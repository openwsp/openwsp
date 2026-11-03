"""Durable Experiment 8 stages; one driver owns budget reservations."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from openwsp.budget import BudgetLedger, compute_cost
from openwsp.backends.jobs import RemoteJobRunner
from replications.exp8.runtime import CONFIG, ROOT, atomic_write, evaluate_outputs, read_rows
from replications.exp8.jobs import ProtocolGPUJobs, gpu_instance
from replications.exp8.eval_grid import STAGES as GRID_STAGES


def run(stage, overrides, *, app, driver_ticket=None):
    from openwsp.backends.modal_app import prepare, data_volume, Generator
    if stage not in {"all", "prepare", "calibration", "smoke", "kernel", "live", "evaluate", "controls", "control4", "control5", "damage", "editor_probe", "review_live", "publish", *GRID_STAGES}:
        raise ValueError("Unknown replication stage")
    if overrides:
        unknown = set(overrides) - {"publish_results", "editor", "run_id"}
        if unknown:
            raise ValueError("Protocol overrides are disabled for this pinned replication: " + ", ".join(sorted(unknown)))
    settings = {**CONFIG, **overrides}
    from replications.exp8.run_identity import live_identity
    # CONFIG['editor'] is the pinned model name; only caller transport
    # overrides select a backend or a separately identified attempt.
    identity = live_identity(overrides or {})
    data = Path("/data")
    # The serialized launcher committed this invocation's ticket after any
    # predecessor finished. A warm worker must see that new Volume snapshot.
    data_volume.reload()
    if stage in GRID_STAGES or stage == "damage":
        from replications.exp8.eval_grid import grid_live_identity
        identity = grid_live_identity(overrides or {}, data=data)
    ledger = BudgetLedger(data / "budget.json")
    state_path = data / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    state.update(status="running", active_stage=stage)
    state.pop("error", None)
    state.pop("error_type", None)
    stop = threading.Event()
    checkpoint_lock = threading.RLock()

    def checkpoint():
        with checkpoint_lock:
            state["heartbeat"] = time.time()
            state["budget"] = ledger.snapshot()["summary"]
            atomic_write(state_path, state)
            data_volume.commit()

    def heartbeat():
        while not stop.wait(45):
            checkpoint()

    jobs_runner = RemoteJobRunner(ledger, data / "jobs", commit=data_volume.commit)
    gpu_jobs = ProtocolGPUJobs(ledger, data / "jobs", commit=checkpoint)
    driver_lease = None
    if driver_ticket:
        import modal
        from openwsp.backends.driver_budget import DriverBudgetLease
        driver_lease = DriverBudgetLease.claim(ledger, driver_ticket, owner_id=modal.current_function_call_id(), commit=checkpoint)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        checkpoint()
        if not state["stages"].get("prepare", {}).get("status") == "done" or not (data / "benchmarks/ifeval.jsonl").exists():
            jobs_runner.launch("prepare-v2", lambda: prepare.spawn(), upper_bound_usd=1,
                               cost_options={"cpu_cores": 2, "memory_gib": 8, "startup_seconds": 120, "scaledown_seconds": 2},
                               fingerprint={"revision": CONFIG["revision"], "datasets": ["math500", "gsm8k", "ifeval"]})
            result = jobs_runner.result("prepare-v2", timeout=1920)
            data_volume.reload()
            state["stages"]["prepare"] = result
            checkpoint()
        if stage == "prepare":
            state["status"] = "stage_done"
            return state
        if stage in ("all", "calibration", "smoke"):
            # Rebuild with the authors' unmodified script, whose tensor hashes are assertions.
            if not (data / "p50c/config.json").exists():
                checkpoint_dir = data / "p50c"
                shutil.copytree(data / "original", checkpoint_dir, dirs_exist_ok=True)
                original_files = sorted((data / "original").glob("*.safetensors"))
                if len(original_files) != 1:
                    raise ValueError("Expected pinned checkpoint's single shard")
                subprocess.run([sys.executable, str(ROOT / "vendor/apply_edit.py"), str(original_files[0]),
                                str(data / "reference_edit"), str(checkpoint_dir / original_files[0].name)], check=True)
                data_volume.commit()
            rows = read_rows(data / "benchmarks/math500.jsonl")
            jobs = [("calibration_base", "/data/original", True, {}),
                    ("calibration_p50c", "/data/p50c", True, {})]
            for name, checkpoint_dir, thinking, sampling_changes in jobs:
                if state["stages"].get(name, {}).get("status") == "done":
                    continue
                params = {**settings["sampling"], **sampling_changes, "n": 1}
                # Every configuration first completes four actual problems at full token cap.
                batches = [(name + "_smoke", rows[:4])]
                if stage != "smoke":
                    batches.append((name, rows))
                for job, selected in batches:
                    target = data / "results" / job
                    if (target / "summary.json").exists():
                        state["stages"][job] = json.loads((target / "summary.json").read_text())
                        continue
                    state["active_stage"] = job
                    checkpoint()
                    gpu_jobs.launch(job, lambda gpu, gate: gpu_instance(Generator, gpu, gate, checkpoint=checkpoint_dir).generate.spawn(selected, params, thinking),
                                       fingerprint={"checkpoint": checkpoint_dir, "params": params, "thinking": thinking,
                                                    "ids": [r["id"] for r in selected]})
                    result = gpu_jobs.result(job, timeout=3600)
                    elapsed = result["container_seconds"]
                    cost = result.get("_openwsp_gpu", {}).get("total_settled_cost_usd") or compute_cost(elapsed + 122, gpu=result["gpu"], cpu_cores=4, memory_gib=16)
                    summary = evaluate_outputs(result["outputs"], selected, directory=target, stage=job,
                                               benchmark="math500", samples=1, seconds=elapsed,
                                               cost_usd=float(cost), thinking=thinking,
                                               provenance={"revision": settings["revision"], "sampling": params,
                                                           "thinking": thinking, "gpu": result["gpu"],
                                                           "engine": result["engine"],
                                                           "engine_settings": result.get("engine_settings", {"max_num_seqs": "vLLM default"}),
                                                           "seconds_basis": "observed_container_duration",
                                                           "accounted_gpu_seconds": result.get("_openwsp_gpu", {}).get("billable_seconds", elapsed + 122),
                                                           "transformers": result["transformers"]})
                    state["stages"][job] = summary
                    checkpoint()
        if stage in ("all", "calibration"):
            check = {}
            for name, reference, se in (("calibration_base", .079, .012), ("calibration_p50c", .481, .022)):
                score = state["stages"][name]["accuracy"]
                check[name] = {"accuracy": score, "reference_accuracy": reference, "reference_standard_error": se,
                               "gap": score-reference, "within_two_standard_errors": abs(score-reference) <= 2*se}
            check["status"] = "passed" if all(row["within_two_standard_errors"] for row in check.values()) else "gap_requires_review"
            atomic_write(data / "calibration-check.json", check)
            state["calibration"] = check
            checkpoint()
            if stage == "all" and check["status"] != "passed":
                audit_path = data / "calibration-gap-audit.json"
                audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
                if not audit.get("circuit_breaker_tripped") or not audit.get("checks"):
                    raise RuntimeError("Calibration gap needs a recorded prompt/template/sampling audit before the live run")
                state["calibration"]["audit"] = audit
        if stage in ("all", "kernel", "live") and state["stages"].get("kernel", {}).get("status") != "passed":
            from replications.exp8.live import verify_kernel
            state["active_stage"] = "kernel_verification"
            checkpoint()
            state["stages"]["kernel"] = verify_kernel(ledger, checkpoint, app)
            checkpoint()
        if stage in ("all", "kernel", "live"):
            from replications.exp8.verification import verify_forward
            state["active_stage"] = "forward_verification"
            checkpoint()
            state["stages"]["forward_verification"] = verify_forward(ledger, checkpoint)
        if stage in ("all", "live"):
            from replications.exp8.live import run_live
            state["active_stage"] = "live"
            if identity.run_id:
                active = identity.record()
                atomic_write(data / "active-live.json", active)
                state["active_live"] = active
            checkpoint()
            state["controls_deferred"] = "Current verified allowance preserves priority for the live run and its evaluation. Controls resume afterward."
            result = run_live(state, ledger, checkpoint, app=app, identity=identity)
            state["live"] = {key: result.get(key) for key in ("turn", "runs", "passes", "requests", "done", "submission")}
            if identity.run_id:
                state["live"].update(run_id=identity.run_id, editor=identity.editor)
            checkpoint()
        if stage in ("all", "evaluate"):
            from replications.exp8.evaluation import evaluate_live
            live_state = json.loads((identity.root / "state.json").read_text())
            if not live_state.get("done") or live_state.get("passes") != 50:
                raise ValueError("Required evaluation waits for the actual completed 50-pass submission")
            state["active_stage"] = "evaluation"
            checkpoint()
            state["stages"]["live_evaluation"] = evaluate_live(live_state["submission"], ledger, checkpoint,
                                                               stage=identity.evaluation_stage)
        if stage in ("all", "controls"):
            from replications.exp8.controls import run_controls
            state["active_stage"] = "controls"
            checkpoint()
            state["stages"]["controls"] = run_controls(ledger, checkpoint)
        if stage in ("control4", "control5"):
            from replications.exp8.optional_controls import run_optional_control
            state["active_stage"] = stage
            checkpoint()
            state["stages"][stage] = run_optional_control(stage, ledger, checkpoint)
        if stage in GRID_STAGES:
            from replications.exp8.eval_grid import run_grid_stage
            state["active_stage"] = stage
            checkpoint()
            state["stages"][stage] = run_grid_stage(stage, ledger, checkpoint, identity=identity)
        if stage == "damage":
            from replications.exp8.damage_stage import run_damage_stage
            state["active_stage"] = stage
            checkpoint()
            state["stages"][stage] = run_damage_stage(ledger, checkpoint, identity=identity)
        if stage == "editor_probe":
            from replications.exp8.editor_validation import run_probe
            state["active_stage"] = stage
            checkpoint()
            state["stages"][stage] = run_probe(ledger, checkpoint)
        if stage == "review_live":
            from replications.exp8.editor_validation import review_live_transport
            state["active_stage"] = stage
            checkpoint()
            state["stages"][stage] = review_live_transport(ledger, checkpoint)
        if stage in ("all", "publish"):
            from replications.exp8.artifacts import publish
            state.update(active_stage="publication", status="complete" if stage == "all" else "stage_done")
            checkpoint()
            state["publication"] = publish()
        state["status"] = "complete" if stage == "all" else "stage_done"
        return state
    except Exception as exc:
        state.update(status="blocked", error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
        checkpoint()
        if stage == "all":
            # A detached run must publish partial evidence and its blocker too.
            # This uses the already-reserved CPU driver, and never masks the
            # original stage failure or starts more paid inference work.
            try:
                from replications.exp8.artifacts import publish
                state["publication"] = publish()
            except Exception as publication_error:
                state["publication"] = {"status": "blocked", "error_type": type(publication_error).__name__}
        raise
    finally:
        stop.set()
        if driver_lease:
            state["driver_receipt"] = driver_lease.finish()
        checkpoint()
