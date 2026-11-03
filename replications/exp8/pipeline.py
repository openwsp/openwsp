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
from replications.exp8.runtime import CONFIG, ROOT, atomic_write, evaluate_outputs, read_rows


def run(stage, overrides, *, app):
    from openwsp.backends.modal_app import prepare, data_volume, Generator
    settings = {**CONFIG, **overrides}
    data = Path("/data")
    ledger = BudgetLedger(data / "budget.json")
    state_path = data / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    state.update(status="running", active_stage=stage)
    stop = threading.Event()

    def checkpoint():
        state["heartbeat"] = time.time()
        state["budget"] = ledger.snapshot()["summary"]
        atomic_write(state_path, state)
        data_volume.commit()

    def heartbeat():
        while not stop.wait(45):
            checkpoint()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        checkpoint()
        if not state["stages"].get("prepare", {}).get("status") == "done":
            reservation = ledger.reserve("modal", 1, "prepare", metadata={"estimated_seconds": 1800})
            checkpoint()
            start = time.monotonic()
            result = prepare.remote()
            data_volume.reload()
            elapsed = time.monotonic() - start
            ledger.settle(reservation, compute_cost(elapsed, cpu_cores=2, memory_gib=8), seconds=elapsed)
            state["stages"]["prepare"] = result
            checkpoint()
        if stage == "prepare":
            state["status"] = "stage_done"
            return state
        if stage in ("all", "calibration", "smoke", "controls"):
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
            if stage == "controls":
                jobs = [("control1_nothink", "/data/original", False,
                         {"temperature": 0.7, "top_p": 0.8, "max_tokens": 32768}),
                        ("control3_repetition", "/data/original", True, {"repetition_penalty": 1.1})]
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
                        continue
                    max_seconds = settings["max_job_seconds"]
                    reservation = ledger.reserve("modal", compute_cost(max_seconds + 120, gpu="H100",
                                                  cpu_cores=4, memory_gib=16), job,
                                                  metadata={"estimated_gpu_seconds": max_seconds,
                                                            "problems": len(selected), "sampling": params})
                    state["active_stage"] = job
                    checkpoint()
                    start = time.monotonic()
                    result = Generator(checkpoint=checkpoint_dir).generate.remote(selected, params, thinking)
                    elapsed = time.monotonic() - start
                    cost = compute_cost(elapsed + 2, gpu="H100", cpu_cores=4, memory_gib=16)
                    ledger.settle(reservation, cost, seconds=result["container_seconds"],
                                  metadata={"generation_seconds": result["generation_seconds"]})
                    summary = evaluate_outputs(result["outputs"], selected, directory=target, stage=job,
                                               benchmark="math500", samples=1, seconds=elapsed,
                                               cost_usd=float(cost), thinking=thinking,
                                               provenance={"revision": settings["revision"], "sampling": params,
                                                           "thinking": thinking, "gpu": result["gpu"],
                                                           "engine": result["engine"],
                                                           "transformers": result["transformers"]})
                    state["stages"][job] = summary
                    checkpoint()
        if stage == "all":
            from replications.exp8.live import run_live
            run_live(state, ledger, checkpoint, app=app)
        state["status"] = "stage_done"
        return state
    except Exception as exc:
        state.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        checkpoint()
