"""The released Experiment 8 forward-tool contract, with durable request receipts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np

from openwsp.budget import compute_cost, BudgetExceeded
from openwsp.backends.jobs import JobBlocked, JobPending
from replications.exp8.jobs import ProtocolGPUJobs, gpu_instance
from openwsp.core import WeightStore
from replications.exp8.runtime import CONFIG, ROOT, atomic_write, read_rows, write_rows


def grader():
    import sys
    if str(ROOT / "compat") not in sys.path:
        sys.path.insert(0, str(ROOT / "compat"))
    if str(ROOT / "vendor") not in sys.path:
        sys.path.insert(0, str(ROOT / "vendor"))
    spec = importlib.util.spec_from_file_location("exp8_grader", ROOT / "vendor/mathqa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_checkpoint(work, destination):
    """Validate the file submission in the trusted driver; freeze its exact snapshot."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    store = WeightStore("/model", Path(work) / "edited")
    snapshot = store.validate_submission()
    names = sorted(snapshot)
    hasher = hashlib.sha256()
    for name in names:
        hasher.update(name.encode())
        hasher.update(snapshot[name].numpy().tobytes())
    edit_id = hasher.hexdigest()
    checkpoint = Path(destination) / edit_id
    if (checkpoint / "openwsp-complete.json").exists():
        return str(checkpoint), edit_id, names
    checkpoint.mkdir(parents=True, exist_ok=True)
    original = Path("/data/original")
    for path in original.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copyfile(path, checkpoint / path.name)
    for path in original.glob("*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        for name, tensor in snapshot.items():
            if name in tensors:
                tensors[name] = tensor.to(tensors[name].dtype).contiguous()
        save_file(tensors, checkpoint / path.name, metadata={"format": "pt"})
    atomic_write(checkpoint / "openwsp-complete.json", {"edit_id": edit_id, "tensors": names})
    return str(checkpoint), edit_id, names


def validate_generate(request, practice):
    if request.get("model") not in ("original", "edited", "both"):
        raise ValueError("model must be original, edited, or both")
    ids, prompts = request.get("problems", []), request.get("prompts", [])
    if not isinstance(ids, list) or not isinstance(prompts, list) or any(not isinstance(p, str) for p in prompts):
        raise ValueError("problems and prompts must be lists")
    unknown = [item for item in ids if item not in practice]
    if unknown:
        raise ValueError(f"unknown practice ids: {unknown[:12]}")
    n = request.get("n", 1)
    tokens = request.get("max_tokens", 32768)
    if type(n) is not int or type(tokens) is not int or n < 1 or not 1 <= tokens <= 32768:
        raise ValueError("Invalid generation count or token limit")
    rows = [practice[item] for item in ids]
    rows += [{"id": "prompt_" + hashlib.sha256(prompt.encode()).hexdigest(), "prompt": prompt,
              "source": "custom"} for prompt in prompts]
    if not rows or len(rows)*n > (256 if request["model"] == "both" else 512):
        raise ValueError("Generation request exceeds the pass-size limit")
    activations = request.get("activations")
    if activations:
        layers = activations.get("layers", [])
        if len(layers) != len(set(layers)) or any(type(layer) is not int or not 0 <= layer <= 24 for layer in layers):
            raise ValueError("Invalid activation layers")
    return rows, n, tokens


def base_cache_key(row, sampling):
    return hashlib.sha256(json.dumps({"row": row, "sampling": sampling, "revision": CONFIG["revision"],
                                     "engine": CONFIG["engine"], "seed_by_id": True}, sort_keys=True).encode()).hexdigest()


class ForwardTool:
    def __init__(self, run_dir, work_dir, sandbox, ledger, checkpoint, *, job_prefix="live"):
        self.run_dir, self.work = Path(run_dir), Path(work_dir)
        self.sandbox, self.ledger, self.checkpoint = sandbox, ledger, checkpoint
        self.job_prefix = job_prefix
        self.practice = {row["id"]: row for p in Path("/practice").glob("*.jsonl") for row in read_rows(p)}
        self.grade = grader()
        self.jobs = ProtocolGPUJobs(ledger, self.run_dir / "jobs", commit=checkpoint)
        self.work.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "requests").mkdir(parents=True, exist_ok=True)

    def _launch(self, key, spawn, fingerprint):
        return self.jobs.launch(key, spawn, fingerprint=fingerprint)

    def _result(self, key):
        try:
            return self.jobs.result(key, timeout=3600)
        except JobBlocked as exc:
            record = self.jobs.active_child_record(key)
            if record.get("failure", {}).get("category") != "job_failure":
                raise
            reservation = self.ledger.snapshot()["reservations"][record["reservation_id"]]
            if reservation["status"] == "reserved":
                self.ledger.settle(record["reservation_id"], reservation["upper_bound_usd"], seconds=2520,
                                   metadata={"failed_job_upper_bound": True, "call_id": record.get("call_id")})
            self.checkpoint()
            error_type = record.get("failure", {}).get("exception_type", "unknown")
            raise RuntimeError(f"The remote forward execution failed ({error_type}); its upper-bound cost was retained") from exc

    def _deliver_artifacts(self, request, number, result):
        """Publish only this result's files in the kernel's existing mount.

        A global volume reload can fail while the editor retains a file or
        memory map. Atomic file publication preserves that kernel state and
        does not require credentials or network access inside the sandbox.
        """
        if not result.get("ok"):
            return
        try:
            artifacts = []
            if request["type"] == "generate":
                paired = request["model"] == "both"
                modes = ("original", "edited") if paired else (request["model"],)
                for mode in modes:
                    suffix = f"_{mode}" if paired else ""
                    name = f"gen_{number}{suffix}.jsonl"
                    reported = result["files"][mode] if paired else result["file"]
                    if reported != f"/work/{name}":
                        raise ValueError("Generation result path disagrees with its request")
                    artifacts.append(Path(name))
                    activations = ((result.get("activations") or {}).get(mode) if paired
                                   else result.get("activations"))
                    if activations:
                        folder = f"gen_{number}{suffix}_acts"
                        if activations["dir"].rstrip("/") != f"/work/{folder}":
                            raise ValueError("Activation result path disagrees with its request")
                        for name in [*activations["files"], "index.json"]:
                            if not re.fullmatch(r"[A-Za-z0-9_]+\.(?:npy|json)", name):
                                raise ValueError("Invalid activation artifact name")
                            artifacts.append(Path(folder) / name)
            elif request["type"] == "teacher_forced":
                folder = Path(f"fwd_{number}")
                for name in result["files"]:
                    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                        raise ValueError("Invalid teacher-forced artifact name")
                    artifacts.append(folder / f"{name}.npy")
                artifacts.append(folder / "meta.json")
            else:
                raise ValueError("Unknown forward artifact type")
            receipts = []
            for relative in artifacts:
                receipt = self.sandbox.publish_file(self.work / relative, f"/work/{relative.as_posix()}")
                receipts.append({**receipt, "volume_path": f"/live/{relative.as_posix()}"})
            atomic_write(self.run_dir / "requests" / f"fwd_{number}.delivery.json",
                         {"request": number, "volume": "openwsp-exp8-work", "files": receipts})
            self.checkpoint()
        except BudgetExceeded:
            raise
        except Exception as exc:
            # Delivery is infrastructure recovery, not a model failure. Keep
            # successful GPU receipts and prevent the live loop from converting
            # this exception into a failed request that the editor must repeat.
            raise JobPending("Forward artifacts are saved but sandbox delivery did not complete; resume the existing request") from exc

    def execute(self, request, number, turn):
        from openwsp.backends.modal_app import Generator, TransformerReader, data_volume, work_volume
        started = time.monotonic()
        source = self.run_dir / "requests" / f"fwd_{number}.json.claimed"
        destination = source.with_name(f"fwd_{number}.done.json")
        if source.exists() and json.loads(source.read_text()) != request:
            raise ValueError("Forward request number already belongs to different input")
        if destination.exists():
            # A prior driver may have saved the result just before delivery.
            result = json.loads(destination.read_text())
            self._deliver_artifacts(request, number, result)
            return result
        atomic_write(source, request)
        self.sandbox.commit_work()
        work_volume.reload()
        # Validate even original-only passes: direct file writes must not evade save_edit rules.
        frozen, edit_id, names = freeze_checkpoint(self.work, "/data/live-checkpoints")
        data_volume.commit()
        if request.get("type") == "generate":
            rows, n, max_tokens = validate_generate(request, self.practice)
            modes = ("original", "edited") if request["model"] == "both" else (request["model"],)
            sampling = {**CONFIG["sampling"], "n": n, "max_tokens": max_tokens}
            cache_dir = Path("/data/base-generation-cache")
            cache_dir.mkdir(exist_ok=True)
            cache_paths = [cache_dir / (base_cache_key(row, sampling) + ".json") for row in rows]
            # User-authorized budget fallback; no new information is supplied to
            # the editor. Record its use in the public protocol provenance.
            reuse_base = (request["model"] == "both"
                          and self.ledger.snapshot()["summary"]["modal"]["available_usd"] < 20
                          and all(path.exists() for path in cache_paths))
            swap_path = Path("/data/results/kernel/weight-swap.json")
            use_swap = ("edited" in modes and bool(names) and swap_path.exists()
                        and json.loads(swap_path.read_text()).get("mode") == "verified_native_swap")
            swap_arguments = {}
            if use_swap:
                from transformers import AutoTokenizer
                if not hasattr(self, "tokenizer"):
                    self.tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
                probes = [self.tokenizer.encode(text, add_special_tokens=False) for text in
                          ("One plus one equals two.", "If x = 3, then x squared is 9.",
                           "The sum of 5 and 7 is 12.", "We can solve this equation step by step.")]
                reference_key = f"{self.job_prefix}/fwd_{number}/fresh-reference"
                self._launch(reference_key, lambda gpu, gate: gpu_instance(Generator, gpu, gate, checkpoint=frozen, slot="fresh-reference").reference.spawn(probes, edit_id),
                             {"edit": edit_id, "probes": probes})
                reference = self._result(reference_key)
                # Read only the trusted immutable checkpoint, never mutable
                # editor files after the validation snapshot has been frozen.
                frozen_store = WeightStore(frozen)
                swap_arguments = {"edits": {name: frozen_store.load_original(name) for name in names},
                                  "edit_id": edit_id, "probe_token_ids": probes, "reference": reference["reference"]}
            running = []
            # Check combined bounds before either dispatch; the journal reserves each.
            bound = float(compute_cost(2520, gpu="H100", cpu_cores=4, memory_gib=16))
            missing = sum(not self.jobs.has_job(f"{self.job_prefix}/fwd_{number}/{mode}") for mode in modes
                          if not (reuse_base and mode == "original"))
            if self.ledger.snapshot()["summary"]["modal"]["available_usd"] < bound * missing:
                raise BudgetExceeded("Insufficient reserved capacity for this paired pass")
            for mode in modes:
                if reuse_base and mode == "original":
                    running.append((mode, None))
                    continue
                model_path = "/data/original" if mode == "original" else frozen
                key = f"{self.job_prefix}/fwd_{number}/{mode}"
                arguments = swap_arguments if mode == "edited" else {}
                if arguments:
                    model_path = "/data/original"
                self._launch(key, lambda gpu, gate, model_path=model_path, mode=mode, arguments=arguments: gpu_instance(Generator, gpu, gate, checkpoint=model_path, slot=mode).generate.spawn(rows, sampling, True, True, **arguments),
                             {"request": request, "mode": mode, "edit": edit_id if mode == "edited" else "original"})
                running.append((mode, key))
            all_records, summaries = {}, {}
            for mode, key in running:
                response = self._result(key) if key else {"outputs": [json.loads(path.read_text()) for path in cache_paths]}
                if response.get("needs_fresh_engine"):
                    prior_swap = json.loads(swap_path.read_text())
                    prior_swap.update(mode="fresh_engine_per_pass", status="fallback",
                                      live_failure=response["swap_failure"], failed_request=number)
                    atomic_write(swap_path, prior_swap)
                    self.checkpoint()
                    fallback_key = f"{self.job_prefix}/fwd_{number}/{mode}/fresh-fallback"
                    self._launch(fallback_key, lambda gpu, gate: gpu_instance(Generator, gpu, gate, checkpoint=frozen, slot="fresh-fallback").generate.spawn(rows, sampling, True, True),
                                 {"request": request, "edit": edit_id, "reason": response["swap_failure"]})
                    atomic_write(self.run_dir / "requests" / f"fwd_{number}.swap-fallback.json", response["swap_failure"])
                    response = self._result(fallback_key)
                if mode == "original" and key:
                    for path, generated in zip(cache_paths, response["outputs"], strict=True):
                        atomic_write(path, generated)
                    data_volume.commit()
                records, samples = [], []
                for row, generated in zip(rows, response["outputs"], strict=True):
                    current = []
                    for sample in generated["samples"]:
                        pred, _ = self.grade.extract(sample["text"])
                        valid = "answer" in row
                        good = bool(self.grade.is_correct(pred, row.get("answer"))) if valid else None
                        current.append({"text": sample["text"], "pred": pred, "correct": good,
                                        "tokens": sample["token_count"], "cut": sample["finish_reason"] == "length",
                                        "finish_reason": sample["finish_reason"]})
                    records.append({"id": row["id"], "source": row["source"], "answer": row.get("answer"), "samples": current})
                    samples.extend((row, sample) for sample in current)
                suffix = f"_{mode}" if request["model"] == "both" else ""
                name = f"gen_{number}{suffix}.jsonl"
                write_rows(self.work / name, records)
                work_volume.commit()
                graded = [(row, sample) for row, sample in samples if sample["correct"] is not None]
                sources = sorted({row["source"] for row, _ in graded})
                summary = {"file": f"/work/{name}", "generations": len(samples), "graded": len(graded),
                           "accuracy": np.mean([s["correct"] for _, s in graded]).item() if graded else None,
                           "accuracy_by_source": {source: float(np.mean([s["correct"] for r, s in graded if r["source"] == source]))
                                                  for source in sources},
                           "mean_tokens": float(np.mean([s["tokens"] for _, s in samples])),
                           "cut_share": float(np.mean([s["cut"] for _, s in samples])),
                           "failures": {"wrong_answer": sum(not s["correct"] and s["pred"] is not None and not s["cut"] for _, s in graded),
                                        "no_answer_finished": sum(not s["correct"] and s["pred"] is None and not s["cut"] for _, s in graded),
                                        "cut_at_max_tokens": sum(s["cut"] for _, s in graded)}}
                all_records[mode], summaries[mode] = records, summary
                if request.get("activations"):
                    params = request["activations"]
                    folder = self.work / f"gen_{number}{suffix}_acts"
                    model_path = "/data/original" if mode == "original" else frozen
                    key = f"{self.job_prefix}/fwd_{number}/{mode}/activations"
                    self._launch(key, lambda gpu, gate: gpu_instance(TransformerReader, gpu, gate, checkpoint=model_path).generation_activations.spawn(
                        rows, records, params.get("layers", []), params.get("mlp", False), str(folder)),
                        {"request": request, "mode": mode, "edit": edit_id if mode == "edited" else "original"})
                    result = self._result(key)
                    work_volume.reload()
                    summary["activations"] = {"dir": f"/work/{folder.name}/", "files": result["files"]}
            if request["model"] == "both":
                differences, rescues, losses = [], 0, 0
                for base, edited in zip(all_records["original"], all_records["edited"], strict=True):
                    pairs = [(b["correct"], e["correct"]) for b, e in zip(base["samples"], edited["samples"], strict=True)
                             if b["correct"] is not None and e["correct"] is not None]
                    if pairs:
                        differences.append(float(np.mean([int(e)-int(b) for b, e in pairs])))
                        rescues += sum(not b and e for b, e in pairs)
                        losses += sum(b and not e for b, e in pairs)
                files = {mode: summaries[mode].pop("file") for mode in modes}
                acts = {mode: summaries[mode].pop("activations") for mode in modes if "activations" in summaries[mode]}
                result = {"files": files, "activations": acts or None,
                          "original": summaries["original"], "edited": summaries["edited"],
                          "paired": {"items": len(differences), "accuracy_difference": float(np.mean(differences)) if differences else None,
                                     "standard_error": float(np.std(differences, ddof=1)/math.sqrt(len(differences))) if len(differences)>1 else None,
                                     "samples_wrong_to_right": rescues, "samples_right_to_wrong": losses}}
            else:
                result = summaries[request["model"]]
        elif request.get("type") == "teacher_forced":
            if request.get("model") not in ("original", "edited"):
                raise ValueError("Teacher-forced model must be original or edited")
            from transformers import AutoTokenizer
            from replications.exp8.internals import tokenize_request
            if not hasattr(self, "tokenizer"):
                self.tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
            ids, ret = tokenize_request(self.tokenizer, request)
            batch, length = len(ids), max(map(len, ids))
            byte_bound = batch*length*(8 + 4*bool(ret.get("token_logprobs")) + 12*ret.get("topk", 0)
                         + 2*1024*len(ret.get("hidden_states", [])) + 2*3584*len(ret.get("mlp_act", []))
                         + 2*8*length*len(ret.get("attn_layers", [])))
            if byte_bound > 6*1024**3:
                raise ValueError("Teacher-forced outputs exceed 6 GiB")
            model_path = "/data/original" if request["model"] == "original" else frozen
            key = f"{self.job_prefix}/fwd_{number}/teacher_forced"
            self._launch(key, lambda gpu, gate: gpu_instance(TransformerReader, gpu, gate, checkpoint=model_path, eager=bool(request.get("return", {}).get("attn_layers"))).read.spawn(
                request, str(self.work / f"fwd_{number}")), {"request": request, "edit": edit_id if request["model"] == "edited" else "original"})
            result = self._result(key)
            work_volume.reload()
            result["edited_tensors"] = len(names) if request["model"] == "edited" else 0
        else:
            raise ValueError("type must be generate or teacher_forced")
        result.update(ok=True, type=request["type"], model=request["model"], s=time.monotonic()-started)
        # Provenance stays in harness logs, outside the editor-visible summary.
        atomic_write(self.run_dir / "requests" / f"fwd_{number}.provenance.json",
                     {"turn": turn, "edit_sha256": edit_id, "tensors": names,
                      "base_generations_reused": bool(locals().get("reuse_base", False))})
        work_volume.commit()
        self._deliver_artifacts(request, number, result)
        atomic_write(destination, result)
        self.checkpoint()
        return result
