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

from openwsp.backends.forward_bank import (
    BankError,
    ForwardResultBank,
    GenerationBank,
    checkpoint_hash,
    json_hash,
)
from openwsp.backends.jobs import JobBlocked, JobPending
from openwsp.budget import BudgetExceeded, compute_cost
from openwsp.core import WeightStore
from replications.exp8.jobs import ProtocolGPUJobs, gpu_instance
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
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118 - safetensors API
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

    def counts_as_pass(self, number):
        path = self.run_dir / "requests" / f"fwd_{number}.provenance.json"
        return bool(json.loads(path.read_text()).get("count_pass", True)) if path.exists() else True

    def _banks(self):
        if not hasattr(self, "base_bank"):
            self.base_bank = GenerationBank("/data/forward-bank/base")
        if not hasattr(self, "result_bank"):
            self.result_bank = ForwardResultBank(self.run_dir / "forward-bank")
        return self.base_bank, self.result_bank

    def _model_hashes(self, frozen):
        if not hasattr(self, "_checkpoint_hashes"):
            self._checkpoint_hashes = {}
        for path in ("/data/original", frozen):
            if path not in self._checkpoint_hashes:
                self._checkpoint_hashes[path] = checkpoint_hash(path)
        return {"original": self._checkpoint_hashes["/data/original"], "edited": self._checkpoint_hashes[frozen]}

    def _generation_prompts(self, rows):
        from transformers import AutoTokenizer

        from replications.exp8.runtime import INSTRUCTION
        if not hasattr(self, "generation_tokenizer"):
            self.generation_tokenizer = AutoTokenizer.from_pretrained("/data/original", local_files_only=True)
        return [self.generation_tokenizer.apply_chat_template(
            [{"role": "user", "content": row.get("prompt", row.get("problem", "") + "\n" + INSTRUCTION)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True) for row in rows]

    def _effective_identity(self, request, frozen, edit_id):
        if request.get("type") == "generate":
            rows, _, _ = validate_generate(request, self.practice)
            resolved = {"rows": rows, "rendered_prompts": self._generation_prompts(rows)}
        else:
            resolved = request.get("texts", request.get("chats"))
        return {"schema_version": 1, "request_body_sha256": json_hash(request),
                "resolved_content_sha256": json_hash(resolved), "checkpoints": self._model_hashes(frozen),
                "edit_snapshot_sha256": edit_id, "sampling": CONFIG["sampling"],
                "engine": CONFIG["engine"], "transformers": CONFIG["transformers"],
                "seed_policy": "sha256(problem_id).hexdigest()[:8] modulo 2^31"}

    def _base_plan(self, number, rows, sampling, model_hash, *, paired, identity):
        bank, _ = self._banks()
        prompts = self._generation_prompts(rows)
        keys = [bank.identity(str(row["id"]), prompt,
                              {"sampling": sampling, "thinking": True, "engine": CONFIG["engine"],
                               "transformers": CONFIG["transformers"]}, model_hash,
                              int(hashlib.sha256(str(row["id"]).encode()).hexdigest()[:8], 16) % 2**31)
                for row, prompt in zip(rows, prompts, strict=True)]
        path = self.run_dir / "requests" / f"fwd_{number}.base-plan.json"
        if path.exists():
            try:
                plan = json.loads(path.read_text())
                if (not isinstance(plan, dict) or plan.get("identity") != identity or plan.get("keys") != keys
                        or not isinstance(plan.get("cached_sha256"), list) or len(plan["cached_sha256"]) != len(keys)
                        or any(item is not None and (not isinstance(item, str) or re.fullmatch(r"[a-f0-9]{64}", item) is None)
                               for item in plan["cached_sha256"])
                        or plan.get("missing") != [index for index, digest in enumerate(plan["cached_sha256"]) if digest is None]):
                    raise ValueError("Generation plan binding differs")
            except (OSError, ValueError, TypeError) as error:
                raise JobPending("Saved generation plan is malformed or does not match the effective request; inspect before resuming") from error
        else:
            # An older job already owns its exact rows. Never replace that job's
            # immutable receipt with a new partial plan or new sampling.
            existing = self.jobs.has_job(f"{self.job_prefix}/fwd_{number}/original")
            cached = [bank.get(key) if paired and not existing else None for key in keys]
            plan = {"identity": identity, "keys": keys,
                    "cached_sha256": [entry["output_sha256"] if entry else None for entry in cached],
                    "missing": [index for index, entry in enumerate(cached) if entry is None],
                    "preserved_existing_job": existing}
            atomic_write(path, plan)
            self.checkpoint()
        cached = [bank.get(key) if digest is not None else None
                  for key, digest in zip(keys, plan["cached_sha256"], strict=True)]
        if any(digest is not None and (entry is None or entry["output_sha256"] != digest)
               for digest, entry in zip(plan["cached_sha256"], cached, strict=True)):
            raise JobPending("Saved base generations are unavailable; no replacement sampling was launched")
        return plan, cached

    def _serve_duplicate(self, entry, request, number, turn, identity, started):
        _, bank = self._banks()
        old = entry["source_request"]

        def rename(value):
            return re.sub(rf"^(gen_|fwd_){old}(?=[_./]|$)", rf"\g<1>{number}", value)

        bank.restore(entry, self.work, paths={item["path"]: rename(item["path"]) for item in entry["artifacts"]})

        def rewrite(value):
            if isinstance(value, str) and value.startswith("/work/"):
                return "/work/" + rename(value[6:])
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            return value

        result = rewrite(entry["result"])
        result["s"] = time.monotonic() - started
        atomic_write(self.run_dir / "requests" / f"fwd_{number}.provenance.json",
                     {"turn": turn, "request_body_sha256": identity["request_body_sha256"],
                      "effective_request_sha256": json_hash(identity), "served_from_request": old,
                      "count_pass": old == number, "base_generations_reused": False,
                      "bank_artifacts": entry["artifacts"]})
        from openwsp.backends.modal_app import work_volume
        work_volume.commit()
        self._deliver_artifacts(request, number, result)
        atomic_write(self.run_dir / "requests" / f"fwd_{number}.done.json", result)
        self.checkpoint()
        return result

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
        identity = self._effective_identity(request, frozen, edit_id)
        identity_path = self.run_dir / "requests" / f"fwd_{number}.identity.json"
        if identity_path.exists():
            try:
                prior_identity = json.loads(identity_path.read_text())
            except (OSError, ValueError, TypeError) as error:
                raise JobPending("Pending forward identity is malformed; inspect before resuming") from error
            if prior_identity != identity:
                raise JobPending("Pending forward request has a different effective checkpoint or text; inspect before resuming")
        atomic_write(identity_path, identity)
        self.checkpoint()
        base_bank, result_bank = self._banks()
        try:
            duplicate = result_bank.get(identity)
            if duplicate is not None:
                return self._serve_duplicate(duplicate, request, number, turn, identity, started)
        except BankError as error:
            raise JobPending("A saved forward bank entry could not be verified; no replacement job was launched") from error
        if request.get("type") == "generate":
            rows, n, max_tokens = validate_generate(request, self.practice)
            modes = ("original", "edited") if request["model"] == "both" else (request["model"],)
            sampling = {**CONFIG["sampling"], "n": n, "max_tokens": max_tokens}
            base_plan, cached_base = (None, None)
            if "original" in modes:
                try:
                    base_plan, cached_base = self._base_plan(number, rows, sampling, identity["checkpoints"]["original"],
                                                              paired=request["model"] == "both", identity=identity)
                except BankError as error:
                    raise JobPending("A base bank entry could not be verified; no replacement sampling was launched") from error
            reuse_base = bool(base_plan and any(item is not None for item in cached_base))
            base_complete = bool(base_plan and not base_plan["missing"])
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
                          if not (base_complete and mode == "original"))
            if self.ledger.snapshot()["summary"]["modal"]["available_usd"] < bound * missing:
                raise BudgetExceeded("Insufficient reserved capacity for this paired pass")
            for mode in modes:
                if base_complete and mode == "original":
                    running.append((mode, None))
                    continue
                model_path = "/data/original" if mode == "original" else frozen
                key = f"{self.job_prefix}/fwd_{number}/{mode}"
                arguments = swap_arguments if mode == "edited" else {}
                if arguments:
                    model_path = "/data/original"
                job_rows = [rows[index] for index in base_plan["missing"]] if mode == "original" else rows
                self._launch(key, lambda gpu, gate, model_path=model_path, mode=mode, arguments=arguments, job_rows=job_rows: gpu_instance(Generator, gpu, gate, checkpoint=model_path, slot=mode).generate.spawn(job_rows, sampling, True, True, **arguments),
                             {"request": request, "mode": mode, "edit": edit_id if mode == "edited" else "original"})
                running.append((mode, key))
            all_records, summaries = {}, {}
            for mode, key in running:
                response = self._result(key) if key else {"outputs": []}
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
                if mode == "original":
                    outputs = [entry["output"] if entry else None for entry in cached_base]
                    for index, generated in zip(base_plan["missing"], response["outputs"], strict=True):
                        if generated.get("id") != rows[index]["id"] or len(generated["samples"]) != n:
                            raise JobPending("Generation receipt does not match its saved row plan")
                        try:
                            base_bank.put(base_plan["keys"][index], generated,
                                          provenance={"job_key": key, "run_directory": str(self.run_dir),
                                                      "request": number, "engine": response.get("engine", CONFIG["engine"])})
                        except BankError as error:
                            raise JobPending("Saved generation could not enter the verified bank; retain its job receipt") from error
                        outputs[index] = generated
                    response = {**response, "outputs": outputs}
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
                    self._launch(key, lambda gpu, gate, model_path=model_path, records=records, params=params, folder=folder: gpu_instance(TransformerReader, gpu, gate, checkpoint=model_path).generation_activations.spawn(
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
                      "request_body_sha256": identity["request_body_sha256"],
                      "effective_request_sha256": json_hash(identity), "count_pass": True,
                      "base_generations_reused": bool(locals().get("reuse_base", False)),
                      "base_bank": [{"key": json_hash(key), "reused": cached is not None,
                                     "source": cached["provenance"] if cached else {"request": number}}
                                    for key, cached in zip(base_plan["keys"], cached_base, strict=True)]
                                   if locals().get("base_plan") else []})
        work_volume.commit()
        self._deliver_artifacts(request, number, result)
        delivery = json.loads((self.run_dir / "requests" / f"fwd_{number}.delivery.json").read_text())
        try:
            result_bank.put(identity, result, request_number=number,
                            artifacts={item["path"][6:]: self.work / item["path"][6:] for item in delivery["files"]})
        except BankError as error:
            raise JobPending("Forward result is saved but its bank artifacts require inspection; retain job receipts") from error
        atomic_write(destination, result)
        self.checkpoint()
        return result
