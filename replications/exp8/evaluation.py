"""Resumable evaluation batches with the authors' fixed sampling and graders."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from openwsp.backends.forward_bank import BankError, GenerationBank, checkpoint_hash, json_hash
from openwsp.backends.jobs import JobPending
from openwsp.budget import compute_cost
from openwsp.metrics import finished_metrics
from replications.exp8.jobs import ProtocolGPUJobs, gpu_instance
from replications.exp8.runtime import (
    CONFIG,
    INSTRUCTION,
    atomic_write,
    evaluate_outputs,
    read_rows,
    write_rows,
)


def _base_generation_keys(rows, checkpoint_dir, params, thinking, engine_settings, generation_options):
    """Use the live bank's effective identity, plus any changed decoding settings."""
    from transformers import AutoTokenizer

    seed_by_id = generation_options.get("seed_by_id", False)
    if type(seed_by_id) is not bool:
        raise ValueError("seed_by_id must be boolean")
    if generation_options.get("seed_scheme", "sha256_first8_utf8_problem_id_mod_2**31") != "sha256_first8_utf8_problem_id_mod_2**31":
        raise ValueError("Unsupported generation seed scheme")
    if not seed_by_id and type(params.get("seed")) is not int:
        # Unseeded outputs cannot be substituted for a fresh random draw.
        return None
    model_hash = checkpoint_hash(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    effective = {"sampling": params, "thinking": thinking, "engine": CONFIG["engine"],
                 "transformers": CONFIG["transformers"]}
    changed_settings = {key: value for key, value in engine_settings.items()
                        if value != {"max_model_len": 34816, "max_num_seqs": 128}[key]}
    if changed_settings:
        effective["engine_settings"] = changed_settings
    if generation_options.get("decoding_policy") is not None:
        effective["decoding_policy"] = generation_options["decoding_policy"]
    result = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row.get("prompt", row.get("problem", "") + "\n" + INSTRUCTION)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        seed = (int(hashlib.sha256(str(row["id"]).encode()).hexdigest()[:8], 16) % 2**31
                if seed_by_id else params["seed"])
        result.append(GenerationBank.identity(str(row["id"]), prompt, effective, model_hash, seed))
    return result


def _base_plan(target, bank, keys, runner, job_key, checkpoint):
    """Freeze hits before dispatch; a prior job keeps its original complete row set."""
    path = target / "generation_plan.json"
    if path.exists():
        try:
            plan = json.loads(path.read_text())
            hashes = plan["cached_sha256"]
            if (plan["keys"] != keys or not isinstance(hashes, list) or len(hashes) != len(keys)
                    or type(plan["preserved_existing_job"]) is not bool
                    or plan["missing"] != [i for i, digest in enumerate(hashes) if digest is None]
                    or any(digest is not None and (not isinstance(digest, str) or len(digest) != 64
                           or any(c not in "0123456789abcdef" for c in digest)) for digest in hashes)
                    or (plan["preserved_existing_job"] and any(digest is not None for digest in hashes))):
                raise ValueError("Saved generation plan binding differs")
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise JobPending("Evaluation generation plan is invalid; no replacement sampling launched") from error
    else:
        existing = runner.has_job(job_key)
        # An old job has no checkpoint-content/prompt binding from before its
        # dispatch. Its result may resume evaluation but never seeds this bank.
        cached = [None if existing else bank.get(key) for key in keys]
        plan = {"keys": keys, "cached_sha256": [entry["output_sha256"] if entry else None for entry in cached],
                "missing": [i for i, entry in enumerate(cached) if entry is None],
                "preserved_existing_job": existing}
        atomic_write(path, plan)
        checkpoint()
    cached = [bank.get(key) if digest is not None else None
              for key, digest in zip(keys, plan["cached_sha256"], strict=True)]
    if any(digest is not None and (entry is None or entry["output_sha256"] != digest)
           for digest, entry in zip(plan["cached_sha256"], cached, strict=True)):
        raise JobPending("A planned cached evaluation generation is missing or changed; retain the original plan")
    return plan, cached


def _validate_generation(output, row, samples, max_tokens):
    if (not isinstance(output, dict) or output.get("id") != row["id"]
            or not isinstance(output.get("samples"), list) or len(output["samples"]) != samples):
        raise JobPending("Evaluation generation receipt differs from its saved row plan")
    for sample in output["samples"]:
        if (not isinstance(sample, dict) or not isinstance(sample.get("text"), str)
                or type(sample.get("token_count")) is not int or not 0 <= sample["token_count"] <= max_tokens
                or sample.get("finish_reason") not in {"stop", "length"}):
            raise JobPending("Evaluation receipt contains an incomplete or invalid generation")


def evaluate(name, checkpoint_dir, benchmark, samples, ledger, checkpoint, *, thinking=True, changes=None,
             engine="vllm", problem_limit=None, engine_settings=None, generation_options=None,
             extra_provenance=None):
    from openwsp.backends.modal_app import Generator, TransformerReader
    if engine not in {"vllm", "transformers"}:
        raise ValueError("Unsupported evaluation engine")
    if problem_limit is not None and (type(problem_limit) is not int or problem_limit < 1):
        raise ValueError("Problem limit must be a positive integer")
    engine_settings = dict(engine_settings or {})
    generation_options = dict(generation_options or {})
    extra_provenance = dict(extra_provenance or {})
    if (engine_settings or generation_options) and engine != "vllm":
        raise ValueError("Evaluation grid options require the vLLM engine")
    if set(engine_settings) - {"max_model_len", "max_num_seqs"}:
        raise ValueError("Unknown evaluation engine setting")
    if set(generation_options) - {"seed_by_id", "seed_scheme", "decoding_policy"}:
        raise ValueError("Unknown evaluation generation option")
    generator = Generator if engine == "vllm" else TransformerReader
    directory = Path("/data/results") / name / benchmark
    params = {**CONFIG["sampling"], **(changes or {}), "n": samples}
    # New variants must bind the immutable experiment specification before any
    # smoke/batch output is written. A resumed variant cannot silently reuse a
    # summary from different decoding settings, weights or submission.
    if engine_settings or generation_options or extra_provenance:
        expected = {"checkpoint": checkpoint_dir, "sampling": params, "thinking": thinking,
                    "engine": engine, "engine_settings": engine_settings,
                    "generation_options": generation_options, "provenance": extra_provenance,
                    "problem_limit": problem_limit}
        binding = directory / "evaluation_spec.json"
        if binding.exists():
            if json.loads(binding.read_text()) != expected:
                raise ValueError("Evaluation variant is bound to a different specification")
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError("Cannot attach a variant specification to existing unbound outputs")
        else:
            atomic_write(binding, expected)
            checkpoint()
    if (directory / "summary.json").exists():
        return json.loads((directory / "summary.json").read_text())
    rows = read_rows(Path("/data/benchmarks") / f"{benchmark}.jsonl")
    if problem_limit is not None:
        rows = rows[:problem_limit]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation requires nonempty, distinct problem IDs")
    bank, keys_by_id = None, {}
    if engine == "vllm" and str(checkpoint_dir) == "/data/original":
        identities = _base_generation_keys(rows, checkpoint_dir, params, thinking, engine_settings, generation_options)
        if identities is not None:
            bank = GenerationBank(Path("/data/forward-bank/base"))
            keys_by_id = {row["id"]: identity for row, identity in zip(rows, identities, strict=True)}
            # Detect unusable cache entries before even the independent smoke
            # starts. A corrupt entry never becomes a reason to sample again.
            try:
                for row, identity in zip(rows, identities, strict=True):
                    entry = bank.get(identity)
                    if entry is not None:
                        _validate_generation(entry["output"], row, samples, params["max_tokens"])
            except BankError as error:
                raise JobPending("Evaluation bank could not be verified before preflight") from error
    runner = ProtocolGPUJobs(ledger, "/data/jobs", commit=checkpoint)
    instance_settings = {**engine_settings}
    if generation_options.get("decoding_policy") is not None:
        instance_settings["enable_thinking_close"] = True
    batch_size = max(1, min(256, engine_settings.get("max_num_seqs", 256)) // samples)
    batches = [("smoke", rows[:4])] + [(f"batch_{i:03d}", rows[start:start+batch_size])
               for i, start in enumerate(range(0, len(rows), batch_size))]
    summaries, records, smoke_summary = [], [], None
    for label, selected in batches:
        target = directory / label
        if (target / "summary.json").exists():
            summary = json.loads((target / "summary.json").read_text())
        else:
            key = f"evaluation/{name}/{benchmark}/{label}"
            plan, cached = None, None
            if bank is not None and label != "smoke":
                try:
                    plan, cached = _base_plan(target, bank, [keys_by_id[row["id"]] for row in selected],
                                              runner, key, checkpoint)
                    for row, entry in zip(selected, cached, strict=True):
                        if entry is not None:
                            _validate_generation(entry["output"], row, samples, params["max_tokens"])
                except BankError as error:
                    raise JobPending("Evaluation bank could not be verified; no replacement sampling launched") from error
            missing = plan["missing"] if plan is not None else list(range(len(selected)))
            job_rows = [selected[index] for index in missing]
            fingerprint = {"checkpoint": checkpoint_dir, "sampling": params, "thinking": thinking,
                           "ids": [row["id"] for row in job_rows]}
            if engine != "vllm":
                fingerprint["engine"] = engine
            if engine_settings or generation_options or extra_provenance:
                fingerprint.update(engine_settings=engine_settings, generation_options=generation_options,
                                   extra_provenance=extra_provenance)
            if plan is not None and not plan["preserved_existing_job"]:
                fingerprint["generation_plan_sha256"] = json_hash(plan)
            if job_rows:
                runner.launch(key, lambda gpu, gate, job_rows=job_rows: gpu_instance(generator, gpu, gate, checkpoint=checkpoint_dir, **instance_settings).generate.spawn(job_rows, params, thinking, **generation_options),
                              fingerprint=fingerprint)
                result = runner.result(key, timeout=3600)
                seconds = result.get("_openwsp_gpu", {}).get("billable_seconds", result["container_seconds"] + 122)
                settled = result.get("_openwsp_gpu", {}).get("total_settled_cost_usd")
                cost = float(settled if settled is not None else compute_cost(seconds, gpu=result["gpu"], cpu_cores=4, memory_gib=16))
            else:
                result = {"outputs": [], "container_seconds": 0, "gpu": None,
                          "engine": CONFIG["engine"], "transformers": CONFIG["transformers"],
                          "engine_settings": engine_settings}
                seconds, cost = 0, 0.0
            if len(result["outputs"]) != len(job_rows):
                raise JobPending("Evaluation generation count differs from its saved row plan")
            for generated, row in zip(result["outputs"], job_rows, strict=True):
                _validate_generation(generated, row, samples, params["max_tokens"])
            bank_provenance = None
            if plan is not None:
                outputs = [entry["output"] if entry else None for entry in cached]
                published = {}
                if job_rows and not plan["preserved_existing_job"] and (
                    result["engine"] != CONFIG["engine"] or result["transformers"] != CONFIG["transformers"]
                ):
                    raise JobPending("Evaluation engine version differs from its generation bank identity")
                for index, generated in zip(missing, result["outputs"], strict=True):
                    outputs[index] = generated
                    if not plan["preserved_existing_job"]:
                        try:
                            entry = bank.put(plan["keys"][index], generated, provenance={
                                "job_key": key, "stage": name, "benchmark": benchmark,
                                "engine": result["engine"], "transformers": result["transformers"],
                                "source_batch_seconds": seconds, "source_batch_cost_usd": cost,
                                "cost_scope": "Entire source batch, shared by its rows; not a per-row charge"})
                        except BankError as error:
                            raise JobPending("Successful evaluation receipt could not enter the verified bank") from error
                        if entry["output_sha256"] != json_hash(generated):
                            raise JobPending("An existing bank result differs from this successful job; retain both receipts")
                        published[index] = entry
                for generated, row in zip(outputs, selected, strict=True):
                    _validate_generation(generated, row, samples, params["max_tokens"])
                result = {**result, "outputs": outputs}
                bank_provenance = {
                    "enabled": True, "reused_rows": len(selected) - len(missing), "generated_rows": len(missing),
                    "plan_sha256": json_hash(plan), "historical_job_not_published": plan["preserved_existing_job"],
                    "incremental_scoring_batch_cost_usd": cost,
                    "rows": [{"problem_id": row["id"], "reused": cached[index] is not None,
                              "bank_key": (cached[index] or published.get(index) or {}).get("key"),
                              "output_sha256": json_hash(outputs[index]),
                              "source": (cached[index] or published.get(index) or {}).get("provenance")}
                             for index, row in enumerate(selected)],
                }
                checkpoint()
            summary = evaluate_outputs(result["outputs"], selected, directory=target, stage=name,
                                       benchmark=benchmark, samples=samples, seconds=seconds,
                                       cost_usd=cost,
                                       thinking=thinking, provenance={"revision": CONFIG["revision"], "sampling": params,
                                       "thinking": thinking, "gpu": result["gpu"], "engine": result["engine"],
                                       "engine_settings": result.get("engine_settings", {}),
                                       "seconds_basis": "accounted_resource_seconds_including_margins_and_cancelled_attempts",
                                       "container_seconds": result["container_seconds"],
                                       "transformers": result["transformers"], **extra_provenance,
                                       **({"generation_bank": bank_provenance} if bank_provenance is not None else {})})
            checkpoint()
        if label != "smoke":
            summaries.append(summary)
            records.extend(read_rows(target / "samples.jsonl"))
        else:
            smoke_summary = summary
    n = len(records)
    correct = sum(row["correct"] for row in records)
    accuracy = correct / n
    # Report the same per-sample binomial SE as the release; all samples and
    # problem identities remain available for a clustered reanalysis.
    summary = {**summaries[0], "stage": name, "n": n, "correct": correct,
               "accuracy": accuracy, "standard_error": math.sqrt(accuracy*(1-accuracy)/n),
               **finished_metrics(records),
               "mean_tokens": sum(row["token_count"] for row in records)/n,
               "cut_share": sum(row["finish_reason"] == "length" for row in records)/n,
               "seconds": sum(row["seconds"] for row in summaries),
               "cost_usd": sum(row["cost_usd"] for row in summaries)}
    summary["provenance"].update(batch_generations=batch_size * samples, standard_error="binomial_per_sample",
                                 smoke_included_in_score=False,
                                 smoke_seconds=smoke_summary["seconds"], smoke_cost_usd=smoke_summary["cost_usd"],
                                 batches=[{"gpu": row["provenance"]["gpu"], "seconds": row["seconds"],
                                           "cost_usd": row["cost_usd"]} for row in summaries])
    if any("generation_bank" in row["provenance"] for row in summaries):
        summary["provenance"]["generation_bank"] = {
            "reused_rows": sum(row["provenance"].get("generation_bank", {}).get("reused_rows", 0) for row in summaries),
            "generated_rows": sum(row["provenance"].get("generation_bank", {}).get("generated_rows", 0) for row in summaries),
            "batch_provenance": [row["provenance"].get("generation_bank") for row in summaries],
            "smoke_generations_never_enter_bank": True,
        }
    if problem_limit is not None:
        summary["provenance"].update(problem_selection="first problems in saved benchmark order",
                                     problem_limit=problem_limit, problem_ids=[row["id"] for row in rows])
    write_rows(directory / "samples.jsonl", records)
    atomic_write(directory / "summary.json", summary)
    checkpoint()
    return summary


def evaluate_live(submission, ledger, checkpoint, *, stage="live-run"):
    results = {}
    for benchmark, samples in CONFIG["samples"].items():
        if stage != "live-run":
            directory = Path("/data/results") / stage / benchmark
            binding = directory / "submission.json"
            expected = {"checkpoint": submission["checkpoint"], "submission_sha256": submission["sha256"],
                        "stage": stage, "benchmark": benchmark, "samples_per_problem": samples}
            if binding.exists():
                if json.loads(binding.read_text()) != expected:
                    raise ValueError("Evaluation is bound to a different submitted edit")
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError("Cannot attach a submission hash to pre-existing unbound evaluations")
                atomic_write(binding, expected)
                checkpoint()
        summary = evaluate(stage, submission["checkpoint"], benchmark, samples, ledger, checkpoint)
        if stage != "live-run":
            prior = summary.get("provenance", {}).get("submission_sha256")
            if prior is not None and prior != submission["sha256"]:
                raise ValueError("Evaluation belongs to a different submitted edit")
            summary.setdefault("provenance", {})["submission_sha256"] = submission["sha256"]
            atomic_write(Path("/data/results") / stage / benchmark / "summary.json", summary)
            checkpoint()
        results[benchmark] = summary
    return results
