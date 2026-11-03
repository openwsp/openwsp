"""Resumable evaluation batches with the authors' fixed sampling and graders."""
from __future__ import annotations

import json
import math
from pathlib import Path

from openwsp.budget import compute_cost
from openwsp.metrics import finished_metrics
from replications.exp8.jobs import ProtocolGPUJobs, gpu_instance
from replications.exp8.runtime import CONFIG, atomic_write, evaluate_outputs, read_rows, write_rows


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
    runner = ProtocolGPUJobs(ledger, "/data/jobs", commit=checkpoint)
    batch_size = max(1, min(256, engine_settings.get("max_num_seqs", 256)) // samples)
    batches = [("smoke", rows[:4])] + [(f"batch_{i:03d}", rows[start:start+batch_size])
               for i, start in enumerate(range(0, len(rows), batch_size))]
    summaries, records = [], []
    for label, selected in batches:
        target = directory / label
        if (target / "summary.json").exists():
            summary = json.loads((target / "summary.json").read_text())
        else:
            key = f"evaluation/{name}/{benchmark}/{label}"
            fingerprint = {"checkpoint": checkpoint_dir, "sampling": params, "thinking": thinking,
                           "ids": [row["id"] for row in selected]}
            if engine != "vllm":
                fingerprint["engine"] = engine
            if engine_settings or generation_options or extra_provenance:
                fingerprint.update(engine_settings=engine_settings, generation_options=generation_options,
                                   extra_provenance=extra_provenance)
            runner.launch(key, lambda gpu, gate, selected=selected: gpu_instance(generator, gpu, gate, checkpoint=checkpoint_dir, **engine_settings).generate.spawn(selected, params, thinking, **generation_options),
                          fingerprint=fingerprint)
            result = runner.result(key, timeout=3600)
            seconds = result.get("_openwsp_gpu", {}).get("billable_seconds", result["container_seconds"] + 122)
            summary = evaluate_outputs(result["outputs"], selected, directory=target, stage=name,
                                       benchmark=benchmark, samples=samples, seconds=seconds,
                                       cost_usd=float(result.get("_openwsp_gpu", {}).get("total_settled_cost_usd") or compute_cost(seconds, gpu=result["gpu"], cpu_cores=4, memory_gib=16)),
                                       thinking=thinking, provenance={"revision": CONFIG["revision"], "sampling": params,
                                       "thinking": thinking, "gpu": result["gpu"], "engine": result["engine"],
                                       "engine_settings": result.get("engine_settings", {}),
                                       "seconds_basis": "accounted_resource_seconds_including_margins_and_cancelled_attempts",
                                       "container_seconds": result["container_seconds"],
                                       "transformers": result["transformers"], **extra_provenance})
            checkpoint()
        if label != "smoke":
            summaries.append(summary)
            records.extend(read_rows(target / "samples.jsonl"))
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
                                 batches=[{"gpu": row["provenance"]["gpu"], "seconds": row["seconds"],
                                           "cost_usd": row["cost_usd"]} for row in summaries])
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
