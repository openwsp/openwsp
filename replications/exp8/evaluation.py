"""Resumable evaluation batches with the authors' fixed sampling and graders."""
from __future__ import annotations

import json
import math
from pathlib import Path

from openwsp.backends.jobs import RemoteJobRunner
from openwsp.budget import compute_cost
from replications.exp8.runtime import CONFIG, atomic_write, evaluate_outputs, read_rows, write_rows


def evaluate(name, checkpoint_dir, benchmark, samples, ledger, checkpoint, *, thinking=True, changes=None):
    from openwsp.backends.modal_app import Generator
    directory = Path("/data/results") / name / benchmark
    if (directory / "summary.json").exists():
        return json.loads((directory / "summary.json").read_text())
    rows = read_rows(Path("/data/benchmarks") / f"{benchmark}.jsonl")
    params = {**CONFIG["sampling"], **(changes or {}), "n": samples}
    runner = RemoteJobRunner(ledger, "/data/jobs", commit=checkpoint)
    batch_size = max(1, 256 // samples)
    batches = [("smoke", rows[:4])] + [(f"batch_{i:03d}", rows[start:start+batch_size])
               for i, start in enumerate(range(0, len(rows), batch_size))]
    summaries, records = [], []
    for label, selected in batches:
        target = directory / label
        if (target / "summary.json").exists():
            summary = json.loads((target / "summary.json").read_text())
        else:
            key = f"evaluation/{name}/{benchmark}/{label}"
            runner.launch(key, lambda: Generator(checkpoint=checkpoint_dir).generate.spawn(selected, params, thinking),
                          upper_bound_usd=compute_cost(2520, gpu="H100", cpu_cores=4, memory_gib=16),
                          cost_options={"gpu": "H100", "cpu_cores": 4, "memory_gib": 16,
                                        "startup_seconds": 120, "scaledown_seconds": 2},
                          fingerprint={"checkpoint": checkpoint_dir, "sampling": params, "thinking": thinking,
                                       "ids": [row["id"] for row in selected]})
            result = runner.result(key, timeout=2520)
            seconds = result["container_seconds"] + 122
            summary = evaluate_outputs(result["outputs"], selected, directory=target, stage=name,
                                       benchmark=benchmark, samples=samples, seconds=seconds,
                                       cost_usd=float(compute_cost(seconds, gpu="H100", cpu_cores=4, memory_gib=16)),
                                       thinking=thinking, provenance={"revision": CONFIG["revision"], "sampling": params,
                                       "thinking": thinking, "gpu": result["gpu"], "engine": result["engine"],
                                       "transformers": result["transformers"]})
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
               "mean_tokens": sum(row["token_count"] for row in records)/n,
               "cut_share": sum(row["finish_reason"] == "length" for row in records)/n,
               "seconds": sum(row["seconds"] for row in summaries),
               "cost_usd": sum(row["cost_usd"] for row in summaries)}
    summary["provenance"].update(batch_generations=256, standard_error="binomial_per_sample",
                                 smoke_included_in_score=False)
    write_rows(directory / "samples.jsonl", records)
    atomic_write(directory / "summary.json", summary)
    checkpoint()
    return summary


def evaluate_live(submission, ledger, checkpoint):
    return {benchmark: evaluate("live-run", submission["checkpoint"], benchmark, samples, ledger, checkpoint)
            for benchmark, samples in CONFIG["samples"].items()}
