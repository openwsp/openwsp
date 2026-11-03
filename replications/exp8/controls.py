"""Predeclared mechanical controls, isolated from the live editor."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from openwsp.backends.jobs import RemoteJobRunner
from openwsp.budget import compute_cost
from openwsp.core import WeightStore
from replications.exp8.evaluation import evaluate
from replications.exp8.forward import freeze_checkpoint
from replications.exp8.runtime import CONFIG, INSTRUCTION, atomic_write, read_rows


def loop_fraction(text):
    words = text[-8192:].split()[-1024:]
    grams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
    return 1-len(set(grams))/len(grams) if grams else 0.0


def mechanical_edit(ledger, checkpoint):
    from openwsp.backends.modal_app import TransformerReader, data_volume
    folder = Path("/data/controls/mechanical")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "submission.json").exists():
        return json.loads((folder / "submission.json").read_text())
    rows = {row["id"]: row for row in read_rows("/data/benchmarks/math500.jsonl")}
    raw = read_rows("/data/results/calibration_base/raw_generations.jsonl")
    examples = []
    for record in raw:
        finished = record["finish_reason"] == "stop"
        if not finished and loop_fraction(record["text"]) < 0.5:
            continue
        examples.append({"id": record["problem_id"], "text": record["text"],
                         "prompt": rows[record["problem_id"]]["problem"] + "\n" + INSTRUCTION,
                         "group": "finished" if finished else "looping"})
    if {row["group"] for row in examples} != {"looping", "finished"}:
        raise ValueError("Calibration produced no valid looping/finished contrast for control 2")
    runner = RemoteJobRunner(ledger, "/data/jobs", commit=checkpoint)
    totals = {group: {str(layer): np.zeros(1024) for layer in (15, 19)} for group in ("looping", "finished")}
    counts = {"looping": 0, "finished": 0}
    # Smoke on four genuine examples, then bounded teacher-forced batches.
    batches = [("smoke", examples[:4])] + [(f"batch_{i:03d}", examples[start:start+32])
               for i, start in enumerate(range(0, len(examples), 32))]
    for label, batch in batches:
        key = f"control2/directions/{label}"
        runner.launch(key, lambda: TransformerReader().mean_mlp_outputs.spawn(batch),
                      upper_bound_usd=compute_cost(2520, gpu="H100", cpu_cores=4, memory_gib=16),
                      cost_options={"gpu": "H100", "cpu_cores": 4, "memory_gib": 16,
                                    "startup_seconds": 120, "scaledown_seconds": 2},
                      fingerprint={"ids": [row["id"] for row in batch], "layers": [15, 19],
                                   "answer_tokens": 8192, "loop_tail_duplicate8_min": 0.5})
        result = runner.result(key, timeout=2520)
        if label != "smoke":
            for group in counts:
                counts[group] += result["counts"][group]
                for layer in ("15", "19"):
                    totals[group][layer] += np.array(result["sums"][group][layer])
    reference = WeightStore("/model")
    reference.apply_edit("/data/reference_edit")
    store = WeightStore("/model", folder / "edited")
    metadata = {"counts": counts, "layers": {}, "alpha": 6,
                "protocol": "Equal-weight answer means over first 8192 answer tokens; looping = length-cut and duplicate final-tail 8-gram share >=0.5; finished = stop. No correctness labels used.",
                "columns": "Exactly the columns changed by the released p50c edit", "benchmark_reuse": "MATH-500 calibration answers define this diagnostic control; it is not a held-out estimate."}
    for layer in ("15", "19"):
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        base = store.load_original(name)
        columns = (reference.load(name) != base).any(0)
        direction = torch.tensor(totals["looping"][layer]/counts["looping"] - totals["finished"][layer]/counts["finished"], dtype=torch.float32)
        if not torch.isfinite(direction).all() or direction.norm() == 0:
            raise ValueError("Mechanical direction is zero or nonfinite")
        direction /= direction.norm()
        writer = direction @ base
        writer[~columns] = 0
        store.save_edit(name, base - 6*torch.outer(direction, writer))
        metadata["layers"][layer] = {"columns": columns.nonzero().flatten().tolist(), "direction": direction.tolist()}
    checkpoint_dir, edit_id, names = freeze_checkpoint(folder, "/data/control-checkpoints")
    store.export_edit(folder / "edit", base_model=CONFIG["model"], revision=CONFIG["revision"])
    submission = {"checkpoint": checkpoint_dir, "sha256": edit_id, "tensors": names}
    atomic_write(folder / "method.json", metadata)
    atomic_write(folder / "submission.json", submission)
    data_volume.commit()
    return submission


def run_controls(ledger, checkpoint):
    results = {}
    results["control1_nothink"] = evaluate("control1_nothink", "/data/original", "math500", 1, ledger, checkpoint,
        thinking=False, changes={"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0,
                                 "presence_penalty": 2.0, "repetition_penalty": 1.0})
    mechanical = mechanical_edit(ledger, checkpoint)
    results["control2_mechanical"] = evaluate("control2_mechanical", mechanical["checkpoint"], "math500", 1, ledger, checkpoint)
    results["control3_repetition"] = evaluate("control3_repetition", "/data/original", "math500", 1, ledger, checkpoint,
                                             changes={"repetition_penalty": 1.1})
    return results
