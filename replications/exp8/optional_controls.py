"""Optional controls fixed before evaluation; never supplied to the live editor."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from openwsp.core import WeightStore
from replications.exp8.evaluation import evaluate
from replications.exp8.forward import freeze_checkpoint
from replications.exp8.runtime import CONFIG, atomic_write, read_rows, write_rows


def random_directions(width=1024):
    rng = np.random.default_rng(0)
    directions = {}
    for layer in ("15", "19"):
        value = torch.tensor(rng.standard_normal(width), dtype=torch.float32)
        directions[layer] = value / value.norm()
    return directions


def random_edit(checkpoint):
    folder = Path("/data/controls/random")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "submission.json").exists():
        return json.loads((folder / "submission.json").read_text())
    reference = WeightStore("/model")
    reference.apply_edit("/data/reference_edit")
    store = WeightStore("/model", folder / "edited")
    metadata = {"seed": 0, "rng": "numpy.default_rng / PCG64", "alpha": 6, "layers": {},
                "protocol": "Independent standard-normal vectors at layers 15 then 19, converted to float32 and normalized. Replace the p50c directions, preserving its changed-column mask and alpha. No fitting or selection.",
                "columns": "Exactly the columns changed by the released p50c edit"}
    for layer, direction in random_directions().items():
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        base = store.load_original(name)
        columns = (reference.load(name) != base).any(0)
        writer = direction @ base
        writer[~columns] = 0
        store.save_edit(name, base - 6 * torch.outer(direction, writer))
        metadata["layers"][layer] = {"columns": columns.nonzero().flatten().tolist()}
    checkpoint_dir, edit_id, names = freeze_checkpoint(folder, "/data/control-checkpoints")
    store.export_edit(folder / "edit", base_model=CONFIG["model"], revision=CONFIG["revision"])
    submission = {"checkpoint": checkpoint_dir, "sha256": edit_id, "tensors": names}
    atomic_write(folder / "method.json", metadata)
    atomic_write(folder / "submission.json", submission)
    checkpoint()
    return submission


def reused_vllm_subset(data=Path("/data")):
    """Compare the same fixed fifty IDs without buying duplicate generations."""
    data = Path(data)
    base = data / "results/calibration_base"
    source = json.loads((base / "summary.json").read_text())
    ids = [row["id"] for row in read_rows(data / "benchmarks/math500.jsonl")[:50]]
    by_id = {row["problem_id"]: row for row in read_rows(base / "samples.jsonl")}
    records = [by_id[identifier] for identifier in ids]
    n = len(records)
    correct = sum(row["correct"] for row in records)
    accuracy = correct / n
    summary = {**source, "stage": "control5_vllm_reused", "n": n, "correct": correct,
               "accuracy": accuracy, "standard_error": math.sqrt(accuracy * (1 - accuracy) / n),
               "mean_tokens": sum(row["token_count"] for row in records) / n,
               "cut_share": sum(row["finish_reason"] == "length" for row in records) / n,
               "seconds": 0, "cost_usd": 0,
               "provenance": {**source["provenance"], "problem_ids": ids,
                              "problem_selection": "first 50 problems in saved benchmark order",
                              "standard_error": "binomial_per_sample", "reused": True,
                              "source_stage": "calibration_base",
                              "seconds_basis": "incremental only; no new GPU work; original cost is in calibration_base",
                              "cost_basis": "incremental only; original cost is in calibration_base"}}
    destination = data / "results/control5_vllm_reused/math500"
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "samples.jsonl", records)
    atomic_write(destination / "summary.json", summary)
    return summary


def run_optional_control(stage, ledger, checkpoint):
    # Required controls have priority even if someone invokes this stage directly.
    for name in ("control1_nothink", "control2_mechanical", "control3_repetition"):
        path = Path("/data/results") / name / "math500/summary.json"
        if not path.exists() or json.loads(path.read_text()).get("status") != "done":
            raise ValueError("Optional controls wait for all three required controls")
    if stage == "control4":
        submission = random_edit(checkpoint)
        return evaluate("control4_random", submission["checkpoint"], "math500", 1, ledger, checkpoint)
    if stage == "control5":
        baseline = reused_vllm_subset()
        checkpoint()
        result = evaluate("control5_transformers", "/data/original", "math500", 1, ledger, checkpoint,
                          engine="transformers", problem_limit=50)
        return {"transformers": result, "vllm_reused": baseline}
    raise ValueError("Unknown optional control")
