"""One-shot local continuation for the authorized 484-example CPU extraction.

This process waits only for the PID named by its immutable run receipt. It never
restarts extraction, runs inference, scores an edit, uses Modal, or pushes Git.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path

import numpy as np

from openwsp.budget import atomic_json
from replications.exp8.repair import DOSES, _file_hash, _json_hash, build_repair, repair_tensor

EXPECTED_COUNTS = {"looping": 450, "finished": 34}
LAYERS = (15, 19)
DOSE_SPEC_FIELDS = ("coefficients", "base_variant", "fixed_directions_and_columns", "problem_ids",
                    "ordered_ids_sha256", "selection_source_sha256", "samples_per_problem", "sampling", "seed_scheme")


def _dose_identity(value):
    return _json_hash({field: value[field] for field in DOSE_SPEC_FIELDS})


def verify_extraction(directory, exit_path, *, expected_counts, widths):
    """Verify every completed receipt and independently rederive the mean contrast."""
    directory, exit_path = Path(directory), Path(exit_path)
    if not exit_path.is_file() or exit_path.read_text().strip() != "0":
        raise ValueError("The original extraction has no successful exit receipt")
    selection = json.loads((directory / "selection.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    examples = selection["examples"]
    expected_n = sum(expected_counts.values())
    if (summary.get("status") != "complete" or summary.get("processed") != expected_n
            or summary.get("selected") != expected_n or summary.get("counts") != expected_counts
            or len(examples) != expected_n or selection.get("layers") != list(widths)
            or len({row["id"] for row in examples}) != expected_n
            or summary.get("selection_sha256") != _file_hash(directory / "selection.json")):
        raise ValueError("Extraction completion/selection identity differs")
    counts = {group: 0 for group in expected_counts}
    sums = {group: {str(layer): np.zeros(width, dtype=np.float64) for layer, width in widths.items()}
            for group in expected_counts}
    receipt_keys, array_hashes = [], []
    binding_hash = _json_hash(selection)
    for example in examples:
        if example.get("group") not in counts:
            raise ValueError("Unexpected contrast group")
        for field in ("prompt_sha256", "text_sha256"):
            digest = example.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Selected text identity is malformed")
        key = _json_hash({"binding": binding_hash, "example": example})
        receipt_keys.append(key)
        folder = directory / "examples" / key
        receipt = json.loads((folder / "receipt.json").read_text())
        if (receipt.get("request_sha256") != key or receipt.get("example") != example
                or receipt.get("arrays_sha256") != _file_hash(folder / "means.npz")):
            raise ValueError("Per-example receipt or activation identity differs")
        tokens = receipt.get("tokens", {})
        if (type(tokens.get("prompt_tokens")) is not int or tokens["prompt_tokens"] <= 0
                or type(tokens.get("answer_tokens")) is not int or not 1 <= tokens["answer_tokens"] <= 8192
                or tokens["prompt_tokens"] + tokens["answer_tokens"] > 34816
                or not isinstance(tokens.get("input_ids_sha256"), str)
                or len(tokens["input_ids_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in tokens["input_ids_sha256"])):
            raise ValueError("Per-example token identity is invalid")
        with np.load(folder / "means.npz", allow_pickle=False) as archive:
            if set(archive.files) != {str(layer) for layer in widths}:
                raise ValueError("Activation layers differ")
            for layer, width in widths.items():
                mean = archive[str(layer)]
                if mean.dtype != np.float32 or mean.shape != (width,) or not np.isfinite(mean).all():
                    raise ValueError("Per-example activation shape/dtype/value differs")
                sums[example["group"]][str(layer)] += mean
        counts[example["group"]] += 1
        array_hashes.append(receipt["arrays_sha256"])
    actual_keys = {path.parent.name for path in (directory / "examples").glob("*/receipt.json")}
    if actual_keys != set(receipt_keys) or counts != expected_counts:
        raise ValueError("Missing, extra or misgrouped extraction receipts")
    differences = {f"model.language_model.layers.{layer}.mlp.down_proj.weight":
                   sums["looping"][str(layer)] / counts["looping"]
                   - sums["finished"][str(layer)] / counts["finished"] for layer in widths}
    artifact = directory / "input-differences.npz"
    if summary.get("input_differences_sha256") != _file_hash(artifact):
        raise ValueError("Completed input contrast digest differs")
    with np.load(artifact, allow_pickle=False) as archive:
        if set(archive.files) != set(differences) or any(
            archive[name].dtype != np.float64 or not np.array_equal(archive[name], values)
            for name, values in differences.items()
        ):
            raise ValueError("Completed input contrast differs from the verified per-example means")
    return differences, {
        "counts": counts, "verified_examples": expected_n,
        "selection_sha256": summary["selection_sha256"],
        "input_differences_sha256": summary["input_differences_sha256"],
        "ordered_receipt_keys_sha256": _json_hash(receipt_keys),
        "ordered_activation_hashes_sha256": _json_hash(array_hashes),
    }


def _process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # Field 22 follows the parenthesized command and state at field 3.
        fields = stat.rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        start_ticks = fields[19]
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return {"start_ticks": start_ticks, "argv": [part.decode() for part in command if part]}
    except FileNotFoundError:
        return None


def wait_for_original(root, *, poll_seconds=30):
    """Wait once for the original invocation; never dispatch a replacement."""
    inputs = root / ".openwsp/repair-inputs"
    invocation_path = inputs / "full-extraction-invocation.json"
    invocation = json.loads(invocation_path.read_text())
    pid = invocation["pid"]
    if type(pid) is not int or pid <= 0 or invocation.get("cpu_threads") != 2:
        raise ValueError("Unexpected original extraction invocation")
    expected = shlex.split(invocation["command"])
    expected = expected[expected.index("-m"):]
    if expected[:2] != ["-m", "replications.exp8.repair_activations"] or "--limit" in expected:
        raise ValueError("Invocation does not identify the authorized full extraction")
    identity = _process_identity(pid)
    if identity is not None and identity["argv"][1:] != expected:
        raise ValueError("PID does not match the recorded extraction command")
    binding = {"pid": pid, "process": identity, "invocation_sha256": _file_hash(invocation_path),
               "selection_sha256": _file_hash(root / ".openwsp/repair-input-means/selection.json"),
               "dose_spec_sha256": _dose_identity(json.loads((root / "replications/exp8/results/repair/dose-sweep.json").read_text()))}
    path = inputs / "completion-binding.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if (prior["invocation_sha256"] != binding["invocation_sha256"]
                or prior["selection_sha256"] != binding["selection_sha256"]
                or prior["dose_spec_sha256"] != binding["dose_spec_sha256"]):
            raise ValueError("Continuation is already bound to a different extraction")
        if identity is not None and prior["process"] != identity:
            raise ValueError("The recorded PID belongs to a different process instance")
        binding = prior
    else:
        atomic_json(path, binding)
    deadline = time.monotonic() + 24 * 3600
    while identity is not None:
        if time.monotonic() > deadline:
            raise TimeoutError("One-shot continuation exceeded its wait bound; extraction was not stopped")
        time.sleep(poll_seconds)
        identity = _process_identity(pid)
        if identity is not None and identity != binding["process"]:
            raise ValueError("Original PID was replaced before its exit could be verified")
    # The shell writes its exit receipt immediately after reaping the child.
    for _ in range(10):
        if (inputs / "full-extraction.exit").exists():
            break
        time.sleep(1)
    if _file_hash(invocation_path) != binding["invocation_sha256"]:
        raise ValueError("Original invocation changed during extraction")
    return binding


def complete(root):
    """Verify this exact extraction, build fixed-direction variants, then publish compact files."""
    import torch

    from openwsp.core import WeightStore
    from openwsp.edits import tensor_sha256
    from replications.exp8.repair_activations import select_examples

    root = Path(root).resolve()
    inputs = root / ".openwsp/repair-inputs"
    activation_dir = root / ".openwsp/repair-input-means"
    model_dir = root / ".openwsp/research-model"
    method_path = root / ".openwsp/research-control2-method.json"
    method = json.loads(method_path.read_text())
    selection = json.loads((activation_dir / "selection.json").read_text())
    binding = json.loads((inputs / "completion-binding.json").read_text())
    if method["counts"] != EXPECTED_COUNTS or _file_hash(activation_dir / "selection.json") != binding["selection_sha256"]:
        raise ValueError("The selected calibration contrast differs from this authorized run")
    provenance = selection["provenance"]
    for path, field in ((inputs / "raw_generations.jsonl", "raw_sha256"),
                        (inputs / "math500.jsonl", "benchmarks_sha256"), (method_path, "saved_method_sha256")):
        if _file_hash(path) != provenance[field]:
            raise ValueError("Extraction source changed")
    for mapping in (provenance["checkpoint_files"], provenance["model_assets"]):
        for name, digest in mapping.items():
            if Path(name).name != name or _file_hash(model_dir / name) != digest:
                raise ValueError("Pinned checkpoint/tokenizer asset changed")
    raw = [json.loads(line) for line in (inputs / "raw_generations.jsonl").read_text().splitlines()]
    benchmarks = [json.loads(line) for line in (inputs / "math500.jsonl").read_text().splitlines()]
    examples = select_examples(raw, benchmarks, EXPECTED_COUNTS)
    identities = [{key: row[key] for key in ("id", "group", "prompt_sha256", "text_sha256")} for row in examples]
    if identities != selection["examples"]:
        raise ValueError("Selected text identities do not reproduce the original contrast")
    store = WeightStore(model_dir)
    directions = {f"model.language_model.layers.{layer}.mlp.down_proj.weight": method["layers"][str(layer)]["direction"]
                  for layer in LAYERS}
    widths = {layer: store.load_original(f"model.language_model.layers.{layer}.mlp.down_proj.weight").shape[1]
              for layer in LAYERS}
    differences, verified = verify_extraction(activation_dir, inputs / "full-extraction.exit",
                                              expected_counts=EXPECTED_COUNTS, widths=widths)
    base_identity = json.loads((root / ".openwsp/research-prepare.json").read_text())
    repair_provenance = {
        "source": ".openwsp/repair-input-means", "saved_method_sha256": _file_hash(method_path),
        "activation_reduction": selection["activation_reduction"], "input_verification": verified,
        "output_direction_source": "Existing control2 saved output directions, unchanged across 2a/2b/2c and doses",
        "text_sets": {group: [{"id": row["id"], "text_sha256": row["text_sha256"]} for row in identities
                              if row["group"] == group] for group in EXPECTED_COUNTS},
        "base_model": "Qwen/Qwen3.5-0.8B", "base_revision": base_identity["model_revision"],
        "model_provenance": provenance,
    }
    built = {}
    for alpha in DOSES:
        folder = root / ".openwsp/repair-build" / ("2c" if alpha == 6 else f"dose/alpha_{alpha}")
        if not folder.exists():
            build_repair(store, folder, directions=directions, input_differences=differences,
                         alpha=alpha, column_rule="topk", top_k=64, provenance=repair_provenance)
        saved_method = json.loads((folder / "method.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        if saved_method["provenance"] != repair_provenance:
            raise ValueError("Existing repair bundle has different provenance")
        check = WeightStore(model_dir)
        manifest = check.apply_edit(folder / "edit")
        check.validate_submission()
        for name, direction in directions.items():
            expected, specification = repair_tensor(store.load_original(name).numpy(), direction,
                alpha=alpha, column_rule="topk", top_k=64, input_difference=differences[name])
            if tensor_sha256(check.load(name)) != tensor_sha256(torch.from_numpy(expected)):
                raise ValueError("Built edit differs from the fixed-direction repair formula")
            if any(saved_method["layers"][name][key] != specification[key] for key in ("alpha", "columns", "direction")):
                raise ValueError("Built edit direction, dose or selected columns differ")
            for earlier in ("2a", "2b"):
                original_method = json.loads((root / f".openwsp/repair-build/{earlier}/method.json").read_text())
                if original_method["layers"][name]["direction"] != specification["direction"]:
                    raise ValueError("2c direction differs from the already built 2a/2b direction")
        if manifest["edit_sha256"] != summary["edit_sha256"] or _file_hash(folder / "method.json") != summary["method_sha256"]:
            raise ValueError("Built artifact summary is inconsistent")
        built[alpha] = {"summary": summary, "manifest": manifest, "method": saved_method,
                        "private_bundle_path": str((folder / "edit").relative_to(root))}
    # Every mean and every private bundle has passed before public results change.
    public = root / "replications/exp8/results/repair"
    dose_path = public / "dose-sweep.json"
    dose = json.loads(dose_path.read_text())
    aggregate = json.loads((public / "summary.json").read_text())
    prior = aggregate["variants"]["2c"]
    if prior.get("edit_sha256") not in (None, built[6]["summary"]["edit_sha256"]):
        raise ValueError("Public 2c result already names a different edit")
    if prior.get("accuracy") is not None:
        raise ValueError("A scored 2c result exists; the CPU continuation will not overwrite it")
    if _dose_identity(dose) != binding["dose_spec_sha256"]:
        raise ValueError("Preregistered dose sweep changed during extraction")
    if (dose["coefficients"] != list(DOSES) or dose["base_variant"] != "2c"
            or len(dose["problem_ids"]) != 100 or len(set(dose["problem_ids"])) != 100
            or _json_hash(dose["problem_ids"]) != dose["ordered_ids_sha256"]):
        raise ValueError("Preregistered dose sweep identity differs")
    for alpha, artifact in built.items():
        target = public / ("2c" if alpha == 6 else f"dose/alpha_{alpha}")
        atomic_json(target / "edit-manifest.json", artifact["manifest"])
        compact = {**artifact["summary"], "private_bundle_path": artifact["private_bundle_path"],
                   "public_edit_manifest": "edit-manifest.json", "input_verification": verified}
        compact.pop("edit_path")
        atomic_json(target / "summary.json", compact)
    atomic_json(public / "2c/method.json", built[6]["method"])
    dose.update(build_status="built", input_statistics_sha256=verified["input_differences_sha256"],
                edits=[{"alpha": alpha, "edit_sha256": artifact["summary"]["edit_sha256"],
                        "private_bundle_path": artifact["private_bundle_path"]} for alpha, artifact in built.items()])
    atomic_json(dose_path, dose)
    aggregate["variants"]["2c"] = {**built[6]["summary"], "summary_path": "2c/summary.json",
                                   "private_bundle_path": built[6]["private_bundle_path"]}
    aggregate["variants"]["2c"].pop("edit_path")
    aggregate["interpretation"] = "2a/2b/2c and the fixed-mask dose edits are CPU-built on the pinned base. No repair score is claimed; all GPU scoring remains pending."
    atomic_json(public / "summary.json", aggregate)
    result = {"status": "complete", "input_verification": verified,
              "edit_sha256": built[6]["summary"]["edit_sha256"], "built_doses": list(DOSES),
              "scoring": "not_run", "gpu_calls": 0, "original_pid": binding["pid"]}
    atomic_json(inputs / "completion-result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    os.chdir(root)
    try:
        wait_for_original(root)
        print(json.dumps(complete(root), indent=2, allow_nan=False))
    except Exception as error:
        atomic_json(root / ".openwsp/repair-inputs/completion-error.json",
                    {"status": "blocked", "exception": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
