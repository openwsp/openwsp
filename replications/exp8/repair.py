"""CPU construction of contrastive repetition repairs; no activation extraction.

The numerical builder uses NumPy. The export boundary uses WeightStore's normal
refusals and bf16 format-v1 serialization. Source texts and activations are supplied
by the caller; this module never runs a model or chooses examples from scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

VARIANTS = {
    "2a": {"column_rule": "all", "alpha": 1.0, "top_k": None},
    "2b": {"column_rule": "all", "alpha": 6.0, "top_k": None},
    "2c": {"column_rule": "topk", "alpha": 6.0, "top_k": 64},
}
DOSES = (0, 1, 2, 3, 4, 6, 8)


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _array_identity(value):
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _provenance_counts(provenance):
    if "saved_method_sha256" in provenance:
        digest = provenance["saved_method_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("saved_method_sha256 must be a lowercase SHA256 digest")
    if not provenance.get("text_sets"):
        return None
    sets = provenance["text_sets"]
    if not isinstance(sets, dict) or set(sets) != {"looping", "finished"}:
        raise ValueError("text_sets must identify looping and finished examples")
    seen = set()
    for examples in sets.values():
        if not isinstance(examples, list) or not examples:
            raise ValueError("Each text set requires nonempty ID/hash records")
        for example in examples:
            if not isinstance(example, dict):
                raise TypeError("Text identities require ID/hash records")
            identity, digest = example.get("id"), example.get("text_sha256")
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError("Text identities must be nonempty and distinct")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Text identities require lowercase SHA256 digests")
            seen.add(identity)
    return {group: len(examples) for group, examples in sets.items()}


def _array(value, name, ndim):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim != ndim or not array.size:
        raise ValueError(f"{name} must be a nonempty real {ndim}-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array.astype(np.float64, copy=False)


def contrastive_mean(looping, finished, *, counts=None):
    """Looping-minus-finished means, with equal weight per example.

    Supply matrices of already reduced per-example activations, or two sum
    vectors with explicit positive counts. Token sums are not example sums.
    """
    if counts is None:
        left, right = _array(looping, "looping", 2), _array(finished, "finished", 2)
        if left.shape[1:] != right.shape[1:]:
            raise ValueError("Contrastive activation widths differ")
        return left.mean(axis=0) - right.mean(axis=0), {"looping": len(left), "finished": len(right)}
    if (
        not isinstance(counts, Mapping)
        or set(counts) != {"looping", "finished"}
        or any(type(counts[key]) is not int or counts[key] <= 0 for key in counts)
    ):
        raise ValueError("Sum vectors require positive integer counts for both groups")
    left, right = _array(looping, "looping sums", 1), _array(finished, "finished sums", 1)
    if left.shape != right.shape:
        raise ValueError("Contrastive activation widths differ")
    return left / counts["looping"] - right / counts["finished"], dict(counts)


def repair_tensor(base, direction, *, alpha, column_rule, input_difference=None, top_k=None):
    """Return float32 replacement and provenance for one projection edit.

    Top-k ranks the signed product, not its magnitude. Equal scores use ascending
    column indices. Direction normalization and construction occur in float64;
    the submitted result is float32, then the exporter evaluates it in bf16.
    """
    weights = _array(base, "base", 2)
    unit = _array(direction, "direction", 1)
    if unit.shape != (weights.shape[0],):
        raise ValueError("Direction width does not match the matrix output")
    norm = float(np.linalg.norm(unit))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Contrastive direction must have finite nonzero norm")
    unit = unit / norm
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    projection = unit @ weights
    scores = None
    if column_rule == "all":
        if top_k is not None:
            raise ValueError("all-columns rule does not accept top_k")
        columns = np.arange(weights.shape[1])
    elif column_rule == "topk":
        if type(top_k) is not int or not 1 <= top_k <= weights.shape[1]:
            raise ValueError("top_k must fit the matrix input width")
        if input_difference is None:
            raise ValueError("topk requires measured contrastive input activations")
        difference = _array(input_difference, "input difference", 1)
        if difference.shape != (weights.shape[1],):
            raise ValueError("Input activation width does not match matrix columns")
        scores = projection * difference
        if not np.isfinite(scores).all():
            raise ValueError("Column scores overflowed")
        columns = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
    else:
        raise ValueError("column_rule must be all or topk")
    changed = weights.copy()
    changed[:, columns] -= float(alpha) * np.outer(unit, projection[columns])
    with np.errstate(over="ignore", invalid="ignore"):
        replacement = changed.astype(np.float32)
    if not np.isfinite(replacement).all():
        raise ValueError("Repair cannot be represented as finite float32")
    return replacement, {
        "direction": unit.tolist(),
        "direction_norm_before_normalization": norm,
        "column_rule": column_rule,
        "top_k": top_k,
        "columns": columns.tolist(),
        "selected_scores": scores[columns].tolist() if scores is not None else None,
        "alpha": float(alpha),
        "formula": "W0 - alpha * outer(u, mask * (u @ W0))",
        "score_rule": "signed descending (u @ W0) * (mean_input_looping - mean_input_finished)",
        "tie_break": "ascending column index",
    }


def build_repair(
    store,
    output_dir,
    *,
    activations=None,
    directions=None,
    input_differences=None,
    alpha,
    column_rule,
    top_k=None,
    provenance,
):
    """Export a repair from measured means or explicitly sourced saved directions.

    ``activations`` maps tensor names to output_looping/output_finished and,
    optionally, input_looping/input_finished arrays. Optional ``counts`` selects
    summed per-example vectors instead of per-example matrices. Alternatively,
    ``directions`` maps names to saved output directions, with separately supplied
    ``input_differences`` required for top-k. Existing store submissions are ignored
    and untouched. The output directory must not exist.
    """
    import torch

    from openwsp.budget import atomic_json
    from openwsp.core import WeightStore

    if (activations is None) == (directions is None):
        raise ValueError("Supply exactly one of activations or saved directions")
    if activations is not None and input_differences is not None:
        raise ValueError("input_differences accompany saved directions; activations contain their own inputs")
    if (
        not isinstance(provenance, dict)
        or not provenance.get("source")
        or not provenance.get("activation_reduction")
    ):
        raise ValueError("Provenance must identify source and activation_reduction")
    if not provenance.get("text_sets") and not provenance.get("saved_method_sha256"):
        raise ValueError("Provenance requires text-set identities or a saved-method SHA256")
    # Validate JSON before any artifact can be published.
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    provenance_counts = _provenance_counts(provenance)
    sources = activations if activations is not None else directions
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("At least one target tensor is required")
    if input_differences is not None and set(input_differences) != set(sources):
        raise ValueError("Input-difference tensor names must match saved directions")
    originals, replacements, methods = {}, {}, {}
    for name in sorted(sources):
        original = store.load_original(name)
        originals[name] = original
        if activations is not None:
            statistics = activations[name]
            direction, counts = contrastive_mean(
                statistics["output_looping"], statistics["output_finished"], counts=statistics.get("counts")
            )
            if provenance_counts is not None and provenance_counts != counts:
                raise ValueError("Text-set identities and activation group counts differ")
            difference = None
            if "input_looping" in statistics or "input_finished" in statistics:
                difference, input_counts = contrastive_mean(
                    statistics["input_looping"], statistics["input_finished"], counts=statistics.get("counts")
                )
                if input_counts != counts:
                    raise ValueError("Input and output activation groups have different counts")
        else:
            direction, counts = directions[name], None
            difference = input_differences[name] if input_differences is not None else None
        replacement, method = repair_tensor(
            original.numpy(),
            direction,
            alpha=alpha,
            column_rule=column_rule,
            input_difference=difference,
            top_k=top_k,
        )
        method["counts"] = counts
        method["source_array_identities"] = (
            {key: _array_identity(value) for key, value in statistics.items() if key != "counts"}
            if activations is not None
            else {"direction": _array_identity(direction)}
        )
        if activations is None and difference is not None:
            method["source_array_identities"]["input_difference"] = _array_identity(difference)
        methods[name] = method
        replacements[name] = torch.from_numpy(replacement)
    # Normal public refusals apply, including frozen embeddings/norm/head,
    # nonfinite bf16 and whole-tensor rescaling. No private validator bypass.
    validated = WeightStore(originals)
    for name, replacement in replacements.items():
        validated.save_edit(name, replacement)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("Repair output already exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".repair-", dir=destination.parent))
    try:
        manifest = validated.export_edit(
            temporary / "edit",
            base_model=provenance.get("base_model"),
            revision=provenance.get("base_revision"),
        )
        method = {
            "format": "openwsp.repair",
            "format_version": 1,
            "edit_sha256": manifest["edit_sha256"],
            "layers": methods,
            "provenance": provenance,
            "analysis_dtype": "float64",
            "submitted_dtype": "float32",
            "evaluated_dtype": "bfloat16",
        }
        atomic_json(temporary / "method.json", method)
        result = {
            "format": "openwsp.repair.summary",
            "format_version": 1,
            "build_status": "built",
            "evaluation_status": "pending",
            "accuracy": None,
            "accuracy_finished": None,
            "edit_sha256": manifest["edit_sha256"],
            "changed_tensors": manifest["changed_tensors"],
            "changed_entries_total": manifest["changed_entries_total"],
            "alpha": float(alpha),
            "column_rule": column_rule,
            "top_k": top_k,
            "method_sha256": _file_hash(temporary / "method.json"),
            "provenance_sha256": _json_hash(provenance),
            "edit_path": "edit",
        }
        atomic_json(temporary / "summary.json", result)
        # Publish only after all validation and serialization succeeds.
        temporary.rename(destination)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_activation_manifest(path):
    """Load an NPZ with an explicitly hashed, caller-verified text-set manifest.

    Manifest: format_version=1, arrays={path,sha256}, layers mapping tensor names
    to array-key mappings, provenance={source,activation_reduction,text_sets,...}.
    Each layer may also specify counts for summed per-example arrays.
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported activation manifest version")
    artifact = path.parent / manifest["arrays"]["path"]
    if _file_hash(artifact) != manifest["arrays"]["sha256"]:
        raise ValueError("Activation archive SHA256 mismatch")
    arrays = {}
    with np.load(artifact, allow_pickle=False) as archive:
        for name, fields in manifest["layers"].items():
            if set(fields) - {
                "output_looping",
                "output_finished",
                "input_looping",
                "input_finished",
                "counts",
            }:
                raise ValueError("Unknown activation field")
            arrays[name] = {
                key: value if key == "counts" else archive[value].copy() for key, value in fields.items()
            }
    provenance = dict(manifest["provenance"])
    provenance.update(
        activation_manifest_sha256=_file_hash(path), activation_archive_sha256=_file_hash(artifact)
    )
    return arrays, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--activation-manifest")
    source.add_argument("--saved-method", help="Existing mechanical control method with saved directions")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--alpha", type=float, help="Explicit dose override; no automatic search")
    parser.add_argument("--base-model", help="Optional identity label; tensor hashes enforce compatibility")
    parser.add_argument("--revision", help="Optional pinned revision label")
    args = parser.parse_args(argv)
    from openwsp.core import WeightStore

    store = WeightStore(args.model)
    settings = {**VARIANTS[args.variant]}
    if args.alpha is not None:
        settings["alpha"] = args.alpha
    if args.activation_manifest:
        activations, provenance = load_activation_manifest(args.activation_manifest)
        source_values = {"activations": activations}
    else:
        method = json.loads(Path(args.saved_method).read_text())
        source_values = {
            "directions": {
                f"model.language_model.layers.{layer}.mlp.down_proj.weight": data["direction"]
                for layer, data in method["layers"].items()
            }
        }
        provenance = {
            "source": str(args.saved_method),
            "saved_method_sha256": _file_hash(args.saved_method),
            "activation_reduction": method["protocol"],
            "counts": method["counts"],
            "benchmark_reuse": method.get("benchmark_reuse"),
            "limitation": "Saved directions only; input contrast unavailable. Published columns are ignored.",
        }
    if args.base_model:
        provenance["base_model"] = args.base_model
    if args.revision:
        provenance["base_revision"] = args.revision
    result = build_repair(store, args.output, provenance=provenance, **source_values, **settings)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
