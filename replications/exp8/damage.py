"""CPU reduction of KL(base || edited) over already aligned logit positions.

No model is loaded and no inference is launched. ``valid_mask`` selects logit
positions, not target-token positions: a causal model backend must perform its
next-token shift once before calling this reducer.

``python -m replications.exp8.damage validate --output summary.json`` publishes
only a synthetic correctness check. ``reduce --manifest manifest.json --output
summary.json`` reads hash-verified .npy chunks, with pickle disabled. A manifest
has schema_version=1, synthetic (boolean), examples [{id, text_sha256, source}],
and chunks [{base_logits: {path, sha256}, edited_logits: {path, sha256},
valid_mask: {path, sha256}, reference_ids: [...]}]. Paths are relative to the
manifest directory. Production records additionally identify base_checkpoint,
edited_checkpoint (each with sha256), edit_sha256 and the preregistered scope.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np

_SHA256 = re.compile(r"[0-9a-f]{64}")
MASK_CONVENTION = "valid_mask selects logit positions after the causal next-token shift"
PRODUCTION_SCOPE = {
    "selection": "source_order_256_gsm8k_then_256_math",
    "thinking": True,
    "scoring": "reference_continuation_only",
    "context_limit": 34816,
    "overflow": "reject",
}


def _logits(value, name):
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError(f"{name} must have a nonempty final vocabulary axis")
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric logits")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} logits must be finite")
    return array


def kl_from_logits(base, edited, valid_mask=None):
    """Return token count, KL sum and token-weighted mean in nats.

    Inputs have identical shape (..., vocabulary). A boolean mask matches the
    preceding token axes exactly. Empty selections have an undefined mean.
    All input logits, including padded positions, must be finite.
    """
    base = _logits(base, "Base")
    edited = _logits(edited, "Edited")
    if base.shape != edited.shape:
        raise ValueError("Base and edited logit shapes must match")
    if valid_mask is None:
        valid_mask = np.ones(base.shape[:-1], dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.dtype != np.dtype(bool) or valid_mask.shape != base.shape[:-1]:
            raise ValueError("Valid mask must be boolean and match the token axes")
    token_count = int(valid_mask.sum())
    if not token_count:
        return {"token_count": 0, "kl_sum": 0.0, "kl_mean": None}
    base, edited = base[valid_mask], edited[valid_mask]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            base_shifted = base - base.max(axis=-1, keepdims=True)
            edited_shifted = edited - edited.max(axis=-1, keepdims=True)
            base_logp = base_shifted - np.log(np.exp(base_shifted).sum(axis=-1, keepdims=True))
            edited_logp = edited_shifted - np.log(np.exp(edited_shifted).sum(axis=-1, keepdims=True))
            per_token = (np.exp(base_logp) * (base_logp - edited_logp)).sum(axis=-1)
    except FloatingPointError as exc:
        raise ValueError("KL cannot be represented by the float64 reduction") from exc
    if not np.isfinite(per_token).all() or np.any(per_token < -1e-12):
        raise ValueError("KL reduction produced an invalid divergence")
    # Roundoff can produce a tiny negative divergence between equal distributions.
    try:
        kl_sum = math.fsum(np.maximum(per_token, 0.0).ravel().tolist())
    except OverflowError as exc:
        raise ValueError("KL sum exceeds float64 range") from exc
    if not math.isfinite(kl_sum):
        raise ValueError("KL sum cannot be represented as a finite float")
    return {"token_count": token_count, "kl_sum": kl_sum, "kl_mean": kl_sum / token_count}


def reduce_kl_chunks(chunks):
    """Reduce (base_logits, edited_logits, valid_mask) chunks by token count."""
    count, sums = 0, []
    for base, edited, mask in chunks:
        result = kl_from_logits(base, edited, mask)
        count += result["token_count"]
        sums.append(result["kl_sum"])
    try:
        total = math.fsum(sums)
    except OverflowError as exc:
        raise ValueError("Aggregate KL sum exceeds float64 range") from exc
    return {"token_count": count, "kl_sum": total, "kl_mean": total / count if count else None}


def _digest(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} requires a lowercase SHA-256 digest")
    return value


def _array_from_record(record, directory):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("Array records require exactly path and sha256")
    name = record["path"]
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Array path must be a relative .npy file")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
        raise ValueError("Array path must stay within the manifest directory")
    path = directory / relative
    if not path.resolve().is_relative_to(directory):
        raise ValueError("Array path escapes the manifest directory")
    expected = _digest(record["sha256"], "Array")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Array SHA-256 does not match the manifest")
    # Hash and decode the same bytes so a changed local file cannot bypass verification.
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise TypeError("Logit and mask artifacts must be single NumPy arrays")
    return array


def _manifest_examples(manifest):
    synthetic = manifest.get("synthetic")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or type(synthetic) is not bool:
        raise ValueError("Manifest requires schema_version 1 and explicit synthetic boolean")
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Manifest must identify every reference example")
    ids = []
    for row in examples:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("Reference examples require nonempty string IDs")
        ids.append(row["id"])
        _digest(row.get("text_sha256"), "Reference text")
    if len(set(ids)) != len(ids):
        raise ValueError("Reference example IDs must be unique")
    if not synthetic:
        sources = [row.get("source") for row in examples]
        if sources != ["gsm8k"] * 256 + ["math"] * 256:
            raise ValueError("Production KL requires 512 references: 256 GSM8K then 256 MATH")
        if manifest.get("scope") != PRODUCTION_SCOPE:
            raise ValueError("Production KL must declare the preregistered continuation-only scope")
        for key in ("base_checkpoint", "edited_checkpoint"):
            identity = manifest.get(key)
            if not isinstance(identity, dict):
                raise TypeError("Production KL requires both checkpoint identities")
            _digest(identity.get("sha256"), key)
        _digest(manifest.get("edit_sha256"), "Edit")
    return ids


def reduce_manifest(path):
    """Validate a private logits manifest and return compact provenance and KL."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise TypeError("KL manifest must be an object")
    ids = _manifest_examples(manifest)
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("KL manifest requires at least one paired chunk")
    covered, scored = set(), set()

    def arrays():
        for chunk in chunks:
            if not isinstance(chunk, dict) or set(chunk) != {"base_logits", "edited_logits", "valid_mask", "reference_ids"}:
                raise ValueError("Chunks require paired logits, a mask and reference IDs")
            base = _array_from_record(chunk["base_logits"], path.parent)
            edited = _array_from_record(chunk["edited_logits"], path.parent)
            mask = _array_from_record(chunk["valid_mask"], path.parent)
            references = chunk["reference_ids"]
            sequences = base.shape[0] if base.ndim == 3 else 1
            if base.ndim not in (2, 3) or not isinstance(references, list) or len(references) != sequences:
                raise ValueError("Manifest chunks must map each 2D logit sequence to one reference ID")
            if any(not isinstance(item, str) or item not in ids for item in references):
                raise ValueError("Chunk references must be declared example IDs")
            # Validate shapes/mask before inspecting scored-token coverage.
            if mask.dtype != np.dtype(bool) or mask.shape != base.shape[:-1]:
                raise ValueError("Chunk mask must be boolean and match logit token axes")
            covered.update(references)
            nonempty = mask.any(axis=-1) if base.ndim == 3 else [mask.any()]
            scored.update(item for item, present in zip(references, nonempty, strict=True) if present)
            yield base, edited, mask

    metrics = reduce_kl_chunks(arrays())
    if covered != set(ids) or scored != set(ids):
        raise ValueError("Every declared reference must have supplied logits and valid continuation tokens")
    synthetic = manifest["synthetic"]
    provenance = {
        "synthetic": synthetic, "scope": "Synthetic supplied-logit reduction; not a model benchmark" if synthetic else
        "KL from supplied logits for the declared practice references; no model inference launched by this command",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "mask_convention": MASK_CONVENTION,
        "reference_count": len(ids), "reference_identities": manifest["examples"],
        "array_hashes": [{key: chunk[key]["sha256"] for key in ("base_logits", "edited_logits", "valid_mask")} for chunk in chunks],
        "identity_verification": "Logit and mask bytes verified; checkpoint and reference hashes are declared provenance",
    }
    if not synthetic:
        provenance.update({key: manifest[key] for key in ("base_checkpoint", "edited_checkpoint", "edit_sha256")})
        provenance["evaluation_scope"] = manifest["scope"]
    return {"schema_version": 1, "stage": "damage", "status": "measured_from_supplied_logits",
            "metric": "kl_base_to_edited", "units": "nats", **metrics, "provenance": provenance}


def validate_cpu():
    """Evaluate analytic distributions, padding exclusion and unequal chunks."""
    p = np.array([[0.75, 0.25], [0.2, 0.8], [0.5, 0.5]])
    q = np.array([[0.5, 0.5], [0.6, 0.4], [0.5, 0.5]])
    base, edited = np.log(p), np.log(q)
    expected = float((p[:2] * np.log(p[:2] / q[:2])).sum())
    measured = kl_from_logits(base, edited, np.array([True, True, False]))
    split = reduce_kl_chunks([(base[:1], edited[:1], None), (base[1:2], edited[1:2], None)])
    empty = kl_from_logits(base, edited, np.zeros(3, dtype=bool))
    checks = {
        "identical_logits_zero": kl_from_logits(base, base)["kl_sum"] == 0,
        "analytic_two_tokens": math.isclose(measured["kl_sum"], expected, abs_tol=1e-14),
        "padding_excluded": measured["token_count"] == 2,
        "chunk_equivalence": split == measured,
        "empty_mask_undefined_mean": empty == {"token_count": 0, "kl_sum": 0.0, "kl_mean": None},
        "large_offset_stable": math.isclose(kl_from_logits(base + 10000, edited + 10000)["kl_sum"], expected, abs_tol=1e-11),
    }
    if not all(checks.values()):
        raise ValueError("CPU KL validation failed")
    return {"schema_version": 1, "stage": "damage_cpu_validation", "status": "validated", "checks": checks,
            "metric": "kl_base_to_edited", "units": "nats", "fixture": measured,
            "model_measurement": {"status": "pending", "kl_mean": None},
            "provenance": {"synthetic": True, "scope": "Analytic CPU array validation only; no model benchmark, training or GPU calls",
                           "mask_convention": MASK_CONVENTION,
                           "command": "python -m replications.exp8.damage validate --output OUTPUT.json"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reduce_parser = sub.add_parser("reduce", help="Reduce a private hash-verified logits manifest")
    reduce_parser.add_argument("--manifest", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    validate_parser = sub.add_parser("validate", help="Run a tiny synthetic CPU correctness check")
    validate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = reduce_manifest(args.manifest) if args.command == "reduce" else validate_cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    main()
