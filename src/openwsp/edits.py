"""Versioned, content-addressed diffs compatible with the authors' edit files.

The evaluated contract uses bfloat16 bits, even when submitted tensors are float32.
The payload hash is over logical arrays, not the NPZ ZIP container, so timestamps
and compression do not affect an edit's identity. No pickle is ever loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT = "openwsp.edit"
FORMAT_VERSION = 1


class EditError(ValueError):
    """An edit is invalid, corrupted, or incompatible with its base."""


def _bits(tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        raise EditError("Weights must be torch tensors")
    value = tensor.detach().cpu().contiguous()
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise EditError("Weights must contain finite floating point values")
    rounded = value.to(torch.bfloat16)
    if not bool(torch.isfinite(rounded).all()):
        raise EditError("Weights overflow bfloat16")
    return rounded.view(torch.uint16).numpy().astype("<u2", copy=True)


def _sha_bits(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits, dtype="<u2").tobytes()).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA256 of a tensor's contiguous, little-endian bfloat16 bytes."""
    return _sha_bits(_bits(tensor))


def _float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    # Match st_numpy.bf16_bits: round float32 to nearest, with ties to even.
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u += 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype("<u2")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        # Explicit length framing prevents concatenation ambiguity.
        header = _canonical({"name": name, "dtype": array.dtype.str, "shape": list(array.shape)})
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.nbytes.to_bytes(8, "little"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def edit_sha256(manifest: Mapping[str, Any]) -> str:
    """Compute the v1 address; the payload digest is part of this manifest."""
    return hashlib.sha256(_canonical({k: v for k, v in manifest.items() if k != "edit_sha256"})).hexdigest()


def export_edit(
    originals: Mapping[str, torch.Tensor],
    edited: Mapping[str, torch.Tensor],
    path: str | Path,
    *,
    base_model: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Write a sparse v1 diff against the bf16-rounded base.

    The input mappings are never modified. Tensor names and shapes are unchanged;
    hash identities refer to evaluated values, not float32 submission precision.
    """
    arrays: dict[str, np.ndarray] = {}
    tensors: list[dict[str, Any]] = []
    if any(not isinstance(name, str) or not name for name in edited):
        raise EditError("Tensor names must be nonempty strings")
    for name in sorted(edited):
        if name not in originals:
            raise EditError(f"Unknown base tensor: {name}")
        base, changed = _bits(originals[name]), _bits(edited[name])
        if base.shape != changed.shape:
            raise EditError(f"Shape mismatch: {name}")
        indices = np.flatnonzero(base.reshape(-1) != changed.reshape(-1))
        if not len(indices):
            continue
        if base.size > np.iinfo(np.int32).max:
            raise EditError(f"Tensor is too large for legacy-compatible int32 indices: {name}")
        i = len(tensors)
        arrays[f"t{i}_idx"] = indices.astype("<i4")
        arrays[f"t{i}_val"] = changed.reshape(-1)[indices].astype("<u2")
        b, e = _float32(base), _float32(changed)
        norm = float(np.linalg.norm(b.astype(np.float64)))
        delta = float(np.linalg.norm(e.astype(np.float64) - b.astype(np.float64)))
        entry: dict[str, Any] = {
            "name": name,
            "shape": list(base.shape),
            "dtype": "bfloat16",
            "form": "sparse",
            "changed_entries": len(indices),
            "base_sha256": _sha_bits(base),
            "edited_sha256": _sha_bits(changed),
            "submitted_file": name + ".npy",
            "relative_change": delta / norm if norm else None,
        }
        if base.ndim == 2:
            rows, columns = np.unravel_index(indices, base.shape)
            entry.update(changed_rows=len(np.unique(rows)), changed_columns=len(np.unique(columns)))
        tensors.append(entry)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "tensors": tensors,
        "changed_tensors": len(tensors),
        "changed_entries_total": sum(t["changed_entries"] for t in tensors),
        "payload_sha256": _payload_sha256(arrays),
    }
    if base_model is not None:
        manifest["base_model"] = base_model
    if revision is not None:
        manifest["base_revision"] = revision
    manifest["edit_sha256"] = edit_sha256(manifest)
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    # Publish the manifest last. Readers either validate the previous pair or
    # reject an interrupted replacement; an incomplete file is never accepted.
    with tempfile.NamedTemporaryFile(dir=target, suffix=".npz", delete=False) as f:
        payload_temp = Path(f.name)
        np.savez_compressed(f, **arrays)
    with tempfile.NamedTemporaryFile(mode="w", dir=target, suffix=".json", delete=False) as f:
        manifest_temp = Path(f.name)
        json.dump(manifest, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    try:
        os.replace(payload_temp, target / "edit.npz")
        os.replace(manifest_temp, target / "manifest.json")
    finally:
        payload_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    return manifest


def _checked_fixups(arrays: Mapping[str, np.ndarray], i: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    indices, values = arrays[f"t{i}_idx"], arrays[f"t{i}_val"]
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise EditError("Sparse indices must be a one-dimensional integer array")
    if values.ndim != 1 or values.dtype.kind != "u" or values.dtype.itemsize != 2:
        raise EditError("Sparse values must be a one-dimensional uint16 array")
    if len(indices) != len(values):
        raise EditError("Sparse indices and values have different lengths")
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= size):
        raise EditError("Sparse index is outside its tensor")
    if len(np.unique(indices)) != len(indices):
        raise EditError("Duplicate sparse indices are not allowed")
    return indices.astype(np.int64, copy=False), values


def read_edit(
    path: str | Path, originals: Mapping[str, torch.Tensor]
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Validate all data, then return reconstructed float32 tensors.

    Unversioned authors' manifests are accepted as legacy. Their lowrank
    reconstruction is reproduced rank-by-rank in float64 before float32/bf16
    rounding and sparse fixups. No input tensor is mutated on success or failure.
    """
    target = Path(path)
    try:
        manifest = json.loads((target / "manifest.json").read_text())
        if not isinstance(manifest, dict):
            raise EditError("Manifest must be an object")
        versioned = "format" in manifest or "format_version" in manifest
        if versioned and (
            manifest.get("format") != FORMAT
            or type(manifest.get("format_version")) is not int
            or manifest["format_version"] != FORMAT_VERSION
        ):
            raise EditError("Unsupported edit format or version")
        if not isinstance(manifest.get("tensors"), list):
            raise EditError("Manifest tensors must be a list")
        with np.load(target / "edit.npz", allow_pickle=False) as archive:
            if len(set(archive.files)) != len(archive.files):
                raise EditError("Duplicate payload array names")
            arrays = {name: archive[name] for name in archive.files}
        if versioned:
            if manifest.get("payload_sha256") != _payload_sha256(arrays):
                raise EditError("Edit payload sha256 mismatch")
            if manifest.get("edit_sha256") != edit_sha256(manifest):
                raise EditError("Edit address sha256 mismatch")
        rebuilt: dict[str, torch.Tensor] = {}
        expected_arrays: set[str] = set()
        changed_total = 0
        for i, tensor in enumerate(manifest["tensors"]):
            if not isinstance(tensor, dict):
                raise EditError("Tensor entries must be objects")
            name = tensor["name"]
            if not isinstance(name, str) or name not in originals or name in rebuilt:
                raise EditError(f"Unknown or duplicate tensor: {name}")
            base = _bits(originals[name])
            if tensor.get("dtype") != "bfloat16" or tensor.get("shape") != list(base.shape):
                raise EditError(f"Tensor shape or dtype mismatch: {name}")
            if tensor.get("base_sha256") != _sha_bits(base):
                raise EditError(f"Base tensor sha256 mismatch: {name}")
            expected_arrays.update((f"t{i}_idx", f"t{i}_val"))
            indices, values = _checked_fixups(arrays, i, base.size)
            if tensor.get("form") == "lowrank":
                a, b = arrays[f"t{i}_a"], arrays[f"t{i}_b"]
                expected_arrays.update((f"t{i}_a", f"t{i}_b"))
                if (
                    base.ndim != 2
                    or a.ndim != 2
                    or b.ndim != 2
                    or a.shape[0] != base.shape[0]
                    or b.shape[1] != base.shape[1]
                    or a.shape[1] != b.shape[0]
                    or a.dtype.kind != "f"
                    or b.dtype.kind != "f"
                    or not np.isfinite(a).all()
                    or not np.isfinite(b).all()
                ):
                    raise EditError(f"Invalid lowrank factors: {name}")
                weights = _float32(base).astype(np.float64)
                # Intentionally mirror upstream apply_edit.py, not a BLAS matmul.
                for rank in range(a.shape[1]):
                    weights = weights + np.outer(a[:, rank], b[rank])
                edited_bits = _bf16_bits(weights.astype(np.float32)).reshape(-1)
            elif tensor.get("form") == "sparse":
                edited_bits = base.copy().reshape(-1)
            else:
                raise EditError(f"Unsupported tensor edit form: {tensor.get('form')}")
            edited_bits[indices] = values
            edited_bits = edited_bits.reshape(base.shape)
            if tensor.get("edited_sha256") != _sha_bits(edited_bits):
                raise EditError(f"Edited tensor sha256 mismatch: {name}")
            as_float32 = _float32(edited_bits)
            if not np.isfinite(as_float32).all():
                raise EditError(f"Nonfinite evaluated tensor: {name}")
            changed = int(np.count_nonzero(base != edited_bits))
            if tensor.get("changed_entries") != changed:
                raise EditError(f"Changed entry count mismatch: {name}")
            changed_total += changed
            rebuilt[name] = torch.from_numpy(as_float32.copy())
        if versioned and set(arrays) != expected_arrays:
            raise EditError("Unexpected payload arrays")
        if manifest.get("changed_tensors") != len(rebuilt):
            raise EditError("Changed tensor count mismatch")
        if manifest.get("changed_entries_total") != changed_total:
            raise EditError("Total changed entry count mismatch")
        return manifest, rebuilt
    except EditError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        OverflowError,
        EOFError,
        zipfile.BadZipFile,
    ) as exc:
        raise EditError(f"Invalid edit: {type(exc).__name__}: {exc}") from exc
