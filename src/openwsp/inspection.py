"""Read-only edit geometry, computed with NumPy on CPU.

SVD describes the supplied numeric values, including any checkpoint rounding.
Projection coefficients use Δ[:,j] ≈ -alpha[j] u (u.T @ W0[:,j]); neither a
dominant singular vector nor a small residual establishes a causal mechanism.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return array


def inspect_tensor(name: str, base, edited, *, top_directions=3, rank_rtol=None, projection_rtol=0.01):
    """Describe a tensor delta; matrices additionally receive exact SVD diagnostics."""
    if type(top_directions) is not int or not 1 <= top_directions <= 16:
        raise ValueError("top_directions must be an integer from 1 to 16")
    for value in (rank_rtol, projection_rtol):
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value < 0):
            raise ValueError("Relative tolerances must be finite and nonnegative")
    original, replacement = _array(base, "base"), _array(edited, "edited")
    if original.shape != replacement.shape:
        raise ValueError("Base and edited tensor shapes differ")
    original = original.astype(np.float64, copy=False)
    delta = replacement.astype(np.float64, copy=False) - original
    changed = delta != 0
    base_norm, delta_norm = float(np.linalg.norm(original.ravel())), float(np.linalg.norm(delta.ravel()))
    result: dict[str, Any] = {
        "name": name,
        "shape": list(original.shape),
        "changed_entries": int(np.count_nonzero(changed)),
        "base_frobenius": base_norm,
        "delta_frobenius": delta_norm,
        "relative_frobenius_change": delta_norm / base_norm if base_norm else None,
        "changed_rows": None,
        "changed_columns": None,
        "row_indices": None,
        "column_indices": None,
        "svd": None,
        "projection": None,
    }
    if original.ndim != 2:
        result["matrix_analysis_unavailable"] = "Tensor is not a matrix"
        return result
    rows, columns = np.flatnonzero(changed.any(axis=1)), np.flatnonzero(changed.any(axis=0))
    result.update(
        changed_rows=len(rows),
        changed_columns=len(columns),
        row_indices=rows.tolist(),
        column_indices=columns.tolist(),
    )
    relative_tolerance = float(
        rank_rtol if rank_rtol is not None else max(original.shape, default=0) * np.finfo(np.float64).eps
    )
    if not len(rows) or not len(columns):
        result["svd"] = {
            "rank": 0,
            "tolerance": 0.0,
            "relative_tolerance": relative_tolerance,
            "singular_values": [],
            "rank_one_relative_error": 0.0,
            "energy_rank_99pct": 0,
            "directions": [],
        }
        return result
    # Unchanged rows/columns contribute only zero singular values. Cropping keeps
    # sparse column edits (e.g. 1024 x 64) cheap without approximating their SVD.
    cropped = delta[np.ix_(rows, columns)]
    left, singular, _ = np.linalg.svd(cropped, full_matrices=False)
    tolerance = relative_tolerance * float(singular[0])
    rank = int(np.count_nonzero(singular > tolerance))
    energy = np.square(singular)
    total_energy = float(energy.sum())
    directions = []
    for index in range(min(top_directions, rank)):
        direction = np.zeros(original.shape[0], dtype=np.float64)
        direction[rows] = left[:, index]
        if direction[int(np.argmax(np.abs(direction)))] < 0:
            direction *= -1
        directions.append(
            {
                "index": index,
                "singular_value": float(singular[index]),
                "energy_fraction": float(energy[index] / total_energy),
                "vector": direction.tolist(),
            }
        )
    result["svd"] = {
        "rank": rank,
        "tolerance": tolerance,
        "relative_tolerance": relative_tolerance,
        "singular_values": singular.tolist(),
        "rank_one_relative_error": float(np.sqrt(energy[1:].sum() / total_energy)),
        "energy_rank_99pct": int(np.searchsorted(np.cumsum(energy) / total_energy, 0.99) + 1),
        "directions": directions,
        "sign_convention": "Largest-magnitude coordinate is positive; tied singular subspaces remain nonunique",
        "omitted_zero_singular_values": min(original.shape) - len(singular),
    }
    if not directions:
        return result
    direction = np.asarray(directions[0]["vector"])
    base_projection = direction @ original[:, columns]
    delta_projection = direction @ delta[:, columns]
    denominator_tolerance = (
        np.finfo(np.float64).eps * max(original.shape) * np.linalg.norm(original[:, columns], axis=0)
    )
    identifiable = np.abs(base_projection) > denominator_tolerance
    alpha = np.divide(
        -delta_projection, base_projection, out=np.zeros_like(delta_projection), where=identifiable
    )
    projected = -np.outer(direction, alpha * base_projection)
    relative_error = float(np.linalg.norm(delta[:, columns] - projected) / delta_norm)
    coefficients = [
        {
            "column": int(column),
            "alpha": float(alpha[index]) if identifiable[index] else None,
            "base_projection": float(base_projection[index]),
            "delta_projection": float(delta_projection[index]),
        }
        for index, column in enumerate(columns)
    ]
    known = alpha[identifiable]
    result["projection"] = {
        "formula": "delta[:,j] = -alpha[j] * u * (u.T @ base[:,j])",
        "direction_index": 0,
        "fit_rtol": float(projection_rtol),
        "relative_error": relative_error,
        "consistent": bool(relative_error <= projection_rtol and identifiable.all()),
        "coefficients": coefficients,
        "unchanged_columns_alpha": 0.0,
        "coefficient_summary": {
            "identifiable_columns": len(known),
            "unidentifiable_columns": int((~identifiable).sum()),
            "min": float(known.min()) if len(known) else None,
            "max": float(known.max()) if len(known) else None,
            "median": float(np.median(known)) if len(known) else None,
            "std": float(known.std()) if len(known) else None,
        },
    }
    return result


def _identity(direction):
    return {key: direction[key] for key in ("edit", "tensor", "index")}


def directions_from_edits(edits: Sequence[Mapping[str, Any]]):
    return [
        {"edit": edit["label"], "tensor": tensor["name"], **direction}
        for edit in edits
        for tensor in edit["tensors"]
        for direction in (tensor.get("svd") or {}).get("directions", [])
    ]


def direction_cosines(directions: Sequence[Mapping[str, Any]]):
    """Compare equally sized left singular directions; signs are not intrinsic."""
    comparisons = []
    for index, left in enumerate(directions):
        a = _array(left["vector"], "direction").astype(np.float64, copy=False)
        for right in directions[index + 1 :]:
            b = _array(right["vector"], "direction").astype(np.float64, copy=False)
            if a.shape != b.shape:
                continue
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosine = float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator else None
            comparisons.append(
                {
                    "left": _identity(left),
                    "right": _identity(right),
                    "cosine": cosine,
                    "absolute_cosine": abs(cosine) if cosine is not None else None,
                }
            )
    return comparisons


def logit_lens(directions: Sequence[Mapping[str, Any]], embedding, token_ids: Mapping[str, int]):
    """Cosines against supplied unembedding rows and their vocabulary mean.

    The caller verifies tying and token IDs. This is a weight-direction cosine,
    not a model forward or a contextual prediction of the named words.
    """
    weights = _array(embedding, "embedding")
    if weights.ndim != 2 or not weights.shape[0]:
        raise ValueError("embedding must be a nonempty vocabulary-by-hidden matrix")
    if any(
        not isinstance(label, str) or type(token) is not int or not 0 <= token < weights.shape[0]
        for label, token in token_ids.items()
    ):
        raise ValueError("Token labels must map to valid integer vocabulary rows")
    mean = np.mean(weights, axis=0, dtype=np.float64)
    mean_norm = float(np.linalg.norm(mean))
    selected = {label: np.asarray(weights[token], dtype=np.float64) for label, token in token_ids.items()}

    def cosine(a, b):
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator else None

    output = []
    for direction in directions:
        vector = _array(direction["vector"], "direction").astype(np.float64, copy=False)
        if vector.shape != (weights.shape[1],):
            output.append({**_identity(direction), "available": False, "reason": "Hidden dimensions differ"})
            continue
        named = {label: cosine(vector, row) for label, row in selected.items()}
        strongest = max(
            (label for label in named if named[label] is not None),
            key=lambda label: abs(named[label]),
            default=None,
        )
        output.append(
            {
                **_identity(direction),
                "available": True,
                "token_cosines": named,
                "mean_row_cosine": cosine(vector, mean),
                "largest_named_absolute_cosine": abs(named[strongest]) if strongest else None,
                "largest_named_token": strongest,
            }
        )
    return {
        "token_ids": dict(token_ids),
        "vocabulary_rows": weights.shape[0],
        "hidden_size": weights.shape[1],
        "mean_row_norm": mean_norm,
        "directions": output,
        "interpretation": "Signed cosines use canonical SVD orientation; absolute values ignore arbitrary direction sign. Named rows do not search the whole vocabulary.",
    }
