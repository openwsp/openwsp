"""Metrics over compact, graded sample records; no model or provider access."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def finished_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, int | float | None]:
    """Count non-truncated answers using canonical ``stop``/``length`` reasons.

    An absent or unknown termination reason cannot establish that an answer
    finished. In particular, do not derive this metric from all-sample accuracy:
    a token-limited generation can still contain a correctly graded answer.
    """
    finished_n = finished_correct = 0
    for row in records:
        reason = row.get("finish_reason")
        if reason not in ("stop", "length"):
            raise ValueError("Finished metrics require canonical stop/length finish_reason values")
        correct = row.get("correct")
        if type(correct) not in (bool, int) or correct not in (0, 1):
            raise ValueError("Finished metrics require boolean or zero/one sample grades")
        if reason == "stop":
            finished_n += 1
            finished_correct += int(correct)
    return {"finished_n": finished_n, "finished_correct": finished_correct,
            "accuracy_finished": finished_correct / finished_n if finished_n else None}
