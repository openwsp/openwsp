"""Finished-answer accuracy must use the matching sample subset."""

import pytest

from openwsp.metrics import finished_metrics


def test_finished_metric_excludes_even_correct_token_limited_answers():
    assert finished_metrics(iter([
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": False},
        {"finish_reason": "length", "correct": True},
        {"finish_reason": "length", "correct": False},
    ])) == {"finished_n": 2, "finished_correct": 1, "accuracy_finished": 0.5}


@pytest.mark.parametrize("rows", [[], [{"finish_reason": "length", "correct": True}]])
def test_no_finished_answers_has_undefined_accuracy(rows):
    assert finished_metrics(rows) == {"finished_n": 0, "finished_correct": 0, "accuracy_finished": None}


@pytest.mark.parametrize("reason", [None, "", "error", "timeout", "max_tokens"])
def test_unrecognized_finish_reason_is_not_assumed_finished(reason):
    with pytest.raises(ValueError, match="finish_reason"):
        finished_metrics([{"finish_reason": reason, "correct": True}])


@pytest.mark.parametrize("grade", [None, "true", 2, -1, 1.0])
def test_invalid_grades_are_not_coerced(grade):
    with pytest.raises(ValueError, match="sample grades"):
        finished_metrics([{"finish_reason": "stop", "correct": grade}])
