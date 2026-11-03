from pathlib import Path

import pytest

from replications.exp8.forward import grader
from replications.exp8.runtime import evaluate_outputs, grade_ifeval


def test_authors_real_extraction_and_math_verify(tmp_path):
    pytest.importorskip("math_verify")
    module = grader()
    answer, _ = module.extract(r"<think>guess \boxed{3}</think> The answer is \boxed{1/2}.")
    assert answer == "1/2"
    assert module.is_correct(answer, "0.5")
    assert not module.is_correct("3", "0.5")
    result = evaluate_outputs(
        [{"id": "a", "samples": [{"text": r"\boxed{1/2}", "token_count": 7, "finish_reason": "stop"}]}],
        [{"id": "a", "answer": "0.5"}], directory=tmp_path, stage="test", benchmark="math500", samples=1,
        seconds=1, cost_usd=0, provenance={})
    assert result["correct"] == 1
    assert result["finished_n"] == result["finished_correct"] == 1
    assert result["accuracy_finished"] == 1
    assert Path(tmp_path, "samples.jsonl").exists()
    with pytest.raises(NotImplementedError):
        module.mcqa_loop_signals("unused")


def test_finished_grading_keeps_correct_cutoff_in_overall_accuracy(tmp_path):
    pytest.importorskip("math_verify")
    result = evaluate_outputs(
        [{"id": "a", "samples": [
            {"text": r"\boxed{1}", "token_count": 7, "finish_reason": "length"},
            {"text": r"\boxed{2}", "token_count": 3, "finish_reason": "stop"},
        ]}], [{"id": "a", "answer": "1"}], directory=tmp_path, stage="test", benchmark="math500", samples=2,
        seconds=1, cost_usd=0, provenance={})
    assert result["accuracy"] == result["cut_share"] == 0.5
    assert result["finished_n"] == 1 and result["finished_correct"] == 0
    assert result["accuracy_finished"] == 0


def test_all_cutoff_grading_keeps_finished_accuracy_undefined(tmp_path):
    pytest.importorskip("math_verify")
    result = evaluate_outputs(
        [{"id": "a", "samples": [{"text": r"\boxed{1}", "token_count": 7, "finish_reason": "length"}]}],
        [{"id": "a", "answer": "1"}], directory=tmp_path, stage="test", benchmark="math500", samples=1,
        seconds=1, cost_usd=0, provenance={})
    assert result["accuracy"] == 1
    assert result["finished_n"] == result["finished_correct"] == 0
    assert result["accuracy_finished"] is None


def test_ifeval_strict_response_after_thinking():
    pytest.importorskip("langdetect")
    row = {"prompt": "Write in uppercase", "instruction_id_list": ["change_case:english_capital"], "kwargs": [{}]}
    english = "THE WEATHER IS BEAUTIFUL TODAY AND I AM GOING TO WALK THROUGH THE PARK WITH MY FRIENDS."
    assert grade_ifeval(row, "reasoning</think>" + english)[0]
    assert not grade_ifeval(row, "reasoning</think>" + english.lower())[0]
    assert not grade_ifeval(row, english, thinking=True)[0]
