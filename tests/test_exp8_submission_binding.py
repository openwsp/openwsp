import json

import pytest

from replications.exp8 import evaluation


@pytest.fixture
def bound_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "Path", lambda value: tmp_path / str(value).lstrip("/"))
    monkeypatch.setattr(evaluation, "CONFIG", {"samples": {"math500": 4}})
    calls = []

    def evaluate(stage, checkpoint_dir, benchmark, samples, ledger, checkpoint):
        calls.append((stage, checkpoint_dir, benchmark, samples))
        return {"stage": stage, "benchmark": benchmark, "provenance": {}}

    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    return tmp_path, calls


def test_separate_attempt_evaluation_is_bound_before_generation(bound_evaluation):
    root, calls = bound_evaluation
    submission = {"checkpoint": "/data/submissions/example", "sha256": "a" * 64}
    result = evaluation.evaluate_live(submission, object(), lambda: None, stage="live-runs/example")
    directory = root / "data/results/live-runs/example/math500"
    binding = json.loads((directory / "submission.json").read_text())
    assert binding["submission_sha256"] == submission["sha256"]
    assert calls == [("live-runs/example", submission["checkpoint"], "math500", 4)]
    assert result["math500"]["provenance"]["submission_sha256"] == submission["sha256"]
    with pytest.raises(ValueError, match="bound to a different"):
        evaluation.evaluate_live({**submission, "sha256": "b" * 64}, object(), lambda: None,
                                 stage="live-runs/example")
    assert len(calls) == 1


def test_existing_unbound_results_cannot_be_stamped_as_a_submission(bound_evaluation):
    root, calls = bound_evaluation
    directory = root / "data/results/live-runs/example/math500"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text('{"n":2000}')
    with pytest.raises(ValueError, match="unbound"):
        evaluation.evaluate_live({"checkpoint": "elsewhere", "sha256": "a" * 64}, object(),
                                 lambda: None, stage="live-runs/example")
    assert not calls


def test_legacy_evaluation_keeps_its_original_namespace(bound_evaluation):
    _, calls = bound_evaluation
    evaluation.evaluate_live({"checkpoint": "/legacy"}, object(), lambda: None)
    assert calls[0][0] == "live-run"
