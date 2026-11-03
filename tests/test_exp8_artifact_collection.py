"""Publish forward-file references while keeping full texts on the volume."""

import hashlib
import json

from openwsp.publishing import _validate_file
from replications.exp8.artifacts import collect


def test_collect_recomputes_finished_metrics_from_compact_samples_without_changing_source(tmp_path):
    data, staging = tmp_path / "data", tmp_path / "staging"
    run = data / "results/calibration_base"
    run.mkdir(parents=True)
    summary = run / "summary.json"
    summary.write_text(json.dumps({"n": 2, "correct": 1, "accuracy": 0.5}))
    original = summary.read_bytes()
    samples = run / "samples.jsonl"
    samples.write_text("\n".join(json.dumps(row) for row in [
        {"problem_id": "a", "correct": True, "finish_reason": "length"},
        {"problem_id": "b", "correct": False, "finish_reason": "stop"},
    ]))
    collect(data, staging)
    result = json.loads((staging / "replications/exp8/results/calibration_base/summary.json").read_text())
    assert result["accuracy"] == 0.5
    assert result["finished_n"] == 1 and result["finished_correct"] == 0 and result["accuracy_finished"] == 0
    assert result["provenance"]["finished_metrics"]["samples_sha256"] == hashlib.sha256(samples.read_bytes()).hexdigest()
    assert summary.read_bytes() == original


def test_collect_publishes_verified_file_references_without_raw_responses(tmp_path):
    data, staging = tmp_path / "data", tmp_path / "staging"
    requests = data / "live-run/requests"
    requests.mkdir(parents=True)
    receipt = {"request": 7, "volume": "openwsp-exp8-work", "files": [
        {"path": "/work/gen_7.jsonl", "volume_path": "/live/gen_7.jsonl",
         "bytes": 14, "sha256": "a" * 64},
    ]}
    (requests / "fwd_7.delivery.json").write_text(json.dumps(receipt))
    (requests / "private-response.json").write_text('{"generated_text":"private generation"}')
    _, paths = collect(data, staging)
    manifest = staging / "replications/exp8/results/live-run/forward-artifacts.json"
    assert json.loads(manifest.read_text()) == {"receipts": [receipt]}
    assert all("private-response" not in path for path in paths)
    for relative in paths:
        published = staging / relative
        assert "private generation" not in published.read_text()
        _validate_file(published)
