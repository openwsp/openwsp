"""Publish forward-file references while keeping full texts on the volume."""

import json

from openwsp.publishing import _validate_file
from replications.exp8.artifacts import collect


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
