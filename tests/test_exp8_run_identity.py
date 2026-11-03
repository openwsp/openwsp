import json

import pytest

from replications.exp8.run_identity import LiveIdentity, live_identity


def test_original_paths_stay_unchanged():
    identity = live_identity({})
    assert str(identity.root) == "/data/live-run"
    assert str(identity.work) == "/work/live"
    assert identity.work_subpath == "live"
    assert identity.job_prefix == "live"
    assert identity.evaluation_stage == "live-run"


def test_codex_attempt_is_isolated_and_binding_is_immutable(tmp_path):
    identity = live_identity({"editor": "codex", "run_id": "codex-20261103"})
    assert str(identity.root) == "/data/live-runs/codex-20261103"
    assert str(identity.work) == "/work/live-runs/codex-20261103"
    assert identity.job_prefix == "live/codex-20261103"
    first = identity.record(tmp_path)
    assert identity.record(tmp_path) == first
    assert not (tmp_path / "live-run").exists()
    path = tmp_path / identity.relative / "identity.json"
    path.write_text(json.dumps({**first, "task_sha256": "different"}))
    with pytest.raises(ValueError, match="conflicts"):
        identity.record(tmp_path)


@pytest.mark.parametrize("run_id", [None, "../live-run", "/data/live-run", "a/b", "A", "", "a" * 65])
def test_codex_rejects_unsafe_or_missing_identity(run_id):
    with pytest.raises(ValueError):
        LiveIdentity("codex", run_id)


def test_cannot_adopt_unbound_existing_state(tmp_path):
    identity = LiveIdentity("codex", "new-run")
    root = tmp_path / identity.relative
    root.mkdir(parents=True)
    (root / "state.json").write_text('{"passes":4}')
    with pytest.raises(ValueError, match="adopt"):
        identity.record(tmp_path)
