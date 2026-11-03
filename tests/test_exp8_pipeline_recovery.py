"""A detached failure publishes its durable blocker without resubmitting work."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from openwsp.budget import BudgetExceeded, BudgetLedger
from replications.exp8 import artifacts, pipeline, verification


@pytest.mark.parametrize("publication_fails", [False, True])
def test_failed_detached_run_publishes_blocker_after_reload(monkeypatch, tmp_path, publication_fails):
    data = tmp_path / "data"
    (data / "benchmarks").mkdir(parents=True)
    (data / "benchmarks/ifeval.jsonl").write_text("")
    (data / "benchmarks/math500.jsonl").write_text("")
    (data / "p50c").mkdir()
    (data / "p50c/config.json").write_text("{}")
    state_path = data / "state.json"
    state_path.write_text(json.dumps({"stages": {
        "prepare": {"status": "done"}, "kernel": {"status": "passed"},
        "calibration_base": {"status": "done", "accuracy": .079},
        "calibration_p50c": {"status": "done", "accuracy": .481},
    }}))
    events = []
    module = ModuleType("openwsp.backends.modal_app")
    module.data_volume = SimpleNamespace(reload=lambda: events.append("reload"), commit=lambda: None)
    module.prepare = module.Generator = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(pipeline, "Path", lambda value: data if value == "/data" else Path(value))

    def ledger(path):
        assert events == ["reload"]
        events.append("ledger")
        return BudgetLedger(path)

    monkeypatch.setattr(pipeline, "BudgetLedger", ledger)
    monkeypatch.setattr(pipeline, "ProtocolGPUJobs", lambda *args, **kwargs: object())

    def fail(*args):
        raise BudgetExceeded("No remaining authorized GPU headroom")

    def publish():
        assert json.loads(state_path.read_text())["status"] == "blocked"
        events.append("published_blocker")
        if publication_fails:
            raise OSError("Git unavailable")
        return {"status": "pushed"}

    monkeypatch.setattr(verification, "verify_forward", fail)
    monkeypatch.setattr(artifacts, "publish", publish)
    with pytest.raises(BudgetExceeded, match="authorized GPU headroom"):
        pipeline.run("all", {}, app=object())
    saved = json.loads(state_path.read_text())
    assert events == ["reload", "ledger", "published_blocker"]
    assert saved["error_type"] == "BudgetExceeded"
    assert saved["publication"]["status"] == ("blocked" if publication_fails else "pushed")
