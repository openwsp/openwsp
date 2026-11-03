import json
from decimal import Decimal

import pytest

from openwsp.budget import BudgetExceeded, BudgetLedger
from replications.exp8 import editor_validation as validation


def existing_ledger(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    rid = ledger.reserve("editor", 79, "prior-live", reservation_id="prior-live")
    ledger.settle(rid, "78.292765", usage={"prior": True})
    return ledger


def test_checked_in_authorization_preserves_plan_b_and_old_spend(tmp_path):
    ledger = existing_ledger(tmp_path)
    receipt = validation.apply_authorization(ledger)
    assert receipt["new_limit_usd"] == "239.142765"
    assert ledger.snapshot()["summary"]["editor"]["available_usd"] == 160.85
    rid = ledger.reserve("editor", 1, "validation")
    ledger.settle(rid, "0.5")
    assert validation.apply_authorization(ledger) == receipt
    assert ledger.snapshot()["summary"]["editor"]["available_usd"] == 160.35
    assert ledger.snapshot()["reservations"]["prior-live"]["actual_usd"] == "78.292765"


def test_existing_task_authorization_does_not_silently_expand_fork_budget(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    with pytest.raises(ValueError, match="not a fresh run"):
        validation.apply_authorization(ledger)
    assert ledger.snapshot()["limits"]["editor"] == "100"


def test_probe_cap_includes_spend_and_unknown_reservations_across_restarts(tmp_path):
    ledger = existing_ledger(tmp_path)
    validation.apply_authorization(ledger)
    budget = validation.ProbeBudget(ledger)
    rid = budget.reserve("editor", "1.2", "editor:transport-probe:low:1", reservation_id="probe1")
    budget.settle(rid, "0.8")
    budget.reserve("editor", "1.0", "editor:transport-probe:default:1", reservation_id="probe2")
    restarted = validation.ProbeBudget(ledger)
    with pytest.raises(BudgetExceeded, match="separate"):
        restarted.reserve("editor", "0.21", "editor:transport-probe:default:2")
    assert restarted.reserve("editor", "1.0", "editor:transport-probe:default:1", reservation_id="probe2") == "probe2"
    assert restarted.totals() == {"spent_usd": "0.8", "reserved_usd": "1.0", "limit_usd": "2.00"}
    with pytest.raises(ValueError, match="only admits"):
        restarted.reserve("modal", "0.1", "anything")


def test_probe_runs_two_isolated_modes_and_resumes_without_more_calls(monkeypatch, tmp_path):
    ledger = existing_ledger(tmp_path)
    calls = []

    class FakeEditor:
        def __init__(self, root, budget, *, max_output_tokens, verbosity, commit):
            self.root, self.budget, self.verbosity = root, budget, verbosity
            self.root.mkdir(parents=True, exist_ok=True)
            assert max_output_tokens == 4096

        def complete(self, prompt, *, request_key):
            calls.append((request_key, prompt, self.verbosity))
            rid = self.budget.reserve("editor", Decimal("0.2"), "editor:" + request_key)
            self.budget.settle(rid, Decimal("0.05"), usage={"input_tokens": 10, "output_tokens": 20})
            self.last_response = {"response_id": request_key, "response_file": "response.json",
                                  "usage": {"input_tokens": 10, "output_tokens": 20}, "cost_usd": "0.05"}
            (self.root / "response.json").write_text(json.dumps({"text": {"verbosity": self.verbosity or "medium"}}))
            return "A sentence.\n```python\nprint(1 + 1)\n```" if request_key.endswith(":1") else 'A sentence.\n```forward\n{"type":"probe","value":2}\n```'

    monkeypatch.setattr(validation, "AstraEditor", FakeEditor)
    result = validation.run_probe(ledger, lambda: None, tmp_path / "probe")
    assert result["status"] == "passed_transport_smoke"
    assert [row[2] for row in calls] == ["low", "low", None, None]
    assert calls[0][1] == calls[2][1] and calls[1][1] == calls[3][1]
    assert result["budget"]["spent_usd"] == "0.20"
    assert all("text" not in row and row["ended_after_single_fenced_block"] for row in result["cases"].values())
    validation.run_probe(ledger, lambda: None, tmp_path / "probe")
    assert len(calls) == 4


def test_live_review_requires_checks_and_preserves_trajectory(monkeypatch, tmp_path):
    ledger = existing_ledger(tmp_path)
    probe, live = tmp_path / "probe", tmp_path / "live"
    probe.mkdir()
    (live / "editor").mkdir(parents=True)
    snapshot = {"turn": 26, "passes": 4, "runs": 10, "done": False, "pending": {"kind": "editor", "turn": 27}}
    (live / "state.json").write_text(json.dumps(snapshot))
    incident = json.loads((validation.ROOT / "results/live-run/editor-format-loop.json").read_text())
    response_id = next(row["response_id"] for row in incident["turns"] if row["turn"] == "t026")
    previous = {"previous_response_id": response_id, "pending": None, "context_tokens": 830000}
    (live / "editor/editor_state.json").write_text(json.dumps(previous))
    (probe / "summary.json").write_text(json.dumps({"status": "blocked", "cases": {}}))
    with pytest.raises(ValueError, match="all four"):
        validation.review_live_transport(ledger, lambda: None, probe, live)
    assert ledger.snapshot()["limits"]["editor"] == "100"
    (probe / "summary.json").write_text(json.dumps({"status": "passed_transport_smoke", "cases": {
        f"transport-probe:{mode}:{turn}": {"status": "completed", "ended_after_single_fenced_block": True}
        for mode in ("low", "default") for turn in (1, 2)}}))
    acknowledged = []

    class ReviewOnly:
        def __init__(self, root, budget, *, verbosity, commit):
            assert root == live / "editor" and verbosity == "low"

        def acknowledge_runaway(self, rid, **kwargs):
            assert rid == response_id
            acknowledged.append(kwargs)
            return {"response_id": rid, **kwargs}

    monkeypatch.setattr(validation, "AstraEditor", ReviewOnly)
    result = validation.review_live_transport(ledger, lambda: None, probe, live)
    assert result["editor_options"] == {"verbosity": "low", "max_output_tokens": 32768}
    assert result["preserved"]["context_tokens"] == 830000
    assert json.loads((live / "state.json").read_text()) == snapshot
    assert json.loads((live / "editor/editor_state.json").read_text()) == previous
    assert len(acknowledged) == 1
    snapshot["turn"] = 27
    (live / "state.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="only to the recorded"):
        validation.review_live_transport(ledger, lambda: None, probe, live)
