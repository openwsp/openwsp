from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from openwsp.budget import (
    BudgetExceeded,
    BudgetLedger,
    BudgetOverrun,
    astra_cost,
    astra_upper_bound,
    compute_cost,
)


def test_astra_usage_bills_cache_categories_and_reasoning_once():
    usage = {"input_tokens": 12000, "output_tokens": 400,
             "input_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 7000,
                                      "cache_write_tokens_details": {"ttl_30m_tokens": 3000,
                                                                     "ttl_12h_tokens": 4000}},
             "output_tokens_details": {"reasoning_tokens": 300}}
    assert astra_cost(usage) == Decimal("0.1715")
    assert astra_cost({"input_tokens": 272001, "output_tokens": 1000}) == Decimal("5.51502")
    assert astra_upper_bound(1000, 1000) == Decimal("0.0625")
    with pytest.raises(ValueError, match="Inconsistent"):
        astra_cost({"input_tokens": 1, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 2}})


def test_modal_limit_does_not_rise_without_observation(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    first = ledger.reserve("modal", 41, "calibration")
    ledger.settle(first, 32)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("modal", 10, "live")
    assert ledger.snapshot()["summary"]["modal"]["available_usd"] == pytest.approx(9.95)
    with pytest.raises(ValueError, match="source"):
        ledger.observe_modal_allowance(200, 32.05, {})
    ledger.observe_modal_allowance(200, 32.05,
                                  {"source": "modal.billing", "observed_at": datetime.now(UTC).isoformat()})
    assert ledger.snapshot()["summary"]["modal"]["available_usd"] == 118
    with pytest.raises(BudgetExceeded):
        ledger.reserve("modal", 119, "too-much")


def test_reservations_survive_restart_and_actuals_release_only_unused_funds(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path)
    rid = ledger.reserve("editor", 90, "live", reservation_id="stable")
    assert ledger.reserve("editor", 90, "live", reservation_id="stable") == rid
    with pytest.raises(ValueError):
        ledger.reserve("modal", 1, "other", reservation_id="stable")
    resumed = BudgetLedger(path)
    with pytest.raises(BudgetExceeded):
        resumed.reserve("editor", 11, "next")
    resumed.settle(rid, 20, usage={"input_tokens": 100, "output_tokens": 10})
    resumed.settle(rid, 20)
    with pytest.raises(ValueError, match="finished"):
        resumed.reserve("editor", 90, "live", reservation_id="stable")
    snapshot = resumed.snapshot()
    assert snapshot["summary"]["editor"] == {"spent_usd": 20, "reserved_usd": 0, "available_usd": 80}
    assert snapshot["reservations"][rid]["usage"]["input_tokens"] == 100


def test_lagging_billing_observation_does_not_erase_known_usage(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    rid = ledger.reserve("modal", 30, "job")
    ledger.settle(rid, 30)
    ledger.observe_modal_allowance(50, 5,
                                  {"source": "modal.billing", "observed_at": datetime.now(UTC).isoformat()})
    assert ledger.snapshot()["summary"]["modal"]["available_usd"] == pytest.approx(19.95)


def test_actual_overrun_is_saved_and_halts_future_launches(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    rid = ledger.reserve("modal", 1, "job")
    with pytest.raises(BudgetOverrun):
        ledger.settle(rid, 2)
    assert ledger.snapshot()["summary"]["modal"]["spent_usd"] == 2
    with pytest.raises(BudgetExceeded, match="halted"):
        ledger.reserve("editor", 1, "another")


def _try_reservation(path):
    try:
        BudgetLedger(path, editor_limit=1).reserve("editor", 1, "concurrent")
        return True
    except BudgetExceeded:
        return False


def test_concurrent_reservations_cannot_overspend(tmp_path):
    path = str(tmp_path / "budget.json")
    BudgetLedger(path, editor_limit=1)
    with ProcessPoolExecutor(max_workers=3) as pool:
        assert sum(pool.map(_try_reservation, [path] * 6)) == 1


def test_compute_cost_includes_sandbox_memory_and_cpu():
    assert compute_cost(3600, gpu="H100", cpu_cores=2, memory_gib=16) == Decimal("4.1726")
    assert compute_cost(3600, cpu_cores=2, memory_gib=96, sandbox=True) == Decimal("2.5878")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_costs_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        BudgetLedger(tmp_path / "budget.json").reserve("modal", value, "invalid")


def authorization_evidence(**extra):
    return {
        "source": "user-message:explicit-budget-change",
        "authorized_at": "2026-11-03T12:30:00+00:00",
        "basis": "explicit_user_authorization",
        **extra,
    }


def test_explicit_editor_authorization_sets_absolute_cap_without_rewriting_usage(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path)
    settled = ledger.reserve("editor", 80, "earlier")
    ledger.settle(settled, "78.292765", usage={"response_id": "saved-receipt"})
    ledger.reserve("editor", 10, "still-pending", reservation_id="pending")
    ledger.reserve("modal", 2, "compute", reservation_id="modal-work")
    before = ledger.snapshot()
    evidence = authorization_evidence(
        additional_authorized_usd="160.85", reserved_for_other_work_usd="160.85",
        prior_settled_usd="78.292765",
    )
    record = ledger.authorize_editor_limit(
        "239.142765", authorization_id="exp8-additional-160.85", evidence=evidence,
    )
    after = ledger.snapshot()
    assert record["prior_limit_usd"] == "100" and record["new_limit_usd"] == "239.142765"
    assert after["limits"]["editor"] == "239.142765"
    assert after["summary"]["editor"] == {
        "spent_usd": 78.292765, "reserved_usd": 10.0, "available_usd": 150.85,
    }
    assert after["reservations"] == before["reservations"]
    for field in ("modal_allowance", "modal_prior_usage_floor", "halted"):
        assert after[field] == before[field]
    assert after["limits"]["modal"] == before["limits"]["modal"]
    assert after["summary"]["modal"] == before["summary"]["modal"]
    assert after["events"][:-1] == before["events"]
    assert after["events"][-1] == {"event": "editor_limit_authorized", **record}
    assert after["editor_authorizations"] == {"exp8-additional-160.85": record}
    evidence["additional_authorized_usd"] = "changed outside ledger"
    record["evidence"]["source"] = "changed return value"
    assert BudgetLedger(path).snapshot() == after
    with pytest.raises(ValueError, match="authorized task budgets"):
        BudgetLedger(path, editor_limit="239.142765")


def test_authorization_replay_never_restores_spent_funds_or_a_superseded_cap(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    evidence = authorization_evidence()
    original = ledger.authorize_editor_limit(200, authorization_id="first", evidence=evidence)
    rid = ledger.reserve("editor", 30, "after-authorization")
    ledger.settle(rid, 20)
    consumed = ledger.snapshot()
    assert ledger.authorize_editor_limit("200.00", authorization_id="first", evidence=evidence) == original
    assert ledger.snapshot() == consumed
    assert consumed["summary"]["editor"]["available_usd"] == 180
    for new_cap, key in ((250, "higher"), (150, "lower")):
        ledger.authorize_editor_limit(new_cap, authorization_id=key, evidence=evidence)
        updated = ledger.snapshot()
        assert ledger.authorize_editor_limit(200, authorization_id="first", evidence=evidence) == original
        assert ledger.snapshot() == updated
        assert updated["summary"]["editor"]["available_usd"] == new_cap - 20


def test_conflicting_authorization_replays_fail_without_mutation(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    evidence = authorization_evidence(approved=True)
    ledger.authorize_editor_limit(200, authorization_id="same", evidence=evidence)
    before = ledger.snapshot()
    for limit, changed in (
        (201, evidence),
        (200, {**evidence, "source": "different"}),
        (200, {**evidence, "approved": 1}),
    ):
        with pytest.raises(ValueError, match="different payload"):
            ledger.authorize_editor_limit(limit, authorization_id="same", evidence=changed)
        assert ledger.snapshot() == before


def test_new_cap_must_cover_spend_and_unresolved_reservations(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    rid = ledger.reserve("editor", 70, "finished")
    ledger.settle(rid, 60)
    ledger.reserve("editor", 30, "uncertain", reservation_id="unknown")
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="settled spend plus retained reservations"):
        ledger.authorize_editor_limit("89.99", authorization_id="too-low", evidence=authorization_evidence())
    assert ledger.snapshot() == before
    ledger.authorize_editor_limit(90, authorization_id="exact", evidence=authorization_evidence())
    assert ledger.snapshot()["summary"]["editor"]["available_usd"] == 0


def test_editor_authorization_cannot_clear_a_budget_halt(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    rid = ledger.reserve("editor", 1, "overrun")
    with pytest.raises(BudgetOverrun):
        ledger.settle(rid, 2)
    ledger.authorize_editor_limit(200, authorization_id="raised", evidence=authorization_evidence())
    assert ledger.snapshot()["halted"] is True
    with pytest.raises(BudgetExceeded, match="halted"):
        ledger.reserve("editor", 1, "still-blocked")


@pytest.mark.parametrize("limit", [True, -1, "NaN", "Infinity", "not-money"])
def test_editor_authorization_rejects_invalid_caps(tmp_path, limit):
    ledger = BudgetLedger(tmp_path / "budget.json")
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ledger.authorize_editor_limit(limit, authorization_id="bad", evidence=authorization_evidence())
    assert ledger.snapshot() == before


@pytest.mark.parametrize("key,evidence", [
    ("", authorization_evidence()),
    ("   ", authorization_evidence()),
    (None, authorization_evidence()),
    ("id", {}),
    ("id", authorization_evidence(source="")),
    ("id", authorization_evidence(basis="provider balance")),
    ("id", authorization_evidence(authorized_at="2026-11-03T12:30:00")),
    ("id", authorization_evidence(authorized_at="not-a-time")),
    ("id", authorization_evidence(detail=float("nan"))),
])
def test_editor_authorization_requires_explicit_auditable_evidence(tmp_path, key, evidence):
    ledger = BudgetLedger(tmp_path / "budget.json")
    before = ledger.snapshot()
    with pytest.raises(ValueError):
        ledger.authorize_editor_limit(200, authorization_id=key, evidence=evidence)
    assert ledger.snapshot() == before


def _apply_same_editor_authorization(path):
    return BudgetLedger(path).authorize_editor_limit(
        200, authorization_id="one-user-message", evidence=authorization_evidence(),
    )


def test_concurrent_authorization_replays_add_one_event(tmp_path):
    path = str(tmp_path / "budget.json")
    BudgetLedger(path)
    with ProcessPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(_apply_same_editor_authorization, [path] * 6))
    assert all(record == records[0] for record in records)
    snapshot = BudgetLedger(path).snapshot()
    assert snapshot["limits"]["editor"] == "200"
    assert len(snapshot["editor_authorizations"]) == len(snapshot["events"]) == 1
