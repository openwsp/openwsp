from decimal import Decimal

import pytest

from openwsp.backends.driver_budget import DriverBudgetLease, DriverLeaseError
from openwsp.budget import BudgetExceeded, BudgetLedger, compute_cost


def test_driver_full_timeout_is_reserved_before_dispatch(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    committed = []
    ticket = DriverBudgetLease.prepare(ledger, commit=lambda: committed.append(ledger.snapshot()))
    assert committed[0]["reservations"][ticket["reservation_id"]]["status"] == "reserved"
    assert Decimal(ticket["upper_bound_usd"]) == compute_cost(86_525, cpu_cores=1, memory_gib=4)
    assert float(ticket["upper_bound_usd"]) < 1.907


def test_driver_cannot_claim_changed_resource_bounds_or_reuse_ticket(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ticket = DriverBudgetLease.prepare(ledger)
    with pytest.raises(DriverLeaseError, match="differs"):
        DriverBudgetLease.claim(ledger, dict(ticket, memory_gib=96), owner_id="call-1")
    DriverBudgetLease.claim(ledger, ticket, owner_id="call-1")
    with pytest.raises(DriverLeaseError, match="twice"):
        DriverBudgetLease.claim(ledger, ticket, owner_id="call-2")


def test_worker_finish_retains_funds_until_provider_completion(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ticket = DriverBudgetLease.prepare(ledger)
    driver = DriverBudgetLease.claim(ledger, ticket, owner_id="call-1")
    receipt = driver.finish(elapsed_seconds=3600)
    assert ledger.snapshot()["reservations"][ticket["reservation_id"]]["status"] == "reserved"
    assert driver.finish() == receipt
    evidence = {"source": "modal.FunctionCall.get", "observed_at": "2026-11-03T06:00:00Z",
                "provider_call_id": "call-1", "provider_terminal_state": "completed", "elapsed_seconds_bound": 3610}
    with pytest.raises(ValueError, match="identify"):
        DriverBudgetLease.settle_completed(ledger, ticket, receipt, evidence=dict(evidence, provider_call_id="other"))
    cost = DriverBudgetLease.settle_completed(ledger, ticket, receipt, evidence=evidence)
    assert cost == compute_cost(3735, cpu_cores=1, memory_gib=4)
    assert ledger.snapshot()["summary"]["modal"]["reserved_usd"] == 0
    assert DriverBudgetLease.settle_completed(ledger, ticket, receipt, evidence=evidence) == cost


def test_missing_duration_uses_full_driver_bound(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ticket = DriverBudgetLease.prepare(ledger)
    receipt = DriverBudgetLease.claim(ledger, ticket, owner_id="call-1").finish(elapsed_seconds=10)
    with pytest.raises(ValueError, match="terminal-state"):
        DriverBudgetLease.settle_completed(ledger, ticket, receipt, evidence={})
    cost = DriverBudgetLease.settle_completed(ledger, ticket, receipt,
                                             evidence={"source": "modal.FunctionCall.get",
                                                       "observed_at": "2026-11-03T06:00:00Z",
                                                       "provider_call_id": "call-1",
                                                       "provider_terminal_state": "completed"})
    assert cost == Decimal(ticket["upper_bound_usd"])


def test_lost_driver_is_charged_full_bound_once(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ticket = DriverBudgetLease.prepare(ledger)
    DriverBudgetLease.claim(ledger, ticket, owner_id="call-1")
    cost = DriverBudgetLease.settle_worst_case(ledger, ticket, reason="Provider preempted the call; no receipt")
    assert cost == Decimal(ticket["upper_bound_usd"])
    assert DriverBudgetLease.settle_worst_case(ledger, ticket, reason="Repeated reconciliation") == cost


def test_prepare_stable_id_is_idempotent_only_before_claim(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    ticket = DriverBudgetLease.prepare(ledger, reservation_id="stable")
    assert DriverBudgetLease.prepare(ledger, reservation_id="stable") == ticket
    DriverBudgetLease.claim(ledger, ticket, owner_id="call-1")
    with pytest.raises(DriverLeaseError, match="claimed"):
        DriverBudgetLease.prepare(ledger, reservation_id="stable")


def test_insufficient_funds_block_driver_dispatch(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", modal_cycle_limit=1)
    with pytest.raises(BudgetExceeded):
        DriverBudgetLease.prepare(ledger)
