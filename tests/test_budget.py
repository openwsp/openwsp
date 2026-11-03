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
