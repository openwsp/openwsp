from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from openwsp.backends.modal_runtime import allowance


def test_summary_zero_does_not_mean_zero_or_increased_allowance():
    summary = SimpleNamespace(start=datetime(2026, 11, 1, tzinfo=UTC),
                              end=datetime(2026, 12, 1, tzinfo=UTC),
                              metered_cost=Decimal(0), billed_cost=Decimal(0),
                              adjustments={"Credits": Decimal(0)}, metered_cost_breakdown={})
    billing = SimpleNamespace(summary=lambda: summary,
                              rates=lambda: {"gpu_hour_cost_h100": Decimal("3.95")})
    result = allowance(workspace=SimpleNamespace(billing=billing))
    assert result["status"] == "unknown"
    assert result["cycle_limit_usd"] is None
    assert result["current_cycle_usage_usd"] == "0"
    assert result["rates"]["gpu_hour_cost_h100"] == "3.95"
    assert result["evidence"]["usage_may_lag"]


def test_unavailable_billing_preserves_unknown_without_error_payload():
    def denied():
        raise PermissionError("Private request details must not reach logs")
    result = allowance(workspace=SimpleNamespace(billing=SimpleNamespace(summary=denied, rates=denied)))
    assert result["status"] == "unavailable"
    assert result["cycle_limit_usd"] is None
    assert result["current_cycle_usage_usd"] is None
    assert result["errors"] == [{"operation": "billing.summary", "type": "PermissionError"},
                                {"operation": "billing.rates", "type": "PermissionError"}]
    assert "Private" not in str(result)
