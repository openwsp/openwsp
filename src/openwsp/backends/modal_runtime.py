"""Modal client setup for proxied environments and read-only budget observations."""
from __future__ import annotations

import os


def configure_tls():
    """Add the caller's configured CA bundle to grpclib without disabling verification.

    grpclib explicitly selects certifi and otherwise ignores SSL_CERT_FILE. Modal
    retains its native HTTP CONNECT proxy support; no routes or proxy variables change.
    """
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if not ca:
        return
    from modal._utils.grpc_utils import ModalChannel
    if getattr(ModalChannel, "_openwsp_ca_configured", False):
        return
    original = ModalChannel._get_default_ssl_context

    def trusted_context(self, **kwargs):
        context = original(self, **kwargs)
        context.load_verify_locations(cafile=ca)
        return context

    ModalChannel._get_default_ssl_context = trusted_context
    ModalChannel._openwsp_ca_configured = True


def allowance(*, workspace=None):
    """Observe public billing data without guessing the workspace's usage limit.

    Modal 1.6.3 exposes rates and invoiced/metered usage through Workspace.billing,
    but no workspace usage-limit field. Environment budgets are a distinct Team /
    Enterprise feature and their zero/unset fields are not a Starter limit.
    Consequently cycle_limit_usd is unknown, even when billing.summary succeeds.
    Retain the last independently verified allowance in the driver's ledger.
    """
    from datetime import UTC, datetime
    from importlib.metadata import PackageNotFoundError, version

    if workspace is None:
        configure_tls()
        from modal import Workspace
        workspace = Workspace.from_context()
    try:
        sdk_version = version("modal")
    except PackageNotFoundError:
        sdk_version = None
    result = {
        "status": "unknown",
        "cycle_limit_usd": None,
        "current_cycle_usage_usd": None,
        "billed_cost_usd": None,
        "billing_summary": None,
        "rates": None,
        "errors": [],
        "evidence": {
            "source": "modal.Workspace.billing.summary/rates",
            "observed_at": datetime.now(UTC).isoformat(),
            "sdk_version": sdk_version,
            "reason": "Public billing APIs do not expose the workspace usage limit; dashboard evidence is required.",
            "docs": "https://modal.com/docs/guide/budgets",
            "usage_may_lag": True,
        },
    }
    try:
        summary = workspace.billing.summary()
        result["current_cycle_usage_usd"] = str(summary.metered_cost)
        result["billed_cost_usd"] = str(summary.billed_cost)
        result["billing_summary"] = {
            "start": summary.start.isoformat(),
            "end": summary.end.isoformat(),
            "metered_cost": str(summary.metered_cost),
            "billed_cost": str(summary.billed_cost),
            "adjustments": {key: str(value) for key, value in summary.adjustments.items()},
            "metered_cost_breakdown": {key: str(value) for key, value in summary.metered_cost_breakdown.items()},
        }
    except Exception as error:  # noqa: BLE001 - billing may be unavailable on the current plan
        result["errors"].append({"operation": "billing.summary", "type": type(error).__name__})
    try:
        rates = workspace.billing.rates()
        result["rates"] = {key: str(value) for key, value in rates.items()
                           if key.startswith(("gpu_hour_", "cpu_hour_", "mem_gib_", "egress_", "volume_storage_"))}
    except Exception as error:  # noqa: BLE001 - preserve unknown instead of inventing a limit
        result["errors"].append({"operation": "billing.rates", "type": type(error).__name__})
    if result["billing_summary"] is None and result["rates"] is None:
        result["status"] = "unavailable"
    return result
