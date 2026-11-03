"""Durable reservations and usage receipts for a single budget-owning driver.

The lock coordinates processes sharing a filesystem. A Modal Volume is not a
distributed lock: one driver must own this ledger and commit it before dispatch.
Workers receive bounded deadlines; only the driver mutates reservations.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    """A proposed launch would exceed available funds."""


class BudgetOverrun(BudgetExceeded):
    """Recorded actual usage exceeded its reservation; further work must stop."""


def _money(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Costs and limits must be finite and nonnegative")
    return result


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    """Replace a JSON record durably without exposing partial state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class BudgetLedger:
    def __init__(self, path, *, editor_limit=100, modal_task_limit=150,
                 modal_cycle_limit=42, modal_prior_usage=0.05):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if _money(editor_limit) > 100 or _money(modal_task_limit) > 150:
            raise ValueError("Limits exceed the user's authorized task budgets")
        if _money(modal_cycle_limit) > 42:
            raise ValueError("An increased Modal allowance requires an observed billing record")
        self.initial = {
            "schema_version": 1,
            "limits": {"editor": str(_money(editor_limit)), "modal": str(_money(modal_task_limit))},
            "modal_allowance": {"cycle_limit": str(_money(modal_cycle_limit)),
                                "cycle_usage": str(_money(modal_prior_usage)),
                                "task_spend_at_observation": "0", "observed_at": _utc(),
                                "evidence": {"source": "docs/TASK.md", "basis": "user report"}},
            "modal_prior_usage_floor": str(_money(modal_prior_usage)),
            "reservations": {}, "events": [], "halted": False,
        }
        with self._locked() as state:
            if state["schema_version"] != 1:
                raise ValueError("Unsupported budget ledger version")

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else self.initial
            yield state
            atomic_json(self.path, state)

    @staticmethod
    def _totals(state, kind):
        rows = [r for r in state["reservations"].values() if r["kind"] == kind]
        spent = sum((_money(r["actual_usd"]) for r in rows if r["status"] == "settled"), Decimal(0))
        reserved = sum((_money(r["upper_bound_usd"]) for r in rows if r["status"] == "reserved"), Decimal(0))
        return spent, reserved

    @classmethod
    def _available(cls, state, kind):
        spent, reserved = cls._totals(state, kind)
        available = _money(state["limits"][kind]) - spent - reserved
        if kind == "modal":
            allowance = state["modal_allowance"]
            cycle_remaining = (_money(allowance["cycle_limit"]) - _money(allowance["cycle_usage"])
                               - (spent - _money(allowance["task_spend_at_observation"])) - reserved)
            available = min(available, cycle_remaining)
        return max(Decimal(0), available)

    def reserve(self, kind, upper_bound_usd, stage, *, reservation_id=None, metadata=None):
        """Reserve the entire worst-case charge before launching any work."""
        if kind not in ("editor", "modal"):
            raise ValueError("Budget kind must be editor or modal")
        cost = _money(upper_bound_usd)
        rid = reservation_id or uuid.uuid4().hex
        with self._locked() as state:
            if state["halted"]:
                raise BudgetExceeded("Ledger halted after a measured budget overrun")
            if rid in state["reservations"]:
                old = state["reservations"][rid]
                if (old["status"] == "reserved" and old["kind"] == kind
                        and _money(old["upper_bound_usd"]) == cost and old["stage"] == stage):
                    return rid
                raise ValueError("Reservation ID already belongs to different or finished work")
            remaining = self._available(state, kind)
            if cost > remaining:
                raise BudgetExceeded(f"{kind} stage {stage}: reserve ${cost}, only ${remaining} available")
            state["reservations"][rid] = {
                "kind": kind, "stage": str(stage), "upper_bound_usd": str(cost),
                "status": "reserved", "created_at": _utc(), "metadata": metadata or {},
            }
            state["events"].append({"event": "reserved", "id": rid, "at": _utc()})
        return rid

    def settle(self, reservation_id, actual_usd, *, usage=None, seconds=None, metadata=None):
        """Record actual usage, retaining every response's complete usage object."""
        cost = _money(actual_usd)
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Seconds must be finite and nonnegative")
        overrun = False
        with self._locked() as state:
            row = state["reservations"][reservation_id]
            if row["status"] == "settled":
                if _money(row["actual_usd"]) != cost:
                    raise ValueError("Cannot change a settled receipt")
                return
            if row["status"] != "reserved":
                raise ValueError("Cannot settle a released reservation")
            row.update(status="settled", actual_usd=str(cost), completed_at=_utc(), usage=usage,
                       seconds=seconds, receipt_metadata=metadata or {})
            overrun = cost > _money(row["upper_bound_usd"])
            state["halted"] = state["halted"] or overrun
            state["events"].append({"event": "settled", "id": reservation_id, "at": _utc()})
        if overrun:
            raise BudgetOverrun("Actual charge exceeded its pre-launch reservation; receipt saved, ledger halted")

    def release(self, reservation_id, reason):
        """Release only work known not to have run; uncertain calls stay reserved."""
        if not reason:
            raise ValueError("A release needs an evidence-based reason")
        with self._locked() as state:
            row = state["reservations"][reservation_id]
            if row["status"] == "settled":
                raise ValueError("Cannot release settled usage")
            row.update(status="released", release_reason=str(reason), completed_at=_utc())

    def authorize_editor_limit(self, total_limit, *, authorization_id, evidence):
        """Apply an audited, explicit user authorization for a cumulative cap.

        This is not a top-up or a provider-balance observation. The caller must
        verify the authorization and commit its storage before further dispatch.
        Evidence needs a source, timezone-aware authorized_at, and
        basis='explicit_user_authorization'. Replaying the same ID and payload
        returns its original audit record without changing the current cap,
        even if another authorization has since changed it.
        """
        if not isinstance(authorization_id, str) or not authorization_id.strip():
            raise ValueError("A nonempty stable authorization_id is required")
        try:
            limit = _money(total_limit)
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError("The cumulative editor limit must be finite and nonnegative") from error
        if (
            not isinstance(evidence, dict)
            or not isinstance(evidence.get("source"), str)
            or not evidence["source"].strip()
            or not isinstance(evidence.get("authorized_at"), str)
            or evidence.get("basis") != "explicit_user_authorization"
        ):
            raise ValueError("Evidence requires source, authorized_at, and explicit_user_authorization basis")
        try:
            authorized_at = datetime.fromisoformat(evidence["authorized_at"])
            serialized = json.dumps(evidence, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Authorization evidence must have a valid timestamp and JSON values") from error
        if authorized_at.tzinfo is None:
            raise ValueError("Authorization timestamp must include a timezone")
        evidence = json.loads(serialized)
        with self._locked() as state:
            previous = state.get("editor_authorizations", {}).get(authorization_id)
            if previous is not None:
                if (
                    _money(previous["new_limit_usd"]) != limit
                    or json.dumps(previous["evidence"], sort_keys=True, allow_nan=False) != serialized
                ):
                    raise ValueError("Authorization ID was already used with a different payload")
                return json.loads(json.dumps(previous))
            spent, reserved = self._totals(state, "editor")
            if limit < spent + reserved:
                raise ValueError("The editor limit cannot be below settled spend plus retained reservations")
            record = {
                "authorization_id": authorization_id,
                "prior_limit_usd": state["limits"]["editor"],
                "new_limit_usd": str(limit),
                "evidence": evidence,
                "applied_at": _utc(),
            }
            state["limits"]["editor"] = str(limit)
            state.setdefault("editor_authorizations", {})[authorization_id] = record
            state["events"].append({"event": "editor_limit_authorized", **record})
            return json.loads(json.dumps(record))

    def observe_modal_allowance(self, cycle_limit, cycle_usage, evidence):
        """Apply a fresh authoritative billing observation, never a predicted increase.

        evidence requires source and observed_at; its payload is retained for audit.
        This method must be called by the driver after reading the account's billing
        state, not merely after a successful job or a $10 usage threshold.
        """
        limit, usage = _money(cycle_limit), _money(cycle_usage)
        if not isinstance(evidence, dict) or not evidence.get("source") or not evidence.get("observed_at"):
            raise ValueError("Billing observation needs source and observed_at evidence")
        observed = datetime.fromisoformat(evidence["observed_at"])
        if observed.tzinfo is None:
            raise ValueError("Billing observation timestamp must include a timezone")
        age = (datetime.now(UTC) - observed).total_seconds()
        if age < -60 or age > 900:
            raise ValueError("Billing observation must be current (within 15 minutes)")
        with self._locked() as state:
            spent, _ = self._totals(state, "modal")
            # Provider billing observations can lag jobs we have already timed.
            # An increased allowance never erases our own measured consumption.
            usage = max(usage, spent + _money(state.get("modal_prior_usage_floor", "0.05")))
            state["modal_allowance"] = {"cycle_limit": str(limit), "cycle_usage": str(usage),
                                        "task_spend_at_observation": str(spent),
                                        "observed_at": evidence["observed_at"], "evidence": evidence}
            state["events"].append({"event": "modal_allowance_observed", "at": _utc(), "evidence": evidence})

    def snapshot(self):
        with self._locked() as state:
            result = json.loads(json.dumps(state))
            result["summary"] = {}
            for kind in ("editor", "modal"):
                spent, reserved = self._totals(state, kind)
                result["summary"][kind] = {"spent_usd": float(spent), "reserved_usd": float(reserved),
                                           "available_usd": float(self._available(state, kind))}
            return result


def astra_cost(usage: dict) -> Decimal:
    """Standard Astra price; output_tokens already includes reasoning tokens.

    Source: https://developers.openai.com/api/docs/models/gpt-6-astra
    Cache reads and writes are disjoint portions of input_tokens. A 30m write
    costs 1.25x input; a 12h write costs 2.5x. No double-counted writes.
    """
    n_input = int(usage["input_tokens"])
    n_output = int(usage["output_tokens"])
    detail = usage.get("input_tokens_details") or {}
    cached = int(detail.get("cached_tokens", 0))
    written = int(detail.get("cache_write_tokens", 0))
    lifetimes = detail.get("cache_write_tokens_details") or {}
    write12 = int(lifetimes.get("ttl_12h_tokens", 0))
    write30 = int(lifetimes.get("ttl_30m_tokens", written - write12))
    ordinary = n_input - cached - written
    if min(n_input, n_output, cached, written, ordinary, write12, write30) < 0 or write30 + write12 != written:
        raise ValueError("Inconsistent token usage; refusing to guess its cost")
    in_multiplier = Decimal(2) if n_input > 272_000 else Decimal(1)
    out_multiplier = Decimal("1.5") if n_input > 272_000 else Decimal(1)
    return ((ordinary * 10 + cached + Decimal("12.5") * write30 + write12 * 25) * in_multiplier
            + n_output * 50 * out_multiplier) / 1_000_000


def astra_upper_bound(input_tokens: int, max_output_tokens: int) -> Decimal:
    """Bound a 30m-cache response assuming all input is newly cache-written."""
    if min(input_tokens, max_output_tokens) < 0:
        raise ValueError("Token bounds must be nonnegative")
    return astra_cost({"input_tokens": input_tokens, "output_tokens": max_output_tokens,
                       "input_tokens_details": {"cache_write_tokens": input_tokens}})


GPU_USD_PER_HOUR = {"H100": "3.95", "A100-80GB": "2.50", "A100": "2.50",
                   "L40S": "1.95", "L4": "0.80"}


def compute_cost(seconds, *, gpu=None, cpu_cores=2, memory_gib=16, sandbox=False) -> Decimal:
    """Modal list-price bound; cpu_cores are physical cores, not vCPU threads.

    Include startup, idle/scaledown and CPU/RAM in the supplied lifetime. Use
    maximum allocated resources for a pre-launch bound, measured billable
    resource-seconds for settlement. Exact hourly rates were observed through
    Workspace.billing.rates() on 2026-11-03; the pricing page's per-second display
    rounds some rates down. https://modal.com/docs/sdk/py/latest/Workspace#billingrates
    """
    duration = _money(seconds)
    cpu_rate = Decimal("0.141900" if sandbox else "0.04730")
    memory_rate = Decimal("0.024000" if sandbox else "0.00800")
    rate = _money(cpu_cores) * cpu_rate + _money(memory_gib) * memory_rate
    if gpu:
        rate += Decimal(GPU_USD_PER_HOUR[gpu])
    return duration * rate / 3600
