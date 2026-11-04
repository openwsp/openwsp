"""Durable reservations and usage receipts for a single budget-owning driver.

The lock coordinates processes sharing a filesystem. A Modal Volume is not a
distributed lock: one driver must own this ledger and commit it before dispatch.
Workers receive bounded deadlines; only the driver mutates reservations.
"""
from __future__ import annotations

import fcntl
import hashlib
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


def _allowance_binding(cycle_limit, cycle_usage, evidence, *, observation_id=None, cycle_id=None):
    try:
        limit, usage = _money(cycle_limit), _money(cycle_usage)
        evidence = json.loads(json.dumps(evidence, sort_keys=True, allow_nan=False))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("Billing observation requires finite costs and JSON evidence") from error
    if (not isinstance(evidence, dict) or not isinstance(evidence.get("source"), str)
            or not evidence["source"].strip() or not isinstance(evidence.get("observed_at"), str)):
        raise ValueError("Billing observation needs source and observed_at evidence")
    observed = datetime.fromisoformat(evidence["observed_at"])
    if observed.tzinfo is None:
        raise ValueError("Billing observation timestamp must include a timezone")
    for name, value in (("observation_id", observation_id), ("cycle_id", cycle_id)):
        if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 256):
            raise ValueError(f"{name} must be a nonempty identifier of at most 256 characters")
    binding = {"cycle_limit": format(limit.normalize(), "f"), "cycle_usage": format(usage.normalize(), "f"),
               "evidence": evidence, "cycle_id": cycle_id}
    if observation_id is None:
        encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False)
        observation_id = "legacy-" + hashlib.sha256(encoded.encode()).hexdigest()
    return {"observation_id": observation_id, **binding}


def _provider_usage_anchor(binding):
    """Validate the explicit distinction between provider usage and estimates."""
    evidence = binding["evidence"]
    if "usage_anchor" not in evidence:
        return None
    anchor = evidence["usage_anchor"]
    required = {"basis", "settled_task_estimate_usd", "all_dispatch_terminal"}
    if (evidence.get("basis") != "user_confirmed_dashboard"
            or not isinstance(anchor, dict) or set(anchor) != required
            or anchor["basis"] != "provider_actual_cycle_usage"
            or anchor["all_dispatch_terminal"] is not True):
        raise ValueError("Provider usage anchors require explicit dashboard evidence and terminal-work confirmation")
    try:
        estimate = _money(anchor["settled_task_estimate_usd"])
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("Provider usage anchor needs a finite settled task estimate") from error
    if _money(binding["cycle_usage"]) >= estimate:
        raise ValueError("A provider usage anchor is only needed below conservative settled estimates")
    return estimate


def _check_operator_topups(binding, task_authorization=None):
    if binding["evidence"].get("basis") != "operator_automatic_topups":
        return
    evidence = binding["evidence"]
    if (evidence.get("automatic_topups_authorized") is not True
            or evidence.get("provider_limit_observed") is not False):
        raise ValueError("Automatic top-ups require explicit authorization, not a provider-limit claim")
    limit = _money(binding["cycle_limit"])
    if ((task_authorization is None and limit > 150)
            or (task_authorization is not None and _money(task_authorization["total_limit"]) < limit)):
        raise ValueError("Automatic top-ups require a bound explicit task authorization covering the funding envelope; "
                         "the standalone task cap is $150")


def validate_modal_allowance_observation(observation, *, task_authorization=None):
    """Validate dashboard evidence or explicit operator funding authorization.

    Freshness belongs to the ledger: an identical acknowledged observation may
    safely be retried after its evidence ages. This function authenticates neither
    the dashboard evidence nor the caller's authority to apply it.
    """
    try:
        observation = json.loads(json.dumps(observation, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Observation must contain only finite JSON values") from error
    required = {"observation_id", "cycle_limit", "cycle_usage", "evidence"}
    if (not isinstance(observation, dict) or not required <= observation.keys()
            or observation.keys() - required - {"cycle_id"}):
        raise ValueError("Observation requires observation_id, cycle_limit, cycle_usage and evidence")
    if observation["observation_id"] is None:
        raise ValueError("A stable observation_id is required")
    binding = _allowance_binding(**observation)
    basis = binding["evidence"].get("basis")
    if basis not in {"user_confirmed_dashboard", "operator_automatic_topups"}:
        raise ValueError("Observation requires user_confirmed_dashboard or operator_automatic_topups evidence")
    authorization = (validate_modal_task_authorization(task_authorization)
                     if task_authorization is not None else None)
    # A larger funding envelope is permitted only with the separately bound
    # explicit task authorization. It remains distinct from a provider limit.
    _check_operator_topups(binding, authorization)
    observed = datetime.fromisoformat(binding["evidence"]["observed_at"])
    if observed.utcoffset().total_seconds() != 0:
        raise ValueError("Allowance evidence timestamp must be UTC")
    _provider_usage_anchor(binding)
    return binding


def _task_authorization_binding(kind, total_limit, *, authorization_id, evidence):
    if not isinstance(authorization_id, str) or not authorization_id.strip():
        raise ValueError("A nonempty stable authorization_id is required")
    try:
        limit = _money(total_limit)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"The cumulative {kind} limit must be finite and nonnegative") from error
    if (not isinstance(evidence, dict) or not isinstance(evidence.get("source"), str)
            or not evidence["source"].strip() or not isinstance(evidence.get("authorized_at"), str)
            or evidence.get("basis") != "explicit_user_authorization"):
        raise ValueError("Evidence requires source, authorized_at, and explicit_user_authorization basis")
    try:
        authorized_at = datetime.fromisoformat(evidence["authorized_at"])
        serialized = json.dumps(evidence, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Authorization evidence must have a valid timestamp and JSON values") from error
    if authorized_at.tzinfo is None:
        raise ValueError("Authorization timestamp must include a timezone")
    return {"total_limit": str(limit), "authorization_id": authorization_id, "evidence": json.loads(serialized)}


def validate_modal_task_authorization(authorization):
    """Validate a detached Modal task-cap authorization without reading a ledger."""
    if not isinstance(authorization, dict) or set(authorization) != {"total_limit", "authorization_id", "evidence"}:
        raise ValueError("Task authorization requires total_limit, authorization_id and evidence")
    binding = _task_authorization_binding("modal", **authorization)
    binding["total_limit"] = format(_money(binding["total_limit"]).normalize(), "f")
    return binding


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
    @classmethod
    def from_existing(cls, path):
        """Open an existing ledger without initializing, rewriting, or reserving."""
        instance = cls.__new__(cls)
        instance.path, instance.initial = Path(path), None
        if not instance.path.is_file():
            raise ValueError("An existing budget ledger is required")
        state = json.loads(instance.path.read_text())
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("Unsupported budget ledger version")
        return instance

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
            if state is None:
                raise ValueError("The existing budget ledger disappeared; initialization is forbidden")
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
        return self._authorize_limit("editor", total_limit, authorization_id=authorization_id, evidence=evidence)

    def authorize_modal_limit(self, total_limit, *, authorization_id, evidence):
        """Authorize a cumulative Modal task cap without changing its allowance.

        Uses the same immutable authorization rules as the editor cap. A task
        authorization does not assert or raise a provider account limit, settle
        any prior job, or clear a measured-overrun halt.
        """
        return self._authorize_limit("modal", total_limit, authorization_id=authorization_id, evidence=evidence)

    def _authorize_limit(self, kind, total_limit, *, authorization_id, evidence):
        binding = _task_authorization_binding(kind, total_limit, authorization_id=authorization_id, evidence=evidence)
        with self._locked() as state:
            return self._apply_limit_authorization(state, kind, binding)

    @classmethod
    def _check_limit_authorization(cls, state, kind, binding):
        limit = _money(binding["total_limit"])
        previous = state.get(f"{kind}_authorizations", {}).get(binding["authorization_id"])
        if previous is not None:
            if (_money(previous["new_limit_usd"]) != limit
                    or json.dumps(previous["evidence"], sort_keys=True, allow_nan=False)
                    != json.dumps(binding["evidence"], sort_keys=True, allow_nan=False)):
                raise ValueError("Authorization ID was already used with a different payload")
            return json.loads(json.dumps(previous))
        spent, reserved = cls._totals(state, kind)
        if limit < spent + reserved:
            raise ValueError(f"The {kind} limit cannot be below settled spend plus retained reservations")
        return None

    @classmethod
    def _apply_limit_authorization(cls, state, kind, binding):
        previous = cls._check_limit_authorization(state, kind, binding)
        if previous is not None:
            return previous
        record = {
            "authorization_id": binding["authorization_id"],
            "prior_limit_usd": state["limits"][kind],
            "new_limit_usd": binding["total_limit"],
            "evidence": binding["evidence"],
            "applied_at": _utc(),
        }
        state["limits"][kind] = binding["total_limit"]
        state.setdefault(f"{kind}_authorizations", {})[binding["authorization_id"]] = record
        state["events"].append({"event": f"{kind}_limit_authorized", **record})
        return json.loads(json.dumps(record))

    @classmethod
    def _check_modal_observation(cls, state, binding):
        previous = state.get("modal_observations", {}).get(binding["observation_id"])
        if previous is not None:
            if (json.dumps(previous.get("observation"), sort_keys=True, allow_nan=False)
                    != json.dumps(binding, sort_keys=True, allow_nan=False)):
                raise ValueError("Observation ID was already used with a different payload")
            return json.loads(json.dumps(previous))
        observed = datetime.fromisoformat(binding["evidence"]["observed_at"])
        age = (datetime.now(UTC) - observed).total_seconds()
        if age < -60 or age > 900:
            raise ValueError("Billing observation must be current (within 15 minutes)")
        allowance = state["modal_allowance"]
        # Initial user-reported allowance is not a measured dashboard observation.
        if allowance.get("evidence", {}).get("basis") != "user report":
            previous_at = datetime.fromisoformat(allowance["observed_at"])
            if observed <= previous_at:
                raise ValueError("New billing observations must be newer than the recorded observation")
        if (binding["cycle_id"] is not None and allowance.get("cycle_id") is not None
                and binding["cycle_id"] != allowance["cycle_id"]):
            raise ValueError("A different billing cycle requires explicit rollover handling")
        _check_operator_topups(binding, binding.get("task_authorization"))
        estimate = _provider_usage_anchor(binding)
        if estimate is not None and estimate != cls._totals(state, "modal")[0]:
            raise ValueError("Provider usage anchor does not match current settled task estimates")
        if "task_authorization" in binding:
            cls._check_limit_authorization(state, "modal", binding["task_authorization"])
        return None

    def preview_modal_allowance(self, observation, *, task_authorization=None):
        """Validate an observation against existing state without writing files."""
        binding = validate_modal_allowance_observation(observation, task_authorization=task_authorization)
        if task_authorization is not None:
            binding["task_authorization"] = validate_modal_task_authorization(task_authorization)
        state = json.loads(self.path.read_text())
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported budget ledger version")
        return self._check_modal_observation(state, binding)

    def observe_modal_allowance(self, cycle_limit, cycle_usage, evidence, *, observation_id=None, cycle_id=None,
                               task_authorization=None, idle_owner_verified=False):
        """Apply fresh billing evidence or explicit operator funding authorization.

        Evidence requires source and observed_at; its payload is retained for audit.
        Funding authorization is distinguished from observed provider limits in
        evidence; successful jobs or usage thresholds never authorize an increase.
        Stable IDs bind immutable receipts; identical retries never reapply an old
        observation. A task authorization, when supplied, is applied atomically
        with this observation. A provider-usage anchor additionally requires the
        serialized caller to verify every driver owner is idle/terminal; it never
        rewrites job estimates or removes reservations.
        """
        binding = _allowance_binding(cycle_limit, cycle_usage, evidence,
                                     observation_id=observation_id, cycle_id=cycle_id)
        if task_authorization is not None:
            binding["task_authorization"] = validate_modal_task_authorization(task_authorization)
        with self._locked() as state:
            previous = self._check_modal_observation(state, binding)
            if previous is not None:
                return previous
            estimate = _provider_usage_anchor(binding)
            if estimate is not None and idle_owner_verified is not True:
                raise ValueError("A provider usage anchor requires independently verified idle ownership")
            spent, _ = self._totals(state, "modal")
            prior = state["modal_allowance"]
            known_usage = (_money(prior["cycle_usage"]) + spent
                           - _money(prior["task_spend_at_observation"]))
            anchor_id = prior.get("provider_usage_anchor_id")
            if estimate is not None:
                usage = _money(binding["cycle_usage"])
                anchor_id = binding["observation_id"]
            else:
                usage = max(_money(binding["cycle_usage"]), known_usage)
                if anchor_id is None:
                    usage = max(usage, spent + _money(state.get("modal_prior_usage_floor", "0.05")))
            applied = {"cycle_limit": binding["cycle_limit"], "cycle_usage": str(usage),
                       "task_spend_at_observation": str(spent),
                       "observed_at": binding["evidence"]["observed_at"], "evidence": binding["evidence"],
                       "cycle_id": cycle_id or prior.get("cycle_id")}
            if anchor_id is not None:
                applied["provider_usage_anchor_id"] = anchor_id
            receipt = {"observation_id": binding["observation_id"], "observation": binding,
                       "prior_allowance": json.loads(json.dumps(prior)), "applied_allowance": applied,
                       "applied_at": _utc()}
            if task_authorization is not None:
                receipt["task_authorization"] = self._apply_limit_authorization(
                    state, "modal", binding["task_authorization"],
                )
            state["modal_allowance"] = applied
            state.setdefault("modal_observations", {})[binding["observation_id"]] = receipt
            state["events"].append({"event": "modal_allowance_observed", **receipt})
            return json.loads(json.dumps(receipt))

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
