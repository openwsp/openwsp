"""Durable ownership of a single provider-hosted budget driver.

Route launch and watchdog requests through the SAME serialized Modal function
(``max_containers=1``, ``max_inputs=1``), and disable provider retries on the
driver. A local flock is not a distributed lock for a Modal Volume. The driver
alone owns the ledger while running; pass a ledger factory so even construction
cannot rewrite a stale Volume snapshot. This coordinator reads that ledger only
before dispatch or after the provider confirms the previous call is terminal.

A POST with a lost response is deliberately blocked: its reservation and launch
intent remain available for operator reconciliation, with no duplicate dispatch.
The watchdog may poll every 15 minutes; stale heartbeats default to 600 seconds.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ..budget import BudgetExceeded, atomic_json
from .driver_budget import DriverBudgetLease, DriverLeaseError


def _provider_call(call_id):
    import modal
    return modal.FunctionCall.from_id(call_id)


def _remote_failure(error):
    """Only Modal's reconstructed remote traceback establishes a user-code failure."""
    try:
        from modal._traceback import traceback_contains_remote_call
    except ImportError:
        return False
    return traceback_contains_remote_call(error.__traceback__)


def _failure_kind(error):
    """Classify without persisting provider messages, which can contain secrets."""
    name = type(error).__name__
    message = str(error).lower()
    if any(word in message for word in (
        "unauthorized", "unauthenticated", "permission denied", "invalid api key",
        "insufficient_quota", "quota exceeded", "quota exhausted", "billing limit",
        "usage limit", "budget exceeded", "401", "403",
    )) or name in {
        "AuthError", "AuthenticationError", "PermissionDeniedError", "BudgetExceeded", "ResourceExhaustedError",
    }:
        return "auth_quota_budget"
    if name == "InputCancellation":
        return "cancelled"
    # These SDK exceptions follow a returned function result. OutputExpiredError
    # means the SDK observed no outputs AND no unfinished inputs. RPC failures
    # instead map to the communication errors below and never establish exit.
    if name in {"FunctionTimeoutError", "RemoteError", "InternalFailure", "OutputExpiredError"} or _remote_failure(error):
        return "failed"
    if isinstance(error, TimeoutError) or name == "TimeoutError":
        return "running"
    if isinstance(error, (ConnectionError, OSError)) or name in {
        "ConnectionError", "ServiceError", "NotFoundError", "GRPCError", "InternalError", "DataLossError",
        "InvalidError", "ConflictError", "UnimplementedError", "VersionError", "ClientClosed", "CancelledError",
    }:
        return "unknown"
    if any(base.__name__ == "Error" and base.__module__.startswith("modal") for base in type(error).__mro__):
        return "unknown"
    return "unknown"


class DriverCoordinator:
    def __init__(self, ledger, path, *, state_path, commit=lambda: None,
                 reload=lambda: None, call_factory=None, clock=time.time,
                 lease_options=None, stale_seconds=600, max_restarts=3):
        if not math.isfinite(stale_seconds) or stale_seconds <= 0:
            raise ValueError("stale_seconds must be finite and positive")
        if not isinstance(max_restarts, int) or not 0 <= max_restarts <= 3:
            raise ValueError("At most three automatic restarts per stage are allowed")
        self.ledger_source, self.path = ledger, Path(path)
        self.state_path = Path(state_path)
        self.commit, self.reload, self.clock = commit, reload, clock
        self.call_factory = call_factory or _provider_call
        self.lease_options = dict(lease_options or {})
        if {"stage", "reservation_id", "commit"} & self.lease_options.keys():
            raise ValueError("Coordinator owns lease stage, reservation ID and commit")
        self.stale_seconds, self.max_restarts = stale_seconds, max_restarts
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _ledger(self):
        return self.ledger_source() if callable(self.ledger_source) else self.ledger_source

    @contextmanager
    def _locked(self):
        # Volume.reload refuses open files on that Volume. Keep this process-local
        # lock on the container filesystem; provider serialization owns the rest.
        lock_name = "openwsp-coordinator-" + hashlib.sha256(str(self.path.resolve()).encode()).hexdigest() + ".lock"
        with (Path(tempfile.gettempdir()) / lock_name).open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            self.reload()
            self.record = self._read(self.path) or {
                "schema_version": 1, "current": None, "restart_counts": {}, "history": [],
            }
            if self.record.get("schema_version") != 1:
                raise ValueError("Unsupported driver coordinator journal")
            yield

    @staticmethod
    def _read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    def _write(self):
        atomic_json(self.path, self.record)
        self.commit()

    def _view(self, status=None, reason=None):
        current = self.record.get("current") or {}
        result = {
            "status": status or current.get("phase", "idle"),
            "stage": current.get("stage"), "call_id": current.get("call_id"),
            "restart_counts": dict(self.record["restart_counts"]),
        }
        reason = reason or current.get("reason")
        if reason:
            result["reason"] = reason
        return result

    def _block(self, reason):
        self.record["current"].update(phase="blocked", reason=reason)
        self._write()
        return self._view("blocked")

    @staticmethod
    def _fingerprint(stage, config):
        encoded = json.dumps({"stage": stage, "config": config}, sort_keys=True, allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def launch(self, stage, config, spawn, *, resume_reason=None):
        """Reserve and dispatch once, or recover an already recorded invocation.

        ``spawn(stage, config, ticket)`` must return a FunctionCall with object_id.
        The worker must claim the ticket with that actual FunctionCall ID.
        An explicit nonempty resume_reason audits a repaired blocked stage. It
        cannot resolve ambiguous dispatch, release another owner's ticket, or
        bypass a still-running invocation. Watchdog recover never unblocks work.
        """
        if not isinstance(stage, str) or not stage:
            raise ValueError("A driver stage is required")
        if resume_reason is not None and (not isinstance(resume_reason, str) or not resume_reason.strip()):
            raise ValueError("An explicit resume needs a nonempty audit reason")
        fingerprint = self._fingerprint(stage, config)
        with self._locked():
            current = self.record["current"]
            state = self._read(self.state_path)
            if resume_reason is not None and (
                (current and current["phase"] == "blocked")
                or (state.get("status") == "blocked" and (not current or current["phase"] != "complete"))
            ):
                self.record.setdefault("manual_resumes", []).append({
                    "at": self.clock(), "stage": stage, "reason": resume_reason,
                    "previous_state_status": state.get("status"),
                    "previous_state_error": state.get("error") or state.get("reason"),
                    "previous_coordinator_reason": current.get("reason") if current else None,
                    "previous_call_id": current.get("call_id") if current else None,
                })
                self._write()
                return self._resume(stage, config, fingerprint, spawn, resume_reason)
            if current:
                if current["fingerprint"] == fingerprint:
                    return self._recover(spawn)
                if current["phase"] != "complete":
                    # Stage transitions need not wait for the periodic watchdog.
                    # Observe/reconcile only: a different request cannot cancel
                    # or restart the previous owner as a side effect of launch.
                    self._recover(spawn, completion_only=True)
                    current = self.record["current"]
                    if current["phase"] != "complete":
                        raise ValueError("Another driver request still owns this coordinator")
                    state = self._read(self.state_path)
            if not current and state.get("status") == "running":
                return self._view("legacy_running", "Existing driver has no coordinator ticket; left untouched")
            if state.get("status") == "blocked":
                return self._view("blocked", "Pipeline state is blocked; automatic launch disabled")
            return self._start(stage, config, fingerprint, spawn)

    def _resume(self, stage, config, fingerprint, spawn, reason):
        current = self.record["current"]
        if not current:
            return self._start(stage, config, fingerprint, spawn)
        if current["phase"] == "terminal":
            current["manual_resume"] = {"stage": stage, "config": config, "fingerprint": fingerprint, "reason": reason}
            self._write()
            return self._reconcile(spawn)
        if not current.get("call_id"):
            if current.get("ticket") or self._ledger().snapshot()["reservations"].get(current["reservation_id"]):
                return self._view("blocked", "Ambiguous dispatch cannot be resolved by a resume reason")
            return self._start(stage, config, fingerprint, spawn)
        try:
            call = self.call_factory(current["call_id"])
        except Exception:  # noqa: BLE001 — manual resume also needs positive terminal evidence
            return self._view("blocked", "Provider owner could not be checked; manual resume remains blocked")
        outcome, result = self._poll(call)
        if outcome not in {"completed", "failed", "cancelled"}:
            return self._view("blocked", "Previous invocation is not confirmed terminal; no replacement dispatched")
        current["manual_resume"] = {"stage": stage, "config": config, "fingerprint": fingerprint, "reason": reason}
        return self._terminal(outcome, result, spawn)

    def recover(self, spawn):
        """Observe completion or replace a stale owner only after confirmed exit."""
        with self._locked():
            return self._recover(spawn)

    def _start(self, stage, config, fingerprint, spawn):
        old = self.record["current"]
        if old:
            self.record["history"].append(old)
        self.record["current"] = {
            "stage": stage, "config": config, "fingerprint": fingerprint,
            "phase": "preparing", "created_at": self.clock(),
            "reservation_id": "driver-" + uuid.uuid4().hex,
        }
        self._write()
        return self._dispatch(spawn)

    def _dispatch(self, spawn):
        current = self.record["current"]
        try:
            ticket = DriverBudgetLease.prepare(
                self._ledger(), stage=current["stage"], reservation_id=current["reservation_id"],
                commit=self.commit, **self.lease_options,
            )
        except (BudgetExceeded, DriverLeaseError) as error:
            return self._block("Driver reservation refused: " + type(error).__name__)
        # A crash after this commit is an ambiguous POST, even if POST never ran.
        current.update(ticket=ticket, phase="dispatching", dispatched_at=self.clock())
        self._write()
        try:
            call = spawn(current["stage"], current["config"], ticket)
            call_id = call.object_id
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Spawn returned no provider call ID")
        except Exception as error:  # noqa: BLE001 — any failed POST can have an uncertain outcome
            return self._block("Dispatch outcome unknown; reservation retained: " + _failure_kind(error))
        current.update(phase="running", call_id=call_id)
        self._write()
        return self._view("running")

    def _poll(self, call):
        try:
            result = call.get(timeout=0)
            return "completed", result
        except BaseException as error:
            # Modal InputCancellation inherits BaseException rather than Exception.
            if not isinstance(error, Exception) and type(error).__name__ != "InputCancellation":
                raise
            kind = _failure_kind(error)
            result = {"error_type": type(error).__name__}
            if kind == "auth_quota_budget" and _remote_failure(error):
                # The invocation ended, but its failure must never cause an
                # automatic retry. Reconcile its receipt, then block the stage.
                result["failure_category"] = kind
                return "failed", result
            return kind, result

    def _recover(self, spawn, *, completion_only=False):
        current = self.record["current"]
        if not current:
            state = self._read(self.state_path)
            if state.get("status") == "running":
                return self._view("legacy_running", "Existing driver has no coordinator ticket; left untouched")
            return self._view("blocked" if state.get("status") == "blocked" else "idle")
        phase = current["phase"]
        if phase in {"complete", "blocked"}:
            return self._view()
        if phase == "preparing":
            return self._view() if completion_only else self._dispatch(spawn)
        if phase == "dispatching" or not current.get("call_id"):
            return self._block("Dispatch outcome unknown; reservation retained and automatic retry disabled")
        if phase == "terminal":
            return self._reconcile(spawn, allow_restart=not completion_only)
        try:
            call = self.call_factory(current["call_id"])
        except Exception as error:  # noqa: BLE001 — lookup failures cannot establish provider exit
            if _failure_kind(error) == "auth_quota_budget":
                return self._block("Provider access blocked; reservation retained")
            return self._view("pending", "Provider call lookup failed; owner and reservation retained")
        outcome, result = self._poll(call)
        if outcome in {"completed", "failed", "cancelled"}:
            return self._terminal(outcome, result, spawn, allow_restart=not completion_only)
        if outcome == "auth_quota_budget":
            return self._block("Provider authentication, quota or budget blocked; no automatic retry")
        if outcome == "unknown":
            return self._view("pending", "Provider state unknown; owner and reservation retained")
        if completion_only:
            return self._view("running", "Previous driver still owns this coordinator")
        state = self._read(self.state_path)
        if phase == "cancelling":
            # Cancel is idempotent for this exact call ID. An intent may have
            # committed before its RPC was sent, or the RPC response was lost.
            return self._cancel(call, spawn)
        heartbeat = state.get("heartbeat", current["dispatched_at"])
        if not isinstance(heartbeat, (int, float)) or not math.isfinite(heartbeat):
            return self._view("pending", "Invalid heartbeat; cannot establish stale ownership")
        stale = self.clock() - max(heartbeat, current["dispatched_at"]) > self.stale_seconds
        if not stale:
            return self._view("blocked" if state.get("status") == "blocked" else "running")
        # Capture stage and blocked state before cancellation can change state.json.
        current.update(phase="cancelling", stale_stage=state.get("active_stage") or current["stage"],
                       no_restart=state.get("status") == "blocked", cancel_requested_at=self.clock())
        self._write()
        return self._cancel(call, spawn)

    def _cancel(self, call, spawn):
        current = self.record["current"]
        try:
            call.cancel(terminate_containers=True)
        except Exception as error:  # noqa: BLE001 — preserve ownership after every ambiguous cancellation
            current["cancel_error_type"] = type(error).__name__
            self._write()
            if _failure_kind(error) == "auth_quota_budget":
                return self._block("Cancellation blocked by provider access; reservation retained")
            return self._view("pending", "Cancellation outcome unknown; owner and reservation retained")
        outcome, result = self._poll(call)
        if outcome in {"completed", "failed", "cancelled"}:
            return self._terminal(outcome, result, spawn)
        if outcome == "auth_quota_budget":
            return self._block("Provider access blocked after cancellation; reservation retained")
        return self._view("pending", "Cancellation requested; provider exit is not confirmed")

    def _terminal(self, outcome, result, spawn, *, allow_restart=True):
        current = self.record["current"]
        # Save the provider observation before touching reservations. No raw return
        # value is needed for coordination; explicit status is its only contract.
        current.update(phase="terminal", terminal_state=outcome, terminal_at=self.clock(),
                       result_status=result.get("status") if isinstance(result, dict) else None)
        if isinstance(result, dict) and result.get("error_type"):
            current["error_type"] = result["error_type"]
        if isinstance(result, dict) and result.get("failure_category") == "auth_quota_budget":
            current["no_restart"] = True
        self._write()
        return self._reconcile(spawn, allow_restart=allow_restart)

    def _reconcile(self, spawn, *, allow_restart=True):
        current = self.record["current"]
        # Terminal evidence establishes transfer of ledger ownership from worker.
        self.reload()
        ledger = self._ledger()
        row = ledger.snapshot()["reservations"].get(current["reservation_id"])
        if not row:
            return self._block("Missing driver reservation; cannot reconcile provider completion")
        owner = row.get("metadata", {}).get("driver_owner_id")
        if owner and owner != current["call_id"]:
            return self._block("Reservation belongs to a different provider invocation; funds retained")
        receipt = row.get("metadata", {}).get("driver_finish_receipt")
        try:
            if current.get("cancel_requested_at") is not None or receipt is None:
                DriverBudgetLease.settle_worst_case(
                    ledger, current["ticket"], reason="Provider terminal after cancellation or without a worker receipt",
                    commit=self.commit,
                )
            else:
                evidence = {
                    "source": "modal.FunctionCall.get", "provider_call_id": current["call_id"],
                    "provider_terminal_state": current["terminal_state"],
                    "observed_at": datetime.fromtimestamp(current["terminal_at"], UTC).isoformat(),
                }
                elapsed = current["terminal_at"] - current["dispatched_at"]
                if elapsed >= receipt["observed_driver_seconds"]:
                    evidence["elapsed_seconds_bound"] = elapsed
                DriverBudgetLease.settle_completed(
                    ledger, current["ticket"], receipt, evidence=evidence, commit=self.commit,
                )
        except (ValueError, BudgetExceeded, DriverLeaseError) as error:
            return self._block("Driver reconciliation refused: " + type(error).__name__)
        if current.get("manual_resume"):
            if not allow_restart:
                return self._view("pending", "Previous manual resume still owns this coordinator")
            request = current["manual_resume"]
            return self._start(request["stage"], request["config"], request["fingerprint"], spawn)
        state = self._read(self.state_path)
        if current.get("no_restart") or state.get("status") == "blocked" or current.get("result_status") == "blocked":
            return self._block("Pipeline blocked; previous owner reconciled and automatic restart disabled")
        done = {"complete", "completed", "done", "stage_done"}
        if (current["terminal_state"] == "completed"
                and (current.get("result_status") in done or state.get("status") in done)):
            current.update(phase="complete", completed_at=self.clock())
            self._write()
            return self._view("complete")
        if not allow_restart:
            return self._view("pending", "Previous driver ended without completing its stage")
        stage = current.get("stale_stage") or state.get("active_stage") or current["stage"]
        attempts = self.record["restart_counts"].get(stage, 0)
        if attempts >= self.max_restarts:
            return self._block(f"Automatic restart limit reached for stage {stage}")
        # Count and replacement intent are committed together by _start, making a
        # crash recover the same preparing ticket rather than count/dispatch twice.
        self.record["restart_counts"][stage] = attempts + 1
        return self._start(current["stage"], current["config"], current["fingerprint"], spawn)
