"""Renew a sandbox's bounded spend reservation before renewing its idle timer.

This is a controller-side accounting bound, not a Modal invoice. It relies on an
idle_timeout of at most 180 seconds and no exec activity longer than 55 seconds.
The transport must call before_activity before every idle-renewing operation.
An unresponsive controller leaves the last reservation in place; Modal's idle
timeout stops the sandbox. Never attach an unbounded filesystem exec helper.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from ..budget import BudgetLedger, atomic_json, compute_cost

GIB = 1024 ** 3


class SandboxLeaseError(RuntimeError):
    """The sandbox must not receive further activity under this lease."""


class SandboxBudgetLease:
    def __init__(self, ledger: BudgetLedger, state_path, *, stage="sandbox", request_gib=8,
                 limit_gib=96, cpu_cores=2, lease_seconds=300, heartbeat_interval=60,
                 idle_timeout=180, max_exec_seconds=55, background=True, on_error=None,
                 clock=time.monotonic, commit=lambda: None):
        if not 0 < request_gib <= limit_gib <= 96 or not 0 < cpu_cores <= 2:
            raise ValueError("Sandbox resources exceed the documented protocol limits")
        if not 0 < heartbeat_interval <= 60 or not 0 < idle_timeout <= 180 or not 0 < max_exec_seconds <= 55:
            raise ValueError("Lease requires heartbeat<=60s, idle<=180s, bounded exec<=55s")
        if not 300 <= lease_seconds or heartbeat_interval + idle_timeout + max_exec_seconds >= lease_seconds:
            raise ValueError("Lease must cover heartbeat spacing, bounded exec, and subsequent idle expiry")
        self.ledger, self.path = ledger, Path(state_path)
        self.stage, self.request_gib, self.limit_gib, self.cpu_cores = stage, request_gib, limit_gib, cpu_cores
        self.lease_seconds, self.heartbeat_interval = lease_seconds, heartbeat_interval
        self.idle_timeout, self.max_exec_seconds = idle_timeout, max_exec_seconds
        self.background, self.on_error, self.clock = background, on_error, clock
        self.commit = commit
        self.transport = None
        self.last_status = None
        self.failure = None
        self._termination_error_type = None
        self._current = None
        self._pending = None
        self._state = "new"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._internal_owner = None
        self._largest_peak = 0
        self._trust_lost = False
        self._orphaned = []
        self._prior_journal = None
        self._sandbox_id = None
        self._reconciled = []
        if self.path.exists():
            previous = json.loads(self.path.read_text())
            self._prior_journal = previous
            self._orphaned = previous.get("orphaned_reservations", [])
            # A resumed driver never erases reservations from an uncertain prior
            # controller. Its old sandbox expires independently via idle_timeout.
            if previous.get("state") not in ("closed", "new"):
                self._orphaned += [r["id"] for r in previous.get("reservations", [])]

    def _cost(self, seconds, memory_gib=None):
        return compute_cost(seconds, cpu_cores=self.cpu_cores,
                            memory_gib=self.limit_gib if memory_gib is None else memory_gib, sandbox=True)

    def _reserve(self, phase):
        start = self.clock()
        rid = self.ledger.reserve("modal", self._cost(self.lease_seconds), f"{self.stage}:{phase}",
                                  reservation_id=uuid.uuid4().hex,
                                  metadata={"kind": "sandbox_resource_upper_bound", "invoice": False,
                                            "seconds_bound": self.lease_seconds,
                                            "memory_gib_bound": self.limit_gib, "cpu_cores_bound": self.cpu_cores,
                                            "idle_timeout": self.idle_timeout,
                                            "max_exec_seconds": self.max_exec_seconds})
        return {"id": rid, "start": start, "wall_started_at": time.time(), "phase": phase}

    def _persist(self):
        reservations = [dict(row) for row in (self._current, self._pending) if row]
        atomic_json(self.path, {"schema_version": 1, "state": self._state,
                               "sandbox_id": getattr(self.transport, "sandbox_id", self._sandbox_id),
                               "reservations": reservations, "orphaned_reservations": sorted(set(self._orphaned)),
                               "reconciled_prior_reservations": self._reconciled,
                               "last_status_usage": (self.last_status or {}).get("usage"),
                               "failure_type": type(self.failure).__name__ if self.failure else None,
                               "termination_error_type": self._termination_error_type,
                               "billing_basis": "resource_upper_bound_not_invoice"})
        self.commit()  # publish reservations and intent before any remote activity

    def start(self, create_callable):
        """Reserve before creation; factory must configure idle_timeout<=180.

        Returns the transport, whose before_activity callback now enforces the
        lease. The factory must use bounded startup and no unbounded background
        exec commands. It must not renew stdin before returning the transport.
        """
        with self._lock:
            if self._state != "new":
                raise SandboxLeaseError("A lease can create only one sandbox")
            self._reconcile_prior()
            self._current = self._reserve("startup")
            self._state = "starting"
            self._persist()
            try:
                self.transport = create_callable()
                self.transport.before_activity = self.check
                self._state = "active"
                self.heartbeat()
            except Exception as error:
                self._abort(error)
                raise
            if self.background:
                self._thread = threading.Thread(target=self._loop, daemon=True, name="openwsp-sandbox-budget")
                self._thread.start()
            return self.transport

    def _reconcile_prior(self):
        """Charge unresolved earlier horizons at their full recorded upper bound.

        Cross-process monotonic clocks cannot reconstruct earlier intervals.
        The ledger therefore keeps the conservative full amount; it never
        silently releases it or labels this as an observed provider charge.
        """
        rows = self.ledger.snapshot()["reservations"]
        for rid in sorted(set(self._orphaned)):
            if rid not in rows:
                raise SandboxLeaseError("Prior sandbox reservation is missing from this ledger")
            row = rows[rid]
            if row["kind"] != "modal" or row.get("metadata", {}).get("kind") != "sandbox_resource_upper_bound":
                raise SandboxLeaseError("Prior journal references a non-sandbox reservation")
            if row["status"] == "reserved":
                self.ledger.settle(rid, row["upper_bound_usd"],
                                   seconds=row.get("metadata", {}).get("seconds_bound"),
                                   metadata={"billing_basis": "unresolved_reservation_upper_bound_not_invoice",
                                             "invoice": False, "reconciliation": "controller_restart",
                                             "may_overlap_later_reservations": True,
                                             "sandbox_id": (self._prior_journal or {}).get("sandbox_id")})
            self._reconciled.append(rid)
        self._orphaned = []

    def attach(self, attach_callable, *, sandbox_id=None):
        """Reattach a journaled sandbox under a newly funded idle horizon.

        ``attach_callable(id)`` must attach without probing; use
        ``ModalSandboxTransport.from_id(id, probe=False)``. One controller must
        own the sandbox; the stage watchdog is responsible for excluding a
        concurrently running old driver. Prior uncertain intervals are charged
        at their full bounds and visibly marked as such, never charged twice.
        """
        with self._lock:
            if self._state != "new":
                raise SandboxLeaseError("A lease can attach only once")
            prior_id = (self._prior_journal or {}).get("sandbox_id")
            target = sandbox_id or prior_id
            if not target or target != prior_id:
                raise SandboxLeaseError("Reattachment requires the same sandbox ID recorded in the lease journal")
            self._reconcile_prior()
            self._sandbox_id = target
            self._current = self._reserve("reattach")
            self._state = "starting"
            self._persist()
            try:
                candidate = attach_callable(target)
                if getattr(candidate, "sandbox_id", None) != target:
                    raise SandboxLeaseError("Attach factory returned a different sandbox")
                self.transport = candidate
                self.transport.before_activity = self.check
                self._state = "active"
                self.heartbeat()
            except Exception as error:
                self._abort(error)
                raise
            if self.background:
                self._thread = threading.Thread(target=self._loop, daemon=True, name="openwsp-sandbox-budget")
                self._thread.start()
            return self.transport

    def check(self, operation="stdin", timeout_seconds=None):
        """Transport guard, called before (never inside) its I/O mutex.

        Unknown exec durations, including Modal filesystem helpers, are refused.
        Only the lease controller may issue its renewal/termination operations
        while it owns the controller lock.
        """
        with self._lock:
            if operation not in ("stdin", "exec", "filesystem"):
                raise SandboxLeaseError("Unknown sandbox activity cannot be budgeted")
            if operation in ("exec", "filesystem") and (
                timeout_seconds is None or not 0 < timeout_seconds <= self.max_exec_seconds
            ):
                raise SandboxLeaseError("Lease forbids unbounded sandbox exec/filesystem activity")
            if self._internal_owner == threading.get_ident():
                return
            if self.failure:
                raise SandboxLeaseError("Sandbox lease stopped after a controller failure") from self.failure
            if self._state != "active":
                raise SandboxLeaseError(f"Sandbox lease is {self._state}")
            if self.clock() - self._current["start"] >= self.heartbeat_interval:
                self.heartbeat()

    def _memory_bound(self, status):
        usage = status.get("usage") or {}
        peak, limit = usage.get("cgroup_memory_peak_bytes"), usage.get("cgroup_memory_limit_bytes")
        verified = usage.get("cgroup_memory_scope_verified") is True
        if not verified or self._trust_lost:
            return self.limit_gib, "configured_hard_limit"
        if (not isinstance(peak, int) or isinstance(peak, bool) or peak < 0
                or limit != int(self.limit_gib * GIB) or peak > limit):
            self._trust_lost = True
            return self.limit_gib, "invalid_cgroup_observation_hard_limit"
        if peak < self._largest_peak:
            self._trust_lost = True
            return self.limit_gib, "nonmonotonic_cgroup_peak_hard_limit"
        self._largest_peak = peak
        return max(self.request_gib, peak / GIB), "verified_lifetime_cgroup_peak"

    def _settle(self, row, end, memory_gib, basis):
        elapsed = max(0, end - row["start"])
        if elapsed > self.lease_seconds:
            raise SandboxLeaseError("Reservation horizon elapsed; retain its full uncertain bound")
        self.ledger.settle(row["id"], self._cost(elapsed, memory_gib), seconds=elapsed,
                           metadata={"billing_basis": "resource_upper_bound_not_invoice", "invoice": False,
                                     "memory_gib_bound": memory_gib, "memory_basis": basis,
                                     "cpu_cores_bound": self.cpu_cores,
                                     "sandbox_id": getattr(self.transport, "sandbox_id", None)})

    def heartbeat(self):
        """Reserve the next horizon BEFORE stdin renews idle expiry.

        After status observes the lifetime peak, settle the previous interval at
        that peak. Partition time at the start of this RPC: observations happen
        later, so they cover the whole settled interval, including network gaps.
        """
        with self._lock:
            if self._state != "active" or self.failure:
                raise SandboxLeaseError("Cannot renew an inactive or failed sandbox lease")
            try:
                if self.clock() - self._current["start"] > self.lease_seconds:
                    raise SandboxLeaseError("Sandbox lease expired; no further idle renewal is allowed")
                self._pending = self._reserve("heartbeat")
                self._persist()
                self._internal_owner = threading.get_ident()
                try:
                    status = self.transport.status()
                finally:
                    self._internal_owner = None
                if not isinstance(status, dict) or status.get("ok") is False:
                    raise SandboxLeaseError("Sandbox did not provide a successful status response")
                if self.clock() - self._pending["start"] > self.lease_seconds:
                    raise SandboxLeaseError("Status exceeded the reserved horizon")
                memory, basis = self._memory_bound(status)
                self._settle(self._current, self._pending["start"], memory, basis)
                self._current, self._pending = self._pending, None
                self.last_status = status
                self._persist()
                return status
            except Exception as error:
                self._abort(error)
                raise

    def _abort(self, error):
        self.failure = error
        self._state = "uncertain"
        self._stop.set()
        # Never call close(): its volume sync is new exec activity. Direct
        # termination neither renews idle nor requires an additional reservation.
        if self.transport is not None:
            try:
                self.transport.sandbox.terminate(wait=True)
            except Exception as termination_error:  # noqa: BLE001 - uncertain termination retains reservations
                self._termination_error_type = type(termination_error).__name__
        self._persist()

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                remaining = self.heartbeat_interval - (self.clock() - self._current["start"])
            if self._stop.wait(max(0, remaining)):
                break
            try:
                self.heartbeat()
            except Exception as error:  # noqa: BLE001 - communicate background failure to owner/check()
                if self.on_error:
                    self.on_error(error)
                break

    def close(self):
        """Observe the last interval, then terminate with a 96 GiB tail bound.

        If renewal/termination is uncertain, retain the full reservation(s).
        The final transport close may execute sync for at most 55 seconds.
        """
        self._stop.set()
        with self._lock:
            if self._state in ("closed", "new"):
                return
            if self.failure:
                return  # already aborted; never release its uncertain reservations
            try:
                self.heartbeat()
                if self.clock() - self._current["start"] > self.heartbeat_interval:
                    raise SandboxLeaseError("Final status was too slow to permit a bounded sync")
                self._state = "closing"
                self._internal_owner = threading.get_ident()
                try:
                    self.transport.close()
                finally:
                    self._internal_owner = None
                self._settle(self._current, self.clock(), self.limit_gib, "termination_tail_hard_limit")
                self._current = None
                self._state = "closed"
                self._persist()
            except Exception as error:
                self._abort(error)
                raise
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
