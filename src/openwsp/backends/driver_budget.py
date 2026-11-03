"""Pre-dispatch CPU-driver reservations with provider-enforced runtime bounds.

A periodic heartbeat cannot bound a function with a 24-hour hard timeout: a
stalled controller can still incur that full cost. Reserve the whole timeout,
plus startup/scaledown margin, before dispatch; settle unused time after exit.
Use cpu=(1, 1), memory=(4096, 4096), timeout=86400, startup_timeout<=120, and
scaledown_window<=5 in the actual function, or configure explicit margins here.
"""
from __future__ import annotations

import math
import time
import uuid
from decimal import Decimal

from ..budget import BudgetLedger, compute_cost


class DriverLeaseError(RuntimeError):
    pass


class DriverBudgetLease:
    @staticmethod
    def prepare(ledger: BudgetLedger, *, stage="driver", timeout_seconds=86_400,
                startup_seconds=120, scaledown_seconds=5, cpu_cores=1, memory_gib=4,
                reservation_id=None, commit=lambda: None):
        """Return a JSON ticket only after its full cost is durably reserved.

        Persist/commit the ledger before spawning. Pass the ticket to exactly one
        invocation; a retried/preempted invocation needs a new ticket unless the
        provider proves it is the same running invocation.
        """
        if not 0 < timeout_seconds <= 86_400 or not 0 <= startup_seconds <= 3600:
            raise ValueError("Invalid driver runtime/startup bound")
        if not 0 <= scaledown_seconds <= 3600 or not 0 < cpu_cores <= 1 or not 0 < memory_gib <= 4:
            raise ValueError("Invalid driver resources or scaledown bound")
        rid = reservation_id or uuid.uuid4().hex
        ticket = {"schema_version": 1, "reservation_id": rid, "stage": stage,
                  "timeout_seconds": timeout_seconds, "startup_seconds": startup_seconds,
                  "scaledown_seconds": scaledown_seconds, "cpu_cores": cpu_cores,
                  "memory_gib": memory_gib, "prepared_at": time.time()}
        seconds = timeout_seconds + startup_seconds + scaledown_seconds
        cost = compute_cost(seconds, cpu_cores=cpu_cores, memory_gib=memory_gib)
        ticket["upper_bound_usd"] = str(cost)
        existing = ledger.snapshot()["reservations"].get(rid)
        if existing:
            old = existing.get("metadata", {}).get("driver_ticket")
            if (existing["status"] != "reserved" or existing["metadata"].get("driver_owner_id")
                    or old is None or any(old.get(k) != value for k, value in ticket.items() if k != "prepared_at")):
                raise DriverLeaseError("Driver reservation already belongs to claimed or different work")
            commit()
            return old
        ledger.reserve("modal", cost, stage, reservation_id=rid,
                       metadata={"kind": "driver_resource_upper_bound", "invoice": False,
                                 "driver_ticket": ticket})
        commit()
        return ticket

    @classmethod
    def claim(cls, ledger: BudgetLedger, ticket, *, owner_id, clock=time.monotonic, commit=lambda: None):
        """Claim inside the invocation with its unique provider call/attempt ID."""
        if not owner_id:
            raise ValueError("A driver claim requires a provider invocation ID")
        started_at = clock()
        rid = ticket["reservation_id"]
        # The same lock protects ownership and reservations on the single driver
        # filesystem. A Volume is not a distributed lock; watchdog ownership is
        # additionally required when controllers run on different containers.
        with ledger._locked() as state:
            row = state["reservations"].get(rid)
            if row is None or row["status"] != "reserved" or row["kind"] != "modal":
                raise DriverLeaseError("Driver ticket has no active pre-dispatch reservation")
            if row.get("metadata", {}).get("driver_ticket") != ticket:
                raise DriverLeaseError("Driver ticket differs from its reserved resource/time bounds")
            metadata = row["metadata"]
            if metadata.get("driver_owner_id"):
                raise DriverLeaseError("A driver reservation cannot be claimed twice; resume the existing owner")
            metadata["driver_owner_id"] = str(owner_id)
            metadata["driver_claimed_at"] = time.time()
        commit()
        instance = cls()
        instance.ledger, instance.ticket, instance.owner_id = ledger, dict(ticket), str(owner_id)
        instance.clock, instance.started_at = clock, started_at
        instance.commit = commit
        instance._receipt = None
        return instance

    def finish(self, *, elapsed_seconds=None):
        """Publish an elapsed-time receipt; retain funds until provider completion.

        The worker is still running while it publishes this receipt. Its full
        reservation must remain in place in case that publication or its exit
        stalls. The launcher/watchdog uses settle_completed after observing the
        provider's terminal call state, or settle_worst_case after a lost owner.
        """
        if self._receipt is not None:
            return self._receipt
        elapsed = self.clock() - self.started_at if elapsed_seconds is None else elapsed_seconds
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Driver elapsed time must be finite and nonnegative")
        total = elapsed + self.ticket["startup_seconds"] + self.ticket["scaledown_seconds"]
        cost = compute_cost(total, cpu_cores=self.ticket["cpu_cores"], memory_gib=self.ticket["memory_gib"])
        if elapsed > self.ticket["timeout_seconds"]:
            raise DriverLeaseError("Observed driver time exceeds its provider timeout; retain reservation")
        receipt = {"reservation_id": self.ticket["reservation_id"], "owner_id": self.owner_id,
                   "reservation_upper_bound_usd": self.ticket["upper_bound_usd"],
                   "observed_subtotal_usd": str(cost), "observed_driver_seconds": elapsed,
                   "receipt_status": "awaiting_provider_completion",
                   "billing_basis": "resource_upper_bound_not_invoice"}
        with self.ledger._locked() as state:
            row = state["reservations"][self.ticket["reservation_id"]]
            if row["status"] != "reserved" or row["metadata"].get("driver_owner_id") != self.owner_id:
                raise DriverLeaseError("Driver reservation is no longer owned by this invocation")
            row["metadata"]["driver_finish_receipt"] = receipt
        self.commit()
        self._receipt = receipt
        return self._receipt

    @staticmethod
    def settle_completed(ledger: BudgetLedger, ticket, receipt, *, evidence, commit=lambda: None):
        """Settle only after the provider confirms the invocation is terminal.

        Evidence needs source, observed_at, provider_call_id matching the claimed
        owner, and provider_terminal_state. An
        optional elapsed_seconds_bound must cover the entire dispatched call,
        including receipt publication; add startup/scaledown margins here. If
        no trustworthy duration exists, charge the full reservation upper bound.
        """
        if (not evidence.get("source") or not evidence.get("observed_at")
                or evidence.get("provider_terminal_state") not in ("completed", "failed", "cancelled", "terminated")):
            raise ValueError("Provider terminal-state evidence is required before releasing driver funds")
        if evidence.get("provider_call_id") != receipt.get("owner_id"):
            raise ValueError("Completion evidence must identify this driver invocation")
        rid = ticket["reservation_id"]
        row = ledger.snapshot()["reservations"].get(rid)
        if (row is None or row.get("metadata", {}).get("driver_ticket") != ticket
                or row["metadata"].get("driver_finish_receipt") != receipt):
            raise DriverLeaseError("Unknown or mismatched driver completion receipt")
        if row["status"] == "settled":
            return Decimal(row["actual_usd"])
        full_seconds = ticket["timeout_seconds"] + ticket["startup_seconds"] + ticket["scaledown_seconds"]
        elapsed = evidence.get("elapsed_seconds_bound")
        if elapsed is None:
            seconds, cost = full_seconds, Decimal(row["upper_bound_usd"])
        else:
            if not math.isfinite(elapsed) or elapsed < receipt["observed_driver_seconds"]:
                raise ValueError("Completion duration must cover the worker's observed elapsed time")
            seconds = min(full_seconds, elapsed + ticket["startup_seconds"] + ticket["scaledown_seconds"])
            cost = compute_cost(seconds, cpu_cores=ticket["cpu_cores"], memory_gib=ticket["memory_gib"])
        ledger.settle(rid, cost, seconds=seconds,
                       metadata={"billing_basis": "driver_resource_upper_bound_not_invoice", "invoice": False,
                                 "provider_completion_evidence": evidence, "worker_receipt": receipt})
        commit()
        return cost

    @staticmethod
    def settle_worst_case(ledger: BudgetLedger, ticket, *, reason, commit=lambda: None):
        """After a lost/preempted owner, retain its whole bound as estimated spend."""
        if not reason:
            raise ValueError("A lost driver receipt needs a reconciliation reason")
        rid = ticket["reservation_id"]
        row = ledger.snapshot()["reservations"].get(rid)
        if row is None or row.get("metadata", {}).get("driver_ticket") != ticket:
            raise DriverLeaseError("Unknown or mismatched driver ticket")
        if row["status"] == "settled":
            return Decimal(row["actual_usd"])
        ledger.settle(rid, row["upper_bound_usd"],
                       seconds=ticket["timeout_seconds"] + ticket["startup_seconds"] + ticket["scaledown_seconds"],
                       metadata={"billing_basis": "unresolved_driver_upper_bound_not_invoice", "invoice": False,
                                 "reconciliation_reason": str(reason)})
        commit()
        return Decimal(row["upper_bound_usd"])
