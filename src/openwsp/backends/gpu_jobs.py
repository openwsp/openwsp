"""Budgeted GPU dispatch with an atomic queue-timeout fallback.

One controller owns this journal and its RemoteJobRunner children. Workers must
claim the supplied start-gate key before expensive initialization. A cancellation
winner never launches a fallback until an actual provider GET confirms the old
call is terminal. The cancelled attempt is charged its full resource bound; these
resource estimates are not provider invoices.
"""

from __future__ import annotations

import fcntl
import json
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from openwsp.backends.jobs import (
    JobBlocked,
    JobPending,
    RemoteJobRunner,
    _canonical,
    _classification,
    _digest,
    _duration,
)
from openwsp.backends.start_gate import StartGate
from openwsp.budget import atomic_json, compute_cost


def _terminal_cancel_error(error: BaseException) -> bool:
    """Recognize terminal results, never transport cancellation or GET timeout.

    Modal 1.6.3 deserializes remote failures or raises RemoteError from a returned
    GenericResult. It has no FunctionCancelledError. InputCancellation is a
    BaseException, so it must be distinguished from process interrupts.
    """
    name = type(error).__name__
    module = type(error).__module__
    if module == "modal.exception" and name in {"RemoteError", "InputCancellation", "FunctionTimeoutError"}:
        return True
    return type(error) is RuntimeError and str(error) == "Queued attempt did not win its start gate: cancelled"


class GPUJobRunner:
    """Wrap a RemoteJobRunner with one H100-to-A100-80GB fallback per attempt.

    ``spawn_for(gpu, gate_key)`` submits one bounded call with that key as a class
    parameter. Repeat ``launch`` with the same fingerprint after controller
    restart to reconstruct this callable; persisted call IDs prevent duplicates.
    The remote worker must enforce the configured resource and timeout maxima,
    use no automatic retries, and stop immediately if its start claim loses.
    """

    def __init__(
        self, ledger, directory, gate, commit=lambda: None, *, call_factory=None,
        clock=None, sleep=None, timeout_seconds=1800, startup_seconds=600,
        margin_seconds=120, settlement_startup_seconds=120, scaledown_seconds=2,
        cpu_cores=4, memory_gib=16, queue_timeout_seconds=600, poll_seconds=30,
    ):
        self.ledger = ledger
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep
        self.gate = gate if isinstance(gate, StartGate) else StartGate(gate, clock=self.clock)
        self.resources = {
            name: _duration(value, name) for name, value in {
                "timeout_seconds": timeout_seconds, "startup_seconds": startup_seconds,
                "margin_seconds": margin_seconds, "settlement_startup_seconds": settlement_startup_seconds,
                "scaledown_seconds": scaledown_seconds, "cpu_cores": cpu_cores, "memory_gib": memory_gib,
                "queue_timeout_seconds": queue_timeout_seconds, "poll_seconds": poll_seconds,
            }.items()
        }
        if min(timeout_seconds, poll_seconds, cpu_cores, memory_gib) <= 0:
            raise ValueError("Timeout, poll interval, CPU and memory bounds must be positive")
        if settlement_startup_seconds > startup_seconds or scaledown_seconds > margin_seconds:
            raise ValueError("Settlement allowances must fit the full reservation bounds")
        self.remote = RemoteJobRunner(
            ledger, self.directory / "attempts", commit=commit,
            call_factory=call_factory, clock=self.clock, sleep=self.sleep,
        )
        self._factories = {}

    @contextmanager
    def _locked(self, key):
        if not isinstance(key, str) or not key:
            raise ValueError("job key must be a nonempty string")
        path = self.directory / (_digest(key) + ".gpu.json")
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield path

    def _persist(self, path, record):
        atomic_json(path, record)
        self.commit()

    @staticmethod
    def _read(path):
        record = json.loads(path.read_text())
        if record.get("schema_version") != 1:
            raise JobBlocked("Unsupported GPU job journal version")
        return record

    def _block(self, path, record, category, *, phase, error=None):
        record.update(status="blocked", failure={"category": category, "phase": phase})
        if error is not None:
            record["failure"]["exception_type"] = type(error).__name__
        self._persist(path, record)
        raise JobBlocked(f"GPU job {record['key']} is blocked ({category}); its funds remain accounted") from None

    def _new_attempt(self, record, gpu):
        index = len(record["attempts"])
        gate_key = f"openwsp-gpu:{record['run_id']}:{index}"
        return {
            "index": index, "gpu": gpu, "gate_key": gate_key,
            "child_key": record["key"] + "/" + gate_key,
            "status": "planned", "created_at": self.clock(),
        }

    def _launch_attempt(self, path, record):
        attempt = record["attempts"][-1]
        if attempt.get("call_id"):
            # Retry a failed Volume commit before returning a known remote ID.
            self._persist(path, record)
            return attempt["call_id"]
        resources = record["resources"]
        upper = compute_cost(
            resources["timeout_seconds"] + resources["startup_seconds"] + resources["margin_seconds"],
            gpu=attempt["gpu"], cpu_cores=resources["cpu_cores"], memory_gib=resources["memory_gib"],
        )
        options = {
            "gpu": attempt["gpu"], "cpu_cores": resources["cpu_cores"], "memory_gib": resources["memory_gib"],
            "startup_seconds": resources["settlement_startup_seconds"],
            "scaledown_seconds": resources["scaledown_seconds"],
        }
        factory = self._factories[record["key"]]
        call_id = self.remote.launch(
            attempt["child_key"], lambda: factory(attempt["gpu"], attempt["gate_key"]),
            upper_bound_usd=upper, cost_options=options,
            fingerprint={"parent": record["fingerprint_sha256"], "gpu": attempt["gpu"],
                         "gate_key": attempt["gate_key"], "resources": resources},
        )
        # Recover the original dispatch timestamp if the controller died after
        # the child saved its call ID but before this wrapper saved the mapping.
        child = self.remote._read(self.remote._path(attempt["child_key"]))
        attempt.update(call_id=call_id, submitted_at=child["submitted_at"], status="pending",
                       upper_bound_usd=str(upper), reservation_id=child["reservation_id"])
        record["status"] = "active"
        self._persist(path, record)
        return call_id

    def launch(self, key, spawn_for, *, fingerprint, initial_gpu="H100") -> str:
        if initial_gpu not in {"H100", "A100-80GB"}:
            raise ValueError("initial_gpu must be H100 or A100-80GB")
        if not callable(spawn_for):
            raise TypeError("spawn_for must submit a bounded job for (gpu, gate_key)")
        fingerprint_hash = _digest(_canonical(fingerprint))
        with self._locked(key) as path:
            if path.exists():
                record = self._read(path)
                if (record["key"] != key or record["fingerprint_sha256"] != fingerprint_hash
                        or record["initial_gpu"] != initial_gpu or record["resources"] != self.resources):
                    raise ValueError("GPU job key already belongs to different inputs or resource bounds")
                if record["status"] == "blocked":
                    raise JobBlocked("GPU job is blocked; automatic replacement is disabled")
            else:
                record = {
                    "schema_version": 1, "key": key, "fingerprint_sha256": fingerprint_hash,
                    "run_id": uuid.uuid4().hex, "initial_gpu": initial_gpu, "resources": self.resources,
                    "created_at": self.clock(), "status": "dispatching", "attempts": [],
                }
                record["attempts"].append(self._new_attempt(record, initial_gpu))
                self._persist(path, record)
            self._factories[key] = spawn_for
            return self._launch_attempt(path, record)

    def has_job(self, key) -> bool:
        """Whether this wrapper has a dispatch journal, including blocked intents."""
        with self._locked(key) as path:
            return path.exists()

    def active_child_record(self, key) -> dict:
        """Read the last child's journal, including its original failure category.

        A planned attempt can lack a child journal when its reservation was
        refused. In that case its returned status is ``planned`` and it has no
        reservation ID. Wrapper context is separate from child fields.
        """
        with self._locked(key) as path:
            if not path.exists():
                raise KeyError(f"No recorded GPU job: {key}")
            record = self._read(path)
            attempt = record["attempts"][-1]
            with self.remote._locked(attempt["child_key"]) as child_path:
                child = (self.remote._read(child_path) if child_path.exists()
                         else {"key": attempt["child_key"], "status": "planned"})
            child["_openwsp_gpu"] = {
                "actual_gpu": attempt["gpu"], "attempt_count": len(record["attempts"]),
                "wrapper_status": record["status"], "attempt_status": attempt["status"],
                "wrapper_failure": record.get("failure"),
            }
            return child

    def _cancel(self, path, record):
        attempt = record["attempts"][-1]
        # Recheck before every request, including crash recovery. A gate that
        # expired or became unreadable cannot authorize killing a started worker.
        if not self.gate.observe(attempt["gate_key"]).may_cancel:
            self._block(path, record, "unknown_cancellation_authority", phase="cancel")
        attempt.update(status="cancel_intent", cancel_requested_at=self.clock())
        self._persist(path, record)
        try:
            self.remote.call_factory(attempt["call_id"]).cancel(terminate_containers=True)
        except Exception as exc:  # noqa: BLE001 — provider failures must never cause a fallback dispatch.
            category = _classification(exc)
            if category in {"authentication", "quota"}:
                self._block(path, record, category, phase="cancel", error=exc)
            attempt.update(status="cancel_uncertain", cancel_error_type=type(exc).__name__)
            self._persist(path, record)
            raise JobPending("Cancellation acknowledgement is uncertain; resume GET of the existing call") from None
        attempt["status"] = "cancel_requested"
        self._persist(path, record)

    def _confirm_cancel(self, path, record, wait_seconds):
        attempt = record["attempts"][-1]
        try:
            response = self.remote.call_factory(attempt["call_id"]).get(timeout=wait_seconds)
        except BaseException as exc:  # Modal InputCancellation is a BaseException; process interrupts propagate.
            if not isinstance(exc, Exception) and not _terminal_cancel_error(exc):
                raise
            category = _classification(exc)
            if category in {"authentication", "quota"}:
                self._block(path, record, category, phase="cancel_confirmation", error=exc)
            if _terminal_cancel_error(exc):
                evidence = {"outcome": "terminal_exception", "exception_type": type(exc).__name__}
            elif category == "transient_get":
                attempt["last_confirmation_error"] = {"exception_type": type(exc).__name__, "at": self.clock()}
                self._persist(path, record)
                return False
            else:
                self._block(path, record, "uncertain_terminal_state", phase="cancel_confirmation", error=exc)
        else:
            evidence = {"outcome": "returned_result", "result_type": type(response).__name__}
        attempt.update(status="terminal_cancelled", terminal_evidence={
            "source": "FunctionCall.get", "call_id": attempt["call_id"], "observed_at": self.clock(), **evidence,
        })
        self._persist(path, record)
        return True

    def _settle_cancelled(self, path, record):
        attempt = record["attempts"][-1]
        if not attempt.get("terminal_evidence"):
            raise JobBlocked("Cancellation lacks actual provider terminal evidence")
        reservation = self.ledger.snapshot()["reservations"][attempt["reservation_id"]]
        upper = Decimal(attempt["upper_bound_usd"])
        if reservation["status"] == "reserved":
            bound_seconds = sum(record["resources"][name] for name in (
                "timeout_seconds", "startup_seconds", "margin_seconds"))
            self.ledger.settle(attempt["reservation_id"], upper, seconds=bound_seconds, metadata={
                "source": "cancelled_gpu_resource_upper_bound_not_invoice",
                "terminal_evidence": attempt["terminal_evidence"], "key": attempt["child_key"],
            })
            self.commit()
        elif reservation["status"] != "settled" or Decimal(reservation["actual_usd"]) != upper:
            self._block(path, record, "conflicting_cancelled_cost", phase="settle_cancelled")
        attempt["status"] = "cancelled_settled"
        self._persist(path, record)

    def _response(self, record, response):
        reservations = self.ledger.snapshot()["reservations"]
        total_cost, total_seconds = Decimal(0), 0.0
        for attempt in record["attempts"]:
            row = reservations[attempt["reservation_id"]]
            if row["status"] != "settled":
                raise JobBlocked("A completed GPU job still has an unsettled attempt")
            total_cost += Decimal(row["actual_usd"])
            seconds = row.get("seconds")
            # Resume journals created before cancellation receipts included
            # seconds; their full bound was already charged and remains valid.
            if seconds is None and attempt["status"] == "cancelled_settled":
                seconds = sum(record["resources"][name] for name in (
                    "timeout_seconds", "startup_seconds", "margin_seconds"))
            if seconds is None:
                raise JobBlocked("GPU receipt lacks an accounted duration")
            total_seconds += seconds
        return {**response, "_openwsp_gpu": {
            "actual_gpu": record["attempts"][-1]["gpu"], "attempt_count": len(record["attempts"]),
            "total_settled_cost_usd": float(total_cost), "billable_seconds": total_seconds,
            "cost_basis": "resource_estimate_not_provider_invoice",
            "duration_basis": "container_seconds_plus_margins_or_cancelled_full_bound",
        }}

    def result(self, key, timeout=None) -> dict:
        if key not in self._factories:
            raise JobBlocked("Call launch with the original spawn_for and fingerprint before resuming result")
        if timeout is None:
            timeout = 2 * sum(self.resources[name] for name in (
                "timeout_seconds", "startup_seconds", "margin_seconds", "queue_timeout_seconds"))
        deadline = self.clock() + _duration(timeout, "timeout")
        with self._locked(key) as path:
            record = self._read(path)
            while True:
                if record["status"] == "blocked":
                    raise JobBlocked("GPU job is blocked; automatic replacement is disabled")
                attempt = record["attempts"][-1]
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise JobPending("GPU job remains pending; resume its recorded attempt")
                wait_seconds = min(remaining, record["resources"]["poll_seconds"])
                if attempt["status"] == "cancel_intent":
                    # The process may have died after committing intent but
                    # before the RPC. Repeating cancel for this owned attempt is
                    # safe; an actual terminal GET is still required afterward.
                    self._cancel(path, record)
                elif attempt["status"] == "cancel_uncertain" and not self._confirm_cancel(path, record, 0):
                    # Prefer an already available terminal result. A missing
                    # acknowledgement may also mean the RPC never arrived.
                    self._cancel(path, record)
                if attempt["status"] == "cancel_requested":
                    started = self.clock()
                    if not self._confirm_cancel(path, record, wait_seconds):
                        # A provider may fail instantly; avoid busy polling while
                        # keeping each read/heartbeat interval bounded.
                        self.sleep(min(max(0.01, wait_seconds - (self.clock() - started)),
                                       max(0.01, deadline - self.clock())))
                        continue
                if attempt["status"] in {"terminal_cancelled", "cancelled_settled"}:
                    self._settle_cancelled(path, record)
                    if attempt["gpu"] != "H100" or attempt["index"] != 0:
                        self._block(path, record, "fallback_already_used", phase="fallback")
                    record["attempts"].append(self._new_attempt(record, "A100-80GB"))
                    record["status"] = "dispatching"
                    self._persist(path, record)
                    self._launch_attempt(path, record)
                    continue
                if attempt["status"] == "planned":
                    self._launch_attempt(path, record)
                    continue
                elapsed = self.clock() - attempt["submitted_at"]
                if (attempt["gpu"] == "H100" and not attempt.get("start_observed")
                        and elapsed >= record["resources"]["queue_timeout_seconds"]
                        and record["status"] != "complete"):
                    decision = self.gate.try_cancel_queued(
                        attempt["gate_key"], attempt["submitted_at"], now=self.clock(),
                        after_seconds=record["resources"]["queue_timeout_seconds"],
                    )
                    if decision.may_cancel:
                        attempt["gate_cancellation"] = decision.record
                        self._cancel(path, record)
                        continue
                    if decision.state != "started":
                        self._block(path, record, "unknown_start_gate", phase="queue_timeout")
                    attempt["start_observed"] = True
                    self._persist(path, record)
                try:
                    response = self.remote.result(attempt["child_key"], timeout=wait_seconds)
                except JobPending:
                    continue
                except JobBlocked:
                    self._block(path, record, "child_job_blocked", phase="get")
                attempt["status"] = "complete"
                record["status"] = "complete"
                self._persist(path, record)
                return self._response(record, response)
