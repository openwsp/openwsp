"""Durable, budgeted dispatch and retrieval of Modal jobs.

One driver owns a ledger and these journals. Filesystem locks coordinate local
processes, not separate Modal Volume mounts. Every dispatch commits its intent
first; a lost dispatch receipt blocks instead of risking duplicate paid work.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import time
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

from openwsp.budget import BudgetOverrun, atomic_json, compute_cost


class JobBlocked(RuntimeError):
    """The job needs reconciliation; automatically repeating it could spend twice."""


class JobPending(TimeoutError):
    """The existing call has not yielded a durable result before this read deadline."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _duration(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _classification(exc: Exception) -> str:
    """Classify without putting potentially credential-bearing exception messages in journals."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)
    grpc_status = str(getattr(exc, "status", "")).lower()
    if status in (401, 403) or any(
        term in name for term in ("authentication", "permissiondenied", "autherror")
    ):
        return "authentication"
    if any(
        term in message
        for term in ("unauthorized", "invalid token", "authentication failed", "permission denied")
    ):
        return "authentication"
    if status == 402 or any(
        term in message
        for term in (
            "insufficient_quota",
            "insufficient quota",
            "usage limit",
            "billing limit",
            "payment required",
            "out of credits",
            "quota exhausted",
            "budget exceeded",
        )
    ):
        return "quota"
    if any(term in grpc_status for term in ("unauthenticated", "permission_denied")):
        return "authentication"
    if "resource_exhausted" in grpc_status:
        return "quota"
    # A remote function's own execution timeout is terminal, unlike a GET timeout.
    if "functiontimeout" in name or "executiontimeout" in name:
        return "job_failure"
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)) or name in {
        "timeouterror",
        "connectionerror",
    }:
        return "transient_get"
    if isinstance(status, int) and (status == 429 or status >= 500):
        return "transient_get"
    if any(term in grpc_status for term in ("unavailable", "deadline_exceeded", "internal")):
        return "transient_get"
    return "job_failure"


def _modal_call(call_id: str):
    import modal

    return modal.FunctionCall.from_id(call_id)


class RemoteJobRunner:
    """Dispatch once, journal the call ID, and settle a persisted response once.

    ``spawn_callable`` must submit one already bounded remote call and return its
    FunctionCall handle. The remote timeout and resource maxima must match the
    supplied reservation. ``commit`` flushes both the ledger and job directory's
    Volume; the caller must mount both on that Volume.
    """

    def __init__(self, ledger, directory, commit=lambda: None, *, call_factory=None, clock=None, sleep=None):
        self.ledger = ledger
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.call_factory = call_factory or _modal_call
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not key:
            raise ValueError("job key must be a nonempty string")
        return self.directory / f"{_digest(key)}.json"

    @contextmanager
    def _locked(self, key):
        path = self._path(key)
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield path

    def _persist(self, path: Path, record: dict) -> None:
        atomic_json(path, record)
        self.commit()

    @staticmethod
    def _read(path: Path) -> dict:
        record = json.loads(path.read_text())
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported remote job journal version")
        return record

    def _blocked(self, path, record, category, *, phase, exception=None):
        record.update(status="blocked", blocked_at=self.clock())
        record["failure"] = {"category": category, "phase": phase}
        if exception is not None:
            record["failure"]["exception_type"] = type(exception).__name__
        self._persist(path, record)

    @staticmethod
    def _options(options: dict) -> dict:
        allowed = {"gpu", "cpu_cores", "memory_gib", "sandbox", "startup_seconds", "scaledown_seconds"}
        if not isinstance(options, dict):
            raise TypeError("cost_options must be an object")
        if set(options) - allowed:
            raise ValueError("Unknown cost option: " + ", ".join(sorted(set(options) - allowed)))
        normalized = {
            "gpu": None,
            "cpu_cores": 2,
            "memory_gib": 16,
            "sandbox": False,
            "startup_seconds": 0,
            "scaledown_seconds": 0,
            **options,
        }
        for key in ("startup_seconds", "scaledown_seconds", "cpu_cores", "memory_gib"):
            normalized[key] = _duration(normalized[key], key)
        if not isinstance(normalized["sandbox"], bool):
            raise TypeError("sandbox must be a boolean")
        # Reject an unknown GPU or invalid rate parameters before dispatching.
        compute_cost(0, **{key: value for key, value in normalized.items() if not key.endswith("_seconds")})
        return normalized

    def launch(self, key, spawn_callable, *, upper_bound_usd, cost_options, fingerprint) -> str:
        fingerprint_hash = _digest(_canonical(fingerprint))
        with self._locked(key) as path:
            if path.exists():
                record = self._read(path)
                if record["key"] != key or record["fingerprint_sha256"] != fingerprint_hash:
                    raise ValueError("Job key already belongs to a different fingerprint")
                if record.get("call_id"):
                    # A prior local commit may have failed: durably save the known ID before returning it.
                    self._persist(path, record)
                    return record["call_id"]
                self._blocked(path, record, "uncertain_dispatch", phase="resume")
                raise JobBlocked(
                    f"Job {key} has a dispatch intent without a call ID; refusing duplicate work"
                )
            options = self._options(cost_options)
            reservation_id = "modal-job-" + _digest(key + "\0" + fingerprint_hash)
            self.ledger.reserve(
                "modal",
                upper_bound_usd,
                key,
                reservation_id=reservation_id,
                metadata={"fingerprint_sha256": fingerprint_hash, "cost_options": options},
            )
            record = {
                "schema_version": 1,
                "key": key,
                "fingerprint_sha256": fingerprint_hash,
                "reservation_id": reservation_id,
                "upper_bound_usd": str(upper_bound_usd),
                "cost_options": options,
                "created_at": self.clock(),
                "status": "dispatching",
            }
            self._persist(path, record)
            try:
                call = spawn_callable()
                call_id = call if isinstance(call, str) else call.object_id
                if not isinstance(call_id, str) or not call_id:
                    raise TypeError("Spawn did not return a FunctionCall ID")
            except Exception as exc:
                category = _classification(exc)
                if category not in {"authentication", "quota"}:
                    category = "uncertain_dispatch"
                self._blocked(path, record, category, phase="spawn", exception=exc)
                raise JobBlocked(
                    f"Job {key} dispatch failed; reservation retained and automatic retry disabled"
                ) from exc
            record.update(call_id=call_id, status="pending", submitted_at=self.clock())
            self._persist(path, record)
            return call_id

    def _receipt(self, record: dict, response: dict) -> dict:
        options = record["cost_options"]
        elapsed = max(0.0, self.clock() - record["created_at"])
        measured = response.get("container_seconds")
        source = "container_seconds" if measured is not None else "elapsed_wall_upper_estimate"
        seconds = _duration(measured, "container_seconds") if measured is not None else elapsed
        for key in ("seconds", "gpu_seconds", "billable_seconds"):
            if response.get(key) is not None:
                seconds = max(seconds, _duration(response[key], key))
        charged_seconds = seconds + options["startup_seconds"] + options["scaledown_seconds"]
        cost = compute_cost(
            charged_seconds, **{key: value for key, value in options.items() if not key.endswith("_seconds")}
        )
        if response.get("cost_usd") is not None:
            stated = Decimal(str(response["cost_usd"]))
            if not stated.is_finite() or stated < 0:
                raise ValueError("cost_usd must be finite and nonnegative")
            cost = max(cost, stated)
        return {
            "actual_usd": str(cost),
            "seconds": charged_seconds,
            "usage_seconds": seconds,
            "duration_source": source,
            "startup_seconds": options["startup_seconds"],
            "scaledown_seconds": options["scaledown_seconds"],
            "cost_options": options,
        }

    def _stored_response(self, path: Path, record: dict) -> dict:
        response_path = self.directory / record["response_file"]
        raw = response_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["response_sha256"]:
            raise JobBlocked("Persisted job response failed its sha256 check")
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise TypeError("Persisted job response must be an object")
        receipt = record["receipt"]
        reserved = self.ledger.snapshot()["reservations"][record["reservation_id"]]
        actual = Decimal(receipt["actual_usd"])
        if reserved["status"] == "settled":
            if Decimal(reserved["actual_usd"]) != actual:
                raise JobBlocked("Journal receipt disagrees with settled ledger cost")
        else:
            # The response and receipt must already be durable before updating the ledger.
            self.commit()
            try:
                self.ledger.settle(
                    record["reservation_id"],
                    actual,
                    seconds=receipt["seconds"],
                    metadata={"key": record["key"], "call_id": record["call_id"], **receipt},
                )
            except BudgetOverrun:
                record.update(status="overrun", settled=True, completed_at=self.clock())
                self._persist(path, record)
                raise
        overrun = actual > Decimal(record["upper_bound_usd"])
        record.update(
            status="overrun" if overrun else "complete",
            settled=True,
            completed_at=record.get("completed_at", self.clock()),
        )
        self._persist(path, record)
        if overrun:
            raise BudgetOverrun("Remote job exceeded its reservation; result saved and ledger halted")
        return response

    def result(self, key, timeout=1800) -> dict:
        timeout = _duration(timeout, "timeout")
        with self._locked(key) as path:
            if not path.exists():
                raise KeyError(f"No recorded job: {key}")
            record = self._read(path)
            if record.get("response_file"):
                return self._stored_response(path, record)
            if not record.get("call_id"):
                self._blocked(path, record, "uncertain_dispatch", phase="resume")
                raise JobBlocked(f"Job {key} has no safely recorded call ID; automatic dispatch is disabled")
            if record["status"] == "blocked":
                raise JobBlocked(
                    f"Job {key} is blocked; dispatch was not safely recorded or retrieval failed"
                )
            deadline = self.clock() + timeout
            attempt = 0
            while True:
                remaining = max(0.0, deadline - self.clock())
                if remaining <= 0:
                    raise JobPending(f"Job {key} still pending; resume its existing call ID")
                try:
                    call = self.call_factory(record["call_id"])
                    response = call.get(timeout=min(remaining, 30))
                    if not isinstance(response, dict):
                        raise TypeError("Remote job must return a JSON object")
                    receipt = self._receipt(record, response)
                    # Serialize before receipt mutation; a non-JSON result is a failure, not a fake receipt.
                    _canonical(response)
                    break
                except Exception as exc:
                    category = _classification(exc)
                    if category != "transient_get":
                        self._blocked(path, record, category, phase="get", exception=exc)
                        raise JobBlocked(
                            f"Job {key} retrieval failed ({category}); reservation retained"
                        ) from exc
                    record["last_get_failure"] = {
                        "category": category,
                        "exception_type": type(exc).__name__,
                        "at": self.clock(),
                    }
                    self._persist(path, record)
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        raise JobPending(f"Job {key} still pending; resume its existing call ID") from exc
                    self.sleep(min(2 ** min(attempt, 5), remaining))
                    attempt += 1
            response_path = path.with_suffix(".result.json")
            atomic_json(response_path, response)
            record.update(
                status="received",
                response_file=response_path.name,
                response_sha256=hashlib.sha256(response_path.read_bytes()).hexdigest(),
                receipt=receipt,
                received_at=self.clock(),
            )
            self._persist(path, record)
            return self._stored_response(path, record)
