"""Atomic ownership of the boundary before an expensive job starts.

The backend contract is ``put(key, value, skip_if_exists=True) -> bool`` and
``get(key, default=None)``; Modal Dict provides these public operations. A worker
and its controller race to insert one immutable record. Every attempt needs a
fresh job key, and every worker needs a distinct owner ID. Never delete, overwrite
or reuse a key while an attempt can still run. Modal Dict entries expire after
seven inactive days; attempts must finish within that window or keep reading it.

The worker checks this gate *before* model imports, loading or compilation and
does no expensive work unless ``may_start`` is true. The controller can request
cancellation only if ``may_cancel`` is true. It must still confirm the original
provider call is terminal before dispatching any fallback, and retain its budget
reservation until reconciliation. This gate does not claim that a briefly
allocated GPU container costs nothing.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateDecision:
    state: str
    may_start: bool = False
    may_cancel: bool = False
    record: dict[str, Any] | None = None
    reason: str | None = None


class StartGate:
    """Fail-closed start/cancel arbitration without observing provider internals."""

    def __init__(self, backend: Any, *, namespace: str = "openwsp:start:v1", clock=time.time):
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a nonempty string")
        self.backend = backend
        self.namespace = namespace
        self.clock = clock

    def _key(self, job_key: str) -> str:
        if not isinstance(job_key, str) or not job_key:
            raise ValueError("job_key must identify a unique attempt")
        return self.namespace + ":" + hashlib.sha256(job_key.encode()).hexdigest()

    @staticmethod
    def _time(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite timestamp or duration")
        return float(value)

    @staticmethod
    def _decision(record: Any, job_key: str, *, starter: str | None = None) -> GateDecision:
        if record is None:
            return GateDecision("absent")
        if (
            not isinstance(record, dict)
            or record.get("version") != 1
            or record.get("job_key") != job_key
            or record.get("state") not in ("started", "cancelled")
            or not isinstance(record.get("owner_id"), str)
            or not record["owner_id"]
        ):
            return GateDecision("unknown", reason="invalid_gate_record")
        state = record["state"]
        return GateDecision(
            state,
            may_start=state == "started" and starter is not None and record["owner_id"] == starter,
            may_cancel=state == "cancelled",
            record=dict(record),
        )

    def observe(self, job_key: str) -> GateDecision:
        """Read evidence; an observation never grants permission to start work."""
        key = self._key(job_key)
        if self.backend is None:
            return GateDecision("unknown", reason="gate_unavailable")
        try:
            return self._decision(self.backend.get(key, None), job_key)
        except Exception as exc:  # noqa: BLE001 — any provider failure must deny permission.
            # Do not expose SDK exception messages, which may contain credentials.
            return GateDecision("unknown", reason="gate_read_" + type(exc).__name__)

    def _claim(self, job_key: str, *, state: str, owner_id: str, now: float) -> GateDecision:
        key = self._key(job_key)
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner_id must identify one worker or controller")
        if self.backend is None:
            return GateDecision("unknown", reason="gate_unavailable")
        record = {"version": 1, "job_key": job_key, "state": state, "owner_id": owner_id, "created_at": now}
        starter = owner_id if state == "started" else None
        try:
            added = self.backend.put(key, record, skip_if_exists=True)
        except Exception as exc:  # noqa: BLE001 — ambiguous writes require reconciliation.
            # A lost write acknowledgement is not a failed claim. Read the
            # immutable winner, but never infer ownership from an absent read.
            result = self.observe(job_key)
            if result.record is not None:
                return self._decision(result.record, job_key, starter=starter)
            return GateDecision("unknown", reason="gate_write_" + type(exc).__name__)
        if added is True:
            return self._decision(record, job_key, starter=starter)
        if added is not False:
            return GateDecision("unknown", reason="unsupported_atomic_put_result")
        result = self.observe(job_key)
        if result.record is not None:
            return self._decision(result.record, job_key, starter=starter)
        # False means an existing winner. Its absence or unreadability cannot
        # authorize either side to cross the boundary.
        return GateDecision("unknown", reason=result.reason or "gate_winner_unavailable")

    def try_start(self, job_key: str, owner_id: str) -> GateDecision:
        """Claim before any heavy worker initialization; fail closed on errors."""
        return self._claim(job_key, state="started", owner_id=owner_id, now=self._time(self.clock(), "now"))

    def try_cancel_queued(
        self,
        job_key: str,
        submitted_at: float,
        *,
        now: float | None = None,
        after_seconds: float = 600,
        owner_id: str = "controller",
    ) -> GateDecision:
        """Claim cancellation after a deadline only if no worker has claimed start.

        A cancelled decision permits a provider cancellation request, not a new
        dispatch. Provider terminal confirmation remains the caller's duty.
        """
        self._key(job_key)
        submitted_at = self._time(submitted_at, "submitted_at")
        now = self._time(self.clock() if now is None else now, "now")
        after_seconds = self._time(after_seconds, "after_seconds")
        if after_seconds < 0:
            raise ValueError("after_seconds must be nonnegative")
        if now - submitted_at < after_seconds:
            result = self.observe(job_key)
            return GateDecision("waiting", reason="queue_deadline_not_reached") if result.state == "absent" else result
        return self._claim(job_key, state="cancelled", owner_id=owner_id, now=now)
