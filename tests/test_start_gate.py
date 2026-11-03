from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from openwsp.backends.start_gate import StartGate


class AtomicDict:
    def __init__(self):
        self.values = {}
        self.lock = Lock()
        self.writes = 0

    def put(self, key, value, *, skip_if_exists=False):
        assert skip_if_exists is True
        with self.lock:
            self.writes += 1
            if key in self.values:
                return False
            self.values[key] = value
            return True

    def get(self, key, default=None):
        with self.lock:
            return self.values.get(key, default)


def test_started_worker_is_protected_even_after_queue_deadline():
    gate = StartGate(AtomicDict())
    assert gate.try_start("h100-attempt-1", "worker-1").may_start
    decision = gate.try_cancel_queued("h100-attempt-1", 100, now=1000)
    assert decision.state == "started"
    assert not decision.may_cancel
    assert not decision.may_start


def test_cancelled_attempt_never_initializes_and_fallback_has_fresh_key():
    gate = StartGate(AtomicDict())
    decision = gate.try_cancel_queued("h100-attempt-1", 100, now=701)
    assert decision.may_cancel
    assert not gate.try_start("h100-attempt-1", "late-worker").may_start
    assert gate.try_start("a100-attempt-2", "fallback-worker").may_start


def test_deadline_check_does_not_mutate_or_use_negative_elapsed_time():
    backend = AtomicDict()
    gate = StartGate(backend)
    assert gate.try_cancel_queued("job", 100, now=699.9).state == "waiting"
    assert gate.try_cancel_queued("job", 100, now=99).state == "waiting"
    assert backend.writes == 0
    assert gate.try_cancel_queued("job", 100, now=700).may_cancel


def test_two_workers_cannot_both_start_the_same_attempt():
    gate = StartGate(AtomicDict())
    assert gate.try_start("job", "worker-1").may_start
    assert not gate.try_start("job", "worker-2").may_start
    assert gate.try_start("job", "worker-1").may_start
    assert not gate.observe("job").may_start


def test_concurrent_start_and_cancellation_have_exactly_one_winner():
    for index in range(40):
        gate = StartGate(AtomicDict())
        barrier = Barrier(2)

        def start(barrier=barrier, gate=gate, index=index):
            barrier.wait()
            return gate.try_start(str(index), "worker")

        def cancel(barrier=barrier, gate=gate, index=index):
            barrier.wait()
            return gate.try_cancel_queued(str(index), 0, now=601)

        with ThreadPoolExecutor(max_workers=2) as pool:
            worker = pool.submit(start)
            controller = pool.submit(cancel)
            started, cancelled = worker.result(), controller.result()
        assert started.may_start != cancelled.may_cancel
        assert started.state == cancelled.state


def test_lost_acknowledgement_recovers_immutable_winner():
    class LostAcknowledgement(AtomicDict):
        def put(self, *args, **kwargs):
            super().put(*args, **kwargs)
            raise ConnectionError("secret transport detail")

    gate = StartGate(LostAcknowledgement())
    assert gate.try_start("job", "worker").may_start
    assert not gate.try_cancel_queued("job", 0, now=601).may_cancel


def test_unavailable_gate_and_ambiguous_absence_fail_closed():
    class Unavailable(AtomicDict):
        def put(self, *args, **kwargs):
            raise ConnectionError("secret token")

    for backend in (None, Unavailable()):
        gate = StartGate(backend)
        for decision in (gate.try_start("job", "worker"), gate.try_cancel_queued("job", 0, now=601)):
            assert decision.state == "unknown"
            assert not decision.may_start and not decision.may_cancel
            assert "secret" not in repr(decision)


def test_unreadable_existing_winner_is_not_a_new_queued_job():
    class Unreadable(AtomicDict):
        def put(self, *args, **kwargs):
            return False

    gate = StartGate(Unreadable())
    assert gate.try_start("job", "worker").state == "unknown"
    assert not gate.try_cancel_queued("job", 0, now=601).may_cancel


def test_invalid_existing_record_and_incompatible_backend_fail_closed():
    class Invalid(AtomicDict):
        def get(self, *args):
            return {"version": 1, "state": "cancelled", "job_key": "other", "owner_id": "controller"}

        def put(self, *args, **kwargs):
            return False

    class Incompatible(AtomicDict):
        def put(self, *args, **kwargs):
            return None

    for backend in (Invalid(), Incompatible()):
        result = StartGate(backend).try_cancel_queued("job", 0, now=601)
        assert result.state == "unknown" and not result.may_cancel


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "601"])
def test_invalid_clock_cannot_authorize_cancellation(value):
    with pytest.raises(ValueError):
        StartGate(AtomicDict()).try_cancel_queued("job", 0, now=value)
