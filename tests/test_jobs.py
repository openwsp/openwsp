"""Crash and retry behavior for durable remote calls, without paid providers."""

from decimal import Decimal

import pytest

from openwsp.backends.jobs import JobBlocked, JobPending, RemoteJobRunner
from openwsp.budget import BudgetLedger, BudgetOverrun, compute_cost

COST_OPTIONS = {
    "gpu": "H100",
    "cpu_cores": 2,
    "memory_gib": 16,
    "sandbox": False,
    "startup_seconds": 2,
    "scaledown_seconds": 3,
}


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds


class Call:
    object_id = "fc-test-job"

    def __init__(self, response=None, errors=()):
        self.response = response if response is not None else {"seconds": 10, "answer": "done"}
        self.errors = list(errors)
        self.timeouts = []

    def get(self, *, timeout):
        assert timeout > 0
        self.timeouts.append(timeout)
        if self.errors:
            raise self.errors.pop(0)
        return self.response


@pytest.fixture
def setup(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    clock = Clock()
    call = Call()
    launches = []
    commits = []

    def commit():
        commits.append(ledger.snapshot())

    def spawn():
        # A provider launch must never precede its funded, committed reservation.
        assert ledger.snapshot()["summary"]["modal"]["reserved_usd"] > 0
        assert commits
        launches.append(call.object_id)
        return call

    def factory(call_id):
        assert call_id == call.object_id
        return call

    def runner():
        return RemoteJobRunner(
            ledger,
            tmp_path / "jobs",
            commit=commit,
            call_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )

    return ledger, clock, call, launches, commits, spawn, runner


def launch(runner, spawn, *, fingerprint="request-v1", bound=1):
    return runner.launch(
        "calibration/base/shard0",
        spawn,
        upper_bound_usd=bound,
        cost_options=COST_OPTIONS,
        fingerprint=fingerprint,
    )


def reservation(ledger):
    reservations = ledger.snapshot()["reservations"]
    assert len(reservations) == 1
    return next(iter(reservations.values()))


def test_resuming_a_persisted_call_never_spawns_again(setup):
    ledger, _, call, launches, _, spawn, runner = setup
    assert launch(runner(), spawn) == call.object_id
    assert launch(runner(), spawn) == call.object_id
    assert launches == [call.object_id]
    assert runner().result("calibration/base/shard0") == call.response
    gets = len(call.timeouts)
    assert runner().result("calibration/base/shard0") == call.response
    assert len(call.timeouts) == gets
    assert reservation(ledger)["status"] == "settled"


def test_reusing_a_key_with_different_inputs_is_refused(setup):
    ledger, _, call, launches, _, spawn, runner = setup
    launch(runner(), spawn)
    with pytest.raises((JobBlocked, ValueError), match="(?i)fingerprint|different|mismatch"):
        launch(runner(), spawn, fingerprint="different-model-or-seed")
    assert launches == [call.object_id]
    assert reservation(ledger)["status"] == "reserved"


def test_uncertain_dispatch_keeps_its_funds_and_cannot_be_relaunched(setup):
    ledger, _, _, _, _, _, runner = setup
    attempts = []

    def lost_response():
        attempts.append(True)
        raise ConnectionError("connection lost after provider accepted the job")

    with pytest.raises(JobBlocked):
        launch(runner(), lost_response)
    with pytest.raises(JobBlocked):
        launch(runner(), lost_response)
    assert len(attempts) == 1
    assert reservation(ledger)["status"] == "reserved"
    assert ledger.snapshot()["summary"]["modal"]["reserved_usd"] == 1


def test_crash_after_receipt_settlement_does_not_settle_or_fetch_twice(setup, monkeypatch):
    ledger, _, call, _, _, spawn, runner = setup
    launch(runner(), spawn)
    settle = ledger.settle
    settlements = []

    class SimulatedCrash(BaseException):
        pass

    def settle_then_crash(*args, **kwargs):
        settlements.append(args[0])
        settle(*args, **kwargs)
        raise SimulatedCrash()

    monkeypatch.setattr(ledger, "settle", settle_then_crash)
    with pytest.raises(SimulatedCrash):
        runner().result("calibration/base/shard0")
    assert reservation(ledger)["status"] == "settled"
    # Keep the crashing wrapper installed: any duplicate settlement would fail.
    assert runner().result("calibration/base/shard0") == call.response
    assert len(settlements) == len(call.timeouts) == 1
    events = ledger.snapshot()["events"]
    assert len([event for event in events if event["event"] == "settled"]) == 1


def test_connection_failure_retries_retrieval_without_new_compute(setup):
    ledger, clock, call, launches, _, spawn, runner = setup
    call.errors = [ConnectionError("temporary disconnected result stream")]
    launch(runner(), spawn)
    assert runner().result("calibration/base/shard0", timeout=30) == call.response
    assert len(call.timeouts) == 2
    assert clock.sleeps
    assert launches == [call.object_id]
    assert reservation(ledger)["status"] == "settled"


@pytest.mark.parametrize("exception_name", ["TimeoutError", "ConnectionError"])
def test_provider_get_errors_need_not_inherit_builtin_exceptions(setup, exception_name):
    ledger, _, call, launches, _, spawn, runner = setup
    # Modal declares these as its own Error subclasses, not the Python builtins.
    provider_exception = type(exception_name, (Exception,), {})
    call.errors = [provider_exception("temporary retrieval failure")]
    launch(runner(), spawn)
    assert runner().result("calibration/base/shard0", timeout=30) == call.response
    assert len(call.timeouts) == 2
    assert launches == [call.object_id]
    assert reservation(ledger)["status"] == "settled"


def test_remote_execution_timeout_is_terminal_not_a_get_retry(setup):
    ledger, clock, call, _, _, spawn, runner = setup
    execution_timeout = type("FunctionTimeoutError", (Exception,), {})
    call.errors = [execution_timeout("function exceeded its execution timeout")]
    launch(runner(), spawn)
    with pytest.raises(JobBlocked):
        runner().result("calibration/base/shard0", timeout=30)
    assert len(call.timeouts) == 1
    assert not clock.sleeps
    assert reservation(ledger)["status"] == "reserved"


def test_pending_timeout_retains_the_call_for_later_retrieval(setup):
    ledger, clock, call, launches, _, spawn, runner = setup
    call.errors = [TimeoutError("still running") for _ in range(100)]
    launch(runner(), spawn)
    with pytest.raises(JobPending):
        runner().result("calibration/base/shard0", timeout=2)
    assert clock.now >= 102
    assert len(call.timeouts) < 100
    assert reservation(ledger)["status"] == "reserved"
    call.errors.clear()
    assert launch(runner(), spawn) == call.object_id
    assert runner().result("calibration/base/shard0", timeout=30) == call.response
    assert launches == [call.object_id]
    assert reservation(ledger)["status"] == "settled"


@pytest.mark.parametrize("message", ["401 Unauthorized", "insufficient_quota", "budget exceeded"])
def test_auth_or_quota_errors_are_terminal_without_retry_or_refund(setup, message):
    ledger, clock, call, _, _, spawn, runner = setup
    call.errors = [RuntimeError(message)]
    launch(runner(), spawn)
    with pytest.raises(JobBlocked):
        runner().result("calibration/base/shard0", timeout=30)
    assert len(call.timeouts) == 1
    assert not clock.sleeps
    assert reservation(ledger)["status"] == "reserved"
    with pytest.raises(JobBlocked):
        runner().result("calibration/base/shard0", timeout=30)
    assert len(call.timeouts) == 1


def test_charge_uses_longest_observation_plus_startup_and_scaledown(setup):
    ledger, clock, call, _, _, spawn, runner = setup
    call.response = {"seconds": 10, "container_seconds": 12, "gpu_seconds": 20, "billable_seconds": 25}
    launch(runner(), spawn)
    clock.now += 15
    runner().result("calibration/base/shard0")
    options = {key: value for key, value in COST_OPTIONS.items() if not key.endswith("_seconds")}
    expected = compute_cost(30, **options)
    assert Decimal(reservation(ledger)["actual_usd"]) == expected


@pytest.mark.parametrize("reported", [0.0, 0.50])
def test_charge_is_maximum_of_reported_and_measured_compute_cost(setup, reported):
    ledger, _, call, _, _, spawn, runner = setup
    call.response = {"seconds": 10, "cost_usd": reported}
    launch(runner(), spawn)
    runner().result("calibration/base/shard0")
    options = {key: value for key, value in COST_OPTIONS.items() if not key.endswith("_seconds")}
    expected = max(Decimal(str(reported)), compute_cost(15, **options))
    assert Decimal(reservation(ledger)["actual_usd"]) == expected


def test_longer_observed_duration_records_overrun_and_halts_budget(setup):
    ledger, _, call, _, _, spawn, runner = setup
    call.response = {"seconds": 1, "container_seconds": 2, "gpu_seconds": 100}
    launch(runner(), spawn, bound=0.01)
    with pytest.raises(BudgetOverrun):
        runner().result("calibration/base/shard0")
    assert reservation(ledger)["status"] == "settled"
    assert Decimal(reservation(ledger)["actual_usd"]) > Decimal("0.01")
    assert ledger.snapshot()["halted"] is True
