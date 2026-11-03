"""Provider state and budget invariants of a restartable driver coordinator."""

import json
import sys
from decimal import Decimal
from types import ModuleType

import pytest

from openwsp.backends.coordinator import DriverCoordinator
from openwsp.backends.driver_budget import DriverBudgetLease
from openwsp.budget import BudgetExceeded, BudgetLedger

LEASE_OPTIONS = {
    "timeout_seconds": 1800,
    "startup_seconds": 2,
    "scaledown_seconds": 3,
    "cpu_cores": 1,
    "memory_gib": 4,
}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class InputCancellation(BaseException):
    pass


class Call:
    def __init__(self, call_id):
        self.object_id = call_id
        self.complete = False
        self.cancelled = False
        self.cancel_sticks = True
        self.get_error = None
        self.events = []
        self.response = {"status": "complete"}

    def get(self, *, timeout):
        assert timeout == 0
        self.events.append("get")
        if self.get_error:
            raise self.get_error
        if self.cancelled:
            raise InputCancellation()
        if self.complete:
            return self.response
        raise TimeoutError("still running")

    def cancel(self, *, terminate_containers):
        assert terminate_containers is True
        self.events.append("cancel")
        self.cancelled = self.cancel_sticks


@pytest.fixture
def setup(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    path, state_path = tmp_path / "coordinator.json", tmp_path / "state.json"
    clock = Clock()
    calls, tickets, commits, reloads = [], [], [], []

    def write_state(**changes):
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state.update(changes)
        state_path.write_text(json.dumps(state))

    def commit():
        commits.append({
            "ledger": ledger.snapshot(),
            "coordinator": json.loads(path.read_text()) if path.exists() else None,
        })

    def reload():
        reloads.append(True)

    def factory(call_id):
        return next(call for call in calls if call.object_id == call_id)

    def spawn(stage, config, ticket):
        assert stage == "calibration" and config == {"seed": 0}
        row = ledger.snapshot()["reservations"][ticket["reservation_id"]]
        assert row["status"] == "reserved"
        # The commit preceding POST must contain both its reservation and intent.
        assert any(
            ticket["reservation_id"] in json.dumps(item["coordinator"])
            and ticket["reservation_id"] in item["ledger"]["reservations"]
            for item in commits
        )
        call = Call(f"fc-driver-{len(calls)}")
        call.lease = DriverBudgetLease.claim(
            ledger, ticket, owner_id=call.object_id, clock=clock, commit=commit
        )
        calls.append(call)
        tickets.append(ticket)
        write_state(status="running", heartbeat=clock(), active_stage="calibration")
        return call

    def coordinator():
        return DriverCoordinator(
            ledger,
            path,
            state_path=state_path,
            commit=commit,
            reload=reload,
            call_factory=factory,
            clock=clock,
            lease_options=LEASE_OPTIONS,
            stale_seconds=600,
            max_restarts=3,
        )

    return ledger, clock, calls, tickets, write_state, spawn, coordinator, reloads


def start(coordinator, spawn):
    return coordinator().launch("calibration", {"seed": 0}, spawn)


def rows(ledger):
    return ledger.snapshot()["reservations"]


def test_fresh_live_call_is_recovered_without_another_dispatch(setup):
    ledger, _, calls, _, _, spawn, coordinator, reloads = setup
    start(coordinator, spawn)
    assert coordinator().recover(spawn)["status"] == "running"
    assert coordinator().recover(spawn)["status"] == "running"
    assert len(calls) == len(rows(ledger)) == 1
    assert not calls[0].cancelled
    assert reloads


def test_legacy_running_stage_without_owned_ticket_is_not_duplicated(setup):
    ledger, clock, calls, _, write_state, spawn, coordinator, _ = setup
    write_state(status="running", heartbeat=clock(), active_stage="calibration")
    assert coordinator().recover(spawn)["status"] == "legacy_running"
    assert not calls
    assert not rows(ledger)


def test_ambiguous_dispatch_is_blocked_with_its_reservation_intact(setup):
    ledger, _, calls, _, _, _, coordinator, _ = setup
    attempts = []

    def uncertain_spawn(stage, config, ticket):
        attempts.append(ticket)
        raise ConnectionError("provider accepted POST but its response was lost")

    result = start(coordinator, uncertain_spawn)
    assert result["status"] == "blocked"
    assert coordinator().recover(uncertain_spawn)["status"] == "blocked"
    assert len(attempts) == 1
    assert not calls
    assert len(rows(ledger)) == 1
    assert next(iter(rows(ledger).values()))["status"] == "reserved"


def test_stale_owner_is_cancelled_and_settled_before_its_replacement(setup):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    old = calls[0]
    clock.now += 601
    old_id = tickets[0]["reservation_id"]

    def checked_spawn(stage, config, ticket):
        assert old.cancelled
        assert old.events[-2:] == ["cancel", "get"]
        row = rows(ledger)[old_id]
        assert row["status"] == "settled"
        assert Decimal(row["actual_usd"]) == Decimal(row["upper_bound_usd"])
        return spawn(stage, config, ticket)

    assert coordinator().recover(checked_spawn)["status"] == "running"
    assert len(calls) == 2
    assert tickets[0]["reservation_id"] != tickets[1]["reservation_id"]
    assert rows(ledger)[tickets[1]["reservation_id"]]["status"] == "reserved"


def test_unconfirmed_cancellation_cannot_release_funds_or_spawn(setup):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    calls[0].cancel_sticks = False
    clock.now += 601
    assert coordinator().recover(spawn)["status"] == "pending"
    assert len(calls) == 1
    assert "cancel" in calls[0].events
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "reserved"


@pytest.mark.parametrize("interruption", ["before-cancel-rpc", "uncertain-cancel-rpc"])
def test_interrupted_cancellation_is_reissued_and_confirmed_before_replacement(setup, interruption):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    old, old_ticket = calls[0], tickets[0]
    clock.now += 601
    interrupted = coordinator()

    class SimulatedCrash(BaseException):
        pass

    if interruption == "before-cancel-rpc":
        commit = interrupted.commit

        def commit_then_crash():
            commit()
            if json.loads(interrupted.path.read_text())["current"]["phase"] == "cancelling":
                raise SimulatedCrash()

        interrupted.commit = commit_then_crash
        with pytest.raises(SimulatedCrash):
            interrupted.recover(spawn)
        assert old.events == ["get"]
    else:
        cancel = old.cancel

        def uncertain_cancel(*, terminate_containers):
            assert terminate_containers is True
            old.events.append("cancel")
            raise ConnectionError("Cancellation acknowledgement lost before delivery was known")

        old.cancel = uncertain_cancel
        assert interrupted.recover(spawn)["status"] == "pending"
        old.cancel = cancel
        assert old.events == ["get", "cancel"]
    assert not old.cancelled and len(calls) == 1
    assert rows(ledger)[old_ticket["reservation_id"]]["status"] == "reserved"

    def checked_spawn(stage, config, ticket):
        assert old.cancelled and old.events[-2:] == ["cancel", "get"]
        row = rows(ledger)[old_ticket["reservation_id"]]
        assert row["status"] == "settled"
        assert Decimal(row["actual_usd"]) == Decimal(row["upper_bound_usd"])
        return spawn(stage, config, ticket)

    assert coordinator().recover(checked_spawn)["status"] == "running"
    assert len(calls) == len(rows(ledger)) == 2
    assert tickets[1]["reservation_id"] != old_ticket["reservation_id"]
    assert old.events.count("cancel") == (1 if interruption == "before-cancel-rpc" else 2)


@pytest.mark.parametrize("observation", ["different-stage", "unknown-get"])
def test_existing_cancellation_is_not_reissued_by_observation_or_unknown_get(setup, observation):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    old = calls[0]
    old.cancel_sticks = False
    clock.now += 601
    assert coordinator().recover(spawn)["status"] == "pending"
    old.events.clear()
    if observation == "different-stage":
        with pytest.raises(ValueError, match="owns this coordinator"):
            coordinator().launch("controls", {"seed": 1}, spawn)
    else:
        old.get_error = ConnectionError("Provider status is unavailable")
        assert coordinator().recover(spawn)["status"] == "pending"
    assert old.events == ["get"]
    assert len(calls) == len(rows(ledger)) == 1
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "reserved"


@pytest.mark.parametrize("remote_traceback", [False, True])
def test_runtime_error_requires_remote_traceback_evidence_before_releasing_owner(
    setup, monkeypatch, remote_traceback
):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    traceback_module = ModuleType("modal._traceback")
    inspected = []

    def contains_remote_call(traceback):
        inspected.append(traceback)
        return remote_traceback

    traceback_module.traceback_contains_remote_call = contains_remote_call
    monkeypatch.setitem(sys.modules, "modal._traceback", traceback_module)
    start(coordinator, spawn)
    error = RuntimeError("A failure whose source must be established")
    calls[0].get_error = error
    clock.now += 601
    result = coordinator().recover(spawn)
    assert result["status"] == ("running" if remote_traceback else "pending")
    assert inspected == [error.__traceback__]
    assert calls[0].events == ["get"]
    assert len(calls) == (2 if remote_traceback else 1)
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == (
        "settled" if remote_traceback else "reserved"
    )


def test_remote_quota_failure_reconciles_terminal_owner_without_automatic_retry(setup, monkeypatch):
    ledger, _, calls, tickets, _, spawn, coordinator, _ = setup
    traceback_module = ModuleType("modal._traceback")
    traceback_module.traceback_contains_remote_call = lambda traceback: True
    monkeypatch.setitem(sys.modules, "modal._traceback", traceback_module)
    start(coordinator, spawn)
    calls[0].get_error = RuntimeError("insufficient_quota")
    assert coordinator().recover(spawn)["status"] == "blocked"
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "settled"
    assert coordinator().recover(spawn)["status"] == "blocked"
    assert len(calls) == 1


def test_stage_is_blocked_after_three_confirmed_restarts(setup):
    _, clock, calls, _, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    for _ in range(3):
        clock.now += 601
        assert coordinator().recover(spawn)["status"] == "running"
    assert len(calls) == 4
    clock.now += 601
    assert coordinator().recover(spawn)["status"] == "blocked"
    assert coordinator().recover(spawn)["status"] == "blocked"
    assert len(calls) == 4


def test_restart_limit_is_per_active_stage(setup):
    _, clock, calls, _, write_state, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    for _ in range(3):
        clock.now += 601
        assert coordinator().recover(spawn)["status"] == "running"
    write_state(active_stage="live")
    clock.now += 601
    assert coordinator().recover(spawn)["status"] == "running"
    assert len(calls) == 5


def test_blocked_pipeline_state_prevents_watchdog_restart(setup):
    _, clock, calls, _, write_state, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    write_state(status="blocked", reason="provider quota exhausted")
    clock.now += 601
    assert coordinator().recover(spawn)["status"] == "blocked"
    assert len(calls) == 1


def test_exhausted_budget_never_dispatches_a_driver(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", modal_cycle_limit=0.01, modal_prior_usage=0)
    coordinator = DriverCoordinator(
        ledger, tmp_path / "coordinator.json", state_path=tmp_path / "state.json", lease_options=LEASE_OPTIONS
    )
    calls = []

    def spawn(*args):
        calls.append(args)
        raise AssertionError("must not dispatch unfunded work")

    try:
        result = coordinator.launch("calibration", {}, spawn)
    except BudgetExceeded:
        pass
    else:
        assert result["status"] == "blocked"
    assert not calls
    assert not rows(ledger)


def test_terminal_provider_call_settles_worker_receipt_once(setup):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    clock.now += 10
    calls[0].lease.finish(elapsed_seconds=10)
    calls[0].complete = True
    assert coordinator().recover(spawn)["status"] == "complete"
    row = rows(ledger)[tickets[0]["reservation_id"]]
    assert row["status"] == "settled"
    assert Decimal(row["actual_usd"]) < Decimal(row["upper_bound_usd"])
    evidence = row["receipt_metadata"]["provider_completion_evidence"]
    assert evidence["provider_call_id"] == calls[0].object_id
    assert evidence["provider_terminal_state"] == "completed"
    assert coordinator().recover(spawn)["status"] == "complete"
    assert len(calls) == 1
    assert len([event for event in ledger.snapshot()["events"] if event["event"] == "settled"]) == 1


@pytest.mark.parametrize("termination", ["completed", "stale-cancellation"])
def test_termination_cannot_release_a_ticket_claimed_by_a_different_provider_call(setup, termination):
    ledger, clock, calls, tickets, write_state, _, coordinator, _ = setup

    def wrong_owner_spawn(stage, config, ticket):
        call = Call("fc-provider-actual")
        call.lease = DriverBudgetLease.claim(ledger, ticket, owner_id="fc-other", clock=clock)
        calls.append(call)
        tickets.append(ticket)
        write_state(status="running", heartbeat=clock(), active_stage=stage)
        return call

    start(coordinator, wrong_owner_spawn)
    if termination == "completed":
        clock.now += 10
        calls[0].lease.finish(elapsed_seconds=10)
        calls[0].complete = True
    else:
        clock.now += 601
    assert coordinator().recover(wrong_owner_spawn)["status"] == "blocked"
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "reserved"
    assert len(calls) == 1


@pytest.mark.parametrize("error", ["401 Unauthorized", "insufficient_quota"])
def test_auth_and_quota_failures_are_terminal_without_retries(setup, error):
    ledger, _, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    calls[0].get_error = RuntimeError(error)
    assert coordinator().recover(spawn)["status"] == "blocked"
    reads = calls[0].events.count("get")
    assert coordinator().recover(spawn)["status"] == "blocked"
    assert calls[0].events.count("get") == reads
    assert len(calls) == 1
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "reserved"


def test_uncertain_get_transport_failure_cannot_cancel_or_duplicate_a_driver(setup):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    calls[0].get_error = ConnectionError("cannot read provider state")
    clock.now += 601
    assert coordinator().recover(spawn)["status"] == "pending"
    assert calls[0].events == ["get"]
    assert len(calls) == 1
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "reserved"


@pytest.mark.parametrize(
    ("exception_name", "expected_status", "expected_calls", "reservation_status"),
    [
        ("ResourceExhaustedError", "blocked", 1, "reserved"),
        ("InternalError", "pending", 1, "reserved"),
        ("OutputExpiredError", "running", 2, "settled"),
    ],
)
def test_modal_error_contract_distinguishes_access_transport_and_terminal_results(
    setup, exception_name, expected_status, expected_calls, reservation_status
):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    provider_error = type(exception_name, (Exception,), {})
    calls[0].get_error = provider_error("provider failure")
    clock.now += 601
    assert coordinator().recover(spawn)["status"] == expected_status
    assert len(calls) == expected_calls
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == reservation_status
    assert "cancel" not in calls[0].events


def test_crash_after_settlement_cannot_spawn_or_settle_a_second_time(setup, monkeypatch):
    ledger, clock, calls, _, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    clock.now += 10
    calls[0].lease.finish(elapsed_seconds=10)
    calls[0].complete = True
    settle = ledger.settle
    settlements = []

    class SimulatedCrash(BaseException):
        pass

    def crashing_settle(*args, **kwargs):
        settlements.append(args[0])
        settle(*args, **kwargs)
        raise SimulatedCrash()

    monkeypatch.setattr(ledger, "settle", crashing_settle)
    with pytest.raises(SimulatedCrash):
        coordinator().recover(spawn)
    assert coordinator().recover(spawn)["status"] == "complete"
    assert len(settlements) == len(calls) == 1


def test_crash_after_prepare_reuses_the_undispatched_ticket(setup, monkeypatch):
    ledger, _, calls, tickets, _, spawn, coordinator, _ = setup
    prepare = DriverBudgetLease.prepare

    class SimulatedCrash(BaseException):
        pass

    def prepare_then_crash(*args, **kwargs):
        prepare(*args, **kwargs)
        raise SimulatedCrash()

    with monkeypatch.context() as patch:
        patch.setattr(DriverBudgetLease, "prepare", staticmethod(prepare_then_crash))
        with pytest.raises(SimulatedCrash):
            start(coordinator, spawn)
    assert not calls
    assert len(rows(ledger)) == 1
    existing_id = next(iter(rows(ledger)))
    assert coordinator().recover(spawn)["status"] == "running"
    assert len(calls) == len(rows(ledger)) == 1
    assert tickets[0]["reservation_id"] == existing_id


def test_crash_after_terminal_journal_reconciles_without_a_new_provider_read(setup):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    clock.now += 10
    calls[0].lease.finish(elapsed_seconds=10)
    calls[0].complete = True
    interrupted = coordinator()
    commit = interrupted.commit

    class SimulatedCrash(BaseException):
        pass

    def fail_terminal_commit():
        record = json.loads(interrupted.path.read_text())
        if record["current"]["phase"] == "terminal":
            raise SimulatedCrash()
        commit()

    interrupted.commit = fail_terminal_commit
    with pytest.raises(SimulatedCrash):
        interrupted.recover(spawn)
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "reserved"
    gets = calls[0].events.count("get")
    assert coordinator().recover(spawn)["status"] == "complete"
    assert calls[0].events.count("get") == gets
    assert len(calls) == 1


def test_ledger_factory_is_not_opened_while_the_remote_driver_owns_it(setup):
    ledger, _, calls, _, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    template = coordinator()
    opened = []

    def factory():
        assert calls[0].complete or calls[0].cancelled
        opened.append(True)
        return ledger

    watcher = DriverCoordinator(
        factory, template.path, state_path=template.state_path,
        commit=template.commit, reload=template.reload, call_factory=template.call_factory,
        clock=template.clock, lease_options=LEASE_OPTIONS,
    )
    assert watcher.recover(spawn)["status"] == "running"
    assert not opened
    calls[0].complete = True
    assert watcher.recover(spawn)["status"] == "complete"
    assert opened


def test_legacy_blocked_state_requires_a_nonblank_audited_resume_reason(setup):
    ledger, _, calls, _, write_state, spawn, coordinator, _ = setup
    write_state(status="blocked", active_stage="calibration", reason="configuration invalid")
    assert start(coordinator, spawn)["status"] == "blocked"
    for reason in ("", "  "):
        with pytest.raises(ValueError):
            coordinator().launch("calibration", {"seed": 0}, spawn, resume_reason=reason)
    assert not calls and not rows(ledger)
    reason = "Fixed the configuration and verified the repair"
    resumed = coordinator()
    assert resumed.launch("calibration", {"seed": 0}, spawn, resume_reason=reason)["status"] == "running"
    assert reason in resumed.path.read_text()
    assert len(calls) == 1


def test_explicit_resume_cannot_retry_an_uncertain_dispatch(setup):
    ledger, _, _, _, _, _, coordinator, _ = setup
    attempts = []

    def lost_post_response(stage, config, ticket):
        attempts.append(ticket)
        raise ConnectionError("dispatch may have succeeded")

    assert start(coordinator, lost_post_response)["status"] == "blocked"
    resumed = coordinator().launch(
        "calibration", {"seed": 0}, lost_post_response,
        resume_reason="Network repaired, but old dispatch remains uncertain",
    )
    assert resumed["status"] == "blocked"
    assert len(attempts) == len(rows(ledger)) == 1
    assert next(iter(rows(ledger).values()))["status"] == "reserved"


def test_explicit_resume_cannot_replace_a_known_call_that_is_still_running(setup):
    ledger, _, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    calls[0].get_error = RuntimeError("insufficient_quota")
    assert coordinator().recover(spawn)["status"] == "blocked"
    calls[0].get_error = None
    assert coordinator().recover(spawn)["status"] == "blocked"
    resumed = coordinator().launch(
        "calibration", {"seed": 0}, spawn, resume_reason="Billing access restored"
    )
    assert resumed["status"] == "blocked"
    assert len(calls) == 1
    assert not calls[0].cancelled
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "reserved"


def test_audited_resume_of_a_terminal_failed_call_gets_a_fresh_ticket(setup):
    ledger, _, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    calls[0].get_error = RuntimeError("401 Unauthorized")
    assert coordinator().recover(spawn)["status"] == "blocked"
    function_timeout = type("FunctionTimeoutError", (Exception,), {})
    calls[0].get_error = function_timeout("the prior invocation has ended")
    assert coordinator().recover(spawn)["status"] == "blocked"
    reason = "Credentials repaired and the prior invocation is terminal"
    resumed = coordinator()
    result = resumed.launch("calibration", {"seed": 0}, spawn, resume_reason=reason)
    assert result["status"] == "running"
    assert len(calls) == 2
    assert tickets[0]["reservation_id"] != tickets[1]["reservation_id"]
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "settled"
    assert rows(ledger)[tickets[1]["reservation_id"]]["status"] == "reserved"
    assert reason in resumed.path.read_text()


def test_explicit_resume_observes_a_newly_blocked_worker_before_watchdog_has_polled(setup):
    ledger, clock, calls, tickets, write_state, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    write_state(status="blocked", reason="worker encountered a repaired configuration problem")
    clock.now += 10
    calls[0].lease.finish(elapsed_seconds=10)
    calls[0].response = {"status": "blocked"}
    calls[0].complete = True
    # No recover() call has yet copied the worker's status into the journal.
    result = coordinator().launch(
        "calibration", {"seed": 0}, spawn, resume_reason="Configuration repair verified"
    )
    assert result["status"] == "running"
    assert len(calls) == 2
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == "settled"
    assert rows(ledger)[tickets[1]["reservation_id"]]["status"] == "reserved"


def test_next_stage_observes_unpolled_completion_and_settles_before_dispatch(setup):
    ledger, clock, calls, tickets, write_state, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    old_call, old_ticket = calls[0], tickets[0]
    clock.now += 10
    old_call.lease.finish(elapsed_seconds=10)
    old_call.complete = True

    def spawn_controls(stage, config, ticket):
        assert stage == "controls" and config == {"seed": 1}
        assert rows(ledger)[old_ticket["reservation_id"]]["status"] == "settled"
        call = Call("fc-controls")
        call.lease = DriverBudgetLease.claim(ledger, ticket, owner_id=call.object_id, clock=clock)
        calls.append(call)
        tickets.append(ticket)
        write_state(status="running", heartbeat=clock(), active_stage="controls")
        return call

    # Deliberately omit recover(): stage transitions must observe provider completion themselves.
    result = coordinator().launch("controls", {"seed": 1}, spawn_controls)
    assert result["status"] == "running" and result["stage"] == "controls"
    assert len(calls) == len(rows(ledger)) == 2
    assert old_call.events == ["get"]
    assert rows(ledger)[tickets[1]["reservation_id"]]["status"] == "reserved"


@pytest.mark.parametrize("provider_state", ["running", "unknown", "failed"])
def test_next_stage_cannot_cancel_restart_or_skip_an_unfinished_predecessor(setup, provider_state):
    ledger, clock, calls, tickets, _, spawn, coordinator, _ = setup
    start(coordinator, spawn)
    # A request for another stage must not trigger the stale-owner restart workflow.
    clock.now += 601
    if provider_state == "unknown":
        calls[0].get_error = ConnectionError("provider status cannot be read")
    elif provider_state == "failed":
        failure = type("FunctionTimeoutError", (Exception,), {})
        calls[0].get_error = failure("prior stage did not complete")
    attempted_dispatches = []

    def must_not_spawn(*args):
        attempted_dispatches.append(args)
        raise AssertionError("A different-stage request must not dispatch replacement work")

    with pytest.raises(ValueError):
        coordinator().launch("controls", {"seed": 1}, must_not_spawn)
    assert not attempted_dispatches
    assert len(calls) == len(rows(ledger)) == 1
    assert calls[0].events == ["get"]
    expected_status = "settled" if provider_state == "failed" else "reserved"
    assert rows(ledger)[tickets[0]["reservation_id"]]["status"] == expected_status
