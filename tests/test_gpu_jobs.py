"""CPU checks for queue races, terminal confirmation, budgets and crash recovery."""

import json
from decimal import Decimal

import pytest

from openwsp.backends.gpu_jobs import GPUJobRunner
from openwsp.backends.jobs import JobBlocked, JobPending
from openwsp.backends.start_gate import StartGate
from openwsp.budget import BudgetExceeded, BudgetLedger, compute_cost


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class AtomicDict:
    def __init__(self):
        self.values = {}

    def put(self, key, value, *, skip_if_exists):
        assert skip_if_exists
        if key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key, default=None):
        return self.values.get(key, default)


class Call:
    def __init__(self, call_id, clock, *, ready=False):
        self.object_id = call_id
        self.clock = clock
        self.ready = ready
        self.cancelled = False
        self.terminal = False
        self.cancel_count = 0
        self.get_count = 0
        self.cancel_error = None
        self.confirmation_errors = []
        self.response = {"container_seconds": 10, "answer": call_id}

    def get(self, *, timeout):
        self.get_count += 1
        if self.cancelled:
            if self.confirmation_errors:
                raise self.confirmation_errors.pop(0)
            self.terminal = True
            raise RuntimeError("Queued attempt did not win its start gate: cancelled")
        if self.ready:
            self.terminal = True
            return self.response
        self.clock.sleep(timeout)
        raise TimeoutError("pending")

    def cancel(self, *, terminate_containers):
        assert terminate_containers
        self.cancel_count += 1
        self.cancelled = True
        if self.cancel_error:
            raise self.cancel_error


@pytest.fixture
def setup(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json")
    clock = Clock()
    gate = StartGate(AtomicDict(), clock=clock)
    calls = {}
    spawned = []
    commits = []

    def commit():
        commits.append(ledger.snapshot())

    def spawn(gpu, gate_key):
        assert commits
        assert ledger.snapshot()["summary"]["modal"]["reserved_usd"] > 0
        if gpu == "A100-80GB" and calls:
            assert calls["fc-0"].terminal
            old = next(row for row in ledger.snapshot()["reservations"].values()
                       if row.get("metadata", {}).get("cost_options", {}).get("gpu") == "H100")
            assert old["status"] == "settled" and Decimal(old["actual_usd"]) > 0
        call = Call(f"fc-{len(calls)}", clock, ready=gpu == "A100-80GB")
        calls[call.object_id] = call
        spawned.append((gpu, gate_key))
        return call

    def runner(**kwargs):
        return GPUJobRunner(ledger, tmp_path / "gpu", gate, commit=commit,
                            call_factory=lambda call_id: calls[call_id], clock=clock, sleep=clock.sleep, **kwargs)

    return ledger, clock, gate, calls, spawned, spawn, runner


def launch(runner, spawn, **kwargs):
    return runner.launch("probe/base", spawn, fingerprint={"model": "pinned", "input": [1, 2]}, **kwargs)


def test_queue_timeout_confirms_terminal_charges_h100_bound_then_funds_a100(setup):
    ledger, clock, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    result = jobs.result("probe/base")
    assert result["answer"] == "fc-1"
    assert clock.now >= 700
    assert calls["fc-0"].cancel_count == 1
    assert [gpu for gpu, _ in spawned] == ["H100", "A100-80GB"]
    assert spawned[0][1] != spawned[1][1]
    reservations = {row["metadata"]["cost_options"]["gpu"]: row
                    for row in ledger.snapshot()["reservations"].values()}
    assert Decimal(reservations["H100"]["actual_usd"]) == compute_cost(2520, gpu="H100", cpu_cores=4, memory_gib=16)
    assert Decimal(reservations["A100-80GB"]["actual_usd"]) == compute_cost(132, gpu="A100-80GB", cpu_cores=4,
                                                                       memory_gib=16)
    gets = calls["fc-1"].get_count
    assert jobs.result("probe/base") == result
    assert calls["fc-1"].get_count == gets


def test_started_worker_is_never_cancelled_during_long_compilation(setup):
    _, clock, gate, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    assert gate.try_start(spawned[0][1], "worker-1").may_start
    clock.sleep(601)
    with pytest.raises(JobPending):
        jobs.result("probe/base", timeout=30)
    assert calls["fc-0"].cancel_count == 0
    assert len(spawned) == 1
    calls["fc-0"].ready = True
    assert jobs.result("probe/base")["answer"] == "fc-0"


def test_resume_requires_factory_and_reuses_saved_child_call(setup):
    _, _, _, calls, spawned, spawn, runner = setup
    launch(runner(), spawn)
    resumed = runner()
    with pytest.raises(JobBlocked, match="launch"):
        resumed.result("probe/base")
    assert launch(resumed, spawn) == "fc-0"
    assert len(spawned) == 1
    calls["fc-0"].ready = True
    assert resumed.result("probe/base")["answer"] == "fc-0"


def test_lost_dispatch_receipt_never_repeats_paid_spawn(setup):
    ledger, _, _, _, _, _, runner = setup
    attempts = []

    def uncertain(gpu, gate_key):
        attempts.append((gpu, gate_key))
        raise ConnectionError("accepted but acknowledgement lost")

    with pytest.raises(JobBlocked):
        launch(runner(), uncertain)
    with pytest.raises(JobBlocked):
        launch(runner(), uncertain)
    assert len(attempts) == 1
    assert len(ledger.snapshot()["reservations"]) == 1


def test_cancel_acknowledgement_loss_blocks_until_actual_get_confirms_terminal(setup):
    _, clock, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    calls["fc-0"].cancel_error = ConnectionError("uncertain cancellation")
    clock.sleep(601)
    with pytest.raises(JobPending, match="acknowledgement"):
        jobs.result("probe/base")
    assert len(spawned) == 1
    resumed = runner()
    launch(resumed, spawn)
    assert resumed.result("probe/base")["answer"] == "fc-1"
    assert calls["fc-0"].cancel_count == 1


def test_cancellation_get_timeout_does_not_authorize_fallback(setup):
    ledger, clock, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    calls["fc-0"].confirmation_errors = [ConnectionError("disconnected") for _ in range(10)]
    clock.sleep(601)
    with pytest.raises(JobPending):
        jobs.result("probe/base", timeout=60)
    assert len(spawned) == 1
    assert next(iter(ledger.snapshot()["reservations"].values()))["status"] == "reserved"
    calls["fc-0"].confirmation_errors.clear()
    assert jobs.result("probe/base")["answer"] == "fc-1"


@pytest.mark.parametrize("error", [RuntimeError("401 Unauthorized"), RuntimeError("quota exhausted"),
                                 RuntimeError("unclassified local SDK failure")])
def test_auth_quota_and_unclassified_confirmation_errors_stop_without_fallback(setup, error):
    ledger, clock, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    calls["fc-0"].confirmation_errors = [error]
    clock.sleep(601)
    with pytest.raises(JobBlocked):
        jobs.result("probe/base")
    assert len(spawned) == 1
    assert next(iter(ledger.snapshot()["reservations"].values()))["status"] == "reserved"
    with pytest.raises(JobBlocked):
        launch(runner(), spawn)


def test_missing_gate_does_not_cancel_a_possibly_running_job(setup):
    _, clock, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    jobs.gate = StartGate(None)
    launch(jobs, spawn)
    clock.sleep(601)
    with pytest.raises(JobBlocked, match="unknown_start_gate"):
        jobs.result("probe/base")
    assert calls["fc-0"].cancel_count == 0
    assert len(spawned) == 1


def test_fallback_must_fit_budget_after_charging_cancelled_h100(setup):
    ledger, _, _, calls, spawned, spawn, runner = setup
    # Enough for the H100 bound; not enough for that bound plus an A100 attempt.
    prior = ledger.reserve("modal", "38.90", "prior-work")
    ledger.settle(prior, "38.90")
    jobs = runner()
    launch(jobs, spawn)
    with pytest.raises(BudgetExceeded):
        jobs.result("probe/base")
    assert calls["fc-0"].terminal
    assert len(spawned) == 1
    assert all(row["status"] == "settled" for row in ledger.snapshot()["reservations"].values())


def test_resume_after_terminal_settlement_crash_does_not_charge_or_cancel_twice(setup, monkeypatch):
    ledger, _, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    settle = ledger.settle

    class Crash(BaseException):
        pass

    def settle_and_crash(*args, **kwargs):
        settle(*args, **kwargs)
        raise Crash()

    monkeypatch.setattr(ledger, "settle", settle_and_crash)
    with pytest.raises(Crash):
        jobs.result("probe/base")
    monkeypatch.setattr(ledger, "settle", settle)
    resumed = runner()
    launch(resumed, spawn)
    assert resumed.result("probe/base")["answer"] == "fc-1"
    assert len(spawned) == 2 and calls["fc-0"].cancel_count == 1
    settled = [event for event in ledger.snapshot()["events"] if event["event"] == "settled"]
    assert len(settled) == 2


def test_resume_after_child_call_id_saved_recovers_original_dispatch_time(setup, tmp_path, monkeypatch):
    _, clock, _, _, spawned, spawn, runner = setup
    jobs = runner()
    persist = jobs._persist

    class Crash(BaseException):
        pass

    def crash_mapping(path, record):
        if record["attempts"][-1].get("call_id"):
            raise Crash()
        persist(path, record)

    monkeypatch.setattr(jobs, "_persist", crash_mapping)
    with pytest.raises(Crash):
        launch(jobs, spawn)
    original = clock.now
    clock.sleep(601)
    resumed = runner()
    assert launch(resumed, spawn) == "fc-0"
    state = json.loads(next((tmp_path / "gpu").glob("*.gpu.json")).read_text())
    assert state["attempts"][0]["submitted_at"] == original
    assert len(spawned) == 1


def test_changed_fingerprint_or_resource_maximum_is_rejected(setup):
    _, _, _, _, spawned, spawn, runner = setup
    launch(runner(), spawn)
    with pytest.raises(ValueError, match="different"):
        runner().launch("probe/base", spawn, fingerprint="changed")
    with pytest.raises(ValueError, match="different"):
        launch(runner(memory_gib=32), spawn)
    assert len(spawned) == 1


def test_input_cancellation_baseexception_is_terminal_but_interrupt_is_not(setup):
    _, clock, _, calls, _, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    cancellation = type("InputCancellation", (BaseException,), {"__module__": "modal.exception"})
    calls["fc-0"].confirmation_errors = [KeyboardInterrupt(), cancellation()]
    clock.sleep(601)
    with pytest.raises(KeyboardInterrupt):
        jobs.result("probe/base")
    # Mark provider terminal when its synthetic terminal cancellation is delivered.
    calls["fc-0"].terminal = True
    assert jobs.result("probe/base")["answer"] == "fc-1"


def test_initial_a100_has_no_second_fallback_even_after_queue_deadline(setup):
    _, clock, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn, initial_gpu="A100-80GB")
    calls["fc-0"].ready = False
    clock.sleep(601)
    with pytest.raises(JobPending):
        jobs.result("probe/base", timeout=30)
    assert calls["fc-0"].cancel_count == 0
    assert [gpu for gpu, _ in spawned] == ["A100-80GB"]


def test_public_modal_remote_error_confirms_cancelled_get_but_service_error_does_not():
    from openwsp.backends.gpu_jobs import _terminal_cancel_error

    # ServiceError may mean a cancelled network request; it says nothing about
    # termination of the paid invocation. RemoteError comes from its result.
    terminal = type("RemoteError", (Exception,), {"__module__": "modal.exception"})
    transport = type("ServiceError", (Exception,), {"__module__": "modal.exception"})
    assert _terminal_cancel_error(terminal("Input terminated"))
    assert not _terminal_cancel_error(transport("CANCELLED"))
    assert not _terminal_cancel_error(TimeoutError("still pending"))


def test_returned_cancelled_result_is_terminal_evidence_before_fallback(setup, monkeypatch):
    _, clock, _, calls, _, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    old = calls["fc-0"]

    def get(*, timeout):
        assert old.cancelled
        old.terminal = True
        return {"cancelled_before_start": True}

    monkeypatch.setattr(old, "get", get)
    clock.sleep(601)
    assert jobs.result("probe/base")["answer"] == "fc-1"


def test_introspection_preserves_original_failure_classification_and_reservation(setup):
    _, _, _, calls, _, spawn, runner = setup
    jobs = runner()
    assert not jobs.has_job("probe/base")
    launch(jobs, spawn)
    assert jobs.has_job("probe/base")
    before = jobs.active_child_record("probe/base")
    assert before["call_id"] == "fc-0" and before["reservation_id"]

    def failed(*, timeout):
        raise RuntimeError("worker computation failed")

    calls["fc-0"].get = failed
    with pytest.raises(JobBlocked):
        jobs.result("probe/base")
    after = jobs.active_child_record("probe/base")
    assert after["failure"]["category"] == "job_failure"
    assert after["reservation_id"] == before["reservation_id"]
    assert after["_openwsp_gpu"]["wrapper_status"] == "blocked"


def test_result_metadata_totals_both_attempts_without_mutating_persisted_response(setup, tmp_path):
    ledger, _, _, _, _, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    response = jobs.result("probe/base")
    metadata = response["_openwsp_gpu"]
    assert metadata["actual_gpu"] == "A100-80GB"
    assert metadata["attempt_count"] == 2
    assert metadata["billable_seconds"] == 2520 + 132
    assert metadata["total_settled_cost_usd"] == pytest.approx(sum(
        float(row["actual_usd"]) for row in ledger.snapshot()["reservations"].values()))
    persisted = [json.loads(path.read_text()) for path in (tmp_path / "gpu" / "attempts").glob("*.result.json")]
    assert len(persisted) == 1 and "_openwsp_gpu" not in persisted[0]
    assert jobs.result("probe/base") == response


def test_introspection_of_unfunded_fallback_retains_planned_state(setup):
    ledger, _, _, _, _, spawn, runner = setup
    prior = ledger.reserve("modal", "38.90", "prior-work")
    ledger.settle(prior, "38.90")
    jobs = runner()
    launch(jobs, spawn)
    with pytest.raises(BudgetExceeded):
        jobs.result("probe/base")
    active = jobs.active_child_record("probe/base")
    assert active["status"] == "planned" and "reservation_id" not in active
    assert active["_openwsp_gpu"]["actual_gpu"] == "A100-80GB"


def test_resume_after_cancel_intent_commit_sends_the_missing_cancel_rpc(setup, monkeypatch):
    _, clock, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    persist = jobs._persist

    class Crash(BaseException):
        pass

    def crash_before_cancel(path, record):
        persist(path, record)
        if record["attempts"][-1]["status"] == "cancel_intent":
            raise Crash()

    monkeypatch.setattr(jobs, "_persist", crash_before_cancel)
    clock.sleep(601)
    with pytest.raises(Crash):
        jobs.result("probe/base")
    assert calls["fc-0"].cancel_count == 0 and len(spawned) == 1
    resumed = runner()
    launch(resumed, spawn)
    assert resumed.result("probe/base")["answer"] == "fc-1"
    assert calls["fc-0"].cancel_count == 1 and len(spawned) == 2


def test_resume_retries_cancel_that_failed_before_reaching_provider(setup, monkeypatch):
    _, clock, _, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)
    original_cancel = calls["fc-0"].cancel
    attempts = []

    def cancel(*, terminate_containers):
        attempts.append(True)
        if len(attempts) == 1:
            raise ConnectionError("connection failed before cancellation was sent")
        return original_cancel(terminate_containers=terminate_containers)

    monkeypatch.setattr(calls["fc-0"], "cancel", cancel)
    clock.sleep(601)
    with pytest.raises(JobPending):
        jobs.result("probe/base")
    assert not calls["fc-0"].cancelled and len(spawned) == 1
    resumed = runner()
    launch(resumed, spawn)
    assert resumed.result("probe/base")["answer"] == "fc-1"
    assert len(attempts) == 2


def test_resume_does_not_reissue_cancel_without_current_gate_authority(setup, monkeypatch):
    _, clock, gate, calls, spawned, spawn, runner = setup
    jobs = runner()
    launch(jobs, spawn)

    def cancel(*, terminate_containers):
        raise ConnectionError("not delivered")

    monkeypatch.setattr(calls["fc-0"], "cancel", cancel)
    clock.sleep(601)
    with pytest.raises(JobPending):
        jobs.result("probe/base")
    # Simulate an expired record; absence cannot grant cancellation authority.
    gate.backend.values.clear()
    resumed = runner()
    launch(resumed, spawn)
    with pytest.raises(JobBlocked, match="unknown_cancellation_authority"):
        resumed.result("probe/base")
    assert calls["fc-0"].cancel_count == 0 and len(spawned) == 1
