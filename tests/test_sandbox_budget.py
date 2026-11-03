import threading
import time
from decimal import Decimal

import pytest

from openwsp.backends.sandbox_budget import GIB, SandboxBudgetLease, SandboxLeaseError
from openwsp.budget import BudgetExceeded, BudgetLedger, compute_cost


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeSandbox:
    def __init__(self):
        self.terminations = 0
        self.fail = False

    def terminate(self, wait=True):
        self.terminations += 1
        if self.fail:
            raise TimeoutError("Uncertain termination")


class FakeTransport:
    sandbox_id = "sandbox-test"

    def __init__(self, clock, ledger):
        self.clock, self.ledger = clock, ledger
        self.sandbox = FakeSandbox()
        self.before_activity = None
        self.peak = 8 * GIB
        self.verified = True
        self.status_delay = 2
        self.status_error = None
        self.snapshots = []
        self.status_calls = 0
        self.closed = False

    def status(self):
        self.before_activity("stdin")
        self.status_calls += 1
        self.snapshots.append(self.ledger.snapshot())
        self.clock.advance(self.status_delay)
        if self.status_error:
            raise self.status_error
        return {"ok": True, "usage": {"cgroup_memory_peak_bytes": self.peak,
                                       "cgroup_memory_limit_bytes": 96 * GIB,
                                       "cgroup_memory_scope_verified": self.verified}}

    def close(self):
        self.before_activity("exec", timeout_seconds=55)
        self.clock.advance(3)
        self.sandbox.terminate(wait=True)
        self.closed = True


def setup(tmp_path, **kwargs):
    clock = Clock()
    ledger = BudgetLedger(tmp_path / "budget.json")
    transport = FakeTransport(clock, ledger)
    lease = SandboxBudgetLease(ledger, tmp_path / "lease.json", clock=clock, background=False, **kwargs)
    def create():
        assert ledger.snapshot()["summary"]["modal"]["reserved_usd"] > 0
        clock.advance(20)
        return transport
    return clock, ledger, transport, lease, create


def test_reserves_before_creation_and_overlaps_before_status(tmp_path):
    clock, ledger, transport, lease, create = setup(tmp_path)
    assert lease.start(create) is transport
    rows = ledger.snapshot()["reservations"]
    assert sorted(row["status"] for row in rows.values()) == ["reserved", "settled"]
    at_status = transport.snapshots[0]["summary"]["modal"]
    assert at_status["reserved_usd"] == pytest.approx(float(compute_cost(600, sandbox=True, memory_gib=96)))
    startup = next(row for row in rows.values() if row["status"] == "settled")
    assert startup["seconds"] == 20
    assert Decimal(startup["actual_usd"]) == compute_cost(20, sandbox=True, memory_gib=8)
    assert startup["receipt_metadata"]["invoice"] is False
    assert clock() == 22


def test_settles_at_verified_lifetime_peak_and_partitions_before_rpc(tmp_path):
    clock, ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    clock.advance(58)
    transport.peak = 16 * GIB
    lease.heartbeat()
    rows = sorted(ledger.snapshot()["reservations"].values(), key=lambda row: row["created_at"])
    settled = [r for r in rows if r["status"] == "settled"]
    assert settled[-1]["seconds"] == 60
    assert Decimal(settled[-1]["actual_usd"]) == compute_cost(60, sandbox=True, memory_gib=16)
    assert len([r for r in rows if r["status"] == "reserved"]) == 1


def test_unverified_or_decreasing_peak_uses_configured_hard_limit(tmp_path):
    clock, ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    clock.advance(10)
    transport.peak = 7 * GIB  # lifetime peak must not shrink
    lease.heartbeat()
    settled = sorted((r for r in ledger.snapshot()["reservations"].values() if r["status"] == "settled"), key=lambda row: row["created_at"])
    assert settled[-1]["receipt_metadata"]["memory_gib_bound"] == 96
    clock.advance(10)
    transport.peak = 9 * GIB
    lease.heartbeat()
    settled = sorted((r for r in ledger.snapshot()["reservations"].values() if r["status"] == "settled"), key=lambda row: row["created_at"])
    assert settled[-1]["receipt_metadata"]["memory_gib_bound"] == 96


def test_no_observability_charges_hard_limit_not_rss(tmp_path):
    _clock, ledger, transport, lease, create = setup(tmp_path)
    transport.verified = False
    lease.start(create)
    row = next(r for r in ledger.snapshot()["reservations"].values() if r["status"] == "settled")
    assert Decimal(row["actual_usd"]) == compute_cost(20, sandbox=True, memory_gib=96)


def test_budget_failure_terminates_without_new_stdin_and_keeps_previous_bound(tmp_path):
    clock, ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    available = ledger.snapshot()["summary"]["modal"]["available_usd"]
    ledger.reserve("modal", available - 0.1, "other-work")
    clock.advance(58)
    previous_calls = transport.status_calls
    with pytest.raises(BudgetExceeded):
        lease.heartbeat()
    assert transport.status_calls == previous_calls
    assert transport.sandbox.terminations == 1
    assert lease.failure
    with pytest.raises(SandboxLeaseError, match="stopped"):
        lease.check()
    rows = ledger.snapshot()["reservations"].values()
    assert any(r["stage"].endswith(":heartbeat") and r["status"] == "reserved" for r in rows)


def test_unknown_status_failure_keeps_both_reservations(tmp_path):
    clock, ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    clock.advance(58)
    transport.status_error = TimeoutError("Response lost after stdin may have renewed idle")
    transport.sandbox.fail = True
    with pytest.raises(TimeoutError):
        lease.heartbeat()
    active = [r for r in ledger.snapshot()["reservations"].values() if r["status"] == "reserved"]
    assert len(active) == 2
    lease.close()
    assert len([r for r in ledger.snapshot()["reservations"].values() if r["status"] == "reserved"]) == 2


def test_guard_renews_before_late_activity_and_refuses_unbounded_exec(tmp_path):
    clock, _ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    clock.advance(60)
    lease.check("stdin")
    assert transport.status_calls == 2
    with pytest.raises(SandboxLeaseError, match="unbounded"):
        lease.check("filesystem")
    with pytest.raises(SandboxLeaseError, match="unbounded"):
        lease.check("exec", timeout_seconds=56)
    lease.check("exec", timeout_seconds=55)


def test_close_observes_prior_interval_then_uses_hard_limit_for_tail(tmp_path):
    clock, ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    clock.advance(20)
    lease.close()
    assert transport.closed
    rows = sorted(ledger.snapshot()["reservations"].values(), key=lambda row: row["created_at"])
    assert all(r["status"] == "settled" for r in rows)
    tail = rows[-1]
    assert tail["seconds"] == 5  # status round trip + sync and confirmed termination
    assert Decimal(tail["actual_usd"]) == compute_cost(5, sandbox=True, memory_gib=96)
    assert tail["receipt_metadata"]["memory_basis"] == "termination_tail_hard_limit"
    lease.close()  # idempotent
    assert transport.sandbox.terminations == 1


def test_expired_reservation_cannot_renew_idle(tmp_path):
    clock, _ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    clock.advance(301)
    with pytest.raises(SandboxLeaseError, match="expired"):
        lease.heartbeat()
    assert transport.status_calls == 1
    assert transport.sandbox.terminations == 1


def test_creation_uncertain_keeps_entire_startup_reservation(tmp_path):
    _clock, ledger, _transport, lease, _create = setup(tmp_path)
    def fail():
        raise TimeoutError("creation response lost")
    with pytest.raises(TimeoutError):
        lease.start(fail)
    rows = sorted(ledger.snapshot()["reservations"].values(), key=lambda row: row["created_at"])
    assert len(rows) == 1 and rows[0]["status"] == "reserved"


def test_background_controller_renews_and_stops_on_close(tmp_path):
    class RealtimeClock:
        def __call__(self):
            return time.monotonic()

        def advance(self, _seconds):
            return None

    clock = RealtimeClock()
    ledger = BudgetLedger(tmp_path / "budget.json")
    transport = FakeTransport(clock, ledger)
    renewed = threading.Event()
    original_status = transport.status

    def status():
        result = original_status()
        if transport.status_calls >= 2:
            renewed.set()
        return result

    transport.status = status
    lease = SandboxBudgetLease(ledger, tmp_path / "lease.json", clock=clock,
                               heartbeat_interval=0.02, background=True)
    lease.start(lambda: transport)
    assert renewed.wait(timeout=1), "Background controller did not renew its reservation"
    lease.close()
    assert not lease._thread.is_alive()
    assert ledger.snapshot()["summary"]["modal"]["reserved_usd"] == 0


def test_replacement_lease_retains_orphan_reservations(tmp_path):
    clock, ledger, _transport, lease, create = setup(tmp_path)
    lease.start(create)
    old_id = next(k for k, r in ledger.snapshot()["reservations"].items() if r["status"] == "reserved")
    replacement = SandboxBudgetLease(ledger, tmp_path / "lease.json", clock=clock, background=False)
    assert old_id in replacement._orphaned
    assert ledger.snapshot()["reservations"][old_id]["status"] == "reserved"


def test_reattach_charges_prior_uncertain_horizon_and_funds_before_probe(tmp_path):
    _clock, ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    old_id = next(k for k, r in ledger.snapshot()["reservations"].items() if r["status"] == "reserved")
    committed = []
    replacement = SandboxBudgetLease(ledger, tmp_path / "lease.json", clock=Clock(), background=False,
                                     commit=lambda: committed.append(ledger.snapshot()))
    def attach(target):
        assert target == transport.sandbox_id
        assert committed[-1]["summary"]["modal"]["reserved_usd"] > 0
        return transport
    replacement.attach(attach)
    old = ledger.snapshot()["reservations"][old_id]
    assert old["status"] == "settled"
    assert old["actual_usd"] == old["upper_bound_usd"]
    assert old["receipt_metadata"]["may_overlap_later_reservations"] is True
    assert transport.before_activity == replacement.check
    replacement.close()


def test_reattach_requires_journaled_sandbox_identity(tmp_path):
    clock, ledger, transport, lease, create = setup(tmp_path)
    lease.start(create)
    replacement = SandboxBudgetLease(ledger, tmp_path / "lease.json", clock=clock, background=False)
    with pytest.raises(SandboxLeaseError, match="same sandbox"):
        replacement.attach(lambda _: transport, sandbox_id="different-sandbox")


@pytest.mark.parametrize("kwargs", [{"heartbeat_interval": 90}, {"max_exec_seconds": 300},
                                    {"idle_timeout": 600}, {"lease_seconds": 290}])
def test_rejects_configs_that_cannot_bound_idle_tail(tmp_path, kwargs):
    with pytest.raises(ValueError):
        setup(tmp_path, **kwargs)
