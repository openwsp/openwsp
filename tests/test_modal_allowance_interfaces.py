"""Allowance-only interfaces use fake RPCs; no Modal app import or provider calls."""

import ast
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from openwsp import budget, cli, replication
from openwsp.backends import coordinator, modal_runtime


@pytest.fixture
def observation():
    return {
        "observation_id": "dashboard-observation-001",
        "cycle_id": "billing-cycle-fixture",
        "cycle_limit": "150",
        "cycle_usage": "42.05",
        "evidence": {
            "source": "user dashboard confirmation",
            "basis": "user_confirmed_dashboard",
            "observed_at": datetime.now(UTC).isoformat(),
        },
    }


@pytest.fixture
def task_authorization():
    return {
        "total_limit": "225",
        "authorization_id": "modal-task-cap-225",
        "evidence": {
            "source": "explicit operator reply",
            "basis": "explicit_user_authorization",
            "authorized_at": datetime.now(UTC).isoformat(),
        },
    }


def _launch_function(volume):
    source = Path(__file__).resolve().parents[1] / "src/openwsp/backends/modal_app.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "launch"
    )
    decorators = {}

    def decorate(name, **options):
        decorators[name] = options
        return lambda function: function

    def forbidden(*args, **kwargs):
        pytest.fail("Allowance recording must never spawn a pipeline")

    namespace = {
        "app": SimpleNamespace(function=lambda **kw: decorate("function", **kw)),
        "modal": SimpleNamespace(concurrent=lambda **kw: decorate("concurrent", **kw)),
        "driver_image": object(),
        "data_volume": volume,
        "pipeline": SimpleNamespace(spawn=forbidden),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace["launch"], decorators


@pytest.fixture
def remote_harness(monkeypatch):
    calls = []
    existing = object()
    receipt = {"status": "allowance_recorded", "observation_id": "dashboard-observation-001"}

    def forbidden(*args, **kwargs):
        pytest.fail("Allowance-only branch must not initialize, reserve, launch, or recover")

    def reload():
        calls.append("reload")

    volume = SimpleNamespace(reload=reload, commit=lambda: calls.append("commit"))

    class ExistingLedger:
        __init__ = forbidden

        @classmethod
        def from_existing(cls, path):
            assert calls == ["reload"]
            assert str(path) == "/data/budget.json"
            calls.append("existing_ledger")
            return existing

    class FakeCoordinator:
        def __init__(self, ledger, path, *, state_path, commit, reload):
            assert ledger is existing and not callable(ledger)
            assert str(path) == "/data/coordinator.json"
            assert str(state_path) == "/data/state.json"
            assert commit is volume.commit and reload is volume.reload
            calls.append("coordinator")

        def observe_modal_allowance(self, value, **kwargs):
            event = ("observe", copy.deepcopy(value))
            calls.append((*event, copy.deepcopy(kwargs)) if kwargs else event)
            return receipt

        launch = recover = forbidden

    monkeypatch.setattr(budget, "BudgetLedger", ExistingLedger)
    monkeypatch.setattr(coordinator, "DriverCoordinator", FakeCoordinator)
    function, decorators = _launch_function(volume)
    return SimpleNamespace(
        launch=function,
        decorators=decorators,
        calls=calls,
        volume=volume,
        ledger=ExistingLedger,
        receipt=receipt,
    )


def test_remote_allowance_branch_only_reloads_existing_ledger_and_records(remote_harness, observation):
    original = copy.deepcopy(observation)
    assert remote_harness.launch(allowance_observation=observation) is remote_harness.receipt
    assert remote_harness.calls == ["reload", "existing_ledger", "coordinator", ("observe", original)]
    assert observation == original
    assert remote_harness.decorators["concurrent"] == {"max_inputs": 1}
    assert remote_harness.decorators["function"]["max_containers"] == 1


def test_remote_combined_funding_update_uses_same_serialized_owner(
    remote_harness, observation, task_authorization
):
    assert remote_harness.launch(
        allowance_observation=observation, modal_task_authorization=task_authorization,
    ) is remote_harness.receipt
    assert remote_harness.calls == [
        "reload", "existing_ledger", "coordinator",
        ("observe", observation, {"task_authorization": task_authorization}),
    ]


def test_task_authorization_without_allowance_cannot_fall_through_to_dispatch(remote_harness, task_authorization):
    with pytest.raises(ValueError):
        remote_harness.launch(modal_task_authorization=task_authorization)
    assert remote_harness.calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"stage": "live"},
        {"stage": None},
        {"config": {}},
        {"config": {"resume_reason": "reviewed"}},
        {"recover": True},
        {"recover": None},
        {"recover": 0},
    ],
)
def test_allowance_cannot_be_combined_with_execution_before_any_volume_access(
    remote_harness, observation, options
):
    with pytest.raises(ValueError):
        remote_harness.launch(allowance_observation=observation, **options)
    assert remote_harness.calls == []


def test_missing_existing_ledger_does_not_initialize_or_construct_coordinator(
    remote_harness, observation, monkeypatch
):
    def missing(path):
        raise ValueError("An existing budget ledger is required")

    monkeypatch.setattr(remote_harness.ledger, "from_existing", missing)
    with pytest.raises(ValueError, match="existing budget ledger"):
        remote_harness.launch(allowance_observation=observation)
    assert remote_harness.calls == ["reload"]


def test_failed_volume_reload_never_reads_ledger_or_records(remote_harness, observation):
    failure = OSError("volume unavailable")

    def failed():
        raise failure

    remote_harness.volume.reload = failed
    with pytest.raises(OSError) as raised:
        remote_harness.launch(allowance_observation=observation)
    assert raised.value is failure
    assert remote_harness.calls == []


@pytest.fixture
def fake_rpc(monkeypatch):
    calls = []
    result = {"status": "allowance_recorded", "observation_id": "dashboard-observation-001"}

    def forbidden(*args, **kwargs):
        pytest.fail("Recording an allowance must not deploy or inspect a replication checkout")

    monkeypatch.setattr(replication, "repository", forbidden)
    monkeypatch.setattr(replication.subprocess, "run", forbidden)
    monkeypatch.setattr(modal_runtime, "configure_tls", lambda: calls.append("tls"))

    def remote(*args, **kwargs):
        calls.append(("rpc", args, copy.deepcopy(kwargs)))
        return result

    def from_name(app_name, function_name):
        calls.append(("function", app_name, function_name))
        return SimpleNamespace(remote=remote)

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Function=SimpleNamespace(from_name=from_name)))
    return SimpleNamespace(calls=calls, result=result)


def test_local_entrypoint_validates_then_makes_exactly_one_allowance_rpc(tmp_path, observation, fake_rpc):
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(observation))
    assert replication.record_modal_allowance(observation=path) is fake_rpc.result
    assert fake_rpc.calls == [
        "tls",
        ("function", "openwsp-exp8", "launch"),
        ("rpc", (), {"allowance_observation": observation}),
    ]
    assert json.loads(path.read_text()) == observation


def test_local_combined_funding_validates_both_files_then_makes_one_rpc(
    tmp_path, observation, task_authorization, fake_rpc
):
    path, authorization_path = tmp_path / "observation.json", tmp_path / "authorization.json"
    path.write_text(json.dumps(observation))
    authorization_path.write_text(json.dumps(task_authorization))
    assert replication.record_modal_allowance(
        observation=path, task_authorization=authorization_path,
    ) is fake_rpc.result
    assert fake_rpc.calls == [
        "tls", ("function", "openwsp-exp8", "launch"),
        ("rpc", (), {"allowance_observation": observation, "modal_task_authorization": task_authorization}),
    ]


@pytest.mark.parametrize("invalid", ["relative", "malformed", "inferred", "missing_id", "extra_key"])
def test_invalid_task_authorization_fails_before_any_rpc(
    tmp_path, observation, task_authorization, fake_rpc, invalid
):
    path, authorization_path = tmp_path / "observation.json", tmp_path / "authorization.json"
    path.write_text(json.dumps(observation))
    if invalid == "inferred":
        task_authorization["evidence"]["basis"] = "estimated affordability"
    elif invalid == "missing_id":
        task_authorization.pop("authorization_id")
    elif invalid == "extra_key":
        task_authorization["recover"] = True
    authorization_path.write_text("{" if invalid == "malformed" else json.dumps(task_authorization))
    with pytest.raises(ValueError):
        replication.record_modal_allowance(
            observation=path,
            task_authorization="authorization.json" if invalid == "relative" else authorization_path,
        )
    assert fake_rpc.calls == []


@pytest.mark.parametrize(
    "invalid",
    [
        "json",
        "not_object",
        "no_evidence",
        "no_id",
        "inferred_evidence",
        "non_utc",
        "negative",
        "extra_key",
    ],
)
def test_malformed_observation_rejected_locally_before_tls_or_rpc(tmp_path, observation, fake_rpc, invalid):
    if invalid == "not_object":
        observation = []
    elif invalid == "no_evidence":
        observation.pop("evidence")
    elif invalid == "no_id":
        observation.pop("observation_id")
    elif invalid == "inferred_evidence":
        observation["evidence"]["basis"] = "predicted_after_payment"
    elif invalid == "non_utc":
        observation["evidence"]["observed_at"] = "2026-11-03T10:00:00+02:00"
    elif invalid == "negative":
        observation["cycle_usage"] = "-1"
    elif invalid == "extra_key":
        observation["recover"] = True
    path = tmp_path / "observation.json"
    path.write_text("{" if invalid == "json" else json.dumps(observation))
    with pytest.raises(ValueError):
        replication.record_modal_allowance(observation=path)
    assert fake_rpc.calls == []


def test_other_experiment_rejected_before_reading_observation_or_rpc(tmp_path, fake_rpc):
    with pytest.raises(ValueError, match="exp8|Unknown"):
        replication.record_modal_allowance("exp9", observation=tmp_path / "does-not-exist.json")
    assert fake_rpc.calls == []


def test_relative_observation_path_rejected_before_tls_or_rpc(fake_rpc):
    with pytest.raises(ValueError, match="absolute"):
        replication.record_modal_allowance(observation="observation.json")
    assert fake_rpc.calls == []


@pytest.mark.parametrize("experiment_args", [[], ["exp8"]])
def test_cli_routes_only_observation_and_default_experiment(tmp_path, monkeypatch, experiment_args):
    path = tmp_path / "observation.json"
    path.write_text("{}")
    calls = []

    def record(experiment="exp8", *, observation):
        calls.append((experiment, Path(observation)))
        return {"status": "allowance_recorded"}

    monkeypatch.setattr(replication, "record_modal_allowance", record)
    result = CliRunner().invoke(
        cli.app, ["record-modal-allowance", *experiment_args, "--observation", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"status": "allowance_recorded"}
    assert calls == [("exp8", path)]


def test_cli_routes_optional_task_authorization_without_execution_flags(tmp_path, monkeypatch):
    path, authorization_path = tmp_path / "observation.json", tmp_path / "authorization.json"
    calls = []

    def record(experiment="exp8", *, observation, task_authorization):
        calls.append((experiment, Path(observation), Path(task_authorization)))
        return {"status": "allowance_recorded"}

    monkeypatch.setattr(replication, "record_modal_allowance", record)
    result = CliRunner().invoke(cli.app, [
        "record-modal-allowance", "exp8", "--observation", str(path),
        "--task-authorization", str(authorization_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls == [("exp8", path, authorization_path)]


def test_cli_requires_explicit_observation_file(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No observation file must not reach allowance RPC")

    monkeypatch.setattr(replication, "record_modal_allowance", forbidden)
    result = CliRunner().invoke(cli.app, ["record-modal-allowance"])
    assert result.exit_code == 2
    assert "observation" in result.output


@pytest.mark.parametrize(
    "extra", [["--stage", "live"], ["--recover"], ["--config", "other.json"], ["--foreground"]]
)
def test_cli_does_not_accept_execution_flags(tmp_path, monkeypatch, extra):
    path = tmp_path / "observation.json"
    path.write_text("{}")

    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected execution flags must not reach allowance RPC")

    monkeypatch.setattr(replication, "record_modal_allowance", forbidden)
    result = CliRunner().invoke(cli.app, ["record-modal-allowance", "--observation", str(path), *extra])
    assert result.exit_code == 2
