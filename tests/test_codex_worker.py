"""Local Codex worker tests with fake editors and no provider or authentication I/O."""

import copy
import json
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from openwsp import replication
from openwsp.backends import codex_worker
from openwsp.backends.codex_worker import CodexWorker, split_codex_settings
from openwsp.cli import app
from openwsp.editors.bridge import BridgeConflict, BridgeStore


class PollPending(Exception):
    pass


@pytest.fixture(autouse=True)
def fake_pending_timeout(monkeypatch):
    monkeypatch.setattr(codex_worker, "_pending_timeout", lambda error: type(error) is PollPending)


@pytest.fixture
def local_settings(tmp_path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    return {
        "home": str(home),
        "state_dir": str(tmp_path / "state"),
        "work_dir": str(tmp_path / "work"),
        "executable": "fake-codex",
    }


class EditorFactory:
    def __init__(self):
        self.calls = []
        self.editors = {}

    def __call__(self, state, **kwargs):
        state = Path(state)
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        Path(kwargs["work_dir"]).mkdir(mode=0o700, parents=True, exist_ok=True)
        self.calls.append((state, kwargs))
        return self.editors.setdefault(state, FakeEditor())


class FakeEditor:
    def __init__(self):
        self.records = {}
        self.generations = 0
        self.last_response = None

    def has_request(self, key):
        return key in self.records

    def complete(self, messages, *, request_key):
        if request_key not in self.records:
            self.generations += 1
            receipt = {"status": "completed", "billing": "subscription", "usage": {"output_tokens": 2}}
            self.records[request_key] = (copy.deepcopy(messages), "Final answer", receipt)
        original, text, self.last_response = self.records[request_key]
        assert original == messages
        return text


class DriverCall:
    """Emulate a remote driver submitting one request after the prior completion."""

    def __init__(self, store, requests, *, run_id="run-a"):
        self.store, self.run_id = store, run_id
        self.requests = iter(requests)
        self.current = None
        self.polls = 0
        self.result = {"status": "complete", "run_id": run_id}

    def get(self, *, timeout):
        assert timeout == 0
        self.polls += 1
        if self.current:
            record = self.store.get(self.run_id, self.current)
            if record["status"] == "blocked":
                return {"status": "blocked", "request_key": self.current}
            if record["status"] != "completed":
                raise PollPending()
        try:
            key, conversation, message = next(self.requests)
        except StopIteration:
            return self.result
        self.store.submit(self.run_id, key, conversation_index=conversation, messages=message)
        self.current = key
        raise PollPending()


def test_split_preserves_api_default_and_never_mutates_caller_config(local_settings):
    default = {"seed": 5, "resources": {"gpu": "H100"}}
    remote, local = split_codex_settings(default, detach=True)
    assert remote == default and local is None
    remote["resources"]["gpu"] = "L4"
    assert default["resources"]["gpu"] == "H100"
    config = {"editor": "codex", "run_id": "run-a", "codex_local": local_settings, "seed": 5}
    before = copy.deepcopy(config)
    remote, local = split_codex_settings(config, detach=False)
    assert remote == {"editor": "codex", "run_id": "run-a", "seed": 5}
    assert local == local_settings
    local["home"] = "/different"
    assert config == before


@pytest.mark.parametrize("run_id", [None, "", "../escape", "Run-a", "a/b", "a" * 65, 123])
def test_malformed_run_id_rejected(local_settings, run_id):
    with pytest.raises(ValueError, match="run_id"):
        split_codex_settings(
            {"editor": "codex", "run_id": run_id, "codex_local": local_settings}, detach=False
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("home", "relative"),
        ("state_dir", "relative"),
        ("work_dir", "relative"),
        ("home", None),
        ("state_dir", "/tmp/bad\x00path"),
        ("executable", ""),
        ("executable", "relative/codex"),
        ("api_key", "DO-NOT-TRANSMIT"),
        ("environment", {"TOKEN": "DO-NOT-TRANSMIT"}),
    ],
)
def test_invalid_or_secret_local_config_rejected(local_settings, key, value):
    local_settings[key] = value
    with pytest.raises(ValueError) as raised:
        split_codex_settings(
            {"editor": "codex", "run_id": "run-a", "codex_local": local_settings}, detach=False
        )
    assert "DO-NOT-TRANSMIT" not in str(raised.value)


def test_worker_identity_is_private_stable_exclusive_and_settings_bound(local_settings):
    first = CodexWorker("run-a", **local_settings)
    second = CodexWorker("run-a", **local_settings)
    with first.locked():
        identity = first.worker_id
        assert identity
        with pytest.raises(RuntimeError, match="already owns"), second.locked():
            pytest.fail("Second worker acquired an owned run")
        assert stat.S_IMODE(first.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(first.work.stat().st_mode) == 0o700
        assert stat.S_IMODE((first.root / "worker.json").stat().st_mode) == 0o600
    with second.locked():
        assert second.worker_id == identity
        assert second.work == first.work
    altered = CodexWorker("run-a", **{**local_settings, "executable": "different-codex"})
    with pytest.raises(ValueError, match="durable identity"), altered.locked():
        pytest.fail("Changed local settings reused an owner identity")


def test_worker_reuses_conversation_editor_and_isolates_handoffs(tmp_path, local_settings):
    store = BridgeStore(tmp_path / "bridge")
    factory = EditorFactory()
    sleeps = []
    worker = CodexWorker("run-a", **local_settings, store=store, editor_factory=factory, sleep=sleeps.append)
    call = DriverCall(store, [("t001", 0, "first"), ("t002", 0, "second"), ("t041", 1, "handoff")])
    with worker.locked():
        assert worker.run(call) == call.result
    assert len(factory.calls) == 2
    first_path, first_options = factory.calls[0]
    second_path, second_options = factory.calls[1]
    assert first_path != second_path
    assert first_options["work_dir"] != second_options["work_dir"]
    assert first_options["codex_home"] == second_options["codex_home"] == Path(local_settings["home"])
    assert [editor.generations for editor in factory.editors.values()] == [2, 1]
    assert sleeps == [5, 5, 5]
    for key in ("t001", "t002", "t041"):
        record = store.get("run-a", key)
        assert record["status"] == "completed"
        assert record["claimed_by"] == worker.worker_id
        assert record["completion"]["text"] == "Final answer"
    assert worker._lock is None


def test_lost_completion_upload_resumes_same_owner_without_regeneration(tmp_path, local_settings):
    store = BridgeStore(tmp_path / "bridge")
    factory = EditorFactory()

    class LostUpload:
        def __getattr__(self, name):
            return getattr(store, name)

        def finish(self, *args, **kwargs):
            raise ConnectionError("upload lost")

    first = CodexWorker(
        "run-a", **local_settings, store=LostUpload(), editor_factory=factory, sleep=lambda _: None
    )
    with pytest.raises(ConnectionError):
        first.run(DriverCall(store, [("t001", 0, "first")]))
    assert store.get("run-a", "t001")["status"] == "dispatched"
    resumed = CodexWorker(
        "run-a", **local_settings, store=store, editor_factory=factory, sleep=lambda _: None
    )
    assert resumed.run(DriverCall(store, [("t001", 0, "first")]))["status"] == "complete"
    assert resumed.worker_id == first.worker_id
    assert sum(editor.generations for editor in factory.editors.values()) == 1


def test_lost_local_journal_blocks_dispatched_request_without_inference(tmp_path, local_settings):
    store = BridgeStore(tmp_path / "bridge")
    worker = CodexWorker(
        "run-a", **local_settings, store=store, editor_factory=EditorFactory(), sleep=lambda _: None
    )
    record = store.submit("run-a", "t001", conversation_index=0, messages="first")
    with worker.locked():
        owner = {"worker_id": worker.worker_id, "request_hash": record["request_hash"]}
        store.claim("run-a", "t001", **owner)
        store.dispatch("run-a", "t001", **owner)
    assert worker.run(DriverCall(store, [("t001", 0, "first")]))["status"] == "blocked"
    assert store.get("run-a", "t001")["blocked_error"]["error_type"] == "MissingLocalRequest"
    assert sum(editor.generations for editor in worker.editor_factory.editors.values()) == 0


def test_remote_execution_error_is_not_treated_as_poll_pending(local_settings):
    def fail(*, timeout):
        raise RuntimeError("remote execution failed")

    def unexpected(*args):
        pytest.fail("Worker polled bridge after a terminal remote error")

    worker = CodexWorker("run-a", **local_settings, store=SimpleNamespace(pending=unexpected))
    with pytest.raises(RuntimeError, match="remote execution failed"):
        worker.run(SimpleNamespace(get=fail))
    assert worker._lock is None


@pytest.mark.parametrize(
    "record",
    [{"run_id": "another-run", "conversation_index": 0}, {"run_id": "run-a", "conversation_index": -1}],
)
def test_malformed_bridge_identity_never_creates_editor(local_settings, record):
    factory = EditorFactory()
    worker = CodexWorker(
        "run-a", **local_settings, store=SimpleNamespace(pending=lambda _: [record]), editor_factory=factory
    )

    def pending(*, timeout):
        raise PollPending()

    with pytest.raises(BridgeConflict):
        worker.run(SimpleNamespace(get=pending))
    assert not factory.calls


@pytest.mark.parametrize("invalid", ["detached", "secret"])
def test_invalid_codex_launch_rejected_before_repository_or_deployment(
    tmp_path, local_settings, monkeypatch, invalid
):
    config = {"editor": "codex", "run_id": "run-a", "codex_local": local_settings}
    if invalid == "secret":
        config["codex_local"]["api_key"] = "DO-NOT-TRANSMIT"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    def forbidden(*args, **kwargs):
        pytest.fail("Validation must precede repository lookup or deployment")

    monkeypatch.setattr(replication, "repository", forbidden)
    monkeypatch.setattr(replication.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        replication.replicate("exp8", config=path, detach=invalid == "detached")
    arguments = ["replicate", "exp8", "--config", str(path)]
    if invalid != "detached":
        arguments.append("--foreground")
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2
    assert "foreground" in result.output if invalid == "detached" else "codex_local" in result.output
    assert "DO-NOT-TRANSMIT" not in result.output


@pytest.mark.parametrize("codex,interrupt", [(True, False), (False, False), (True, True)])
def test_foreground_replication_keeps_auth_local_and_locks_before_deploy(
    tmp_path, local_settings, monkeypatch, codex, interrupt
):
    root = tmp_path / "repo"
    for name in ("src", "replications", "scripts"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(replication, "repository", lambda: root)
    from openwsp.backends import modal_runtime

    monkeypatch.setattr(modal_runtime, "configure_tls", lambda: None)
    events = []
    local_received = []
    remote_received = []
    call = object()

    class FakeWorker:
        def __init__(self, run_id, **kwargs):
            local_received.append((run_id, kwargs))
            self.held = False

        @contextmanager
        def locked(self):
            self.held = True
            events.append("lock")
            try:
                yield self
            finally:
                self.held = False
                events.append("unlock")

        def run(self, source_call):
            assert source_call is call and self.held
            events.append("work")
            assert json.loads((root / ".openwsp/launch.json").read_text())["call_id"] == "fc-test"
            if interrupt:
                raise KeyboardInterrupt()
            return {"status": "complete", "worker": "local"}

    monkeypatch.setattr(codex_worker, "CodexWorker", FakeWorker)

    def deploy(command, *, cwd, env, check):
        events.append("deploy")
        assert check is True
        if codex:
            assert events[:2] == ["lock", "deploy"]
            assert env["OPENWSP_EDITOR_TRANSPORT"] == "codex"
        else:
            assert env["OPENWSP_EDITOR_TRANSPORT"] == "openai"
        deployed_config = json.loads((cwd / "scripts/launch-config.json").read_text())
        assert "codex_local" not in deployed_config
        assert all(value not in json.dumps(deployed_config) for value in local_settings.values())
        assert "DO-NOT-TRANSMIT" not in json.dumps(command)

    monkeypatch.setattr(replication.subprocess, "run", deploy)

    def launch(stage, settings):
        events.append("launch")
        remote_received.append((stage, settings))
        assert "codex_local" not in settings
        assert all(value not in json.dumps(settings) for value in local_settings.values())
        return {"status": "running", "call_id": "fc-test"}

    def from_name(app_name, function_name):
        assert (app_name, function_name) == ("openwsp-exp8", "launch")
        return SimpleNamespace(remote=launch)

    def from_id(call_id):
        assert call_id == "fc-test"
        return call if codex else SimpleNamespace(get=lambda **kw: {"status": "complete", "worker": "api"})

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Function=SimpleNamespace(from_name=from_name),
            FunctionCall=SimpleNamespace(from_id=from_id),
        ),
    )
    settings = {"editor": "codex", "run_id": "run-a", "codex_local": local_settings} if codex else {}
    path = root / "scripts/launch-config.json"
    path.write_text(json.dumps(settings))
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            replication.replicate("exp8", stage="live", config=path, detach=False)
        result = json.loads((root / ".openwsp/launch.json").read_text())
        assert result == {"status": "running", "call_id": "fc-test", "experiment": "exp8"}
    else:
        result = replication.replicate("exp8", stage="live", config=path, detach=False)
        assert result["status"] == "complete"
    assert json.loads(path.read_text()) == settings
    assert json.loads((root / ".openwsp/launch.json").read_text()) == result
    if codex:
        assert local_received == [("run-a", local_settings)]
        assert remote_received == [("live", {"editor": "codex", "run_id": "run-a"})]
        assert events == ["lock", "deploy", "launch", "work", "unlock"]
    else:
        assert local_received == [] and remote_received == [("live", {})]
        assert events == ["deploy", "launch"]


@pytest.mark.parametrize("key", ["home", "state_dir", "work_dir"])
def test_local_codex_directories_cannot_enter_deployment_snapshot(tmp_path, local_settings, monkeypatch, key):
    root = tmp_path / "repo"
    unsafe = root / "src/private-local-state"
    unsafe.mkdir(parents=True, mode=0o700)
    local_settings[key] = str(unsafe)
    monkeypatch.setattr(replication, "repository", lambda: root)
    from openwsp.backends import modal_runtime

    def forbidden(*args, **kwargs):
        pytest.fail("Local directory admission must precede TLS setup and deployment")

    monkeypatch.setattr(modal_runtime, "configure_tls", forbidden)
    monkeypatch.setattr(replication.subprocess, "run", forbidden)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"editor": "codex", "run_id": "run-a", "codex_local": local_settings}))
    with pytest.raises(ValueError, match="outside the deployed source"):
        replication.replicate("exp8", config=path, detach=False)
