import errno
import json
import os
import queue
import socket
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from openwsp.backends.sandbox import (
    ModalSandboxTransport,
    SandboxTransportError,
    _inspect_work_snapshot,
    _probe_sandbox_isolation,
)


class FakeVolume:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def with_mount_options(self, **options):
        self.calls.append(options)
        return (self.name, options)


class FakeStdout:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        while True:
            line = self.queue.get()
            if line is None:
                return
            yield line

    def close(self):
        self.queue.put(None)


class FakeSandbox:
    def __init__(self):
        self.stdout = FakeStdout()
        self.stdout.queue.put(json.dumps({"type": "ready", "ok": True, "runs": 0}) + "\n")
        self.stdin = SimpleNamespace(write=self.write, drain=lambda: None)
        self.object_id = "sb-test"
        self.files = {}
        self.directories = []
        self.commands = []
        self.returncode = 0
        self.closed = False
        self.reloads = 0
        self.requests = []
        self.filesystem = SimpleNamespace(
            write_text=lambda text, path: self.files.__setitem__(path, text),
            read_text=lambda path: self.files[path],
            make_directory=self.directories.append,
            copy_from_local=lambda local, remote: self.files.__setitem__(remote, Path(local).read_text()),
            copy_to_local=lambda remote, local: Path(local).write_text(self.files[remote]),
        )

    def write(self, message):
        request = json.loads(message)
        self.requests.append(request)
        self.stdout.queue.put(
            json.dumps(
                {
                    "id": request["id"],
                    "ok": True,
                    "output": "test",
                    "runs": 1,
                    "restarted": request["op"] == "restart",
                    "variables_lost": request["op"] == "restart",
                }
            )
            + "\n"
        )

    def exec(self, *args, **kwargs):
        self.commands.append((args, kwargs))
        return SimpleNamespace(wait=lambda: self.returncode, returncode=self.returncode)

    def reload_volumes(self, **kwargs):
        self.reloads += 1

    def terminate(self, **kwargs):
        self.closed = True
        self.stdout.close()


@pytest.fixture
def sandbox_setup(monkeypatch):
    sandbox = FakeSandbox()
    calls = []

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return sandbox

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Sandbox=SimpleNamespace(create=create)))
    return sandbox, calls, [FakeVolume(name) for name in ("models", "practice", "work")]


def test_create_preserves_isolation_and_resource_limits(sandbox_setup):
    sandbox, calls, volumes = sandbox_setup
    transport = ModalSandboxTransport.create(
        "app", "image", *volumes, model_subpath="qwen", practice_subpath="exp8", work_subpath="run001"
    )
    args, options = calls[0]
    assert "--verify-read-only" in args
    assert options["block_network"] is True
    assert options["secrets"] == []
    assert options["include_oidc_identity_token"] is False
    assert options["cpu"] == (2, 2)
    assert options["memory"] == (8192, 98_304)
    assert options["timeout"] == 86_400
    assert options["idle_timeout"] == 180
    assert set(options["volumes"]) == {"/model", "/practice", "/work"}
    assert volumes[0].calls == [{"read_only": True, "sub_path": "qwen"}]
    assert volumes[1].calls == [{"read_only": True, "sub_path": "exp8"}]
    assert volumes[2].calls == [{"read_only": False, "sub_path": "run001"}]
    assert all("TOKEN" not in key or key == "TOKENIZERS_PARALLELISM" for key in options["env"])
    assert transport.run("1 + 1", request_id="turn001-cell001")["ok"]
    assert sandbox.requests[-1]["id"] == "turn001-cell001"
    assert transport.runs == 1
    transport.close()
    assert sandbox.closed


def test_commit_reload_and_artifact_paths(sandbox_setup, tmp_path):
    sandbox, _, volumes = sandbox_setup
    transport = ModalSandboxTransport.create("app", "image", *volumes)
    transport.write_text("notes.md", "hypothesis")
    assert transport.read_text("/work/notes.md") == "hypothesis"
    source = tmp_path / "source.txt"
    source.write_text("forward artifact")
    transport.upload(source, "fwd_1/meta.json")
    destination = tmp_path / "copy" / "artifact.txt"
    transport.download("fwd_1/meta.json", destination)
    assert destination.read_text() == "forward artifact"
    for path in ("../model/x", "/model/x", "/etc/passwd", "/work/../model/x", "/work"):
        with pytest.raises(ValueError, match="under /work"):
            transport.write_text(path, "bad")
    transport.commit_work()
    assert sandbox.commands[-1] == (("sync", "/work"), {"timeout": 55})
    transport.reload_work()
    assert sandbox.reloads == 1
    transport.close()


@pytest.mark.parametrize(
    "limits",
    [
        {"work_volume_version": 1},
        {"memory_limit_mib": 100_000},
        {"memory_request_mib": 0},
        {"cell_timeout": 901},
        {"cpu": 3},
        {"initial_runs": 301},
        {"max_runs": 301},
    ],
)
def test_invalid_limits_rejected_before_launch(sandbox_setup, limits):
    _, calls, volumes = sandbox_setup
    with pytest.raises(ValueError):
        ModalSandboxTransport.create("app", "image", *volumes, **limits)
    assert calls == []


def test_failed_commit_is_not_reported_as_success(sandbox_setup):
    sandbox, _, volumes = sandbox_setup
    transport = ModalSandboxTransport.create("app", "image", *volumes)
    sandbox.returncode = 1
    with pytest.raises(SandboxTransportError, match="commit failed"):
        transport.commit_work()
    with pytest.raises(SandboxTransportError, match="commit failed"):
        transport.close()
    assert sandbox.closed


def test_status_can_complete_while_run_is_waiting(sandbox_setup):
    sandbox, _, volumes = sandbox_setup
    transport = ModalSandboxTransport.create("app", "image", *volumes)
    started = threading.Event()
    original_write = sandbox.write

    def hold_run(message):
        if json.loads(message)["op"] == "run":
            started.set()
        else:
            original_write(message)

    sandbox.stdin.write = hold_run
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(transport.run, "slow native computation", request_id="slow")
        assert started.wait(timeout=1)
        assert transport.status()["ok"]
        assert not running.done()
        sandbox.stdout.queue.put(json.dumps({"id": "slow", "ok": True, "runs": 1, "output": "42"}) + "\n")
        assert running.result(timeout=1)["output"] == "42"
    transport.close()


def test_activity_guard_prevents_unreserved_requests_and_filesystem_exec(sandbox_setup, tmp_path):
    sandbox, _, volumes = sandbox_setup
    transport = ModalSandboxTransport.create("app", "image", *volumes)
    calls = []

    def guard(operation, timeout_seconds=None):
        calls.append((operation, timeout_seconds))
        raise RuntimeError("budget reservation unavailable")

    transport.before_activity = guard
    with pytest.raises(RuntimeError, match="reservation"):
        transport.run("1")
    assert sandbox.requests == []
    with pytest.raises(RuntimeError, match="reservation"):
        transport.upload(tmp_path / "file", "artifact")
    assert sandbox.directories == []
    with pytest.raises(RuntimeError, match="reservation"):
        transport.read_text("notes.md")
    assert sandbox.commands == []
    assert calls == [("stdin", None), ("filesystem", None), ("exec", 55)]
    with pytest.raises(RuntimeError, match="reservation"):
        transport.close()
    assert sandbox.closed


def test_replayed_batch_is_bounded_per_response_not_per_chunk(sandbox_setup):
    sandbox, _, volumes = sandbox_setup
    transport = ModalSandboxTransport.create("app", "image", *volumes)
    historical = json.dumps({"id": "old-request", "ok": True, "output": "x" * 20_000}) + "\n"
    sandbox.stdout.queue.put(historical * 60)
    assert transport.status()["ok"]
    transport.close()


def test_inspect_work_snapshot_lists_regular_files_without_following_symlinks(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    edited = work / "edited"
    edited.mkdir()
    (edited / "layer.weight.npy").write_bytes(b"weights")
    (edited / ".npy").write_bytes(b"invalid name")
    (edited / "bad\\name.npy").write_bytes(b"invalid name")
    (edited / "directory.npy").mkdir()
    nested = work / "fwd_1"
    nested.mkdir()
    (nested / "meta.json").write_text("{}")
    notes = "prefix:" + "hypothesis\n" * 1000
    (work / "notes.md").write_text(notes)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden.txt").write_text("must not be inspected")
    (work / "escape").symlink_to(outside, target_is_directory=True)
    (work / "linked.txt").symlink_to(outside / "hidden.txt")
    (edited / "linked.weight.npy").symlink_to(outside / "hidden.txt")
    result = _inspect_work_snapshot(str(work))
    assert result["edited"] == ["layer.weight"]
    assert result["files"] == sorted(
        ["edited/.npy", "edited/bad\\name.npy", "edited/layer.weight.npy", "fwd_1/meta.json", "notes.md"]
    )
    assert result["notes"] == notes[-8000:]
    assert result["notes_path"] == "/work/notes.md"
    assert result["notes_truncated"] is True
    assert result["files_truncated"] is False
    assert result["edited_truncated"] is False
    (work / "notes.md").unlink()
    (work / "notes.md").symlink_to(outside / "hidden.txt")
    assert _inspect_work_snapshot(str(work))["notes"] == ""
    with pytest.raises(OSError):
        _inspect_work_snapshot(str(work / "escape"))


def test_inspect_work_snapshot_truncates_large_file_inventory_explicitly(tmp_path):
    for index in range(3000):
        (tmp_path / (f"{index:05d}-" + "x" * 240)).touch()
    result = _inspect_work_snapshot(str(tmp_path))
    assert result["files_truncated"] is True
    assert result["files"] == sorted(result["files"])
    assert len(result["files"]) < 3000
    assert len(json.dumps(result).encode()) < 1_048_576


def test_inspect_work_uses_guarded_bounded_exec_without_kernel_run(sandbox_setup):
    sandbox, _, volumes = sandbox_setup
    transport = ModalSandboxTransport.create("app", "image", *volumes)
    expected = {"edited": ["layer.weight"], "files": ["edited/layer.weight.npy"], "notes": "hypothesis"}
    activities = []

    def guard(operation, timeout_seconds=None):
        activities.append((operation, timeout_seconds))
        # The budget guard can need another thread which uses this lock.
        with ThreadPoolExecutor(max_workers=1) as pool:

            def available():
                acquired = transport._lock.acquire(timeout=0.2)
                if acquired:
                    transport._lock.release()
                return acquired

            assert pool.submit(available).result(timeout=1)

    def execute(*args, **kwargs):
        sandbox.commands.append((args, kwargs))
        return SimpleNamespace(
            stdout=SimpleNamespace(read=lambda: json.dumps(expected)), wait=lambda: 0, returncode=0
        )

    sandbox.exec = execute
    transport.before_activity = guard
    assert transport.inspect_work() == expected
    args, options = sandbox.commands[-1]
    assert args[:3] == ("python", "-I", "-c")
    assert "_inspect_work_snapshot('/work')" in args[3]
    assert options == {"timeout": 55}
    assert activities == [("exec", 55)]
    assert sandbox.requests == []
    assert transport.runs == 0
    transport.close()


@pytest.mark.parametrize("payload", ["not-json", "[]", "x" * 1_048_577])
def test_inspect_work_rejects_invalid_or_oversized_response(sandbox_setup, payload):
    sandbox, _, volumes = sandbox_setup
    transport = ModalSandboxTransport.create("app", "image", *volumes)
    sandbox.exec = lambda *args, **kwargs: SimpleNamespace(
        stdout=SimpleNamespace(read=lambda: payload), wait=lambda: 0, returncode=0
    )
    with pytest.raises(SandboxTransportError):
        transport.inspect_work()
    transport.close()


def test_isolation_probe_reports_measured_denials_and_secret_names_only(monkeypatch):
    import torch

    targets = []

    def blocked(target, timeout):
        targets.append((target, timeout))
        raise TimeoutError("synthetic network denial")

    def readonly(*args, **kwargs):
        raise OSError(errno.EROFS, "synthetic read-only mount")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(tempfile, "mkstemp", readonly)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(os, "environ", {"OPENAI_API_KEY": "synthetic-value-must-not-be-returned"})
    result = _probe_sandbox_isolation()
    assert targets == [(("1.1.1.1", 443), 2)]
    assert result["network"]["connected"] is False
    assert result["network"]["error_type"] == "TimeoutError"
    assert result["network"]["target"] == "1.1.1.1:443"
    assert "other destinations" in result["network"]["scope"]
    assert all(record["expected_denial"] for record in result["mounts"].values())
    assert result["secret_environment"]["present_names"] == ["OPENAI_API_KEY"]
    assert "synthetic-value-must-not-be-returned" not in json.dumps(result)
    assert result["cuda_available"] is False


def test_isolation_probe_cleans_up_unexpected_writes_and_reports_connectivity(monkeypatch, tmp_path):
    import torch

    original_mkstemp = tempfile.mkstemp
    created = []

    def writable(*args, **kwargs):
        descriptor, path = original_mkstemp(prefix=kwargs["prefix"], dir=tmp_path)
        created.append(Path(path))
        return descriptor, path

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(tempfile, "mkstemp", writable)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(os, "environ", {})
    result = _probe_sandbox_isolation()
    assert result["network"]["connected"] is True
    assert result["cuda_available"] is True
    assert len(created) == 2
    assert all(not path.exists() for path in created)
    assert all(not record["write_denied"] for record in result["mounts"].values())
    assert result["secret_environment"]["present_names"] == []


def test_isolation_probe_does_not_mistake_missing_mount_for_readonly(monkeypatch):
    import torch

    def missing(*args, **kwargs):
        raise FileNotFoundError(errno.ENOENT, "synthetic missing mount")

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(tempfile, "mkstemp", missing)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = _probe_sandbox_isolation()
    assert all(not record["expected_denial"] for record in result["mounts"].values())


def test_verify_isolation_uses_guarded_bounded_exec_without_kernel_run(sandbox_setup):
    sandbox, _, volumes = sandbox_setup
    transport = ModalSandboxTransport.create("app", "image", *volumes)
    expected = {
        "network": {"connected": False},
        "mounts": {},
        "secret_environment": {"present_names": []},
        "cuda_available": False,
    }
    activities = []
    transport.before_activity = lambda operation, timeout_seconds=None: activities.append(
        (operation, timeout_seconds)
    )

    def execute(*args, **kwargs):
        sandbox.commands.append((args, kwargs))
        return SimpleNamespace(
            stdout=SimpleNamespace(read=lambda: json.dumps(expected).encode()), wait=lambda: 0, returncode=0
        )

    sandbox.exec = execute
    assert transport.verify_isolation() == expected
    args, options = sandbox.commands[-1]
    assert args[:3] == ("python", "-I", "-c")
    assert "_probe_sandbox_isolation()" in args[3]
    assert options == {"timeout": 55}
    assert activities == [("exec", 55)]
    assert sandbox.requests == []
    assert transport.runs == 0
    transport.close()
