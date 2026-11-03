import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openwsp.backends.sandbox import ModalSandboxTransport, SandboxTransportError


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
        self.filesystem = SimpleNamespace(
            write_text=lambda text, path: self.files.__setitem__(path, text),
            read_text=lambda path: self.files[path],
            make_directory=self.directories.append,
            copy_from_local=lambda local, remote: self.files.__setitem__(remote, Path(local).read_text()),
            copy_to_local=lambda remote, local: Path(local).write_text(self.files[remote]),
        )

    def write(self, message):
        request = json.loads(message)
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
    assert set(options["volumes"]) == {"/model", "/practice", "/work"}
    assert volumes[0].calls == [{"read_only": True, "sub_path": "qwen"}]
    assert volumes[1].calls == [{"read_only": True, "sub_path": "exp8"}]
    assert volumes[2].calls == [{"read_only": False, "sub_path": "run001"}]
    assert all("TOKEN" not in key or key == "TOKENIZERS_PARALLELISM" for key in options["env"])
    assert transport.run("1 + 1")["ok"]
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
