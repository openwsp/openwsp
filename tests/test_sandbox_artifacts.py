"""Exercise bounded artifact publication against the real local receiver."""

import hashlib
import io
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openwsp.backends.sandbox import (
    ModalSandboxTransport,
    SandboxTransportError,
    _artifact_receiver,
)

CHUNK_LIMIT = 16 * 1024**2


class ReceiverProcess:
    def __init__(self, sandbox, request):
        self.sandbox, self.request = sandbox, request
        self.data = io.BytesIO()
        self.events = []
        self.returncode = None
        self.stdin = SimpleNamespace(
            write=self.write,
            write_eof=lambda: self.events.append("eof"),
            drain=lambda: self.events.append("drain"),
        )
        self.stdout = SimpleNamespace(read=self.read)

    def write(self, data):
        assert isinstance(data, bytes)
        assert len(data) <= CHUNK_LIMIT
        self.events.append("write")
        self.sandbox.chunk_lengths.append(len(data))
        self.data.write(data)

    def read(self):
        if self.request["op"] == "chunk":
            assert self.events == ["write", "eof", "drain"]
        self.data.seek(0)
        try:
            result = _artifact_receiver(str(self.sandbox.root), self.request, self.data)
        except (ValueError, OSError) as exc:
            self.returncode = 1
            self.sandbox.errors.append(exc)
            return b""
        self.returncode = 0
        if self.request["op"] == "finish" and self.sandbox.corrupt_finish_receipt:
            result["sha256"] = "0" * 64
        return json.dumps(result).encode()

    def wait(self):
        assert self.returncode is not None
        return self.returncode


class ReceiverSandbox:
    def __init__(self, root):
        self.root = root
        self.requests, self.chunk_lengths, self.errors = [], [], []
        self.fail_operation = None
        self.corrupt_finish = False
        self.corrupt_finish_receipt = False

    def exec(self, *args, **kwargs):
        assert args[:3] == ("python", "-I", "-c")
        assert "def _artifact_receiver(" in args[3]
        assert kwargs == {"timeout": 55, "text": False}
        request = json.loads(args[4])
        self.requests.append(request)
        if request["op"] == self.fail_operation:
            raise OSError("simulated exec failure")
        if request["op"] == "finish" and self.corrupt_finish:
            request["sha256"] = "0" * 64
        return ReceiverProcess(self, request)


@pytest.fixture
def publication(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    sandbox = ReceiverSandbox(root)
    transport = object.__new__(ModalSandboxTransport)
    transport.sandbox = sandbox
    transport._lock = threading.Lock()
    guarded = []

    def guard(operation, *, timeout_seconds=None):
        assert transport._lock.acquire(blocking=False), "budget guard ran under the transport lock"
        transport._lock.release()
        assert (operation, timeout_seconds) == ("exec", 55)
        guarded.append(operation)

    transport.before_activity = guard
    return transport, sandbox, guarded


@pytest.mark.parametrize("size", [0, CHUNK_LIMIT + 4099])
def test_binary_publication_is_bounded_verified_and_complete(tmp_path, publication, size):
    transport, sandbox, guarded = publication
    source = tmp_path / "source.bin"
    payload = (bytes(range(256)) * (size // 256 + 1))[:size]
    source.write_bytes(payload)

    result = transport.publish_file(source, "/work/forward/activations.bin")

    assert result == {
        "path": "/work/forward/activations.bin",
        "bytes": size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert (sandbox.root / "forward/activations.bin").read_bytes() == payload
    assert sandbox.chunk_lengths == ([CHUNK_LIMIT, 4099] if size else [])
    assert len(guarded) == len(sandbox.requests)
    assert [r["op"] for r in sandbox.requests] == (
        ["init", "chunk", "chunk", "finish"] if size else ["init", "finish"]
    )
    assert not list(sandbox.root.rglob(".openwsp-artifact-*"))


def test_atomic_replacement_preserves_existing_open_memmap(tmp_path, publication):
    transport, sandbox, _ = publication
    target = sandbox.root / "values.bin"
    target.write_bytes(b"old bytes")
    existing = np.memmap(target, dtype=np.uint8, mode="r")
    old_inode = target.stat().st_ino
    source = tmp_path / "replacement.bin"
    source.write_bytes(b"new bytes")
    try:
        transport.publish_file(source, "values.bin")
        assert existing.tobytes() == b"old bytes"
        assert target.read_bytes() == b"new bytes"
        assert target.stat().st_ino != old_inode
    finally:
        existing._mmap.close()


def test_final_digest_failure_cleans_temporary_and_preserves_target(tmp_path, publication):
    transport, sandbox, _ = publication
    target = sandbox.root / "artifact.bin"
    target.write_bytes(b"keep old artifact")
    source = tmp_path / "new.bin"
    source.write_bytes(b"replacement")
    sandbox.corrupt_finish = True

    with pytest.raises(SandboxTransportError, match="Artifact finish failed"):
        transport.publish_file(source, "artifact.bin")

    assert "final sha256 mismatch" in str(sandbox.errors[0])
    assert target.read_bytes() == b"keep old artifact"
    assert sandbox.requests[-1]["op"] == "abort"
    assert not list(sandbox.root.rglob(".openwsp-artifact-*"))


@pytest.mark.parametrize("destination_kind", ["parent_symlink", "target_symlink", "fifo", "directory"])
def test_destination_symlinks_and_special_files_are_refused(tmp_path, publication, destination_kind):
    transport, sandbox, _ = publication
    source = tmp_path / "source"
    source.write_bytes(b"new")
    outside = tmp_path / "outside"
    outside.mkdir()
    untouched = outside / "artifact"
    untouched.write_bytes(b"untouched")
    remote = "artifact"
    target = sandbox.root / remote
    if destination_kind == "parent_symlink":
        (sandbox.root / "linked").symlink_to(outside, target_is_directory=True)
        remote = "linked/artifact"
    elif destination_kind == "target_symlink":
        target.symlink_to(untouched)
    elif destination_kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()

    with pytest.raises(SandboxTransportError, match="Artifact init failed"):
        transport.publish_file(source, remote)

    assert untouched.read_bytes() == b"untouched"
    assert not list(sandbox.root.rglob(".openwsp-artifact-*"))


@pytest.mark.parametrize("source_kind", ["file_symlink", "parent_symlink", "fifo"])
def test_source_symlinks_and_special_files_fail_before_exec(tmp_path, publication, source_kind):
    transport, sandbox, guarded = publication
    directory = tmp_path / "source_directory"
    directory.mkdir()
    real = directory / "file"
    real.write_bytes(b"source")
    source = tmp_path / "source"
    if source_kind == "file_symlink":
        source.symlink_to(real)
    elif source_kind == "parent_symlink":
        source.symlink_to(directory, target_is_directory=True)
        source /= "file"
    else:
        os.mkfifo(source)

    with pytest.raises((ValueError, OSError)):
        transport.publish_file(source, "artifact")
    assert sandbox.requests == []
    assert guarded == []


def test_guard_rejection_precedes_exec(tmp_path, publication):
    transport, sandbox, _ = publication
    source = tmp_path / "source"
    source.write_bytes(b"content")

    def reject(*args, **kwargs):
        raise RuntimeError("budget exhausted")

    transport.before_activity = reject
    with pytest.raises(RuntimeError, match="budget exhausted"):
        transport.publish_file(source, "artifact")
    assert sandbox.requests == []
    assert list(sandbox.root.iterdir()) == []


def test_exec_failure_aborts_staged_transfer(tmp_path, publication):
    transport, sandbox, _ = publication
    source = tmp_path / "source"
    source.write_bytes(b"replacement")
    target = sandbox.root / "artifact"
    target.write_bytes(b"old")
    sandbox.fail_operation = "chunk"

    with pytest.raises(OSError, match="simulated exec failure"):
        transport.publish_file(source, "artifact")

    assert [r["op"] for r in sandbox.requests] == ["init", "chunk", "abort"]
    assert target.read_bytes() == b"old"
    assert not list(sandbox.root.rglob(".openwsp-artifact-*"))


@pytest.mark.parametrize("remote", ["../escape", "/tmp/escape", "/work/../escape", "/work"])
def test_destination_escape_is_rejected_before_exec(tmp_path, publication, remote):
    transport, sandbox, _ = publication
    source = tmp_path / "source"
    source.write_bytes(b"content")
    with pytest.raises(ValueError):
        transport.publish_file(source, remote)
    assert sandbox.requests == []


@pytest.mark.parametrize("case", ["utf8", "notice51", "empty", "byte_limit"])
def test_leased_text_uses_atomic_binary_receiver_and_verified_sha(publication, case):
    transport, sandbox, guarded = publication
    notice = Path(__file__).parents[1] / "replications/exp8/prompts/eval_prompt_format.txt"
    text = {"utf8": "café — λ\n", "notice51": notice.read_text(), "empty": "",
            "byte_limit": "é" * (1_048_576 // 2)}[case]
    payload = text.encode("utf-8")
    if case == "notice51":
        assert len(payload) == 482
    captured = []
    publish = transport.publish_file

    def observe(source, remote):
        assert source.read_bytes() == payload
        receipt = publish(source, remote)
        captured.append((source, receipt))
        return receipt

    transport.publish_file = observe
    assert transport.write_text("eval_prompt_format.txt", text) is None
    assert (sandbox.root / "eval_prompt_format.txt").read_bytes() == payload
    source, receipt = captured[0]
    assert not source.exists() and not source.parent.exists()
    assert receipt == {"path": "/work/eval_prompt_format.txt", "bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()}
    assert [row["op"] for row in sandbox.requests] == (["init", "chunk", "finish"] if payload else ["init", "finish"])
    assert len(guarded) == len(sandbox.requests)
    assert not list(sandbox.root.rglob(".openwsp-artifact-*"))


def test_text_byte_ceiling_rejects_multibyte_overflow_before_remote_activity(publication):
    transport, sandbox, guarded = publication
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        transport.write_text("notice.txt", "é" * (1_048_576 // 2 + 1))
    assert sandbox.requests == [] and guarded == []


@pytest.mark.parametrize("remote", ["../escape", "/tmp/escape", "/work/../escape", "/work"])
def test_leased_text_path_guard_precedes_publication(publication, remote):
    transport, sandbox, guarded = publication
    with pytest.raises(ValueError, match="under /work"):
        transport.write_text(remote, "notice")
    assert sandbox.requests == [] and guarded == []


def test_leased_text_rejects_hash_failure_and_preserves_existing_destination(publication):
    transport, sandbox, _ = publication
    target = sandbox.root / "notice.txt"
    target.write_bytes(b"previous complete text")
    sandbox.corrupt_finish = True
    with pytest.raises(SandboxTransportError, match="Artifact finish failed"):
        transport.write_text("notice.txt", "new notice")
    assert target.read_bytes() == b"previous complete text"
    assert sandbox.requests[-1]["op"] == "abort"
    assert not list(sandbox.root.rglob(".openwsp-artifact-*"))


def test_leased_text_never_accepts_a_mismatched_final_receipt(publication):
    transport, sandbox, _ = publication
    sandbox.corrupt_finish_receipt = True
    with pytest.raises(SandboxTransportError, match="final verification receipt"):
        transport.write_text("notice.txt", "complete UTF-8 notice: é")
    # A lost/bad acknowledgement cannot prove the atomic rename did not happen.
    assert (sandbox.root / "notice.txt").read_text() == "complete UTF-8 notice: é"


def test_text_local_temporary_is_cleaned_after_guard_rejection(publication):
    transport, sandbox, _ = publication
    sources = []
    publish = transport.publish_file

    def observe(source, remote):
        sources.append(source)
        return publish(source, remote)

    def reject(*args, **kwargs):
        raise RuntimeError("budget unavailable")

    transport.publish_file = observe
    transport.before_activity = reject
    with pytest.raises(RuntimeError, match="budget unavailable"):
        transport.write_text("notice.txt", "notice")
    assert sandbox.requests == []
    assert sources and not sources[0].exists() and not sources[0].parent.exists()
