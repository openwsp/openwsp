"""Modal transport for the persistent, isolated weight-analysis worker.

Modal is imported only when creating or attaching a transport. The worker and
core remain provider-independent. This adapter targets the documented Modal
1.6 API: Volume.with_mount_options, Sandbox.filesystem, and v2-volume sync.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import queue
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


class SandboxTransportError(RuntimeError):
    pass


def _artifact_receiver(root: str, request: dict, input_stream=None) -> dict:
    """Receive one bounded artifact operation, with no symlink traversal."""
    import hashlib
    import os
    import stat
    import sys
    from pathlib import PurePosixPath

    relative = PurePosixPath(request["path"])
    transfer = request["transfer_id"]
    if relative.is_absolute() or not relative.parts or any(part in (".", "..") for part in relative.parts):
        raise ValueError("Artifact path must be a file under the work directory")
    if (
        not isinstance(transfer, str)
        or len(transfer) != 32
        or any(c not in "0123456789abcdef" for c in transfer)
    ):
        raise ValueError("Invalid artifact transfer id")
    operation = request["op"]
    if operation not in ("init", "chunk", "finish", "abort"):
        raise ValueError("Unknown artifact operation")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(root, flags)
    temporary = ".openwsp-artifact-" + transfer
    descriptor = None
    owned = False
    try:
        for part in relative.parts[:-1]:
            if operation == "init":
                try:
                    os.mkdir(part, mode=0o755, dir_fd=directory)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        name = relative.name
        try:
            target = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            target = None
        if operation == "init" and target is not None and not stat.S_ISREG(target.st_mode):
            raise ValueError("Artifact destination must be a regular file, not a symlink or special file")
        if operation == "init":
            descriptor = os.open(
                temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
            )
            owned = True
        else:
            try:
                descriptor = os.open(temporary, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            except FileNotFoundError:
                if operation == "abort":
                    return {"removed": False}
                raise
        info = os.fstat(descriptor)
        identity = [info.st_dev, info.st_ino]
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Artifact temporary file is not a private regular file")
        if operation != "init":
            if request.get("identity") != identity:
                raise ValueError("Artifact temporary file changed identity")
            owned = True
        if operation == "init":
            return {"identity": identity, "bytes": 0}
        if operation == "abort":
            os.unlink(temporary, dir_fd=directory)
            owned = False
            return {"removed": True}
        expected_size = request["size"]
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError("Invalid artifact size")
        if operation == "chunk":
            offset = request["offset"]
            if type(offset) is not int or offset < 0 or info.st_size != offset:
                raise ValueError("Artifact chunk offset does not match the temporary file")
            if expected_size > 16 * 1024**2:
                raise ValueError("Artifact chunk exceeds 16 MiB")
            stream = sys.stdin.buffer if input_stream is None else input_stream
            chunk = stream.read(expected_size + 1)
            if len(chunk) != expected_size or hashlib.sha256(chunk).hexdigest() != request["sha256"]:
                raise ValueError("Artifact chunk length or sha256 mismatch")
            os.lseek(descriptor, offset, os.SEEK_SET)
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("Artifact write made no progress")
                view = view[written:]
            return {"identity": identity, "bytes": offset + len(chunk), "sha256": request["sha256"]}
        if info.st_size != expected_size:
            raise ValueError("Artifact final length mismatch")
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, 1024**2):
            digest.update(chunk)
        if digest.hexdigest() != request["sha256"]:
            raise ValueError("Artifact final sha256 mismatch")
        os.fsync(descriptor)
        try:
            target = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            target = None
        if target is not None and not stat.S_ISREG(target.st_mode):
            raise ValueError("Artifact destination must be a regular file, not a symlink or special file")
        # Rename replaces this directory entry, so an existing reader/memmap
        # keeps its old inode while later opens see the complete new artifact.
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        owned = False
        return {
            "identity": identity,
            "path": str(relative),
            "bytes": expected_size,
            "sha256": digest.hexdigest(),
        }
    except Exception:
        if owned:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _open_artifact_source(path: str | Path):
    """Open a regular local source without following symlink parents or files."""
    absolute = Path(os.path.abspath(path))
    directory = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Artifact source must be a regular file")
        stream = os.fdopen(descriptor, "rb")
        descriptor = None
        return stream
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _inspect_work_snapshot(root: str) -> dict:
    """Inspect regular work files without following editor-created symlinks."""
    import json
    import os
    import stat

    root = os.path.abspath(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(root, flags)
    result = {
        "edited": [],
        "files": [],
        "notes": "",
        "notes_path": "/work/notes.md",
        "files_truncated": False,
        "edited_truncated": False,
        "notes_truncated": False,
    }
    try:
        try:
            edit_fd = os.open("edited", flags, dir_fd=root_fd)
        except (FileNotFoundError, NotADirectoryError, OSError):
            edit_fd = None
        if edit_fd is not None:
            try:
                edited_bytes = 0
                for name in sorted(os.listdir(edit_fd)):
                    stem = name.removesuffix(".npy")
                    if not name.endswith(".npy") or not stem or stem in {".", ".."} or "\\" in stem:
                        continue
                    try:
                        mode = os.stat(name, dir_fd=edit_fd, follow_symlinks=False).st_mode
                    except FileNotFoundError:
                        continue
                    if stat.S_ISREG(mode):
                        size = len(json.dumps(stem, ensure_ascii=True).encode()) + 2
                        if len(result["edited"]) >= 4096 or edited_bytes + size > 200_000:
                            result["edited_truncated"] = True
                            break
                        result["edited"].append(stem)
                        edited_bytes += size
            finally:
                os.close(edit_fd)
        used = 0
        for directory, dirs, files, directory_fd in os.fwalk(".", dir_fd=root_fd, follow_symlinks=False):
            dirs.sort()
            for name in sorted(files):
                try:
                    mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(mode):
                    continue
                relative = os.path.normpath(os.path.join(directory, name))
                if os.path.isabs(relative) or ".." in relative.split(os.sep):
                    raise ValueError("Work inspection encountered a path outside its root")
                size = len(json.dumps(relative, ensure_ascii=True).encode()) + 2
                if len(result["files"]) >= 10_000 or used + size > 700_000:
                    result["files_truncated"] = True
                    break
                result["files"].append(relative)
                used += size
            if result["files_truncated"]:
                break
        result["files"].sort()
        try:
            note_fd = os.open("notes.md", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        except (FileNotFoundError, OSError):
            note_fd = None
        if note_fd is not None:
            try:
                metadata = os.fstat(note_fd)
                if stat.S_ISREG(metadata.st_mode):
                    start = max(0, metadata.st_size - 32_000)
                    os.lseek(note_fd, start, os.SEEK_SET)
                    text = os.read(note_fd, 32_000).decode("utf-8", errors="replace")
                    result["notes"] = text[-8000:]
                    result["notes_truncated"] = start > 0 or len(text) > 8000
            finally:
                os.close(note_fd)
        # Names can be non-ASCII. Enforce the limit on the serialized bytes,
        # including the edited-name list rather than only ordinary filenames.
        if len(json.dumps(result, ensure_ascii=True).encode()) > 1_000_000:
            raise ValueError("Work inspection cannot fit its output limit")
        return result
    finally:
        os.close(root_fd)


def _probe_sandbox_isolation() -> dict:
    """Measure specific sandbox restrictions without reading secret values."""
    import errno
    import os
    import socket
    import tempfile
    import time

    import torch

    target = ("1.1.1.1", 443)
    network = {"target": "1.1.1.1:443", "timeout_seconds": 2, "connected": False}
    started = time.monotonic()
    try:
        with socket.create_connection(target, timeout=2):
            network["connected"] = True
    except OSError as exc:
        network.update(error_type=type(exc).__name__, errno=exc.errno)
    network["seconds"] = time.monotonic() - started
    network["scope"] = "One outbound TCP connection probe; does not establish policy for other destinations."
    mounts = {}
    for directory in ("/model", "/practice"):
        descriptor = None
        probe = None
        record = {"write_denied": False, "expected_denial": False}
        try:
            descriptor, probe = tempfile.mkstemp(prefix=".openwsp-isolation-probe-", dir=directory)
        except OSError as exc:
            record.update(
                write_denied=True,
                expected_denial=exc.errno in {errno.EROFS, errno.EACCES, errno.EPERM},
                error_type=type(exc).__name__,
                errno=exc.errno,
            )
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if probe is not None:
                os.unlink(probe)
        mounts[directory] = record
    secret_names = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_ACCESS_TOKEN",
        "GH_TOKEN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_IDENTITY_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_CLIENT_SECRET",
    }
    return {
        "network": network,
        "mounts": mounts,
        "secret_environment": {
            "checked_names": sorted(secret_names),
            "present_names": sorted(name for name in secret_names if name in os.environ),
        },
        "cuda_available": bool(torch.cuda.is_available()),
    }


class ModalSandboxTransport:
    def __init__(self, sandbox: Any, *, cell_timeout: float = 900, startup_timeout: float = 180):
        self.sandbox = sandbox
        self.cell_timeout = cell_timeout
        self.startup_timeout = startup_timeout
        self.runs = 0
        self.before_activity = None
        self._messages: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._response_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._stream_error: SandboxTransportError | None = None
        self._closed = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True, name="openwsp-sandbox-stdout")
        self._reader.start()

    @classmethod
    def create(
        cls,
        app,
        image,
        model_volume,
        practice_volume,
        work_volume,
        *,
        model_subpath: str | None = None,
        practice_subpath: str | None = None,
        work_subpath: str | None = None,
        tokenizer_path: str | None = "/model",
        memory_request_mib: int = 8192,
        memory_limit_mib: int = 98_304,
        cpu: float = 2,
        timeout: int = 86_400,
        idle_timeout: int | None = 180,
        cell_timeout: float = 900,
        startup_timeout: float = 180,
        initial_runs: int = 0,
        max_runs: int = 300,
        work_volume_version: int = 2,
    ) -> ModalSandboxTransport:
        """Create with only the selected model, practice and run-work mounts.

        Pass dedicated Volumes or explicit subpaths; never pass a full volume
        containing evaluation data, reference edits, credentials or driver logs.
        The writable work Volume must be v2, whose documented ``sync`` operation
        makes files visible to GPU workers before the sandbox terminates.
        """
        import modal

        if work_volume_version != 2:
            raise ValueError("The sandbox work volume must be v2 for explicit commit/reload semantics")
        if not 0 < memory_request_mib <= memory_limit_mib <= 98_304:
            raise ValueError("Memory request/limit must be positive and at most 96 GiB")
        if not 0 < cpu <= 2 or not 0 < cell_timeout <= 900 or not 0 < timeout <= 86_400:
            raise ValueError("Sandbox limits exceed the configured protocol")
        if not 0 <= initial_runs <= max_runs <= 300:
            raise ValueError("Invalid kernel code-run budget")
        mounts = {
            "/model": model_volume.with_mount_options(read_only=True, sub_path=model_subpath),
            "/practice": practice_volume.with_mount_options(read_only=True, sub_path=practice_subpath),
            "/work": work_volume.with_mount_options(read_only=False, sub_path=work_subpath),
        }
        args = [
            "python",
            "-u",
            "-m",
            "openwsp.backends.sandbox_worker",
            "--model",
            "/model",
            "--practice",
            "/practice",
            "--work",
            "/work",
            "--cell-timeout",
            str(cell_timeout),
            "--startup-timeout",
            str(startup_timeout),
            "--initial-runs",
            str(initial_runs),
            "--max-runs",
            str(max_runs),
            "--memory-limit-mib",
            str(memory_limit_mib),
            "--verify-read-only",
        ]
        if tokenizer_path is not None:
            args.extend(["--tokenizer", tokenizer_path])
        sandbox = modal.Sandbox.create(
            *args,
            app=app,
            image=image,
            volumes=mounts,
            secrets=[],
            env={
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            },
            block_network=True,
            include_oidc_identity_token=False,
            cpu=(cpu, cpu),
            memory=(memory_request_mib, memory_limit_mib),
            timeout=timeout,
            idle_timeout=idle_timeout,
            workdir="/work",
        )
        transport = cls(sandbox, cell_timeout=cell_timeout, startup_timeout=startup_timeout)
        try:
            transport._await_ready()
        except BaseException:
            sandbox.terminate(wait=True)
            raise
        return transport

    @classmethod
    def from_id(
        cls, sandbox_id: str, *, cell_timeout: float = 900, startup_timeout: float = 180, probe: bool = True
    ):
        """Reconnect after a driver restart; request IDs skip replayed output."""
        import modal

        transport = cls(
            modal.Sandbox.from_id(sandbox_id), cell_timeout=cell_timeout, startup_timeout=startup_timeout
        )
        if probe:
            transport.status()
        return transport

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.object_id

    def _read_stdout(self):
        pending = ""
        try:
            for chunk in self.sandbox.stdout:
                pending += chunk.decode() if isinstance(chunk, bytes) else chunk
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    if len(line) > 1_048_576:
                        raise SandboxTransportError("Sandbox response exceeded 1 MiB")
                    if line:
                        self._deliver(json.loads(line))
                if len(pending) > 1_048_576:
                    raise SandboxTransportError("Sandbox response exceeded 1 MiB")
            if pending.strip():
                self._deliver(json.loads(pending))
        except Exception as exc:  # noqa: BLE001 -- relay reader errors to the calling thread.
            self._stream_failed(SandboxTransportError(f"Sandbox stream failed: {type(exc).__name__}: {exc}"))
        finally:
            self._stream_failed(SandboxTransportError("Sandbox output stream closed"))

    def _deliver(self, message):
        if not isinstance(message, dict):
            raise SandboxTransportError("Sandbox returned a non-object JSON response")
        with self._response_lock:
            if message.get("type") == "ready":
                self._messages.put(message)
            elif message.get("id") in self._pending:
                self._pending[message["id"]].put(message)

    def _stream_failed(self, error):
        with self._response_lock:
            if self._stream_error is None:
                self._stream_error = error
                self._messages.put(error)
                for destination in self._pending.values():
                    destination.put(error)

    def _receive(self, deadline: float, messages: queue.Queue | None = None) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxTransportError("Timed out waiting for sandbox response")
        try:
            result = (self._messages if messages is None else messages).get(timeout=remaining)
        except queue.Empty as exc:
            raise SandboxTransportError("Timed out waiting for sandbox response") from exc
        if isinstance(result, BaseException):
            raise result
        if not isinstance(result, dict):
            raise SandboxTransportError("Sandbox returned a non-object JSON response")
        return result

    def _await_ready(self):
        deadline = time.monotonic() + self.startup_timeout + 30
        while True:
            result = self._receive(deadline)
            if result.get("type") == "ready":
                if not result.get("ok"):
                    raise SandboxTransportError(result.get("error", "Sandbox initialization failed"))
                self.runs = result["runs"]
                return

    def _request(self, op: str, *, request_id: str | None = None, **fields) -> dict:
        self._guard_activity("stdin")
        identifier = request_id or uuid.uuid4().hex
        request = {"id": identifier, "op": op, **fields}
        wire = json.dumps(request, ensure_ascii=True) + "\n"
        if len(wire.encode()) > 1_048_576:
            raise ValueError("Kernel request exceeds 1 MiB")
        messages: queue.Queue = queue.Queue()
        with self._response_lock:
            if self._closed:
                raise SandboxTransportError("Sandbox transport is closed")
            if self._stream_error is not None:
                raise self._stream_error
            if identifier in self._pending:
                raise SandboxTransportError("Request id is already in flight")
            self._pending[identifier] = messages
        try:
            with self._write_lock:
                self.sandbox.stdin.write(wire)
                self.sandbox.stdin.drain()
            wait = 55 if op == "status" else self.cell_timeout + self.startup_timeout + 30
            result = self._receive(time.monotonic() + wait, messages)
            with self._response_lock:
                self.runs = max(self.runs, result.get("runs", self.runs))
            return result
        finally:
            with self._response_lock:
                self._pending.pop(identifier, None)

    def run(self, code: str, *, request_id: str | None = None) -> dict:
        """Use a driver-persisted ID to recover a completed cell after reconnect.

        IDs are deduplicated while this sandbox's supervisor lives. A replacement
        sandbox has no completed-request journal; the driver must resolve an
        interrupted cell from its durable logs rather than blindly retry it.
        """
        return self._request("run", code=code, request_id=request_id)

    def _guard_activity(self, operation: str, timeout_seconds: float | None = None) -> None:
        if self.before_activity is not None:
            self.before_activity(operation, timeout_seconds=timeout_seconds)

    def status(self) -> dict:
        return self._request("status")

    def restart(self, *, request_id: str | None = None) -> dict:
        return self._request("restart", request_id=request_id)

    def inspect_work(self) -> dict:
        """Read handoff evidence outside the editor kernel without spending a run.

        Only regular files under /work are inspected; symlinks are excluded.
        Inspection is bounded to 55 seconds and a response smaller than 1 MiB.
        """
        script = "import json\n" + inspect.getsource(_inspect_work_snapshot)
        script += "\nprint(json.dumps(_inspect_work_snapshot('/work'), ensure_ascii=True))\n"
        self._guard_activity("exec", timeout_seconds=55)
        with self._lock:
            process = self.sandbox.exec("python", "-I", "-c", script, timeout=55)
            output = process.stdout.read()
            process.wait()
        if process.returncode != 0:
            raise SandboxTransportError(f"Work inspection failed with exit code {process.returncode}")
        encoded = output if isinstance(output, bytes) else output.encode()
        if len(encoded) > 1_048_576:
            raise SandboxTransportError("Work inspection response exceeded 1 MiB")
        try:
            result = json.loads(encoded)
        except ValueError as exc:
            raise SandboxTransportError("Work inspection returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise SandboxTransportError("Work inspection returned a non-object response")
        return result

    def verify_isolation(self) -> dict:
        """Measure one TCP probe, mount write denials, secret names and CUDA state.

        This is for the verification sandbox, not an editor-visible tool. It
        does not read credentials or spend a kernel code run. A failed TCP probe
        is evidence about its named destination, not proof of all network rules.
        """
        script = "import json\n" + inspect.getsource(_probe_sandbox_isolation)
        script += "\nprint(json.dumps(_probe_sandbox_isolation(), ensure_ascii=True))\n"
        self._guard_activity("exec", timeout_seconds=55)
        with self._lock:
            process = self.sandbox.exec("python", "-I", "-c", script, timeout=55)
            output = process.stdout.read()
            process.wait()
        if process.returncode != 0:
            raise SandboxTransportError(f"Isolation verification failed with exit code {process.returncode}")
        encoded = output if isinstance(output, bytes) else output.encode()
        if len(encoded) > 1_048_576:
            raise SandboxTransportError("Isolation verification response exceeded 1 MiB")
        try:
            result = json.loads(encoded)
        except ValueError as exc:
            raise SandboxTransportError("Isolation verification returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise SandboxTransportError("Isolation verification returned a non-object response")
        return result

    @staticmethod
    def _work_path(path: str) -> str:
        relative = PurePosixPath(path)
        if relative.is_absolute():
            try:
                relative = relative.relative_to("/work")
            except ValueError as exc:
                raise ValueError("Artifact transfers must remain under /work") from exc
        if ".." in relative.parts or not relative.parts:
            raise ValueError("Artifact transfers require a file path under /work")
        return str(PurePosixPath("/work") / relative)

    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a forward artifact; caller is responsible for excluding eval data."""
        remote = self._work_path(remote_path)
        self._guard_activity("filesystem")
        self.sandbox.filesystem.make_directory(str(PurePosixPath(remote).parent))
        self.sandbox.filesystem.copy_from_local(local_path, remote)

    def _artifact_operation(self, request: dict, chunk: bytes | None = None) -> dict:
        script = "import json, sys\n" + inspect.getsource(_artifact_receiver)
        script += "\nprint(json.dumps(_artifact_receiver('/work', json.loads(sys.argv[1]))))\n"
        self._guard_activity("exec", timeout_seconds=55)
        with self._lock:
            process = self.sandbox.exec(
                "python", "-I", "-c", script, json.dumps(request), timeout=55, text=False
            )
            if chunk is not None:
                process.stdin.write(chunk)
                process.stdin.write_eof()
                process.stdin.drain()
            output = process.stdout.read()
            process.wait()
        if process.returncode != 0:
            raise SandboxTransportError(
                f"Artifact {request['op']} failed with exit code {process.returncode}"
            )
        encoded = output if isinstance(output, bytes) else output.encode()
        if len(encoded) > 16_384:
            raise SandboxTransportError("Artifact receiver response exceeded 16 KiB")
        try:
            result = json.loads(encoded)
        except ValueError as exc:
            raise SandboxTransportError("Artifact receiver returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise SandboxTransportError("Artifact receiver returned a non-object response")
        return result

    def publish_file(self, local_path: str | Path, remote_path: str) -> dict:
        """Publish one verified artifact into /work without reloading any mount.

        Sources and destination parents must not be symlinks. Each receiver exec
        has a 55-second bound and at most one 16 MiB chunk in memory. The final
        receiver verifies the complete SHA256 before atomically replacing the
        destination. Existing open readers and kernel variables remain intact.
        Cleanup is attempted on failure under the same budget guard; if remote
        access is lost, an incomplete hidden temporary file may remain.
        """
        remote = self._work_path(remote_path)
        relative = str(PurePosixPath(remote).relative_to("/work"))
        common = {"path": relative, "transfer_id": uuid.uuid4().hex}
        identity = None
        finished = False
        with _open_artifact_source(local_path) as source:
            before = os.fstat(source.fileno())
            try:
                result = self._artifact_operation({**common, "op": "init"})
                identity = result.get("identity")
                if (
                    not isinstance(identity, list)
                    or len(identity) != 2
                    or any(type(part) is not int or part < 0 for part in identity)
                    or result.get("bytes") != 0
                ):
                    raise SandboxTransportError("Artifact receiver returned an invalid file identity")
                common["identity"] = identity
                digest, total = hashlib.sha256(), 0
                while total < before.st_size:
                    chunk = source.read(min(16 * 1024**2, before.st_size - total))
                    if not chunk:
                        raise SandboxTransportError("Artifact source changed during publication")
                    chunk_sha = hashlib.sha256(chunk).hexdigest()
                    result = self._artifact_operation(
                        {**common, "op": "chunk", "offset": total, "size": len(chunk), "sha256": chunk_sha},
                        chunk,
                    )
                    total += len(chunk)
                    digest.update(chunk)
                    if (
                        result.get("identity") != identity
                        or result.get("bytes") != total
                        or result.get("sha256") != chunk_sha
                    ):
                        raise SandboxTransportError("Artifact chunk was not acknowledged correctly")
                after = os.fstat(source.fileno())
                if source.read(1) or (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise SandboxTransportError("Artifact source changed during publication")
                expected = {"path": relative, "bytes": total, "sha256": digest.hexdigest()}
                result = self._artifact_operation(
                    {**common, "op": "finish", "size": total, "sha256": digest.hexdigest()}
                )
                if result.get("identity") != identity or any(
                    result.get(key) != value for key, value in expected.items()
                ):
                    raise SandboxTransportError("Artifact final verification receipt did not match")
                finished = True
                return {**expected, "path": remote}
            finally:
                if identity is not None and not finished:
                    try:
                        self._artifact_operation({**common, "op": "abort"})
                    except Exception:  # noqa: BLE001, S110 -- preserve original failure if access is unavailable.
                        pass

    def download(self, remote_path: str, local_path: str | Path) -> None:
        self._guard_activity("filesystem")
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.sandbox.filesystem.copy_to_local(self._work_path(remote_path), local)

    def read_text(self, remote_path: str) -> str:
        if self.before_activity is not None:
            return self._bounded_text_file(remote_path, None)
        return self.sandbox.filesystem.read_text(self._work_path(remote_path))

    def write_text(self, remote_path: str, text: str) -> None:
        if self.before_activity is not None:
            remote = self._work_path(remote_path)
            payload = text.encode("utf-8")
            if len(payload) > 1_048_576:
                raise ValueError("Text transfer exceeds 1 MiB; use the shared work volume")
            # Use the same bounded binary receiver as forward artifacts. It
            # verifies SHA256 and atomically publishes before acknowledging;
            # each operation obtains its own lease guard without a reload.
            with tempfile.TemporaryDirectory(prefix="openwsp-text-") as directory:
                source = Path(directory) / "text.utf8"
                source.write_bytes(payload)
                self.publish_file(source, remote)
            return
        remote = self._work_path(remote_path)
        self.sandbox.filesystem.make_directory(str(PurePosixPath(remote).parent))
        self.sandbox.filesystem.write_text(text, remote)

    def _bounded_text_file(self, remote_path: str, text: str | None) -> str:
        remote = self._work_path(remote_path)
        if text is not None and len(text.encode()) > 1_048_576:
            raise ValueError("Text transfer exceeds 1 MiB; use the shared work volume")
        script = (
            "import sys; from pathlib import Path; "
            "p=Path(sys.argv[1]).resolve(); "
            "assert p.is_relative_to('/work') and p != Path('/work'), 'path outside /work'; "
            "mode=sys.argv[2]; "
            "p.parent.mkdir(parents=True,exist_ok=True) if mode=='write' else None; "
            "data=sys.stdin.buffer.read(1048577) if mode=='write' else p.open('rb').read(1048577); "
            "assert len(data)<=1048576, 'text exceeds 1 MiB'; "
            "p.write_bytes(data) if mode=='write' else sys.stdout.buffer.write(data)"
        )
        self._guard_activity("exec", timeout_seconds=55)
        process = self.sandbox.exec(
            "python", "-I", "-c", script, remote, "write" if text is not None else "read", timeout=55
        )
        if text is not None:
            process.stdin.write(text)
            process.stdin.write_eof()
            process.stdin.drain()
        output = process.stdout.read()
        process.wait()
        if process.returncode != 0:
            raise SandboxTransportError(f"Text transfer failed with exit code {process.returncode}")
        return output.decode() if isinstance(output, bytes) else output

    def commit_work(self) -> None:
        """Commit v2 Volume writes without giving the editor any Modal credential."""
        self._guard_activity("exec", timeout_seconds=55)
        with self._lock:
            process = self.sandbox.exec("sync", "/work", timeout=55)
            process.wait()
            if process.returncode != 0:
                raise SandboxTransportError(f"Volume commit failed with exit code {process.returncode}")

    def reload_work(self) -> None:
        """Reload ALL mounted volumes; fails when any mount has open files.

        Use publish_file for forward artifacts when a persistent kernel may
        retain open files or memory maps. Modal has no work-only reload API.
        """
        self._guard_activity("exec", timeout_seconds=55)
        with self._lock:
            self.sandbox.reload_volumes(timeout=55)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.commit_work()
        finally:
            self.sandbox.terminate(wait=True)
            self._closed = True
            self.sandbox.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
