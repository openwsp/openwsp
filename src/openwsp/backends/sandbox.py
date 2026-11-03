"""Modal transport for the persistent, isolated weight-analysis worker.

Modal is imported only when creating or attaching a transport. The worker and
core remain provider-independent. This adapter targets the documented Modal
1.6 API: Volume.with_mount_options, Sandbox.filesystem, and v2-volume sync.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


class SandboxTransportError(RuntimeError):
    pass


class ModalSandboxTransport:
    def __init__(self, sandbox: Any, *, cell_timeout: float = 900, startup_timeout: float = 180):
        self.sandbox = sandbox
        self.cell_timeout = cell_timeout
        self.startup_timeout = startup_timeout
        self.runs = 0
        self._messages: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
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
    def from_id(cls, sandbox_id: str, *, cell_timeout: float = 900, startup_timeout: float = 180):
        """Reconnect after a driver restart; request IDs skip replayed output."""
        import modal

        transport = cls(
            modal.Sandbox.from_id(sandbox_id), cell_timeout=cell_timeout, startup_timeout=startup_timeout
        )
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
                if len(pending) > 1_048_576:
                    raise SandboxTransportError("Sandbox response exceeded 1 MiB")
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    if line:
                        self._messages.put(json.loads(line))
            if pending.strip():
                self._messages.put(json.loads(pending))
        except Exception as exc:  # noqa: BLE001 -- relay reader errors to the calling thread.
            self._messages.put(SandboxTransportError(f"Sandbox stream failed: {type(exc).__name__}: {exc}"))
        finally:
            self._messages.put(SandboxTransportError("Sandbox output stream closed"))

    def _receive(self, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxTransportError("Timed out waiting for sandbox response")
        try:
            result = self._messages.get(timeout=remaining)
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

    def _request(self, op: str, **fields) -> dict:
        with self._lock:
            if self._closed:
                raise SandboxTransportError("Sandbox transport is closed")
            identifier = uuid.uuid4().hex
            request = {"id": identifier, "op": op, **fields}
            wire = json.dumps(request, ensure_ascii=True) + "\n"
            if len(wire.encode()) > 1_048_576:
                raise ValueError("Kernel request exceeds 1 MiB")
            self.sandbox.stdin.write(wire)
            self.sandbox.stdin.drain()
            deadline = time.monotonic() + self.cell_timeout + self.startup_timeout + 30
            while True:
                result = self._receive(deadline)
                if result.get("id") == identifier:
                    self.runs = result.get("runs", self.runs)
                    return result

    def run(self, code: str) -> dict:
        return self._request("run", code=code)

    def status(self) -> dict:
        return self._request("status")

    def restart(self) -> dict:
        return self._request("restart")

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
        self.sandbox.filesystem.make_directory(str(PurePosixPath(remote).parent))
        self.sandbox.filesystem.copy_from_local(local_path, remote)

    def download(self, remote_path: str, local_path: str | Path) -> None:
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.sandbox.filesystem.copy_to_local(self._work_path(remote_path), local)

    def read_text(self, remote_path: str) -> str:
        return self.sandbox.filesystem.read_text(self._work_path(remote_path))

    def write_text(self, remote_path: str, text: str) -> None:
        remote = self._work_path(remote_path)
        self.sandbox.filesystem.make_directory(str(PurePosixPath(remote).parent))
        self.sandbox.filesystem.write_text(text, remote)

    def commit_work(self) -> None:
        """Commit v2 Volume writes without giving the editor any Modal credential."""
        with self._lock:
            process = self.sandbox.exec("sync", "/work", timeout=55)
            process.wait()
            if process.returncode != 0:
                raise SandboxTransportError(f"Volume commit failed with exit code {process.returncode}")

    def reload_work(self) -> None:
        """Call after the producing GPU/container commits its forward artifacts."""
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
