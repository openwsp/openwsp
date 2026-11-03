"""Foreground bridge worker; ChatGPT authentication and Codex state stay local."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import re
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

from ..budget import atomic_json
from ..editors.bridge import BridgeConflict, process_request


def _run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("Codex run_id must match [a-z][a-z0-9_-]{0,63}")
    return value


def split_codex_settings(settings, *, detach):
    """Remove all local settings before a deployment or remote config exists."""
    if not isinstance(settings, dict):
        raise ValueError("Replication config must be a JSON object")  # noqa: TRY004 - CLI validation error
    remote = copy.deepcopy(settings)
    local = remote.pop("codex_local", None)
    if remote.get("editor", "openai") != "codex":
        if local is not None:
            raise ValueError("codex_local is only valid with editor='codex'")
        return remote, None
    if detach:
        raise ValueError("Codex requires --foreground so the authenticated local worker remains running")
    _run_id(remote.get("run_id"))
    allowed = {"home", "state_dir", "work_dir", "executable"}
    if not isinstance(local, dict) or set(local) - allowed or not {"home", "state_dir"} <= set(local):
        raise ValueError("codex_local requires home/state_dir and allows only work_dir/executable additionally")
    for key in ("home", "state_dir", "work_dir"):
        if key not in local:
            continue
        value = local[key]
        if not isinstance(value, str) or not value or "\x00" in value or not Path(value).is_absolute():
            raise ValueError(f"codex_local.{key} must be an absolute local directory path")
    if "executable" in local:
        value = local["executable"]
        if (not isinstance(value, str) or not value or "\x00" in value
                or ("/" in value and not Path(value).is_absolute())):
            raise ValueError("codex_local.executable must be a command name or absolute local path")
    return remote, local


def _private_directory(value, *, create=True):
    path = Path(value).absolute()
    if path.resolve() != path:
        raise ValueError("Local worker directories must not traverse symbolic links")
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Local worker directories must be owned by this user with mode 0700")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError("Local worker directories must be readable and writable")
    return path


def _pending_timeout(error):
    """Accept only the SDK's observed empty-output, still-running poll result."""
    from modal._functions import _Invocation
    from modal._traceback import traceback_contains_remote_call
    from modal.exception import TimeoutError as ModalTimeoutError
    from modal_proto import api_pb2

    if type(error) not in (TimeoutError, ModalTimeoutError) or traceback_contains_remote_call(error.__traceback__):
        return False
    # A remote user TimeoutError may lose its serialized traceback, and network
    # failures can use the same built-in class. Neither proves the call is still
    # running. Modal 1.6.3 raises its polling timeout directly in this SDK frame
    # after receiving no outputs and a positive unfinished-input count.
    traceback = error.__traceback__
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next
    code = getattr(_Invocation.poll_function, "__code__", None)
    if traceback is None or code is None or traceback.tb_frame.f_code is not code:
        return False
    response = traceback.tb_frame.f_locals.get("response")
    return (isinstance(response, api_pb2.FunctionGetOutputsResponse)
            and len(response.outputs) == 0 and response.num_unfinished_inputs > 0)


class CodexWorker:
    """One durable owner for a run, with an isolated editor per conversation."""

    def __init__(self, run_id, *, home, state_dir, work_dir=None, executable="codex", store=None,
                 editor_factory=None, poll_interval=5, sleep=time.sleep):
        self.run_id = _run_id(run_id)
        if (isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float))
                or not math.isfinite(poll_interval) or not 0 < poll_interval <= 60):
            raise ValueError("Worker polling interval must be positive and at most 60 seconds")
        self.home = _private_directory(home, create=False)
        base = _private_directory(state_dir)
        self.root = _private_directory(base / self.run_id)
        self.explicit_work = _private_directory(work_dir) if work_dir is not None else None
        self.executable = os.fspath(executable)
        self.store, self.editor_factory = store, editor_factory
        self.poll_interval, self.sleep = poll_interval, sleep
        self.worker_id = None
        self.work = None
        self._lock = None
        self._editors = {}

    @contextmanager
    def locked(self):
        """Hold before dispatch and throughout polling; fail if another worker owns it."""
        if self._lock is not None:
            raise RuntimeError("Codex worker lock is already held")
        descriptor = os.open(self.root / "worker.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another local Codex worker already owns this run") from None
            self._lock = lock
            try:
                self._load_identity()
                yield self
            finally:
                self._lock = None
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _load_identity(self):
        path = self.root / "worker.json"
        settings = {"run_id": self.run_id, "home": str(self.home), "executable": self.executable,
                    "explicit_work_dir": str(self.explicit_work) if self.explicit_work else None}
        if path.is_symlink():
            raise ValueError("Worker identity must not be a symbolic link")
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("schema_version") != 1 or record.get("settings") != settings:
                raise ValueError("Local worker settings differ from the durable identity")
            identity = record.get("worker_id")
            if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity):
                raise ValueError("Invalid durable worker identity")
        else:
            identity = uuid.uuid4().hex
            record = {"schema_version": 1, "worker_id": identity, "settings": settings}
            atomic_json(path, record)
        self.worker_id = identity
        if self.explicit_work:
            self.work = _private_directory(self.explicit_work / self.run_id)
        else:
            temporary = Path(tempfile.gettempdir()).resolve()
            base = _private_directory(temporary / f"openwsp-codex-{os.getuid()}")
            self.work = _private_directory(base / identity)
        # Project configuration is distinct from AGENTS/rules and would add
        # client instructions or tools even with --ignore-user-config.
        for ancestor in (self.work, *self.work.parents):
            if os.path.lexists(ancestor / ".codex" / "config.toml"):
                raise ValueError("Codex work_dir must be isolated from project .codex/config.toml")

    def _editor(self, index):
        if type(index) is not int or index < 0:
            raise BridgeConflict("Bridge conversation_index must be a nonnegative integer")
        if index not in self._editors:
            from ..editors.profiles import ASTRA_CODEX_PROFILE
            factory = self.editor_factory
            if factory is None:
                from ..editors.codex import CodexEditor
                factory = CodexEditor
            state = _private_directory(self.root / "conversations")
            self._editors[index] = factory(
                state / str(index), codex_home=self.home, executable=self.executable,
                work_dir=self.work / str(index), profile=ASTRA_CODEX_PROFILE,
            )
        return self._editors[index]

    def run(self, source_call):
        """Return only the driver's result; exceptions retain their original type."""
        with self.locked() if self._lock is None else nullcontext(self):
            if self.store is None:
                from .editor_bridge import ModalBridgeStore
                self.store = ModalBridgeStore()
            while True:
                try:
                    return source_call.get(timeout=0)
                except Exception as error:
                    if not _pending_timeout(error):
                        raise
                records = self.store.pending(self.run_id)
                if not isinstance(records, list):
                    raise BridgeConflict("Bridge pending response must be a list")
                for record in records:
                    if not isinstance(record, dict) or record.get("run_id") != self.run_id:
                        raise BridgeConflict("Bridge returned a request for another run")
                    editor = self._editor(record.get("conversation_index"))
                    process_request(self.store, record, worker_id=self.worker_id, editor=editor)
                self.sleep(self.poll_interval)
