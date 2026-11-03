"""Pinned, tool-free Codex CLI transport using caller-provisioned ChatGPT login.

No credentials are read or copied here, and subscription tokens are not priced
as API usage. A persisted launch intent is never automatically launched again.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..budget import atomic_json
from .base import EditorError, EditorRefused, EditorRunaway, ProviderUnavailable, UncertainRequest
from .output_guard import detect_runaway_formatting
from .profiles import ASTRA_CODEX_PROFILE, CodexProfile

# Backwards-compatible names for the default profile, not transport restrictions.
CLI_VERSION = ASTRA_CODEX_PROFILE.cli_version
MODEL = ASTRA_CODEX_PROFILE.model
INSTRUCTIONS = ASTRA_CODEX_PROFILE.instructions
CATALOG_SHA256 = ASTRA_CODEX_PROFILE.catalog_sha256
_DISABLED_FEATURES = (
    "code_mode",
    "collaboration_modes",
    "shell_tool",
    "unified_exec",
    "unified_exec_tty",
    "shell_snapshot",
    "view_image",
    "image_generation",
    "multi_agent",
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "code_mode_host",
    "skill_search",
    "skill_mcp_dependency_install",
    "sleep_tool",
    "tool_suggest",
    "goals",
    "hooks",
    "worktrees",
    "workspace_dependencies",
    "recommended_plugins",
    "code_mode_only",
    "current_time_reminder",
)
_ENVIRONMENT = {
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
}
_COUNTERS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def _private_directory(value, *, create=False):
    path = Path(value).absolute()
    if path.resolve() != path:
        raise ValueError("Private directories must not traverse symbolic links")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Private directories must be owned by this user with mode 0700")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError("Private directories must be readable and writable")
    return path


def _private_write(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _snapshot(raw, previous, *, new_thread):
    if not isinstance(raw, dict) or any(type(raw.get(key)) is not int or raw[key] < 0 for key in _COUNTERS):
        raise ValueError("Missing or malformed cumulative usage")
    counts = {key: raw[key] for key in _COUNTERS}
    if not any(counts.values()):
        return {"cumulative": counts, "delta": None, "basis": "zero_or_unavailable_cli_usage"}
    if previous is None:
        return {
            "cumulative": counts,
            "delta": counts.copy() if new_thread else None,
            "basis": "new_thread_total" if new_thread else "previous_snapshot_unavailable",
        }
    if any(counts[key] < previous[key] for key in _COUNTERS):
        return {"cumulative": counts, "delta": None, "basis": "cumulative_counter_regression"}
    return {
        "cumulative": counts,
        "delta": {key: counts[key] - previous[key] for key in _COUNTERS},
        "basis": "same_thread_cumulative_difference",
    }


class CodexEditor:
    """Subscription editor with explicit conversation handoff and fail-closed recovery.

    ``output_validator`` may reject semantic refusals using ``EditorRefused``;
    the CLI itself has no structured refusal event. No generated text executes.
    """

    def __init__(
        self,
        state_dir,
        *,
        codex_home,
        profile=ASTRA_CODEX_PROFILE,
        executable="codex",
        catalog_path=None,
        work_dir=None,
        max_wait_seconds=3600,
        poll_interval=1,
        on_poll=None,
        output_validator=None,
        commit=lambda: None,
    ):
        for number in (max_wait_seconds, poll_interval):
            if isinstance(number, bool) or not math.isfinite(number) or number <= 0:
                raise ValueError("Timeout and polling interval must be finite and positive")
        if poll_interval > 60:
            raise ValueError("Polling interval must not exceed 60 seconds")
        if not isinstance(profile, CodexProfile):
            raise TypeError("profile must be an immutable CodexProfile")
        self._profile = profile
        self.root = _private_directory(state_dir, create=True)
        self.home = _private_directory(codex_home)
        self.work = _private_directory(work_dir or self.root / "work", create=True)
        self.path = self.root / "codex_state.json"
        self.catalog = Path(catalog_path or profile.catalog_path)
        self.executable = shutil.which(os.fspath(executable))
        if not self.executable:
            raise ValueError("Codex CLI executable is unavailable")
        self.executable = str(Path(self.executable).resolve())
        self.max_wait_seconds, self.poll_interval = max_wait_seconds, poll_interval
        self.on_poll, self.output_validator, self.commit = on_poll, output_validator, commit
        self.last_response = None
        self.settings = {
            "cli_version": profile.cli_version,
            "model": profile.model,
            "effort": profile.effort,
            "verbosity": profile.verbosity,
            "catalog_sha256": profile.catalog_sha256,
            "codex_home": str(self.home),
            "work_dir": str(self.work),
            "instructions_sha256": hashlib.sha256(profile.instructions.encode()).hexdigest(),
        }
        # Preserve the already-deployed Astra state identity exactly. Explicit
        # alternatives still bind their changed context settings to the state.
        for name in ("context_window", "auto_compact_token_limit"):
            if getattr(profile, name) != getattr(ASTRA_CODEX_PROFILE, name):
                self.settings[name] = getattr(profile, name)

    def _environment(self):
        environment = {key: value for key, value in os.environ.items() if key in _ENVIRONMENT}
        environment["CODEX_HOME"] = str(self.home)
        return environment

    @property
    def profile(self):
        return self._profile

    def _check_installation(self):
        _private_directory(self.home)
        _private_directory(self.work)
        for ancestor in (self.work, *self.work.parents):
            if os.path.lexists(ancestor / ".codex" / "config.toml"):
                raise EditorError(
                    "Project Codex configuration exists above work_dir; use an isolated directory"
                )
        catalog_bytes = self.catalog.read_bytes()
        if hashlib.sha256(catalog_bytes).hexdigest() != self.profile.catalog_sha256:
            raise EditorError("Codex model catalog does not match the pinned fixture")
        try:
            models = json.loads(catalog_bytes)["models"]
            if (
                not isinstance(models, list)
                or sum(model["slug"] == self.profile.model for model in models) != 1
            ):
                raise ValueError("Selected model missing or ambiguous")
        except (ValueError, KeyError, TypeError):
            raise EditorError("Codex catalog must identify the selected profile model exactly once") from None
        result = subprocess.run(
            [self.executable, "--version"],
            env=self._environment(),
            cwd=self.work,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode or result.stdout.decode().strip() != f"codex-cli {self.profile.cli_version}":
            raise EditorError(f"This profile requires codex-cli {self.profile.cli_version}")

    def _new_state(self):
        return {
            "schema_version": 1,
            "adapter": "codex_cli_chatgpt",
            "settings": self.settings,
            "conversation_id": uuid.uuid4().hex,
            "thread_id": None,
            "usage_cumulative": None,
            "pending": None,
            "completed": {},
            "handoffs": 0,
            "blocked": None,
        }

    @contextmanager
    def _locked_state(self):
        descriptor = os.open(self.root / "codex.lock", os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else self._new_state()
            if state.get("schema_version") != 1 or state.get("adapter") != "codex_cli_chatgpt":
                raise EditorError("Unsupported Codex editor state")
            yield state

    def _save(self, state):
        atomic_json(self.path, state)
        self.commit()

    def has_request(self, request_key):
        """Read durable request identity without creating state or launching a CLI.

        Unknown or damaged state is not proof that dispatch is safe. Orphaned
        intents and inconsistent links raise ``UncertainRequest``; callers must
        not use a false result to replay an already-dispatched remote request.
        """
        if not isinstance(request_key, str) or not request_key:
            raise ValueError("request_key must be a nonempty string")
        lock_path = self.root / "codex.lock"
        try:
            if not os.path.lexists(lock_path):
                return self._recorded_request(request_key, lock_missing=True)
            descriptor = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                return self._recorded_request(request_key, lock_missing=False)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise UncertainRequest("Local Codex request records are busy, missing, or inconsistent") from None

    def _recorded_request(self, key, *, lock_missing):
        def read(path):
            if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("Record must be a regular file")
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise TypeError("Record must be an object")
            return value

        matches = []
        for path in self.root.glob("*/request.json"):
            if path.parent.is_symlink():
                raise ValueError("Intent directory must not be a symbolic link")
            intent = read(path)
            if intent.get("key") == key:
                if (
                    path.parent.is_symlink()
                    or intent["directory"] != path.parent.name
                    or intent["request_id"] != path.parent.name
                    or uuid.UUID(hex=intent["request_id"]).hex != intent["request_id"]
                ):
                    raise ValueError("Intent directory identity mismatch")
                prompt_path = path.parent / "prompt.txt"
                if (
                    prompt_path.is_symlink()
                    or _digest(prompt_path.read_bytes().decode()) != intent["input_sha256"]
                ):
                    raise ValueError("Intent prompt identity mismatch")
                matches.append((path, intent))
        if len(matches) > 1:
            raise ValueError("More than one intent uses the request key")
        if lock_missing and (self.path.exists() or matches):
            raise ValueError("Durable records have no matching lock")
        if not self.path.exists():
            if matches:
                raise ValueError("Orphaned request intent")
            return False
        state = read(self.path)
        if state["schema_version"] != 1 or state["adapter"] != "codex_cli_chatgpt":
            raise ValueError("Unsupported request state")
        completed, pending = state["completed"], state["pending"]
        if not isinstance(completed, dict) or (pending is not None and not isinstance(pending, dict)):
            raise ValueError("Malformed request index")
        is_pending = pending is not None and pending.get("key") == key
        is_complete = key in completed
        if is_pending and is_complete:
            raise ValueError("Request is both pending and completed")
        if not is_pending and not is_complete:
            if matches:
                raise ValueError("Request intent is missing from state")
            return False
        if not matches:
            raise ValueError("State has no matching request intent")
        path, intent = matches[0]
        if is_pending:
            if any(
                pending[name] != intent[name]
                for name in ("key", "request_id", "directory", "input_sha256", "thread_id")
            ):
                raise ValueError("Pending request differs from its intent")
        else:
            receipt_path = path.parent / "receipt.json"
            if completed[key] != str(receipt_path.relative_to(self.root)):
                raise ValueError("Receipt path differs from its intent")
            receipt = read(receipt_path)
            if (
                receipt["request_key"] != key
                or receipt["response_id"] != intent["request_id"]
                or receipt["input_sha256"] != intent["input_sha256"]
                or receipt["status"] not in ("completed", "failed", "refused", "invalid_output", "runaway")
            ):
                raise ValueError("Receipt differs from its intent")
        return True

    def new_conversation(self):
        with self._locked_state() as state:
            if state["pending"]:
                raise UncertainRequest("Cannot hand off while a Codex launch remains unresolved")
            fresh = self._new_state()
            fresh["completed"], fresh["handoffs"] = state["completed"], state["handoffs"] + 1
            self._save(fresh)

    @staticmethod
    def _input(messages):
        if isinstance(messages, str) and messages:
            return messages
        if (
            isinstance(messages, list)
            and len(messages) == 1
            and isinstance(messages[0], dict)
            and set(messages[0]) == {"role", "content"}
            and messages[0]["role"] == "user"
            and isinstance(messages[0]["content"], str)
            and messages[0]["content"]
        ):
            return messages[0]["content"]
        raise ValueError("Provide one new, nonempty user text message; the CLI retains prior turns")

    def _command(self, directory, thread_id):
        config = {
            "model_provider": "openai",
            "forced_login_method": "chatgpt",
            "cli_auth_credentials_store": "file",
            "model_reasoning_effort": self.profile.effort,
            "model_context_window": self.profile.context_window,
            "model_auto_compact_token_limit": self.profile.auto_compact_token_limit,
            "model_catalog_json": str(self.catalog.resolve()),
            "model_instructions_file": str(directory / "instructions.txt"),
            "approval_policy": "never",
            "sandbox_mode": "read-only",
            "project_doc_max_bytes": 0,
            "instructions": "",
            "developer_instructions": "",
            "skills.include_instructions": False,
            "skills.bundled.enabled": False,
            "include_permissions_instructions": False,
            "include_apps_instructions": False,
            "include_collaboration_mode_instructions": False,
            "include_environment_context": False,
            "include_environment_context_time": False,
            "web_search": "disabled",
            "agents.enabled": False,
            "tools.experimental_request_user_input.enabled": False,
            "tools.update_plan.enabled": False,
            **{f"features.{name}": False for name in _DISABLED_FEATURES},
        }
        if self.profile.verbosity is not None:
            config["model_verbosity"] = self.profile.verbosity
        command = [self.executable, "exec", "--sandbox", "read-only"]
        if thread_id:
            command += ["resume"]
        command += [
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--skip-git-repo-check",
            "--json",
            "-m",
            self.profile.model,
            "--output-last-message",
            str(directory / "final.txt"),
        ]
        for key, value in config.items():
            command += ["-c", f"{key}={json.dumps(value)}"]
        if thread_id:
            command += [thread_id]
        else:
            command += ["-C", str(self.work)]
        return command + ["-"]

    def complete(self, messages, *, request_key=None):
        prompt = self._input(messages)
        fingerprint = _digest(prompt)
        key = str(request_key if request_key is not None else fingerprint)
        if not key:
            raise ValueError("request_key must be nonempty")
        with self._locked_state() as state:
            if key in state["completed"]:
                receipt = json.loads((self.root / state["completed"][key]).read_text())
                if receipt["input_sha256"] != fingerprint:
                    raise EditorError("Request key was reused with different input")
                return self._deliver(receipt)
            if state["settings"] != self.settings:
                raise EditorError("Codex settings changed; an explicit new_conversation is required")
            pending = state["pending"]
            if pending:
                if pending["key"] != key or pending["input_sha256"] != fingerprint:
                    raise UncertainRequest("Resolve the persisted Codex launch before another request")
                return self._finish(state)
            if state["blocked"]:
                raise EditorError(
                    "Codex conversation is blocked; inspect its receipt before explicit handoff"
                )
            self._check_installation()
            if self.on_poll:
                self.on_poll()
            request_id = uuid.uuid4().hex
            directory = _private_directory(self.root / request_id, create=True)
            _private_write(directory / "prompt.txt", prompt.encode())
            _private_write(directory / "instructions.txt", self.profile.instructions.encode())
            command = self._command(directory, state["thread_id"])
            pending = {
                "key": key,
                "input_sha256": fingerprint,
                "request_id": request_id,
                "directory": request_id,
                "thread_id": state["thread_id"],
                "created_at": time.time(),
            }
            atomic_json(
                directory / "request.json",
                {**pending, "command": command, "settings": self.settings, "profile": self.profile.record()},
            )
            state["pending"] = pending
            self._save(state)  # After this point an interrupted launch is never repeated.
            self._launch(directory, command, state["thread_id"])
            return self._finish(state)

    def _launch(self, directory, command, expected_thread):
        started = time.monotonic()
        process = None
        reason = None
        violation = None
        position = 0
        try:
            with (
                (directory / "prompt.txt").open("rb") as source,
                (directory / "events.jsonl").open("xb") as events,
                (directory / "stderr.txt").open("xb") as errors,
            ):
                os.chmod(directory / "events.jsonl", 0o600)
                os.chmod(directory / "stderr.txt", 0o600)
                process = subprocess.Popen(
                    command,
                    stdin=source,
                    stdout=events,
                    stderr=errors,
                    cwd=self.work,
                    env=self._environment(),
                    start_new_session=True,
                )
                atomic_json(directory / "process.json", {"pid": process.pid, "started_at": time.time()})
                while process.poll() is None:
                    position, violation = self._scan_live(
                        directory / "events.jsonl", position, expected_thread
                    )
                    if violation:
                        reason = "prohibited_event"
                        self._stop(process)
                        break
                    remaining = self.max_wait_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        reason = "timeout"
                        self._stop(process)
                        break
                    if self.on_poll:
                        self.on_poll()
                    try:
                        process.wait(timeout=min(self.poll_interval, remaining))
                    except subprocess.TimeoutExpired:
                        pass
                events.flush()
                errors.flush()
                os.fsync(events.fileno())
                os.fsync(errors.fileno())
                final_path = directory / "final.txt"
                if final_path.exists():
                    os.chmod(final_path, 0o600)
                    with final_path.open("rb") as final_message:
                        os.fsync(final_message.fileno())
        except BaseException as error:
            if process is not None:
                self._stop(process)
            atomic_json(
                directory / "exit.json",
                {
                    "returncode": None if process is None else process.returncode,
                    "reason": "launch_or_poll_exception",
                    "exception_class": type(error).__name__,
                },
            )
            self.commit()
            raise
        atomic_json(
            directory / "exit.json",
            {
                "returncode": process.returncode,
                "reason": reason,
                "violation": violation,
                "seconds": time.monotonic() - started,
            },
        )
        self.commit()

    @staticmethod
    def _scan_live(path, position, expected_thread):
        """Inspect complete JSONL records without blocking on an unfinished line."""
        with path.open("rb") as stream:
            stream.seek(position)
            while line := stream.readline(16 * 1024 * 1024 + 1):
                if len(line) > 16 * 1024 * 1024:
                    return position, {"kind": "oversized_event"}
                if not line.endswith(b"\n"):
                    break
                position = stream.tell()
                try:
                    event = json.loads(line)
                    kind = event["type"]
                    if kind in ("error", "turn.failed"):
                        return position, {"kind": kind}
                    if kind.startswith("item.") and event["item"]["type"] not in (
                        "agent_message",
                        "reasoning",
                    ):
                        return position, {"kind": "prohibited_item", "item_type": event["item"]["type"]}
                    if kind == "thread.started":
                        thread = str(uuid.UUID(event["thread_id"]))
                        if expected_thread and thread != expected_thread:
                            return position, {"kind": "thread_identity_changed"}
                except (ValueError, KeyError, TypeError, AttributeError):
                    return position, {"kind": "malformed_event"}
        return position, None

    @staticmethod
    def _stop(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2)

    def _finish(self, state):
        pending = state["pending"]
        directory = self.root / pending["directory"]
        if not (directory / "exit.json").exists():
            raise UncertainRequest("Codex launch has no durable exit record; inspect it without relaunching")
        exit_record = json.loads((directory / "exit.json").read_text())
        if exit_record["reason"]:
            raise UncertainRequest("Codex launch was interrupted; completion and usage remain uncertain")
        try:
            parsed = self._events(directory, pending["thread_id"])
        except (OSError, ValueError, KeyError, TypeError):
            raise UncertainRequest(
                "Codex terminal evidence is missing or inconsistent; no relaunch"
            ) from None
        if not parsed["completed"] and not parsed["failed"]:
            raise UncertainRequest("Codex has no terminal turn receipt; no partial output is deliverable")
        outcome = "failed" if parsed["failed"] else "completed"
        text = parsed["text"]
        if outcome == "completed":
            if exit_record["returncode"] != 0:
                raise UncertainRequest(
                    "Codex completed event has a nonzero process exit; inspect before continuing"
                )
            try:
                if (directory / "final.txt").stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("Oversized final message")
                saved = (directory / "final.txt").read_text()
                if not text or saved not in (text, text + "\n"):
                    raise ValueError("Final message mismatch")
                usage = _snapshot(
                    parsed["usage"], state["usage_cumulative"], new_thread=pending["thread_id"] is None
                )
            except (OSError, ValueError, TypeError):
                raise UncertainRequest("Codex final message or usage evidence is inconsistent") from None
        else:
            usage = {"cumulative": None, "delta": None, "basis": "failed_turn_usage_unavailable"}
            if parsed["completed"]:
                try:
                    usage = _snapshot(
                        parsed["usage"], state["usage_cumulative"], new_thread=pending["thread_id"] is None
                    )
                except ValueError:
                    pass
        runaway = detect_runaway_formatting(text) if text else None
        if runaway:
            outcome = "runaway"
        if outcome == "completed" and self.output_validator:
            try:
                self.output_validator(text)
            except EditorRefused:
                outcome = "refused"
            except (EditorError, ValueError):
                outcome = "invalid_output"
        receipt = {
            "schema_version": 1,
            "request_key": pending["key"],
            "input_sha256": pending["input_sha256"],
            "response_id": pending["request_id"],
            "thread_id": parsed["thread_id"],
            "status": outcome,
            "text": text if outcome == "completed" else "",
            "usage": usage["cumulative"],
            "usage_snapshot": usage,
            "billing": "chatgpt_subscription_no_api_price_inferred",
            "event_file": str((directory / "events.jsonl").relative_to(self.root)),
            "final_file": str((directory / "final.txt").relative_to(self.root)),
            "runaway": runaway,
            "exit": exit_record,
            "settings": self.settings,
            "profile": self.profile.record(),
            "recorded_at": time.time(),
        }
        receipt_path = directory / "receipt.json"
        atomic_json(receipt_path, receipt)
        state["completed"][pending["key"]] = str(receipt_path.relative_to(self.root))
        state["pending"] = None
        state["thread_id"] = parsed["thread_id"]
        state["usage_cumulative"] = (
            usage["cumulative"]
            if usage["basis"] not in ("zero_or_unavailable_cli_usage", "cumulative_counter_regression")
            else None
        )
        if outcome != "completed":
            state["blocked"] = {"request_key": pending["key"], "status": outcome}
        self._save(state)
        return self._deliver(receipt)

    @staticmethod
    def _events(directory, expected_thread):
        parsed = {"thread_id": None, "completed": False, "failed": False, "usage": None, "text": ""}
        with (directory / "events.jsonl").open("rb") as stream:
            while line := stream.readline(16 * 1024 * 1024 + 1):
                if len(line) > 16 * 1024 * 1024:
                    raise ValueError("Oversized event")
                event = json.loads(line)
                kind = event["type"]
                if kind == "thread.started":
                    thread = str(uuid.UUID(event["thread_id"]))
                    if parsed["thread_id"] or (expected_thread and thread != expected_thread):
                        raise ValueError("Thread identity changed")
                    parsed["thread_id"] = thread
                elif kind == "turn.completed":
                    if parsed["completed"]:
                        raise ValueError("Multiple terminal events")
                    parsed["completed"], parsed["usage"] = True, event["usage"]
                elif kind in ("turn.failed", "error"):
                    parsed["failed"] = True
                elif kind in ("item.started", "item.updated", "item.completed"):
                    item = event["item"]
                    if item["type"] not in ("agent_message", "reasoning"):
                        parsed["failed"] = True
                    elif kind == "item.completed" and item["type"] == "agent_message":
                        if parsed["completed"] or not isinstance(item["text"], str):
                            raise ValueError("Invalid final message ordering")
                        parsed["text"] = item["text"]
                elif kind != "turn.started":
                    raise ValueError("Unknown CLI event")
        if not parsed["thread_id"]:
            raise ValueError("Missing thread identity")
        return parsed

    def _deliver(self, receipt):
        self.last_response = receipt
        status = receipt["status"]
        if status == "runaway":
            raise EditorRunaway(
                receipt["runaway"]["kind"],
                receipt["response_id"],
                receipt["runaway"],
                usage_pending=receipt["usage"] is None,
            )
        if status == "refused":
            raise EditorRefused("Codex output validator identified a refusal; conversation blocked")
        if status == "failed":
            raise ProviderUnavailable(
                "Codex reported an error or prohibited item; inspect private event records"
            )
        if status != "completed":
            raise EditorError("Codex output was not accepted; conversation blocked")
        return receipt["text"]
