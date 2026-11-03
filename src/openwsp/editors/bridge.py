"""Durable, provider-independent exchange with an authenticated local editor.

The server must serialize mutations across machines (a filesystem lock is not a
distributed Volume lock). Credentials stay with the worker; this module never
reads an environment or authentication file. Owners and dispatch markers never
expire. A worker must retain the same durable idempotent editor state on resume.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

from ..budget import atomic_json
from .base import EditorError, EditorIncomplete, EditorRefused, ProviderUnavailable, UncertainRequest

_IDENTITY_FIELDS = ("schema_version", "run_id", "request_key", "conversation_index", "messages", "prompt_sha256")
_TERMINAL = ("completed", "blocked")
_SECRET_KEYS = frozenset({"api_key", "authorization", "access_token", "refresh_token", "id_token",
                          "token_secret", "credentials", "password", "secret", "auth_json", "environment", "env"})


class BridgeConflict(EditorError):
    """A request, owner, or immutable result differs from the recorded identity."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty identifier of at most 256 characters")
    return value


def _messages(value):
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, list) or not value:
        raise ValueError("Provide nonempty text or a list of text messages")
    for item in value:
        if (not isinstance(item, dict) or set(item) != {"role", "content"}
                or item["role"] not in ("user", "assistant") or not isinstance(item["content"], str)):
            raise ValueError("Only user/assistant role-content text messages are accepted")
    return _copy(value)


def _intent(run_id, request_key, conversation_index, messages):
    if isinstance(conversation_index, bool) or not isinstance(conversation_index, int) or conversation_index < 0:
        raise ValueError("conversation_index must be a nonnegative integer")
    messages = _messages(messages)
    prompt = messages if isinstance(messages, str) else _json(messages)
    intent = {"schema_version": 1, "run_id": _identifier(run_id, "run_id"),
              "request_key": _identifier(request_key, "request_key"), "conversation_index": conversation_index,
              "messages": messages, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    return {**intent, "request_hash": hashlib.sha256(_json(intent).encode()).hexdigest()}


def _validated(record, expected=None):
    if not isinstance(record, dict):
        raise BridgeConflict("Bridge returned no valid request record")
    try:
        identity = _intent(record["run_id"], record["request_key"], record["conversation_index"], record["messages"])
    except (KeyError, TypeError, ValueError):
        raise BridgeConflict("Bridge request identity is malformed") from None
    if any(record.get(k) != v for k, v in identity.items()):
        raise BridgeConflict("Bridge request hash or prompt digest does not match its payload")
    if expected and any(record.get(k) != expected[k] for k in (*_IDENTITY_FIELDS, "request_hash")):
        raise BridgeConflict("Bridge returned a different request identity")
    status = record.get("status")
    if status not in ("queued", "claimed", "dispatched", *_TERMINAL):
        raise BridgeConflict("Bridge request has an unknown state")
    if status in ("dispatched", *_TERMINAL):
        dispatch = record.get("dispatch")
        if (not isinstance(dispatch, dict) or dispatch.get("worker_id") != record.get("claimed_by")
                or not record.get("claimed_by") or dispatch.get("request_hash") != identity["request_hash"]):
            raise BridgeConflict("Bridge dispatch does not match its owner and request")
    if status == "completed":
        result = record.get("completion")
        if (not isinstance(result, dict) or result.get("request_hash") != identity["request_hash"]
                or not isinstance(result.get("text"), str) or not result["text"]):
            raise BridgeConflict("Bridge completion does not match its request")
        try:
            _final_receipt(result.get("receipt"), result["text"])
        except (TypeError, ValueError):
            raise BridgeConflict("Bridge completion has no valid credential-free receipt") from None
    if status == "blocked":
        error = record.get("blocked_error")
        if (not isinstance(error, dict) or error.get("request_hash") != identity["request_hash"]
                or any(not isinstance(error.get(k), str) for k in ("category", "error_type"))):
            raise BridgeConflict("Bridge error does not match its request")
    return record


def _receipt(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("A completion requires a nonempty JSON receipt")
    value = _copy(value)

    def inspect(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key.lower().replace("-", "_") in _SECRET_KEYS:
                    raise ValueError("Credential-bearing receipt fields are forbidden")
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)

    inspect(value)
    return value


def _final_receipt(value, text):
    receipt = _receipt(value)
    if (not isinstance(text, str) or not text or receipt.get("status") != "completed"
            or ("text" in receipt and receipt["text"] != text)):
        raise ValueError("A final receipt must attest to the same completed text")
    return receipt


class BridgeStore:
    """Atomic state per run, with permanent request and conversation ownership."""

    def __init__(self, root, *, commit=lambda: None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.commit = commit

    def _path(self, run_id):
        return self.root / (hashlib.sha256(_identifier(run_id, "run_id").encode()).hexdigest() + ".json")

    @contextmanager
    def _state(self, run_id):
        path = self._path(run_id)
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(path.read_text()) if path.exists() else {
                "schema_version": 1, "run_id": run_id, "conversation_owners": {}, "requests": {}}
            if state.get("schema_version") != 1 or state.get("run_id") != run_id:
                raise BridgeConflict("Bridge run identity changed")
            yield path, state

    def _save(self, path, state):
        atomic_json(path, state)
        self.commit()

    def submit(self, run_id, request_key, *, conversation_index, messages):
        identity = _intent(run_id, request_key, conversation_index, messages)
        with self._state(run_id) as (path, state):
            old = state["requests"].get(request_key)
            if old:
                _validated(old, identity)
                self.commit()
                return _copy(old)
            if any(r["status"] != "completed" for r in state["requests"].values()):
                raise BridgeConflict("Resolve the existing bridge request before submitting another")
            record = {**identity, "status": "queued", "claimed_by": None, "dispatch": None,
                      "completion": None, "blocked_error": None, "created_at": time.time()}
            state["requests"][request_key] = record
            self._save(path, state)
            return _copy(record)

    def get(self, run_id, request_key):
        _identifier(request_key, "request_key")
        with self._state(run_id) as (_, state):
            record = state["requests"].get(request_key)
            return _copy(_validated(record)) if record else None

    def pending(self, run_id):
        with self._state(run_id) as (_, state):
            return [_copy(_validated(record)) for record in state["requests"].values()
                    if record["status"] not in _TERMINAL]

    @staticmethod
    def _owned(state, request_key, request_hash, worker_id, *, require_claim=True):
        _identifier(worker_id, "worker_id")
        record = state["requests"].get(request_key)
        if (record is None or _validated(record)["request_hash"] != request_hash
                or record["run_id"] != state["run_id"] or record["request_key"] != request_key):
            raise BridgeConflict("Bridge request hash does not match the recorded intent")
        owner = record["claimed_by"]
        if owner not in (None, worker_id) or (require_claim and owner != worker_id):
            raise BridgeConflict("Bridge request belongs to another worker or is unclaimed")
        return record

    def claim(self, run_id, request_key, *, request_hash, worker_id):
        with self._state(run_id) as (path, state):
            record = self._owned(state, request_key, request_hash, worker_id, require_claim=False)
            conversation = str(record["conversation_index"])
            owner = state["conversation_owners"].get(conversation)
            if owner not in (None, worker_id):
                raise BridgeConflict("Bridge conversation belongs to another worker")
            if record["claimed_by"] is None:
                state["conversation_owners"][conversation] = worker_id
                record.update(status="claimed", claimed_by=worker_id, claimed_at=time.time())
                self._save(path, state)
            else:
                self.commit()
            return _copy(record)

    def dispatch(self, run_id, request_key, *, request_hash, worker_id):
        with self._state(run_id) as (path, state):
            record = self._owned(state, request_key, request_hash, worker_id)
            if record["dispatch"] is None:
                if record["status"] != "claimed":
                    raise BridgeConflict("Only an observed claim can be dispatched")
                record["dispatch"] = {"worker_id": worker_id, "request_hash": request_hash, "at": time.time()}
                record["status"] = "dispatched"
                self._save(path, state)
            else:
                self.commit()
            return _copy(record)

    def finish(self, run_id, request_key, *, request_hash, worker_id, text, receipt):
        if not isinstance(text, str) or not text:
            raise ValueError("A completion requires nonempty final text")
        completion = {"request_hash": request_hash, "text": text, "receipt": _final_receipt(receipt, text)}
        return self._terminal(run_id, request_key, request_hash, worker_id, "completed", "completion", completion)

    def block(self, run_id, request_key, *, request_hash, worker_id, error):
        if (not isinstance(error, dict) or set(error) != {"category", "error_type"}
                or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", v)
                       for v in error.values())):
            raise ValueError("Blocked errors contain only category and error_type identifiers, never exception text")
        blocked = {"request_hash": request_hash, **error}
        return self._terminal(run_id, request_key, request_hash, worker_id, "blocked", "blocked_error", blocked)

    def _terminal(self, run_id, request_key, request_hash, worker_id, status, field, value):
        with self._state(run_id) as (path, state):
            record = self._owned(state, request_key, request_hash, worker_id)
            if record["dispatch"] is None:
                raise BridgeConflict("A terminal result requires a durable dispatch marker")
            if record["status"] in _TERMINAL:
                if record["status"] != status or _json(record[field]) != _json(value):
                    raise BridgeConflict("Bridge terminal outcome is immutable")
                self.commit()
                return _copy(record)
            record.update(status=status, finished_at=time.time())
            record[field] = value
            self._save(path, state)
            return _copy(record)


class BridgeEditor:
    """Remote-side editor facade; the broker may be a BridgeStore or RPC proxy."""

    def __init__(self, state_dir, *, run_id, store, poll_interval=5, max_wait_seconds=3600,
                 reload=lambda: None, commit=lambda: None, on_poll=None, sleep=time.sleep,
                 max_output_bytes=None):
        if any(isinstance(n, bool) or not math.isfinite(n) or n <= 0 for n in (poll_interval, max_wait_seconds)):
            raise ValueError("Bridge polling and timeout must be finite and positive")
        if poll_interval > 60:
            raise ValueError("Bridge polling interval must not exceed 60 seconds")
        if max_output_bytes is not None and (isinstance(max_output_bytes, bool)
                or not isinstance(max_output_bytes, int) or max_output_bytes <= 0):
            raise ValueError("Output byte ceiling must be a positive integer")
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "bridge_editor_state.json"
        self.run_id, self.store = _identifier(run_id, "run_id"), store
        self.poll_interval, self.max_wait_seconds = poll_interval, max_wait_seconds
        self.reload, self.commit, self.on_poll, self.sleep = reload, commit, on_poll, sleep
        self.last_response = None
        self.max_output_bytes = max_output_bytes

    @contextmanager
    def _state(self):
        with self.root.joinpath("bridge_editor.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                "schema_version": 1, "run_id": self.run_id, "conversation_index": 0,
                "pending": None, "completed": {}, "blocked": None}
            if state.get("schema_version") != 1 or state.get("run_id") != self.run_id:
                raise BridgeConflict("Bridge editor run identity changed")
            yield state

    def _save(self, state):
        atomic_json(self.path, state)
        self.commit()

    def new_conversation(self):
        with self._state() as state:
            if state["pending"]:
                raise UncertainRequest("Cannot hand off while a bridge request remains unresolved")
            if state["blocked"]:
                raise ProviderUnavailable("The bridge conversation is blocked for inspection")
            state["conversation_index"] += 1
            self._save(state)

    def _call(self, operation, *args, **kwargs):
        try:
            self.reload()
            return getattr(self.store, operation)(*args, **kwargs)
        except EditorError:
            raise
        except Exception as error:  # noqa: BLE001 - never copy transport strings or credentials
            raise ProviderUnavailable(f"Bridge unavailable ({type(error).__name__}); persisted request may remain active") from None

    def complete(self, messages, *, request_key=None):
        messages = _messages(messages)
        with self._state() as state:
            key = str(request_key) if request_key is not None else hashlib.sha256(
                _json([state["conversation_index"], messages]).encode()).hexdigest()
            if key in state["completed"]:
                record = json.loads((self.root / state["completed"][key]).read_text())
                if record["messages"] != messages:
                    raise BridgeConflict("Bridge request key was reused with different input")
                expected = _intent(self.run_id, key, record["conversation_index"], messages)
                _validated(record, expected)
                self.commit()
                return self._deliver(record)
            if state["blocked"]:
                raise ProviderUnavailable("The bridge conversation is blocked for inspection")
            identity = _intent(self.run_id, key, state["conversation_index"], messages)
            if state["pending"] and state["pending"] != identity:
                raise UncertainRequest("Resolve the persisted bridge request before another request")
            if not state["pending"]:
                state["pending"] = identity
                self._save(state)
            # The broker implements idempotency for this exact hash. Unlike a
            # provider create, repeating submit after a lost reply cannot dispatch
            # another inference or change ownership.
            record = self._call("submit", self.run_id, key, conversation_index=state["conversation_index"], messages=messages)
            deadline = time.monotonic() + self.max_wait_seconds
            while True:
                _validated(record, identity)
                if record["status"] in _TERMINAL:
                    break
                if self.on_poll:
                    self.on_poll({k: record[k] for k in (
                        "run_id", "request_key", "conversation_index", "request_hash", "status", "claimed_by")})
                if time.monotonic() >= deadline:
                    raise UncertainRequest("Bridge wait expired; ownership and any inference remain unresolved")
                self.sleep(self.poll_interval)
                record = self._call("get", self.run_id, key)
                if record is None:
                    raise UncertainRequest("A submitted bridge request disappeared; no replacement was submitted")
            name = f"result_{identity['request_hash']}.json"
            atomic_json(self.root / name, record)
            state["completed"][key], state["pending"] = name, None
            if record["status"] == "blocked":
                state["blocked"] = record["blocked_error"]
            self._save(state)
            return self._deliver(record)

    def _deliver(self, record):
        if record["status"] == "blocked":
            error = record["blocked_error"]
            if not error or error["request_hash"] != record["request_hash"]:
                raise BridgeConflict("Blocked result does not match its request")
            self.last_response = {"status": "blocked", "bridge": _copy(error)}
            exception = {"uncertain_request": UncertainRequest, "editor_refused": EditorRefused}.get(
                error["category"], ProviderUnavailable)
            raise exception(f"Local editor blocked: {error['category']} ({error['error_type']})")
        result = record["completion"]
        if (not result or result["request_hash"] != record["request_hash"]
                or not isinstance(result.get("text"), str) or not result["text"]):
            raise BridgeConflict("Final result does not match its request")
        self.last_response = {**_receipt(result["receipt"]), "_openwsp_bridge": {
            k: record[k] for k in ("run_id", "request_key", "conversation_index", "request_hash", "claimed_by")}}
        if self.max_output_bytes is not None and len(result["text"].encode("utf-8")) > self.max_output_bytes:
            # The immutable completed receipt and its original usage are already
            # durable. Reject execution, never resample this request or bill twice.
            self.last_response.update(text=result["text"], execution_rejected="output_byte_limit",
                                      output_bytes=len(result["text"].encode("utf-8")),
                                      max_output_bytes=self.max_output_bytes)
            raise EditorIncomplete("output_byte_limit", self.last_response.get("response_id", record["request_hash"]))
        return result["text"]


def process_request(store, record, *, worker_id, editor):
    """Process one request using the same durable editor for its conversation.

    The caller selects persistent editor state by (run_id, conversation_index).
    ``complete`` must be idempotent by request_key and must never relaunch an
    uncertain prior inference. Already-dispatched requests also require a true
    ``has_request(request_key)`` predicate proving that local state survived.
    This function does not reset conversations or read/copy authentication.
    Exceptions upload identifiers, never their text.
    """
    identity = _validated(record)
    run_id, key = identity["run_id"], identity["request_key"]
    ownership = {"request_hash": identity["request_hash"], "worker_id": worker_id}
    claimed = _validated(store.claim(run_id, key, **ownership), identity)
    recovering = claimed["dispatch"] is not None
    _validated(store.dispatch(run_id, key, **ownership), identity)
    observed = _validated(store.get(run_id, key), identity)
    if observed["claimed_by"] != worker_id or observed["dispatch"] is None:
        raise BridgeConflict("Worker did not observe its durable exclusive dispatch")
    if observed["status"] in _TERMINAL:
        return observed
    if observed["status"] != "dispatched":
        raise BridgeConflict("Worker has no observed dispatched request")
    editor_key = "bridge:" + observed["request_hash"]
    if recovering:
        # Stable worker identity alone cannot establish that the local journal
        # survived. Never turn an ambiguous remote dispatch into a fresh model
        # call after disk loss, even when it happened before local launch intent.
        try:
            check = getattr(editor, "has_request", None)
            known = callable(check) and check(editor_key) is True
        except Exception as error:  # noqa: BLE001 - keep provider/transport exception text private
            return store.block(run_id, key, **ownership,
                               error={"category": "uncertain_request", "error_type": type(error).__name__})
        if not known:
            return store.block(run_id, key, **ownership,
                               error={"category": "uncertain_request", "error_type": "MissingLocalRequest"})
    try:
        text = editor.complete(observed["messages"], request_key=editor_key)
    except EditorError as error:
        category = ("uncertain_request" if isinstance(error, UncertainRequest)
                    else "editor_refused" if isinstance(error, EditorRefused) else "provider_unavailable")
        return store.block(run_id, key, **ownership, error={"category": category, "error_type": type(error).__name__})
    except Exception as error:  # noqa: BLE001 - outcome unknown, never retry a different request
        return store.block(run_id, key, **ownership,
                           error={"category": "uncertain_request", "error_type": type(error).__name__})
    try:
        receipt = _final_receipt(editor.last_response, text)
    except (TypeError, ValueError):
        return store.block(run_id, key, **ownership,
                           error={"category": "uncertain_request", "error_type": "InvalidReceipt"})
    return _validated(store.finish(run_id, key, **ownership, text=text, receipt=receipt), identity)
