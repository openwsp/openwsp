"""Resumable, tool-free Astra Responses adapter.

Requests use only the new message and previous_response_id. The provider keeps
reasoning items between turns; a handoff deliberately creates a new chain. The
driver should pass a stable turn key, save its result, then move to the next key.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from ..budget import BudgetLedger, astra_cost, astra_upper_bound, atomic_json
from .base import (
    EditorError,
    EditorIncomplete,
    EditorRefused,
    EditorRunaway,
    ProviderUnavailable,
    UncertainRequest,
)
from .output_guard import MAX_SCAN_CHARS, detect_runaway_formatting

INSTRUCTIONS = "Answer the user's request."
EFFORTS = ("max", "xhigh", "high", "medium", "low")
ACTIVE_STATUSES = ("queued", "in_progress")
TERMINAL_STATUSES = ("completed", "incomplete", "failed", "cancelled")
STREAM_CHECKPOINT_SECONDS = 1.0
STREAM_CHECKPOINT_EVENTS = 256


def _dict(value):
    if isinstance(value, dict):
        return value
    return value.model_dump(mode="json")


def _error_detail(error):
    """Keep identifiers, never an exception string that may contain request data."""
    body = getattr(error, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    body = body if isinstance(body, dict) else {}
    return {"class": type(error).__name__, "status_code": getattr(error, "status_code", None),
            "code": body.get("code"), "param": body.get("param"),
            "request_id": getattr(error, "request_id", None)}


class AstraEditor:
    def __init__(self, state_dir, budget: BudgetLedger, *, client=None, model="gpt-6-astra",
                 effort="max", max_output_tokens=32_768, poll_interval=5,
                 max_wait_seconds=3600, on_poll=None, sleep=time.sleep, commit=lambda: None,
                 verbosity=None):
        if model != "gpt-6-astra":
            raise ValueError("This adapter's verified billing schedule is for gpt-6-astra only")
        if effort not in EFFORTS or not 1 <= max_output_tokens <= 128_000:
            raise ValueError("Invalid effort or output token ceiling")
        if not 0 <= poll_interval <= 60 or max_wait_seconds <= 0:
            raise ValueError("Invalid polling interval or wait deadline")
        if verbosity not in (None, "low", "medium", "high"):
            raise ValueError("Verbosity must be omitted, low, medium, or high")
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "editor_state.json"
        self.budget = budget
        self.client = client
        self.model, self.effort = model, effort
        self.max_output_tokens = max_output_tokens
        self.verbosity = verbosity
        self.poll_interval, self.max_wait_seconds = poll_interval, max_wait_seconds
        self.on_poll, self.sleep = on_poll, sleep
        self.commit = commit
        self.last_response = None

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI
            # SDK automatic POST retries can duplicate requests after lost replies.
            # HTTP client inherits the environment proxy and CA configuration.
            self.client = OpenAI(max_retries=0, timeout=900)
        elif hasattr(self.client, "with_options") and getattr(self.client, "max_retries", 0) != 0:
            self.client = self.client.with_options(max_retries=0)
        return self.client

    @contextmanager
    def _locked_state(self):
        with self.root.joinpath("editor.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else self._new_state()
            yield state

    def _new_state(self):
        return {"schema_version": 1, "conversation_id": uuid.uuid4().hex,
                "previous_response_id": None, "context_tokens": 0, "effort": self.effort,
                "pending": None, "completed": {}, "handoffs": 0}

    def new_conversation(self):
        with self._locked_state() as state:
            self._check_previous_runaway(state)
            if state.get("blocked"):
                self._raise_runaway(state)
            if state["pending"]:
                raise EditorError("Cannot hand off while an editor request is unresolved")
            new = self._new_state()
            new["handoffs"] = state["handoffs"] + 1
            atomic_json(self.path, new)
            self.commit()

    def acknowledge_runaway(self, response_id, *, reason, evidence):
        """Record explicit review of a settled response without changing its history.

        This does not execute old output, submit another request, or certify the
        caller's protocol decision. Cancelled/failed/unknown outcomes cannot be
        acknowledged as a preserved conversation trajectory.
        """
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("Review requires a response ID")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Review requires a nonempty reason")
        if (not isinstance(evidence, dict) or not isinstance(evidence.get("source"), str)
                or not evidence["source"].strip()):
            raise ValueError("Review evidence must be a JSON object with a nonempty source")
        try:
            evidence = json.loads(json.dumps(evidence, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError):
            raise ValueError("Review evidence must be finite JSON") from None
        with self._locked_state() as state:
            if state["pending"]:
                raise EditorError("Cannot acknowledge a runaway with a pending or unknown usage receipt")
            reviewed = self._reviewed_runaway(state, response_id)
            if reviewed:
                if reviewed["reason"] != reason or reviewed["evidence"] != evidence:
                    raise EditorError("Conflicting runaway review; the existing audit is immutable")
                self.commit()  # retry a previously interrupted Volume commit, without rewriting its audit
                return reviewed
            blocked = state.get("blocked")
            if blocked and blocked["response_id"] != response_id:
                raise EditorError("Review does not match the blocked response")
            matches = []
            for key, name in state["completed"].items():
                record = json.loads((self.root / name).read_text())
                if record["response_id"] == response_id:
                    matches.append((key, name, record))
            if len(matches) != 1:
                raise EditorError("Review requires one matching saved terminal receipt")
            key, name, record = matches[0]
            if record["status"] not in ("completed", "incomplete") or not record.get("usage"):
                raise EditorError("Only completed/incomplete responses with known usage can be reviewed")
            diagnostic = record.get("runaway") or detect_runaway_formatting(record["text"])
            if not diagnostic:
                raise EditorError("Response has no recorded runaway formatting to acknowledge")
            rid = Path(name).stem.removeprefix("receipt_")
            row = self.budget.snapshot()["reservations"].get(rid)
            if (not row or row["kind"] != "editor" or row["status"] != "settled"
                    or row.get("usage") != record["usage"]
                    or row.get("receipt_metadata", {}).get("response_id") != response_id
                    or Decimal(row["actual_usd"]) != Decimal(record["cost_usd"])):
                raise EditorError("Review requires the matching settled editor usage receipt")
            review_path = self._review_path(response_id)
            identity = {"response_id": response_id, "request_key": key, "reservation_id": rid,
                        "terminal_status": record["status"], "receipt_file": name,
                        "receipt_sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest(),
                        "source": evidence["source"], "reason": reason, "evidence": evidence}
            if review_path.exists():
                # A crash may leave the immutable audit before its state index.
                reviewed = json.loads(review_path.read_text())
                if any(reviewed.get(k) != v for k, v in identity.items()):
                    raise EditorError("Conflicting runaway review; the existing audit is immutable")
            else:
                reviewed = {"schema_version": 1, **identity, "reviewed_at": time.time(),
                            "diagnostic": diagnostic, "action": "AstraEditor.acknowledge_runaway"}
                atomic_json(review_path, reviewed)
            state.setdefault("runaway_reviews", {})[response_id] = hashlib.sha256(review_path.read_bytes()).hexdigest()
            if blocked:
                state.pop("blocked")
            self._persist(state)
            return reviewed

    def _review_path(self, response_id):
        return self.root / f"runaway_review_{hashlib.sha256(response_id.encode()).hexdigest()}.json"

    def _reviewed_runaway(self, state, response_id):
        digest = state.get("runaway_reviews", {}).get(response_id)
        if digest is None:
            return None
        try:
            raw = self._review_path(response_id).read_bytes()
            reviewed = json.loads(raw)
            receipt_raw = (self.root / reviewed["receipt_file"]).read_bytes()
        except (OSError, ValueError, KeyError, TypeError):
            raise EditorError("Runaway review audit is missing or unreadable") from None
        if (hashlib.sha256(raw).hexdigest() != digest or reviewed.get("response_id") != response_id
                or hashlib.sha256(receipt_raw).hexdigest() != reviewed.get("receipt_sha256")):
            raise EditorError("Runaway review audit changed; request requires inspection")
        return reviewed

    @staticmethod
    def _input(messages):
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if not isinstance(messages, list) or not messages:
            raise ValueError("Provide a nonempty list of new text messages")
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("Only role/content text messages are accepted")
            if message["role"] not in ("user", "assistant") or not isinstance(message["content"], str):
                raise ValueError("Only user/assistant text messages are accepted")
        return messages

    def complete(self, messages, *, request_key=None):
        inputs = self._input(messages)
        fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        with self._locked_state() as state:
            # An explicit tNNN key distinguishes identical messages on two turns.
            key = str(request_key or fingerprint)
            self._check_previous_runaway(state)
            if key in state["completed"]:
                record = json.loads((self.root / state["completed"][key]).read_text())
                if record["input_sha256"] != fingerprint:
                    raise EditorError("A completed request key was reused with different input")
                self.last_response = record
                diagnostic = record.get("runaway") or detect_runaway_formatting(record["text"])
                if diagnostic:
                    if self._reviewed_runaway(state, record["response_id"]):
                        # Reviewing continuity never turns this old partial output
                        # into a deliverable answer or re-blocks a later request.
                        raise EditorRunaway(diagnostic["kind"], record["response_id"], diagnostic, usage_pending=False)
                    self._block(state, diagnostic, record["response_id"], key)
                    self._raise_runaway(state)
                return self._deliver(record)
            if state.get("blocked") and not (state["pending"] and state["pending"]["key"] == key):
                self._raise_runaway(state)
            pending = state["pending"]
            if pending and (pending["key"] != key or pending["input_sha256"] != fingerprint):
                raise EditorError("Finish the persisted pending request before submitting another turn")
            if pending and not pending.get("response_id"):
                raise UncertainRequest("A prior create has no response ID; reservation retained, inspect its request ID")
            deadline = time.monotonic() + self.max_wait_seconds
            if pending:
                diagnostic = detect_runaway_formatting(pending.get("stream_tail", ""))
                if diagnostic and not state.get("blocked"):
                    self._block(state, diagnostic, pending["response_id"], key)
                if state.get("blocked"):
                    return self._finish(self._cancel_runaway(state), state, fingerprint)
            if not pending:
                # Text token counts cannot exceed UTF-8 bytes; add message/system
                # framing allowance. Last actual input+output bounds carried context.
                new_bytes = len(json.dumps(inputs, ensure_ascii=False).encode("utf-8"))
                input_bound = state["context_tokens"] + new_bytes + len(INSTRUCTIONS) + 4096 + 64 * len(inputs)
                bound = astra_upper_bound(input_bound, self.max_output_tokens)
                rid = uuid.uuid4().hex
                self.budget.reserve("editor", bound, f"editor:{key}", reservation_id=rid,
                                    metadata={"input_token_bound": input_bound,
                                              "max_output_tokens": self.max_output_tokens,
                                              "conversation_id": state["conversation_id"]})
                pending = {"key": key, "input_sha256": fingerprint, "reservation_id": rid,
                           "client_request_id": uuid.uuid4().hex, "response_id": None,
                           "input_token_bound": input_bound, "status": "submitting",
                           "started_at": time.time(), "stream": True, "sequence_number": None,
                           "stream_tail": ""}
                state["pending"] = pending
                atomic_json(self.path, state)  # durable BEFORE the possibly billed POST
                request = {
                    "model": self.model, "instructions": INSTRUCTIONS, "input": inputs,
                    "reasoning": {"effort": state["effort"]}, "background": True, "store": True, "stream": True,
                    "max_output_tokens": self.max_output_tokens,
                    "prompt_cache_key": state["conversation_id"],
                    "prompt_cache_options": {"ttl": "30m"}, "tools": [],
                    "truncation": "disabled", "service_tier": "default",
                    "metadata": {"openwsp_request": pending["client_request_id"]},
                    "extra_headers": {"X-Client-Request-Id": pending["client_request_id"]},
                }
                if state["previous_response_id"]:
                    request["previous_response_id"] = state["previous_response_id"]
                if self.verbosity is not None:
                    request["text"] = {"verbosity": self.verbosity}
                request_path = self.root / f"request_{rid}.json"
                atomic_json(request_path, request)
                self.commit()  # Volume ledger and request intent must precede billed POST
                created = self._create(request, state)
                if isinstance(created, dict) or hasattr(created, "model_dump"):
                    # Preserve recovery for nonstreaming clients and old requests.
                    response = _dict(created)
                    pending["stream"] = False
                    self._save_response(response, state)
                else:
                    response = self._consume_stream(created, state, deadline)
            else:
                if pending.get("stream") and pending.get("status") not in TERMINAL_STATUSES:
                    stream = self._retrieve(pending["response_id"], stream=True,
                                            starting_after=pending.get("sequence_number"))
                    response = self._consume_stream(stream, state, deadline)
                else:
                    response = self._retrieve(pending["response_id"])
            while response.get("status") in ACTIVE_STATUSES:
                pending["status"] = response["status"]
                atomic_json(self.path, state)
                self.commit()
                if self.on_poll:
                    self.on_poll(dict(pending))
                if time.monotonic() >= deadline:
                    raise EditorError("Editor polling deadline reached; response remains resumable, reservation retained")
                self.sleep(self.poll_interval)
                response = self._retrieve(pending["response_id"])
            return self._finish(response, state, fingerprint)

    def _create(self, request, state):
        pending = state["pending"]
        while True:
            try:
                return self._get_client().responses.create(**request)
            except Exception as error:  # noqa: BLE001 - preserve reservations for any uncertain transport
                detail = _error_detail(error)
                pending["error"] = detail
                atomic_json(self.path, state)
                self.commit()
                status, code = detail["status_code"], detail["code"]
                # Rejected effort requests have not generated output. Only this
                # specific validation error permits changing the requested effort.
                if status == 400 and detail["param"] in ("reasoning.effort", "reasoning"):
                    current = EFFORTS.index(state["effort"])
                    if current + 1 < len(EFFORTS):
                        state["effort"] = EFFORTS[current + 1]
                        request["reasoning"]["effort"] = state["effort"]
                        pending.setdefault("effort_rejections", []).append(detail)
                        atomic_json(self.path, state)
                        atomic_json(self.root / f"request_{pending['reservation_id']}.json", request)
                        self.commit()
                        continue
                if status in (400, 401, 403, 404, 422, 429):
                    self.budget.release(pending["reservation_id"], f"Provider rejected create: HTTP {status}, {code}")
                    state["pending"] = None
                    atomic_json(self.path, state)
                    self.commit()
                    if status in (401, 403) or code in ("insufficient_quota", "billing_hard_limit_reached"):
                        raise ProviderUnavailable(f"Editor blocked: HTTP {status}, code={code}") from None
                    if code in ("content_filter", "content_policy_violation", "moderation_blocked"):
                        raise EditorRefused("Provider filtered the response before delivery") from None
                    raise EditorError(f"Editor request rejected: HTTP {status}, code={code}") from None
                # No documented POST idempotency guarantee: even a server error
                # may follow successful creation. GET retries are safe; POST isn't.
                raise UncertainRequest("Editor create outcome is uncertain; request ID saved, no duplicate submitted") from None

    def _retrieve(self, response_id, *, stream=False, starting_after=None):
        options = {"stream": True} if stream else {}
        if stream and starting_after is not None:
            options["starting_after"] = starting_after
        for attempt in range(6):
            try:
                result = self._get_client().responses.retrieve(response_id, **options)
                return result if stream else _dict(result)
            except Exception as error:  # noqa: BLE001 - preserve reservations for any uncertain transport
                detail = _error_detail(error)
                status, code = detail["status_code"], detail["code"]
                if status in (401, 403) or code in ("insufficient_quota", "billing_hard_limit_reached"):
                    raise ProviderUnavailable(f"Editor retrieval blocked: HTTP {status}, code={code}") from None
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt == 5:
                    raise EditorError(f"Editor retrieval failed: HTTP {status}, code={code}; response remains resumable") from None
                if self.on_poll:
                    self.on_poll({"response_id": response_id, "status": "retrieval_retry", "attempt": attempt + 1})
                self.sleep(min(30, 2 ** attempt))
        raise AssertionError("unreachable")

    def _persist(self, state):
        atomic_json(self.path, state)
        self.commit()

    def _save_response(self, response, state, *, commit=True):
        pending = state["pending"]
        if pending.get("response_id") not in (None, response["id"]):
            raise EditorError("Provider response ID changed; request requires inspection")
        pending["response_id"] = response["id"]
        pending["status"] = response.get("status")
        atomic_json(self.root / f"response_{pending['reservation_id']}.json", response)
        if commit:
            self._persist(state)

    def _block(self, state, diagnostic, response_id, key):
        state.setdefault("blocked", {"reason": diagnostic["kind"], "response_id": response_id,
                                     "request_key": key, "diagnostic": diagnostic, "detected_at": time.time()})
        self._persist(state)

    def _raise_runaway(self, state):
        blocked = state["blocked"]
        receipt = state["completed"].get(blocked["request_key"])
        if receipt:
            self.last_response = json.loads((self.root / receipt).read_text())
        raise EditorRunaway(blocked["reason"], blocked["response_id"], blocked["diagnostic"],
                            usage_pending=state["pending"] is not None)

    def _check_previous_runaway(self, state):
        """Guard legacy saved responses before another POST, without rewriting them."""
        if state.get("blocked") or state["pending"] or not state.get("previous_response_id"):
            return
        if self._reviewed_runaway(state, state["previous_response_id"]):
            return
        for key, path in reversed(list(state["completed"].items())):
            record = json.loads((self.root / path).read_text())
            if record["response_id"] == state["previous_response_id"]:
                diagnostic = record.get("runaway") or detect_runaway_formatting(record["text"])
                if diagnostic:
                    self.last_response = record
                    self._block(state, diagnostic, record["response_id"], key)
                return

    def _consume_stream(self, stream, state, deadline):
        """Reconnect GETs only; each durable cursor includes the detector's tail."""
        pending = state["pending"]
        for reconnect in range(6):
            try:
                response = self._consume_events(stream, state, deadline)
            finally:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001, S110 - reader close must not hide a saved terminal result
                    pass
            if response is not None:
                return response
            if not pending.get("response_id"):
                raise UncertainRequest("Editor stream ended before its response ID; no duplicate submitted")
            if reconnect == 5 or time.monotonic() >= deadline:
                raise EditorError("Editor stream interrupted; saved response ID and cursor remain resumable")
            if self.on_poll:
                self.on_poll({"response_id": pending["response_id"], "status": "stream_reconnect"})
            self.sleep(min(30, 2 ** reconnect))
            stream = self._retrieve(pending["response_id"], stream=True,
                                    starting_after=pending.get("sequence_number"))
        raise AssertionError("unreachable")

    def _consume_events(self, stream, state, deadline):
        pending = state["pending"]
        iterator = iter(stream)
        last_checkpoint = last_poll = time.monotonic()
        since_checkpoint = 0
        poll_seconds = max(1.0, self.poll_interval)
        with (self.root / f"stream_{pending['reservation_id']}.jsonl").open("a") as journal:
            def checkpoint():
                nonlocal last_checkpoint, since_checkpoint
                journal.flush()
                os.fsync(journal.fileno())
                # Cursor and detector tail become durable together. Until this
                # checkpoint they are only in memory; a restart replays from the
                # older persisted pair, never from the append-only evidence.
                self._persist(state)
                last_checkpoint = time.monotonic()
                since_checkpoint = 0

            while time.monotonic() < deadline:
                try:
                    event = _dict(next(iterator))
                except StopIteration:
                    checkpoint()  # reconnect only after the buffered cursor is durable
                    return None
                except Exception as error:  # noqa: BLE001 - read errors only, not durable mutations
                    detail = _error_detail(error)
                    pending["stream_error"] = detail
                    checkpoint()
                    if detail["status_code"] in (401, 403) or detail["code"] in ("insufficient_quota", "billing_hard_limit_reached"):
                        raise ProviderUnavailable("Editor stream blocked; response remains resumable") from None
                    return None
                if event.get("type") == "keepalive":
                    # Keepalive counters belong to this connection, not the
                    # stored response event log used by starting_after.
                    now = time.monotonic()
                    poll_due = self.on_poll is not None and now - last_poll >= poll_seconds
                    if poll_due or (since_checkpoint and now - last_checkpoint >= STREAM_CHECKPOINT_SECONDS):
                        checkpoint()
                    if poll_due:
                        self.on_poll({"response_id": pending.get("response_id"), "status": pending["status"],
                                      "sequence_number": pending.get("sequence_number")})
                        last_poll = time.monotonic()
                    continue
                response = event.get("response")
                new_id = bool(response and pending.get("response_id") is None)
                if new_id:
                    # Retain the ID even if the following cursor is malformed.
                    self._save_response(response, state, commit=False)
                if response and response["id"] != pending.get("response_id"):
                    checkpoint()
                    raise EditorError("Provider stream response ID changed; request requires inspection")
                sequence = event.get("sequence_number")
                if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
                    checkpoint()
                    raise EditorError("Editor stream has no valid sequence number; reservation retained")
                if pending.get("sequence_number") is not None and sequence <= pending["sequence_number"]:
                    continue
                if response and not new_id:
                    self._save_response(response, state, commit=False)
                if not pending.get("response_id"):
                    checkpoint()
                    raise UncertainRequest("Editor stream has no response ID; no duplicate submitted")
                if event.get("type") == "response.output_text.delta":
                    delta = event.get("delta")
                    if not isinstance(delta, str):
                        checkpoint()
                        raise EditorError("Invalid editor text delta; reservation retained")
                    pending["stream_tail"] = (pending.get("stream_tail", "") + delta)[-(MAX_SCAN_CHARS + 1):]
                journal.write(json.dumps(event, sort_keys=True) + "\n")
                pending["sequence_number"] = sequence
                since_checkpoint += 1
                diagnostic = detect_runaway_formatting(pending.get("stream_tail", ""))
                if diagnostic:
                    journal.flush()
                    os.fsync(journal.fileno())
                    self._block(state, diagnostic, pending["response_id"], pending["key"])
                    if response and response.get("status") not in ACTIVE_STATUSES:
                        return response
                    return self._cancel_runaway(state)
                now = time.monotonic()
                poll_due = self.on_poll is not None and now - last_poll >= poll_seconds
                terminal = bool(response and response.get("status") not in ACTIVE_STATUSES)
                stream_error = event.get("type") == "error"
                if (new_id or terminal or stream_error or poll_due
                        or since_checkpoint >= STREAM_CHECKPOINT_EVENTS
                        or now - last_checkpoint >= STREAM_CHECKPOINT_SECONDS):
                    checkpoint()
                if response and response.get("status") not in ACTIVE_STATUSES:
                    return response
                if stream_error:
                    if event.get("code") in ("insufficient_quota", "billing_hard_limit_reached",
                                             "invalid_api_key", "authentication_error"):
                        raise ProviderUnavailable("Editor stream blocked; response and reservation retained")
                    raise EditorError("Provider emitted a stream error; reservation and response ID retained")
                if poll_due:
                    self.on_poll({"response_id": pending["response_id"], "status": pending["status"],
                                  "sequence_number": sequence})
                    last_poll = time.monotonic()
            checkpoint()
            return None

    def _cancel_runaway(self, state):
        """Cancel the known response; uncertain/missing usage keeps the reservation."""
        pending = state["pending"]
        if (pending.get("cancel_status") in ("uncertain", "response_received")
                or pending.get("status") in TERMINAL_STATUSES):
            try:
                response = self._retrieve(pending["response_id"])
            except EditorError:
                self._raise_runaway(state)
            self._save_response(response, state)
            if response.get("status") not in ACTIVE_STATUSES:
                return response
        pending["cancel_status"] = "intent"
        self._persist(state)  # must survive a crash before the idempotent cancel RPC
        try:
            response = _dict(self._get_client().responses.cancel(pending["response_id"], timeout=30))
        except Exception as error:  # noqa: BLE001 - never resubmit generation after an uncertain cancel
            pending["cancel_status"] = "uncertain"
            pending["cancel_error"] = detail = _error_detail(error)
            self._persist(state)
            if detail["status_code"] in (401, 403) or detail["code"] in ("insufficient_quota", "billing_hard_limit_reached"):
                self._raise_runaway(state)
            try:
                response = self._retrieve(pending["response_id"])
            except EditorError:
                self._raise_runaway(state)
        else:
            pending["cancel_status"] = "response_received"
            atomic_json(self.root / f"cancel_{pending['reservation_id']}.json", response)
            self._persist(state)
        self._save_response(response, state)
        if response.get("status") in ACTIVE_STATUSES:
            self._raise_runaway(state)
        return response

    def _finish(self, response, state, fingerprint):
        pending = state["pending"]
        rid = pending["reservation_id"]
        atomic_json(self.root / f"response_{rid}.json", response)
        response_text = "".join(part.get("text", "") for item in response.get("output", [])
                                if item.get("type") == "message" for part in item.get("content", [])
                                if part.get("type") == "output_text")
        diagnostic = detect_runaway_formatting(response_text)
        if diagnostic:
            self._block(state, diagnostic, response["id"], pending["key"])
        usage = response.get("usage")
        if usage is None:
            pending["status"] = response.get("status")
            atomic_json(self.path, state)
            self.commit()
            if state.get("blocked"):
                self._raise_runaway(state)
            error = response.get("error") or {}
            if error.get("code") in ("insufficient_quota", "billing_hard_limit_reached",
                                     "invalid_api_key", "authentication_error"):
                raise ProviderUnavailable(f"Editor blocked: {error['code']}; missing usage reservation retained")
            raise EditorError("Terminal response has no usage receipt; reservation retained until billing is known")
        cost = astra_cost(usage)
        self.budget.settle(rid, cost, usage=usage, seconds=max(0, time.time() - pending["started_at"]),
                           metadata={"response_id": response["id"], "model": response.get("model", self.model)})
        text_parts, refusals = [], []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text_parts.append(part["text"])
                elif part.get("type") == "refusal":
                    refusals.append(part.get("refusal", "Provider refused the output"))
        record = {"response_id": response["id"], "request_id": pending["client_request_id"],
                  "input_sha256": fingerprint, "request_key": pending["key"], "usage": usage,
                  "cost_usd": str(cost), "status": response.get("status"),
                  "text": "".join(text_parts), "refusals": refusals,
                  "error": response.get("error"), "incomplete_details": response.get("incomplete_details"),
                  "effort": state["effort"], "conversation_id": state["conversation_id"],
                  "response_file": f"response_{rid}.json"}
        if state.get("blocked"):
            record["runaway"] = state["blocked"]["diagnostic"]
        receipt = f"receipt_{rid}.json"
        atomic_json(self.root / receipt, record)
        state["completed"][pending["key"]] = receipt
        state["previous_response_id"] = response["id"]
        state["context_tokens"] = int(usage["input_tokens"]) + int(usage["output_tokens"])
        state["pending"] = None
        atomic_json(self.path, state)
        self.commit()
        self.last_response = record
        if state.get("blocked"):
            self._raise_runaway(state)
        return self._deliver(record)

    @staticmethod
    def _deliver(record):
        if record["refusals"]:
            raise EditorRefused("Provider returned a refusal; its usage and response are recorded")
        if record["status"] != "completed":
            detail = record.get("incomplete_details") or record.get("error") or {}
            if detail.get("reason") == "content_filter" or detail.get("code") in ("content_filter", "content_policy_violation"):
                raise EditorRefused("Provider filtered the response; its usage is recorded")
            if record["status"] == "incomplete" and detail.get("reason") == "max_output_tokens":
                raise EditorIncomplete(detail["reason"], record["response_id"])
            raise EditorError(f"Editor response status {record['status']}; inspect the saved response before continuing")
        if not record["text"]:
            raise EditorError("Editor returned no text; response and usage are recorded")
        return record["text"]
