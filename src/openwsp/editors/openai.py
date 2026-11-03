"""Resumable, tool-free Astra Responses adapter.

Requests use only the new message and previous_response_id. The provider keeps
reasoning items between turns; a handoff deliberately creates a new chain. The
driver should pass a stable turn key, save its result, then move to the next key.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..budget import BudgetLedger, astra_cost, astra_upper_bound, atomic_json
from .base import EditorError, EditorRefused, ProviderUnavailable, UncertainRequest

INSTRUCTIONS = "Answer the user's request."
EFFORTS = ("max", "xhigh", "high", "medium", "low")


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
                 effort="max", max_output_tokens=32768, poll_interval=5,
                 max_wait_seconds=3600, on_poll=None, sleep=time.sleep, commit=lambda: None):
        if model != "gpt-6-astra":
            raise ValueError("This adapter's verified billing schedule is for gpt-6-astra only")
        if effort not in EFFORTS or not 1 <= max_output_tokens <= 128_000:
            raise ValueError("Invalid effort or output token ceiling")
        if not 0 <= poll_interval <= 60 or max_wait_seconds <= 0:
            raise ValueError("Invalid polling interval or wait deadline")
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "editor_state.json"
        self.budget = budget
        self.client = client
        self.model, self.effort = model, effort
        self.max_output_tokens = max_output_tokens
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
            if state["pending"]:
                raise EditorError("Cannot hand off while an editor request is unresolved")
            new = self._new_state()
            new["handoffs"] = state["handoffs"] + 1
            atomic_json(self.path, new)
            self.commit()

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
            if key in state["completed"]:
                record = json.loads((self.root / state["completed"][key]).read_text())
                if record["input_sha256"] != fingerprint:
                    raise EditorError("A completed request key was reused with different input")
                self.last_response = record
                return self._deliver(record)
            pending = state["pending"]
            if pending and (pending["key"] != key or pending["input_sha256"] != fingerprint):
                raise EditorError("Finish the persisted pending request before submitting another turn")
            if pending and not pending.get("response_id"):
                raise UncertainRequest("A prior create has no response ID; reservation retained, inspect its request ID")
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
                           "started_at": time.time()}
                state["pending"] = pending
                atomic_json(self.path, state)  # durable BEFORE the possibly billed POST
                request = {
                    "model": self.model, "instructions": INSTRUCTIONS, "input": inputs,
                    "reasoning": {"effort": state["effort"]}, "background": True, "store": True,
                    "max_output_tokens": self.max_output_tokens,
                    "prompt_cache_key": state["conversation_id"],
                    "prompt_cache_options": {"ttl": "30m"}, "tools": [],
                    "truncation": "disabled", "service_tier": "default",
                    "metadata": {"openwsp_request": pending["client_request_id"]},
                    "extra_headers": {"X-Client-Request-Id": pending["client_request_id"]},
                }
                if state["previous_response_id"]:
                    request["previous_response_id"] = state["previous_response_id"]
                request_path = self.root / f"request_{rid}.json"
                atomic_json(request_path, request)
                self.commit()  # Volume ledger and request intent must precede billed POST
                response = self._create(request, state)
                pending["response_id"] = response["id"]
                pending["status"] = response.get("status")
                atomic_json(self.root / f"response_{rid}.json", response)
                atomic_json(self.path, state)
                self.commit()  # durably retain response ID before polling/restart
            else:
                response = self._retrieve(pending["response_id"])
            deadline = time.monotonic() + self.max_wait_seconds
            while response.get("status") in ("queued", "in_progress"):
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
                return _dict(self._get_client().responses.create(**request))
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

    def _retrieve(self, response_id):
        for attempt in range(6):
            try:
                return _dict(self._get_client().responses.retrieve(response_id))
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

    def _finish(self, response, state, fingerprint):
        pending = state["pending"]
        rid = pending["reservation_id"]
        atomic_json(self.root / f"response_{rid}.json", response)
        usage = response.get("usage")
        if usage is None:
            pending["status"] = response.get("status")
            atomic_json(self.path, state)
            self.commit()
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
        receipt = f"receipt_{rid}.json"
        atomic_json(self.root / receipt, record)
        state["completed"][pending["key"]] = receipt
        state["previous_response_id"] = response["id"]
        state["context_tokens"] = int(usage["input_tokens"]) + int(usage["output_tokens"])
        state["pending"] = None
        atomic_json(self.path, state)
        self.commit()
        self.last_response = record
        return self._deliver(record)

    @staticmethod
    def _deliver(record):
        if record["refusals"]:
            raise EditorRefused("Provider returned a refusal; its usage and response are recorded")
        if record["status"] != "completed":
            detail = record.get("incomplete_details") or record.get("error") or {}
            if detail.get("reason") == "content_filter" or detail.get("code") in ("content_filter", "content_policy_violation"):
                raise EditorRefused("Provider filtered the response; its usage is recorded")
            raise EditorError(f"Editor response status {record['status']}; inspect the saved response before continuing")
        if not record["text"]:
            raise EditorError("Editor returned no text; response and usage are recorded")
        return record["text"]
