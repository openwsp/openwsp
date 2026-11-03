"""Durable, text-only Anthropic Messages adapter; never used by Experiment 8.

The caller supplies pricing and a verified input/context ceiling. Messages has
no response-retrieval API: uncertain POSTs are never silently repeated.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..budget import BudgetLedger, BudgetOverrun, atomic_json
from .base import EditorError, EditorIncomplete, EditorRefused, ProviderUnavailable, UncertainRequest


@dataclass(frozen=True, kw_only=True)
class AnthropicTokenRates:
    """Caller-verified USD per million tokens, including any applicable tiers."""

    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal
    cache_creation_5m_per_million: Decimal
    cache_creation_1h_per_million: Decimal

    def __post_init__(self):
        for name, raw in asdict(self).items():
            try:
                value = Decimal(str(raw))
            except InvalidOperation as exc:
                raise ValueError("Token rates must be finite nonnegative numbers") from exc
            if not value.is_finite() or value < 0:
                raise ValueError("Token rates must be finite nonnegative numbers")
            object.__setattr__(self, name, value)

    def record(self):
        return {name: str(value) for name, value in asdict(self).items()}

    def upper_bound(self, input_tokens: int, output_tokens: int) -> Decimal:
        input_rate = max(
            self.input_per_million,
            self.cache_read_per_million,
            self.cache_creation_5m_per_million,
            self.cache_creation_1h_per_million,
        )
        return (input_tokens * input_rate + output_tokens * self.output_per_million) / 1_000_000


def _count(value, name):
    if type(value) is not int or value < 0:
        raise EditorError(f"Invalid usage counter: {name}; reservation retained")
    return value


def _priced_usage(usage, rates):
    if not isinstance(usage, dict):
        raise EditorError("Response has no usage receipt; reservation retained")
    ordinary = _count(usage.get("input_tokens"), "input_tokens")
    output = _count(usage.get("output_tokens"), "output_tokens")
    read_value = usage.get("cache_read_input_tokens")
    created_value = usage.get("cache_creation_input_tokens")
    read = _count(0 if read_value is None else read_value, "cache_read_input_tokens")
    created = _count(0 if created_value is None else created_value, "cache_creation_input_tokens")
    breakdown = usage.get("cache_creation")
    if breakdown is None:
        if created and rates.cache_creation_5m_per_million != rates.cache_creation_1h_per_million:
            raise EditorError(
                "Cache creation TTL is missing; explicit billing resolution required; reservation retained"
            )
        write5, write1 = created, 0
        basis = "equal_creation_rates" if created else "no_cache_creation"
    else:
        if not isinstance(breakdown, dict):
            raise EditorError("Invalid cache creation breakdown; reservation retained")
        write5 = _count(breakdown.get("ephemeral_5m_input_tokens"), "ephemeral_5m_input_tokens")
        write1 = _count(breakdown.get("ephemeral_1h_input_tokens"), "ephemeral_1h_input_tokens")
        if write5 + write1 != created:
            raise EditorError("Cache creation totals disagree; reservation retained")
        basis = "reported_creation_ttl"
    if usage.get("service_tier") not in (None, "standard") or usage.get("speed") not in (None, "standard"):
        raise EditorError(
            "Unexpected service tier or speed; explicit billing resolution required; reservation retained"
        )
    tools = usage.get("server_tool_use") or {}
    if not isinstance(tools, dict) or any(value for value in tools.values()):
        raise EditorError(
            "Unexpected server-tool usage; explicit billing resolution required; reservation retained"
        )
    cost = (
        ordinary * rates.input_per_million
        + read * rates.cache_read_per_million
        + write5 * rates.cache_creation_5m_per_million
        + write1 * rates.cache_creation_1h_per_million
        + output * rates.output_per_million
    ) / 1_000_000
    return cost, ordinary + read + created, output, basis


def anthropic_cost(usage: dict, rates: AnthropicTokenRates) -> Decimal:
    """Price disjoint usage categories without adding thinking or cache totals twice."""
    return _priced_usage(usage, rates)[0]


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


class AnthropicEditor:
    def __init__(
        self,
        state_dir,
        budget: BudgetLedger,
        *,
        model: str,
        rates: AnthropicTokenRates,
        max_output_tokens: int,
        max_input_tokens: int,
        client=None,
        thinking=None,
        output_effort: str | None = None,
        system: str | None = None,
        timeout: float = 900,
        commit=lambda: None,
    ):
        if not isinstance(model, str) or not model.strip() or not isinstance(rates, AnthropicTokenRates):
            raise ValueError("Explicit model and AnthropicTokenRates are required")
        if any(type(value) is not int or value <= 0 for value in (max_input_tokens, max_output_tokens)):
            raise ValueError("Input/context ceiling and output limit must be positive integers")
        if system is not None and not isinstance(system, str):
            raise ValueError("system must be text")
        if output_effort not in (None, "low", "medium", "high", "xhigh", "max"):
            raise ValueError("output_effort must be low, medium, high, xhigh, max, or omitted")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if thinking is not None:
            if not isinstance(thinking, dict):
                raise ValueError("thinking must be an explicit documented setting")
            kind = thinking.get("type")
            if kind in ("disabled", "adaptive"):
                valid = set(thinking) == {"type"}
            elif kind == "enabled":
                value = thinking.get("budget_tokens")
                valid = (
                    set(thinking) == {"type", "budget_tokens"}
                    and type(value) is int
                    and 1024 <= value < max_output_tokens
                )
            else:
                valid = False
            if not valid:
                raise ValueError(
                    "thinking supports disabled, adaptive, or enabled with 1024 <= budget_tokens < max_output_tokens"
                )
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "anthropic_state.json"
        self.budget, self.client, self.rates = budget, client, rates
        self.timeout, self.commit, self.last_response = timeout, commit, None
        self.settings = {
            "model": model,
            "rates": rates.record(),
            "max_output_tokens": max_output_tokens,
            "max_input_tokens": max_input_tokens,
            "thinking": copy.deepcopy(thinking),
            "system": system,
        }
        # Omission preserves existing conversation fingerprints. This is the
        # SDK's documented shape, not a promise every model accepts each level.
        if output_effort is not None:
            self.settings["output_config"] = {"effort": output_effort}

    def _get_client(self):
        if self.client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise EditorError(
                    "Install the optional anthropic SDK before making Messages requests"
                ) from exc
            self.client = Anthropic(max_retries=0, timeout=self.timeout)
        if hasattr(self.client, "with_options"):
            return self.client.with_options(max_retries=0, timeout=self.timeout)
        return self.client

    def _new_state(self):
        return {
            "schema_version": 1,
            "adapter": "anthropic_messages",
            "settings": copy.deepcopy(self.settings),
            "conversation_id": uuid.uuid4().hex,
            "messages": [],
            "pending": None,
            "completed": {},
            "handoffs": 0,
            "blocked": None,
        }

    @contextmanager
    def _locked_state(self):
        with (self.root / "anthropic.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else self._new_state()
            if state.get("schema_version") != 1 or state.get("adapter") != "anthropic_messages":
                raise EditorError("Unsupported Anthropic state format")
            yield state

    def _save(self, state):
        atomic_json(self.path, state)
        self.commit()

    def new_conversation(self):
        with self._locked_state() as state:
            if state["pending"]:
                raise EditorError("Cannot hand off while an Anthropic request is unresolved")
            fresh = self._new_state()
            fresh.update(completed=state["completed"], handoffs=state["handoffs"] + 1)
            self._save(fresh)

    @staticmethod
    def _input(messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if not isinstance(messages, list) or not messages:
            raise ValueError("Provide nonempty new text messages")
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message["role"] not in ("user", "assistant")
                or not isinstance(message["content"], str)
            ):
                raise ValueError("Only user/assistant role and plain-text content are accepted")
        if messages[-1]["role"] != "user":
            raise ValueError("New messages must end with a user turn; assistant prefills are unsupported")
        return copy.deepcopy(messages)

    def complete(self, messages, *, request_key=None):
        inputs = self._input(messages)
        fingerprint = _fingerprint(inputs)
        key = str(request_key) if request_key is not None else fingerprint
        if not key:
            raise ValueError("request_key must not be empty")
        with self._locked_state() as state:
            if key in state["completed"]:
                record = json.loads((self.root / state["completed"][key]).read_text())
                if record["input_sha256"] != fingerprint:
                    raise EditorError("A completed request key was reused with different input")
                self.last_response = record
                return self._deliver(record)
            if state["settings"] != self.settings:
                raise EditorError("Settings changed; explicitly start a new conversation before submitting")
            if state["blocked"]:
                if state["blocked"] == "caller_bounds_exceeded":
                    raise BudgetOverrun(
                        "Prior usage exceeded caller bounds; review the receipt before a new conversation"
                    )
                raise EditorError(
                    "Prior response cannot be continued; inspect it and start a new conversation"
                )
            pending = state["pending"]
            if pending:
                if pending["key"] != key or pending["input_sha256"] != fingerprint:
                    raise EditorError("Finish the persisted pending request before submitting another turn")
                raw_path = self.root / pending["response_file"]
                if not raw_path.exists():
                    raise UncertainRequest(
                        "Messages create outcome is uncertain; reservation retained; do not repeat the POST"
                    )
                return self._finish(json.loads(raw_path.read_text()), state)
            if not state["messages"] and inputs[0]["role"] != "user":
                raise ValueError("A conversation must start with a user message")
            client = self._get_client()
            rid = uuid.uuid4().hex
            bound = self.rates.upper_bound(
                self.settings["max_input_tokens"], self.settings["max_output_tokens"]
            )
            self.budget.reserve(
                "editor",
                bound,
                f"anthropic:{key}",
                reservation_id=rid,
                metadata={
                    "provider": "anthropic",
                    "settings": self.settings,
                    "bound_basis": "caller_verified_input_ceiling_and_output_limit",
                    "conversation_id": state["conversation_id"],
                },
            )
            pending = {
                "key": key,
                "input_sha256": fingerprint,
                "inputs": inputs,
                "reservation_id": rid,
                "upper_bound_usd": str(bound),
                "started_at": time.time(),
                "response_file": f"response_{rid}.json",
            }
            state["pending"] = pending
            request = {
                "model": self.settings["model"],
                "max_tokens": self.settings["max_output_tokens"],
                "messages": copy.deepcopy(state["messages"]) + inputs,
                "tools": [],
                "service_tier": "standard_only",
                "stream": False,
            }
            for name in ("system", "thinking"):
                if self.settings[name] is not None:
                    request[name] = copy.deepcopy(self.settings[name])
            if "output_config" in self.settings:
                request["output_config"] = copy.deepcopy(self.settings["output_config"])
            atomic_json(self.root / f"request_{rid}.json", request)
            self._save(state)  # reservation + intent precede the possibly billed POST
            try:
                response = client.messages.create(**request)
            except Exception as error:  # noqa: BLE001 -- an uncertain POST must never be recreated.
                self._create_failed(error, state)
                raise AssertionError("unreachable") from None
            try:
                raw = copy.deepcopy(
                    response if isinstance(response, dict) else response.model_dump(mode="json")
                )
                if not isinstance(raw, dict):
                    raise TypeError("Non-object response")
            except Exception as error:  # noqa: BLE001 -- the request may already have incurred charges.
                pending["error"] = {"class": type(error).__name__}
                self._save(state)
                raise UncertainRequest("Cannot persist Messages response; reservation retained") from None
            atomic_json(self.root / pending["response_file"], raw)
            self._save(state)  # response exists durably before any usage settlement
            return self._finish(raw, state)

    def _create_failed(self, error, state):
        pending = state["pending"]
        body = getattr(error, "body", None)
        body = body.get("error", body) if isinstance(body, dict) else {}
        body = body if isinstance(body, dict) else {}
        status = getattr(error, "status_code", None)
        detail = {
            "class": type(error).__name__,
            "status_code": status,
            "type": body.get("type"),
            "request_id": getattr(error, "request_id", None),
        }
        pending["error"] = detail
        if status in (400, 401, 402, 403, 404, 413, 422, 429):
            quota = any(
                word in str(body.get("message", "")).lower()
                for word in ("credit balance", "insufficient credits", "quota")
            )
            unavailable = status in (401, 402, 403, 429) or quota
            rejection = {"_provider_rejection": detail, "unavailable": unavailable}
            atomic_json(self.root / pending["response_file"], rejection)
            self._save(state)
            return self._finish_rejection(rejection, state)
        self._save(state)
        raise UncertainRequest(
            "Messages create outcome is uncertain; reservation retained; no retry was submitted"
        ) from None

    def resolve_pending(self, response: dict, *, request_key):
        """Record an operator-verified receipt; this method does not authenticate it.

        The operator must establish that the complete response and usage belong
        to this pending request. No provider fetch or duplicate POST is made.
        """
        if not isinstance(response, dict):
            raise TypeError("Provide the complete provider response")
        raw = copy.deepcopy(response)
        if not isinstance(raw.get("id"), str) or not isinstance(raw.get("usage"), dict):
            raise TypeError("Resolution requires a response ID and usage receipt")
        with self._locked_state() as state:
            pending = state["pending"]
            if not pending or pending["key"] != str(request_key):
                raise EditorError("Resolution must identify the currently pending request")
            if state["settings"] != self.settings:
                raise EditorError("Use the original settings to price the pending receipt")
            path = self.root / pending["response_file"]
            if path.exists():
                prior = json.loads(path.read_text())
                if prior.get("id") != raw["id"] or {k: v for k, v in prior.items() if k != "usage"} != {
                    k: v for k, v in raw.items() if k != "usage"
                }:
                    raise EditorError(
                        "Billing resolution cannot replace the saved response content or identity"
                    )
                atomic_json(
                    self.root / f"unresolved_{pending['reservation_id']}_{_fingerprint(prior)[:16]}.json",
                    prior,
                )
            atomic_json(path, raw)
            self._save(state)
            return self._finish(raw, state)

    def _finish(self, raw, state):
        if "_provider_rejection" in raw:
            return self._finish_rejection(raw, state)
        pending = state["pending"]
        cost, input_tokens, output_tokens, basis = _priced_usage(raw.get("usage"), self.rates)
        exceeded = (
            input_tokens > self.settings["max_input_tokens"]
            or output_tokens > self.settings["max_output_tokens"]
            or cost > Decimal(pending["upper_bound_usd"])
        )
        try:
            self.budget.settle(
                pending["reservation_id"],
                cost,
                usage=raw["usage"],
                seconds=max(0, time.time() - pending["started_at"]),
                metadata={
                    "provider": "anthropic",
                    "model": raw.get("model"),
                    "response_id": raw.get("id"),
                    "rates": self.rates.record(),
                    "cache_creation_basis": basis,
                    "caller_bounds_exceeded": exceeded,
                },
            )
        except BudgetOverrun:
            exceeded = True  # settlement is already durable; retain its actual charge
        content = raw.get("content")
        reason = raw.get("stop_reason")
        outcome, text = "error", ""
        if isinstance(content, list) and all(isinstance(block, dict) for block in content):
            safe_types = all(
                block.get("type") == "text"
                and isinstance(block.get("text"), str)
                or block.get("type") == "thinking"
                and isinstance(block.get("thinking"), str)
                and isinstance(block.get("signature"), str)
                or block.get("type") == "redacted_thinking"
                and isinstance(block.get("data"), str)
                for block in content
            )
            parts = [block.get("text") for block in content if block.get("type") == "text"]
            if reason == "refusal":
                outcome = "refused"
            elif reason in ("max_tokens", "model_context_window_exceeded", "pause_turn", "stop_sequence"):
                outcome = "incomplete"
            elif (
                reason == "end_turn" and safe_types and parts and all(isinstance(part, str) for part in parts)
            ):
                text = "".join(parts)
                outcome = "complete" if text else "error"
            if not safe_types:
                state["blocked"] = "unsupported_response_content"
        if (
            raw.get("type") != "message"
            or raw.get("role") != "assistant"
            or not isinstance(raw.get("id"), str)
        ):
            outcome, text = "error", ""
        if outcome == "error" or reason == "pause_turn":
            state["blocked"] = "unsupported_response_continuation"
        if exceeded:
            outcome, text = "overrun", ""
            state["blocked"] = "caller_bounds_exceeded"
        record = {
            "request_key": pending["key"],
            "input_sha256": pending["input_sha256"],
            "response_id": raw.get("id"),
            "response_file": pending["response_file"],
            "usage": raw["usage"],
            "cost_usd": str(cost),
            "cache_creation_basis": basis,
            "stop_reason": reason,
            "outcome": outcome,
            "text": text,
            "conversation_id": state["conversation_id"],
            "settings": self.settings,
        }
        self._complete_record(state, record, append_response=content)
        return self._deliver(record)

    def _finish_rejection(self, rejection, state):
        pending = state["pending"]
        row = self.budget.snapshot()["reservations"][pending["reservation_id"]]
        if row["status"] != "released":
            self.budget.release(
                pending["reservation_id"],
                f"Anthropic rejected Messages create: HTTP {rejection['_provider_rejection']['status_code']}",
            )
        record = {
            "input_sha256": pending["input_sha256"],
            "request_key": pending["key"],
            "outcome": "unavailable" if rejection["unavailable"] else "rejected",
            "error": rejection["_provider_rejection"],
            "response_id": None,
            "response_file": pending["response_file"],
            "cost_usd": "0",
            "usage": None,
            "text": "",
        }
        self._complete_record(state, record, append_response=None)
        return self._deliver(record)

    def _complete_record(self, state, record, append_response):
        pending = state["pending"]
        receipt = f"receipt_{pending['reservation_id']}.json"
        atomic_json(self.root / receipt, record)
        if isinstance(append_response, list):
            state["messages"].extend(copy.deepcopy(pending["inputs"]))
            state["messages"].append({"role": "assistant", "content": copy.deepcopy(append_response)})
        state["completed"][pending["key"]] = receipt
        state["pending"] = None
        self._save(state)
        self.last_response = record

    @staticmethod
    def _deliver(record):
        outcome = record["outcome"]
        if outcome == "complete":
            return record["text"]
        if outcome == "refused":
            raise EditorRefused("Anthropic refused the response; receipt and usage are recorded")
        if outcome == "incomplete":
            raise EditorIncomplete(record["stop_reason"], record.get("response_id"))
        if outcome == "unavailable":
            raise ProviderUnavailable(
                "Anthropic authentication, quota, or rate limit rejected the request; no retry"
            )
        if outcome == "overrun":
            raise BudgetOverrun("Anthropic usage exceeded caller bounds; actual usage and cost are recorded")
        raise EditorError("Anthropic response was not a complete text answer; inspect the saved receipt")
