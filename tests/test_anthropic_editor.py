"""Offline Messages tests; the optional real-SDK test uses in-process HTTP only."""

import copy
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from openwsp.budget import BudgetExceeded, BudgetLedger, BudgetOverrun
from openwsp.editors.anthropic import AnthropicEditor, AnthropicTokenRates, anthropic_cost
from openwsp.editors.base import (
    EditorError,
    EditorIncomplete,
    EditorRefused,
    ProviderUnavailable,
    UncertainRequest,
)

RATES = AnthropicTokenRates(
    input_per_million="3",
    output_per_million="15",
    cache_read_per_million="0.3",
    cache_creation_5m_per_million="3.75",
    cache_creation_1h_per_million="6",
)


def response(identifier="msg_1", *, stop="end_turn", content=None, usage=None):
    return {
        "id": identifier,
        "type": "message",
        "role": "assistant",
        "model": "caller-model",
        "content": content if content is not None else [{"type": "text", "text": "Complete answer"}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 100,
            "output_tokens": 30,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


class Failure(Exception):
    def __init__(self, status, kind="invalid_request_error", message="REDACTED-ERROR-DATA"):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"type": kind, "message": message}}
        self.request_id = "req_provider"


class FakeMessages:
    def __init__(self, responses=None, before_create=None):
        self.responses = list(responses or [response()])
        self.calls = []
        self.before_create = before_create

    def create(self, **kwargs):
        if self.before_create:
            self.before_create()
        self.calls.append(copy.deepcopy(kwargs))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)


def editor(tmp_path, fake, **kwargs):
    settings = {
        "model": "caller-model",
        "rates": RATES,
        "max_input_tokens": 200_000,
        "max_output_tokens": 4096,
        **kwargs,
    }
    budget = settings.pop("budget", None) or BudgetLedger(tmp_path / "budget.json")
    return AnthropicEditor(tmp_path / "editor", budget, client=SimpleNamespace(messages=fake), **settings)


def test_full_signed_history_and_usage_categories_survive_restart(tmp_path):
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "opaque+signed=="},
        {"type": "redacted_thinking", "data": "encrypted=="},
        {"type": "text", "text": "First ", "citations": None},
        {"type": "text", "text": "answer"},
    ]
    usage = {
        "input_tokens": 100,
        "output_tokens": 30,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": 20,
        "cache_creation": {"ephemeral_5m_input_tokens": 12, "ephemeral_1h_input_tokens": 8},
        "output_tokens_details": {"reasoning_tokens": 25},
        "service_tier": "standard",
        "speed": "standard",
    }
    fake = FakeMessages([response(content=blocks, usage=usage), response("msg_2")])
    config = {"thinking": {"type": "adaptive"}, "system": "Keep this prefix exactly."}
    instance = editor(tmp_path, fake, **config)
    assert instance.complete("Task", request_key="turn1") == "First answer"
    first = fake.calls[0]
    assert first["tools"] == [] and first["stream"] is False and first["service_tier"] == "standard_only"
    assert first["thinking"] == config["thinking"] and first["system"] == config["system"]
    assert "tool_choice" not in first and "cache_control" not in first
    restarted = editor(tmp_path, fake, **config)
    assert restarted.complete("Tool output", request_key="turn2") == "Complete answer"
    assert fake.calls[1]["messages"] == [
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": blocks},
        {"role": "user", "content": "Tool output"},
    ]
    expected = Decimal("0.000867")  # 100*3 +80*.3 +12*3.75 +8*6 +30*15, divided by 1M
    assert anthropic_cost(usage, RATES) == expected
    receipt = json.loads(next((tmp_path / "editor").glob("receipt_*.json")).read_text())
    first_receipt = instance.last_response
    assert Decimal(first_receipt["cost_usd"]) == expected
    assert first_receipt["usage"] == usage
    assert json.loads((instance.root / first_receipt["response_file"]).read_text())["content"] == blocks
    assert receipt["usage"] is not None


def test_real_sdk_serializes_signed_history_without_native_tools(tmp_path):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    blocks = [
        {"type": "thinking", "thinking": "Checked the supplied statistics.", "signature": "opaque+signed=="},
        {"type": "redacted_thinking", "data": "opaque-encrypted=="},
        {"type": "text", "text": "First answer", "citations": None},
    ]
    replies = [response(content=blocks), response("msg_2")]
    requests = []

    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1/messages"
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=replies[len(requests) - 1])

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as http_client:
        client = anthropic.Anthropic(http_client=http_client, api_key="dummy", max_retries=0)
        instance = AnthropicEditor(
            tmp_path / "editor",
            BudgetLedger(tmp_path / "budget.json"),
            model="caller-model",
            rates=RATES,
            max_input_tokens=200_000,
            max_output_tokens=4096,
            thinking={"type": "adaptive"},
            system="Keep this prefix exactly.",
            client=client,
        )
        assert instance.complete("Task", request_key="one") == "First answer"
        assert instance.complete("Next observation", request_key="two") == "Complete answer"

    assert len(requests) == 2
    assert requests[0]["messages"] == [{"role": "user", "content": "Task"}]
    assert requests[1]["messages"] == [
        {"role": "user", "content": "Task"},
        {"role": "assistant", "content": blocks},
        {"role": "user", "content": "Next observation"},
    ]
    for request in requests:
        assert request["tools"] == [] and "tool_choice" not in request
        assert request["thinking"] == {"type": "adaptive"}
        assert request["system"] == "Keep this prefix exactly."
        assert request["service_tier"] == "standard_only" and request["stream"] is False
    state = json.loads(instance.path.read_text())
    assert state["messages"][1] == {"role": "assistant", "content": blocks}
    assert [event["event"] for event in instance.budget.snapshot()["events"]] == [
        "reserved",
        "settled",
        "reserved",
        "settled",
    ]


def test_real_sdk_fable_profile_sends_documented_effort_and_persists_it(tmp_path):
    from dataclasses import replace

    from openwsp.editors.profiles import FABLE_5_1_PROFILE, ThinkingConfig, create_editor

    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "claude-fable-5-1"
        assert body["output_config"] == {"effort": "max"}
        assert body["thinking"] == {"type": "adaptive"}
        assert body["tools"] == []
        return httpx2.Response(200, json={**response(), "model": "claude-fable-5-1"})

    profile = replace(FABLE_5_1_PROFILE, thinking=ThinkingConfig(type="adaptive"), output_effort="max")
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as http_client:
        client = anthropic.Anthropic(http_client=http_client, api_key="dummy", max_retries=0)
        options = {
            "state_dir": tmp_path / "editor",
            "budget": BudgetLedger(tmp_path / "budget.json"),
            "rates": RATES,
            "max_input_tokens": 200000,
            "max_output_tokens": 4096,
            "client": client,
        }
        instance = create_editor(profile, **options)
        assert instance.complete("Task", request_key="one") == "Complete answer"
        saved = json.loads(next(instance.root.glob("request_*.json")).read_text())
        assert saved["output_config"] == {"effort": "max"}
        assert instance.last_response["settings"]["output_config"] == {"effort": "max"}
        changed = create_editor(replace(profile, output_effort="high"), **options)
        with pytest.raises(EditorError, match="Settings changed"):
            changed.complete("Next", request_key="two")
        assert len(requests) == 1


def test_omitted_output_effort_preserves_existing_settings_and_request_shape(tmp_path):
    fake = FakeMessages()
    instance = editor(tmp_path, fake, output_effort=None)
    assert "output_config" not in instance.settings
    instance.complete("Task", request_key="one")
    assert "output_config" not in fake.calls[0]
    assert editor(tmp_path, fake).complete("Task", request_key="one") == "Complete answer"
    assert len(fake.calls) == 1


def test_reservation_and_request_are_committed_before_create(tmp_path):
    checkpoints = []
    fake = FakeMessages()
    instance = editor(tmp_path, fake, commit=lambda: checkpoints.append("commit"))

    def check():
        state = json.loads(instance.path.read_text())
        rid = state["pending"]["reservation_id"]
        row = instance.budget.snapshot()["reservations"][rid]
        assert row["status"] == "reserved"
        assert Decimal(row["upper_bound_usd"]) == RATES.upper_bound(200_000, 4096)
        assert (instance.root / f"request_{rid}.json").exists()
        assert checkpoints

    fake.before_create = check
    instance.complete("Task", request_key="one")


def test_completed_key_is_idempotent_and_handoff_keeps_receipts(tmp_path):
    fake = FakeMessages([response(), response("msg_2")])
    instance = editor(tmp_path, fake)
    assert instance.complete("Task", request_key="one") == "Complete answer"
    ledger = instance.budget.snapshot()
    restarted = editor(tmp_path, fake)
    assert restarted.complete("Task", request_key="one") == "Complete answer"
    assert restarted.budget.snapshot() == ledger
    assert len(fake.calls) == 1
    with pytest.raises(EditorError, match="different input"):
        restarted.complete("Changed", request_key="one")
    changed = editor(tmp_path, fake, model="other-model", system="Different prefix")
    with pytest.raises(EditorError, match="Settings changed"):
        changed.complete("Next", request_key="two")
    changed.new_conversation()
    changed.complete("Next", request_key="two")
    assert fake.calls[1]["messages"] == [{"role": "user", "content": "Next"}]
    assert fake.calls[1]["model"] == "other-model"
    assert changed.complete("Task", request_key="one") == "Complete answer"
    assert len(fake.calls) == 2


@pytest.mark.parametrize("failure", [TimeoutError("sensitive detail"), Failure(500), Failure(529)])
def test_uncertain_create_never_duplicates_and_can_resolve_verified_receipt(tmp_path, failure):
    fake = FakeMessages([failure])
    instance = editor(tmp_path, fake)
    with pytest.raises(UncertainRequest):
        instance.complete("Task", request_key="one")
    restarted = editor(tmp_path, fake)
    with pytest.raises(UncertainRequest):
        restarted.complete("Task", request_key="one")
    with pytest.raises(EditorError, match="unresolved"):
        restarted.new_conversation()
    with pytest.raises(EditorError, match="currently pending"):
        restarted.resolve_pending(response(), request_key="wrong")
    assert restarted.budget.snapshot()["summary"]["editor"]["reserved_usd"] > 0
    assert restarted.resolve_pending(response(), request_key="one") == "Complete answer"
    assert restarted.complete("Task", request_key="one") == "Complete answer"
    assert len(fake.calls) == 1
    assert restarted.budget.snapshot()["summary"]["editor"]["reserved_usd"] == 0
    assert "sensitive detail" not in restarted.path.read_text()


@pytest.mark.parametrize(
    "failure",
    [Failure(401), Failure(403), Failure(429), Failure(400, message="Your credit balance is too low")],
)
def test_auth_and_quota_rejections_are_durable_and_never_retried(tmp_path, failure):
    fake = FakeMessages([failure])
    instance = editor(tmp_path, fake)
    for _ in range(2):
        with pytest.raises(ProviderUnavailable):
            instance.complete("Task", request_key="one")
    assert len(fake.calls) == 1
    assert instance.budget.snapshot()["summary"]["editor"]["reserved_usd"] == 0
    assert instance.budget.snapshot()["summary"]["editor"]["spent_usd"] == 0
    assert json.loads(instance.path.read_text())["messages"] == []


@pytest.mark.parametrize(
    "reason,exception",
    [
        ("refusal", EditorRefused),
        ("max_tokens", EditorIncomplete),
        ("model_context_window_exceeded", EditorIncomplete),
        ("pause_turn", EditorIncomplete),
        ("tool_use", EditorError),
        (None, EditorError),
    ],
)
def test_noncomplete_outputs_are_settled_but_never_delivered(tmp_path, reason, exception):
    partial = response(stop=reason, content=[{"type": "text", "text": "UNFINISHED-CODE"}])
    fake = FakeMessages([partial])
    instance = editor(tmp_path, fake)
    for _ in range(2):
        with pytest.raises(exception) as raised:
            instance.complete("Task", request_key="one")
        assert "UNFINISHED-CODE" not in str(raised.value)
    rows = instance.budget.snapshot()
    assert [event["event"] for event in rows["events"]] == ["reserved", "settled"]
    assert len(fake.calls) == 1
    assert json.loads(instance.path.read_text())["messages"][-1]["content"] == partial["content"]


def test_unrequested_tool_block_is_not_delivered(tmp_path):
    fake = FakeMessages(
        [response(content=[{"type": "tool_use", "id": "tool1", "name": "exec", "input": {}}])]
    )
    with pytest.raises(EditorError, match="not a complete text"):
        editor(tmp_path, fake).complete("Task")
    assert fake.calls[0]["tools"] == []


def test_missing_cache_ttl_retains_reservation_until_explicit_billing_resolution(tmp_path):
    raw = response(usage={"input_tokens": 10, "output_tokens": 2, "cache_creation_input_tokens": 20})
    fake = FakeMessages([raw])
    instance = editor(tmp_path, fake)
    for _ in range(2):
        with pytest.raises(EditorError, match="TTL is missing"):
            instance.complete("Task", request_key="one")
    assert len(fake.calls) == 1
    assert instance.budget.snapshot()["summary"]["editor"]["reserved_usd"] > 0
    wrong = copy.deepcopy(raw)
    wrong["id"] = "wrong-id"
    with pytest.raises(EditorError, match="content or identity"):
        instance.resolve_pending(wrong, request_key="one")
    resolved = copy.deepcopy(raw)
    resolved["usage"]["cache_creation"] = {"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 15}
    assert instance.resolve_pending(resolved, request_key="one") == "Complete answer"
    assert len(list(instance.root.glob("unresolved_*.json"))) == 1
    assert instance.budget.snapshot()["summary"]["editor"]["spent_usd"] == float(
        anthropic_cost(resolved["usage"], RATES)
    )


def test_equal_creation_rates_make_unsplit_usage_unambiguous():
    rates = AnthropicTokenRates(**{**RATES.record(), "cache_creation_1h_per_million": "3.75"})
    usage = {"input_tokens": 10, "output_tokens": 2, "cache_creation_input_tokens": 20}
    assert anthropic_cost(usage, rates) == Decimal("0.000135")


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_tokens", True),
        ("output_tokens", -1),
        ("cache_read_input_tokens", False),
        ("cache_creation_input_tokens", "1"),
    ],
)
def test_invalid_usage_is_not_guessed(field, value):
    usage = {"input_tokens": 10, "output_tokens": 2, field: value}
    with pytest.raises(EditorError, match="Invalid usage"):
        anthropic_cost(usage, RATES)


@pytest.mark.parametrize("crash_point", ["before_settlement", "after_settlement"])
def test_crash_with_saved_response_recovers_without_post_or_duplicate_charge(
    tmp_path, monkeypatch, crash_point
):
    fake = FakeMessages()
    instance = editor(tmp_path, fake)

    def crash(*args, **kwargs):
        raise RuntimeError("simulated process loss")

    monkeypatch.setattr(
        instance, "_finish" if crash_point == "before_settlement" else "_complete_record", crash
    )
    with pytest.raises(RuntimeError, match="simulated"):
        instance.complete("Task", request_key="one")
    resumed = editor(tmp_path, fake)
    assert resumed.complete("Task", request_key="one") == "Complete answer"
    assert len(fake.calls) == 1
    assert [event["event"] for event in resumed.budget.snapshot()["events"]] == ["reserved", "settled"]
    assert len(json.loads(resumed.path.read_text())["messages"]) == 2


def test_rejection_crash_after_release_recovers_without_post(tmp_path, monkeypatch):
    fake = FakeMessages([Failure(401)])
    instance = editor(tmp_path, fake)
    monkeypatch.setattr(
        instance, "_complete_record", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("crash"))
    )
    with pytest.raises(RuntimeError, match="crash"):
        instance.complete("Task", request_key="one")
    resumed = editor(tmp_path, fake)
    with pytest.raises(ProviderUnavailable):
        resumed.complete("Task", request_key="one")
    assert len(fake.calls) == 1
    assert resumed.budget.snapshot()["summary"]["editor"]["reserved_usd"] == 0


def test_zero_budget_rejects_before_post(tmp_path):
    fake = FakeMessages()
    instance = editor(tmp_path, fake, budget=BudgetLedger(tmp_path / "zero.json", editor_limit=0))
    with pytest.raises(BudgetExceeded):
        instance.complete("Task")
    assert not fake.calls


@pytest.mark.parametrize("input_tokens,halts_ledger", [(101, False), (100_000, True)])
def test_over_bound_usage_is_saved_and_never_delivered(tmp_path, input_tokens, halts_ledger):
    fake = FakeMessages([response(usage={"input_tokens": input_tokens, "output_tokens": 1})])
    instance = editor(tmp_path, fake, max_input_tokens=100, max_output_tokens=100)
    for _ in range(2):
        with pytest.raises(BudgetOverrun):
            instance.complete("Task", request_key="one")
    ledger = instance.budget.snapshot()
    assert ledger["halted"] is halts_ledger
    expected_cost = anthropic_cost({"input_tokens": input_tokens, "output_tokens": 1}, RATES)
    assert ledger["summary"]["editor"]["spent_usd"] == float(expected_cost)
    assert json.loads(instance.path.read_text())["blocked"] == "caller_bounds_exceeded"
    with pytest.raises(BudgetOverrun):
        instance.complete("Next", request_key="two")
    assert len(fake.calls) == 1


def test_injected_sdk_options_disable_automatic_retries(tmp_path):
    fake = FakeMessages()
    options = []
    instance = editor(tmp_path, fake)
    instance.client = SimpleNamespace(
        with_options=lambda **kw: options.append(kw) or SimpleNamespace(messages=fake)
    )
    instance.complete("Task")
    assert options == [{"max_retries": 0, "timeout": 900}]


@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "enabled", "budget_tokens": 1023},
        {"type": "enabled", "budget_tokens": 4096},
        {"type": "adaptive", "unknown": 1},
        {"type": "unknown"},
    ],
)
def test_invalid_thinking_rejected_before_any_request(tmp_path, thinking):
    with pytest.raises(ValueError, match="thinking supports"):
        editor(tmp_path, FakeMessages(), thinking=thinking)


def test_missing_usage_retains_raw_response_and_resolves_without_post(tmp_path):
    raw = response()
    raw["usage"] = None
    fake = FakeMessages([raw])
    instance = editor(tmp_path, fake)
    for _ in range(2):
        with pytest.raises(EditorError, match="no usage"):
            instance.complete("Task", request_key="one")
    assert len(fake.calls) == 1
    assert instance.budget.snapshot()["summary"]["editor"]["spent_usd"] == 0
    assert instance.budget.snapshot()["summary"]["editor"]["reserved_usd"] > 0
    assert instance.resolve_pending(response(), request_key="one") == "Complete answer"
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "blocks",
    [
        [{"type": "thinking", "thinking": "Missing signature"}, {"type": "text", "text": "Not deliverable"}],
        [{"type": "text", "text": None}],
        ["invalid block"],
    ],
)
def test_malformed_content_blocks_do_not_become_next_provider_input(tmp_path, blocks):
    fake = FakeMessages([response(content=blocks)])
    instance = editor(tmp_path, fake)
    with pytest.raises(EditorError, match="not a complete text"):
        instance.complete("Task", request_key="one")
    with pytest.raises(EditorError, match="cannot be continued"):
        instance.complete("Next", request_key="two")
    assert len(fake.calls) == 1
    assert json.loads(instance.path.read_text())["messages"][-1]["content"] == blocks
    assert instance.budget.snapshot()["summary"]["editor"]["spent_usd"] > 0


def test_manual_thinking_settings_are_explicit_and_preserved(tmp_path):
    fake = FakeMessages()
    instance = editor(tmp_path, fake, thinking={"type": "enabled", "budget_tokens": 2048})
    assert instance.complete("Task") == "Complete answer"
    assert fake.calls[0]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in fake.calls[0] and "top_p" not in fake.calls[0]


def test_caller_supplied_native_content_blocks_are_rejected(tmp_path):
    fake = FakeMessages()
    with pytest.raises(ValueError, match="plain-text"):
        editor(tmp_path, fake).complete([{"role": "user", "content": [{"type": "tool_result"}]}])
    assert not fake.calls
