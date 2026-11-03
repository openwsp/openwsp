import json
from types import SimpleNamespace

import pytest

from openwsp.budget import BudgetLedger
from openwsp.editors import AstraEditor, EditorError, EditorRefused, ProviderUnavailable, UncertainRequest


def response(rid="resp_1", text="A hypothesis.\n```python\n1 + 1\n```", status="completed", refusal=False):
    return {"id": rid, "status": status, "model": "gpt-6-astra",
            "usage": {"input_tokens": 100, "output_tokens": 30,
                      "input_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 20},
                      "output_tokens_details": {"reasoning_tokens": 20}},
            "output": [{"type": "message", "content": [
                {"type": "refusal", "refusal": "Refused"} if refusal else {"type": "output_text", "text": text}
            ]}]}


class Failure(Exception):
    def __init__(self, status, code=None, param=None):
        self.status_code = status
        self.body = {"code": code, "param": param}


class FakeResponses:
    def __init__(self, creates=None, retrieves=None):
        self.creates = list(creates or [response()])
        self.retrieves = list(retrieves or [])
        self.calls = []
        self.gets = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.creates.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def retrieve(self, response_id):
        self.gets.append(response_id)
        item = self.retrieves.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def editor(tmp_path, fake):
    budget = BudgetLedger(tmp_path / "budget.json")
    return AstraEditor(tmp_path / "editor", budget, client=SimpleNamespace(responses=fake),
                       poll_interval=0, sleep=lambda _: None)


def test_exact_editor_contract_and_turn_reasoning_chain(tmp_path):
    fake = FakeResponses([response(), response("resp_2")])
    e = editor(tmp_path, fake)
    e.complete("task", request_key="t001")
    first = fake.calls[0]
    assert first["instructions"] == "Answer the user's request."
    assert first["reasoning"] == {"effort": "max"}
    assert first["tools"] == []
    assert first["prompt_cache_options"] == {"ttl": "30m"}
    assert "temperature" not in first and "top_p" not in first and "previous_response_id" not in first
    e.complete("tool output", request_key="t002")
    second = fake.calls[1]
    assert second["previous_response_id"] == "resp_1"
    assert second["prompt_cache_key"] == first["prompt_cache_key"]
    assert second["input"] == [{"role": "user", "content": "tool output"}]
    assert len(e.budget.snapshot()["reservations"]) == 2


def test_handoff_uses_fresh_cache_and_response_chain(tmp_path):
    fake = FakeResponses([response(), response("resp_2")])
    e = editor(tmp_path, fake)
    e.complete("task", request_key="t001")
    e.new_conversation()
    e.complete("task and notes", request_key="t041")
    assert fake.calls[0]["prompt_cache_key"] != fake.calls[1]["prompt_cache_key"]
    assert "previous_response_id" not in fake.calls[1]


def test_repeated_stable_turn_returns_saved_output_without_charge(tmp_path):
    fake = FakeResponses()
    e = editor(tmp_path, fake)
    answer = e.complete("task", request_key="t001")
    restarted = editor(tmp_path, fake)
    assert restarted.complete("task", request_key="t001") == answer
    assert len(fake.calls) == 1
    with pytest.raises(EditorError, match="different input"):
        restarted.complete("changed", request_key="t001")


def test_background_retrieval_retries_without_recreating(tmp_path):
    fake = FakeResponses([response(status="queued")], [Failure(503), response(status="in_progress"), response()])
    e = editor(tmp_path, fake)
    assert e.complete("task")
    assert len(fake.calls) == 1 and len(fake.gets) == 3
    assert e.budget.snapshot()["summary"]["editor"]["reserved_usd"] == 0


def test_crash_pending_response_resumes_get_only(tmp_path):
    fake = FakeResponses([response(status="queued")], [Failure(401)])
    e = editor(tmp_path, fake)
    with pytest.raises(ProviderUnavailable):
        e.complete("task", request_key="t001")
    fake.retrieves = [response()]
    assert editor(tmp_path, fake).complete("task", request_key="t001")
    assert len(fake.calls) == 1


def test_ambiguous_post_is_not_duplicated_and_keeps_reservation(tmp_path):
    fake = FakeResponses([TimeoutError("uncertain")])
    e = editor(tmp_path, fake)
    with pytest.raises(UncertainRequest):
        e.complete("task", request_key="t001")
    with pytest.raises(UncertainRequest):
        editor(tmp_path, fake).complete("task", request_key="t001")
    assert len(fake.calls) == 1
    assert e.budget.snapshot()["summary"]["editor"]["reserved_usd"] > 0


@pytest.mark.parametrize("failure", [Failure(401), Failure(429, "insufficient_quota")])
def test_auth_and_quota_stop_without_retry(tmp_path, failure):
    fake = FakeResponses([failure])
    e = editor(tmp_path, fake)
    with pytest.raises(ProviderUnavailable):
        e.complete("task")
    assert len(fake.calls) == 1
    assert e.budget.snapshot()["summary"]["editor"]["reserved_usd"] == 0


def test_unsupported_effort_uses_highest_accepted(tmp_path):
    fake = FakeResponses([Failure(400, "unsupported_value", "reasoning.effort"), response()])
    e = editor(tmp_path, fake)
    assert e.complete("task")
    assert len(fake.calls) == 2
    assert e.last_response["effort"] == "xhigh"


def test_refusal_is_recorded_and_charged(tmp_path):
    fake = FakeResponses([response(refusal=True)])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRefused):
        e.complete("task")
    assert e.last_response["refusals"] == ["Refused"]
    assert e.budget.snapshot()["summary"]["editor"]["spent_usd"] > 0


def test_missing_usage_does_not_guess_or_release_cost(tmp_path):
    missing = response()
    missing["usage"] = None
    e = editor(tmp_path, FakeResponses([missing]))
    with pytest.raises(EditorError, match="no usage"):
        e.complete("task")
    assert e.budget.snapshot()["summary"]["editor"]["reserved_usd"] > 0
    state = json.loads(e.path.read_text())
    assert state["pending"]["response_id"] == "resp_1"


def test_budget_rejection_happens_before_network_call(tmp_path):
    fake = FakeResponses()
    budget = BudgetLedger(tmp_path / "budget.json", editor_limit=0)
    e = AstraEditor(tmp_path / "editor", budget, client=SimpleNamespace(responses=fake))
    from openwsp.budget import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        e.complete("task")
    assert not fake.calls


def test_editor_commits_intent_before_post_and_id_before_poll(tmp_path):
    fake = FakeResponses([response(status="queued")], [response()])
    snapshots = []
    e = editor(tmp_path, fake)
    e.commit = lambda: snapshots.append(json.loads(e.path.read_text()))
    original_create, original_retrieve = fake.create, fake.retrieve

    def create(**kwargs):
        assert snapshots and snapshots[-1]["pending"]["response_id"] is None
        return original_create(**kwargs)

    def retrieve(response_id):
        assert snapshots[-1]["pending"]["response_id"] == response_id
        return original_retrieve(response_id)

    fake.create, fake.retrieve = create, retrieve
    e.complete("task", request_key="t001")
    assert snapshots[-1]["pending"] is None and "t001" in snapshots[-1]["completed"]
