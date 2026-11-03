import json
from itertools import pairwise
from types import SimpleNamespace

import pytest

from openwsp.budget import BudgetLedger, astra_cost
from openwsp.editors import (
    AstraEditor,
    EditorError,
    EditorIncomplete,
    EditorRefused,
    EditorRunaway,
    ProviderUnavailable,
    UncertainRequest,
)


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
    def __init__(self, creates=None, retrieves=None, cancels=None):
        self.creates = list(creates or [response()])
        self.retrieves = list(retrieves or [])
        self.calls = []
        self.gets = []
        self.get_options = []
        self.cancel_responses = list(cancels or [])
        self.cancel_calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.creates.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def retrieve(self, response_id, **kwargs):
        self.gets.append(response_id)
        self.get_options.append(kwargs)
        item = self.retrieves.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


    def cancel(self, response_id, **kwargs):
        self.cancel_calls.append((response_id, kwargs))
        item = self.cancel_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.events:
            raise StopIteration
        item = self.events.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


def response_event(value, sequence=0):
    kind = "created" if value["status"] == "queued" else value["status"]
    return {"type": f"response.{kind}", "sequence_number": sequence, "response": value}


def text_delta(text, sequence=1):
    return {"type": "response.output_text.delta", "sequence_number": sequence, "delta": text}


RUNAWAY_TEXT = "A short answer.\n```python\n1 + 1\n```\n" + "`\n\n" * 400


def runaway_stream():
    return FakeStream([response_event(response(status="queued")), text_delta(RUNAWAY_TEXT)])


def editor(tmp_path, fake, **kwargs):
    budget = BudgetLedger(tmp_path / "budget.json")
    poll_interval = kwargs.pop("poll_interval", 0)
    return AstraEditor(tmp_path / "editor", budget, client=SimpleNamespace(responses=fake),
                       poll_interval=poll_interval, sleep=lambda _: None, **kwargs)


def test_exact_editor_contract_and_turn_reasoning_chain(tmp_path):
    fake = FakeResponses([response(), response("resp_2")])
    e = editor(tmp_path, fake)
    e.complete("task", request_key="t001")
    first = fake.calls[0]
    assert first["instructions"] == "Answer the user's request."
    assert first["reasoning"] == {"effort": "max"}
    assert first["max_output_tokens"] == 32_768
    assert first["stream"] is True and first["background"] is True
    assert "text" not in first
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


def test_incomplete_output_is_settled_once_and_replayed_without_partial_text(tmp_path):
    partial = response("resp_incomplete", text="UNFINISHED CODE", status="incomplete")
    partial["incomplete_details"] = {"reason": "max_output_tokens"}
    partial["usage"]["output_tokens"] = 32_768
    fake = FakeResponses([partial])
    e = editor(tmp_path, fake)
    # A receipt created with the prior ceiling must remain recoverable after
    # restarting with the new default, without submitting another paid attempt.
    e.max_output_tokens = 128_000
    with pytest.raises(EditorIncomplete) as raised:
        e.complete("task", request_key="t011")
    assert raised.value.reason == "max_output_tokens"
    assert raised.value.response_id == "resp_incomplete"
    assert "UNFINISHED CODE" not in str(raised.value)
    state = json.loads(e.path.read_text())
    assert state["pending"] is None
    assert state["previous_response_id"] == "resp_incomplete"
    receipt = json.loads((e.root / state["completed"]["t011"]).read_text())
    assert receipt["usage"] == partial["usage"]
    assert json.loads((e.root / receipt["response_file"]).read_text()) == partial
    settled = e.budget.snapshot()
    assert settled["summary"]["editor"]["spent_usd"] == float(astra_cost(partial["usage"]))
    assert settled["summary"]["editor"]["reserved_usd"] == 0
    assert [event["event"] for event in settled["events"]] == ["reserved", "settled"]
    restarted = editor(tmp_path, fake)
    for _ in range(2):
        with pytest.raises(EditorIncomplete) as replayed:
            restarted.complete("task", request_key="t011")
        assert replayed.value.response_id == raised.value.response_id
        assert replayed.value.reason == raised.value.reason
    assert restarted.budget.snapshot() == settled
    assert len(fake.calls) == 1 and not fake.gets
    assert json.loads(e.path.read_text()) == state
    with pytest.raises(EditorError, match="different input"):
        restarted.complete("changed task", request_key="t011")


def test_next_caller_message_chains_to_incomplete_response_with_lower_default_ceiling(tmp_path):
    partial = response("resp_incomplete", status="incomplete")
    partial["incomplete_details"] = {"reason": "max_output_tokens"}
    fake = FakeResponses([response("resp_complete"), partial, response("resp_next")])
    e = editor(tmp_path, fake)
    e.max_output_tokens = 128_000
    e.complete("task", request_key="t010")
    with pytest.raises(EditorIncomplete):
        e.complete("first tool output", request_key="t011")
    prior_state = json.loads(e.path.read_text())
    restarted = editor(tmp_path, fake)
    assert restarted.complete("next caller message", request_key="t012")
    assert fake.calls[1]["previous_response_id"] == "resp_complete"
    assert fake.calls[2]["previous_response_id"] == "resp_incomplete"
    assert fake.calls[2]["input"] == [{"role": "user", "content": "next caller message"}]
    assert fake.calls[2]["max_output_tokens"] == 32_768
    assert fake.calls[2]["prompt_cache_key"] == fake.calls[0]["prompt_cache_key"]
    final_state = json.loads(restarted.path.read_text())
    assert final_state["previous_response_id"] == "resp_next"
    assert all(final_state["completed"][key] == receipt for key, receipt in prior_state["completed"].items())
    reservations = restarted.budget.snapshot()["reservations"].values()
    assert len(reservations) == 3 and all(row["status"] == "settled" for row in reservations)
    assert next(row for row in reservations if row["stage"] == "editor:t012")["metadata"]["max_output_tokens"] == 32_768


@pytest.mark.parametrize("status,reason,exception", [
    ("incomplete", "content_filter", EditorRefused),
    ("incomplete", "max_messages", EditorError),
    ("failed", "max_output_tokens", EditorError),
])
def test_only_terminal_token_limit_incomplete_gets_typed_error(tmp_path, status, reason, exception):
    terminal = response(status=status)
    terminal["incomplete_details"] = {"reason": reason}
    e = editor(tmp_path, FakeResponses([terminal]))
    with pytest.raises(exception) as raised:
        e.complete("task")
    assert not isinstance(raised.value, EditorIncomplete)


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_missing_usage_does_not_guess_or_release_cost(tmp_path, status):
    missing = response(status=status)
    if status == "incomplete":
        missing["incomplete_details"] = {"reason": "max_output_tokens"}
    missing["usage"] = None
    e = editor(tmp_path, FakeResponses([missing]))
    with pytest.raises(EditorError, match="no usage") as raised:
        e.complete("task")
    assert not isinstance(raised.value, EditorIncomplete)
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


@pytest.mark.parametrize("verbosity", ["low", "medium", "high"])
def test_explicit_verbosity_only_changes_requested_text_setting(tmp_path, verbosity):
    fake = FakeResponses()
    editor(tmp_path, fake, verbosity=verbosity).complete("task")
    assert fake.calls[0]["text"] == {"verbosity": verbosity}
    assert fake.calls[0]["instructions"] == "Answer the user's request."
    assert fake.calls[0]["reasoning"] == {"effort": "max"}


def test_invalid_verbosity_rejected_before_create(tmp_path):
    fake = FakeResponses()
    with pytest.raises(ValueError, match="Verbosity"):
        editor(tmp_path, fake, verbosity="automatic")
    assert not fake.calls


def test_streamed_completed_answer_has_durable_id_cursor_and_one_charge(tmp_path):
    terminal = response()
    stream = FakeStream([response_event(response(status="queued")),
                         text_delta("A hypothesis."), response_event(terminal, 2)])
    fake = FakeResponses([stream])
    e = editor(tmp_path, fake)
    snapshots = []
    e.commit = lambda: snapshots.append(json.loads(e.path.read_text()))
    assert e.complete("task", request_key="t001") == terminal["output"][0]["content"][0]["text"]
    assert any(s["pending"] and s["pending"]["response_id"] == "resp_1"
               and s["pending"].get("sequence_number") == 0 for s in snapshots)
    assert any(s["pending"] and s["pending"].get("sequence_number") == 2
               and s["pending"]["stream_tail"] == "A hypothesis." for s in snapshots)
    assert snapshots[-1]["pending"] is None
    assert stream.closed and not fake.cancel_calls
    assert len(fake.calls) == 1
    assert e.budget.snapshot()["summary"]["editor"]["spent_usd"] == float(astra_cost(terminal["usage"]))


def test_stream_reconnect_uses_committed_cursor_and_ignores_duplicate_delta(tmp_path):
    first = FakeStream([response_event(response(status="queued")), text_delta("`\n\n" * 200),
                        ConnectionError("disconnected")])
    second = FakeStream([text_delta("`\n\n" * 200), text_delta("`\n\n" * 200, 2)])
    fake = FakeResponses([first], [second], [response(status="cancelled", text=RUNAWAY_TEXT)])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway) as raised:
        e.complete("task", request_key="t001")
    assert fake.get_options == [{"stream": True, "starting_after": 1}]
    assert raised.value.diagnostic["backtick_lines"] == 400
    assert len(fake.calls) == 1 and len(fake.cancel_calls) == 1
    assert first.closed and second.closed
    events = [json.loads(line) for path in e.root.glob("stream_*.jsonl") for line in path.read_text().splitlines()]
    assert [event["sequence_number"] for event in events] == [0, 1, 2]


def test_keepalive_counters_never_advance_reconnect_cursor_or_drop_real_events(tmp_path, monkeypatch):
    monkeypatch.setattr("openwsp.editors.openai.time.monotonic", lambda: 0.0)
    keepalive = {"type": "keepalive", "sequence_number": 1_000_000}
    first = FakeStream([{"type": "keepalive"}, response_event(response(status="queued")),
                        text_delta("first"), keepalive, keepalive, {"type": "keepalive"},
                        ConnectionError("disconnected")])
    second = FakeStream([keepalive, text_delta("second", 2), {"type": "keepalive", "sequence_number": 2},
                         response_event(response(text="firstsecond"), 3)])
    fake = FakeResponses([first], [second])
    e = editor(tmp_path, fake)
    snapshots = []
    e.commit = lambda: snapshots.append(json.loads(e.path.read_text()))
    assert e.complete("task", request_key="t001") == "firstsecond"
    assert fake.get_options == [{"stream": True, "starting_after": 1}]
    assert all(s["pending"]["sequence_number"] in (None, 0, 1, 2, 3) for s in snapshots if s["pending"])
    events = [json.loads(line) for path in e.root.glob("stream_*.jsonl") for line in path.read_text().splitlines()]
    assert [event["sequence_number"] for event in events] == [0, 1, 2, 3]
    assert all(event["type"] != "keepalive" for event in events)
    assert len(fake.calls) == 1 and not fake.cancel_calls
    assert [event["event"] for event in e.budget.snapshot()["events"]] == ["reserved", "settled"]


def test_keepalives_preserve_batched_heartbeat_and_real_cursor(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("openwsp.editors.openai.time.monotonic", lambda: now[0])

    class TimedStream(FakeStream):
        def __next__(self):
            now[0] += 1.0
            return super().__next__()

    stream = TimedStream([response_event(response(status="queued")), text_delta("answer"),
                          *({"type": "keepalive", "sequence_number": i + 10_000} for i in range(13)),
                          response_event(response(text="answer"), 2)])
    e = editor(tmp_path, FakeResponses([stream]), poll_interval=5)
    heartbeats = []
    e.on_poll = lambda meta: heartbeats.append((now[0], meta))
    assert e.complete("task", request_key="t001") == "answer"
    assert [t for t, _ in heartbeats] == [5.0, 10.0, 15.0]
    assert all(meta["sequence_number"] == 1 for _, meta in heartbeats)


def test_process_restart_preserves_stream_tail_before_cancellation(tmp_path, monkeypatch):
    class Crash(BaseException):
        pass

    monkeypatch.setattr("openwsp.editors.openai.STREAM_CHECKPOINT_EVENTS", 1)
    first = FakeStream([response_event(response(status="queued")), text_delta("`\n\n" * 200), Crash()])
    fake = FakeResponses([first])
    e = editor(tmp_path, fake)
    with pytest.raises(Crash):
        e.complete("task", request_key="t001")
    pending = json.loads(e.path.read_text())["pending"]
    assert pending["sequence_number"] == 1 and pending["stream_tail"] == "`\n\n" * 200
    fake.retrieves = [FakeStream([text_delta("`\n\n" * 200, 2)])]
    fake.cancel_responses = [response(status="cancelled")]
    with pytest.raises(EditorRunaway) as raised:
        editor(tmp_path, fake).complete("task", request_key="t001")
    assert not raised.value.usage_pending
    assert fake.get_options == [{"stream": True, "starting_after": 1}]
    assert len(fake.calls) == 1 and len(fake.cancel_calls) == 1


def test_crash_replays_uncommitted_chunks_from_old_cursor_without_double_counting(tmp_path, monkeypatch):
    class Crash(BaseException):
        pass

    monkeypatch.setattr("openwsp.editors.openai.time.monotonic", lambda: 0.0)
    first_chunk = "`\n\n" * 200
    first = FakeStream([response_event(response(status="queued")), text_delta(first_chunk), Crash()])
    fake = FakeResponses([first])
    e = editor(tmp_path, fake)
    with pytest.raises(Crash):
        e.complete("task", request_key="t001")
    durable = json.loads(e.path.read_text())["pending"]
    assert durable["response_id"] == "resp_1"
    assert durable["sequence_number"] == 0 and durable["stream_tail"] == ""
    fake.retrieves = [FakeStream([response_event(response(status="queued")),
                                 text_delta(first_chunk), text_delta(first_chunk, 2)])]
    fake.cancel_responses = [response(status="cancelled")]
    restarted = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway) as raised:
        restarted.complete("task", request_key="t001")
    assert fake.get_options == [{"stream": True, "starting_after": 0}]
    assert raised.value.diagnostic["backtick_lines"] == 400
    assert len(fake.calls) == len(fake.cancel_calls) == 1
    assert [event["event"] for event in restarted.budget.snapshot()["events"]] == ["reserved", "settled"]


def test_stream_batches_commits_by_event_count_instead_of_per_delta(tmp_path, monkeypatch):
    monkeypatch.setattr("openwsp.editors.openai.time.monotonic", lambda: 0.0)
    terminal = response(text="x" * 1024)
    stream = FakeStream([response_event(response(status="queued")),
                         *(text_delta("x", sequence) for sequence in range(1, 1025)),
                         response_event(terminal, 1025)])
    fake = FakeResponses([stream])
    e = editor(tmp_path, fake)
    snapshots, heartbeats = [], []
    e.commit = lambda: snapshots.append(json.loads(e.path.read_text()))
    e.on_poll = heartbeats.append
    assert e.complete("task", request_key="t001") == "x" * 1024
    cursors = [s["pending"]["sequence_number"] for s in snapshots if s["pending"]]
    assert cursors == [None, 0, 256, 512, 768, 1024, 1025]
    assert len(snapshots) == 8 and not heartbeats
    assert all(len(s["pending"]["stream_tail"]) == s["pending"]["sequence_number"]
               for s in snapshots if s["pending"] and s["pending"]["sequence_number"] in (256, 512, 768, 1024))


def test_stream_checkpoint_time_and_heartbeat_cadences_are_independent(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("openwsp.editors.openai.time.monotonic", lambda: now[0])

    class TimedStream(FakeStream):
        def __next__(self):
            now[0] += 0.25
            return super().__next__()

    stream = TimedStream([response_event(response(status="queued")),
                          *(text_delta("x", sequence) for sequence in range(1, 25)),
                          response_event(response(text="x" * 24), 25)])
    e = editor(tmp_path, FakeResponses([stream]), poll_interval=5)
    checkpoints, heartbeats = [], []
    e.commit = lambda: checkpoints.append((now[0], json.loads(e.path.read_text())["pending"]))
    e.on_poll = lambda _: heartbeats.append(now[0])
    assert e.complete("task", request_key="t001") == "x" * 24
    stream_times = [t for t, p in checkpoints if p and p["response_id"]]
    assert stream_times[0] == 0.25  # first response ID is committed immediately
    assert max(b - a for a, b in pairwise(stream_times)) <= 1.0
    assert heartbeats == [5.0]


def test_crash_after_terminal_cursor_recovers_with_nonstream_get(tmp_path):
    class Crash(BaseException):
        pass

    terminal = response()
    stream = FakeStream([response_event(response(status="queued")), response_event(terminal, 1)])
    fake = FakeResponses([stream], [terminal])
    e = editor(tmp_path, fake)

    def crash_after_terminal_event():
        pending = json.loads(e.path.read_text())["pending"]
        if pending and pending.get("status") == "completed" and pending.get("sequence_number") == 1:
            raise Crash()

    e.commit = crash_after_terminal_event
    with pytest.raises(Crash):
        e.complete("task", request_key="t001")
    restarted = editor(tmp_path, fake)
    assert restarted.complete("task", request_key="t001")
    assert fake.get_options == [{}] and len(fake.calls) == 1
    assert [event["event"] for event in restarted.budget.snapshot()["events"]] == ["reserved", "settled"]


def test_streamed_terminal_missing_usage_reconciles_without_replaying_exhausted_stream(tmp_path):
    missing = response()
    missing["usage"] = None
    stream = FakeStream([response_event(response(status="queued")), response_event(missing, 1)])
    fake = FakeResponses([stream], [response()])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorError, match="no usage"):
        e.complete("task", request_key="t001")
    assert e.budget.snapshot()["summary"]["editor"]["reserved_usd"] > 0
    restarted = editor(tmp_path, fake)
    assert restarted.complete("task", request_key="t001")
    assert fake.get_options == [{}] and len(fake.calls) == 1
    assert restarted.budget.snapshot()["summary"]["editor"]["reserved_usd"] == 0


def test_legacy_terminal_runaway_missing_usage_gets_receipt_without_cancel(tmp_path):
    missing = response(status="incomplete", text=RUNAWAY_TEXT)
    missing["usage"] = None
    fake = FakeResponses([missing], [response(status="incomplete", text=RUNAWAY_TEXT)])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway) as pending:
        e.complete("task", request_key="t001")
    assert pending.value.usage_pending
    with pytest.raises(EditorRunaway) as reconciled:
        editor(tmp_path, fake).complete("task", request_key="t001")
    assert not reconciled.value.usage_pending
    assert fake.get_options == [{}] and not fake.cancel_calls


def test_runaway_cancellation_is_durable_and_replay_cannot_launch_another_turn(tmp_path):
    stream = runaway_stream()
    fake = FakeResponses([stream], cancels=[response(status="cancelled", text=RUNAWAY_TEXT)])
    e = editor(tmp_path, fake)
    cancel = fake.cancel

    def checked_cancel(response_id, **kwargs):
        state = json.loads(e.path.read_text())
        assert state["blocked"]["response_id"] == response_id
        assert state["pending"]["cancel_status"] == "intent"
        assert state["pending"]["sequence_number"] == 1
        return cancel(response_id, **kwargs)

    fake.cancel = checked_cancel
    with pytest.raises(EditorRunaway) as raised:
        e.complete("task", request_key="t001")
    assert raised.value.reason == "runaway_backtick_suffix" and not raised.value.usage_pending
    assert "A short answer" not in str(raised.value)
    assert fake.cancel_calls == [("resp_1", {"timeout": 30})]
    settled = e.budget.snapshot()
    restarted = editor(tmp_path, fake)
    for key in ["t001", "t002"]:
        with pytest.raises(EditorRunaway):
            restarted.complete("task", request_key=key)
    with pytest.raises(EditorRunaway):
        restarted.new_conversation()
    assert restarted.budget.snapshot() == settled
    assert len(fake.calls) == 1 and len(fake.cancel_calls) == 1 and not fake.gets
    assert restarted.last_response["runaway"]["kind"] == "runaway_backtick_suffix"


def test_block_preserves_read_only_replay_of_earlier_successful_keys(tmp_path):
    fake = FakeResponses([response("resp_earlier"), response(text=RUNAWAY_TEXT)])
    e = editor(tmp_path, fake)
    earlier = e.complete("first input", request_key="t001")
    with pytest.raises(EditorRunaway):
        e.complete("second input", request_key="t002")
    settled = e.budget.snapshot()
    restarted = editor(tmp_path, fake)
    assert restarted.complete("first input", request_key="t001") == earlier
    with pytest.raises(EditorRunaway):
        restarted.complete("third input", request_key="t003")
    assert len(fake.calls) == 2 and restarted.budget.snapshot() == settled


def test_crash_after_cancel_intent_reissues_same_idempotent_cancel(tmp_path):
    class Crash(BaseException):
        pass

    fake = FakeResponses([runaway_stream()], cancels=[response(status="cancelled")])
    e = editor(tmp_path, fake)

    def crash_at_intent():
        state = json.loads(e.path.read_text())
        if state["pending"] and state["pending"].get("cancel_status") == "intent":
            raise Crash()

    e.commit = crash_at_intent
    with pytest.raises(Crash):
        e.complete("task", request_key="t001")
    assert not fake.cancel_calls
    with pytest.raises(EditorRunaway) as raised:
        editor(tmp_path, fake).complete("task", request_key="t001")
    assert not raised.value.usage_pending
    assert len(fake.calls) == len(fake.cancel_calls) == 1


@pytest.mark.parametrize("cancel_result", [response(status="completed"), Failure(409)])
def test_cancel_race_with_completion_settles_but_never_delivers_text(tmp_path, cancel_result):
    fake = FakeResponses([runaway_stream()], [response()], [cancel_result])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway) as raised:
        e.complete("task", request_key="t001")
    assert not raised.value.usage_pending
    assert e.last_response["status"] == "completed"
    assert e.last_response["runaway"]["kind"] == "runaway_backtick_suffix"
    assert [event["event"] for event in e.budget.snapshot()["events"]] == ["reserved", "settled"]


def test_cancelled_missing_usage_retains_reservation_then_reconciles_get_only(tmp_path):
    missing = response(status="cancelled")
    missing["usage"] = None
    fake = FakeResponses([runaway_stream()], cancels=[missing])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway) as raised:
        e.complete("task", request_key="t001")
    assert raised.value.usage_pending
    before = e.budget.snapshot()
    assert before["summary"]["editor"]["spent_usd"] == 0
    assert before["summary"]["editor"]["reserved_usd"] > 0
    restarted = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway):
        restarted.complete("next input", request_key="t002")
    with pytest.raises(EditorRunaway):
        restarted.new_conversation()
    assert restarted.budget.snapshot() == before
    fake.retrieves = [response(status="cancelled")]
    with pytest.raises(EditorRunaway) as reconciled:
        restarted.complete("task", request_key="t001")
    assert not reconciled.value.usage_pending
    assert len(fake.calls) == len(fake.cancel_calls) == 1
    assert fake.get_options == [{}]
    assert restarted.budget.snapshot()["summary"]["editor"]["reserved_usd"] == 0


def test_uncertain_cancel_keeps_budget_and_uses_same_id_on_resume(tmp_path):
    fake = FakeResponses([runaway_stream()], [response(status="in_progress")], [TimeoutError("unknown")])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway) as raised:
        e.complete("task", request_key="t001")
    assert raised.value.usage_pending
    assert e.budget.snapshot()["summary"]["editor"]["reserved_usd"] > 0
    fake.retrieves = [response(status="in_progress")]
    fake.cancel_responses = [response(status="cancelled")]
    with pytest.raises(EditorRunaway) as reconciled:
        editor(tmp_path, fake).complete("task", request_key="t001")
    assert not reconciled.value.usage_pending
    assert [call[0] for call in fake.cancel_calls] == ["resp_1", "resp_1"]
    assert len(fake.calls) == 1


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_terminal_response_guard_blocks_and_charges_even_without_stream_deltas(tmp_path, status):
    terminal = response(status=status, text=RUNAWAY_TEXT)
    terminal["incomplete_details"] = {"reason": "max_output_tokens"} if status == "incomplete" else None
    fake = FakeResponses([terminal])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway) as raised:
        e.complete("task", request_key="t001")
    assert not raised.value.usage_pending
    assert e.budget.snapshot()["summary"]["editor"]["spent_usd"] == float(astra_cost(terminal["usage"]))
    assert not fake.cancel_calls


def test_legacy_runaway_receipt_blocks_next_post_without_rewriting_history(tmp_path, monkeypatch):
    terminal = response(status="incomplete", text=RUNAWAY_TEXT)
    terminal["incomplete_details"] = {"reason": "max_output_tokens"}
    fake = FakeResponses([terminal])
    e = editor(tmp_path, fake)
    with monkeypatch.context() as old:
        old.setattr("openwsp.editors.openai.detect_runaway_formatting", lambda _: None)
        with pytest.raises(EditorIncomplete):
            e.complete("task", request_key="t026")
    state = json.loads(e.path.read_text())
    receipt = e.root / state["completed"]["t026"]
    original_receipt = receipt.read_bytes()
    original_budget = e.budget.snapshot()
    with pytest.raises(EditorRunaway):
        editor(tmp_path, fake).complete("next input", request_key="t027")
    assert len(fake.calls) == 1 and not fake.cancel_calls and not fake.gets
    assert receipt.read_bytes() == original_receipt
    assert e.budget.snapshot() == original_budget
    assert json.loads(e.path.read_text())["previous_response_id"] == state["previous_response_id"]


def test_stream_lost_before_response_id_cannot_duplicate_create(tmp_path):
    stream = FakeStream([ConnectionError("lost before first event")])
    fake = FakeResponses([stream])
    e = editor(tmp_path, fake)
    with pytest.raises(UncertainRequest):
        e.complete("task", request_key="t001")
    with pytest.raises(UncertainRequest):
        editor(tmp_path, fake).complete("task", request_key="t001")
    assert len(fake.calls) == 1 and not fake.gets and stream.closed
    assert e.budget.snapshot()["summary"]["editor"]["reserved_usd"] > 0


def test_actual_sdk_sse_reconnect_cancel_and_usage_reconciliation_without_network(tmp_path):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx2")
    requests = []

    def sse(events):
        body = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"
                       for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    def handle(request):
        requests.append((request.method, request.url.path, dict(request.url.params)))
        if request.method == "POST" and request.url.path == "/v1/responses":
            payload = json.loads(request.content)
            assert payload["stream"] is True and payload["background"] is True
            return sse([{"type": "keepalive"}, response_event(response(status="queued")),
                        text_delta("`\n\n" * 200), {"type": "keepalive", "sequence_number": 999}])
        if request.method == "GET" and request.url.params.get("stream") == "true":
            assert request.url.params["starting_after"] == "1"
            return sse([{"type": "keepalive", "sequence_number": 9999}, text_delta("`\n\n" * 200, 2)])
        if request.method == "POST" and request.url.path == "/v1/responses/resp_1/cancel":
            value = response(status="cancelled")
            value["usage"] = None
            return httpx.Response(200, json=value)
        assert request.method == "GET" and request.url.path == "/v1/responses/resp_1"
        return httpx.Response(200, json=response(status="cancelled"))

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        # The adapter must also disable automatic POST retries on supplied SDK clients.
        client = openai.OpenAI(api_key="test-key", base_url="https://provider.invalid/v1", http_client=http_client)
        e = AstraEditor(tmp_path / "editor", BudgetLedger(tmp_path / "budget.json"), client=client,
                        poll_interval=0, sleep=lambda _: None)
        with pytest.raises(EditorRunaway) as detected:
            e.complete("task", request_key="t001")
        assert detected.value.usage_pending and e.client.max_retries == 0
        with pytest.raises(EditorRunaway) as reconciled:
            e.complete("task", request_key="t001")
        assert not reconciled.value.usage_pending
    assert requests == [
        ("POST", "/v1/responses", {}),
        ("GET", "/v1/responses/resp_1", {"starting_after": "1", "stream": "true"}),
        ("POST", "/v1/responses/resp_1/cancel", {}),
        ("GET", "/v1/responses/resp_1", {}),
    ]
    assert e.budget.snapshot()["summary"]["editor"]["reserved_usd"] == 0


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_explicit_runaway_review_preserves_history_billing_and_blocks_old_output(tmp_path, status):
    failed = response(text=RUNAWAY_TEXT, status=status)
    fake = FakeResponses([failed, response("resp_2"), response("resp_3", text=RUNAWAY_TEXT)])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway):
        e.complete("first input", request_key="t001")
    before = json.loads(e.path.read_text())
    receipt = e.root / before["completed"]["t001"]
    receipt_bytes = receipt.read_bytes()
    budget_before = e.budget.snapshot()
    evidence = {"source": "tests/isolated-probe.json", "status": "passed"}
    review = e.acknowledge_runaway("resp_1", reason="Explicit review after separate validation", evidence=evidence)
    after = json.loads(e.path.read_text())
    assert set(after["runaway_reviews"]) == {"resp_1"} and "blocked" not in after
    assert {k: v for k, v in before.items() if k != "blocked"} == {
        k: v for k, v in after.items() if k != "runaway_reviews"}
    assert receipt.read_bytes() == receipt_bytes and e.budget.snapshot() == budget_before
    assert review["source"] == evidence["source"] and review["reviewed_at"] > 0
    assert review["terminal_status"] == status and review["response_id"] == "resp_1"
    assert len(fake.calls) == 1 and not fake.gets and not fake.cancel_calls
    with pytest.raises(EditorRunaway):
        e.complete("first input", request_key="t001")
    assert json.loads(e.path.read_text()) == after  # historical replay must not re-block continuity
    assert editor(tmp_path, fake).complete("next input", request_key="t002")
    assert fake.calls[1]["previous_response_id"] == "resp_1"
    assert fake.calls[1]["prompt_cache_key"] == fake.calls[0]["prompt_cache_key"]
    with pytest.raises(EditorRunaway):
        e.complete("future input", request_key="t003")
    later = json.loads(e.path.read_text())
    assert later["blocked"]["response_id"] == "resp_3"
    assert e.acknowledge_runaway("resp_1", reason=review["reason"], evidence=evidence) == review
    assert json.loads(e.path.read_text()) == later  # old acknowledgement cannot clear a new loop
    with pytest.raises(EditorRunaway):
        e.complete("another input", request_key="t004")
    assert len(fake.calls) == 3


def test_runaway_review_is_immutable_and_idempotent(tmp_path):
    fake = FakeResponses([response(text=RUNAWAY_TEXT)])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway):
        e.complete("task", request_key="t001")
    evidence = {"source": "tests/probe.json", "cost_usd": "0.01"}
    first = e.acknowledge_runaway("resp_1", reason="Reviewed", evidence=evidence)
    audit = next(e.root.glob("runaway_review_*.json"))
    audit_bytes, state_bytes = audit.read_bytes(), e.path.read_bytes()
    budget_before = e.budget.snapshot()
    assert editor(tmp_path, fake).acknowledge_runaway("resp_1", reason="Reviewed", evidence=evidence) == first
    for reason, details in [("Different reason", evidence), ("Reviewed", {"source": "other.json"})]:
        with pytest.raises(EditorError, match="Conflicting"):
            e.acknowledge_runaway("resp_1", reason=reason, evidence=details)
    assert audit.read_bytes() == audit_bytes and e.path.read_bytes() == state_bytes
    assert e.budget.snapshot() == budget_before and len(fake.calls) == 1


@pytest.mark.parametrize("status", ["cancelled", "failed"])
def test_runaway_review_rejects_cancelled_or_failed_continuity(tmp_path, status):
    e = editor(tmp_path, FakeResponses([response(status=status, text=RUNAWAY_TEXT)]))
    with pytest.raises(EditorRunaway):
        e.complete("task", request_key="t001")
    before = e.path.read_bytes(), e.budget.snapshot()
    with pytest.raises(EditorError, match="Only completed/incomplete"):
        e.acknowledge_runaway("resp_1", reason="Review", evidence={"source": "fixture.json"})
    assert (e.path.read_bytes(), e.budget.snapshot()) == before
    assert not list(e.root.glob("runaway_review_*.json"))


def test_runaway_review_rejects_unknown_usage_and_pending_request(tmp_path):
    missing = response(status="incomplete", text=RUNAWAY_TEXT)
    missing["usage"] = None
    fake = FakeResponses([missing])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway):
        e.complete("task", request_key="t001")
    before = e.path.read_bytes(), e.budget.snapshot()
    with pytest.raises(EditorError, match="pending or unknown usage"):
        e.acknowledge_runaway("resp_1", reason="Review", evidence={"source": "fixture.json"})
    assert (e.path.read_bytes(), e.budget.snapshot()) == before
    assert len(fake.calls) == 1 and not fake.gets and not fake.cancel_calls


@pytest.mark.parametrize("evidence", [None, {}, [], {"source": ""}, {"source": 1},
                                      {"source": "file.json", "value": float("nan")},
                                      {"source": "file.json", "value": object()}])
def test_runaway_review_requires_finite_json_with_a_source(tmp_path, evidence):
    fake = FakeResponses()
    e = editor(tmp_path, fake)
    with pytest.raises(ValueError, match="evidence"):
        e.acknowledge_runaway("resp_1", reason="Review", evidence=evidence)
    assert not fake.calls and not e.path.exists()


def test_runaway_review_rejects_missing_reason_wrong_id_and_unsettled_receipt(tmp_path, monkeypatch):
    e = editor(tmp_path, FakeResponses([response(text=RUNAWAY_TEXT)]))
    with pytest.raises(EditorRunaway):
        e.complete("task", request_key="t001")
    with pytest.raises(ValueError, match="reason"):
        e.acknowledge_runaway("resp_1", reason=" ", evidence={"source": "fixture.json"})
    with pytest.raises(EditorError, match="does not match"):
        e.acknowledge_runaway("resp_other", reason="Review", evidence={"source": "fixture.json"})
    unsettled = e.budget.snapshot()
    next(iter(unsettled["reservations"].values()))["status"] = "reserved"
    monkeypatch.setattr(e.budget, "snapshot", lambda: unsettled)
    with pytest.raises(EditorError, match="matching settled"):
        e.acknowledge_runaway("resp_1", reason="Review", evidence={"source": "fixture.json"})
    assert not list(e.root.glob("runaway_review_*.json"))


def test_runaway_review_recovers_audit_written_before_state_update(tmp_path, monkeypatch):
    from openwsp.editors import openai as adapter

    class Crash(BaseException):
        pass

    e = editor(tmp_path, FakeResponses([response(text=RUNAWAY_TEXT)]))
    with pytest.raises(EditorRunaway):
        e.complete("task", request_key="t001")
    atomic = adapter.atomic_json

    def crash_before_index(path, value):
        if path == e.path and value.get("runaway_reviews"):
            raise Crash()
        return atomic(path, value)

    with monkeypatch.context() as fault:
        fault.setattr(adapter, "atomic_json", crash_before_index)
        with pytest.raises(Crash):
            e.acknowledge_runaway("resp_1", reason="Review", evidence={"source": "fixture.json"})
    audit = next(e.root.glob("runaway_review_*.json"))
    original = audit.read_bytes()
    assert "blocked" in json.loads(e.path.read_text())
    review = e.acknowledge_runaway("resp_1", reason="Review", evidence={"source": "fixture.json"})
    assert audit.read_bytes() == original and review == json.loads(original)
    assert "blocked" not in json.loads(e.path.read_text())


def test_changed_runaway_review_audit_cannot_authorize_a_new_request(tmp_path):
    fake = FakeResponses([response(text=RUNAWAY_TEXT)])
    e = editor(tmp_path, fake)
    with pytest.raises(EditorRunaway):
        e.complete("task", request_key="t001")
    e.acknowledge_runaway("resp_1", reason="Review", evidence={"source": "fixture.json"})
    audit = next(e.root.glob("runaway_review_*.json"))
    value = json.loads(audit.read_text())
    value["reason"] = "changed"
    audit.write_text(json.dumps(value))
    with pytest.raises(EditorError, match="audit changed"):
        e.complete("next input", request_key="t002")
    assert len(fake.calls) == 1
