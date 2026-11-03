import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from openwsp.editors.base import EditorIncomplete, EditorRefused, ProviderUnavailable, UncertainRequest
from openwsp.editors.bridge import BridgeConflict, BridgeEditor, BridgeStore, process_request


class LocalEditor:
    """Mock durable idempotent editor; attempts are distinct from generations."""

    def __init__(self, error=None):
        self.records = {}
        self.attempts = self.generations = 0
        self.last_response = None
        self.error = error

    def has_request(self, request_key):
        return request_key in self.records

    def complete(self, messages, *, request_key):
        self.attempts += 1
        if self.error:
            raise self.error
        if request_key in self.records:
            old_messages, text, self.last_response = self.records[request_key]
            assert old_messages == messages
            return text
        self.generations += 1
        text = "A final local-editor response."
        self.last_response = {"status": "completed", "response_id": f"local-{self.generations}",
                              "billing": "chatgpt_subscription_no_api_price_inferred",
                              "usage": {"input_tokens": 10, "output_tokens": 5}}
        self.records[request_key] = (messages, text, self.last_response)
        return text


def submit(store, key="t001", *, conversation=0, message="exact prompt"):
    return store.submit("run-a", key, conversation_index=conversation, messages=message)


def owner(record, worker="worker-a"):
    return {"request_hash": record["request_hash"], "worker_id": worker}


def test_bridge_output_byte_ceiling_rejects_execution_without_repeating_or_rebilling(tmp_path):
    store = BridgeStore(tmp_path / "store")
    record = submit(store)
    store.claim("run-a", "t001", **owner(record))
    store.dispatch("run-a", "t001", **owner(record))
    receipt = {"status": "completed", "response_id": "one-response", "usage": {"output_tokens": 7}}
    store.finish("run-a", "t001", **owner(record), text="é" * 5, receipt=receipt)
    editor = BridgeEditor(tmp_path / "editor", run_id="run-a", store=store, max_output_bytes=9)
    for _ in range(2):
        with pytest.raises(EditorIncomplete) as raised:
            editor.complete("exact prompt", request_key="t001")
        assert raised.value.reason == "output_byte_limit"
        assert editor.last_response["usage"] == {"output_tokens": 7}
        assert editor.last_response["output_bytes"] == 10
    assert store.get("run-a", "t001")["completion"]["receipt"] == receipt
    # A completed but non-executed response does not strand the conversation.
    assert submit(store, key="t002", message="Neutral failure")['status'] == 'queued'


def test_bridge_output_exact_byte_ceiling_is_delivered(tmp_path):
    store = BridgeStore(tmp_path / "store")
    record = submit(store)
    store.claim("run-a", "t001", **owner(record))
    store.dispatch("run-a", "t001", **owner(record))
    store.finish("run-a", "t001", **owner(record), text="é" * 5, receipt={"status": "completed"})
    editor = BridgeEditor(tmp_path / "editor", run_id="run-a", store=store, max_output_bytes=10)
    assert editor.complete("exact prompt", request_key="t001") == "é" * 5


def test_store_request_and_terminal_payloads_are_immutable(tmp_path):
    store = BridgeStore(tmp_path)
    record = submit(store)
    assert record["prompt_sha256"] == hashlib.sha256(b"exact prompt").hexdigest()
    assert submit(store) == record
    with pytest.raises(BridgeConflict, match="different request"):
        submit(store, message="different prompt")
    with pytest.raises(BridgeConflict, match="different request"):
        submit(store, conversation=1)
    with pytest.raises(BridgeConflict, match="hash"):
        store.claim("run-a", "t001", request_hash="wrong", worker_id="worker-a")
    with pytest.raises(BridgeConflict, match="unclaimed"):
        store.dispatch("run-a", "t001", **owner(record))
    claimed = store.claim("run-a", "t001", **owner(record))
    assert store.claim("run-a", "t001", **owner(record)) == claimed
    dispatched = store.dispatch("run-a", "t001", **owner(record))
    assert store.dispatch("run-a", "t001", **owner(record)) == dispatched
    receipt = {"status": "completed", "usage": {"output_tokens": 10}}
    final = store.finish("run-a", "t001", **owner(record), text="answer", receipt=receipt)
    assert store.finish("run-a", "t001", **owner(record), text="answer", receipt=receipt) == final
    assert submit(store) == final and store.pending("run-a") == []
    with pytest.raises(BridgeConflict, match="immutable"):
        store.finish("run-a", "t001", **owner(record), text="changed", receipt=receipt)
    with pytest.raises(BridgeConflict, match="another worker"):
        store.finish("run-a", "t001", **owner(record, "worker-b"), text="answer", receipt=receipt)
    assert store.get("run-a", "t001") == final


def test_two_workers_cannot_claim_or_take_over_one_request(tmp_path):
    store = BridgeStore(tmp_path)
    record = submit(store)

    def claim(worker):
        try:
            return store.claim("run-a", "t001", **owner(record, worker))["claimed_by"]
        except BridgeConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        winners = list(executor.map(claim, ["worker-a", "worker-b"]))
    assert sum(winner is not None for winner in winners) == 1
    winner = next(worker for worker in winners if worker)
    store.dispatch("run-a", "t001", **owner(record, winner))
    with pytest.raises(BridgeConflict, match="another worker"):
        store.claim("run-a", "t001", **owner(record, "worker-c"))
    assert store.get("run-a", "t001")["claimed_by"] == winner
    assert store.pending("run-a")[0]["status"] == "dispatched"


def test_conversation_owner_cannot_silently_change_on_later_turn(tmp_path):
    store = BridgeStore(tmp_path)
    editor = LocalEditor()
    process_request(store, submit(store), worker_id="worker-a", editor=editor)
    second = submit(store, "t002")
    with pytest.raises(BridgeConflict, match="conversation"):
        store.claim("run-a", "t002", **owner(second, "worker-b"))
    process_request(store, second, worker_id="worker-a", editor=editor)
    new_conversation = submit(store, "t003", conversation=1)
    assert store.claim("run-a", "t003", **owner(new_conversation, "worker-b"))["claimed_by"] == "worker-b"


def test_worker_observes_committed_claim_and_dispatch_before_inference(tmp_path):
    committed = []
    store = BridgeStore(tmp_path, commit=lambda: committed.append(True))
    record = submit(store)
    local = LocalEditor()
    complete = local.complete

    def checked_complete(messages, *, request_key):
        observed = store.get("run-a", "t001")
        assert len(committed) >= 3
        assert observed["status"] == "dispatched" and observed["claimed_by"] == "worker-a"
        assert observed["dispatch"]["request_hash"] == record["request_hash"]
        assert messages == record["messages"] and request_key == "bridge:" + record["request_hash"]
        return complete(messages, request_key=request_key)

    local.complete = checked_complete
    result = process_request(store, record, worker_id="worker-a", editor=local)
    assert result["status"] == "completed" and local.generations == 1
    assert process_request(store, record, worker_id="worker-a", editor=local) == result
    assert local.attempts == 1


def test_lost_completion_upload_recovers_same_local_editor_key_without_regeneration(tmp_path):
    store = BridgeStore(tmp_path)
    record = submit(store)
    local = LocalEditor()

    class LostUpload:
        def __getattr__(self, name):
            return getattr(store, name)

        def finish(self, *args, **kwargs):
            raise ConnectionError("upload did not reach server")

    with pytest.raises(ConnectionError):
        process_request(LostUpload(), record, worker_id="worker-a", editor=local)
    assert store.get("run-a", "t001")["status"] == "dispatched"
    assert local.generations == 1
    result = process_request(store, record, worker_id="worker-a", editor=local)
    assert result["status"] == "completed" and local.generations == 1 and local.attempts == 2


def test_worker_does_not_infer_when_dispatch_readback_is_unavailable(tmp_path):
    store = BridgeStore(tmp_path)
    record = submit(store)
    local = LocalEditor()

    class NoReadback:
        def __getattr__(self, name):
            return getattr(store, name)

        def get(self, *args):
            raise ConnectionError("unavailable")

    with pytest.raises(ConnectionError):
        process_request(NoReadback(), record, worker_id="worker-a", editor=local)
    assert local.generations == 0
    assert store.get("run-a", "t001")["status"] == "dispatched"
    recovered = process_request(store, record, worker_id="worker-a", editor=local)
    assert recovered["status"] == "blocked"
    assert recovered["blocked_error"]["error_type"] == "MissingLocalRequest"
    assert local.generations == 0


@pytest.mark.parametrize("predicate", ["missing", "false", "uncertain"])
def test_dispatched_request_cannot_infer_without_a_verified_local_journal(tmp_path, predicate):
    store = BridgeStore(tmp_path)
    record = submit(store)
    store.claim("run-a", "t001", **owner(record))
    store.dispatch("run-a", "t001", **owner(record))
    calls = []
    local = SimpleNamespace(complete=lambda *args, **kwargs: calls.append(True))
    if predicate == "false":
        local.has_request = lambda _: False
    elif predicate == "uncertain":
        def uncertain(_):
            raise UncertainRequest("Local journal identity is inconsistent")
        local.has_request = uncertain
    blocked = process_request(store, record, worker_id="worker-a", editor=local)
    assert blocked["status"] == "blocked" and blocked["blocked_error"]["category"] == "uncertain_request"
    assert not calls


def test_completed_bridge_result_can_be_replayed_without_local_editor_state(tmp_path):
    store = BridgeStore(tmp_path)
    record = submit(store)
    final = process_request(store, record, worker_id="worker-a", editor=LocalEditor())
    assert process_request(store, record, worker_id="worker-a", editor=SimpleNamespace()) == final


@pytest.mark.parametrize("error,exception", [
    (UncertainRequest("sensitive-test-token"), UncertainRequest),
    (ProviderUnavailable("sensitive-test-token"), ProviderUnavailable),
    (EditorRefused("sensitive-test-token"), EditorRefused),
])
def test_blocked_error_is_immutable_and_contains_no_exception_text(tmp_path, error, exception):
    store = BridgeStore(tmp_path / "store")
    record = submit(store)
    result = process_request(store, record, worker_id="worker-a", editor=LocalEditor(error))
    assert result["status"] == "blocked"
    assert "sensitive-test-token" not in json.dumps(result)
    editor = BridgeEditor(tmp_path / "remote", run_id="run-a", store=store)
    with pytest.raises(exception):
        editor.complete("exact prompt", request_key="t001")
    with pytest.raises(ProviderUnavailable):
        editor.complete("next prompt", request_key="t002")
    with pytest.raises(BridgeConflict, match="immutable"):
        store.finish("run-a", "t001", **owner(record), text="partial text", receipt={"status": "completed"})


def test_receipts_cannot_copy_authentication_into_bridge_records(tmp_path):
    store = BridgeStore(tmp_path)
    record = submit(store)
    store.claim("run-a", "t001", **owner(record))
    store.dispatch("run-a", "t001", **owner(record))
    with pytest.raises(ValueError, match="Credential"):
        store.finish("run-a", "t001", **owner(record), text="answer",
                     receipt={"metadata": {"authorization": "sensitive-test-token"}})
    assert "sensitive-test-token" not in store._path("run-a").read_text()


@pytest.mark.parametrize("receipt", [
    {"status": "incomplete", "text": "partial"},
    {"status": "cancelled", "usage": None},
    {"status": "completed", "text": "different text"},
])
def test_partial_or_mismatched_receipt_never_becomes_a_completion(tmp_path, receipt):
    store = BridgeStore(tmp_path)
    record = submit(store)
    local = LocalEditor()

    def complete(*args, **kwargs):
        local.last_response = receipt
        return "partial"

    local.complete = complete
    result = process_request(store, record, worker_id="worker-a", editor=local)
    assert result["status"] == "blocked" and result["completion"] is None
    assert result["blocked_error"]["error_type"] == "InvalidReceipt"


def test_remote_editor_round_trip_replay_and_explicit_handoff(tmp_path):
    store = BridgeStore(tmp_path / "store")
    local = {}
    commits, reloads = [], []

    def work(meta):
        record = store.get(meta["run_id"], meta["request_key"])
        editor = local.setdefault(record["conversation_index"], LocalEditor())
        process_request(store, record, worker_id="worker-a", editor=editor)

    editor = BridgeEditor(tmp_path / "remote", run_id="run-a", store=store, on_poll=work,
                          sleep=lambda _: None, commit=lambda: commits.append(True), reload=lambda: reloads.append(True))
    answer = editor.complete("exact prompt", request_key="t001")
    assert editor.last_response["_openwsp_bridge"]["request_hash"] == store.get("run-a", "t001")["request_hash"]
    assert editor.last_response["billing"] == "chatgpt_subscription_no_api_price_inferred"
    assert len(commits) == 2 and len(reloads) == 2
    restarted = BridgeEditor(tmp_path / "remote", run_id="run-a", store=store, on_poll=work, sleep=lambda _: None)
    assert restarted.complete("exact prompt", request_key="t001") == answer
    assert local[0].generations == 1
    with pytest.raises(BridgeConflict, match="different input"):
        restarted.complete("changed", request_key="t001")
    restarted.new_conversation()
    assert restarted.complete("new context", request_key="t041") == answer
    assert store.get("run-a", "t041")["conversation_index"] == 1 and local[1].generations == 1
    assert restarted.complete("exact prompt", request_key="t001") == answer


def test_cached_replay_rejects_a_valid_record_for_a_different_key(tmp_path):
    store = BridgeStore(tmp_path / "store")
    first = process_request(store, submit(store), worker_id="worker-a", editor=LocalEditor())
    second = process_request(store, submit(store, "t002"), worker_id="worker-a", editor=LocalEditor())
    editor = BridgeEditor(tmp_path / "remote", run_id="run-a", store=store)
    editor.complete("exact prompt", request_key="t001")
    state = json.loads(editor.path.read_text())
    receipt_path = editor.root / state["completed"]["t001"]
    receipt_path.write_text(json.dumps(second))
    assert first["messages"] == second["messages"]
    with pytest.raises(BridgeConflict, match="different request"):
        editor.complete("exact prompt", request_key="t001")
    assert editor.last_response["_openwsp_bridge"]["request_key"] == "t001"


def test_lost_submit_reply_keeps_same_intent_and_resumes_without_duplicate_request(tmp_path):
    store = BridgeStore(tmp_path / "store")

    class LostReply:
        def submit(self, *args, **kwargs):
            store.submit(*args, **kwargs)
            raise ConnectionError("sensitive-test-token")

    editor = BridgeEditor(tmp_path / "remote", run_id="run-a", store=LostReply())
    with pytest.raises(ProviderUnavailable) as raised:
        editor.complete("exact prompt", request_key="t001")
    assert "sensitive-test-token" not in str(raised.value)
    pending = json.loads(editor.path.read_text())["pending"]
    assert pending["request_hash"] == store.get("run-a", "t001")["request_hash"]
    local = LocalEditor()
    restarted = BridgeEditor(tmp_path / "remote", run_id="run-a", store=store, sleep=lambda _: None,
                             on_poll=lambda _: process_request(store, store.get("run-a", "t001"),
                                                               worker_id="worker-a", editor=local))
    assert restarted.complete("exact prompt", request_key="t001")
    assert local.generations == 1 and len(json.loads(store._path("run-a").read_text())["requests"]) == 1


def test_timeout_preserves_pending_request_and_refuses_handoff_or_another_key(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("openwsp.editors.bridge.time.monotonic", lambda: now[0])
    store = BridgeStore(tmp_path / "store")
    editor = BridgeEditor(tmp_path / "remote", run_id="run-a", store=store, max_wait_seconds=1,
                          poll_interval=1, sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    with pytest.raises(UncertainRequest, match="expired"):
        editor.complete("exact prompt", request_key="t001")
    with pytest.raises(UncertainRequest):
        editor.complete("next", request_key="t002")
    with pytest.raises(UncertainRequest):
        editor.new_conversation()
    assert len(store.pending("run-a")) == 1


def test_hash_mismatched_terminal_result_does_not_advance_remote_state(tmp_path):
    store = BridgeStore(tmp_path / "store")
    record = submit(store)
    result = process_request(store, record, worker_id="worker-a", editor=LocalEditor())
    result["completion"]["request_hash"] = "different"
    broker = SimpleNamespace(submit=lambda *args, **kwargs: result)
    editor = BridgeEditor(tmp_path / "remote", run_id="run-a", store=broker)
    with pytest.raises(BridgeConflict, match="completion"):
        editor.complete("exact prompt", request_key="t001")
    state = json.loads(editor.path.read_text())
    assert state["pending"] is not None and not state["completed"]


def test_idempotent_dispatch_recommits_after_lost_commit_acknowledgement(tmp_path):
    store = BridgeStore(tmp_path)
    record = submit(store)
    store.claim("run-a", "t001", **owner(record))
    calls = []

    def lost_commit():
        calls.append(True)
        if len(calls) == 1:
            raise ConnectionError("commit acknowledgement lost")

    store.commit = lost_commit
    with pytest.raises(ConnectionError):
        store.dispatch("run-a", "t001", **owner(record))
    assert store.dispatch("run-a", "t001", **owner(record))["status"] == "dispatched"
    assert len(calls) == 2
