"""Driver crash recovery tests with no provider, model, or sandbox execution."""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from replications.exp8 import live


def test_new_subscription_attempt_does_not_adopt_old_state_or_work(driver, monkeypatch, tmp_path):
    from openwsp.backends import editor_bridge, sandbox
    from openwsp.editors import bridge
    from openwsp.editors.base import ProviderUnavailable
    from replications.exp8.run_identity import LiveIdentity

    driver.write_state(passes=4, turn=26, runs=10, next="Previous attempt's private reasoning")
    old_state = (driver.root / "state.json").read_bytes()
    old_work = tmp_path / "work/live"
    old_work.mkdir(parents=True)
    (old_work / "notes.md").write_text("Previous attempt's private reasoning")
    observed = {}
    original_create = sandbox.ModalSandboxTransport.create

    def create(*args, **kwargs):
        observed["work_subpath"] = kwargs["work_subpath"]
        return original_create(*args, **kwargs)

    class Editor:
        last_response = None

        def __init__(self, *args, **kwargs):
            observed["run_id"] = kwargs["run_id"]

        def complete(self, message, **kwargs):
            observed["prompt"] = message
            raise ProviderUnavailable("Synthetic stop before any provider call")

    monkeypatch.setattr(sandbox.ModalSandboxTransport, "create", create)
    monkeypatch.setattr(editor_bridge, "ModalBridgeStore", lambda: object())
    monkeypatch.setattr(bridge, "BridgeEditor", Editor)
    identity = LiveIdentity("codex", "separate-attempt")
    with pytest.raises(ProviderUnavailable):
        live.run_live({"stages": {"kernel": {"status": "passed"}}}, object(), lambda: None,
                      app=object(), identity=identity)
    assert (driver.root / "state.json").read_bytes() == old_state
    assert observed["work_subpath"] == "live-runs/separate-attempt"
    assert driver.tool_options == [{"job_prefix": "live/separate-attempt", "work_subpath": "live-runs/separate-attempt"}]
    assert observed["run_id"] == "separate-attempt"
    assert observed["prompt"] == (live.ROOT / "prompts/message.txt").read_text()
    state = json.loads((tmp_path / "data/live-runs/separate-attempt/state.json").read_text())
    assert (state["passes"], state["turn"], state["runs"]) == (0, 0, 0)
    assert not (tmp_path / "work/live-runs/separate-attempt/notes.md").exists()


@pytest.fixture
def driver(monkeypatch, tmp_path):
    root = tmp_path / "data/live-run"
    root.mkdir(parents=True)
    state_path = root / "state.json"
    initial = {
        "budget": 50,
        "mode": "qwen35think",
        "prompt": "v3-final",
        "turn": 0,
        "runs": 0,
        "passes": 50,
        "requests": 0,
        "handoffs": 0,
        "since_handoff": 0,
        "done": False,
        "notices": [],
        "unexplained_streak": 0,
        "next": "Continue",
        "pending": None,
    }
    initial_runs, editor_messages, tool_calls, tool_options = [], [], [], []
    replies = {"t001": "The submitted edit is final."}
    volume = SimpleNamespace(commit=lambda: None, reload=lambda: None)
    application = ModuleType("openwsp.backends.modal_app")
    for name in ("data_volume", "model_volume", "practice_volume", "work_volume"):
        setattr(application, name, volume)
    application.sandbox_image = object()
    monkeypatch.setitem(sys.modules, application.__name__, application)

    def mapped_path(value):
        value = str(value)
        if value in ("/data", "/work") or value.startswith(("/data/", "/work/")):
            return tmp_path / value.lstrip("/")
        return Path(value)

    monkeypatch.setattr(live, "Path", mapped_path)

    class Transport:
        def __init__(self, runs):
            self.sandbox = SimpleNamespace(object_id="sb-test")
            self.runs = runs

        @classmethod
        def create(cls, *args, **kwargs):
            initial_runs.append(kwargs["initial_runs"])
            return cls(kwargs["initial_runs"])

        def commit_work(self):
            pass

        def run(self, code, *, request_id):
            raise AssertionError("Cached/unknown Python cells must never be re-executed")

        def inspect_work(self):
            return {"notes": "", "edited": [], "files": []}

    class Lease:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, factory):
            return factory()

        def check(self):
            pass

        def close(self):
            pass

    class Editor:
        def __init__(self, *args, **kwargs):
            self.last_response = None

        def complete(self, message, *, request_key):
            editor_messages.append((request_key, message))
            return replies.get(request_key, "The submitted edit is final.")

        def new_conversation(self):
            pass

    class Tool:
        def __init__(self, *args, **kwargs):
            tool_options.append(kwargs)

        def counts_as_pass(self, number):
            return True

        def execute(self, request, number, turn):
            tool_calls.append((request, number, turn))
            return {"ok": True, "type": "generate", "file": "/work/gen_1.jsonl"}

    import openwsp.backends.sandbox as sandbox_module
    import openwsp.backends.sandbox_budget as budget_module
    from openwsp import core

    monkeypatch.setattr(sandbox_module, "ModalSandboxTransport", Transport)
    monkeypatch.setattr(budget_module, "SandboxBudgetLease", Lease)
    monkeypatch.setattr(live, "AstraEditor", Editor)
    monkeypatch.setattr(live, "ForwardTool", Tool)
    monkeypatch.setattr(live, "freeze_checkpoint", lambda *args: ("/data/submissions/test", "edit-sha", []))
    monkeypatch.setattr(core, "WeightStore", lambda *args: SimpleNamespace(export_edit=lambda *a, **kw: None))
    monkeypatch.setattr(live, "CONFIG", {**live.CONFIG, "notices": {}})

    def write_state(**changes):
        state = {**initial, **changes}
        state_path.write_text(json.dumps(state))
        return state

    def run():
        return live.run_live(
            {"stages": {"kernel": {"status": "passed"}}}, object(), lambda: None, app=object()
        )

    return SimpleNamespace(
        root=root,
        write_state=write_state,
        read_state=lambda: json.loads(state_path.read_text()),
        run=run,
        replies=replies,
        initial_runs=initial_runs,
        editor_messages=editor_messages,
        tool_calls=tool_calls,
        tool_options=tool_options,
    )


def test_failed_final_freeze_does_not_mark_run_done(driver, monkeypatch):
    driver.write_state()
    original = live.freeze_checkpoint

    def fail_freeze(*args):
        raise OSError("Interrupted final checkpoint write")

    monkeypatch.setattr(live, "freeze_checkpoint", fail_freeze)
    with pytest.raises(OSError, match="Interrupted final checkpoint"):
        driver.run()
    assert driver.read_state()["done"] is False
    monkeypatch.setattr(live, "freeze_checkpoint", original)
    result = driver.run()
    assert result["done"] is True
    assert result["submission"]["sha256"] == "edit-sha"


@pytest.mark.parametrize("admitted", [False, True])
def test_restart_notice_precedes_only_a_never_admitted_editor_turn(driver, admitted):
    driver.write_state(turn=2, pending={"kind": "editor", "turn": 3})
    turn = driver.root / "turns/t003"
    turn.mkdir(parents=True)
    (turn / "prompt.txt").write_text("Saved request input")
    (driver.root / "editor").mkdir()
    (driver.root / "editor/editor_state.json").write_text(json.dumps({
        "pending": {"key": "t003", "response_id": "existing"} if admitted else None, "completed": {}}))
    driver.run()
    received = driver.editor_messages[0][1]
    if admitted:
        assert received == "Saved request input"
    else:
        assert received.startswith("The kernel was restarted; variables were lost")
        assert received.endswith("Saved request input")
    assert (turn / "prompt.txt").read_text() == received


def test_completed_forward_receipt_counts_once_after_logging_failure(driver, monkeypatch):
    driver.write_state(passes=49)
    driver.replies["t001"] = 'I will inspect the result.\n```forward\n{"type":"generate"}\n```'
    original = live.append_record

    def fail_append(*args):
        raise OSError("Interrupted transcript append")

    monkeypatch.setattr(live, "append_record", fail_append)
    with pytest.raises(OSError, match="Interrupted transcript"):
        driver.run()
    monkeypatch.setattr(live, "append_record", original)
    result = driver.run()
    assert result["done"] is True
    assert result["passes"] == 50
    assert result["requests"] == 1
    assert result["counted_requests"] == [1]
    assert "variables were lost" in driver.editor_messages[-1][1]


def test_effective_duplicate_is_served_without_incrementing_pass_count(driver, monkeypatch):
    driver.write_state(passes=49)
    reply = 'The result will inform the next action.\n```forward\n{"type":"generate"}\n```'
    driver.replies.update(t001=reply, t002=reply)
    monkeypatch.setattr(live.ForwardTool, "counts_as_pass", lambda self, number: number != 1)
    result = driver.run()
    assert result["done"] and result["passes"] == 50 and result["requests"] == 2
    assert result["counted_requests"] == [2]
    assert "FORWARD PASS 49 of 50" in driver.editor_messages[1][1]
    assert "duplicate" not in driver.editor_messages[1][1].lower()


def test_oversized_turn_never_executes_a_complete_code_prefix(driver, monkeypatch):
    driver.write_state()
    monkeypatch.setattr(live, "OUTPUT_BYTE_LIMIT", 128)
    driver.replies["t001"] = 'Explanation.\n```python\nraise RuntimeError("must not execute")\n```' + "é" * 64
    result = driver.run()
    assert result["done"] and result["turn"] == 2 and result["runs"] == 0
    event = json.loads((driver.root / "turns/t001/editor_event.json").read_text())
    assert event["reason"] == "output_byte_limit" and event["executed"] is False
    assert "No code or forward pass" in driver.editor_messages[1][1]
    assert not (driver.root / "turns/t001/answer.md").exists()


def test_saved_oversized_turn_preserves_its_original_usage_receipt(driver, monkeypatch):
    driver.write_state(pending={"kind": "editor", "turn": 1})
    monkeypatch.setattr(live, "OUTPUT_BYTE_LIMIT", 128)
    directory = driver.root / "turns/t001"
    directory.mkdir(parents=True)
    (directory / "prompt.txt").write_text("Exact original input")
    (directory / "answer.md").write_text("é" * 100)
    receipt = {"status": "completed", "response_id": "original", "usage": {"output_tokens": 100}, "cost_usd": 0.25}
    (directory / "run.json").write_text(json.dumps(receipt))
    result = driver.run()
    saved = json.loads((directory / "run.json").read_text())
    assert result["done"] and result["turn"] == 2
    assert all(saved[key] == value for key, value in receipt.items())
    assert [key for key, _ in driver.editor_messages] == ["t002"]
    assert saved["execution_rejected"] == "output_byte_limit"


def test_delivery_interruption_keeps_the_same_pending_forward_request(driver, monkeypatch):
    driver.write_state(passes=49)
    driver.replies["t001"] = 'I will inspect the result.\n```forward\n{"type":"generate"}\n```'
    original = live.ForwardTool.execute

    def interrupted_delivery(*args):
        raise live.JobPending("Saved artifacts are not delivered yet")

    monkeypatch.setattr(live.ForwardTool, "execute", interrupted_delivery)
    with pytest.raises(live.JobPending, match="not delivered"):
        driver.run()
    state = driver.read_state()
    assert state["passes"] == 49 and state["requests"] == 1 and state["turn"] == 0
    assert state["pending"] == {"kind": "forward", "request": 1, "turn": 1}
    assert not (driver.root / "requests/fwd_1.done.json").exists()
    monkeypatch.setattr(live.ForwardTool, "execute", original)
    result = driver.run()
    assert result["passes"] == 50 and result["requests"] == 1
    assert result["counted_requests"] == [1]
    assert [turn for turn, _ in driver.editor_messages] == ["t001", "t002"]


def test_incomplete_editor_turn_is_recorded_without_executing_partial_code(driver, monkeypatch):
    driver.write_state(runs=3, requests=69, turn=0,
                       pending={"kind": "editor", "turn": 1})
    turn = driver.root / "turns/t001"
    turn.mkdir(parents=True)
    (turn / "prompt.txt").write_text("Original pending input")
    partial = "I will inspect weights.\n```python\nraise AssertionError('must not run')"
    original = live.AstraEditor.complete

    def complete(self, message, *, request_key):
        result = original(self, message, request_key=request_key)
        if request_key == "t001":
            self.last_response = {"status": "incomplete", "text": partial,
                                  "usage": {"output_tokens": 32768}, "cost_usd": "1.683131",
                                  "response_file": "response_saved.json"}
            raise live.EditorIncomplete("max_output_tokens", "resp-saved")
        return result

    monkeypatch.setattr(live.AstraEditor, "complete", complete)
    result = driver.run()
    assert result["done"] is True and result["turn"] == 2
    assert result["passes"] == 50 and result["runs"] == 3 and result["requests"] == 69
    assert driver.tool_calls == []
    assert not (turn / "answer.md").exists()
    event = json.loads((turn / "editor_event.json").read_text())
    assert event["executed"] is False and event["reason"] == "max_output_tokens"
    assert event["partial_text_sha256"] == hashlib.sha256(partial.encode()).hexdigest()
    assert "text" not in event and "must not run" not in json.dumps(event)
    assert json.loads((turn / "run.json").read_text())["text"] == partial
    assert driver.editor_messages[0] == ("t001", "Original pending input")
    assert "No code or forward pass" in driver.editor_messages[1][1]
    assert "variables were lost" in driver.editor_messages[1][1]


def test_cached_refused_cell_preserves_replacement_run_counter(driver):
    driver.write_state(pending={"kind": "python", "turn": 1, "request_id": "live-t001"})
    turn = driver.root / "turns/t001"
    turn.mkdir(parents=True)
    (turn / "prompt.txt").write_text("Continue")
    (turn / "answer.md").write_text("I will inspect a tensor.\n```python\n1 + 1\n```")
    (turn / "tool_result.json").write_text(
        json.dumps(
            {
                "ok": False,
                "output": "",
                "error": "ProtocolRefused",
                "seconds": 0,
                "runs": 0,
                "restarted": False,
            }
        )
    )
    result = driver.run()
    assert driver.initial_runs == [0]
    assert result["runs"] == 0
    assert result["done"] is True
    assert "variables were lost" in driver.editor_messages[-1][1]


def test_unknown_claimed_cell_is_not_repeated_in_new_sandbox(driver):
    driver.write_state(pending={"kind": "python", "turn": 1, "request_id": "live-t001"}, runs=4)
    turn = driver.root / "turns/t001"
    turn.mkdir(parents=True)
    (turn / "prompt.txt").write_text("Continue")
    (turn / "answer.md").write_text("I will inspect a tensor.\n```python\n1 + 1\n```")
    result = driver.run()
    receipt = json.loads((turn / "tool_result.json").read_text())
    assert driver.initial_runs == [5]
    assert result["runs"] == 5
    assert receipt["ok"] is False
    assert "not repeated" in receipt["error"]
    assert "variables were lost" in driver.editor_messages[-1][1]
