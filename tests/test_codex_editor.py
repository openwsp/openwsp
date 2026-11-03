"""Subprocess-backed offline CLI tests: the executable is a local Python stub."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from openwsp.editors.base import (
    EditorError,
    EditorRefused,
    EditorRunaway,
    ProviderUnavailable,
    UncertainRequest,
)
from openwsp.editors.codex import CATALOG_SHA256, INSTRUCTIONS, CodexEditor, _snapshot

THREAD = "e66b4164-83c5-40a1-8355-5b282b8c929a"
OTHER_THREAD = "b4100299-5452-43fa-b0cd-cedd7a8faf67"
USAGE = {
    "input_tokens": 10,
    "cached_input_tokens": 3,
    "cache_write_input_tokens": 2,
    "output_tokens": 5,
    "reasoning_output_tokens": 2,
}

STUB = r"""
import json, os, pathlib, sys, time
if "--version" in sys.argv:
    print("codex-cli 0.163.0")
    raise SystemExit(0)
home = pathlib.Path(os.environ["CODEX_HOME"])
scenario = json.loads((home / "scenario.json").read_text())
log = home / "launches.jsonl"
number = len(log.read_text().splitlines()) if log.exists() else 0
with log.open("a") as stream:
    stream.write(json.dumps({"argv": sys.argv[1:], "env_keys": sorted(os.environ),
                             "cwd": os.getcwd(), "prompt": sys.stdin.read()}) + "\n")
current = scenario[number] if isinstance(scenario, list) else scenario
def emit(event):
    print(json.dumps(event), flush=True)
emit({"type": "thread.started", "thread_id": current.get("thread", "e66b4164-83c5-40a1-8355-5b282b8c929a")})
emit({"type": "turn.started"})
if current.get("sleep"):
    time.sleep(current["sleep"])
if current.get("bad_json"):
    print("not json", flush=True)
    raise SystemExit(0)
if current.get("error"):
    emit({"type": "turn.failed", "error": {"message": current["error"]}})
    raise SystemExit(1)
if current.get("item"):
    emit({"type": "item.completed", "item": current["item"]})
    if current.get("after_item_sleep"):
        time.sleep(current["after_item_sleep"])
answer = current.get("text", "Final answer")
emit({"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": answer}})
if not current.get("omit_final"):
    target = pathlib.Path(sys.argv[sys.argv.index("--output-last-message") + 1])
    target.write_text(current.get("saved", answer) + "\n")
if current.get("interrupted"):
    raise SystemExit(1)
usage = current.get("usage", {"input_tokens": 10 * (number + 1), "cached_input_tokens": 3 * (number + 1),
                             "cache_write_input_tokens": 2 * (number + 1), "output_tokens": 5 * (number + 1),
                             "reasoning_output_tokens": 2 * (number + 1)})
emit({"type": "turn.completed", "usage": usage})
raise SystemExit(current.get("exit_code", 0))
"""


def make_editor(tmp_path, scenario=None, **kwargs):
    home = tmp_path / "cli-home"
    home.mkdir(mode=0o700, exist_ok=True)
    (home / "scenario.json").write_text(json.dumps(scenario if scenario is not None else {}))
    executable = tmp_path / "fake-codex"
    executable.write_text(f"#!{sys.executable}\n" + STUB)
    executable.chmod(0o700)
    return CodexEditor(
        tmp_path / "state", codex_home=home, executable=executable, poll_interval=0.01, **kwargs
    )


def launches(instance):
    path = instance.home / "launches.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def config(argv):
    return {
        argument.split("=", 1)[0]: json.loads(argument.split("=", 1)[1])
        for index, argument in enumerate(argv)
        if index and argv[index - 1] == "-c"
    }


def test_persistent_exact_uuid_resume_honest_usage_and_completed_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-inherit")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-inherit")
    monkeypatch.setenv("OPENAI_BASE_URL", "must-not-inherit")
    monkeypatch.setenv("CODEX_CONFIG", "must-not-inherit")
    editor = make_editor(tmp_path)
    assert editor.complete("First input", request_key="one") == "Final answer"
    assert editor.last_response["usage_snapshot"] == {
        "cumulative": USAGE,
        "delta": USAGE,
        "basis": "new_thread_total",
    }
    assert editor.complete("First input", request_key="one") == "Final answer"
    restarted = make_editor(tmp_path)
    assert (
        restarted.complete([{"role": "user", "content": "Next input"}], request_key="two") == "Final answer"
    )
    assert restarted.last_response["usage_snapshot"]["delta"] == USAGE
    assert restarted.last_response["usage_snapshot"]["basis"] == "same_thread_cumulative_difference"
    calls = launches(editor)
    assert len(calls) == 2
    assert calls[0]["prompt"] == "First input" and calls[1]["prompt"] == "Next input"
    assert calls[0]["argv"][:3] == ["exec", "--sandbox", "read-only"]
    assert calls[1]["argv"][:4] == ["exec", "--sandbox", "read-only", "resume"]
    assert calls[1]["argv"][-2:] == [THREAD, "-"]
    for call in calls:
        assert "--ephemeral" not in call["argv"] and "--last" not in call["argv"]
        assert {"--ignore-user-config", "--ignore-rules", "--strict-config"} <= set(call["argv"])
        assert not {"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_CONFIG"} & set(
            call["env_keys"]
        )
        assert "CODEX_HOME" in call["env_keys"]
        options = config(call["argv"])
        assert options["forced_login_method"] == "chatgpt"
        assert options["model_provider"] == "openai"
        assert options["cli_auth_credentials_store"] == "file"
        assert options["model_reasoning_effort"] == "max" and options["model_verbosity"] == "low"
        assert options["project_doc_max_bytes"] == 0 and options["developer_instructions"] == ""
        assert options["tools.update_plan.enabled"] is False
        assert options["features.shell_tool"] is False and options["features.code_mode_only"] is False
        assert Path(options["model_instructions_file"]).read_text() == INSTRUCTIONS
        assert call["cwd"] == str(editor.work)
    assert "cost_usd" not in restarted.last_response
    assert restarted.last_response["billing"] == "chatgpt_subscription_no_api_price_inferred"


def test_neutral_config_matches_the_proven_offline_transport_fixture(tmp_path):
    from scripts.check_codex_transport import build_config

    editor = make_editor(tmp_path)
    directory = editor.root / "example"
    actual = config(editor._command(directory, None))
    for key, value in build_config(directory / "instructions.txt", editor.catalog).items():
        assert actual[key] == value
    import hashlib

    assert hashlib.sha256(editor.catalog.read_bytes()).hexdigest() == CATALOG_SHA256


def test_new_conversation_is_explicit_and_old_keys_remain_idempotent(tmp_path):
    editor = make_editor(tmp_path, [{}, {"thread": OTHER_THREAD, "usage": USAGE}])
    editor.complete("Original", request_key="one")
    editor.new_conversation()
    assert editor.complete("Original", request_key="one") == "Final answer"
    editor.complete("Handoff", request_key="two")
    calls = launches(editor)
    assert len(calls) == 2 and "resume" not in calls[1]["argv"]
    assert editor.last_response["usage_snapshot"]["basis"] == "new_thread_total"
    with pytest.raises(EditorError, match="different input"):
        editor.complete("Changed", request_key="one")


def test_has_request_is_read_only_and_checks_persisted_completed_identity(tmp_path):
    editor = make_editor(tmp_path)
    before = sorted(str(path.relative_to(editor.root)) for path in editor.root.rglob("*"))
    assert editor.has_request("one") is False
    assert sorted(str(path.relative_to(editor.root)) for path in editor.root.rglob("*")) == before
    assert not (editor.root / "codex.lock").exists()
    editor.complete("Preserve\r\nnewlines", request_key="one")
    restarted = make_editor(tmp_path)
    before = {
        str(path.relative_to(editor.root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in editor.root.rglob("*")
        if path.is_file()
    }
    assert restarted.has_request("one") is True
    assert restarted.has_request("missing") is False
    assert restarted.last_response is None
    after = {
        str(path.relative_to(editor.root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in editor.root.rglob("*")
        if path.is_file()
    }
    assert before == after and len(launches(editor)) == 1
    restarted.new_conversation()
    assert restarted.has_request("one") is True


def test_has_request_recognizes_pending_intent_without_relaunching(tmp_path):
    editor = make_editor(tmp_path, {"interrupted": True})
    with pytest.raises(UncertainRequest):
        editor.complete("Task", request_key="one")
    restarted = make_editor(tmp_path)
    assert restarted.has_request("one") is True
    with pytest.raises(UncertainRequest):
        restarted.complete("Task", request_key="one")
    assert len(launches(editor)) == 1


def test_has_request_detects_lost_state_with_orphaned_intent(tmp_path):
    editor = make_editor(tmp_path)
    editor.complete("Task", request_key="one")
    editor.path.unlink()
    restarted = make_editor(tmp_path)
    with pytest.raises(UncertainRequest, match="inconsistent"):
        restarted.has_request("one")
    assert not restarted.path.exists() and len(launches(editor)) == 1


def test_has_request_returns_false_after_all_local_state_is_lost(tmp_path):
    editor = make_editor(tmp_path)
    editor.complete("Task", request_key="one")
    shutil.rmtree(editor.root)
    restarted = make_editor(tmp_path)
    assert restarted.has_request("one") is False
    assert not restarted.path.exists() and len(launches(editor)) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "missing_intent",
        "missing_receipt",
        "wrong_key",
        "wrong_prompt",
        "wrong_receipt",
        "missing_lock",
        "malformed_state",
    ],
)
def test_has_request_rejects_inconsistent_linked_records(tmp_path, damage):
    editor = make_editor(tmp_path)
    editor.complete("Task", request_key="one")
    request_path = next(editor.root.glob("*/request.json"))
    receipt_path = request_path.parent / "receipt.json"
    if damage == "missing_intent":
        request_path.unlink()
    elif damage == "missing_receipt":
        receipt_path.unlink()
    elif damage == "wrong_key":
        intent = json.loads(request_path.read_text())
        intent["key"] = "someone-else"
        request_path.write_text(json.dumps(intent))
    elif damage == "wrong_prompt":
        (request_path.parent / "prompt.txt").write_text("Changed prompt")
    elif damage == "wrong_receipt":
        receipt = json.loads(receipt_path.read_text())
        receipt["response_id"] = "unrelated"
        receipt_path.write_text(json.dumps(receipt))
    elif damage == "missing_lock":
        (editor.root / "codex.lock").unlink()
    else:
        editor.path.write_text("invalid JSON")
    with pytest.raises(UncertainRequest):
        editor.has_request("one")
    assert len(launches(editor)) == 1


def test_has_request_does_not_wait_or_change_an_actively_locked_editor(tmp_path):
    editor = make_editor(tmp_path)
    with editor._locked_state(), pytest.raises(UncertainRequest, match="busy"):
        editor.has_request("one")
    assert launches(editor) == []


@pytest.mark.parametrize(
    "scenario",
    [
        {"interrupted": True},
        {"omit_final": True},
        {"saved": "Different output"},
        {"exit_code": 1},
        {"bad_json": True},
        {"usage": None},
    ],
)
def test_partial_or_inconsistent_result_is_never_delivered_or_relaunched(tmp_path, scenario):
    editor = make_editor(tmp_path, scenario)
    for _ in range(2):
        with pytest.raises(UncertainRequest):
            editor.complete("Task", request_key="one")
    with pytest.raises(UncertainRequest):
        editor.complete("New task", request_key="two")
    with pytest.raises(UncertainRequest):
        editor.new_conversation()
    assert len(launches(editor)) == 1


def test_resume_cannot_silently_start_another_thread(tmp_path):
    editor = make_editor(tmp_path, [{}, {"thread": OTHER_THREAD}])
    editor.complete("First", request_key="one")
    with pytest.raises(UncertainRequest):
        editor.complete("Second", request_key="two")
    assert json.loads(editor.path.read_text())["thread_id"] == THREAD


@pytest.mark.parametrize(
    "scenario",
    [
        {"error": "Unauthorized"},
        {"error": "Usage limit reached"},
        {"item": {"id": "tool", "type": "command_execution", "command": "echo forbidden"}},
        {"item": {"id": "warning", "type": "error", "message": "Model rerouted"}},
    ],
)
def test_provider_errors_and_native_or_rerouted_items_stop_the_conversation(tmp_path, scenario):
    editor = make_editor(tmp_path, scenario)
    for _ in range(2):
        with pytest.raises((ProviderUnavailable, UncertainRequest)):
            editor.complete("Task", request_key="one")
    with pytest.raises(EditorError):
        editor.complete("Another task", request_key="two")
    assert len(launches(editor)) == 1
    if editor.last_response is not None:
        assert editor.last_response["text"] == ""
        if "item" in scenario:
            assert editor.last_response["usage"] == USAGE
        else:
            assert editor.last_response["usage_snapshot"]["delta"] is None
    else:
        assert json.loads(editor.path.read_text())["pending"] is not None


def test_completed_formatting_runaway_is_saved_but_not_delivered(tmp_path):
    text = "Some answer\n" + "```````\n" * 128
    editor = make_editor(tmp_path, {"text": text})
    with pytest.raises(EditorRunaway) as raised:
        editor.complete("Task", request_key="one")
    assert raised.value.usage_pending is False
    assert editor.last_response["usage"] == USAGE and editor.last_response["text"] == ""
    assert (editor.root / editor.last_response["final_file"]).read_text() == text + "\n"
    with pytest.raises(EditorError, match="blocked"):
        editor.complete("Next", request_key="two")
    assert len(launches(editor)) == 1


def test_caller_can_refuse_semantic_output_without_partial_delivery(tmp_path):
    def validate(text):
        assert text == "Cannot comply"
        raise EditorRefused("Caller identified refusal")

    editor = make_editor(tmp_path, {"text": "Cannot comply"}, output_validator=validate)
    with pytest.raises(EditorRefused):
        editor.complete("Task", request_key="one")
    assert editor.last_response["status"] == "refused" and editor.last_response["text"] == ""


@pytest.mark.parametrize("crash_at", [2, 3])
def test_saved_terminal_evidence_recovers_after_commit_failure_without_launch(tmp_path, crash_at):
    commits = []

    def commit():
        commits.append(1)
        if len(commits) == crash_at:
            raise RuntimeError("simulated storage commit failure")

    editor = make_editor(tmp_path, commit=commit)
    with pytest.raises(RuntimeError):
        editor.complete("Task", request_key="one")
    recovered = make_editor(tmp_path)
    assert recovered.complete("Task", request_key="one") == "Final answer"
    assert len(launches(editor)) == 1


def test_saved_intent_before_launch_is_ambiguous_and_never_repeated(tmp_path):
    def commit():
        raise RuntimeError("simulated process death before dispatch")

    editor = make_editor(tmp_path, commit=commit)
    with pytest.raises(RuntimeError):
        editor.complete("Task", request_key="one")
    recovered = make_editor(tmp_path)
    with pytest.raises(UncertainRequest, match="no durable exit"):
        recovered.complete("Task", request_key="one")
    assert launches(editor) == []


def test_timeout_and_poll_failure_stop_child_and_keep_pending(tmp_path):
    polls = []
    editor = make_editor(tmp_path, {"sleep": 2}, max_wait_seconds=0.1, on_poll=lambda: polls.append(1))
    with pytest.raises(UncertainRequest, match="interrupted"):
        editor.complete("Task", request_key="one")
    assert len(polls) >= 2
    pending = json.loads(editor.path.read_text())["pending"]
    exit_record = json.loads((editor.root / pending["directory"] / "exit.json").read_text())
    assert exit_record["reason"] == "timeout" and exit_record["returncode"] != 0
    with pytest.raises(UncertainRequest):
        editor.complete("Task", request_key="one")
    assert len(launches(editor)) == 1


def test_poll_callback_failure_cannot_silently_continue_or_retry(tmp_path):
    calls = []

    def poll():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("external admission guard closed")

    editor = make_editor(tmp_path, {"sleep": 2}, on_poll=poll)
    with pytest.raises(RuntimeError, match="admission"):
        editor.complete("Task", request_key="one")
    with pytest.raises(UncertainRequest):
        editor.complete("Task", request_key="one")
    pending = json.loads(editor.path.read_text())["pending"]
    result = json.loads((editor.root / pending["directory"] / "exit.json").read_text())
    assert result["returncode"] != 0


@pytest.mark.parametrize("item_type", ["error", "command_execution", "mcp_tool_call"])
def test_live_native_or_reroute_event_stops_before_remaining_generation(tmp_path, item_type):
    editor = make_editor(
        tmp_path,
        {"item": {"id": "bad", "type": item_type, "message": "Model rerouted"}, "after_item_sleep": 2},
    )
    with pytest.raises(UncertainRequest):
        editor.complete("Task", request_key="one")
    pending = json.loads(editor.path.read_text())["pending"]
    directory = editor.root / pending["directory"]
    result = json.loads((directory / "exit.json").read_text())
    assert result["reason"] == "prohibited_event"
    assert result["violation"] == {"kind": "prohibited_item", "item_type": item_type}
    assert result["returncode"] != 0
    assert "turn.completed" not in (directory / "events.jsonl").read_text()
    with pytest.raises(UncertainRequest):
        editor.complete("Task", request_key="one")
    assert len(launches(editor)) == 1


def test_project_config_ancestor_is_refused_and_explicit_isolated_work_dir_is_supported(tmp_path):
    editor = make_editor(tmp_path)
    project_config = editor.root / ".codex" / "config.toml"
    project_config.parent.mkdir()
    project_config.write_text("untrusted project config must not be loaded")
    with pytest.raises(EditorError, match="Project Codex configuration"):
        editor.complete("Task", request_key="one")
    assert launches(editor) == []
    isolated = CodexEditor(
        editor.root,
        codex_home=editor.home,
        executable=editor.executable,
        work_dir=tmp_path / "isolated",
        poll_interval=0.01,
    )
    assert isolated.complete("Task", request_key="one") == "Final answer"
    assert launches(editor)[0]["cwd"] == str(tmp_path / "isolated")


def test_live_scanner_waits_for_complete_lines_without_losing_the_prefix(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"type":"item.completed","item":')
    assert CodexEditor._scan_live(path, 0, None) == (0, None)
    with path.open("ab") as stream:
        stream.write(b'{"type":"command_execution"}}\n')
    offset, violation = CodexEditor._scan_live(path, 0, None)
    assert offset == path.stat().st_size and violation["kind"] == "prohibited_item"


def test_zero_or_regressing_usage_never_claims_a_precise_per_turn_delta(tmp_path):
    zeros = {key: 0 for key in USAGE}
    assert _snapshot(zeros, USAGE, new_thread=False)["delta"] is None
    assert (
        _snapshot(USAGE, {key: value * 2 for key, value in USAGE.items()}, new_thread=False)["delta"] is None
    )
    editor = make_editor(
        tmp_path, [{}, {"usage": zeros}, {"usage": {key: value * 3 for key, value in USAGE.items()}}]
    )
    editor.complete("One", request_key="one")
    editor.complete("Two", request_key="two")
    assert editor.last_response["usage_snapshot"]["basis"] == "zero_or_unavailable_cli_usage"
    editor.complete("Three", request_key="three")
    assert editor.last_response["usage_snapshot"]["basis"] == "previous_snapshot_unavailable"
    assert editor.last_response["usage_snapshot"]["delta"] is None


def test_counter_regression_invalidates_the_next_delta_baseline(tmp_path):
    editor = make_editor(
        tmp_path,
        [
            {"usage": {key: value * 2 for key, value in USAGE.items()}},
            {"usage": USAGE},
            {"usage": {key: value * 3 for key, value in USAGE.items()}},
        ],
    )
    editor.complete("One", request_key="one")
    editor.complete("Two", request_key="two")
    assert editor.last_response["usage_snapshot"]["basis"] == "cumulative_counter_regression"
    editor.complete("Three", request_key="three")
    assert editor.last_response["usage_snapshot"]["basis"] == "previous_snapshot_unavailable"
    assert editor.last_response["usage_snapshot"]["delta"] is None


def test_pin_and_private_directory_checks_fail_before_dispatch(tmp_path):
    editor = make_editor(tmp_path)
    bad_catalog = tmp_path / "catalog.json"
    bad_catalog.write_text("{}")
    editor.catalog = bad_catalog
    with pytest.raises(EditorError, match="catalog"):
        editor.complete("Task")
    assert launches(editor) == []
    editor.home.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        make_editor(tmp_path)


def test_cli_version_drift_fails_before_dispatch(tmp_path):
    editor = make_editor(tmp_path)
    executable = Path(editor.executable)
    executable.write_text(executable.read_text().replace("codex-cli 0.163.0", "codex-cli 0.164.0"))
    with pytest.raises(EditorError, match="requires codex-cli"):
        editor.complete("Task")
    assert launches(editor) == []


def test_journal_directories_and_sensitive_files_remain_private(tmp_path):
    editor = make_editor(tmp_path)
    editor.complete("Private input", request_key="one")
    assert editor.root.stat().st_mode & 0o077 == 0
    for name in (
        "prompt.txt",
        "instructions.txt",
        "events.jsonl",
        "stderr.txt",
        "final.txt",
        "request.json",
        "receipt.json",
    ):
        path = next(editor.root.glob(f"*/{name}"))
        assert path.stat().st_mode & 0o077 == 0
        assert path.parent.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize(
    "value",
    [
        "",
        [],
        [{"role": "assistant", "content": "old turn"}],
        [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
    ],
)
def test_input_cannot_reconstruct_or_silently_change_cli_conversation(tmp_path, value):
    editor = make_editor(tmp_path)
    with pytest.raises(ValueError):
        editor.complete(value)
    assert launches(editor) == []
