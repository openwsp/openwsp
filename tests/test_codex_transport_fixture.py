"""Offline integrity and request checks; never execute Codex or open a socket."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def fixture_module():
    path = Path(__file__).resolve().parents[1] / "scripts/check_codex_transport.py"
    spec = importlib.util.spec_from_file_location("check_codex_transport_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profile_files(tmp_path, fixture_module):
    catalog = tmp_path / "catalog.json"
    provenance = tmp_path / "provenance.json"
    catalog.write_bytes(fixture_module.DEFAULT_CATALOG.read_bytes())
    provenance.write_bytes(fixture_module.DEFAULT_PROVENANCE.read_bytes())
    return catalog, provenance


@pytest.fixture
def request_body(fixture_module):
    return {
        "model": "gpt-6-astra",
        "reasoning": {"effort": "max", "context": "all_turns"},
        "text": {"verbosity": "low"},
        "tools": [],
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": fixture_module.INSTRUCTIONS}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": fixture_module.PROMPT}],
            },
        ],
    }


def test_profile_retains_native_metadata_except_three_audited_tool_fields(fixture_module):
    catalog = fixture_module.load_profile()
    provenance = json.loads(fixture_module.DEFAULT_PROVENANCE.read_text())
    assert len(catalog["models"]) == 1
    model = catalog["models"][0]
    assert model["slug"] == "gpt-6-astra"
    assert model["tool_mode"] == "direct"
    assert model["apply_patch_tool_type"] is None
    assert model["experimental_supported_tools"] == ["send_user_message_async"]
    original = copy.deepcopy(model)
    original.update(
        tool_mode="code_mode_only",
        apply_patch_tool_type="freeform",
        experimental_supported_tools=["send_user_message_async", "clock"],
    )
    assert (
        hashlib.sha256(fixture_module.canonical_json(original)).hexdigest()
        == provenance["source_model_canonical_sha256"]
    )
    assert provenance["cli_version"] == "0.163.0"
    assert set(provenance["changes"]) == {
        "tool_mode",
        "apply_patch_tool_type",
        "experimental_supported_tools",
    }


def test_canonical_json_is_stable_and_preserves_unicode(fixture_module):
    assert fixture_module.canonical_json({"z": "λ", "a": [1, False]}) == ('{"a":[1,false],"z":"λ"}'.encode())


def test_catalog_tampering_rejected_even_with_matching_manifest_hash(profile_files, fixture_module):
    catalog_path, provenance_path = profile_files
    catalog = json.loads(catalog_path.read_text())
    catalog["models"][0]["experimental_supported_tools"].append("clock")
    data = fixture_module.canonical_json(catalog)
    catalog_path.write_bytes(data)
    provenance = json.loads(provenance_path.read_text())
    provenance["catalog_sha256"] = hashlib.sha256(data).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="Pinned catalog checksum"):
        fixture_module.load_profile(*profile_files)


def test_recomputed_source_hash_cannot_authorize_changed_original(profile_files, fixture_module):
    catalog_path, provenance_path = profile_files
    catalog = json.loads(catalog_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    provenance["changes"]["tool_mode"]["before"] = "unverified_original"
    fake_original = copy.deepcopy(catalog["models"][0])
    for name, change in provenance["changes"].items():
        fake_original[name] = change["before"]
    provenance["source_model_canonical_sha256"] = hashlib.sha256(
        fixture_module.canonical_json(fake_original)
    ).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="(?i)source.*checksum"):
        fixture_module.load_profile(*profile_files)


@pytest.mark.parametrize(
    "case",
    ["version", "catalog_checksum", "source_checksum", "extra_change", "wrong_after", "wrong_before"],
)
def test_invalid_provenance_rejected(profile_files, fixture_module, case):
    _, provenance_path = profile_files
    provenance = json.loads(provenance_path.read_text())
    if case == "version":
        provenance["cli_version"] = "0.164.0"
    elif case == "catalog_checksum":
        provenance["catalog_sha256"] = "0" * 64
    elif case == "source_checksum":
        provenance["source_model_canonical_sha256"] = "0" * 64
    elif case == "extra_change":
        provenance["changes"]["shell_type"] = {"before": "unified_exec", "after": "unified_exec"}
    elif case == "wrong_after":
        provenance["changes"]["tool_mode"]["after"] = "code_mode_only"
    else:
        provenance["changes"]["tool_mode"]["before"] = "unverified_original"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        fixture_module.load_profile(*profile_files)


def test_config_disables_tools_and_added_context_without_selecting_auth(tmp_path, fixture_module):
    instructions = tmp_path / "instructions.md"
    catalog = tmp_path / "catalog.json"
    config = fixture_module.build_config(instructions, catalog)
    assert config["model_instructions_file"] == str(instructions.resolve())
    assert config["model_catalog_json"] == str(catalog.resolve())
    assert config["model_reasoning_effort"] == "max"
    assert config["model_verbosity"] == "low"
    assert config["project_doc_max_bytes"] == 0
    assert config["instructions"] == config["developer_instructions"] == ""
    assert config["web_search"] == "disabled"
    for key in (
        "skills.include_instructions",
        "skills.bundled.enabled",
        "include_permissions_instructions",
        "include_apps_instructions",
        "include_collaboration_mode_instructions",
        "include_environment_context",
        "include_environment_context_time",
        "agents.enabled",
        "tools.experimental_request_user_input.enabled",
        "tools.update_plan.enabled",
        "features.shell_tool",
        "features.unified_exec",
        "features.code_mode",
        "features.view_image",
        "features.multi_agent",
        "features.apps",
        "features.plugins",
    ):
        assert config[key] is False, key
    assert "model_provider" not in config
    assert not any("auth" in key or "api_key" in key for key in config)


def test_expected_request_accepts_provider_ids_without_changing_content(request_body, fixture_module):
    expected = copy.deepcopy(request_body["input"])
    for index, item in enumerate(request_body["input"]):
        item["id"] = f"item_{index}"
    report = fixture_module.assert_request(request_body, expected)
    assert report["input_roles"] == ["developer", "user"]
    assert report["model"] == "gpt-6-astra"
    assert report["max_output_tokens_present"] is False
    assert report["reasoning"] == {"effort": "max", "context": "all_turns"}


@pytest.mark.parametrize(
    "kind",
    [
        "top_level_tool",
        "additional_tools",
        "top_level_instruction",
        "developer",
        "system",
        "changed_instruction",
        "changed_user_input",
    ],
)
def test_extra_tools_or_instructions_rejected(request_body, fixture_module, kind):
    expected = copy.deepcopy(request_body["input"])
    if kind == "top_level_tool":
        request_body["tools"] = [{"type": "function", "name": "exec_command"}]
    elif kind == "additional_tools":
        request_body["input"].append({"type": "additional_tools", "tools": []})
    elif kind == "top_level_instruction":
        request_body["instructions"] = "Extra instructions"
    elif kind in ("developer", "system"):
        request_body["input"].append(
            {
                "type": "message",
                "role": kind,
                "content": [{"type": "input_text", "text": "Extra instructions"}],
            }
        )
    elif kind == "changed_instruction":
        request_body["input"][0]["content"][0]["text"] += "\nExtra instructions"
    else:
        request_body["input"][1]["content"][0]["text"] += "\nExtra context"
    with pytest.raises(AssertionError):
        fixture_module.assert_request(request_body, expected)


@pytest.mark.parametrize(
    "key,value",
    [
        ("model", "another-model"),
        ("reasoning", {"effort": "high", "context": "all_turns"}),
        ("reasoning", {"effort": "max", "context": "current_turn"}),
        ("text", {"verbosity": "high"}),
    ],
)
def test_model_or_reasoning_drift_rejected(request_body, fixture_module, key, value):
    request_body[key] = value
    with pytest.raises(AssertionError):
        fixture_module.assert_request(request_body)


def test_saved_two_turn_capture_preserves_wire_contract_and_cumulative_usage(fixture_module):
    directory = Path(__file__).parent / "fixtures/codex_transport_0_163_0"
    captures = json.loads((directory / "captures.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert len(captures) == 2
    assert summary["cli_version"] == "0.163.0"
    assert summary["real_provider_calls"] == 0
    assert summary["resume_requested"] is True
    observed_usage = []
    thread_ids = []
    for index, capture in enumerate(captures):
        assert capture["path"] == "/v1/responses"
        assert capture["fake_auth_matches"] is True
        report = fixture_module.assert_request(capture["body"])
        assert report == summary["requests"][index]
        assert report["tools_present"] is False
        assert report["store"] is False
        assert report["include"] == ["reasoning.encrypted_content"]
        events = [json.loads(line) for line in (directory / f"stdout-{index}.jsonl").read_text().splitlines()]
        started = [event["thread_id"] for event in events if event["type"] == "thread.started"]
        completed = [event["usage"] for event in events if event["type"] == "turn.completed"]
        assert len(started) == len(completed) == 1
        thread_ids.extend(started)
        observed_usage.extend(completed)
    assert thread_ids == [summary["thread_id"]] * 2
    assert observed_usage == summary["cli_turn_usage"]
    assert [(usage["input_tokens"], usage["output_tokens"]) for usage in observed_usage] == [(10, 2), (20, 4)]
    assert summary["mock_usage_per_response"] == {"input_tokens": 10, "output_tokens": 2}
    assert captures[0]["body"]["input"] == captures[1]["body"]["input"][:2]
    assert [item["role"] for item in captures[1]["body"]["input"]] == [
        "developer",
        "user",
        "assistant",
        "user",
    ]
