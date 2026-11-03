"""Provider/model selection stays explicit and independent of experiment code."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from test_codex_editor import config, launches, make_editor

from openwsp.budget import BudgetLedger
from openwsp.editors.anthropic import AnthropicEditor, AnthropicTokenRates
from openwsp.editors.base import EditorError
from openwsp.editors.openai import AstraEditor
from openwsp.editors.profiles import (
    ASTRA_CODEX_PROFILE,
    ASTRA_RESPONSES_PROFILE,
    FABLE_5_1_PROFILE,
    FABLE_OPENROUTER_MODEL_ID,
    EditorProfile,
    EditorRegistry,
    ThinkingConfig,
    create_editor,
    default_registry,
)


def test_default_codex_profile_preserves_the_deployed_state_identity(tmp_path):
    editor = make_editor(tmp_path)
    assert editor.settings == {
        "cli_version": "0.163.0",
        "model": "gpt-6-astra",
        "effort": "max",
        "verbosity": "low",
        "catalog_sha256": "05723cac0f7da85c31f710f602bb0c47a5588d8bdf68aad24e9dd6ac951c7705",
        "codex_home": str(editor.home),
        "work_dir": str(editor.work),
        "instructions_sha256": hashlib.sha256(b"Answer the user's request.\n").hexdigest(),
    }
    editor.complete("Existing turn", request_key="one")
    # Existing receipts/intents predate the descriptive profile metadata.
    for path in [next(editor.root.glob("*/request.json")), next(editor.root.glob("*/receipt.json"))]:
        record = json.loads(path.read_text())
        record.pop("profile")
        path.write_text(json.dumps(record))
    restarted = make_editor(tmp_path, profile=ASTRA_CODEX_PROFILE)
    assert restarted.has_request("one") is True
    assert restarted.complete("Existing turn", request_key="one") == "Final answer"
    assert len(launches(editor)) == 1


def test_explicit_codex_profile_drives_model_catalog_version_and_generation_settings(tmp_path):
    catalog = tmp_path / "different-model.json"
    catalog.write_text('{"models": [{"slug": "caller-verified-model"}]}')
    profile = replace(
        ASTRA_CODEX_PROFILE,
        name="verified-test-profile",
        model="caller-verified-model",
        cli_version="0.164.0",
        catalog_path=catalog,
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        effort="high",
        verbosity="medium",
        instructions="Use the supplied evidence.\n",
        context_window=100000,
        auto_compact_token_limit=120000,
    )
    editor = make_editor(tmp_path, profile=profile)
    executable = tmp_path / "fake-codex"
    executable.write_text(executable.read_text().replace("codex-cli 0.163.0", "codex-cli 0.164.0"))
    assert editor.complete("Task", request_key="one") == "Final answer"
    argv = launches(editor)[0]["argv"]
    options = config(argv)
    assert argv[argv.index("-m") + 1] == "caller-verified-model"
    assert options["model_catalog_json"] == str(catalog)
    assert options["model_reasoning_effort"] == "high" and options["model_verbosity"] == "medium"
    assert options["model_context_window"] == 100000 and options["model_auto_compact_token_limit"] == 120000
    assert options["forced_login_method"] == "chatgpt" and options["features.shell_tool"] is False
    assert next(editor.root.glob("*/instructions.txt")).read_text() == profile.instructions
    assert editor.last_response["profile"] == profile.record()
    assert editor.settings["context_window"] == 100000


def test_changed_profile_cannot_silently_continue_existing_conversation(tmp_path):
    editor = make_editor(tmp_path)
    editor.complete("First", request_key="one")
    changed = make_editor(tmp_path, profile=replace(ASTRA_CODEX_PROFILE, effort="high"))
    with pytest.raises(EditorError, match="explicit new_conversation"):
        changed.complete("Second", request_key="two")
    assert len(launches(editor)) == 1


def test_explicit_reasoning_summary_is_bound_without_changing_default_identity(tmp_path):
    assert "reasoning_summary" not in ASTRA_CODEX_PROFILE.record()
    plain = make_editor(tmp_path)
    assert "model_reasoning_summary" not in config(plain._command(plain.root / "request", None))
    profile = replace(ASTRA_CODEX_PROFILE, effort="high", reasoning_summary="detailed")
    editor = make_editor(tmp_path, profile=profile)
    assert editor.complete("Task", request_key="one") == "Final answer"
    assert config(launches(editor)[0]["argv"])["model_reasoning_summary"] == "detailed"
    assert editor.settings["reasoning_summary"] == "detailed"
    changed = make_editor(tmp_path, profile=replace(profile, reasoning_summary=None))
    with pytest.raises(EditorError, match="explicit new_conversation"):
        changed.complete("Another task", request_key="two")
    assert len(launches(editor)) == 1


def test_codex_profile_cannot_select_a_model_absent_from_its_verified_catalog(tmp_path):
    editor = make_editor(tmp_path, profile=replace(ASTRA_CODEX_PROFILE, model="not-in-catalog"))
    with pytest.raises(EditorError, match="selected profile model"):
        editor.complete("Task", request_key="one")
    assert launches(editor) == []


def test_profiles_and_nested_thinking_are_immutable(tmp_path):
    with pytest.raises(FrozenInstanceError):
        ASTRA_CODEX_PROFILE.model = "changed"
    thinking = ThinkingConfig(type="enabled", budget_tokens=1024)
    with pytest.raises(FrozenInstanceError):
        thinking.budget_tokens = 2048
    request = thinking.request()
    request["budget_tokens"] = 5000
    assert thinking.budget_tokens == 1024
    editor = make_editor(tmp_path)
    with pytest.raises(AttributeError):
        editor.profile = replace(ASTRA_CODEX_PROFILE, effort="low")


def test_omitted_codex_verbosity_is_an_explicit_profile_choice(tmp_path):
    editor = make_editor(tmp_path, profile=replace(ASTRA_CODEX_PROFILE, verbosity=None))
    assert "model_verbosity" not in config(editor._command(editor.root / "example", None))


def test_factory_preserves_the_existing_astra_responses_implementation(tmp_path):
    budget = BudgetLedger(tmp_path / "budget.json")
    client = SimpleNamespace()
    profile = replace(ASTRA_RESPONSES_PROFILE, effort="high", verbosity="low", max_output_tokens=2048)
    editor = create_editor(profile, state_dir=tmp_path / "api", budget=budget, client=client)
    assert isinstance(editor, AstraEditor)
    assert editor.client is client and editor.budget is budget
    assert editor.model == "gpt-6-astra" and editor.effort == "high"
    assert editor.verbosity == "low" and editor.max_output_tokens == 2048


def test_responses_factory_profile_binding_is_immutable(tmp_path):
    options = {
        "state_dir": tmp_path / "api",
        "budget": BudgetLedger(tmp_path / "budget.json"),
        "client": SimpleNamespace(),
    }
    first = create_editor(ASTRA_RESPONSES_PROFILE, **options)
    binding = (first.root / "responses_profile.json").read_bytes()
    assert isinstance(create_editor(ASTRA_RESPONSES_PROFILE, **options), AstraEditor)
    with pytest.raises(EditorError, match="profile binding differs"):
        create_editor(replace(ASTRA_RESPONSES_PROFILE, effort="high"), **options)
    with pytest.raises(EditorError, match="profile binding differs"):
        create_editor(replace(ASTRA_RESPONSES_PROFILE, max_output_tokens=1024), **options)
    assert (first.root / "responses_profile.json").read_bytes() == binding


def test_responses_factory_does_not_adopt_or_modify_unbound_legacy_state(tmp_path):
    budget = BudgetLedger(tmp_path / "budget.json")
    legacy = AstraEditor(tmp_path / "api", budget, client=SimpleNamespace())
    legacy.new_conversation()
    original = legacy.path.read_bytes()
    with pytest.raises(EditorError, match="no profile binding"):
        create_editor(ASTRA_RESPONSES_PROFILE, state_dir=legacy.root, budget=budget, client=SimpleNamespace())
    assert legacy.path.read_bytes() == original
    assert not (legacy.root / "responses_profile.json").exists()
    # The original direct constructor continues to accept its existing state.
    direct = AstraEditor(legacy.root, budget, client=SimpleNamespace())
    with direct._locked_state() as state:
        assert state["conversation_id"] == json.loads(original)["conversation_id"]


def test_responses_factory_refuses_orphaned_unbound_api_request_records(tmp_path):
    root = tmp_path / "api"
    root.mkdir()
    (root / "request_orphan.json").write_text("{}")
    with pytest.raises(EditorError, match="no profile binding"):
        create_editor(
            ASTRA_RESPONSES_PROFILE,
            state_dir=root,
            budget=BudgetLedger(tmp_path / "budget.json"),
            client=SimpleNamespace(),
        )
    assert not (root / "responses_profile.json").exists()


def test_fable_native_profile_requires_caller_prices_and_bounds(tmp_path):
    assert FABLE_5_1_PROFILE.model == "claude-fable-5-1"
    assert FABLE_OPENROUTER_MODEL_ID == "anthropic/claude-fable-5.1"
    assert FABLE_5_1_PROFILE.model != FABLE_OPENROUTER_MODEL_ID
    assert not hasattr(FABLE_5_1_PROFILE, "rates")
    with pytest.raises(TypeError, match="rates"):
        create_editor(
            FABLE_5_1_PROFILE, state_dir=tmp_path / "native", budget=BudgetLedger(tmp_path / "budget.json")
        )
    rates = AnthropicTokenRates(
        input_per_million=1,
        output_per_million=2,
        cache_read_per_million=1,
        cache_creation_5m_per_million=1,
        cache_creation_1h_per_million=1,
    )
    profile = replace(FABLE_5_1_PROFILE, thinking=ThinkingConfig(type="adaptive"), output_effort="max")
    editor = create_editor(
        profile,
        state_dir=tmp_path / "native",
        budget=BudgetLedger(tmp_path / "budget.json"),
        rates=rates,
        max_input_tokens=2000,
        max_output_tokens=2000,
        client=SimpleNamespace(),
    )
    assert isinstance(editor, AnthropicEditor)
    assert editor.settings["model"] == "claude-fable-5-1"
    assert editor.settings["thinking"] == {"type": "adaptive"}
    assert editor.settings["output_config"] == {"effort": "max"}
    assert editor.settings["rates"] == rates.record()


def test_new_transport_registration_needs_no_experiment_switch_or_fallback():
    registry = EditorRegistry()
    calls = []

    def build(profile, **options):
        calls.append((profile, options))
        return SimpleNamespace(profile=profile, options=options)

    profile = EditorProfile(name="future-router", transport="future_router", model=FABLE_OPENROUTER_MODEL_ID)
    with pytest.raises(ValueError, match="No editor transport"):
        registry.create(profile)
    registry.register("future_router", build)
    result = registry.create(profile, caller_verified_options="explicit")
    assert result.profile is profile and calls == [(profile, {"caller_verified_options": "explicit"})]
    with pytest.raises(ValueError, match="already registered"):
        registry.register("future_router", build)
    with pytest.raises(ValueError, match="No editor transport"):
        create_editor(profile)


def test_wrong_transport_profile_type_is_rejected_before_client_construction():
    profile = EditorProfile(name="invalid", transport="codex_cli_chatgpt", model="some-model")
    with pytest.raises(TypeError, match="CodexProfile"):
        default_registry().create(profile)


@pytest.mark.parametrize(
    "changes",
    [
        {"catalog_sha256": "unverified"},
        {"instructions": "one\ntwo"},
        {"context_window": True},
        {"auto_compact_token_limit": 1},
        {"reasoning_summary": "unsupported"},
    ],
)
def test_invalid_codex_profile_identity_or_limits_are_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(ASTRA_CODEX_PROFILE, **changes)


@pytest.mark.parametrize("effort", ["extreme", "", 1])
def test_undocumented_anthropic_effort_is_rejected(effort):
    with pytest.raises(ValueError, match="output effort"):
        replace(FABLE_5_1_PROFILE, output_effort=effort)
