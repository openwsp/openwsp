# Editor profiles and transports

An editor profile makes the model and transport an explicit choice. `create_editor(profile, **options)` constructs that transport; it does not try another provider, credential source, or billing account after a failure. The adapters return text and retain conversation state. They do not execute generated code or expose model-native tools.

| Profile | Transport | Model identity | Billing configuration |
| --- | --- | --- | --- |
| `ASTRA_CODEX_PROFILE` | `codex_cli_chatgpt` | `gpt-6-astra` | Caller-provisioned ChatGPT subscription; no API price inferred |
| `ASTRA_RESPONSES_PROFILE` | `astra_responses` | `gpt-6-astra` | Existing budgeted Responses adapter and its Astra billing schedule |
| `FABLE_5_1_PROFILE` | `anthropic_messages` | `claude-fable-5-1` | Caller supplies all token rates and request bounds |

The primary subscription route is an explicit `ASTRA_CODEX_PROFILE` selection. The Responses route remains available as a separate, explicitly selected API transport. Subscription failure never authorizes an API fallback. Profiles do not establish provider entitlement, model availability, or permission to spend.

```python
from openwsp.editors.profiles import ASTRA_CODEX_PROFILE, create_editor

editor = create_editor(
    ASTRA_CODEX_PROFILE,
    state_dir=private_state_directory,
    codex_home=caller_provisioned_private_codex_home,
    work_dir=isolated_private_work_directory,
)
reply = editor.complete(new_user_text, request_key="turn-001")
```

The caller provisions the normal CLI login in a private writable `CODEX_HOME`; the adapter does not read or copy its tokens. The profile pins CLI version, catalog digest, model, effort, verbosity, neutral instructions, context window, and compaction setting. The standard CLI is forced to its ChatGPT route. Its own transport retries still apply: the adapter guarantees one CLI launch per durable request, not zero internal network retries. Tool suppression is verified for the pinned configuration and inspected deployment; it is not an operating-system isolation guarantee. See [the Codex adapter contract](codex-editor.md).

`CodexEditor(..., profile=ASTRA_CODEX_PROFILE)` remains a direct construction option. Omitting `profile` preserves that same default and its existing saved-state identity. Earlier request journals and receipts without descriptive profile metadata remain readable. Changing an effective Codex setting requires an explicit `new_conversation()` or a separate state directory; a pending request prevents handoff.

The retained API route is constructed separately:

```python
from openwsp.editors.profiles import ASTRA_RESPONSES_PROFILE, create_editor

api_editor = create_editor(
    ASTRA_RESPONSES_PROFILE,
    state_dir=api_state_directory,
    budget=authorized_api_budget_ledger,
)
```

This returns the existing `AstraEditor`; it does not migrate a Codex thread or share its request journal. The factory binds a new Responses state directory to its immutable profile and rejects a different profile for that directory. It also refuses to adopt an existing API conversation without that binding. Use a separate state directory for a different factory profile. Direct `AstraEditor` construction remains backward compatible with existing unbound API conversations and retains its original recovery and negotiated-effort behavior. See [editor recovery](editor-recovery.md).

## Native Fable configuration

`FABLE_5_1_PROFILE` selects the native Anthropic Messages model identifier `claude-fable-5-1`. `FABLE_OPENROUTER_MODEL_ID` is the distinct reference string `anthropic/claude-fable-5.1`; no OpenRouter transport is implemented or registered. Neither identifier establishes the other's prices, context limits, available options, or authentication route.

```python
from dataclasses import replace

from openwsp.editors.anthropic import AnthropicTokenRates
from openwsp.editors.profiles import FABLE_5_1_PROFILE, ThinkingConfig, create_editor

rates = AnthropicTokenRates(
    input_per_million=verified_input_rate,
    output_per_million=verified_output_rate,
    cache_read_per_million=verified_cache_read_rate,
    cache_creation_5m_per_million=verified_cache_write_5m_rate,
    cache_creation_1h_per_million=verified_cache_write_1h_rate,
)
profile = replace(
    FABLE_5_1_PROFILE,
    thinking=ThinkingConfig(type="adaptive"),
    output_effort="max",
)
native_editor = create_editor(
    profile,
    state_dir=native_state_directory,
    budget=authorized_native_budget_ledger,
    rates=rates,
    max_input_tokens=verified_entire_request_input_ceiling,
    max_output_tokens=verified_output_limit,
)
```

All `verified_*` variables are supplied by the caller. This example quotes no prices or limits and grants no spending allowance. All five rate fields are required, finite, nonnegative USD-per-million-token values. The input ceiling covers the entire request, including saved history and system context; it is an admission-budget bound, not a history truncation parameter. The native adapter reserves against the supplied bounds before a possible paid request.

The default Fable profile omits both thinking and output effort. `ThinkingConfig` accepts `disabled`, `adaptive`, or `enabled` with `budget_tokens`; enabled thinking requires at least 1,024 tokens and, when constructing the adapter, less than the output limit. Optional `output_effort` accepts `low`, `medium`, `high`, `xhigh`, or `max` and is sent as `output_config={"effort": ...}`. These are supported request shapes in the adapter and SDK, not proof that this model accepts every combination. The caller must verify model support before selecting them. The factory never retries rejected options with weaker settings.

Fable has not been validated on the Experiment 8 benchmark. The profile, fake-client serialization tests, and generic adapter support do not establish benchmark performance or a successful live model call. Native requests retain `tools=[]` and `service_tier="standard_only"`. No native or OpenRouter price, context ceiling, or output ceiling is built into the Fable profile.

The [Anthropic adapter contract](anthropic-editor.md) governs signed thinking blocks, exact conversation-prefix replay, separate cache accounting, and uncertain POST recovery. Explicit effort is persisted with the other settings and cannot change mid-conversation. Missing cache-write TTL details with unequal configured TTL rates leave billing unresolved; no charge is guessed. An uncertain request retains its reservation and is never automatically posted again.

## Immutable configuration and extensions

`EditorProfile`, `CodexProfile`, `AstraResponsesProfile`, `AnthropicMessagesProfile`, and nested `ThinkingConfig` are frozen dataclasses. Use `dataclasses.replace` to create another configuration; do not mutate an active profile or its saved state. `profile.record()` produces a separate JSON-compatible description. Profiles hold configuration, while each transport owns authentication, state, usage accounting, and recovery.

`create_editor` uses the module's `EDITOR_REGISTRY`. `EditorRegistry()` creates an empty registry; `default_registry()` creates a fresh registry with the three built-in transports. Register another transport through `registry.register(transport, builder)` or the module-level `register_editor_transport(transport, builder)`. A builder receives the profile as its first argument and caller options as keyword arguments. Unknown transports and duplicate registrations are errors.

```python
from openwsp.editors.profiles import EditorProfile, EditorRegistry

registry = EditorRegistry()
registry.register("my_transport", build_my_editor)
profile = EditorProfile(name="caller-verified-model", transport="my_transport", model=model_id)
editor = registry.create(profile, state_dir=private_state_directory)
```

The extension point keeps provider selection out of experiment-specific control flow. An extension must implement the editor interface and define its own verified billing and durable-recovery contract; registration supplies neither. Keep state directories separate across transports. Stable request keys identify intended turns, but recovery remains adapter-specific: profiles do not convert an unknown outcome into permission to generate again.
