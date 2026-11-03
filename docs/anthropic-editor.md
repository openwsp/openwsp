# Anthropic editor adapter

`openwsp.editors.anthropic.AnthropicEditor` implements the generic editor interface with Anthropic's Messages API. Experiment 8 continues to use its prescribed Astra adapter; this adapter does not select a different editor for that replication.

## Configuration

The caller supplies the model, output limit, verified input-token ceiling, and billing rates. There is no built-in model price table. Install the optional SDK with `uv sync --locked --extra anthropic` (include any other extras you use), then provide `ANTHROPIC_API_KEY` through the environment. Offline tests inject a compatible client. The lockfile pins the SDK; no Anthropic call was made for this release.

```python
from openwsp.budget import BudgetLedger
from openwsp.editors.anthropic import AnthropicEditor, AnthropicTokenRates

# Supply these values from the selected model's current pricing and limits.
rates = AnthropicTokenRates(
    input_per_million=verified_input_rate,
    output_per_million=verified_output_rate,
    cache_read_per_million=verified_cache_read_rate,
    cache_creation_5m_per_million=verified_cache_write_5m_rate,
    cache_creation_1h_per_million=verified_cache_write_1h_rate,
)
editor = AnthropicEditor(
    state_dir="/tmp/openwsp-anthropic/editor",
    budget=BudgetLedger("/tmp/openwsp-anthropic/budget.json", editor_limit=authorized_editor_budget),
    model=verified_model_id,
    rates=rates,
    max_output_tokens=verified_output_limit,
    max_input_tokens=verified_input_ceiling,
    thinking={"type": "disabled"},
    system="Answer the user's request.",
)
reply = editor.complete("Inspect the supplied weight statistics.", request_key="turn-001")
```

The example's `verified_*` variables and `authorized_editor_budget` are caller-provided values, not quoted prices, recommended model limits, or a new spending allowance. All five rate fields are required, accept Decimal-compatible values, and must be finite and nonnegative. Rates are USD per million tokens. Verify applicable model/version, context-length pricing bands, and inference-region charges; the adapter cannot establish that a caller's rate table is current or sufficient.

`max_input_tokens` is a caller-verified ceiling for the entire request, including prior conversation and system context. It is used for admission budgeting, not as an Anthropic request parameter or a promise to truncate history. Before a possible paid POST, the adapter reserves:

```text
(max_input_tokens × highest configured input/cache rate
 + max_output_tokens × configured output rate) / 1,000,000
```

`thinking=None` omits the thinking setting and leaves the provider's model-specific default in effect. Explicit settings are `{"type": "disabled"}`, `{"type": "adaptive"}`, or `{"type": "enabled", "budget_tokens": ...}`. Manual thinking requires a budget of at least 1,024 and less than `max_output_tokens`; thinking consumes the output-token allowance. Model support varies, so select a compatible model and setting. Rejected settings are recorded without silently changing the requested behavior. This adapter does not expose adaptive-effort overrides.

Requests use `service_tier="standard_only"` and `tools=[]`. The adapter does not expose prompt-cache controls, priority or fast-mode selection, built-in tools, or tool execution. Provider-reported cache usage is still accounted for.

## Conversation and recovery

`complete(messages, request_key=...)` accepts new text input through the shared editor interface. The adapter stores the full conversation locally and sends the earlier user and assistant messages on subsequent calls. Only text content is returned to the driver. The raw provider response and complete assistant content blocks are retained separately, including `thinking` signatures and opaque `redacted_thinking` data.

Thinking blocks must be replayed unchanged and in their original order. Current models may bind signatures to the system prompt, tools, and earlier conversation as well. Do not edit persisted history, remove signatures, reconstruct thinking from displayed text, or change request configuration during a conversation. Configuration changes require an explicit `new_conversation()` or a separate state directory.

Use a stable, unique request key for each intended turn. Repeating a completed key with the same input returns its saved outcome without another provider request; changing the input under that key is refused. `new_conversation()` clears conversation history while retaining completed-request receipts, so use new keys for the new conversation. A handoff is refused while a request remains unresolved.

The default client disables automatic POST retries. A request intent and budget reservation are saved before dispatch. If the provider may have received a POST but no durable response is available, the adapter raises `UncertainRequest` and retains the reservation. It does not claim provider-side idempotency or recover a Messages response through a GET endpoint. Resolve an uncertain request explicitly before continuing; deleting its intent and blindly retrying can duplicate usage.

`resolve_pending(response, request_key=...)` accepts an authentic, complete provider response obtained and checked by the operator. The operator must establish that the response belongs to the pending request; the adapter does not verify that association externally. The request key must match, and a previously recorded response ID must also match. This method can supply missing usage or TTL details from verified provider evidence without sending another Messages request. It is not permission to invent usage, edit signed content, or substitute an unrelated response.

The `commit` callback lets a caller make saved intent, response, state, and ledger changes durable on its storage system. `timeout` bounds the SDK request and defaults to 900 seconds. A timeout does not establish that generation stopped or that nothing was billed.

## Outcomes and usage

A nonempty text response ending with `end_turn` is deliverable. `refusal` is recorded and surfaced as `EditorRefused`; output or context limits are surfaced as `EditorIncomplete`. Other stop reasons, including `tool_use`, `pause_turn`, and unexpected stop sequences, are not silently accepted as completed editor text. Malformed or unsupported content, other error outcomes, and `pause_turn` block further turns until an explicit `new_conversation()`; their raw responses and content remain saved unchanged. Terminal usage and response evidence are retained even when the adapter refuses delivery. Empty text is not proof of zero usage.

Anthropic's `input_tokens` excludes `cache_creation_input_tokens` and `cache_read_input_tokens`. These are separate input categories. Cache-creation TTL details, when present, split cache writes between `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`; they are not an additional charge on top of the creation total. If positive cache-creation usage has no TTL breakdown and the two configured creation rates differ, the raw response is saved, the reservation remains held, and explicit billing resolution is required. No TTL or charge is guessed. Zero cache-creation usage or equal creation rates are unambiguous without a TTL split.

`output_tokens` is the inclusive billed output total, including thinking; do not add reasoning or other output-detail counters again. The complete usage object is retained, including provider-reported service tier and speed when supplied. An unsupported billing mode or missing usable receipt requires investigation rather than treating the response as free. A financial reservation overrun is recorded and halts the budget ledger. Usage exceeding a configured token ceiling blocks the conversation; it is not silently accepted under the original limits.

For offline integration tests, pass `client=...` with a fake `messages.create` implementation and a temporary ledger/state directory. Test terminal stop reasons, cache-token receipts, exact content-block replay, completed-key reuse, and lost POST replies without calling Anthropic. Injected clients must preserve the adapter's no-automatic-retry behavior.

## Provider contract sources

The rules above were checked against Anthropic's official generated SDK types and examples. SDK source links below are pinned to commit `5752c132041af71055584632caab5ca2325cae5e`; model availability and prices still need separate verification when configuring a real run.

- [Messages request fields and thinking configuration](https://github.com/anthropics/anthropic-sdk-python/blob/5752c132041af71055584632caab5ca2325cae5e/src/anthropic/types/message_create_params.py)
- [Message stop reasons and total input-token definition](https://github.com/anthropics/anthropic-sdk-python/blob/5752c132041af71055584632caab5ca2325cae5e/src/anthropic/types/message.py)
- [Thinking signatures and unchanged replay](https://github.com/anthropics/anthropic-sdk-python/blob/5752c132041af71055584632caab5ca2325cae5e/src/anthropic/types/thinking_block_param.py), [redacted blocks](https://github.com/anthropics/anthropic-sdk-python/blob/5752c132041af71055584632caab5ca2325cae5e/src/anthropic/types/redacted_thinking_block_param.py), and [manual thinking limits](https://github.com/anthropics/anthropic-sdk-python/blob/5752c132041af71055584632caab5ca2325cae5e/src/anthropic/types/thinking_config_enabled_param.py)
- [Usage fields](https://github.com/anthropics/anthropic-sdk-python/blob/5752c132041af71055584632caab5ca2325cae5e/src/anthropic/types/usage.py) and [cache-write TTL counters](https://github.com/anthropics/anthropic-sdk-python/blob/5752c132041af71055584632caab5ca2325cae5e/src/anthropic/types/cache_creation.py)
- [Official thinking and tool-use examples, including conversation-prefix binding](https://github.com/anthropics/claude-cookbooks/blob/785d7a06457b3413727f386b9c470083feb23c49/extended_thinking/extended_thinking_with_tool_use.ipynb)
- [SDK idempotency option documented as a no-op](https://github.com/anthropics/anthropic-sdk-python/blob/5752c132041af71055584632caab5ca2325cae5e/src/anthropic/_types.py)
- [Current model overview](https://platform.claude.com/docs/en/about-claude/models/overview) and [pricing](https://platform.claude.com/docs/en/about-claude/pricing)
