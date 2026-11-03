# OpenWSP

OpenWSP is open tooling for weight-space programming: reading, editing and versioning the weights of open models. It is an independent replication of the work of [Rikiya Takehi](https://github.com/RikiyaT "Equal contribution"), [Jayden Teoh](https://github.com/JaydenTeoh "Equal contribution"), [Alex Zhang](https://github.com/alexzhang13) and [Omar Khattab](https://github.com/okhat) at [MIT CSAIL](https://csail.mit.edu), with no affiliation to the authors.

**Experiment 8 is running again.** The current [`codex-20261103` attempt](replications/exp8/results/live-runs/codex-20261103/identity.json) uses GPT-6 Astra through Codex and a ChatGPT Pro subscription. The [published resume snapshot](replications/exp8/results/live-runs/codex-20261103/resume-20261103.json) records the same attempt continuing from [1 of 50 passes](replications/exp8/results/live-runs/codex-20261103/state.json); use `openwsp status` for current progress. No final edit or live benchmark result is available. The earlier API attempt remains recorded separately at [4 of 50 passes](replications/exp8/results/live-run/blocker.json).

The authors introduced the term in ["Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"](https://rikiyat.github.io/blog/weight-space-programming/). Their experiments used [GPT-6 Astra](https://openai.com/index/gpt-6-astra/) to read weights, repair networks and construct them from scratch, without writing training loops. The editor reasons about the weights and writes changes directly.

OpenWSP began as an effort to reproduce that work and build the missing tools around it. The goal is to make agentic weight editing as reproducible and reviewable as working with code: inspect a model, make a change, measure its effects and preserve the exact diff. The first target is Experiment 8, "Make a language model better at math". The authors edited [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) to reduce its thinking loops and raised MATH-500 accuracy from [7.9%](replications/exp8/reference/unedited.json) to [48.1%](replications/exp8/reference/p50c.json). Reproducing that live editing process is the test of this tooling.

The tools are a Python kernel over model weights, a forward-pass service and a versioned edit format, exposed through one core API as a CLI, Python SDK and MCP server. The current experiment uses [Codex with subscription billing](docs/subscription-replication.md) for its editor, [Modal](https://modal.com) for computation and [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-0.8B) for the pinned base model. The [Responses API adapter](docs/editor-recovery.md) remains available as a separate route. Further experiments, editors and services can build on the same interfaces.

The [CPU raw-weight replications](#raw-weight-experiments) are complete for Astra: all seven game/shape labels are correct, the one-shot car repair fails, and all three written MinAtar policies have measured scores. Fable awaits native credentials; the remaining language-model experiments follow Experiment 8. For human correspondence, use the maintainer's name at their handle dot com.

## Project Status
<!-- OPENWSP:STATUS:START -->
| Recorded component | State | Evidence |
| --- | --- | --- |
| Required live run | [live goal unmet](replications/exp8/results/live-runs/codex-20261103/state.json) | Active attempt: [codex-20261103](replications/exp8/results/active-live.json); submission requirement pending |
| Pipeline | [blocked](replications/exp8/results/pipeline-state.json) | [live](replications/exp8/results/pipeline-state.json) |
| Recorded blocker | [SandboxTransportError: Text transfer failed with exit code 1](replications/exp8/results/pipeline-state.json) | See source record. |
| Driver recovery | [running](replications/exp8/results/coordinator.json) | [all](replications/exp8/results/coordinator.json) |
| Calibration prompt audit | [passed](replications/exp8/results/calibration-prompt-audit.json) | [calibration-prompt-audit.json](replications/exp8/results/calibration-prompt-audit.json) |
| Calibration comparison | [passed](replications/exp8/results/calibration-check.json) | [calibration-check.json](replications/exp8/results/calibration-check.json) |
| Tensor hashes | [passed](replications/exp8/results/kernel/hash-verification.json) | [kernel/hash-verification.json](replications/exp8/results/kernel/hash-verification.json) |
| Sandbox and first cell | [passed](replications/exp8/results/kernel/sandbox-verification.json) | [kernel/sandbox-verification.json](replications/exp8/results/kernel/sandbox-verification.json) |
| Forward service | [passed](replications/exp8/results/kernel/forward-verification.json) | [kernel/forward-verification.json](replications/exp8/results/kernel/forward-verification.json) |
| Weight swap check | [fallback](replications/exp8/results/kernel/weight-swap.json) | [kernel/weight-swap.json](replications/exp8/results/kernel/weight-swap.json) |
| Recorded diagnostic | [Engine does not expose a compatible weight-reload contract](replications/exp8/results/kernel/weight-swap.json) | See source record. |

Live attempts:

| Attempt | Selection | Editor | Recorded state | Passes | Turns | Runs | Submission | Final evaluations | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| live-run | historical | pending | [blocked](replications/exp8/results/live-run/blocker.json) | [4](replications/exp8/results/live-run/state.json) | [26](replications/exp8/results/live-run/state.json) | [10](replications/exp8/results/live-run/state.json) | not established | math500: pending; gsm8k: pending; ifeval: pending | [state.json](replications/exp8/results/live-run/state.json); [blocker.json](replications/exp8/results/live-run/blocker.json); [transcript.md](replications/exp8/results/live-run/transcript.md) |
| live-runs/codex-20261103 | active | [codex](replications/exp8/results/live-runs/codex-20261103/identity.json) | [incomplete](replications/exp8/results/live-runs/codex-20261103/state.json) | [14](replications/exp8/results/live-runs/codex-20261103/state.json) | [50](replications/exp8/results/live-runs/codex-20261103/state.json) | [26](replications/exp8/results/live-runs/codex-20261103/state.json) | not established | math500: pending; gsm8k: pending; ifeval: pending | [state.json](replications/exp8/results/live-runs/codex-20261103/state.json); [identity.json](replications/exp8/results/live-runs/codex-20261103/identity.json); [blocker.json](replications/exp8/results/live-runs/codex-20261103/blocker.json); [transcript.md](replications/exp8/results/live-runs/codex-20261103/transcript.md) |

The active pointer selects the current attempt. Historical blockers and completed controls do not establish its completion.

States are recorded snapshots; use `openwsp status` to check current liveness.
<!-- OPENWSP:STATUS:END -->

The tables distinguish calibration, controls and each live attempt. A completed component does not establish a completed replication. `openwsp status` shows the current driver state; the published records are snapshots.

## Experiment 8

**TL;DR:** [Calibration passed](replications/exp8/results/calibration-check.json), and the [required controls](#openwsp-controls) have results. The subscription attempt [resumed with the same identity and saved counters](replications/exp8/results/live-runs/codex-20261103/resume-20261103.json) after the [Modal allowance update](replications/exp8/results/live-runs/codex-20261103/modal-allowance.json). The published snapshot records [1 of 50 passes](replications/exp8/results/live-runs/codex-20261103/state.json). The earlier API attempt stopped at [4 of 50 passes](replications/exp8/results/live-run/blocker.json), before a final submission, because the next response could exceed its remaining budget. Neither attempt has a final live-edit benchmark result.

### Setup

The [task message](replications/exp8/prompts/message.txt) defines what the editor can see and do. The [source manifest](replications/exp8/sources.json) pins the released material, and [task decisions](docs/TASK.md) record the budget and implementation choices. Each result retains its runtime configuration.

- **Editor:** `gpt-6-astra`, reasoning effort `max`, through the [pinned Codex CLI configuration](docs/codex-editor.md) with ChatGPT Pro billing. It receives the authors' [one-line instructions](replications/exp8/prompts/instructions.md), task message and text-block interface. Native tools and additional instructions are disabled in the verified configuration. The editor has no access to the authors' findings, calibration results or control results.
- **Model:** the text part of `Qwen/Qwen3.5-0.8B`, pinned to [revision `2fc06364715b967f1860aea9cf38778875588b17`](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).
- **Kernel:** persistent Python over float32 weights, with a [96 GiB memory limit](docs/TASK.md), no GPU, no network and no secrets. The model and practice paths are read-only. Edits to the final norm, embedding and output layer, or whole-tensor rescaling, are refused.
- **Tools and run limits:** [50 forward passes and at most 300 Python runs](replications/exp8/prompts/message.txt). A generation pass has up to [512 answers, or 256 per model for a paired comparison](replications/exp8/prompts/message.txt). A teacher-forced pass returns log probabilities, hidden states, MLP activations or attention under the authors' limits.
- **Practice:** the released GSM8K train, MATH train and AIME files, with [7,469, 7,275 and 714 problems respectively](replications/exp8/prompts/message.txt). The editor never receives benchmark problems or calibration results.
- **Evaluation:** the authors' [`think35` protocol](replications/exp8/vendor/evaluate.py): thinking on, temperature [0.6](replications/exp8/vendor/evaluate.py), top-p [0.95](replications/exp8/vendor/evaluate.py), top-k [20](replications/exp8/vendor/evaluate.py), presence penalty [1.5](replications/exp8/vendor/evaluate.py), and up to [32,768 new tokens](replications/exp8/vendor/evaluate.py). Math uses the last boxed answer and math-verify. IFEval uses strict prompt-level grading after the thinking block.
- **Final samples per problem:** [MATH-500: 4; GSM8K: 2; IFEval: 2](replications/exp8/vendor/evaluate.py). Calibration and the required controls use [one sample per problem](docs/TASK.md).
- **Editor billing:** the current route uses the [ChatGPT Pro subscription](docs/TASK.md). Subscription quota is recorded separately from API charges. The retained API route has an [explicit additional allocation and protected Plan B reserve](replications/exp8/authorizations/editor-budget-20261103.json); it is never selected automatically when subscription quota runs out.
- **Compute budget:** The operator authorizes automatic Modal top-ups within the existing [$150 total task cap](docs/TASK.md). The [allowance receipt](replications/exp8/results/live-runs/codex-20261103/modal-allowance.json) records this funding mode without claiming the provider dashboard limit changed. Job estimates precede launches, and unresolved reservations remain in the ledger. The [allowance procedure](docs/modal-budget.md) preserves the authorization and funding evidence.

### Results

`openwsp report exp8` generates these tables from the [authors' reference files](replications/exp8/reference) and [OpenWSP evaluation files](replications/exp8/results). Every number links to its source. Missing measurements remain pending, and unfinished attempts remain visible alongside completed evaluations. The calibrated p50c score measures the authors' released edit; it is not a result from an OpenWSP live editor.

<!-- OPENWSP:RESULTS:START -->
Authors' released evaluations:

| Run | MATH-500 | GSM8K | IFEval | MATH-500 hit token limit |
| --- | --- | --- | --- | --- |
| unedited | [7.9%](replications/exp8/reference/unedited.json) | [13.8%](replications/exp8/reference/unedited.json) | [21.0%](replications/exp8/reference/unedited.json) | [93.0%](replications/exp8/reference/unedited.json) |
| p10c | [12.3%](replications/exp8/reference/p10c.json) | [25.7%](replications/exp8/reference/p10c.json) | [49.0%](replications/exp8/reference/p10c.json) | [87.2%](replications/exp8/reference/p10c.json) |
| p25c | [14.1%](replications/exp8/reference/p25c.json) | [25.1%](replications/exp8/reference/p25c.json) | [39.9%](replications/exp8/reference/p25c.json) | [86.4%](replications/exp8/reference/p25c.json) |
| p50c | [48.1%](replications/exp8/reference/p50c.json) | [67.1%](replications/exp8/reference/p50c.json) | [62.3%](replications/exp8/reference/p50c.json) | [33.1%](replications/exp8/reference/p50c.json) |

Our evaluations and controls (controls are ours):

| Stage | Model | Benchmark | Author comparator | Accuracy | Standard error | Samples | Hit token limit | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration_base | Qwen/Qwen3.5-0.8B | math500 | unedited: [7.9%](replications/exp8/reference/unedited.json) | [7.2%](replications/exp8/results/calibration_base/summary.json) | [1.16%](replications/exp8/results/calibration_base/summary.json) | [500](replications/exp8/results/calibration_base/summary.json) | [93.2%](replications/exp8/results/calibration_base/summary.json) | [done](replications/exp8/results/calibration_base/summary.json) |
| calibration_p50c | Qwen/Qwen3.5-0.8B | math500 | p50c: [48.1%](replications/exp8/reference/p50c.json) | [50.6%](replications/exp8/results/calibration_p50c/summary.json) | [2.24%](replications/exp8/results/calibration_p50c/summary.json) | [500](replications/exp8/results/calibration_p50c/summary.json) | [31.0%](replications/exp8/results/calibration_p50c/summary.json) | [done](replications/exp8/results/calibration_p50c/summary.json) |
| control1_nothink | Qwen/Qwen3.5-0.8B | math500 | not applicable | [46.8%](replications/exp8/results/control1_nothink/math500/summary.json) | [2.23%](replications/exp8/results/control1_nothink/math500/summary.json) | [500](replications/exp8/results/control1_nothink/math500/summary.json) | [0.6%](replications/exp8/results/control1_nothink/math500/summary.json) | [done](replications/exp8/results/control1_nothink/math500/summary.json) |
| control2_mechanical | Qwen/Qwen3.5-0.8B | math500 | not applicable | [46.0%](replications/exp8/results/control2_mechanical/math500/summary.json) | [2.23%](replications/exp8/results/control2_mechanical/math500/summary.json) | [500](replications/exp8/results/control2_mechanical/math500/summary.json) | [45.2%](replications/exp8/results/control2_mechanical/math500/summary.json) | [done](replications/exp8/results/control2_mechanical/math500/summary.json) |
| control3_repetition | Qwen/Qwen3.5-0.8B | math500 | not applicable | [23.8%](replications/exp8/results/control3_repetition/math500/summary.json) | [1.90%](replications/exp8/results/control3_repetition/math500/summary.json) | [500](replications/exp8/results/control3_repetition/math500/summary.json) | [75.6%](replications/exp8/results/control3_repetition/math500/summary.json) | [done](replications/exp8/results/control3_repetition/math500/summary.json) |
| damage_cpu_validation | pending | pending | pending | pending | pending | pending | pending | [validated](replications/exp8/results/damage_cpu_validation/summary.json) |
| eval_grid_cpu_validation | pending | token_state_machine | pending | pending | pending | pending | pending | [validated](replications/exp8/results/eval_grid_cpu_validation/summary.json) |
| pending | {'checkpoint_hash_note': 'HF file includes vision weights; prepare extracts language-model tensors into a different safetensors artifact. Listed edited tensor base hashes were verified against both edit manifests.', 'config_sha256': 'b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204', 'downloaded_huggingface_checkpoint_sha256': '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696', 'embedding_name': 'model.language_model.embed_tokens.weight', 'id': 'Qwen/Qwen3.5-0.8B', 'prepared_text_only_checkpoint_sha256': '978661a573971be7257815c2554a28c74913c7b7ea51b0a0b9503fd11eb0d2f8', 'revision': '2fc06364715b967f1860aea9cf38778875588b17', 'tied_embeddings_verified_by_config': True, 'tokenizer_sha256': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'} | pending | pending | pending | pending | pending | pending | [measured_cpu_only](replications/exp8/results/inspection/summary.json) |
| pending | pending | pending | pending | pending | pending | pending | pending | [unspecified](replications/exp8/results/repair/2a/summary.json) |
| pending | pending | pending | pending | pending | pending | pending | pending | [unspecified](replications/exp8/results/repair/2b/summary.json) |
| pending | pending | pending | pending | pending | pending | pending | pending | [cpu_builds_only](replications/exp8/results/repair/summary.json) |

Only recorded values are shown. Missing fields remain pending; partial runs retain their recorded status.

Calibration comparison:

| Condition | Accuracy | Reference | Gap (points) | Within prescribed tolerance |
| --- | --- | --- | --- | --- |
| calibration_base | [7.2%](replications/exp8/results/calibration-check.json) | [7.9%](replications/exp8/results/calibration-check.json) | [-0.70](replications/exp8/results/calibration-check.json) | [True](replications/exp8/results/calibration-check.json) |
| calibration_p50c | [50.6%](replications/exp8/results/calibration-check.json) | [48.1%](replications/exp8/results/calibration-check.json) | [2.50](replications/exp8/results/calibration-check.json) | [True](replications/exp8/results/calibration-check.json) |

Recorded runtime and uncertainty:

| Stage / benchmark | Engine | GPU | Transformers | Uncertainty method |
| --- | --- | --- | --- | --- |
| calibration_base / math500 | [vllm==0.33.0](replications/exp8/results/calibration_base/summary.json) | [H100](replications/exp8/results/calibration_base/summary.json) | [5.17.0](replications/exp8/results/calibration_base/summary.json) | pending |
| calibration_p50c / math500 | [vllm==0.33.0](replications/exp8/results/calibration_p50c/summary.json) | [H100](replications/exp8/results/calibration_p50c/summary.json) | [5.17.0](replications/exp8/results/calibration_p50c/summary.json) | pending |
| control1_nothink / math500 | [vllm==0.33.0](replications/exp8/results/control1_nothink/math500/summary.json) | [H100](replications/exp8/results/control1_nothink/math500/summary.json) | [5.17.0](replications/exp8/results/control1_nothink/math500/summary.json) | [binomial_per_sample](replications/exp8/results/control1_nothink/math500/summary.json) |
| control2_mechanical / math500 | [vllm==0.33.0](replications/exp8/results/control2_mechanical/math500/summary.json) | [H100](replications/exp8/results/control2_mechanical/math500/summary.json) | [5.17.0](replications/exp8/results/control2_mechanical/math500/summary.json) | [binomial_per_sample](replications/exp8/results/control2_mechanical/math500/summary.json) |
| control3_repetition / math500 | [vllm==0.33.0](replications/exp8/results/control3_repetition/math500/summary.json) | [H100](replications/exp8/results/control3_repetition/math500/summary.json) | [5.17.0](replications/exp8/results/control3_repetition/math500/summary.json) | [binomial_per_sample](replications/exp8/results/control3_repetition/math500/summary.json) |
| damage_cpu_validation / pending | pending | pending | pending | pending |
| eval_grid_cpu_validation / token_state_machine | pending | pending | pending | pending |
| pending / pending | pending | pending | pending | pending |
| pending / pending | pending | pending | pending | pending |
| pending / pending | pending | pending | pending | pending |
| pending / pending | pending | pending | pending | pending |
<!-- OPENWSP:RESULTS:END -->

### OpenWSP controls

These controls were added for this replication. They answer alternative explanations for the reported gain, using definitions fixed in the [protocol](docs/TASK.md) before their results were read. They are OpenWSP results and are not attributed to the authors.

| Question | Control | Evidence |
| --- | --- | --- |
| Is the baseline broken by the thinking setup? | Base model with thinking off and the model card's recommended non-thinking sampling. | `control1_nothink` in the [generated results](#results). |
| Is it a mechanical direction removal? | Difference of mean MLP outputs for looping versus finished base answers, removed from the same layers and columns in the same form as the published p50c edit. | `control2_mechanical` in the [generated results](#results). |
| Is it a decoding fix? | Base model with repetition penalty [1.1](docs/TASK.md), everything else unchanged. | `control3_repetition` in the [generated results](#results). |
| Does the direction matter? | Published p50c edit with a random unit vector replacing its direction. | Optional; no result published. |
| Is looping specific to vLLM? | Base model in another engine with the same sampling on the prescribed subset. | Optional; no result published. |

The mechanical control constructs its direction from calibration generations on the same benchmark. Its [method record](replications/exp8/results/control2_mechanical/method.json) documents that reuse. It is a diagnostic comparison, not a held-out discovery experiment.

### What happened during the run

The [subscription attempt's transcript](replications/exp8/results/live-runs/codex-20261103/transcript.md) records its explanations, Python cells, forward requests and returned results. Its [identity](replications/exp8/results/live-runs/codex-20261103/identity.json) and [state](replications/exp8/results/live-runs/codex-20261103/state.json) distinguish it from the [earlier API transcript](replications/exp8/results/live-run/transcript.md). Each attempt keeps its failed requests as well as its successful ones. Full generations and activations stay on the volume; public records retain their paths and hashes.

The subscription attempt started from the original pinned weights, empty working state and fresh counters. It does not inherit the API attempt's partial edits or conversation. Within the subscription attempt, Codex resumes the exact saved conversation. A [synthetic check](replications/exp8/results/kernel/codex-subscription-probe.json) verified short formatted output and same-session continuation; it does not establish reliability over a full live run.

The API attempt stopped at its [editor budget admission check](replications/exp8/results/live-run/blocker.json). A later audit found [seven output-limit responses](replications/exp8/results/live-run/editor-format-loop.json) that wrote a complete fenced action and then repeated backticks and blank lines. Their trailing loops contain about [49,000 to 213,000 characters each](replications/exp8/results/live-run/editor-format-loop.json). None of those incomplete actions was executed. Raising the output limit did not solve the loop, and retaining the responses enlarged subsequent context. The [API recovery work](docs/editor-recovery.md) remains available separately.

The subscription attempt paused when its next GPU request needed a [$2.98704 reservation with $2.90964 available](replications/exp8/results/live-runs/codex-20261103/blocker.json). That request was not dispatched at the budget stop. The [resume record](replications/exp8/results/live-runs/codex-20261103/resume-20261103.json) now preserves continuation of the same attempt after the funding update; the earlier blocker remains as a resolved historical event.

No final live edit has been submitted. Candidate edits remain the editor's choices; the operator does not choose a winner or alter its weights. Before live editing, [kernel verification](replications/exp8/results/kernel/hash-verification.json) reproduced the released p50c tensor hashes, and the [sandbox check](replications/exp8/results/kernel/sandbox-verification.json) ran the authors' first Python cell unmodified.

### Cost and time

The cost records distinguish the earlier API charges, Modal resource estimates and unresolved reservations. Subscription usage has its own [token receipts](replications/exp8/results/live-runs/codex-20261103/subscription-usage.json), without assigning an API price or treating it as free. Each duration retains its measurement basis. Adding resource time across jobs does not give end-to-end elapsed time.

<!-- OPENWSP:COSTS:START -->
| Stage | Benchmark | Recorded hours | Recorded cost (USD) |
| --- | --- | --- | --- |
| calibration_base | math500 | [0.465](replications/exp8/results/calibration_base/summary.json) | [2.1280](replications/exp8/results/calibration_base/summary.json) |
| calibration_p50c | math500 | [0.259](replications/exp8/results/calibration_p50c/summary.json) | [1.2512](replications/exp8/results/calibration_p50c/summary.json) |
| control1_nothink | math500 | [0.236](replications/exp8/results/control1_nothink/math500/summary.json) | [1.0059](replications/exp8/results/control1_nothink/math500/summary.json) |
| control2_mechanical | math500 | [0.447](replications/exp8/results/control2_mechanical/math500/summary.json) | [1.9082](replications/exp8/results/control2_mechanical/math500/summary.json) |
| control3_repetition | math500 | [0.514](replications/exp8/results/control3_repetition/math500/summary.json) | [2.1923](replications/exp8/results/control3_repetition/math500/summary.json) |
| damage_cpu_validation | pending | pending | pending |
| eval_grid_cpu_validation | token_state_machine | pending | pending |
| pending | pending | pending | pending |
| pending | pending | pending | pending |
| pending | pending | pending | pending |
| pending | pending | pending | pending |

Costs and durations come from the linked records. Missing costs are not treated as zero; rows are not summed because stages may overlap.

Budget ledger: [source](replications/exp8/results/costs.json). Reservations are retained bounds, not charges.

| Budget / stage / operation | State | Receipts | Recorded hours | Settled amount (USD) | Reserved bound (USD) | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| editor / editor turns | [settled](replications/exp8/results/costs.json) | [26](replications/exp8/results/costs.json) | [1.815](replications/exp8/results/costs.json) | [78.2928](replications/exp8/results/costs.json) | pending | [API usage and recorded token rates](replications/exp8/results/costs.json) |
| modal / all | [reserved](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | pending | pending | [1.9060](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / all | [settled](replications/exp8/results/costs.json) | [5](replications/exp8/results/costs.json) | [4.262](replications/exp8/results/costs.json) | [0.3380](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / calibration | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.906](replications/exp8/results/costs.json) | [0.0718](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / calibration_base | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.499](replications/exp8/results/costs.json) | [2.1280](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / calibration_base_smoke | [reserved](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | pending | pending | [2.2758](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / calibration_base_smoke | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.105](replications/exp8/results/costs.json) | [0.4496](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / calibration_p50c | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.293](replications/exp8/results/costs.json) | [1.2512](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / calibration_p50c_smoke | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.099](replications/exp8/results/costs.json) | [0.4228](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / control1_nothink / math500 / evaluation | [settled](replications/exp8/results/costs.json) | [2](replications/exp8/results/costs.json) | [0.236](replications/exp8/results/costs.json) | [1.0059](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / control1_nothink / math500 / evaluation smoke | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.066](replications/exp8/results/costs.json) | [0.2833](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / control2 / directions | [settled](replications/exp8/results/costs.json) | [16](replications/exp8/results/costs.json) | [0.681](replications/exp8/results/costs.json) | [2.9042](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / control2 / directions smoke | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1613](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / control2_mechanical / math500 / evaluation | [settled](replications/exp8/results/costs.json) | [2](replications/exp8/results/costs.json) | [0.447](replications/exp8/results/costs.json) | [1.9082](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / control2_mechanical / math500 / evaluation smoke | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.090](replications/exp8/results/costs.json) | [0.3856](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / control3_repetition / math500 / evaluation | [settled](replications/exp8/results/costs.json) | [2](replications/exp8/results/costs.json) | [0.514](replications/exp8/results/costs.json) | [2.1923](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / control3_repetition / math500 / evaluation smoke | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.091](replications/exp8/results/costs.json) | [0.3880](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / controls | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [1.612](replications/exp8/results/costs.json) | [0.1278](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / infrastructure | [reserved](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | pending | pending | [2.0000](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / kernel | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.205](replications/exp8/results/costs.json) | [0.0163](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / kernel_verification:heartbeat | [settled](replications/exp8/results/costs.json) | [2](replications/exp8/results/costs.json) | [0.003](replications/exp8/results/costs.json) | [0.0073](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / kernel_verification:startup | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.004](replications/exp8/results/costs.json) | [0.0092](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / live / activation extraction / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / activation extraction / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.071](replications/exp8/results/costs.json) | [0.3049](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_1 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.234](replications/exp8/results/costs.json) | [0.9999](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_1 / original / activations | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.105](replications/exp8/results/costs.json) | [0.4491](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_10 / edited | [reserved](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | pending | pending | [2.9870](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_10 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_11 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.039](replications/exp8/results/costs.json) | [0.1680](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_12 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_13 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.039](replications/exp8/results/costs.json) | [0.1670](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_14 / edited | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_15 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1610](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_16 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1638](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_17 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1639](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_18 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1625](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_19 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1607](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_2 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.039](replications/exp8/results/costs.json) | [0.1645](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_20 / edited | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.098](replications/exp8/results/costs.json) | [0.4199](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_20 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.079](replications/exp8/results/costs.json) | [0.3392](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_21 / edited | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.249](replications/exp8/results/costs.json) | [1.0622](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_21 / edited / activations | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.078](replications/exp8/results/costs.json) | [0.3342](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_21 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.276](replications/exp8/results/costs.json) | [1.1781](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_21 / original / activations | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.085](replications/exp8/results/costs.json) | [0.3608](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_22 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1616](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_3 / edited | [reserved](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | pending | pending | [2.9870](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_3 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_4 / edited | [reserved](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | pending | pending | [2.9870](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_4 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_5 / edited | [reserved](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | pending | pending | [2.9870](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_5 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_6 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.039](replications/exp8/results/costs.json) | [0.1647](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_7 / edited | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_8 / original | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [failed job upper bound; not invoice](replications/exp8/results/costs.json) |
| modal / live / codex-20261103 / fwd_9 / teacher_forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1633](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / generation / edited | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.111](replications/exp8/results/costs.json) | [0.4719](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / generation / original | [settled](replications/exp8/results/costs.json) | [3](replications/exp8/results/costs.json) | [0.637](replications/exp8/results/costs.json) | [2.7181](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / teacher-forced | [settled](replications/exp8/results/costs.json) | [2](replications/exp8/results/costs.json) | [0.079](replications/exp8/results/costs.json) | [0.3373](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / prepare | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.010](replications/exp8/results/costs.json) | [0.0016](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / prepare-v2 | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.0060](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / publish | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.134](replications/exp8/results/costs.json) | [0.0107](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / sandbox:heartbeat | [settled](replications/exp8/results/costs.json) | [305](replications/exp8/results/costs.json) | [4.976](replications/exp8/results/costs.json) | [12.8769](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / sandbox:startup | [settled](replications/exp8/results/costs.json) | [6](replications/exp8/results/costs.json) | [0.022](replications/exp8/results/costs.json) | [0.0581](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / verify / attention | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1626](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify / fresh-reference | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.086](replications/exp8/results/costs.json) | [0.3677](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify / native-swap | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.097](replications/exp8/results/costs.json) | [0.4123](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify / teacher-forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1635](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |

Repeated requests, attempts and batches are grouped by stage and operation; the linked ledger retains each receipt. Ledger rows and evaluation summaries may describe the same jobs; do not add the two tables. Summed resource hours are not end-to-end wall time. Settled resource estimates remain estimates until reconciled with provider billing.
<!-- OPENWSP:COSTS:END -->

### FAQ

**Is the baseline broken?** The thinking baseline scores [7.2%](replications/exp8/results/calibration_base/summary.json), within the prescribed tolerance of the authors' result. The same base weights reach [46.8% with thinking off](replications/exp8/results/control1_nothink/math500/summary.json), and the token-limit share falls from [93.2%](replications/exp8/results/calibration_base/summary.json) to [0.6%](replications/exp8/results/control1_nothink/math500/summary.json). This control changes both thinking mode and sampling to the model card's non-thinking recommendations. It does not isolate either change alone.

**Is it a decoding fix?** Adding repetition penalty [1.1](replications/exp8/results/control3_repetition/math500/summary.json) with the base weights and other thinking settings fixed raises accuracy to [23.8%](replications/exp8/results/control3_repetition/math500/summary.json). The token-limit share remains [75.6%](replications/exp8/results/control3_repetition/math500/summary.json). This decoding change helps, but does not reproduce the [50.6%](replications/exp8/results/calibration_p50c/summary.json) calibrated p50c result.

**What tools did the editor have?** The kernel, practice files and bounded forward-pass interface listed above. It does not see test sets, the reference edit, calibration outcomes, control results or the internet. The live experiment uses the authors' text-block protocol. The public MCP server exposes the core operations for other agent workflows.

**What did it cost?** The stopped API attempt incurred [$78.292765](replications/exp8/results/live-run/editor-cost-analysis.json) in editor charges before its next response failed the budget check. The incomplete requests account for [$60.234243 including input](replications/exp8/results/live-run/editor-cost-analysis.json). The current attempt uses a [ChatGPT Pro subscription](docs/TASK.md), whose quota counters do not establish a dollar cost per run. Modal computation is still paid and appears separately in the generated table. The [budget policy](docs/TASK.md) sets spending limits; it is not a cost measurement.

### Differences from the authors' setup

| Component | Difference and evidence |
| --- | --- |
| Harness | The original Experiment 8 driver, kernel and forward server were not released. OpenWSP reconstructs the documented contract from [their release](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math). |
| Editor transport | The current attempt uses [pinned Codex CLI with ChatGPT billing](docs/codex-editor.md). The authors' released logs show Codex use, but their exact CLI version and billing method were not released. OpenWSP's local worker keeps authentication and conversation files off Modal, and the pinned configuration disables native tools. The earlier API attempt used Responses chaining; it remains a separate recorded attempt. |
| Editor output handling | In the API attempt, the ceiling changed from [32,768 to 128,000 after the first truncated turn](replications/exp8/results/live-run/editor-output-limit.json). Incomplete responses were charged and retained in history, but never executed. The [API streaming repair](docs/editor-recovery.md) requests cancellation when it detects the recorded loop. Its cancellation and recovery behavior has CPU tests but has not been exercised against a live provider response. The Codex adapter checks completed output for the same pattern. Neither check establishes that the subscription attempt cannot loop. |
| Completed-output execution gate | Both editor routes now reject completed replies above [17,734 UTF-8 bytes, twice the largest legitimate author reply of 8,867 bytes across 318 turns](replications/exp8/reference/editor-output-limit.json). The response and usage are retained once, and the text is not executed. This is a completed-text gate, not a guarantee of early streaming cancellation. |
| Sandbox | Modal Sandbox instead of the authors' local isolation. The memory limit follows the authors' evidence. See [architecture](docs/architecture.md) for what the kernel and OS enforce. |
| Inference engine, GPU and dependency versions | The generated runtime table links each recorded version and GPU to its result provenance. Missing metadata remains pending. |
| Weight updates | The production native-update check found an incompatible vLLM interface. This run uses the specified [fresh-engine fallback](replications/exp8/results/kernel/weight-swap.json), loading the submitted checkpoint for each pass. |
| Scheduling | Subsequent jobs cap active vLLM sequences at [128](DECISIONS.md), based on observed KV-cache capacity. Final evaluations use batches of [256 generations](replications/exp8/evaluation.py). Compilation caches persist between containers; GPUs scale down between editor turns. |
| Refusal implementation | The authors did not release the numerical whole-tensor scaling test. OpenWSP's explicit tolerance and refusal tests are documented in [the kernel implementation](src/openwsp/core.py) and [decisions](DECISIONS.md). |
| Sampling and seeds | Evaluation preserves the published sampling settings. The original per-problem seed algorithm was not released. Live passes use the [first eight SHA-256 hex digits of the problem ID, modulo 2^31](src/openwsp/backends/modal_app.py), identically for both models and repeated requests. Calibration uses the published seed. |
| Duplicate requests | [Authorized duplicate handling](docs/STEERING_UPDATE_2.md) restores an identical effective request without incrementing the forward count. The editor receives no duplicate or reuse notice. A weight change makes the request distinct and counts a new pass. |
| Base-generation reuse | [Steering update 2](docs/STEERING_UPDATE_2.md) enables reuse for identical problem, prompt, sampling, model-hash and seed keys. The bank records reuse privately in result provenance; the editor receives the normal result interface. [CPU fixture checks](replications/exp8/results/forward-bank-cpu.json) verify reuse and cache invalidation, not model accuracy. |

## Quickstart

Install the locked dependencies from this repository. The tests and tiny-model examples run on CPU.

```sh
git clone https://github.com/openwsp/openwsp.git
cd openwsp
uv sync --locked --extra dev --extra modal --extra replication
uv run openwsp --help
uv run pytest -q
```

For the primary ChatGPT subscription route, follow the [subscription setup](docs/subscription-replication.md). A persistent local Codex worker supplies editor responses while Modal runs the kernel and forward passes. Its private configuration names a run and the local authentication and state directories. Keep the worker and its host available throughout the run.

Set the variables listed in [.env.example](.env.example) through the environment's secret mechanism. Both routes use Modal for compute, Hugging Face for model access and GitHub for result publication. `OPENAI_API_KEY` is needed only for the Responses API route. ChatGPT credentials stay with the local Codex worker. Keep all credentials out of repository files and remote URLs.

Create the named secrets in the Modal workspace environment before deployment. This command reads exported values at runtime, maps `GITHUB_ACCESS_TOKEN` to the remote `GITHUB_TOKEN` key and retains existing secrets:

```sh
uv run --no-sync python - <<'PY'
import os
import modal
from openwsp.backends.modal_runtime import configure_tls

configure_tls()
secrets = {
    "huggingface": {"HF_TOKEN": os.environ["HF_TOKEN"]},
    "github": {"GITHUB_TOKEN": os.environ["GITHUB_ACCESS_TOKEN"]},
}
if os.environ.get("OPENAI_API_KEY"):
    secrets["openai"] = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
if any(not value for contents in secrets.values() for value in contents.values()):
    raise ValueError("Required secret environment variables must be nonempty")
for name, contents in secrets.items():
    modal.Secret.objects.create(name, contents, allow_existing=True)
PY
```

Update existing secrets in Modal if their credentials have changed; see [Modal's secret setup](https://modal.com/docs/guide/secrets). The local CPU examples below require no credentials.

Launch or resume the named subscription attempt with the private configuration from the setup guide. Modal compute is still paid.

```sh
uv run openwsp replicate exp8 --config /absolute/private/codex-exp8.json --foreground
uv run openwsp status exp8
uv run openwsp report exp8
```

`openwsp status exp8` shows progress on one screen; add `--json` for the full saved state. `openwsp report exp8` rebuilds the README tables from locally available result files. Current progress and the configured resume command are in [STATUS.md](STATUS.md).

Without a subscription configuration, `uv run openwsp replicate exp8` retains the Responses API route and its token billing. It requires the `openai` Modal secret. No automatic switch between billing routes occurs. Reusing a subscription run ID resumes its saved attempt; a new ID starts a separate attempt from the original weights.

For a local CPU example:

```sh
uv run openwsp --tiny --work-dir /tmp/openwsp-demo tensors
uv run openwsp --tiny --work-dir /tmp/openwsp-demo forward '{"inputs": [[1.0, 0.0]], "model": "both"}'
```

For another model, supply its safetensors checkpoint and an installed backend factory:

```sh
uv run openwsp --model /absolute/path/model.safetensors \
  --work-dir /absolute/path/work --backend my_backend:create tensors
```

The factory supplies inference for that model. Loading tensors alone does not configure a forward service. The same read, run, forward, save, diff, inspect, apply and export operations are available through the [Python SDK](docs/architecture.md) and the [MCP server](docs/mcp.md), with connection examples for Codex, Claude Code and Cursor. The [edit format](docs/edit-format.md) defines the portable diff and its verification rules.

Editors sit behind a separate `complete(messages)` interface. The [profile and transport architecture](docs/editor-architecture.md) retains the Astra subscription and Responses routes and adds a Fable profile over the [Anthropic Messages adapter](docs/anthropic-editor.md). Native API use requires explicit prices and token bounds. Fable has mocked transport tests but no live Experiment 8 comparison. OpenRouter is an extension target, not an implemented route. The current experiment continues to use Astra.

## Raw-weight experiments

Astra completed [all 11 released items](replications/raw_weights/results/summary.json) across experiments 1, 2, 4 and 5. Every item received the verbatim user prompt and neutral instruction, a fresh conversation, no tools, one reply and no feedback or best-of selection. Native Fable 5.1 has an [authorized $20 cap for the whole set](replications/raw_weights/results/fable-blocker.json), but remains unstarted because its API key is not configured. That record contains the exact token-count and execution commands; no provider substitution or API fallback was used. The [runner and scoring plan](replications/raw_weights/README.md) give source hashes, setup and licenses.

<!-- OPENWSP:RAW_RESULTS:START -->
| Experiment | Authors' released result | Astra (subscription) | Fable (native Anthropic) |
| --- | --- | --- | --- |
| [1: identify the game](replications/exp1) | [4/4](replications/exp1/reference/scores.json) | [4/4](replications/raw_weights/results/summary.json) | [Pending](replications/raw_weights/results/fable-blocker.json) |
| [2: identify image families](replications/exp2) | [3/3](replications/exp2/reference/scores.json) | [3/3](replications/raw_weights/results/summary.json) | [Pending](replications/raw_weights/results/fable-blocker.json) |
| [4: repair the car](replications/exp4) | [1,024/1,024 successes; 6,073/171,664 weights changed](replications/exp4/reference/oneshot_result.json) | [0/1,024 successes; 2,918/171,664 weights changed](replications/raw_weights/results/summary.json) | [Pending](replications/raw_weights/results/fable-blocker.json) |

Experiment 5 scores are mean ± standard error over [1,000 episodes per policy](replications/exp5/reference/minatar_eval.json). Each editor supplies one reply per game; standard errors describe variability across game episodes.

| Game | Authors' PPO 100M: mean [range], seeds | Authors' PPO 500M: mean [range], seeds | Authors' Astra | Astra (subscription) | Fable (native Anthropic) |
| --- | --- | --- | --- | --- | --- |
| Breakout | [8.21 [5.61, 15.78], 10 seeds](replications/exp5/analysis/ppo-seed-summary.json) | [19.12 [12.19, 25.34], 3 seeds](replications/exp5/analysis/ppo-seed-summary.json) | [517.41 ± 7.42](replications/exp5/reference/minatar_eval.json) | [458.73 ± 4.99](replications/raw_weights/results/astra_subscription/exp5/breakout/result.json) | [Pending](replications/raw_weights/results/fable-blocker.json) |
| Seaquest | [4.82 [4.41, 5.62], 3 seeds](replications/exp5/analysis/ppo-seed-summary.json) | [33.59 [4.38, 88.88], 3 seeds](replications/exp5/analysis/ppo-seed-summary.json) | [53.11 ± 1.36](replications/exp5/reference/minatar_eval.json) | [2.41 ± 0.11](replications/raw_weights/results/astra_subscription/exp5/seaquest/result.json) | [Pending](replications/raw_weights/results/fable-blocker.json) |
| Asterix | [18.19 [13.21, 25.82], 10 seeds](replications/exp5/analysis/ppo-seed-summary.json) | [48.93 [44.71, 55.20], 3 seeds](replications/exp5/analysis/ppo-seed-summary.json) | [53.18 ± 1.02](replications/exp5/reference/minatar_eval.json) | [39.71 ± 0.73](replications/raw_weights/results/astra_subscription/exp5/asterix/result.json) | [Pending](replications/raw_weights/results/fable-blocker.json) |

PPO entries are reference-only, equally weighted means over every released training seed; the ranges and complete seed lists are [retained in the source-derived summary](replications/exp5/analysis/ppo-seed-summary.json). The authors' post reports best-seed PPO values. No PPO training or model selection was repeated.
<!-- OPENWSP:RAW_RESULTS:END -->

The failed car reply and every lower-scoring policy are retained without refinement or replacement. The [car scorer calibration](replications/exp4/calibration/summary.json) reproduced the released base and repaired networks; [MinAtar source-network calibration](replications/raw_weights/calibration/minatar/summary.json) evaluated all three published policies before the new policies were scored. Those calibration records are separate from the new editor outcomes above. A [verification record](replications/raw_weights/results/astra-verification.json) preserves the scored float32 tensor hashes and distinguishes them from the rounded bfloat16 edit exports.

These are prompt-faithful replications with disclosed execution differences. Astra matches the authors' `high` effort for experiments 1/2/4 and `xhigh` for 5, including their reasoning-summary choices; OpenWSP's verified Codex profile also explicitly selects `low` verbosity. The original complete client/configuration envelope and subscription entitlement were not released. Despite identical released prompt bytes, [input-token receipts](replications/raw_weights/results/summary.json) are [3,480 tokens lower on every label, 3,478 lower on the car prompt, and 3,503 lower on every MinAtar prompt](replications/raw_weights/results/summary.json) than the [authors' counts](replications/raw_weights/inventory.json). The released experiment folders contain rules, replies, networks and result files; the scorers are disclosed reconstructions validated against that evidence.

## Limits and unfinished work

This replication does not train the model or compute gradients. The live editor receives no benchmark data, reference edit or control outcomes, and its final edit is not selected by an operator. The original API attempt was not relabeled as a completed run when the subscription attempt began.

The raw-weight experiments retain every Astra outcome above; the Fable comparison remains blocked on native credentials, with its authorized budget unused. Experiments [3](replications/exp3), [6](replications/exp6) and [7](replications/exp7) remain layout only. There is no hosted service, web UI, package-registry release or Hugging Face publication of a new live edit. The expanded [evaluation grid](replications/exp8/eval_grid_configs) has CPU implementation checks, not measured GPU results. Optional controls and additional editor comparisons remain separate from the active Experiment 8 attempt.

## Add a result

The result table can be extended by pull request. Add a separately identified run under [replications/exp8/results](replications/exp8/results), including its pinned configuration, compact sample records, summaries, edit identity, usage, costs and differences from the protocol. Run `openwsp report exp8` to derive the table from those files. Preserve failed, incomplete and lower-scoring attempts as well as completed ones. The [result schema](docs/results-schema.md) describes the required evidence and active-attempt selection. The [contributor guide](docs/contributing-experiments.md) explains how to add a grid cell, repair variant or experiment folder.

## Dual use

The method that removes a loop can remove safety behavior too. This repository ships no such edits. Experiment 8 uses a pinned base model, and OpenWSP edits are versioned, SHA-256-addressed diffs so the changed tensors can be verified. The format validates the listed tensors; callers must also verify the complete base checkpoint and revision. Model and revision labels alone do not enforce that match.

OpenWSP code is [MIT licensed](LICENSE). The authors' scripts, logs and edits retain their respective licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
