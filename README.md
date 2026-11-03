# OpenWSP

OpenWSP is open tooling for weight-space programming: to read, edit and version the weights of open models. The term was introduced by [Rikiya Takehi](https://github.com/RikiyaT "Equal contribution"), [Jayden Teoh](https://github.com/JaydenTeoh "Equal contribution"), [Alex Zhang](https://github.com/alexzhang13) and [Omar Khattab](https://github.com/okhat) from the [MIT CSAIL](https://csail.mit.edu) on November 2, 2026 as [a blog post](https://rikiyat.github.io/blog/weight-space-programming/) titled "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights". They used [GPT-6 Astra](https://openai.com/index/gpt-6-astra/) with natural language and demonstrated and it can read neural network weights, fix them, and write them from scratch. Importantly, GPT-6 Astra demonstrated these capabilities without being allowed to write training loops, it reasoned about the weights demonstrating an alien understanding of neural networks.

I wanted to replicate this groundbreaking approach and realized a lot of the tooling is missing, hence started developing OpenWSP. My goal is to make it easier to read, edit and version LLM weights. I see a future where editing weights of LLMs agentically is as common as we humans were used to reason and commit code. Therefore, there is a lot of tools and services to develop around this new workflow. My first goal is to replicate _Sparks of Weight-Space Programming_'s Experiment 8: "Make a language model better at math". Namely, to edit the [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) weights to fix its thinking loops shortcoming to improve its MATH-500 score from 7.9% to about 48.1%, as demonstrated by the MIT researchers.

To get to there, the tools in development are a Python kernel over model weights, a forward-pass service and a versioned edit format, exposed through one API as a CLI, Python SDK and MCP server. To replicate Experiment 8, it is using [OpenAI API](https://developers.openai.com/api/docs/models/gpt-6-astra) to access GPT-6 Astra and [Modal](https://modal.com) for computer, [Hugging Face](https://huggingface.co) to read the open weight LLM and publish its modification.

Once Experiment 8 is replicated, I plan to replicate all of their experiments and have more fun. Email me by my name at my handle dot com only if you're a human.

## Project Status
<!-- OPENWSP:STATUS:START -->
| Recorded component | State | Evidence |
| --- | --- | --- |
| Required live run | [blocked](replications/exp8/results/live-run/blocker.json) | [live goal unmet](replications/exp8/results/live-run/blocker.json); [editor_budget_admission](replications/exp8/results/live-run/blocker.json) |
| Live pass requirement | [4](replications/exp8/results/live-run/blocker.json) / [50](replications/exp8/results/live-run/blocker.json) | Recorded turn: [26](replications/exp8/results/live-run/blocker.json) |
| Live final evaluation | [not_started_no_completed_50_pass_submission](replications/exp8/results/live-run/blocker.json) | See live-run record. |
| Recorded live blocker | [BudgetExceeded: editor stage editor:t027: reserve $30.478625, only $21.7072350 available](replications/exp8/results/live-run/blocker.json) | See live-run record. |
| Pipeline | [stage_done](replications/exp8/results/pipeline-state.json) | [publication](replications/exp8/results/pipeline-state.json) |
| Driver recovery | [blocked](replications/exp8/results/coordinator.json) | [all](replications/exp8/results/coordinator.json) |
| Recorded blocker | [Pipeline blocked; previous owner reconciled and automatic restart disabled](replications/exp8/results/coordinator.json) | See source record. |
| Calibration prompt audit | [passed](replications/exp8/results/calibration-prompt-audit.json) | [calibration-prompt-audit.json](replications/exp8/results/calibration-prompt-audit.json) |
| Calibration comparison | [passed](replications/exp8/results/calibration-check.json) | [calibration-check.json](replications/exp8/results/calibration-check.json) |
| Tensor hashes | [passed](replications/exp8/results/kernel/hash-verification.json) | [kernel/hash-verification.json](replications/exp8/results/kernel/hash-verification.json) |
| Sandbox and first cell | [passed](replications/exp8/results/kernel/sandbox-verification.json) | [kernel/sandbox-verification.json](replications/exp8/results/kernel/sandbox-verification.json) |
| Forward service | [passed](replications/exp8/results/kernel/forward-verification.json) | [kernel/forward-verification.json](replications/exp8/results/kernel/forward-verification.json) |
| Weight swap check | [fallback](replications/exp8/results/kernel/weight-swap.json) | [kernel/weight-swap.json](replications/exp8/results/kernel/weight-swap.json) |
| Recorded diagnostic | [Engine does not expose a compatible weight-reload contract](replications/exp8/results/kernel/weight-swap.json) | See source record. |
| Live editor | [incomplete](replications/exp8/results/live-run/state.json) | passes: [4](replications/exp8/results/live-run/state.json); turn: [26](replications/exp8/results/live-run/state.json); runs: [10](replications/exp8/results/live-run/state.json) |

Completed controls or publication do not satisfy the blocked live-run goal.

States are recorded snapshots; use `openwsp status` to check current liveness.
<!-- OPENWSP:STATUS:END -->

The tools are a Python kernel over model weights, a forward-pass service and a versioned edit format, exposed through one API as a CLI, Python SDK and MCP server. The test is Experiment 8: can an editor diagnose a model's math failures and repair its weights without gradients or training?

## Experiment 8

**TL;DR:** [Calibration passed](replications/exp8/results/calibration-check.json). The required live run stopped at [4 of 50 passes](replications/exp8/results/live-run/blocker.json) before a final submission because the next editor response could exceed the remaining budget. Controls are reported separately; there is no final live-edit benchmark result.

### Setup

The prescribed editor setup is recorded in the [task message](replications/exp8/prompts/message.txt), [source manifest](replications/exp8/sources.json) and [task decisions](docs/TASK.md). Actual runtime choices and deviations are recorded below and with each result.

- **Editor:** `gpt-6-astra`, reasoning effort `max`, through the Responses API. It receives the authors' [one-line instructions](replications/exp8/prompts/instructions.md), task message and text-block interface. It has no built-in tools or access to the authors' findings.
- **Model:** the text part of `Qwen/Qwen3.5-0.8B`, pinned to [revision `2fc06364715b967f1860aea9cf38778875588b17`](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).
- **Kernel:** persistent Python over float32 weights, with a [96 GiB memory limit](docs/TASK.md), no GPU, no network and no secrets. The model and practice paths are read-only. Edits to the final norm, embedding and output layer, or whole-tensor rescaling, are refused.
- **Tools and run limits:** [50 forward passes and at most 300 Python runs](replications/exp8/prompts/message.txt). A generation pass has up to [512 answers, or 256 per model for a paired comparison](replications/exp8/prompts/message.txt). A teacher-forced pass returns log probabilities, hidden states, MLP activations or attention under the authors' limits.
- **Practice:** the released GSM8K train, MATH train and AIME files, with [7,469, 7,275 and 714 problems respectively](replications/exp8/prompts/message.txt). The editor never receives benchmark problems or calibration results.
- **Evaluation:** the authors' [`think35` protocol](replications/exp8/vendor/evaluate.py): thinking on, temperature [0.6](replications/exp8/vendor/evaluate.py), top-p [0.95](replications/exp8/vendor/evaluate.py), top-k [20](replications/exp8/vendor/evaluate.py), presence penalty [1.5](replications/exp8/vendor/evaluate.py), and up to [32,768 new tokens](replications/exp8/vendor/evaluate.py). Math uses the last boxed answer and math-verify. IFEval uses strict prompt-level grading after the thinking block.
- **Final samples per problem:** [MATH-500: 4; GSM8K: 2; IFEval: 2](replications/exp8/vendor/evaluate.py). Calibration and our required controls begin with [one sample per problem](docs/TASK.md).
- **Spending limits:** [OpenAI initially $100, followed by $160.85 additional for this attempt with a separate $160.85 Plan B reserve](replications/exp8/authorizations/editor-budget-20261103.json); [Modal: the verified billing-cycle allowance, up to $150 total after an observed limit increase](docs/TASK.md). Job estimates precede launches; token usage and compute time are recorded.

### Results

These tables are generated by `openwsp report exp8` from the [authors' reference files](replications/exp8/reference) and [our evaluation files](replications/exp8/results). Missing measurements remain pending. The source link for each number is its evidence.

<!-- OPENWSP:RESULTS:START -->
| Recorded component | State | Evidence |
| --- | --- | --- |
| Required live run | [blocked](replications/exp8/results/live-run/blocker.json) | [live goal unmet](replications/exp8/results/live-run/blocker.json); [editor_budget_admission](replications/exp8/results/live-run/blocker.json) |
| Live pass requirement | [4](replications/exp8/results/live-run/blocker.json) / [50](replications/exp8/results/live-run/blocker.json) | Recorded turn: [26](replications/exp8/results/live-run/blocker.json) |
| Live final evaluation | [not_started_no_completed_50_pass_submission](replications/exp8/results/live-run/blocker.json) | See live-run record. |
| Recorded live blocker | [BudgetExceeded: editor stage editor:t027: reserve $30.478625, only $21.7072350 available](replications/exp8/results/live-run/blocker.json) | See live-run record. |
| Pipeline | [stage_done](replications/exp8/results/pipeline-state.json) | [publication](replications/exp8/results/pipeline-state.json) |
| Driver recovery | [blocked](replications/exp8/results/coordinator.json) | [all](replications/exp8/results/coordinator.json) |
| Recorded blocker | [Pipeline blocked; previous owner reconciled and automatic restart disabled](replications/exp8/results/coordinator.json) | See source record. |
| Calibration prompt audit | [passed](replications/exp8/results/calibration-prompt-audit.json) | [calibration-prompt-audit.json](replications/exp8/results/calibration-prompt-audit.json) |
| Calibration comparison | [passed](replications/exp8/results/calibration-check.json) | [calibration-check.json](replications/exp8/results/calibration-check.json) |
| Tensor hashes | [passed](replications/exp8/results/kernel/hash-verification.json) | [kernel/hash-verification.json](replications/exp8/results/kernel/hash-verification.json) |
| Sandbox and first cell | [passed](replications/exp8/results/kernel/sandbox-verification.json) | [kernel/sandbox-verification.json](replications/exp8/results/kernel/sandbox-verification.json) |
| Forward service | [passed](replications/exp8/results/kernel/forward-verification.json) | [kernel/forward-verification.json](replications/exp8/results/kernel/forward-verification.json) |
| Weight swap check | [fallback](replications/exp8/results/kernel/weight-swap.json) | [kernel/weight-swap.json](replications/exp8/results/kernel/weight-swap.json) |
| Recorded diagnostic | [Engine does not expose a compatible weight-reload contract](replications/exp8/results/kernel/weight-swap.json) | See source record. |
| Live editor | [incomplete](replications/exp8/results/live-run/state.json) | passes: [4](replications/exp8/results/live-run/state.json); turn: [26](replications/exp8/results/live-run/state.json); runs: [10](replications/exp8/results/live-run/state.json) |

Completed controls or publication do not satisfy the blocked live-run goal.

States are recorded snapshots; use `openwsp status` to check current liveness.

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
<!-- OPENWSP:RESULTS:END -->

### Controls — ours

These controls are additions to the replication, not results reported by the authors. Their definitions are fixed before their results are read; see the [protocol](docs/TASK.md).

| Question | Control | Evidence |
| --- | --- | --- |
| Is the baseline broken by the thinking setup? | Base model with thinking off and the model card's recommended non-thinking sampling. | `control1_nothink` in the [generated results](#results). |
| Is it a mechanical direction removal? | Difference of mean MLP outputs for looping versus finished base answers, removed from the same layers and columns in the same form as the published p50c edit. | `control2_mechanical` in the [generated results](#results). |
| Is it a decoding fix? | Base model with repetition penalty [1.1](docs/TASK.md), everything else unchanged. | `control3_repetition` in the [generated results](#results). |
| Does the direction matter? | Published p50c edit with a random unit vector replacing its direction. | Optional; only claimed when its result record is present. |
| Is looping specific to vLLM? | Base model in another engine with the same sampling on the prescribed subset. | Optional; only claimed when its result record is present. |

The mechanical control uses calibration generations to construct its direction. Its provenance must identify that data reuse; it is not a separately held-out discovery experiment.

### What happened during the run

The [readable live-run transcript](replications/exp8/results/live-run/transcript.md) records each available turn's explanation, code or forward request, and returned result. Incomplete runs are labeled, and failed requests remain in the record. Full generations and activations stay on the volume, with paths and hashes in the compact records.

No final live edit was submitted; [the recorded run remains blocked](replications/exp8/results/live-run/blocker.json). We did not select among the editor's candidates or change its live weights. Before the live run, [kernel verification](STATUS.md) established the published p50c tensor hashes and ran the authors' first Python cell unmodified.

The live run stopped at its [editor budget admission check](replications/exp8/results/live-run/blocker.json). A later audit found [seven output-limit responses](replications/exp8/results/live-run/editor-format-loop.json) that wrote a complete fenced action and then repeated backticks and blank lines. Their trailing formatting loops contain about [49,000–213,000 characters each](replications/exp8/results/live-run/editor-format-loop.json). None of these incomplete responses was executed. Raising the output limit did not solve the loop; retaining those responses also enlarged subsequent context.

### Cost and time

Cost records distinguish API usage charges, resource estimates and unresolved reservations. The linked files state what each duration measures. Resource time summed across jobs is not end-to-end wall time.

<!-- OPENWSP:COSTS:START -->
| Stage | Benchmark | Recorded hours | Recorded cost (USD) |
| --- | --- | --- | --- |
| calibration_base | math500 | [0.465](replications/exp8/results/calibration_base/summary.json) | [2.1280](replications/exp8/results/calibration_base/summary.json) |
| calibration_p50c | math500 | [0.259](replications/exp8/results/calibration_p50c/summary.json) | [1.2512](replications/exp8/results/calibration_p50c/summary.json) |
| control1_nothink | math500 | [0.236](replications/exp8/results/control1_nothink/math500/summary.json) | [1.0059](replications/exp8/results/control1_nothink/math500/summary.json) |
| control2_mechanical | math500 | [0.447](replications/exp8/results/control2_mechanical/math500/summary.json) | [1.9082](replications/exp8/results/control2_mechanical/math500/summary.json) |
| control3_repetition | math500 | [0.514](replications/exp8/results/control3_repetition/math500/summary.json) | [2.1923](replications/exp8/results/control3_repetition/math500/summary.json) |

Costs and durations come from the linked records. Missing costs are not treated as zero; rows are not summed because stages may overlap.

Budget ledger: [source](replications/exp8/results/costs.json). Reservations are retained bounds, not charges.

| Budget / stage / operation | State | Receipts | Recorded hours | Settled amount (USD) | Reserved bound (USD) | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| editor / editor turns | [settled](replications/exp8/results/costs.json) | [26](replications/exp8/results/costs.json) | [1.815](replications/exp8/results/costs.json) | [78.2928](replications/exp8/results/costs.json) | pending | [API usage and recorded token rates](replications/exp8/results/costs.json) |
| modal / all | [settled](replications/exp8/results/costs.json) | [4](replications/exp8/results/costs.json) | [3.457](replications/exp8/results/costs.json) | [0.2741](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
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
| modal / live / generation / edited | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.111](replications/exp8/results/costs.json) | [0.4719](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / generation / original | [settled](replications/exp8/results/costs.json) | [3](replications/exp8/results/costs.json) | [0.637](replications/exp8/results/costs.json) | [2.7181](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live / teacher-forced | [settled](replications/exp8/results/costs.json) | [2](replications/exp8/results/costs.json) | [0.079](replications/exp8/results/costs.json) | [0.3373](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / prepare | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.010](replications/exp8/results/costs.json) | [0.0016](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / prepare-v2 | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.0060](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / publish | [reserved](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | pending | pending | [1.9060](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / sandbox:heartbeat | [settled](replications/exp8/results/costs.json) | [188](replications/exp8/results/costs.json) | [3.060](replications/exp8/results/costs.json) | [7.9188](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / sandbox:startup | [settled](replications/exp8/results/costs.json) | [4](replications/exp8/results/costs.json) | [0.015](replications/exp8/results/costs.json) | [0.0400](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / verify / attention | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1626](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify / fresh-reference | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.086](replications/exp8/results/costs.json) | [0.3677](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify / native-swap | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.097](replications/exp8/results/costs.json) | [0.4123](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify / teacher-forced | [settled](replications/exp8/results/costs.json) | [1](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1635](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |

Repeated requests, attempts and batches are grouped by stage and operation; the linked ledger retains each receipt. Ledger rows and evaluation summaries may describe the same jobs; do not add the two tables. Summed resource hours are not end-to-end wall time. Settled resource estimates remain estimates until reconciled with provider billing.
<!-- OPENWSP:COSTS:END -->

### FAQ

**Is the baseline broken?** We reproduce the low thinking baseline: [7.2%](replications/exp8/results/calibration_base/summary.json). The same base weights reach [46.8% with thinking off](replications/exp8/results/control1_nothink/math500/summary.json), and the token-limit share falls from [93.2%](replications/exp8/results/calibration_base/summary.json) to [0.6%](replications/exp8/results/control1_nothink/math500/summary.json). This control changes both thinking mode and sampling to the model card's non-thinking recommendations; it does not isolate either change alone.

**Is it abliteration?** A mechanical direction computed from mean activations reaches [46.0%](replications/exp8/results/control2_mechanical/math500/summary.json), compared with [50.6% for the authors' p50c edit](replications/exp8/results/calibration_p50c/summary.json). It uses the same changed columns and projection form. The [method](replications/exp8/results/control2_mechanical/method.json) derives its direction from the calibration answers on this benchmark, so this is a diagnostic comparison, not a held-out estimate or proof that the two edits work identically.

**Is it a decoding fix?** Adding repetition penalty [1.1](replications/exp8/results/control3_repetition/math500/summary.json) with the base weights and other thinking settings fixed raises accuracy to [23.8%](replications/exp8/results/control3_repetition/math500/summary.json). The token-limit share remains [75.6%](replications/exp8/results/control3_repetition/math500/summary.json). This decoding change helps, but does not reproduce the [50.6%](replications/exp8/results/calibration_p50c/summary.json) calibrated p50c result.

**What tools did the editor have?** The kernel, practice files and bounded forward-pass interface listed above. It does not see test sets, the reference edit, calibration outcomes, our control results or the internet. The live run uses the authors' text-block protocol; the public MCP interface is a separate way to use the same core operations.

**What did it cost?** The editor spent [$78.292765](replications/exp8/results/live-run/editor-cost-analysis.json) before the next response could no longer be admitted under the cap. The seven incomplete requests account for [$60.234243 including input](replications/exp8/results/live-run/editor-cost-analysis.json). The generated cost table records compute separately. The [budget policy](docs/TASK.md) is a spending limit, not a reported cost.

### Differences from the authors' setup

| Component | Difference and evidence |
| --- | --- |
| Harness | The original Experiment 8 driver, kernel and forward server were not released. OpenWSP reconstructs the documented contract from [their release](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math). |
| Editor transport | Responses API with response chaining instead of the Codex CLI. API usage is recorded per response. The output ceiling changed from [32,768 to the documented 128,000 after the first truncated turn](replications/exp8/results/live-run/editor-output-limit.json). The captured incomplete responses were charged, retained in history and never executed. The [CPU-tested API fallback](docs/editor-recovery.md) detects the recorded formatting loop and requests cancellation; it has not processed a live provider stream. Codex subscription feasibility is being checked before further API spending. |
| Sandbox | Modal Sandbox instead of the authors' local isolation. The memory limit follows the authors' evidence. See [architecture](docs/architecture.md) for what the kernel and OS enforce. |
| Inference engine, GPU and dependency versions | The generated runtime table links each recorded version and GPU to its result provenance. Missing metadata remains pending. |
| Weight updates | The production native-update check found an incompatible vLLM interface. This run uses the specified [fresh-engine fallback](replications/exp8/results/kernel/weight-swap.json), loading the submitted checkpoint for each pass. |
| Scheduling | Subsequent jobs cap active vLLM sequences at [128](DECISIONS.md), based on observed KV-cache capacity. Final evaluations use batches of [256 generations](replications/exp8/evaluation.py). Compilation caches persist between containers; GPUs scale down between editor turns. |
| Refusal implementation | The authors did not release the numerical whole-tensor scaling test. OpenWSP's explicit tolerance and refusal tests are documented in [the kernel implementation](src/openwsp/core.py) and [decisions](DECISIONS.md). |
| Sampling and seeds | Preserve the published sampling settings. The original per-problem seed algorithm was not released. Live passes use the [first eight SHA-256 hex digits of the problem ID, modulo 2^31](src/openwsp/backends/modal_app.py), identically for both models and repeated requests. Calibration uses the published seed. |
| Base-generation reuse | Only the budget fallback authorized in [the task](docs/TASK.md) may reuse repeated problem/seed generations. Any use must be disclosed in the affected result's provenance. |

## Quickstart

Install from this repository:

```sh
git clone https://github.com/openwsp/openwsp.git
cd openwsp
uv sync --locked --extra dev --extra modal --extra replication
uv run openwsp --help
uv run pytest -q
```

For ChatGPT subscription billing, follow the [subscription setup](docs/subscription-replication.md): a persistent local Codex worker supplies editor responses while Modal runs the experiment. The [editor architecture](docs/editor-architecture.md) keeps this route, the existing Responses API adapter and the Fable profile separate from the kernel.

Set the environment variables listed in [.env.example](.env.example) using your environment's secret mechanism. Both routes use Modal for compute, Hugging Face for model access, and GitHub for the authorized result push. `OPENAI_API_KEY` is needed only for the Responses API route. Keep credentials out of repository files and remote URLs.

Before deploying, create its named secrets in the same Modal workspace environment. This command reads the exported values at runtime, maps `GITHUB_ACCESS_TOKEN` to the remote `GITHUB_TOKEN` key, and leaves existing secrets unchanged:

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

Existing secret values are retained; update them in Modal if their credentials have changed. See [Modal's secret setup](https://modal.com/docs/guide/secrets). The local CPU examples below require no credentials.

Launch or resume with the private subscription configuration from the setup guide (Modal compute is still paid), inspect its state, and regenerate its report:

```sh
uv run openwsp replicate exp8 --config /absolute/private/codex-exp8.json --foreground
uv run openwsp status exp8
uv run openwsp report exp8
```

Current progress and exact resume commands are in [STATUS.md](STATUS.md).

The original `uv run openwsp replicate exp8` command retains the Responses API route and its API-token billing; it requires the `openai` Modal secret. No automatic switch between billing routes occurs.

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

The factory implements the forward operation for that model; tensor loading alone does not configure inference. The [architecture guide](docs/architecture.md) shows the Python SDK and shared operations. The [MCP guide](docs/mcp.md) shows setup for Codex, Claude Code and Cursor. The [edit format](docs/edit-format.md) defines portable, verified diffs.

For a different editor, the optional [Anthropic Messages adapter](docs/anthropic-editor.md) implements the same `complete(messages)` interface with durable history and usage accounting. It requires explicit model prices and token ceilings; its tests use mocked responses. Experiment 8 continues to use Astra.

## What we did not do

We do not train or compute gradients, give the live editor benchmark data, choose its winning edit, or change the protocol to pursue a score. Experiments before Experiment 8 have layout placeholders only. There is no hosted service, web UI, package-registry publication, or edit intended to remove safety behavior. The optional controls and extra evaluations are not claimed complete while their records are absent.

## Add a result

Results are evidence files, not hand-edited table cells. Add a run directory under [replications/exp8/results](replications/exp8/results), with its pinned configuration, compact sample records, summaries, edit identity, costs and deviations. Run `openwsp report exp8` and submit the files and regenerated table in a pull request. Include every run, including failed, incomplete and lower-scoring runs. See the [result schema](docs/results-schema.md).

## Dual use

The method that removes a loop can remove safety behavior too. This repository ships no such edits. The Experiment 8 base model is pinned. OpenWSP v1 edits are SHA-256-addressed diffs that validate only the listed tensors. Callers must separately verify the complete base checkpoint and revision; optional model/revision labels do not enforce either.

OpenWSP code is [MIT licensed](LICENSE). The authors' scripts, logs and edits retain their respective licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
