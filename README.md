# OpenWSP

<!-- OPENWSP:STATUS:START -->
| Recorded component | State | Evidence |
| --- | --- | --- |
| Pipeline | [running](replications/exp8/results/pipeline-state.json) | [live](replications/exp8/results/pipeline-state.json) |
| Driver recovery | [running](replications/exp8/results/coordinator.json) | [all](replications/exp8/results/coordinator.json) |
| Calibration prompt audit | [passed](replications/exp8/results/calibration-prompt-audit.json) | [calibration-prompt-audit.json](replications/exp8/results/calibration-prompt-audit.json) |
| Calibration comparison | [passed](replications/exp8/results/calibration-check.json) | [calibration-check.json](replications/exp8/results/calibration-check.json) |
| Tensor hashes | [passed](replications/exp8/results/kernel/hash-verification.json) | [kernel/hash-verification.json](replications/exp8/results/kernel/hash-verification.json) |
| Sandbox and first cell | [passed](replications/exp8/results/kernel/sandbox-verification.json) | [kernel/sandbox-verification.json](replications/exp8/results/kernel/sandbox-verification.json) |
| Forward service | [passed](replications/exp8/results/kernel/forward-verification.json) | [kernel/forward-verification.json](replications/exp8/results/kernel/forward-verification.json) |
| Weight swap check | [fallback](replications/exp8/results/kernel/weight-swap.json) | [kernel/weight-swap.json](replications/exp8/results/kernel/weight-swap.json) |
| Recorded diagnostic | [Engine does not expose a compatible weight-reload contract](replications/exp8/results/kernel/weight-swap.json) | See source record. |
| Live editor | [incomplete](replications/exp8/results/live-run/state.json) | passes: [0](replications/exp8/results/live-run/state.json); turn: [4](replications/exp8/results/live-run/state.json); runs: [2](replications/exp8/results/live-run/state.json) |

States are recorded snapshots; use `openwsp status` to check current liveness.
<!-- OPENWSP:STATUS:END -->

OpenWSP is open tooling for weight space programming: agents read, edit and version the weights of open models. This is an independent replication of [the work of Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab](https://rikiyat.github.io/blog/weight-space-programming/), with no affiliation to its authors.

The tools are a Python kernel over model weights, a forward-pass service and a versioned edit format, exposed through one API as a CLI, Python SDK and MCP server. The test is Experiment 8: can an editor diagnose a model's math failures and repair its weights without gradients or training?

## Experiment 8

**TL;DR:** The tables below report calibration, each live run and our controls from their evidence files. Missing stages stay pending, blocked stages retain their reason, and a lower result remains next to the authors' result. A passed component check does not establish completion of the whole replication.

### Setup

The prescribed editor setup is recorded in the [task message](replications/exp8/prompts/message.txt), [source manifest](replications/exp8/sources.json) and [task decisions](docs/TASK.md). Actual runtime choices and deviations are recorded below and with each result.

- **Editor:** `gpt-6-astra`, reasoning effort `max`, through the Responses API. It receives the authors' [one-line instructions](replications/exp8/prompts/instructions.md), task message and text-block interface. It has no built-in tools or access to the authors' findings.
- **Model:** the text part of `Qwen/Qwen3.5-0.8B`, pinned to [revision `2fc06364715b967f1860aea9cf38778875588b17`](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17).
- **Kernel:** persistent Python over float32 weights, with a [96 GiB memory limit](docs/TASK.md), no GPU, no network and no secrets. The model and practice paths are read-only. Edits to the final norm, embedding and output layer, or whole-tensor rescaling, are refused.
- **Tools and run limits:** [50 forward passes and at most 300 Python runs](replications/exp8/prompts/message.txt). A generation pass has up to [512 answers, or 256 per model for a paired comparison](replications/exp8/prompts/message.txt). A teacher-forced pass returns log probabilities, hidden states, MLP activations or attention under the authors' limits.
- **Practice:** the released GSM8K train, MATH train and AIME files, with [7,469, 7,275 and 714 problems respectively](replications/exp8/prompts/message.txt). The editor never receives benchmark problems or calibration results.
- **Evaluation:** the authors' [`think35` protocol](replications/exp8/vendor/evaluate.py): thinking on, temperature [0.6](replications/exp8/vendor/evaluate.py), top-p [0.95](replications/exp8/vendor/evaluate.py), top-k [20](replications/exp8/vendor/evaluate.py), presence penalty [1.5](replications/exp8/vendor/evaluate.py), and up to [32,768 new tokens](replications/exp8/vendor/evaluate.py). Math uses the last boxed answer and math-verify. IFEval uses strict prompt-level grading after the thinking block.
- **Final samples per problem:** [MATH-500: 4; GSM8K: 2; IFEval: 2](replications/exp8/vendor/evaluate.py). Calibration and our required controls begin with [one sample per problem](docs/TASK.md).
- **Spending limits:** [OpenAI: $100; Modal: the verified billing-cycle allowance, up to $150 total for this task after an observed limit increase](docs/TASK.md). Job estimates precede launches; token usage and compute time are recorded.

### Results

These tables are generated by `openwsp report exp8` from the [authors' reference files](replications/exp8/reference) and [our evaluation files](replications/exp8/results). Missing measurements remain pending. The source link for each number is its evidence.

<!-- OPENWSP:RESULTS:START -->
| Recorded component | State | Evidence |
| --- | --- | --- |
| Pipeline | [running](replications/exp8/results/pipeline-state.json) | [live](replications/exp8/results/pipeline-state.json) |
| Driver recovery | [running](replications/exp8/results/coordinator.json) | [all](replications/exp8/results/coordinator.json) |
| Calibration prompt audit | [passed](replications/exp8/results/calibration-prompt-audit.json) | [calibration-prompt-audit.json](replications/exp8/results/calibration-prompt-audit.json) |
| Calibration comparison | [passed](replications/exp8/results/calibration-check.json) | [calibration-check.json](replications/exp8/results/calibration-check.json) |
| Tensor hashes | [passed](replications/exp8/results/kernel/hash-verification.json) | [kernel/hash-verification.json](replications/exp8/results/kernel/hash-verification.json) |
| Sandbox and first cell | [passed](replications/exp8/results/kernel/sandbox-verification.json) | [kernel/sandbox-verification.json](replications/exp8/results/kernel/sandbox-verification.json) |
| Forward service | [passed](replications/exp8/results/kernel/forward-verification.json) | [kernel/forward-verification.json](replications/exp8/results/kernel/forward-verification.json) |
| Weight swap check | [fallback](replications/exp8/results/kernel/weight-swap.json) | [kernel/weight-swap.json](replications/exp8/results/kernel/weight-swap.json) |
| Recorded diagnostic | [Engine does not expose a compatible weight-reload contract](replications/exp8/results/kernel/weight-swap.json) | See source record. |
| Live editor | [incomplete](replications/exp8/results/live-run/state.json) | passes: [0](replications/exp8/results/live-run/state.json); turn: [4](replications/exp8/results/live-run/state.json); runs: [2](replications/exp8/results/live-run/state.json) |

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

The [readable live-run transcript](replications/exp8/results/live-run/transcript.md) records each available turn's explanation, code or forward request, and returned result. Its pending state means no editor turns have been published. Full generations and activations stay on the volume, with paths and hashes in the compact records.

The final submitted edit is the editor's choice. We do not select among its candidates or change its live weights. The [kernel verification](STATUS.md) must establish the published edit's tensor hashes and run the authors' first Python cell unmodified before the live run, or record the specified circuit breaker.

### Cost and time

Cost records distinguish API usage charges, resource estimates and unresolved reservations. The linked files state what each duration measures. Resource time summed across jobs is not end-to-end wall time.

<!-- OPENWSP:COSTS:START -->
| Stage | Benchmark | Recorded hours | Recorded cost (USD) |
| --- | --- | --- | --- |
| calibration_base | math500 | [0.465](replications/exp8/results/calibration_base/summary.json) | [2.1280](replications/exp8/results/calibration_base/summary.json) |
| calibration_p50c | math500 | [0.259](replications/exp8/results/calibration_p50c/summary.json) | [1.2512](replications/exp8/results/calibration_p50c/summary.json) |

Costs and durations come from the linked records. Missing costs are not treated as zero; rows are not summed because stages may overlap.

Budget ledger: [source](replications/exp8/results/costs.json). Reservations are retained bounds, not charges.

| Budget / stage | State | Recorded hours | Settled amount (USD) | Reserved bound (USD) | Basis |
| --- | --- | --- | --- | --- | --- |
| editor / editor turns | [settled](replications/exp8/results/costs.json) | [0.039](replications/exp8/results/costs.json) | [0.4280](replications/exp8/results/costs.json) | pending | [API usage and recorded token rates](replications/exp8/results/costs.json) |
| modal / all | [reserved](replications/exp8/results/costs.json) | pending | pending | [1.9060](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / all | [settled](replications/exp8/results/costs.json) | [0.491](replications/exp8/results/costs.json) | [0.0390](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / calibration | [settled](replications/exp8/results/costs.json) | [0.906](replications/exp8/results/costs.json) | [0.0718](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / calibration_base | [settled](replications/exp8/results/costs.json) | [0.499](replications/exp8/results/costs.json) | [2.1280](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / calibration_base_smoke | [reserved](replications/exp8/results/costs.json) | pending | pending | [2.2758](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / calibration_base_smoke | [settled](replications/exp8/results/costs.json) | [0.105](replications/exp8/results/costs.json) | [0.4496](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / calibration_p50c | [settled](replications/exp8/results/costs.json) | [0.293](replications/exp8/results/costs.json) | [1.2512](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / calibration_p50c_smoke | [settled](replications/exp8/results/costs.json) | [0.099](replications/exp8/results/costs.json) | [0.4228](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / infrastructure | [reserved](replications/exp8/results/costs.json) | pending | pending | [2.0000](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / kernel | [settled](replications/exp8/results/costs.json) | [0.205](replications/exp8/results/costs.json) | [0.0163](replications/exp8/results/costs.json) | pending | [driver_resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / kernel_verification:heartbeat | [settled](replications/exp8/results/costs.json) | [0.003](replications/exp8/results/costs.json) | [0.0073](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / kernel_verification:startup | [settled](replications/exp8/results/costs.json) | [0.004](replications/exp8/results/costs.json) | [0.0092](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / live/fwd_1/original/activations/openwsp-gpu:412d8691f5574c15a0b8d5ab47a99415:0 | [settled](replications/exp8/results/costs.json) | [0.700](replications/exp8/results/costs.json) | [2.9870](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live/fwd_1/original/openwsp-gpu:e382217120784a58b0cb6b8d80fa71e1:0 | [settled](replications/exp8/results/costs.json) | [0.388](replications/exp8/results/costs.json) | [1.6550](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / live/fwd_2/original/openwsp-gpu:6df93dd4a722479d89a783fc1cff9ddd:0 | [reserved](replications/exp8/results/costs.json) | pending | pending | [2.9870](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / prepare | [settled](replications/exp8/results/costs.json) | [0.010](replications/exp8/results/costs.json) | [0.0016](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / prepare-v2 | [settled](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.0060](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / sandbox:heartbeat | [reserved](replications/exp8/results/costs.json) | pending | pending | [0.2157](replications/exp8/results/costs.json) | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / sandbox:heartbeat | [settled](replications/exp8/results/costs.json) | [0.934](replications/exp8/results/costs.json) | [2.4163](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / sandbox:startup | [settled](replications/exp8/results/costs.json) | [0.009](replications/exp8/results/costs.json) | [0.0226](replications/exp8/results/costs.json) | pending | [resource_upper_bound_not_invoice](replications/exp8/results/costs.json) |
| modal / verify/attention/openwsp-gpu:5072ca485aba4672a75e1bda85d260bf:0 | [settled](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1626](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify/fresh-reference/openwsp-gpu:3c4cb1f5c3f949cfb7a7dfb258fcdfd9:0 | [settled](replications/exp8/results/costs.json) | [0.086](replications/exp8/results/costs.json) | [0.3677](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify/native-swap/openwsp-gpu:14322bc224b641a08e8fe7e8bd3268a7:0 | [settled](replications/exp8/results/costs.json) | [0.097](replications/exp8/results/costs.json) | [0.4123](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |
| modal / verify/teacher-forced/openwsp-gpu:7e5f28f65a2246f8ae27081745329234:0 | [settled](replications/exp8/results/costs.json) | [0.038](replications/exp8/results/costs.json) | [0.1635](replications/exp8/results/costs.json) | pending | [resource estimate; not invoice](replications/exp8/results/costs.json) |

Ledger rows and evaluation summaries may describe the same jobs; do not add the two tables. Summed resource hours are not end-to-end wall time. Settled resource estimates remain estimates until reconciled with provider billing.
<!-- OPENWSP:COSTS:END -->

### FAQ

**Is the baseline broken?** The generated calibration table shows whether this implementation reproduces the published base and edited scores, including any unresolved gap. Compare `control1_nothink` with `calibration_base` in the result table to see the effect of the thinking setup. Missing rows leave that question unanswered.

**Is it abliteration?** The published edit removes a direction associated with looping. `control2_mechanical` reports the result of computing a direction from mean activations. Compare it with the calibrated p50c edit in the generated table; a shared edit form does not by itself establish equal behavior.

**Is it a decoding fix?** `control3_repetition` reports decoding changes with the base weights fixed. Compare its accuracy and token-limit share with `calibration_base` and `calibration_p50c`. The table preserves them as separate conditions, including pending or partial evaluations.

**What tools did the editor have?** The kernel, practice files and bounded forward-pass interface listed above. It does not see test sets, the reference edit, calibration outcomes, our control results or the internet. The live run uses the authors' text-block protocol; the public MCP interface is a separate way to use the same core operations.

**What did it cost?** The generated cost table links to per-stage records. The [budget policy](docs/TASK.md) is a spending limit, not a reported cost.

### Differences from the authors' setup

| Component | Difference and evidence |
| --- | --- |
| Harness | The original Experiment 8 driver, kernel and forward server were not released. OpenWSP reconstructs the documented contract from [their release](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math). |
| Editor transport | Responses API with response chaining instead of the Codex CLI. API usage is recorded per response rather than summed from the authors' cumulative turn counters. |
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

Set the environment variables listed in [.env.example](.env.example) using your environment's secret mechanism. The replication uses Modal for compute, OpenAI for the editor, Hugging Face for model access, and GitHub for the authorized result push. Keep credentials out of repository files and remote URLs.

Launch or resume the replication, inspect its state, and regenerate its report:

```sh
uv run openwsp replicate exp8
uv run openwsp status exp8
uv run openwsp report exp8
```

Current progress and exact resume commands are in [STATUS.md](STATUS.md).

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

## What we did not do

We do not train or compute gradients, give the live editor benchmark data, choose its winning edit, or change the protocol to pursue a score. Experiments before Experiment 8 have layout placeholders only. There is no hosted service, web UI, package-registry publication, or edit intended to remove safety behavior. The optional controls and extra evaluations are not claimed complete while their records are absent.

## Add a result

Results are evidence files, not hand-edited table cells. Add a run directory under [replications/exp8/results](replications/exp8/results), with its pinned configuration, compact sample records, summaries, edit identity, costs and deviations. Run `openwsp report exp8` and submit the files and regenerated table in a pull request. Include every run, including failed, incomplete and lower-scoring runs. See the [result schema](docs/results-schema.md).

## Dual use

The method that removes a loop can remove safety behavior too. This repository ships no such edits. Every public edit is a SHA-256-addressed diff against a pinned base, so anyone can verify which weights changed.

OpenWSP code is [MIT licensed](LICENSE). The authors' scripts, logs and edits retain their respective licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
