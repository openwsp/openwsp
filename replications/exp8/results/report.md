# Recorded evaluation results

| Recorded component | State | Evidence |
| --- | --- | --- |
| Required live run | [live goal unmet](live-runs/codex-20261103/state.json) | Active attempt: [codex-20261103](active-live.json); submission requirement pending |
| Pipeline | [running](pipeline-state.json) | [live](pipeline-state.json) |
| Driver recovery | [running](coordinator.json) | [all](coordinator.json) |
| Calibration prompt audit | [passed](calibration-prompt-audit.json) | [calibration-prompt-audit.json](calibration-prompt-audit.json) |
| Calibration comparison | [passed](calibration-check.json) | [calibration-check.json](calibration-check.json) |
| Tensor hashes | [passed](kernel/hash-verification.json) | [kernel/hash-verification.json](kernel/hash-verification.json) |
| Sandbox and first cell | [passed](kernel/sandbox-verification.json) | [kernel/sandbox-verification.json](kernel/sandbox-verification.json) |
| Forward service | [passed](kernel/forward-verification.json) | [kernel/forward-verification.json](kernel/forward-verification.json) |
| Weight swap check | [fallback](kernel/weight-swap.json) | [kernel/weight-swap.json](kernel/weight-swap.json) |
| Recorded diagnostic | [Engine does not expose a compatible weight-reload contract](kernel/weight-swap.json) | See source record. |

Live attempts:

| Attempt | Selection | Editor | Recorded state | Passes | Turns | Runs | Submission | Final evaluations | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| live-run | historical | pending | [blocked](live-run/blocker.json) | [4](live-run/state.json) | [26](live-run/state.json) | [10](live-run/state.json) | not established | math500: pending; gsm8k: pending; ifeval: pending | [state.json](live-run/state.json); [blocker.json](live-run/blocker.json); [transcript.md](live-run/transcript.md) |
| live-runs/codex-20261103 | active | [codex](live-runs/codex-20261103/identity.json) | [incomplete](live-runs/codex-20261103/state.json) | [38](live-runs/codex-20261103/state.json) | [101](live-runs/codex-20261103/state.json) | [52](live-runs/codex-20261103/state.json) | not established | math500: pending; gsm8k: pending; ifeval: pending | [state.json](live-runs/codex-20261103/state.json); [identity.json](live-runs/codex-20261103/identity.json); [blocker.json](live-runs/codex-20261103/blocker.json); [transcript.md](live-runs/codex-20261103/transcript.md) |

The active pointer selects the current attempt. Historical blockers and completed controls do not establish its completion.

States are recorded snapshots; use `openwsp status` to check current liveness.

Authors' released evaluations:

| Run | MATH-500 | GSM8K | IFEval | MATH-500 finished accuracy | MATH-500 finished samples | MATH-500 hit token limit |
| --- | --- | --- | --- | --- | --- | --- |
| unedited | [7.9%](../reference/unedited.json) | [13.8%](../reference/unedited.json) | [21.0%](../reference/unedited.json) | pending | pending | [93.0%](../reference/unedited.json) |
| p10c | [12.3%](../reference/p10c.json) | [25.7%](../reference/p10c.json) | [49.0%](../reference/p10c.json) | pending | pending | [87.2%](../reference/p10c.json) |
| p25c | [14.1%](../reference/p25c.json) | [25.1%](../reference/p25c.json) | [39.9%](../reference/p25c.json) | pending | pending | [86.4%](../reference/p25c.json) |
| p50c | [48.1%](../reference/p50c.json) | [67.1%](../reference/p50c.json) | [62.3%](../reference/p50c.json) | pending | pending | [33.1%](../reference/p50c.json) |

OpenWSP evaluations and controls:

| Stage | Model | Benchmark | Author comparator | Accuracy | Finished accuracy | Hit token limit | Finished samples | Standard error | Samples | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration_base | Qwen/Qwen3.5-0.8B | math500 | unedited: [7.9%](../reference/unedited.json) | [7.2%](calibration_base/summary.json) | [100.0%](calibration_base/summary.json) | [93.2%](calibration_base/summary.json) | [34](calibration_base/summary.json) | [1.16%](calibration_base/summary.json) | [500](calibration_base/summary.json) | [done](calibration_base/summary.json) |
| calibration_p50c | Qwen/Qwen3.5-0.8B | math500 | p50c: [48.1%](../reference/p50c.json) | [50.6%](calibration_p50c/summary.json) | [73.3%](calibration_p50c/summary.json) | [31.0%](calibration_p50c/summary.json) | [345](calibration_p50c/summary.json) | [2.24%](calibration_p50c/summary.json) | [500](calibration_p50c/summary.json) | [done](calibration_p50c/summary.json) |
| control1_nothink | Qwen/Qwen3.5-0.8B | math500 | not applicable | [46.8%](control1_nothink/math500/summary.json) | [47.1%](control1_nothink/math500/summary.json) | [0.6%](control1_nothink/math500/summary.json) | [497](control1_nothink/math500/summary.json) | [2.23%](control1_nothink/math500/summary.json) | [500](control1_nothink/math500/summary.json) | [done](control1_nothink/math500/summary.json) |
| control2_mechanical | Qwen/Qwen3.5-0.8B | math500 | not applicable | [46.0%](control2_mechanical/math500/summary.json) | [83.2%](control2_mechanical/math500/summary.json) | [45.2%](control2_mechanical/math500/summary.json) | [274](control2_mechanical/math500/summary.json) | [2.23%](control2_mechanical/math500/summary.json) | [500](control2_mechanical/math500/summary.json) | [done](control2_mechanical/math500/summary.json) |
| control3_repetition | Qwen/Qwen3.5-0.8B | math500 | not applicable | [23.8%](control3_repetition/math500/summary.json) | [96.7%](control3_repetition/math500/summary.json) | [75.6%](control3_repetition/math500/summary.json) | [122](control3_repetition/math500/summary.json) | [1.90%](control3_repetition/math500/summary.json) | [500](control3_repetition/math500/summary.json) | [done](control3_repetition/math500/summary.json) |

Only recorded values are shown. Missing fields remain pending; partial runs retain their recorded status.
Finished accuracy uses only samples with `finish_reason=stop`; token-limited samples remain in all-sample accuracy. It is undefined when no answers finish, and must be read with the cap-hit rate because the finished subset can change.

Calibration comparison:

| Condition | Accuracy | Reference | Gap (points) | Within prescribed tolerance |
| --- | --- | --- | --- | --- |
| calibration_base | [7.2%](calibration-check.json) | [7.9%](calibration-check.json) | [-0.70](calibration-check.json) | [True](calibration-check.json) |
| calibration_p50c | [50.6%](calibration-check.json) | [48.1%](calibration-check.json) | [2.50](calibration-check.json) | [True](calibration-check.json) |

Recorded runtime and uncertainty:

| Stage / benchmark | Engine | GPU | Transformers | Uncertainty method |
| --- | --- | --- | --- | --- |
| calibration_base / math500 | [vllm==0.33.0](calibration_base/summary.json) | [H100](calibration_base/summary.json) | [5.17.0](calibration_base/summary.json) | pending |
| calibration_p50c / math500 | [vllm==0.33.0](calibration_p50c/summary.json) | [H100](calibration_p50c/summary.json) | [5.17.0](calibration_p50c/summary.json) | pending |
| control1_nothink / math500 | [vllm==0.33.0](control1_nothink/math500/summary.json) | [H100](control1_nothink/math500/summary.json) | [5.17.0](control1_nothink/math500/summary.json) | [binomial_per_sample](control1_nothink/math500/summary.json) |
| control2_mechanical / math500 | [vllm==0.33.0](control2_mechanical/math500/summary.json) | [H100](control2_mechanical/math500/summary.json) | [5.17.0](control2_mechanical/math500/summary.json) | [binomial_per_sample](control2_mechanical/math500/summary.json) |
| control3_repetition / math500 | [vllm==0.33.0](control3_repetition/math500/summary.json) | [H100](control3_repetition/math500/summary.json) | [5.17.0](control3_repetition/math500/summary.json) | [binomial_per_sample](control3_repetition/math500/summary.json) |

### CPU analysis and implementation checks

| Component | Recorded status | Evidence count | Scope |
| --- | --- | --- | --- |
| damage_cpu_validation | [validated](damage_cpu_validation/summary.json) | [6/6 checks passed](damage_cpu_validation/summary.json) | [Analytic CPU array validation only; no model benchmark, training or GPU calls](damage_cpu_validation/summary.json) |
| eval_grid_cpu_validation | [validated](eval_grid_cpu_validation/summary.json) | [7/7 checks passed](eval_grid_cpu_validation/summary.json) | [Tiny implementation validation only; not a model benchmark.](eval_grid_cpu_validation/summary.json) |
| Edit inspection | [measured_cpu_only](inspection/summary.json) | [4 tensor measurements](inspection/summary.json) | [Pinned weights and saved edits; no generation or GPU job](inspection/summary.json) |
| forward_bank_cpu | [passed](forward-bank-cpu.json) | [4/4 checks passed](forward-bank-cpu.json) | [Synthetic CPU cache verification; no scientific model inference or provider calls](forward-bank-cpu.json) |

These records establish the stated CPU measurements or implementation checks. They are not GPU benchmark results or a completed live replication.

Mechanical repair builds:

| Variant | Coefficient | Column rule | Build | Changed entries | Evaluation | Accuracy | Finished accuracy | Hit token limit | Finished samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2a | [1.0](repair/summary.json) | [all](repair/summary.json) | [built](repair/summary.json) | [6289541](repair/summary.json) | [pending](repair/summary.json) | pending | pending | pending | pending |
| 2b | [6.0](repair/summary.json) | [all](repair/summary.json) | [built](repair/summary.json) | [7077322](repair/summary.json) | [pending](repair/summary.json) | pending | pending | pending | pending |
| 2c | [6.0](repair/summary.json) | [topk](repair/summary.json) | [built](repair/summary.json) | [130641](repair/summary.json) | [pending](repair/summary.json) | pending | pending | pending | pending |

Building an edit does not establish its benchmark effect. Unavailable inputs and unrun evaluations remain explicit.

### Per-edit neutral-reference KL

| Record | Edit | Measurement scope | KL(base ∥ edited), nats/token | Scored tokens | Reference sequences | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [damage_cpu_validation](damage_cpu_validation/summary.json) | [synthetic arrays](damage_cpu_validation/summary.json) | [Synthetic CPU validation; not a model measurement](damage_cpu_validation/summary.json) | [0.23280366](damage_cpu_validation/summary.json) | [2](damage_cpu_validation/summary.json) | pending | [validated](damage_cpu_validation/summary.json) |

KL is the sum of base-to-edited token divergences divided by scored tokens, in nats. The registered neutral-reference measurement excludes prompt and padding positions and its identity smoke. Different attempts, edits and reference sets remain separate. Synthetic CPU values validate the reducer only; missing model measurements remain pending.

### Independent results

Published results under their recorded conditions. Rows with different sampling, token caps, engines or benchmarks do not estimate the same effect. These are reference-checkpoint reruns and controls, not completed OpenWSP live-edit results.

| Source / condition | Benchmark / samples | Thinking / token cap | Engine | Accuracy | Finished accuracy | Hit token limit | Finished samples | Comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Authors / unedited ([source](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md#L38-L45)) | math500 / [2000](analysis/independent-results.json) | [on / 32768](analysis/independent-results.json) | [vLLM (version not reported)](analysis/independent-results.json) | [7.9%](analysis/independent-results.json) | pending | [93.0%](analysis/independent-results.json) | pending | [Original think35 evaluation; full termination records unavailable](analysis/independent-results.json) |
| Authors / p50c ([source](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md#L38-L45)) | math500 / [2000](analysis/independent-results.json) | [on / 32768](analysis/independent-results.json) | [vLLM (version not reported)](analysis/independent-results.json) | [48.1%](analysis/independent-results.json) | pending | [33.1%](analysis/independent-results.json) | pending | [Original think35 evaluation; full termination records unavailable](analysis/independent-results.json) |
| akhaliq / published p50c ([source](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming/blob/d1671a178705bcab7644806d23c535517f35723b/eval/scores_edited_math500.json)) | math500 / [2000](analysis/independent-results.json) | [on / 32768](analysis/independent-results.json) | [vLLM 0.33.0](analysis/independent-results.json) | [51.0%](analysis/independent-results.json) | pending | [31.6%](analysis/independent-results.json) | pending | [Matched think35 sampling/cap and vLLM version; different samples, Transformers and GPU](analysis/independent-results.json) |
| hedwigai / unedited ([source](https://www.hedwigai.com/blog/weight-edit-math-or-loops)) | math500 / [2000](analysis/independent-results.json) | [off / 8192](analysis/independent-results.json) | [vLLM (version not reported)](analysis/independent-results.json) | [43.6%](analysis/independent-results.json) | pending | pending | pending | [Changed thinking, sampling and cap; differs from OpenWSP control 1](analysis/independent-results.json) |
| hedwigai / published p50c ([source](https://www.hedwigai.com/blog/weight-edit-math-or-loops)) | math500 / [2000](analysis/independent-results.json) | [off / 8192](analysis/independent-results.json) | [vLLM (version not reported)](analysis/independent-results.json) | [45.2%](analysis/independent-results.json) | pending | pending | pending | [Paired with Hedwig base; changed thinking, sampling and cap](analysis/independent-results.json) |
| hedwigai / unedited ([source](https://www.hedwigai.com/blog/weight-edit-math-or-loops)) | math500 / [1000](analysis/independent-results.json) | [forced close 8192 + answer 4096](analysis/independent-results.json) | [vLLM (version not reported)](analysis/independent-results.json) | [34.3%](analysis/independent-results.json) | pending | pending | pending | [Forced close plus answer budget; not thinking off](analysis/independent-results.json) |
| alisorcorp / base ([source](https://github.com/alisorcorp/inside-a-reasoning-loop/blob/5542f0fdf87a95778c5d4999d5948e388d8752c6/data/eval_summary.txt)) | gsm8k subset / [40](analysis/independent-results.json) | [on / 16384](analysis/independent-results.json) | [llama.cpp (version not reported)](analysis/independent-results.json) | [20.0%](analysis/independent-results.json) | pending | [7.5%](analysis/independent-results.json) | pending | [GSM8K subset; llama.cpp with early-loop abort](analysis/independent-results.json) |
| alisorcorp / alpha1 ([source](https://github.com/alisorcorp/inside-a-reasoning-loop/blob/5542f0fdf87a95778c5d4999d5948e388d8752c6/data/eval_summary.txt)) | gsm8k subset / [40](analysis/independent-results.json) | [on / 16384](analysis/independent-results.json) | [llama.cpp (version not reported)](analysis/independent-results.json) | [32.5%](analysis/independent-results.json) | pending | [12.5%](analysis/independent-results.json) | pending | [Same subset; coefficient-1 variant and early-loop abort](analysis/independent-results.json) |
| alisorcorp / p50c ([source](https://github.com/alisorcorp/inside-a-reasoning-loop/blob/5542f0fdf87a95778c5d4999d5948e388d8752c6/data/eval_summary.txt)) | gsm8k subset / [40](analysis/independent-results.json) | [on / 16384](analysis/independent-results.json) | [llama.cpp (version not reported)](analysis/independent-results.json) | [47.5%](analysis/independent-results.json) | pending | [30.0%](analysis/independent-results.json) | pending | [Same subset; p50c and early-loop abort](analysis/independent-results.json) |
| OpenWSP / unedited | math500 / [500](calibration_base/summary.json) | [on / 32768](calibration_base/summary.json) | [vllm==0.33.0](calibration_base/summary.json) | [7.2%](calibration_base/summary.json) | [100.0%](calibration_base/summary.json) | [93.2%](calibration_base/summary.json) | [34](calibration_base/summary.json) | [think35 sampling/cap; one sample, H100](analysis/independent-results.json) |
| OpenWSP / published p50c | math500 / [500](calibration_p50c/summary.json) | [on / 32768](calibration_p50c/summary.json) | [vllm==0.33.0](calibration_p50c/summary.json) | [50.6%](calibration_p50c/summary.json) | [73.3%](calibration_p50c/summary.json) | [31.0%](calibration_p50c/summary.json) | [345](calibration_p50c/summary.json) | [Same local calibration settings as base](analysis/independent-results.json) |
| OpenWSP / thinking off | math500 / [500](control1_nothink/math500/summary.json) | [off / 32768](control1_nothink/math500/summary.json) | [vllm==0.33.0](control1_nothink/math500/summary.json) | [46.8%](control1_nothink/math500/summary.json) | [47.1%](control1_nothink/math500/summary.json) | [0.6%](control1_nothink/math500/summary.json) | [497](control1_nothink/math500/summary.json) | [Thinking and sampling both changed; same local cap/engine](analysis/independent-results.json) |
| OpenWSP / mechanical control | math500 / [500](control2_mechanical/math500/summary.json) | [on / 32768](control2_mechanical/math500/summary.json) | [vllm==0.33.0](control2_mechanical/math500/summary.json) | [46.0%](control2_mechanical/math500/summary.json) | [83.2%](control2_mechanical/math500/summary.json) | [45.2%](control2_mechanical/math500/summary.json) | [274](control2_mechanical/math500/summary.json) | [think35; direction uses calibration answers and borrowed columns](analysis/independent-results.json) |
| OpenWSP / repetition penalty | math500 / [500](control3_repetition/math500/summary.json) | [on / 32768](control3_repetition/math500/summary.json) | [vllm==0.33.0](control3_repetition/math500/summary.json) | [23.8%](control3_repetition/math500/summary.json) | [96.7%](control3_repetition/math500/summary.json) | [75.6%](control3_repetition/math500/summary.json) | [122](control3_repetition/math500/summary.json) | [think35 plus repetition penalty 1.1](analysis/independent-results.json) |

[Source verification, exact settings and limitations](analysis/independent-results.json). Different thinking modes, token caps, engines, sample counts and early-stop rules are separate conditions. An unavailable finished metric is not inferred from overall accuracy and cutoff share.

### Questions about the result

**Did the prompt point at the fix?**

Yes. The verbatim task identifies repetition as an in-scope failure and names token-limit cutoffs as a diagnostic. It does not supply a direction or an edit. These are the actual sentences:

> Repairing a specific failure mechanism is in scope: for example, if the model often fails because it repeats itself and never finishes, finding and fixing what drives that repetition counts. [Source](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/message.txt#L5)

> Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for, and neither is making the model skip its thinking or close it early regardless of the problem; repairing a specific failure inside the thinking (such as repetition) is in scope. [Source](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/message.txt#L5)

> Look at how the model fails on practice problems: a wrong final answer, finishing without an answer, or being cut off at the token limit (often while repeating itself in its thinking). [Source](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/message.txt#L19)

[The published excerpt of the authors’ private results table](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/eval/RESULTS_excerpt.md#L3-L21) records [no submitted edit at 10 passes, but 14.1% MATH-500 at 25 passes](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/eval/RESULTS_excerpt.md#L20-L21) for prompt v4, labeled shorter, DEV/HOLDOUT and no failure example. [The source also records older limits: 128 answers and no paired “both” request](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/profiles.py#L109-L111). This changes several conditions and does not isolate the effect of removing the repetition example. The faithful run keeps the verbatim message.

[The source audit](analysis/authors-source-verification.json) preserves exact quotations, hashes and the limits of these comparisons.

**Is it abliteration?**

It uses a related directional weight edit, but its coefficient is not the textbook orthogonal projection. The authors recorded [W' = W0 - 6 outer(u, mask64 * (uᵀ W0))](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/tool_calls.jsonl#L150), with a separate unit direction for each target layer. [The selected 64 columns are in layers 15 and 19](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/tool_calls.jsonl#L128).

For a selected column and unit u, [uᵀw' = (1 − α)uᵀw](analysis/interpretation.json). Thus [α = 1 removes the parallel component, while α = 6 reverses it and multiplies it by 5](analysis/interpretation.json). This algebra explains the coefficient; it does not establish the behavioral mechanism.

[CPU inspection finds rank 64 after bfloat16 rounding, with over 99.9991% of each p50c delta’s energy in its leading direction](inspection/p50c-vs-control2.json). The edits are approximately rank one, not exactly rank one as stored. [The p50c layer-to-layer absolute cosine is 0.8490; its same-layer cosines with the mechanical control are 0.3293 and 0.3879](inspection/p50c-vs-control2.json).

[The limited logit lens](inspection/p50c-vs-control2.json) does not support strong alignment to the tested Wait, But, Hmm, So or Okay embedding rows, or the mean row. [Maximum named-token absolute cosines are 0.0203, 0.0390; mean-row absolute cosines are 0.0468, 0.0400](inspection/p50c-vs-control2.json). This does not rule out other token directions or a more distributed mechanism.

**What is the primary metric?**

For the mechanism question, report accuracy among finished answers together with the cap-hit rate. The faithful benchmark result remains accuracy over all samples, including capped ones. [accuracy_finished = finished_correct / finished_n](analysis/interpretation.json), where canonical `stop` samples finish and `length` samples hit the cap. An empty finished set has undefined accuracy.

In local calibration, base finished accuracy is [100.0%](calibration_base/summary.json) over [34](calibration_base/summary.json) finished samples; p50c is [73.3%](calibration_p50c/summary.json) over [345](calibration_p50c/summary.json). Their cap-hit rates are [93.2%](calibration_base/summary.json) and [31.0%](calibration_p50c/summary.json). These are different subsets, so the conditional comparison does not establish a paired accuracy regression. [The recomputation audit](analysis/finished-metrics.json) retains sample hashes.

A capped answer can still grade correct, so finished accuracy cannot be recovered by dividing overall accuracy by one minus the cap-hit rate. Full author termination records are unavailable; their finished metric remains pending.

**What does the agent add over a script?**

The released p50c edit reaches [50.6%](calibration_p50c/summary.json) versus [46.0%](control2_mechanical/math500/summary.json) for the mechanical control under the local think35 evaluation. The control borrows p50c’s column set and derives its direction from calibration answers on this benchmark. It is not a held-out discovery experiment.

The [agent-free 2c build](repair/2c/summary.json) selects its own columns from [450 looping and 34 finished base answers](repair/2c/verification.json). Its mask shares [25 of 64 columns in layer 15 and 18 of 64 in layer 19](repair/2c/column-comparison.json) with the p50c mask used by the earlier control. [All seven fixed-mask dose bundles](repair/dose-sweep.json) are built; benchmark accuracy is pending. Until that comparison and the live attempt are complete, these results show that a simple directional intervention recovers much of the reported benchmark gain, not how much an agent adds over a fully independent script.


| Stage | Benchmark | Recorded hours | Recorded cost (USD) |
| --- | --- | --- | --- |
| calibration_base | math500 | [0.465](calibration_base/summary.json) | [2.1280](calibration_base/summary.json) |
| calibration_p50c | math500 | [0.259](calibration_p50c/summary.json) | [1.2512](calibration_p50c/summary.json) |
| control1_nothink | math500 | [0.236](control1_nothink/math500/summary.json) | [1.0059](control1_nothink/math500/summary.json) |
| control2_mechanical | math500 | [0.447](control2_mechanical/math500/summary.json) | [1.9082](control2_mechanical/math500/summary.json) |
| control3_repetition | math500 | [0.514](control3_repetition/math500/summary.json) | [2.1923](control3_repetition/math500/summary.json) |

Costs and durations come from the linked records. Missing costs are not treated as zero; rows are not summed because stages may overlap.

Budget ledger: [source](costs.json). Reservations are retained bounds, not charges.

| Budget / stage / operation | State | Receipts | Recorded hours | Settled amount (USD) | Reserved bound (USD) | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| editor / editor turns | [settled](costs.json) | [26](costs.json) | [1.815](costs.json) | [78.2928](costs.json) | pending | [API usage and recorded token rates](costs.json) |
| modal / all | [reserved](costs.json) | [1](costs.json) | pending | pending | [1.9060](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / all | [settled](costs.json) | [6](costs.json) | [5.769](costs.json) | [0.4575](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
| modal / calibration | [settled](costs.json) | [1](costs.json) | [0.906](costs.json) | [0.0718](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
| modal / calibration_base | [settled](costs.json) | [1](costs.json) | [0.499](costs.json) | [2.1280](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / calibration_base_smoke | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.2758](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / calibration_base_smoke | [settled](costs.json) | [1](costs.json) | [0.105](costs.json) | [0.4496](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / calibration_p50c | [settled](costs.json) | [1](costs.json) | [0.293](costs.json) | [1.2512](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / calibration_p50c_smoke | [settled](costs.json) | [1](costs.json) | [0.099](costs.json) | [0.4228](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / control1_nothink / math500 / evaluation | [settled](costs.json) | [2](costs.json) | [0.236](costs.json) | [1.0059](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / control1_nothink / math500 / evaluation smoke | [settled](costs.json) | [1](costs.json) | [0.066](costs.json) | [0.2833](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / control2 / directions | [settled](costs.json) | [16](costs.json) | [0.681](costs.json) | [2.9042](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / control2 / directions smoke | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1613](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / control2_mechanical / math500 / evaluation | [settled](costs.json) | [2](costs.json) | [0.447](costs.json) | [1.9082](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / control2_mechanical / math500 / evaluation smoke | [settled](costs.json) | [1](costs.json) | [0.090](costs.json) | [0.3856](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / control3_repetition / math500 / evaluation | [settled](costs.json) | [2](costs.json) | [0.514](costs.json) | [2.1923](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / control3_repetition / math500 / evaluation smoke | [settled](costs.json) | [1](costs.json) | [0.091](costs.json) | [0.3880](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / controls | [settled](costs.json) | [1](costs.json) | [1.612](costs.json) | [0.1278](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
| modal / infrastructure | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.0000](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / kernel | [settled](costs.json) | [1](costs.json) | [0.205](costs.json) | [0.0163](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
| modal / kernel_verification:heartbeat | [settled](costs.json) | [2](costs.json) | [0.003](costs.json) | [0.0073](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / kernel_verification:startup | [settled](costs.json) | [1](costs.json) | [0.004](costs.json) | [0.0092](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / live / activation extraction / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / activation extraction / original | [settled](costs.json) | [1](costs.json) | [0.071](costs.json) | [0.3049](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_1 / original | [settled](costs.json) | [1](costs.json) | [0.234](costs.json) | [0.9999](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_1 / original / activations | [settled](costs.json) | [1](costs.json) | [0.105](costs.json) | [0.4491](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_10 / edited | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.9870](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_10 / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_11 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1680](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_12 / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_13 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1670](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_14 / edited | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_15 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1610](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_16 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1638](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_17 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1639](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_18 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1625](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_19 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1607](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_2 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1645](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_20 / edited | [settled](costs.json) | [1](costs.json) | [0.098](costs.json) | [0.4199](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_20 / original | [settled](costs.json) | [1](costs.json) | [0.079](costs.json) | [0.3392](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_21 / edited | [settled](costs.json) | [1](costs.json) | [0.249](costs.json) | [1.0622](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_21 / edited / activations | [settled](costs.json) | [1](costs.json) | [0.078](costs.json) | [0.3342](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_21 / original | [settled](costs.json) | [1](costs.json) | [0.276](costs.json) | [1.1781](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_21 / original / activations | [settled](costs.json) | [1](costs.json) | [0.085](costs.json) | [0.3608](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_22 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1616](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_23 / edited | [settled](costs.json) | [1](costs.json) | [0.275](costs.json) | [1.1723](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_24 / edited | [settled](costs.json) | [1](costs.json) | [0.279](costs.json) | [1.1925](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_25 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1662](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_26 / edited | [settled](costs.json) | [1](costs.json) | [0.275](costs.json) | [1.1748](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_27 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1657](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_28 / edited | [settled](costs.json) | [1](costs.json) | [0.278](costs.json) | [1.1869](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_29 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1673](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_3 / edited | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.9870](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_3 / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_30 / edited | [settled](costs.json) | [1](costs.json) | [0.143](costs.json) | [0.6122](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_30 / original | [settled](costs.json) | [1](costs.json) | [0.283](costs.json) | [1.2064](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_31 / edited | [settled](costs.json) | [1](costs.json) | [0.280](costs.json) | [1.1956](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_32 / edited | [settled](costs.json) | [1](costs.json) | [0.279](costs.json) | [1.1891](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_33 / edited | [settled](costs.json) | [1](costs.json) | [0.269](costs.json) | [1.1476](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_33 / edited / activations | [settled](costs.json) | [1](costs.json) | [0.083](costs.json) | [0.3525](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_33 / original | [settled](costs.json) | [1](costs.json) | [0.271](costs.json) | [1.1575](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_33 / original / activations | [settled](costs.json) | [1](costs.json) | [0.084](costs.json) | [0.3598](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_34 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1663](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_35 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.040](costs.json) | [0.1696](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_36 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1682](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_37 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1632](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_38 / edited | [settled](costs.json) | [1](costs.json) | [0.268](costs.json) | [1.1453](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_38 / original | [settled](costs.json) | [1](costs.json) | [0.269](costs.json) | [1.1495](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_39 / edited | [settled](costs.json) | [1](costs.json) | [0.258](costs.json) | [1.1005](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_39 / original | [settled](costs.json) | [1](costs.json) | [0.283](costs.json) | [1.2058](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_4 / edited | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.9870](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_4 / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_40 / edited | [settled](costs.json) | [1](costs.json) | [0.288](costs.json) | [1.2299](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_40 / edited / activations | [settled](costs.json) | [1](costs.json) | [0.088](costs.json) | [0.3759](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_40 / original / activations | [settled](costs.json) | [1](costs.json) | [0.135](costs.json) | [0.5750](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_41 / edited | [settled](costs.json) | [1](costs.json) | [0.291](costs.json) | [1.2436](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_41 / edited / activations | [settled](costs.json) | [1](costs.json) | [0.085](costs.json) | [0.3638](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_41 / original / activations | [settled](costs.json) | [1](costs.json) | [0.087](costs.json) | [0.3720](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_42 / edited | [settled](costs.json) | [1](costs.json) | [0.261](costs.json) | [1.1152](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_42 / edited / activations | [settled](costs.json) | [1](costs.json) | [0.084](costs.json) | [0.3568](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_42 / original / activations | [settled](costs.json) | [1](costs.json) | [0.091](costs.json) | [0.3890](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_43 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1650](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_44 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1635](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_45 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1623](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_46 / edited | [settled](costs.json) | [1](costs.json) | [0.265](costs.json) | [1.1292](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_46 / original | [settled](costs.json) | [1](costs.json) | [0.277](costs.json) | [1.1826](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_5 / edited | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.9870](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_5 / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_6 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.039](costs.json) | [0.1647](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_7 / edited | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_8 / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_9 / teacher_forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1633](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / generation / edited | [settled](costs.json) | [1](costs.json) | [0.111](costs.json) | [0.4719](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / generation / original | [settled](costs.json) | [3](costs.json) | [0.637](costs.json) | [2.7181](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / teacher-forced | [settled](costs.json) | [2](costs.json) | [0.079](costs.json) | [0.3373](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / prepare | [settled](costs.json) | [1](costs.json) | [0.010](costs.json) | [0.0016](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / prepare-v2 | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.0060](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / publish | [settled](costs.json) | [1](costs.json) | [0.134](costs.json) | [0.0107](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
| modal / sandbox:heartbeat | [settled](costs.json) | [595](costs.json) | [9.789](costs.json) | [25.3311](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / sandbox:startup | [settled](costs.json) | [7](costs.json) | [0.026](costs.json) | [0.0677](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / verify / attention | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1626](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / fresh-reference | [settled](costs.json) | [1](costs.json) | [0.086](costs.json) | [0.3677](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / native-swap | [settled](costs.json) | [1](costs.json) | [0.097](costs.json) | [0.4123](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / teacher-forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1635](costs.json) | pending | [resource estimate; not invoice](costs.json) |

Repeated requests, attempts and batches are grouped by stage and operation; the linked ledger retains each receipt. Ledger rows and evaluation summaries may describe the same jobs; do not add the two tables. Summed resource hours are not end-to-end wall time. Settled resource estimates remain estimates until reconciled with provider billing.
