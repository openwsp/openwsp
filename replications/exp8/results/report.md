# Recorded evaluation results

| Recorded component | State | Evidence |
| --- | --- | --- |
| Required live run | [live goal unmet](live-runs/codex-20261103/state.json) | Active attempt: [codex-20261103](active-live.json); submission requirement pending |
| Pipeline | [blocked](pipeline-state.json) | [live](pipeline-state.json) |
| Recorded blocker | [SandboxTransportError: Text transfer failed with exit code 1](pipeline-state.json) | See source record. |
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
| live-runs/codex-20261103 | active | [codex](live-runs/codex-20261103/identity.json) | [incomplete](live-runs/codex-20261103/state.json) | [14](live-runs/codex-20261103/state.json) | [50](live-runs/codex-20261103/state.json) | [26](live-runs/codex-20261103/state.json) | not established | math500: pending; gsm8k: pending; ifeval: pending | [state.json](live-runs/codex-20261103/state.json); [identity.json](live-runs/codex-20261103/identity.json); [blocker.json](live-runs/codex-20261103/blocker.json); [transcript.md](live-runs/codex-20261103/transcript.md) |

The active pointer selects the current attempt. Historical blockers and completed controls do not establish its completion.

States are recorded snapshots; use `openwsp status` to check current liveness.

Authors' released evaluations:

| Run | MATH-500 | GSM8K | IFEval | MATH-500 hit token limit |
| --- | --- | --- | --- | --- |
| unedited | [7.9%](../reference/unedited.json) | [13.8%](../reference/unedited.json) | [21.0%](../reference/unedited.json) | [93.0%](../reference/unedited.json) |
| p10c | [12.3%](../reference/p10c.json) | [25.7%](../reference/p10c.json) | [49.0%](../reference/p10c.json) | [87.2%](../reference/p10c.json) |
| p25c | [14.1%](../reference/p25c.json) | [25.1%](../reference/p25c.json) | [39.9%](../reference/p25c.json) | [86.4%](../reference/p25c.json) |
| p50c | [48.1%](../reference/p50c.json) | [67.1%](../reference/p50c.json) | [62.3%](../reference/p50c.json) | [33.1%](../reference/p50c.json) |

Our evaluations and controls (controls are ours):

| Stage | Model | Benchmark | Author comparator | Accuracy | Standard error | Samples | Hit token limit | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration_base | Qwen/Qwen3.5-0.8B | math500 | unedited: [7.9%](../reference/unedited.json) | [7.2%](calibration_base/summary.json) | [1.16%](calibration_base/summary.json) | [500](calibration_base/summary.json) | [93.2%](calibration_base/summary.json) | [done](calibration_base/summary.json) |
| calibration_p50c | Qwen/Qwen3.5-0.8B | math500 | p50c: [48.1%](../reference/p50c.json) | [50.6%](calibration_p50c/summary.json) | [2.24%](calibration_p50c/summary.json) | [500](calibration_p50c/summary.json) | [31.0%](calibration_p50c/summary.json) | [done](calibration_p50c/summary.json) |
| control1_nothink | Qwen/Qwen3.5-0.8B | math500 | not applicable | [46.8%](control1_nothink/math500/summary.json) | [2.23%](control1_nothink/math500/summary.json) | [500](control1_nothink/math500/summary.json) | [0.6%](control1_nothink/math500/summary.json) | [done](control1_nothink/math500/summary.json) |
| control2_mechanical | Qwen/Qwen3.5-0.8B | math500 | not applicable | [46.0%](control2_mechanical/math500/summary.json) | [2.23%](control2_mechanical/math500/summary.json) | [500](control2_mechanical/math500/summary.json) | [45.2%](control2_mechanical/math500/summary.json) | [done](control2_mechanical/math500/summary.json) |
| control3_repetition | Qwen/Qwen3.5-0.8B | math500 | not applicable | [23.8%](control3_repetition/math500/summary.json) | [1.90%](control3_repetition/math500/summary.json) | [500](control3_repetition/math500/summary.json) | [75.6%](control3_repetition/math500/summary.json) | [done](control3_repetition/math500/summary.json) |
| damage_cpu_validation | pending | pending | pending | pending | pending | pending | pending | [validated](damage_cpu_validation/summary.json) |
| eval_grid_cpu_validation | pending | token_state_machine | pending | pending | pending | pending | pending | [validated](eval_grid_cpu_validation/summary.json) |
| pending | {'checkpoint_hash_note': 'HF file includes vision weights; prepare extracts language-model tensors into a different safetensors artifact. Listed edited tensor base hashes were verified against both edit manifests.', 'config_sha256': 'b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204', 'downloaded_huggingface_checkpoint_sha256': '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696', 'embedding_name': 'model.language_model.embed_tokens.weight', 'id': 'Qwen/Qwen3.5-0.8B', 'prepared_text_only_checkpoint_sha256': '978661a573971be7257815c2554a28c74913c7b7ea51b0a0b9503fd11eb0d2f8', 'revision': '2fc06364715b967f1860aea9cf38778875588b17', 'tied_embeddings_verified_by_config': True, 'tokenizer_sha256': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'} | pending | pending | pending | pending | pending | pending | [measured_cpu_only](inspection/summary.json) |
| pending | pending | pending | pending | pending | pending | pending | pending | [unspecified](repair/2a/summary.json) |
| pending | pending | pending | pending | pending | pending | pending | pending | [unspecified](repair/2b/summary.json) |
| pending | pending | pending | pending | pending | pending | pending | pending | [cpu_builds_only](repair/summary.json) |

Only recorded values are shown. Missing fields remain pending; partial runs retain their recorded status.

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
| damage_cpu_validation / pending | pending | pending | pending | pending |
| eval_grid_cpu_validation / token_state_machine | pending | pending | pending | pending |
| pending / pending | pending | pending | pending | pending |
| pending / pending | pending | pending | pending | pending |
| pending / pending | pending | pending | pending | pending |
| pending / pending | pending | pending | pending | pending |

| Stage | Benchmark | Recorded hours | Recorded cost (USD) |
| --- | --- | --- | --- |
| calibration_base | math500 | [0.465](calibration_base/summary.json) | [2.1280](calibration_base/summary.json) |
| calibration_p50c | math500 | [0.259](calibration_p50c/summary.json) | [1.2512](calibration_p50c/summary.json) |
| control1_nothink | math500 | [0.236](control1_nothink/math500/summary.json) | [1.0059](control1_nothink/math500/summary.json) |
| control2_mechanical | math500 | [0.447](control2_mechanical/math500/summary.json) | [1.9082](control2_mechanical/math500/summary.json) |
| control3_repetition | math500 | [0.514](control3_repetition/math500/summary.json) | [2.1923](control3_repetition/math500/summary.json) |
| damage_cpu_validation | pending | pending | pending |
| eval_grid_cpu_validation | token_state_machine | pending | pending |
| pending | pending | pending | pending |
| pending | pending | pending | pending |
| pending | pending | pending | pending |
| pending | pending | pending | pending |

Costs and durations come from the linked records. Missing costs are not treated as zero; rows are not summed because stages may overlap.

Budget ledger: [source](costs.json). Reservations are retained bounds, not charges.

| Budget / stage / operation | State | Receipts | Recorded hours | Settled amount (USD) | Reserved bound (USD) | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| editor / editor turns | [settled](costs.json) | [26](costs.json) | [1.815](costs.json) | [78.2928](costs.json) | pending | [API usage and recorded token rates](costs.json) |
| modal / all | [reserved](costs.json) | [1](costs.json) | pending | pending | [1.9060](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / all | [settled](costs.json) | [5](costs.json) | [4.262](costs.json) | [0.3380](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
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
| modal / live / codex-20261103 / fwd_3 / edited | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.9870](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_3 / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_4 / edited | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.9870](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / live / codex-20261103 / fwd_4 / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
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
| modal / sandbox:heartbeat | [settled](costs.json) | [305](costs.json) | [4.976](costs.json) | [12.8769](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / sandbox:startup | [settled](costs.json) | [6](costs.json) | [0.022](costs.json) | [0.0581](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / verify / attention | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1626](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / fresh-reference | [settled](costs.json) | [1](costs.json) | [0.086](costs.json) | [0.3677](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / native-swap | [settled](costs.json) | [1](costs.json) | [0.097](costs.json) | [0.4123](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / teacher-forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1635](costs.json) | pending | [resource estimate; not invoice](costs.json) |

Repeated requests, attempts and batches are grouped by stage and operation; the linked ledger retains each receipt. Ledger rows and evaluation summaries may describe the same jobs; do not add the two tables. Summed resource hours are not end-to-end wall time. Settled resource estimates remain estimates until reconciled with provider billing.
