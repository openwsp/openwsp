# Recorded evaluation results

| Recorded component | State | Evidence |
| --- | --- | --- |
| Pipeline | [running](pipeline-state.json) | [live](pipeline-state.json) |
| Driver recovery | [running](coordinator.json) | [all](coordinator.json) |
| Calibration prompt audit | [passed](calibration-prompt-audit.json) | [calibration-prompt-audit.json](calibration-prompt-audit.json) |
| Calibration comparison | [passed](calibration-check.json) | [calibration-check.json](calibration-check.json) |
| Tensor hashes | [passed](kernel/hash-verification.json) | [kernel/hash-verification.json](kernel/hash-verification.json) |
| Sandbox and first cell | [passed](kernel/sandbox-verification.json) | [kernel/sandbox-verification.json](kernel/sandbox-verification.json) |
| Forward service | [passed](kernel/forward-verification.json) | [kernel/forward-verification.json](kernel/forward-verification.json) |
| Weight swap check | [fallback](kernel/weight-swap.json) | [kernel/weight-swap.json](kernel/weight-swap.json) |
| Recorded diagnostic | [Engine does not expose a compatible weight-reload contract](kernel/weight-swap.json) | See source record. |
| Live editor | [incomplete](live-run/state.json) | passes: [2](live-run/state.json); turn: [13](live-run/state.json); runs: [6](live-run/state.json) |

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

| Stage | Benchmark | Recorded hours | Recorded cost (USD) |
| --- | --- | --- | --- |
| calibration_base | math500 | [0.465](calibration_base/summary.json) | [2.1280](calibration_base/summary.json) |
| calibration_p50c | math500 | [0.259](calibration_p50c/summary.json) | [1.2512](calibration_p50c/summary.json) |

Costs and durations come from the linked records. Missing costs are not treated as zero; rows are not summed because stages may overlap.

Budget ledger: [source](costs.json). Reservations are retained bounds, not charges.

| Budget / stage / operation | State | Receipts | Recorded hours | Settled amount (USD) | Reserved bound (USD) | Basis |
| --- | --- | --- | --- | --- | --- | --- |
| editor / editor turns | [reserved](costs.json) | [1](costs.json) | pending | pending | [7.1685](costs.json) | [API usage and recorded token rates](costs.json) |
| editor / editor turns | [settled](costs.json) | [13](costs.json) | [0.161](costs.json) | [3.3555](costs.json) | pending | [API usage and recorded token rates](costs.json) |
| modal / all | [reserved](costs.json) | [1](costs.json) | pending | pending | [1.9060](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / all | [settled](costs.json) | [3](costs.json) | [1.505](costs.json) | [0.1194](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
| modal / calibration | [settled](costs.json) | [1](costs.json) | [0.906](costs.json) | [0.0718](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
| modal / calibration_base | [settled](costs.json) | [1](costs.json) | [0.499](costs.json) | [2.1280](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / calibration_base_smoke | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.2758](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / calibration_base_smoke | [settled](costs.json) | [1](costs.json) | [0.105](costs.json) | [0.4496](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / calibration_p50c | [settled](costs.json) | [1](costs.json) | [0.293](costs.json) | [1.2512](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / calibration_p50c_smoke | [settled](costs.json) | [1](costs.json) | [0.099](costs.json) | [0.4228](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / infrastructure | [reserved](costs.json) | [1](costs.json) | pending | pending | [2.0000](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / kernel | [settled](costs.json) | [1](costs.json) | [0.205](costs.json) | [0.0163](costs.json) | pending | [driver_resource_upper_bound_not_invoice](costs.json) |
| modal / kernel_verification:heartbeat | [settled](costs.json) | [2](costs.json) | [0.003](costs.json) | [0.0073](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / kernel_verification:startup | [settled](costs.json) | [1](costs.json) | [0.004](costs.json) | [0.0092](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / live / activation extraction / original | [settled](costs.json) | [1](costs.json) | [0.700](costs.json) | [2.9870](costs.json) | pending | [failed job upper bound; not invoice](costs.json) |
| modal / live / activation extraction / original | [settled](costs.json) | [1](costs.json) | [0.071](costs.json) | [0.3049](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / generation / original | [settled](costs.json) | [2](costs.json) | [0.539](costs.json) | [2.3013](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / live / teacher-forced | [settled](costs.json) | [1](costs.json) | [0.041](costs.json) | [0.1745](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / prepare | [settled](costs.json) | [1](costs.json) | [0.010](costs.json) | [0.0016](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / prepare-v2 | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.0060](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / sandbox:heartbeat | [reserved](costs.json) | [1](costs.json) | pending | pending | [0.2157](costs.json) | [resource estimate; not invoice](costs.json) |
| modal / sandbox:heartbeat | [settled](costs.json) | [85](costs.json) | [1.358](costs.json) | [3.5139](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / sandbox:startup | [settled](costs.json) | [4](costs.json) | [0.015](costs.json) | [0.0400](costs.json) | pending | [resource_upper_bound_not_invoice](costs.json) |
| modal / verify / attention | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1626](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / fresh-reference | [settled](costs.json) | [1](costs.json) | [0.086](costs.json) | [0.3677](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / native-swap | [settled](costs.json) | [1](costs.json) | [0.097](costs.json) | [0.4123](costs.json) | pending | [resource estimate; not invoice](costs.json) |
| modal / verify / teacher-forced | [settled](costs.json) | [1](costs.json) | [0.038](costs.json) | [0.1635](costs.json) | pending | [resource estimate; not invoice](costs.json) |

Repeated requests, attempts and batches are grouped by stage and operation; the linked ledger retains each receipt. Ledger rows and evaluation summaries may describe the same jobs; do not add the two tables. Summed resource hours are not end-to-end wall time. Settled resource estimates remain estimates until reconciled with provider billing.
