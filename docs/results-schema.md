# Result records and contributions

`openwsp report exp8` reads evaluation files and regenerates the README's marked status, result and cost tables. It does not infer success from the presence of a run directory. Missing measurements remain pending, and an interrupted or blocked run must retain its status.

## Layout

Place one run under `replications/exp8/results/<run-id>/`. Keep a separate `summary.json` for each stage, model condition and benchmark, for example `<run-id>/evaluation/<condition>/<benchmark>/summary.json`. The reporter discovers `summary.json` files recursively under the results directory.

Named live attempts use `replications/exp8/results/live-runs/<run-id>/`, with `identity.json`, `state.json`, `transcript.md` and final summaries at `<benchmark>/summary.json`. The original API attempt remains under `live-run/`. A new attempt does not overwrite its state, costs or transcript.

Commit compact per-sample records, summaries, configuration/provenance, the edit manifest and diff, and the readable transcript. Do not commit weights, checkpoints, activations or full generated text. Full text remains on the volume, referenced by path and SHA-256. Preserve the original artifacts when a run is repeated; do not overwrite a lower result.

## Evaluation summary

A summary is one JSON object. Rates use fractions in the file; the report formats them as percentages. Omitted numeric fields mean unavailable, not zero.

| Field | Type | Meaning |
| --- | --- | --- |
| `stage` | string | The stage or condition, such as calibration, live evaluation or a named control. |
| `model` | string | Model/condition identifier, with exact checkpoint provenance recorded separately. |
| `benchmark` | string | Benchmark key, such as `math500`, `gsm8k` or `ifeval`. |
| `samples_per_problem` | integer | Number of samples requested per problem. |
| `n` | integer | Number of scored samples actually present. |
| `correct` | integer | Number of those samples graded correct. |
| `accuracy` | number | `correct / n`, on the interval from zero to one. |
| `finished_n` | integer | Samples with canonical `finish_reason: "stop"`; excludes `"length"` token-limit termination. |
| `finished_correct` | integer | Correct grades among those finished samples. |
| `accuracy_finished` | number or null | `finished_correct / finished_n`; null when no samples finish. Missing termination evidence leaves this metric unavailable. |
| `standard_error` | number | Standard error as a fraction, using the method declared in provenance. |
| `mean_tokens` | number | Mean generated token count over scored samples. |
| `cut_share` | number | Fraction of scored samples that reached the generation token limit. |
| `seconds` | number | Measured duration for this record, with its scope stated in provenance. |
| `cost_usd` | number | Cost allocated to this stage in US dollars; distinguish measured charges from estimates. |
| `status` | string | Explicit run state. Never label partial evaluation complete. |
| `provenance` | object | Sources, configuration, runtime versions, hardware, sample files, cost method and deviations. |
| `reference` | string, optional | Explicit author comparison: `unedited`, `p10c`, `p25c` or `p50c`. |

The reference comparison must describe the intended comparator. It does not assert that a different control is a reproduction of that condition. The authors' table is generated separately from `replications/exp8/reference/*.json`.

Compute the finished fields with `openwsp.metrics.finished_metrics(records)` from the complete compact sample records. The helper rejects unknown termination reasons and invalid grades rather than treating them as completed answers. Recompute the fields after combining batches; never average batch accuracies or inherit a batch's denominator. Overall accuracy retains capped answers, including any that still grade correct. Finished accuracy therefore cannot be inferred from overall accuracy and `cut_share`. Compare it with the cap-hit rate: different conditions can finish different subsets, so a change in conditional accuracy alone is not a paired regression.

The published summaries record the helper, termination convention and compact sample SHA-256 under `provenance.finished_metrics`. The artifact collector recomputes these fields from collected samples, preserving the original private summary. The reporter checks any recorded finished fields against the sample file before changing published tables. `analysis/finished-metrics.json` records the initial recomputation over previously published samples.

Record problem coverage as well as sample count in provenance. Multiple samples of the same problem are not independent problems: report the uncertainty method and preserve the per-problem grouping. An exact ratio with no uncertainty method is insufficient for a calibration claim.

## Per-sample record

Use JSON Lines, one object for every scored sample:

| Field | Type | Meaning |
| --- | --- | --- |
| `problem_id` | string | Stable benchmark problem identifier. |
| `sample_index` | integer | Sample index within the problem. |
| `answer` | string or null | Extracted final answer, not the full generated reasoning. |
| `correct` | boolean | Grade under the recorded official protocol. |
| `token_count` | integer | Generated tokens for this sample. |
| `finish_reason` | string | Canonical `stop` or `length`. Preserve and document the mapping from the engine; never silently classify an unknown or failed generation as finished. |
| `raw_path` | string | Volume path of the full generation artifact. |
| `raw_sha256` | string | SHA-256 of the referenced artifact's exact bytes. |

If a shared raw artifact contains multiple samples, record how the problem ID and sample index locate the row. Do not silently drop unsuccessful generations or retries. Record failures separately so reported coverage and spend can be reconciled.

## Cost and transcript records

Stage summaries may contain `seconds` and `cost_usd`. A separate `cost.json` can record billing for a stage that has no evaluation summary, with `stage`, `seconds`, `cost_usd`, and provenance identifying the job and measurement method. Avoid counting the same charge in both files. Store GPU seconds, CPU/memory use or rate assumptions, and the pre-launch estimate with the underlying job record.

`costs.json` is the durable budget ledger. The report reads its `reservations` object, groups receipts by budget, stage, state and billing basis, and combines editor-turn receipts for display. Settled amounts and reserved upper bounds appear in separate columns. Numeric strings from the ledger are accepted only when finite and nonnegative. A missing amount stays pending; an unresolved reservation is not a charge or a zero-cost job. The ledger table and evaluation summaries may refer to the same work and must not be added together. Resource estimates do not become invoices merely because their reservations are settled.

## Published state and calibration evidence

The report reads `pipeline-state.json`, optional `coordinator.json`, `live-run/state.json`, the kernel verification files, `calibration-prompt-audit.json`, `calibration-check.json` and optional `calibration-gap-audit.json`. A top-of-README table uses optional `OPENWSP:STATUS` markers. Publication may change only that table body and the existing `RESULTS` and `COSTS` table bodies; surrounding prose stays unchanged.

`active-live.json` selects the current named attempt. Its version, editor, run ID, relative path, task hash and instructions hash must match the attempt's identity record. The relative path must be exactly `live-runs/<run-id>`, with a run ID matching `[a-z][a-z0-9_-]{0,63}`. Historical attempts remain in the report. Without an active pointer, the original `live-run/` remains the default.

Missing active state or identity keeps the current goal pending. A recorded live completion requires exactly fifty passes, `done: true`, a concrete submission checkpoint and hash, and full MATH-500, GSM8K and IFEval summaries for that same attempt and submitted edit. Each final summary records the matching `provenance.submission_sha256`; subset or batch results do not establish completion. Summary paths must resolve within their own attempt directory.

Pipeline and recovery states are shown as recorded. The live editor's `done` flag means a submission was recorded, not that evaluation, controls, CI and publication all passed. The report never promotes a passed hash check or completed evaluation into an overall completion claim. The saved heartbeat is a snapshot, so current liveness must be checked through `openwsp status`.

Calibration comparison records retain observed/reference accuracy, gap and the prescribed tolerance outcome. A recorded gap stays visible when a circuit-breaker audit permits continuation. Publish the audit file with its checks and unresolved discrepancy; an absent audit is shown explicitly. Runtime metadata and standard-error methods are read from each summary's `provenance` object.

## Independent comparisons and interpretation

Optional `analysis/independent-results.json` supplies the generated independent-results table. Each external row records its primary source, immutable revision where available, retrieved source hash, benchmark coverage, sampling, thinking mode, token cap, engine and comparison limits. Local rows use a `summary` path relative to the results directory, so refreshed tables read the actual measurement. Missing referenced summaries remain pending even if the comparison file contains copied values. Do not pool different benchmarks, forced-close policies, thinking settings or sampling configurations.

Optional `analysis/interpretation.json` supplies the four generated FAQ answers within the existing `RESULTS` section. Paragraph fragments reference summary metrics, source permalinks or contained result artifacts; exact task quotations retain their source links. `within` selects nested variant metadata in a summary. Unmeasured variants remain pending. Source-verification files preserve quotations, calculations and caveats without including private generations. These analysis files and every linked local artifact are included in the report's source-hash receipt. They are orchestrator evidence and must never be sent to the live editor.

The readable transcript is generated from `turns/` and `tool_calls.jsonl`. Each turn retains the editor's explanation, code or forward block and returned result. Redact credentials; link to large outputs by their artifact identity instead of embedding arrays or full datasets. The current Experiment 8 transcript is selected by [`active-live.json`](../replications/exp8/results/active-live.json), presently [`live-runs/codex-20261103/transcript.md`](../replications/exp8/results/live-runs/codex-20261103/transcript.md). The [`live-run/transcript.md`](../replications/exp8/results/live-run/transcript.md) file preserves the earlier API attempt.

## Add a run by pull request

1. Choose a new run ID and commit its exact model revision, editor settings, prompt/source hashes, dataset revisions and code revision.
2. Include all sample records and summaries for that run, the final edit identity, runtime provenance, costs and every departure from the prescribed protocol.
3. Run `uv run openwsp report exp8`. Review the generated table's links and verify that counts and ratios match the sample files.
4. Run `uv run pytest -q` and `python scripts/check_secrets.py --all`, then submit the evidence and regenerated tables together.

Do not write numbers directly into the README. Record a failed or incomplete attempt honestly, with its missing stages and resume command. A result from another model or changed protocol belongs under a distinct configuration and must be labeled accordingly.

The [contributor guide](contributing-experiments.md) explains how to add an evaluation cell, mechanical repair variant or replication folder.
