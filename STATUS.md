# Status: CPU repair builds verified; Modal workspace disabled at 38 passes

Read [docs/TASK.md](docs/TASK.md) and [steering update 2](docs/STEERING_UPDATE_2.md). Main is the authorized target: test logical changes, commit and push directly. README voice follows user commit 0a41b05: no singular first person or em dashes. User-facing timestamps use America/Buenos_Aires.

**The overall goal is unmet.** The faithful run has no submission or final evaluation. Modal execution is externally blocked; the Fable comparison needs native credentials. The independent local CPU extraction, edit construction and audits are complete. No local experiment or observer remains running.

## Modal blocker and preserved attempt

The same **codex-20261103** subscription attempt is paused at **turn 101, 38/50 passes, 46 requests, 52 runs and 2 handoffs**. Its next action is **editor turn 102**. The provider returned `ConflictError: workspace is disabled` to the worker's bridge call and a subsequent status invocation. The error does not establish a billing, payment or quota cause. Raw pipeline/coordinator snapshots retain their old running phases; the reporter labels them as last persisted states.

Driver **fc-01M6HJFWKABFATSH2NKQKVB6FN** is provider-confirmed terminal: one failed input, zero pending, finished at **22:35:04 on November 3**. `FunctionCallGetInfo` and the complete call graph agree; the older output endpoint's unfinished count was contradictory. All 37 GPU children succeeded. Local worker PID 50018 has exited. Its frozen source was **7542fe5e15b1d0f8d7cbbb17e21e836a3831c112**, checkout `/tmp/openwsp-notice-resume-7542fe5-17_b8pae`. Former sandbox **sb-01M6HJGHTRF1G3N5F4PQRQCWZ9** is stopped with exit 137; that code does not establish OOM or another cause.

Turn 102 is queued with `dispatch`, `claimed_by` and `completion` all null. All **101 prior local replies** have verified completion receipts and final text, with conversation counts **40/40/21**. No unfinished local model request needs recovery. Preserve the existing worker identity and conversation 2; do not rewrite the queued prompt or replay a response. Normal kernel-restart handling remains in place.

Public evidence: [blocker](replications/exp8/results/live-runs/codex-20261103/blocker.json), [provider metadata](replications/exp8/results/live-runs/codex-20261103/workspace-disabled-provider-20261104.json), [editor audit](replications/exp8/results/live-runs/codex-20261103/workspace-disabled-editor-audit-20261104.json), [state](replications/exp8/results/live-runs/codex-20261103/state.json). Private logs: `logs/modal-workspace-disabled-*` and `logs/modal-codex-notice-resume.log`. Direct Volume reads at 05:30 on November 4 still found the same terminal driver and unchanged heartbeat, with no watchdog replacement (`logs/modal-disabled-direct-observation-latest.json`). No new function invocation or model request followed the recorded disabled-status failure.

## Exact resume path

The operator has been asked to report when Modal accepts calls again. Do not blindly retry rejected invocations. Read-only Volume access still works. Once access returns, read the coordinator first because the existing scheduled watchdog may reconcile after restoration, then use the normal serialized launch path:

```bash
.venv/bin/openwsp replicate exp8 --stage all --config /workspace/openwsp/.openwsp/codex-workspace-resume.json --foreground
```

This prepared configuration has the incident-specific `resume_reason`, unchanged run ID and existing private home/state/executable. It has not been launched. Deploy from a tested source snapshot, record the resulting driver call ID and keep the foreground worker alive through final evaluation/publication. Use the CLI to attach or resume, never directly spawn a duplicate driver. No additional permission is needed for already-authorized work within the current cap.

Subscription Astra remains primary: Codex 0.163.0, `gpt-6-astra`, effort max, native tools disabled. Authentication is private in `.openwsp/codex-subscription`; worker journals are in `.openwsp/codex-workers/codex-20261103`. Never send these credentials to Modal. The latest recorded quota RPC showed 69% remaining and one unused reset credit (`logs/codex-subscription-quota-latest.json`); no reset was redeemed.

## Accounting and pending replies

The saved ledger retains **$104.89222352871018 Modal spent, $18.129953472222223 reserved and $26.927822999067608 available** under the **$150 total task cap**. These are conservative resource estimates. Automatic account top-ups are authorized within that cap. Preserve failed-job bounds and unresolved reservations; no ledger, cap or account setting changed during the CPU work.

The [published provider snapshot](replications/exp8/results/provider-billing-20261104.json), observed at 23:02 on November 3, reports $51.37595920 metered and $20.41000000 after credits/allowances. Its 24 app/hour rows total $50.36126872 for completed intervals, all attributed to openwsp-exp8. A later read-only refresh at 02:19 on November 4 reports $51.28595920 metered and $20.32000000 after adjustments (`logs/modal-provider-billing-summary-latest.json`). These cycle totals do not establish payment, the current provider limit, the disabled-workspace cause or individual job costs. Do not settle reservations from aggregate billing.

The separate 21:54 question about increasing the task cap to $225 remains unanswered and is not authorization. It was based on conservative estimates; the provider figures do not establish that a higher allocation is needed. Historical API spend is **$78.292765**, remote cap $100, reserved zero. The authorized additional $160.85 remains unapplied; the separate **$160.85 Plan B reserve is protected**. No automatic API fallback. Fable's separate $20 cap is unused. Native Fable credentials remain absent at the latest environment check; its credential question is pending.

## Verified CPU repair completion

Original extraction **PID 39594** and bound continuation **PID 44670** both exited **0** around 05:32 on November 4. All **484 examples**, comprising **450 looping and 34 finished**, were retained with the original first-8192-answer-token reduction. The original four benchmark receipts were reused unchanged. The public [extraction summary](replications/exp8/results/repair/input-activations-complete.json) reports summed per-example durations, including those reused receipts, not full-process wall time. No extraction was restarted or shortened.

The continuation built variant **2c** and coefficients **0, 1, 2, 3, 4, 6, 8**, using the same saved output directions and independently selected signed top-64 columns. The [independent audit](replications/exp8/results/repair/2c/verification.json) rederived every selected identity, token-ID hash, activation receipt, group mean, column mask and BF16 formula. The standalone chat template matches its hash-bound tokenizer configuration. Alpha 0 changes no tensors. All private/public bundle identities agree and scores remain null/pending.

2c identity: **8114394c5620c8d7f1f3616e927feba74130925ac57fb0574b3a5cd8b3d0e350**; **130,641 changed entries across two tensors**. Its masks overlap the earlier borrowed p50c mask in **25/64 columns at layer 15 and 18/64 at layer 19** ([comparison](replications/exp8/results/repair/2c/column-comparison.json)). [Seven bundle identities](replications/exp8/results/repair/dose-sweep.json) are published; no repair benchmark score is claimed.

Private extraction receipts remain in `.openwsp/repair-input-means`; private bundles in `.openwsp/repair-build/{2c,dose}`. Exit receipts and construction result are `.openwsp/repair-inputs/{full-extraction,completion}.exit` and `completion-result.json`. The one-shot independent audit is `/tmp/openwsp-repair-final-audit.py`, SHA256 `7d09e851ac811d60cd1f0dc36b2b9ace595188fa8b218ad53c9a3457f96d983e`; it passed before the compact publication copies were added. Its strict original file allowlist means it should not be rerun unchanged against the expanded public metadata set. Private result: `logs/repair-final-independent-audit.json`. Full weights, activation arrays and raw generations remain private.

Publication review verified both receipt copies, all mask intersections, links, unchanged preregistration and pending-score language. **177 focused repair, extraction, completion, reporting and publishing tests passed** (`logs/tests-repair-completion.log`). README/report were regenerated from source files; report receipt: `logs/report-repair-completion.json`. CI evidence for this publication is recorded after push in `logs/ci-repair-completion.json`.

## Other completed evidence

Calibration passed: base 36/500 (7.2%), p50c 253/500 (50.6%). Controls 1–3: thinking off 234/500 (46.8%), mechanical 230/500 (46.0%), repetition 1.1 119/500 (23.8%). All 2,500 compact records and finished-only metrics are public. Kernel hash/refusal/isolation and forward-internals checks passed. Native vLLM reload is unavailable; fresh engines are the documented fallback. Calibration and kernel-verified tags exist; live-run and v0.1.0 do not.

Astra raw CPU comparisons are complete: 11 replies and scores, one per released prompt, no tools or feedback. Games 4/4; shapes 3/3; car 0/1024. MinAtar, 1,000 episodes each: Breakout 458.730±4.994560; Asterix 39.709±0.730104; Seaquest 2.407±0.105937. Exact scored policies were float32; public v1 exports are BF16 with rounding provenance. [Comparison](replications/raw_weights/results/comparison.md). Fable remains pending; experiments 3, 6 and 7 are layout only.

CLI/SDK/MCP share the provider-independent core. Inspect/geometry, all 11 grid definitions, repair, KL reduction/backend/stage, generation banks and the 17,734-byte execution ceiling are implemented and CPU-tested. Preserve true SHA256 problem seeds, the prescribed activation reduction and bounded hash-verified transfer. No global sandbox Volume reload, new ledgers, watchdog modes or reporting automations.

Generation startup was fixed and verified on actual requests 20/21. Turn 51's notice-transfer repair was verified against its exact 482-byte file hash. History and costs remain intact. Namespace-aware forward receipt and reporting fixes are on main but were absent from frozen driver 7542fe5; they deploy on the next tested launch. `forward-bank-live.json` preserves corrections to historical wrong receipt locators.

The 38-pass transcript is 681,971 bytes, SHA256 `352c73a0f9792b7299af948ad6848e4db9da804a475dcaa9ea31b4a2eb170ecf`. Prior turns 1–74 remain an exact prefix. Only the 8,421-byte raw sample dump in turn 83 and its turn 84 echo are omitted with hashes in `transcript-omissions.json`; private history is unchanged. Requests 23, 24, 26, 28, 31, 32, 40, 41 and 42 reused base generations. Incident commit 439c64a passed both CI jobs (`logs/ci-workspace-disabled.json`); provider-billing commit eb4c2ae also passed (`logs/ci-provider-billing.json`).

## Remaining execution order

1. Restore Modal execution, resume this same faithful attempt, complete 50 passes and the actual submission, then MATH500×4, GSM8K×2 and IFEval×2. Stage all handles evaluation and publication.
2. Control 4, the 11 preregistered grid cells, then neutral-reference KL, within the authorized cap. Use `.openwsp/codex-post-faithful.json` and the normal foreground CLI. Individual stages require the existing publish stage afterward. Forced-close cells still require the real four-problem GPU smoke. No optional repair scoring before this queue.
3. When native Fable credentials are available, follow `replications/raw_weights/results/fable-blocker.json`: native model/token preflight, the existing one-reply/no-tools runner, then CPU scoring under its $20 cap.
4. Run 2 requires fresh operator confirmation after run 1. Tag only when the original criteria are met. Never present the current incomplete attempt or CPU builds as a completed faithful replication.
