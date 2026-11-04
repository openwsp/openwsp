# Status: Modal workspace disabled at 38 passes; CPU extraction continues

Read [docs/TASK.md](docs/TASK.md) and [steering update 2](docs/STEERING_UPDATE_2.md). Main is the authorized target: test logical changes, commit and push directly. README voice follows user commit 0a41b05: no singular first person or em dashes. User-facing timestamps use America/Buenos_Aires.

## Current blocker and preserved state

The same **codex-20261103** subscription attempt is paused at **turn 101, 38/50 passes, 46 requests, 52 runs and 2 handoffs**. Its next action is **editor turn 102**. The provider returned `ConflictError: workspace is disabled`; both the worker and a subsequent status invocation were rejected. No billing-specific cause is established. The saved pipeline and coordinator snapshots still say running from before the interruption; these raw records are retained and the reporter labels them as last persisted states.

Driver **fc-01M6HJFWKABFATSH2NKQKVB6FN** is provider-confirmed terminal: one failed input, zero pending, finished at **22:35:04 on November 3**. The older output endpoint reported a contradictory unfinished count; `FunctionCallGetInfo` and the call graph agree on failure. All 37 GPU children from this driver succeeded. Local worker **PID 50018** has exited. Its frozen source was **7542fe5e15b1d0f8d7cbbb17e21e836a3831c112**, checkout `/tmp/openwsp-notice-resume-7542fe5-17_b8pae`; its `.openwsp/launch.json` retains the call ID. Former sandbox: **sb-01M6HJGHTRF1G3N5F4PQRQCWZ9**. No replacement has been launched.

Turn 102 is queued in the bridge with `dispatch`, `claimed_by` and `completion` all null. All **101 prior local replies** have verified completion receipts and final text, with conversation counts **40/40/21**. There is no unfinished local model call to recover. Preserve the private worker identity and conversation 2. The existing bridge will admit turn 102's first request; do not rewrite its immutable prompt or manually replay any response. Normal kernel-restart handling remains in place.

Public evidence: [blocker](replications/exp8/results/live-runs/codex-20261103/blocker.json), [provider metadata](replications/exp8/results/live-runs/codex-20261103/workspace-disabled-provider-20261104.json), [editor audit](replications/exp8/results/live-runs/codex-20261103/workspace-disabled-editor-audit-20261104.json), [state](replications/exp8/results/live-runs/codex-20261103/state.json). Private snapshots and logs are `logs/modal-workspace-disabled-*`; full run log is `logs/modal-codex-notice-resume.log`. The old passive observer session 93922 ended with the worker.

## Resume and pending operator replies

The operator has been asked to check Modal's workspace/account status and report when calls are accepted again. Do not blindly retry failed invocations while the workspace is disabled. Read-only Volume access still works. Once access is restored, the normal serialized launch path can reconcile the terminal driver and resume the same attempt:

```bash
.venv/bin/openwsp replicate exp8 --stage all --config /workspace/openwsp/.openwsp/codex-workspace-resume.json --foreground
```

The local configuration is prepared with an incident-specific `resume_reason`, the existing private home/state/executable and unchanged run ID. It has **not been launched**. Read the coordinator again before launch, since the existing scheduled watchdog may reconcile after account access returns. Use the serialized CLI to attach or resume, never a direct duplicate spawn. Deploy from a tested source snapshot; record the resulting driver call ID before monitoring. Keep the foreground worker running through final evaluation and publication. No separate approval is needed to resume work already authorized within the current cap.

A separate async question at 21:54 asked whether the **total Modal cap may rise from $150 to $225**. This is pending, not authorization. The attached steering retains the total cap separately from account top-ups. Keep **$150** until an explicit answer. No budget code, configuration or ledger limit has been changed. The earlier native Fable credential question is also pending; no Anthropic or OpenRouter key was present at the latest check.

## Accounting and editor transport

Saved ledger after the interruption: **$104.89222352871018 Modal spent, $18.129953472222223 reserved, $26.927822999067608 available**. These are conservative resource estimates, not provider invoices. Automatic account top-ups are authorized within the unchanged $150 task cap; no dashboard increase or payment status is inferred. Preserve historical failed-job bounds and unresolved reservations. The normal serialized coordinator will reconcile the terminal driver's ownership on a later authorized launch.

Subscription Astra remains primary: Codex 0.163.0, `gpt-6-astra`, effort max, native tools disabled. Private authentication is in `.openwsp/codex-subscription`; immutable worker identity and response journals are in `.openwsp/codex-workers/codex-20261103`. Never send these credentials to Modal. Historical API spend is **$78.292765**, remote cap $100, reserved zero. The additional $160.85 allocation remains unapplied; the separate **$160.85 Plan B reserve is protected**. No API fallback or new paid probe. The read-only quota RPC at 19:52 showed 69% subscription quota remaining and one unused reset credit (`logs/codex-subscription-quota-latest.json`). Fable's separate $20 cap is unused.

## Local CPU work still running

Full variant 2c input extraction is **PID 39594**, two threads, started around 16:40 Buenos Aires time. Latest progress: **232/484 examples** at about 23:00. All 450 looping and 34 finished examples are required. The projection remains roughly 13.5 hours total; do not restart, duplicate, shorten the selection or build from partial statistics. Logs: `.openwsp/repair-inputs/full-extraction.{pid,log,exit}`. Receipts: `.openwsp/repair-input-means`.

The single continuation **PID 44670** waits for that exact extraction and validates all receipt, source, checkpoint, tokenizer, template, direction and dose bindings before building 2c and the seven coefficient bundles. Logs: `.openwsp/repair-inputs/completion.{pid,log,exit}`. It performs no GPU scoring, new inference, remote calls or push. Both original processes remain healthy. A prior independent 70/484 receipt audit passed. Full weights, activation arrays and raw generations remain private.

## Completed evidence and preserved fixes

Calibration passed: base 36/500 (7.2%), p50c 253/500 (50.6%). Controls 1–3: thinking off 234/500 (46.8%), mechanical 230/500 (46.0%), repetition 1.1 119/500 (23.8%). All 2,500 compact records and finished-only metrics are public. Kernel hash/refusal/isolation and forward internals evidence passed; native vLLM weight reload is unavailable, so fresh engines are used. Calibration and kernel-verified tags exist; live-run and v0.1.0 do not.

Astra raw CPU comparisons are complete: 11 replies and 11 scores, one per released prompt, no tools or feedback. Games 4/4; shapes 3/3; car 0/1024. MinAtar, 1,000 episodes each: Breakout 458.730±4.994560; Asterix 39.709±0.730104; Seaquest 2.407±0.105937. Exact scored policies were float32; public v1 edits are bfloat16 interchange exports with hashes and provenance. [Comparison](replications/raw_weights/results/comparison.md). Fable is the pending comparison; experiments 3, 6 and 7 remain layout only.

Inspect CLI/SDK/MCP, geometry, finished-only metrics, 11 grid definitions, repair 2a/2b, KL reduction/backend/stage, generation banks and the 17,734-byte execution ceiling are implemented and tested. Preserve true SHA256 problem seeds, first 8,192 answer-token float16/NumPy float32 reduction, and bounded hash-verified transfer. No global sandbox Volume reload or new ledgers, budget abstractions, watchdog modes or reporting automations.

The ordinary-generation startup failure was fixed and verified on real requests 20 and 21. The turn 51 notice-transfer repair was verified against its exact 482-byte file hash. Neither failure's history or costs were erased. Later namespace-aware forward receipt and reporting fixes are on main but were not in frozen driver 7542fe5; they deploy with the next tested launch. Historical wrong receipt locators have explicit correction evidence in `forward-bank-live.json`.

## Publication, checks and remaining order

The previous published halfway checkpoint is **1d617ce**, with 25 passes and transcript through turn 74; it passed 130 focused tests and both Python 3.11/3.12 CI jobs: https://github.com/openwsp/openwsp/actions/runs/41738624497. The current incident publication records 38 passes and transcript through turn 101, with no submission or final benchmark claim. Its transcript is 681,971 bytes, SHA256 `352c73a0f9792b7299af948ad6848e4db9da804a475dcaa9ea31b4a2eb170ecf`. Nine completed requests reused base generations: 23, 24, 26, 28, 31, 32, 40, 41 and 42. Reporter clarity and raw-sample omission changes passed 148 reporting/publication/artifact tests and Ruff. Prior transcript turns 1–74 remain an exact byte prefix. Only the 8,421-byte raw sample dump in turn 83 and its turn 84 echo are omitted, with identical hashes and source evidence in `transcript-omissions.json`; original journals, protocol examples, compact metadata, code and replies are unchanged. Independent review passed, including exact omission/source hashes and preserved prior records. The incident publication is ready to commit and push.

1. Resume this same faithful attempt when Modal accepts calls, complete 50 passes and its actual submission, then MATH500×4, GSM8K×2 and IFEval×2. The stage-all pipeline handles final evaluation and publication.
2. Control 4, then the 11 preregistered grid cells, then neutral-reference KL, within the authorized cap. Use `.openwsp/codex-post-faithful.json` and normal foreground CLI. Individual stages do not auto-publish; use the existing publish stage afterward. Forced-close cells still require the real four-problem GPU smoke.
3. Finish the full CPU extraction and verify/publish 2c and dose metadata. No optional repair-scoring GPU work before the required queue. Run 2 requires fresh operator confirmation after run 1. Tag only when the original criteria are met. The overall goal is not complete while this work remains.
