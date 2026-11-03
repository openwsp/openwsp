# Status: faithful attempt recovery at 14 of 50 passes

Read [docs/TASK.md](docs/TASK.md) and [steering update 2](docs/STEERING_UPDATE_2.md). Main is the authorized target: test logical changes, commit and push directly. README voice follows user commit0a41b05: no singular first person or em dashes. User-facing timestamps use America/Buenos_Aires.

## Current action

The same **codex-20261103** subscription attempt is saved at **turn50,14/50passes,22requests,26runs,one handoff,pending=null,done=false**. Driver **fc-01M6HCQ89PPXXFSZB647FGB9VF** is provider-confirmed terminal with `SandboxTransportError: Text transfer failed with exit code1` while delivering the prescribed turn51 `eval_prompt_format.txt`, before editor admission. Local worker PID32617 exited. Sandbox sb-01M6HCQY8QNQZN5JRPJ6W9D83T is terminated. Historical stderr was not retained; do not infer OOM from its later exit137. [Incident](replications/exp8/results/live-runs/codex-20261103/text-transfer-incident.json), private terminal proof `logs/modal-text-transfer-terminal.json`.

Leased text writes now use the existing bounded SHA-verified atomic artifact receiver, with the1MiB UTF-8 cap and per-operation lease guards.71 focused tests and independent review passed; an additional182 sandbox/artifact/reporting tests passed. The full suite passed1166tests with3optional-JAXskips (logs/tests-notice-transfer-final.log). Commit, freeze and deploy this source, then use the normal serialized CLI launch with `.openwsp/codex-text-transfer-resume.json`. Resume the same conversation, journals, saved edits, work files and counters; the recreated kernel gets the existing neutral restart notice. Verify turn51 notice admission once. No model request or GPU failed at this notice step.

The earlier ordinary-generation integration failure is fixed and verified on actual paired generation requests20 and21. The forced-close hook is omitted from ordinary engines; grid use explicitly opts into the separately tested V2 adapter. [Startup incident](replications/exp8/results/live-runs/codex-20261103/generation-startup-incident.json). Eight failed requests remain in the trajectory and costs. Twelve affected child calls are provider-confirmed terminal; four unresolved edited-partner accounting bounds remain reserved. Do not erase those bounds without billing evidence.

## Funding and editor transport

Latest ledger: **$65.77025538028428 Modal spent,$18.129953472222223 reserved,$66.0497911474935 available** under the **$150 total task cap**. These are conservative accounting values, not invoices. Automatic-top-up authorization `operator-auto-topups-20261103-steering2` supersedes the old$42 gate. No provider dashboard increase is claimed. This notice failure is not a budget block. Preserve all costs and reservations; the next serialized launch reconciles the terminal driver owner.

Subscription Astra remains primary. Codex0.163.0, model gpt-6-astra, effortmax, native tools disabled, credentials local only. `.openwsp/codex-subscription` holds private authentication; `.openwsp/codex-workers/codex-20261103` holds immutable worker identity and response journals. Never print or send credentials to Modal. Do not replay an admitted request or combine subscription and API attempts.

Historical API spend **$78.292765**, remote cap100, reserved0. The authorized additional$160.85 allocation remains unapplied; the separate **$160.85 PlanB reserve is protected**. No API fallback or new paid probe. Native Fable has a separate **$20 total cap** for CPU experiments, but its credential is absent; the earlier async configuration question remains pending. No Fable calls have been made.

## Completed work

Calibration: base36/500(7.2%), p50c253/500(50.6%), both within prescribed tolerance. Mandatory controls: thinking-off234/500(46.8%), mechanical230/500(46.0%), repetition1.1 119/500(23.8%). All2500 compact records are public. Finished-only metrics are recomputed from canonical stop/length evidence and sample hashes; old publication briefly removed these fields, now restored. Mechanical control uses calibration answers and is a diagnostic.

Kernel hash, refusal/isolation, real forward internals and fresh-engine fallback evidence are public. Native vLLM weight reload is unavailable. Calibration and kernel-verified tags exist; live-run and v0.1.0 do not.

Astra raw CPU comparisons are complete:11 replies and11 scores, one reply per released prompt, no tools or feedback. Labels4/4; shapes3/3; car0/1024. MinAtar over1000episodes each: Breakout458.730±4.994560; Asterix39.709±0.730104; Seaquest2.407±0.105937. All-source-seed PPO and source-network calibration are separate reference comparisons. Public v1 edits are bfloat16 interchange exports; exact scored policies were float32, with hashes and parser/source provenance published and full archives private. [Comparison](replications/raw_weights/results/comparison.md). Fable is the only pending raw comparison; experiments3,6,7 remain layout only.

Inspect CLI/SDK/MCP, actual CPU geometry, finished-only metrics,11 grid definitions, repair2a/2b, KL math/backend/stage, forward and evaluation base banks, and17734-byte execution ceiling are implemented and tested. Preserve true SHA256-based problem seeds and first8192 answer-token float16/NumPyfloat32 activation reduction. No global sandbox volume reload. No new ledgers, budget abstractions, watchdog modes or reporting automations.

## Long CPU extraction

Variant2c input extraction is **PID39594**, two threads, projected about13.5hours; latest snapshot38/484 examples. All484 original control2 examples are required. Logs `.openwsp/repair-inputs/full-extraction.{pid,log,exit}`; receipts `.openwsp/repair-input-means`. Single continuation **PID44670** waits for that exact process and verifies all receipts before building2c and seven dose bundles. Logs `.openwsp/repair-inputs/completion.{pid,log,exit}`. No duplicate process, partial contrast, GPU scoring or automatic push. Environment refreshes can discard exec-session handles while both PIDs survive.

## Remaining order

1. Resume this faithful attempt and reach its actual50-pass submission; run MATH500×4, GSM8K×2 and IFEval×2.
2. Control4, then the11 preregistered grid cells in `replications/exp8/eval_grid.py`, then neutral-reference KL. Use `.openwsp/codex-post-faithful.json` with the normal CLI and foreground worker. Ordinary evaluation remains processor-free; forced-close cells require the real four-problem GPU smoke.
3. Publish final evidence and report, meet remaining done criteria and tag only when warranted. Run2 requires fresh operator confirmation. No optional repair-scoring GPU work before the required priority order.

Frozen8a3b9f3 passed1139tests,3optional-JAXskips. Completion revision2bb69da passed Python3.11/3.12 CI at https://github.com/openwsp/openwsp/actions/runs/41709165531. The current notice-transfer fix passed1166tests with3optional-JAXskips; deployment is next. All prior history remains in Git and DECISIONS.md.
