# Status

Goal in progress; none of the six completion criteria is complete in full. The exact task and the user's answers are in [docs/TASK.md](docs/TASK.md). Current Modal allowance: $42 including credits. The task may use up to $150 only after evidence of an increased allowance. Editor cap: $100. No further questions.

- Main commit `b9ebfb5e4ed9c5378b6a4c27d91cded1c9555b85` is pushed and CI is green: https://github.com/openwsp/openwsp/actions/runs/41590756728. The latest working-tree CPU suite passed **268 tests** in 48.76 seconds; Ruff and secret checks pass. Protocol changes are being committed next.
- The authors' p50c edit rebuilt the exact two published tensor hashes through `save_edit` and checkpoint export on the real pinned weights: [hash verification](replications/exp8/results/kernel/hash-verification.json). Production sandbox first-cell/refusal/isolation verification remains pending.
- Full base calibration is running. Launched with `openwsp replicate exp8 --stage calibration --config /tmp/openwsp-resume-calibration.json`; call `fc-01M6G1PPP266WAAV5S8J975H8H`; log `logs/modal-calibration-resume.log`. The four-problem smoke completed and scored 0/4; this is a diagnostic, not the 500-problem result. Its saved GPU receipt was reused after fixing an unreleased, grading-independent import in the authors' grader. No full calibration score yet.
- Earlier GPU startup failed because nvcc was missing; that app was stopped. The CUDA 13.0.3 devel image now works. Failed-startup reservation $2.27584 remains retained. Latest observed ledger: Modal estimated spend $0.45725, reserved $7.16883, available $34.32392; editor $0. Provider billing lags and does not expose the Starter allowance. See `logs/modal-build-billing-observation.json`; do not infer an increased cap.
- Generic tooling now includes durable GPU jobs, sandbox and driver budget leases, a serialized coordinator, a 15-minute watchdog with at most three restarts per stage, a native vLLM swap verifier, and a publisher restricted to compact results and generated README tables. CI's real MCP client exercises every tool.
- Protocol modules are written: `runtime`, `pipeline`, `forward`, `internals`, `live`, `evaluation`, `controls`, `verification`, `artifacts`. The live driver has 50 passes, 300 Python runs, notices at turns 46/51, handoffs every 40 turns, exact saved request formats, and crash-recovery tests. Final evaluation is MATH-500 x4, GSM8K x2, IFEval x2. Controls 1–3 are implemented. The official IFEval checkers are pinned and attributed.
- These production paths still need real sandbox/GPU checks. New `verify_forward` checks native swaps against fresh-engine log probabilities, teacher-forced arrays and eager attention. On swap incompatibility it records the permitted fresh-engine fallback. The current remote deployment predates these newest checks; deploy the committed version before starting the live run.
- `all` is wired through calibration, kernel/forward checks, live run, evaluation, controls and publication. A calibration gap requires a recorded prompt/template/sampling audit and circuit-breaker evidence before continuing. Controls currently defer to the user's spending priority; reconsider parallel execution only if verified headroom permits. No live editor request has been sent.

## Next

1. Commit/push the tested protocol implementation. Monitor calibration every 10–15 minutes with `.venv/bin/openwsp status`.
2. Inspect full base/p50c scores against 7.9 ± 2.4 and 48.1 ± 4.4 percentage points. Debug prompt/template/sampling first, within the two-hour limit; record any unresolved gap honestly.
3. Deploy and run production kernel/forward verification (`openwsp replicate exp8 --stage kernel`), then launch `openwsp replicate exp8` as soon as calibration and kernel checks pass or their specified breakers trip.
4. Finish the actual 50-pass run, its evaluation, required controls and published report. Update README FAQ/deviations from evidence. Tag only completed milestones. If an external budget/provider blocks a criterion, record exact resume commands and mark the goal unmet.

## Environment and recovery

Workspace: `/workspace/openwsp`; home is read-only. `.venv` contains locked CPU/dev/Modal/replication dependencies; `/tmp/openwsp-venv` is the bootstrap venv. Preserve the proxy and CA bundle. `configure_tls()` adds the environment CA to grpclib's trust; Modal's proxy extra is installed. Exec network permission is also needed for asyncio test socketpairs here.

Modal app: `openwsp-exp8`; v2 volumes: `openwsp-exp8-data`, `-model`, `-practice`, `-work`. The editor sees only read-only model/practice and its work subpath. All three required secrets exist. Auto-review rejected refreshing the GitHub secret, so the existing secret was left untouched; deployment validates its `GITHUB_TOKEN` key. Never print credentials.

Source checkout: `/tmp/openwsp-upstream`, pinned `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Local real weights: `/tmp/openwsp-model`, `/tmp/openwsp-text-model`; restored p50c: `/tmp/openwsp-p50c-submission`. Deployments use immutable temporary source snapshots. Git's credential helper reads `GITHUB_ACCESS_TOKEN` at runtime. The original uploaded task explicitly authorizes direct commits and pushes to main after tests; no PR is requested.

Coordinator state is `/data/coordinator.json`; stage state `/data/state.json`; costs `/data/budget.json`. Ambiguous paid POSTs retain reservations and never automatically redispatch. An explicit repaired-stage resume can use a config containing `resume_reason`; automatic watchdog recovery never clears authentication, quota or budget blocks. The detached pipeline publishes only results and README tables. Latest code must be on main before its publisher clones main.
