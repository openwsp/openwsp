# Status

Goal in progress; initial questions answered in docs/TASK.md. All completion criteria still pending in full.

- Core weights/kernel/versioned edits, shared API/CLI/SDK/MCP, budget ledger/editor adapter, isolated sandbox transport, report/transcript tooling, docs and CI implemented. CPU suite previously 87 passed; latest expanded suite running. MCP real client works with exec network enabled (required here for asyncio socketpair wakeups).
- Exact p50c hash verification passed using real pinned Qwen weights through save_edit and checkpoint export: replications/exp8/results/kernel/hash-verification.json. Production sandbox first-cell/refusal check remains.
- Modal connection works: use scripts/modal_cli.py or configure_tls() to extend grpclib trust with environment CA; proxy-support extra required. All three secrets exist. Refreshing github secret rejected by auto-review; left untouched, driver requires existing GITHUB_TOKEN key.
- Current launch: `openwsp replicate exp8 --stage prepare`, log logs/modal-prepare-launch.log. Builds images; no generation/editor call yet. Earlier image dependency resolution failed; now transformers5.17.0 matches authors and vLLM0.33.0.
- Protocol scaffolding: replications/exp8/{config,runtime,pipeline}.py/json and src/openwsp/backends/modal_app.py. Preparation/calibration smoke/full loops written, not yet validated remotely. `all` calls live module NOT YET IMPLEMENTED. No watchdog/controls2/teacher-forced/live/final eval/publish implementation yet. Do not claim completed pipeline.
- Spend: editor $0/$100; Modal image build costs pending billing reconciliation, no GPU jobs. Cap remains $42 cycle (user's prior few cents conservatively $0.05); $150 task maximum only after verified higher allowance. Billing SDK has usage/rates but no Starter limit; logs/modal-billing-observation.json.
- No first commit yet; must run tests then commit/push logical tooling step. git main tracks origin/main; runtime credential helper configured. Secret hook installed.

Next: finish image/preparation launch, commit tested tooling, smoke generation4 then calibration500 base/p50c. Build live driver while calibration runs; complete production kernel check. No question to user; record defaults in QUESTIONS/DECISIONS.

Resume: /workspace/openwsp; read docs/TASK.md and this file. `.venv` holds all CPU/test packages; /tmp/openwsp-venv holds Modal/bootstrap tools. Source /tmp/openwsp-upstream pinned4d09ba6d6a7e891dd8063974dfe81b3d78e3494f. Model /tmp/openwsp-model (filename model.safetensors-00001-of-00001.safetensors), text subset /tmp/openwsp-text-model, p50c restored /tmp/openwsp-p50c-submission. Preserve proxy+CA. Network tool commands require with_additional_permissions network.enabled=true. Never print credentials.
