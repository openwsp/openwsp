# Recording a confirmed Modal allowance

The configured Modal billing-cycle allowance remains **$42** until the user
confirms a higher limit and current usage in the Modal billing dashboard. An
expected increase after a payment is not evidence that the increase occurred.
The separate **$150 total task ceiling** remains unchanged by this operation.

`record-modal-allowance` records that confirmation in the existing budget ledger.
It uses the already deployed, serialized coordinator. It does not deploy an app,
create a ledger, start or restart a driver, cancel work, or run recovery. The
coordinator may inspect the recorded provider call to establish that its owner
is terminal; an active or uncertain owner prevents the update.

The deployment must already contain this command. The CLI does not update an
older deployment or fall back to a launch operation. Revision `89c321c` predates
the allowance keyword and will reject it until the definitions are updated.

## Prepare the observation

After the user confirms the dashboard values, save a JSON file at an absolute
local path, for example `/tmp/modal-allowance.json`. Replace every angle-bracket
placeholder below with the confirmed value; this template is not a valid
observation as written. Amounts are USD. Use the dashboard's total cycle limit
and its current cycle usage, including credits as represented there.

```json
{
  "observation_id": "<unique-stable-observation-id>",
  "cycle_limit": "<confirmed-total-cycle-limit>",
  "cycle_usage": "<confirmed-current-cycle-usage>",
  "evidence": {
    "source": "Modal billing dashboard, confirmed by the user",
    "observed_at": "<timezone-aware-UTC-timestamp>",
    "basis": "user_confirmed_dashboard"
  },
  "cycle_id": "<dashboard-billing-cycle-identifier>"
}
```

`cycle_id` is optional; omit it if unavailable. The evidence file must contain
the observation fields only, with no credentials, authentication material, or
model conversation context. Optional evidence metadata can identify audit records.
New observations must be current. Keep the original file and observation ID for
an exact retry if the acknowledgement is lost: replaying the same observation
does not restore consumed allowance, while changing its payload under the same
ID is rejected.

## Update definitions if needed

Before updating an older deployment, confirm that its previous driver owner is
terminal at the provider and complete the local tests and review. A recorded
`blocked` snapshot alone is not proof of provider termination. From a frozen
checkout, the current subscription route uses this definition-only deployment:

```sh
OPENWSP_EDITOR_TRANSPORT=codex .venv/bin/python scripts/modal_cli.py deploy -m openwsp.backends.modal_app
```

This command updates the deployed definitions; it does not invoke `launch`,
record an allowance observation, or resume a driver. Keep the `codex` transport
selection for this subscription run. User confirmation of the dashboard increase
is still required before recording the observation or resuming work.

## Record, then resume separately

Once the user has confirmed the increase and the file contains that evidence:

```sh
openwsp record-modal-allowance exp8 --observation /tmp/modal-allowance.json
```

The command prints an audit receipt. It changes the observed Modal allowance;
it does not change the task ceiling, editor budget, usage receipts, outstanding
reservations, or a halted ledger. Recording the allowance also leaves any
blocked run blocked.

Resumption is a separate explicit action. Preserve the existing run's config,
run ID, selected stage, and local Codex state; add a nonempty `resume_reason`
explaining the confirmed allowance change and identifying the accepted
`observation_id`. Preserve the same `run_id`, `codex_local` paths, and existing
editor journal; do not start a fresh conversation. For a Codex run originally
launched with stage `all`, add a concrete reason to a copy of its existing config,
replacing the placeholder with the accepted observation ID:

```json
{
  "resume_reason": "Modal dashboard allowance observation <accepted-observation-id> recorded; previous driver confirmed terminal; resume the existing run."
}
```

The fragment above is an additional field, not a replacement for the existing
run config. The later resume uses the normal foreground command:

```sh
openwsp replicate exp8 --stage all --config /absolute/path/to/resume-config.json --foreground
```

Unlike the allowance command, this normal replication command deploys and may
launch work. Use it only after the allowance receipt is accepted and within the
user's authorized task scope. Do not substitute a new run ID or a fresh local editor
journal to get around a blocked or uncertain run.

## Explicit automatic top-up authorization

An operator may explicitly authorize ongoing automatic top-ups within the existing task cap. The same command accepts `basis: "operator_automatic_topups"`, `automatic_topups_authorized: true` and `provider_limit_observed: false` in `evidence`. In this case `cycle_limit` is the authorized funding envelope, at most$150, not an observed provider dashboard limit. `cycle_usage` remains a conservative known-usage figure; the ledger always retains its greater existing usage floor. Include the exact instruction and source in evidence. This route does not infer permission from a successful charge or expected limit increase, and does not change the task cap.

The operator's direct instruction accompanying steering update2 supplies that authorization for this task. Record it, retain the returned receipt ID in `resume_reason`, and resume the same `codex-20261103` journal. The earlier dashboard-only instructions describe the prior gate.
