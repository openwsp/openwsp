# Recording Modal funding evidence

New ledgers start with a **$42 billing-cycle allowance** and a separate **$150
cumulative task cap**. These are initial defaults. A confirmed account-limit
observation can change the allowance; only an explicit task authorization can
change the task cap. Expected payments and automatic limit increases are not
observations that those events occurred.

`record-modal-allowance` records evidence through the existing serialized
coordinator. It does not deploy, create a ledger, launch or restart a driver,
cancel work, or run recovery. A driver that is active, has uncertain provider
state, or has an unmatched retained ownership ticket prevents all mutations.
A saved `blocked` status alone does not prove termination.

The current task's [authorization](../replications/exp8/authorizations/modal-budget-20261104.json)
and [dashboard evidence](../replications/exp8/authorizations/modal-dashboard-20261104.json)
are source records, not proof that a ledger update was applied. See
[STATUS.md](../STATUS.md) for the current owner, applied limits and pending work.

## Prepare the account observation

Save the evidence at an absolute local path, such as
`/tmp/modal-allowance.json`. Replace every placeholder with confirmed evidence;
the following template is intentionally invalid as written. Amounts are USD.
Use the dashboard's total usage limit and current usage, including credits as
represented there. Its separate charge-based spend limit is not the same field.

```json
{
  "observation_id": "<unique-stable-observation-id>",
  "cycle_limit": "<confirmed-total-cycle-usage-limit>",
  "cycle_usage": "<confirmed-current-cycle-usage>",
  "evidence": {
    "source": "<specific-dashboard-confirmation-or-evidence-record>",
    "observed_at": "<timezone-aware-UTC-timestamp>",
    "basis": "user_confirmed_dashboard"
  },
  "cycle_id": "<confirmed-billing-cycle-identifier>"
}
```

`cycle_id` is optional when unavailable. New observations must be within
15 minutes and newer than the last observation. Evidence contains no credentials
or editor conversation material. A billing summary can establish reported usage;
it does not by itself establish the account limit or payment status.

## Optional cumulative task authorization

To apply a separately authorized task cap in the same transaction, prepare a
second absolute-path JSON file. Its top-level fields are exactly `total_limit`,
`authorization_id` and `evidence`:

```json
{
  "total_limit": "<explicitly-authorized-cumulative-task-cap>",
  "authorization_id": "<unique-stable-authorization-id>",
  "evidence": {
    "source": "<specific-operator-instruction-or-authorization-record>",
    "authorized_at": "<timezone-aware-authorization-timestamp>",
    "basis": "explicit_user_authorization"
  }
}
```

The amount is a cumulative ceiling, not an additional allowance. It cannot be
below retained settled estimates plus reservations. It does not assert or change
the provider's account limit. The CLI validates both files before contacting
Modal; the coordinator applies the authorization and observation atomically
only after proving that the ledger has no active driver owner.

## Provider usage below conservative estimates

Ordinary observations preserve the ledger's greater known-usage floor. When
fresh, confirmed provider usage is lower than the conservative settled task
estimates, an explicit anchor can distinguish the two without rewriting any job
receipt. Add this object inside the observation's `evidence` only after verifying
that all prior dispatches are terminal:

```json
"usage_anchor": {
  "basis": "provider_actual_cycle_usage",
  "settled_task_estimate_usd": "<exact-current-settled-Modal-estimate-total>",
  "all_dispatch_terminal": true
}
```

The outer evidence basis must remain `user_confirmed_dashboard`. The exact
settled estimate must still match when the update is applied, and the observation
must still be fresh. The terminal flag records an actual verification; it is not
a substitute for one. The serialized coordinator independently checks current
and retained driver ownership before passing its idle-owner proof to the ledger.

This operation anchors cycle usage to the reported provider amount and retains
the unchanged task-estimate total as its baseline. Later estimated spend is added
to that baseline and every outstanding reservation remains deducted. It neither
settles individual jobs from aggregate billing nor changes the task cap unless
the separate explicit authorization is supplied. Existing receipts, historical
events, editor limits and a measured-overrun halt remain intact.

## Record and resume separately

The deployed definitions must already support the requested arguments. This
command will not update an older deployment or fall back to launching work. If
an update is needed, first establish that the previous driver is terminal, test
and review the source, then deploy definitions from a frozen checkout:

```sh
OPENWSP_EDITOR_TRANSPORT=codex .venv/bin/python scripts/modal_cli.py deploy -m openwsp.backends.modal_app
```

Record both files together, or omit `--task-authorization` to record only the
account observation:

```sh
openwsp record-modal-allowance exp8 \
  --observation /tmp/modal-allowance.json \
  --task-authorization /tmp/modal-task-authorization.json
```

The returned audit receipt identifies what was applied. Preserve both original files and IDs for an exact
retry after a lost acknowledgement. An identical retry does not restore spent
funds or a superseded task cap; changing either payload under its existing ID is
rejected. The optional authorization is bound into the observation receipt.

Recording funding evidence leaves a blocked run blocked. Resume separately with
the existing run ID, stage, config, local Codex paths and conversation journal.
Add a nonempty `resume_reason` identifying the accepted observation and repaired
blocker, then use the normal foreground command:

```sh
openwsp replicate exp8 --stage all --config /absolute/path/to/resume-config.json --foreground
```

This replication command deploys and may launch work. Read the current
coordinator first so an existing watchdog recovery is attached to rather than
duplicated. Never replace the run ID or editor journal to bypass an uncertain
owner or budget boundary.

## Earlier automatic top-up authorization

The earlier [operator top-up record](../replications/exp8/authorizations/modal-topups-20261103.json)
uses `basis: "operator_automatic_topups"`, `automatic_topups_authorized: true`
and `provider_limit_observed: false` inside `evidence`. That existing route accepts
a funding envelope of at most $150; it does not observe a provider limit or raise
the cumulative task cap. A larger explicitly authorized task cap uses the
separate authorization above. Neither route infers permission from a successful
charge or an expected future limit increase.
