# Experiment 8 with a ChatGPT subscription

The subscription route uses Codex CLI on a persistent local worker and Modal for the kernel, forward passes and evaluation. The worker keeps ChatGPT authentication and Codex conversation files locally. Modal exchanges immutable prompts and responses with it; ChatGPT credentials are not copied into a Modal secret or container. An OpenAI API key is not required by the Codex deployment.

Install the pinned Codex CLI version `0.163.0` and sign in through its standard ChatGPT login. Provision a private writable Codex home and worker-state directory, each owned by your user with mode `0700`. Keep both outside `src/`, `replications/` and `scripts/`, which are deployed to Modal. In this managed workspace, the ordinary home is read-only, so the CLI uses a separate ignored directory under `.openwsp`.

Put absolute local paths in a private configuration file:

```json
{
  "editor": "codex",
  "run_id": "codex-attempt-1",
  "codex_local": {
    "home": "/absolute/private/codex-home",
    "state_dir": "/absolute/private/editor-workers"
  }
}
```

Launch with:

```sh
openwsp replicate exp8 --config /absolute/private/codex-exp8.json --foreground
```

Keep this process and its host available. The command deploys the Modal pipeline, launches or resumes its serialized driver, and services the editor requests in the foreground. Local-only paths are removed before the remote configuration is submitted. The worker uses the exact pinned Astra profile; it does not select a model based on quota or silently use an API key. The [model/profile layer](editor-architecture.md) supplies extension points for other editors without changing the kernel or forward contract.

Each `run_id` has isolated live state, kernel files, GPU request identifiers, Codex conversations and final evaluations. Reusing the same configuration resumes that attempt. A new ID means a separate attempt from the original weights and zero counters; it is not a migration of another editor's reasoning or partial edit. The earlier API attempt remains recorded. All attempts appear in the generated report, and the active attempt must finish exactly fifty passes and all required evaluations before its goal is reported complete.

The bridge binds each response to its exact prompt hash and worker. A durable dispatch is not reassigned after a timeout. A restarted worker can deliver a saved response without generating it again, but it must retain the same identity and Codex journal. Missing local evidence after dispatch blocks the request instead of starting another model call. Use the same configuration to resume only after resolving the recorded error; an authentication, quota or uncertain-request failure is not a reason to delete the worker state.

The local [wire fixture](../scripts/check_codex_transport.py) verifies the pinned initial and resumed request with no native tools or added experiment instructions. The [real synthetic check](../replications/exp8/results/kernel/codex-subscription-probe.json) verifies subscription authentication, short fenced output, exact-session resume and persisted reasoning. These checks do not establish long-run reliability. Codex's own supported transport retries still apply, and stopping its process is not proof that a remote generation incurred no usage.

Subscription counters are reported separately from API dollars. They do not make GPU computation free or expand the Modal cap. Quota exhaustion pauses the editor; the API adapter and its [explicit recovery workflow](editor-recovery.md) remain available, but switching transports requires an honest conversation boundary or a separately reported attempt. No failed output or abandoned partial edit is selected as a final submission.
