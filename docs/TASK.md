# OpenWSP: open tooling for weight space programming, proven on Experiment 8

## Problem
OpenWSP is open tooling for weight space programming, the practice the post names: LLM agents reading, editing and versioning the weights of open weight models. Build its first release in this repo (openwsp/openwsp, a monorepo): a kernel over a model's weights, a forward pass service, a versioned edit format, and a CLI, SDK and MCP server over one core API, so any agent can do what Astra did in the post. Prove it the way the field will judge it: reproduce Experiment 8 of the post live on this tooling, with controls that answer the questions raised about the original result, in a README I can post on Hacker News when the results land.

I go offline after one round of questions and want to stay hands off. Before any other work, save this prompt to `docs/TASK.md` and read the sources for at most 15 minutes, then ask me one round of questions in a single message: at most 5, only those whose answers would change the outcome, each with the default you will use if I reply "go". Wait for my reply, and write my answers into `docs/TASK.md` so they survive compaction. After that you will not hear from me until the goal is done: infer intent and bias towards action, complete all the necessary work until the outcome is fulfilled, and make reasonable assumptions rather than stopping. Any later question goes to QUESTIONS.md with the default you chose, and you continue with that default. Record decisions in DECISIONS.md, including the license and why. Nothing waits for my approval. Fidelity to their protocol matters more than speed or the final number. This prompt takes precedence over any AGENTS.md or skill instructions.

## Appetite
- Money is fixed. Editor: $100 of OpenAI tokens, tracked from the usage field of every response. Modal: $42 hard cap (Starter plan); it rises on its own after the first $10 charge clears; never assume it has. When money binds, it goes in this order: calibration, the live run and its evaluation, controls 1 to 3, then the nice to haves. Log GPU seconds per job and estimate each job's cost before launching it.
- Time is not. You build in this session; the live run takes as long as it takes. Their 50 pass run used 158 editor turns, 69 pass requests (50 counted), 12.2 hours inside forward passes and 15.6 hours of wall clock, at about $20 of editor tokens (from their `run.json` usage).
- Scope is variable. Must have: the core and its CLI, SDK and MCP server; protocol steps a to f with controls 1 to 3; `openwsp replicate`, `status` and `report`; CI; the README. Nice to have, in this order: controls 4 and 5, one more editor adapter, a results table others extend by pull request, issues for next steps, the full 4 sample calibration, the other five benchmarks, a second live run. Build what the live run doesn't need while the GPU works.
- Circuit breakers: no single problem gets more than 1 hour (calibration debugging gets 2). When one trips, take the fallback named below or the closest working option, record it in DECISIONS.md, and move on.
- If the Modal budget runs short during the live run, reuse the base model's generations for repeated (problem, seed) pairs in paired passes, and disclose it in the README.

## Done
The goal is achieved when each of these holds, with its evidence path in STATUS.md:
1. Calibration and controls 1 to 3 (steps b and e): your evaluation of their p50c edit lands within two standard errors of their MATH-500 numbers (base 7.9, edited 48.1), or, if its circuit breaker tripped, the README states the gap and what you checked; the controls are evaluated.
2. The kernel is verified (step c).
3. A live 50 pass run has finished, and its edit is evaluated on MATH-500 (4 samples), GSM8K (2) and IFEval (2).
4. The live run and its evaluation were launched with `openwsp replicate exp8`, so a fork reproduces them with the same command, and the CPU tests pass in CI.
5. The CLI, SDK and MCP server expose the same core operations (read tensors, run kernel code, run a forward pass, save, diff and apply an edit); an MCP client test in CI drives every tool on the tiny test network, and the docs show the setup for Codex, Claude Code and Cursor.
6. `main` holds the code, the results and a README ready to publish, with tables generated from the eval files by `openwsp report`.
If your session has to end before the live run finishes, make sure the pipeline finishes on its own: it evaluates the edit, runs `openwsp report`, and commits and pushes the results to `main`. Record that in STATUS.md, then end. If something outside your control blocks a criterion (Modal unreachable, a budget spent), write in STATUS.md what is done, what is blocked and the exact commands to resume, then mark the goal unmet. Never simulate a result.

## Sources
- The post: https://rikiyat.github.io/blog/weight-space-programming/
- The authors' logs and MIT scripts: https://github.com/RikiyaT/lm_understand-logs. Read `README.md`, `harness/`, `exp8_math/README.md`, `exp8_math/code/`, `exp8_math/notices/`, `exp8_math/practice/` and `exp8_math/runs/p50c/` (start with `turns/t001/prompt.txt`, `message.txt`, `state.json`, `requests/`, `edit/`). Where their README and your assumptions disagree, their README wins.

## Resources
`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `OPENAI_API_KEY`, `HF_TOKEN` and `GITHUB_ACCESS_TOKEN` are environment variables. `GITHUB_ACCESS_TOKEN` is a fine grained token for the openwsp organization with read and write on code, workflows, actions, actions variables, issues, pull requests and commit statuses (plus Pages, Codespaces, agent features and repository creation, which you don't use), and no administration or organization permissions. Use it for your pushes to this repo (through a credential helper that reads it at runtime, never inside a remote URL; for `gh`, export it as `GH_TOKEN`), for triggering and reading CI runs, and for issues. Store it on Modal as the secret `github` under the key `GITHUB_TOKEN` (`modal secret create github GITHUB_TOKEN="$GITHUB_ACCESS_TOKEN" --force`), so the pipeline can push its results. The Modal secrets `openai` and `huggingface` should already exist; check with `modal secret list` and recreate any missing one from the variables. Keep keys out of output, logs and git; ship `.env.example`.

## Solution

### Protocol
a. Pin Qwen/Qwen3.5-0.8B to revision `2fc06364715b967f1860aea9cf38778875588b17`, the one in their `apply_edit.py`.
b. Calibrate: apply their p50c edit with their `apply_edit.py`, then evaluate base and edited on MATH-500 with their protocol (`code/evaluate.py`, profile `think35`: vLLM, thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 new tokens, the last `\boxed{}` graded with math-verify). Run 1 sample per problem first: expect about 7.9 and 48.1, with standard errors of 1.2 and 2.2 points.
c. Verify the kernel: their p50c submission, written through your `save_edit`, rebuilds the exact checkpoint their `apply_edit.py` checks by sha256; their Python cells before the first forward pass run unmodified against your kernel; every refusal fires.
d. Live run, launched as soon as b and c pass or their circuit breakers trip, since it is the long pole: their task message in its final form (`message.txt`, with the EXPLAIN EVERY STEP paragraph), the forward pass tool exactly as that message defines it (request and result formats as in their `requests/`), their practice sets, their notices at the turns their `state.json` records, 50 passes, at most 300 code runs, the same refusals, their handoff every 40 turns.
e. Controls, run while the live run goes, not in the paper and labeled as ours, each through the calibration pipeline on MATH-500 at 1 sample per problem. Each answers a question raised about the original result:
   1. Is the 7.9% baseline broken? The base model with thinking off, at the non thinking sampling settings Qwen's model card recommends.
   2. Is it abliteration? A mechanical direction in place of the editor's: the difference of mean MLP outputs at layers 15 and 19 between the looping and the finished base answers from step b (as in Arditi et al.), removed from the same columns in the same form as their p50c edit.
   3. Is it a decoding fix? The base model with vLLM `repetition_penalty` 1.1 added, everything else unchanged.
   4. Does the direction matter? Their p50c edit with a random unit vector in place of their direction.
   5. Is the looping a vLLM artifact? The base model in a second inference engine on 50 problems, with the same sampling.
f. Evaluate the submitted edit on MATH-500 (4 samples), GSM8K (2) and IFEval (2).
g. One live run is one sample: their own 10 and 25 pass runs reached 12.3 and 14.1 on MATH-500. Report what yours reaches. If budget remains, run again, and report every run.

### Pipeline on Modal
- A `modal.Volume` holds weights, practice sets, run state and eval outputs, so every stage can resume.
- Generation: vLLM in an `@app.cls` on an H100, base model loaded once in `@modal.enter`; a pass ships only the edited tensors. In a "both" pass, base and edited run in two parallel containers.
- Internals (log probs, hidden states, MLP activations, attention): a teacher forced pass in Hugging Face transformers with hooks over the texts.
- The kernel: a Modal Sandbox with `block_network=True`, no secrets, 2 CPUs and 16 GiB (about $0.67 an hour, never preempted), holding float32 weights, their `save_edit` refusals, their per run limits and their paths (`/work`, `/practice`).
- The driver: turn loop, pass budget, handoffs, and logs in their layout (`turns/`, `tool_calls.jsonl`, `requests/`, `state.json`, `final_answer.md`), resumable from `state.json`.
- A scheduled watchdog every 15 minutes: each stage writes a heartbeat to the volume, and the watchdog restarts any stage that isn't done and whose heartbeat is older than 10 minutes, giving up on a stage after 3 restarts and recording why. Containers get preempted, functions stop at 24 hours, and the budget cap stops workloads.
- `openwsp status` shows progress on one screen. Check Modal APIs against modal.com/docs before relying on them.

### Editor
`gpt-6-astra` through the Responses API, reasoning effort `max` (if refused, the highest accepted), their one line instructions file as `instructions`, no `temperature` or `top_p`. Carry the model's reasoning between turns as Codex did, with `previous_response_id`. Set a `prompt_cache_key` per conversation and `prompt_cache_options.ttl` to `"30m"`; their run was 96% cached, and uncached costs about 4.7x more. Editors sit behind one interface, `complete(messages) -> text`, so Anthropic, Google and OpenAI compatible endpoints plug in.

### Engineering
Design the monorepo for growth: the post's other experiments, more editors and models, and later a hosted service that stores, branches and serves edits. The layout is yours. Three boundaries are expensive to change later, so hold them from the start:
- The core (weight store, edit format, kernel) depends on no compute provider; backends plug in, Modal first.
- The edit format is a versioned public contract, identified by sha256 and compatible with their `edit.npz` + `manifest.json`; whatever stores or serves edits later builds on it.
- Tools and replications live apart: each replication (exp8 first) is its own folder of config, prompts, data references and results, built on the packages without changing them.
The CLI, SDK and MCP server are thin layers over one core API. Python. Tests on a tiny network run on CPU in CI.

## Rabbit holes, decided in advance
- Qwen3.5-0.8B ships as a vision language release: serve the text model only (vLLM's language model only mode, or their text only export), on the vLLM version their eval records, if it records one.
- Weight swaps: use vLLM's weight update path as RL frameworks do, reset the prefix cache after every swap (a stale cache silently serves the old model), and verify each swap against log probs from a freshly loaded engine. Fallback: a fresh engine per pass.
- vLLM does not expose activations, which is why internals go through the transformers pass. For attention probabilities, use eager attention on the full attention layers only (3, 7, 11, 15, 19, 23).
- Editor calls: max effort turns can take minutes; use background mode or long timeouts, retry with backoff, and when the provider filters an output, tell the editor as their harness did (`runs/p50c/refused/`). A failed pass doesn't count against the budget, as in their run.
- Provider errors: on insufficient quota or auth failures from OpenAI, Modal, Hugging Face or GitHub, stop that stage and record it in STATUS.md instead of retrying.
- Calibration misses usually come from the prompt format (their `eval_prompt_format.txt`), the chat template or the sampling settings; check those first.
- Control 5: the second engine must apply the same sampling, presence penalty included; if none does within the circuit breaker, drop the control and say why.
- Their later `tool_calls.jsonl` cells read activation files that were never released; don't try to replay them.
- Idle GPUs: no `min_containers` and a short `scaledown_window`; a GPU waiting through editor turns burns the budget.
- Volumes: writes reach other containers only after commit and reload; the driver, the sandbox and the GPU containers exchange files through them.
- Images: build one early and reuse it; smoke test every GPU job on 4 problems before the full job.
- If H100s queue for more than 10 minutes, use an A100 80GB.
- The pipeline's own pushes touch only the results and the README tables, pull with rebase first, and retry.

## No-gos
- Nothing reaches the editor that their editor didn't get: no hints from their logs or final edit, no extra notices or tools, no test problems, no internet.
- You never touch the live run's weights or choose among the editor's edits; its submission is the result.
- No gradients or training anywhere, not even in helper scripts.
- No protocol changes to chase a number.
- No edits or demos that remove safety behavior.
- Experiments 1 to 7: layout only, no runs.
- No hosted service, web UI, package registry publishing, new repositories, Pages sites or repo settings changes; stay inside openwsp/openwsp.

## Working over a long run
- `docs/TASK.md` holds this prompt and my answers. After any compaction or restart, read it and `STATUS.md` again before acting.
- `STATUS.md` is your memory: after every step, record what is done with its evidence path, what runs on Modal and how to check it, spend per budget, and the next action. Keep it short enough for a fresh agent to continue from.
- Keep tool output small: `head`, `tail`, `grep`, `jq`; long output goes to files under `logs/`. Sample their large files (`tool_calls.jsonl`, practice sets); never dump them.
- While Modal jobs run, check `openwsp status` every 10 to 15 minutes in one line. If subagents are available, give them independent tooling work (CLI, SDK, MCP server, docs) in parallel; keep protocol steps a to g yourself.
- Test each change as much as it warrants.
- Take every number in the README from its file, never from memory or a summary.

## Git
- This is a greenfield repo and you are its only developer: commit straight to `main` and push after every commit. Keep `main` green; tests pass before every push. If a push fails, record why in STATUS.md and keep working.
- One commit per logical step, Conventional Commits messages, the evidence path in the body when a step produces a result. Tags: `calibration`, `kernel-verified`, `live-run`, `v0.1.0`.
- Before the first commit: `.gitignore` and a git hook that blocks secrets (the `sk-`, `hf_`, `ak-`, `as-`, `ghp_` and `github_pat_` prefixes).
- No weights, checkpoints, activations or raw generations in git. Commit the edit diffs and one record per sample (problem id, answer, grade, token count). Full texts stay on the volume, referenced by path and sha256. Pin dependencies with `uv.lock`.

## README
Lead with what OpenWSP is, in two plain sentences, then the proof. Copy the language of the blog post: its register, structure and vocabulary, plain declarative sentences, no adjectives about results. Credit the authors in the first paragraph and state that this is an independent replication with no affiliation to them. For Experiment 8: a TL;DR; a Setup list stating exactly what the editor had (kernel, forward passes, practice sets, budgets); tables with their numbers next to ours; the controls, labeled as ours; a short FAQ that answers the main questions raised about the original result with data from this repo (is the baseline broken, is it abliteration, is it a decoding fix, what tools did the editor have, what did it cost); a link to a readable transcript of the live run, with each turn's explanation, code and result; what each step cost in dollars and hours; and every difference from their setup (engine versions, GPU, sandbox, anything else). Every number links to the file it comes from. A run that falls short is reported as what it is, next to theirs. Then a quickstart: install, `openwsp replicate exp8`, pointing the harness at another model, and connecting the MCP server to Codex, Claude Code and Cursor. Then a "What we did not do" section, the results table others add to, and one short paragraph on dual use: the method that removes a loop can remove safety behavior too, this repo ships no such edits, and every edit is a sha256 addressed diff against a pinned base, so anyone can verify which weights changed. That restraint is what works on Hacker News. No pitch. If results are still running, say so at the top.

## Session environment note

The user says home is read-only. Install Modal in a venv under `/tmp`, preserve the container proxy, and trust the environment CA bundle.

## Clarification answers

Answered by the user on 2026-11-03 (verbatim):

1. Budget: $42 is Modal's usage limit for this billing cycle, credits included (about $22.50 of credits plus up to $20 of charges). Only the environment check has used anything, a few cents. Modal raises the limit after the first $10 charge clears; when it does, use the extra, up to $150 total for this task.
2. Memory: their finding are better evidence; I was just assuming from my armchair.
3. GPU: you decide on the evidence on the logs; the blog post; and then best practices for such uses.

Interpretation: match the authors' 96 GiB execution limit; retain their message verbatim. Choose GPU by evidence and measured cost/throughput. Current Modal ceiling remains $42 until an increase is observed, then spend at most $150 total on this task. Editor budget remains $100. No further questions are required; record later choices in QUESTIONS.md and DECISIONS.md.

## Initial source review (2026-11-03)

- Reviewed the blog and authors' release at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, including root/Experiment 8 READMEs, harness, final and original task messages, state, early Python/forward requests, edit manifest/application, evaluation/profile/grading/build/diff scripts, notices and practice metadata. Source review stayed within the requested 15 minutes.
- The authors explicitly did not release the Experiment 8 driver/kernel/forward server or IFEval checker imports. Reconstruct the public contract from prompts and logs; do not claim access to the original implementation. The generic harness sandbox is not the Experiment 8 kernel.
- The final task message promises 96 GB per execution, while this brief specifies a 16 GiB Modal Sandbox. Resolve in the initial question round.
- The final state lists delivered notices, but not their delivery turns. Recover timing from turn prompts rather than claiming it is in state.json.
- The first tool call is one Python cell followed by the first forward request. Test that first cell unmodified; do not replay later cells that depend on unreleased activations.
- Upstream code is MIT; logs/data/networks are CC BY 4.0; benchmark licenses remain separate. Default OpenWSP code license will be MIT with upstream notices preserved.
- Current official Modal list rates: H100 $0.001097/GPU-second, A100 80GB $0.000694/GPU-second; sandbox CPU $0.00003942/core-second and RAM $0.00000667/GiB-second. CPU/RAM are additional to GPU charges. Source: https://modal.com/pricing
- Current official Astra Standard prices per million tokens: input $10, cached input $1, cache writes $12.50, output $50. Above 272K input tokens, input/cache rates double and output rate is 1.5x. Max reasoning effort is documented. Source: https://developers.openai.com/api/docs/models/gpt-6-astra
- No paid jobs or editor requests have been launched; no code changes or commits yet.

## Initial question round

1. Does the $42 Modal ceiling mean total metered usage or out-of-pocket after credits, and is any allowance already consumed? Default: $42 total metered usage for this task, credits do not expand it, and any lower verified remaining account allowance also binds.
2. Which kernel memory limit wins: the brief's 16 GiB or the original prompt's 96 GB? Default: 16 GiB, correct only that resource statement in the editor prompt, and disclose the difference.
3. If smoke tests show another supported GPU costs less per completed pass, may GPU choice change even without a 10-minute H100 queue? Default: retain the specified H100 / queue-triggered A100 fallback and stop at the budget cap if necessary.

User replied; answers above supersede the initial defaults. Implementation is authorized.

## Later user steering (2026-11-03)

After the live budget-admission block, the user asked “what to top up? openai and or modal?” and “why is it taking too expensive to run Astra? maybe the approach is to be changed”. Audit the editor cost/output handling and recommend a concrete change before further live spend. No higher editor budget has been authorized; independent controls continue.

## Additional OpenAI allocation (2026-11-03)

User: “I added $300 more for OpenAI for a total of $321.70 but that doesnt mean you are free to spend it; budget half of it to complete the experiment 8 for yourself and the rest is plan b”.

Interpretation stated to the user: at most **$160.85 additional OpenAI spend** for completing Experiment 8, including diagnostics and recovery; **$160.85 remains protected for Plan B** and is not authorized for this attempt. Existing settled editor spend is $78.292765, so the cumulative ledger cap becomes **$239.142765**. This is an absolute cap, not a fresh $239.142765 allowance, and old charges/reservations remain. Apply it once through the serialized driver with a durable authorization record. Modal remains $42 until a higher current allowance is actually observed, at most $150 total. No blind live restart: validate the streaming guard and inexpensive isolated formatting checks first.

User also suggested Codex might subsidize usage instead of API pricing. Released logs establish Codex CLI use but not authentication/billing method. Their cumulative token snapshots are broadly compatible with the quoted API cost, with reconstruction caveats; our confirmed formatting loops explain most of our larger output bill regardless of billing route.

## Subscription billing preference (2026-11-03)

User: “i've ChatGPT Pro 100 and can upgrade that if needed but you're right to figure out the core of the problem. however, even for trying to fix that problem, it may make more sense to use the subsidized plan billing vs api billing for my budget and also to make it more attractive for people to fork and try”.

Prefer a supported Codex CLI route using the existing ChatGPT subscription if it can preserve the experiment contract. Check authentication mode, Astra availability, disabled native tools, exact instructions, reasoning continuity and durable resume before committing the live run to it. Hold the prepared paid API formatting probe during this investigation. The installed CLI reports ChatGPT-class authentication; this alone does not establish the user's plan, quota, model entitlement or billing account. Do not upgrade a subscription, silently fall back to API-key billing, discard the live history or claim a migrated conversation preserves reasoning without evidence. The previously authorized attempt allocation remains a ceiling, and the $160.85 Plan B reserve remains protected.

The initial environment credentials failed the authenticated quota endpoint with HTTP 401. The user enabled device sign-in, requested a retry, and then completed standard Codex device authentication for this workspace. Credentials are managed by Codex in a private ignored directory; no API-key fallback, subscription upgrade or credential transfer to Modal is authorized by that sign-in. Verify the resulting account and model access before inference.

User: “I updated the readme btw”. Their introduction is commit `0a41b05`; preserve it. Restrict current README edits to generated evidence and the controls' findings.

## Subscription upgrade and editor expansion (2026-11-03)

User: “upgraded to $200/m chatgpt pro 200 plan now. 100% usage left and i can also do a full reset as we exhaust that (expires in 2 days so we better use it than lose it). let's use the subscription instead of api pricing, afaik it is way cheaper. otoh do not lose your api implementation; it can be used when limits are reached or to use openrouter etc in the future. we will also need a fable implementation as well there; to compare it to astra on these capabilities. therefore the architecture of there should be accomodating to expansion instead of calcified.”

Use the upgraded subscription as the primary editor route. Retain the tested API implementation and its durable billing/recovery records, with the existing protected budget unchanged. Preserve provider-independent editor/bridge interfaces; separate transport, authentication, model profiles and pricing. Add a Fable implementation/profile over the Anthropic transport and an extension point for future routes such as OpenRouter; do not confuse native and router model identifiers or invent pricing equivalence. This steering does not authorize a paid Fable comparison now or an unrecorded mid-conversation model switch.

The live backend quota endpoint confirms `pro`, 99% of its weekly allowance remaining, and one reset credit; observed expiry is 2026-11-06 23:52:26 UTC. The older cached credential metadata still says `prolite`; backend quota is the newer evidence. No reset was redeemed. Record quota exhaustion and the available reset when action is needed. Source: `logs/codex-subscription-upgraded.json`. Do not purchase additional credits or upgrade beyond the user's completed change.

The user reconfirmed the Modal dashboard still shows $42: “still 42 but it will increase dont worry about it i will keep track of that; just report here periodically”. Retain the $42 ceiling until they confirm the increase. During the live run, report progress and compute spend in this chat about every fifteen minutes, and report budget/quota stops immediately.

## README style follow-up

The user requested a README overhaul using their commit0a41b05 as the style reference. Remove first-person 'I'; neutral or passive constructions are acceptable. Do not use em dashes anywhere, especially in titles. This supersedes the earlier instruction to preserve that introduction byte-for-byte while retaining its intent, attribution and scientific evidence.
