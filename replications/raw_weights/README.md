# One-reply CPU replications

Experiments 1, 2, 4 and 5 contain 11 released prompts: four game labels, three image-family labels, one car repair, and three MinAtar policies. [inventory.json](inventory.json) records the exact prompt/instruction hashes, byte counts, source paths and authors' effort settings at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` of [the authors' release](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f). Every released item is planned once for Astra through ChatGPT subscription and once for native Anthropic Fable 5.1. No new model results are implied by the source-validation file.

The complete source checkout stays in the ignored `.openwsp/upstream` cache. No source prompt is rewritten: its UTF-8 bytes and the neutral instruction file must match the manifest before execution. Each item gets a fresh persistent editor state and one scientific reply, zero tools, and no feedback. A retry after interruption may retrieve a durable completed reply; an uncertain dispatch never creates a replacement inference. The existing Codex and Anthropic adapters enforce this behavior.

Astra uses the authors' `high` effort for experiments 1, 2 and 4, `xhigh` for experiment 5, and detailed reasoning summaries only where the released command requested them (1, 2 and 5). Our shared profile explicitly selects `low` verbosity; the authors' recorded commands did not pin verbosity. Our authentication is ChatGPT subscription; the authors released their Codex command and model identity, but did not establish the same subscription entitlement. These transport/settings differences are disclosed, not presented as exact execution equivalence. The installed Codex version, catalog hash and tool-free configuration remain pinned by the shared profile. Subscription tokens are reported as usage, never assigned fictitious API dollars.

Fable uses `claude-fable-5-1`, adaptive thinking and the corresponding native effort names. These names do not imply equal reasoning compute across providers. Official [model specifications](https://platform.claude.com/docs/en/models/fable-5-1/overview) list 1M context, 128K maximum output and standard $10/M input, $50/M output; [pricing](https://platform.claude.com/docs/en/about-claude/pricing) applies standard rates across the full context. [Effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort) confirms native `high` and `xhigh`. These sources were checked on 2026-11-03. Account entitlement and token counts still require verification before dispatch.

All 11 Fable items share one existing `BudgetLedger` configured with `editor_limit=20`. The five explicit rates are input 10, output 50, cache read 0.25, 5-minute cache creation 12.50 and 1-hour creation 20 dollars per million. The adapter reserves the conservative maximum input rate and output ceiling before each request, and settles actual disjoint usage. No prompt caching, batch discount, US-only routing, API fallback or best-of selection is requested. Stop when the next conservative bound cannot fit; preserve the unattempted item as pending. The fixed output ceiling is 8,192 for experiments 1/2 and 32,768 for 4/5, including thinking. Before Fable execution, save native token-count evidence and a conservative input ceiling for each exact prompt in the limits file. Source authors' token counts are not Fable counts. A length-limited reply is retained as incomplete, never continued or replaced.

From the OpenWSP repository root, prepare the pinned source cache once, using sparse checkout for the four raw-experiment folders. These commands do not call an editor or run a scorer:

```sh
mkdir -p .openwsp
git clone --filter=blob:none --no-checkout \
  https://github.com/RikiyaT/lm_understand-logs.git .openwsp/upstream
git -C .openwsp/upstream sparse-checkout init --cone
git -C .openwsp/upstream sparse-checkout set \
  exp1_which_game exp2_diffusion exp4_fix_car exp5_write_from_scratch
git -C .openwsp/upstream checkout --detach 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
git -C .openwsp/upstream rev-parse HEAD
```

If `.openwsp/upstream` already exists, reuse it when `rev-parse HEAD` reports that exact revision and the inventory hash check below passes. Preserve existing changes and sparse-checkout selections. For a different or modified checkout, use a new cache directory and pass its path with `--source-root` to the runner, token counter and source validator. The runner verifies existing source files; it never downloads or rewrites them.

Run inventory and hash verification without inference:

```sh
.venv/bin/python -m replications.raw_weights.runner \
  --editor astra_subscription --state-root .openwsp/raw-weights \
  --output-root replications/raw_weights/results
```

Actual execution is explicit; the existing subscription login stays local:

```sh
.venv/bin/python -m replications.raw_weights.runner \
  --editor astra_subscription --codex-home .openwsp/codex-subscription \
  --state-root .openwsp/raw-weights --output-root replications/raw_weights/results \
  --execute
```

`--experiment exp1 --item breakout` selects one released item without modifying its prompt. Reuse the same state/output roots for every item and recovery. Fable replaces `--editor` with `fable_native_anthropic` and requires `--fable-limits /absolute/verified-limits.json`; its SDK reads `ANTHROPIC_API_KEY` normally. No credential appears in config or results. Each limits entry is keyed by `exp1/breakout` (and similarly for the other items), with `prompt_sha256`, positive `max_input_tokens`, the fixed `max_output_tokens`, and `count_evidence` containing the native counting provenance. `--rollout` enables the CPU MinAtar scorer. For deferred scoring, replace `--execute` with `--score-only --rollout`: this path requires saved hash-verified answers and never constructs an editor or reads credentials. It preserves the original answer and receipt.

Astra completed [all eleven replies and scores](results/summary.json). Fable remains [unstarted pending native credentials](results/fable-blocker.json); its authorized $20 cap has not been used.

The [completed-run verification](results/astra-verification.json) checks eleven distinct conversations, one completed turn per item, no native tool calls, and source/answer hashes. CPU scoring uses float32 parsed tensors. The generic version-1 edit exports round to bfloat16 and are not the exact scored networks. The verification record preserves each scored float32 tensor's shape and hash; the immutable answer, shown base and pinned parser reconstruct those tensors. Full weights stay outside git.

Regenerate the [new-outcome inventory](results/summary.json), [comparison](results/comparison.md) and marked main README tables from saved files with `.venv/bin/python -m replications.raw_weights.report --output-root replications/raw_weights/results --update-readme`. It distinguishes missing replies, saved replies awaiting rollout scoring, and completed scores; it never substitutes the authors' reference outcomes.

## Reconstructed scoring

The authors released rules, replies, networks and result JSON, but no scorer implementation in these four folders. Our scorer is therefore a disclosed reconstruction. [source-validation.json](source-validation.json) validates the seven released label answers and all four edited networks. It is a parser calibration, not independent evidence of new editor performance.

Experiments 1 and 2 score only the requested `Game` or `Images` answer section. Missing or ambiguous labels fail. Freeway accepts Frogger. Shape labels follow the released circle/disc/round/oval, square/rectangle/box, and triangle/pyramid/wedge families. Shape-family correctness does not establish correct filled-versus-hollow image descriptions.

Experiments 4 and 5 parse only the final complete `edits` fence, with numeric literals, scalar fills, exact-shape lists, indexed slices, `=`, `+=`, `-=`, `*=`, comments and the car's `ZERO` command. Invalid statements are skipped in order. The parser never executes generated Python. Weight changes use the shared `WeightStore` and version-1 edit export. MinAtar starts at all zero; car starts from the rounded integers and scales in the prompt. The released car archive is float64; shared storage converts to float32. Our reconstructed car matches every element of the released archive after that conversion, and all three MinAtar archives match bit for bit.

MinAtar is pinned to `minatar==1.0.15`, with prompt-specific channel mappings, the 600→32→32→6 tanh actor, full six-action softmax sampling, sticky actions 0.1, difficulty ramping, 1,000 episodes and a 10,000-step limit. Our predeclared evaluation seed is 20261103: independent environment seeds `seed + episode`, independent policy RNG `SeedSequence([seed, episode, 1])`, and a fresh environment per episode. Original evaluation seeds were not released, so exact numerical equality is not expected. Per-episode scores and lengths are retained. No PPO training is performed. Install the CPU game dependency with `UV_CACHE_DIR=/tmp/openwsp-uv-cache uv pip install --python .venv/bin/python --no-deps minatar==1.0.15`; display packages are unnecessary.

Full 1,000-episode calibration of all three published MinAtar networks is complete. [The file-derived summary](calibration/minatar/summary.json) links every episode record and the source-network hashes. Each mean is within two combined standard errors of its authors' reference mean; this validates the reconstructed scorer at the released precision without claiming identical RNG streams or new editor performance.

The car's training record pins Kinetix commit `80ee9c292837c825cafaf9889323fdcc212efd1b` and JAX `0.4.38`; its scoring implementation and original evaluation RNG seeds were not released. The reconstructed CPU scorer uses those pins and now reproduces the shown base's 0/256 successes and the published repair's 1,024/1,024 successes, with every repaired episode ending at 73 steps. [Calibration records](../exp4/calibration/summary.json) retain all episode outcomes; [setup and differences](../exp4/SCORER.md) disclose seed and precision choices. Deferred `--score-only --rollout` uses the isolated JAX environment and never invokes the editor.

Once native credentials are configured, `.venv/bin/python -m replications.raw_weights.fable_limits --source-root .openwsp/upstream --output .openwsp/raw-weights/fable-limits.json --count-native-tokens` verifies native model metadata and counts the exact messages. It does not create a Message or generate a reply. Each input ceiling is `ceil(native_count * 1.05) + 1024`, with the count receipt and model limits retained. The counting endpoint returns an estimate; the adapter still checks actual usage and halts on any ceiling overrun. Generation uses a 3,600-second client timeout, zero provider retries and the fixed output ceilings above.

Run the source calibration without a model:

```sh
.venv/bin/python -m replications.raw_weights.validate_sources \
  --source-root .openwsp/upstream --output replications/raw_weights/source-validation.json
.venv/bin/pytest -q tests/test_raw_weights.py tests/test_editor_profiles.py
```

Upstream logs, data and networks are CC BY 4.0; [LICENSE-upstream](LICENSE-upstream) preserves attribution and the upstream MIT code distinction. MinAtar source embedded verbatim in the experiment-5 prompts is GPL-3.0; [the full notice](../exp5/LICENSE-MinAtar) is retained. Prompts and full network archives remain in the source cache; checked-in manifests link them by immutable source revision and hash.
