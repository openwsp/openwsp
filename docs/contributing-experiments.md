# Contributing experiments

Each contribution starts with a declared question and produces evidence in the [results schema](results-schema.md). Keep protocol changes in replication folders and build on the shared weight store, edit format and editor interfaces. A CPU implementation check is not a model result. Missing measurements stay pending; failed and interrupted attempts remain visible.

## Add an evaluation cell

Record the hypothesis, comparison and settings in [QUESTIONS.md](../QUESTIONS.md) before reading results. Copy a JSON definition from [eval_grid_configs](../replications/exp8/eval_grid_configs), choose a new stage name and register it in `STAGES` in [eval_grid.py](../replications/exp8/eval_grid.py). The order in `STAGES` also controls admission, so a new cell must preserve the declared spending order.

Specify the model condition, benchmark, samples per problem, thinking flag, complete sampling settings, token cap, engine context length and sequence limit, decoding policy and seed scheme. The current loader supports MATH-500 with one sample per problem and base, p50c or live conditions. A new benchmark or policy also requires an explicit loader and execution change. Forced closure and thinking off are distinct conditions. Extend the CPU tests for changed settings, natural closure, total-token limits and resume/config identity before scheduling a GPU run.

From the repository root:

```sh
uv run python -m replications.exp8.eval_grid list
uv run python -m replications.exp8.eval_grid validate --output /tmp/openwsp-grid-validation.json
uv run pytest -q tests/test_exp8_eval_grid.py tests/test_thinking_close.py
```

For a funded run whose prerequisites are complete, the existing launch interface is:

```sh
uv run openwsp replicate exp8 --stage grid_p3_base --foreground
```

The stage checks the saved active attempt, its actual 50-pass submission and final evaluations, control 4 and earlier grid cells. It does not bypass those requirements. Results go under `replications/exp8/results/<stage>/math500/` with the cell configuration hash and runtime provenance.

## Add a mechanical repair variant

Declare the two contrastive text sets, their source and split, target tensors, direction construction, column-selection rule and coefficient before scoring. The repetition repair uses `W' = W - alpha * outer(u, mask * (u.T @ W))` with a unit contrast direction. Record whether columns are all selected or ranked by projection onto that direction times the mean activation difference. A borrowed column set must be identified explicitly.

The specified variants are all columns at coefficient 1, all columns at coefficient 6, and 64 mechanically selected columns at coefficient 6. Keep each variant and dose separate. The repair builder is under implementation; no repair command is documented as available yet. Its acceptance checks must cover a planted contrast, deterministic column selection, coefficient-zero identity, untouched columns, finite outputs, protected tensors and versioned edit round trips. Export the [v1 edit](edit-format.md) with contrast-source hashes, selected columns and coefficient provenance. Scoring requires separate evaluation evidence.

The implemented inspection command can audit a built edit on CPU:

```sh
uv run openwsp --model /path/model.safetensors --work-dir /tmp/openwsp-inspect inspect --path /path/edit --output /tmp/openwsp-inspect.json
uv run pytest -q tests/test_inspection.py
```

This repository ships repetition repair only. Refusal-removal texts, edits and demonstrations are outside its scope.

## Add a replication folder

Create `replications/<experiment>/` with its own configuration, verbatim prompts, pinned source manifest, reference records, evaluator, tests and results. Record source commits, file hashes and license notices; separate reference numbers from new measurements. The [raw-weight inventory](../replications/raw_weights/inventory.json) and experiment folders 1, 2, 4 and 5 provide source/configuration examples; their runner is under implementation. The pinned release provides rules and results, but no scoring implementation for these experiments, so reconstructed evaluators must disclose missing upstream RNG and version pins. Experiments 3, 6 and 7 remain layout only.

Preserve the experiment's editor contract. For raw-weight experiments, each item uses a fresh conversation, one reply and no tools or training. Parse the permitted answer or weight format without executing arbitrary editor text. Keep evaluator keys, controls and outside findings out of editor inputs. Use distinct run identities for different editors or protocols; never reset an existing attempt's counters, overwrite its submission or merge its history with another run.

## Publish evidence

Store compact sample records, summaries, configurations, edit identities and all attempt statuses. Record model/data revisions, prompt hashes, seeds, sampling, token limits, engine versions, hardware and cost basis. Compute `finished_n` and `accuracy_finished` from recorded termination reasons, alongside overall accuracy and `cut_share`; retain capped samples in the overall denominator. An empty finished set has undefined accuracy, not zero. Missing termination evidence cannot establish finished-only accuracy. Use experiment-specific metrics where accuracy does not apply.

Keep checkpoints, activations, full generations and credentials out of Git. Reference private artifacts by path and SHA-256. For Experiment 8, regenerate the tables from files, then review their source links:

```sh
uv run openwsp report exp8
uv run pytest -q
uv run python scripts/check_secrets.py --all
```

Submit code, CPU tests, provenance and generated evidence together. A blocked run includes its blocker and resume procedure; a successful check never substitutes for an unrun experiment.
