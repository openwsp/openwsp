# Experiment 8 KL on a local CPU

The registered damage measurement can run on an existing CPU workspace without
starting a Modal job. The normal Modal stage remains available. This command is
an alternative execution location for the same four comparisons, not a shorter
measurement or a way around the evaluation queue.

The command first checks the actual completed 50-pass submission, all required
final evaluation samples, control 4, and all 11 registered grid cells. Missing or
inconsistent evidence stops execution before tokenization, reference preparation
or model loading. There is no override for this check.

Use the replication environment with the pinned `transformers==5.17.0`. Supply
an already-materialized private local mirror of the relevant Modal data and
practice files. No download, checkpoint reconstruction, provider authentication
or synchronization is performed by the command. Both mirror roots must be
absolute existing directories owned by the caller with mode `0700`; symlinks
are rejected on resolved input paths. Do not use a Hugging Face cache snapshot
containing links as a checkpoint mirror.

The data mirror needs these original files and directories:

- The selected run's state and exported edit bundle, with its saved submission
  and original `/data/...` checkpoint path unchanged.
- `original`, `p50c`, `reference_edit`, both control submissions and edit
  bundles, and every checkpoint selected by those submissions.
- The benchmark files, complete final evaluation receipts and sample files,
  control 4, and all 11 grid cells required by `require_damage_priority`.
- The four source-frozen `damage/<run-relative-path>` manifests and token records.
  These are required even for the first CPU run. Preserve any completed
  `results/damage/<run-relative-path>` summaries when resuming; conflicting
  evidence is rejected rather than overwritten.

The practice mirror needs the pinned `gsm8k_train.jsonl` and `math_train.jsonl`
files. Their full file hashes are checked against the registered sources.
The mirror roots replace `/data` and `/practice` for local file access only.
Saved submissions and manifests retain their original source paths, checkpoint
and edit hashes, tokenization identity and faithful submission binding. Moving
the private mirror therefore does not change the scientific manifest.

The four manifest hashes must be independently recorded from the authoritative
source `/data` manifests. They mean `openwsp.backends.forward_bank.json_hash`
of the parsed manifest object, matching the existing measurement identity;
they are not the output of `sha256sum` over the formatted JSON file. Do not
derive these expected hashes from the local mirror being checked. All four
are checked before tokenizer loading or preparation, then local checkpoint,
edit and reference bytes must reproduce those anchored manifests exactly.
Local hashing by itself cannot establish equivalence to the source checkpoint.

Source preparation is an explicit normal-pipeline stage, separate from the
local CPU command. After the full required queue has completed, the existing
coordinator and CPU driver reservation admit this command:

```sh
openwsp replicate exp8 --stage damage_prepare \
  --config /absolute/private/codex-exp8.json --foreground
```

Use the same editor and run identity as the completed faithful submission.
The stage runs the full priority gate before loading a local tokenizer,
freezes the existing four source manifests, and commits them. It dispatches
no GPU jobs and loads no model. It can incur the existing Modal CPU-driver
charges. It is never included implicitly in `all`. The normal Modal `damage`
stage reuses the same preparation helper before its existing measurement jobs.

The returned `stages.damage_prepare.manifests` entries contain each source path
and canonical hash. Record these directly from that authoritative source
response, then transfer the required files into the private mirror using the
existing operator workflow. The local CPU command does not launch preparation
or transfer files. No authoritative production source manifests or source
hashes have been verified for this command yet. Production preparation requires
the completed queue and normal budget admission. Source preparation and file
transfers can have costs separate from the local measurement; end-to-end savings
have not been measured.

From the repository root, after the queue and source preparation have completed,
substitute the independently recorded canonical hashes:

```sh
.venv/bin/python -m replications.exp8.damage_cpu \
  --data-mirror /absolute/private/exp8-data \
  --practice-mirror /absolute/private/exp8-practice \
  --editor codex --run-id codex-20261103 --threads 2 \
  --manifest-sha256 'p50c=<source canonical hash>' \
  --manifest-sha256 'mechanical=<source canonical hash>' \
  --manifest-sha256 'random=<source canonical hash>' \
  --manifest-sha256 'live=<source canonical hash>'
```

All 512 registered references are used in each comparison: the first 256 pinned
GSM8K training references, then the first 256 pinned MATH training references.
Full prompts and unchanged solutions are tokenized once and never truncated.
The comparison order is `p50c`, `mechanical`, `random`, `live`. The existing
four-reference base-to-base smoke runs before each edited model is loaded and
does not contribute to its mean.

The loader uses BF16, SDPA and CPU explicitly. The existing kernel processes
one full reference at a time, projects 32 scored positions at a time, and
accumulates token-weighted KL(base || edited) in float64, in nats. One base and
one edited model are resident together. There is no fallback to CUDA or FP32.
CPU and GPU floating-point execution need not produce bit-identical results;
the device, library versions, thread count and implementation hashes are saved
with the result. Bounded hardware metadata also records the available processor
model, visible logical CPU count, affinity count and cgroup v2 quota when these
are readable. Unavailable values remain null.

Completed summaries are written to the corresponding private
`results/damage/<run-relative-path>/<edit>/summary.json`. Only complete results
with the exact current manifest and CPU execution provenance are reused. A
saved GPU result is not relabeled as a CPU result. A process lock prevents two
instances of this CPU command from writing the same mirror concurrently.
An interrupted comparison restarts that comparison; earlier complete matching
summaries remain reusable. There is no partial-reference cache.

Each summary records elapsed local wall time, including input verification,
identity smoke and model loading. The first computed comparison also includes
the shared base model load; initial reference preparation and preflight are
outside these per-comparison timings. `incremental_modal_cost_usd` and the
summary's provider-cost field `cost_usd` are zero because no Modal job runs.
`local_cpu_cost_usd` is null: the existing workspace's CPU resources are unpriced,
not free. No provider budget ledger is changed.

This entry point has only been verified with tiny CPU fixtures. No production
KL runtime or memory measurement is implied. The production run may take many
hours and needs space for the materialized checkpoints. The completion,
source-verification and budget gates still apply.
