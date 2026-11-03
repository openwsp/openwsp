# Reconstructed Kinetix scorer

The released `eval/ppo_training_result.json` pins the official [Kinetix source](https://github.com/FLAIROx/Kinetix/tree/80ee9c292837c825cafaf9889323fdcc212efd1b) to `80ee9c292837c825cafaf9889323fdcc212efd1b` and JAX to `0.4.38`. That commit declares `jax2d==1.0.1`. The harness implementation and its evaluation RNG seeds were not released, so this is an explicit reconstruction, not the authors' scoring script.

The scorer loads `kinetix/levels/l/h2_car_ramp.json` and uses the official multi-discrete actions, symbolic flat observation and physics engine with auto-reset disabled. The saved level's dense reward scale is 0.1; the experiment prompt specifies 0.2, which the scorer sets explicitly. Other checked settings are 12 polygon slots, four circle slots, six joints, two thrusters, two physics substeps per action, 1/60-second substeps and a 256-action episode limit. The resulting observation has 679 float32 entries. The actor has the prompt's ReLU embedding, five tanh layers and 16 logits; all six action entries are sampled independently, including the inactive bindings.

The declared new evaluation seed is 20261103, with each episode key `jax.random.fold_in(jax.random.PRNGKey(seed), episode_index)`. Each step splits independent keys for the environment and six categorical samples. Evaluation never changes weights or asks the editor for feedback. Per-episode outcomes, returns and lengths are retained. The published repaired archive is float64; the shared store and scorer use float32, matching the explicitly float32 observations and released training computation. Calibration is against the rounded shown base and released repaired actor, not the full-precision training checkpoint.

Use a separate CPU environment to avoid changing the main project's dependencies:

```sh
UV_CACHE_DIR=/tmp/openwsp-uv-cache uv venv .openwsp/kinetix-venv
UV_CACHE_DIR=/tmp/openwsp-uv-cache uv pip install \
  --python .openwsp/kinetix-venv/bin/python 'jax[cpu]==0.4.38' 'flax==0.10.4' \
  .openwsp/kinetix-source
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/openwsp-matplotlib \
  XDG_CACHE_HOME=/tmp/openwsp-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .openwsp/kinetix-venv/bin/python -m replications.raw_weights.kinetix_score \
  --network /absolute/actor.npz --source-root .openwsp/kinetix-source \
  --episodes 1024 --output /absolute/result.json
```

Clone the linked Kinetix repository into `.openwsp/kinetix-source` and check out the exact commit before installation. The scorer rejects a different commit, modified relevant source files, installed source that differs from the checkout, incompatible JAX/JAX2D versions and GPU devices. Kinetix is MIT licensed; its dependency JAX2D retains its package license. This code imports installed packages; it does not vendor or rewrite their physics implementation.

The files in `calibration/` are evaluations of the authors' released networks only. They must not be presented as new Astra or Fable results. Original RNG seeds were not published, so numerical mean-return equality is not expected even when success rates reproduce.

[The completed calibration](calibration/summary.json) reproduces 0/256 successes for the shown base (all timeouts) and 1,024/1,024 for the published repair (all 73 steps). Mean returns are 0.151905944 and 1.373848140, respectively, versus the authors' 0.151900858 and 1.373849869. The small differences are retained in the summary; they are not rounded away to imply exact trajectory equality. Full per-episode records are linked from that file.
