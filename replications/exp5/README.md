# Experiment 5: Write from scratch

Three released MinAtar network-writing prompts are inventoried in [config.json](config.json) and [sources.json](sources.json). Every network begins with all 20,486 weights at zero. Each editor receives every prompt verbatim, once, with no tools and no feedback. The [shared CPU runner and scoring plan](../raw_weights/README.md) specify transport differences, hash checks, numeric edits and a CPU-only scorer using MinAtar 1.0.15. No PPO training is performed.

The parser reproduces all three published networks bit for bit. [Source validation](../raw_weights/source-validation.json) records this check. Independent [1,000-episode CPU calibration](../raw_weights/calibration/minatar/summary.json) of all three published networks is complete using the disclosed seed 20261103; original evaluation seeds were not released. All three new Astra replies have been scored; Fable remains [pending native credentials](../raw_weights/results/fable-blocker.json). [Reference](reference) contains only upstream outcomes. The MinAtar source embedded in the prompts retains its [GPL-3.0 notice](LICENSE-MinAtar).

Construct a game policy directly in weight space. See the [authors' release](https://github.com/RikiyaT/lm_understand-logs) for the original experiment.

## Per-editor results and PPO references

Experiment 5 scores are mean ± standard error over [1,000 episodes per policy](../../replications/exp5/reference/minatar_eval.json). Each editor supplies one reply per game; standard errors describe variability across game episodes.

| Game | Authors' PPO 100M: mean [range], seeds | Authors' PPO 500M: mean [range], seeds | Authors' Astra | Astra (subscription) | Fable (native Anthropic) |
| --- | --- | --- | --- | --- | --- |
| Breakout | [8.21 [5.61, 15.78], 10 seeds](../../replications/exp5/analysis/ppo-seed-summary.json) | [19.12 [12.19, 25.34], 3 seeds](../../replications/exp5/analysis/ppo-seed-summary.json) | [517.41 ± 7.42](../../replications/exp5/reference/minatar_eval.json) | [458.73 ± 4.99](../../replications/raw_weights/results/astra_subscription/exp5/breakout/result.json) | [Pending](../../replications/raw_weights/results/fable-blocker.json) |
| Seaquest | [4.82 [4.41, 5.62], 3 seeds](../../replications/exp5/analysis/ppo-seed-summary.json) | [33.59 [4.38, 88.88], 3 seeds](../../replications/exp5/analysis/ppo-seed-summary.json) | [53.11 ± 1.36](../../replications/exp5/reference/minatar_eval.json) | [2.41 ± 0.11](../../replications/raw_weights/results/astra_subscription/exp5/seaquest/result.json) | [Pending](../../replications/raw_weights/results/fable-blocker.json) |
| Asterix | [18.19 [13.21, 25.82], 10 seeds](../../replications/exp5/analysis/ppo-seed-summary.json) | [48.93 [44.71, 55.20], 3 seeds](../../replications/exp5/analysis/ppo-seed-summary.json) | [53.18 ± 1.02](../../replications/exp5/reference/minatar_eval.json) | [39.71 ± 0.73](../../replications/raw_weights/results/astra_subscription/exp5/asterix/result.json) | [Pending](../../replications/raw_weights/results/fable-blocker.json) |

PPO entries are reference-only, equally weighted means over every released training seed; the ranges and complete seed lists are [retained in the source-derived summary](../../replications/exp5/analysis/ppo-seed-summary.json). The authors' post reports best-seed PPO values. No PPO training or model selection was repeated.
