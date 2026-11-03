# Experiment 5: Write from scratch

Three released MinAtar network-writing prompts are inventoried in [config.json](config.json) and [sources.json](sources.json). Every network begins with all 20,486 weights at zero. Each editor receives every prompt verbatim, once, with no tools and no feedback. The [shared CPU runner and scoring plan](../raw_weights/README.md) specify transport differences, hash checks, numeric edits and a CPU-only scorer using MinAtar 1.0.15. No PPO training is performed.

The parser reproduces all three published networks bit for bit. [Source validation](../raw_weights/source-validation.json) records this check. Independent [1,000-episode CPU calibration](../raw_weights/calibration/minatar/summary.json) of all three published networks is complete using the disclosed seed 20261103; original evaluation seeds were not released. New Astra and Fable replies/results remain pending. [Reference](reference) contains only upstream outcomes. The MinAtar source embedded in the prompts retains its [GPL-3.0 notice](LICENSE-MinAtar).

Construct a game policy directly in weight space. See the [authors' release](https://github.com/RikiyaT/lm_understand-logs) for the original experiment.
