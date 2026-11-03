# Experiment 1: Which game?

Four released game-identification prompts are inventoried in [config.json](config.json) and [sources.json](sources.json). Each editor receives every prompt verbatim, once, with no tools and no feedback. The [shared CPU runner and scoring plan](../raw_weights/README.md) specify transport differences, hash checks, one-reply recovery and result locations.

The reconstructed `Game`-section scorer reproduces all four authors' labels; [source validation](../raw_weights/source-validation.json) is calibration of published answers. The new Astra subscription run scored 4/4 correct; its saved answers, hashes and usage appear in [the outcome inventory](../raw_weights/results/summary.json). Fable remains pending. The small JSON files under [reference](reference) are upstream evidence, never our scores.

Identify which game a network was trained on. See the [authors' release](https://github.com/RikiyaT/lm_understand-logs) for the original experiment.
