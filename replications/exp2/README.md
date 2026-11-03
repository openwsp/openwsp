# Experiment 2: Diffusion

Three released diffusion-weight prompts are inventoried in [config.json](config.json) and [sources.json](sources.json). Each editor receives every prompt verbatim, once, with no tools and no feedback. The [shared CPU runner and scoring plan](../raw_weights/README.md) specify transport differences, hash checks, one-reply recovery and result locations.

The reconstructed `Images`-section scorer reproduces the authors' three shape-family labels. Correct shape-family labels do not imply a correct filled-versus-hollow description. [Source validation](../raw_weights/source-validation.json) is calibration of published answers. The new Astra subscription run scored 3/3 shape families correctly; saved answers, hashes and usage appear in [the outcome inventory](../raw_weights/results/summary.json). Fable remains pending; [reference](reference) contains upstream result evidence.

Read what a diffusion model generates. See the [authors' release](https://github.com/RikiyaT/lm_understand-logs) for the original experiment.
