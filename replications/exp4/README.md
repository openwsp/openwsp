# Experiment 4: Fix the car

One released car-repair prompt is inventoried in [config.json](config.json) and [sources.json](sources.json). It contains the complete rounded actor and task description. Each editor receives it verbatim, once, with no tools and no feedback. The [shared CPU runner and scoring plan](../raw_weights/README.md) specify the one-reply contract and reconstruct the numeric edit language without executing generated code.

All 51 statements in the published answer reproduce the published repaired actor after conversion to the shared float32 store. [Source validation](../raw_weights/source-validation.json) records this parser check. The Kinetix rollout scorer is being reconstructed from the released training pin; new success rates remain pending until source-network calibration completes. [Reference](reference) contains the authors' 0/256 base and 1024/1024 repair results, not new outcomes.

Edit a failing policy. See the [authors' release](https://github.com/RikiyaT/lm_understand-logs) for the original experiment.
