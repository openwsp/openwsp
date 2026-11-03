# Experiment 4: Fix the car

One released car-repair prompt is inventoried in [config.json](config.json) and [sources.json](sources.json). It contains the complete rounded actor and task description. Each editor receives it verbatim, once, with no tools and no feedback. The [shared CPU runner and scoring plan](../raw_weights/README.md) specify the one-reply contract and reconstruct the numeric edit language without executing generated code.

All 51 statements in the published answer reproduce the published repaired actor after conversion to the shared float32 store. [Source validation](../raw_weights/source-validation.json) records this parser check. The reconstructed Kinetix CPU scorer [reproduces 0/256 base successes and 1024/1024 published-repair successes](calibration/summary.json); every repaired episode takes 73 steps. [Scorer documentation](SCORER.md) discloses the source pins, evaluation seed and precision differences. These are source-network calibrations; new editor results are recorded separately in [the outcome inventory](../raw_weights/results/summary.json). [Reference](reference) contains only the authors' results.

The new Astra subscription attempt did not reproduce the repair: [its sole reply](../raw_weights/results/astra_subscription/exp4/oneshot_repair/answer.md) supplied 31 valid statements changing 2,918 weights, and [the 1,024-episode evaluation](../raw_weights/results/astra_subscription/exp4/oneshot_repair/kinetix-evaluation.json) had no successes. Every episode hit red at step 32. The failed attempt is retained without feedback, refinement or a replacement reply. Fable remains pending.

Edit a failing policy. See the [authors' release](https://github.com/RikiyaT/lm_understand-logs) for the original experiment.
