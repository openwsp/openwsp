# Later questions and chosen defaults

No further user input is required. Record uncertainties and the chosen defaults here.

- Control 1: the pinned model card distinguishes its benchmark settings (0.7/0.8/1.5) from recommended non-thinking text settings (temperature1.0/top_p1.0/presence2.0/top_k20/min_p0/repetition1). Use the dedicated text recommendation, thinking disabled, keep32768 token cap for comparison. Source: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md . No result-dependent tuning.

- Control 2 needs a definition of looping and token aggregation. Default chosen before scores: length-cut answers with at least 50% duplicate 8-grams in the final 1,024-word tail, contrasted with all stop-finished answers; equal weight per answer, mean MLP output over its first 8,192 answer tokens. Normalize each layer's mean difference, remove alpha=6 times its projection from exactly the 64 columns changed by p50c. This uses calibration MATH-500 answers as requested and is labeled a diagnostic, not a held-out estimate.
- Controls run alongside the live run only when the verified budget can reserve headroom for the live run and required evaluation. Otherwise the user's spending priority takes precedence and controls wait. Missing controls are never reported as evaluated.
- When available Modal headroom falls below $20, paired passes may reuse previously recorded base generations for exactly matching problem/prompt, sampling, pinned model/engine and fixed seed. Its use is recorded per forward request and disclosed in the final report.
