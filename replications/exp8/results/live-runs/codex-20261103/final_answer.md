Retained one edit: **`model.language_model.layers.23.mlp.down_proj.weight`**

\[
W' = W-4(Wv)v^\top
\]

Here \(v\) is a unit direction derived from late-versus-early repeated-check activations, targeting repetitive thinking rather than globally scaling outputs.

**Path:** Initial diagnosis found 175/192 generations hit the token limit; attention-head and single-neuron candidates failed validation, while the layer-23 direction passed strength tests and outperformed matched random-direction controls. Stronger and opposite-sign edits were rejected.

**Independent confirmation:** The frozen edit was tested on 2,048 held-out practice problems, with 3,584 generations per model and equal weighting per problem.

| Practice source | Original | Edited |
|---|---:|---:|
| GSM8K | 15.09% | **25.34%** |
| MATH | 9.28% | **13.43%** |
| Combined | 12.18% | **19.38%** |

The paired gain was **7.20 percentage points**, SE **0.75**, with an approximate 95% interval of **[5.72, 8.68]** points; cutoffs fell from **88.94% to 75.98%**.

This supports expecting improvement through reduced repetitive noncompletion, not a general mathematical-capability repair: many failures remain, legitimate iterative reasoning can also be suppressed, and AIME accuracy did not improve. No forbidden tensors, configuration, or decoding settings changed, and no training was used.

Details are saved in `/work/final_report.md` and `/work/final_results.json`; the selected weights remain in `/work/edited`.