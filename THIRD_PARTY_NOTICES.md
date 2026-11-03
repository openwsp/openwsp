# Third-party notices

OpenWSP is an independent replication of Experiment 8 in [Sparks of Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), with no affiliation to its authors. The source release is [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), pinned to commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

OpenWSP's [MIT license](LICENSE) covers its own code. It does not relicense third-party material.

| Material | License and attribution |
| --- | --- |
| Authors' Python and shell scripts | MIT, copyright 2026 the authors of “LLMs can understand and write neural network weights”; full notice below. |
| Authors' task messages, logs, notices, edit artifacts and other original data | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode), attributed to the authors of the linked post and release. Copied material retains its provenance; OpenWSP-generated results are identified separately. |
| Benchmark and practice problem text | The original dataset's terms continue to apply. The authors' license does not replace the terms for MATH, GSM8K, IFEval, or other quoted datasets. Consult each dataset's pinned source before redistributing it. |
| Qwen model weights and tokenizer files | The terms distributed with [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17). Full weights and checkpoints are downloaded to the compute volume, not committed to this repository. |
| MinAtar source in the authors' Experiment 5 prompts | GPL-3.0 in the upstream release. OpenWSP's Experiment 5 folder is a layout placeholder and contains none of that source. |
| Official IFEval instruction checkers | Apache-2.0, from [google-research/google-research](https://github.com/google-research/google-research/tree/83f1e7b59e762648c953de751d31f75720683622/instruction_following_eval), pinned to commit `83f1e7b59e762648c953de751d31f75720683622`. The vendored files retain their copyright headers and [license](replications/exp8/vendor/instruction_following_eval/LICENSE); file hashes are recorded in [sources.json](replications/exp8/vendor/instruction_following_eval/sources.json). |
| Codex Astra model metadata | Apache-2.0, OpenAI's [Codex CLI](https://github.com/openai/codex/tree/83086188ae0614bd882818ff1aa23f3194962012), release `rust-v0.163.0`. The [catalog provenance](src/openwsp/editors/data/codex-models-0.163.0.provenance.json) records the exported source hashes and three modified tool-exposure fields. The [upstream license](src/openwsp/editors/data/CODEX_LICENSE) is included. |

The pinned upstream [license](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE) is the authority for the source release. Changes to imported code and differences from the experiment are described in the replication configuration and decision log.

## Authors' script license

MIT License

Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
