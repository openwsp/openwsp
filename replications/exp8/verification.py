"""Private production checks before exposing the forward service to an editor."""
from __future__ import annotations

import json
from pathlib import Path

from replications.exp8.jobs import ProtocolGPUJobs, gpu_instance
from openwsp.budget import compute_cost
from openwsp.core import WeightStore
from replications.exp8.runtime import atomic_write


def verify_forward(ledger, checkpoint):
    from transformers import AutoTokenizer
    from openwsp.backends.modal_app import Generator, TransformerReader, work_volume
    root = Path("/data/results/kernel")
    root.mkdir(parents=True, exist_ok=True)
    runner = ProtocolGPUJobs(ledger, "/data/jobs", commit=checkpoint)

    def run(key, spawn, fingerprint):
        runner.launch(key, spawn, fingerprint=fingerprint)
        return runner.result(key, timeout=3600)

    if not (root / "weight-swap.json").exists():
        tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
        probes = [tokenizer.encode(text, add_special_tokens=False) for text in
                  ("One plus one equals two.", "If x = 3, then x squared is 9.", "The sum of 5 and 7 is 12.", "We can solve this equation step by step.")]
        store = WeightStore("/model")
        store.apply_edit("/data/reference_edit")
        edits = store.validate_submission()
        edit_id = "authors-p50c-pinned"
        reference = run("verify/fresh-reference", lambda gpu, gate: gpu_instance(Generator, gpu, gate, checkpoint="/data/p50c", slot="verification-reference").reference.spawn(probes, edit_id),
                        {"edit": edit_id, "probes": probes})
        swapped = run("verify/native-swap", lambda gpu, gate: gpu_instance(Generator, gpu, gate, slot="verification-swap").audit_swap.spawn(edits, edit_id, probes, reference["reference"]),
                      {"edit": edit_id, "probes": probes})
        atomic_write(root / "weight-swap.json", {"status": "passed" if swapped["ok"] else "fallback",
                     "mode": "verified_native_swap" if swapped["ok"] else "fresh_engine_per_pass", **swapped})
        checkpoint()
    if not (root / "forward-verification.json").exists():
        request = {"texts": ["1+1=2", "2+3=5", "3+4=7", "4+5=9"],
                   "return": {"token_logprobs": True, "topk": 3, "hidden_states": [0, 4, 24], "mlp_act": [15, 19]}}
        result = run("verify/teacher-forced", lambda gpu, gate: gpu_instance(TransformerReader, gpu, gate).read.spawn(request, "/work/verification/teacher"), request)
        attention = {"texts": ["One plus one equals two.", "Two plus three equals five."],
                     "return": {"attn_layers": [3, 7, 11, 15, 19, 23]}}
        attn = run("verify/attention", lambda gpu, gate: gpu_instance(TransformerReader, gpu, gate, eager=True).read.spawn(attention, "/work/verification/attention"), attention)
        work_volume.reload()
        import numpy as np
        for directory in ("teacher", "attention"):
            for path in (Path("/work/verification") / directory).glob("*.npy"):
                values = np.load(path, mmap_mode="r")
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite production internal: {path.name}")
                if path.name.startswith("attn_"):
                    lengths = attn["lengths"]
                    for i, length in enumerate(lengths):
                        if not np.allclose(values[i, :, :length, :length].sum(-1), 1, atol=.015):
                            raise ValueError("Attention probabilities do not sum to one")
                del values
        # Public evidence names the actual files and records only their shapes.
        # Bare input_ids is reserved by the publisher for forbidden raw arrays.
        for metadata in (result, attn):
            metadata["files"] = {name + ".npy": shape for name, shape in metadata["files"].items()}
        atomic_write(root / "forward-verification.json", {"status": "passed", "teacher_forced": result, "attention": attn})
        checkpoint()
    return {"status": "passed", "weight_swap": json.loads((root / "weight-swap.json").read_text()),
            "internals": json.loads((root / "forward-verification.json").read_text())}
