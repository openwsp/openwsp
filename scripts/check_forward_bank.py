"""Verify the forward bank on a synthetic CPU fixture without provider calls."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from openwsp.backends.forward_bank import ForwardResultBank, GenerationBank, json_hash
from openwsp.budget import atomic_json


def check(output):
    with tempfile.TemporaryDirectory(prefix="openwsp-bank-check-") as temporary:
        root = Path(temporary)
        bank = GenerationBank(root / "base")
        model_hash = json_hash({"tiny_linear_weights": [[1.0, 2.0], [0.0, 1.0]]})
        key = bank.identity("tiny-1", "synthetic prompt", {"n": 1, "temperature": 0}, model_hash, 7)
        generated = {"id": "tiny-1", "samples": [{"text": "3", "token_count": 1, "finish_reason": "stop"}]}
        first = bank.put(key, generated, provenance={"source": "synthetic_cpu_fixture"})
        assert bank.get(key) == first
        altered = {**key, "model_sha256": "b" * 64}
        assert bank.get(altered) is None
        full = ForwardResultBank(root / "forward")
        artifact = root / "gen_1.jsonl"
        artifact.write_text("synthetic artifact\n")
        effective = {"request_body_sha256": json_hash({"type": "generate", "model": "both"}),
                     "model_sha256": model_hash, "prompt_sha256": key["prompt_sha256"]}
        entry = full.put(effective, {"ok": True, "file": "/work/gen_1.jsonl"}, request_number=1,
                         artifacts={"gen_1.jsonl": artifact})
        artifact.write_text("modified editor-owned copy\n")
        full.restore(entry, root / "restored", paths={"gen_1.jsonl": "gen_2.jsonl"})
        assert (root / "restored/gen_2.jsonl").read_text() == "synthetic artifact\n"
        report = {"schema_version": 1, "stage": "forward_bank_cpu", "status": "passed", "backend": "cpu_fixture",
                  "checks": {"identical_generation_hit": True, "changed_weights_miss": True,
                             "immutable_artifact_restored": True, "request_body_hashed": True},
                  "generation_key_fields": list(key), "synthetic_generation_entries": 1,
                  "synthetic_forward_entries": 1, "model_sha256": model_hash,
                  "provenance": {"command": "python scripts/check_forward_bank.py --output OUTPUT.json",
                                 "scope": "Synthetic CPU cache verification; no scientific model inference or provider calls"}}
        atomic_json(Path(output), report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(check(parser.parse_args().output)["status"])
