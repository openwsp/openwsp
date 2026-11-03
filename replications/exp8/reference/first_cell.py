# Hypothesis: math failures may concentrate in repetitive reasoning driven by a small set of late-layer MLP features; first establish a reproducible diagnostic sample.
import os, json, random, math, collections, numpy as np, torch
from pathlib import Path
torch.set_num_threads(8)
rng = random.Random(4217)
practice = {}
for source, fn in [("gsm8k", "gsm8k_train.jsonl"), ("math", "math_train.jsonl"), ("aime", "aime.jsonl")]:
    rows = [json.loads(s) for s in open("/practice/" + fn)]
    practice[source] = rows
    print(source, len(rows), "example fields:", sorted(rows[0]), "example id:", rows[0]["id"])
byid = {r["id"]: r for rows in practice.values() for r in rows}
diagnostic_ids = [r["id"] for r in rng.sample(practice["gsm8k"], 32)] + [r["id"] for r in rng.sample(practice["math"], 64)]
unused = [r for s in ["gsm8k", "math"] for r in practice[s] if r["id"] not in set(diagnostic_ids)]
print("Diagnostic ids:", json.dumps(diagnostic_ids))
print("Tensor count:", len(NAMES))
print("\n".join(NAMES[:45]))
print("Existing edits:", edited())
