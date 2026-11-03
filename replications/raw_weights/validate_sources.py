"""Validate reconstructed readers against pinned published answers; no inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openwsp.budget import atomic_json

from .edits import apply_numeric_edits, shown_car_store, zero_minatar_store
from .runner import inventory, verified_source
from .scoring import score_label


def validate(source_root):
    source_root = Path(source_root)
    rows = []
    mapping = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias",
               "W2": "actor/dense2/kernel", "b2": "actor/dense2/bias",
               "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
    for item in inventory()["items"]:
        source = verified_source(source_root, item)
        answer_path = Path(item["prompt"]["path"]).with_name("answer.md")
        answer_bytes = (source_root / answer_path).read_bytes()
        answer = answer_bytes.decode()
        row = {"experiment": item["experiment"], "item_id": item["item_id"],
               "prompt_sha256": item["prompt"]["sha256"], "answer_path": str(answer_path),
               "answer_sha256": hashlib.sha256(answer_bytes).hexdigest()}
        if item["experiment"] in ("exp1", "exp2"):
            row["score"] = score_label(item["experiment"], item["item_id"], answer)
            if not row["score"]["correct"]:
                raise AssertionError("Published label no longer reproduces")
        else:
            car = item["experiment"] == "exp4"
            store = shown_car_store(source["prompt"]) if car else zero_minatar_store()
            row["parser"] = apply_numeric_edits(answer, store)
            row["changed_parameters"] = sum(int((store.load(n) != store.load_original(n)).count_nonzero()) for n in store.names)
            row["nonzero_parameters"] = sum(int(store.load(n).count_nonzero()) for n in store.names)
            network = ("exp4_fix_car/networks/edited_actor.npz" if car else
                       f"exp5_write_from_scratch/networks/{item['item_id']}/hand.npz")
            row["network_path"] = network
            row["network_sha256"] = hashlib.sha256((source_root / network).read_bytes()).hexdigest()
            with np.load(source_root / network, allow_pickle=False) as archive:
                row["source_dtypes"] = sorted({str(archive[name if car else mapping[name]].dtype) for name in store.names})
                row["equal_after_float32_conversion"] = all(
                    np.array_equal(store.load(name).numpy(), archive[name if car else mapping[name]].astype(np.float32))
                    for name in store.names)
            if row["parser"]["errors"] or not row["equal_after_float32_conversion"]:
                raise AssertionError("Published network no longer reproduces")
        rows.append(row)
    return {"schema_version": 1, "source_revision": inventory()["revision"],
            "validation": "released answers only; not new model outcomes", "items": rows,
            "rollout_validation": "pending", "scorer_source": "OpenWSP reconstruction from released rules"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source_root)
    atomic_json(args.output, result)
    print(json.dumps({"validated_items": len(result["items"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
