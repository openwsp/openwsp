"""Summarize saved one-reply artifacts; never request or re-score a response."""

import argparse
import hashlib
import json
from pathlib import Path

from openwsp.budget import atomic_json

from .runner import inventory


def generate(output_root):
    root = Path(output_root)
    rows = []
    for editor in inventory()["editors"]:
        for item in inventory()["items"]:
            path = root / editor / item["experiment"] / item["item_id"] / "result.json"
            row = {"editor": editor, "experiment": item["experiment"], "item_id": item["item_id"],
                   "reply_status": "pending", "score_status": "pending", "prompt_sha256": item["prompt"]["sha256"]}
            if path.exists():
                data = path.read_bytes()
                record = json.loads(data)
                if record["binding"]["prompt_sha256"] != row["prompt_sha256"]:
                    raise ValueError("Result prompt differs from the released manifest")
                if "answer_sha256" in record:
                    answer = path.with_name("answer.md").read_bytes()
                    if hashlib.sha256(answer).hexdigest() != record["answer_sha256"]:
                        raise ValueError("Saved answer differs from its scored receipt")
                row.update(reply_status=record["status"], result_path=str(path.relative_to(root)),
                           result_sha256=hashlib.sha256(data).hexdigest())
                for key in ("answer_sha256", "answer_bytes", "error_type", "editor_contract"):
                    if key in record:
                        row[key] = record[key]
                receipt = record.get("receipt") or {}
                if receipt.get("usage") is not None:
                    row["usage"] = receipt["usage"]
                if "cost_usd" in receipt:
                    row["cost_usd"] = receipt["cost_usd"]
                if "billing" in receipt:
                    row["billing"] = receipt["billing"]
                score = record.get("score", {})
                if "correct" in score:
                    row.update(score_status="completed", correct=score["correct"], predicted=score["predicted"])
                evaluation = score.get("evaluation", {})
                if evaluation and evaluation.get("status") != "pending":
                    row["score_status"] = "completed"
                    row["evaluation"] = {key: value for key, value in evaluation.items() if key != "records"}
                if "parser" in score:
                    row["parser"] = score["parser"]
            rows.append(row)
    result = {"schema_version": 1, "source_revision": inventory()["revision"], "items_per_editor": inventory()["items_per_editor"],
              "basis": "Saved new editor responses only; source-network calibration is separate", "items": rows}
    atomic_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.output_root)
    print(json.dumps({"saved_replies": sum(row["reply_status"] == "completed" for row in result["items"]),
                      "completed_scores": sum(row["score_status"] == "completed" for row in result["items"]),
                      "planned_items": len(result["items"])}))


if __name__ == "__main__":
    main()
