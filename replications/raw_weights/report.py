"""Summarize saved one-reply artifacts; never request or re-score a response."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from openwsp.budget import atomic_json

from .runner import ROOT, inventory


def ppo_reference_summary():
    """Average every declared released training seed; never select a winner."""
    reference_path = ROOT / "replications/exp5/reference/minatar_eval.json"
    raw = reference_path.read_bytes()
    sources = json.loads((ROOT / "replications/exp5/sources.json").read_text())
    expected = sources["files"]["exp5_write_from_scratch/eval/minatar_eval.json"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("PPO reference differs from the pinned authors' source")
    reference, rows = json.loads(raw), []
    for game in ("breakout", "seaquest", "asterix"):
        for steps in (100, 500):
            prefix = f"ppo{steps}_s"
            count = 10 if steps == 100 and game != "seaquest" else 3
            names = [prefix + str(seed) for seed in range(count)]
            if {name for name in reference[game] if name.startswith(prefix)} != set(names):
                raise ValueError("Released PPO training seed set differs from its declaration")
            if any(reference[game][name]["episodes"] != 1000 for name in names):
                raise ValueError("PPO reference episode counts changed")
            means = [reference[game][name]["mean"] for name in names]
            rows.append({"game": game, "training_steps_millions": steps, "seed_count": count,
                         "seed_names": names, "episodes_per_seed": 1000,
                         "mean_of_seed_means": statistics.fmean(means), "minimum_seed_mean": min(means),
                         "maximum_seed_mean": max(means), "individual_means": dict(zip(names, means, strict=True))})
    return {"schema_version": 1, "basis": "Authors' reference only; no new PPO training or evaluation",
            "aggregation": "Equal weight over every released training seed; no best-seed selection",
            "source_path": str(reference_path.relative_to(ROOT)), "source_sha256": expected,
            "source_revision": sources["revision"], "rows": rows}


def comparison_markdown(result):
    """Render the public comparison from saved outcomes and the pinned references."""
    base = "replications/raw_weights/results/"
    link = lambda value, path: f"[{value}]({path})"
    summary_path = base + "summary.json"
    rows = result["items"]

    def pending(editor):
        return link("Pending", base + "fable-blocker.json") if editor == "fable_native_anthropic" else link("Pending", summary_path)

    def selected(editor, experiment, item=None):
        return [row for row in rows if row["editor"] == editor and row["experiment"] == experiment
                and (item is None or row["item_id"] == item)]

    def labels(editor, experiment):
        values = selected(editor, experiment)
        planned = sum(item["experiment"] == experiment for item in inventory()["items"])
        if len(values) != planned or any(row["score_status"] != "completed" for row in values):
            return pending(editor)
        return link(f"{sum(row['correct'] for row in values)}/{len(values)}", summary_path)

    def car(editor):
        values = selected(editor, "exp4")
        if len(values) != 1 or values[0]["score_status"] != "completed":
            return pending(editor)
        row, metric = values[0], values[0]["evaluation"]
        return link(f"{metric['successes']:,}/{metric['episodes']:,} successes; "
                    f"{row['changed_parameters']:,}/{row['parameters']:,} weights changed", summary_path)

    def minatar(editor, game):
        values = selected(editor, "exp5", game)
        if len(values) != 1 or values[0]["score_status"] != "completed":
            return pending(editor)
        metric = values[0]["evaluation"]
        return link(f"{metric['mean']:.2f} ± {metric['standard_error']:.2f}", base + values[0]["result_path"])

    editors = inventory()["editors"]
    lines = ["| Experiment | Authors' released result | Astra (subscription) | Fable (native Anthropic) |",
             "| --- | --- | --- | --- |"]
    for experiment, label in (("exp1", "1: identify the game"), ("exp2", "2: identify image families")):
        path = f"replications/{experiment}/reference/scores.json"
        reference = json.loads((ROOT / path).read_text())
        author = link(f"{sum(row['correct'] for row in reference)}/{len(reference)}", path)
        lines.append(f"| {link(label, 'replications/' + experiment)} | {author} | "
                     + " | ".join(labels(editor, experiment) for editor in editors) + " |")
    path = "replications/exp4/reference/oneshot_result.json"
    reference = json.loads((ROOT / path).read_text())
    test, changed = reference["test"], reference["changes"]
    author = link(f"{round(test['success'] * test['episodes']):,}/{test['episodes']:,} successes; "
                  f"{changed['changed']:,}/{changed['entries']:,} weights changed", path)
    lines.append(f"| {link('4: repair the car', 'replications/exp4')} | {author} | "
                 + " | ".join(car(editor) for editor in editors) + " |")
    lines += ["", "Experiment 5 scores are mean ± standard error over "
              + link("1,000 episodes per policy", "replications/exp5/reference/minatar_eval.json")
              + ". Each editor supplies one reply per game; standard errors describe variability across game episodes.", "",
              "| Game | Authors' PPO 100M: mean [range], seeds | Authors' PPO 500M: mean [range], seeds | Authors' Astra | Astra (subscription) | Fable (native Anthropic) |",
              "| --- | --- | --- | --- | --- | --- |"]
    reference = json.loads((ROOT / "replications/exp5/reference/minatar_eval.json").read_text())
    ppo = ppo_reference_summary()["rows"]
    for game in ("breakout", "seaquest", "asterix"):
        baselines = []
        for steps in (100, 500):
            row = next(row for row in ppo if row["game"] == game and row["training_steps_millions"] == steps)
            baselines.append(link(f"{row['mean_of_seed_means']:.2f} [{row['minimum_seed_mean']:.2f}, "
                                  f"{row['maximum_seed_mean']:.2f}], {row['seed_count']} seeds",
                                  "replications/exp5/analysis/ppo-seed-summary.json"))
        hand = reference[game]["hand"]
        author = link(f"{hand['mean']:.2f} ± {hand['se']:.2f}", "replications/exp5/reference/minatar_eval.json")
        lines.append(f"| {game.title()} | " + " | ".join(baselines + [author] + [minatar(editor, game) for editor in editors]) + " |")
    lines += ["", "PPO entries are reference-only, equally weighted means over every released training seed; the ranges and complete seed lists are "
              + link("retained in the source-derived summary", "replications/exp5/analysis/ppo-seed-summary.json")
              + ". The authors' post reports best-seed PPO values. No PPO training or model selection was repeated."]
    return "\n".join(lines) + "\n"


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
                expected_binding = {"source_revision": inventory()["revision"], "prompt_sha256": item["prompt"]["sha256"],
                    "instructions_sha256": item["instructions"]["sha256"], "editor": editor,
                    "experiment": item["experiment"], "item_id": item["item_id"]}
                if record["binding"] != expected_binding:
                    raise ValueError("Result editor/item/source binding differs from the released manifest")
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
                    authors_input = item.get("authors_usage", {}).get("input_tokens")
                    our_input = receipt["usage"].get("input_tokens")
                    if type(authors_input) is int and type(our_input) is int:
                        row["authors_input_tokens"] = authors_input
                        row["ours_minus_authors_input_tokens"] = our_input - authors_input
                if "cost_usd" in receipt:
                    row["cost_usd"] = receipt["cost_usd"]
                if "billing" in receipt:
                    row["billing"] = receipt["billing"]
                score = record.get("score", {})
                if score and ("answer_sha256" not in record or record["status"] != "completed"):
                    raise ValueError("A published score requires a completed hash-verified answer")
                if "correct" in score:
                    row.update(score_status="completed", correct=score["correct"], predicted=score["predicted"])
                evaluation = score.get("evaluation", {})
                if evaluation and evaluation.get("status") in (None, "completed"):
                    metric = "success" if item["experiment"] == "exp4" else "mean"
                    if type(evaluation.get("episodes")) is not int or evaluation["episodes"] < 1 or metric not in evaluation:
                        raise ValueError("Completed evaluation is missing its episode count or metric")
                    row["score_status"] = "completed"
                    row["evaluation"] = {key: value for key, value in evaluation.items() if key != "records"}
                elif evaluation.get("status") not in (None, "pending"):
                    row["score_status"] = evaluation["status"]
                if "parser" in score:
                    row["parser"] = score["parser"]
                for key in ("parameters", "changed_parameters", "nonzero_parameters"):
                    if key in score:
                        row[key] = score[key]
            rows.append(row)
    result = {"schema_version": 1, "source_revision": inventory()["revision"], "items_per_editor": inventory()["items_per_editor"],
              "basis": "Saved new editor responses only; source-network calibration is separate", "items": rows}
    result["editors"] = {}
    for editor in inventory()["editors"]:
        selected = [row for row in rows if row["editor"] == editor]
        result["editors"][editor] = {
            "planned_items": len(selected),
            "completed_replies": sum(row["reply_status"] == "completed" for row in selected),
            "completed_scores": sum(row["score_status"] == "completed" for row in selected),
            "blocked_items": sum(row["reply_status"] == "blocked" for row in selected),
            "pending_items": sum(row["reply_status"] == "pending" for row in selected),
            "usage_totals": {
                key: sum(row["usage"][key] for row in selected if type(row.get("usage", {}).get(key)) is int)
                for key in sorted({key for row in selected for key, value in row.get("usage", {}).items()
                                   if type(value) is int})
            },
        }
    atomic_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--update-readme", action="store_true", help="Refresh the marked public raw-results tables")
    args = parser.parse_args()
    result = generate(args.output_root)
    atomic_json(ROOT / "replications/exp5/analysis/ppo-seed-summary.json", ppo_reference_summary())
    public_root = ROOT / "replications/raw_weights/results"
    if args.output_root.resolve() == public_root:
        table = comparison_markdown(result)
        (public_root / "comparison.md").write_text(table.replace("](replications/", "](../../../replications/"))
        if args.update_readme:
            readme = ROOT / "README.md"
            text = readme.read_text()
            start, end = "<!-- OPENWSP:RAW_RESULTS:START -->", "<!-- OPENWSP:RAW_RESULTS:END -->"
            if text.count(start) != 1 or text.count(end) != 1:
                raise ValueError("README raw-results markers are missing or duplicated")
            readme.write_text(text.split(start, 1)[0] + start + "\n" + table + end + text.split(end, 1)[1])
    elif args.update_readme:
        raise ValueError("README publication requires the repository's public raw-results directory")
    print(json.dumps({"saved_replies": sum(row["reply_status"] == "completed" for row in result["items"]),
                      "completed_scores": sum(row["score_status"] == "completed" for row in result["items"]),
                      "planned_items": len(result["items"])}))


if __name__ == "__main__":
    main()
