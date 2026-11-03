"""Experiment 8 data and evaluation protocol. Never mounted in the editor sandbox."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import urllib.request
from pathlib import Path

from openwsp.metrics import finished_metrics

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text())
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch_source(relative, destination):
    url = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{CONFIG['source_revision']}/{relative}"
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / "sources.json").read_text())["files"][relative]
    if not target.exists() or digest(target) != expected:
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if digest(target) != expected:
        raise ValueError(f"Source hash mismatch: {relative}")
    return target


def prepare(data="/data", model="/model", practice="/practice"):
    """Download pinned base, preserve exact names, fetch practice and held-out data."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from datasets import load_dataset
    data, model, practice = map(Path, (data, model, practice))
    for path in (data, model, practice):
        path.mkdir(parents=True, exist_ok=True)
    original = data / "original"
    snapshot_download(CONFIG["model"], revision=CONFIG["revision"], local_dir=original,
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    if not (model / "model.safetensors").exists():
        tensors = {}
        for file in original.glob("*.safetensors"):
            with safe_open(file, framework="pt") as handle:
                for name in handle.keys():
                    if name.startswith("model.language_model.") or name == "lm_head.weight":
                        tensors[name] = handle.get_tensor(name)
        save_file(tensors, model / "model.safetensors", metadata={"format": "pt"})
    for file in original.iterdir():
        if file.is_file() and file.suffix != ".safetensors" and "index" not in file.name:
            shutil.copyfile(file, model / file.name)
    # Authors' text-only config and chat assets define the CPU kernel's model view.
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"):
        fetch_source(f"exp8_math/base_model/Qwen3.5-0.8B-text/{name}", model / name)
    for name in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        fetch_source(f"exp8_math/practice/{name}", practice / name)
    edit = data / "reference_edit"
    for name in ("manifest.json", "edit.npz"):
        fetch_source(f"exp8_math/runs/p50c/edit/{name}", edit / name)
    benchmarks = data / "benchmarks"
    benchmarks.mkdir(exist_ok=True)
    if not (benchmarks / "math500.jsonl").exists():
        rows = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
        write_rows(benchmarks / "math500.jsonl", rows)
    if not (benchmarks / "gsm8k.jsonl").exists():
        rows = [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))]
        write_rows(benchmarks / "gsm8k.jsonl", rows)
    if not (benchmarks / "ifeval.jsonl").exists():
        rows = [{"id": f"ifeval_{i}", "kind": "ifeval", "prompt": row["prompt"],
                 "instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]}
                for i, row in enumerate(load_dataset("google/IFEval", split="train"))]
        write_rows(benchmarks / "ifeval.jsonl", rows)
    return {"status": "done", "model_revision": CONFIG["revision"],
            "model_sha256": digest(model / "model.safetensors"),
            "practice": {p.name: digest(p) for p in practice.glob("*.jsonl")}}


def write_rows(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate_outputs(outputs, rows, *, directory, stage, benchmark, samples, seconds, cost_usd,
                     provenance, thinking=True):
    """Keep full text on the volume, with an auditable compact row for every sample."""
    import importlib.util
    import sys
    if str(ROOT / "compat") not in sys.path:
        sys.path.insert(0, str(ROOT / "compat"))
    if str(ROOT / "vendor") not in sys.path:
        sys.path.insert(0, str(ROOT / "vendor"))
    spec = importlib.util.spec_from_file_location("exp8_mathqa", ROOT / "vendor/mathqa.py")
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw, records = [], []
    if len(outputs) != len(rows):
        raise ValueError("Missing model outputs")
    for row, generated in zip(rows, outputs, strict=True):
        if len(generated["samples"]) != samples:
            raise ValueError("Missing evaluation samples")
        for index, output in enumerate(generated["samples"]):
            if output["finish_reason"] not in ("stop", "length"):
                raise ValueError("Incomplete generation cannot be scored")
            if benchmark == "ifeval":
                correct, pred, extra = grade_ifeval(row, output["text"], thinking)
            else:
                pred, _ = grader.extract(output["text"])
                correct = bool(grader.is_correct(pred, row["answer"]))
                extra = {}
            raw.append({"problem_id": row["id"], "sample_index": index, **output})
            records.append({"problem_id": row["id"], "sample_index": index, "answer": pred,
                            "correct": correct, "token_count": output["token_count"],
                            "finish_reason": output["finish_reason"], **extra})
    raw_path = directory / "raw_generations.jsonl"
    write_rows(raw_path, raw)
    sha = digest(raw_path)
    for record in records:
        record.update(raw_path=str(raw_path), raw_sha256=sha)
    write_rows(directory / "samples.jsonl", records)
    n = len(records)
    accuracy = sum(r["correct"] for r in records) / n
    summary = {"status": "done", "stage": stage, "model": CONFIG["model"], "benchmark": benchmark,
               "samples_per_problem": samples, "n": n, "correct": sum(r["correct"] for r in records),
               "accuracy": accuracy, "standard_error": math.sqrt(accuracy * (1-accuracy) / n),
               "mean_tokens": sum(r["token_count"] for r in records)/n,
               "cut_share": sum(r["finish_reason"] == "length" for r in records)/n,
               **finished_metrics(records),
               "seconds": seconds, "cost_usd": cost_usd, "provenance": provenance}
    atomic_write(directory / "summary.json", summary)
    return summary


def grade_ifeval(row, text, thinking=True):
    """Authors' strict prompt-level adapter over the official pinned checkers."""
    import sys
    vendor = str(ROOT / "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from instruction_following_eval import instructions_registry as registry
    response = text.split("</think>")[-1].strip() if "</think>" in text else ("" if thinking else text.strip())
    followed = []
    for identifier, kwargs in zip(row["instruction_id_list"], row["kwargs"], strict=True):
        instruction = registry.INSTRUCTION_DICT[identifier](identifier)
        instruction.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
        arguments = instruction.get_instruction_args()
        if arguments and "prompt" in arguments:
            instruction.build_description(prompt=row["prompt"])
        followed.append(bool(response) and bool(instruction.check_following(response)))
    return all(followed), None, {"inst_ok": sum(followed), "inst_total": len(followed)}
