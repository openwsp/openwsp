"""CPU-only MLP input means for the existing mechanical calibration contrast.

No generation or vocabulary projection is run. Saved output directions stay
fixed; this supplies the previously missing input statistics for variant 2c.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from openwsp.budget import atomic_json
from replications.exp8.repair import _file_hash, _json_hash
from replications.exp8.runtime import INSTRUCTION

LAYERS = (15, 19)
ANSWER_TOKENS = 8192


def _text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def select_examples(raw, benchmarks, expected_counts):
    """Use exactly controls.mechanical_edit's group rule and source order.

    Correctness fields are deliberately never accessed. The raw generation
    file is private and is not copied to any editor or public result directory.
    """
    from replications.exp8.controls import loop_fraction

    problems = {row["id"]: row["problem"] for row in benchmarks}
    if len(problems) != len(benchmarks):
        raise ValueError("Duplicate benchmark IDs")
    selected, seen = [], set()
    counts = {"looping": 0, "finished": 0}
    for record in raw:
        if record["finish_reason"] not in {"stop", "length"}:
            raise ValueError("Calibration contains an incomplete generation")
        finished = record["finish_reason"] == "stop"
        if not finished and loop_fraction(record["text"]) < 0.5:
            continue
        identity = record["problem_id"]
        if identity in seen:
            raise ValueError("Expected one calibration generation per problem")
        seen.add(identity)
        prompt = problems[identity] + "\n" + INSTRUCTION
        group = "finished" if finished else "looping"
        counts[group] += 1
        selected.append(
            {
                "id": identity,
                "group": group,
                "prompt": prompt,
                "text": record["text"],
                "prompt_sha256": _text_hash(prompt),
                "text_sha256": _text_hash(record["text"]),
            }
        )
    if counts != expected_counts:
        raise ValueError(f"Contrast groups differ from saved control2 method: {counts} != {expected_counts}")
    return selected


def _text_model(model):
    base = getattr(model, "model", model)
    text = getattr(base, "language_model", base)
    if not hasattr(text, "layers") or not hasattr(text, "embed_tokens"):
        raise ValueError("Model must expose its text decoder")
    return text


def input_means(model, tokenizer, example, *, layers=LAYERS, answer_tokens=ANSWER_TOKENS):
    """Teacher-force prompt IDs + first answer IDs, reducing down-proj inputs.

    Each complete answer activation array is rounded to float16 then converted
    to NumPy float32 before its mean. This mirrors the public activation-file
    reduction convention. It does not recompute the saved output direction.
    """
    import torch

    if type(answer_tokens) is not int or not 1 <= answer_tokens <= ANSWER_TOKENS:
        raise ValueError("answer_tokens must be from 1 through 8192")
    decoder = _text_model(model)
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(type(layer) is not int or not 0 <= layer < len(decoder.layers) for layer in layers)
    ):
        raise ValueError("Invalid target layers")
    if decoder.embed_tokens.weight.device.type != "cpu":
        raise ValueError("This extraction command requires a CPU model")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(example["text"], add_special_tokens=False)[:answer_tokens]
    if not prompt_ids or not answer_ids or len(prompt_ids) + len(answer_ids) > 34816:
        raise ValueError("Empty prompt/answer or teacher-forcing context overflow")
    result, hooks = {}, []
    expected_length = len(prompt_ids) + len(answer_ids)

    def capture(layer, arguments):
        values = arguments[0]
        width = decoder.layers[layer].mlp.down_proj.weight.shape[1]
        if tuple(values.shape) != (1, expected_length, width) or str(layer) in result:
            raise RuntimeError("Unexpected or repeated down-projection input")
        answer = values[0, len(prompt_ids) :].detach().float().numpy().astype(np.float16)
        if not np.isfinite(answer).all():
            raise ValueError("MLP input cannot be represented as finite float16")
        result[str(layer)] = answer.astype(np.float32).mean(axis=0)

    try:
        for layer in layers:

            def hook(module, arguments, layer=layer):
                capture(layer, arguments)

            hooks.append(decoder.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        model.eval()
        with torch.inference_mode():
            tokens = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device="cpu")
            decoder(input_ids=tokens, use_cache=False, output_attentions=False)
        if set(result) != {str(layer) for layer in layers}:
            raise RuntimeError("A requested MLP input was not observed")
        return result, {
            "prompt_tokens": len(prompt_ids),
            "answer_tokens": len(answer_ids),
            "input_ids_sha256": hashlib.sha256(
                np.asarray(prompt_ids + answer_ids, dtype="<i8").tobytes()
            ).hexdigest(),
        }
    finally:
        for hook in hooks:
            hook.remove()


def resources():
    """Read local cgroup limits without assuming host-wide CPU or memory access."""
    result = {"peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    for name in ("cpu.max", "memory.max", "memory.current", "memory.peak"):
        try:
            result[name] = (Path("/sys/fs/cgroup") / name).read_text().strip()
        except OSError:
            result[name] = None
    return result


def extract(model, tokenizer, examples, output, *, provenance, limit=None, layers=LAYERS):
    """Resume durable per-example CPU measurements bound to exact source inputs.

    A limit benchmarks the first examples without publishing a complete input
    contrast. A subsequent unbounded call reuses verified individual receipts.
    """
    if not examples or (limit is not None and (type(limit) is not int or limit <= 0)):
        raise ValueError("Extraction requires examples and a positive optional limit")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identities = [
        {key: row[key] for key in ("id", "group", "prompt_sha256", "text_sha256")} for row in examples
    ]
    if len({row["id"] for row in identities}) != len(identities):
        raise ValueError("Duplicate selected example identities")
    for example in examples:
        if example["group"] not in {"looping", "finished"} or any(
            _text_hash(example[field]) != example[field + "_sha256"] for field in ("prompt", "text")
        ):
            raise ValueError("Example identity or source text changed")
    binding = {
        "format": "openwsp.repair.input-activations",
        "format_version": 1,
        "examples": identities,
        "provenance": provenance,
        "layers": list(layers),
        "answer_token_cap": ANSWER_TOKENS,
        "input_concatenation": "prompt_ids + first8192_answer_ids",
        "activation_reduction": "down_proj input float16 intermediate, NumPy float32 per-example answer mean, float64 group sums",
    }
    manifest = output / "selection.json"
    if manifest.exists() and json.loads(manifest.read_text()) != binding:
        raise ValueError("Extraction source/model settings differ from the existing selection")
    if not manifest.exists():
        atomic_json(manifest, binding)
    selected = examples if limit is None else examples[:limit]
    counts = {"looping": 0, "finished": 0}
    sums = {group: {} for group in counts}
    receipts = []
    for index, example in enumerate(selected):
        key = _json_hash({"binding": _json_hash(binding), "example": identities[index]})
        folder = output / "examples" / key
        receipt_path = folder / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["request_sha256"] != key
                or _file_hash(folder / "means.npz") != receipt["arrays_sha256"]
            ):
                raise ValueError("Saved activation receipt or arrays changed")
            with np.load(folder / "means.npz", allow_pickle=False) as archive:
                means = {layer: archive[layer].copy() for layer in archive.files}
        else:
            start = time.monotonic()
            means, tokens = input_means(model, tokenizer, example, layers=layers)
            folder.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=folder, suffix=".npz")
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.savez_compressed(stream, **means)
                os.replace(temporary, folder / "means.npz")
            finally:
                Path(temporary).unlink(missing_ok=True)
            receipt = {
                "request_sha256": key,
                "example": identities[index],
                "tokens": tokens,
                "seconds": time.monotonic() - start,
                "arrays_sha256": _file_hash(folder / "means.npz"),
                "resources": resources(),
            }
            atomic_json(receipt_path, receipt)
        if set(means) != {str(layer) for layer in layers} or any(
            value.ndim != 1 or value.dtype != np.float32 or not np.isfinite(value).all()
            for value in means.values()
        ):
            raise ValueError("Invalid saved per-example means")
        group = example["group"]
        counts[group] += 1
        for layer, mean in means.items():
            if layer not in sums[group]:
                sums[group][layer] = np.zeros_like(mean, dtype=np.float64)
            if sums[group][layer].shape != mean.shape:
                raise ValueError("MLP input width changed between examples")
            sums[group][layer] += mean
        receipts.append(receipt)
        atomic_json(
            output / "progress.json",
            {"processed": len(receipts), "selected": len(examples), "counts": counts},
        )
    seconds = sum(receipt["seconds"] for receipt in receipts)
    total_tokens = sum(
        sum(receipt["tokens"][key] for key in ("prompt_tokens", "answer_tokens")) for receipt in receipts
    )
    result = {
        "format": "openwsp.repair.input-activations.summary",
        "format_version": 1,
        "status": "complete" if len(receipts) == len(examples) else "benchmark_only",
        "processed": len(receipts),
        "selected": len(examples),
        "counts": counts,
        "measured_seconds": seconds,
        "measured_tokens": total_tokens,
        "estimated_total_seconds_by_example_count": seconds * len(examples) / len(receipts),
        "selection_sha256": _file_hash(manifest),
        "resources": resources(),
        "model_inference": "CPU text decoder only, no vocabulary logits or generation",
    }
    if result["status"] == "complete":
        if not all(counts.values()):
            raise ValueError("Both contrast groups are required")
        differences = {
            f"model.language_model.layers.{layer}.mlp.down_proj.weight": sums["looping"][str(layer)]
            / counts["looping"]
            - sums["finished"][str(layer)] / counts["finished"]
            for layer in layers
        }
        np.savez_compressed(output / "input-differences.npz", **differences)
        result["input_differences_sha256"] = _file_hash(output / "input-differences.npz")
    atomic_json(output / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "raw", "benchmarks", "method", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--limit", type=int, help="Benchmark first N selected examples; full contrast remains unavailable"
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 4:
        raise ValueError("Use one through four CPU threads")
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    raw = [json.loads(line) for line in Path(args.raw).read_text().splitlines() if line.strip()]
    benchmarks = [json.loads(line) for line in Path(args.benchmarks).read_text().splitlines() if line.strip()]
    method = json.loads(Path(args.method).read_text())
    examples = select_examples(raw, benchmarks, method["counts"])
    model_path = Path(args.model)
    checkpoint_files = sorted(model_path.glob("*.safetensors"))
    if not checkpoint_files:
        raise ValueError("No local checkpoint files")
    provenance = {
        "raw_sha256": _file_hash(args.raw),
        "benchmarks_sha256": _file_hash(args.benchmarks),
        "saved_method_sha256": _file_hash(args.method),
        "checkpoint_files": {path.name: _file_hash(path) for path in checkpoint_files},
        "model_assets": {path.name: _file_hash(path) for path in sorted(model_path.glob("*.json"))},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "cpu_threads": args.threads,
        "selection_rule": "stop-finished vs length-cut with duplicate final-tail 8-gram share >=0.5, original calibration order",
        "output_directions": "Saved control2 directions are reused unchanged; no output-direction recomputation",
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to("cpu")
        .eval()
    )
    result = extract(model, tokenizer, examples, args.output, provenance=provenance, limit=args.limit)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
