"""Registered-model execution for the public Modal forward transport.

The deployed gateway must serialize inputs (max_containers=1, max_inputs=1),
disable retries, and enforce PUBLIC_LIMITS on its own CPU resources and timeout.
Its GPU children use the separately bounded Generator and TransformerReader.
No provider object is constructed and no paid work is launched at import time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from ..budget import BudgetExceeded, BudgetLedger, atomic_json
from ..core import WeightStore
from .jobs import RemoteJobRunner
from .modal_api import MAX_ARTIFACT_BYTES, apply_payload_edit, make_response, reservation_bound

GPU_LIMITS = {"gpu": "H100", "cpu_cores": 4, "memory_gib": 16,
              "timeout_seconds": 1800, "startup_seconds": 600, "scaledown_seconds": 120}
GATEWAY_LIMITS = {"gpu": None, "cpu_cores": 1, "memory_gib": 4,
                  "timeout_seconds": 3600, "startup_seconds": 120, "scaledown_seconds": 5}
PUBLIC_LIMITS = {**GATEWAY_LIMITS, "children": [dict(GPU_LIMITS), dict(GPU_LIMITS)]}
ARTIFACT_VOLUME = "openwsp-exp8-data"


def _runtime():
    import modal
    from transformers import AutoTokenizer

    from replications.exp8.runtime import CONFIG

    from .modal_app import Generator, TransformerReader, data_volume

    return SimpleNamespace(
        registration={"model": CONFIG["model"], "revision": CONFIG["revision"]},
        Generator=Generator, TransformerReader=TransformerReader,
        tokenizer=lambda path: AutoTokenizer.from_pretrained(str(path), local_files_only=True),
        commit=data_volume.commit, reload=data_volume.reload,
        call_factory=modal.FunctionCall.from_id,
    )


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_checkpoint(store, base_path, destination, manifest):
    """Publish an immutable checkpoint made from the fully validated v1 edit.

    The complete marker is written last. No submitted paths, model IDs, or
    request strings become filenames. Unmodified safetensors retain their exact
    tensor names; changed values use the base tensor dtype. JSON, Jinja and text
    assets cover the registered Qwen tokenizer; other model registrations must
    extend that asset allowlist when their tokenizer needs other file formats.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_path, destination = Path(base_path), Path(destination)
    edit_id = manifest["edit_sha256"]
    names = store.edited()
    if not names:
        return str(base_path)
    checkpoint = destination / edit_id
    marker = checkpoint / "openwsp-complete.json"
    if marker.exists():
        if json.loads(marker.read_text()) != {"edit_sha256": edit_id, "base": {
            "model": manifest["base_model"], "revision": manifest["base_revision"],
        }}:
            raise ValueError("Frozen checkpoint identity does not match the validated edit")
        return str(checkpoint)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=destination) as temporary:
        temporary = Path(temporary)
        for source in base_path.iterdir():
            if source.is_symlink():
                raise ValueError("Pinned base checkpoint cannot contain symlinks")
            if not source.is_file():
                continue
            if source.suffix == ".safetensors":
                with safe_open(source, framework="pt", device="cpu") as handle:
                    tensors = {name: handle.get_tensor(name) for name in handle.keys()}  # noqa: SIM118 -- safe_open is not iterable.
                for name in names:
                    if name in tensors:
                        tensors[name] = store.load(name).to(tensors[name].dtype).contiguous()
                save_file(tensors, temporary / source.name, metadata={"format": "pt"})
            elif source.suffix in {".json", ".jinja", ".txt"}:
                shutil.copyfile(source, temporary / source.name)
        atomic_json(temporary / marker.name, {"edit_sha256": edit_id, "base": {
            "model": manifest["base_model"], "revision": manifest["base_revision"],
        }})
        if checkpoint.exists():
            # A prior interrupted publication never had a completion marker.
            shutil.rmtree(checkpoint)
        os.replace(temporary, checkpoint)
    return str(checkpoint)


def _number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} exceeds the registered backend's limits")
    return value


def _generate_request(request, tokenizer):
    allowed = {"type", "model", "prompts", "n", "max_tokens", "temperature", "top_p", "top_k",
               "presence_penalty", "seed", "enable_thinking"}
    if set(request) - allowed:
        raise ValueError("Unknown public generation option; supply prompts rather than private dataset IDs")
    prompts, model = request.get("prompts"), request.get("model")
    if model not in {"original", "edited", "both"}:
        raise ValueError("Generation model must be original, edited or both")
    if not isinstance(prompts, list) or not prompts or any(not isinstance(text, str) for text in prompts):
        raise ValueError("prompts must be a nonempty list of strings")
    n, maximum = request.get("n", 1), request.get("max_tokens", 1024)
    if type(n) is not int or n < 1 or type(maximum) is not int or not 1 <= maximum <= 32768:
        raise ValueError("Invalid generation sample count or token bound")
    if len(prompts) * n * (2 if model == "both" else 1) > 512:
        raise ValueError("Generation requests allow at most 512 total samples")
    thinking = request.get("enable_thinking", True)
    if type(thinking) is not bool:
        raise ValueError("enable_thinking must be boolean")
    rows = []
    for index, prompt in enumerate(prompts):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking,
        )
        length = len(tokenizer.encode(formatted, add_special_tokens=False))
        if not length or length + maximum > 34816:
            raise ValueError("Prompt and generation exceed the registered model context length")
        rows.append({"id": str(index) + "-" + hashlib.sha256(prompt.encode()).hexdigest(), "prompt": prompt})
    top_k, seed = request.get("top_k", 20), request.get("seed", 0)
    if type(top_k) is not int or (top_k != -1 and not 1 <= top_k <= 1000):
        raise ValueError("top_k must be -1 or an integer in 1..1000")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in 0..2147483647")
    sampling = {
        "n": n, "max_tokens": maximum, "seed": seed, "top_k": top_k,
        "temperature": _number(request.get("temperature", 0.6), "temperature", 0, 2),
        "top_p": _number(request.get("top_p", 0.95), "top_p", 0.000001, 1),
        "presence_penalty": _number(request.get("presence_penalty", 0), "presence_penalty", -2, 2),
    }
    return rows, sampling, thinking


def _teacher_request(request, tokenizer, base_path):
    from replications.exp8.internals import tokenize_request

    if set(request) - {"type", "model", "texts", "chats", "return", "enable_thinking"}:
        raise ValueError("Unknown public teacher-forced option")
    if request.get("model") not in {"original", "edited"}:
        raise ValueError("Teacher-forced model must be original or edited")
    config = json.loads((base_path / "config.json").read_text())
    config = config.get("text_config", config)
    layers = config.get("num_hidden_layers", 24)
    layer_types = config.get("layer_types")
    full_attention = (tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
                      if layer_types is not None else tuple(range(3, layers, 4)))
    ids, ret = tokenize_request(tokenizer, request, num_layers=layers, full_attention_layers=full_attention)
    if ret.get("topk", 0) > config["vocab_size"]:
        raise ValueError("topk exceeds the registered model vocabulary")
    batch, length = len(ids), max(map(len, ids))
    elements = batch * length
    byte_bound = elements * (
        8 + 4 * bool(ret.get("token_logprobs")) + 12 * ret.get("topk", 0)
        + 2 * config["hidden_size"] * len(ret.get("hidden_states", []))
        + 2 * config["intermediate_size"] * len(ret.get("mlp_act", []))
        + 2 * config["num_attention_heads"] * length * len(ret.get("attn_layers", []))
    )
    # Include array headers and the small metadata document in the transport bound.
    if byte_bound + 1024**2 > MAX_ARTIFACT_BYTES:
        raise ValueError("Teacher-forced outputs exceed the 6 GiB transport bound")


def _artifacts(directory, data_root):
    descriptors, total = [], 0
    if directory.is_symlink():
        raise ValueError("Public forward artifact directory cannot be a symlink")
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix not in {".npy", ".json"}:
            raise ValueError("Unexpected or unsafe public forward artifact")
        size = path.stat().st_size
        total += size
        if total > MAX_ARTIFACT_BYTES:
            raise ValueError("Public forward artifacts exceed 6 GiB")
        descriptors.append({"volume": ARTIFACT_VOLUME, "path": path.relative_to(data_root).as_posix(),
                            "name": path.name, "sha256": _digest(path), "size_bytes": size})
    if not descriptors:
        raise ValueError("Teacher-forced worker produced no artifacts")
    return descriptors


def execute(payload, *, data_root="/data", runtime=None):
    """Run one explicit public request with private durable state and budget.

    runtime/data_root are dependency-injection seams for CPU contract tests and
    deployments; they cannot be selected through the remote payload. Callers must
    retain the same request ID and bytes when recovering an existing request.
    """
    started = time.monotonic()
    if (not isinstance(payload, dict) or not isinstance(payload.get("request_id"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", payload["request_id"])):
        raise ValueError("Invalid public forward request ID")
    runtime = runtime or _runtime()
    runtime.reload()
    data_root = Path(data_root)
    base_path = data_root / "original"
    state_dir = data_root / "public-forward-state" / payload["request_id"]
    artifact_dir = data_root / "public-forward" / payload["request_id"]
    state_dir.mkdir(parents=True, exist_ok=True)
    # Check immutable request identity before applying any new submitted tensors
    # to this request's persisted store. Full hash/base validation still follows.
    try:
        header = json.loads(payload["edit"]["manifest_json"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid public forward edit manifest") from error
    if not isinstance(header, dict):
        raise TypeError("Public forward edit manifest must be an object")
    identity = {"base": payload.get("base"), "request": payload.get("request"), "limits": payload.get("limits"),
                "edit_sha256": header.get("edit_sha256")}
    request_path, response_path = state_dir / "request.json", state_dir / "response.json"
    if request_path.exists() and json.loads(request_path.read_text()) != identity:
        raise ValueError("Request ID already belongs to a different public forward request")
    store = WeightStore(base_path, edit_dir=state_dir / "edited")
    manifest = apply_payload_edit(
        payload, store, base_model=runtime.registration["model"], base_revision=runtime.registration["revision"],
        limits=PUBLIC_LIMITS,
    )
    if manifest["edit_sha256"] != identity["edit_sha256"]:
        raise ValueError("Validated edit identity differs from the submitted manifest")
    atomic_json(request_path, identity)
    runtime.commit()
    if response_path.exists():
        response = json.loads(response_path.read_text())
        if (response.get("request_id") != payload["request_id"] or response.get("base") != payload["base"]
                or response.get("edit_sha256") != manifest["edit_sha256"]):
            raise ValueError("Stored public response identity is invalid")
        return response
    request = payload["request"]
    tokenizer = runtime.tokenizer(base_path)
    if request.get("type") == "generate":
        rows, sampling, thinking = _generate_request(request, tokenizer)
    elif request.get("type") == "teacher_forced":
        _teacher_request(request, tokenizer, base_path)
    else:
        raise ValueError("Public forward type must be generate or teacher_forced")
    modes = ("original", "edited") if request["model"] == "both" else (request["model"],)
    checkpoints = {"original": str(base_path)}
    if "edited" in modes:
        checkpoints["edited"] = freeze_checkpoint(
            store, base_path, data_root / "public-forward-checkpoints", manifest,
        )
        runtime.commit()
    budget = reservation_bound(PUBLIC_LIMITS)
    ledger = BudgetLedger(state_dir / "budget.json", modal_task_limit=budget,
                          modal_cycle_limit=min(budget, Decimal(42)), modal_prior_usage=0)
    jobs = RemoteJobRunner(ledger, state_dir / "jobs", commit=runtime.commit, call_factory=runtime.call_factory)
    child_bound = reservation_bound(GPU_LIMITS)
    child_options = {key: value for key, value in GPU_LIMITS.items() if key != "timeout_seconds"}
    keys = {mode: "public/" + payload["request_id"] + "/" + mode for mode in modes}
    missing = sum(not jobs._path(key).exists() for key in keys.values())
    available = Decimal(str(ledger.snapshot()["summary"]["modal"]["available_usd"]))
    if available < child_bound * missing:
        raise BudgetExceeded("Insufficient reserved capacity for all public forward children")
    # Launch every arm before collecting any result so paired passes run together.
    for mode in modes:
        if request["type"] == "generate":
            def spawn(mode=mode):
                return runtime.Generator(checkpoint=checkpoints[mode], slot="public-" + mode).generate.spawn(
                    rows, sampling, thinking, False,
                )
        else:
            artifact_dir.mkdir(parents=True, exist_ok=True)

            def spawn(mode=mode):
                return runtime.TransformerReader(
                    checkpoint=checkpoints[mode], eager=bool(request.get("return", {}).get("attn_layers")),
                ).read.spawn(request, str(artifact_dir))
        jobs.launch(keys[mode], spawn, upper_bound_usd=child_bound, cost_options=child_options,
                    fingerprint={**identity, "mode": mode})
    results = {}
    for mode in modes:
        remaining = GATEWAY_LIMITS["timeout_seconds"] - (time.monotonic() - started) - 60
        if remaining <= 0:
            raise TimeoutError("Gateway deadline reached; child reservations and call IDs retained")
        results[mode] = jobs.result(keys[mode], timeout=min(2520, remaining))
    artifacts = []
    if request["type"] == "generate":
        public = {}
        for mode, value in results.items():
            outputs = value.get("outputs")
            if (not isinstance(outputs, list) or len(outputs) != len(rows)
                    or [row.get("id") for row in outputs] != [row["id"] for row in rows]):
                raise ValueError("Generation output does not correspond to the submitted prompts")
            public[mode] = {"outputs": outputs}
        result = public if len(modes) > 1 else public[modes[0]]
    else:
        # The worker commits its /data artifacts before returning. Publish only
        # files in this request's private directory, never arbitrary worker paths.
        runtime.reload()
        artifacts = _artifacts(artifact_dir, data_root)
        value = results[modes[0]]
        result = {key: value[key] for key in ("lengths", "mean_token_logprob", "bytes") if key in value}
        result["arrays"] = value.get("files", {})
    result.update(ok=True, type=request["type"], model=request["model"])
    child_cost = sum((Decimal(row["actual_usd"]) for row in ledger.snapshot()["reservations"].values()
                      if row["status"] == "settled"), Decimal(0))
    # Charge the whole CPU-gateway bound: publishing a receipt occurs before the
    # invocation exits, so the measured interval alone cannot bound a stuck exit.
    total_cost = child_cost + reservation_bound(GATEWAY_LIMITS)
    response = make_response(payload, result, container_seconds=time.monotonic() - started,
                             cost_usd=total_cost, artifacts=artifacts)
    atomic_json(response_path, response)
    runtime.commit()
    return response
