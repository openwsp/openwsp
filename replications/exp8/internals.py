"""Teacher-forced reads without training, gradients, or autoregressive generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FULL_ATTENTION = (3, 7, 11, 15, 19, 23)


def tokenize_request(tokenizer, request, *, num_layers=24, full_attention_layers=FULL_ATTENTION):
    """Validate the released request schema; model dimensions may vary in other uses."""
    if not isinstance(request, dict) or ("texts" in request) == ("chats" in request):
        raise ValueError("Supply exactly one of texts or chats")
    if type(request.get("enable_thinking", False)) is not bool:
        raise ValueError("enable_thinking must be boolean")
    if "texts" in request:
        texts = request["texts"]
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must be a list of strings")
    else:
        chats = request["chats"]
        if not isinstance(chats, list) or any(not isinstance(chat, list) for chat in chats):
            raise ValueError("chats must be a list of message lists")
        texts = [
            tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=request.get("enable_thinking", False),
            )
            for chat in chats
        ]
    if not 1 <= len(texts) <= 1024:
        raise ValueError("Teacher-forced requests allow 1..1024 texts")
    ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    if any(not row or len(row) > 32768 for row in ids) or sum(map(len, ids)) > 524288:
        raise ValueError("Teacher-forced token limit exceeded")
    ret = request.get("return", {})
    if not isinstance(ret, dict) or set(ret) - {
        "token_logprobs",
        "topk",
        "hidden_states",
        "mlp_act",
        "attn_layers",
    }:
        raise ValueError("Unknown teacher-forced output")
    for key, allowed in (
        ("hidden_states", range(num_layers + 1)),
        ("mlp_act", range(num_layers)),
        ("attn_layers", full_attention_layers),
    ):
        indices = ret.get(key, [])
        if (
            not isinstance(indices, list)
            or any(type(index) is not int or index not in allowed for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(f"Invalid {key}")
    if type(ret.get("token_logprobs", False)) is not bool:
        raise ValueError("token_logprobs must be boolean")
    if type(ret.get("topk", 0)) is not int or not 0 <= ret.get("topk", 0) <= 50:
        raise ValueError("topk must be in 0..50")
    if ret.get("attn_layers") and (len(ids) > 2 or max(map(len, ids)) > 2048):
        raise ValueError("Attention reads allow at most 2 texts of at most 2048 tokens")
    return ids, ret


def _components(model):
    """Accept the HF causal-LM and multimodal wrappers sharing this text backbone."""
    base = getattr(model, "model", model)
    text_model = getattr(base, "language_model", base)
    if not all(hasattr(text_model, name) for name in ("embed_tokens", "layers", "norm")):
        raise ValueError("Model must expose a decoder with embed_tokens, layers, and norm")
    head = getattr(model, "lm_head", None)
    if head is None and hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise ValueError("Model must expose its output projection")
    return text_model, head


def generation_summary(model, tokenizer, text, prompt_length, hidden_layers, mlp_layers):
    """Reduce generation activations without writing discarded token arrays.

    Match capture's float16 storage followed by NumPy float32 reduction, while
    retaining only the two vectors the generation contract actually returns.
    """
    import torch

    text_model, _ = _components(model)
    layers = text_model.layers
    request = {"texts": [text], "return": {"hidden_states": hidden_layers, "mlp_act": mlp_layers}}
    ids, _ = tokenize_request(tokenizer, request, num_layers=len(layers),
                             full_attention_layers=tuple(i for i, layer in enumerate(layers) if hasattr(layer, "self_attn")))
    if not 0 < prompt_length <= len(ids[0]):
        raise ValueError("Prompt length is outside the teacher-forced sequence")
    hidden_size = text_model.embed_tokens.weight.shape[1]
    mlp_size = layers[mlp_layers[0]].mlp.down_proj.weight.shape[1] if mlp_layers else 0
    result = {"hidden_last_prompt.npy": np.zeros((len(hidden_layers), hidden_size), dtype=np.float32),
              "hidden_mean_answer.npy": np.zeros((len(hidden_layers), hidden_size), dtype=np.float32)}
    if mlp_layers:
        result.update({"mlp_last_prompt.npy": np.zeros((len(mlp_layers), mlp_size), dtype=np.float32),
                       "mlp_mean_answer.npy": np.zeros((len(mlp_layers), mlp_size), dtype=np.float32)})
    hooks = []

    def reduce(key, index, value):
        if value.shape[:2] != (1, len(ids[0])):
            raise RuntimeError("Unexpected generation activation shape")
        values = value[0].detach().float().cpu().numpy().astype(np.float16)
        result[f"{key}_last_prompt.npy"][index] = values[prompt_length - 1]
        answer = values[prompt_length:prompt_length + 8192]
        if len(answer):
            result[f"{key}_mean_answer.npy"][index] = answer.astype(np.float32).mean(0)

    try:
        for index, layer in enumerate(hidden_layers):
            if layer == len(layers):
                continue
            module = text_model.embed_tokens if layer == 0 else layers[layer - 1]
            def hidden_hook(module, arguments, value, index=index):
                reduce("hidden", index, value[0] if isinstance(value, tuple) else value)
            hooks.append(module.register_forward_hook(hidden_hook))
        for index, layer in enumerate(mlp_layers):
            def mlp_hook(module, arguments, index=index):
                reduce("mlp", index, arguments[0])
            hooks.append(layers[layer].mlp.down_proj.register_forward_pre_hook(mlp_hook))
        model.eval()
        with torch.inference_mode():
            tokens = torch.tensor(ids, dtype=torch.long, device=text_model.embed_tokens.weight.device)
            output = text_model(input_ids=tokens, use_cache=False, output_attentions=False)
            if len(layers) in hidden_layers:
                reduce("hidden", hidden_layers.index(len(layers)), output.last_hidden_state)
        return result
    finally:
        for hook in hooks:
            hook.remove()


def capture(model, tokenizer, request, destination):
    import torch

    text_model, head = _components(model)
    layers = text_model.layers
    full_attention = tuple(i for i, layer in enumerate(layers) if hasattr(layer, "self_attn"))
    ids, ret = tokenize_request(
        tokenizer, request, num_layers=len(layers), full_attention_layers=full_attention
    )
    vocab_size = head.weight.shape[0]
    if ret.get("topk", 0) > vocab_size:
        raise ValueError("topk exceeds the model vocabulary")
    if any(type(token) is not int or not 0 <= token < vocab_size for row in ids for token in row):
        raise ValueError("Tokenizer produced an invalid token id")
    n, length = len(ids), max(map(len, ids))
    hidden_size = text_model.embed_tokens.weight.shape[1]
    specs = {"input_ids": ((n, length), np.int64)}
    if ret.get("token_logprobs"):
        specs["token_logprobs"] = ((n, max(0, length - 1)), np.float32)
    if ret.get("topk"):
        specs["topk_logprobs"] = ((n, length, ret["topk"]), np.float32)
        specs["topk_ids"] = ((n, length, ret["topk"]), np.int64)
    for layer in ret.get("hidden_states", []):
        specs[f"hidden_{layer}"] = ((n, length, hidden_size), np.float16)
    for layer in ret.get("mlp_act", []):
        intermediate_size = layers[layer].mlp.down_proj.weight.shape[1]
        specs[f"mlp_act_{layer}"] = ((n, length, intermediate_size), np.float16)
    for layer in ret.get("attn_layers", []):
        attention = layers[layer].self_attn
        config = getattr(attention, "config", getattr(text_model, "config", None))
        heads = getattr(config, "num_attention_heads", getattr(attention, "num_heads", None))
        if heads is None:
            raise ValueError("Cannot determine the model's number of attention heads")
        specs[f"attn_{layer}"] = ((n, heads, length, length), np.float16)
    byte_count = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in specs.values())
    if byte_count > 6 * 1024**3:
        raise ValueError("Teacher-forced outputs exceed 6 GiB")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # A failed retry must not leave an earlier successful manifest for partial arrays.
    (destination / "meta.json").unlink(missing_ok=True)
    arrays = {
        name: np.lib.format.open_memmap(destination / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in specs.items()
    }
    for array in arrays.values():
        array[:] = 0
    model.eval()
    hooks, current = [], {"row": 0, "length": 0}

    def save_sequence(name, value):
        expected = (1, current["length"], arrays[name].shape[-1])
        if tuple(value.shape) != expected:
            raise RuntimeError(f"Unexpected {name} shape {tuple(value.shape)}; expected {expected}")
        arrays[name][current["row"], : current["length"]] = value[0].detach().float().cpu().numpy()

    def save_hidden(name):
        def hook(module, arguments, result):
            save_sequence(name, result[0] if isinstance(result, tuple) else result)

        return hook

    means = []
    try:
        for layer in ret.get("hidden_states", []):
            # HF hidden_states[-1] is AFTER final normalization, unlike the last
            # decoder block's output. Save it from last_hidden_state below.
            if layer == len(layers):
                continue
            module = text_model.embed_tokens if layer == 0 else layers[layer - 1]
            hooks.append(module.register_forward_hook(save_hidden(f"hidden_{layer}")))
        for layer in ret.get("mlp_act", []):

            def pre_hook(module, arguments, name=f"mlp_act_{layer}"):
                save_sequence(name, arguments[0])

            hooks.append(layers[layer].mlp.down_proj.register_forward_pre_hook(pre_hook))
        for layer in ret.get("attn_layers", []):

            def attention_hook(module, arguments, result, name=f"attn_{layer}"):
                value = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                if value is None:
                    raise RuntimeError("Attention probabilities unavailable; eager attention required")
                size = current["length"]
                expected = (1, arrays[name].shape[1], size, size)
                if tuple(value.shape) != expected:
                    raise RuntimeError(f"Unexpected {name} shape {tuple(value.shape)}; expected {expected}")
                arrays[name][current["row"], :, :size, :size] = value[0].detach().float().cpu().numpy()

            hooks.append(layers[layer].self_attn.register_forward_hook(attention_hook))
        with torch.inference_mode():
            for i, row in enumerate(ids):
                current.update(row=i, length=len(row))
                arrays["input_ids"][i, : len(row)] = row
                tokens = torch.tensor([row], dtype=torch.long, device=text_model.embed_tokens.weight.device)
                result = text_model(
                    input_ids=tokens, use_cache=False, output_attentions=bool(ret.get("attn_layers"))
                )
                if len(layers) in ret.get("hidden_states", []):
                    save_sequence(f"hidden_{len(layers)}", result.last_hidden_state)
                # Do not materialize a 32768 x vocabulary logits tensor.
                score_sum, score_n = 0.0, 0
                for start in range(0, len(row), 256):
                    end = min(len(row), start + 256)
                    logp = head(result.last_hidden_state[:, start:end]).float().log_softmax(-1)[0]
                    if ret.get("topk"):
                        values, indices = logp.topk(ret["topk"], dim=-1)
                        arrays["topk_logprobs"][i, start:end] = values.cpu().numpy()
                        arrays["topk_ids"][i, start:end] = indices.cpu().numpy()
                    stop = min(end, len(row) - 1)
                    if stop > start:
                        chosen = (
                            logp[: stop - start].gather(1, tokens[0, start + 1 : stop + 1, None]).squeeze(1)
                        )
                        score_sum += chosen.sum().item()
                        score_n += chosen.numel()
                        if ret.get("token_logprobs"):
                            arrays["token_logprobs"][i, start:stop] = chosen.cpu().numpy()
                means.append(score_sum / score_n if score_n else None)
    finally:
        for hook in hooks:
            hook.remove()
        for array in arrays.values():
            array.flush()
    meta = {
        "ok": True,
        "type": "teacher_forced",
        "lengths": list(map(len, ids)),
        "files": {name: list(array.shape) for name, array in arrays.items()},
        "mean_token_logprob": means,
        "bytes": byte_count,
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
