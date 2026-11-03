"""Native Hugging Face generation over already-loaded causal models.

No model loading, prompt rendering, provider calls, or experiment policy belongs
here. Install the replication extra for the pinned Transformers implementation.
"""

from __future__ import annotations

import math
import threading

import torch

_RNG_LOCK = threading.RLock()


class TokenPenalties:
    """Repetition over real context, then presence over generated tokens only."""

    def __init__(self, attention_mask, *, repetition_penalty=1.0, presence_penalty=0.0):
        self.attention_mask = attention_mask
        self.prompt_width = attention_mask.shape[1]
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_width :]
        if self.repetition_penalty != 1.0:
            seen = torch.zeros_like(scores, dtype=torch.int64)
            seen.scatter_add_(1, input_ids[:, : self.prompt_width], self.attention_mask.to(scores.device))
            seen.scatter_(1, generated, 1)
            penalized = torch.where(
                scores < 0, scores * self.repetition_penalty, scores / self.repetition_penalty
            )
            scores = torch.where(seen.bool(), penalized, scores)
        if self.presence_penalty and generated.shape[1]:
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, generated, True)
            scores = scores - seen.to(scores.dtype) * self.presence_penalty
        return scores


def _finite(value, name, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its supported range")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _special(model, tokenizer, name):
    for source in (getattr(model, "generation_config", None), getattr(model, "config", None), tokenizer):
        value = getattr(source, name, None)
        if value is not None:
            return value
    return None


def generate(
    model,
    tokenizer,
    rendered_prompts,
    *,
    temperature,
    top_p,
    top_k,
    max_tokens,
    seed,
    n=1,
    repetition_penalty=1.0,
    presence_penalty=0.0,
):
    """Generate one completion per rendered prompt, preserving their batch order.

    Prompts already contain any chat template/BOS, so tokenization adds no special
    tokens. Inputs are left padded without changing the tokenizer. EOS counts as
    a generated token; padding after EOS does not. Decoding uses the tokenizer's
    normal ``skip_special_tokens=True`` behavior, with no extra text stripping.

    ``top_k=-1`` or ``0`` disables top-k. Temperature zero selects greedy decoding
    (top-k/top-p are then inactive). Only decoder-only CPU/CUDA models and n=1
    are supported. The seed is reproducible for the same model, prompts, batch
    layout, device and library versions; CPU/selected-CUDA RNG state is restored.
    """
    import transformers
    from transformers import LogitsProcessorList

    if type(n) is not int or n != 1:
        raise ValueError("Only n=1 is supported")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(top_k) is not int or top_k < -1:
        raise ValueError("top_k must be -1, 0, or a positive integer")
    temperature = _finite(temperature, "temperature", minimum=0)
    top_p = _finite(top_p, "top_p", positive=True, maximum=1)
    repetition_penalty = _finite(repetition_penalty, "repetition_penalty", positive=True)
    presence_penalty = _finite(presence_penalty, "presence_penalty", minimum=-2, maximum=2)
    if not isinstance(rendered_prompts, (list, tuple)) or not rendered_prompts:
        raise ValueError("rendered_prompts must be a nonempty list of strings")
    if any(not isinstance(prompt, str) for prompt in rendered_prompts):
        raise ValueError("rendered_prompts must contain only strings")
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("Only decoder-only models are supported")
    device = model.get_input_embeddings().weight.device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA input devices are supported")
    # A model split across CUDA devices would consume RNG states outside this
    # call's restoration scope; require one device rather than promise isolation.
    if any(parameter.device != device for parameter in model.parameters()):
        raise ValueError("The model must be loaded on one device")

    eos = _special(model, tokenizer, "eos_token_id")
    eos_ids = [] if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if any(type(token) is not int or token < 0 for token in eos_ids):
        raise ValueError("EOS IDs must be nonnegative integers")
    eos_ids = list(dict.fromkeys(eos_ids))
    pad = _special(model, tokenizer, "pad_token_id")
    if pad is None and eos_ids:
        pad = eos_ids[0]
    if type(pad) is not int or pad < 0:
        raise ValueError("A pad token or EOS token is required for batching")
    rows = tokenizer(
        list(rendered_prompts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    if len(rows) != len(rendered_prompts) or any(not row for row in rows):
        raise ValueError("Every rendered prompt must encode to at least one token")
    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, -len(row) :] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[index, -len(row) :] = 1

    # Explicit kwargs override the model's generation_config *after* HF merges
    # defaults. A fresh GenerationConfig alone is insufficient in Transformers
    # 5.17: its None fields inherit model settings, including forced tokens.
    options = {
        "max_new_tokens": max_tokens,
        "min_new_tokens": None,
        "min_length": 0,
        "max_time": None,
        "stop_strings": None,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": top_p if temperature > 0 else 1.0,
        "top_k": max(0, top_k) if temperature > 0 else 0,
        "min_p": None,
        "top_h": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "early_stopping": False,
        "diversity_penalty": 0.0,
        "length_penalty": 1.0,
        "penalty_alpha": None,
        "dola_layers": None,
        "constraints": None,
        "force_words_ids": None,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "bad_words_ids": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "exponential_decay_length_penalty": None,
        "sequence_bias": None,
        "guidance_scale": 1.0,
        "watermarking_config": None,
        "token_healing": False,
        "renormalize_logits": False,
        "remove_invalid_values": False,
        "use_cache": True,
        "is_assistant": False,
        "use_mtp": False,
        "prompt_lookup_num_tokens": None,
        "assistant_early_exit": None,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "output_logits": False,
        "return_dict_in_generate": False,
        "pad_token_id": pad,
        "eos_token_id": eos_ids or None,
        "bos_token_id": _special(model, tokenizer, "bos_token_id"),
    }
    processors = LogitsProcessorList(
        [
            TokenPenalties(
                attention_mask,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
            )
        ]
    )
    cuda_devices = [device.index] if device.type == "cuda" else []
    with _RNG_LOCK, torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            torch.cuda.default_generators[device.index].manual_seed(seed)
        training_states = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            sequences = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, logits_processor=processors, **options
            )
        finally:
            # A caller can keep selected children in eval mode while the root
            # is training. Recursive model.train(previous) would erase that.
            for module, training in training_states:
                module.training = training
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != len(rows):
        raise ValueError("generate() must return one token sequence per prompt")
    if not torch.equal(sequences[:, :width], input_ids):
        raise ValueError("generate() did not preserve the input prefix")
    tails = sequences[:, width:].tolist()
    outputs = []
    for tokens in tails:
        if not 1 <= len(tokens) <= max_tokens:
            raise ValueError("generate() returned an invalid completion length")
        stop = next((index for index, token in enumerate(tokens) if token in eos_ids), None)
        if stop is None and len(tokens) != max_tokens:
            raise ValueError("generate() stopped before max_tokens without EOS")
        tokens = tokens if stop is None else tokens[: stop + 1]
        outputs.append(
            {
                "text": tokenizer.decode(tokens, skip_special_tokens=True),
                "token_count": len(tokens),
                "finish_reason": "length" if stop is None else "stop",
            }
        )
    return {
        "outputs": outputs,
        "engine": "transformers",
        "transformers_version": transformers.__version__,
        "eos_token_ids": eos_ids,
        "pad_token_id": pad,
        "sampling": {
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "seed": seed,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        },
    }
