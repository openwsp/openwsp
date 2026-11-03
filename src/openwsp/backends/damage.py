"""Bounded teacher-forced KL for already-loaded causal text models.

The caller supplies one shared tokenization and vocabulary for both models. Each
model must expose its normalized decoder hidden state and exact output projection
as ``model[.language_model]`` and ``lm_head`` (the pinned Qwen text wrappers do).
Architectures with additional logit transformations need an equivalent projection
adapter; this module must not silently omit them. No logits or input text leave
the function, and no generation, sampling, training, or provider I/O occurs.
"""
from __future__ import annotations

import itertools
import math

import torch


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parts(model):
    backbone = getattr(model, "model", model)
    backbone = getattr(backbone, "language_model", backbone)
    head = getattr(model, "lm_head", None)
    embedding = getattr(backbone, "embed_tokens", None)
    if (not callable(backbone) or not callable(head) or embedding is None
            or not isinstance(getattr(head, "weight", None), torch.Tensor)
            or not isinstance(getattr(embedding, "weight", None), torch.Tensor)
            or head.weight.ndim != 2 or embedding.weight.ndim != 2
            or not head.weight.is_floating_point() or not embedding.weight.is_floating_point()):
        raise ValueError("Model needs a text decoder, input embedding, and exact vocabulary projection")
    if head.weight.device != embedding.weight.device:
        raise ValueError("This backend requires each model on one device")
    for config in (getattr(model, "config", None), getattr(backbone, "config", None)):
        if config is not None and (
            getattr(config, "final_logit_softcapping", None) not in (None, 0)
            or any(getattr(config, name, None) not in (None, 1.0) for name in ("logits_scaling", "logit_scale"))
        ):
            raise ValueError("Model has a logit transform; provide an exact projection adapter")
    return backbone, head, embedding


def _mask(value, length, name, *, allow_binary=False):
    result = torch.as_tensor(value, device="cpu")
    if result.ndim != 1 or len(result) != length:
        raise ValueError(f"{name} must match the input token positions")
    if result.dtype != torch.bool and (
        not allow_binary or result.dtype.is_floating_point or result.dtype.is_complex
        or not bool(((result == 0) | (result == 1)).all())
    ):
        raise ValueError(f"{name} must be a boolean mask")
    return result.to(torch.bool)


def _prepare(examples, vocabulary_sizes, max_sequence_tokens, max_examples):
    records = list(itertools.islice(iter(examples), max_examples + 1))
    if len(records) > max_examples:
        raise ValueError("Too many reference sequences; no model was run")
    prepared = []
    for record in records:
        if not isinstance(record, dict) or "input_ids" not in record or "score_mask" not in record:
            raise ValueError("Each reference needs input_ids and an explicit token-aligned score_mask")
        tokens = torch.as_tensor(record["input_ids"], device="cpu")
        if (tokens.ndim != 1 or tokens.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or not 1 <= len(tokens) <= max_sequence_tokens):
            raise ValueError("Reference token IDs must be a nonempty integer sequence within the context limit")
        tokens = tokens.to(torch.long)
        if bool((tokens < 0).any()) or bool((tokens >= min(vocabulary_sizes)).any()):
            raise ValueError("Reference token IDs are outside the shared model vocabulary")
        attention = _mask(record.get("attention_mask", [True] * len(tokens)), len(tokens),
                          "attention_mask", allow_binary=True)
        score = _mask(record["score_mask"], len(tokens), "score_mask")
        # Logits at t predict token t+1, never the token at their own position.
        selected = score[1:] & attention[1:] & attention[:-1]
        prepared.append((tokens, attention, selected))
    return prepared


def _hidden(backbone, embedding, tokens, attention):
    device = embedding.weight.device
    ids = tokens.to(device).unsqueeze(0)
    mask = attention.to(device).unsqueeze(0)
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    result = backbone(input_ids=ids, attention_mask=mask.long(), position_ids=positions,
                      use_cache=False, output_attentions=False, return_dict=True)
    hidden = getattr(result, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[:2] != ids.shape:
        raise ValueError("Decoder did not return aligned normalized hidden states")
    return hidden[0]


def teacher_forced_kl(base_model, edited_model, examples, *, chunk_tokens=32,
                      max_sequence_tokens=34816, max_examples=512):
    """Return token-weighted KL(base || edited), in nats, over reference tokens.

    ``examples`` contains 1-D ``input_ids``, optional binary ``attention_mask``,
    and a required boolean ``score_mask`` aligned with *input token positions*.
    To score only reference continuations, mark prompt positions false. A target
    token is scored only when its own attention bit and its predecessor's bit
    are true. The first token and padding never contribute.

    Both full contexts run sequentially, one example at a time. Only selected
    prediction positions reach the vocabulary projection, in bounded token
    chunks. Softmax/KL arithmetic and the token-weighted total use float64.
    Inputs are preflighted before either model runs; nothing is silently cut.
    The caller must verify that base and edited token-to-vocabulary mappings are
    identical, not merely the same size, and must fingerprint that tokenization.
    """
    chunk_tokens = _positive_integer(chunk_tokens, "chunk_tokens")
    max_sequence_tokens = _positive_integer(max_sequence_tokens, "max_sequence_tokens")
    max_examples = _positive_integer(max_examples, "max_examples")
    base, base_head, base_embedding = _parts(base_model)
    edited, edited_head, edited_embedding = _parts(edited_model)
    if base_head.weight.shape[0] != edited_head.weight.shape[0]:
        raise ValueError("Base and edited projections must cover the same full vocabulary")
    if base_head.weight.device != edited_head.weight.device:
        raise ValueError("Base and edited models must share one device")
    vocabulary = base_head.weight.shape[0]
    prepared = _prepare(examples, (vocabulary, base_embedding.weight.shape[0], edited_embedding.weight.shape[0]),
                        max_sequence_tokens, max_examples)
    base_model.eval()
    edited_model.eval()
    totals, token_count, scored_sequences = [], 0, 0
    with torch.inference_mode():
        for tokens, attention, selected in prepared:
            positions = selected.nonzero(as_tuple=False).flatten()
            if not len(positions):
                continue
            base_hidden = _hidden(base, base_embedding, tokens, attention)
            edited_hidden = _hidden(edited, edited_embedding, tokens, attention)
            chunk_totals = []
            for start in range(0, len(positions), chunk_tokens):
                indices = positions[start:start + chunk_tokens].to(base_hidden.device)
                base_logits = base_head(base_hidden[indices])
                edited_logits = edited_head(edited_hidden[indices])
                expected = (len(indices), vocabulary)
                if (not isinstance(base_logits, torch.Tensor) or not isinstance(edited_logits, torch.Tensor)
                        or base_logits.shape != expected or edited_logits.shape != expected
                        or not base_logits.is_floating_point() or not edited_logits.is_floating_point()):
                    raise ValueError("Projection did not return full-vocabulary logits for each selected position")
                base_logits, edited_logits = base_logits.to(torch.float64), edited_logits.to(torch.float64)
                if not bool(torch.isfinite(base_logits).all()) or not bool(torch.isfinite(edited_logits).all()):
                    raise ValueError("Selected reference logits must be finite")
                logp = base_logits.log_softmax(-1)
                logq = edited_logits.log_softmax(-1)
                per_token = (logp.exp() * (logp - logq)).sum(-1, dtype=torch.float64)
                if not bool(torch.isfinite(per_token).all()) or bool((per_token < -1e-12).any()):
                    raise ValueError("Reference KL is nonfinite or negative beyond float64 roundoff")
                # Match the independent reducer: only tiny negative roundoff is clipped.
                value = float(per_token.clamp_min(0).sum(dtype=torch.float64).item())
                if not math.isfinite(value):
                    raise ValueError("Reference KL sum is nonfinite")
                chunk_totals.append(value)
                del base_logits, edited_logits, logp, logq, per_token
            totals.append(math.fsum(chunk_totals))
            token_count += len(positions)
            scored_sequences += 1
            del base_hidden, edited_hidden
    total = math.fsum(totals)
    return {"token_count": token_count, "kl_sum": total, "kl_mean": total / token_count if token_count else None,
            "sequence_count": len(prepared), "scored_sequence_count": scored_sequences,
            "input_token_count": sum(int(attention.sum()) for _, attention, _ in prepared),
            "chunk_tokens": chunk_tokens}
