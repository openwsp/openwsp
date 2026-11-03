"""vLLM0.33 Model Runner V2 adapter for registered thinking-close policies.

Ordinary faithful generation must not register this optional grid processor.
Only committed generated tokens enter its state machine; requests and prompt
positions are recovered from the actual V2 buffers and per-step row mapping.
"""
from __future__ import annotations

from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

from replications.exp8.eval_grid import ThinkingCloseState, validate_policy

EXTRA_ARGS_KEY = "openwsp_thinking_close"


def _settings(params):
    settings = (params.extra_args or {}).get(EXTRA_ARGS_KEY)
    if settings is None:
        return None
    if (not isinstance(settings, dict) or set(settings) != {"policy", "close_token_id"}
            or type(settings["close_token_id"]) is not int or settings["close_token_id"] < 0):
        raise ValueError("Thinking-close extra_args require policy and a nonnegative integer close_token_id")
    validate_policy(settings["policy"])
    if settings["policy"] is None or type(params.max_tokens) is not int or params.max_tokens < 1:
        raise ValueError("Thinking-close requires a policy and a positive total output-token cap")
    return settings


class RequestThinkingClose:
    """One instance per generated sample, including each member of n>1."""

    def __init__(self, settings, max_tokens):
        self.state = ThinkingCloseState(settings["policy"], settings["close_token_id"], max_tokens=max_tokens)

    def __call__(self, output_ids, logits):
        if len(output_ids) < self.state.generated_tokens:
            raise ValueError("Thinking-close request history moved backwards")
        return self.advance(output_ids[self.state.generated_tokens:], logits)

    def advance(self, new_ids, logits):
        self.state.observe_many(new_ids)
        if self.state.reason_to_close:
            import torch

            token = self.state.close_token_id
            if token >= logits.shape[-1]:
                raise ValueError("Thinking-close token is outside the model vocabulary")
            if not bool(torch.isfinite(logits[token])):
                raise ValueError("Thinking-close token is masked or nonfinite; cannot override another constraint")
            logits.fill_(float("-inf"))
            logits[token] = 0.0
        return logits


class ThinkingCloseLogitsProcessor(LogitsProcessor):
    """V2 persistent-slot state, independent of this step's logits row order.

The registered grid does not use speculative decoding. Reject it explicitly:
draft rows require a rollback-aware state machine and cannot safely consume
the committed-history stream used here. The per-step GPU-to-host copy carries
only row metadata and newly committed output tokens, not a growing full prefix.
"""

    def __init__(self, vllm_config, req_states):
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError("Thinking-close grid adapter does not support speculative decoding")
        self.req_states = req_states
        self.requests = {}

    @classmethod
    def validate_params(cls, sampling_params):
        _settings(sampling_params)

    def add_request(self, req_idx, params):
        if type(req_idx) is not int or not 0 <= req_idx < self.req_states.max_num_reqs:
            raise ValueError("Thinking-close request slot is outside the runtime capacity")
        # Recycled or preempted slots must never inherit the former occupant.
        self.requests.pop(req_idx, None)
        settings = _settings(params)
        if settings is None:
            return False
        if settings["close_token_id"] >= self.req_states.vocab_size:
            raise ValueError("Thinking-close token is outside the model vocabulary")
        self.requests[req_idx] = RequestThinkingClose(settings, params.max_tokens)
        return True

    def apply(self, logits, ctx):
        import torch

        if not self.requests or not any(int(slot) in self.requests for slot in ctx.idx_mapping_np):
            return logits
        slots = ctx.expanded_idx_mapping.long()
        if len(slots) != len(logits):
            raise ValueError("Thinking-close logits rows do not match the V2 context")
        metadata = torch.stack((slots, ctx.pos + 1, ctx.expanded_local_pos,
                                self.req_states.total_len.gpu[slots]), dim=1).cpu().tolist()
        if len({row[0] for row in metadata}) != len(metadata) or any(row[2] != 0 for row in metadata):
            raise ValueError("Thinking-close grid adapter received speculative or duplicate request rows")
        pending, pieces = [], []
        for row, (slot, visible, _, committed) in enumerate(metadata):
            callback = self.requests.get(slot)
            if callback is None:
                continue
            # V2 samples partial-prefill rows too, then discards those tokens.
            # A resumed request's prefill includes already-generated history;
            # consume none of it until the whole (re)fill is visible.
            if visible < int(self.req_states.prefill_len.np[slot]):
                continue
            prompt = int(self.req_states.prompt_len.np[slot])
            output_len = visible - prompt
            if (prompt < 0 or visible > committed or output_len < callback.state.generated_tokens
                    or output_len > callback.state.max_tokens):
                raise ValueError("Thinking-close visible committed history is inconsistent or moved backwards")
            start = prompt + callback.state.generated_tokens
            piece = self.req_states.all_token_ids.gpu[slot, start:visible]
            if len(piece) != output_len - callback.state.generated_tokens:
                raise ValueError("Thinking-close token history exceeds its runtime buffer")
            pieces.append(piece)
            pending.append((row, callback, len(piece)))
        new_tokens = torch.cat(pieces).cpu().tolist() if pieces else []
        offset = 0
        for row, callback, count in pending:
            callback.advance(new_tokens[offset:offset + count], logits[row])
            offset += count
        return logits
