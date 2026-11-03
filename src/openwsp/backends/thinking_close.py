"""vLLM 0.33 adapter for the preregistered generated-token close policies.

Import this module only inside the GPU runtime: the CPU package does not depend
on vLLM. The two-argument request callback receives generated tokens only.
"""
from __future__ import annotations

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

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
        # AdapterLogitsProcessor recognizes this two-argument signature and
        # binds only the mutable generated-token list, excluding prompt IDs.
        if len(output_ids) < self.state.generated_tokens:
            raise ValueError("Thinking-close request history moved backwards")
        self.state.observe_many(output_ids[self.state.generated_tokens:])
        if self.state.reason_to_close:
            token = self.state.close_token_id
            if token >= logits.shape[-1]:
                raise ValueError("Thinking-close token is outside the model vocabulary")
            logits.fill_(float("-inf"))
            logits[token] = 0.0
        return logits


class ThinkingCloseLogitsProcessor(AdapterLogitsProcessor):
    """vLLM's adapter owns batch moves; each request owns independent state."""

    @classmethod
    def validate_params(cls, sampling_params):
        _settings(sampling_params)

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        settings = _settings(params)
        return None if settings is None else RequestThinkingClose(settings, params.max_tokens)
