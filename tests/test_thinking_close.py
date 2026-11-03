import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from replications.exp8.eval_grid import close_metadata


@pytest.fixture
def module(monkeypatch):
    # vLLM itself requires CUDA. Its pinned adapter factory contract is enough
    # to exercise the actual request callback on CPU logits tensors.
    vllm = ModuleType("vllm.v1.sample.logits_processor")
    vllm.AdapterLogitsProcessor = type("AdapterLogitsProcessor", (), {})
    monkeypatch.setitem(sys.modules, vllm.__name__, vllm)
    path = Path(__file__).parents[1] / "src/openwsp/backends/thinking_close.py"
    spec = importlib.util.spec_from_file_location("thinking_close_cpu_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def params(module, *, policy=None, max_tokens=20, close=3):
    return SimpleNamespace(max_tokens=max_tokens, extra_args={module.EXTRA_ARGS_KEY: {
        "policy": policy or {"kind": "thinking_token_cap", "thinking_tokens": 3}, "close_token_id": close}})


def test_vllm_callback_forces_exactly_one_token_without_restarting(module):
    adapter = module.ThinkingCloseLogitsProcessor()
    assert adapter.is_argmax_invariant() is False
    callback = adapter.new_req_logits_processor(params(module))
    assert len(inspect.signature(callback).parameters) == 2  # Generated-only adapter contract.
    output = [1, 1]
    original = torch.arange(5, dtype=torch.float32)
    assert torch.equal(callback(output, original.clone()), original)
    output.append(2)
    forced = callback(output, original.clone())
    assert forced[3] == 0 and torch.isneginf(forced[[0, 1, 2, 4]]).all()
    # Calling twice on the same prefix does not consume the threshold twice.
    assert torch.equal(callback(output, original.clone()), forced)
    output.append(3)
    assert torch.equal(callback(output, original.clone()), original)
    output.extend([1, 1, 1, 1])
    assert torch.equal(callback(output, original.clone()), original)
    assert callback.state.generated_tokens == len(output)
    assert close_metadata(output, {"kind": "thinking_token_cap", "thinking_tokens": 3}, 3, max_tokens=20) == {
        "forced_close": True, "forced_close_reason": "thinking_token_cap",
        "forced_close_token_index": 3, "forced_close_token_count": 1}


def test_n_samples_receive_independent_state_and_no_policy_is_noop(module):
    adapter = module.ThinkingCloseLogitsProcessor()
    shared_params = params(module)
    first = adapter.new_req_logits_processor(shared_params)
    second = adapter.new_req_logits_processor(shared_params)
    original = torch.ones(5)
    assert first is not second and first.state is not second.state
    assert torch.isneginf(first([1, 2, 1], original.clone())[0])
    assert torch.equal(second([2], original.clone()), original)
    assert second.state.generated_tokens == 1 and not second.state.closed
    assert adapter.new_req_logits_processor(SimpleNamespace(extra_args=None)) is None
    assert adapter.new_req_logits_processor(SimpleNamespace(extra_args={"unrelated": True})) is None


def test_repeat_callback_ignores_prompt_and_natural_close(module):
    policy = {"kind": "repeat_ngram", "ngram_size": 8, "repeats": 5, "window_tokens": 2048}
    callback = module.ThinkingCloseLogitsProcessor().new_req_logits_processor(params(module, policy=policy))
    original = torch.arange(5, dtype=torch.float32)
    # An arbitrarily repetitive prompt has no input path to this callback.
    assert torch.equal(callback([], original.clone()), original)
    assert torch.equal(callback([1] * 11, original.clone()), original)
    forced = callback([1] * 12, original.clone())
    assert forced[3] == 0 and torch.isneginf(forced[0])
    tokens = [1] * 12 + [3, 4]
    assert torch.equal(callback(tokens, original.clone()), original)
    assert close_metadata(tokens, policy, 3, max_tokens=20)["forced_close_reason"] == "repeat_ngram"
    natural = module.ThinkingCloseLogitsProcessor().new_req_logits_processor(params(module, policy=policy))
    assert torch.equal(natural([3] + [1] * 12, original.clone()), original)
    assert close_metadata([3] + [1] * 12, policy, 3, max_tokens=20)["forced_close"] is False


def test_total_cap_does_not_create_extra_output_or_false_finished_claim(module):
    callback = module.ThinkingCloseLogitsProcessor().new_req_logits_processor(params(module, max_tokens=3))
    logits = torch.ones(5)
    assert torch.equal(callback([1, 2, 1], logits.clone()), logits)
    assert close_metadata([1, 2, 1], {"kind": "thinking_token_cap", "thinking_tokens": 3}, 3, max_tokens=3) == {
        "forced_close": False, "forced_close_reason": None,
        "forced_close_token_index": None, "forced_close_token_count": 0}


def test_audit_rejects_missing_enforcement_and_truncated_prefix_does_not_invent_close():
    policy = {"kind": "thinking_token_cap", "thinking_tokens": 3}
    with pytest.raises(ValueError, match="did not enforce"):
        close_metadata([1, 1, 1, 1], policy, 3, max_tokens=20)
    assert close_metadata([1, 1, 1], policy, 3, max_tokens=20)["forced_close"] is False


def test_invalid_vllm_options_fail_before_sampling(module):
    cls = module.ThinkingCloseLogitsProcessor
    bad = params(module)
    bad.extra_args[module.EXTRA_ARGS_KEY]["close_token_id"] = True
    with pytest.raises(ValueError, match="close_token_id"):
        cls.validate_params(bad)
    bad = params(module, max_tokens=None)
    with pytest.raises(ValueError, match="output-token cap"):
        cls.validate_params(bad)
    callback = cls().new_req_logits_processor(params(module, close=999))
    with pytest.raises(ValueError, match="vocabulary"):
        callback([1, 2, 1], torch.ones(5))
    callback = cls().new_req_logits_processor(params(module))
    callback([1, 2], torch.ones(5))
    with pytest.raises(ValueError, match="backwards"):
        callback([1], torch.ones(5))
