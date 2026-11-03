"""CPU policy checks against vLLM0.33's actual V2 interface and loader."""
import hashlib
import importlib.machinery
import importlib.util
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from replications.exp8.eval_grid import close_metadata

FIXTURES = Path(__file__).parent / "fixtures/vllm033"
PACKAGE = "vllm.v1.worker.gpu.sample.logits_processor"


def load_file(monkeypatch, name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    monkeypatch.setitem(sys.modules, name, module)
    loader.exec_module(module)
    return module


@pytest.fixture
def module(monkeypatch):
    # Exact pinned source; only logging, plugin discovery and CUDA-init guard
    # dependencies are inert. The interface and loader are real upstream code.
    for record in json.loads((FIXTURES / "sources.json").read_text())["files"]:
        assert hashlib.sha256((FIXTURES / record["file"]).read_bytes()).hexdigest() == record["sha256"]
    interface = load_file(monkeypatch, PACKAGE + ".interface", FIXTURES / "interface.py.txt")
    package = ModuleType(PACKAGE)
    package.LogitsProcessor = interface.LogitsProcessor
    monkeypatch.setitem(sys.modules, PACKAGE, package)
    for name, values in {
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.exceptions": {"VLLMValidationError": type("VLLMValidationError", (ValueError,), {})},
        "vllm.utils.torch_utils": {"guard_cuda_initialization": nullcontext},
        "vllm.v1.sample.logits_processor": {"STR_POOLING_REJECTS_LOGITSPROCS": "pooling unsupported"},
    }.items():
        inert = ModuleType(name)
        inert.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, inert)
    loader = load_file(monkeypatch, PACKAGE + ".loader", FIXTURES / "loader.py.txt")
    monkeypatch.setattr(loader, "entry_points", lambda **kwargs: [])
    path = Path(__file__).parents[1] / "src/openwsp/backends/thinking_close.py"
    spec = importlib.util.spec_from_file_location("thinking_close_cpu_test", path)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, result)
    spec.loader.exec_module(result)
    result.upstream = SimpleNamespace(interface=interface, loader=loader)
    return result


def params(module, *, policy=None, max_tokens=20, close=3):
    return SimpleNamespace(max_tokens=max_tokens, extra_args={module.EXTRA_ARGS_KEY: {
        "policy": policy or {"kind": "thinking_token_cap", "thinking_tokens": 3}, "close_token_id": close}})


def state_buffers():
    prompt = np.zeros(4, dtype=np.int32)
    return SimpleNamespace(device=torch.device("cpu"), max_num_reqs=4, vocab_size=5,
                           all_token_ids=SimpleNamespace(gpu=torch.zeros((4, 64), dtype=torch.int32)),
                           prompt_len=SimpleNamespace(np=prompt), prefill_len=SimpleNamespace(np=prompt.copy()),
                           total_len=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32)))


def set_history(buffers, slot, prompt, generated):
    all_tokens = prompt + generated
    buffers.all_token_ids.gpu[slot, :len(all_tokens)] = torch.tensor(all_tokens)
    buffers.prompt_len.np[slot] = len(prompt)
    buffers.prefill_len.np[slot] = len(prompt)
    buffers.total_len.gpu[slot] = len(all_tokens)


def context(module, buffers, slots, *, visible=None):
    lengths = [int(buffers.total_len.gpu[slot]) for slot in slots] if visible is None else visible
    return module.upstream.interface.LogitsContext(
        expanded_idx_mapping=torch.tensor(slots), idx_mapping=torch.tensor(slots), idx_mapping_np=np.array(slots),
        expanded_local_pos=torch.zeros(len(slots), dtype=torch.int32),
        input_ids=torch.tensor([int(buffers.all_token_ids.gpu[slot, length - 1]) for slot, length in zip(slots, lengths, strict=True)]),
        pos=torch.tensor(lengths) - 1, seq_lens_upper_bound_np=np.array(lengths))


def processor(module, buffers=None):
    buffers = buffers or state_buffers()
    instances = module.upstream.loader.build_custom_logits_processors(
        SimpleNamespace(speculative_config=None), buffers, False, [module.ThinkingCloseLogitsProcessor])
    return instances[0], buffers


def test_actual_v2_loader_validates_class_fqcn_constructor_and_rejects_old_base(module):
    loader, cls = module.upstream.loader, module.ThinkingCloseLogitsProcessor
    assert issubclass(cls, module.upstream.interface.LogitsProcessor)
    assert loader._load_v2_logitsprocs_by_fqcns([cls, "thinking_close_cpu_test:ThinkingCloseLogitsProcessor"]) == [cls, cls]
    actual, buffers = processor(module)
    assert actual.req_states.all_token_ids is buffers.all_token_ids and actual.requests == {}
    # Reproduce the exact startup failure missed by the previous fake-base tests.
    old_adapter = type("AdapterLogitsProcessor", (), {})
    old_processor = type("ThinkingCloseLogitsProcessor", (old_adapter,), {})
    with pytest.raises(ValueError, match="not a subclass of vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor"):
        loader._load_v2_logitsprocs_by_fqcns([old_processor])
    validator = loader.build_custom_logits_processors_params_validator([cls])
    validator(params(module))
    with pytest.raises(ValueError, match="output-token cap"):
        validator(params(module, max_tokens=None))


def test_v2_forces_one_token_and_preserves_repeated_prefix_state(module):
    adapter, buffers = processor(module)
    assert adapter.add_request(0, params(module)) is True
    original = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    set_history(buffers, 0, [3, 1], [1, 2])
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), original)
    set_history(buffers, 0, [3, 1], [1, 2, 1])
    forced = adapter.apply(original.clone(), context(module, buffers, [0]))
    assert forced[0, 3] == 0 and torch.isneginf(forced[0, [0, 1, 2, 4]]).all()
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), forced)
    assert adapter.requests[0].state.generated_tokens == 3
    set_history(buffers, 0, [3, 1], [1, 2, 1, 3, 4])
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), original)
    assert adapter.requests[0].state.generated_tokens == 5
    assert close_metadata([1, 2, 1, 3, 4], {"kind": "thinking_token_cap", "thinking_tokens": 3}, 3, max_tokens=20)["forced_close"]


def test_actual_slot_mapping_handles_reordering_n_samples_and_unmodified_rows(module):
    adapter, buffers = processor(module)
    shared = params(module)
    assert adapter.add_request(0, shared) and adapter.add_request(2, shared)
    assert adapter.requests[0].state is not adapter.requests[2].state
    assert adapter.add_request(1, SimpleNamespace(extra_args=None)) is False
    set_history(buffers, 0, [4, 4], [1, 1, 1])
    set_history(buffers, 1, [4], [2, 2, 2, 2])
    set_history(buffers, 2, [4, 4], [2])
    original = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    result = adapter.apply(original.clone(), context(module, buffers, [2, 0, 1]))
    assert torch.equal(result[0], original[0]) and torch.equal(result[2], original[2])
    assert result[1, 3] == 0 and torch.isneginf(result[1, 0])
    set_history(buffers, 2, [4, 4], [2, 2, 2])
    result = adapter.apply(original.clone(), context(module, buffers, [1, 2, 0]))
    assert torch.equal(result[0], original[0]) and torch.isneginf(result[1:, 0]).all()


def test_slot_reuse_clears_policy_and_resume_replays_committed_output_once(module):
    adapter, buffers = processor(module)
    adapter.add_request(0, params(module))
    set_history(buffers, 0, [4], [1, 1, 1])
    adapter.apply(torch.ones((1, 5)), context(module, buffers, [0]))
    assert adapter.add_request(0, SimpleNamespace(extra_args={"unrelated": True})) is False
    assert 0 not in adapter.requests
    original = torch.ones((1, 5))
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), original)
    adapter.add_request(0, params(module))
    set_history(buffers, 0, [4], [3, 1, 1, 1])
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), original)
    assert adapter.requests[0].state.closed and adapter.requests[0].state.generated_tokens == 4


def test_repeat_policy_uses_generated_tokens_only_and_stops_after_natural_close(module):
    policy = {"kind": "repeat_ngram", "ngram_size": 8, "repeats": 5, "window_tokens": 2048}
    adapter, buffers = processor(module)
    adapter.add_request(0, params(module, policy=policy))
    original = torch.ones((1, 5))
    set_history(buffers, 0, [1] * 20, [1] * 11)
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), original)
    set_history(buffers, 0, [1] * 20, [1] * 12)
    forced = adapter.apply(original.clone(), context(module, buffers, [0]))
    assert forced[0, 3] == 0 and torch.isneginf(forced[0, 0])
    adapter.add_request(0, params(module, policy=policy))
    set_history(buffers, 0, [1] * 20, [3] + [1] * 12)
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), original)


def test_visible_prefix_excludes_future_committed_tokens(module):
    adapter, buffers = processor(module)
    adapter.add_request(0, params(module))
    set_history(buffers, 0, [4, 4], [1, 1, 1, 3])
    original = torch.ones((1, 5))
    forced = adapter.apply(original.clone(), context(module, buffers, [0], visible=[5]))
    assert torch.isneginf(forced[0, 0]) and not adapter.requests[0].state.closed
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), original)


def test_chunked_initial_prefill_never_consumes_prompt_or_changes_discarded_samples(module):
    adapter, buffers = processor(module)
    adapter.add_request(0, params(module))
    set_history(buffers, 0, [1, 1, 1, 1, 1, 1], [])
    original = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    for visible in (2, 4, 6):
        result = adapter.apply(original.clone(), context(module, buffers, [0], visible=[visible]))
        assert torch.equal(result, original)
        assert adapter.requests[0].state.generated_tokens == 0
    set_history(buffers, 0, [1] * 6, [1, 1, 1])
    result = adapter.apply(original.clone(), context(module, buffers, [0]))
    assert result[0, 3] == 0 and torch.isneginf(result[0, 0])
    assert adapter.requests[0].state.generated_tokens == 3


def test_preempted_history_refill_does_not_advance_or_force_until_complete(module):
    adapter, buffers = processor(module)
    adapter.add_request(0, params(module))
    set_history(buffers, 0, [4, 4, 4], [1, 1, 1])
    buffers.prefill_len.np[0] = 6  # Prompt plus the resumed generated prefix.
    original = torch.ones((1, 5))
    for visible in (2, 4, 5):
        result = adapter.apply(original.clone(), context(module, buffers, [0], visible=[visible]))
        assert torch.equal(result, original)
        assert adapter.requests[0].state.generated_tokens == 0
        assert adapter.requests[0].state.reason_to_close is None
    result = adapter.apply(original.clone(), context(module, buffers, [0], visible=[6]))
    assert result[0, 3] == 0 and torch.isneginf(result[0, 0])
    assert adapter.requests[0].state.generated_tokens == 3
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), result)


def test_total_cap_cannot_extend_generation(module):
    adapter, buffers = processor(module)
    adapter.add_request(0, params(module, max_tokens=3))
    set_history(buffers, 0, [4], [1, 2, 1])
    original = torch.ones((1, 5))
    assert torch.equal(adapter.apply(original.clone(), context(module, buffers, [0])), original)
    assert close_metadata([1, 2, 1], {"kind": "thinking_token_cap", "thinking_tokens": 3}, 3, max_tokens=3)["forced_close"] is False


def test_speculation_bad_vocab_and_invalid_settings_are_rejected(module):
    with pytest.raises(ValueError, match="speculative"):
        module.ThinkingCloseLogitsProcessor(SimpleNamespace(speculative_config=object()), state_buffers())
    adapter, buffers = processor(module)
    with pytest.raises(ValueError, match="vocabulary"):
        adapter.add_request(0, params(module, close=999))
    with pytest.raises(ValueError, match="close_token_id"):
        adapter.add_request(0, params(module, close=True))
    with pytest.raises(ValueError, match="slot"):
        adapter.add_request(-1, params(module))
    adapter.add_request(0, params(module))
    set_history(buffers, 0, [4], [1])
    with pytest.raises(ValueError, match="duplicate request rows"):
        adapter.apply(torch.ones((2, 5)), context(module, buffers, [0, 0]))
    ctx = context(module, buffers, [0])
    ctx.expanded_local_pos[0] = 1
    with pytest.raises(ValueError, match="speculative"):
        adapter.apply(torch.ones((1, 5)), ctx)


def test_backwards_or_uncommitted_history_cannot_corrupt_policy(module):
    adapter, buffers = processor(module)
    adapter.add_request(0, params(module))
    set_history(buffers, 0, [4], [1, 2])
    adapter.apply(torch.ones((1, 5)), context(module, buffers, [0]))
    with pytest.raises(ValueError, match="backwards"):
        adapter.apply(torch.ones((1, 5)), context(module, buffers, [0], visible=[2]))
    with pytest.raises(ValueError, match="committed history"):
        adapter.apply(torch.ones((1, 5)), context(module, buffers, [0], visible=[4]))


def test_forcing_does_not_reinflate_a_grammar_masked_token(module):
    adapter, buffers = processor(module)
    adapter.add_request(0, params(module))
    set_history(buffers, 0, [4], [1, 1, 1])
    logits = torch.tensor([[0.0, 1.0, 2.0, -torch.inf, 4.0]])
    before = logits.clone()
    with pytest.raises(ValueError, match="masked"):
        adapter.apply(logits, context(module, buffers, [0]))
    assert torch.equal(before, logits)


def test_audit_rejects_missing_enforcement_and_does_not_invent_close():
    policy = {"kind": "thinking_token_cap", "thinking_tokens": 3}
    with pytest.raises(ValueError, match="did not enforce"):
        close_metadata([1, 1, 1, 1], policy, 3, max_tokens=20)
    assert close_metadata([1, 1, 1], policy, 3, max_tokens=20)["forced_close"] is False
