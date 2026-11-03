"""Exercise the deployed Generator body on CPU without importing Modal or CUDA."""
import ast
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from replications.exp8.eval_grid import SEED_SCHEME, problem_seed


@pytest.fixture
def generator(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src/openwsp/backends/modal_app.py"
    node = next(item for item in ast.parse(source.read_text()).body
                if isinstance(item, ast.ClassDef) and item.name == "Generator")
    calls = {"loads": [], "generate": [], "templates": [], "encodes": [], "commits": []}
    outputs = []

    class Tokenizer:
        prompt_tokens = 3
        close_ids = (9,)

        def apply_chat_template(self, messages, **kwargs):
            calls["templates"].append((messages, kwargs))
            return "rendered:" + messages[0]["content"]

        def encode(self, text, **kwargs):
            calls["encodes"].append((text, kwargs))
            return self.close_ids if text == "</think>" else list(range(self.prompt_tokens))

    tokenizer = Tokenizer()

    def generate(prompts, params, **kwargs):
        calls["generate"].append((prompts, params, kwargs))
        return outputs

    engine = SimpleNamespace(generate=generate)

    def load(**kwargs):
        calls["loads"].append(kwargs)
        return engine

    vllm = ModuleType("vllm")
    vllm.LLM = load
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(options=kwargs)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda checkpoint: tokenizer)
    processor = ModuleType("openwsp.backends.thinking_close")
    processor.ThinkingCloseLogitsProcessor = object()
    for module in (vllm, transformers, processor):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = {
        "app": SimpleNamespace(cls=lambda **kwargs: lambda cls: cls),
        "modal": SimpleNamespace(parameter=lambda **kwargs: kwargs["default"],
                                 enter=lambda: lambda fn: fn, method=lambda: lambda fn: fn),
        "gpu_image": object(), "data_volume": object(), "time": time,
        "kernel_cache_volume": SimpleNamespace(commit=lambda: calls["commits"].append(True)),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    instance = namespace["Generator"]()
    instance.load()
    assert instance.load_error is None
    return SimpleNamespace(instance=instance, tokenizer=tokenizer, outputs=outputs, calls=calls,
                           processor=processor.ThinkingCloseLogitsProcessor)


def sample(tokens, *, finish="stop"):
    return SimpleNamespace(token_ids=tokens, text="decoded text", finish_reason=finish)


def test_loader_uses_explicit_context_sequence_cap_and_registered_processor(generator):
    fixture = generator
    assert fixture.calls["loads"][0]["max_model_len"] == 34816
    assert fixture.calls["loads"][0]["max_num_seqs"] == 128
    fixture.instance.max_model_len = 83968
    fixture.instance.max_num_seqs = 32
    fixture.instance._load()
    assert fixture.calls["loads"][-1]["max_model_len"] == 83968
    assert fixture.calls["loads"][-1]["max_num_seqs"] == 32
    assert fixture.calls["loads"][-1]["logits_processors"] == [fixture.processor]
    assert fixture.calls["loads"][-1]["enable_prefix_caching"] is False


def test_grid_sampling_seeds_and_per_sample_forced_metadata_are_preserved(generator):
    fixture = generator
    fixture.outputs.extend([SimpleNamespace(outputs=[sample([1, 2, 1, 9, 4], finish="length"),
                                                    sample([2, 9, 4])]),
                           SimpleNamespace(outputs=[sample([9, 4]), sample([9, 5])])])
    rows = [{"id": "problem-1", "problem": "1+1"}, {"id": "problem-2", "prompt": "Custom prompt"}]
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                "max_tokens": 5, "n": 2, "seed": 0}
    policy = {"kind": "thinking_token_cap", "thinking_tokens": 3}
    result = fixture.instance.generate(rows, sampling, True, True, seed_scheme=SEED_SCHEME, decoding_policy=policy)
    prompts, params, options = fixture.calls["generate"][0]
    assert options == {"use_tqdm": False}
    assert prompts[1] == "rendered:Custom prompt"
    assert all(call[1]["enable_thinking"] is True for call in fixture.calls["templates"])
    for row, param in zip(rows, params, strict=True):
        assert param.options == {**sampling, "seed": problem_seed(row["id"]),
                                 "extra_args": {"openwsp_thinking_close": {"policy": policy, "close_token_id": 9}}}
    forced, natural = result["outputs"][0]["samples"]
    assert forced["forced_close"] is True and forced["forced_close_token_index"] == 3
    assert forced["token_count"] == 5 and forced["finish_reason"] == "length"
    assert natural["forced_close"] is False and natural["token_count"] == 3
    assert natural["finish_reason"] == "stop"
    assert sampling["seed"] == 0 and "extra_args" not in sampling


def test_thinking_off_has_no_close_policy_and_keeps_sampling(generator):
    fixture = generator
    fixture.outputs.append(SimpleNamespace(outputs=[sample([1, 2])]))
    sampling = {"temperature": 0.6, "presence_penalty": 1.5, "max_tokens": 32768, "n": 1, "seed": 0}
    result = fixture.instance.generate([{"id": "a", "prompt": "answer"}], sampling, False)
    assert fixture.calls["generate"][0][1].options == sampling
    assert fixture.calls["templates"][0][1]["enable_thinking"] is False
    assert "forced_close" not in result["outputs"][0]["samples"][0]


def test_long_context_retains_full_output_allowance_at_exact_boundary(generator):
    fixture = generator
    fixture.instance.max_model_len = 83968
    fixture.instance.max_num_seqs = 32
    fixture.tokenizer.prompt_tokens = 2048
    fixture.outputs.append(SimpleNamespace(outputs=[sample([1])]))
    result = fixture.instance.generate([{"id": "a", "prompt": "answer"}], {"max_tokens": 81920})
    assert fixture.calls["generate"][0][1].options["max_tokens"] == 81920
    assert result["engine_settings"]["max_model_len"] == 83968
    assert result["engine_settings"]["max_num_seqs"] == 32
    fixture.tokenizer.prompt_tokens = 2049
    with pytest.raises(ValueError, match="no output cap was shortened"):
        fixture.instance.generate([{"id": "a", "prompt": "answer"}], {"max_tokens": 81920})
    assert len(fixture.calls["generate"]) == 1


@pytest.mark.parametrize("change,match", [("thinking", "thinking enabled"),
    ("close_token", "single-token"), ("seed", "seed scheme")])
def test_invalid_grid_requests_fail_before_generation(generator, change, match):
    fixture = generator
    options = {"decoding_policy": {"kind": "thinking_token_cap", "thinking_tokens": 3}}
    if change == "thinking":
        options["thinking"] = False
    elif change == "close_token":
        fixture.tokenizer.close_ids = [9, 10]
    else:
        options["seed_scheme"] = "different"
    with pytest.raises(ValueError, match=match):
        fixture.instance.generate([{"id": "a", "prompt": "answer"}], {"max_tokens": 5}, **options)
    assert fixture.calls["generate"] == []


def test_output_audit_detects_decoder_that_did_not_enforce_policy(generator):
    fixture = generator
    fixture.outputs.append(SimpleNamespace(outputs=[sample([1, 1, 1, 1])]))
    with pytest.raises(ValueError, match="did not enforce"):
        fixture.instance.generate([{"id": "a", "prompt": "answer"}], {"max_tokens": 5},
                                  decoding_policy={"kind": "thinking_token_cap", "thinking_tokens": 3})
