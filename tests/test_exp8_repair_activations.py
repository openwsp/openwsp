from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from replications.exp8.repair_activations import _text_hash, extract, input_means, select_examples


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
        return "1 2"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [int(value) for value in text.split()]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(3, 5, bias=False)
        self.down_proj = nn.Linear(5, 3, bias=False)

    def forward(self, hidden):
        self.activation = torch.nn.functional.silu(self.up_proj(hidden))
        return self.down_proj(self.activation)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 3)
        self.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.layers:
            layer.mlp = MLP()

    def forward(self, input_ids, *, use_cache, output_attentions):
        assert not torch.is_grad_enabled() and use_cache is False and output_attentions is False
        self.observed_ids = input_ids.tolist()[0]
        value = self.embed_tokens(input_ids)
        for layer in self.layers:
            value = value + layer.mlp(value)
        return SimpleNamespace(last_hidden_state=value)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = Decoder()

    def forward(self, *args, **kwargs):
        raise AssertionError("Must bypass the LM wrapper and vocabulary projection")


def example(identity="one", text="3 4 5", group="looping"):
    return {
        "id": identity,
        "prompt": "original prompt",
        "text": text,
        "group": group,
        "prompt_sha256": _text_hash("original prompt"),
        "text_sha256": _text_hash(text),
    }


def test_selection_matches_control_rule_without_correctness_or_subsetting():
    looping = "a b c d e f g h " * 180
    raw = [
        {"problem_id": "loop", "finish_reason": "length", "text": looping},
        {"problem_id": "excluded", "finish_reason": "length", "text": "short nonrepetition"},
        {"problem_id": "done", "finish_reason": "stop", "text": "answer"},
    ]
    benchmark = [{"id": key, "problem": key + " question"} for key in ("loop", "excluded", "done")]
    selected = select_examples(raw, benchmark, {"looping": 1, "finished": 1})
    assert [row["id"] for row in selected] == ["loop", "done"]
    assert selected[0]["group"] == "looping" and selected[1]["group"] == "finished"
    with pytest.raises(ValueError, match="differ from saved"):
        select_examples(raw, benchmark, {"looping": 2, "finished": 1})


def test_hook_means_use_answer_inputs_float16_then_numpy_float32_and_no_head():
    torch.manual_seed(78)
    model = Model()
    result, tokens = input_means(model, Tokenizer(), example(), layers=(0, 1), answer_tokens=2)
    decoder = model.model.language_model
    assert decoder.observed_ids == [1, 2, 3, 4]
    assert tokens["prompt_tokens"] == tokens["answer_tokens"] == 2
    for layer in (0, 1):
        expected = (
            decoder.layers[layer].mlp.activation[0, 2:].numpy().astype(np.float16).astype(np.float32).mean(0)
        )
        np.testing.assert_array_equal(result[str(layer)], expected)
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_hooks_are_removed_after_forward_failure():
    model = Model()
    decoder = model.model.language_model

    def fail(*args, **kwargs):
        raise RuntimeError("broken decoder")

    decoder.forward = fail
    with pytest.raises(RuntimeError, match="broken"):
        input_means(model, Tokenizer(), example(), layers=(0,))
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_benchmark_then_resume_reuses_exact_receipts_and_finishes_contrast(tmp_path, monkeypatch):
    import replications.exp8.repair_activations as module

    model = Model()
    rows = [example(), example("two", "6 7", "finished")]
    first = extract(
        model, Tokenizer(), rows, tmp_path, provenance={"fixture": "same"}, limit=1, layers=(0, 1)
    )
    assert first["status"] == "benchmark_only" and first["processed"] == 1
    assert not (tmp_path / "input-differences.npz").exists()
    actual = module.input_means
    calls = []

    def counted(model, tokenizer, row, **kwargs):
        calls.append(row["id"])
        return actual(model, tokenizer, row, **kwargs)

    monkeypatch.setattr(module, "input_means", counted)
    result = extract(model, Tokenizer(), rows, tmp_path, provenance={"fixture": "same"}, layers=(0, 1))
    assert result["status"] == "complete" and result["counts"] == {"looping": 1, "finished": 1}
    assert calls == ["two"]
    with np.load(tmp_path / "input-differences.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in (0, 1)}
    with pytest.raises(ValueError, match="differ from the existing"):
        extract(model, Tokenizer(), rows, tmp_path, provenance={"fixture": "changed"}, layers=(0, 1))


def test_source_or_saved_array_tampering_refuses_resume(tmp_path):
    rows = [example(), example("two", "6", "finished")]
    extract(Model(), Tokenizer(), rows, tmp_path, provenance={}, limit=1, layers=(0,))
    next(tmp_path.glob("examples/*/means.npz")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="arrays changed"):
        extract(Model(), Tokenizer(), rows, tmp_path, provenance={}, layers=(0,))
    rows[0]["text"] = "different text"
    with pytest.raises(ValueError, match="source text changed"):
        extract(Model(), Tokenizer(), rows, tmp_path, provenance={}, layers=(0,))
