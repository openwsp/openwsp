"""CPU-only sampling contracts; no pretrained checkpoints or provider clients."""

from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from openwsp.backends.hf_generation import TokenPenalties, generate


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = 1

    def __init__(self):
        self.calls = []
        self.decoded = []
        self.padding_side = "right"

    def __call__(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return {"input_ids": [[int(token) for token in prompt.split()] for prompt in prompts]}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        self.decoded.append(list(ids))
        words = {5: "<think>", 6: "reasoning", 7: "</think>", 8: "answer"}
        return " ".join(words.get(token, str(token)) for token in ids if token not in {0, 9})


class ScriptedModel(torch.nn.Module):
    def __init__(self, tails):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.generation_config = transformers.GenerationConfig(eos_token_id=[0, 9], pad_token_id=0)
        self.tails = tails
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        return torch.cat((kwargs["input_ids"], torch.tensor(self.tails)), dim=1)


def run(model, tokenizer, prompts, **kwargs):
    return generate(
        model,
        tokenizer,
        prompts,
        **{
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 4,
            "max_tokens": 4,
            "seed": 41,
            **kwargs,
        },
    )


def test_penalties_ignore_padding_and_presence_counts_generated_ids_once():
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    processor = TokenPenalties(mask, repetition_penalty=2.0, presence_penalty=1.5)
    ids = torch.tensor([[0, 2, 3, 3, 3, 4], [0, 2, 2, 5, 5, 5]])
    scores = torch.tensor([[8.0, 7.0, 6.0, -4.0, 5.0, 9.0], [8.0, 7.0, 6.0, -4.0, 5.0, 9.0]])
    actual = processor(ids, scores)
    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [8.0, 7.0, 3.0, -9.5, 1.0, 9.0],
                [4.0, 7.0, 3.0, -4.0, 5.0, 3.0],
            ]
        ),
    )
    # EOS padding is ignored in row 0, while real EOS inside row 1's prompt
    # participates in repetition. Presence does not touch either prompt alone.
    first_step = TokenPenalties(mask, presence_penalty=1.5)(ids[:, :3], scores)
    torch.testing.assert_close(first_step, scores)
    assert scores[0, 3] == -4  # The caller's logits are not mutated.


def test_eos_counts_left_padding_and_normal_thinking_decode():
    model = ScriptedModel([[9, 0, 0, 0], [5, 6, 7, 8], [2, 4, 6, 9], [2, 0, 0, 0]])
    model.embedding.eval()
    tokenizer = Tokenizer()
    result = run(model, tokenizer, ["0 2", "3 4 5", "2", "3 4"])
    assert result["outputs"] == [
        {"text": "", "token_count": 1, "finish_reason": "stop"},
        {"text": "<think> reasoning </think> answer", "token_count": 4, "finish_reason": "length"},
        {"text": "2 4 reasoning", "token_count": 4, "finish_reason": "stop"},
        {"text": "2", "token_count": 2, "finish_reason": "stop"},
    ]
    assert tokenizer.decoded == [[9], [5, 6, 7, 8], [2, 4, 6, 9], [2, 0]]
    call = model.calls[0]
    assert call["input_ids"].tolist() == [[0, 0, 2], [3, 4, 5], [0, 0, 2], [0, 3, 4]]
    assert call["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1]]
    assert tokenizer.calls[0][1] == {
        "add_special_tokens": False,
        "padding": False,
        "truncation": False,
        "return_attention_mask": False,
    }
    assert tokenizer.padding_side == "right"
    assert model.training  # Restore the caller's previous mode.
    assert not model.embedding.training
    assert result["eos_token_ids"] == [0, 9]


def tiny_model():
    config = transformers.GPT2Config(
        vocab_size=16,
        n_positions=64,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    return transformers.GPT2LMHeadModel(config)


def test_actual_transformers_overrides_loaded_config_and_orders_penalties_before_warpers(monkeypatch):
    model = tiny_model()
    model.generation_config = transformers.GenerationConfig(
        eos_token_id=[0, 9],
        pad_token_id=0,
        do_sample=True,
        num_beams=4,
        num_return_sequences=3,
        temperature=1.9,
        top_k=2,
        top_p=0.1,
        min_p=0.9,
        top_h=0.1,
        typical_p=0.1,
        epsilon_cutoff=0.2,
        eta_cutoff=0.2,
        min_new_tokens=20,
        min_length=30,
        stop_strings=["unexpected"],
        repetition_penalty=9.0,
        no_repeat_ngram_size=2,
        forced_bos_token_id=8,
        forced_eos_token_id=9,
        suppress_tokens=[4],
        bad_words_ids=[[5]],
        sequence_bias={(7,): -10.0},
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )
    captured = {}
    original = model._get_logits_processor

    def record(**kwargs):
        captured["config"] = kwargs["generation_config"]
        captured["processors"] = original(**kwargs)
        return captured["processors"]

    monkeypatch.setattr(model, "_get_logits_processor", record)
    result = run(model, Tokenizer(), ["1 2 3", "4"], repetition_penalty=1.1, presence_penalty=1.5)
    config = captured["config"]
    assert config.num_beams == config.num_return_sequences == 1
    assert config.temperature == 0.6 and config.top_p == 0.95 and config.top_k == 4
    assert config.min_new_tokens is None
    assert config.min_length == config.no_repeat_ngram_size == 0
    assert config.repetition_penalty == 1  # Explicit custom processor handles it once.
    for field in (
        "min_p",
        "top_h",
        "stop_strings",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "suppress_tokens",
        "bad_words_ids",
        "sequence_bias",
    ):
        assert getattr(config, field) is None
    assert config.typical_p == 1 and config.epsilon_cutoff == config.eta_cutoff == 0
    assert not config.output_scores and not config.output_logits and not config.return_dict_in_generate
    assert [type(item).__name__ for item in captured["processors"]] == [
        "TokenPenalties",
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
    ]
    assert len(result["outputs"]) == 2
    assert model.generation_config.num_beams == 4  # No caller configuration mutation.


def test_native_sampling_seed_is_stable_and_rng_state_restored():
    model, tokenizer = tiny_model(), Tokenizer()
    tokenizer.eos_token_id = None
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)  # Uniform logits make seeds observable.
    before = torch.random.get_rng_state().clone()
    first = run(model, tokenizer, ["1 2", "3"], top_k=-1, top_p=1, max_tokens=8, seed=10)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    repeated = run(model, tokenizer, ["1 2", "3"], top_k=-1, top_p=1, max_tokens=8, seed=10)
    different = run(model, tokenizer, ["1 2", "3"], top_k=-1, top_p=1, max_tokens=8, seed=11)
    assert first["outputs"] == repeated["outputs"]
    assert first["outputs"] != different["outputs"]
    assert all(row["token_count"] == 8 and row["finish_reason"] == "length" for row in first["outputs"])


def test_tiny_qwen_conditional_sdpa_with_left_padding_and_cached_decode():
    text_config = transformers.Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=31,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
    )
    vision_config = transformers.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=1,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config = transformers.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=26,
        vision_end_token_id=27,
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = transformers.Qwen3_5ForConditionalGeneration(config).eval()
    calls = []

    def record(module, args, kwargs):
        calls.append(
            {
                "ids": kwargs["input_ids"].clone(),
                "mask": kwargs["attention_mask"].clone(),
                "cache": kwargs.get("past_key_values") is not None,
                "inference": torch.is_inference_mode_enabled(),
            }
        )

    hook = model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        result = run(model, Tokenizer(), ["1 2 3", "4"], repetition_penalty=1.1, presence_penalty=1.5)
    finally:
        hook.remove()
    assert model.config.text_config._attn_implementation == "sdpa"
    assert next(model.parameters()).device.type == "cpu" and not model.training
    assert calls[0]["ids"].tolist() == [[1, 2, 3], [0, 0, 4]]
    assert calls[0]["mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert len(calls) > 1
    assert all(call["inference"] and call["cache"] for call in calls)
    assert all(call["ids"].shape == (2, 1) for call in calls[1:])
    assert len(result["outputs"]) == 2 and result["eos_token_ids"] == [31]
    assert all(1 <= item["token_count"] <= 4 for item in result["outputs"])


def test_temperature_zero_and_disabled_top_k_propagate_explicitly():
    model = ScriptedModel([[2, 3, 4, 5]])
    run(model, Tokenizer(), ["1"], temperature=0, top_k=-1, repetition_penalty=1.2, presence_penalty=-0.5)
    call = model.calls[0]
    assert call["do_sample"] is False
    assert call["temperature"] == call["top_p"] == 1
    assert call["top_k"] == 0 and call["max_new_tokens"] == 4
    assert call["num_beams"] == call["num_return_sequences"] == call["repetition_penalty"] == 1
    processor = call["logits_processor"][0]
    assert processor.repetition_penalty == 1.2 and processor.presence_penalty == -0.5


def test_generate_failure_restores_rng_and_training_mode(monkeypatch):
    model = ScriptedModel([[0]])
    model.embedding.eval()
    before = torch.random.get_rng_state().clone()

    def fail(**kwargs):
        assert not model.training
        torch.rand(10)
        raise RuntimeError("generation failed")

    monkeypatch.setattr(model, "generate", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        run(model, Tokenizer(), ["1"])
    assert model.training
    assert not model.embedding.training
    torch.testing.assert_close(torch.random.get_rng_state(), before)


@pytest.mark.parametrize(
    "change",
    [
        {"n": 2},
        {"n": True},
        {"temperature": float("nan")},
        {"temperature": -1},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_k": -2},
        {"top_k": True},
        {"max_tokens": 0},
        {"seed": -1},
        {"seed": True},
        {"repetition_penalty": 0},
        {"presence_penalty": 3},
    ],
)
def test_unsupported_sampling_is_rejected_before_generate(change):
    model = ScriptedModel([[0]])
    with pytest.raises(ValueError):
        run(model, Tokenizer(), ["1"], **change)
    assert not model.calls


def test_unsupported_extra_parameter_and_short_non_eos_output_are_rejected():
    model = ScriptedModel([[2]])
    with pytest.raises(TypeError):
        run(model, Tokenizer(), ["1"], frequency_penalty=1)
    with pytest.raises(ValueError, match="without EOS"):
        run(model, Tokenizer(), ["1"])
