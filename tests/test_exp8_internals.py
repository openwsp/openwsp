from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from replications.exp8.internals import capture, generation_summary, tokenize_request


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [int(token) for token in text.split()]

    def apply_chat_template(self, chat, **options):
        assert options == {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
        return " ".join(message["content"] for message in chat)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2)
        self.supply_probabilities = True

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        heads = hidden.reshape(batch, length, 2, 2).transpose(1, 2)
        scores = heads @ heads.transpose(-1, -2) / 2**0.5
        causal_mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        probabilities = scores.masked_fill(causal_mask, -torch.inf).softmax(-1)
        output = (probabilities @ heads).transpose(1, 2).reshape(batch, length, 4)
        return output, probabilities if self.supply_probabilities else None


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, hidden):
        self.activation = torch.nn.functional.silu(self.up_proj(hidden))
        return self.down_proj(self.activation)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()

    def forward(self, hidden):
        attended, self.attention = self.self_attn(hidden)
        hidden = hidden + attended
        return hidden + self.mlp(hidden)


class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(12, 4)
        self.layers = nn.ModuleList([Layer(), Layer()])
        self.norm = nn.LayerNorm(4)

    def forward(self, input_ids, *, use_cache, output_attentions):
        assert use_cache is False
        assert not torch.is_grad_enabled()
        hidden = self.embed_tokens(input_ids)
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        self.pre_norm = hidden
        states[-1] = self.norm(hidden)
        return SimpleNamespace(
            last_hidden_state=states[-1],
            hidden_states=states,
            attentions=[layer.attention for layer in self.layers],
        )


class Model(nn.Module):
    def __init__(self, multimodal):
        super().__init__()
        self.model = nn.Module() if multimodal else TextModel()
        if multimodal:
            self.model.language_model = TextModel()
        self.lm_head = nn.Linear(4, 12, bias=False)

    @property
    def text_model(self):
        return getattr(self.model, "language_model", self.model)


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("prompt_length", [2, 5])
def test_generation_summary_exactly_matches_saved_arrays(tmp_path, multimodal, prompt_length):
    torch.manual_seed(31)
    model = Model(multimodal)
    text = "1 2 3 4 5"
    capture(model, Tokenizer(), {"texts": [text], "return": {
        "hidden_states": [0, 1, 2], "mlp_act": [0, 1]}}, tmp_path)

    def no_logits(*args):
        raise AssertionError("Activation summaries must not calculate unrequested vocabulary logits")

    hook = model.lm_head.register_forward_pre_hook(no_logits)
    actual = generation_summary(model, Tokenizer(), text, prompt_length, [0, 1, 2], [0, 1])
    hook.remove()
    for key, layers in (("hidden", [0, 1, 2]), ("mlp", [0, 1])):
        for index, layer in enumerate(layers):
            name = f"hidden_{layer}" if key == "hidden" else f"mlp_act_{layer}"
            values = np.load(tmp_path / f"{name}.npy")[0]
            np.testing.assert_array_equal(actual[f"{key}_last_prompt.npy"][index], values[prompt_length - 1])
            expected = values[prompt_length:].astype(np.float32).mean(0) if prompt_length < len(values) else np.zeros(values.shape[-1])
            np.testing.assert_array_equal(actual[f"{key}_mean_answer.npy"][index], expected)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("multimodal", [False, True])
def test_capture_matches_model_states_logits_mlp_and_attention(tmp_path, multimodal):
    torch.manual_seed(22)
    model = Model(multimodal)
    request = {
        "texts": ["1 2 3 4", "3"],
        "return": {
            "token_logprobs": True,
            "topk": 3,
            "hidden_states": [0, 1, 2],
            "mlp_act": [0, 1],
            "attn_layers": [0, 1],
        },
    }
    meta = capture(model, Tokenizer(), request, tmp_path)
    arrays = {name: np.load(tmp_path / f"{name}.npy") for name in meta["files"]}
    assert meta["lengths"] == [4, 1]
    assert meta["bytes"] == sum(array.nbytes for array in arrays.values())
    assert arrays["hidden_2"].shape == (2, 4, 4)
    assert arrays["mlp_act_0"].shape == (2, 4, 6)
    assert arrays["attn_1"].shape == (2, 2, 4, 4)
    assert arrays["hidden_2"].dtype == np.float16
    assert arrays["token_logprobs"].dtype == np.float32
    with torch.inference_mode():
        tokens = torch.tensor([[1, 2, 3, 4]])
        expected = model.text_model(tokens, use_cache=False, output_attentions=True)
        logp = model.lm_head(expected.last_hidden_state)[0].log_softmax(-1)
        chosen = logp[:-1].gather(1, tokens[0, 1:, None]).squeeze(1)
        values, indices = logp.topk(3, dim=-1)
    for index, state in enumerate(expected.hidden_states):
        np.testing.assert_array_equal(arrays[f"hidden_{index}"][0], state[0].numpy().astype(np.float16))
    assert not np.allclose(arrays["hidden_2"][0], model.text_model.pre_norm[0].numpy(), atol=1e-3)
    for index, layer in enumerate(model.text_model.layers):
        np.testing.assert_array_equal(
            arrays[f"mlp_act_{index}"][0], layer.mlp.activation[0].numpy().astype(np.float16)
        )
        np.testing.assert_array_equal(
            arrays[f"attn_{index}"][0], expected.attentions[index][0].numpy().astype(np.float16)
        )
    np.testing.assert_allclose(arrays["token_logprobs"][0], chosen.numpy(), atol=1e-6)
    np.testing.assert_allclose(arrays["topk_logprobs"][0], values.numpy(), atol=1e-6)
    np.testing.assert_array_equal(arrays["topk_ids"][0], indices.numpy())
    assert meta["mean_token_logprob"][0] == pytest.approx(chosen.mean().item())
    assert meta["mean_token_logprob"][1] is None
    assert not arrays["token_logprobs"][1].any()
    assert not arrays["hidden_2"][1, 1:].any()
    assert not arrays["attn_1"][1, :, 1:].any()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training


def test_next_token_scoring_across_logit_chunk_boundary(tmp_path):
    torch.manual_seed(2)
    model = Model(False)
    row = [index % 12 for index in range(259)]
    request = {"texts": [" ".join(map(str, row))], "return": {"token_logprobs": True}}
    meta = capture(model, Tokenizer(), request, tmp_path)
    with torch.inference_mode():
        tokens = torch.tensor([row])
        output = model.text_model(tokens, use_cache=False, output_attentions=False)
        logp = model.lm_head(output.last_hidden_state)[0].log_softmax(-1)
        expected = logp[:-1].gather(1, tokens[0, 1:, None]).squeeze(1)
    actual = np.load(tmp_path / "token_logprobs.npy")[0]
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6)
    assert meta["mean_token_logprob"][0] == pytest.approx(expected.mean().item())


def test_failed_capture_removes_hooks_and_does_not_publish_success(tmp_path):
    model = Model(False)
    model.text_model.layers[1].self_attn.supply_probabilities = False
    (tmp_path / "meta.json").write_text('{"ok": true}')
    with pytest.raises(RuntimeError, match="eager attention required"):
        capture(
            model,
            Tokenizer(),
            {
                "texts": ["1 2"],
                "return": {
                    "hidden_states": [0, 1, 2],
                    "mlp_act": [0],
                    "attn_layers": [1],
                },
            },
            tmp_path,
        )
    assert not (tmp_path / "meta.json").exists()
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": "1 2"},
        {"texts": None},
        {"texts": [1]},
        {"texts": []},
        {"chats": ["1 2"]},
        {"texts": ["1"], "chats": []},
        {"texts": ["1"], "return": {"hidden_states": [True]}},
        {"texts": ["1"], "return": {"hidden_states": [1, 1]}},
        {"texts": ["1"], "return": {"hidden_states": [25]}},
        {"texts": ["1"], "return": {"mlp_act": 3}},
        {"texts": ["1"], "return": {"attn_layers": [0]}},
        {"texts": ["1"], "return": {"topk": True}},
        {"texts": ["1"], "return": {"topk": 51}},
        {"texts": ["1"], "return": {"token_logprobs": "yes"}},
        {"texts": ["1"], "enable_thinking": "yes"},
    ],
)
def test_request_schema_rejects_ambiguous_or_invalid_values(payload):
    with pytest.raises(ValueError):
        tokenize_request(Tokenizer(), payload)


def test_chat_template_and_single_token_output(tmp_path):
    meta = capture(
        Model(False),
        Tokenizer(),
        {
            "chats": [[{"role": "user", "content": "1"}]],
            "return": {"token_logprobs": True},
        },
        tmp_path,
    )
    assert meta["mean_token_logprob"] == [None]
    assert np.load(tmp_path / "token_logprobs.npy").shape == (1, 0)


def test_real_tiny_qwen_hidden_states_match_hf_final_norm(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"] * 2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
    )
    config._attn_implementation = "eager"
    torch.manual_seed(19)
    model = transformers.Qwen3_5ForCausalLM(config).eval()
    capture(
        model,
        Tokenizer(),
        {
            "texts": ["1 2 3"],
            "return": {
                "hidden_states": [0, 1, 2],
                "attn_layers": [0, 1],
                "mlp_act": [1],
            },
        },
        tmp_path,
    )
    with torch.inference_mode():
        expected = model.model(
            torch.tensor([[1, 2, 3]]), use_cache=False, output_hidden_states=True, output_attentions=True
        )
    for layer, hidden in enumerate(expected.hidden_states):
        np.testing.assert_array_equal(
            np.load(tmp_path / f"hidden_{layer}.npy")[0], hidden[0].float().numpy().astype(np.float16)
        )
    for layer, attention in enumerate(expected.attentions):
        np.testing.assert_array_equal(
            np.load(tmp_path / f"attn_{layer}.npy")[0], attention[0].float().numpy().astype(np.float16)
        )
