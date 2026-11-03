"""Tiny CPU model checks; no pretrained model, provider, or GPU calls."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from openwsp.backends.damage import teacher_forced_kl
from replications.exp8.damage import kl_from_logits, reduce_kl_chunks


class Decoder(nn.Module):
    def __init__(self, vocab=7, width=3):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width, dtype=torch.float64)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.arange(vocab * width).reshape(vocab, width) / 19)
        self.calls = 0
        self.poison_padding = False

    def forward(self, input_ids, attention_mask, position_ids, *, use_cache, output_attentions, return_dict):
        assert not torch.is_grad_enabled() and not self.training
        assert use_cache is False and output_attentions is False and return_dict is True
        self.calls += 1
        values = self.embed_tokens(input_ids) * attention_mask[..., None]
        hidden = values + values.cumsum(1) * 0.2 + position_ids[..., None] * 0.01
        if self.poison_padding:
            hidden = hidden.masked_fill(~attention_mask.bool()[..., None], torch.nan)
        return SimpleNamespace(last_hidden_state=hidden)


class Head(nn.Linear):
    def __init__(self, width=3, vocab=7):
        super().__init__(width, vocab, bias=False, dtype=torch.float64)
        self.rows = []
        with torch.no_grad():
            self.weight.copy_(torch.arange(vocab * width).reshape(vocab, width).sin())

    def forward(self, values):
        self.rows.append(values.shape[-2])
        return super().forward(values)


class Model(nn.Module):
    def __init__(self, *, edited=False, multimodal=False, vocab=7):
        super().__init__()
        decoder = Decoder(vocab)
        self.model = nn.Module() if multimodal else decoder
        if multimodal:
            self.model.language_model = decoder
        self.lm_head = Head(vocab=vocab)
        if edited:
            with torch.no_grad():
                self.lm_head.weight[2] += torch.tensor([0.7, -0.4, 1.2])

    @property
    def decoder(self):
        return getattr(self.model, "language_model", self.model)

    def forward(self, *args, **kwargs):
        raise AssertionError("Do not materialize sequence-by-vocabulary logits through the full model")


def record(ids, score=None, attention=None):
    return {"input_ids": ids, "score_mask": score if score is not None else [True] * len(ids),
            "attention_mask": attention if attention is not None else [1] * len(ids)}


def full_logits(model, item):
    tokens = torch.tensor([item["input_ids"]])
    mask = torch.tensor([item["attention_mask"]])
    model.eval()
    with torch.inference_mode():
        states = model.decoder(tokens, mask, (mask.cumsum(-1) - 1).clamp_min(0),
                               use_cache=False, output_attentions=False, return_dict=True).last_hidden_state
        return model.lm_head(states)[0].numpy()


def oracle(base, edited, examples):
    chunks = []
    for item in examples:
        mask = np.asarray(item["score_mask"], dtype=bool)[1:]
        attention = np.asarray(item["attention_mask"], dtype=bool)
        mask &= attention[1:] & attention[:-1]
        chunks.append((full_logits(base, item)[:-1], full_logits(edited, item)[:-1], mask))
    return reduce_kl_chunks(chunks)


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("chunk_tokens", [1, 2, 32])
def test_chunked_teacher_force_matches_independent_numpy_full_vocabulary_kl(multimodal, chunk_tokens):
    base, edited = Model(multimodal=multimodal), Model(edited=True, multimodal=multimodal)
    examples = [record([1, 2, 3, 4], [False, False, True, True]), record([2, 1, 5])]
    expected = oracle(base, edited, examples)
    base.lm_head.rows.clear()
    edited.lm_head.rows.clear()
    actual = teacher_forced_kl(base, edited, examples, chunk_tokens=chunk_tokens)
    assert actual["token_count"] == expected["token_count"] == 4
    assert actual["kl_sum"] == pytest.approx(expected["kl_sum"], rel=1e-12, abs=1e-12)
    assert actual["kl_mean"] == pytest.approx(expected["kl_mean"], rel=1e-12, abs=1e-12)
    assert actual["sequence_count"] == 2
    assert max(base.lm_head.rows + edited.lm_head.rows) <= chunk_tokens
    assert all(parameter.grad is None for model in (base, edited) for parameter in model.parameters())


def test_mask_scores_the_preceding_prediction_position_and_excludes_prompt():
    base, edited = Model(), Model(edited=True)
    item = record([1, 2, 3, 4], [False, False, True, False])
    expected = kl_from_logits(full_logits(base, item)[1:2], full_logits(edited, item)[1:2])
    wrong_unshifted = kl_from_logits(full_logits(base, item)[2:3], full_logits(edited, item)[2:3])
    actual = teacher_forced_kl(base, edited, [item])
    assert actual["token_count"] == 1
    assert actual["kl_mean"] == pytest.approx(expected["kl_mean"], abs=1e-12)
    assert abs(actual["kl_mean"] - wrong_unshifted["kl_mean"]) > 1e-4


def test_padding_is_not_scored_and_nonfinite_unselected_padding_is_ignored():
    base, edited = Model(), Model(edited=True)
    plain = record([1, 2, 3], [False, True, True])
    expected = teacher_forced_kl(base, edited, [plain])
    base.decoder.poison_padding = edited.decoder.poison_padding = True
    padded = record([0, 0, 1, 2, 3, 0], [True] * 6, [0, 0, 1, 1, 1, 0])
    actual = teacher_forced_kl(base, edited, [padded])
    assert actual["token_count"] == expected["token_count"] == 2
    assert actual["kl_mean"] == pytest.approx(expected["kl_mean"], abs=1e-12)
    assert actual["input_token_count"] == 3


def test_result_is_token_weighted_not_an_average_of_sequence_means():
    base, edited = Model(), Model(edited=True)
    examples = [record([1, 2]), record([5, 4, 3, 2, 1])]
    separate = [teacher_forced_kl(base, edited, [item]) for item in examples]
    actual = teacher_forced_kl(base, edited, examples)
    expected = sum(item["kl_sum"] for item in separate) / sum(item["token_count"] for item in separate)
    assert actual["token_count"] == 5 and actual["kl_mean"] == pytest.approx(expected, abs=1e-12)
    assert abs(actual["kl_mean"] - np.mean([item["kl_mean"] for item in separate])) > 1e-4


def test_kl_direction_and_distribution_metric_not_target_cross_entropy():
    base, edited = Model(), Model(edited=True)
    forward = teacher_forced_kl(base, edited, [record([1, 2, 3])])
    reverse = teacher_forced_kl(edited, base, [record([1, 2, 3])])
    changed_last_target = teacher_forced_kl(base, edited, [record([1, 2, 6])])
    assert abs(forward["kl_sum"] - reverse["kl_sum"]) > 1e-4
    assert changed_last_target["kl_sum"] == forward["kl_sum"]
    assert teacher_forced_kl(base, base, [record([1, 2, 3])])["kl_sum"] == 0


def test_nearly_equal_distributions_clip_only_tiny_per_token_roundoff():
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn((3, 7), generator=generator, dtype=torch.float64)
    changed = logits + torch.randn((3, 7), generator=generator, dtype=torch.float64) * 1e-9
    base, edited = Model(), Model()
    base.lm_head.forward = lambda hidden: logits
    edited.lm_head.forward = lambda hidden: changed
    expected = kl_from_logits(logits.numpy(), changed.numpy())
    actual = teacher_forced_kl(base, edited, [record([1, 2, 3, 4])])
    assert 0 <= actual["kl_sum"] < 1e-12
    assert actual["kl_sum"] == pytest.approx(expected["kl_sum"], abs=1e-14)


@pytest.mark.parametrize("projection", [None, torch.ones((1, 6)), torch.ones((1, 7), dtype=torch.long)])
def test_projection_must_return_real_full_vocabulary_logits(projection):
    base, edited = Model(), Model()
    edited.lm_head.forward = lambda hidden: projection
    with pytest.raises(ValueError, match="full-vocabulary logits"):
        teacher_forced_kl(base, edited, [record([1, 2])])


def test_empty_selection_does_not_run_models_or_invent_a_mean():
    base, edited = Model(), Model(edited=True)
    actual = teacher_forced_kl(base, edited, [record([1]), record([1, 2], [False, False])])
    assert actual["token_count"] == 0 and actual["kl_sum"] == 0 and actual["kl_mean"] is None
    assert base.decoder.calls == edited.decoder.calls == 0


@pytest.mark.parametrize("bad", [
    {"input_ids": [1, 2]}, record([1, 2], [False]), record([1, 2], [0, 1]),
    record([1, 99]), record([1.0, 2.0]), record([1, 2], attention=[1, 2]),
    record([1, 2, 3, 4, 5]),
])
def test_invalid_later_example_fails_preflight_before_any_model_work(bad):
    base, edited = Model(), Model(edited=True)
    with pytest.raises(ValueError):
        teacher_forced_kl(base, edited, [record([1, 2]), bad], max_sequence_tokens=4)
    assert base.decoder.calls == edited.decoder.calls == 0


def test_sequence_bound_and_vocabulary_mismatch_fail_before_forward():
    base, edited = Model(), Model(edited=True)
    with pytest.raises(ValueError, match="Too many"):
        teacher_forced_kl(base, edited, [record([1, 2])] * 2, max_examples=1)
    with pytest.raises(ValueError, match="same full vocabulary"):
        teacher_forced_kl(base, Model(vocab=8), [record([1, 2])])
    assert base.decoder.calls == 0


def test_nonfinite_selected_logits_and_unsupported_logit_transform_are_rejected():
    base, edited = Model(), Model(edited=True)
    with torch.no_grad():
        edited.lm_head.weight[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        teacher_forced_kl(base, edited, [record([1, 2])])
    edited = Model()
    edited.config = SimpleNamespace(final_logit_softcapping=1.0)
    with pytest.raises(ValueError, match="logit transform"):
        teacher_forced_kl(base, edited, [record([1, 2])])
