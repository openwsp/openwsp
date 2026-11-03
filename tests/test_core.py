import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from openwsp.core import EditRefused, WeightStore


@pytest.fixture
def weights():
    return {
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "model.norm.weight": torch.ones(2),
        "model.embed_tokens.weight": torch.ones(3, 2),
        "lm_head.weight": torch.ones(3, 2),
    }


def test_reads_and_source_are_immutable(weights, tmp_path):
    store = WeightStore(weights, tmp_path)
    name = "model.layers.0.mlp.down_proj.weight"
    original = store.load_original(name)
    weights[name].zero_()
    store.load_original(name).zero_()
    assert torch.equal(store.load_original(name), original)
    value = original.clone()
    value[0, 0] += 0.1234567
    store.save_edit(name, value)
    store.load(name).zero_()
    assert torch.equal(store.load(name), value)
    assert torch.equal(store.load_original(name), original)
    assert np.load(tmp_path / f"{name}.npy").dtype == np.float32


def test_refusals(weights):
    store = WeightStore(weights)
    name = "model.layers.0.mlp.down_proj.weight"
    for frozen in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"):
        with pytest.raises(EditRefused, match="Frozen"):
            store.save_edit(frozen, weights[frozen] + 1)
    with pytest.raises(EditRefused, match="Unknown"):
        store.save_edit("missing", torch.ones(2))
    with pytest.raises(EditRefused, match="Shape"):
        store.save_edit(name, torch.ones(3))
    with pytest.raises(EditRefused, match="Non-finite"):
        store.save_edit(name, torch.full((2, 2), float("nan")))
    with pytest.raises(EditRefused, match="rescaling"):
        store.save_edit(name, weights[name] * 2)
    with pytest.raises(EditRefused, match="rescaling"):
        store.save_edit(name, weights[name] * 0)
    with pytest.raises(EditRefused, match="gradients"):
        store.save_edit(name, weights[name].clone().requires_grad_())
    assert store.edited() == []


def test_file_backed_restore_and_drop(weights, tmp_path):
    model = tmp_path / "model.safetensors"
    save_file({name: value.bfloat16() for name, value in weights.items()}, model)
    edits = tmp_path / "edited"
    store = WeightStore(model, edits)
    name = "model.layers.0.mlp.down_proj.weight"
    value = store.load(name)
    value[0, 1] = 7.1234
    store.save_edit(name, value)
    resumed = WeightStore(model, edits)
    assert resumed.edited() == [name]
    assert torch.equal(resumed.load(name), value)
    resumed.save_edit(name, resumed.load_original(name))
    assert not resumed.edited()
    assert not resumed.drop_edit(name)


def test_tampered_submission_is_refused(weights, tmp_path):
    store = WeightStore(weights, tmp_path)
    name = "model.layers.0.mlp.down_proj.weight"
    np.save(tmp_path / f"{name}.npy", weights[name].numpy() * 2)
    with pytest.raises(EditRefused, match="rescaling"):
        store.load(name)


def test_apply_validates_all_before_mutating(weights, tmp_path):
    from openwsp.edits import export_edit

    store = WeightStore(weights, tmp_path / "current")
    name = "model.layers.0.mlp.down_proj.weight"
    current = weights[name].clone()
    current[0, 0] += 3
    store.save_edit(name, current)
    replacement = weights[name].clone()
    replacement[1, 1] += 5
    export_edit(
        weights,
        {name: replacement, "model.norm.weight": torch.tensor([3.0, 1.0])},
        tmp_path / "bad-edit",
    )
    with pytest.raises(EditRefused, match="Frozen"):
        store.apply_edit(tmp_path / "bad-edit")
    assert torch.equal(store.load(name), current)
    assert store.edited() == [name]
