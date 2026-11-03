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


def test_submission_reloads_external_backup_copies_and_returns_snapshot(weights, tmp_path):
    import shutil

    store = WeightStore(weights, tmp_path / "edited")
    name = "model.layers.0.mlp.down_proj.weight"
    value = weights[name].clone()
    value[0, 0] += 3
    store.save_edit(name, value)
    shutil.copytree(tmp_path / "edited", tmp_path / "backup")
    later = value.clone()
    later[1, 0] += 5
    np.save(tmp_path / "edited" / f"{name}.npy", later.numpy())
    snapshot = store.validate_submission()
    assert torch.equal(snapshot[name], later)
    assert torch.equal(store.load(name), later)
    shutil.copyfile(tmp_path / "backup" / f"{name}.npy", tmp_path / "edited" / f"{name}.npy")
    assert torch.equal(store.validate_submission()[name], value)
    assert torch.equal(snapshot[name], later)
    (tmp_path / "edited" / f"{name}.npy").unlink()
    assert store.validate_submission() == {}
    assert torch.equal(store.load(name), weights[name])


@pytest.mark.parametrize(
    "kind", ["frozen", "unknown", "shape", "nan", "inf", "complex", "rescale", "bf16-overflow"]
)
def test_external_submission_refusals_block_forward_checkpoint(weights, tmp_path, kind):
    store = WeightStore(weights, tmp_path / "edited")
    name = "model.layers.0.mlp.down_proj.weight"
    array = weights[name].numpy().copy()
    array[0, 0] += 1
    if kind == "frozen":
        name, array = "model.norm.weight", np.array([2.0, 1.0], dtype=np.float32)
    elif kind == "unknown":
        name = "not.a.model.tensor"
    elif kind == "shape":
        array = np.zeros((5,), dtype=np.float32)
    elif kind == "nan":
        array[0, 0] = float("nan")
    elif kind == "inf":
        array[0, 0] = float("inf")
    elif kind == "complex":
        array = array.astype(np.complex64) + 1j
    elif kind == "rescale":
        array = weights[name].numpy() * 2
    elif kind == "bf16-overflow":
        array[0, 0] = np.finfo(np.float32).max
    np.save(tmp_path / "edited" / f"{name}.npy", array)
    destination = tmp_path / "checkpoint.safetensors"
    destination.write_bytes(b"previous valid checkpoint remains")
    with pytest.raises(EditRefused):
        store.validate_submission()
    with pytest.raises(EditRefused):
        store.export_checkpoint(destination)
    assert destination.read_bytes() == b"previous valid checkpoint remains"


def test_submission_rejects_malformed_archive_and_symlink(weights, tmp_path):
    store = WeightStore(weights, tmp_path / "edited")
    name = "model.layers.0.mlp.down_proj.weight"
    path = tmp_path / "edited" / f"{name}.npy"
    with path.open("wb") as file:
        np.savez(file, unrelated=np.zeros((2, 2)))
    with pytest.raises(EditRefused, match="one numpy array"):
        store.validate_submission()
    path.unlink()
    target = tmp_path / "backup.npy"
    np.save(target, np.array([[7.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    path.symlink_to(target)
    with pytest.raises(EditRefused, match="regular"):
        store.validate_submission()


def test_safe_checkpoint_cannot_overwrite_original(weights, tmp_path):
    model = tmp_path / "original.safetensors"
    save_file(weights, model)
    original_bytes = model.read_bytes()
    store = WeightStore(model, tmp_path / "edited")
    with pytest.raises(EditRefused, match="original"):
        store.export_checkpoint(model)
    assert model.read_bytes() == original_bytes


def test_snapshot_keeps_one_validated_edit_view_after_external_changes(weights, tmp_path):
    name = "model.layers.0.mlp.down_proj.weight"
    store = WeightStore(weights, tmp_path / "edited")
    edited = store.load_original(name)
    edited[0, 0] = 7
    store.save_edit(name, edited)
    snapshot = store.snapshot()
    later = edited.clone()
    later[0, 1] = 9
    np.save(tmp_path / "edited" / f"{name}.npy", later.numpy())
    assert torch.equal(snapshot.load(name), edited)
    assert torch.equal(store.load(name), later)
    snapshot.drop_edit(name)
    assert torch.equal(store.load(name), later)
    assert torch.equal(snapshot.load_original(name), weights[name])


def test_apply_unknown_existing_submission_refuses_before_replacing_edits(weights, tmp_path):
    from openwsp.edits import export_edit

    name = "model.layers.0.mlp.down_proj.weight"
    store = WeightStore(weights, tmp_path / "edited")
    current = store.load_original(name)
    current[0, 0] = 7
    store.save_edit(name, current)
    replacement = current.clone()
    replacement[0, 1] = 9
    export_edit(weights, {name: replacement}, tmp_path / "replacement")
    np.save(tmp_path / "edited/unknown.npy", np.ones(2, dtype=np.float32))
    with pytest.raises(EditRefused, match="Unknown"):
        store.apply_edit(tmp_path / "replacement")
    assert torch.equal(store.load(name), current)
