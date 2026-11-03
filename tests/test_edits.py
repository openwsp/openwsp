import hashlib
import json

import numpy as np
import pytest
import torch

from openwsp.edits import EditError, edit_sha256, export_edit, read_edit, tensor_sha256


def fixture_weights():
    base = {"layer.weight": torch.tensor([[1.0, 0.5, -2.0], [0.0, 4.0, 8.0]])}
    edited = {"layer.weight": base["layer.weight"].clone()}
    edited["layer.weight"][0, 1] += 0.251
    edited["layer.weight"][1, 2] -= 1.25
    return base, edited


def test_roundtrip_records_evaluated_bits_and_does_not_mutate(tmp_path):
    base, edited = fixture_weights()
    snapshot = base["layer.weight"].clone()
    manifest = export_edit(base, edited, tmp_path, base_model="tiny", revision="pinned")
    loaded, tensors = read_edit(tmp_path, base)
    assert loaded == manifest
    assert torch.equal(tensors["layer.weight"], edited["layer.weight"].bfloat16().float())
    assert torch.equal(base["layer.weight"], snapshot)
    assert manifest["changed_entries_total"] == 2
    assert manifest["tensors"][0]["changed_columns"] == 2
    assert manifest["base_model"] == "tiny"
    assert manifest["base_revision"] == "pinned"
    expected = hashlib.sha256(
        edited["layer.weight"].bfloat16().view(torch.uint16).numpy().tobytes()
    ).hexdigest()
    assert tensor_sha256(edited["layer.weight"]) == expected


def test_digest_is_stable_across_order_and_archive_metadata(tmp_path):
    base, edited = fixture_weights()
    base["second"] = torch.tensor([1.0, 2.0])
    edited["second"] = torch.tensor([1.0, 3.0])
    first = export_edit(base, edited, tmp_path / "one")
    second = export_edit(
        dict(reversed(list(base.items()))), dict(reversed(list(edited.items()))), tmp_path / "two"
    )
    assert first["edit_sha256"] == second["edit_sha256"]
    # Changing compression changes bytes, but not the logical content address.
    with np.load(tmp_path / "one" / "edit.npz") as z:
        arrays = {k: z[k] for k in z.files}
    np.savez(tmp_path / "one" / "edit.npz", **arrays)
    loaded, _ = read_edit(tmp_path / "one", base)
    assert loaded["edit_sha256"] == first["edit_sha256"]


def test_base_and_payload_corruption_are_rejected(tmp_path):
    base, edited = fixture_weights()
    export_edit(base, edited, tmp_path)
    wrong_base = {"layer.weight": base["layer.weight"] + 1}
    with pytest.raises(EditError, match="Base tensor sha256"):
        read_edit(tmp_path, wrong_base)
    with np.load(tmp_path / "edit.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["t0_val"][0] ^= 1
    np.savez_compressed(tmp_path / "edit.npz", **arrays)
    with pytest.raises(EditError, match="payload sha256"):
        read_edit(tmp_path, base)


@pytest.mark.parametrize(
    "field,value", [("format_version", 2), ("format_version", True), ("format", "other")]
)
def test_unknown_version_or_format_is_refused(tmp_path, field, value):
    base, edited = fixture_weights()
    manifest = export_edit(base, edited, tmp_path)
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EditError, match="Unsupported edit format"):
        read_edit(tmp_path, base)


def test_legacy_sparse_format_and_edited_hash_check(tmp_path):
    base, edited = fixture_weights()
    manifest = export_edit(base, edited, tmp_path)
    for field in ["format", "format_version", "payload_sha256", "edit_sha256"]:
        manifest.pop(field)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, rebuilt = read_edit(tmp_path, base)
    assert torch.equal(rebuilt["layer.weight"], edited["layer.weight"].bfloat16().float())
    manifest["tensors"][0]["edited_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EditError, match="Edited tensor sha256"):
        read_edit(tmp_path, base)


def test_legacy_lowrank_reconstructs_rank_by_rank_with_fixups(tmp_path):
    base = {"weight": torch.tensor([[1.0, -0.5], [3.0, 4.0]])}
    a = np.array([[0.123456789, 0.987654321], [0.75, -0.125]], dtype=np.float64)
    b = np.array([[0.25, -0.875], [-0.333333333, 0.625]], dtype=np.float64)
    expected = base["weight"].bfloat16().float().numpy().astype(np.float64)
    for i in range(a.shape[1]):
        expected = expected + np.outer(a[:, i], b[i])
    target = torch.from_numpy(expected.astype(np.float32)).bfloat16()
    target[1, 0] = -1.75  # a released fixup wins over reconstructed lowrank bits
    indices = np.array([2], dtype=np.int32)
    values = target.view(torch.uint16).numpy().reshape(-1)[indices]
    np.savez(tmp_path / "edit.npz", t0_a=a, t0_b=b, t0_idx=indices, t0_val=values)
    count = int((base["weight"].bfloat16() != target).sum())
    manifest = {
        "tensors": [
            {
                "name": "weight",
                "dtype": "bfloat16",
                "shape": [2, 2],
                "form": "lowrank",
                "base_sha256": tensor_sha256(base["weight"]),
                "edited_sha256": tensor_sha256(target),
                "changed_entries": count,
            }
        ],
        "changed_tensors": 1,
        "changed_entries_total": count,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, actual = read_edit(tmp_path, base)
    assert torch.equal(actual["weight"], target.float())


@pytest.mark.parametrize("indices", [np.array([-1]), np.array([99]), np.array([0, 0])])
def test_legacy_invalid_indices_are_refused(tmp_path, indices):
    base, edited = fixture_weights()
    manifest = export_edit(base, edited, tmp_path)
    for field in ["format", "format_version", "payload_sha256", "edit_sha256"]:
        manifest.pop(field)
    np.savez(tmp_path / "edit.npz", t0_idx=indices, t0_val=np.ones(len(indices), dtype=np.uint16))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EditError, match="index|indices"):
        read_edit(tmp_path, base)


def test_empty_and_bf16_unchanged_edits_have_valid_identity(tmp_path):
    base = {"w": torch.tensor([1.0])}
    manifest = export_edit(base, {"w": torch.tensor([1.00001])}, tmp_path)
    assert manifest["changed_tensors"] == 0
    assert read_edit(tmp_path, base)[1] == {}
    assert edit_sha256(manifest) == manifest["edit_sha256"]


def test_export_refuses_invalid_values_and_shapes(tmp_path):
    base = {"w": torch.tensor([1.0])}
    with pytest.raises(EditError, match="finite"):
        export_edit(base, {"w": torch.tensor([float("nan")])}, tmp_path)
    with pytest.raises(EditError, match="overflow"):
        export_edit(base, {"w": torch.tensor([torch.finfo(torch.float32).max])}, tmp_path)
    with pytest.raises(EditError, match="Shape mismatch"):
        export_edit(base, {"w": torch.tensor([1.0, 2.0])}, tmp_path)


def test_metadata_tampering_changes_the_address(tmp_path):
    base, edited = fixture_weights()
    manifest = export_edit(base, edited, tmp_path, revision="original")
    manifest["base_revision"] = "altered"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(EditError, match="address sha256"):
        read_edit(tmp_path, base)


def test_pickle_payload_is_never_loaded(tmp_path):
    base, edited = fixture_weights()
    export_edit(base, edited, tmp_path)
    np.savez(tmp_path / "edit.npz", t0_idx=np.array([{"unsafe": "object"}], dtype=object))
    with pytest.raises(EditError, match="allow_pickle=False"):
        read_edit(tmp_path, base)


def test_truncated_zip_is_invalid_edit(tmp_path):
    base, edited = fixture_weights()
    export_edit(base, edited, tmp_path)
    payload = tmp_path / "edit.npz"
    payload.write_bytes(payload.read_bytes()[:30])
    with pytest.raises(EditError, match="Invalid edit"):
        read_edit(tmp_path, base)
