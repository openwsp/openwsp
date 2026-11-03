import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from openwsp.core import EditRefused, WeightStore
from replications.exp8.repair import (
    DOSES,
    VARIANTS,
    build_repair,
    contrastive_mean,
    load_activation_manifest,
    repair_tensor,
)

NAME = "model.layers.0.mlp.down_proj.weight"
BASE = np.array([[2, 2, -3, 1], [1, -1, 2, 3]], dtype=np.float32)
PROVENANCE = {
    "source": "synthetic repetition contrast",
    "activation_reduction": "equal-weight per-example means",
    "text_sets": {
        "looping": [{"id": "l", "text_sha256": "a" * 64}],
        "finished": [{"id": "f", "text_sha256": "b" * 64}],
    },
}


def test_contrastive_means_are_equal_weight_per_example_or_explicit_sums():
    looping = np.array([[4, 1], [2, 3]])
    finished = np.array([[1, 1]])
    mean, counts = contrastive_mean(looping, finished)
    summed, same_counts = contrastive_mean(looping.sum(0), finished.sum(0), counts=counts)
    np.testing.assert_array_equal(mean, [2, 1])
    np.testing.assert_array_equal(summed, mean)
    assert counts == same_counts == {"looping": 2, "finished": 1}
    with pytest.raises(ValueError, match="positive integer"):
        contrastive_mean(looping.sum(0), finished.sum(0), counts={"looping": 0, "finished": 1})


def test_rank_one_repair_uses_signed_scores_and_stable_ties_without_mutating_inputs():
    before = BASE.copy()
    direction = np.array([3.0, 0.0])
    replacement, method = repair_tensor(
        BASE, direction, alpha=6, column_rule="topk", input_difference=[1, 1, 10, 1], top_k=2
    )
    assert method["columns"] == [0, 1]  # -30 has largest magnitude but is last by signed score.
    assert method["selected_scores"] == [2, 2]
    np.testing.assert_array_equal(replacement[0], [-10, -10, -3, 1])
    np.testing.assert_array_equal(replacement[1], BASE[1])
    np.testing.assert_array_equal(BASE, before)
    np.testing.assert_array_equal(direction, [3, 0])
    assert replacement.dtype == np.float32


def test_all_column_alpha_one_removes_only_the_direction_component_and_zero_is_identity():
    direction = np.array([1, 2], dtype=np.float64)
    ablated, method = repair_tensor(BASE, direction, alpha=1, column_rule="all")
    np.testing.assert_allclose(np.asarray(method["direction"]) @ ablated, 0, atol=2e-7)
    zero, _ = repair_tensor(BASE, direction, alpha=0, column_rule="all")
    np.testing.assert_array_equal(zero, BASE)
    assert DOSES == (0, 1, 2, 3, 4, 6, 8)
    assert VARIANTS["2a"]["alpha"] == 1 and VARIANTS["2b"]["alpha"] == 6
    assert VARIANTS["2c"] == {"column_rule": "topk", "alpha": 6, "top_k": 64}


def test_build_exports_real_bf16_v1_and_does_not_change_callers_edits(tmp_path):
    store = WeightStore({NAME: torch.from_numpy(BASE)})
    unrelated = torch.from_numpy(BASE.copy())
    unrelated[1, 0] += 1
    store.save_edit(NAME, unrelated)
    inputs = {
        NAME: {
            "output_looping": [[3, 1], [1, -1]],
            "output_finished": [[0, 0]],
            "input_looping": [[1, 1, 10, 1], [1, 1, 10, 1]],
            "input_finished": [[0, 0, 0, 0]],
        }
    }
    result = build_repair(
        store,
        tmp_path / "built",
        activations=inputs,
        alpha=6,
        column_rule="topk",
        top_k=2,
        provenance={
            **PROVENANCE,
            "text_sets": {
                "looping": [{"id": "l1", "text_sha256": "a" * 64}, {"id": "l2", "text_sha256": "c" * 64}],
                "finished": PROVENANCE["text_sets"]["finished"],
            },
        },
    )
    assert result["evaluation_status"] == "pending" and result["accuracy"] is None
    restored = WeightStore({NAME: torch.from_numpy(BASE)})
    manifest = restored.apply_edit(tmp_path / "built/edit")
    assert manifest["format_version"] == 1 and manifest["edit_sha256"] == result["edit_sha256"]
    np.testing.assert_array_equal(restored.load(NAME)[0], [-10, -10, -3, 1])
    assert torch.equal(store.load(NAME), unrelated)
    method = json.loads((tmp_path / "built/method.json").read_text())
    assert method["layers"][NAME]["counts"] == {"looping": 2, "finished": 1}
    assert method["layers"][NAME]["columns"] == [0, 1]
    assert (
        result["method_sha256"] == hashlib.sha256((tmp_path / "built/method.json").read_bytes()).hexdigest()
    )


def test_missing_measured_inputs_does_not_borrow_a_mask_or_leave_partial_output(tmp_path):
    store = WeightStore({NAME: torch.from_numpy(BASE)})
    destination = tmp_path / "missing"
    with pytest.raises(ValueError, match="measured contrastive input"):
        build_repair(
            store,
            destination,
            directions={NAME: [1, 0]},
            alpha=6,
            column_rule="topk",
            top_k=2,
            provenance=PROVENANCE,
        )
    assert not destination.exists()


def test_build_obeys_frozen_and_global_rescaling_refusals(tmp_path):
    frozen = "model.embed_tokens.weight"
    with pytest.raises(EditRefused, match="Frozen"):
        build_repair(
            WeightStore({frozen: torch.from_numpy(BASE)}),
            tmp_path / "frozen",
            directions={frozen: [1, 0]},
            alpha=1,
            column_rule="all",
            provenance=PROVENANCE,
        )
    rank_one = np.array([[1, 2], [0, 0]], dtype=np.float32)
    with pytest.raises(EditRefused, match="rescaling"):
        build_repair(
            WeightStore({NAME: torch.from_numpy(rank_one)}),
            tmp_path / "rescale",
            directions={NAME: [1, 0]},
            alpha=6,
            column_rule="all",
            provenance=PROVENANCE,
        )
    assert not (tmp_path / "frozen").exists() and not (tmp_path / "rescale").exists()


def test_zero_dose_exports_an_empty_identity_bundle_and_refuses_overwrite(tmp_path):
    store = WeightStore({NAME: torch.from_numpy(BASE)})
    kwargs = {"directions": {NAME: [1, 0]}, "alpha": 0, "column_rule": "all", "provenance": PROVENANCE}
    result = build_repair(store, tmp_path / "zero", **kwargs)
    assert result["changed_tensors"] == result["changed_entries_total"] == 0
    store.apply_edit(tmp_path / "zero/edit")
    assert store.edited() == []
    with pytest.raises(FileExistsError):
        build_repair(store, tmp_path / "zero", **kwargs)


@pytest.mark.parametrize(
    "changes",
    [
        {"direction": [0, 0]},
        {"direction": [1, np.nan]},
        {"alpha": -1},
        {"alpha": True},
        {"column_rule": "mystery"},
        {"column_rule": "topk", "top_k": 5},
    ],
)
def test_invalid_direction_dose_and_rule_are_rejected(changes):
    kwargs = {"direction": [1, 0], "alpha": 1, "column_rule": "all", **changes}
    with pytest.raises(ValueError):
        repair_tensor(BASE, **kwargs)


def test_manifest_cli_binds_arrays_and_builds_on_cpu(tmp_path):
    save_file({NAME: torch.from_numpy(BASE)}, tmp_path / "model.safetensors")
    np.savez(tmp_path / "acts.npz", output_looping=np.array([[1, 0]]), output_finished=np.array([[0, 0]]))
    manifest = {
        "format_version": 1,
        "arrays": {
            "path": "acts.npz",
            "sha256": hashlib.sha256((tmp_path / "acts.npz").read_bytes()).hexdigest(),
        },
        "layers": {NAME: {"output_looping": "output_looping", "output_finished": "output_finished"}},
        "provenance": PROVENANCE,
    }
    (tmp_path / "inputs.json").write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "replications.exp8.repair",
            "--model",
            str(tmp_path / "model.safetensors"),
            "--output",
            str(tmp_path / "built"),
            "--activation-manifest",
            str(tmp_path / "inputs.json"),
            "--variant",
            "2a",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["build_status"] == "built"
    (tmp_path / "acts.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_activation_manifest(tmp_path / "inputs.json")


def test_provenance_counts_and_saved_method_hash_must_be_valid(tmp_path):
    store = WeightStore({NAME: torch.from_numpy(BASE)})
    with pytest.raises(ValueError, match="group counts differ"):
        build_repair(
            store,
            tmp_path / "wrong-counts",
            alpha=1,
            column_rule="all",
            provenance=PROVENANCE,
            activations={NAME: {"output_looping": [[1, 0], [2, 0]], "output_finished": [[0, 0]]}},
        )
    with pytest.raises(ValueError, match="saved_method_sha256"):
        build_repair(
            store,
            tmp_path / "wrong-hash",
            alpha=1,
            column_rule="all",
            directions={NAME: [1, 0]},
            provenance={
                "source": "method",
                "activation_reduction": "means",
                "saved_method_sha256": "unknown",
            },
        )
    assert not (tmp_path / "wrong-counts").exists() and not (tmp_path / "wrong-hash").exists()
