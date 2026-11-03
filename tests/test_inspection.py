"""CPU-only edit geometry and cross-interface inspection of planted repairs."""

import json
import shutil

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from openwsp.cli import app
from openwsp.core import WeightStore
from openwsp.edits import EditError
from openwsp.inspection import direction_cosines, directions_from_edits, inspect_tensor, logit_lens
from openwsp.sdk import OpenWSP


def planted():
    base = np.array([[1.0, 2.0, -3.0, 4.0], [2.0, -1.0, 4.0, 2.0], [1.0, 3.0, 2.0, -1.0]])
    direction = np.array([0.6, 0.8, 0.0])
    projection = direction @ base
    projection[[1, 3]] = 0
    return base, base - 6 * np.outer(direction, projection), direction


def test_planted_rank_one_projection_recovers_direction_columns_and_dose():
    base, edited, direction = planted()
    original, replacement = base.copy(), edited.copy()
    result = inspect_tensor("down.weight", base, edited)
    assert result["changed_rows"] == 2 and result["row_indices"] == [0, 1]
    assert result["changed_columns"] == 2 and result["column_indices"] == [0, 2]
    assert result["svd"]["rank"] == 1
    assert result["svd"]["energy_rank_99pct"] == 1
    assert result["svd"]["rank_one_relative_error"] < 1e-14
    assert np.allclose(result["svd"]["directions"][0]["vector"], direction)
    assert result["projection"]["consistent"]
    assert [row["alpha"] for row in result["projection"]["coefficients"]] == pytest.approx([6, 6])
    assert result["projection"]["relative_error"] < 1e-14
    assert result["relative_frobenius_change"] == pytest.approx(
        np.linalg.norm(edited - base) / np.linalg.norm(base)
    )
    assert np.array_equal(base, original) and np.array_equal(edited, replacement)


def test_rounded_rank_one_edit_reports_numerical_noise_without_claiming_exact_rank_one():
    rng = np.random.default_rng(8)
    base = torch.from_numpy(rng.normal(size=(20, 12)).astype(np.float32)).bfloat16().float().numpy()
    u = rng.normal(size=20)
    u /= np.linalg.norm(u)
    target = base - 6 * np.outer(u, u @ base)
    rounded = torch.from_numpy(target.astype(np.float32)).bfloat16().float().numpy()
    result = inspect_tensor("down.weight", base, rounded)
    assert result["svd"]["rank"] > 1
    assert result["svd"]["energy_rank_99pct"] == 1
    assert result["svd"]["directions"][0]["energy_fraction"] > 0.999
    assert result["projection"]["consistent"]
    assert result["projection"]["coefficient_summary"]["median"] == pytest.approx(6, abs=0.03)
    assert result["svd"]["rank_one_relative_error"] > 0


def test_projection_refuses_an_unidentifiable_zero_base_projection():
    base = np.array([[0.0, 0.0], [1.0, 2.0]])
    result = inspect_tensor("weight", base, base + np.array([[1.0, 2.0], [0.0, 0.0]]))
    assert result["svd"]["rank"] == 1
    assert result["projection"]["consistent"] is False
    assert result["projection"]["relative_error"] == pytest.approx(1)
    assert all(row["alpha"] is None for row in result["projection"]["coefficients"])


def test_zero_and_nonmatrix_edits_have_finite_explicit_results():
    base = np.zeros((3, 2))
    result = inspect_tensor("zero", base, base)
    assert result["svd"]["rank"] == 0 and result["svd"]["directions"] == []
    assert result["relative_frobenius_change"] is None
    assert result["changed_rows"] == result["changed_columns"] == 0
    vector = inspect_tensor("bias", np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert vector["changed_entries"] == 1 and vector["svd"] is None
    assert vector["matrix_analysis_unavailable"]
    json.dumps([result, vector], allow_nan=False)


def test_direction_cosines_are_sign_explicit_and_lens_uses_named_rows_and_mean():
    base, edited, u = planted()
    tensor = inspect_tensor("down", base, edited, top_directions=1)
    directions = directions_from_edits(
        [{"label": "first", "tensors": [tensor]}, {"label": "second", "tensors": [tensor]}]
    )
    comparisons = direction_cosines(directions)
    assert len(comparisons) == 1 and comparisons[0]["absolute_cosine"] == pytest.approx(1)
    directions[1]["vector"] = (-u).tolist()
    assert direction_cosines(directions)[0]["cosine"] == pytest.approx(-1)
    embedding = np.array([u, -u, [0.0, 0.0, 1.0]])
    result = logit_lens(directions, embedding, {"Wait": 0, "But": 1})
    assert result["directions"][0]["token_cosines"] == pytest.approx({"Wait": 1, "But": -1})
    assert result["directions"][0]["mean_row_cosine"] == pytest.approx(0)
    assert result["mean_row_norm"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "bad", [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1j]]), np.array([["x"]])]
)
def test_invalid_numeric_inputs_are_rejected(bad):
    with pytest.raises(ValueError, match="finite real"):
        inspect_tensor("weight", np.ones((1, 1)), bad)


def test_store_inspects_legacy_and_v1_without_applying_either(tmp_path):
    base, edited, u = planted()
    store = WeightStore(
        {
            "down.weight": torch.from_numpy(base.astype(np.float32)),
            "model.embed_tokens.weight": torch.from_numpy(
                np.array([u, -u, [0.0, 0.0, 1.0]], dtype=np.float32)
            ),
        }
    )
    store.save_edit("down.weight", edited)
    bundle = tmp_path / "v1"
    manifest = store.export_edit(bundle, base_model="tiny", revision="test")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    shutil.copyfile(bundle / "edit.npz", legacy / "edit.npz")
    (legacy / "manifest.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"format", "format_version", "edit_sha256", "payload_sha256"}
            }
        )
    )
    store.drop_edit("down.weight")
    result = store.inspect(
        legacy,
        compare=bundle,
        top_directions=1,
        embedding_name="model.embed_tokens.weight",
        token_ids={"Wait": 0},
    )
    assert result["format"] == "openwsp.inspect"
    assert result["edits"][0]["source_format"] == "legacy"
    assert result["edits"][1]["edit_sha256"] == manifest["edit_sha256"]
    assert result["direction_cosines"][0]["absolute_cosine"] == pytest.approx(1)
    assert result["logit_lens"]["directions"][0]["token_cosines"]["Wait"] > 0.999
    assert store.edited() == []
    assert torch.equal(store.load("down.weight"), torch.from_numpy(base.astype(np.float32)))
    bad = WeightStore({"down.weight": torch.from_numpy(base.astype(np.float32) + 1)})
    with pytest.raises(EditError, match="Base tensor"):
        bad.inspect(bundle)


def test_sdk_and_cli_inspect_current_and_exported_edits_share_the_core(tmp_path):
    api = OpenWSP.tiny(tmp_path / "sdk")
    api.save_edit("layer1.weight", [[-5, -10], [-1, 1]])
    current = api.inspect(top_directions=1)
    assert current["edits"][0]["tensors"][0]["projection"]["coefficient_summary"]["median"] == 6
    bundle = tmp_path / "edit"
    api.export_edit(bundle)
    result = CliRunner().invoke(
        app,
        [
            "--tiny",
            "--work-dir",
            str(tmp_path / "cli"),
            "inspect",
            "--path",
            str(bundle),
            "--compare",
            str(bundle),
            "--top-directions",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    inspected = json.loads(result.output)
    assert inspected["edits"][0]["tensors"] == current["edits"][0]["tensors"]
    assert inspected["direction_cosines"][0]["absolute_cosine"] == pytest.approx(1)
    assert list((tmp_path / "cli" / "edited").glob("*.npy")) == []
