"""Cross-interface contracts on a real small network, with no external services."""

import json
import sys
import types

import pytest
import torch
from typer.testing import CliRunner

from openwsp.cli import app
from openwsp.sdk import OpenWSP, load_backend


def test_sdk_edit_changes_forward_and_roundtrips(tmp_path):
    api = OpenWSP.tiny(tmp_path / "first")
    request = {"inputs": [[1, 0], [0, 1]], "model": "both"}
    before = api.forward(request)
    assert before["original"]["logits"] == before["edited"]["logits"]
    api.save_edit("layer1.weight", [[2, 2], [-1, 1]])
    after = api.forward(request)
    assert after["edited"]["logits"] != after["original"]["logits"]
    assert before["original"] == after["original"]
    assert len(api.diff()) == 1
    api.export_edit(tmp_path / "export", base_model="openwsp/tiny", revision="test-v1")
    restored = OpenWSP.tiny(tmp_path / "second")
    restored.apply_edit(tmp_path / "export")
    assert restored.forward(request) == after


def test_tiny_forward_roundtrip_preserves_bf16_evaluated_decimal_edits(tmp_path):
    api = OpenWSP.tiny(tmp_path / "submitted")
    values = [[1.123456, 2.987654], [-1, 1]]
    api.save_edit("layer1.weight", values)
    request = {"inputs": [[0.125, 0.25]], "model": "both"}
    expected = api.forward(request)
    api.export_edit(tmp_path / "export")
    restored = OpenWSP.tiny(tmp_path / "restored")
    restored.apply_edit(tmp_path / "export")
    assert restored.forward(request) == expected
    assert torch.equal(api.store.load("layer1.weight"), torch.tensor(values))


def test_bounded_reads_and_persistent_kernel(tmp_path):
    api = OpenWSP.tiny(tmp_path)
    assert len(api.list_tensors()) == 4
    read = api.read_tensor("layer1.weight", offset=1, limit=2)
    assert read["values"] == [2.0, -1.0]
    assert read["shape"] == [2, 2]
    assert read["truncated"]
    for kwargs in ({"limit": 0}, {"limit": 4097}, {"offset": -1}, {"offset": True}):
        with pytest.raises(ValueError):
            api.read_tensor("layer1.weight", **kwargs)
    assert api.run_code("x = load('layer1.weight'); x[0, 0] = 3; save_edit('layer1.weight', x)")["ok"]
    assert "3" in api.run_code("x[0, 0].item()")["output"]


def test_cli_persists_edit_across_invocations(tmp_path):
    runner = CliRunner()
    prefix = ["--tiny", "--work-dir", str(tmp_path / "work")]
    result = runner.invoke(app, [*prefix, "save", "layer1.weight", "[[2,2],[-1,1]]"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, [*prefix, "read", "layer1.weight", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["values"] == [2.0]
    result = runner.invoke(app, [*prefix, "run", "--code", "1 / 0"])
    assert result.exit_code == 1
    assert not json.loads(result.output)["ok"]


def test_cli_replication_commands_delegate(monkeypatch):
    calls = []
    fake = types.ModuleType("openwsp.replication")
    fake.replicate = lambda *args, **kwargs: calls.append((args, kwargs)) or {"launched": True}
    fake.status = lambda experiment: {"experiment": experiment, "state": "pending"}
    fake.report = lambda experiment: {"experiment": experiment, "written": True}
    monkeypatch.setitem(sys.modules, "openwsp.replication", fake)
    runner = CliRunner()
    result = runner.invoke(app, ["replicate", "exp8", "--stage", "calibration", "--foreground"])
    assert result.exit_code == 0, result.output
    assert calls == [(("exp8",), {"stage": "calibration", "detach": False, "config": None})]
    assert runner.invoke(app, ["status"]).exit_code == 0
    assert runner.invoke(app, ["report", "exp8"]).exit_code == 0


def test_backend_errors_and_input_validation(tmp_path):
    api = OpenWSP.tiny(tmp_path)
    with pytest.raises(ValueError):
        api.forward({"inputs": [[1, 2, 3]]})
    with pytest.raises(ValueError):
        api.forward({"inputs": [[1, 2]], "model": "unknown"})
    api.backend = None
    with pytest.raises(RuntimeError, match="No forward backend"):
        api.forward({"inputs": [[1, 2]]})
    with pytest.raises(ValueError, match="module:factory"):
        load_backend("invalid")
    with pytest.raises(ValueError, match="real"):
        api.save_edit("layer1.weight", torch.ones((2, 2), dtype=torch.complex64))


def test_forward_validates_even_unread_submission_files(tmp_path):
    import numpy as np

    api = OpenWSP.tiny(tmp_path)
    np.save(tmp_path / "edited/unknown.npy", np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError, match="Unknown"):
        api.forward({"inputs": [[1, 0]], "model": "original"})


def test_forward_uses_one_snapshot_and_does_not_mutate_caller_request(tmp_path):
    import numpy as np

    api = OpenWSP.tiny(tmp_path)
    name = "layer1.weight"
    api.save_edit(name, [[2, 2], [-1, 1]])
    expected = api.store.load(name)
    request = {"inputs": [[1, 0]], "model": "edited"}

    class Backend:
        def forward(self, received, store):
            assert not torch.is_grad_enabled()
            received["inputs"][0][0] = 99
            before = store.load(name)
            np.save(tmp_path / "edited" / f"{name}.npy", np.array([[3, 2], [-1, 1]], dtype=np.float32))
            after = store.load(name)
            store.drop_edit(name)
            return {"before": before.tolist(), "after": after.tolist()}

    api.backend = Backend()
    result = api.forward(request)
    assert result["before"] == result["after"] == expected.tolist()
    assert request == {"inputs": [[1, 0]], "model": "edited"}
    assert api.store.load(name)[0, 0].item() == 3
    assert api.store.edited() == [name]
