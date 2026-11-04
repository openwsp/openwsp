"""Authoritative manifest preparation uses the existing driver, with no dispatch."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import test_exp8_damage_stage as stage_tests

from openwsp.backends.forward_bank import json_hash
from replications.exp8 import damage_stage as stage
from replications.exp8.runtime import atomic_write

setup = stage_tests.setup


@pytest.fixture
def source(setup, monkeypatch):
    import transformers

    prepare = stage.prepare_references
    monkeypatch.setattr(stage, "prepare_references", lambda tokenizer, practice, **kwargs:
                        prepare(tokenizer, practice, expected_hashes=setup.expected))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: setup.tokenizer)
    monkeypatch.setattr(transformers.Qwen3_5ForConditionalGeneration, "from_pretrained",
                        lambda *a, **kw: pytest.fail("Preparation must not load a model"))
    monkeypatch.setattr(stage, "gpu_instance", lambda *a, **kw: pytest.fail("Preparation must not dispatch"))
    return setup


def test_source_preparation_returns_only_committed_ordered_manifest_anchors(source):
    commits = []

    def commit():
        directory = source.data / "damage" / source.identity.relative
        assert len(list(directory.glob("*.manifest.json"))) == 5  # Four edits plus the shared references.
        commits.append(True)

    result = stage.prepare_damage_stage(commit, identity=source.identity, data=source.data, practice=source.practice)
    assert commits == [True]
    assert result["status"] == "done" and result["stage"] == "damage_prepare"
    assert result["faithful_submission_sha256"] == source.submission["sha256"]
    assert result["hash_algorithm"] == "canonical_json_sha256"
    assert [row["edit"] for row in result["manifests"]] == ["p50c", "mechanical", "random", "live"]
    assert all(set(row) == {"edit", "path", "sha256"} for row in result["manifests"])
    for row in result["manifests"]:
        manifest = json.loads(Path(row["path"]).read_text())
        assert row["sha256"] == json_hash(manifest)
        assert len(manifest["dataset"]["examples"]) == 512
    assert "Unchanged reference" not in json.dumps(result)
    assert "measurement" not in result and "cost_usd" not in result


def test_source_preparation_gate_runs_before_tokenizer_or_writes(source, monkeypatch):
    import transformers

    (source.data / source.identity.relative / "state.json").unlink()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *a, **kw: pytest.fail("Gate must precede tokenizer loading"))
    with pytest.raises(ValueError, match="integer 50-pass"):
        stage.prepare_damage_stage(lambda: pytest.fail("No preparation should be committed"),
                                   identity=source.identity, data=source.data, practice=source.practice)
    assert not (source.data / "damage").exists()


@pytest.mark.parametrize("complete", [False, True])
def test_explicit_pipeline_preparation_never_dispatches_generic_preparation_or_gpu(source, monkeypatch, complete):
    from replications.exp8 import pipeline

    # Even with no generic prepare completion flag, this stage must gate first
    # and must never dispatch prepare-v2 to repair missing prerequisites.
    atomic_write(source.data / "state.json", {"stages": {}})
    atomic_write(source.data / "active-live.json", {"editor": source.identity.editor, "run_id": source.identity.run_id})
    if not complete:
        (source.data / source.identity.relative / "state.json").unlink()
        (source.data / "benchmarks/ifeval.jsonl").unlink()
    module = ModuleType("openwsp.backends.modal_app")
    module.data_volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    module.prepare = SimpleNamespace(spawn=lambda: pytest.fail("No generic preparation dispatch"))
    module.Generator = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(pipeline, "Path", lambda value: source.data if value == "/data" else Path(value))
    no_jobs = SimpleNamespace(launch=lambda *a, **kw: pytest.fail("No remote job dispatch"))
    monkeypatch.setattr(pipeline, "RemoteJobRunner", lambda *a, **kw: no_jobs)
    monkeypatch.setattr(pipeline, "ProtocolGPUJobs", lambda *a, **kw: no_jobs)
    prepare = stage.prepare_damage_stage
    monkeypatch.setattr(stage, "prepare_damage_stage", lambda checkpoint, *, identity:
                        prepare(checkpoint, identity=identity, data=source.data, practice=source.practice))
    monkeypatch.setattr(stage, "run_damage_stage", lambda *a, **kw: pytest.fail("No measurement dispatch"))
    if not complete:
        with pytest.raises(ValueError, match="integer 50-pass"):
            pipeline.run("damage_prepare", {}, app=object())
        assert not (source.data / "damage").exists()
    else:
        result = pipeline.run("damage_prepare", {}, app=object())
        assert result["status"] == "stage_done"
        assert set(result["stages"]) == {"damage_prepare"}
        assert result["stages"]["damage_prepare"]["faithful_submission_sha256"] == source.submission["sha256"]
