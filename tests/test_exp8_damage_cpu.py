"""Local mirror and execution checks with existing synthetic fixtures only."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_exp8_damage_stage as stage_tests
import torch
from test_damage_backend import Model

from openwsp.backends.forward_bank import json_hash
from replications.exp8 import damage_cpu as cpu
from replications.exp8 import damage_stage as stage
from replications.exp8.runtime import atomic_write

setup = stage_tests.setup


@pytest.fixture
def mirror(setup, monkeypatch):
    # The original stage fixture uses local paths; a Volume mirror must retain
    # the original /data paths in its saved submissions.
    for directory in (setup.data, setup.practice):
        directory.chmod(0o700)
    setup.submission["checkpoint"] = "/data/final"
    atomic_write(setup.data / setup.identity.relative / "state.json",
                 {"done": True, "passes": 50, "submission": setup.submission})
    for name in ("mechanical", "random"):
        atomic_write(setup.data / "controls" / name / "submission.json",
                     {"sha256": name[0] * 64, "checkpoint": f"/data/{name}"})
    prepare_refs = stage.prepare_references
    monkeypatch.setattr(stage, "prepare_references", lambda tokenizer, practice, **kwargs:
                        prepare_refs(tokenizer, practice, expected_hashes=setup.expected))
    monkeypatch.setattr(cpu, "_tokenizer", lambda path: setup.tokenizer)
    monkeypatch.setattr(cpu, "_model", lambda path: pytest.fail("Unexpected model load"))
    # The synthetic fixture represents the authoritative source. Capture its
    # canonical manifest hashes before any mirror mutation in a test.
    setup.anchors = {manifest["edit"]: json_hash(manifest) for _, manifest in prepare(setup)}
    return setup


def run(mirror):
    return cpu.run_cpu(data=mirror.data, practice=mirror.practice, identity=mirror.identity,
                       expected_manifest_sha256=mirror.anchors)


def prepare(mirror):
    return stage.prepare_manifests(mirror.tokenizer, identity=mirror.identity, submission=mirror.submission,
                                  resolve_path=cpu.mirror_paths(mirror.data, mirror.practice))


def test_priority_gate_precedes_preparation_model_load_and_output(mirror, monkeypatch):
    (mirror.data / mirror.identity.relative / "state.json").unlink()
    shutil.rmtree(mirror.data / "damage")
    for owner, name in ((cpu, "_execution"), (cpu, "_tokenizer"), (cpu, "_model"), (stage, "prepare_manifests")):
        monkeypatch.setattr(owner, name, lambda *a, **kw: pytest.fail("Gate must run first"))
    with pytest.raises(ValueError, match="integer 50-pass"):
        run(mirror)
    assert not (mirror.data / "damage").exists()
    assert not (mirror.data / ".damage-cpu.lock").exists()


@pytest.mark.parametrize("failure", ["missing", "hash", "partial"])
def test_source_anchors_are_required_before_tokenization_or_preparation(mirror, monkeypatch, failure):
    path = mirror.data / "damage" / mirror.identity.relative / "live.manifest.json"
    if failure == "missing":
        path.unlink()
    elif failure == "hash":
        manifest = json.loads(path.read_text())
        manifest["edited_checkpoint"] = manifest["base_checkpoint"]
        atomic_write(path, manifest)
    else:
        mirror.anchors.pop("live")
    for owner, name in ((cpu, "_execution"), (cpu, "_tokenizer"), (stage, "prepare_manifests")):
        monkeypatch.setattr(owner, name, lambda *a, **kw: pytest.fail("Source bindings must be checked first"))
    with pytest.raises(ValueError, match="independently recorded source manifest"):
        run(mirror)


@pytest.mark.parametrize("relative", ["live-runs/damage-test/state.json", "benchmarks/math500.jsonl",
                                      "results/control4_random/math500/samples.jsonl", ".damage-cpu.lock"])
def test_prerequisite_and_lock_symlinks_are_rejected(mirror, monkeypatch, tmp_path, relative):
    path = mirror.data / relative
    outside = tmp_path / "outside"
    outside.write_bytes(path.read_bytes() if path.exists() else b"")
    path.unlink(missing_ok=True)
    path.symlink_to(outside)
    monkeypatch.setattr(cpu, "_tokenizer", lambda path: pytest.fail("Symlink must be rejected first"))
    with pytest.raises(ValueError, match="symlinks"):
        run(mirror)


def test_mirror_paths_reject_escape_and_nonprivate_roots(mirror, tmp_path):
    resolve = cpu.mirror_paths(mirror.data, mirror.practice)
    assert resolve("/data/original") == mirror.data / "original"
    for source in ("relative", "/other/model", "/data/../practice", "/data2/model"):
        with pytest.raises(ValueError):
            resolve(source)
    (mirror.data / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        resolve("/data/linked/anything")
    mirror.data.chmod(0o755)
    with pytest.raises(ValueError, match="private"):
        cpu.mirror_paths(mirror.data, mirror.practice)


def test_source_manifests_survive_physical_relocation(mirror, tmp_path):
    manifests = prepare(mirror)
    data = tmp_path / "relocated-data"
    practice = tmp_path / "relocated-practice"
    shutil.copytree(mirror.data, data)
    shutil.copytree(mirror.practice, practice)
    resolve = cpu.mirror_paths(data, practice)
    repeated = stage.prepare_manifests(mirror.tokenizer, identity=mirror.identity, submission=mirror.submission,
                                      resolve_path=resolve)
    assert repeated == manifests
    for path, manifest in manifests:
        assert str(path).startswith("/data/damage/")
        assert manifest["base_checkpoint"]["path"] == "/data/original"
        assert manifest["edited_checkpoint"]["path"].startswith("/data/")
        assert manifest["edit_bundle"]["path"].startswith("/data/")
        assert str(tmp_path) not in json.dumps(manifest)
        loaded, records = stage.load_measurement_inputs(path, json_hash(manifest), base_checkpoint="/data/original",
                                                        tokenizer=mirror.tokenizer, resolve_path=resolve)
        assert loaded == manifest and len(records) == 512


@pytest.mark.parametrize("relative", ["final/model.safetensors", "controls/random/edit/edit.npz",
                                      "damage/live-runs/damage-test/references.tokens.json"])
def test_mirror_tampering_stops_before_any_model_load(mirror, relative):
    prepare(mirror)
    path = mirror.data / relative
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        value["records"][0]["input_ids"][0] += 1
        atomic_write(path, value)
    else:
        path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="conflicting immutable identity"):
        run(mirror)


def test_records_must_remain_siblings_in_both_path_namespaces(mirror):
    path, manifest = prepare(mirror)[0]
    resolve = cpu.mirror_paths(mirror.data, mirror.practice)
    other = mirror.data / "other-tokens.json"
    shutil.copyfile(resolve(manifest["records_path"]), other)

    def relocated_records(source):
        return other if str(source) == manifest["records_path"] else resolve(source)

    with pytest.raises(ValueError, match="beside the manifest"):
        stage.load_measurement_inputs(path, json_hash(manifest), base_checkpoint="/data/original",
                                      tokenizer=mirror.tokenizer, resolve_path=relocated_records)
    changed = {**manifest, "records_path": "/data/other-tokens.json"}
    atomic_write(resolve(path), changed)
    with pytest.raises(ValueError, match="beside the manifest"):
        stage.load_measurement_inputs(path, json_hash(changed), base_checkpoint="/data/original",
                                      tokenizer=mirror.tokenizer, resolve_path=resolve)


def test_complete_tiny_bf16_measurement_reuses_only_exact_cpu_receipts(mirror, monkeypatch):
    loads = []

    def load(path):
        loads.append(path.name)
        model = Model(vocab=32, edited=path.name != "original").to(dtype=torch.bfloat16).eval()
        # This synthetic decoder's positional arithmetic otherwise promotes to
        # FP32; keep its output in the same dtype as its BF16 vocabulary head.
        model.model.register_forward_hook(lambda module, args, output:
            SimpleNamespace(last_hidden_state=output.last_hidden_state.to(torch.bfloat16)))
        return model

    monkeypatch.setattr(cpu, "_model", load)
    previous_threads = torch.get_num_threads()
    try:
        result = run(mirror)
    finally:
        torch.set_num_threads(previous_threads)
    assert list(result["edits"]) == ["p50c", "mechanical", "random", "live"]
    assert loads == ["original", "p50c", "mechanical", "random", "final"]
    for summary in result["edits"].values():
        assert summary["sequence_count"] == summary["scored_sequence_count"] == 512
        assert summary["kl_mean"] > 0 and summary["chunk_tokens"] == 32
        assert summary["smoke"]["sequence_count"] == 4 and summary["smoke"]["kl_sum"] == 0
        assert summary["smoke"]["included_in_measurement"] is False
        assert summary["seconds"] > 0 and summary["cost_usd"] == summary["incremental_modal_cost_usd"] == 0
        assert summary["local_cpu_cost_usd"] is None
        provenance = summary["provenance"]
        assert provenance["engine_settings"] == stage.ENGINE_SETTINGS
        assert provenance["gpu"] is None and provenance["execution"]["device"] == "cpu"
        assert provenance["execution"]["threads"] == 2
        assert set(provenance["execution"]["implementation_sha256"]) == {"runner", "stage", "kernel"}
        assert "unpriced, not free" in provenance["cost_basis"]
        assert provenance["faithful_submission_sha256"] == mirror.submission["sha256"]
        assert str(mirror.data) not in json.dumps(summary)

    monkeypatch.setattr(cpu, "_model", lambda path: pytest.fail("Completed measurements must be reused"))
    assert run(mirror) == result
    target = mirror.data / "results/damage" / mirror.identity.relative / "live/summary.json"
    original = json.loads(target.read_text())
    # Neither a provider receipt nor a different implementation may be relabeled
    # as a zero-incremental-Modal CPU measurement.
    for update in ({"gpu": "H100"}, {"execution": {"device": "cuda"}}):
        atomic_write(target, {**original, "provenance": {**original["provenance"], **update}})
        with pytest.raises(ValueError, match="differs from this CPU measurement"):
            run(mirror)
    atomic_write(target, {**original, "kl_mean": original["kl_mean"] + 0.5})
    with pytest.raises(ValueError, match="exact registered measurement"):
        run(mirror)


def test_loader_has_no_network_gpu_or_dtype_fallback(monkeypatch):
    import transformers

    calls = []

    class Loaded:
        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return Loaded()

    monkeypatch.setattr(transformers.Qwen3_5ForConditionalGeneration, "from_pretrained", from_pretrained)
    cpu._model(Path("/private/original"))
    assert calls == [(Path("/private/original"), {"dtype": torch.bfloat16, "attn_implementation": "sdpa",
                                               "local_files_only": True}), ("device", "cpu"), ("eval",)]
    monkeypatch.setattr(transformers.Qwen3_5ForConditionalGeneration, "from_pretrained",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("CPU unsupported")))
    with pytest.raises(RuntimeError, match="CPU unsupported"):
        cpu._model(Path("/private/original"))
