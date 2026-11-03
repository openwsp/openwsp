"""Damage orchestration checks with synthetic files and no model/provider calls."""
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from openwsp.backends.forward_bank import json_hash
from replications.exp8 import damage_stage as stage
from replications.exp8.eval_grid import STAGES, load_cell
from replications.exp8.run_identity import LiveIdentity
from replications.exp8.runtime import CONFIG, atomic_write, write_rows


class Tokenizer:
    all_special_ids = (0,)
    chat_template = "synthetic thinking template"

    def get_vocab(self):
        return {str(i): i for i in range(32)}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is True
        return "<think>" + messages[0]["content"]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) % 31 + 1 for character in text]


def prerequisites(data, identity):
    submission = {"sha256": "a" * 64, "checkpoint": str(data / "final")}
    atomic_write(data / identity.relative / "state.json", {"done": True, "passes": 50, "submission": submission})

    def completed(directory, benchmark, samples, provenance=None):
        atomic_write(directory / "summary.json", {"status": "done", "samples_per_problem": samples,
                                                   "n": stage.BENCHMARK_SIZES[benchmark] * samples,
                                                   "provenance": provenance or {}})
        write_rows(directory / "samples.jsonl",
                   [{"problem_id": f"{benchmark}_{i}", "sample_index": index}
                    for i in range(stage.BENCHMARK_SIZES[benchmark]) for index in range(samples)])

    for benchmark, samples in CONFIG["samples"].items():
        (data / "benchmarks").mkdir(exist_ok=True)
        write_rows(data / "benchmarks" / f"{benchmark}.jsonl", [{"id": f"{benchmark}_{i}"} for i in range(stage.BENCHMARK_SIZES[benchmark])])
        completed(data / "results" / identity.evaluation_stage / benchmark, benchmark, samples,
                  {"submission_sha256": submission["sha256"]})
    completed(data / "results/control4_random/math500", "math500", 1)
    for cell in STAGES:
        completed(data / "results" / cell / "math500", "math500", 1,
                  {"eval_grid": {"config_sha256": load_cell(cell).config_sha256}, "faithful_submission_sha256": submission["sha256"]})
    return submission


@pytest.fixture
def setup(tmp_path):
    data, practice = tmp_path / "data", tmp_path / "practice"
    practice.mkdir()
    identity = LiveIdentity("codex", "damage-test")
    submission = prerequisites(data, identity)
    expected = {}
    for source in ("gsm8k", "math"):
        filename = f"{source}_train.jsonl"
        rows = [{"id": f"{source}_{i}", "source": source, "problem": f"Problem {i}",
                 "solution": f"Unchanged reference {i}.\n", "answer": "unused"} for i in range(260)]
        write_rows(practice / filename, rows)
        expected[f"exp8_math/practice/{filename}"] = hashlib.sha256((practice / filename).read_bytes()).hexdigest()
    for name in ("original", "p50c", "mechanical", "random", "final"):
        directory = data / name
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(name.encode())
        (directory / "config.json").write_text("{}")
        (directory / "tokenizer.json").write_text('{"synthetic": true}')
    for name in ("mechanical", "random"):
        atomic_write(data / "controls" / name / "submission.json",
                     {"sha256": name[0] * 64, "checkpoint": str(data / name)})
    for relative in ("reference_edit", "controls/mechanical/edit", "controls/random/edit", f"{identity.relative}/edit"):
        directory = data / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text("{}")
        (directory / "edit.npz").write_bytes(relative.encode())
    return SimpleNamespace(data=data, practice=practice, identity=identity, submission=submission,
                           expected=expected, tokenizer=Tokenizer())


def prepared(setup):
    return stage.prepare_manifests(setup.tokenizer, identity=setup.identity, submission=setup.submission,
                                   data=setup.data, practice=setup.practice, expected_hashes=setup.expected)


def test_every_prior_stage_must_bind_to_the_actual_completed_submission(setup):
    assert stage.require_damage_priority(identity=setup.identity, data=setup.data) == setup.submission
    paths = [setup.data / setup.identity.relative / "state.json",
             *[setup.data / "results" / setup.identity.evaluation_stage / name / "summary.json" for name in CONFIG["samples"]],
             setup.data / "results/control4_random/math500/summary.json",
             *[setup.data / "results" / cell / "math500/summary.json" for cell in STAGES]]
    for path in paths:
        raw = path.read_bytes()
        path.unlink()
        with pytest.raises(ValueError):
            stage.require_damage_priority(identity=setup.identity, data=setup.data)
        path.write_bytes(raw)
    path = setup.data / "results" / STAGES[-1] / "math500/summary.json"
    value = json.loads(path.read_text())
    value["provenance"]["faithful_submission_sha256"] = "wrong"
    atomic_write(path, value)
    with pytest.raises(ValueError, match="complete preregistered"):
        stage.require_damage_priority(identity=setup.identity, data=setup.data)


@pytest.mark.parametrize("relative", ["live-runs/damage-test/math500", "live-runs/damage-test/gsm8k",
                                     "live-runs/damage-test/ifeval", "control4_random/math500", STAGES[-1] + "/math500"])
def test_done_headers_cannot_hide_missing_or_duplicated_benchmark_samples(setup, relative):
    path = setup.data / "results" / relative / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    write_rows(path, rows[:-1] + [rows[0]])
    with pytest.raises(ValueError, match="full benchmark sample coverage"):
        stage.require_damage_priority(identity=setup.identity, data=setup.data)
    write_rows(path, rows[:50])
    with pytest.raises(ValueError, match="completed subset"):
        stage.require_damage_priority(identity=setup.identity, data=setup.data)


@pytest.mark.parametrize("changed", [{"done": 1}, {"passes": 50.0}])
def test_faithful_completion_fields_are_strict(setup, changed):
    path = setup.data / setup.identity.relative / "state.json"
    atomic_write(path, {**json.loads(path.read_text()), **changed})
    with pytest.raises(ValueError, match="integer 50-pass"):
        stage.require_damage_priority(identity=setup.identity, data=setup.data)


def test_reference_selection_full_hashes_and_continuation_masks_are_fixed(setup):
    manifest, tokens = stage.prepare_references(setup.tokenizer, setup.practice, expected_hashes=setup.expected)
    rows = manifest["examples"]
    assert [row["id"] for row in rows] == [f"{source}_{i}" for source in ("gsm8k", "math") for i in range(256)]
    assert manifest["scope"]["overflow"] == "reject" and manifest["smoke_indices"] == [0, 1, 256, 257]
    assert manifest["smoke_in_measurement"] is False
    assert rows[0]["reference_sha256"] == hashlib.sha256(b"Unchanged reference 0.\n").hexdigest()
    for meta, row in zip(rows, tokens["records"], strict=True):
        assert row["score_mask"] == [False] * meta["prompt_tokens"] + [True] * meta["reference_tokens"]
        assert json_hash(row) == meta["token_sha256"]
    assert "Unchanged reference" not in json.dumps(manifest)
    assert manifest["record_sha256"] == json_hash(tokens["records"])


def test_reference_content_hash_and_full_context_are_checked_before_writing(setup):
    path = setup.practice / "gsm8k_train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned source"):
        prepared(setup)
    setup.expected["exp8_math/practice/gsm8k_train.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    setup.tokenizer.encode = lambda text, **kwargs: [1] * 34816
    with pytest.raises(ValueError, match="truncation is forbidden"):
        prepared(setup)
    assert not (setup.data / "damage").exists()


def test_four_edit_manifests_are_immutable_and_share_the_same_dataset(setup):
    manifests = prepared(setup)
    assert [value["edit"] for _, value in manifests] == ["p50c", "mechanical", "random", "live"]
    assert len({json_hash(value["dataset"]) for _, value in manifests}) == 1
    assert len({value["edited_checkpoint"]["sha256"] for _, value in manifests}) == 4
    assert all(value["edit_bundle"]["edit_sha256"] is None for _, value in manifests)
    assert all(value["edit_bundle"]["edit_address_format"] == "legacy_unversioned" for _, value in manifests)
    assert prepared(setup) == manifests
    path, manifest = manifests[0]
    loaded, records = stage.load_measurement_inputs(path, json_hash(manifest),
        base_checkpoint=setup.data / "original", tokenizer=setup.tokenizer)
    assert loaded == manifest and len(records) == 512
    (setup.data / "p50c/model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Checkpoint bytes changed"):
        stage.load_measurement_inputs(path, json_hash(manifest), base_checkpoint=setup.data / "original", tokenizer=setup.tokenizer)
    with pytest.raises(ValueError, match="conflicting immutable identity"):
        prepared(setup)


def test_versioned_edit_address_is_distinct_from_exported_bundle_bytes(setup):
    from openwsp.edits import edit_sha256

    path = setup.data / "controls/mechanical/edit/manifest.json"
    manifest = {"format": "openwsp.edit", "format_version": 1, "tensors": [], "payload_sha256": "b" * 64}
    manifest["edit_sha256"] = edit_sha256(manifest)
    atomic_write(path, manifest)
    _, paired = prepared(setup)[1]
    bundle = paired["edit_bundle"]
    assert bundle["edit_sha256"] == manifest["edit_sha256"]
    assert bundle["sha256"] != bundle["edit_sha256"] and bundle["edit_address_format"] == "openwsp.edit.v1"
    atomic_write(path, {**manifest, "edit_sha256": "0" * 64})
    with pytest.raises(ValueError, match="manifest address"):
        prepared(setup)


def test_tokenizer_edit_and_manifest_mismatches_stop_before_model_calls(setup):
    path, manifest = prepared(setup)[0]
    with pytest.raises(ValueError, match="manifest hash"):
        stage.load_measurement_inputs(path, "0" * 64, base_checkpoint=setup.data / "original", tokenizer=setup.tokenizer)
    setup.tokenizer.chat_template = "changed"
    with pytest.raises(ValueError, match="tokenizer binding"):
        stage.load_measurement_inputs(path, json_hash(manifest), base_checkpoint=setup.data / "original", tokenizer=setup.tokenizer)
    setup.tokenizer.chat_template = Tokenizer.chat_template
    (setup.data / "reference_edit/edit.npz").write_bytes(b"changed edit")
    with pytest.raises(ValueError, match="Edit bytes changed"):
        stage.load_measurement_inputs(path, json_hash(manifest), base_checkpoint=setup.data / "original", tokenizer=setup.tokenizer)


def metrics(records, kl=0):
    count = sum(sum(record["score_mask"]) for record in records)
    return {"token_count": count, "kl_sum": kl * count, "kl_mean": kl,
            "sequence_count": len(records), "scored_sequence_count": len(records),
            "input_token_count": sum(len(record["input_ids"]) for record in records), "chunk_tokens": 32}


def test_identity_smoke_precedes_edited_load_and_is_not_merged_into_measurement(monkeypatch, setup):
    from openwsp.backends import damage
    path, manifest = prepared(setup)[0]
    base, edited, calls = object(), object(), []

    def measure(left, right, rows, **kwargs):
        calls.append((left is right, len(rows)))
        return metrics(rows, 0 if left is right else 0.25)

    def load(checkpoint):
        assert calls == [(True, 4)] and checkpoint == str(setup.data / "p50c")
        return edited

    monkeypatch.setattr(damage, "teacher_forced_kl", measure)
    result = stage.measure_manifest(base, setup.tokenizer, path, json_hash(manifest),
                                   base_checkpoint=setup.data / "original", load_edited=load)
    assert calls == [(True, 4), (False, 512)]
    assert result["measurement"]["sequence_count"] == 512
    assert result["measurement"]["kl_mean"] == 0.25 and result["smoke"]["included_in_measurement"] is False
    monkeypatch.setattr(damage, "teacher_forced_kl", lambda a, b, rows, **kwargs: metrics(rows, 0.1))
    with pytest.raises(ValueError, match="smoke failed"):
        stage.measure_manifest(base, setup.tokenizer, path, json_hash(manifest),
                               base_checkpoint=setup.data / "original", load_edited=lambda path: pytest.fail("Must not load edited model"))


def test_lost_result_resume_uses_same_existing_job_identity_and_never_remeasures(monkeypatch, setup):
    import transformers
    manifests = prepared(setup)
    monkeypatch.setattr(stage, "prepare_manifests", lambda *args, **kwargs: manifests)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: setup.tokenizer)
    module = ModuleType("openwsp.backends.modal_app")
    module.TransformerReader = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls, saved, commits = [], {}, []

    def instance(cls, gpu, gate, **kwargs):
        assert cls is module.TransformerReader and gpu == "H100" and gate == "test-gate"
        assert kwargs == {"checkpoint": str(setup.data / "original")}

        def spawn(path, manifest_sha):
            assert commits and all(path.exists() for path, _ in manifests)
            manifest = json.loads(Path(path).read_text())
            assert manifest_sha == json_hash(manifest)
            rows = json.loads((Path(path).parent / "references.tokens.json").read_text())["records"]
            return {"status": "done", "manifest_sha256": manifest_sha, "measurement": metrics(rows, 0.2),
                    "smoke": {**metrics([rows[i] for i in stage.SMOKE_INDICES]), "included_in_measurement": False,
                              "reference_indices": stage.SMOKE_INDICES},
                    "container_seconds": 30, "gpu": "H100", "_openwsp_gpu": {"billable_seconds": 152,
                                                                                "total_settled_cost_usd": "0.18"}}

        return SimpleNamespace(damage=SimpleNamespace(spawn=spawn))

    monkeypatch.setattr(stage, "gpu_instance", instance)

    class Runner:
        def launch(self, key, spawn_for, *, fingerprint):
            calls.append((key, fingerprint))
            if key not in saved:
                saved[key] = spawn_for("H100", "test-gate")
            assert fingerprint == {"manifest_sha256": saved[key]["manifest_sha256"]}

        def result(self, key, **kwargs):
            return saved[key]

    monkeypatch.setattr(stage, "ProtocolGPUJobs", lambda *args, **kwargs: Runner())
    checkpoint = lambda: commits.append(True)
    result = stage.run_damage_stage(object(), checkpoint, identity=setup.identity, data=setup.data, practice=setup.practice)
    assert result["status"] == "done" and len(calls) == 4 and len(commits) >= 5
    assert all(row["cost_usd"] == 0.18 and row["sequence_count"] == 512 for row in result["edits"].values())
    target = setup.data / "results/damage" / setup.identity.relative / "live/summary.json"
    target.unlink()  # Simulate losing local completion persistence, not the durable provider receipt.
    repeated = stage.run_damage_stage(object(), checkpoint, identity=setup.identity, data=setup.data, practice=setup.practice)
    assert repeated == result and len(calls) == 5 and len(saved) == 4
    assert calls[-1] == calls[-2]
    corrupt = json.loads(target.read_text())
    corrupt["kl_mean"] = 0.75
    atomic_write(target, corrupt)
    with pytest.raises(ValueError, match="exact registered"):
        stage.run_damage_stage(object(), checkpoint, identity=setup.identity, data=setup.data, practice=setup.practice)
    assert len(calls) == 5  # Corrupt completed evidence cannot trigger resampling.


def test_modal_method_preserves_sdpa_and_existing_resource_accounting(monkeypatch):
    import transformers
    source = Path(__file__).resolve().parents[1] / "src/openwsp/backends/modal_app.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "TransformerReader")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "damage")
    namespace = {"modal": SimpleNamespace(method=lambda: lambda function: function), "time": time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    calls = []
    edited = SimpleNamespace(to=lambda device: calls.append(("device", device)) or SimpleNamespace(eval=lambda: "edited"))
    monkeypatch.setattr(transformers.Qwen3_5ForConditionalGeneration, "from_pretrained",
                        lambda path, **kwargs: calls.append((path, kwargs)) or edited)
    worker = SimpleNamespace(load_error=None, eager=False, checkpoint="/data/original", model=SimpleNamespace(device="cpu"),
                             tokenizer=object(), gpu_kind="H100", _seconds=lambda: 123)

    def measure(base, tokenizer, path, digest, *, base_checkpoint, load_edited):
        assert base is worker.model and tokenizer is worker.tokenizer
        assert (path, digest, base_checkpoint) == ("/private/manifest", "a" * 64, "/data/original")
        assert load_edited("/data/edited") == "edited"
        return {"status": "done", "measurement": {"kl_mean": 0.2}}

    monkeypatch.setattr(stage, "measure_manifest", measure)
    result = namespace["damage"](worker, "/private/manifest", "a" * 64)
    assert result["container_seconds"] == 123 and result["gpu"] == "H100"
    assert calls[0][1]["local_files_only"] is True and calls[0][1]["attn_implementation"] == "sdpa"
    worker.eager = True
    with pytest.raises(ValueError, match="sdpa"):
        namespace["damage"](worker, "/private/manifest", "a" * 64)
    assert len(calls) == 2


def test_pipeline_damage_stage_resolves_saved_active_identity_without_running_other_stages(monkeypatch, setup):
    from openwsp.budget import BudgetLedger
    from replications.exp8 import pipeline

    atomic_write(setup.data / "state.json", {"stages": {"prepare": {"status": "done"}}})
    atomic_write(setup.data / "active-live.json", {"editor": setup.identity.editor, "run_id": setup.identity.run_id})
    module = ModuleType("openwsp.backends.modal_app")
    module.data_volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    module.prepare = module.Generator = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(pipeline, "Path", lambda value: setup.data if value == "/data" else Path(value))
    monkeypatch.setattr(pipeline, "BudgetLedger", lambda path: BudgetLedger(path))
    monkeypatch.setattr(pipeline, "ProtocolGPUJobs", lambda *args, **kwargs: object())
    calls = []
    monkeypatch.setattr(stage, "run_damage_stage", lambda ledger, checkpoint, **kwargs: calls.append(kwargs) or {"status": "done"})
    result = pipeline.run("damage", {}, app=object())
    assert calls == [{"identity": setup.identity}]
    assert result["status"] == "stage_done" and result["stages"]["damage"] == {"status": "done"}
    assert set(result["stages"]) == {"prepare", "damage"}
