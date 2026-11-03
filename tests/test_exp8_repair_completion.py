import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from openwsp.budget import atomic_json
from openwsp.core import WeightStore
from replications.exp8 import repair_completion as completion
from replications.exp8.repair import DOSES, _file_hash, _json_hash, build_repair
from replications.exp8.repair_activations import select_examples


@pytest.fixture
def extraction(tmp_path, monkeypatch):
    counts = {"looping": 1, "finished": 1}
    monkeypatch.setattr(completion, "EXPECTED_COUNTS", counts)
    model = tmp_path / ".openwsp/research-model"
    model.mkdir(parents=True)
    rng = np.random.default_rng(71)
    names = [f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in (15, 19)]
    weights = {name: torch.from_numpy(rng.normal(size=(3, 70)).astype(np.float32)) for name in names}
    save_file(weights, model / "model.safetensors")
    (model / "config.json").write_text("{}")
    inputs = tmp_path / ".openwsp/repair-inputs"
    inputs.mkdir()
    raw = [{"problem_id": "l", "finish_reason": "length", "text": "a b c d e f g h " * 180},
           {"problem_id": "f", "finish_reason": "stop", "text": "finished"}]
    benchmarks = [{"id": "l", "problem": "loop question"}, {"id": "f", "problem": "done question"}]
    for filename, rows in (("raw_generations.jsonl", raw), ("math500.jsonl", benchmarks)):
        (inputs / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    method_path = tmp_path / ".openwsp/research-control2-method.json"
    method = {"counts": counts, "protocol": "synthetic CPU fixture", "layers": {
        str(layer): {"direction": [1, 0, 0]} for layer in (15, 19)}}
    atomic_json(method_path, method)
    atomic_json(tmp_path / ".openwsp/research-prepare.json", {"model_revision": "synthetic-pin"})
    examples = select_examples(raw, benchmarks, counts)
    identities = [{key: row[key] for key in ("id", "group", "prompt_sha256", "text_sha256")} for row in examples]
    provenance = {
        "raw_sha256": _file_hash(inputs / "raw_generations.jsonl"),
        "benchmarks_sha256": _file_hash(inputs / "math500.jsonl"),
        "saved_method_sha256": _file_hash(method_path),
        "checkpoint_files": {"model.safetensors": _file_hash(model / "model.safetensors")},
        "model_assets": {"config.json": _file_hash(model / "config.json")},
    }
    directory = tmp_path / ".openwsp/repair-input-means"
    selection = {"examples": identities, "layers": [15, 19], "provenance": provenance,
                 "activation_reduction": "synthetic per-example means"}
    atomic_json(directory / "selection.json", selection)
    means = []
    for index, identity in enumerate(identities):
        key = _json_hash({"binding": _json_hash(selection), "example": identity})
        folder = directory / "examples" / key
        folder.mkdir(parents=True)
        values = {str(layer): rng.normal(size=70).astype(np.float32) for layer in (15, 19)}
        means.append(values)
        np.savez_compressed(folder / "means.npz", **values)
        atomic_json(folder / "receipt.json", {
            "request_sha256": key, "example": identity, "arrays_sha256": _file_hash(folder / "means.npz"),
            "tokens": {"prompt_tokens": 3, "answer_tokens": index + 2, "input_ids_sha256": "a" * 64},
        })
    differences = {name: means[0][str(layer)].astype(np.float64) - means[1][str(layer)].astype(np.float64)
                   for layer, name in zip((15, 19), names, strict=True)}
    np.savez_compressed(directory / "input-differences.npz", **differences)
    atomic_json(directory / "summary.json", {
        "status": "complete", "processed": 2, "selected": 2, "counts": counts,
        "selection_sha256": _file_hash(directory / "selection.json"),
        "input_differences_sha256": _file_hash(directory / "input-differences.npz"),
    })
    (inputs / "full-extraction.exit").write_text("0\n")
    public = tmp_path / "replications/exp8/results/repair"
    ids = [str(i) for i in range(100)]
    dose = {"coefficients": list(DOSES), "base_variant": "2c", "problem_ids": ids,
            "ordered_ids_sha256": _json_hash(ids), "fixed_directions_and_columns": True,
            "selection_source_sha256": "b" * 64, "samples_per_problem": 1,
            "sampling": "think35 unchanged", "seed_scheme": "sha256_first8_utf8_problem_id_mod_2**31"}
    atomic_json(public / "dose-sweep.json", dose)
    atomic_json(public / "summary.json", {"variants": {"2c": {"accuracy": None}}})
    atomic_json(inputs / "completion-binding.json", {
        "pid": 123, "selection_sha256": _file_hash(directory / "selection.json"),
        "dose_spec_sha256": completion._dose_identity(dose),
    })
    for variant, alpha in (("2a", 1), ("2b", 6)):
        build_repair(WeightStore(model), tmp_path / f".openwsp/repair-build/{variant}",
            directions={name: [1, 0, 0] for name in names}, alpha=alpha, column_rule="all",
            provenance={"source": "synthetic", "activation_reduction": "synthetic",
                        "saved_method_sha256": _file_hash(method_path)})
    return tmp_path, directory, inputs, differences


def verify(extraction):
    _, directory, inputs, _ = extraction
    return completion.verify_extraction(directory, inputs / "full-extraction.exit",
        expected_counts={"looping": 1, "finished": 1}, widths={15: 70, 19: 70})


def test_verifier_rederives_every_input_mean_and_matches_complete_artifact(extraction):
    differences, receipt = verify(extraction)
    for name, values in differences.items():
        np.testing.assert_array_equal(values, extraction[3][name])
    assert receipt["verified_examples"] == 2
    assert receipt["counts"] == {"looping": 1, "finished": 1}


@pytest.mark.parametrize("corruption", ["exit", "missing", "bytes", "contrast", "counts", "extra"])
def test_verifier_refuses_incomplete_tampered_or_inconsistent_inputs(extraction, corruption):
    _, directory, inputs, _ = extraction
    folder = next((directory / "examples").iterdir())
    if corruption == "exit":
        (inputs / "full-extraction.exit").write_text("137\n")
    elif corruption == "missing":
        (folder / "receipt.json").unlink()
    elif corruption == "bytes":
        (folder / "means.npz").write_bytes(b"changed")
    elif corruption == "extra":
        atomic_json(directory / "examples/extra/receipt.json", {})
    else:
        summary = json.loads((directory / "summary.json").read_text())
        if corruption == "contrast":
            changed = {name: values + 1 for name, values in extraction[3].items()}
            np.savez_compressed(directory / "input-differences.npz", **changed)
            summary["input_differences_sha256"] = _file_hash(directory / "input-differences.npz")
        else:
            summary["counts"] = {"looping": 2, "finished": 0}
        atomic_json(directory / "summary.json", summary)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(extraction)


def test_complete_builds_all_doses_only_after_verification_and_can_resume(extraction):
    root, _, _, _ = extraction
    first = completion.complete(root)
    assert first["built_doses"] == list(DOSES) and first["scoring"] == "not_run"
    public = root / "replications/exp8/results/repair"
    aggregate = json.loads((public / "summary.json").read_text())
    assert aggregate["variants"]["2c"]["accuracy"] is None
    assert aggregate["variants"]["2c"]["edit_sha256"] == first["edit_sha256"]
    dose = json.loads((public / "dose-sweep.json").read_text())
    assert [entry["alpha"] for entry in dose["edits"]] == list(DOSES)
    zero = json.loads((public / "dose/alpha_0/edit-manifest.json").read_text())
    assert zero["changed_tensors"] == 0
    assert completion.complete(root) == first


def test_invalid_inputs_do_not_publish_a_partial_2c_result(extraction):
    root, directory, _, _ = extraction
    (directory / "input-differences.npz").write_bytes(b"corrupt")
    public = root / "replications/exp8/results/repair"
    before = (public / "summary.json").read_bytes()
    with pytest.raises(ValueError):
        completion.complete(root)
    assert not (root / ".openwsp/repair-build/2c").exists()
    assert (public / "summary.json").read_bytes() == before and not (public / "2c").exists()


def test_scored_results_are_never_overwritten_by_cpu_continuation(extraction):
    root, _, _, _ = extraction
    completion.complete(root)
    public = root / "replications/exp8/results/repair"
    aggregate = json.loads((public / "summary.json").read_text())
    aggregate["variants"]["2c"]["accuracy"] = 0.5
    atomic_json(public / "summary.json", aggregate)
    before = {str(path): path.read_bytes() for path in public.rglob("*.json")}
    with pytest.raises(ValueError, match="scored 2c"):
        completion.complete(root)
    assert before == {str(path): path.read_bytes() for path in public.rglob("*.json")}


@pytest.mark.parametrize("replacement", [False, True])
def test_one_shot_wait_is_bound_to_original_process_and_never_restarts_it(extraction, monkeypatch, replacement):
    root, _, inputs, _ = extraction
    (inputs / "completion-binding.json").unlink()
    argv = ["python", "-m", "replications.exp8.repair_activations", "--threads", "2"]
    atomic_json(inputs / "full-extraction-invocation.json",
                {"pid": 123, "cpu_threads": 2, "command": " ".join(argv)})
    original = {"argv": argv, "start_ticks": "100"}
    changed = {"argv": argv, "start_ticks": "200"} if replacement else None
    states = iter([original, changed])
    monkeypatch.setattr(completion, "_process_identity", lambda pid: next(states))
    sleeps = []
    monkeypatch.setattr(completion.time, "sleep", sleeps.append)
    if replacement:
        with pytest.raises(ValueError, match="PID was replaced"):
            completion.wait_for_original(root, poll_seconds=1)
    else:
        result = completion.wait_for_original(root, poll_seconds=1)
        assert result["pid"] == 123 and result["process"] == original
    assert sleeps == [1]
