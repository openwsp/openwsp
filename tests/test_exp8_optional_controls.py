import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from replications.exp8 import evaluation, optional_controls
from replications.exp8.runtime import write_rows


def test_random_directions_are_fixed_independent_unit_vectors():
    first, second = optional_controls.random_directions(), optional_controls.random_directions()
    assert set(first) == {"15", "19"}
    for layer in first:
        assert torch.equal(first[layer], second[layer])
        assert first[layer].dtype == torch.float32
        assert first[layer].norm().item() == pytest.approx(1, abs=1e-6)
    assert abs(torch.dot(first["15"], first["19"]).item()) < 0.1


def test_reused_baseline_uses_fixed_problem_order_and_retains_raw_provenance(tmp_path):
    (tmp_path / "benchmarks").mkdir()
    base = tmp_path / "results/calibration_base"
    base.mkdir(parents=True)
    rows = [{"id": str(i)} for i in range(60)]
    records = [{"problem_id": str(i), "correct": i >= 40, "token_count": i + 1,
                "finish_reason": "length" if i < 30 else "stop",
                "raw_path": "/data/results/calibration_base/raw_generations.jsonl", "raw_sha256": "a" * 64}
               for i in reversed(range(60))]
    write_rows(tmp_path / "benchmarks/math500.jsonl", rows)
    write_rows(base / "samples.jsonl", records)
    (base / "summary.json").write_text(json.dumps({"n": 60, "accuracy": 1 / 3,
        "cost_usd": 8, "seconds": 100, "provenance": {"engine": "vllm"}}))
    result = optional_controls.reused_vllm_subset(tmp_path)
    assert result["n"] == 50 and result["correct"] == 10
    assert result["accuracy"] == 0.2 and result["cut_share"] == 0.6
    assert result["cost_usd"] == result["seconds"] == 0
    assert result["provenance"]["problem_ids"] == [str(i) for i in range(50)]
    assert result["provenance"]["reused"] is True
    assert "original cost" in result["provenance"]["cost_basis"]
    saved = [json.loads(line) for line in (tmp_path / "results/control5_vllm_reused/math500/samples.jsonl").read_text().splitlines()]
    assert [row["problem_id"] for row in saved] == [str(i) for i in range(50)]
    assert all(row["raw_sha256"] == "a" * 64 for row in saved)


def test_optional_controls_cannot_bypass_required_control_priority(monkeypatch, tmp_path):
    monkeypatch.setattr(optional_controls, "Path", lambda value: tmp_path / str(value).removeprefix("/data/"))
    monkeypatch.setattr(optional_controls, "random_edit", lambda _: pytest.fail("must not construct optional edit"))
    with pytest.raises(ValueError, match="all three required"):
        optional_controls.run_optional_control("control4", object(), lambda: None)


@pytest.mark.parametrize("engine", ["vllm", "transformers"])
def test_engine_evaluation_preserves_sampling_smoke_and_fixed_subset(monkeypatch, tmp_path, engine):
    (tmp_path / "benchmarks").mkdir()
    write_rows(tmp_path / "benchmarks/math500.jsonl", [{"id": str(i), "problem": "1?", "answer": "1"} for i in range(60)])
    monkeypatch.setattr(evaluation, "Path", lambda value: tmp_path / str(value).removeprefix("/data/"))
    module = ModuleType("openwsp.backends.modal_app")
    module.Generator, module.TransformerReader = object(), object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    from openwsp.backends.forward_bank import GenerationBank
    monkeypatch.setattr(evaluation, "_base_generation_keys", lambda rows, *args:
                        [GenerationBank.identity(row["id"], row["problem"], {}, "a" * 64, 0) for row in rows])
    calls, results = [], {}

    def instance(cls, gpu, gate, **kwargs):
        assert cls is (module.Generator if engine == "vllm" else module.TransformerReader)
        assert kwargs["checkpoint"] == "/data/original"

        def spawn(rows, params, thinking):
            calls.append((rows, params, thinking))
            return {"outputs": [{"id": row["id"], "samples": [{"text": r"\boxed{1}", "token_count": 3, "finish_reason": "stop"}]} for row in rows],
                    "container_seconds": 1, "gpu": gpu, "engine": evaluation.CONFIG["engine"] if engine == "vllm" else engine,
                    "transformers": evaluation.CONFIG["transformers"]}
        return SimpleNamespace(generate=SimpleNamespace(spawn=spawn))

    class Jobs:
        def __init__(self, *args, **kwargs):
            pass

        def launch(self, key, spawn_for, *, fingerprint):
            assert fingerprint.get("engine", "vllm") == engine
            results[key] = spawn_for("H100", "gate")

        def has_job(self, key):
            return key in results

        def result(self, key, timeout):
            return results[key]

    monkeypatch.setattr(evaluation, "ProtocolGPUJobs", Jobs)
    monkeypatch.setattr(evaluation, "gpu_instance", instance)
    result = evaluation.evaluate("test", "/data/original", "math500", 1, object(), lambda: None,
                                 engine=engine, problem_limit=50)
    assert [len(call[0]) for call in calls] == [4, 50]
    assert all(call[1] == {**evaluation.CONFIG["sampling"], "n": 1} and call[2] is True for call in calls)
    assert result["n"] == result["correct"] == 50
    assert result["provenance"]["smoke_included_in_score"] is False
    assert result["provenance"]["problem_ids"] == [str(i) for i in range(50)]
