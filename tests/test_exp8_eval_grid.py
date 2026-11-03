import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from replications.exp8 import eval_grid, evaluation
from replications.exp8.run_identity import LiveIdentity
from replications.exp8.runtime import CONFIG, atomic_write, write_rows


def test_preregistered_grid_only_changes_named_factors():
    baseline = {k: v for k, v in CONFIG["sampling"].items() if k != "seed"}
    cells = {name: eval_grid.load_cell(name).definition for name in eval_grid.STAGES}
    assert len(cells) == 11
    for model in ("base", "p50c", "live"):
        off = cells[f"grid_p3_{model}"]
        assert off["thinking"] is False and off["sampling"] == baseline
        assert off["decoding_policy"] is None
        close = cells[f"grid_p4_{model}"]
        assert close["thinking"] is True and close["sampling"] == baseline
        assert close["decoding_policy"] == {"kind": "thinking_token_cap", "thinking_tokens": 8192}
    for model in ("base", "p50c"):
        long = cells[f"grid_p1_{model}"]
        assert long["thinking"] is True and long["sampling"] == {**baseline, "max_tokens": 81920}
        assert long["engine_settings"] == {"max_model_len": 83968, "max_num_seqs": 32}
    assert cells["grid_p5_presence_base"]["sampling"] == {**baseline, "presence_penalty": 2.0}
    assert cells["grid_p5_repetition_base"]["sampling"] == {**baseline, "repetition_penalty": 1.2}
    assert cells["grid_p5_repeat_close_base"]["sampling"] == baseline
    assert all(cell["engine_settings"] == {"max_model_len": 34816, "max_num_seqs": 128}
               for cell in cells.values() if cell["protocol"] != "P1")


def test_seed_preserves_existing_digest_prefix_and_is_stable_across_cells():
    value = "math500/test/17"
    expected = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % (2**31)
    assert eval_grid.problem_seed(value) == expected
    assert eval_grid.problem_seed(value) != int(hashlib.sha256(value.encode()).hexdigest(), 16) % (2**31)
    assert all(eval_grid.load_cell(name).definition["seed_scheme"] == eval_grid.SEED_SCHEME
               for name in eval_grid.STAGES)


def test_forced_close_counts_generated_tokens_then_allows_answer():
    state = eval_grid.ThinkingCloseState({"kind": "thinking_token_cap", "thinking_tokens": 8192}, 99, max_tokens=32768)
    assert state.observe_many([1] * 8191) is None
    assert state.observe(2) == "thinking_token_cap"
    assert state.generated_tokens == 8192
    assert state.observe(99) is None
    assert state.generated_tokens == 8193 and state.closed
    assert state.observe_many([3] * 24575) is None
    assert state.generated_tokens == 32768
    with pytest.raises(ValueError, match="output cap"):
        state.observe(4)


def test_natural_close_disables_both_interventions():
    for policy in ({"kind": "thinking_token_cap", "thinking_tokens": 3},
                   {"kind": "repeat_ngram", "ngram_size": 2, "repeats": 3, "window_tokens": 8}):
        state = eval_grid.ThinkingCloseState(policy, 99, max_tokens=20)
        assert state.observe_many([1, 99, 1, 1, 1, 1, 1]) is None
        assert state.closed and state.generated_tokens == 7


def test_no_close_can_extend_total_output_cap():
    state = eval_grid.ThinkingCloseState({"kind": "thinking_token_cap", "thinking_tokens": 3}, 99, max_tokens=3)
    assert state.observe_many([1, 2, 3]) is None
    assert not state.closed


def test_eightgram_counts_five_occurrences_including_first_and_overlaps():
    policy = {"kind": "repeat_ngram", "ngram_size": 8, "repeats": 5, "window_tokens": 2048}
    state = eval_grid.ThinkingCloseState(policy, 99, max_tokens=32768)
    assert state.observe_many(list(range(1, 9)) * 4 + list(range(1, 8))) is None
    assert state.observe(8) == "repeat_ngram"
    overlapping = eval_grid.ThinkingCloseState(policy, 99, max_tokens=32768)
    assert overlapping.observe_many([1] * 11) is None
    assert overlapping.observe(1) == "repeat_ngram"


def test_repeat_window_excludes_old_tokens_and_cross_boundary_ngrams():
    policy = {"kind": "repeat_ngram", "ngram_size": 2, "repeats": 3, "window_tokens": 6}
    state = eval_grid.ThinkingCloseState(policy, 99, max_tokens=100)
    # The third global occurrence is too late: its first occurrence was evicted.
    assert state.observe_many([1, 2, 1, 2, 3, 4, 1, 2]) is None
    assert len(state._grams) == 5
    assert state.observe_many([1, 2, 1, 2]) == "repeat_ngram"
    assert len(state._grams) == 5


def test_repeat_policy_matches_reference_at_every_step():
    import random
    rng = random.Random(417)
    policy = {"kind": "repeat_ngram", "ngram_size": 3, "repeats": 4, "window_tokens": 17}
    for _ in range(40):
        state = eval_grid.ThinkingCloseState(policy, 99, max_tokens=201)
        generated = []
        for _ in range(200):
            generated.append(rng.randrange(2))
            tail = generated[-17:]
            grams = [tuple(tail[i:i + 3]) for i in range(len(tail) - 2)]
            triggered = bool(grams) and grams.count(grams[-1]) >= 4
            assert bool(state.observe(generated[-1])) == triggered
            if triggered:
                assert state.observe(99) is None
                break


@pytest.mark.parametrize("policy", [{}, {"kind": "thinking_token_cap", "thinking_tokens": True},
    {"kind": "thinking_token_cap", "thinking_tokens": 0},
    {"kind": "repeat_ngram", "ngram_size": 8, "repeats": 5, "window_tokens": 8},
    {"kind": "repeat_ngram", "ngram_size": 8, "repeats": 1, "window_tokens": 2048}])
def test_invalid_decoding_policies_fail_before_inference(policy):
    with pytest.raises(ValueError):
        eval_grid.validate_policy(policy)


def test_cell_rejects_implicit_context_and_missing_sampling(tmp_path):
    stage = "grid_p1_base"
    cell = eval_grid.load_cell(stage).definition
    cell["engine_settings"]["max_model_len"] = 34816
    atomic_write(tmp_path / f"{stage}.json", cell)
    with pytest.raises(ValueError, match="prompt allowance"):
        eval_grid.load_cell(stage, directory=tmp_path)
    cell["engine_settings"]["max_model_len"] = 83968
    del cell["sampling"]["top_p"]
    cell["sampling"]["repetition_penalty"] = 1.2
    atomic_write(tmp_path / f"{stage}.json", cell)
    with pytest.raises(ValueError, match="Missing think35"):
        eval_grid.load_cell(stage, directory=tmp_path)


def write_prerequisites(data, identity):
    submission = {"sha256": "b" * 64, "checkpoint": "/data/final-checkpoint"}
    atomic_write(data / identity.relative / "state.json", {"done": True, "passes": 50, "submission": submission})
    for benchmark, samples in CONFIG["samples"].items():
        atomic_write(data / "results" / identity.evaluation_stage / benchmark / "summary.json",
                     {"status": "done", "samples_per_problem": samples,
                      "provenance": {"submission_sha256": submission["sha256"]}})
    atomic_write(data / "results/control4_random/math500/summary.json", {"status": "done"})
    return submission


def test_priority_blocks_missing_submission_final_evals_control_and_prior_cell(tmp_path):
    identity = LiveIdentity("codex", "codex-20261103")
    with pytest.raises(ValueError, match="50-pass"):
        eval_grid.require_grid_priority("grid_p3_base", identity=identity, data=tmp_path)
    submission = write_prerequisites(tmp_path, identity)
    path = tmp_path / "results" / identity.evaluation_stage / "math500/summary.json"
    original = json.loads(path.read_text())
    atomic_write(path, {**original, "provenance": {"submission_sha256": "wrong"}})
    with pytest.raises(ValueError, match="final evaluations"):
        eval_grid.require_grid_priority("grid_p3_base", identity=identity, data=tmp_path)
    atomic_write(path, original)
    control = tmp_path / "results/control4_random/math500/summary.json"
    control.unlink()
    with pytest.raises(ValueError, match="control4"):
        eval_grid.require_grid_priority("grid_p3_base", identity=identity, data=tmp_path)
    atomic_write(control, {"status": "done"})
    assert eval_grid.require_grid_priority("grid_p3_base", identity=identity, data=tmp_path) == submission
    with pytest.raises(ValueError, match="grid_p3_base"):
        eval_grid.require_grid_priority("grid_p3_p50c", identity=identity, data=tmp_path)
    atomic_write(tmp_path / "results/grid_p3_base/math500/summary.json",
                 {"status": "done", "provenance": {"eval_grid": eval_grid.load_cell("grid_p3_base").provenance,
                                                    "faithful_submission_sha256": submission["sha256"]}})
    assert eval_grid.require_grid_priority("grid_p3_p50c", identity=identity, data=tmp_path) == submission


def test_grid_defaults_to_active_attempt_but_explicit_identity_wins(tmp_path):
    assert eval_grid.grid_live_identity({}, data=tmp_path) == LiveIdentity()
    atomic_write(tmp_path / "active-live.json", {"editor": "codex", "run_id": "codex-20261103"})
    assert eval_grid.grid_live_identity({}, data=tmp_path) == LiveIdentity("codex", "codex-20261103")
    assert eval_grid.grid_live_identity({"publish_results": False}, data=tmp_path).run_id == "codex-20261103"
    assert eval_grid.grid_live_identity({"editor": "openai"}, data=tmp_path) == LiveIdentity()


def test_run_cell_passes_seed_policy_context_and_live_identity(monkeypatch, tmp_path):
    identity = LiveIdentity("codex", "codex-20261103")
    submission = {"checkpoint": "/data/final", "sha256": "a" * 64}
    monkeypatch.setattr(eval_grid, "require_grid_priority", lambda *a, **kw: submission)
    called = []
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kwargs: called.append((args, kwargs)) or {"status": "done"})
    assert eval_grid.run_grid_stage("grid_p4_live", object(), lambda: None, identity=identity, data=tmp_path) == {"status": "done"}
    args, kwargs = called[0]
    assert args[:4] == ("grid_p4_live", "/data/final", "math500", 1)
    assert kwargs["generation_options"] == {"seed_by_id": True, "seed_scheme": eval_grid.SEED_SCHEME,
                                            "decoding_policy": {"kind": "thinking_token_cap", "thinking_tokens": 8192}}
    assert kwargs["extra_provenance"]["submission_sha256"] == submission["sha256"]


def test_grid_evaluation_binds_provenance_preserves_options_and_aggregates_finished(monkeypatch, tmp_path):
    (tmp_path / "benchmarks").mkdir()
    write_rows(tmp_path / "benchmarks/math500.jsonl", [{"id": str(i), "problem": "1?", "answer": "1"} for i in range(3)])
    monkeypatch.setattr(evaluation, "Path", lambda value: tmp_path / str(value).removeprefix("/data/"))
    module = ModuleType("openwsp.backends.modal_app")
    module.Generator, module.TransformerReader = object(), object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls, results = [], {}

    def instance(cls, gpu, gate, **kwargs):
        assert kwargs == {"checkpoint": "/data/original", "max_model_len": 83968, "max_num_seqs": 2}

        def spawn(rows, params, thinking, **options):
            calls.append((rows, params, thinking, options))
            return {"outputs": [{"id": row["id"], "samples": [{"text": r"\boxed{1}", "token_count": 3,
                "finish_reason": "length" if row["id"] == "0" else "stop"}]} for row in rows],
                "container_seconds": 1, "gpu": gpu, "engine": "vllm", "transformers": "test"}
        return SimpleNamespace(generate=SimpleNamespace(spawn=spawn))

    class Jobs:
        def __init__(self, *args, **kwargs):
            pass

        def launch(self, key, spawn_for, *, fingerprint):
            assert fingerprint["generation_options"]["seed_scheme"] == eval_grid.SEED_SCHEME
            results[key] = spawn_for("H100", "gate")

        def result(self, key, timeout):
            return results[key]

    monkeypatch.setattr(evaluation, "ProtocolGPUJobs", Jobs)
    monkeypatch.setattr(evaluation, "gpu_instance", instance)
    kwargs = {"engine_settings": {"max_model_len": 83968, "max_num_seqs": 2},
              "generation_options": {"seed_by_id": True, "seed_scheme": eval_grid.SEED_SCHEME, "decoding_policy": None},
              "extra_provenance": {"eval_grid": {"config_sha256": "a" * 64}}, "changes": {"max_tokens": 81920}}
    result = evaluation.evaluate("test", "/data/original", "math500", 1, object(), lambda: None, **kwargs)
    assert [len(call[0]) for call in calls] == [3, 2, 1]  # Smoke, then scored batches.
    assert all(call[1]["max_tokens"] == 81920 and call[3] == kwargs["generation_options"] for call in calls)
    assert result["n"] == result["correct"] == 3
    assert result["finished_n"] == result["finished_correct"] == 2
    assert result["accuracy_finished"] == 1 and result["cut_share"] == pytest.approx(1 / 3)
    assert result["provenance"]["eval_grid"] == kwargs["extra_provenance"]["eval_grid"]
    assert result["provenance"]["batch_generations"] == 2
    assert evaluation.evaluate("test", "/data/original", "math500", 1, object(), lambda: None, **kwargs) == result
    with pytest.raises(ValueError, match="different specification"):
        evaluation.evaluate("test", "/data/changed", "math500", 1, object(), lambda: None, **kwargs)


def test_tiny_validation_is_not_a_model_benchmark():
    result = eval_grid.tiny_validation()
    assert result["status"] == "validated"
    assert result["provenance"]["synthetic"] is True
    assert result["provenance"]["model_calls"] == result["provenance"]["gpu_calls"] == 0
    assert result["checks_passed"] == len(result["checks"])
    assert "accuracy" not in result and "cost_usd" not in result
