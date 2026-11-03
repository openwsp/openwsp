import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from openwsp.backends.forward_bank import GenerationBank, json_hash
from openwsp.backends.jobs import JobPending
from replications.exp8 import evaluation
from replications.exp8.runtime import CONFIG, write_rows


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        return f"thinking={enable_thinking}:" + messages[0]["content"]


@pytest.fixture
def setup(monkeypatch, tmp_path):
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(evaluation, "checkpoint_hash", lambda path: "a" * 64)
    monkeypatch.setattr(evaluation, "Path", lambda value: tmp_path / str(value).removeprefix("/data/"))
    (tmp_path / "benchmarks").mkdir()
    rows = [{"id": str(i), "problem": f"{i}+1?", "answer": "1"} for i in range(3)]
    write_rows(tmp_path / "benchmarks/math500.jsonl", rows)
    module = ModuleType("openwsp.backends.modal_app")
    module.Generator, module.TransformerReader = object(), object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls, jobs, instances = [], {}, []

    def response(selected):
        return {"outputs": [{"id": row["id"], "samples": [
            {"text": r"\boxed{1}", "token_count": 3, "finish_reason": "stop"}
        ]} for row in selected], "container_seconds": 10, "gpu": "H100",
            "engine": CONFIG["engine"], "transformers": CONFIG["transformers"],
            "_openwsp_gpu": {"billable_seconds": 12, "total_settled_cost_usd": 5}}

    def instance(cls, gpu, gate, **settings):
        instances.append(settings)
        def spawn(selected, params, thinking, **options):
            calls.append([row["id"] for row in selected])
            return response(selected)
        return SimpleNamespace(generate=SimpleNamespace(spawn=spawn))

    class Jobs:
        def __init__(self, *args, **kwargs):
            pass

        def has_job(self, key):
            return key in jobs

        def launch(self, key, spawn, *, fingerprint):
            if key in jobs:
                assert jobs[key]["fingerprint"] in (None, fingerprint)
            else:
                jobs[key] = {"fingerprint": fingerprint, "result": spawn("H100", "gate")}

        def result(self, key, timeout):
            return jobs[key]["result"]

    monkeypatch.setattr(evaluation, "ProtocolGPUJobs", Jobs)
    monkeypatch.setattr(evaluation, "gpu_instance", instance)
    return SimpleNamespace(root=tmp_path, rows=rows, calls=calls, jobs=jobs, response=response, instances=instances)


def run(name, **kwargs):
    return evaluation.evaluate(name, "/data/original", "math500", 1, object(), lambda: None, **kwargs)


def test_base_identity_matches_live_bank_and_distinguishes_effective_inputs(setup, monkeypatch):
    params = {**CONFIG["sampling"], "n": 1}
    options = {"seed_by_id": True}
    row = setup.rows[0]
    base = evaluation._base_generation_keys([row], "/data/original", params, True, {}, options)[0]
    prompt = Tokenizer().apply_chat_template(
        [{"content": row["problem"] + "\n" + evaluation.INSTRUCTION}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % 2**31
    assert base == GenerationBank.identity(row["id"], prompt, {
        "sampling": params, "thinking": True, "engine": CONFIG["engine"],
        "transformers": CONFIG["transformers"]}, "a" * 64, seed)
    assert evaluation._base_generation_keys([row], "/data/original", params, True,
        {"max_model_len": 34816, "max_num_seqs": 128}, options)[0] == base
    alternatives = [
        ([{**row, "problem": "different"}], params, True, {}, options),
        ([row], {**params, "n": 2}, True, {}, options),
        ([row], params, False, {}, options),
        ([row], params, True, {"max_num_seqs": 32}, options),
        ([row], params, True, {}, {**options, "decoding_policy": {"kind": "close"}}),
        ([row], params, True, {}, {"seed_by_id": False}),
    ]
    for rows, sampling, thinking, settings, generation in alternatives:
        assert evaluation._base_generation_keys(rows, "/data/original", sampling, thinking, settings, generation)[0] != base
    monkeypatch.setattr(evaluation, "checkpoint_hash", lambda path: "b" * 64)
    assert evaluation._base_generation_keys([row], "/data/original", params, True, {}, options)[0] != base


def test_partial_and_complete_reuse_launch_only_missing_scored_rows_and_charge_once(setup):
    first = run("first", problem_limit=2)
    assert setup.calls == [["0", "1"], ["0", "1"]]  # Smoke never seeds scoring cache.
    second = run("second")
    assert setup.calls[-2:] == [["0", "1", "2"], ["2"]]
    assert second["provenance"]["generation_bank"]["reused_rows"] == 2
    assert second["provenance"]["generation_bank"]["generated_rows"] == 1
    assert second["cost_usd"] == first["cost_usd"] == 5
    third = run("third")
    assert setup.calls[-1] == ["0", "1", "2"] and len(setup.calls) == 5  # Only new smoke.
    assert third["seconds"] == third["cost_usd"] == 0
    assert third["provenance"]["smoke_cost_usd"] == 5
    assert third["provenance"]["generation_bank"]["reused_rows"] == 3
    assert third["n"] == third["correct"] == 3
    sources = third["provenance"]["generation_bank"]["batch_provenance"][0]["rows"]
    assert [row["source"]["stage"] for row in sources] == ["first", "first", "second"]
    assert all(row["reused"] for row in sources)


def test_resume_keeps_original_misses_even_after_their_results_enter_bank(setup, monkeypatch):
    run("first", problem_limit=2)
    real = evaluation.evaluate_outputs
    def fail_after_bank(*args, **kwargs):
        if kwargs["directory"].name == "batch_000":
            raise RuntimeError("grading interrupted")
        return real(*args, **kwargs)
    monkeypatch.setattr(evaluation, "evaluate_outputs", fail_after_bank)
    with pytest.raises(RuntimeError, match="grading interrupted"):
        run("partial")
    calls_before = list(setup.calls)
    plan = json.loads((setup.root / "results/partial/math500/batch_000/generation_plan.json").read_text())
    assert plan["missing"] == [2]
    monkeypatch.setattr(evaluation, "evaluate_outputs", real)
    result = run("partial")
    assert setup.calls == calls_before
    assert result["cost_usd"] == 5 and result["provenance"]["generation_bank"]["generated_rows"] == 1


def test_disappearing_planned_hit_blocks_resume_without_a_replacement_job(setup, monkeypatch):
    run("first", problem_limit=2)
    real = evaluation.evaluate_outputs
    def fail_scoring(*args, **kwargs):
        if kwargs["directory"].name == "batch_000":
            raise RuntimeError("grading interrupted")
        return real(*args, **kwargs)
    monkeypatch.setattr(evaluation, "evaluate_outputs", fail_scoring)
    with pytest.raises(RuntimeError):
        run("partial")
    plan = json.loads((setup.root / "results/partial/math500/batch_000/generation_plan.json").read_text())
    (setup.root / "forward-bank/base" / f"{json_hash(plan['keys'][0])}.json").unlink()
    monkeypatch.setattr(evaluation, "evaluate_outputs", real)
    before = list(setup.calls)
    with pytest.raises(JobPending, match="missing or changed"):
        run("partial")
    assert setup.calls == before


def test_multiple_batches_aggregate_cache_counts_without_recharging_source_cost(setup):
    settings = {"engine_settings": {"max_num_seqs": 2}}
    first = run("small_batches", **settings)
    assert first["cost_usd"] == 10
    assert first["provenance"]["generation_bank"]["generated_rows"] == 3
    second = run("cached_batches", **settings)
    assert second["cost_usd"] == 0
    bank = second["provenance"]["generation_bank"]
    assert bank["reused_rows"] == 3 and bank["generated_rows"] == 0
    assert len(bank["batch_provenance"]) == 2
    json.dumps(second, allow_nan=False)


@pytest.mark.parametrize("corruption", ["bad_hash", "wrong_n", "incomplete"])
def test_invalid_bank_blocks_before_smoke_and_never_resamples(setup, corruption):
    run("first")
    path = next((setup.root / "forward-bank/base").glob("*.json"))
    record = json.loads(path.read_text())
    if corruption == "bad_hash":
        record["output"]["samples"][0]["text"] = "changed"
    elif corruption == "wrong_n":
        record["output"]["samples"] *= 2
        record["output_sha256"] = json_hash(record["output"])
    else:
        record["output"]["samples"][0]["finish_reason"] = "error"
        record["output_sha256"] = json_hash(record["output"])
    path.write_text(json.dumps(record))
    before = list(setup.calls)
    with pytest.raises(JobPending):
        run("blocked")
    assert setup.calls == before


def test_preexisting_unbound_job_keeps_rows_and_is_not_imported_into_bank(setup):
    key = "evaluation/historical/math500/batch_000"
    setup.jobs[key] = {"fingerprint": None, "result": setup.response(setup.rows)}
    result = run("historical")
    assert setup.calls == [["0", "1", "2"]]  # Smoke only; saved job is reused as its own receipt.
    assert not list((setup.root / "forward-bank/base").glob("*.json"))
    batch = result["provenance"]["generation_bank"]["batch_provenance"][0]
    assert batch["historical_job_not_published"] and batch["reused_rows"] == 0


def test_nonnull_policy_alone_enables_thinking_close(setup):
    policy = {"kind": "thinking_token_cap", "thinking_tokens": 8192}
    run("plain")
    assert all("enable_thinking_close" not in settings for settings in setup.instances)
    setup.instances.clear()
    run("forced", generation_options={"seed_by_id": True, "decoding_policy": policy})
    assert setup.instances and all(settings["enable_thinking_close"] is True for settings in setup.instances)
