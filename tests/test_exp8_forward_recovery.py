"""Forward receipt recovery with tiny fake providers; no remote calls or grading data."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openwsp.backends.forward_bank import GenerationBank
from openwsp.backends.jobs import JobPending
from openwsp.budget import compute_cost
from replications.exp8 import forward


@pytest.fixture
def tool_setup(tmp_path, monkeypatch):
    events = []
    dispatched = []
    sent_rows, response_rows = [], {}
    delivered = {}
    run = tmp_path / "run"
    (run / "requests").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    jobs_dir = run / "jobs"
    jobs_dir.mkdir()

    class Jobs:
        def _path(self, key):
            return jobs_dir / (key.replace("/", "_") + ".json")

        def has_job(self, key):
            return self._path(key).exists()

        def launch(self, key, spawn, **kwargs):
            if self._path(key).exists():
                events.append(("reused_job", key))
                return "existing-call"
            call = spawn("H100", "fake-gate")
            self._path(key).write_text(json.dumps({"call_id": call.object_id}))
            events.append(("new_job", key))
            return call.object_id

        def result(self, key, **kwargs):
            events.append(("result", key))
            call_id = json.loads(self._path(key).read_text())["call_id"]
            ids = response_rows.get(call_id, ["practice-1"])
            return {
                "outputs": [
                    {
                        "id": problem_id,
                        "samples": [{"text": "2", "token_count": 1, "finish_reason": "stop"}],
                    } for problem_id in ids
                ]
            }

    def generator(*, checkpoint, slot="generation", job_key="", gpu_kind="H100"):
        def spawn(*args, **kwargs):
            dispatched.append(checkpoint)
            call_id = f"fake-call-{len(dispatched)}"
            sent_rows.append([row["id"] for row in args[0]])
            response_rows[call_id] = sent_rows[-1]
            return SimpleNamespace(object_id=call_id)

        return SimpleNamespace(generate=SimpleNamespace(spawn=spawn))

    generator.with_options = lambda **options: generator

    def publish_file(local, remote):
        delivered[remote] = Path(local).read_bytes()
        events.append(("publish", remote))
        return {"path": remote, "bytes": len(delivered[remote]),
                "sha256": hashlib.sha256(delivered[remote]).hexdigest()}

    sandbox = SimpleNamespace(
        commit_work=lambda: events.append("sandbox_commit"),
        publish_file=publish_file,
    )
    work_volume = SimpleNamespace(
        commit=lambda: events.append("work_commit"), reload=lambda: events.append("work_reload")
    )
    data_volume = SimpleNamespace(commit=lambda: events.append("data_commit"))
    providers = SimpleNamespace(
        Generator=generator, TransformerReader=None, work_volume=work_volume, data_volume=data_volume
    )
    monkeypatch.setitem(sys.modules, "openwsp.backends.modal_app", providers)
    monkeypatch.setattr(forward, "freeze_checkpoint", lambda *args: ("/frozen/edit", "edit-sha", ["w"]))
    # Only redirect the fixed cache root; artifact and receipt paths remain real
    # tmp_path files so publication ordering is exercised rather than mocked.
    monkeypatch.setattr(
        forward,
        "Path",
        lambda value: tmp_path / "base-cache" if str(value) == "/data/base-generation-cache" else Path(value),
    )
    tool = object.__new__(forward.ForwardTool)
    tool.job_prefix = "live"
    tool.run_dir, tool.work = run, work
    tool.sandbox = sandbox
    tool.ledger = SimpleNamespace(snapshot=lambda: {"summary": {"modal": {"available_usd": 100}}})
    tool.checkpoint = lambda: events.append("checkpoint")
    tool.practice = {"practice-1": {"id": "practice-1", "source": "gsm8k", "problem": "1+1", "answer": "2"}}
    tool.grade = SimpleNamespace(
        extract=lambda text: (text, None), is_correct=lambda pred, answer: pred == answer
    )
    tool.jobs = Jobs()
    tool.base_bank = GenerationBank(tmp_path / "base-bank")
    tool._model_hashes = lambda _: {"original": "a" * 64, "edited": "b" * 64}
    tool._generation_prompts = lambda rows: [row.get("prompt", row.get("problem", "")) for row in rows]
    request = {"type": "generate", "model": "original", "problems": ["practice-1"], "n": 1}
    return SimpleNamespace(
        tool=tool,
        request=request,
        events=events,
        dispatched=dispatched,
        sent_rows=sent_rows,
        delivered=delivered,
        source=run / "requests/fwd_1.json.claimed",
        done=run / "requests/fwd_1.done.json",
    )


def test_separate_attempt_never_reuses_another_attempt_job_identifier(tool_setup):
    setup = tool_setup
    prior = setup.tool.jobs._path("live/fwd_1/original")
    prior.write_text('{"call_id":"prior-attempt"}')
    setup.tool.job_prefix = "live/separate-attempt"
    result = setup.tool.execute(setup.request, 1, 1)
    assert result["ok"]
    assert len(setup.dispatched) == 1
    assert ("new_job", "live/separate-attempt/fwd_1/original") in setup.events
    assert json.loads(prior.read_text()) == {"call_id": "prior-attempt"}


def test_delivery_failure_reuses_gpu_receipt_without_counting_a_failed_request(tool_setup):
    setup = tool_setup
    publish = setup.tool.sandbox.publish_file

    def fail_publication(local, remote):
        assert not setup.done.exists()
        assert (setup.tool.work / "gen_1.jsonl").exists()
        assert "work_commit" in setup.events
        raise RuntimeError("synthetic delivery failure")

    setup.tool.sandbox.publish_file = fail_publication
    with pytest.raises(JobPending, match="delivery did not complete"):
        setup.tool.execute(setup.request, 1, 1)
    assert not setup.done.exists()
    assert setup.source.exists()
    assert len(setup.dispatched) == 1
    setup.tool.sandbox.publish_file = publish
    recovered = setup.tool.execute(setup.request, 1, 1)
    assert recovered["ok"] is True
    assert setup.done.exists()
    assert len(setup.dispatched) == 1
    assert json.loads(setup.delivered["/work/gen_1.jsonl"])["id"] == "practice-1"


def test_cached_success_publishes_artifacts_without_launching_jobs(tool_setup):
    setup = tool_setup
    cached = {"ok": True, "type": "generate", "model": "original", "file": "/work/gen_1.jsonl"}
    setup.source.write_text(json.dumps(setup.request))
    setup.done.write_text(json.dumps(cached))
    (setup.tool.work / "gen_1.jsonl").write_text("saved generation record\n")
    assert setup.tool.execute(setup.request, 1, 1) == cached
    assert setup.events == [("publish", "/work/gen_1.jsonl"), "checkpoint"]
    assert setup.delivered["/work/gen_1.jsonl"] == b"saved generation record\n"
    receipt = json.loads((setup.tool.run_dir / "requests/fwd_1.delivery.json").read_text())
    assert receipt["files"][0]["volume_path"] == "/live/gen_1.jsonl"
    assert receipt["files"][0]["sha256"] == hashlib.sha256(setup.delivered["/work/gen_1.jsonl"]).hexdigest()
    assert setup.dispatched == []


def test_delivery_copies_only_the_named_paired_generation_artifacts(tool_setup):
    setup = tool_setup
    request = dict(setup.request, model="both", activations={"layers": [0], "mlp": False})
    result = {"ok": True, "files": {}, "activations": {}}
    expected = {}
    for mode in ("original", "edited"):
        name = f"gen_1_{mode}.jsonl"
        result["files"][mode] = f"/work/{name}"
        expected[name] = mode.encode()
        folder = f"gen_1_{mode}_acts"
        result["activations"][mode] = {"dir": f"/work/{folder}/", "files": {"hidden_last_prompt.npy": [1, 1, 2]}}
        expected[f"{folder}/hidden_last_prompt.npy"] = b"exact array bytes"
        expected[f"{folder}/index.json"] = b'{"samples":[]}'
    for relative, payload in expected.items():
        path = setup.tool.work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (setup.tool.work / "notes.md").write_text("editor-owned")
    setup.tool._deliver_artifacts(request, 1, result)
    assert setup.delivered == {f"/work/{path}": payload for path, payload in expected.items()}
    assert (setup.tool.work / "notes.md").read_text() == "editor-owned"


def test_teacher_forced_delivery_includes_arrays_and_metadata(tool_setup):
    setup = tool_setup
    folder = setup.tool.work / "fwd_7"
    folder.mkdir()
    for name in ("input_ids.npy", "hidden_0.npy", "meta.json"):
        (folder / name).write_bytes(name.encode())
    setup.tool._deliver_artifacts({"type": "teacher_forced"}, 7,
                                  {"ok": True, "files": {"input_ids": [1, 2], "hidden_0": [1, 2, 3]}})
    assert set(setup.delivered) == {"/work/fwd_7/input_ids.npy", "/work/fwd_7/hidden_0.npy", "/work/fwd_7/meta.json"}


def test_unexpected_artifact_path_is_not_delivered(tool_setup):
    setup = tool_setup
    with pytest.raises(JobPending, match="delivery did not complete"):
        setup.tool._deliver_artifacts({"type": "teacher_forced"}, 7,
                                      {"ok": True, "files": {"../edited/weight": [1]}})
    assert setup.delivered == {}


def test_cached_failure_needs_no_artifact_delivery(tool_setup):
    setup = tool_setup
    cached = {"ok": False, "error": "remote execution failed"}
    setup.source.write_text(json.dumps(setup.request))
    setup.done.write_text(json.dumps(cached))
    assert setup.tool.execute(setup.request, 1, 1) == cached
    assert setup.events == []
    assert setup.dispatched == []


def test_request_number_cannot_reuse_receipt_for_different_input(tool_setup):
    setup = tool_setup
    setup.source.write_text(json.dumps(setup.request))
    setup.done.write_text(json.dumps({"ok": True}))
    different = dict(setup.request, n=2)
    with pytest.raises(ValueError, match="different input"):
        setup.tool.execute(different, 1, 1)
    assert setup.events == []
    assert setup.dispatched == []
    assert json.loads(setup.source.read_text()) == setup.request


def test_paired_resume_only_needs_budget_and_dispatch_for_missing_job(tool_setup):
    setup = tool_setup
    bound = float(compute_cost(2520, gpu="H100", cpu_cores=4, memory_gib=16))
    setup.tool.ledger.snapshot = lambda: {"summary": {"modal": {"available_usd": bound + 0.01}}}
    setup.tool.jobs._path("live/fwd_1/original").write_text('{"call_id":"prior-call"}')
    request = dict(setup.request, model="both")
    result = setup.tool.execute(request, 1, 1)
    assert result["ok"] is True
    assert setup.dispatched == ["/frozen/edit"]
    assert ("reused_job", "live/fwd_1/original") in setup.events
    assert ("new_job", "live/fwd_1/edited") in setup.events
    assert setup.done.exists()


def test_exact_effective_duplicate_restores_result_without_new_pass_or_editor_hint(tool_setup):
    setup = tool_setup
    first = setup.tool.execute(setup.request, 1, 1)
    (setup.tool.work / "gen_1.jsonl").write_text("editor modified its local copy")
    second = setup.tool.execute(setup.request, 2, 2)
    assert len(setup.dispatched) == 1
    assert setup.tool.counts_as_pass(1) is True
    assert setup.tool.counts_as_pass(2) is False
    assert second["file"] == "/work/gen_2.jsonl"
    assert second["accuracy"] == first["accuracy"]
    assert setup.delivered["/work/gen_2.jsonl"] == setup.delivered["/work/gen_1.jsonl"]
    assert not any(word in json.dumps(second).lower() for word in ("reuse", "duplicate", "cache", "bank"))
    provenance = json.loads((setup.tool.run_dir / "requests/fwd_2.provenance.json").read_text())
    assert provenance["served_from_request"] == 1 and provenance["count_pass"] is False
    assert len(provenance["request_body_sha256"]) == 64
    # A restarted driver must recover the same non-counting result.
    assert setup.tool.execute(setup.request, 2, 2) == second
    assert setup.tool.counts_as_pass(2) is False and len(setup.dispatched) == 1


@pytest.mark.parametrize("change", ["weights", "practice_text", "sampling", "rendered_prompt"])
def test_same_body_after_effective_input_change_is_a_new_pass(tool_setup, monkeypatch, change):
    setup = tool_setup
    setup.tool.execute(setup.request, 1, 1)
    if change == "weights":
        setup.tool._model_hashes = lambda _: {"original": "a" * 64, "edited": "c" * 64}
    elif change == "practice_text":
        setup.tool.practice["practice-1"]["problem"] = "different content under the same ID"
    elif change == "rendered_prompt":
        setup.tool._generation_prompts = lambda rows: ["changed instruction or rendering" for _ in rows]
    else:
        monkeypatch.setitem(forward.CONFIG, "sampling", {**forward.CONFIG["sampling"], "temperature": 0.7})
    setup.tool.execute(setup.request, 2, 2)
    assert len(setup.dispatched) == 2 and setup.tool.counts_as_pass(2) is True


def test_paired_base_reuse_is_on_even_with_abundant_budget(tool_setup):
    setup = tool_setup
    setup.tool.execute(setup.request, 1, 1)
    result = setup.tool.execute(dict(setup.request, model="both"), 2, 2)
    assert setup.dispatched == ["/data/original", "/frozen/edit"]
    assert result["ok"] and setup.tool.counts_as_pass(2) is True
    provenance = json.loads((setup.tool.run_dir / "requests/fwd_2.provenance.json").read_text())
    assert provenance["base_generations_reused"] is True
    assert provenance["base_bank"][0]["reused"] is True


def test_partial_base_hits_generate_only_missing_rows_and_pin_plan_through_resume(tool_setup):
    setup = tool_setup
    setup.tool.execute(setup.request, 1, 1)
    setup.tool.practice["practice-2"] = {"id": "practice-2", "source": "gsm8k", "problem": "2+0", "answer": "2"}
    request = dict(setup.request, model="both", problems=["practice-1", "practice-2"])
    deliver = setup.tool._deliver_artifacts

    def fail(*args):
        raise JobPending("synthetic delivery interruption")

    setup.tool._deliver_artifacts = fail
    with pytest.raises(JobPending):
        setup.tool.execute(request, 2, 2)
    assert setup.sent_rows == [["practice-1"], ["practice-2"], ["practice-1", "practice-2"]]
    plan = json.loads((setup.tool.run_dir / "requests/fwd_2.base-plan.json").read_text())
    assert plan["missing"] == [1]
    setup.tool._deliver_artifacts = deliver
    result = setup.tool.execute(request, 2, 2)
    assert result["original"]["generations"] == 2 and len(setup.dispatched) == 3
    assert json.loads((setup.tool.run_dir / "requests/fwd_2.base-plan.json").read_text()) == plan


def test_pending_request_refuses_changed_weights_instead_of_rebinding_old_job(tool_setup):
    setup = tool_setup
    setup.tool._deliver_artifacts = lambda *args: (_ for _ in ()).throw(JobPending("interrupted"))
    with pytest.raises(JobPending):
        setup.tool.execute(setup.request, 1, 1)
    setup.tool._model_hashes = lambda _: {"original": "a" * 64, "edited": "c" * 64}
    with pytest.raises(JobPending, match="different effective"):
        setup.tool.execute(setup.request, 1, 1)
    assert len(setup.dispatched) == 1


@pytest.mark.parametrize("artifact,payload", [
    ("identity", "corrupt"), ("base-plan", "corrupt"), ("base-plan", "{}"),
])
def test_corrupt_pending_bank_metadata_blocks_without_a_new_pass(tool_setup, artifact, payload):
    setup = tool_setup
    setup.tool._deliver_artifacts = lambda *args: (_ for _ in ()).throw(JobPending("interrupted"))
    with pytest.raises(JobPending):
        setup.tool.execute(setup.request, 1, 1)
    path = setup.tool.run_dir / "requests" / f"fwd_1.{artifact}.json"
    path.write_text(payload)
    with pytest.raises(JobPending, match="malformed"):
        setup.tool.execute(setup.request, 1, 1)
    assert len(setup.dispatched) == 1 and not setup.done.exists()


def test_corrupt_bank_does_not_silently_resample(tool_setup):
    setup = tool_setup
    setup.tool.execute(setup.request, 1, 1)
    manifest = next((setup.tool.run_dir / "forward-bank").glob("*/entry.json"))
    manifest.write_text("corrupt")
    with pytest.raises(JobPending, match="could not be verified"):
        setup.tool.execute(setup.request, 2, 2)
    assert len(setup.dispatched) == 1


def test_teacher_forced_duplicate_restores_arrays_and_changed_text_is_new_pass(tool_setup, monkeypatch):
    setup = tool_setup
    setup.tool.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [1, 2])

    def reader(**parameters):
        def spawn(request, directory):
            setup.dispatched.append(parameters["checkpoint"])
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            (target / "input_ids.npy").write_bytes(b"synthetic array")
            (target / "meta.json").write_text('{"files":{"input_ids":[1,2]}}')
            return SimpleNamespace(object_id=f"teacher-{len(setup.dispatched)}")
        return SimpleNamespace(read=SimpleNamespace(spawn=spawn))

    reader.with_options = lambda **kwargs: reader
    providers = sys.modules["openwsp.backends.modal_app"]
    monkeypatch.setattr(providers, "TransformerReader", reader)
    setup.tool.jobs.result = lambda *args, **kwargs: {"files": {"input_ids": [1, 2]}, "lengths": [2]}
    request = {"type": "teacher_forced", "model": "original", "texts": ["first content"], "return": {}}
    setup.tool.execute(request, 1, 1)
    setup.tool.execute(request, 2, 2)
    assert len(setup.dispatched) == 1 and setup.tool.counts_as_pass(2) is False
    assert setup.delivered["/work/fwd_2/input_ids.npy"] == b"synthetic array"
    setup.tool.execute({**request, "texts": ["new content"]}, 3, 3)
    assert len(setup.dispatched) == 2 and setup.tool.counts_as_pass(3) is True
