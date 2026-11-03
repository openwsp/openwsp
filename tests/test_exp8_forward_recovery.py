"""Forward receipt recovery with tiny fake providers; no remote calls or grading data."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openwsp.backends.jobs import JobPending
from openwsp.budget import compute_cost
from replications.exp8 import forward


@pytest.fixture
def tool_setup(tmp_path, monkeypatch):
    events = []
    dispatched = []
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
            return {
                "outputs": [
                    {
                        "id": "practice-1",
                        "samples": [{"text": "2", "token_count": 1, "finish_reason": "stop"}],
                    }
                ]
            }

    def generator(*, checkpoint, slot="generation", job_key="", gpu_kind="H100"):
        def spawn(*args, **kwargs):
            dispatched.append(checkpoint)
            return SimpleNamespace(object_id=f"fake-call-{len(dispatched)}")

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
    request = {"type": "generate", "model": "original", "problems": ["practice-1"], "n": 1}
    return SimpleNamespace(
        tool=tool,
        request=request,
        events=events,
        dispatched=dispatched,
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
