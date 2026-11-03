"""Forward receipt recovery with tiny fake providers; no remote calls or grading data."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openwsp.budget import compute_cost
from replications.exp8 import forward


@pytest.fixture
def tool_setup(tmp_path, monkeypatch):
    events = []
    dispatched = []
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

    sandbox = SimpleNamespace(
        commit_work=lambda: events.append("sandbox_commit"),
        reload_work=lambda: events.append("sandbox_reload"),
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
        source=run / "requests/fwd_1.json.claimed",
        done=run / "requests/fwd_1.done.json",
    )


def test_reload_failure_does_not_publish_success_receipt(tool_setup):
    setup = tool_setup

    def fail_reload():
        assert not setup.done.exists()
        assert (setup.tool.work / "gen_1.jsonl").exists()
        assert "work_commit" in setup.events
        raise RuntimeError("synthetic delivery failure")

    setup.tool.sandbox.reload_work = fail_reload
    with pytest.raises(RuntimeError, match="delivery failure"):
        setup.tool.execute(setup.request, 1, 1)
    assert not setup.done.exists()
    assert setup.source.exists()
    assert len(setup.dispatched) == 1
    setup.tool.sandbox.reload_work = lambda: setup.events.append("sandbox_reload_recovered")
    recovered = setup.tool.execute(setup.request, 1, 1)
    assert recovered["ok"] is True
    assert setup.done.exists()
    assert len(setup.dispatched) == 1
    assert "sandbox_reload_recovered" in setup.events


def test_cached_success_reloads_artifacts_without_launching_jobs(tool_setup):
    setup = tool_setup
    cached = {"ok": True, "type": "generate", "model": "original", "file": "/work/gen_1.jsonl"}
    setup.source.write_text(json.dumps(setup.request))
    setup.done.write_text(json.dumps(cached))
    assert setup.tool.execute(setup.request, 1, 1) == cached
    assert setup.events == ["sandbox_reload"]
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
