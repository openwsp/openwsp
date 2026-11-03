"""Reporting must preserve evidence, pending states and source provenance."""

import json
from pathlib import Path

import pytest

from openwsp.reporting import generate_report, render_transcript


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup_repo(tmp_path):
    (tmp_path / "README.md").write_text(
        "# OpenWSP\n\nBefore.\n<!-- OPENWSP:RESULTS:START -->\nstale results\n<!-- OPENWSP:RESULTS:END -->\n"
        "Between.\n<!-- OPENWSP:COSTS:START -->\nstale costs\n<!-- OPENWSP:COSTS:END -->\nAfter.\n"
    )
    return tmp_path / "replications/exp8/results"


def test_report_reads_evidence_and_replaces_stale_tables(tmp_path):
    results = setup_repo(tmp_path)
    reference = tmp_path / "replications/exp8/reference/p50c.json"
    write_json(reference, {"math500": {"acc": 0.375, "cut_share": 0.25}})
    summary = results / "live-run/math500/summary.json"
    write_json(
        summary,
        {
            "stage": "live-run",
            "model": "test-model",
            "benchmark": "math500",
            "n": 2,
            "correct": 1,
            "accuracy": 0.5,
            "standard_error": 0.1,
            "seconds": 3600,
            "cost_usd": 2.5,
            "status": "complete",
        },
    )
    summary.with_name("samples.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"problem_id": "a", "sample_index": 0, "correct": True},
                {"problem_id": "b", "sample_index": 0, "correct": False},
            ]
        )
    )
    generated = generate_report(tmp_path, results)
    readme = (tmp_path / "README.md").read_text()
    assert "stale" not in readme
    assert "Before." in readme and "Between." in readme and "After." in readme
    assert "[37.5%](replications/exp8/reference/p50c.json)" in readme
    assert "[50.0%](replications/exp8/results/live-run/math500/summary.json)" in readme
    assert "[2.5000]" in readme and "[1.000]" in readme
    assert "pending" in readme  # Missing GSM8K/IFEval and cut_share are not invented.
    assert generated["summaries"] == 1 and len(generated["sources"]) == 2
    first = readme
    generate_report(tmp_path, results)
    assert (tmp_path / "README.md").read_text() == first
    write_json(reference, {"math500": {"acc": 0.25}})
    generate_report(tmp_path, results)
    assert "[37.5%]" not in (tmp_path / "README.md").read_text()


def test_empty_report_does_not_invent_results_or_zero_cost(tmp_path):
    results = setup_repo(tmp_path)
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "pending" in text
    assert "[0.0%]" not in text and "[0.0000]" not in text


@pytest.mark.parametrize("standalone_status", [False, True])
def test_status_appears_once_in_each_readme_layout_and_standalone_report(tmp_path, standalone_status):
    results = setup_repo(tmp_path)
    readme = tmp_path / "README.md"
    if standalone_status:
        readme.write_text("<!-- OPENWSP:STATUS:START -->\nstale status\n<!-- OPENWSP:STATUS:END -->\n"
                          + readme.read_text())
    write_json(results / "pipeline-state.json", {"status": "blocked", "active_stage": "live"})
    generate_report(tmp_path, results)
    text = readme.read_text()
    status_heading = "| Recorded component | State | Evidence |"
    assert text.count(status_heading) == 1
    assert "[blocked](replications/exp8/results/pipeline-state.json)" in text
    result_section = text.split("<!-- OPENWSP:RESULTS:START -->", 1)[1].split(
        "<!-- OPENWSP:RESULTS:END -->", 1)[0]
    assert (status_heading in result_section) is not standalone_status
    report = (results / "report.md").read_text()
    assert report.count(status_heading) == 1
    assert "[blocked](pipeline-state.json)" in report
    generate_report(tmp_path, results)
    assert readme.read_text() == text
    assert (results / "report.md").read_text() == report


def test_stale_sample_summary_does_not_modify_readme(tmp_path):
    results = setup_repo(tmp_path)
    summary = results / "calibration_base/summary.json"
    write_json(summary, {"n": 2, "correct": 2, "accuracy": 1.0})
    summary.with_name("samples.jsonl").write_text(
        json.dumps({"problem_id": "a", "sample": 0, "correct": 0}) + "\n"
    )
    original = (tmp_path / "README.md").read_text()
    with pytest.raises(ValueError, match="disagrees"):
        generate_report(tmp_path, results)
    assert (tmp_path / "README.md").read_text() == original


def test_report_preserves_blocked_and_recovering_state_with_source_links(tmp_path):
    results = setup_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- OPENWSP:STATUS:START -->\nstale status\n<!-- OPENWSP:STATUS:END -->\n" + readme.read_text())
    write_json(results / "pipeline-state.json", {"status": "blocked", "active_stage": "calibration_p50c", "error": "quota exhausted"})
    write_json(results / "coordinator.json", {"current": {"phase": "recovering"}})
    write_json(results / "kernel/hash-verification.json", {"status": "passed"})
    write_json(results / "live-run/state.json", {"done": False, "passes": 2, "turn": 3, "runs": 4})
    generated = generate_report(tmp_path, results)
    text = readme.read_text()
    assert "[blocked](replications/exp8/results/pipeline-state.json)" in text
    assert "[recovering](replications/exp8/results/coordinator.json)" in text
    assert "[quota exhausted](replications/exp8/results/pipeline-state.json)" in text
    assert "[incomplete](replications/exp8/results/live-run/state.json)" in text
    assert "[2](replications/exp8/results/live-run/state.json)" in text
    assert "stale status" not in text
    assert len(generated["sources"]) == 4


def test_component_success_is_not_whole_pipeline_success(tmp_path):
    results = setup_repo(tmp_path)
    write_json(results / "kernel/hash-verification.json", {"status": "passed"})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "[passed]" in text
    assert "Component checks do not establish overall completion" in text


def test_fallback_diagnostic_is_not_labeled_as_an_active_blocker(tmp_path):
    results = setup_repo(tmp_path)
    write_json(results / "kernel/weight-swap.json", {"status": "fallback", "error": "Native swap unavailable; fresh engine selected"})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "Recorded diagnostic" in text
    assert "Recorded blocker" not in text
    assert "[fallback](replications/exp8/results/kernel/weight-swap.json)" in text
    assert "States are recorded snapshots" in text


def test_coordinator_block_reason_is_retained(tmp_path):
    results = setup_repo(tmp_path)
    write_json(results / "coordinator.json", {"current": {"phase": "blocked", "stage": "all", "reason": "Provider outcome remains uncertain"}})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "Recorded blocker" in text
    assert "[Provider outcome remains uncertain](replications/exp8/results/coordinator.json)" in text


@pytest.mark.parametrize("stage,status", [("controls", "stage_done"), ("publication", "complete")])
def test_later_stage_success_cannot_hide_recorded_live_blocker(tmp_path, stage, status):
    results = setup_repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- OPENWSP:STATUS:START -->\nstale status\n<!-- OPENWSP:STATUS:END -->\n" + readme.read_text())
    write_json(results / "pipeline-state.json", {"status": status, "active_stage": stage})
    write_json(results / "coordinator.json", {"current": {"phase": "complete", "stage": stage}})
    write_json(results / "live-run/state.json", {"done": False, "passes": 4, "turn": 26, "runs": 10})
    blocker = results / "live-run/blocker.json"
    write_json(blocker, {
        "status": "blocked", "reason": "editor_budget_admission", "error": "Next response exceeds remaining allowance",
        "live": {"passes": 4, "turn": 26, "done": False, "submission": None}, "required_live_passes": 50,
        "final_evaluation_status": "not_started_no_completed_50_pass_submission", "publication": {"status": "pushed"},
    })
    write_json(results / "control1_nothink/math500/summary.json", {
        "stage": "control1_nothink", "benchmark": "math500", "status": "done", "n": 10, "correct": 4, "accuracy": 0.4,
    })
    generated = generate_report(tmp_path, results)
    text = readme.read_text()
    top = text.split("<!-- OPENWSP:STATUS:START -->", 1)[1].split("<!-- OPENWSP:STATUS:END -->", 1)[0]
    source = "replications/exp8/results/live-run/blocker.json"
    assert f"| Required live run | [blocked]({source}) | [live goal unmet]({source})" in top
    assert f"[4]({source}) / [50]({source})" in top
    assert f"[not_started_no_completed_50_pass_submission]({source})" in top
    assert f"[{status}](replications/exp8/results/pipeline-state.json)" in top
    assert top.index("Required live run") < top.index("| Pipeline |")
    assert "Completed controls or publication do not satisfy the blocked live-run goal" in top
    assert "[40.0%]" in text  # Completed controls still appear as their own result.
    report = (results / "report.md").read_text()
    assert "[blocked](live-run/blocker.json)" in report and "live goal unmet" in report
    assert str(blocker) in {item["path"] for item in generated["sources"]}


def test_explicitly_resolved_live_blocker_is_not_reported_as_unmet_goal(tmp_path):
    results = setup_repo(tmp_path)
    write_json(results / "live-run/blocker.json", {"status": "resolved", "reason": "resolved in recorded follow-up"})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "[resolved](replications/exp8/results/live-run/blocker.json)" in text
    assert "live goal unmet" not in text
    assert "Completed controls or publication do not satisfy" not in text


def active_attempt(results, *, run_id="codex-first", state=None):
    name = f"live-runs/{run_id}"
    identity = {"schema_version": 1, "editor": "codex", "run_id": run_id, "relative_path": name,
                "task_sha256": "b" * 64, "instructions_sha256": "c" * 64}
    write_json(results / "active-live.json", identity)
    write_json(results / name / "identity.json", identity)
    if state is not None:
        write_json(results / name / "state.json", state)
    return results / name


def submitted_state(**changes):
    return {"passes": 50, "turn": 72, "runs": 20, "done": True,
            "submission": {"checkpoint": "/data/submissions/final", "sha256": "a" * 64}, **changes}


def final_summaries(directory):
    for benchmark, samples, n in (("math500", 4, 2000), ("gsm8k", 2, 2638), ("ifeval", 2, 1082)):
        write_json(directory / benchmark / "summary.json", {
            "stage": f"live-runs/{directory.name}", "benchmark": benchmark, "status": "done",
            "samples_per_problem": samples, "n": n, "correct": n // 2, "accuracy": 0.5,
            "provenance": {"submission_sha256": "a" * 64},
        })


def test_active_attempt_supersedes_historical_goal_but_preserves_every_attempt(tmp_path):
    results = setup_repo(tmp_path)
    write_json(results / "live-run/state.json", {"passes": 4, "turn": 26, "runs": 10, "done": False})
    blocker = results / "live-run/blocker.json"
    write_json(blocker, {"status": "blocked", "reason": "editor_budget_admission", "required_live_passes": 50})
    legacy = blocker.read_bytes()
    selected = active_attempt(results, state={"passes": 7, "turn": 12, "runs": 5, "done": False})
    transcript = selected / "transcript.md"
    transcript.write_text("# Saved transcript\n")
    write_json(results / "live-runs/codex-earlier/state.json", {"passes": 1, "turn": 2, "runs": 1, "done": False})
    write_json(results / "pipeline-state.json", {"status": "complete", "active_stage": "publication"})
    generated = generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "| Required live run | [live goal unmet](replications/exp8/results/live-runs/codex-first/state.json)" in text
    assert "| live-run | historical | pending | [blocked](replications/exp8/results/live-run/blocker.json)" in text
    assert "| live-runs/codex-first | active | [codex]" in text
    assert "| live-runs/codex-earlier | historical |" in text
    assert "[7](replications/exp8/results/live-runs/codex-first/state.json)" in text
    assert "[4](replications/exp8/results/live-run/state.json)" in text
    assert "[transcript.md](replications/exp8/results/live-runs/codex-first/transcript.md)" in text
    assert "Completed controls or publication do not satisfy the blocked live-run goal" not in text
    assert "Before." in text and "Between." in text and "After." in text
    assert blocker.read_bytes() == legacy
    assert str(transcript) in {row["path"] for row in generated["sources"]}
    assert "[7](live-runs/codex-first/state.json)" in (results / "report.md").read_text()


@pytest.mark.parametrize("identity_exists", [False, True])
def test_missing_active_state_never_falls_back_to_old_blocker(tmp_path, identity_exists):
    results = setup_repo(tmp_path)
    selected = active_attempt(results)
    if not identity_exists:
        (selected / "identity.json").unlink()
        selected.rmdir()
    write_json(results / "live-run/blocker.json", {"status": "blocked"})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "| Required live run | [live goal unmet](replications/exp8/results/active-live.json)" in text
    assert "pending: active state missing" in text and "pending: state missing" in text
    assert "[0]" not in text


def test_unselected_attempt_does_not_override_legacy_goal_without_pointer(tmp_path):
    results = setup_repo(tmp_path)
    selected = active_attempt(results, state=submitted_state())
    (results / "active-live.json").unlink()
    final_summaries(selected)
    write_json(results / "live-run/blocker.json", {"status": "blocked", "reason": "legacy blocker"})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "| Required live run | [blocked](replications/exp8/results/live-run/blocker.json)" in text
    assert "| live-run | current (legacy default) |" in text
    assert "| live-runs/codex-first | unselected |" in text
    assert "Completed controls or publication do not satisfy the blocked live-run goal" in text


@pytest.mark.parametrize("changes", [
    {"passes": 49}, {"passes": 51}, {"passes": 49.5}, {"passes": 50.0}, {"passes": True},
    {"done": 1}, {"submission": None}, {"submission": True}, {"submission": {}},
    {"submission": {"checkpoint": " ", "sha256": "a" * 64}},
    {"submission": {"checkpoint": "/data/submission", "sha256": "pending"}},
])
def test_active_submission_requires_exact_passes_done_and_concrete_identity(tmp_path, changes):
    results = setup_repo(tmp_path)
    selected = active_attempt(results, state=submitted_state(**changes))
    final_summaries(selected)
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "live goal unmet" in text
    assert "submission and final evaluations recorded" not in text


def test_active_submission_without_final_evaluations_stays_unmet(tmp_path):
    results = setup_repo(tmp_path)
    active_attempt(results, state=submitted_state())
    final_summaries(results / "live-runs/other")
    final_summaries(results / "live-run")
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "live goal unmet" in text and "submission recorded; final evaluation pending" in text


def test_only_full_matching_attempt_evaluations_establish_recorded_completion(tmp_path):
    results = setup_repo(tmp_path)
    selected = active_attempt(results, state=submitted_state())
    final_summaries(selected)
    write_json(results / "live-run/blocker.json", {"status": "blocked"})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "submission and final evaluations recorded" in text
    assert "live goal unmet" not in text
    assert "| live-run | historical | pending | [blocked]" in text
    for benchmark in ("math500", "gsm8k", "ifeval"):
        assert f"{benchmark}: [recorded](replications/exp8/results/live-runs/codex-first/{benchmark}/summary.json)" in text


def test_missing_active_identity_keeps_complete_state_and_evaluations_pending(tmp_path):
    results = setup_repo(tmp_path)
    selected = active_attempt(results, state=submitted_state())
    final_summaries(selected)
    (selected / "identity.json").unlink()
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "live goal unmet" in text and "pending: active identity missing" in text
    assert "submission and final evaluations recorded" not in text
    for benchmark in ("math500", "gsm8k", "ifeval"):
        assert f"{benchmark}: pending" in text


@pytest.mark.parametrize("outside_results", [False, True])
def test_final_evaluation_symlink_outside_attempt_is_rejected_before_reading(tmp_path, monkeypatch, outside_results):
    results = setup_repo(tmp_path)
    selected = active_attempt(results, state=submitted_state())
    final_summaries(selected)
    summary = selected / "math500/summary.json"
    outside = tmp_path / "outside.json" if outside_results else results / "other-summary.json"
    summary.rename(outside)
    summary.symlink_to(outside)
    readme_before = (tmp_path / "README.md").read_bytes()
    report = results / "report.md"
    report.write_text("Previously published report.\n")

    from openwsp import reporting
    original_read = reporting._read_json

    def guarded_read(path):
        assert path != summary, "External summary must be rejected before its contents are read"
        return original_read(path)

    monkeypatch.setattr(reporting, "_read_json", guarded_read)
    with pytest.raises(ValueError, match="summary escapes its attempt directory"):
        generate_report(tmp_path, results)
    assert (tmp_path / "README.md").read_bytes() == readme_before
    assert report.read_text() == "Previously published report.\n"


@pytest.mark.parametrize("change", [
    {"status": "partial"}, {"stage": "live-run"}, {"samples_per_problem": 2},
    {"n": 4, "correct": 2}, {"correct": None}, {"accuracy": None},
    {"provenance": {}}, {"provenance": {"submission_sha256": "b" * 64}},
    {"provenance": {"submission_sha256": "a" * 64, "problem_limit": 500}},
    {"provenance": {"submission_sha256": "a" * 64, "problem_selection": "subset"}},
])
def test_partial_or_mismatched_evaluation_cannot_complete_active_goal(tmp_path, change):
    results = setup_repo(tmp_path)
    selected = active_attempt(results, state=submitted_state())
    final_summaries(selected)
    summary = selected / "math500/summary.json"
    write_json(summary, {**json.loads(summary.read_text()), **change})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "live goal unmet" in text and "math500: pending" in text
    assert "submission and final evaluations recorded" not in text


def test_batch_summary_cannot_replace_attempt_aggregate(tmp_path):
    results = setup_repo(tmp_path)
    selected = active_attempt(results, state=submitted_state())
    final_summaries(selected)
    summary = selected / "math500/summary.json"
    write_json(selected / "math500/batch_000/summary.json", json.loads(summary.read_text()))
    summary.unlink()
    generate_report(tmp_path, results)
    assert "live goal unmet" in (tmp_path / "README.md").read_text()


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"run_id": "../escape"},
    {"run_id": "Upper"}, {"run_id": "a" * 65}, {"editor": ""},
    {"relative_path": "/tmp/outside"}, {"relative_path": "live-runs/../codex-first"},
    {"relative_path": "live-runs/other"}, {"relative_path": "live-runs\\codex-first"},
    {"task_sha256": None}, {"instructions_sha256": "not-a-digest"},
])
def test_invalid_active_pointer_does_not_change_published_files(tmp_path, change):
    results = setup_repo(tmp_path)
    active_attempt(results)
    pointer = results / "active-live.json"
    write_json(pointer, {**json.loads(pointer.read_text()), **change})
    before = (tmp_path / "README.md").read_bytes()
    with pytest.raises(ValueError, match="active-live pointer"):
        generate_report(tmp_path, results)
    assert (tmp_path / "README.md").read_bytes() == before
    assert not (results / "report.md").exists()


@pytest.mark.parametrize("change", [{"run_id": "other"}, {"editor": "astra"},
                                     {"task_sha256": "d" * 64}, {"instructions_sha256": "e" * 64},
                                     {"schema_version": True}])
def test_active_identity_must_match_selection(tmp_path, change):
    results = setup_repo(tmp_path)
    selected = active_attempt(results)
    identity_path = selected / "identity.json"
    write_json(identity_path, {**json.loads(identity_path.read_text()), **change})
    with pytest.raises(ValueError, match="mismatch"):
        generate_report(tmp_path, results)


@pytest.mark.parametrize("target", ["directory", "state", "pointer"])
def test_attempt_symlink_escape_is_rejected_before_publication(tmp_path, target):
    results = setup_repo(tmp_path)
    selected = active_attempt(results)
    outside = tmp_path / "outside"
    outside.mkdir()
    write_json(outside / "state.json", submitted_state())
    if target == "directory":
        (selected / "identity.json").unlink()
        selected.rmdir()
        selected.symlink_to(outside, target_is_directory=True)
    elif target == "state":
        (selected / "state.json").symlink_to(outside / "state.json")
    else:
        pointer = results / "active-live.json"
        pointer.rename(outside / "pointer.json")
        pointer.symlink_to(outside / "pointer.json")
    with pytest.raises(ValueError, match="escapes"):
        generate_report(tmp_path, results)
    assert not (results / "report.md").exists()


def test_calibration_gap_audit_and_runtime_are_source_linked(tmp_path):
    results = setup_repo(tmp_path)
    write_json(results / "calibration-check.json", {
        "status": "gap_requires_review", "calibration_base": {"accuracy": 0.1, "reference_accuracy": 0.2,
        "gap": -0.1, "within_two_standard_errors": False},
    })
    write_json(results / "calibration-gap-audit.json", {"circuit_breaker_tripped": False, "checks": {"prompt": True}})
    write_json(results / "calibration_base/summary.json", {"stage": "calibration_base", "benchmark": "math500",
        "accuracy": 0.1, "provenance": {"engine": "test-engine", "gpu": "test-gpu", "transformers": "test-version",
        "standard_error": "binomial_per_sample"}})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "[-10.00](replications/exp8/results/calibration-check.json)" in text
    assert "[audit](replications/exp8/results/calibration-gap-audit.json)" in text
    assert "[test-engine](replications/exp8/results/calibration_base/summary.json)" in text
    assert "[binomial_per_sample](replications/exp8/results/calibration_base/summary.json)" in text


def test_cost_ledger_separates_settled_estimates_and_reserved_bounds(tmp_path):
    results = setup_repo(tmp_path)
    write_json(results / "costs.json", {"reservations": {
        "a": {"kind": "modal", "stage": "calibration_base", "status": "settled", "actual_usd": "2.50",
              "seconds": 3600, "receipt_metadata": {"billing_basis": "resource_upper_bound_not_invoice"}},
        "b": {"kind": "editor", "stage": "editor:1", "status": "reserved", "upper_bound_usd": "1.25"},
    }})
    write_json(results / "setup/cost.json", {"stage": "setup", "cost_usd": 0.25})
    generated = generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "[2.5000](replications/exp8/results/costs.json)" in text
    assert "[1.2500](replications/exp8/results/costs.json)" in text
    assert "[0.2500](replications/exp8/results/setup/cost.json)" in text
    assert "[resource_upper_bound_not_invoice](replications/exp8/results/costs.json)" in text
    assert "Reservations are retained bounds, not charges" in text
    assert "resource hours are not end-to-end wall time" in text
    assert "[0.0000]" not in text
    assert len(generated["sources"]) == 2


def test_costs_aggregate_requests_and_batches_without_merging_conditions_or_bases(tmp_path):
    results = setup_repo(tmp_path)
    reservations = {}

    def receipt(stage, amount, *, seconds=1200, status="settled", metadata=None):
        reservations[str(len(reservations))] = {
            "kind": "modal", "stage": stage, "status": status, "seconds": seconds,
            "actual_usd" if status == "settled" else "upper_bound_usd": str(amount),
            "receipt_metadata": metadata or {},
        }

    # Two attempts for one request and a later request share one operation row.
    receipt("live/fwd_1/original/openwsp-gpu:aaa:0", 1)
    receipt("live/fwd_1/original/openwsp-gpu:aaa:1", 2)
    receipt("live/fwd_2/original/openwsp-gpu:bbb:0", 3)
    receipt("live/fwd_3/original/openwsp-gpu:ccc:0", 4, status="reserved", seconds=None)
    receipt("live/fwd_4/original/openwsp-gpu:ddd:0", 5, metadata={"failed_job_upper_bound": True})
    receipt("live/fwd_5/original/openwsp-gpu:eee:0", 6, metadata={"invoice": True})
    receipt("live/fwd_6/original/openwsp-gpu:fff:0", 7,
            metadata={"source": "cancelled_gpu_resource_upper_bound_not_invoice"})
    receipt("live/fwd_1/edited/openwsp-gpu:ggg:0", 2)
    receipt("live/fwd_2/edited/fresh-fallback/openwsp-gpu:hhh:0", 3)
    receipt("live/fwd_1/original/activations/openwsp-gpu:iii:0", 2)
    receipt("live/fwd_2/original/activations/openwsp-gpu:jjj:0", 3)
    receipt("live/fwd_7/teacher_forced/openwsp-gpu:kkk:0", 0.5)
    receipt("live/fwd_8/teacher_forced/openwsp-gpu:lll:0", 1)
    for condition, benchmark in [
        ("live-run", "math500"), ("live-run", "gsm8k"),
        ("control1_nothink", "math500"), ("control2_mechanical", "math500"),
        ("calibration_base", "math500"), ("calibration_p50c", "math500"),
    ]:
        receipt(f"evaluation/{condition}/{benchmark}/batch_000/openwsp-gpu:mmm:0", 1)
        receipt(f"evaluation/{condition}/{benchmark}/batch_001/openwsp-gpu:nnn:0", 2)
    receipt("evaluation/live-run/math500/smoke/openwsp-gpu:ooo:0", 0.25)
    receipt("control2/directions/batch_000/openwsp-gpu:ppp:0", 0.1)
    receipt("control2/directions/batch_001/openwsp-gpu:qqq:0", 0.2)
    receipt("control2/directions/smoke/openwsp-gpu:rrr:0", 0.05)
    ledger = results / "costs.json"
    write_json(ledger, {"reservations": reservations})
    original_ledger = ledger.read_bytes()
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    rows = [line for line in text.splitlines() if line.startswith("| modal /")]
    source = "replications/exp8/results/costs.json"

    def group(stage, state="settled", basis="resource estimate; not invoice"):
        matches = [line for line in rows if line.startswith(f"| modal / {stage} | [{state}]")
                   and f"[{basis}]({source})" in line]
        assert len(matches) == 1
        return matches[0]

    generation = group("live / generation / original")
    assert f"[3]({source})" in generation  # Receipts, including both attempts.
    assert f"[1.000]({source})" in generation  # Sum of all three resource durations.
    assert f"[6.0000]({source})" in generation
    reserved = group("live / generation / original", "reserved")
    assert f"[1]({source}) | pending | pending | [4.0000]({source})" in reserved
    assert f"[5.0000]({source})" in group("live / generation / original", basis="failed job upper bound; not invoice")
    assert f"[6.0000]({source})" in group("live / generation / original", basis="provider invoice")
    assert f"[7.0000]({source})" in group("live / generation / original", basis="cancelled job upper bound; not invoice")
    assert f"[5.0000]({source})" in group("live / generation / edited")
    assert f"[5.0000]({source})" in group("live / activation extraction / original")
    assert f"[1.5000]({source})" in group("live / teacher-forced")
    for stage in ("live-run / math500", "live-run / gsm8k", "control1_nothink / math500",
                  "control2_mechanical / math500", "calibration_base / math500", "calibration_p50c / math500"):
        assert f"[2]({source})" in group(stage + " / evaluation")
        assert f"[3.0000]({source})" in group(stage + " / evaluation")
    assert f"[0.2500]({source})" in group("live-run / math500 / evaluation smoke")
    assert f"[0.3000]({source})" in group("control2 / directions")
    assert f"[0.0500]({source})" in group("control2 / directions smoke")
    assert "openwsp-gpu:" not in text and "fwd_" not in text and "batch_" not in text
    assert ledger.read_bytes() == original_ledger


@pytest.mark.parametrize("amount", ["NaN", "-1", True])
def test_invalid_settled_ledger_does_not_replace_published_files(tmp_path, amount):
    results = setup_repo(tmp_path)
    write_json(results / "costs.json", {"reservations": {"a": {
        "kind": "modal", "stage": "calibration_base", "status": "settled", "actual_usd": amount,
    }}})
    original = (tmp_path / "README.md").read_text()
    with pytest.raises(ValueError, match="cost value"):
        generate_report(tmp_path, results)
    assert (tmp_path / "README.md").read_text() == original
    assert not (results / "report.md").exists()


def test_missing_settled_amount_stays_pending(tmp_path):
    results = setup_repo(tmp_path)
    write_json(results / "costs.json", {"reservations": {"a": {
        "kind": "modal", "stage": "calibration_base", "status": "settled",
    }}})
    generate_report(tmp_path, results)
    text = (tmp_path / "README.md").read_text()
    assert "[settled]" in text
    assert "[0.0000]" not in text


def test_transcript_contains_actual_turns_tools_and_redacts_tokens(tmp_path):
    run = tmp_path / "run"
    turn = run / "turns/t001"
    turn.mkdir(parents=True)
    (turn / "prompt.txt").write_text("Read the weights.")
    (turn / "answer.md").write_text("I will inspect a column.\n```python\nload('x')\n```\n")
    credential = "sk-" + "x" * 30
    (run / "tool_calls.jsonl").write_text(
        json.dumps(
            {
                "turn": 1,
                "code": "print('shape')",
                "out": "shape (2, 2) " + credential,
            }
        )
        + "\n"
    )
    write_json(run / "requests/fwd_1.done.json", {"ok": True, "accuracy": 0.25, "texts": ["RAW GENERATION"]})
    # This file must not be loaded or embedded by the renderer.
    (run / "gen_1.jsonl").write_text("RAW GENERATION")
    path = render_transcript(run, run / "transcript.md")
    text = path.read_text()
    assert "Turn 1" in text and "inspect a column" in text and "shape (2, 2)" in text
    assert credential not in text and "[REDACTED]" in text
    assert "RAW GENERATION" not in text
    assert '"accuracy": 0.25' in text


def test_transcript_records_incomplete_editor_event_without_reading_private_response(tmp_path, monkeypatch):
    run = tmp_path / "run"
    turn = run / "turns/t011"
    turn.mkdir(parents=True)
    private_response = run / "editor/response_private.json"
    write_json(private_response, {"text": "PRIVATE PARTIAL CODE"})
    write_json(turn / "run.json", {"text": "PRIVATE PARTIAL CODE"})
    write_json(turn / "editor_event.json", {
        "status": "incomplete", "reason": "max_output_tokens", "response_id": "resp_saved",
        "executed": False, "usage": {"output_tokens": 32768}, "cost_usd": "1.683131",
        "partial_text_sha256": "a" * 64, "partial_text_characters": 20,
        "response_volume": "openwsp-exp8-data", "response_volume_path": str(private_response),
        "partial_text": "PRIVATE PARTIAL CODE", "text": "PRIVATE PARTIAL CODE",
    })
    original_read = Path.read_text
    forbidden = {private_response.resolve(), (turn / "run.json").resolve()}

    def read_text(path, *args, **kwargs):
        assert path.resolve() not in forbidden, "Renderer opened a private response"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    output = render_transcript(run, run / "transcript.md").read_text()
    assert "Turn 11" in output and "### Editor event" in output
    assert '"status": "incomplete"' in output and '"reason": "max_output_tokens"' in output
    assert "No code or forward pass from this response was executed" in output
    assert '"output_tokens": 32768' in output and '"cost_usd": "1.683131"' in output
    assert '"partial_text_sha256": "' + "a" * 64 + '"' in output
    assert "resp_saved" in output and "response_volume_path" in output
    assert "Editor reply pending" not in output and "PRIVATE PARTIAL CODE" not in output
