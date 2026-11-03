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
