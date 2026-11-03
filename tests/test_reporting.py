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
