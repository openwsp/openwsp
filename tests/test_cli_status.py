"""Compact status is bounded; the explicit JSON mode retains all evidence."""

import copy
import json
import os

import pytest
from click import unstyle
from typer.testing import CliRunner

from openwsp import cli, replication


@pytest.fixture
def status_state(monkeypatch):
    state = {
        "status": "blocked", "active_stage": "live", "heartbeat": 1_700_000_000,
        "live": {"turn": 158, "passes": 49, "runs": 210, "requests": 68, "done": False},
        "budget": {
            "modal": {"spent_usd": 8.43841, "reserved_usd": 9.38448, "available_usd": 24.12711},
            "editor": {"spent_usd": 0.214501, "reserved_usd": 0, "available_usd": 99.785499},
        },
        "error": "Volume conflict: " + "long diagnostic\n" * 1000,
        "controls_deferred": "Required controls defer until live evaluation finishes.",
        "publication": {"status": "pushed", "commit": "test-commit"},
        "stages": {"kernel": {"first_cell": {"output": "VERBOSE_KERNEL_OUTPUT" * 5000}}},
        "calibration": {"complete": True, "raw_details": list(range(1000))},
    }
    monkeypatch.setattr(replication, "status", lambda experiment: state)
    monkeypatch.setattr(cli.time, "time", lambda: 1_700_000_120)
    monkeypatch.setattr(cli.shutil, "get_terminal_size", lambda fallback: os.terminal_size((80, 24)))
    return state


def test_default_status_fits_one_screen_without_losing_progress(status_state):
    original = copy.deepcopy(status_state)
    result = CliRunner().invoke(cli.app, ["status", "exp8"])
    assert result.exit_code == 0, result.output
    assert "exp8: blocked | stage: live" in result.output
    assert "turn 158 | passes 49 | runs 210 | requests 68 | submitted no" in result.output
    assert "Modal USD: spent $8.4384 | reserved $9.3845 | available $24.1271" in result.output
    assert "Editor USD: spent $0.2145 | reserved $0.0000 | available $99.7855" in result.output
    assert "2023-11-14 22:13:20 UTC (120s ago)" in result.output
    assert "Last error: Volume conflict" in result.output
    assert "Controls note: Required controls defer" in result.output
    assert "Last publication: pushed" in result.output
    assert "openwsp status exp8 --json" in result.output
    assert "VERBOSE_KERNEL_OUTPUT" not in result.output
    assert len(result.output.splitlines()) <= 12
    assert max(map(len, result.output.splitlines())) <= 80
    assert status_state == original


def test_json_status_preserves_the_complete_machine_readable_state(status_state):
    result = CliRunner().invoke(cli.app, ["status", "--json"])
    assert result.exit_code == 0, result.output[:300]
    assert json.loads(result.output) == status_state
    assert "VERBOSE_KERNEL_OUTPUT" in result.output
    assert "Full state:" not in result.output


def test_not_started_does_not_invent_zero_counters_or_budget(monkeypatch):
    monkeypatch.setattr(replication, "status", lambda experiment: {"status": "not_started"})
    result = CliRunner().invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "exp8: not_started" in result.output
    assert "Live: not recorded" in result.output
    assert "Modal USD: not recorded" in result.output
    assert "Editor USD: not recorded" in result.output
    assert "Heartbeat: not recorded or invalid" in result.output
    assert "$0" not in result.output and "passes 0" not in result.output


def test_future_heartbeat_is_clock_skew_not_negative_elapsed(status_state):
    status_state["heartbeat"] = 1_700_000_180
    result = CliRunner().invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "60s ahead of local clock" in result.output
    assert "-60s ago" not in result.output


def test_status_help_explains_full_json_mode():
    result = CliRunner().invoke(cli.app, ["status", "--help"])
    assert result.exit_code == 0
    visible = unstyle(result.output)
    assert "--json" in visible
    assert "complete machine-readable state" in visible
