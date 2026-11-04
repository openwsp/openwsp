"""The hook and CI scan detect values without leaking them in diagnostics."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCANNER = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"
TRANSCRIPT = "replications/exp8/results/live-runs/test-run/transcript.md"


def git(directory, *args):
    return subprocess.run(["git", *args], cwd=directory, check=True, capture_output=True)


def scan(directory, *args):
    return subprocess.run([sys.executable, str(SCANNER), *args], cwd=directory, capture_output=True, text=True, check=False)


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    return tmp_path


@pytest.mark.parametrize("prefix", ["sk-", "hf_", "ak-", "as-", "ghp_", "github_pat_"])
def test_staged_secret_is_rejected_without_echoing_value(repository, prefix):
    value = prefix + "a" * 40
    (repository / "config.txt").write_text("token=" + value)
    git(repository, "add", "config.txt")
    result = scan(repository, "--staged")
    assert result.returncode == 1
    assert "config.txt" in result.stderr
    assert value not in result.stdout + result.stderr


def test_all_checks_committed_files_and_staged_default_ignores_them(repository):
    (repository / "config.txt").write_text("token=" + "sk-" + "b" * 40)
    git(repository, "add", "config.txt")
    git(repository, "commit", "-qm", "fixture")
    assert scan(repository).returncode == 0
    assert scan(repository, "--all").returncode == 1


def test_documented_prefixes_and_unstaged_values_are_not_staged_secrets(repository):
    (repository / "readme.txt").write_text("Blocked prefixes: sk-, hf_, ak-, as-, ghp_, github_pat_.")
    git(repository, "add", "readme.txt")
    (repository / "untracked.txt").write_text("sk-" + "c" * 40)
    assert scan(repository, "--staged").returncode == 0
    assert scan(repository, "--all").returncode == 0


def stage_transcript(repository, text, path=TRANSCRIPT):
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    git(repository, "add", path)
    return target


def printed_samples():
    # Match the printed nested sample field in t83 and its echoed t84 input.
    # Actual generation text remains private and is not a committed fixture.
    return "Generation nontext fields: " + repr({
        "id": "practice_1", "samples": [{"text": "PRIVATE SAMPLE SENTINEL café ] } ' text",
                                          "finish_reason": "stop", "tokens": 12}],
        "shape": [2, 3], "path": "/work/gen_1.jsonl",
    })


@pytest.mark.parametrize("kind", ["plain", "nested_json", "echo", "truncated"])
def test_staged_transcript_rejects_old_renderer_sample_dump(repository, kind):
    output = printed_samples()
    if kind == "nested_json":
        output = json.dumps({"result": {"stdout": output}, "elapsed": 1}, indent=2)
    elif kind == "truncated":
        output = output[:output.index("finish_reason")] + "\n[output truncated]"
    section = "### Input" if kind == "echo" else "### Tool result"
    if kind == "echo":
        output = "OUTPUT of Python run 44 of at most 100:\n" + output
    target = stage_transcript(repository, "# Live run transcript\n\n## Turn 83\n\n"
                              + section + "\n\n```text\n" + output + "\n```\n")
    # Read the index, even if the working tree was subsequently corrected.
    target.write_text("# Corrected working tree\n")
    result = scan(repository, "--staged")
    assert result.returncode == 1
    assert "Raw sample payload detected in: " + TRANSCRIPT in result.stderr
    assert "PRIVATE SAMPLE SENTINEL" not in result.stdout + result.stderr


@pytest.mark.parametrize("path", [TRANSCRIPT, "replications/exp8/results/live-run/transcript.md"])
def test_current_renderer_transcript_passes_with_examples_and_metadata(repository, path):
    from openwsp.reporting import render_transcript

    run = repository / "private-run"
    first, second = run / "turns/t083", run / "turns/t084"
    first.mkdir(parents=True)
    second.mkdir()
    example = "{'samples': [{'text': 'EXAMPLE IN CODE'}]}"
    (first / "prompt.txt").write_text("Original task:\nUse " + example)
    (first / "answer.md").write_text("Example: " + example + "\n```python\nprint(" + example + ")\n```")
    (second / "prompt.txt").write_text("OUTPUT of Python run 44 of at most 100:\n" + printed_samples())
    rows = [
        {"turn": 83, "code": "print(" + example + ")", "out": printed_samples()},
        {"turn": 83, "result": {"output": printed_samples(), "generated_text": "private structured text"}},
        {"turn": 84, "out": "{'samples': [{'item': 'practice_1', 'correct': True, 'tokens': 12}], "
                              "'generations': 2, 'path': '/work/raw_generations/gen_1.jsonl'}"},
    ]
    (run / "tool_calls.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    transcript = render_transcript(run, repository / "rendered.md").read_text()
    assert "PRIVATE SAMPLE SENTINEL" not in transcript
    assert example in transcript and "sha256=" in transcript
    target = stage_transcript(repository, transcript, path)
    target.write_text(printed_samples())  # Unstaged text does not enter publication.
    assert scan(repository, "--staged").returncode == 0
    assert scan(repository, "--all").returncode == 0


def test_transcript_guard_uses_fresh_clone_detector_over_frozen_pythonpath(repository, monkeypatch):
    frozen = repository / "frozen/openwsp"
    frozen.mkdir(parents=True)
    (frozen / "__init__.py").write_text("")
    (frozen / "reporting.py").write_text("raise RuntimeError('frozen detector must not be imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(frozen.parent))
    stage_transcript(repository, "## Turn 83\n\n### Tool result\n\n```text\n"
                     + printed_samples() + "\n```\n")
    result = scan(repository, "--staged")
    assert result.returncode == 1
    assert "Raw sample payload detected" in result.stderr
    assert "RuntimeError" not in result.stderr


def test_raw_transcript_guard_does_not_scan_source_or_documentation_examples(repository):
    (repository / "example.py").write_text("output = " + repr(printed_samples()) + "\n")
    git(repository, "add", "example.py")
    assert scan(repository, "--staged").returncode == 0
