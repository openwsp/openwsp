"""The hook and CI scan detect values without leaking them in diagnostics."""
import subprocess
import sys
from pathlib import Path

import pytest

SCANNER = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"


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
