"""Publication tests use local clones and intercept every push; no remote writes."""

import json
import subprocess
from pathlib import Path

import pytest

from openwsp import publishing as P
from openwsp.api import OpenWSP

PREFIX = "replications/exp8/results"


def git(directory, *args):
    return subprocess.run(["git", *args], cwd=directory, check=True, capture_output=True, text=True)


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text(
        "# Project\nHuman prose stays.\n"
        "<!-- OPENWSP:STATUS:START -->\nold state\n<!-- OPENWSP:STATUS:END -->\n"
        "<!-- OPENWSP:RESULTS:START -->\nold result\n<!-- OPENWSP:RESULTS:END -->\n"
        "<!-- OPENWSP:COSTS:START -->\nold cost\n<!-- OPENWSP:COSTS:END -->\n"
    )
    (repo / "source.py").write_text("human_source = True\n")
    (repo / "scripts").mkdir()
    scanner = Path(__file__).resolve().parents[1] / "scripts/check_secrets.py"
    (repo / "scripts/check_secrets.py").write_text(scanner.read_text())
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "initial")
    return repo


@pytest.fixture
def staging(tmp_path):
    root = tmp_path / "staging"
    directory = root / PREFIX / "live/eval/math500"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps({
        "stage": "live-run", "model": "edited", "benchmark": "math500", "n": 1,
        "correct": 1, "accuracy": 1.0, "status": "partial",
    }))
    (directory / "samples.jsonl").write_text(json.dumps({
        "problem_id": "p0", "sample_index": 0, "answer": "7", "correct": True,
        "token_count": 8, "finish_reason": "stop", "raw_path": "/volume/raw.jsonl",
        "raw_sha256": "0" * 64,
    }) + "\n")
    return root


def fake_transport(monkeypatch, source_repo, *, rejects=0, failure=None, concurrent=None):
    real_run = P._run
    calls, pushes = [], []

    def run(args, *, cwd, env, check=True):
        calls.append(args)
        assert env["GITHUB_TOKEN"] == "dummy-value-for-test"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert all("dummy-value-for-test" not in arg for arg in args)
        if args[0] == "git" and "clone" in args:
            args = [str(source_repo) if arg == P.REPOSITORY else arg for arg in args]
        if args[0] == "git" and "push" in args:
            # Record the commit contents while the private clone still exists.
            pushes.append({
                "readme": (cwd / "README.md").read_text(),
                "source": (cwd / "source.py").read_text(),
                "status": (cwd / "STATUS.md").read_text() if (cwd / "STATUS.md").exists() else None,
                "evidence": (cwd / PREFIX / "kernel/local-evidence.json").read_text()
                if (cwd / PREFIX / "kernel/local-evidence.json").exists() else None,
                "files": git(cwd, "show", "--format=", "--name-only", "HEAD").stdout.splitlines(),
            })
            if concurrent and len(pushes) == 1:
                concurrent()
            message = failure or ("[rejected] fetch first" if len(pushes) <= rejects else "")
            return subprocess.CompletedProcess(args, int(bool(message)), "", message)
        return real_run(args, cwd=cwd, env=env, check=check)

    monkeypatch.setenv("GITHUB_TOKEN", "dummy-value-for-test")
    monkeypatch.setattr(P, "_run", run)
    return calls, pushes


def artifact_paths(staging):
    return [str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()]


def test_publish_changes_only_allowlisted_results_and_readme_tables(tmp_path, monkeypatch, source_repo, staging):
    calls, pushes = fake_transport(monkeypatch, source_repo)
    (staging / "source.py").write_text("should not be copied\n")
    paths = artifact_paths(staging / PREFIX)
    paths = [str(Path(PREFIX) / path) for path in paths]
    result = P.publish_results(staging, paths, checkout_dir=tmp_path / "checkouts")
    assert result["status"] == "pushed"
    assert result["attempts"] == 1
    assert "Human prose stays." in pushes[0]["readme"]
    assert "old state" not in pushes[0]["readme"]
    assert "pending: no published state record" in pushes[0]["readme"]
    assert pushes[0]["source"] == "human_source = True\n"
    assert all(path == "README.md" or path.startswith(PREFIX + "/") for path in result["paths"])
    assert any("pull" in command and "--rebase" in command for command in calls)
    assert any("scripts/check_secrets.py" in command for command in calls)


def test_concurrent_push_reclones_and_preserves_new_human_prose(tmp_path, monkeypatch, source_repo, staging):
    def concurrent():
        readme = source_repo / "README.md"
        readme.write_text(readme.read_text().replace("Human prose stays.", "New human prose stays."))
        git(source_repo, "add", "README.md")
        git(source_repo, "commit", "-qm", "human update")

    calls, pushes = fake_transport(monkeypatch, source_repo, rejects=1, concurrent=concurrent)
    result = P.publish_results(staging, artifact_paths(staging), checkout_dir=tmp_path / "checkouts")
    assert result["attempts"] == 2
    assert "New human prose stays." in pushes[-1]["readme"]
    assert sum("clone" in call for call in calls) == 2


def test_blocked_publication_preserves_concurrent_status_and_unrelated_evidence(tmp_path, monkeypatch, source_repo, staging):
    state = staging / PREFIX / "pipeline-state.json"
    state.write_text(json.dumps({"status": "blocked", "active_stage": "live", "error": "Budget exhausted"}))

    def concurrent():
        (source_repo / "STATUS.md").write_text("Human status update after deployment.\n")
        evidence = source_repo / PREFIX / "kernel/local-evidence.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text('{"status": "passed", "source": "local probe"}\n')
        git(source_repo, "add", "STATUS.md", str(evidence.relative_to(source_repo)))
        git(source_repo, "commit", "-qm", "record independent evidence")

    _, pushes = fake_transport(monkeypatch, source_repo, rejects=1, concurrent=concurrent)
    result = P.publish_results(staging, artifact_paths(staging), checkout_dir=tmp_path / "checkouts")
    assert result["attempts"] == 2
    assert pushes[-1]["status"] == "Human status update after deployment.\n"
    assert json.loads(pushes[-1]["evidence"])["source"] == "local probe"
    assert "[blocked](replications/exp8/results/pipeline-state.json)" in pushes[-1]["readme"]
    assert "[Budget exhausted](replications/exp8/results/pipeline-state.json)" in pushes[-1]["readme"]
    assert "[partial](replications/exp8/results/live/eval/math500/summary.json)" in pushes[-1]["readme"]
    assert "STATUS.md" not in result["paths"]
    assert PREFIX + "/kernel/local-evidence.json" not in result["paths"]


def test_authentication_failure_is_not_retried_or_exposed(tmp_path, monkeypatch, source_repo, staging):
    _, pushes = fake_transport(monkeypatch, source_repo, failure="Authentication failed dummy-value-for-test")
    with pytest.raises(P.PublishingError) as error:
        P.publish_results(staging, artifact_paths(staging), checkout_dir=tmp_path / "checkouts")
    assert "dummy-value-for-test" not in str(error.value)
    assert len(pushes) == 1


@pytest.mark.parametrize("path", [
    "README.md", "src/openwsp/core.py", "../outside.json", "/tmp/outside.json",
    PREFIX + "/../config.json", PREFIX + "/weights/model.safetensors",
    PREFIX + "/live/gen_1.jsonl", PREFIX + "/live/activations/a.npy",
    PREFIX + "/live/array.npz",
])
def test_refuses_paths_outside_public_artifact_contract(tmp_path, monkeypatch, path):
    monkeypatch.setenv("GITHUB_TOKEN", "dummy")
    with pytest.raises(P.PublishingError):
        P.publish_results(tmp_path, [path], checkout_dir=tmp_path / "checkouts")


def test_refuses_symlinked_source(tmp_path, monkeypatch, staging):
    monkeypatch.setenv("GITHUB_TOKEN", "dummy")
    path = staging / PREFIX / "linked.json"
    path.symlink_to(staging / PREFIX / "live/eval/math500/summary.json")
    with pytest.raises(P.PublishingError, match="Symlink"):
        P.publish_results(staging, [str(path.relative_to(staging))], checkout_dir=tmp_path / "checkouts")


def test_rejects_raw_generation_fields_and_secret_values(tmp_path, monkeypatch, staging):
    monkeypatch.setenv("GITHUB_TOKEN", "dummy")
    path = staging / PREFIX / "unsafe.json"
    for value in ({"generated_text": "raw generation"}, {"key": "sk-" + "a" * 40}):
        path.write_text(json.dumps(value))
        with pytest.raises(P.PublishingError):
            P.publish_results(staging, [str(path.relative_to(staging))], checkout_dir=tmp_path / "checkouts")


def test_data_inconsistency_prevents_commit_and_push(tmp_path, monkeypatch, source_repo, staging):
    calls, pushes = fake_transport(monkeypatch, source_repo)
    summary = staging / PREFIX / "live/eval/math500/summary.json"
    data = json.loads(summary.read_text())
    data.update(correct=0, accuracy=0)
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disagrees"):
        P.publish_results(staging, artifact_paths(staging), checkout_dir=tmp_path / "checkouts")
    assert not pushes
    assert not any("commit" in call for call in calls)


def test_versioned_edit_and_transcript_are_publishable(tmp_path, monkeypatch, source_repo, staging):
    _, pushes = fake_transport(monkeypatch, source_repo)
    edit = staging / PREFIX / "live/edit"
    api = OpenWSP.tiny(tmp_path / "tiny")
    api.save_edit("layer1.weight", [[1.5, 2], [-1, 1]])
    api.export_edit(edit, base_model="tiny", revision="test-fixture")
    (edit.parent / "transcript.md").write_text("# Turn\n\nExplanation.\n```python\nx = 1\n```\nResult: 1\n")
    result = P.publish_results(staging, artifact_paths(staging), checkout_dir=tmp_path / "checkouts")
    assert result["status"] == "pushed"
    assert PREFIX + "/live/edit/edit.npz" in pushes[0]["files"]


def test_readme_without_markers_stops_before_push(tmp_path, monkeypatch, source_repo, staging):
    (source_repo / "README.md").write_text("# Human changed the document structure\n")
    git(source_repo, "add", "README.md")
    git(source_repo, "commit", "-qm", "remove markers")
    _, pushes = fake_transport(monkeypatch, source_repo)
    with pytest.raises(P.PublishingError, match="marker"):
        P.publish_results(staging, artifact_paths(staging), checkout_dir=tmp_path / "checkouts")
    assert not pushes


def test_corrupted_edit_hash_is_refused_before_git(tmp_path, monkeypatch, staging):
    monkeypatch.setenv("GITHUB_TOKEN", "dummy")
    edit = staging / PREFIX / "live/edit"
    api = OpenWSP.tiny(tmp_path / "tiny")
    api.save_edit("layer1.weight", [[1.5, 2], [-1, 1]])
    api.export_edit(edit)
    manifest = edit / "manifest.json"
    data = json.loads(manifest.read_text())
    data["edit_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(P.PublishingError, match="address mismatch"):
        P.publish_results(staging, artifact_paths(staging), checkout_dir=tmp_path / "checkouts")
    assert not (tmp_path / "checkouts").exists()


def test_identical_results_are_not_committed_or_pushed(tmp_path, monkeypatch, source_repo, staging):
    import shutil

    for relative in artifact_paths(staging):
        destination = source_repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staging / relative, destination)
    P.generate_report(source_repo, source_repo / PREFIX)
    git(source_repo, "add", ".")
    git(source_repo, "commit", "-qm", "existing results")
    calls, pushes = fake_transport(monkeypatch, source_repo)
    result = P.publish_results(staging, artifact_paths(staging), checkout_dir=tmp_path / "checkouts")
    assert result["status"] == "unchanged"
    assert not pushes
    assert not any("commit" in call for call in calls)
