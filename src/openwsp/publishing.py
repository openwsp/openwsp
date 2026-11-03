"""Publish explicit result artifacts and generated README tables to the project repo.

No source file or README prose is copied from staging. Credentials are read by
Git's helper at runtime, never embedded in a remote URL, command or log.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .edits import _payload_sha256, edit_sha256
from .reporting import generate_report

REPOSITORY = "https://github.com/openwsp/openwsp.git"
RESULTS = PurePosixPath("replications/exp8/results")
_SECRET = re.compile(r"(?:sk-|hf_|ak-|as-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}")
_RAW_NAMES = {"raw", "raw_generations", "generations", "activations", "weights", "checkpoints", "models"}
_RAW_FIELDS = {"raw_text", "generated_text", "token_ids", "input_ids", "hidden_states", "mlp_activations"}


class PublishingError(RuntimeError):
    """Publication stopped without changing source files or exposing credentials."""


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or any(ord(char) < 32 for char in value) or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise PublishingError("Publication paths must be relative result paths")
    if not path.is_relative_to(RESULTS) or path == RESULTS:
        raise PublishingError("Only replications/exp8/results artifacts may be supplied")
    if any(part.lower() in _RAW_NAMES for part in path.parts):
        raise PublishingError("Raw generations, activations and full weights are not publishable")
    if re.match(r"(?:gen_\d+|raw[_-])", path.name.lower()):
        raise PublishingError("Raw generation files are not publishable")
    if path.suffix not in {".json", ".jsonl", ".md", ".npz"}:
        raise PublishingError("Unsupported result artifact type")
    if path.suffix == ".npz" and path.name != "edit.npz":
        raise PublishingError("Only versioned edit.npz payloads may be published as arrays")
    return path


def _source(root: Path, relative: PurePosixPath) -> Path:
    path = root.joinpath(*relative.parts)
    if any(root.joinpath(*relative.parts[:i]).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise PublishingError("Symlinked staging artifacts are not allowed")
    if not path.is_file():
        raise PublishingError(f"Missing result artifact: {relative}")
    return path


def _validate_json(value: Any) -> None:
    if isinstance(value, dict):
        if _RAW_FIELDS.intersection(value):
            raise PublishingError("Raw generation or activation fields found in result data")
        for item in value.values():
            _validate_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item)


def _validate_file(path: Path) -> None:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise PublishingError("Result artifact exceeds the publication size limit")
    if path.suffix == ".npz":
        try:
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > 256 * 1024 * 1024:
                    raise PublishingError("Edit payload expands beyond the publication size limit")
                if not all(re.fullmatch(r"t\d+_(?:idx|val|a|b)\.npy", name) for name in archive.namelist()):
                    raise PublishingError("Edit payload contains unexpected arrays")
        except zipfile.BadZipFile as exc:
            raise PublishingError("Edit payload is not a valid NPZ archive") from exc
        return
    text = path.read_text()
    if _SECRET.search(text):
        raise PublishingError("Credential-shaped content found in result artifact")
    try:
        if path.suffix == ".json":
            _validate_json(json.loads(text))
        elif path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise PublishingError("JSONL result records must be objects")
                    _validate_json(row)
    except (ValueError, UnicodeError) as exc:
        raise PublishingError("Result artifact is not valid text data") from exc


def _readme_prose(text: str) -> str:
    """Remove only table bodies; the remaining bytes must not change."""
    for section in ("STATUS", "RESULTS", "COSTS"):
        start, end = f"<!-- OPENWSP:{section}:START -->", f"<!-- OPENWSP:{section}:END -->"
        if section == "STATUS" and start not in text and end not in text:
            continue
        if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
            raise PublishingError("README must contain exactly one ordered pair of each table marker")
        before, rest = text.split(start)
        _, after = rest.split(end)
        text = before + start + end + after
    return text


def _validate_edit(payload: Path, manifest: Path) -> None:
    data = json.loads(manifest.read_text())
    if data.get("format") != "openwsp.edit" or data.get("format_version") != 1:
        raise PublishingError("Published edits must use the versioned OpenWSP contract")
    try:
        with np.load(payload, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if any(array.dtype.kind not in "iuf" for array in arrays.values()):
            raise PublishingError("Edit payload must contain numeric arrays")
        if data.get("payload_sha256") != _payload_sha256(arrays):
            raise PublishingError("Edit payload hash mismatch")
        if data.get("edit_sha256") != edit_sha256(data):
            raise PublishingError("Edit manifest address mismatch")
    except (ValueError, KeyError) as exc:
        raise PublishingError("Invalid versioned edit payload") from exc


def _run(args: list[str], *, cwd: Path, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=180, check=False)
    if check and result.returncode:
        # Git may echo server diagnostics containing authorization information.
        # Keep those bytes private; the operation and exit code are sufficient.
        operation = args[1] if len(args) > 1 else args[0]
        raise PublishingError(f"Publication operation {operation} failed with exit code {result.returncode}")
    return result


def _retryable_push(result: subprocess.CompletedProcess) -> bool:
    message = (result.stderr + result.stdout).lower()
    if any(word in message for word in ("authentication", "permission denied", "403", "401", "quota")):
        return False
    return any(word in message for word in ("non-fast-forward", "fetch first", "[rejected]"))


def publish_results(
    staging_dir: str | Path,
    paths: Sequence[str],
    *,
    checkout_dir: str | Path,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Clone main, validate explicit artifacts, render tables, commit and push.

    ``staging_dir`` mirrors repository-relative paths. ``checkout_dir`` is a
    writable parent for a private temporary clone. Only the fixed project repo
    and main branch are used. A concurrent push causes a pull/rebase attempt and
    a fresh clone/re-render; authentication or quota failures stop immediately.
    """
    if not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between one and three")
    if not os.environ.get("GITHUB_TOKEN"):
        raise PublishingError("GITHUB_TOKEN is required for result publication")
    staging = Path(staging_dir).resolve()
    relative_paths = sorted({_relative_path(str(path)) for path in paths})
    if not relative_paths:
        raise PublishingError("An explicit nonempty result allowlist is required")
    for relative in relative_paths:
        path = _source(staging, relative)
        _validate_file(path)
        if path.suffix == ".npz":
            manifest = relative.with_name("manifest.json")
            if manifest not in relative_paths:
                raise PublishingError("An edit payload must be accompanied by its manifest")
            _validate_edit(path, _source(staging, manifest))
    checkout_parent = Path(checkout_dir).resolve()
    checkout_parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    for attempt in range(1, max_attempts + 1):
        with tempfile.TemporaryDirectory(prefix="openwsp-publish-", dir=checkout_parent) as temporary:
            directory = Path(temporary)
            helper = directory / "credential-helper"
            helper.write_text(
                "#!/usr/bin/env python3\nimport os, sys\n"
                "if len(sys.argv) > 1 and sys.argv[1] == 'get':\n"
                "    print('username=x-access-token')\n"
                "    print('password=' + os.environ['GITHUB_TOKEN'])\n"
            )
            helper.chmod(0o700)
            # The command contains a helper path, never the credential value.
            helper_command = "!" + shlex.quote(sys.executable) + " " + shlex.quote(str(helper))
            git = ["git", "-c", "credential.helper=", "-c", f"credential.helper={helper_command}"]
            clone = directory / "repo"
            _run(git + ["clone", "--branch", "main", "--single-branch", REPOSITORY, str(clone)], cwd=directory, env=env)
            _run(git + ["pull", "--rebase", "origin", "main"], cwd=clone, env=env)
            for key, value in (("user.name", "OpenWSP results"), ("user.email", "openwsp@users.noreply.github.com"), ("core.hooksPath", ".githooks")):
                _run(git + ["config", key, value], cwd=clone, env=env)
            readme = clone / "README.md"
            if readme.is_symlink():
                raise PublishingError("README must not be a symlink")
            original_prose = _readme_prose(readme.read_text())
            for relative in relative_paths:
                destination = clone.joinpath(*relative.parts)
                if any(clone.joinpath(*relative.parts[:i]).is_symlink() for i in range(1, len(relative.parts) + 1)):
                    raise PublishingError("Repository result destination must not contain symlinks")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_source(staging, relative), destination)
            for relative in relative_paths:
                if relative.suffix == ".npz":
                    _validate_edit(clone / str(relative), clone / str(relative.with_name("manifest.json")))
            generate_report(clone, clone / str(RESULTS))  # Also validates summary/sample consistency.
            if _readme_prose(readme.read_text()) != original_prose:
                raise PublishingError("Report generation changed README prose outside table markers")
            generated = {str(path) for path in relative_paths} | {"README.md", str(RESULTS / "report.md")}
            for relative in generated:
                _validate_file(clone / relative)
            _run(git + ["add", "--", *sorted(generated)], cwd=clone, env=env)
            changed = _run(git + ["diff", "--cached", "--name-only", "-z"], cwd=clone, env=env)
            staged = set(filter(None, changed.stdout.split("\0")))
            if not staged.issubset(generated):
                raise PublishingError("Unexpected staged files outside publication allowlist")
            if not staged:
                return {"status": "unchanged", "attempts": attempt, "paths": []}
            _run([sys.executable, "scripts/check_secrets.py", "--staged"], cwd=clone, env=env)
            # Live transcripts preserve captured editor code/output, whose
            # trailing spaces may be meaningful evidence. Exempt only those files
            # from cosmetic whitespace lint; every content/secret guard above
            # still covers them, and all other staged files retain this check.
            _run(git + ["diff", "--cached", "--check", "--", ".",
                        ":(exclude,literal)replications/exp8/results/live-run/transcript.md",
                        ":(exclude,glob)replications/exp8/results/live-runs/*/transcript.md"], cwd=clone, env=env)
            evidence = "Evidence:\n" + "\n".join(sorted(path for path in staged if path != "README.md"))
            _run(git + ["commit", "-m", "docs(results): publish recorded replication evidence", "-m", evidence], cwd=clone, env=env)
            pushed = _run(git + ["push", "origin", "HEAD:main"], cwd=clone, env=env, check=False)
            if pushed.returncode == 0:
                commit = _run(git + ["rev-parse", "HEAD"], cwd=clone, env=env).stdout.strip()
                return {"status": "pushed", "commit": commit, "attempts": attempt, "paths": sorted(staged)}
            if not _retryable_push(pushed):
                raise PublishingError("Result push failed; authentication, quota or non-retryable Git error")
            rebased = _run(git + ["pull", "--rebase", "origin", "main"], cwd=clone, env=env, check=False)
            if rebased.returncode:
                _run(git + ["rebase", "--abort"], cwd=clone, env=env, check=False)
            # Re-render from a fresh main so concurrent new results and README
            # prose are preserved even when Git's textual rebase could succeed.
    raise PublishingError("Concurrent result pushes exhausted the publication retry limit")
