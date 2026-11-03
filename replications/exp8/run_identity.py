"""Isolate separately reported live attempts without changing their task protocol."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from replications.exp8.runtime import ROOT, atomic_write, digest


@dataclass(frozen=True)
class LiveIdentity:
    editor: str = "openai"
    run_id: str | None = None

    def __post_init__(self):
        if self.editor not in {"openai", "codex"}:
            raise ValueError("Editor must be openai or codex")
        if self.editor == "openai" and self.run_id is not None:
            raise ValueError("The existing API trajectory keeps its original identity")
        if self.editor == "codex" and (not isinstance(self.run_id, str)
                or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.run_id)):
            raise ValueError("Codex requires a separate lowercase run_id")

    @property
    def relative(self):
        return "live-run" if self.run_id is None else f"live-runs/{self.run_id}"

    @property
    def root(self):
        return Path("/data") / self.relative

    @property
    def work_subpath(self):
        return "live" if self.run_id is None else self.relative

    @property
    def work(self):
        return Path("/work") / self.work_subpath

    @property
    def job_prefix(self):
        return "live" if self.run_id is None else f"live/{self.run_id}"

    @property
    def evaluation_stage(self):
        return "live-run" if self.run_id is None else f"live-runs/{self.run_id}"

    def record(self, data="/data"):
        """Bind a fresh namespace once; never rename/reset the original attempt."""
        root = Path(data) / self.relative
        record = {"schema_version": 1, "editor": self.editor, "run_id": self.run_id,
                  "relative_path": self.relative,
                  "task_sha256": digest(ROOT / "prompts/message.txt"),
                  "instructions_sha256": digest(ROOT / "prompts/instructions.md"),
                  "initialization": "Original pinned weights, empty isolated work namespace, zero live counters.",
                  "relationship": "Separate attempt; no prior API response history, partial edit or control result is supplied."}
        path = root / "identity.json"
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError("Live identity conflicts with the recorded attempt")
        elif self.editor == "codex":
            if (root / "state.json").exists():
                raise ValueError("Cannot adopt an existing unbound trajectory as a Codex run")
            atomic_write(path, record)
        return record


def live_identity(settings):
    return LiveIdentity(settings.get("editor", "openai"), settings.get("run_id"))
