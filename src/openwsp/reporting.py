"""Generate published tables and readable transcripts from recorded evidence only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

_REFERENCES = ("unedited", "p10c", "p25c", "p50c")
_BENCHMARKS = ("math500", "gsm8k", "ifeval")
_COMPARATORS = {"calibration_base": "unedited", "calibration_p50c": "p50c", "live-run": "p50c"}
_RUN_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_LIVE_EVALUATIONS = {"math500": (4, 2000), "gsm8k": (2, 2638), "ifeval": (2, 1082)}
_SECRET = re.compile(r"(?:sk-|hf_|ak-|as-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}")
_BEARER = re.compile(r"(?i)(authorization[\"'\s:]+bearer\s+)[A-Za-z0-9._~+/-]+")
_STATE_FILES = (
    "pipeline-state.json", "coordinator.json", "live-run/state.json", "live-run/blocker.json", "calibration-check.json", "calibration-prompt-audit.json",
    "calibration-gap-audit.json", "kernel/hash-verification.json", "kernel/sandbox-verification.json",
    "kernel/forward-verification.json", "kernel/weight-swap.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".report-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value: Any, default: str = "pending") -> str:
    if value is None:
        return default
    return str(value).replace("|", "\\|").replace("\n", " ")


def _link(path: Path, relative_to: Path) -> str:
    return quote(os.path.relpath(path, relative_to), safe="/._-")


def _metric(value: Any, path: Path, relative_to: Path, *, percent: bool = False, digits: int = 1) -> str:
    if not _number(value):
        return "pending"
    rendered = f"{value * 100:.{digits}f}%" if percent else f"{value:.{digits}f}"
    return f"[{rendered}]({_link(path, relative_to)})"


def _recorded(value: Any, path: Path, relative_to: Path) -> str:
    return f"[{_text(_redact(str(value)))}]({_link(path, relative_to)})" if value is not None else "pending"


def _attempt_evidence(results: Path) -> tuple[dict, dict, list[Path]]:
    """Validate the explicit selection before any generated file is changed."""
    pointer_path = results / "active-live.json"
    if not pointer_path.resolve().is_relative_to(results):
        raise ValueError("Active-live pointer escapes results")
    pointer = _read_json(pointer_path) if pointer_path.exists() else {}
    if pointer:
        run_id = pointer.get("run_id")
        if (type(pointer.get("schema_version")) is not int or pointer["schema_version"] != 1
                or not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id)
                or pointer.get("relative_path") != f"live-runs/{run_id}"
                or not isinstance(pointer.get("editor"), str) or not pointer["editor"].strip()
                or any(not isinstance(pointer.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", pointer[key])
                       for key in ("task_sha256", "instructions_sha256"))):
            raise ValueError("Invalid active-live pointer")
    elif pointer_path.exists():
        raise ValueError("Invalid active-live pointer")
    paths = [pointer_path] if pointer else []
    names = ({f"live-runs/{path.name}" for path in (results / "live-runs").iterdir() if path.is_dir()}
             if (results / "live-runs").exists() else set())
    if pointer:
        names.add(pointer["relative_path"])
    names.add("live-run")
    attempts = {}
    for name in sorted(names):
        directory = results / name
        if name != "live-run" and not _RUN_ID.fullmatch(directory.name):
            raise ValueError("Invalid live attempt directory name")
        if not directory.resolve().is_relative_to(results):
            raise ValueError("Live attempt directory escapes results")
        records = {}
        for filename in ("state.json", "identity.json", "blocker.json", "transcript.md"):
            path = directory / filename
            if not path.exists():
                continue
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Live attempt evidence escapes its directory")
            data = {} if filename == "transcript.md" else _read_json(path)
            if name != "live-run" and data.get("run_id", directory.name) != directory.name:
                raise ValueError("Live attempt evidence run_id mismatch")
            records[filename] = (path, data)
            paths.append(path)
        if records or name != "live-run":
            attempts[name] = records
    if pointer:
        identity = attempts[pointer["relative_path"]].get("identity.json")
        keys = ("schema_version", "editor", "run_id", "relative_path", "task_sha256", "instructions_sha256",
                "initialization", "relationship")
        if identity and (type(identity[1].get("schema_version")) is not int
                         or any(identity[1].get(key) != pointer[key] for key in keys if key in pointer)):
            raise ValueError("Active live attempt identity mismatch")
    return pointer, attempts, paths


def _submission_ready(state: dict) -> bool:
    submission = state.get("submission")
    return (type(state.get("passes")) is int and state["passes"] == 50 and state.get("done") is True
            and isinstance(submission, dict) and isinstance(submission.get("checkpoint"), str)
            and bool(submission["checkpoint"].strip()) and isinstance(submission.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", submission["sha256"]) is not None)


def _attempt_evaluations(name: str, records: dict, summaries: list[tuple[Path, dict]]) -> dict[str, Path]:
    state_item = records.get("state.json")
    if (state_item is None or not _submission_ready(state_item[1])
            or (name != "live-run" and "identity.json" not in records)):
        return {}
    directory = state_item[0].parent
    completed = {}
    for path, summary in summaries:
        benchmark = summary.get("benchmark")
        if benchmark not in _LIVE_EVALUATIONS or path != directory / benchmark / "summary.json":
            continue
        samples, count = _LIVE_EVALUATIONS[benchmark]
        provenance = summary.get("provenance") or {}
        if (summary.get("stage") == name and summary.get("status") in {"done", "complete", "completed"}
                and type(summary.get("samples_per_problem")) is int and summary["samples_per_problem"] == samples
                and type(summary.get("n")) is int and summary["n"] == count
                and type(summary.get("correct")) is int and _number(summary.get("accuracy"))
                and provenance.get("submission_sha256") == state_item[1]["submission"]["sha256"]
                and provenance.get("problem_limit") is None and provenance.get("problem_selection") is None):
            completed[benchmark] = path
    return completed


def _attempt_table(pointer: dict, attempts: dict, summaries: list, relative_to: Path, pointer_path: Path | None) -> str:
    lines = ["Live attempts:", "", "| Attempt | Selection | Editor | Recorded state | Passes | Turns | Runs | Submission | Final evaluations | Evidence |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, records in attempts.items():
        state_item = records.get("state.json")
        blocker = records.get("blocker.json")
        identity = records.get("identity.json")
        source, state = state_item or blocker or identity or (None, {})
        counts = state if state_item else state.get("live") or {}
        status = "submission recorded" if _submission_ready(counts) else "incomplete"
        if blocker and blocker[1].get("status") == "blocked":
            status = "blocked"
        state_cell = (_recorded(status, blocker[0] if status == "blocked" else source, relative_to)
                      if state_item or blocker else "pending: state missing")
        editor = _recorded(identity[1].get("editor"), identity[0], relative_to) if identity else "pending"
        if not identity and name == pointer.get("relative_path"):
            editor = _recorded(pointer["editor"], pointer_path, relative_to)
        completed = _attempt_evaluations(name, records, summaries)
        evaluation = "; ".join(f"{benchmark}: " + (_recorded("recorded", completed[benchmark], relative_to)
                                if benchmark in completed else "pending") for benchmark in _BENCHMARKS)
        links = "; ".join(f"[{filename}]({_link(path, relative_to)})" for filename, (path, _) in records.items()) or "pending"
        selection = ("active" if name == pointer.get("relative_path") else "historical" if pointer
                     else "current (legacy default)" if name == "live-run" else "unselected")
        cells = [name, selection,
                 editor, state_cell, *[_recorded(counts.get(key), source, relative_to) for key in ("passes", "turn", "runs")],
                 _recorded("recorded", source, relative_to) if _submission_ready(counts) else "not established", evaluation, links]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _status_table(evidence: dict[str, tuple[Path, dict]], relative_to: Path, *, pointer: dict | None = None,
                  attempts: dict | None = None, summaries: list | None = None) -> str:
    pointer, attempts, summaries = pointer or {}, attempts or {}, summaries or []
    lines = ["| Recorded component | State | Evidence |", "| --- | --- | --- |"]
    live_blocker = evidence.get("live-run/blocker.json") if not pointer else None
    if pointer:
        selected = attempts[pointer["relative_path"]]
        source, state = selected.get("state.json", evidence["active-live.json"])
        ready = "state.json" in selected and _submission_ready(state)
        evaluations = _attempt_evaluations(pointer["relative_path"], selected, summaries)
        status = "submission and final evaluations recorded" if ready and len(evaluations) == 3 else "live goal unmet"
        detail = "submission recorded; final evaluation pending" if ready else "submission requirement pending"
        if "state.json" not in selected:
            detail = "pending: active state missing"
        elif "identity.json" not in selected:
            detail = "pending: active identity missing"
        lines.append(f"| Required live run | {_recorded(status, source, relative_to)} | Active attempt: {_recorded(pointer['run_id'], evidence['active-live.json'][0], relative_to)}; {detail if len(evaluations) != 3 else 'see linked attempt evidence'} |")
    if live_blocker:
        path, data = live_blocker
        state = data.get("status")
        details = _recorded(data.get("reason"), path, relative_to)
        if state == "blocked":
            details = _recorded("live goal unmet", path, relative_to) + "; " + details
        lines.append(f"| Required live run | {_recorded(state, path, relative_to)} | {details} |")
        live = data.get("live") or {}
        passes = _metric(live.get("passes"), path, relative_to, digits=0)
        required = _metric(data.get("required_live_passes"), path, relative_to, digits=0)
        turn = _metric(live.get("turn"), path, relative_to, digits=0)
        lines.append(f"| Live pass requirement | {passes} / {required} | Recorded turn: {turn} |")
        if data.get("final_evaluation_status") is not None:
            lines.append(f"| Live final evaluation | {_recorded(data['final_evaluation_status'], path, relative_to)} | See live-run record. |")
        if data.get("error"):
            label = "Recorded live blocker" if state == "blocked" else "Recorded live diagnostic"
            lines.append(f"| {label} | {_recorded(str(data['error'])[:500], path, relative_to)} | See live-run record. |")
    labels = {
        "pipeline-state.json": "Pipeline", "coordinator.json": "Driver recovery",
        "calibration-prompt-audit.json": "Calibration prompt audit",
        "calibration-check.json": "Calibration comparison", "kernel/hash-verification.json": "Tensor hashes",
        "kernel/sandbox-verification.json": "Sandbox and first cell", "kernel/forward-verification.json": "Forward service",
        "kernel/weight-swap.json": "Weight swap check",
    }
    pipeline = evidence.get("pipeline-state.json")
    for name, label in labels.items():
        if name not in evidence:
            continue
        path, data = evidence[name]
        state = data.get("status")
        diagnostic = data.get("error")
        details = data.get("active_stage") or data.get("error_type") or name
        if name == "coordinator.json":
            current = data.get("current") or {}
            state = current.get("phase", state)
            diagnostic = current.get("reason") or diagnostic
            details = current.get("stage") or details
        lines.append(f"| {label} | {_recorded(state, path, relative_to)} | {_recorded(details, path, relative_to)} |")
        if diagnostic:
            label = "Recorded blocker" if state in {"blocked", "failed", "error"} else "Recorded diagnostic"
            lines.append(f"| {label} | {_recorded(str(diagnostic)[:500], path, relative_to)} | See source record. |")
    if "live-run/state.json" in evidence and not pointer:
        path, data = evidence["live-run/state.json"]
        state = "submission recorded" if data.get("done") is True else "incomplete"
        counts = [f"{key}: {_metric(data.get(key), path, relative_to, digits=0)}" for key in ("passes", "turn", "runs")]
        lines.append(f"| Live editor | {_recorded(state, path, relative_to)} | {'; '.join(counts)} |")
        reused = data.get("base_generation_reuse_requests")
        if isinstance(reused, list) and reused:
            lines.append(f"| Base-generation reuse | {_recorded('used', path, relative_to)} | Requests: {_metric(len(reused), path, relative_to, digits=0)} |")
    if len(lines) == 2:
        lines.append("| Pipeline | pending: no published state record | No completion claim. |")
    elif pipeline is None:
        lines.append("| Pipeline | pending: no published pipeline state | Component checks do not establish overall completion. |")
    if live_blocker and live_blocker[1].get("status") == "blocked":
        lines += ["", "Completed controls or publication do not satisfy the blocked live-run goal."]
    if pointer or any(name != "live-run" for name in attempts):
        lines += ["", _attempt_table(pointer, attempts, summaries, relative_to, evidence.get("active-live.json", (None,))[0])]
        if pointer:
            lines += ["", "The active pointer selects the current attempt. Historical blockers and completed controls do not establish its completion."]
    return "\n".join(lines) + "\n\nStates are recorded snapshots; use `openwsp status` to check current liveness."


def _calibration_table(evidence: dict[str, tuple[Path, dict]], relative_to: Path) -> str:
    item = evidence.get("calibration-check.json")
    if item is None:
        return "Calibration comparison: pending."
    path, data = item
    lines = ["Calibration comparison:", "", "| Condition | Accuracy | Reference | Gap (points) | Within prescribed tolerance |",
             "| --- | --- | --- | --- | --- |"]
    for condition, row in data.items():
        if not isinstance(row, dict) or "accuracy" not in row:
            continue
        gap = row.get("gap")
        lines.append("| " + " | ".join((
            _text(condition), _metric(row.get("accuracy"), path, relative_to, percent=True),
            _metric(row.get("reference_accuracy"), path, relative_to, percent=True),
            _metric(gap * 100 if _number(gap) else None, path, relative_to, digits=2),
            _recorded(row.get("within_two_standard_errors"), path, relative_to),
        )) + " |")
    if data.get("status") != "passed":
        audit = evidence.get("calibration-gap-audit.json")
        if audit is None:
            lines += ["", "Calibration has a recorded gap; no circuit-breaker audit artifact is published."]
        else:
            audit_path, audit_data = audit
            lines += ["", (f"Circuit-breaker audit: {_recorded(audit_data.get('circuit_breaker_tripped'), audit_path, relative_to)}. "
                           f"Checks and remaining gap: [audit]({_link(audit_path, relative_to)}).")]
    return "\n".join(lines)


def _runtime_table(summaries: list[tuple[Path, dict]], relative_to: Path) -> str:
    lines = ["Recorded runtime and uncertainty:", "", "| Stage / benchmark | Engine | GPU | Transformers | Uncertainty method |",
             "| --- | --- | --- | --- | --- |"]
    for path, data in summaries:
        provenance = data.get("provenance") or {}
        lines.append("| " + " | ".join([
            _text(f"{data.get('stage', 'pending')} / {data.get('benchmark', 'pending')}"),
            *[_recorded(provenance.get(key), path, relative_to) for key in ("engine", "gpu", "transformers", "standard_error")],
        ]) + " |")
    if not summaries:
        lines.append("| pending | pending | pending | pending | pending |")
    return "\n".join(lines)


def _validate_summary(summary: dict[str, Any], path: Path) -> None:
    """Refuse impossible scores and stale totals where per-sample evidence is present."""
    for key in ("accuracy", "standard_error", "cut_share"):
        value = summary.get(key)
        if value is not None and (not _number(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid {key} in {path}")
    for key in ("n", "correct", "samples_per_problem", "mean_tokens", "seconds", "cost_usd"):
        value = summary.get(key)
        if value is not None and (not _number(value) or value < 0):
            raise ValueError(f"Invalid {key} in {path}")
    if _number(summary.get("n")) and _number(summary.get("correct")):
        if summary["correct"] > summary["n"]:
            raise ValueError(f"Correct count exceeds sample count in {path}")
        if summary["n"] > 0 and _number(summary.get("accuracy")):
            expected = summary["correct"] / summary["n"]
            if not math.isclose(summary["accuracy"], expected, abs_tol=1e-9):
                raise ValueError(f"Accuracy and counts disagree in {path}")
    samples = path.with_name("samples.jsonl")
    if samples.exists():
        count, correct = 0, 0
        seen = set()
        with samples.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("problem_id"), row.get("sample_index", row.get("sample")))
                if key in seen:
                    raise ValueError(f"Duplicate sample identity in {samples}: {key}")
                seen.add(key)
                if row.get("correct") not in (0, 1, False, True):
                    raise ValueError(f"Invalid sample grade in {samples}")
                count += 1
                correct += int(row["correct"])
        for key, measured in (("n", count), ("correct", correct)):
            if summary.get(key) is not None and summary[key] != measured:
                raise ValueError(f"Summary {key} disagrees with {samples}")


def _results_table(
    references: dict[str, tuple[Path, dict[str, Any]]],
    summaries: list[tuple[Path, dict[str, Any]]],
    relative_to: Path,
) -> str:
    lines = [
        "Authors' released evaluations:",
        "",
        "| Run | MATH-500 | GSM8K | IFEval | MATH-500 hit token limit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in _REFERENCES:
        if name not in references:
            lines.append(f"| {name} | pending | pending | pending | pending |")
            continue
        path, data = references[name]
        scores = [
            _metric(data.get(bench, {}).get("acc"), path, relative_to, percent=True) for bench in _BENCHMARKS
        ]
        cut = _metric(data.get("math500", {}).get("cut_share"), path, relative_to, percent=True)
        lines.append(f"| {name} | {' | '.join(scores)} | {cut} |")
    lines += [
        "",
        "Our evaluations and controls (controls are ours):",
        "",
        "| Stage | Model | Benchmark | Author comparator | Accuracy | Standard error | Samples | Hit token limit | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for path, data in summaries:
        stage, benchmark = data.get("stage"), data.get("benchmark")
        reference = data.get("reference", "p50c" if str(stage).startswith("live-runs/") else _COMPARATORS.get(stage))
        comparator = "not applicable" if str(stage).startswith("control") else "pending"
        if reference in references:
            ref_path, ref_data = references[reference]
            acc = _metric(ref_data.get(benchmark, {}).get("acc"), ref_path, relative_to, percent=True)
            comparator = f"{_text(reference)}: {acc}"
        values = [
            _text(stage),
            _text(data.get("model")),
            _text(benchmark),
            comparator,
            _metric(data.get("accuracy"), path, relative_to, percent=True),
            _metric(data.get("standard_error"), path, relative_to, percent=True, digits=2),
            _metric(data.get("n"), path, relative_to, digits=0),
            _metric(data.get("cut_share"), path, relative_to, percent=True),
            f"[{_text(data.get('status'), 'unspecified')}]({_link(path, relative_to)})",
        ]
        lines.append("| " + " | ".join(values) + " |")
    if not summaries:
        lines.append(
            "| pending | pending | pending | pending | pending | pending | pending | pending | pending |"
        )
    lines += [
        "",
        "Only recorded values are shown. Missing fields remain pending; partial runs retain their recorded status.",
    ]
    return "\n".join(lines)


def _cost_stage(stage: str) -> str:
    """Group known job identities without merging conditions or benchmarks."""
    stage = re.sub(r"/openwsp-gpu:[A-Za-z0-9_-]+:\d+$", "", stage)
    live = re.fullmatch(r"live/fwd_\d+/(original|edited)(?:/(activations|fresh-fallback))?", stage)
    if live:
        model, operation = live.groups()
        operation = "activation extraction" if operation == "activations" else "generation"
        return f"live / {operation} / {model}"
    if re.fullmatch(r"live/fwd_\d+/teacher_forced", stage):
        return "live / teacher-forced"
    evaluation = re.fullmatch(r"evaluation/([^/]+)/([^/]+)/(batch_\d+|smoke)", stage)
    if evaluation:
        condition, benchmark, batch = evaluation.groups()
        operation = "evaluation smoke" if batch == "smoke" else "evaluation"
        return f"{condition} / {benchmark} / {operation}"
    directions = re.fullmatch(r"control2/directions/(batch_\d+|smoke)", stage)
    if directions:
        return "control2 / directions smoke" if directions[1] == "smoke" else "control2 / directions"
    return stage.replace("/", " / ")


def _cost_basis(row: dict[str, Any], kind: str) -> str:
    receipt = row.get("receipt_metadata") or {}
    basis = receipt.get("billing_basis") or receipt.get("cost_basis")
    if basis:
        return str(basis)
    if receipt.get("invoice") is True:
        return "provider invoice"
    if receipt.get("failed_job_upper_bound") is True:
        return "failed job upper bound; not invoice"
    if receipt.get("source") == "cancelled_gpu_resource_upper_bound_not_invoice":
        return "cancelled job upper bound; not invoice"
    return "API usage and recorded token rates" if kind == "editor" else "resource estimate; not invoice"


def _costs_table(summaries: list[tuple[Path, dict[str, Any]]], relative_to: Path,
                 ledgers: list[tuple[Path, dict]] | None = None) -> str:
    lines = [
        "| Stage | Benchmark | Recorded hours | Recorded cost (USD) |",
        "| --- | --- | --- | --- |",
    ]
    for path, data in summaries:
        seconds = data.get("seconds")
        hours = seconds / 3600 if _number(seconds) else None
        lines.append(
            "| "
            + " | ".join(
                [
                    _text(data.get("stage")),
                    _text(data.get("benchmark")),
                    _metric(hours, path, relative_to, digits=3),
                    _metric(data.get("cost_usd"), path, relative_to, digits=4),
                ]
            )
            + " |"
        )
    if not summaries:
        lines.append("| pending | pending | pending | pending |")
    lines += [
        "",
        "Costs and durations come from the linked records. Missing costs are not treated as zero; rows are not summed because stages may overlap.",
    ]
    for path, ledger in ledgers or []:
        reservations = ledger.get("reservations")
        if not isinstance(reservations, dict):
            raise TypeError(f"Cost ledger reservations must be an object in {path}")
        groups: dict[tuple, list[dict]] = {}
        for row in reservations.values():
            if not isinstance(row, dict):
                raise TypeError(f"Cost ledger entries must be objects in {path}")
            kind = row.get("kind", "pending")
            stage = "editor turns" if kind == "editor" else _cost_stage(str(row.get("stage", "pending")))
            basis = _cost_basis(row, kind)
            key = (kind, stage, row.get("status", "unspecified"), basis)
            groups.setdefault(key, []).append(row)
        lines += ["", f"Budget ledger: [source]({_link(path, relative_to)}). Reservations are retained bounds, not charges.", "",
                  "| Budget / stage / operation | State | Receipts | Recorded hours | Settled amount (USD) | Reserved bound (USD) | Basis |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for (kind, stage, status, basis), rows in sorted(groups.items()):
            seconds = [_ledger_number(row.get("seconds"), path) for row in rows]
            duration = sum(seconds) / 3600 if all(value is not None for value in seconds) else None
            amounts = [_ledger_number(row.get("actual_usd"), path) for row in rows]
            bounds = [_ledger_number(row.get("upper_bound_usd"), path) for row in rows]
            settled = sum(amounts) if status == "settled" and all(value is not None for value in amounts) else None
            reserved = sum(bounds) if status == "reserved" and all(value is not None for value in bounds) else None
            lines.append("| " + " | ".join((
                _text(f"{kind} / {stage}"), _recorded(status, path, relative_to),
                _metric(len(rows), path, relative_to, digits=0),
                _metric(duration, path, relative_to, digits=3), _metric(settled, path, relative_to, digits=4),
                _metric(reserved, path, relative_to, digits=4), _recorded(basis, path, relative_to),
            )) + " |")
        if not groups:
            lines.append("| No recorded jobs | pending | pending | pending | pending | pending | Ledger has no receipts. |")
        lines += ["", ("Repeated requests, attempts and batches are grouped by stage and operation; the linked ledger retains each receipt. "
                       "Ledger rows and evaluation summaries may describe the same jobs; do not add the two tables. "
                       "Summed resource hours are not end-to-end wall time. Settled resource estimates remain estimates until reconciled with provider billing.")]
    return "\n".join(lines)


def _ledger_number(value: Any, path: Path) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid cost value in {path}") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid cost value in {path}")
    return number


def _replace_section(readme: str, name: str, body: str) -> str:
    start, end = f"<!-- OPENWSP:{name}:START -->", f"<!-- OPENWSP:{name}:END -->"
    if start not in readme and end not in readme:
        return readme.rstrip() + f"\n\n{start}\n{body}\n{end}\n"
    if readme.count(start) != 1 or readme.count(end) != 1 or readme.index(start) >= readme.index(end):
        raise ValueError(f"README has malformed or duplicate {name} markers")
    before, rest = readme.split(start)
    _, after = rest.split(end)
    return before + start + "\n" + body + "\n" + end + after


def generate_report(repo_root: str | Path, results_dir: str | Path) -> dict[str, Any]:
    """Refresh README tables and a standalone report from summaries and upstream files."""
    repo = Path(repo_root).resolve()
    results = Path(results_dir)
    results = results.resolve() if results.is_absolute() else (repo / results).resolve()
    pointer, attempts, attempt_sources = _attempt_evidence(results)
    references = {}
    reference_dir = repo / "replications" / "exp8" / "reference"
    for name in _REFERENCES:
        path = reference_dir / f"{name}.json"
        if path.exists():
            references[name] = (path, _read_json(path))
    summaries = []
    for path in sorted(results.rglob("summary.json")):
        parts = path.relative_to(results).parts
        if len(parts) >= 3 and parts[0] == "live-runs":
            attempt_dir = results / parts[0] / parts[1]
            if not path.resolve().is_relative_to(attempt_dir.resolve()):
                raise ValueError("Live evaluation summary escapes its attempt directory")
        data = _read_json(path)
        _validate_summary(data, path)
        summaries.append((path, data))
    evidence = {name: (results / name, _read_json(results / name)) for name in _STATE_FILES if (results / name).exists()}
    if pointer:
        evidence["active-live.json"] = (results / "active-live.json", pointer)
    ledgers = [(path, _read_json(path)) for path in sorted(results.rglob("costs.json"))]
    extra_costs = []
    for path in sorted(results.rglob("cost.json")):
        data = _read_json(path)
        _validate_summary(data, path)
        extra_costs.append((path, data))
    def result_body(relative_to):
        return "\n\n".join((status_body(relative_to), _results_table(references, summaries, relative_to),
                            _calibration_table(evidence, relative_to), _runtime_table(summaries, relative_to)))
    def status_body(relative_to):
        return _status_table(evidence, relative_to, pointer=pointer, attempts=attempts, summaries=summaries)
    def cost_body(relative_to):
        return _costs_table(summaries + extra_costs, relative_to, ledgers)
    readme_path = repo / "README.md"
    readme = readme_path.read_text() if readme_path.exists() else "# OpenWSP\n"
    if "<!-- OPENWSP:STATUS:" in readme:
        readme = _replace_section(readme, "STATUS", status_body(repo))
    readme = _replace_section(readme, "RESULTS", result_body(repo))
    readme = _replace_section(readme, "COSTS", cost_body(repo))
    report_path = results / "report.md"
    report = "# Recorded evaluation results\n\n" + result_body(results)
    report += "\n\n" + cost_body(results) + "\n"
    # Validate all inputs and both replacements before changing published files.
    _atomic_write(report_path, report)
    _atomic_write(readme_path, readme)
    return {
        "readme": str(readme_path),
        "report": str(report_path),
        "summaries": len(summaries),
        "sources": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in dict.fromkeys([*(item[0] for item in references.values()), *(item[0] for item in summaries),
                         *(item[0] for item in evidence.values()), *attempt_sources,
                         *(item[0] for item in ledgers), *(item[0] for item in extra_costs)])
        ],
    }


def _redact(text: str) -> str:
    return _BEARER.sub(r"\1[REDACTED]", _SECRET.sub("[REDACTED]", text))


def _fence(text: str, language: str = "") -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{_redact(text).rstrip()}\n{fence}"


def _compact(value: Any) -> Any:
    """Keep shapes, counts and paths while excluding structured raw text/array payloads."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = key.lower()
            if lowered in {"api_key", "access_token", "token_secret", "password", "authorization"}:
                result[key] = "[REDACTED]"
            elif lowered in {
                "texts",
                "chats",
                "prompts",
                "samples",
                "token_ids",
                "raw_text",
                "generated_text",
            }:
                result[key] = "[Raw content omitted; see run volume.]"
            elif lowered == "generations" and isinstance(item, list):
                result[key] = "[Raw generations omitted; see run volume.]"
            else:
                result[key] = _compact(item)
        return result
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def _tool_record(record: dict[str, Any]) -> str:
    parts = []
    if record.get("code"):
        parts.append(_fence(str(record["code"]), "python"))
    for key in ("out", "output", "error", "result"):
        if record.get(key) is not None:
            value = record[key]
            text = value if isinstance(value, str) else json.dumps(_compact(value), indent=2)
            if len(text) > 20000:
                text = text[:20000] + "\n[Long tool output omitted; see source record.]"
            parts.append(_fence(text, "text"))
    return "\n\n".join(parts) or "No code or compact output was recorded."


def _editor_event(record: dict[str, Any]) -> str:
    """Render public event metadata without opening its private response reference."""
    fields = ("status", "reason", "response_id", "executed", "usage", "cost_usd",
              "partial_text_characters", "partial_text_sha256", "response_volume", "response_volume_path")
    metadata = {key: record[key] for key in fields if key in record}
    execution = ("No code or forward pass from this response was executed."
                 if record.get("executed") is False else "Execution status is shown only as recorded below.")
    return "\n\n".join((execution, "Partial response text remains private and is not included here.",
                        _fence(json.dumps(_compact(metadata), indent=2), "json")))


def render_transcript(run_dir: str | Path, output_path: str | Path) -> Path:
    """Render editor messages and compact tool outputs; never load generation/activation files."""
    run, output = Path(run_dir).resolve(), Path(output_path).resolve()
    calls_by_turn: dict[int, list[dict[str, Any]]] = {}
    unassigned = []
    calls = run / "tool_calls.jsonl"
    if calls.exists():
        with calls.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                turn = str(record.get("turn", ""))
                if re.fullmatch(r"t?\d+", turn):
                    calls_by_turn.setdefault(int(turn.lstrip("t")), []).append(record)
                else:
                    unassigned.append(record)
    parts = [
        "# Live run transcript",
        "This transcript contains recorded editor replies and compact tool outputs. Raw generations and activation arrays remain in the run volume.",
    ]
    turns = sorted(
        (path for path in (run / "turns").glob("t[0-9]*") if re.fullmatch(r"t\d+", path.name)),
        key=lambda path: int(path.name[1:]),
    )
    if not turns:
        parts.append("Pending: no editor turns have been recorded.")
    for directory in turns:
        if not re.fullmatch(r"t\d+", directory.name):
            continue
        number = int(directory.name[1:])
        parts.append(f"## Turn {number}")
        prompt = directory / "prompt.txt"
        if prompt.exists():
            text = prompt.read_text()
            if len(text) > 20000:
                text = text[:20000] + "\n[Long input omitted; see source record.]"
            parts += ["### Input", _fence(text, "text")]
        answer = directory / "answer.md"
        if answer.exists():
            parts += ["### Editor", _redact(answer.read_text()).strip()]
        elif (directory / "editor_event.json").exists():
            parts += ["### Editor event", _editor_event(_read_json(directory / "editor_event.json"))]
        else:
            parts.append("Editor reply pending.")
        for record in calls_by_turn.pop(number, []):
            parts += ["### Tool result", _tool_record(record)]
    # Do not guess which turn an old harness record belongs to.
    unassigned += [record for records in calls_by_turn.values() for record in records]
    if unassigned:
        parts += [
            "## Other tool records",
            "These records lack a matching turn identifier and are kept separately.",
        ]
        for index, record in enumerate(unassigned, 1):
            parts += [f"### Tool record {index}", _tool_record(record)]
    requests = sorted(
        (run / "requests").glob("*.done.json"),
        key=lambda path: (
            int(re.search(r"\d+", path.name).group()) if re.search(r"\d+", path.name) else 0,
            path.name,
        ),
    )
    if requests:
        parts.append("## Forward-pass summaries")
        for path in requests:
            summary = _read_json(path)
            # A summary may reference raw files, but they are never opened here.
            parts += [f"### {path.name}", _tool_record({"result": summary})]
    final = run / "final_answer.md"
    if final.exists():
        parts += ["## Final submission", _redact(final.read_text()).strip()]
    _atomic_write(output, "\n\n".join(parts) + "\n")
    return output
