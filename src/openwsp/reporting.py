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
_SECRET = re.compile(r"(?:sk-|hf_|ak-|as-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}")
_BEARER = re.compile(r"(?i)(authorization[\"'\s:]+bearer\s+)[A-Za-z0-9._~+/-]+")
_STATE_FILES = (
    "pipeline-state.json", "coordinator.json", "live-run/state.json", "calibration-check.json", "calibration-prompt-audit.json",
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


def _status_table(evidence: dict[str, tuple[Path, dict]], relative_to: Path) -> str:
    lines = ["| Recorded component | State | Evidence |", "| --- | --- | --- |"]
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
        if name == "coordinator.json":
            state = (data.get("current") or {}).get("phase", state)
        details = data.get("active_stage") or data.get("error_type") or name
        lines.append(f"| {label} | {_recorded(state, path, relative_to)} | {_recorded(details, path, relative_to)} |")
        if data.get("error"):
            lines.append(f"| Recorded blocker | {_recorded(str(data['error'])[:500], path, relative_to)} | See source record. |")
    if "live-run/state.json" in evidence:
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
    return "\n".join(lines)


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
        reference = data.get("reference", _COMPARATORS.get(stage))
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
            stage = str(row.get("stage", "pending"))
            if kind == "editor":
                stage = "editor turns"
            receipt = row.get("receipt_metadata") or {}
            basis = receipt.get("billing_basis") or receipt.get("cost_basis")
            if not basis:
                basis = "API usage and recorded token rates" if kind == "editor" else "resource estimate; not invoice"
            key = (kind, stage, row.get("status", "unspecified"), str(basis))
            groups.setdefault(key, []).append(row)
        lines += ["", f"Budget ledger: [source]({_link(path, relative_to)}). Reservations are retained bounds, not charges.", "",
                  "| Budget / stage | State | Recorded hours | Settled amount (USD) | Reserved bound (USD) | Basis |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for (kind, stage, status, basis), rows in sorted(groups.items()):
            seconds = [_ledger_number(row.get("seconds"), path) for row in rows]
            duration = sum(seconds) / 3600 if all(value is not None for value in seconds) else None
            amounts = [_ledger_number(row.get("actual_usd"), path) for row in rows]
            bounds = [_ledger_number(row.get("upper_bound_usd"), path) for row in rows]
            settled = sum(amounts) if status == "settled" and all(value is not None for value in amounts) else None
            reserved = sum(bounds) if status == "reserved" and all(value is not None for value in bounds) else None
            lines.append("| " + " | ".join((
                _text(f"{kind} / {stage}"), _recorded(status, path, relative_to),
                _metric(duration, path, relative_to, digits=3), _metric(settled, path, relative_to, digits=4),
                _metric(reserved, path, relative_to, digits=4), _recorded(basis, path, relative_to),
            )) + " |")
        if not groups:
            lines.append("| No recorded jobs | pending | pending | pending | pending | Ledger has no receipts. |")
        lines += ["", ("Ledger rows and evaluation summaries may describe the same jobs; do not add the two tables. "
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
    references = {}
    reference_dir = repo / "replications" / "exp8" / "reference"
    for name in _REFERENCES:
        path = reference_dir / f"{name}.json"
        if path.exists():
            references[name] = (path, _read_json(path))
    summaries = []
    for path in sorted(results.rglob("summary.json")):
        data = _read_json(path)
        _validate_summary(data, path)
        summaries.append((path, data))
    evidence = {name: (results / name, _read_json(results / name)) for name in _STATE_FILES if (results / name).exists()}
    ledgers = [(path, _read_json(path)) for path in sorted(results.rglob("costs.json"))]
    extra_costs = []
    for path in sorted(results.rglob("cost.json")):
        data = _read_json(path)
        _validate_summary(data, path)
        extra_costs.append((path, data))
    def result_body(relative_to):
        return "\n\n".join((_status_table(evidence, relative_to), _results_table(references, summaries, relative_to),
                            _calibration_table(evidence, relative_to), _runtime_table(summaries, relative_to)))
    def cost_body(relative_to):
        return _costs_table(summaries + extra_costs, relative_to, ledgers)
    readme_path = repo / "README.md"
    readme = readme_path.read_text() if readme_path.exists() else "# OpenWSP\n"
    if "<!-- OPENWSP:STATUS:" in readme:
        readme = _replace_section(readme, "STATUS", _status_table(evidence, repo))
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
            for path in [*(item[0] for item in references.values()), *(item[0] for item in summaries),
                         *(item[0] for item in evidence.values()), *(item[0] for item in ledgers), *(item[0] for item in extra_costs)]
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
