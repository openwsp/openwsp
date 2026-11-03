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


def _costs_table(summaries: list[tuple[Path, dict[str, Any]]], relative_to: Path) -> str:
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
    return "\n".join(lines)


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
    readme_path = repo / "README.md"
    readme = readme_path.read_text() if readme_path.exists() else "# OpenWSP\n"
    readme = _replace_section(readme, "RESULTS", _results_table(references, summaries, repo))
    readme = _replace_section(readme, "COSTS", _costs_table(summaries, repo))
    report_path = results / "report.md"
    report = "# Recorded evaluation results\n\n" + _results_table(references, summaries, results)
    report += "\n\n" + _costs_table(summaries, results) + "\n"
    # Validate all inputs and both replacements before changing published files.
    _atomic_write(report_path, report)
    _atomic_write(readme_path, readme)
    return {
        "readme": str(readme_path),
        "report": str(report_path),
        "summaries": len(summaries),
        "sources": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [*(item[0] for item in references.values()), *(item[0] for item in summaries)]
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
