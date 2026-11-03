"""Command line adapters for the shared core API and replication pipelines."""

from __future__ import annotations

import json
import math
import shutil
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(no_args_is_help=True, help="Read, edit, execute and version model weights.")


def _emit(value: Any) -> None:
    typer.echo(value if isinstance(value, str) else json.dumps(value, indent=2, default=str))


def _json(value: str) -> Any:
    try:
        return json.loads(Path(value[1:]).read_text() if value.startswith("@") else value)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(f"Expected JSON or @path/to/file.json: {exc}") from exc


def _status_text(experiment: str, state: dict[str, Any]) -> str:
    """Show bounded, recorded progress without changing the SDK state."""
    width = min(100, max(40, shutil.get_terminal_size(fallback=(80, 24)).columns))

    def amount(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "pending"
        return f"${number:.4f}" if not isinstance(value, bool) and math.isfinite(number) else "pending"

    status = state.get("status", state.get("state", "unknown"))
    stage = state.get("active_stage", state.get("stage", "not recorded"))
    lines = [f"{experiment}: {status} | stage: {stage}"]
    live = state.get("live")
    if isinstance(live, dict) and live:
        counts = " | ".join(f"{key} {live.get(key, 'pending')}" for key in ("turn", "passes", "runs", "requests"))
        submitted = "yes" if live.get("done") is True else "no" if live.get("done") is False else "pending"
        lines.append(f"Live: {counts} | submitted {submitted}")
    else:
        lines.append("Live: not recorded")
    budgets = state.get("budget") or {}
    for kind in ("modal", "editor"):
        budget = budgets.get(kind) if isinstance(budgets, dict) else None
        values = " | ".join(f"{label} {amount(budget.get(key))}" for label, key in (
            ("spent", "spent_usd"), ("reserved", "reserved_usd"), ("available", "available_usd"),
        )) if isinstance(budget, dict) else "not recorded"
        lines.append(f"{kind.title()} USD: {values}")
    heartbeat = state.get("heartbeat")
    try:
        if isinstance(heartbeat, bool):
            raise TypeError("A boolean is not a heartbeat")
        timestamp = float(heartbeat)
        utc = datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        age = time.time() - timestamp
        elapsed = f"{age:.0f}s ago" if age >= 0 else f"{-age:.0f}s ahead of local clock"
        lines.append(f"Heartbeat: {utc} ({elapsed})")
    except (TypeError, ValueError, OverflowError, OSError):
        lines.append("Heartbeat: not recorded or invalid")
    if state.get("error") or state.get("error_type"):
        lines.append(f"Last error: {state.get('error') or state['error_type']}")
    if state.get("controls_deferred"):
        lines.append(f"Controls note: {state['controls_deferred']}")
    publication = state.get("publication")
    if isinstance(publication, dict) and publication:
        lines.append(f"Last publication: {publication.get('status', 'unknown')}")
    lines.extend(("Recorded snapshot; 'running' does not confirm current liveness.",
                  f"Full state: openwsp status {experiment} --json"))
    return "\n".join(textwrap.shorten("".join(char if char.isprintable() else " " for char in line),
                                     width=width, placeholder="...") for line in lines)


@app.callback()
def configure(
    ctx: typer.Context,
    model: Annotated[Path | None, typer.Option(help="Model safetensors path.")] = None,
    work_dir: Annotated[Path, typer.Option(help="Persistent edits and kernel working directory.")] = Path(
        ".openwsp"
    ),
    practice_dir: Annotated[Path | None, typer.Option(help="Practice data path.")] = None,
    backend: Annotated[str | None, typer.Option(help="Installed backend module:factory.")] = None,
    tiny: Annotated[bool, typer.Option(help="Use the tiny CPU demonstration network.")] = False,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj.update(model=model, work_dir=work_dir, practice_dir=practice_dir, backend=backend, tiny=tiny)


def _api(ctx: typer.Context):
    from .api import OpenWSP, load_backend

    options = ctx.obj
    if "api" not in options:
        if options["tiny"]:
            if options["model"] is not None or options["backend"] is not None:
                raise typer.BadParameter("--tiny cannot be combined with --model or --backend")
            options["api"] = OpenWSP.tiny(options["work_dir"])
        else:
            if options["model"] is None:
                raise typer.BadParameter("Supply --model or --tiny before the command")
            options["api"] = OpenWSP(
                options["model"],
                work_dir=options["work_dir"],
                practice_dir=options["practice_dir"],
                backend=load_backend(options["backend"]),
            )
    return options["api"]


@app.command("tensors")
def tensors(ctx: typer.Context) -> None:
    """List tensor names and shapes."""
    _emit(_api(ctx).list_tensors())


@app.command("read")
def read_tensor(ctx: typer.Context, name: str, offset: int = 0, limit: int = 256) -> None:
    """Read a flattened tensor slice (at most 4096 values)."""
    _emit(_api(ctx).read_tensor(name, offset=offset, limit=limit))


@app.command("run")
def run_code(ctx: typer.Context, code: str | None = None, file: Path | None = None) -> None:
    """Execute a Python cell with --code or --file."""
    if (code is None) == (file is None):
        raise typer.BadParameter("Supply exactly one of --code or --file")
    result = _api(ctx).run_code(code if code is not None else file.read_text())
    _emit(result)
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command("forward")
def forward(ctx: typer.Context, request: str) -> None:
    """Execute a JSON forward request, or read one from @file.json."""
    _emit(_api(ctx).forward(_json(request)))


@app.command("save")
def save_edit(ctx: typer.Context, name: str, values: str) -> None:
    """Save a tensor from JSON values or @file.json."""
    _emit(_api(ctx).save_edit(name, _json(values)))


@app.command("diff")
def diff(ctx: typer.Context) -> None:
    """Describe changes from the original weights."""
    _emit(_api(ctx).diff())


@app.command("apply")
def apply_edit(ctx: typer.Context, path: Path) -> None:
    """Validate and apply an edit manifest and payload."""
    _emit(_api(ctx).apply_edit(path))


@app.command("inspect")
def inspect_edit(ctx: typer.Context, path: Path | None = None, compare: Path | None = None,
                 top_directions: int = 3, rank_rtol: float | None = None,
                 embedding_name: str | None = None, token_ids: str | None = None,
                 output: Path | None = None) -> None:
    """Inspect a legacy/v1 edit (--path) or current edits, with optional comparison and logit lens."""
    result = _api(ctx).inspect(path, compare=compare, top_directions=top_directions, rank_rtol=rank_rtol,
                               embedding_name=embedding_name, token_ids=_json(token_ids) if token_ids else None)
    if output is None:
        _emit(result)
    else:
        from .budget import atomic_json
        atomic_json(output, result)
        _emit({"written": str(output), "format": result["format"]})


@app.command("export")
def export_edit(
    ctx: typer.Context, path: Path, base_model: str | None = None, revision: str | None = None
) -> None:
    """Export the current edit as a versioned, hashed diff."""
    _emit(_api(ctx).export_edit(path, base_model=base_model, revision=revision))


@app.command("replicate")
def replicate(
    experiment: str,
    stage: str = "all",
    config: Path | None = None,
    foreground: Annotated[
        bool, typer.Option(help="Wait locally for completion; required for editor='codex'.")
    ] = False,
) -> None:
    """Launch or resume a replication pipeline."""
    from .replication import replicate as run_replication

    try:
        result = run_replication(
            experiment, stage=stage, detach=not foreground, config=str(config) if config else None
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    _emit(result)


@app.command("status")
def status(
    experiment: Annotated[str, typer.Argument()] = "exp8",
    json_output: Annotated[bool, typer.Option("--json", help="Print the complete machine-readable state.")] = False,
) -> None:
    """Show replication progress on one screen; use --json for full details."""
    from .replication import status as get_status

    state = get_status(experiment)
    _emit(state if json_output else _status_text(experiment, state))


@app.command("report")
def report(experiment: Annotated[str, typer.Argument()] = "exp8") -> None:
    """Generate result tables from saved evaluation records."""
    from .replication import report as write_report

    _emit(write_report(experiment))


@app.command("record-modal-allowance")
def record_modal_allowance(
    observation: Annotated[Path, typer.Option(help="Absolute path to user-confirmed dashboard evidence JSON.")],
    experiment: Annotated[str, typer.Argument()] = "exp8",
) -> None:
    """Record Modal allowance evidence without deploying or restarting work."""
    from .replication import record_modal_allowance as record_allowance

    try:
        result = record_allowance(experiment, observation=observation)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _emit(result)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
