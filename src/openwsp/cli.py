"""Command line adapters for the shared core API and replication pipelines."""

from __future__ import annotations

import json
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
    foreground: bool = False,
) -> None:
    """Launch or resume a replication pipeline."""
    from .replication import replicate as run_replication

    _emit(
        run_replication(
            experiment, stage=stage, detach=not foreground, config=str(config) if config else None
        )
    )


@app.command("status")
def status(experiment: Annotated[str, typer.Argument()] = "exp8") -> None:
    """Show current replication progress and budget."""
    from .replication import status as get_status

    _emit(get_status(experiment))


@app.command("report")
def report(experiment: Annotated[str, typer.Argument()] = "exp8") -> None:
    """Generate result tables from saved evaluation records."""
    from .replication import report as write_report

    _emit(write_report(experiment))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
