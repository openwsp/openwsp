"""The MCP transport for the same API used by Python and the CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from openwsp.api import OpenWSP, load_backend


def create_server(api: OpenWSP) -> FastMCP:
    """Bind one persistent weight session to an MCP stdio server."""
    server = FastMCP("OpenWSP")

    @server.tool()
    def list_tensors() -> dict[str, Any]:
        """List tensor names, shapes, and dtypes without returning their values."""
        return {"tensors": api.list_tensors()}

    @server.tool()
    def read_tensor(name: str, offset: int = 0, limit: int = 256) -> dict[str, Any]:
        """Read a bounded flat slice of a current tensor (at most 4096 values)."""
        return api.read_tensor(name, offset=offset, limit=limit)

    @server.tool()
    async def run_code(code: str) -> dict[str, Any]:
        """Execute a Python cell in this session's persistent weight kernel."""
        # Keep the serial kernel on the main thread so its signal deadline also
        # interrupts native calls; Keep native-call deadlines independent of transport thread behavior.
        return api.run_code(code)

    @server.tool()
    def forward(request: dict[str, Any]) -> dict[str, Any]:
        """Run a forward request through the configured compute backend."""
        return api.forward(request)

    @server.tool()
    def save_edit(name: str, values: list[Any]) -> dict[str, Any]:
        """Save replacement values for one tensor after the core's refusal checks."""
        return api.save_edit(name, values)

    @server.tool()
    def diff() -> dict[str, Any]:
        """Describe changed tensors relative to the pinned base weights."""
        return {"tensors": api.diff()}

    @server.tool()
    def apply_edit(path: str) -> dict[str, Any]:
        """Validate a manifest and payload against this base, then apply its edit."""
        return api.apply_edit(path)

    @server.tool()
    def export_edit(
        path: str, base_model: str | None = None, revision: str | None = None
    ) -> dict[str, Any]:
        """Write a versioned edit manifest and payload, identified by sha256."""
        return api.export_edit(path, base_model=base_model, revision=revision)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenWSP MCP server (stdio)")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=Path, help="Base safetensors file or model directory")
    model.add_argument("--tiny", action="store_true", help="Use the CPU test network")
    parser.add_argument("--work-dir", type=Path, default=Path(".openwsp"))
    parser.add_argument("--practice-dir", type=Path)
    parser.add_argument("--backend", help="Importable module:factory for a forward backend")
    args = parser.parse_args()
    if args.tiny:
        if args.backend:
            parser.error("--tiny supplies its own backend")
        api = OpenWSP.tiny(args.work_dir)
    else:
        api = OpenWSP(
            args.model,
            work_dir=args.work_dir,
            practice_dir=args.practice_dir,
            backend=load_backend(args.backend),
        )
    create_server(api).run(transport="stdio")


if __name__ == "__main__":
    main()
