"""Drive every MCP tool over its actual stdio transport on a CPU network."""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def decoded(result):
    assert not result.isError, result.content
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(next(item.text for item in result.content if item.type == "text"))


@pytest.mark.asyncio
async def test_every_tool_through_stdio(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "openwsp.mcp_server", "--tiny", "--work-dir", str(tmp_path / "session")],
        env=env,
    )
    async with stdio_client(params) as (reader, writer), ClientSession(
        reader, writer, read_timeout_seconds=timedelta(seconds=20)
    ) as client:
        await client.initialize()
        tools = {tool.name for tool in (await client.list_tools()).tools}
        assert tools == {
            "list_tensors", "read_tensor", "run_code", "forward", "save_edit",
            "diff", "apply_edit", "export_edit",
        }
        listed = decoded(await client.call_tool("list_tensors", {}))
        assert {row["name"] for row in listed["tensors"]} == {
            "layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"
        }
        read = decoded(await client.call_tool(
            "read_tensor", {"name": "layer1.weight", "offset": 1, "limit": 2}
        ))
        assert read["values"] == [2.0, -1.0]
        assert read["shape"] == [2, 2]
        before = decoded(await client.call_tool(
            "forward", {"request": {"inputs": [[1.0, 0.0]], "model": "edited"}}
        ))
        assert before["backend"] == "tiny-cpu"
        cell = decoded(await client.call_tool("run_code", {"code": "sentinel = 7\nsentinel + 3"}))
        assert cell["ok"] and "10" in cell["output"]
        cell = decoded(await client.call_tool("run_code", {"code": "sentinel + 4"}))
        assert cell["ok"] and "11" in cell["output"]
        decoded(await client.call_tool(
            "save_edit", {"name": "layer1.weight", "values": [[2.0, 2.0], [-1.0, 1.0]]}
        ))
        diff = decoded(await client.call_tool("diff", {}))
        assert [row["name"] for row in diff["tensors"]] == ["layer1.weight"]
        after = decoded(await client.call_tool(
            "forward", {"request": {"inputs": [[1.0, 0.0]], "model": "both"}}
        ))
        assert after["original"]["logits"] == before["logits"]
        assert after["edited"]["logits"] != before["logits"]
        bundle = tmp_path / "exported-edit"
        exported = decoded(await client.call_tool(
            "export_edit", {"path": str(bundle), "base_model": "tiny", "revision": "test-v1"}
        ))
        assert (bundle / "manifest.json").is_file()
        assert (bundle / "edit.npz").is_file()
        decoded(await client.call_tool("run_code", {"code": "drop_edit('layer1.weight')"}))
        assert decoded(await client.call_tool("diff", {}))["tensors"] == []
        applied = decoded(await client.call_tool("apply_edit", {"path": str(bundle)}))
        assert applied == exported
        restored = decoded(await client.call_tool(
            "forward", {"request": {"inputs": [[1.0, 0.0]], "model": "edited"}}
        ))
        assert restored["logits"] == after["edited"]["logits"]
        # Validate real error handling across transport, not just success wrappers.
        invalid = await client.call_tool("read_tensor", {"name": "layer1.weight", "limit": 4097})
        assert invalid.isError
