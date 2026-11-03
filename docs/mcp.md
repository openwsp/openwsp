# Connect an agent through MCP

The stdio server exposes the same operations as the Python SDK and CLI. It runs locally; no hosted service or account is required for the tiny CPU example. Install from a checkout with `uv sync --locked --extra dev` before adding it to a client.

Replace `/absolute/path/openwsp` and `/absolute/path/work` in the examples with your checkout and a writable work directory. `uv run --no-sync` uses that checkout's installed, locked environment and does not install packages when the client starts.

## Codex

```sh
codex mcp add openwsp -- uv run --project /absolute/path/openwsp --no-sync \
  openwsp-mcp --tiny --work-dir /absolute/path/work
```

Run `codex mcp list` to check the configuration. The command uses the stdio registration syntax in [Codex's MCP documentation](https://developers.openai.com/codex/mcp/).

## Claude Code

```sh
claude mcp add --transport stdio openwsp -- uv run \
  --project /absolute/path/openwsp --no-sync \
  openwsp-mcp --tiny --work-dir /absolute/path/work
```

Run `claude mcp get openwsp` or `/mcp` in Claude Code to check the connection. See [Claude Code's MCP documentation](https://code.claude.com/docs/en/mcp) for configuration scopes.

## Cursor

Add this entry to the project's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "openwsp": {
      "command": "uv",
      "args": [
        "run", "--project", "/absolute/path/openwsp", "--no-sync",
        "openwsp-mcp", "--tiny", "--work-dir", "/absolute/path/work"
      ]
    }
  }
}
```

Check the server in Cursor's MCP settings. This uses the project configuration location and stdio format described in [Cursor's documentation](https://cursor.com/docs/mcp).

## Tools and a first run

Ask the client to list the OpenWSP tensors, read `layer1.weight`, and call `forward` with `{"inputs": [[1.0, 0.0]], "model": "both"}`. Then ask it to change one value, save the edit, inspect the diff, export it, and apply the exported edit to a fresh session.

| Tool | Operation |
| --- | --- |
| `list_tensors` | List tensor names and metadata. |
| `read_tensor` | Read a bounded flat slice, with shape and dtype. |
| `run_code` | Run Python in the persistent kernel. |
| `forward` | Send a request to the configured forward backend. |
| `save_edit` | Validate and save replacement values for one tensor. |
| `diff` | Compare submitted edits with the immutable base. |
| `apply_edit` | Verify and apply an edit directory. |
| `export_edit` | Write the versioned manifest and compact tensor diff. |

The tiny example is a functional CPU network used by the tests. It produces no Experiment 8 measurements. CI starts the server as a subprocess and drives every tool through an MCP client.

For a checkpoint, replace `--tiny` with `--model /absolute/path/model.safetensors`. Configure inference with `--backend installed_module:factory`; the trusted factory takes no arguments and returns a backend with `forward(request, store) -> dict`. A backend must support that model before forward requests can run; loading its tensors alone does not configure inference. See [architecture](architecture.md) for the backend boundary and [edit format](edit-format.md) for portable artifacts.

`run_code` in a local MCP server has the local process's access. Run the server in an isolated environment when executing untrusted code. The Experiment 8 editor instead receives the authors' text-block interface inside its dedicated network-blocked sandbox; connecting a general-purpose MCP client is a separate use of the tooling.
