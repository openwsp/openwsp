#!/usr/bin/env python3
"""Capture Codex 0.163.0's request with a loopback provider and a fake key.

This is a transport fixture, never a subscription or model-quality test. It
creates a private empty Codex state directory, ignores external configuration,
and routes every child HTTP proxy to its local server. No existing login is read.
Run: python scripts/check_codex_transport.py --output-dir /tmp/codex-wire-check
Add --resume to capture a second turn using the first turn's exact thread ID.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "src/openwsp/editors/data/codex-models-0.163.0.json"
DEFAULT_PROVENANCE = DEFAULT_CATALOG.with_suffix(".provenance.json")
PINNED_CATALOG_SHA256 = "05723cac0f7da85c31f710f602bb0c47a5588d8bdf68aad24e9dd6ac951c7705"
PINNED_SOURCE_MODEL_SHA256 = "b62992cf13d7d24c9fb2c9ae030f887ebd319cf072b120b8fccd21e051d5845b"
PINNED_CLI_VERSION = "0.163.0"
INSTRUCTIONS = "Answer the user's request."
PROMPT = "Return exactly MOCK_OK. This is a local synthetic transport fixture."
FAKE_KEY = "openwsp-local-mock-not-a-real-key"
FEATURES_OFF = (
    "code_mode", "code_mode_only", "current_time_reminder", "collaboration_modes",
    "shell_tool", "unified_exec", "unified_exec_tty", "shell_snapshot", "view_image",
    "image_generation", "multi_agent", "apps", "plugins", "remote_plugin", "browser_use",
    "browser_use_external", "browser_use_full_cdp_access", "in_app_browser", "computer_use",
    "code_mode_host", "skill_search", "skill_mcp_dependency_install", "sleep_tool", "tool_suggest",
    "goals", "hooks", "worktrees", "workspace_dependencies", "recommended_plugins",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def load_profile(
    catalog_path: Path = DEFAULT_CATALOG, provenance_path: Path = DEFAULT_PROVENANCE
) -> dict:
    """Check bytes and that reversing exactly three overrides restores the source."""
    data = Path(catalog_path).read_bytes()
    provenance = json.loads(Path(provenance_path).read_text())
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINNED_CATALOG_SHA256 or digest != provenance["catalog_sha256"]:
        raise ValueError("Pinned catalog checksum mismatch")
    if provenance["cli_version"] != PINNED_CLI_VERSION:
        raise ValueError("Unsupported catalog CLI version")
    if provenance["source_model_canonical_sha256"] != PINNED_SOURCE_MODEL_SHA256:
        raise ValueError("Source model checksum mismatch")
    catalog = json.loads(data)
    if len(catalog["models"]) != 1 or catalog["models"][0]["slug"] != "gpt-6-astra":
        raise ValueError("Catalog must contain only the pinned Astra model")
    model = copy.deepcopy(catalog["models"][0])
    changes = provenance["changes"]
    if set(changes) != {"tool_mode", "apply_patch_tool_type", "experimental_supported_tools"}:
        raise ValueError("Unexpected catalog changes")
    for key, change in changes.items():
        if model[key] != change["after"]:
            raise ValueError("Catalog does not match its change audit")
        model[key] = change["before"]
    if hashlib.sha256(canonical_json(model)).hexdigest() != PINNED_SOURCE_MODEL_SHA256:
        raise ValueError("Source model checksum mismatch")
    return catalog


def build_config(instructions_file: Path, catalog_path: Path = DEFAULT_CATALOG) -> dict:
    """Provider-independent CLI configuration proved by this loopback capture."""
    return {
        "model_reasoning_effort": "max", "model_verbosity": "low",
        "model_context_window": 872000, "model_auto_compact_token_limit": 1000000,
        "model_instructions_file": str(Path(instructions_file).resolve()),
        "model_catalog_json": str(Path(catalog_path).resolve()),
        "approval_policy": "never", "project_doc_max_bytes": 0,
        "instructions": "", "developer_instructions": "",
        "skills.include_instructions": False, "skills.bundled.enabled": False,
        "include_permissions_instructions": False, "include_apps_instructions": False,
        "include_collaboration_mode_instructions": False, "include_environment_context": False,
        "include_environment_context_time": False, "web_search": "disabled", "agents.enabled": False,
        "tools.experimental_request_user_input.enabled": False, "tools.update_plan.enabled": False,
        **{"features." + flag: False for flag in FEATURES_OFF},
    }


def assert_request(body: dict, expected_input: list[dict] | None = None) -> dict:
    """Reject additional instructions or advertised tools, including nested tools."""
    if body.get("tools") or body.get("instructions"):
        raise AssertionError("Unexpected top-level tools or instructions")
    items = body["input"]
    if any(item.get("type") == "additional_tools" for item in items):
        raise AssertionError("Unexpected additional_tools input")
    instruction_messages = [x for x in items if x.get("role") in ("developer", "system")]
    expected_instruction = {"role": "developer", "type": "message",
                            "content": [{"type": "input_text", "text": INSTRUCTIONS}]}
    simplified = [{key: item[key] for key in ("role", "type", "content")} for item in instruction_messages]
    if simplified != [expected_instruction]:
        raise AssertionError("Unexpected instruction text or extra instruction messages")
    if expected_input is not None:
        actual = [{key: value for key, value in item.items() if key != "id"} for item in items]
        if actual != expected_input:
            raise AssertionError("Unexpected first-turn input")
    if body["model"] != "gpt-6-astra" or body["reasoning"] != {"effort": "max", "context": "all_turns"}:
        raise AssertionError("Unexpected model or reasoning settings")
    if body["text"] != {"verbosity": "low"}:
        raise AssertionError("Unexpected verbosity")
    return {"model": body["model"], "tools_present": "tools" in body,
            "input_roles": [item.get("role", item.get("type")) for item in items],
            "reasoning": body["reasoning"], "text": body["text"],
            "store": body.get("store"), "include": body.get("include"),
            "max_output_tokens_present": "max_output_tokens" in body}


def toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}={toml(item)}" for key, item in value.items()) + "}"
    return json.dumps(value)


def run_fixture(output_dir: Path, *, codex: str = "codex", resume: bool = False) -> dict:
    load_profile()
    output_dir = output_dir.resolve()
    # Reuse could bring an existing login/config into scope; refuse it entirely.
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    codex_fixture_root, work = output_dir / "codex-state", output_dir / "work"
    codex_fixture_root.mkdir(mode=0o700)
    work.mkdir(mode=0o700)
    instruction_file = output_dir / "instructions.md"
    instruction_file.write_text(INSTRUCTIONS + "\n")
    executable = shutil.which(codex)
    if executable is None:
        raise ValueError("Codex CLI executable not found")
    captures: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_CONNECT(self) -> None:
            self.send_error(403, "External destinations are disabled")

        def do_GET(self) -> None:
            self.send_error(404, "Only loopback Responses POST is supported")

        def do_POST(self) -> None:
            if self.path != "/v1/responses":
                self.send_error(403, "Only loopback Responses is permitted")
                return
            encoded = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.headers.get("Content-Encoding") == "gzip":
                encoded = gzip.decompress(encoded)
            body = json.loads(encoded)
            captures.append({"path": self.path, "fake_auth_matches": self.headers.get("Authorization") ==
                             "Bearer " + FAKE_KEY, "body": body})
            # Usage is intentionally identical on each response, to expose whether
            # resumed JSONL counters are cumulative or per-turn.
            item = {"id": "msg_mock", "type": "message", "role": "assistant", "status": "completed",
                    "phase": "final_answer", "content": [{"type": "output_text", "text": "MOCK_OK",
                                                             "annotations": []}]}
            response = {"id": f"resp_mock_{len(captures)}", "object": "response", "created_at": 0,
                        "status": "completed", "output": [item],
                        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
            events = [
                {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
                {"type": "response.output_item.added", "output_index": 0,
                 "item": {**item, "status": "in_progress", "content": []}},
                {"type": "response.content_part.added", "item_id": "msg_mock", "output_index": 0,
                 "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
                {"type": "response.output_text.delta", "item_id": "msg_mock", "output_index": 0,
                 "content_index": 0, "delta": "MOCK_OK"},
                {"type": "response.output_text.done", "item_id": "msg_mock", "output_index": 0,
                 "content_index": 0, "text": "MOCK_OK"},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": response},
            ]
            data = "".join("event: " + event["type"] + "\ndata: " +
                           json.dumps({**event, "sequence_number": index}) + "\n\n"
                           for index, event in enumerate(events)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # Nothing is inherited except the executable's location. Actual credentials,
    # provider settings, managed-host tools, and user configuration are absent.
    environment = {
        "PATH": str(Path(executable).parent) + ":/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
        "CODEX_HOME": str(codex_fixture_root), "OPENWSP_MOCK_API_KEY": FAKE_KEY,
        "HTTP_PROXY": endpoint, "HTTPS_PROXY": endpoint, "ALL_PROXY": endpoint,
        "NO_PROXY": "127.0.0.1,localhost",
    }
    config = build_config(instruction_file)
    config.update({
        "model_provider": "openwsp_mock",
        "model_providers.openwsp_mock": {
            "name": "OpenWSP loopback mock", "base_url": endpoint + "/v1",
            "env_key": "OPENWSP_MOCK_API_KEY", "wire_api": "responses", "requires_openai_auth": False,
            "request_max_retries": 0, "stream_max_retries": 0, "stream_idle_timeout_ms": 10000,
        },
    })
    flags = ["--ignore-user-config", "--ignore-rules", "--strict-config",
             "--skip-git-repo-check", "--json", "-m", "gpt-6-astra"]
    for key, value in config.items():
        flags += ["-c", key + "=" + toml(value)]
    outputs = []
    try:
        version = subprocess.run([executable, "--version"], check=False, env=environment, cwd=work,
                                 stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
        if version.returncode or version.stdout.strip() != f"codex-cli {PINNED_CLI_VERSION}":
            raise ValueError("This fixture requires Codex CLI " + PINNED_CLI_VERSION)
        thread_id = None
        for index in range(2 if resume else 1):
            command = [executable, "exec", "--sandbox", "read-only"]
            if index:
                command += ["resume", *flags, thread_id, PROMPT]
            else:
                command += flags + ([] if resume else ["--ephemeral"]) + ["-C", str(work), PROMPT]
            (output_dir / f"command-{index}.json").write_text(json.dumps(command, indent=2) + "\n")
            completed = subprocess.run(command, check=False, env=environment, cwd=work, stdin=subprocess.DEVNULL,
                                       capture_output=True, text=True, timeout=75)
            (output_dir / f"stdout-{index}.jsonl").write_text(completed.stdout)
            (output_dir / f"stderr-{index}.txt").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"Mock CLI exited {completed.returncode}; see fixture stderr")
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            started = next(event["thread_id"] for event in events if event["type"] == "thread.started")
            if thread_id is not None and started != thread_id:
                raise AssertionError("Resume returned a different thread ID")
            thread_id = started
            if not any(event["type"] == "item.completed" and event["item"].get("text") == "MOCK_OK"
                       for event in events):
                raise AssertionError("Mock final answer missing")
            outputs.append(next(event["usage"] for event in events if event["type"] == "turn.completed"))
        if len(captures) != len(outputs) or not all(item["fake_auth_matches"] for item in captures):
            raise AssertionError("Unexpected request count or fake-key authentication")
        expected = [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": INSTRUCTIONS}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": PROMPT}]},
        ]
        summaries = [assert_request(item["body"], expected if index == 0 else None)
                     for index, item in enumerate(captures)]
        summary = {"cli_version": PINNED_CLI_VERSION, "catalog_sha256": PINNED_CATALOG_SHA256,
                   "provider": "loopback mock only", "real_provider_calls": 0,
                   "requests": summaries, "cli_turn_usage": outputs,
                   "mock_usage_per_response": {"input_tokens": 10, "output_tokens": 2},
                   "resume_requested": resume, "thread_id": thread_id,
                   "limitations": ["Does not test subscription authentication or model availability",
                                   "No inference output cap or prompt-cache TTL appears in captured requests"]}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        server.shutdown()
        server.server_close()
        (output_dir / "captures.json").write_text(json.dumps(captures, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New private output directory")
    parser.add_argument("--codex", default="codex", help="Installed Codex 0.163.0 executable")
    parser.add_argument("--resume", action="store_true", help="Also capture an exact-ID resumed turn")
    args = parser.parse_args()
    print(json.dumps(run_fixture(args.output_dir, codex=args.codex, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
