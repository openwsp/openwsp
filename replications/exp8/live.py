"""Tool-free editor turn loop. Reference edits and evaluation data stay outside it."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from openwsp.editors.base import EditorIncomplete, EditorRefused, ProviderUnavailable
from openwsp.budget import BudgetExceeded
from openwsp.backends.jobs import JobBlocked, JobPending
from openwsp.editors.openai import AstraEditor
from replications.exp8.forward import ForwardTool, freeze_checkpoint
from replications.exp8.runtime import CONFIG, ROOT, atomic_write

BLOCK = re.compile(r"```(python|forward)\s*\n(.*?)```", re.DOTALL)
EXPLANATION = ("Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). "
               + (ROOT / "prompts/message.txt").read_text().split("EXPLAIN EVERY STEP.", 1)[1].split("\n\nRULES.")[0])


def parse_reply(text):
    matches = list(BLOCK.finditer(text))
    if not matches:
        return None, None, text.strip()
    if len(matches) != 1 or text.count("```") != 2:
        raise ValueError("Reply with exactly one Python or forward block")
    match = matches[0]
    if text[match.end():].strip():
        raise ValueError("Put the explanation before the block")
    return match.group(1), match.group(2), text[:match.start()].strip()


def append_record(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush()


def handoff(task, state, sandbox, last_result):
    info = sandbox.inspect_work()
    notes = info["notes"]
    return (task + f"\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {state['turn']} replies happened before. "
            f"Forward passes used: {state['passes']} of 50; Python runs used: {state['runs']} of 300. "
            "Your kernel is still running with its variables. "
            f"Tensors currently edited: {info.get('edited', [])}. Files in /work: {info.get('files', [])}.\n"
            f"YOUR NOTEBOOK (/work/notes.md, last 8,000 characters):\n```\n{notes}\n```\nContinue from here.\n\n"
            f"Last result:\n{last_result}")


def verify_kernel(ledger, checkpoint, app):
    from openwsp.backends.modal_app import sandbox_image, model_volume, practice_volume, work_volume
    from openwsp.backends.sandbox import ModalSandboxTransport
    from openwsp.backends.sandbox_budget import SandboxBudgetLease
    lease = SandboxBudgetLease(ledger, "/data/kernel-lease.json", stage="kernel_verification", commit=checkpoint)
    sandbox = lease.start(lambda: ModalSandboxTransport.create(app, sandbox_image, model_volume, practice_volume, work_volume,
                                          work_subpath="verification", timeout=600))
    try:
        first = sandbox.run((ROOT / "reference/first_cell.py").read_text(), request_id="authors-first-cell")
        if not first["ok"]:
            raise RuntimeError(f"Original first cell failed: {first.get('error')}")
        probes = {
            "frozen_norm": "save_edit('model.language_model.norm.weight', load_original('model.language_model.norm.weight'))",
            "frozen_embedding": "save_edit('model.language_model.embed_tokens.weight', load_original('model.language_model.embed_tokens.weight'))",
            "frozen_output": "save_edit('lm_head.weight', torch.ones(1))",
            "rescale": "n='model.language_model.layers.15.mlp.down_proj.weight'; save_edit(n, 2*load_original(n))",
            "nonfinite": "n='model.language_model.layers.15.mlp.down_proj.weight'; w=load_original(n); w[0,0]=float('nan'); save_edit(n,w)",
            "shape": "save_edit('model.language_model.layers.15.mlp.down_proj.weight', torch.ones(1))",
            "gradient": "torch.ones(1, requires_grad=True).sum().backward()",
            "optimizer": "torch.optim.SGD([torch.ones(1)],lr=0.1)",
            "model_forward": "torch.nn.Linear(2,2)(torch.ones(2))",
            "network": "import urllib.request; urllib.request.urlopen('https://example.com')",
        }
        refusals = {}
        for name, code in probes.items():
            result = sandbox.run(code, request_id="refusal-"+name)
            refusals[name] = {"refused": not result["ok"], "error": result.get("error")}
        if not all(record["refused"] for record in refusals.values()):
            raise RuntimeError("A production kernel refusal failed")
        isolation = sandbox.verify_isolation()
        if (isolation["network"]["connected"] or any(not r["expected_denial"] for r in isolation["mounts"].values())
                or isolation["secret_environment"]["present_names"] or isolation["cuda_available"]):
            raise RuntimeError("Production sandbox isolation probe failed")
        evidence = {"status": "passed", "first_cell": first, "refusals": refusals,
                    "isolation": isolation, "block_network_configured": True}
        atomic_write("/data/results/kernel/sandbox-verification.json", evidence)
        return evidence
    finally:
        lease.close()
        checkpoint()


def run_live(outer_state, ledger, checkpoint, *, app):
    from openwsp.backends.modal_app import sandbox_image, model_volume, practice_volume, work_volume, data_volume
    from openwsp.backends.sandbox import ModalSandboxTransport
    from openwsp.backends.sandbox_budget import SandboxBudgetLease
    root = Path("/data/live-run")
    root.mkdir(parents=True, exist_ok=True)
    task = (ROOT / "prompts/message.txt").read_text()
    path = root / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {
        "budget": 50, "mode": "qwen35think", "prompt": "v3-final", "turn": 0, "runs": 0,
        "passes": 0, "requests": 0, "handoffs": 0, "since_handoff": 0, "done": False,
        "notices": [], "unexplained_streak": 0, "next": task, "pending": None}
    if state["done"]:
        return state
    if not outer_state["stages"].get("kernel", {}).get("status") == "passed":
        outer_state["stages"]["kernel"] = verify_kernel(ledger, checkpoint, app)
    work = Path("/work/live")
    work.mkdir(parents=True, exist_ok=True)
    # A claimed cell may have changed files before a lost driver response. Never
    # replay it in a different kernel: keep files and report the unknown outcome.
    old_pending = state.get("pending")
    uncertain_cell = bool(old_pending and old_pending["kind"] == "python")
    initial_runs = min(300, state["runs"] + int(uncertain_cell))
    if uncertain_cell:
        old_receipt = root / "turns" / f"t{old_pending['turn']:03d}" / "tool_result.json"
        if old_receipt.exists():
            initial_runs = json.loads(old_receipt.read_text())["runs"]
    work_volume.commit()
    lease = SandboxBudgetLease(ledger, root / "sandbox-lease.json", commit=checkpoint)
    sandbox = lease.start(lambda: ModalSandboxTransport.create(
        app, sandbox_image, model_volume, practice_volume, work_volume, work_subpath="live",
        timeout=86400, idle_timeout=180, initial_runs=initial_runs))
    state["sandbox_id"] = sandbox.sandbox.object_id

    def save():
        atomic_write(path, state)
        outer_state["live"] = {key: state[key] for key in ("turn", "runs", "passes", "requests", "done", "sandbox_id")}
        checkpoint()

    try:
        transport_path = root / "transport-reviewed.json"
        editor_options = {}
        if transport_path.exists():
            transport = json.loads(transport_path.read_text())
            if transport.get("status") != "reviewed_for_resume" or transport.get("editor_options") != {"verbosity": "low", "max_output_tokens": 32768}:
                raise ValueError("Unrecognized editor transport recovery record")
            editor_options = transport["editor_options"]
        editor = AstraEditor(root / "editor", ledger, on_poll=lambda _: save(), commit=save, **editor_options)
        restart_notice_due = bool(state["turn"] or state.get("pending"))
        if state["turn"] and not state.get("pending"):
            state["next"] = "The kernel was restarted; variables were lost, but files in /work and saved edits remain.\n" + state["next"]
            restart_notice_due = False
        tool = ForwardTool(root, work, sandbox, ledger, save)
        while not state["done"]:
            lease.check()
            turn = state["turn"] + 1
            directory = root / "turns" / f"t{turn:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            message = state["next"]
            pending = state.get("pending")
            if not pending and state["since_handoff"] >= 40:
                message = handoff(task, state, sandbox, message)
                editor.new_conversation()
                state["handoffs"] += 1
                state["since_handoff"] = 0
            notice = CONFIG["notices"].get(str(turn))
            if not pending and notice and notice not in state["notices"]:
                if notice == "002_tool_facts.txt":
                    sandbox.write_text("eval_prompt_format.txt", (ROOT / "prompts/eval_prompt_format.txt").read_text())
                message = "[note from the experimenters] " + (ROOT / "prompts" / notice).read_text() + "\n\n" + message
                state["notices"].append(notice)
            if pending:
                message = (directory / "prompt.txt").read_text()
                # A budget rejection before request admission has no provider
                # input to preserve. Tell the editor about this fresh kernel
                # before its next action; never rewrite admitted request input.
                editor_state_path = root / "editor/editor_state.json"
                if (restart_notice_due and pending["kind"] == "editor" and editor_state_path.exists()
                        and not (directory / "answer.md").exists()):
                    editor_state = json.loads(editor_state_path.read_text())
                    if editor_state.get("pending") is None and f"t{turn:03d}" not in editor_state.get("completed", {}):
                        notice_text = "The kernel was restarted; variables were lost, but files in /work and saved edits remain.\n"
                        if not message.startswith(notice_text):
                            message = notice_text + message
                            (directory / "prompt.txt").write_text(message)
                            save()
                        restart_notice_due = False
            else:
                (directory / "prompt.txt").write_text(message)
                state["pending"] = {"kind": "editor", "turn": turn}
                save()
            try:
                answer = ((directory / "answer.md").read_text() if (directory / "answer.md").exists()
                          else editor.complete(message, request_key=f"t{turn:03d}"))
            except EditorIncomplete as exc:
                # A terminal, billed output-limit response is an editor turn,
                # not an ambiguous POST or executable partial program. Keep its
                # provider lineage; never sample a replacement for this key.
                record = editor.last_response or {}
                partial = record.get("text", "")
                atomic_write(directory / "run.json", record)
                atomic_write(directory / "editor_event.json", {
                    "status": "incomplete", "reason": exc.reason,
                    "response_id": exc.response_id, "executed": False,
                    "usage": record.get("usage"), "cost_usd": record.get("cost_usd"),
                    "partial_text_characters": len(partial),
                    "partial_text_sha256": hashlib.sha256(partial.encode()).hexdigest(),
                    "response_volume": "openwsp-exp8-data",
                    "response_volume_path": f"/live-run/editor/{record['response_file']}" if record.get("response_file") else None,
                })
                state["next"] = ("The editor API returned an incomplete response (max_output_tokens). "
                                 "No code or forward pass from that response was executed.")
                if restart_notice_due:
                    state["next"] = "The kernel was restarted; variables were lost, but files in /work and saved edits remain.\n" + state["next"]
                    restart_notice_due = False
                state["turn"] = turn
                state["since_handoff"] += 1
                state["pending"] = None
                save()
                continue
            except EditorRefused:
                state["next"] = (ROOT / "prompts/provider_refused.txt").read_text()
                state["turn"] = turn
                state["since_handoff"] += 1
                state["pending"] = None
                save()
                continue
            (directory / "answer.md").write_text(answer)
            if editor.last_response and not (directory / "run.json").exists():
                atomic_write(directory / "run.json", editor.last_response)
            try:
                kind, code, explanation = parse_reply(answer)
            except ValueError as exc:
                kind, code, explanation = "invalid", str(exc), ""
            if kind in ("python", "forward") and not explanation and state["unexplained_streak"] < 2:
                state["unexplained_streak"] += 1
                state["next"] = "EXPLAIN EVERY STEP. " + EXPLANATION
            elif kind == "python":
                state["unexplained_streak"] = 0
                state["pending"] = {"kind": "python", "turn": turn, "request_id": f"live-t{turn:03d}"}
                save()
                receipt = directory / "tool_result.json"
                if receipt.exists():
                    result = json.loads(receipt.read_text())
                elif uncertain_cell:
                    result = {"ok": False, "output": "", "runs": initial_runs, "seconds": 0,
                              "error": "The driver lost the result of this Python run. It may have changed files; it was not repeated.",
                              "restarted": True}
                else:
                    result = sandbox.run(code, request_id=f"live-t{turn:03d}")
                atomic_write(receipt, result)
                checkpoint()
                uncertain_cell = False
                state["runs"] = result["runs"]
                append_record(root / "tool_calls.jsonl", {"turn": turn, "kind": "python", "code": code,
                                                          "out": result["output"], "error": result.get("error"), "s": result["seconds"]})
                state["next"] = f"OUTPUT of Python run {state['runs']} of at most 300:\n```\n{result['output']}\n{result.get('error') or ''}\n```"
                if result.get("restarted"):
                    state["next"] += "\nThe kernel was restarted; variables were lost, saved files remain."
            elif kind == "forward":
                state["unexplained_streak"] = 0
                if state["passes"] >= 50 and not (pending and pending["kind"] == "forward"):
                    state["next"] = "All 50 forward passes are used. Make your final choice and reply without a code block."
                else:
                    if not pending or pending["kind"] != "forward":
                        state["requests"] += 1
                    state["pending"] = {"kind": "forward", "request": state["requests"], "turn": turn}
                    save()
                    try:
                        request = json.loads(code)
                        result = tool.execute(request, state["requests"], turn)
                    except (BudgetExceeded, JobBlocked, JobPending):
                        raise
                    except (ValueError, RuntimeError, KeyError) as exc:
                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        atomic_write(root / "requests" / f"fwd_{state['requests']}.done.json", result)
                    if result.get("ok"):
                        applied = state.setdefault("counted_requests", [])
                        if state["requests"] not in applied:
                            state["passes"] += 1
                            applied.append(state["requests"])
                        state["next"] = f"FORWARD PASS {state['passes']} of 50:\n```json\n{json.dumps(result)}\n```"
                    else:
                        state["next"] = "The forward pass failed and was not counted: " + result["error"]
                    append_record(root / "tool_calls.jsonl", {"turn": turn, "kind": "forward", "request": code, "result": result})
            elif kind is None and state["passes"] >= 50:
                (root / "final_answer.md").write_text(answer)
                sandbox.commit_work()
                work_volume.reload()
                frozen, edit_id, names = freeze_checkpoint(work, "/data/submissions")
                state["submission"] = {"checkpoint": frozen, "sha256": edit_id, "tensors": names}
                from openwsp.core import WeightStore
                WeightStore("/model", work / "edited").export_edit(root / "edit", base_model=CONFIG["model"], revision=CONFIG["revision"])
                data_volume.commit()
                state["done"] = True
            elif kind == "invalid":
                state["next"] = code
            else:
                state["next"] = f"Continue: {state['passes']} of 50 forward passes are used. Replying without code does not end the run before all passes are used."
            if restart_notice_due:
                state["next"] = "The kernel was restarted; variables were lost, but files in /work and saved edits remain.\n" + state["next"]
                restart_notice_due = False
            state["turn"] = turn
            state["since_handoff"] += 1
            state["pending"] = None
            save()
        return state
    except ProviderUnavailable:
        outer_state["status"] = "blocked"
        save()
        raise
    finally:
        try:
            lease.close()
        finally:
            save()
