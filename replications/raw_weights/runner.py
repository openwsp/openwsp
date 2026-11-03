"""Explicit local one-reply execution; importing or inventorying never calls a model."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from openwsp.budget import BudgetLedger, atomic_json
from openwsp.editors.anthropic import AnthropicTokenRates
from openwsp.editors.profiles import ASTRA_CODEX_PROFILE, FABLE_5_1_PROFILE, ThinkingConfig, create_editor

from .edits import apply_numeric_edits, shown_car_store, zero_minatar_store
from .scoring import score_label

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = Path(__file__).with_name("inventory.json")


def inventory():
    return json.loads(INVENTORY.read_text())


def verified_source(source_root, item):
    root = Path(source_root).resolve()
    contents = {}
    for name in ("prompt", "instructions"):
        path = (root / item[name]["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source path escapes pinned checkout")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item[name]["sha256"]:
            raise ValueError(f"Pinned {name} hash mismatch")
        contents[name] = data.decode("utf-8")
    return contents


def score_answer(item, answer, prompt, output_dir, *, rollout=False):
    experiment = item["experiment"]
    if experiment in ("exp1", "exp2"):
        return score_label(experiment, item["item_id"], answer)
    store = zero_minatar_store() if experiment == "exp5" else shown_car_store(prompt)
    parser = apply_numeric_edits(answer, store)
    edit_dir = Path(output_dir) / "edit"
    store.export_edit(edit_dir, base_model=f"{experiment}:{item['item_id']}:shown", revision=inventory()["revision"])
    result = {"parser": parser, "edit_path": str(edit_dir), "parameters": sum(store.load(n).numel() for n in store.names),
              "nonzero_parameters": sum(int(store.load(n).count_nonzero()) for n in store.names),
              "changed_parameters": sum(int((store.load(n) != store.load_original(n)).count_nonzero()) for n in store.names)}
    if experiment == "exp5" and rollout:
        from .minatar_score import evaluate
        result["evaluation"] = evaluate(item["item_id"], store)
    elif experiment == "exp4" and rollout:
        result["evaluation"] = evaluate_car(store, output_dir)
    else:
        result["evaluation"] = {"status": "pending", "reason": "CPU rollout has not been run"}
    return result


def evaluate_car(store, output_dir):
    """Keep the pinned JAX environment separate from the shared torch installation."""
    executable = ROOT / ".openwsp/kinetix-venv/bin/python"
    source_root = ROOT / ".openwsp/kinetix-source"
    if not executable.is_file() or not source_root.is_dir():
        raise RuntimeError("Prepare the isolated Kinetix environment in replications/exp4/SCORER.md")
    output = Path(output_dir).resolve() / "kinetix-evaluation.json"
    env = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH") if name in os.environ}
    env.update(JAX_PLATFORMS="cpu", MPLCONFIGDIR="/tmp/openwsp-matplotlib", XDG_CACHE_HOME="/tmp/openwsp-cache",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="openwsp-car-score-") as directory:
        network = Path(directory) / "actor.npz"
        np.savez(network, **{name: store.load(name).numpy() for name in store.names})
        subprocess.run([str(executable), "-m", "replications.raw_weights.kinetix_score", "--network", str(network),
                        "--source-root", str(source_root), "--output", str(output), "--episodes", "1024",
                        "--batch-size", "32"], cwd=ROOT, env=env, check=True)
    return json.loads(output.read_text())


def _private(path):
    path = Path(path).absolute()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("State directory must be owned by the current user without a symlink")
    path.chmod(0o700)
    return path.resolve()


def run_item(item, *, source_root, state_root, output_root, editor_name, editor_factory,
             rollout=False, score_only=False):
    """Durable editor replay owns retry safety; no second reply is ever requested."""
    source = verified_source(source_root, item)
    target = Path(output_root) / editor_name / item["experiment"] / item["item_id"]
    target.mkdir(parents=True, exist_ok=True)
    state = _private(Path(state_root) / editor_name / item["experiment"] / item["item_id"])
    binding = {"source_revision": inventory()["revision"], "prompt_sha256": item["prompt"]["sha256"],
               "instructions_sha256": item["instructions"]["sha256"], "editor": editor_name,
               "experiment": item["experiment"], "item_id": item["item_id"]}
    request_key = "raw-weights:" + hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    with (state / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = target / "result.json"
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if old.get("binding") != binding:
                raise ValueError("Existing item belongs to a different source/editor")
            if old.get("status") in ("answered", "completed"):
                answer = (target / "answer.md").read_bytes()
                if hashlib.sha256(answer).hexdigest() != old["answer_sha256"]:
                    raise ValueError("Completed answer hash changed")
                pending_rollout = old.get("score", {}).get("evaluation", {}).get("status") == "pending"
                if old["status"] == "answered" or (rollout and pending_rollout) or (score_only and "score" not in old):
                    old["score"] = score_answer(item, answer.decode("utf-8"), source["prompt"], target, rollout=rollout)
                    old["status"] = "completed"
                    atomic_json(record_path, old)
                return old
        if score_only:
            raise ValueError("Scoring requires a saved, hash-verified answer; no editor was invoked")
        record = {"binding": binding, "request_key": request_key, "status": "pending",
                  "tools": [], "scientific_replies_requested": 1,
                  "editor_contract": {"effort": item.get("authors_config", {}).get("effort"),
                      "reasoning_summary": item.get("authors_config", {}).get("reasoning_summary"),
                      "verbosity": "low" if editor_name == "astra_subscription" else None,
                      "transport": "codex_cli_chatgpt" if editor_name == "astra_subscription" else "anthropic_messages",
                      "instructions_sha256": item["instructions"]["sha256"]}}
        atomic_json(record_path, record)
        editor = editor_factory(item, source["instructions"], state)
        try:
            answer = editor.complete(source["prompt"], request_key=request_key)
        except Exception as exc:
            record.update(status="blocked", error_type=type(exc).__name__, receipt=editor.last_response)
            # Terminal incomplete/refused receipts remain evidence. A saved
            # partial answer is never treated as complete or scored by this path.
            partial = (editor.last_response or {}).get("text")
            if isinstance(partial, str) and partial:
                data = partial.encode("utf-8")
                (target / "incomplete-answer.md").write_bytes(data)
                record.update(incomplete_answer_sha256=hashlib.sha256(data).hexdigest(), incomplete_answer_bytes=len(data))
            atomic_json(record_path, record)
            raise
        data = answer.encode("utf-8")
        (target / "answer.md").write_bytes(data)
        record.update(answer_sha256=hashlib.sha256(data).hexdigest(), answer_bytes=len(data),
                      receipt=editor.last_response, status="answered")
        atomic_json(record_path, record)
        record["score"] = score_answer(item, answer, source["prompt"], target, rollout=rollout)
        record["status"] = "completed"
        atomic_json(record_path, record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", choices=("astra_subscription", "fable_native_anthropic"), required=True)
    parser.add_argument("--experiment", choices=("exp1", "exp2", "exp4", "exp5"))
    parser.add_argument("--item")
    parser.add_argument("--source-root", type=Path, default=ROOT / ".openwsp/upstream")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-home", type=Path)
    parser.add_argument("--fable-limits", type=Path, help="Verified per-item input ceiling/output cap JSON")
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--score-only", action="store_true", help="Score saved answers; never construct an editor")
    parser.add_argument("--execute", action="store_true", help="Actually request the one scientific reply")
    args = parser.parse_args(argv)
    items = [item for item in inventory()["items"] if
             (args.experiment is None or item["experiment"] == args.experiment)
             and (args.item is None or item["item_id"] == args.item)]
    if not items:
        parser.error("No released item matches")
    for item in items:
        verified_source(args.source_root, item)
    if args.execute and args.score_only:
        parser.error("--execute and --score-only are mutually exclusive")
    if not args.execute and not args.score_only:
        print(json.dumps({"status": "inventory_only", "items": items}, indent=2))
        return
    if args.score_only:
        factory = None
    elif args.editor == "astra_subscription":
        if args.codex_home is None:
            parser.error("--codex-home must identify an existing subscription login")
        def factory(item, instructions, state):
            config = item["authors_config"]
            profile = replace(ASTRA_CODEX_PROFILE, name=f"astra-{item['experiment']}-authors",
                              effort=config["effort"], reasoning_summary=config["reasoning_summary"],
                              instructions=instructions)
            isolated = Path("/tmp") / ("openwsp-raw-" + hashlib.sha256(str(state).encode()).hexdigest()[:24])
            return create_editor(profile, state_dir=state / "editor", codex_home=args.codex_home,
                                 work_dir=_private(isolated))
    else:
        if args.fable_limits is None:
            parser.error("Fable requires a verified --fable-limits file before any request")
        limits = json.loads(args.fable_limits.read_text())
        budget = BudgetLedger(_private(args.state_root) / "fable-budget.json", editor_limit=20)
        if float(budget.snapshot()["limits"]["editor"]) != 20:
            raise ValueError("This comparison must use its dedicated existing $20 editor budget")
        rates = AnthropicTokenRates(input_per_million=10, output_per_million=50, cache_read_per_million="0.25",
                                    cache_creation_5m_per_million="12.5", cache_creation_1h_per_million=20)
        def factory(item, instructions, state):
            key = item["experiment"] + "/" + item["item_id"]
            bound = limits[key]
            if bound["prompt_sha256"] != item["prompt"]["sha256"] or not bound.get("count_evidence"):
                raise ValueError("Fable token ceiling needs matching prompt and count evidence")
            if bound["max_input_tokens"] + bound["max_output_tokens"] > 1_000_000:
                raise ValueError("Fable request exceeds verified context window")
            expected_output = 8192 if item["experiment"] in ("exp1", "exp2") else 32768
            if bound["max_output_tokens"] != expected_output:
                raise ValueError("Fable output ceiling differs from the preregistered comparison")
            profile = replace(FABLE_5_1_PROFILE, system=instructions, thinking=ThinkingConfig(type="adaptive"),
                              output_effort=item["authors_config"]["effort"])
            return create_editor(profile, state_dir=state / "editor", budget=budget, rates=rates,
                                 max_input_tokens=bound["max_input_tokens"], max_output_tokens=bound["max_output_tokens"],
                                 timeout=3600)
    for item in items:
        result = run_item(item, source_root=args.source_root, state_root=args.state_root,
                          output_root=args.output_root, editor_name=args.editor, editor_factory=factory,
                          rollout=args.rollout, score_only=args.score_only)
        print(json.dumps({"experiment": item["experiment"], "item": item["item_id"], "status": result["status"]}))


if __name__ == "__main__":
    main()
