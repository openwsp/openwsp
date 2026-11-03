"""Small, separately recorded transport checks with no experiment inputs."""
from __future__ import annotations

import ast
import json
import re
from decimal import Decimal
from pathlib import Path

from openwsp.budget import BudgetExceeded
from openwsp.editors import AstraEditor, EditorError
from replications.exp8.runtime import ROOT, atomic_write, digest

AUTHORIZATION = ROOT / "authorizations/editor-budget-20261103.json"
PROBE_LIMIT = Decimal("2.00")
PROMPTS = (
    ("This is a synthetic formatting check, with no code execution. Write one sentence explaining "
    "that 1+1=2, followed by exactly one fenced python block containing print(1 + 1). "
    "End after the closing fence."),
    ('This is a second synthetic formatting check, with no tool dispatch. Write one sentence, '
    'followed by exactly one fenced forward block containing JSON {"type":"probe","value":2}. '
    'End after the closing fence.'),
)


def apply_authorization(ledger, authorization=AUTHORIZATION):
    grant = json.loads(Path(authorization).read_text())
    baseline = Decimal(grant["settled_spend_baseline_usd"])
    extra = Decimal(grant["additional_authorized_usd"])
    total = Decimal(grant["total_editor_limit_usd"])
    if total != baseline + extra or extra != Decimal(grant["protected_plan_b_usd"]):
        raise ValueError("Authorization does not preserve the user's half-allocation")
    snapshot = ledger.snapshot()
    if grant["authorization_id"] not in snapshot.get("editor_authorizations", {}):
        spent = sum((Decimal(row["actual_usd"]) for row in snapshot["reservations"].values()
                     if row["kind"] == "editor" and row["status"] == "settled"), Decimal(0))
        if spent != baseline:
            raise ValueError("This authorization belongs to the recorded existing task spend, not a fresh run")
    return ledger.authorize_editor_limit(total, authorization_id=grant["authorization_id"],
                                         evidence=grant["evidence"])


class ProbeBudget:
    """A tighter admission cap within the existing serialized driver ledger."""

    def __init__(self, ledger):
        self.ledger = ledger

    def snapshot(self):
        return self.ledger.snapshot()

    def totals(self):
        spent, reserved = Decimal(0), Decimal(0)
        for row in self.snapshot()["reservations"].values():
            if row["kind"] == "editor" and row["stage"].startswith("editor:transport-probe:"):
                if row["status"] == "settled":
                    spent += Decimal(row["actual_usd"])
                elif row["status"] == "reserved":
                    reserved += Decimal(row["upper_bound_usd"])
        return {"spent_usd": str(spent), "reserved_usd": str(reserved), "limit_usd": str(PROBE_LIMIT)}

    def reserve(self, kind, amount, stage, **kwargs):
        if kind != "editor" or not stage.startswith("editor:transport-probe:"):
            raise ValueError("Probe ledger only admits its own editor requests")
        existing = self.snapshot()["reservations"].get(kwargs.get("reservation_id"))
        if existing is None:
            totals = self.totals()
            if Decimal(totals["spent_usd"]) + Decimal(totals["reserved_usd"]) + Decimal(amount) > PROBE_LIMIT:
                raise BudgetExceeded("Editor transport validation exceeds its separate $2 cap")
        return self.ledger.reserve(kind, amount, stage, **kwargs)

    def settle(self, *args, **kwargs):
        return self.ledger.settle(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self.ledger.release(*args, **kwargs)


def run_probe(ledger, checkpoint, directory="/data/editor-transport-probe"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write(directory / "authorization.json", apply_authorization(ledger))
    checkpoint()  # authorization must be durable before any paid request
    budget = ProbeBudget(ledger)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "status": "running", "model": "gpt-6-astra", "effort": "max", "max_output_tokens": 4096,
        "probe_cap_usd": str(PROBE_LIMIT), "mode_order": ["low", "default"], "cases": {},
        "scope": "Synthetic formatting only; no weights, practice problems or benchmark results supplied. No code or tool dispatch.",
        "limitation": "A short formatting check does not establish long-run reliability or prove the cause of prior loops.",
    }

    def save():
        summary["budget"] = budget.totals()
        atomic_write(summary_path, summary)
        checkpoint()

    save()
    try:
        for label, verbosity in (("low", "low"), ("default", None)):
            editor = AstraEditor(directory / label, budget, max_output_tokens=4096,
                                 verbosity=verbosity, commit=checkpoint)
            for index, prompt in enumerate(PROMPTS, 1):
                key = f"transport-probe:{label}:{index}"
                if summary["cases"].get(key, {}).get("status") == "completed":
                    continue
                try:
                    text = editor.complete(prompt, request_key=key)
                except Exception as exc:
                    summary["cases"][key] = {"status": "blocked", "error_type": type(exc).__name__,
                                             "response_id": (editor.last_response or {}).get("response_id")}
                    save()
                    raise
                receipt = editor.last_response
                raw_path = directory / label / receipt["response_file"]
                raw = json.loads(raw_path.read_text())
                match = re.fullmatch(r"([\s\S]*?)```(python|forward)\n([\s\S]*?)```\s*", text)
                valid = bool(match and match.group(1).strip() and match.group(2) == ("python" if index == 1 else "forward")
                             and "```" not in match.group(3))
                if valid:
                    try:
                        ast.parse(match.group(3)) if index == 1 else json.loads(match.group(3))
                    except (SyntaxError, ValueError):
                        valid = False
                summary["cases"][key] = {
                    "status": "completed" if valid else "invalid_format", "response_id": receipt["response_id"],
                    "requested_verbosity": verbosity, "returned_text_configuration": raw.get("text"),
                    "characters": len(text), "usage": receipt["usage"], "cost_usd": receipt["cost_usd"],
                    "response_volume": "openwsp-exp8-data", "response_volume_path": str(raw_path).removeprefix("/data"),
                    "ended_after_single_fenced_block": valid,
                }
                save()
                if not valid:
                    raise EditorError("Synthetic editor response did not obey the single-block format")
        summary["status"] = "passed_transport_smoke"
        return summary
    except Exception as exc:
        summary.update(status="blocked", error_type=type(exc).__name__)
        raise
    finally:
        save()


def review_live_transport(ledger, checkpoint, directory="/data/editor-transport-probe", live_dir="/data/live-run"):
    """Explicitly review this recorded legacy incident without rewriting history."""
    directory, live_dir = Path(directory), Path(live_dir)
    probe = json.loads((directory / "summary.json").read_text())
    expected = {f"transport-probe:{mode}:{turn}" for mode in ("low", "default") for turn in (1, 2)}
    if (probe.get("status") != "passed_transport_smoke" or set(probe.get("cases", {})) != expected
            or any(case.get("status") != "completed" or not case.get("ended_after_single_fenced_block")
                   for case in probe["cases"].values())):
        raise ValueError("Live recovery requires all four completed synthetic format checks")
    grant = apply_authorization(ledger)
    state = json.loads((live_dir / "state.json").read_text())
    if state.get("done") or (state.get("turn"), state.get("passes"), state.get("runs")) != (26, 4, 10):
        raise ValueError("This recovery applies only to the recorded paused trajectory")
    editor_state = json.loads((live_dir / "editor/editor_state.json").read_text())
    incident = json.loads((ROOT / "results/live-run/editor-format-loop.json").read_text())
    response_id = next(row["response_id"] for row in incident["turns"] if row["turn"] == "t026")
    if editor_state.get("previous_response_id") != response_id or editor_state.get("pending"):
        raise ValueError("The original settled response lineage must remain unchanged")
    reason = "Reviewed the recorded formatting loop after transport checks; preserve the original history and forty-turn handoff."
    evidence = {"source": str(directory / "summary.json"), "sha256": digest(directory / "summary.json"),
                "authorization_id": grant["authorization_id"], "guard": "streamed_formatting_runaway",
                "future_max_output_tokens": 32768, "future_verbosity": "low"}
    editor = AstraEditor(live_dir / "editor", ledger, verbosity="low", commit=checkpoint)
    review = editor.acknowledge_runaway(response_id, reason=reason, evidence=evidence)
    record = {"status": "reviewed_for_resume", "review": review, "editor_options": {"verbosity": "low", "max_output_tokens": 32768},
              "preserved": {"turn": state["turn"], "passes": state["passes"], "runs": state["runs"],
                            "previous_response_id": response_id, "context_tokens": editor_state["context_tokens"]},
              "limitation": "The synthetic check is small. A new formatting loop still cancels and blocks; no discarded action was executed and no history was trimmed."}
    atomic_write(live_dir / "transport-reviewed.json", record)
    checkpoint()
    return record
