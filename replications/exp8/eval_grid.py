"""Preregistered decoding cells. This module is never an editor input.

``python -m replications.exp8.eval_grid validate --output summary.json`` runs
only tiny deterministic token-policy checks. GPU cells use replication stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from replications.exp8.runtime import CONFIG, ROOT, atomic_write

CONFIG_DIR = ROOT / "eval_grid_configs"
STAGES = (
    "grid_p3_base", "grid_p3_p50c", "grid_p3_live",
    "grid_p4_base", "grid_p4_p50c", "grid_p4_live",
    "grid_p1_base", "grid_p1_p50c", "grid_p5_presence_base",
    "grid_p5_repetition_base", "grid_p5_repeat_close_base",
)
SEED_SCHEME = "sha256_first8_utf8_problem_id_mod_2**31"


def problem_seed(problem_id):
    """Preserve the existing pinned evaluator's first-eight-hex seed conversion."""
    return int(hashlib.sha256(str(problem_id).encode("utf-8")).hexdigest()[:8], 16) % (2**31)


def validate_policy(policy):
    if policy is None:
        return
    if not isinstance(policy, dict):
        raise TypeError("Decoding policy must be a mapping")
    if policy.get("kind") == "thinking_token_cap":
        if set(policy) != {"kind", "thinking_tokens"} or type(policy["thinking_tokens"]) is not int or policy["thinking_tokens"] < 1:
            raise ValueError("Thinking cap must be a positive integer")
    elif policy.get("kind") == "repeat_ngram":
        if set(policy) != {"kind", "ngram_size", "repeats", "window_tokens"}:
            raise ValueError("Repeat policy requires ngram_size, repeats and window_tokens")
        if any(type(policy[key]) is not int or policy[key] < 1 for key in ("ngram_size", "repeats", "window_tokens")):
            raise ValueError("Repeat policy values must be positive integers")
        if policy["repeats"] < 2 or policy["window_tokens"] < policy["ngram_size"] + policy["repeats"] - 1:
            raise ValueError("Repeat policy cannot reach its trigger within the window")
    else:
        raise ValueError("Unknown decoding policy")


@dataclass(frozen=True)
class EvaluationCell:
    definition: dict
    config_sha256: str

    @property
    def stage(self):
        return self.definition["stage"]

    @property
    def provenance(self):
        return {"cell": self.definition, "config_sha256": self.config_sha256,
                "seed_scheme": SEED_SCHEME, "preregistration": "QUESTIONS.md#evaluation-grid-preregistration-steering-update-2-2026-11-03",
                "protocol_variant": True, "source": "docs/STEERING_UPDATE_2.md Part 3.2"}


def load_cell(stage, *, directory=CONFIG_DIR):
    if stage not in STAGES:
        raise ValueError("Unknown evaluation grid stage")
    raw = (Path(directory) / f"{stage}.json").read_bytes()
    cell = json.loads(raw)
    expected = {"schema_version", "stage", "protocol", "model", "benchmark", "samples_per_problem",
                "thinking", "sampling", "engine_settings", "decoding_policy", "seed_scheme"}
    if set(cell) != expected or cell["stage"] != stage or cell["schema_version"] != 1:
        raise ValueError("Invalid cell schema or stage binding")
    if cell["model"] not in {"base", "p50c", "live"} or cell["benchmark"] != "math500" or cell["samples_per_problem"] != 1:
        raise ValueError("Unregistered grid model or benchmark")
    if type(cell["thinking"]) is not bool or cell["seed_scheme"] != SEED_SCHEME:
        raise ValueError("Invalid thinking or seed policy")
    sampling = cell["sampling"]
    if set(sampling) - {"temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty", "max_tokens"}:
        raise ValueError("Unregistered sampling option")
    if not {"temperature", "top_p", "top_k", "presence_penalty", "max_tokens"} <= set(sampling):
        raise ValueError("Missing think35 sampling options")
    if type(sampling["max_tokens"]) is not int or sampling["max_tokens"] < 1:
        raise ValueError("Invalid output-token cap")
    engine = cell["engine_settings"]
    if set(engine) != {"max_model_len", "max_num_seqs"} or any(type(v) is not int or v < 1 for v in engine.values()):
        raise ValueError("Invalid engine settings")
    if engine["max_model_len"] != sampling["max_tokens"] + 2048:
        raise ValueError("Grid requires an explicit 2048-token prompt allowance")
    validate_policy(cell["decoding_policy"])
    if cell["decoding_policy"] is not None and not cell["thinking"]:
        raise ValueError("Forced closure requires thinking enabled")
    return EvaluationCell(cell, hashlib.sha256(raw).hexdigest())


class ThinkingCloseState:
    """Incremental generated-token policy with a bounded repetition window.

    Call ``observe`` only for generated token IDs, never the templated prompt.
    A decoder can force its next sampled token to ``close_token_id`` when
    ``reason_to_close`` is set. Sampling/penalty state stays in the same request.
    A naturally observed close permanently disables interventions.
    """

    def __init__(self, policy, close_token_id, *, max_tokens):
        validate_policy(policy)
        if policy is None or type(close_token_id) is not int or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("A policy, integer close token and positive output cap are required")
        self.policy = dict(policy)
        self.close_token_id = close_token_id
        self.max_tokens = max_tokens
        self.generated_tokens = 0
        self.closed = False
        self.reason_to_close = None
        self._tail = deque(maxlen=policy.get("ngram_size", 1))
        self._grams = deque()
        self._counts = Counter()

    def observe(self, token_id):
        if self.generated_tokens >= self.max_tokens:
            raise ValueError("Generated tokens exceed the total output cap")
        if type(token_id) is not int:
            raise ValueError("Token IDs must be integers")
        self.generated_tokens += 1
        if token_id == self.close_token_id:
            self.closed = True
        if self.closed or self.generated_tokens == self.max_tokens:
            self.reason_to_close = None
            return None
        if self.policy["kind"] == "thinking_token_cap":
            if self.generated_tokens >= self.policy["thinking_tokens"]:
                self.reason_to_close = "thinking_token_cap"
        else:
            self._tail.append(token_id)
            if len(self._tail) == self.policy["ngram_size"]:
                gram = tuple(self._tail)
                self._grams.append(gram)
                self._counts[gram] += 1
                # Only n-grams wholly within the trailing window are retained.
                capacity = self.policy["window_tokens"] - self.policy["ngram_size"] + 1
                if len(self._grams) > capacity:
                    oldest = self._grams.popleft()
                    self._counts[oldest] -= 1
                    if not self._counts[oldest]:
                        del self._counts[oldest]
                if self._counts[gram] >= self.policy["repeats"]:
                    self.reason_to_close = "repeat_ngram"
        return self.reason_to_close

    def observe_many(self, token_ids):
        for token_id in token_ids:
            self.observe(int(token_id))
        return self.reason_to_close


def close_metadata(token_ids, policy, close_token_id, *, max_tokens):
    """Replay the exact token prefix to audit whether the decoder intervened.

    The forced position uses zero-based generated-token indexing. A naturally
    stopped sequence is not extended just because its final prefix meets a
    threshold. Actual enforcement is checked whenever a next token exists.
    """
    state = ThinkingCloseState(policy, close_token_id, max_tokens=max_tokens)
    reason, position = None, None
    for index, token in enumerate(token_ids):
        if state.reason_to_close:
            if token != close_token_id:
                raise ValueError("Decoder did not enforce the preregistered thinking-close rule")
            reason, position = state.reason_to_close, index
        state.observe(int(token))
    return {"forced_close": position is not None, "forced_close_reason": reason,
            "forced_close_token_index": position, "forced_close_token_count": int(position is not None)}


def grid_live_identity(overrides, *, data=Path("/data")):
    """Default grid commands evaluate the saved active attempt's submission."""
    from replications.exp8.run_identity import live_identity

    if "editor" in overrides or "run_id" in overrides:
        return live_identity(overrides)
    path = Path(data) / "active-live.json"
    if path.exists():
        active = json.loads(path.read_text())
        return live_identity({key: active[key] for key in ("editor", "run_id") if key in active})
    return live_identity(overrides)


def require_grid_priority(stage, *, identity, data=Path("/data")):
    """A direct stage invocation cannot skip the operator's spending order."""
    data = Path(data)
    state_path = data / identity.relative / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    submission = state.get("submission") or {}
    if not state.get("done") or state.get("passes") != 50 or not submission.get("sha256") or not submission.get("checkpoint"):
        raise ValueError("Evaluation grid waits for the actual 50-pass faithful submission")
    for benchmark, samples in CONFIG["samples"].items():
        path = data / "results" / identity.evaluation_stage / benchmark / "summary.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        if (result.get("status") != "done" or result.get("samples_per_problem") != samples
                or result.get("provenance", {}).get("submission_sha256") != submission["sha256"]):
            raise ValueError("Evaluation grid waits for all final evaluations bound to the faithful submission")
    path = data / "results/control4_random/math500/summary.json"
    if not path.exists() or json.loads(path.read_text()).get("status") != "done":
        raise ValueError("Evaluation grid waits for control4")
    for earlier in STAGES[:STAGES.index(stage)]:
        path = data / "results" / earlier / "math500/summary.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        if (result.get("status") != "done"
                or result.get("provenance", {}).get("eval_grid", {}).get("config_sha256") != load_cell(earlier).config_sha256
                or result.get("provenance", {}).get("faithful_submission_sha256") != submission["sha256"]):
            raise ValueError(f"Evaluation grid waits for preregistered {earlier}")
    return submission


def run_grid_stage(stage, ledger, checkpoint, *, identity, data=Path("/data")):
    from replications.exp8.evaluation import evaluate

    cell = load_cell(stage)
    submission = require_grid_priority(stage, identity=identity, data=data)
    definition = cell.definition
    model = definition["model"]
    checkpoint_dir = {"base": str(Path(data) / "original"), "p50c": str(Path(data) / "p50c"),
                      "live": submission["checkpoint"]}[model]
    provenance = {"eval_grid": cell.provenance, "faithful_submission_sha256": submission["sha256"]}
    if model == "live":
        provenance["submission_sha256"] = submission["sha256"]
    return evaluate(stage, checkpoint_dir, "math500", 1, ledger, checkpoint,
                    thinking=definition["thinking"], changes=definition["sampling"],
                    engine_settings=definition["engine_settings"],
                    generation_options={"seed_by_id": True, "seed_scheme": SEED_SCHEME,
                                        "decoding_policy": definition["decoding_policy"]},
                    extra_provenance=provenance)


def tiny_validation():
    """Published, deterministic implementation checks; no model calls or scores."""
    close = 999
    cap = ThinkingCloseState({"kind": "thinking_token_cap", "thinking_tokens": 3}, close, max_tokens=8)
    before = cap.observe_many([1, 2]) is None
    trigger = cap.observe(3) == "thinking_token_cap"
    after = cap.observe_many([close, 4, 5]) is None and cap.closed and cap.generated_tokens == 6
    natural = ThinkingCloseState({"kind": "thinking_token_cap", "thinking_tokens": 3}, close, max_tokens=8)
    natural_ok = natural.observe_many([1, close, 2, 3]) is None
    repeat = ThinkingCloseState({"kind": "repeat_ngram", "ngram_size": 8, "repeats": 5, "window_tokens": 2048}, close, max_tokens=32768)
    overlap_before = repeat.observe_many([7] * 11) is None
    overlap_at = repeat.observe(7) == "repeat_ngram"
    exhausted = ThinkingCloseState({"kind": "thinking_token_cap", "thinking_tokens": 3}, close, max_tokens=3)
    total_cap = exhausted.observe_many([1, 2, 3]) is None
    checks = {"no_early_close": before, "close_after_thinking_cap": trigger,
              "one_close_then_answer": after, "natural_close_disables_intervention": natural_ok,
              "overlapping_eightgram_occurrences": overlap_before and overlap_at,
              "no_injected_token_after_total_cap": total_cap,
              "all_registered_configs_load": all(load_cell(name).stage == name for name in STAGES)}
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"schema_version": 1, "status": "validated", "stage": "eval_grid_cpu_validation",
            "benchmark": "token_state_machine", "checks": checks, "checks_passed": sum(checks.values()),
            "provenance": {"synthetic": True, "scope": "Tiny implementation validation only; not a model benchmark.",
                           "command": "python -m replications.exp8.eval_grid validate --output replications/exp8/results/eval_grid_cpu_validation/summary.json",
                           "config_sha256": {name: load_cell(name).config_sha256 for name in STAGES},
                           "model_calls": 0, "gpu_calls": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "validate"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "list":
        print(json.dumps([load_cell(name).definition for name in STAGES], indent=2))
    else:
        result = tiny_validation()
        if args.output:
            atomic_write(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
