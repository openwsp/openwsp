"""Collect compact evidence; never publish benchmark generations or weight files."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from openwsp.reporting import render_transcript
from replications.exp8.runtime import atomic_write


def collect(data="/data", staging="/data/publication"):
    data, staging = Path(data), Path(staging)
    root = staging / "replications/exp8/results"
    root.mkdir(parents=True, exist_ok=True)
    for path in (data / "results").rglob("summary.json"):
        # Only complete evaluation aggregates appear in published score tables.
        relative = path.relative_to(data / "results")
        if any(part == "smoke" or part.endswith("_smoke") or part.startswith("batch_") for part in relative.parts):
            continue
        for name in ("summary.json", "samples.jsonl"):
            source = path.with_name(name)
            if source.exists():
                destination = root / relative.parent / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    for name in ("sandbox-verification.json", "hash-verification.json", "forward-verification.json", "weight-swap.json"):
        source = data / "results/kernel" / name
        if source.exists():
            (root / "kernel").mkdir(exist_ok=True)
            shutil.copyfile(source, root / "kernel" / name)
    for name, target in (("summary.json", "editor-transport-probe.json"),
                         ("authorization.json", "editor-budget-authorization.json")):
        source = data / "editor-transport-probe" / name
        if source.exists():
            (root / "kernel").mkdir(exist_ok=True)
            shutil.copyfile(source, root / "kernel" / target)
    for name in ("calibration-check.json", "calibration-gap-audit.json", "budget.json"):
        if (data / name).exists():
            shutil.copyfile(data / name, root / ("costs.json" if name == "budget.json" else name))
    live = data / "live-run"
    if live.exists():
        public_live = root / "live-run"
        public_live.mkdir(exist_ok=True)
        render_transcript(live, public_live / "transcript.md")
        state = json.loads((live / "state.json").read_text()) if (live / "state.json").exists() else {}
        compact = {key: state.get(key) for key in ("turn", "runs", "passes", "requests", "handoffs", "done", "submission")}
        compact["base_generation_reuse_requests"] = [p.stem.split(".")[0] for p in (live / "requests").glob("*.provenance.json")
            if json.loads(p.read_text()).get("base_generations_reused")]
        atomic_write(public_live / "state.json", compact)
        deliveries = sorted((live / "requests").glob("*.delivery.json"))
        if deliveries:
            atomic_write(public_live / "forward-artifacts.json",
                         {"receipts": [json.loads(path.read_text()) for path in deliveries]})
        if (live / "edit/manifest.json").exists():
            shutil.copytree(live / "edit", public_live / "edit", dirs_exist_ok=True)
        if (live / "final_answer.md").exists():
            shutil.copyfile(live / "final_answer.md", public_live / "final_answer.md")
        if (live / "transport-reviewed.json").exists():
            shutil.copyfile(live / "transport-reviewed.json", public_live / "transport-reviewed.json")
    control = data / "controls/mechanical"
    if (control / "method.json").exists():
        method = json.loads((control / "method.json").read_text())
        for layer in method["layers"].values():
            layer.pop("direction", None)  # derived vectors remain on the volume
        atomic_write(root / "control2_mechanical/method.json", method)
        shutil.copytree(control / "edit", root / "control2_mechanical/edit", dirs_exist_ok=True)
    random_control = data / "controls/random"
    if (random_control / "method.json").exists():
        atomic_write(root / "control4_random/method.json", json.loads((random_control / "method.json").read_text()))
        shutil.copytree(random_control / "edit", root / "control4_random/edit", dirs_exist_ok=True)
    if (data / "state.json").exists():
        state = json.loads((data / "state.json").read_text())
        atomic_write(root / "pipeline-state.json", {key: state.get(key) for key in
                     ("status", "active_stage", "heartbeat", "error", "error_type", "budget", "controls_deferred")})
    return staging, [str(path.relative_to(staging)) for path in root.rglob("*") if path.is_file()]


def publish():
    from openwsp.publishing import publish_results
    staging, paths = collect()
    return publish_results(staging, paths, checkout_dir="/tmp/openwsp-publish")
