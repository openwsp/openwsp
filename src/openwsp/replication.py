"""CLI entrypoints for separately versioned replication workflows."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def repository():
    root = Path.cwd()
    if not (root / "replications/exp8/config.json").exists():
        root = Path(__file__).resolve().parents[2]
    if not (root / "replications/exp8/config.json").exists():
        raise FileNotFoundError("Run replication commands from the OpenWSP repository checkout")
    return root


def replicate(experiment, *, stage="all", detach=True, config=None):
    if experiment != "exp8":
        raise ValueError("Only exp8 is implemented; experiments 1–7 are layout only")
    root = repository()
    settings = json.loads(Path(config).read_text()) if config else {}
    from openwsp.backends.modal_runtime import configure_tls
    configure_tls()
    # Image builds can take minutes while development continues. Modal rejects
    # mutated source mounts; snapshot precisely the files this deployment uses.
    with tempfile.TemporaryDirectory(prefix="openwsp-deploy-") as temporary:
        snapshot = Path(temporary)
        for name in ("src", "replications", "scripts"):
            shutil.copytree(root / name, snapshot / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        environment = {**os.environ, "PYTHONPATH": str(snapshot / "src")}
        subprocess.run([sys.executable, str(snapshot / "scripts/modal_cli.py"), "deploy",
                        "-m", "openwsp.backends.modal_app"], cwd=snapshot, env=environment, check=True)
    import modal
    result = modal.Function.from_name("openwsp-exp8", "launch").remote(stage, settings)
    result["experiment"] = experiment
    if not detach and result.get("call_id"):
        result = modal.FunctionCall.from_id(result["call_id"]).get(timeout=86400)
    path = root / ".openwsp/launch.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def status(experiment="exp8"):
    if experiment != "exp8":
        raise ValueError("Unknown replication")
    from openwsp.backends.modal_runtime import configure_tls
    configure_tls()
    import modal
    return modal.Function.from_name("openwsp-exp8", "status").remote()


def report(experiment="exp8"):
    if experiment != "exp8":
        raise ValueError("Unknown replication")
    from openwsp.reporting import generate_report
    root = repository()
    return generate_report(root, root / "replications/exp8/results")
