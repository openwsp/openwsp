"""CLI entrypoints for separately versioned replication workflows."""
from __future__ import annotations

import json
import subprocess
import sys
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
    subprocess.run([sys.executable, str(root / "scripts/modal_cli.py"), "deploy",
                    str(root / "src/openwsp/backends/modal_app.py")], cwd=root, check=True)
    import modal
    function = modal.Function.from_name("openwsp-exp8", "pipeline")
    if detach:
        call = function.spawn(stage, settings)
        result = {"experiment": experiment, "stage": stage, "call_id": call.object_id, "status": "launched"}
    else:
        result = function.remote(stage, settings)
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
