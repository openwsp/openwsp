"""CLI entrypoints for separately versioned replication workflows."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
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
        raise ValueError("This Modal entry point supports exp8; CPU experiments 1, 2, 4 and 5 use python -m replications.raw_weights.runner")
    settings = json.loads(Path(config).read_text()) if config else {}
    from openwsp.backends.codex_worker import CodexWorker, split_codex_settings
    settings, local = split_codex_settings(settings, detach=detach)
    root = repository()
    worker = CodexWorker(settings["run_id"], **local) if local is not None else None
    if local is not None:
        for key in ("home", "state_dir", "work_dir"):
            if key in local and any(Path(local[key]).resolve().is_relative_to(root.resolve() / name)
                                    for name in ("src", "replications", "scripts")):
                raise ValueError("Local Codex directories must be outside the deployed source directories")
    from openwsp.backends.modal_runtime import configure_tls
    configure_tls()
    with worker.locked() if worker is not None else nullcontext():
        # Image builds can take minutes while development continues. Modal rejects
        # mutated source mounts; snapshot precisely the files this deployment uses.
        with tempfile.TemporaryDirectory(prefix="openwsp-deploy-") as temporary:
            snapshot = Path(temporary)
            for name in ("src", "replications", "scripts"):
                shutil.copytree(root / name, snapshot / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            # A caller may keep its config in a mounted source tree. Strip local
            # paths there too, rather than sending the original config file.
            if local is not None and config:
                try:
                    relative = Path(config).resolve().relative_to(root.resolve())
                except ValueError:
                    relative = None
                if relative is not None and (snapshot / relative).is_file():
                    (snapshot / relative).write_text(json.dumps(settings, indent=2) + "\n")
            environment = {**os.environ, "PYTHONPATH": str(snapshot / "src"),
                           "OPENWSP_EDITOR_TRANSPORT": settings.get("editor", "openai")}
            subprocess.run([sys.executable, str(snapshot / "scripts/modal_cli.py"), "deploy",
                            "-m", "openwsp.backends.modal_app"], cwd=snapshot, env=environment, check=True)
        import modal
        result = modal.Function.from_name("openwsp-exp8", "launch").remote(stage, settings)
        result["experiment"] = experiment
        from openwsp.budget import atomic_json
        path = root / ".openwsp/launch.json"
        path.parent.mkdir(exist_ok=True)
        # Preserve the provider ID before a foreground worker can be interrupted.
        atomic_json(path, result)
        if not detach and result.get("call_id"):
            call = modal.FunctionCall.from_id(result["call_id"])
            result = worker.run(call) if worker is not None else call.get(timeout=86400)
            atomic_json(path, result)
        return result


def status(experiment="exp8"):
    if experiment != "exp8":
        raise ValueError("Unknown replication")
    from openwsp.backends.modal_runtime import configure_tls
    configure_tls()
    import modal
    return modal.Function.from_name("openwsp-exp8", "status").remote()


def record_modal_allowance(experiment="exp8", *, observation):
    """Record explicit dashboard evidence through the existing serialized owner.

    This command never deploys, creates a ledger, or launches/restarts work.
    """
    if experiment != "exp8":
        raise ValueError("Only exp8 supports Modal allowance observations")
    path = Path(observation)
    if not path.is_absolute():
        raise ValueError("--observation must be an absolute path to a JSON evidence file")
    from openwsp.budget import validate_modal_allowance_observation
    value = validate_modal_allowance_observation(json.loads(path.read_text()))
    from openwsp.backends.modal_runtime import configure_tls
    configure_tls()
    import modal
    return modal.Function.from_name("openwsp-exp8", "launch").remote(allowance_observation=value)


def report(experiment="exp8"):
    if experiment != "exp8":
        raise ValueError("Unknown replication")
    from openwsp.reporting import generate_report
    root = repository()
    return generate_report(root, root / "replications/exp8/results")
