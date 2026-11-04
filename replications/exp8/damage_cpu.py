"""Run the registered KL measurement on a private local mirror, without Modal.

The required evaluation queue must already be complete. Source manifests retain
their /data and /practice paths; only filesystem access is relocated. No network,
generation, GPU fallback, budget ledger or download operation is provided.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path

from openwsp.backends.forward_bank import file_hash, json_hash
from replications.exp8 import damage_stage as stage
from replications.exp8.run_identity import LiveIdentity
from replications.exp8.runtime import CONFIG

EDITS = ("p50c", "mechanical", "random", "live")


def mirror_paths(data, practice):
    """Resolve only canonical source paths inside private, symlink-free roots."""
    roots = {}
    for source, value in ((Path("/data"), data), (Path("/practice"), practice)):
        root = Path(value)
        if not root.is_absolute() or root.resolve() != root or not root.is_dir():
            raise ValueError("Mirror roots must be existing absolute directories without symlinks")
        info = root.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Mirror roots must be private (mode 0700) and owned by the caller")
        roots[source] = root

    def resolve(source):
        source = Path(source)
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("Mirror source paths must be absolute and cannot traverse parents")
        for prefix, root in roots.items():
            if source.is_relative_to(prefix):
                target = root / source.relative_to(prefix)
                if target.resolve() != target:
                    raise ValueError("Mirror paths cannot contain symlinks")
                return target
        raise ValueError("Source path is outside /data and /practice")

    return resolve


def _require_priority(identity, resolve):
    # The existing gate uses physical paths. Validate all of its inputs in the
    # source namespace first so mirror-local prerequisite symlinks cannot escape.
    paths = [Path("/data") / identity.relative / "state.json"]
    paths += [Path("/data/benchmarks") / f"{name}.jsonl" for name in stage.BENCHMARK_SIZES]
    directories = [Path("/data/results") / identity.evaluation_stage / name for name in CONFIG["samples"]]
    directories += [Path("/data/results") / name / "math500" for name in ("control4_random", *stage.GRID_STAGES)]
    paths += [directory / name for directory in directories for name in ("summary.json", "samples.jsonl")]
    for path in paths:
        resolve(path)
    return stage.require_damage_priority(identity=identity, data=resolve("/data"))


def _source_manifests(identity, expected, resolve):
    """Require caller-supplied anchors captured independently from source /data."""
    if (not isinstance(expected, dict) or set(expected) != set(EDITS)
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in expected.values())):
        raise ValueError("Supply all four independently recorded source manifest SHA-256 values")
    for name in EDITS:
        path = resolve(Path("/data/damage") / identity.relative / f"{name}.manifest.json")
        if not path.is_file() or json_hash(stage._read(path)) != expected[name]:
            raise ValueError("Mirrored manifest does not match the independently recorded source manifest hash")


def _tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _model(path):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    # Same pinned BF16/SDPA loader as the completed local repair extraction.
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    ).to("cpu").eval()


def _execution(threads):
    from openwsp.backends import damage

    if version("transformers") != CONFIG["transformers"]:
        raise ValueError("CPU damage requires the pinned Transformers version")
    hardware = {"visible_logical_cpus": os.cpu_count(), "affinity_logical_cpus": None,
                "processor_model": None, "cgroup_v2_cpu_quota_cores": None}
    try:
        hardware["affinity_logical_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        with Path("/proc/cpuinfo").open() as handle:
            lines = handle.read(4096).splitlines()
        hardware["processor_model"] = next((line.partition(":")[2].strip()[:200]
            for line in lines if line.partition(":")[0].strip() in ("model name", "Hardware")), None)
    except OSError:
        pass
    try:
        with Path("/sys/fs/cgroup/cpu.max").open() as handle:
            quota, period = handle.read(64).split()
        if quota != "max" and int(period) > 0:
            hardware["cgroup_v2_cpu_quota_cores"] = int(quota) / int(period)
    except (OSError, ValueError):
        pass
    return {"device": "cpu", "threads": threads, "torch": version("torch"),
            "transformers": version("transformers"), "machine": platform.machine(),
            "cpu_hardware": hardware,
            "implementation_sha256": {"runner": file_hash(__file__),
                                      "stage": file_hash(stage.__file__), "kernel": file_hash(damage.__file__)}}


def _provenance(manifest, execution):
    return {"manifest_sha256": json_hash(manifest), "dataset_sha256": json_hash(manifest["dataset"]),
            "dataset": manifest["dataset"], "base_checkpoint": manifest["base_checkpoint"],
            "edited_checkpoint": manifest["edited_checkpoint"],
            "edit_bundle_sha256": manifest["edit_bundle"]["sha256"],
            "edit_sha256": manifest["edit_bundle"]["edit_sha256"],
            "edit_address_format": manifest["edit_bundle"]["edit_address_format"],
            "faithful_submission_sha256": manifest["faithful_submission_sha256"],
            "engine_settings": manifest["engine_settings"], "execution": execution, "gpu": None,
            "source_binding": "caller-supplied independently recorded source manifest canonical JSON SHA-256, checked before preparation",
            "seconds_basis": "local wall time including input verification, identity smoke and model loading; "
                             "the first computed edit includes the shared base model load; "
                             "excludes initial source-anchor checks, reference preparation and global preflight",
            "cost_basis": "zero incremental Modal usage; existing workspace CPU resources are unpriced, not free"}


def _completed(path, manifest, execution):
    if not path.exists():
        return None
    summary = stage._read(path)
    if (summary.get("schema_version") != 1 or summary.get("status") != "done" or summary.get("stage") != "damage"
            or summary.get("edit") != manifest["edit"]
            or summary.get("metric") != "kl_base_to_edited" or summary.get("units") != "nats"
            or summary.get("provenance") != _provenance(manifest, execution)
            or summary.get("cost_usd") != 0 or summary.get("incremental_modal_cost_usd") != 0
            or summary.get("local_cpu_cost_usd", "missing") is not None
            or type(summary.get("seconds")) not in (int, float)
            or not math.isfinite(summary["seconds"]) or summary["seconds"] < 0):
        raise ValueError("Existing damage result differs from this CPU measurement; no replacement is run")
    stage._validate_result(summary, summary.get("smoke", {}), manifest)
    return summary


def run_cpu(*, data, practice, identity, expected_manifest_sha256, threads=2):
    """Complete missing exact-bound edit measurements; never start the queue."""
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("Use one through four CPU threads")
    resolve = mirror_paths(data, practice)
    # This must precede tokenization, manifest preparation and model loading.
    submission = _require_priority(identity, resolve)
    with resolve("/data/.damage-cpu.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        submission = _require_priority(identity, resolve)
        _source_manifests(identity, expected_manifest_sha256, resolve)
        execution = _execution(threads)
        tokenizer = _tokenizer(resolve("/data/original"))
        manifests = stage.prepare_manifests(tokenizer, identity=identity, submission=submission,
                                            resolve_path=resolve)
        summaries, pending = {}, []
        # Validate all inputs and existing results before the first model load.
        for path, manifest in manifests:
            stage.load_measurement_inputs(path, json_hash(manifest),
                base_checkpoint=manifest["base_checkpoint"]["path"], tokenizer=tokenizer, resolve_path=resolve)
            target = resolve(Path("/data/results/damage") / identity.relative / manifest["edit"] / "summary.json")
            existing = _completed(target, manifest, execution)
            if existing is not None:
                summaries[manifest["edit"]] = existing
            else:
                pending.append((path, manifest, target))
        if pending:
            import torch

            torch.set_num_threads(threads)
            base = None
            try:
                for path, manifest, target in pending:
                    started = time.monotonic()
                    if base is None:
                        base = _model(resolve(manifest["base_checkpoint"]["path"]))
                    result = stage.measure_manifest(base, tokenizer, path, json_hash(manifest),
                        base_checkpoint=manifest["base_checkpoint"]["path"],
                        load_edited=lambda source: _model(resolve(source)), resolve_path=resolve)
                    if result.get("status") != "done" or result.get("manifest_sha256") != json_hash(manifest):
                        raise ValueError("CPU damage did not complete the exact registered measurement")
                    stage._validate_result(result["measurement"], result["smoke"], manifest)
                    summary = {"schema_version": 1, "stage": "damage", "status": "done",
                               "edit": manifest["edit"], "metric": "kl_base_to_edited", "units": "nats",
                               **result["measurement"], "smoke": result["smoke"],
                               "seconds": time.monotonic() - started, "cost_usd": 0.0,
                               "incremental_modal_cost_usd": 0.0, "local_cpu_cost_usd": None,
                               "provenance": _provenance(manifest, execution)}
                    stage._immutable(target, summary)
                    summaries[manifest["edit"]] = summary
            finally:
                del base
        return {"status": "done", "edits": {manifest["edit"]: summaries[manifest["edit"]]
                                               for _, manifest in manifests},
                "faithful_submission_sha256": submission["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-mirror", required=True, type=Path)
    parser.add_argument("--practice-mirror", required=True, type=Path)
    parser.add_argument("--editor", choices=("codex", "openai"), default="codex")
    parser.add_argument("--run-id")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--manifest-sha256", action="append", required=True, metavar="EDIT=HASH",
                        help="Repeat for all four canonical JSON hashes independently recorded from source /data manifests")
    args = parser.parse_args()
    anchors = {}
    for value in args.manifest_sha256:
        name, separator, digest = value.partition("=")
        if not separator or name in anchors:
            parser.error("Supply each source manifest hash once as EDIT=HASH")
        anchors[name] = digest
    result = run_cpu(data=args.data_mirror, practice=args.practice_mirror,
                     identity=LiveIdentity(args.editor, args.run_id), threads=args.threads,
                     expected_manifest_sha256=anchors)
    print(json.dumps({"status": result["status"], "edits": list(result["edits"]),
                      "faithful_submission_sha256": result["faithful_submission_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
