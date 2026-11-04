"""Private, prerequisite-gated teacher-forced damage measurement.

The driver freezes reference tokenization and byte identities before launching
an existing accounted GPU job. Neither reference text nor raw logits are
returned by the worker. A four-reference identity smoke shares the container
with the full comparison and is never included in its mean.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from openwsp.backends.forward_bank import file_hash, json_hash
from openwsp.budget import compute_cost
from replications.exp8.damage import PRODUCTION_SCOPE
from replications.exp8.eval_grid import STAGES as GRID_STAGES
from replications.exp8.eval_grid import load_cell, require_grid_priority
from replications.exp8.jobs import ProtocolGPUJobs, gpu_instance
from replications.exp8.runtime import CONFIG, INSTRUCTION, ROOT, atomic_write, read_rows

STAGE = "damage"
SMOKE_INDICES = [0, 1, 256, 257]
TOKENIZATION = "encode(rendered thinking prompt, add_special_tokens=False) + encode(unchanged solution, add_special_tokens=False)"
ENGINE_SETTINGS = {"dtype": "bfloat16", "attention": "sdpa", "use_cache": False,
                   "kl_accumulation": "float64", "direction": "base||edited", "units": "nats"}
BENCHMARK_SIZES = {"math500": 500, "gsm8k": 1319, "ifeval": 541}


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("Damage metadata must be a JSON object")
    return value


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if path.is_symlink() or _read(path) != value:
            raise ValueError("Existing damage evidence has a conflicting immutable identity")
    else:
        atomic_write(path, value)


def require_damage_priority(*, identity, data=Path("/data")):
    data = Path(data)
    live_path = data / identity.relative / "state.json"
    live = _read(live_path) if live_path.exists() else {}
    if live.get("done") is not True or type(live.get("passes")) is not int or live["passes"] != 50:
        raise ValueError("Damage requires the actual completed integer 50-pass faithful submission")
    submission = require_grid_priority(GRID_STAGES[-1], identity=identity, data=data)
    # The grid helper checks predecessors; damage also requires its final cell.
    last = GRID_STAGES[-1]
    path = Path(data) / "results" / last / "math500/summary.json"
    result = _read(path) if path.exists() else {}
    if (result.get("status") != "done"
            or result.get("provenance", {}).get("eval_grid", {}).get("config_sha256") != load_cell(last).config_sha256
            or result.get("provenance", {}).get("faithful_submission_sha256") != submission["sha256"]):
        raise ValueError("Damage measurement waits for the complete preregistered evaluation grid")
    benchmarks = {}
    for benchmark, count in BENCHMARK_SIZES.items():
        rows = read_rows(data / "benchmarks" / f"{benchmark}.jsonl")
        ids = [row["id"] for row in rows]
        if len(ids) != count or len(set(ids)) != count or any(not isinstance(value, str) for value in ids):
            raise ValueError("Damage prerequisites require the full pinned benchmark coverage")
        benchmarks[benchmark] = ids
    required = [(data / "results" / identity.evaluation_stage / benchmark, benchmark, samples)
                for benchmark, samples in CONFIG["samples"].items()]
    required += [(data / "results" / name / "math500", "math500", 1)
                 for name in ("control4_random", *GRID_STAGES)]
    for directory, benchmark, samples in required:
        summary = _read(directory / "summary.json")
        expected = {(identifier, index) for identifier in benchmarks[benchmark] for index in range(samples)}
        sample_path = directory / "samples.jsonl"
        records = read_rows(sample_path) if sample_path.exists() else []
        if (type(summary.get("n")) is not int or summary["n"] != len(expected)
                or type(summary.get("samples_per_problem")) is not int or summary["samples_per_problem"] != samples
                or len(records) != len(expected)
                or any(not isinstance(row.get("problem_id"), str) or type(row.get("sample_index")) is not int for row in records)
                or {(row["problem_id"], row["sample_index"]) for row in records} != expected):
            raise ValueError("Damage waits for full benchmark sample coverage, not a completed subset")
    return submission


def checkpoint_identity(directory):
    directory = Path(directory)
    files = sorted(path for path in directory.iterdir()
                   if path.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model"}
                   and path.name != "openwsp-complete.json")
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("Damage requires an actual checkpoint with weight files")
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("Damage checkpoint identities require regular files")
    hashes = {path.name: file_hash(path) for path in files}
    return {"path": str(directory), "sha256": json_hash(hashes), "files": hashes}


def _tokenizer_files(checkpoint):
    files = {name: value for name, value in checkpoint["files"].items()
             if name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "chat_template"))
             or name.endswith(".model")}
    if not files:
        raise ValueError("Damage requires fingerprinted tokenizer assets")
    return files


def tokenizer_identity(tokenizer):
    return {"vocabulary_sha256": json_hash(tokenizer.get_vocab()),
            "special_token_ids": list(tokenizer.all_special_ids),
            "chat_template_sha256": json_hash(tokenizer.chat_template)}


def prepare_references(tokenizer, practice, *, expected_hashes=None):
    """Select exactly the saved first256 rows per source; never trim solutions."""
    if expected_hashes is None:
        expected_hashes = _read(ROOT / "sources.json")["files"]
    examples, records, sources = [], [], {}
    for source, filename in (("gsm8k", "gsm8k_train.jsonl"), ("math", "math_train.jsonl")):
        path = Path(practice) / filename
        expected = expected_hashes[f"exp8_math/practice/{filename}"]
        raw = path.read_bytes()
        if path.is_symlink() or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Practice reference file differs from the pinned source")
        sources[filename] = expected
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) < 256:
            raise ValueError("Damage requires 256 complete references per practice source")
        for position, row in enumerate(rows[:256]):
            if (row.get("source") != source or any(not isinstance(row.get(key), str) or not row[key].strip()
                                                   for key in ("id", "problem", "solution"))):
                raise ValueError("Reference rows require source, ID, problem and unchanged full solution")
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["problem"] + "\n" + INSTRUCTION}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            reference_ids = tokenizer.encode(row["solution"], add_special_tokens=False)
            if not prompt_ids or not reference_ids or len(prompt_ids) + len(reference_ids) > PRODUCTION_SCOPE["context_limit"]:
                raise ValueError("Full prompt/reference exceeds context or has no tokens; truncation is forbidden")
            record = {"input_ids": prompt_ids + reference_ids,
                      "score_mask": [False] * len(prompt_ids) + [True] * len(reference_ids)}
            records.append(record)
            examples.append({"id": row["id"], "source": source, "source_row": position,
                             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                             "reference_sha256": hashlib.sha256(row["solution"].encode()).hexdigest(),
                             "token_sha256": json_hash(record), "prompt_tokens": len(prompt_ids),
                             "reference_tokens": len(reference_ids)})
    if len({row["id"] for row in examples}) != 512:
        raise ValueError("Damage reference IDs must be unique")
    manifest = {"schema_version": 1, "scope": PRODUCTION_SCOPE, "source_sha256": sources,
                "tokenization": TOKENIZATION, "tokenizer": tokenizer_identity(tokenizer), "examples": examples,
                "smoke_indices": SMOKE_INDICES, "smoke_in_measurement": False,
                "record_sha256": json_hash(records)}
    return manifest, {"records": records}


def _edit_identity(directory):
    directory = Path(directory)
    paths = [directory / name for name in ("manifest.json", "edit.npz")]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Damage requires the immutable exported edit bundle")
    files = {path.name: file_hash(path) for path in paths}
    manifest = _read(directory / "manifest.json")
    address, address_format = None, "legacy_unversioned"
    if "format" in manifest or "format_version" in manifest:
        from openwsp.edits import FORMAT, FORMAT_VERSION, edit_sha256

        if (manifest.get("format") != FORMAT or type(manifest.get("format_version")) is not int
                or manifest["format_version"] != FORMAT_VERSION or manifest.get("edit_sha256") != edit_sha256(manifest)):
            raise ValueError("Versioned edit manifest address is invalid")
        address, address_format = manifest["edit_sha256"], "openwsp.edit.v1"
    return {"path": str(directory), "sha256": json_hash(files), "files": files,
            "edit_sha256": address, "edit_address_format": address_format}


def _source_identity(inspect, source, resolve_path):
    """Hash local bytes while retaining the manifest's original path namespace."""
    return {**inspect(resolve_path(source)), "path": str(source)}


def prepare_manifests(tokenizer, *, identity, submission, data=Path("/data"), practice=Path("/practice"),
                      expected_hashes=None, resolve_path=Path):
    data = Path(data)
    directory = data / "damage" / identity.relative
    dataset, records = prepare_references(tokenizer, resolve_path(practice), expected_hashes=expected_hashes)
    base = _source_identity(checkpoint_identity, data / "original", resolve_path)
    choices = [("p50c", {"checkpoint": str(data / "p50c")}, data / "reference_edit"),
               ("mechanical", _read(resolve_path(data / "controls/mechanical/submission.json")), data / "controls/mechanical/edit"),
               ("random", _read(resolve_path(data / "controls/random/submission.json")), data / "controls/random/edit"),
               ("live", submission, data / identity.relative / "edit")]
    manifests = []
    for name, selected, edit_path in choices:
        edited = _source_identity(checkpoint_identity, selected["checkpoint"], resolve_path)
        if _tokenizer_files(base) != _tokenizer_files(edited):
            raise ValueError("Base and edited tokenizer/template bytes differ")
        manifest = {"schema_version": 1, "stage": STAGE, "edit": name, "dataset": dataset,
                    "base_checkpoint": base, "edited_checkpoint": edited,
                    "edit_bundle": _source_identity(_edit_identity, edit_path, resolve_path),
                    "declared_submission_sha256": selected.get("sha256"),
                    "faithful_submission_sha256": submission["sha256"], "identity": {"editor": identity.editor, "run_id": identity.run_id},
                    "records_path": str(directory / "references.tokens.json"), "chunk_tokens": 32,
                    "engine_settings": ENGINE_SETTINGS}
        manifests.append((directory / f"{name}.manifest.json", manifest))
    # Validate every edit before creating any new immutable record.
    for path, value in [(directory / "references.tokens.json", records), (directory / "references.manifest.json", dataset), *manifests]:
        path = resolve_path(path)
        if path.exists() and _read(path) != value:
            raise ValueError("Existing damage evidence has a conflicting immutable identity")
    for path, value in [(directory / "references.tokens.json", records), (directory / "references.manifest.json", dataset), *manifests]:
        _immutable(resolve_path(path), value)
    return manifests


def load_measurement_inputs(path, expected_sha256, *, base_checkpoint, tokenizer, resolve_path=Path):
    """Verify bytes and all 512 token masks before any teacher-forced call."""
    path = Path(path)
    manifest = _read(resolve_path(path))
    if resolve_path(path).is_symlink() or json_hash(manifest) != expected_sha256:
        raise ValueError("Damage manifest hash differs from the accounted request")
    dataset = manifest["dataset"]
    if (manifest.get("schema_version") != 1 or manifest.get("stage") != STAGE
            or dataset.get("scope") != PRODUCTION_SCOPE or dataset.get("tokenization") != TOKENIZATION
            or dataset.get("smoke_indices") != SMOKE_INDICES or dataset.get("smoke_in_measurement") is not False
            or dataset.get("tokenizer") != tokenizer_identity(tokenizer) or manifest.get("chunk_tokens") != 32
            or manifest.get("engine_settings") != ENGINE_SETTINGS
            or str(base_checkpoint) != manifest["base_checkpoint"]["path"]):
        raise ValueError("Damage manifest protocol/tokenizer binding differs")
    for key in ("base_checkpoint", "edited_checkpoint"):
        if _source_identity(checkpoint_identity, manifest[key]["path"], resolve_path) != manifest[key]:
            raise ValueError("Checkpoint bytes changed after damage preregistration")
    if _tokenizer_files(manifest["base_checkpoint"]) != _tokenizer_files(manifest["edited_checkpoint"]):
        raise ValueError("Damage vocabulary assets differ")
    if _source_identity(_edit_identity, manifest["edit_bundle"]["path"], resolve_path) != manifest["edit_bundle"]:
        raise ValueError("Edit bytes changed after damage preregistration")
    records_path = Path(manifest["records_path"])
    if (records_path.parent != path.parent or resolve_path(records_path).is_symlink()
            or resolve_path(records_path).parent != resolve_path(path).parent):
        raise ValueError("Damage token records must stay beside the manifest")
    records = _read(resolve_path(records_path))["records"]
    examples = dataset["examples"]
    if (len(records) != 512 or len(examples) != 512 or json_hash(records) != dataset["record_sha256"]
            or [row["source"] for row in examples] != ["gsm8k"] * 256 + ["math"] * 256
            or len({row["id"] for row in examples}) != 512):
        raise ValueError("Damage references differ from the balanced 512-example manifest")
    vocabulary = set(tokenizer.get_vocab().values())
    for index, (record, example) in enumerate(zip(records, examples, strict=True)):
        prompt, reference = example["prompt_tokens"], example["reference_tokens"]
        if (type(prompt) is not int or type(reference) is not int or min(prompt, reference) < 1
                or prompt + reference > PRODUCTION_SCOPE["context_limit"] or example["source_row"] != index % 256
                or set(record) != {"input_ids", "score_mask"} or len(record["input_ids"]) != prompt + reference
                or any(type(token) is not int or token not in vocabulary for token in record["input_ids"])
                or any(type(bit) is not bool for bit in record["score_mask"])
                or record["score_mask"] != [False] * prompt + [True] * reference
                or json_hash(record) != example["token_sha256"]):
            raise ValueError("Damage continuation tokens/masks do not match their immutable reference")
    return manifest, records


def measure_manifest(base_model, tokenizer, path, expected_sha256, *, base_checkpoint, load_edited,
                     resolve_path=Path):
    from openwsp.backends.damage import teacher_forced_kl

    manifest, records = load_measurement_inputs(path, expected_sha256, base_checkpoint=base_checkpoint,
                                               tokenizer=tokenizer, resolve_path=resolve_path)
    smoke = teacher_forced_kl(base_model, base_model, [records[index] for index in SMOKE_INDICES])
    if smoke["kl_sum"] != 0 or smoke["token_count"] < 4 or smoke["scored_sequence_count"] != 4:
        raise ValueError("Damage identity smoke failed; the edited measurement was not started")
    edited = load_edited(manifest["edited_checkpoint"]["path"])
    try:
        measurement = teacher_forced_kl(base_model, edited, records, chunk_tokens=manifest["chunk_tokens"])
    finally:
        del edited
    return {"status": "done", "manifest_sha256": expected_sha256, "measurement": measurement,
            "smoke": {**smoke, "reference_indices": SMOKE_INDICES, "included_in_measurement": False}}


def _validate_result(metrics, smoke, manifest):
    examples = manifest["dataset"]["examples"]
    for value, selected in ((metrics, examples), (smoke, [examples[index] for index in SMOKE_INDICES])):
        expected_tokens = sum(row["reference_tokens"] for row in selected)
        if (value.get("sequence_count") != len(selected) or value.get("scored_sequence_count") != len(selected)
                or value.get("token_count") != expected_tokens or value.get("chunk_tokens") != 32
                or value.get("input_token_count") != sum(row["prompt_tokens"] + row["reference_tokens"] for row in selected)
                or any(type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0
                       for key in ("kl_sum", "kl_mean"))
                or not math.isclose(value["kl_mean"], value["kl_sum"] / expected_tokens, rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError("Damage worker did not complete the exact registered measurement")
    if (smoke.get("kl_sum") != 0 or smoke.get("included_in_measurement") is not False
            or smoke.get("reference_indices") != SMOKE_INDICES):
        raise ValueError("Damage identity smoke is missing or included in the measurement")


def prepare_damage_stage(checkpoint, *, identity, data=Path("/data"), practice=Path("/practice")):
    """Freeze authoritative manifests on the existing CPU driver, without dispatch."""
    submission = require_damage_priority(identity=identity, data=data)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(data) / "original", local_files_only=True)
    manifests = prepare_manifests(tokenizer, identity=identity, submission=submission, data=data, practice=practice)
    checkpoint()  # All dataset/edit identities must be durable before the first job.
    return {"status": "done", "stage": "damage_prepare", "faithful_submission_sha256": submission["sha256"],
            "manifests": [{"edit": manifest["edit"], "path": str(path), "sha256": json_hash(manifest)}
                          for path, manifest in manifests], "hash_algorithm": "canonical_json_sha256"}


def run_damage_stage(ledger, checkpoint, *, identity, data=Path("/data"), practice=Path("/practice")):
    from openwsp.backends.modal_app import TransformerReader

    prepared = prepare_damage_stage(checkpoint, identity=identity, data=data, practice=practice)
    manifests = [(Path(row["path"]), _read(row["path"])) for row in prepared["manifests"]]
    if any(json_hash(manifest) != row["sha256"] for (_, manifest), row in zip(manifests, prepared["manifests"], strict=True)):
        raise ValueError("Damage manifests changed after source preparation")
    runner = ProtocolGPUJobs(ledger, Path(data) / "jobs", commit=checkpoint)
    summaries = {}
    for manifest_path, manifest in manifests:
        name, manifest_sha = manifest["edit"], json_hash(manifest)
        target = Path(data) / "results/damage" / identity.relative / name / "summary.json"
        if target.exists():
            summary = _read(target)
            if summary.get("status") != "done" or summary.get("provenance", {}).get("manifest_sha256") != manifest_sha:
                raise ValueError("Saved damage result belongs to different evidence")
            _validate_result(summary, summary["smoke"], manifest)
            summaries[name] = summary
            continue
        key = f"damage/{identity.relative}/{name}"
        runner.launch(key, lambda gpu, gate, path=manifest_path, sha=manifest_sha, base=manifest["base_checkpoint"]["path"]: gpu_instance(
            TransformerReader, gpu, gate, checkpoint=base).damage.spawn(str(path), sha),
            fingerprint={"manifest_sha256": manifest_sha})
        result = runner.result(key, timeout=3600)
        metrics = result["measurement"]
        if result.get("status") != "done" or result.get("manifest_sha256") != manifest_sha:
            raise ValueError("Damage worker did not complete the exact registered measurement")
        _validate_result(metrics, result["smoke"], manifest)
        seconds = result.get("_openwsp_gpu", {}).get("billable_seconds", result["container_seconds"] + 122)
        cost = result.get("_openwsp_gpu", {}).get("total_settled_cost_usd")
        summary = {"schema_version": 1, "stage": "damage", "status": "done", "edit": name,
                   "metric": "kl_base_to_edited", "units": "nats", **metrics, "smoke": result["smoke"],
                   "seconds": seconds, "cost_usd": float(cost if cost is not None else compute_cost(
                       seconds, gpu=result["gpu"], cpu_cores=4, memory_gib=16)),
                   "provenance": {"manifest_sha256": manifest_sha, "dataset_sha256": json_hash(manifest["dataset"]),
                                  "dataset": manifest["dataset"], "base_checkpoint": manifest["base_checkpoint"],
                                  "edited_checkpoint": manifest["edited_checkpoint"],
                                  "edit_bundle_sha256": manifest["edit_bundle"]["sha256"],
                                  "edit_sha256": manifest["edit_bundle"]["edit_sha256"],
                                  "edit_address_format": manifest["edit_bundle"]["edit_address_format"],
                                  "faithful_submission_sha256": prepared["faithful_submission_sha256"], "engine_settings": manifest["engine_settings"],
                                  "gpu": result["gpu"], "seconds_basis": "accounted_resource_seconds_including_smoke_and_margins",
                                  "cost_basis": "existing_job_ledger_estimate_not_provider_invoice"}}
        _immutable(target, summary)
        summaries[name] = summary
        checkpoint()
    return {"status": "done", "edits": summaries, "faithful_submission_sha256": prepared["faithful_submission_sha256"]}
