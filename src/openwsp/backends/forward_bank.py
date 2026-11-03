"""Content-addressed forward results for a serialized, provider-independent caller.

Raw outputs and artifacts belong on private storage, not in public result files.
Entries never overwrite an earlier result for the same key. Bank hits carry
provenance separately from the result delivered to a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from ..budget import atomic_json


class BankError(RuntimeError):
    """An existing cache entry cannot be verified; do not silently regenerate."""


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hash(directory):
    """Bind checkpoint, tokenizer, templates and configuration to actual bytes."""
    directory = Path(directory)
    files = sorted(path for path in directory.iterdir()
                   if path.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model"}
                   and path.name != "openwsp-complete.json")
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise BankError("Checkpoint has no weight files to fingerprint")
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise BankError("Checkpoint identity requires regular files")
    return json_hash({path.name: file_hash(path) for path in files})


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _relative(value):
    if not isinstance(value, str):
        raise BankError("Artifact path must be text")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise BankError("Artifact path must remain inside its bank")
    return path


def _validate_output(output, problem_id):
    if (not isinstance(output, dict) or output.get("id") != problem_id
            or not isinstance(output.get("samples"), list) or not output["samples"]):
        raise BankError("Generation output does not match its problem identity")
    for sample in output["samples"]:
        if (not isinstance(sample, dict) or not isinstance(sample.get("text"), str)
                or type(sample.get("token_count")) is not int or sample["token_count"] < 0
                or not isinstance(sample.get("finish_reason"), str)):
            raise BankError("Generation sample is malformed")


class GenerationBank:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def identity(problem_id, prompt, sampling, model_hash, seed):
        if not isinstance(problem_id, str) or not isinstance(prompt, str) or type(seed) is not int:
            raise ValueError("Generation identity needs a problem ID, rendered prompt, and integer seed")
        if not isinstance(model_hash, str) or re.fullmatch(r"[a-f0-9]{64}", model_hash) is None:
            raise ValueError("Generation identity requires a checkpoint SHA-256")
        return {"problem_id": problem_id, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "sampling_sha256": json_hash(sampling), "model_sha256": model_hash, "seed": seed}

    def get(self, identity):
        key = json_hash(identity)
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
            if (path.is_symlink() or record["schema_version"] != 1 or record["key"] != key
                    or record["identity"] != identity or json_hash(record["output"]) != record["output_sha256"]):
                raise BankError("Generation bank identity or payload hash differs")
            _validate_output(record["output"], identity["problem_id"])
            return record
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise BankError("Generation bank record is unreadable or malformed") from error

    def put(self, identity, output, *, provenance):
        existing = self.get(identity)
        if existing is not None:
            return existing
        _validate_output(output, identity["problem_id"])
        record = _copy({"schema_version": 1, "key": json_hash(identity), "identity": identity,
                        "output": output, "output_sha256": json_hash(output), "provenance": provenance})
        atomic_json(self.root / f"{record['key']}.json", record)
        return _copy(record)


class ForwardResultBank:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, identity):
        key = json_hash(identity)
        directory = self.root / key
        if not directory.exists():
            return None
        try:
            if directory.resolve() != directory.absolute():
                raise BankError("Forward bank must not traverse symbolic links")
            record = json.loads((directory / "entry.json").read_text())
            if (record["schema_version"] != 1 or record["key"] != key or record["identity"] != identity
                    or json_hash(record["result"]) != record["result_sha256"]):
                raise BankError("Forward bank identity or result hash differs")
            for item in record["artifacts"]:
                path = directory / "files" / _relative(item["path"])
                if (path.resolve() != path.absolute() or not path.is_file() or path.stat().st_size != item["bytes"]
                        or file_hash(path) != item["sha256"]):
                    raise BankError("Forward bank artifact changed")
            return record
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise BankError("Forward bank is incomplete or malformed") from error

    def put(self, identity, result, *, request_number, artifacts):
        existing = self.get(identity)
        if existing is not None:
            return existing
        if result.get("ok") is not True or type(request_number) is not int or request_number < 1:
            raise ValueError("Only successful, numbered forward results enter the bank")
        key = json_hash(identity)
        temporary = Path(tempfile.mkdtemp(prefix=".bank-", dir=self.root))
        try:
            files = []
            for relative, source in sorted(artifacts.items()):
                relative = _relative(relative)
                source = Path(source)
                if source.is_symlink() or not source.is_file():
                    raise BankError("Bank artifacts must be regular files")
                destination = temporary / "files" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                digest = file_hash(destination)
                if digest != file_hash(source):
                    raise BankError("Artifact changed while entering the bank")
                with destination.open("rb") as stream:
                    os.fsync(stream.fileno())
                files.append({"path": relative.as_posix(), "sha256": digest, "bytes": destination.stat().st_size})
            record = _copy({"schema_version": 1, "key": key, "identity": identity, "result": result,
                            "result_sha256": json_hash(result), "source_request": request_number, "artifacts": files})
            atomic_json(temporary / "entry.json", record)
            temporary.rename(self.root / key)
            descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return _copy(record)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def restore(self, entry, destination, *, paths):
        """Restore verified artifacts under caller-supplied safe relative names."""
        verified = self.get(entry["identity"])
        if verified is None or verified != entry:
            raise BankError("Forward bank entry changed before artifact restoration")
        expected = {item["path"] for item in entry["artifacts"]}
        if set(paths) != expected or len(set(paths.values())) != len(paths):
            raise BankError("Artifact mapping must cover every stored file exactly once")
        destination = Path(destination)
        for item in entry["artifacts"]:
            target = destination / _relative(paths[item["path"]])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.parent.resolve() != target.parent.absolute() or target.is_symlink():
                raise BankError("Artifact restoration must not traverse symbolic links")
            descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
            os.close(descriptor)
            try:
                shutil.copyfile(self.root / entry["key"] / "files" / item["path"], temporary)
                if file_hash(temporary) != item["sha256"]:
                    raise BankError("Artifact changed during restoration")
                with open(temporary, "rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
