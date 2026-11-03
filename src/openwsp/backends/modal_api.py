"""Built-in, budgeted Modal transport for the public SDK, CLI and MCP API.

Construction never contacts Modal. A configured and explicitly enabled backend
dispatches only on forward(). Its entry point accepts the v1 envelope documented
in docs/architecture.md; model execution remains the deployment's responsibility.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from decimal import Decimal
from pathlib import Path, PurePosixPath

from ..budget import BudgetExceeded, BudgetLedger, compute_cost
from ..edits import FORMAT, FORMAT_VERSION
from .jobs import JobBlocked, JobPending, RemoteJobRunner

MAX_ARTIFACT_BYTES = 6 * 1024**3
MAX_EDIT_BYTES = 1024**3
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SECRET = re.compile(r"(?:sk-|hf_|ak-|as-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


class ModalForwardError(RuntimeError):
    """A forward request or provider response failed the public transport contract."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_copy(value):
    return json.loads(_canonical(value))


def _sanitize(value):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z]", "", key.lower())
            secret = normalized in {
                "token", "accesstoken", "refreshtoken", "authorization", "password", "credentials",
                "modaltokenid", "modaltokensecret", "hftoken", "secret", "secretkey",
            } or "secret" in normalized or normalized.endswith(("apikey", "apitoken", "password", "credentials"))
            output[key] = "[REDACTED]" if secret else _sanitize(item)
        return output
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _BEARER.sub(r"\1[REDACTED]", _SECRET.sub("[REDACTED]", value))
    return value


def _limits(value, *, child=False):
    keys = {"timeout_seconds", "startup_seconds", "scaledown_seconds", "gpu", "cpu_cores", "memory_gib"}
    if not isinstance(value, dict) or set(value) - {"children"} != keys or (child and "children" in value):
        raise ValueError("limits must specify timeout, startup, scaledown seconds, gpu, cpu_cores and memory_gib")
    result = dict(value)
    for name in keys - {"gpu"}:
        number = result[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError("Resource and time limits must be finite numbers")
        if number < 0 or (name in {"timeout_seconds", "cpu_cores", "memory_gib"} and number == 0):
            raise ValueError("Timeout and resources must be positive; margins must be nonnegative")
    if result["timeout_seconds"] > 86400 or max(result["startup_seconds"], result["scaledown_seconds"]) > 3600:
        raise ValueError("Runtime bounds exceed the supported provider limits")
    compute_cost(0, gpu=result["gpu"], cpu_cores=result["cpu_cores"], memory_gib=result["memory_gib"])
    if "children" in result:
        if not isinstance(result["children"], list) or len(result["children"]) > 16:
            raise ValueError("A gateway may declare at most sixteen bounded child workers")
        result["children"] = [_limits(item, child=True) for item in result["children"]]
    return result


def reservation_bound(limits):
    """Conservative full-runtime bound for a worker or a gateway and its children."""
    limits = _limits(limits)
    total = Decimal(0)
    for item in [limits, *limits.get("children", [])]:
        total += compute_cost(sum(item[key] for key in ("timeout_seconds", "startup_seconds", "scaledown_seconds")),
                              gpu=item["gpu"], cpu_cores=item["cpu_cores"], memory_gib=item["memory_gib"])
    return total


def _base(model, revision):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A pinned base model ID is required")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("base_revision must be an immutable 40- or 64-character hexadecimal revision")
    return {"model": model, "revision": revision}


def apply_payload_edit(payload, store, *, base_model, base_revision, limits=None, max_edit_bytes=MAX_EDIT_BYTES):
    """Validate a v1 transport and replace the store's complete edit snapshot.

    The deployment must load its base from this immutable revision and pass its
    actual provider-enforced resource/time bounds as limits. Tensor hashes, shapes,
    dtype and edit restrictions are checked again by WeightStore.apply_edit.
    Empty edits reset any previous edits in a reused store.
    """
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported Modal forward envelope")
    if payload.get("base") != _base(base_model, base_revision):
        raise ValueError("Forward base model or pinned revision does not match this deployment")
    if not isinstance(payload.get("request_id"), str) or not _HEX.fullmatch(payload["request_id"]):
        raise ValueError("Invalid forward request ID")
    if not isinstance(payload.get("request"), dict):
        raise TypeError("Forward request must be a JSON object")
    _json_copy(payload["request"])
    supplied_limits = _limits(payload.get("limits"))
    if limits is not None and supplied_limits != _limits(limits):
        raise ValueError("Requested resource bounds do not match the deployed function")
    edit = payload.get("edit")
    if not isinstance(edit, dict) or set(edit) != {"manifest_json", "edit_npz"}:
        raise ValueError("Forward edit must contain the exact v1 manifest and NPZ bytes")
    if not all(isinstance(edit[key], bytes) for key in edit):
        raise ValueError("Forward edit files must be bytes")
    if len(edit["manifest_json"]) > 8 * 1024**2 or len(edit["edit_npz"]) > max_edit_bytes:
        raise ValueError("Forward edit exceeds transport size limits")
    manifest = json.loads(edit["manifest_json"])
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION
            or manifest.get("base_model") != base_model or manifest.get("base_revision") != base_revision):
        raise ValueError("Forward edit is not v1 or does not identify the pinned base")
    with tempfile.TemporaryDirectory(prefix="openwsp-received-edit-") as directory:
        path = Path(directory)
        (path / "manifest.json").write_bytes(edit["manifest_json"])
        (path / "edit.npz").write_bytes(edit["edit_npz"])
        try:
            with zipfile.ZipFile(path / "edit.npz") as archive:
                if sum(item.file_size for item in archive.infolist()) > max_edit_bytes:
                    raise ValueError("Expanded edit exceeds transport size limits")
        except zipfile.BadZipFile:
            raise ValueError("Invalid NPZ edit archive") from None
        return store.apply_edit(path)


def make_response(payload, result, *, container_seconds, cost_usd=None, artifacts=None):
    """Build a response after apply_payload_edit succeeded and execution finished.

    container_seconds must include the whole billed GPU invocation. A gateway
    with additional workers must also supply their total known cost_usd.
    """
    response = {
        "schema_version": 1, "base": payload["base"], "request_id": payload["request_id"],
        "edit_sha256": json.loads(payload["edit"]["manifest_json"])["edit_sha256"],
        "limits": payload["limits"], "result": result, "container_seconds": container_seconds,
        "artifacts": artifacts or [],
    }
    if cost_usd is not None:
        response["cost_usd"] = str(cost_usd)
    return response


class _ResponseCall:
    def __init__(self, call, expected):
        self.call, self.expected = call, expected

    def get(self, *, timeout):
        response = self.call.get(timeout=timeout)
        if not isinstance(response, dict) or type(response.get("schema_version")) is not int or response["schema_version"] != 1:
            raise ModalForwardError("Unsupported Modal forward response")
        for key in ("base", "request_id", "edit_sha256", "limits"):
            if response.get(key) != self.expected[key]:
                raise ModalForwardError("Response identity or resource bounds do not match the request")
        if not isinstance(response.get("result"), dict) or not isinstance(response.get("artifacts", []), list):
            raise ModalForwardError("Response result and artifact descriptors have invalid types")
        if self.expected["limits"].get("children") and "cost_usd" not in response:
            raise ModalForwardError("Gateway response must account for its child workers' total cost")
        seconds = response.get("container_seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ModalForwardError("Response must include finite nonnegative container_seconds")
        # Do not persist arbitrary top-level server fields, credentials, or debug
        # environment data. Duration fields are preserved for conservative cost.
        clean = {key: response[key] for key in self.expected}
        clean.update(schema_version=1, result=_sanitize(_json_copy(response["result"])),
                     container_seconds=seconds, artifacts=_sanitize(_json_copy(response.get("artifacts", []))))
        for key in ("cost_usd", "seconds", "gpu_seconds", "billable_seconds"):
            if key in response:
                clean[key] = response[key]
        _canonical(clean)
        return clean


def _function(app_name, function_name):
    import modal

    from .modal_runtime import configure_tls

    configure_tls()
    return modal.Function.from_name(app_name, function_name)


def _call(call_id):
    import modal

    from .modal_runtime import configure_tls

    configure_tls()
    return modal.FunctionCall.from_id(call_id)


def _volume(name):
    import modal

    from .modal_runtime import configure_tls

    configure_tls()
    return modal.Volume.from_name(name, create_if_missing=False)


class ModalForwardBackend:
    def __init__(self, *, app_name, function_name, base_model, base_revision, work_dir, limits,
                 budget_usd=42, enabled=False, artifact_volumes=(), max_artifact_bytes=MAX_ARTIFACT_BYTES,
                 function_factory=None, call_factory=None, volume_factory=None):
        self.base, self.limits = _base(base_model, base_revision), _limits(limits)
        if not all(isinstance(name, str) and name.strip() for name in (app_name, function_name)):
            raise ValueError("Modal app and entry function names are required")
        budget = Decimal(str(budget_usd))
        if not budget.is_finite() or not 0 < budget <= 150:
            raise ValueError("The Modal task budget must be positive and at most $150")
        if type(enabled) is not bool:
            raise ValueError("enabled must be an explicit boolean")
        if (type(max_artifact_bytes) is not int or not 0 <= max_artifact_bytes <= MAX_ARTIFACT_BYTES):
            raise ValueError("Artifact downloads must be bounded to at most 6 GiB")
        if isinstance(artifact_volumes, str) or not all(isinstance(name, str) and name for name in artifact_volumes):
            raise ValueError("artifact_volumes must contain allowed volume names")
        self.app_name, self.function_name, self.budget_usd = app_name, function_name, budget
        self.work_dir, self.enabled = Path(work_dir).resolve(), enabled
        self.artifact_volumes, self.max_artifact_bytes = frozenset(artifact_volumes), max_artifact_bytes
        self.function_factory, self.call_factory = function_factory or _function, call_factory or _call
        self.volume_factory = volume_factory or _volume

    def forward(self, request, store):
        if not self.enabled:
            raise ModalForwardError("Modal forward is disabled; explicitly enable the configured backend")
        if not isinstance(request, dict):
            raise TypeError("Forward request must be a JSON object")
        request = _json_copy(request)
        user_id = request.pop("request_id", None)
        if user_id is not None and (not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 256):
            raise ValueError("request_id must be a nonempty string of at most 256 characters")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".forward-edit-", dir=self.work_dir) as directory:
            manifest = store.export_edit(directory, base_model=self.base["model"], revision=self.base["revision"])
            edit = {"manifest_json": (Path(directory) / "manifest.json").read_bytes(),
                    "edit_npz": (Path(directory) / "edit.npz").read_bytes()}
        if len(edit["edit_npz"]) > MAX_EDIT_BYTES:
            raise ValueError("Exported edit exceeds transport size limits")
        fingerprint = {"app": self.app_name, "function": self.function_name, "base": self.base,
                       "request": request, "edit_sha256": manifest["edit_sha256"], "limits": self.limits}
        identity = "user:" + user_id if user_id is not None else "content:" + _canonical(fingerprint)
        request_id = hashlib.sha256(identity.encode()).hexdigest()
        key = "public-forward/" + request_id
        payload = {"schema_version": 1, "base": self.base, "request_id": request_id,
                   "request": request, "edit": edit, "limits": self.limits}
        expected = {"base": self.base, "request_id": request_id, "limits": self.limits,
                    "edit_sha256": manifest["edit_sha256"]}
        ledger = BudgetLedger(self.work_dir / "budget.json", modal_task_limit=self.budget_usd,
                              modal_cycle_limit=min(self.budget_usd, Decimal(42)), modal_prior_usage=0)
        if Decimal(ledger.snapshot()["limits"]["modal"]) > self.budget_usd:
            raise ValueError("Existing ledger limit exceeds this configured task budget")
        runner = RemoteJobRunner(ledger, self.work_dir / "jobs",
                                 call_factory=lambda call_id: _ResponseCall(self.call_factory(call_id), expected))
        resources = {key: value for key, value in self.limits.items() if key not in {"timeout_seconds", "children"}}
        bound = reservation_bound(self.limits)
        try:
            call_id = runner.launch(
                key, lambda: self.function_factory(self.app_name, self.function_name).spawn(payload),
                upper_bound_usd=bound, cost_options=resources, fingerprint=fingerprint,
            )
            response = runner.result(key, timeout=self.limits["timeout_seconds"] + self.limits["startup_seconds"])
            result = _sanitize(response["result"])
        except (JobBlocked, JobPending) as error:
            raise type(error)(str(error)) from None
        except (BudgetExceeded, ValueError):
            raise
        except Exception as error:  # noqa: BLE001 — never expose provider exception text containing credentials
            raise ModalForwardError("Modal forward failed: " + type(error).__name__) from None
        if response.get("artifacts"):
            try:
                downloaded = self._download_artifacts(request_id, response["artifacts"])
            except ModalForwardError:
                raise
            except Exception as error:  # noqa: BLE001 — volume errors can also contain authentication details
                raise ModalForwardError("Artifact download failed: " + type(error).__name__) from None
            result["artifacts"] = downloaded
            result["files"] = {entry["name"]: entry["local_path"] for entry in downloaded}
        result["_openwsp"] = {**expected, "call_id": call_id}
        return result

    def _download_artifacts(self, request_id, descriptors):
        seen, checked, total = set(), [], 0
        prefix = "public-forward/" + request_id + "/"
        for item in descriptors:
            if not isinstance(item, dict) or set(item) != {"volume", "path", "name", "sha256", "size_bytes"}:
                raise ModalForwardError("Invalid artifact descriptor")
            name, path, size, digest = item["name"], item["path"], item["size_bytes"], item["sha256"]
            if item["volume"] not in self.artifact_volumes:
                raise ModalForwardError("Artifact volume is not configured for this backend")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) or name in seen:
                raise ModalForwardError("Artifact name is unsafe or duplicated")
            if (not isinstance(path, str) or not path.startswith(prefix) or "\\" in path
                    or ".." in PurePosixPath(path).parts or path != str(PurePosixPath(path))):
                raise ModalForwardError("Artifact path is outside the current request")
            if type(size) is not int or size < 0 or not isinstance(digest, str) or not _HEX.fullmatch(digest):
                raise ModalForwardError("Invalid artifact size or sha256")
            total += size
            if total > self.max_artifact_bytes:
                raise ModalForwardError("Artifact response exceeds the configured download bound")
            checked.append(dict(item))
            seen.add(name)
        directory = self.work_dir / "forward" / request_id
        for part in (directory.parent, directory):
            if part.is_symlink():
                raise ModalForwardError("Artifact destination cannot contain symlinks")
            part.mkdir(exist_ok=True)
        volumes, listings = {}, {}
        for item in checked:
            destination = directory / item["name"]
            if destination.is_symlink():
                raise ModalForwardError("Artifact destination cannot be a symlink")
            if destination.exists() and self._matches(destination, item):
                item["local_path"] = str(destination)
                continue
            if item["volume"] not in volumes:
                volumes[item["volume"]] = self.volume_factory(item["volume"])
            volume = volumes[item["volume"]]
            parts = PurePosixPath(item["path"]).parts
            for index in range(len(parts)):
                parent, current = "/".join(parts[:index]) or "/", "/".join(parts[:index + 1])
                listing_key = (item["volume"], parent)
                if listing_key not in listings:
                    listings[listing_key] = list(volume.listdir(parent, recursive=False))
                entries = [entry for entry in listings[listing_key] if entry.path.strip("/") == current]
                expected_type = 1 if index == len(parts) - 1 else 2  # Modal FileEntryType FILE / DIRECTORY
                if len(entries) != 1 or entries[0].type != expected_type:
                    raise ModalForwardError("Artifact path contains a symlink or a nonregular entry")
                if expected_type == 1 and entries[0].size != item["size_bytes"]:
                    raise ModalForwardError("Artifact size changed before download")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=".download-", delete=False) as output:
                    temporary = Path(output.name)
                    size, digest = 0, hashlib.sha256()
                    for chunk in volume.read_file(item["path"]):
                        if not isinstance(chunk, bytes):
                            raise ModalForwardError("Artifact transport returned non-byte content")
                        size += len(chunk)
                        if size > item["size_bytes"]:
                            raise ModalForwardError("Artifact exceeds its declared size")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if size != item["size_bytes"] or digest.hexdigest() != item["sha256"]:
                    raise ModalForwardError("Artifact hash or size verification failed")
                os.replace(temporary, destination)
                item["local_path"] = str(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return checked

    @staticmethod
    def _matches(path, descriptor):
        if not path.is_file() or path.stat().st_size != descriptor["size_bytes"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                digest.update(chunk)
        return digest.hexdigest() == descriptor["sha256"]


def from_env():
    """Trusted zero-argument factory: --backend openwsp.backends.modal_api:from_env."""
    prefix = "OPENWSP_MODAL_"
    required = ("APP", "FUNCTION", "BASE_MODEL", "BASE_REVISION", "WORK_DIR", "BUDGET_USD", "LIMITS")
    if any(not os.environ.get(prefix + key) for key in required):
        raise ValueError("Configure OPENWSP_MODAL_" + ", OPENWSP_MODAL_".join(required))
    return ModalForwardBackend(
        app_name=os.environ[prefix + "APP"], function_name=os.environ[prefix + "FUNCTION"],
        base_model=os.environ[prefix + "BASE_MODEL"], base_revision=os.environ[prefix + "BASE_REVISION"],
        work_dir=os.environ[prefix + "WORK_DIR"], budget_usd=os.environ[prefix + "BUDGET_USD"],
        limits=json.loads(os.environ[prefix + "LIMITS"]), enabled=os.environ.get(prefix + "ENABLED") == "1",
        artifact_volumes=json.loads(os.environ.get(prefix + "ARTIFACT_VOLUMES", "[]")),
        max_artifact_bytes=int(os.environ.get(prefix + "MAX_ARTIFACT_BYTES", str(MAX_ARTIFACT_BYTES))),
    )
