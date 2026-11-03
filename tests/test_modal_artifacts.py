"""Verified public artifact transport; all provider reads are local fakes."""

import hashlib
from types import SimpleNamespace

import pytest

from openwsp.api import OpenWSP
from openwsp.backends.modal_api import ModalForwardBackend, ModalForwardError, make_response

REQUEST_ID = "a" * 64
BODY = b"binary tensor artifact"


class Volume:
    def __init__(self):
        self.calls = []
        self.body = BODY
        self.types = {"public-forward": 2, "public-forward/" + REQUEST_ID: 2,
                      f"public-forward/{REQUEST_ID}/values.npy": 1}

    def listdir(self, path, *, recursive):
        assert not recursive
        self.calls.append(("list", path))
        parent = path.strip("/")
        return [SimpleNamespace(path=name, type=kind, size=len(BODY) if kind == 1 else 0)
                for name, kind in self.types.items() if name.rpartition("/")[0] == parent]

    def read_file(self, path):
        self.calls.append(("read", path))
        yield self.body[:5]
        yield self.body[5:]


@pytest.fixture
def artifacts(tmp_path):
    volume = Volume()
    backend = ModalForwardBackend(
        app_name="test", function_name="forward", base_model="test/tiny", base_revision="b" * 40,
        work_dir=tmp_path, limits={"gpu": None, "cpu_cores": 1, "memory_gib": 4,
                                  "timeout_seconds": 10, "startup_seconds": 1, "scaledown_seconds": 1},
        artifact_volumes=["test-volume"], volume_factory=lambda name: volume,
    )
    descriptor = {"volume": "test-volume", "path": f"public-forward/{REQUEST_ID}/values.npy",
                  "name": "values.npy", "sha256": hashlib.sha256(BODY).hexdigest(), "size_bytes": len(BODY)}
    return backend, volume, descriptor


def test_artifacts_download_verified_bytes_and_reuse_verified_local_file(artifacts):
    backend, volume, descriptor = artifacts
    result = backend._download_artifacts(REQUEST_ID, [descriptor])
    destination = backend.work_dir / "forward" / REQUEST_ID / "values.npy"
    assert destination.read_bytes() == BODY
    assert result[0]["local_path"] == str(destination)
    calls = list(volume.calls)
    assert backend._download_artifacts(REQUEST_ID, [descriptor]) == result
    assert volume.calls == calls


@pytest.mark.parametrize("change", [
    {"path": "public-forward/" + "b" * 64 + "/values.npy"},
    {"path": f"public-forward/{REQUEST_ID}/../values.npy"},
    {"path": f"public-forward/{REQUEST_ID}/./values.npy"},
    {"name": "../values.npy"},
    {"volume": "another-volume"},
    {"size_bytes": -1},
    {"size_bytes": 6 * 1024**3 + 1},
])
def test_unsafe_artifacts_refused_before_provider_reads(artifacts, change):
    backend, volume, descriptor = artifacts
    descriptor.update(change)
    with pytest.raises(ModalForwardError):
        backend._download_artifacts(REQUEST_ID, [descriptor])
    assert not volume.calls


@pytest.mark.parametrize("name", ["public-forward", "public-forward/" + REQUEST_ID,
                                 f"public-forward/{REQUEST_ID}/values.npy"])
def test_remote_symlink_at_any_path_component_is_refused(artifacts, name):
    backend, volume, descriptor = artifacts
    volume.types[name] = 3
    with pytest.raises(ModalForwardError, match="symlink"):
        backend._download_artifacts(REQUEST_ID, [descriptor])
    assert not any(action == "read" for action, _ in volume.calls)


def test_local_symlink_is_refused_without_provider_reads(artifacts, tmp_path):
    backend, volume, descriptor = artifacts
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (backend.work_dir / "forward").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ModalForwardError, match="symlink"):
        backend._download_artifacts(REQUEST_ID, [descriptor])
    assert not volume.calls
    assert not list(outside.iterdir())


@pytest.mark.parametrize("body", [b"too short", BODY + b"extra", b"x" * len(BODY)])
def test_failed_hash_or_size_does_not_publish_an_artifact(artifacts, body):
    backend, volume, descriptor = artifacts
    volume.body = body
    with pytest.raises(ModalForwardError):
        backend._download_artifacts(REQUEST_ID, [descriptor])
    directory = backend.work_dir / "forward" / REQUEST_ID
    assert not list(directory.iterdir())


def test_corrupted_cached_artifact_is_downloaded_again(artifacts):
    backend, volume, descriptor = artifacts
    backend._download_artifacts(REQUEST_ID, [descriptor])
    (backend.work_dir / "forward" / REQUEST_ID / "values.npy").write_bytes(b"corrupt")
    backend._download_artifacts(REQUEST_ID, [descriptor])
    assert sum(action == "read" for action, _ in volume.calls) == 2


def test_retry_after_failed_download_reuses_paid_result_without_dispatch(artifacts, tmp_path):
    backend, volume, descriptor = artifacts
    backend.enabled = True
    calls = []

    def spawn(payload):
        request_id = payload["request_id"]
        item = {**descriptor, "path": f"public-forward/{request_id}/values.npy"}
        volume.types = {name.replace(REQUEST_ID, request_id): kind for name, kind in volume.types.items()}
        response = make_response(payload, {"ok": True}, container_seconds=1, artifacts=[item])
        call = SimpleNamespace(object_id="fc-artifact", get=lambda **kwargs: response)
        calls.append(call)
        return call

    backend.function_factory = lambda *args: SimpleNamespace(spawn=spawn)
    backend.call_factory = lambda call_id: calls[0]
    api = OpenWSP.tiny(tmp_path / "tiny")
    volume.body = b"corrupt"
    with pytest.raises(ModalForwardError):
        backend.forward({"type": "teacher_forced"}, api.store)
    volume.body = BODY
    result = backend.forward({"type": "teacher_forced"}, api.store)
    assert len(calls) == 1
    assert result["files"]["values.npy"] == result["artifacts"][0]["local_path"]
    assert result["artifacts"][0]["sha256"] == hashlib.sha256(BODY).hexdigest()
