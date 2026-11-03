"""Exercise the real JSONL supervisor and its isolated child process on CPU."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


class WorkerClient:
    def __init__(self, root: Path, *, cell_timeout=2.0, max_runs=300, initial_runs=0):
        self.root = root
        self.root.mkdir(parents=True)
        model = root / "model.safetensors"
        save_file({"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, model)
        practice = root / "practice"
        practice.mkdir()
        (practice / "tiny.jsonl").write_text('{"id":"practice_1","problem":"1+1","answer":"2"}\n')
        argv = [
            sys.executable,
            "-m",
            "openwsp.backends.sandbox_worker",
            "--model",
            str(model),
            "--work",
            str(root / "work"),
            "--practice",
            str(practice),
            "--cell-timeout",
            str(cell_timeout),
            "--startup-timeout",
            "30",
            "--max-runs",
            str(max_runs),
            "--initial-runs",
            str(initial_runs),
        ]
        # In particular, never inherit Modal, OpenAI, HF, or GitHub credentials
        # into the kernel integration-test process.
        environment = {
            "PATH": os.defpath,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        self.stderr = tempfile.TemporaryFile()  # noqa: SIM115 -- lifetime is managed by close().
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            bufsize=0,
            env=environment,
            cwd=root,
            start_new_session=True,
        )
        self.buffer = b""
        self.counter = 0
        try:
            self.ready = self.read(timeout=35)
        except BaseException:
            self.close()
            raise

    def _stderr_tail(self):
        self.stderr.seek(0)
        return self.stderr.read().decode(errors="replace")[-3000:]

    def read(self, timeout=15):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    pytest.fail(f"Worker response timed out; stderr: {self._stderr_tail()}")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    pytest.fail(f"Worker closed stdout; stderr: {self._stderr_tail()}")
                self.buffer += chunk
            line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            return json.loads(line)
        except ValueError:
            pytest.fail(f"Worker polluted its JSONL stdout with {line!r}")

    def raw(self, line):
        self.process.stdin.write(line.encode() + b"\n")
        self.process.stdin.flush()
        return self.read()

    def request(self, op, **fields):
        self.counter += 1
        request_id = f"request-{self.counter}"
        response = self.raw(json.dumps({"id": request_id, "op": op, **fields}))
        assert response["id"] == request_id
        return response

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.request("shutdown")
                    self.process.wait(timeout=5)
                except (AssertionError, OSError, subprocess.TimeoutExpired, pytest.fail.Exception):
                    pass
        finally:
            # Ensure an interrupted test cannot leave the kernel child running.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
            self.stderr.close()


@pytest.fixture
def worker_factory(tmp_path):
    workers = []

    def create(**kwargs):
        worker = WorkerClient(tmp_path / f"worker-{len(workers)}", **kwargs)
        workers.append(worker)
        return worker

    yield create
    for worker in workers:
        worker.close()


def test_namespace_persists_and_edits_survive_explicit_restart(worker_factory):
    worker = worker_factory()
    assert worker.ready["type"] == "ready"
    assert worker.ready["ok"] is True
    assert worker.ready["runs"] == 0
    first = worker.request("run", code="counter = 40\ncounter + 2")
    assert first["ok"] is True
    assert first["output"].strip() == "42"
    saved = worker.request(
        "run",
        code="w = load('layer.weight')\nw[0, 0] = 1.25\nsave_edit('layer.weight', w)\nprint(counter)",
    )
    assert saved["ok"] is True
    assert saved["output"].strip() == "40"
    assert saved["runs"] == 2
    restarted = worker.request("restart")
    assert restarted["ok"] is True
    assert restarted["restarted"] is True
    assert restarted["variables_lost"] is True
    assert restarted["runs"] == 2
    recovered = worker.request(
        "run", code="print('counter' in globals())\nprint(float(load('layer.weight')[0, 0]))\nprint(edited())"
    )
    assert recovered["ok"] is True
    assert recovered["output"].splitlines() == ["False", "1.25", "['layer.weight']"]
    status = worker.request("status")
    assert status["runs"] == 3
    assert status["restarts"] == 1


def test_native_call_is_killed_and_kernel_recovers(worker_factory):
    worker = worker_factory(cell_timeout=0.3)
    before = worker.request("run", code="sentinel = 'before timeout'")
    assert before["ok"] is True
    timed_out = worker.request(
        "run",
        code=(
            "import ctypes, signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nctypes.CDLL(None).sleep(3)"
        ),
    )
    assert timed_out["ok"] is False
    assert timed_out["restarted"] is True
    assert timed_out["variables_lost"] is True
    assert timed_out["runs"] == 2
    assert "timeout" in timed_out["error"].lower() or "exceed" in timed_out["error"].lower()
    recovered = worker.request("run", code="print('sentinel' in globals())\n6 * 7")
    assert recovered["ok"] is True
    assert recovered["output"].splitlines() == ["False", "42"]
    assert recovered["runs"] == 3


def test_native_stdout_cannot_corrupt_supervisor_protocol(worker_factory):
    worker = worker_factory()
    response = worker.request("run", code="import os\nos.write(1, b'bad\\n')\nprint('visible')")
    assert response["ok"] is True
    assert "visible" in response["output"]
    status = worker.request("status")
    assert status["runs"] == 1


def test_protocol_refusal_does_not_consume_run(worker_factory):
    worker = worker_factory()
    refused = worker.request("run", code="import socket")
    assert refused["ok"] is False
    assert refused["runs"] == 0
    assert refused["error"]
    allowed = worker.request("run", code="1 + 1")
    assert allowed["ok"] is True
    assert allowed["runs"] == 1


def test_run_limit_survives_resumed_worker_and_restart(worker_factory):
    worker = worker_factory(max_runs=3, initial_runs=2)
    assert worker.ready["runs"] == 2
    assert worker.request("run", code="2 + 2")["runs"] == 3
    assert worker.request("restart")["runs"] == 3
    exhausted = worker.request("run", code="99")
    assert exhausted["ok"] is False
    assert exhausted["runs"] == 3
    assert "run" in exhausted["error"].lower() or "budget" in exhausted["error"].lower()


def test_invalid_messages_do_not_crash_or_consume_budget(worker_factory):
    worker = worker_factory()
    malformed = worker.raw("{not-json")
    assert malformed["ok"] is False
    assert malformed["error"]
    unknown = worker.request("unsupported")
    assert unknown["ok"] is False
    assert unknown["error"]
    assert worker.request("status")["runs"] == 0
    assert worker.request("run", code="'still alive'")["ok"] is True


def test_shutdown_acknowledges_then_exits(worker_factory):
    worker = worker_factory()
    response = worker.request("shutdown")
    assert response["ok"] is True
    assert worker.process.wait(timeout=5) == 0
