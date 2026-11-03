"""JSONL supervisor for a persistent kernel inside an isolated sandbox.

The supervisor owns stdin/stdout and the run counter. Editor code runs in a
separate persistent process whose OS stdout is disconnected from the transport.
A wall-clock timeout kills that process, including native code, and starts a
fresh kernel. Only files in /work survive this restart.
"""

from __future__ import annotations

import argparse
import errno
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

_SECRET_NAMES = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GITHUB_TOKEN",
    "GITHUB_ACCESS_TOKEN",
    "GH_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_IDENTITY_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
}


def _assert_read_only(path: str) -> None:
    """Verify the mount rejects creation; never change existing model/practice files."""
    try:
        fd, probe = tempfile.mkstemp(prefix=".openwsp-readonly-probe-", dir=path)
    except OSError as exc:
        if exc.errno in {errno.EROFS, errno.EACCES, errno.EPERM}:
            return
        raise
    else:
        os.close(fd)
        os.unlink(probe)
        raise RuntimeError(f"Required read-only mount is writable: {path}")


def _child_main(connection, config: dict) -> None:
    # Native library diagnostics and direct os.write cannot corrupt supervisor JSONL.
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    os.dup2(null, 2)
    os.close(null)
    for name in _SECRET_NAMES:
        os.environ.pop(name, None)
    try:
        from openwsp.core import WeightStore
        from openwsp.kernel import Kernel

        if config["verify_read_only"]:
            _assert_read_only(
                str(Path(config["model"]).parent) if Path(config["model"]).is_file() else config["model"]
            )
            _assert_read_only(config["practice"])
        tokenizer = None
        if config["tokenizer"]:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config["tokenizer"], local_files_only=True, trust_remote_code=False
            )
        store = WeightStore(config["model"], Path(config["work"]) / "edited")
        kernel = Kernel(
            store,
            config["work"],
            config["practice"],
            tokenizer,
            max_runs=config["max_runs"],
            timeout=config["cell_timeout"],
            output_limit=10_000,
        )
        kernel.runs = config["initial_runs"]
        connection.send({"ok": True, "event": "ready", "runs": kernel.runs})
        while True:
            command = connection.recv()
            if command["op"] == "shutdown":
                return
            kernel.runs = command["runs"]
            result = kernel.run(command["code"])
            connection.send(result)
    except EOFError:
        return
    except BaseException as exc:  # noqa: BLE001 -- report worker failure to its supervisor.
        # No environment/credential values are included in diagnostics.
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "event": "fatal"})
    finally:
        connection.close()


class Supervisor:
    def __init__(self, config: dict):
        self.config = config
        self.runs = config["initial_runs"]
        self.restarts = 0
        self.process = None
        self.connection = None
        self._start()

    def _start(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        config = dict(self.config, initial_runs=self.runs)
        process = context.Process(target=_child_main, args=(child, config), daemon=True)
        process.start()
        child.close()
        self.process, self.connection = process, parent
        if not parent.poll(self.config["startup_timeout"]):
            self._stop()
            raise TimeoutError("Kernel initialization exceeded startup timeout")
        try:
            result = parent.recv()
        except EOFError as exc:
            self._stop()
            raise RuntimeError("Kernel process exited during initialization") from exc
        if not result.get("ok"):
            self._stop()
            raise RuntimeError(result.get("error", "Kernel initialization failed"))

    def _stop(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.kill()
            self.process.join(timeout=5)
            if self.process.is_alive():
                raise RuntimeError("Kernel process did not terminate")
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def restart(self):
        self._stop()
        self._start()
        self.restarts += 1

    def handle(self, request: dict) -> dict:
        identifier = request.get("id")
        op = request.get("op")
        common = {"id": identifier, "runs": self.runs, "restarted": False, "variables_lost": False}
        if op == "status":
            return dict(common, ok=True, restarts=self.restarts, alive=self.process.is_alive())
        if op == "restart":
            self.restart()
            return dict(common, ok=True, restarted=True, variables_lost=True, restarts=self.restarts)
        if op == "shutdown":
            self._stop()
            return dict(common, ok=True)
        if op != "run" or not isinstance(request.get("code"), str):
            return dict(
                common, ok=False, error="Expected op='run' with a code string, status, restart or shutdown"
            )
        if self.runs >= self.config["max_runs"]:
            return dict(common, ok=False, output="", error="Code-run budget exhausted", seconds=0.0)
        started = time.monotonic()
        try:
            self.connection.send({"op": "run", "code": request["code"], "runs": self.runs})
            if not self.connection.poll(self.config["cell_timeout"]):
                raise TimeoutError("Kernel execution exceeded its wall-clock limit")
            result = self.connection.recv()
            if result.get("event") == "fatal":
                raise RuntimeError(result.get("error", "Kernel process failed"))
        except (EOFError, BrokenPipeError, OSError, TimeoutError, RuntimeError) as exc:
            self.runs += 1
            self.restart()
            return {
                "id": identifier,
                "ok": False,
                "output": "",
                "error": f"{type(exc).__name__}: {exc}. Kernel restarted; variables were lost; /work files remain.",
                "seconds": time.monotonic() - started,
                "runs": self.runs,
                "restarted": True,
                "variables_lost": True,
            }
        self.runs = result["runs"]
        # The in-process deadline can fire just before the supervisor timeout;
        # both timeout paths have identical restart semantics.
        restarted = "KernelTimeout:" in (result.get("error") or "")
        if restarted:
            self.restart()
            result["error"] += ". Kernel restarted; variables were lost; /work files remain."
        return dict(result, id=identifier, restarted=restarted, variables_lost=restarted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/model")
    parser.add_argument("--practice", default="/practice")
    parser.add_argument("--work", default="/work")
    parser.add_argument("--tokenizer")
    parser.add_argument("--cell-timeout", type=float, default=900)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--max-runs", type=int, default=300)
    parser.add_argument("--initial-runs", type=int, default=0)
    parser.add_argument("--verify-read-only", action="store_true")
    config = vars(parser.parse_args())
    if config["cell_timeout"] <= 0 or not 0 <= config["initial_runs"] <= config["max_runs"]:
        parser.error("Timeout must be positive and run counters must be within budget")
    for name in _SECRET_NAMES:
        os.environ.pop(name, None)
    supervisor = Supervisor(config)
    print(json.dumps({"type": "ready", "ok": True, "runs": supervisor.runs}), flush=True)
    try:
        while line := sys.stdin.readline(1_048_577):
            request = None
            try:
                if len(line) > 1_048_576:
                    while not line.endswith("\n"):
                        line = sys.stdin.readline(1_048_577)
                        if not line:
                            break
                    raise ValueError("JSONL request exceeds 1 MiB")
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise TypeError("Request must be a JSON object")
                result = supervisor.handle(request)
            except (ValueError, KeyError, TypeError) as exc:
                result = {
                    "id": None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "runs": supervisor.runs,
                }
            print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
            if isinstance(request, dict) and request.get("op") == "shutdown":
                break
    finally:
        supervisor._stop()


if __name__ == "__main__":
    main()
