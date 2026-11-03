"""JSONL supervisor for a persistent kernel inside an isolated sandbox.

The supervisor owns stdin/stdout and the run counter. Editor code runs in a
separate persistent process whose OS stdout is disconnected from the transport.
A wall-clock timeout kills that process, including native code, and starts a
fresh kernel. Only files in /work survive this restart.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import resource
import sys
import tempfile
import threading
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


def _send_frame(connection, value):
    connection.send_bytes(json.dumps(value, ensure_ascii=True, allow_nan=False).encode())


def _receive_frame(connection):
    value = json.loads(connection.recv_bytes(maxlength=1_048_576))
    if not isinstance(value, dict):
        raise TypeError("Kernel frame must be a JSON object")
    return value


def _cgroup_usage(expected_limit_bytes: int) -> dict:
    """Read a container cgroup, with explicit unknowns instead of RSS guesses."""
    unknown = {
        "cgroup_memory_current_bytes": None,
        "cgroup_memory_peak_bytes": None,
        "cgroup_memory_limit_bytes": None,
        "cgroup_cpu_seconds": None,
        "cgroup_memory_scope_verified": False,
        "cgroup_version": None,
        "cgroup_path": None,
    }
    candidates = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, membership = line.split(":", 2)
            relative = Path(membership.lstrip("/"))
            if ".." in relative.parts:
                continue
            if not controllers:
                candidates.append((2, Path("/sys/fs/cgroup") / relative))
            elif "memory" in controllers.split(","):
                candidates.append((1, Path("/sys/fs/cgroup/memory") / relative))
    except (OSError, ValueError):
        pass
    candidates.extend([(2, Path("/sys/fs/cgroup")), (1, Path("/sys/fs/cgroup/memory"))])
    for version, directory in candidates:

        def counter(name, directory=directory):
            try:
                return int((directory / name).read_text().strip())
            except (OSError, ValueError):
                return None

        current = counter("memory.current" if version == 2 else "memory.usage_in_bytes")
        peak = counter("memory.peak" if version == 2 else "memory.max_usage_in_bytes")
        limit = counter("memory.max" if version == 2 else "memory.limit_in_bytes")
        if current is None and peak is None:
            continue
        cpu = None
        if version == 2:
            try:
                for line in (directory / "cpu.stat").read_text().splitlines():
                    key, value = line.split()
                    if key == "usage_usec":
                        cpu = int(value) / 1_000_000
                        break
            except (OSError, ValueError):
                pass
        verified = False
        try:
            members = {int(value) for value in (directory / "cgroup.procs").read_text().split()}
            read_only = bool(os.statvfs(directory).f_flag & os.ST_RDONLY)
            verified = os.getpid() in members and limit == expected_limit_bytes and read_only
        except (OSError, ValueError):
            pass
        return {
            "cgroup_memory_current_bytes": current,
            "cgroup_memory_peak_bytes": peak,
            "cgroup_memory_limit_bytes": limit,
            "cgroup_cpu_seconds": cpu,
            "cgroup_memory_scope_verified": verified,
            "cgroup_version": version,
            "cgroup_path": str(directory),
        }
    return unknown


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
        _send_frame(connection, {"ok": True, "event": "ready", "runs": kernel.runs})
        while True:
            command = _receive_frame(connection)
            if command["op"] == "shutdown":
                return
            kernel.runs = command["runs"]
            result = kernel.run(command["code"])
            _send_frame(connection, result)
    except EOFError:
        return
    except BaseException as exc:  # noqa: BLE001 -- report worker failure to its supervisor.
        # No environment/credential values are included in diagnostics.
        _send_frame(
            connection, {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:10_000], "event": "fatal"}
        )
    finally:
        connection.close()


class Supervisor:
    def __init__(self, config: dict):
        self.started_at = time.monotonic()
        from openwsp.kernel import validate_code

        self.config = config
        self._validate_code = validate_code
        self._completed: dict[str, tuple[str, dict]] = {}
        self.runs = config["initial_runs"]
        self.restarts = 0
        self.process = None
        self.connection = None
        self._start()

    def usage(self) -> dict:
        """Read OS counters outside the editor process; RSS is not a billing total."""
        own = resource.getrusage(resource.RUSAGE_SELF)
        exited = resource.getrusage(resource.RUSAGE_CHILDREN)
        kernel_peak = None
        live_cpu = 0.0
        try:
            if self.process is not None and self.process.is_alive():
                status = Path(f"/proc/{self.process.pid}/status").read_text()
                for line in status.splitlines():
                    if line.startswith("VmHWM:"):
                        kernel_peak = int(line.split()[1]) * 1024
                        break
                stat_fields = Path(f"/proc/{self.process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                live_cpu = (int(stat_fields[11]) + int(stat_fields[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError, AttributeError):
            pass
        return {
            "supervisor_elapsed_seconds": time.monotonic() - self.started_at,
            "supervisor_peak_rss_bytes": int(own.ru_maxrss) * 1024,
            "kernel_peak_rss_bytes": kernel_peak,
            "exited_kernel_peak_rss_bytes": int(exited.ru_maxrss) * 1024,
            "tracked_process_cpu_seconds": own.ru_utime
            + own.ru_stime
            + exited.ru_utime
            + exited.ru_stime
            + live_cpu,
            **_cgroup_usage(self.config["memory_limit_mib"] * 1024**2),
            "observation_note": "OS counters, not provider billing. RSS excludes some container memory; cgroup scope must be verified before settlement.",
        }

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
            result = _receive_frame(parent)
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
        if identifier is not None and (not isinstance(identifier, str) or len(identifier) > 200):
            return {
                "id": identifier,
                "ok": False,
                "error": "Request id must be at most 200 characters",
                "runs": self.runs,
            }
        if request.get("op") in {"run", "restart"} and identifier is not None:
            digest = hashlib.sha256(
                json.dumps(request, sort_keys=True, ensure_ascii=True).encode()
            ).hexdigest()
            if identifier in self._completed:
                previous_digest, result = self._completed[identifier]
                if previous_digest != digest:
                    return {
                        "id": identifier,
                        "ok": False,
                        "error": "Request id was already used for different content",
                        "runs": self.runs,
                    }
                return dict(result)
            result = self._handle_once(request)
            self._completed[identifier] = (digest, dict(result))
            return result
        return self._handle_once(request)

    def _handle_once(self, request: dict) -> dict:
        identifier = request.get("id")
        op = request.get("op")
        common = {"id": identifier, "runs": self.runs, "restarted": False, "variables_lost": False}
        if op == "status":
            try:
                alive = self.process is not None and self.process.is_alive()
            except ValueError:
                alive = False
            return dict(common, ok=True, restarts=self.restarts, alive=alive, usage=self.usage())
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
        try:
            self._validate_code(request["code"])
        except (ValueError, SyntaxError) as exc:
            return dict(
                common, ok=False, output="", error=f"{type(exc).__name__}: {exc}"[:10_000], seconds=0.0
            )
        # Own the budget in this supervisor. The editor process can change its
        # own Python state, so its reported counter is never authoritative.
        self.runs += 1
        started = time.monotonic()
        try:
            _send_frame(self.connection, {"op": "run", "code": request["code"], "runs": self.runs - 1})
            if not self.connection.poll(self.config["cell_timeout"]):
                raise TimeoutError("Kernel execution exceeded its wall-clock limit")
            result = _receive_frame(self.connection)
            if result.get("event") == "fatal":
                raise RuntimeError(result.get("error", "Kernel process failed"))
            if (
                type(result.get("ok")) is not bool
                or not isinstance(result.get("output"), str)
                or result.get("error") is not None
                and not isinstance(result["error"], str)
                or not isinstance(result.get("seconds"), (float, int))
                or not math.isfinite(result["seconds"])
            ):
                raise ValueError("Malformed kernel response")
            result = {
                "ok": result["ok"],
                "output": result["output"][:10_000],
                "error": result["error"][:10_000] if result.get("error") is not None else None,
                "seconds": time.monotonic() - started,
                "runs": self.runs,
            }
        except (EOFError, BrokenPipeError, OSError, TimeoutError, RuntimeError, ValueError, TypeError) as exc:
            self.restart()
            return {
                "id": identifier,
                "ok": False,
                "output": "",
                "error": f"{type(exc).__name__}: {exc}"[:9_900]
                + ". Kernel restarted; variables were lost; /work files remain.",
                "seconds": time.monotonic() - started,
                "runs": self.runs,
                "restarted": True,
                "variables_lost": True,
            }
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
    parser.add_argument("--memory-limit-mib", type=int, default=98_304)
    parser.add_argument("--verify-read-only", action="store_true")
    config = vars(parser.parse_args())
    if config["cell_timeout"] <= 0 or not 0 <= config["initial_runs"] <= config["max_runs"]:
        parser.error("Timeout must be positive and run counters must be within budget")
    for name in _SECRET_NAMES:
        os.environ.pop(name, None)
    supervisor = Supervisor(config)
    print(json.dumps({"type": "ready", "ok": True, "runs": supervisor.runs}), flush=True)
    requests: queue.Queue = queue.Queue()
    output_lock = threading.Lock()

    def respond(result):
        with output_lock:
            print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)

    def read_requests():
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
                if request.get("op") == "status":
                    respond(supervisor.handle(request))
                    continue
                requests.put(request)
                if request.get("op") == "shutdown":
                    return
            except (ValueError, KeyError, TypeError) as exc:
                result = {
                    "id": None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "runs": supervisor.runs,
                }
                respond(result)
        requests.put(None)

    threading.Thread(target=read_requests, daemon=True, name="openwsp-kernel-input").start()
    try:
        while (request := requests.get()) is not None:
            respond(supervisor.handle(request))
            if request.get("op") == "shutdown":
                break
    finally:
        supervisor._stop()


if __name__ == "__main__":
    main()
