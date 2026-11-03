"""Persistent CPU analysis kernel with Experiment 8's public namespace.

This module enforces the protocol and execution limits, not an operating-system
security boundary. Run untrusted editors inside a network-blocked sandbox with
no secrets and read-only model/practice mounts. The driver must audit code for
semantic violations (for example manually implementing a forward pass).
"""

from __future__ import annotations

import ast
import io
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core import WeightStore


class ProtocolRefused(ValueError):
    """Code requested a capability outside the weight-analysis protocol."""


class KernelTimeout(TimeoutError):
    pass


_FORBIDDEN_IMPORTS = {
    "requests",
    "httpx",
    "urllib",
    "http",
    "socket",
    "ftplib",
    "aiohttp",
    "subprocess",
    "multiprocessing",
    "transformers",
    "vllm",
    "tensorflow",
    "jax",
}
_FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__", "breakpoint", "input"}
_FORBIDDEN_ATTRIBUTES = {
    "backward",
    "autograd",
    "optim",
    "optimizer",
    "enable_grad",
    "forward",
    "system",
    "popen",
    "spawn",
    "fork",
    "forkpty",
    "execl",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "exit",
    "_exit",
    "quit",
}
_EXECUTION_LOCK = threading.RLock()
_LAYER_CALLS = {
    "Linear",
    "Embedding",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "RNN",
    "GRU",
    "LSTM",
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Sequential",
    "Module",
    "linear",
    "embedding",
    "conv1d",
    "conv2d",
    "conv3d",
}
_GRAD_TOGGLES = {"set_grad_enabled", "requires_grad_"}


def validate_code(code: str, *, aliases: dict[str, str] | None = None) -> ast.Module:
    """Reject explicit training, model loading/execution, and process/network tools.

    The AST filter is intentionally described as a protocol guard. It cannot
    prove arbitrary Python is safe or recognize every handwritten network.
    """
    tree = ast.parse(code, mode="exec")
    callable_aliases = dict(aliases or {})
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"torch", "torch.nn.functional"}:
            for imported in node.names:
                if imported.name == "*":
                    raise ProtocolRefused("Wildcard torch imports cannot be checked; import names explicitly")
                if imported.name in _LAYER_CALLS | _GRAD_TOGGLES:
                    callable_aliases[imported.asname or imported.name] = imported.name
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            )
            for module in modules:
                parts = module.split(".")
                if parts[0] in _FORBIDDEN_IMPORTS or any(part in {"optim", "autograd"} for part in parts):
                    raise ProtocolRefused(f"Import not allowed by the analysis protocol: {module}")
                if module.startswith("torch.nn") and module != "torch.nn.functional":
                    if isinstance(node, ast.ImportFrom) and all(
                        alias.name == "functional" for alias in node.names
                    ):
                        continue
                    raise ProtocolRefused(
                        "Running model layers in the kernel is forbidden; use the forward tool"
                    )
                if (
                    isinstance(node, ast.ImportFrom)
                    and module == "torch"
                    and any(alias.name in {"optim", "autograd", "nn", "enable_grad"} for alias in node.names)
                ):
                    raise ProtocolRefused("Model layers and training modules are not available")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            raise ProtocolRefused(f"Operation not allowed by the analysis protocol: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise ProtocolRefused(f"Dynamic execution or interactive input is forbidden: {node.func.id}")
            if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) > 1:
                key = node.args[1]
                if isinstance(key, ast.Constant) and key.value in _FORBIDDEN_ATTRIBUTES:
                    raise ProtocolRefused("Accessing a forbidden operation dynamically is not allowed")
            callable_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else callable_aliases.get(node.func.id)
                if isinstance(node.func, ast.Name)
                else None
            )
            if callable_name in _LAYER_CALLS:
                raise ProtocolRefused("Running model layers in the kernel is forbidden; use the forward tool")
            if callable_name in _GRAD_TOGGLES:
                args = list(node.args) + [keyword.value for keyword in node.keywords]
                if not args or not all(isinstance(arg, ast.Constant) and arg.value is False for arg in args):
                    raise ProtocolRefused("Enabling gradients is forbidden")
            for keyword in node.keywords:
                if keyword.arg == "requires_grad" and not (
                    isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                ):
                    raise ProtocolRefused("Creating tensors with gradients is forbidden")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Attribute) and target.attr == "requires_grad" for target in targets
            ):
                raise ProtocolRefused("Changing gradient flags is forbidden")
    if aliases is not None:
        aliases.clear()
        aliases.update(callable_aliases)
    return tree


class _BoundedOutput(io.TextIOBase):
    def __init__(self, limit: int):
        self.limit = limit
        self.parts: list[str] = []
        self.length = 0
        self.truncated = False

    def write(self, value: str) -> int:
        available = max(0, self.limit - self.length)
        if available:
            self.parts.append(value[:available])
        self.length += min(available, len(value))
        self.truncated |= len(value) > available
        return len(value)

    def flush(self):
        pass

    def getvalue(self) -> str:
        text = "".join(self.parts)
        if self.truncated:
            marker = "\n[output truncated]"
            text = text[: max(0, self.limit - len(marker))] + marker[: self.limit]
        return text.encode("ascii", errors="replace").decode("ascii")


@contextmanager
def _deadline(seconds: float):
    if seconds <= 0:
        raise ValueError("Kernel timeout must be positive")
    started = time.monotonic()

    def expired(*_):
        raise KernelTimeout(f"Execution exceeded {seconds:g} seconds")

    if threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer"):
        handler = signal.getsignal(signal.SIGALRM)
        timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            if timer[0]:
                signal.setitimer(
                    signal.ITIMER_REAL, max(0.001, timer[0] - (time.monotonic() - started)), timer[1]
                )
    else:
        # Thread-hosted local clients can interrupt Python bytecode. Production
        # kernel workers run in the main thread; the sandbox supervisor also
        # enforces a wall-clock deadline on native-code execution.
        previous = sys.gettrace()

        def trace(frame, event, arg):
            if time.monotonic() - started > seconds:
                expired()
            return trace

        sys.settrace(trace)
        try:
            yield
        finally:
            sys.settrace(previous)


class Kernel:
    def __init__(
        self,
        store: WeightStore,
        work_dir: str | Path,
        practice_dir: str | Path,
        tokenizer=None,
        *,
        max_runs: int = 300,
        timeout: float = 900,
        output_limit: int = 10_000,
    ):
        self.store = store
        self.work_dir = Path(work_dir).resolve()
        self.practice_dir = Path(practice_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_runs = max_runs
        self.timeout = timeout
        self.output_limit = output_limit
        self.runs = 0
        self._call_aliases: dict[str, str] = {}
        self._lock = threading.RLock()
        self.namespace: dict[str, Any] = {
            "__name__": "__openwsp_kernel__",
            "NAMES": store.names,
            "load": store.load,
            "load_original": store.load_original,
            "save_edit": store.save_edit,
            "drop_edit": store.drop_edit,
            "edited": store.edited,
            "tok": tokenizer,
            "np": np,
            "torch": torch,
        }

    def run(self, code: str) -> dict[str, Any]:
        with self._lock, _EXECUTION_LOCK:
            started = time.monotonic()
            output = _BoundedOutput(self.output_limit)
            if self.runs >= self.max_runs:
                return {
                    "ok": False,
                    "output": "",
                    "error": "Code-run budget exhausted",
                    "seconds": 0.0,
                    "runs": self.runs,
                }
            # A refused block was not executed and does not consume a code run.
            try:
                tree = validate_code(code, aliases=self._call_aliases)
            except (ProtocolRefused, SyntaxError) as exc:
                return {
                    "ok": False,
                    "output": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.monotonic() - started,
                    "runs": self.runs,
                }
            self.runs += 1
            error = None
            cwd = Path.cwd()
            try:
                os.chdir(self.work_dir)
                with (
                    redirect_stdout(output),
                    redirect_stderr(output),
                    torch.no_grad(),
                    _deadline(self.timeout),
                ):
                    expression = None
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        expression = ast.Expression(tree.body.pop().value)
                        ast.fix_missing_locations(expression)
                    # This is the kernel's purpose; untrusted code needs external OS isolation.
                    exec(compile(tree, "<openwsp>", "exec"), self.namespace, self.namespace)  # noqa: S102
                    if expression is not None:
                        value = eval(compile(expression, "<openwsp>", "eval"), self.namespace, self.namespace)
                        self.namespace["_"] = value
                        if value is not None:
                            print(repr(value))
            except (Exception, SystemExit, KeyboardInterrupt) as exc:  # noqa: BLE001 -- return cell errors.
                error = f"{type(exc).__name__}: {exc}"[: self.output_limit]
                error = error.encode("ascii", errors="replace").decode("ascii")
            finally:
                os.chdir(cwd)
            return {
                "ok": error is None,
                "output": output.getvalue(),
                "error": error,
                "seconds": time.monotonic() - started,
                "runs": self.runs,
            }
