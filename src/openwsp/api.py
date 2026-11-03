"""One provider-independent API shared by Python, the CLI and MCP.

Local kernel code runs with the caller's filesystem permissions. Use an isolated
backend/sandbox for untrusted editors; this API is not an operating-system sandbox.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch

from .core import WeightStore
from .kernel import Kernel


class ForwardBackend(Protocol):
    """A compute backend receives an immutable request and the current weight view."""

    def forward(self, request: dict[str, Any], store: WeightStore) -> dict[str, Any]: ...


def load_backend(spec: str | None) -> ForwardBackend | None:
    """Load a trusted, installed ``module:factory`` plugin; the factory takes no arguments."""
    if spec is None:
        return None
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("backend must be an installed module:factory")
    backend = getattr(importlib.import_module(module), name)()
    if not callable(getattr(backend, "forward", None)):
        raise TypeError("backend must implement forward(request, store)")
    return backend


class OpenWSP:
    """An editable weight store, persistent kernel and optional forward backend."""

    def __init__(
        self,
        model: str | Path | Mapping[str, torch.Tensor],
        work_dir: str | Path = ".openwsp",
        practice_dir: str | Path | None = None,
        backend: ForwardBackend | None = None,
        tokenizer: Any = None,
    ):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.store = WeightStore(model, edit_dir=self.work_dir / "edited")
        self.backend = backend
        self.kernel = Kernel(
            self.store,
            work_dir=self.work_dir,
            practice_dir=Path(practice_dir).resolve() if practice_dir else self.work_dir / "practice",
            tokenizer=tokenizer,
        )

    @classmethod
    def tiny(cls, work_dir: str | Path = ".openwsp") -> OpenWSP:
        """A real, deterministic two-layer CPU network for interface tests and examples."""
        tensors = {
            "layer1.weight": torch.tensor([[1.0, 2.0], [-1.0, 1.0]]),
            "layer1.bias": torch.zeros(2),
            "layer2.weight": torch.tensor([[1.0, -1.0], [0.5, 1.0]]),
            "layer2.bias": torch.zeros(2),
        }
        return cls(tensors, work_dir=work_dir, backend=TinyBackend())

    def list_tensors(self) -> list[dict[str, Any]]:
        result = []
        for name in self.store.names:
            tensor = self.store.load(name)
            result.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        return result

    def read_tensor(self, name: str, offset: int = 0, limit: int = 256) -> dict[str, Any]:
        """Read a bounded flattened slice without dumping a whole large tensor."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4096:
            raise ValueError("limit must be an integer between 1 and 4096")
        tensor = self.store.load(name)
        total = tensor.numel()
        if offset > total:
            raise ValueError("offset exceeds tensor length")
        values = tensor.reshape(-1)[offset : offset + limit].tolist()
        return {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "offset": offset,
            "values": values,
            "total": total,
            "truncated": offset + len(values) < total,
        }

    def run_code(self, code: str) -> dict[str, Any]:
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        return self.kernel.run(code)

    def forward(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError("forward request must be a JSON object")
        if self.backend is None:
            raise RuntimeError("No forward backend configured; supply a backend or use the tiny CPU example")
        return self.backend.forward(request, self.store)

    def save_edit(self, name: str, values: Any) -> dict[str, Any]:
        # Preserve complex/gradient-bearing inputs for the core's refusal checks.
        return self.store.save_edit(name, torch.as_tensor(values))

    def diff(self) -> list[dict[str, Any]]:
        return self.store.diff()

    def apply_edit(self, path: str | Path) -> dict[str, Any]:
        return self.store.apply_edit(path)

    def export_edit(
        self, path: str | Path, base_model: str | None = None, revision: str | None = None
    ) -> dict[str, Any]:
        return self.store.export_edit(path, base_model=base_model, revision=revision)


class TinyBackend:
    """Two linear maps with tanh, executed without autograd; never used as an experiment result."""

    def forward(self, request: dict[str, Any], store: WeightStore) -> dict[str, Any]:
        model = request.get("model", "edited")
        if model not in ("original", "edited", "both"):
            raise ValueError("model must be original, edited, or both")
        inputs = torch.as_tensor(request.get("inputs", []), dtype=torch.float32)
        if inputs.ndim != 2 or inputs.shape[1] != 2 or not 1 <= inputs.shape[0] <= 512:
            raise ValueError("tiny forward inputs must have shape [1..512, 2]")
        if not torch.isfinite(inputs).all():
            raise ValueError("tiny forward inputs must be finite")
        if model == "both":
            return {arm: self.forward({**request, "model": arm}, store) for arm in ("original", "edited")}
        load = store.load_original if model == "original" else store.load
        with torch.inference_mode():
            hidden = torch.tanh(inputs @ load("layer1.weight").T + load("layer1.bias"))
            logits = hidden @ load("layer2.weight").T + load("layer2.bias")
            return {
                "backend": "tiny-cpu",
                "model": model,
                "logits": logits.tolist(),
                "probabilities": torch.softmax(logits, dim=-1).tolist(),
            }
