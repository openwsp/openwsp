"""Provider-independent model weights and edits.

The store never exposes a mutable view of the base weights. Submitted edits remain
float32; exported public edits and checkpoints describe their bfloat16 evaluation.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


class EditRefused(ValueError):
    """An edit violates the configured weight-editing protocol."""


DEFAULT_FROZEN = frozenset(
    {
        "model.language_model.norm.weight",
        "model.language_model.embed_tokens.weight",
        "model.norm.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }
)


class _Originals(Mapping):
    def __init__(self, store: WeightStore):
        self.store = store

    def __getitem__(self, name):
        return self.store._read_original(name)

    def __iter__(self):
        return iter(self.store.names)

    def __len__(self):
        return len(self.store.names)


class WeightStore:
    """Weights backed by safetensors or a small in-memory tensor mapping.

    File-backed stores read tensors lazily. ``edit_dir`` uses the authors' one
    float32 ``<tensor-name>.npy`` per submitted tensor layout, so copying a saved
    edit directory restores it without losing precision.
    """

    def __init__(
        self,
        model_path: str | Path | Mapping[str, torch.Tensor],
        edit_dir: str | Path | None = None,
        *,
        frozen_names: set[str] | frozenset[str] | None = None,
        refuse_rescale: bool = True,
    ):
        self._lock = threading.RLock()
        self._memory: dict[str, torch.Tensor] | None = None
        self._files: dict[str, Path] = {}
        self._edits: dict[str, torch.Tensor] = {}
        self.frozen_names = DEFAULT_FROZEN if frozen_names is None else frozenset(frozen_names)
        self.refuse_rescale = refuse_rescale
        if isinstance(model_path, Mapping):
            self._memory = {
                name: torch.as_tensor(tensor).detach().to("cpu").clone().contiguous()
                for name, tensor in model_path.items()
            }
            names = self._memory.keys()
        else:
            path = Path(model_path)
            files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
            if not files:
                raise FileNotFoundError(f"No safetensors files in {path}")
            for file in files:
                with safe_open(file, framework="pt", device="cpu") as handle:
                    for name in handle.keys():  # noqa: SIM118 -- safe_open is not a mapping iterator.
                        if name in self._files:
                            raise ValueError(f"Duplicate tensor in checkpoint: {name}")
                        self._files[name] = file
            names = self._files.keys()
        self._names = tuple(sorted(names))
        for name in self._names:
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise ValueError(f"Unsafe tensor name: {name!r}")
        self.edit_dir = Path(edit_dir) if edit_dir is not None else None
        if self.edit_dir is not None:
            self.edit_dir.mkdir(parents=True, exist_ok=True)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def _check_name(self, name: str) -> None:
        if name not in (self._memory if self._memory is not None else self._files):
            raise EditRefused(f"Unknown tensor: {name}")

    def _read_original(self, name: str) -> torch.Tensor:
        self._check_name(name)
        if self._memory is not None:
            return self._memory[name].clone()
        with safe_open(self._files[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).clone()

    def load_original(self, name: str) -> torch.Tensor:
        return self._read_original(name).float().detach().clone()

    def _edit_path(self, name: str) -> Path:
        self._check_name(name)
        assert self.edit_dir is not None
        return self.edit_dir / f"{name}.npy"

    def load(self, name: str) -> torch.Tensor:
        self._check_name(name)
        with self._lock:
            if self.edit_dir is not None:
                file = self._edit_path(name)
                if file.exists():
                    tensor = torch.from_numpy(np.load(file, allow_pickle=False)).float()
                    self._validate(name, tensor)
                    return tensor.detach().clone()
            elif name in self._edits:
                return self._edits[name].clone()
        return self.load_original(name)

    def _validate(self, name: str, tensor: Any) -> torch.Tensor:
        self._check_name(name)
        if name in self.frozen_names:
            raise EditRefused(f"Frozen tensor (final norm, embedding or output layer): {name}")
        if not isinstance(tensor, (torch.Tensor, np.ndarray)):
            raise EditRefused("An edit must be a torch tensor or numpy array")
        if torch.is_tensor(tensor) and (tensor.is_complex() or tensor.requires_grad):
            raise EditRefused("Edits must be real tensors without gradients")
        if isinstance(tensor, np.ndarray) and np.iscomplexobj(tensor):
            raise EditRefused("Edits must be real tensors")
        value = torch.as_tensor(tensor).detach().to(device="cpu", dtype=torch.float32).contiguous()
        original = self.load_original(name)
        if value.shape != original.shape:
            raise EditRefused(f"Shape mismatch for {name}: {tuple(value.shape)} != {tuple(original.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise EditRefused(f"Non-finite edit values: {name}")
        if self.refuse_rescale and not torch.equal(value, original):
            flat = original.reshape(-1)
            denom = torch.dot(flat.double(), flat.double())
            if denom > 0:
                scale = torch.dot(flat.double(), value.reshape(-1).double()) / denom
                # Chosen tolerance is explicit because the original harness was not released.
                if torch.allclose(value, original * scale.float(), rtol=1e-5, atol=1e-7):
                    raise EditRefused(f"Whole-tensor scalar rescaling is forbidden: {name}")
        return value.clone()

    def _persist(self, name: str, value: torch.Tensor) -> None:
        if self.edit_dir is None:
            self._edits[name] = value.clone()
            return
        path = self._edit_path(name)
        fd, temporary = tempfile.mkstemp(prefix=".edit-", suffix=".tmp", dir=self.edit_dir)
        try:
            with os.fdopen(fd, "wb") as file:
                np.save(file, value.numpy(), allow_pickle=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_edit(self, name: str, tensor: Any) -> dict[str, Any]:
        with self._lock:
            value = self._validate(name, tensor)
            if torch.equal(value, self.load_original(name)):
                self.drop_edit(name)
                return {"name": name, "changed": False, "shape": list(value.shape)}
            self._persist(name, value)
            return {"name": name, "changed": True, "shape": list(value.shape), "dtype": "float32"}

    def drop_edit(self, name: str) -> bool:
        self._check_name(name)
        with self._lock:
            if self.edit_dir is None:
                return self._edits.pop(name, None) is not None
            path = self._edit_path(name)
            exists = path.exists()
            path.unlink(missing_ok=True)
            return exists

    def edited(self) -> list[str]:
        with self._lock:
            if self.edit_dir is None:
                return sorted(self._edits)
            names = sorted(file.name[:-4] for file in self.edit_dir.glob("*.npy"))
            for name in names:
                self._check_name(name)
            return names

    def diff(self) -> list[dict[str, Any]]:
        from .edits import tensor_sha256

        output = []
        for name in self.edited():
            original = self.load_original(name)
            value = self.load(name)
            delta = value - original
            norm = float(original.double().norm())
            changed = value.to(torch.bfloat16) != original.to(torch.bfloat16)
            output.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "changed_entries": int(changed.sum()),
                    "relative_change": float(delta.double().norm()) / norm if norm else None,
                    "base_sha256": tensor_sha256(original),
                    "edited_sha256": tensor_sha256(value),
                }
            )
        return output

    def export_edit(self, path: str | Path, *, base_model=None, revision=None) -> dict:
        from .edits import export_edit

        with self._lock:
            return export_edit(
                _Originals(self),
                {name: self.load(name) for name in self.edited()},
                path,
                base_model=base_model,
                revision=revision,
            )

    def apply_edit(self, path: str | Path) -> dict:
        """Validate the complete diff before replacing the currently submitted edit."""
        from .edits import read_edit

        with self._lock:
            manifest, tensors = read_edit(path, _Originals(self))
            validated = {name: self._validate(name, tensor) for name, tensor in tensors.items()}
            # All validation precedes mutation. Each file is replaced atomically.
            for name, value in validated.items():
                self._persist(name, value)
            for name in set(self.edited()) - validated.keys():
                self.drop_edit(name)
            return manifest

    def export_checkpoint(self, path: str | Path) -> Path:
        """Write the evaluated bfloat16 checkpoint; the base checkpoint is untouched."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tensors = {name: self.load(name).to(torch.bfloat16).contiguous() for name in self.names}
            save_file(tensors, destination, metadata={"format": "pt"})
        return destination
