"""Provider-independent model weights and edits.

The store never exposes a mutable view of the base weights. Submitted edits remain
float32; exported public edits and checkpoints describe their bfloat16 evaluation.
"""

from __future__ import annotations

import copy
import os
import stat
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
                if file.exists() or file.is_symlink():
                    return self._read_submission_file(name, file)
            elif name in self._edits:
                return self._edits[name].clone()
        return self.load_original(name)

    def _read_submission_file(self, name: str, file: Path) -> torch.Tensor:
        """Read once into a validated snapshot, without following file symlinks."""
        try:
            if not stat.S_ISREG(file.lstat().st_mode):
                raise EditRefused(f"Submission must be a regular .npy file: {name}")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(file, flags)
            with os.fdopen(fd, "rb") as handle:
                before = os.fstat(handle.fileno())
                array = np.load(handle, allow_pickle=False)
                if not isinstance(array, np.ndarray):
                    if hasattr(array, "close"):
                        array.close()
                    raise EditRefused(f"Submission must contain one numpy array: {name}")
                after = os.fstat(handle.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise EditRefused(f"Submission changed while being read: {name}")
            # Preserve dtype through validation: casting first would silently
            # discard the imaginary component of a directly written complex edit.
            return self._validate(name, array)
        except EditRefused:
            raise
        except (OSError, ValueError, TypeError, EOFError) as exc:
            raise EditRefused(f"Invalid submission file for {name}: {type(exc).__name__}") from exc

    def validate_submission(self) -> dict[str, torch.Tensor]:
        """Return fresh, fully validated float32 copies of every submitted tensor.

        Call this outside the editor process before every forward pass and final
        freeze. The returned snapshot must be used directly: re-reading mutable
        files after validation would reopen a check/use race. External copies of
        saved edit directories are supported; no previously loaded value is
        cached. Unknown, frozen, malformed and invalid direct writes are refused.
        """
        with self._lock:
            names = self.edited()
            if self.edit_dir is None:
                return {name: self._validate(name, self._edits[name]) for name in names}
            try:
                before = {name: self._edit_path(name).lstat() for name in names}
            except OSError as exc:
                raise EditRefused("Submission changed during validation") from exc
            tensors = {name: self._read_submission_file(name, self._edit_path(name)) for name in names}
            if names != self.edited():
                raise EditRefused("Submitted tensor list changed during validation")
            try:
                for name in names:
                    first, last = before[name], self._edit_path(name).lstat()
                    if (first.st_ino, first.st_size, first.st_mtime_ns, first.st_ctime_ns) != (
                        last.st_ino,
                        last.st_size,
                        last.st_mtime_ns,
                        last.st_ctime_ns,
                    ):
                        raise EditRefused(f"Submission changed during validation: {name}")
            except OSError as exc:
                raise EditRefused("Submission changed during validation") from exc
            return tensors

    def snapshot(self) -> WeightStore:
        """Capture all validated edits for one forward without rereading live files.

        File-backed originals remain lazy and must remain immutable, as for this
        store. In-memory originals are cloned. Changes made through either store
        after this call do not alter the other's submitted tensors.
        """
        with self._lock:
            submitted = self.validate_submission()
            result = copy.copy(self)
            result._lock = threading.RLock()
            result._files = self._files.copy()
            result._memory = (
                None
                if self._memory is None
                else {name: value.clone() for name, value in self._memory.items()}
            )
            result.edit_dir = None
            result._edits = submitted
            return result

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
        try:
            value = torch.as_tensor(tensor).detach().to(device="cpu", dtype=torch.float32).contiguous()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EditRefused("Edits must contain real numeric values") from exc
        original = self.load_original(name)
        if value.shape != original.shape:
            raise EditRefused(f"Shape mismatch for {name}: {tuple(value.shape)} != {tuple(original.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise EditRefused(f"Non-finite edit values: {name}")
        if not bool(torch.isfinite(value.to(torch.bfloat16)).all()):
            raise EditRefused(f"Edit values overflow bfloat16 evaluation: {name}")
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
        self.edit_dir.mkdir(parents=True, exist_ok=True)
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
        for name, value in self.validate_submission().items():
            original = self.load_original(name)
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
                self.validate_submission(),
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
            previous = self.edited()
            # All validation precedes mutation. Each file is replaced atomically.
            for name, value in validated.items():
                self._persist(name, value)
            for name in set(previous) - validated.keys():
                self.drop_edit(name)
            return manifest

    def export_checkpoint(self, path: str | Path) -> Path:
        """Write the evaluated bfloat16 checkpoint; the base checkpoint is untouched."""
        destination = Path(path)
        if any(destination.resolve() == source.resolve() for source in set(self._files.values())):
            raise EditRefused("Cannot overwrite the original model checkpoint")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            submission = self.validate_submission()
            tensors = {
                name: (submission[name] if name in submission else self.load_original(name))
                .to(torch.bfloat16)
                .contiguous()
                for name in self.names
            }
            fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=destination.parent)
            os.close(fd)
            try:
                save_file(tensors, temporary, metadata={"format": "pt"})
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return destination
