"""Verified tensor-only updates for a resident vLLM engine.

Uses vLLM 0.33's documented native reload path, which preserves runtime tensor
storage and CUDA graph references. A reload includes complete fused modules:
untouched sibling weights are read locally, never sent by the editor.

Sources inspected at v0.33.0:
https://docs.vllm.ai/en/latest/training/layerwise/
https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/llm.py
https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/model_loader/reload/inplace.py
https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/model_loader/reload/layerwise.py

Only CPU contract tests run here. GPU log-probability verification is mandatory
at runtime; an incompatible update requires a recorded fresh-engine fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from contextlib import contextmanager

import torch


class FreshEngineRequired(RuntimeError):
    """The resident engine cannot safely serve more requests after this failure."""

    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def probe_fingerprint(token_ids: list[list[int]]) -> str:
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            not isinstance(row, list)
            or len(row) < 2
            or any(type(token) is not int or token < 0 for token in row)
            for row in token_ids
        )
    ):
        raise ValueError("Probes must contain nonempty lists of at least two nonnegative token IDs")
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def probe_logprobs(engine, token_ids: list[list[int]]) -> list[float]:
    """Read teacher-forced input-token scores, using a one-token generation request."""
    from vllm import SamplingParams

    probe_fingerprint(token_ids)
    outputs = engine.generate(
        [{"prompt_token_ids": row} for row in token_ids],
        SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0, seed=0),
        use_tqdm=False,
    )
    scores = []
    if len(outputs) != len(token_ids):
        raise ValueError("Probe engine returned the wrong number of requests")
    for requested, output in zip(token_ids, outputs, strict=True):
        if list(output.prompt_token_ids) != requested or output.prompt_logprobs is None:
            raise ValueError("Probe engine did not return the requested input-token log probabilities")
        if len(output.prompt_logprobs) != len(requested):
            raise ValueError("Incomplete prompt log probabilities")
        for token, probabilities in zip(requested[1:], output.prompt_logprobs[1:], strict=True):
            if probabilities is None or token not in probabilities:
                raise ValueError("Missing observed-token log probability")
            score = float(probabilities[token].logprob)
            if not math.isfinite(score):
                raise ValueError("Non-finite probe log probability")
            scores.append(score)
    return scores


def capture_reference(engine, token_ids: list[list[int]], edit_id: str, *, probe_fn=probe_logprobs) -> dict:
    """Capture reference evidence from an engine the caller has freshly loaded.

    The caller is responsible for loading the exact pinned checkpoint identified
    by edit_id and recording its provenance. This helper does not launch compute.
    """
    values = [float(value) for value in probe_fn(engine, token_ids)]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Fresh-engine probe must return finite log probabilities")
    return {
        "fresh_engine": True,
        "edit_id": edit_id,
        "probe_sha256": probe_fingerprint(token_ids),
        "logprobs": values,
        "engine_api": "vllm==0.33.0",
    }


def _packing(model) -> dict[str, list[str]]:
    return {name: list(members) for name, members in getattr(model, "packed_modules_mapping", {}).items()}


def _tensor(value) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.is_complex() or value.requires_grad:
        raise ValueError("Weight updates must be real tensors without gradients")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Weight updates must be finite")
    value = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Weight update overflows bfloat16")
    return value


def _fingerprint(edits: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(edits.items()):
        hasher.update(json.dumps([name, list(tensor.shape)], separators=(",", ":")).encode())
        hasher.update(tensor.view(torch.uint16).numpy().tobytes())
    return hasher.hexdigest()


class VLLMWeightSwapper:
    """Swap full edit snapshots while retaining a single engine loaded from the base.

    original_store exposes ``names`` and ``load_original(name)``. It must contain
    the same pinned base checkpoint loaded by engine. All generation using this
    engine must be serialized with ``session`` or equivalent external locking.
    Pass only currently edited tensors; omissions restore previous edits to base.
    """

    def __init__(self, engine, original_store, *, probe_fn=probe_logprobs, atol=1e-4, rtol=1e-4):
        if min(atol, rtol) < 0 or not all(math.isfinite(value) for value in (atol, rtol)):
            raise ValueError("Verification tolerances must be finite and nonnegative")
        self.engine, self.store, self.probe_fn = engine, original_store, probe_fn
        self.atol, self.rtol = atol, rtol
        self._lock = threading.RLock()
        self._poisoned = False
        self.last_failure: dict | None = None
        self._current: dict[str, torch.Tensor] = {}
        self._originals: dict[str, torch.Tensor] = {}
        self._seen: dict[str, str] = {}
        self._names = set(original_store.names)
        self._modules: dict[str, set[str]] = {}
        for name in self._names:
            self._modules.setdefault(name.rsplit(".", 1)[0], set()).add(name)
        try:
            mappings = engine.apply_model(_packing)
            if not mappings or any(mapping != mappings[0] for mapping in mappings[1:]):
                raise ValueError("Worker packing maps are unavailable or inconsistent")
            self._groups = list(mappings[0].values())
            if any(not isinstance(part, str) or "." in part for group in self._groups for part in group):
                raise ValueError("Unsupported packed module naming")
            if not callable(getattr(engine, "collective_rpc", None)) or not callable(
                getattr(engine, "reset_prefix_cache", None)
            ):
                raise TypeError("Engine lacks the documented reload/cache-reset API")
        except Exception as exc:
            raise FreshEngineRequired("Engine does not expose a compatible weight-reload contract") from exc

    def _original(self, name):
        if name not in self._originals:
            self._originals[name] = _tensor(self.store.load_original(name))
        return self._originals[name]

    def _snapshot(self, edits):
        if not isinstance(edits, Mapping):
            raise TypeError("edits must be a mapping from checkpoint tensor names to tensors")
        snapshot = {}
        for name, value in edits.items():
            if name not in self._names:
                raise ValueError(f"Unknown checkpoint tensor: {name}")
            tensor, original = _tensor(value), self._original(name)
            if tensor.shape != original.shape:
                raise ValueError(f"Shape mismatch for {name}")
            if not torch.equal(tensor, original):
                snapshot[name] = tensor
        return snapshot

    def _closure(self, changed):
        modules = {name.rsplit(".", 1)[0] for name in changed}
        expanded = True
        while expanded:
            expanded = False
            for module in list(modules):
                parent, _, component = module.rpartition(".")
                for group in self._groups:
                    if component in group:
                        siblings = {f"{parent}.{part}" if parent else part for part in group}
                        siblings &= self._modules.keys()
                        if not siblings <= modules:
                            modules.update(siblings)
                            expanded = True
        return sorted({name for module in modules for name in self._modules[module]})

    def _reference(self, reference, edit_id, token_ids):
        if not isinstance(reference, dict) or reference.get("fresh_engine") is not True:
            raise ValueError("Every weight swap requires log probabilities from a freshly loaded engine")
        if reference.get("edit_id") != edit_id or reference.get("probe_sha256") != probe_fingerprint(
            token_ids
        ):
            raise ValueError("Fresh-engine reference does not match this edit and probe")
        values = reference.get("logprobs")
        if (
            not isinstance(values, list)
            or not values
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in values
            )
        ):
            raise ValueError("Reference log probabilities must be a nonempty finite list")
        return values

    def apply(self, edits, *, edit_id: str, probe_token_ids: list[list[int]], reference: dict | None) -> dict:
        with self._lock:
            return self._apply(edits, edit_id=edit_id, probe_token_ids=probe_token_ids, reference=reference)

    def _apply(self, edits, *, edit_id, probe_token_ids, reference):
        if self._poisoned:
            raise FreshEngineRequired("A prior swap failed; this engine must be discarded")
        if not isinstance(edit_id, str) or not edit_id:
            raise ValueError("edit_id must identify the exact target checkpoint")
        desired = self._snapshot(edits)
        content_hash = _fingerprint(desired)
        if edit_id in self._seen and self._seen[edit_id] != content_hash:
            raise ValueError("The same edit_id was supplied with different weight values")
        changed = {
            name
            for name in self._current.keys() | desired.keys()
            if not torch.equal(
                self._current.get(name, self._original(name)), desired.get(name, self._original(name))
            )
        }
        if not changed:
            self._seen[edit_id] = content_hash
            return {
                "changed": False,
                "edit_id": edit_id,
                "weights_sha256": content_hash,
                "verification": "unchanged",
                "reload_tensors": [],
            }
        expected = self._reference(reference, edit_id, probe_token_ids)
        names = self._closure(changed)
        payload = [(name, desired.get(name, self._original(name))) for name in names]
        evidence = {
            "edit_id": edit_id,
            "weights_sha256": content_hash,
            "reload_tensors": names,
            "probe_sha256": probe_fingerprint(probe_token_ids),
            "atol": self.atol,
            "rtol": self.rtol,
            "phase": "reload",
        }
        try:
            self.engine.collective_rpc(
                "reload_weights", kwargs={"weights_iterator": payload, "is_checkpoint_format": True}
            )
            evidence["phase"] = "cache_reset"
            reset = self.engine.reset_prefix_cache(reset_running_requests=False, reset_connector=True)
            evidence["prefix_cache_reset"] = reset is True
            if reset is not True:
                raise ValueError("Prefix cache reset did not succeed; active requests may still exist")
            evidence["phase"] = "probe"
            actual = [float(value) for value in self.probe_fn(self.engine, probe_token_ids)]
            if len(actual) != len(expected) or not all(math.isfinite(value) for value in actual):
                raise ValueError("Swapped-engine probe has missing or non-finite log probabilities")
            errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
            evidence.update(observed_logprobs=actual, expected_logprobs=expected, max_abs_error=max(errors))
            if any(
                error > self.atol + self.rtol * abs(expected_value)
                for error, expected_value in zip(errors, expected, strict=True)
            ):
                raise ValueError("Swapped log probabilities differ from the freshly loaded checkpoint")
        except Exception as exc:
            self._poisoned = True
            self.last_failure = {**evidence, "exception_type": type(exc).__name__}
            raise FreshEngineRequired(
                "Weight swap/cache reset/fresh-engine verification failed", evidence=self.last_failure
            ) from exc
        self._current, self._seen[edit_id] = desired, content_hash
        return {
            "changed": True,
            "edit_id": edit_id,
            "weights_sha256": content_hash,
            "reload_tensors": names,
            "edited_tensors": sorted(desired),
            "prefix_cache_reset": True,
            "verification": "fresh_engine_logprobs",
            "probe_sha256": probe_fingerprint(probe_token_ids),
            "max_abs_error": max(errors),
            "atol": self.atol,
            "rtol": self.rtol,
            "reference": reference,
            "observed_logprobs": actual,
        }

    @contextmanager
    def session(self, edits, *, edit_id, probe_token_ids, reference):
        """Hold the swap lock through the caller's subsequent generation."""
        with self._lock:
            yield self._apply(edits, edit_id=edit_id, probe_token_ids=probe_token_ids, reference=reference)
