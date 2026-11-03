"""GPU jobs with queue fallback and compatibility for already launched receipts."""
from __future__ import annotations

from pathlib import Path

from openwsp.backends.gpu_jobs import GPUJobRunner
from openwsp.backends.jobs import RemoteJobRunner
from openwsp.budget import compute_cost


class ProtocolGPUJobs:
    def __init__(self, ledger, directory, commit):
        from openwsp.backends.modal_app import start_gate_store
        self.legacy = RemoteJobRunner(ledger, directory, commit=commit)
        self.modern = GPUJobRunner(ledger, Path(directory) / "gated", start_gate_store, commit=commit)
        self._legacy_keys = set()

    def has_job(self, key):
        return self.legacy._path(key).exists() or self.modern.has_job(key)

    def launch(self, key, spawn_for, *, fingerprint):
        # Existing call IDs are immutable. They predate the start gate and must
        # be retrieved without inventing cancellation authority over them.
        if self.legacy._path(key).exists():
            self._legacy_keys.add(key)
            return self.legacy.launch(key, lambda: spawn_for("H100", ""),
                upper_bound_usd=compute_cost(2520, gpu="H100", cpu_cores=4, memory_gib=16),
                cost_options={"gpu": "H100", "cpu_cores": 4, "memory_gib": 16,
                              "startup_seconds": 120, "scaledown_seconds": 2}, fingerprint=fingerprint)
        return self.modern.launch(key, spawn_for, fingerprint=fingerprint)

    def result(self, key, timeout=3600):
        if key in self._legacy_keys:
            return self.legacy.result(key, timeout=timeout)
        return self.modern.result(key, timeout=timeout)

    def active_child_record(self, key):
        if key in self._legacy_keys:
            return self.legacy._read(self.legacy._path(key))
        return self.modern.active_child_record(key)


def gpu_instance(cls, gpu, gate_key, **parameters):
    return cls.with_options(gpu=gpu)(job_key=gate_key, gpu_kind=gpu, **parameters)
