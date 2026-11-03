"""Modal transport for durable editor intents; authentication stays on the worker."""
from __future__ import annotations

from typing import Any


class ModalBridgeStore:
    def __init__(self, function=None):
        if function is None:
            from openwsp.backends.modal_runtime import configure_tls
            configure_tls()
            import modal
            function = modal.Function.from_name("openwsp-exp8", "editor_exchange")
        self.function = function

    def _call(self, operation: str, *args: Any, **kwargs: Any):
        return self.function.remote(operation, list(args), kwargs)

    def submit(self, *args, **kwargs):
        return self._call("submit", *args, **kwargs)

    def get(self, *args, **kwargs):
        return self._call("get", *args, **kwargs)

    def pending(self, *args, **kwargs):
        return self._call("pending", *args, **kwargs)

    def claim(self, *args, **kwargs):
        return self._call("claim", *args, **kwargs)

    def dispatch(self, *args, **kwargs):
        return self._call("dispatch", *args, **kwargs)

    def finish(self, *args, **kwargs):
        return self._call("finish", *args, **kwargs)

    def block(self, *args, **kwargs):
        return self._call("block", *args, **kwargs)


def exchange(operation, args, kwargs, *, directory, commit, reload):
    """Called by one serialized remote owner; never accepts arbitrary methods."""
    if operation not in {"submit", "get", "pending", "claim", "dispatch", "finish", "block"}:
        raise ValueError("Unknown editor bridge operation")
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise TypeError("Malformed editor bridge call")
    from openwsp.editors.bridge import BridgeStore
    reload()
    return getattr(BridgeStore(directory, commit=commit), operation)(*args, **kwargs)
