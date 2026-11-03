from types import SimpleNamespace

import pytest

from openwsp.backends.editor_bridge import ModalBridgeStore, exchange


def test_rpc_boundary_preserves_committed_identity(tmp_path):
    events = []

    def remote(operation, args, kwargs):
        return exchange(operation, args, kwargs, directory=tmp_path,
                        reload=lambda: events.append("reload"), commit=lambda: events.append("commit"))

    store = ModalBridgeStore(SimpleNamespace(remote=remote))
    queued = store.submit("fixture", "turn-1", conversation_index=0, messages="Synthetic transport fixture")
    assert events[0] == "reload" and "commit" in events
    assert store.get("fixture", "turn-1")["request_hash"] == queued["request_hash"]
    assert store.pending("fixture")[0]["messages"] == "Synthetic transport fixture"
    with pytest.raises(ValueError, match="Unknown"):
        remote("__getattribute__", ["root"], {})


def test_rpc_boundary_rejects_malformed_arguments_before_reload(tmp_path):
    def reload():
        pytest.fail("Malformed input must be rejected before accessing a volume")

    with pytest.raises(TypeError, match="Malformed"):
        exchange("get", "wrong", {}, directory=tmp_path, commit=lambda: None, reload=reload)
