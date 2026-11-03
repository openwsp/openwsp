import asyncio
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


def _invocation(response=None, error=None):
    from modal._functions import _Invocation

    async def outputs(*args, **kwargs):
        if error is not None:
            raise error
        return response

    return _Invocation(SimpleNamespace(FunctionGetOutputs=outputs), "fc-local-test", None)


async def test_actual_sdk_empty_unfinished_poll_is_pending():
    from modal_proto import api_pb2

    from openwsp.backends.codex_worker import _pending_timeout

    response = api_pb2.FunctionGetOutputsResponse(num_unfinished_inputs=1)
    with pytest.raises(TimeoutError) as raised:
        await _invocation(response).poll_function(timeout=0)
    assert type(raised.value) is TimeoutError
    assert _pending_timeout(raised.value)


@pytest.mark.parametrize("traceback_kind", ["missing", "corrupt", "valid"])
async def test_actual_sdk_terminal_user_timeout_is_never_pending(traceback_kind):
    from modal._serialization import serialize
    from modal._traceback import extract_traceback, traceback_contains_remote_call
    from modal_proto import api_pb2

    from openwsp.backends.codex_worker import _pending_timeout

    try:
        raise TimeoutError("remote user timeout")
    except TimeoutError as error:
        trace, lines = extract_traceback(error, "ta-" + "A" * 26)
        data = serialize(error)
    serialized = {}
    if traceback_kind == "valid":
        serialized = {"serialized_tb": serialize(trace), "tb_line_cache": serialize(lines)}
    elif traceback_kind == "corrupt":
        serialized = {"serialized_tb": b"not a serialized traceback"}
    result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                                 data=data, **serialized)
    response = api_pb2.FunctionGetOutputsResponse(outputs=[api_pb2.FunctionGetOutputsItem(
        result=result, data_format=api_pb2.DATA_FORMAT_PICKLE)])
    with pytest.raises(TimeoutError) as raised:
        await _invocation(response).poll_function(timeout=0)
    assert traceback_contains_remote_call(raised.value.__traceback__) is (traceback_kind == "valid")
    assert not _pending_timeout(raised.value)


async def test_actual_sdk_platform_execution_timeout_and_expired_output_are_not_pending():
    from modal.exception import FunctionTimeoutError, OutputExpiredError
    from modal_proto import api_pb2

    from openwsp.backends.codex_worker import _pending_timeout

    result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT)
    response = api_pb2.FunctionGetOutputsResponse(outputs=[api_pb2.FunctionGetOutputsItem(result=result)])
    with pytest.raises(FunctionTimeoutError) as execution:
        await _invocation(response).poll_function(timeout=0)
    assert not _pending_timeout(execution.value)
    with pytest.raises(OutputExpiredError) as expired:
        await _invocation(api_pb2.FunctionGetOutputsResponse()).poll_function(timeout=0)
    assert not _pending_timeout(expired.value)


async def test_actual_sdk_transport_timeout_and_unknown_origin_timeout_are_not_pending():
    from modal.exception import TimeoutError as ModalTimeoutError

    from openwsp.backends.codex_worker import _pending_timeout

    with pytest.raises(TimeoutError) as transport:
        await _invocation(error=TimeoutError("transport timeout")).poll_function(timeout=0)
    assert not _pending_timeout(transport.value)
    assert not _pending_timeout(TimeoutError())
    assert not _pending_timeout(ModalTimeoutError())


def test_worker_stops_before_bridge_work_on_sdk_remote_timeout_without_traceback(tmp_path):
    from modal._serialization import serialize
    from modal_proto import api_pb2

    from openwsp.backends.codex_worker import CodexWorker

    result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE,
                                 data=serialize(TimeoutError("remote user timeout")))
    response = api_pb2.FunctionGetOutputsResponse(outputs=[api_pb2.FunctionGetOutputsItem(
        result=result, data_format=api_pb2.DATA_FORMAT_PICKLE)])

    def get(*, timeout):
        return asyncio.run(_invocation(response).poll_function(timeout=timeout))

    def pending(*args):
        pytest.fail("A terminal driver timeout must not admit further bridge/inference work")

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    worker = CodexWorker("run-a", home=home, state_dir=tmp_path / "state",
                         work_dir=tmp_path / "work", store=SimpleNamespace(pending=pending))
    with pytest.raises(TimeoutError, match="remote user timeout"):
        worker.run(SimpleNamespace(get=get))
    assert worker._lock is None
