"""Exercise the deployed status body without importing or contacting Modal."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def status_function(volume, path):
    source = Path(__file__).resolve().parents[1] / "src/openwsp/backends/modal_app.py"
    node = next(item for item in ast.parse(source.read_text()).body
                if isinstance(item, ast.FunctionDef) and item.name == "status")
    decorators = {}

    def decorate(name, **options):
        decorators[name] = options
        return lambda function: function

    namespace = {
        "app": SimpleNamespace(function=lambda **options: decorate("function", **options)),
        "modal": SimpleNamespace(concurrent=lambda **options: decorate("concurrent", **options)),
        "driver_image": object(), "data_volume": volume, "json": json,
        "Path": lambda value: path if value == "/data/state.json" else Path(value),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace["status"], decorators


def test_warm_status_refreshes_each_call_before_reading(tmp_path):
    mounted = tmp_path / "state.json"
    mounted.write_text(json.dumps({"live": {"turn": 2}}))
    committed = {"live": {"turn": 3}}
    refreshes = []

    def reload():
        refreshes.append(True)
        mounted.write_text(json.dumps(committed))

    status, decorators = status_function(SimpleNamespace(reload=reload), mounted)
    assert status() == {"live": {"turn": 3}}
    committed["live"]["turn"] = 4
    assert status() == {"live": {"turn": 4}}
    assert len(refreshes) == 2
    assert decorators["concurrent"] == {"max_inputs": 1}


def test_status_refreshes_before_checking_for_missing_state(tmp_path):
    mounted = tmp_path / "state.json"
    mounted.write_text('{"status":"stale"}')
    status, _ = status_function(SimpleNamespace(reload=mounted.unlink), mounted)
    assert status() == {"status": "not_started"}


def test_failed_refresh_never_returns_cached_status(tmp_path):
    mounted = tmp_path / "state.json"
    mounted.write_text('{"status":"stale"}')
    failure = OSError("Refresh failed")

    def reload():
        raise failure

    status, _ = status_function(SimpleNamespace(reload=reload), mounted)
    with pytest.raises(OSError) as raised:
        status()
    assert raised.value is failure
