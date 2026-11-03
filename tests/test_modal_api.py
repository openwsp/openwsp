"""Exercise the public Modal wire contract with real edits and no paid provider."""

import copy
import json
from decimal import Decimal

import pytest
import torch
from typer.testing import CliRunner

from openwsp.api import OpenWSP, TinyBackend, load_backend
from openwsp.backends import modal_api
from openwsp.backends.modal_api import (
    ModalForwardBackend,
    apply_payload_edit,
    make_response,
    reservation_bound,
)
from openwsp.budget import BudgetExceeded, compute_cost
from openwsp.cli import app
from openwsp.mcp_server import create_server

MODEL = "openwsp/tiny"
REVISION = "a" * 40
LIMITS = {
    "timeout_seconds": 1800,
    "startup_seconds": 2,
    "scaledown_seconds": 3,
    "gpu": "H100",
    "cpu_cores": 4,
    "memory_gib": 16,
}
REQUEST = {"inputs": [[1.0, 0.0], [0.0, 1.0]], "model": "edited"}


class Call:
    def __init__(self, call_id, response):
        self.object_id = call_id
        self.response = response
        self.gets = 0

    def get(self, *, timeout):
        assert timeout > 0
        self.gets += 1
        return self.response


@pytest.fixture
def setup(tmp_path):
    local = OpenWSP.tiny(tmp_path / "local")
    work_dir = tmp_path / "client"
    calls, payloads, lookups, remotes = [], [], [], []
    behavior = {"spawn_error": None, "response_change": None, "result_extra": {}}

    class Function:
        def spawn(self, payload):
            payloads.append(copy.deepcopy(payload))
            if behavior["spawn_error"]:
                raise behavior["spawn_error"]
            remote = OpenWSP.tiny(tmp_path / f"remote-{len(calls)}")
            apply_payload_edit(payload, remote.store, base_model=MODEL, base_revision=REVISION)
            result = TinyBackend().forward(payload["request"], remote.store)
            result.update(behavior["result_extra"])
            response = make_response(payload, result, container_seconds=1)
            if behavior["response_change"]:
                behavior["response_change"](response)
            call = Call(f"fc-public-{len(calls)}", response)
            calls.append(call)
            remotes.append(remote)
            return call

    def function_factory(app_name, function_name):
        assert (app_name, function_name) == ("test-openwsp", "public_forward")
        lookups.append((app_name, function_name))
        return Function()

    def call_factory(call_id):
        return next(call for call in calls if call.object_id == call_id)

    def backend(**overrides):
        return ModalForwardBackend(**{
            "app_name": "test-openwsp",
            "function_name": "public_forward",
            "base_model": MODEL,
            "base_revision": REVISION,
            "work_dir": work_dir,
            "limits": LIMITS,
            "budget_usd": 10,
            "enabled": True,
            "function_factory": function_factory,
            "call_factory": call_factory,
            **overrides,
        })

    return local, work_dir, calls, payloads, lookups, remotes, behavior, backend


def ledger(work_dir):
    return json.loads((work_dir / "budget.json").read_text())


def change_weight(local, amount=1):
    tensor = local.store.load_original("layer1.weight")
    tensor[0, 0] += amount
    local.store.save_edit("layer1.weight", tensor)


def test_client_constructor_has_no_remote_side_effects_and_disabled_forward_never_posts(setup):
    local, _, calls, payloads, lookups, _, _, backend = setup
    disabled = backend(enabled=False)
    assert not calls and not payloads and not lookups
    with pytest.raises((RuntimeError, ValueError), match="(?i)enable|disabled|opt.in"):
        disabled.forward(REQUEST, local.store)
    assert not calls and not payloads and not lookups


def test_real_v1_edit_roundtrip_changes_remote_forward_and_preserves_provenance(setup):
    local, _, calls, payloads, _, remotes, _, backend = setup
    original = TinyBackend().forward(REQUEST, local.store)
    change_weight(local)
    expected = TinyBackend().forward(REQUEST, local.store)
    result = backend().forward(REQUEST, local.store)
    assert result["logits"] == expected["logits"] != original["logits"]
    payload = payloads[0]
    assert payload["schema_version"] == 1
    assert payload["base"] == {"model": MODEL, "revision": REVISION}
    assert payload["request"] == REQUEST
    assert len(payload["request_id"]) == 64
    assert isinstance(payload["edit"]["manifest_json"], bytes)
    assert isinstance(payload["edit"]["edit_npz"], bytes)
    manifest = json.loads(payload["edit"]["manifest_json"])
    assert manifest["format"] == "openwsp.edit" and manifest["format_version"] == 1
    assert manifest["changed_tensors"] == 1
    assert manifest["base_model"] == MODEL and manifest["base_revision"] == REVISION
    assert result["_openwsp"]["request_id"] == payload["request_id"]
    assert result["_openwsp"]["edit_sha256"] == manifest["edit_sha256"]
    assert result["_openwsp"]["base"] == payload["base"]
    assert result["_openwsp"]["call_id"] == calls[0].object_id
    assert torch.equal(remotes[0].store.load("layer1.weight"), local.store.load("layer1.weight"))


def test_empty_edit_is_valid_and_restores_a_previously_edited_remote_store(setup):
    local, _, _, payloads, _, _, _, backend = setup
    change_weight(local)
    client = backend()
    client.forward(REQUEST, local.store)
    edited_payload = payloads[0]
    local.store.drop_edit("layer1.weight")
    original = TinyBackend().forward(REQUEST, local.store)
    result = client.forward(REQUEST, local.store)
    empty_payload = payloads[1]
    assert result["logits"] == original["logits"]
    assert json.loads(empty_payload["edit"]["manifest_json"])["changed_tensors"] == 0
    apply_payload_edit(edited_payload, local.store, base_model=MODEL, base_revision=REVISION)
    assert local.store.edited()
    apply_payload_edit(empty_payload, local.store, base_model=MODEL, base_revision=REVISION)
    assert local.store.edited() == []
    assert TinyBackend().forward(REQUEST, local.store)["logits"] == original["logits"]


def test_recreated_client_reuses_durable_response_without_a_second_post_or_receipt(setup):
    local, work_dir, calls, payloads, _, _, _, backend = setup
    change_weight(local)
    first = backend().forward(REQUEST, local.store)
    snapshot = ledger(work_dir)
    second = backend().forward(copy.deepcopy(REQUEST), local.store)
    assert first == second
    assert len(calls) == len(payloads) == 1
    assert calls[0].gets == 1
    assert ledger(work_dir)["reservations"] == snapshot["reservations"]
    assert len([event for event in ledger(work_dir)["events"] if event["event"] == "settled"]) == 1


def test_changed_edit_content_has_a_different_request_identity_and_job(setup):
    local, work_dir, calls, payloads, _, _, _, backend = setup
    first = backend().forward(REQUEST, local.store)
    change_weight(local)
    second = backend().forward(REQUEST, local.store)
    assert first["logits"] != second["logits"]
    assert payloads[0]["request_id"] != payloads[1]["request_id"]
    assert len(calls) == len(ledger(work_dir)["reservations"]) == 2


@pytest.mark.parametrize("change", ["request", "edit"])
def test_explicit_request_id_cannot_be_rebound_to_other_inputs_or_weights(setup, change):
    local, _, calls, payloads, _, _, _, backend = setup
    request = {**REQUEST, "request_id": "my-retry-key"}
    backend().forward(request, local.store)
    assert "request_id" not in payloads[0]["request"]
    if change == "request":
        request["inputs"] = [[2.0, 0.0]]
    else:
        change_weight(local)
    with pytest.raises((ValueError, RuntimeError), match="(?i)different|fingerprint|mismatch|identity"):
        backend().forward(request, local.store)
    assert len(calls) == len(payloads) == 1


def test_insufficient_local_budget_refuses_before_provider_post(setup):
    local, work_dir, calls, payloads, _, _, _, backend = setup
    with pytest.raises(BudgetExceeded):
        backend(budget_usd=0.01).forward(REQUEST, local.store)
    assert not calls and not payloads
    assert not ledger(work_dir)["reservations"]


@pytest.mark.parametrize("corruption", ["base-model", "base-revision", "manifest-pin", "manifest-hash", "payload"])
def test_remote_rejects_pin_or_edit_corruption_before_mutating_weights(setup, corruption):
    local, _, _, payloads, _, _, _, backend = setup
    change_weight(local)
    backend().forward(REQUEST, local.store)
    payload = copy.deepcopy(payloads[0])
    if corruption == "base-model":
        payload["base"]["model"] = "another/model"
    elif corruption == "base-revision":
        payload["base"]["revision"] = "b" * 40
    elif corruption in {"manifest-pin", "manifest-hash"}:
        manifest = json.loads(payload["edit"]["manifest_json"])
        manifest["base_revision" if corruption == "manifest-pin" else "edit_sha256"] = "b" * 40
        payload["edit"]["manifest_json"] = json.dumps(manifest).encode()
    else:
        payload["edit"]["edit_npz"] = b"not an NPZ archive"
    local.store.drop_edit("layer1.weight")
    with pytest.raises(ValueError):
        apply_payload_edit(payload, local.store, base_model=MODEL, base_revision=REVISION)
    assert local.store.edited() == []


@pytest.mark.parametrize("field", ["request_id", "edit_sha256", "base"])
def test_client_rejects_a_response_for_another_request_edit_or_base(setup, field):
    local, _, calls, _, _, _, behavior, backend = setup

    def corrupt(response):
        response[field] = {"model": "wrong", "revision": REVISION} if field == "base" else "0" * 64

    behavior["response_change"] = corrupt
    with pytest.raises((ValueError, RuntimeError)):
        backend().forward(REQUEST, local.store)
    assert len(calls) == 1


def test_credential_like_results_are_redacted_without_losing_token_counts(setup):
    local, work_dir, _, _, _, _, behavior, backend = setup
    synthetic_secret = "sk-" + "z" * 24
    behavior["result_extra"] = {
        "api_key": synthetic_secret,
        "diagnostic": "provider echoed " + synthetic_secret,
        "token_count": 123,
    }
    result = backend().forward(REQUEST, local.store)
    assert result["api_key"] == "[REDACTED]"
    assert synthetic_secret not in json.dumps(result)
    assert result["token_count"] == 123
    for path in work_dir.rglob("*.json"):
        assert synthetic_secret not in path.read_text()


def test_provider_error_credentials_are_not_exposed_or_written_to_journals(setup):
    local, work_dir, _, payloads, _, _, behavior, backend = setup
    synthetic_secret = "github_pat_" + "z" * 24
    behavior["spawn_error"] = RuntimeError("Unauthorized " + synthetic_secret)
    with pytest.raises(RuntimeError) as error:
        backend().forward(REQUEST, local.store)
    assert synthetic_secret not in str(error.value)
    assert len(payloads) == 1
    for path in work_dir.rglob("*.json"):
        assert synthetic_secret not in path.read_text()
    assert all(row["status"] == "reserved" for row in ledger(work_dir)["reservations"].values())
    assert sum(Decimal(row["upper_bound_usd"]) for row in ledger(work_dir)["reservations"].values()) > 0


def gateway_limits():
    return {**LIMITS, "gpu": None, "cpu_cores": 1, "memory_gib": 4, "children": [LIMITS, LIMITS]}


def test_gateway_reservation_includes_every_child_before_dispatch(setup):
    local, _, calls, payloads, _, _, _, backend = setup
    limits = gateway_limits()
    expected = compute_cost(1805, cpu_cores=1, memory_gib=4)
    expected += 2 * compute_cost(1805, gpu="H100", cpu_cores=4, memory_gib=16)
    assert reservation_bound(limits) == expected
    # The gateway CPU alone fits; its two GPU children do not.
    with pytest.raises(BudgetExceeded):
        backend(limits=limits, budget_usd=0.1).forward(REQUEST, local.store)
    assert not calls and not payloads


def test_gateway_cannot_settle_without_accounting_for_child_costs(setup):
    local, work_dir, calls, _, _, _, _, backend = setup
    with pytest.raises(RuntimeError):
        backend(limits=gateway_limits()).forward(REQUEST, local.store)
    assert len(calls) == 1
    assert all(row["status"] == "reserved" for row in ledger(work_dir)["reservations"].values())


def test_gateway_records_explicit_aggregate_cost_in_its_single_receipt(setup):
    local, work_dir, calls, _, _, _, behavior, backend = setup
    behavior["response_change"] = lambda response: response.update(cost_usd="0.25")
    result = backend(limits=gateway_limits()).forward(REQUEST, local.store)
    assert result["logits"] == TinyBackend().forward(REQUEST, local.store)["logits"]
    records = ledger(work_dir)["reservations"]
    assert len(records) == len(calls) == 1
    receipt = next(iter(records.values()))
    assert Decimal(receipt["upper_bound_usd"]) == reservation_bound(gateway_limits())
    assert Decimal(receipt["actual_usd"]) == Decimal("0.25")


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sdk", "cli", "mcp"])
async def test_builtin_factory_drives_all_public_interfaces_without_custom_backend_code(
    setup, monkeypatch, tmp_path, transport
):
    local, work_dir, calls, payloads, _, _, _, backend = setup
    dependencies = backend()
    monkeypatch.setattr(modal_api, "_function", dependencies.function_factory)
    monkeypatch.setattr(modal_api, "_call", dependencies.call_factory)
    values = {
        "APP": "test-openwsp", "FUNCTION": "public_forward", "BASE_MODEL": MODEL,
        "BASE_REVISION": REVISION, "WORK_DIR": str(work_dir), "BUDGET_USD": "10",
        "LIMITS": json.dumps(LIMITS), "ENABLED": "1",
    }
    for key, value in values.items():
        monkeypatch.setenv("OPENWSP_MODAL_" + key, value)
    model_path = tmp_path / "tiny-base.safetensors"
    local.store.export_checkpoint(model_path)
    change_weight(local)
    expected = TinyBackend().forward(REQUEST, local.store)
    factory = "openwsp.backends.modal_api:from_env"
    if transport == "cli":
        response = CliRunner().invoke(app, [
            "--model", str(model_path), "--work-dir", str(local.work_dir),
            "--backend", factory, "forward", json.dumps(REQUEST),
        ])
        assert response.exit_code == 0, response.output
        result = json.loads(response.stdout)
    else:
        api = OpenWSP(model_path, work_dir=local.work_dir, backend=load_backend(factory))
        if transport == "sdk":
            result = api.forward(REQUEST)
        else:
            _, result = await create_server(api).call_tool("forward", {"request": REQUEST})
    assert result["logits"] == expected["logits"]
    assert result["_openwsp"]["call_id"] == calls[0].object_id
    assert len(calls) == len(payloads) == 1
