"""Run the public gateway locally with real checkpoint edits and fake GPU calls."""

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openwsp.api import OpenWSP, TinyBackend
from openwsp.backends.modal_api import reservation_bound
from openwsp.backends.public_forward import GATEWAY_LIMITS, GPU_LIMITS, PUBLIC_LIMITS, execute
from openwsp.budget import BudgetExceeded, BudgetLedger
from openwsp.core import WeightStore

MODEL, REVISION, REQUEST_ID = "Qwen/test", "a" * 40, "1" * 64


class Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert isinstance(text, str)
        return [1, 2]

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert tokenize is False
        return " ".join(row["content"] for row in messages)


@pytest.fixture
def setup(tmp_path):
    data_root = tmp_path / "data"
    original = data_root / "original"
    original.mkdir(parents=True)
    local = OpenWSP.tiny(tmp_path / "local")
    local.store.export_checkpoint(original / "model.safetensors")
    (original / "config.json").write_text(json.dumps({"text_config": {
        "hidden_size": 2, "intermediate_size": 4, "num_hidden_layers": 2,
        "num_attention_heads": 1, "vocab_size": 8,
    }}))
    # A gateway request must use its private ledger, never the live experiment ledger.
    (data_root / "budget.json").write_text('{"live_pipeline":"do not touch"}')
    calls, events, checkpoints, commits, reloads = {}, [], [], [], []
    behavior = {"artifact_symlink": False, "wrong_generation_id": False}

    class Call:
        def __init__(self, result, kind):
            self.object_id = f"fc-gateway-{len(calls)}"
            self.result, self.kind = result, kind
            calls[self.object_id] = self

        def get(self, *, timeout):
            assert timeout > 0
            events.append("get:" + self.kind)
            return self.result

    def generator(*, checkpoint, slot):
        checkpoints.append(("generate", checkpoint, slot))

        def spawn(rows, sampling, thinking, seed_by_id):
            events.append("spawn:generate")
            store = WeightStore(checkpoint)
            value = TinyBackend().forward({"inputs": [[1.0, 0.0]], "model": "edited"}, store)["logits"][0][0]
            outputs = [{
                "id": "wrong-id" if behavior["wrong_generation_id"] else row["id"],
                "samples": [{"text": str(value), "token_count": 3, "finish_reason": "stop"}
                            for _ in range(sampling["n"])],
            } for row in rows]
            return Call({"outputs": outputs, "container_seconds": 1}, "generate")

        return SimpleNamespace(generate=SimpleNamespace(spawn=spawn))

    def reader(*, checkpoint, eager):
        checkpoints.append(("teacher_forced", checkpoint, eager))

        def spawn(request, directory):
            events.append("spawn:teacher_forced")
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / "input_ids.npy", np.asarray([[1, 2]], dtype=np.int64), allow_pickle=False)
            (path / "meta.json").write_text(json.dumps({"lengths": [2], "files": {"input_ids": [1, 2]}}))
            if behavior["artifact_symlink"]:
                (path / "outside.npy").symlink_to(data_root / "budget.json")
            return Call({
                "ok": True, "type": "teacher_forced", "files": {"input_ids": [1, 2]},
                "lengths": [2], "mean_token_logprob": [-1.0], "bytes": 16, "container_seconds": 1,
            }, "teacher_forced")

        return SimpleNamespace(read=SimpleNamespace(spawn=spawn))

    runtime = SimpleNamespace(
        registration={"model": MODEL, "revision": REVISION},
        Generator=generator, TransformerReader=reader, tokenizer=lambda path: Tokenizer(),
        commit=lambda: commits.append(True), reload=lambda: reloads.append(True),
        call_factory=lambda call_id: calls[call_id],
    )

    def payload(request=None, *, changed=True, request_id=REQUEST_ID):
        if changed:
            value = local.store.load_original("layer1.weight")
            value[0, 0] += 1
            local.store.save_edit("layer1.weight", value)
        else:
            local.store.drop_edit("layer1.weight")
        edit_path = tmp_path / "edit"
        local.store.export_edit(edit_path, base_model=MODEL, revision=REVISION)
        return {
            "schema_version": 1, "base": {"model": MODEL, "revision": REVISION},
            "request_id": request_id,
            "request": request or {"type": "generate", "model": "both", "prompts": ["hello"], "n": 1, "max_tokens": 8},
            "edit": {"manifest_json": (edit_path / "manifest.json").read_bytes(),
                     "edit_npz": (edit_path / "edit.npz").read_bytes()},
            "limits": copy.deepcopy(PUBLIC_LIMITS),
        }

    return data_root, local, runtime, payload, calls, events, checkpoints, behavior, commits, reloads


def private_ledger(data_root):
    return json.loads((data_root / "public-forward-state" / REQUEST_ID / "budget.json").read_text())


def test_both_arms_dispatch_before_any_get_and_use_the_correct_frozen_weights(setup):
    data_root, local, runtime, payload, calls, events, checkpoints, _, commits, reloads = setup
    request = payload()
    response = execute(request, data_root=data_root, runtime=runtime)
    assert events == ["spawn:generate", "spawn:generate", "get:generate", "get:generate"]
    assert len(calls) == 2
    result = response["result"]
    original_text = result["original"]["outputs"][0]["samples"][0]["text"]
    edited_text = result["edited"]["outputs"][0]["samples"][0]["text"]
    expected = TinyBackend().forward({"inputs": [[1.0, 0.0]], "model": "both"}, local.store)
    assert float(original_text) == expected["original"]["logits"][0][0]
    assert float(edited_text) == expected["edited"]["logits"][0][0]
    assert original_text != edited_text
    assert checkpoints[0][1] == str(data_root / "original")
    assert checkpoints[1][1] != checkpoints[0][1]
    records = private_ledger(data_root)["reservations"]
    assert len(records) == 2 and all(row["status"] == "settled" for row in records.values())
    total = sum(Decimal(row["actual_usd"]) for row in records.values()) + reservation_bound(GATEWAY_LIMITS)
    assert Decimal(response["cost_usd"]) == total
    assert (data_root / "budget.json").read_text() == '{"live_pipeline":"do not touch"}'
    assert commits and reloads


def test_pair_budget_is_checked_before_either_gpu_can_launch(setup):
    data_root, _, runtime, payload, calls, events, _, _, _, _ = setup
    cap = reservation_bound(GPU_LIMITS) * Decimal("1.5")
    budget_path = data_root / "public-forward-state" / REQUEST_ID / "budget.json"
    BudgetLedger(budget_path, modal_task_limit=cap, modal_cycle_limit=cap, modal_prior_usage=0)
    with pytest.raises(BudgetExceeded):
        execute(payload(), data_root=data_root, runtime=runtime)
    assert not calls and not events
    assert not private_ledger(data_root)["reservations"]


def test_completed_request_reuses_saved_response_without_new_gpu_calls_or_receipts(setup):
    data_root, _, runtime, payload, calls, events, _, _, _, _ = setup
    request = payload()
    first = execute(request, data_root=data_root, runtime=runtime)
    before = private_ledger(data_root)
    event_count = len(events)
    second = execute(copy.deepcopy(request), data_root=data_root, runtime=runtime)
    assert second == first
    assert len(calls) == 2 and len(events) == event_count
    assert private_ledger(data_root) == before


@pytest.mark.parametrize("changed", ["prompt", "edit"])
def test_same_request_id_cannot_silently_rebind_the_prompt_or_edit(setup, changed):
    data_root, _, runtime, payload, calls, _, _, _, _, _ = setup
    request = payload()
    execute(request, data_root=data_root, runtime=runtime)
    edit_dir = data_root / "public-forward-state" / REQUEST_ID / "edited"
    persisted = WeightStore(data_root / "original", edit_dir=edit_dir)
    original_edit = persisted.load("layer1.weight")
    if changed == "prompt":
        altered = copy.deepcopy(request)
        altered["request"]["prompts"] = ["different prompt"]
    else:
        altered = payload(changed=False)
    with pytest.raises(ValueError, match="different"):
        execute(altered, data_root=data_root, runtime=runtime)
    assert WeightStore(data_root / "original", edit_dir=edit_dir).load("layer1.weight").equal(original_edit)
    assert len(calls) == 2


@pytest.mark.parametrize("defect", ["model-pin", "resource-bounds", "sample-limit", "private-data", "bad-type"])
def test_invalid_request_is_rejected_before_provider_construction_or_dispatch(setup, defect):
    data_root, _, runtime, payload, calls, events, checkpoints, _, _, _ = setup
    request = payload()
    if defect == "model-pin":
        request["base"]["revision"] = "b" * 40
    elif defect == "resource-bounds":
        request["limits"]["children"].pop()
    elif defect == "sample-limit":
        request["request"]["n"] = 257
    elif defect == "private-data":
        request["request"]["problems"] = ["held-out-example"]
    else:
        request["request"]["type"] = "unknown"
    with pytest.raises(ValueError):
        execute(request, data_root=data_root, runtime=runtime)
    assert not calls and not events and not checkpoints


def test_teacher_forced_artifacts_are_bounded_hashed_and_owned_by_this_request(setup):
    data_root, _, runtime, payload, calls, _, checkpoints, _, _, _ = setup
    request = payload({
        "type": "teacher_forced", "model": "edited", "texts": ["one two"],
        "return": {"token_logprobs": True},
    })
    response = execute(request, data_root=data_root, runtime=runtime)
    assert len(calls) == 1
    assert checkpoints[0][0] == "teacher_forced"
    result = response["result"]
    assert result["ok"] and result["arrays"] == {"input_ids": [1, 2]}
    assert {item["name"] for item in response["artifacts"]} == {"input_ids.npy", "meta.json"}
    for item in response["artifacts"]:
        assert item["path"].startswith("public-forward/" + REQUEST_ID + "/")
        path = data_root / item["path"]
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(private_ledger(data_root)["reservations"]) == 1


def test_worker_symlink_cannot_publish_files_from_outside_its_artifact_directory(setup):
    data_root, _, runtime, payload, calls, _, _, behavior, _, _ = setup
    behavior["artifact_symlink"] = True
    request = payload({"type": "teacher_forced", "model": "edited", "texts": ["one two"], "return": {}})
    with pytest.raises(ValueError, match="unsafe|Unexpected"):
        execute(request, data_root=data_root, runtime=runtime)
    assert len(calls) == 1


def test_generation_cannot_be_reported_under_the_wrong_prompt_id(setup):
    data_root, _, runtime, payload, calls, _, _, behavior, _, _ = setup
    behavior["wrong_generation_id"] = True
    with pytest.raises(ValueError, match="submitted prompts"):
        execute(payload(), data_root=data_root, runtime=runtime)
    assert len(calls) == 2
