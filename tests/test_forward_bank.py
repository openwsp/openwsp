"""CPU fixtures for content identity and immutable forward-bank artifacts."""
import json

import pytest

from openwsp.backends.forward_bank import (
    BankError,
    ForwardResultBank,
    GenerationBank,
    checkpoint_hash,
    json_hash,
)


def identity(**changes):
    values = {"problem_id": "tiny-1", "prompt": "1+1", "sampling": {"n": 1, "temperature": 0.6},
              "model_hash": "a" * 64, "seed": 123}
    values.update(changes)
    return GenerationBank.identity(**values)


def output(text="2"):
    return {"id": "tiny-1", "samples": [{"text": text, "token_count": 1, "finish_reason": "stop"}]}


@pytest.mark.parametrize("change", [
    {"problem_id": "tiny-2"}, {"prompt": "2+2"}, {"sampling": {"n": 2, "temperature": 0.6}},
    {"model_hash": "b" * 64}, {"seed": 124},
])
def test_every_effective_generation_key_component_prevents_wrong_reuse(tmp_path, change):
    bank = GenerationBank(tmp_path)
    entry = bank.put(identity(), output(), provenance={"job_key": "first"})
    assert bank.get(identity()) == entry
    assert bank.get(identity(**change)) is None


def test_generation_bank_keeps_first_result_and_source_immutable(tmp_path):
    bank = GenerationBank(tmp_path)
    first = bank.put(identity(), output(), provenance={"job_key": "original-receipt"})
    assert bank.put(identity(), output("different"), provenance={"job_key": "other"}) == first
    first["output"]["samples"][0]["text"] = "locally changed"
    assert bank.get(identity())["output"] == output()


def test_generation_bank_rejects_corrupted_payload(tmp_path):
    bank = GenerationBank(tmp_path)
    entry = bank.put(identity(), output(), provenance={})
    path = tmp_path / f"{entry['key']}.json"
    changed = json.loads(path.read_text())
    changed["output"]["samples"][0]["text"] = "changed"
    path.write_text(json.dumps(changed))
    with pytest.raises(BankError, match="hash"):
        bank.get(identity())


def test_checkpoint_identity_binds_weight_and_tokenizer_bytes_not_directory_name(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "weights.safetensors").write_bytes(b"synthetic fixture, not model weights")
        (directory / "tokenizer_config.json").write_text('{"template":"one"}')
    assert checkpoint_hash(first) == checkpoint_hash(second)
    (second / "tokenizer_config.json").write_text('{"template":"two"}')
    assert checkpoint_hash(first) != checkpoint_hash(second)
    (second / "tokenizer_config.json").write_text('{"template":"one"}')
    (second / "weights.safetensors").write_bytes(b"different synthetic fixture")
    assert checkpoint_hash(first) != checkpoint_hash(second)


def test_full_bank_restores_private_verified_copy_after_editor_mutates_work_file(tmp_path):
    bank = ForwardResultBank(tmp_path / "private-bank")
    source = tmp_path / "gen_1.jsonl"
    source.write_bytes(b"synthetic result\n")
    key = {"body": "same request", "checkpoint": "effective weights", "texts": "exact content"}
    result = {"ok": True, "file": "/work/gen_1.jsonl"}
    entry = bank.put(key, result, request_number=1, artifacts={"gen_1.jsonl": source})
    source.write_bytes(b"editor changed work file")
    bank.restore(entry, tmp_path / "work", paths={"gen_1.jsonl": "gen_2.jsonl"})
    assert (tmp_path / "work/gen_2.jsonl").read_bytes() == b"synthetic result\n"
    assert bank.get({**key, "checkpoint": "changed weights"}) is None
    assert bank.get({**key, "texts": "changed text"}) is None
    assert bank.put(key, {"ok": True, "file": "different"}, request_number=2, artifacts={}) == entry


def test_full_bank_corruption_and_unsafe_restore_paths_fail_closed(tmp_path):
    bank = ForwardResultBank(tmp_path / "private")
    source = tmp_path / "array.npy"
    source.write_bytes(b"synthetic array")
    entry = bank.put({"request": 1}, {"ok": True}, request_number=1, artifacts={"fwd_1/array.npy": source})
    with pytest.raises(BankError, match="inside"):
        bank.restore(entry, tmp_path / "work", paths={"fwd_1/array.npy": "../escape"})
    stored = bank.root / entry["key"] / "files/fwd_1/array.npy"
    stored.write_bytes(b"corrupted array")
    with pytest.raises(BankError, match="changed"):
        bank.get({"request": 1})


def test_json_request_hash_is_canonical_and_rejects_nonfinite_numbers():
    assert json_hash({"a": 1, "b": 2}) == json_hash({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        json_hash({"invalid": float("nan")})


def test_cpu_verification_command_writes_a_non_scientific_result(tmp_path):
    from scripts.check_forward_bank import check
    destination = tmp_path / "result.json"
    result = check(destination)
    assert json.loads(destination.read_text()) == result
    assert result["status"] == "passed" and all(result["checks"].values())
    assert result["backend"] == "cpu_fixture" and "no scientific model inference" in result["provenance"]["scope"]
