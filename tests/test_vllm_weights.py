"""Exercise weight-swap invariants without vLLM, a GPU, or paid calls."""

import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from openwsp.backends.vllm_weights import (
    FreshEngineRequired,
    VLLMWeightSwapper,
    probe_fingerprint,
    probe_logprobs,
)

PREFIX = "model.layers.0."
Q = PREFIX + "self_attn.q_proj.weight"
GATE = PREFIX + "mlp.gate_proj.weight"
UP = PREFIX + "mlp.up_proj.weight"
PROBES = [[1, 2, 3], [1, 4]]


class Store:
    def __init__(self):
        self.tensors = {}
        for index, module in enumerate(("q_proj", "k_proj", "v_proj", "o_proj"), start=1):
            self.tensors[PREFIX + f"self_attn.{module}.weight"] = torch.full((2, 2), float(index))
            self.tensors[PREFIX + f"self_attn.{module}.bias"] = torch.full((2,), float(index))
        for index, module in enumerate(("gate_proj", "up_proj", "down_proj"), start=5):
            self.tensors[PREFIX + f"mlp.{module}.weight"] = torch.full((2, 2), float(index))
            self.tensors[PREFIX + f"mlp.{module}.bias"] = torch.full((2,), float(index))
        self.tensors["model.layers.1.mlp.down_proj.weight"] = torch.ones((2, 2))
        self.names = sorted(self.tensors)
        self.loaded = []

    def load_original(self, name):
        self.loaded.append(name)
        return self.tensors[name].clone()


def logprobs(weights):
    # These deterministic probes depend on both independently packed groups.
    return [-float(weights[Q].sum()), -float(weights[GATE].sum() + weights[UP].sum())]


class Engine:
    def __init__(self, store):
        self.weights = {name: value.clone() for name, value in store.tensors.items()}
        self.model = SimpleNamespace(
            packed_modules_mapping={
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                "gate_up_proj": ["gate_proj", "up_proj"],
            }
        )
        self.events = []
        self.updates = []
        self.reset_result = True
        self.rpc_error = None
        self.probe_error = None

    def apply_model(self, callback):
        return [callback(self.model)]

    def collective_rpc(self, method, *, kwargs):
        assert method == "reload_weights"
        assert kwargs["is_checkpoint_format"] is True
        self.events.append("reload")
        update = {name: value.clone() for name, value in kwargs["weights_iterator"]}
        self.updates.append(update)
        if self.rpc_error:
            raise self.rpc_error
        for name, value in update.items():
            self.weights[name].copy_(value)
        return [None]

    def reset_prefix_cache(self, *, reset_running_requests=False, reset_connector=True):
        assert reset_running_requests is False and reset_connector is True
        self.events.append("reset")
        return self.reset_result


def probe(engine, token_ids):
    assert token_ids == PROBES
    engine.events.append("probe")
    if engine.probe_error:
        raise engine.probe_error
    return logprobs(engine.weights)


@pytest.fixture
def setup():
    store = Store()
    engine = Engine(store)
    swapper = VLLMWeightSwapper(engine, store, probe_fn=probe)
    return store, engine, swapper


def reference(store, edits, edit_id):
    weights = {**store.tensors, **edits}
    return {
        "fresh_engine": True,
        "edit_id": edit_id,
        "probe_sha256": probe_fingerprint(PROBES),
        "logprobs": logprobs(weights),
    }


def apply(swapper, store, edits, edit_id):
    return swapper.apply(
        edits, edit_id=edit_id, probe_token_ids=PROBES, reference=reference(store, edits, edit_id)
    )


def test_swap_expands_packed_modules_and_biases_then_flushes_before_probe(setup):
    store, engine, swapper = setup
    edited = store.tensors[Q].clone()
    edited[0, 0] += 0.5
    pointers = {name: tensor.data_ptr() for name, tensor in engine.weights.items()}
    apply(swapper, store, {Q: edited}, "q-edit")
    expected = {
        PREFIX + f"self_attn.{module}.{suffix}"
        for module in ("q_proj", "k_proj", "v_proj")
        for suffix in ("weight", "bias")
    }
    assert set(engine.updates[0]) == expected
    assert engine.events == ["reload", "reset", "probe"]
    assert torch.equal(engine.weights[Q], edited)
    for name in expected - {Q}:
        assert torch.equal(engine.weights[name], store.tensors[name])
    assert not any("layers.1." in name or ".o_proj." in name or ".mlp." in name for name in store.loaded)
    assert all(engine.weights[name].data_ptr() == pointer for name, pointer in pointers.items())


def test_full_snapshots_preserve_retained_edits_and_restore_dropped_edits(setup):
    store, engine, swapper = setup
    q = store.tensors[Q] + 0.5
    gate = store.tensors[GATE] + 0.25
    up = store.tensors[UP] - 0.125
    apply(swapper, store, {Q: q, GATE: gate}, "first")
    apply(swapper, store, {Q: q, UP: up}, "second")
    assert torch.equal(engine.weights[Q], q)
    assert torch.equal(engine.weights[GATE], store.tensors[GATE])
    assert torch.equal(engine.weights[UP], up)
    packed_mlp = {
        PREFIX + f"mlp.{module}.{suffix}"
        for module in ("gate_proj", "up_proj")
        for suffix in ("weight", "bias")
    }
    assert packed_mlp <= set(engine.updates[-1])
    apply(swapper, store, {}, "restored-base")
    assert all(torch.equal(engine.weights[name], value) for name, value in store.tensors.items())


def test_identical_snapshots_are_noops_but_an_id_cannot_describe_different_weights(setup):
    store, engine, swapper = setup
    swapper.apply({}, edit_id="original", probe_token_ids=PROBES, reference=None)
    assert not engine.events
    edited = store.tensors[Q] + 0.5
    apply(swapper, store, {Q: edited}, "same-id")
    calls = list(engine.events)
    swapper.apply({Q: edited.clone()}, edit_id="same-id", probe_token_ids=PROBES, reference=None)
    assert engine.events == calls
    with pytest.raises(ValueError, match="(?i)edit|payload|different|identity"):
        apply(swapper, store, {Q: edited + 1}, "same-id")
    assert engine.events == calls


@pytest.mark.parametrize("defect", ["missing", "not-fresh", "wrong-edit", "wrong-probes"])
def test_a_swap_requires_a_reference_for_the_same_edit_and_probes(setup, defect):
    store, engine, swapper = setup
    edits = {Q: store.tensors[Q] + 0.5}
    ref = reference(store, edits, "edited")
    if defect == "missing":
        ref = None
    elif defect == "not-fresh":
        ref["fresh_engine"] = False
    elif defect == "wrong-edit":
        ref["edit_id"] = "another-checkpoint"
    else:
        ref["probe_sha256"] = probe_fingerprint([[999, 998]])
    with pytest.raises((ValueError, FreshEngineRequired)):
        swapper.apply(edits, edit_id="edited", probe_token_ids=PROBES, reference=ref)
    assert not engine.updates


@pytest.mark.parametrize(
    "bad", [torch.ones(3), torch.full((2, 2), float("nan")), torch.ones((2, 2), dtype=torch.complex64)]
)
def test_invalid_weights_are_rejected_before_touching_the_engine(setup, bad):
    store, engine, swapper = setup
    valid = {Q: store.tensors[Q] + 0.5}
    with pytest.raises((ValueError, TypeError)):
        swapper.apply(
            {Q: bad}, edit_id="bad", probe_token_ids=PROBES, reference=reference(store, valid, "bad")
        )
    assert not engine.updates
    apply(swapper, store, valid, "valid-after-rejection")
    assert len(engine.updates) == 1


@pytest.mark.parametrize("failure", ["mismatch", "reset-failed", "rpc-exception", "probe-exception"])
def test_failed_verification_poisoning_prevents_reuse_of_uncertain_weights(setup, failure):
    store, engine, swapper = setup
    edits = {Q: store.tensors[Q] + 0.5}
    ref = reference(store, edits, "edited")
    if failure == "mismatch":
        ref["logprobs"][0] -= 1
    elif failure == "reset-failed":
        engine.reset_result = False
    elif failure == "rpc-exception":
        engine.rpc_error = RuntimeError("worker disconnected during reload")
    else:
        engine.probe_error = RuntimeError("probe failed after weights changed")
    with pytest.raises(FreshEngineRequired):
        swapper.apply(edits, edit_id="edited", probe_token_ids=PROBES, reference=ref)
    calls = list(engine.events)
    with pytest.raises(FreshEngineRequired):
        apply(swapper, store, {}, "try-base")
    assert engine.events == calls


def test_session_holds_the_edit_stable_until_generation_finishes(setup):
    store, engine, swapper = setup
    first = {Q: store.tensors[Q] + 0.5}
    second = {Q: store.tensors[Q] + 1}
    started, finished = threading.Event(), threading.Event()
    errors = []

    def swap_again():
        started.set()
        try:
            apply(swapper, store, second, "second")
        except BaseException as exc:  # noqa: BLE001 - report worker failures in the parent test
            errors.append(exc)
        finally:
            finished.set()

    with swapper.session(
        first, edit_id="first", probe_token_ids=PROBES, reference=reference(store, first, "first")
    ):
        thread = threading.Thread(target=swap_again)
        thread.start()
        assert started.wait(1)
        assert not finished.wait(0.05)
        assert torch.equal(engine.weights[Q], first[Q])
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not errors
    assert torch.equal(engine.weights[Q], second[Q])


@pytest.mark.parametrize("defect", [None, "wrong-tokens", "missing-observed-token", "non-finite"])
def test_probe_reads_observed_input_token_scores_with_fixed_sampling(monkeypatch, defect):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    rows = [
        SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=[
                None,
                {2: SimpleNamespace(logprob=-1.1), 99: SimpleNamespace(logprob=-0.001)},
                {3: SimpleNamespace(logprob=-1.2)},
            ],
        ),
        SimpleNamespace(prompt_token_ids=[1, 4], prompt_logprobs=[None, {4: SimpleNamespace(logprob=-0.5)}]),
    ]
    if defect == "wrong-tokens":
        rows[0].prompt_token_ids = [1, 2, 99]
    elif defect == "missing-observed-token":
        rows[0].prompt_logprobs[1].pop(2)
    elif defect == "non-finite":
        rows[0].prompt_logprobs[1][2].logprob = float("nan")

    def generate(prompts, sampling, *, use_tqdm):
        assert prompts == [{"prompt_token_ids": row} for row in PROBES]
        assert vars(sampling) == {"temperature": 0, "max_tokens": 1, "prompt_logprobs": 0, "seed": 0}
        assert use_tqdm is False
        return rows

    engine = SimpleNamespace(generate=generate)
    if defect is None:
        assert probe_logprobs(engine, PROBES) == [-1.1, -1.2, -0.5]
    else:
        with pytest.raises(ValueError):
            probe_logprobs(engine, PROBES)
