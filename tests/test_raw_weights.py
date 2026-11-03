import hashlib
import json

import pytest
import torch

from openwsp.core import WeightStore
from replications.raw_weights.edits import apply_numeric_edits
from replications.raw_weights.scoring import score_label


def test_numeric_edits_are_data_only_and_last_fence_wins(tmp_path):
    store = WeightStore({"W": torch.zeros(3, 2)}, refuse_rescale=False)
    attack = tmp_path / "executed"
    answer = f'''```edits
W[:, :] = 55
```
```edits
W[0:2, :] = [ [1, 2],
 [3, 4] ] # multiline lists with spaces
W[1, :] *= 2
W[2, :] = __import__('pathlib').Path('{attack}').touch()
W[2, 0] = 7
W[2, 1] = 1000001
```
'''
    result = apply_numeric_edits(answer, store)
    assert result["statements"] == 5 and result["applied"] == 3
    assert len(result["errors"]) == 2
    assert not attack.exists()
    assert torch.equal(store.load("W"), torch.tensor([[1., 2.], [6., 8.], [7., 0.]]))
    assert torch.count_nonzero(store.load_original("W")) == 0


def test_invalid_statement_is_atomic_and_zero_is_allowed():
    store = WeightStore({"a.W": torch.ones(2, 2)}, refuse_rescale=False)
    result = apply_numeric_edits("```edits\na.W[:, :] = [[2, 3, 4]]\nZERO a.W\na.W[0, 1] += 4\n```", store)
    assert result["applied"] == 2 and len(result["errors"]) == 1
    assert torch.equal(store.load("a.W"), torch.tensor([[0., 4.], [0., 0.]]))


def test_numeric_ranges_support_negative_steps_and_reject_zero_step():
    store = WeightStore({"b": torch.zeros(5)}, refuse_rescale=False)
    result = apply_numeric_edits("```edits\nb[4::-2] = [1, 2, 3]\nb[::0] = 99\nb[3:0:-1] += 1\n```", store)
    assert result["applied"] == 2 and len(result["errors"]) == 1
    assert torch.equal(store.load("b"), torch.tensor([3., 1., 3., 1., 1.]))


def test_unterminated_list_does_not_swallow_next_valid_statement():
    store = WeightStore({"b": torch.zeros(2)}, refuse_rescale=False)
    result = apply_numeric_edits("```edits\nb[0] = [1\nb[1] = 2\n```", store)
    assert result["statements"] == 2 and result["applied"] == 1
    assert torch.equal(store.load("b"), torch.tensor([0., 2.]))


def test_label_scorer_ignores_reasoning_and_ambiguous_final_labels():
    assert score_label("exp1", "freeway", "Reasoning: pong.\n## Game\nFrogger")['correct']
    assert not score_label("exp1", "pong", "I considered Pong.\nGame: Breakout")['correct']
    assert not score_label("exp1", "pong", "Game: Pong or Breakout")['correct']
    assert not score_label("exp2", "circle", "Reasoning: a circle")['correct']
    assert score_label("exp2", "circle", "Images: hollow circular discs")['correct']


def test_runner_one_reply_and_source_hash_guard(tmp_path):
    from replications.raw_weights.runner import run_item
    source = tmp_path / "source"
    source.mkdir()
    prompt, instructions = b"Identify this network.\n", b"Answer the user's request.\n"
    (source / "prompt.txt").write_bytes(prompt)
    (source / "instructions.md").write_bytes(instructions)
    item = {"experiment": "exp1", "item_id": "pong",
            "prompt": {"path": "prompt.txt", "sha256": hashlib.sha256(prompt).hexdigest()},
            "instructions": {"path": "instructions.md", "sha256": hashlib.sha256(instructions).hexdigest()}}
    calls = []

    class Editor:
        def __init__(self):
            self.last_response = {"status": "completed", "usage": {"output_tokens": 4}}

        def complete(self, text, *, request_key):
            calls.append((text, request_key))
            return "Game: Pong"

    def factory(received, system, state):
        assert received == item and system.encode() == instructions
        assert state.stat().st_mode & 0o777 == 0o700
        return Editor()

    kwargs = {"source_root": source, "state_root": tmp_path / "state", "output_root": tmp_path / "output",
              "editor_name": "astra_subscription", "editor_factory": factory}
    first = run_item(item, **kwargs)
    assert first["score"]["correct"]
    assert run_item(item, **kwargs) == first
    assert len(calls) == 1 and calls[0][0].encode() == prompt
    assert json.loads((tmp_path / "output/astra_subscription/exp1/pong/result.json").read_text())["tools"] == []
    (source / "prompt.txt").write_text("Changed prompt")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_item(item, **kwargs)
    assert len(calls) == 1


def test_deferred_scoring_never_constructs_an_editor(tmp_path, monkeypatch):
    from replications.raw_weights import runner
    source = tmp_path / "source"
    source.mkdir()
    data = b"fixed source"
    (source / "prompt.txt").write_bytes(data)
    item = {"experiment": "exp5", "item_id": "asterix", "prompt": {"path": "prompt.txt", "sha256": hashlib.sha256(data).hexdigest()},
            "instructions": {"path": "prompt.txt", "sha256": hashlib.sha256(data).hexdigest()}}
    class Editor:
        last_response = None

        def complete(self, text, *, request_key):
            return "```edits\nb3[0] = 1\n```"
    kwargs = {"source_root": source, "state_root": tmp_path / "state", "output_root": tmp_path / "out",
              "editor_name": "astra_subscription"}
    first = runner.run_item(item, editor_factory=lambda *args: Editor(), **kwargs)
    assert first["score"]["evaluation"]["status"] == "pending"
    seen = []
    def scorer(received_item, answer, prompt, target, *, rollout):
        seen.append((answer, rollout))
        return {"evaluation": {"episodes": 1000, "mean": 3.0}}
    monkeypatch.setattr(runner, "score_answer", scorer)
    result = runner.run_item(item, editor_factory=None, score_only=True, rollout=True, **kwargs)
    assert seen == [("```edits\nb3[0] = 1\n```", True)]
    assert result["receipt"] == first["receipt"] and result["answer_sha256"] == first["answer_sha256"]
    assert result["score"]["evaluation"]["episodes"] == 1000
    runner.run_item(item, editor_factory=None, score_only=True, rollout=True, **kwargs)
    assert len(seen) == 1


def test_blocked_reply_retains_receipt_without_scoring(tmp_path):
    from openwsp.editors.base import EditorIncomplete
    from replications.raw_weights.runner import run_item
    source = tmp_path / "source"
    source.mkdir()
    data = b"fixed source"
    (source / "prompt.txt").write_bytes(data)
    item = {"experiment": "exp1", "item_id": "pong", "prompt": {"path": "prompt.txt", "sha256": hashlib.sha256(data).hexdigest()},
            "instructions": {"path": "prompt.txt", "sha256": hashlib.sha256(data).hexdigest()}}
    class Editor:
        def __init__(self):
            self.last_response = {"outcome": "incomplete", "stop_reason": "max_tokens", "text": "Partial text",
                                  "usage": {"input_tokens": 20, "output_tokens": 10}, "cost_usd": "0.001"}

        def complete(self, text, *, request_key):
            raise EditorIncomplete("max_tokens", "message-id")
    with pytest.raises(EditorIncomplete):
        run_item(item, source_root=source, state_root=tmp_path / "state", output_root=tmp_path / "out",
                 editor_name="fable_native_anthropic", editor_factory=lambda *args: Editor())
    target = tmp_path / "out/fable_native_anthropic/exp1/pong"
    record = json.loads((target / "result.json").read_text())
    assert record["status"] == "blocked" and "score" not in record
    assert record["receipt"]["cost_usd"] == "0.001"
    assert (target / "incomplete-answer.md").read_text() == "Partial text"
    def forbidden(*args):
        raise AssertionError("Terminal failed attempt must never construct another editor")
    replay = run_item(item, source_root=source, state_root=tmp_path / "changed-state", output_root=tmp_path / "out",
                     editor_name="fable_native_anthropic", editor_factory=forbidden)
    assert replay == record


def test_uncertain_attempt_cannot_dispatch_after_private_state_loss(tmp_path):
    from openwsp.editors.base import UncertainRequest
    from replications.raw_weights.runner import run_item
    data = b"source"
    (tmp_path / "prompt.txt").write_bytes(data)
    item = {"experiment": "exp1", "item_id": "pong", "prompt": {"path": "prompt.txt", "sha256": hashlib.sha256(data).hexdigest()},
            "instructions": {"path": "prompt.txt", "sha256": hashlib.sha256(data).hexdigest()}}
    calls = []
    class Editor:
        last_response = None

        def has_request(self, key):
            return False  # Simulates lost private journal or an unrelated state root.

        def complete(self, *args, **kwargs):
            calls.append("dispatch")
            raise UncertainRequest("Provider outcome is unknown")
    options = {"source_root": tmp_path, "state_root": tmp_path / "state", "output_root": tmp_path / "out",
               "editor_name": "astra_subscription", "editor_factory": lambda *args: Editor()}
    with pytest.raises(UncertainRequest, match="unknown"):
        run_item(item, **options)
    record = (tmp_path / "out/astra_subscription/exp1/pong/result.json").read_bytes()
    with pytest.raises(UncertainRequest, match="durable request evidence"):
        run_item(item, **options)
    assert calls == ["dispatch"]
    assert (tmp_path / "out/astra_subscription/exp1/pong/result.json").read_bytes() == record
    options["state_root"] = tmp_path / "different-state"
    with pytest.raises(UncertainRequest, match="different private state"):
        run_item(item, **options)
    assert calls == ["dispatch"]


def test_car_scoring_subprocess_uses_pinned_cpu_environment(tmp_path, monkeypatch):
    from replications.raw_weights import runner
    executable = tmp_path / ".openwsp/kinetix-venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (tmp_path / ".openwsp/kinetix-source").mkdir()
    output = tmp_path / "result"
    output.mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key-must-not-reach-scorer")
    def run(args, *, cwd, env, check):
        import numpy as np
        assert args[0] == str(executable) and check and cwd == tmp_path
        assert env["JAX_PLATFORMS"] == "cpu" and "ANTHROPIC_API_KEY" not in env
        with np.load(args[args.index("--network") + 1], allow_pickle=False) as arrays:
            assert arrays["w"].dtype == np.float32
        Path(args[args.index("--output") + 1]).write_text('{"episodes":1024,"success_rate":1.0}')
    from pathlib import Path
    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.evaluate_car(WeightStore({"w": torch.ones(2)}), output)["episodes"] == 1024


def test_report_preserves_pending_and_verifies_answer_bytes(tmp_path, monkeypatch):
    from replications.raw_weights import report
    manifest = {"editors": ["astra_subscription", "fable_native_anthropic"], "revision": "pinned",
                "items_per_editor": 1, "items": [{"experiment": "exp1", "item_id": "pong", "prompt": {"sha256": "prompt-hash"},
                                                  "instructions": {"sha256": "instructions-hash"}}]}
    monkeypatch.setattr(report, "inventory", lambda: manifest)
    target = tmp_path / "astra_subscription/exp1/pong"
    target.mkdir(parents=True)
    answer = b"Game: Pong"
    (target / "answer.md").write_bytes(answer)
    binding = {"source_revision": "pinned", "prompt_sha256": "prompt-hash", "instructions_sha256": "instructions-hash",
               "editor": "astra_subscription", "experiment": "exp1", "item_id": "pong"}
    (target / "result.json").write_text(json.dumps({"binding": binding, "status": "completed",
        "answer_sha256": hashlib.sha256(answer).hexdigest(), "score": {"correct": True, "predicted": ["pong"]}}))
    rows = report.generate(tmp_path)["items"]
    assert rows[0]["correct"] is True and "cost_usd" not in rows[0]
    assert rows[1]["reply_status"] == "pending" and "correct" not in rows[1]
    (target / "answer.md").write_text("Changed answer")
    with pytest.raises(ValueError, match="scored receipt"):
        report.generate(tmp_path)
    (target / "answer.md").write_bytes(answer)
    wrong = json.loads((target / "result.json").read_text())
    wrong["binding"]["editor"] = "fable_native_anthropic"
    (target / "result.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="binding differs"):
        report.generate(tmp_path)
    wrong["binding"] = binding
    wrong.pop("answer_sha256")
    (target / "result.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="hash-verified answer"):
        report.generate(tmp_path)


def test_fable_limits_count_exact_messages_without_generation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from replications.raw_weights import fable_limits
    data = b"source prompt\n"
    (tmp_path / "prompt.txt").write_bytes(data)
    item = {"experiment": "exp5", "item_id": "asterix", "authors_config": {"effort": "xhigh"},
            "prompt": {"path": "prompt.txt", "sha256": hashlib.sha256(data).hexdigest()},
            "instructions": {"path": "prompt.txt", "sha256": hashlib.sha256(data).hexdigest()}}
    monkeypatch.setattr(fable_limits, "inventory", lambda: {"items": [item]})
    calls = []
    def count(**request):
        calls.append(request)
        return SimpleNamespace(input_tokens=1000)
    client = SimpleNamespace(models=SimpleNamespace(retrieve=lambda model: SimpleNamespace(id=model, lifecycle="active", max_input_tokens=1000000, max_tokens=128000)),
                             messages=SimpleNamespace(count_tokens=count))
    result = fable_limits.prepare(tmp_path, tmp_path / "limits.json", client=client)
    assert calls[0]["messages"] == [{"role": "user", "content": data.decode()}]
    assert calls[0]["system"] == data.decode() and calls[0]["tools"] == []
    assert calls[0]["output_config"] == {"effort": "xhigh"}
    assert result["exp5/asterix"]["max_input_tokens"] == 2074
    assert result["exp5/asterix"]["max_output_tokens"] == 32768


def test_minatar_input_channels_and_seeded_rollouts():
    import numpy as np

    from replications.raw_weights.edits import zero_minatar_store
    from replications.raw_weights.minatar_score import evaluate, observation

    raw = np.zeros((10, 10, 3), dtype=bool)
    raw[2, 4, 0] = raw[8, 1, 1] = True
    mapped = observation("asterix", raw, {"player": 0, "enemy": 1, "gold": 2, "trail": 2}).reshape(10, 10, 6)
    assert mapped[2, 4, 0] == 1 and mapped[8, 1, 1] == 1
    assert mapped.sum() == 2

    class FakeGame:
        def __init__(self):
            self.channels = {"player": 0, "enemy": 1, "gold": 2, "trail": 2}

    class Environment:
        def __init__(self, game, **options):
            assert game == "asterix" and options == {"sticky_action_prob": 0.1, "difficulty_ramping": True}
            self.env = FakeGame()

        def seed(self, seed):
            self.rng = np.random.default_rng(seed)

        def reset(self):
            self.steps = 0

        def state(self):
            return raw

        def act(self, action):
            self.steps += 1
            return int(action == self.rng.integers(6)), self.steps == 4

    kwargs = {"episodes": 8, "seed": 1234, "max_steps": 20, "env_factory": Environment}
    first = evaluate("asterix", zero_minatar_store(), **kwargs)
    assert first == evaluate("asterix", zero_minatar_store(), **kwargs)
    assert first["mean_steps"] == 4 and len(first["records"]) == 8
    assert all(row["finish_reason"] == "terminated" for row in first["records"])
