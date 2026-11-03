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
