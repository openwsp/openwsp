"""CPU-only MinAtar 1.0.15 evaluation reconstructed from the released prompts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect

import numpy as np

CHANNELS = {
    "breakout": (("paddle",), (), ("brick",), ("ball",), ("trail",), ()),
    "asterix": (("player",), ("enemy",), ("gold",), (), ("trail",), ()),
    "seaquest": (("sub_front", "sub_back"), ("enemy_fish", "enemy_sub", "enemy_bullet"),
                 ("enemy_fish", "enemy_sub", "diver"), ("friendly_bullet",),
                 ("trail", "sub_back"), ("oxygen_guage", "diver_guage")),
}


def observation(game, state, names):
    result = np.zeros((10, 10, 6), dtype=np.float32)
    for dest, sources in enumerate(CHANNELS[game]):
        for source in sources:
            result[:, :, dest] = np.maximum(result[:, :, dest], state[:, :, names[source]])
    return result.reshape(600)


def policy_logits(store):
    arrays = {name: store.load(name).numpy() for name in store.names}

    def forward(x):
        h1 = np.tanh(x @ arrays["W1"] + arrays["b1"])
        h2 = np.tanh(h1 @ arrays["W2"] + arrays["b2"])
        return h2 @ arrays["W3"] + arrays["b3"]

    return forward


def evaluate(game, store, *, episodes=1000, seed=20261103, max_steps=10000, env_factory=None):
    if game not in CHANNELS or type(episodes) is not int or episodes < 1:
        raise ValueError("Known game and positive episode count required")
    if type(seed) is not int or seed < 0 or type(max_steps) is not int or max_steps < 1:
        raise ValueError("Explicit nonnegative seed and positive step limit required")
    if env_factory is None:
        if importlib.metadata.version("minatar") != "1.0.15":
            raise RuntimeError("This scorer requires MinAtar 1.0.15")
        from minatar import Environment
        env_factory = Environment
    forward = policy_logits(store)
    records = []
    source_hash = None
    for episode in range(episodes):
        env = env_factory(game, sticky_action_prob=0.1, difficulty_ramping=True)
        env.seed(seed + episode)
        env.reset()
        if source_hash is None:
            source_hash = hashlib.sha256(inspect.getsource(type(env.env)).encode()).hexdigest()
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 1]))
        score, terminated = 0.0, False
        for step in range(1, max_steps + 1):
            logits = forward(observation(game, env.state(), env.env.channels)).astype(np.float64)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            action = int(rng.choice(6, p=probabilities))
            reward, terminated = env.act(action)
            score += float(reward)
            if terminated:
                break
        records.append({"episode": episode, "seed": seed + episode, "score": score,
                        "steps": step, "finish_reason": "terminated" if terminated else "step_limit"})
    values = np.asarray([row["score"] for row in records])
    return {"scorer": "openwsp-reconstructed-minatar-v1", "minatar_version": "1.0.15",
            "environment_class_sha256": source_hash, "game": game, "episodes": episodes, "seed": seed,
            "seed_matches_authors": False, "sticky_action_probability": 0.1, "max_steps": max_steps,
            "mean": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(episodes)) if episodes > 1 else None,
            "mean_steps": float(np.mean([row["steps"] for row in records])), "records": records}
