"""CPU Kinetix evaluation reconstructed from the Experiment 4 released rules.

Uses the released simulator commit and JAX version, never generated executable
code. Evaluation seeds were not published and are explicitly new here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

KINETIX_REVISION = "80ee9c292837c825cafaf9889323fdcc212efd1b"
JAX_VERSION = "0.4.38"
JAX2D_VERSION = "1.0.1"
ACTION_CARDINALITIES = (3, 3, 3, 3, 2, 2)
DEFAULT_SEED = 20261103


def actor_logits(observation, tensors, *, array_module=np):
    """The published 679→128→5×128→16 actor, with no value head."""
    xp = array_module
    hidden = xp.maximum(observation @ tensors["embed.W"] + tensors["embed.b"], 0)
    for layer in range(5):
        hidden = xp.tanh(hidden @ tensors[f"actor{layer}.W"] + tensors[f"actor{layer}.b"])
    return hidden @ tensors["actor_head.W"] + tensors["actor_head.b"]


def validate_tensors(tensors):
    expected = {"embed.W": (679, 128), "embed.b": (128,),
                "actor_head.W": (128, 16), "actor_head.b": (16,)}
    expected.update({f"actor{layer}.{suffix}": shape for layer in range(5)
                     for suffix, shape in (("W", (128, 128)), ("b", (128,)))})
    if set(tensors) != set(expected):
        raise ValueError("A complete actor with the published names is required")
    result = {}
    for name, shape in expected.items():
        value = np.asarray(tensors[name], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid actor tensor: {name}")
        result[name] = value
    return result


def source_provenance(source_root):
    root = Path(source_root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != KINETIX_REVISION:
        raise ValueError("Kinetix source must match the released training commit")
    paths = ["kinetix/levels/l/h2_car_ramp.json", "kinetix/environment/env.py",
             "kinetix/environment/spaces.py", "kinetix/render/renderer_symbolic_flat.py",
             "kinetix/render/renderer_symbolic_common.py", "pyproject.toml"]
    hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}
    dirty = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD", "--", *paths], text=True)
    if dirty.strip():
        raise ValueError("Pinned scorer source contains modified files")
    return {"repository": "https://github.com/FLAIROx/Kinetix", "revision": revision, "sha256": hashes}


def load_environment(source_root):
    """Create the exact released level with the prompt's explicit reward scale."""
    # Explicitly reject a previously initialized GPU backend; the task is CPU only.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.spaces import ActionType, ObservationType
    from kinetix.util.saving import load_from_json_file

    if (jax.__version__ != JAX_VERSION or importlib.metadata.version("jaxlib") != JAX_VERSION
            or importlib.metadata.version("jax2d") != JAX2D_VERSION):
        raise RuntimeError("Use JAX 0.4.38 and the pinned Kinetix dependency jax2d 1.0.1")
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("The raw-weight scorer must use CPU devices only")
    provenance = source_provenance(source_root)
    # Verify the installed implementation matches the source tree, even if the
    # editable-install path was not used.
    import kinetix.environment.env as env_module
    installed = Path(env_module.__file__).resolve().parents[1]
    for relative, expected in provenance["sha256"].items():
        if relative.startswith("kinetix/"):
            candidate = installed / relative.removeprefix("kinetix/")
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
                raise ValueError("Installed Kinetix differs from the pinned source")
    level = Path(source_root) / "kinetix/levels/l/h2_car_ramp.json"
    state, static, params = load_from_json_file(str(level.resolve()))
    if (static.num_polygons, static.num_circles, static.num_joints, static.num_thrusters,
            static.frame_skip, params.max_timesteps) != (12, 4, 6, 2, 2, 256):
        raise ValueError("Level dimensions or timing differ from the released prompt")
    if not np.isclose(params.dt, 1 / 60):
        raise ValueError("Unexpected physics timestep")
    # The saved level defaults to 0.1; the authors' prompt explicitly says 0.2.
    params = params.replace(dense_reward_scale=0.2)
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None,
                           env_params=params, static_env_params=static, auto_reset=False)
    obs = np.asarray(env.get_obs(state))
    if obs.shape != (679,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
        raise ValueError("Unexpected symbolic observation")
    if not (np.allclose(np.asarray(state.polygon.position[5]), (2.8, 0.725), atol=1e-6)
            and np.allclose(np.asarray(state.circle.position[0]), (2.537499666, 0.734999955), atol=1e-6)):
        raise ValueError("Car scene does not match the released level")
    provenance.update(jax=jax.__version__, jaxlib=importlib.metadata.version("jaxlib"),
                      jax2d=importlib.metadata.version("jax2d"), numpy=np.__version__,
                      backend="cpu", dense_reward_scale=0.2, frame_skip=2, max_steps=256,
                      observation_sha256=hashlib.sha256(obs.tobytes()).hexdigest())
    return env, state, params, provenance


def _make_batch_runner(env, initial_state, params, batch_size, *, mode="sample"):
    import jax
    import jax.numpy as jnp

    if mode not in ("sample", "argmax"):
        raise ValueError("mode must be sample or argmax")
    state0 = jax.tree.map(lambda value: jnp.broadcast_to(value, (batch_size,) + jnp.shape(value)), initial_state)
    obs0 = jnp.broadcast_to(env.get_obs(initial_state), (batch_size, 679))
    splits = np.cumsum((0,) + ACTION_CARDINALITIES)

    @jax.jit
    def run(tensors, keys):
        def step(carry, _):
            state, obs, keys, ended, outcomes, returns, lengths = carry
            all_keys = jax.vmap(lambda key: jax.random.split(key, 8))(keys)
            keys, step_keys = all_keys[:, 0], all_keys[:, 1]
            logits = actor_logits(obs, tensors, array_module=jnp)
            actions = []
            for i in range(6):
                part = logits[:, splits[i]:splits[i + 1]]
                action = (jax.vmap(jax.random.categorical)(all_keys[:, i + 2], part)
                          if mode == "sample" else jnp.argmax(part, axis=-1))
                actions.append(action)
            actions = jnp.stack(actions, axis=1)
            obs_new, state_new, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_keys, state, actions, params)
            finite = jax.tree.reduce(jnp.logical_and,
                                    jax.tree.map(lambda value: jnp.isfinite(value).reshape(batch_size, -1).all(axis=1), state_new))
            dense = jnp.where(state.last_distance == -1, 0.0,
                              -(info["distance"] - state.last_distance) * params.dense_reward_scale)
            dense = jnp.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)
            sparse_reward = reward - dense
            # Outcome codes: 1 success, 2 red, 3 invalid, 4 timeout. Red takes
            # precedence over green within a substep in the official simulator.
            outcome = jnp.where(sparse_reward < -0.5, 2,
                                jnp.where(info["GoalR"], 1, jnp.where(~finite, 3, 4)))
            finished = done | ~finite
            new_finish = ~ended & finished
            outcomes = jnp.where(new_finish, outcome, outcomes)
            returns = returns + jnp.where(ended, 0.0, reward)
            lengths = lengths + (~ended).astype(jnp.int32)
            ended = ended | finished
            return (state_new, obs_new, keys, ended, outcomes, returns, lengths), None

        carry = (state0, obs0, keys, jnp.zeros(batch_size, dtype=bool),
                 jnp.zeros(batch_size, dtype=jnp.int32), jnp.zeros(batch_size),
                 jnp.zeros(batch_size, dtype=jnp.int32))
        # Once every episode has terminated, subsequent transitions cannot
        # affect a score. Avoid simulating those discarded physics steps.
        def condition(value):
            count, current = value
            return (count < 256) & ~jnp.all(current[3])

        def advance(value):
            count, current = value
            updated, _ = step(current, None)
            return count + 1, updated

        _, final = jax.lax.while_loop(condition, advance, (jnp.array(0), carry))
        return final[4], final[5], final[6]

    return run


def evaluate(tensors, *, source_root, episodes=1024, seed=DEFAULT_SEED, batch_size=32, mode="sample"):
    """Evaluate fixed weights once, preserving every episode's result and seed."""
    if any(type(value) is not int or value < 1 for value in (episodes, batch_size)):
        raise ValueError("episodes and batch_size must be positive integers")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a uint32 integer")
    arrays = validate_tensors(tensors)
    env, initial_state, params, provenance = load_environment(source_root)
    import jax
    import jax.numpy as jnp

    arrays = {name: jnp.asarray(value) for name, value in arrays.items()}
    run = _make_batch_runner(env, initial_state, params, batch_size, mode=mode)
    root_key = jax.random.PRNGKey(seed)
    records, started = [], time.monotonic()
    names = {1: "success", 2: "red", 3: "invalid", 4: "timeout"}
    for offset in range(0, episodes, batch_size):
        indices = jnp.arange(offset, offset + batch_size, dtype=jnp.uint32)
        keys = jax.vmap(lambda index: jax.random.fold_in(root_key, index))(indices)
        outcomes, returns, lengths = jax.device_get(run(arrays, keys))
        for i in range(min(batch_size, episodes - offset)):
            records.append({"episode": offset + i, "outcome": names[int(outcomes[i])],
                            "return": float(returns[i]), "length": int(lengths[i])})
    successes = sum(row["outcome"] == "success" for row in records)
    fraction = successes / episodes
    return {"scorer": "openwsp-reconstructed-kinetix-v1", "provenance": provenance,
            "episodes": episodes, "batch_size": batch_size, "seed": seed, "seed_matches_authors": False,
            "seed_rule": "jax.random.fold_in(PRNGKey(seed), episode_index)", "mode": mode,
            "successes": successes, "success": fraction,
            "standard_error": float(np.sqrt(fraction * (1 - fraction) / episodes)),
            "red": sum(row["outcome"] == "red" for row in records) / episodes,
            "invalid": sum(row["outcome"] == "invalid" for row in records) / episodes,
            "timeout": sum(row["outcome"] == "timeout" for row in records) / episodes,
            "return": float(np.mean([row["return"] for row in records])),
            "length": float(np.mean([row["length"] for row in records])),
            "seconds": time.monotonic() - started, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True, help="Numeric actor NPZ; never executable code")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", choices=("sample", "argmax"), default="sample")
    args = parser.parse_args()
    with np.load(args.network, allow_pickle=False) as archive:
        tensors = dict(archive)
    result = evaluate(tensors, source_root=args.source_root, episodes=args.episodes,
                      seed=args.seed, batch_size=args.batch_size, mode=args.mode)
    result["network_sha256"] = hashlib.sha256(args.network.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "provenance")}))


if __name__ == "__main__":
    main()
