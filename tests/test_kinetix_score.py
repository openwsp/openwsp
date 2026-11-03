import numpy as np
import pytest

from replications.raw_weights.kinetix_score import (
    ACTION_CARDINALITIES,
    actor_logits,
    validate_tensors,
)


def actor():
    tensors = {"embed.W": np.zeros((679, 128), np.float32), "embed.b": np.zeros(128, np.float32),
               "actor_head.W": np.zeros((128, 16), np.float32), "actor_head.b": np.zeros(16, np.float32)}
    for layer in range(5):
        tensors[f"actor{layer}.W"] = np.eye(128, dtype=np.float32)
        tensors[f"actor{layer}.b"] = np.zeros(128, dtype=np.float32)
    return tensors


def test_published_actor_formula_orientation_relu_and_depth():
    tensors = actor()
    tensors["embed.W"][52, 6] = 2
    tensors["embed.W"][54, 7] = -1
    tensors["actor_head.W"][6, 2] = 3
    tensors["actor_head.W"][7, 0] = 100
    tensors["actor_head.b"][1] = 4
    x = np.zeros((2, 679), np.float32)
    x[0, 52], x[0, 54], x[1, 52] = 1, 2, -1
    got = actor_logits(x, tensors)
    expected = 2.0
    for _ in range(5):
        expected = np.tanh(expected)
    assert got.shape == (2, 16)
    assert got[0, 2] == pytest.approx(3 * expected)
    assert got[0, 0] == 0 and got[1, 2] == 0
    assert np.array_equal(got[:, 1], [4, 4])
    assert sum(ACTION_CARDINALITIES) == 16 and ACTION_CARDINALITIES == (3, 3, 3, 3, 2, 2)


def test_actor_validation_checks_all_shapes_and_finiteness():
    tensors = actor()
    tensors["embed.W"] = tensors["embed.W"].astype(np.float64)
    assert validate_tensors(tensors)["embed.W"].dtype == np.float32
    tensors["embed.W"] = np.zeros((128, 679), np.float32)
    with pytest.raises(ValueError, match="Invalid actor tensor"):
        validate_tensors(tensors)
    tensors = actor()
    tensors["actor_head.b"][0] = np.nan
    with pytest.raises(ValueError, match="Invalid actor tensor"):
        validate_tensors(tensors)
    tensors = actor()
    tensors["value_head.W"] = np.zeros((128, 1))
    with pytest.raises(ValueError, match="complete actor"):
        validate_tensors(tensors)


@pytest.mark.parametrize("terminal", ["success", "red", "invalid"])
def test_completed_episode_is_counted_once_while_other_episodes_continue(terminal):
    # Optional integration coverage runs in the pinned scorer environment. The
    # core suite needs no JAX installation for the formula/shape tests above.
    import os
    from collections import namedtuple

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from replications.raw_weights.kinetix_score import _make_batch_runner

    State = namedtuple("State", ["last_distance", "timestep", "target"])
    Params = namedtuple("Params", ["dense_reward_scale"])

    class FakeEnvironment:
        def get_obs(self, state):
            return jnp.zeros(679, jnp.float32)

        def step(self, key, state, action, params):
            target = jnp.where(state.timestep == 0, jax.random.randint(key, (), 1, 4), state.target)
            updated = State(jnp.array(-1.), state.timestep + 1, target)
            done = updated.timestep >= target
            sign = 1 if terminal == "success" else -1
            reward = sign * done.astype(jnp.float32)
            goal = done & (terminal == "success")
            if terminal == "invalid":
                updated = updated._replace(last_distance=jnp.where(done, jnp.nan, -1.))
                done, reward = jnp.array(False), jnp.array(0.)
            # Keep returning a terminal reward deliberately: the rollout must
            # retain only the first terminal transition.
            return self.get_obs(updated), updated, reward, done, {
                "GoalR": goal, "distance": jnp.array(0.)}

    run = _make_batch_runner(FakeEnvironment(), State(jnp.array(-1.), jnp.array(0), jnp.array(0)),
                             Params(0.2), 8, mode="argmax")
    keys = jax.random.split(jax.random.PRNGKey(4), 8)
    outcomes, returns, lengths = run({k: jnp.asarray(v) for k, v in actor().items()},
                                     keys)
    expected = jax.vmap(lambda key: jax.random.randint(jax.random.split(key, 8)[1], (), 1, 4))(keys)
    assert len(set(np.asarray(expected).tolist())) > 1
    np.testing.assert_array_equal(outcomes, np.full(8, {"success": 1, "red": 2, "invalid": 3}[terminal]))
    np.testing.assert_array_equal(returns, np.full(8, {"success": 1., "red": -1., "invalid": 0.}[terminal]))
    np.testing.assert_array_equal(lengths, expected)
