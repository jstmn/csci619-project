"""Gradient-based action optimization through the PushT env.

Demonstrates the differentiable API with :class:`pusht619.core.Action` and
``env.step_pure``.  After each gradient step the environment is reset so the
cost is always the rollout from a fresh initial state (same distribution when
``reset(seed=...)`` is fixed).

TODOS:
1. [done] Discrete face made differentiable via softmax-weighted face points
    (``step_pure_soft``). Optimizer treats `face_logits` as free params and
    we snap to argmax for the final hard-face rollout.
2. Add a neural network to predict the action (contact_point, angle) from the state
    (t_poses, t_distances, joint_positions)
3. Randomize the target poses and see if the agent can learn to push from any configuration.


# Run it:

unset LD_LIBRARY_PATH
python scripts/optimize_action.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from pusht619.core import Action, PushTEnv 

jax.config.update("jax_enable_x64", True) #

class MLP:
    """Maps a context vector to two unconstrained action parameters (u_contact, u_angle).

    Apply sigmoid / pi*sigmoid downstream to recover valid contact_point and angle.

    Params are a list of (W, b) tuples — a plain JAX pytree, jit/grad compatible.
    """

    def __init__(self, context_dim: int, hidden_dims: Sequence[int] = (64, 64)):
        self.layer_sizes = [context_dim, *hidden_dims, 2]

    def init(self, key: jax.Array) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        params = []
        for i in range(len(self.layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            w = jax.random.normal(subkey, (fan_in, fan_out), dtype=jnp.float64) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros(fan_out, dtype=jnp.float64)
            params.append((w, b))
        return params

    def apply(
        self, params: list[tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass. Returns shape (..., 2) — columns are [u_contact, u_angle]."""
        for i, (w, b) in enumerate(params):
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        return x


N_ENVS = 100
N_SIM_STEPS = 50
N_OPT_STEPS = 100
LR_ACTION = 0.25  # for u_contact / u_angle (smooth)
RESET_SEED = 0  # Same layout every reset; change or use ``seed + it`` for variety.


def main():
    env = PushTEnv(nenvs=N_ENVS, record_video=True, visualize=False)
    env.reset(seed=RESET_SEED)

    # Context vector (dim=11): T_target_pose(3) | T_pose(3) | T_velocity(3) | pusher_xy(2)
    mlp = MLP(context_dim=11)
    params = mlp.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(seed=1)
    faces = jnp.asarray(rng.integers(0, 6, size=(N_ENVS, 1), dtype=np.int32))

    def cost(params, data):
        out = mlp.apply(params, env.get_context_vector(data))  # (N_ENVS, 2)
        contact_point = jax.nn.sigmoid(out[:, 0])
        angle = jnp.pi * jax.nn.sigmoid(out[:, 1])
        _, _, t_distances, jpos_traj = env.step_pure(
            data=data,
            face=faces[:, 0].astype(jnp.int32),
            contact_point=contact_point,
            angle=angle,
            n_sim_steps=N_SIM_STEPS,
        )
        return t_distances[:, -1].mean(), (t_distances, jpos_traj)

    cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=0, has_aux=True))

    print("Optimizing MLP params by gradient descent")
    t_start = time.time()
    for it in range(N_OPT_STEPS):
        env.reset(seed=RESET_SEED)

        t0 = time.time()
        (loss, (t_distances, jpos_traj)), g_params = cost_and_grad(params, env.data)
        params = jax.tree.map(lambda p, g: p - LR_ACTION * g, params, g_params)
        dt = time.time() - t0
        if it == 0 or (it + 1) % 5 == 0:
            print(f"  iter {it + 1:4d}: mean dist = {loss:.4f} m  ({dt * 1000:.1f} ms)")

        save_filepath = Path(f"learned_action_{it:03d}.mp4")
        env.save_video_from_jpos_traj(save_filepath, np.asarray(jpos_traj))
        print(f"  saved {save_filepath}")

    print(f"Optimization took {time.time() - t_start:.2f} s total")

    env.reset(seed=RESET_SEED)
    out = mlp.apply(params, env.get_context_vector(env.data))
    contact_point = jax.nn.sigmoid(out[:, 0])
    angle = jnp.pi * jax.nn.sigmoid(out[:, 1])
    final_action = Action(
        face=np.asarray(faces),
        contact_point=np.asarray(contact_point).reshape(N_ENVS, 1),
        angle=np.asarray(angle).reshape(N_ENVS, 1),
    )
    result = env.step(final_action, n_sim_steps=N_SIM_STEPS)
    t_distances = result.t_distances

    print("\nLearned per-env action and resulting final distance:")
    for i in range(N_ENVS):
        print(
            f"  env {i}: face={int(final_action.face[i, 0])}  "
            f"contact_point={float(final_action.contact_point[i, 0]):.3f}  "
            f"angle={float(final_action.angle[i, 0]):.3f}  "
            f"final_distance={float(t_distances[i, -1]):.4f} m"
        )

    env.save_video_from_jpos_traj(Path("/tmp/learned_action.mp4"), np.asarray(result.jpos_traj))


if __name__ == "__main__":
    main()
