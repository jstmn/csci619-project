"""Gradient-based action optimization through the PushT env.

Demonstrates the differentiable API with :class:`pusht619.core.Action` and
``env.step_pure``.  After each gradient step the environment is reset so the
cost is always the rollout from a fresh initial state (same distribution when
``reset(seed=...)`` is fixed).

TODOS:
1. Add parameter for the face (how to do this exactly? face needs to be discrete, but we want to optimize it)
2. Add a neural network to predict the action (contact_point, angle) from the state
    (t_poses, t_distances, joint_positions)
3. Randomize the target poses and see if the agent can learn to push from any configuration.


Run it:
    python scripts/optimize_action.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pusht619.core import Action, PushTEnv


N_ENVS = 100
N_SIM_STEPS = 50
N_OPT_STEPS = 1000
LR = 0.25
RESET_SEED = 0  # Same layout every reset; change or use ``seed + it`` for variety.


def main():
    env = PushTEnv(nenvs=N_ENVS, record_video=True, visualize=False)
    env.reset(seed=RESET_SEED)

    rng = np.random.default_rng(seed=1)
    face_arr = jnp.asarray(rng.integers(0, 6, size=(N_ENVS, 1), dtype=np.int32))

    # Unconstrained optimizer variables; sigmoid maps them into valid ranges.
    u_contact = jnp.zeros((N_ENVS,), dtype=jnp.float64)  # → contact = 0.5
    u_angle = jnp.zeros((N_ENVS,), dtype=jnp.float64)  # → angle = π/2

    def cost(u_contact, u_angle, data0, target_xy):
        contact_point = jax.nn.sigmoid(u_contact).reshape(N_ENVS, 1)
        angle = (jnp.pi * jax.nn.sigmoid(u_angle)).reshape(N_ENVS, 1)
        action = Action(face=face_arr, contact_point=contact_point, angle=angle)
        _, _, t_distances, _ = env.step_pure(
            data=data0,
            face=action.face[:, 0].astype(jnp.int32),
            contact_point=action.contact_point[:, 0],
            angle=action.angle[:, 0],
            target_xy=target_xy,
            n_sim_steps=N_SIM_STEPS,
        )
        return t_distances[:, -1].sum()

    cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=(0, 1)))

    print("Optimizing `contact_point` and `angle` by gradient descent")
    t_start = time.time()
    for it in range(N_OPT_STEPS):
        env.reset(seed=RESET_SEED)
        data0 = env.data
        target_xy = jnp.asarray(env.target_poses[:, :2])

        t0 = time.time()
        loss, (g_contact, g_angle) = cost_and_grad(u_contact, u_angle, data0, target_xy)
        u_contact = u_contact - LR * g_contact
        u_angle = u_angle - LR * g_angle
        dt = time.time() - t0
        if it == 0 or (it + 1) % 5 == 0:
            mean_dist = float(loss) / N_ENVS
            print(
                f"  iter {it + 1:4d}: mean final-distance = {mean_dist:.4f} m  "
                f"(sum = {float(loss):.4f}, {dt * 1000:.1f} ms)"
            )
    print(f"Optimization took {time.time() - t_start:.2f} s total")

    contact_point = jax.nn.sigmoid(u_contact)
    angle = jnp.pi * jax.nn.sigmoid(u_angle)
    final_action = Action(
        face=np.asarray(face_arr),
        contact_point=np.asarray(contact_point).reshape(N_ENVS, 1),
        angle=np.asarray(angle).reshape(N_ENVS, 1),
    )

    env.reset(seed=RESET_SEED)
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

    env.save_video(Path("/tmp/learned_action.mp4"))


if __name__ == "__main__":
    main()
