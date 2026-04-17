"""Gradient-based action optimization through the PushT env.

Demonstrates the differentiable API:

    data_new, t_poses, t_distances, _ = env.step_pure(
        data, face, contact_point, angle, target_xy, n_sim_steps
    )

For each environment we keep the discrete `face` fixed (ints → zero grad)
and use `jax.grad` to optimize the continuous parameters (`contact_point`,
`angle`) so that the T ends as close as possible to the target.

Run it:
    python scripts/optimize_action.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from pusht619.core import PushTEnv


N_ENVS = 9
N_SIM_STEPS = 100
N_OPT_STEPS = 30
LR = 0.25


def main():
    env = PushTEnv(nenvs=N_ENVS, record_video=False, visualize=False)
    env.reset(seed=0)

    data0 = env.data                                 # snapshot initial state
    target_xy = jnp.asarray(env.target_poses[:, :2])

    rng = np.random.default_rng(seed=1)
    face = jnp.asarray(rng.integers(0, 6, size=(N_ENVS,), dtype=np.int32))

    # Unconstrained optimizer variables; sigmoid maps them into valid ranges.
    u_contact = jnp.zeros((N_ENVS,), dtype=jnp.float64)  # → contact = 0.5
    u_angle = jnp.zeros((N_ENVS,), dtype=jnp.float64)    # → angle = π/2

    def cost(u_contact, u_angle):
        contact_point = jax.nn.sigmoid(u_contact)        # (0, 1)
        angle = jnp.pi * jax.nn.sigmoid(u_angle)         # (0, π)
        _, _, t_distances, _ = env.step_pure(
            data=data0,
            face=face,
            contact_point=contact_point,
            angle=angle,
            target_xy=target_xy,
            n_sim_steps=N_SIM_STEPS,
        )
        # Sum-of-final-distances; scalar → we can take `jax.grad` of it.
        return t_distances[:, -1].sum()

    cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=(0, 1)))

    print("Optimizing `contact_point` and `angle` by gradient descent …")
    t_start = time.time()
    for it in range(N_OPT_STEPS):
        t0 = time.time()
        loss, (g_contact, g_angle) = cost_and_grad(u_contact, u_angle)
        # Plain SGD step
        u_contact = u_contact - LR * g_contact
        u_angle = u_angle - LR * g_angle
        dt = time.time() - t0
        if it == 0 or (it + 1) % 5 == 0:
            mean_dist = float(loss) / N_ENVS
            print(
                f"  iter {it + 1:3d}: mean final-distance = {mean_dist:.4f} m  "
                f"(sum = {float(loss):.4f}, {dt * 1000:.1f} ms)"
            )
    print(f"Optimization took {time.time() - t_start:.2f} s total")

    contact_point = jax.nn.sigmoid(u_contact)
    angle = jnp.pi * jax.nn.sigmoid(u_angle)

    _, _, t_distances, _ = env.step_pure(
        data=data0,
        face=face,
        contact_point=contact_point,
        angle=angle,
        target_xy=target_xy,
        n_sim_steps=N_SIM_STEPS,
    )
    print("\nLearned per-env action and resulting final distance:")
    for i in range(N_ENVS):
        print(
            f"  env {i}: face={int(face[i])}  "
            f"contact_point={float(contact_point[i]):.3f}  "
            f"angle={float(angle[i]):.3f}  "
            f"final_distance={float(t_distances[i, -1]):.4f} m"
        )


if __name__ == "__main__":
    main()
