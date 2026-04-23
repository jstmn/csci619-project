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

jax.config.update("jax_enable_x64", True) #

N_ENVS = 100
N_SIM_STEPS = 50
N_OPT_STEPS = 100
LR_ACTION = 0.25  # for u_contact / u_angle (smooth)
LR_FACE = 1.0  # for face_logits — keep modest so distance gradient can steer face choice
LAMBDA_ENT = 0.005  # entropy reg — repels flat weights but weak enough to allow face switching
RESET_SEED = 0  # Same layout every reset; change or use ``seed + it`` for variety.


def main():
    env = PushTEnv(nenvs=N_ENVS, record_video=True, visualize=False)
    env.reset(seed=RESET_SEED)

    rng = np.random.default_rng(seed=1)
    init_face = rng.integers(0, 6, size=(N_ENVS,), dtype=np.int32)
    # Warm-start face_logits with a +FACE_BIAS on a random face per env. The
    # bias is a tradeoff: too low (~0) → near-uniform weights put the contact
    # body *inside* the T block and blow up spring-damper contacts; too high
    # (~5) → softmax saturates and the face gradient vanishes (Jacobian scales
    # like p(1-p)). 1.5 is peaked (p≈0.44) but leaves meaningful gradient flow.
    FACE_BIAS = 3.0
    face_logits = jnp.zeros((N_ENVS, 6), dtype=jnp.float64).at[jnp.arange(N_ENVS), jnp.asarray(init_face)].set(FACE_BIAS)

    # Unconstrained optimizer variables; sigmoid maps them into valid ranges.
    u_contact = jnp.zeros((N_ENVS,), dtype=jnp.float64)  # → contact = 0.5
    u_angle = jnp.zeros((N_ENVS,), dtype=jnp.float64)  # → angle = π/2

    def cost(face_logits, u_contact, u_angle, data0, target_xy):
        face_weights = jax.nn.softmax(face_logits, axis=-1)
        contact_point = jax.nn.sigmoid(u_contact)
        angle = jnp.pi * jax.nn.sigmoid(u_angle)
        _, _, t_distances, _ = env.step_pure_soft(
            data=data0,
            face_weights=face_weights,
            contact_point=contact_point,
            angle=angle,
            target_xy=target_xy,
            n_sim_steps=N_SIM_STEPS,
        )
        # Entropy = -Σ w log w  (max at uniform, 0 at one-hot). Add λ·H to cost
        # so the optimizer is repelled from flat weights — flat puts the
        # contact body inside the T and explodes the spring-damper.
        dist_sum = t_distances[:, -1].sum()
        entropy = -(face_weights * jnp.log(face_weights + 1e-12)).sum(axis=-1).sum()
        total = dist_sum + LAMBDA_ENT * entropy
        return total, (dist_sum, entropy)

    cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=(0, 1, 2), has_aux=True))

    print("Optimizing `face_logits`, `contact_point`, `angle` by gradient descent")
    t_start = time.time()
    for it in range(N_OPT_STEPS):
        env.reset(seed=RESET_SEED)
        data0 = env.data
        target_xy = jnp.asarray(env.target_poses[:, :2])

        t0 = time.time()
        (_, (dist_sum, entropy)), (g_face, g_contact, g_angle) = cost_and_grad(
            face_logits, u_contact, u_angle, data0, target_xy
        )
        face_logits = face_logits - LR_FACE * g_face
        u_contact = u_contact - LR_ACTION * g_contact
        u_angle = u_angle - LR_ACTION * g_angle
        dt = time.time() - t0
        if it == 0 or (it + 1) % 5 == 0:
            mean_dist = float(dist_sum) / N_ENVS
            mean_ent = float(entropy) / N_ENVS
            print(
                f"  iter {it + 1:4d}: mean dist = {mean_dist:.4f} m  "
                f"mean entropy = {mean_ent:.3f}  ({dt * 1000:.1f} ms)"
            )
    print(f"Optimization took {time.time() - t_start:.2f} s total")

    # Snap to hard argmax face for the visualization rollout.
    face_final = np.asarray(jnp.argmax(face_logits, axis=-1)).astype(np.int32).reshape(N_ENVS, 1)
    face_weights_final = jax.nn.softmax(face_logits, axis=-1)
    face_confidence = np.asarray(jnp.max(face_weights_final, axis=-1))
    contact_point = jax.nn.sigmoid(u_contact)
    angle = jnp.pi * jax.nn.sigmoid(u_angle)
    final_action = Action(
        face=face_final,
        contact_point=np.asarray(contact_point).reshape(N_ENVS, 1),
        angle=np.asarray(angle).reshape(N_ENVS, 1),
    )

    env.reset(seed=RESET_SEED)
    result = env.step(final_action, n_sim_steps=N_SIM_STEPS)
    t_distances = result.t_distances

    print("\nLearned per-env action and resulting final distance:")
    for i in range(N_ENVS):
        print(
            f"  env {i}: face={int(final_action.face[i, 0])} (p={float(face_confidence[i]):.2f})  "
            f"contact_point={float(final_action.contact_point[i, 0]):.3f}  "
            f"angle={float(final_action.angle[i, 0]):.3f}  "
            f"final_distance={float(t_distances[i, -1]):.4f} m"
        )

    env.save_video(Path("/tmp/learned_action.mp4"))


if __name__ == "__main__":
    main()
