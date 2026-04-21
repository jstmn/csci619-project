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

unset LD_LIBRARY_PATH; python scripts/optimize_action.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from pusht619.core import Action, PushTEnv

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


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

    def apply(self, params: list[tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. Returns shape (..., 2) — columns are [u_contact, u_angle]."""
        for i, (w, b) in enumerate(params):
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        return x


N_ENVS = 9
N_SIM_STEPS = 25
N_OPT_STEPS = 10
LR_ACTION = 0.25  # for u_contact / u_angle (smooth)
RESET_SEED = 0  # Same layout every reset; change or use ``seed + it`` for variety.


def main():
    env = PushTEnv(nenvs=N_ENVS, record_video=True, visualize=False)
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%H:%M:%S")
    video_dir = Path(f"videos/")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Context vector (dim=11): T_target_pose(3) | T_pose(3) | T_velocity(3) | pusher_xy(2)
    mlp = MLP(context_dim=11)
    params = mlp.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(seed=1)
    faces = jnp.asarray(rng.integers(0, 6, size=(N_ENVS, 1), dtype=np.int32))

    def cost(params, data):
        ctx = env.get_context_vector(data)
        out = mlp.apply(params, ctx)  # (N_ENVS, 2)
        contact_point = 0.1 + 0.8 * jax.nn.sigmoid(out[:, 0])  # CONTACT_POINT_BOUNDS = (0.1, 0.9)
        angle = jnp.pi / 6 + (4 * jnp.pi / 6) * jax.nn.sigmoid(out[:, 1])  # ANGLE_BOUNDS = (pi/6, 5*pi/6)

        jax.debug.print("context  any_nan={n} min={lo:.3f} max={hi:.3f}", n=jnp.any(jnp.isnan(ctx)), lo=ctx.min(), hi=ctx.max())
        jax.debug.print("mlp_out  any_nan={n}", n=jnp.any(jnp.isnan(out)))
        jax.debug.print("contact_point, raw:  {pre}", pre=out[:, 0])
        jax.debug.print("contact_point, post: {post}", post=contact_point)
        jax.debug.print("angle        , raw:  {pre}", pre=out[:, 1])
        jax.debug.print("angle        , post: {post}", post=angle)

        _, _, t_distances, jpos_traj = env.step_pure(
            data=data,
            face=faces[:, 0].astype(jnp.int32),
            contact_point=contact_point,
            angle=angle,
            n_sim_steps=N_SIM_STEPS,
        )

        jax.debug.print("t_dist   any_nan={n} final={d}",
            n=jnp.any(jnp.isnan(t_distances)), d=t_distances[:, -1])
        return t_distances[:, -1].mean(), (t_distances, jpos_traj, contact_point, angle)

    cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=0, has_aux=True))

    print("Optimizing MLP params by gradient descent")
    t_start = time.time()
    for it in range(N_OPT_STEPS):
        env.reset(seed=RESET_SEED)

        t0 = time.time()
        t_poses_0 = env.t_poses
        env_data_0 = env.data
        (loss, (t_distances, jpos_traj, contact_point, angle)), g_params = cost_and_grad(params, env_data_0)
        params = jax.tree.map(lambda p, g: p - LR_ACTION * g, params, g_params)
        dt = time.time() - t0
        print()
        print(f"  iter {it + 1:4d}: mean dist = {loss:.4f} m  ({dt * 1000:.1f} ms)")

        # Check for NaNs in gradients
        has_nan = any(jnp.any(jnp.isnan(g)).item() for layer in g_params for g in layer)
        if has_nan:
            print("  WARNING: NaN detected in gradients!")

        # Check gradient statistics
        grad_abs_values = [jnp.abs(g) for layer in g_params for g in layer]
        max_grad = max(jnp.max(g).item() for g in grad_abs_values)
        mean_abs_change = jnp.mean(jnp.array([jnp.mean(LR_ACTION * jnp.abs(g)).item() for g in grad_abs_values])).item()
        std_change = jnp.std(jnp.array([jnp.std(LR_ACTION * jnp.abs(g)).item() for g in grad_abs_values])).item()
        print(f"  max |grad|: {max_grad:.6f}, mean |change|: {mean_abs_change:.6f}, std |change|: {std_change:.6f}")

        # Save videos
        save_filepath = video_dir / f"{now}__learned_action_{it:03d}.mp4"
        env.save_video_from_jpos_traj(save_filepath, np.asarray(jpos_traj))
        print(f"  saved {save_filepath}")

        # On NaN: save debug JSON + per-failing-env video
        final_dists = np.asarray(t_distances[:, -1])
        nan_envs = np.where(np.isnan(final_dists))[0].tolist()
        if nan_envs:
            for env_idx in nan_envs:
                print(f"==== i: {env_idx} ====")
                print("  T_pose before:       ", t_poses_0[env_idx])
                print("  T_pose after:        ", env.t_poses[env_idx])
                print("  contact_poin:        ", contact_point[env_idx])
                print("  angle:               ", angle[env_idx])
                print("  pusher position_0:   ", env_data_0.base_position[env_idx])
                print("  pusher orientation_0:", env_data_0.base_orientation[env_idx])
                print("  joint positions_0:   ", env_data_0.joint_positions[env_idx])
                print("  joint velocities_0:  ", env_data_0.joint_velocities[env_idx])
            exit()

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
