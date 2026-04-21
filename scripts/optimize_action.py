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
python scripts/optimize_action.py --random-side --random-t-pose --record-video --verbosity 1
python scripts/optimize_action.py --verbosity 1 --record-video
"""

from __future__ import annotations

from time import time

PROGRAM_START_TIME = time()
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from termcolor import cprint
import jax

jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/jax_pusht619"))
import jax.numpy as jnp
import numpy as np
import argparse

from pusht619.core import PushTEnv, ANGLE_BOUNDS, CONTACT_POINT_BOUNDS


class MLP:
    """Maps a context vector to two unconstrained action parameters (u_contact, u_angle).

    Apply sigmoid / pi*sigmoid downstream to recover valid contact_point and angle.

    Params are a list of (W, b) tuples — a plain JAX pytree, jit/grad compatible.
    """

    def __init__(
        self, context_dim: int, output_ranges: list[tuple[float, float]], hidden_dims: Sequence[int] = (64, 64)
    ):
        self.output_ranges = output_ranges
        self.layer_sizes = [context_dim, *hidden_dims, 2]

    def init(self, key: jax.Array) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        params = []
        for i in range(len(self.layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            w = jax.random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros(fan_out, dtype=jnp.float32)
            params.append((w, b))
        return params

    def apply(self, params: list[tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. Returns shape (..., 2) — columns are [u_contact, u_angle]."""
        for i, (w, b) in enumerate(params):
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        for i, (lo, hi) in enumerate(self.output_ranges):
            x = x.at[:, i].set(lo + (hi - lo) * jax.nn.sigmoid(x[:, i]))
        return x


N_OPT_STEPS = 50
LR = 2.0
N_ENVS = 64
# N_ENVS = 9
N_SIM_STEPS = 50
RESET_SEED = 0


def main(random_side: bool, random_t_pose: bool, record_video: bool, verbosity: int):
    env = PushTEnv(nenvs=N_ENVS, record_video=record_video, visualize=False)
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%d__%H:%M:%S")
    save_dir = Path(f"videos/{now}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Context vector (dim=11): T_target_pose(3) | T_pose(3) | T_velocity(3) | pusher_xy(2)
    mlp = MLP(context_dim=11, output_ranges=[CONTACT_POINT_BOUNDS, ANGLE_BOUNDS])
    params = mlp.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(seed=RESET_SEED)

    def cost(params, data, faces):
        ctx = env.get_context_vector(data)
        out = mlp.apply(params, ctx)  # (N_ENVS, 2)
        contact_point, angle = out[:, 0], out[:, 1]

        if verbosity > 1:
            jax.debug.print(
                "context  any_nan={n} min={lo:.3f} max={hi:.3f}", n=jnp.any(jnp.isnan(ctx)), lo=ctx.min(), hi=ctx.max()
            )
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
        assert t_distances.shape == (N_ENVS, N_SIM_STEPS), (
            f"t_distances must be ({N_ENVS}, {N_SIM_STEPS}), got {t_distances.shape}"
        )
        # Mean final distance; NaNs from diverged rollouts are ignored (cannot use
        # jnp.where(cond)[0] here — dynamic nonzero is illegal inside jit).
        final_dists = t_distances[:, -1]
        loss = jnp.nanmean(final_dists)
        if verbosity > 0:
            jax.debug.print(
                "t_dist   sum(is_nan)={n} final_dists={d}", n=jnp.sum(jnp.isnan(final_dists)), d=final_dists
            )
        return loss, (t_distances, jpos_traj, contact_point, angle)

    cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=0, has_aux=True))

    print("Optimizing MLP params by gradient descent")
    means = []
    stds = []
    maxs = []
    mins = []
    angles = []
    contact_points = []
    initial_mean_dist = None
    t_start = time()
    faces = rng.integers(0, 6, size=(N_ENVS, 1), dtype=np.int32)

    for it in range(N_OPT_STEPS):
        if it == 0:
            print(f"Program loading time: {time() - PROGRAM_START_TIME:.2f} s")

        if random_side:
            faces = rng.integers(0, 6, size=(N_ENVS, 1), dtype=np.int32)

        if random_t_pose:
            env.reset()
        else:
            env.reset(seed=RESET_SEED)

        t0 = time()
        t_poses_0 = env.t_poses
        env_data_0 = env.data
        (loss, (t_distances, jpos_traj, contact_point, angle)), g_raw = cost_and_grad(params, env_data_0, faces)
        n_bad_grads = sum(not jnp.all(jnp.isfinite(x)).item() for layer in g_raw for x in layer)
        g_params = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), g_raw)
        params = jax.tree.map(lambda p, g: p - LR * g, params, g_params)
        dt = time() - t0
        if initial_mean_dist is None:
            initial_mean_dist = loss
        print()
        print(
            f"===  iter {it + 1:2d}  ===  |  mean dist: {loss:.6f} [m] | delta from initial: {100 * (loss - initial_mean_dist):.4f} [cm] | {dt * 1000:.1f} ms"
        )

        if n_bad_grads > 0:
            cprint(
                f"  WARNING: {n_bad_grads} non-finite values in raw gradients (sanitized to 0 for this step).", "red"
            )

        # Check gradient statistics
        if verbosity > 0:
            grad_abs_values = [jnp.abs(g) for layer in g_params for g in layer]
            max_grad = max(jnp.max(g).item() for g in grad_abs_values)
            mean_abs_change = jnp.mean(jnp.array([jnp.mean(LR * jnp.abs(g)).item() for g in grad_abs_values])).item()
            std_change = jnp.std(jnp.array([jnp.std(LR * jnp.abs(g)).item() for g in grad_abs_values])).item()
            print(f"  max |grad|: {max_grad:.6f}, mean |change|: {mean_abs_change:.6f}, std |change|: {std_change:.6f}")

        # Save stats
        means.append(t_distances[:, -1].mean())
        stds.append(t_distances[:, -1].std())
        maxs.append(t_distances[:, -1].max())
        mins.append(t_distances[:, -1].min())
        angles.append(np.asarray(angle))
        contact_points.append(np.asarray(contact_point))

        # Save videos
        if record_video:
            save_filepath = save_dir / f"{it:03d}.mp4"
            env.save_video_from_jpos_traj(save_filepath, np.asarray(jpos_traj))
            if verbosity > 1:
                print(f"  saved {save_filepath}")

        # On NaN: save debug JSON + per-failing-env video
        final_dists = np.asarray(t_distances[:, -1])
        nan_envs = np.where(np.isnan(final_dists))[0].tolist()
        if nan_envs:
            for env_idx in nan_envs:
                print(f"  Env: {env_idx} - T distance is NaN")
                print("    T_pose before:       ", t_poses_0[env_idx])
                print("    T_pose after:        ", env.t_poses[env_idx])
                print("    contact_poin:        ", contact_point[env_idx])
                print("    angle:               ", angle[env_idx])
                print("    pusher position_0:   ", env_data_0.base_position[env_idx])
                print("    pusher orientation_0:", env_data_0.base_orientation[env_idx])
                print("    joint positions_0:   ", env_data_0.joint_positions[env_idx])
                print("    joint velocities_0:  ", env_data_0.joint_velocities[env_idx])

        if it == 0:
            print(f"First iteration time: {time() - t_start:.2f} s")

    print(f"Optimization took {time() - t_start:.2f} s total")

    # ================================================
    # Plot results
    #
    # Determine layout based on randomization settings
    show_action_plots = not random_side and not random_t_pose
    if show_action_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax3 = axes[0, 0], axes[0, 1]
        ax2, ax4 = axes[1, 0], axes[1, 1]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(
        f"Optimization Results - N_ENVS={N_ENVS}, N_SIM_STEPS={N_SIM_STEPS}, N_OPT_STEPS={N_OPT_STEPS}, RANDOM_SIDE={random_side}, RANDOM_T_POSE={random_t_pose}"
    )
    # Left plot: mean with min and max
    ax1.axhline(initial_mean_dist, label="initial mean", color="black")
    ax1.plot(means, label="mean")
    ax1.legend()
    ax1.set_title("Mean Distance")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Distance [m]")
    ax1.grid(True, alpha=0.3)

    # Right plot: std
    ax2.plot(stds, label="std")
    ax2.legend()
    ax2.set_title("Standard Deviation")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Distance Standard Deviation [m]")
    ax2.grid(True, alpha=0.3)

    # If not randomizing, show angle and contact_point for first environment
    if show_action_plots:
        # Extract first environment's angle and contact_point over iterations
        # angles and contact_points are lists of arrays with shape (nenvs, 1)
        for env_idx in range(5):
            env_angles = [a[env_idx] for a in angles]
            env_contact_points = [cp[env_idx] for cp in contact_points]
            ax3.plot(env_angles, label=f"Angle [env {env_idx}]")
            ax4.plot(env_contact_points, label=f"Contact Point [env {env_idx}]")
        ax3.legend()
        ax3.set_title("Angle")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Angle [rad]")
        ax3.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_title("Contact Point")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Contact Point")
        ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    save_filepath = save_dir / "optimization.png"
    plt.savefig(save_filepath, bbox_inches="tight")
    print(f"Saved plot to {save_filepath}")
    print(f"xdg-open {save_filepath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-side", action="store_true", help="Randomize the side of the target object")
    parser.add_argument("--random-t-pose", action="store_true", help="Randomize the target pose")
    parser.add_argument("--record-video", action="store_true", help="Record a video of the optimization process")
    parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level")
    args = parser.parse_args()
    main(
        random_side=args.random_side,
        random_t_pose=args.random_t_pose,
        record_video=args.record_video,
        verbosity=args.verbosity,
    )
