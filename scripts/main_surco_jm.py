"""This script trains a SurCo model to solve the PushT problem.

Definitions, single-step problem:
  - y: problem instance / context vector / Dim=11:
        [0:3]:   T_target_pose
        [4:6]:   T_pose
        [7:9]:   T_velocity
        [10:11]: pusher_xy

  - x: decision variable vector AKA 'action' / Dim=8
        [0:6]: face. One hot encoding of the face.
        [7]:   contact_point (0.2, 0.8)
        [8]:   angle (0.2*pi, 0.8*pi)

  - c: surrogate cost vector / Dim=8
        [0:6]: face logits (softmax → face weights for differentiable solver)
        [7:8]: contact_point logit, angle logit (sigmoid-scaled to bounds)

  - omega: set of feasible solutions.
        Action must be one hot in [0:6], action[6] in (0.2, 0.8),
        action[7] in (0.2*pi, 0.8*pi) for each step.

  - f(x; y): objective function / Dim=1
        The cost is the sum squared distance of the corners of the target block
        (p0, p3, p4, p7) to the target T's corners at the final timestep.


Surco methods:
- SurCo-zero:
    Gradient descent performed directly on c for a single fixed problem instance y.
- SurCo-prior:
    A neural network is trained to predict c conditioned on y, where y is drawn
    from the distribution of problem instances (random resets each iteration).
- SurCo-hybrid:
    Run SurCo-prior to get the NN y → c, then fine-tune c via gradient descent
    on a specific instance y.


System design (SurCo-prior):
1. NN: y → c  (8-dim surrogate cost vector)
2. Differentiable combinatorial solver: softmax(c[:6]) → face_weights
   Continuous params: sigmoid-scaled c[6] → contact_point, c[7] → angle
3. Rollout via step_pure_soft (differentiable physics)
4. Loss: mean sum-of-squared corner distances at final timestep
5. Backprop through rollout + solver + NN to update NN params


Run:
    python scripts/main_surco_jm.py --random-t-pose --verbosity 1
    python scripts/main_surco_jm.py --verbosity 1 --record-video
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
    """Maps context y (dim=11) → surrogate cost c (dim=8).

    c[:6] = face logits  (unbounded; softmax gives differentiable face weights)
    c[6]  = contact_point logit (sigmoid-scaled to CONTACT_POINT_BOUNDS)
    c[7]  = angle logit         (sigmoid-scaled to ANGLE_BOUNDS)

    Params: list of (W, b) tuples — plain JAX pytree, jit/grad compatible.
    """

    def __init__(self, context_dim: int, hidden_dims: Sequence[int] = (64, 64)):
        self.layer_sizes = [context_dim, *hidden_dims, 8]

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
        """Forward pass. Returns shape (N, 8) raw surrogate costs."""
        for i, (w, b) in enumerate(params):
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        return x


N_OPT_STEPS = 50
LR = 2.0
N_ENVS = 64
N_SIM_STEPS = 50
RESET_SEED = 0


def main(problem_type: str, verbosity: int, random_t_pose: bool, record_video: bool):
    assert problem_type in ["single_step"], "Only single step problem is supported for now."
    assert verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."

    env = PushTEnv(nenvs=N_ENVS, record_video=record_video, visualize=False)
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%d__%H:%M:%S")
    save_dir = Path(f"videos/{now}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # SurCo-prior: NN predicts 8-dim surrogate cost c conditioned on context y.
    mlp = MLP(context_dim=11, hidden_dims=(64, 64))
    params = mlp.init(jax.random.PRNGKey(0))

    lo_cp, hi_cp = CONTACT_POINT_BOUNDS
    lo_ang, hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])

    def cost(params, data):
        ctx = env.get_context_vector(data)  # (N_ENVS, 11)
        c = mlp.apply(params, ctx)          # (N_ENVS, 8) raw surrogate costs

        # Differentiable combinatorial solver: softmax relaxation over face logits.
        face_weights = jax.nn.softmax(c[:, :6], axis=-1)                        # (N_ENVS, 6)
        contact_point = lo_cp + (hi_cp - lo_cp) * jax.nn.sigmoid(c[:, 6])      # (N_ENVS,)
        angle = lo_ang + (hi_ang - lo_ang) * jax.nn.sigmoid(c[:, 7])           # (N_ENVS,)

        if verbosity > 1:
            jax.debug.print(
                "context  any_nan={n} min={lo:.3f} max={hi:.3f}",
                n=jnp.any(jnp.isnan(ctx)), lo=ctx.min(), hi=ctx.max(),
            )
            jax.debug.print("face_weights: {fw}", fw=face_weights)
            jax.debug.print("contact_point: {cp}", cp=contact_point)
            jax.debug.print("angle: {a}", a=angle)

        _, _, t_distances, jpos_traj = env.step_pure_soft(
            data=data,
            face_weights=face_weights,
            contact_point=contact_point,
            angle=angle,
            n_sim_steps=N_SIM_STEPS,
            check_t_displacement=False,
        )

        final_dists = t_distances[:, -1]
        loss = jnp.nanmean(final_dists)

        if verbosity > 0:
            jax.debug.print(
                "t_dist   sum(is_nan)={n} final_dists={d}",
                n=jnp.sum(jnp.isnan(final_dists)),
                d=final_dists,
            )
        return loss, (t_distances, jpos_traj, contact_point, angle, face_weights)

    cost_and_grad = jax.jit(jax.value_and_grad(cost, argnums=0, has_aux=True))

    print("SurCo-prior: training NN  y → c  (softmax face relaxation)")
    means = []
    stds = []
    initial_mean_dist = None
    t_start = time()

    for it in range(N_OPT_STEPS):
        if it == 0:
            print(f"Program loading time: {time() - PROGRAM_START_TIME:.2f} s")

        if random_t_pose:
            env.reset()
        else:
            env.reset(seed=RESET_SEED)

        t0 = time()
        env_data_0 = env.data
        (loss, (t_distances, jpos_traj, contact_point, angle, face_weights)), g_raw = cost_and_grad(
            params, env_data_0
        )

        n_bad_grads = sum(not jnp.all(jnp.isfinite(x)).item() for layer in g_raw for x in layer)
        g_params = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), g_raw)
        params = jax.tree.map(lambda p, g: p - LR * g, params, g_params)
        dt = time() - t0

        if initial_mean_dist is None:
            initial_mean_dist = loss

        print()
        print(
            f"===  iter {it + 1:2d}  ===  |  mean dist: {loss:.6f} [m] | "
            f"delta from initial: {100 * (loss - initial_mean_dist):.4f} [cm] | {dt * 1000:.1f} ms"
        )

        if n_bad_grads > 0:
            cprint(
                f"  WARNING: {n_bad_grads} non-finite values in raw gradients (sanitized to 0 for this step).",
                "red",
            )

        if verbosity > 0:
            grad_abs_values = [jnp.abs(g) for layer in g_params for g in layer]
            max_grad = max(jnp.max(g).item() for g in grad_abs_values)
            mean_abs_change = jnp.mean(
                jnp.array([jnp.mean(LR * jnp.abs(g)).item() for g in grad_abs_values])
            ).item()
            std_change = jnp.std(
                jnp.array([jnp.std(LR * jnp.abs(g)).item() for g in grad_abs_values])
            ).item()
            print(
                f"  max |grad|: {max_grad:.6f}, mean |change|: {mean_abs_change:.6f}, "
                f"std |change|: {std_change:.6f}"
            )

        final_dists_np = np.asarray(t_distances[:, -1])
        means.append(float(np.nanmean(final_dists_np)))
        stds.append(float(np.nanstd(final_dists_np)))

        if record_video:
            save_filepath = save_dir / f"{it:03d}.mp4"
            env.save_video_from_jpos_traj(save_filepath, np.asarray(jpos_traj))
            if verbosity > 1:
                print(f"  saved {save_filepath}")

        nan_envs = np.where(np.isnan(final_dists_np))[0].tolist()
        if nan_envs:
            for env_idx in nan_envs:
                print(f"  Env: {env_idx} - T distance is NaN")

        if it == 0:
            print(f"First iteration time: {time() - t_start:.2f} s")

    print(f"\nOptimization took {time() - t_start:.2f} s total")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"SurCo-prior  N_ENVS={N_ENVS}  N_SIM_STEPS={N_SIM_STEPS}  "
        f"N_OPT_STEPS={N_OPT_STEPS}  RANDOM_T_POSE={random_t_pose}"
    )
    ax1.axhline(initial_mean_dist, label="initial mean", color="black", linestyle="--")
    ax1.plot(means, label="mean")
    ax1.legend()
    ax1.set_title("Mean Distance")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Distance [m]")
    ax1.grid(True, alpha=0.3)

    ax2.plot(stds, label="std")
    ax2.legend()
    ax2.set_title("Standard Deviation")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Std [m]")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_filepath = save_dir / "surco_prior.png"
    plt.savefig(save_filepath, bbox_inches="tight")
    print(f"Saved plot to {save_filepath}")
    print(f"xdg-open {save_filepath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem_type", type=str, default="single_step", choices=["single_step", "multi_step"]
    )
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--random-t-pose", action="store_true", help="Randomize problem instances each iteration")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    main(args.problem_type, args.verbosity, args.random_t_pose, args.record_video)
