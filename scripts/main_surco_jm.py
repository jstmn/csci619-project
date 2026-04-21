"""This script trains a SurCo model to solve the PushT problem.

Definitions, single-step problem:
  - y: problem instance / context vector / Dim=11:
        [0:3]:   T_target_pose
        [4:6]:   T_pose
        [7:9]:   T_velocity
        [10:11]: pusher_xy

  - x: decision variable vector AKA 'action' / Dim=8
        [0:6]: face. One hot encoding of the face.
        [6]:   contact_point in CONTACT_POINT_BOUNDS = (0.2, 0.8)
        [7]:   angle in ANGLE_BOUNDS = (0.2*pi, 0.8*pi)

  - c: surrogate cost vector / Dim=8
        [0:6]: face logits  (fed to Gurobi MILP; gradient via Pogančić)
        [6]:   contact_point cost coefficient
        [7]:   angle cost coefficient

  - omega: set of feasible solutions.
        x[0:6] must be one-hot, x[6] in CONTACT_POINT_BOUNDS, x[7] in ANGLE_BOUNDS.
        All constraints enforced by the MILP.

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
2. Combinatorial solver: Gurobi MILP  argmin c^T x  s.t.
       sum(x[0:6])=1, x[0:6]∈{0,1}^6
       x[6] ∈ CONTACT_POINT_BOUNDS  (continuous)
       x[7] ∈ ANGLE_BOUNDS          (continuous)
   Gradient via Pogančić et al. 2019 blackbox differentiation.
3. Rollout via step_pure_soft (differentiable physics; one-hot face is valid input).
4. Loss: nanmean of sum-of-squared corner distances at final timestep.
5. Backprop through rollout + Pogančić VJP + NN to update NN params.


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
import gurobipy as gp
from gurobipy import GRB

from pusht619.core import PushTEnv, ANGLE_BOUNDS, CONTACT_POINT_BOUNDS


# ── Hyperparameters ───────────────────────────────────────────────────────────

N_OPT_STEPS = 50
LR = 1.0
N_ENVS = 4
N_SIM_STEPS = 50
RESET_SEED = 0
N_FACES = 6

# Pogančić perturbation magnitude.
# Too small → solver returns same x* for c and c', gradient ≈ 0.
# Too large → gradient biased toward a faraway solution.
PERTURB_LAMBDA = 0.5


# ── Gurobi MILP action solver ─────────────────────────────────────────────────

_lo_cp, _hi_cp = CONTACT_POINT_BOUNDS
_lo_ang, _hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])


class _PersistentActionSolver:
    """One Gurobi model, reused across every solve for speed.

    Variables:
        x[0:6]: binary one-hot face selection, sum == 1
        cp:     continuous contact_point in [lo_cp, hi_cp]
        ang:    continuous angle in [lo_ang, hi_ang]

    Input c is (8,): [face_costs(6), cp_cost, ang_cost]
    Output x is (8,): [face_onehot(6), cp, ang]
    """

    def __init__(self):
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model(env=self.env)
        self.model.setParam("Threads", 1)
        self.model.setParam("Presolve", 0)
        self.xf = self.model.addVars(N_FACES, vtype=GRB.BINARY, name="xf")
        self.cp = self.model.addVar(lb=_lo_cp, ub=_hi_cp, vtype=GRB.CONTINUOUS, name="cp")
        self.ang = self.model.addVar(lb=_lo_ang, ub=_hi_ang, vtype=GRB.CONTINUOUS, name="ang")
        self.model.addConstr(gp.quicksum(self.xf[i] for i in range(N_FACES)) == 1)
        self.model.update()

    def solve(self, c: np.ndarray) -> np.ndarray:
        """Update objective coefficients from c (8,), solve, return x (8,)."""
        if not np.all(np.isfinite(c)):
            c = np.zeros(8, dtype=np.float32)
        for i in range(N_FACES):
            self.xf[i].Obj = float(c[i])
        self.cp.Obj = float(c[6])
        self.ang.Obj = float(c[7])
        self.model.update()
        self.model.optimize()
        face_vals = np.array([self.xf[i].X for i in range(N_FACES)], dtype=np.float32)
        return np.append(face_vals, [self.cp.X, self.ang.X]).astype(np.float32)


_SOLVER = _PersistentActionSolver()


def _gurobi_solve_batch(c_batch: np.ndarray) -> np.ndarray:
    """Solve the MILP for every env. Returns (N, 8): [face_onehot(6), cp, ang]."""
    N = c_batch.shape[0]
    out = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        out[i] = _SOLVER.solve(c_batch[i])
    return out


def _solve_pure_callback(c: jnp.ndarray) -> jnp.ndarray:
    """Forward-only JAX wrapper around _gurobi_solve_batch. c: (N,8) → x: (N,8)."""
    c = c.astype(jnp.float32)
    shape = jax.ShapeDtypeStruct(c.shape, jnp.float32)
    return jax.pure_callback(
        lambda cb_c: _gurobi_solve_batch(np.asarray(cb_c)).astype(np.float32),
        shape,
        c,
    )


@jax.custom_vjp
def milp_solver(c: jnp.ndarray) -> jnp.ndarray:
    """Differentiable Gurobi: c (N,8) → x_star (N,8).

    Gradient via Pogančić et al. 2019:
        x_prime = solve(c + λ · grad_x L)
        ∂L/∂c   = -(x_star - x_prime) / λ
    """
    return _solve_pure_callback(c)


def _milp_fwd(c):
    x_star = _solve_pure_callback(c)
    return x_star, (c, x_star)


def _milp_bwd(res, grad_x):
    c, x_star = res
    grad_x_safe = jnp.where(jnp.isfinite(grad_x), grad_x, 0.0).astype(jnp.float32)
    c_prime = (c + PERTURB_LAMBDA * grad_x_safe).astype(jnp.float32)
    x_prime = _solve_pure_callback(c_prime)
    grad_c = -(x_star - x_prime) / PERTURB_LAMBDA
    return (grad_c,)


milp_solver.defvjp(_milp_fwd, _milp_bwd)


# ── MLP: y → c ────────────────────────────────────────────────────────────────


class MLP:
    """Maps context y (dim=11) → surrogate cost c (dim=8).

    c[:6] = face cost coefficients  (fed to Gurobi MILP; Pogančić gradient)
    c[6]  = contact_point cost coefficient
    c[7]  = angle cost coefficient

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


# ── Main ──────────────────────────────────────────────────────────────────────


def main(problem_type: str, verbosity: int, random_t_pose: bool, record_video: bool):
    assert problem_type in ["single_step"], "Only single step problem is supported for now."
    assert verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."

    env = PushTEnv(nenvs=N_ENVS, record_video=record_video, visualize=False)
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%d__%H:%M:%S")
    save_dir = Path(f"videos/{now}")
    save_dir.mkdir(parents=True, exist_ok=True)

    mlp = MLP(context_dim=11, hidden_dims=(128, 128))
    params = mlp.init(jax.random.PRNGKey(0))

    def cost(params, data):
        ctx = env.get_context_vector(data)  # (N_ENVS, 11)
        c = mlp.apply(params, ctx)          # (N_ENVS, 8) raw surrogate costs

        # MILP: argmin c^T x s.t. one-hot face + continuous bounds (all enforced inside Gurobi).
        x = milp_solver(c)                  # (N_ENVS, 8)
        face_onehot = x[:, :N_FACES]        # (N_ENVS, 6)
        contact_point = x[:, 6]             # (N_ENVS,)
        angle = x[:, 7]                     # (N_ENVS,)

        if verbosity > 1:
            jax.debug.print(
                "context  any_nan={n} min={lo:.3f} max={hi:.3f}",
                n=jnp.any(jnp.isnan(ctx)), lo=ctx.min(), hi=ctx.max(),
            )
            jax.debug.print("contact_point: {cp}", cp=contact_point)
            jax.debug.print("angle: {a}", a=angle)

        # step_pure_soft accepts one-hot face_weights — one-hot is valid input.
        _, _, t_distances, jpos_traj = env.step_pure_soft(
            data=data,
            face_weights=face_onehot,
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
        return loss, (t_distances, jpos_traj, face_onehot, contact_point, angle)

    # Do NOT wrap in jax.jit: pure_callback dispatches to Python per Gurobi
    # solve, so a JIT wrapper adds overhead without benefit. step_pure_soft is
    # already JIT'd internally, so physics stays compiled.
    cost_and_grad = jax.value_and_grad(cost, argnums=0, has_aux=True)

    print("SurCo-prior: training NN  y → c  (Gurobi MILP + Pogančić VJP)")

    means = []
    stds = []
    face_hist = []       # list of (N_ENVS,) int — argmax face per env per iter
    cp_hist = []         # list of (N_ENVS,) float — contact_point per env per iter
    ang_hist = []        # list of (N_ENVS,) float — angle per env per iter

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
        (loss, (t_distances, jpos_traj, face_onehot, cp_batch, ang_batch)), g_raw = cost_and_grad(params, env_data_0)

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
        face_hist.append(np.asarray(jnp.argmax(face_onehot, axis=-1)))  # (N_ENVS,)
        cp_hist.append(np.asarray(cp_batch))
        ang_hist.append(np.asarray(ang_batch))

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
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"SurCo-prior  N_ENVS={N_ENVS}  N_SIM_STEPS={N_SIM_STEPS}  "
        f"N_OPT_STEPS={N_OPT_STEPS}  RANDOM_T_POSE={random_t_pose}"
    )
    ax_mean, ax_std = axes[0, 0], axes[0, 1]
    ax_cp, ax_ang = axes[1, 0], axes[1, 1]
    ax_face = axes[2, 0]
    axes[2, 1].set_visible(False)

    ax_mean.axhline(float(initial_mean_dist), label="initial mean", color="black", linestyle="--")
    ax_mean.plot(means, label="mean")
    ax_mean.legend()
    ax_mean.set_title("Mean Distance")
    ax_mean.set_xlabel("Iteration")
    ax_mean.set_ylabel("Distance [m]")
    ax_mean.grid(True, alpha=0.3)

    ax_std.plot(stds, label="std")
    ax_std.legend()
    ax_std.set_title("Standard Deviation")
    ax_std.set_xlabel("Iteration")
    ax_std.set_ylabel("Std [m]")
    ax_std.grid(True, alpha=0.3)

    for env_idx in range(N_ENVS):
        ax_cp.plot([cp[env_idx] for cp in cp_hist], label=f"env {env_idx}")
        ax_ang.plot([a[env_idx] for a in ang_hist], label=f"env {env_idx}")
        ax_face.plot([f[env_idx] for f in face_hist], marker=".", linestyle="-", label=f"env {env_idx}")

    lo_cp, hi_cp = CONTACT_POINT_BOUNDS
    lo_ang, hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
    ax_cp.axhline(lo_cp, color="gray", linestyle="--", linewidth=0.8)
    ax_cp.axhline(hi_cp, color="gray", linestyle="--", linewidth=0.8)
    ax_cp.legend()
    ax_cp.set_title("Contact Point")
    ax_cp.set_xlabel("Iteration")
    ax_cp.set_ylabel("contact_point")
    ax_cp.grid(True, alpha=0.3)

    ax_ang.axhline(lo_ang, color="gray", linestyle="--", linewidth=0.8)
    ax_ang.axhline(hi_ang, color="gray", linestyle="--", linewidth=0.8)
    ax_ang.legend()
    ax_ang.set_title("Angle")
    ax_ang.set_xlabel("Iteration")
    ax_ang.set_ylabel("angle [rad]")
    ax_ang.grid(True, alpha=0.3)

    ax_face.set_yticks(range(N_FACES))
    ax_face.legend()
    ax_face.set_title("Face Chosen (argmax)")
    ax_face.set_xlabel("Iteration")
    ax_face.set_ylabel("face index")
    ax_face.grid(True, alpha=0.3)

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
