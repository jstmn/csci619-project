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

  - c: surrogate solver parameters / Dim=8
        [0:6]: face logits  (fed to Gurobi; gradient via randomized smoothing)
        [6]:   contact_point target in CONTACT_POINT_BOUNDS
        [7]:   angle target in ANGLE_BOUNDS

  - omega: set of feasible solutions.
        x[0:6] must be one-hot, x[6] in CONTACT_POINT_BOUNDS, x[7] in ANGLE_BOUNDS.
        All constraints enforced by the Gurobi solve.

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
1. NN: y → c  (6 face logits + bounded targets for contact_point/angle)
2. Combinatorial solver: Gurobi MIQP
       argmin face_costs^T x_face
            + w_cp (cp - cp_target)^2
            + w_ang (ang - ang_target)^2
       s.t.
       sum(x[0:6])=1, x[0:6]∈{0,1}^6
       x[6] ∈ CONTACT_POINT_BOUNDS  (continuous)
       x[7] ∈ ANGLE_BOUNDS          (continuous)
   Gradient via Berthet et al. 2020 randomized smoothing through the solve.
3. Rollout via step_pure_soft (differentiable physics; one-hot face is valid input).
4. Loss: run one rollout using the clean Gurobi solution x* for the predicted
   face logits and continuous targets.
5. Backprop through the rollout, then use randomized smoothing only in the
   VJP to estimate gradients through the combinatorial solve before the NN.


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

VERBOSITY = 0
N_OPT_STEPS = 50
LR = 0.1
N_ENVS = 9
N_SIM_STEPS = 50
RESET_SEED = 0
N_FACES = 6
RANDOMZED_SMOOTHING_K = 10
OUTPUT_REG_BETA = 1e-3
CP_TARGET_WEIGHT = 1.0
ANG_TARGET_WEIGHT = 1.0

# Randomized smoothing scale: perturbed costs are c + λ ε, ε ~ N(0, I).
# Too small → perturbed solves often match x*; estimator variance high.
# Too large → x_k far from x*; gradient bias grows.
PERTURB_LAMBDA = 0.1


# ── Gurobi action solver ──────────────────────────────────────────────────────

_lo_cp, _hi_cp = CONTACT_POINT_BOUNDS
_lo_ang, _hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
_mid_cp = 0.5 * (_lo_cp + _hi_cp)
_mid_ang = 0.5 * (_lo_ang + _hi_ang)


class _PersistentActionSolver:
    """One Gurobi model, reused across every solve for speed.

    Variables:
        x[0:6]: binary one-hot face selection, sum == 1
        cp:     continuous contact_point in [lo_cp, hi_cp]
        ang:    continuous angle in [lo_ang, hi_ang]

    Input c is (8,): [face_costs(6), cp_target, ang_target]
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
        """Update objective from c (8,), solve, return x (8,)."""
        if not np.all(np.isfinite(c)):
            c = np.zeros(8, dtype=np.float32)
            c[6] = _mid_cp
            c[7] = _mid_ang
        cp_target = float(np.clip(c[6], _lo_cp, _hi_cp))
        ang_target = float(np.clip(c[7], _lo_ang, _hi_ang))
        face_obj = gp.quicksum(float(c[i]) * self.xf[i] for i in range(N_FACES))
        cp_obj = CP_TARGET_WEIGHT * (self.cp - cp_target) * (self.cp - cp_target)
        ang_obj = ANG_TARGET_WEIGHT * (self.ang - ang_target) * (self.ang - ang_target)
        self.model.setObjective(face_obj + cp_obj + ang_obj, GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()
        face_vals = np.array([self.xf[i].X for i in range(N_FACES)], dtype=np.float32)
        x = np.append(face_vals, [self.cp.X, self.ang.X]).astype(np.float32)
        assert x.shape == (8,), f"x must be (8,), got {x.shape}"
        return x


_SOLVER = _PersistentActionSolver()


def _gurobi_solve_batch(c_batch: np.ndarray) -> np.ndarray:
    """Solve the Gurobi objective for every env. Returns (N, 8)."""
    N = c_batch.shape[0]
    out = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        out[i] = _SOLVER.solve(c_batch[i])
    return out


def _solve_milp_pure_callback(c: jnp.ndarray) -> jnp.ndarray:
    """Forward-only JAX wrapper around _gurobi_solve_batch. c: (N,8) → x: (N,8)."""
    c = c.astype(jnp.float32)
    shape = jax.ShapeDtypeStruct(c.shape, jnp.float32)
    return jax.pure_callback(
        lambda cb_c: _gurobi_solve_batch(np.asarray(cb_c)).astype(np.float32),
        shape,
        c,
    )


@jax.custom_vjp
def milp_solver(c: jnp.ndarray, rng: jnp.ndarray, verbosity: int) -> jnp.ndarray:
    """Differentiable Gurobi solve: c (N,8) → x_star (N,8).

    Backward (Berthet et al. 2020): Monte Carlo over K_in = RANDOMZED_SMOOTHING_K draws
        ε_j ~ N(0, I),  x_j = solve(c + λ ε_j)
        ∂L/∂c[n,i] ≈ (1/(K_in λ)) Σ_j ε_j[n,i] · (grad_x[n] · x_j[n])
    where grad_x is ∂L/∂x from the same forward branch's rollout (see cost()).
    """
    return _solve_milp_pure_callback(c)


def _milp_forward(c, rng, verbosity):
    # Primal evaluation solves the Gurobi objective at the branch parameters c.
    # The K noisy solves used for the randomized-smoothing VJP are deferred to
    # _milp_backward and only happen during backpropagation.
    x_star = _solve_milp_pure_callback(c)
    rng, sample_rng = jax.random.split(rng)
    return x_star, (c, x_star, sample_rng, int(verbosity))


def _milp_backward(res, grad_x):
    c, _x_star, sample_rng, verbosity = res
    grad_x_safe = jnp.where(jnp.isfinite(grad_x), grad_x, 0.0).astype(jnp.float32)
    grad_c = jnp.zeros_like(c)
    key = sample_rng
    # Randomized-smoothing estimator for dL/dc: solve the perturbed objective K times
    # at c + lambda * eps_j, then average their contributions to the VJP.
    if verbosity > 0:
        jax.debug.print("  c={c}", c=c)
        jax.debug.print("\n")

    #
    for k_i in range(RANDOMZED_SMOOTHING_K):
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, c.shape, dtype=jnp.float32)
        c_pert = (c + PERTURB_LAMBDA * eps).astype(jnp.float32)
        x_k = _solve_milp_pure_callback(c_pert)
        face_k = jnp.argmax(x_k[:, :N_FACES], axis=-1)
        if verbosity > 0:
            jax.debug.print("\n  {k_i}", k_i=k_i)
            jax.debug.print("       c_pert={c_pert}", c_pert=c_pert)
            jax.debug.print(
                "       x_perturbed face={face} contact_point={cp} angle={a}",
                face=face_k,
                cp=x_k[:, 6],
                a=x_k[:, 7]
            )
        inner = jnp.sum(grad_x_safe * x_k, axis=-1, keepdims=True)  # (N, 1)
        grad_c = grad_c + eps * inner
    grad_c = grad_c / (RANDOMZED_SMOOTHING_K * PERTURB_LAMBDA)
    return (grad_c, None, None)


milp_solver.defvjp(_milp_forward, _milp_backward)


# ── MLP: y → c ────────────────────────────────────────────────────────────────


class MLP:
    """Maps context y (dim=11) → solver parameters c (dim=8).

    c[:6] = face cost coefficients
    c[6]  = contact_point target, squashed into CONTACT_POINT_BOUNDS
    c[7]  = angle target, squashed into ANGLE_BOUNDS

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
        """Forward pass. Returns face logits plus bounded continuous targets."""
        for i, (w, b) in enumerate(params):
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        face_logits = x[:, :N_FACES]
        cp_target = _lo_cp + (_hi_cp - _lo_cp) * jax.nn.sigmoid(x[:, 6:7])
        ang_target = _lo_ang + (_hi_ang - _lo_ang) * jax.nn.sigmoid(x[:, 7:8])
        return jnp.concatenate([face_logits, cp_target, ang_target], axis=-1)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(problem_type: str, verbosity: int, random_t_pose: bool, record_video: bool):
    assert problem_type in ["single_step"], "Only single step problem is supported for now."
    assert verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."

    env = PushTEnv(nenvs=N_ENVS, record_video=record_video, visualize=False)
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%d__%H:%M:%S")
    save_dir = Path(f"videos/{now}__{problem_type}__SurCo-prior")
    save_dir.mkdir(parents=True, exist_ok=True)

    mlp = MLP(context_dim=11, hidden_dims=(128, 128))
    params = mlp.init(jax.random.PRNGKey(0))

    def cost(params, data, rng_solve):
        ctx = env.get_context_vector(data)  # (N_ENVS, 11)
        c = mlp.apply(params, ctx)  # (N_ENVS, 8) face logits + bounded targets

        if verbosity > 1:
            jax.debug.print(
                "context  any_nan={n} min={lo:.3f} max={hi:.3f}",
                n=jnp.any(jnp.isnan(ctx)),
                lo=ctx.min(),
                hi=ctx.max(),
            )

        # Do one clean forward solve/rollout. The only smoothing Monte Carlo lives
        # in milp_solver's custom VJP, where it estimates dL/dc.
        x_star = milp_solver(c, rng_solve, verbosity)  # (N_ENVS, 8)
        face_onehot = x_star[:, :N_FACES]
        contact_point = x_star[:, 6]
        angle = x_star[:, 7]

        if verbosity > 0:
            face_idx = jnp.argmax(face_onehot, axis=-1)
            jax.debug.print(
                "rollout action\n  face={face}\n  contact_point={cp}\n  angle={a}",
                face=face_idx,
                cp=contact_point,
                a=angle,
            )

        # _, _, t_distances, jpos_traj = env.step_pure_soft(
        _, _, t_distances, jpos_traj = env.step_pure(
            data=data,
            face=face_onehot,
            contact_point=contact_point,
            angle=angle,
            n_sim_steps=N_SIM_STEPS,
            check_t_displacement=False,
        )

        final_dists = t_distances[:, -1]
        task_loss = jnp.nanmean(final_dists)
        # The face logits are scale-invariant up to ordering, so lightly
        # penalize them to keep smoothing effective. The targets are already
        # bounded by the sigmoid head.
        c_reg = OUTPUT_REG_BETA * jnp.mean(jnp.square(c[:, :N_FACES]))
        loss = task_loss + c_reg

        if verbosity > 0:
            jax.debug.print(
                "  sum(is_nan)={n} task_loss={task_loss:.6f} c_reg={c_reg:.6f}",
                n=jnp.sum(jnp.isnan(t_distances[:, -1])),
                task_loss=task_loss,
                c_reg=c_reg,
            )
        if verbosity > 1:
            jax.debug.print("  final_dists={d}", d=t_distances[:, -1])
        return loss, (t_distances, jpos_traj, face_onehot, contact_point, angle)

    # Do NOT wrap in jax.jit: pure_callback dispatches to Python per Gurobi
    # solve, so a JIT wrapper adds overhead without benefit. step_pure_soft is
    # already JIT'd internally, so physics stays compiled.
    cost_and_grad = jax.value_and_grad(cost, argnums=0, has_aux=True)

    print("SurCo-prior: training NN  y → solver params  (Gurobi + randomized-smoothing VJP)")

    means = []
    stds = []
    face_hist = []  # list of (N_ENVS,) int — argmax face per env per iter
    cp_hist = []  # list of (N_ENVS,) float — contact_point per env per iter
    ang_hist = []  # list of (N_ENVS,) float — angle per env per iter

    initial_mean_dist = None
    t_start = time()

    for it in range(N_OPT_STEPS):
        if it == 0:
            print(f"Program loading time: {time() - PROGRAM_START_TIME:.2f} s")

        print()
        print(f"|  ===  iter {it + 1:2d}  ===  |")

        if random_t_pose:
            env.reset()
        else:
            env.reset(seed=RESET_SEED)

        t0 = time()
        env_data_0 = env.data
        step_key = jax.random.PRNGKey(it)
        (loss, (t_distances, jpos_traj, face_onehot, cp_batch, ang_batch)), g_raw = cost_and_grad(
            params, env_data_0, step_key
        )

        n_bad_grads = sum(not jnp.all(jnp.isfinite(x)).item() for layer in g_raw for x in layer)
        g_params = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), g_raw)
        params = jax.tree.map(lambda p, g: p - LR * g, params, g_params)
        dt = time() - t0

        if initial_mean_dist is None:
            initial_mean_dist = loss

        print(f"mean dist: {loss:.6f} [m] | delta from initial: {100 * (loss - initial_mean_dist):.4f} [cm] | {dt * 1000:.1f} ms")

        if n_bad_grads > 0:
            cprint(
                f"  WARNING: {n_bad_grads} non-finite values in raw gradients (sanitized to 0 for this step).",
                "red",
            )

        if verbosity > 0:
            grad_abs_values = [jnp.abs(g) for layer in g_params for g in layer]
            max_grad = max(jnp.max(g).item() for g in grad_abs_values)
            mean_abs_change = jnp.mean(jnp.array([jnp.mean(LR * jnp.abs(g)).item() for g in grad_abs_values])).item()
            std_change = jnp.std(jnp.array([jnp.std(LR * jnp.abs(g)).item() for g in grad_abs_values])).item()
            cprint(f"  max |grad|: {max_grad:.6f}, mean |change|: {mean_abs_change:.6f}, std |change|: {std_change:.6f}", "yellow")

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
                cprint(f"  saved {save_filepath}", "green")

        nan_envs = np.where(np.isnan(final_dists_np))[0].tolist()
        if nan_envs:
            for env_idx in nan_envs:
                cprint(f"  Env: {env_idx} - T distance is NaN", "red")

        if it == 0:
            cprint(f"First iteration time: {time() - t_start:.2f} s", "yellow")

    cprint(f"\nOptimization took {time() - t_start:.2f} s total", "green")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"SurCo-prior  N_ENVS={N_ENVS}  N_SIM_STEPS={N_SIM_STEPS}  "
        f"N_OPT_STEPS={N_OPT_STEPS}  K={RANDOMZED_SMOOTHING_K}  λ={PERTURB_LAMBDA}  "
        f"RANDOM_T_POSE={random_t_pose}"
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

    for env_idx in range(min(N_ENVS, 3)):
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
    parser.add_argument("--problem_type", type=str, default="single_step", choices=["single_step", "multi_step"])
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--random-t-pose", action="store_true", help="Randomize problem instances each iteration")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    VERBOSITY = args.verbosity
    main(args.problem_type, args.verbosity, args.random_t_pose, args.record_video)
