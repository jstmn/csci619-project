"""This script trains a SurCo model to solve the PushT problem.

Definitions, single-step problem:
  - y: problem instance / context vector / Dim=9:
        [0:3]:   T_target_pose
        [4:6]:   T_pose
        [7:9]:   T_velocity
        # [10:11]: pusher_xy

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
1. NN: y → c  (6 face logits + per-face bounded targets for contact_point/angle)
2. Combinatorial solver: Gurobi MIQP
       argmin face_costs^T x_face
            + w_cp (cp - cp_ref(face))^2
            + w_ang (ang - ang_ref(face))^2
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


# --- Example usage
# - Note: poor performance observed with n-envs < 16. This is likely due to the gradients being too noisy.

# Random t-pose
python scripts/main_surco_jm.py --n-envs 100 --random-t-pose
python scripts/main_surco_jm.py --n-envs 64 --random-t-pose
python scripts/main_surco_jm.py --n-envs 16 --random-t-pose
python scripts/main_surco_jm.py --n-envs 25  --random-t-pose --verbosity 1 --record-video

# Fixed t-pose
python scripts/main_surco_jm.py --n-envs 25 --verbosity 1 --record-video
python scripts/main_surco_jm.py --n-envs 25 --verbosity 1 --record-video --multi-step-n-actions 2
"""

from __future__ import annotations
import os
from time import time

PROGRAM_START_TIME = time()
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from termcolor import cprint
import jax

jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/jax_pusht619"))
import jax.numpy as jnp
import numpy as np
import argparse

import optax
from pusht619.models import MLP, ActionSolver, ActionSolverMultiStep
from pusht619.core import PushTEnv, ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES


# ── Hyperparameters ───────────────────────────────────────────────────────────

N_OPT_STEPS = 500
LR = 0.01
N_SIM_STEPS = 25
RESET_SEED = 0
ACTION_DIM = NUM_FACES + 2
RANDOMZED_SMOOTHING_K = 20
FACE_OUTPUT_REG_BETA = 0.01
CP_TARGET_WEIGHT = 1.0
ANG_TARGET_WEIGHT = 1.0
RANDOM_T_POSE_EVAL_EVERY = 25

# Randomized smoothing scale: perturbed costs are c + λ ε, ε ~ N(0, I).
# Too small → perturbed solves often match x*; estimator variance high.
# Too large → x_k far from x*; gradient bias grows.
PERTURB_LAMBDA = 0.75


_SOLVER = ActionSolver()


def _configure_solver(multi_step_n_actions: int | None) -> None:
    global _SOLVER
    if multi_step_n_actions is None:
        _SOLVER = ActionSolver()
        return
    if multi_step_n_actions < 1:
        raise ValueError("multi_step_n_actions must be >= 1")
    _SOLVER = ActionSolverMultiStep(n_actions=multi_step_n_actions)


def _n_action_blocks(action_dim: int) -> int:
    if action_dim % ACTION_DIM != 0:
        raise ValueError(f"Action dimension must be divisible by {ACTION_DIM}, got {action_dim}")
    return action_dim // ACTION_DIM


def _gurobi_solve_batch(c_batch: np.ndarray) -> np.ndarray:
    """Solve the Gurobi objective for every env. Returns (N, action_dim)."""
    return _SOLVER.solve_batch(c_batch)


def _solve_milp_pure_callback(c: jnp.ndarray) -> jnp.ndarray:
    """Forward-only JAX wrapper around _gurobi_solve_batch. c: (N,D) → x: (N,D)."""
    c = c.astype(jnp.float32)
    shape = jax.ShapeDtypeStruct(c.shape, jnp.float32)
    return jax.pure_callback(
        lambda cb_c: _gurobi_solve_batch(np.asarray(cb_c)).astype(np.float32),
        shape,
        c,
    )


@jax.custom_vjp
def milp_solver(c: jnp.ndarray, rng: jnp.ndarray, verbosity: int) -> jnp.ndarray:
    """Differentiable Gurobi solve: c (N,D) → x_star (N,D).

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
    n_action_blocks = _n_action_blocks(c.shape[1])

    # Direct gradient for continuous outputs: x_star[NUM_FACES] = cp_target and
    # x_star[NUM_FACES + 1] = ang_target within each action block.
    grad_c_continuous = jnp.zeros_like(c)
    for action_idx in range(n_action_blocks):
        lo = action_idx * ACTION_DIM
        grad_c_continuous = grad_c_continuous.at[:, lo + NUM_FACES : lo + ACTION_DIM].set(
            grad_x_safe[:, lo + NUM_FACES : lo + ACTION_DIM]
        )

    # Randomized-smoothing estimator for face logits in each action block.
    grad_c_face = jnp.zeros_like(c)
    key = sample_rng
    if verbosity > 0:
        jax.debug.print("  c={c}", c=c)
        jax.debug.print("\n")

    x_star_blocks = _x_star.reshape(_x_star.shape[0], n_action_blocks, ACTION_DIM)
    angle_0 = x_star_blocks[:, :, NUM_FACES + 1]
    contact_point_0 = x_star_blocks[:, :, NUM_FACES]
    face_0 = jnp.argmax(x_star_blocks[:, :, :NUM_FACES], axis=-1)
    grad_x_face = grad_x_safe.reshape(grad_x_safe.shape[0], n_action_blocks, ACTION_DIM)[:, :, :NUM_FACES]

    for k_i in range(RANDOMZED_SMOOTHING_K):
        eps_face = jnp.zeros_like(c)
        for action_idx in range(n_action_blocks):
            key, subkey = jax.random.split(key)
            lo = action_idx * ACTION_DIM
            eps_face = eps_face.at[:, lo : lo + NUM_FACES].set(
                jax.random.normal(subkey, (c.shape[0], NUM_FACES), dtype=jnp.float32)
            )
        c_pert = (c + PERTURB_LAMBDA * eps_face).astype(jnp.float32)
        x_k = _solve_milp_pure_callback(c_pert)
        x_k_blocks = x_k.reshape(x_k.shape[0], n_action_blocks, ACTION_DIM)
        face_k = jnp.argmax(x_k_blocks[:, :, :NUM_FACES], axis=-1)
        inner = jnp.sum(grad_x_face * x_k_blocks[:, :, :NUM_FACES], axis=(1, 2), keepdims=True)  # (N, 1, 1)
        inner = inner.reshape(c.shape[0], 1)
        grad_c_face = grad_c_face + eps_face * inner

        #
        if verbosity > 1:
            jax.debug.print("\n  {k_i}", k_i=k_i)
            jax.debug.print("x_perturbed deltas")
            jax.debug.print("|__ face_0={face_0}", face_0=face_0)
            jax.debug.print("|__ face_k={face_k}", face_k=face_k)
            jax.debug.print("|__ d_face={d_face}", d_face=face_k - face_0)
            jax.debug.print("|__ cp=    {delta_cp}", delta_cp=x_k_blocks[:, :, NUM_FACES] - contact_point_0)
            jax.debug.print("|__ a=     {delta_a}", delta_a=x_k_blocks[:, :, NUM_FACES + 1] - angle_0)
    grad_c_face = grad_c_face / (RANDOMZED_SMOOTHING_K * PERTURB_LAMBDA)
    grad_c = grad_c_face + grad_c_continuous
    return (grad_c, None, None)


milp_solver.defvjp(_milp_forward, _milp_backward)


def plot_results(
    save_dir,
    means,
    stds,
    dist_delta_hist,
    face_hist,
    cp_hist,
    ang_hist,
    n_envs,
    n_sim_steps,
    n_opt_steps,
    random_t_pose,
    save_filepath=None,
    open_after_save=False,
):
    initial_mean_loss = means[0]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"SurCo-prior  n_envs={n_envs}  n_sim_steps={n_sim_steps}  "
        f"n_opt_steps={n_opt_steps}  K={RANDOMZED_SMOOTHING_K}  λ={PERTURB_LAMBDA}  "
        f"RANDOM_T_POSE={random_t_pose}"
    )
    ax_mean, ax_std = axes[0, 0], axes[0, 1]
    ax_cp, ax_ang = axes[1, 0], axes[1, 1]
    ax_face, ax_delta = axes[2, 0], axes[2, 1]

    ax_mean.axhline(float(initial_mean_loss), label="initial mean", color="black", linestyle="--")
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

    mean_delta = [float(np.nanmean(delta)) for delta in dist_delta_hist]
    ax_delta.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_delta.plot(mean_delta, color="black", linewidth=2.0, label="mean")
    for env_idx in range(min(n_envs, 7)):
        ax_delta.plot([delta[env_idx] for delta in dist_delta_hist], label=f"env {env_idx}", alpha=0.8)
    ax_delta.legend()
    ax_delta.set_title("Distance Change Per Env")
    ax_delta.set_xlabel("Iteration")
    ax_delta.set_ylabel("Delta from Iter 1 [m]")
    ax_delta.grid(True, alpha=0.3)

    for env_idx in range(min(n_envs, 5)):
        x1 = np.arange(len(cp_hist))
        p1 = ax_cp.plot([cp[env_idx] for cp in cp_hist], label=f"env {env_idx}", alpha=0.5)
        ax_cp.scatter(x1, [cp[env_idx] for cp in cp_hist], color=p1[0].get_color())
        p2 = ax_ang.plot([a[env_idx] for a in ang_hist], label=f"env {env_idx}", alpha=0.5)
        ax_ang.scatter(x1, [a[env_idx] for a in ang_hist], color=p2[0].get_color())
        p3 = ax_face.plot([f[env_idx] for f in face_hist], marker=".", linestyle="-", label=f"env {env_idx}", alpha=0.5)
        ax_face.scatter(x1, [f[env_idx] for f in face_hist], color=p3[0].get_color())

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

    ax_face.set_yticks(range(NUM_FACES))
    ax_face.legend()
    ax_face.set_title("Face Chosen (argmax)")
    ax_face.set_xlabel("Iteration")
    ax_face.set_ylabel("face index")
    ax_face.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_filepath is None:
        save_filepath = save_dir / "surco_prior.png"
    plt.savefig(save_filepath, bbox_inches="tight")
    print(f"Saved plot to {save_filepath}")
    plt.close()
    if open_after_save:
        print(f"xdg-open {save_filepath}")
        os.system(f"xdg-open {save_filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    problem_type: str,
    n_envs: int,
    verbosity: int,
    random_t_pose: bool,
    record_video: bool,
    multi_step_n_actions: int | None,
):
    assert problem_type in ["single_step", "multi_step"], "problem_type must be 'single_step' or 'multi_step'."
    assert verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."
    is_multi_step = multi_step_n_actions is not None
    n_action_blocks = multi_step_n_actions if is_multi_step else 1
    effective_problem_type = "multi_step" if is_multi_step else problem_type
    _configure_solver(multi_step_n_actions)

    env = PushTEnv(nenvs=n_envs, record_video=record_video, visualize=False)
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%d__%H:%M:%S")
    solver_output_dim = ACTION_DIM * n_action_blocks
    save_dir = Path(
        f"logs/{now}__{effective_problem_type}__n-envs:{n_envs}__SurCo-prior__Adam__{solver_output_dim}dim__lr:{LR}__grad-clip:1.0__random-t-pose:{random_t_pose}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"xdg-open {save_dir}")

    mlp = MLP(context_dim=9, hidden_dims=(128, 128), output_dim=solver_output_dim)
    params = mlp.init(jax.random.PRNGKey(0))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LR),
    )
    opt_state = optimizer.init(params)

    def cost(params, data, rng_solve):
        ctx = env.get_context_vector(data)  # (n_envs, 9)
        c = mlp.apply(params, ctx)  # (n_envs, action_dim)

        if verbosity > 1:
            jax.debug.print(
                "context  any_nan={n} min={lo:.3f} max={hi:.3f}",
                n=jnp.any(jnp.isnan(ctx)),
                lo=ctx.min(),
                hi=ctx.max(),
            )

        # Do one clean forward solve/rollout. The only smoothing Monte Carlo lives
        # in milp_solver's custom VJP, where it estimates dL/dc.
        x_star = milp_solver(c, rng_solve, verbosity)  # (n_envs, action_dim)
        x_star_blocks = x_star.reshape((x_star.shape[0], n_action_blocks, ACTION_DIM))
        c_blocks = c.reshape((c.shape[0], n_action_blocks, ACTION_DIM))
        face_weights = x_star_blocks[:, :, :NUM_FACES]
        contact_points = x_star_blocks[:, :, NUM_FACES]
        angles = x_star_blocks[:, :, NUM_FACES + 1]

        if verbosity > 0:
            face_idx = jnp.argmax(face_weights, axis=-1)
            jax.debug.print("rollout action")
            jax.debug.print(
                "|__ face={face}\n  |__ contact_point={cp}\n  |__ angle={a}",
                face=face_idx,
                cp=contact_points,
                a=angles,
            )
            jax.debug.print("|__")

        rollout_data = data
        t_distances_parts = []
        jpos_traj_parts = []
        for action_idx in range(n_action_blocks):
            rollout_data, _, t_distances_step, jpos_traj_step = env.step_pure_soft(
                data=rollout_data,
                face_weights=face_weights[:, action_idx, :],
                contact_point=contact_points[:, action_idx],
                angle=angles[:, action_idx],
                n_sim_steps=N_SIM_STEPS,
                check_t_displacement=False,
            )
            t_distances_parts.append(t_distances_step)
            jpos_traj_parts.append(jpos_traj_step)
        t_distances = jnp.concatenate(t_distances_parts, axis=1)
        jpos_traj = jnp.concatenate(jpos_traj_parts, axis=1)

        final_dists = t_distances[:, -1]
        task_loss = jnp.nanmean(final_dists)
        # The face logits are scale-invariant up to ordering, so lightly
        # penalize them to keep smoothing effective. The per-face targets are
        # already bounded by the sigmoid head.
        c_reg = FACE_OUTPUT_REG_BETA * jnp.mean(jnp.square(c_blocks[:, :, :NUM_FACES]))
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
        return loss, (t_distances, jpos_traj, face_weights, contact_points, angles)

    # Do NOT wrap in jax.jit: pure_callback dispatches to Python per Gurobi
    # solve, so a JIT wrapper adds overhead without benefit. step_pure_soft is
    # already JIT'd internally, so physics stays compiled.
    cost_and_grad = jax.value_and_grad(cost, argnums=0, has_aux=True)

    print("SurCo-prior: training NN  y → solver params  (Gurobi + randomized-smoothing VJP)")
    if n_envs < 16:
        cprint(
            f"WARNING: n_envs={n_envs} may produce noisy gradients; training is typically more stable with larger batches (e.g. 16+ envs).",
            "yellow",
        )

    means = []
    stds = []
    dist_delta_hist = []  # list of (n_envs,) float — change in final distance from iter 1 per env
    face_hist = []  # list of (n_envs,) int — argmax face per env per iter
    cp_hist = []  # list of (n_envs,) float — contact_point per env per iter
    ang_hist = []  # list of (n_envs,) float — angle per env per iter

    n_envs_better_0 = None
    initial_mean_loss = None
    initial_final_dists = None
    initial_faces = None
    t_start = time()

    for it in range(N_OPT_STEPS):
        if it == 0:
            print(f"Program loading time: {time() - PROGRAM_START_TIME:.2f} s")

        print()
        print(f"|  ===  iter {it + 1:2d}  ===  |")

        is_eval_step = (it % RANDOM_T_POSE_EVAL_EVERY) == 0
        if random_t_pose:
            if is_eval_step:
                env.reset(0)
            else:
                env.reset()
        else:
            env.reset(seed=RESET_SEED)

        t0 = time()
        env_data_0 = env.data
        step_key = jax.random.PRNGKey(it)
        (loss, (t_distances, jpos_traj, face_weights, cp_batch, ang_batch)), g_raw = cost_and_grad(
            params, env_data_0, step_key
        )
        final_dists_np = np.asarray(t_distances[:, -1])
        face_idx_np = np.asarray(jnp.argmax(face_weights, axis=-1))
        face_hist_current = face_idx_np[:, 0] if is_multi_step else face_idx_np
        cp_hist_current = np.asarray(cp_batch[:, 0] if is_multi_step else cp_batch)
        ang_hist_current = np.asarray(ang_batch[:, 0] if is_multi_step else ang_batch)
        g_params = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), g_raw)
        updates, opt_state = optimizer.update(g_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        dt = time() - t0

        # Print NaN environments / gradients
        #
        nan_envs = np.where(np.isnan(final_dists_np))[0].tolist()
        if nan_envs:
            for env_idx in nan_envs:
                cprint(f"  Env: {env_idx} - T distance is NaN", "red")
        n_bad_grads = sum(not jnp.all(jnp.isfinite(x)).item() for layer in g_raw for x in layer)
        if n_bad_grads > 0:
            cprint(
                f"WARNING: {n_bad_grads} non-finite values in raw gradients (sanitized to 0 for this step).",
                "red",
            )

        # Print results
        #
        if initial_mean_loss is None:
            initial_mean_loss = loss
        if initial_final_dists is None:
            initial_final_dists = final_dists_np.copy()
        if initial_faces is None:
            initial_faces = face_idx_np.copy()
        if random_t_pose:
            if is_eval_step:
                cprint(f"|____ eval step: using environment from iteration 0", "cyan")
        if n_envs_better_0 is None:
            n_envs_better_0 = sum(final_dists_np < initial_mean_loss - 0.05)
        delta_cm = 100 * (loss - initial_mean_loss)
        cprint(
            f"|____ mean dist: {loss:.5f} [m] | delta from initial: {delta_cm:.3f} [cm] | initial mean: {initial_mean_loss:.5f} [m] | {dt * 1000:.1f} ms",
            "green" if delta_cm < 0 else "red",
        )
        n_envs_better = sum(final_dists_np < initial_mean_loss - 0.05)
        cprint(
            f"|____ {n_envs_better} / {n_envs} envs are better than the initial mean, initial: {n_envs_better_0}",
            "green" if n_envs_better > n_envs_better_0 else "red",
        )
        cprint(f"|____ face initial={initial_faces.tolist()}", "yellow")
        cprint(f"|____ face current={face_idx_np.tolist()}", "yellow")
        cprint(f"|____ face diff=   {(face_idx_np - initial_faces).tolist()}", "yellow")
        cprint(f"|____ angles=      {ang_hist_current.tolist()}", "yellow")
        cprint(f"|____ cp=          {cp_hist_current.tolist()}", "yellow")

        # Print gradient statistics
        #
        if verbosity > 0:
            grad_abs_values = [jnp.abs(g) for layer in g_params for g in layer]
            max_grad = max(jnp.max(g).item() for g in grad_abs_values)
            mean_grad = jnp.mean(jnp.array([jnp.mean(g).item() for g in grad_abs_values]))
            cprint(f"|____ max |grad|: {max_grad:.6f}, mean |grad|: {mean_grad:.6f}", "yellow")

        # Log results for plotting
        #
        means.append(float(np.nanmean(final_dists_np)))
        stds.append(float(np.nanstd(final_dists_np)))
        dist_delta_hist.append(final_dists_np - initial_final_dists)
        face_hist.append(face_hist_current)
        cp_hist.append(cp_hist_current)
        ang_hist.append(ang_hist_current)

        # Save weights and snapshot plot
        #
        if (it + 1) % 5 == 0:
            filepath = checkpoints_dir / f"mlp_iter_{it + 1:03d}.npz"
            mlp.save_mlp_weights(filepath, params)
            cprint(f"|____ saved weights to {filepath}", "yellow")
            plot_results(
                save_dir,
                means,
                stds,
                dist_delta_hist,
                face_hist,
                cp_hist,
                ang_hist,
                n_envs,
                N_SIM_STEPS,
                N_OPT_STEPS,
                random_t_pose,
                save_filepath=save_dir / f"{it + 1:03d}.png",
                open_after_save=False,
            )

        # Save video
        #
        if record_video and it % 5 == 0:
            save_filepath = save_dir / f"{it + 1:03d}.mp4"
            env.save_video_from_jpos_traj(save_filepath, np.asarray(jpos_traj))
            print(f"  Saved video to {save_filepath}")

        if it == 0:
            cprint(f"First iteration time: {time() - t_start:.2f} s", "yellow")

    cprint(f"\nOptimization took {time() - t_start:.2f} s total", "green")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_results(
        save_dir,
        means,
        stds,
        dist_delta_hist,
        face_hist,
        cp_hist,
        ang_hist,
        n_envs,
        N_SIM_STEPS,
        N_OPT_STEPS,
        random_t_pose,
        open_after_save=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", type=str, default="single_step", choices=["single_step", "multi_step"])
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--random-t-pose", action="store_true", help="Randomize problem instances each iteration")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--multi-step-n-actions", type=int)
    args = parser.parse_args()
    assert args.verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."
    assert args.n_envs is not None, "n_envs must be specified"
    main(
        args.problem_type, args.n_envs, args.verbosity, args.random_t_pose, args.record_video, args.multi_step_n_actions
    )
