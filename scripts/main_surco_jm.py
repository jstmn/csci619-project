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


# Example usage:
# - Note: poor performance observed with n-envs < 16. This is likely due to the gradients being too noisy.
python scripts/main_surco_jm.py --n-envs 100 --random-t-pose
python scripts/main_surco_jm.py --n-envs 16 --verbosity 1 --record-video

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

from pusht619.models import MLP, ActionSolver
from pusht619.core import PushTEnv, ANGLE_BOUNDS, CONTACT_POINT_BOUNDS


# ── Hyperparameters ───────────────────────────────────────────────────────────

N_OPT_STEPS = 200
LR = 0.1
N_SIM_STEPS = 50
RESET_SEED = 0
N_FACES = 6
RANDOMZED_SMOOTHING_K = 20
# RANDOMZED_SMOOTHING_K = 10
FACE_OUTPUT_REG_BETA = 0.01
CP_TARGET_WEIGHT = 1.0
ANG_TARGET_WEIGHT = 1.0

# Randomized smoothing scale: perturbed costs are c + λ ε, ε ~ N(0, I).
# Too small → perturbed solves often match x*; estimator variance high.
# Too large → x_k far from x*; gradient bias grows.
PERTURB_LAMBDA = 3.0


_SOLVER = ActionSolver()


def _gurobi_solve_batch(c_batch: np.ndarray) -> np.ndarray:
    """Solve the Gurobi objective for every env. Returns (N, 8)."""
    return _SOLVER.solve_batch(c_batch)


def _solve_milp_pure_callback(c: jnp.ndarray) -> jnp.ndarray:
    """Forward-only JAX wrapper around _gurobi_solve_batch. c: (N,18) → x: (N,8)."""
    c = c.astype(jnp.float32)
    shape = jax.ShapeDtypeStruct((c.shape[0], 8), jnp.float32)
    return jax.pure_callback(
        lambda cb_c: _gurobi_solve_batch(np.asarray(cb_c)).astype(np.float32),
        shape,
        c,
    )


@jax.custom_vjp
def milp_solver(c: jnp.ndarray, rng: jnp.ndarray, verbosity: int) -> jnp.ndarray:
    """Differentiable Gurobi solve: c (N,18) → x_star (N,8).

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

    angle_0 = _x_star[:, 7]
    contact_point_0 = _x_star[:, 6]
    face_0 = jnp.argmax(_x_star[:, :N_FACES], axis=-1)

    #
    for k_i in range(RANDOMZED_SMOOTHING_K):
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, c.shape, dtype=jnp.float32)
        c_pert = (c + PERTURB_LAMBDA * eps).astype(jnp.float32)
        x_k = _solve_milp_pure_callback(c_pert)
        face_k = jnp.argmax(x_k[:, :N_FACES], axis=-1)
        if verbosity > 0:
            jax.debug.print("\n  {k_i}", k_i=k_i)
            # jax.debug.print("c_perturbed={c_pert}", c_pert=c_pert)
            jax.debug.print("x_perturbed deltas")
            jax.debug.print("|__ face_0={face_0}", face_0=face_0)
            jax.debug.print("|__ face_k={face_k}", face_k=face_k)
            jax.debug.print("|__ d_face={d_face}", d_face=face_k - face_0)
            jax.debug.print("|__ cp=    {delta_cp}", delta_cp=x_k[:, 6] - contact_point_0)
            jax.debug.print("|__ a=     {delta_a}", delta_a=x_k[:, 7] - angle_0)
        inner = jnp.sum(grad_x_safe * x_k, axis=-1, keepdims=True)  # (N, 1)
        grad_c = grad_c + eps * inner
    grad_c = grad_c / (RANDOMZED_SMOOTHING_K * PERTURB_LAMBDA)
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
    initial_mean_dist = means[0]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"SurCo-prior  n_envs={n_envs}  n_sim_steps={n_sim_steps}  "
        f"n_opt_steps={n_opt_steps}  K={RANDOMZED_SMOOTHING_K}  λ={PERTURB_LAMBDA}  "
        f"RANDOM_T_POSE={random_t_pose}"
    )
    ax_mean, ax_std = axes[0, 0], axes[0, 1]
    ax_cp, ax_ang = axes[1, 0], axes[1, 1]
    ax_face, ax_delta = axes[2, 0], axes[2, 1]

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

    for env_idx in range(min(n_envs, 7)):
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
    if save_filepath is None:
        save_filepath = save_dir / "surco_prior.png"
    plt.savefig(save_filepath, bbox_inches="tight")
    print(f"Saved plot to {save_filepath}")
    plt.close()
    if open_after_save:
        print(f"xdg-open {save_filepath}")
        os.system(f"xdg-open {save_filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(problem_type: str, n_envs: int, verbosity: int, random_t_pose: bool, record_video: bool):
    assert problem_type in ["single_step"], "Only single step problem is supported for now."
    assert verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."

    env = PushTEnv(nenvs=n_envs, record_video=record_video, visualize=False)
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%d__%H:%M:%S")
    save_dir = Path(f"logs/{now}__{problem_type}__n-envs:{n_envs}__SurCo-prior")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"xdg-open {save_dir}")

    mlp = MLP(context_dim=11, hidden_dims=(128, 128))
    params = mlp.init(jax.random.PRNGKey(0))

    def cost(params, data, rng_solve):
        ctx = env.get_context_vector(data)  # (n_envs, 11)
        c = mlp.apply(params, ctx)  # (n_envs, 18) face logits + per-face bounded targets

        if verbosity > 1:
            jax.debug.print(
                "context  any_nan={n} min={lo:.3f} max={hi:.3f}",
                n=jnp.any(jnp.isnan(ctx)),
                lo=ctx.min(),
                hi=ctx.max(),
            )

        # Do one clean forward solve/rollout. The only smoothing Monte Carlo lives
        # in milp_solver's custom VJP, where it estimates dL/dc.
        x_star = milp_solver(c, rng_solve, verbosity)  # (n_envs, 8)
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
        # penalize them to keep smoothing effective. The per-face targets are
        # already bounded by the sigmoid head.
        c_reg = FACE_OUTPUT_REG_BETA * jnp.mean(jnp.square(c[:, :N_FACES]))
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
    initial_mean_dist = None
    initial_final_dists = None
    initial_faces = None
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
        final_dists_np = np.asarray(t_distances[:, -1])
        face_idx_np = np.asarray(jnp.argmax(face_onehot, axis=-1))
        g_params = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), g_raw)
        params = jax.tree.map(lambda p, g: p - LR * g, params, g_params)
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
        if initial_mean_dist is None:
            initial_mean_dist = loss
        if initial_final_dists is None:
            initial_final_dists = final_dists_np.copy()
        if initial_faces is None:
            initial_faces = face_idx_np.copy()
        if n_envs_better_0 is None:
            n_envs_better_0 = sum(final_dists_np < initial_mean_dist - 0.05)
        delta_cm = 100 * (loss - initial_mean_dist)
        cprint(f"|____ mean dist: {loss:.5f} [m] | delta from initial: {delta_cm:.3f} [cm] | initial mean: {initial_mean_dist:.5f} [m] | {dt * 1000:.1f} ms", "green" if delta_cm < 0 else "red")
        n_envs_better = sum(final_dists_np < initial_mean_dist - 0.05)
        cprint(f"|____ {n_envs_better} / {n_envs} envs are better than the initial mean, initial: {n_envs_better_0}", "green" if n_envs_better > n_envs_better_0 else "red")
        cprint(f"|____ face initial={initial_faces.tolist()}", "yellow")
        cprint(f"|____ face current={face_idx_np.tolist()}", "yellow")
        cprint(f"|____ face diff=   {(face_idx_np - initial_faces).tolist()}", "yellow")

        # Print gradient statistics
        #
        if verbosity > 0:
            grad_abs_values = [jnp.abs(g) for layer in g_params for g in layer]
            max_grad = max(jnp.max(g).item() for g in grad_abs_values)
            mean_abs_change = jnp.mean(jnp.array([jnp.mean(LR * jnp.abs(g)).item() for g in grad_abs_values])).item()
            std_change = jnp.std(jnp.array([jnp.std(LR * jnp.abs(g)).item() for g in grad_abs_values])).item()
            cprint(f"|____ max |grad|: {max_grad:.6f}, mean |change|: {mean_abs_change:.6f}, std |change|: {std_change:.6f}", "yellow")

        # Log results for plotting
        #
        means.append(float(np.nanmean(final_dists_np)))
        stds.append(float(np.nanstd(final_dists_np)))
        dist_delta_hist.append(final_dists_np - initial_final_dists)
        face_hist.append(face_idx_np)  # (n_envs,)
        cp_hist.append(np.asarray(cp_batch))
        ang_hist.append(np.asarray(ang_batch))

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
        if record_video:
            save_filepath = save_dir / f"{it:03d}.mp4"
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
    args = parser.parse_args()
    assert args.problem_type in ["single_step"], "Only single step problem is supported for now."
    assert args.verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."
    assert args.n_envs is not None, "n_envs must be specified"
    main(args.problem_type, args.n_envs, args.verbosity, args.random_t_pose, args.record_video)
