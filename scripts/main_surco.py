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
python scripts/main_surco.py --n-envs 64 --random-t-pose
python scripts/main_surco.py --n-envs 16 --relative-coordinates --random-t-pose --relative-coordinates
python scripts/main_surco.py --n-envs 25 --relative-coordinates --random-t-pose --verbosity 1 --record-video
python scripts/main_surco.py --n-envs 25 --relative-coordinates --random-t-pose --verbosity 1 --record-video --multi-step-n-actions 2
python scripts/main_surco.py --n-envs 1 --use-soft-face --random-t-pose --verbosity 1 --record-video
python scripts/main_surco.py --n-envs 1 --use-hard-face --random-t-pose --verbosity 1 --record-video
python scripts/main_surco.py --n-envs 1 --relative-coordinates --use-soft-face --random-t-pose --verbosity 1 --record-video
python scripts/main_surco.py --n-envs 1 --relative-coordinates --use-hard-face --random-t-pose --verbosity 1 --record-video


# Fixed t-pose
python scripts/main_surco.py --n-envs 25 --verbosity 1 --record-video

python scripts/main_surco.py --n-envs 1 --use-soft-face --verbosity 1 --record-video
python scripts/main_surco.py --n-envs 1 --use-hard-face --verbosity 1 --record-video
python scripts/main_surco.py --n-envs 1 --relative-coordinates --use-soft-face --verbosity 1 --record-video
python scripts/main_surco.py --n-envs 1 --relative-coordinates --use-hard-face --verbosity 1 --record-video

"""

from __future__ import annotations
import json
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
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]

import optax
from pusht619.models import MLP, ActionSolver, ActionSolverMultiStep
from pusht619.core import Action, PushTEnv, ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES


# ── Hyperparameters ───────────────────────────────────────────────────────────

N_OPT_STEPS = 250
LR = 0.01
N_SIM_STEPS = 25
RESET_SEED = 0
ACTION_DIM = NUM_FACES + 2
M_ROLLOUTS = 10
RANDOM_ACTION_SAMPLE_K = 5
FACE_OUTPUT_REG_BETA = 0.05
CONT_OUTPUT_REG_BETA = 0.005
CP_TARGET_WEIGHT = 1.0
ANG_TARGET_WEIGHT = 1.0
RANDOM_T_POSE_EVAL_EVERY = 25

# Randomized smoothing scale: perturbed costs are c + λ ε, ε ~ N(0, I).
# Too small → perturbed solves often match x*; estimator variance high.
# Too large → x_k far from x*; gradient bias grows.
PERTURB_LAMBDA = 1.25


_SOLVER = ActionSolver()
_ENV: "PushTEnv | None" = None
_ENV_BACKWARD: "PushTEnv | None" = None
_CP_MID = 0.5 * (CONTACT_POINT_BOUNDS[0] + CONTACT_POINT_BOUNDS[1])
_ANG_MID = 0.5 * (float(ANGLE_BOUNDS[0]) + float(ANGLE_BOUNDS[1]))
_CP_SCALE = CONTACT_POINT_BOUNDS[1] - CONTACT_POINT_BOUNDS[0]
_ANG_SCALE = float(ANGLE_BOUNDS[1]) - float(ANGLE_BOUNDS[0])


def _configure_solver(multi_step_n_actions: int | None) -> None:
    global _SOLVER
    if multi_step_n_actions is None:
        _SOLVER = ActionSolver()
        return
    if multi_step_n_actions < 1:
        raise ValueError("multi_step_n_actions must be >= 1")
    _SOLVER = ActionSolverMultiStep(n_actions=multi_step_n_actions)


def _configure_env(env: "PushTEnv") -> None:
    global _ENV
    _ENV = env

def _configure_backward_env(env: "PushTEnv") -> None:
    global _ENV_BACKWARD
    _ENV_BACKWARD = env


def _run_rollout(
    face_weights: jnp.ndarray,  # (N, n_actions, NUM_FACES) — soft weights or one-hot
    cp: jnp.ndarray,            # (N, n_actions)
    ang: jnp.ndarray,           # (N, n_actions)
    data,
    env: "PushTEnv | None" = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run all action blocks sequentially through step_pure_soft.

    Returns (mean_final_dist scalar, t_distances (N, total_steps), jpos_traj (N, total_steps, dofs)).
    Differentiable w.r.t. cp and ang; face_weights treated as constants when one-hot.
    Pass env explicitly to use a different environment (e.g. _ENV_BACKWARD).
    """
    env = env or _ENV
    assert env is not None, "call _configure_env(env) before training"
    n_actions = face_weights.shape[1]
    rollout_data = data
    t_distances_parts = []
    jpos_traj_parts = []
    for action_idx in range(n_actions):
        rollout_data, _, t_dists, jpos = env.step_pure_soft(
            data=rollout_data,
            face_weights=face_weights[:, action_idx, :],
            contact_point=cp[:, action_idx],
            angle=ang[:, action_idx],
            n_sim_steps=N_SIM_STEPS,
            check_t_displacement=False,
        )
        t_distances_parts.append(t_dists)
        jpos_traj_parts.append(jpos)
    t_distances = jnp.concatenate(t_distances_parts, axis=1)
    jpos_traj = jnp.concatenate(jpos_traj_parts, axis=1)
    return jnp.nanmean(t_distances[:, -1]), t_distances, jpos_traj


def _n_actions(action_dim: int) -> int:
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
def milp_solver(c: jnp.ndarray, data, rng: jnp.ndarray, verbosity: int) -> jnp.ndarray:
    """Differentiable Gurobi solve: c (N,D) → x_star (N,D).

    Backward: M_ROLLOUTS Monte Carlo draws, each with an actual physics rollout.
        ε_k ~ N(0, I) (face logits only),  x_k = solve(c + λ ε_k)
        L_k = rollout_cost(x_k)            (true physics cost)
        ∂L/∂c_face  ≈ (1/(M λ)) Σ_k ε_k (L_k - mean(L))   [MC estimator, mean-baseline]
        ∂L/∂c_cont  ≈ (1/M) Σ_k ∂L_k/∂c_cont               [averaged analytical grad]
    """
    return _solve_milp_pure_callback(c)


def _milp_forward(c, data, rng, verbosity):
    x_star = _solve_milp_pure_callback(c)
    rng, sample_rng = jax.random.split(rng)
    return x_star, (c, x_star, data, sample_rng, int(verbosity))


def _milp_backward(res, grad_x):
    c, x_star, data, sample_rng, verbosity = res
    n_envs = c.shape[0]
    n_actions = _n_actions(c.shape[1])
    assert _ENV_BACKWARD is not None, "call _configure_backward_env before training"

    # Continuous params are the same for all M perturbed solves (only face logits are perturbed).
    x_star_blocks = x_star.reshape(n_envs, n_actions, ACTION_DIM)
    cp_0 = x_star_blocks[:, :, NUM_FACES]       # (N, n_actions)
    ang_0 = x_star_blocks[:, :, NUM_FACES + 1]  # (N, n_actions)

    key = sample_rng
    face_weights_ks: list = []
    eps_faces: list = []

    if verbosity > 0:
        jax.debug.print("  c={c}", c=c)

    # Gurobi pure_callbacks are inherently sequential — loop only for solves.
    for k_i in range(M_ROLLOUTS):
        eps_face = jnp.zeros_like(c)
        for action_idx in range(n_actions):
            key, subkey = jax.random.split(key)
            lo = action_idx * ACTION_DIM
            eps_face = eps_face.at[:, lo : lo + NUM_FACES].set(
                jax.random.normal(subkey, (n_envs, NUM_FACES), dtype=jnp.float32)
            )
        c_pert = (c + PERTURB_LAMBDA * eps_face).astype(jnp.float32)
        x_k = _solve_milp_pure_callback(c_pert)
        x_k_blocks = x_k.reshape(n_envs, n_actions, ACTION_DIM)
        face_weights_ks.append(x_k_blocks[:, :, :NUM_FACES])  # (N, n_actions, F) one-hot
        eps_faces.append(eps_face)

    # Stack all M face-weight arrays → (M*N, n_actions, F); tile data → (M*N, ...).
    # All M rollouts share the same cp/ang (only face differs), so we tile those too
    # and differentiate through the tiling to get the averaged continuous gradient.
    face_weights_all = jnp.concatenate(face_weights_ks, axis=0)   # (M*N, n_actions, F)
    data_tiled = jax.tree.map(lambda x: jnp.repeat(x, M_ROLLOUTS, axis=0), data)

    def all_rollouts_cost(cp, ang):
        # cp: (N, n_actions) — tiled inside so grad flows back to this shape.
        cp_tiled = jnp.repeat(cp, M_ROLLOUTS, axis=0)   # (M*N, n_actions)
        ang_tiled = jnp.repeat(ang, M_ROLLOUTS, axis=0)
        _, t_dists, _ = _run_rollout(face_weights_all, cp_tiled, ang_tiled, data_tiled, _ENV_BACKWARD)
        # t_dists: (M*N, total_steps) → per-rollout costs (M,)
        L_ks = jnp.nanmean(t_dists[:, -1].reshape(M_ROLLOUTS, n_envs), axis=1)
        return L_ks.mean(), L_ks  # scalar for grad, L_ks as aux

    # One parallel backward pass: L_ks for MC face estimator, grads for continuous.
    (_, L_ks), (grad_cp_0, grad_ang_0) = jax.value_and_grad(
        all_rollouts_cost, argnums=(0, 1), has_aux=True
    )(cp_0, ang_0)
    # grad_cp_0 = (1/M) Σ_k ∂L_k/∂cp  (chain rule through jnp.repeat averages over M)

    # Face MC gradient — mean-baseline control variate reduces variance.
    L_mean = L_ks.mean()
    grad_c_face = jnp.zeros_like(c)
    for k_i in range(M_ROLLOUTS):
        grad_c_face = grad_c_face + eps_faces[k_i] * (L_ks[k_i] - L_mean)
    grad_c_face = grad_c_face / (M_ROLLOUTS * PERTURB_LAMBDA)

    # Continuous gradient: map (N, n_actions) grad arrays back into the c layout.
    grad_c_continuous = jnp.zeros_like(c)
    for action_idx in range(n_actions):
        lo = action_idx * ACTION_DIM
        grad_c_continuous = grad_c_continuous.at[:, lo + NUM_FACES].set(grad_cp_0[:, action_idx])
        grad_c_continuous = grad_c_continuous.at[:, lo + NUM_FACES + 1].set(grad_ang_0[:, action_idx])

    if verbosity > 0:
        jax.debug.print("  L_mean={L_mean}", L_mean=L_mean)

    grad_c = grad_c_face + grad_c_continuous
    return (grad_c, None, None, None)


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
    relative_coordinates: bool,
    use_soft_face: bool,
    random_means=None,
    random_stds=None,
    save_filepath=None,
    save_filepath2=None,
    open_after_save=False,
):
    initial_mean_loss = means[0]
    x_iters = np.arange(len(means))
    has_random_baseline = random_means is not None and random_stds is not None
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"SurCo-prior  n_envs={n_envs}  n_sim_steps={n_sim_steps}  "
        f"n_opt_steps={n_opt_steps}  M={M_ROLLOUTS}  λ={PERTURB_LAMBDA}  "
        f"RANDOM_T_POSE={random_t_pose}  FACE_MODE={'soft' if use_soft_face else 'hard'}  RELATIVE_COORDINATES={relative_coordinates}",
        fontweight="bold",
    )
    ax_mean, ax_std = axes[0, 0], axes[0, 1]
    ax_cp, ax_ang = axes[1, 0], axes[1, 1]
    ax_face, ax_delta = axes[2, 0], axes[2, 1]

    ax_mean.axhline(float(initial_mean_loss), label="initial mean", color="black", linestyle="--")
    ax_mean.plot(x_iters, means, label="training mean", color="tab:red")
    if has_random_baseline:
        ax_mean.plot(x_iters, random_means, label="random mean", color="tab:blue")
        ax_mean.fill_between(
            x_iters,
            np.asarray(random_means) - np.asarray(random_stds),
            np.asarray(random_means) + np.asarray(random_stds),
            color="tab:blue",
            alpha=0.2,
            label="random mean ± std",
        )
    ax_mean.legend()
    ax_mean.set_title("Mean Distance")
    ax_mean.set_xlabel("Iteration")
    ax_mean.set_ylabel("Distance [m]")
    ax_mean.grid(True, alpha=0.3)

    ax_std.plot(x_iters, stds, label="training std", color="tab:red")
    if has_random_baseline:
        ax_std.plot(x_iters, random_stds, label="random std", color="tab:blue")
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
    if save_filepath2 is not None:
        plt.savefig(save_filepath2, bbox_inches="tight")
    print(f"Saved plot to {save_filepath}")
    plt.close()
    if open_after_save:
        print(f"xdg-open {save_filepath}")
        os.system(f"xdg-open {save_filepath}")


def save_json(
    iteration_json_path,
    iteration,
    loss,
    mean_dist,
    final_dists_np,
    c_batch,
    x_batch,
    grad_c,
    random_action_mean_final_distance=None,
    random_action_std_final_distance=None,
):
    iteration_payload = {
        "iteration": iteration,
        "loss": float(loss),
        "mean_final_distance": mean_dist,
        "final_distance_per_env": final_dists_np.tolist(),
        "c": np.asarray(c_batch).tolist(),
        "x": x_batch.tolist(),
        "dloss_dc": grad_c.tolist(),
        "random_action_mean_final_distance": (
            None if random_action_mean_final_distance is None else float(random_action_mean_final_distance)
        ),
        "random_action_std_final_distance": (
            None if random_action_std_final_distance is None else float(random_action_std_final_distance)
        ),
        "random_action_sample_k": None if random_action_mean_final_distance is None else RANDOM_ACTION_SAMPLE_K,
    }
    iteration_json_path.write_text(json.dumps(iteration_payload, indent=2))


def random_action_mean_variance(
    eval_env: PushTEnv,
    target_pose: np.ndarray,
    t_pose: np.ndarray,
    seed: int,
    random_sample_k: int,
    n_action_steps: int = 1,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed=seed)
    results: list[float] = []
    target_poses = np.asarray(target_pose, dtype=np.float32).reshape(1, 3)
    t_poses = np.asarray(t_pose, dtype=np.float32).reshape(1, 3)
    for k in tqdm(range(random_sample_k), desc="Random actions", unit="action", leave=False):
        eval_env.reset(seed=seed + k, target_poses=target_poses, t_poses=t_poses)
        final_distance = None
        for _action_idx in range(n_action_steps):
            action = Action(
                face=np.array([[rng.integers(NUM_FACES)]], dtype=np.int32),
                contact_point=np.array(
                    [[rng.uniform(CONTACT_POINT_BOUNDS[0], CONTACT_POINT_BOUNDS[1])]],
                    dtype=np.float32,
                ),
                angle=np.array(
                    [[rng.uniform(float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1]))]],
                    dtype=np.float32,
                ),
            )
            result = eval_env.step(action=action, n_sim_steps=N_SIM_STEPS, check_t_displacement=False)
            t_distances = np.asarray(result.t_distances)[0]
            final_distance = float(t_distances[-1])
        assert final_distance is not None, "n_action_steps must be positive"
        results.append(final_distance)
    return float(np.mean(results)), float(np.std(results))


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    problem_type: str,
    n_envs: int,
    verbosity: int,
    random_t_pose: bool,
    record_video: bool,
    use_soft_face: bool,
    use_hard_face: bool,
    multi_step_n_actions: int | None,
    disable_random: bool,
    relative_coordinates: bool,
):
    assert problem_type in ["single_step", "multi_step"], "problem_type must be 'single_step' or 'multi_step'."
    assert verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."
    assert use_soft_face or use_hard_face, "At least one of use_soft_face or use_hard_face must be True."
    assert not (use_soft_face and use_hard_face), "use_soft_face and use_hard_face cannot be True at the same time."
    is_multi_step = multi_step_n_actions is not None
    n_actions = multi_step_n_actions if is_multi_step else 1
    _configure_solver(multi_step_n_actions)

    env = PushTEnv(nenvs=n_envs, record_video=record_video, visualize=False, use_relative_coordinates=relative_coordinates)
    _configure_env(env)
    backward_env = PushTEnv(nenvs=n_envs * M_ROLLOUTS, record_video=False, visualize=False, use_relative_coordinates=relative_coordinates)
    _configure_backward_env(backward_env)
    random_eval_env = None if disable_random else PushTEnv(nenvs=1, record_video=False, visualize=False)
    env.reset(seed=RESET_SEED)
    now = datetime.now().strftime("%d__%H:%M:%S")
    solver_output_dim = ACTION_DIM * n_actions
    random_pose_str = "random-t-pose" if random_t_pose else "fixed-t-pose"
    multi_step_str = "multi-step" if is_multi_step else "single-step"
    save_dir = Path(f"logs/{now}__n-envs:{n_envs}__lr:{LR}__{random_pose_str}__{multi_step_str}")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    iterations_dir = save_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"xdg-open {save_dir}")

    mlp = MLP(context_dim=9, hidden_dims=(128, 128), output_dim=solver_output_dim)
    params = mlp.init(jax.random.PRNGKey(0))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LR),
    )
    opt_state = optimizer.init(params)

    def cost_from_c(c, data, rng_solve, solver_verbosity: int, log_forward: bool):
        if verbosity > 1:
            pass

        # Do one clean forward solve/rollout. The M_ROLLOUTS Monte Carlo rollouts
        # used for gradient estimation live in milp_solver's custom VJP.
        x_star = milp_solver(c, data, rng_solve, solver_verbosity)  # (n_envs, action_dim)
        x_star_blocks = x_star.reshape((x_star.shape[0], n_actions, ACTION_DIM))
        c_blocks = c.reshape((c.shape[0], n_actions, ACTION_DIM))
        face_weights = x_star_blocks[:, :, :NUM_FACES]
        contact_points = x_star_blocks[:, :, NUM_FACES]
        angles = x_star_blocks[:, :, NUM_FACES + 1]

        if log_forward and verbosity > 0:
            face_idx = jnp.argmax(face_weights, axis=-1)
            jax.debug.print("rollout action")
            jax.debug.print(
                "|__ face={face}\n|__ contact_point (lims: {lo_cp:.3f}, {hi_cp:.3f})={cp}\n|__ angle (lims: {lo_ang:.3f}, {hi_ang:.3f})={a}",
                face=face_idx,
                cp=contact_points,
                a=angles,
                lo_cp=CONTACT_POINT_BOUNDS[0],
                hi_cp=CONTACT_POINT_BOUNDS[1],
                lo_ang=ANGLE_BOUNDS[0],
                hi_ang=ANGLE_BOUNDS[1],
            )
            jax.debug.print("|__")

        # Hard face: pass one-hot weights so _run_rollout can always use step_pure_soft.
        # One-hot @ face_geometry == hard gather, so physics is identical to step_pure.
        if use_hard_face:
            face_weights_in = jax.nn.one_hot(jnp.argmax(face_weights, axis=-1), NUM_FACES)
        else:
            face_weights_in = face_weights
        task_loss, t_distances, jpos_traj = _run_rollout(face_weights_in, contact_points, angles, data)

        final_dists = t_distances[:, -1]
        # The face logits are scale-invariant up to ordering, so lightly
        # penalize them to keep smoothing effective. The per-face targets are
        # already bounded by the sigmoid head.
        c_reg = FACE_OUTPUT_REG_BETA * jnp.mean(jnp.square(c_blocks[:, :, :NUM_FACES]))
        cont_reg = CONT_OUTPUT_REG_BETA * jnp.mean(
            jnp.square((contact_points - _CP_MID) / _CP_SCALE) + jnp.square((angles - _ANG_MID) / _ANG_SCALE)
        )
        loss = task_loss + c_reg + cont_reg

        if log_forward and verbosity > 0:
            jax.debug.print(
                "  sum(is_nan)={n} task_loss={task_loss:.6f} c_reg={c_reg:.6f} cont_reg={cont_reg:.6f}",
                n=jnp.sum(jnp.isnan(t_distances[:, -1])),
                task_loss=task_loss,
                c_reg=c_reg,
                cont_reg=cont_reg,
            )
        if log_forward and verbosity > 1:
            jax.debug.print("  final_dists={d}", d=t_distances[:, -1])
        return loss, (t_distances, jpos_traj, face_weights, contact_points, angles, c)

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
        return cost_from_c(c, data, rng_solve, verbosity, True)

    # Do NOT wrap in jax.jit: pure_callback dispatches to Python per Gurobi
    # solve, so a JIT wrapper adds overhead without benefit. step_pure_soft is
    # already JIT'd internally, so physics stays compiled.
    cost_and_grad = jax.value_and_grad(cost, argnums=0, has_aux=True)
    grad_loss_wrt_c = jax.grad(lambda c, data, rng_solve: cost_from_c(c, data, rng_solve, 0, False)[0], argnums=0)

    print("SurCo-prior: training NN  y → solver params  (Gurobi + randomized-smoothing VJP)")
    if n_envs < 16:
        cprint(
            f"WARNING: n_envs={n_envs} may produce noisy gradients; training is typically more stable with larger batches (e.g. 16+ envs).",
            "yellow",
        )

    means = []
    stds = []
    random_means = []
    random_stds = []
    dist_delta_hist = []  # list of (n_envs,) float — change in final distance from iter 1 per env
    face_hist = []  # list of (n_envs,) int — argmax face per env per iter
    cp_hist = []  # list of (n_envs,) float — contact_point per env per iter
    ang_hist = []  # list of (n_envs,) float — angle per env per iter

    n_envs_better_0 = None
    initial_mean_dist = None
    initial_final_dists = None
    initial_faces = None
    lowest_mean_dist = float("inf")
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
        (loss, (t_distances, jpos_traj, face_weights, cp_batch, ang_batch, c_batch)), g_raw = cost_and_grad(
            params, env_data_0, step_key
        )
        final_dists_np = np.asarray(t_distances[:, -1])
        initial_dists_np = np.asarray(t_distances[:, 0])
        mean_dist = float(np.nanmean(final_dists_np))
        face_idx_np = np.asarray(jnp.argmax(face_weights, axis=-1))
        face_hist_current = face_idx_np[:, 0] if is_multi_step else face_idx_np
        cp_hist_current = np.asarray(cp_batch[:, 0] if is_multi_step else cp_batch)
        ang_hist_current = np.asarray(ang_batch[:, 0] if is_multi_step else ang_batch)
        grad_c = np.asarray(grad_loss_wrt_c(c_batch, env_data_0, step_key))
        g_params = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), g_raw)
        updates, opt_state = optimizer.update(g_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        dt = time() - t0
        x_batch = np.concatenate(
            [
                np.asarray(face_weights),
                np.asarray(cp_batch)[..., None],
                np.asarray(ang_batch)[..., None],
            ],
            axis=-1,
        ).reshape(n_envs, -1)

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
            initial_mean_dist = float(np.nanmean(initial_dists_np))
        if initial_final_dists is None:
            initial_final_dists = final_dists_np.copy()
        if initial_faces is None:
            initial_faces = face_idx_np.copy()
        if random_t_pose:
            if is_eval_step:
                cprint(f"|____ eval step: using environment from iteration 0", "cyan")
        if n_envs_better_0 is None:
            n_envs_better_0 = sum(final_dists_np < initial_mean_dist - 0.05)
        delta_cm = 100 * (mean_dist - initial_mean_dist)
        cprint(
            f"|____ mean dist: {mean_dist:.5f} [m] | delta from initial: {delta_cm:.3f} [cm] | initial mean @ t=0: {initial_mean_dist:.5f} [m] | {dt * 1000:.1f} ms",
            "green" if delta_cm < 0 else "red",
        )
        n_envs_better = sum(final_dists_np < initial_mean_dist - 0.05)
        cprint(
            f"|____ {n_envs_better} / {n_envs} envs are better than the initial mean, initial: {n_envs_better_0}",
            "green" if n_envs_better > n_envs_better_0 else "red",
        )
        cprint(f"|____ face initial= {initial_faces.tolist()}", "yellow")
        cprint(f"|____ face current= {face_idx_np.tolist()}", "yellow")
        cprint(f"|____ face diff=    {(face_idx_np - initial_faces).tolist()}", "yellow")
        cprint(f"|____ angles=       {ang_hist_current.tolist()}", "yellow")
        cprint(f"|____ contact-point={cp_hist_current.tolist()}", "yellow")
        if verbosity > 1:
            cprint(f"|____ c=           {np.asarray(c_batch).tolist()}", "yellow")
            cprint(f"|____ dloss/dc=    {grad_c.tolist()}", "yellow")
        random_mean = None
        random_std = None
        if not disable_random:
            assert random_eval_env is not None
            random_mean, random_std = random_action_mean_variance(
                random_eval_env,
                target_pose=env.target_poses[0],
                t_pose=env.t_poses[0],
                seed=it,
                random_sample_k=RANDOM_ACTION_SAMPLE_K,
                n_action_steps=n_actions,
            )
            cprint(
                f"|____ random baseline (env 0): mean={random_mean:.5f} [m], std={random_std:.5f} [m], k={RANDOM_ACTION_SAMPLE_K}",
                "cyan",
            )
        save_json(
            iterations_dir / f"{it:03d}.json",
            it,
            loss,
            mean_dist,
            final_dists_np,
            c_batch,
            x_batch,
            grad_c,
            random_mean,
            random_std,
        )

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
        if random_mean is not None and random_std is not None:
            random_means.append(random_mean)
            random_stds.append(random_std)
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
                save_dir=save_dir,
                means=means,
                stds=stds,
                dist_delta_hist=dist_delta_hist,
                face_hist=face_hist,
                cp_hist=cp_hist,
                ang_hist=ang_hist,
                n_envs=n_envs,
                n_sim_steps=N_SIM_STEPS,
                n_opt_steps=N_OPT_STEPS,
                random_t_pose=random_t_pose,
                use_soft_face=use_soft_face,
                relative_coordinates=relative_coordinates,
                random_means=random_means if random_means else None,
                random_stds=random_stds if random_stds else None,
                save_filepath=save_dir / f"{it + 1:03d}.png",
                save_filepath2=save_dir / f"latest.png",
                open_after_save=False,
            )
        if mean_dist < lowest_mean_dist:
            lowest_mean_dist = mean_dist
            cprint(f"New lowest mean dist: {lowest_mean_dist:.5f} [m]", "green")
            filepath = checkpoints_dir / f"mlp_lowest_mean_dist.npz"
            mlp.save_mlp_weights(filepath, params)
            if record_video:
                save_filepath = save_dir / f"best.mp4"
                env.save_video_from_jpos_traj(save_filepath, np.asarray(jpos_traj))


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
        save_dir=save_dir,
        means=means,
        stds=stds,
        dist_delta_hist=dist_delta_hist,
        face_hist=face_hist,
        cp_hist=cp_hist,
        ang_hist=ang_hist,
        n_envs=n_envs,
        n_sim_steps=N_SIM_STEPS,
        n_opt_steps=N_OPT_STEPS,
        random_t_pose=random_t_pose,
        random_means=random_means if random_means else None,
        random_stds=random_stds if random_stds else None,
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
    parser.add_argument("--disable-random", action="store_true", help="Skip random action baseline sampling")
    parser.add_argument("--use-soft-face", action="store_true", help="Use soft face mode")
    parser.add_argument("--use-hard-face", action="store_true", help="Use hard face mode")
    parser.add_argument("--relative-coordinates", action="store_true", help="Use relative coordinates")
    args = parser.parse_args()
    assert args.verbosity in [0, 1, 2], "Verbosity must be 0, 1, or 2."
    assert args.n_envs is not None, "n_envs must be specified"
    main(
        problem_type=args.problem_type,
        n_envs=args.n_envs,
        verbosity=args.verbosity,
        random_t_pose=args.random_t_pose,
        record_video=args.record_video,
        multi_step_n_actions=args.multi_step_n_actions,
        disable_random=args.disable_random,
        use_soft_face=args.use_soft_face,
        use_hard_face=args.use_hard_face,
        relative_coordinates=args.relative_coordinates,
    )
