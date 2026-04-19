"""NN-policy version of `optimize_action.py`.

One shared MLP sees per-env state and outputs (face_logits, u_contact, u_angle).
Differentiable rollout through `step_pure_soft`; gradients flow from final
distance all the way back to MLP params.

State (8-dim, per env):
    [init_T_x, init_T_y, cos(init_T_θ), sin(init_T_θ),
     target_x,  target_y,  cos(target_θ),  sin(target_θ)]

Output (8-dim, per env):
    6 face_logits + u_contact + u_angle

Debug mode (USE_NN_FACE=False):
    Face is fixed to a per-env random one-hot (no grad). MLP only predicts
    (u_contact, u_angle). Isolates the NN/Adam/rollout pipeline from the
    soft-face NaN cliff — if loss descends here, the NN stack is sound and
    any NaN with face-on is a face-head init problem, not a plumbing problem.

Run it:
    python scripts/optimize_action_nn.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pusht619.core import Action, PushTEnv

jax.config.update("jax_enable_x64", True)


N_ENVS = 100
N_SIM_STEPS = 50
N_OPT_STEPS = 200
LR = 3e-3
RESET_SEED = 0

HIDDEN = 64
FACE_OUT_SCALE = 5.0  # sharpens MLP face output (used when USE_NN_FACE=True).

# Structural safety: clip action ranges away from endpoints where the contact
# spring-damper blows up. Observed: face=0, contact≈0.8, angle≈2.5 → NaN.
# Shrinking sigmoid ranges makes those actions *unreachable* by the MLP.
CONTACT_LO, CONTACT_HI = 0.15, 0.85
# angle capped at 2.5 (~143°): all observed spring-damper blowups were at
# angle > 2.55 (near-tangential pushes near π graze concave corners). Lose
# the far-tangential swings; keep perpendicular + mild tangential.
ANGLE_LO, ANGLE_HI = 0.4, 2.5

# Debug toggle. False → face is a fixed per-env random one-hot; MLP only
# controls (u_contact, u_angle). Verifies NN pipeline independent of face.
USE_NN_FACE = False
DEBUG_EVERY = 5


# ----- MLP -----


def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, n_in, n_out in zip(keys, sizes[:-1], sizes[1:]):
        W = jax.random.normal(k, (n_in, n_out), dtype=jnp.float64) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros((n_out,), dtype=jnp.float64)
        params.append((W, b))
    return params


def mlp(params, x):
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1:
            x = jax.nn.tanh(x)
    return x


def param_norms(params):
    """Per-layer (W_norm, b_norm) — for grads or params."""
    return [(float(jnp.linalg.norm(W)), float(jnp.linalg.norm(b))) for W, b in params]


# ----- Adam -----


def init_adam(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return m, v


def adam_step(params, grads, m, v, step, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
    step = step + 1
    m = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * (gi * gi), v, grads)
    bc1 = 1.0 - b1**step
    bc2 = 1.0 - b2**step
    params = jax.tree_util.tree_map(
        lambda p, mi, vi: p - lr * (mi / bc1) / (jnp.sqrt(vi / bc2) + eps),
        params,
        m,
        v,
    )
    return params, m, v, step


# ----- state encoding -----


def encode_state(env: PushTEnv) -> jnp.ndarray:
    init_t = jnp.asarray(env.poses)
    target = jnp.asarray(env.target_poses)
    return jnp.stack(
        [
            init_t[:, 0],
            init_t[:, 1],
            jnp.cos(init_t[:, 2]),
            jnp.sin(init_t[:, 2]),
            target[:, 0],
            target[:, 1],
            jnp.cos(target[:, 2]),
            jnp.sin(target[:, 2]),
        ],
        axis=-1,
    )


def main():
    env = PushTEnv(nenvs=N_ENVS, record_video=True, visualize=False)
    env.reset(seed=RESET_SEED)

    key = jax.random.key(0)
    params = init_mlp(key, [8, HIDDEN, HIDDEN, 8])
    m, v = init_adam(params)
    step_count = jnp.asarray(0, dtype=jnp.int32)

    # Fixed per-env face one-hot for debug mode (no grad). Reproducible.
    rng = np.random.default_rng(seed=1)
    fixed_face_idx = rng.integers(0, 6, size=(N_ENVS,), dtype=np.int32)
    fixed_face_onehot = jnp.zeros((N_ENVS, 6), dtype=jnp.float64).at[
        jnp.arange(N_ENVS), jnp.asarray(fixed_face_idx)
    ].set(1.0)
    print(f"Debug mode: USE_NN_FACE={USE_NN_FACE}")
    if not USE_NN_FACE:
        print(f"  fixed face distribution: {np.bincount(fixed_face_idx, minlength=6)}")

    def compute_action(params, state):
        out = mlp(params, state)
        if USE_NN_FACE:
            face_weights = jax.nn.softmax(out[:, :6] * FACE_OUT_SCALE, axis=-1)
        else:
            face_weights = fixed_face_onehot
        contact_point = CONTACT_LO + (CONTACT_HI - CONTACT_LO) * jax.nn.sigmoid(out[:, 6])
        angle = ANGLE_LO + (ANGLE_HI - ANGLE_LO) * jax.nn.sigmoid(out[:, 7])
        return face_weights, contact_point, angle, out

    def cost(params, state, data0, target_xy):
        face_weights, contact_point, angle, _ = compute_action(params, state)
        _, _, t_distances, _ = env.step_pure_soft(
            data=data0,
            face_weights=face_weights,
            contact_point=contact_point,
            angle=angle,
            target_xy=target_xy,
            n_sim_steps=N_SIM_STEPS,
        )
        per_env = t_distances[:, -1]
        # Mask NaN envs in the loss so one blowup doesn't nuke logging.
        # This alone is NOT enough — the NaN originates inside the rollout,
        # so the backward pass still sees NaN intermediates. We also scrub
        # grads with nan_to_num below before the Adam update.
        finite = jnp.isfinite(per_env)
        loss = jnp.where(finite, per_env, 0.0).sum()
        return loss, per_env

    value_and_grad = jax.jit(jax.value_and_grad(cost, has_aux=True))

    @jax.jit
    def opt_step(params, m, v, step_count, state, data0, target_xy):
        (loss, per_env), grads = value_and_grad(params, state, data0, target_xy)
        # Sever NaN/Inf grads so one blowup env can't poison Adam's m/v state
        # (which would then nuke every env on subsequent iters).
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
        )
        params, m, v, step_count = adam_step(params, grads, m, v, step_count)
        # max|grad| element across whole pytree — catches single-weight explosions
        grad_max_abs = jnp.max(
            jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(grads)])
        )
        return params, m, v, step_count, loss, grads, per_env, grad_max_abs

    @jax.jit
    def forward_stats(params, state):
        """Distributions of MLP outputs + derived action — for debugging."""
        fw, cp, ang, out = compute_action(params, state)
        return {
            "out_min": out.min(),
            "out_max": out.max(),
            "out_mean_abs": jnp.abs(out).mean(),
            "face_max_prob_mean": jnp.max(fw, axis=-1).mean(),
            "face_max_prob_min": jnp.max(fw, axis=-1).min(),
            "contact_mean": cp.mean(),
            "contact_std": cp.std(),
            "angle_mean": ang.mean(),
            "angle_std": ang.std(),
        }

    print("Training MLP policy by gradient descent through physics rollout")
    t_start = time.time()

    # iter-0 sanity snapshot
    env.reset(seed=RESET_SEED)
    state0 = encode_state(env)
    stats0 = forward_stats(params, state0)
    print("  pre-train forward pass:")
    for k_, v_ in stats0.items():
        print(f"    {k_}: {float(v_):+.4f}")
    print(f"    param W norms: {[round(w, 3) for w, _ in param_norms(params)]}")

    # Track NaN history so we can see whether masking+grad-scrub actually
    # lets the optimizer recover, or envs keep going bad.
    nan_dump_budget = 3  # full per-env dumps, then we go quiet
    prev_params = params
    prev_state = state0
    for it in range(N_OPT_STEPS):
        # Fresh target layout every iter so no single env is a fixed NaN
        # attractor — forces the MLP to generalize across poses, and makes
        # NaN incidents one-off instead of self-reinforcing.
        env.reset(seed=RESET_SEED + it)
        data0 = env.data
        target_xy = jnp.asarray(env.target_poses[:, :2])
        state = encode_state(env)

        t0 = time.time()
        params_new, m, v, step_count, loss, grads, per_env, grad_max_abs = opt_step(
            params, m, v, step_count, state, data0, target_xy
        )
        dt = time.time() - t0

        per_env_np = np.asarray(per_env)
        finite_mask = np.isfinite(per_env_np)
        n_bad = int(np.sum(~finite_mask))
        n_good = int(finite_mask.sum())
        # Loss sums over finite envs only — report mean over those envs so the
        # number stays comparable as envs drop in/out of the finite set.
        mean_dist_finite = float(loss) / max(n_good, 1)
        grad_W_norms = [round(w, 4) for w, _ in param_norms(grads)]

        print(
            f"  iter {it + 1:4d}: mean_dist[finite]={mean_dist_finite:.4f}  "
            f"n_nan={n_bad:3d}/{N_ENVS}  "
            f"grad|W|={grad_W_norms}  max|grad|={float(grad_max_abs):.3e}  ({dt * 1000:.0f} ms)"
        )

        # Post-mortem on any iter that produced NaN envs. Throttled so the
        # log doesn't explode if the same envs stay bad. We still get the
        # first few incidents in detail to diagnose the blowup geometry.
        if n_bad > 0 and nan_dump_budget > 0:
            nan_dump_budget -= 1
            bad_idx = np.where(~finite_mask)[0]
            fw, cp, ang, _ = compute_action(params, state)
            cp_np = np.asarray(cp)
            ang_np = np.asarray(ang)
            face_np = np.asarray(jnp.argmax(fw, axis=-1))
            face_conf_np = np.asarray(jnp.max(fw, axis=-1))
            print(f"  !! NaN at iter {it + 1}: {len(bad_idx)} env(s); dumping up to 10:")
            for i in bad_idx[:10]:
                print(
                    f"     env {int(i):3d}: face={int(face_np[i])} "
                    f"(p={float(face_conf_np[i]):.2f})  "
                    f"contact={float(cp_np[i]):.4f}  angle={float(ang_np[i]):.4f}  "
                    f"dist={per_env_np[i]}"
                )
            if len(bad_idx) > 10:
                print(f"     ... {len(bad_idx) - 10} more")
            if nan_dump_budget == 0:
                print("  (further NaN incidents will be counted but not dumped)")
        # Grad scrub is done inside opt_step via nan_to_num; if max|grad| is
        # still NaN/Inf something upstream leaked — flag loudly.
        if not np.isfinite(float(grad_max_abs)):
            print("  !! grad_max_abs is non-finite AFTER nan_to_num — grad scrub failed")

        prev_params = params
        prev_state = state
        params = params_new
    _ = (prev_params, prev_state)

    print(f"Training took {time.time() - t_start:.2f} s total")

    # Final rollout with hard-int face.
    env.reset(seed=RESET_SEED)
    state = encode_state(env)
    fw, cp, ang, _ = compute_action(params, state)
    face_confidence = np.asarray(jnp.max(fw, axis=-1))
    if USE_NN_FACE:
        face_final = np.asarray(jnp.argmax(fw, axis=-1)).astype(np.int32).reshape(N_ENVS, 1)
    else:
        face_final = fixed_face_idx.reshape(N_ENVS, 1)
    contact_point = np.asarray(cp).reshape(N_ENVS, 1)
    angle = np.asarray(ang).reshape(N_ENVS, 1)

    final_action = Action(face=face_final, contact_point=contact_point, angle=angle)
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
    print(f"\nMean final distance (hard-face rollout): {float(t_distances[:, -1].mean()):.4f} m")

    env.save_video(Path("/tmp/learned_action_nn.mp4"))


if __name__ == "__main__":
    main()
