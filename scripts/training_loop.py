"""SurCo training for JAXSim PushT, with Gurobi as the combinatorial solver.

Mapping from the paper (Ferber et al. 2023, Alg. 2 / 4) to this setting:

    y = instance description: per-env JAXSim state + target xy
    c = c_hat(y; θ)      : 8-dim output of Flax MLP, split into
                           c_face = c[:6]   → fed to Gurobi
                           c[6:8] → (u_contact, u_angle), stays differentiable
    g_Ω(c_face)          : Gurobi solves  argmin c_face^T x  s.t.  sum(x)=1,
                                                                   x ∈ {0,1}^6
                          Returns one-hot face ∈ {0,1}^6.
    f(x; y)              : mean final distance after N_SIM_STEPS of physics.

Forward:  c → (Gurobi one-hot face,  sigmoid contact,  sigmoid angle) → rollout
Backward: ∂f/∂face backprops through Gurobi via Pogančić et al. 2019
          blackbox differentiation (call solver again with perturbed costs,
          finite-diff the Jacobian).  λ = PERTURB_LAMBDA hyperparameter.
          Continuous params keep their normal smooth gradients.

NOTE on smoothness: the one-hot output means ∂face/∂c is a.e. zero, so the
only gradient signal on c_face comes from Pogančić's perturbation. That's
fine — randomized smoothing can be layered on later for lower-variance grads.

Run:
    python scripts/training_loop.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import gurobipy as gp
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from gurobipy import GRB

from pusht619.core import Action, PushTEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────

N_ENVS = 64
N_SIM_STEPS = 50
N_SIM_STEPS_EVAL = 5000

N_EPOCHS = 200
LR_THETA = 1e-3

# Pogančić blackbox-differentiation perturbation magnitude.
# Too small → solver returns the same x_star for c and c', gradient = 0.
# Too large → gradient biased toward a faraway solution.
# NN-output c_face is unbounded and can grow during training, so a smaller
# λ is safer. Bump it up if you see grad_c going to zero (no face flips).
PERTURB_LAMBDA = 0.5

# Gradient clipping on NN updates. The rollout occasionally produces large
# gradients when physics nearly explodes; clipping stops one bad batch from
# destroying the NN weights.
GRAD_CLIP = 1.0

# Hybrid fine-tune
RUN_HYBRID = True
N_HYBRID_STEPS = 30
LR_HYBRID_FACE = 1.0
LR_HYBRID_ACTION = 0.25

EVAL_EVERY = 10
N_EVAL_ENVS = 16
CKPT_PATH = Path("checkpoints/surco_prior_params.npz")


NUM_FACES = 6
# ── 1. Gurobi: the combinatorial solver g_Ω ───────────────────────────────────
#
# Persistent solver: build the model once at module load, reuse across all
# calls. Each solve just mutates the objective coefficients in place (Gurobi
# supports this via Var.Obj and Model.update()). This is ~10× faster than
# rebuilding the model per call, which is critical because we trigger
# 2 × N_ENVS = 128 solves per training step.


class _PersistentFaceSolver:
    """One Gurobi model, reused across every solve. Thread-safe enough for
    our single-threaded training loop. If you ever parallelize Gurobi calls,
    give each worker its own instance."""

    def __init__(self):
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model(env=self.env)
        self.model.setParam("Threads", 1)
        # Presolve is unhelpful for a 6-var problem and adds overhead.
        self.model.setParam("Presolve", 0)
        self.x = self.model.addVars(NUM_FACES, vtype=GRB.BINARY, name="x")
        self.model.addConstr(gp.quicksum(self.x[i] for i in range(NUM_FACES)) == 1)
        # Objective will be mutated per solve; initialize to zeros.
        self.model.setObjective(gp.quicksum(0.0 * self.x[i] for i in range(NUM_FACES)), GRB.MINIMIZE)
        self.model.update()

    def solve(self, c_face: np.ndarray) -> np.ndarray:
        """Update obj coefs in place, solve, return one-hot (6,)."""
        # NaN/Inf guard: if upstream produced garbage coefficients (typically
        # from a NaN rollout earlier in the batch), fall back to a safe
        # uniform cost so Gurobi still returns *something*. We also surface
        # this upstream so the training loop can skip the step.
        if not np.all(np.isfinite(c_face)):
            c_face = np.zeros(NUM_FACES, dtype=np.float32)
        for i in range(NUM_FACES):
            self.x[i].Obj = float(c_face[i])
        self.model.update()
        self.model.optimize()
        return np.array([self.x[i].X for i in range(NUM_FACES)], dtype=np.float32)


_SOLVER = _PersistentFaceSolver()


def gurobi_solve_batch(c_face_batch: np.ndarray) -> np.ndarray:
    """Solve the MILP for each env in the batch. Returns (N, 6) one-hot.

    NaN/Inf coefficients are sanitized per-env inside _SOLVER.solve; this
    prevents a single poisoned env from killing the whole training run.
    """
    N = c_face_batch.shape[0]
    out = np.zeros((N, NUM_FACES), dtype=np.float32)
    for i in range(N):
        out[i] = _SOLVER.solve(c_face_batch[i])
    return out


# ── 2. JAX wrapper with Pogančić blackbox VJP ────────────────────────────────


def _solve_pure_callback(c_face: jnp.ndarray) -> jnp.ndarray:
    """Forward-only JAX wrapper around gurobi_solve_batch."""
    shape = jax.ShapeDtypeStruct(c_face.shape, c_face.dtype)
    return jax.pure_callback(
        lambda cb_c: gurobi_solve_batch(np.asarray(cb_c)).astype(c_face.dtype),
        shape,
        c_face,
    )


@jax.custom_vjp
def milp_face_solver(c_face: jnp.ndarray) -> jnp.ndarray:
    """Differentiable Gurobi: c_face (N,6) → x_star (N,6) one-hot.

    Gradient via Pogančić et al. 2019:
        x_prime = solve(c + λ · grad_x L)
        ∂L/∂c   = -(x_star - x_prime) / λ
    """
    return _solve_pure_callback(c_face)


def _milp_fwd(c_face):
    x_star = _solve_pure_callback(c_face)
    return x_star, (c_face, x_star)


def _milp_bwd(res, grad_x):
    c_face, x_star = res
    # Sanitize upstream grad: a NaN rollout anywhere in the batch will
    # produce NaN grad_x, which would make c_prime = NaN and crash Gurobi.
    # Replace NaN/Inf with 0 so those rows get zero surrogate gradient
    # (safer than killing the step).
    grad_x_safe = jnp.where(jnp.isfinite(grad_x), grad_x, 0.0)
    # Pogančić: perturb costs in the direction of the upstream solution gradient.
    c_prime = c_face + PERTURB_LAMBDA * grad_x_safe
    x_prime = _solve_pure_callback(c_prime)
    grad_c = -(x_star - x_prime) / PERTURB_LAMBDA
    return (grad_c,)


milp_face_solver.defvjp(_milp_fwd, _milp_bwd)


# ── 3. Feature extraction ─────────────────────────────────────────────────────


def extract_y(data, target_xy: jnp.ndarray) -> jnp.ndarray:
    """Flatten per-env JAXSim state + target into a feature vector (N, feat)."""
    base_pos = data.base_position
    base_quat = data.base_orientation
    joint_pos = data.joint_positions
    parts = [base_pos, base_quat, target_xy]
    if joint_pos.shape[-1] > 0:
        parts.insert(2, joint_pos)
    return jnp.concatenate(parts, axis=-1)


# ── 4. Prior network: y → c ∈ R^8 ─────────────────────────────────────────────


class SurCoPrior(nn.Module):
    hidden: tuple[int, ...] = (128, 128, 64)

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        h = y
        for d in self.hidden:
            h = nn.Dense(d)(h)
            h = nn.relu(h)
        return nn.Dense(8)(h)


# ── 5. Surrogate solver + rollout cost ────────────────────────────────────────


def c_to_action(c: jnp.ndarray):
    """c (N,8) → (face_onehot, contact_point, angle). Gurobi on face, sigmoid
    on the rest. Returned face is one-hot; step_pure_soft accepts this as a
    valid `face_weights` argument (weight 1 on chosen face, 0 elsewhere)."""
    c_face = c[:, :NUM_FACES]
    u_contact = c[:, 6]
    u_angle = c[:, 7]
    face_onehot = milp_face_solver(c_face)  # (N, 6), differentiable
    contact_point = jax.nn.sigmoid(u_contact)
    angle = jnp.pi * jax.nn.sigmoid(u_angle)
    return face_onehot, contact_point, angle


def rollout_cost(c: jnp.ndarray, env: PushTEnv, data0):
    """f(g_Ω(c); y) = mean final distance after the JAXSim rollout.

    No entropy reg: the face output is already one-hot, so entropy is
    identically 0 and the term would do nothing.
    """

    face_weights, contact_point, angle = c_to_action(c)
    _, _, t_distances, _ = env.step_pure_soft(
        data=data0,
        face_weights=face_weights,
        contact_point=contact_point,
        angle=angle,
        n_sim_steps=N_SIM_STEPS,
    )
    dist_sum = t_distances[:, -1].sum()

    # at the start of rollout_cost, after the step_pure_soft call:
    # jax.debug.print(
    #     "t_distances finite: {}  c finite: {}",
    #     jnp.all(jnp.isfinite(t_distances)),
    #     jnp.all(jnp.isfinite(c)),
    # )
    return dist_sum / c.shape[0], dist_sum


# ── 6. Train step factory ─────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    env: PushTEnv


def make_train_step(cfg: TrainConfig):
    """Build the prior-training step. We do NOT jit over the whole step
    because jax.pure_callback + custom_vjp round-trips to Python per Gurobi
    solve. step_pure_soft is already jitted internally, so the heavy physics
    compute stays on device.

    Returns (loss, dist_sum, grads) — the caller decides whether to apply
    the update (skip on NaN)."""
    env = cfg.env

    def loss_fn(params, y, data0):
        c = SurCoPrior().apply(params, y)
        loss, dist_sum = rollout_cost(c, env, data0)
        return loss, dist_sum

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def step(state: train_state.TrainState, y, data0):
        (loss, dist_sum), grads = grad_fn(state.params, y, data0)
        return loss, dist_sum, grads

    return step


# ── 7. Offline training (SurCo-prior, Alg. 2) ────────────────────────────────


def train_prior():
    print("=" * 70)
    print("SurCo-prior training  (Gurobi MILP face solver + blackbox VJP)")
    print("=" * 70)

    env = PushTEnv(nenvs=N_ENVS, record_video=False, visualize=False)
    env.reset(seed=0)
    target_xy = jnp.asarray(env.target_poses[:, :2])
    y_sample = extract_y(env.data, target_xy)
    print(f"Feature dim y: {y_sample.shape[-1]}  |  c dim: 8  (6 MILP + 2 cont.)")
    print(f"Pogančić perturbation λ = {PERTURB_LAMBDA}")

    model = SurCoPrior()
    key = jax.random.PRNGKey(0)
    params = model.init(key, y_sample)
    # Clip first, then adam. optax.chain applies left-to-right.
    tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR_THETA))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    step_fn = make_train_step(TrainConfig(env=env))

    # Dedicated eval env with video recording on. Kept separate from the
    # training env so we don't pay the video-recording overhead on every
    # training step. Uses a smaller nenvs so the mp4 is manageable.
    eval_env = PushTEnv(nenvs=N_EVAL_ENVS, record_video=True, visualize=False)
    EVAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Eval videos will be saved to: {EVAL_VIDEO_DIR}/")

    rng = np.random.default_rng(seed=42)
    t_start = time.time()
    n_skipped = 0

    for epoch in range(N_EPOCHS):
        seed = int(rng.integers(0, 10**9))
        env.reset(seed=seed)
        target_xy = jnp.asarray(env.target_poses[:, :2])
        data0 = env.data
        y = extract_y(data0, target_xy)

        t0 = time.time()
        # Each step triggers 2 Gurobi batches: one on fwd, one on bwd with
        # perturbed costs. ~100 ms total for 64 envs × 2 solves.
        loss, dist_sum, grads = step_fn(state, y, data0)

        # NaN check: if the rollout exploded, skip the update so we don't
        # poison the NN weights. The next batch gets a fresh seed and the
        # physics usually recovers.
        grads_flat = jnp.concatenate([jnp.ravel(g) for g in jax.tree.leaves(grads)])
        grads_finite = bool(jnp.all(jnp.isfinite(grads_flat)))
        loss_finite = bool(jnp.isfinite(loss))
        if grads_finite and loss_finite:
            state = state.apply_gradients(grads=grads)
        else:
            n_skipped += 1
            print(
                f"  epoch {epoch + 1:4d}: ⚠ NaN in "
                f"{'loss' if not loss_finite else 'grads'} — step skipped "
                f"(total skipped: {n_skipped})"
            )
            continue
        dt = time.time() - t0

        mean_dist = float(dist_sum) / N_ENVS
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"  epoch {epoch + 1:4d}/{N_EPOCHS} | "
                f"loss={float(loss):.4f}  mean_dist={mean_dist:.4f} m  "
                f"({dt * 1000:.0f} ms)"
            )

        if (epoch + 1) % EVAL_EVERY == 0:
            video_path = EVAL_VIDEO_DIR / f"surco_eval_epoch_{epoch + 1:04d}.mp4"
            ed = evaluate_prior(
                state.params,
                eval_env,
                n_eval=N_EVAL_ENVS,
                seed=EVAL_VIDEO_SEED,
                video_path=video_path,
            )
            print(f"    [eval] mean_dist = {ed:.4f} m  |  video → {video_path}")

    print(f"\nTotal training time: {time.time() - t_start:.1f} s")
    print(f"Skipped steps (NaN in loss/grads): {n_skipped} / {N_EPOCHS}")
    np.savez(CKPT_PATH, **_flatten_params(jax.tree.map(np.asarray, state.params)))
    print(f"Saved prior params → {CKPT_PATH}")
    return state.params, env


def _flatten_params(params, prefix=""):
    out = {}
    for k, v in params.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        else:
            out[key] = np.asarray(v)
    return out


# ── 8. Deployment (Alg. 3) ────────────────────────────────────────────────────


def evaluate_prior(
    params,
    env: PushTEnv,
    n_eval: int = 16,
    seed: int = 1234,
    video_path: Path | None = None,
    n_sim_steps: int = N_SIM_STEPS_EVAL,
) -> float:
    """Forward pass through NN → Gurobi → hard env.step rollout.

    n_sim_steps defaults to N_SIM_STEPS_EVAL (longer than training) so the
    T has time to reach the target in the video. If video_path is provided
    AND env was constructed with record_video=True, saves the rollout video
    to that path.
    """
    env.reset(seed=seed)
    target_xy = jnp.asarray(env.target_poses[:, :2])
    y = extract_y(env.data, target_xy)

    c = SurCoPrior().apply(params, y)
    face_onehot = np.asarray(_solve_pure_callback(c[:, :NUM_FACES]))
    face = np.argmax(face_onehot, axis=-1).astype(np.int32).reshape(-1, 1)
    contact = np.asarray(jax.nn.sigmoid(c[:, 6])).reshape(-1, 1)
    angle = np.asarray(jnp.pi * jax.nn.sigmoid(c[:, 7])).reshape(-1, 1)

    result = env.step(Action(face=face, contact_point=contact, angle=angle), n_sim_steps=n_sim_steps)

    if video_path is not None:
        # NOTE: this assumes env.save_video dumps the current recording
        # buffer. If videos across eval calls end up concatenated, the fix
        # is to rebuild `env` each eval (slower but clean). Check behavior
        # on first run and adjust if needed.
        video_path.parent.mkdir(parents=True, exist_ok=True)
        env.save_video(video_path)

    return float(np.asarray(result.t_distances[:n_eval, -1]).mean())


# ── 9. SurCo-hybrid (Alg. 4) ──────────────────────────────────────────────────


def hybrid_finetune(params, env: PushTEnv, seed: int = 9999, save_video: bool = True):
    """Warm-start c from the prior NN, then take SurCo-zero steps on c itself
    (grads through the same Gurobi+rollout pipeline)."""
    print("\n" + "=" * 70)
    print(f"SurCo-hybrid fine-tune  ({N_HYBRID_STEPS} steps, seed={seed})")
    print("=" * 70)

    env.reset(seed=seed)
    target_xy = jnp.asarray(env.target_poses[:, :2])
    data0 = env.data
    y = extract_y(data0, target_xy)

    c_init = SurCoPrior().apply(params, y)
    c_face = c_init[:, :NUM_FACES]
    u_contact = c_init[:, 6]
    u_angle = c_init[:, 7]

    def cost(c_face, u_contact, u_angle):
        c = jnp.concatenate([c_face, u_contact[:, None], u_angle[:, None]], axis=-1)
        loss, dist_sum = rollout_cost(c, env, data0)
        return loss, dist_sum

    cost_and_grad = jax.value_and_grad(cost, argnums=(0, 1, 2), has_aux=True)

    for step in range(N_HYBRID_STEPS):
        (loss, dist_sum), (g_face, g_c, g_a) = cost_and_grad(c_face, u_contact, u_angle)
        c_face = c_face - LR_HYBRID_FACE * g_face
        u_contact = u_contact - LR_HYBRID_ACTION * g_c
        u_angle = u_angle - LR_HYBRID_ACTION * g_a

        if step == 0 or (step + 1) % 5 == 0:
            mean_dist = float(dist_sum) / c_face.shape[0]
            print(f"  hybrid step {step + 1:3d}/{N_HYBRID_STEPS}: mean_dist = {mean_dist:.4f} m")

    face_onehot = np.asarray(_solve_pure_callback(c_face))
    face = np.argmax(face_onehot, axis=-1).astype(np.int32).reshape(-1, 1)
    contact = np.asarray(jax.nn.sigmoid(u_contact)).reshape(-1, 1)
    angle = np.asarray(jnp.pi * jax.nn.sigmoid(u_angle)).reshape(-1, 1)

    env.reset(seed=seed)
    result = env.step(Action(face=face, contact_point=contact, angle=angle), n_sim_steps=N_SIM_STEPS)
    final = np.asarray(result.t_distances[:, -1])
    print(f"\n  [hybrid final] mean_dist = {final.mean():.4f} m  (min={final.min():.4f}  max={final.max():.4f})")

    if save_video:
        out = Path("/tmp/surco_hybrid.mp4")
        env.save_video(out)
        print(f"  video → {out}")


# ── 10. Main ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    params, env = train_prior()
    held_out = evaluate_prior(params, env, n_eval=N_ENVS, seed=77777)
    print(f"\n[Held-out prior eval] mean_dist = {held_out:.4f} m")

    if RUN_HYBRID:
        env_vid = PushTEnv(nenvs=N_ENVS, record_video=True, visualize=False)
        hybrid_finetune(params, env_vid, seed=77777, save_video=True)
