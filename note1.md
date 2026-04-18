quick question:
1. main.py face=np.random.randint(0, 6, size=(9, 1)), 6 faces or 8 faces, are we forgoing the two inner faces for simplicity?
2. n_sim_steps = 100, whats sim dt here? JaxSimModel.build_from_model_description(time_step=0.001)
3. looks like its 1ms
4. looks like the white wall around the env has no collision?

---

# Progress 1: Apr 17

## Goal

From `scripts/optimize_action.py` TODOs:
1. Make `face` (discrete, 0–5) differentiable so it can be optimized.
2. Add a neural network policy mapping state → action `(contact_point, angle[, face])`.
3. Randomize target poses so the agent generalizes across configurations.

Working one step at a time, with tests.

## Step 1 — Differentiable face via `step_pure_soft` ✅

**Approach:**  `_plan_push_jax` needed to branch
between the integer-gather and weighted-sum paths.

### Changes in [src/pusht619/core.py](src/pusht619/core.py)

- [`_plan_push_jax`](src/pusht619/core.py#L99-L122): dispatch on `face.ndim`.
  - `ndim == 1` (int) → gather: `_FACE_START_POINTS_JAX[face]`
  - `ndim == 2` (float weights, shape `(nenvs, 6)`) → matmul:
    `face @ _FACE_START_POINTS_JAX` 
- New `PushTEnv.step_pure_soft(...)` method. Mirrors `step_pure` but takes
  `face_weights: (nenvs, 6) float` and skips the `astype(int32)` cast so
  gradients propagate.

### Tests in [tests/test_core.py](tests/test_core.py)

- `test_step_pure_soft_matches_hard`: one-hot `face_weights` must reproduce
  the integer-face rollout bit-for-bit (`atol=1e-8`).
- `test_step_pure_soft_is_differentiable_in_face`: `jax.value_and_grad` through
  `softmax(face_logits) → step_pure_soft → final distance` produces finite,
  non-zero gradients w.r.t. `face_logits`.

### BUGS from testing

- **`nenvs` must be a perfect square.** `PushTEnv.__init__` builds a grid via
  `row_length = int(jnp.sqrt(nenvs))`. For `nenvs=6` this gives a 2×2 grid (4
  slots) and broadcasting fails. Used `nenvs=9` with faces `[0,1,2,3,4,5,0,2,4]`
  to cover all 6 faces.
- **Uniform softmax blows up.** `face_logits = zeros` → weights `[1/6]*6` →
  convex-combined contact body lands at `(0, 0.025)`, *inside* the T block's
  stem. Contact-force spring-damper explodes → NaN loss. Fix: bias one logit
  per env (value 5.0) so the softmaxed weights are near-one-hot and land the
  pusher on an actual edge.

### Status

- New tests: both pass.
- Existing tests (`test_step_pure_is_differentiable`, `test_step_executes_push_action`): still pass.
- jaxsim init-time RuntimeWarnings about casts are pre-existing, unrelated.

## Step 2 — NN policy (not started)

Proposed design:
- State vector (8-dim): `[t_x, t_y, cos(t_θ), sin(t_θ), target_x, target_y, cos(target_θ), sin(target_θ)]`
- MLP ~3 layers, 128 hidden, outputs `(face_logits[6], u_contact, u_angle)`.
- Optax Adam over MLP params; loss = final-step distance through `step_pure_soft`.
- Open question: depend on `optax`, or hand-roll Adam?

## Step 3 — Randomize targets (not started)

One-line change: use `env.reset(seed=it)` instead of fixed `RESET_SEED=0` inside
the training loop so each iteration sees a fresh target configuration.

---

# Progress 2: Apr 17 — plugging soft face into the optimizer

## Setup

`scripts/optimize_action.py` was using hard int `face` (random, fixed). Switched
it to optimize `face_logits: (N_ENVS, 6)` jointly with `u_contact`, `u_angle`,
going through `step_pure_soft`. For the final video rollout we snap to hard
`argmax(face_logits)` and call the original `step`.

Config: `N_ENVS=100`, `N_SIM_STEPS=50`, `N_OPT_STEPS=100`, `RESET_SEED=0` (same
layout every iter so optimizer sees a stationary objective).

## Experiments & results

All starts: `mean dist ≈ 0.6355 m` at iter 1 (random-face baseline).

| Run | FACE_BIAS | LR_FACE | LAMBDA_ENT | iter 100 mean dist | iter 100 entropy | Outcome |
|---|---|---|---|---|---|---|
| 1 | 5.0 | 0.25 (shared LR) | 0 | 0.6320 | ~0 (saturated init) | No face movement. Softmax Jacobian `p(1-p) ≈ 0.03` at p=0.97 → face grad vanishes. |
| 2 | 1.5 | 5.0 | 0 | NaN iter 1 | — | Bias too low; convex combo of face points lands *inside* T block → spring-damper explodes. |
| 3 | 3.0 | 5.0 | 0 | NaN iter 50 | — | Good progress (0.6355 → 0.6250 over 45 iters) then optimizer pushed face_logits toward flat → same NaN as run 2. One bad step poisons all subsequent iters. |
| 4 | 3.0 | 5.0 | 0.05 | 0.6317 | 0.048 | No NaN (entropy penalty keeps weights peaked). But entropy gradient *dominated* — face_logits just sharpened the random init instead of switching. Distance barely moved. |
| 5 | 3.0 | 1.0 | 0.005 | 0.6281 | 0.698 | Weaker entropy + smaller face LR. Entropy stayed high, `p` values spread (0.62–0.95). **Still no face switched.** 0.004 m better than run 4. |

## The core finding

**Softmax relaxation of face does not enable face *switching*.** Across every
non-NaN run, the argmax face at iter 100 matched the random init. The gradient
flows (we confirmed in the unit test) — it just always points toward further
committing to the current face basin. Switching B → A would require passing
through a mixed state that gives worse distance, so the gradient says "don't."

We were tuning knobs around a structural limitation of gradient-descending
through a relaxed discrete variable. Further λ / LR tuning would be lipstick.

## What we did learn

- There's a hard NaN cliff at `FACE_BIAS ≲ 2` (convex combination of face
  points ends up inside the T block → spring-damper explosion).
- Entropy regularization works as a soft barrier against that cliff but
  fighting it also dominates the face gradient signal.
- Softmax Jacobian saturation (`p(1-p)` → 0 as p → 1) makes separate face/action
  learning rates a necessity if we keep this approach.
- The ~0.63 m single-push ceiling is mostly **geometric** — for many envs the
  target pose is simply unreachable in one shove regardless of face choice.
  Several envs end with final distance > 1.0 m.

## Paths forward

1. **Brute-force enumerate face.** 6 choices per env, run 6 parallel
   optimizations over `(contact_point, angle)`, pick the winner per env. Gives
   the *true* best-face-per-env baseline. Cost: 6× wall time. ~10 min to wire up.
2. **Step 2 — NN policy.** An MLP mapping state → `(face_logits, u_contact,
   u_angle)` sees different (state, best-face) correspondences across resets and
   can *learn* a state→face map. 

**Thoughts:** skip further soft-face tuning. Go to (2) NN policy. Optionally run (1) first
as a baseline reference number, I still don't understand the env and pipeline fully, I think I should
not get much performance just from optimize without NN here?

