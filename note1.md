quick question:
1. main.py face=np.random.randint(0, 6, size=(9, 1)), 6 faces or 8 faces, are we forgoing the two inner faces for simplicity?
2. n_sim_steps = 100, whats sim dt here? JaxSimModel.build_from_model_description(time_step=0.001)
3. looks like its 1ms
4. looks like the white wall around the env has no collision?

---

# Progress

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
    `face @ _FACE_START_POINTS_JAX` (convex combination, differentiable)
- New `PushTEnv.step_pure_soft(...)` method. Mirrors `step_pure` but takes
  `face_weights: (nenvs, 6) float` and skips the `astype(int32)` cast so
  gradients propagate.

### Tests in [tests/test_core.py](tests/test_core.py)

- `test_step_pure_soft_matches_hard`: one-hot `face_weights` must reproduce
  the integer-face rollout bit-for-bit (`atol=1e-8`).
- `test_step_pure_soft_is_differentiable_in_face`: `jax.value_and_grad` through
  `softmax(face_logits) → step_pure_soft → final distance` produces finite,
  non-zero gradients w.r.t. `face_logits`.

### Gotchas found while testing

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

