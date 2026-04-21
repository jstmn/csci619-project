import json
import numpy as np
from pathlib import Path
import pytest
import jaxsim as js
import jax.numpy as jnp
import jax

from pusht619.core import Action, PushTEnv


# python -m pytest tests/test_core.py::test_PushTEnv_smoke_test --capture=no
def test_PushTEnv_smoke_test() -> None:
    env = PushTEnv(nenvs=9, record_video=True, visualize=False)
    action = Action(
        face=np.array([0, 1, 2, 3, 4, 5, 0, 1, 2]).reshape(9, 1),
        contact_point=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(9, 1),
        angle=np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]
        ).reshape(9, 1),
        # push_distance=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).reshape(9, 1),
    )
    n_sim_steps = 100
    result = env.step(action, n_sim_steps=n_sim_steps)
    assert result.action == action
    assert result.t_poses.shape == (9, n_sim_steps, 3)
    assert result.t_distances.shape == (9, n_sim_steps)

    # Next, save a video
    save_filepath = Path("/tmp/test_PushTEnv_smoke_test.mp4")
    env.save_video(save_filepath)
    assert save_filepath.exists()
    print(f"Saved video to {save_filepath}")
    print(f"xdg-open {save_filepath}")

    # Now, reset the environment
    env.reset()

    # Try saving a video again, it should fail because the environment has not been stepped
    with pytest.raises(AssertionError):
        env.save_video("/tmp/test_PushTEnv_smoke_test_2.mp4")


def test_step_executes_push_action() -> None:
    env = PushTEnv(nenvs=1, record_video=False, visualize=False)
    initial_t_pose = env.poses.copy()

    action = Action(
        face=np.array([[2]], dtype=np.int32),
        contact_point=np.array([[0.5]], dtype=np.float32),
        angle=np.array([[np.pi / 2]], dtype=np.float32),
    )
    env.step(action, n_sim_steps=100)

    final_t_pose = env.poses.copy()
    final_pusher_xy = np.asarray(env._data.joint_positions[0, [env._pusher_x_idx, env._pusher_y_idx]])

    assert np.linalg.norm(final_t_pose - initial_t_pose) > 1e-4
    assert np.linalg.norm(final_pusher_xy - np.array([0.75, 0.75])) > 0.1


# python -m pytest tests/test_core.py::test_step_pure_is_differentiable --capture=no
def test_step_pure_is_differentiable() -> None:
    """`env.step_pure` should be `jax.grad`-able end-to-end through a full
    100-step physics rollout, and the gradient w.r.t. the continuous action
    parameters should be finite and (for at least one env) non-zero.
    """
    nenvs = 4
    n_sim_steps = 50

    env = PushTEnv(nenvs=nenvs, record_video=False, visualize=False)
    env.reset(seed=0)

    data0 = env.data

    # Choose faces that point roughly toward the targets so the pusher actually
    # makes contact, ensuring a non-trivial gradient signal.
    face = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    def cost(contact_point: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
        _, _, t_distances, _ = env.step_pure(
            data=data0,
            face=face,
            contact_point=contact_point,
            angle=angle,
            n_sim_steps=n_sim_steps,
        )
        # Scalar: sum of final-step distances across envs.
        return t_distances[:, -1].sum()

    contact_point = jnp.full((nenvs,), 0.5, dtype=jnp.float32)
    angle = jnp.full((nenvs,), np.pi / 2, dtype=jnp.float32)

    # Forward pass is a scalar cost.
    loss = cost(contact_point, angle)
    assert loss.shape == ()
    assert jnp.isfinite(loss), f"loss is not finite: {loss}"

    # Backward pass: analytic gradient w.r.t. the continuous action params.
    grad_fn = jax.jit(jax.grad(cost, argnums=(0, 1)))
    g_contact, g_angle = grad_fn(contact_point, angle)

    assert g_contact.shape == (nenvs,)
    assert g_angle.shape == (nenvs,)
    assert jnp.all(jnp.isfinite(g_contact)), f"g_contact has NaN/Inf: {g_contact}"
    assert jnp.all(jnp.isfinite(g_angle)), f"g_angle has NaN/Inf: {g_angle}"

    # At least one env should produce a non-zero gradient — otherwise the
    # chain rule is broken or every pusher missed the T.
    max_grad_magnitude = max(float(jnp.abs(g_contact).max()), float(jnp.abs(g_angle).max()))
    assert max_grad_magnitude > 1e-8, (
        f"gradients are identically zero; autodiff isn't flowing through the rollout "
        f"(max |g| = {max_grad_magnitude:.3e})"
    )

    # Finite-difference sanity check on one env: along the combined
    # gradient direction, a small +ε step should *increase* the loss
    # (first-order Taylor with a non-zero gradient).
    eps = 1e-4
    cp_fd = contact_point + eps * g_contact
    ang_fd = angle + eps * g_angle
    loss_plus = cost(cp_fd, ang_fd)
    expected_delta = eps * (float(jnp.dot(g_contact, g_contact)) + float(jnp.dot(g_angle, g_angle)))
    actual_delta = float(loss_plus) - float(loss)
    # Loose tolerance — physics is stiff and we're only doing a 1st-order check.
    assert actual_delta > 0.1 * expected_delta, (
        f"finite-difference check failed: actual Δloss={actual_delta:.3e}, "
        f"expected ≈ {expected_delta:.3e} along the gradient direction"
    )


# python -m pytest tests/test_core.py::test_step_pure_soft_matches_hard --capture=no
def test_step_pure_soft_matches_hard() -> None:
    """One-hot face_weights should reproduce the hard integer-face result."""
    nenvs = 9  # must be a perfect square for env layout
    n_sim_steps = 30

    env = PushTEnv(nenvs=nenvs, record_video=False, visualize=False)
    env.reset(seed=0)
    data0 = env.data

    # Cover all 6 faces + 3 repeats.
    face_int = jnp.array([0, 1, 2, 3, 4, 5, 0, 2, 4], dtype=jnp.int32)
    contact_point = jnp.full((nenvs,), 0.5, dtype=jnp.float32)
    angle = jnp.full((nenvs,), np.pi / 2, dtype=jnp.float32)

    _, t_poses_hard, t_dists_hard, _ = env.step_pure(
        data=data0,
        face=face_int,
        contact_point=contact_point,
        angle=angle,
        n_sim_steps=n_sim_steps,
    )

    face_weights = jax.nn.one_hot(face_int, num_classes=6, dtype=jnp.float32)
    _, t_poses_soft, t_dists_soft, _ = env.step_pure_soft(
        data=data0,
        face_weights=face_weights,
        contact_point=contact_point,
        angle=angle,
        n_sim_steps=n_sim_steps,
    )

    # Tiny numerical drift acceptable (matmul vs gather path through physics).
    np.testing.assert_allclose(np.asarray(t_poses_hard), np.asarray(t_poses_soft), atol=1e-8)
    np.testing.assert_allclose(np.asarray(t_dists_hard), np.asarray(t_dists_soft), atol=1e-8)


# python -m pytest tests/test_core.py::test_step_pure_soft_is_differentiable_in_face --capture=no
def test_step_pure_soft_is_differentiable_in_face() -> None:
    """Gradient must flow into `face_weights` (the whole point of the soft path)."""
    nenvs = 4
    n_sim_steps = 50

    env = PushTEnv(nenvs=nenvs, record_video=False, visualize=False)
    env.reset(seed=0)
    data0 = env.data

    contact_point = jnp.full((nenvs,), 0.5, dtype=jnp.float32)
    angle = jnp.full((nenvs,), np.pi / 2, dtype=jnp.float32)
    # Bias each env toward a different face so the softmaxed weights are sharp
    # enough to put the pusher on an actual edge (uniform weights would land
    # the contact point inside the T block and blow up physics).
    face_logits = jnp.zeros((nenvs, 6), dtype=jnp.float32).at[jnp.arange(nenvs), jnp.array([0, 1, 2, 3])].set(5.0)

    def cost(face_logits: jnp.ndarray) -> jnp.ndarray:
        face_weights = jax.nn.softmax(face_logits, axis=-1)
        _, _, t_distances, _ = env.step_pure_soft(
            data=data0,
            face_weights=face_weights,
            contact_point=contact_point,
            angle=angle,
            n_sim_steps=n_sim_steps,
        )
        return t_distances[:, -1].sum()

    loss, g_logits = jax.jit(jax.value_and_grad(cost))(face_logits)

    assert g_logits.shape == (nenvs, 6)
    assert jnp.isfinite(loss), f"loss not finite: {loss}"
    assert jnp.all(jnp.isfinite(g_logits)), f"g_logits has NaN/Inf: {g_logits}"

    max_grad_magnitude = float(jnp.abs(g_logits).max())
    assert max_grad_magnitude > 1e-8, (
        f"face-weight gradients are identically zero; autodiff isn't flowing "
        f"through the soft face gather (max |g| = {max_grad_magnitude:.3e})"
    )


# python -m pytest tests/test_core.py::test_t_pose_unchanged --capture=no
def test_t_pose_unchanged():
    """Debug a specific test"""
    env = PushTEnv(nenvs=1, record_video=True, visualize=False)
    env.reset(seed=0, t_poses=np.array([[0.38668, 1.22616, -2.68507]], dtype=np.float32))

    data: js.data.JaxSimModelData = env.data
    # step_pure / _plan_push_jax expect per-env scalars shaped (nenvs,), not (nenvs, 1);
    # column vectors make contact_point[:, None] rank-3 and break the rotation einsum.
    faces = np.array([0], dtype=np.int32).reshape(-1)
    contact_point = np.array([0.6577187796214783], dtype=np.float32).reshape(-1)
    angle = np.array([0.7242890193265389], dtype=np.float32).reshape(-1)
    _, t_poses, t_dists, jpos_traj = env.step_pure(
        data=data,
        face=jnp.asarray(faces, dtype=jnp.int32).reshape(-1),
        contact_point=jnp.asarray(contact_point, dtype=jnp.float32).reshape(-1),
        angle=jnp.asarray(angle, dtype=jnp.float32).reshape(-1),
        n_sim_steps=50,
    )
    print(f"t_dists: {t_dists}")
    print(f"t_poses: {t_poses}")
    env.save_video_from_jpos_traj("/tmp/test_t_pose_unchanged.mp4", jpos_traj)
    print("xdg-open /tmp/test_t_pose_unchanged.mp4")
