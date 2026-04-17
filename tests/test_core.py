import numpy as np
from pathlib import Path
import pytest

import jax
import jax.numpy as jnp

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
        contact_point=np.array([[0.5]], dtype=np.float64),
        angle=np.array([[np.pi / 2]], dtype=np.float64),
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
    target_xy = jnp.asarray(env.target_poses[:, :2])

    # Choose faces that point roughly toward the targets so the pusher actually
    # makes contact, ensuring a non-trivial gradient signal.
    face = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    def cost(contact_point: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
        _, _, t_distances, _ = env.step_pure(
            data=data0,
            face=face,
            contact_point=contact_point,
            angle=angle,
            target_xy=target_xy,
            n_sim_steps=n_sim_steps,
        )
        # Scalar: sum of final-step distances across envs.
        return t_distances[:, -1].sum()

    contact_point = jnp.full((nenvs,), 0.5, dtype=jnp.float64)
    angle = jnp.full((nenvs,), np.pi / 2, dtype=jnp.float64)

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
