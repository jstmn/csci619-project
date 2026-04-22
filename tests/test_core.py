from time import time

PROGRAM_START_TIME = time()
import numpy as np
from pathlib import Path
import pytest
from uuid import uuid4
import jaxsim as js
import jax.numpy as jnp
import jax
from termcolor import cprint

from pusht619.core import (
    ANGLE_BOUNDS,
    CONTACT_POINT_BOUNDS,
    WORKSPACE_HEIGHT,
    WORKSPACE_WIDTH,
    NUM_FACES,
    Action,
    PushTEnv,
)

np.set_printoptions(precision=4, suppress=True)


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


# python -m pytest tests/test_core.py::test_step_pure_sweeps_faces_and_action_grid_from_center --capture=no
def test_step_pure_sweeps_faces_and_action_grid_from_center() -> None:
    """Sweep all faces over a contact/angle grid from a centered T pose."""
    print(f"Test load time: {time() - PROGRAM_START_TIME:.2f} s")
    env = PushTEnv(nenvs=1, record_video=False, visualize=False)
    n_sim_steps = 50
    center_t_pose = np.array([[WORKSPACE_WIDTH / 2, WORKSPACE_HEIGHT / 2, 0.0]], dtype=np.float32)
    contact_points = np.linspace(CONTACT_POINT_BOUNDS[0], CONTACT_POINT_BOUNDS[1], num=5, dtype=np.float32)
    angles = np.linspace(float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1]), num=5, dtype=np.float32)

    moved_cases = 0
    moved_by_face = np.zeros(NUM_FACES, dtype=bool)

    for face in range(NUM_FACES):
        for contact_point in contact_points:
            for angle in angles:
                print(f"face/cp/angle: {face}/{contact_point:.3f}/{angle:.3f}", end=" | ")
                env.reset(seed=0, t_poses=center_t_pose)
                np.testing.assert_allclose(env.t_poses, center_t_pose, atol=1e-6)
                _, t_poses, t_dists, _ = env.step_pure(
                    data=env.data,
                    face=jnp.array([face], dtype=jnp.int32),
                    contact_point=jnp.array([contact_point], dtype=jnp.float32),
                    angle=jnp.array([angle], dtype=jnp.float32),
                    n_sim_steps=n_sim_steps,
                    check_t_displacement=False,
                )
                t_poses_np = np.asarray(t_poses)
                t_dists_np = np.asarray(t_dists)
                assert np.all(np.isfinite(t_poses_np)), (
                    f"Non-finite T poses for face={face}, contact_point={contact_point:.3f}, angle={angle:.3f}"
                )
                assert np.all(np.isfinite(t_dists_np)), (
                    f"Non-finite T distances for face={face}, contact_point={contact_point:.3f}, angle={angle:.3f}"
                )
                displacement = float(np.linalg.norm(t_poses_np[0, -1, :2] - center_t_pose[0, :2]))
                if displacement <= 1e-3:
                    cprint(f"DID NOT MOVE, displacement={100 * displacement:.3f} [cm]", "red")
                    continue
                else:
                    print(f"moved, displacement={100 * displacement:.3f} [cm]")

                moved_cases += 1
                moved_by_face[face] = True

    assert moved_cases > 0, "Sweep never moved the T block from the centered pose."
    assert moved_by_face.all(), f"At least one face never moved the T block: {np.where(~moved_by_face)[0].tolist()}"


# python -m pytest tests/test_core.py::test_step_pure_sweeps_faces_and_action_grid_multi_step --capture=no
def test_step_pure_sweeps_faces_and_action_grid_multi_step() -> None:
    """Sweep all faces over a contact/angle grid from a centered T pose using a multi-step solver."""
    print(f"Test load time: {time() - PROGRAM_START_TIME:.2f} s")
    jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/jax_pusht619"))
    env = PushTEnv(nenvs=1, record_video=True, visualize=False)
    n_sim_steps = 50
    logs_dir = Path("logs/test_step_pure_sweeps_faces_and_action_grid_multi_step")
    logs_dir.mkdir(parents=True, exist_ok=True)
    center_t_pose = np.array([[WORKSPACE_WIDTH / 2, WORKSPACE_HEIGHT / 2, 0.0]], dtype=np.float32)
    contact_points = np.linspace(CONTACT_POINT_BOUNDS[0], CONTACT_POINT_BOUNDS[1], num=3, dtype=np.float32)
    angles = np.linspace(float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1]), num=3, dtype=np.float32)

    moved_cases = 0
    moved_by_first_face = np.zeros(NUM_FACES, dtype=bool)
    moved_by_second_face = np.zeros(NUM_FACES, dtype=bool)

    def save_rollout_video(
        stage: str,
        jpos_traj: np.ndarray,
        face_1: int,
        contact_point_1: float,
        angle_1: float,
        face_2: int,
        contact_point_2: float,
        angle_2: float,
    ) -> Path:
        save_path = logs_dir / (
            f"multi_step__{stage}"
            f"__a1:({face_1}, {contact_point_1:.3f}, {angle_1:.3f})"
            f"__a2:({face_2}, {contact_point_2:.3f}, {angle_2:.3f}).mp4"
        )
        env.save_video_from_jpos_traj(save_path, np.asarray(jpos_traj))
        print(f"saved video to {save_path}")
        return save_path

    for face_1 in range(NUM_FACES):
        for contact_point_1 in contact_points:
            for angle_1 in angles:
                for face_2 in range(NUM_FACES):
                    for contact_point_2 in contact_points:
                        for angle_2 in angles:
                            print(
                                f"faces: ({face_1}, {face_2}), cp: ({contact_point_1:.3f}, {contact_point_2:.3f}), angle: ({angle_1:.3f}, {angle_2:.3f})",
                                end=" | ",
                            )
                            env.reset(seed=0, t_poses=center_t_pose)
                            np.testing.assert_allclose(env.t_poses, center_t_pose, atol=1e-6)

                            data_after_1, t_poses_1, t_dists_1, jpos_traj_1 = env.step_pure(
                                data=env.data,
                                face=jnp.array([face_1], dtype=jnp.int32),
                                contact_point=jnp.array([contact_point_1], dtype=jnp.float32),
                                angle=jnp.array([angle_1], dtype=jnp.float32),
                                n_sim_steps=n_sim_steps,
                                check_t_displacement=False,
                            )
                            t_poses_1_np = np.asarray(t_poses_1)
                            t_dists_1_np = np.asarray(t_dists_1)
                            assert np.all(np.isfinite(t_poses_1_np)), (
                                f"Non-finite T poses for first action "
                                f"face={face_1}, contact_point={contact_point_1:.3f}, angle={angle_1:.3f}"
                            )
                            assert np.all(np.isfinite(t_dists_1_np)), (
                                f"Non-finite T distances for first action "
                                f"face={face_1}, contact_point={contact_point_1:.3f}, angle={angle_1:.3f}"
                            )

                            displacement_1 = float(np.linalg.norm(t_poses_1_np[0, -1, :2] - center_t_pose[0, :2]))
                            action_1_did_not_move = displacement_1 <= 1e-3
                            if displacement_1 <= 1e-3:
                                cprint(f"ACTION 1 DID NOT MOVE, displacement={100 * displacement_1:.3f} [cm]", "red")

                            _, t_poses_2, t_dists_2, jpos_traj_2 = env.step_pure(
                                data=data_after_1,
                                face=jnp.array([face_2], dtype=jnp.int32),
                                contact_point=jnp.array([contact_point_2], dtype=jnp.float32),
                                angle=jnp.array([angle_2], dtype=jnp.float32),
                                n_sim_steps=n_sim_steps,
                                check_t_displacement=False,
                            )
                            t_poses_2_np = np.asarray(t_poses_2)
                            t_dists_2_np = np.asarray(t_dists_2)
                            assert np.all(np.isfinite(t_poses_2_np)), (
                                f"Non-finite T poses for second action "
                                f"face={face_2}, contact_point={contact_point_2:.3f}, angle={angle_2:.3f}"
                            )
                            assert np.all(np.isfinite(t_dists_2_np)), (
                                f"Non-finite T distances for second action "
                                f"face={face_2}, contact_point={contact_point_2:.3f}, angle={angle_2:.3f}"
                            )

                            displacement_2 = float(np.linalg.norm(t_poses_2_np[0, -1, :2] - t_poses_1_np[0, -1, :2]))
                            total_displacement = float(np.linalg.norm(t_poses_2_np[0, -1, :2] - center_t_pose[0, :2]))
                            action_2_did_not_move = displacement_2 <= 1e-3
                            if action_1_did_not_move and action_2_did_not_move:
                                video_stage = "no_move_both"
                            elif action_1_did_not_move:
                                video_stage = "no_move_action1"
                            elif action_2_did_not_move:
                                video_stage = "no_move_action2"
                            else:
                                video_stage = "moved"
                            save_rollout_video(
                                video_stage,
                                np.concatenate([np.asarray(jpos_traj_1), np.asarray(jpos_traj_2)], axis=1),
                                face_1,
                                float(contact_point_1),
                                float(angle_1),
                                face_2,
                                float(contact_point_2),
                                float(angle_2),
                            )
                            if displacement_2 <= 1e-3:
                                cprint(f"ACTION 2 DID NOT MOVE, displacement={100 * displacement_2:.3f} [cm]", "red")
                                continue

                            print(
                                f"disp_1={100 * displacement_1:.3f} [cm], "
                                f"disp_2={100 * displacement_2:.3f} [cm], "
                                f"disp_total={100 * total_displacement:.3f} [cm]"
                            )
                            moved_cases += 1
                            moved_by_first_face[face_1] = True
                            moved_by_second_face[face_2] = True

    assert moved_cases > 0, "Sweep never moved the T block from the centered pose."
    assert moved_by_first_face.all(), (
        f"At least one first-action face never moved the T block: {np.where(~moved_by_first_face)[0].tolist()}"
    )
    assert moved_by_second_face.all(), (
        f"At least one second-action face never moved the T block: {np.where(~moved_by_second_face)[0].tolist()}"
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
