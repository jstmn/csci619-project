import numpy as np
from pathlib import Path
import pytest


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
