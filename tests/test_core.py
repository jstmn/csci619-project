import numpy as np
from pathlib import Path
import pytest


from pusht619.core import Action, PushTEnv



# python -m pytest tests/test_core.py::test_PushTEnv_smoke_test
def test_PushTEnv_smoke_test() -> None:
    env = PushTEnv(nenvs=9, record_video=True)
    action = Action(
        face=np.array([0, 1, 2, 3, 4, 5, 0, 1, 2]).reshape(9, 1),
        contact_point=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).reshape(9, 1),
        angle=np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]).reshape(9, 1),
        push_distance=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).reshape(9, 1),
    )
    n_sim_steps = 100
    result = env.step(action, n_sim_steps=n_sim_steps)
    assert result.action == action
    assert result.t_poses.shape == (9, n_sim_steps, 3)
    assert result.t_distances.shape == (9, n_sim_steps)


    # Next, save a video
    env.save_video("/tmp/test_PushTEnv_smoke_test.mp4")
    assert Path("/tmp/test_PushTEnv_smoke_test.mp4").exists()

    # Now, reset the environment
    env.reset()

    # Try saving a video again, it should fail because the environment has not been stepped
    with pytest.raises(AssertionError):
        env.save_video("/tmp/test_PushTEnv_smoke_test_2.mp4")