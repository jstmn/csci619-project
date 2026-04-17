from pusht619.core import Action, PushTEnv
import numpy as np
from pathlib import Path


"""Example usage:

python scripts/main.py
"""




if __name__ == "__main__":
    env = PushTEnv(nenvs=9, record_video=True, visualize=True)

    for i in range(100):
        print(f"Step {i}")
        action = Action(
            face=np.random.randint(0, 6, size=(9, 1)),
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
    save_filepath = Path("/tmp/test_PushTEnv_smoke_test.mp4")
    env.save_video(save_filepath)
    assert save_filepath.exists()
    print(f"Saved video to {save_filepath}")
    print(f"xdg-open {save_filepath}")