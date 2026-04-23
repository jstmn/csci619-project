"""Generate (state, action, next_state) pairs with random T poses and actions.

Small-scale debug dataset. For each sample we:
  1. reset() — randomizes T pose and goal pose inside workspace,
  2. draw a random Action (face, contact_point, angle),
  3. env.step(...) — record T pose before and after.

Output: npz with
  t_before : (N, 3)  [x, y, theta]
  goal     : (N, 3)
  face     : (N,) int
  contact  : (N,)
  angle    : (N,)
  t_after  : (N, 3)

Run:
    python scripts/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from pusht619.core import Action, PushTEnv


N_ENVS = 16  # samples per batch (one env.step produces N_ENVS samples)
N_BATCHES = 4  # total samples = N_ENVS * N_BATCHES
N_SIM_STEPS = 50
OUT_PATH = Path(__file__).resolve().parent.parent / "debug_dataset.npz"


def main():
    env = PushTEnv(nenvs=N_ENVS, record_video=False, visualize=False)
    rng = np.random.default_rng(seed=0)

    t_before_list, goal_list = [], []
    face_list, contact_list, angle_list = [], [], []
    t_after_list = []

    for b in range(N_BATCHES):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        t_before = env._poses.copy()
        goal = env.target_poses.copy()

        face = rng.integers(0, 6, size=(N_ENVS, 1)).astype(np.int32)
        contact = rng.uniform(0.0, 1.0, size=(N_ENVS, 1)).astype(np.float32)
        angle = rng.uniform(0.0, np.pi, size=(N_ENVS, 1)).astype(np.float32)

        result = env.step(
            Action(face=face, contact_point=contact, angle=angle),
            n_sim_steps=N_SIM_STEPS,
        )
        t_after = np.asarray(result.t_poses[:, -1, :])

        t_before_list.append(t_before)
        goal_list.append(goal)
        face_list.append(face.reshape(-1))
        contact_list.append(contact.reshape(-1))
        angle_list.append(angle.reshape(-1))
        t_after_list.append(t_after)

        nan_mask = ~np.isfinite(t_after).all(axis=-1)
        disp = np.linalg.norm(t_after[:, :2] - t_before[:, :2], axis=-1)
        mean_disp = float(np.nanmean(disp))
        print(f"  batch {b + 1}/{N_BATCHES}: mean |Δxy| = {mean_disp:.4f} m  NaN envs = {int(nan_mask.sum())}/{N_ENVS}")
        for i in np.where(nan_mask)[0]:
            print(
                f"    NaN env {i}: t_before={t_before[i].round(3).tolist()}  "
                f"goal={goal[i].round(3).tolist()}  "
                f"face={int(face[i, 0])}  contact={float(contact[i, 0]):.3f}  "
                f"angle={float(angle[i, 0]):.3f}"
            )

    data = dict(
        t_before=np.concatenate(t_before_list, axis=0),
        goal=np.concatenate(goal_list, axis=0),
        face=np.concatenate(face_list, axis=0),
        contact=np.concatenate(contact_list, axis=0),
        angle=np.concatenate(angle_list, axis=0),
        t_after=np.concatenate(t_after_list, axis=0),
    )
    np.savez(OUT_PATH, **data)
    print(f"\nSaved {data['t_before'].shape[0]} samples → {OUT_PATH}")


if __name__ == "__main__":
    main()
