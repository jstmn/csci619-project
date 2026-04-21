"""Chained deployment rollout of the trained SurCo prior.

Loads the saved policy from CKPT_PATH, applies it for N_PUSHES consecutive
pushes per env (state persists between pushes — mirrors real deployment),
and writes one video showing the full chain. Run as a script:

    python tests/test_chain_deploy.py

Skipped under pytest if the checkpoint doesn't exist yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pusht619.core import ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, Action, PushTEnv
from scripts.training_loop import (
    CKPT_PATH,
    N_FACES,
    N_SIM_STEPS,
    SurCoPrior,
    _solve_pure_callback,
    extract_y,
    load_prior_params,
)


N_ENVS = 4
N_PUSHES = 6
SEED = 2024
VIDEO_FPS = 60  # slowed down from the 1000 fps default (= 1/time_step)
VIDEO_PATH = Path(__file__).resolve().parent.parent / "surco_eval_videos" / "chain_deploy.mp4"


def _set_overhead_camera(env: PushTEnv, distance: float = 3.0):
    """Mutate the recorded camera to look straight down at the workspace.

    The mjcf-baked "t_block_cam" starts at azimuth=150, elevation=-30. We
    override its world pose on every MjModel helper: position directly above
    the workspace center, orientation = identity (mujoco cameras look along
    their local -Z, so identity quat = looking straight down with +Y up).
    """
    lookat = np.array([0.75, 0.75, 0.1], dtype=np.float32)
    cam_pos = lookat + np.array([0.0, 0.0, distance])
    cam_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    for helper in env._mj_model_helpers:
        model = helper.model
        cam_id = 0  # only one camera in the mjcf
        model.cam_pos[cam_id] = cam_pos
        model.cam_quat[cam_id] = cam_quat


def run_chain(
    params,
    n_envs: int = N_ENVS,
    n_pushes: int = N_PUSHES,
    n_sim_steps: int = N_SIM_STEPS,
    seed: int = SEED,
    video_path: Path | None = VIDEO_PATH,
):
    env = PushTEnv(nenvs=n_envs, record_video=video_path is not None, visualize=False)
    if video_path is not None:
        _set_overhead_camera(env)
        env._recorder.fps = VIDEO_FPS
    env.reset(seed=seed)
    target_xy = jnp.asarray(env.target_poses[:, :2])

    dist_history = np.zeros((n_envs, n_pushes + 1), dtype=np.float32)
    t_xy0 = np.asarray(env._poses[:, :2])
    dist_history[:, 0] = np.linalg.norm(t_xy0 - np.asarray(target_xy), axis=-1)

    for k in range(n_pushes):
        y = extract_y(env.data, target_xy)
        c = SurCoPrior().apply(params, y)
        face_onehot = np.asarray(_solve_pure_callback(c[:, :N_FACES]))
        face = np.argmax(face_onehot, axis=-1).astype(np.int32).reshape(-1, 1)
        lo_cp, hi_cp = CONTACT_POINT_BOUNDS
        lo_ang, hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
        contact = np.asarray(lo_cp + (hi_cp - lo_cp) * jax.nn.sigmoid(c[:, 6])).reshape(-1, 1)
        angle = np.asarray(lo_ang + (hi_ang - lo_ang) * jax.nn.sigmoid(c[:, 7])).reshape(-1, 1)

        result = env.step(
            Action(face=face, contact_point=contact, angle=angle),
            n_sim_steps=n_sim_steps,
        )
        dist_history[:, k + 1] = np.asarray(result.t_distances[:, -1])

        mean_d = float(dist_history[:, k + 1].mean())
        print(f"  push {k + 1}/{n_pushes}: mean_dist = {mean_d:.4f} m")

    print("\n  Per-env distance trajectory (push 0..N):")
    for i in range(n_envs):
        traj = "  ".join(f"{d:.3f}" for d in dist_history[i])
        print(f"    env {i}: {traj}")

    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        env.save_video(video_path)
        print(f"\n  video → {video_path}")

    return dist_history


@pytest.mark.skipif(not CKPT_PATH.exists(), reason="no trained checkpoint yet")
def test_chain_deploy_monotone_on_average():
    """Mean distance after N pushes should not be worse than initial."""
    params = load_prior_params()
    dists = run_chain(params, video_path=None)
    assert np.isfinite(dists).all(), "chain produced NaN"
    assert dists[:, -1].mean() <= dists[:, 0].mean() + 1e-6, (
        f"policy made it worse: {dists[:, 0].mean():.4f} → {dists[:, -1].mean():.4f}"
    )


if __name__ == "__main__":
    if not CKPT_PATH.exists():
        raise SystemExit(f"No checkpoint at {CKPT_PATH}. Run `python scripts/training_loop.py` first.")
    print("=" * 70)
    print(f"Chained deployment  ({N_ENVS} envs × {N_PUSHES} pushes, seed={SEED})")
    print("=" * 70)
    params = load_prior_params()
    run_chain(params)
