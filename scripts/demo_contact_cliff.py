"""Demo: the spring-damper NaN cliff on face 0.

For each angle in ANGLES we build a single-env PushTEnv, run one push
(face=0, contact=0.75), report whether the rollout stayed finite and at
which sim step it first blew up, and save a per-angle video.

Two safe angles + two in the NaN zone makes the cliff obvious.

Run it:
    python scripts/demo_contact_cliff.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pusht619.core import Action, PushTEnv

jax.config.update("jax_enable_x64", True)


N_SIM_STEPS = 50
FACE = 0
CONTACT = 0.75
ANGLES = [2.0, 2.4, 2.7, 2.9]  # two safe, two in the NaN zone
RESET_SEED = 0
VIDEO_FPS = 30  # baked is 1000 fps — playback would be 0.05 s otherwise
OUT_DIR = Path("/tmp")


def run_one(angle: float) -> tuple[int, float]:
    env = PushTEnv(nenvs=1, record_video=True, visualize=False)
    env.reset(seed=RESET_SEED)

    face_j = jnp.asarray([FACE], dtype=jnp.int32)
    contact_j = jnp.asarray([CONTACT], dtype=jnp.float64)
    angle_j = jnp.asarray([angle], dtype=jnp.float64)

    _, t_poses, t_distances, _ = env.step_pure(
        data=env.data,
        face=face_j,
        contact_point=contact_j,
        angle=angle_j,
        n_sim_steps=N_SIM_STEPS,
    )
    t_poses_np = np.asarray(t_poses)[0]
    t_distances_np = np.asarray(t_distances)[0]
    pose_bad = ~np.isfinite(t_poses_np).all(axis=-1)
    first_bad = int(np.argmax(pose_bad)) if pose_bad.any() else -1
    final = float(t_distances_np[-1])

    env.reset(seed=RESET_SEED)
    action = Action(
        face=np.asarray([[FACE]], dtype=np.int32),
        contact_point=np.asarray([[CONTACT]], dtype=np.float64),
        angle=np.asarray([[angle]], dtype=np.float64),
    )
    env.step(action, n_sim_steps=N_SIM_STEPS)
    env._recorder.fps = VIDEO_FPS
    out = OUT_DIR / f"demo_contact_cliff_angle_{angle:.2f}.mp4"
    env.save_video(out)
    return first_bad, final, out


def main():
    print(f"face={FACE}, contact={CONTACT}, N_SIM_STEPS={N_SIM_STEPS}")
    print(f"{'angle':>6}  {'first_bad_step':>14}  {'final_dist':>10}  video")
    for a in ANGLES:
        first_bad, final, out = run_one(a)
        tag = str(first_bad) if first_bad >= 0 else "—"
        final_str = "NaN" if not np.isfinite(final) else f"{final:.4f}"
        print(f"{a:>6.3f}  {tag:>14}  {final_str:>10}  {out}")


if __name__ == "__main__":
    main()
