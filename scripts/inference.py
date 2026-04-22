import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from pusht619.core import Action, PushTEnv
from pusht619.models import MLP, ActionSolver


N_SIM_STEPS = 50
RESET_SEED = 0

def load_checkpoint(checkpoint_path: Path) -> tuple[MLP, list[tuple[jnp.ndarray, jnp.ndarray]]]:
    checkpoint = np.load(checkpoint_path)
    layer_sizes = checkpoint["layer_sizes"].astype(np.int32).tolist()
    # assert layer_sizes[0] == 11, f"Expected context dim 11, got {layer_sizes[0]}"
    hidden_dims = tuple(layer_sizes[1:-1])
    mlp = MLP(context_dim=layer_sizes[0], hidden_dims=hidden_dims)

    cp_bounds = tuple(float(x) for x in checkpoint["cp_bounds"])
    ang_bounds = tuple(float(x) for x in checkpoint["ang_bounds"])
    mlp.cp_bounds = cp_bounds
    mlp.ang_bounds = ang_bounds

    params = []
    n_layers = len(layer_sizes) - 1
    for layer_idx in range(n_layers):
        w = jnp.asarray(checkpoint[f"layer_{layer_idx}_w"], dtype=jnp.float32)
        b = jnp.asarray(checkpoint[f"layer_{layer_idx}_b"], dtype=jnp.float32)
        params.append((w, b))
    return mlp, params


def format_solver_params(c: np.ndarray) -> str:
    n_faces = 6
    face_costs = np.array2string(c[:n_faces], precision=4, floatmode="fixed")
    cp_targets = np.array2string(c[n_faces : 2 * n_faces], precision=4, floatmode="fixed")
    ang_targets = np.array2string(c[2 * n_faces : 3 * n_faces], precision=4, floatmode="fixed")
    return (
        f"face_costs={face_costs}\n"
        f"    cp_targets={cp_targets}\n"
        f"    ang_targets={ang_targets}"
    )


def main(
    checkpoint: Path,
    n_envs: int,
    seed: int,
    random_t_pose: bool,
    record_video: bool,
    save_dir: Path | None,
    verbosity: int,
    n_planning_steps: int,
):
    mlp, params = load_checkpoint(checkpoint)
    solver = ActionSolver()

    env = PushTEnv(nenvs=n_envs, record_video=record_video, visualize=False)
    if random_t_pose:
        env.reset()
    else:
        env.reset(seed=seed)

    if save_dir is None:
        timestamp = datetime.now().strftime("%d__%H:%M:%S")
        save_dir = Path(f"logs/{timestamp}__inference__n-envs:{n_envs}")
    save_dir.mkdir(parents=True, exist_ok=True)

    all_distances = []  # list of (n_envs, N_SIM_STEPS+1) arrays

    for i in range(n_planning_steps):
        print(f"Running planning step {i + 1}/{n_planning_steps}")
        ctx = env.get_context_vector(env.data)
        c = np.asarray(mlp.apply(params, ctx))
        x_star = solver.solve_batch(c)
        face_onehot = x_star[:, :6]
        face_idx = np.argmax(face_onehot, axis=-1)
        contact_point = x_star[:, 6]
        angle = x_star[:, 7]

        print("Chosen actions:")
        print(f"  face={face_idx.tolist()}")
        print(f"  contact_point={np.asarray(contact_point).round(5).tolist()}")
        print(f"  angle={np.asarray(angle).round(5).tolist()}")

        if verbosity > 0:
            for env_idx in range(n_envs):
                print(f"\nEnv {env_idx}")
                print(f"  context={np.asarray(ctx[env_idx]).round(5).tolist()}")
                print(f"  {format_solver_params(c[env_idx])}")

        action = Action(
            face=face_idx.astype(np.int32).reshape(-1, 1),
            contact_point=np.asarray(contact_point, dtype=np.float32).reshape(-1, 1),
            angle=np.asarray(angle, dtype=np.float32).reshape(-1, 1),
        )
        result = env.step(
            action=action,
            n_sim_steps=N_SIM_STEPS,
            check_t_displacement=False,
        )
        t_distances = np.asarray(result.t_distances)  # (n_envs, N_SIM_STEPS+1)
        all_distances.append(t_distances)
        final_dists = t_distances[:, -1]
        print(f"\nFinal distances={np.round(final_dists, 6).tolist()}")
        print(f"Mean final distance={float(np.nanmean(final_dists)):.6f} [m]")
        # print(f"Std final distance={float(np.nanstd(final_dists)):.6f} [m]")

    # ── Distance plot ──────────────────────────────────────────────────────────
    # Concatenate across planning steps: (n_envs, total_timesteps)
    dist_all = np.concatenate(all_distances, axis=1)
    timesteps = np.arange(dist_all.shape[1])

    fig, ax = plt.subplots(figsize=(10, 4))
    for env_idx in range(n_envs):
        ax.plot(timesteps, dist_all[env_idx], alpha=0.4, linewidth=0.8)
    ax.plot(timesteps, np.nanmean(dist_all, axis=0), color="black", linewidth=2.0, label="mean")
    for step in range(1, n_planning_steps):
        ax.axvline(step * (N_SIM_STEPS + 1), color="gray", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance [m]")
    ax.set_title(f"T distance over time  (n_envs={n_envs}, n_planning_steps={n_planning_steps})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    video_path = save_dir / f"inference__n-steps:{n_planning_steps}__n-envs:{n_envs}.mp4"
    plot_path = str(video_path).replace(".mp4", ".png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved distance plot to {plot_path}")
    os.system(f"xdg-open {plot_path}")

    if record_video:
        env.save_video(video_path)
        print(f"Saved video to {video_path}")
        os.system(f"xdg-open {save_dir}")

""" Example usage:

CKPT=logs/22__00:59:01__single_step__n-envs:64__SurCo-prior__Adam__8dim__lr:0.01__grad-clip:1.0/checkpoints/mlp_iter_1000.npz
CKPT=logs/22__01:03:11__single_step__n-envs:64__SurCo-prior__Adam__8dim__lr:0.01__grad-clip:1.0/checkpoints/mlp_iter_1000.npz
CKPT=logs/22__01:08:13__single_step__n-envs:16__SurCo-prior__Adam__8dim__lr:0.01__grad-clip:1.0/checkpoints/mlp_iter_1000.npz

python scripts/inference.py --checkpoint ${CKPT} --n-envs 16 --random-t-pose --record-video
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved .npz MLP checkpoint")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=RESET_SEED)
    parser.add_argument("--random-t-pose", action="store_true", help="Use a random reset instead of a fixed seed")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--n-planning-steps", type=int, default=10)
    args = parser.parse_args()

    save_dir = str(args.checkpoint).replace(".npz", "")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"xdg-open {save_dir}")
    assert args.n_envs > 0, "n_envs must be positive"
    assert args.verbosity in [0, 1, 2], "verbosity must be 0, 1, or 2"
    main(
        checkpoint=args.checkpoint,
        n_envs=args.n_envs,
        seed=args.seed,
        random_t_pose=args.random_t_pose,
        record_video=args.record_video,
        save_dir=save_dir,
        verbosity=args.verbosity,
        n_planning_steps=args.n_planning_steps,
    )
