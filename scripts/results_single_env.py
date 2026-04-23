"""Evaluate a single PushT problem with exhaustive and random action search."""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]

from pusht619.core import ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES, Action, PushTEnv


N_SIM_STEPS = 25
N_PLOT_ITERS = 10


@dataclass(frozen=True)
class EvaluatedAction:
    face: int
    contact_point: float
    angle: float
    final_distance: float


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    loss: float
    mean_final_distance: float
    final_distance_per_env: list[float]


def build_action_grid(contact_point_bins: int, angle_bins: int) -> list[tuple[int, float, float]]:
    contact_points = np.linspace(CONTACT_POINT_BOUNDS[0], CONTACT_POINT_BOUNDS[1], num=contact_point_bins, dtype=np.float32)
    angles = np.linspace(float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1]), num=angle_bins, dtype=np.float32)

    grid: list[tuple[int, float, float]] = []
    for face in range(NUM_FACES):
        for contact_point in contact_points:
            for angle in angles:
                grid.append((face, float(contact_point), float(angle)))
    return grid


def rollout_action(env: PushTEnv, seed: int, face: int, contact_point: float, angle: float) -> tuple[float, float]:
    # Recreate the exact same single-env problem for every candidate action.
    env.reset(seed=seed)
    action = Action(
        face=np.array([[face]], dtype=np.int32),
        contact_point=np.array([[contact_point]], dtype=np.float32),
        angle=np.array([[angle]], dtype=np.float32),
    )
    result = env.step(action=action, n_sim_steps=N_SIM_STEPS, check_t_displacement=False)
    t_distances = np.asarray(result.t_distances)[0]
    return float(t_distances[0]), float(t_distances[-1])


def evaluate_optimal_action(env: PushTEnv, seed: int, action_grid: list[tuple[int, float, float]]) -> tuple[EvaluatedAction, float]:
    best_action: EvaluatedAction | None = None
    starting_distance: float | None = None
    for face, contact_point, angle in tqdm(action_grid, desc="Optimal actions", unit="action"):
        rollout_starting_distance, final_distance = rollout_action(env, seed, face, contact_point, angle)
        if starting_distance is None:
            starting_distance = rollout_starting_distance
        candidate = EvaluatedAction(
            face=face,
            contact_point=contact_point,
            angle=angle,
            final_distance=final_distance,
        )
        if best_action is None or candidate.final_distance < best_action.final_distance:
            best_action = candidate
    assert best_action is not None, "Action grid must not be empty."
    assert starting_distance is not None, "Action grid must not be empty."
    return best_action, starting_distance


def evaluate_random_actions(
    env: PushTEnv,
    seed: int,
    action_grid: list[tuple[int, float, float]],
    random_sample_k: int,
) -> tuple[list[EvaluatedAction], float]:
    rng = np.random.default_rng(seed=seed)
    replace = random_sample_k > len(action_grid)
    sampled_indices = rng.choice(len(action_grid), size=random_sample_k, replace=replace)

    results: list[EvaluatedAction] = []
    starting_distance: float | None = None
    for idx in tqdm(np.asarray(sampled_indices).tolist(), desc="Random actions", unit="action"):
        face, contact_point, angle = action_grid[idx]
        rollout_starting_distance, final_distance = rollout_action(env, seed, face, contact_point, angle)
        if starting_distance is None:
            starting_distance = rollout_starting_distance
        results.append(
            EvaluatedAction(
                face=face,
                contact_point=contact_point,
                angle=angle,
                final_distance=final_distance,
            )
        )
    assert starting_distance is not None, "random_sample_k must be positive"
    return results, starting_distance


def load_iteration_records(iterations_dir: Path) -> list[IterationRecord]:
    json_paths = sorted(iterations_dir.glob("*.json"))
    assert json_paths, f"No iteration JSON files found in {iterations_dir}"

    records: list[IterationRecord] = []
    for json_path in json_paths:
        payload = json.loads(json_path.read_text())
        records.append(
            IterationRecord(
                iteration=int(payload["iteration"]),
                loss=float(payload["loss"]),
                mean_final_distance=float(payload["mean_final_distance"]),
                final_distance_per_env=[float(x) for x in payload["final_distance_per_env"]],
            )
        )
    return records


def save_plot(
    save_path: Path,
    starting_distance: float,
    optimal_final_distance: float,
    iteration_records: list[IterationRecord],
    random_mean_final_distance: float,
    random_std_final_distance: float,
) -> None:
    training_mean_curve = np.array([record.mean_final_distance for record in iteration_records], dtype=np.float32)
    training_final_dists = np.array([record.final_distance_per_env for record in iteration_records], dtype=np.float32)
    best_iter_pos = int(np.argmin(training_mean_curve))
    last_iter_pos = min(best_iter_pos + 10, len(iteration_records) - 1)

    iterations = np.array([record.iteration for record in iteration_records[: last_iter_pos + 1]], dtype=np.int32)
    training_mean_curve = training_mean_curve[: last_iter_pos + 1]
    training_final_dists = training_final_dists[: last_iter_pos + 1]
    n_iters = len(iterations)
    starting_curve = np.full(n_iters, starting_distance, dtype=np.float32)
    optimal_curve = np.full(n_iters, optimal_final_distance, dtype=np.float32)
    random_mean_curve = np.full(n_iters, random_mean_final_distance, dtype=np.float32)
    random_upper = np.full(n_iters, random_mean_final_distance + random_std_final_distance, dtype=np.float32)
    random_lower = np.full(n_iters, random_mean_final_distance - random_std_final_distance, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9, 5))
    for env_idx in range(training_final_dists.shape[1]):
        ax.plot(
            iterations,
            training_final_dists[:, env_idx],
            color="tab:orange",
            linewidth=1.0,
            alpha=0.3,
            label="Training final distance per env" if env_idx == 0 else None,
        )
    ax.plot(iterations, starting_curve, label="Starting distance", color="tab:green", linewidth=2.0, linestyle="--")
    ax.plot(iterations, optimal_curve, label="Optimal final distance", color="black", linewidth=2.0)
    ax.plot(iterations, training_mean_curve, label="Training mean final distance", color="tab:red", linewidth=2.0)
    ax.plot(iterations, random_mean_curve, label="Random mean final distance", color="tab:blue", linewidth=2.0)
    ax.fill_between(iterations, random_lower, random_upper, color="tab:blue", alpha=0.25, label="Random mean ± std")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Final distance [m]")
    ax.set_title("Single-environment action search results")
    ax.set_xlim(iterations[0], iterations[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_optimal_action_video(save_path: Path, seed: int, optimal_action: EvaluatedAction) -> None:
    """Replay the best action once on a video-enabled env and save the rollout."""
    video_env = PushTEnv(nenvs=1, record_video=True, visualize=False)
    video_env.reset(seed=seed)
    action = Action(
        face=np.array([[optimal_action.face]], dtype=np.int32),
        contact_point=np.array([[optimal_action.contact_point]], dtype=np.float32),
        angle=np.array([[optimal_action.angle]], dtype=np.float32),
    )
    video_env.step(action=action, n_sim_steps=N_SIM_STEPS, check_t_displacement=False)
    video_env.save_video(str(save_path))


def main(
    seed: int,
    random_sample_k: int,
    contact_point_bins: int,
    angle_bins: int,
    iterations_dir: Path,
) -> None:
    assert random_sample_k > 0, "random_sample_k must be positive"
    assert contact_point_bins > 0, "contact_point_bins must be positive"
    assert angle_bins > 0, "angle_bins must be positive"
    save_dir = iterations_dir.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    env = PushTEnv(nenvs=1, record_video=False, visualize=False)
    action_grid = build_action_grid(contact_point_bins=contact_point_bins, angle_bins=angle_bins)

    print(f"Evaluating {len(action_grid)} discretized actions for the optimum.")
    optimal_action, starting_distance = evaluate_optimal_action(env=env, seed=seed, action_grid=action_grid)

    print(f"Sampling {random_sample_k} random actions from the discretized grid.")
    random_results, _ = evaluate_random_actions(
        env=env,
        seed=seed,
        action_grid=action_grid,
        random_sample_k=random_sample_k,
    )
    iteration_records = load_iteration_records(iterations_dir)
    random_final_distances = np.array([result.final_distance for result in random_results], dtype=np.float32)
    random_mean_final_distance = float(np.mean(random_final_distances))
    random_std_final_distance = float(np.std(random_final_distances))


    filename = f"results__k:{random_sample_k}__cp-bins:{contact_point_bins}__ang-bins:{angle_bins}__seed:{seed}"
    plot_path = save_dir / f"{filename}.png"
    video_path = save_dir / f"{filename}__optimal.mp4"
    save_plot(
        save_path=plot_path,
        starting_distance=starting_distance,
        optimal_final_distance=optimal_action.final_distance,
        iteration_records=iteration_records,
        random_mean_final_distance=random_mean_final_distance,
        random_std_final_distance=random_std_final_distance,
    )
    save_optimal_action_video(save_path=video_path, seed=seed, optimal_action=optimal_action)
    os.system(f"xdg-open {plot_path}")

    results = {
        "args": {
            "seed": seed,
            "random_sample_k": random_sample_k,
            "contact_point_bins": contact_point_bins,
            "angle_bins": angle_bins,
            "n_sim_steps": N_SIM_STEPS,
            "n_plot_iterations": N_PLOT_ITERS,
            "num_faces": NUM_FACES,
            "iterations_dir": str(iterations_dir),
        },
        "summary": {
            "num_discretized_actions": len(action_grid),
            "num_training_iterations": len(iteration_records),
            "starting_distance": starting_distance,
            "optimal_final_distance": optimal_action.final_distance,
            "random_final_distances": random_final_distances.tolist(),
            "random_mean_final_distance": random_mean_final_distance,
            "random_std_final_distance": random_std_final_distance,
        },
        "optimal_action": asdict(optimal_action),
        "random_actions": [asdict(result) for result in random_results],
        "training_iterations": [asdict(record) for record in iteration_records],
        "artifacts": {
            "plot": str(plot_path),
            "optimal_action_video": str(video_path),
        },
    }
    json_path = save_dir / f"{filename}.json"
    json_path.write_text(json.dumps(results, indent=2))

    print(f"Saved plot to {plot_path}")
    print(f"Saved optimal action video to {video_path}")
    print(f"Saved results to {json_path}")
    print(f"Starting distance: {starting_distance:.6f}")
    print("Optimal action:")
    print(
        f"  face={optimal_action.face}, contact_point={optimal_action.contact_point:.5f}, "
        f"angle={optimal_action.angle:.5f}, final_distance={optimal_action.final_distance:.6f}"
    )
    print(
        f"Random baseline: mean={random_mean_final_distance:.6f}, std={random_std_final_distance:.6f}, "
        f"k={random_sample_k}"
    )


""" Example usage:

python scripts/results_single_env.py --seed 0 --random-sample-k 50 --contact-point-bins 5 --angle-bins 5 --iterations-dir logs/22__11:11:30__n-envs:1__lr:0.05__fixed-t-pose__single-step/iterations
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed for the single environment and random action sampling")
    parser.add_argument("--random-sample-k", type=int, default=50, help="Number of random actions to sample and roll out")
    parser.add_argument(
        "--contact-point-bins",
        type=int,
        default=21,
        help="Number of discretization bins for contact_point when searching the action grid",
    )
    parser.add_argument(
        "--angle-bins",
        type=int,
        default=21,
        help="Number of discretization bins for angle when searching the action grid",
    )
    parser.add_argument(
        "--iterations-dir",
        type=Path,
        required=True,
        help="Directory containing per-iteration JSON files saved by main_surco.py",
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        random_sample_k=args.random_sample_k,
        contact_point_bins=args.contact_point_bins,
        angle_bins=args.angle_bins,
        iterations_dir=args.iterations_dir,
    )
