"""Evaluate a single PushT problem with exhaustive and random multi-step search."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm  # pyright: ignore[reportMissingModuleSource]

# Disable JIT for this exhaustive search script to avoid XLA compile-time memory blowups.
# jax.config.update("jax_disable_jit", True)

from pusht619.core import ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES, Action, PushTEnv


N_SIM_STEPS = 25


@dataclass(frozen=True)
class ActionSpec:
    face: int
    contact_point: float
    angle: float


@dataclass(frozen=True)
class EvaluatedSequence:
    actions: list[ActionSpec]
    final_distance: float


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    loss: float
    mean_final_distance: float
    final_distance_per_env: list[float]


def _evaluated_sequence_from_payload(payload: dict) -> EvaluatedSequence:
    return EvaluatedSequence(
        actions=[
            ActionSpec(
                face=int(action_payload["face"]),
                contact_point=float(action_payload["contact_point"]),
                angle=float(action_payload["angle"]),
            )
            for action_payload in payload["actions"]
        ],
        final_distance=float(payload["final_distance"]),
    )


def build_action_grid(contact_point_bins: int, angle_bins: int) -> list[ActionSpec]:
    contact_points = np.linspace(CONTACT_POINT_BOUNDS[0], CONTACT_POINT_BOUNDS[1], num=contact_point_bins, dtype=np.float32)
    angles = np.linspace(float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1]), num=angle_bins, dtype=np.float32)

    grid: list[ActionSpec] = []
    for face in range(NUM_FACES):
        for contact_point in contact_points:
            for angle in angles:
                grid.append(
                    ActionSpec(
                        face=face,
                        contact_point=float(contact_point),
                        angle=float(angle),
                    )
                )
    return grid


def rollout_action_sequence(env: PushTEnv, seed: int, action_sequence: tuple[ActionSpec, ...]) -> tuple[float, float]:
    # Recreate the exact same single-env problem for every candidate action sequence.
    env.reset(seed=seed)
    starting_distance: float | None = None
    final_distance: float | None = None
    for action_spec in action_sequence:
        action = Action(
            face=np.array([[action_spec.face]], dtype=np.int32),
            contact_point=np.array([[action_spec.contact_point]], dtype=np.float32),
            angle=np.array([[action_spec.angle]], dtype=np.float32),
        )
        result = env.step(action=action, n_sim_steps=N_SIM_STEPS, check_t_displacement=False)
        t_distances = np.asarray(result.t_distances)[0]
        if starting_distance is None:
            starting_distance = float(t_distances[0])
        final_distance = float(t_distances[-1])
    assert starting_distance is not None, "Action sequence must not be empty."
    assert final_distance is not None, "Action sequence must not be empty."
    return starting_distance, final_distance


def evaluate_optimal_sequence(
    env: PushTEnv,
    seed: int,
    action_grid: list[ActionSpec],
    n_action_steps: int,
) -> tuple[EvaluatedSequence, float]:
    best_sequence: EvaluatedSequence | None = None
    starting_distance: float | None = None
    total_sequences = len(action_grid) ** n_action_steps
    sequence_iter = product(action_grid, repeat=n_action_steps)
    for action_sequence in tqdm(sequence_iter, total=total_sequences, desc="Optimal sequences", unit="sequence"):
        rollout_starting_distance, final_distance = rollout_action_sequence(env, seed, action_sequence)
        if starting_distance is None:
            starting_distance = rollout_starting_distance
        candidate = EvaluatedSequence(actions=list(action_sequence), final_distance=final_distance)
        if best_sequence is None or candidate.final_distance < best_sequence.final_distance:
            best_sequence = candidate
    assert best_sequence is not None, "Action grid must not be empty."
    assert starting_distance is not None, "Action grid must not be empty."
    return best_sequence, starting_distance


def evaluate_random_sequences(
    env: PushTEnv,
    seed: int,
    action_grid: list[ActionSpec],
    random_sample_k: int,
    n_action_steps: int,
) -> tuple[list[EvaluatedSequence], float]:
    rng = np.random.default_rng(seed=seed)
    sampled_indices = rng.integers(0, len(action_grid), size=(random_sample_k, n_action_steps))

    results: list[EvaluatedSequence] = []
    starting_distance: float | None = None
    for sampled_sequence in tqdm(sampled_indices.tolist(), desc="Random sequences", unit="sequence"):
        action_sequence = tuple(action_grid[idx] for idx in sampled_sequence)
        rollout_starting_distance, final_distance = rollout_action_sequence(env, seed, action_sequence)
        if starting_distance is None:
            starting_distance = rollout_starting_distance
        results.append(EvaluatedSequence(actions=list(action_sequence), final_distance=final_distance))
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


def load_existing_results(existing_results_path: Path) -> dict:
    return json.loads(existing_results_path.read_text())


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
    training_std_curve = np.std(training_final_dists, axis=1)
    finite_mean_mask = np.isfinite(training_mean_curve)
    assert np.any(finite_mean_mask), "No finite mean_final_distance values found in iteration records."
    finite_mean_curve = np.where(finite_mean_mask, training_mean_curve, np.inf)
    best_iter_pos = int(np.argmin(finite_mean_curve))
    last_iter_pos = min(best_iter_pos + 10, len(iteration_records) - 1)

    iterations = np.array([record.iteration for record in iteration_records[: last_iter_pos + 1]], dtype=np.int32)
    training_mean_curve = training_mean_curve[: last_iter_pos + 1]
    training_final_dists = training_final_dists[: last_iter_pos + 1]
    training_std_curve = training_std_curve[: last_iter_pos + 1]
    n_iters = len(iterations)
    starting_curve = np.full(n_iters, starting_distance, dtype=np.float32)
    optimal_curve = np.full(n_iters, optimal_final_distance, dtype=np.float32)
    random_mean_curve = np.full(n_iters, random_mean_final_distance, dtype=np.float32)
    random_upper = np.full(n_iters, random_mean_final_distance + random_std_final_distance, dtype=np.float32)
    random_lower = np.full(n_iters, random_mean_final_distance - random_std_final_distance, dtype=np.float32)
    # training_upper = training_mean_curve + training_std_curve
    # training_lower = training_mean_curve - training_std_curve

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
    ax.plot(iterations, optimal_curve, label="Optimal sequence final distance", color="black", linewidth=2.0)
    ax.plot(iterations, training_mean_curve, label="Training mean final distance", color="tab:red", linewidth=2.0)
    # ax.fill_between(
    #     iterations,
    #     training_lower,
    #     training_upper,
    #     color="tab:red",
    #     alpha=0.15,
    #     label="Training mean ± std",
    # )
    ax.plot(iterations, random_mean_curve, label="Random sequence mean final distance", color="tab:blue", linewidth=2.0)
    ax.fill_between(iterations, random_lower, random_upper, color="tab:blue", alpha=0.25)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Final distance [m]")
    ax.set_title("Single-environment multi-step action search results")
    ax.set_xlim(iterations[0], iterations[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_optimal_action_video(save_path: Path, seed: int, optimal_sequence: EvaluatedSequence) -> None:
    """Replay the best action sequence once on a video-enabled env and save the rollout."""
    video_env = PushTEnv(nenvs=1, record_video=True, visualize=False)
    video_env.reset(seed=seed)
    for action_spec in optimal_sequence.actions:
        action = Action(
            face=np.array([[action_spec.face]], dtype=np.int32),
            contact_point=np.array([[action_spec.contact_point]], dtype=np.float32),
            angle=np.array([[action_spec.angle]], dtype=np.float32),
        )
        video_env.step(action=action, n_sim_steps=N_SIM_STEPS, check_t_displacement=False)
    video_env.save_video(str(save_path))


def main(
    seed: int | None,
    random_sample_k: int | None,
    contact_point_bins: int | None,
    angle_bins: int | None,
    n_action_steps: int | None,
    iterations_dir: Path,
    existing_results: Path | None,
) -> None:
    existing_results_payload: dict | None = None
    if existing_results is not None:
        existing_results_payload = load_existing_results(existing_results)
        existing_args = existing_results_payload["args"]
        if seed is None:
            seed = int(existing_args["seed"])
        if random_sample_k is None:
            random_sample_k = int(existing_args["random_sample_k"])
        if contact_point_bins is None:
            contact_point_bins = int(existing_args["contact_point_bins"])
        if angle_bins is None:
            angle_bins = int(existing_args["angle_bins"])
        if n_action_steps is None:
            n_action_steps = int(existing_args["n_action_steps"])

    assert seed is not None, "seed must be provided unless --existing-results is used"
    assert random_sample_k is not None and random_sample_k > 0, "random_sample_k must be positive"
    assert contact_point_bins is not None and contact_point_bins > 0, "contact_point_bins must be positive"
    assert angle_bins is not None and angle_bins > 0, "angle_bins must be positive"
    assert n_action_steps is not None and n_action_steps > 0, "n_action_steps must be positive"
    save_dir = iterations_dir.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    action_grid = build_action_grid(contact_point_bins=contact_point_bins, angle_bins=angle_bins)
    iteration_records = load_iteration_records(iterations_dir)

    if existing_results_payload is None:
        env = PushTEnv(nenvs=1, record_video=False, visualize=False)
        print(
            f"Evaluating {len(action_grid) ** n_action_steps} discretized action sequences "
            f"({len(action_grid)}^{n_action_steps}) for the optimum."
        )
        optimal_sequence, starting_distance = evaluate_optimal_sequence(
            env=env,
            seed=seed,
            action_grid=action_grid,
            n_action_steps=n_action_steps,
        )

        print(f"Sampling {random_sample_k} random action sequences from the discretized grid.")
        random_sequences, _ = evaluate_random_sequences(
            env=env,
            seed=seed,
            action_grid=action_grid,
            random_sample_k=random_sample_k,
            n_action_steps=n_action_steps,
        )
        random_final_distances = np.array([result.final_distance for result in random_sequences], dtype=np.float32)
        random_mean_final_distance = float(np.mean(random_final_distances))
        random_std_final_distance = float(np.std(random_final_distances))
    else:
        summary_payload = existing_results_payload["summary"]
        optimal_sequence = _evaluated_sequence_from_payload(existing_results_payload["optimal_sequence"])
        random_sequences = [
            _evaluated_sequence_from_payload(sequence_payload)
            for sequence_payload in existing_results_payload.get("random_sequences", [])
        ]
        starting_distance = float(summary_payload["starting_distance"])
        random_mean_final_distance = float(summary_payload["random_mean_final_distance"])
        random_std_final_distance = float(summary_payload["random_std_final_distance"])
        random_final_distances = np.array(summary_payload.get("random_final_distances", []), dtype=np.float32)
        print(f"Loaded optimal/random sequence statistics from {existing_results}.")

    filename = (
        f"results_multistep__k:{random_sample_k}__cp-bins:{contact_point_bins}"
        f"__ang-bins:{angle_bins}__n-steps:{n_action_steps}__seed:{seed}"
    )
    if existing_results_payload is None:
        plot_path = save_dir / f"{filename}.png"
        video_path = save_dir / f"{filename}__optimal.mp4"
    else:
        plot_path = save_dir / f"{filename}__replot.png"
        video_path = save_dir / f"{filename}__optimal.mp4"
    save_plot(
        save_path=plot_path,
        starting_distance=starting_distance,
        optimal_final_distance=optimal_sequence.final_distance,
        iteration_records=iteration_records,
        random_mean_final_distance=random_mean_final_distance,
        random_std_final_distance=random_std_final_distance,
    )
    os.system(f"xdg-open {plot_path}")

    if existing_results_payload is None:
        save_optimal_action_video(save_path=video_path, seed=seed, optimal_sequence=optimal_sequence)
        results = {
            "args": {
                "seed": seed,
                "random_sample_k": random_sample_k,
                "contact_point_bins": contact_point_bins,
                "angle_bins": angle_bins,
                "n_action_steps": n_action_steps,
                "n_sim_steps": N_SIM_STEPS,
                "num_faces": NUM_FACES,
                "iterations_dir": str(iterations_dir),
            },
            "summary": {
                "num_single_actions": len(action_grid),
                "num_discretized_action_sequences": len(action_grid) ** n_action_steps,
                "num_training_iterations": len(iteration_records),
                "starting_distance": starting_distance,
                "optimal_final_distance": optimal_sequence.final_distance,
                "random_final_distances": random_final_distances.tolist(),
                "random_mean_final_distance": random_mean_final_distance,
                "random_std_final_distance": random_std_final_distance,
            },
            "optimal_sequence": asdict(optimal_sequence),
            "random_sequences": [asdict(result) for result in random_sequences],
            "training_iterations": [asdict(record) for record in iteration_records],
            "artifacts": {
                "plot": str(plot_path),
                "optimal_action_video": str(video_path),
            },
        }

    print(f"Saved plot to {plot_path}")
    if existing_results_payload is None:
        json_path = save_dir / f"{filename}.json"
        json_path.write_text(json.dumps(results, indent=2))
        print(f"Saved optimal action video to {video_path}")
        print(f"Saved results to {json_path}")
    else:
        print(f"Kept existing results JSON unchanged: {existing_results}")
    print(f"Starting distance: {starting_distance:.6f}")
    print("Optimal action sequence:")
    for action_idx, action_spec in enumerate(optimal_sequence.actions, start=1):
        print(
            f"  step {action_idx}: face={action_spec.face}, contact_point={action_spec.contact_point:.5f}, "
            f"angle={action_spec.angle:.5f}"
        )
    print(f"  final_distance={optimal_sequence.final_distance:.6f}")
    print(
        f"Random baseline: mean={random_mean_final_distance:.6f}, std={random_std_final_distance:.6f}, "
        f"k={random_sample_k}"
    )


"""Example usage:


python scripts/results_single_env_multistep.py \
    --seed 0 --random-sample-k 20 --contact-point-bins 2 --angle-bins 3 --n-action-steps 2 \
    --iterations-dir logs/22__12:38:48__n-envs:1__lr:0.01__fixed-t-pose__multi-step/iterations

python scripts/results_single_env_multistep.py \
    --iterations-dir logs/22__12:38:48__n-envs:1__lr:0.01__fixed-t-pose__multi-step/iterations \
    --existing-results logs/22__12:38:48__n-envs:1__lr:0.01__fixed-t-pose__multi-step/results_multistep__k:20__cp-bins:2__ang-bins:3__n-steps:2__seed:0.json
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed for the single environment and random action sampling")
    parser.add_argument("--random-sample-k", type=int, help="Number of random action sequences to sample")
    parser.add_argument(
        "--contact-point-bins",
        type=int,
        help="Number of discretization bins for contact_point when searching the action grid",
    )
    parser.add_argument(
        "--angle-bins",
        type=int,
        help="Number of discretization bins for angle when searching the action grid",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        help="Number of consecutive actions in each evaluated sequence",
    )
    parser.add_argument(
        "--iterations-dir",
        type=Path,
        required=True,
        help="Directory containing per-iteration JSON files saved by main_surco.py",
    )
    parser.add_argument(
        "--existing-results",
        type=Path,
        help="Existing results JSON to reuse for optimal/random statistics while reloading training iterations from --iterations-dir",
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        random_sample_k=args.random_sample_k,
        contact_point_bins=args.contact_point_bins,
        angle_bins=args.angle_bins,
        n_action_steps=args.n_action_steps,
        iterations_dir=args.iterations_dir,
        existing_results=args.existing_results,
    )
