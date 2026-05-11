from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stride.data import load_atomistic_dataset_npz, load_pdb_trajectory
from stride.goals import GoalSpec
from stride.training import load_atomistic_checkpoint, score_atomistic_dataset
from stride.training.evaluation import (
    dihedral_window_baseline_scores,
    random_baseline_scores,
    resolve_dihedral_indices,
    write_evaluation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate STRIDE checkpoint evaluation reports and simple baselines."
    )
    parser.add_argument("dataset_npz", type=Path, help="Atomistic STRIDE .npz dataset.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained PyTorch checkpoint to score. Required unless --scores-npz is given.",
    )
    parser.add_argument(
        "--scores-npz",
        type=Path,
        default=None,
        help="Existing score artifact from scripts/score_atomistic.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/atomistic_eval"),
        help="Directory for report tables and plots.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--rank-key",
        default="p_event",
        help="Checkpoint score key to treat as the primary STRIDE ranker.",
    )
    parser.add_argument(
        "--goal-yaml",
        type=Path,
        default=None,
        help="Goal YAML for dihedral baseline bounds and topology selections.",
    )
    parser.add_argument(
        "--topology-pdb",
        type=Path,
        default=None,
        help="PDB topology used to resolve goal selections for dihedral baselines.",
    )
    parser.add_argument(
        "--dihedral-indices",
        default=None,
        help="Comma-separated zero-based atom indices for phi baseline, e.g. 4,6,8,14.",
    )
    parser.add_argument(
        "--dihedral-bounds",
        default=None,
        help="Comma-separated lower,upper degrees if --goal-yaml is not provided.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    if args.checkpoint is None and args.scores_npz is None:
        raise ValueError("Provide --checkpoint or --scores-npz.")

    dataset = load_atomistic_dataset_npz(args.dataset_npz)
    rankers: dict[str, np.ndarray] = {}

    if args.scores_npz is not None:
        score_data = np.load(args.scores_npz)
        if args.rank_key not in score_data:
            raise ValueError(f"Score artifact does not contain key: {args.rank_key}")
        rankers["STRIDE"] = score_data[args.rank_key]

    checkpoint_metrics = {}
    if args.checkpoint is not None:
        model, checkpoint_metrics = load_atomistic_checkpoint(
            args.checkpoint,
            device=args.device,
        )
        scores = score_atomistic_dataset(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=args.device,
        )
        if args.rank_key not in scores:
            raise ValueError(f"Checkpoint scoring did not produce key: {args.rank_key}")
        rankers["STRIDE"] = scores[args.rank_key]

    baseline_info = _build_dihedral_baselines(dataset, args)
    rankers.update(baseline_info)
    rankers["random"] = random_baseline_scores(len(dataset.event_labels), seed=args.random_seed)

    paths = write_evaluation_report(
        output_dir=args.output_dir,
        y_true=dataset.event_labels,
        rankers=rankers,
        dataset_name=str(args.dataset_npz),
        checkpoint_name=str(args.checkpoint or args.scores_npz or ""),
    )

    print(f"Dataset: {args.dataset_npz}")
    if args.checkpoint is not None:
        print(f"Checkpoint: {args.checkpoint}")
    if checkpoint_metrics:
        print(f"Checkpoint train_loss: {checkpoint_metrics.get('train_loss', float('nan')):.6g}")
    print(f"Report: {paths['markdown']}")
    print(f"Metrics: {paths['metrics']}")
    print(f"Quantile precision/recall: {paths['quantiles']}")
    print(f"Score summary: {paths['summary']}")


def _build_dihedral_baselines(
    dataset,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    goal = GoalSpec.from_yaml(args.goal_yaml) if args.goal_yaml is not None else None
    atom_indices = _parse_dihedral_indices(args.dihedral_indices)
    if atom_indices is None and goal is not None and args.topology_pdb is not None:
        _, atoms = load_pdb_trajectory(args.topology_pdb)
        atom_indices = resolve_dihedral_indices(atoms, goal)

    bounds = _parse_dihedral_bounds(args.dihedral_bounds)
    if bounds is None and goal is not None:
        if goal.lower_bound is None or goal.upper_bound is None:
            raise ValueError("Goal YAML does not define dihedral bounds.")
        bounds = (goal.lower_bound, goal.upper_bound)

    if atom_indices is None or bounds is None:
        return {}

    lower, upper = bounds
    return {
        "phi_window_min_distance": dihedral_window_baseline_scores(
            dataset,
            atom_indices=atom_indices,
            lower_bound=lower,
            upper_bound=upper,
            mode="window_min",
        ),
        "last_frame_phi_proximity": dihedral_window_baseline_scores(
            dataset,
            atom_indices=atom_indices,
            lower_bound=lower,
            upper_bound=upper,
            mode="last_frame",
        ),
    }


def _parse_dihedral_indices(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(indices) != 4:
        raise ValueError("--dihedral-indices must contain exactly four indices.")
    return indices  # type: ignore[return-value]


def _parse_dihedral_bounds(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != 2:
        raise ValueError("--dihedral-bounds must contain lower,upper.")
    return parts


if __name__ == "__main__":
    main()
