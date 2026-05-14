from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stride.data import load_atomistic_dataset_npz, load_pdb_trajectory
from stride.goals import GoalSpec
from stride.training import (
    atom_pair_distance_baseline_scores,
    load_atomistic_checkpoint,
    pcoord_baseline_rankers,
    score_atomistic_dataset,
    split_atomistic_indices,
    truncate_atomistic_history,
)
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
        "--history-frames",
        type=int,
        default=None,
        help=(
            "Use only the most recent N frames from each window before scoring. "
            "Use 1 to evaluate a last-frame-only ablation checkpoint."
        ),
    )
    parser.add_argument(
        "--eval-split",
        choices=("all", "train", "validation"),
        default="all",
        help="Evaluate all examples or the train/validation subset from a split.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=(
            "contiguous",
            "random",
            "blocked",
            "blocked_tail",
            "iteration_tail",
            "iteration_random",
            "iteration_balanced",
        ),
        default="contiguous",
        help="Split strategy used when --eval-split is train or validation.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Validation fraction used when --eval-split is train or validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Split seed used when --eval-split is train or validation.",
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
    parser.add_argument(
        "--atom-pair-indices",
        "--distance-indices",
        dest="atom_pair_indices",
        default=None,
        help="Comma-separated zero-based atom indices for distance baselines, e.g. 0,1.",
    )
    parser.add_argument(
        "--distance-direction",
        choices=("low", "high", "proximity"),
        default="low",
        help=(
            "How atom-pair distances become rank scores: low for association, "
            "high for dissociation, proximity for closeness to --distance-target."
        ),
    )
    parser.add_argument(
        "--distance-target",
        type=float,
        default=None,
        help="Target distance for proximity baselines. Defaults to goal threshold when possible.",
    )
    parser.add_argument(
        "--pcoord-target",
        type=float,
        default=None,
        help="Target pcoord value for WESTPA pcoord proximity baselines.",
    )
    parser.add_argument(
        "--pcoord-dim",
        type=int,
        default=0,
        help="Progress-coordinate dimension for WESTPA pcoord baselines.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    if args.checkpoint is None and args.scores_npz is None:
        raise ValueError("Provide --checkpoint or --scores-npz.")

    dataset = load_atomistic_dataset_npz(args.dataset_npz)
    dataset = truncate_atomistic_history(dataset, args.history_frames)
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
    rankers.update(_build_distance_baselines(dataset, args))
    rankers.update(_build_westpa_pcoord_baselines(args))
    rankers["random"] = random_baseline_scores(len(dataset.event_labels), seed=args.random_seed)
    eval_indices = _evaluation_indices(dataset, args)
    eval_labels = dataset.event_labels[eval_indices]
    eval_rankers = {
        name: np.asarray(scores)[eval_indices]
        for name, scores in rankers.items()
    }
    dataset_name = str(args.dataset_npz)
    if args.eval_split != "all":
        dataset_name = (
            f"{dataset_name} [{args.eval_split} split: "
            f"{args.split_strategy}, validation_fraction={args.validation_fraction}, "
            f"seed={args.seed}]"
        )

    paths = write_evaluation_report(
        output_dir=args.output_dir,
        y_true=eval_labels,
        rankers=eval_rankers,
        dataset_name=dataset_name,
        checkpoint_name=str(args.checkpoint or args.scores_npz or ""),
    )

    print(f"Dataset: {args.dataset_npz}")
    print(
        "Evaluation examples: "
        f"{len(eval_indices)} ({args.eval_split}, positive_rate={eval_labels.mean():.6g})"
    )
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
    if args.dihedral_indices is None and (goal is None or goal.type != "dihedral_window"):
        return {}
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


def _build_distance_baselines(
    dataset,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    atom_indices = _parse_atom_pair_indices(args.atom_pair_indices)
    if atom_indices is None:
        return {}

    goal = GoalSpec.from_yaml(args.goal_yaml) if args.goal_yaml is not None else None
    target = args.distance_target
    if target is None and args.distance_direction == "proximity" and goal is not None:
        target = goal.threshold

    direction = args.distance_direction
    window_mode = "window_max" if direction == "high" else "window_min"
    return {
        f"last_frame_atom_pair_distance_{direction}": atom_pair_distance_baseline_scores(
            dataset,
            atom_indices=atom_indices,
            mode="last_frame",
            direction=direction,
            target=target,
        ),
        f"{window_mode}_atom_pair_distance_{direction}": atom_pair_distance_baseline_scores(
            dataset,
            atom_indices=atom_indices,
            mode=window_mode,
            direction=direction,
            target=target,
        ),
    }


def _build_westpa_pcoord_baselines(args: argparse.Namespace) -> dict[str, np.ndarray]:
    data = np.load(args.dataset_npz)
    pcoord_key = _first_present_key(data, ("westpa_pcoord_windows", "pcoord_windows"))
    mask_key = _first_present_key(data, ("westpa_pcoord_window_mask", "window_mask"))
    if pcoord_key is None or mask_key is None:
        return {}

    goal = GoalSpec.from_yaml(args.goal_yaml) if args.goal_yaml is not None else None
    target = args.pcoord_target
    if target is None and goal is not None and goal.type == "distance_threshold":
        target = goal.threshold

    return pcoord_baseline_rankers(
        data[pcoord_key],
        data[mask_key],
        target=target,
        pcoord_dim=args.pcoord_dim,
    )


def _first_present_key(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> str | None:
    available = set(data.files)
    for key in keys:
        if key in available:
            return key
    return None


def _evaluation_indices(dataset, args: argparse.Namespace) -> np.ndarray:
    if args.eval_split == "all":
        return np.arange(len(dataset.event_labels), dtype=np.int64)

    train_indices, val_indices = split_atomistic_indices(
        dataset=dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        split_strategy=args.split_strategy,
    )
    if args.eval_split == "train":
        return train_indices.astype(np.int64)
    if args.eval_split == "validation":
        return val_indices.astype(np.int64)
    raise ValueError(f"Unknown eval split: {args.eval_split}")


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


def _parse_atom_pair_indices(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(indices) != 2:
        raise ValueError("--atom-pair-indices must contain exactly two indices.")
    return indices  # type: ignore[return-value]


if __name__ == "__main__":
    main()
