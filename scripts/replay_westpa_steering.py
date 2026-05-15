from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stride.westpa_plugin import ReplayConfig, replay_westpa_steering


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay STRIDE steering choices on saved WESTPA lineage artifacts."
    )
    parser.add_argument("lineage_npz", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/westpa_steering_replay"))
    parser.add_argument("--stride-scores-npz", type=Path, default=None)
    parser.add_argument("--rank-key", default="p_event")
    parser.add_argument("--baseline-key", default="last_pcoord_low")
    parser.add_argument(
        "--eval-split",
        choices=("all", "train", "validation"),
        default="validation",
    )
    parser.add_argument(
        "--iteration-split-strategy",
        choices=(
            "tail",
            "random_block",
            "heldout_goal",
            "heldout_cell",
            "random_goal",
            "random_cell",
        ),
        default="tail",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--binning", choices=("quantile", "fixed"), default="quantile")
    parser.add_argument("--pcoord-dim", type=int, default=0)
    args = parser.parse_args()

    stride_scores = None
    if args.stride_scores_npz is not None:
        score_data = np.load(args.stride_scores_npz)
        if args.rank_key not in score_data:
            raise ValueError(f"Score artifact has no key: {args.rank_key}")
        stride_scores = score_data[args.rank_key]

    paths = replay_westpa_steering(
        lineage_npz=args.lineage_npz,
        output_dir=args.output_dir,
        stride_scores=stride_scores,
        config=ReplayConfig(
            eval_split=args.eval_split,
            split_strategy=args.iteration_split_strategy,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            num_bins=args.num_bins,
            binning=args.binning,
            pcoord_dim=args.pcoord_dim,
            baseline_key=args.baseline_key,
        ),
    )

    print(f"Dataset: {args.lineage_npz}")
    if args.stride_scores_npz is not None:
        print(f"STRIDE scores: {args.stride_scores_npz} [{args.rank_key}]")
    print(f"Report: {paths['markdown']}")
    print(f"Metrics: {paths['metrics']}")
    print(f"Bin occupancy: {paths['bins']}")
    print(f"Grouped metrics: {paths['grouped_metrics']}")
    print(f"WESTPA arrays: {paths['assignments']}")


if __name__ == "__main__":
    main()
