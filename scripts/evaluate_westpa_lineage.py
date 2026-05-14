from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stride.training.westpa_evaluation import write_westpa_lineage_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate WESTPA pcoord lineage artifacts against simple baselines."
    )
    parser.add_argument("lineage_npz", type=Path, help="Output from extract_westpa_dataset.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/westpa_lineage"))
    parser.add_argument("--stride-scores-npz", type=Path, default=None)
    parser.add_argument("--rank-key", default="p_event")
    parser.add_argument(
        "--eval-split",
        choices=("all", "train", "validation"),
        default="all",
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
    parser.add_argument("--pcoord-target", type=float, default=None)
    parser.add_argument("--pcoord-dim", type=int, default=0)
    args = parser.parse_args()

    stride_scores = None
    if args.stride_scores_npz is not None:
        score_data = np.load(args.stride_scores_npz)
        if args.rank_key not in score_data:
            raise ValueError(f"Score artifact has no key: {args.rank_key}")
        stride_scores = score_data[args.rank_key]

    paths = write_westpa_lineage_report(
        lineage_npz=args.lineage_npz,
        output_dir=args.output_dir,
        stride_scores=stride_scores,
        eval_split=args.eval_split,
        validation_fraction=args.validation_fraction,
        split_strategy=args.iteration_split_strategy,
        seed=args.seed,
        pcoord_target=args.pcoord_target,
        pcoord_dim=args.pcoord_dim,
    )

    print(f"Dataset: {args.lineage_npz}")
    print(f"Report: {paths['markdown']}")
    print(f"Metrics: {paths['metrics']}")
    print(f"Quantile precision/recall: {paths['quantiles']}")
    print(f"Score summary: {paths['summary']}")
    if "grouped_metrics" in paths:
        print(f"Grouped metrics: {paths['grouped_metrics']}")
    if "grouped_markdown" in paths:
        print(f"Grouped report: {paths['grouped_markdown']}")


if __name__ == "__main__":
    main()
