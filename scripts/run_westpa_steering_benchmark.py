from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run train -> score -> evaluate -> replay for WESTPA steering benchmarks."
    )
    parser.add_argument("lineage_npz", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument(
        "--mode",
        choices=("tail", "heldout_cell", "heldout_goal"),
        default="tail",
        help="Generalization split to benchmark.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--event-positive-weight", default="auto")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rank-key", default="p_event")
    parser.add_argument("--baseline-key", default="last_pcoord_low")
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--binning", choices=("quantile", "fixed"), default="quantile")
    parser.add_argument("--bin-reference", choices=("train", "eval"), default="train")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    args = parser.parse_args()

    split_strategy = args.mode
    checkpoint = args.output_prefix.with_suffix(".pt")
    best_checkpoint = _default_best_checkpoint(checkpoint)
    scores_npz = args.output_prefix.with_name(args.output_prefix.name + "_scores.npz")
    eval_report_dir = Path("outputs/reports") / f"{args.output_prefix.name}_evaluation"
    replay_report_dir = Path("outputs/reports") / f"{args.output_prefix.name}_steering_replay"

    _run(
        [
            sys.executable,
            "scripts/train_westpa_lineage.py",
            str(args.lineage_npz),
            str(checkpoint),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--hidden-dim",
            str(args.hidden_dim),
            "--transformer-layers",
            str(args.transformer_layers),
            "--transformer-heads",
            str(args.transformer_heads),
            "--device",
            args.device,
            "--event-positive-weight",
            str(args.event_positive_weight),
            "--split-strategy",
            split_strategy,
            "--validation-fraction",
            str(args.validation_fraction),
            "--seed",
            str(args.seed),
            "--save-best-metric",
            "val_auprc",
            "--save-best-mode",
            "max",
            "--early-stopping-patience",
            str(args.early_stopping_patience),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/score_westpa_lineage.py",
            str(args.lineage_npz),
            str(best_checkpoint),
            str(scores_npz),
            "--device",
            args.device,
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/evaluate_westpa_lineage.py",
            str(args.lineage_npz),
            "--stride-scores-npz",
            str(scores_npz),
            "--rank-key",
            args.rank_key,
            "--eval-split",
            "validation",
            "--iteration-split-strategy",
            split_strategy,
            "--validation-fraction",
            str(args.validation_fraction),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(eval_report_dir),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/replay_westpa_steering.py",
            str(args.lineage_npz),
            "--stride-scores-npz",
            str(scores_npz),
            "--rank-key",
            args.rank_key,
            "--baseline-key",
            args.baseline_key,
            "--eval-split",
            "validation",
            "--iteration-split-strategy",
            split_strategy,
            "--validation-fraction",
            str(args.validation_fraction),
            "--seed",
            str(args.seed),
            "--num-bins",
            str(args.num_bins),
            "--binning",
            args.binning,
            "--bin-reference",
            args.bin_reference,
            "--per-iteration",
            "--checkpoint",
            str(best_checkpoint),
            "--output-dir",
            str(replay_report_dir),
        ]
    )

    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Scores: {scores_npz}")
    print(f"Evaluation report: {eval_report_dir / 'report.md'}")
    print(f"Replay report: {replay_report_dir / 'report.md'}")


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _default_best_checkpoint(path: Path) -> Path:
    suffix = path.suffix
    if suffix:
        return path.with_name(path.name[: -len(suffix)] + ".best" + suffix)
    return path.with_name(path.name + ".best.pt")


if __name__ == "__main__":
    main()
