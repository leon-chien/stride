from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from stride.westpa_plugin import decide_promotion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an online STRIDE challenger and write a promotion decision."
    )
    parser.add_argument("lineage_npz", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--champion-metrics-csv", type=Path, default=None)
    parser.add_argument("--mode", default="tail")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--baseline-key", default="last_pcoord_low")
    args = parser.parse_args()

    checkpoint = args.output_prefix.with_suffix(".pt")
    best_checkpoint = _default_best_checkpoint(checkpoint)
    scores_npz = args.output_prefix.with_name(args.output_prefix.name + "_scores.npz")
    replay_dir = Path("outputs/reports") / f"{args.output_prefix.name}_online_replay"
    decision_path = args.output_prefix.with_name(args.output_prefix.name + "_promotion.json")

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
            "--validation-fraction",
            str(args.validation_fraction),
            "--split-strategy",
            args.mode,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--event-positive-weight",
            "auto",
            "--save-best-metric",
            "val_auprc",
            "--save-best-mode",
            "max",
            "--early-stopping-patience",
            "5",
            "--feature-mode",
            "engineered",
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
            "scripts/replay_westpa_steering.py",
            str(args.lineage_npz),
            "--stride-scores-npz",
            str(scores_npz),
            "--rank-key",
            "p_event",
            "--baseline-key",
            args.baseline_key,
            "--eval-split",
            "validation",
            "--iteration-split-strategy",
            args.mode,
            "--validation-fraction",
            str(args.validation_fraction),
            "--seed",
            str(args.seed),
            "--bin-reference",
            "train",
            "--per-iteration",
            "--checkpoint",
            str(best_checkpoint),
            "--stride-fusion-alpha",
            "auto",
            "--output-dir",
            str(replay_dir),
        ]
    )

    rows = _read_metrics(replay_dir / "steering_metrics.csv")
    challenger = rows["STRIDE"]
    baseline = rows[args.baseline_key]
    champion = None
    if args.champion_metrics_csv is not None:
        champion_rows = _read_metrics(args.champion_metrics_csv)
        champion = champion_rows.get("STRIDE")
    decision = decide_promotion(challenger, baseline, champion=champion)
    decision.write_json(decision_path)

    print(f"Challenger checkpoint: {best_checkpoint}")
    print(f"Replay report: {replay_dir / 'report.md'}")
    print(f"Promotion decision: {decision_path}")
    print(f"promote: {decision.promote}")
    print(f"reason: {decision.reason}")


def _read_metrics(path: Path) -> dict[str, dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: dict[str, dict[str, float]] = {}
    for row in rows:
        ranker = row["ranker"]
        parsed[ranker] = {
            key: float(value)
            for key, value in row.items()
            if key != "ranker" and value not in {"", "nan"}
        }
    return parsed


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
