from __future__ import annotations

import argparse
from pathlib import Path

from stride.training import (
    describe_atomistic_split,
    load_dataset_and_make_config,
    train_atomistic_value_model,
    truncate_atomistic_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train STRIDE's atomistic goal-conditioned value model."
    )
    parser.add_argument("dataset_npz", type=Path, help="Atomistic STRIDE .npz dataset.")
    parser.add_argument("checkpoint", type=Path, help="Output PyTorch checkpoint.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
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
        help=(
            "Train/validation split. Use iteration_balanced for small WESTPA "
            "lineage datasets where purged tail splits can remove all positives."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--egnn-layers", type=int, default=3)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help=(
            "Use only the most recent N frames from each window. Use 1 for a "
            "last-frame-only temporal ablation."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--event-positive-weight",
        default="auto",
        help="Positive event class weight. Use 'auto' for negative/positive ratio.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final metrics, not per-epoch progress.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume model and optimizer state from an existing checkpoint.",
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=None,
        help="Path for the best checkpoint. Defaults to '<checkpoint>.best.pt'.",
    )
    parser.add_argument(
        "--no-save-best",
        action="store_true",
        help="Disable saving a separate best checkpoint during training.",
    )
    parser.add_argument(
        "--save-best-metric",
        default="val_auroc",
        help="Metric used to choose the best checkpoint.",
    )
    parser.add_argument(
        "--save-best-mode",
        choices=("max", "min"),
        default="max",
        help="Whether higher or lower save-best metric values are better.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without improving the save-best metric.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum save-best metric improvement required to reset patience.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine", "plateau"),
        default="none",
        help="Optional learning-rate scheduler.",
    )
    args = parser.parse_args()

    dataset, config = load_dataset_and_make_config(
        dataset_path=args.dataset_npz,
        hidden_dim=args.hidden_dim,
        egnn_layers=args.egnn_layers,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        radius=args.radius,
    )
    dataset = truncate_atomistic_history(dataset, args.history_frames)
    best_checkpoint = None
    if not args.no_save_best:
        best_checkpoint = args.best_checkpoint or _default_best_checkpoint(args.checkpoint)

    split_stats = describe_atomistic_split(
        dataset=dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        split_strategy=args.split_strategy,
    )
    print(f"Dataset: {args.dataset_npz}", flush=True)
    print(f"Examples: {int(split_stats['num_examples'])}", flush=True)
    print(f"Overall positive rate: {split_stats['positive_rate']:.6g}", flush=True)
    print(
        "Train split: "
        f"{int(split_stats['train_examples'])} examples, "
        f"{int(split_stats['train_positives'])} positives, "
        f"positive_rate={split_stats['train_positive_rate']:.6g}",
        flush=True,
    )
    print(
        "Validation split: "
        f"{int(split_stats['val_examples'])} examples, "
        f"{int(split_stats['val_positives'])} positives, "
        f"positive_rate={split_stats['val_positive_rate']:.6g}",
        flush=True,
    )
    if best_checkpoint is not None:
        print(
            f"Best checkpoint: {best_checkpoint} "
            f"({args.save_best_mode} {args.save_best_metric})",
            flush=True,
        )
    if args.resume_from is not None:
        print(f"Resume from: {args.resume_from}", flush=True)

    metadata = {
        "dataset_npz": str(args.dataset_npz),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "device": args.device,
        "event_positive_weight": args.event_positive_weight,
        "history_frames": args.history_frames,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "lr_scheduler": args.lr_scheduler,
    }
    model, metrics = train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
        split_strategy=args.split_strategy,
        event_positive_weight=_parse_event_positive_weight(args.event_positive_weight),
        progress_callback=None if args.quiet else _print_epoch_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_metadata=metadata,
        best_checkpoint_path=best_checkpoint,
        save_best_metric=args.save_best_metric,
        save_best_mode=args.save_best_mode,
        resume_from=args.resume_from,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lr_scheduler=args.lr_scheduler,
    )

    print(f"Final checkpoint: {args.checkpoint}")
    if best_checkpoint is not None:
        print(f"Best checkpoint: {best_checkpoint}")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6g}")


def _print_epoch_progress(epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
    fields = [
        f"epoch {epoch}/{total_epochs}",
        f"train_loss={metrics.get('train_loss', float('nan')):.6g}",
        f"event_positive_weight={metrics.get('event_positive_weight', float('nan')):.6g}",
    ]
    for key in (
        "val_loss",
        "val_auroc",
        "val_auprc",
        "val_top25_enrichment",
        "learning_rate",
    ):
        if key in metrics:
            fields.append(f"{key}={metrics[key]:.6g}")
    print(" | ".join(fields), flush=True)


def _parse_event_positive_weight(value: str) -> float | str:
    if value == "auto":
        return value
    return float(value)


def _default_best_checkpoint(path: Path) -> Path:
    suffix = path.suffix
    if suffix:
        return path.with_name(path.name[: -len(suffix)] + ".best" + suffix)
    return path.with_name(path.name + ".best.pt")


if __name__ == "__main__":
    main()
