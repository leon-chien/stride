from __future__ import annotations

import argparse
from pathlib import Path

from stride.training import (
    load_dataset_and_make_config,
    save_atomistic_checkpoint,
    train_atomistic_value_model,
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--egnn-layers", type=int, default=3)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--device", default=None)
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
    model, metrics = train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )
    save_atomistic_checkpoint(args.checkpoint, model, metrics)

    print(f"Dataset: {args.dataset_npz}")
    print(f"Checkpoint: {args.checkpoint}")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6g}")


if __name__ == "__main__":
    main()
