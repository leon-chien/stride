from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stride.data import load_atomistic_dataset_npz
from stride.training import load_atomistic_checkpoint, score_atomistic_dataset
from stride.training.metrics import compute_binary_metrics, top_k_enrichment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score an atomistic STRIDE dataset with a trained checkpoint."
    )
    parser.add_argument("dataset_npz", type=Path, help="Atomistic STRIDE .npz dataset.")
    parser.add_argument("checkpoint", type=Path, help="Trained PyTorch checkpoint.")
    parser.add_argument("output_npz", type=Path, help="Output .npz score artifact.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dataset = load_atomistic_dataset_npz(args.dataset_npz)
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

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        **scores,
        event_labels=dataset.event_labels,
        flux_labels=dataset.flux_labels,
        source_frame_start=dataset.source_frame_start,
    )

    metrics = compute_binary_metrics(dataset.event_labels, scores["p_event"])
    metrics["top25_enrichment"] = top_k_enrichment(
        dataset.event_labels,
        scores["p_event"],
        k=0.25,
    )

    print(f"Dataset: {args.dataset_npz}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Scores: {args.output_npz}")
    if checkpoint_metrics:
        print(f"Checkpoint train_loss: {checkpoint_metrics.get('train_loss', float('nan')):.6g}")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6g}")


if __name__ == "__main__":
    main()
