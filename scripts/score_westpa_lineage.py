from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stride.training import (
    load_pcoord_lineage_checkpoint,
    load_pcoord_lineage_dataset_npz,
    score_pcoord_lineage_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score WESTPA pcoord lineage windows with a trained STRIDE checkpoint."
    )
    parser.add_argument("lineage_npz", type=Path, help="Output from extract_westpa_dataset.py.")
    parser.add_argument("checkpoint", type=Path, help="Trained pcoord lineage checkpoint.")
    parser.add_argument("output_npz", type=Path, help="Output score .npz.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset = load_pcoord_lineage_dataset_npz(args.lineage_npz)
    model, metrics = load_pcoord_lineage_checkpoint(args.checkpoint, device=args.device)
    scores = score_pcoord_lineage_dataset(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **scores)

    print(f"Dataset: {args.lineage_npz}")
    print(f"Checkpoint: {args.checkpoint}")
    if metrics:
        print(f"Checkpoint train_loss: {metrics.get('train_loss', float('nan')):.6g}")
    for key, values in scores.items():
        print(f"{key}: shape={values.shape} mean={float(np.mean(values)):.6g}")
    print(f"Output: {args.output_npz}")


if __name__ == "__main__":
    main()
