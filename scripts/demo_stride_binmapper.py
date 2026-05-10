from __future__ import annotations

import numpy as np

from stride.westpa_plugin.stride_binmapper import (
    StrideQuantileBinMapper,
    StrideScoreBinMapper,
)


def load_demo_windows(
    dataset_path: str = "outputs/nacl/nacl_dataset.npz",
    num_examples: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a few NaCl windows for demonstrating learned bin assignment.
    """
    data = np.load(dataset_path)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    # Pick examples spread across the dataset.
    indices = np.linspace(0, len(X) - 1, num_examples, dtype=int)

    return X[indices], y[indices]


def main() -> None:
    checkpoint_path = "checkpoints/nacl_gru.pt"
    dataset_path = "outputs/nacl/nacl_dataset.npz"

    windows, labels = load_demo_windows(
        dataset_path=dataset_path,
        num_examples=12,
    )

    print("Demo: STRIDE learned BinMapper for NaCl")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print()

    print("Fixed probability-width bins:")
    fixed_mapper = StrideScoreBinMapper(
        checkpoint_path=checkpoint_path,
        num_bins=8,
    )

    bin_ids, scores = fixed_mapper.assign_batch(windows)

    print(
        f"{'example':>7} | {'label':>5} | {'score':>10} | {'bin_id':>6}"
    )
    print("-" * 40)

    for i, (label, score, bin_id) in enumerate(zip(labels, scores, bin_ids)):
        print(
            f"{i:>7} | "
            f"{int(label):>5} | "
            f"{score:>10.4f} | "
            f"{int(bin_id):>6}"
        )

    print("\nQuantile bins using dataset reference windows:")

    # Use part of the dataset as reference to learn quantile score edges.
    data = np.load(dataset_path)
    reference_windows = data["X"][:5000].astype(np.float32)

    quantile_mapper = StrideQuantileBinMapper(
        checkpoint_path=checkpoint_path,
        reference_windows=reference_windows,
        num_bins=8,
    )

    q_bin_ids, q_scores = quantile_mapper.assign_batch(windows)

    print(
        f"{'example':>7} | {'label':>5} | {'score':>10} | {'q_bin_id':>8}"
    )
    print("-" * 44)

    for i, (label, score, bin_id) in enumerate(zip(labels, q_scores, q_bin_ids)):
        print(
            f"{i:>7} | "
            f"{int(label):>5} | "
            f"{score:>10.4f} | "
            f"{int(bin_id):>8}"
        )

    print("\nQuantile bin edges:")
    print(quantile_mapper.edges)


if __name__ == "__main__":
    main()