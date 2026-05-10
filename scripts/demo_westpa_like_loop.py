from __future__ import annotations

import numpy as np

from stride.westpa_plugin.nacl_adapter import (
    NaClAdapterConfig,
    NaClWESTPAAdapter,
)


def summarize_bins(bin_ids: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    """
    Summarize bin occupancy and score statistics.
    """
    rows: list[dict[str, float]] = []

    for bin_id in sorted(np.unique(bin_ids)):
        mask = bin_ids == bin_id

        rows.append(
            {
                "bin_id": int(bin_id),
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "mean_score": float(scores[mask].mean()),
                "min_score": float(scores[mask].min()),
                "max_score": float(scores[mask].max()),
            }
        )

    return rows


def main() -> None:
    dataset_path = "outputs/nacl/nacl_dataset.npz"
    checkpoint_path = "checkpoints/nacl_gru.pt"

    data = np.load(dataset_path)

    distance_trajectories = data["distance_trajectories"].astype(np.float32)
    reference_windows = data["X"][:5000].astype(np.float32)

    config = NaClAdapterConfig(
        dt=0.02,
        window_size=25,
        num_bins=8,
    )

    adapter = NaClWESTPAAdapter(
        checkpoint_path=checkpoint_path,
        config=config,
        mode="quantile",
        reference_windows=reference_windows,
    )

    # Pretend these are WESTPA walkers.
    num_walkers = 64

    # Pick 64 trajectories as fake walkers.
    walker_indices = np.linspace(
        0,
        len(distance_trajectories) - 1,
        num_walkers,
        dtype=int,
    )

    print("Demo: WESTPA-like STRIDE batch bin assignment")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num fake walkers: {num_walkers}")
    print()

    # Simulate several WESTPA iterations.
    # At each iteration, WESTPA would have recent pcoord history for each walker.
    # Here we take progressively later chunks from the saved distance trajectories.
    iteration_end_frames = [25, 50, 75, 100, 150, 200, 300]

    for iteration, end_frame in enumerate(iteration_end_frames, start=1):
        if end_frame < config.window_size:
            continue

        distance_histories = np.stack(
            [
                distance_trajectories[i, :end_frame]
                for i in walker_indices
            ]
        ).astype(np.float32)

        bin_ids, scores = adapter.assign_distance_histories(distance_histories)

        last_distances = distance_histories[:, -1]

        print(f"Iteration {iteration} | end_frame={end_frame}")
        print(
            f"  mean_last_distance={last_distances.mean():.4f} | "
            f"min_last_distance={last_distances.min():.4f} | "
            f"mean_score={scores.mean():.4f} | "
            f"max_score={scores.max():.4f}"
        )

        print(
            f"  {'bin':>5} | {'count':>6} | {'frac':>7} | "
            f"{'mean_score':>10} | {'min_score':>10} | {'max_score':>10}"
        )
        print("  " + "-" * 65)

        for row in summarize_bins(bin_ids, scores):
            print(
                f"  {row['bin_id']:>5} | "
                f"{row['count']:>6} | "
                f"{row['fraction']:>7.3f} | "
                f"{row['mean_score']:>10.4f} | "
                f"{row['min_score']:>10.4f} | "
                f"{row['max_score']:>10.4f}"
            )

        print()

    print("Done.")
    print(
        "\nThis script mimics the key WESTPA-facing behavior: "
        "a batch of walker pcoord histories goes in, and STRIDE returns bin IDs."
    )


if __name__ == "__main__":
    main()