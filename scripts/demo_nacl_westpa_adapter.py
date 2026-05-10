from __future__ import annotations

import numpy as np

from stride.westpa_plugin.nacl_adapter import (
    NaClAdapterConfig,
    NaClWESTPAAdapter,
)


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

    print("Demo: NaCl WESTPA-style STRIDE adapter")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    fixed_adapter = NaClWESTPAAdapter(
        checkpoint_path=checkpoint_path,
        config=config,
        mode="fixed",
    )

    quantile_adapter = NaClWESTPAAdapter(
        checkpoint_path=checkpoint_path,
        config=config,
        mode="quantile",
        reference_windows=reference_windows,
    )

    # Choose a few pseudo-walkers from different trajectories.
    walker_indices = [0, 5, 10, 25, 50, 100, 250, 500]

    # Pretend these are recent WESTPA pcoord histories.
    # Use first 80 frames, then the adapter will take the last 25.
    distance_histories = np.stack(
        [
            distance_trajectories[i, :80]
            for i in walker_indices
        ]
    ).astype(np.float32)

    fixed_bins, fixed_scores = fixed_adapter.assign_distance_histories(
        distance_histories
    )
    q_bins, q_scores = quantile_adapter.assign_distance_histories(
        distance_histories
    )

    print(
        f"{'walker':>8} | {'last_dist':>10} | "
        f"{'score':>10} | {'fixed_bin':>9} | {'q_bin':>6}"
    )
    print("-" * 62)

    for idx, distance_history, score, fixed_bin, q_bin in zip(
        walker_indices,
        distance_histories,
        fixed_scores,
        fixed_bins,
        q_bins,
    ):
        last_dist = float(distance_history[-1])

        print(
            f"{idx:>8} | "
            f"{last_dist:>10.4f} | "
            f"{score:>10.4f} | "
            f"{int(fixed_bin):>9} | "
            f"{int(q_bin):>6}"
        )

    print("\nSingle-walker assignment example:")

    assignment = fixed_adapter.assign_distance_history(distance_histories[0])

    print(
        f"walker={walker_indices[0]}, "
        f"score={assignment.score:.4f}, "
        f"bin_id={assignment.bin_id}"
    )


if __name__ == "__main__":
    main()