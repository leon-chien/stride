from __future__ import annotations

from pathlib import Path

import yaml

from stride.features.nacl import (
    NaClConfig,
    build_nacl_windows,
    compute_feature_dataset,
    event_rate,
    save_nacl_dataset,
    simulate_nacl_dataset,
)
from stride.replay.nacl_replay import run_nacl_replay


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_heldout_dataset(
    config_path: str | Path = "configs/nacl.yaml",
    heldout_seed: int = 123,
    output_dir: str | Path = "outputs/nacl_heldout_seed_123",
) -> Path:
    """
    Generate a new synthetic NaCl dataset using a different seed.

    This dataset is not used for training. It is only used to evaluate
    whether the frozen trained model generalizes to new simulations.
    """
    config = load_config(config_path)

    system_config_dict = dict(config["system"])
    system_config_dict["seed"] = heldout_seed

    system_cfg = NaClConfig(**system_config_dict)
    dataset_cfg = config["dataset"]

    print("Generating held-out NaCl association dataset...")
    print(f"Held-out seed: {heldout_seed}")
    print(f"Event type: {system_cfg.event_type}")
    print(f"Association distance: {system_cfg.association_distance}")

    distance_trajectories = simulate_nacl_dataset(system_cfg)

    feature_trajectories = compute_feature_dataset(
        distance_trajectories=distance_trajectories,
        cfg=system_cfg,
    )

    rate = event_rate(
        distance_trajectories=distance_trajectories,
        cfg=system_cfg,
    )

    X, y = build_nacl_windows(
        feature_trajectories=feature_trajectories,
        distance_trajectories=distance_trajectories,
        cfg=system_cfg,
        window_size=dataset_cfg["window_size"],
        horizon=dataset_cfg["horizon"],
        stride=dataset_cfg["stride"],
    )

    print("Held-out NaCl dataset generated")
    print(f"Distance trajectories shape: {distance_trajectories.shape}")
    print(f"Feature trajectories shape: {feature_trajectories.shape}")
    print(f"Trajectory-level association rate: {rate:.3f}")
    print(f"Window dataset X shape: {X.shape}")
    print(f"Window labels y shape: {y.shape}")
    print(f"Window positive rate: {y.mean():.3f}")

    output_dir = Path(output_dir)

    save_nacl_dataset(
        output_dir=output_dir,
        distance_trajectories=distance_trajectories,
        feature_trajectories=feature_trajectories,
        X=X,
        y=y,
    )

    dataset_path = output_dir / "nacl_dataset.npz"

    print(f"Saved held-out dataset to {dataset_path}")

    return dataset_path


def main() -> None:
    heldout_seed = 123
    output_dir = f"outputs/nacl_heldout_seed_{heldout_seed}"

    dataset_path = generate_heldout_dataset(
        config_path="configs/nacl.yaml",
        heldout_seed=heldout_seed,
        output_dir=output_dir,
    )

    print("\nRunning frozen-model replay on held-out seed dataset...")

    results = run_nacl_replay(
        dataset_path=dataset_path,
        checkpoint_path="checkpoints/nacl_gru.pt",
        batch_size=512,
        num_bins=4,
        top_k=0.10,
        seed=heldout_seed,
    )

    metrics = results["metrics"]

    print("\nHeld-out seed replay results:")
    print(f"Held-out seed: {heldout_seed}")
    print(f"Overall positive rate: {metrics['positive_rate']:.4f}")
    print(f"Random top 10% positive rate: {metrics['random_top10_positive_rate']:.4f}")
    print(f"Model top 10% positive rate: {metrics['top10_positive_rate']:.4f}")
    print(f"Top 10% enrichment: {metrics['top10_enrichment']:.2f}x")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")

    print("\nBin summary:")
    print(
        f"{'bin':>5} | {'n':>8} | {'frac':>8} | "
        f"{'pos_rate':>10} | {'mean_score':>10} | {'min_score':>10} | {'max_score':>10}"
    )
    print("-" * 86)

    for row in results["bin_summary"]:
        print(
            f"{int(row['bin_id']):>5} | "
            f"{int(row['n']):>8} | "
            f"{row['fraction']:>8.3f} | "
            f"{row['positive_rate']:>10.4f} | "
            f"{row['mean_score']:>10.4f} | "
            f"{row['min_score']:>10.4f} | "
            f"{row['max_score']:>10.4f}"
        )


if __name__ == "__main__":
    main()