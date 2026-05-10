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


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config("configs/nacl.yaml")

    system_cfg = NaClConfig(**config["system"])
    dataset_cfg = config["dataset"]

    print("Generating synthetic NaCl association dataset...")
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

    print("NaCl dataset generated")
    print(f"Distance trajectories shape: {distance_trajectories.shape}")
    print(f"Feature trajectories shape: {feature_trajectories.shape}")
    print(f"Trajectory-level association rate: {rate:.3f}")
    print(f"Window dataset X shape: {X.shape}")
    print(f"Window labels y shape: {y.shape}")
    print(f"Window positive rate: {y.mean():.3f}")

    output_cfg = config["output"]

    if output_cfg.get("save_npz", True):
        save_nacl_dataset(
            output_dir=output_cfg["dir"],
            distance_trajectories=distance_trajectories,
            feature_trajectories=feature_trajectories,
            X=X,
            y=y,
        )
        print(f"Saved dataset to {output_cfg['dir']}/nacl_dataset.npz")


if __name__ == "__main__":
    main()