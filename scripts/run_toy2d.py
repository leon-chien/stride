from __future__ import annotations

from pathlib import Path

import yaml

from stride.features.toy2d import (
    Toy2DConfig,
    build_windows,
    event_rate,
    save_dataset,
    simulate_dataset,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config("configs/toy2d.yaml")

    sim_cfg = Toy2DConfig(**config["simulation"])

    trajectories = simulate_dataset(sim_cfg)

    rate = event_rate(
        trajectories,
        sim_cfg,
    )

    dataset_cfg = config["dataset"]

    X, y = build_windows(
        trajectories=trajectories,
        cfg=sim_cfg,
        window_size=dataset_cfg["window_size"],
        horizon=dataset_cfg["horizon"],
        stride=dataset_cfg["stride"],
    )

    print("Toy2D dataset generated")
    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Trajectory-level event rate: {rate:.3f}")
    print(f"Window dataset X shape: {X.shape}")
    print(f"Window labels y shape: {y.shape}")
    print(f"Window positive rate: {y.mean():.3f}")
    print(f"Event type: {sim_cfg.event_type}")
    if sim_cfg.event_type == "upper_gate":
        print(f"Gate y min: {sim_cfg.gate_y_min}")

    output_cfg = config["output"]

    if output_cfg.get("save_npz", True):
        save_dataset(
            output_dir=output_cfg["dir"],
            trajectories=trajectories,
            X=X,
            y=y,
        )
        print(f"Saved dataset to {output_cfg['dir']}/toy2d_dataset.npz")


if __name__ == "__main__":
    main()