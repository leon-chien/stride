from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev

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


def safe_mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    if not clean:
        return float("nan")
    return mean(clean)


def safe_std(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    if len(clean) < 2:
        return 0.0
    return stdev(clean)


def format_mean_std(values: list[float], decimals: int = 4) -> str:
    m = safe_mean(values)
    s = safe_std(values)

    if m != m:
        return "nan"

    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def generate_heldout_dataset_for_seed(
    config: dict,
    heldout_seed: int,
    output_dir: str | Path,
) -> Path:
    """
    Generate one held-out NaCl dataset for a specific seed.
    """
    system_config_dict = dict(config["system"])
    system_config_dict["seed"] = heldout_seed

    system_cfg = NaClConfig(**system_config_dict)
    dataset_cfg = config["dataset"]

    distance_trajectories = simulate_nacl_dataset(system_cfg)

    feature_trajectories = compute_feature_dataset(
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

    output_dir = Path(output_dir)

    save_nacl_dataset(
        output_dir=output_dir,
        distance_trajectories=distance_trajectories,
        feature_trajectories=feature_trajectories,
        X=X,
        y=y,
    )

    rate = event_rate(distance_trajectories, system_cfg)

    print(
        f"Seed {heldout_seed:03d} dataset | "
        f"trajectory_rate={rate:.3f} | "
        f"window_positive_rate={y.mean():.3f}"
    )

    return output_dir / "nacl_dataset.npz"


def main() -> None:
    config = load_config("configs/nacl.yaml")

    checkpoint_path = "checkpoints/nacl_gru.pt"

    # These seeds should be different from the training seed, which is currently 42.
    heldout_seeds = list(range(100, 120))

    output_root = Path("outputs/nacl_heldout_multiseed")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []

    print("Running held-out multi-seed NaCl replay...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Held-out seeds: {heldout_seeds}")
    print()

    for heldout_seed in heldout_seeds:
        seed_output_dir = output_root / f"seed_{heldout_seed}"

        dataset_path = generate_heldout_dataset_for_seed(
            config=config,
            heldout_seed=heldout_seed,
            output_dir=seed_output_dir,
        )

        results = run_nacl_replay(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            batch_size=512,
            num_bins=4,
            top_k=0.10,
            seed=heldout_seed,
        )

        metrics = results["metrics"]

        row = {
            "seed": float(heldout_seed),
            "positive_rate": float(metrics["positive_rate"]),
            "random_top10_positive_rate": float(metrics["random_top10_positive_rate"]),
            "model_top10_positive_rate": float(metrics["top10_positive_rate"]),
            "top10_enrichment": float(metrics["top10_enrichment"]),
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
        }

        rows.append(row)

        print(
            f"Seed {heldout_seed:03d} replay | "
            f"pos={row['positive_rate']:.4f} | "
            f"random_top10={row['random_top10_positive_rate']:.4f} | "
            f"model_top10={row['model_top10_positive_rate']:.4f} | "
            f"enrichment={row['top10_enrichment']:.2f}x | "
            f"auroc={row['auroc']:.4f} | "
            f"auprc={row['auprc']:.4f}"
        )

        print("-" * 90)

    print("\nHeld-out multi-seed summary:")
    print(
        f"{'metric':>30} | {'mean ± std':>20}"
    )
    print("-" * 56)

    for metric in [
        "positive_rate",
        "random_top10_positive_rate",
        "model_top10_positive_rate",
        "top10_enrichment",
        "auroc",
        "auprc",
    ]:
        values = [row[metric] for row in rows]
        print(f"{metric:>30} | {format_mean_std(values):>20}")

    # Save summary CSV manually without pandas dependency.
    summary_path = output_root / "heldout_replay_results.csv"

    with open(summary_path, "w") as f:
        header = [
            "seed",
            "positive_rate",
            "random_top10_positive_rate",
            "model_top10_positive_rate",
            "top10_enrichment",
            "auroc",
            "auprc",
        ]
        f.write(",".join(header) + "\n")

        for row in rows:
            f.write(
                ",".join(str(row[key]) for key in header)
                + "\n"
            )

    print(f"\nSaved held-out replay results to {summary_path}")


if __name__ == "__main__":
    main()