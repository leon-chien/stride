from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev

import yaml

from stride.features.toy2d import Toy2DConfig
from stride.sampling.we_toy import WEToyConfig, run_we_comparison


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def first_iteration_reached(history: list[dict[str, float]]) -> int | None:
    """
    Return the first iteration where at least one walker reached the target.
    """
    for row in history:
        if row["population_reached"] > 0:
            return int(row["iteration"])
    return None


def summarize_history(history: list[dict[str, float]]) -> dict[str, float]:
    """
    Extract final and summary metrics from one method's history.
    """
    final = history[-1]
    first_reached = first_iteration_reached(history)

    return {
        "first_reached": float(first_reached) if first_reached is not None else float("nan"),
        "final_target_weight": float(final["target_weight"]),
        "final_population_reached": float(final["population_reached"]),
        "final_fraction_reached": float(final["fraction_reached"]),
        "final_unique_lineages": float(final["cumulative_unique_reached"]),
        "final_total_weight": float(final["total_weight"]),
    }


def safe_mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # removes NaN
    if not clean:
        return float("nan")
    return mean(clean)


def safe_std(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # removes NaN
    if len(clean) < 2:
        return 0.0
    return stdev(clean)


def format_mean_std(values: list[float], decimals: int = 4) -> str:
    m = safe_mean(values)
    s = safe_std(values)

    if m != m:
        return "nan"

    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def main() -> None:
    config = load_config("configs/toy2d.yaml")
    sim_cfg = Toy2DConfig(**config["simulation"])

    checkpoint_path = "checkpoints/toy_gru.pt"

    # Increase this later to 50 or 100.
    seeds = list(range(20))

    all_results: dict[str, list[dict[str, float]]] = {
        "static": [],
        "model": [],
        "hybrid": [],
    }

    print("Running multi-seed WE-style toy benchmark...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Number of seeds: {len(seeds)}")
    print(f"Seeds: {seeds}")
    print()

    for seed in seeds:
        we_cfg = WEToyConfig(
            num_walkers=256,
            num_iterations=80,
            segment_length=5,
            window_size=config["dataset"]["window_size"],
            num_bins=8,
            num_score_bins=4,
            num_distance_bins=4,
            target_per_bin=32,
            diversity_priority_weight=0.4,
            score_priority_weight=0.4,
            probability_priority_weight=0.2,
            min_count_per_bin=8,
            seed=seed,
        )

        results = run_we_comparison(
            sim_cfg=sim_cfg,
            we_cfg=we_cfg,
            checkpoint_path=checkpoint_path,
        )

        for method in ["static", "model", "hybrid"]:
            summary = summarize_history(results[method])
            summary["seed"] = float(seed)
            all_results[method].append(summary)

        static_final = all_results["static"][-1]
        model_final = all_results["model"][-1]
        hybrid_final = all_results["hybrid"][-1]

        print(
            f"Seed {seed:02d} | "
            f"static_w={static_final['final_target_weight']:.4f} | "
            f"model_w={model_final['final_target_weight']:.4f} | "
            f"hybrid_w={hybrid_final['final_target_weight']:.4f}"
        )

    print("\nMulti-seed summary:")
    print(
        f"{'method':>10} | "
        f"{'target_weight':>20} | "
        f"{'unique_lineages':>20} | "
        f"{'pop_reached':>20} | "
        f"{'first_reached':>20}"
    )
    print("-" * 104)

    for method in ["static", "model", "hybrid"]:
        rows = all_results[method]

        target_weights = [r["final_target_weight"] for r in rows]
        unique_lineages = [r["final_unique_lineages"] for r in rows]
        population_reached = [r["final_population_reached"] for r in rows]
        first_reached = [r["first_reached"] for r in rows]

        print(
            f"{method:>10} | "
            f"{format_mean_std(target_weights):>20} | "
            f"{format_mean_std(unique_lineages):>20} | "
            f"{format_mean_std(population_reached):>20} | "
            f"{format_mean_std(first_reached):>20}"
        )

    print("\nWin rates by final target weight:")

    methods = ["static", "model", "hybrid"]
    wins = {method: 0 for method in methods}

    for i in range(len(seeds)):
        seed_weights = {
            method: all_results[method][i]["final_target_weight"]
            for method in methods
        }

        winner = max(seed_weights, key=seed_weights.get)
        wins[winner] += 1

    for method in methods:
        print(f"{method:>10}: {wins[method]}/{len(seeds)}")

    print("\nPairwise improvement over static target weight:")

    static_weights = [r["final_target_weight"] for r in all_results["static"]]
    model_weights = [r["final_target_weight"] for r in all_results["model"]]
    hybrid_weights = [r["final_target_weight"] for r in all_results["hybrid"]]

    model_ratios = [
        m / s if s > 0 else float("nan")
        for m, s in zip(model_weights, static_weights)
    ]

    hybrid_ratios = [
        h / s if s > 0 else float("nan")
        for h, s in zip(hybrid_weights, static_weights)
    ]

    print(f"{'model/static':>15}: {format_mean_std(model_ratios)}x")
    print(f"{'hybrid/static':>15}: {format_mean_std(hybrid_ratios)}x")


if __name__ == "__main__":
    main()