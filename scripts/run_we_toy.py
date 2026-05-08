from __future__ import annotations

from pathlib import Path

import yaml

from stride.features.toy2d import Toy2DConfig
from stride.sampling.we_toy import WEToyConfig, run_we_comparison


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def first_iteration_reached(history: list[dict[str, float]]) -> int | None:
    for row in history:
        if row["population_reached"] > 0:
            return int(row["iteration"])
    return None


def print_we_table(
    static_history: list[dict[str, float]],
    model_history: list[dict[str, float]],
    hybrid_history: list[dict[str, float]],
    every: int = 5,
) -> None:
    print("\nWE-style toy comparison:")
    print(
        f"{'iter':>5} | "
        f"{'static_w':>9} | {'model_w':>9} | {'hybrid_w':>9} | "
        f"{'static_u':>8} | {'model_u':>8} | {'hybrid_u':>8} | "
        f"{'static_pop':>10} | {'model_pop':>9} | {'hybrid_pop':>10}"
    )
    print("-" * 112)

    for s, m, h in zip(static_history, model_history, hybrid_history):
        iteration = int(s["iteration"])

        if iteration % every != 0 and iteration != 1:
            continue

        print(
            f"{iteration:>5} | "
            f"{s['target_weight']:>9.4f} | "
            f"{m['target_weight']:>9.4f} | "
            f"{h['target_weight']:>9.4f} | "
            f"{int(s['cumulative_unique_reached']):>8} | "
            f"{int(m['cumulative_unique_reached']):>8} | "
            f"{int(h['cumulative_unique_reached']):>8} | "
            f"{int(s['population_reached']):>10} | "
            f"{int(m['population_reached']):>9} | "
            f"{int(h['population_reached']):>10}"
        )


def main() -> None:
    config = load_config("configs/toy2d.yaml")

    sim_cfg = Toy2DConfig(**config["simulation"])

    we_cfg = WEToyConfig(
        num_walkers=256,
        num_iterations=80,
        segment_length=5,
        window_size=config["dataset"]["window_size"],
        num_bins=8,
        target_per_bin=32,
        seed=123,
    )

    checkpoint_path = "checkpoints/toy_gru.pt"

    print("Running WE-style toy comparison...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num walkers: {we_cfg.num_walkers}")
    print(f"Static/model bins: {we_cfg.num_bins}")
    print(f"Hybrid score bins: {we_cfg.num_score_bins}")
    print(f"Hybrid distance bins: {we_cfg.num_distance_bins}")
    print(f"Target total walkers preserved per method: {we_cfg.num_walkers}")
    print(f"Priority alpha: {we_cfg.priority_alpha}")
    print(f"Minimum walkers per occupied bin: {we_cfg.min_count_per_bin}")
    print(f"Num iterations: {we_cfg.num_iterations}")
    print(f"Segment length: {we_cfg.segment_length}")
    print(f"Window size: {we_cfg.window_size}")

    results = run_we_comparison(
        sim_cfg=sim_cfg,
        we_cfg=we_cfg,
        checkpoint_path=checkpoint_path,
    )

    static_history = results["static"]
    model_history = results["model"]
    hybrid_history = results["hybrid"]

    print_we_table(static_history, model_history, hybrid_history, every=5)

    static_first = first_iteration_reached(static_history)
    model_first = first_iteration_reached(model_history)
    hybrid_first = first_iteration_reached(hybrid_history)

    print("\nSummary:")
    print(f"Static first reached iteration: {static_first}")
    print(f"Model first reached iteration: {model_first}")
    print(f"Hybrid first reached iteration: {hybrid_first}")

    static_final = static_history[-1]
    model_final = model_history[-1]
    hybrid_final = hybrid_history[-1]

    print("\nFinal weighted population:")
    print(
        f"Static: target_weight={static_final['target_weight']:.4f}, "
        f"population_reached={int(static_final['population_reached'])}/"
        f"{int(static_final['num_walkers'])}, "
        f"unique_lineages={int(static_final['cumulative_unique_reached'])}, "
        f"total_weight={static_final['total_weight']:.4f}"
    )

    print(
        f"Model:  target_weight={model_final['target_weight']:.4f}, "
        f"population_reached={int(model_final['population_reached'])}/"
        f"{int(model_final['num_walkers'])}, "
        f"unique_lineages={int(model_final['cumulative_unique_reached'])}, "
        f"total_weight={model_final['total_weight']:.4f}"
    )

    print(
        f"Hybrid: target_weight={hybrid_final['target_weight']:.4f}, "
        f"population_reached={int(hybrid_final['population_reached'])}/"
        f"{int(hybrid_final['num_walkers'])}, "
        f"unique_lineages={int(hybrid_final['cumulative_unique_reached'])}, "
        f"total_weight={hybrid_final['total_weight']:.4f}"
    )


if __name__ == "__main__":
    main()