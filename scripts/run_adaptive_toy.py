from __future__ import annotations

from pathlib import Path

import yaml

from stride.features.toy2d import Toy2DConfig
from stride.sampling.adaptive_toy import AdaptiveToyConfig, run_comparison


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def first_iteration_reached(history: list[dict[str, float]]) -> int | None:
    """
    Return the first iteration where at least one walker reached the target.
    """
    for row in history:
        if row["num_reached"] > 0:
            return int(row["iteration"])

    return None


def print_history_table(
    random_history: list[dict[str, float]],
    model_history: list[dict[str, float]],
    every: int = 5,
) -> None:
    """
    Print compact side-by-side comparison.
    """
    print("\nAdaptive toy comparison:")
    print(
        f"{'iter':>5} | "
        f"{'rand_pop':>8} | {'model_pop':>9} | "
        f"{'rand_new':>8} | {'model_new':>9} | "
        f"{'rand_unique':>11} | {'model_unique':>12} | "
        f"{'model_score':>11}"
    )
    print("-" * 96)

    for r, m in zip(random_history, model_history):
        iteration = int(r["iteration"])

        if iteration % every != 0 and iteration != 1:
            continue

        print(
            f"{iteration:>5} | "
            f"{int(r['num_reached']):>8} | "
            f"{int(m['num_reached']):>9} | "
            f"{int(r['new_reached_this_iteration']):>8} | "
            f"{int(m['new_reached_this_iteration']):>9} | "
            f"{int(r['cumulative_unique_reached']):>11} | "
            f"{int(m['cumulative_unique_reached']):>12} | "
            f"{m.get('mean_score', 0.0):>11.4f}"
        )


def main() -> None:
    config = load_config("configs/toy2d.yaml")

    sim_cfg = Toy2DConfig(**config["simulation"])

    adaptive_cfg = AdaptiveToyConfig(
        num_walkers=256,
        num_iterations=80,
        segment_length=5,
        window_size=config["dataset"]["window_size"],
        resample_temperature=1.0,
        seed=123,
    )

    checkpoint_path = "checkpoints/toy_gru.pt"

    print("Running adaptive toy comparison...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num walkers: {adaptive_cfg.num_walkers}")
    print(f"Num iterations: {adaptive_cfg.num_iterations}")
    print(f"Segment length: {adaptive_cfg.segment_length}")
    print(f"Window size: {adaptive_cfg.window_size}")

    results = run_comparison(
        sim_cfg=sim_cfg,
        adaptive_cfg=adaptive_cfg,
        checkpoint_path=checkpoint_path,
    )

    random_history = results["random"]
    model_history = results["model"]

    print_history_table(random_history, model_history, every=5)

    random_first = first_iteration_reached(random_history)
    model_first = first_iteration_reached(model_history)

    print("\nSummary:")
    print(f"Random first reached iteration: {random_first}")
    print(f"Model first reached iteration: {model_first}")

    random_final = random_history[-1]
    model_final = model_history[-1]

    print("\nFinal population:")
    print(
        f"Random: population_reached={int(random_final['num_reached'])}/"
        f"{adaptive_cfg.num_walkers}, "
        f"fraction={random_final['fraction_reached']:.3f}, "
        f"unique_lineages={int(random_final['cumulative_unique_reached'])}/"
        f"{adaptive_cfg.num_walkers}, "
        f"min_distance={random_final['min_distance']:.4f}"
    )
    print(
        f"Model:  population_reached={int(model_final['num_reached'])}/"
        f"{adaptive_cfg.num_walkers}, "
        f"fraction={model_final['fraction_reached']:.3f}, "
        f"unique_lineages={int(model_final['cumulative_unique_reached'])}/"
        f"{adaptive_cfg.num_walkers}, "
        f"min_distance={model_final['min_distance']:.4f}"
    )


if __name__ == "__main__":
    main()