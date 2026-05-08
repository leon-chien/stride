from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stride.binning.quantile_binner import QuantileBinner
from stride.features.toy2d import (
    Toy2DConfig,
    distance_to_target,
    force,
    potential,
    reaches_target,
)
from stride.models.gru_ranker import GRURanker
from stride.sampling.weighted_resampler import weighted_ensemble_resample


@dataclass
class WEWalker:
    """
    Weighted toy walker.

    lineage_id tracks the original starting walker.
    reached_target records whether this lineage has ever reached the target.
    """

    x: float
    y: float
    prev_x: float
    prev_y: float
    history: list[np.ndarray]
    lineage_id: int
    reached_target: bool = False

    def clone(self) -> "WEWalker":
        return WEWalker(
            x=self.x,
            y=self.y,
            prev_x=self.prev_x,
            prev_y=self.prev_y,
            history=[frame.copy() for frame in self.history],
            lineage_id=self.lineage_id,
            reached_target=self.reached_target,
        )


@dataclass
class WEToyConfig:
    """
    Configuration for WESTPA-like toy resampling.
    """

    num_walkers: int = 256
    num_iterations: int = 80
    segment_length: int = 5
    window_size: int = 25

    # Used by static and pure model-score binning.
    num_bins: int = 8

    # Used by hybrid score + distance binning.
    num_score_bins: int = 4
    num_distance_bins: int = 4

    # Fallback target walkers per bin.
    target_per_bin: int = 32

    # Version 1.2 priority-aware allocation settings.
    priority_alpha: float = 0.7
    min_count_per_bin: int = 4

    seed: int = 123


def make_feature_vector(
    x: float,
    y: float,
    prev_x: float,
    prev_y: float,
    dt: float,
    target_center: tuple[float, float],
) -> np.ndarray:
    """
    Same feature representation used by the toy GRU.

    Features:
        x, y, vx, vy, potential energy, distance to target
    """
    vx = (x - prev_x) / dt
    vy = (y - prev_y) / dt
    u = potential(x, y)
    d_target = distance_to_target(x, y, target_center)

    return np.array([x, y, vx, vy, u, d_target], dtype=np.float32)


def initialize_we_walkers(
    sim_cfg: Toy2DConfig,
    we_cfg: WEToyConfig,
    rng: np.random.Generator,
) -> tuple[list[WEWalker], np.ndarray]:
    """
    Initialize weighted walkers near the starting basin.
    """
    walkers: list[WEWalker] = []

    for i in range(we_cfg.num_walkers):
        x = float(rng.normal(sim_cfg.start_center[0], sim_cfg.start_std))
        y = float(rng.normal(sim_cfg.start_center[1], sim_cfg.start_std))

        feature = make_feature_vector(
            x=x,
            y=y,
            prev_x=x,
            prev_y=y,
            dt=sim_cfg.dt,
            target_center=sim_cfg.target_center,
        )

        walkers.append(
            WEWalker(
                x=x,
                y=y,
                prev_x=x,
                prev_y=y,
                history=[feature],
                lineage_id=i,
                reached_target=False,
            )
        )

    weights = np.full(we_cfg.num_walkers, 1.0 / we_cfg.num_walkers, dtype=np.float64)

    return walkers, weights


def step_we_walker(
    walker: WEWalker,
    sim_cfg: Toy2DConfig,
    rng: np.random.Generator,
) -> None:
    """
    Advance one walker by one toy dynamics step.
    """
    old_x = walker.x
    old_y = walker.y

    fx, fy = force(walker.x, walker.y)

    new_x = (
        walker.x
        + sim_cfg.dt * fx
        + sim_cfg.noise_std * np.sqrt(sim_cfg.dt) * rng.normal()
    )
    new_y = (
        walker.y
        + sim_cfg.dt * fy
        + sim_cfg.noise_std * np.sqrt(sim_cfg.dt) * rng.normal()
    )

    walker.prev_x = old_x
    walker.prev_y = old_y
    walker.x = float(new_x)
    walker.y = float(new_y)

    feature = make_feature_vector(
        x=walker.x,
        y=walker.y,
        prev_x=walker.prev_x,
        prev_y=walker.prev_y,
        dt=sim_cfg.dt,
        target_center=sim_cfg.target_center,
    )

    walker.history.append(feature)

    if reaches_target(
        walker.x,
        walker.y,
        sim_cfg.target_center,
        sim_cfg.target_radius,
    ):
        walker.reached_target = True


def run_we_segment(
    walkers: list[WEWalker],
    sim_cfg: Toy2DConfig,
    we_cfg: WEToyConfig,
    rng: np.random.Generator,
) -> None:
    """
    Run all walkers forward for one WE segment.
    """
    for _ in range(we_cfg.segment_length):
        for walker in walkers:
            step_we_walker(walker, sim_cfg, rng)


def walkers_to_windows(
    walkers: list[WEWalker],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert walker histories into fixed-length windows.
    """
    windows: list[np.ndarray] = []
    ready_indices: list[int] = []

    for i, walker in enumerate(walkers):
        if len(walker.history) >= window_size:
            windows.append(np.stack(walker.history[-window_size:]).astype(np.float32))
            ready_indices.append(i)

    if not windows:
        return np.empty((0, window_size, 6), dtype=np.float32), np.array([], dtype=int)

    return np.stack(windows).astype(np.float32), np.array(ready_indices, dtype=int)


def load_gru_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[GRURanker, np.ndarray, np.ndarray]:
    """
    Load trained GRU trajectory-value model.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = GRURanker(
        num_features=int(checkpoint["num_features"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["mean"], checkpoint["std"]


def score_we_walkers(
    walkers: list[WEWalker],
    model: GRURanker,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    window_size: int,
) -> np.ndarray:
    """
    Score walkers using the trained GRU.

    During warmup, walkers without enough history receive a neutral score.
    """
    scores = np.full(len(walkers), 0.5, dtype=np.float32)

    X, ready_indices = walkers_to_windows(walkers, window_size)

    if len(ready_indices) == 0:
        return scores

    X_norm = ((X - mean) / std).astype(np.float32)

    x_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    scores[ready_indices] = probs

    return scores


def assign_static_distance_bins(
    walkers: list[WEWalker],
    sim_cfg: Toy2DConfig,
    num_bins: int,
) -> np.ndarray:
    """
    Assign bins using distance to target.

    This is a hand-designed baseline progress coordinate.
    """
    xs = np.array([w.x for w in walkers], dtype=float)
    ys = np.array([w.y for w in walkers], dtype=float)

    distances = distance_to_target(xs, ys, sim_cfg.target_center)

    # Smaller distance = closer to target.
    # Use quantile bins so each occupied region has some walkers.
    binner = QuantileBinner(num_bins=num_bins)
    return binner.fit_transform(-distances)


def assign_model_score_bins(
    scores: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """
    Assign bins using model event-probability scores.

    Higher score = more likely future event.
    """
    binner = QuantileBinner(num_bins=num_bins)
    return binner.fit_transform(scores)


def assign_hybrid_score_distance_bins(
    walkers: list[WEWalker],
    sim_cfg: Toy2DConfig,
    scores: np.ndarray,
    num_score_bins: int,
    num_distance_bins: int,
) -> np.ndarray:
    """
    Assign hybrid bins using both model value and physical progress.

    This is Version 1.1's key idea.

    Pure model-score bins can over-exploit a few high-value lineages.
    Hybrid bins preserve more diversity by separating walkers using:

        1. model event probability
        2. distance-to-target progress coordinate

    The combined bin ID is:

        bin_id = score_bin * num_distance_bins + distance_bin

    So two walkers with similar model scores but different physical
    locations can still occupy different bins.
    """
    if num_score_bins < 2:
        raise ValueError("num_score_bins must be at least 2.")

    if num_distance_bins < 2:
        raise ValueError("num_distance_bins must be at least 2.")

    scores = np.asarray(scores, dtype=float)

    xs = np.array([w.x for w in walkers], dtype=float)
    ys = np.array([w.y for w in walkers], dtype=float)

    distances = distance_to_target(xs, ys, sim_cfg.target_center)

    score_binner = QuantileBinner(num_bins=num_score_bins)
    score_bins = score_binner.fit_transform(scores)

    # Smaller distance means closer to target, so use -distance.
    distance_binner = QuantileBinner(num_bins=num_distance_bins)
    distance_bins = distance_binner.fit_transform(-distances)

    combined_bin_ids = score_bins * num_distance_bins + distance_bins

    return combined_bin_ids.astype(int)


def compute_score_based_bin_priorities(
    bin_ids: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.7,
) -> dict[int, float]:
    """
    Compute bin priorities from model scores.

    Each bin priority combines:

        1. a uniform diversity floor
        2. the average model score in that bin

    alpha controls exploitation:

        alpha = 0.0 → uniform allocation
        alpha = 1.0 → fully score-driven allocation

    Recommended starting value:

        alpha = 0.7

    This is Version 1.2's core improvement.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    bin_ids = np.asarray(bin_ids, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if bin_ids.shape != scores.shape:
        raise ValueError(
            f"Shape mismatch: bin_ids has {bin_ids.shape}, scores has {scores.shape}"
        )

    unique_bins = sorted(int(b) for b in np.unique(bin_ids))
    num_bins = len(unique_bins)

    if num_bins == 0:
        raise ValueError("No occupied bins.")

    mean_scores = {}

    for bin_id in unique_bins:
        mask = bin_ids == bin_id
        mean_scores[bin_id] = float(scores[mask].mean())

    score_values = np.array([mean_scores[b] for b in unique_bins], dtype=np.float64)

    # Normalize scores into a nonnegative priority distribution.
    score_values = score_values - score_values.min()

    if score_values.sum() > 0:
        score_priority = score_values / score_values.sum()
    else:
        score_priority = np.ones(num_bins, dtype=np.float64) / num_bins

    uniform_priority = np.ones(num_bins, dtype=np.float64) / num_bins

    mixed_priority = (1.0 - alpha) * uniform_priority + alpha * score_priority

    return {
        bin_id: float(priority)
        for bin_id, priority in zip(unique_bins, mixed_priority)
    }


def find_new_lineages(
    walkers: list[WEWalker],
    discovered_lineages: set[int],
) -> set[int]:
    """
    Identify lineages that reached target for the first time.
    """
    newly_discovered: set[int] = set()

    for walker in walkers:
        if walker.reached_target and walker.lineage_id not in discovered_lineages:
            newly_discovered.add(walker.lineage_id)

    return newly_discovered


def summarize_we_population(
    walkers: list[WEWalker],
    weights: np.ndarray,
    sim_cfg: Toy2DConfig,
    scores: np.ndarray | None,
    bin_ids: np.ndarray,
    iteration: int,
    method: str,
    newly_discovered: set[int],
    discovered_lineages: set[int],
) -> dict[str, float]:
    """
    Summarize current weighted ensemble population.
    """
    xs = np.array([w.x for w in walkers], dtype=float)
    ys = np.array([w.y for w in walkers], dtype=float)

    distances = distance_to_target(xs, ys, sim_cfg.target_center)
    reached = np.array([w.reached_target for w in walkers], dtype=bool)

    weights = weights.astype(np.float64)

    reached_weight = float(weights[reached].sum()) if reached.any() else 0.0

    summary = {
        "iteration": iteration,
        "method": method,
        "num_walkers": len(walkers),
        "num_bins": int(len(np.unique(bin_ids))),
        "total_weight": float(weights.sum()),
        "population_reached": int(reached.sum()),
        "fraction_reached": float(reached.mean()),
        "target_weight": reached_weight,
        "new_reached_this_iteration": int(len(newly_discovered)),
        "cumulative_unique_reached": int(len(discovered_lineages)),
        "mean_distance": float(distances.mean()),
        "min_distance": float(distances.min()),
    }

    if scores is not None:
        summary["mean_score"] = float(np.mean(scores))
        summary["max_score"] = float(np.max(scores))
    else:
        summary["mean_score"] = float("nan")
        summary["max_score"] = float("nan")

    return summary


def run_we_toy_experiment(
    sim_cfg: Toy2DConfig,
    we_cfg: WEToyConfig,
    method: str,
    checkpoint_path: str | Path = "checkpoints/toy_gru.pt",
) -> list[dict[str, float]]:
    """
    Run toy weighted ensemble simulation.

    method:
        "static" = hand-designed distance bins
        "model" = trained GRU score bins
    """
    if method not in {"static", "model", "hybrid"}:
        raise ValueError(f"Unknown method: {method}")

    rng = np.random.default_rng(we_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: GRURanker | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    if method in {"model", "hybrid"}:
        model, mean, std = load_gru_model(checkpoint_path, device)

    walkers, weights = initialize_we_walkers(sim_cfg, we_cfg, rng)

    discovered_lineages: set[int] = set()
    history: list[dict[str, float]] = []

    for iteration in range(1, we_cfg.num_iterations + 1):
        run_we_segment(walkers, sim_cfg, we_cfg, rng)

        if method in {"model", "hybrid"}:
            assert model is not None
            assert mean is not None
            assert std is not None

            scores = score_we_walkers(
                walkers=walkers,
                model=model,
                mean=mean,
                std=std,
                device=device,
                window_size=we_cfg.window_size,
            )

            if method == "model":
                bin_ids = assign_model_score_bins(
                    scores=scores,
                    num_bins=we_cfg.num_bins,
                )
            else:
                bin_ids = assign_hybrid_score_distance_bins(
                    walkers=walkers,
                    sim_cfg=sim_cfg,
                    scores=scores,
                    num_score_bins=we_cfg.num_score_bins,
                    num_distance_bins=we_cfg.num_distance_bins,
                )

        else:
            scores = None
            bin_ids = assign_static_distance_bins(
                walkers=walkers,
                sim_cfg=sim_cfg,
                num_bins=we_cfg.num_bins,
            )

        newly_discovered = find_new_lineages(walkers, discovered_lineages)
        discovered_lineages.update(newly_discovered)

        summary = summarize_we_population(
            walkers=walkers,
            weights=weights,
            sim_cfg=sim_cfg,
            scores=scores,
            bin_ids=bin_ids,
            iteration=iteration,
            method=method,
            newly_discovered=newly_discovered,
            discovered_lineages=discovered_lineages,
        )
        history.append(summary)

        if scores is not None:
            bin_priorities = compute_score_based_bin_priorities(
                bin_ids=bin_ids,
                scores=scores,
                alpha=we_cfg.priority_alpha,
            )
        else:
            bin_priorities = None

        walkers, weights, _ = weighted_ensemble_resample(
            walkers=walkers,
            weights=weights,
            bin_ids=bin_ids,
            target_per_bin=we_cfg.target_per_bin,
            rng=rng,
            target_total_count=we_cfg.num_walkers,
            bin_priorities=bin_priorities,
            min_count_per_bin=we_cfg.min_count_per_bin,
        )

    return history


def run_we_comparison(
    sim_cfg: Toy2DConfig,
    we_cfg: WEToyConfig,
    checkpoint_path: str | Path = "checkpoints/toy_gru.pt",
) -> dict[str, list[dict[str, float]]]:
    """
    Compare static distance bins, model-score bins, and hybrid bins.
    """
    static_cfg = WEToyConfig(**we_cfg.__dict__)
    model_cfg = WEToyConfig(**we_cfg.__dict__)
    hybrid_cfg = WEToyConfig(**we_cfg.__dict__)

    static_cfg.seed = we_cfg.seed
    model_cfg.seed = we_cfg.seed
    hybrid_cfg.seed = we_cfg.seed

    # Static/model use 8 bins × 32 walkers/bin = ~256 walkers.
    static_cfg.num_bins = 8
    model_cfg.num_bins = 8
    static_cfg.target_per_bin = 32
    model_cfg.target_per_bin = 32

    # Hybrid uses 4 score bins × 4 distance bins = 16 bins.
    # Use 16 walkers/bin to keep total around 256.
    hybrid_cfg.num_score_bins = 4
    hybrid_cfg.num_distance_bins = 4
    hybrid_cfg.target_per_bin = 32

    static_history = run_we_toy_experiment(
        sim_cfg=sim_cfg,
        we_cfg=static_cfg,
        method="static",
        checkpoint_path=checkpoint_path,
    )

    model_history = run_we_toy_experiment(
        sim_cfg=sim_cfg,
        we_cfg=model_cfg,
        method="model",
        checkpoint_path=checkpoint_path,
    )

    hybrid_history = run_we_toy_experiment(
        sim_cfg=sim_cfg,
        we_cfg=hybrid_cfg,
        method="hybrid",
        checkpoint_path=checkpoint_path,
    )

    return {
        "static": static_history,
        "model": model_history,
        "hybrid": hybrid_history,
    }