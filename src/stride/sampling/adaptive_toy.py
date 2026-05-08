from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stride.features.toy2d import (
    Toy2DConfig,
    distance_to_target,
    force,
    potential,
    reaches_target,
)
from stride.models.gru_ranker import GRURanker


@dataclass
class WalkerState:
    """
    State for one toy 2D walker.

    lineage_id tracks the original starting walker.
    When a walker is cloned during resampling, its children keep
    the same lineage_id. This lets us count unique discoveries
    without being fooled by cloning.
    """

    x: float
    y: float
    prev_x: float
    prev_y: float
    history: list[np.ndarray]
    lineage_id: int
    reached_target: bool = False


@dataclass
class AdaptiveToyConfig:
    """
    Configuration for adaptive toy resampling.
    """

    num_walkers: int = 256
    num_iterations: int = 80
    segment_length: int = 5
    window_size: int = 25
    resample_temperature: float = 1.0
    seed: int = 123


def load_gru_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[GRURanker, np.ndarray, np.ndarray]:
    """
    Load trained toy GRU model and normalization stats.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run `python src/stride/training/train_toy.py` first."
        )

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

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    return model, mean, std


def make_feature_vector(
    x: float,
    y: float,
    prev_x: float,
    prev_y: float,
    dt: float,
    target_center: tuple[float, float],
) -> np.ndarray:
    """
    Convert current toy state into the same 6-feature representation
    used during training.

    Features:
        0: x
        1: y
        2: vx
        3: vy
        4: potential energy
        5: distance to target
    """
    vx = (x - prev_x) / dt
    vy = (y - prev_y) / dt
    u = potential(x, y)
    d_target = distance_to_target(x, y, target_center)

    return np.array([x, y, vx, vy, u, d_target], dtype=np.float32)


def initialize_walkers(
    sim_cfg: Toy2DConfig,
    adaptive_cfg: AdaptiveToyConfig,
    rng: np.random.Generator,
) -> list[WalkerState]:
    """
    Initialize walkers near the starting basin.
    """
    walkers: list[WalkerState] = []

    for _ in range(adaptive_cfg.num_walkers):
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
            WalkerState(
                x=x,
                y=y,
                prev_x=x,
                prev_y=y,
                history=[feature],
                lineage_id=len(walkers),
                reached_target=False,
            )
        )

    return walkers


def step_walker(
    walker: WalkerState,
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


def run_segment(
    walkers: list[WalkerState],
    sim_cfg: Toy2DConfig,
    adaptive_cfg: AdaptiveToyConfig,
    rng: np.random.Generator,
) -> None:
    """
    Run each walker for segment_length dynamics steps.
    """
    for _ in range(adaptive_cfg.segment_length):
        for walker in walkers:
            step_walker(walker, sim_cfg, rng)


def walkers_to_windows(
    walkers: list[WalkerState],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert current walker histories into model windows.

    Returns:
        X: [num_ready_walkers, window_size, num_features]
        ready_indices: original walker indices corresponding to X
    """
    windows: list[np.ndarray] = []
    ready_indices: list[int] = []

    for i, walker in enumerate(walkers):
        if len(walker.history) >= window_size:
            window = np.stack(walker.history[-window_size:]).astype(np.float32)
            windows.append(window)
            ready_indices.append(i)

    if not windows:
        return np.empty((0, window_size, 6), dtype=np.float32), np.array([], dtype=int)

    X = np.stack(windows).astype(np.float32)
    indices = np.array(ready_indices, dtype=int)

    return X, indices


def score_walkers(
    walkers: list[WalkerState],
    model: GRURanker,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    window_size: int,
) -> np.ndarray:
    """
    Score walkers by predicted rare-event probability.

    Walkers that do not yet have enough history receive score 1.0
    so early warmup does not accidentally eliminate them.
    """
    scores = np.ones(len(walkers), dtype=np.float32)

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


def clone_walker(walker: WalkerState) -> WalkerState:
    """
    Clone a walker during resampling.

    The history is copied so each child can evolve independently afterward.
    """
    return WalkerState(
        x=walker.x,
        y=walker.y,
        prev_x=walker.prev_x,
        prev_y=walker.prev_y,
        history=[frame.copy() for frame in walker.history],
        lineage_id=walker.lineage_id,
        reached_target=walker.reached_target,
    )


def resample_walkers(
    walkers: list[WalkerState],
    scores: np.ndarray,
    method: str,
    adaptive_cfg: AdaptiveToyConfig,
    rng: np.random.Generator,
) -> list[WalkerState]:
    """
    Resample walkers to keep population size fixed.

    method:
        "random" = uniform parent selection
        "model" = score-weighted parent selection
    """
    n = adaptive_cfg.num_walkers

    if method == "random":
        probs = np.ones(len(walkers), dtype=np.float64) / len(walkers)

    elif method == "model":
        # Convert model scores into sampling probabilities.
        # Small epsilon prevents zero-probability walkers.
        adjusted = np.asarray(scores, dtype=np.float64) + 1e-6

        # Temperature controls selection sharpness.
        # Lower = more aggressive exploitation.
        adjusted = adjusted ** (1.0 / adaptive_cfg.resample_temperature)

        probs = adjusted / adjusted.sum()

    else:
        raise ValueError(f"Unknown resampling method: {method}")

    parent_indices = rng.choice(
        len(walkers),
        size=n,
        replace=True,
        p=probs,
    )

    return [clone_walker(walkers[i]) for i in parent_indices]


def summarize_population(
    walkers: list[WalkerState],
    sim_cfg: Toy2DConfig,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute population summary statistics.
    """
    xs = np.array([w.x for w in walkers], dtype=float)
    ys = np.array([w.y for w in walkers], dtype=float)

    distances = distance_to_target(xs, ys, sim_cfg.target_center)
    reached = np.array([w.reached_target for w in walkers], dtype=bool)

    summary = {
        "mean_x": float(xs.mean()),
        "mean_y": float(ys.mean()),
        "mean_distance": float(distances.mean()),
        "min_distance": float(distances.min()),
        "fraction_reached": float(reached.mean()),
        "num_reached": int(reached.sum()),
    }

    if scores is not None:
        summary["mean_score"] = float(np.mean(scores))
        summary["max_score"] = float(np.max(scores))

    return summary


def find_newly_discovered_lineages(
    walkers: list[WalkerState],
    discovered_lineages: set[int],
) -> set[int]:
    """
    Find unique lineages that reached the target for the first time.

    This avoids counting cloned descendants as new independent discoveries.
    """
    newly_discovered: set[int] = set()

    for walker in walkers:
        if walker.reached_target and walker.lineage_id not in discovered_lineages:
            newly_discovered.add(walker.lineage_id)

    return newly_discovered


def run_adaptive_toy_experiment(
    sim_cfg: Toy2DConfig,
    adaptive_cfg: AdaptiveToyConfig,
    method: str,
    checkpoint_path: str | Path = "checkpoints/toy_gru.pt",
) -> list[dict[str, float]]:
    """
    Run one toy adaptive sampling experiment.

    method:
        "random" = random resampling baseline
        "model" = GRU-guided adaptive resampling

    Returns:
        List of per-iteration summaries.
    """
    rng = np.random.default_rng(adaptive_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: GRURanker | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    if method == "model":
        model, mean, std = load_gru_model(checkpoint_path, device)

    walkers = initialize_walkers(sim_cfg, adaptive_cfg, rng)

    history: list[dict[str, float]] = []
    discovered_lineages: set[int] = set()
    
    for iteration in range(1, adaptive_cfg.num_iterations + 1):
        run_segment(walkers, sim_cfg, adaptive_cfg, rng)

        if method == "model":
            assert model is not None
            assert mean is not None
            assert std is not None

            scores = score_walkers(
                walkers=walkers,
                model=model,
                mean=mean,
                std=std,
                device=device,
                window_size=adaptive_cfg.window_size,
            )
        else:
            scores = np.ones(len(walkers), dtype=np.float32)

        newly_discovered = find_newly_discovered_lineages(
            walkers=walkers,
            discovered_lineages=discovered_lineages,
        )
        discovered_lineages.update(newly_discovered)

        summary = summarize_population(walkers, sim_cfg, scores)
        summary["iteration"] = iteration
        summary["method"] = method
        summary["new_reached_this_iteration"] = len(newly_discovered)
        summary["cumulative_unique_reached"] = len(discovered_lineages)

        history.append(summary)

        walkers = resample_walkers(
            walkers=walkers,
            scores=scores,
            method=method,
            adaptive_cfg=adaptive_cfg,
            rng=rng,
        )

    return history


def run_comparison(
    sim_cfg: Toy2DConfig,
    adaptive_cfg: AdaptiveToyConfig,
    checkpoint_path: str | Path = "checkpoints/toy_gru.pt",
) -> dict[str, list[dict[str, float]]]:
    """
    Run random baseline and model-guided adaptive sampling.
    """
    random_cfg = AdaptiveToyConfig(**adaptive_cfg.__dict__)
    model_cfg = AdaptiveToyConfig(**adaptive_cfg.__dict__)

    random_cfg.seed = adaptive_cfg.seed
    model_cfg.seed = adaptive_cfg.seed

    random_history = run_adaptive_toy_experiment(
        sim_cfg=sim_cfg,
        adaptive_cfg=random_cfg,
        method="random",
        checkpoint_path=checkpoint_path,
    )

    model_history = run_adaptive_toy_experiment(
        sim_cfg=sim_cfg,
        adaptive_cfg=model_cfg,
        method="model",
        checkpoint_path=checkpoint_path,
    )

    return {
        "random": random_history,
        "model": model_history,
    }