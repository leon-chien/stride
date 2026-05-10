from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NaClConfig:
    """
    Synthetic Na+/Cl- association benchmark configuration.

    This is not yet full molecular dynamics. It is a reduced-distance
    benchmark that mimics Na-Cl association as a 1D stochastic distance
    process.

    Version 2 uses this first so STRIDE can build the NaCl training and
    learned-binning pipeline before reading actual WESTPA output.
    """

    name: str = "nacl_association"
    
    num_trajectories: int = 1000
    trajectory_length: int = 400
    dt: float = 0.02
    noise_std: float = 0.08
    drift_strength: float = 0.35

    start_distance_mean: float = 1.25
    start_distance_std: float = 0.08
    min_distance: float = 0.20
    max_distance: float = 1.60

    association_distance: float = 0.35

    event_type: str = "association"
    seed: int = 42


def association_event(
    distances: float | np.ndarray,
    association_distance: float,
) -> bool | np.ndarray:
    """
    Association occurs when Na-Cl distance is below threshold.
    """
    return distances < association_distance


def synthetic_distance_force(
    distance: float,
    cfg: NaClConfig,
) -> float:
    """
    Simple reduced dynamics force for NaCl-like association.

    The force gently biases distances toward smaller values, while noise
    allows stochastic association/dissociation-like behavior.

    This is intentionally simple. It is only a Version 2 bridge benchmark.
    """
    center = cfg.association_distance
    force = -cfg.drift_strength * (distance - center)

    return float(force)


def simulate_nacl_distance_trajectory(
    cfg: NaClConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one synthetic Na-Cl distance trajectory.

    Returns:
        distances: shape [trajectory_length]
    """
    distances = np.zeros(cfg.trajectory_length, dtype=np.float32)

    d = float(rng.normal(cfg.start_distance_mean, cfg.start_distance_std))
    d = float(np.clip(d, cfg.min_distance, cfg.max_distance))

    for t in range(cfg.trajectory_length):
        f = synthetic_distance_force(d, cfg)

        d = (
            d
            + cfg.dt * f
            + cfg.noise_std * np.sqrt(cfg.dt) * rng.normal()
        )

        # Reflect/clip into physically reasonable range.
        d = float(np.clip(d, cfg.min_distance, cfg.max_distance))

        distances[t] = d

    return distances


def simulate_nacl_dataset(cfg: NaClConfig) -> np.ndarray:
    """
    Simulate many synthetic NaCl-like distance trajectories.

    Returns:
        distances: shape [num_trajectories, trajectory_length]
    """
    rng = np.random.default_rng(cfg.seed)

    trajectories = np.zeros(
        (cfg.num_trajectories, cfg.trajectory_length),
        dtype=np.float32,
    )

    for i in range(cfg.num_trajectories):
        trajectories[i] = simulate_nacl_distance_trajectory(cfg, rng)

    return trajectories


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute simple moving average with edge padding.
    """
    if window <= 1:
        return values.astype(np.float32)

    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window

    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_nacl_features(
    distances: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Convert a distance trajectory into frame-level features.

    Features:
        0: distance
        1: delta_distance
        2: radial_velocity
        3: inverse_distance
        4: moving_average_distance

    Args:
        distances: shape [trajectory_length]
        dt: simulation time step

    Returns:
        features: shape [trajectory_length, 5]
    """
    distances = np.asarray(distances, dtype=np.float32)

    delta = np.zeros_like(distances, dtype=np.float32)
    delta[1:] = distances[1:] - distances[:-1]

    radial_velocity = delta / dt

    inverse_distance = 1.0 / np.clip(distances, 1e-6, None)

    ma_distance = moving_average(distances, window=5)

    features = np.stack(
        [
            distances,
            delta,
            radial_velocity,
            inverse_distance,
            ma_distance,
        ],
        axis=-1,
    ).astype(np.float32)

    return features


def compute_feature_dataset(
    distance_trajectories: np.ndarray,
    cfg: NaClConfig,
) -> np.ndarray:
    """
    Convert all distance trajectories into feature trajectories.

    Args:
        distance_trajectories: shape [num_trajectories, trajectory_length]

    Returns:
        feature_trajectories: shape [num_trajectories, trajectory_length, 5]
    """
    feature_trajectories = []

    for distances in distance_trajectories:
        features = compute_nacl_features(distances, dt=cfg.dt)
        feature_trajectories.append(features)

    return np.stack(feature_trajectories).astype(np.float32)


def trajectory_reached_association(
    distances: np.ndarray,
    cfg: NaClConfig,
) -> bool:
    """
    Whether a full trajectory ever reaches association.
    """
    reached = association_event(
        distances,
        association_distance=cfg.association_distance,
    )

    return bool(np.any(reached))


def event_rate(
    distance_trajectories: np.ndarray,
    cfg: NaClConfig,
) -> float:
    """
    Fraction of trajectories that ever reach association.
    """
    events = [
        trajectory_reached_association(distances, cfg)
        for distances in distance_trajectories
    ]

    return float(np.mean(events))


def build_nacl_windows(
    feature_trajectories: np.ndarray,
    distance_trajectories: np.ndarray,
    cfg: NaClConfig,
    window_size: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert NaCl trajectories into delayed-label training examples.

    X = recent feature window
    y = whether association occurs in the future horizon
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []

    num_traj, traj_len, _ = feature_trajectories.shape

    for i in range(num_traj):
        features = feature_trajectories[i]
        distances = distance_trajectories[i]

        for start in range(0, traj_len - window_size - horizon, stride):
            end = start + window_size
            future_end = end + horizon

            window = features[start:end]

            future_distances = distances[end:future_end]

            future_reached = association_event(
                future_distances,
                association_distance=cfg.association_distance,
            )

            label = int(np.any(future_reached))

            xs.append(window)
            ys.append(label)

    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)

    return X, y


def save_nacl_dataset(
    output_dir: str | Path,
    distance_trajectories: np.ndarray,
    feature_trajectories: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Save synthetic NaCl dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "nacl_dataset.npz",
        distance_trajectories=distance_trajectories,
        feature_trajectories=feature_trajectories,
        X=X,
        y=y,
    )