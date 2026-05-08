from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Toy2DConfig:
    num_trajectories: int = 1000
    trajectory_length: int = 200
    dt: float = 0.02
    noise_std: float = 0.25
    start_center: tuple[float, float] = (-1.0, 0.0)
    start_std: float = 0.08
    target_center: tuple[float, float] = (1.0, 0.0)
    target_radius: float = 0.25
    seed: int = 42


def potential(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """
    Double-well potential.

    Left basin: x ≈ -1
    Right basin: x ≈ +1
    """
    return (x**2 - 1.0) ** 2 + 0.5 * y**2


def force(x: float, y: float) -> tuple[float, float]:
    """
    Force = -gradient of potential.

    U(x, y) = (x^2 - 1)^2 + 0.5 y^2
    dU/dx = 4x(x^2 - 1)
    dU/dy = y
    """
    fx = -4.0 * x * (x**2 - 1.0)
    fy = -y
    return fx, fy


def distance_to_target(
    x: float | np.ndarray,
    y: float | np.ndarray,
    target_center: tuple[float, float],
) -> float | np.ndarray:
    tx, ty = target_center
    return np.sqrt((x - tx) ** 2 + (y - ty) ** 2)


def reaches_target(
    x: float | np.ndarray,
    y: float | np.ndarray,
    target_center: tuple[float, float],
    target_radius: float,
) -> bool | np.ndarray:
    return distance_to_target(x, y, target_center) < target_radius


def simulate_trajectory(
    cfg: Toy2DConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one noisy 2D trajectory.

    Returns:
        trajectory: shape [trajectory_length, 6]

    Features:
        0: x
        1: y
        2: vx
        3: vy
        4: potential energy
        5: distance to target
    """
    length = cfg.trajectory_length
    traj = np.zeros((length, 6), dtype=np.float32)

    x = rng.normal(cfg.start_center[0], cfg.start_std)
    y = rng.normal(cfg.start_center[1], cfg.start_std)

    prev_x, prev_y = x, y

    for t in range(length):
        fx, fy = force(x, y)

        # Overdamped Langevin-style update
        x = x + cfg.dt * fx + cfg.noise_std * np.sqrt(cfg.dt) * rng.normal()
        y = y + cfg.dt * fy + cfg.noise_std * np.sqrt(cfg.dt) * rng.normal()

        vx = (x - prev_x) / cfg.dt
        vy = (y - prev_y) / cfg.dt

        u = potential(x, y)
        d_target = distance_to_target(x, y, cfg.target_center)

        traj[t] = np.array([x, y, vx, vy, u, d_target], dtype=np.float32)

        prev_x, prev_y = x, y

    return traj


def simulate_dataset(cfg: Toy2DConfig) -> np.ndarray:
    """
    Simulate many independent trajectories.

    Returns:
        trajectories: shape [num_trajectories, trajectory_length, 6]
    """
    rng = np.random.default_rng(cfg.seed)

    trajectories = np.zeros(
        (cfg.num_trajectories, cfg.trajectory_length, 6),
        dtype=np.float32,
    )

    for i in range(cfg.num_trajectories):
        trajectories[i] = simulate_trajectory(cfg, rng)

    return trajectories


def trajectory_reached_target(
    trajectory: np.ndarray,
    target_center: tuple[float, float],
    target_radius: float,
) -> bool:
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    reached = reaches_target(x, y, target_center, target_radius)
    return bool(np.any(reached))


def event_rate(
    trajectories: np.ndarray,
    target_center: tuple[float, float],
    target_radius: float,
) -> float:
    events = [
        trajectory_reached_target(traj, target_center, target_radius)
        for traj in trajectories
    ]
    return float(np.mean(events))


def build_windows(
    trajectories: np.ndarray,
    target_center: tuple[float, float],
    target_radius: float,
    window_size: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert trajectories into delayed-label supervised examples.

    X = recent trajectory window
    y = whether target is reached within the next horizon frames
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []

    num_traj, traj_len, _ = trajectories.shape

    for i in range(num_traj):
        traj = trajectories[i]

        for start in range(0, traj_len - window_size - horizon, stride):
            end = start + window_size
            future_end = end + horizon

            window = traj[start:end]

            future = traj[end:future_end]
            future_x = future[:, 0]
            future_y = future[:, 1]

            future_reached = reaches_target(
                future_x,
                future_y,
                target_center,
                target_radius,
            )

            label = int(np.any(future_reached))

            xs.append(window)
            ys.append(label)

    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)

    return X, y


def save_dataset(
    output_dir: str | Path,
    trajectories: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "toy2d_dataset.npz",
        trajectories=trajectories,
        X=X,
        y=y,
    )

