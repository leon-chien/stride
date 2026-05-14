from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from stride.goals import GoalSpec
from stride.westpa_plugin.h5_reader import build_lineage_windows, load_segment_records


@dataclass(frozen=True)
class MultiGoalBuildReport:
    benchmark_name: str
    num_cells: int
    num_goals: int
    num_examples: int
    positive_rate: float
    flux_sum: float


def build_multigoal_lineage_dataset_from_yaml(
    benchmark_yaml: str | Path,
    output_npz: str | Path,
) -> MultiGoalBuildReport:
    """
    Build one pcoord-lineage artifact from many WESTPA cells and goal specs.
    """
    config_path = Path(benchmark_yaml)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    benchmark = config.get("benchmark", config)
    base_dir = config_path.parent

    name = str(benchmark.get("name", config_path.stem))
    window_iterations = int(benchmark["window_iterations"])
    pcoord_frame_index = int(benchmark.get("pcoord_frame_index", -1))
    require_full_window = bool(benchmark.get("require_full_window", True))
    include_self_in_label = bool(benchmark.get("include_self_in_label", False))
    cells = _require_sequence(benchmark, "cells")
    goals = _require_sequence(benchmark, "goals")

    pcoord_windows: list[np.ndarray] = []
    window_masks: list[np.ndarray] = []
    goal_features: list[np.ndarray] = []
    event_labels: list[float] = []
    flux_labels: list[float] = []
    weights: list[float] = []
    n_iters: list[int] = []
    seg_ids: list[int] = []
    cell_ids: list[str] = []
    goal_ids: list[str] = []
    pcoord_dims: list[int] = []
    thresholds: list[float] = []
    horizons: list[int] = []

    expected_pcoord_dim: tuple[int, ...] | None = None
    for cell in cells:
        cell_id = str(cell["cell_id"])
        h5_path = _resolve_path(base_dir, cell["west_h5"])
        records = load_segment_records(h5_path)
        for goal_entry in goals:
            goal_id = str(goal_entry["goal_id"])
            goal = _goal_from_entry(goal_entry)
            pcoord_dim = int(goal_entry.get("pcoord_dim", benchmark.get("pcoord_dim", 0)))
            horizon = int(goal_entry.get("horizon_iterations", goal.horizon_iterations))
            windows = build_lineage_windows(
                records=records,
                goal=goal,
                window_iterations=window_iterations,
                horizon_iterations=horizon,
                pcoord_frame_index=pcoord_frame_index,
                pcoord_dim=pcoord_dim,
                require_full_window=require_full_window,
                include_self_in_label=include_self_in_label,
            )
            for window in windows:
                if expected_pcoord_dim is None:
                    expected_pcoord_dim = window.pcoord_window.shape[1:]
                elif window.pcoord_window.shape[1:] != expected_pcoord_dim:
                    raise ValueError(
                        "All pcoord windows must have the same feature shape: "
                        f"{window.pcoord_window.shape[1:]} != {expected_pcoord_dim}"
                    )
                padded = np.zeros(
                    (window_iterations, *window.pcoord_window.shape[1:]),
                    dtype=np.float32,
                )
                mask = np.zeros((window_iterations,), dtype=bool)
                length = window.pcoord_window.shape[0]
                padded[-length:] = window.pcoord_window
                mask[-length:] = True

                pcoord_windows.append(padded)
                window_masks.append(mask)
                goal_features.append(window.goal_features)
                event_labels.append(float(window.event))
                flux_labels.append(float(window.flux))
                weights.append(float(window.weight))
                n_iters.append(int(window.key.n_iter))
                seg_ids.append(int(window.key.seg_id))
                cell_ids.append(cell_id)
                goal_ids.append(goal_id)
                pcoord_dims.append(pcoord_dim)
                thresholds.append(float(goal.threshold))
                horizons.append(horizon)

    if not pcoord_windows:
        raise ValueError("No multi-goal lineage examples were built.")

    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pcoord_windows=np.stack(pcoord_windows).astype(np.float32),
        window_mask=np.stack(window_masks).astype(bool),
        goal_features=np.stack(goal_features).astype(np.float32),
        event_labels=np.asarray(event_labels, dtype=np.float32),
        flux_labels=np.asarray(flux_labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float64),
        n_iter=np.asarray(n_iters, dtype=np.int64),
        seg_id=np.asarray(seg_ids, dtype=np.int64),
        cell_id=np.asarray(cell_ids),
        goal_id=np.asarray(goal_ids),
        pcoord_dim=np.asarray(pcoord_dims, dtype=np.int64),
        threshold=np.asarray(thresholds, dtype=np.float32),
        horizon_iterations=np.asarray(horizons, dtype=np.int64),
    )

    labels = np.asarray(event_labels, dtype=np.float32)
    return MultiGoalBuildReport(
        benchmark_name=name,
        num_cells=len(cells),
        num_goals=len(goals),
        num_examples=int(labels.shape[0]),
        positive_rate=float(np.mean(labels)),
        flux_sum=float(np.sum(flux_labels)),
    )


def _goal_from_entry(entry: dict[str, Any]) -> GoalSpec:
    if "goal" in entry:
        return GoalSpec.from_dict(entry["goal"])
    goal_id = str(entry["goal_id"])
    threshold = float(entry["threshold"])
    horizon = int(entry["horizon_iterations"])
    return GoalSpec.from_dict(
        {
            "name": goal_id,
            "type": str(entry.get("type", "distance_threshold")),
            "selections": tuple(entry.get("selections", ("pcoord",))),
            "operator": str(entry.get("operator", "less_than")),
            "threshold": threshold,
            "horizon_iterations": horizon,
            "value_target": str(entry.get("value_target", "event_and_flux")),
        }
    )


def _require_sequence(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Benchmark config requires non-empty list: {key}")
    return value


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
