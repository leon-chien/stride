from __future__ import annotations

from pathlib import Path

import numpy as np

from stride.training.evaluation import write_evaluation_report


def pcoord_baseline_rankers(
    pcoord_windows: np.ndarray,
    window_mask: np.ndarray,
    target: float | None = None,
    pcoord_dim: int = 0,
) -> dict[str, np.ndarray]:
    """
    Build simple WESTPA pcoord baselines for lineage-window artifacts.
    """
    pcoord_windows = np.asarray(pcoord_windows, dtype=np.float32)
    window_mask = np.asarray(window_mask, dtype=bool)
    if pcoord_windows.ndim != 3:
        raise ValueError("pcoord_windows must have shape [examples, window, dims].")
    if window_mask.shape != pcoord_windows.shape[:2]:
        raise ValueError("window_mask must match pcoord_windows first two dimensions.")

    last_values = np.empty((pcoord_windows.shape[0],), dtype=np.float32)
    min_values = np.empty((pcoord_windows.shape[0],), dtype=np.float32)
    for example_index in range(pcoord_windows.shape[0]):
        valid = np.flatnonzero(window_mask[example_index])
        if len(valid) == 0:
            raise ValueError("Every pcoord window must contain at least one valid frame.")
        values = pcoord_windows[example_index, valid, pcoord_dim]
        last_values[example_index] = values[-1]
        min_values[example_index] = float(np.min(values))

    rankers = {
        "last_pcoord_low": -last_values,
        "window_min_pcoord_low": -min_values,
    }
    if target is not None:
        rankers["last_pcoord_target_proximity"] = -np.abs(last_values - float(target))
        rankers["window_min_target_proximity"] = -np.min(
            np.abs(pcoord_windows[:, :, pcoord_dim] - float(target))
            + (~window_mask).astype(np.float32) * 1.0e6,
            axis=1,
        )
    return rankers


def westpa_iteration_split_indices(
    n_iter: np.ndarray,
    validation_fraction: float = 0.2,
    split_strategy: str = "tail",
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split examples by whole WESTPA iterations to avoid iteration leakage.
    """
    n_iter = np.asarray(n_iter, dtype=np.int64)
    unique_iters = np.unique(n_iter)
    if unique_iters.size < 3:
        return np.arange(len(n_iter), dtype=np.int64), np.empty((0,), dtype=np.int64)

    val_count = int(round(unique_iters.size * validation_fraction))
    val_count = min(max(val_count, 1), unique_iters.size - 1)
    if split_strategy == "tail":
        val_iters = unique_iters[-val_count:]
    elif split_strategy == "random_block":
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, unique_iters.size - val_count + 1))
        val_iters = unique_iters[start : start + val_count]
    else:
        raise ValueError("split_strategy must be 'tail' or 'random_block'.")

    val_mask = np.isin(n_iter, val_iters)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def write_westpa_lineage_report(
    lineage_npz: str | Path,
    output_dir: str | Path,
    stride_scores: np.ndarray | None = None,
    eval_split: str = "all",
    validation_fraction: float = 0.2,
    split_strategy: str = "tail",
    seed: int = 7,
    pcoord_target: float | None = None,
    pcoord_dim: int = 0,
) -> dict[str, Path]:
    """
    Generate an offline report for pcoord lineage-window WESTPA artifacts.
    """
    data = np.load(lineage_npz)
    labels = data["event_labels"].astype(np.float32)
    rankers = pcoord_baseline_rankers(
        data["pcoord_windows"],
        data["window_mask"],
        target=pcoord_target,
        pcoord_dim=pcoord_dim,
    )
    if stride_scores is not None:
        rankers = {"STRIDE": np.asarray(stride_scores, dtype=np.float32), **rankers}

    indices = np.arange(len(labels), dtype=np.int64)
    if eval_split != "all":
        train_indices, val_indices = westpa_iteration_split_indices(
            data["n_iter"],
            validation_fraction=validation_fraction,
            split_strategy=split_strategy,
            seed=seed,
        )
        if eval_split == "train":
            indices = train_indices
        elif eval_split == "validation":
            indices = val_indices
        else:
            raise ValueError("eval_split must be 'all', 'train', or 'validation'.")

    return write_evaluation_report(
        output_dir=output_dir,
        y_true=labels[indices],
        rankers={name: scores[indices] for name, scores in rankers.items()},
        dataset_name=f"{lineage_npz} [{eval_split}]",
        checkpoint_name="WESTPA lineage artifact",
    )
