from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stride.training.evaluation import evaluate_rankers, write_evaluation_report


def pcoord_baseline_rankers(
    pcoord_windows: np.ndarray,
    window_mask: np.ndarray,
    target: float | np.ndarray | None = None,
    pcoord_dim: int | np.ndarray = 0,
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

    num_examples = pcoord_windows.shape[0]
    pcoord_dims = _as_per_example_int(pcoord_dim, num_examples, "pcoord_dim")
    targets = None
    if target is not None:
        targets = _as_per_example_float(target, num_examples, "target")

    last_values = np.empty((num_examples,), dtype=np.float32)
    min_values = np.empty((num_examples,), dtype=np.float32)
    target_last = np.empty((num_examples,), dtype=np.float32)
    target_min = np.empty((num_examples,), dtype=np.float32)
    for example_index in range(pcoord_windows.shape[0]):
        valid = np.flatnonzero(window_mask[example_index])
        if len(valid) == 0:
            raise ValueError("Every pcoord window must contain at least one valid frame.")
        dim = int(pcoord_dims[example_index])
        if dim < 0 or dim >= pcoord_windows.shape[-1]:
            raise ValueError(f"pcoord_dim {dim} out of range for example {example_index}.")
        values = pcoord_windows[example_index, valid, dim]
        last_values[example_index] = values[-1]
        min_values[example_index] = float(np.min(values))
        if targets is not None:
            target_value = float(targets[example_index])
            target_last[example_index] = -abs(float(values[-1]) - target_value)
            target_min[example_index] = -float(np.min(np.abs(values - target_value)))

    rankers = {
        "last_pcoord_low": -last_values,
        "window_min_pcoord_low": -min_values,
    }
    if targets is not None:
        rankers["last_pcoord_target_proximity"] = target_last
        rankers["window_min_target_proximity"] = target_min
    return rankers


def westpa_iteration_split_indices(
    n_iter: np.ndarray,
    validation_fraction: float = 0.2,
    split_strategy: str = "tail",
    seed: int = 7,
    goal_id: np.ndarray | None = None,
    cell_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split examples by whole WESTPA iterations to avoid iteration leakage.
    """
    n_iter = np.asarray(n_iter, dtype=np.int64)
    if split_strategy == "heldout_goal":
        if goal_id is None:
            raise ValueError("heldout_goal split requires goal_id metadata.")
        return _metadata_split_indices(goal_id, validation_fraction, seed, random=False)
    if split_strategy == "heldout_cell":
        if cell_id is None:
            raise ValueError("heldout_cell split requires cell_id metadata.")
        return _metadata_split_indices(cell_id, validation_fraction, seed, random=False)
    if split_strategy == "random_goal":
        if goal_id is None:
            raise ValueError("random_goal split requires goal_id metadata.")
        return _metadata_split_indices(goal_id, validation_fraction, seed, random=True)
    if split_strategy == "random_cell":
        if cell_id is None:
            raise ValueError("random_cell split requires cell_id metadata.")
        return _metadata_split_indices(cell_id, validation_fraction, seed, random=True)

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
        raise ValueError(
            "split_strategy must be 'tail', 'random_block', 'heldout_goal', "
            "'heldout_cell', 'random_goal', or 'random_cell'."
        )

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
    pcoord_dim_values = data["pcoord_dim"].astype(np.int64) if "pcoord_dim" in data else pcoord_dim
    target_values = data["threshold"].astype(np.float32) if "threshold" in data and pcoord_target is None else pcoord_target
    rankers = pcoord_baseline_rankers(
        data["pcoord_windows"],
        data["window_mask"],
        target=target_values,
        pcoord_dim=pcoord_dim_values,
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
            goal_id=data["goal_id"].astype(str) if "goal_id" in data else None,
            cell_id=data["cell_id"].astype(str) if "cell_id" in data else None,
        )
        if eval_split == "train":
            indices = train_indices
        elif eval_split == "validation":
            indices = val_indices
        else:
            raise ValueError("eval_split must be 'all', 'train', or 'validation'.")

    paths = write_evaluation_report(
        output_dir=output_dir,
        y_true=labels[indices],
        rankers={name: scores[indices] for name, scores in rankers.items()},
        dataset_name=f"{lineage_npz} [{eval_split}]",
        checkpoint_name="WESTPA lineage artifact",
    )
    paths.update(
        _write_grouped_reports(
            output_dir=Path(output_dir),
            labels=labels,
            rankers=rankers,
            indices=indices,
            data=data,
        )
    )
    return paths


def _metadata_split_indices(
    values: np.ndarray,
    validation_fraction: float,
    seed: int,
    random: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values).astype(str)
    unique_values = np.unique(values)
    if unique_values.size < 2:
        raise ValueError("Metadata split requires at least two unique values.")
    val_count = int(round(unique_values.size * validation_fraction))
    val_count = min(max(val_count, 1), unique_values.size - 1)
    if random:
        rng = np.random.default_rng(seed)
        val_values = rng.choice(unique_values, size=val_count, replace=False)
    else:
        val_values = unique_values[-val_count:]
    val_mask = np.isin(values, val_values)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def _write_grouped_reports(
    output_dir: Path,
    labels: np.ndarray,
    rankers: dict[str, np.ndarray],
    indices: np.ndarray,
    data: np.lib.npyio.NpzFile,
) -> dict[str, Path]:
    group_fields = [field for field in ("goal_id", "cell_id") if field in data.files]
    if not group_fields:
        return {}

    metric_rows: list[dict[str, float | str]] = []
    for field in group_fields:
        values = data[field].astype(str)
        for group in np.unique(values[indices]):
            group_indices = indices[values[indices] == group]
            rows, _, _ = evaluate_rankers(
                labels[group_indices],
                {name: scores[group_indices] for name, scores in rankers.items()},
            )
            for row in rows:
                metric_rows.append({"group_type": field, "group_id": group, **row})

    grouped_metrics = output_dir / "grouped_metrics.csv"
    grouped_markdown = output_dir / "grouped_report.md"
    _write_csv(grouped_metrics, metric_rows)
    _write_grouped_markdown(grouped_markdown, metric_rows)
    return {"grouped_metrics": grouped_metrics, "grouped_markdown": grouped_markdown}


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_grouped_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines = ["# STRIDE Grouped Evaluation", ""]
    for group_type in sorted({str(row["group_type"]) for row in rows}):
        lines.extend([f"## {group_type}", ""])
        group_ids = sorted({str(row["group_id"]) for row in rows if row["group_type"] == group_type})
        for group_id in group_ids:
            lines.extend([f"### {group_id}", ""])
            lines.extend(
                [
                    "| Ranker | AUROC | AUPRC | Top 5% | Top 25% |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in rows:
                if row["group_type"] != group_type or row["group_id"] != group_id:
                    continue
                lines.append(
                    "| {ranker} | {auroc:.4g} | {auprc:.4g} | {top5:.4g} | {top25:.4g} |".format(
                        ranker=row["ranker"],
                        auroc=float(row.get("auroc", float("nan"))),
                        auprc=float(row.get("auprc", float("nan"))),
                        top5=float(row.get("top5_enrichment", float("nan"))),
                        top25=float(row.get("top25_enrichment", float("nan"))),
                    )
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _as_per_example_int(value: int | np.ndarray, num_examples: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim == 0:
        return np.full((num_examples,), int(array), dtype=np.int64)
    if array.shape != (num_examples,):
        raise ValueError(f"{name} must be scalar or shape [examples].")
    return array


def _as_per_example_float(value: float | np.ndarray, num_examples: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full((num_examples,), float(array), dtype=np.float32)
    if array.shape != (num_examples,):
        raise ValueError(f"{name} must be scalar or shape [examples].")
    return array
