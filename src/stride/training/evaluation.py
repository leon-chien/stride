from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np

from stride.data import AtomisticDataset, atom_selection_mask, compute_dihedral_degrees
from stride.goals import GoalSpec
from stride.training.metrics import (
    compute_binary_metrics,
    precision_recall_at_score_quantiles,
    top_k_enrichment,
    top_k_positive_rate,
)


TOP_K_FRACTIONS = (0.01, 0.05, 0.1, 0.25)
SCORE_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)


def dihedral_window_baseline_scores(
    dataset: AtomisticDataset,
    atom_indices: tuple[int, int, int, int],
    lower_bound: float,
    upper_bound: float,
    mode: str = "last_frame",
) -> np.ndarray:
    """
    Score examples by angular proximity to a target dihedral window.

    Higher is better. Inside the target window has score 0; outside examples
    are ranked by negative angular distance to the nearest window boundary.
    """
    dataset.validate()
    if mode not in {"last_frame", "window_min"}:
        raise ValueError("mode must be 'last_frame' or 'window_min'.")
    if len(atom_indices) != 4:
        raise ValueError("atom_indices must contain exactly four atom indices.")

    scores = np.empty((dataset.coordinates.shape[0],), dtype=np.float32)
    for example_index, window in enumerate(dataset.coordinates):
        frame_distances: list[float] = []
        frame_indices = range(window.shape[0])
        if mode == "last_frame":
            valid_frames = np.flatnonzero(dataset.frame_mask[example_index])
            frame_indices = [int(valid_frames[-1])] if len(valid_frames) else [window.shape[0] - 1]
        for frame_index in frame_indices:
            angle = compute_dihedral_degrees(window[frame_index, list(atom_indices), :])
            frame_distances.append(_distance_to_angle_window(angle, lower_bound, upper_bound))
        scores[example_index] = -float(min(frame_distances))
    return scores


def random_baseline_scores(num_examples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(num_examples, dtype=np.float32)


def evaluate_rankers(
    y_true: np.ndarray,
    rankers: dict[str, np.ndarray],
    top_k_fractions: tuple[float, ...] = TOP_K_FRACTIONS,
    score_quantiles: tuple[float, ...] = SCORE_QUANTILES,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], list[dict[str, float | str]]]:
    metric_rows: list[dict[str, float | str]] = []
    quantile_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for name, scores in rankers.items():
        scores = np.asarray(scores, dtype=float)
        metrics = compute_binary_metrics(y_true, scores)
        row: dict[str, float | str] = {"ranker": name, **metrics}
        for fraction in top_k_fractions:
            percent = int(round(fraction * 100))
            row[f"top{percent}_enrichment"] = top_k_enrichment(
                y_true,
                scores,
                k=fraction,
            )
            row[f"top{percent}_positive_rate"] = top_k_positive_rate(
                y_true,
                scores,
                k=fraction,
            )
        metric_rows.append(row)

        for quantile_row in precision_recall_at_score_quantiles(
            y_true,
            scores,
            quantiles=score_quantiles,
        ):
            quantile_rows.append({"ranker": name, **quantile_row})

        summary_rows.append({"ranker": name, **score_distribution_summary(scores)})

    return metric_rows, quantile_rows, summary_rows


def score_distribution_summary(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    quantiles = np.quantile(scores, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": float(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": float(np.max(scores)),
    }


def write_evaluation_report(
    output_dir: str | Path,
    y_true: np.ndarray,
    rankers: dict[str, np.ndarray],
    dataset_name: str = "",
    checkpoint_name: str = "",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, quantile_rows, summary_rows = evaluate_rankers(y_true, rankers)

    paths = {
        "metrics": output_dir / "metrics.csv",
        "quantiles": output_dir / "quantile_precision_recall.csv",
        "summary": output_dir / "score_summary.csv",
        "markdown": output_dir / "report.md",
        "distribution_plot": output_dir / "score_distributions.png",
        "pr_plot": output_dir / "precision_recall.png",
    }
    _write_csv(paths["metrics"], metric_rows)
    _write_csv(paths["quantiles"], quantile_rows)
    _write_csv(paths["summary"], summary_rows)
    _write_markdown_report(
        paths["markdown"],
        metric_rows,
        dataset_name=dataset_name,
        checkpoint_name=checkpoint_name,
    )
    _write_plots(paths, y_true, rankers)
    return paths


def resolve_dihedral_indices(
    atoms: list[object],
    goal: GoalSpec,
) -> tuple[int, int, int, int]:
    if goal.type != "dihedral_window":
        raise ValueError("Only dihedral_window goals can resolve dihedral indices.")
    indices: list[int] = []
    for selection in goal.selections:
        mask = atom_selection_mask(atoms, selection)
        matches = np.flatnonzero(mask)
        if len(matches) != 1:
            raise ValueError(
                "Dihedral selections must each match exactly one atom; "
                f"{selection!r} matched {len(matches)} atoms."
            )
        indices.append(int(matches[0]))
    return tuple(indices)  # type: ignore[return-value]


def _distance_to_angle_window(angle: float, lower: float, upper: float) -> float:
    angle = _normalize_angle(angle)
    lower = _normalize_angle(lower)
    upper = _normalize_angle(upper)
    if _angle_inside(angle, lower, upper):
        return 0.0
    return min(_circular_distance(angle, lower), _circular_distance(angle, upper))


def _angle_inside(angle: float, lower: float, upper: float) -> bool:
    if lower <= upper:
        return lower <= angle <= upper
    return angle >= lower or angle <= upper


def _normalize_angle(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def _circular_distance(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


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


def _write_markdown_report(
    path: Path,
    metric_rows: list[dict[str, float | str]],
    dataset_name: str,
    checkpoint_name: str,
) -> None:
    lines = ["# STRIDE Evaluation Report", ""]
    if dataset_name:
        lines.append(f"Dataset: `{dataset_name}`")
    if checkpoint_name:
        lines.append(f"Checkpoint: `{checkpoint_name}`")
    if dataset_name or checkpoint_name:
        lines.append("")

    lines.extend(
        [
            "| Ranker | AUROC | AUPRC | Top 1% | Top 5% | Top 10% | Top 25% |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metric_rows:
        lines.append(
            "| {ranker} | {auroc} | {auprc} | {top1} | {top5} | {top10} | {top25} |".format(
                ranker=row["ranker"],
                auroc=_fmt(row.get("auroc")),
                auprc=_fmt(row.get("auprc")),
                top1=_fmt(row.get("top1_enrichment")),
                top5=_fmt(row.get("top5_enrichment")),
                top10=_fmt(row.get("top10_enrichment")),
                top25=_fmt(row.get("top25_enrichment")),
            )
        )
    lines.append("")
    lines.extend(_baseline_comparison_lines(metric_rows))
    if lines[-1] != "":
        lines.append("")
    lines.append(
        "Top-k values are enrichment over the dataset positive rate; higher is better."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(paths: dict[str, Path], y_true: np.ndarray, rankers: dict[str, np.ndarray]) -> None:
    cache_dir = paths["distribution_plot"].parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except Exception:
        return

    plt.figure(figsize=(8, 5))
    for name, scores in rankers.items():
        values = np.asarray(scores, dtype=float)
        plt.hist(values, bins=_histogram_bins(values), alpha=0.45, label=name)
    plt.xlabel("Score")
    plt.ylabel("Examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["distribution_plot"], dpi=150)
    plt.close()

    if len(np.unique(np.asarray(y_true).astype(int))) != 2:
        return
    plt.figure(figsize=(6, 5))
    for name, scores in rankers.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["pr_plot"], dpi=150)
    plt.close()


def _histogram_bins(values: np.ndarray) -> int:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1
    score_range = float(np.max(finite) - np.min(finite))
    if score_range <= 1e-12:
        return 1
    return min(40, max(1, int(finite.size)))


def _baseline_comparison_lines(
    metric_rows: list[dict[str, float | str]],
) -> list[str]:
    stride_row = next((row for row in metric_rows if row.get("ranker") == "STRIDE"), None)
    if stride_row is None:
        return []
    baseline_rows = [
        row
        for row in metric_rows
        if row.get("ranker") not in {"STRIDE", "random"}
    ]
    if not baseline_rows:
        return []

    lines = ["## Baseline Comparison", ""]
    for metric in ("auroc", "auprc", "top25_enrichment"):
        stride_value = _finite_float(stride_row.get(metric))
        baseline_values = [
            _finite_float(row.get(metric))
            for row in baseline_rows
            if _finite_float(row.get(metric)) is not None
        ]
        if stride_value is None or not baseline_values:
            continue
        best_baseline = max(baseline_values)
        verdict = "beats" if stride_value > best_baseline else "does not beat"
        lines.append(
            f"- STRIDE {verdict} the best non-random baseline on {metric}: "
            f"{stride_value:.4g} vs {best_baseline:.4g}."
        )
    return lines


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return "nan"
    return f"{number:.4g}"
