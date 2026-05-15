from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stride.binning import scores_to_quantile_bins
from stride.training.metrics import (
    compute_binary_metrics,
    top_k_enrichment,
    top_k_positive_rate,
)
from stride.training.westpa_evaluation import (
    pcoord_baseline_rankers,
    westpa_iteration_split_indices,
)
from stride.westpa_plugin.value_mapper import StrideValueBinMapper, ValueMapperConfig


@dataclass(frozen=True)
class ReplayConfig:
    eval_split: str = "validation"
    split_strategy: str = "tail"
    validation_fraction: float = 0.2
    seed: int = 7
    num_bins: int = 8
    binning: str = "quantile"
    pcoord_dim: int = 0
    baseline_key: str = "last_pcoord_low"
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25)


def assign_score_bins(
    scores: np.ndarray,
    num_bins: int,
    binning: str = "quantile",
    reference_scores: np.ndarray | None = None,
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ranking scores into WESTPA-facing integer bin IDs.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError("scores must have shape [examples].")
    if len(scores) == 0:
        raise ValueError("Cannot assign bins for an empty score array.")
    if binning == "quantile":
        return scores_to_quantile_bins(scores, num_bins, reference_scores=reference_scores)
    if binning == "fixed":
        mapper = StrideValueBinMapper(
            ValueMapperConfig(
                num_bins=num_bins,
                min_score=float(min_score),
                max_score=float(max_score),
            )
        )
        edges = np.linspace(float(min_score), float(max_score), num_bins + 1)[1:-1]
        return mapper.assign(scores), edges.astype(np.float32)
    raise ValueError("binning must be 'quantile' or 'fixed'.")


def priority_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Return one-indexed priority ranks, where 1 is the highest score.
    """
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranks = np.empty((len(scores),), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def replay_westpa_steering(
    lineage_npz: str | Path,
    output_dir: str | Path,
    stride_scores: np.ndarray | None = None,
    config: ReplayConfig | None = None,
) -> dict[str, Path]:
    """
    Replay held-out WESTPA iterations as a steering decision surface.

    The replay ranks active examples by STRIDE and by a simple pcoord baseline,
    assigns bins, and writes WESTPA-facing assignment arrays plus diagnostics.
    """
    cfg = config or ReplayConfig()
    data = np.load(lineage_npz)
    labels = data["event_labels"].astype(np.float32)
    flux = data["flux_labels"].astype(np.float32) if "flux_labels" in data else np.zeros_like(labels)
    indices = _eval_indices(data, cfg)

    pcoord_dims = data["pcoord_dim"].astype(np.int64) if "pcoord_dim" in data else cfg.pcoord_dim
    targets = data["threshold"].astype(np.float32) if "threshold" in data else None
    baselines = pcoord_baseline_rankers(
        data["pcoord_windows"],
        data["window_mask"],
        target=targets,
        pcoord_dim=pcoord_dims,
    )
    if cfg.baseline_key not in baselines:
        raise ValueError(
            f"Baseline {cfg.baseline_key!r} is not available. "
            f"Available baselines: {sorted(baselines)}"
        )

    rankers: dict[str, np.ndarray] = {cfg.baseline_key: baselines[cfg.baseline_key]}
    if stride_scores is not None:
        stride_scores = np.asarray(stride_scores, dtype=np.float32)
        if stride_scores.shape != labels.shape:
            raise ValueError("stride_scores must have one score per lineage example.")
        rankers = {"STRIDE": stride_scores, **rankers}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, float | str]] = []
    bin_rows: list[dict[str, float | str]] = []
    grouped_rows: list[dict[str, float | str]] = []
    assignments: dict[str, np.ndarray] = {
        "eval_indices": indices.astype(np.int64),
        "event_labels": labels[indices].astype(np.float32),
        "flux_labels": flux[indices].astype(np.float32),
        "n_iter": data["n_iter"][indices].astype(np.int64),
        "seg_id": data["seg_id"][indices].astype(np.int64),
    }

    for ranker_name, all_scores in rankers.items():
        scores = np.asarray(all_scores, dtype=np.float32)[indices]
        bins, _ = assign_score_bins(scores, cfg.num_bins, binning=cfg.binning)
        ranks = priority_ranks(scores)
        key = _assignment_prefix(ranker_name)
        assignments[f"{key}_score"] = scores.astype(np.float32)
        assignments[f"{key}_bin"] = bins.astype(np.int64)
        assignments[f"{key}_priority_rank"] = ranks.astype(np.int64)

        metric_rows.append(
            _ranker_metrics(
                ranker_name,
                labels[indices],
                flux[indices],
                scores,
                bins,
                data["seg_id"][indices],
                cfg.top_fractions,
            )
        )
        bin_rows.extend(_bin_rows(ranker_name, labels[indices], flux[indices], scores, bins))
        grouped_rows.extend(
            _grouped_rows(
                ranker_name,
                data,
                indices,
                labels,
                flux,
                all_scores,
                cfg.num_bins,
                cfg.binning,
                cfg.top_fractions,
            )
        )

    paths = {
        "markdown": output_dir / "report.md",
        "metrics": output_dir / "steering_metrics.csv",
        "bins": output_dir / "bin_occupancy.csv",
        "grouped_metrics": output_dir / "grouped_steering_metrics.csv",
        "assignments": output_dir / "steering_assignments.npz",
    }
    _write_csv(paths["metrics"], metric_rows)
    _write_csv(paths["bins"], bin_rows)
    _write_csv(paths["grouped_metrics"], grouped_rows)
    np.savez_compressed(paths["assignments"], **assignments)
    _write_markdown(paths["markdown"], lineage_npz, cfg, metric_rows)
    return paths


def _eval_indices(data: np.lib.npyio.NpzFile, cfg: ReplayConfig) -> np.ndarray:
    labels = data["event_labels"]
    if cfg.eval_split == "all":
        return np.arange(len(labels), dtype=np.int64)
    train_indices, val_indices = westpa_iteration_split_indices(
        data["n_iter"],
        validation_fraction=cfg.validation_fraction,
        split_strategy=cfg.split_strategy,
        seed=cfg.seed,
        goal_id=data["goal_id"].astype(str) if "goal_id" in data else None,
        cell_id=data["cell_id"].astype(str) if "cell_id" in data else None,
    )
    if cfg.eval_split == "train":
        return train_indices
    if cfg.eval_split == "validation":
        return val_indices
    raise ValueError("eval_split must be 'all', 'train', or 'validation'.")


def _ranker_metrics(
    ranker: str,
    labels: np.ndarray,
    flux: np.ndarray,
    scores: np.ndarray,
    bins: np.ndarray,
    seg_id: np.ndarray,
    top_fractions: tuple[float, ...],
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {"ranker": ranker}
    metrics.update(compute_binary_metrics(labels, scores))
    for fraction in top_fractions:
        label = _fraction_label(fraction)
        metrics[f"top{label}_positive_rate"] = top_k_positive_rate(labels, scores, fraction)
        metrics[f"top{label}_enrichment"] = top_k_enrichment(labels, scores, fraction)

    counts = np.bincount(bins.astype(np.int64), minlength=int(np.max(bins)) + 1)
    nonempty = counts[counts > 0]
    metrics["nonempty_bins"] = float(len(nonempty))
    metrics["occupancy_cv"] = _safe_cv(nonempty)
    metrics["score_event_corr"] = _safe_corr(scores, labels)
    metrics["score_flux_corr"] = _safe_corr(scores, flux)
    metrics["bin_event_rate_gradient"] = _bin_event_rate_gradient(labels, bins)

    top_n = max(1, int(np.ceil(0.25 * len(scores))))
    top_indices = np.argsort(scores)[::-1][:top_n]
    metrics["top25_unique_seg_fraction"] = float(len(np.unique(seg_id[top_indices])) / top_n)
    return metrics


def _bin_rows(
    ranker: str,
    labels: np.ndarray,
    flux: np.ndarray,
    scores: np.ndarray,
    bins: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    total = len(labels)
    for bin_id in sorted(np.unique(bins.astype(np.int64))):
        mask = bins == bin_id
        rows.append(
            {
                "ranker": ranker,
                "bin": float(bin_id),
                "count": float(np.sum(mask)),
                "fraction": float(np.mean(mask)) if total else float("nan"),
                "positive_rate": float(np.mean(labels[mask])) if np.any(mask) else float("nan"),
                "flux_sum": float(np.sum(flux[mask])) if np.any(mask) else 0.0,
                "mean_score": float(np.mean(scores[mask])) if np.any(mask) else float("nan"),
            }
        )
    return rows


def _grouped_rows(
    ranker: str,
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    labels: np.ndarray,
    flux: np.ndarray,
    all_scores: np.ndarray,
    num_bins: int,
    binning: str,
    top_fractions: tuple[float, ...],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for field in ("goal_id", "cell_id"):
        if field not in data.files:
            continue
        values = data[field].astype(str)
        for group in sorted(np.unique(values[indices])):
            group_indices = indices[values[indices] == group]
            scores = np.asarray(all_scores, dtype=np.float32)[group_indices]
            bins, _ = assign_score_bins(scores, num_bins, binning=binning)
            row = _ranker_metrics(
                ranker,
                labels[group_indices],
                flux[group_indices],
                scores,
                bins,
                data["seg_id"][group_indices],
                top_fractions,
            )
            row["group_type"] = field
            row["group_id"] = group
            rows.append(row)
    return rows


def _bin_event_rate_gradient(labels: np.ndarray, bins: np.ndarray) -> float:
    rates: list[tuple[int, float]] = []
    for bin_id in sorted(np.unique(bins.astype(np.int64))):
        mask = bins == bin_id
        if np.any(mask):
            rates.append((int(bin_id), float(np.mean(labels[mask]))))
    if len(rates) < 2:
        return float("nan")
    return rates[-1][1] - rates[0][1]


def _safe_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    mean = float(np.mean(values))
    if mean == 0.0:
        return float("nan")
    return float(np.std(values) / mean)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _fraction_label(fraction: float) -> str:
    percentage = fraction * 100.0
    if abs(percentage - round(percentage)) < 1e-9:
        return str(int(round(percentage)))
    return f"{percentage:g}".replace(".", "_")


def _assignment_prefix(ranker: str) -> str:
    if ranker == "STRIDE":
        return "stride"
    return ranker.lower().replace(" ", "_").replace("-", "_")


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


def _write_markdown(
    path: Path,
    lineage_npz: str | Path,
    cfg: ReplayConfig,
    rows: list[dict[str, float | str]],
) -> None:
    lines = [
        "# STRIDE WESTPA Steering Replay",
        "",
        f"Dataset: `{lineage_npz}`",
        f"Evaluation split: `{cfg.eval_split}` using `{cfg.split_strategy}`",
        f"Binning: `{cfg.binning}` with `{cfg.num_bins}` bins",
        "",
        "| Ranker | AUROC | AUPRC | Top 5% enrichment | Top 25% enrichment | Bin gradient | Occupancy CV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {ranker} | {auroc:.4g} | {auprc:.4g} | {top5:.4g} | {top25:.4g} | {gradient:.4g} | {cv:.4g} |".format(
                ranker=row["ranker"],
                auroc=float(row.get("auroc", float("nan"))),
                auprc=float(row.get("auprc", float("nan"))),
                top5=float(row.get("top5_enrichment", float("nan"))),
                top25=float(row.get("top25_enrichment", float("nan"))),
                gradient=float(row.get("bin_event_rate_gradient", float("nan"))),
                cv=float(row.get("occupancy_cv", float("nan"))),
            )
        )

    stride = next((row for row in rows if row["ranker"] == "STRIDE"), None)
    baseline = next((row for row in rows if row["ranker"] == cfg.baseline_key), None)
    if stride is not None and baseline is not None:
        lines.extend(["", "## Steering Comparison", ""])
        for key, label in (
            ("top5_enrichment", "top-5% enrichment"),
            ("top25_enrichment", "top-25% enrichment"),
            ("auprc", "AUPRC"),
        ):
            stride_value = float(stride.get(key, float("nan")))
            baseline_value = float(baseline.get(key, float("nan")))
            verb = "beats" if stride_value > baseline_value else "does not beat"
            lines.append(
                f"- STRIDE {verb} `{cfg.baseline_key}` on {label}: "
                f"{stride_value:.4g} vs {baseline_value:.4g}."
            )

    lines.extend(
        [
            "",
            "Generated arrays include `stride_score`, `stride_bin`, and "
            "`stride_priority_rank` when STRIDE scores are provided.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
