from __future__ import annotations

import csv
import json
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
    bin_reference: str = "train"
    per_iteration: bool = False
    pcoord_dim: int = 0
    baseline_key: str = "last_pcoord_low"
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25)
    score_key: str = "p_event"
    stride_scores_path: str | None = None
    checkpoint_path: str | None = None


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


def assign_score_bins_from_edges(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Assign scores using precomputed frozen bin edges.
    """
    scores = np.asarray(scores, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError("scores must have shape [examples].")
    if edges.ndim != 1:
        raise ValueError("edges must be one-dimensional.")
    return np.digitize(scores, edges, right=False).astype(np.int64)


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
    if cfg.bin_reference not in {"train", "eval"}:
        raise ValueError("bin_reference must be 'train' or 'eval'.")
    data = np.load(lineage_npz)
    labels = data["event_labels"].astype(np.float32)
    flux = data["flux_labels"].astype(np.float32) if "flux_labels" in data else np.zeros_like(labels)
    train_indices, eval_indices = _split_indices(data, cfg)
    reference_indices = train_indices if cfg.bin_reference == "train" else eval_indices

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

    random_scores = np.random.default_rng(cfg.seed).random(len(labels)).astype(np.float32)
    rankers: dict[str, np.ndarray] = {
        cfg.baseline_key: baselines[cfg.baseline_key],
        "random": random_scores,
    }
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
    iteration_rows: list[dict[str, float | str]] = []
    iteration_summary_rows: list[dict[str, float | str]] = []
    assignments: dict[str, np.ndarray] = {
        "eval_indices": eval_indices.astype(np.int64),
        "calibration_indices": reference_indices.astype(np.int64),
        "event_labels": labels[eval_indices].astype(np.float32),
        "flux_labels": flux[eval_indices].astype(np.float32),
        "n_iter": data["n_iter"][eval_indices].astype(np.int64),
        "seg_id": data["seg_id"][eval_indices].astype(np.int64),
        "bin_reference": np.asarray(cfg.bin_reference),
        "split_strategy": np.asarray(cfg.split_strategy),
        "eval_split": np.asarray(cfg.eval_split),
        "validation_fraction": np.asarray(cfg.validation_fraction, dtype=np.float32),
    }
    control_rankers: dict[str, dict[str, object]] = {}

    for ranker_name, all_scores in rankers.items():
        all_scores = np.asarray(all_scores, dtype=np.float32)
        scores = all_scores[eval_indices]
        _, edges = assign_score_bins(
            scores,
            cfg.num_bins,
            binning=cfg.binning,
            reference_scores=all_scores[reference_indices],
        )
        bins = assign_score_bins_from_edges(scores, edges)
        ranks = priority_ranks(scores)
        iteration_ranks = _per_iteration_priority_ranks(scores, data["n_iter"][eval_indices])
        key = _assignment_prefix(ranker_name)
        assignments[f"{key}_score"] = scores.astype(np.float32)
        assignments[f"{key}_bin"] = bins.astype(np.int64)
        assignments[f"{key}_priority_rank"] = ranks.astype(np.int64)
        assignments[f"{key}_iteration_priority_rank"] = iteration_ranks.astype(np.int64)
        assignments[f"{key}_bin_edges"] = edges.astype(np.float32)
        control_rankers[key] = {
            "ranker": ranker_name,
            "score_key": cfg.score_key if ranker_name == "STRIDE" else key,
            "bin_edges": _json_list(edges),
            "num_bins": int(len(edges) + 1),
        }

        metric_rows.append(
            _ranker_metrics(
                ranker_name,
                labels[eval_indices],
                flux[eval_indices],
                scores,
                bins,
                data["seg_id"][eval_indices],
                cfg.top_fractions,
            )
        )
        bin_rows.extend(_bin_rows(ranker_name, labels[eval_indices], flux[eval_indices], scores, bins))
        grouped_rows.extend(
            _grouped_rows(
                ranker_name,
                data,
                eval_indices,
                labels,
                flux,
                all_scores,
                edges,
                cfg.top_fractions,
            )
        )
        if cfg.per_iteration:
            iteration_rows.extend(
                _iteration_rows(
                    ranker_name,
                    data["n_iter"][eval_indices],
                    labels[eval_indices],
                    flux[eval_indices],
                    scores,
                    bins,
                    data["seg_id"][eval_indices],
                    cfg.top_fractions,
                )
            )

    if iteration_rows:
        iteration_summary_rows = _summarize_iteration_rows(iteration_rows, cfg.top_fractions)
    paths = {
        "markdown": output_dir / "report.md",
        "metrics": output_dir / "steering_metrics.csv",
        "bins": output_dir / "bin_occupancy.csv",
        "grouped_metrics": output_dir / "grouped_steering_metrics.csv",
        "iteration_metrics": output_dir / "iteration_steering_metrics.csv",
        "iteration_summary": output_dir / "iteration_steering_summary.csv",
        "assignments": output_dir / "steering_assignments.npz",
        "control_config": output_dir / "stride_control_config.json",
    }
    _write_csv(paths["metrics"], metric_rows)
    _write_csv(paths["bins"], bin_rows)
    _write_csv(paths["grouped_metrics"], grouped_rows)
    _write_csv(paths["iteration_metrics"], iteration_rows)
    _write_csv(paths["iteration_summary"], iteration_summary_rows)
    np.savez_compressed(paths["assignments"], **assignments)
    _write_control_config(
        paths["control_config"],
        lineage_npz=lineage_npz,
        cfg=cfg,
        rankers=control_rankers,
        train_count=len(train_indices),
        eval_count=len(eval_indices),
    )
    _write_markdown(paths["markdown"], lineage_npz, cfg, metric_rows, iteration_summary_rows)
    return paths


def _split_indices(
    data: np.lib.npyio.NpzFile,
    cfg: ReplayConfig,
) -> tuple[np.ndarray, np.ndarray]:
    labels = data["event_labels"]
    if cfg.eval_split == "all":
        indices = np.arange(len(labels), dtype=np.int64)
        return indices, indices
    train_indices, val_indices = westpa_iteration_split_indices(
        data["n_iter"],
        validation_fraction=cfg.validation_fraction,
        split_strategy=cfg.split_strategy,
        seed=cfg.seed,
        goal_id=data["goal_id"].astype(str) if "goal_id" in data else None,
        cell_id=data["cell_id"].astype(str) if "cell_id" in data else None,
    )
    if cfg.eval_split == "train":
        return train_indices, train_indices
    if cfg.eval_split == "validation":
        return train_indices, val_indices
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
    edges: np.ndarray,
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
            bins = assign_score_bins_from_edges(scores, edges)
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


def _iteration_rows(
    ranker: str,
    n_iter: np.ndarray,
    labels: np.ndarray,
    flux: np.ndarray,
    scores: np.ndarray,
    bins: np.ndarray,
    seg_id: np.ndarray,
    top_fractions: tuple[float, ...],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for iteration in sorted(np.unique(n_iter.astype(np.int64))):
        mask = n_iter == iteration
        row = _ranker_metrics(
            ranker,
            labels[mask],
            flux[mask],
            scores[mask],
            bins[mask],
            seg_id[mask],
            top_fractions,
        )
        row["n_iter"] = float(iteration)
        row["count"] = float(np.sum(mask))
        row["positive_count"] = float(np.sum(labels[mask] >= 0.5))
        rows.append(row)
    return rows


def _summarize_iteration_rows(
    rows: list[dict[str, float | str]],
    top_fractions: tuple[float, ...],
) -> list[dict[str, float | str]]:
    summary_rows: list[dict[str, float | str]] = []
    metric_keys = ["auroc", "auprc", "bin_event_rate_gradient"]
    for fraction in top_fractions:
        label = _fraction_label(fraction)
        metric_keys.extend([f"top{label}_positive_rate", f"top{label}_enrichment"])
    for ranker in sorted({str(row["ranker"]) for row in rows}):
        ranker_rows = [row for row in rows if row["ranker"] == ranker]
        summary: dict[str, float | str] = {
            "ranker": ranker,
            "n_iterations": float(len(ranker_rows)),
        }
        for key in metric_keys:
            values = np.asarray([float(row.get(key, float("nan"))) for row in ranker_rows])
            values = values[np.isfinite(values)]
            summary[f"{key}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            summary[f"{key}_std"] = float(np.std(values)) if len(values) else float("nan")
        summary_rows.append(summary)
    return summary_rows


def _per_iteration_priority_ranks(scores: np.ndarray, n_iter: np.ndarray) -> np.ndarray:
    ranks = np.empty((len(scores),), dtype=np.int64)
    for iteration in np.unique(n_iter):
        indices = np.flatnonzero(n_iter == iteration)
        ranks[indices] = priority_ranks(scores[indices])
    return ranks


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
    iteration_summary_rows: list[dict[str, float | str]],
) -> None:
    lines = [
        "# STRIDE WESTPA Steering Replay",
        "",
        f"Dataset: `{lineage_npz}`",
        f"Evaluation split: `{cfg.eval_split}` using `{cfg.split_strategy}`",
        f"Binning: `{cfg.binning}` with `{cfg.num_bins}` bins calibrated on `{cfg.bin_reference}`",
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
    stride_iteration = next(
        (row for row in iteration_summary_rows if row["ranker"] == "STRIDE"),
        None,
    )
    baseline_iteration = next(
        (row for row in iteration_summary_rows if row["ranker"] == cfg.baseline_key),
        None,
    )
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
        pass_run = _passes_replay_acceptance(stride, baseline, stride_iteration, baseline_iteration)
        lines.append(f"- Replay acceptance: `{'PASS' if pass_run else 'FAIL'}`.")

    if iteration_summary_rows:
        lines.extend(
            [
                "",
                "## Per-Iteration Summary",
                "",
                "| Ranker | Iterations | Mean Top 5% enrichment | Mean Top 25% enrichment | Mean AUPRC |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in iteration_summary_rows:
            lines.append(
                "| {ranker} | {n:.0f} | {top5:.4g} | {top25:.4g} | {auprc:.4g} |".format(
                    ranker=row["ranker"],
                    n=float(row.get("n_iterations", 0.0)),
                    top5=float(row.get("top5_enrichment_mean", float("nan"))),
                    top25=float(row.get("top25_enrichment_mean", float("nan"))),
                    auprc=float(row.get("auprc_mean", float("nan"))),
                )
            )

    lines.extend(
        [
            "",
            "Generated arrays include `stride_score`, `stride_bin`, and "
            "`stride_priority_rank` when STRIDE scores are provided.",
            "Frozen bin edges and deployment metadata are saved in "
            "`stride_control_config.json`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _passes_replay_acceptance(
    stride: dict[str, float | str],
    baseline: dict[str, float | str],
    stride_iteration: dict[str, float | str] | None,
    baseline_iteration: dict[str, float | str] | None,
) -> bool:
    if float(stride.get("auprc", float("nan"))) <= float(baseline.get("auprc", float("nan"))):
        return False
    if stride_iteration is None or baseline_iteration is None:
        return float(stride.get("top25_enrichment", float("nan"))) > float(
            baseline.get("top25_enrichment", float("nan"))
        )
    return float(stride_iteration.get("top25_enrichment_mean", float("nan"))) > float(
        baseline_iteration.get("top25_enrichment_mean", float("nan"))
    )


def _write_control_config(
    path: Path,
    lineage_npz: str | Path,
    cfg: ReplayConfig,
    rankers: dict[str, dict[str, object]],
    train_count: int,
    eval_count: int,
) -> None:
    payload = {
        "lineage_npz": str(lineage_npz),
        "stride_scores_npz": cfg.stride_scores_path,
        "checkpoint_path": cfg.checkpoint_path,
        "score_key": cfg.score_key,
        "baseline_key": cfg.baseline_key,
        "num_bins": cfg.num_bins,
        "binning": cfg.binning,
        "bin_reference": cfg.bin_reference,
        "eval_split": cfg.eval_split,
        "split_strategy": cfg.split_strategy,
        "validation_fraction": cfg.validation_fraction,
        "seed": cfg.seed,
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "rankers": rankers,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float32)]
