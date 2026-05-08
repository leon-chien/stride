from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Binary labels, shape [num_examples]
        y_score: Predicted probabilities, shape [num_examples]
        threshold: Probability cutoff for accuracy/F1

    Returns:
        Dictionary of metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true.shape}, y_score has {y_score.shape}"
        )

    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, float] = {}

    # AUROC fails if only one class is present.
    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["positive_rate"] = float(np.mean(y_true))
    metrics["mean_score"] = float(np.mean(y_score))

    return metrics


def top_k_enrichment(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: float = 0.1,
) -> float:
    """
    Compute top-k enrichment.

    This asks:
    Among the top k fraction of model-ranked examples,
    how much higher is the positive rate compared to random?

    Example:
        overall positive rate = 0.08
        top 10% positive rate = 0.32
        enrichment = 4.0x
    """
    if not 0.0 < k <= 1.0:
        raise ValueError(f"k must be in (0, 1], got {k}")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true.shape}, y_score has {y_score.shape}"
        )

    baseline_rate = float(np.mean(y_true))

    if baseline_rate == 0.0:
        return float("nan")

    n = len(y_true)
    top_n = max(1, int(np.ceil(k * n)))

    ranked_indices = np.argsort(y_score)[::-1]
    top_indices = ranked_indices[:top_n]

    top_rate = float(np.mean(y_true[top_indices]))

    return top_rate / baseline_rate


def top_k_positive_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: float = 0.1,
) -> float:
    """
    Positive rate among the top-k model-ranked examples.
    """
    if not 0.0 < k <= 1.0:
        raise ValueError(f"k must be in (0, 1], got {k}")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n = len(y_true)
    top_n = max(1, int(np.ceil(k * n)))

    ranked_indices = np.argsort(y_score)[::-1]
    top_indices = ranked_indices[:top_n]

    return float(np.mean(y_true[top_indices]))


if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.7])

    metrics = compute_binary_metrics(y_true, y_score)
    enrichment = top_k_enrichment(y_true, y_score, k=0.5)
    top_rate = top_k_positive_rate(y_true, y_score, k=0.5)

    print(metrics)
    print(f"Top-k enrichment: {enrichment:.2f}x")
    print(f"Top-k positive rate: {top_rate:.3f}")