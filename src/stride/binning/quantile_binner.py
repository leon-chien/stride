from __future__ import annotations

import numpy as np


class QuantileBinner:
    """
    Convert continuous model scores into discrete bins using quantiles.

    This is the toy version of a WESTPA-compatible adaptive binner.

    Example:
        scores = [0.1, 0.3, 0.9, 0.6]
        binner = QuantileBinner(num_bins=4)
        bin_ids = binner.fit_transform(scores)

    Higher model scores should usually end up in higher-numbered bins.
    """

    def __init__(self, num_bins: int = 4) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2.")

        self.num_bins = num_bins
        self.edges_: np.ndarray | None = None

    def fit(self, scores: np.ndarray) -> "QuantileBinner":
        """
        Learn quantile bin edges from scores.

        Args:
            scores: shape [num_examples]

        Returns:
            self
        """
        scores = np.asarray(scores, dtype=float)

        if scores.ndim != 1:
            raise ValueError(f"Expected 1D scores, got shape {scores.shape}")

        quantiles = np.linspace(0.0, 1.0, self.num_bins + 1)

        # Internal edges only. We do not need 0% and 100% endpoints.
        edges = np.quantile(scores, quantiles[1:-1])

        # Remove duplicate edges to avoid weird behavior when many scores are identical.
        self.edges_ = np.unique(edges)

        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Assign scores to bins.

        Args:
            scores: shape [num_examples]

        Returns:
            bin_ids: shape [num_examples]
        """
        if self.edges_ is None:
            raise RuntimeError("QuantileBinner must be fit before transform.")

        scores = np.asarray(scores, dtype=float)

        if scores.ndim != 1:
            raise ValueError(f"Expected 1D scores, got shape {scores.shape}")

        # np.digitize returns 0 for below first edge, 1 for between first/second, etc.
        bin_ids = np.digitize(scores, self.edges_, right=False)

        return bin_ids.astype(int)

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Fit bin edges and assign scores to bins.
        """
        return self.fit(scores).transform(scores)


def summarize_bins(
    bin_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """
    Summarize positive rate and optional score statistics for each bin.

    Args:
        bin_ids: Discrete bin IDs, shape [num_examples]
        labels: Binary labels, shape [num_examples]
        scores: Optional model scores, shape [num_examples]

    Returns:
        List of dictionaries, one per bin.
    """
    bin_ids = np.asarray(bin_ids)
    labels = np.asarray(labels).astype(int)

    if bin_ids.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: bin_ids has {bin_ids.shape}, labels has {labels.shape}"
        )

    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != labels.shape:
            raise ValueError(
                f"Shape mismatch: scores has {scores.shape}, labels has {labels.shape}"
            )

    summary: list[dict[str, float]] = []

    for bin_id in np.unique(bin_ids):
        mask = bin_ids == bin_id
        n = int(mask.sum())

        row: dict[str, float] = {
            "bin_id": int(bin_id),
            "n": n,
            "fraction": float(n / len(labels)),
            "positive_rate": float(labels[mask].mean()) if n > 0 else float("nan"),
        }

        if scores is not None:
            row["mean_score"] = float(scores[mask].mean()) if n > 0 else float("nan")
            row["min_score"] = float(scores[mask].min()) if n > 0 else float("nan")
            row["max_score"] = float(scores[mask].max()) if n > 0 else float("nan")

        summary.append(row)

    return summary


if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    scores = np.array([0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.3, 0.95])

    binner = QuantileBinner(num_bins=4)
    bin_ids = binner.fit_transform(scores)

    print("Scores:", scores)
    print("Bin IDs:", bin_ids)
    print("Edges:", binner.edges_)

    for row in summarize_bins(bin_ids, y_true, scores):
        print(row)