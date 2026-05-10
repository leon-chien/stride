from __future__ import annotations

import numpy as np


def scores_to_quantile_bins(
    scores: np.ndarray,
    num_bins: int,
    reference_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert continuous STRIDE value scores into WESTPA-compatible integer bins.

    If reference_scores is provided, edges are computed from that reference
    distribution. Otherwise edges come from the current score batch.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(f"Expected scores shape [num_walkers], got {scores.shape}")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")

    edge_source = scores if reference_scores is None else np.asarray(reference_scores)
    if edge_source.ndim != 1:
        raise ValueError("reference_scores must be one-dimensional.")
    if len(edge_source) == 0:
        raise ValueError("Cannot build quantile bins from an empty score set.")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(edge_source, quantiles[1:-1])
    edges = np.unique(edges)

    bin_ids = np.digitize(scores, edges, right=False).astype(int)
    return bin_ids, edges.astype(np.float32)


def combine_value_heads(
    p_event: np.ndarray,
    flux_value: np.ndarray,
    uncertainty: np.ndarray,
    flux_weight: float = 1.0,
    uncertainty_weight: float = 0.1,
) -> np.ndarray:
    """
    Default WESTPA control score from STRIDE's value heads.
    """
    p_event = np.asarray(p_event, dtype=np.float32)
    flux_value = np.asarray(flux_value, dtype=np.float32)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)

    if p_event.shape != flux_value.shape or p_event.shape != uncertainty.shape:
        raise ValueError("All value heads must have the same shape.")

    return p_event + flux_weight * flux_value + uncertainty_weight * uncertainty
