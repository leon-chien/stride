from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WESTPA_INDEX_DTYPE = np.uint16


@dataclass(frozen=True)
class ValueMapperConfig:
    """
    Configuration for mapping STRIDE scores to WESTPA bins.
    """

    num_bins: int = 8
    score_coord_dim: int = 0
    min_score: float = 0.0
    max_score: float = 1.0


class StrideValueBinMapper:
    """
    WESTPA-compatible bin mapper for scalar STRIDE value scores.

    WESTPA calls `assign(coords, mask=None, output=None)` where coords are the
    current progress-coordinate values. This mapper expects one coordinate
    dimension to contain a STRIDE score and maps that score to integer bins.

    In early integrations, the score dimension can be a pcoord-derived proxy.
    In the full plugin path, STRIDE should compute the score from walker history
    before WESTPA bin assignment and expose it as the configured score dimension.
    """

    def __init__(
        self,
        config: ValueMapperConfig | None = None,
        edges: np.ndarray | None = None,
    ) -> None:
        self.config = config or ValueMapperConfig()
        if self.config.num_bins < 2:
            raise ValueError("num_bins must be at least 2.")
        if self.config.max_score <= self.config.min_score:
            raise ValueError("max_score must be greater than min_score.")

        if edges is None:
            self.edges = np.linspace(
                self.config.min_score,
                self.config.max_score,
                self.config.num_bins + 1,
                dtype=np.float32,
            )[1:-1]
        else:
            edges = np.asarray(edges, dtype=np.float32)
            if edges.ndim != 1:
                raise ValueError("edges must be one-dimensional.")
            if np.any(np.diff(edges) <= 0):
                raise ValueError("edges must be strictly increasing.")
            self.edges = edges

        self.nbins = len(self.edges) + 1
        self.labels = [f"stride_value_{i}" for i in range(self.nbins)]

    def construct_bins(self, type_=None) -> np.ndarray:
        """
        Match WESTPA BinMapper's construct_bins behavior without importing WESTPA.
        """
        if type_ is None:
            return np.empty((self.nbins,), dtype=object)
        return np.array([type_() for _ in range(self.nbins)], dtype=object)

    def assign(
        self,
        coords,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Assign each pcoord row to a STRIDE value bin.
        """
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords[:, None]
        if coords.ndim != 2:
            raise ValueError(f"Expected coords shape [n_points, ndim], got {coords.shape}")
        if self.config.score_coord_dim >= coords.shape[1]:
            raise ValueError(
                f"score_coord_dim={self.config.score_coord_dim} is out of bounds "
                f"for coords with shape {coords.shape}"
            )

        if output is None:
            output = np.zeros((len(coords),), dtype=WESTPA_INDEX_DTYPE)
        else:
            output = np.asarray(output)
            if output.shape != (len(coords),):
                raise ValueError(f"Expected output shape {(len(coords),)}, got {output.shape}")

        if mask is None:
            mask = np.ones((len(coords),), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (len(coords),):
                raise ValueError(f"Expected mask shape {(len(coords),)}, got {mask.shape}")

        scores = coords[:, self.config.score_coord_dim]
        clipped_scores = np.clip(
            scores,
            self.config.min_score,
            self.config.max_score,
        )
        output[mask] = np.digitize(
            clipped_scores[mask],
            self.edges,
            right=False,
        ).astype(WESTPA_INDEX_DTYPE)

        return output
