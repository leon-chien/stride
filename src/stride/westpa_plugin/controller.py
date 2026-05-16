from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stride.westpa_plugin.runtime_scorer import (
    PcoordLineageRuntimeScorer,
    PcoordRuntimeScoringInput,
    RuntimeScoringResult,
)
from stride.westpa_plugin.steering_replay import assign_score_bins_from_edges


@dataclass(frozen=True)
class ControllerAssignment:
    """
    WESTPA-facing controller result for one active walker batch.
    """

    bin_ids: np.ndarray
    scores: np.ndarray
    priority_rank: np.ndarray
    used_fallback: bool
    message: str


class StrideWestpaController:
    """
    Product-facing pcoord STRIDE controller.

    This class is intentionally small: it loads the frozen steering config from
    replay, scores active pcoord histories when a checkpoint is available, and
    falls back to pcoord baseline bins when scoring is unavailable or invalid.
    """

    def __init__(
        self,
        control_config: dict[str, object],
        scorer: PcoordLineageRuntimeScorer | None = None,
        fallback_score: float = 0.0,
    ) -> None:
        self.control_config = dict(control_config)
        self.score_key = str(self.control_config.get("score_key", "p_event"))
        self.baseline_key = str(self.control_config.get("baseline_key", "last_pcoord_low"))
        rankers = self.control_config.get("rankers", {})
        if not isinstance(rankers, dict):
            raise ValueError("control config 'rankers' must be a mapping.")
        self.stride_edges = _edges_from_rankers(rankers, "stride")
        self.fallback_edges = _edges_from_rankers(rankers, self.baseline_key)
        self.scorer = scorer
        self.fallback_score = float(fallback_score)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 256,
        fallback_score: float = 0.0,
    ) -> "StrideWestpaController":
        with open(path, "r", encoding="utf-8") as handle:
            control_config = json.load(handle)
        checkpoint = checkpoint_path or control_config.get("checkpoint_path")
        scorer = PcoordLineageRuntimeScorer(
            checkpoint,
            device=device,
            batch_size=batch_size,
            fallback_score=fallback_score,
        )
        return cls(control_config, scorer=scorer, fallback_score=fallback_score)

    def assign(
        self,
        active_histories: PcoordRuntimeScoringInput,
        pcoords: np.ndarray | None = None,
        metadata: dict[str, np.ndarray] | None = None,
        baseline_scores: np.ndarray | None = None,
    ) -> ControllerAssignment:
        del pcoords, metadata
        num_walkers = int(active_histories.pcoord_windows.shape[0])
        if baseline_scores is not None:
            baseline_scores = np.asarray(baseline_scores, dtype=np.float32)
            if baseline_scores.shape != (num_walkers,):
                raise ValueError("baseline_scores must have shape [num_walkers].")

        if self.scorer is None:
            return self._fallback(num_walkers, baseline_scores, "No scorer configured.")

        result = self.scorer.score(active_histories)
        if result.used_fallback:
            return self._fallback(num_walkers, baseline_scores, result.message)
        if self.score_key not in result.scores:
            return self._fallback(
                num_walkers,
                baseline_scores,
                f"Score key {self.score_key!r} missing; using fallback.",
            )

        scores = np.asarray(result.scores[self.score_key], dtype=np.float32)
        if scores.shape != (num_walkers,) or not np.all(np.isfinite(scores)):
            return self._fallback(
                num_walkers,
                baseline_scores,
                "Invalid STRIDE scores; using fallback.",
            )

        bin_ids = assign_score_bins_from_edges(scores, self.stride_edges)
        return ControllerAssignment(
            bin_ids=bin_ids.astype(np.int64),
            scores=scores,
            priority_rank=_priority_ranks(scores),
            used_fallback=False,
            message=result.message,
        )

    def _fallback(
        self,
        num_walkers: int,
        baseline_scores: np.ndarray | None,
        message: str,
    ) -> ControllerAssignment:
        if baseline_scores is None:
            scores = np.full((num_walkers,), self.fallback_score, dtype=np.float32)
        else:
            scores = np.asarray(baseline_scores, dtype=np.float32)
        bin_ids = assign_score_bins_from_edges(scores, self.fallback_edges)
        return ControllerAssignment(
            bin_ids=bin_ids.astype(np.int64),
            scores=scores.astype(np.float32),
            priority_rank=_priority_ranks(scores),
            used_fallback=True,
            message=message,
        )


def _edges_from_rankers(rankers: dict[str, object], key: str) -> np.ndarray:
    value = rankers.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"control config missing ranker {key!r}.")
    edges = np.asarray(value.get("bin_edges", []), dtype=np.float32)
    if edges.ndim != 1:
        raise ValueError(f"ranker {key!r} bin_edges must be one-dimensional.")
    return edges


def _priority_ranks(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores, kind="mergesort")[::-1]
    ranks = np.empty((len(scores),), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks
