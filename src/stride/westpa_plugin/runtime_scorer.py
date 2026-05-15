from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stride.data import AtomisticDataset
from stride.training import (
    PcoordLineageDataset,
    load_atomistic_checkpoint,
    load_pcoord_lineage_checkpoint,
    score_atomistic_dataset,
    score_pcoord_lineage_dataset,
)
from stride.westpa_plugin.value_mapper import StrideValueBinMapper


@dataclass(frozen=True)
class RuntimeScoringInput:
    """
    Active walker coordinate histories prepared for STRIDE runtime scoring.
    """

    coordinates: np.ndarray
    atom_features: np.ndarray
    atom_mask: np.ndarray
    frame_mask: np.ndarray
    goal_features: np.ndarray

    def to_dataset(self) -> AtomisticDataset:
        num_examples = int(self.coordinates.shape[0])
        dataset = AtomisticDataset(
            coordinates=np.asarray(self.coordinates, dtype=np.float32),
            atom_features=np.asarray(self.atom_features, dtype=np.float32),
            atom_mask=np.asarray(self.atom_mask, dtype=bool),
            frame_mask=np.asarray(self.frame_mask, dtype=bool),
            goal_features=np.asarray(self.goal_features, dtype=np.float32),
            event_labels=np.zeros((num_examples,), dtype=np.float32),
            flux_labels=np.zeros((num_examples,), dtype=np.float32),
            source_frame_start=np.zeros((num_examples,), dtype=np.int64),
        )
        dataset.validate()
        return dataset


@dataclass(frozen=True)
class PcoordRuntimeScoringInput:
    """
    Active walker pcoord lineage histories prepared for STRIDE runtime scoring.
    """

    pcoord_windows: np.ndarray
    window_mask: np.ndarray
    goal_features: np.ndarray

    def to_dataset(self) -> PcoordLineageDataset:
        num_examples = int(self.pcoord_windows.shape[0])
        dataset = PcoordLineageDataset(
            pcoord_windows=np.asarray(self.pcoord_windows, dtype=np.float32),
            window_mask=np.asarray(self.window_mask, dtype=bool),
            goal_features=np.asarray(self.goal_features, dtype=np.float32),
            event_labels=np.zeros((num_examples,), dtype=np.float32),
            flux_labels=np.zeros((num_examples,), dtype=np.float32),
            n_iter=np.arange(num_examples, dtype=np.int64),
            seg_id=np.arange(num_examples, dtype=np.int64),
            weights=np.ones((num_examples,), dtype=np.float64),
        )
        dataset.validate()
        return dataset


@dataclass(frozen=True)
class RuntimeScoringResult:
    """
    Runtime STRIDE score outputs plus diagnostics.
    """

    scores: dict[str, np.ndarray]
    used_fallback: bool
    message: str


class StrideRuntimeScorer:
    """
    Load a STRIDE checkpoint and score active walker history windows.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        device: str | None = None,
        batch_size: int = 64,
        fallback_score: float = 0.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.batch_size = batch_size
        self.fallback_score = float(fallback_score)
        self._model = None
        self._load_error: Exception | None = None

        if self.checkpoint_path is not None:
            try:
                self._model = load_atomistic_checkpoint(
                    self.checkpoint_path,
                    device=device,
                )[0]
            except Exception as exc:
                self._load_error = exc

    def score(self, scoring_input: RuntimeScoringInput) -> RuntimeScoringResult:
        dataset = scoring_input.to_dataset()
        if self._model is None:
            message = "No checkpoint configured; using fallback scores."
            if self._load_error is not None:
                message = f"Model load failed; using fallback scores: {self._load_error}"
            return RuntimeScoringResult(
                scores=_fallback_scores(len(dataset.event_labels), self.fallback_score),
                used_fallback=True,
                message=message,
            )

        try:
            scores = score_atomistic_dataset(
                model=self._model,
                dataset=dataset,
                batch_size=self.batch_size,
                device=self.device,
            )
        except Exception as exc:
            return RuntimeScoringResult(
                scores=_fallback_scores(len(dataset.event_labels), self.fallback_score),
                used_fallback=True,
                message=f"Model scoring failed; using fallback scores: {exc}",
            )

        return RuntimeScoringResult(
            scores=scores,
            used_fallback=False,
            message="Scored with STRIDE checkpoint.",
        )

    def assign_bins(
        self,
        scoring_input: RuntimeScoringInput,
        mapper: StrideValueBinMapper,
        score_key: str = "stride_score",
    ) -> tuple[np.ndarray, RuntimeScoringResult]:
        result = self.score(scoring_input)
        if score_key not in result.scores:
            raise KeyError(f"Runtime scores do not contain {score_key!r}.")
        assignments = mapper.assign(result.scores[score_key])
        return assignments, result


class PcoordLineageRuntimeScorer:
    """
    Load a pcoord-lineage STRIDE checkpoint and score active walker histories.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        device: str | None = None,
        batch_size: int = 256,
        fallback_score: float = 0.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.batch_size = batch_size
        self.fallback_score = float(fallback_score)
        self._model = None
        self._load_error: Exception | None = None

        if self.checkpoint_path is not None:
            try:
                self._model = load_pcoord_lineage_checkpoint(
                    self.checkpoint_path,
                    device=device,
                )[0]
            except Exception as exc:
                self._load_error = exc

    def score(self, scoring_input: PcoordRuntimeScoringInput) -> RuntimeScoringResult:
        dataset = scoring_input.to_dataset()
        if self._model is None:
            message = "No checkpoint configured; using fallback scores."
            if self._load_error is not None:
                message = f"Model load failed; using fallback scores: {self._load_error}"
            return RuntimeScoringResult(
                scores=_fallback_scores(len(dataset.event_labels), self.fallback_score),
                used_fallback=True,
                message=message,
            )

        try:
            scores = score_pcoord_lineage_dataset(
                model=self._model,
                dataset=dataset,
                batch_size=self.batch_size,
                device=self.device,
            )
        except Exception as exc:
            return RuntimeScoringResult(
                scores=_fallback_scores(len(dataset.event_labels), self.fallback_score),
                used_fallback=True,
                message=f"Model scoring failed; using fallback scores: {exc}",
            )

        return RuntimeScoringResult(
            scores=scores,
            used_fallback=False,
            message="Scored with STRIDE pcoord-lineage checkpoint.",
        )

    def assign_bins(
        self,
        scoring_input: PcoordRuntimeScoringInput,
        mapper: StrideValueBinMapper,
        score_key: str = "stride_score",
    ) -> tuple[np.ndarray, RuntimeScoringResult]:
        result = self.score(scoring_input)
        if score_key not in result.scores:
            raise KeyError(f"Runtime scores do not contain {score_key!r}.")
        assignments = mapper.assign(result.scores[score_key])
        return assignments, result


def _fallback_scores(num_examples: int, fallback_score: float) -> dict[str, np.ndarray]:
    values = np.full((num_examples,), fallback_score, dtype=np.float32)
    return {
        "p_event": values.copy(),
        "flux_value": np.zeros((num_examples,), dtype=np.float32),
        "uncertainty": np.ones((num_examples,), dtype=np.float32),
        "stride_score": values.copy(),
    }
