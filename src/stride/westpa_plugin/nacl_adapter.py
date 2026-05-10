from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stride.features.nacl import compute_nacl_features
from stride.westpa_plugin.stride_binmapper import (
    BinAssignment,
    StrideQuantileBinMapper,
    StrideScoreBinMapper,
)


@dataclass
class NaClAdapterConfig:
    """
    Configuration for adapting NaCl pcoord histories into STRIDE windows.

    In WESTPA's NaCl tutorial, the progress coordinate is typically the
    Na-Cl distance. This adapter treats a recent distance history as a
    pcoord-like input and converts it into the feature format expected by
    the trained STRIDE NaCl GRU.
    """

    dt: float = 0.02
    window_size: int = 25
    num_bins: int = 8


class NaClWESTPAAdapter:
    """
    WESTPA-style adapter for STRIDE's NaCl learned BinMapper.

    This class is not yet a full WESTPA plugin. It is a bridge object that
    accepts pcoord-like Na-Cl distance histories and returns learned bin IDs.

    Input:
        distance_history: [window_size] or longer

    Processing:
        distance history -> NaCl features -> GRU score -> bin ID

    Output:
        BinAssignment(bin_id, score)
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "checkpoints/nacl_gru.pt",
        config: NaClAdapterConfig | None = None,
        mode: str = "fixed",
        reference_windows: np.ndarray | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or NaClAdapterConfig()
        self.mode = mode

        if mode == "fixed":
            self.mapper = StrideScoreBinMapper(
                checkpoint_path=checkpoint_path,
                num_bins=self.config.num_bins,
                device=device,
            )

        elif mode == "quantile":
            if reference_windows is None:
                raise ValueError(
                    "reference_windows must be provided when mode='quantile'."
                )

            self.mapper = StrideQuantileBinMapper(
                checkpoint_path=checkpoint_path,
                reference_windows=reference_windows,
                num_bins=self.config.num_bins,
                device=device,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def distance_history_to_window(self, distance_history: np.ndarray) -> np.ndarray:
        """
        Convert a Na-Cl distance history into a STRIDE feature window.

        Args:
            distance_history:
                1D array of recent Na-Cl distances. It may be longer than
                window_size; the last window_size points are used.

        Returns:
            feature_window:
                Shape [window_size, num_features]
        """
        distance_history = np.asarray(distance_history, dtype=np.float32)

        if distance_history.ndim != 1:
            raise ValueError(
                f"Expected 1D distance history, got shape {distance_history.shape}"
            )

        if len(distance_history) < self.config.window_size:
            raise ValueError(
                f"Need at least {self.config.window_size} distance values, "
                f"got {len(distance_history)}."
            )

        recent_distances = distance_history[-self.config.window_size :]

        features = compute_nacl_features(
            distances=recent_distances,
            dt=self.config.dt,
        )

        return features.astype(np.float32)

    def assign_distance_history(self, distance_history: np.ndarray) -> BinAssignment:
        """
        Assign one distance history to a STRIDE learned bin.
        """
        window = self.distance_history_to_window(distance_history)
        return self.mapper.assign(window)

    def assign_distance_histories(
        self,
        distance_histories: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign multiple distance histories to STRIDE learned bins.

        Args:
            distance_histories:
                Shape [num_walkers, history_length]

        Returns:
            bin_ids:
                Shape [num_walkers]
            scores:
                Shape [num_walkers]
        """
        distance_histories = np.asarray(distance_histories, dtype=np.float32)

        if distance_histories.ndim != 2:
            raise ValueError(
                "Expected distance_histories shape [num_walkers, history_length], "
                f"got {distance_histories.shape}"
            )

        windows = np.stack(
            [
                self.distance_history_to_window(distance_history)
                for distance_history in distance_histories
            ]
        ).astype(np.float32)

        return self.mapper.assign_batch(windows)