from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stride.models.gru_ranker import GRURanker


@dataclass
class BinAssignment:
    """
    Output from STRIDE's learned binning interface.
    """

    bin_id: int
    score: float


class StrideScoreBinMapper:
    """
    Prototype learned BinMapper for STRIDE.

    This is not yet a full WESTPA plugin, but it has the core behavior:

        trajectory window -> trained model -> event probability -> bin ID

    For Version 2, this uses the NaCl GRU checkpoint.

    Later, this class can be wrapped/adapted into an actual WESTPA BinMapper.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        num_bins: int = 8,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.num_bins = num_bins

        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2.")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model, self.mean, self.std = self._load_checkpoint(self.checkpoint_path)

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> tuple[GRURanker, np.ndarray, np.ndarray]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                "Run `python scripts/train_nacl.py` first."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        model = GRURanker(
            num_features=int(checkpoint["num_features"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        mean = checkpoint["mean"]
        std = checkpoint["std"]

        return model, mean, std

    def score_window(self, window: np.ndarray) -> float:
        """
        Score one trajectory window.

        Args:
            window:
                Shape [window_size, num_features]

        Returns:
            Event probability score in [0, 1].
        """
        window = np.asarray(window, dtype=np.float32)

        if window.ndim != 2:
            raise ValueError(
                f"Expected window shape [window_size, num_features], got {window.shape}"
            )

        # Add batch dimension.
        X = window[None, :, :]

        X_norm = ((X - self.mean) / self.std).astype(np.float32)

        x_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():
            logits = self.model(x_tensor)
            prob = torch.sigmoid(logits).item()

        return float(prob)

    def score_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Score a batch of trajectory windows.

        Args:
            windows:
                Shape [num_windows, window_size, num_features]

        Returns:
            scores:
                Shape [num_windows]
        """
        windows = np.asarray(windows, dtype=np.float32)

        if windows.ndim != 3:
            raise ValueError(
                f"Expected windows shape [batch, window_size, num_features], "
                f"got {windows.shape}"
            )

        X_norm = ((windows - self.mean) / self.std).astype(np.float32)

        x_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs.astype(np.float32)

    def score_to_bin(self, score: float) -> int:
        """
        Convert event probability score to a discrete bin ID.

        Version 2 uses uniform probability bins:

            score in [0.00, 0.125) -> bin 0
            score in [0.125, 0.25) -> bin 1
            ...
            score in [0.875, 1.00] -> bin 7

        Later versions can replace this with quantile bins or learned bin edges.
        """
        score = float(score)

        if score < 0.0 or score > 1.0:
            raise ValueError(f"Score must be in [0, 1], got {score}")

        bin_id = int(np.floor(score * self.num_bins))

        # score = 1.0 would otherwise map to num_bins.
        bin_id = min(bin_id, self.num_bins - 1)

        return bin_id

    def assign(self, window: np.ndarray) -> BinAssignment:
        """
        Assign one trajectory window to a learned STRIDE bin.

        Args:
            window:
                Shape [window_size, num_features]

        Returns:
            BinAssignment containing bin ID and model score.
        """
        score = self.score_window(window)
        bin_id = self.score_to_bin(score)

        return BinAssignment(
            bin_id=bin_id,
            score=score,
        )

    def assign_batch(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign a batch of windows to bins.

        Args:
            windows:
                Shape [batch, window_size, num_features]

        Returns:
            bin_ids:
                Shape [batch]
            scores:
                Shape [batch]
        """
        scores = self.score_batch(windows)

        bin_ids = np.array(
            [self.score_to_bin(float(score)) for score in scores],
            dtype=int,
        )

        return bin_ids, scores


class StrideQuantileBinMapper(StrideScoreBinMapper):
    """
    Quantile-edge learned BinMapper.

    Unlike StrideScoreBinMapper, which uses fixed probability-width bins,
    this mapper learns bin edges from a reference score distribution.

    This is often better when model scores are not uniformly distributed.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        reference_windows: np.ndarray,
        num_bins: int = 8,
        device: str | None = None,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            num_bins=num_bins,
            device=device,
        )

        reference_scores = self.score_batch(reference_windows)
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)

        # Internal edges only.
        self.edges = np.quantile(reference_scores, quantiles[1:-1])
        self.edges = np.unique(self.edges)

    def score_to_bin(self, score: float) -> int:
        score = float(score)

        if score < 0.0 or score > 1.0:
            raise ValueError(f"Score must be in [0, 1], got {score}")

        bin_id = int(np.digitize(score, self.edges, right=False))

        return bin_id


if __name__ == "__main__":
    # Basic import/load sanity check.
    mapper = StrideScoreBinMapper(
        checkpoint_path="checkpoints/nacl_gru.pt",
        num_bins=8,
    )

    print("Loaded STRIDE NaCl BinMapper")
    print(f"Device: {mapper.device}")
    print(f"Number of bins: {mapper.num_bins}")