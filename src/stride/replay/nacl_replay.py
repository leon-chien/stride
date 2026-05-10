from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stride.binning.quantile_binner import QuantileBinner, summarize_bins
from stride.models.gru_ranker import GRURanker
from stride.training.metrics import (
    compute_binary_metrics,
    top_k_enrichment,
    top_k_positive_rate,
)


def load_nacl_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load NaCl windows and labels from a saved .npz file.

    Expected keys:
        X: [num_examples, window_size, num_features]
        y: [num_examples]
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"NaCl dataset not found at {dataset_path}. "
            "Run `python scripts/run_nacl_dataset.py` first."
        )

    data = np.load(dataset_path)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    return X, y


def load_nacl_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[GRURanker, np.ndarray, np.ndarray, dict]:
    """
    Load trained NaCl GRU model and normalization stats.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run `python scripts/train_nacl.py` first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = GRURanker(
        num_features=int(checkpoint["num_features"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    return model, mean, std, checkpoint


def normalize_with_checkpoint_stats(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Normalize replay data using training-set statistics saved in checkpoint.
    """
    return ((X - mean) / std).astype(np.float32)


def score_windows(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Run frozen model inference on NaCl trajectory windows.

    Returns:
        probabilities: shape [num_examples]
    """
    x_tensor = torch.tensor(X, dtype=torch.float32)
    dummy_y = torch.zeros(len(X), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x_tensor, dummy_y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_scores: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            all_scores.append(probs.cpu().numpy())

    return np.concatenate(all_scores)


def random_selection_positive_rate(
    y_true: np.ndarray,
    k: float = 0.10,
    seed: int = 42,
) -> float:
    """
    Estimate positive rate from randomly selecting k fraction of examples.
    """
    if not 0.0 < k <= 1.0:
        raise ValueError(f"k must be in (0, 1], got {k}")

    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true).astype(int)

    n = len(y_true)
    top_n = max(1, int(np.ceil(k * n)))

    selected = rng.choice(n, size=top_n, replace=False)

    return float(y_true[selected].mean())


def run_nacl_replay(
    dataset_path: str | Path = "outputs/nacl/nacl_dataset.npz",
    checkpoint_path: str | Path = "checkpoints/nacl_gru.pt",
    batch_size: int = 512,
    num_bins: int = 4,
    top_k: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Offline replay analysis for the synthetic NaCl benchmark.

    This asks:
        If STRIDE used the trained NaCl GRU to rank trajectory windows,
        how enriched would the selected high-value windows be for future
        association events?

    Note:
        This is a frozen-model inference audit. No model training happens here.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading dataset: {dataset_path}")
    X, y = load_nacl_dataset(dataset_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, mean, std, checkpoint = load_nacl_checkpoint(checkpoint_path, device)

    X_norm = normalize_with_checkpoint_stats(X, mean, std)

    print("Scoring NaCl trajectory windows...")
    scores = score_windows(
        model=model,
        X=X_norm,
        device=device,
        batch_size=batch_size,
    )

    metrics = compute_binary_metrics(y, scores)
    metrics["top10_enrichment"] = top_k_enrichment(y, scores, k=top_k)
    metrics["top10_positive_rate"] = top_k_positive_rate(y, scores, k=top_k)
    metrics["random_top10_positive_rate"] = random_selection_positive_rate(
        y,
        k=top_k,
        seed=seed,
    )

    binner = QuantileBinner(num_bins=num_bins)
    bin_ids = binner.fit_transform(scores)
    bin_summary = summarize_bins(bin_ids, y, scores)

    results = {
        "metrics": metrics,
        "bin_summary": bin_summary,
        "bin_edges": binner.edges_,
        "scores": scores,
        "labels": y,
        "checkpoint_val_metrics": checkpoint.get("val_metrics", {}),
    }

    return results