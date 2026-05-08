from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stride.features.toy2d import Toy2DConfig, build_windows, simulate_dataset
from stride.models.gru_ranker import GRURanker
from stride.training.metrics import (
    compute_binary_metrics,
    top_k_enrichment,
    top_k_positive_rate,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Random window-level split for toy MVP.

    Later, for real trajectories, split by trajectory/run to avoid leakage.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)

    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize using train-set statistics only.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_scores.append(probs)
            all_labels.append(y_batch.numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    metrics = compute_binary_metrics(y_true, y_score)
    metrics["top10_enrichment"] = top_k_enrichment(y_true, y_score, k=0.10)
    metrics["top10_positive_rate"] = top_k_positive_rate(y_true, y_score, k=0.10)

    return metrics, y_true, y_score


def main() -> None:
    config = load_config("configs/toy2d.yaml")

    sim_cfg = Toy2DConfig(**config["simulation"])
    dataset_cfg = config["dataset"]

    training_cfg = config.get("training", {})

    batch_size = int(training_cfg.get("batch_size", 128))
    epochs = int(training_cfg.get("epochs", 20))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    hidden_dim = int(training_cfg.get("hidden_dim", 128))
    seed = int(sim_cfg.seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Generating toy trajectories...")
    trajectories = simulate_dataset(sim_cfg)

    X, y = build_windows(
        trajectories=trajectories,
        cfg=sim_cfg,
        window_size=dataset_cfg["window_size"],
        horizon=dataset_cfg["horizon"],
        stride=dataset_cfg["stride"],
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Window positive rate: {y.mean():.3f}")
    print(f"Event type: {sim_cfg.event_type}")
    if sim_cfg.event_type == "upper_gate":
        print(f"Gate y min: {sim_cfg.gate_y_min}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X,
        y,
        train_frac=float(training_cfg.get("train_split", 0.7)),
        val_frac=float(training_cfg.get("val_split", 0.15)),
        seed=seed,
    )

    X_train, X_val, X_test, mean, std = normalize_features(
        X_train,
        X_val,
        X_test,
    )

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    num_features = X.shape[-1]

    model = GRURanker(
        num_features=num_features,
        hidden_dim=hidden_dim,
    ).to(device)

    num_positive = float(y_train.sum())
    num_negative = float(len(y_train) - num_positive)

    if num_positive == 0:
        raise RuntimeError("Training set has zero positive examples.")

    pos_weight = torch.tensor(num_negative / num_positive, dtype=torch.float32).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Val positive rate: {y_val.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")
    print(f"pos_weight: {pos_weight.item():.2f}")

    best_val_auprc = -1.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "toy_gru.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size_actual = len(X_batch)
            total_loss += float(loss.item()) * batch_size_actual
            total_examples += batch_size_actual

        train_loss = total_loss / total_examples

        val_metrics, _, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_auroc={val_metrics['auroc']:.3f} | "
            f"val_auprc={val_metrics['auprc']:.3f} | "
            f"top10_enrichment={val_metrics['top10_enrichment']:.2f}x | "
            f"top10_pos={val_metrics['top10_positive_rate']:.3f}"
        )

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_features": num_features,
                    "hidden_dim": hidden_dim,
                    "mean": mean,
                    "std": std,
                    "config": config,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

    print(f"Best checkpoint saved to {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, _, _ = evaluate(model, test_loader, device)

    print("\nFinal test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()