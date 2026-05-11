from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stride.data import AtomisticDataset, load_atomistic_dataset_npz
from stride.models import StrideModelConfig, StrideValueModel
from stride.training.metrics import compute_binary_metrics, top_k_enrichment
from stride.training.stride_value import (
    StrideValueLossConfig,
    StrideValueTargets,
    stride_value_loss,
)


def train_atomistic_value_model(
    dataset: AtomisticDataset,
    config: StrideModelConfig,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 7,
    device: str | None = None,
    split_strategy: str = "contiguous",
    event_positive_weight: float | str = "auto",
    progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
    best_checkpoint_path: str | Path | None = None,
    save_best_metric: str = "val_auroc",
    save_best_mode: str = "max",
    resume_from: str | Path | None = None,
) -> tuple[StrideValueModel, dict[str, float]]:
    """
    Train STRIDE's goal-conditioned value model on an AtomisticDataset.
    """
    dataset.validate()
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device_obj = resolve_device(device)
    train_loader, val_loader = _make_loaders(
        dataset=dataset,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
    )

    model = StrideValueModel(config).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_epoch = 1
    best_metric_value: float | None = None

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device_obj)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        previous_epoch = int(checkpoint.get("epoch", checkpoint.get("metrics", {}).get("epoch", 0)))
        start_epoch = previous_epoch + 1
        best_metric_value = _metadata_float(
            checkpoint.get("metadata", {}),
            "best_metric_value",
        )
        if start_epoch > epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {previous_epoch}, "
                f"but requested epochs={epochs}."
            )

    loss_config = StrideValueLossConfig(
        event_positive_weight=_resolve_event_positive_weight(
            dataset.event_labels,
            event_positive_weight,
        )
    )
    metrics: dict[str, float] = {}

    _validate_best_mode(save_best_mode)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch in train_loader:
            coordinates, atom_features, atom_mask, frame_mask, goal_features, event, flux = [
                tensor.to(device_obj) for tensor in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                coordinates=coordinates,
                atom_features=atom_features,
                goal_features=goal_features,
                atom_mask=atom_mask,
                frame_mask=frame_mask,
            )
            loss, loss_metrics = stride_value_loss(
                outputs,
                StrideValueTargets(event=event, flux=flux),
                config=loss_config,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_metrics["loss"])

        metrics = {"epoch": float(epoch), "train_loss": float(np.mean(epoch_losses))}
        if val_loader is not None:
            metrics.update(
                _evaluate_model(
                    model,
                    val_loader,
                    device_obj,
                    prefix="val_",
                    loss_config=loss_config,
                )
            )
        metrics["event_positive_weight"] = float(loss_config.event_positive_weight)
        metrics["start_epoch"] = float(start_epoch)

        metric_value = metrics.get(save_best_metric)
        is_best = _is_better_metric(metric_value, best_metric_value, save_best_mode)
        if is_best:
            best_metric_value = float(metric_value)
            metrics["best_metric_value"] = best_metric_value
            metrics["best_metric_epoch"] = float(epoch)
            if best_checkpoint_path is not None:
                save_atomistic_checkpoint(
                    best_checkpoint_path,
                    model,
                    metrics,
                    metadata={
                        **(checkpoint_metadata or {}),
                        "checkpoint_type": "best",
                        "best_metric": save_best_metric,
                        "best_metric_mode": save_best_mode,
                        "best_metric_value": best_metric_value,
                        "best_metric_epoch": epoch,
                    },
                    optimizer=optimizer,
                    epoch=epoch,
                )
        elif best_metric_value is not None:
            metrics["best_metric_value"] = best_metric_value

        if progress_callback is not None:
            progress_callback(epoch, epochs, dict(metrics))

    if checkpoint_path is not None:
        save_atomistic_checkpoint(
            checkpoint_path,
            model,
            metrics,
            metadata={
                **(checkpoint_metadata or {}),
                "checkpoint_type": "final",
                "best_metric": save_best_metric,
                "best_metric_mode": save_best_mode,
                "best_metric_value": best_metric_value,
            },
            optimizer=optimizer,
            epoch=int(metrics.get("epoch", 0)),
        )

    return model, metrics


def score_atomistic_dataset(
    model: StrideValueModel,
    dataset: AtomisticDataset,
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Score every example in an AtomisticDataset with a trained STRIDE model.
    """
    dataset.validate()
    device_obj = resolve_device(device)
    model = model.to(device_obj)
    model.eval()

    tensor_dataset = _to_tensor_dataset(dataset)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    outputs: dict[str, list[np.ndarray]] = {
        "p_event": [],
        "flux_value": [],
        "uncertainty": [],
        "stride_score": [],
    }

    with torch.no_grad():
        for batch in loader:
            coordinates, atom_features, atom_mask, frame_mask, goal_features, _, _ = [
                tensor.to(device_obj) for tensor in batch
            ]
            batch_outputs = model(
                coordinates=coordinates,
                atom_features=atom_features,
                goal_features=goal_features,
                atom_mask=atom_mask,
                frame_mask=frame_mask,
            )
            for key in outputs:
                outputs[key].append(batch_outputs[key].detach().cpu().numpy())

    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}


def save_atomistic_checkpoint(
    path: str | Path,
    model: StrideValueModel,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, object] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
        "metrics": metrics or {},
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    torch.save(checkpoint, path)


def load_atomistic_checkpoint(
    path: str | Path,
    device: str | None = None,
) -> tuple[StrideValueModel, dict[str, float]]:
    device_obj = resolve_device(device)
    checkpoint = torch.load(path, map_location=device_obj)
    config = StrideModelConfig(**checkpoint["config"])
    model = StrideValueModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    return model, dict(checkpoint.get("metrics", {}))


def resolve_device(device: str | None = None) -> torch.device:
    """
    Resolve auto/cpu/cuda/mps device selection for local and remote training.
    """
    value = (device or "auto").lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if value == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
    return torch.device(value)


def _resolve_event_positive_weight(
    event_labels: np.ndarray,
    event_positive_weight: float | str,
) -> float:
    if isinstance(event_positive_weight, str):
        if event_positive_weight != "auto":
            raise ValueError("event_positive_weight must be a float or 'auto'.")
        positive = float(np.sum(event_labels >= 0.5))
        negative = float(np.sum(event_labels < 0.5))
        if positive == 0.0:
            return 1.0
        return max(1.0, negative / positive)

    value = float(event_positive_weight)
    if value <= 0.0:
        raise ValueError("event_positive_weight must be positive.")
    return value


def load_dataset_and_make_config(
    dataset_path: str | Path,
    hidden_dim: int = 128,
    egnn_layers: int = 3,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    dropout: float = 0.1,
    radius: float | None = None,
) -> tuple[AtomisticDataset, StrideModelConfig]:
    dataset = load_atomistic_dataset_npz(dataset_path)
    config = StrideModelConfig(
        atom_feature_dim=dataset.atom_features.shape[-1],
        goal_feature_dim=dataset.goal_features.shape[-1],
        hidden_dim=hidden_dim,
        egnn_layers=egnn_layers,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        dropout=dropout,
        radius=radius,
    )
    return dataset, config


def describe_atomistic_split(
    dataset: AtomisticDataset,
    validation_fraction: float = 0.2,
    seed: int = 7,
    split_strategy: str = "contiguous",
) -> dict[str, float]:
    """
    Summarize train/validation event balance before training.
    """
    dataset.validate()
    train_indices, val_indices = _split_indices(
        num_examples=len(dataset.event_labels),
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
    )

    train_labels = dataset.event_labels[train_indices.numpy()]
    val_labels = dataset.event_labels[val_indices.numpy()] if len(val_indices) else np.array([])

    return {
        "num_examples": float(len(dataset.event_labels)),
        "positive_rate": float(np.mean(dataset.event_labels)),
        "train_examples": float(len(train_indices)),
        "train_positive_rate": float(np.mean(train_labels)) if len(train_labels) else float("nan"),
        "train_positives": float(np.sum(train_labels >= 0.5)),
        "val_examples": float(len(val_indices)),
        "val_positive_rate": float(np.mean(val_labels)) if len(val_labels) else float("nan"),
        "val_positives": float(np.sum(val_labels >= 0.5)) if len(val_labels) else 0.0,
    }


def _make_loaders(
    dataset: AtomisticDataset,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    split_strategy: str,
) -> tuple[DataLoader, DataLoader | None]:
    tensor_dataset = _to_tensor_dataset(dataset)
    train_indices, val_indices = _split_indices(
        num_examples=len(tensor_dataset),
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
    )

    train_subset = torch.utils.data.Subset(tensor_dataset, train_indices.tolist())
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if len(val_indices) > 0:
        val_subset = torch.utils.data.Subset(tensor_dataset, val_indices.tolist())
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def _split_indices(
    num_examples: int,
    validation_fraction: float,
    seed: int,
    split_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    val_count = int(round(num_examples * validation_fraction))
    if num_examples < 3:
        val_count = 0
    else:
        val_count = min(max(val_count, 1), num_examples - 1)

    if split_strategy == "random":
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(num_examples, generator=generator)
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
    elif split_strategy == "contiguous":
        split_at = num_examples - val_count
        train_indices = torch.arange(0, split_at)
        val_indices = torch.arange(split_at, num_examples)
    else:
        raise ValueError("split_strategy must be 'contiguous' or 'random'.")

    return train_indices, val_indices


def _to_tensor_dataset(dataset: AtomisticDataset) -> TensorDataset:
    return TensorDataset(
        torch.tensor(dataset.coordinates, dtype=torch.float32),
        torch.tensor(dataset.atom_features, dtype=torch.float32),
        torch.tensor(dataset.atom_mask, dtype=torch.bool),
        torch.tensor(dataset.frame_mask, dtype=torch.bool),
        torch.tensor(dataset.goal_features, dtype=torch.float32),
        torch.tensor(dataset.event_labels, dtype=torch.float32),
        torch.tensor(dataset.flux_labels, dtype=torch.float32),
    )


def _evaluate_model(
    model: StrideValueModel,
    loader: DataLoader,
    device: torch.device,
    prefix: str,
    loss_config: StrideValueLossConfig | None = None,
) -> dict[str, float]:
    model.eval()
    batch_losses: list[dict[str, float]] = []
    score_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            coordinates, atom_features, atom_mask, frame_mask, goal_features, event, flux = [
                tensor.to(device) for tensor in batch
            ]
            outputs = model(
                coordinates=coordinates,
                atom_features=atom_features,
                goal_features=goal_features,
                atom_mask=atom_mask,
                frame_mask=frame_mask,
            )
            _, loss_metrics = stride_value_loss(
                outputs,
                StrideValueTargets(event=event, flux=flux),
                config=loss_config,
            )
            batch_losses.append(loss_metrics)
            score_batches.append(outputs["p_event"].detach().cpu().numpy())
            label_batches.append(event.detach().cpu().numpy())

    scores = np.concatenate(score_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    binary_metrics = compute_binary_metrics(labels, scores)
    enrichment = top_k_enrichment(labels, scores, k=0.25)

    loss_metrics = {
        key: float(np.mean([batch[key] for batch in batch_losses]))
        for key in batch_losses[0]
    }
    metrics = {f"{prefix}{key}": value for key, value in loss_metrics.items()}
    metrics.update({f"{prefix}{key}": value for key, value in binary_metrics.items()})
    metrics[f"{prefix}top25_enrichment"] = float(enrichment)
    return metrics


def _is_better_metric(
    metric_value: float | None,
    best_metric_value: float | None,
    mode: str,
) -> bool:
    if metric_value is None or not np.isfinite(metric_value):
        return False
    if best_metric_value is None or not np.isfinite(best_metric_value):
        return True
    if mode == "max":
        return metric_value > best_metric_value
    if mode == "min":
        return metric_value < best_metric_value
    raise ValueError("mode must be 'max' or 'min'.")


def _validate_best_mode(mode: str) -> None:
    if mode not in {"max", "min"}:
        raise ValueError("save_best_mode must be 'max' or 'min'.")


def _metadata_float(metadata: object, key: str) -> float | None:
    if not isinstance(metadata, dict) or key not in metadata:
        return None
    value = metadata[key]
    if value is None:
        return None
    return float(value)
