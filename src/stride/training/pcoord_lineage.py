from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stride.models import PcoordLineageModelConfig, PcoordLineageValueModel
from stride.training.atomistic import (
    _is_better_metric,
    _make_scheduler,
    _metadata_float,
    _resolve_event_positive_weight,
    _step_scheduler,
    _validate_best_mode,
    resolve_device,
)
from stride.training.metrics import compute_binary_metrics, top_k_enrichment
from stride.training.stride_value import (
    StrideValueLossConfig,
    StrideValueTargets,
    stride_value_loss,
)
from stride.training.westpa_evaluation import westpa_iteration_split_indices


@dataclass(frozen=True)
class PcoordLineageDataset:
    pcoord_windows: np.ndarray
    window_mask: np.ndarray
    goal_features: np.ndarray
    event_labels: np.ndarray
    flux_labels: np.ndarray
    n_iter: np.ndarray
    seg_id: np.ndarray
    weights: np.ndarray | None = None

    def validate(self) -> None:
        if self.pcoord_windows.ndim != 3:
            raise ValueError("pcoord_windows must have shape [examples, window, dims].")
        num_examples, window_size, _ = self.pcoord_windows.shape
        if self.window_mask.shape != (num_examples, window_size):
            raise ValueError("window_mask must match pcoord_windows first two dimensions.")
        if self.goal_features.shape[0] != num_examples:
            raise ValueError("goal_features must have one row per example.")
        if self.event_labels.shape != (num_examples,):
            raise ValueError("event_labels must have shape [examples].")
        if self.flux_labels.shape != (num_examples,):
            raise ValueError("flux_labels must have shape [examples].")
        if self.n_iter.shape != (num_examples,):
            raise ValueError("n_iter must have shape [examples].")
        if self.seg_id.shape != (num_examples,):
            raise ValueError("seg_id must have shape [examples].")
        if self.weights is not None and self.weights.shape != (num_examples,):
            raise ValueError("weights must have shape [examples].")
        if not np.all(np.any(self.window_mask, axis=1)):
            raise ValueError("Every pcoord window must contain at least one valid frame.")


def load_pcoord_lineage_dataset_npz(path: str | Path) -> PcoordLineageDataset:
    data = np.load(path)
    required = (
        "pcoord_windows",
        "window_mask",
        "goal_features",
        "event_labels",
        "flux_labels",
        "n_iter",
        "seg_id",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Lineage artifact missing required arrays: {missing}")
    weights = data["weights"].astype(np.float64) if "weights" in data else None
    dataset = PcoordLineageDataset(
        pcoord_windows=data["pcoord_windows"].astype(np.float32),
        window_mask=data["window_mask"].astype(bool),
        goal_features=data["goal_features"].astype(np.float32),
        event_labels=data["event_labels"].astype(np.float32),
        flux_labels=data["flux_labels"].astype(np.float32),
        n_iter=data["n_iter"].astype(np.int64),
        seg_id=data["seg_id"].astype(np.int64),
        weights=weights,
    )
    dataset.validate()
    return dataset


def load_lineage_dataset_and_make_config(
    dataset_path: str | Path,
    hidden_dim: int = 64,
    transformer_layers: int = 1,
    transformer_heads: int = 4,
    dropout: float = 0.1,
) -> tuple[PcoordLineageDataset, PcoordLineageModelConfig]:
    dataset = load_pcoord_lineage_dataset_npz(dataset_path)
    config = PcoordLineageModelConfig(
        pcoord_dim=dataset.pcoord_windows.shape[-1],
        goal_feature_dim=dataset.goal_features.shape[-1],
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        dropout=dropout,
    )
    return dataset, config


def describe_pcoord_lineage_split(
    dataset: PcoordLineageDataset,
    validation_fraction: float = 0.2,
    seed: int = 7,
    split_strategy: str = "tail",
) -> dict[str, float]:
    dataset.validate()
    train_indices, val_indices = split_pcoord_lineage_indices(
        dataset,
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
    )
    train_labels = dataset.event_labels[train_indices]
    val_labels = dataset.event_labels[val_indices] if len(val_indices) else np.array([])
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


def split_pcoord_lineage_indices(
    dataset: PcoordLineageDataset,
    validation_fraction: float = 0.2,
    seed: int = 7,
    split_strategy: str = "tail",
) -> tuple[np.ndarray, np.ndarray]:
    dataset.validate()
    return westpa_iteration_split_indices(
        dataset.n_iter,
        validation_fraction=validation_fraction,
        split_strategy=split_strategy,
        seed=seed,
    )


def train_pcoord_lineage_value_model(
    dataset: PcoordLineageDataset,
    config: PcoordLineageModelConfig,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.2,
    seed: int = 7,
    device: str | None = None,
    split_strategy: str = "tail",
    event_positive_weight: float | str = "auto",
    progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
    best_checkpoint_path: str | Path | None = None,
    save_best_metric: str = "val_auprc",
    save_best_mode: str = "max",
    resume_from: str | Path | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    lr_scheduler: str | None = None,
) -> tuple[PcoordLineageValueModel, dict[str, float]]:
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

    model = PcoordLineageValueModel(config).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = _make_scheduler(optimizer, lr_scheduler, epochs, save_best_mode)
    start_epoch = 1
    best_metric_value: float | None = None
    epochs_without_improvement = 0

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device_obj)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        previous_epoch = int(checkpoint.get("epoch", checkpoint.get("metrics", {}).get("epoch", 0)))
        start_epoch = previous_epoch + 1
        best_metric_value = _metadata_float(checkpoint.get("metadata", {}), "best_metric_value")
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
    if early_stopping_patience is not None and early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch in train_loader:
            pcoord_windows, window_mask, goal_features, event, flux = [
                tensor.to(device_obj) for tensor in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pcoord_windows=pcoord_windows,
                window_mask=window_mask,
                goal_features=goal_features,
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
        metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])

        metric_value = metrics.get(save_best_metric)
        is_best = _is_better_metric(
            metric_value,
            best_metric_value,
            save_best_mode,
            min_delta=early_stopping_min_delta,
        )
        if is_best:
            best_metric_value = float(metric_value)
            epochs_without_improvement = 0
            metrics["best_metric_value"] = best_metric_value
            metrics["best_metric_epoch"] = float(epoch)
            if best_checkpoint_path is not None:
                save_pcoord_lineage_checkpoint(
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
                    scheduler=scheduler,
                    epoch=epoch,
                )
        elif best_metric_value is not None:
            epochs_without_improvement += 1
            metrics["best_metric_value"] = best_metric_value

        _step_scheduler(scheduler, lr_scheduler, metric_value)
        metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        metrics["epochs_without_improvement"] = float(epochs_without_improvement)

        if progress_callback is not None:
            progress_callback(epoch, epochs, dict(metrics))

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            metrics["early_stopped"] = 1.0
            break

    if checkpoint_path is not None:
        save_pcoord_lineage_checkpoint(
            checkpoint_path,
            model,
            metrics,
            metadata={
                **(checkpoint_metadata or {}),
                "checkpoint_type": "final",
                "best_metric": save_best_metric,
                "best_metric_mode": save_best_mode,
                "best_metric_value": best_metric_value,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "lr_scheduler": lr_scheduler or "none",
            },
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(metrics.get("epoch", 0)),
        )

    return model, metrics


def score_pcoord_lineage_dataset(
    model: PcoordLineageValueModel,
    dataset: PcoordLineageDataset,
    batch_size: int = 256,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    dataset.validate()
    device_obj = resolve_device(device)
    model = model.to(device_obj)
    model.eval()

    loader = DataLoader(_to_tensor_dataset(dataset), batch_size=batch_size, shuffle=False)
    outputs: dict[str, list[np.ndarray]] = {
        "p_event": [],
        "flux_value": [],
        "uncertainty": [],
        "stride_score": [],
    }
    with torch.no_grad():
        for batch in loader:
            pcoord_windows, window_mask, goal_features, _, _ = [
                tensor.to(device_obj) for tensor in batch
            ]
            batch_outputs = model(
                pcoord_windows=pcoord_windows,
                window_mask=window_mask,
                goal_features=goal_features,
            )
            for key in outputs:
                outputs[key].append(batch_outputs[key].detach().cpu().numpy())

    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}


def save_pcoord_lineage_checkpoint(
    path: str | Path,
    model: PcoordLineageValueModel,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, object] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
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
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    torch.save(checkpoint, path)


def load_pcoord_lineage_checkpoint(
    path: str | Path,
    device: str | None = None,
) -> tuple[PcoordLineageValueModel, dict[str, float]]:
    device_obj = resolve_device(device)
    checkpoint = torch.load(path, map_location=device_obj)
    config = PcoordLineageModelConfig(**checkpoint["config"])
    model = PcoordLineageValueModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    return model, dict(checkpoint.get("metrics", {}))


def _make_loaders(
    dataset: PcoordLineageDataset,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    split_strategy: str,
) -> tuple[DataLoader, DataLoader | None]:
    tensor_dataset = _to_tensor_dataset(dataset)
    train_indices, val_indices = split_pcoord_lineage_indices(
        dataset=dataset,
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


def _to_tensor_dataset(dataset: PcoordLineageDataset) -> TensorDataset:
    return TensorDataset(
        torch.tensor(dataset.pcoord_windows, dtype=torch.float32),
        torch.tensor(dataset.window_mask, dtype=torch.bool),
        torch.tensor(dataset.goal_features, dtype=torch.float32),
        torch.tensor(dataset.event_labels, dtype=torch.float32),
        torch.tensor(dataset.flux_labels, dtype=torch.float32),
    )


def _evaluate_model(
    model: PcoordLineageValueModel,
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
            pcoord_windows, window_mask, goal_features, event, flux = [
                tensor.to(device) for tensor in batch
            ]
            outputs = model(
                pcoord_windows=pcoord_windows,
                window_mask=window_mask,
                goal_features=goal_features,
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
