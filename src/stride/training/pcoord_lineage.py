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
    cell_id: np.ndarray | None = None
    goal_id: np.ndarray | None = None
    pcoord_dim: np.ndarray | None = None
    threshold: np.ndarray | None = None
    horizon_iterations: np.ndarray | None = None

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
        for name, value in (
            ("cell_id", self.cell_id),
            ("goal_id", self.goal_id),
            ("pcoord_dim", self.pcoord_dim),
            ("threshold", self.threshold),
            ("horizon_iterations", self.horizon_iterations),
        ):
            if value is not None and value.shape != (num_examples,):
                raise ValueError(f"{name} must have shape [examples].")
        if not np.all(np.any(self.window_mask, axis=1)):
            raise ValueError("Every pcoord window must contain at least one valid frame.")


PcoordFeatureTransform = dict[str, object]


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
        cell_id=data["cell_id"].astype(str) if "cell_id" in data else None,
        goal_id=data["goal_id"].astype(str) if "goal_id" in data else None,
        pcoord_dim=data["pcoord_dim"].astype(np.int64) if "pcoord_dim" in data else None,
        threshold=data["threshold"].astype(np.float32) if "threshold" in data else None,
        horizon_iterations=(
            data["horizon_iterations"].astype(np.int64)
            if "horizon_iterations" in data
            else None
        ),
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
        goal_id=dataset.goal_id,
        cell_id=dataset.cell_id,
    )


def build_pcoord_feature_transform(
    dataset: PcoordLineageDataset,
    train_indices: np.ndarray,
    mode: str = "engineered",
) -> PcoordFeatureTransform:
    """
    Build train-calibrated pcoord feature normalization metadata.
    """
    dataset.validate()
    if mode not in {"raw", "engineered"}:
        raise ValueError("feature_mode must be 'raw' or 'engineered'.")
    raw_dim = int(dataset.pcoord_windows.shape[-1])
    if mode == "raw":
        return {
            "mode": "raw",
            "raw_pcoord_dim": raw_dim,
            "feature_dim": raw_dim,
        }

    train_indices = np.asarray(train_indices, dtype=np.int64)
    train_windows = dataset.pcoord_windows[train_indices]
    train_mask = dataset.window_mask[train_indices]
    valid_values = train_windows[train_mask]
    if valid_values.size == 0:
        raise ValueError("Cannot build pcoord feature transform without valid train frames.")
    mean = np.mean(valid_values, axis=0).astype(np.float32)
    std = np.std(valid_values, axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    feature_dim = raw_dim * 8 + 3
    return {
        "mode": "engineered",
        "raw_pcoord_dim": raw_dim,
        "feature_dim": int(feature_dim),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def transform_pcoord_lineage_dataset(
    dataset: PcoordLineageDataset,
    transform: PcoordFeatureTransform | None,
) -> PcoordLineageDataset:
    """
    Apply raw or engineered pcoord features while preserving labels/provenance.
    """
    dataset.validate()
    transform = transform or {"mode": "raw"}
    mode = str(transform.get("mode", "raw"))
    if mode == "raw":
        return dataset
    if mode != "engineered":
        raise ValueError(f"Unknown pcoord feature transform mode: {mode}")

    features = engineer_pcoord_window_features(
        dataset.pcoord_windows,
        dataset.window_mask,
        transform=transform,
        pcoord_dim=dataset.pcoord_dim,
        threshold=dataset.threshold,
    )
    return PcoordLineageDataset(
        pcoord_windows=features,
        window_mask=dataset.window_mask,
        goal_features=dataset.goal_features,
        event_labels=dataset.event_labels,
        flux_labels=dataset.flux_labels,
        n_iter=dataset.n_iter,
        seg_id=dataset.seg_id,
        weights=dataset.weights,
        cell_id=dataset.cell_id,
        goal_id=dataset.goal_id,
        pcoord_dim=dataset.pcoord_dim,
        threshold=dataset.threshold,
        horizon_iterations=dataset.horizon_iterations,
    )


def engineer_pcoord_window_features(
    pcoord_windows: np.ndarray,
    window_mask: np.ndarray,
    transform: PcoordFeatureTransform,
    pcoord_dim: np.ndarray | None = None,
    threshold: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build robust per-frame lineage features from raw pcoord histories.

    Features include raw pcoord, train z-scores, temporal deltas, window summary
    statistics, and goal-dimension distance-to-threshold features.
    """
    raw = np.asarray(pcoord_windows, dtype=np.float32)
    mask = np.asarray(window_mask, dtype=bool)
    if raw.ndim != 3:
        raise ValueError("pcoord_windows must have shape [examples, window, dims].")
    if mask.shape != raw.shape[:2]:
        raise ValueError("window_mask must match pcoord_windows first two dimensions.")
    num_examples, window, raw_dim = raw.shape

    expected_raw_dim = int(transform.get("raw_pcoord_dim", raw_dim))
    if raw_dim != expected_raw_dim:
        raise ValueError(f"Expected raw pcoord dim {expected_raw_dim}, got {raw_dim}.")
    mean = np.asarray(transform["mean"], dtype=np.float32)
    std = np.asarray(transform["std"], dtype=np.float32)
    if mean.shape != (raw_dim,) or std.shape != (raw_dim,):
        raise ValueError("Feature transform mean/std must match raw pcoord dim.")
    std = np.maximum(std, np.float32(1e-6))

    valid_raw = np.where(mask[:, :, None], raw, 0.0).astype(np.float32)
    z = (raw - mean[None, None, :]) / std[None, None, :]

    first_indices = np.argmax(mask, axis=1)
    last_indices = window - 1 - np.argmax(mask[:, ::-1], axis=1)
    rows = np.arange(num_examples)
    first_values = raw[rows, first_indices]
    last_values = raw[rows, last_indices]
    delta_first = raw - first_values[:, None, :]

    previous = np.concatenate([first_values[:, None, :], raw[:, :-1, :]], axis=1)
    delta_previous = raw - previous

    masked_min = np.where(mask[:, :, None], raw, np.inf).min(axis=1)
    masked_max = np.where(mask[:, :, None], raw, -np.inf).max(axis=1)
    value_range = masked_max - masked_min
    last_repeated = np.repeat(last_values[:, None, :], window, axis=1)
    min_repeated = np.repeat(masked_min[:, None, :], window, axis=1)
    max_repeated = np.repeat(masked_max[:, None, :], window, axis=1)
    range_repeated = np.repeat(value_range[:, None, :], window, axis=1)

    if pcoord_dim is not None and threshold is not None:
        dims = np.asarray(pcoord_dim, dtype=np.int64)
        thresholds = np.asarray(threshold, dtype=np.float32)
        if dims.shape != (num_examples,) or thresholds.shape != (num_examples,):
            raise ValueError("pcoord_dim and threshold must have one value per example.")
        selected = np.take_along_axis(raw, dims[:, None, None].repeat(window, axis=1), axis=2)
        target_delta = selected - thresholds[:, None, None]
        target_abs = np.abs(target_delta)
    else:
        selected = np.zeros((num_examples, window, 1), dtype=np.float32)
        target_delta = np.zeros_like(selected)
        target_abs = np.zeros_like(selected)

    features = np.concatenate(
        [
            valid_raw,
            z,
            delta_first,
            delta_previous,
            last_repeated,
            min_repeated,
            max_repeated,
            range_repeated,
            selected,
            target_delta,
            target_abs,
        ],
        axis=-1,
    ).astype(np.float32)
    features[~mask] = 0.0
    return features


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
    feature_mode: str = "engineered",
) -> tuple[PcoordLineageValueModel, dict[str, float]]:
    dataset.validate()
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_indices, _ = split_pcoord_lineage_indices(
        dataset,
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
    )
    feature_transform = build_pcoord_feature_transform(dataset, train_indices, mode=feature_mode)
    training_dataset = transform_pcoord_lineage_dataset(dataset, feature_transform)
    config = PcoordLineageModelConfig(
        pcoord_dim=training_dataset.pcoord_windows.shape[-1],
        goal_feature_dim=config.goal_feature_dim,
        hidden_dim=config.hidden_dim,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        dropout=config.dropout,
    )

    device_obj = resolve_device(device)
    train_loader, val_loader = _make_loaders(
        dataset=training_dataset,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
    )

    model = PcoordLineageValueModel(config).to(device_obj)
    model.pcoord_feature_transform = feature_transform
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
                        "pcoord_feature_transform": feature_transform,
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
                "pcoord_feature_transform": feature_transform,
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
    feature_transform = getattr(model, "pcoord_feature_transform", {"mode": "raw"})
    dataset = transform_pcoord_lineage_dataset(dataset, feature_transform)
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
    model.pcoord_feature_transform = checkpoint.get("metadata", {}).get(
        "pcoord_feature_transform",
        {"mode": "raw"},
    )
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
