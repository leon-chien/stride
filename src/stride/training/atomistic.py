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
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    lr_scheduler: str | None = None,
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
    if early_stopping_patience is not None and early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative.")

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
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "lr_scheduler": lr_scheduler or "none",
            },
            optimizer=optimizer,
            scheduler=scheduler,
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


def truncate_atomistic_history(
    dataset: AtomisticDataset,
    history_frames: int | None,
) -> AtomisticDataset:
    """
    Keep only the most recent frames in each atomistic lineage window.

    This supports controlled temporal ablations: history_frames=1 gives a
    last-frame-only model while preserving the same atom features and labels.
    """
    dataset.validate()
    if history_frames is None:
        return dataset
    history_frames = int(history_frames)
    if history_frames <= 0:
        raise ValueError("history_frames must be positive.")
    window_size = dataset.coordinates.shape[1]
    if history_frames >= window_size:
        return dataset

    return AtomisticDataset(
        coordinates=dataset.coordinates[:, -history_frames:, :, :].copy(),
        atom_features=dataset.atom_features.copy(),
        atom_mask=dataset.atom_mask.copy(),
        frame_mask=dataset.frame_mask[:, -history_frames:].copy(),
        goal_features=dataset.goal_features.copy(),
        event_labels=dataset.event_labels.copy(),
        flux_labels=dataset.flux_labels.copy(),
        source_frame_start=dataset.source_frame_start.copy(),
    )


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
    train_indices_np, val_indices_np = split_atomistic_indices(
        dataset=dataset,
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
    )

    train_labels = dataset.event_labels[train_indices_np]
    val_labels = dataset.event_labels[val_indices_np] if len(val_indices_np) else np.array([])

    return {
        "num_examples": float(len(dataset.event_labels)),
        "positive_rate": float(np.mean(dataset.event_labels)),
        "train_examples": float(len(train_indices_np)),
        "train_positive_rate": float(np.mean(train_labels)) if len(train_labels) else float("nan"),
        "train_positives": float(np.sum(train_labels >= 0.5)),
        "val_examples": float(len(val_indices_np)),
        "val_positive_rate": float(np.mean(val_labels)) if len(val_labels) else float("nan"),
        "val_positives": float(np.sum(val_labels >= 0.5)) if len(val_labels) else 0.0,
    }


def split_atomistic_indices(
    dataset: AtomisticDataset,
    validation_fraction: float = 0.2,
    seed: int = 7,
    split_strategy: str = "contiguous",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return train and validation indices using the same split logic as training.
    """
    dataset.validate()
    train_indices, val_indices = _split_indices(
        num_examples=len(dataset.event_labels),
        validation_fraction=validation_fraction,
        seed=seed,
        split_strategy=split_strategy,
        source_frame_start=dataset.source_frame_start,
        window_size=dataset.coordinates.shape[1],
        event_labels=dataset.event_labels,
    )
    return train_indices.numpy(), val_indices.numpy()


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
        source_frame_start=dataset.source_frame_start,
        window_size=dataset.coordinates.shape[1],
        event_labels=dataset.event_labels,
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
    source_frame_start: np.ndarray | None = None,
    window_size: int | None = None,
    event_labels: np.ndarray | None = None,
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
    elif split_strategy == "blocked":
        train_indices, val_indices = _blocked_split_indices(
            num_examples=num_examples,
            val_count=val_count,
            seed=seed,
            source_frame_start=source_frame_start,
            window_size=window_size,
        )
    elif split_strategy == "blocked_tail":
        train_indices, val_indices = _blocked_split_indices(
            num_examples=num_examples,
            val_count=val_count,
            seed=seed,
            source_frame_start=source_frame_start,
            window_size=window_size,
            block_start="tail",
        )
    elif split_strategy in {"iteration_tail", "iteration_random", "iteration_balanced"}:
        train_indices, val_indices = _iteration_split_indices(
            num_examples=num_examples,
            validation_fraction=validation_fraction,
            seed=seed,
            split_strategy=split_strategy,
            source_frame_start=source_frame_start,
            event_labels=event_labels,
        )
    else:
        raise ValueError(
            "split_strategy must be 'contiguous', 'random', 'blocked', "
            "'blocked_tail', 'iteration_tail', 'iteration_random', "
            "or 'iteration_balanced'."
        )

    return train_indices, val_indices


def _blocked_split_indices(
    num_examples: int,
    val_count: int,
    seed: int,
    source_frame_start: np.ndarray | None,
    window_size: int | None,
    block_start: int | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if val_count == 0:
        return torch.arange(num_examples), torch.empty(0, dtype=torch.long)
    if source_frame_start is None:
        source_frame_start = np.arange(num_examples, dtype=np.int64)
    starts = np.asarray(source_frame_start, dtype=np.int64)
    if starts.shape != (num_examples,):
        raise ValueError("source_frame_start must have one value per example.")
    if window_size is None or window_size <= 0:
        window_size = 1

    ordered = np.argsort(starts, kind="mergesort")
    max_block_start = num_examples - val_count
    if block_start == "tail":
        resolved_block_start = max_block_start
    elif block_start is None:
        rng = np.random.default_rng(seed)
        resolved_block_start = int(rng.integers(0, max_block_start + 1))
    else:
        resolved_block_start = int(block_start)
    if resolved_block_start < 0 or resolved_block_start > max_block_start:
        raise ValueError("blocked split start is out of range.")
    val_indices_np = np.sort(
        ordered[resolved_block_start : resolved_block_start + val_count]
    )

    val_starts = starts[val_indices_np]
    val_interval_start = int(np.min(val_starts))
    val_interval_end = int(np.max(val_starts) + window_size)
    train_mask = np.ones(num_examples, dtype=bool)
    train_mask[val_indices_np] = False
    train_intervals_overlap = (
        (starts < val_interval_end) & ((starts + int(window_size)) > val_interval_start)
    )
    train_mask[train_intervals_overlap] = False
    train_indices_np = np.flatnonzero(train_mask)

    if train_indices_np.size == 0:
        raise ValueError(
            "Blocked split removed all training examples; reduce validation_fraction "
            "or use a longer trajectory."
        )

    return (
        torch.tensor(train_indices_np, dtype=torch.long),
        torch.tensor(val_indices_np, dtype=torch.long),
    )


def _iteration_split_indices(
    num_examples: int,
    validation_fraction: float,
    seed: int,
    split_strategy: str,
    source_frame_start: np.ndarray | None,
    event_labels: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_frame_start is None:
        source_frame_start = np.arange(num_examples, dtype=np.int64)
    starts = np.asarray(source_frame_start, dtype=np.int64)
    if starts.shape != (num_examples,):
        raise ValueError("source_frame_start must have one value per example.")

    unique_starts = np.unique(starts)
    if unique_starts.size < 3:
        return _split_indices(
            num_examples=num_examples,
            validation_fraction=validation_fraction,
            seed=seed,
            split_strategy="random",
        )

    val_group_count = int(round(unique_starts.size * validation_fraction))
    val_group_count = min(max(val_group_count, 1), unique_starts.size - 1)
    if split_strategy == "iteration_tail":
        val_starts = unique_starts[-val_group_count:]
    elif split_strategy == "iteration_random":
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, unique_starts.size - val_group_count + 1))
        val_starts = unique_starts[start : start + val_group_count]
    elif split_strategy == "iteration_balanced":
        if event_labels is None:
            raise ValueError("iteration_balanced requires event_labels.")
        val_starts = _balanced_iteration_block(
            unique_starts=unique_starts,
            starts=starts,
            labels=np.asarray(event_labels, dtype=np.float32),
            val_group_count=val_group_count,
        )
    else:
        raise ValueError(f"Unknown iteration split strategy: {split_strategy}")

    val_mask = np.isin(starts, val_starts)
    train_indices_np = np.flatnonzero(~val_mask)
    val_indices_np = np.flatnonzero(val_mask)
    if train_indices_np.size == 0 or val_indices_np.size == 0:
        raise ValueError("Iteration split produced an empty train or validation set.")
    return (
        torch.tensor(train_indices_np, dtype=torch.long),
        torch.tensor(val_indices_np, dtype=torch.long),
    )


def _balanced_iteration_block(
    unique_starts: np.ndarray,
    starts: np.ndarray,
    labels: np.ndarray,
    val_group_count: int,
) -> np.ndarray:
    if labels.shape != starts.shape:
        raise ValueError("event_labels must have one value per example.")
    overall_rate = float(np.mean(labels))
    best_starts = unique_starts[:val_group_count]
    best_score = float("inf")
    for start in range(0, unique_starts.size - val_group_count + 1):
        candidate = unique_starts[start : start + val_group_count]
        val_mask = np.isin(starts, candidate)
        train_labels = labels[~val_mask]
        val_labels = labels[val_mask]
        if train_labels.size == 0 or val_labels.size == 0:
            continue
        train_rate = float(np.mean(train_labels))
        val_rate = float(np.mean(val_labels))
        train_has_both = np.unique(train_labels >= 0.5).size == 2
        val_has_both = np.unique(val_labels >= 0.5).size == 2
        class_penalty = 0.0 if train_has_both and val_has_both else 10.0
        score = (
            abs(train_rate - overall_rate)
            + abs(val_rate - overall_rate)
            + class_penalty
        )
        if score < best_score:
            best_score = score
            best_starts = candidate
    return best_starts


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
    min_delta: float = 0.0,
) -> bool:
    if metric_value is None or not np.isfinite(metric_value):
        return False
    if best_metric_value is None or not np.isfinite(best_metric_value):
        return True
    if mode == "max":
        return metric_value > best_metric_value + min_delta
    if mode == "min":
        return metric_value < best_metric_value - min_delta
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


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: str | None,
    epochs: int,
    metric_mode: str,
) -> object | None:
    value = (lr_scheduler or "none").lower()
    if value == "none":
        return None
    if value == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
        )
    if value == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=metric_mode,
            factor=0.5,
            patience=2,
        )
    raise ValueError("lr_scheduler must be one of: none, cosine, plateau.")


def _step_scheduler(
    scheduler: object | None,
    lr_scheduler: str | None,
    metric_value: float | None,
) -> None:
    if scheduler is None:
        return
    value = (lr_scheduler or "none").lower()
    if value == "plateau":
        if metric_value is not None and np.isfinite(metric_value):
            scheduler.step(float(metric_value))
    else:
        scheduler.step()
