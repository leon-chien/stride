from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stride.data import AtomisticDataset, load_atomistic_dataset_npz
from stride.models import StrideModelConfig, StrideValueModel
from stride.training.metrics import compute_binary_metrics, top_k_enrichment
from stride.training.stride_value import StrideValueTargets, stride_value_loss


def train_atomistic_value_model(
    dataset: AtomisticDataset,
    config: StrideModelConfig,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 7,
    device: str | None = None,
) -> tuple[StrideValueModel, dict[str, float]]:
    """
    Train STRIDE's goal-conditioned value model on an AtomisticDataset.
    """
    dataset.validate()
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_tensors = _make_loaders(
        dataset=dataset,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    model = StrideValueModel(config).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
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
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_metrics["loss"])

        metrics = {"epoch": float(epoch), "train_loss": float(np.mean(epoch_losses))}
        if val_tensors is not None:
            metrics.update(_evaluate_model(model, val_tensors, device_obj, prefix="val_"))

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
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "metrics": metrics or {},
        },
        path,
    )


def load_atomistic_checkpoint(
    path: str | Path,
    device: str | None = None,
) -> tuple[StrideValueModel, dict[str, float]]:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(path, map_location=device_obj)
    config = StrideModelConfig(**checkpoint["config"])
    model = StrideValueModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    return model, dict(checkpoint.get("metrics", {}))


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


def _make_loaders(
    dataset: AtomisticDataset,
    batch_size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[DataLoader, tuple[torch.Tensor, ...] | None]:
    tensor_dataset = _to_tensor_dataset(dataset)
    num_examples = len(tensor_dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_examples, generator=generator)

    val_count = int(round(num_examples * validation_fraction))
    if num_examples < 3:
        val_count = 0
    else:
        val_count = min(max(val_count, 1), num_examples - 1)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_subset = torch.utils.data.Subset(tensor_dataset, train_indices.tolist())
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    val_tensors = None
    if val_count > 0:
        tensors = tuple(tensor[val_indices] for tensor in tensor_dataset.tensors)
        val_tensors = tensors

    return train_loader, val_tensors


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
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    model.eval()
    coordinates, atom_features, atom_mask, frame_mask, goal_features, event, flux = [
        tensor.to(device) for tensor in tensors
    ]
    with torch.no_grad():
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
        )

    scores = outputs["p_event"].detach().cpu().numpy()
    labels = event.detach().cpu().numpy()
    binary_metrics = compute_binary_metrics(labels, scores)
    enrichment = top_k_enrichment(labels, scores, k=0.25)

    metrics = {f"{prefix}{key}": value for key, value in loss_metrics.items()}
    metrics.update({f"{prefix}{key}": value for key, value in binary_metrics.items()})
    metrics[f"{prefix}top25_enrichment"] = float(enrichment)
    return metrics
