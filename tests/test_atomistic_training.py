from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from stride.data import build_sample_ligand_contact_dataset
from stride.training.atomistic import _evaluate_model, _to_tensor_dataset
from stride.training.stride_value import StrideValueLossConfig
from stride.training import (
    load_atomistic_checkpoint,
    save_atomistic_checkpoint,
    score_atomistic_dataset,
    train_atomistic_value_model,
)
from stride.models import StrideModelConfig


def test_atomistic_training_checkpoint_and_scoring(tmp_path) -> None:
    dataset = build_sample_ligand_contact_dataset(
        window_size=4,
        horizon=2,
        num_frames=12,
    )
    config = StrideModelConfig(
        atom_feature_dim=dataset.atom_features.shape[-1],
        goal_feature_dim=dataset.goal_features.shape[-1],
        hidden_dim=16,
        egnn_layers=1,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )

    model, metrics = train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=1,
        batch_size=4,
        validation_fraction=0.25,
        seed=11,
        device="cpu",
    )
    assert metrics["train_loss"] > 0.0

    checkpoint = tmp_path / "stride_atomistic.pt"
    save_atomistic_checkpoint(checkpoint, model, metrics)
    loaded_model, loaded_metrics = load_atomistic_checkpoint(checkpoint, device="cpu")
    scores = score_atomistic_dataset(loaded_model, dataset, batch_size=3, device="cpu")

    assert loaded_metrics["train_loss"] == metrics["train_loss"]
    assert scores["p_event"].shape == dataset.event_labels.shape
    assert scores["stride_score"].shape == dataset.event_labels.shape
    assert np.isfinite(scores["p_event"]).all()


def test_validation_evaluation_uses_batches() -> None:
    dataset = build_sample_ligand_contact_dataset(
        window_size=4,
        horizon=2,
        num_frames=12,
    )
    config = StrideModelConfig(
        atom_feature_dim=dataset.atom_features.shape[-1],
        goal_feature_dim=dataset.goal_features.shape[-1],
        hidden_dim=16,
        egnn_layers=1,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )
    model = train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=1,
        batch_size=2,
        validation_fraction=0.25,
        seed=3,
        device="cpu",
    )[0]
    loader = DataLoader(_to_tensor_dataset(dataset), batch_size=2, shuffle=False)

    metrics = _evaluate_model(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        prefix="val_",
        loss_config=StrideValueLossConfig(),
    )

    assert "val_loss" in metrics
    assert "val_auroc" in metrics
