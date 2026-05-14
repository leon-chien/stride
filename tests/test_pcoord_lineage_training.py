from __future__ import annotations

import numpy as np
import torch

from stride.models import PcoordLineageModelConfig, PcoordLineageValueModel
from stride.training import (
    load_lineage_dataset_and_make_config,
    load_pcoord_lineage_checkpoint,
    load_pcoord_lineage_dataset_npz,
    score_pcoord_lineage_dataset,
    split_pcoord_lineage_indices,
    train_pcoord_lineage_value_model,
)
from stride.training.westpa_evaluation import write_westpa_lineage_report


def test_pcoord_lineage_loader_validates_shapes(tmp_path) -> None:
    artifact = _write_tiny_lineage_artifact(tmp_path / "lineage.npz")

    dataset = load_pcoord_lineage_dataset_npz(artifact)

    assert dataset.pcoord_windows.shape == (12, 4, 2)
    assert dataset.window_mask.shape == (12, 4)
    assert dataset.event_labels.shape == (12,)
    assert dataset.window_mask[0, 0] == np.False_


def test_pcoord_lineage_model_outputs_one_score_per_example() -> None:
    model = PcoordLineageValueModel(
        PcoordLineageModelConfig(
            pcoord_dim=2,
            goal_feature_dim=15,
            hidden_dim=16,
            transformer_layers=1,
            transformer_heads=4,
            dropout=0.0,
        )
    )
    pcoord_windows = torch.randn(3, 4, 2)
    window_mask = torch.ones(3, 4, dtype=torch.bool)
    window_mask[0, 0] = False
    goal_features = torch.ones(3, 15)

    outputs = model(
        pcoord_windows=pcoord_windows,
        window_mask=window_mask,
        goal_features=goal_features,
    )

    assert set(outputs) == {"p_event", "flux_value", "uncertainty", "stride_score"}
    assert outputs["p_event"].shape == (3,)
    assert torch.isfinite(outputs["stride_score"]).all()


def test_pcoord_lineage_training_checkpoint_scoring_and_report(tmp_path) -> None:
    artifact = _write_tiny_lineage_artifact(tmp_path / "lineage.npz")
    dataset, config = load_lineage_dataset_and_make_config(
        artifact,
        hidden_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )
    final_checkpoint = tmp_path / "lineage.pt"
    best_checkpoint = tmp_path / "lineage.best.pt"

    _, metrics = train_pcoord_lineage_value_model(
        dataset=dataset,
        config=config,
        epochs=1,
        batch_size=4,
        validation_fraction=0.33,
        split_strategy="tail",
        seed=11,
        device="cpu",
        checkpoint_path=final_checkpoint,
        best_checkpoint_path=best_checkpoint,
        save_best_metric="val_auprc",
        save_best_mode="max",
    )
    assert final_checkpoint.exists()
    assert best_checkpoint.exists()
    assert "val_auprc" in metrics

    loaded_model, loaded_metrics = load_pcoord_lineage_checkpoint(
        best_checkpoint,
        device="cpu",
    )
    scores = score_pcoord_lineage_dataset(
        loaded_model,
        dataset,
        batch_size=3,
        device="cpu",
    )

    assert loaded_metrics["train_loss"] == metrics["train_loss"]
    assert scores["p_event"].shape == dataset.event_labels.shape
    assert np.isfinite(scores["p_event"]).all()

    paths = write_westpa_lineage_report(
        artifact,
        tmp_path / "report",
        stride_scores=scores["p_event"],
        eval_split="validation",
        validation_fraction=0.33,
        split_strategy="tail",
        pcoord_target=0.5,
        pcoord_dim=1,
    )
    text = paths["markdown"].read_text()
    assert "STRIDE" in text
    assert "last_pcoord_low" in text


def test_pcoord_lineage_split_holds_out_whole_iterations(tmp_path) -> None:
    dataset = load_pcoord_lineage_dataset_npz(
        _write_tiny_lineage_artifact(tmp_path / "lineage.npz")
    )

    train_indices, val_indices = split_pcoord_lineage_indices(
        dataset,
        validation_fraction=0.33,
        split_strategy="tail",
    )

    assert set(dataset.n_iter[train_indices]).isdisjoint(set(dataset.n_iter[val_indices]))
    assert set(dataset.n_iter[val_indices]) == {5, 6}


def _write_tiny_lineage_artifact(path) -> object:
    num_examples = 12
    window = 4
    dims = 2
    pcoord_windows = np.zeros((num_examples, window, dims), dtype=np.float32)
    for index in range(num_examples):
        pcoord_windows[index, :, 0] = np.linspace(1.0, 0.2 + index / 20.0, window)
        pcoord_windows[index, :, 1] = np.linspace(0.8, 0.1 + (index % 4) / 5.0, window)
    window_mask = np.ones((num_examples, window), dtype=bool)
    window_mask[::3, 0] = False
    event_labels = np.asarray(
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        pcoord_windows=pcoord_windows,
        window_mask=window_mask,
        event_labels=event_labels,
        flux_labels=event_labels * 0.25,
        weights=np.ones((num_examples,), dtype=np.float64),
        n_iter=np.repeat(np.arange(1, 7, dtype=np.int64), 2),
        seg_id=np.tile(np.arange(2, dtype=np.int64), 6),
        goal_features=np.ones((num_examples, 15), dtype=np.float32),
    )
    return path
