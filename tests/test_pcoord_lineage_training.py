from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from stride.models import PcoordLineageModelConfig, PcoordLineageValueModel
from stride.training import (
    build_pcoord_feature_transform,
    engineer_pcoord_window_features,
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


def test_pcoord_feature_transform_uses_train_statistics(tmp_path) -> None:
    artifact = _write_tiny_lineage_artifact(tmp_path / "lineage.npz")
    dataset = load_pcoord_lineage_dataset_npz(artifact)

    transform = build_pcoord_feature_transform(dataset, np.arange(8), mode="engineered")
    features = engineer_pcoord_window_features(
        dataset.pcoord_windows,
        dataset.window_mask,
        transform,
        pcoord_dim=np.ones((len(dataset.event_labels),), dtype=np.int64),
        threshold=np.full((len(dataset.event_labels),), 0.5, dtype=np.float32),
    )

    assert transform["raw_pcoord_dim"] == 2
    assert transform["feature_dim"] == 19
    assert features.shape == (12, 4, 19)
    assert np.all(features[~dataset.window_mask] == 0.0)
    np.testing.assert_allclose(
        transform["mean"],
        dataset.pcoord_windows[:8][dataset.window_mask[:8]].mean(axis=0),
    )


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
    assert loaded_model.config.pcoord_dim > dataset.pcoord_windows.shape[-1]
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


def test_default_best_checkpoint_preserves_decimal_threshold_names() -> None:
    helper = _load_train_script_helper()
    path = helper(Path("outputs/tutorial35_cell0_dim1_thr0.5.pt"))

    assert path.name == "tutorial35_cell0_dim1_thr0.5.best.pt"


def _load_train_script_helper():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_westpa_lineage.py"
    spec = importlib.util.spec_from_file_location("train_westpa_lineage", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load train_westpa_lineage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._default_best_checkpoint


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
