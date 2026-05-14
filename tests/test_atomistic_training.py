from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from stride.data import build_sample_ligand_contact_dataset
from stride.training.atomistic import _evaluate_model, _to_tensor_dataset
from stride.training.atomistic import _split_indices
from stride.training.evaluation import (
    atom_pair_distance_baseline_scores,
    dihedral_window_baseline_scores,
    write_evaluation_report,
)
from stride.training.stride_value import StrideValueLossConfig
from stride.training import (
    describe_atomistic_split,
    load_atomistic_checkpoint,
    save_atomistic_checkpoint,
    score_atomistic_dataset,
    split_atomistic_indices,
    train_atomistic_value_model,
    truncate_atomistic_history,
)
from stride.models import StrideModelConfig


def test_atomistic_training_progress_callback_runs_each_epoch() -> None:
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
    calls = []

    train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=2,
        batch_size=4,
        validation_fraction=0.25,
        seed=17,
        device="cpu",
        progress_callback=lambda epoch, total, metrics: calls.append(
            (epoch, total, metrics["train_loss"])
        ),
    )

    assert len(calls) == 2
    assert calls[0][0:2] == (1, 2)
    assert calls[1][0:2] == (2, 2)


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


def test_atomistic_training_saves_best_checkpoint_and_resume(tmp_path) -> None:
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
    final_checkpoint = tmp_path / "final.pt"
    best_checkpoint = tmp_path / "best.pt"

    _, metrics = train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=1,
        batch_size=4,
        validation_fraction=0.25,
        seed=13,
        device="cpu",
        checkpoint_path=final_checkpoint,
        best_checkpoint_path=best_checkpoint,
        save_best_metric="val_loss",
        save_best_mode="min",
    )

    assert final_checkpoint.exists()
    assert best_checkpoint.exists()
    assert metrics["best_metric_value"] == metrics["val_loss"]

    _, resumed_metrics = train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=2,
        batch_size=4,
        validation_fraction=0.25,
        seed=13,
        device="cpu",
        resume_from=final_checkpoint,
    )

    assert resumed_metrics["start_epoch"] == 2.0
    assert resumed_metrics["epoch"] == 2.0


def test_describe_atomistic_split_reports_positive_rates() -> None:
    dataset = build_sample_ligand_contact_dataset(
        window_size=4,
        horizon=2,
        num_frames=12,
    )
    stats = describe_atomistic_split(
        dataset,
        validation_fraction=0.25,
        seed=3,
        split_strategy="random",
    )

    assert stats["num_examples"] == len(dataset.event_labels)
    assert stats["train_examples"] > 0
    assert stats["val_examples"] > 0
    assert 0.0 <= stats["positive_rate"] <= 1.0


def test_blocked_split_avoids_overlapping_windows() -> None:
    source_frame_start = np.arange(30, dtype=np.int64)
    train_indices, val_indices = _split_indices(
        num_examples=30,
        validation_fraction=0.2,
        seed=5,
        split_strategy="blocked",
        source_frame_start=source_frame_start,
        window_size=4,
    )

    assert len(train_indices) > 0
    assert len(val_indices) > 0
    for train_index in train_indices.numpy():
        train_start = source_frame_start[train_index]
        train_end = train_start + 4
        for val_index in val_indices.numpy():
            val_start = source_frame_start[val_index]
            val_end = val_start + 4
            assert train_end <= val_start or val_end <= train_start

    tail_train_indices, tail_val_indices = _split_indices(
        num_examples=30,
        validation_fraction=0.2,
        seed=5,
        split_strategy="blocked_tail",
        source_frame_start=source_frame_start,
        window_size=4,
    )
    assert source_frame_start[tail_val_indices.numpy()].min() > source_frame_start[
        tail_train_indices.numpy()
    ].max()

    dataset = build_sample_ligand_contact_dataset(
        window_size=4,
        horizon=2,
        num_frames=12,
    )
    train_np, val_np = split_atomistic_indices(
        dataset,
        validation_fraction=0.25,
        seed=5,
        split_strategy="blocked_tail",
    )
    assert train_np.dtype == np.int64
    assert val_np.dtype == np.int64
    assert len(train_np) > 0
    assert len(val_np) > 0


def test_iteration_balanced_split_keeps_positive_examples_in_both_splits() -> None:
    dataset = build_sample_ligand_contact_dataset(
        window_size=4,
        horizon=2,
        num_frames=16,
    )
    labels = dataset.event_labels.copy()
    starts = np.asarray([4, 4, 5, 5, 6, 6, 7, 7, 8, 8], dtype=np.int64)
    labels = np.asarray([0, 0, 0, 1, 0, 1, 0, 1, 1, 1], dtype=np.float32)
    dataset = type(dataset)(
        coordinates=dataset.coordinates[: len(labels)],
        atom_features=dataset.atom_features[: len(labels)],
        atom_mask=dataset.atom_mask[: len(labels)],
        frame_mask=dataset.frame_mask[: len(labels)],
        goal_features=dataset.goal_features[: len(labels)],
        event_labels=labels,
        flux_labels=labels,
        source_frame_start=starts,
    )

    train_indices, val_indices = split_atomistic_indices(
        dataset,
        validation_fraction=0.4,
        seed=7,
        split_strategy="iteration_balanced",
    )

    assert np.unique(dataset.source_frame_start[val_indices]).size == 2
    assert dataset.event_labels[train_indices].sum() > 0
    assert dataset.event_labels[val_indices].sum() > 0
    assert dataset.event_labels[train_indices].sum() < len(train_indices)
    assert dataset.event_labels[val_indices].sum() < len(val_indices)


def test_truncate_atomistic_history_keeps_last_frames() -> None:
    dataset = build_sample_ligand_contact_dataset(
        window_size=4,
        horizon=2,
        num_frames=12,
    )

    truncated = truncate_atomistic_history(dataset, history_frames=1)

    assert truncated.coordinates.shape[1] == 1
    assert truncated.frame_mask.shape[1] == 1
    assert np.allclose(truncated.coordinates[:, 0], dataset.coordinates[:, -1])
    assert np.array_equal(truncated.event_labels, dataset.event_labels)


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


def test_dihedral_baseline_scores_one_value_per_example() -> None:
    coordinates = np.zeros((8, 4, 3), dtype=np.float32)
    base = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = base.copy()
    target[3] = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    coordinates[:5] = base
    coordinates[5:] = target

    from stride.data import AtomRecord, build_atomistic_windows
    from stride.goals import GoalSpec

    atoms = [
        AtomRecord(atom_name="A", element="C", residue_name="ALA", residue_id=1),
        AtomRecord(atom_name="B", element="C", residue_name="ALA", residue_id=1),
        AtomRecord(atom_name="C", element="C", residue_name="ALA", residue_id=1),
        AtomRecord(atom_name="D", element="C", residue_name="ALA", residue_id=1),
    ]
    dataset = build_atomistic_windows(
        coordinates=coordinates,
        atoms=atoms,
        goal=GoalSpec(
            name="toy_dihedral",
            type="dihedral_window",
            selections=("atom:A", "atom:B", "atom:C", "atom:D"),
            operator="inside",
            threshold=0.0,
            lower_bound=-10.0,
            upper_bound=10.0,
            horizon_iterations=2,
        ),
        window_size=3,
        horizon=2,
    )

    scores = dihedral_window_baseline_scores(
        dataset,
        atom_indices=(0, 1, 2, 3),
        lower_bound=-10.0,
        upper_bound=10.0,
    )

    assert scores.shape == dataset.event_labels.shape
    assert np.isfinite(scores).all()


def test_atom_pair_distance_baseline_scores_one_value_per_example() -> None:
    dataset = build_sample_ligand_contact_dataset(
        window_size=4,
        horizon=2,
        num_frames=12,
    )

    last_scores = atom_pair_distance_baseline_scores(
        dataset,
        atom_indices=(0, 1),
        mode="last_frame",
        direction="low",
    )
    window_scores = atom_pair_distance_baseline_scores(
        dataset,
        atom_indices=(0, 1),
        mode="window_min",
        direction="low",
    )

    assert last_scores.shape == dataset.event_labels.shape
    assert window_scores.shape == dataset.event_labels.shape
    assert np.isfinite(last_scores).all()
    assert np.isfinite(window_scores).all()


def test_evaluation_report_handles_rare_positives(tmp_path) -> None:
    y_true = np.asarray([0, 0, 0, 0, 1], dtype=np.float32)
    rankers = {
        "STRIDE": np.asarray([0.1, 0.2, 0.3, 0.4, 0.9], dtype=np.float32),
        "random": np.asarray([0.5, 0.1, 0.8, 0.2, 0.4], dtype=np.float32),
    }

    paths = write_evaluation_report(tmp_path, y_true=y_true, rankers=rankers)

    assert paths["metrics"].exists()
    assert paths["summary"].exists()
    assert paths["quantiles"].exists()
    assert "STRIDE" in paths["markdown"].read_text()


def test_early_stopping_preserves_manual_best_checkpoint_selection(tmp_path) -> None:
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
    manual_best = tmp_path / "manual_best.pt"
    early_best = tmp_path / "early_best.pt"

    _, manual_metrics = train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=2,
        batch_size=4,
        validation_fraction=0.25,
        seed=19,
        device="cpu",
        best_checkpoint_path=manual_best,
        save_best_metric="val_loss",
        save_best_mode="min",
    )
    _, early_metrics = train_atomistic_value_model(
        dataset=dataset,
        config=config,
        epochs=2,
        batch_size=4,
        validation_fraction=0.25,
        seed=19,
        device="cpu",
        best_checkpoint_path=early_best,
        save_best_metric="val_loss",
        save_best_mode="min",
        early_stopping_patience=10,
    )

    assert manual_best.exists()
    assert early_best.exists()
    assert manual_metrics["best_metric_value"] == early_metrics["best_metric_value"]
