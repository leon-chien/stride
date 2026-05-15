from __future__ import annotations

import csv
import json

import numpy as np

from stride.models import PcoordLineageModelConfig, PcoordLineageValueModel
from stride.training import save_pcoord_lineage_checkpoint
from stride.westpa_plugin import (
    PcoordLineageRuntimeScorer,
    PcoordRuntimeScoringInput,
    ReplayConfig,
    assign_score_bins,
    assign_score_bins_from_edges,
    priority_ranks,
    replay_westpa_steering,
)


def test_pcoord_runtime_scorer_loads_checkpoint_and_scores(tmp_path) -> None:
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
    checkpoint = tmp_path / "lineage.pt"
    save_pcoord_lineage_checkpoint(checkpoint, model, metrics={"train_loss": 0.1})

    scoring_input = _runtime_input(num_examples=5)
    scorer = PcoordLineageRuntimeScorer(checkpoint, device="cpu", batch_size=2)
    result = scorer.score(scoring_input)

    assert not result.used_fallback
    assert result.scores["p_event"].shape == (5,)
    assert np.isfinite(result.scores["stride_score"]).all()


def test_pcoord_runtime_scorer_fallback_is_deterministic() -> None:
    scoring_input = _runtime_input(num_examples=4)
    scorer = PcoordLineageRuntimeScorer(None, fallback_score=0.25)

    first = scorer.score(scoring_input)
    second = scorer.score(scoring_input)

    assert first.used_fallback
    np.testing.assert_allclose(first.scores["p_event"], np.full(4, 0.25))
    np.testing.assert_allclose(first.scores["p_event"], second.scores["p_event"])
    np.testing.assert_allclose(first.scores["uncertainty"], np.ones(4))


def test_quantile_bins_and_priority_ranks_are_stable() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)

    bins, edges = assign_score_bins(scores, num_bins=4, binning="quantile")
    ranks = priority_ranks(scores)

    assert len(np.unique(bins)) == 4
    assert edges.shape == (3,)
    assert ranks[np.argmax(scores)] == 1
    assert ranks[np.argmin(scores)] == len(scores)


def test_replay_preserves_tail_validation_iterations(tmp_path) -> None:
    artifact = _write_tiny_multigoal_artifact(tmp_path / "lineage.npz")
    stride_scores = np.linspace(0.0, 1.0, 24, dtype=np.float32)

    paths = replay_westpa_steering(
        artifact,
        tmp_path / "replay",
        stride_scores=stride_scores,
        config=ReplayConfig(
            eval_split="validation",
            split_strategy="tail",
            validation_fraction=0.25,
            num_bins=4,
        ),
    )

    assignments = np.load(paths["assignments"])
    assert set(assignments["n_iter"]) == {5, 6}
    assert assignments["stride_score"].shape == assignments["event_labels"].shape
    assert assignments["stride_bin"].shape == assignments["event_labels"].shape
    assert assignments["stride_priority_rank"].min() == 1


def test_replay_train_calibrated_bins_use_only_train_indices(tmp_path) -> None:
    artifact = _write_tiny_multigoal_artifact(tmp_path / "lineage.npz")
    stride_scores = np.arange(24, dtype=np.float32)

    paths = replay_westpa_steering(
        artifact,
        tmp_path / "replay",
        stride_scores=stride_scores,
        config=ReplayConfig(
            eval_split="validation",
            split_strategy="tail",
            validation_fraction=0.25,
            num_bins=4,
            bin_reference="train",
        ),
    )

    assignments = np.load(paths["assignments"])
    expected_edges = np.unique(np.quantile(stride_scores[:16], [0.25, 0.5, 0.75])).astype(
        np.float32
    )
    eval_scores = stride_scores[assignments["eval_indices"]]
    np.testing.assert_allclose(assignments["stride_bin_edges"], expected_edges)
    np.testing.assert_array_equal(
        assignments["stride_bin"],
        assign_score_bins_from_edges(eval_scores, expected_edges),
    )
    assert np.all(assignments["calibration_indices"] < 16)


def test_replay_per_iteration_metrics_do_not_mix_iterations(tmp_path) -> None:
    artifact = _write_tiny_multigoal_artifact(tmp_path / "lineage.npz")
    stride_scores = np.linspace(0.0, 1.0, 24, dtype=np.float32)

    paths = replay_westpa_steering(
        artifact,
        tmp_path / "replay",
        stride_scores=stride_scores,
        config=ReplayConfig(
            eval_split="validation",
            split_strategy="tail",
            validation_fraction=0.25,
            num_bins=4,
            bin_reference="train",
            per_iteration=True,
        ),
    )

    iteration_rows = _read_csv(paths["iteration_metrics"])
    stride_rows = [row for row in iteration_rows if row["ranker"] == "STRIDE"]
    assert {float(row["n_iter"]) for row in stride_rows} == {5.0, 6.0}
    assert all(float(row["count"]) == 4.0 for row in stride_rows)

    summary_rows = _read_csv(paths["iteration_summary"])
    assert any(row["ranker"] == "STRIDE" for row in summary_rows)
    assert "Per-Iteration Summary" in paths["markdown"].read_text()


def test_replay_writes_deployment_control_config(tmp_path) -> None:
    artifact = _write_tiny_multigoal_artifact(tmp_path / "lineage.npz")
    stride_scores = np.linspace(0.0, 1.0, 24, dtype=np.float32)

    paths = replay_westpa_steering(
        artifact,
        tmp_path / "replay",
        stride_scores=stride_scores,
        config=ReplayConfig(
            eval_split="validation",
            split_strategy="tail",
            validation_fraction=0.25,
            num_bins=4,
            bin_reference="train",
            per_iteration=True,
            score_key="p_event",
            stride_scores_path="outputs/scores.npz",
            checkpoint_path="outputs/model.best.pt",
        ),
    )

    payload = json.loads(paths["control_config"].read_text())
    assert payload["bin_reference"] == "train"
    assert payload["checkpoint_path"] == "outputs/model.best.pt"
    assert payload["stride_scores_npz"] == "outputs/scores.npz"
    assert payload["rankers"]["stride"]["score_key"] == "p_event"
    assert len(payload["rankers"]["stride"]["bin_edges"]) == 3


def test_replay_outputs_diagnostics_for_rare_single_class_groups(tmp_path) -> None:
    artifact = _write_tiny_multigoal_artifact(tmp_path / "lineage.npz", rare_single_class=True)
    stride_scores = np.linspace(0.0, 1.0, 24, dtype=np.float32)

    paths = replay_westpa_steering(
        artifact,
        tmp_path / "replay",
        stride_scores=stride_scores,
        config=ReplayConfig(
            eval_split="validation",
            split_strategy="tail",
            validation_fraction=0.25,
            num_bins=4,
        ),
    )

    assert paths["markdown"].exists()
    assert paths["metrics"].exists()
    assert paths["bins"].exists()
    assert paths["grouped_metrics"].exists()
    assert paths["iteration_metrics"].exists()
    assert paths["control_config"].exists()

    metric_rows = _read_csv(paths["metrics"])
    grouped_rows = _read_csv(paths["grouped_metrics"])
    assert {row["ranker"] for row in metric_rows} == {"STRIDE", "last_pcoord_low", "random"}
    assert any(row["group_type"] == "goal_id" for row in grouped_rows)
    assert "STRIDE beats" in paths["markdown"].read_text() or "STRIDE does not beat" in paths[
        "markdown"
    ].read_text()


def _runtime_input(num_examples: int) -> PcoordRuntimeScoringInput:
    return PcoordRuntimeScoringInput(
        pcoord_windows=np.ones((num_examples, 4, 2), dtype=np.float32),
        window_mask=np.ones((num_examples, 4), dtype=bool),
        goal_features=np.ones((num_examples, 15), dtype=np.float32),
    )


def _write_tiny_multigoal_artifact(path, rare_single_class: bool = False):
    num_examples = 24
    window = 4
    dims = 2
    pcoord_windows = np.zeros((num_examples, window, dims), dtype=np.float32)
    for index in range(num_examples):
        pcoord_windows[index, :, 0] = np.linspace(1.2, 0.1 + index / 30.0, window)
        pcoord_windows[index, :, 1] = np.linspace(0.9, 0.2 + (index % 6) / 10.0, window)

    labels = (np.arange(num_examples) % 4 == 0).astype(np.float32)
    if rare_single_class:
        labels[-8:] = 0.0

    np.savez_compressed(
        path,
        pcoord_windows=pcoord_windows,
        window_mask=np.ones((num_examples, window), dtype=bool),
        goal_features=np.ones((num_examples, 15), dtype=np.float32),
        event_labels=labels,
        flux_labels=labels * 0.2,
        weights=np.ones((num_examples,), dtype=np.float64),
        n_iter=np.repeat(np.arange(1, 7, dtype=np.int64), 4),
        seg_id=np.tile(np.arange(4, dtype=np.int64), 6),
        goal_id=np.asarray(["goal_a", "goal_b"] * 12),
        cell_id=np.asarray(["cell_0"] * 12 + ["cell_1"] * 12),
        pcoord_dim=np.asarray([0, 1] * 12, dtype=np.int64),
        threshold=np.asarray([0.5, 0.25] * 12, dtype=np.float32),
        horizon_iterations=np.full((num_examples,), 5, dtype=np.int64),
    )
    return path


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
