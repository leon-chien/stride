from __future__ import annotations

import numpy as np

from stride.training.westpa_evaluation import (
    pcoord_baseline_rankers,
    westpa_iteration_split_indices,
    write_westpa_lineage_report,
)
from stride.westpa_plugin import (
    RuntimeScoringInput,
    StrideRuntimeScorer,
    StrideValueBinMapper,
    ValueMapperConfig,
)


def test_westpa_iteration_split_holds_out_whole_iterations() -> None:
    n_iter = np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    train_indices, val_indices = westpa_iteration_split_indices(
        n_iter,
        validation_fraction=0.25,
        split_strategy="tail",
    )

    assert set(n_iter[val_indices]) == {4}
    assert set(n_iter[train_indices]).isdisjoint(set(n_iter[val_indices]))


def test_pcoord_baselines_and_report_handle_lineage_artifact(tmp_path) -> None:
    artifact = tmp_path / "lineage.npz"
    pcoord_windows = np.asarray(
        [
            [[0.9], [0.8]],
            [[0.7], [0.4]],
            [[0.3], [0.2]],
            [[0.9], [0.85]],
        ],
        dtype=np.float32,
    )
    window_mask = np.ones((4, 2), dtype=bool)
    np.savez_compressed(
        artifact,
        pcoord_windows=pcoord_windows,
        window_mask=window_mask,
        event_labels=np.asarray([0, 1, 1, 0], dtype=np.float32),
        flux_labels=np.asarray([0.0, 0.2, 0.3, 0.0], dtype=np.float32),
        weights=np.ones((4,), dtype=np.float64),
        n_iter=np.asarray([1, 2, 3, 4], dtype=np.int64),
        seg_id=np.asarray([0, 0, 0, 0], dtype=np.int64),
        goal_features=np.ones((4, 15), dtype=np.float32),
    )

    rankers = pcoord_baseline_rankers(
        pcoord_windows,
        window_mask,
        target=0.25,
    )
    assert rankers["last_pcoord_low"].shape == (4,)
    assert rankers["last_pcoord_target_proximity"][2] > rankers[
        "last_pcoord_target_proximity"
    ][0]

    paths = write_westpa_lineage_report(
        artifact,
        tmp_path / "report",
        eval_split="validation",
        validation_fraction=0.25,
    )
    assert paths["metrics"].exists()
    assert "window_min_pcoord_low" in paths["markdown"].read_text()


def test_runtime_scorer_fallback_assigns_bins() -> None:
    scoring_input = RuntimeScoringInput(
        coordinates=np.zeros((3, 2, 2, 3), dtype=np.float32),
        atom_features=np.ones((3, 2, 4), dtype=np.float32),
        atom_mask=np.ones((3, 2), dtype=bool),
        frame_mask=np.ones((3, 2), dtype=bool),
        goal_features=np.ones((3, 15), dtype=np.float32),
    )
    scorer = StrideRuntimeScorer(
        checkpoint_path=None,
        fallback_score=0.4,
    )
    mapper = StrideValueBinMapper(
        ValueMapperConfig(num_bins=4, min_score=0.0, max_score=1.0)
    )

    assignments, result = scorer.assign_bins(scoring_input, mapper)

    assert result.used_fallback
    assert np.allclose(result.scores["stride_score"], [0.4, 0.4, 0.4])
    assert assignments.shape == (3,)
    assert assignments.tolist() == [1, 1, 1]
