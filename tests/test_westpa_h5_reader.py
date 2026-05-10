from __future__ import annotations

import h5py
import numpy as np

from stride.goals import GoalSpec
from stride.westpa_plugin.h5_reader import (
    SegmentKey,
    build_lineage_windows,
    compute_delayed_labels,
    extract_pcoord_window,
    load_segment_records,
    save_lineage_windows_npz,
)


def test_westpa_lineage_reconstruction_and_delayed_labels(tmp_path) -> None:
    h5_path = tmp_path / "west.h5"
    _write_tiny_westpa_h5(h5_path)

    records = load_segment_records(h5_path)

    assert len(records) == 7
    assert records[SegmentKey(2, 1)].parent_key == SegmentKey(1, 0)
    assert records[SegmentKey(1, 0)].parent_key is None

    window = extract_pcoord_window(
        records,
        key=SegmentKey(3, 0),
        window_iterations=3,
    )

    assert window.shape == (3, 1)
    assert np.allclose(window[:, 0], [0.8, 0.3, 0.2])

    goal = GoalSpec(
        name="association",
        type="distance_threshold",
        selections=("a", "b"),
        operator="less_than",
        threshold=0.35,
        horizon_iterations=2,
    )
    labels = compute_delayed_labels(records, goal)

    label = labels[SegmentKey(1, 0)]
    assert label.event == 1
    assert label.num_event_descendants == 2
    assert np.isclose(label.flux, 0.5)

    no_event_label = labels[SegmentKey(1, 1)]
    assert no_event_label.event == 0
    assert no_event_label.flux == 0.0


def test_build_lineage_windows_for_training(tmp_path) -> None:
    h5_path = tmp_path / "west.h5"
    _write_tiny_westpa_h5(h5_path)

    records = load_segment_records(h5_path)
    goal = GoalSpec(
        name="association",
        type="distance_threshold",
        selections=("a", "b"),
        operator="less_than",
        threshold=0.35,
        horizon_iterations=2,
    )

    windows = build_lineage_windows(
        records=records,
        goal=goal,
        window_iterations=2,
        require_full_window=True,
    )

    assert len(windows) == 5
    assert windows[0].pcoord_window.shape == (2, 1)
    assert windows[0].goal_features.shape == (goal.feature_dim,)

    by_key = {window.key: window for window in windows}
    assert by_key[SegmentKey(2, 1)].event == 1
    assert np.isclose(by_key[SegmentKey(2, 1)].flux, 0.25)


def test_save_lineage_windows_pads_short_windows(tmp_path) -> None:
    h5_path = tmp_path / "west.h5"
    output_path = tmp_path / "stride_dataset.npz"
    _write_tiny_westpa_h5(h5_path)

    records = load_segment_records(h5_path)
    goal = GoalSpec(
        name="association",
        type="distance_threshold",
        selections=("a", "b"),
        operator="less_than",
        threshold=0.35,
        horizon_iterations=2,
    )

    windows = build_lineage_windows(
        records=records,
        goal=goal,
        window_iterations=3,
        require_full_window=False,
    )
    save_lineage_windows_npz(output_path, windows)

    data = np.load(output_path)
    assert data["pcoord_windows"].shape == (7, 3, 1)
    assert data["window_mask"].shape == (7, 3)
    assert data["window_mask"][0].tolist() == [False, False, True]


def _write_tiny_westpa_h5(path) -> None:
    seg_index_dtype = np.dtype(
        [
            ("weight", np.float64),
            ("parent_id", np.int64),
            ("endpoint_type", np.uint8),
        ]
    )

    with h5py.File(path, "w") as h5:
        iterations = h5.create_group("iterations")
        _write_iter(
            iterations,
            n_iter=1,
            parent_ids=[-1, -2],
            weights=[0.5, 0.5],
            final_pcoords=[0.8, 0.9],
            seg_index_dtype=seg_index_dtype,
        )
        _write_iter(
            iterations,
            n_iter=2,
            parent_ids=[0, 0, 1],
            weights=[0.25, 0.25, 0.5],
            final_pcoords=[0.6, 0.3, 0.85],
            seg_index_dtype=seg_index_dtype,
        )
        _write_iter(
            iterations,
            n_iter=3,
            parent_ids=[1, 2],
            weights=[0.25, 0.5],
            final_pcoords=[0.2, 0.7],
            seg_index_dtype=seg_index_dtype,
        )


def _write_iter(
    iterations,
    n_iter: int,
    parent_ids: list[int],
    weights: list[float],
    final_pcoords: list[float],
    seg_index_dtype: np.dtype,
) -> None:
    group = iterations.create_group(f"iter_{n_iter:08d}")

    pcoord = np.zeros((len(parent_ids), 2, 1), dtype=np.float32)
    pcoord[:, 0, 0] = 1.0
    pcoord[:, 1, 0] = np.asarray(final_pcoords, dtype=np.float32)
    group.create_dataset("pcoord", data=pcoord)

    seg_index = np.zeros((len(parent_ids),), dtype=seg_index_dtype)
    seg_index["weight"] = np.asarray(weights, dtype=np.float64)
    seg_index["parent_id"] = np.asarray(parent_ids, dtype=np.int64)
    seg_index["endpoint_type"] = 0
    group.create_dataset("seg_index", data=seg_index)
