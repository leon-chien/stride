from __future__ import annotations

import h5py
import numpy as np

from stride.goals import GoalSpec
from stride.westpa_plugin import (
    SegmentKey,
    build_coordinate_atomistic_dataset,
    build_segment_coordinate_store,
    load_segment_coordinate_store_npz,
    load_segment_records,
    save_segment_coordinate_store_npz,
    segment_trajectory_path,
)


def test_segment_trajectory_path_formats_westpa_keys(tmp_path) -> None:
    path = segment_trajectory_path(
        tmp_path,
        "{n_iter:06d}/{seg_id:06d}/seg.pdb",
        SegmentKey(12, 3),
    )

    assert path == tmp_path / "000012" / "000003" / "seg.pdb"


def test_build_segment_coordinate_store_from_pdb_segments(tmp_path) -> None:
    h5_path = tmp_path / "west.h5"
    topology_path = tmp_path / "topology.pdb"
    traj_root = tmp_path / "traj_segs"
    output_npz = tmp_path / "segment_coordinates.npz"
    _write_tiny_westpa_h5(h5_path)
    _write_pdb(topology_path, x_offset=0.0)

    records = load_segment_records(h5_path)
    for key in records:
        path = segment_trajectory_path(
            traj_root,
            "{n_iter:06d}/{seg_id:06d}/seg.pdb",
            key,
        )
        _write_pdb(path, x_offset=float(key.n_iter) + float(key.seg_id) / 10.0)

    store, report = build_segment_coordinate_store(
        records=records,
        topology_path=topology_path,
        trajectory_root=traj_root,
        trajectory_pattern="{n_iter:06d}/{seg_id:06d}/seg.pdb",
        coordinate_units="angstrom",
    )
    save_segment_coordinate_store_npz(output_npz, store)
    loaded = load_segment_coordinate_store_npz(output_npz)

    assert report.total_segments == 3
    assert report.saved_segments == 3
    assert report.missing_segments == 0
    assert loaded.coordinates.shape == (3, 2, 3)
    assert loaded.atom_features.shape[0] == 3
    assert loaded.atom_mask.all()

    goal = GoalSpec(
        name="association",
        type="distance_threshold",
        selections=("pcoord",),
        operator="less_than",
        threshold=0.5,
        horizon_iterations=1,
    )
    dataset, provenance = build_coordinate_atomistic_dataset(
        records=records,
        coordinates=loaded,
        goal=goal,
        window_iterations=2,
    )
    assert dataset.coordinates.shape[0] == 2
    assert provenance["westpa_n_iter"].tolist() == [2, 2]
    assert provenance["westpa_pcoord_windows"].shape == (2, 2, 1)
    assert provenance["westpa_pcoord_window_mask"].shape == (2, 2)
    assert provenance["westpa_pcoord_window_mask"].all()


def test_build_segment_coordinate_store_reports_missing_segments(tmp_path) -> None:
    h5_path = tmp_path / "west.h5"
    topology_path = tmp_path / "topology.pdb"
    traj_root = tmp_path / "traj_segs"
    _write_tiny_westpa_h5(h5_path)
    _write_pdb(topology_path, x_offset=0.0)

    records = load_segment_records(h5_path)
    key = SegmentKey(1, 0)
    path = segment_trajectory_path(
        traj_root,
        "{n_iter:06d}/{seg_id:06d}/seg.pdb",
        key,
    )
    _write_pdb(path, x_offset=0.0)

    store, report = build_segment_coordinate_store(
        records=records,
        topology_path=topology_path,
        trajectory_root=traj_root,
        trajectory_pattern="{n_iter:06d}/{seg_id:06d}/seg.pdb",
        coordinate_units="angstrom",
        require_all=False,
    )

    assert store.coordinates.shape[0] == 1
    assert report.missing_segments == 2
    assert SegmentKey(2, 0) in report.missing_examples


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
            parent_ids=[-1],
            weights=[1.0],
            final_pcoords=[0.8],
            seg_index_dtype=seg_index_dtype,
        )
        _write_iter(
            iterations,
            n_iter=2,
            parent_ids=[0, 0],
            weights=[0.5, 0.5],
            final_pcoords=[0.4, 0.9],
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


def _write_pdb(path, x_offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(
            "ATOM      1  C   ALA A   1    "
            f"{x_offset:8.3f}{0.0:8.3f}{0.0:8.3f}"
            "  1.00  0.00           C\n"
        )
        handle.write(
            "ATOM      2  O   ALA A   1    "
            f"{x_offset + 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
            "  1.00  0.00           O\n"
        )
        handle.write("END\n")
