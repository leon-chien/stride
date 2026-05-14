from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stride.data import compute_atom_features
from stride.data.mdanalysis_converter import load_mdanalysis_trajectory
from stride.westpa_plugin.h5_reader import (
    SegmentCoordinateStore,
    SegmentKey,
    SegmentRecord,
)


@dataclass(frozen=True)
class SegmentCoordinateBuildReport:
    """
    Summary of a WESTPA segment-coordinate build.
    """

    total_segments: int
    saved_segments: int
    missing_segments: int
    missing_examples: tuple[SegmentKey, ...]


def build_segment_coordinate_store(
    records: dict[SegmentKey, SegmentRecord],
    topology_path: str | Path,
    trajectory_root: str | Path,
    trajectory_pattern: str,
    frame_index: int = -1,
    mda_selection: str | None = None,
    coordinate_units: str = "nm",
    require_all: bool = True,
) -> tuple[SegmentCoordinateStore, SegmentCoordinateBuildReport]:
    """
    Build a per-segment coordinate store from WESTPA segment trajectory files.

    The pattern is formatted with `n_iter` and `seg_id`, for example:
        "{n_iter:06d}/{seg_id:06d}/seg.xtc"
    """
    topology_path = Path(topology_path)
    trajectory_root = Path(trajectory_root)
    if not topology_path.exists():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")
    if not trajectory_root.exists():
        raise FileNotFoundError(f"Trajectory root not found: {trajectory_root}")

    coordinates: list[np.ndarray] = []
    n_iters: list[int] = []
    seg_ids: list[int] = []
    atom_features_reference: np.ndarray | None = None
    atom_mask_reference: np.ndarray | None = None
    missing: list[SegmentKey] = []

    for key in sorted(records, key=lambda item: (item.n_iter, item.seg_id)):
        trajectory_path = segment_trajectory_path(
            trajectory_root=trajectory_root,
            trajectory_pattern=trajectory_pattern,
            key=key,
        )
        if not trajectory_path.exists():
            missing.append(key)
            continue

        frames, atoms = load_mdanalysis_trajectory(
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            mda_selection=mda_selection,
            coordinate_units=coordinate_units,
        )
        selected_frame = frames[frame_index]
        atom_features = compute_atom_features(atoms)
        atom_mask = np.ones((selected_frame.shape[0],), dtype=bool)

        if atom_features_reference is None:
            atom_features_reference = atom_features
            atom_mask_reference = atom_mask
        elif atom_features.shape != atom_features_reference.shape:
            raise ValueError(
                "Segment trajectory atom feature shape changed: "
                f"{atom_features.shape} != {atom_features_reference.shape}"
            )
        elif selected_frame.shape != coordinates[0].shape:
            raise ValueError(
                "Segment trajectory coordinate shape changed: "
                f"{selected_frame.shape} != {coordinates[0].shape}"
            )

        coordinates.append(selected_frame.astype(np.float32))
        n_iters.append(int(key.n_iter))
        seg_ids.append(int(key.seg_id))

    if missing and require_all:
        examples = ", ".join(
            f"({key.n_iter}, {key.seg_id})" for key in missing[:5]
        )
        raise FileNotFoundError(
            f"Missing {len(missing)} segment trajectory files; examples: {examples}"
        )
    if not coordinates or atom_features_reference is None or atom_mask_reference is None:
        raise ValueError("No segment coordinates were loaded.")

    num_segments = len(coordinates)
    atom_features = np.broadcast_to(
        atom_features_reference[None, :, :],
        (num_segments, *atom_features_reference.shape),
    ).copy()
    atom_mask = np.broadcast_to(
        atom_mask_reference[None, :],
        (num_segments, atom_mask_reference.shape[0]),
    ).copy()
    store = SegmentCoordinateStore(
        coordinates=np.stack(coordinates).astype(np.float32),
        n_iter=np.asarray(n_iters, dtype=np.int64),
        seg_id=np.asarray(seg_ids, dtype=np.int64),
        atom_features=atom_features.astype(np.float32),
        atom_mask=atom_mask.astype(bool),
    )
    store.validate()
    report = SegmentCoordinateBuildReport(
        total_segments=len(records),
        saved_segments=num_segments,
        missing_segments=len(missing),
        missing_examples=tuple(missing[:10]),
    )
    return store, report


def save_segment_coordinate_store_npz(
    path: str | Path,
    store: SegmentCoordinateStore,
) -> None:
    """
    Save a segment-coordinate store for `extract_westpa_dataset.py`.
    """
    store.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coordinates=store.coordinates,
        n_iter=store.n_iter,
        seg_id=store.seg_id,
        atom_features=store.atom_features,
        atom_mask=store.atom_mask,
    )


def segment_trajectory_path(
    trajectory_root: str | Path,
    trajectory_pattern: str,
    key: SegmentKey,
) -> Path:
    """
    Resolve one segment trajectory path from a format pattern.
    """
    relative = trajectory_pattern.format(
        n_iter=int(key.n_iter),
        seg_id=int(key.seg_id),
    )
    return Path(trajectory_root) / relative
