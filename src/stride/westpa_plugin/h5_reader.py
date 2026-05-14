from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from stride.goals import GoalSpec
from stride.data import AtomisticDataset


@dataclass
class H5DatasetInfo:
    """
    Metadata for one HDF5 dataset.
    """

    path: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SegmentKey:
    """
    Stable identity for one WESTPA segment.
    """

    n_iter: int
    seg_id: int


@dataclass
class SegmentRecord:
    """
    WESTPA segment metadata needed for lineage reconstruction and labeling.
    """

    key: SegmentKey
    parent_id: int
    weight: float
    endpoint_type: int | None
    pcoord: np.ndarray

    @property
    def parent_key(self) -> SegmentKey | None:
        """
        Parent segment key, or None for initial-state parents.

        WESTPA stores initial-state parents as negative IDs. Nonnegative parent
        IDs refer to segment IDs in the previous iteration.
        """
        if self.parent_id < 0 or self.key.n_iter <= 1:
            return None
        return SegmentKey(
            n_iter=self.key.n_iter - 1,
            seg_id=int(self.parent_id),
        )


@dataclass(frozen=True)
class DelayedLabel:
    """
    Delayed descendant label for one segment.
    """

    event: int
    flux: float
    num_event_descendants: int


@dataclass(frozen=True)
class LineageWindow:
    """
    Training window extracted from a WESTPA lineage.
    """

    key: SegmentKey
    pcoord_window: np.ndarray
    event: int
    flux: float
    weight: float
    goal_features: np.ndarray


@dataclass(frozen=True)
class SegmentCoordinateStore:
    """
    Coordinate frames keyed by WESTPA segment identity.
    """

    coordinates: np.ndarray
    n_iter: np.ndarray
    seg_id: np.ndarray
    atom_features: np.ndarray
    atom_mask: np.ndarray

    def validate(self) -> None:
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [segments, atoms, 3].")
        num_segments, num_atoms, _ = self.coordinates.shape
        if self.n_iter.shape != (num_segments,):
            raise ValueError("n_iter must have one value per coordinate frame.")
        if self.seg_id.shape != (num_segments,):
            raise ValueError("seg_id must have one value per coordinate frame.")
        if self.atom_features.shape[:2] != (num_segments, num_atoms):
            raise ValueError(
                "atom_features must have shape [segments, atoms, features]."
            )
        if self.atom_mask.shape != (num_segments, num_atoms):
            raise ValueError("atom_mask must have shape [segments, atoms].")


def list_h5_datasets(h5_path: str | Path) -> list[H5DatasetInfo]:
    """
    List all datasets inside an HDF5 file.

    This is useful because WESTPA .h5 files can contain groups such as:
        /iterations/iter_00000001/pcoord
        /iterations/iter_00000001/seg_index
        /summary
        etc.

    This function does not assume a specific layout. It simply inspects.
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    datasets: list[H5DatasetInfo] = []

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(
                H5DatasetInfo(
                    path="/" + name,
                    shape=tuple(obj.shape),
                    dtype=str(obj.dtype),
                )
            )

    with h5py.File(h5_path, "r") as h5:
        h5.visititems(visitor)

    return datasets


def print_h5_tree(h5_path: str | Path) -> None:
    """
    Print a simple HDF5 dataset tree.
    """
    datasets = list_h5_datasets(h5_path)

    print(f"HDF5 file: {h5_path}")
    print(f"Datasets found: {len(datasets)}")
    print()

    print(f"{'dataset path':<70} | {'shape':<25} | {'dtype'}")
    print("-" * 115)

    for dataset in datasets:
        print(
            f"{dataset.path:<70} | "
            f"{str(dataset.shape):<25} | "
            f"{dataset.dtype}"
        )


def find_candidate_pcoord_paths(h5_path: str | Path) -> list[H5DatasetInfo]:
    """
    Find datasets whose path suggests they may contain WESTPA progress coordinates.
    """
    datasets = list_h5_datasets(h5_path)

    candidates = []

    for dataset in datasets:
        lower_path = dataset.path.lower()

        if "pcoord" in lower_path or "progress" in lower_path:
            candidates.append(dataset)

    return candidates


def read_dataset(h5_path: str | Path, dataset_path: str) -> np.ndarray:
    """
    Read one dataset from an HDF5 file.
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        if dataset_path not in h5:
            raise KeyError(f"Dataset path not found in HDF5 file: {dataset_path}")

        data = h5[dataset_path][...]

    return np.asarray(data)


def list_iteration_groups(h5_path: str | Path) -> list[str]:
    """
    List WESTPA-style iteration groups if present.

    Common WESTPA files have groups like:
        /iterations/iter_00000001
        /iterations/iter_00000002

    This function returns those group paths if they exist.
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        if "/iterations" not in h5:
            return []

        iteration_group = h5["/iterations"]

        names = sorted(
            iteration_group.keys(),
            key=_iteration_name_to_number,
        )

    return [f"/iterations/{name}" for name in names]


def _iteration_name_to_number(name: str) -> int:
    try:
        return int(name.rsplit("_", maxsplit=1)[-1])
    except ValueError:
        return -1


def iteration_group_to_number(iteration_group_path: str) -> int:
    """
    Convert '/iterations/iter_00000012' to 12.
    """
    return _iteration_name_to_number(Path(iteration_group_path).name)


def read_iteration_pcoord(
    h5_path: str | Path,
    iteration_group_path: str,
) -> np.ndarray:
    """
    Read pcoord from a single WESTPA iteration group.

    Expected path:
        /iterations/iter_xxxxxxxx/pcoord

    Returns:
        pcoord array, usually shaped like:
            [num_segments, pcoord_len, pcoord_dim]
    """
    dataset_path = f"{iteration_group_path}/pcoord"

    return read_dataset(h5_path, dataset_path)


def read_iteration_seg_index(
    h5_path: str | Path,
    iteration_group_path: str,
) -> np.ndarray:
    """
    Read WESTPA seg_index from one iteration group.

    Expected path:
        /iterations/iter_xxxxxxxx/seg_index
    """
    dataset_path = f"{iteration_group_path}/seg_index"
    return read_dataset(h5_path, dataset_path)


def read_iteration_records(
    h5_path: str | Path,
    iteration_group_path: str,
) -> list[SegmentRecord]:
    """
    Read one iteration as SegmentRecord objects.
    """
    n_iter = iteration_group_to_number(iteration_group_path)
    if n_iter < 0:
        raise ValueError(f"Could not parse iteration number from {iteration_group_path}")

    pcoord = read_iteration_pcoord(h5_path, iteration_group_path)
    seg_index = read_iteration_seg_index(h5_path, iteration_group_path)

    if len(pcoord) != len(seg_index):
        raise ValueError(
            f"pcoord and seg_index length mismatch for {iteration_group_path}: "
            f"{len(pcoord)} != {len(seg_index)}"
        )

    dtype_names = set(seg_index.dtype.names or ())
    if "parent_id" not in dtype_names:
        raise KeyError(f"{iteration_group_path}/seg_index has no parent_id field")
    if "weight" not in dtype_names:
        raise KeyError(f"{iteration_group_path}/seg_index has no weight field")

    records: list[SegmentRecord] = []
    for seg_id, row in enumerate(seg_index):
        endpoint_type = None
        if "endpoint_type" in dtype_names:
            endpoint_type = int(row["endpoint_type"])

        records.append(
            SegmentRecord(
                key=SegmentKey(n_iter=n_iter, seg_id=int(seg_id)),
                parent_id=int(row["parent_id"]),
                weight=float(row["weight"]),
                endpoint_type=endpoint_type,
                pcoord=np.asarray(pcoord[seg_id]),
            )
        )

    return records


def load_segment_records(
    h5_path: str | Path,
    max_iterations: int | None = None,
) -> dict[SegmentKey, SegmentRecord]:
    """
    Load all available WESTPA segment records keyed by (iteration, segment ID).
    """
    iteration_groups = list_iteration_groups(h5_path)
    if max_iterations is not None:
        iteration_groups = iteration_groups[-max_iterations:]

    records: dict[SegmentKey, SegmentRecord] = {}
    for group_path in iteration_groups:
        for record in read_iteration_records(h5_path, group_path):
            records[record.key] = record
    return records


def build_child_index(
    records: dict[SegmentKey, SegmentRecord],
) -> dict[SegmentKey, list[SegmentKey]]:
    """
    Build parent -> children mapping for descendant traversal.
    """
    child_index: dict[SegmentKey, list[SegmentKey]] = {}
    for record in records.values():
        parent_key = record.parent_key
        if parent_key is None:
            continue
        if parent_key not in records:
            continue
        child_index.setdefault(parent_key, []).append(record.key)

    for children in child_index.values():
        children.sort(key=lambda key: (key.n_iter, key.seg_id))

    return child_index


def trace_ancestor_keys(
    records: dict[SegmentKey, SegmentRecord],
    key: SegmentKey,
    window_iterations: int,
) -> list[SegmentKey]:
    """
    Return ancestor keys ordered oldest -> current for one segment.
    """
    if window_iterations <= 0:
        raise ValueError("window_iterations must be positive.")
    if key not in records:
        raise KeyError(f"Segment not found: {key}")

    lineage = [key]
    current = records[key]

    while len(lineage) < window_iterations:
        parent_key = current.parent_key
        if parent_key is None or parent_key not in records:
            break
        lineage.append(parent_key)
        current = records[parent_key]

    lineage.reverse()
    return lineage


def extract_pcoord_window(
    records: dict[SegmentKey, SegmentRecord],
    key: SegmentKey,
    window_iterations: int,
    pcoord_frame_index: int = -1,
) -> np.ndarray:
    """
    Extract pcoord history for one segment lineage.

    The returned array is ordered oldest -> current and shaped
    [available_window, pcoord_dim]. The window may be shorter near the beginning
    of a simulation.
    """
    lineage = trace_ancestor_keys(
        records=records,
        key=key,
        window_iterations=window_iterations,
    )
    values = [records[lineage_key].pcoord[pcoord_frame_index] for lineage_key in lineage]
    return np.asarray(values, dtype=np.float32)


def descendant_keys_within_horizon(
    child_index: dict[SegmentKey, list[SegmentKey]],
    key: SegmentKey,
    horizon_iterations: int,
) -> list[SegmentKey]:
    """
    Return descendant keys within a future iteration horizon.
    """
    if horizon_iterations <= 0:
        raise ValueError("horizon_iterations must be positive.")

    descendants: list[SegmentKey] = []
    queue = list(child_index.get(key, []))
    max_iter = key.n_iter + horizon_iterations

    while queue:
        child = queue.pop(0)
        if child.n_iter > max_iter:
            continue
        descendants.append(child)
        queue.extend(child_index.get(child, []))

    descendants.sort(key=lambda item: (item.n_iter, item.seg_id))
    return descendants


def pcoord_satisfies_goal(
    pcoord: np.ndarray,
    goal: GoalSpec,
    pcoord_dim: int = 0,
) -> bool:
    """
    Evaluate a structured goal against a pcoord value.

    This supports pcoord-based labels. Coordinate/RMSD/contact goals should be
    evaluated by a separate coordinate-aware labeler once coordinate extraction
    is available.
    """
    goal.validate()

    if goal.type != "distance_threshold":
        raise NotImplementedError(
            "Only pcoord distance_threshold goals are supported by h5_reader. "
            "Coordinate-aware goals should use a coordinate labeler."
        )

    value = float(np.asarray(pcoord)[pcoord_dim])
    if goal.operator == "less_than":
        return value < goal.threshold
    if goal.operator == "greater_than":
        return value > goal.threshold
    raise NotImplementedError(
        f"Operator {goal.operator!r} is not supported for scalar pcoord labels."
    )


def compute_delayed_labels(
    records: dict[SegmentKey, SegmentRecord],
    goal: GoalSpec,
    horizon_iterations: int | None = None,
    pcoord_frame_index: int = -1,
    pcoord_dim: int = 0,
    include_self: bool = False,
) -> dict[SegmentKey, DelayedLabel]:
    """
    Compute event/flux labels from future descendants.

    Event is 1 if any descendant reaches the target within the horizon. Flux is
    the sum of descendant segment weights that satisfy the target condition.
    """
    horizon = horizon_iterations or goal.horizon_iterations
    child_index = build_child_index(records)

    labels: dict[SegmentKey, DelayedLabel] = {}
    for key, record in records.items():
        candidates = descendant_keys_within_horizon(child_index, key, horizon)
        if include_self:
            candidates = [key] + candidates

        flux = 0.0
        num_event_descendants = 0
        for descendant_key in candidates:
            descendant = records[descendant_key]
            pcoord = descendant.pcoord[pcoord_frame_index]
            if pcoord_satisfies_goal(pcoord, goal=goal, pcoord_dim=pcoord_dim):
                num_event_descendants += 1
                flux += float(descendant.weight)

        labels[key] = DelayedLabel(
            event=int(num_event_descendants > 0),
            flux=float(flux),
            num_event_descendants=num_event_descendants,
        )

    return labels


def build_lineage_windows(
    records: dict[SegmentKey, SegmentRecord],
    goal: GoalSpec,
    window_iterations: int,
    horizon_iterations: int | None = None,
    pcoord_frame_index: int = -1,
    pcoord_dim: int = 0,
    require_full_window: bool = True,
    include_self_in_label: bool = False,
) -> list[LineageWindow]:
    """
    Build pcoord lineage windows and delayed labels for training.
    """
    labels = compute_delayed_labels(
        records=records,
        goal=goal,
        horizon_iterations=horizon_iterations,
        pcoord_frame_index=pcoord_frame_index,
        pcoord_dim=pcoord_dim,
        include_self=include_self_in_label,
    )

    windows: list[LineageWindow] = []
    goal_features = goal.to_feature_vector()

    for key in sorted(records, key=lambda item: (item.n_iter, item.seg_id)):
        pcoord_window = extract_pcoord_window(
            records=records,
            key=key,
            window_iterations=window_iterations,
            pcoord_frame_index=pcoord_frame_index,
        )
        if require_full_window and len(pcoord_window) < window_iterations:
            continue

        label = labels[key]
        windows.append(
            LineageWindow(
                key=key,
                pcoord_window=pcoord_window,
                event=label.event,
                flux=label.flux,
                weight=records[key].weight,
                goal_features=goal_features,
            )
        )

    return windows


def load_segment_coordinate_store_npz(path: str | Path) -> SegmentCoordinateStore:
    """
    Load per-segment coordinate frames keyed by WESTPA (n_iter, seg_id).

    Expected arrays:
        coordinates: [segments, atoms, 3]
        n_iter: [segments]
        seg_id: [segments]
        atom_features: [segments, atoms, features] or [atoms, features]
        atom_mask: optional [segments, atoms] or [atoms]
    """
    data = np.load(path)
    coordinates = data["coordinates"].astype(np.float32)
    n_iter = data["n_iter"].astype(np.int64)
    seg_id = data["seg_id"].astype(np.int64)
    num_segments, num_atoms, _ = coordinates.shape

    if "atom_features" in data:
        atom_features = data["atom_features"].astype(np.float32)
        if atom_features.ndim == 2:
            atom_features = np.broadcast_to(
                atom_features[None, :, :],
                (num_segments, num_atoms, atom_features.shape[-1]),
            ).copy()
    else:
        atom_features = np.ones((num_segments, num_atoms, 1), dtype=np.float32)

    if "atom_mask" in data:
        atom_mask = data["atom_mask"].astype(bool)
        if atom_mask.ndim == 1:
            atom_mask = np.broadcast_to(
                atom_mask[None, :],
                (num_segments, num_atoms),
            ).copy()
    else:
        atom_mask = np.ones((num_segments, num_atoms), dtype=bool)

    store = SegmentCoordinateStore(
        coordinates=coordinates,
        n_iter=n_iter,
        seg_id=seg_id,
        atom_features=atom_features,
        atom_mask=atom_mask,
    )
    store.validate()
    return store


def build_coordinate_atomistic_dataset(
    records: dict[SegmentKey, SegmentRecord],
    coordinates: SegmentCoordinateStore,
    goal: GoalSpec,
    window_iterations: int,
    horizon_iterations: int | None = None,
    pcoord_frame_index: int = -1,
    pcoord_dim: int = 0,
    require_full_window: bool = True,
    include_self_in_label: bool = False,
) -> tuple[AtomisticDataset, dict[str, np.ndarray]]:
    """
    Build coordinate lineage windows with delayed WESTPA pcoord labels.

    This is the coordinate-aware bridge from WESTPA segment identities to the
    canonical atomistic model input. Labels still use the auditable delayed
    descendant machinery; coordinate-aware labelers can be added behind the same
    output contract later.
    """
    coordinates.validate()
    coordinate_index = {
        SegmentKey(int(n_iter), int(seg_id)): index
        for index, (n_iter, seg_id) in enumerate(
            zip(coordinates.n_iter, coordinates.seg_id, strict=True)
        )
    }
    labels = compute_delayed_labels(
        records=records,
        goal=goal,
        horizon_iterations=horizon_iterations,
        pcoord_frame_index=pcoord_frame_index,
        pcoord_dim=pcoord_dim,
        include_self=include_self_in_label,
    )
    goal_features = goal.to_feature_vector()
    num_atoms = coordinates.coordinates.shape[1]
    atom_feature_dim = coordinates.atom_features.shape[-1]

    window_coordinates: list[np.ndarray] = []
    window_atom_features: list[np.ndarray] = []
    window_atom_masks: list[np.ndarray] = []
    frame_masks: list[np.ndarray] = []
    event_labels: list[float] = []
    flux_labels: list[float] = []
    source_frame_start: list[int] = []
    current_n_iter: list[int] = []
    current_seg_id: list[int] = []
    lineage_n_iter: list[np.ndarray] = []
    lineage_seg_id: list[np.ndarray] = []
    lineage_pcoord: list[np.ndarray] = []
    lineage_pcoord_mask: list[np.ndarray] = []
    weights: list[float] = []
    first_record = next(iter(records.values()))
    pcoord_shape = np.asarray(first_record.pcoord[pcoord_frame_index]).shape

    for key in sorted(records, key=lambda item: (item.n_iter, item.seg_id)):
        lineage = trace_ancestor_keys(records, key, window_iterations)
        available = [lineage_key for lineage_key in lineage if lineage_key in coordinate_index]
        if require_full_window and len(available) < window_iterations:
            continue
        if not available:
            continue

        padded_coords = np.zeros(
            (window_iterations, num_atoms, 3),
            dtype=np.float32,
        )
        padded_atom_features = np.zeros(
            (num_atoms, atom_feature_dim),
            dtype=np.float32,
        )
        padded_atom_mask = np.zeros((num_atoms,), dtype=bool)
        frame_mask = np.zeros((window_iterations,), dtype=bool)
        padded_lineage_n_iter = np.full((window_iterations,), -1, dtype=np.int64)
        padded_lineage_seg_id = np.full((window_iterations,), -1, dtype=np.int64)
        padded_pcoord = np.zeros((window_iterations, *pcoord_shape), dtype=np.float32)
        pcoord_mask = np.zeros((window_iterations,), dtype=bool)

        trimmed = available[-window_iterations:]
        offset = window_iterations - len(trimmed)
        for window_index, lineage_key in enumerate(trimmed, start=offset):
            coordinate_index_value = coordinate_index[lineage_key]
            padded_coords[window_index] = coordinates.coordinates[coordinate_index_value]
            padded_lineage_n_iter[window_index] = lineage_key.n_iter
            padded_lineage_seg_id[window_index] = lineage_key.seg_id
            padded_pcoord[window_index] = np.asarray(
                records[lineage_key].pcoord[pcoord_frame_index],
                dtype=np.float32,
            )
            frame_mask[window_index] = True
            pcoord_mask[window_index] = True

        current_coordinate_index = coordinate_index[trimmed[-1]]
        padded_atom_features[:] = coordinates.atom_features[current_coordinate_index]
        padded_atom_mask[:] = coordinates.atom_mask[current_coordinate_index]

        label = labels[key]
        window_coordinates.append(padded_coords)
        window_atom_features.append(padded_atom_features)
        window_atom_masks.append(padded_atom_mask)
        frame_masks.append(frame_mask)
        event_labels.append(float(label.event))
        flux_labels.append(float(label.flux))
        source_frame_start.append(int(key.n_iter))
        current_n_iter.append(int(key.n_iter))
        current_seg_id.append(int(key.seg_id))
        lineage_n_iter.append(padded_lineage_n_iter)
        lineage_seg_id.append(padded_lineage_seg_id)
        lineage_pcoord.append(padded_pcoord)
        lineage_pcoord_mask.append(pcoord_mask)
        weights.append(float(records[key].weight))

    if not window_coordinates:
        raise ValueError("No coordinate lineage windows could be built.")

    dataset = AtomisticDataset(
        coordinates=np.stack(window_coordinates).astype(np.float32),
        atom_features=np.stack(window_atom_features).astype(np.float32),
        atom_mask=np.stack(window_atom_masks).astype(bool),
        frame_mask=np.stack(frame_masks).astype(bool),
        goal_features=np.broadcast_to(
            goal_features[None, :],
            (len(window_coordinates), goal_features.shape[0]),
        ).copy(),
        event_labels=np.asarray(event_labels, dtype=np.float32),
        flux_labels=np.asarray(flux_labels, dtype=np.float32),
        source_frame_start=np.asarray(source_frame_start, dtype=np.int64),
    )
    dataset.validate()
    provenance = {
        "westpa_n_iter": np.asarray(current_n_iter, dtype=np.int64),
        "westpa_seg_id": np.asarray(current_seg_id, dtype=np.int64),
        "westpa_lineage_n_iter": np.stack(lineage_n_iter).astype(np.int64),
        "westpa_lineage_seg_id": np.stack(lineage_seg_id).astype(np.int64),
        "westpa_pcoord_windows": np.stack(lineage_pcoord).astype(np.float32),
        "westpa_pcoord_window_mask": np.stack(lineage_pcoord_mask).astype(bool),
        "westpa_weights": np.asarray(weights, dtype=np.float64),
    }
    return dataset, provenance


def save_westpa_atomistic_dataset_npz(
    output_path: str | Path,
    dataset: AtomisticDataset,
    provenance: dict[str, np.ndarray],
) -> None:
    """
    Save a canonical atomistic dataset plus WESTPA provenance arrays.
    """
    dataset.validate()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        coordinates=dataset.coordinates,
        atom_features=dataset.atom_features,
        atom_mask=dataset.atom_mask,
        frame_mask=dataset.frame_mask,
        goal_features=dataset.goal_features,
        event_labels=dataset.event_labels,
        flux_labels=dataset.flux_labels,
        source_frame_start=dataset.source_frame_start,
        **provenance,
    )


def save_lineage_windows_npz(
    output_path: str | Path,
    windows: list[LineageWindow],
) -> None:
    """
    Save pcoord lineage windows as a compact training artifact.
    """
    if not windows:
        raise ValueError("No lineage windows to save.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcoord_dims = {window.pcoord_window.shape[1:] for window in windows}
    if len(pcoord_dims) != 1:
        raise ValueError("All pcoord windows must have the same non-time dimensions.")

    max_window = max(window.pcoord_window.shape[0] for window in windows)
    pcoord_dim = next(iter(pcoord_dims))
    padded_windows = np.zeros((len(windows), max_window, *pcoord_dim), dtype=np.float32)
    window_mask = np.zeros((len(windows), max_window), dtype=bool)

    for i, window in enumerate(windows):
        length = window.pcoord_window.shape[0]
        padded_windows[i, -length:] = window.pcoord_window
        window_mask[i, -length:] = True

    np.savez_compressed(
        output_path,
        pcoord_windows=padded_windows,
        window_mask=window_mask,
        event_labels=np.asarray([window.event for window in windows], dtype=np.float32),
        flux_labels=np.asarray([window.flux for window in windows], dtype=np.float32),
        weights=np.asarray([window.weight for window in windows], dtype=np.float64),
        n_iter=np.asarray([window.key.n_iter for window in windows], dtype=np.int64),
        seg_id=np.asarray([window.key.seg_id for window in windows], dtype=np.int64),
        goal_features=np.stack([window.goal_features for window in windows]).astype(
            np.float32
        ),
    )


def extract_latest_pcoord_histories(
    h5_path: str | Path,
    max_iterations: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Extract pcoord arrays from iteration groups.

    Returns:
        dictionary mapping iteration group path -> pcoord array

    This does not yet reconstruct full parent-child walker histories.
    It simply reads per-iteration pcoord arrays.

    Later, STRIDE will use seg_index/parent information to reconstruct
    descendant histories.
    """
    iteration_groups = list_iteration_groups(h5_path)

    if max_iterations is not None:
        iteration_groups = iteration_groups[-max_iterations:]

    pcoords: dict[str, np.ndarray] = {}

    for group_path in iteration_groups:
        try:
            pcoords[group_path] = read_iteration_pcoord(h5_path, group_path)
        except KeyError:
            continue

    return pcoords
