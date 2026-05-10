from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class H5DatasetInfo:
    """
    Metadata for one HDF5 dataset.
    """

    path: str
    shape: tuple[int, ...]
    dtype: str


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

        names = sorted(iteration_group.keys())

    return [f"/iterations/{name}" for name in names]


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