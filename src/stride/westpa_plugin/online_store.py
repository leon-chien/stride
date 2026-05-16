from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stride.training import PcoordLineageDataset, load_pcoord_lineage_dataset_npz


@dataclass(frozen=True)
class OnlineStoreSummary:
    path: Path
    num_examples: int
    positive_rate: float


class OnlineLineageStore:
    """
    Append-only pcoord lineage store for one adaptive WESTPA run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> PcoordLineageDataset:
        return load_pcoord_lineage_dataset_npz(self.path)

    def append(
        self,
        dataset: PcoordLineageDataset,
        metadata: dict[str, object] | None = None,
    ) -> OnlineStoreSummary:
        dataset.validate()
        if self.path.exists():
            dataset = concatenate_pcoord_lineage_datasets([self.load(), dataset])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _save_dataset(self.path, dataset)
        if metadata is not None:
            meta_path = self.path.with_suffix(self.path.suffix + ".metadata.json")
            meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return OnlineStoreSummary(
            path=self.path,
            num_examples=int(dataset.event_labels.shape[0]),
            positive_rate=float(np.mean(dataset.event_labels)),
        )


def concatenate_pcoord_lineage_datasets(
    datasets: list[PcoordLineageDataset],
) -> PcoordLineageDataset:
    if not datasets:
        raise ValueError("At least one dataset is required.")
    for dataset in datasets:
        dataset.validate()
    return PcoordLineageDataset(
        pcoord_windows=np.concatenate([d.pcoord_windows for d in datasets], axis=0),
        window_mask=np.concatenate([d.window_mask for d in datasets], axis=0),
        goal_features=np.concatenate([d.goal_features for d in datasets], axis=0),
        event_labels=np.concatenate([d.event_labels for d in datasets], axis=0),
        flux_labels=np.concatenate([d.flux_labels for d in datasets], axis=0),
        n_iter=np.concatenate([d.n_iter for d in datasets], axis=0),
        seg_id=np.concatenate([d.seg_id for d in datasets], axis=0),
        weights=_concat_optional(datasets, "weights", np.float64),
        cell_id=_concat_optional(datasets, "cell_id", str),
        goal_id=_concat_optional(datasets, "goal_id", str),
        pcoord_dim=_concat_optional(datasets, "pcoord_dim", np.int64),
        threshold=_concat_optional(datasets, "threshold", np.float32),
        horizon_iterations=_concat_optional(datasets, "horizon_iterations", np.int64),
    )


def _concat_optional(
    datasets: list[PcoordLineageDataset],
    field: str,
    dtype,
) -> np.ndarray | None:
    values = [getattr(dataset, field) for dataset in datasets]
    if all(value is None for value in values):
        return None
    filled: list[np.ndarray] = []
    for dataset, value in zip(datasets, values, strict=True):
        if value is None:
            filled.append(_default_optional(field, len(dataset.event_labels), dtype))
        else:
            filled.append(np.asarray(value, dtype=dtype))
    return np.concatenate(filled, axis=0)


def _default_optional(field: str, length: int, dtype) -> np.ndarray:
    if field in {"cell_id", "goal_id"}:
        return np.asarray(["unknown"] * length, dtype=str)
    return np.zeros((length,), dtype=dtype)


def _save_dataset(path: Path, dataset: PcoordLineageDataset) -> None:
    arrays: dict[str, np.ndarray] = {
        "pcoord_windows": dataset.pcoord_windows.astype(np.float32),
        "window_mask": dataset.window_mask.astype(bool),
        "goal_features": dataset.goal_features.astype(np.float32),
        "event_labels": dataset.event_labels.astype(np.float32),
        "flux_labels": dataset.flux_labels.astype(np.float32),
        "n_iter": dataset.n_iter.astype(np.int64),
        "seg_id": dataset.seg_id.astype(np.int64),
    }
    for name in (
        "weights",
        "cell_id",
        "goal_id",
        "pcoord_dim",
        "threshold",
        "horizon_iterations",
    ):
        value = getattr(dataset, name)
        if value is not None:
            arrays[name] = np.asarray(value)
    np.savez_compressed(path, **arrays)
