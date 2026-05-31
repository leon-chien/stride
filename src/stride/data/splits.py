import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_topology_splits(
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    seed: int = 1729,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, list[str]]:
    topologies = sorted(str(value) for value in metadata["cath_topology"].unique())
    shuffled = _deterministic_shuffle(topologies, seed)
    n_total = len(shuffled)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    topology_splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    traj_splits = {
        split: sorted(
            str(row.traj_id)
            for row in metadata.itertuples()
            if str(row.cath_topology) in topology_values
        )
        for split, topology_values in topology_splits.items()
    }
    payload: dict[str, Any] = {
        "seed": seed,
        "split_unit": "cath_topology",
        "topologies": topology_splits,
        "trajectories": traj_splits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return traj_splits


def _deterministic_shuffle(values: list[str], seed: int) -> list[str]:
    import random

    rng = random.Random(seed)
    out = list(values)
    rng.shuffle(out)
    return out
