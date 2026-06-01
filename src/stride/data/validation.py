from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import zarr

from stride.data.domains import mdcath_h5_path, read_domain_list
from stride.data.features import decode_h5_scalar, load_domain_topology


@dataclass(frozen=True, slots=True)
class StageAValidationConfig:
    data_root: Path
    input_root: Path
    domains_path: Path
    benchmark_windows: int = 100
    window_frames: int = 64
    seed: int = 0
    rmsd_threshold_angstrom: float = 0.05
    p50_threshold_ms: float = 50.0
    p99_threshold_ms: float = 200.0


def validate_stage_a(config: StageAValidationConfig) -> dict[str, Any]:
    metadata = pd.read_parquet(config.data_root / "metadata.parquet")
    manifest = json.loads((config.data_root / "manifest.json").read_text(encoding="utf-8"))
    shapes = zarr_shapes(config.data_root)
    domain_check = domain_list_check(metadata, config.domains_path)
    hash_check = manifest_hash_check(metadata, manifest)
    rmsd = roundtrip_ca_rmsd(config.data_root, config.input_root, metadata)
    benchmark = random_window_benchmark(
        config.data_root,
        metadata,
        n_windows=config.benchmark_windows,
        window_frames=config.window_frames,
        seed=config.seed,
    )
    summary = {
        "dataset_hash": manifest["dataset_hash"],
        "n_domains": int(metadata["domain_id"].nunique()),
        "n_trajectories": len(metadata),
        "max_frames": int(manifest["max_frames"]),
        "max_residues": int(manifest["max_residues"]),
        "shapes": shapes,
        "domain_list": domain_check,
        "manifest_determinism": hash_check,
        "roundtrip_rmsd": {
            **rmsd,
            "threshold_angstrom": config.rmsd_threshold_angstrom,
            "passed": bool(rmsd["rmsd_angstrom"] < config.rmsd_threshold_angstrom),
        },
        "random_window_benchmark": {
            **benchmark,
            "p50_threshold_ms": config.p50_threshold_ms,
            "p99_threshold_ms": config.p99_threshold_ms,
            "passed": bool(
                benchmark["p50_ms"] < config.p50_threshold_ms
                and benchmark["p99_ms"] < config.p99_threshold_ms
            ),
        },
    }
    summary["passed"] = bool(
        domain_check["passed"]
        and hash_check["passed"]
        and summary["roundtrip_rmsd"]["passed"]
        and summary["random_window_benchmark"]["passed"]
    )
    return summary


def zarr_shapes(data_root: Path) -> dict[str, tuple[int, ...]]:
    coords = zarr.open_group(data_root / "coords.zarr", mode="r")
    features = zarr.open_group(data_root / "features.zarr", mode="r")
    mask = zarr.open_group(data_root / "residue_mask.zarr", mode="r")
    return {
        "coords.ca": tuple(coords["ca"].shape),
        "coords.cb": tuple(coords["cb"].shape),
        "features.backbone_torsions": tuple(features["backbone_torsions"].shape),
        "features.dssp": tuple(features["dssp"].shape),
        "features.sasa": tuple(features["sasa"].shape),
        "residue_mask.mask": tuple(mask["mask"].shape),
    }


def domain_list_check(metadata: pd.DataFrame, domains_path: Path) -> dict[str, Any]:
    expected = read_domain_list(domains_path)
    observed = sorted(metadata["domain_id"].astype(str).unique().tolist())
    return {
        "expected_count": len(expected),
        "observed_count": len(observed),
        "missing": sorted(set(expected) - set(observed)),
        "unexpected": sorted(set(observed) - set(expected)),
        "passed": set(expected) == set(observed),
    }


def manifest_hash_check(metadata: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    recomputed = hashlib.sha256(
        "".join(sorted(metadata["sha256_chunk0"].astype(str).tolist())).encode("utf-8")
    ).hexdigest()
    expected = str(manifest["dataset_hash"])
    return {
        "expected": expected,
        "recomputed": recomputed,
        "passed": recomputed == expected,
    }


def roundtrip_ca_rmsd(data_root: Path, input_root: Path, metadata: pd.DataFrame) -> dict[str, Any]:
    row = metadata.iloc[0]
    traj_index = int(row["traj_index"])
    domain_id = str(row["domain_id"])
    temperature_kelvin = int(row["temperature_kelvin"])
    replicate = int(row["replicate"])
    n_frames = int(row["n_frames"])
    n_res = int(row["n_res"])
    raw_h5_path = mdcath_h5_path(input_root, domain_id)
    raw_ca = _raw_ca(raw_h5_path, domain_id, temperature_kelvin, replicate)[:n_frames, :n_res, :]
    coords = zarr.open_group(data_root / "coords.zarr", mode="r")
    zarr_ca = np.asarray(coords["ca"][traj_index, :n_frames, :n_res, :], dtype=np.float32)
    diff = raw_ca.astype(np.float32) - zarr_ca
    rmsd = float(np.sqrt(np.mean(np.square(diff))))
    return {
        "traj_id": str(row["traj_id"]),
        "traj_index": traj_index,
        "n_frames": n_frames,
        "n_res": n_res,
        "rmsd_angstrom": rmsd,
    }


def random_window_benchmark(
    data_root: Path,
    metadata: pd.DataFrame,
    *,
    n_windows: int,
    window_frames: int,
    seed: int,
) -> dict[str, Any]:
    eligible = metadata[metadata["n_frames"] >= window_frames]
    if eligible.empty:
        msg = f"no trajectories with at least {window_frames} frames"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    coords = zarr.open_group(data_root / "coords.zarr", mode="r")["ca"]
    durations_ms: list[float] = []
    rows = eligible.reset_index(drop=True)
    for _ in range(n_windows):
        row = rows.iloc[int(rng.integers(0, len(rows)))]
        n_frames = int(row["n_frames"])
        n_res = int(row["n_res"])
        start = int(rng.integers(0, n_frames - window_frames + 1))
        traj_index = int(row["traj_index"])
        before = time.perf_counter()
        _ = np.asarray(coords[traj_index, start : start + window_frames, :n_res, :])
        durations_ms.append((time.perf_counter() - before) * 1000.0)
    values = np.asarray(durations_ms, dtype=np.float64)
    return {
        "n_windows": int(n_windows),
        "window_frames": int(window_frames),
        "seed": int(seed),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def _raw_ca(
    h5_path: Path, domain_id: str, temperature_kelvin: int, replicate: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    with h5py.File(h5_path, "r") as h5:
        topology = load_domain_topology(h5, domain_id)
        group = h5[f"/{domain_id}/{temperature_kelvin}/{replicate}"]
        coords = np.asarray(group["coords"][()], dtype=np.float32)
        rmsd_dataset = group.get("rmsd")
        if temperature_kelvin == 450 and rmsd_dataset is not None:
            rmsd = np.asarray(rmsd_dataset[()], dtype=np.float32)
            unit = decode_h5_scalar(rmsd_dataset.attrs.get("unit", ""))
            if unit.lower() == "nm":
                rmsd *= 10.0
            coords = coords[rmsd <= 4.0]
        return np.asarray(coords[:, topology.ca_index, :], dtype=np.float32)
