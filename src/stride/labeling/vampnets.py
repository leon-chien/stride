from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from deeptime.decomposition import TICA
from sklearn.cluster import MiniBatchKMeans

from stride.data.domains import read_domain_list


@dataclass(frozen=True, slots=True)
class VampnetPretrainConfig:
    data_root: Path
    out_root: Path
    domains_path: Path | None = None
    resolutions: tuple[int, ...] = (4, 16, 64)
    max_domains: int | None = None
    lagtime: int = 1
    tica_dim: int = 8
    feature_residues: int = 32
    max_frames_per_domain: int = 20_000
    health_margin: float = 0.0
    seed: int = 0
    force: bool = False


def pretrain_vampnets(config: VampnetPretrainConfig) -> dict[str, Any]:
    manifest_path = config.out_root / "vampnet_manifest.json"
    if manifest_path.exists() and not config.force:
        msg = f"label manifest already exists: {manifest_path}; pass --force to overwrite"
        raise FileExistsError(msg)
    config.out_root.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(config.data_root / "metadata.parquet")
    stage_a_manifest = json.loads((config.data_root / "manifest.json").read_text(encoding="utf-8"))
    domains = _selected_domains(metadata, config.domains_path, config.max_domains)
    domain_results = []
    for domain_id in domains:
        features, offsets = _domain_features(metadata, config.data_root, domain_id, config)
        labels_by_resolution: dict[str, dict[str, Any]] = {}
        domain_dir = config.out_root / domain_id
        domain_dir.mkdir(parents=True, exist_ok=True)
        for resolution in config.resolutions:
            labels, result = _fit_labels(features, resolution, config)
            np.save(domain_dir / f"vamp_{resolution}.npy", labels.astype(np.int64))
            labels_by_resolution[str(resolution)] = result
        domain_results.append(
            {
                "domain_id": domain_id,
                "n_frames": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "trajectory_offsets": offsets,
                "resolutions": labels_by_resolution,
            }
        )
    manifest = {
        "stage": "A0",
        "label_type": "vampnet_tier1_pilot",
        "dataset_hash": stage_a_manifest["dataset_hash"],
        "data_root": str(config.data_root),
        "out_root": str(config.out_root),
        "domains_path": str(config.domains_path) if config.domains_path else None,
        "resolutions": list(config.resolutions),
        "lagtime": config.lagtime,
        "tica_dim": config.tica_dim,
        "health_margin": config.health_margin,
        "seed": config.seed,
        "domains": domain_results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _selected_domains(
    metadata: pd.DataFrame, domains_path: Path | None, max_domains: int | None
) -> list[str]:
    if domains_path is None:
        domains = sorted(metadata["domain_id"].astype(str).unique().tolist())
    else:
        requested = read_domain_list(domains_path)
        available = set(metadata["domain_id"].astype(str).unique().tolist())
        domains = [domain for domain in requested if domain in available]
        missing = sorted(set(requested) - available)
        if missing:
            msg = f"requested domains missing from processed data: {missing[:10]}"
            raise ValueError(msg)
    if max_domains is not None:
        domains = domains[:max_domains]
    if not domains:
        msg = "no domains selected for VAMPnet pretraining"
        raise ValueError(msg)
    return domains


def _domain_features(
    metadata: pd.DataFrame, data_root: Path, domain_id: str, config: VampnetPretrainConfig
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], list[dict[str, Any]]]:
    rows = metadata[metadata["domain_id"] == domain_id].sort_values("traj_index")
    coords = zarr.open_group(data_root / "coords.zarr", mode="r")["ca"]
    features: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    offsets: list[dict[str, Any]] = []
    cursor = 0
    for _, row in rows.iterrows():
        n_frames = int(row["n_frames"])
        if config.max_frames_per_domain > 0:
            remaining = config.max_frames_per_domain - cursor
            if remaining <= 0:
                break
            n_frames = min(n_frames, remaining)
        n_res = int(row["n_res"])
        traj_index = int(row["traj_index"])
        ca = np.asarray(coords[traj_index, :n_frames, :n_res, :], dtype=np.float32)
        frame_features = _ca_features(ca, config.feature_residues)
        features.append(frame_features)
        offsets.append(
            {
                "traj_id": str(row["traj_id"]),
                "traj_index": traj_index,
                "start": cursor,
                "stop": cursor + int(frame_features.shape[0]),
            }
        )
        cursor += int(frame_features.shape[0])
    if not features:
        msg = f"no frames available for domain {domain_id}"
        raise ValueError(msg)
    return np.concatenate(features, axis=0), offsets


def _ca_features(
    ca: np.ndarray[Any, np.dtype[np.float32]], feature_residues: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    n_frames, n_res, _ = ca.shape
    count = min(feature_residues, n_res)
    residue_indices = np.linspace(0, n_res - 1, count, dtype=np.int64)
    selected = ca[:, residue_indices, :]
    centered = selected - selected.mean(axis=1, keepdims=True)
    flattened = centered.reshape(n_frames, count * 3)
    if count < feature_residues:
        padded = np.zeros((n_frames, feature_residues * 3), dtype=np.float32)
        padded[:, : flattened.shape[1]] = flattened
        flattened = padded
    scale = flattened.std(axis=0, keepdims=True).clip(min=1e-6)
    return np.asarray((flattened - flattened.mean(axis=0, keepdims=True)) / scale, dtype=np.float32)


def _fit_labels(
    features: np.ndarray[Any, np.dtype[np.float32]], resolution: int, config: VampnetPretrainConfig
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], dict[str, Any]]:
    embedded, score = _tica_embedding(features, config.lagtime, config.tica_dim)
    baseline = _random_projection_score(features, config)
    passed = score > baseline + config.health_margin
    method = "vampnet" if passed else "tica_kmeans"
    labels = _cluster(embedded, resolution, config.seed)
    result = {
        "method": method,
        "n_states": int(min(resolution, len(features))),
        "health_score": score,
        "random_projection_baseline": baseline,
        "health_margin": config.health_margin,
        "health_passed": bool(passed),
        "fallback_reason": None if passed else "health_score_not_above_random_projection_baseline",
    }
    return labels, result


def _tica_embedding(
    features: np.ndarray[Any, np.dtype[np.float32]], lagtime: int, tica_dim: int
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
    dim = max(1, min(tica_dim, features.shape[1], features.shape[0] - 1))
    estimator = TICA(lagtime=max(1, min(lagtime, features.shape[0] - 2)), dim=dim).fit(features)
    model = estimator.fetch_model()
    if model is None:
        msg = "deeptime TICA did not produce a model"
        raise ValueError(msg)
    embedded = np.asarray(model.transform(features), dtype=np.float32)
    singular_values = np.asarray(model.singular_values[:dim], dtype=np.float64)
    score = float(np.sum(np.square(singular_values)))
    return embedded, score


def _random_projection_score(
    features: np.ndarray[Any, np.dtype[np.float32]], config: VampnetPretrainConfig
) -> float:
    rng = np.random.default_rng(config.seed)
    dim = max(1, min(config.tica_dim, features.shape[1]))
    projection = rng.normal(size=(features.shape[1], dim)).astype(np.float32)
    projected = features @ projection
    _, score = _tica_embedding(projected, config.lagtime, dim)
    return score


def _cluster(
    embedded: np.ndarray[Any, np.dtype[np.float32]], resolution: int, seed: int
) -> np.ndarray[Any, np.dtype[np.int64]]:
    n_clusters = max(1, min(resolution, embedded.shape[0]))
    if n_clusters == 1:
        return np.zeros(embedded.shape[0], dtype=np.int64)
    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(1024, max(n_clusters * 4, 16)),
        n_init=5,
    )
    return np.asarray(clusterer.fit_predict(embedded), dtype=np.int64)
