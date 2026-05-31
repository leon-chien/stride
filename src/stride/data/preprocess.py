from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import zarr

from stride.data.domains import mdcath_h5_path, read_domain_list
from stride.data.features import (
    TrajectoryFeatures,
    inspect_h5_schema,
    iter_trajectory_groups,
    load_domain_topology,
    load_trajectory_features,
)
from stride.data.splits import write_topology_splits


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    input_root: Path
    output_root: Path
    domains_path: Path
    workers: int = 1
    force: bool = False
    dry_run: bool = False
    frame_dt_ps: float = 1000.0
    filter_high_temp_unfolded: bool = True


@dataclass(frozen=True, slots=True)
class PlannedTrajectory:
    domain_id: str
    h5_path: Path
    temperature_kelvin: int
    replicate: int
    group_path: str
    n_frames: int
    n_residues: int

    @property
    def traj_id(self) -> str:
        return f"{self.domain_id}_{self.temperature_kelvin}_{self.replicate}"


def preprocess_mdcath(config: PreprocessConfig) -> dict[str, Any]:
    domains = read_domain_list(config.domains_path)
    planned = plan_mdcath_preprocess(config.input_root, domains)
    summary = {
        "domains": domains,
        "n_domains": len(domains),
        "n_trajectories": len(planned),
        "max_frames": max((traj.n_frames for traj in planned), default=0),
        "max_residues": max((traj.n_residues for traj in planned), default=0),
    }
    if config.dry_run:
        return summary
    if not planned:
        msg = "no trajectories planned"
        raise ValueError(msg)
    if config.output_root.exists() and not config.force:
        manifest = config.output_root / "manifest.json"
        if manifest.exists():
            msg = f"output already exists: {config.output_root}; pass --force to overwrite"
            raise FileExistsError(msg)
    config.output_root.mkdir(parents=True, exist_ok=True)
    max_frames = _round_up(summary["max_frames"], 256)
    max_residues = _round_up(summary["max_residues"], 512)
    writer = ZarrDatasetWriter(config.output_root, len(planned), max_frames, max_residues)
    metadata: list[dict[str, Any]] = []
    trajectory_hashes: list[str] = []
    for index, item in enumerate(planned):
        with h5py.File(item.h5_path, "r") as h5:
            topology = load_domain_topology(h5, item.domain_id)
            features = load_trajectory_features(
                h5,
                item.domain_id,
                item.temperature_kelvin,
                item.replicate,
                item.group_path,
                topology,
                frame_dt_ps=config.frame_dt_ps,
                filter_high_temp_unfolded=config.filter_high_temp_unfolded,
            )
        writer.write(index, features)
        traj_hash = hash_trajectory(features)
        trajectory_hashes.append(traj_hash)
        metadata.append(
            {
                "traj_index": index,
                "traj_id": features.traj_id,
                "domain_id": features.domain_id,
                "cath_topology": topology_key(features.domain_id),
                "temperature_kelvin": features.temperature_kelvin,
                "replicate": features.replicate,
                "n_frames": features.n_frames,
                "n_res": features.n_residues,
                "frame_dt_ps": features.frame_dt_ps,
                "ff_tag": 0,
                "raw_h5_path": str(item.h5_path),
                "sha256_chunk0": traj_hash,
            }
        )
    writer.close()
    metadata_frame = pd.DataFrame(metadata)
    metadata_frame.to_parquet(config.output_root / "metadata.parquet", index=False)
    write_topology_splits(metadata_frame, config.output_root / "splits" / "by_topology.json")
    manifest = build_manifest(
        config=config,
        metadata=metadata,
        trajectory_hashes=trajectory_hashes,
        max_frames=max_frames,
        max_residues=max_residues,
    )
    (config.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def plan_mdcath_preprocess(input_root: Path, domains: list[str]) -> list[PlannedTrajectory]:
    planned: list[PlannedTrajectory] = []
    for domain_id in domains:
        h5_path = mdcath_h5_path(input_root, domain_id)
        if not h5_path.exists():
            msg = f"missing mdCATH HDF5 for {domain_id}: expected {h5_path}"
            raise FileNotFoundError(msg)
        try:
            trajectories = iter_trajectory_groups(h5_path, domain_id)
            with h5py.File(h5_path, "r") as h5:
                n_residues = int(h5[domain_id].attrs["numResidues"])
                for temperature_kelvin, replicate, group_path in trajectories:
                    n_frames = int(
                        h5[group_path].attrs.get("numFrames", h5[group_path]["coords"].shape[0])
                    )
                    planned.append(
                        PlannedTrajectory(
                            domain_id=domain_id,
                            h5_path=h5_path,
                            temperature_kelvin=temperature_kelvin,
                            replicate=replicate,
                            group_path=group_path,
                            n_frames=n_frames,
                            n_residues=n_residues,
                        )
                    )
        except Exception as exc:
            schema_path = h5_path.with_suffix(".schema.json")
            schema_path.write_text(
                json.dumps(inspect_h5_schema(h5_path), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            msg = f"failed to inspect {h5_path}; wrote schema dump to {schema_path}"
            raise ValueError(msg) from exc
    return planned


class ZarrDatasetWriter:
    def __init__(self, output_root: Path, n_traj: int, max_frames: int, max_residues: int) -> None:
        self.output_root = output_root
        self.n_traj = n_traj
        self.max_frames = max_frames
        self.max_residues = max_residues
        self.coords = zarr.open_group(output_root / "coords.zarr", mode="w")
        self.features = zarr.open_group(output_root / "features.zarr", mode="w")
        self.mask = zarr.open_group(output_root / "residue_mask.zarr", mode="w")
        self.ca = _create_array(
            self.coords,
            "ca",
            shape=(n_traj, max_frames, max_residues, 3),
            chunks=(1, 256, 512, 3),
            dtype="f2",
            fill_value=np.nan,
        )
        self.cb = _create_array(
            self.coords,
            "cb",
            shape=(n_traj, max_frames, max_residues, 3),
            chunks=(1, 256, 512, 3),
            dtype="f2",
            fill_value=np.nan,
        )
        self.torsions = _create_array(
            self.features,
            "backbone_torsions",
            shape=(n_traj, max_frames, max_residues, 6),
            chunks=(1, 256, 512, 6),
            dtype="f2",
            fill_value=0,
        )
        self.dssp = _create_array(
            self.features,
            "dssp",
            shape=(n_traj, max_frames, max_residues),
            chunks=(1, 256, 512),
            dtype="u1",
            fill_value=0,
        )
        self.sasa = _create_array(
            self.features,
            "sasa",
            shape=(n_traj, max_frames, max_residues),
            chunks=(1, 256, 512),
            dtype="f2",
            fill_value=np.nan,
        )
        self.residue_type = _create_array(
            self.features,
            "residue_type",
            shape=(n_traj, max_residues),
            chunks=(1, 512),
            dtype="u1",
            fill_value=0,
        )
        self.ff_tag = _create_array(
            self.features,
            "ff_tag",
            shape=(n_traj,),
            chunks=(1024,),
            dtype="u1",
            fill_value=0,
        )
        self.residue_mask = _create_array(
            self.mask,
            "mask",
            shape=(n_traj, max_residues),
            chunks=(1, 512),
            dtype="bool",
            fill_value=False,
        )

    def write(self, index: int, features: TrajectoryFeatures) -> None:
        frame_slice = slice(0, features.n_frames)
        residue_slice = slice(0, features.n_residues)
        self.ca[index, frame_slice, residue_slice, :] = features.ca.astype(np.float16)
        self.cb[index, frame_slice, residue_slice, :] = features.cb.astype(np.float16)
        self.torsions[index, frame_slice, residue_slice, :] = features.backbone_torsions.astype(
            np.float16
        )
        self.dssp[index, frame_slice, residue_slice] = features.dssp
        self.sasa[index, frame_slice, residue_slice] = features.sasa.astype(np.float16)
        self.residue_type[index, residue_slice] = features.residue_type
        self.ff_tag[index] = 0
        self.residue_mask[index, residue_slice] = features.residue_mask

    def close(self) -> None:
        return None


def hash_trajectory(features: TrajectoryFeatures) -> str:
    digest = hashlib.sha256()
    for array in [
        features.ca,
        features.cb,
        features.backbone_torsions,
        features.dssp,
        features.sasa,
        features.residue_type,
        features.residue_mask,
    ]:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_manifest(
    *,
    config: PreprocessConfig,
    metadata: list[dict[str, Any]],
    trajectory_hashes: list[str],
    max_frames: int,
    max_residues: int,
) -> dict[str, Any]:
    dataset_hash = hashlib.sha256("".join(sorted(trajectory_hashes)).encode("utf-8")).hexdigest()
    return {
        "dataset": "mdCATH",
        "source_repo": "compsciencelab/mdCATH",
        "domains_path": str(config.domains_path),
        "frame_dt_policy": "read_from_hdf5_when_available; default 1000 ps",
        "frame_dt_default_ps": config.frame_dt_ps,
        "filter_high_temp_unfolded": config.filter_high_temp_unfolded,
        "n_trajectories": len(metadata),
        "max_frames": max_frames,
        "max_residues": max_residues,
        "dataset_hash": dataset_hash,
        "trajectories": metadata,
    }


def topology_key(domain_id: str) -> str:
    return domain_id


def _create_array(
    group: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str,
    fill_value: float | int | bool,
) -> Any:
    kwargs: dict[str, Any] = _compression_kwargs()
    try:
        return group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            overwrite=True,
            **kwargs,
        )
    except TypeError:
        return group.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            overwrite=True,
            **kwargs,
        )


def _compression_kwargs() -> dict[str, Any]:
    try:
        from zarr.codecs import BloscCodec, BloscShuffle

        return {
            "compressors": [BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)]
        }
    except Exception:
        from numcodecs import Blosc

        return {"compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)}


def _round_up(value: int, multiple: int) -> int:
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def config_to_dict(config: PreprocessConfig) -> dict[str, Any]:
    out = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in out.items()}
