from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

AA3_TO_INDEX = {
    "ALA": 1,
    "ARG": 2,
    "ASN": 3,
    "ASP": 4,
    "CYS": 5,
    "GLN": 6,
    "GLU": 7,
    "GLY": 8,
    "HIS": 9,
    "ILE": 10,
    "LEU": 11,
    "LYS": 12,
    "MET": 13,
    "PHE": 14,
    "PRO": 15,
    "SER": 16,
    "THR": 17,
    "TRP": 18,
    "TYR": 19,
    "VAL": 20,
}

DSSP_TO_INDEX = {
    "C": 0,
    " ": 0,
    "-": 0,
    "H": 1,
    "B": 2,
    "E": 3,
    "G": 4,
    "I": 5,
    "T": 6,
    "S": 7,
}


@dataclass(frozen=True, slots=True)
class AtomRecord:
    index: int
    name: str
    residue_index: int
    residue_name: str
    residue_id: int
    chain: str
    element: str


@dataclass(frozen=True, slots=True)
class ResidueTopology:
    residue_ids: npt.NDArray[np.int64]
    residue_names: list[str]
    residue_type: npt.NDArray[np.uint8]
    atom_names: list[str]
    atom_indices_by_residue: list[npt.NDArray[np.int64]]
    n_index: npt.NDArray[np.int64]
    ca_index: npt.NDArray[np.int64]
    c_index: npt.NDArray[np.int64]
    cb_index: npt.NDArray[np.int64]

    @property
    def n_residues(self) -> int:
        return len(self.residue_names)


@dataclass(frozen=True, slots=True)
class TrajectoryFeatures:
    traj_id: str
    domain_id: str
    temperature_kelvin: int
    replicate: int
    frame_dt_ps: float
    ca: npt.NDArray[np.float32]
    cb: npt.NDArray[np.float32]
    backbone_torsions: npt.NDArray[np.float32]
    dssp: npt.NDArray[np.uint8]
    sasa: npt.NDArray[np.float32]
    residue_type: npt.NDArray[np.uint8]
    residue_mask: npt.NDArray[np.bool_]
    source_rmsd_angstrom: npt.NDArray[np.float32]

    @property
    def n_frames(self) -> int:
        return int(self.ca.shape[0])

    @property
    def n_residues(self) -> int:
        return int(self.ca.shape[1])


def decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return decode_h5_scalar(value.item())
    return str(value)


def inspect_h5_schema(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        schema: dict[str, Any] = {"attrs": _json_attrs(h5.attrs), "items": {}}

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if name.count("/") > 3:
                return
            entry: dict[str, Any] = {"type": type(obj).__name__, "attrs": _json_attrs(obj.attrs)}
            if isinstance(obj, h5py.Dataset):
                entry["shape"] = list(obj.shape)
                entry["dtype"] = str(obj.dtype)
            schema["items"][name] = entry

        h5.visititems(visitor)
        return schema


def load_domain_topology(h5: h5py.File, domain_id: str) -> ResidueTopology:
    group = h5[domain_id]
    pdb_text = decode_h5_scalar(group["pdbProteinAtoms"][()])
    atom_names = _atom_names_from_pdb(pdb_text)
    resnames = _decode_array(group["resname"][()])
    chains = _decode_array(group["chain"][()])
    elements = _decode_array(group["element"][()])
    resids = np.asarray(group["resid"][()], dtype=np.int64)
    if len(atom_names) != len(resids):
        msg = f"PDB atom count {len(atom_names)} does not match metadata atom count {len(resids)}"
        raise ValueError(msg)

    residue_keys: list[tuple[str, int, str]] = []
    residue_index_by_key: dict[tuple[str, int, str], int] = {}
    atoms: list[AtomRecord] = []
    atom_indices_by_residue: list[list[int]] = []
    for atom_index, (name, residue_name, chain, element, residue_id) in enumerate(
        zip(atom_names, resnames, chains, elements, resids, strict=True)
    ):
        key = (chain, int(residue_id), residue_name)
        if key not in residue_index_by_key:
            residue_index_by_key[key] = len(residue_keys)
            residue_keys.append(key)
            atom_indices_by_residue.append([])
        residue_index = residue_index_by_key[key]
        atom_indices_by_residue[residue_index].append(atom_index)
        atoms.append(
            AtomRecord(
                index=atom_index,
                name=name,
                residue_index=residue_index,
                residue_name=residue_name,
                residue_id=int(residue_id),
                chain=chain,
                element=element,
            )
        )

    n_residues = len(residue_keys)
    n_index = np.full(n_residues, -1, dtype=np.int64)
    ca_index = np.full(n_residues, -1, dtype=np.int64)
    c_index = np.full(n_residues, -1, dtype=np.int64)
    cb_index = np.full(n_residues, -1, dtype=np.int64)
    for atom in atoms:
        if atom.name == "N":
            n_index[atom.residue_index] = atom.index
        elif atom.name == "CA":
            ca_index[atom.residue_index] = atom.index
        elif atom.name == "C":
            c_index[atom.residue_index] = atom.index
        elif atom.name == "CB":
            cb_index[atom.residue_index] = atom.index
    if np.any(ca_index < 0):
        missing = np.flatnonzero(ca_index < 0)[:10].tolist()
        msg = f"missing CA atoms for residues {missing}"
        raise ValueError(msg)

    residue_names = [key[2] for key in residue_keys]
    residue_type = np.asarray([AA3_TO_INDEX.get(name, 0) for name in residue_names], dtype=np.uint8)
    residue_ids = np.asarray([key[1] for key in residue_keys], dtype=np.int64)
    atom_index_arrays = [np.asarray(indices, dtype=np.int64) for indices in atom_indices_by_residue]
    return ResidueTopology(
        residue_ids=residue_ids,
        residue_names=residue_names,
        residue_type=residue_type,
        atom_names=atom_names,
        atom_indices_by_residue=atom_index_arrays,
        n_index=n_index,
        ca_index=ca_index,
        c_index=c_index,
        cb_index=cb_index,
    )


def iter_trajectory_groups(path: Path, domain_id: str) -> list[tuple[int, int, str]]:
    with h5py.File(path, "r") as h5:
        if domain_id not in h5:
            msg = f"domain {domain_id!r} not found in {path}"
            raise ValueError(msg)
        domain_group = h5[domain_id]
        trajectory_paths: list[tuple[int, int, str]] = []
        for temperature_key in sorted(domain_group.keys()):
            if not temperature_key.isdigit():
                continue
            temp_group = domain_group[temperature_key]
            if not isinstance(temp_group, h5py.Group):
                continue
            for replicate_key in sorted(temp_group.keys(), key=int):
                rep_group = temp_group[replicate_key]
                if isinstance(rep_group, h5py.Group) and "coords" in rep_group:
                    trajectory_paths.append(
                        (int(temperature_key), int(replicate_key), rep_group.name)
                    )
        if not trajectory_paths:
            msg = f"no trajectory groups found for {domain_id} in {path}"
            raise ValueError(msg)
        return trajectory_paths


def load_trajectory_features(
    h5: h5py.File,
    domain_id: str,
    temperature_kelvin: int,
    replicate: int,
    group_path: str,
    topology: ResidueTopology,
    *,
    frame_dt_ps: float,
    filter_high_temp_unfolded: bool,
) -> TrajectoryFeatures:
    group = h5[group_path]
    coords = np.asarray(group["coords"][()], dtype=np.float32)
    rmsd_angstrom = _rmsd_angstrom(group)
    keep = np.ones(coords.shape[0], dtype=bool)
    if filter_high_temp_unfolded and temperature_kelvin == 450 and rmsd_angstrom.size:
        keep = rmsd_angstrom <= 4.0
    coords = coords[keep]
    rmsd_angstrom = rmsd_angstrom[keep] if rmsd_angstrom.size else rmsd_angstrom
    ca = coords[:, topology.ca_index, :]
    cb = virtual_cb(coords, topology)
    torsions = backbone_torsion_sincos(coords, topology)
    dssp = dssp_to_uint8(np.asarray(group["dssp"][()]))[keep]
    sasa = residue_sasa(coords, topology)
    residue_mask = np.ones(topology.n_residues, dtype=np.bool_)
    return TrajectoryFeatures(
        traj_id=f"{domain_id}_{temperature_kelvin}_{replicate}",
        domain_id=domain_id,
        temperature_kelvin=temperature_kelvin,
        replicate=replicate,
        frame_dt_ps=frame_dt_ps,
        ca=ca,
        cb=cb,
        backbone_torsions=torsions,
        dssp=dssp,
        sasa=sasa,
        residue_type=topology.residue_type,
        residue_mask=residue_mask,
        source_rmsd_angstrom=rmsd_angstrom,
    )


def virtual_cb(
    coords: npt.NDArray[np.float32], topology: ResidueTopology
) -> npt.NDArray[np.float32]:
    ca = coords[:, topology.ca_index, :]
    cb = np.empty_like(ca)
    has_real_cb = topology.cb_index >= 0
    if np.any(has_real_cb):
        cb[:, has_real_cb, :] = coords[:, topology.cb_index[has_real_cb], :]
    need_virtual = ~has_real_cb
    valid = need_virtual & (topology.n_index >= 0) & (topology.c_index >= 0)
    if np.any(valid):
        n = coords[:, topology.n_index[valid], :]
        ca_valid = ca[:, valid, :]
        c = coords[:, topology.c_index[valid], :]
        b = ca_valid - n
        c_vec = c - ca_valid
        a = np.cross(b, c_vec)
        cb[:, valid, :] = ca_valid - 0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec
    invalid = need_virtual & ~valid
    if np.any(invalid):
        cb[:, invalid, :] = ca[:, invalid, :]
    return cb


def backbone_torsion_sincos(
    coords: npt.NDArray[np.float32], topology: ResidueTopology
) -> npt.NDArray[np.float32]:
    n_frames = coords.shape[0]
    n_residues = topology.n_residues
    out = np.zeros((n_frames, n_residues, 6), dtype=np.float32)
    n = coords[:, topology.n_index.clip(min=0), :]
    ca = coords[:, topology.ca_index, :]
    c = coords[:, topology.c_index.clip(min=0), :]
    for residue in range(n_residues):
        if residue > 0 and _has_atoms(
            topology.c_index[residue - 1],
            topology.n_index[residue],
            topology.ca_index[residue],
            topology.c_index[residue],
        ):
            angle = _dihedral(c[:, residue - 1], n[:, residue], ca[:, residue], c[:, residue])
            out[:, residue, 0] = np.sin(angle)
            out[:, residue, 1] = np.cos(angle)
        if residue + 1 < n_residues and _has_atoms(
            topology.n_index[residue],
            topology.ca_index[residue],
            topology.c_index[residue],
            topology.n_index[residue + 1],
        ):
            angle = _dihedral(n[:, residue], ca[:, residue], c[:, residue], n[:, residue + 1])
            out[:, residue, 2] = np.sin(angle)
            out[:, residue, 3] = np.cos(angle)
        if residue + 1 < n_residues and _has_atoms(
            topology.ca_index[residue],
            topology.c_index[residue],
            topology.n_index[residue + 1],
            topology.ca_index[residue + 1],
        ):
            angle = _dihedral(ca[:, residue], c[:, residue], n[:, residue + 1], ca[:, residue + 1])
            out[:, residue, 4] = np.sin(angle)
            out[:, residue, 5] = np.cos(angle)
    return out


def dssp_to_uint8(raw: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    out = np.zeros(raw.shape, dtype=np.uint8)
    for value, index in np.ndenumerate(raw):
        label = decode_h5_scalar(index).strip() or "C"
        out[value] = DSSP_TO_INDEX.get(label[0], 0)
    return out


def residue_sasa(
    coords: npt.NDArray[np.float32], topology: ResidueTopology
) -> npt.NDArray[np.float32]:
    try:
        import freesasa
    except ImportError:
        return np.full((coords.shape[0], topology.n_residues), np.nan, dtype=np.float32)

    classifier = freesasa.Classifier()
    radii = [
        classifier.radius(record_residue_name, record_atom_name)
        for record_residue_name, record_atom_name in _residue_atom_pairs(topology)
    ]
    radii_array = np.asarray(radii, dtype=np.float64)
    out = np.zeros((coords.shape[0], topology.n_residues), dtype=np.float32)
    for frame_index, frame_coords in enumerate(coords):
        result = freesasa.calcCoord(
            frame_coords.reshape(-1).astype(float).tolist(), radii_array.tolist()
        )
        atom_areas = np.asarray(
            [result.atomArea(i) for i in range(frame_coords.shape[0])], dtype=np.float32
        )
        for residue_index, atom_indices in enumerate(topology.atom_indices_by_residue):
            out[frame_index, residue_index] = float(atom_areas[atom_indices].sum())
    return out


def _residue_atom_pairs(topology: ResidueTopology) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    atom_count = len(topology.atom_names)
    residue_for_atom = ["UNK"] * atom_count
    for residue_index, atom_indices in enumerate(topology.atom_indices_by_residue):
        for atom_index in atom_indices:
            residue_for_atom[atom_index] = topology.residue_names[residue_index]
    for residue_name, atom_name in zip(residue_for_atom, topology.atom_names, strict=True):
        pairs.append((residue_name, atom_name))
    return pairs


def _dihedral(
    p0: npt.NDArray[np.float32],
    p1: npt.NDArray[np.float32],
    p2: npt.NDArray[np.float32],
    p3: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = b1 / np.linalg.norm(b1, axis=1, keepdims=True).clip(min=1e-8)
    v = b0 - (b0 * b1_norm).sum(axis=1, keepdims=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(axis=1, keepdims=True) * b1_norm
    x = (v * w).sum(axis=1)
    y = (np.cross(b1_norm, v) * w).sum(axis=1)
    return np.asarray([atan2(yy, xx) for yy, xx in zip(y, x, strict=True)], dtype=np.float32)


def _has_atoms(*indices: np.int64) -> bool:
    return all(int(index) >= 0 for index in indices)


def _rmsd_angstrom(group: h5py.Group) -> npt.NDArray[np.float32]:
    if "rmsd" not in group:
        return np.asarray([], dtype=np.float32)
    values = np.asarray(group["rmsd"][()], dtype=np.float32)
    unit = group["rmsd"].attrs.get("unit", "")
    if decode_h5_scalar(unit).lower() == "nm":
        values *= 10.0
    return values


def _atom_names_from_pdb(pdb_text: str) -> list[str]:
    names: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            names.append(line[12:16].strip())
    return names


def _decode_array(values: npt.NDArray[Any]) -> list[str]:
    return [decode_h5_scalar(value).strip() for value in values]


def _json_attrs(attrs: h5py.AttributeManager) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, bytes):
            out[key] = value.decode("utf-8", errors="replace")
        else:
            out[key] = value
    return out
