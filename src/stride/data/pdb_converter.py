from __future__ import annotations

from pathlib import Path

import numpy as np

from stride.data.atomistic import (
    AMINO_ACIDS,
    AtomRecord,
    AtomisticDataset,
    atom_selection_mask,
    build_atomistic_windows,
)
from stride.goals import GoalSpec


WATER_RESNAMES = {"HOH", "WAT", "TIP", "TIP3", "SOL"}
ION_ELEMENTS = {"NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU"}


def load_pdb_trajectory(path: str | Path) -> tuple[np.ndarray, list[AtomRecord]]:
    """
    Load a single-model or multi-model PDB file into coordinates and atom records.

    This intentionally covers the lightweight interchange format needed for
    early STRIDE datasets. Larger production trajectories should eventually use
    an MDAnalysis/OpenMM adapter that returns the same output contract.
    """
    path = Path(path)
    atoms: list[AtomRecord] | None = None
    frames: list[np.ndarray] = []
    current_atoms: list[AtomRecord] = []
    current_coordinates: list[list[float]] = []
    saw_model = False

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            record = line[:6].strip().upper()

            if record == "MODEL":
                saw_model = True
                current_atoms = []
                current_coordinates = []
                continue

            if record == "ENDMDL":
                _append_frame(frames, current_coordinates, atoms)
                if atoms is None:
                    atoms = current_atoms
                current_atoms = []
                current_coordinates = []
                continue

            if record not in {"ATOM", "HETATM"}:
                continue

            atom, xyz = _parse_pdb_atom_line(line, is_hetatm=record == "HETATM")
            current_atoms.append(atom)
            current_coordinates.append(xyz)

    if current_coordinates:
        _append_frame(frames, current_coordinates, atoms)
        if atoms is None:
            atoms = current_atoms

    if not frames or atoms is None:
        raise ValueError(f"No ATOM/HETATM coordinates found in {path}.")

    coordinates = np.stack(frames).astype(np.float32)
    if coordinates.shape[1] != len(atoms):
        raise ValueError("PDB models contain inconsistent atom counts.")
    if not saw_model and coordinates.shape[0] != 1:
        raise ValueError("Internal error while parsing single-model PDB.")

    return coordinates, atoms


def build_atomistic_dataset_from_pdb(
    pdb_path: str | Path,
    goal: GoalSpec,
    window_size: int,
    horizon: int | None = None,
    stride: int = 1,
    atom_selection: str | list[str] | tuple[str, ...] | None = None,
    flux_label_mode: str = "event",
) -> AtomisticDataset:
    """
    Convert a PDB trajectory directly into the canonical STRIDE dataset schema.
    """
    coordinates, atoms = load_pdb_trajectory(pdb_path)

    if atom_selection is not None:
        mask = atom_selection_mask(atoms, atom_selection)
        if not mask.any():
            raise ValueError(f"Atom selection matched no atoms: {atom_selection}")
        atoms = [atom for atom, keep in zip(atoms, mask, strict=True) if keep]
        coordinates = coordinates[:, mask, :]

    return build_atomistic_windows(
        coordinates=coordinates,
        atoms=atoms,
        goal=goal,
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        flux_label_mode=flux_label_mode,
    )


def _append_frame(
    frames: list[np.ndarray],
    coordinates: list[list[float]],
    reference_atoms: list[AtomRecord] | None,
) -> None:
    if not coordinates:
        return
    if reference_atoms is not None and len(coordinates) != len(reference_atoms):
        raise ValueError("PDB models contain inconsistent atom counts.")
    frames.append(np.asarray(coordinates, dtype=np.float32))


def _parse_pdb_atom_line(line: str, is_hetatm: bool) -> tuple[AtomRecord, list[float]]:
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip().upper()
    chain_id = line[21:22].strip()
    residue_id = int(line[22:26].strip())
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    element = _guess_element(line, atom_name)

    is_water = residue_name in WATER_RESNAMES
    is_ion = element in ION_ELEMENTS and residue_name not in AMINO_ACIDS[:-1]
    is_ligand = (
        is_hetatm
        and not is_water
        and not is_ion
        and residue_name not in AMINO_ACIDS[:-1]
    )

    atom = AtomRecord(
        atom_name=atom_name,
        element=element,
        residue_name=residue_name,
        residue_id=residue_id,
        chain_id=chain_id,
        is_ligand=is_ligand,
        is_water=is_water,
        is_ion=is_ion,
    )
    return atom, [x, y, z]


def _guess_element(line: str, atom_name: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element

    cleaned = "".join(char for char in atom_name.upper() if char.isalpha())
    if len(cleaned) >= 2 and cleaned[:2] in ION_ELEMENTS:
        return cleaned[:2]
    if cleaned:
        return cleaned[0]
    return "OTHER"
