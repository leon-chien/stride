from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stride.data.atomistic import (
    AMINO_ACIDS,
    AtomRecord,
    AtomisticDataset,
    build_atomistic_windows,
)
from stride.goals import GoalSpec


WATER_RESNAMES = {"HOH", "WAT", "TIP", "TIP3", "SOL", "WATER"}
ION_ELEMENTS = {"NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU"}


def load_mdanalysis_trajectory(
    topology_path: str | Path,
    trajectory_path: str | Path | None = None,
    mda_selection: str | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    coordinate_units: str = "nm",
) -> tuple[np.ndarray, list[AtomRecord]]:
    """
    Load topology + trajectory files through MDAnalysis.

    STRIDE stores real-data coordinates in nanometers. MDAnalysis positions are
    usually Angstroms, so the default conversion divides positions by 10.
    """
    mda = _import_mdanalysis()
    universe = mda.Universe(str(topology_path), str(trajectory_path)) if trajectory_path else mda.Universe(str(topology_path))
    atoms = universe.select_atoms(mda_selection) if mda_selection else universe.atoms
    if len(atoms) == 0:
        raise ValueError(f"MDAnalysis selection matched no atoms: {mda_selection}")

    atom_records = [_atom_record_from_mda_atom(atom) for atom in atoms]
    scale = _coordinate_scale(coordinate_units)

    frames: list[np.ndarray] = []
    for timestep in universe.trajectory[slice(start, stop, step)]:
        _ = timestep
        frames.append(np.asarray(atoms.positions, dtype=np.float32) * scale)

    if not frames:
        raise ValueError("No trajectory frames were loaded.")

    return np.stack(frames).astype(np.float32), atom_records


def build_atomistic_dataset_from_mdanalysis(
    topology_path: str | Path,
    trajectory_path: str | Path | None,
    goal: GoalSpec,
    window_size: int,
    horizon: int | None = None,
    stride: int = 1,
    mda_selection: str | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    coordinate_units: str = "nm",
    flux_label_mode: str = "event",
) -> AtomisticDataset:
    coordinates, atoms = load_mdanalysis_trajectory(
        topology_path=topology_path,
        trajectory_path=trajectory_path,
        mda_selection=mda_selection,
        start=start,
        stop=stop,
        step=step,
        coordinate_units=coordinate_units,
    )
    return build_atomistic_windows(
        coordinates=coordinates,
        atoms=atoms,
        goal=goal,
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        flux_label_mode=flux_label_mode,
    )


def _import_mdanalysis() -> Any:
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise ImportError(
            "MDAnalysis is required for topology+trajectory conversion. "
            "Install project dependencies with `pip install -e .` or "
            "`conda env update -f environment.yml`."
        ) from exc
    return mda


def _atom_record_from_mda_atom(atom: Any) -> AtomRecord:
    residue_name = str(_safe_attr(atom, "resname", "UNK")).upper()
    element = str(_safe_attr(atom, "element", "") or _safe_attr(atom, "type", "OTHER"))
    element = _canonical_mda_element(element)
    is_water = residue_name in WATER_RESNAMES
    is_ion = element in ION_ELEMENTS and residue_name not in AMINO_ACIDS[:-1]
    is_protein_residue = residue_name in AMINO_ACIDS[:-1]
    is_ligand = not is_protein_residue and not is_water and not is_ion

    return AtomRecord(
        atom_name=str(_safe_attr(atom, "name", "")),
        element=element,
        residue_name=residue_name,
        residue_id=int(_safe_attr(atom, "resid", 0)),
        chain_id=str(_safe_attr(atom, "segid", "") or ""),
        atom_type=str(_safe_attr(atom, "type", "") or ""),
        charge=float(_safe_attr(atom, "charge", 0.0) or 0.0),
        mass=float(_safe_attr(atom, "mass", 0.0) or 0.0),
        is_ligand=is_ligand,
        is_water=is_water,
        is_ion=is_ion,
    )


def _safe_attr(obj: Any, name: str, default: Any) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _coordinate_scale(coordinate_units: str) -> float:
    value = coordinate_units.strip().lower()
    if value in {"nm", "nanometer", "nanometers"}:
        return 0.1
    if value in {"angstrom", "angstroms", "a"}:
        return 1.0
    raise ValueError("coordinate_units must be 'nm' or 'angstrom'.")


def _canonical_mda_element(element: str) -> str:
    value = "".join(char for char in element.strip().upper() if char.isalpha())
    if len(value) >= 2 and value[:2] in ION_ELEMENTS:
        return value[:2]
    if value:
        return value[0]
    return "OTHER"
