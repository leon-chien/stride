from __future__ import annotations

from pathlib import Path

import numpy as np

from stride.data.atomistic import (
    AtomRecord,
    AtomisticDataset,
    build_atomistic_windows,
    save_atomistic_dataset_npz,
)
from stride.goals import GoalSpec


def sample_ligand_contact_atoms() -> list[AtomRecord]:
    """
    Tiny protein-ligand topology used for local smoke tests.
    """
    return [
        AtomRecord(
            atom_name="OD1",
            element="O",
            residue_name="ASP",
            residue_id=42,
            chain_id="A",
            mass=16.0,
        ),
        AtomRecord(
            atom_name="CG",
            element="C",
            residue_name="ASP",
            residue_id=42,
            chain_id="A",
            mass=12.0,
        ),
        AtomRecord(
            atom_name="O1",
            element="O",
            residue_name="LIG",
            residue_id=1,
            chain_id="L",
            is_ligand=True,
            mass=16.0,
        ),
        AtomRecord(
            atom_name="C1",
            element="C",
            residue_name="LIG",
            residue_id=1,
            chain_id="L",
            is_ligand=True,
            mass=12.0,
        ),
    ]


def sample_ligand_contact_goal() -> GoalSpec:
    return GoalSpec(
        name="ligand_contact_asp42",
        type="contact",
        selections=("ligand", "ASP42"),
        operator="less_than",
        threshold=0.45,
        horizon_iterations=2,
    )


def sample_ligand_contact_coordinates(num_frames: int = 12) -> np.ndarray:
    """
    Generate a short trajectory where a two-atom ligand approaches ASP42.
    """
    if num_frames < 6:
        raise ValueError("num_frames must be at least 6.")

    coordinates = np.zeros((num_frames, 4, 3), dtype=np.float32)
    coordinates[:, 0, :] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    coordinates[:, 1, :] = np.asarray([0.25, 0.0, 0.0], dtype=np.float32)

    ligand_distances = np.linspace(1.6, 0.22, num_frames, dtype=np.float32)
    for frame_index, distance in enumerate(ligand_distances):
        coordinates[frame_index, 2, :] = np.asarray(
            [float(distance), 0.0, 0.0],
            dtype=np.float32,
        )
        coordinates[frame_index, 3, :] = np.asarray(
            [float(distance), 0.8, 0.0],
            dtype=np.float32,
        )

    return coordinates


def build_sample_ligand_contact_dataset(
    goal: GoalSpec | None = None,
    window_size: int = 4,
    horizon: int = 2,
    stride: int = 1,
    num_frames: int = 12,
) -> AtomisticDataset:
    return build_atomistic_windows(
        coordinates=sample_ligand_contact_coordinates(num_frames=num_frames),
        atoms=sample_ligand_contact_atoms(),
        goal=goal or sample_ligand_contact_goal(),
        window_size=window_size,
        horizon=horizon,
        stride=stride,
    )


def write_sample_ligand_contact_dataset(
    output_npz: str | Path,
    pdb_output: str | Path | None = None,
    goal: GoalSpec | None = None,
    window_size: int = 4,
    horizon: int = 2,
    stride: int = 1,
    num_frames: int = 12,
) -> AtomisticDataset:
    dataset = build_sample_ligand_contact_dataset(
        goal=goal,
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        num_frames=num_frames,
    )
    save_atomistic_dataset_npz(output_npz, dataset)

    if pdb_output is not None:
        write_sample_ligand_contact_pdb(pdb_output, num_frames=num_frames)

    return dataset


def write_sample_ligand_contact_pdb(
    path: str | Path,
    num_frames: int = 12,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    atoms = sample_ligand_contact_atoms()
    coordinates = sample_ligand_contact_coordinates(num_frames=num_frames)

    with open(path, "w") as handle:
        for model_index, frame in enumerate(coordinates, start=1):
            handle.write(f"MODEL     {model_index:4d}\n")
            for atom_index, (atom, xyz) in enumerate(
                zip(atoms, frame, strict=True),
                start=1,
            ):
                record = "HETATM" if atom.is_ligand else "ATOM  "
                handle.write(
                    f"{record}{atom_index:5d} {atom.atom_name:<4s}"
                    f" {atom.residue_name:>3s} {atom.chain_id:1s}"
                    f"{atom.residue_id:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"  1.00  0.00          {atom.element:>2s}\n"
                )
            handle.write("ENDMDL\n")
        handle.write("END\n")
