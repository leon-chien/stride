from __future__ import annotations

import numpy as np

from stride.data import (
    AtomRecord,
    atom_selection_mask,
    build_atomistic_windows,
    compute_dihedral_degrees,
)
from stride.goals import GoalSpec


def test_combined_atom_selection_and_dihedral_labeling() -> None:
    atoms = [
        AtomRecord(atom_name="C", element="C", residue_name="ACE", residue_id=1),
        AtomRecord(atom_name="N", element="N", residue_name="ALA", residue_id=2),
        AtomRecord(atom_name="CA", element="C", residue_name="ALA", residue_id=2),
        AtomRecord(atom_name="C", element="C", residue_name="ALA", residue_id=2),
    ]

    mask = atom_selection_mask(atoms, "resid:2&atom:CA")
    assert mask.tolist() == [False, False, True, False]

    coordinates = _dihedral_trajectory()
    assert compute_dihedral_degrees(coordinates[-1]) == 0.0

    dataset = build_atomistic_windows(
        coordinates=coordinates,
        atoms=atoms,
        goal=GoalSpec(
            name="alanine_phi_window",
            type="dihedral_window",
            selections=(
                "resid:1&atom:C",
                "resid:2&atom:N",
                "resid:2&atom:CA",
                "resid:2&atom:C",
            ),
            operator="inside",
            threshold=0.0,
            lower_bound=-10.0,
            upper_bound=10.0,
            horizon_iterations=2,
        ),
        window_size=3,
        horizon=2,
    )

    assert dataset.event_labels.tolist() == [0.0, 1.0, 1.0]


def _dihedral_trajectory() -> np.ndarray:
    coordinates = np.zeros((7, 4, 3), dtype=np.float32)
    base = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = base.copy()
    target[3] = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)

    for frame in range(5):
        coordinates[frame] = base
    coordinates[5:] = target
    return coordinates
