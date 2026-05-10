from __future__ import annotations

import numpy as np

from stride.data import (
    build_atomistic_dataset_from_pdb,
    load_pdb_trajectory,
    write_sample_ligand_contact_pdb,
)
from stride.goals import GoalSpec


def test_multi_model_pdb_converts_to_atomistic_dataset(tmp_path) -> None:
    pdb_path = tmp_path / "sample_ligand_contact.pdb"
    write_sample_ligand_contact_pdb(pdb_path, num_frames=12)

    coordinates, atoms = load_pdb_trajectory(pdb_path)

    assert coordinates.shape == (12, 4, 3)
    assert atoms[0].residue_name == "ASP"
    assert atoms[2].is_ligand
    assert np.isclose(coordinates[-1, 2, 0], 0.22, atol=0.001)

    dataset = build_atomistic_dataset_from_pdb(
        pdb_path=pdb_path,
        goal=_goal(),
        window_size=4,
        horizon=2,
    )

    assert dataset.coordinates.shape == (7, 4, 4, 3)
    assert dataset.atom_features.shape[0:2] == (7, 4)
    assert dataset.event_labels.mean() > 0.0
    assert dataset.event_labels.mean() < 1.0


def test_pdb_converter_supports_atom_subset_selection(tmp_path) -> None:
    pdb_path = tmp_path / "sample_ligand_contact.pdb"
    write_sample_ligand_contact_pdb(pdb_path, num_frames=12)

    dataset = build_atomistic_dataset_from_pdb(
        pdb_path=pdb_path,
        goal=_goal(),
        window_size=4,
        horizon=2,
        atom_selection=("ASP42", "ligand"),
    )

    assert dataset.coordinates.shape[2] == 4
    assert dataset.atom_mask.all()


def _goal() -> GoalSpec:
    return GoalSpec(
        name="ligand_contact_asp42",
        type="contact",
        selections=("ligand", "ASP42"),
        operator="less_than",
        threshold=0.45,
        horizon_iterations=2,
    )
