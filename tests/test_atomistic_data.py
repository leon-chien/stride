from __future__ import annotations

import numpy as np
import torch

from stride.data import (
    AtomRecord,
    atom_selection_mask,
    build_atomistic_windows,
    compute_atom_features,
    load_atomistic_dataset_npz,
    save_atomistic_dataset_npz,
)
from stride.goals import GoalSpec
from stride.models import StrideModelConfig, StrideValueModel
from stride.training import StrideValueTargets, stride_value_loss


def test_atom_features_and_selection_masks_include_molecular_identity() -> None:
    atoms = _example_atoms()

    features = compute_atom_features(atoms)

    assert features.shape[0] == len(atoms)
    assert features.dtype == np.float32
    assert features.shape[1] > 30

    asp_mask = atom_selection_mask(atoms, "ASP42")
    ligand_mask = atom_selection_mask(atoms, "ligand")
    oxygen_mask = atom_selection_mask(atoms, "element:O")

    assert asp_mask.tolist() == [True, True, False, False]
    assert ligand_mask.tolist() == [False, False, True, True]
    assert oxygen_mask.tolist() == [True, False, True, False]


def test_build_atomistic_windows_and_proxy_labels(tmp_path) -> None:
    atoms = _example_atoms()
    coordinates = _example_coordinates()
    goal = _example_goal()

    dataset = build_atomistic_windows(
        coordinates=coordinates,
        atoms=atoms,
        goal=goal,
        window_size=3,
        horizon=2,
        stride=1,
    )

    assert dataset.coordinates.shape == (3, 3, 4, 3)
    assert dataset.atom_features.shape[0:2] == (3, 4)
    assert dataset.atom_mask.all()
    assert dataset.frame_mask.all()
    assert dataset.event_labels.tolist() == [0.0, 1.0, 1.0]
    assert dataset.flux_labels.tolist() == [0.0, 1.0, 1.0]

    output_path = tmp_path / "atomistic_stride_dataset.npz"
    save_atomistic_dataset_npz(output_path, dataset)
    loaded = load_atomistic_dataset_npz(output_path)

    assert np.allclose(loaded.coordinates, dataset.coordinates)
    assert np.array_equal(loaded.event_labels, dataset.event_labels)


def test_atomistic_dataset_runs_through_stride_value_model() -> None:
    torch.manual_seed(5)

    atoms = _example_atoms()
    dataset = build_atomistic_windows(
        coordinates=_example_coordinates(),
        atoms=atoms,
        goal=_example_goal(),
        window_size=3,
        horizon=2,
    )

    config = StrideModelConfig(
        atom_feature_dim=dataset.atom_features.shape[-1],
        goal_feature_dim=dataset.goal_features.shape[-1],
        hidden_dim=32,
        egnn_layers=2,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )
    model = StrideValueModel(config)
    model.train()

    outputs = model(
        coordinates=torch.tensor(dataset.coordinates, dtype=torch.float32),
        atom_features=torch.tensor(dataset.atom_features, dtype=torch.float32),
        goal_features=torch.tensor(dataset.goal_features, dtype=torch.float32),
        atom_mask=torch.tensor(dataset.atom_mask, dtype=torch.bool),
        frame_mask=torch.tensor(dataset.frame_mask, dtype=torch.bool),
    )
    loss, metrics = stride_value_loss(
        outputs,
        StrideValueTargets(
            event=torch.tensor(dataset.event_labels, dtype=torch.float32),
            flux=torch.tensor(dataset.flux_labels, dtype=torch.float32),
        ),
    )

    assert loss.ndim == 0
    assert metrics["loss"] > 0.0
    for value in outputs.values():
        assert value.shape == (len(dataset.event_labels),)
        assert torch.isfinite(value).all()


def _example_atoms() -> list[AtomRecord]:
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


def _example_coordinates() -> np.ndarray:
    coordinates = np.zeros((7, 4, 3), dtype=np.float32)
    coordinates[:, 0, :] = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    coordinates[:, 1, :] = np.asarray([0.2, 0.0, 0.0], dtype=np.float32)
    coordinates[:, 3, :] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    ligand_distances = [1.4, 1.2, 1.0, 0.9, 0.7, 0.35, 0.25]
    for i, distance in enumerate(ligand_distances):
        coordinates[i, 2, :] = np.asarray([distance, 0.0, 0.0], dtype=np.float32)

    return coordinates


def _example_goal() -> GoalSpec:
    return GoalSpec(
        name="ligand_contact_asp42",
        type="contact",
        selections=("ligand", "ASP42"),
        operator="less_than",
        threshold=0.45,
        horizon_iterations=2,
    )
