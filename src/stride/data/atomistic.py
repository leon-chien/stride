from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stride.goals import GoalSpec


ELEMENTS = (
    "H",
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "CL",
    "BR",
    "I",
    "NA",
    "MG",
    "ZN",
    "CA",
    "FE",
    "OTHER",
)

AMINO_ACIDS = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "OTHER",
)


@dataclass(frozen=True)
class AtomRecord:
    """
    Topology metadata for one atom.
    """

    atom_name: str
    element: str
    residue_name: str
    residue_id: int
    chain_id: str = ""
    atom_type: str = ""
    charge: float = 0.0
    mass: float = 0.0
    is_ligand: bool = False
    is_water: bool = False
    is_ion: bool = False

    @property
    def is_protein(self) -> bool:
        return (
            self.residue_name.upper() in AMINO_ACIDS[:-1]
            and not self.is_ligand
            and not self.is_water
            and not self.is_ion
        )


@dataclass(frozen=True)
class AtomisticDataset:
    """
    Canonical atomistic STRIDE training artifact.
    """

    coordinates: np.ndarray
    atom_features: np.ndarray
    atom_mask: np.ndarray
    frame_mask: np.ndarray
    goal_features: np.ndarray
    event_labels: np.ndarray
    flux_labels: np.ndarray
    source_frame_start: np.ndarray

    def validate(self) -> None:
        if self.coordinates.ndim != 4 or self.coordinates.shape[-1] != 3:
            raise ValueError(
                "coordinates must have shape [examples, window, atoms, 3]."
            )
        num_examples, window_size, num_atoms, _ = self.coordinates.shape

        if self.atom_features.shape[:2] != (num_examples, num_atoms):
            raise ValueError(
                "atom_features must have shape [examples, atoms, features]."
            )
        if self.atom_mask.shape != (num_examples, num_atoms):
            raise ValueError("atom_mask must have shape [examples, atoms].")
        if self.frame_mask.shape != (num_examples, window_size):
            raise ValueError("frame_mask must have shape [examples, window].")
        if self.goal_features.shape[0] != num_examples:
            raise ValueError("goal_features must have one row per example.")
        if self.event_labels.shape != (num_examples,):
            raise ValueError("event_labels must have shape [examples].")
        if self.flux_labels.shape != (num_examples,):
            raise ValueError("flux_labels must have shape [examples].")
        if self.source_frame_start.shape != (num_examples,):
            raise ValueError("source_frame_start must have shape [examples].")


def compute_atom_features(atoms: list[AtomRecord]) -> np.ndarray:
    """
    Convert topology metadata into model-ready atom identity features.

    These are molecular identity inputs, not hand-picked event features.
    """
    features = []
    max_residue_id = max((abs(atom.residue_id) for atom in atoms), default=1)
    max_residue_id = max(max_residue_id, 1)

    for atom in atoms:
        element = _canonical_element(atom.element)
        residue = atom.residue_name.upper()

        row: list[float] = []
        row.extend(_one_hot(element, ELEMENTS))
        row.extend(_one_hot(residue if residue in AMINO_ACIDS else "OTHER", AMINO_ACIDS))
        row.extend(
            [
                float(atom.is_protein),
                float(atom.is_ligand),
                float(atom.is_water),
                float(atom.is_ion),
                float(atom.residue_id) / float(max_residue_id),
                float(atom.charge),
                float(atom.mass) / 100.0,
            ]
        )
        features.append(row)

    return np.asarray(features, dtype=np.float32)


def atom_selection_mask(
    atoms: list[AtomRecord],
    selection: str | list[str] | tuple[str, ...],
) -> np.ndarray:
    """
    Resolve a small structured selection vocabulary into an atom mask.

    Supported forms include:
        protein, ligand, water, ion
        element:C
        chain:A
        resid:42
        resname:ASP
        ASP42
        atom:CA
    """
    selections = [selection] if isinstance(selection, str) else list(selection)
    mask = np.zeros((len(atoms),), dtype=bool)

    for item in selections:
        item_mask = np.asarray([_atom_matches(atom, item) for atom in atoms], dtype=bool)
        mask = mask | item_mask

    return mask


def build_atomistic_windows(
    coordinates: np.ndarray,
    atoms: list[AtomRecord],
    goal: GoalSpec,
    window_size: int,
    horizon: int | None = None,
    stride: int = 1,
    flux_label_mode: str = "event",
) -> AtomisticDataset:
    """
    Build fixed atomistic trajectory windows with proxy future-event labels.

    Public MD trajectories do not provide WESTPA probability flux. For proxy
    training, flux labels can mirror event labels. True flux labels should come
    from WESTPA descendant weights.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [frames, atoms, 3].")
    if coordinates.shape[1] != len(atoms):
        raise ValueError("coordinates atom count must match atoms metadata.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    horizon = horizon or goal.horizon_iterations
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    atom_features = compute_atom_features(atoms)
    goal_features = goal.to_feature_vector()
    num_frames = coordinates.shape[0]

    windows: list[np.ndarray] = []
    event_labels: list[float] = []
    flux_labels: list[float] = []
    source_frame_start: list[int] = []

    max_start = num_frames - window_size - horizon + 1
    for start in range(0, max(0, max_start), stride):
        end = start + window_size
        future_end = end + horizon
        future = coordinates[end:future_end]
        event = _future_event_label(future, atoms, goal)

        windows.append(coordinates[start:end])
        event_labels.append(float(event))
        if flux_label_mode == "event":
            flux_labels.append(float(event))
        elif flux_label_mode == "zero":
            flux_labels.append(0.0)
        else:
            raise ValueError(f"Unknown flux_label_mode: {flux_label_mode}")
        source_frame_start.append(start)

    if not windows:
        raise ValueError("No windows could be built from the provided trajectory.")

    num_examples = len(windows)
    num_atoms = len(atoms)
    feature_dim = atom_features.shape[-1]

    dataset = AtomisticDataset(
        coordinates=np.stack(windows).astype(np.float32),
        atom_features=np.broadcast_to(
            atom_features[None, :, :],
            (num_examples, num_atoms, feature_dim),
        ).copy(),
        atom_mask=np.ones((num_examples, num_atoms), dtype=bool),
        frame_mask=np.ones((num_examples, window_size), dtype=bool),
        goal_features=np.broadcast_to(
            goal_features[None, :],
            (num_examples, goal_features.shape[0]),
        ).copy(),
        event_labels=np.asarray(event_labels, dtype=np.float32),
        flux_labels=np.asarray(flux_labels, dtype=np.float32),
        source_frame_start=np.asarray(source_frame_start, dtype=np.int64),
    )
    dataset.validate()
    return dataset


def save_atomistic_dataset_npz(path: str | Path, dataset: AtomisticDataset) -> None:
    dataset.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coordinates=dataset.coordinates,
        atom_features=dataset.atom_features,
        atom_mask=dataset.atom_mask,
        frame_mask=dataset.frame_mask,
        goal_features=dataset.goal_features,
        event_labels=dataset.event_labels,
        flux_labels=dataset.flux_labels,
        source_frame_start=dataset.source_frame_start,
    )


def load_atomistic_dataset_npz(path: str | Path) -> AtomisticDataset:
    data = np.load(path)
    dataset = AtomisticDataset(
        coordinates=data["coordinates"].astype(np.float32),
        atom_features=data["atom_features"].astype(np.float32),
        atom_mask=data["atom_mask"].astype(bool),
        frame_mask=data["frame_mask"].astype(bool),
        goal_features=data["goal_features"].astype(np.float32),
        event_labels=data["event_labels"].astype(np.float32),
        flux_labels=data["flux_labels"].astype(np.float32),
        source_frame_start=data["source_frame_start"].astype(np.int64),
    )
    dataset.validate()
    return dataset


def _future_event_label(
    future_coordinates: np.ndarray,
    atoms: list[AtomRecord],
    goal: GoalSpec,
) -> int:
    if goal.type not in {"contact", "distance_threshold"}:
        raise NotImplementedError(
            "Atomistic proxy labels currently support contact and distance_threshold goals."
        )
    if goal.operator != "less_than":
        raise NotImplementedError("Atomistic proxy labels currently support less_than.")
    if len(goal.selections) < 2:
        raise ValueError("Atomistic goals require at least two selections.")

    mask_a = atom_selection_mask(atoms, goal.selections[0])
    mask_b = atom_selection_mask(atoms, goal.selections[1])
    if not mask_a.any():
        raise ValueError(f"Selection matched no atoms: {goal.selections[0]}")
    if not mask_b.any():
        raise ValueError(f"Selection matched no atoms: {goal.selections[1]}")

    for frame in future_coordinates:
        min_distance = _min_pair_distance(frame[mask_a], frame[mask_b])
        if min_distance < goal.threshold:
            return 1
    return 0


def _min_pair_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    deltas = coords_a[:, None, :] - coords_b[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    return float(distances.min())


def _atom_matches(atom: AtomRecord, selection: str) -> bool:
    text = selection.strip()
    lower = text.lower()

    if lower == "protein":
        return atom.is_protein
    if lower == "ligand":
        return atom.is_ligand
    if lower == "water":
        return atom.is_water
    if lower == "ion":
        return atom.is_ion
    if lower.startswith("element:"):
        return _canonical_element(atom.element) == _canonical_element(text.split(":", 1)[1])
    if lower.startswith("chain:"):
        return atom.chain_id == text.split(":", 1)[1]
    if lower.startswith("resid:"):
        return atom.residue_id == int(text.split(":", 1)[1])
    if lower.startswith("resname:"):
        return atom.residue_name.upper() == text.split(":", 1)[1].upper()
    if lower.startswith("atom:"):
        return atom.atom_name.upper() == text.split(":", 1)[1].upper()

    residue_name, residue_id = _parse_residue_token(text)
    if residue_name is not None and residue_id is not None:
        return atom.residue_name.upper() == residue_name and atom.residue_id == residue_id

    return (
        atom.atom_name.upper() == text.upper()
        or atom.residue_name.upper() == text.upper()
        or _canonical_element(atom.element) == _canonical_element(text)
    )


def _parse_residue_token(text: str) -> tuple[str | None, int | None]:
    letters = "".join(char for char in text if char.isalpha())
    digits = "".join(char for char in text if char.isdigit() or char == "-")
    if not letters or not digits:
        return None, None
    return letters.upper(), int(digits)


def _canonical_element(element: str) -> str:
    value = element.strip().upper()
    if value in ELEMENTS:
        return value
    if len(value) > 1 and value[:2] in ELEMENTS:
        return value[:2]
    if value[:1] in ELEMENTS:
        return value[:1]
    return "OTHER"


def _one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]
