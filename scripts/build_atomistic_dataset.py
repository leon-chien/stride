from __future__ import annotations

import argparse
from pathlib import Path

from stride.data import build_atomistic_dataset_from_pdb, save_atomistic_dataset_npz
from stride.goals import GoalSpec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a STRIDE atomistic .npz dataset from a multi-model PDB."
    )
    parser.add_argument("pdb_path", type=Path, help="Single-model or multi-model PDB.")
    parser.add_argument("goal_yaml", type=Path, help="Structured STRIDE goal YAML.")
    parser.add_argument("output_npz", type=Path, help="Output .npz dataset path.")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--atom-selection",
        default=None,
        help="Optional atom subset, for example 'protein' or 'protein,ligand'.",
    )
    args = parser.parse_args()

    atom_selection = None
    if args.atom_selection:
        atom_selection = tuple(item.strip() for item in args.atom_selection.split(","))

    goal = GoalSpec.from_yaml(args.goal_yaml)
    dataset = build_atomistic_dataset_from_pdb(
        pdb_path=args.pdb_path,
        goal=goal,
        window_size=args.window_size,
        horizon=args.horizon,
        stride=args.stride,
        atom_selection=atom_selection,
    )
    save_atomistic_dataset_npz(args.output_npz, dataset)

    print(f"Input PDB: {args.pdb_path}")
    print(f"Output dataset: {args.output_npz}")
    print(f"Examples: {dataset.coordinates.shape[0]}")
    print(f"Window size: {dataset.coordinates.shape[1]}")
    print(f"Atoms: {dataset.coordinates.shape[2]}")
    print(f"Event positive rate: {dataset.event_labels.mean():.4f}")


if __name__ == "__main__":
    main()
