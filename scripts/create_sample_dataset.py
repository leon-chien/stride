from __future__ import annotations

import argparse
from pathlib import Path

from stride.data import write_sample_ligand_contact_dataset
from stride.goals import GoalSpec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a tiny atomistic STRIDE dataset for local smoke tests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sample_ligand_contact.npz"),
        help="Output .npz dataset path.",
    )
    parser.add_argument(
        "--pdb-output",
        type=Path,
        default=Path("outputs/sample_ligand_contact.pdb"),
        help="Optional multi-model PDB path for converter smoke tests.",
    )
    parser.add_argument(
        "--goal-yaml",
        type=Path,
        default=Path("configs/goals/ligand_contact_asp42.yaml"),
        help="Goal YAML used for label generation.",
    )
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=12)
    args = parser.parse_args()

    goal = GoalSpec.from_yaml(args.goal_yaml)
    dataset = write_sample_ligand_contact_dataset(
        output_npz=args.output,
        pdb_output=args.pdb_output,
        goal=goal,
        window_size=args.window_size,
        horizon=args.horizon,
        stride=args.stride,
        num_frames=args.num_frames,
    )

    print(f"Output dataset: {args.output}")
    print(f"Output PDB: {args.pdb_output}")
    print(f"Examples: {dataset.coordinates.shape[0]}")
    print(f"Window size: {dataset.coordinates.shape[1]}")
    print(f"Atoms: {dataset.coordinates.shape[2]}")
    print(f"Atom feature dim: {dataset.atom_features.shape[-1]}")
    print(f"Goal feature dim: {dataset.goal_features.shape[-1]}")
    print(f"Event positive rate: {dataset.event_labels.mean():.4f}")


if __name__ == "__main__":
    main()
