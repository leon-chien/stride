from __future__ import annotations

import argparse
from pathlib import Path

from stride.data import (
    build_atomistic_dataset_from_mdanalysis,
    save_atomistic_dataset_npz,
)
from stride.goals import GoalSpec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a STRIDE atomistic .npz dataset from topology + trajectory files."
    )
    parser.add_argument("topology", type=Path, help="Topology file, usually PDB/PSF/PRMTOP.")
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Trajectory file, usually XTC/DCD/NC/TRR.",
    )
    parser.add_argument("goal_yaml", type=Path, help="Structured STRIDE goal YAML.")
    parser.add_argument("output_npz", type=Path, help="Output .npz dataset path.")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--mda-selection",
        default=None,
        help="Optional MDAnalysis selection string, e.g. 'protein and not name H*'.",
    )
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--coordinate-units",
        choices=("nm", "angstrom"),
        default="nm",
        help="Store STRIDE coordinates in nm by default.",
    )
    args = parser.parse_args()

    goal = GoalSpec.from_yaml(args.goal_yaml)
    dataset = build_atomistic_dataset_from_mdanalysis(
        topology_path=args.topology,
        trajectory_path=args.trajectory,
        goal=goal,
        window_size=args.window_size,
        horizon=args.horizon,
        stride=args.stride,
        mda_selection=args.mda_selection,
        start=args.start,
        stop=args.stop,
        step=args.step,
        coordinate_units=args.coordinate_units,
    )
    save_atomistic_dataset_npz(args.output_npz, dataset)

    print(f"Topology: {args.topology}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Output dataset: {args.output_npz}")
    print(f"Examples: {dataset.coordinates.shape[0]}")
    print(f"Window size: {dataset.coordinates.shape[1]}")
    print(f"Atoms: {dataset.coordinates.shape[2]}")
    print(f"Event positive rate: {dataset.event_labels.mean():.4f}")


if __name__ == "__main__":
    main()
