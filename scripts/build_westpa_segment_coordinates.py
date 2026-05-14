from __future__ import annotations

import argparse
from pathlib import Path

from stride.westpa_plugin.h5_reader import load_segment_records
from stride.westpa_plugin.segment_coordinates import (
    build_segment_coordinate_store,
    save_segment_coordinate_store_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a STRIDE segment-coordinate .npz from WESTPA segment "
            "trajectory files."
        )
    )
    parser.add_argument("west_h5", type=Path, help="Path to WESTPA west.h5.")
    parser.add_argument("topology", type=Path, help="Topology file for MDAnalysis.")
    parser.add_argument(
        "trajectory_root",
        type=Path,
        help="Root directory containing per-segment trajectory files.",
    )
    parser.add_argument("output_npz", type=Path, help="Output segment-coordinate .npz.")
    parser.add_argument(
        "--trajectory-pattern",
        default="{n_iter:06d}/{seg_id:06d}/seg.xtc",
        help=(
            "Format string relative to trajectory_root. Available fields: "
            "n_iter, seg_id."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Frame from each segment trajectory to store.",
    )
    parser.add_argument(
        "--mda-selection",
        default=None,
        help="Optional MDAnalysis atom selection.",
    )
    parser.add_argument(
        "--coordinate-units",
        choices=("nm", "angstrom"),
        default="nm",
        help="Store STRIDE coordinates in nm by default.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing segment trajectory files instead of failing.",
    )
    parser.add_argument(
        "--atom-count-warning",
        type=int,
        default=512,
        help=(
            "Warn when the selected atom count exceeds this value. The current "
            "eGNN path is intended for selected atoms, not full-solvent systems."
        ),
    )
    args = parser.parse_args()

    records = load_segment_records(args.west_h5)
    store, report = build_segment_coordinate_store(
        records=records,
        topology_path=args.topology,
        trajectory_root=args.trajectory_root,
        trajectory_pattern=args.trajectory_pattern,
        frame_index=args.frame_index,
        mda_selection=args.mda_selection,
        coordinate_units=args.coordinate_units,
        require_all=not args.allow_missing,
    )
    save_segment_coordinate_store_npz(args.output_npz, store)

    print(f"WESTPA segments: {report.total_segments}")
    print(f"Saved coordinate frames: {report.saved_segments}")
    print(f"Missing coordinate frames: {report.missing_segments}")
    if report.missing_examples:
        examples = ", ".join(
            f"({key.n_iter}, {key.seg_id})" for key in report.missing_examples
        )
        print(f"Missing examples: {examples}")
    print(f"Atoms: {store.coordinates.shape[1]}")
    if store.coordinates.shape[1] > args.atom_count_warning:
        selection_hint = (
            " Add --mda-selection for model training."
            if args.mda_selection is None
            else " Consider a narrower --mda-selection before training."
        )
        print(
            "WARNING: selected atom count is high for the current pairwise eGNN "
            f"({store.coordinates.shape[1]} atoms).{selection_hint}"
        )
    print(f"Atom feature dim: {store.atom_features.shape[-1]}")
    print(f"Output: {args.output_npz}")


if __name__ == "__main__":
    main()
