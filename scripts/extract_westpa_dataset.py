from __future__ import annotations

import argparse
from pathlib import Path

from stride.goals import GoalSpec
from stride.westpa_plugin.h5_reader import (
    build_lineage_windows,
    load_segment_records,
    save_lineage_windows_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract STRIDE pcoord lineage windows from a WESTPA west.h5 file."
    )
    parser.add_argument("west_h5", type=Path, help="Path to WESTPA west.h5")
    parser.add_argument("goal_yaml", type=Path, help="Path to STRIDE goal YAML")
    parser.add_argument("output_npz", type=Path, help="Output .npz training artifact")
    parser.add_argument("--window-iterations", type=int, default=8)
    parser.add_argument("--horizon-iterations", type=int, default=None)
    parser.add_argument("--pcoord-dim", type=int, default=0)
    parser.add_argument("--pcoord-frame-index", type=int, default=-1)
    parser.add_argument(
        "--allow-short-windows",
        action="store_true",
        help="Include early-iteration windows shorter than --window-iterations.",
    )

    args = parser.parse_args()

    goal = GoalSpec.from_yaml(args.goal_yaml)
    records = load_segment_records(args.west_h5)
    windows = build_lineage_windows(
        records=records,
        goal=goal,
        window_iterations=args.window_iterations,
        horizon_iterations=args.horizon_iterations,
        pcoord_frame_index=args.pcoord_frame_index,
        pcoord_dim=args.pcoord_dim,
        require_full_window=not args.allow_short_windows,
    )

    save_lineage_windows_npz(args.output_npz, windows)

    positive_rate = sum(window.event for window in windows) / len(windows)
    total_flux = sum(window.flux for window in windows)
    print(f"Loaded segments: {len(records)}")
    print(f"Saved windows: {len(windows)}")
    print(f"Positive event rate: {positive_rate:.4f}")
    print(f"Total labeled flux: {total_flux:.6g}")
    print(f"Output: {args.output_npz}")


if __name__ == "__main__":
    main()
