from __future__ import annotations

import argparse
from pathlib import Path

from stride.westpa_plugin.h5_reader import (
    find_candidate_pcoord_paths,
    list_iteration_groups,
    print_h5_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a WESTPA HDF5 file for STRIDE integration."
    )

    parser.add_argument(
        "h5_path",
        type=str,
        help="Path to WESTPA HDF5 file, usually west.h5",
    )

    args = parser.parse_args()

    h5_path = Path(args.h5_path)

    print_h5_tree(h5_path)

    print("\nCandidate pcoord/progress-coordinate datasets:")
    candidates = find_candidate_pcoord_paths(h5_path)

    if not candidates:
        print("  No obvious pcoord/progress datasets found.")
    else:
        for dataset in candidates:
            print(
                f"  {dataset.path} | shape={dataset.shape} | dtype={dataset.dtype}"
            )

    print("\nWESTPA-style iteration groups:")
    iteration_groups = list_iteration_groups(h5_path)

    if not iteration_groups:
        print("  No /iterations/iter_* groups found.")
    else:
        print(f"  Found {len(iteration_groups)} iteration groups.")
        print("  First few:")
        for group in iteration_groups[:5]:
            print(f"    {group}")

        print("  Last few:")
        for group in iteration_groups[-5:]:
            print(f"    {group}")


if __name__ == "__main__":
    main()