from __future__ import annotations

import argparse
from pathlib import Path

from stride.westpa_plugin import build_multigoal_lineage_dataset_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a multi-goal WESTPA pcoord-lineage STRIDE dataset."
    )
    parser.add_argument("benchmark_yaml", type=Path, help="Multi-goal benchmark YAML.")
    parser.add_argument("output_npz", type=Path, help="Output multi-goal lineage .npz.")
    args = parser.parse_args()

    report = build_multigoal_lineage_dataset_from_yaml(
        args.benchmark_yaml,
        args.output_npz,
    )
    print(f"Benchmark: {report.benchmark_name}")
    print(f"Cells: {report.num_cells}")
    print(f"Goals: {report.num_goals}")
    print(f"Examples: {report.num_examples}")
    print(f"Positive event rate: {report.positive_rate:.6g}")
    print(f"Total labeled flux: {report.flux_sum:.6g}")
    print(f"Output: {args.output_npz}")


if __name__ == "__main__":
    main()
