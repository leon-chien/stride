from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ALANINE_FILES = (
    "alanine-dipeptide-nowater.pdb",
    "alanine-dipeptide-0-250ns-nowater.xtc",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download small public MD datasets used by STRIDE examples."
    )
    parser.add_argument(
        "dataset",
        choices=("alanine_dipeptide",),
        help="Dataset alias to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mdshare/alanine_dipeptide"),
    )
    args = parser.parse_args()

    mdshare = _import_mdshare()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = ALANINE_FILES
    copied_paths = []
    for filename in files:
        source = Path(mdshare.fetch(filename))
        destination = args.output_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied_paths.append(destination)

    print(f"Dataset: {args.dataset}")
    for path in copied_paths:
        print(f"File: {path}")


def _import_mdshare():
    try:
        import mdshare
    except ImportError as exc:
        raise ImportError(
            "mdshare is required to download the alanine dipeptide example. "
            "Install project dependencies with `pip install -e .` or "
            "`conda env update -f environment.yml`."
        ) from exc
    return mdshare


if __name__ == "__main__":
    main()
