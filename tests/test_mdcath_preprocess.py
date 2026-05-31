from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr

from stride.data.preprocess import PreprocessConfig, preprocess_mdcath


def test_preprocess_synthetic_mdcath_writes_training_artifacts(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    input_root.mkdir()
    domain = "12asA00"
    _write_synthetic_mdcath(input_root / f"mdcath_dataset_{domain}.h5", domain)
    domains_path = tmp_path / "domains.txt"
    domains_path.write_text(f"{domain}\n", encoding="utf-8")

    manifest = preprocess_mdcath(
        PreprocessConfig(
            input_root=input_root,
            output_root=output_root,
            domains_path=domains_path,
            force=True,
        )
    )

    assert manifest["n_trajectories"] == 2
    assert (output_root / "manifest.json").exists()
    assert (output_root / "metadata.parquet").exists()
    assert (output_root / "splits" / "by_topology.json").exists()
    coords = zarr.open_group(output_root / "coords.zarr", mode="r")
    features = zarr.open_group(output_root / "features.zarr", mode="r")
    mask = zarr.open_group(output_root / "residue_mask.zarr", mode="r")
    assert coords["ca"].shape == (2, 256, 512, 3)
    assert coords["cb"].shape == (2, 256, 512, 3)
    np.testing.assert_allclose(coords["ca"][0, :3, :3], _expected_ca(), atol=2e-3)
    assert features["backbone_torsions"].shape == (2, 256, 512, 6)
    assert features["dssp"][0, 0, :3].tolist() == [1, 3, 0]
    assert features["residue_type"][0, :3].tolist() == [1, 8, 16]
    assert mask["mask"][0, :3].tolist() == [True, True, True]
    metadata = pd.read_parquet(output_root / "metadata.parquet")
    assert metadata["frame_dt_ps"].tolist() == [1000.0, 1000.0]


def test_preprocess_manifest_hash_is_deterministic(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    domain = "12asA00"
    _write_synthetic_mdcath(input_root / f"mdcath_dataset_{domain}.h5", domain)
    domains_path = tmp_path / "domains.txt"
    domains_path.write_text(f"{domain}\n", encoding="utf-8")

    first = preprocess_mdcath(
        PreprocessConfig(
            input_root=input_root,
            output_root=tmp_path / "out1",
            domains_path=domains_path,
        )
    )
    second = preprocess_mdcath(
        PreprocessConfig(
            input_root=input_root,
            output_root=tmp_path / "out2",
            domains_path=domains_path,
        )
    )

    assert first["dataset_hash"] == second["dataset_hash"]


def test_preprocess_dry_run_reports_planned_work(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    domain = "12asA00"
    _write_synthetic_mdcath(input_root / f"mdcath_dataset_{domain}.h5", domain)
    domains_path = tmp_path / "domains.txt"
    domains_path.write_text(f"{domain}\n", encoding="utf-8")

    summary = preprocess_mdcath(
        PreprocessConfig(
            input_root=input_root,
            output_root=tmp_path / "processed",
            domains_path=domains_path,
            dry_run=True,
        )
    )

    assert summary["n_domains"] == 1
    assert summary["n_trajectories"] == 2
    assert summary["max_residues"] == 3


def _write_synthetic_mdcath(path: Path, domain: str) -> None:
    coords = _coords()
    with h5py.File(path, "w") as h5:
        group = h5.create_group(domain)
        group.attrs["numChains"] = 1
        group.attrs["numProteinAtoms"] = 12
        group.attrs["numResidues"] = 3
        group.create_dataset("pdbProteinAtoms", data=np.bytes_(_pdb()))
        group.create_dataset("pdb", data=np.bytes_(_pdb()))
        group.create_dataset("psf", data=np.bytes_("PSF"))
        group.create_dataset("resid", data=np.asarray([1] * 4 + [2] * 3 + [3] * 5))
        group.create_dataset(
            "resname",
            data=np.asarray([b"ALA"] * 4 + [b"GLY"] * 3 + [b"SER"] * 5, dtype="S3"),
        )
        group.create_dataset("chain", data=np.asarray([b"A"] * 12, dtype="S1"))
        group.create_dataset(
            "element",
            data=np.asarray(
                [b"N", b"C", b"C", b"C", b"N", b"C", b"C", b"N", b"C", b"C", b"C", b"O"]
            ),
        )
        for temperature in ["320", "450"]:
            temp_group = group.create_group(temperature)
            rep_group = temp_group.create_group("0")
            rep_group.attrs["numFrames"] = coords.shape[0]
            coord_dataset = rep_group.create_dataset("coords", data=coords)
            coord_dataset.attrs["unit"] = "Angstrom"
            rep_group.create_dataset("box", data=np.eye(3, dtype=np.float32))
            dssp = rep_group.create_dataset(
                "dssp",
                data=np.asarray([[b"H", b"E", b"C"], [b"H", b"E", b"C"], [b"H", b"E", b"C"]]),
            )
            dssp.attrs["unit"] = "none"
            rmsd = rep_group.create_dataset(
                "rmsd", data=np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
            )
            rmsd.attrs["unit"] = "nm"


def _pdb() -> str:
    lines = [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  ",
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C  ",
        "ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00  0.00           C  ",
        "ATOM      4  CB  ALA A   1       1.000  -1.000   0.000  1.00  0.00           C  ",
        "ATOM      5  N   GLY A   2       2.000   1.000   0.000  1.00  0.00           N  ",
        "ATOM      6  CA  GLY A   2       2.000   2.000   0.000  1.00  0.00           C  ",
        "ATOM      7  C   GLY A   2       3.000   2.000   0.000  1.00  0.00           C  ",
        "ATOM      8  N   SER A   3       3.000   3.000   0.000  1.00  0.00           N  ",
        "ATOM      9  CA  SER A   3       4.000   3.000   0.000  1.00  0.00           C  ",
        "ATOM     10  C   SER A   3       4.000   4.000   0.000  1.00  0.00           C  ",
        "ATOM     11  CB  SER A   3       4.000   2.000   0.000  1.00  0.00           C  ",
        "ATOM     12  O   SER A   3       5.000   4.000   0.000  1.00  0.00           O  ",
    ]
    return "\n".join(lines)


def _coords() -> np.ndarray:
    base = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, -1, 0],
            [2, 1, 0],
            [2, 2, 0],
            [3, 2, 0],
            [3, 3, 0],
            [4, 3, 0],
            [4, 4, 0],
            [4, 2, 0],
            [5, 4, 0],
        ],
        dtype=np.float32,
    )
    return np.stack([base, base + 0.1, base + 0.2])


def _expected_ca() -> np.ndarray:
    return np.asarray(
        [
            [[1, 0, 0], [2, 2, 0], [4, 3, 0]],
            [[1.1, 0.1, 0.1], [2.1, 2.1, 0.1], [4.1, 3.1, 0.1]],
            [[1.2, 0.2, 0.2], [2.2, 2.2, 0.2], [4.2, 3.2, 0.2]],
        ],
        dtype=np.float32,
    )
