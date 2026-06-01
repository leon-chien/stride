from pathlib import Path

import pandas as pd

from stride.data.preprocess import PreprocessConfig, preprocess_mdcath
from stride.data.validation import StageAValidationConfig, validate_stage_a
from test_mdcath_preprocess import _write_synthetic_mdcath


def test_validate_stage_a_checks_roundtrip_hash_and_benchmark(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    input_root.mkdir()
    domain = "12asA00"
    _write_synthetic_mdcath(input_root / f"mdcath_dataset_{domain}.h5", domain)
    domains_path = tmp_path / "domains.txt"
    domains_path.write_text(f"{domain}\n", encoding="utf-8")
    preprocess_mdcath(
        PreprocessConfig(
            input_root=input_root,
            output_root=output_root,
            domains_path=domains_path,
        )
    )

    result = validate_stage_a(
        StageAValidationConfig(
            data_root=output_root,
            input_root=input_root,
            domains_path=domains_path,
            benchmark_windows=3,
            window_frames=2,
        )
    )

    assert result["passed"] is True
    assert result["n_domains"] == 1
    assert result["n_trajectories"] == 2
    assert result["manifest_determinism"]["passed"] is True
    assert result["roundtrip_rmsd"]["passed"] is True
    assert result["roundtrip_rmsd"]["rmsd_angstrom"] < 0.05
    assert result["random_window_benchmark"]["passed"] is True
    assert result["random_window_benchmark"]["n_windows"] == 3
    assert result["shapes"]["coords.ca"] == (2, 256, 512, 3)


def test_validate_stage_a_detects_manifest_hash_mismatch(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    input_root.mkdir()
    domain = "12asA00"
    _write_synthetic_mdcath(input_root / f"mdcath_dataset_{domain}.h5", domain)
    domains_path = tmp_path / "domains.txt"
    domains_path.write_text(f"{domain}\n", encoding="utf-8")
    preprocess_mdcath(
        PreprocessConfig(
            input_root=input_root,
            output_root=output_root,
            domains_path=domains_path,
        )
    )
    metadata_path = output_root / "metadata.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata.loc[0, "sha256_chunk0"] = "0" * 64
    metadata.to_parquet(metadata_path, index=False)

    result = validate_stage_a(
        StageAValidationConfig(
            data_root=output_root,
            input_root=input_root,
            domains_path=domains_path,
            benchmark_windows=1,
            window_frames=2,
        )
    )

    assert result["passed"] is False
    assert result["manifest_determinism"]["passed"] is False
