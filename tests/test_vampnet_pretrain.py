from pathlib import Path

import numpy as np

from stride.data.preprocess import PreprocessConfig, preprocess_mdcath
from stride.labeling.vampnets import VampnetPretrainConfig, pretrain_vampnets
from test_mdcath_preprocess import _write_synthetic_mdcath


def test_pretrain_vampnets_writes_labels_and_manifest(tmp_path: Path) -> None:
    data_root, domains_path = _processed_synthetic(tmp_path)
    out_root = tmp_path / "labels"

    manifest = pretrain_vampnets(
        VampnetPretrainConfig(
            data_root=data_root,
            out_root=out_root,
            domains_path=domains_path,
            resolutions=(2, 3),
            max_frames_per_domain=6,
            force=True,
        )
    )

    assert manifest["dataset_hash"]
    assert manifest["resolutions"] == [2, 3]
    assert manifest["domains"][0]["domain_id"] == "12asA00"
    for resolution in [2, 3]:
        labels_path = out_root / "12asA00" / f"vamp_{resolution}.npy"
        labels = np.load(labels_path)
        assert labels.shape == (6,)
        assert labels.min() >= 0
        assert labels.max() < resolution
        assert str(resolution) in manifest["domains"][0]["resolutions"]
    assert (out_root / "vampnet_manifest.json").exists()


def test_pretrain_vampnets_refuses_existing_manifest_without_force(tmp_path: Path) -> None:
    data_root, domains_path = _processed_synthetic(tmp_path)
    out_root = tmp_path / "labels"
    config = VampnetPretrainConfig(
        data_root=data_root,
        out_root=out_root,
        domains_path=domains_path,
        resolutions=(2,),
        max_frames_per_domain=6,
        force=True,
    )
    pretrain_vampnets(config)

    try:
        pretrain_vampnets(
            VampnetPretrainConfig(
                data_root=data_root,
                out_root=out_root,
                domains_path=domains_path,
                resolutions=(2,),
                max_frames_per_domain=6,
            )
        )
    except FileExistsError:
        return
    raise AssertionError("expected existing manifest to require --force")


def test_pretrain_vampnets_records_tica_fallback_when_health_gate_fails(tmp_path: Path) -> None:
    data_root, domains_path = _processed_synthetic(tmp_path)

    manifest = pretrain_vampnets(
        VampnetPretrainConfig(
            data_root=data_root,
            out_root=tmp_path / "labels",
            domains_path=domains_path,
            resolutions=(2,),
            max_frames_per_domain=6,
            health_margin=1_000.0,
            force=True,
        )
    )

    result = manifest["domains"][0]["resolutions"]["2"]
    assert result["method"] == "tica_kmeans"
    assert result["health_passed"] is False
    assert result["fallback_reason"] == "health_score_not_above_random_projection_baseline"


def _processed_synthetic(tmp_path: Path) -> tuple[Path, Path]:
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
            force=True,
        )
    )
    return output_root, domains_path
