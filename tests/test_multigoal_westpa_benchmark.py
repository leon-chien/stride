from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np

from stride.training import load_pcoord_lineage_dataset_npz, split_pcoord_lineage_indices
from stride.training.evaluation import write_evaluation_report
from stride.training.westpa_evaluation import write_westpa_lineage_report
from stride.westpa_plugin import build_multigoal_lineage_dataset_from_yaml


def test_multigoal_builder_creates_one_row_per_lineage_goal_pair(tmp_path) -> None:
    h5_path = tmp_path / "cell0" / "west.h5"
    output_path = tmp_path / "multigoal.npz"
    config_path = tmp_path / "benchmark.yaml"
    _write_tiny_westpa_h5(h5_path)
    _write_benchmark_yaml(config_path, h5_path)

    report = build_multigoal_lineage_dataset_from_yaml(config_path, output_path)
    dataset = load_pcoord_lineage_dataset_npz(output_path)

    assert report.num_cells == 1
    assert report.num_goals == 2
    assert dataset.pcoord_windows.shape[0] == 8
    assert dataset.goal_id is not None
    assert dataset.cell_id is not None
    assert dataset.pcoord_dim is not None
    assert dataset.threshold is not None
    assert dataset.horizon_iterations is not None
    assert set(dataset.goal_id.tolist()) == {"dim0_low", "dim1_low"}
    assert set(dataset.cell_id.tolist()) == {"cell0"}
    assert dataset.pcoord_dim.tolist().count(0) == 4
    assert dataset.pcoord_dim.tolist().count(1) == 4


def test_multigoal_split_holds_out_goal_or_cell_without_leakage(tmp_path) -> None:
    output_path = tmp_path / "multigoal.npz"
    config_path = tmp_path / "benchmark.yaml"
    h5_a = tmp_path / "cell0" / "west.h5"
    h5_b = tmp_path / "cell1" / "west.h5"
    _write_tiny_westpa_h5(h5_a)
    _write_tiny_westpa_h5(h5_b)
    _write_benchmark_yaml(config_path, h5_a, h5_b)
    dataset = load_pcoord_lineage_dataset_npz(
        _build(config_path, output_path)
    )

    train, val = split_pcoord_lineage_indices(
        dataset,
        validation_fraction=0.5,
        split_strategy="heldout_goal",
    )
    assert set(dataset.goal_id[train]).isdisjoint(set(dataset.goal_id[val]))

    train, val = split_pcoord_lineage_indices(
        dataset,
        validation_fraction=0.5,
        split_strategy="heldout_cell",
    )
    assert set(dataset.cell_id[train]).isdisjoint(set(dataset.cell_id[val]))


def test_grouped_report_handles_multigoal_artifact(tmp_path) -> None:
    output_path = tmp_path / "multigoal.npz"
    config_path = tmp_path / "benchmark.yaml"
    h5_path = tmp_path / "cell0" / "west.h5"
    _write_tiny_westpa_h5(h5_path)
    _write_benchmark_yaml(config_path, h5_path)
    _build(config_path, output_path)
    data = np.load(output_path)
    stride_scores = np.linspace(0.0, 1.0, len(data["event_labels"]), dtype=np.float32)

    paths = write_westpa_lineage_report(
        output_path,
        tmp_path / "report",
        stride_scores=stride_scores,
        eval_split="all",
    )

    assert paths["grouped_metrics"].exists()
    assert paths["grouped_markdown"].exists()
    text = paths["grouped_markdown"].read_text()
    assert "goal_id" in text
    assert "cell_id" in text
    assert "dim0_low" in text


def test_seed_summary_computes_mean_and_baseline_delta(tmp_path) -> None:
    report_a = tmp_path / "report_a"
    report_b = tmp_path / "report_b"
    write_evaluation_report(
        report_a,
        y_true=np.asarray([0, 1, 0, 1], dtype=np.float32),
        rankers={
            "STRIDE": np.asarray([0.1, 0.9, 0.2, 0.8]),
            "baseline": np.asarray([0.2, 0.7, 0.3, 0.6]),
        },
    )
    write_evaluation_report(
        report_b,
        y_true=np.asarray([0, 1, 0, 1], dtype=np.float32),
        rankers={
            "STRIDE": np.asarray([0.1, 0.8, 0.2, 0.7]),
            "baseline": np.asarray([0.2, 0.6, 0.3, 0.5]),
        },
    )
    module = _load_summary_script()
    rows = module.summarize_metric_rows(module.load_metric_rows([report_a, report_b]))
    stride = next(row for row in rows if row["ranker"] == "STRIDE")

    assert stride["n"] == 2.0
    assert stride["auroc_delta_vs_best_baseline"] >= 0.0


def _build(config_path: Path, output_path: Path) -> Path:
    build_multigoal_lineage_dataset_from_yaml(config_path, output_path)
    return output_path


def _write_benchmark_yaml(path: Path, h5_path: Path, h5_path_2: Path | None = None) -> None:
    cells = [
        f"    - cell_id: cell0\n      west_h5: {h5_path}\n",
    ]
    if h5_path_2 is not None:
        cells.append(f"    - cell_id: cell1\n      west_h5: {h5_path_2}\n")
    path.write_text(
        "benchmark:\n"
        "  name: tiny_multigoal\n"
        "  window_iterations: 2\n"
        "  pcoord_frame_index: -1\n"
        "  cells:\n"
        + "".join(cells)
        + "  goals:\n"
        "    - goal_id: dim0_low\n"
        "      pcoord_dim: 0\n"
        "      threshold: 0.55\n"
        "      horizon_iterations: 1\n"
        "    - goal_id: dim1_low\n"
        "      pcoord_dim: 1\n"
        "      threshold: 0.55\n"
        "      horizon_iterations: 1\n",
        encoding="utf-8",
    )


def _write_tiny_westpa_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seg_index_dtype = np.dtype(
        [
            ("weight", np.float64),
            ("parent_id", np.int64),
            ("endpoint_type", np.uint8),
        ]
    )
    with h5py.File(path, "w") as h5:
        iterations = h5.create_group("iterations")
        _write_iter(
            iterations,
            n_iter=1,
            parent_ids=[-1],
            weights=[1.0],
            final_pcoords=[[0.8, 0.8]],
            seg_index_dtype=seg_index_dtype,
        )
        _write_iter(
            iterations,
            n_iter=2,
            parent_ids=[0, 0],
            weights=[0.5, 0.5],
            final_pcoords=[[0.4, 0.9], [0.9, 0.4]],
            seg_index_dtype=seg_index_dtype,
        )
        _write_iter(
            iterations,
            n_iter=3,
            parent_ids=[0, 1],
            weights=[0.25, 0.25],
            final_pcoords=[[0.3, 0.9], [0.9, 0.3]],
            seg_index_dtype=seg_index_dtype,
        )


def _write_iter(
    iterations,
    n_iter: int,
    parent_ids: list[int],
    weights: list[float],
    final_pcoords: list[list[float]],
    seg_index_dtype: np.dtype,
) -> None:
    group = iterations.create_group(f"iter_{n_iter:08d}")
    pcoord = np.zeros((len(parent_ids), 2, 2), dtype=np.float32)
    pcoord[:, 0, :] = 1.0
    pcoord[:, 1, :] = np.asarray(final_pcoords, dtype=np.float32)
    group.create_dataset("pcoord", data=pcoord)

    seg_index = np.zeros((len(parent_ids),), dtype=seg_index_dtype)
    seg_index["weight"] = np.asarray(weights, dtype=np.float64)
    seg_index["parent_id"] = np.asarray(parent_ids, dtype=np.int64)
    seg_index["endpoint_type"] = 0
    group.create_dataset("seg_index", data=seg_index)


def _load_summary_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_westpa_reports.py"
    spec = importlib.util.spec_from_file_location("summarize_westpa_reports", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load summarize_westpa_reports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
