from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from stride.models import FrozenPretrainedFrameEncoder
from stride.training import PcoordLineageDataset
from stride.westpa_plugin import (
    OnlineLineageStore,
    PcoordRuntimeScoringInput,
    StrideWestpaController,
    decide_promotion,
)


def test_controller_loads_control_config_and_falls_back_to_pcoord(tmp_path) -> None:
    config_path = _write_control_config(tmp_path / "stride_control_config.json")
    controller = StrideWestpaController.from_json(config_path, checkpoint_path=None)
    active = PcoordRuntimeScoringInput(
        pcoord_windows=np.ones((4, 3, 2), dtype=np.float32),
        window_mask=np.ones((4, 3), dtype=bool),
        goal_features=np.ones((4, 15), dtype=np.float32),
    )

    result = controller.assign(
        active,
        baseline_scores=np.asarray([-1.0, 0.2, 0.8, 2.0], dtype=np.float32),
    )

    assert result.used_fallback
    assert result.bin_ids.tolist() == [0, 1, 2, 3]
    assert result.priority_rank.tolist() == [4, 3, 2, 1]


def test_controller_rejects_bad_baseline_shape(tmp_path) -> None:
    controller = StrideWestpaController.from_json(
        _write_control_config(tmp_path / "stride_control_config.json"),
        checkpoint_path=None,
    )
    active = PcoordRuntimeScoringInput(
        pcoord_windows=np.ones((4, 3, 2), dtype=np.float32),
        window_mask=np.ones((4, 3), dtype=bool),
        goal_features=np.ones((4, 15), dtype=np.float32),
    )

    try:
        controller.assign(active, baseline_scores=np.ones((3,), dtype=np.float32))
    except ValueError as exc:
        assert "baseline_scores" in str(exc)
    else:
        raise AssertionError("Expected baseline shape validation failure.")


def test_online_lineage_store_appends_and_reloads(tmp_path) -> None:
    store = OnlineLineageStore(tmp_path / "online.npz")
    first = _dataset(0, 3)
    second = _dataset(3, 2)

    summary = store.append(first, metadata={"run_id": "test"})
    assert summary.num_examples == 3
    summary = store.append(second)

    loaded = store.load()
    assert summary.num_examples == 5
    assert loaded.pcoord_windows.shape[0] == 5
    assert loaded.n_iter.tolist() == [0, 1, 2, 3, 4]
    assert (tmp_path / "online.npz.metadata.json").exists()


def test_promotion_rejects_weaker_challenger() -> None:
    decision = decide_promotion(
        challenger={
            "auprc": 0.4,
            "top25_enrichment": 1.1,
            "bin_event_rate_gradient": 0.1,
            "occupancy_cv": 0.2,
        },
        baseline={
            "auprc": 0.5,
            "top25_enrichment": 1.0,
            "bin_event_rate_gradient": 0.05,
            "occupancy_cv": 0.2,
        },
    )

    assert not decision.promote
    assert "baseline" in decision.reason


def test_pretrained_frame_encoder_adapter_can_be_mocked() -> None:
    class MockEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 4)

        def forward(self, coordinates, atom_features, atom_mask):
            del atom_features
            masked = coordinates[..., :2] * atom_mask[..., None]
            return self.linear(masked.mean(dim=1))

    adapter = FrozenPretrainedFrameEncoder(MockEncoder(), output_dim=4, freeze=True)
    coordinates = torch.ones(3, 5, 3)
    atom_features = torch.ones(3, 5, 2)
    atom_mask = torch.ones(3, 5, dtype=torch.bool)

    output = adapter(coordinates, atom_features, atom_mask)

    assert output.shape == (3, 4)
    assert all(not parameter.requires_grad for parameter in adapter.module.parameters())


def _write_control_config(path):
    path.write_text(
        json.dumps(
            {
                "score_key": "p_event",
                "baseline_key": "last_pcoord_low",
                "checkpoint_path": None,
                "rankers": {
                    "stride": {"bin_edges": [0.25, 0.5, 0.75]},
                    "last_pcoord_low": {"bin_edges": [0.0, 0.5, 1.0]},
                },
            }
        )
    )
    return path


def _dataset(start_iter: int, count: int) -> PcoordLineageDataset:
    labels = (np.arange(count) % 2).astype(np.float32)
    return PcoordLineageDataset(
        pcoord_windows=np.ones((count, 3, 2), dtype=np.float32),
        window_mask=np.ones((count, 3), dtype=bool),
        goal_features=np.ones((count, 15), dtype=np.float32),
        event_labels=labels,
        flux_labels=labels * 0.1,
        n_iter=np.arange(start_iter, start_iter + count, dtype=np.int64),
        seg_id=np.arange(count, dtype=np.int64),
        weights=np.ones((count,), dtype=np.float64),
        cell_id=np.asarray(["cell_0"] * count),
        goal_id=np.asarray(["goal_0"] * count),
        pcoord_dim=np.zeros((count,), dtype=np.int64),
        threshold=np.ones((count,), dtype=np.float32),
        horizon_iterations=np.ones((count,), dtype=np.int64),
    )
