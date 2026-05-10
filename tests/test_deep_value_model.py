from __future__ import annotations

import numpy as np
import torch

from stride.goals import GoalSpec
from stride.binning import combine_value_heads, scores_to_quantile_bins
from stride.models import EGNNFrameEncoder, StrideModelConfig, StrideValueModel
from stride.training import StrideValueTargets, stride_value_loss


def test_goal_spec_feature_vector_is_deterministic() -> None:
    spec = GoalSpec.from_dict(
        {
            "goal": {
                "name": "ligand_contact_asp42",
                "type": "contact",
                "selection_a": "ligand",
                "selection_b": "ASP42",
                "operator": "less_than",
                "threshold": 0.45,
                "horizon_iterations": 50,
                "value_target": "event_and_flux",
            }
        }
    )

    features = spec.to_feature_vector()

    assert features.dtype == np.float32
    assert features.shape == (spec.feature_dim,)
    assert spec.selections == ("ligand", "ASP42")
    assert np.array_equal(features, spec.to_feature_vector())


def test_egnn_frame_embedding_is_translation_and_rotation_invariant() -> None:
    torch.manual_seed(7)

    encoder = EGNNFrameEncoder(
        atom_feature_dim=4,
        hidden_dim=32,
        num_layers=2,
    )
    encoder.eval()

    coordinates = torch.randn(2, 5, 3)
    atom_features = torch.randn(2, 5, 4)
    atom_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    translation = torch.tensor([3.0, -2.0, 0.5])
    angle = torch.tensor(0.7)
    c = torch.cos(angle)
    s = torch.sin(angle)
    rotation = torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = coordinates @ rotation.T + translation

    with torch.no_grad():
        original_embedding = encoder(coordinates, atom_features, atom_mask)
        transformed_embedding = encoder(transformed, atom_features, atom_mask)

    assert torch.allclose(original_embedding, transformed_embedding, atol=1e-5)


def test_stride_value_model_outputs_westpa_scoring_heads() -> None:
    torch.manual_seed(11)

    spec = GoalSpec(
        name="ligand_binding",
        type="contact",
        selections=("ligand", "ASP42"),
        operator="less_than",
        threshold=0.4,
        horizon_iterations=25,
    )

    config = StrideModelConfig(
        atom_feature_dim=6,
        goal_feature_dim=spec.feature_dim,
        hidden_dim=32,
        egnn_layers=2,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )
    model = StrideValueModel(config)
    model.eval()

    batch_size = 3
    window = 4
    atoms = 5

    coordinates = torch.randn(batch_size, window, atoms, 3)
    atom_features = torch.randn(batch_size, atoms, 6)
    goal_features = torch.tensor(
        np.stack([spec.to_feature_vector() for _ in range(batch_size)]),
        dtype=torch.float32,
    )
    atom_mask = torch.ones(batch_size, atoms, dtype=torch.bool)
    frame_mask = torch.ones(batch_size, window, dtype=torch.bool)

    with torch.no_grad():
        outputs = model(
            coordinates=coordinates,
            atom_features=atom_features,
            goal_features=goal_features,
            atom_mask=atom_mask,
            frame_mask=frame_mask,
        )

    assert set(outputs) == {"p_event", "flux_value", "uncertainty", "stride_score"}

    for value in outputs.values():
        assert value.shape == (batch_size,)
        assert torch.isfinite(value).all()

    assert torch.all(outputs["p_event"] >= 0.0)
    assert torch.all(outputs["p_event"] <= 1.0)
    assert torch.all(outputs["flux_value"] >= 0.0)
    assert torch.all(outputs["uncertainty"] >= 0.0)


def test_stride_value_loss_and_quantile_binning_interfaces() -> None:
    outputs = {
        "p_event": torch.tensor([0.1, 0.8, 0.6]),
        "flux_value": torch.tensor([0.0, 0.3, 0.2]),
        "uncertainty": torch.tensor([0.5, 0.1, 0.2]),
        "stride_score": torch.tensor([0.15, 1.11, 0.82]),
    }
    targets = StrideValueTargets(
        event=torch.tensor([0.0, 1.0, 1.0]),
        flux=torch.tensor([0.0, 0.25, 0.3]),
    )

    loss, metrics = stride_value_loss(outputs, targets)

    assert loss.ndim == 0
    assert metrics["loss"] > 0.0

    scores = combine_value_heads(
        outputs["p_event"].numpy(),
        outputs["flux_value"].numpy(),
        outputs["uncertainty"].numpy(),
    )
    bin_ids, edges = scores_to_quantile_bins(scores, num_bins=3)

    assert bin_ids.shape == (3,)
    assert edges.ndim == 1
    assert bin_ids.min() >= 0
    assert bin_ids.max() < 3
