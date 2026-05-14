from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from stride.models.stride_value_model import NumericGoalEncoder, TemporalTransformer


@dataclass(frozen=True)
class PcoordLineageModelConfig:
    pcoord_dim: int
    goal_feature_dim: int
    hidden_dim: int = 64
    transformer_layers: int = 1
    transformer_heads: int = 4
    dropout: float = 0.1


class PcoordLineageValueModel(nn.Module):
    """
    Goal-conditioned temporal value model for WESTPA pcoord lineage windows.
    """

    def __init__(self, config: PcoordLineageModelConfig) -> None:
        super().__init__()
        if config.pcoord_dim <= 0:
            raise ValueError("pcoord_dim must be positive.")
        self.config = config

        self.pcoord_encoder = nn.Sequential(
            nn.Linear(config.pcoord_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.temporal_encoder = TemporalTransformer(
            hidden_dim=config.hidden_dim,
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            dropout=config.dropout,
        )
        self.goal_encoder = NumericGoalEncoder(
            goal_feature_dim=config.goal_feature_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        joint_dim = config.hidden_dim * 3
        self.trunk = nn.Sequential(
            nn.Linear(joint_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.event_head = nn.Linear(config.hidden_dim, 1)
        self.flux_head = nn.Linear(config.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        pcoord_windows: torch.Tensor,
        window_mask: torch.Tensor,
        goal_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pcoord_windows.ndim != 3:
            raise ValueError(
                "Expected pcoord_windows shape [batch, window, dims], "
                f"got {pcoord_windows.shape}"
            )
        batch_size, window, pcoord_dim = pcoord_windows.shape
        if pcoord_dim != self.config.pcoord_dim:
            raise ValueError(
                f"Expected pcoord dim {self.config.pcoord_dim}, got {pcoord_dim}."
            )
        if window_mask.shape != (batch_size, window):
            raise ValueError(
                f"Expected window_mask shape {(batch_size, window)}, got {window_mask.shape}"
            )

        frame_embeddings = self.pcoord_encoder(pcoord_windows)
        trajectory_embedding = self.temporal_encoder(
            frame_embeddings,
            frame_mask=window_mask,
        )
        goal_embedding = self.goal_encoder(goal_features)
        joint = torch.cat(
            [
                trajectory_embedding,
                goal_embedding,
                trajectory_embedding * goal_embedding,
            ],
            dim=-1,
        )
        hidden = self.trunk(joint)

        p_event = torch.sigmoid(self.event_head(hidden)).squeeze(-1)
        flux_value = torch.nn.functional.softplus(self.flux_head(hidden)).squeeze(-1)
        uncertainty = torch.nn.functional.softplus(
            self.uncertainty_head(hidden)
        ).squeeze(-1)
        stride_score = p_event + flux_value + 0.1 * uncertainty

        return {
            "p_event": p_event,
            "flux_value": flux_value,
            "uncertainty": uncertainty,
            "stride_score": stride_score,
        }
