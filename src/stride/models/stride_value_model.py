from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from stride.models.egnn import EGNNFrameEncoder


@dataclass(frozen=True)
class StrideModelConfig:
    atom_feature_dim: int
    goal_feature_dim: int
    hidden_dim: int = 128
    egnn_layers: int = 3
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1
    radius: float | None = None


class NumericGoalEncoder(nn.Module):
    """
    Learned embedding for deterministic structured goal vectors.
    """

    def __init__(
        self,
        goal_feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(goal_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, goal_features: torch.Tensor) -> torch.Tensor:
        if goal_features.ndim != 2:
            raise ValueError(
                f"Expected goal_features shape [batch, features], got {goal_features.shape}"
            )
        return self.net(goal_features)


class TemporalTransformer(nn.Module):
    """
    Learn trajectory direction and delayed commitment from frame embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_window: int = 512,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_window + 1, hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError(
                "Expected frame_embeddings shape [batch, window, hidden], "
                f"got {frame_embeddings.shape}"
            )

        batch_size, window, hidden_dim = frame_embeddings.shape
        if window + 1 > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Window length {window} exceeds max supported "
                f"{self.pos_embedding.shape[1] - 1}."
            )

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, frame_embeddings], dim=1)
        x = x + self.pos_embedding[:, : window + 1, :]

        key_padding_mask = None
        if frame_mask is not None:
            if frame_mask.shape != (batch_size, window):
                raise ValueError(
                    f"Expected frame_mask shape {(batch_size, window)}, got {frame_mask.shape}"
                )
            cls_mask = torch.ones(
                batch_size,
                1,
                dtype=torch.bool,
                device=frame_mask.device,
            )
            valid_mask = torch.cat([cls_mask, frame_mask.to(torch.bool)], dim=1)
            key_padding_mask = ~valid_mask

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(encoded[:, 0, :])


class StrideValueModel(nn.Module):
    """
    Goal-conditioned eGNN + temporal Transformer model for STRIDE.
    """

    def __init__(self, config: StrideModelConfig) -> None:
        super().__init__()
        self.config = config

        self.frame_encoder = EGNNFrameEncoder(
            atom_feature_dim=config.atom_feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.egnn_layers,
            radius=config.radius,
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
        coordinates: torch.Tensor,
        atom_features: torch.Tensor,
        goal_features: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
            raise ValueError(
                "Expected coordinates shape [batch, window, atoms, 3], "
                f"got {coordinates.shape}"
            )

        batch_size, window, num_atoms, _ = coordinates.shape

        if atom_features.ndim == 3:
            atom_features = atom_features[:, None, :, :].expand(-1, window, -1, -1)
        elif atom_features.ndim != 4:
            raise ValueError(
                "Expected atom_features shape [batch, atoms, features] or "
                f"[batch, window, atoms, features], got {atom_features.shape}"
            )

        if atom_features.shape[:3] != (batch_size, window, num_atoms):
            raise ValueError("atom_features must match coordinates batch/window/atom shape.")

        flat_coordinates = coordinates.reshape(batch_size * window, num_atoms, 3)
        flat_atom_features = atom_features.reshape(
            batch_size * window,
            num_atoms,
            atom_features.shape[-1],
        )

        flat_atom_mask = None
        if atom_mask is not None:
            if atom_mask.shape != (batch_size, num_atoms):
                raise ValueError(
                    f"Expected atom_mask shape {(batch_size, num_atoms)}, got {atom_mask.shape}"
                )
            flat_atom_mask = atom_mask[:, None, :].expand(-1, window, -1)
            flat_atom_mask = flat_atom_mask.reshape(batch_size * window, num_atoms)

        frame_embeddings = self.frame_encoder(
            flat_coordinates,
            flat_atom_features,
            atom_mask=flat_atom_mask,
        ).reshape(batch_size, window, self.config.hidden_dim)

        trajectory_embedding = self.temporal_encoder(
            frame_embeddings,
            frame_mask=frame_mask,
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
