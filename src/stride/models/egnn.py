from __future__ import annotations

import torch
from torch import nn


class EGNNLayer(nn.Module):
    """
    Minimal dense eGNN message-passing layer.

    The node update depends on invariant pair distances. The coordinate update
    is equivariant because it is a learned scalar multiplied by relative
    coordinate vectors.
    """

    def __init__(
        self,
        hidden_dim: int,
        message_dim: int | None = None,
        coord_update_scale: float = 0.1,
    ) -> None:
        super().__init__()
        message_dim = message_dim or hidden_dim
        self.coord_update_scale = coord_update_scale

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, message_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        radius: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 3:
            raise ValueError(f"Expected h shape [batch, atoms, hidden], got {h.shape}")
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected x shape [batch, atoms, 3], got {x.shape}")

        batch_size, num_atoms, _ = h.shape

        h_i = h[:, :, None, :].expand(batch_size, num_atoms, num_atoms, -1)
        h_j = h[:, None, :, :].expand(batch_size, num_atoms, num_atoms, -1)

        rel = x[:, :, None, :] - x[:, None, :, :]
        dist2 = (rel * rel).sum(dim=-1, keepdim=True)

        messages = self.message_mlp(torch.cat([h_i, h_j, dist2], dim=-1))

        edge_mask = torch.ones(
            batch_size,
            num_atoms,
            num_atoms,
            1,
            dtype=h.dtype,
            device=h.device,
        )
        eye = torch.eye(num_atoms, dtype=h.dtype, device=h.device)[None, :, :, None]
        edge_mask = edge_mask * (1.0 - eye)

        if node_mask is not None:
            node_mask = node_mask.to(dtype=h.dtype, device=h.device)
            pair_mask = node_mask[:, :, None] * node_mask[:, None, :]
            edge_mask = edge_mask * pair_mask[..., None]

        if radius is not None:
            edge_mask = edge_mask * (dist2 <= float(radius) ** 2).to(dtype=h.dtype)

        messages = messages * edge_mask
        message_sum = messages.sum(dim=2)

        h_next = h + self.node_mlp(torch.cat([h, message_sum], dim=-1))

        coord_weight = self.coord_mlp(messages) * edge_mask
        denom = edge_mask.sum(dim=2).clamp(min=1.0)
        coord_update = (coord_weight * rel).sum(dim=2) / denom
        x_next = x + self.coord_update_scale * coord_update

        if node_mask is not None:
            mask = node_mask[..., None]
            h_next = h_next * mask
            x_next = x_next * mask + x * (1.0 - mask)

        return h_next, x_next


class EGNNFrameEncoder(nn.Module):
    """
    Encode one molecular frame into a fixed-size invariant embedding.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        radius: float | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.radius = radius
        self.input_proj = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim=hidden_dim) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        coordinates: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError(
                f"Expected coordinates shape [batch, atoms, 3], got {coordinates.shape}"
            )
        if atom_features.ndim != 3:
            raise ValueError(
                "Expected atom_features shape [batch, atoms, features], "
                f"got {atom_features.shape}"
            )
        if coordinates.shape[:2] != atom_features.shape[:2]:
            raise ValueError("coordinates and atom_features must share batch/atom shape.")

        h = self.input_proj(atom_features)
        x = coordinates

        if atom_mask is not None:
            atom_mask = atom_mask.to(dtype=h.dtype, device=h.device)
            h = h * atom_mask[..., None]

        for layer in self.layers:
            h, x = layer(h, x, node_mask=atom_mask, radius=self.radius)

        if atom_mask is None:
            pooled = h.mean(dim=1)
        else:
            denom = atom_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (h * atom_mask[..., None]).sum(dim=1) / denom

        return self.output_norm(pooled)
