from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from stride.models.egnn import EGNNFrameEncoder


class FrameEncoder(Protocol):
    """
    Common interface for from-scratch and pretrained atomistic frame encoders.
    """

    output_dim: int

    def __call__(
        self,
        coordinates: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class FrameEncoderConfig:
    encoder_type: str = "egnn"
    input_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 2
    freeze: bool = True


class FrozenPretrainedFrameEncoder(nn.Module):
    """
    Adapter for optional pretrained geometric encoders.

    The wrapped module must return one frame embedding per batch item. This keeps
    heavyweight dependencies outside the core STRIDE install and makes the
    interface easy to mock in tests.
    """

    def __init__(self, module: nn.Module, output_dim: int, freeze: bool = True) -> None:
        super().__init__()
        self.module = module
        self.output_dim = int(output_dim)
        if freeze:
            for parameter in self.module.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        coordinates: torch.Tensor,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.module(coordinates, atom_features, atom_mask)
        if output.ndim != 2:
            raise ValueError("Pretrained frame encoder must return [batch, hidden_dim].")
        if output.shape[-1] != self.output_dim:
            raise ValueError(
                f"Expected pretrained output dim {self.output_dim}, got {output.shape[-1]}."
            )
        return output


def build_frame_encoder(config: FrameEncoderConfig) -> nn.Module:
    if config.encoder_type == "egnn":
        return EGNNFrameEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
    raise ValueError(
        "Only the built-in 'egnn' encoder can be constructed without providing "
        "an optional pretrained module adapter."
    )
