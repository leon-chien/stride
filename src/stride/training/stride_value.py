from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StrideValueLossConfig:
    event_weight: float = 1.0
    flux_weight: float = 1.0
    uncertainty_weight: float = 0.01
    score_weight: float = 0.0


@dataclass(frozen=True)
class StrideValueTargets:
    """
    Delayed descendant training labels for one STRIDE batch.
    """

    event: torch.Tensor
    flux: torch.Tensor
    score: torch.Tensor | None = None


def stride_value_loss(
    outputs: dict[str, torch.Tensor],
    targets: StrideValueTargets,
    config: StrideValueLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Multi-head objective for goal-conditioned trajectory value learning.

    The event and flux targets are delayed descendant labels: whether any
    descendant reached the conditioned target, and how much probability flux
    descendants carried into that target.
    """
    config = config or StrideValueLossConfig()

    event_pred = _require_output(outputs, "p_event")
    flux_pred = _require_output(outputs, "flux_value")
    uncertainty = _require_output(outputs, "uncertainty")

    event_target = targets.event.to(dtype=event_pred.dtype, device=event_pred.device)
    flux_target = targets.flux.to(dtype=flux_pred.dtype, device=flux_pred.device)

    event_loss = nn.functional.binary_cross_entropy(
        event_pred.clamp(min=1e-6, max=1.0 - 1e-6),
        event_target,
    )
    flux_loss = nn.functional.mse_loss(flux_pred, flux_target)
    uncertainty_loss = uncertainty.mean()

    total = (
        config.event_weight * event_loss
        + config.flux_weight * flux_loss
        + config.uncertainty_weight * uncertainty_loss
    )

    score_loss = torch.zeros((), dtype=total.dtype, device=total.device)
    if config.score_weight > 0.0 and targets.score is not None:
        score_pred = _require_output(outputs, "stride_score")
        score_target = targets.score.to(dtype=score_pred.dtype, device=score_pred.device)
        score_loss = nn.functional.mse_loss(score_pred, score_target)
        total = total + config.score_weight * score_loss

    metrics = {
        "loss": float(total.detach().cpu()),
        "event_loss": float(event_loss.detach().cpu()),
        "flux_loss": float(flux_loss.detach().cpu()),
        "uncertainty_loss": float(uncertainty_loss.detach().cpu()),
        "score_loss": float(score_loss.detach().cpu()),
    }

    return total, metrics


def _require_output(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in outputs:
        raise KeyError(f"Model output missing required key: {key}")
    value = outputs[key]
    if value.ndim != 1:
        raise ValueError(f"Expected output '{key}' shape [batch], got {value.shape}")
    return value
