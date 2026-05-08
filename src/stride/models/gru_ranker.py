from __future__ import annotations

import torch
from torch import nn


class GRURanker(nn.Module):
    """
    GRU-based trajectory-value model.

    Input:
        x: [batch_size, window_size, num_features]

    Output:
        logits: [batch_size]

    The model predicts whether a trajectory window is likely to
    reach the target event within a future horizon.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_layers == 1:
            gru_dropout = 0.0
        else:
            gru_dropout = dropout

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [batch_size, window_size, num_features]

        Returns:
            logits: Tensor with shape [batch_size]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [batch, window, features], got {x.shape}"
            )

        _, hidden = self.gru(x)

        # hidden shape: [num_layers, batch_size, hidden_dim]
        last_hidden = hidden[-1]

        logits = self.head(last_hidden)

        return logits.squeeze(-1)


def predict_probability(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Convert model logits into probabilities.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    return probs


if __name__ == "__main__":
    model = GRURanker(num_features=6)

    dummy_x = torch.randn(32, 25, 6)
    dummy_logits = model(dummy_x)

    print(f"Input shape: {dummy_x.shape}")
    print(f"Output shape: {dummy_logits.shape}")

    assert dummy_logits.shape == (32,)

    print("GRURanker sanity check passed.")