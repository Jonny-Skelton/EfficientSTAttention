import torch
import torch.nn as nn
from TSMixer import MixerBlock


class RPMixer(nn.Module):
    """
    RPMixer: Shaking Up Time Series Forecasting with Random Projections.
    Yeh et al., KDD 2024

    Adds a frozen random projection before TSMixer blocks. The projection
    acts as a structured regularizer and diversifies entity representations.

    Input shape:  (batch, history_len, num_entities)
    Output shape: (batch, forecast_len, num_entities)

    NOT inductive to N — the MixerBlock feature MLP has N hardcoded.
    """

    def __init__(self, history_len, forecast_len, num_entities, num_blocks=2,
                 ff_dim=64, dropout=0.1):
        super().__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.num_entities = num_entities

        # Frozen random projection across entities
        self.random_proj = nn.Linear(num_entities, num_entities)
        for param in self.random_proj.parameters():
            param.requires_grad = False

        self.blocks = nn.ModuleList([
            MixerBlock(history_len, num_entities, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.temporal_proj = nn.Linear(history_len, forecast_len)

    def forward(self, x):
        x = self.random_proj(x)          # (B, H, N) — applied to last dim

        for block in self.blocks:
            x = block(x)

        x = x.transpose(-1, -2)          # (B, N, H)
        x = self.temporal_proj(x)        # (B, N, F)
        return x.transpose(-1, -2)       # (B, F, N)
