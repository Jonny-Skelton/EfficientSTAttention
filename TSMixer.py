import torch
import torch.nn as nn


class TSMixer(nn.Module):
    """
    TSMixer: An All-MLP Architecture for Time Series Forecasting.
    Chen et al., arXiv:2303.06053

    Input shape:  (batch, history_len, num_entities)
    Output shape: (batch, forecast_len, num_entities)

    NOT inductive to N — the feature MLP has N hardcoded in its weight matrix.
    """

    def __init__(self, history_len, forecast_len, num_entities, num_blocks=2,
                 ff_dim=64, dropout=0.1):
        super().__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.num_entities = num_entities

        self.blocks = nn.ModuleList([
            MixerBlock(history_len, num_entities, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.temporal_proj = nn.Linear(history_len, forecast_len)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.transpose(-1, -2)         # (B, N, H)
        x = self.temporal_proj(x)       # (B, N, F)
        return x.transpose(-1, -2)      # (B, F, N)


class MixerBlock(nn.Module):
    """One TSMixer block: time mixing then feature mixing, each with residual + LayerNorm."""

    def __init__(self, history_len, num_entities, ff_dim=64, dropout=0.1):
        super().__init__()

        # Time mixing: project each entity's history independently
        self.time_fc = nn.Linear(history_len, history_len)
        self.time_norm = nn.LayerNorm(num_entities)

        # Feature mixing: two-layer FFN across entities at each timestep
        self.feat_fc1 = nn.Linear(num_entities, ff_dim)
        self.feat_fc2 = nn.Linear(ff_dim, num_entities)
        self.feat_norm = nn.LayerNorm(num_entities)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, H, N)
        r = x.transpose(-1, -2)         # (B, N, H)
        r = self.act(self.time_fc(r))
        r = self.dropout(r)
        r = r.transpose(-1, -2)         # (B, H, N)
        x = self.time_norm(x + r)

        r = self.act(self.feat_fc1(x))
        r = self.dropout(r)
        r = self.feat_fc2(r)
        r = self.dropout(r)
        x = self.feat_norm(x + r)

        return x
