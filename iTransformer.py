import torch.nn as nn


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.
    Liu et al., ICLR 2024 Spotlight

    Each entity's full time series becomes one token; standard self-attention
    then captures inter-entity correlations. Complexity is O(N²).

    Input shape:  (batch, history_len, num_entities)
    Output shape: (batch, forecast_len, num_entities)

    Inductive to N — no N-specific parameters.
    """

    def __init__(self, history_len, forecast_len, num_entities, d_model=64,
                 n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        # num_entities not used in weights — iTransformer is inductive to N
        self.num_entities = num_entities

        self.embed = nn.Linear(history_len, d_model)
        self.blocks = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        x = x.transpose(-1, -2)         # (B, N, H)
        x = self.embed(x)               # (B, N, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)            # (B, N, F)
        return x.transpose(-1, -2)      # (B, F, N)


class iTransformerBlock(nn.Module):
    """Pre-norm transformer block with standard N×N self-attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
