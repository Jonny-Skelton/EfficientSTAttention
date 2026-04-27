import torch
import torch.nn as nn
import torch.nn.functional as F


class EiFormer(nn.Module):
    """
    EiFormer: Towards Efficient Large Scale Spatial-Temporal Time Series Forecasting
    via Improved Inverted Transformers. Sun et al., arXiv:2503.10858

    Replaces N×N self-attention with N×M latent attention (M << N), reducing
    complexity from O(N²) to O(N·M) ≈ O(N). The first block uses a frozen
    random projection for K; subsequent blocks learn K.

    Input shape:  (batch, history_len, num_entities)
    Output shape: (batch, forecast_len, num_entities)

    Inductive to N — no N-specific parameters.
    """

    def __init__(self, history_len, forecast_len, num_entities, d_model=64,
                 n_heads=4, num_latent=16, num_layers=2, dropout=0.1):
        super().__init__()
        # num_entities not used in weights — EiFormer is inductive to N
        self.num_entities = num_entities

        self.embed = nn.Linear(history_len, d_model)
        self.blocks = nn.ModuleList([
            EiFormerBlock(d_model, n_heads, num_latent, dropout, freeze_K=(i == 0))
            for i in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        x = x.transpose(-1, -2)         # (B, N, H)
        x = self.embed(x)               # (B, N, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)            # (B, N, F)
        return x.transpose(-1, -2)      # (B, F, N)


class EiFormerBlock(nn.Module):
    """
    EiFormer block: multi-head latent attention + feedforward.

    Q comes from input; K and V are learnable parameters of shape
    (n_heads, M, d_head), so the attention map is N×M instead of N×N.
    When freeze_K=True, K is a fixed random buffer (random projection).
    """

    def __init__(self, d_model, n_heads, num_latent, dropout=0.1, freeze_K=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model)

        K_init = torch.randn(n_heads, num_latent, self.d_head)
        if freeze_K:
            # Frozen: part of state_dict but excluded from optimizer updates
            self.register_buffer('K', K_init)
        else:
            self.K = nn.Parameter(K_init)

        self.V = nn.Parameter(torch.randn(n_heads, num_latent, self.d_head))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, _ = x.shape

        normed = self.norm1(x)
        Q = self.W_Q(normed)
        Q = Q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)   # (B, h, N, d_head)

        # scores: (B, h, N, M)
        scores = torch.matmul(Q, self.K.transpose(-1, -2)) * self.scale
        A = self.dropout(F.softmax(scores, dim=-1))

        # attn_out: (B, h, N, d_head) -> (B, N, d_model)
        attn_out = torch.matmul(A, self.V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        attn_out = self.out_proj(attn_out)

        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x
