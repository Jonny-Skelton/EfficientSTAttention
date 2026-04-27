"""
EiFormer Paper: From-Scratch Implementations
=============================================
Four spatial-temporal forecasting models from the EiFormer paper,
all in one file for clarity.

Problem setup (Section 3 of the paper):
    Input:  X ∈ R^{B × H × N × C}  (batch, history, entities, channels)
    Output: Y ∈ R^{B × F × N × C}  (batch, forecast, entities, channels)

Architecture summary (Figure 2 of the paper):
    TSMixer:      Feature MLP (across entities) + Time MLP
    RPMixer:      Random Projection + Feature MLP + Time MLP
    iTransformer: Entity Embedding + Standard Self-Attention (N×N) + Time MLP
    EiFormer:     Entity Embedding + RP Latent Attention + Learnable Latent Attention + Time MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# TSMixer
# =============================================================================
# Paper reference: Chen et al., "TSMixer: An All-MLP Architecture for Time
# Series Forecasting" (arXiv:2303.06053)
#
# Key idea: Two alternating MLPs — one mixes across TIME, one mixes across
# FEATURES (entities). No attention at all.
#
# From Figure 2: TSMixer = component (6) Time MLP + component (5) Feature MLP
#
# IMPORTANT LIMITATION: The Feature MLP uses nn.Linear(N, N), which means N
# is baked into the model weights. If entities appear or vanish at test time,
# this model breaks.
# =============================================================================

class TSMixer(nn.Module):
    """
    Input shape:  (batch, history_len, num_entities)  — univariate, C=1 absorbed
    Output shape: (batch, forecast_len, num_entities)

    For simplicity we treat C=1 (univariate). The public TSMixer code handles
    multi-channel, but the EiFormer paper benchmarks are univariate.
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
        """
        x: (batch, history_len, num_entities)

        Pipeline:
        1. Pass through each MixerBlock (time mixing then feature mixing)
        2. Transpose to (batch, num_entities, history_len) so time is last dim
        3. Apply temporal projection: (batch, num_entities, history_len) -> (batch, num_entities, forecast_len)
        4. Transpose back to (batch, forecast_len, num_entities)
        """
        for block in self.blocks:
            x = block(x)
        x = x.transpose(-1, -2)          # (B, N, H)
        x = self.temporal_proj(x)        # (B, N, F)
        return x.transpose(-1, -2)       # (B, F, N)


class MixerBlock(nn.Module):
    """One block of TSMixer: TimeMixing followed by FeatureMixing.

    This corresponds to MixerLayer in the public code (layers.py lines 229-262).
    """

    def __init__(self, history_len, num_entities, ff_dim=64, dropout=0.1):
        super().__init__()

        # ---- Time Mixing ----
        self.time_fc = nn.Linear(history_len, history_len)
        self.time_norm = nn.LayerNorm(num_entities)

        # ---- Feature Mixing ----
        # N is hardcoded here — TSMixer cannot handle changing entity counts.
        self.feat_fc1 = nn.Linear(num_entities, ff_dim)
        self.feat_fc2 = nn.Linear(ff_dim, num_entities)
        self.feat_norm = nn.LayerNorm(num_entities)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        x: (batch, history_len, num_entities)
        Returns: same shape
        """
        # Time mixing: each entity's time series is mixed independently
        r = x.transpose(-1, -2)          # (B, N, H)
        r = self.act(self.time_fc(r))
        r = self.dropout(r)
        r = r.transpose(-1, -2)          # (B, H, N)
        x = self.time_norm(x + r)

        # Feature mixing: entities are mixed at each timestep
        r = self.act(self.feat_fc1(x))
        r = self.dropout(r)
        r = self.feat_fc2(r)
        r = self.dropout(r)
        x = self.feat_norm(x + r)

        return x


# =============================================================================
# iTransformer
# =============================================================================
# Paper reference: Liu et al., "iTransformer: Inverted Transformers Are
# Effective for Time Series Forecasting" (ICLR 2024 Spotlight)
#
# Key idea: "Invert" the transformer — instead of each token being one timestep
# across all entities, each token is one ENTITY across all timesteps.
# Then self-attention captures inter-entity relationships.
#
# From Figure 2: iTransformer = component (2) Entity Embedding
#                             + component (4) Entity Correlation Attention
#                             + component (6) Time MLP
#
# Complexity: O(N²) because the attention map is N×N (every entity attends
# to every other entity).
#
# ADVANTAGE: Inductive to N — attention works with any number of tokens,
# so new/vanished entities are handled naturally.
# =============================================================================

class iTransformer(nn.Module):
    """
    Input shape:  (batch, history_len, num_entities)
    Output shape: (batch, forecast_len, num_entities)
    """

    def __init__(self, history_len, forecast_len, num_entities, d_model=64,
                 n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        # num_entities not used in weights — iTransformer is inductive to N
        self.num_entities = num_entities

        # Entity embedding: maps each entity's full history to a d_model token
        self.embed = nn.Linear(history_len, d_model)

        self.blocks = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

        # Maps each entity's d_model embedding to forecast_len predictions
        self.out_proj = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        """
        x: (batch, history_len, num_entities)

        Pipeline:
        1. Transpose to (batch, num_entities, history_len) — each entity is a "token"
        2. Entity embed: (batch, N, history_len) -> (batch, N, d_model)
        3. Pass through iTransformerBlocks
        4. Project: (batch, N, d_model) -> (batch, N, forecast_len)
        5. Transpose to (batch, forecast_len, N)
        """
        x = x.transpose(-1, -2)         # (B, N, H)
        x = self.embed(x)               # (B, N, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)            # (B, N, F)
        return x.transpose(-1, -2)      # (B, F, N)


class iTransformerBlock(nn.Module):
    """One transformer block for iTransformer.

    Standard pre-norm transformer block:
        x = x + MultiheadAttention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        # N×N attention — O(N²) complexity
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
        """
        x: (batch, num_entities, d_model)
        Returns: same shape

        Pre-norm residual pattern:
            x = x + Attention(LayerNorm(x))
            x = x + FFN(LayerNorm(x))
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# EiFormer
# =============================================================================
# Paper reference: Sun et al., "Towards Efficient Large Scale Spatial-Temporal
# Time Series Forecasting via Improved Inverted Transformers" (arXiv:2503.10858)
#
# Key idea: Replace standard N×N self-attention with latent attention (N×M)
# where M << N is a fixed number of latent factors. K and V are learnable
# parameters independent of input, so the attention map scales linearly with N.
#
# Architecture (Figure 3 + Figure 4 design 3):
#   1. Entity embedding (same as iTransformer)
#   2. Random Projection block: latent attention with FROZEN K
#   3. Learnable Latent Attention block: latent attention with LEARNED K
#   4. Output projection
#
# Complexity: O(N·M) ≈ O(N) since M is fixed
# =============================================================================

class EiFormer(nn.Module):
    """
    Input shape:  (batch, history_len, num_entities)
    Output shape: (batch, forecast_len, num_entities)
    """

    def __init__(self, history_len, forecast_len, num_entities, d_model=64,
                 n_heads=4, num_latent=16, num_layers=2, dropout=0.1):
        super().__init__()
        # num_entities not used in weights — EiFormer is inductive to N
        self.num_entities = num_entities

        self.embed = nn.Linear(history_len, d_model)

        # First block: random projection (frozen K); remaining: learnable K
        self.blocks = nn.ModuleList([
            EiFormerBlock(d_model, n_heads, num_latent, dropout, freeze_K=(i == 0))
            for i in range(num_layers)
        ])

        self.out_proj = nn.Linear(d_model, forecast_len)

    def forward(self, x):
        """
        x: (batch, history_len, num_entities)

        Pipeline: same as iTransformer but with latent attention instead of self-attention
        1. Transpose to (batch, N, history_len)
        2. Entity embed -> (batch, N, d_model)
        3. Pass through EiFormerBlocks
        4. Project -> (batch, N, forecast_len)
        5. Transpose -> (batch, forecast_len, N)
        """
        x = x.transpose(-1, -2)         # (B, N, H)
        x = self.embed(x)               # (B, N, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)            # (B, N, F)
        return x.transpose(-1, -2)      # (B, F, N)


class EiFormerBlock(nn.Module):
    """One EiFormer block: latent attention + temporal MLP.

    This is the core innovation. Instead of N×N self-attention,
    we use N×M latent attention where M is the number of latent factors.
    """

    def __init__(self, d_model, n_heads, num_latent, dropout=0.1, freeze_K=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model)

        # K: (n_heads, M, d_head) — frozen (random projection) or learnable
        K_init = torch.randn(n_heads, num_latent, self.d_head)
        if freeze_K:
            # register_buffer: tracked in state_dict but not updated by optimizer
            self.register_buffer('K', K_init)
        else:
            self.K = nn.Parameter(K_init)

        # V is always learnable
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
        """
        x: (batch, num_entities, d_model)
        Returns: same shape

        Latent attention (multi-head):
            Q = W_Q(LayerNorm(x))    (B, N, d_model) -> (B, h, N, d_head)
            scores = Q @ K^T / sqrt(d_head)           (B, h, N, M)
            A = softmax(scores, dim=-1)               (B, h, N, M)
            out = A @ V                               (B, h, N, d_head)
            out = concat heads -> W_O                 (B, N, d_model)
        """
        B, N, _ = x.shape

        normed = self.norm1(x)
        Q = self.W_Q(normed)                                   # (B, N, d_model)
        Q = Q.view(B, N, self.n_heads, self.d_head)            # (B, N, h, d_head)
        Q = Q.transpose(1, 2)                                  # (B, h, N, d_head)

        # K: (h, M, d_head), scores: (B, h, N, M)
        scores = torch.matmul(Q, self.K.transpose(-1, -2)) * self.scale
        A = F.softmax(scores, dim=-1)                          # (B, h, N, M)
        A = self.dropout(A)

        # V: (h, M, d_head), attn_out: (B, h, N, d_head)
        attn_out = torch.matmul(A, self.V)
        attn_out = attn_out.transpose(1, 2).contiguous()       # (B, N, h, d_head)
        attn_out = attn_out.view(B, N, -1)                     # (B, N, d_model)
        attn_out = self.out_proj(attn_out)

        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# RPMixer
# =============================================================================
# Paper reference: Yeh et al., "RPMixer: Shaking Up Time Series Forecasting
# with Random Projections for Large Spatial-Temporal Data" (KDD 2024)
#
# Key idea: Add a random projection layer before TSMixer's feature MLP.
# The random projection is a FROZEN linear layer (weights never update).
# This acts as regularization and produces diverse representations.
#
# Architecture:
#   1. Random projection (frozen linear across entities)
#   2. TSMixer blocks (Feature MLP + Time MLP)
#
# Still has the Feature MLP with nn.Linear(N, N), so NOT inductive.
# =============================================================================

class RPMixer(nn.Module):
    """
    Input shape:  (batch, history_len, num_entities)
    Output shape: (batch, forecast_len, num_entities)
    """

    def __init__(self, history_len, forecast_len, num_entities, num_blocks=2,
                 ff_dim=64, dropout=0.1):
        super().__init__()

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
        """
        x: (batch, history_len, num_entities)

        Pipeline:
        1. Apply random projection across entities (transpose, linear, transpose)
        2. Pass through MixerBlocks
        3. Temporal projection
        """
        x = self.random_proj(x)          # (B, H, N) — linear applied to last dim

        for block in self.blocks:
            x = block(x)

        x = x.transpose(-1, -2)          # (B, N, H)
        x = self.temporal_proj(x)        # (B, N, F)
        return x.transpose(-1, -2)       # (B, F, N)


# =============================================================================
# Testing
# =============================================================================
if __name__ == "__main__":
    # Dimensions
    B = 4          # batch
    H = 12         # history
    F_out = 12     # forecast
    N = 100        # entities

    x = torch.randn(B, H, N)

    # --- TSMixer ---
    model = TSMixer(H, F_out, N)
    y = model(x)
    assert y.shape == (B, F_out, N), f"Expected {(B, F_out, N)}, got {y.shape}"
    print(f"TSMixer OK: {y.shape}, params: {sum(p.numel() for p in model.parameters())}")

    # --- iTransformer ---
    model = iTransformer(H, F_out, N)
    y = model(x)
    assert y.shape == (B, F_out, N), f"Expected {(B, F_out, N)}, got {y.shape}"
    print(f"iTransformer OK: {y.shape}, params: {sum(p.numel() for p in model.parameters())}")

    # --- EiFormer ---
    model = EiFormer(H, F_out, N)
    y = model(x)
    assert y.shape == (B, F_out, N), f"Expected {(B, F_out, N)}, got {y.shape}"
    print(f"EiFormer OK: {y.shape}, params: {sum(p.numel() for p in model.parameters())}")

    # --- RPMixer ---
    model = RPMixer(H, F_out, N)
    y = model(x)
    assert y.shape == (B, F_out, N), f"Expected {(B, F_out, N)}, got {y.shape}"
    print(f"RPMixer OK: {y.shape}, params: {sum(p.numel() for p in model.parameters())}")

    # --- Inductiveness test ---
    # TSMixer and RPMixer have N hardcoded in their feature MLP — they crash here.
    # iTransformer and EiFormer work for any N.
    x_bigger = torch.randn(B, H, N + 10)
    print("\nInductiveness test (N + 10 entities):")
    for name, cls, kwargs in [
        ("iTransformer", iTransformer, {"history_len": H, "forecast_len": F_out, "num_entities": N}),
        ("EiFormer",     EiFormer,     {"history_len": H, "forecast_len": F_out, "num_entities": N}),
    ]:
        try:
            m = cls(**kwargs)
            out = m(x_bigger)
            print(f"  {name}: inductive ✓ ({out.shape})")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    for name, cls, kwargs in [
        ("TSMixer",  TSMixer,  {"history_len": H, "forecast_len": F_out, "num_entities": N}),
        ("RPMixer",  RPMixer,  {"history_len": H, "forecast_len": F_out, "num_entities": N}),
    ]:
        try:
            m = cls(**kwargs)
            out = m(x_bigger)
            print(f"  {name}: inductive ✓ ({out.shape})")
        except Exception as e:
            print(f"  {name}: not inductive (expected) — {type(e).__name__}")

    # --- Scalability test ---
    import time
    print("\nScalability test (forward pass time, batch=1):")
    for n in [100, 500, 1000, 5000]:
        x_test = torch.randn(1, H, n)
        row = f"  N={n:5d}:"
        for name, cls in [("TSMixer", TSMixer), ("RPMixer", RPMixer),
                           ("iTransformer", iTransformer), ("EiFormer", EiFormer)]:
            try:
                m = cls(H, F_out, n)
                m.eval()
                with torch.no_grad():
                    t0 = time.perf_counter()
                    m(x_test)
                    dt = (time.perf_counter() - t0) * 1000
                row += f"  {name}={dt:.1f}ms"
            except Exception:
                row += f"  {name}=N/A"
        print(row)
