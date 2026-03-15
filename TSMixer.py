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
        # Store dimensions you'll need in forward()
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.num_entities = num_entities

        # ---- Build num_blocks MixerBlocks ----
        # Each block contains: TimeMixing then FeatureMixing (see MixerLayer in layers.py)
        # Use nn.ModuleList so PyTorch registers them properly
        # Docs: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html

        # TODO: Create a nn.ModuleList of MixerBlock instances (you'll define MixerBlock below)

        # ---- Temporal Projection ----
        # After all mixer blocks, project from history_len to forecast_len
        # This operates on the TIME dimension
        # Docs: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        #
        # TODO: Create nn.Linear(history_len, forecast_len)
        #       This will be applied after transposing so time is the last dimension

        pass

    def forward(self, x):
        """
        x: (batch, history_len, num_entities)

        Pipeline:
        1. Pass through each MixerBlock (time mixing then feature mixing)
        2. Transpose to (batch, num_entities, history_len) so time is last dim
        3. Apply temporal projection: (batch, num_entities, history_len) -> (batch, num_entities, forecast_len)
        4. Transpose back to (batch, forecast_len, num_entities)
        """
        # TODO: Implement the forward pass

        pass


class MixerBlock(nn.Module):
    """One block of TSMixer: TimeMixing followed by FeatureMixing.

    This corresponds to MixerLayer in the public code (layers.py lines 229-262).
    """

    def __init__(self, history_len, num_entities, ff_dim=64, dropout=0.1):
        super().__init__()

        # ---- Time Mixing ----
        # Mixes information ACROSS TIME for each entity independently.
        # Operates on shape (batch, history_len, num_entities)
        #
        # Steps in forward:
        #   1. Transpose to (batch, num_entities, history_len) — time becomes last dim
        #   2. Apply linear: history_len -> history_len
        #   3. Apply activation (ReLU)
        #   4. Apply dropout
        #   5. Transpose back to (batch, history_len, num_entities)
        #   6. Add residual connection (input + output)
        #   7. Apply LayerNorm
        #
        # TODO: Create nn.Linear(history_len, history_len) for time mixing
        # TODO: Create nn.Dropout(dropout)
        # TODO: Create nn.LayerNorm(num_entities) for normalization after residual
        # Docs LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

        pass

        # ---- Feature Mixing ----
        # Mixes information ACROSS ENTITIES (features) at each timestep.
        # Operates on the entity dimension — this is the "spatial" component.
        #
        # This is a two-layer feedforward: num_entities -> ff_dim -> num_entities
        # with activation and dropout between.
        #
        # Steps in forward:
        #   1. Apply linear: num_entities -> ff_dim
        #   2. Apply activation (ReLU)
        #   3. Apply dropout
        #   4. Apply linear: ff_dim -> num_entities
        #   5. Apply dropout
        #   6. Add residual connection
        #   7. Apply LayerNorm
        #
        # NOTE: nn.Linear(num_entities, ...) — N is HARDCODED in the weight matrix.
        #       This is why TSMixer can't handle changing entity counts.
        #
        # TODO: Create nn.Linear(num_entities, ff_dim)
        # TODO: Create nn.Linear(ff_dim, num_entities)
        # TODO: Create nn.Dropout(dropout)
        # TODO: Create nn.LayerNorm(num_entities)

        pass

    def forward(self, x):
        """
        x: (batch, history_len, num_entities)
        Returns: same shape
        """
        # TODO: Implement time mixing (transpose, linear, activation, dropout, transpose, residual, norm)
        # TODO: Implement feature mixing (linear, activation, dropout, linear, dropout, residual, norm)

        pass

