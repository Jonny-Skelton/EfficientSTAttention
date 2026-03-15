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

    def __init__(self, history_len, forecast_len, num_entities, num_blocks=2, hidden_channels=64,
                 ff_dim=64, dropout=0.1):
        super().__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.num_entities = num_entities

        # ---- Build num_blocks MixerBlocks ----
        # Each block contains: TimeMixing then FeatureMixing (see MixerLayer in layers.py)
        # Use nn.ModuleList so PyTorch registers them properly

        self.mixer = nn.ModuleList(
            [
                MixerBlock(
                    history_len=history_len,
                    num_entities=num_entities,
                )
                for _ in range(num_blocks)
            ]
        )
        
        # ---- Temporal Projection ----
        # After all mixer blocks, project from history_len to forecast_len

        self.temporal_projection = nn.Linear(history_len, forecast_len)

    def forward(self, x):
        """
        x: (batch, history_len, num_entities)

        Pipeline:
        1. Pass through each MixerBlock (time mixing then feature mixing)
        2. Transpose to (batch, num_entities, history_len) so time is last dim
        3. Apply temporal projection: (batch, num_entities, history_len) -> (batch, num_entities, forecast_len)
        4. Transpose back to (batch, forecast_len, num_entities)
        """
        for layer in self.mixer:
            x = layer(x)
        x = x.permute(0, 2, 1)
        x_temp = self.temporal_projection(x)
        x = x_temp.permute(0, 2, 1)

        return x


class MixerBlock(nn.Module):
    """One block of TSMixer: TimeMixing followed by FeatureMixing.

    This corresponds to MixerLayer in the public code (layers.py lines 229-262).
    """

    def __init__(self, history_len, num_entities, ff_dim=64, dropout=0.1):
        super().__init__()

        # ---- Time Mixing ----
        # Mixes information ACROSS TIME for each entity independently.
        # Operates on shape (batch, history_len, num_entities)

        self.t_fc1 = nn.Linear(history_len, history_len) # fully connected layer 1.
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_entities) # TODO: may need to be nn.LayerNorm(history_len, num_entities)
        self.activation = nn.ReLU()

        # ---- Feature Mixing ----
        # Mixes information ACROSS ENTITIES (features) at each timestep.
        # Operates on the entity dimension — this is the "spatial" component.
        #
        # This is a two-layer feedforward: num_entities -> ff_dim -> num_entities
        # with activation and dropout between.

        self.f_fc1 = nn.Linear(num_entities, ff_dim)
        self.f_fc2 = nn.Linear(ff_dim, num_entities)
        self.projection = nn.Identity()

    def forward(self, x):
        """
        x: (batch, history_len, num_entities)
        Returns: same shape
        """
        # Time mixing
        x_temp = x.permute(0, 2, 1) #transpose so time is in last dimension
        x_temp = self.activation(self.t_fc1(x_temp)) #linear projection then activation (ReLU)
        x_temp = self.dropout(x_temp)
        x_res = x_temp.permute(0, 2, 1) #transpose back to match original

        x = self.norm(x + x_res) # Adds residual connection then normalizes


        # Feature mixing (linear, activation, dropout, linear, dropout, residual, norm)
        # assumes num_input features = num_output features
        x_proj = self.projection(x)

        x = self.norm(x)

        x = self.f_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.f_fc2(x)

        x = self.dropout(x)

        x = x_proj + x
        x = self.projection(x)

        return x

