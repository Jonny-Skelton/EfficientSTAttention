import torch
import torch.nn as nn
import torch.nn.functional as F
from TSMixer import MixerBlock

class RPMixer(nn.Module):
    """
    Input shape:  (batch, history_len, num_entities)
    Output shape: (batch, forecast_len, num_entities)
    """

    def __init__(self, history_len, forecast_len, num_entities, num_blocks=2,
                 ff_dim=64, dropout=0.1):
        super().__init__()

        # ---- Random Projection ----
        # A frozen linear layer that mixes across entities.
        # Create a nn.Linear(num_entities, num_entities) and then FREEZE its parameters.

        self.rp = nn.Linear(num_entities, num_entities)
        for param in self.rp.parameters():
            param.requires_grad = False
        # ---- TSMixer blocks (same as TSMixer) ----
        self.rpMixer = nn.ModuleList(
            [
                MixerBlock(
                    history_len=history_len,
                    num_entities=num_entities
                )
                for _ in range(num_blocks)
            ]
        )
        # ---- Temporal Projection ----
        self.temporal_proj = nn.Linear(history_len, forecast_len)

        pass

    def forward(self, x):
        """
        x: (batch, history_len, num_entities)

        Pipeline:
        1. Apply random projection across entities (transpose, linear, transpose)
        2. Pass through MixerBlocks
        3. Temporal projection
        """
        x = self.rp(x)

        for layer in self.rpMixer:
            x = layer(x)
        
        x = x.permute(0, 2, 1)
        x = self.temporal_proj(x)
        x = x.permute(0, 2, 1)

        return x