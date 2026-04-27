"""
Smoke-test all four models with a synthetic batch.
Run: python main.py
"""

import torch
from TSMixer     import TSMixer
from RPMixer     import RPMixer
from iTransformer import iTransformer
from EiFormer    import EiFormer

B, H, F_out, N = 4, 12, 12, 100
x = torch.randn(B, H, N)

models = [
    ("TSMixer",      TSMixer(H, F_out, N)),
    ("RPMixer",      RPMixer(H, F_out, N)),
    ("iTransformer", iTransformer(H, F_out, N)),
    ("EiFormer",     EiFormer(H, F_out, N)),
]

for name, model in models:
    y = model(x)
    assert y.shape == (B, F_out, N), f"{name}: expected {(B, F_out, N)}, got {y.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name:15s}  output={tuple(y.shape)}  params={n_params:,}")

# Inductiveness test — which models generalise to unseen N?
print("\nInductiveness (N+10 entities, model trained on N):")
x_big = torch.randn(B, H, N + 10)
for name, model in models:
    try:
        out = model(x_big)
        print(f"  {name:15s}  inductive OK  {tuple(out.shape)}")
    except (RuntimeError, Exception) as e:
        print(f"  {name:15s}  not inductive ({type(e).__name__})")
