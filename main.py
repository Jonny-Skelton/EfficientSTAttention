from TSMixer import *

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