"""
Training and evaluation script for LargeST benchmarks.

Usage:
    python train.py --model EiFormer --dataset SD --horizon 12

The script trains for --epochs epochs with early stopping on validation MAE,
then evaluates on the test set and writes a JSON result to --results_dir.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_model(name, history_len, forecast_len, num_entities, args):
    name = name.lower()
    if name == "tsmixer":
        from TSMixer import TSMixer
        return TSMixer(history_len, forecast_len, num_entities,
                       num_blocks=args.num_blocks, ff_dim=args.d_model, dropout=args.dropout)
    if name == "rpmixer":
        from RPMixer import RPMixer
        return RPMixer(history_len, forecast_len, num_entities,
                       num_blocks=args.num_blocks, ff_dim=args.d_model, dropout=args.dropout)
    if name == "itransformer":
        from iTransformer import iTransformer
        return iTransformer(history_len, forecast_len, num_entities,
                            d_model=args.d_model, n_heads=args.n_heads,
                            num_layers=args.num_layers, dropout=args.dropout)
    if name == "eiformer":
        from EiFormer import EiFormer
        return EiFormer(history_len, forecast_len, num_entities,
                        d_model=args.d_model, n_heads=args.n_heads,
                        num_latent=args.num_latent, num_layers=args.num_layers,
                        dropout=args.dropout)
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def masked_mae(pred, true, null_val=0.0):
    mask = true.abs() > abs(null_val) + 1e-5
    return (pred - true).abs()[mask].mean()


def masked_rmse(pred, true, null_val=0.0):
    mask = true.abs() > abs(null_val) + 1e-5
    return ((pred - true)[mask] ** 2).mean().sqrt()


def masked_mape(pred, true, null_val=0.0):
    mask = true.abs() > abs(null_val) + 1e-5
    return ((pred - true).abs() / true.abs())[mask].mean() * 100.0


@torch.no_grad()
def evaluate(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        # Inverse-transform to original scale for metric computation
        out_np = scaler.inverse_transform(out.cpu().numpy())
        y_np   = scaler.inverse_transform(y.cpu().numpy())
        preds.append(out_np)
        trues.append(y_np)

    pred = torch.from_numpy(np.concatenate(preds))
    true = torch.from_numpy(np.concatenate(trues))
    return {
        "mae":  masked_mae(pred, true).item(),
        "rmse": masked_rmse(pred, true).item(),
        "mape": masked_mape(pred, true).item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, scaler, num_entities = get_dataloaders(
        args.data_root, args.dataset,
        history_len=args.history, forecast_len=args.horizon,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Dataset: {args.dataset}  N={num_entities}  "
          f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")

    model = build_model(args.model, args.history, args.horizon, num_entities, args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=1.0)

    best_val_mae = math.inf
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(x)

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, scaler, device)
        scheduler.step(val_metrics["mae"])

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_mae={val_metrics['mae']:.4f}  val_rmse={val_metrics['rmse']:.4f}  "
              f"val_mape={val_metrics['mape']:.2f}%  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"t={elapsed:.1f}s")

        # Early stopping
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    # Restore best checkpoint and evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, scaler, device)
    print(f"\nTest results (best epoch {best_epoch}):")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")

    result = {
        "model":       args.model,
        "dataset":     args.dataset,
        "horizon":     args.horizon,
        "history":     args.history,
        "num_entities": num_entities,
        "n_params":    n_params,
        "best_epoch":  best_epoch,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"val_{k}":  v for k, v in evaluate(model, val_loader, scaler, device).items()},
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_file = os.path.join(
        args.results_dir,
        f"{args.model}_{args.dataset}_h{args.horizon}.json"
    )
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Optionally save checkpoint
    if args.save_model:
        ckpt_dir = os.path.join(args.results_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{args.model}_{args.dataset}_h{args.horizon}.pt")
        torch.save(best_state, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LargeST benchmark training")

    # Experiment identity
    parser.add_argument("--model",   required=True,
                        choices=["TSMixer", "RPMixer", "iTransformer", "EiFormer"])
    parser.add_argument("--dataset", required=True,
                        choices=["SD", "GBA", "GLA", "CA"])
    parser.add_argument("--horizon", type=int, default=12,
                        help="Forecast horizon in timesteps (5 min each)")
    parser.add_argument("--history", type=int, default=12,
                        help="Input history length in timesteps")

    # Paths
    parser.add_argument("--data_root",   default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--save_model",  action="store_true")

    # Model hyperparameters
    parser.add_argument("--d_model",    type=int, default=64)
    parser.add_argument("--n_heads",    type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_latent", type=int, default=16,
                        help="Number of latent factors M for EiFormer")
    parser.add_argument("--dropout",    type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping patience in epochs")
    parser.add_argument("--num_workers",  type=int,   default=4)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
