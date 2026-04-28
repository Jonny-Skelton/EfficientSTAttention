"""
LargeST Dataset Loader
======================
LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting
Liu et al., NeurIPS 2023  (github.com/liuxu77/LargeST)
Kaggle:  https://www.kaggle.com/datasets/liuxu77/largest

Expected data layout after download:
    <data_root>/
      sd/sd.h5      (716 sensors)
      gba/gba.h5    (2,352 sensors)
      gla/gla.h5    (3,834 sensors)
      ca/ca.h5      (8,600 sensors)

Each .h5 file contains the key 'raw_data' with shape (T, N, C).
For traffic speed, C=1 so we squeeze to (T, N).

Download (requires kaggle CLI and API token — see README):
    python dataset.py --download --data_root /scratch/$USER/largeST
"""

import argparse
import os
import subprocess
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

KAGGLE_DATASET = "liuxu77/largest"

DATASET_FILES = {
    "SD":  "sd/sd.h5",
    "GBA": "gba/gba.h5",
    "GLA": "gla/gla.h5",
    "CA":  "ca/ca.h5",
}

DATASET_SIZES = {"SD": 716, "GBA": 2352, "GLA": 3834, "CA": 8600}


# ---------------------------------------------------------------------------
# Download from Kaggle
# ---------------------------------------------------------------------------

def download_from_kaggle(data_root):
    """
    Download the full LargeST dataset from Kaggle and unzip it.

    Requires:
      - kaggle CLI installed:  pip install kaggle
      - API token at:          ~/.kaggle/kaggle.json
        (download from https://www.kaggle.com/settings -> API -> Create New Token)
    """
    os.makedirs(data_root, exist_ok=True)

    # Check all four h5 files already present
    missing = [
        name for name, rel in DATASET_FILES.items()
        if not os.path.exists(os.path.join(data_root, rel))
    ]
    if not missing:
        print("All LargeST h5 files already present — skipping download.")
        return

    zip_path = os.path.join(data_root, "largest.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading {KAGGLE_DATASET} from Kaggle ...")
        _check_kaggle_credentials()
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", KAGGLE_DATASET,
             "-p", data_root],
            check=True,
        )
        print(f"Download complete: {zip_path}")
    else:
        print(f"Archive already present: {zip_path}")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)
    print(f"Extracted to {data_root}/")

    # Verify expected files exist after extraction
    for name, rel in DATASET_FILES.items():
        full = os.path.join(data_root, rel)
        if os.path.exists(full):
            size_mb = os.path.getsize(full) / 1e6
            print(f"  {name}: {full}  ({size_mb:.0f} MB)")
        else:
            print(f"  WARNING: {full} not found after extraction.")
            print(f"  Check the archive contents with: unzip -l {zip_path}")


def _check_kaggle_credentials():
    import json
    cred = os.path.expanduser("~/.kaggle/kaggle.json")
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        # KAGGLE_API_TOKEN may be a JSON blob {"username":..,"key":..} or a bare key string.
        try:
            parsed = json.loads(token)
        except json.JSONDecodeError:
            parsed = {"username": "", "key": token}
        os.makedirs(os.path.dirname(cred), exist_ok=True)
        with open(cred, "w") as f:
            json.dump(parsed, f)
        os.chmod(cred, 0o600)
        return
    if not os.path.exists(cred):
        raise FileNotFoundError(
            "Kaggle API token not found. Either:\n"
            "  - Set env var:  export KAGGLE_API_TOKEN=<your_token>\n"
            "  - Or place credentials at ~/.kaggle/kaggle.json\n"
            "    (download from https://www.kaggle.com/settings -> API -> Create New Token)\n"
        )


# ---------------------------------------------------------------------------
# H5 loader
# ---------------------------------------------------------------------------

def load_h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        data = f["raw_data"][:]          # (T, N, C) or (T, N)
    if data.ndim == 3:
        data = data[..., 0]              # take speed channel -> (T, N)
    data = data.astype(np.float32)
    np.nan_to_num(data, nan=0.0, copy=False)  # missing sensors -> 0 (masked in metrics)
    return data


# ---------------------------------------------------------------------------
# Train / val / test split (by time)
# ---------------------------------------------------------------------------

def split_data(data, train_ratio=0.7, val_ratio=0.1):
    """Split along the time axis: 70 / 10 / 20."""
    T = len(data)
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio))
    return data[:t1], data[t1:t2], data[t2:]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class StandardScaler:
    """Zero-mean, unit-variance normalisation fitted on training data."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()
        return self

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return data * (self.std + 1e-8) + self.mean


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """
    Yields (x, y) pairs from a sliding window over a (T, N) array.

    x: (history_len, N)
    y: (forecast_len, N)
    """

    def __init__(self, data, history_len, forecast_len):
        self.data = torch.from_numpy(data)
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.n_samples = len(data) - history_len - forecast_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.history_len]
        y = self.data[idx + self.history_len : idx + self.history_len + self.forecast_len]
        return x, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(data_root, dataset_name, history_len=12, forecast_len=12,
                    batch_size=64, num_workers=4):
    """
    Load and preprocess one LargeST sub-dataset.

    Returns:
        train_loader, val_loader, test_loader, scaler, num_entities
    """
    path = os.path.join(data_root, DATASET_FILES[dataset_name.upper()])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run:  python dataset.py --download --data_root {data_root}"
        )

    data = load_h5(path)
    train_raw, val_raw, test_raw = split_data(data)

    scaler = StandardScaler().fit(train_raw)
    train_data = scaler.transform(train_raw)
    val_data   = scaler.transform(val_raw)
    test_data  = scaler.transform(test_raw)

    train_ds = SlidingWindowDataset(train_data, history_len, forecast_len)
    val_ds   = SlidingWindowDataset(val_data,   history_len, forecast_len)
    test_ds  = SlidingWindowDataset(test_data,  history_len, forecast_len)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    num_entities = data.shape[1]
    return train_loader, val_loader, test_loader, scaler, num_entities


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the LargeST dataset from Kaggle."
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download and extract the full dataset from Kaggle."
    )
    parser.add_argument(
        "--data_root", default="data",
        help="Directory to download and extract data into."
    )
    args = parser.parse_args()

    if args.download:
        download_from_kaggle(args.data_root)
    else:
        parser.print_help()
