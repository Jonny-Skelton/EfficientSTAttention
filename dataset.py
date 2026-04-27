"""
LargeST Dataset Loader
======================
LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting
Liu et al., NeurIPS 2023  (github.com/liuxu77/LargeST)

Expected data layout:
    data/
      sd/sd.h5      (716 sensors)
      gba/gba.h5    (2,352 sensors)
      gla/gla.h5    (3,834 sensors)
      ca/ca.h5      (8,600 sensors)

Each .h5 file contains the key 'raw_data' with shape (T, N, C).
For traffic speed, C=1 so we squeeze to (T, N).

Download script:
    python dataset.py --download --data_root data/
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATASET_FILES = {
    "SD":  "sd/sd.h5",
    "GBA": "gba/gba.h5",
    "GLA": "gla/gla.h5",
    "CA":  "ca/ca.h5",
}

# Approximate sensor counts for sanity checking
DATASET_SIZES = {"SD": 716, "GBA": 2352, "GLA": 3834, "CA": 8600}


def load_h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        data = f["raw_data"][:]          # (T, N, C) or (T, N)
    if data.ndim == 3:
        data = data[..., 0]              # take speed channel, shape (T, N)
    return data.astype(np.float32)


def split_data(data, train_ratio=0.7, val_ratio=0.1):
    """Split along the time axis: 70 / 10 / 20."""
    T = len(data)
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio))
    return data[:t1], data[t1:t2], data[t2:]


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
            f"{path} not found. Run: python dataset.py --download --data_root {data_root}"
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
# Optional download helper
# ---------------------------------------------------------------------------
LARGESET_URLS = {
    "SD":  "https://zenodo.org/record/8070680/files/sd.h5",
    "GBA": "https://zenodo.org/record/8070680/files/gba.h5",
    "GLA": "https://zenodo.org/record/8070680/files/gla.h5",
    "CA":  "https://zenodo.org/record/8070680/files/ca.h5",
}


def download_datasets(data_root, datasets=None):
    import urllib.request
    datasets = datasets or list(LARGESET_URLS.keys())
    for name in datasets:
        url = LARGESET_URLS[name]
        subdir = os.path.join(data_root, name.lower())
        os.makedirs(subdir, exist_ok=True)
        dest = os.path.join(subdir, f"{name.lower()}.h5")
        if os.path.exists(dest):
            print(f"[skip] {dest} already exists")
            continue
        print(f"Downloading {name} from {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> saved to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--datasets", nargs="+", default=["SD", "GBA", "GLA", "CA"])
    args = parser.parse_args()

    if args.download:
        download_datasets(args.data_root, args.datasets)
    else:
        parser.print_help()
