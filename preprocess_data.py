"""
preprocess_data.py — Convert raw LargeST CA h5 files to the format expected by dataset.py.

Creates:
  <data_root>/ca/ca.h5    (T, 8600, 1)
  <data_root>/sd/sd.h5    (T,   716, 1)  District 11
  <data_root>/gba/gba.h5  (T, 2352, 1)  District 4
  <data_root>/gla/gla.h5  (T, 3834, 1)  Districts 7 + 8 + 12

Reads in time chunks to stay within login-node memory limits.

Usage:
  python preprocess_data.py --data_root /lustre/work/helab/jonnyskel/data/largeST --year 2019
"""

import argparse
import csv
import os

import numpy as np
import h5py

SUBSETS = {
    "CA":  None,
    "SD":  {11},
    "GBA": {4},
    "GLA": {7, 8, 12},
}

OUTPUT = {
    "CA":  "ca/ca.h5",
    "SD":  "sd/sd.h5",
    "GBA": "gba/gba.h5",
    "GLA": "gla/gla.h5",
}


def load_sensor_indices(meta_path, districts):
    indices = []
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if districts is None or int(row["District"]) in districts:
                indices.append(int(row["ID2"]))
    return np.array(sorted(indices), dtype=np.intp)


def preprocess_subset(name, indices, src_ds, out_path, chunk_size=5000):
    T = src_ds.shape[0]
    N = len(indices)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as dst:
        ds = dst.create_dataset("raw_data", shape=(T, N, 1), dtype=np.float32,
                                chunks=(min(chunk_size, T), min(N, 512), 1))
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = src_ds[start:end, :]
            chunk = chunk[:, indices].astype(np.float32)
            ds[start:end, :, 0] = chunk
            print(f"    {name}: {end}/{T} timesteps written", end="\r", flush=True)
    print()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  {name:4s}  shape=({T}, {N}, 1)  ->  {out_path}  ({size_mb:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/lustre/work/helab/jonnyskel/data/largeST")
    parser.add_argument("--year", type=int, default=2019,
                        help="Which year to use (2017-2021); each year = 105120 timesteps")
    parser.add_argument("--chunk_size", type=int, default=5000,
                        help="Timesteps to read at once (~5000*8600*4 ≈ 172 MB)")
    args = parser.parse_args()

    meta_path = os.path.join(args.data_root, "ca_meta.csv")
    raw_path = os.path.join(args.data_root, f"ca_his_raw_{args.year}.h5")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    print(f"Source: {raw_path}")
    with h5py.File(raw_path, "r") as src:
        src_ds = src["t/block0_values"]
        print(f"  Shape: {src_ds.shape}  dtype: {src_ds.dtype}")

        for name, districts in SUBSETS.items():
            out_path = os.path.join(args.data_root, OUTPUT[name])
            if os.path.exists(out_path):
                print(f"  {name:4s}  already exists, skipping ({out_path})")
                continue
            indices = load_sensor_indices(meta_path, districts)
            preprocess_subset(name, indices, src_ds, out_path, args.chunk_size)

    print("Done.")


if __name__ == "__main__":
    main()
