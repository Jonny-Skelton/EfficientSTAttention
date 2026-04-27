# EfficientSTAttention

From-scratch PyTorch implementations of four spatial-temporal forecasting models from the [EiFormer paper](https://arxiv.org/abs/2503.10858), evaluated on the [LargeST benchmark](https://github.com/liuxu77/LargeST).

## Models

| Model | Complexity | Inductive to N | Key idea |
|---|---|---|---|
| **TSMixer** | O(N) | No | Alternating time and entity MLPs — no attention |
| **RPMixer** | O(N) | No | Frozen random projection before TSMixer blocks |
| **iTransformer** | O(N²) | Yes | Each entity is a token; standard self-attention |
| **EiFormer** | O(N·M) | Yes | Latent attention: K and V are M learnable parameters, not derived from input |

> **Inductive to N** means the model can run inference on a different number of entities than it was trained on. TSMixer and RPMixer bake N into their feature-MLP weight matrices; iTransformer and EiFormer do not.

### Architecture (Figure 2 of the EiFormer paper)

```
Input (B, H, N)
      │
      ├─ TSMixer:      TimeMLP ─► FeatureMLP ─► temporal projection
      ├─ RPMixer:      RandProj ─► TimeMLP ─► FeatureMLP ─► temporal projection
      ├─ iTransformer: Embed ─► N×N Attention ─► FFN ─► output proj
      └─ EiFormer:     Embed ─► N×M LatentAttn(frozen K) ─► N×M LatentAttn(learned K) ─► output proj
                                                                                         │
Output (B, F, N)                                                                         ▼
```

### EiFormer latent attention

Standard self-attention computes Q, K, V all from the input → N×N attention map (quadratic).

EiFormer fixes K and V as learnable parameters of shape (M, d) where M ≪ N:

```
Q = embed(x)  W_Q           (B, N, d)
scores = Q @ K^T / √d        (B, N, M)  — linear in N!
A = softmax(scores, dim=-1)  (B, N, M)
out = A @ V                  (B, N, d)
```

The first block uses a **frozen** random K (random projection); subsequent blocks learn K.

## Dataset — LargeST

Four California traffic speed sub-datasets at 5-minute resolution:

| Sub-dataset | Sensors (N) | Timesteps |
|---|---|---|
| SD (San Diego) | 716 | ~105,000 |
| GBA (Greater Bay Area) | 2,352 | ~105,000 |
| GLA (Greater Los Angeles) | 3,834 | ~105,000 |
| CA (California) | 8,600 | ~105,000 |

Source: https://www.kaggle.com/datasets/liuxu77/largest

After download and extraction the layout should be:
```
<data_root>/
  sd/sd.h5
  gba/gba.h5
  gla/gla.h5
  ca/ca.h5
```

## Setup

```bash
conda create -n eiformer python=3.11
conda activate eiformer
pip install torch torchvision h5py numpy kaggle
```

## Downloading the dataset on an SSH cluster

The dataset is hosted on Kaggle and must be downloaded via the Kaggle API.
Follow these steps **once**, then the SLURM script handles the rest automatically.

### Step 1 — Get a Kaggle API token (do this on your local machine)

1. Log in at [kaggle.com](https://www.kaggle.com) and go to **Settings → API → Create New Token**.
2. This downloads `kaggle.json` to your local machine (usually `~/Downloads/kaggle.json`).

### Step 2 — Copy the token to the cluster

```bash
# On your local machine:
ssh <user>@<cluster> "mkdir -p ~/.kaggle"
scp ~/Downloads/kaggle.json <user>@<cluster>:~/.kaggle/kaggle.json

# On the cluster — restrict permissions (required by the Kaggle CLI):
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3 — Download and extract

```bash
# On the cluster, inside this repo:
export DATA_ROOT=/scratch/$USER/largeST

python dataset.py --download --data_root $DATA_ROOT
```

This runs `kaggle datasets download -d liuxu77/largest`, saves `largest.zip` to
`$DATA_ROOT`, then extracts it in place. The zip is ~10 GB so allow a few minutes.
If the job is interrupted it will resume from the zip if it was already downloaded.

### Step 4 — Verify

```bash
ls $DATA_ROOT/sd/ $DATA_ROOT/gba/ $DATA_ROOT/gla/ $DATA_ROOT/ca/
# Expect: sd.h5  gba.h5  gla.h5  ca.h5
```

## Smoke test

```bash
python main.py
```

Expected output:
```
TSMixer          output=(4, 12, 100)  params=...
RPMixer          output=(4, 12, 100)  params=...
iTransformer     output=(4, 12, 100)  params=...
EiFormer         output=(4, 12, 100)  params=...

Inductiveness (N+10 entities, model trained on N):
  TSMixer          not inductive (expected for MLP-based models)
  RPMixer          not inductive (expected for MLP-based models)
  iTransformer     inductive ✓
  EiFormer         inductive ✓
```

## Training

```bash
python train.py --model EiFormer --dataset SD --horizon 12 --data_root data/
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | required | TSMixer, RPMixer, iTransformer, EiFormer |
| `--dataset` | required | SD, GBA, GLA, CA |
| `--horizon` | 12 | Forecast horizon in timesteps (×5 min) |
| `--history` | 12 | Input window length |
| `--d_model` | 64 | Hidden dimension |
| `--num_latent` | 16 | Latent factors M (EiFormer only) |
| `--epochs` | 100 | Max epochs (early stopping at `--patience 15`) |
| `--batch_size` | 64 | |
| `--lr` | 1e-3 | Adam learning rate |

Results are written as JSON to `results/<model>_<dataset>_h<horizon>.json`.

## SLURM experiment

Runs all 48 combinations (4 models × 4 datasets × 3 horizons: 15/30/60 min) with at most 16 concurrent jobs.

```bash
# Download data first (see "Downloading the dataset" above)
export DATA_ROOT=/scratch/$USER/largeST
python dataset.py --download --data_root $DATA_ROOT

# Submit the array job
sbatch forecast.sbatch
```

The array index maps to `(model, dataset, horizon)` as:
```
index = model_idx * 12 + dataset_idx * 3 + horizon_idx
models:   TSMixer, RPMixer, iTransformer, EiFormer
datasets: SD, GBA, GLA, CA
horizons: 3, 6, 12 timesteps  (15 min / 30 min / 1 hr)
```

> **Note:** iTransformer on CA (N≈8,600) is automatically skipped in the SLURM script because O(N²) attention at that scale exceeds GPU memory.

Results land in `results/<model>_<dataset>_h<horizon>.json`. Logs go to `logs/forecast/`.

## Metrics

All metrics are computed after inverting the training-set normalisation, with zero values masked out (common for traffic speed data where 0 indicates missing sensors):

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

## File structure

```
.
├── TSMixer.py        # TSMixer + MixerBlock
├── RPMixer.py        # RPMixer (imports MixerBlock from TSMixer)
├── iTransformer.py   # iTransformer + iTransformerBlock
├── EiFormer.py       # EiFormer + EiFormerBlock
├── allModels.py      # All four models in one file (reference / self-test)
├── dataset.py        # LargeST data loader + download helper
├── train.py          # Training loop, evaluation, result saving
├── main.py           # Smoke test
└── forecast.sbatch   # SLURM job array (48 experiments)
```

## References

- Sun et al., "Towards Efficient Large Scale Spatial-Temporal Time Series Forecasting via Improved Inverted Transformers", arXiv:2503.10858
- Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024
- Chen et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting", arXiv:2303.06053
- Yeh et al., "RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data", KDD 2024
- Liu et al., "LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting", NeurIPS 2023
