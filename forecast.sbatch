#!/bin/bash
# =============================================================================
# forecast.sbatch — LargeST benchmark: 4 models × 4 datasets × 3 horizons
# =============================================================================
# Job array index -> (model, dataset, horizon) mapping:
#   index = model_idx * 12 + dataset_idx * 3 + horizon_idx
#   models:   TSMixer, RPMixer, iTransformer, EiFormer   (4)
#   datasets: SD, GBA, GLA, CA                           (4)
#   horizons: 3, 6, 12 timesteps (15 min, 30 min, 1 hr)  (3)
#   total:    4 × 4 × 3 = 48 jobs
#
# Submit: sbatch forecast.sbatch
# =============================================================================

#SBATCH --job-name=eiformer_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_l40s&gpu_48gb'
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-47%16
#SBATCH --output=logs/forecast/%x_%A_%a.out
#SBATCH --error=logs/forecast/%x_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Decode array index into (MODEL, DATASET, HORIZON)
# ---------------------------------------------------------------------------
MODELS=(TSMixer RPMixer iTransformer EiFormer)
DATASETS=(SD GBA GLA CA)
HORIZONS=(3 6 12)

IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$((IDX / 12))]}
DATASET=${DATASETS[$(( (IDX % 12) / 3 ))]}
HORIZON=${HORIZONS[$((IDX % 3))]}

echo "============================================"
echo "Job array index : $IDX"
echo "Model           : $MODEL"
echo "Dataset         : $DATASET"
echo "Horizon         : $HORIZON timesteps"
echo "Node            : $(hostname)"
echo "GPU             : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load cuda/13.0

source "/util/opt/anaconda/4.9.2/etc/profile.d/conda.sh"
conda activate eiformer          # or: source venv/bin/activate

# ---------------------------------------------------------------------------
# Working directory and data
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/forecast results

DATA_ROOT="${DATA_ROOT:-/lustre/work/helab/jonnyskel/data/largeST}"

# Download the full LargeST archive if the required sub-dataset is not present.
# Kaggle delivers one zip for all four sub-datasets, so this only runs once
# across the whole array (subsequent jobs skip it immediately).
if [ ! -f "$DATA_ROOT/${DATASET,,}/${DATASET,,}.h5" ]; then
    echo "Dataset not found — downloading from Kaggle (requires ~/.kaggle/kaggle.json) ..."
    python dataset.py --download --data_root "$DATA_ROOT"
fi

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# iTransformer scales O(N²); skip on CA (N≈8600) to avoid OOM
if [[ "$MODEL" == "iTransformer" && "$DATASET" == "CA" ]]; then
    echo "Skipping iTransformer on CA (O(N²) attention, N≈8600 would OOM)"
    exit 0
fi

python train.py \
    --model       "$MODEL"      \
    --dataset     "$DATASET"    \
    --horizon     "$HORIZON"    \
    --history     12            \
    --data_root   "$DATA_ROOT"  \
    --results_dir results       \
    --d_model     64            \
    --n_heads     4             \
    --num_layers  2             \
    --num_blocks  2             \
    --num_latent  16            \
    --dropout     0.1           \
    --epochs      100           \
    --batch_size  64            \
    --lr          1e-3          \
    --weight_decay 1e-4         \
    --patience    15            \
    --num_workers 4             \
    --save_model

echo "Done: $MODEL / $DATASET / horizon=$HORIZON"
