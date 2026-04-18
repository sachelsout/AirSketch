#!/bin/bash
#SBATCH --job-name=airsketch-train
#SBATCH --partition=gpu
#SBATCH --account=msml612-class
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rdawkhar@umd.edu

# ── Environment ──────────────────────────────────────────────────────────────
unset PYTHONPATH
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
source $HOME/scratch/envs/airsketch/bin/activate
export PYTHONPATH=$HOME/scratch/envs/airsketch/lib/python3.10/site-packages:$PYTHONPATH
export WANDB_ERROR_REPORTING=false
export WANDB_API_KEY=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

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR=$HOME/scratch/AirSketch
CONFIG=${1:-configs/default.yaml}
cd $REPO_DIR

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs checkpoints
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Config:     $CONFIG"
echo "Start time: $(date)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Running import check..."
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
import mediapipe
import cv2
import onnxruntime
print('All imports OK')
" || { echo "Import check failed — aborting job"; exit 1; }

export WANDB_MODE=offline

# ── Run ───────────────────────────────────────────────────────────────────────
python src/train.py --config $CONFIG

echo "End time: $(date)"