#!/bin/bash
#SBATCH --job-name=airsketch-preprocess
#SBATCH --partition=standard
#SBATCH --account=msml612-class
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rdawkhar@umd.edu

# ── Environment ──────────────────────────────────────────────────────────────
unset PYTHONPATH
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
source $HOME/envs/airsketch/bin/activate
export PYTHONPATH=$HOME/envs/airsketch/lib/python3.10/site-packages:$PYTHONPATH
export WANDB_ERROR_REPORTING=false
export WANDB_API_KEY=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

# ── Paths ─────────────────────────────────────────────────────────────────────
cd $HOME/scratch/AirSketch
mkdir -p logs data/processed/freihand data/processed/egohands

# ── Logging ───────────────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Running import check..."
python -c "
import mediapipe
import cv2
import numpy
print('All imports OK')
" || { echo "Import check failed — aborting job"; exit 1; }

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Processing FreiHand..."
python src/landmark_extract.py \
  --input  data/raw/freihand/ \
  --output data/processed/freihand/

echo "Processing EgoHands..."
python src/landmark_extract.py \
  --input  data/raw/egohands/ \
  --output data/processed/egohands/

echo "End time: $(date)"