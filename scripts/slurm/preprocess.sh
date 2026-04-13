#!/bin/bash
#SBATCH --job-name=airsketch-preprocess
#SBATCH --partition=standard
#SBATCH --account=msml612-class
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rdawkhar@umd.edu

# ── Environment ───────────────────────────────────────────────────────────────
unset PYTHONPATH
source $HOME/scratch/envs/airsketch/bin/activate
export PYTHONPATH=$HOME/scratch/envs/airsketch/lib/python3.10/site-packages:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1

# ── Paths ─────────────────────────────────────────────────────────────────────
cd /scratch/zt1/project/msml612/user/rdawkhar/AirSketch
mkdir -p logs data/processed/freihand

# ── Logging ───────────────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"

# ── Sanity check ──────────────────────────────────────────────────────────────
python -c "import mediapipe; import cv2; import numpy; print('Imports OK')" \
    || { echo "Import check failed"; exit 1; }

# ── FreiHAND extraction ───────────────────────────────────────────────────────
python scripts/extract_landmarks.py \
    --input   data/raw/freihand/training/rgb/ \
    --output  data/processed/freihand/ \
    --mode    images \
    --splits  data/splits/freihand_splits.json \
    --workers 16

echo "End time: $(date)"