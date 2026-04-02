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
#SBATCH --mail-user=<your-email>@umd.edu

# ── Environment ──────────────────────────────────────────────────────────────
unset PYTHONPATH
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
source $HOME/envs/airsketch/bin/activate

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

# ── Run ───────────────────────────────────────────────────────────────────────
python src/train.py --config $CONFIG

echo "End time: $(date)"