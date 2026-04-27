#!/bin/bash
#SBATCH --job-name=airsketch-sweep
#SBATCH --partition=gpu
#SBATCH --account=msml612-class
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rdawkhar@umd.edu

# ── Environment ───────────────────────────────────────────────────────────────
unset PYTHONPATH
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
source $HOME/scratch/envs/airsketch/bin/activate
export PYTHONPATH=$HOME/scratch/envs/airsketch/lib/python3.10/site-packages:$PYTHONPATH
export WANDB_MODE=offline
export WANDB_API_KEY=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

REPO_DIR=$HOME/scratch/AirSketch
cd $REPO_DIR
mkdir -p logs checkpoints/sweep

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Config:     hidden=${HIDDEN_DIM} layers=${NUM_LAYERS} lr=${LR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/train.py \
    --config configs/default.yaml \
    --override model.hidden_dim=${HIDDEN_DIM} \
    --override model.num_layers=${NUM_LAYERS} \
    --override training.learning_rate=${LR} \
    --run-name sweep_h${HIDDEN_DIM}_l${NUM_LAYERS}_lr${LR}

echo "End time: $(date)"