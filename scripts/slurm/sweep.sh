#!/bin/bash
#SBATCH --job-name=airsketch-sweep
#SBATCH --partition=gpu
#SBATCH --account=msml612-class
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-8          # 9 configs: 3 hidden_dims × 3 num_layers
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --mail-type=ARRAY_TASKS_END,FAIL
#SBATCH --mail-user=<your-email>@umd.edu

# ── Environment ──────────────────────────────────────────────────────────────
unset PYTHONPATH
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
source $HOME/envs/airsketch/bin/activate

# ── Paths ─────────────────────────────────────────────────────────────────────
cd $HOME/AirSketch

# ── Hyperparameter grid ───────────────────────────────────────────────────────
HIDDEN_DIMS=(64 128 256)
NUM_LAYERS=(2 4 6)

HIDDEN=${HIDDEN_DIMS[$((SLURM_ARRAY_TASK_ID / 3))]}
LAYERS=${NUM_LAYERS[$((SLURM_ARRAY_TASK_ID % 3))]}

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "hidden_dim=$HIDDEN  num_layers=$LAYERS"

python src/train.py \
  --config configs/default.yaml \
  --override model.hidden_dim=$HIDDEN \
  --override model.num_layers=$LAYERS \
  --run-name "sweep_h${HIDDEN}_l${LAYERS}"