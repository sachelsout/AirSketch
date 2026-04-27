#!/bin/bash
# Submit 5 targeted sweep configs

cd /home/rdawkhar/scratch/AirSketch

# Config 1: hidden=128, layers=4, lr=5e-4 (likely best per spec)
HIDDEN_DIM=128 NUM_LAYERS=4 LR=0.0005 sbatch scripts/slurm/sweep_agent.sh

# Config 2: hidden=256, layers=6, lr=1e-4 (more capacity)
HIDDEN_DIM=256 NUM_LAYERS=6 LR=0.0001 sbatch scripts/slurm/sweep_agent.sh

# Config 3: hidden=128, layers=6, lr=5e-4 (more layers)
HIDDEN_DIM=128 NUM_LAYERS=6 LR=0.0005 sbatch scripts/slurm/sweep_agent.sh

# Config 4: hidden=256, layers=4, lr=5e-4 (more dim)
HIDDEN_DIM=256 NUM_LAYERS=4 LR=0.0005 sbatch scripts/slurm/sweep_agent.sh

# Config 5: hidden=64, layers=2, lr=1e-3 (small baseline)
HIDDEN_DIM=64 NUM_LAYERS=2 LR=0.001 sbatch scripts/slurm/sweep_agent.sh

echo "All 5 configs submitted."