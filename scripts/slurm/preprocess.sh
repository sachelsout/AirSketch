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

# ── Environment ──────────────────────────────────────────────────────────────
unset PYTHONPATH
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
source $HOME/envs/airsketch/bin/activate

# ── Paths ─────────────────────────────────────────────────────────────────────
cd $HOME/AirSketch

echo "Start: $(date)"
python src/landmark_extract.py --input data/raw/freihand/ --output data/processed/freihand/
python src/landmark_extract.py --input data/raw/egohands/ --output data/processed/egohands/
echo "End: $(date)"