# SLURM Job Scripts

All job submission scripts follow a consistent pattern: load modules, activate conda environment, then run the command.

## Available Scripts

### `train.sh` — Single Training Run

Submits a single training job with a config file.

**Usage:**
```bash
# With default config
sbatch scripts/slurm/train.sh

# With custom config
sbatch scripts/slurm/train.sh configs/experiment_A.yaml
```

**Resources:**
- 1 × A100 GPU
- 8 CPU cores
- 32 GB RAM
- 4 hours

**Output:** `logs/train_<job_id>.out`, `logs/train_<job_id>.err`

---

### `sweep.sh` — Hyperparameter Sweep

Submits 9 parallel jobs for hyperparameter tuning. Each task maps to a combination of `hidden_dim` and `num_layers`.

**Hyperparameter grid:**
- `hidden_dim`: [64, 128, 256]
- `num_layers`: [2, 4, 6]

**Usage:**
```bash
sbatch scripts/slurm/sweep.sh
```

This creates a job array with indices 0–8. SLURM parallelizes them automatically.

**Resources:** (per task)
- 1 × A100 GPU
- 8 CPU cores
- 32 GB RAM
- 6 hours

**Output:** `logs/sweep_<master_job_id>_<array_task_id>.out`

**Monitor array jobs:**
```bash
# See all array tasks
squeue -j <master_job_id>

# Cancel all tasks in array
scancel <master_job_id>
```

---

### `preprocess.sh` — Data Preprocessing (CPU Only)

Runs feature extraction on raw datasets. Does NOT use GPU.

**Usage:**
```bash
sbatch scripts/slurm/preprocess.sh
```

This extracts landmarks from all raw video files and saves processed outputs.

**Resources:**
- Standard partition (CPU only)
- 16 CPU cores
- 64 GB RAM
- 3 hours

**Output:** `logs/preprocess_<job_id>.out`

---

## Before First Submission

1. **Edit each script** to add your account name and email:
   ```bash
   #SBATCH --account=<your-account>
   #SBATCH --mail-user=<your-email>@umd.edu
   ```

2. **Verify resource requests** match your needs:
   - Check GPU availability: `sinfo -p gpu`
   - Review cluster usage: `squeue -p gpu | wc -l`

3. **Test with interactive session first:**
   ```bash
   salloc --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \
          --mem=32G --time=00:30:00 --account=<account>
   cd /scratch/$USER/airsketch
   python src/train.py --config configs/default.yaml --debug
   ```

---

## Template: Custom Training Script

To create a new training job script:

```bash
#!/bin/bash
#SBATCH --job-name=airsketch-custom
#SBATCH --partition=gpu
#SBATCH --account=<account>
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/custom_%j.out
#SBATCH --error=logs/custom_%j.err

module load python/3.10.8 cuda/12.1.0 cudnn/8.9.0-cuda12.1
conda activate airsketch
cd /scratch/$USER/airsketch

python src/train.py --config configs/myconfig.yaml
```

---

## Debugging Failed Jobs

1. **Check error log immediately:** `cat logs/train_<job_id>.err`
2. **View full output:** `cat logs/train_<job_id>.out`
3. **Check resource efficiency:** `seff <job_id>`
4. **Common issues:**
   - Missing modules: Check `module list` in script
   - Conda env not found: Verify `conda activate airsketch` works
   - OOM/timeout: Check `seff` output, increase `--mem` or `--time`

---

## Resources

- [SLURM Quick Reference](https://slurm.schedmd.com/pdfs/summary.pdf)
- [Zaratan Job Scheduler Docs](https://zaratan.documentation.umd.edu/)
