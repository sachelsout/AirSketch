# Zaratan HPC Cluster Setup Guide

This document consolidates all Zaratan-specific configuration, commands, and resources needed for the AirSketch team to submit GPU training jobs.

**Last Updated:** April 2026  
**Zaratan Documentation:** https://zaratan.documentation.umd.edu/  
**Support:** hpcsupport@umd.edu

---

## Prerequisites (Must Complete Before First Job)

Before attempting to run any training on Zaratan, ensure the following are done:

- [ ] **All team members have active Zaratan accounts**
  - Request access at `hpcsupport@umd.edu` if needed
  - Allow 2–3 business days for approval
  
- [ ] **Each member can SSH in successfully**
  ```bash
  ssh <uid>@login.zaratan.umd.edu
  ```
  
- [ ] **Compute account name is confirmed**
  - Used in every SLURM script as `#SBATCH --account=<account>`
  - Allocated by PI or course instructor
  - **Account name: msml612-class**
  
- [ ] **GPU partition is verified**
  - Standard GPU partition is `gpu`
  - Verify with: `sinfo -p gpu` (after logging in)

---

## Cluster Architecture & GPU Specifications

### GPU Nodes
| Nodes | GPU | GPUs/node |
|-------|-----|-----------|
| gpu-b9/b10/b11-[1-5] | A100 40GB | 4 |
| gpu-a6-[2-9] | H100 | 4 |
| gpu-a5-1 | V100 | 4 |
| gpu-b11-6 | A100 MIG 5GB | 28 slices |

Target partition: gpu, prefer A100 nodes

**Run these on Zaratan login node to gather cluster info:**

```bash
# Available partitions and their state
sinfo

# GPUs available on GPU nodes (name, GPU model, memory)
sinfo -p gpu -o "%N %G %m"

# Your group's current allocation and usage
sacctmgr show assoc where user=$USER

# Current queue status (how busy is the GPU partition?)
squeue -p gpu --format="%.10i %.12j %.8u %.8T %.10M %.6D %R" | head -20
```

---

## Module System & Environment Setup

Zaratan uses the `module` system to expose compilers, CUDA, and Python. Always load modules **before** creating conda environments.

### Step 1: Load Required Modules

```bash
ssh <uid>@login.zaratan.umd.edu

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2

# Verify
nvcc --version      # CUDA Toolkit version
```

### Step 2: Add Modules to ~/.bashrc (Permanent)

These modules should auto-load every time you log in:

```bash
unset PYTHONPATH
echo "module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2" >> ~/.bashrc
echo "module load cuda/12.3.0/gcc/11.3.0/zen2" >> ~/.bashrc
echo "module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2" >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Create Virtual Environment

```bash
# First, make sure the python module is loaded (should be in ~/.bashrc already)
python --version  # should show 3.10.10

python -m venv $HOME/envs/airsketch
source $HOME/envs/airsketch/bin/activate

# Verify you're in the right environment
which python
# Expected: ~/envs/airsketch/bin/python
python --version
# Expected: Python 3.10.10
```

### Step 4: Install PyTorch with CUDA 12.3

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

**GPU verification** (must run from a compute node):

First, get an interactive GPU session:
```bash
salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --account=msml612-class
```

Once allocated, activate the environment and verify:
```bash
source $HOME/envs/airsketch/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected:
# True
# NVIDIA A100-SXM4-40GB  (or H100/V100 depending on which node is allocated)
```

Exit when done:
```bash
exit
```

### Step 5: Install Project Dependencies

```bash
cd ~/scratch
git clone https://github.com/sachelsout/AirSketch.git
cd AirSketch

pip install -r requirements.txt

# Verify critical dependencies
python -c "import mediapipe; print(mediapipe.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

### Step 6: Export Environment Lockfiles

For reproducibility across team members:

```bash
# Using venv instead of conda, so just export pip packages
pip freeze > requirements.lock

# Commit to repo
git add requirements.lock
git commit -m "chore: add Zaratan pip environment lockfile"
git push
```

---

## Interactive GPU Sessions (for Debugging)

Never run training on the login node—it will be killed. Use an interactive session for GPU debugging:

```bash
# Request 1 GPU, 8 CPUs, 32GB RAM for 1 hour
salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 \
       --mem=32G --time=01:00:00 --account=msml612-class

source $HOME/envs/airsketch/bin/activate
python -c "import torch; print(torch.cuda.is_available())"

# Exit when done (releases allocation immediately)
exit
```

### Shell Alias (add to ~/.bashrc)

```bash
alias gpu-shell='salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --account=msml612-class'

# Usage: just type `gpu-shell` to get an interactive session
```

---

## Job Submission & Monitoring

All SLURM scripts are in `scripts/slurm/` in the repo.

### Submitting Jobs

```bash
# Single training run with default config
sbatch scripts/slurm/train.sh

# Single run with custom config
sbatch scripts/slurm/train.sh configs/experiment_A.yaml

# Hyperparameter sweep (9 parallel jobs)
sbatch scripts/slurm/sweep.sh
```

### Checking Job Status

```bash
# List your queued/running jobs
squeue -u $USER

# Watch live (refreshes every 5 seconds)
watch -n 5 squeue -u $USER

# View job output in real-time
tail -f logs/train_<job_id>.out

# Check resource efficiency after job finishes
seff <job_id>
```

### Job Control

```bash
# Cancel a specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

---

## Storage Guidelines

```
Scratch is provisioned at /scratch/zt1/project/msml612/user/<uid>/.
Home includes symlinks:
scratch -> scratch.msml612/
scratch.msml612 -> /scratch/zt1/project/msml612/user/<uid>/
Use ~/scratch/ as your working scratch location.
```

Zaratan has three storage tiers:

| Location | Path | Use For | Quota | Notes |
|----------|------|---------|-------|-------|
| **Home** | `/home/<uid>/` | Code, configs, small files | 20 GB | Backed up. **Slow**—don't train from here |
| **Scratch** | `/scratch/zt1/project/msml612/user/<uid>/` | Datasets, checkpoints, outputs | 1 TB | Not backed up. **Fast**—train from here |
| **Group** | `/scratch/zt1/<group>/` | Shared datasets across team | Shared | Ask your PI for group path |

**Key Rules:**
- Clone the repo and all data go in `~/scratch/AirSketch/`
- **Back up important checkpoints** to GitHub Releases or Google Drive (scratch is periodically wiped)
- Never store raw datasets in home directory

---

## Troubleshooting

### CUDA not detected on compute node
- Confirm you loaded all three modules: `module list`
- Reactivate your venv: deactivate && source $HOME/envs/airsketch/bin/activate
- If still failing, request interactive session with salloc and test directly

### Job killed immediately (exceeds time, memory, or GPU)
- Check `seff <job_id>` to see actual usage
- Increase `--time`, `--mem`, or reduce batch size in training script

### Environment differences between local and Zaratan
- Use `environment.yml` and `requirements.lock` (committed to repo) as single source of truth
- Always `pip install -r requirements.txt` on Zaratan, not `requirements-dev.txt`

### Out of quota
- Check usage: `quota`
- Clean up old checkpoints from `~/scratch/`
- Old job logs can be deleted: `rm logs/*`

---

## Quick Reference Checklist

**Team member setup (one-time):**
- [ ] SSH access confirmed
- [ ] Modules added to `~/.bashrc`
- [ ] Venv created at $HOME/envs/airsketch and torch.cuda.is_available() returns True from a compute node
- [ ] `requirements.txt` installed successfully
- [ ] GPU shell alias added to `.bashrc` (optional but recommended)

**Before every training submission:**
- [ ] Code changes pushed to `main` branch
- [ ] Config file created/updated in `configs/`
- [ ] Review SLURM script for correct `--account` and resource requests
- [ ] Confirm `~/scratch/AirSketch/` clone is up-to-date

**After job completes:**
- [ ] Check `seff <job_id>` to optimize resource requests
- [ ] **Back up important checkpoints** (copy to GitHub Releases or Google Drive)
- [ ] Review logs in `logs/train_<job_id>.out` for any errors or warnings

---

## Support & Additional Resources

- **Zaratan Documentation:** https://zaratan.documentation.umd.edu/
- **SLURM Quick Reference:** https://slurm.schedmd.com/pdfs/summary.pdf
- **PyTorch CUDA Setup:** https://pytorch.org/get-started/locally/
- **Support Email:** hpcsupport@umd.edu (2–3 business day response typical)

---

## Team Notes

**Account Name:** msml612-class

**Primary PI/Instructor:** Dr. Samet Ayhan (sayhan@umd.edu)

**Team Members & Zaratan UIDs:**
- [ ] Rohan Dawkhar: [rdawkhar]
- [ ] Member 2: [UID]
- [ ] Member 3: [UID]
- [ ] Member 4: [UID]

**Last Verified:** 04/01/2026 
**Verified By:** Rohan Dawkhar
