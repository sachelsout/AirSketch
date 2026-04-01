# Getting Started with Zaratan GPU Training

This guide walks a new team member through one-time setup and first job submission on Zaratan.

## Prerequisites

Complete these **before** starting:

1. **SSH Access**: You have a Zaratan account and can `ssh <uid>@login.zaratan.umd.edu`
2. **Compute Allocation**: Your PI/instructor has allocated GPU hours to your group
3. **Account Name**: You know your account name (ask PI or check `docs/zaratan_notes.md`)

If any of these are missing, contact `hpcsupport@umd.edu` first.

---

## One-Time Setup (15–20 minutes per person)

### 1. SSH into Zaratan

```bash
ssh <uid>@login.zaratan.umd.edu
```

### 2. Update ~/.bashrc with Module Loads

These modules must be loaded every session. Add them to your shell startup:

```bash
cat >> ~/.bashrc << 'EOF'
unset PYTHONPATH
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
module load cuda/12.3.0/gcc/11.3.0/zen2
module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
EOF

source ~/.bashrc
```

Verify: `nvcc --version` should show CUDA 12.3

### 3. Create venv

```bash
python -m venv $HOME/envs/airsketch
source $HOME/envs/airsketch/bin/activate

# Verify environment path
which python
# Expected: ~/envs/airsketch/bin/python
```

### 4. Clone Repo to Home (Will do SCRATCH once we get the permissions)

```bash
cd $HOME
git clone https://github.com/sachelsout/AirSketch.git
cd AirSketch
```

### 5. Install PyTorch with CUDA 12.1

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

### 6. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 7. Generate Lockfiles (for team reproducibility)

```bash
pip freeze > requirements.lock

git add requirements.lock
git commit -m "chore: add Zaratan pip environment lockfile"
git push
```

**First person to do this**: Team members thereafter can skip steps 3–6 and use:

```bash
pip install -r requirements.lock
```

### 8. Test GPU Access (Optional)

Request a quick interactive session:

```bash
salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 \
       --mem=32G --time=00:10:00 --account=msml612-class

# Once allocated (you'll see a new prompt on a compute node):
source $HOME/envs/airsketch/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

exit  # Release allocation
```

---

## Your First Training Job

### 1. Update SLURM Scripts with Your Account

Edit `scripts/slurm/train.sh`, `sweep.sh`, and `preprocess.sh`:

Replace:
```bash
#SBATCH --account=<account>
#SBATCH --mail-user=<your-email>@umd.edu
```

With your actual account and email.

### 2. Verify Your Config

Check that `configs/default.yaml` has the parameters you want:

```bash
cat configs/default.yaml
```

### 3. Submit Job

```bash
cd /scratch/$USER/airsketch
sbatch scripts/slurm/train.sh
```

You'll see output like:
```
Submitted batch job 12345
```

### 4. Monitor Job

```bash
# Quick check
squeue -u $USER

# Watch every 5 seconds
watch -n 5 squeue -u $USER

# See live output
tail -f logs/train_12345.out
```

### 5. After Job Completes

```bash
# Check resource usage and efficiency
seff 12345

# View full output
cat logs/train_12345.out

# Check for errors
cat logs/train_12345.err
```

---

## Common Commands

See [docs/zaratan_notes.md](zaratan_notes.md) for full reference. Quick versions:

```bash
# Request interactive GPU session (1 hour)
salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --account=msml612-class

# Submit training job
sbatch scripts/slurm/train.sh
sbatch scripts/slurm/train.sh configs/custom.yaml

# Hyperparameter sweep (9 parallel jobs)
sbatch scripts/slurm/sweep.sh

# Check queued/running jobs
squeue -u $USER

# Cancel a job
scancel 12345

# View job efficiency (after job finishes)
seff 12345
```

---

## Troubleshooting

### Can't SSH to Zaratan
- Check you have a Zaratan account (request at hpcsupport@umd.edu)
- Try: `ssh -vvv <uid>@login.zaratan.umd.edu` to see detailed errors

### CUDA not detected in interactive session
- Verify modules loaded: module list
- Reload: module purge && unset PYTHONPATH && module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2 cuda/12.3.0/gcc/11.3.0/zen2 cudnn/8.9.7.29-12/gcc/11.3.0/zen2
- Reactivate venv: deactivate && source $HOME/envs/airsketch/bin/activate

### Job dies immediately
- Check error log: `cat logs/train_<job_id>.err`
- Common causes: module not loaded, conda env not found, config file missing
- Use interactive session to debug: `salloc ...` first

### Need to update dependencies
- Update `requirements.txt` locally
- On Zaratan: `pip install -r requirements.txt`
- Re-generate lockfiles:
  ```bash
  pip install -r requirements.txt
  pip freeze > requirements.lock
  git add requirements.lock
  git commit -m "chore: update pip lockfile"
  git push
  ```

---

## Important: Storage & Backups
```
Note: /scratch/$USER is not yet provisioned. Currently using $HOME/AirSketch as the working directory. Move to scratch once provisioned for faster I/O.
```

- Train from: `/scratch/$USER/airsketch/`
- **Back up checkpoints** to GitHub Releases or Google Drive (scratch is periodically wiped)
- Logs: Feel free to delete after reviewing (they're not backed up)

---

## Full Reference

See [docs/zaratan_notes.md](zaratan_notes.md) for:
- Cluster architecture & GPU specs
- Module system details
- All SLURM commands
- Storage tier breakdown
- Troubleshooting guide

---

## Questions?

Contact `hpcsupport@umd.edu` for Zaratan-specific issues, or check the docs.
