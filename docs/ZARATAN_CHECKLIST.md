# Zaratan Setup Completion Checklist

Use this checklist to verify that all prerequisites and setup steps are complete before submitting training jobs.

## Team-Level Prerequisites

Before **anyone** attempts to run jobs:

- [ ] **All 4 team members have Zaratan accounts**
  - Requested at: hpcsupport@umd.edu
  - Confirmation emails received by: [DATE]
  
- [ ] **Account name confirmed with PI/instructor**
  - Account name: ______________________________________
  - Documented in: `docs/zaratan_notes.md` (line: "Account Name")
  
- [ ] **GPU partition verified**
  - Run on Zaratan: `sinfo -p gpu`
  - Partition is available: YES / NO
  - GPU type: NVIDIA A100 40GB (expected)

---

## Individual One-Time Setup

Each team member completes this once. First person should generate `environment.yml` and `requirements.lock`.

### Prerequisites
- [ ] Zaratan account active (can SSH)
- [ ] Know the account name from PI

### SSH & Module Setup
- [ ] SSH into Zaratan: `ssh <uid>@zaratan.umd.edu`
- [ ] Modules added to `~/.bashrc`:
  ```bash
  module load python/3.10.8
  module load cuda/12.1.0
  module load cudnn/8.9.0-cuda12.1
  ```
- [ ] Modules loadable: `source ~/.bashrc && module list` shows all 3 modules
- [ ] CUDA verified: `nvcc --version` shows 12.1.0

### Conda Environment
- [ ] Conda environment created: `conda create -n airsketch python=3.10 -y`
- [ ] Environment activatable: `conda activate airsketch`
- [ ] Python path correct: `which python` shows `/home/<uid>/.conda/envs/airsketch/bin/python`

### Repository & Dependencies
- [ ] Repo cloned to `/scratch/$USER/airsketch/`
- [ ] Dependencies installed: `pip install -r requirements.txt` ✓
- [ ] PyTorch installed with CUDA: `pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121` ✓
- [ ] All imports working: `python -c "import torch; import cv2; import mediapipe"`

### First-Time: Generate Lockfiles
(Only first person does this; generates `environment.yml` and `requirements.lock`)

- [ ] Lockfiles generated:
  ```bash
  conda env export > environment.yml
  pip freeze > requirements.lock
  ```
- [ ] Lockfiles committed to repo:
  ```bash
  git add environment.yml requirements.lock
  git commit -m "chore: add Zaratan environment lockfiles"
  git push
  ```

### Subsequent Team Members: Use Lockfiles
- [ ] Lockfiles copied from repo (via `git pull`)
- [ ] Conda environment created from lockfile:
  ```bash
  conda env create -f environment.yml -n airsketch
  ```
- [ ] Dependencies installed from lockfile:
  ```bash
  pip install -r requirements.lock
  ```

### GPU Verification (Optional but Recommended)
- [ ] Interactive GPU session requested:
  ```bash
  salloc --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \
         --mem=32G --time=00:10:00 --account=<account>
  ```
- [ ] GPU detected from compute node:
  ```bash
  conda activate airsketch
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
  ```
  Should print: `True` and `NVIDIA A100-SXM4-40GB`
- [ ] Interactive session exited: `exit`

---

## SLURM Script Preparation

Before submitting any jobs:

- [ ] `scripts/slurm/train.sh` edited with:
  - [ ] `--account=<account>` replaced with actual account
  - [ ] `--mail-user=<your-email>@umd.edu` replaced with your email
  
- [ ] `scripts/slurm/sweep.sh` edited with:
  - [ ] `--account=<account>` replaced with actual account
  - [ ] `--mail-user=<your-email>@umd.edu` replaced with your email
  
- [ ] `scripts/slurm/preprocess.sh` edited with:
  - [ ] `--account=<account>` replaced with actual account
  - [ ] `--mail-user=<your-email>@umd.edu` replaced with your email

---

## First Job Submission & Test

Before running real training:

### Test with Interactive Session
- [ ] SSH into Zaratan: `ssh <uid>@zaratan.umd.edu`
- [ ] Navigate to repo: `cd /scratch/$USER/airsketch`
- [ ] Request interactive GPU:
  ```bash
  salloc --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \
         --mem=32G --time=00:30:00 --account=<account>
  ```
- [ ] Test training code:
  ```bash
  conda activate airsketch
  python src/train.py --config configs/default.yaml --debug
  ```
- [ ] Code runs without errors (check for 1–2 iterations, then Ctrl+C)
- [ ] GPU is utilized: `nvidia-smi` during training shows GPU usage
- [ ] Exit session: `exit`

### Submit First Test Job
- [ ] Config file verified: `configs/default.yaml` exists and looks correct
- [ ] Submit job:
  ```bash
  cd /scratch/$USER/airsketch
  sbatch scripts/slurm/train.sh
  ```
- [ ] Job submitted successfully (shows `Submitted batch job <job_id>`)
- [ ] Job status checked: `squeue -u $USER` shows job in queue
- [ ] Job status monitored: `watch -n 5 squeue -u $USER` shows job progressing
- [ ] Job output monitored: `tail -f logs/train_<job_id>.out` shows training starting
- [ ] Job completes successfully (exits without timeout/OOM/crash)

### Verify Job Completion
- [ ] Job finished: `squeue -u $USER` no longer shows the job
- [ ] Resource efficiency checked: `seff <job_id>` shows CPU/GPU/memory realistic usage
- [ ] Job logs reviewed:
  ```bash
  cat logs/train_<job_id>.out    # Training metrics, model info
  cat logs/train_<job_id>.err    # Any warnings or errors
  ```
- [ ] Model checkpoint saved (verify in training logs)
- [ ] **Important:** Checkpoint backed up to GitHub Releases or Google Drive
  (Scratch storage is not permanent)

---

## Team Verification & Documentation

After all members are set up:

- [ ] All 4 members have completed "Individual One-Time Setup"
- [ ] At least one `environment.yml` and `requirements.lock` generated and in repo
- [ ] `docs/zaratan_notes.md` updated with:
  - [ ] Account name filled in
  - [ ] Team member names & UIDs filled in
  - [ ] Last verified date & person filled in
- [ ] Quick test job submitted & completed by at least 2 team members
- [ ] `seff <test_job_id>` output captured and documented (attach to GitHub issue)

---

## Definition of "Done"

✅ Any team member can run:

```bash
ssh <uid>@zaratan.umd.edu
cd /scratch/$USER/airsketch
conda activate airsketch
sbatch scripts/slurm/train.sh
squeue -u $USER   # Job appears with state R (running) or PD (pending)
```

Without asking for help, and the job runs to completion without environment errors.

---

## Documentation References

- **Setup Guide:** `docs/GETTING_STARTED_ZARATAN.md`
- **Full Reference:** `docs/zaratan_notes.md`
- **SLURM Scripts:** `scripts/README.md`
- **Support:** hpcsupport@umd.edu
