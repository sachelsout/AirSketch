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
  - GPU type: NVIDIA A100 40GB, H100, or V100 (depends on node allocated)

---

## Individual One-Time Setup

Each team member completes this once. First person should generate `environment.yml` and `requirements.lock`.

### Prerequisites
- [ ] Zaratan account active (can SSH)
- [ ] Know the account name from PI

### SSH & Module Setup
- [ ] SSH into Zaratan: `ssh <uid>@login.zaratan.umd.edu`
- [ ] unset PYTHONPATH added to ~/.bashrc before module loads
- [ ] Modules added to `~/.bashrc`:
  ```bash
    module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
    module load cuda/12.3.0/gcc/11.3.0/zen2
    module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
  ```
- [ ] CUDA verified: `nvcc --version` shows 12.3

### Venv Setup
- [ ] Venv created: python -m venv $HOME/envs/airsketch
- [ ] Venv activatable: source $HOME/envs/airsketch/bin/activate
- [ ] Python path correct: which python shows ~/envs/airsketch/bin/python
- [ ] Python version correct: python --version shows 3.10.10

### Repository & Dependencies
- [ ] Repo cloned to `$HOME/AirSketch/` (Note: /scratch/$USER may not be provisioned. Contact hpcsupport@umd.edu to request it.)
- [ ] Dependencies installed: `pip install -r requirements.txt` ✓
- [ ] PyTorch installed with CUDA: `pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121` ✓
- [ ] All imports working: `python -c "import torch; import cv2; import mediapipe"`

### First-Time: Generate Lockfiles
(Only first person does this; generates `requirements.lock`)

- [ ] Lockfiles generated:
  ```bash
  pip freeze > requirements.lock
  ```
- [ ] Lockfiles committed to repo:
  ```bash
  git add requirements.lock
  git commit -m "chore: add Zaratan pip environment lockfiles"
  git push
  ```

### Subsequent Team Members: Use Lockfiles
- [ ] Lockfiles copied from repo (via `git pull`)
- [ ] Venv created:
  ```bash
  python -m venv $HOME/envs/airsketch
  ```
- [ ] Dependencies installed from lockfile:
  ```bash
  pip install -r requirements.lock
  ```

### GPU Verification (Optional but Recommended)
- [ ] Interactive GPU session requested:
  ```bash
  salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 \
           --mem=32G --time=00:10:00 --account=msml612-class
  ```
- [ ] GPU detected from compute node:
  ```bash
  source $HOME/envs/airsketch/bin/activate
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
  ```
  Should print: `True` and `NVIDIA A100-SXM4-40GB (or H100/V100)`
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
- [ ] SSH into Zaratan: `ssh <uid>@login.zaratan.umd.edu`
- [ ] Navigate to repo: `cd $HOME/AirSketch`
- [ ] Request interactive GPU:
  ```bash
  salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 \
           --mem=32G --time=00:30:00 --account=msml612-class
  ```
- [ ] Test training code:
  ```bash
  source $HOME/envs/airsketch/bin/activate
  python src/train.py --config configs/default.yaml --debug
  ```
- [ ] Code runs without errors (check for 1–2 iterations, then Ctrl+C)
- [ ] GPU is utilized: `nvidia-smi` during training shows GPU usage
- [ ] Exit session: `exit`

### Submit First Test Job
- [ ] Config file verified: `configs/default.yaml` exists and looks correct
- [ ] Submit job:
  ```bash
  cd $HOME/AirSketch
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
ssh <uid>@login.zaratan.umd.edu
cd $HOME/AirSketch
source $HOME/envs/airsketch/bin/activate
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
