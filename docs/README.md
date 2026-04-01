# Zaratan Documentation Index

This directory contains all information needed to set up and run AirSketch training on the Zaratan HPC cluster at University of Maryland.

## Start Here

**New to Zaratan?** Follow this order:

1. **[GETTING_STARTED_ZARATAN.md](GETTING_STARTED_ZARATAN.md)** ← Read this first
   - One-time setup walkthrough (15–20 minutes)
   - Step-by-step instructions for your first job
   - Prerequisites checklist

2. **[ZARATAN_CHECKLIST.md](ZARATAN_CHECKLIST.md)** ← Use while setting up
   - Detailed verification checklist for each step
   - Team-level prerequisites
   - Definition of "done"

3. **[zaratan_notes.md](zaratan_notes.md)** ← Reference during work
   - Comprehensive reference material
   - Cluster architecture & specs
   - All SLURM commands with examples
   - Storage guidelines
   - Module system details

4. **[ZARATAN_TROUBLESHOOTING.md](ZARATAN_TROUBLESHOOTING.md)** ← Use if something breaks
   - FAQ for common issues
   - Debugging strategies
   - Solutions organized by problem type

5. **[../scripts/README.md](../scripts/README.md)** ← Reference for job scripts
   - How to use each SLURM script
   - Resource requests & timeouts
   - Job template

---

## Document Purposes

| Document | Purpose | When to Use |
|----------|---------|------------|
| **GETTING_STARTED_ZARATAN.md** | Step-by-step setup guide | First time on Zaratan (read top to bottom) |
| **ZARATAN_CHECKLIST.md** | Verification checklist | Check off items during setup, before first job |
| **zaratan_notes.md** | Comprehensive reference | Look up commands, cluster specs, troubleshooting |
| **ZARATAN_TROUBLESHOOTING.md** | FAQ & debugging | Something went wrong, look up the error |
| **../scripts/README.md** | SLURM script guide | Questions about train.sh, sweep.sh, preprocess.sh |

---

## Key Files In This Directory

```
docs/
├── GETTING_STARTED_ZARATAN.md        ← Read first
├── ZARATAN_CHECKLIST.md              ← Use during setup
├── zaratan_notes.md                  ← Reference & commands
├── ZARATAN_TROUBLESHOOTING.md        ← Debug problems
└── README.md (this file)
```

---

## Quick Links

### Common Tasks

**I want to...**
- **Set up my Zaratan account the first time** ➜ [GETTING_STARTED_ZARATAN.md](GETTING_STARTED_ZARATAN.md)
- **Submit my first training job** ➜ [GETTING_STARTED_ZARATAN.md](GETTING_STARTED_ZARATAN.md#your-first-training-job) then [../scripts/README.md](../scripts/README.md)
- **Monitor a running job** ➜ [zaratan_notes.md](zaratan_notes.md#job-submission--monitoring)
- **Debug a job that failed** ➜ [ZARATAN_TROUBLESHOOTING.md](ZARATAN_TROUBLESHOOTING.md)
- **Understand SLURM scripts** ➜ [../scripts/README.md](../scripts/README.md)
- **Learn cluster architecture** ➜ [zaratan_notes.md](zaratan_notes.md#cluster-architecture--gpu-specifications)
- **Fix CUDA/GPU issues** ➜ [ZARATAN_TROUBLESHOOTING.md](ZARATAN_TROUBLESHOOTING.md#module--environment-issues)
- **Check my storage quota** ➜ [zaratan_notes.md](zaratan_notes.md#storage-guidelines)

---

## Important Information

### Account Name & Contact

Before starting, you need:
- **Account name:** (from your PI, used in all SLURM scripts)
- **Your Zaratan UID:** (used in `ssh <uid>@zaratan.umd.edu`)
- **Support email:** hpcsupport@umd.edu (for technical issues)

See [zaratan_notes.md](zaratan_notes.md#team-notes) to fill in team details.

### Critical Paths

| Path | Use | Notes |
|------|-----|-------|
| `/home/<uid>/` | Code, configs | Backed up, slow I/O (20 GB quota) |
| `/scratch/$USER/airsketch/` | Clone repo here | Not backed up, fast I/O, (1 TB quota) |
| `/scratch/zt1/<group>/` | Shared data | Ask PI for access |

**Rule: Always train from `/scratch/$USER/airsketch/`, not from home dir**

### SLURM Scripts

All jobs use templates in `../scripts/slurm/`:
- `train.sh` — Single training run
- `sweep.sh` — Hyperparameter sweep (9 parallel jobs)
- `preprocess.sh` — Data preprocessing (CPU only)

[See ../scripts/README.md for details](../scripts/README.md)

---

## Hardware Specs

- **GPU Nodes:** NVIDIA A100 40GB (4 GPUs per node)
- **Typical Request:** 1 GPU + 8 CPU cores + 32 GB RAM
- **Training Job Timeout:** 4–6 hours (adjust as needed)
- **Partition:** `gpu` (standard for model training)

---

## Checklist: Ready to Submit Your First Job?

- [ ] SSH access verified (`ssh <uid>@zaratan.umd.edu` works)
- [ ] Conda environment created and working
- [ ] `pytorch` GPU support confirmed (`torch.cuda.is_available()` returns `True`)
- [ ] `environment.yml` and `requirements.lock` committed to repo
- [ ] SLURM scripts updated with account name and your email
- [ ] Config file created or verified (`configs/default.yaml`)
- [ ] Training code tested locally or in interactive session
- [ ] First job submitted (`sbatch scripts/slurm/train.sh`)

---

## Support & Resources

- **Zaratan Official Docs:** https://zaratan.documentation.umd.edu/
- **SLURM Quick Reference:** https://slurm.schedmd.com/pdfs/summary.pdf
- **PyTorch CUDA Setup:** https://pytorch.org/get-started/locally/
- **This Project's GitHub:** [INSERT REPO URL]
- **Zaratan Support Email:** hpcsupport@umd.edu

---

## FAQ: Which document should I read?

**Q: I've never used Zaratan before. Where do I start?**
A: Read [GETTING_STARTED_ZARATAN.md](GETTING_STARTED_ZARATAN.md) top to bottom.

**Q: My job failed. What's wrong?**
A: Check [ZARATAN_TROUBLESHOOTING.md](ZARATAN_TROUBLESHOOTING.md) for your error message.

**Q: I want to understand the cluster and all available commands.**
A: See [zaratan_notes.md](zaratan_notes.md) — this is the complete reference.

**Q: How do I submit a hyperparameter sweep?**
A: Use `sbatch scripts/slurm/sweep.sh` — details in [../scripts/README.md](../scripts/README.md#sweepsh--hyperparameter-sweep).

**Q: How much time should I request for my job?**
A: See [GETTING_STARTED_ZARATAN.md](GETTING_STARTED_ZARATAN.md#job-execution) or estimate & use `seff` to adjust next time.

**Q: I keep hitting "Out of Memory". What do I do?**
A: See [ZARATAN_TROUBLESHOOTING.md](ZARATAN_TROUBLESHOOTING.md#out-of-memory-or-cuda-out-of-memory).

---

## Team Coordination

- **Account name:** [TO BE FILLED IN BY PI]
- **Team members set up:** [_] [_] [_] [_]
- **Environment lockfiles committed:** [DATE]
- **First successful job:** Job ID _______ by _______ on [DATE]

---

## Last Updated

- **Date:** March 2026
- **By:** [TO BE FILLED IN]
- **Status:** Ready for team

---

**Need help?** Start with [GETTING_STARTED_ZARATAN.md](GETTING_STARTED_ZARATAN.md) or email hpcsupport@umd.edu
