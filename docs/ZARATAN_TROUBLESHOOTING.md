# Zaratan FAQ & Troubleshooting

Common issues and solutions for running AirSketch on the Zaratan cluster.

## Account & Access

### "Permission denied (publickey)"
**Problem:** Can't SSH to Zaratan

**Solution:**
1. Verify you have a Zaratan account: check email for account confirmation
2. If not, request at: hpcsupport@umd.edu (2–3 business days)
3. Try verbose SSH to debug: `ssh -vvv <uid>@zaratan.umd.edu`
4. Verify your `<uid>` is correct (check account confirmation email)
5. If using SSH key: `ssh-keygen -t rsa -b 4096` then follow Zaratan docs to upload key

### "Connection refused" or "Connection timed out"
**Solution:**
- Verify hostname is correct: `zaratan.umd.edu` (not `zaratan.cs.umd.edu`)
- Check network: `ping zaratan.umd.edu`
- Try SSH with explicit port: `ssh -p 22 <uid>@zaratan.umd.edu`

---

## Module & Environment Issues

### "ModuleNotFoundError: No module named 'torch'"
**Problem:** PyTorch not installed or not found by Python

**Solution:**
1. Verify venv is active: `source $HOME/envs/airsketch/bin/activate && which python`
2. Check PyTorch is installed: `pip show torch`
3. If missing, reinstall: `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121`
4. Reactivate venv: `deactivate && source $HOME/envs/airsketch/bin/activate`

### "nvcc: command not found" or CUDA not detected
**Problem:** CUDA modules not loaded

**Solution:**
1. Load CUDA modules: 
   ```bash
   unset PYTHONPATH
   module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
   module load cuda/12.3.0/gcc/11.3.0/zen2
   module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2```
2. Verify: `nvcc --version` should show CUDA 12.3
3. Check ~/.bashrc has unset PYTHONPATH and all three module load lines
4. Reload shell: `source ~/.bashrc && module list`

### "torch.cuda.is_available() returns False"
**Problem:** CUDA is installed but PyTorch can't detect GPUs

**Causes & Solutions:**
1. **Not on a compute node:** `nvidia-smi` only works on compute nodes, not login node
   - Solution: Use `salloc` to get interactive GPU session first
2. **PyTorch built for different CUDA:** Version mismatch
   - Solution: `pip install --force-reinstall torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121`
3. **Conda environment not activated properly:**
   - Solution: Reactivate venv: deactivate && source $HOME/envs/airsketch/bin/activate && python -c "import torch; print(torch.cuda.is_available())"

### "Error: Could not find cudnn header"
**Problem:** cuDNN not loaded or not found

**Solution:**
1. Load cuDNN module: `module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2`
2. Verify: `echo $CUDNN_HOME` should show a path
3. Add to ~/.bashrc if missing

---

## Job Submission Issues

### "sbatch: command not found"
**Problem:** SLURM tools not available

**Solution:**
- SLURM should always be available on Zaratan. Try:
  ```bash
  module purge
  which sbatch
  ```
- If still missing, contact: hpcsupport@umd.edu

### "sbatch: error: Invalid account specified"
**Problem:** `--account=<account>` is wrong or missing

**Solution:**
1. Check SLURM script: `grep "account=" scripts/slurm/train.sh`
2. Verify account name with PI (check `docs/zaratan_notes.md`)
3. Update all scripts: `<account>` should be the actual account, not literally `<account>`
4. Resubmit: `sbatch scripts/slurm/train.sh`

### "sbatch: fatal: can't find a suitable node"
**Problem:** Requested resources not available

**Solution:**
1. Check GPU availability: `sinfo -p gpu`
2. Check queue: `squeue -p gpu | head -20`
3. Options:
   - Reduce resources (fewer CPUs, less RAM, shorter time)
   - Try `--time=00:30:00` for quick test first
   - Increase `--time` (longer jobs have lower priority but are more likely to start)

### "Submitted batch job 12345" but job never runs
**Problem:** Job stuck in queue (state=PD)

**Solution:**
1. Check queue status: `squeue -j 12345` (see REASON column)
2. Common reasons & fixes:
   - **Priority**: Your job has lower priority. Wait or reduce resource request to make jobs smaller
   - **Resources**: GPUs unavailable. Check `squeue -p gpu` to see current usage
   - **Dependency**: Job depends on another job. Check with `squeue -j 12345 -l` for details
3. Cancel and resubmit if needed: `scancel 12345 && sbatch scripts/slurm/train.sh`

---

## Job Execution Issues

### Job dies immediately with no output
**Problem:** Job fails before training starts

**Solution:**
1. Check error log: `cat logs/train_<job_id>.err`
2. Common causes:
   - Python not found: modules not loaded or venv not activated
   - Config file missing: `cd ~/scratch/AirSketch` didn't work or directory is wrong
   -  Venv doesn't exist: run python -m venv $HOME/envs/airsketch first
3. Debug with interactive session first:
   ```bash
   salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:30:00 --account=msml612-class
   unset PYTHONPATH
   module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
   module load cuda/12.3.0/gcc/11.3.0/zen2
   module load cudnn/8.9.7.29-12/gcc/11.3.0/zen2
   source $HOME/envs/airsketch/bin/activate
   cd ~/scratch/AirSketch
   python src/train.py --config configs/default.yaml
   ```

### "Out of memory" or "CUDA out of memory"
**Problem:** GPU ran out of VRAM

**Solution:**
1. Check what you requested: `seff <job_id>` (look at %CPU and GPU memory columns)
2. Options:
   - Reduce batch size in config: `configs/default.yaml` → `batch_size: 32` (was 64?)
   - Reduce model size: `hidden_dim: 128` (was 256?)
   - Request more time (may allow smaller batch sizes)
3. Resubmit with updated config:
   ```bash
   sbatch scripts/slurm/train.sh configs/default.yaml
   ```

### Job killed after 4 hours (or `--time` value)
**Problem:** Job hit time limit

**Solution:**
1. Check actual runtime: `seff <job_id>` (look at "Elapsed" time)
2. If job was still training:
   - Increase `--time` in SLURM script
   - Try: `--time=06:00:00` instead of `--time=04:00:00`
   - Longer jobs have lower priority but may be needed for large models
3. For sweep jobs: `--time=06:00:00` in sweep.sh

---

## Data & Storage Issues

```
Scratch is available at /scratch/zt1/project/msml612/user/<uid>/ and exposed in home as ~/scratch/.
```

### "data/raw/: No such file or directory"
**Problem:** Raw data not found at expected path

**Solution:**
1. Verify data location: `ls ~/scratch/AirSketch/data/raw/`
2. If empty, data needs to be uploaded to Zaratan:
   ```bash
   # From your laptop/local machine:
   scp -r data/raw/* <uid>@zaratan.umd.edu:/scratch/zt1/project/msml612/user/<uid>/AirSketch/data/raw/
   ```
3. Or download on Zaratan:
   ```bash
   cd ~/scratch/AirSketch/data/raw/
   wget <data-url>
   ```

### "Disk quota exceeded"
**Problem:** Scratch storage full

**Solution:**
1. Check usage: `quota`
2. Find large files: `du -sh ~/scratch/AirSketch/* | sort -h`
3. Clean up:
   - Delete old checkpoints: `rm checkpoints/old_*.pt` (keep recent ones!)
   - Delete old logs: `rm logs/*.out logs/*.err`
   - **Back up important checkpoints** before deleting
4. If still over quota:
   - Contact PI about using group storage: `/scratch/zt1/<group>/`

### "Scratch data disappeared" or "I/O error"
**Problem:** Scratch files were wiped (periodic cleanup) or filesystem issue

**Solution:**
1. **Always back up important checkpoints:**
   ```bash
   # Copy to GitHub Releases or Google Drive
   gsutil cp checkpoints/trained_model.pt gs://your-bucket/
   # or
   aws s3 cp checkpoints/trained_model.pt s3://your-bucket/
   # or manually download: scp <uid>@zaratan.umd.edu:...</path>
   ```
2. Don't rely on Zaratan scratch for long-term storage
3. For persistent data, use `/home/$USER/` (backed up, slower I/O) for small files

---

## Monitoring & Debugging

### "How do I see my job output in real-time?"
```bash
tail -f logs/train_<job_id>.out
```

Press Ctrl+C to stop following (job keeps running).

### "How do I know why my job failed?"
```bash
cat logs/train_<job_id>.err    # Error messages
cat logs/train_<job_id>.out    # Full output
seff <job_id>                   # Resource usage & efficiency
```

### "Can I run multiple jobs at the same time?"
Yes! SLURM will queue them. Options:
1. Submit multiple separate jobs: `sbatch train.sh` (multiple times)
2. Use array job: `sbatch sweep.sh` (runs 9 jobs in parallel)
3. Check queue: `squeue -u $USER` to see all submitted jobs

### "How do I stop a running job?"
```bash
scancel <job_id>

# Or cancel all your jobs:
scancel -u $USER
```

---

## General Tips

1. **Always test with interactive session first** before submitting batch jobs:
   ```bash
   salloc --partition=gpu --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:30:00 --account=msml612-class
   ```

2. **Save resource requests for future reference:** After job completes, run `seff <job_id>` and note:
   - Actual CPU usage (helps tune `--cpus-per-task`)
   - Actual memory usage (helps tune `--mem`)
   - GPU utilization (if <50%, model might be bottlenecked by CPU)

3. **Use meaningful job names:** Makes monitoring easier
   ```bash
   #SBATCH --job-name=airsketch-exp-v2
   ```

4. **Enable email notifications:** So you don't have to keep checking
   ```bash
   #SBATCH --mail-type=END,FAIL
   #SBATCH --mail-user=you@umd.edu
   ```

5. **Commit code before submitting:** If job fails, you want to know what code ran
   ```bash
   git status
   git add -A
   git commit -m "exp: test config v2"
   git push
   sbatch scripts/slurm/train.sh
   ```

---

## Still Stuck?

1. Check `docs/zaratan_notes.md` for reference
2. Check official Zaratan docs: https://zaratan.documentation.umd.edu/
3. Email support: hpcsupport@umd.edu
4. Ask teammates in Slack/Discord

Provide when asking for help:
- Job ID: `<job_id>`
- Full error message from `cat logs/train_<job_id>.err`
- `seff <job_id>` output
- What config/command was run
- Any recent changes to code or config
