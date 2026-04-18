"""
src/train.py

Training loop for the AirSketch ST-Transformer.

Trains the model with a joint loss (L1 regression + weighted cross-entropy
gesture classification), AdamW optimizer, and cosine learning rate schedule.
Logs all metrics to Weights & Biases per epoch. Checkpoints the best model
by validation MPJPE.

Usage:
    # Standard run with default config
    python src/train.py --config configs/default.yaml

    # Override specific hyperparameters (used by W&B sweep in issue #11)
    python src/train.py --config configs/default.yaml \
        --override model.hidden_dim=256 \
        --override model.num_layers=6 \
        --run-name sweep_h256_l6

    # Debug run -- small dataset subset, no W&B logging
    python src/train.py --config configs/default.yaml --debug

    # Resume from a checkpoint
    python src/train.py --config configs/default.yaml \
        --resume checkpoints/best_model.pt
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.dataset import build_dataloaders_merged
from src.evaluate import compute_all_metrics
from src.logger import ExperimentLogger
from src.model import AirSketchLoss, build_model


# -- Reproducibility -----------------------------------------------------------


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility across restarts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -- Config helpers ------------------------------------------------------------


def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    """
    Load YAML config and apply any --override key=value arguments.

    Override format: "section.key=value"
    Example: "model.hidden_dim=256" sets config["model"]["hidden_dim"] = 256

    Numeric values are auto-cast: "0.001" -> float, "128" -> int.
    Booleans are cast: "true" -> True.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for override in overrides or []:
        key_path, _, raw_value = override.partition("=")
        parts = key_path.strip().split(".")

        value: int | float | bool | str
        if raw_value.lower() == "true":
            value = True
        elif raw_value.lower() == "false":
            value = False
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value

        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
        print(f"  Override: {key_path} = {value} ({type(value).__name__})")

    return config


# -- Checkpoint helpers --------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_mpjpe: float,
    config: dict,
) -> None:
    """Save full training state to disk for resumption or evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_mpjpe": best_mpjpe,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: str = "cpu",
) -> tuple[int, float]:
    """
    Load training state from a checkpoint file.

    Returns:
        (start_epoch, best_mpjpe) -- training resumes from start_epoch + 1.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_mpjpe = checkpoint.get("best_mpjpe", float("inf"))

    print(f"  Resumed from epoch {start_epoch}  |  best val MPJPE: {best_mpjpe:.4f} px")
    return start_epoch, best_mpjpe


# -- One epoch -----------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: AirSketchLoss,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    logger: ExperimentLogger | None,
    epoch: int,
    is_train: bool,
    log_every: int = 10,
) -> dict[str, float]:
    """
    Run one full pass (train or eval) over a DataLoader.

    During training: computes gradients, clips them, steps the optimizer.
    During eval: runs under torch.no_grad() for speed and memory efficiency.

    Accumulates predictions and ground-truth tensors across all batches,
    then computes MPJPE, jitter index, and gesture accuracy at the end of
    the epoch over the full split — not a running average of per-batch values,
    which would be biased when the final batch is smaller than the rest.

    Returns:
        Dict of metrics: loss, regression_loss, gesture_loss,
                         mpjpe, jitter_index, gesture_acc.
    """
    model.train() if is_train else model.eval()
    context = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    total_reg_loss = 0.0
    total_ges_loss = 0.0
    n_batches = 0

    all_pred_xy = []
    all_gt_xy = []
    all_pred_labels = []
    all_gt_labels = []

    global_step = (epoch - 1) * len(loader)

    with context:
        for batch_idx, batch in enumerate(loader):
            sequence = batch["sequence"].to(device)
            gt_xy = batch["target"].to(device)
            gesture_labels = batch["gesture"].to(device)

            out = model(sequence)

            losses = loss_fn(
                pred_xy=out["pred_xy"],
                gt_xy=gt_xy,
                gesture_logits=out["gesture_logits"],
                gesture_labels=gesture_labels,
            )

            if is_train:
                optimizer.zero_grad()
                losses["loss"].backward()

                # Gradient clipping prevents exploding gradients while
                # the CLS token is settling in early training epochs
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if logger and batch_idx % log_every == 0:
                    logger.log_step(
                        step=global_step + batch_idx,
                        loss=losses["loss"].item(),
                        gesture_loss=losses["gesture_loss"].item(),
                        regression_loss=losses["regression_loss"].item(),
                    )

            total_loss += losses["loss"].item()
            total_reg_loss += losses["regression_loss"].item()
            total_ges_loss += losses["gesture_loss"].item()
            n_batches += 1

            all_pred_xy.append(out["pred_xy"].detach().cpu())
            all_gt_xy.append(gt_xy.detach().cpu())

            pred_labels = out["gesture_logits"].argmax(dim=1).detach().cpu()
            all_pred_labels.append(pred_labels)
            all_gt_labels.append(gesture_labels.detach().cpu())

    all_pred_xy_np = torch.cat(all_pred_xy, dim=0).numpy()
    all_gt_xy_np = torch.cat(all_gt_xy, dim=0).numpy()
    all_pred_lbl_np = torch.cat(all_pred_labels, dim=0).numpy()
    all_gt_lbl_np = torch.cat(all_gt_labels, dim=0).numpy()

    # Scale normalized [0,1] coords to pixel space for MPJPE in px
    # (model trains and evaluates at 224px resolution)
    metrics = compute_all_metrics(
        pred_xy=all_pred_xy_np * 224,
        gt_xy=all_gt_xy_np * 224,
        pred_labels=all_pred_lbl_np,
        gt_labels=all_gt_lbl_np,
    )

    metrics["loss"] = total_loss / n_batches
    metrics["regression_loss"] = total_reg_loss / n_batches
    metrics["gesture_loss"] = total_ges_loss / n_batches

    return metrics


# -- Main training function ----------------------------------------------------


def train(
    config_path: str,
    overrides: list[str] | None = None,
    run_name: str | None = None,
    resume_path: str | None = None,
    debug: bool = False,
) -> float:
    """
    Full training run: data -> model -> train loop -> checkpoint.

    Returns:
        Best validation MPJPE achieved (used as the W&B sweep objective).
    """
    print("Loading config ...")
    config = load_config(config_path, overrides)

    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    print(f"  Seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    print("\nBuilding DataLoaders ...")
    train_loader, val_loader = build_dataloaders_merged(config_path, config=config)

    # Retrieve inverse-frequency class weights from training dataset.
    # Passed to AirSketchLoss to compensate for FreiHAND's ~80% idle skew.
    if hasattr(train_loader.dataset, "get_class_weights"):
        class_weights = train_loader.dataset.get_class_weights().to(device)
    elif hasattr(train_loader.dataset, "datasets"):
        # ConcatDataset — get weights from the first sub-dataset (FreiHAND)
        class_weights = train_loader.dataset.datasets[0].get_class_weights().to(device)
    else:
        class_weights = None

    if debug:
        from itertools import islice

        train_loader = list(islice(train_loader, 2))
        val_loader = list(islice(val_loader, 2))
        print("  [debug] Using 2 batches per split.")

    print("\nBuilding model ...")
    model = build_model(config, device=device)

    gesture_lam = config.get("model", {}).get("gesture_loss_weight", 0.5)
    loss_fn = AirSketchLoss(
        gesture_loss_weight=gesture_lam,
        class_weights=class_weights,
    )
    print(f"  gesture_loss_weight (lambda): {gesture_lam}")
    if class_weights is not None:
        print(f"  gesture class weights:        {class_weights.tolist()}")

    # AdamW with weight decay excluded from bias and LayerNorm parameters.
    # Applying weight decay to these is incorrect (Loshchilov & Hutter 2019)
    # and degrades convergence.
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": config.get("training", {}).get("weight_decay", 0.01),
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    lr = config.get("training", {}).get("learning_rate", 1e-3)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    print(
        f"\n  Optimizer: AdamW  lr={lr}  "
        f"wd={config.get('training', {}).get('weight_decay', 0.01)}"
    )

    # Cosine annealing from initial LR to eta_min over the full run.
    # No warmup needed -- pre-norm residuals keep training stable from epoch 1.
    epochs = config.get("training", {}).get("epochs", 50)
    min_lr = lr / 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_lr,
    )
    print(f"  Scheduler: CosineAnnealingLR  T_max={epochs}  eta_min={min_lr:.2e}")

    ckpt_dir = Path(config.get("training", {}).get("checkpoint_dir", "checkpoints"))
    best_ckpt = ckpt_dir / "best_model.pt"
    latest_ckpt = ckpt_dir / "latest_model.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_mpjpe = float("inf")

    if resume_path:
        print(f"\nResuming from: {resume_path}")
        start_epoch, best_mpjpe = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, device
        )

    logger = None
    if not debug:
        tags = ["training"]
        if run_name and "sweep" in run_name.lower():
            tags.append("sweep")
        logger = ExperimentLogger(config=config, run_name=run_name, tags=tags)

    print(f"\n{'--'*30}")
    print(f"  Training: {epochs} epochs  |  {'debug' if debug else 'full run'}")
    print(f"{'--'*30}\n")

    log_every = config.get("logging", {}).get("log_every_n_steps", 10)

    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            logger=logger,
            epoch=epoch,
            is_train=True,
            log_every=log_every,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            optimizer=None,
            device=device,
            logger=None,
            epoch=epoch,
            is_train=False,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if logger:
            logger.log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
            )

        val_mpjpe = val_metrics["mpjpe"]
        is_best = val_mpjpe < best_mpjpe

        if is_best:
            best_mpjpe = val_mpjpe
            save_checkpoint(
                best_ckpt, model, optimizer, scheduler, epoch, best_mpjpe, config
            )
            if logger:
                logger.log_model(best_ckpt, metric_value=best_mpjpe)
            marker = "  <- best"
        else:
            marker = ""

        # Always save latest for resumption
        save_checkpoint(
            latest_ckpt, model, optimizer, scheduler, epoch, best_mpjpe, config
        )

        print(
            f"  [{epoch:3d}/{epochs}]  "
            f"loss {train_metrics['loss']:.4f} -> {val_metrics['loss']:.4f}  |  "
            f"MPJPE {train_metrics['mpjpe']:.2f} -> {val_metrics['mpjpe']:.2f} px  |  "
            f"gesture {train_metrics['gesture_acc']*100:.1f} -> "
            f"{val_metrics['gesture_acc']*100:.1f}%  |  "
            f"jitter {val_metrics['jitter_index']:.3f} px2  |  "
            f"lr {current_lr:.2e}  |  {time.time()-t0:.1f}s"
            f"{marker}"
        )

        if _targets_met(val_metrics) and not debug:
            print(f"\n  All metric targets met at epoch {epoch}. Stopping early.")
            break

    print(f"\n{'--'*30}")
    print("  Training complete.")
    print(f"  Best val MPJPE:  {best_mpjpe:.4f} px  (target: < 8 px)")
    print(f"  Best checkpoint: {best_ckpt}")
    print(f"{'--'*30}\n")

    if logger:
        logger.log_summary({"best_val_mpjpe": best_mpjpe})
        logger.finish()

    return best_mpjpe


# -- Target checker ------------------------------------------------------------


def _targets_met(val_metrics: dict) -> bool:
    """
    Return True if all three primary metric targets from the proposal are met.

    Targets:
        MPJPE        < 8.0 px
        jitter_index < 2.0 px2
        gesture_acc  > 0.92 (strict inequality)
    """
    return (
        val_metrics.get("mpjpe", float("inf")) < 8.0
        and val_metrics.get("jitter_index", float("inf")) < 2.0
        and val_metrics.get("gesture_acc", 0.0) > 0.92
    )


# -- CLI -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the AirSketch ST-Transformer.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override config. Repeatable. E.g. --override model.hidden_dim=256",
    )
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    best = train(
        config_path=args.config,
        overrides=args.overrides,
        run_name=args.run_name,
        resume_path=args.resume,
        debug=args.debug,
    )
    sys.exit(0 if best < 8.0 else 1)
