"""
src/dataset.py

PyTorch Dataset for AirSketch ST-Transformer training.

Each sample is a sliding window of T=16 consecutive landmark frames
(shape: 16 x 42) and the regression target is the index fingertip
(x, y) at frame T+1 (shape: 2).

A parallel binary gesture label (draw=1 / idle=0) is derived from
fingertip velocity and returned alongside each sample for the gesture
classification head.

Augmentations applied during training:
  - Random horizontal flip (p=0.5, applied consistently across all frames)
  - Gaussian noise on landmark coordinates (sigma=0.005)

Usage:
    from src.dataset import AirSketchDataset, build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders("configs/default.yaml")
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import get_valid_sequence_ranges, interpolate_missing_landmarks


# ── Constants ──────────────────────────────────────────────────────────────────

WINDOW_SIZE = 16  # T — number of input frames per sample
NUM_LANDMARKS = 21
FEATURE_DIM = 42  # 21 landmarks x 2 (x, y), flattened
INDEX_FINGERTIP = 8  # MediaPipe / FreiHAND landmark index for index fingertip

# Velocity threshold (normalized coords/frame) separating draw from idle.
# Tuned empirically: slow deliberate strokes are ~0.008/frame,
# idle hand tremor is typically < 0.003/frame.
DRAW_VELOCITY_THRESHOLD = 0.005


# ── Dataset ────────────────────────────────────────────────────────────────────


class AirSketchDataset(Dataset):
    """
    Sliding-window landmark sequence dataset for AirSketch.

    Args:
        landmarks_path: Path to landmarks.npy — shape (N, 21, 2) float32.
                        NaN entries mark frames where MediaPipe detection failed.
        split_indices:  List of frame indices belonging to this split.
                        Must be pre-filtered to only include frames from
                        the correct train/val/test split (issue #4).
        window_size:    Number of input frames per sample (default: 16).
        stride:         Step size between consecutive windows (default: 1).
                        Use stride > 1 to reduce dataset size during debugging.
        augment:        Whether to apply data augmentation. Should be True
                        only for the training split.
        noise_sigma:    Standard deviation of Gaussian noise added to landmark
                        coordinates during augmentation (default: 0.005).
        flip_prob:      Probability of random horizontal flip per sample (default: 0.5).
        max_gap:        Maximum consecutive NaN frames to interpolate over
                        before treating a gap as a sequence break (default: 2).
    """

    def __init__(
        self,
        landmarks_path: str | Path,
        split_indices: list[int],
        window_size: int = WINDOW_SIZE,
        stride: int = 1,
        augment: bool = False,
        noise_sigma: float = 0.005,
        flip_prob: float = 0.5,
        max_gap: int = 2,
    ):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.flip_prob = flip_prob

        # ── Load and filter landmarks ─────────────────────────────────────────
        all_landmarks = np.load(landmarks_path)  # (N_total, 21, 2)

        # Remap split_indices to a contiguous sub-array.
        # split_indices may be non-contiguous (e.g. every 4th scene in FreiHAND).
        # We extract only the relevant rows so window math stays simple.
        split_indices_sorted = sorted(split_indices)
        self._landmarks_raw = all_landmarks[split_indices_sorted]  # (N_split, 21, 2)
        self._index_map = split_indices_sorted  # original → local index

        # ── Interpolate short gaps, then find valid sequence ranges ───────────
        self._landmarks, nan_mask = interpolate_missing_landmarks(
            self._landmarks_raw, max_gap=max_gap
        )
        valid_ranges = get_valid_sequence_ranges(nan_mask, window_size=window_size + 1)
        # +1 because we need T+1 frames: T for input, 1 for target

        # ── Build the flat window index list ──────────────────────────────────
        # Each entry is the start frame index (local) of one valid window.
        self._windows: list[int] = []
        for range_start, range_end in valid_ranges:
            window_start = range_start
            while window_start + window_size + 1 <= range_end:
                self._windows.append(window_start)
                window_start += stride

        # ── Pre-compute gesture labels ────────────────────────────────────────
        # Gesture label for window starting at frame i is based on the
        # mean fingertip velocity across that window.
        self._gesture_labels = self._compute_gesture_labels()

        print(
            f"AirSketchDataset | split frames: {len(split_indices_sorted):,} | "
            f"valid windows: {len(self._windows):,} | "
            f"augment: {augment}"
        )

    # ── Gesture label computation ──────────────────────────────────────────────

    def _compute_gesture_labels(self) -> np.ndarray:
        """
        Pre-compute a binary draw/idle label for every window.

        A window is labelled draw=1 if the mean frame-to-frame Euclidean
        displacement of the index fingertip across the window exceeds
        DRAW_VELOCITY_THRESHOLD. Otherwise idle=0.

        Returns:
            np.ndarray of shape (num_windows,) with dtype int64.
        """
        labels = np.zeros(len(self._windows), dtype=np.int64)
        tips = self._landmarks[:, INDEX_FINGERTIP, :]  # (N, 2)

        for idx, start in enumerate(self._windows):
            window_tips = tips[start : start + self.window_size]  # (T, 2)
            displacements = np.linalg.norm(np.diff(window_tips, axis=0), axis=1)
            mean_velocity = float(np.mean(displacements))
            labels[idx] = 1 if mean_velocity > DRAW_VELOCITY_THRESHOLD else 0

        return labels

    # ── PyTorch Dataset interface ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return one sample as a dict with keys: 'sequence', 'target', 'gesture'.

        sequence: (T, 42)  float32  — input landmark window, flattened per frame.
        target:   (2,)     float32  — index fingertip (x, y) at frame T+1.
        gesture:  ()       int64    — draw=1 / idle=0.
        """
        start = self._windows[idx]
        window_frames = self._landmarks[start : start + self.window_size]  # (T, 21, 2)
        target_frame = self._landmarks[start + self.window_size]  # (21, 2)
        target_xy = target_frame[INDEX_FINGERTIP].copy()  # (2,)
        gesture_label = self._gesture_labels[idx]

        # ── Augmentation (training only) ───────────────────────────────────────
        if self.augment:
            window_frames, target_xy = self._augment(window_frames, target_xy)

        # ── Flatten (T, 21, 2) → (T, 42) ──────────────────────────────────────
        sequence = window_frames.reshape(self.window_size, FEATURE_DIM)  # (T, 42)

        return {
            "sequence": torch.from_numpy(sequence.copy()),
            "target": torch.from_numpy(target_xy.copy()),
            "gesture": torch.tensor(gesture_label, dtype=torch.int64),
        }

    # ── Augmentation ───────────────────────────────────────────────────────────

    def _augment(
        self,
        window: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations consistently across all T frames and the target.

        Augmentations applied:
          1. Random horizontal flip (p=flip_prob)
             Flips x coordinate: x_new = 1.0 - x
             Applied identically to every frame in the window AND the target.
             The y coordinate and all landmark indices are unchanged.

          2. Gaussian noise (always applied when augment=True)
             Adds independent N(0, noise_sigma) noise to every (x, y) coordinate.
             Applied to window frames only — NOT to the target. The target
             must remain a clean ground-truth coordinate.
             Coordinates are clipped to [0, 1] after noise addition.

        Args:
            window: (T, 21, 2) float32 — landmark window (will be modified in-place copy).
            target: (2,) float32 — fingertip target coordinate.

        Returns:
            Augmented (window, target) as new arrays.
        """
        window = window.copy()
        target = target.copy()

        # ── 1. Random horizontal flip ──────────────────────────────────────────
        if random.random() < self.flip_prob:
            # Flip x coordinate for all landmarks in all frames
            window[:, :, 0] = 1.0 - window[:, :, 0]
            # Flip x coordinate of target fingertip
            target[0] = 1.0 - target[0]

        # ── 2. Gaussian noise on window frames (not target) ────────────────────
        noise = np.random.normal(0.0, self.noise_sigma, size=window.shape).astype(
            np.float32
        )
        window = window + noise
        window = np.clip(window, 0.0, 1.0)

        return window, target

    # ── Utility methods ────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for the gesture head loss.
        Pass these weights to nn.CrossEntropyLoss(weight=...) in the training loop.

        Returns:
            Tensor of shape (2,) with weights for [idle, draw].
        """
        n_total = len(self._gesture_labels)
        n_draw = int(self._gesture_labels.sum())
        n_idle = n_total - n_draw

        if n_draw == 0 or n_idle == 0:
            return torch.ones(2, dtype=torch.float32)

        weight_idle = n_total / (2 * n_idle)
        weight_draw = n_total / (2 * n_draw)
        return torch.tensor([weight_idle, weight_draw], dtype=torch.float32)

    def gesture_distribution(self) -> dict[str, int | float]:
        """Return a summary of draw/idle label counts. Useful for sanity checks."""
        n_draw = int(self._gesture_labels.sum())
        n_idle = len(self._gesture_labels) - n_draw
        return {
            "total": len(self._gesture_labels),
            "draw": n_draw,
            "idle": n_idle,
            "draw_frac": round(n_draw / max(len(self._gesture_labels), 1), 4),
        }


# ── DataLoader factory ─────────────────────────────────────────────────────────


def build_dataloaders(
    config_path: str | Path,
    landmarks_path: str | Path | None = None,
    splits_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders from config + landmark files.

    Args:
        config_path:    Path to configs/default.yaml.
        landmarks_path: Override landmarks file path (default: read from config).
        splits_path:    Override splits JSON path (default: read from config).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    lm_path = (
        Path(landmarks_path or config["data"]["processed_dir"])
        / "freihand"
        / "landmarks.npy"
    )
    sp_path = Path(splits_path or config["data"]["splits_dir"]) / "freihand_splits.json"
    window_size = config["data"]["window_size"]
    stride = config["data"].get("stride", 1)
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)

    with open(sp_path) as f:
        splits = json.load(f)

    train_ds = AirSketchDataset(
        landmarks_path=lm_path,
        split_indices=splits["train"],
        window_size=window_size,
        stride=stride,
        augment=True,
    )
    val_ds = AirSketchDataset(
        landmarks_path=lm_path,
        split_indices=splits["val"],
        window_size=window_size,
        stride=stride,
        augment=False,
    )
    test_ds = AirSketchDataset(
        landmarks_path=lm_path,
        split_indices=splits["test"],
        window_size=window_size,
        stride=stride,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # avoid incomplete final batch during training
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,  # no grad — can afford larger batches
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    _print_dataloader_summary(train_ds, val_ds, test_ds)

    return train_loader, val_loader, test_loader


def build_dataloaders_merged(
    config_path: str | Path,
    merged_split_path: str | Path | None = None,
    config: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders from the merged FreiHAND + EgoHands training split.

    For training: concatenates windows from FreiHAND train + all EgoHands clips.
    For validation: FreiHAND val only (EgoHands has no separate val split).
    For test: custom gesture dataset (loaded separately in issue #16).

    Args:
        config_path:       Path to configs/default.yaml.
        merged_split_path: Path to merged_train_split.json.
                           Defaults to data/splits/merged_train_split.json.

    Returns:
        (train_loader, val_loader, test_loader)
        Note: test_loader here is FreiHAND test only.
        The custom gesture test split is evaluated separately in issue #16.
    """
    import yaml
    from torch.utils.data import ConcatDataset

    if config is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

    merged_path = Path(merged_split_path or "data/splits/merged_train_split.json")
    with open(merged_path) as f:
        merged = json.load(f)

    window_size = config["data"]["window_size"]
    stride = config["data"].get("stride", 1)
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)

    # ── FreiHAND splits ────────────────────────────────────────────────────────
    fh_source = merged["sources"]["freihand"]
    fh_lm_path = fh_source["landmarks_path"]

    freihand_train_ds = AirSketchDataset(
        landmarks_path=fh_lm_path,
        split_indices=fh_source["train_indices"],
        window_size=window_size,
        stride=stride,
        augment=True,
    )
    freihand_val_ds = AirSketchDataset(
        landmarks_path=fh_lm_path,
        split_indices=fh_source["val_indices"],
        window_size=window_size,
        stride=stride,
        augment=False,
    )

    # ── EgoHands clips — stratified 40/8 train/val split ─────────────────────
    VAL_CLIPS = {
        "CARDS_COURTYARD_B_T",
        "CARDS_LIVINGROOM_B_T",
        "CHESS_COURTYARD_B_T",
        "CHESS_LIVINGROOM_B_S",
        "JENGA_COURTYARD_B_H",
        "JENGA_OFFICE_B_S",
        "PUZZLE_COURTYARD_B_S",
        "PUZZLE_LIVINGROOM_B_T",
    }

    egohands_train_datasets = []
    egohands_val_datasets = []

    for clip_name, clip_info in merged["sources"]["egohands"]["clips"].items():
        if clip_info["detection_rate"] < 0.70:
            print(
                f"  Skipping low-detection clip: {clip_name} "
                f"({clip_info['detection_rate']*100:.1f}%)"
            )
            continue
        ds = AirSketchDataset(
            landmarks_path=clip_info["landmarks_path"],
            split_indices=list(range(clip_info["n_frames"])),
            window_size=window_size,
            stride=stride,
            augment=clip_name not in VAL_CLIPS,
        )
        if clip_name in VAL_CLIPS:
            egohands_val_datasets.append(ds)
        else:
            egohands_train_datasets.append(ds)

    # ── Build train dataset ────────────────────────────────────────────────────
    egohands_only = config.get("training", {}).get("egohands_only", False)

    if egohands_only:
        if egohands_train_datasets:
            train_ds = ConcatDataset(egohands_train_datasets)
            eg_windows = sum(len(d) for d in egohands_train_datasets)
            print("  [egohands_only] FreiHAND excluded.")
            print(
                f"  EgoHands train clips: {len(egohands_train_datasets)} ({eg_windows:,} windows)"
            )
        else:
            raise RuntimeError("egohands_only=true but no EgoHands train clips loaded.")
    elif egohands_train_datasets:
        train_ds = ConcatDataset([freihand_train_ds] + egohands_train_datasets)
        eg_windows = sum(len(d) for d in egohands_train_datasets)
        print(
            f"  EgoHands train clips: {len(egohands_train_datasets)} ({eg_windows:,} windows)"
        )
    else:
        train_ds = freihand_train_ds
        print("  WARNING: No EgoHands clips loaded — using FreiHAND only.")

    # ── Build val dataset ──────────────────────────────────────────────────────
    if egohands_val_datasets:
        val_ds = ConcatDataset(egohands_val_datasets)
        val_windows = sum(len(d) for d in egohands_val_datasets)
        print(
            f"  EgoHands val clips:   {len(egohands_val_datasets)} ({val_windows:,} windows)"
        )
    else:
        val_ds = freihand_val_ds
        print("  WARNING: No EgoHands val clips — falling back to FreiHAND val.")

    print(f"  Total train windows: {len(train_ds):,}")
    print(f"  Total val windows:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def _print_dataloader_summary(
    train_ds: AirSketchDataset,
    val_ds: AirSketchDataset,
    test_ds: AirSketchDataset,
) -> None:
    print("\n── Dataset summary ─────────────────────────────────────────────────")
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        dist = ds.gesture_distribution()
        print(
            f"  {name:5s} | windows: {len(ds):7,} | "
            f"draw: {dist['draw']:6,} ({dist['draw_frac']*100:.1f}%) | "
            f"idle: {dist['idle']:6,} ({(1-dist['draw_frac'])*100:.1f}%)"
        )
    print()
