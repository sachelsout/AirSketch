import os
import json
import yaml
import numpy as np

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from src.dataset import AirSketchDataset

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
with open("data/splits/freihand_splits.json") as f:
    splits = json.load(f)

SUBSET = 5000
train_ds = AirSketchDataset(
    landmarks_path="data/processed/freihand/landmarks.npy",
    split_indices=splits["train"][:SUBSET],
    augment=False,
)
print(f"Windows: {len(train_ds):,}")
print(f"Gesture distribution: {train_ds.gesture_distribution()}")
print(f"Class weights: {train_ds.get_class_weights()}")

# Gesture distribution plot

os.makedirs("report/figures", exist_ok=True)
dist = train_ds.gesture_distribution()
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(
    ["idle (0)", "draw (1)"],
    [dist["idle"], dist["draw"]],
    color=["#4A90D9", "#E84040"],
    edgecolor="none",
)
ax.set_title(
    f"Gesture label distribution\n(draw fraction: {dist['draw_frac']*100:.1f}%)"
)
ax.set_ylabel("Window count")
plt.tight_layout()
plt.savefig("report/figures/gesture_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: report/figures/gesture_distribution.png")

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]
INDEX_FINGERTIP = 8

# Dataset windows figure
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
fig.suptitle("4 sample windows", fontsize=10)
for row, sample_idx in enumerate([0, 10, 50, 200]):
    sample = train_ds[sample_idx]
    seq = sample["sequence"].numpy().reshape(16, 21, 2)
    target = sample["target"].numpy()
    label = "draw" if sample["gesture"].item() == 1 else "idle"
    for col, fi in enumerate([0, 5, 10, 15]):
        ax = axes[row, col]
        uv = seq[fi]
        for a, b in HAND_CONNECTIONS:
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], "b-", lw=0.8, alpha=0.6)
        ax.scatter(uv[:, 0], uv[:, 1], s=15, c="white", edgecolors="steelblue", lw=0.5)
        ax.scatter(
            uv[INDEX_FINGERTIP, 0], uv[INDEX_FINGERTIP, 1], s=50, c="red", zorder=6
        )
        if col == 3:
            ax.scatter(target[0], target[1], s=80, c="lime", marker="+", lw=2, zorder=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(f"sample {sample_idx}\n({label})", fontsize=8)
        if row == 0:
            ax.set_title(f"frame {fi}", fontsize=8)
plt.tight_layout()
plt.savefig("report/figures/dataset_windows.png", dpi=150, bbox_inches="tight")
print("Saved: report/figures/dataset_windows.png")

# Augmentation flip figure
train_aug = AirSketchDataset(
    "data/processed/freihand/landmarks.npy",
    splits["train"][:SUBSET],
    augment=True,
    flip_prob=1.0,
    noise_sigma=0.0,
)
idx = 5
orig = train_ds[idx]["sequence"].numpy().reshape(16, 21, 2)[0]
flip = train_aug[idx]["sequence"].numpy().reshape(16, 21, 2)[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
for ax, uv, title in [(ax1, orig, "Original"), (ax2, flip, "Flipped (x → 1-x)")]:
    for a, b in HAND_CONNECTIONS:
        ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], "b-", lw=0.8, alpha=0.6)
    ax.scatter(uv[:, 0], uv[:, 1], s=20, c="white", edgecolors="steelblue", lw=0.5)
    ax.scatter(uv[INDEX_FINGERTIP, 0], uv[INDEX_FINGERTIP, 1], s=60, c="red", zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("report/figures/augmentation_flip.png", dpi=150, bbox_inches="tight")
np.testing.assert_allclose(flip[:, 0], 1.0 - orig[:, 0], atol=1e-6)
print("Saved: report/figures/augmentation_flip.png")
print("✓ Flip verified: x_flipped == 1 - x_original")
