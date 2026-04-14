import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path
from PIL import Image

os.makedirs("notebooks/report/figures", exist_ok=True)

with open("data/processed/egohands/clip_index.json") as f:
    clip_index = json.load(f)

clip_names = list(clip_index.keys())
det_rates = [clip_index[c]["summary"]["detection_rate"] for c in clip_names]
activities = [c.split("_")[0] for c in clip_names]

color_map = {
    "CHESS": "#4A90D9",
    "CARDS": "#E84040",
    "JENGA": "#2DB37A",
    "PUZZLE": "#E8A830",
}
colors = [color_map.get(a, "#888888") for a in activities]

# ── Figure 1: Detection rate per clip ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(len(clip_names)), det_rates, color=colors, edgecolor="none")
ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1.2)
ax.axhline(y=0.70, color="orange", linestyle="--", linewidth=1.0)
ax.set_xticks(range(len(clip_names)))
ax.set_xticklabels([c.replace("_", "\n") for c in clip_names], fontsize=5)
ax.set_ylabel("Detection rate")
ax.set_ylim(0, 1.05)
ax.set_title(
    f"MediaPipe detection rate per EgoHands clip\n"
    f"Mean: {np.mean(det_rates)*100:.1f}%  |  "
    f"Clips below 70%: {sum(r < 0.70 for r in det_rates)}"
)
legend_patches = [Patch(color=v, label=k) for k, v in color_map.items()]
ax.legend(
    handles=legend_patches
    + [
        plt.Line2D([0], [0], color="red", linestyle="--", label="80% threshold"),
        plt.Line2D(
            [0], [0], color="orange", linestyle="--", label="70% exclusion threshold"
        ),
    ],
    fontsize=8,
)
plt.tight_layout()
plt.savefig(
    "notebooks/report/figures/egohands_detection_rates.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved: egohands_detection_rates.png")
print(f"Clips above 80%: {sum(r >= 0.80 for r in det_rates)}")
print(f"Clips 70-80%:    {sum(0.70 <= r < 0.80 for r in det_rates)}")
print(f"Clips below 70%: {sum(r < 0.70 for r in det_rates)}  (will be excluded)")

# ── Figure 2: Sample frames with landmarks ────────────────────────────────────
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

sample_clips = list(clip_index.keys())[::16][:3]

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle(
    "EgoHands sample frames with MediaPipe landmarks\n"
    "(red dot = index fingertip, blue = skeleton)",
    fontsize=10,
)

for row, clip_name in enumerate(sample_clips):
    lm_path = clip_index[clip_name]["landmarks_path"]
    landmarks = np.load(lm_path)
    img_dir = Path("data/raw/egohands/_LABELLED_SAMPLES") / clip_name
    frames = sorted(img_dir.glob("frame_*.jpg"))

    for col, frame_idx in enumerate([0, 49, 99]):
        ax = axes[row, col]
        img = np.array(Image.open(frames[frame_idx]).convert("RGB"))
        uv = landmarks[frame_idx]
        ax.imshow(img)

        if not np.any(np.isnan(uv)):
            h, w = img.shape[:2]
            uv_px = uv * np.array([w, h])
            for a, b in HAND_CONNECTIONS:
                ax.plot(
                    [uv_px[a, 0], uv_px[b, 0]],
                    [uv_px[a, 1], uv_px[b, 1]],
                    "b-",
                    linewidth=0.8,
                    alpha=0.6,
                )
            ax.scatter(
                uv_px[:, 0],
                uv_px[:, 1],
                s=10,
                c="white",
                edgecolors="steelblue",
                linewidths=0.4,
                zorder=5,
            )
            ax.scatter(uv_px[8, 0], uv_px[8, 1], s=40, c="red", zorder=6)
            status = "detected"
        else:
            status = "no detection"

        ax.set_title(f"frame {frame_idx+1} — {status}", fontsize=7)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(clip_name.replace("_", "\n"), fontsize=6)

plt.tight_layout()
plt.savefig(
    "notebooks/report/figures/egohands_samples.png", dpi=150, bbox_inches="tight"
)
print("Saved: egohands_samples.png")

# ── Overlap check ─────────────────────────────────────────────────────────────
with open("data/splits/merged_train_split.json") as f:
    merged = json.load(f)

print("\nOverlap check result:")
print(f"  overlap_check_passed: {merged['overlap_check_passed']}")
print("\nTraining sources:")
print(
    f"  FreiHAND train: {len(merged['sources']['freihand']['train_indices']):,} frames"
)
print(
    f"  EgoHands:       {merged['sources']['egohands']['total_frames']:,} frames "
    f"across {merged['sources']['egohands']['total_clips']} clips"
)
print("\nTest source (held-out, never seen during training):")
print(f"  {merged['test_source']['note']}")

assert merged[
    "overlap_check_passed"
], "Overlap check failed — do not proceed to training"
print("\n✓ All checks passed. Safe to proceed to issue #9.")
