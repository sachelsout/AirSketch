"""
scripts/label_custom_gestures.py

Applies velocity-based draw/idle labelling to custom gesture sessions
and assembles the held-out test split for issue #16.

For each session:
  1. Load landmarks.npy from extract_landmarks.py output
  2. Interpolate short detection gaps (max 2 frames)
  3. Compute per-frame fingertip velocity
  4. Label each frame draw=1 if velocity > threshold, idle=0 otherwise
  5. Save labelled sequences and metadata

Then merge all sessions into a single custom_test_split.json that
issue #16 can load directly.

Usage:
    python scripts/label_custom_gestures.py \
        --input-dir  data/processed/custom/ \
        --output-dir data/processed/custom_labeled/ \
        --splits-out data/splits/custom_test_split.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils import interpolate_missing_landmarks, get_valid_sequence_ranges


# ── Constants ──────────────────────────────────────────────────────────────────

INDEX_FINGERTIP = 8
DRAW_VELOCITY_THRESHOLD = 0.005  # normalized coords/frame — matches dataset.py
WINDOW_SIZE = 16
MIN_SEQUENCE_LENGTH = WINDOW_SIZE + 1


def label_session(
    landmarks_path: Path,
    session_name: str,
) -> dict:
    """
    Load a session's landmarks.npy and produce per-frame draw/idle labels.

    Returns a dict with keys:
        name:           Session identifier string.
        landmarks:      (N, 21, 2) float32 array after interpolation.
        gesture_labels: (N,) int64 array — 1=draw, 0=idle.
        nan_mask:       (N,) bool array — True where landmarks are still NaN.
        valid_ranges:   List of (start, end) tuples for contiguous valid regions.
        stats:          Summary statistics dict.
    """
    raw_landmarks = np.load(landmarks_path)  # (N, 21, 2)
    N = len(raw_landmarks)

    # Interpolate short gaps
    landmarks, nan_mask = interpolate_missing_landmarks(raw_landmarks, max_gap=2)

    # Compute per-frame fingertip velocity
    tips = landmarks[:, INDEX_FINGERTIP, :]  # (N, 2)
    velocity = np.zeros(N, dtype=np.float32)

    # Forward difference: velocity[i] = distance from frame i to frame i+1
    for i in range(N - 1):
        if not nan_mask[i] and not nan_mask[i + 1]:
            velocity[i] = float(np.linalg.norm(tips[i + 1] - tips[i]))

    # Apply threshold
    gesture_labels = (velocity > DRAW_VELOCITY_THRESHOLD).astype(np.int64)

    # NaN frames get idle label (they will be excluded by the dataset anyway)
    gesture_labels[nan_mask] = 0

    # Find valid contiguous ranges for sliding windows
    valid_ranges = get_valid_sequence_ranges(nan_mask, window_size=MIN_SEQUENCE_LENGTH)

    # Summary statistics
    n_valid = int((~nan_mask).sum())
    n_draw = int(gesture_labels.sum())
    n_idle = n_valid - n_draw
    n_nan = int(nan_mask.sum())
    det_rate = n_valid / N if N > 0 else 0.0

    stats = {
        "total_frames": N,
        "valid_frames": n_valid,
        "nan_frames": n_nan,
        "detection_rate": round(det_rate, 4),
        "draw_frames": n_draw,
        "idle_frames": n_idle,
        "draw_fraction": round(n_draw / max(n_valid, 1), 4),
        "valid_ranges": len(valid_ranges),
        "total_windows": sum(
            (e - s - WINDOW_SIZE)
            for s, e in valid_ranges
            if (e - s) >= MIN_SEQUENCE_LENGTH
        ),
    }

    print(
        f"  {session_name}: {N:,} frames | "
        f"det: {det_rate*100:.1f}% | "
        f"draw: {n_draw:,} ({stats['draw_fraction']*100:.1f}%) | "
        f"idle: {n_idle:,} | "
        f"windows: {stats['total_windows']:,}"
    )

    return {
        "name": session_name,
        "landmarks": landmarks,
        "gesture_labels": gesture_labels,
        "nan_mask": nan_mask,
        "valid_ranges": valid_ranges,
        "stats": stats,
    }


def main(input_dir: str, output_dir: str, splits_out: str) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not session_dirs:
        raise RuntimeError(f"No session subdirectories found in {input_dir}")

    print(f"\nLabelling {len(session_dirs)} sessions:\n")

    all_stats = {}
    split_index = {}  # session_name → list of (landmarks_path, gesture_labels_path)

    for session_dir in session_dirs:
        lm_path = session_dir / "landmarks.npy"
        if not lm_path.exists():
            print(f"  WARNING: No landmarks.npy in {session_dir} — skipping.")
            continue

        session_name = session_dir.name
        result = label_session(lm_path, session_name)

        # Save labeled outputs
        session_out = output_dir / session_name
        session_out.mkdir(parents=True, exist_ok=True)

        lm_out = session_out / "landmarks.npy"
        labels_out = session_out / "gesture_labels.npy"
        nanmask_out = session_out / "nan_mask.npy"
        stats_out = session_out / "stats.json"

        np.save(lm_out, result["landmarks"])
        np.save(labels_out, result["gesture_labels"])
        np.save(nanmask_out, result["nan_mask"])

        with open(stats_out, "w") as f:
            json.dump(result["stats"], f, indent=2)

        all_stats[session_name] = result["stats"]
        split_index[session_name] = {
            "landmarks_path": str(lm_out),
            "gesture_labels_path": str(labels_out),
            "nan_mask_path": str(nanmask_out),
            "valid_ranges": result["valid_ranges"],
            "stats": result["stats"],
        }

    # Aggregate summary
    total_windows = sum(s["total_windows"] for s in all_stats.values())
    total_frames = sum(s["total_frames"] for s in all_stats.values())
    mean_det_rate = np.mean([s["detection_rate"] for s in all_stats.values()])
    mean_draw_frac = np.mean([s["draw_fraction"] for s in all_stats.values()])

    split_meta = {
        "sessions": split_index,
        "aggregate": {
            "total_sessions": len(split_index),
            "total_frames": total_frames,
            "total_windows": total_windows,
            "mean_det_rate": round(float(mean_det_rate), 4),
            "mean_draw_frac": round(float(mean_draw_frac), 4),
        },
    }

    with open(splits_out, "w") as f:
        json.dump(split_meta, f, indent=2)

    print("\n── Aggregate summary ───────────────────────────────────────────────")
    print(f"  Sessions:          {len(split_index)}")
    print(f"  Total frames:      {total_frames:,}")
    print(f"  Total windows:     {total_windows:,}")
    print(f"  Mean detection:    {mean_det_rate*100:.1f}%")
    print(f"  Mean draw frac:    {mean_draw_frac*100:.1f}%")
    print(f"\n  Split index saved: {splits_out}")
    print(
        "\n✓ Custom dataset ready. Use data/splits/custom_test_split.json in issue #16."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="data/processed/custom/")
    p.add_argument("--output-dir", default="data/processed/custom_labeled/")
    p.add_argument("--splits-out", default="data/splits/custom_test_split.json")
    main(**vars(p.parse_args()))
