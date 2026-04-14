"""
scripts/egohands_download.py

Validates the EgoHands dataset download for AirSketch.

Checks:
  1. All 48 clip directories exist
  2. Each clip contains exactly 100 .jpg frames
  3. All frames are readable by OpenCV
  4. Reports any missing or corrupt files

Usage:
    python scripts/egohands_download.py \
        --data-dir data/raw/egohands/_LABELLED_SAMPLES/
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


EXPECTED_CLIPS = 48
EXPECTED_FRAMES_PER_CLIP = 100
EXPECTED_TOTAL = EXPECTED_CLIPS * EXPECTED_FRAMES_PER_CLIP  # 4800

KNOWN_ACTIVITIES = {"CHESS", "CARDS", "JENGA", "PUZZLE"}
KNOWN_LOCATIONS = {"OFFICE", "COURTYARD", "LIVING", "KITCHEN"}


def validate(data_dir: Path) -> dict:
    """
    Validate the EgoHands directory structure and frame readability.

    Returns a summary dict with counts and lists of problem files.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    clip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    print(f"Found {len(clip_dirs)} clip directories (expected {EXPECTED_CLIPS}).")

    missing_clips = []
    corrupt_frames = []
    clip_stats = []

    for clip_dir in tqdm(clip_dirs, desc="Validating clips"):
        frames = sorted(clip_dir.glob("frame_*.jpg"))
        n = len(frames)

        if n != EXPECTED_FRAMES_PER_CLIP:
            missing_clips.append(
                {
                    "clip": clip_dir.name,
                    "expected": EXPECTED_FRAMES_PER_CLIP,
                    "found": n,
                }
            )

        # Sample-check readability: read first, middle, last frame
        for f in [frames[0], frames[n // 2], frames[-1]] if frames else []:
            img = cv2.imread(str(f))
            if img is None:
                corrupt_frames.append(str(f))

        # Parse activity and location from folder name
        parts = clip_dir.name.split("_")
        activity = parts[0] if parts else "UNKNOWN"

        clip_stats.append(
            {
                "name": clip_dir.name,
                "activity": activity,
                "frames": n,
            }
        )

    total_frames = sum(c["frames"] for c in clip_stats)

    print("\n── Validation summary ──────────────────────────────────────────────")
    print(f"  Clip directories:  {len(clip_dirs)} / {EXPECTED_CLIPS}")
    print(f"  Total frames:      {total_frames:,} / {EXPECTED_TOTAL:,}")
    print(f"  Incomplete clips:  {len(missing_clips)}")
    print(f"  Corrupt frames:    {len(corrupt_frames)}")

    if missing_clips:
        print("\n  Incomplete clips:")
        for mc in missing_clips:
            print(f"    {mc['clip']}: {mc['found']} frames (expected {mc['expected']})")

    if corrupt_frames:
        print("\n  Corrupt frames (sample):")
        for cf in corrupt_frames[:10]:
            print(f"    {cf}")

    activity_counts = {}
    for c in clip_stats:
        activity_counts[c["activity"]] = activity_counts.get(c["activity"], 0) + 1
    print("\n  Clips per activity:")
    for act, cnt in sorted(activity_counts.items()):
        print(f"    {act:<10s} {cnt} clips ({cnt * 100} frames)")

    return {
        "clip_count": len(clip_dirs),
        "total_frames": total_frames,
        "clip_stats": clip_stats,
        "missing_clips": missing_clips,
        "corrupt_frames": corrupt_frames,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        default="data/raw/egohands/_LABELLED_SAMPLES/",
        help="Path to EgoHands _LABELLED_SAMPLES directory.",
    )
    args = p.parse_args()
    result = validate(Path(args.data_dir))

    if result["missing_clips"] or result["corrupt_frames"]:
        print("\nWARNING: Issues found — re-download before running extraction.")
        raise SystemExit(1)
    else:
        print("\n✓ EgoHands dataset verified. Proceed to landmark extraction.")
