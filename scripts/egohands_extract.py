"""
scripts/egohands_extract.py

Runs MediaPipe landmark extraction on every EgoHands clip independently,
preserving clip boundaries for the temporal sliding window dataset.

For each clip, produces:
    data/processed/egohands/<clip_name>/landmarks.npy
    data/processed/egohands/<clip_name>/detection_log.json
    data/processed/egohands/<clip_name>/summary.json

After all clips are processed, writes a merged index:
    data/processed/egohands/clip_index.json

Usage:
    python scripts/egohands_extract.py \
        --data-dir   data/raw/egohands/_LABELLED_SAMPLES/ \
        --output-dir data/processed/egohands/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def extract_clip(
    clip_dir: Path,
    output_dir: Path,
    min_confidence: float = 0.5,
) -> dict:
    """
    Run extract_landmarks.py on one EgoHands clip (image folder mode).

    EgoHands frames are named frame_0001.jpg...frame_0100.jpg — the
    extract_landmarks.py script handles any sorted image folder, so we
    pass it directly in images mode (no splits file needed here).

    Returns the summary dict from the extraction.
    """
    clip_output = output_dir / clip_dir.name
    clip_output.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/extract_landmarks.py",
            "--input",
            str(clip_dir),
            "--output",
            str(clip_output),
            "--mode",
            "images",
            "--min-confidence",
            str(min_confidence),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ERROR processing clip {clip_dir.name}:")
        print(result.stderr[-500:])  # last 500 chars of stderr
        return {"clip": clip_dir.name, "error": True}

    summary_path = clip_output / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary["clip"] = clip_dir.name
        return summary
    else:
        return {"clip": clip_dir.name, "error": True, "reason": "no summary.json"}


def main(data_dir: str, output_dir: str, min_confidence: float) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Processing {len(clip_dirs)} EgoHands clips ...\n")

    clip_summaries = []
    failed_clips = []

    for clip_dir in tqdm(clip_dirs, desc="Clips", unit="clip"):
        summary = extract_clip(clip_dir, output_dir, min_confidence)
        if summary.get("error"):
            failed_clips.append(clip_dir.name)
        else:
            clip_summaries.append(summary)

    # Build clip index — maps clip name to its output paths and stats
    clip_index = {}
    for s in clip_summaries:
        clip_name = s["clip"]
        clip_index[clip_name] = {
            "landmarks_path": str(output_dir / clip_name / "landmarks.npy"),
            "log_path": str(output_dir / clip_name / "detection_log.json"),
            "summary": s,
        }

    index_path = output_dir / "clip_index.json"
    with open(index_path, "w") as f:
        json.dump(clip_index, f, indent=2)

    # Aggregate stats
    total_frames = sum(s.get("total_frames", 0) for s in clip_summaries)
    total_detected = sum(s.get("detected", 0) for s in clip_summaries)
    mean_det_rate = total_detected / max(total_frames, 1)

    print("\n── EgoHands extraction summary ─────────────────────────────────────")
    print(f"  Clips processed:   {len(clip_summaries)} / {len(clip_dirs)}")
    print(f"  Failed clips:      {len(failed_clips)}")
    print(f"  Total frames:      {total_frames:,}")
    print(f"  Detected:          {total_detected:,} ({mean_det_rate*100:.1f}%)")
    print(f"  Clip index saved:  {index_path}")

    if failed_clips:
        print("\n  Failed clips:")
        for c in failed_clips:
            print(f"    {c}")

    if mean_det_rate < 0.80:
        print("\n  WARNING: Mean detection rate below 80%.")
        print("     EgoHands is egocentric footage — hands may be partially")
        print("     occluded or at unusual angles. Detection rate of 75-85%")
        print("     is expected and acceptable for this dataset.")
    else:
        print("\n✓ EgoHands extraction complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw/egohands/_LABELLED_SAMPLES/")
    p.add_argument("--output-dir", default="data/processed/egohands/")
    p.add_argument("--min-confidence", type=float, default=0.5)
    main(**vars(p.parse_args()))
