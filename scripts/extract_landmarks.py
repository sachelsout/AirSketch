"""
scripts/extract_landmarks.py

MediaPipe Hands landmark extraction pipeline for AirSketch.

Accepts a folder of images or a single video file, runs MediaPipe hand
landmark detection on each frame, and saves results as .npy arrays.

Outputs (written to --output directory):
    landmarks.npy       float32 (N, 21, 2)  — normalized [0,1] (x,y) per frame.
                        Frames where detection failed are filled with NaN.
    detection_log.json  Per-frame detection status, confidence, and failure reason.
    summary.json        Aggregate statistics: total frames, detection rate, etc.

Usage:
    python scripts/extract_landmarks.py --input <path> --output <dir> --mode <images|video>

See --help for all options.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

NUM_LANDMARKS = 21
INDEX_FINGERTIP = 8
NAN_LANDMARK = np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)

# Failure reason codes written to detection_log.json
REASON_OK = "ok"
REASON_NO_DETECTION = "no_detection"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_UNREADABLE = "unreadable_frame"
REASON_OUT_OF_BOUNDS = "out_of_bounds"


# ── Landmark extraction ────────────────────────────────────────────────────────


def extract_landmarks_from_result(
    result,
    frame_w: int,
    frame_h: int,
    min_confidence: float,
) -> tuple[np.ndarray, str, float]:
    if not result.multi_hand_landmarks:
        return NAN_LANDMARK.copy(), REASON_NO_DETECTION, 0.0

    best_hand_idx = 0
    best_confidence = 0.0

    if result.multi_handedness:
        for i, handedness in enumerate(result.multi_handedness):
            score = handedness.classification[0].score
            if score > best_confidence:
                best_confidence = score
                best_hand_idx = i

    if best_confidence < min_confidence:
        return NAN_LANDMARK.copy(), REASON_LOW_CONFIDENCE, best_confidence

    hand_landmarks = result.multi_hand_landmarks[best_hand_idx]

    uv = np.array(
        [[lm.x, lm.y] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    if np.any(uv < -0.05) or np.any(uv > 1.05):
        return NAN_LANDMARK.copy(), REASON_OUT_OF_BOUNDS, best_confidence

    uv = np.clip(uv, 0.0, 1.0)
    return uv, REASON_OK, best_confidence


# ── Parallel worker ────────────────────────────────────────────────────────────


def _process_chunk(args: tuple) -> list[dict]:
    """
    Worker function for multiprocessing — each worker gets its own MediaPipe
    instance and processes a chunk of image paths independently.
    """
    chunk_paths, min_confidence = args

    # Prevent OpenBLAS from spawning threads inside worker processes
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    results = []
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.5,
    ) as hands:
        for img_path in chunk_paths:
            entry = {"path": str(img_path.name)}

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                entry.update(
                    {"reason": REASON_UNREADABLE, "confidence": 0.0, "uv": None}
                )
                results.append(entry)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            result = hands.process(rgb)

            uv, reason, confidence = extract_landmarks_from_result(
                result, w, h, min_confidence
            )
            entry.update(
                {
                    "reason": reason,
                    "confidence": round(float(confidence), 4),
                    "uv": uv.tolist() if reason == REASON_OK else None,
                }
            )
            results.append(entry)

    return results


# ── Image folder mode ──────────────────────────────────────────────────────────


def process_image_folder(
    input_dir: Path,
    output_dir: Path,
    splits_path: Path | None,
    max_frames: int,
    min_confidence: float,
    dry_run: bool,
    num_workers: int = 1,
) -> dict:
    if splits_path is not None:
        with open(splits_path) as f:
            splits = json.load(f)
        all_indices = sorted(
            splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
        )
        # FreiHAND uses 8-digit zero-padded filenames
        image_paths = [input_dir / f"{i:08d}.jpg" for i in all_indices]
        print(f"Split file provided — processing {len(image_paths):,} indexed images.")
    else:
        image_paths = sorted(
            list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        )
        print(
            f"No split file — processing all {len(image_paths):,} images in {input_dir}."
        )

    if max_frames > 0:
        image_paths = image_paths[:max_frames]
        print(f"--max-frames {max_frames}: truncated to {len(image_paths):,} images.")

    n_frames = len(image_paths)
    landmarks = np.full((n_frames, NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    log = []

    if num_workers > 1:
        print(f"Running with {num_workers} parallel workers ...")

        chunk_size = max(1, n_frames // num_workers)
        chunks = [
            image_paths[i : i + chunk_size] for i in range(0, n_frames, chunk_size)
        ]
        pool_args = [(chunk, min_confidence) for chunk in chunks]

        with Pool(processes=num_workers) as pool:
            chunk_results = list(
                tqdm(
                    pool.imap(_process_chunk, pool_args),
                    total=len(chunks),
                    desc="Chunks",
                    unit="chunk",
                )
            )

        # Flatten results back in original order
        flat_results = [r for chunk in chunk_results for r in chunk]
        for frame_idx, entry in enumerate(flat_results):
            log.append(
                {
                    "frame": frame_idx,
                    "path": entry["path"],
                    "reason": entry["reason"],
                    "confidence": entry["confidence"],
                }
            )
            if entry["uv"] is not None:
                landmarks[frame_idx] = np.array(entry["uv"], dtype=np.float32)

    else:
        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
        ) as hands:
            for frame_idx, img_path in enumerate(
                tqdm(image_paths, desc="Extracting", unit="img")
            ):
                entry = {"frame": frame_idx, "path": str(img_path.name)}

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    entry.update({"reason": REASON_UNREADABLE, "confidence": 0.0})
                    log.append(entry)
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                result = hands.process(rgb)

                uv, reason, confidence = extract_landmarks_from_result(
                    result, w, h, min_confidence
                )
                landmarks[frame_idx] = uv
                entry.update(
                    {"reason": reason, "confidence": round(float(confidence), 4)}
                )
                log.append(entry)

    return _save_outputs(output_dir, landmarks, log, n_frames, dry_run)


# ── Video mode ─────────────────────────────────────────────────────────────────


def process_video(
    video_path: Path,
    output_dir: Path,
    max_frames: int,
    min_confidence: float,
    dry_run: bool,
    start_frame: int = 0,
    end_frame: int = 0,
) -> dict:
    """
    Video mode is always single-threaded (frames must be read sequentially).
    Uses static_image_mode=False so MediaPipe uses tracking between frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine frame range
    frame_start = max(0, start_frame)
    frame_end = end_frame if end_frame > 0 else total_frames
    n_frames = frame_end - frame_start

    if max_frames > 0:
        n_frames = min(n_frames, max_frames)
        frame_end = frame_start + n_frames

    print(f"Video: {video_path.name}")
    print(f"  {total_frames} frames  |  {fps:.1f} fps  |  {width}x{height}")
    print(
        f"  Processing frames {frame_start:,} to {frame_end:,} ({n_frames:,} frames) ..."
    )

    landmarks = np.full((n_frames, NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    log = []

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.5,
    ) as hands:
        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        for output_idx in tqdm(range(n_frames), desc="Extracting", unit="frame"):
            frame_idx = frame_start + output_idx
            ret, bgr = cap.read()
            entry = {
                "frame": frame_idx,
                "timestamp_ms": round(frame_idx / fps * 1000, 1),
            }

            if not ret or bgr is None:
                entry.update({"reason": REASON_UNREADABLE, "confidence": 0.0})
                log.append(entry)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            uv, reason, confidence = extract_landmarks_from_result(
                result, width, height, min_confidence
            )
            landmarks[output_idx] = uv
            entry.update({"reason": reason, "confidence": round(float(confidence), 4)})
            log.append(entry)

    cap.release()
    return _save_outputs(output_dir, landmarks, log, n_frames, dry_run)


# ── Output saving ──────────────────────────────────────────────────────────────


def _save_outputs(
    output_dir: Path,
    landmarks: np.ndarray,
    log: list[dict],
    n_frames: int,
    dry_run: bool,
) -> dict:
    reason_counts: dict[str, int] = {}
    for entry in log:
        r = entry.get("reason", REASON_NO_DETECTION)
        reason_counts[r] = reason_counts.get(r, 0) + 1

    n_ok = reason_counts.get(REASON_OK, 0)
    n_failed = n_frames - n_ok
    detection_rate = n_ok / n_frames if n_frames > 0 else 0.0

    failed_mask = np.all(np.isnan(landmarks[:, :, 0]), axis=-1)
    max_gap = _longest_run(failed_mask)
    mean_gap = _mean_run_length(failed_mask) if n_failed > 0 else 0.0

    summary = {
        "total_frames": n_frames,
        "detected": n_ok,
        "failed": n_failed,
        "detection_rate": round(detection_rate, 4),
        "failure_reasons": reason_counts,
        "max_gap_frames": int(max_gap),
        "mean_gap_frames": round(float(mean_gap), 2),
        "output_shape": list(landmarks.shape),
    }

    print("\n── Extraction summary ──────────────────────────────────────────────")
    print(f"  Total frames:    {n_frames:,}")
    print(f"  Detected:        {n_ok:,}  ({detection_rate*100:.1f}%)")
    print(f"  Failed:          {n_failed:,}")
    for reason, count in reason_counts.items():
        if reason != REASON_OK:
            print(f"    {reason:<25s} {count:,}")
    print(f"  Longest gap:     {max_gap} consecutive failed frames")
    print(f"  Mean gap length: {mean_gap:.1f} frames")

    if detection_rate < 0.80:
        print("\n  WARNING: Detection rate below 80%.")
        print("     Consider re-recording under better lighting, or lowering")
        print("     --min-confidence. Sequences with long gaps will be")
        print("     filtered out by the sliding window dataset in issue #6.")

    if dry_run:
        print("\n  [dry-run] No files written.")
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)

    landmarks_path = output_dir / "landmarks.npy"
    log_path = output_dir / "detection_log.json"
    summary_path = output_dir / "summary.json"

    np.save(landmarks_path, landmarks)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    size_mb = landmarks_path.stat().st_size / 1e6
    print(f"\n  Saved: {landmarks_path}  ({size_mb:.1f} MB)")
    print(f"  Saved: {log_path}")
    print(f"  Saved: {summary_path}")

    return summary


# ── Gap analysis helpers ───────────────────────────────────────────────────────


def _longest_run(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    max_run = cur_run = 0
    for val in mask:
        if val:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def _mean_run_length(mask: np.ndarray) -> float:
    runs, cur = [], 0
    for val in mask:
        if val:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MediaPipe landmark extraction pipeline for AirSketch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to image folder or video file.")
    p.add_argument(
        "--output", required=True, help="Directory to write landmarks.npy and logs."
    )
    p.add_argument(
        "--mode",
        choices=["images", "video"],
        default="images",
        help="Input type: 'images' for a folder, 'video' for a .mp4/.avi file.",
    )
    p.add_argument(
        "--splits",
        default=None,
        help="Path to freihand_splits.json (images mode only).",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe handedness confidence to accept a detection.",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start processing from this frame index (video mode only). 0 = beginning.",
    )
    p.add_argument(
        "--end-frame",
        type=int,
        default=0,
        help="Stop processing at this frame index (exclusive; video mode only). 0 = no limit.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process at most this many frames. 0 = no limit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection but do not write any output files.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (images mode only). "
        "Default 1 = single-threaded. Set to match --cpus-per-task in SLURM.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    start_time = time.time()

    print("AirSketch — MediaPipe landmark extraction")
    print(f"  input:          {input_path}")
    print(f"  output:         {output_dir}")
    print(f"  mode:           {args.mode}")
    print(f"  min_confidence: {args.min_confidence}")
    print(f"  max_frames:     {args.max_frames or 'unlimited'}")
    print(f"  workers:        {args.workers}")
    print(f"  dry_run:        {args.dry_run}\n")

    if args.mode == "images":
        if not input_path.is_dir():
            raise ValueError(
                f"--input must be a directory in images mode. Got: {input_path}"
            )
        summary = process_image_folder(
            input_dir=input_path,
            output_dir=output_dir,
            splits_path=Path(args.splits) if args.splits else None,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
            num_workers=args.workers,
        )
    else:
        if not input_path.is_file():
            raise ValueError(
                f"--input must be a video file in video mode. Got: {input_path}"
            )
        summary = process_video(
            video_path=input_path,
            output_dir=output_dir,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )

    elapsed = time.time() - start_time
    fps_achieved = summary["total_frames"] / elapsed
    print(f"\n  Elapsed: {elapsed:.1f}s  ({fps_achieved:.1f} frames/sec)")
    print(
        f"\n{'Done.' if summary['detection_rate'] >= 0.80 else 'Done with warnings.'}"
    )


if __name__ == "__main__":
    main()
