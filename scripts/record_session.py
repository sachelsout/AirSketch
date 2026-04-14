"""
scripts/record_session.py

Guided webcam recording script for AirSketch gesture data collection.

Displays task prompts on screen and records to a timestamped .mp4 file.
The session is split into task blocks — an on-screen timer and task label
are burned into the video to aid post-hoc review.

Usage:
    python scripts/record_session.py --name rohan --output data/raw/custom/
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


# ── Task schedule ──────────────────────────────────────────────────────────────
# Each entry: (task_label, duration_seconds, is_drawing)
# is_drawing=True  → frame should eventually be labelled draw
# is_drawing=False → frame should eventually be labelled idle
#
# Total: 1800 seconds (30 minutes)
# Note: keep durations summing to exactly 1800 when editing.

TASK_SCHEDULE = [
    # ── Warm-up / idle baseline (3 min) ───────────────────────────────────────
    ("REST — hand at side, look at camera", 60, False),
    ("IDLE — hand raised, held still", 60, False),
    ("IDLE — hand raised, natural small movements", 60, False),
    # ── Basic strokes (3.5 min) ───────────────────────────────────────────────
    ("DRAW — horizontal lines, left to right", 60, True),
    ("DRAW — vertical lines, top to bottom", 60, True),
    ("DRAW — diagonal lines, both directions", 60, True),
    ("REST — lower hand, relax", 30, False),
    # ── Curved shapes (5 min) ─────────────────────────────────────────────────
    ("DRAW — circles, clockwise", 60, True),
    ("DRAW — circles, counter-clockwise", 60, True),
    ("DRAW — ovals and ellipses", 60, True),
    ("DRAW — arcs and half-circles", 60, True),
    ("REST — lower hand, relax", 30, False),
    # ── Arrows and connectors (3.5 min) ─────────────────────────────────────────
    ("DRAW — straight arrows (draw line then tip)", 60, True),
    ("DRAW — curved arrows", 60, True),
    ("DRAW — double-headed arrows", 60, True),
    ("REST — lower hand, relax", 30, False),
    # ── Flowchart shapes (4.5 min) ──────────────────────────────────────────────
    ("DRAW — rectangles (boxes)", 60, True),
    ("DRAW — rectangles connected by arrows", 60, True),
    ("DRAW — diamond shapes (decision nodes)", 60, True),
    ("DRAW — full 3-node flowchart: box→arrow→box", 60, True),
    ("REST — lower hand, relax", 30, False),
    # ── System design shapes (4.5 min) ──────────────────────────────────────────
    ("DRAW — cylinders / database symbols", 60, True),
    ("DRAW — cloud / server outlines", 60, True),
    ("DRAW — client→server→database diagram", 60, True),
    ("DRAW — labels and short text annotations", 60, True),
    ("REST — lower hand, relax", 30, False),
    # ── Mixed natural drawing (3.5 min) ─────────────────────────────────────────
    ("DRAW — freestyle: draw whatever comes to mind", 120, True),
    ("DRAW — reproduce a system design from memory", 90, True),
    # ── Final idle baseline (3 min) ───────────────────────────────────────────
    ("IDLE — hand raised, held still", 60, False),
    ("IDLE — natural idle: shift, gesture, fidget", 60, False),
    ("REST — hand at side", 60, False),
]

assert sum(d for _, d, _ in TASK_SCHEDULE) == 1800, "Schedule must total 1800 seconds"


def main(name: str, output_dir: str, camera: int) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{name}_{timestamp}.mp4"

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(filename),
        fourcc,
        actual_fps,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    print(f"Recording to: {filename}")
    print(
        f"Resolution:   {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
        f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {actual_fps:.0f}fps"
    )
    print("Press Q at any time to stop early.\n")

    # ── 5-second countdown before starting ────────────────────────────────────
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(
            frame,
            f"Starting in {i}...",
            (400, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            3,
        )
        cv2.imshow("AirSketch Recording", frame)
        cv2.waitKey(1000)

    # ── Main recording loop ────────────────────────────────────────────────────
    session_start = time.time()
    task_idx = 0
    task_start = session_start
    elapsed_offset = 0.0

    while task_idx < len(TASK_SCHEDULE):
        task_label, task_duration, is_drawing = TASK_SCHEDULE[task_idx]
        task_elapsed = time.time() - task_start

        if task_elapsed >= task_duration:
            task_idx += 1
            task_start = time.time()
            elapsed_offset += task_duration
            continue

        ret, frame = cap.read()
        if not ret:
            break

        writer.write(frame)

        # ── Overlay task info on display (not burned into the saved video) ────
        display = frame.copy()
        time_remaining = task_duration - task_elapsed
        total_elapsed = time.time() - session_start

        color = (0, 200, 80) if is_drawing else (200, 180, 0)
        cv2.rectangle(display, (0, 0), (1280, 80), (0, 0, 0), -1)
        cv2.putText(
            display, task_label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
        )
        cv2.putText(
            display,
            f"Task: {int(time_remaining)}s remaining  |  "
            f"Session: {int(total_elapsed//60):02d}:{int(total_elapsed%60):02d} / 30:00  |  "
            f"Task {task_idx+1}/{len(TASK_SCHEDULE)}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )

        cv2.imshow("AirSketch Recording (Q to stop)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Recording stopped early by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    duration = time.time() - session_start
    print("\nRecording complete.")
    print(f"  File:     {filename}")
    print(f"  Duration: {duration/60:.1f} minutes")
    print(f"  Tasks completed: {task_idx}/{len(TASK_SCHEDULE)}")
    print("\nNext step: run extract_landmarks.py on this file (issue #5).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--name", required=True, help="Your first name (used in filename). E.g. rohan"
    )
    p.add_argument(
        "--output",
        default="data/raw/custom/",
        help="Directory to save the recorded video.",
    )
    p.add_argument(
        "--camera", type=int, default=0, help="Camera device index (default: 0)."
    )
    args = p.parse_args()
    main(args.name, args.output, args.camera)
