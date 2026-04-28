"""
src/inference.py

Real-time inference loop for AirSketch.

Captures webcam frames, extracts hand landmarks with MediaPipe,
maintains a rolling T=16 frame buffer, runs the ONNX model on each
complete buffer, and returns predicted fingertip coordinates and
gesture classification per frame.

This module is imported by:
    src/overlay.py   -- adds stroke rendering on top of the frame
    src/stream.py    -- routes the annotated frame to pyvirtualcam

Usage (standalone -- shows live preview without overlay):
    python src/inference.py --model checkpoints/best_model.optimized.onnx

    # With explicit camera index and resolution
    python src/inference.py \
        --model    checkpoints/best_model.optimized.onnx \
        --camera   0 \
        --width    1280 \
        --height   720 \
        --fps      30

    # Headless mode (no preview window -- for integration testing)
    python src/inference.py \
        --model    checkpoints/best_model.optimized.onnx \
        --headless \
        --max-frames 300
"""

import argparse
import collections
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort


# -- Constants -----------------------------------------------------------------

WINDOW_SIZE = 16  # T -- must match training config
NUM_LANDMARKS = 21
INDEX_FINGERTIP = 8  # MediaPipe landmark index for index fingertip
IMAGE_SIZE = 224  # normalization reference (matches training)
LATENCY_WARN_MS = 45.0  # warn in log when a frame exceeds this threshold

# ONNX session I/O names (must match export in issue #12)
INPUT_NAME = "sequence"
OUTPUT_NAMES = ["pred_xy", "gesture_logits"]


# -- Data classes --------------------------------------------------------------


@dataclass
class InferenceResult:
    """
    Output produced for every frame that completes a full pipeline pass.

    Attributes:
        pred_xy:          (2,) float32 -- predicted index fingertip (x, y)
                          in normalized [0, 1] coordinates.
                          Multiply by frame width/height for pixel coords.
        gesture:          int -- 0=idle, 1=draw.
        gesture_conf:     float -- softmax confidence for the predicted class.
        is_drawing:       bool -- convenience alias for gesture == 1.
        landmark_detected: bool -- True if MediaPipe found a hand this frame.
        latency_ms:       float -- wall-clock time from frame capture to
                          this result being returned, in milliseconds.
        frame_idx:        int -- monotonic frame counter since loop start.
        timestamp_ms:     float -- time.perf_counter() * 1000 at frame capture.
    """

    pred_xy: np.ndarray  # (2,) float32
    gesture: int
    gesture_conf: float
    is_drawing: bool
    landmark_detected: bool
    latency_ms: float
    frame_idx: int
    timestamp_ms: float
    is_pointing: bool = False
    is_fist: bool = False

    def pred_xy_px(self, width: int = 1280, height: int = 720):
        """Return pred_xy scaled to pixel coordinates."""
        return np.array(
            [
                self.pred_xy[0] * width,
                self.pred_xy[1] * height,
            ],
            dtype=np.float32,
        )


@dataclass
class LoopStats:
    """
    Running statistics for the inference loop.
    Updated in-place every frame. Printed to console periodically.
    """

    frame_count: int = 0
    detect_count: int = 0
    draw_count: int = 0
    total_latency: float = 0.0
    max_latency: float = 0.0
    latency_over_budget: int = 0  # frames where latency > LATENCY_TARGET_MS
    start_time: float = field(default_factory=time.perf_counter)

    @property
    def fps(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return self.frame_count / max(elapsed, 1e-6)

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency / max(self.frame_count, 1)

    @property
    def detection_rate(self) -> float:
        return self.detect_count / max(self.frame_count, 1)

    def update(
        self, result: "InferenceResult", latency_target_ms: float = 50.0
    ) -> None:
        self.frame_count += 1
        self.detect_count += int(result.landmark_detected)
        self.draw_count += int(result.is_drawing)
        self.total_latency += result.latency_ms
        self.max_latency = max(self.max_latency, result.latency_ms)
        if result.latency_ms > latency_target_ms:
            self.latency_over_budget += 1

    def summary_line(self) -> str:
        return (
            f"FPS: {self.fps:5.1f}  |  "
            f"Latency: {self.mean_latency_ms:5.1f} ms mean / "
            f"{self.max_latency:5.1f} ms max  |  "
            f"Detection: {self.detection_rate*100:4.1f}%  |  "
            f"Draw: {self.draw_count}/{self.frame_count} frames  |  "
            f"Over budget: {self.latency_over_budget}"
        )


# -- Rolling frame buffer ------------------------------------------------------


class FrameBuffer:
    """
    Rolling deque of T=16 MediaPipe landmark arrays in capture order.

    Maintains the invariant that buffer[0] is oldest and buffer[-1] is
    newest. Pre-filled with NaN arrays at construction so the buffer is
    always exactly T frames deep -- even before T real frames have arrived.

    Thread safety: this class is not thread-safe. It is designed for
    single-threaded use within the inference loop. If threading is added
    (see section 6), wrap buffer.push() and buffer.as_model_input() with
    a threading.Lock().
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=window_size)
        self._reset()
        self._frames_since_reset = 0

    def _reset(self) -> None:
        """Fill the buffer with NaN arrays (used at init and on reset)."""
        self._buf.clear()
        nan_frame = np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
        for _ in range(self.window_size):
            self._buf.append(nan_frame.copy())
        self._frames_since_reset = 0

    def push(self, landmarks: np.ndarray | None) -> None:
        """
        Add one frame to the buffer.

        Args:
            landmarks: (21, 2) float32 normalized landmark array from
                       MediaPipe, or None if detection failed. None is
                       stored as a NaN-filled array -- the slot is never
                       skipped.
        """
        if landmarks is None or np.any(np.isnan(landmarks)):
            frame = np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
        else:
            frame = landmarks.astype(np.float32)
        self._buf.append(frame)
        self._frames_since_reset += 1

    def as_model_input(self) -> np.ndarray:
        """
        Return the buffer as a (1, T, 42) float32 array ready for ONNX.

        Flattens each (21, 2) frame to 42 features.
        NaN values are kept -- the ONNX model was trained to handle
        occasional NaN frames via the interpolation logic in issue #5.

        Returns:
            (1, T, 42) float32 numpy array.
        """
        # Stack along the time axis: list of T × (21, 2) -> (T, 21, 2)
        stacked = np.stack(list(self._buf), axis=0)  # (T, 21, 2)
        flat = stacked.reshape(self.window_size, -1)  # (T, 42)
        return flat[np.newaxis].astype(np.float32)  # (1, T, 42)

    @property
    def is_ready(self) -> bool:
        """
        True when the buffer has accumulated at least T real frames.
        Before this point, the buffer is mostly NaN and model outputs
        are unreliable. The loop skips model inference until ready.
        """
        return self._frames_since_reset >= self.window_size

    @property
    def nan_fraction(self) -> float:
        """Fraction of buffer frames that are NaN (detection failures)."""
        nan_count = sum(1 for f in self._buf if np.any(np.isnan(f)))
        return nan_count / self.window_size

    def reset(self) -> None:
        """Clear the buffer and restart warmup. Called after long pauses."""
        self._reset()


# -- MediaPipe session ---------------------------------------------------------


class LandmarkExtractor:
    """
    Thin wrapper around MediaPipe Hands for the real-time loop.

    Reuses a single MediaPipe Hands session across frames (static_image_mode
    is False so tracking is used between frames -- faster than per-frame
    detection). The session is opened as a context manager in the loop.

    Args:
        min_detection_confidence: Minimum confidence to accept a new detection.
        min_tracking_confidence:  Minimum confidence to continue tracking.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._hands = None

    def open(self) -> "LandmarkExtractor":
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,  # tracking mode -- faster
            max_num_hands=1,  # AirSketch uses one hand
            min_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
        )
        return self

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def __enter__(self) -> "LandmarkExtractor":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray | None:
        """
        Run MediaPipe on one BGR frame and return normalized (21, 2) landmarks.

        Args:
            bgr_frame: (H, W, 3) uint8 BGR frame from cv2.VideoCapture.

        Returns:
            (21, 2) float32 array with x, y in [0, 1], or None if no hand.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        # Take the first (and only -- max_num_hands=1) detected hand
        hand = result.multi_hand_landmarks[0]
        uv = np.array(
            [[lm.x, lm.y] for lm in hand.landmark],
            dtype=np.float32,
        )  # (21, 2) already normalized to [0, 1] by MediaPipe

        # Light bounds check -- reject detections with extreme outliers
        if np.any(uv < -0.05) or np.any(uv > 1.05):
            return None

        return np.clip(uv, 0.0, 1.0)

    @staticmethod
    def is_pointing(landmarks: np.ndarray) -> bool:
        """
        Check if the hand is in a pointing pose -- index finger extended,
        other fingers curled.

        Uses y-coordinate relationships between fingertip and MCP joints.
        In normalized coords, smaller y = higher in frame.

        Landmark indices:
            Index:  MCP=5, PIP=6, DIP=7, TIP=8
            Middle: MCP=9, TIP=12
            Ring:   MCP=13, TIP=16
            Pinky:  MCP=17, TIP=20
            Thumb:  MCP=2, TIP=4
        """
        if landmarks is None or np.any(np.isnan(landmarks)):
            return False

        # Use PIP joints and a small margin to reduce flicker from jitter.
        margin = 0.015

        # Index finger extended: tip is clearly above PIP.
        index_extended = landmarks[8, 1] < (landmarks[6, 1] - margin)

        # Other fingers curled: tips are clearly below their PIP joints.
        middle_curled = landmarks[12, 1] > (landmarks[10, 1] + margin)
        ring_curled = landmarks[16, 1] > (landmarks[14, 1] + margin)
        pinky_curled = landmarks[20, 1] > (landmarks[18, 1] + margin)

        return index_extended and middle_curled and ring_curled and pinky_curled

    @staticmethod
    def is_fist(landmarks: np.ndarray) -> bool:
        if landmarks is None or np.any(np.isnan(landmarks)):
            return False
        index_curled = landmarks[8, 1] > landmarks[5, 1]
        middle_curled = landmarks[12, 1] > landmarks[9, 1]
        ring_curled = landmarks[16, 1] > landmarks[13, 1]
        pinky_curled = landmarks[20, 1] > landmarks[17, 1]
        return index_curled and middle_curled and ring_curled and pinky_curled


# -- ONNX inference session ----------------------------------------------------


class ONNXPredictor:
    """
    Wraps an ONNX Runtime inference session for AirSketch.

    Loads the optimized ONNX model from issue #12 and exposes a single
    predict() method that accepts a (1, T, 42) buffer and returns
    (pred_xy, gesture, gesture_conf).

    The session is created once at construction time -- do not create a
    new InferenceSession per frame (it takes ~50 ms to initialize).

    Args:
        onnx_path:      Path to best_model.optimized.onnx.
        num_threads:    Number of ORT intra-op threads. Default: 0 (auto).
                        Set to 4 for a typical laptop quad-core.
    """

    def __init__(self, onnx_path: str | Path, num_threads: int = 0):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1

        self._sess = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Verify I/O names match the export from issue #12
        input_names = [i.name for i in self._sess.get_inputs()]
        assert INPUT_NAME in input_names, (
            f"Expected input '{INPUT_NAME}', found {input_names}. "
            f"Was the model exported with the correct input_names in issue #12?"
        )

    def predict(self, buffer: np.ndarray) -> tuple[np.ndarray, int, float]:
        """
        Run one inference pass on the current frame buffer.

        Args:
            buffer: (1, T, 42) float32 numpy array from FrameBuffer.as_model_input().

        Returns:
            pred_xy:      (2,) float32 -- fingertip (x, y) in [0, 1].
            gesture:      int -- 0=idle, 1=draw.
            gesture_conf: float -- softmax confidence for predicted class.
        """
        outputs = self._sess.run(OUTPUT_NAMES, {INPUT_NAME: buffer})
        pred_xy = outputs[0][0]  # (2,) -- strip batch dim
        ges_logits = outputs[1][0]  # (2,) -- strip batch dim

        # Softmax for confidence score
        exp_logits = np.exp(ges_logits - ges_logits.max())
        probs = exp_logits / exp_logits.sum()
        gesture = int(np.argmax(probs))
        gesture_conf = float(probs[gesture])

        return pred_xy.astype(np.float32), gesture, gesture_conf


# -- Latency logger ------------------------------------------------------------


class LatencyLogger:
    """
    Logs per-frame latency and FPS to the console at a configurable interval.

    Also writes a structured CSV log to disk when a log_path is provided,
    which is used by issue #16's evaluation script to report end-to-end
    latency in the paper.

    Args:
        print_every:  Print a summary line every N frames.
        log_path:     Optional CSV file path. If provided, every frame's
                      latency is appended to the file.
    """

    def __init__(self, print_every: int = 30, log_path: str | Path | None = None):
        self.print_every = print_every
        self.log_path = Path(log_path) if log_path else None
        self._file = None

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_path, "w", buffering=1)  # line-buffered
            self._file.write(
                "frame_idx,timestamp_ms,latency_ms,detected,gesture," "pred_x,pred_y\n"
            )

    def log(self, result: InferenceResult, stats: LoopStats) -> None:
        """Log one frame result. Called from the main loop every frame."""
        if self._file:
            self._file.write(
                f"{result.frame_idx},{result.timestamp_ms:.1f},"
                f"{result.latency_ms:.2f},{int(result.landmark_detected)},"
                f"{result.gesture},{result.pred_xy[0]:.4f},{result.pred_xy[1]:.4f}\n"
            )

        if result.latency_ms > LATENCY_WARN_MS:
            print(
                f"  [WARN] Frame {result.frame_idx}: "
                f"latency {result.latency_ms:.1f} ms > {LATENCY_WARN_MS:.0f} ms"
            )

        if stats.frame_count % self.print_every == 0:
            print(f"  [{stats.frame_count:6d}]  {stats.summary_line()}")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# -- Main inference loop -------------------------------------------------------


class InferenceLoop:
    """
    Full real-time inference pipeline: capture -> extract -> buffer -> predict.

    Designed to be used as a generator that yields InferenceResult objects,
    one per frame. The caller (overlay.py or stream.py) iterates over the
    generator and handles rendering.

    Args:
        model_path:               Path to best_model.optimized.onnx.
        camera_index:             OpenCV camera device index (default: 0).
        width:                    Capture width in pixels (default: 1280).
        height:                   Capture height in pixels (default: 720).
        target_fps:               Target capture frame rate (default: 30).
        min_detection_confidence: MediaPipe detection threshold.
        min_tracking_confidence:  MediaPipe tracking threshold.
        num_ort_threads:          ORT intra-op threads (0 = auto).
        log_path:                 Optional CSV path for per-frame latency log.
        print_every:              Console log interval in frames.
        frame_callback:           Optional callback(bgr_frame, result) called
                                  after each inference. Used by overlay.py to
                                  draw strokes without subclassing.

    Usage:
        loop = InferenceLoop("checkpoints/best_model.optimized.onnx")
        for result in loop.run():
            # result.pred_xy, result.is_drawing, result.latency_ms
            pass
    """

    def __init__(
        self,
        model_path: str | Path,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_ort_threads: int = 0,
        log_path: str | Path | None = None,
        print_every: int = 30,
        frame_callback: Callable | None = None,
        key_callback: Callable | None = None,
        ema_alpha: float = 0.3,
    ):
        self.model_path = Path(model_path)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.log_path = log_path
        self.print_every = print_every
        self.frame_callback = frame_callback
        self.key_callback = key_callback
        self._extractor = LandmarkExtractor(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._predictor = ONNXPredictor(self.model_path, num_ort_threads)
        self._buffer = FrameBuffer(WINDOW_SIZE)
        self._ema_xy = None
        self._ema_alpha = ema_alpha
        self._stats = LoopStats()
        self._logger = LatencyLogger(print_every, log_path)

    def _open_camera(self) -> cv2.VideoCapture:
        """Open the webcam and configure resolution and FPS."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.camera_index}. "
                f"Check that the webcam is connected and not in use by another app."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(
            f"  Camera {self.camera_index} opened: "
            f"{actual_w}x{actual_h} @ {actual_fps:.0f} fps"
        )

        if actual_w != self.width or actual_h != self.height:
            print(
                f"  WARNING: Requested {self.width}x{self.height} but got "
                f"{actual_w}x{actual_h}. "
                f"Using actual resolution for coordinate scaling."
            )
            self.width = actual_w
            self.height = actual_h

        return cap

    def _process_frame(
        self,
        bgr_frame: np.ndarray,
        frame_idx: int,
        capture_ts: float,
    ) -> InferenceResult:
        """
        Run the full pipeline on one frame and return an InferenceResult.

        This method is the hot path -- every allocation and call here
        contributes to per-frame latency. MediaPipe and ONNX are the
        two dominant costs.

        Args:
            bgr_frame:  (H, W, 3) uint8 BGR frame from cv2.read().
            frame_idx:  Monotonic frame counter.
            capture_ts: time.perf_counter() at the moment cap.read() returned.

        Returns:
            InferenceResult with all fields populated.
        """
        # -- Stage 2: MediaPipe landmark extraction ---------------------------
        landmarks = self._extractor.extract(bgr_frame)
        detected = landmarks is not None
        pointing = LandmarkExtractor.is_pointing(landmarks) if detected else False
        is_fist = LandmarkExtractor.is_fist(landmarks) if detected else False

        # -- Stage 3: Buffer update -------------------------------------------
        self._buffer.push(landmarks)

        # -- Stage 4: ONNX inference ------------------------------------------
        if self._buffer.is_ready:
            model_input = self._buffer.as_model_input()
            pred_xy, gesture, gesture_conf = self._predictor.predict(model_input)
        else:
            # Buffer still warming up -- return a neutral result
            pred_xy = np.array([0.5, 0.5], dtype=np.float32)
            gesture = 0
            gesture_conf = 0.5

        # -- EMA smoothing ------------------------------------------------
        if not np.any(np.isnan(pred_xy)):
            if self._ema_xy is None:
                self._ema_xy = pred_xy.copy()
            else:
                self._ema_xy = (
                    self._ema_alpha * pred_xy + (1 - self._ema_alpha) * self._ema_xy
                )
            pred_xy = self._ema_xy.copy()

        # Guard against invalid model outputs to keep downstream rendering safe.
        if not np.all(np.isfinite(pred_xy)):
            if self._ema_xy is not None and np.all(np.isfinite(self._ema_xy)):
                pred_xy = self._ema_xy.copy()
            else:
                pred_xy = np.array([0.5, 0.5], dtype=np.float32)
        pred_xy = np.clip(pred_xy, 0.0, 1.0).astype(np.float32)

        # -- Stage 5: Pose-driven draw-state gating ----------------------------
        # Draw when a valid hand is detected and the pointing pose is active.
        # This avoids interruptions caused by transient model gesture flips.
        is_drawing = detected and np.all(np.isfinite(pred_xy)) and pointing

        # -- Stage 6: Output packaging ----------------------------------------
        latency_ms = (time.perf_counter() - capture_ts) * 1000.0

        return InferenceResult(
            pred_xy=pred_xy,
            gesture=gesture,
            gesture_conf=gesture_conf,
            is_drawing=is_drawing,
            landmark_detected=detected,
            latency_ms=latency_ms,
            frame_idx=frame_idx,
            timestamp_ms=capture_ts * 1000.0,
            is_pointing=pointing,
            is_fist=is_fist,
        )

    def run(
        self,
        max_frames: int = 0,
        show_preview: bool = True,
    ) -> Iterator[InferenceResult]:
        """
        Run the inference loop as a generator.

        Yields one InferenceResult per frame. The caller handles display
        and overlay (issue #14) or virtual camera output (issue #15).

        Args:
            max_frames:   Stop after this many frames. 0 = run until Q pressed.
            show_preview: Show a minimal OpenCV preview window. Set False
                          when the overlay in issue #14 handles display.

        Yields:
            InferenceResult for each frame.
        """
        cap = self._open_camera()
        frame_idx = 0

        print("\nInference loop starting. Press Q to quit.")
        print(f"  Model: {self.model_path}")
        print(
            f"  Buffer warmup: {WINDOW_SIZE} frames (~"
            f"{WINDOW_SIZE / self.target_fps:.1f}s at {self.target_fps} fps)\n"
        )

        try:
            with self._extractor:
                while True:
                    # -- Stage 1: Frame capture --------------------------------
                    capture_ts = time.perf_counter()
                    ret, bgr = cap.read()
                    if ret and bgr is not None:
                        bgr = cv2.flip(bgr, 1)

                    if not ret or bgr is None:
                        print(f"  WARNING: Frame {frame_idx} read failed -- skipping.")
                        continue

                    # -- Full pipeline -----------------------------------------
                    result = self._process_frame(bgr, frame_idx, capture_ts)

                    # -- Stats and logging ------------------------------------
                    self._stats.update(result)
                    self._logger.log(result, self._stats)

                    # -- Frame callback (used by overlay.py) ------------------
                    if self.frame_callback is not None:
                        self.frame_callback(bgr, result)

                    # -- Minimal preview (when no overlay is active) ----------
                    if show_preview:
                        _draw_preview_hud(bgr, result, self._stats)
                        cv2.imshow("AirSketch Inference (Q to quit)", bgr)

                    yield result

                    frame_idx += 1

                    if max_frames > 0 and frame_idx >= max_frames:
                        print(f"\n  max_frames={max_frames} reached. Stopping.")
                        break

                    # -- Q to quit / key callback -----------------------------
                    # cv2.waitKey() returns -1 when no key is pressed.
                    # Avoid forwarding synthetic 255 values to key handlers.
                    raw_key = cv2.waitKey(1)
                    if raw_key != -1:
                        key = raw_key & 0xFF
                        if key == ord("q"):
                            print("\n  Q pressed. Stopping.")
                            break
                        if self.key_callback is not None:
                            self.key_callback(key)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._logger.close()
            self._print_final_stats()

    def _print_final_stats(self) -> None:
        """Print a summary of the full session to console."""
        s = self._stats
        print("\n-- Session summary ------------------------------------------")
        print(f"  Total frames:     {s.frame_count:,}")
        print(f"  Mean FPS:         {s.fps:.1f}")
        print(f"  Mean latency:     {s.mean_latency_ms:.1f} ms")
        print(f"  Max latency:      {s.max_latency:.1f} ms")
        print(
            f"  Over budget:      {s.latency_over_budget} frames "
            f"({s.latency_over_budget/max(s.frame_count,1)*100:.1f}%)"
        )
        print(f"  Detection rate:   {s.detection_rate*100:.1f}%")
        print(
            f"  Draw frames:      {s.draw_count:,} "
            f"({s.draw_count/max(s.frame_count,1)*100:.1f}%)"
        )
        print("-----------------------------------------------------------\n")


# -- Preview HUD ---------------------------------------------------------------


def _draw_preview_hud(
    bgr: np.ndarray,
    result: InferenceResult,
    stats: LoopStats,
) -> None:
    """
    Draw a minimal heads-up display on the preview frame.

    Shows FPS, latency, gesture state, and a dot at the predicted
    fingertip position. This is only used in standalone mode -- issue
    #14's overlay replaces this with the full stroke renderer.

    All drawing is in-place on bgr.
    """
    h, w = bgr.shape[:2]

    # -- Status bar (top of frame) --------------------------------------------
    bar_color = (0, 0, 0)
    cv2.rectangle(bgr, (0, 0), (w, 52), bar_color, -1)

    gesture_text = "DRAW" if result.is_drawing else "IDLE"
    detect_text = "hand" if result.landmark_detected else "no hand"
    latency_color = (0, 200, 80) if result.latency_ms < 50 else (0, 80, 220)
    gesture_color = (0, 200, 80) if result.is_drawing else (180, 180, 180)

    cv2.putText(
        bgr,
        f"FPS: {stats.fps:.1f}  |  "
        f"Latency: {result.latency_ms:.1f} ms  |  "
        f"{detect_text}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        latency_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        f"Gesture: {gesture_text} ({result.gesture_conf*100:.0f}%)  |  "
        f"Buffer: {'ready' if result.frame_idx >= WINDOW_SIZE else 'warming up'}",
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        gesture_color,
        1,
        cv2.LINE_AA,
    )

    # -- Predicted fingertip dot ----------------------------------------------
    if result.landmark_detected and not np.any(np.isnan(result.pred_xy)):
        px = int(result.pred_xy[0] * w)
        py = int(result.pred_xy[1] * h)
        # Outer ring (white)
        cv2.circle(bgr, (px, py), 10, (255, 255, 255), 2, cv2.LINE_AA)
        # Inner fill (color by gesture)
        dot_color = (0, 200, 80) if result.is_drawing else (100, 100, 100)
        cv2.circle(bgr, (px, py), 6, dot_color, -1, cv2.LINE_AA)


# -- CLI -----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AirSketch real-time inference loop (standalone mode).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="checkpoints/best_model.optimized.onnx")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--log-path", default=None, help="CSV file path for per-frame latency log."
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until Q).",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Disable preview window (for CI / integration tests).",
    )
    p.add_argument(
        "--num-threads", type=int, default=0, help="ORT intra-op threads. 0 = auto."
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.05,
        help="EMA smoothing factor. Lower = smoother.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loop = InferenceLoop(
        model_path=args.model,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        log_path=args.log_path,
        num_ort_threads=args.num_threads,
        ema_alpha=args.ema_alpha,
    )
    for _ in loop.run(
        max_frames=args.max_frames,
        show_preview=not args.headless,
    ):
        pass  # overlay.py attaches a frame_callback instead
