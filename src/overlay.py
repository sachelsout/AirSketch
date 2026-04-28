"""
src/overlay.py

Drawing stroke overlay for AirSketch.

Receives per-frame InferenceResult objects from the inference loop,
maintains a persistent stroke buffer, detects the clear-canvas gesture,
and renders all accumulated strokes onto the live webcam frame using
cv2.polylines() with anti-aliasing.

This module is the primary rendering layer. It is imported by:
    src/stream.py   -- the pyvirtualcam integration in issue #15
    src/inference.py -- via the frame_callback mechanism

Usage:
    from src.inference import InferenceLoop
    from src.overlay import StrokeOverlay

    overlay = StrokeOverlay()
    loop    = InferenceLoop(
        model_path     = "checkpoints/best_model.optimized.onnx",
        frame_callback = overlay.process,
    )
    for result in loop.run():
        pass   # overlay.process() is called inside loop.run()
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, NamedTuple

import cv2
import numpy as np

from src.inference import InferenceResult


# -- Color palette -------------------------------------------------------------


class StrokeColor(Enum):
    """
    Available stroke colors. Values are BGR tuples for OpenCV.

    The palette is designed for visibility on typical video-call backgrounds
    (neutral walls, home offices). Neon colors are avoided because they look
    harsh on compressed video streams.
    """

    WHITE = (255, 255, 255)
    RED = (60, 60, 220)  # BGR -- warm red
    GREEN = (60, 180, 60)  # BGR -- mid green
    BLUE = (220, 100, 40)  # BGR -- medium blue
    YELLOW = (0, 210, 220)  # BGR -- warm yellow
    CYAN = (200, 180, 0)  # BGR -- soft cyan

    @classmethod
    def cycle(cls, current: "StrokeColor") -> "StrokeColor":
        """Return the next color in the palette (wraps around)."""
        members = list(cls)
        idx = members.index(current)
        return members[(idx + 1) % len(members)]


# -- Stroke data structures ----------------------------------------------------


class Point(NamedTuple):
    """A single (x, y) point in normalized [0, 1] coordinates."""

    x: float
    y: float

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized coords to integer pixel coords."""
        return (int(self.x * width), int(self.y * height))


@dataclass
class Stroke:
    """
    One continuous stroke -- a sequence of points drawn without lifting.

    A new Stroke is created each time the gesture transitions from idle
    to draw. Points are appended while gesture == draw. The stroke is
    finalized when gesture returns to idle (after debounce).

    Attributes:
        points:    Ordered list of normalized (x, y) positions.
        color:     BGR tuple for rendering.
        thickness: Line thickness in pixels.
        active:    True while the user is still drawing this stroke.
                   False once finalized (gesture returned to idle).
    """

    points: list[Point] = field(default_factory=list)
    color: tuple[int, int, int] = StrokeColor.WHITE.value
    thickness: int = 3
    active: bool = True

    def add_point(self, x: float, y: float) -> None:
        self.points.append(Point(x, y))

    def finalize(self) -> None:
        self.active = False

    def to_pixel_array(self, width: int, height: int) -> np.ndarray | None:
        """
        Convert points to an (N, 1, 2) int32 pixel array for cv2.polylines().

        Returns None if the stroke has fewer than 2 points (nothing to draw).
        """
        if len(self.points) < 2:
            return None

        valid_points = [p for p in self.points if np.isfinite(p.x) and np.isfinite(p.y)]
        if len(valid_points) < 2:
            return None

        pts = np.array(
            [[p.to_pixel(width, height)] for p in valid_points],
            dtype=np.int32,
        )  # shape: (N, 1, 2)
        return pts


# -- Stroke buffer -------------------------------------------------------------


class StrokeBuffer:
    """
    Persistent collection of all strokes drawn since the last canvas clear.

    Manages the active stroke lifecycle:
      - Transitions: idle -> draw opens a new stroke
      - Transitions: draw -> idle (after debounce) finalizes the active stroke
      - Clear: discards all strokes and resets state

    Args:
        debounce_frames: Number of consecutive idle frames required before
                         a stroke is considered finished. Default: 4.
                         Higher values produce smoother strokes with fewer
                         breaks, but increase latency before a new stroke
                         can start after lifting.
        max_points_per_stroke: Hard cap on points per stroke. Prevents memory
                               growth during very long continuous strokes.
        color:           Initial stroke color.
        thickness:       Stroke line thickness in pixels.
    """

    def __init__(
        self,
        debounce_frames: int = 4,
        max_points_per_stroke: int = 2000,
        color: StrokeColor = StrokeColor.WHITE,
        thickness: int = 3,
    ):
        self.debounce_frames = debounce_frames
        self.max_points_per_stroke = max_points_per_stroke
        self.color = color
        self.thickness = thickness

        self._strokes: list[Stroke] = []
        self._active_stroke: Stroke | None = None
        self._idle_frame_count: int = 0

    # -- Public API ------------------------------------------------------------

    def update(self, result: InferenceResult) -> None:
        """
        Process one InferenceResult and update stroke state accordingly.

        Called once per frame from StrokeOverlay.process().

        State machine:
            If is_drawing and landmark_detected:
                - If no active stroke: open a new one
                - Append pred_xy to the active stroke
                - Reset idle counter

            If not is_drawing (or no landmark):
                - Increment idle counter
                - If idle counter >= debounce_frames: finalize active stroke
        """
        if result.is_drawing and result.landmark_detected:
            # Start a new stroke on transition from idle to draw
            if self._active_stroke is None:
                self._active_stroke = Stroke(
                    color=self.color.value,
                    thickness=self.thickness,
                    active=True,
                )
                self._strokes.append(self._active_stroke)

            # Append this frame's predicted fingertip to the active stroke.
            # Ignore invalid model outputs to avoid NaNs in render path.
            x, y = float(result.pred_xy[0]), float(result.pred_xy[1])
            if np.isfinite(x) and np.isfinite(y):
                x = float(np.clip(x, 0.0, 1.0))
                y = float(np.clip(y, 0.0, 1.0))
                if len(self._active_stroke.points) < self.max_points_per_stroke:
                    self._active_stroke.add_point(x, y)
                self._idle_frame_count = 0
            else:
                # Treat invalid coords as a temporary idle frame.
                self._idle_frame_count += 1

        else:
            # Accumulate idle frames
            self._idle_frame_count += 1

            # Finalize the stroke after debounce threshold
            if (
                self._active_stroke is not None
                and self._idle_frame_count >= self.debounce_frames
            ):
                self._active_stroke.finalize()
                self._active_stroke = None
                self._idle_frame_count = 0

    def clear(self) -> None:
        """Discard all strokes and reset to empty state."""
        self._strokes.clear()
        self._active_stroke = None
        self._idle_frame_count = 0

    def set_color(self, color: StrokeColor) -> None:
        """Change stroke color for subsequent strokes (not retroactively)."""
        self.color = color

    def cycle_color(self) -> StrokeColor:
        """Advance to the next color and return it."""
        self.color = StrokeColor.cycle(self.color)
        return self.color

    # -- Accessors -------------------------------------------------------------

    @property
    def strokes(self) -> list[Stroke]:
        return self._strokes

    @property
    def total_points(self) -> int:
        return sum(len(s.points) for s in self._strokes)

    @property
    def has_content(self) -> bool:
        return any(len(s.points) >= 2 for s in self._strokes)

    @property
    def is_drawing(self) -> bool:
        return self._active_stroke is not None


# -- Clear-canvas gesture detector ---------------------------------------------


class ClearGestureDetector:
    """
    Detects double-fist gesture to clear the canvas.

    Triggers a clear when the user closes their fist twice in succession.
    A fist is detected when all four fingertips are below their MCP joints.
    The second fist must occur within max_interval_seconds of the first.

    Args:
        cooldown_seconds:    Minimum time between successive clear events.
        max_interval_seconds: Max time allowed between first and second fist.
    """

    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        max_interval_seconds: float = 2.0,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.max_interval_seconds = max_interval_seconds

        self._fist_count: int = 0
        self._was_fist: bool = False
        self._first_fist_ts: float | None = None
        self._last_clear: float = 0.0
        self._progress: float = 0.0

    def update(self, result: InferenceResult) -> bool:
        """
        Process one frame and return True if double-fist clear triggered.
        """
        now = time.perf_counter()

        # Respect cooldown
        if now - self._last_clear < self.cooldown_seconds:
            self._fist_count = 0
            self._was_fist = False
            self._first_fist_ts = None
            self._progress = 0.0
            return False

        is_fist = getattr(result, "is_fist", False)

        # Reset if first fist timed out
        if self._first_fist_ts is not None:
            if now - self._first_fist_ts > self.max_interval_seconds:
                self._fist_count = 0
                self._first_fist_ts = None
                self._progress = 0.0

        # Detect fist open->close transition
        if is_fist and not self._was_fist:
            self._fist_count += 1
            if self._fist_count == 1:
                self._first_fist_ts = now
                self._progress = 0.5  # show half progress after first fist

            if self._fist_count >= 2:
                self._fist_count = 0
                self._was_fist = False
                self._first_fist_ts = None
                self._progress = 0.0
                self._last_clear = now
                return True

        self._was_fist = is_fist
        return False

    @property
    def progress(self) -> float:
        return self._progress


# -- Renderer ------------------------------------------------------------------


class OverlayRenderer:
    """
    Renders accumulated strokes and UI elements onto a BGR frame.

    All rendering is in-place (modifies the frame array directly).
    Uses cv2.polylines() with LINE_AA for anti-aliased stroke rendering.

    Args:
        show_hud:         If True, draw status text and gesture indicators.
        show_fingertip:   If True, draw a dot at the current pred_xy position.
        stroke_alpha:     Opacity of strokes blended over the webcam frame.
                          1.0 = fully opaque (no blending), 0.6 = semi-transparent.
                          Values below 1.0 require an extra frame copy per render.
    """

    def __init__(
        self,
        show_hud: bool = True,
        show_fingertip: bool = True,
        stroke_alpha: float = 1.0,
    ):
        self.show_hud = show_hud
        self.show_fingertip = show_fingertip
        self.stroke_alpha = stroke_alpha

    def render(
        self,
        frame: np.ndarray,
        buffer: StrokeBuffer,
        detector: ClearGestureDetector,
        result: InferenceResult,
    ) -> np.ndarray:
        """
        Render strokes and HUD onto the frame.

        Args:
            frame:    (H, W, 3) uint8 BGR frame -- modified in place.
            buffer:   StrokeBuffer with accumulated strokes.
            detector: ClearGestureDetector for progress overlay.
            result:   Current frame's InferenceResult.

        Returns:
            The annotated frame (same array as input, modified in place).
        """
        h, w = frame.shape[:2]

        # -- Stroke rendering -------------------------------------------------
        if buffer.has_content:
            if self.stroke_alpha < 1.0:
                # Semi-transparent: draw onto a copy and blend
                canvas = frame.copy()
                self._draw_strokes(canvas, buffer.strokes, w, h)
                cv2.addWeighted(
                    canvas, self.stroke_alpha, frame, 1.0 - self.stroke_alpha, 0, frame
                )
            else:
                self._draw_strokes(frame, buffer.strokes, w, h)

        # -- Fingertip dot ----------------------------------------------------
        if self.show_fingertip and result.landmark_detected:
            self._draw_fingertip(frame, result, w, h)

        # -- HUD overlay ------------------------------------------------------
        if self.show_hud:
            self._draw_hud(frame, buffer, detector, result, w, h)

        return frame

    def _draw_strokes(
        self,
        frame: np.ndarray,
        strokes: list[Stroke],
        width: int,
        height: int,
    ) -> None:
        """
        Draw all strokes using cv2.polylines() with anti-aliasing.

        Strokes with fewer than 2 points are skipped (nothing to draw).
        The active (in-progress) stroke is drawn with slightly higher
        thickness to distinguish it from completed strokes.
        """
        max_render_points = 800
        for stroke in strokes:
            pts = stroke.to_pixel_array(width, height)
            if pts is None:
                continue

            # Downsample very long strokes to keep per-frame render cost bounded.
            if len(pts) > max_render_points:
                step = max(1, len(pts) // max_render_points)
                pts = pts[::step]

            thickness = stroke.thickness + (1 if stroke.active else 0)

            cv2.polylines(
                frame,
                [pts],
                isClosed=False,
                color=stroke.color,
                thickness=thickness,
                lineType=cv2.LINE_AA,  # anti-aliasing -- essential for quality
            )

    def _draw_fingertip(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        width: int,
        height: int,
    ) -> None:
        """
        Draw a small indicator dot at the predicted fingertip position.

        Color:  green when drawing, white when idle.
        Size:   slightly larger when drawing to give tactile feedback.
        """
        if np.any(np.isnan(result.pred_xy)):
            return

        px = int(result.pred_xy[0] * width)
        py = int(result.pred_xy[1] * height)

        if result.is_drawing:
            dot_color = (60, 200, 60)  # green -- drawing active
            dot_radius = 7
        else:
            dot_color = (200, 200, 200)  # light grey -- idle
            dot_radius = 5

        # Outer ring
        cv2.circle(frame, (px, py), dot_radius + 3, (255, 255, 255), 1, cv2.LINE_AA)
        # Filled inner dot
        cv2.circle(frame, (px, py), dot_radius, dot_color, -1, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        buffer: StrokeBuffer,
        detector: ClearGestureDetector,
        result: InferenceResult,
        width: int,
        height: int,
    ) -> None:
        """
        Draw status bar and clear-gesture progress arc.

        Status bar: stroke count, total points, current color, draw/idle.
        Progress arc: fills a circle in the top-right corner as the clear
                      gesture is held, giving the user visual feedback that
                      the gesture is being recognized.
        """
        # -- Status bar (bottom-left) -----------------------------------------
        bar_y = height - 10
        cv2.putText(
            frame,
            f"Strokes: {len(buffer.strokes)}  "
            f"pts: {buffer.total_points}  "
            f"{'DRAWING' if buffer.is_drawing else 'idle'}",
            (12, bar_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            buffer.color.value,
            1,
            cv2.LINE_AA,
        )

        # -- Clear gesture progress arc (top-right corner) --------------------
        progress = detector.progress
        if progress > 0.02:
            cx, cy, r = width - 40, 40, 22

            # Background circle (dark)
            cv2.circle(frame, (cx, cy), r, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), r, (120, 120, 120), 1, cv2.LINE_AA)

            # Progress arc -- drawn as a filled ellipse sector
            angle = int(360 * progress)
            cv2.ellipse(
                frame,
                center=(cx, cy),
                axes=(r - 2, r - 2),
                angle=-90,  # start from top
                startAngle=0,
                endAngle=angle,
                color=(60, 200, 220),  # warm yellow
                thickness=4,
                lineType=cv2.LINE_AA,
            )

            # Center icon: "X" to indicate clear action
            cv2.putText(
                frame,
                "X",
                (cx - 6, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # -- Color indicator dot (bottom-right) --------------------------------
        cv2.circle(
            frame,
            (width - 20, height - 20),
            10,
            buffer.color.value,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            (width - 20, height - 20),
            10,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


# -- Top-level overlay class ---------------------------------------------------


class StrokeOverlay:
    """
    Assembles StrokeBuffer, ClearGestureDetector, and OverlayRenderer into
    a single object that can be used as a frame_callback with InferenceLoop.

    This is the primary public API for the overlay system.

    Args:
        initial_color:        Starting stroke color.
        thickness:            Stroke line thickness in pixels (default: 3).
        debounce_frames:      Idle frames before stroke is finalized (default: 4).
        enable_clear_gesture: Enable automatic clear by gesture (default: False).
        show_hud:             Display status bar and progress arc (default: True).
        show_fingertip:       Display fingertip dot (default: True).
        stroke_alpha:         Stroke opacity, 0.0–1.0 (default: 1.0).
        on_clear:             Optional callback() invoked after every clear event.
                              Useful for logging or integration tests.

    Usage with InferenceLoop:
        overlay = StrokeOverlay(initial_color=StrokeColor.WHITE, thickness=3)
        loop    = InferenceLoop(
            model_path     = "checkpoints/best_model.optimized.onnx",
            frame_callback = overlay.process,
        )
        for result in loop.run():
            pass

    Usage standalone (pass frames manually):
        overlay = StrokeOverlay()
        annotated_bgr = overlay.process(bgr_frame, inference_result)
    """

    def __init__(
        self,
        initial_color: StrokeColor = StrokeColor.WHITE,
        thickness: int = 3,
        debounce_frames: int = 4,
        enable_clear_gesture: bool = False,
        show_hud: bool = True,
        show_fingertip: bool = True,
        stroke_alpha: float = 1.0,
        on_clear: Callable | None = None,
    ):
        self._buffer = StrokeBuffer(
            debounce_frames=debounce_frames,
            color=initial_color,
            thickness=thickness,
        )
        self._detector = ClearGestureDetector(
            cooldown_seconds=2.0,
            max_interval_seconds=2.0,
        )
        self._renderer = OverlayRenderer(
            show_hud=show_hud,
            show_fingertip=show_fingertip,
            stroke_alpha=stroke_alpha,
        )
        self._enable_clear_gesture = enable_clear_gesture
        self._on_clear = on_clear
        self._clear_count = 0

    def process(
        self,
        bgr_frame: np.ndarray,
        result: InferenceResult,
    ) -> np.ndarray:
        """
        Process one frame: update state and render strokes onto the frame.

        This method is passed as frame_callback to InferenceLoop.

        Args:
            bgr_frame: (H, W, 3) uint8 BGR frame from the webcam.
            result:    InferenceResult for this frame from the loop.

        Returns:
            Annotated frame with strokes and HUD rendered in-place.
        """
        # -- Update stroke buffer with this frame's gesture -------------------
        self._buffer.update(result)

        # -- Check for clear gesture (optional, disabled by default) ----------
        if self._enable_clear_gesture and self._detector.update(result):
            self._buffer.clear()
            self._clear_count += 1
            if self._on_clear is not None:
                self._on_clear()

        # -- Render strokes and HUD onto the frame ----------------------------
        return self._renderer.render(
            bgr_frame,
            self._buffer,
            self._detector,
            result,
        )

    # -- Controls (called from keyboard handler in stream.py) -----------------

    def clear_canvas(self) -> None:
        """Programmatically clear the canvas (e.g. on keypress)."""
        self._buffer.clear()
        self._clear_count += 1
        if self._on_clear is not None:
            self._on_clear()

    def cycle_color(self) -> StrokeColor:
        """Advance to the next stroke color and return it."""
        return self._buffer.cycle_color()

    def set_color(self, color: StrokeColor) -> None:
        """Set stroke color directly."""
        self._buffer.set_color(color)

    def set_thickness(self, thickness: int) -> None:
        """Set stroke line thickness for subsequent strokes."""
        self._buffer.thickness = max(1, min(thickness, 20))

    # -- Accessors ------------------------------------------------------------

    @property
    def stroke_count(self) -> int:
        return len(self._buffer.strokes)

    @property
    def total_points(self) -> int:
        return self._buffer.total_points

    @property
    def clear_count(self) -> int:
        return self._clear_count

    @property
    def current_color(self) -> StrokeColor:
        return self._buffer.color
