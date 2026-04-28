"""
tests/test_overlay.py

Unit tests for the stroke overlay system.
No webcam or ONNX model required -- all tests use synthetic InferenceResult
objects and small numpy frames.
"""

import time

import numpy as np

from src.inference import InferenceResult
from src.overlay import (
    ClearGestureDetector,
    OverlayRenderer,
    Point,
    Stroke,
    StrokeBuffer,
    StrokeColor,
    StrokeOverlay,
)


# -- Fixtures ------------------------------------------------------------------


def fake_result(
    is_drawing: bool = False,
    landmark_detected: bool = True,
    pred_x: float = 0.5,
    pred_y: float = 0.5,
    frame_idx: int = 0,
    latency_ms: float = 20.0,
    is_fist: bool = False,
) -> InferenceResult:
    return InferenceResult(
        pred_xy=np.array([pred_x, pred_y], dtype=np.float32),
        gesture=1 if is_drawing else 0,
        gesture_conf=0.9,
        is_drawing=is_drawing,
        landmark_detected=landmark_detected,
        latency_ms=latency_ms,
        frame_idx=frame_idx,
        timestamp_ms=frame_idx * 33.3,
        is_fist=is_fist,
    )


def blank_frame(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# -- Point tests ---------------------------------------------------------------


class TestPoint:

    def test_to_pixel_center(self):
        p = Point(0.5, 0.5)
        assert p.to_pixel(640, 480) == (320, 240)

    def test_to_pixel_origin(self):
        p = Point(0.0, 0.0)
        assert p.to_pixel(640, 480) == (0, 0)

    def test_to_pixel_far_corner(self):
        p = Point(1.0, 1.0)
        assert p.to_pixel(640, 480) == (640, 480)


# -- Stroke tests --------------------------------------------------------------


class TestStroke:

    def test_add_point_increments_count(self):
        s = Stroke()
        s.add_point(0.1, 0.2)
        s.add_point(0.3, 0.4)
        assert len(s.points) == 2

    def test_finalize_sets_active_false(self):
        s = Stroke()
        assert s.active is True
        s.finalize()
        assert s.active is False

    def test_to_pixel_array_none_for_single_point(self):
        s = Stroke()
        s.add_point(0.5, 0.5)
        assert s.to_pixel_array(640, 480) is None

    def test_to_pixel_array_shape(self):
        s = Stroke()
        s.add_point(0.1, 0.1)
        s.add_point(0.5, 0.5)
        s.add_point(0.9, 0.9)
        pts = s.to_pixel_array(640, 480)
        assert pts is not None
        assert pts.shape == (3, 1, 2)
        assert pts.dtype == np.int32

    def test_to_pixel_array_empty_returns_none(self):
        s = Stroke()
        assert s.to_pixel_array(640, 480) is None


# -- StrokeBuffer tests --------------------------------------------------------


class TestStrokeBuffer:

    def test_initially_empty(self):
        buf = StrokeBuffer()
        assert len(buf.strokes) == 0
        assert buf.total_points == 0
        assert not buf.has_content
        assert not buf.is_drawing

    def test_drawing_frame_opens_stroke(self):
        buf = StrokeBuffer()
        buf.update(fake_result(is_drawing=True))
        assert len(buf.strokes) == 1
        assert buf.is_drawing

    def test_points_accumulate_while_drawing(self):
        buf = StrokeBuffer(debounce_frames=2)
        for i in range(10):
            buf.update(
                fake_result(is_drawing=True, pred_x=i * 0.05, pred_y=0.5, frame_idx=i)
            )
        assert buf.total_points == 10

    def test_idle_after_debounce_finalizes_stroke(self):
        buf = StrokeBuffer(debounce_frames=3)
        # Draw 5 frames
        for i in range(5):
            buf.update(fake_result(is_drawing=True, frame_idx=i))
        # Then 3 idle frames (exactly debounce threshold)
        for i in range(3):
            buf.update(fake_result(is_drawing=False, frame_idx=5 + i))
        assert not buf.is_drawing
        assert len(buf.strokes) == 1
        assert not buf.strokes[0].active

    def test_idle_before_debounce_does_not_finalize(self):
        buf = StrokeBuffer(debounce_frames=4)
        for i in range(5):
            buf.update(fake_result(is_drawing=True, frame_idx=i))
        # Only 2 idle frames -- below debounce threshold of 4
        for i in range(2):
            buf.update(fake_result(is_drawing=False, frame_idx=5 + i))
        assert buf.is_drawing, "Stroke should still be active before debounce"

    def test_new_draw_after_idle_opens_new_stroke(self):
        buf = StrokeBuffer(debounce_frames=2)
        # First stroke
        for i in range(3):
            buf.update(fake_result(is_drawing=True, frame_idx=i))
        # Idle to finalize
        for i in range(2):
            buf.update(fake_result(is_drawing=False, frame_idx=3 + i))
        # Second stroke
        for i in range(3):
            buf.update(fake_result(is_drawing=True, frame_idx=5 + i))
        assert len(buf.strokes) == 2

    def test_clear_empties_all_strokes(self):
        buf = StrokeBuffer()
        for i in range(5):
            buf.update(fake_result(is_drawing=True, frame_idx=i))
        buf.clear()
        assert len(buf.strokes) == 0
        assert buf.total_points == 0
        assert not buf.has_content
        assert not buf.is_drawing

    def test_no_stroke_on_undetected_frame(self):
        buf = StrokeBuffer()
        buf.update(fake_result(is_drawing=True, landmark_detected=False))
        assert len(buf.strokes) == 0

    def test_max_points_per_stroke_respected(self):
        buf = StrokeBuffer(max_points_per_stroke=10)
        for i in range(50):
            buf.update(fake_result(is_drawing=True, pred_x=i * 0.01, frame_idx=i))
        assert buf.total_points <= 10

    def test_cycle_color_advances(self):
        buf = StrokeBuffer(color=StrokeColor.WHITE)
        color2 = buf.cycle_color()
        assert color2 != StrokeColor.WHITE

    def test_set_color_affects_new_strokes(self):
        buf = StrokeBuffer(color=StrokeColor.WHITE)
        buf.set_color(StrokeColor.RED)
        buf.update(fake_result(is_drawing=True))
        assert buf.strokes[0].color == StrokeColor.RED.value

    def test_has_content_false_with_single_point_strokes(self):
        buf = StrokeBuffer(debounce_frames=1)
        buf.update(fake_result(is_drawing=True))
        buf.update(fake_result(is_drawing=False))
        # One stroke with 1 point -- has_content should be False
        assert not buf.has_content


# -- ClearGestureDetector tests ------------------------------------------------


def _do_fist_transition(det: ClearGestureDetector) -> None:
    """Simulate one fist open->close transition."""
    det.update(fake_result(is_fist=False))  # hand open
    det.update(fake_result(is_fist=True))  # hand closes


class TestClearGestureDetector:

    def test_no_trigger_on_single_fist(self):
        """One fist transition must NOT trigger a clear."""
        det = ClearGestureDetector()
        _do_fist_transition(det)
        assert det.progress == 0.5, "Progress should be 0.5 after first fist"

    def test_trigger_on_double_fist(self):
        """Two fist transitions within the interval MUST trigger a clear."""
        det = ClearGestureDetector(cooldown_seconds=0.0, max_interval_seconds=5.0)
        _do_fist_transition(det)
        det.update(fake_result(is_fist=False))  # open between fists
        triggered = det.update(fake_result(is_fist=True))  # second close
        assert triggered, "Double-fist should trigger a clear"

    def test_no_trigger_when_no_fist(self):
        """Frames with no fist should never trigger."""
        det = ClearGestureDetector()
        for _ in range(20):
            assert not det.update(fake_result(is_fist=False))

    def test_progress_zero_initially(self):
        det = ClearGestureDetector()
        assert det.progress == 0.0

    def test_progress_half_after_first_fist(self):
        det = ClearGestureDetector()
        _do_fist_transition(det)
        assert det.progress == 0.5

    def test_progress_resets_after_trigger(self):
        det = ClearGestureDetector(cooldown_seconds=0.0, max_interval_seconds=5.0)
        _do_fist_transition(det)
        det.update(fake_result(is_fist=False))
        det.update(fake_result(is_fist=True))  # triggers
        assert det.progress == 0.0

    def test_no_trigger_when_interval_exceeded(self):
        """Second fist arriving after max_interval_seconds should NOT trigger."""
        det = ClearGestureDetector(cooldown_seconds=0.0, max_interval_seconds=0.01)
        _do_fist_transition(det)  # first fist
        time.sleep(0.05)  # wait past max_interval
        det.update(fake_result(is_fist=False))
        triggered = det.update(fake_result(is_fist=True))  # second fist, too late
        assert not triggered, "Second fist after interval expiry should not trigger"

    def test_cooldown_prevents_immediate_retrigger(self):
        det = ClearGestureDetector(cooldown_seconds=5.0, max_interval_seconds=10.0)
        # First double-fist
        _do_fist_transition(det)
        det.update(fake_result(is_fist=False))
        det.update(fake_result(is_fist=True))  # triggers
        # Immediately try again -- cooldown should block
        _do_fist_transition(det)
        det.update(fake_result(is_fist=False))
        second = det.update(fake_result(is_fist=True))
        assert not second, "Cooldown should prevent immediate retrigger"


# -- OverlayRenderer tests -----------------------------------------------------


class TestOverlayRenderer:

    def test_render_returns_frame_array(self):
        renderer = OverlayRenderer()
        frame = blank_frame()
        buf = StrokeBuffer()
        det = ClearGestureDetector()
        result = fake_result()
        out = renderer.render(frame, buf, det, result)
        assert out is frame  # in-place modification

    def test_render_with_strokes_does_not_crash(self):
        renderer = OverlayRenderer()
        frame = blank_frame()
        buf = StrokeBuffer()
        det = ClearGestureDetector()
        # Draw a stroke
        for i in range(10):
            buf.update(fake_result(is_drawing=True, pred_x=i * 0.05, frame_idx=i))
        result = fake_result()
        renderer.render(frame, buf, det, result)  # should not raise

    def test_render_modifies_frame_when_strokes_present(self):
        renderer = OverlayRenderer(show_hud=False, show_fingertip=False)
        frame = blank_frame()
        buf = StrokeBuffer()
        det = ClearGestureDetector()
        # Draw a clear horizontal stroke across the frame
        for i in range(20):
            buf.update(
                fake_result(is_drawing=True, pred_x=i * 0.04, pred_y=0.5, frame_idx=i)
            )
        renderer.render(frame, buf, det, fake_result())
        # Frame should no longer be all zeros after rendering
        assert frame.sum() > 0, "Rendering strokes should modify the frame"

    def test_empty_buffer_leaves_frame_unchanged(self):
        renderer = OverlayRenderer(show_hud=False, show_fingertip=False)
        frame = blank_frame()
        original = frame.copy()
        buf = StrokeBuffer()
        det = ClearGestureDetector()
        renderer.render(frame, buf, det, fake_result(landmark_detected=False))
        np.testing.assert_array_equal(frame, original)


# -- StrokeOverlay integration tests -------------------------------------------


class TestStrokeOverlay:

    def test_process_returns_frame(self):
        overlay = StrokeOverlay()
        frame = blank_frame()
        result = fake_result()
        out = overlay.process(frame, result)
        assert out is frame

    def test_stroke_count_increments_while_drawing(self):
        overlay = StrokeOverlay(debounce_frames=2)
        for i in range(5):
            overlay.process(blank_frame(), fake_result(is_drawing=True, frame_idx=i))
        assert overlay.stroke_count == 1

    def test_clear_canvas_resets_strokes(self):
        overlay = StrokeOverlay()
        for i in range(5):
            overlay.process(blank_frame(), fake_result(is_drawing=True, frame_idx=i))
        overlay.clear_canvas()
        assert overlay.stroke_count == 0
        assert overlay.total_points == 0

    def test_on_clear_callback_fires(self):
        calls = []
        overlay = StrokeOverlay(on_clear=lambda: calls.append(1))
        overlay.clear_canvas()
        assert len(calls) == 1

    def test_cycle_color_changes_color(self):
        overlay = StrokeOverlay(initial_color=StrokeColor.WHITE)
        original_color = overlay.current_color
        overlay.cycle_color()
        assert overlay.current_color != original_color

    def test_set_color_is_applied(self):
        overlay = StrokeOverlay()
        overlay.set_color(StrokeColor.RED)
        assert overlay.current_color == StrokeColor.RED

    def test_clear_count_increments(self):
        overlay = StrokeOverlay()
        overlay.clear_canvas()
        overlay.clear_canvas()
        assert overlay.clear_count == 2
