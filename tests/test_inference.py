"""
tests/test_inference.py

Unit tests for the real-time inference pipeline components.
Tests run on CPU using a tiny ONNX model -- no webcam required.
"""

from pathlib import Path

import numpy as np

from src.model import STTransformer, STTransformerConfig
from scripts.export_onnx import export_to_onnx
from src.inference import (
    FrameBuffer,
    InferenceResult,
    LoopStats,
    ONNXPredictor,
    NUM_LANDMARKS,
)


# -- Fixtures ------------------------------------------------------------------


def make_tiny_onnx(tmp_path: Path) -> Path:
    """Export a tiny (hidden_dim=32) model to ONNX for testing."""
    cfg = STTransformerConfig(hidden_dim=32, num_layers=1, num_heads=4)
    model = STTransformer(cfg)
    model.eval()
    out = tmp_path / "tiny.onnx"
    export_to_onnx(model, out)
    return out


def random_landmarks() -> np.ndarray:
    """Return a valid (21, 2) float32 landmark array in [0, 1]."""
    return np.random.rand(NUM_LANDMARKS, 2).astype(np.float32)


def fake_result(**kwargs) -> InferenceResult:
    gesture = kwargs.get("gesture", 0)
    defaults = dict(
        pred_xy=np.array([0.5, 0.5], dtype=np.float32),
        gesture=gesture,
        gesture_conf=0.8,
        is_drawing=gesture == 1,  # derive from gesture
        landmark_detected=True,
        latency_ms=20.0,
        frame_idx=1,
        timestamp_ms=1000.0,
    )
    defaults.update(kwargs)
    return InferenceResult(**defaults)


# -- FrameBuffer tests ---------------------------------------------------------


class TestFrameBuffer:

    def test_initial_length_is_window_size(self):
        buf = FrameBuffer(window_size=16)
        arr = buf.as_model_input()
        assert arr.shape == (1, 16, 42)

    def test_initial_buffer_all_nan(self):
        buf = FrameBuffer(window_size=4)
        arr = buf.as_model_input()
        assert np.all(np.isnan(arr)), "Initial buffer should be all NaN"

    def test_push_real_landmark_replaces_nan(self):
        buf = FrameBuffer(window_size=4)
        lm = random_landmarks()
        buf.push(lm)
        arr = buf.as_model_input()
        # Last frame (most recent) should not be NaN
        assert not np.any(np.isnan(arr[0, -1]))

    def test_push_none_stores_nan(self):
        buf = FrameBuffer(window_size=4)
        # Push real frames then a None
        for _ in range(3):
            buf.push(random_landmarks())
        buf.push(None)
        arr = buf.as_model_input()
        assert np.any(np.isnan(arr[0, -1])), "None push should store NaN"

    def test_buffer_length_stays_fixed_after_many_pushes(self):
        buf = FrameBuffer(window_size=8)
        for _ in range(50):
            buf.push(random_landmarks())
        arr = buf.as_model_input()
        assert arr.shape == (1, 8, 42), f"Expected (1, 8, 42), got {arr.shape}"

    def test_model_input_shape(self):
        buf = FrameBuffer(window_size=16)
        for _ in range(16):
            buf.push(random_landmarks())
        arr = buf.as_model_input()
        assert arr.shape == (1, 16, 42)
        assert arr.dtype == np.float32

    def test_is_ready_false_before_window_frames(self):
        buf = FrameBuffer(window_size=16)
        for i in range(15):
            buf.push(random_landmarks())
            assert not buf.is_ready, f"Should not be ready after {i+1} frames"

    def test_is_ready_true_after_window_frames(self):
        buf = FrameBuffer(window_size=16)
        for _ in range(16):
            buf.push(random_landmarks())
        assert buf.is_ready

    def test_chronological_order_preserved(self):
        """Most recently pushed frame must be at index [-1]."""
        buf = FrameBuffer(window_size=4)
        sentinel = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
        sentinel[:] = 0.99  # unique value
        for _ in range(3):
            buf.push(random_landmarks())
        buf.push(sentinel)
        arr = buf.as_model_input()
        np.testing.assert_allclose(arr[0, -1, 0], 0.99, atol=1e-5)

    def test_reset_clears_to_nan(self):
        buf = FrameBuffer(window_size=8)
        for _ in range(8):
            buf.push(random_landmarks())
        buf.reset()
        arr = buf.as_model_input()
        assert np.all(np.isnan(arr)), "After reset, buffer should be all NaN"
        assert not buf.is_ready, "After reset, is_ready must be False"

    def test_nan_fraction_all_nan(self):
        buf = FrameBuffer(window_size=4)
        assert buf.nan_fraction == 1.0

    def test_nan_fraction_half_real(self):
        buf = FrameBuffer(window_size=4)
        buf.push(random_landmarks())
        buf.push(random_landmarks())
        # 2 real frames pushed into a 4-frame buffer (initialized with NaN)
        # buffer now has 2 NaN + 2 real
        assert buf.nan_fraction == 0.5


# -- ONNXPredictor tests -------------------------------------------------------


class TestONNXPredictor:

    def test_predict_output_shapes(self, tmp_path):
        onnx_path = make_tiny_onnx(tmp_path)
        pred = ONNXPredictor(onnx_path)
        buf = np.random.rand(1, 16, 42).astype(np.float32)
        xy, ges, conf = pred.predict(buf)
        assert xy.shape == (2,), f"pred_xy shape: {xy.shape}"
        assert isinstance(ges, int)
        assert isinstance(conf, float)

    def test_pred_xy_in_unit_range(self, tmp_path):
        onnx_path = make_tiny_onnx(tmp_path)
        pred = ONNXPredictor(onnx_path)
        for _ in range(20):
            buf = np.random.rand(1, 16, 42).astype(np.float32)
            xy, _, _ = pred.predict(buf)
            assert xy.min() >= 0.0 and xy.max() <= 1.0

    def test_gesture_is_binary(self, tmp_path):
        onnx_path = make_tiny_onnx(tmp_path)
        pred = ONNXPredictor(onnx_path)
        for _ in range(10):
            buf = np.random.rand(1, 16, 42).astype(np.float32)
            _, ges, _ = pred.predict(buf)
            assert ges in (0, 1), f"Gesture must be 0 or 1, got {ges}"

    def test_gesture_conf_in_unit_range(self, tmp_path):
        onnx_path = make_tiny_onnx(tmp_path)
        pred = ONNXPredictor(onnx_path)
        buf = np.random.rand(1, 16, 42).astype(np.float32)
        _, _, conf = pred.predict(buf)
        assert 0.0 <= conf <= 1.0

    def test_deterministic_on_same_input(self, tmp_path):
        """Same input must produce identical output every time."""
        onnx_path = make_tiny_onnx(tmp_path)
        pred = ONNXPredictor(onnx_path)
        buf = np.random.rand(1, 16, 42).astype(np.float32)
        xy1, g1, c1 = pred.predict(buf)
        xy2, g2, c2 = pred.predict(buf)
        np.testing.assert_array_equal(xy1, xy2)
        assert g1 == g2
        assert c1 == c2


# -- LoopStats tests -----------------------------------------------------------


class TestLoopStats:

    def test_fps_increases_with_frames(self):
        import time

        stats = LoopStats()
        time.sleep(0.05)
        for i in range(10):
            stats.update(fake_result(frame_idx=i, latency_ms=20.0))
        assert stats.fps > 0

    def test_mean_latency_computed_correctly(self):
        stats = LoopStats()
        for i in range(4):
            stats.update(fake_result(frame_idx=i, latency_ms=float(10 * (i + 1))))
        # Latencies: 10, 20, 30, 40 -- mean = 25
        assert abs(stats.mean_latency_ms - 25.0) < 0.01

    def test_detection_rate_all_detected(self):
        stats = LoopStats()
        for i in range(5):
            stats.update(fake_result(frame_idx=i, landmark_detected=True))
        assert stats.detection_rate == 1.0

    def test_detection_rate_none_detected(self):
        stats = LoopStats()
        for i in range(5):
            stats.update(fake_result(frame_idx=i, landmark_detected=False))
        assert stats.detection_rate == 0.0

    def test_over_budget_counted_correctly(self):
        stats = LoopStats()
        stats.update(fake_result(frame_idx=0, latency_ms=20.0))  # under
        stats.update(fake_result(frame_idx=1, latency_ms=60.0))  # over
        stats.update(fake_result(frame_idx=2, latency_ms=55.0))  # over
        assert stats.latency_over_budget == 2

    def test_summary_line_is_string(self):
        stats = LoopStats()
        stats.update(fake_result())
        assert isinstance(stats.summary_line(), str)


# -- InferenceResult tests -----------------------------------------------------


class TestInferenceResult:

    def test_is_drawing_matches_gesture(self):
        r_draw = fake_result(gesture=1)
        r_idle = fake_result(gesture=0)
        assert r_draw.is_drawing is True
        assert r_idle.is_drawing is False

    def test_fields_accessible(self):
        r = fake_result()
        assert hasattr(r, "pred_xy")
        assert hasattr(r, "gesture")
        assert hasattr(r, "latency_ms")
        assert hasattr(r, "frame_idx")
        assert hasattr(r, "timestamp_ms")
