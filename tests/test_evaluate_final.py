"""
tests/test_evaluate_final.py

Unit tests for metric functions and data utilities in evaluate_final.py.
No model or real dataset required.
"""

import numpy as np

from src.evaluate_final import (
    mpjpe_per_window,
    jitter_index_per_sequence,
    gesture_accuracy,
    IMAGE_SIZE,
)


class TestMpjpePerWindow:

    def test_zero_error_on_identical_inputs(self):
        xy = np.random.rand(50, 2).astype(np.float32)
        errors = mpjpe_per_window(xy, xy)
        np.testing.assert_allclose(errors, 0.0, atol=1e-6)

    def test_known_error_value(self):
        # Predicted is 1/224 away from GT in x -- MPJPE should be 1.0 px
        pred = np.array([[0.5 + 1.0 / IMAGE_SIZE, 0.5]], dtype=np.float32)
        gt = np.array([[0.5, 0.5]], dtype=np.float32)
        err = mpjpe_per_window(pred, gt)
        np.testing.assert_allclose(err, 1.0, atol=1e-4)

    def test_output_shape(self):
        N = 100
        pred = np.random.rand(N, 2).astype(np.float32)
        gt = np.random.rand(N, 2).astype(np.float32)
        assert mpjpe_per_window(pred, gt).shape == (N,)

    def test_values_are_non_negative(self):
        pred = np.random.rand(200, 2).astype(np.float32)
        gt = np.random.rand(200, 2).astype(np.float32)
        assert np.all(mpjpe_per_window(pred, gt) >= 0)

    def test_scales_with_image_size(self):
        # Moving by 0.5 in normalized space = IMAGE_SIZE/2 in pixel space
        pred = np.array([[0.0, 0.0]], dtype=np.float32)
        gt = np.array([[0.5, 0.0]], dtype=np.float32)
        err = mpjpe_per_window(pred, gt)
        np.testing.assert_allclose(err[0], IMAGE_SIZE * 0.5, atol=0.5)


class TestJitterIndexPerSequence:

    def test_zero_jitter_on_static_sequence(self):
        # Same point every frame -- zero displacement variance
        seq = np.full((20, 2), 0.5, dtype=np.float32)
        assert jitter_index_per_sequence(seq) < 1e-6

    def test_higher_jitter_on_noisy_sequence(self):
        smooth = (
            np.linspace(0, 1, 30).reshape(-1, 1).repeat(2, axis=1).astype(np.float32)
        )
        noisy = smooth + np.random.randn(30, 2).astype(np.float32) * 0.05
        assert jitter_index_per_sequence(noisy) > jitter_index_per_sequence(smooth)

    def test_returns_float(self):
        seq = np.random.rand(16, 2).astype(np.float32)
        assert isinstance(jitter_index_per_sequence(seq), float)

    def test_single_frame_returns_zero(self):
        seq = np.array([[0.5, 0.5]], dtype=np.float32)
        assert jitter_index_per_sequence(seq) == 0.0

    def test_non_negative(self):
        for _ in range(20):
            seq = np.random.rand(16, 2).astype(np.float32)
            assert jitter_index_per_sequence(seq) >= 0.0


class TestGestureAccuracy:

    def test_perfect_accuracy(self):
        labels = np.array([0, 1, 1, 0, 1], dtype=np.int64)
        logits = np.array(
            [
                [2.0, -2.0],  # predict 0
                [-2.0, 2.0],  # predict 1
                [-2.0, 2.0],  # predict 1
                [2.0, -2.0],  # predict 0
                [-2.0, 2.0],  # predict 1
            ],
            dtype=np.float32,
        )
        assert gesture_accuracy(logits, labels) == 1.0

    def test_zero_accuracy(self):
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        logits = np.array(
            [
                [-2.0, 2.0],  # predict 1 (wrong)
                [2.0, -2.0],  # predict 0 (wrong)
                [-2.0, 2.0],  # predict 1 (wrong)
                [2.0, -2.0],  # predict 0 (wrong)
            ],
            dtype=np.float32,
        )
        assert gesture_accuracy(logits, labels) == 0.0

    def test_fifty_percent_accuracy(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        logits = np.array(
            [
                [2.0, -2.0],  # correct
                [-2.0, 2.0],  # wrong
                [-2.0, 2.0],  # correct
                [2.0, -2.0],  # wrong
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(gesture_accuracy(logits, labels), 0.5)

    def test_return_type_is_float(self):
        labels = np.zeros(10, dtype=np.int64)
        logits = np.zeros((10, 2), dtype=np.float32)
        logits[:, 0] = 1.0
        assert isinstance(gesture_accuracy(logits, labels), float)

    def test_output_in_unit_range(self):
        for _ in range(20):
            n = np.random.randint(10, 100)
            labels = np.random.randint(0, 2, size=n, dtype=np.int64)
            logits = np.random.randn(n, 2).astype(np.float32)
            acc = gesture_accuracy(logits, labels)
            assert 0.0 <= acc <= 1.0
