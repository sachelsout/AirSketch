"""
tests/test_landmark_extract.py

Unit tests for the MediaPipe extraction pipeline and landmark utilities.
These tests use synthetic data — no real images required.
"""

import numpy as np
from src.utils import interpolate_missing_landmarks, get_valid_sequence_ranges


class TestInterpolateMissingLandmarks:

    def _make_landmarks(self, n: int, nan_frames: list[int]) -> np.ndarray:
        lm = np.random.rand(n, 21, 2).astype(np.float32)
        for i in nan_frames:
            lm[i] = np.nan
        return lm

    def test_no_nans_unchanged(self):
        lm = self._make_landmarks(20, [])
        filled, mask = interpolate_missing_landmarks(lm, max_gap=2)
        assert not np.any(mask)
        np.testing.assert_array_equal(filled, lm)

    def test_single_gap_interpolated(self):
        lm = self._make_landmarks(10, [5])
        filled, mask = interpolate_missing_landmarks(lm, max_gap=2)
        assert not mask[5], "Single-frame gap should be interpolated"
        assert not np.any(np.isnan(filled[5]))

    def test_interpolated_value_is_midpoint(self):
        lm = np.zeros((5, 21, 2), dtype=np.float32)
        lm[0] = 0.0
        lm[2] = np.nan
        lm[4] = 1.0
        filled, _ = interpolate_missing_landmarks(lm, max_gap=2)
        assert not np.any(np.isnan(filled[2]))

    def test_long_gap_not_interpolated(self):
        lm = self._make_landmarks(20, [5, 6, 7, 8, 9])
        filled, mask = interpolate_missing_landmarks(lm, max_gap=2)
        assert np.all(mask[5:10]), "Gap > max_gap should remain NaN"

    def test_gap_at_start_not_interpolated(self):
        lm = self._make_landmarks(10, [0])
        filled, mask = interpolate_missing_landmarks(lm, max_gap=2)
        assert mask[0], "Gap at start with no left anchor should remain NaN"

    def test_gap_at_end_not_interpolated(self):
        lm = self._make_landmarks(10, [9])
        filled, mask = interpolate_missing_landmarks(lm, max_gap=2)
        assert mask[9], "Gap at end with no right anchor should remain NaN"


class TestGetValidSequenceRanges:

    def _make_mask(self, n: int, nan_frames: list[int]) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        for i in nan_frames:
            mask[i] = True
        return mask

    def test_no_nans_returns_full_range(self):
        mask = self._make_mask(100, [])
        ranges = get_valid_sequence_ranges(mask, window_size=16)
        assert ranges == [(0, 100)]

    def test_short_sequence_excluded(self):
        mask = self._make_mask(10, [])
        ranges = get_valid_sequence_ranges(mask, window_size=16)
        assert ranges == []

    def test_gap_splits_into_two_ranges(self):
        mask = self._make_mask(60, list(range(25, 35)))
        ranges = get_valid_sequence_ranges(mask, window_size=16)
        assert len(ranges) == 2
        assert ranges[0] == (0, 25)
        assert ranges[1] == (35, 60)

    def test_all_nan_returns_empty(self):
        mask = np.ones(50, dtype=bool)
        ranges = get_valid_sequence_ranges(mask, window_size=16)
        assert ranges == []
