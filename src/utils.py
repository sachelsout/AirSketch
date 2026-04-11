"""Shared helper utilities for normalization, logging, and common tasks."""

import numpy as np


def interpolate_missing_landmarks(
    landmarks: np.ndarray,
    max_gap: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill short gaps (NaN frames) in a landmark sequence by linear interpolation.
    Gaps longer than max_gap are left as NaN — handled by the dataset builder.

    Args:
        landmarks: (N, 21, 2) float32 array where failed frames contain NaN.
        max_gap:   Maximum consecutive NaN frames to interpolate over.

    Returns:
        filled:    (N, 21, 2) array with short gaps filled.
        nan_mask:  (N,) bool array — True where landmarks are still NaN after filling.
    """
    filled = landmarks.copy()
    N = len(filled)
    nan_mask = np.all(np.isnan(filled[:, :, 0]), axis=-1)

    i = 0
    while i < N:
        if not nan_mask[i]:
            i += 1
            continue

        gap_start = i
        while i < N and nan_mask[i]:
            i += 1
        gap_end = i

        gap_len = gap_end - gap_start
        has_left_anchor = gap_start > 0
        has_right_anchor = gap_end < N

        if gap_len <= max_gap and has_left_anchor and has_right_anchor:
            left = filled[gap_start - 1]
            right = filled[gap_end]

            for offset in range(gap_len):
                t = (offset + 1) / (gap_len + 1)
                filled[gap_start + offset] = (1 - t) * left + t * right
                nan_mask[gap_start + offset] = False

    return filled, nan_mask


def get_valid_sequence_ranges(
    nan_mask: np.ndarray,
    window_size: int = 16,
) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs for all contiguous non-NaN regions
    long enough to produce at least one sliding window of `window_size`.

    Args:
        nan_mask:    (N,) bool array — True where landmarks are NaN.
        window_size: Minimum required sequence length.

    Returns:
        List of (start_idx, end_idx) tuples (end is exclusive).
    """
    ranges = []
    N = len(nan_mask)
    i = 0

    while i < N:
        if nan_mask[i]:
            i += 1
            continue

        run_start = i
        while i < N and not nan_mask[i]:
            i += 1
        run_end = i

        if (run_end - run_start) >= window_size:
            ranges.append((run_start, run_end))

    return ranges
