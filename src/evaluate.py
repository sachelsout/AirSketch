"""
src/evaluate.py
Metric computation functions for AirSketch.
All functions return Python floats. Units are documented in the docstring.
"""

import numpy as np


def compute_mpjpe(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """
    Mean Per-Joint Position Error — average Euclidean distance between
    predicted and ground-truth index fingertip coordinates in pixels.

    Target: < 8 px on 224×224 frames.

    Args:
        pred_xy: (N, 2) array of predicted (x, y) coordinates.
        gt_xy:   (N, 2) array of ground-truth (x, y) coordinates.

    Returns:
        MPJPE in pixels (float).
    """
    assert (
        pred_xy.shape == gt_xy.shape
    ), "Shape mismatch between predictions and ground truth"
    return float(np.mean(np.linalg.norm(pred_xy - gt_xy, axis=1)))


def compute_jitter_index(pred_xy: np.ndarray) -> float:
    """
    Stroke Smoothness — mean frame-to-frame displacement variance of
    predicted fingertip positions. Lower = smoother strokes.

    Target: < 2 px² variance.

    Args:
        pred_xy: (N, 2) array of predicted (x, y) coordinates in sequence order.

    Returns:
        Jitter index in px² (float).
    """
    if len(pred_xy) < 2:
        return 0.0
    frame_deltas = np.linalg.norm(np.diff(pred_xy, axis=0), axis=1)
    return float(np.var(frame_deltas))


def compute_gesture_accuracy(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    Binary classification accuracy for draw vs idle gesture.

    Target: > 0.92 (92%).

    Args:
        pred_labels: (N,) array of predicted labels {0: idle, 1: draw}.
        gt_labels:   (N,) array of ground-truth labels {0: idle, 1: draw}.

    Returns:
        Accuracy as a float in [0, 1].
    """
    assert (
        pred_labels.shape == gt_labels.shape
    ), "Shape mismatch between predictions and ground truth"
    return float(np.mean(pred_labels == gt_labels))


def compute_all_metrics(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
) -> dict:
    """
    Compute all three tracked metrics in one call.
    Returns a dict ready to pass directly to ExperimentLogger.log_epoch().

    Args:
        pred_xy:     (N, 2) predicted fingertip coordinates.
        gt_xy:       (N, 2) ground-truth fingertip coordinates.
        pred_labels: (N,) predicted gesture labels {0: idle, 1: draw}.
        gt_labels:   (N,) ground-truth gesture labels {0: idle, 1: draw}.

    Returns:
        Dict with keys: mpjpe, jitter_index, gesture_acc.
    """
    return {
        "mpjpe": compute_mpjpe(pred_xy, gt_xy),
        "jitter_index": compute_jitter_index(pred_xy),
        "gesture_acc": compute_gesture_accuracy(pred_labels, gt_labels),
    }
