"""
src/evaluate_final.py

Final held-out evaluation for AirSketch.

Evaluates the best ONNX model on both the FreiHAND test split and the
custom gesture test set. Produces per-metric aggregate results, per-window
detailed results, failure-case analysis, and publication-ready figures.

Outputs (all written to report/eval/):
    results_summary.json              Aggregate metrics for both splits
    freihand_test_results.csv         Per-window metrics, FreiHAND test
    custom_test_results.csv           Per-window metrics, custom gesture
    figure_metric_histograms.png      Distribution plots for all 3 metrics
    figure_failure_cases.png          Top-20 worst MPJPE windows
    figure_gesture_confusion.png      Confusion matrix for gesture head
    figure_ood_comparison.png         In-distribution vs OOD comparison
    results_table.md                  Markdown table for the final report

Usage:
    python src/evaluate_final.py \
        --model    checkpoints/best_model.optimized.onnx \
        --config   configs/default.yaml

    # Evaluate FreiHAND only (faster -- skips custom gesture loading)
    python src/evaluate_final.py --model checkpoints/best_model.optimized.onnx \
        --skip-custom

    # Debug: evaluate on 500 windows per split
    python src/evaluate_final.py --model checkpoints/best_model.optimized.onnx \
        --max-windows 500
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml

matplotlib.use("Agg")  # headless -- no display required on Zaratan


# -- Constants -----------------------------------------------------------------

IMAGE_SIZE = 224
INDEX_FINGERTIP = 8
WINDOW_SIZE = 16
INPUT_NAME = "sequence"
OUTPUT_NAMES = ["pred_xy", "gesture_logits"]

TARGETS = {
    "mpjpe_px": 8.0,
    "jitter_px2": 2.0,
    "gesture_acc": 0.92,
}

# Color palette for figures (colorblind-safe)
COLORS = {
    "freihand": "#4A90D9",
    "custom": "#E84040",
    "target": "#2DB37A",
    "fail": "#E8A830",
}


# -- ONNX inference session ----------------------------------------------------


def load_onnx_session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def run_batch(
    sess: ort.InferenceSession,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a batch of windows.

    Args:
        sess:    ONNX Runtime session.
        windows: (B, T, 42) float32 batch.

    Returns:
        pred_xy:  (B, 2) float32 -- predicted fingertip in [0, 1].
        ges_logits: (B, 2) float32 -- gesture logits.
    """
    outputs = sess.run(OUTPUT_NAMES, {INPUT_NAME: windows})
    pred_xy = outputs[0]  # (B, 2)
    ges_logits = outputs[1]  # (B, 2)
    return pred_xy, ges_logits


# -- Metric functions ----------------------------------------------------------


def mpjpe_per_window(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
) -> np.ndarray:
    """
    Per-window MPJPE in pixels (Euclidean distance × IMAGE_SIZE).

    Args:
        pred_xy: (N, 2) float32 in [0, 1].
        gt_xy:   (N, 2) float32 in [0, 1].

    Returns:
        (N,) float32 MPJPE values in pixels.
    """
    return np.linalg.norm((pred_xy - gt_xy) * IMAGE_SIZE, axis=1)


def jitter_index_per_sequence(
    pred_xy_seq: np.ndarray,
) -> float:
    """
    Jitter index for one predicted sequence (lower = smoother).

    Computes the variance of frame-to-frame displacement magnitudes.

    Args:
        pred_xy_seq: (T, 2) float32 predicted positions in [0, 1].

    Returns:
        Jitter index in px².
    """
    if len(pred_xy_seq) < 2:
        return 0.0
    deltas = np.linalg.norm(np.diff(pred_xy_seq * IMAGE_SIZE, axis=0), axis=1)
    return float(np.var(deltas))


def gesture_accuracy(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Binary gesture accuracy.

    Args:
        logits: (N, 2) float32 gesture logits.
        labels: (N,) int64 ground-truth {0=idle, 1=draw}.

    Returns:
        Accuracy in [0, 1].
    """
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == labels))


# -- Dataset loader ------------------------------------------------------------


def load_freihand_test_windows(
    config: dict,
    max_windows: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows from the FreiHAND test split.

    Returns:
        windows:  (N, T, 42) float32
        targets:  (N, 2) float32 -- normalized gt fingertip
        gestures: (N,) int64 -- draw/idle labels
    """
    from src.dataset import AirSketchDataset

    lm_path = Path(config["data"]["processed_dir"]) / "freihand" / "landmarks.npy"
    sp_path = Path(config["data"]["splits_dir"]) / "freihand_splits.json"

    with open(sp_path) as f:
        splits = json.load(f)

    ds = AirSketchDataset(
        landmarks_path=lm_path,
        split_indices=splits["test"],
        window_size=WINDOW_SIZE,
        stride=1,
        augment=False,
    )

    n = min(len(ds), max_windows) if max_windows > 0 else len(ds)
    print(f"  FreiHAND test: {n:,} windows from {len(ds):,} available")

    windows = np.zeros((n, WINDOW_SIZE, 42), dtype=np.float32)
    targets = np.zeros((n, 2), dtype=np.float32)
    gestures = np.zeros(n, dtype=np.int64)

    for i in range(n):
        sample = ds[i]
        windows[i] = sample["sequence"].numpy()
        targets[i] = sample["target"].numpy()
        gestures[i] = sample["gesture"].numpy()

    return windows, targets, gestures


def load_custom_test_windows(
    config: dict,
    max_windows: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build sliding windows from all custom gesture test sessions.

    Returns:
        windows:      (N, T, 42) float32
        targets:      (N, 2) float32
        gestures:     (N,) int64
        session_ids:  (N,) list of session name strings
    """
    from src.utils import (
        get_valid_sequence_ranges,
        interpolate_missing_landmarks,
    )

    sp_path = Path(config["data"]["splits_dir"]) / "custom_test_split.json"
    with open(sp_path) as f:
        split = json.load(f)

    all_windows: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_gestures: list[int] = []
    all_sessions: list[str] = []

    for session_name, session_data in split["sessions"].items():
        lm_path = Path(session_data["landmarks_path"])
        ges_path = Path(session_data["gesture_labels_path"])

        landmarks = np.load(lm_path)  # (N, 21, 2)
        ges_labels = np.load(ges_path)  # (N,)

        landmarks, nan_mask = interpolate_missing_landmarks(landmarks, max_gap=2)
        valid_ranges = get_valid_sequence_ranges(nan_mask, window_size=WINDOW_SIZE + 1)

        for start, end in valid_ranges:
            idx = start
            while idx + WINDOW_SIZE + 1 <= end:
                window = landmarks[idx : idx + WINDOW_SIZE]  # (T, 21, 2)
                target = landmarks[idx + WINDOW_SIZE, INDEX_FINGERTIP]  # (2,)
                label = int(ges_labels[idx + WINDOW_SIZE])

                flat = window.reshape(WINDOW_SIZE, 42)
                all_windows.append(flat)
                all_targets.append(target)
                all_gestures.append(label)
                all_sessions.append(session_name)

                idx += 1

                if max_windows > 0 and len(all_windows) >= max_windows:
                    break
            if max_windows > 0 and len(all_windows) >= max_windows:
                break
        if max_windows > 0 and len(all_windows) >= max_windows:
            break

    n = len(all_windows)
    print(f"  Custom test:  {n:,} windows from " f"{len(split['sessions'])} sessions")

    return (
        np.stack(all_windows, axis=0).astype(np.float32),
        np.stack(all_targets, axis=0).astype(np.float32),
        np.array(all_gestures, dtype=np.int64),
        all_sessions,
    )


# -- Evaluation runner ---------------------------------------------------------


def _apply_ema(pred_xy: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply exponential moving average smoothing over a sequence of predictions.

    Args:
        pred_xy: (N, 2) float32 raw predictions in order.
        alpha:   EMA weight for new observation (0 < alpha <= 1).
                 alpha=1.0 means no smoothing.

    Returns:
        (N, 2) float32 smoothed predictions.
    """
    smoothed = pred_xy.copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * pred_xy[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def evaluate_split(
    sess: ort.InferenceSession,
    windows: np.ndarray,
    targets: np.ndarray,
    gestures: np.ndarray,
    split_name: str,
    batch_size: int = 256,
    ema_alpha: float = 0.8,
    session_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run inference and compute per-window metrics for one test split.

    Processes windows in batches to avoid OOM on large test sets.

    Returns:
        DataFrame with columns:
            window_idx, pred_x, pred_y, gt_x, gt_y,
            mpjpe_px, pred_gesture, gt_gesture, correct_gesture
    """
    N = len(windows)
    print(f"\nEvaluating {split_name}: {N:,} windows ...")

    all_pred_xy = np.zeros((N, 2), dtype=np.float32)
    all_ges_logits = np.zeros((N, 2), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = windows[start:end]
        px, gl = run_batch(sess, batch)
        all_pred_xy[start:end] = px
        all_ges_logits[start:end] = gl

        if (start // batch_size) % 20 == 0:
            pct = end / N * 100
            print(f"  {end:>7,} / {N:,}  ({pct:.0f}%)", end="\r", flush=True)

    print()

    # EMA smoothing + jitter: apply per-session when session IDs are available,
    # otherwise treat the whole sequence as one (e.g. FreiHAND contiguous split).
    if session_ids is not None:
        session_arr = np.array(session_ids)
        unique_sessions = list(dict.fromkeys(session_ids))  # preserves order

        if ema_alpha < 1.0:
            for sid in unique_sessions:
                mask = session_arr == sid
                all_pred_xy[mask] = _apply_ema(all_pred_xy[mask], ema_alpha)
            print(f"  EMA smoothing applied per-session (alpha={ema_alpha})")

        jitter_vals = [
            jitter_index_per_sequence(all_pred_xy[session_arr == sid])
            for sid in unique_sessions
        ]
        jitter = float(np.mean(jitter_vals))
    else:
        if ema_alpha < 1.0:
            all_pred_xy = _apply_ema(all_pred_xy, ema_alpha)
            print(f"  EMA smoothing applied (alpha={ema_alpha})")
        jitter = jitter_index_per_sequence(all_pred_xy)

    # Per-window metrics
    mpjpe = mpjpe_per_window(all_pred_xy, targets)  # (N,)
    preds = np.argmax(all_ges_logits, axis=1)  # (N,)
    correct = (preds == gestures).astype(np.int32)  # (N,)

    df = pd.DataFrame(
        {
            "window_idx": np.arange(N),
            "pred_x": all_pred_xy[:, 0],
            "pred_y": all_pred_xy[:, 1],
            "gt_x": targets[:, 0],
            "gt_y": targets[:, 1],
            "mpjpe_px": mpjpe,
            "pred_gesture": preds,
            "gt_gesture": gestures,
            "correct_gesture": correct,
        }
    )

    # Aggregate statistics
    mean_mpjpe = float(mpjpe.mean())
    p50_mpjpe = float(np.percentile(mpjpe, 50))
    p95_mpjpe = float(np.percentile(mpjpe, 95))
    ges_acc = float(correct.mean())

    # Jitter was already computed above (per-session or global)
    print(
        f"  MPJPE mean:      {mean_mpjpe:.2f} px  "
        f"({'PASS' if mean_mpjpe < TARGETS['mpjpe_px'] else 'FAIL'})"
    )
    print(f"  MPJPE P50/P95:   {p50_mpjpe:.2f} / {p95_mpjpe:.2f} px")
    print(
        f"  Jitter index:    {jitter:.3f} px²  "
        f"({'PASS' if jitter < TARGETS['jitter_px2'] else 'FAIL'})"
    )
    print(
        f"  Gesture acc:     {ges_acc*100:.2f}%  "
        f"({'PASS' if ges_acc > TARGETS['gesture_acc'] else 'FAIL'})"
    )

    df.attrs["split_name"] = split_name
    df.attrs["mean_mpjpe"] = mean_mpjpe
    df.attrs["p50_mpjpe"] = p50_mpjpe
    df.attrs["p95_mpjpe"] = p95_mpjpe
    df.attrs["jitter"] = jitter
    df.attrs["ges_acc"] = ges_acc

    return df


# -- Figures -------------------------------------------------------------------


def plot_metric_histograms(
    df_fh: pd.DataFrame,
    df_cust: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    """
    Three-panel histogram: MPJPE, jitter (approximated per-window),
    and gesture accuracy per session.

    Each panel shows both splits (FreiHAND blue, custom red) and the
    target threshold as a green vertical line.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -- Panel 1: MPJPE distribution ------------------------------------------
    ax = axes[0]
    ax.hist(
        df_fh["mpjpe_px"],
        bins=60,
        color=COLORS["freihand"],
        alpha=0.7,
        label="FreiHAND test",
        density=True,
    )
    if df_cust is not None:
        ax.hist(
            df_cust["mpjpe_px"],
            bins=60,
            color=COLORS["custom"],
            alpha=0.7,
            label="Custom gesture",
            density=True,
        )
    ax.axvline(
        TARGETS["mpjpe_px"],
        color=COLORS["target"],
        linestyle="--",
        linewidth=2,
        label=f"Target: {TARGETS['mpjpe_px']} px",
    )
    ax.set_xlabel("MPJPE (px)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("MPJPE Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, min(df_fh["mpjpe_px"].quantile(0.99) * 1.2, 60))

    # Annotate mean values
    fh_mean = df_fh["mpjpe_px"].mean()
    ax.axvline(
        fh_mean,
        color=COLORS["freihand"],
        linestyle=":",
        linewidth=1.5,
        label=f"FH mean: {fh_mean:.1f}",
    )
    if df_cust is not None:
        cust_mean = df_cust["mpjpe_px"].mean()
        ax.axvline(
            cust_mean,
            color=COLORS["custom"],
            linestyle=":",
            linewidth=1.5,
            label=f"Custom mean: {cust_mean:.1f}",
        )
    ax.legend(fontsize=8)

    # -- Panel 2: Per-window displacement variance (proxy for jitter) ---------
    ax = axes[1]

    def per_window_disp(df):
        """Frame-to-frame displacement within each window (px)."""
        preds = df[["pred_x", "pred_y"]].values * IMAGE_SIZE
        # Approximate: use consecutive predicted positions as a sequence
        diffs = np.linalg.norm(np.diff(preds, axis=0), axis=1)
        return diffs

    disp_fh = per_window_disp(df_fh)
    ax.hist(
        disp_fh,
        bins=50,
        color=COLORS["freihand"],
        alpha=0.7,
        label="FreiHAND test",
        density=True,
    )
    if df_cust is not None:
        disp_cu = per_window_disp(df_cust)
        ax.hist(
            disp_cu,
            bins=50,
            color=COLORS["custom"],
            alpha=0.7,
            label="Custom gesture",
            density=True,
        )
    ax.axvline(
        np.sqrt(TARGETS["jitter_px2"]),
        color=COLORS["target"],
        linestyle="--",
        linewidth=2,
        label=f"Target (sqrt): {np.sqrt(TARGETS['jitter_px2']):.1f} px",
    )
    ax.set_xlabel("Frame displacement (px)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Stroke Smoothness", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 20)

    # -- Panel 3: Gesture accuracy per session (custom only) ------------------
    ax = axes[2]
    if df_cust is not None and "session" in df_cust.columns:
        session_accs = (
            df_cust.groupby("session")["correct_gesture"].mean().sort_values()
        )
        bars = ax.barh(
            range(len(session_accs)),
            session_accs.values * 100,
            color=COLORS["custom"],
            alpha=0.85,
        )
        ax.axvline(
            TARGETS["gesture_acc"] * 100,
            color=COLORS["target"],
            linestyle="--",
            linewidth=2,
            label=f"Target: {TARGETS['gesture_acc']*100:.0f}%",
        )
        ax.set_yticks(range(len(session_accs)))
        ax.set_yticklabels([s[-12:] for s in session_accs.index], fontsize=9)
        ax.set_xlabel("Gesture accuracy (%)", fontsize=11)
        ax.set_title("Gesture Accuracy by Session", fontsize=12)
        ax.set_xlim(0, 105)
        ax.legend(fontsize=9)
        for i, (bar, acc) in enumerate(zip(bars, session_accs.values)):
            ax.text(
                bar.get_width() + 0.5, i, f"{acc*100:.1f}%", va="center", fontsize=8
            )
    else:
        # Fall back to overall accuracy bar
        accs = {
            "FreiHAND test": df_fh["correct_gesture"].mean() * 100,
        }
        if df_cust is not None:
            accs["Custom gesture"] = df_cust["correct_gesture"].mean() * 100
        ax.bar(
            accs.keys(),
            accs.values(),
            color=[COLORS["freihand"], COLORS["custom"]][: len(accs)],
            alpha=0.85,
        )
        ax.axhline(
            TARGETS["gesture_acc"] * 100,
            color=COLORS["target"],
            linestyle="--",
            linewidth=2,
            label=f"Target: {TARGETS['gesture_acc']*100:.0f}%",
        )
        ax.set_ylabel("Gesture Accuracy (%)", fontsize=11)
        ax.set_title("Gesture Classification Accuracy", fontsize=12)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)

    plt.suptitle("AirSketch Held-Out Test Set Metrics", fontsize=14, y=1.02)
    plt.tight_layout()
    out = out_dir / "figure_metric_histograms.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_failure_cases(
    df: pd.DataFrame,
    windows: np.ndarray,
    split_name: str,
    out_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Plot the top_n windows with the highest MPJPE as failure cases.

    Each panel shows the window's predicted trajectory vs ground-truth
    fingertip position as a 2D scatter in normalized coords.
    """
    worst = df.nlargest(top_n, "mpjpe_px").reset_index(drop=True)
    ncols = 5
    nrows = (top_n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, (_, row) in enumerate(worst.iterrows()):
        ax = axes[i]
        idx = int(row["window_idx"])

        # Extract landmark positions for index fingertip across the window
        window_flat = windows[idx]  # (T, 42)
        window_3d = window_flat.reshape(WINDOW_SIZE, 21, 2)
        tips_traj = window_3d[:, INDEX_FINGERTIP, :]  # (T, 2)

        # Plot trajectory of fingertip across the window
        ax.plot(
            tips_traj[:, 0],
            tips_traj[:, 1],
            "o-",
            color=COLORS["freihand"],
            markersize=3,
            linewidth=1,
            alpha=0.7,
            label="Input traj.",
        )

        # Ground-truth target (next frame)
        ax.scatter(
            row["gt_x"],
            row["gt_y"],
            c=COLORS["target"],
            s=80,
            zorder=5,
            marker="*",
            label="GT target",
        )

        # Model prediction
        ax.scatter(
            row["pred_x"],
            row["pred_y"],
            c=COLORS["fail"],
            s=60,
            zorder=6,
            marker="X",
            label="Prediction",
        )

        # Error arrow
        ax.annotate(
            "",
            xy=(row["pred_x"], row["pred_y"]),
            xytext=(row["gt_x"], row["gt_y"]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.8),
        )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.invert_yaxis()
        ax.set_title(
            f"#{i+1}  MPJPE={row['mpjpe_px']:.1f}px\n"
            f"ges={row['gt_gesture']}→{row['pred_gesture']}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Top {top_n} Failure Cases — {split_name}\n"
        f"(orange X = prediction, green * = ground truth, "
        f"blue = fingertip trajectory)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    out = out_dir / f"figure_failure_cases_{split_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_gesture_confusion(
    df_fh: pd.DataFrame,
    df_cust: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    """
    2×2 confusion matrix for gesture classification on both splits.
    Rows = ground truth, columns = prediction.
    """
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2 if df_cust is not None else 1, figsize=(8, 4))
    if df_cust is None:
        axes = [axes]

    datasets = [("FreiHAND test", df_fh, COLORS["freihand"])]
    if df_cust is not None:
        datasets.append(("Custom gesture", df_cust, COLORS["custom"]))

    for ax, (name, df, color) in zip(axes, datasets):
        cm = confusion_matrix(
            df["gt_gesture"],
            df["pred_gesture"],
            labels=[0, 1],
        )

        # Normalize rows
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm.astype(np.float64),
            row_sums,
            out=np.zeros_like(cm, dtype=np.float64),
            where=row_sums != 0,
        )

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)

        labels = ["Idle (0)", "Draw (1)"]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Ground Truth", fontsize=11)
        ax.set_title(f"Gesture Confusion Matrix\n{name}", fontsize=11)

        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    f"{cm_norm[r,c]:.2f}\n({cm[r,c]:,})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if cm_norm[r, c] > 0.5 else "black",
                )

    plt.tight_layout()
    out = out_dir / "figure_gesture_confusion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_ood_comparison(
    fh_mpjpe: np.ndarray,
    cust_mpjpe: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Side-by-side violin plot comparing MPJPE distribution on
    in-distribution (FreiHAND) vs OOD (custom gesture) data.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    data = [
        fh_mpjpe[fh_mpjpe < 40],  # clip extreme outliers for display
        cust_mpjpe[cust_mpjpe < 40],
    ]
    labels = [
        "FreiHAND test\n(in-distribution)",
        "Custom gesture\n(out-of-distribution)",
    ]
    colors = [COLORS["freihand"], COLORS["custom"]]

    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=True)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for part_name in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[part_name].set_edgecolor("gray")

    ax.axhline(
        TARGETS["mpjpe_px"],
        color=COLORS["target"],
        linestyle="--",
        linewidth=2,
        label=f"Target: {TARGETS['mpjpe_px']} px",
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("MPJPE (px)", fontsize=11)
    ax.set_title("In-Distribution vs Out-of-Distribution MPJPE", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 40)

    # Annotate medians
    for i, arr in enumerate(data, start=1):
        med = np.median(arr)
        ax.text(
            i,
            med + 0.8,
            f"median\n{med:.1f}px",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    plt.tight_layout()
    out = out_dir / "figure_ood_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# -- Results table -------------------------------------------------------------


def write_results_table(
    fh_stats: dict,
    cust_stats: dict | None,
    out_dir: Path,
) -> None:
    """
    Write a Markdown results table for the final report.
    """

    def check(val, target, lower_better=True):
        passed = (val < target) if lower_better else (val > target)
        return (
            f"{'**' if passed else ''}{val:.2f}{'**' if passed else ''} "
            f"{'✓' if passed else '✗'}"
        )

    custom_p50 = f"{cust_stats['p50_mpjpe']:.2f}" if cust_stats else "N/A"
    custom_p95 = f"{cust_stats['p95_mpjpe']:.2f}" if cust_stats else "N/A"

    lines = [
        "# AirSketch Quantitative Evaluation Results\n",
        "Targets: MPJPE < 8 px | Jitter < 2 px² | Gesture Acc > 92%\n",
        "Bold values indicate target met.\n",
        "| Metric | Target | FreiHAND Test | Custom Gesture |",
        "|--------|--------|---------------|----------------|",
        f"| MPJPE (px) ↓ | < {TARGETS['mpjpe_px']} | "
        f"{check(fh_stats['mean_mpjpe'], TARGETS['mpjpe_px'])} | "
        f"{check(cust_stats['mean_mpjpe'], TARGETS['mpjpe_px']) if cust_stats else 'N/A'} |",
        f"| MPJPE P50 (px) | — | {fh_stats['p50_mpjpe']:.2f} | " f"{custom_p50} |",
        f"| MPJPE P95 (px) | — | {fh_stats['p95_mpjpe']:.2f} | " f"{custom_p95} |",
        f"| Jitter Index (px²) ↓ | < {TARGETS['jitter_px2']} | "
        f"{check(fh_stats['jitter'], TARGETS['jitter_px2'])} | "
        f"{check(cust_stats['jitter'], TARGETS['jitter_px2']) if cust_stats else 'N/A'} |",
        f"| Gesture Acc (%) ↑ | > {TARGETS['gesture_acc']*100:.0f} | "
        f"{check(fh_stats['ges_acc']*100, TARGETS['gesture_acc']*100, lower_better=False)} | "
        f"{check(cust_stats['ges_acc']*100, TARGETS['gesture_acc']*100, lower_better=False) if cust_stats else 'N/A'} |",
        "",
        "## Notes",
        "- FreiHAND test = in-distribution benchmark (19,536 windows)",
        (
            f"- Custom gesture = OOD benchmark ({cust_stats['n_windows']:,} windows "
            f"across 4 sessions)"
            if cust_stats
            else ""
        ),
        "- MPJPE computed in pixel space on 224×224 frames",
        "- Jitter index = variance of frame-to-frame displacement magnitudes",
        "- Gesture accuracy = binary draw/idle classification over all windows",
    ]

    out = out_dir / "results_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out}")

    # Also print to console
    print("\n" + "\n".join(lines[3:9]))


# -- Summary JSON --------------------------------------------------------------


def write_summary_json(
    fh_stats: dict,
    cust_stats: dict | None,
    model_path: Path,
    out_dir: Path,
) -> None:
    summary = {
        "model": str(model_path),
        "targets": TARGETS,
        "freihand_test": {
            **fh_stats,
            "pass_mpjpe": fh_stats["mean_mpjpe"] < TARGETS["mpjpe_px"],
            "pass_jitter": fh_stats["jitter"] < TARGETS["jitter_px2"],
            "pass_gesture": fh_stats["ges_acc"] > TARGETS["gesture_acc"],
        },
    }
    if cust_stats:
        summary["custom_gesture"] = {
            **cust_stats,
            "pass_mpjpe": cust_stats["mean_mpjpe"] < TARGETS["mpjpe_px"],
            "pass_jitter": cust_stats["jitter"] < TARGETS["jitter_px2"],
            "pass_gesture": cust_stats["ges_acc"] > TARGETS["gesture_acc"],
        }

    out = out_dir / "results_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")


# -- Main ----------------------------------------------------------------------


def main(
    model_path: str,
    config_path: str,
    out_dir: str,
    skip_custom: bool,
    max_windows: int,
    batch_size: int,
    ema_alpha: float = 0.8,
) -> None:
    model_path = Path(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loading ONNX session: {model_path}")
    sess = load_onnx_session(model_path)
    print("  Session loaded.")

    # -- FreiHAND test ---------------------------------------------------------
    print("\nLoading FreiHAND test split ...")
    fh_windows, fh_targets, fh_gestures = load_freihand_test_windows(
        config, max_windows
    )
    df_fh = evaluate_split(
        sess,
        fh_windows,
        fh_targets,
        fh_gestures,
        "FreiHAND test",
        batch_size,
        ema_alpha,
    )
    df_fh.to_csv(out_dir / "freihand_test_results.csv", index=False)

    fh_stats = {
        "n_windows": len(df_fh),
        "mean_mpjpe": df_fh.attrs["mean_mpjpe"],
        "p50_mpjpe": df_fh.attrs["p50_mpjpe"],
        "p95_mpjpe": df_fh.attrs["p95_mpjpe"],
        "jitter": df_fh.attrs["jitter"],
        "ges_acc": df_fh.attrs["ges_acc"],
    }

    # -- Custom gesture test ---------------------------------------------------
    df_cust = None
    cust_stats = None

    if not skip_custom:
        custom_split = Path(config["data"]["splits_dir"]) / "custom_test_split.json"
        if custom_split.exists():
            print("\nLoading custom gesture test split ...")
            cust_windows, cust_targets, cust_gestures, cust_sessions = (
                load_custom_test_windows(config, max_windows)
            )

            df_cust = evaluate_split(
                sess,
                cust_windows,
                cust_targets,
                cust_gestures,
                "Custom gesture",
                batch_size,
                ema_alpha,
                cust_sessions,
            )
            df_cust["session"] = cust_sessions[: len(df_cust)]
            df_cust.to_csv(out_dir / "custom_test_results.csv", index=False)

            cust_stats = {
                "n_windows": len(df_cust),
                "mean_mpjpe": df_cust.attrs["mean_mpjpe"],
                "p50_mpjpe": df_cust.attrs["p50_mpjpe"],
                "p95_mpjpe": df_cust.attrs["p95_mpjpe"],
                "jitter": df_cust.attrs["jitter"],
                "ges_acc": df_cust.attrs["ges_acc"],
            }
        else:
            print(f"\nWARNING: {custom_split} not found -- skipping custom eval.")

    # -- Figures ---------------------------------------------------------------
    print("\nGenerating figures ...")
    plot_metric_histograms(df_fh, df_cust, out_dir)
    plot_failure_cases(df_fh, fh_windows, "FreiHAND test", out_dir)
    if df_cust is not None:
        plot_failure_cases(df_cust, cust_windows, "Custom gesture", out_dir)
    plot_gesture_confusion(df_fh, df_cust, out_dir)
    if df_cust is not None:
        plot_ood_comparison(
            df_fh["mpjpe_px"].values,
            df_cust["mpjpe_px"].values,
            out_dir,
        )

    # -- Results table + JSON --------------------------------------------------
    print("\nWriting results table and summary JSON ...")
    write_results_table(fh_stats, cust_stats, out_dir)
    write_summary_json(fh_stats, cust_stats, model_path, out_dir)

    # -- Final pass/fail summary -----------------------------------------------
    print(f"\n{'--'*30}")
    print("  FINAL EVALUATION RESULTS")
    print(f"{'--'*30}")

    all_passed = True
    for split_name, stats in [
        ("FreiHAND test", fh_stats),
        ("Custom gesture", cust_stats),
    ]:
        if stats is None:
            continue
        p1 = stats["mean_mpjpe"] < TARGETS["mpjpe_px"]
        p2 = stats["jitter"] < TARGETS["jitter_px2"]
        p3 = stats["ges_acc"] > TARGETS["gesture_acc"]
        all_passed = all_passed and p1 and p2 and p3

        print(f"\n  {split_name}:")
        print(
            f"    MPJPE:   {stats['mean_mpjpe']:.2f} px  "
            f"({'PASS' if p1 else 'FAIL'} < {TARGETS['mpjpe_px']})"
        )
        print(
            f"    Jitter:  {stats['jitter']:.3f} px²  "
            f"({'PASS' if p2 else 'FAIL'} < {TARGETS['jitter_px2']})"
        )
        print(
            f"    Gesture: {stats['ges_acc']*100:.2f}%  "
            f"({'PASS' if p3 else 'FAIL'} > {TARGETS['gesture_acc']*100:.0f}%)"
        )

    print(f"\n{'--'*30}")
    print(f"  Overall: {'ALL TARGETS MET' if all_passed else 'SOME TARGETS MISSED'}")
    print(f"{'--'*30}\n")

    if not all_passed:
        raise SystemExit(1)


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Final held-out evaluation for AirSketch.")
    p.add_argument("--model", default="checkpoints/best_model.optimized.onnx")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out-dir", default="report/eval")
    p.add_argument(
        "--skip-custom",
        action="store_true",
        help="Evaluate FreiHAND only (skip custom gesture set).",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Cap on windows per split (0 = all). Use for debugging.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.8,
        help=(
            "EMA smoothing factor for predicted fingertip positions "
            "(0 < alpha <= 1.0). 1.0 = no smoothing. "
            "Default is 0.8 for a lower-lag smoothed output."
        ),
    )
    args = p.parse_args()
    main(
        model_path=args.model,
        config_path=args.config,
        out_dir=args.out_dir,
        skip_custom=args.skip_custom,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
        ema_alpha=args.ema_alpha,
    )
