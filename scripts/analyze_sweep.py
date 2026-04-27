"""
scripts/analyze_sweep.py

Fetches all completed sweep runs from W&B, produces a ranked results table, and generates the figures required for the final report.

Outputs:
    report/figures/sweep_results_table.csv
    report/figures/sweep_heatmap_mpjpe.png
    report/figures/sweep_heatmap_gesture.png
    report/figures/sweep_parallel_coords.png
    report/figures/sweep_best_curves.png

Usage:
    python scripts/analyze_sweep.py --sweep-id <sweep-id>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


ENTITY = "airsketch"
PROJECT = "AirSketch"

# Proposal targets -- drawn as reference lines on all metric plots
TARGETS = {
    "val/mpjpe": 8.0,  # px
    "val/jitter_index": 2.0,  # px2
    "val/gesture_acc": 0.92,
}


def fetch_runs(sweep_id: str) -> pd.DataFrame:
    """
    Fetch all finished runs from the given sweep and return a DataFrame.
    Includes hyperparameters and final-epoch summary metrics.
    """
    api = wandb.Api()
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
    runs = [r for r in sweep.runs if r.state == "finished"]

    print(f"Fetched {len(runs)} finished runs from sweep {sweep_id}.")
    if len(runs) < 27:
        print(
            f"  WARNING: Only {len(runs)}/27 runs finished. "
            f"Results table may be incomplete."
        )

    records = []
    for run in runs:
        records.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "hidden_dim": run.config.get("model.hidden_dim", "?"),
                "num_layers": run.config.get("model.num_layers", "?"),
                "lr": run.config.get("training.learning_rate", "?"),
                "val_mpjpe": run.summary.get("val/mpjpe", float("nan")),
                "val_jitter": run.summary.get("val/jitter_index", float("nan")),
                "val_gesture": run.summary.get("val/gesture_acc", float("nan")),
                "train_loss": run.summary.get("train/loss", float("nan")),
                "best_epoch": run.summary.get("epoch", float("nan")),
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("val_mpjpe").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def print_results_table(df: pd.DataFrame) -> None:
    """Print a formatted top-10 results table to stdout."""
    print("\n── Top 10 configs by val MPJPE ─────────────────────────────────────")
    print(
        f"  {'Rank':>4}  {'hidden':>6}  {'layers':>6}  {'lr':>8}  "
        f"{'MPJPE':>8}  {'jitter':>8}  {'gesture':>8}  {'epoch':>5}"
    )
    print(
        f"  {'':->4}  {'':->6}  {'':->6}  {'':->8}  "
        f"{'':->8}  {'':->8}  {'':->8}  {'':->5}"
    )

    for _, row in df.head(10).iterrows():
        mpjpe_flag = " *" if row["val_mpjpe"] < TARGETS["val/mpjpe"] else "  "
        gesture_flag = " *" if row["val_gesture"] > TARGETS["val/gesture_acc"] else "  "
        jitter_flag = " *" if row["val_jitter"] < TARGETS["val/jitter_index"] else "  "
        print(
            f"  {int(row['rank']):>4}  "
            f"{int(row['hidden_dim']):>6}  "
            f"{int(row['num_layers']):>6}  "
            f"{row['lr']:>8.0e}  "
            f"{row['val_mpjpe']:>7.2f}{mpjpe_flag}  "
            f"{row['val_jitter']:>7.3f}{jitter_flag}  "
            f"{row['val_gesture']:>7.3f}{gesture_flag}  "
            f"{int(row['best_epoch']):>5}"
        )

    print("\n  * = target met")
    print(
        f"  Targets: MPJPE < {TARGETS['val/mpjpe']} px  |  "
        f"jitter < {TARGETS['val/jitter_index']} px2  |  "
        f"gesture > {TARGETS['val/gesture_acc']}"
    )


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    target: float,
    higher_better: bool,
    out_path: Path,
) -> None:
    """
    Plot a 3x3 heatmap of a metric over (hidden_dim x num_layers),
    one subplot per learning rate.
    """
    lrs = sorted(df["lr"].unique())
    fig, axes = plt.subplots(1, len(lrs), figsize=(5 * len(lrs), 4.5), sharey=True)

    if len(lrs) == 1:
        axes = [axes]

    cmap = "RdYlGn" if higher_better else "RdYlGn_r"

    all_vals = df[metric].dropna()
    vmin = all_vals.min()
    vmax = all_vals.max()

    for ax, lr in zip(axes, lrs):
        subset = df[df["lr"] == lr]
        pivot = subset.pivot_table(
            index="num_layers",
            columns="hidden_dim",
            values=metric,
            aggfunc="min" if not higher_better else "max",
        )

        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_xlabel("hidden_dim", fontsize=10)
        if lr == lrs[0]:
            ax.set_ylabel("num_layers", fontsize=10)
        ax.set_title(f"lr = {lr:.0e}", fontsize=11)

        # Annotate each cell with the metric value
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    meets = (val < target) if not higher_better else (val > target)
                    color = "white" if meets else "black"
                    ax.text(
                        c,
                        r,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                        fontweight="bold" if meets else "normal",
                    )

        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        f"{title}\n(* = target met, best cell per subplot highlighted)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_parallel_coords(df: pd.DataFrame, out_path: Path) -> None:
    """
    Parallel coordinates plot: one line per run, colored by val MPJPE.
    Helps spot interactions between hyperparameters.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Normalize each axis to [0, 1] for plotting
    cols = ["hidden_dim", "num_layers", "lr", "val_mpjpe"]
    norms = {}
    for col in cols:
        lo, hi = df[col].min(), df[col].max()
        norms[col] = (df[col] - lo) / (hi - lo + 1e-9)

    cmap = plt.cm.RdYlGn_r
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(df["val_mpjpe"].min(), df["val_mpjpe"].max()),
    )
    sm.set_array([])

    for _, row in df.iterrows():
        ys = [norms[c][row.name] for c in cols]
        color = cmap(norms["val_mpjpe"][row.name])
        ax.plot(range(len(cols)), ys, color=color, alpha=0.5, linewidth=1.0)

    # Axis labels with tick marks
    for i, col in enumerate(cols):
        vals = sorted(df[col].unique())
        lo, hi = df[col].min(), df[col].max()
        ax.axvline(i, color="gray", linewidth=0.5, linestyle="--")
        for v in vals:
            yn = (v - lo) / (hi - lo + 1e-9)
            ax.text(
                i,
                yn,
                f"{v:.1e}" if col == "lr" else str(int(v)),
                ha="center",
                va="center",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
            )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        ["hidden_dim", "num_layers", "lr", "val MPJPE (px)"], fontsize=10
    )
    ax.set_yticks([])
    ax.set_title(
        "Parallel coordinates — all 27 sweep configs\n"
        "(green = low MPJPE, red = high)",
        fontsize=11,
    )
    plt.colorbar(sm, ax=ax, label="val MPJPE (px)", shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_best_curves(sweep_id: str, df: pd.DataFrame, out_path: Path) -> None:
    """
    Fetch epoch-by-epoch history for the top 5 runs and plot training curves
    for val MPJPE, val gesture accuracy, and val jitter index.
    """
    api = wandb.Api()
    top5_ids = df.head(5)["run_id"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["val/mpjpe", "val/gesture_acc", "val/jitter_index"]
    titles = [
        "Val MPJPE (px) -- lower is better",
        "Val Gesture Acc -- higher is better",
        "Val Jitter Index (px2) -- lower is better",
    ]

    for ax, metric, title, target in zip(
        axes, metrics, titles, [TARGETS[m] for m in metrics]
    ):
        for run_id in top5_ids:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            hist = run.history(keys=[metric, "epoch"])
            if metric not in hist.columns:
                continue
            label = (
                f"h={run.config.get('model.hidden_dim')} "
                f"l={run.config.get('model.num_layers')} "
                f"lr={run.config.get('training.learning_rate'):.0e}"
            )
            ax.plot(hist["epoch"], hist[metric], linewidth=1.5, alpha=0.85, label=label)

        ax.axhline(
            y=target,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Target: {target}",
        )
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Top 5 configs — training curves", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main(sweep_id: str) -> None:
    figures_dir = Path("report/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching results for sweep: {sweep_id}")
    df = fetch_runs(sweep_id)

    # -- Save CSV results table ------------------------------------------------
    csv_path = figures_dir / "sweep_results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # -- Print top-10 table to console ----------------------------------------
    print_results_table(df)

    # -- MPJPE heatmap --------------------------------------------------------
    plot_heatmap(
        df,
        metric="val_mpjpe",
        title="Val MPJPE (px) by config -- lower is better",
        target=TARGETS["val/mpjpe"],
        higher_better=False,
        out_path=figures_dir / "sweep_heatmap_mpjpe.png",
    )

    # -- Gesture accuracy heatmap ---------------------------------------------
    plot_heatmap(
        df,
        metric="val_gesture",
        title="Val Gesture Accuracy by config -- higher is better",
        target=TARGETS["val/gesture_acc"],
        higher_better=True,
        out_path=figures_dir / "sweep_heatmap_gesture.png",
    )

    # -- Parallel coordinates -------------------------------------------------
    plot_parallel_coords(df, out_path=figures_dir / "sweep_parallel_coords.png")

    # -- Top-5 training curves ------------------------------------------------
    plot_best_curves(
        sweep_id,
        df,
        out_path=figures_dir / "sweep_best_curves.png",
    )

    # -- Print winning config --------------------------------------------------
    best = df.iloc[0]
    print("\n── Best config ─────────────────────────────────────────────────────")
    print(f"  hidden_dim:  {int(best['hidden_dim'])}")
    print(f"  num_layers:  {int(best['num_layers'])}")
    print(f"  lr:          {best['lr']:.0e}")
    print(
        f"  val MPJPE:   {best['val_mpjpe']:.2f} px  "
        f"({'PASS' if best['val_mpjpe'] < TARGETS['val/mpjpe'] else 'FAIL'} -- target < 8 px)"
    )
    print(
        f"  val jitter:  {best['val_jitter']:.3f} px2  "
        f"({'PASS' if best['val_jitter'] < TARGETS['val/jitter_index'] else 'FAIL'} -- target < 2 px2)"
    )
    print(
        f"  val gesture: {best['val_gesture']:.3f}  "
        f"({'PASS' if best['val_gesture'] > TARGETS['val/gesture_acc'] else 'FAIL'} -- target > 0.92)"
    )
    print("\n  Update configs/default.yaml with these values before issue #12.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-id",
        required=True,
        help="W&B sweep ID (the short alphanumeric ID, not the full URL).",
    )
    main(p.parse_args().sweep_id)
