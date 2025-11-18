#!/usr/bin/env python3
"""
Plot h_max_eig, h_stable_rank, and h_trace from the CSV produced by your scan.

Usage:
  python plot_hidden_metrics.py \
    --csv pythia_hidden_metrics.csv \
    --out metrics.png \
    --logy \
    --ma 5

Arguments:
  --csv   Path to CSV (default: pythia_hidden_metrics.csv)
  --out   Output image path (default: hidden_metrics.png)
  --logy  Use log scale on y-axis for eig/trace (stable rank stays linear)
  --ma    Moving average window (int >=1). 1 = no smoothing. Default: 1
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, k: int):
    if k <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    # centered moving average; reflect pad to keep length
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(k) / k
    y = np.convolve(xpad, kernel, mode="valid")
    # if k even, trim one extra at end to keep exact length
    if len(y) > len(x):
        y = y[: len(x)]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="pythia_hidden_metrics.csv")
    ap.add_argument("--out", type=str, default="hidden_metrics_logscale.png")
    ap.add_argument("--logy", action="store_true", help="log scale for eig/trace")
    ap.add_argument("--ma", type=int, default=1, help="moving average window")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    steps, h_max_eig, h_trace, h_stable_rank = [], [], [], []

    with open("pythia_hidden_metrics.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            s = int(row["step"])
            if s == 0:   # avoid log(0)
                continue
            steps.append(s)
            h_max_eig.append(float(row["h_max_eig"]))
            h_trace.append(float(row["h_trace"]))
            h_stable_rank.append(float(row["h_stable_rank"]))

    # Sort by step (just in case)
    order = np.argsort(steps)
    steps = np.array(steps)[order]
    h_max_eig = np.array(h_max_eig)[order]
    h_trace = np.array(h_trace)[order]
    h_stable_rank = np.array(h_stable_rank)[order]

    # Smoothing (optional)
    k = max(1, int(args.ma))
    h_max_eig_s = moving_average(h_max_eig, k)
    h_trace_s = moving_average(h_trace, k)
    h_sr_s = moving_average(h_stable_rank, k)

    # Plot
    plt.figure(figsize=(11, 6))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Distinct colors for clarity
    color_eig = "#1f77b4"     # blue
    color_trace = "#ff7f0e"   # orange
    color_sr = "#2ca02c"      # green

    l1, = ax1.plot(steps, h_max_eig_s, label="h_max_eig", color=color_eig, linewidth=1.8)
    l2, = ax1.plot(steps, h_trace_s, label="h_trace", color=color_trace, linestyle="--", linewidth=1.8)
    l3, = ax2.plot(steps, h_sr_s, label="h_stable_rank", color=color_sr, linewidth=1.8)

    ax1.set_xscale("log") 
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Eigenvalue / Trace" + (" (log)" if args.logy else ""))
    ax2.set_ylabel("Stable Rank")

    if args.logy:
        ax1.set_yscale("log")

    # Build combined legend
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    ax1.grid(True, linestyle=":", alpha=0.5)
    plt.title("Hidden-State Hessian Metrics over Checkpoints")
    if k > 1:
        plt.suptitle(f"Centered moving average window = {k}", y=0.97, fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()