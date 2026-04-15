#!/usr/bin/env python3
"""Generate Figure 7: Speedup Comparison (MC+burnup vs Surrogate).

Log-scale bar chart comparing wall-clock times.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# Data from runtime_benchmark.md (Task 2.6)
# Physics: ~25-45 min per sample on 8 HPC cores (SugonHB AMD EPYC 7H12, 2.6GHz)
# Surrogate: 10.7ms single, 28.6ms for 8640 batch
PHYSICS_PER_SAMPLE_S = 35 * 60  # ~35 min median → 2100s
SURROGATE_SINGLE_S = 0.0107  # 10.7 ms
SURROGATE_BATCH_S = 28.6e-3 / 8640  # per-prediction in batch mode
BATCH_SIZE = 144  # full dataset

# For parameter sweep of N=144 designs:
PHYSICS_TOTAL_S = PHYSICS_PER_SAMPLE_S * BATCH_SIZE  # sequential
SURROGATE_TOTAL_S = 28.6e-3  # batch of 8640, N=144 needs << 8640


def main():
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=150, constrained_layout=True)

    categories = ["Per sample\n(single query)", f"Full sweep\n(N={BATCH_SIZE})"]
    physics_vals = [PHYSICS_PER_SAMPLE_S, PHYSICS_TOTAL_S]
    surrogate_vals = [SURROGATE_SINGLE_S, SURROGATE_TOTAL_S]

    x = np.arange(len(categories))
    w = 0.3

    bars_phys = ax.bar(x - w / 2, physics_vals, w, label="OpenMC + FISPACT",
                       color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_surr = ax.bar(x + w / 2, surrogate_vals, w, label="HGBR Surrogate",
                       color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Wall-Clock Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)

    # Annotate speedup
    for i in range(len(categories)):
        speedup = physics_vals[i] / surrogate_vals[i]
        mid_x = x[i]
        top_y = max(physics_vals[i], surrogate_vals[i])
        ax.annotate(f"{speedup:.0e}×",
                    xy=(mid_x, top_y * 2),
                    fontsize=9, fontweight="bold", ha="center", color="#1f77b4")

    # Annotate absolute values on bars
    for bar, val in zip(bars_phys, physics_vals):
        if val >= 3600:
            label = f"{val/3600:.1f} h"
        elif val >= 60:
            label = f"{val/60:.0f} min"
        else:
            label = f"{val:.1f} s"
        ax.text(bar.get_x() + bar.get_width() / 2, val * 0.3, label,
                ha="center", va="top", fontsize=7, color="white", fontweight="bold")

    for bar, val in zip(bars_surr, surrogate_vals):
        if val >= 1:
            label = f"{val:.1f} s"
        else:
            label = f"{val*1000:.1f} ms"
        ax.text(bar.get_x() + bar.get_width() / 2, val * 3, label,
                ha="center", va="bottom", fontsize=7, color="black")

    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Computational Cost: Physics vs Surrogate", fontsize=10, fontweight="bold")

    # Hardware note
    ax.text(0.02, 0.02,
            "Physics: 8 cores, AMD EPYC 7H12 @ 2.6 GHz\n"
            "Surrogate: 1 core, Intel i9-10900X @ 3.7 GHz",
            transform=ax.transAxes, fontsize=6, va="bottom", color="gray")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig07_speedup.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig07_speedup.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved fig07_speedup.pdf/png")


if __name__ == "__main__":
    main()
