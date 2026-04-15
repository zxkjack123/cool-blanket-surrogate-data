#!/usr/bin/env python3
"""Generate Figure 2: TBR Constraint Satisfaction Landscape (C1 main figure).

Scatter plot of Li6 enrichment vs PbLi thickness, colored by Pr(TBR > theta).
Multi-panel for different theta values.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def main():
    df = pd.read_csv(ARTIFACTS / "constraint_landscape.csv")

    # Select two discriminative thresholds for multi-panel
    thetas = [1.40, 1.50]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=150, sharey=True,
                             gridspec_kw={"right": 0.88})

    cmap = plt.cm.RdYlGn  # Red (low Pr) to Green (high Pr) — colorblind-aware
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for ax, theta in zip(axes, thetas):
        col = f"pr_tbr_gt_{theta:.2f}"
        sc = ax.scatter(
            df["LI6_ENRICH_ATOM_FRAC"],
            df["PBLI_THICK_CM"],
            c=df[col],
            cmap=cmap,
            norm=norm,
            s=25,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.85,
        )
        ax.set_xlabel("Li-6 Enrichment (atom frac)")
        ax.set_title(f"$\\theta = {theta:.2f}$", fontsize=10)

        # Annotate safe/risk zones
        safe = df[df[col] > 0.95]
        risk = df[df[col] < 0.50]
        if not safe.empty:
            ax.text(0.95, 0.95, f"Safe: {len(safe)}", transform=ax.transAxes,
                    fontsize=7, ha="right", va="top", color="green",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
        if not risk.empty:
            ax.text(0.95, 0.85, f"Risk: {len(risk)}", transform=ax.transAxes,
                    fontsize=7, ha="right", va="top", color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    axes[0].set_ylabel("PbLi Thickness (cm)")

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Pr(TBR > $\\theta$)")

    fig.suptitle("TBR Constraint Satisfaction Probability Landscape", fontsize=10, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.08)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig02_tbr_landscape.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig02_tbr_landscape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig02_tbr_landscape.pdf/png")


if __name__ == "__main__":
    main()
