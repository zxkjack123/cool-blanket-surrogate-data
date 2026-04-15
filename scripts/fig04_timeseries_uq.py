#!/usr/bin/env python3
"""Generate Figure 4: Representative Time-Series Curves with UQ Bands.

3×2 panels: 3 targets (H3, Li6, Li7) × 2 samples (safe vs edge).
Each panel: truth, prediction, 90% CI band.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

TARGETS = ["atoms_H3", "atoms_Li6", "atoms_Li7"]
TARGET_LABELS = {"atoms_H3": "H-3 (atoms)", "atoms_Li6": "Li-6 (atoms)", "atoms_Li7": "Li-7 (atoms)"}

# Samples: 14 = safe (TBR=1.54), 20 = edge (TBR=1.41)
SAMPLES = [14, 20]
SAMPLE_LABELS = {14: "Safe design (TBR=1.54)", 20: "Edge design (TBR=1.41)"}

COOLING_ONSET_YR = 10.0  # Irradiation ends at ~10 FPY


def main():
    curves = pd.read_csv(ARTIFACTS / "timeseries_curves.csv")

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.0), dpi=150, sharex=True,
                             constrained_layout=True)

    for col, sid in enumerate(SAMPLES):
        for row, target in enumerate(TARGETS):
            ax = axes[row, col]
            sub = curves[
                (curves["region"] == "cell_3")
                & (curves["target"] == target)
                & (curves["sample_id"] == sid)
            ].sort_values("time_yr")

            if sub.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            t = sub["time_yr"].values
            y_true = sub["true"].values
            y_pred = sub["pred"].values
            ci_lo = sub["ci_lower"].values
            ci_hi = sub["ci_upper"].values

            ax.fill_between(t, ci_lo, ci_hi, alpha=0.25, color="C0", label="90% CI")
            ax.plot(t, y_true, "ko-", markersize=3, linewidth=1.0, label="FISPACT (truth)")
            ax.plot(t, y_pred, "C0s--", markersize=3, linewidth=1.0, label="Surrogate")

            ax.set_yscale("log")
            ax.set_xscale("log")

            # Cooling onset vertical line
            ax.axvline(COOLING_ONSET_YR, color="gray", linestyle=":", linewidth=0.8)
            # Use axes-relative coordinate for text to avoid layout explosion
            ax.text(0.73, 0.92, "cooling", transform=ax.transAxes,
                    fontsize=6, color="gray", ha="left", va="top")

            # Labels
            if col == 0:
                ax.set_ylabel(TARGET_LABELS[target], fontsize=9)
            if row == 0:
                ax.set_title(SAMPLE_LABELS[sid], fontsize=9, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Time (yr)")
            if row == 0 and col == 1:
                ax.legend(fontsize=6.5, loc="lower right")

    fig.suptitle("Time-Series Predictions with Conformal UQ (cell_3)",
                 fontsize=11, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig04_timeseries_uq.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig04_timeseries_uq.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved fig04_timeseries_uq.pdf/png")


if __name__ == "__main__":
    main()
