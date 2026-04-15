"""Graphical abstract for FED submission.

Two-panel layout (left-right):
  Left:  Simplified TBR constraint landscape scatter (θ=1.40)
  Right: Workflow arrow diagram (5 params → ML → Safe/Risk zones)

Output: ≥531×1328 px (FED requirement).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_DIR = ROOT / "figures"
ART_DIR = ROOT / "artifacts"

LANDSCAPE_CSV = ART_DIR / "constraint_landscape.csv"


def main():
    df = pd.read_csv(LANDSCAPE_CSV)
    tbr_pred = df["tbr_pred"].values
    qhat = (df["ci_upper"].values - df["ci_lower"].values) / 2.0
    theta = 1.40
    lower = tbr_pred - qhat
    upper = tbr_pred + qhat
    pr = np.clip((upper - theta) / (upper - lower + 1e-12), 0.0, 1.0)

    # Classify
    safe = pr > 0.95
    risk = pr < 0.50
    uncertain = ~safe & ~risk

    # ── Create figure: 1328 wide × 531 tall at 150 dpi ──
    # Use 300 dpi for delivery quality; figsize in inches
    dpi = 300
    fig_w_in = 1328 / dpi * 1.5  # ~6.64 in → 1992 px at 300 dpi
    fig_h_in = 531 / dpi * 1.5   # ~2.66 in → 797 px at 300 dpi

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(fig_w_in, fig_h_in), dpi=dpi,
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35},
    )

    # ── Left panel: TBR landscape scatter ──
    f_li6 = df["LI6_ENRICH_ATOM_FRAC"].values
    d_pbli = df["PBLI_THICK_CM"].values

    ax_left.scatter(f_li6[risk], d_pbli[risk], c="#F44336", s=30, alpha=0.8,
                    edgecolors="white", linewidths=0.3, label="Risk", zorder=2)
    ax_left.scatter(f_li6[uncertain], d_pbli[uncertain], c="#FFC107", s=30,
                    alpha=0.8, edgecolors="white", linewidths=0.3,
                    label="Uncertain", zorder=2)
    ax_left.scatter(f_li6[safe], d_pbli[safe], c="#4CAF50", s=30, alpha=0.8,
                    edgecolors="white", linewidths=0.3, label="Safe", zorder=2)

    ax_left.set_xlabel("Li-6 enrichment", fontsize=8)
    ax_left.set_ylabel("PbLi thickness (cm)", fontsize=8)
    ax_left.set_title(r"TBR constraint map ($\theta$ = 1.40)", fontsize=9,
                      fontweight="bold")
    ax_left.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax_left.tick_params(labelsize=7)

    # ── Right panel: conceptual workflow ──
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 6)
    ax_right.axis("off")
    ax_right.set_title("Surrogate design analysis", fontsize=9,
                       fontweight="bold")

    # Boxes
    box_style = dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD",
                     edgecolor="#1565C0", linewidth=1)
    result_safe = dict(boxstyle="round,pad=0.3", facecolor="#C8E6C9",
                       edgecolor="#2E7D32", linewidth=1)
    result_risk = dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2",
                       edgecolor="#C62828", linewidth=1)

    # Input parameters
    ax_right.text(1.5, 5.0, "5 Design\nParameters", fontsize=7, ha="center",
                  va="center", bbox=box_style)

    # ML surrogate
    ax_right.text(5.0, 5.0, "ML Surrogate\n+ Conformal UQ", fontsize=7,
                  ha="center", va="center", bbox=box_style)

    # Output branches
    ax_right.text(8.5, 5.0, "TBR +\nUncertainty", fontsize=6.5,
                  ha="center", va="center", bbox=box_style)

    ax_right.text(5.0, 2.0, "Constraint\nSatisfaction\nMap", fontsize=7,
                  ha="center", va="center", bbox=box_style)

    ax_right.text(2.0, 0.8, "Safe\nZone", fontsize=7, ha="center",
                  va="center", bbox=result_safe, fontweight="bold")
    ax_right.text(5.0, 0.8, "Uncertain", fontsize=7, ha="center",
                  va="center",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4",
                            edgecolor="#F9A825", linewidth=1))
    ax_right.text(8.0, 0.8, "Risk\nZone", fontsize=7, ha="center",
                  va="center", bbox=result_risk, fontweight="bold")

    # Arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                    color="#333", lw=1.2)
    ax_right.annotate("", xy=(3.5, 5.0), xytext=(2.8, 5.0),
                      arrowprops=arrow_kw)
    ax_right.annotate("", xy=(6.8, 5.0), xytext=(6.2, 5.0),
                      arrowprops=arrow_kw)
    ax_right.annotate("", xy=(5.0, 3.0), xytext=(5.0, 4.2),
                      arrowprops=arrow_kw)
    # Down from constraint map to zones
    ax_right.annotate("", xy=(2.0, 1.5), xytext=(4.0, 2.0),
                      arrowprops=arrow_kw)
    ax_right.annotate("", xy=(5.0, 1.5), xytext=(5.0, 1.5),
                      arrowprops=arrow_kw)
    ax_right.annotate("", xy=(8.0, 1.5), xytext=(6.0, 2.0),
                      arrowprops=arrow_kw)

    fig.tight_layout()

    out_pdf = FIG_DIR / "graphical_abstract.pdf"
    out_png = FIG_DIR / "graphical_abstract.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300, facecolor="white")

    # Verify size
    from PIL import Image
    img = Image.open(out_png)
    w, h = img.size
    print(f"Graphical abstract: {w}×{h} px")
    assert w >= 531 and h >= 531, f"Too small: {w}×{h}"
    print(f"Saved to {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
