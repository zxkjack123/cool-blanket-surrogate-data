#!/usr/bin/env python3
"""Generate Figure 1: End-to-end workflow overview diagram.

Creates a workflow figure showing:
OpenMC -> 709g flux -> FISPACT burnup -> Labels -> ML Surrogate + Conformal UQ
-> Q1 (TBR constraint landscape) + Q2 (H3 sensitivity analysis)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"


def draw_workflow():
    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Color scheme
    sim_color = "#4BACC6"    # simulation stages
    data_color = "#F79646"   # data stage
    ml_color = "#9BBB59"     # ML stage
    app_color = "#8064A2"    # application stages (C1, C2)

    def add_box(x, y, w, h, text, color, fontsize=7, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, wrap=True)

    def add_arrow(x1, y1, x2, y2, style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color="black", lw=1.2))

    # Row 1: Physics pipeline (top)
    y1 = 5.0
    bh = 1.2
    add_box(0.3, y1, 2.0, bh, "OpenMC\nMonte Carlo\n(CSG, 150M)", sim_color, 6.5)
    add_box(3.0, y1, 2.0, bh, "709-group\nFlux\nExtraction", sim_color, 6.5)
    add_box(5.7, y1, 2.0, bh, "FISPACT-II\nBurnup\n(4 regions)", sim_color, 6.5)
    add_box(8.4, y1, 2.3, bh, "Labels\n(TBR, H3, Li6,\nLi7, ...)", data_color, 6.5)

    add_arrow(2.3, y1+bh/2, 3.0, y1+bh/2)
    add_arrow(5.0, y1+bh/2, 5.7, y1+bh/2)
    add_arrow(7.7, y1+bh/2, 8.4, y1+bh/2)

    # Annotation: "144 LHS samples, 5 design params"
    ax.text(5.5, y1+bh+0.3, "144 LHS samples, 5 design parameters",
            ha="center", va="bottom", fontsize=6, style="italic", color="gray")

    # Row 2: ML + UQ (middle)
    y2 = 2.8
    add_box(3.3, y2, 3.2, bh, "HGBR Surrogate\n+ Split Conformal UQ\n(per target, per timepoint)", ml_color, 6.5)

    add_arrow(9.5, y1, 4.9, y2+bh, "-|>")  # data -> ML

    # Row 3: Applications (bottom), fork to Q1 and Q2
    y3 = 0.5
    add_box(0.5, y3, 3.5, bh, "Q1: TBR Constraint\nSatisfaction Landscape\nPr(TBR > \u03b8 | x)", app_color, 6.5, bold=True)
    add_box(6.5, y3, 3.5, bh, "Q2: H3 Peak Inventory\nSensitivity Analysis\n(SHAP + PDP)", app_color, 6.5, bold=True)

    # Fork arrows from ML to Q1 and Q2
    add_arrow(4.0, y2, 2.25, y3+bh, "-|>")
    add_arrow(5.5, y2, 8.25, y3+bh, "-|>")

    # Labels: C1, C2
    ax.text(0.5 + 3.5/2, y3-0.15, "Contribution 1 (C1)", ha="center",
            fontsize=6, fontweight="bold", color="#5B3C8C")
    ax.text(6.5 + 3.5/2, y3-0.15, "Contribution 2 (C2)", ha="center",
            fontsize=6, fontweight="bold", color="#5B3C8C")

    # Code/data provenance annotations
    ax.text(1.3, y1-0.15, "openmc 0.15", ha="center", fontsize=5, color="gray")
    ax.text(6.7, y1-0.15, "FISPACT-II 5.0", ha="center", fontsize=5, color="gray")
    ax.text(9.55, y1-0.15, "expandC n=144", ha="center", fontsize=5, color="gray")
    ax.text(4.9, y2-0.15, "scikit-learn HGBR", ha="center", fontsize=5, color="gray")

    # Title
    ax.text(6.5, 6.8, "Figure 1: End-to-end workflow for ML-accelerated blanket burnup analysis",
            ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = draw_workflow()

    # Save PDF (publication) and PNG (preview)
    pdf_path = OUTPUT_DIR / "fig01_workflow.pdf"
    png_path = OUTPUT_DIR / "fig01_workflow.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
