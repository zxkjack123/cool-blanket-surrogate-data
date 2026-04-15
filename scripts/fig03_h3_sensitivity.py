#!/usr/bin/env python3
"""Generate Figure 3: H3 Peak Sensitivity (C2 main figure).

Left panels: Permutation importance bar charts (cell_2, cell_3).
Right panels: PDP curves for top-2 features per region.
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

FEATURE_LABELS = {
    "FW_THICK_CM": "FW Thickness",
    "LI6_ENRICH_ATOM_FRAC": "Li-6 Enrichment",
    "PBLI_THICK_CM": "PbLi Thickness",
    "SHIELD_THICK_CM": "Shield Thickness",
    "VV_THICK_CM": "VV Thickness",
}

REGION_TITLES = {"cell_2": "Cell 2 (FW-adjacent)", "cell_3": "Cell 3 (PbLi bulk)"}

PHYSICS_NOTES = {
    ("cell_2", "FW_THICK_CM"): "thicker FW\n→ more moderation\n→ H-3↑",
    ("cell_3", "LI6_ENRICH_ATOM_FRAC"): "Li-6↑ → more\n⁶Li(n,α)T → H-3↑",
    ("cell_3", "PBLI_THICK_CM"): "thicker PbLi\n→ more Li target\n→ H-3↑",
}


def main():
    imp = pd.read_csv(ARTIFACTS / "h3_sensitivity.csv")
    pdp = pd.read_csv(ARTIFACTS / "h3_pdp_data.csv")

    regions = ["cell_2", "cell_3"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), dpi=150)

    colors = plt.cm.tab10(np.arange(5))

    for row, region in enumerate(regions):
        # --- Left: bar chart ---
        ax_bar = axes[row, 0]
        sub = imp[imp["region"] == region].sort_values("perm_importance_mean", ascending=True)
        labels = [FEATURE_LABELS.get(f, f) for f in sub["feature"]]
        vals = sub["perm_importance_mean"].values
        stds = sub["perm_importance_std"].values
        bar_colors = ["#2ca02c" if v > 0.01 else "#999999" for v in vals]
        ax_bar.barh(labels, vals, xerr=stds, color=bar_colors, edgecolor="black",
                    linewidth=0.5, capsize=2, height=0.6)
        ax_bar.set_xlabel("Permutation Importance")
        ax_bar.set_title(f"{REGION_TITLES[region]} — Importance", fontsize=9)
        ax_bar.axvline(0, color="black", linewidth=0.5)

        # --- Right: PDP for top-2 features ---
        ax_pdp = axes[row, 1]
        top2 = imp[imp["region"] == region].sort_values("rank").head(2)["feature"].tolist()
        for i, feat in enumerate(top2):
            sub_pdp = pdp[(pdp["region"] == region) & (pdp["feature"] == feat)]
            label = FEATURE_LABELS.get(feat, feat)
            ax_pdp.plot(sub_pdp["grid_value"], sub_pdp["pdp_log10_h3"],
                        label=label, linewidth=1.8, color=colors[i])

            # Physics annotation on the curve
            note = PHYSICS_NOTES.get((region, feat))
            if note:
                mid = len(sub_pdp) * 3 // 4
                ax_pdp.annotate(
                    note,
                    xy=(sub_pdp["grid_value"].iloc[mid], sub_pdp["pdp_log10_h3"].iloc[mid]),
                    xytext=(15, 10 if i == 0 else -25),
                    textcoords="offset points",
                    fontsize=6.5,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.9),
                )

        ax_pdp.set_xlabel(f"Feature Value")
        ax_pdp.set_ylabel("PDP: log₁₀(H-3 peak)")
        ax_pdp.set_title(f"{REGION_TITLES[region]} — Partial Dependence", fontsize=9)
        ax_pdp.legend(fontsize=7, loc="best")

    fig.suptitle("H-3 Peak Inventory Sensitivity Analysis", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig03_h3_sensitivity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig03_h3_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig03_h3_sensitivity.pdf/png")


if __name__ == "__main__":
    main()
