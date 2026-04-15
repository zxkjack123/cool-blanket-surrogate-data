#!/usr/bin/env python3
"""Generate Figure 5: Parity Plots (TBR + H3 peak + Li6 at EOC).

3 panels: (a) TBR scalar, (b) H3 peak inventory, (c) Li6 at end of irradiation.
Test set only with ±10% lines and R²/MAE annotations.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
DATA_ROOT = Path("/home/gw/ComputeData/CFETR/COOL-PbLi-Burnup/datasets")
LABELS_CSV = DATA_ROOT / "wp2_batches/expandC_act9_n144_20260404/labels_long_expandC_act9.csv"
LHS_CSV = DATA_ROOT / "wp2_lhs_cool_csg_expandC_n144_seed20260330.csv"

FEATURES = ["FW_THICK_CM", "LI6_ENRICH_ATOM_FRAC", "PBLI_THICK_CM",
            "SHIELD_THICK_CM", "VV_THICK_CM"]


def extract_a_idx(run_id: str) -> int:
    """Extract a_idx from run_id like '..._a143'."""
    return int(run_id.rsplit("_a", 1)[1])


def load_targets():
    """Load per-sample TBR, H3 peak, Li6 at EOC."""
    labels = pd.read_csv(LABELS_CSV)
    labels["a_idx"] = labels["run_id"].apply(extract_a_idx)

    records = []
    for a_idx, grp in labels.groupby("a_idx"):
        c3 = grp[grp["region"] == "cell_3"]
        # TBR (time-constant, take first)
        tbr = c3["tbr"].iloc[0]
        # H3 peak (max over time)
        h3_peak = c3["atoms_H3"].max()
        # Li6 at end of irradiation (last non-cooling step)
        irrad = c3[~c3["is_cooling"]].sort_values("time_s")
        li6_eoc = irrad["atoms_Li6"].iloc[-1] if len(irrad) > 0 else np.nan
        records.append({"a_idx": a_idx, "tbr": tbr, "h3_peak": h3_peak, "li6_eoc": li6_eoc})

    return pd.DataFrame(records)


def main():
    # Load splits
    with open(ARTIFACTS / "splits.json") as f:
        splits = json.load(f)
    train_ids = set(splits["train"])
    test_ids = set(splits["test"])

    # Load LHS design matrix
    lhs = pd.read_csv(LHS_CSV)
    lhs["a_idx"] = lhs["sample_id"].str.replace("s", "").astype(int) - 200

    # Load targets
    targets = load_targets()

    # Merge
    df = targets.merge(lhs[["a_idx"] + FEATURES], on="a_idx")
    df["split"] = df["a_idx"].apply(
        lambda x: "train" if x in train_ids else ("test" if x in test_ids else "cal")
    )

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    X_train = train[FEATURES].values
    X_test = test[FEATURES].values

    # --- (a) TBR from existing results ---
    tbr_res = pd.read_csv(ARTIFACTS / "tbr_conformal_results.csv")
    tbr_test = tbr_res[tbr_res["split"] == "test"]
    tbr_true = tbr_test["tbr"].values
    tbr_pred = tbr_test["tbr_pred"].values

    # --- (b) H3 peak: train HGBR on log10 ---
    y_h3_train = np.log10(train["h3_peak"].values)
    y_h3_test_true = test["h3_peak"].values
    model_h3 = HistGradientBoostingRegressor(random_state=42, max_iter=200)
    model_h3.fit(X_train, y_h3_train)
    h3_pred_log = model_h3.predict(X_test)
    h3_pred = 10 ** h3_pred_log
    h3_r2 = r2_score(np.log10(y_h3_test_true), h3_pred_log)

    # --- (c) Li6 at EOC ---
    y_li6_train = train["li6_eoc"].values
    y_li6_test_true = test["li6_eoc"].values
    model_li6 = HistGradientBoostingRegressor(random_state=42, max_iter=200)
    model_li6.fit(X_train, y_li6_train)
    li6_pred = model_li6.predict(X_test)
    li6_r2 = r2_score(y_li6_test_true, li6_pred)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), dpi=150, constrained_layout=True)

    panels = [
        ("(a) TBR", tbr_true, tbr_pred, False),
        ("(b) H-3 Peak (atoms)", y_h3_test_true, h3_pred, True),
        ("(c) Li-6 at EOI (atoms)", y_li6_test_true, li6_pred, False),
    ]

    for ax, (title, y_true, y_pred, use_log) in zip(axes, panels):
        ax.scatter(y_true, y_pred, s=30, edgecolors="black", linewidths=0.5,
                   alpha=0.8, zorder=3)

        # Diagonal
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "k-", linewidth=0.8, label="y=x")

        # ±10% bands
        x_range = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_range, x_range * 1.10, "r--", linewidth=0.6, alpha=0.5, label="±10%")
        ax.plot(x_range, x_range * 0.90, "r--", linewidth=0.6, alpha=0.5)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.text(0.05, 0.92, f"R²={r2:.4f}\nMAE={mae:.2e}", transform=ax.transAxes,
                fontsize=7, va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))

        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")

    axes[2].legend(fontsize=6, loc="lower right")

    fig.suptitle("Parity Plots — Test Set", fontsize=11, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig05_parity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig05_parity.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved fig05_parity.pdf/png")


if __name__ == "__main__":
    main()
