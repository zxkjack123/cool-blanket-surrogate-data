#!/usr/bin/env python3
"""Generate Figure 8: Design What-If Demo (Li-6 Enrichment Sweep).

Dual panel: TBR + UQ (top) and H3 peak + UQ (bottom) vs Li-6 enrichment sweep.
Other parameters fixed at median of training set.
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

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
DATA_ROOT = Path("/home/gw/ComputeData/CFETR/COOL-PbLi-Burnup/datasets")
LABELS_CSV = DATA_ROOT / "wp2_batches/expandC_act9_n144_20260404/labels_long_expandC_act9.csv"
LHS_CSV = DATA_ROOT / "wp2_lhs_cool_csg_expandC_n144_seed20260330.csv"

FEATURES = ["FW_THICK_CM", "LI6_ENRICH_ATOM_FRAC", "PBLI_THICK_CM",
            "SHIELD_THICK_CM", "VV_THICK_CM"]


def extract_a_idx(run_id: str) -> int:
    return int(run_id.rsplit("_a", 1)[1])


def main():
    # Load data
    labels = pd.read_csv(LABELS_CSV)
    labels["a_idx"] = labels["run_id"].apply(extract_a_idx)
    with open(ARTIFACTS / "splits.json") as f:
        splits = json.load(f)
    train_ids = set(splits["train"])
    cal_ids = set(splits["calibration"]) if "calibration" in splits else set(splits.get("cal", []))

    lhs = pd.read_csv(LHS_CSV)
    lhs["a_idx"] = lhs["sample_id"].str.replace("s", "").astype(int) - 200

    # ---- TBR target ----
    c3 = labels[labels["region"] == "cell_3"].copy()
    # TBR is time-constant; take first timepoint per sample
    tbr_df = c3.groupby("a_idx").first().reset_index()[["a_idx", "tbr"]]
    tbr_df = tbr_df.merge(lhs[["a_idx"] + FEATURES], on="a_idx")

    train_tbr = tbr_df[tbr_df["a_idx"].isin(train_ids)]
    cal_tbr = tbr_df[tbr_df["a_idx"].isin(cal_ids)]

    X_train_tbr = train_tbr[FEATURES].values
    y_train_tbr = train_tbr["tbr"].values
    X_cal_tbr = cal_tbr[FEATURES].values
    y_cal_tbr = cal_tbr["tbr"].values

    model_tbr = HistGradientBoostingRegressor(random_state=42, max_iter=200)
    model_tbr.fit(X_train_tbr, y_train_tbr)

    # Conformal qhat for TBR
    cal_pred_tbr = model_tbr.predict(X_cal_tbr)
    residuals_tbr = np.abs(y_cal_tbr - cal_pred_tbr)
    n_cal = len(residuals_tbr)
    q_level = min(np.ceil((n_cal + 1) * 0.9) / n_cal, 1.0)
    qhat_tbr = np.quantile(residuals_tbr, q_level, method="higher")

    # ---- H3 peak target ----
    h3_peak = c3.groupby("a_idx")["atoms_H3"].max().reset_index()
    h3_peak.columns = ["a_idx", "h3_peak"]
    h3_peak = h3_peak.merge(lhs[["a_idx"] + FEATURES], on="a_idx")

    train_h3 = h3_peak[h3_peak["a_idx"].isin(train_ids)]
    cal_h3 = h3_peak[h3_peak["a_idx"].isin(cal_ids)]

    X_train_h3 = train_h3[FEATURES].values
    y_train_h3_log = np.log10(train_h3["h3_peak"].values)
    X_cal_h3 = cal_h3[FEATURES].values
    y_cal_h3_log = np.log10(cal_h3["h3_peak"].values)

    model_h3 = HistGradientBoostingRegressor(random_state=42, max_iter=200)
    model_h3.fit(X_train_h3, y_train_h3_log)

    cal_pred_h3_log = model_h3.predict(X_cal_h3)
    residuals_h3 = np.abs(y_cal_h3_log - cal_pred_h3_log)
    qhat_h3_log = np.quantile(residuals_h3, q_level, method="higher")

    # ---- Sweep Li6 enrichment ----
    li6_range = np.linspace(0.30, 0.90, 50)
    # Fix other params at median of training set
    medians = train_tbr[FEATURES].median()
    X_sweep = np.tile(medians.values, (len(li6_range), 1))
    li6_idx = FEATURES.index("LI6_ENRICH_ATOM_FRAC")
    X_sweep[:, li6_idx] = li6_range

    tbr_pred = model_tbr.predict(X_sweep)
    h3_pred_log = model_h3.predict(X_sweep)
    h3_pred = 10 ** h3_pred_log

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.0), dpi=150,
                                    sharex=True, constrained_layout=True)

    # TBR panel
    ax1.plot(li6_range, tbr_pred, "C0-", linewidth=1.5, label="TBR prediction")
    ax1.fill_between(li6_range, tbr_pred - qhat_tbr, tbr_pred + qhat_tbr,
                     alpha=0.25, color="C0", label="90% CI")
    ax1.axhline(1.05, color="red", linestyle="--", linewidth=1.0, label="TBR = 1.05")
    ax1.set_ylabel("TBR")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_title("(a) TBR vs Li-6 Enrichment", fontsize=9)

    # H3 peak panel
    ax2.plot(li6_range, h3_pred, "C1-", linewidth=1.5, label="H-3 peak prediction")
    h3_ci_lo = 10 ** (h3_pred_log - qhat_h3_log)
    h3_ci_hi = 10 ** (h3_pred_log + qhat_h3_log)
    ax2.fill_between(li6_range, h3_ci_lo, h3_ci_hi,
                     alpha=0.25, color="C1", label="90% CI")
    ax2.set_yscale("log")
    ax2.set_ylabel("H-3 Peak Inventory (atoms)")
    ax2.set_xlabel("Li-6 Enrichment (atom fraction)")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.set_title("(b) H-3 Peak vs Li-6 Enrichment", fontsize=9)

    # Annotation: other params fixed
    note = (f"Other params at median:\n"
            f"FW={medians['FW_THICK_CM']:.1f} cm, "
            f"PbLi={medians['PBLI_THICK_CM']:.1f} cm\n"
            f"Shield={medians['SHIELD_THICK_CM']:.1f} cm, "
            f"VV={medians['VV_THICK_CM']:.1f} cm")
    ax1.text(0.02, 0.02, note, transform=ax1.transAxes, fontsize=6, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    fig.suptitle("Design What-If: Li-6 Enrichment Sweep", fontsize=11, fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig08_what_if.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig08_what_if.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Saved fig08_what_if.pdf/png")


if __name__ == "__main__":
    main()
