#!/usr/bin/env python3
"""H3 Peak Inventory Sensitivity Analysis (C2 core).

Analyzes tritium (H3) peak inventory sensitivity to blanket design parameters
using HGBR surrogates + permutation importance + partial dependence.

Usage:
    python scripts/h3_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

DATA_ROOT = Path("/home/gw/ComputeData/CFETR/COOL-PbLi-Burnup")
BATCH_ROOT = DATA_ROOT / "datasets" / "wp2_batches"
LHS_CSV = DATA_ROOT / "datasets" / "wp2_lhs_cool_csg_expandC_n144_seed20260330.csv"

DESIGN_COLS = [
    "FW_THICK_CM",
    "PBLI_THICK_CM",
    "SHIELD_THICK_CM",
    "VV_THICK_CM",
    "LI6_ENRICH_ATOM_FRAC",
]

REGIONS = ["cell_2", "cell_3"]


def load_h3_peaks(batch_name: str) -> pd.DataFrame:
    """Extract H3 peak (max over time) per sample per region, joined with design params."""
    batch_dir = BATCH_ROOT / batch_name
    csv_path = next(batch_dir.glob("labels_long_*.csv"))
    df = pd.read_csv(csv_path)
    df["a_idx"] = df["run_id"].str.extract(r"_a(\d+)$").astype(int)

    # For each region, get peak H3 and peak time
    records = []
    for region in REGIONS:
        sub = df[df["region"] == region].copy()
        if sub.empty or "atoms_H3" not in sub.columns:
            continue
        # Peak per sample
        peak_idx = sub.groupby("a_idx")["atoms_H3"].idxmax()
        peaks = sub.loc[peak_idx, ["a_idx", "atoms_H3", "time_s"]].copy()
        peaks = peaks.rename(columns={"atoms_H3": "h3_peak", "time_s": "h3_peak_time_s"})
        peaks["region"] = region
        records.append(peaks)

    peaks_df = pd.concat(records, ignore_index=True)

    # Join with LHS design params
    lhs = pd.read_csv(LHS_CSV)
    lhs["s_idx"] = lhs["sample_id"].str.extract(r"s(\d+)$").astype(int)
    base_offset = int(lhs["s_idx"].min())
    lhs["a_idx"] = lhs["s_idx"] - base_offset

    peaks_df = peaks_df.merge(lhs[["a_idx"] + DESIGN_COLS], on="a_idx", how="left")
    return peaks_df


def train_and_analyze(
    data: pd.DataFrame,
    region: str,
    feat_cols: list[str],
    splits: dict,
    seed: int = 42,
) -> dict:
    """Train HGBR for H3_peak and compute importance + PDP."""
    reg_data = data[data["region"] == region].copy()
    # Log-transform target for better HGBR performance (H3 spans orders of magnitude)
    reg_data["log_h3_peak"] = np.log10(reg_data["h3_peak"])

    train_mask = reg_data["a_idx"].isin(splits["train"])
    test_mask = reg_data["a_idx"].isin(splits["test"])
    cal_mask = reg_data["a_idx"].isin(splits["calibration"])
    trainval_mask = train_mask | cal_mask  # use train+cal for surrogate (no conformal needed here)

    X_tv = reg_data.loc[trainval_mask, feat_cols].values
    y_tv = reg_data.loc[trainval_mask, "log_h3_peak"].values
    X_test = reg_data.loc[test_mask, feat_cols].values
    y_test = reg_data.loc[test_mask, "log_h3_peak"].values

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=1200,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=60,
        min_samples_leaf=2,
    )
    model.fit(X_tv, y_tv)

    y_tv_pred = model.predict(X_tv)
    y_test_pred = model.predict(X_test)

    r2_tv = r2_score(y_tv, y_tv_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"\n  [{region}] R² (train+cal): {r2_tv:.4f}, R² (test): {r2_test:.4f}, MAE (test, log10): {mae_test:.4f}")

    # Permutation importance on test set
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=seed, scoring="r2"
    )
    importance_mean = perm_result.importances_mean
    importance_std = perm_result.importances_std

    # Ranking
    ranking = np.argsort(-importance_mean)
    print(f"  Permutation importance ranking:")
    for rank, idx in enumerate(ranking):
        print(f"    {rank+1}. {feat_cols[idx]}: {importance_mean[idx]:.4f} +/- {importance_std[idx]:.4f}")

    # Partial dependence
    pdp_data = {}
    X_all = reg_data[feat_cols].values
    for i, col in enumerate(feat_cols):
        pd_result = partial_dependence(model, X_all, features=[i], kind="average", grid_resolution=50)
        pdp_data[col] = {
            "grid": pd_result["grid_values"][0].tolist(),
            "mean": pd_result["average"][0].tolist(),
        }

    return {
        "region": region,
        "r2_trainval": float(r2_tv),
        "r2_test": float(r2_test),
        "mae_test_log10": float(mae_test),
        "n_trainval": int(len(y_tv)),
        "n_test": int(len(y_test)),
        "importance": {
            feat_cols[i]: {"mean": float(importance_mean[i]), "std": float(importance_std[i])}
            for i in range(len(feat_cols))
        },
        "ranking": [feat_cols[i] for i in ranking],
        "pdp": pdp_data,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description="H3 sensitivity analysis")
    parser.add_argument("--batch", default="expandC_act9_n144_20260404")
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "splits.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts",
    )
    args = parser.parse_args()

    print(f"Loading H3 peaks: {args.batch}")
    peaks = load_h3_peaks(args.batch)
    print(f"  Total records: {len(peaks)}, regions: {peaks['region'].unique().tolist()}")

    splits_raw = json.loads(args.splits_json.read_text(encoding="utf-8"))
    splits = {
        "train": [int(x) for x in splits_raw["train"]],
        "calibration": [int(x) for x in splits_raw["calibration"]],
        "test": [int(x) for x in splits_raw["test"]],
    }

    results = {}
    for region in REGIONS:
        if region not in peaks["region"].values:
            print(f"\n  [{region}] Skipped (no data)")
            continue
        results[region] = train_and_analyze(peaks, region, DESIGN_COLS, splits, args.seed)

    # Build sensitivity output CSV
    sens_rows = []
    for region, res in results.items():
        for col in DESIGN_COLS:
            imp = res["importance"][col]
            rank = res["ranking"].index(col) + 1
            sens_rows.append({
                "region": region,
                "feature": col,
                "perm_importance_mean": imp["mean"],
                "perm_importance_std": imp["std"],
                "rank": rank,
            })
    sens_df = pd.DataFrame(sens_rows)
    sens_path = args.output_dir / "h3_sensitivity.csv"
    sens_path.parent.mkdir(parents=True, exist_ok=True)
    sens_df.to_csv(sens_path, index=False)
    print(f"\nSensitivity written to {sens_path}")

    # Build PDP output CSV (long format)
    pdp_rows = []
    for region, res in results.items():
        for col, pdp in res["pdp"].items():
            for g, m in zip(pdp["grid"], pdp["mean"]):
                pdp_rows.append({
                    "region": region,
                    "feature": col,
                    "grid_value": g,
                    "pdp_log10_h3": m,
                })
    pdp_df = pd.DataFrame(pdp_rows)
    pdp_path = args.output_dir / "h3_pdp_data.csv"
    pdp_df.to_csv(pdp_path, index=False)
    print(f"PDP data written to {pdp_path}")

    # Summary metrics
    metrics = {}
    for region, res in results.items():
        metrics[region] = {
            "r2_trainval": res["r2_trainval"],
            "r2_test": res["r2_test"],
            "mae_test_log10": res["mae_test_log10"],
            "ranking": res["ranking"],
            "top3": res["ranking"][:3],
            "importance": res["importance"],
        }

    # Acceptance checks
    print(f"\n=== Acceptance ===")
    any_r2_pass = any(r["r2_test"] > 0.8 for r in results.values())
    r2_summary = ", ".join(f"{k}={v['r2_test']:.3f}" for k, v in metrics.items())
    print(f"  AC3 R2 > 0.8:          {'PASS' if any_r2_pass else 'FAIL'} ({r2_summary})")

    # AC1: top 3 identified
    top3_all = set()
    for reg_m in metrics.values():
        top3_all.update(reg_m["top3"])
    ac1 = len(top3_all) >= 3
    print(f"  AC1 top3 identified:    {'PASS' if ac1 else 'FAIL'} ({top3_all})")

    # AC2: PDP shows monotonic/non-monotonic trends for >= 2 key params
    n_monotonic_check = 0
    for region, res in results.items():
        for col in res["ranking"][:3]:
            pdp = res["pdp"][col]
            vals = pdp["mean"]
            diffs = np.diff(vals)
            mostly_increasing = (diffs > 0).sum() / max(len(diffs), 1) > 0.7
            mostly_decreasing = (diffs < 0).sum() / max(len(diffs), 1) > 0.7
            if mostly_increasing or mostly_decreasing:
                n_monotonic_check += 1
    ac2 = n_monotonic_check >= 2
    print(f"  AC2 PDP trends (>=2):   {'PASS' if ac2 else 'FAIL'} ({n_monotonic_check} monotonic trends found)")

    metrics["acceptance"] = {"ac1": ac1, "ac2": ac2, "ac3_r2_gt_0_8": any_r2_pass}
    metrics_path = args.output_dir / "h3_sensitivity_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    main()
