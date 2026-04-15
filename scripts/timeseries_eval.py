#!/usr/bin/env python3
"""Time-series prediction evaluation with UQ (supports C1+C2).

Trains per-timepoint HGBR surrogates for H3, Li6, Li7 and evaluates
accuracy + conformal UQ coverage on the expandC dataset.

Usage:
    python scripts/timeseries_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
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

TARGETS = {
    "cell_3": ["atoms_H3", "atoms_Li6", "atoms_Li7"],
    "cell_2": ["atoms_H3"],
}

# Select ~15 representative timepoints (log-spaced through 130 steps)
N_TIMEPOINTS = 15


def load_data(batch_name: str) -> pd.DataFrame:
    """Load full labels + join LHS design params."""
    batch_dir = BATCH_ROOT / batch_name
    csv_path = next(batch_dir.glob("labels_long_*.csv"))
    df = pd.read_csv(csv_path)
    df["a_idx"] = df["run_id"].str.extract(r"_a(\d+)$").astype(int)

    lhs = pd.read_csv(LHS_CSV)
    lhs["s_idx"] = lhs["sample_id"].str.extract(r"s(\d+)$").astype(int)
    base_offset = int(lhs["s_idx"].min())
    lhs["a_idx"] = lhs["s_idx"] - base_offset
    df = df.merge(lhs[["a_idx"] + DESIGN_COLS], on="a_idx", how="left")
    return df


def select_timepoints(times: np.ndarray, n: int = N_TIMEPOINTS) -> np.ndarray:
    """Select n log-spaced timepoints from available times."""
    unique_times = np.sort(np.unique(times))
    if len(unique_times) <= n:
        return unique_times
    indices = np.unique(np.round(np.linspace(0, len(unique_times) - 1, n)).astype(int))
    return unique_times[indices]


def split_conformal_qhat(y_cal: np.ndarray, y_cal_pred: np.ndarray, alpha: float = 0.10) -> float:
    """Compute conformal quantile from calibration residuals."""
    residuals = np.abs(y_cal - y_cal_pred)
    n = len(residuals)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(residuals, q_level, method="higher"))


def evaluate_target_timeseries(
    df: pd.DataFrame,
    region: str,
    target: str,
    timepoints: np.ndarray,
    splits: dict,
    seed: int = 42,
    alpha: float = 0.10,
) -> dict:
    """Train per-timepoint HGBR and evaluate accuracy + UQ."""
    sub = df[df["region"] == region].copy()
    # Drop NaN rows for this target
    sub = sub.dropna(subset=[target])
    if sub.empty:
        return {"skip": True, "reason": f"no data for {target} in {region}"}

    results = []
    representative_curves = {}  # sample_id -> list of (time, true, pred, ci_lo, ci_hi)

    # Pick 3 test samples for representative curves
    test_ids = splits["test"][:3]

    for t in timepoints:
        t_data = sub[sub["time_s"] == t].copy()
        if len(t_data) < 10:
            continue

        y_raw = t_data[target].values
        # Use log10(1 + y) transform for numerical stability
        y = np.log10(1 + y_raw)

        X = t_data[DESIGN_COLS].values
        a_idx = t_data["a_idx"].values

        train_mask = np.isin(a_idx, splits["train"])
        cal_mask = np.isin(a_idx, splits["calibration"])
        test_mask = np.isin(a_idx, splits["test"])

        if train_mask.sum() < 5 or cal_mask.sum() < 3 or test_mask.sum() < 3:
            continue

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=4,
            max_iter=800,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            min_samples_leaf=2,
        )
        model.fit(X[train_mask], y[train_mask])

        y_pred = model.predict(X)
        y_cal_pred = y_pred[cal_mask]
        y_test_pred = y_pred[test_mask]

        qhat = split_conformal_qhat(y[cal_mask], y_cal_pred, alpha=alpha)

        # Metrics on test set
        y_t = y[test_mask]
        mae = float(mean_absolute_error(y_t, y_test_pred))
        r2 = float(r2_score(y_t, y_test_pred)) if len(y_t) > 1 else float("nan")
        covered = float(((y_t >= y_test_pred - qhat) & (y_t <= y_test_pred + qhat)).mean())

        # abs log10 ratio (on original scale)
        y_raw_test = y_raw[test_mask]
        y_raw_pred = 10 ** y_test_pred - 1
        eps = 1e-12
        alr = np.abs(np.log10((y_raw_pred + eps) / (y_raw_test + eps)))
        median_alr = float(np.median(alr))
        q90_alr = float(np.percentile(alr, 90))

        results.append({
            "region": region,
            "target": target,
            "time_s": float(t),
            "time_yr": float(t / 3.1536e7),
            "mae_log": mae,
            "r2": r2,
            "picp_90": covered,
            "qhat": float(qhat),
            "median_alr": median_alr,
            "q90_alr": q90_alr,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })

        # Representative curves
        for sid in test_ids:
            idx = np.where(a_idx == sid)[0]
            if len(idx) == 0:
                continue
            i = idx[0]
            if sid not in representative_curves:
                representative_curves[sid] = []
            representative_curves[sid].append({
                "time_s": float(t),
                "time_yr": float(t / 3.1536e7),
                "true": float(y_raw[i]),
                "pred": float(10 ** y_pred[i] - 1),
                "ci_lower": float(10 ** (y_pred[i] - qhat) - 1),
                "ci_upper": float(10 ** (y_pred[i] + qhat) - 1),
            })

    return {
        "skip": False,
        "metrics": results,
        "curves": representative_curves,
    }


def check_monotonicity(df: pd.DataFrame, region: str, target: str, splits: dict) -> dict:
    """Check if Li6/Li7 are monotonically decreasing over time for each sample."""
    sub = df[(df["region"] == region)].dropna(subset=[target])
    violations = 0
    total = 0
    for sid in splits["test"]:
        s_data = sub[sub["a_idx"] == sid].sort_values("time_s")
        if len(s_data) < 2:
            continue
        y = s_data[target].values
        diffs = np.diff(y)
        total += 1
        if (diffs > 0).any():
            violations += 1
    return {"violations": violations, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Time-series evaluation with UQ")
    parser.add_argument("--batch", default="expandC_act9_n144_20260404")
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "splits.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts",
    )
    args = parser.parse_args()

    print(f"Loading data: {args.batch}")
    df = load_data(args.batch)
    print(f"  Total rows: {len(df)}")

    splits_raw = json.loads(args.splits_json.read_text(encoding="utf-8"))
    splits = {
        "train": [int(x) for x in splits_raw["train"]],
        "calibration": [int(x) for x in splits_raw["calibration"]],
        "test": [int(x) for x in splits_raw["test"]],
    }

    all_metrics = []
    all_curves = {}  # (region, target) -> {sample_id: [curve_pts]}

    for region, targets in TARGETS.items():
        sub = df[df["region"] == region]
        times = sub["time_s"].unique()
        timepoints = select_timepoints(times)
        print(f"\n[{region}] {len(timepoints)} timepoints")

        for target in targets:
            print(f"  {target}...")
            result = evaluate_target_timeseries(
                df, region, target, timepoints, splits, args.seed, args.alpha
            )
            if result["skip"]:
                print(f"    SKIPPED: {result['reason']}")
                continue

            all_metrics.extend(result["metrics"])
            all_curves[(region, target)] = result["curves"]

            # Summary
            mets = result["metrics"]
            avg_picp = np.mean([m["picp_90"] for m in mets])
            avg_mae = np.mean([m["mae_log"] for m in mets])
            avg_r2 = np.mean([m["r2"] for m in mets if not np.isnan(m["r2"])])
            print(f"    avg PICP@90: {avg_picp:.1%}, avg MAE(log): {avg_mae:.4f}, avg R2: {avg_r2:.3f}")

    # Write accuracy metrics CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = args.output_dir / "timeseries_accuracy.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nAccuracy metrics written to {metrics_path}")

    # Write representative curves CSV
    curve_rows = []
    for (region, target), curves in all_curves.items():
        for sid, pts in curves.items():
            for pt in pts:
                curve_rows.append({
                    "region": region,
                    "target": target,
                    "sample_id": sid,
                    **pt,
                })
    curves_df = pd.DataFrame(curve_rows)
    curves_path = args.output_dir / "timeseries_curves.csv"
    curves_df.to_csv(curves_path, index=False)
    print(f"Curves written to {curves_path} ({len(curves_df)} points, {curves_df['sample_id'].nunique()} samples)")

    # Monotonicity check for Li6/Li7 in raw data
    print(f"\n=== Monotonicity Check (raw data) ===")
    mono_results = {}
    for target in ["atoms_Li6", "atoms_Li7"]:
        mc = check_monotonicity(df, "cell_3", target, splits)
        mono_results[target] = mc
        print(f"  {target}: violations={mc['violations']}/{mc['total']}")

    # Acceptance
    print(f"\n=== Acceptance ===")
    ac1 = len(metrics_df) > 0
    print(f"  AC1 accuracy CSV:       {'PASS' if ac1 else 'FAIL'} ({len(metrics_df)} rows)")

    n_curve_samples = curves_df["sample_id"].nunique() if not curves_df.empty else 0
    ac2 = n_curve_samples >= 3
    print(f"  AC2 >= 3 curves:        {'PASS' if ac2 else 'FAIL'} ({n_curve_samples} samples)")

    ac3 = all(v["violations"] == 0 for v in mono_results.values())
    print(f"  AC3 monotonic Li6/Li7:  {'PASS' if ac3 else 'FAIL'}")

    # H3 coverage gate
    h3_mets = [m for m in all_metrics if "H3" in m["target"]]
    if h3_mets:
        avg_h3_picp = np.mean([m["picp_90"] for m in h3_mets])
        h3_pass = avg_h3_picp >= 0.85
        print(f"  H3 avg PICP@90:         {'PASS' if h3_pass else 'FAIL'} ({avg_h3_picp:.1%})")

    # Write summary metrics JSON
    summary = {
        "n_metrics_rows": len(metrics_df),
        "n_curve_samples": n_curve_samples,
        "monotonicity": mono_results,
        "ac1": ac1,
        "ac2": ac2,
        "ac3": ac3,
    }
    summary_path = args.output_dir / "timeseries_eval_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
