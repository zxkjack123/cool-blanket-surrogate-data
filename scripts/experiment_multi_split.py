#!/usr/bin/env python3
"""100× multi-split validation experiment.

Repeats the train/calibration/test split 100 times with different seeds,
trains HGBR each time, and reports the distribution of R², MAE, qhat, PICP.
This quantifies how sensitive the reported metrics are to the random split.

Usage:
    python scripts/experiment_multi_split.py
    python scripts/experiment_multi_split.py --n-splits 100
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from create_splits import create_splits
from train_tbr_with_uq import load_data, split_conformal

ART_DIR = SCRIPT_DIR.parent / "artifacts"
BATCH_NAME = "expandC_act9_n144_20260404"


def run_one_split(
    per_sample: pd.DataFrame,
    feat_cols: list[str],
    sample_ids: np.ndarray,
    seed: int,
    alpha: float = 0.10,
) -> dict:
    """Train HGBR on one random split and return metrics."""
    splits = create_splits(
        n_samples=len(sample_ids),
        sample_ids=sample_ids,
        seed=seed,
    )

    train_mask = per_sample["sample_id"].isin(splits["train"])
    cal_mask = per_sample["sample_id"].isin(splits["calibration"])
    test_mask = per_sample["sample_id"].isin(splits["test"])

    X_train = per_sample.loc[train_mask, feat_cols].values
    y_train = per_sample.loc[train_mask, "tbr"].values
    X_cal = per_sample.loc[cal_mask, feat_cols].values
    y_cal = per_sample.loc[cal_mask, "tbr"].values
    X_test = per_sample.loc[test_mask, feat_cols].values
    y_test = per_sample.loc[test_mask, "tbr"].values

    # Train HGBR with same hyperparameters as main script
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
    model.fit(X_train, y_train)

    y_cal_pred = model.predict(X_cal)
    y_test_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    # Conformal
    qhat = split_conformal(y_cal, y_cal_pred, alpha=alpha)
    ci_width = 2.0 * qhat
    test_covered = (
        (y_test >= y_test_pred - qhat) & (y_test <= y_test_pred + qhat)
    ).mean()

    # Response range and interval/range ratio
    y_range = float(y_test.max() - y_test.min())
    width_range_ratio = ci_width / y_range if y_range > 0 else float("inf")

    return {
        "seed": int(seed),
        "r2": float(r2),
        "mae": float(mae),
        "qhat": float(qhat),
        "ci_width": float(ci_width),
        "picp": float(test_covered),
        "y_range": float(y_range),
        "width_range_ratio": float(width_range_ratio),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    per_sample, feat_cols = load_data(BATCH_NAME)
    sample_ids = per_sample["sample_id"].values

    print(f"Running {args.n_splits} random splits ...")
    results = []
    for i in range(args.n_splits):
        seed = i + 1
        row = run_one_split(per_sample, feat_cols, sample_ids, seed, args.alpha)
        results.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.n_splits} done")

    df = pd.DataFrame(results)

    # Save full results
    csv_path = ART_DIR / "multi_split_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    # Compute summary statistics
    summary = {}
    for col in ["r2", "mae", "qhat", "ci_width", "picp", "width_range_ratio"]:
        vals = df[col].values
        summary[f"{col}_mean"] = float(np.mean(vals))
        summary[f"{col}_std"] = float(np.std(vals))
        summary[f"{col}_p5"] = float(np.percentile(vals, 5))
        summary[f"{col}_p25"] = float(np.percentile(vals, 25))
        summary[f"{col}_p50"] = float(np.percentile(vals, 50))
        summary[f"{col}_p75"] = float(np.percentile(vals, 75))
        summary[f"{col}_p95"] = float(np.percentile(vals, 95))

    summary["n_splits"] = args.n_splits
    summary["alpha"] = args.alpha

    # Check if seed=42 (original) falls within p5-p95
    orig_seed42 = df[df["seed"] == 42]
    if not orig_seed42.empty:
        r2_42 = orig_seed42.iloc[0]["r2"]
        summary["seed42_r2"] = float(r2_42)
        summary["seed42_in_p5_p95"] = bool(
            summary["r2_p5"] <= r2_42 <= summary["r2_p95"]
        )
    else:
        summary["seed42_r2"] = None
        summary["seed42_in_p5_p95"] = None

    json_path = ART_DIR / "multi_split_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {json_path}")

    # Print summary
    print(f"\n=== Multi-Split Summary ({args.n_splits} splits) ===")
    print(f"  R²:   {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}  "
          f"[p5={summary['r2_p5']:.4f}, p95={summary['r2_p95']:.4f}]")
    print(f"  MAE:  {summary['mae_mean']:.6f} ± {summary['mae_std']:.6f}")
    print(f"  qhat: {summary['qhat_mean']:.4f} ± {summary['qhat_std']:.4f}")
    print(f"  PICP: {summary['picp_mean']:.3f} ± {summary['picp_std']:.3f}  "
          f"[p5={summary['picp_p5']:.3f}, p95={summary['picp_p95']:.3f}]")
    if summary.get("seed42_in_p5_p95") is not None:
        print(f"  Seed-42 R² = {summary['seed42_r2']:.4f}, "
              f"in p5-p95 range: {summary['seed42_in_p5_p95']}")


if __name__ == "__main__":
    main()
