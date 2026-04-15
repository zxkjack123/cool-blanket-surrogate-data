#!/usr/bin/env python3
"""GAM additive model comparison experiment for C3 validation.

Trains a purely additive surrogate (sum of univariate spline transforms
with Ridge regression) and compares out-of-sample R² to the HGBR model.
If the additive model performs comparably, this provides model-independent
evidence that TBR depends approximately additively on the design parameters.

Usage:
    python scripts/experiment_gam_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

# Reuse data-loading infrastructure from the training script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_tbr_with_uq import load_data, load_splits

ART_DIR = SCRIPT_DIR.parent / "artifacts"
SPLITS_JSON = ART_DIR / "splits.json"
BATCH_NAME = "expandC_act9_n144_20260404"


def build_additive_pipeline():
    """Build a purely additive model: per-feature cubic splines + Ridge.

    Each feature is independently transformed by SplineTransformer,
    then all are concatenated and fed to Ridge regression.
    No interaction terms are created.
    """
    return make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, extrapolation="continue"),
        Ridge(alpha=1.0),
    )


def main():
    # Load data and splits (same as main training script)
    per_sample, feat_cols = load_data(BATCH_NAME)
    splits = load_splits(SPLITS_JSON)

    train_mask = per_sample["sample_id"].isin(splits["train"])
    test_mask = per_sample["sample_id"].isin(splits["test"])

    X_train = per_sample.loc[train_mask, feat_cols].values
    y_train = per_sample.loc[train_mask, "tbr"].values
    X_test = per_sample.loc[test_mask, feat_cols].values
    y_test = per_sample.loc[test_mask, "tbr"].values

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"Features: {feat_cols}")

    # ── Train additive (GAM-like) model ──
    print("\nTraining additive model (SplineTransformer + Ridge) ...")
    additive_model = build_additive_pipeline()
    additive_model.fit(X_train, y_train)
    y_test_pred_add = additive_model.predict(X_test)
    r2_gam = r2_score(y_test, y_test_pred_add)
    mae_gam = mean_absolute_error(y_test, y_test_pred_add)
    print(f"  Additive R² = {r2_gam:.4f}, MAE = {mae_gam:.6f}")

    # ── Train HGBR (same hyperparameters as main training script) ──
    print("\nTraining HGBR ...")
    hgbr = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=4,
        max_iter=1200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=60,
        min_samples_leaf=2,
    )
    hgbr.fit(X_train, y_train)
    y_test_pred_hgbr = hgbr.predict(X_test)
    r2_hgbr = r2_score(y_test, y_test_pred_hgbr)
    mae_hgbr = mean_absolute_error(y_test, y_test_pred_hgbr)
    print(f"  HGBR R² = {r2_hgbr:.4f}, MAE = {mae_hgbr:.6f}")

    # ── Compare ──
    r2_diff = r2_hgbr - r2_gam
    print(f"\n=== Comparison ===")
    print(f"  R² difference (HGBR - Additive) = {r2_diff:.4f}")
    print(f"  MAE difference (HGBR - Additive) = {mae_hgbr - mae_gam:.6f}")

    if abs(r2_diff) < 0.02:
        conclusion = "C3_independently_confirmed"
        print("  → Additive model matches HGBR: C3 (near-additivity) independently confirmed")
    else:
        conclusion = "C3_qualified_as_model_observation"
        print(f"  → Additive model differs from HGBR by {r2_diff:.4f}: "
              "C3 should be qualified as model observation")

    if r2_gam < 0.50:
        print("  WARNING: Additive R² < 0.50 — main effects alone explain less than half the variance")

    # ── Save results ──
    results = {
        "r2_gam": float(r2_gam),
        "r2_hgbr": float(r2_hgbr),
        "r2_difference": float(r2_diff),
        "mae_gam": float(mae_gam),
        "mae_hgbr": float(mae_hgbr),
        "conclusion": conclusion,
        "additive_model": "SplineTransformer(n_knots=5, degree=3) + Ridge(alpha=1.0)",
        "hgbr_params": "lr=0.05, max_depth=4, max_iter=1200, early_stopping=True",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    out_path = ART_DIR / "gam_comparison_metrics.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
