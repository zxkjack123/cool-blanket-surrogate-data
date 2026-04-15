#!/usr/bin/env python3
"""P-hat sensitivity analysis: how safe/uncertain/risk fractions change with qhat.

Uses the seed-42 model predictions (tbr_pred from tbr_conformal_results.csv)
and the 100 qhat values from multi_split_results.csv to recompute the
constraint satisfaction partition under each qhat, for thresholds 1.40/1.45/1.50.

This is *not* an end-to-end repeated experiment (each qhat comes from a different
model trained on a different split). It isolates the effect of conformal interval
width on the zone classification, holding the point predictions fixed.

Usage:
    python scripts/experiment_phat_sensitivity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ART_DIR = SCRIPT_DIR.parent / "artifacts"

THRESHOLDS = [1.40, 1.45, 1.50]
SAFE_CUTOFF = 0.95
RISK_CUTOFF = 0.50


def compute_pr_exceeds(
    tbr_pred: np.ndarray, qhat: float, theta: float
) -> np.ndarray:
    """Compute P-hat(TBR > theta) using the linear ramp from conformal interval."""
    ci_lower = tbr_pred - qhat
    ci_upper = tbr_pred + qhat
    ci_width = ci_upper - ci_lower
    pr = np.where(
        ci_lower > theta,
        1.0,
        np.where(
            ci_upper < theta,
            0.0,
            (ci_upper - theta) / np.maximum(ci_width, 1e-12),
        ),
    )
    return np.clip(pr, 0.0, 1.0)


def classify(pr: np.ndarray) -> tuple[int, int, int]:
    """Classify into safe/uncertain/risk counts."""
    n_safe = int(np.sum(pr > SAFE_CUTOFF))
    n_risk = int(np.sum(pr < RISK_CUTOFF))
    n_uncertain = len(pr) - n_safe - n_risk
    return n_safe, n_uncertain, n_risk


def main():
    # Load seed-42 predictions
    conformal_csv = ART_DIR / "tbr_conformal_results.csv"
    if not conformal_csv.exists():
        raise FileNotFoundError(f"Missing: {conformal_csv}")
    df_conf = pd.read_csv(conformal_csv)
    for col in ("tbr_pred", "ci_lower", "ci_upper"):
        if col not in df_conf.columns:
            raise KeyError(f"Missing column '{col}' in {conformal_csv}")
    tbr_pred = df_conf["tbr_pred"].values
    n_samples = len(tbr_pred)
    print(f"Loaded {n_samples} predictions from {conformal_csv.name}")

    # Load 100 qhat values
    multi_csv = ART_DIR / "multi_split_results.csv"
    if not multi_csv.exists():
        raise FileNotFoundError(f"Missing: {multi_csv}")
    df_multi = pd.read_csv(multi_csv)
    if "qhat" not in df_multi.columns:
        raise KeyError(f"Missing column 'qhat' in {multi_csv}")
    qhat_values = df_multi["qhat"].values
    print(f"Loaded {len(qhat_values)} qhat values from {multi_csv.name}")

    # Compute partition for each (qhat, theta) pair
    rows = []
    for qhat in qhat_values:
        for theta in THRESHOLDS:
            pr = compute_pr_exceeds(tbr_pred, qhat, theta)
            n_safe, n_uncertain, n_risk = classify(pr)
            rows.append({
                "qhat": qhat,
                "theta": theta,
                "n_safe": n_safe,
                "n_uncertain": n_uncertain,
                "n_risk": n_risk,
                "frac_safe": n_safe / n_samples,
                "frac_uncertain": n_uncertain / n_samples,
                "frac_risk": n_risk / n_samples,
            })

    df_out = pd.DataFrame(rows)
    csv_path = ART_DIR / "phat_sensitivity.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df_out)} rows)")

    # Summary statistics per threshold
    summary = {}
    for theta in THRESHOLDS:
        sub = df_out[df_out["theta"] == theta]
        key = f"theta_{theta:.2f}"
        for col in ("frac_safe", "frac_uncertain", "frac_risk"):
            vals = sub[col].values
            summary[f"{key}_{col}_mean"] = float(np.mean(vals))
            summary[f"{key}_{col}_std"] = float(np.std(vals))
            summary[f"{key}_{col}_min"] = float(np.min(vals))
            summary[f"{key}_{col}_max"] = float(np.max(vals))
            summary[f"{key}_{col}_p5"] = float(np.percentile(vals, 5))
            summary[f"{key}_{col}_p95"] = float(np.percentile(vals, 95))

    summary["n_qhat_values"] = len(qhat_values)
    summary["n_samples"] = n_samples
    summary["thresholds"] = THRESHOLDS

    json_path = ART_DIR / "phat_sensitivity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
