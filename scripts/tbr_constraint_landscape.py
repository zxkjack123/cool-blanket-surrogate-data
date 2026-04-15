#!/usr/bin/env python3
"""Compute TBR constraint satisfaction probability landscape (C1 core).

For each sample and threshold theta, computes Pr(TBR > theta | x) using
the conformal prediction intervals from Task 2.1.

Usage:
    python scripts/tbr_constraint_landscape.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DESIGN_COLS = [
    "FW_THICK_CM",
    "PBLI_THICK_CM",
    "SHIELD_THICK_CM",
    "VV_THICK_CM",
    "LI6_ENRICH_ATOM_FRAC",
]

# Standard engineering TBR thresholds + discriminative thresholds for this dataset
DEFAULT_THRESHOLDS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]


def compute_pr_exceeds(ci_lower: np.ndarray, ci_upper: np.ndarray,
                       theta: float) -> np.ndarray:
    """Compute Pr(TBR > theta) for each sample using linear interpolation
    within the conformal interval.

    Returns array of probabilities in [0, 1].
    """
    pr = np.where(
        ci_lower > theta,
        1.0,
        np.where(
            ci_upper < theta,
            0.0,
            (ci_upper - theta) / np.maximum(ci_upper - ci_lower, 1e-12),
        ),
    )
    return np.clip(pr, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="TBR constraint landscape")
    parser.add_argument(
        "--conformal-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "tbr_conformal_results.csv",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "constraint_landscape.csv",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    print(f"Loading conformal results: {args.conformal_csv}")
    df = pd.read_csv(args.conformal_csv)
    print(f"  Samples: {len(df)}, CI width: {df['ci_width'].iloc[0]:.6f}")

    ci_lower = df["ci_lower"].values
    ci_upper = df["ci_upper"].values

    # Compute Pr(TBR > theta) for each threshold
    rows = []
    for theta in args.thresholds:
        pr = compute_pr_exceeds(ci_lower, ci_upper, theta)
        col_name = f"pr_tbr_gt_{theta:.2f}"
        df[col_name] = pr

    # Build output
    out_cols = ["sample_id", "split"] + DESIGN_COLS + ["tbr", "tbr_pred", "ci_lower", "ci_upper"]
    pr_cols = [c for c in df.columns if c.startswith("pr_tbr_gt_")]
    out_cols += pr_cols
    out_df = df[out_cols].sort_values("sample_id")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\nLandscape written to {args.output}")
    print(f"  Shape: {out_df.shape}")

    # Diagnostics
    print(f"\n=== Discrimination Analysis ===")
    discriminative_thresholds = []
    for theta in args.thresholds:
        col = f"pr_tbr_gt_{theta:.2f}"
        vals = df[col]
        n_zero = (vals == 0).sum()
        n_one = (vals == 1).sum()
        n_between = len(vals) - n_zero - n_one
        span = vals.max() - vals.min()
        print(f"  theta={theta:.2f}: Pr=0:{n_zero}, Pr in (0,1):{n_between}, Pr=1:{n_one}, span={span:.3f}")
        if n_between >= 5 and span > 0.3:
            discriminative_thresholds.append(theta)

    print(f"\n  Discriminative thresholds (>=5 intermediate, span>0.3): {discriminative_thresholds}")

    # Physical consistency check: Pr(TBR>theta) should correlate positively with Li6 enrichment
    print(f"\n=== Physical Consistency ===")
    li6 = df["LI6_ENRICH_ATOM_FRAC"]
    correlation_results = {}
    for theta in discriminative_thresholds[:3]:  # check top 3
        col = f"pr_tbr_gt_{theta:.2f}"
        r, p = stats.spearmanr(li6, df[col])
        print(f"  Spearman(Li6, Pr(TBR>{theta:.2f})): r={r:.3f}, p={p:.4f}")
        correlation_results[f"spearman_li6_theta_{theta:.2f}"] = {"r": float(r), "p": float(p)}

    # Acceptance checks
    print(f"\n=== Acceptance ===")
    ac1 = len(out_df) == len(df) and len(pr_cols) == len(args.thresholds)
    print(f"  AC1 n_samples x n_thresholds:  {'PASS' if ac1 else 'FAIL'} ({len(out_df)} x {len(pr_cols)})")

    ac2 = len(discriminative_thresholds) >= 1
    print(f"  AC2 discriminative theta exists: {'PASS' if ac2 else 'FAIL'} ({len(discriminative_thresholds)} found)")

    # AC3: Pr positively correlates with Li6
    ac3 = any(v["r"] > 0 and v["p"] < 0.05 for v in correlation_results.values())
    print(f"  AC3 Li6-Pr positive corr:       {'PASS' if ac3 else 'FAIL'}")

    # Write metrics
    metrics = {
        "n_samples": len(out_df),
        "n_thresholds": len(args.thresholds),
        "thresholds": args.thresholds,
        "discriminative_thresholds": discriminative_thresholds,
        "correlation_results": correlation_results,
        "ac1_pass": ac1,
        "ac2_pass": ac2,
        "ac3_pass": ac3,
    }
    metrics_path = args.metrics_output or (args.output.parent / "constraint_landscape_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    main()
