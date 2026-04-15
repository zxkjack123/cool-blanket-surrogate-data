#!/usr/bin/env python3
"""Create train/calibration/test splits for expandC n144 dataset.

Usage:
    python scripts/create_splits.py --batch expandC_act9_n144_20260404 --seed 42
    python scripts/create_splits.py --batch expandC_act9_n144_20260404 --seed 42 --output artifacts/splits.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BATCH_ROOT = Path("/home/gw/ComputeData/CFETR/COOL-PbLi-Burnup/datasets/wp2_batches")
BREEDING_REGION = "cell_3"

DESIGN_PARAMS = [
    "FW_THICK_CM",
    "PBLI_THICK_CM",
    "SHIELD_THICK_CM",
    "VV_THICK_CM",
    "LI6_ENRICH_ATOM_FRAC",
]


def extract_design_params(df: pd.DataFrame) -> pd.DataFrame:
    """Extract design-level parameters from the label data.

    Since we don't have a separate design matrix CSV, we infer design params
    from per-sample TBR and Li6 enrichment. For the full 5-param extraction,
    we'd need the parameter_space YAML or a design CSV.

    For now, use sample_id as the grouping key and TBR (cell_3) as a proxy.
    """
    cell3 = df[df["region"] == BREEDING_REGION].copy()
    per_sample = cell3.groupby("sample_id").agg(
        tbr=("tbr", "first"),
        h3_peak=("atoms_H3", "max"),
    ).reset_index()
    return per_sample


def create_splits(
    n_samples: int,
    sample_ids: np.ndarray,
    seed: int = 42,
    train_frac: float = 0.80,
    cal_frac: float = 0.10,
) -> dict:
    """Create stratified train/calibration/test split.

    Split is 80/10/10 by default. Calibration set is used for conformal
    prediction (separate from validation).
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)

    n_test = max(int(n_samples * (1 - train_frac - cal_frac)), 1)
    n_cal = max(int(n_samples * cal_frac), 1)
    n_train = n_samples - n_test - n_cal

    train_idx = indices[:n_train]
    cal_idx = indices[n_train : n_train + n_cal]
    test_idx = indices[n_train + n_cal :]

    return {
        "train": sorted(sample_ids[train_idx].tolist()),
        "calibration": sorted(sample_ids[cal_idx].tolist()),
        "test": sorted(sample_ids[test_idx].tolist()),
        "metadata": {
            "seed": seed,
            "n_total": n_samples,
            "n_train": n_train,
            "n_calibration": n_cal,
            "n_test": len(test_idx),
            "train_frac": n_train / n_samples,
            "cal_frac": n_cal / n_samples,
            "test_frac": len(test_idx) / n_samples,
        },
    }


def validate_splits(splits: dict, df: pd.DataFrame) -> list[str]:
    """Validate split quality."""
    issues = []

    all_ids = set(splits["train"]) | set(splits["calibration"]) | set(splits["test"])
    train_set = set(splits["train"])
    cal_set = set(splits["calibration"])
    test_set = set(splits["test"])

    # No leakage
    if train_set & test_set:
        issues.append(f"LEAKAGE: {len(train_set & test_set)} samples in both train and test")
    if train_set & cal_set:
        issues.append(f"LEAKAGE: {len(train_set & cal_set)} samples in both train and cal")
    if cal_set & test_set:
        issues.append(f"LEAKAGE: {len(cal_set & test_set)} samples in both cal and test")

    # Test set size
    if len(test_set) < 14:
        issues.append(f"Test set too small: {len(test_set)} (need ≥ 14)")

    # TBR distribution check (KS test: train vs test)
    cell3 = df[df["region"] == BREEDING_REGION]
    tbr_per_sample = cell3.groupby("sample_id")["tbr"].first()

    train_tbr = tbr_per_sample[tbr_per_sample.index.isin(train_set)].values
    test_tbr = tbr_per_sample[tbr_per_sample.index.isin(test_set)].values

    if len(train_tbr) > 0 and len(test_tbr) > 0:
        ks_stat, ks_pval = stats.ks_2samp(train_tbr, test_tbr)
        if ks_pval < 0.05:
            issues.append(f"TBR distribution mismatch (KS p={ks_pval:.4f})")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Create splits for expandC dataset")
    parser.add_argument("--batch", required=True, help="Batch directory name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output path for splits.json")
    args = parser.parse_args()

    # Load data
    batch_dir = BATCH_ROOT / args.batch
    csv_files = list(batch_dir.glob("labels_long_*.csv"))
    if not csv_files:
        sys.exit(f"No labels CSV in {batch_dir}")

    print(f"Loading {csv_files[0]} ...")
    df = pd.read_csv(csv_files[0])
    df["sample_id"] = df["run_id"].str.extract(r"_a(\d+)$").astype(int)

    sample_ids = np.sort(df["sample_id"].unique())
    n_samples = len(sample_ids)
    print(f"Found {n_samples} samples")

    # Create splits
    splits = create_splits(n_samples, sample_ids, seed=args.seed)
    print(f"Split: train={splits['metadata']['n_train']}, "
          f"cal={splits['metadata']['n_calibration']}, "
          f"test={splits['metadata']['n_test']}")

    # Validate
    issues = validate_splits(splits, df)
    if issues:
        print("VALIDATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Validation passed: no leakage, test ≥ 14, TBR distribution OK")

    # Compute SHA256
    splits_json = json.dumps(splits, indent=2, sort_keys=True)
    sha256 = hashlib.sha256(splits_json.encode()).hexdigest()
    splits["metadata"]["sha256"] = sha256

    # Write output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Written to {out_path}")
        print(f"SHA256: {sha256}")
    else:
        print(json.dumps(splits, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
