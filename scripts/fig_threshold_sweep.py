"""TBR threshold continuous sweep: design space feasibility vs target θ.

Stacked area chart showing fraction of 144 designs classified as
safe/uncertain/risk as function of TBR threshold θ (1.05–1.60).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_DIR = ROOT / "figures"
ART_DIR = ROOT / "artifacts"
FIG_DIR.mkdir(exist_ok=True)

LANDSCAPE_CSV = ART_DIR / "constraint_landscape.csv"
assert LANDSCAPE_CSV.exists(), f"Missing {LANDSCAPE_CSV}"


def compute_pr(tbr_pred: np.ndarray, qhat: np.ndarray, theta: float) -> np.ndarray:
    """Compute Pr(TBR > theta) via conformal linear ramp.

    Pr = 1 if lower > theta (safe)
    Pr = 0 if upper < theta (risk)
    Pr linear in between.
    """
    lower = tbr_pred - qhat
    upper = tbr_pred + qhat
    pr = np.clip((upper - theta) / (upper - lower + 1e-12), 0.0, 1.0)
    return pr


def main():
    import matplotlib.pyplot as plt

    df = pd.read_csv(LANDSCAPE_CSV)
    n = len(df)
    print(f"Loaded {n} samples from {LANDSCAPE_CSV.name}")

    tbr_pred = df["tbr_pred"].values
    qhat = (df["ci_upper"].values - df["ci_lower"].values) / 2.0

    # Sweep θ
    thetas = np.linspace(1.05, 1.60, 111)
    fracs = {"theta": [], "frac_safe": [], "frac_uncertain": [], "frac_risk": []}

    for theta in thetas:
        pr = compute_pr(tbr_pred, qhat, theta)
        n_safe = np.sum(pr > 0.95)
        n_risk = np.sum(pr < 0.50)
        n_uncertain = n - n_safe - n_risk
        fracs["theta"].append(theta)
        fracs["frac_safe"].append(n_safe / n * 100)
        fracs["frac_uncertain"].append(n_uncertain / n * 100)
        fracs["frac_risk"].append(n_risk / n * 100)

    df_sweep = pd.DataFrame(fracs)
    df_sweep.to_csv(ART_DIR / "threshold_sweep.csv", index=False)
    print(f"Saved threshold sweep to {ART_DIR / 'threshold_sweep.csv'}")

    # Verify known points
    for t_check in [1.40, 1.50]:
        row = df_sweep.iloc[(df_sweep["theta"] - t_check).abs().idxmin()]
        print(f"θ={t_check:.2f}: safe={row['frac_safe']:.1f}%, "
              f"uncertain={row['frac_uncertain']:.1f}%, risk={row['frac_risk']:.1f}%")

    # ── Plot stacked area ──
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    theta_arr = df_sweep["theta"].values
    safe_arr = df_sweep["frac_safe"].values
    uncertain_arr = df_sweep["frac_uncertain"].values
    risk_arr = df_sweep["frac_risk"].values

    ax.fill_between(theta_arr, 0, safe_arr,
                    color="#4CAF50", alpha=0.7, label="Safe (Pr > 0.95)")
    ax.fill_between(theta_arr, safe_arr, safe_arr + uncertain_arr,
                    color="#FFC107", alpha=0.7, label="Uncertain (0.50 ≤ Pr ≤ 0.95)")
    ax.fill_between(theta_arr, safe_arr + uncertain_arr, 100,
                    color="#F44336", alpha=0.7, label="Risk (Pr < 0.50)")

    # Reference lines for key engineering thresholds
    ref_thetas = [
        (1.05, "Min. self-sufficiency", "left"),
        (1.15, "Fuel cycle losses", "left"),
        (1.40, "Conservative margin", "right"),
    ]
    for t_ref, label, side in ref_thetas:
        ax.axvline(t_ref, color="black", ls="--", lw=0.8, alpha=0.6)
        if side == "right":
            ax.text(t_ref - 0.005, 102, label, fontsize=7.5, rotation=0,
                    va="bottom", ha="right", color="#333333")
        else:
            ax.text(t_ref + 0.005, 102, label, fontsize=7.5, rotation=0,
                    va="bottom", ha="left", color="#333333")

    ax.set_xlabel(r"TBR threshold $\theta$", fontsize=11)
    ax.set_ylabel("Fraction of design space (%)", fontsize=11)
    ax.set_xlim(1.05, 1.60)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("Design space feasibility vs TBR target threshold", fontsize=12)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_threshold_sweep.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_threshold_sweep.png", bbox_inches="tight",
                dpi=300, facecolor="white")
    print(f"Saved figure to {FIG_DIR / 'fig_threshold_sweep.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
