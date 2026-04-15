"""Parameter interaction heatmap for TBR surrogate.

Computes pairwise interaction strength using 2D partial-dependence variance
decomposition (simplified H-statistic), plus diagonal = permutation importance.
Produces 5×5 symmetric heatmap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import r2_score

# ── paths ──
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_DIR = ROOT / "figures"
ART_DIR = ROOT / "artifacts"
FIG_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)

DATA_ROOT = Path("/home/gw/ComputeData/CFETR/COOL-PbLi-Burnup")
BATCH_ROOT = DATA_ROOT / "datasets" / "wp2_batches"
LHS_CSV = DATA_ROOT / "datasets" / "wp2_lhs_cool_csg_expandC_n144_seed20260330.csv"
SPLITS_PATH = ART_DIR / "splits.json"

FEATURE_COLS = [
    "FW_THICK_CM",
    "PBLI_THICK_CM",
    "SHIELD_THICK_CM",
    "VV_THICK_CM",
    "LI6_ENRICH_ATOM_FRAC",
]

NICE_NAMES = [
    r"$d_\mathrm{FW}$",
    r"$d_\mathrm{PbLi}$",
    r"$d_\mathrm{Sh}$",
    r"$d_\mathrm{VV}$",
    r"$f_\mathrm{Li6}$",
]


def load_data():
    """Load TBR per-sample data with design features."""
    batch_dir = BATCH_ROOT / "expandC_act9_n144_20260404"
    csv_path = next(batch_dir.glob("labels_long_*.csv"))
    df = pd.read_csv(csv_path)
    df["a_idx"] = df["run_id"].str.extract(r"_a(\d+)$").astype(int)

    cell3 = df[df["region"] == "cell_3"].copy()
    per_sample = cell3.groupby("a_idx").agg(tbr=("tbr", "first")).reset_index()

    lhs = pd.read_csv(LHS_CSV)
    lhs["s_idx"] = lhs["sample_id"].str.extract(r"s(\d+)$").astype(int)
    base_offset = int(lhs["s_idx"].min())
    lhs["a_idx"] = lhs["s_idx"] - base_offset

    per_sample = per_sample.merge(lhs[["a_idx"] + FEATURE_COLS], on="a_idx", how="left")
    assert per_sample[FEATURE_COLS].notna().all().all(), "Missing design params"
    return per_sample


def main():
    import matplotlib.pyplot as plt

    # ── Load & split ──
    data = load_data()
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_ids = [int(x) for x in splits["train"]]
    test_ids = [int(x) for x in splits["test"]]

    train = data[data["a_idx"].isin(train_ids)]
    test = data[data["a_idx"].isin(test_ids)]

    X_train = train[FEATURE_COLS].values
    y_train = train["tbr"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["tbr"].values

    # ── Train HGBR ──
    model = HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"TBR model R² on test: {r2:.4f}")

    n_features = len(FEATURE_COLS)

    # ── Permutation importance (diagonal) ──
    perm = permutation_importance(model, X_test, y_test, n_repeats=30,
                                  random_state=42, scoring="r2")
    diag_importance = perm.importances_mean   # shape (n_features,)

    # ── Pairwise interaction via PDP variance decomposition ──
    # For each pair (i,j), compute:
    #   interaction_ij = Var(PDP_2d(i,j)) - Var(PDP_1d(i)) - Var(PDP_1d(j))
    # Simplified H-statistic approach

    # Precompute 1D PDPs
    pdp_1d_var = {}
    for i in range(n_features):
        result = partial_dependence(model, X_train, features=[i],
                                    kind="average", grid_resolution=20)
        pdp_1d_var[i] = np.var(result["average"][0])

    # Compute 2D PDP and interaction for each pair
    interaction_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        interaction_matrix[i, i] = diag_importance[i]
        for j in range(i + 1, n_features):
            result_2d = partial_dependence(model, X_train, features=[i, j],
                                           kind="average", grid_resolution=20)
            var_2d = np.var(result_2d["average"][0])
            interaction_ij = max(0.0, var_2d - pdp_1d_var[i] - pdp_1d_var[j])
            interaction_matrix[i, j] = interaction_ij
            interaction_matrix[j, i] = interaction_ij

    # ── Save CSV ──
    df_mat = pd.DataFrame(interaction_matrix, index=FEATURE_COLS, columns=FEATURE_COLS)
    df_mat.to_csv(ART_DIR / "param_interaction_matrix.csv")
    print(f"Saved interaction matrix to {ART_DIR / 'param_interaction_matrix.csv'}")
    print("Interaction matrix:\n", df_mat.round(6))

    # ── Find strongest interaction pair ──
    mask_upper = np.triu(np.ones_like(interaction_matrix, dtype=bool), k=1)
    off_diag = interaction_matrix.copy()
    off_diag[~mask_upper] = -1
    idx_flat = np.argmax(off_diag)
    row, col = np.unravel_index(idx_flat, off_diag.shape)
    print(f"Strongest interaction: {FEATURE_COLS[row]} × {FEATURE_COLS[col]} "
          f"= {interaction_matrix[row, col]:.6f}")

    # ── Plot heatmap ──
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # Mask diagonal for the off-diagonal colormap
    off_diag_vals = interaction_matrix.copy()
    np.fill_diagonal(off_diag_vals, np.nan)

    im = ax.imshow(interaction_matrix, cmap="YlOrRd", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Interaction strength / importance", fontsize=10)

    # Annotate cells
    for i in range(n_features):
        for j in range(n_features):
            val = interaction_matrix[i, j]
            text_color = "white" if val > 0.5 * interaction_matrix.max() else "black"
            if i == j:
                ax.text(j, i, f"{val:.4f}\n(perm.imp.)",
                        ha="center", va="center", fontsize=7.5, color=text_color)
            else:
                ax.text(j, i, f"{val:.5f}",
                        ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(NICE_NAMES, fontsize=10)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(NICE_NAMES, fontsize=10)
    ax.set_title("TBR parameter interaction strength", fontsize=12, pad=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_param_interaction.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_param_interaction.png", bbox_inches="tight",
                dpi=300, facecolor="white")
    print(f"Saved figure to {FIG_DIR / 'fig_param_interaction.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
