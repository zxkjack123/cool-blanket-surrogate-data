# Open Data: Surrogate-Based Design Space Exploration for a CFETR PbLi Blanket

This repository contains the training data, analysis scripts, and intermediate artifacts for the paper:

> **Surrogate-Based Design Space Exploration with Conformal Uncertainty Quantification for a CFETR PbLi Blanket**
>
> Xiaokang Zhang, Xilong Tong, Long Gui, Yanshi Wei, Wei Sun, Shanliang Zheng\*
>
> Submitted to *Fusion Engineering and Design* (2026)

## Repository structure

```
├── data/
│   ├── wp2_lhs_cool_csg_v2_n144_merged.csv          # 144 LHS samples: 5 design params + per-cell TBR
│   ├── wp2_lhs_cool_csg_v2_n144_seed20260414.csv     # LHS parameter table (v2, 144 samples)
│   ├── wp2_lhs_cool_csg_expandC_n144_seed20260330.csv # LHS parameter table (expandC, 144 samples)
│   ├── labels_long_expandC_act9.csv                   # Time-series labels: 74,880 rows (144 samples × 4 regions × ~130 time steps)
│   ├── splits.json                                    # Train/test split definition
│   ├── fispact_mats/                                  # FISPACT-II material templates
│   │   ├── w.mat                                      # Tungsten armor (cell_2)
│   │   ├── pbli.mat                                   # PbLi breeder (cell_3)
│   │   ├── clam.mat                                   # CLAM steel first wall struct (cell_4)
│   │   └── ss304l.mat                                 # SS304L back plate (cell_5)
│   └── openmc_model/
│       └── build_cool_openmc_v2.py                    # OpenMC model builder (22.5° ZTorus sector)
├── scripts/                                           # Figure and analysis scripts
│   ├── train_tbr_with_uq.py                           # HGBR training + conformal UQ
│   ├── fig01_blanket_geometry.py                      # Fig 1: radial build
│   ├── fig01_workflow.py                              # Fig 2: workflow diagram
│   ├── fig02_tbr_landscape.py                         # Fig 4: TBR constraint landscape
│   ├── fig03_h3_sensitivity.py                        # Fig 6: H-3 sensitivity
│   ├── fig04_timeseries_uq.py                         # Fig 8: time-series UQ bands
│   ├── fig05_parity.py                                # Fig 3: parity plots
│   ├── fig_param_interaction.py                       # Fig 7: parameter interaction
│   ├── fig_threshold_sweep.py                         # Fig 5: threshold sweep
│   ├── experiment_multi_split.py                      # 100-split cross-validation
│   ├── experiment_phat_sensitivity.py                 # ĥ_P sensitivity analysis
│   ├── gold_comparison.py                             # Gold-standard validation
│   └── ...                                            # Additional analysis scripts
├── artifacts/                                         # Intermediate analysis outputs (CSV + JSON)
│   ├── tbr_conformal_results.csv                      # TBR surrogate predictions + conformal bands
│   ├── constraint_landscape.csv                       # Constraint satisfaction probabilities
│   ├── multi_split_results.csv                        # 100-split R²/MAE/PICP
│   ├── phat_sensitivity.csv                           # ĥ_P threshold sweep (300 rows)
│   ├── h3_sensitivity.csv                             # H-3 permutation importances
│   └── ...                                            # Additional metrics and results
├── LICENSE
├── .gitignore
└── README.md
```

## Data description

### Design parameter table (`data/wp2_lhs_cool_csg_v2_n144_merged.csv`)

144 blanket configurations sampled via Latin Hypercube Sampling (LHS) with rejection constraint (total radial envelope ≤ 150 cm). Columns include:

| Column | Description | Unit |
|--------|-------------|------|
| `sample_id` | Unique sample identifier | — |
| `FW_THICK_CM` | First-wall thickness | cm |
| `LI6_ENRICH_ATOM_FRAC` | Li-6 enrichment (atom fraction) | — |
| `PBLI_THICK_CM` | PbLi breeding zone thickness | cm |
| `SHIELD_THICK_CM` | Neutron shield thickness | cm |
| `VV_THICK_CM` | Vacuum vessel thickness | cm |
| `tbr_total` | Total tritium breeding ratio | — |
| `tbr_cell_*` | Per-cell TBR contributions | — |

### Time-series labels (`data/labels_long_expandC_act9.csv`)

74,880 rows covering 144 samples × 4 material regions × ~130 time steps (irradiation + cooling). Key targets:

- **Tritium (H-3)**: Atoms per region at each time step
- **Li-6, Li-7**: Isotopic inventory evolution (burnup tracers)
- **Activation products**: ⁵⁴Mn, ⁶⁰Co, ⁵⁵Fe, ²¹⁰Po, ²¹⁰Bi, ²⁰³Hg, ²⁰⁴Tl, ²⁰⁷Bi, ²⁰³Pb
- **Integral quantities**: Activity (Bq), contact dose (Sv/h), decay heat (W)

### OpenMC model (`data/openmc_model/build_cool_openmc_v2.py`)

Parametric OpenMC model builder for a 22.5° toroidal sector of the CFETR COOL PbLi blanket. Takes the five design parameters as input and generates a CSG geometry with reflecting boundary conditions and (n,Xt) TBR tallies.

- Source: 14.06 MeV D-T isotropic ring source
- Transport: 50 batches × 1M particles = 50M histories
- Cross sections: FENDL-3.1d
- Tally: 709-group flux per cell + (n,Xt) reaction rate

### FISPACT-II materials (`data/fispact_mats/`)

Material composition templates for the four activation regions with full impurity specifications.

## Reproducing the analysis

```bash
pip install numpy pandas scikit-learn matplotlib scipy

# Train surrogate and generate predictions
python scripts/train_tbr_with_uq.py

# Generate individual figures
python scripts/fig05_parity.py           # Fig 3
python scripts/fig02_tbr_landscape.py    # Fig 4
python scripts/fig_threshold_sweep.py    # Fig 5
python scripts/fig03_h3_sensitivity.py   # Fig 6
python scripts/fig_param_interaction.py  # Fig 7
python scripts/fig04_timeseries_uq.py   # Fig 8
```

## Related repositories

- [zxkjack123/sc-piml-plus-data](https://github.com/zxkjack123/sc-piml-plus-data) — Data and scripts for the companion MLST paper on time-aware conformal UQ
- [zxkjack123/coolburnup-bench-fed](https://github.com/zxkjack123/coolburnup-bench-fed) — COOLBurnup-Bench: OpenMC → FISPACT coupled workflow and benchmark pipeline

## Citation

If you use this data or code, please cite:

```bibtex
@article{zhang2026surrogate,
  title   = {Surrogate-Based Design Space Exploration with Conformal Uncertainty
             Quantification for a {CFETR} {PbLi} Blanket},
  author  = {Zhang, Xiaokang and Tong, Xilong and Gui, Long and Wei, Yanshi
             and Sun, Wei and Zheng, Shanliang},
  journal = {Fusion Engineering and Design},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Code: MIT License. Data: CC-BY-4.0.
