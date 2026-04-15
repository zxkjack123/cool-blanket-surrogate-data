"""Fig 1: CFETR COOL PbLi blanket radial geometry schematic.

Horizontal stacked-bar style showing Plasma | FW | PbLi | Shield | VV
with design parameter annotations and thickness ranges.
"""
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──
HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── design parameters (from Table 1) ──
# Use mid-range values for the schematic geometry
zones = [
    # (label, symbol, min_cm, max_cm, color, hatch)
    ("Plasma",  None,                   15, 15,  "#FFD6E0", None),   # fixed, not a parameter
    ("FW",      r"$d_\mathrm{FW}$",     1,   5,  "#4A90D9", None),
    ("PbLi",    r"$d_\mathrm{PbLi}$",  30,  80,  "#E8A838", ".."),
    ("Shield",  r"$d_\mathrm{Sh}$",    10,  50,  "#7CB342", None),
    ("VV",      r"$d_\mathrm{VV}$",    10,  30,  "#9E9E9E", "//"),
]

# Use representative mid-range thicknesses for visual layout
mid_widths = {
    "Plasma": 15,
    "FW":     3,
    "PbLi":  55,
    "Shield": 30,
    "VV":     20,
}

# ── figure ──
fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)

y_bot = 0.0
y_top = 1.0
x_cursor = 0.0

rects = []
for label, symbol, vmin, vmax, color, hatch in zones:
    w = mid_widths[label]
    rect = mpatches.FancyBboxPatch(
        (x_cursor, y_bot), w, y_top - y_bot,
        boxstyle="square,pad=0",
        facecolor=color, edgecolor="black", linewidth=1.2,
        hatch=hatch, alpha=0.85,
    )
    ax.add_patch(rect)

    # Zone label (centered)
    cx = x_cursor + w / 2
    cy = 0.55
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color="black")

    # For variable zones: add range annotation below
    if symbol is not None:
        range_str = f"{symbol}: {vmin}–{vmax} cm"
        ax.text(cx, 0.25, range_str, ha="center", va="center",
                fontsize=8.5, color="#333333", style="italic")
        # Double-headed arrow showing variable width
        arrow_y = -0.15
        ax.annotate("", xy=(x_cursor + w, arrow_y), xytext=(x_cursor, arrow_y),
                     arrowprops=dict(arrowstyle="<->", color=color,
                                     lw=2.0, shrinkA=0, shrinkB=0))

    rects.append((label, x_cursor, w))
    x_cursor += w

total_width = x_cursor

# ── Li-6 enrichment annotation inside PbLi zone ──
pbli_x = mid_widths["Plasma"] + mid_widths["FW"]
pbli_w = mid_widths["PbLi"]
ax.text(pbli_x + pbli_w / 2, 0.82,
        r"$f_\mathrm{Li6}$: 0.30 – 0.90",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#8B4513",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#8B4513", alpha=0.9))

# ── Total thickness constraint bracket ──
# Bracket spans FW + PbLi + Shield + VV
brace_x0 = mid_widths["Plasma"]
brace_x1 = total_width
brace_y = -0.35
ax.annotate("", xy=(brace_x1, brace_y), xytext=(brace_x0, brace_y),
            arrowprops=dict(arrowstyle="|-|", color="black", lw=1.5,
                            shrinkA=0, shrinkB=0))
ax.text((brace_x0 + brace_x1) / 2, brace_y - 0.12,
        r"$d_\mathrm{FW} + d_\mathrm{PbLi} + d_\mathrm{Sh} + d_\mathrm{VV} \leq 150\,\mathrm{cm}$",
        ha="center", va="center", fontsize=10, fontweight="bold")

# ── Radial direction arrow ──
ax.annotate("Radial direction →", xy=(total_width + 2, -0.5),
            fontsize=9, color="gray", va="center")

# ── Axis formatting ──
ax.set_xlim(-3, total_width + 20)
ax.set_ylim(-0.65, 1.15)
ax.axis("off")

# ── Title ──
ax.set_title("CFETR COOL PbLi Blanket — Radial Build (schematic, not to scale)",
             fontsize=11, pad=10)

fig.tight_layout()

# ── Save ──
fig.savefig(FIG_DIR / "fig01_blanket_geometry.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig01_blanket_geometry.png", bbox_inches="tight",
            dpi=300, facecolor="white")
print(f"Saved to {FIG_DIR / 'fig01_blanket_geometry.pdf'}")
print(f"Saved to {FIG_DIR / 'fig01_blanket_geometry.png'}")
plt.close(fig)
