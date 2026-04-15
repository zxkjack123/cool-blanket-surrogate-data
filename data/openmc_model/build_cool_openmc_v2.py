#!/usr/bin/env python3
"""CFETR COOL PbLi blanket — semi-engineering CSG model (v2, 709-group).

Upgrades over v1 (full-torus idealized model):
  - 22.5° toroidal sector with reflecting boundary conditions
  - Poloidal void gap (±30° from midplane = 60° divertor opening)
  - Homogenized FW structure: CLAM + 15 vol% H₂O (replaces pure W)
  - Thin W armor layer (2 mm, fixed)
  - Homogenized PbLi zone: PbLi + 15 vol% CLAM + 5 vol% SiC (cooling
    plates + functional channel insert)
  - Back plate: CLAM + 10 vol% H₂O (5 cm, fixed)
  - Homogenized shield: CLAM + 30 vol% H₂O
  - TBR tally via (n,Xt) to capture both ⁶Li(n,t) and ⁷Li(n,n't)

The five parametric design variables are unchanged:
  FW_THICK_CM, PBLI_THICK_CM, SHIELD_THICK_CM, VV_THICK_CM,
  LI6_ENRICH_ATOM_FRAC

Geometry notes
--------------
The sector is bounded by two planes through the Z-axis at ±11.25° from
the X-axis.  Both planes carry reflecting boundary conditions so that the
sector represents 1/16 of the full torus (matching the 16-module CFETR
layout).

A poloidal void gap removes the bottom ±30° cone (from the midplane
outward) to represent the divertor region, giving ~83% poloidal
coverage.  This is implemented via two cone surfaces that clip the
blanket layers at the lower portion of the torus cross-section.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import openmc

# ---------------------------------------------------------------------------
# Default geometry constants (cm)
# ---------------------------------------------------------------------------
R_MAJOR_DEFAULT = 720.0   # 7.2 m
A_MINOR_DEFAULT = 220.0   # 2.2 m
ELONG_DEFAULT = 2.0

# Layer thicknesses (cm) — parameterised
FW_THICK_DEFAULT = 2.0
PBLI_THICK_DEFAULT = 60.0
SHIELD_THICK_DEFAULT = 40.0
VV_THICK_DEFAULT = 5.0
LI6_ENRICH_ATOM_FRAC_DEFAULT = 0.90

# Fixed structural layers (not parameterised)
W_ARMOR_THICK = 0.2       # 2 mm tungsten plasma-facing armor
BACKPLATE_THICK = 5.0      # 5 cm CLAM + 10% H₂O

# Sector angle
SECTOR_HALF_ANGLE_DEG = 11.25   # 22.5° / 2

# Poloidal void gap: the blanket extends DIVERTOR_POLOIDAL_HALF_ANGLE_DEG
# below the equatorial midplane.  Below that, a Z-plane cut creates the
# divertor void.  z_cut = ELONG * A_MINOR * sin(angle).
# At 45°: z_cut = 0.707 * ELONG * A_MINOR → void area ≈ 9% of cross-section
# → ~91% poloidal blanket coverage (compensates for unmodeled ports/penetrations).
DIVERTOR_POLOIDAL_HALF_ANGLE_DEG = 45.0

COOL_CSG_PARAMS_ENV = "COOL_CSG_PARAMS"
OUTDIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Material densities for homogenized zones (g/cm³)
# ---------------------------------------------------------------------------
# CLAM steel: ρ = 7.87 g/cm³
# H₂O: ρ = 1.0 g/cm³
# SiC: ρ = 2.72 g/cm³ (CVD SiC FCI)
# PbLi: ρ = 9.4 g/cm³ (Li₁₇Pb₈₃, with impurities)
# SS316L: ρ = 7.9 g/cm³

RHO_CLAM = 7.87
RHO_WATER = 1.0
RHO_SIC = 2.72
RHO_PBLI_PURE = 9.4    # literature PbLi density (used for homogenization)
RHO_W = 19.25
RHO_SS316L = 7.9

# Volume fractions for homogenized zones
FW_STRUCT_CLAM_VFRAC = 0.85
FW_STRUCT_WATER_VFRAC = 0.15

PBLI_PBLI_VFRAC = 0.85
PBLI_CLAM_VFRAC = 0.10
PBLI_SIC_VFRAC = 0.05

BACKPLATE_CLAM_VFRAC = 0.90
BACKPLATE_WATER_VFRAC = 0.10

SHIELD_CLAM_VFRAC = 0.70
SHIELD_WATER_VFRAC = 0.30

# Homogenized densities
RHO_FW_STRUCT = FW_STRUCT_CLAM_VFRAC * RHO_CLAM + FW_STRUCT_WATER_VFRAC * RHO_WATER
RHO_PBLI_HOMOG = PBLI_PBLI_VFRAC * RHO_PBLI_PURE + PBLI_CLAM_VFRAC * RHO_CLAM + PBLI_SIC_VFRAC * RHO_SIC
RHO_BACKPLATE = BACKPLATE_CLAM_VFRAC * RHO_CLAM + BACKPLATE_WATER_VFRAC * RHO_WATER
RHO_SHIELD = SHIELD_CLAM_VFRAC * RHO_CLAM + SHIELD_WATER_VFRAC * RHO_WATER

# CLAM composition (weight fractions — from FISPACT clam.mat)
CLAM_COMPOSITION = {
    "Fe": 0.8841, "Cr": 0.0911, "W": 0.0152, "Ta": 0.0020,
    "V": 0.0019, "C": 0.0012, "Si": 0.0003, "Mn": 0.0041,
    "P": 0.00003, "S": 0.00003, "O": 0.00001, "N": 0.00002,
}

# SiC composition (weight fractions — from FISPACT sic.mat, simplified)
SIC_COMPOSITION = {"Si": 0.6909, "C": 0.2942, "O": 0.0134, "Fe": 0.00102}

# SS316L composition (weight fractions)
SS316L_COMPOSITION = {
    "Fe": 0.65, "Cr": 0.17, "Ni": 0.12, "Mo": 0.025,
    "Mn": 0.015, "Si": 0.01, "C": 0.001, "N": 0.001,
}


# ---------------------------------------------------------------------------
# Parameter loading (unchanged interface from v1)
# ---------------------------------------------------------------------------
def _load_params_from_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text) or {}
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML needed for YAML params; use JSON.") from exc
        return yaml.safe_load(text) or {}
    raise ValueError(f"Unsupported params file extension: {path.suffix}")


def _get_float(payload: dict, key: str, default: float) -> float:
    if key not in payload:
        return float(default)
    return float(payload[key])


def load_model_parameters() -> dict[str, float]:
    """Load model parameters from env var, falling back to defaults."""
    params_path = os.environ.get(COOL_CSG_PARAMS_ENV)
    payload: dict[str, Any] = {}
    if params_path:
        p = Path(params_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"{COOL_CSG_PARAMS_ENV} → missing: {p}")
        payload = _load_params_from_file(p)

    r_major = _get_float(payload, "R_MAJOR", R_MAJOR_DEFAULT)
    a_minor = _get_float(payload, "A_MINOR", A_MINOR_DEFAULT)
    elong = _get_float(payload, "ELONG", ELONG_DEFAULT)
    fw = _get_float(payload, "FW_THICK_CM", FW_THICK_DEFAULT)
    pbli = _get_float(payload, "PBLI_THICK_CM", PBLI_THICK_DEFAULT)
    shield = _get_float(payload, "SHIELD_THICK_CM", SHIELD_THICK_DEFAULT)
    vv = _get_float(payload, "VV_THICK_CM", VV_THICK_DEFAULT)
    li6 = _get_float(payload, "LI6_ENRICH_ATOM_FRAC", LI6_ENRICH_ATOM_FRAC_DEFAULT)

    for name, v in {"R_MAJOR": r_major, "A_MINOR": a_minor, "ELONG": elong,
                     "FW_THICK_CM": fw, "PBLI_THICK_CM": pbli,
                     "SHIELD_THICK_CM": shield, "VV_THICK_CM": vv}.items():
        if v <= 0.0:
            raise ValueError(f"{name} must be > 0, got {v}")
    if not (0.0 < li6 < 1.0):
        raise ValueError(f"LI6_ENRICH_ATOM_FRAC must be in (0,1), got {li6}")

    total_build = fw + pbli + shield + vv
    if total_build > 150.0:
        raise ValueError(f"Total radial build exceeds 150 cm: {total_build:.3f}")

    return {
        "R_MAJOR": r_major, "A_MINOR": a_minor, "ELONG": elong,
        "FW_THICK_CM": fw, "PBLI_THICK_CM": pbli,
        "SHIELD_THICK_CM": shield, "VV_THICK_CM": vv,
        "LI6_ENRICH_ATOM_FRAC": li6,
    }


# ---------------------------------------------------------------------------
# Module-level params (loaded once at import time, as in v1)
# ---------------------------------------------------------------------------
_PARAMS = load_model_parameters()
R_MAJOR = _PARAMS["R_MAJOR"]
A_MINOR = _PARAMS["A_MINOR"]
ELONG = _PARAMS["ELONG"]
FW_THICK = _PARAMS["FW_THICK_CM"]
PBLI_THICK = _PARAMS["PBLI_THICK_CM"]
SHIELD_THICK = _PARAMS["SHIELD_THICK_CM"]
VV_THICK = _PARAMS["VV_THICK_CM"]


# ---------------------------------------------------------------------------
# Geometry builder
# ---------------------------------------------------------------------------
def build_geometry() -> tuple[openmc.Geometry, dict[str, openmc.Cell]]:
    """Build a 22.5° toroidal sector CSG model with reflecting boundaries.

    Returns
    -------
    geometry : openmc.Geometry
    cells : dict mapping cell names to openmc.Cell objects
    """
    # --- Toroidal sector boundary planes ---
    # Two planes through the Z-axis at ±11.25° from the +X axis.
    # A plane through Z-axis at angle θ from +X has normal (sin θ, -cos θ, 0)
    # for the +θ side and (-sin θ, cos θ, 0) for the -θ side.
    theta = math.radians(SECTOR_HALF_ANGLE_DEG)
    sin_t, cos_t = math.sin(theta), math.cos(theta)

    # Plane 1: φ = +11.25°  →  normal pointing inward (sin θ, -cos θ, 0)
    plane_pos = openmc.Plane(a=sin_t, b=-cos_t, c=0, d=0,
                             boundary_type="reflective")
    # Plane 2: φ = -11.25°  →  normal pointing inward (-sin θ, -cos θ, 0)
    plane_neg = openmc.Plane(a=-sin_t, b=-cos_t, c=0, d=0,
                             boundary_type="reflective")

    # The sector region is: +plane_neg & -plane_pos (between the two planes,
    # on the +X side).  We need the region where both half-spaces overlap.
    # plane_pos: sin_t * x - cos_t * y = 0  →  region "-" means
    #   sin_t*x - cos_t*y < 0 for points at y>0 side.
    # For the 22.5° wedge centred on +X axis, the correct combination is:
    #   -plane_pos & +plane_neg
    sector_region = -plane_pos & +plane_neg

    # --- Poloidal void gap (divertor) ---
    # Remove the bottom portion of the torus cross-section.
    # Use a Z-plane to cut at z = -z_div, where z_div = minor_outer * elong
    # × sin(divertor_half_angle) measured from the plasma centre.
    # For simplicity, use a single Z-plane at z = -Z_DIV cutting across all
    # blanket layers.  Everything below this plane is void (divertor space).
    # The divertor opening corresponds to ~60° of poloidal angle on the
    # lower half → ~83% blanket poloidal coverage.
    z_div_frac = math.sin(math.radians(DIVERTOR_POLOIDAL_HALF_ANGLE_DEG))
    # Cut at the outermost blanket extent to ensure all layers are clipped
    outer_minor = A_MINOR + W_ARMOR_THICK + FW_THICK + PBLI_THICK + BACKPLATE_THICK + SHIELD_THICK + VV_THICK
    z_div = ELONG * A_MINOR * z_div_frac  # cut at fraction of plasma height
    div_plane = openmc.ZPlane(z0=-z_div)
    blanket_z_region = +div_plane  # blanket exists above this plane

    # --- Torus layer surfaces ---
    def torus(minor_r: float) -> openmc.ZTorus:
        return openmc.ZTorus(a=R_MAJOR, b=ELONG * minor_r, c=minor_r)

    s_plasma = torus(A_MINOR)
    r1 = A_MINOR + W_ARMOR_THICK
    s_armor = torus(r1)
    r2 = r1 + FW_THICK
    s_fw = torus(r2)
    r3 = r2 + PBLI_THICK
    s_pbli = torus(r3)
    r4 = r3 + BACKPLATE_THICK
    s_bp = torus(r4)
    r5 = r4 + SHIELD_THICK
    s_shield = torus(r5)
    r6 = r5 + VV_THICK
    s_vv = torus(r6)

    # Bounding box as graveyard
    r_bound = R_MAJOR + r6 + 200.0
    z_bound = ELONG * r6 + 200.0
    # Use a rectangular parallelepiped to bound the sector tightly
    boundary = openmc.Sphere(r=r_bound + z_bound, boundary_type="vacuum")

    # --- Cells ---
    # All blanket cells are clipped to the sector AND the poloidal blanket
    # region (above divertor cut).
    def blanket_cell(inner, outer, cell_id, name):
        region = +inner & -outer & sector_region & blanket_z_region
        return openmc.Cell(cell_id=cell_id, name=name, region=region)

    cell_plasma = openmc.Cell(cell_id=1, name="plasma",
                              region=-s_plasma & sector_region & blanket_z_region)
    cell_armor  = blanket_cell(s_plasma, s_armor, 2, "armor")
    cell_fw     = blanket_cell(s_armor, s_fw, 3, "fw")
    cell_pbli   = blanket_cell(s_fw, s_pbli, 4, "pbli")
    cell_bp     = blanket_cell(s_pbli, s_bp, 5, "backplate")
    cell_shield = blanket_cell(s_bp, s_shield, 6, "shield")
    cell_vv     = blanket_cell(s_shield, s_vv, 7, "vv")

    # Void region: everything inside the boundary sphere that is not a blanket
    # cell.  This includes the divertor void below the Z-cut and the space
    # outside the VV but inside the bounding sphere.
    # Build it as: inside boundary, outside VV OR below divertor cut, AND in
    # sector.
    # Actually, simplest: void = boundary & ~(union of all above cells).
    # OpenMC doesn't support that syntax directly, so we define void as the
    # complement.
    void_region = (+s_vv | ~blanket_z_region) & -boundary & sector_region
    cell_void = openmc.Cell(cell_id=8, name="void", region=void_region)

    # Also need plasma/blanket below divertor cut → void
    divertor_void_region = ~blanket_z_region & -s_vv & sector_region
    cell_div_void = openmc.Cell(cell_id=9, name="divertor_void",
                                region=divertor_void_region)

    root = openmc.Universe(cells=[
        cell_plasma, cell_armor, cell_fw, cell_pbli, cell_bp,
        cell_shield, cell_vv, cell_void, cell_div_void,
    ])
    return openmc.Geometry(root), {
        "plasma": cell_plasma,
        "armor": cell_armor,
        "fw": cell_fw,
        "pbli": cell_pbli,
        "backplate": cell_bp,
        "shield": cell_shield,
        "vv": cell_vv,
        "void": cell_void,
        "divertor_void": cell_div_void,
    }


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------
def _homogenized_material(
    name: str,
    density: float,
    components: list[tuple[dict[str, float], float, float]],
    *,
    li6_enrich: float | None = None,
) -> openmc.Material:
    """Create a homogenized material from volume-fraction-weighted components.

    Parameters
    ----------
    name : str
        Material name.
    density : float
        Homogenized density in g/cm³.
    components : list of (composition_dict, component_density, volume_fraction)
        Each composition_dict maps element names to mass fractions within that
        component.  component_density and volume_fraction are used to compute
        mass fractions in the homogenized mix.
    li6_enrich : float or None
        If not None, lithium in PbLi will be split into Li6/Li7 nuclides
        at this atom-fraction enrichment.  Only applied to "Li" entries.
    """
    mat = openmc.Material(name=name)
    mat.set_density("g/cm3", density)

    # Compute mass fraction of each element in the mix
    # mass_i = ρ_component × vfrac × wf_element
    # total mass per unit volume = density (pre-computed)
    elem_mass: dict[str, float] = {}
    has_li = False
    for comp, rho_comp, vfrac in components:
        for elem, wf in comp.items():
            mass_contrib = rho_comp * vfrac * wf
            if elem == "Li":
                has_li = True
            elem_mass[elem] = elem_mass.get(elem, 0.0) + mass_contrib

    total_mass = sum(elem_mass.values())

    for elem in sorted(elem_mass.keys()):
        wf = elem_mass[elem] / total_mass
        if wf < 1e-10:
            continue
        if elem == "Li" and li6_enrich is not None:
            # Split Li into Li-6 and Li-7 by atom fraction
            li6_aw, li7_aw = 6.015122, 7.016004
            li_avg_aw = li6_enrich * li6_aw + (1 - li6_enrich) * li7_aw
            li6_mf = (li6_enrich * li6_aw) / li_avg_aw
            mat.add_nuclide("Li6", wf * li6_mf, percent_type="wo")
            mat.add_nuclide("Li7", wf * (1 - li6_mf), percent_type="wo")
        else:
            mat.add_element(elem, wf, percent_type="wo")

    return mat


def build_materials() -> openmc.Materials:
    """Build all materials for the v2 model."""
    li6 = float(_PARAMS["LI6_ENRICH_ATOM_FRAC"])
    materials = openmc.Materials()

    # 1. Tungsten armor (pure W, thin fixed layer)
    armor = openmc.Material(name="armor")
    armor.set_density("g/cm3", RHO_W)
    armor.add_element("W", 1.0)
    materials.append(armor)

    # 2. FW structure: CLAM (85 vol%) + H₂O (15 vol%)
    fw = _homogenized_material(
        "fw", RHO_FW_STRUCT,
        [(CLAM_COMPOSITION, RHO_CLAM, FW_STRUCT_CLAM_VFRAC),
         ({"H": 0.1119, "O": 0.8881}, RHO_WATER, FW_STRUCT_WATER_VFRAC)],
    )
    materials.append(fw)

    # 3. PbLi breeder zone: PbLi (80 vol%) + CLAM (15 vol%) + SiC (5 vol%)
    # PbLi composition: Li 6.2 wt% (with enrichment), Pb ~93.8 wt%
    # Using the FISPACT pbli.mat simplified: Li 0.062, Pb 0.938
    pbli_comp = {"Li": 0.062, "Pb": 0.938}
    pbli = _homogenized_material(
        "pbli", RHO_PBLI_HOMOG,
        [(pbli_comp, RHO_PBLI_PURE, PBLI_PBLI_VFRAC),
         (CLAM_COMPOSITION, RHO_CLAM, PBLI_CLAM_VFRAC),
         (SIC_COMPOSITION, RHO_SIC, PBLI_SIC_VFRAC)],
        li6_enrich=li6,
    )
    materials.append(pbli)

    # 4. Back plate: CLAM (90 vol%) + H₂O (10 vol%)
    bp = _homogenized_material(
        "backplate", RHO_BACKPLATE,
        [(CLAM_COMPOSITION, RHO_CLAM, BACKPLATE_CLAM_VFRAC),
         ({"H": 0.1119, "O": 0.8881}, RHO_WATER, BACKPLATE_WATER_VFRAC)],
    )
    materials.append(bp)

    # 5. Shield: CLAM (70 vol%) + H₂O (30 vol%)
    shield = _homogenized_material(
        "shield", RHO_SHIELD,
        [(CLAM_COMPOSITION, RHO_CLAM, SHIELD_CLAM_VFRAC),
         ({"H": 0.1119, "O": 0.8881}, RHO_WATER, SHIELD_WATER_VFRAC)],
    )
    materials.append(shield)

    # 6. Vacuum vessel: SS316L
    vv = openmc.Material(name="vv")
    vv.set_density("g/cm3", RHO_SS316L)
    for elem, wf in SS316L_COMPOSITION.items():
        vv.add_element(elem, wf, percent_type="wo")
    materials.append(vv)

    return materials


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def build_settings(plasma_cell: openmc.Cell) -> openmc.Settings:
    settings = openmc.Settings()
    settings.batches = 50
    settings.particles = 1_000_000
    settings.run_mode = "fixed source"

    # Box source with rejection to plasma cell.
    # The box covers the full torus extent; rejection sampling selects only
    # points inside the plasma cell (22.5° sector, above divertor Z-cut).
    r_max = R_MAJOR + A_MINOR
    z_max = ELONG * A_MINOR
    src = openmc.IndependentSource()
    src.space = openmc.stats.Box(
        lower_left=(-r_max, -r_max, -z_max),
        upper_right=(r_max, r_max, z_max),
        only_fissionable=False,
    )
    src.constraints = {
        "domains": [plasma_cell],
        "rejection_strategy": "resample",
    }
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.06e6], [1.0])
    settings.source = src

    settings.max_lost_particles = 50_000
    # Box source over full torus has ~2.5% acceptance for 22.5° sector;
    # lower the threshold from the default 0.05 to avoid stochastic failures.
    settings.source_rejection_fraction = 0.005
    settings.output = {"tallies": True}
    return settings


# ---------------------------------------------------------------------------
# Tallies
# ---------------------------------------------------------------------------
def build_tallies(cells: dict[str, openmc.Cell]) -> openmc.Tallies:
    tallies = openmc.Tallies()

    # Material cells for tallies (armor, fw, pbli, backplate, shield, vv)
    tally_cell_names = ("armor", "fw", "pbli", "backplate", "shield", "vv")
    material_cells = [cells[k] for k in tally_cell_names]

    cell_filter = openmc.CellFilter(material_cells)

    # 709-group flux
    group_edges = openmc.mgxs.GROUP_STRUCTURES["CCFE-709"]
    efilter = openmc.EnergyFilter(group_edges)

    flux = openmc.Tally(name="flux_709g")
    flux.filters = [cell_filter, efilter]
    flux.scores = ["flux"]
    tallies.append(flux)

    # TBR tally — (n,Xt) captures all tritium-producing reactions including
    # both ⁶Li(n,t)α and ⁷Li(n,n't)α.
    tbr = openmc.Tally(name="tbr")
    tbr.filters = [cell_filter]
    tbr.scores = ["(n,Xt)"]
    tallies.append(tbr)

    return tallies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if hasattr(openmc, "reset_auto_ids"):
        openmc.reset_auto_ids()

    geom, cells = build_geometry()
    materials = build_materials()

    # Fill cells with materials
    name_to_mat = {m.name: m for m in materials}
    cells["armor"].fill = name_to_mat["armor"]
    cells["fw"].fill = name_to_mat["fw"]
    cells["pbli"].fill = name_to_mat["pbli"]
    cells["backplate"].fill = name_to_mat["backplate"]
    cells["shield"].fill = name_to_mat["shield"]
    cells["vv"].fill = name_to_mat["vv"]
    # plasma, void, divertor_void remain unfilled (vacuum)

    materials.export_to_xml(path=OUTDIR / "materials.xml")
    geom.export_to_xml(path=OUTDIR / "geometry.xml")

    settings = build_settings(cells["plasma"])
    settings.export_to_xml(path=OUTDIR / "settings.xml")

    tallies = build_tallies(cells)
    tallies.export_to_xml(path=OUTDIR / "tallies.xml")

    print(f"OpenMC v2 inputs written to {OUTDIR}")
    print(f"  Cells: {len([c for c in geom.root_universe.cells.values()])}")
    print(f"  Materials: {len(materials)}")
    print(f"  Sector: ±{SECTOR_HALF_ANGLE_DEG}° ({2*SECTOR_HALF_ANGLE_DEG}° sector)")
    print(f"  Divertor void: z < -{z_div_frac * ELONG * A_MINOR:.1f} cm")
    print(f"  Homog. densities (g/cm³):")
    print(f"    FW struct:  {RHO_FW_STRUCT:.3f}")
    print(f"    PbLi zone:  {RHO_PBLI_HOMOG:.3f}")
    print(f"    Back plate: {RHO_BACKPLATE:.3f}")
    print(f"    Shield:     {RHO_SHIELD:.3f}")
    if os.environ.get(COOL_CSG_PARAMS_ENV):
        print(f"  Params from ${COOL_CSG_PARAMS_ENV}:")
        for k in sorted(_PARAMS.keys()):
            print(f"    {k}: {_PARAMS[k]}")


# Module-level z_div_frac for main() print
z_div_frac = math.sin(math.radians(DIVERTOR_POLOIDAL_HALF_ANGLE_DEG))

if __name__ == "__main__":
    main()
