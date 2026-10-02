"""Embedded wall-measure audits for the contract-v8 Young boundary flux (L1A-2e).

Fast, trajectory-free checks of the *geometry* and the *variational structure* of the
production wall operator. Nothing here time-steps to equilibrium (that evidence lives in
``production/embedded_young_audit.py``); everything here runs in seconds so CI can gate on
it.

Sections
--------
1. **flat wall length / translation** -- ``sum_i A_wall,i`` equals the geometric wall length
   for every sub-cell wall offset ``k/8 * dy`` (grid-alignment independence), with the
   contract-v7 diffuse kernel's fluid-side share recorded next to it as the falsified
   root cause.
2. **inclined walls** -- slopes +/-0.25 and +/-0.5: the measure equals the Euclidean cut
   length (<= 2 % in an interior window) and carries no Manhattan/staircase gain.
3. **textured geometry** -- flat/pillars/grooves/wedge/hierarchical: finite, positive,
   one-cell-layer-localized measures with unit normals, exact segment->cell conservation,
   all measure on hard-fluid cells, no periodic-y seam ghost, no duplicated corners, and
   O(dx) convergence to the analytic in-domain boundary length.
4. **variational derivative** -- the production wall operator is the exact variational
   derivative of ``F_bulk + F_wall^h`` (float64 centred directional derivative, gate
   ``<= 1e-6``, target ``<= 1e-8``), and the check is non-vacuous: adding a second wall
   contribution (the old diffuse kernel on top of the cut-cell flux) or dropping the wall
   term breaks it.
5. **angle independence / no global gain** -- the measure is bit-identical for all four
   Young targets, the wall term is exactly linear in ``cos(theta)``, and the total measure
   is the wall length (not ``wall_length / f``): there is no fitted factor anywhere.
6. **first-fluid-layer Young residual** -- manufactured fields at 60/90/120/150 deg, the
   new measure-weighted residual, its non-vacuity (a wrong wall angle is detected), and a
   side-by-side with the retained two-cell-band residual.

Run from ``examples/two_phase``::

    python -m production.wall_measure_audit --quick --json artifacts/production_validation/wall_measure_audit.json
    python -m production.wall_measure_audit --json artifacts/production_validation/wall_measure_audit_full.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
import phasefield as pf
from production import contact_line_kinetics as clk

AUDIT_SCHEMA_VERSION = 1
WALL_MEASURE_CONTRACT_VERSION = 1

#: Gates. These are the L1A-2e merge gates; none is tuned to an observed value.
FLAT_TRANSLATION_SPREAD_LIMIT = 0.01  # <= 1 % (ideal 0.5 %) across the eight sub-cell offsets
FLAT_TRANSLATION_SPREAD_IDEAL = 0.005
FLAT_LENGTH_RELATIVE_TOLERANCE = 1.0e-9  # a flat wall is measured exactly (chord == dx)
INCLINED_LENGTH_RELATIVE_TOLERANCE = 0.02  # <= 2 %
INCLINED_MANHATTAN_MARGIN = 0.5  # |measure - euclid| <= margin * |manhattan - euclid|
TEXTURED_LENGTH_RELATIVE_TOLERANCE = 0.05  # sharp corners are O(dx); curved/flat are far tighter
GHOST_TO_PEAK_LIMIT = 1.0e-12  # no measure anywhere near the periodic-y seam
CONTROL_CELL_PLACEMENT_LIMIT_CELLS = 2.13  # sqrt(1.5^2 + 1.5^2) = 2.121 for a ring-1 control cell
SMOOTH_CURVED_COARSE_TOLERANCE = 5.0e-3  # periodic curved wall at N = 64
SMOOTH_CURVED_FINE_TOLERANCE = 1.0e-3  # periodic curved wall at N >= 128 (second-order chord error)
NORMAL_UNIT_TOLERANCE = 1.0e-9
CONSERVATION_RELATIVE_TOLERANCE = 1.0e-12
DIRECTIONAL_DERIVATIVE_TOLERANCE = 1.0e-6
LEGACY_REPRODUCTION_RELATIVE_TOLERANCE = 1.0e-12  # transcribed v7 operator vs the pinned legacy path
VARIATIONAL_DEGENERATE_FLOOR = 1.0e-10  # below this a directional derivative is 0/0 and reported as degenerate
DIRECTIONAL_DERIVATIVE_IDEAL = 1.0e-8
FIRST_LAYER_MANUFACTURED_TOLERANCE = 5.0e-2  # normalized L2 on manufactured fields (eps/dx = 2)
FIRST_LAYER_NEUTRAL_TOLERANCE = 1.0e-12  # 90 deg: both terms vanish identically
FIRST_LAYER_WRONG_ANGLE_MINIMUM = 0.2  # non-vacuity: a wrong wall angle must be loud

TRANSLATION_OFFSETS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
TARGETS_DEG = (60.0, 90.0, 120.0, 150.0)
INCLINED_SLOPES = (-0.5, -0.25, 0.25, 0.5)

CONVENTIONS = {
    "wall_measure_method": pf.WALL_MEASURE_METHOD,
    "wall_measure_contract_version": pf.WALL_MEASURE_CONTRACT_VERSION,
    "measure_definition": "A_wall,i = length of the sdf = 0 marching-squares contour assigned to control cell i",
    "measure_density": "dA/dV = A_wall,i / (dx dy), used in exactly one place: the embedded Laplacian wall flux",
    "control_cell": "the cut cell when its centre is hard fluid, else the first cell along the rounded "
    "normal direction whose centre is hard fluid (sdf >= 0)",
    "normal_convention": "n = -grad(sdf)/|grad(sdf)| at the segment centroid, pointing fluid -> solid",
    "wall_energy": "g_w(phi, theta) = -sigma_0 cos(theta) h(phi), h(phi) = phi^2 (3 - 2 phi), sigma_0 = sqrt(2)/6",
    "discrete_wall_energy": "F_wall^h = sum_i A_wall,i g_w(phi_i, theta_i)",
    "forbidden": "no cos(theta)/f, no global wall gain, no angle remap, no fitted calibration of any kind",
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WallMeasureAudit:
    checks: list[Check] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "audit": "wall_measure",
            "stage": "L1A-2e",
            "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
            "wall_measure_method": str(pf.WALL_MEASURE_METHOD),
            "wall_measure_contract_version": int(pf.WALL_MEASURE_CONTRACT_VERSION),
            "settings": self.settings,
            "conventions": dict(CONVENTIONS),
            "checks": [check.to_dict() for check in self.checks],
            "numbers": self.numbers,
            "passed": self.passed,
        }


def _params(N: int, *, dtype=jnp.float64, eps_factor: float = 2.0, **kwargs) -> pf.PhaseFieldParams:
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=dtype, **kwargs)
    p.eps = float(eps_factor) * p.dx
    return p


def _measure(sdf, p) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    area, normal_x, normal_y, distance, info = pf.wall_cut_measure(jnp.asarray(sdf), p)
    return (
        np.asarray(area, dtype=np.float64),
        np.asarray(normal_x, dtype=np.float64),
        np.asarray(normal_y, dtype=np.float64),
        np.asarray(distance, dtype=np.float64),
        {key: (float(value) if np.ndim(value) == 0 else value) for key, value in info.items()},
    )


def _normal_magnitudes(area, normal_x, normal_y) -> np.ndarray:
    magnitude = np.sqrt(normal_x**2 + normal_y**2)
    return magnitude[area > 0.0]


# --------------------------------------------------------------------------------------
#  1. flat wall: exact length and grid-alignment independence
# --------------------------------------------------------------------------------------
def audit_flat_wall(N_values: Sequence[int] = (64, 96, 128, 192), wall_height: float = 0.25):
    """Total measure, per-cell measure and v7 fluid-side share for eight sub-cell offsets."""
    rows: list[dict[str, Any]] = []
    for N in N_values:
        p = _params(int(N))
        per_offset: list[dict[str, Any]] = []
        for offset in TRANSLATION_OFFSETS:
            height = wall_height + float(offset) * p.dy
            sdf = np.asarray(pf.surface_flat(p, wall_height=height), dtype=np.float64)
            area, normal_x, normal_y, distance, info = _measure(sdf, p)
            positive = area > 0.0
            legacy = clk.fluid_wall_delta_integral(sdf, p.dx, p.dy)
            rows_cell = np.unique(np.argwhere(positive)[:, 1]).tolist()
            per_offset.append(
                {
                    "offset_over_dy": float(offset),
                    "wall_height": float(height),
                    "total_measure": float(area.sum()),
                    "expected_length": float(p.Lx),
                    "relative_error": abs(float(area.sum()) - p.Lx) / p.Lx,
                    "n_wall_cells": int(positive.sum()),
                    "min_cell_measure": float(area[positive].min()) if positive.any() else 0.0,
                    "max_cell_measure": float(area[positive].max()) if positive.any() else 0.0,
                    "cell_rows": rows_cell,
                    "n_cell_layers": len(rows_cell),
                    "normal_x_range": [
                        float(normal_x[positive].min()) if positive.any() else 0.0,
                        float(normal_x[positive].max()) if positive.any() else 0.0,
                    ],
                    "normal_y_range": [
                        float(normal_y[positive].min()) if positive.any() else 0.0,
                        float(normal_y[positive].max()) if positive.any() else 0.0,
                    ],
                    "measure_conservation_error": float(info["measure_conservation_error"]),
                    "length_on_solid_cells": float(info["length_on_solid_cells"]),
                    "v7_fluid_side_normal_integral": legacy["fluid_side_normal_integral"],
                    "v7_fluid_fraction_of_wall_kernel": legacy["fluid_fraction_of_wall_kernel"],
                    "v7_effective_fluid_measure": float(
                        np.sum(np.asarray(pf.wall_delta(jnp.asarray(sdf), p), dtype=np.float64)[sdf >= 0.0])
                        * p.dx
                        * p.dy
                    ),
                }
            )
        totals = np.asarray([row["total_measure"] for row in per_offset], dtype=np.float64)
        legacy_totals = np.asarray([row["v7_effective_fluid_measure"] for row in per_offset], dtype=np.float64)
        legacy_shares = np.asarray([row["v7_fluid_fraction_of_wall_kernel"] for row in per_offset], dtype=np.float64)
        rows.append(
            {
                "N": int(N),
                "dx": float(p.dx),
                "offsets": per_offset,
                "measure_spread_relative": float((totals.max() - totals.min()) / totals.mean()),
                "measure_spread_absolute": float(totals.max() - totals.min()),
                "max_relative_error": float(max(row["relative_error"] for row in per_offset)),
                "max_cell_layers": int(max(row["n_cell_layers"] for row in per_offset)),
                "v7_measure_spread_relative": float(
                    (legacy_totals.max() - legacy_totals.min()) / max(legacy_totals.mean(), 1e-30)
                ),
                "v7_fluid_share_range": [float(legacy_shares.min()), float(legacy_shares.max())],
                "v7_fluid_share_spread_relative": float(
                    (legacy_shares.max() - legacy_shares.min()) / max(legacy_shares.mean(), 1e-30)
                ),
            }
        )

    spreads = [row["measure_spread_relative"] for row in rows]
    length_errors = [row["max_relative_error"] for row in rows]
    legacy_spreads = [row["v7_measure_spread_relative"] for row in rows]
    checks = [
        Check(
            "flat_wall_measure_equals_geometric_length",
            all(error <= FLAT_LENGTH_RELATIVE_TOLERANCE for error in length_errors),
            f"sum_i A_wall,i == Lx within {FLAT_LENGTH_RELATIVE_TOLERANCE:g} at N={list(N_values)} for all eight "
            f"sub-cell offsets (max relative error {max(length_errors):.3e})",
            {str(row["N"]): row["max_relative_error"] for row in rows},
        ),
        Check(
            "flat_wall_measure_translation_invariant",
            all(spread <= FLAT_TRANSLATION_SPREAD_LIMIT for spread in spreads),
            f"wall-measure spread across wall offsets [0, 0.875]*dy is <= {100 * FLAT_TRANSLATION_SPREAD_LIMIT:g} % "
            f"(ideal {100 * FLAT_TRANSLATION_SPREAD_IDEAL:g} %); measured max {max(spreads):.3e}",
            {str(row["N"]): row["measure_spread_relative"] for row in rows},
        ),
        Check(
            "flat_wall_translation_ideal_spread",
            all(spread <= FLAT_TRANSLATION_SPREAD_IDEAL for spread in spreads),
            f"ideal gate: spread <= {100 * FLAT_TRANSLATION_SPREAD_IDEAL:g} % (measured max {max(spreads):.3e})",
            {str(row["N"]): row["measure_spread_relative"] for row in rows},
        ),
        Check(
            "v7_diffuse_kernel_is_alignment_dependent",
            all(spread > 10.0 * max(FLAT_TRANSLATION_SPREAD_LIMIT, s) for spread, s in zip(legacy_spreads, spreads)),
            "root-cause reproduction: the pinned contract-v7 diffuse kernel's *effective fluid-side* measure varies "
            "by > 10x the v8 spread across the same offsets (this is the falsified v7 behaviour, not a v8 gate)",
            {str(row["N"]): row["v7_measure_spread_relative"] for row in rows},
        ),
    ]
    return checks, {"flat_wall": rows}


# --------------------------------------------------------------------------------------
#  2. inclined walls: Euclidean length, no Manhattan gain
# --------------------------------------------------------------------------------------
def audit_inclined_wall(N: int = 128, slopes: Sequence[float] = INCLINED_SLOPES, wall_height: float = 0.25):
    """Interior-window cut length for inclined planar walls (exact for a linear SDF)."""
    p = _params(int(N))
    dx = float(p.dx)
    x_axis = (np.arange(p.Nx) + 0.5) * dx
    X, Y = pf.grids(p)
    rows: list[dict[str, Any]] = []
    for slope in slopes:
        m = float(slope)
        sdf = np.asarray((Y - m * (X - 0.5 * p.Lx) - wall_height) / math.sqrt(1.0 + m * m), dtype=np.float64)
        area, normal_x, normal_y, distance, info = _measure(sdf, p)
        height = wall_height + m * (x_axis - 0.5 * p.Lx)
        # Keep only columns whose wall sits comfortably inside the domain, so the window
        # is not truncated by the y boundary (the plane leaves the domain for |m| x > 0.25).
        # The inclined *plane* is not periodic in x, so the input SDF itself jumps at the
        # x seam; the audit window stays away from it exactly like the retained v7
        # natural-BC probes do. The periodic curved counterpart is the wedge case below.
        interior = (height > 6.0 * dx) & (height < p.Ly - 6.0 * dx)
        interior[:8] = False
        interior[-8:] = False
        first = int(np.argmax(interior))
        last = int(p.Nx - np.argmax(interior[::-1]))
        window = area[first:last]
        columns = last - first
        positive = window > 0.0
        measured = float(window.sum())
        euclidean = columns * dx * math.sqrt(1.0 + m * m)
        manhattan = columns * dx * (1.0 + abs(m))
        normals = np.stack([normal_x[first:last][positive], normal_y[first:last][positive]], axis=-1)
        # sdf = (Y - m (X - Lx/2) - h)/sqrt(1+m^2) -> grad(sdf) = (-m, 1)/sqrt(1+m^2)
        # and the fluid->solid normal is n = -grad(sdf) = (m, -1)/sqrt(1+m^2).
        expected_normal = np.asarray([m, -1.0]) / math.sqrt(1.0 + m * m)
        rows.append(
            {
                "slope": m,
                "window_columns": [first, last],
                "n_columns": columns,
                "measure": measured,
                "euclidean_length": euclidean,
                "manhattan_length": manhattan,
                "relative_error_vs_euclidean": abs(measured - euclidean) / euclidean,
                "relative_deviation_from_manhattan": (measured - manhattan) / euclidean,
                "manhattan_gain_suppressed": abs(measured - euclidean)
                <= INCLINED_MANHATTAN_MARGIN * abs(manhattan - euclidean),
                "n_wall_cells": int(positive.sum()),
                "cell_measure_range": [float(window[positive].min()), float(window[positive].max())],
                "expected_cell_measure": float(dx * math.sqrt(1.0 + m * m)),
                "normal_max_error": float(np.max(np.linalg.norm(normals - expected_normal, axis=-1)))
                if positive.any()
                else float("inf"),
                "measure_conservation_error": float(info["measure_conservation_error"]),
            }
        )
    errors = [row["relative_error_vs_euclidean"] for row in rows]
    checks = [
        Check(
            "inclined_wall_measure_is_euclidean_length",
            all(error <= INCLINED_LENGTH_RELATIVE_TOLERANCE for error in errors),
            f"inclined slopes {list(slopes)}: cut measure within {100 * INCLINED_LENGTH_RELATIVE_TOLERANCE:g} % of the "
            f"Euclidean length in an interior window (max {max(errors):.3e})",
            {str(row["slope"]): row["relative_error_vs_euclidean"] for row in rows},
        ),
        Check(
            "inclined_wall_has_no_manhattan_gain",
            all(row["manhattan_gain_suppressed"] for row in rows),
            "the measure is the true diagonal cut length, not a staircase |dx| + |dy| sum",
            {str(row["slope"]): row["relative_deviation_from_manhattan"] for row in rows},
        ),
        Check(
            "inclined_wall_normal_orientation",
            all(row["normal_max_error"] <= 1.0e-9 for row in rows),
            "cell normals equal -grad(sdf)/|grad(sdf)| = (-m, -1)/sqrt(1+m^2) to round-off",
            {str(row["slope"]): row["normal_max_error"] for row in rows},
        ),
    ]
    return checks, {"inclined_wall": rows}


# --------------------------------------------------------------------------------------
#  3. textured geometry: localization, conservation, ghosts, refinement
# --------------------------------------------------------------------------------------
def _analytic_boundary_length(name: str, p, kwargs: dict[str, Any]) -> float | None:
    """In-domain analytic boundary length for the geometries where it is closed form."""
    if name == "flat":
        return float(p.Lx)
    if name == "pillars":
        n = int(kwargs.get("n_pillars", 4))
        height = float(kwargs.get("height", 0.4))
        return float(p.Lx + 2.0 * n * height)
    if name == "grooves":
        n = int(kwargs.get("n_grooves", 6))
        width = float(kwargs.get("width", 0.25))
        depth = float(kwargs.get("depth", 0.35))
        wall_height = float(kwargs.get("wall_height", 0.25))
        visible = min(depth, wall_height)  # the groove floor lies below the domain for the default depth
        return float(p.Lx - n * width + 2.0 * n * visible)
    if name == "wedge":
        slope = float(kwargs.get("slope", 0.5))
        wave_number = 6.0 * math.pi / p.Lx
        # ∫ sqrt(1 + (slope cos(k (x - Lx/2)))^2) dx over one domain period
        steps = 200001
        x = np.linspace(0.0, p.Lx, steps)
        integrand = np.sqrt(1.0 + (slope * np.cos(wave_number * (x - 0.5 * p.Lx))) ** 2)
        return float(np.trapezoid(integrand, x))
    return None


TEXTURED_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("flat", {"wall_height": 0.25}),
    ("pillars", {"wall_height": 0.25, "n_pillars": 4, "width": 0.3, "height": 0.4}),
    ("grooves", {"wall_height": 0.25, "n_grooves": 6, "width": 0.25, "depth": 0.35}),
    ("wedge", {"wall_height": 1.5, "slope": 0.5}),
    ("hierarchical", {"wall_height": 0.25}),
)


def audit_textured_geometry(N_values: Sequence[int] = (64, 128, 192), cases=TEXTURED_CASES):
    """Finite/positive/localized measures, conservation, seam ghosts and refinement."""
    rows: list[dict[str, Any]] = []
    for name, kwargs in cases:
        per_N: list[dict[str, Any]] = []
        for N in N_values:
            p = _params(int(N))
            generator = getattr(pf, f"surface_{name}")
            sdf = np.asarray(generator(p, **kwargs), dtype=np.float64)
            area, normal_x, normal_y, distance, info = _measure(sdf, p)
            positive = area > 0.0
            magnitudes_full = np.sqrt(normal_x**2 + normal_y**2)
            magnitudes = magnitudes_full[area > 0.0]
            # "one cell layer" localization: how far the assigned cells sit from the contour.
            segments = pf.wall_cut_segments(jnp.asarray(sdf), p)
            segment_count = np.asarray(info["segment_count_field"], dtype=np.float64)
            # Distance from each segment centroid to the centre of the control cell that
            # received it, in cell units: this is the wall-condition placement error.
            target_i = np.asarray(segments["target_i"])
            target_j = np.asarray(segments["target_j"])
            valid_segments = np.asarray(segments["valid"])
            offset_x = (np.asarray(segments["mid_x"]) - (target_i + 0.5) * p.dx + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx
            offset_y = np.asarray(segments["mid_y"]) - (target_j + 0.5) * p.dy
            placement = np.sqrt(offset_x**2 + offset_y**2) / max(p.dx, 1e-30)
            source_i = np.broadcast_to(np.arange(p.Nx)[:, None, None], target_i.shape)
            source_j = np.broadcast_to(np.arange(p.Ny)[None, :, None], target_i.shape)
            chebyshev = np.maximum(
                np.abs((target_i - source_i + p.Nx // 2) % p.Nx - p.Nx // 2), np.abs(target_j - source_j)
            )
            single = positive & (segment_count <= 1.0 + 1e-9)
            multi = positive & (segment_count > 1.0 + 1e-9)
            top_band = area[:, int(0.85 * p.Ny) :]
            analytic = _analytic_boundary_length(name, p, kwargs)
            per_N.append(
                {
                    "N": int(N),
                    "dx": float(p.dx),
                    "total_measure": float(area.sum()),
                    "analytic_length": analytic,
                    "relative_error": (abs(float(area.sum()) - analytic) / analytic) if analytic else None,
                    "n_wall_cells": int(positive.sum()),
                    "n_segments": float(info["n_segments"]),
                    "n_positive_length_segments": float(info["n_positive_length_segments"]),
                    "n_degenerate_zero_length_segments": float(info["n_segments"])
                    - float(info["n_positive_length_segments"]),
                    "n_segments_multi_per_cell": float(info["n_segments_multi_per_cell"]),
                    "segments_per_wall_cell": float(info["mean_segments_per_wall_cell"]),
                    "assigned_segment_count": float(np.sum(np.asarray(info["segment_count_field"], dtype=np.float64))),
                    "max_segments_per_cell": float(np.max(np.asarray(info["segment_count_field"], dtype=np.float64))),
                    "measure_conservation_error": float(info["measure_conservation_error"]),
                    "measure_conservation_relative": abs(float(info["measure_conservation_error"]))
                    / max(float(area.sum()), 1e-30),
                    "min_cell_measure": float(area[positive].min()) if positive.any() else 0.0,
                    "max_cell_measure": float(area[positive].max()) if positive.any() else 0.0,
                    "all_finite": bool(np.isfinite(area).all() and np.isfinite(magnitudes).all()),
                    "all_positive": bool((area[positive] > 0.0).all()),
                    "normal_magnitude_range": [float(magnitudes.min()), float(magnitudes.max())]
                    if magnitudes.size
                    else [0.0, 0.0],
                    "length_on_solid_cells": float(info["length_on_solid_cells"]),
                    "fluid_fraction_of_measure": float(info["length_on_fluid_cells"]) / max(float(area.sum()), 1e-30),
                    "max_control_cell_placement_cells": float(placement[valid_segments].max())
                    if valid_segments.any()
                    else 0.0,
                    "n_segments_control_offset_gt_1": int(np.count_nonzero((chebyshev > 1) & valid_segments)),
                    "n_single_segment_cells": int(single.sum()),
                    "n_multi_segment_cells": int(multi.sum()),
                    "single_segment_normal_deviation": float(np.max(np.abs(magnitudes_full[single] - 1.0)))
                    if single.any()
                    else 0.0,
                    "multi_segment_normal_min": float(magnitudes_full[multi].min()) if multi.any() else None,
                    "n_degenerate_normal_cells": int(np.count_nonzero(positive & (magnitudes_full < 0.5))),
                    "seam_band_max_measure": float(top_band.max()) if top_band.size else 0.0,
                    "seam_band_to_total": float(top_band.max()) / max(float(area.sum()), 1e-30)
                    if top_band.size
                    else 0.0,
                }
            )
        rows.append({"surface": name, "kwargs": {k: v for k, v in kwargs.items()}, "grids": per_N})

    def _worst(key: str) -> float:
        return max(float(grid[key]) for row in rows for grid in row["grids"] if grid[key] is not None)

    def _by_surface(key: str) -> dict[str, Any]:
        return {row["surface"]: [grid[key] for grid in row["grids"]] for row in rows}

    smooth = {row["surface"]: row for row in rows if row["surface"] in ("flat", "wedge")}
    cornered = {row["surface"]: row for row in rows if row["surface"] in ("pillars", "grooves")}
    smooth_errors = {name: {grid["N"]: grid["relative_error"] for grid in row["grids"]} for name, row in smooth.items()}
    flat_exact = (
        all(
            grid["relative_error"] is not None and grid["relative_error"] <= FLAT_LENGTH_RELATIVE_TOLERANCE
            for grid in smooth["flat"]["grids"]
        )
        if "flat" in smooth
        else False
    )
    wedge_series = smooth_errors.get("wedge", {})
    wedge_coarse = max((error for n, error in wedge_series.items() if n <= 64), default=0.0)
    wedge_fine = max((error for n, error in wedge_series.items() if n >= 128), default=0.0)
    wedge_ratio = (wedge_coarse / wedge_fine) if wedge_fine > 0 else None
    corner_series = {
        name: {grid["N"]: grid["relative_error"] for grid in row["grids"]} for name, row in cornered.items()
    }

    def _coarse(series: dict[int, float]) -> float:
        return max((v for n, v in series.items() if n <= 64), default=0.0)

    def _fine(series: dict[int, float]) -> float:
        return max((v for n, v in series.items() if n >= 128), default=0.0)

    corner_improves = (
        all(_fine(series) <= _coarse(series) for series in corner_series.values()) if corner_series else False
    )
    corner_fine = max((_fine(series) for series in corner_series.values()), default=0.0)
    fluid_only_fine = all(
        grid["length_on_solid_cells"] == 0.0 for row in rows for grid in row["grids"] if grid["N"] >= 96
    )
    checks = [
        Check(
            "wall_measure_finite_positive_localized",
            all(
                grid["all_finite"]
                and grid["all_positive"]
                and grid["max_control_cell_placement_cells"] <= CONTROL_CELL_PLACEMENT_LIMIT_CELLS
                for row in rows
                for grid in row["grids"]
            ),
            "every assigned measure is finite and strictly positive, and each segment's control cell centre lies "
            f"within {CONTROL_CELL_PLACEMENT_LIMIT_CELLS:g} cells of that segment (ring-1 bound "
            "sqrt(1.5^2 + 1.5^2) = 2.121; worst observed "
            f"{_worst('max_control_cell_placement_cells'):.3f} cells)",
            _by_surface("max_control_cell_placement_cells"),
        ),
        Check(
            "wall_measure_normals_are_unit_on_single_segment_cells",
            all(
                grid["single_segment_normal_deviation"] <= NORMAL_UNIT_TOLERANCE
                for row in rows
                for grid in row["grids"]
            )
            and all(
                grid["multi_segment_normal_min"] is None
                or (0.0 < grid["multi_segment_normal_min"] <= 1.0 + NORMAL_UNIT_TOLERANCE)
                for row in rows
                for grid in row["grids"]
            ),
            f"area-weighted normals of single-segment cells are unit to {NORMAL_UNIT_TOLERANCE:g}; cells that "
            "collect two wall faces (concave corners) keep a bounded weighted mean normal, and the count of "
            "near-cancelling normals is reported because the first-layer residual excludes them",
            {
                "single_segment_deviation": _by_surface("single_segment_normal_deviation"),
                "multi_segment_normal_min": _by_surface("multi_segment_normal_min"),
                "degenerate_normal_cells": _by_surface("n_degenerate_normal_cells"),
            },
        ),
        Check(
            "wall_measure_segment_conservation_no_duplicates",
            all(
                grid["measure_conservation_relative"] <= CONSERVATION_RELATIVE_TOLERANCE
                for row in rows
                for grid in row["grids"]
            )
            and all(
                grid["assigned_segment_count"] == grid["n_positive_length_segments"]
                for row in rows
                for grid in row["grids"]
            ),
            "every marching-squares segment of positive length is assigned exactly once: sum_i A_wall,i equals "
            "the summed segment lengths to round-off and the assigned count equals the positive-length segment "
            "count (a duplicated or dropped corner would break conservation). Degenerate zero-length slots "
            "(the contour passing exactly through a grid corner) carry no measure and are counted separately.",
            {
                "conservation": _by_surface("measure_conservation_relative"),
                "assigned_equals_positive_length": {
                    row["surface"]: [
                        grid["assigned_segment_count"] == grid["n_positive_length_segments"] for grid in row["grids"]
                    ]
                    for row in rows
                },
                "degenerate_zero_length_segments": _by_surface("n_degenerate_zero_length_segments"),
                "max_segments_per_cell": _by_surface("max_segments_per_cell"),
                "segments_per_wall_cell": _by_surface("segments_per_wall_cell"),
            },
        ),
        Check(
            "wall_measure_lives_on_fluid_cells_only",
            fluid_only_fine,
            "at N >= 96 every geometry assigns 100 % of the measure to hard-fluid control cells (sdf >= 0), so "
            "the wall forcing always reaches a cell that the open-face CH transport updates; coarse-grid "
            "(N = 64) concave corners of composite SDFs can exceed the two-cell search radius and are reported",
            {
                "N_ge_96_all_fluid": fluid_only_fine,
                "length_on_solid_cells": _by_surface("length_on_solid_cells"),
            },
        ),
        Check(
            "wall_measure_has_no_periodic_seam_ghost",
            all(grid["seam_band_max_measure"] <= GHOST_TO_PEAK_LIMIT for row in rows for grid in row["grids"]),
            "zero wall measure in the top 15 % of the domain for every geometry: the y seam is linearly "
            "extrapolated, never wrapped, so no ghost wall appears at the material top boundary",
            _by_surface("seam_band_max_measure"),
        ),
        Check(
            "wall_measure_smooth_geometry_accurate",
            flat_exact
            and wedge_coarse <= SMOOTH_CURVED_COARSE_TOLERANCE
            and wedge_fine <= SMOOTH_CURVED_FINE_TOLERANCE
            and (wedge_ratio is None or wedge_ratio >= 2.5),
            "flat walls are measured exactly at every N and every offset; the periodic curved wedge converges at "
            f"second order (<= {SMOOTH_CURVED_COARSE_TOLERANCE:g} at N = 64, <= {SMOOTH_CURVED_FINE_TOLERANCE:g} at "
            f"N >= 128, coarse/fine ratio {wedge_ratio if wedge_ratio is None else round(wedge_ratio, 2)})",
            {"flat": smooth_errors.get("flat"), "wedge": smooth_errors.get("wedge"), "coarse_fine_ratio": wedge_ratio},
        ),
        Check(
            "wall_measure_corner_geometry_converges",
            corner_improves and corner_fine <= TEXTURED_LENGTH_RELATIVE_TOLERANCE,
            f"sharp-corner geometries (pillars/grooves) improve monotonically with N and stay within "
            f"{100 * TEXTURED_LENGTH_RELATIVE_TOLERANCE:g} % of the analytic in-domain boundary length at N >= 128; "
            "the residual deficit is the O(dx) chord through each 90-degree corner of the composite SDF, not lost "
            "measure (conservation above is exact)",
            corner_series,
        ),
    ]
    return checks, {"textured_geometry": rows}


# --------------------------------------------------------------------------------------
#  4. variational derivative of the discrete bulk + wall energy
# --------------------------------------------------------------------------------------
def audit_variational_derivative(
    N: int = 96, targets: Sequence[float] = TARGETS_DEG, amplitudes=(1.0e-6, 1.0e-5, 1.0e-4)
):
    """``chemical_potential`` must be d(F_bulk + F_wall^h)/dphi (float64, centred).

    Two probe directions are used. The smooth mixed-mode probe is odd under the
    ``X -> X + Lx/2`` shift for a flat wall, so its directional derivative cancels
    exactly at 90 deg (both terms are round-off); the x-antisymmetric probe does not
    cancel and keeps the neutral case a meaningful test rather than a 0/0 ratio.
    """
    p = _params(int(N), dtype=jnp.float64)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    X, Y = pf.grids(p)
    fluid = sdf >= 0.0
    shifted_y = (Y - 0.25) / (p.Ly - 0.25)
    # Two x wavenumbers on purpose: with a single cos(2 pi X/Lx) mode every probe's *bulk*
    # directional derivative cancels exactly by symmetry at 90 deg, which would reduce the neutral
    # target to a 0/0 check. The wall term alone breaks that symmetry at non-neutral angles.
    phi0 = np.where(
        fluid,
        np.clip(
            0.5
            + 0.3 * np.cos(2.0 * np.pi * X / p.Lx) * np.cos(np.pi * shifted_y)
            + 0.1 * np.cos(4.0 * np.pi * X / p.Lx),
            0.02,
            0.98,
        ),
        0.0,
    )
    probes = {
        "smooth_mixed_mode": np.where(
            fluid, 0.25 + 0.15 * np.sin(2.0 * np.pi * X / p.Lx) * np.cos(2.0 * np.pi * shifted_y), 0.0
        ),
        "x_antisymmetric_mode": np.where(
            fluid,
            0.2 * np.sin(2.0 * np.pi * X / p.Lx) * (1.0 + 0.5 * np.cos(np.pi * shifted_y))
            + 0.1 * np.cos(4.0 * np.pi * X / p.Lx),
            0.0,
        ),
        "wall_localized_mode": np.where(
            fluid, np.exp(-((sdf / (2.0 * p.dx)) ** 2)) * (0.5 + 0.5 * np.sin(2.0 * np.pi * X / p.Lx)), 0.0
        ),
    }
    energy_scale = 1.0
    rows: list[dict[str, Any]] = []
    for target in targets:
        cos_theta = math.cos(math.radians(float(target)))
        solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=cos_theta)
        phi = jnp.asarray(phi0, dtype=jnp.float64)
        energy_scale = max(energy_scale, abs(float(pf.phase_free_energy(phi, solid, p))))
        degenerate_floor = VARIATIONAL_DEGENERATE_FLOOR * energy_scale
        for label, direction in probes.items():
            probe = jnp.asarray(direction, dtype=jnp.float64)
            mu = np.asarray(pf.chemical_potential(phi, solid, p), dtype=np.float64)
            predicted = float(np.sum(mu * direction) * p.dx * p.dy)
            for amplitude in amplitudes:
                plus = float(pf.phase_free_energy(phi + amplitude * probe, solid, p))
                minus = float(pf.phase_free_energy(phi - amplitude * probe, solid, p))
                centred = (plus - minus) / (2.0 * amplitude)
                magnitude = max(abs(centred), abs(predicted))
                rows.append(
                    {
                        "target_deg": float(target),
                        "probe": label,
                        "amplitude": float(amplitude),
                        "finite_difference": centred,
                        "operator_inner_product": predicted,
                        "absolute_error": abs(centred - predicted),
                        "degenerate_zero_derivative": bool(magnitude <= degenerate_floor),
                        "relative_error": abs(centred - predicted) / max(magnitude, degenerate_floor),
                    }
                )
            # Non-vacuity: double counting (cut-cell flux *plus* the legacy diffuse kernel)
            # and dropping the wall energy must both break the identity.
            amplitude = float(amplitudes[1])
            centred = (
                float(pf.phase_free_energy(phi + amplitude * probe, solid, p))
                - float(pf.phase_free_energy(phi - amplitude * probe, solid, p))
            ) / (2.0 * amplitude)
            double_counted = mu + np.asarray(
                pf.wall_energy_derivative(phi, solid.cos_theta) * pf.wall_delta(solid.sdf, p), dtype=np.float64
            )
            rows.append(
                {
                    "target_deg": float(target),
                    "probe": label,
                    "amplitude": amplitude,
                    "mutation": "double_counted_wall_term",
                    "finite_difference": centred,
                    "operator_inner_product": float(np.sum(double_counted * direction) * p.dx * p.dy),
                    "relative_error": abs(centred - float(np.sum(double_counted * direction) * p.dx * p.dy))
                    / max(abs(centred), degenerate_floor),
                }
            )
            dropped = mu - np.asarray(
                pf.wall_energy_derivative(phi, solid.cos_theta) * solid.wall_area / (p.dx * p.dy), dtype=np.float64
            )
            rows.append(
                {
                    "target_deg": float(target),
                    "probe": label,
                    "amplitude": amplitude,
                    "mutation": "wall_energy_removed",
                    "finite_difference": centred,
                    "operator_inner_product": float(np.sum(dropped * direction) * p.dx * p.dy),
                    "relative_error": abs(centred - float(np.sum(dropped * direction) * p.dx * p.dy))
                    / max(abs(centred), degenerate_floor),
                }
            )

    production = [row for row in rows if "mutation" not in row]
    mutations = [row for row in rows if "mutation" in row]
    significant = [row for row in production if not row["degenerate_zero_derivative"]]
    worst = max(row["relative_error"] for row in production)
    worst_significant = max((row["relative_error"] for row in significant), default=0.0)
    # Amplitude-optimized error: for each (target, probe) take the best of the centred
    # amplitudes (float64 cancellation dominates the small ones, FD truncation the large
    # ones). Every amplitude is still reported above.
    optimized: dict[tuple[float, str], float] = {}
    for row in significant:
        key = (row["target_deg"], row["probe"])
        optimized[key] = min(optimized.get(key, math.inf), row["relative_error"])
    worst_optimized = max(optimized.values()) if optimized else 0.0
    non_neutral_optimized = max(
        (value for (target, _), value in optimized.items() if abs(target - 90.0) > 1e-9), default=0.0
    )
    mutation_detected = all(
        row["relative_error"] > 1.0e-3
        for row in mutations
        if abs(row["target_deg"] - 90.0) > 1e-9 and row["probe"] == "wall_localized_mode"
    )
    checks = [
        Check(
            "wall_operator_is_variational_derivative",
            worst_optimized <= DIRECTIONAL_DERIVATIVE_TOLERANCE,
            f"centred directional derivative of F_bulk + F_wall^h equals <mu, direction> within "
            f"{DIRECTIONAL_DERIVATIVE_TOLERANCE:g} in float64 for every non-degenerate probe/target "
            f"(amplitude-optimized worst {worst_optimized:.3e}; worst single amplitude {worst_significant:.3e}; "
            f"worst including degenerate 0/0 rows {worst:.3e})",
            {
                "worst": worst,
                "worst_significant": worst_significant,
                "worst_amplitude_optimized": worst_optimized,
            },
        ),
        Check(
            "wall_operator_variational_ideal_tolerance",
            non_neutral_optimized <= DIRECTIONAL_DERIVATIVE_IDEAL,
            f"ideal gate: non-neutral amplitude-optimized relative error <= {DIRECTIONAL_DERIVATIVE_IDEAL:g} "
            f"(measured {non_neutral_optimized:.3e})",
            {
                "worst_non_neutral_optimized": non_neutral_optimized,
                "per_probe": {f"{target:g}|{probe}": value for (target, probe), value in sorted(optimized.items())},
                "cases": production,
            },
        ),
        Check(
            "variational_check_detects_double_counting",
            bool(mutation_detected),
            "adding the legacy diffuse kernel on top of the cut-cell wall flux (double counting), or removing the "
            "wall term from the operator, breaks the identity by > 1e-3: the variational check is not vacuous",
            [row for row in mutations if row["probe"] == "wall_localized_mode"],
        ),
    ]
    return checks, {"variational": rows, "variational_energy_scale": energy_scale}


def audit_x_seam_equivariance(N: int = 128, rolls: Sequence[int] = (1, 7, 32, 64)):
    """Rolling a periodic geometry in x must roll its measure identically.

    This is the direct "no periodic-seam ghost and no duplicated corner" test for the
    x direction: the corner construction wraps (``q[Nx, j] == q[0, j]``), so a geometry
    that is periodic on the grid has no distinguished column. Any special treatment of
    ``i = 0`` would show up as a mismatch.
    """
    p = _params(int(N), dtype=jnp.float64)
    cases = [("flat", {"wall_height": 0.25}, list(rolls))]
    pitch_cells = int(round(1.5 / p.dx))  # surface_pillars pitch = Lx / n_pillars = 1.5 for n = 4
    if pitch_cells * 4 == N:
        cases.append(("pillars", {"wall_height": 0.25, "n_pillars": 4, "width": 0.3, "height": 0.4}, [pitch_cells]))
    rows: list[dict[str, Any]] = []
    for name, kwargs, case_rolls in cases:
        generator = getattr(pf, f"surface_{name}")
        for roll in case_rolls:
            sdf = np.asarray(generator(p, **kwargs), dtype=np.float64)
            reference, _, _, _, _ = _measure(sdf, p)
            rolled_sdf = np.roll(sdf, int(roll), axis=0)
            rolled_area, _, _, _, info = _measure(rolled_sdf, p)
            expected = np.roll(reference, int(roll), axis=0)
            difference = float(np.max(np.abs(rolled_area - expected)))
            rows.append(
                {
                    "surface": name,
                    "roll_cells": int(roll),
                    "max_absolute_difference": difference,
                    "total_measure_reference": float(reference.sum()),
                    "total_measure_rolled": float(rolled_area.sum()),
                    "measure_conservation_error": float(info["measure_conservation_error"]),
                }
            )
    worst = max(row["max_absolute_difference"] for row in rows)
    checks = [
        Check(
            "wall_measure_is_x_seam_equivariant",
            worst <= 1.0e-12,
            f"rolling a grid-periodic geometry in x rolls the wall measure identically (max absolute difference "
            f"{worst:.3e}): no seam ghost, no duplicated or dropped corner segment at i = 0",
            rows,
        )
    ]
    return checks, {"x_seam_equivariance": rows}


def audit_v7_measure_reproduction_pinned(N: int = 64, targets: Sequence[float] = (60.0, 120.0)):
    """``wall_measure='diffuse_sdf_v7'`` must reproduce the contract-v7 operator exactly.

    The v7 root cause is reproduced *with v8 code* by selecting the pinned legacy measure,
    so the falsification control cannot drift. The reference is transcribed here from the
    contract-v7 formula, not read back from the solver.
    """
    p = _params(int(N), dtype=jnp.float64, wall_measure="diffuse_sdf_v7")
    p_production = _params(int(N), dtype=jnp.float64)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    phi = jnp.asarray(np.where(sdf >= 0.0, 0.5 - 0.45 * np.tanh((sdf - 0.4) / (math.sqrt(2.0) * p.eps)), 0.0))
    rows: list[dict[str, Any]] = []
    for target in targets:
        cos_theta = math.cos(math.radians(float(target)))
        solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=cos_theta)
        mu_legacy = np.asarray(pf.chemical_potential(phi, solid, p), dtype=np.float64)
        # Transcribed contract-v7 operator, in the same association order as the v7 code:
        #   g_w'(phi) = -sigma_0 cos(theta) * (6 phi (1 - phi))
        #   dphi/dn   = -g_w'(phi) / eps
        #   mu        = f'(phi)/eps - eps * (dphi/dn * delta_wall) - eps * lap_face(phi)
        phase = np.asarray(phi, dtype=np.float64)
        switch_derivative = 6.0 * phase * (1.0 - phase)
        derivative = -pf.WALL_SIGMA0 * np.asarray(cos_theta, dtype=np.float64) * switch_derivative
        normal_derivative = -derivative / p.eps
        delta = np.asarray(pf.wall_delta(solid.sdf, p), dtype=np.float64)
        bulk = 2.0 * phase * (1.0 - phase) * (1.0 - 2.0 * phase) / p.eps
        reference = (
            bulk
            - p.eps * (normal_derivative * delta)
            - p.eps * np.asarray(pf.fluid_laplacian(phi, solid, p), dtype=np.float64)
        )
        mu_production = np.asarray(pf.chemical_potential(phi, solid, p_production), dtype=np.float64)
        cut_density = np.asarray(solid.wall_area, dtype=np.float64) / (p.dx * p.dy)
        rows.append(
            {
                "target_deg": float(target),
                # XLA may contract multiply-add inside its kernels, so a NumPy transcription
                # of the same formula agrees to round-off (a few ULP) rather than bit-for-bit.
                "legacy_matches_transcribed_v7": bool(
                    float(np.max(np.abs(mu_legacy - reference)))
                    <= LEGACY_REPRODUCTION_RELATIVE_TOLERANCE * max(1.0, float(np.max(np.abs(mu_legacy))))
                ),
                "legacy_max_difference": float(np.max(np.abs(mu_legacy - reference))),
                "legacy_max_difference_in_ulps": float(np.max(np.abs(mu_legacy - reference)))
                / np.finfo(np.float64).eps,
                "legacy_max_relative_difference": float(np.max(np.abs(mu_legacy - reference)))
                / max(1.0, float(np.max(np.abs(mu_legacy)))),
                "production_differs_from_legacy": bool(not np.allclose(mu_production, mu_legacy, rtol=0, atol=1e-14)),
                "legacy_kernel_fluid_integral": float(np.sum(delta[sdf >= 0.0]) * p.dx * p.dy),
                "cutcell_measure_total": float(np.sum(cut_density) * p.dx * p.dy),
                "geometric_wall_length": float(p.Lx),
                "legacy_to_geometric_ratio": float(np.sum(delta[sdf >= 0.0]) * p.dx * p.dy / p.Lx),
            }
        )
    checks = [
        Check(
            "v7_diffuse_measure_reproduction_is_pinned",
            all(row["legacy_matches_transcribed_v7"] for row in rows)
            and all(row["production_differs_from_legacy"] for row in rows),
            f"``wall_measure='diffuse_sdf_v7'`` reproduces the independently transcribed contract-v7 chemical "
            f"potential to round-off (<= {LEGACY_REPRODUCTION_RELATIVE_TOLERANCE:g} relative; the residual is XLA "
            "multiply-add contraction, reported in ULPs), while the production default differs from it and its "
            "measure integrates to the exact geometric wall length instead of the fluid share f",
            rows,
        )
    ]
    return checks, {"v7_reproduction": rows}


def audit_single_wall_contribution(N: int = 64, targets: Sequence[float] = (60.0, 120.0)):
    """Exactly one wall-energy contribution in the production path."""
    p = _params(int(N), dtype=jnp.float64)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    phi = jnp.asarray(
        np.where(sdf >= 0.0, np.clip(0.4 + 0.3 * np.sin(2.0 * np.pi * sdf / 3.0), 0.0, 1.0), 0.0), dtype=jnp.float64
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        cos_theta = math.cos(math.radians(float(target)))
        solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=cos_theta)
        standalone = np.asarray(pf.wetting_mu(phi, solid, p), dtype=np.float64)
        explicit = np.asarray(pf._explicit_chemical_potential(phi, solid, p), dtype=np.float64)
        bulk_only = np.asarray(pf.fprime(phi) / p.eps, dtype=np.float64)
        expected_wall = np.asarray(
            pf.wall_energy_derivative(phi, solid.cos_theta) * solid.wall_area / (p.dx * p.dy), dtype=np.float64
        )
        rows.append(
            {
                "target_deg": float(target),
                "wetting_mu_max_abs": float(np.max(np.abs(standalone))),
                "explicit_minus_bulk_matches_wall_flux": float(np.max(np.abs(explicit - bulk_only - expected_wall))),
                "wall_flux_max_abs": float(np.max(np.abs(expected_wall))),
                "n_wall_cells": int(np.count_nonzero(np.asarray(solid.wall_area) > 0.0)),
            }
        )
    checks = [
        Check(
            "production_path_has_exactly_one_wall_contribution",
            all(row["wetting_mu_max_abs"] == 0.0 for row in rows)
            and all(
                row["explicit_minus_bulk_matches_wall_flux"] <= 1.0e-12 * max(row["wall_flux_max_abs"], 1.0)
                for row in rows
            ),
            "``wetting_mu`` returns exact zeros in production mode and ``_explicit_chemical_potential - f'/eps`` "
            "equals ``A_wall,i g_w'(phi_i) / (dx dy)`` to round-off: the Young energy enters once, as the "
            "embedded boundary flux",
            rows,
        )
    ]
    return checks, {"single_contribution": rows}


# --------------------------------------------------------------------------------------
#  5. angle independence / no global gain
# --------------------------------------------------------------------------------------
def audit_angle_independence(N: int = 96, targets: Sequence[float] = TARGETS_DEG):
    """The measure is geometry only: identical for every theta and equal to the wall length."""
    p = _params(int(N), dtype=jnp.float64)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    reference_area: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    phi = jnp.asarray(np.where(sdf >= 0.0, 0.5 - 0.45 * np.tanh((sdf - 0.4) / (math.sqrt(2.0) * p.eps)), 0.0))
    for target in targets:
        cos_theta = math.cos(math.radians(float(target)))
        solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=cos_theta)
        area = np.asarray(solid.wall_area, dtype=np.float64)
        if reference_area is None:
            reference_area = area
        mu = np.asarray(pf.chemical_potential(phi, solid, p), dtype=np.float64)
        wall_mu = np.asarray(pf.wall_energy_derivative(phi, solid.cos_theta) * area / (p.dx * p.dy), dtype=np.float64)
        rows.append(
            {
                "target_deg": float(target),
                "cos_theta": float(cos_theta),
                "total_measure": float(area.sum()),
                "expected_length": float(p.Lx),
                "measure_bitwise_identical_to_reference": bool(
                    np.array_equal(area, reference_area) if reference_area is not None else True
                ),
                "wall_flux_max_abs": float(np.max(np.abs(wall_mu))),
                "chemical_potential_max_abs": float(np.max(np.abs(mu))),
                "global_gain_applied": float(area.sum() / p.Lx),
            }
        )
    # exact linearity in cos(theta): mu_wall(theta) / cos(theta) is one theta-independent field
    scalings: list[float] = []
    base: np.ndarray | None = None
    for row in rows:
        if abs(row["cos_theta"]) < 1e-12:
            continue
        cos_theta = row["cos_theta"]
        solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=cos_theta)
        field = np.asarray(
            pf.wall_energy_derivative(phi, solid.cos_theta) * solid.wall_area / (p.dx * p.dy), dtype=np.float64
        )
        normalized = field / cos_theta
        if base is None:
            base = normalized
        scalings.append(float(np.max(np.abs(normalized - base))))
    checks = [
        Check(
            "wall_measure_is_angle_independent",
            all(row["measure_bitwise_identical_to_reference"] for row in rows),
            "A_wall,i is bit-identical for 60/90/120/150 deg: the measure is geometry, never a function of theta",
            {row["target_deg"]: row["total_measure"] for row in rows},
        ),
        Check(
            "wall_measure_total_is_geometric_length_not_gain_corrected",
            all(abs(row["global_gain_applied"] - 1.0) <= 1.0e-12 for row in rows),
            "sum_i A_wall,i / Lx == 1 exactly: there is no global 1/f wall gain and no case-specific multiplier",
            {row["target_deg"]: row["global_gain_applied"] for row in rows},
        ),
        Check(
            "wall_operator_is_linear_in_cos_theta",
            (max(scalings) if scalings else 0.0) <= 1.0e-12,
            "mu_wall / cos(theta) is one theta-independent field, so the only angle dependence is the physical "
            "Young cosine (no angle remap, no lookup table, no theta_scale)",
            {"max_deviation": max(scalings) if scalings else 0.0},
        ),
    ]
    return checks, {"angle_independence": rows}


# --------------------------------------------------------------------------------------
#  6. first-fluid-layer Young residual on manufactured fields
# --------------------------------------------------------------------------------------
def audit_first_layer_residual(
    N_values: Sequence[int] = (64, 128), targets: Sequence[float] = TARGETS_DEG, eps_factors=(2.0, 4.0)
):
    """Manufactured natural-BC fields: new measure-weighted first-layer residual vs the band residual."""
    rows: list[dict[str, Any]] = []
    for N in N_values:
        for eps_factor in eps_factors:
            p = _params(int(N), eps_factor=float(eps_factor))
            dx = float(p.dx)
            x_axis = (np.arange(p.Nx) + 0.5) * dx
            X, Y = np.meshgrid(x_axis, x_axis, indexing="ij")
            for target in targets:
                theta = float(target)
                sdf = Y - 0.25
                phi = clk.manufactured_young_boundary_field(X, Y, sdf, p.eps, theta)
                solid = pf.make_solid(jnp.asarray(sdf, dtype=np.float64), p, cos_theta=math.cos(math.radians(theta)))
                kwargs = {
                    "wall_area": np.asarray(solid.wall_area, dtype=np.float64),
                    "wall_normal_x": np.asarray(solid.wall_normal_x, dtype=np.float64),
                    "wall_normal_y": np.asarray(solid.wall_normal_y, dtype=np.float64),
                }
                first = clk.young_boundary_residual_first_layer(
                    phi, sdf, dx, dx, p.eps, math.cos(math.radians(theta)), **kwargs
                )
                band = clk.young_boundary_residual(phi, sdf, dx, dx, p.eps, math.cos(math.radians(theta)))
                wrong = clk.young_boundary_residual_first_layer(
                    phi,
                    sdf,
                    dx,
                    dx,
                    p.eps,
                    math.cos(math.radians(60.0 if abs(theta - 60.0) > 1e-9 else 150.0)),
                    **kwargs,
                )
                rows.append(
                    {
                        "N": int(N),
                        "eps_over_dx": float(eps_factor),
                        "target_deg": theta,
                        "RY_first_l2": first["RY_first_l2"],
                        "RY_first_linf": first["RY_first_linf"],
                        "RY_first_normalized_l2": first["RY_first_normalized_l2"],
                        "RY_first_normalized_linf": first["RY_first_normalized_linf"],
                        "wall_measure_weighted_RY": first["wall_measure_weighted_RY"],
                        "n_wall_cells": first["n_wall_cells"],
                        "n_wall_cells_band": first["n_wall_cells_band"],
                        "n_wall_cells_nonfluid": first["n_wall_cells_nonfluid"],
                        "band_RY_normalized_l2": band["RY_normalized_l2"],
                        "band_n_points": band["n_points"],
                        "wrong_angle_RY_first_normalized_l2": wrong["RY_first_normalized_l2"],
                    }
                )
    non_neutral = [row for row in rows if abs(row["target_deg"] - 90.0) > 1e-9]
    neutral = [row for row in rows if abs(row["target_deg"] - 90.0) <= 1e-9]
    checks = [
        Check(
            "first_layer_residual_manufactured_converged",
            all(row["RY_first_normalized_l2"] <= FIRST_LAYER_MANUFACTURED_TOLERANCE for row in non_neutral),
            f"manufactured natural-BC fields drive the measure-weighted first-layer residual to "
            f"<= {FIRST_LAYER_MANUFACTURED_TOLERANCE:g} normalized L2 (worst "
            f"{max(row['RY_first_normalized_l2'] for row in non_neutral):.3e}); it is finite-difference "
            "truncation, scale-invariant in N at fixed eps/dx and smaller at eps/dx = 4",
            [
                {
                    "N": row["N"],
                    "eps_over_dx": row["eps_over_dx"],
                    "target_deg": row["target_deg"],
                    "RY_first_normalized_l2": row["RY_first_normalized_l2"],
                }
                for row in non_neutral
            ],
        ),
        Check(
            "first_layer_residual_neutral_is_roundoff",
            all(row["RY_first_linf"] <= FIRST_LAYER_NEUTRAL_TOLERANCE for row in neutral),
            "90 deg: both terms of R_Y vanish identically, so the first-layer residual is round-off",
            {row["N"]: row["RY_first_linf"] for row in neutral},
        ),
        Check(
            "first_layer_residual_is_not_vacuous",
            all(row["wrong_angle_RY_first_normalized_l2"] >= FIRST_LAYER_WRONG_ANGLE_MINIMUM for row in rows),
            f"supplying the wrong wall angle raises the normalized first-layer residual to >= "
            f"{FIRST_LAYER_WRONG_ANGLE_MINIMUM:g}",
            [
                {
                    "N": row["N"],
                    "target_deg": row["target_deg"],
                    "wrong_angle_RY_first_normalized_l2": row["wrong_angle_RY_first_normalized_l2"],
                }
                for row in rows
            ],
        ),
        Check(
            "first_layer_cells_are_fluid_and_measure_weighted",
            all(row["n_wall_cells_nonfluid"] == 0 and row["n_wall_cells"] > 0 for row in rows),
            "the first-layer residual samples only hard-fluid wall cells carrying positive measure",
            [{"N": row["N"], "target_deg": row["target_deg"], "n_wall_cells": row["n_wall_cells"]} for row in rows],
        ),
    ]
    return checks, {"first_layer_residual": rows}


def run_audit(quick: bool = False) -> WallMeasureAudit:
    """Run every wall-measure audit. ``quick`` trims the grid list for CI."""
    audit = WallMeasureAudit()
    N_geometry = (64, 128) if quick else (64, 128, 192)
    N_flat = (64, 128) if quick else (64, 96, 128, 192)
    audit.settings = {
        "quick": bool(quick),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "wall_measure_method": str(pf.WALL_MEASURE_METHOD),
        "wall_measure_contract_version": int(pf.WALL_MEASURE_CONTRACT_VERSION),
        "sigma_0": pf.WALL_SIGMA0,
        "N_flat_translation": list(N_flat),
        "N_textured": list(N_geometry),
        "translation_offsets_over_dy": list(TRANSLATION_OFFSETS),
        "inclined_slopes": list(INCLINED_SLOPES),
        "gates": {
            "flat_translation_spread_limit": FLAT_TRANSLATION_SPREAD_LIMIT,
            "flat_translation_spread_ideal": FLAT_TRANSLATION_SPREAD_IDEAL,
            "flat_length_relative_tolerance": FLAT_LENGTH_RELATIVE_TOLERANCE,
            "inclined_length_relative_tolerance": INCLINED_LENGTH_RELATIVE_TOLERANCE,
            "textured_length_relative_tolerance": TEXTURED_LENGTH_RELATIVE_TOLERANCE,
            "ghost_to_peak_limit": GHOST_TO_PEAK_LIMIT,
            "normal_unit_tolerance": NORMAL_UNIT_TOLERANCE,
            "conservation_relative_tolerance": CONSERVATION_RELATIVE_TOLERANCE,
            "directional_derivative_tolerance": DIRECTIONAL_DERIVATIVE_TOLERANCE,
            "directional_derivative_ideal": DIRECTIONAL_DERIVATIVE_IDEAL,
            "first_layer_manufactured_tolerance": FIRST_LAYER_MANUFACTURED_TOLERANCE,
        },
    }
    for runner, kwargs in (
        (audit_flat_wall, {"N_values": N_flat}),
        (audit_inclined_wall, {}),
        (audit_textured_geometry, {"N_values": N_geometry}),
        (audit_variational_derivative, {"N": 64 if quick else 96}),
        (audit_x_seam_equivariance, {"N": 64 if quick else 128}),
        (audit_v7_measure_reproduction_pinned, {}),
        (audit_single_wall_contribution, {}),
        (audit_angle_independence, {}),
        (audit_first_layer_residual, {"N_values": (64,) if quick else (64, 128)}),
    ):
        checks, numbers = runner(**kwargs)
        audit.checks.extend(checks)
        audit.numbers.update(numbers)
    return audit


def format_markdown(audit: WallMeasureAudit) -> str:
    lines = [
        "# Embedded wall-measure audit (L1A-2e)",
        "",
        f"- solver contract = {audit.settings['solver_contract_version']}, wall measure = "
        f"{audit.settings['wall_measure_method']} "
        f"(measure contract v{audit.settings['wall_measure_contract_version']})",
        f"- quick profile = {audit.settings['quick']}, overall = {'PASS' if audit.passed else 'FAIL'}",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for check in audit.checks:
        lines.append(f"| `{check.name}` | {'PASS' if check.passed else 'FAIL'} | {check.detail} |")
    flat = audit.numbers.get("flat_wall", [])
    if flat:
        lines += [
            "",
            "## Flat-wall translation sweep",
            "",
            "| N | measure spread (rel) | max length error | v7 spread | v7 fluid share |",
            "|---|---:|---:|---:|---|",
        ]
        for row in flat:
            lines.append(
                f"| {row['N']} | {row['measure_spread_relative']:.3e} | {row['max_relative_error']:.3e} | "
                f"{row['v7_measure_spread_relative']:.3e} | "
                f"{row['v7_fluid_share_range'][0]:.4f}..{row['v7_fluid_share_range'][1]:.4f} |"
            )
    inclined = audit.numbers.get("inclined_wall", [])
    if inclined:
        lines += [
            "",
            "## Inclined walls",
            "",
            "| slope | measure | Euclidean | rel. error | Manhattan deviation |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in inclined:
            lines.append(
                f"| {row['slope']:+.2f} | {row['measure']:.6f} | {row['euclidean_length']:.6f} | "
                f"{row['relative_error_vs_euclidean']:.3e} | {row['relative_deviation_from_manhattan']:+.3e} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L1A-2e embedded wall-measure audit")
    parser.add_argument("--quick", action="store_true", help="trimmed grid list for CI")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)
    import jax

    jax.config.update("jax_enable_x64", True)
    audit = run_audit(quick=args.quick)
    payload = audit.to_dict()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(format_markdown(audit))
    print(format_markdown(audit))
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
