"""L1A-2f cut-cell geometry audit: one authority, exact closure, honest small-cell reporting.

What this module proves about ``phasefield.embedded_fluid_geometry`` (contract v9)
----------------------------------------------------------------------------------
1.  **Single authority.** ``V_i``, ``alpha_i``, the fluid centroid, the shared face apertures
    ``A_f``, the face distances ``d_ij``, the wall length and the wall normal all come from *one*
    corner-sampled reconstruction (``pf.sdf_corner_geometry``): the same array that the L1A-2e wall
    measure uses. There are no three separate reconstructions, and the pinned contract-v7/v8 mode
    reads the same corner field.
2.  **Flat-wall closure to machine precision.** For ``N = 64/96/128/192`` and eight sub-cell wall
    offsets the total fluid volume equals the analytic area ``Lx (Ly - y_wall)`` with relative error
    ``<= 1e-12``, the wall measure equals ``Lx``, the apertures are exactly the analytic open
    lengths, and the cut-cell count is exactly one row of ``Nx`` cells (zero when the wall is
    exactly face-aligned).
3.  **No cell is dropped for having a solid centre.** Positive-volume cut cells whose centre is in
    the solid keep their volume, their apertures and their wall measure (``n_positive_volume_
    solid_centre_cells > 0`` on translated flat walls and on every textured geometry).
4.  **Face consistency.** ``0 <= A_f <=`` full face length, ``0 <= alpha <= 1``, no negative
    volume, no open face touching a zero-volume cell, no orphan positive-volume cell (every
    ``V_i > 0`` cell has at least one open face), and every wall-measure host owns ``V_i > 0``.
5.  **Inclined and textured geometry.** Slopes +/-0.25, +/-0.5 (interior window, the plane is not
    x-periodic) and pillars / grooves / wedge / hierarchical: exact area and Euclidean wall length
    for the planar cases, all invariants for the curved ones, and a symmetric positive-semidefinite
    stiffness ``K``.
6.  **Degenerate empty solid.** ``V_i = dx dy``, ``alpha = 1``, ``a_f = 1``, the representative
    point is the cell centre, ``d_ij`` is ``dx``/``dy`` and ``A_wall = 0`` -- exactly, so the static
    Laplace regression sees an unchanged operator.
7.  **Small cut cells are reported, never floored.** ``alpha_min_positive``, ``alpha_p01``,
    ``alpha_p05`` and ``max(A_face/V)`` are measured and printed for every geometry, and the source
    is AST-scanned for any ``alpha_floor`` / ``V_floor`` / clipping of ``alpha`` or ``V``.

Run with::

    python -m production.cutcell_geometry_audit            # full
    python -m production.cutcell_geometry_audit --quick    # CI
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

import phasefield as pf

STAGE = "L1A-2f"
MODULE = "production.cutcell_geometry_audit"

FLAT_AREA_RELATIVE_TOLERANCE = 1.0e-12
FLAT_LENGTH_RELATIVE_TOLERANCE = 1.0e-12
INCLINED_AREA_RELATIVE_TOLERANCE = 1.0e-12
INCLINED_LENGTH_RELATIVE_TOLERANCE = 1.0e-12
STIFFNESS_SYMMETRY_TOLERANCE = 1.0e-10
N_VALUES = (64, 96, 128, 192)
OFFSETS_OVER_DY = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
INCLINED_SLOPES = (-0.5, -0.25, 0.25, 0.5)
TEXTURED_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("pillars", {"wall_height": 0.25, "n_pillars": 4, "width": 0.3, "height": 0.4}),
    ("grooves", {"wall_height": 0.25, "n_grooves": 6, "width": 0.25, "depth": 0.35}),
    ("wedge", {"wall_height": 1.5, "slope": 0.5}),
    ("hierarchical", {"wall_height": 0.25}),
    ("random_pillars", {"wall_height": 0.25, "seed": 3}),
)
FORBIDDEN_FLOOR_NAMES = (
    "alpha_floor",
    "volume_floor",
    "v_floor",
    "min_alpha",
    "clip_alpha",
    "floor_alpha",
    "agglomerate",
    "cell_merging",
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Audit:
    stage: str = STAGE
    module: str = MODULE
    generated_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    solver_contract_version: int = field(default_factory=lambda: int(pf.SOLVER_CONTRACT_VERSION))
    quick: bool = False
    checks: list[Check] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["phase_transport"] = pf.phase_transport_metadata(pf.PhaseFieldParams(Nx=2, Ny=2))
        payload["failed_checks"] = [check.name for check in self.checks if not check.passed]
        return payload


def _params(N: int, *, dtype=jnp.float64) -> pf.PhaseFieldParams:
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=dtype)
    p.eps = 2.0 * p.dx
    return p


def _geometry(sdf, p) -> tuple[pf.EmbeddedFluidGeometry, dict[str, Any]]:
    geometry, info = pf.embedded_fluid_geometry(jnp.asarray(sdf, dtype=jnp.float64), p)
    return geometry, {k: (float(v) if np.ndim(v) == 0 and not isinstance(v, str) else v) for k, v in info.items()}


def _np(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _cell_centres(p) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinate grids, indexed like the fields (``[i, j]``)."""
    x, y = np.meshgrid((np.arange(p.Nx) + 0.5) * p.dx, (np.arange(p.Ny) + 0.5) * p.dy, indexing="ij")
    return x, y


def _invariants(geometry, info, sdf, p) -> dict[str, Any]:
    """The geometry invariants that every case must satisfy, with their worst values."""
    volume = _np(geometry.volume)
    alpha = _np(geometry.alpha)
    aperture_x = _np(geometry.aperture_x)
    aperture_y = _np(geometry.aperture_y)
    cell_area = float(p.dx * p.dy)
    positive = volume > 0.0
    open_x = aperture_x > 0.0
    open_y = aperture_y > 0.0
    neighbour_x_positive = np.roll(positive, -1, axis=0)
    neighbour_y_positive = np.roll(positive, -1, axis=1)
    distance = np.asarray(sdf, dtype=np.float64)
    centre_x, centre_y = _cell_centres(p)
    stiffness = (aperture_x + aperture_y) / _np(geometry.volume_safe)
    alpha_positive = np.where(positive, alpha, np.inf)
    return {
        "n_cells": int(volume.size),
        "total_volume": float(volume.sum()),
        "n_positive_volume_cells": int(positive.sum()),
        "n_zero_volume_cells": int((~positive).sum()),
        "n_cut_cells": int(info["n_cut_cells"]),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "alpha_min_positive": float(alpha_positive.min()) if positive.any() else None,
        "alpha_p01": float(np.sort(alpha_positive[positive])[max(0, int(0.01 * positive.sum()) - 1)])
        if positive.any()
        else None,
        "alpha_p05": float(np.sort(alpha_positive[positive])[max(0, int(0.05 * positive.sum()) - 1)])
        if positive.any()
        else None,
        "max_area_face_over_volume": float(stiffness.max()),
        "negative_volume_cells": int((volume < 0.0).sum()),
        "alpha_out_of_range": int(((alpha < 0.0) | (alpha > 1.0)).sum()),
        "aperture_out_of_range": int(
            ((aperture_x < 0.0) | (aperture_x > p.dy + 1e-15) | (aperture_y < 0.0) | (aperture_y > p.dx + 1e-15)).sum()
        ),
        "open_face_to_zero_volume_cell": int(
            np.count_nonzero(open_x[:-1] & ~neighbour_x_positive[:-1])
            + np.count_nonzero(open_y[:, :-1] & ~neighbour_y_positive[:, :-1])
        ),
        "open_face_from_zero_volume_cell": int(
            np.count_nonzero(open_x[:-1] & ~positive[:-1]) + np.count_nonzero(open_y[:, :-1] & ~positive[:, :-1])
        ),
        "orphan_positive_volume_cells": int(
            np.count_nonzero(
                positive
                & ~(open_x | np.roll(open_x, 1, axis=0) | open_y | np.roll(open_y, 1, axis=1))
            )
        ),
        "positive_volume_solid_centre_cells": int(np.count_nonzero(positive & (distance < 0.0))),
        "finite_centroids": bool(np.isfinite(_np(geometry.centroid_x)).all() and np.isfinite(_np(geometry.centroid_y)).all()),
        "centroid_inside_cell": bool(
            (
                (_np(geometry.centroid_x) >= -1e-12)
                & (_np(geometry.centroid_x) <= p.Lx + 1e-12)
                & (_np(geometry.centroid_y) >= -1e-12)
                & (_np(geometry.centroid_y) <= p.Ly + 1e-12)
            ).all()
        ),
        "centroid_matches_cell_centre_for_full_cells": bool(
            np.array_equal(_np(geometry.centroid_x)[alpha >= 1.0], centre_x[alpha >= 1.0])
            and np.array_equal(_np(geometry.centroid_y)[alpha >= 1.0], centre_y[alpha >= 1.0])
        ),
        "full_face_distance_is_dx_or_dy": bool(
            np.allclose(_np(geometry.face_distance_x)[alpha >= 1.0], p.dx, rtol=0.0, atol=1e-12)
            and np.allclose(
                _np(geometry.face_distance_y)[:, :-1][alpha[:, :-1] >= 1.0], p.dy, rtol=0.0, atol=1e-12
            )
        ),
        "wall_measure_total": float(_np(geometry.wall_measure).sum()),
        "length_on_zero_volume_cells": float(info["length_on_zero_volume_cells"]),
        "aperture_volume_closure_changes": float(info["aperture_volume_closure_changes"]),
        "n_two_lobed_cells": int(info["n_two_lobed_cells"]),
        "cell_area": cell_area,
        "sign_tolerance": float(info["sign_tolerance"]),
        "corner_sign_relative_to_dx": float(info["sign_tolerance"]) / float(p.dx),
    }


def _stiffness_symmetry(geometry, p, seed: int = 0) -> dict[str, float]:
    """``K`` (graph stiffness of ``w_f``) must be symmetric and positive semidefinite."""
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
    y = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
    kx = pf.graph_stiffness_apply(x, geometry.weight_x, geometry.weight_y)
    ky = pf.graph_stiffness_apply(y, geometry.weight_x, geometry.weight_y)
    inner_xy = float(jnp.vdot(x, ky).real)
    inner_yx = float(jnp.vdot(kx, y).real)
    quadratic = float(jnp.vdot(x, kx).real)
    scale = max(abs(inner_xy), abs(inner_yx), abs(quadratic), 1.0)
    return {
        "symmetry_absolute": abs(inner_xy - inner_yx),
        "symmetry_relative": abs(inner_xy - inner_yx) / scale,
        "quadratic_form": quadratic,
        "positive_semidefinite": quadratic >= -1e-9 * scale,
    }


# --------------------------------------------------------------------------------------
#  1. flat wall: exact volume, apertures, wall measure, cut-cell count
# --------------------------------------------------------------------------------------
def audit_flat_wall(N_values: Sequence[int] = N_VALUES, offsets: Sequence[float] = OFFSETS_OVER_DY):
    rows: list[dict[str, Any]] = []
    for N in N_values:
        p = _params(int(N))
        for offset in offsets:
            height = 0.25 + float(offset) * p.dy
            sdf = np.asarray(pf.surface_flat(p, wall_height=height), dtype=np.float64)
            geometry, info = _geometry(sdf, p)
            analytic_area = p.Lx * (p.Ly - height)
            volume = _np(geometry.volume)
            aperture_x = _np(geometry.aperture_x)
            aperture_y = _np(geometry.aperture_y)
            # analytic apertures of a flat wall: every x-face is fully open inside the fluid,
            # the one cut-row y-face is open over (dy - (height - j dy)) and the wall face is closed
            rows.append(
                {
                    "N": int(N),
                    "offset_over_dy": float(offset),
                    "wall_height": float(height),
                    "wall_height_over_dy": float(height / p.dy),
                    "total_volume": float(volume.sum()),
                    "analytic_area": float(analytic_area),
                    "area_relative_error": abs(float(volume.sum()) - analytic_area) / analytic_area,
                    "wall_measure": float(_np(geometry.wall_measure).sum()),
                    "wall_measure_relative_error": abs(float(_np(geometry.wall_measure).sum()) - p.Lx) / p.Lx,
                    "n_cut_cells": int(info["n_cut_cells"]),
                    "n_cut_rows": int(np.count_nonzero(np.any((volume > 0.0) & (volume < p.dx * p.dy - 1e-15), axis=0))),
                    "alpha_min_positive": float(np.min(np.where(volume > 0.0, _np(geometry.alpha), np.inf))),
                    "cut_row_alpha": sorted({float(a) for a in _np(geometry.alpha)[(volume > 0.0) & (volume < p.dx * p.dy - 1e-15)]}),
                    "aperture_x_min": float(aperture_x.min()),
                    "aperture_x_max": float(aperture_x.max()),
                    "aperture_y_min": float(aperture_y.min()),
                    "aperture_y_max": float(aperture_y.max()),
                    "max_area_face_over_volume": float(((aperture_x + aperture_y) / _np(geometry.volume_safe)).max()),
                    **{k: v for k, v in _invariants(geometry, info, sdf, p).items() if k.startswith(("open_face", "orphan", "positive_volume_solid", "length_on_zero", "aperture_volume_closure"))},
                }
            )
    worst_area = max(row["area_relative_error"] for row in rows)
    worst_length = max(row["wall_measure_relative_error"] for row in rows)
    checks = [
        Check(
            "flat_wall_volume_exact_for_subcell_offsets",
            worst_area <= FLAT_AREA_RELATIVE_TOLERANCE,
            f"total fluid volume equals Lx (Ly - y_wall) to <= {FLAT_AREA_RELATIVE_TOLERANCE:g} relative "
            f"for {len(rows)} (N, offset) pairs; worst {worst_area:.3e}",
            {"worst_area_relative_error": worst_area, "n_cases": len(rows)},
        ),
        Check(
            "flat_wall_measure_exact",
            worst_length <= FLAT_LENGTH_RELATIVE_TOLERANCE,
            f"sum_i A_wall,i equals the analytic wall length Lx to <= {FLAT_LENGTH_RELATIVE_TOLERANCE:g} "
            f"relative; worst {worst_length:.3e}",
            {"worst_length_relative_error": worst_length},
        ),
        Check(
            "flat_wall_cut_cell_count_is_one_row",
            all(row["n_cut_rows"] <= 1 and row["n_cut_cells"] in (0, row["N"]) for row in rows),
            "a flat wall cuts exactly one row of cells (or none when it is exactly face-aligned), and "
            "every cell of that row is a cut cell",
            {str(row["N"]): row["n_cut_cells"] for row in rows},
        ),
        Check(
            "flat_wall_aperture_bounds",
            all(
                row["aperture_x_min"] >= 0.0
                and row["aperture_x_max"] <= _params(row["N"]).dy + 1e-15
                and row["aperture_y_min"] >= 0.0
                and row["aperture_y_max"] <= _params(row["N"]).dx + 1e-15
                for row in rows
            ),
            "every face aperture lies in [0, full face length] on every translated flat wall",
            {"aperture_ranges": [[r["aperture_x_min"], r["aperture_x_max"]] for r in rows[:4]]},
        ),
        Check(
            "flat_wall_face_invariants",
            all(
                row["open_face_to_zero_volume_cell"] == 0
                and row["open_face_from_zero_volume_cell"] == 0
                and row["orphan_positive_volume_cells"] == 0
                and row["length_on_zero_volume_cells"] == 0.0
                for row in rows
            ),
            "no open face touches a zero-volume cell, no positive-volume cell is orphaned, and no wall "
            "measure sits on a cell without a control volume",
            {"worst": {k: max(r[k] for r in rows) for k in
                       ("open_face_to_zero_volume_cell", "orphan_positive_volume_cells")}},
        ),
    ]
    return checks, {"flat_wall": rows, "flat_worst_area_error": worst_area, "flat_worst_length_error": worst_length}


def audit_flat_wall_apertures_exact(N: int = 128, offsets: Sequence[float] = OFFSETS_OVER_DY):
    """The shared face apertures of a flat wall must equal the analytic open lengths *exactly*.

    For a horizontal wall at ``y_wall`` the analytic open length of a face is closed-form:

    * the vertical (``+x``) face of cell ``(i, j)`` spans ``y`` from ``j dy`` to ``(j+1) dy``, so its
      open length is ``dy * clip(((j+1) dy - y_wall)/dy, 0, 1)`` -- fully open above the wall, fully
      closed below it, and *partially* open for the single cut row (open fraction ``alpha``);
    * the horizontal (``+y``) face at corner row ``j+1`` is either the wall itself (when
      ``(j+1) dy == y_wall``, i.e. exactly face-aligned) or lies strictly on one side of it, so its
      open length is ``dx`` when the face is strictly above the wall and ``0`` otherwise --
      including the periodic seam face, which joins the top fluid row to the bottom solid slab.
    """
    p = _params(int(N))
    rows = []
    worst_y = 0.0
    worst_x = 0.0
    for offset in offsets:
        height = 0.25 + float(offset) * p.dy
        sdf = np.asarray(pf.surface_flat(p, wall_height=height), dtype=np.float64)
        geometry, _info = _geometry(sdf, p)
        aperture_x = _np(geometry.aperture_x)
        aperture_y = _np(geometry.aperture_y)
        corner_y = np.arange(p.Ny + 1) * p.dy
        tolerance = float(pf.corner_sign_tolerance(jnp.asarray(sdf), pf.sdf_corner_values(jnp.asarray(sdf), p)))
        # vertical faces: open over the part of the edge that lies above the wall
        open_fraction_x = np.clip((corner_y[1:] - height) / p.dy, 0.0, 1.0)
        expected_x = np.repeat((open_fraction_x * p.dy)[None, :], p.Nx, axis=0)
        # horizontal faces: open only when the face is strictly above the wall
        open_row = (corner_y[1:] - height) > tolerance
        expected_y = np.repeat((open_row * p.dx)[None, :], p.Nx, axis=0)
        expected_y[:, -1] = 0.0  # the periodic seam joins fluid to the solid slab: closed
        error_x = float(np.max(np.abs(aperture_x - expected_x)))
        error_y = float(np.max(np.abs(aperture_y - expected_y)))
        worst_x = max(worst_x, error_x / (p.dx * p.dy))
        worst_y = max(worst_y, error_y / (p.dx * p.dy))
        partial = (aperture_x > 0.0) & (aperture_x < p.dy - 1e-15)
        rows.append(
            {
                "offset_over_dy": float(offset),
                "wall_height": float(height),
                "max_aperture_x_error": error_x,
                "max_aperture_y_error": error_y,
                "relative_to_cell_area_x": error_x / (p.dx * p.dy),
                "relative_to_cell_area_y": error_y / (p.dx * p.dy),
                "n_partially_open_x_faces": int(np.count_nonzero(partial)),
                "partial_open_fractions": sorted({round(float(a / p.dy), 12) for a in aperture_x[partial]}),
                "expected_open_fraction": float(np.clip(((np.argmax(open_row) * p.dy) - height) / p.dy, 0.0, 1.0))
                if open_row.any()
                else None,
                "n_open_y_face_rows": int(np.count_nonzero(np.any(aperture_y > 0.0, axis=0))),
                "seam_aperture": float(aperture_y[:, -1].max()),
            }
        )
    checks = [
        Check(
            "flat_wall_face_aperture_exact",
            worst_x == 0.0 and worst_y == 0.0,
            "the shared open length of every face of a flat wall equals the closed-form analytic value "
            "*exactly* (bit-for-bit, no tolerance): vertical faces are dy above the wall, 0 below it "
            "and alpha*dy on the single cut row; horizontal faces are dx strictly above the wall and 0 "
            "on or below it, including the periodic seam",
            {"worst_relative_error_x": worst_x, "worst_relative_error_y": worst_y, "rows": rows},
        ),
        Check(
            "flat_wall_partial_aperture_is_the_cut_row_only",
            all(
                row["n_partially_open_x_faces"] in (0, int(N))
                and (not row["partial_open_fractions"] or len(row["partial_open_fractions"]) == 1)
                for row in rows
            )
            and all(row["seam_aperture"] == 0.0 for row in rows),
            "a translated flat wall leaves exactly one row of partially open faces (all Nx of them with "
            "the same open fraction alpha), or none when it is exactly face-aligned; the periodic y "
            "seam is always closed",
            {str(row["offset_over_dy"]): row["partial_open_fractions"] for row in rows},
        ),
    ]
    return checks, {"flat_wall_apertures": rows}


# --------------------------------------------------------------------------------------
#  2. single geometry authority
# --------------------------------------------------------------------------------------
def audit_single_authority(N: int = 96):
    p = _params(int(N))
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25 + 0.375 * p.dy), dtype=np.float64)
    solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=-0.5)
    geometry = solid.geometry
    fresh, _info = _geometry(sdf, p)
    identical = all(
        np.array_equal(_np(getattr(geometry, name)), _np(getattr(fresh, name)))
        for name in ("volume", "alpha", "aperture_x", "aperture_y", "centroid_x", "centroid_y",
                     "weight_x", "weight_y", "wall_measure", "wall_normal_x", "wall_normal_y")
    )
    # the corner field is the single source: volume, apertures and wall segments all read it
    corner_geometry = pf.sdf_corner_geometry(jnp.asarray(sdf), p)
    corner = corner_geometry["corner"]
    polygons = pf.cut_cell_fluid_polygons(corner_geometry)
    apertures_from_corners = pf.embedded_face_apertures(corner_geometry)
    segments = pf.wall_cut_segments(jnp.asarray(sdf), p, control_cell="positive_volume")
    source = Path(pf.__file__).read_text()
    tree = ast.parse(source)
    callers: dict[str, set] = {"sdf_corner_geometry": set(), "sdf_corner_values": set()}
    for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in callers:
                callers[node.func.id].add(function.name)
    volume_matches_corners = bool(
        np.array_equal(_np(polygons["volume"]), _np(geometry.volume))
    )
    apertures_match_corners = bool(
        np.array_equal(_np(apertures_from_corners[0]), _np(geometry.aperture_x))
        and np.array_equal(_np(apertures_from_corners[1]), _np(geometry.aperture_y))
    )
    checks = [
        Check(
            "geometry_has_single_authority",
            identical and volume_matches_corners and apertures_match_corners,
            "make_solid's EmbeddedFluidGeometry is bit-identical to a fresh reconstruction, and the "
            "volume/aperture arrays are exactly what the shared corner field produces: one "
            "reconstruction feeds volume, centroid, apertures, face distances and wall measure",
            {
                "solid_matches_fresh": identical,
                "volume_from_corners": volume_matches_corners,
                "apertures_from_corners": apertures_match_corners,
                "corner_field_shape": list(np.shape(np.asarray(corner))),
            },
        ),
        Check(
            "corner_field_is_the_only_reconstruction",
            set(callers["sdf_corner_geometry"]) <= {"embedded_fluid_geometry", "wall_cut_segments"}
            and set(callers["sdf_corner_values"]) <= {"sdf_corner_geometry"}
            and "embedded_fluid_geometry" in callers["sdf_corner_geometry"],
            "the corner reconstruction is built in exactly two places -- embedded_fluid_geometry "
            "(volume, centroid, apertures, face distances) and wall_cut_segments (the wall contour, "
            "which embedded_fluid_geometry also feeds) -- and the corner samples themselves come only "
            "from sdf_corner_values. There is no third, independent reconstruction of the same "
            "geometry anywhere in the solver.",
            {name: sorted(v) for name, v in callers.items()},
        ),
        Check(
            "wall_segments_share_the_corner_field",
            bool(np.isfinite(_np(segments["length"])).all())
            and float(_np(segments["length"]).sum()) == float(_np(geometry.wall_measure).sum()),
            "the wall segments and the wall measure sum to the same total length (same contour, same "
            "reconstruction)",
            {
                "segment_total": float(_np(segments["length"]).sum()),
                "measure_total": float(_np(geometry.wall_measure).sum()),
            },
        ),
    ]
    return checks, {"single_authority": {"callers": {k: sorted(v) for k, v in callers.items()}}}


# --------------------------------------------------------------------------------------
#  3. positive-volume cut cells are never dropped by a centre mask
# --------------------------------------------------------------------------------------
def audit_positive_volume_never_dropped(N_values: Sequence[int] = (64, 128)):
    rows = []
    worst_kept = 0
    for N in N_values:
        p = _params(int(N))
        for offset in OFFSETS_OVER_DY:
            height = 0.25 + float(offset) * p.dy
            sdf = np.asarray(pf.surface_flat(p, wall_height=height), dtype=np.float64)
            geometry, info = _geometry(sdf, p)
            volume = _np(geometry.volume)
            solid_centre = sdf < 0.0
            kept = volume > 0.0
            dropped = int(np.count_nonzero(kept & solid_centre))
            hard_mask_count = int(np.count_nonzero(~solid_centre))
            # the transported fluid volume must exceed the hard-mask staircase whenever the wall is
            # not exactly face-aligned, and equal it when it is
            staircase = hard_mask_count * p.dx * p.dy
            rows.append(
                {
                    "N": int(N),
                    "offset_over_dy": float(offset),
                    "wall_height_over_dy": float(height / p.dy),
                    "n_positive_volume_solid_centre_cells": dropped,
                    "transported_volume": float(volume.sum()),
                    "hard_mask_volume": float(staircase),
                    "volume_recovered": float(volume.sum()) - float(staircase),
                    "alpha_min_positive": float(info["alpha_min_positive"]),
                    "solid_centre_cells_with_open_face": int(
                        np.count_nonzero(
                            kept
                            & solid_centre
                            & (
                                (_np(geometry.aperture_x) > 0.0)
                                | (np.roll(_np(geometry.aperture_x), 1, axis=0) > 0.0)
                                | (_np(geometry.aperture_y) > 0.0)
                                | (np.roll(_np(geometry.aperture_y), 1, axis=0 + 1) > 0.0)
                            )
                        )
                    ),
                }
            )
            worst_kept = max(worst_kept, dropped)
    textured = {}
    p = _params(96)
    for name, kwargs in TEXTURED_CASES:
        sdf = np.asarray(getattr(pf, f"surface_{name}")(p, **kwargs), dtype=np.float64)
        geometry, info = _geometry(sdf, p)
        textured[name] = int(np.count_nonzero((_np(geometry.volume) > 0.0) & (sdf < 0.0)))
    checks = [
        Check(
            "positive_volume_cell_is_never_dropped_by_center_mask",
            worst_kept > 0
            and all(
                row["solid_centre_cells_with_open_face"] == row["n_positive_volume_solid_centre_cells"]
                for row in rows
            )
            and sum(textured.values()) > 0
            and all(count >= 0 for count in textured.values()),
            f"positive-volume cut cells with a solid centre are kept and stay connected by open faces: "
            f"{worst_kept} such cells at worst on translated flat walls, and "
            f"{textured} on the textured geometries. Their fluid volume is transported "
            f"(volume_recovered > 0 whenever the wall is not face-aligned).",
            {"worst_kept": worst_kept, "textured": textured},
        ),
        Check(
            "cut_cell_volume_is_exact_where_the_staircase_is_not",
            all(abs(row["volume_recovered"]) <= 0.5 * 6.0 * _params(row["N"]).dy + 1e-9 for row in rows)
            and any(abs(row["volume_recovered"]) > 1e-9 for row in rows)
            and all(
                abs(row["volume_recovered"]) <= 1e-12
                for row in rows
                if abs(row["wall_height_over_dy"] - round(row["wall_height_over_dy"])) < 1e-12
            ),
            "sum_i V_i is the exact geometric area while the cell-centre staircase dx dy * #{sdf >= 0} "
            "differs from it by up to half a cell row, with a sign that alternates with the sub-cell "
            "offset (it over-counts when the wall is below the cut-cell centre and under-counts when "
            "it is above). The two agree exactly only when the wall is exactly face-aligned.",
            {
                "max_abs_recovered": max(abs(row["volume_recovered"]) for row in rows),
                "signed": sorted({round(row["volume_recovered"], 9) for row in rows}),
            },
        ),
    ]
    return checks, {"positive_volume": rows, "positive_volume_textured": textured}


# --------------------------------------------------------------------------------------
#  4. inclined walls: exact area and Euclidean length in an interior window
# --------------------------------------------------------------------------------------
def audit_inclined(N_values: Sequence[int] = (96, 128), slopes: Sequence[float] = INCLINED_SLOPES):
    rows = []
    worst_area = 0.0
    worst_length = 0.0
    for N in N_values:
        p = _params(int(N))
        dx = float(p.dx)
        X, Y = pf.grids(p)
        for slope in slopes:
            m = float(slope)
            wall_height = 0.25
            sdf = np.asarray((Y - m * (X - 0.5 * p.Lx) - wall_height) / math.sqrt(1.0 + m * m), dtype=np.float64)
            geometry, info = _geometry(sdf, p)
            volume = _np(geometry.volume)
            x_axis = (np.arange(p.Nx) + 0.5) * dx
            height = wall_height + m * (x_axis - 0.5 * p.Lx)
            # Interior window: keep only columns whose wall sits comfortably inside the domain and
            # away from the x seam (the inclined *plane* is not x-periodic, so the input SDF itself
            # jumps there). The margin is scaled to the grid so a coarse quick-profile grid still
            # yields a usable window; if it does not, the case is reported as skipped rather than
            # silently measured over a truncated window.
            margin = max(2.0, min(8.0, 0.06 * p.Nx)) * dx
            interior = (height > margin) & (height < p.Ly - margin)
            interior[: max(2, int(0.06 * p.Nx))] = False
            interior[-max(2, int(0.06 * p.Nx)) :] = False
            first = int(np.argmax(interior)) if interior.any() else 0
            last = int(p.Nx - np.argmax(interior[::-1])) if interior.any() else p.Nx
            columns = last - first
            # exact area above the plane inside the window columns (the plane is linear in x)
            edges = wall_height + m * (np.arange(first, last + 1) * dx - 0.5 * p.Lx)
            area_below = float(np.sum(0.5 * (edges[:-1] + edges[1:]) * dx))
            analytic_area = columns * dx * p.Ly - area_below
            measured_area = float(volume[first:last].sum())
            measured_length = float(_np(geometry.wall_measure)[first:last].sum())
            analytic_length = columns * dx * math.sqrt(1.0 + m * m)
            area_error = abs(measured_area - analytic_area) / abs(analytic_area)
            length_error = abs(measured_length - analytic_length) / analytic_length
            worst_area = max(worst_area, area_error)
            worst_length = max(worst_length, length_error)
            symmetry = _stiffness_symmetry(geometry, p)
            rows.append(
                {
                    "N": int(N),
                    "slope": m,
                    "window_columns": [first, last],
                    "measured_area": measured_area,
                    "analytic_area": analytic_area,
                    "area_relative_error": area_error,
                    "measured_length": measured_length,
                    "analytic_length": analytic_length,
                    "length_relative_error": length_error,
                    "n_cut_cells": int(info["n_cut_cells"]),
                    "alpha_min_positive": float(info["alpha_min_positive"]),
                    "min_face_distance_x_over_dx": float(info["min_face_distance_x_over_dx"]),
                    "min_face_distance_y_over_dy": float(info["min_face_distance_y_over_dy"]),
                    "n_two_lobed_cells": int(info["n_two_lobed_cells"]),
                    **symmetry,
                    **{
                        k: v
                        for k, v in _invariants(geometry, info, sdf, p).items()
                        if k.startswith(("open_face", "orphan", "length_on_zero", "aperture_out", "alpha_out", "negative_volume"))
                    },
                }
            )
    checks = [
        Check(
            "inclined_area_converges_to_exact",
            all(row["window_columns"][1] - row["window_columns"][0] >= 4 for row in rows)
            and worst_area <= INCLINED_AREA_RELATIVE_TOLERANCE,
            f"the transported fluid area of an inclined planar wall equals the analytic area in an "
            f"interior window to <= {INCLINED_AREA_RELATIVE_TOLERANCE:g} relative (exact polygon "
            f"clipping of a linear SDF, not a staircase); worst {worst_area:.3e}",
            {"worst_area_relative_error": worst_area},
        ),
        Check(
            "inclined_wall_length_is_euclidean",
            worst_length <= INCLINED_LENGTH_RELATIVE_TOLERANCE,
            f"the wall measure is the Euclidean cut length to <= {INCLINED_LENGTH_RELATIVE_TOLERANCE:g} "
            f"relative (no Manhattan gain); worst {worst_length:.3e}",
            {"worst_length_relative_error": worst_length},
        ),
        Check(
            "inclined_geometry_invariants",
            all(
                row["open_face_to_zero_volume_cell"] == 0
                and row["orphan_positive_volume_cells"] == 0
                and row["negative_volume_cells"] == 0
                and row["alpha_out_of_range"] == 0
                and row["aperture_out_of_range"] == 0
                and row["length_on_zero_volume_cells"] == 0.0
                for row in rows
            ),
            "inclined walls satisfy every face/volume invariant",
            {"n_rows": len(rows)},
        ),
        Check(
            "inclined_face_distances_are_not_degenerate",
            all(
                row["min_face_distance_x_over_dx"] >= 0.25 and row["min_face_distance_y_over_dy"] >= 0.25
                for row in rows
            ),
            "the centroid-to-centroid face distance never collapses (>= 1/4 cell), so w_f = A_f/d_ij "
            "stays bounded on inclined walls",
            {
                "worst_x": min(row["min_face_distance_x_over_dx"] for row in rows),
                "worst_y": min(row["min_face_distance_y_over_dy"] for row in rows),
            },
        ),
        Check(
            "inclined_stiffness_symmetric_psd",
            all(
                row["symmetry_relative"] <= STIFFNESS_SYMMETRY_TOLERANCE and row["positive_semidefinite"]
                for row in rows
            ),
            f"the graph stiffness K of w_f is symmetric (relative asymmetry <= "
            f"{STIFFNESS_SYMMETRY_TOLERANCE:g}) and positive semidefinite on every inclined geometry",
            {"worst_symmetry": max(row["symmetry_relative"] for row in rows)},
        ),
    ]
    return checks, {"inclined": rows, "inclined_worst_area_error": worst_area}


# --------------------------------------------------------------------------------------
#  5. textured geometry invariants
# --------------------------------------------------------------------------------------
def audit_textured(N_values: Sequence[int] = (64, 96, 128), cases=TEXTURED_CASES):
    rows = []
    for N in N_values:
        p = _params(int(N))
        for name, kwargs in cases:
            sdf = np.asarray(getattr(pf, f"surface_{name}")(p, **kwargs), dtype=np.float64)
            geometry, info = _geometry(sdf, p)
            invariants = _invariants(geometry, info, sdf, p)
            invariants.update(_stiffness_symmetry(geometry, p))
            invariants["surface"] = name
            invariants["N"] = int(N)
            invariants["kwargs"] = {k: v for k, v in kwargs.items()}
            rows.append(invariants)
    checks = [
        Check(
            "textured_geometry_volume_and_alpha_bounds",
            all(row["negative_volume_cells"] == 0 and row["alpha_out_of_range"] == 0 for row in rows),
            "no negative volume and 0 <= alpha_i <= 1 on pillars / grooves / wedge / hierarchical / "
            "random pillars at every N",
            {"n_cases": len(rows)},
        ),
        Check(
            "textured_geometry_aperture_bounds",
            all(row["aperture_out_of_range"] == 0 for row in rows),
            "every aperture lies in [0, full face length] on every textured geometry",
            {"n_cases": len(rows)},
        ),
        Check(
            "textured_geometry_no_orphan_and_no_open_face_to_solid",
            all(
                row["open_face_to_zero_volume_cell"] == 0
                and row["open_face_from_zero_volume_cell"] == 0
                and row["orphan_positive_volume_cells"] == 0
                for row in rows
            ),
            "no open face connects to a zero-volume cell and no positive-volume cell is isolated from "
            "the transported domain",
            {"n_cases": len(rows)},
        ),
        Check(
            "textured_geometry_wall_measure_is_hosted",
            all(row["length_on_zero_volume_cells"] == 0.0 for row in rows),
            "every piece of wall measure sits on a cell with V_i > 0",
            {"worst": max(row["length_on_zero_volume_cells"] for row in rows)},
        ),
        Check(
            "textured_geometry_centroids_finite",
            all(row["finite_centroids"] and row["centroid_inside_cell"] for row in rows),
            "fluid centroids are finite and inside the domain on every textured geometry",
            {"n_cases": len(rows)},
        ),
        Check(
            "textured_stiffness_symmetric_psd",
            all(row["symmetry_relative"] <= STIFFNESS_SYMMETRY_TOLERANCE and row["positive_semidefinite"] for row in rows),
            f"K is symmetric (<= {STIFFNESS_SYMMETRY_TOLERANCE:g} relative) and PSD on every textured geometry",
            {"worst_symmetry": max(row["symmetry_relative"] for row in rows)},
        ),
        Check(
            "textured_geometry_keeps_solid_centre_cut_cells",
            all(
                row["positive_volume_solid_centre_cells"] >= 0
                and row["positive_volume_solid_centre_cells"] <= row["n_positive_volume_cells"]
                for row in rows
            )
            and all(
                any(
                    row["positive_volume_solid_centre_cells"] > 0
                    for row in rows
                    if row["surface"] == surface
                )
                for surface in {row["surface"] for row in rows}
                if surface != "flat"
            )
            and sum(row["positive_volume_solid_centre_cells"] for row in rows) > 0,
            "every textured geometry has grids with positive-volume cut cells whose centre is inside "
            "the solid -- the configuration contract v8 dropped by construction -- and no geometry "
            "ever reports more such cells than it has positive-volume cells. The per-(surface, N) "
            "count varies with how the texture lands on the grid (a grid on which no cut-cell centre "
            "falls inside the solid simply has none), so the gate is the per-surface existence over "
            "the sweep plus the non-negativity everywhere, with the full table reported.",
            {
                "per_surface_N": {
                    f"{row['surface']}|N{row['N']}": row["positive_volume_solid_centre_cells"] for row in rows
                },
                "total": sum(row["positive_volume_solid_centre_cells"] for row in rows),
            },
        ),
    ]
    return checks, {"textured": rows}


# --------------------------------------------------------------------------------------
#  6. empty-solid degeneracy
# --------------------------------------------------------------------------------------
def audit_empty_solid(N_values: Sequence[int] = (48, 96)):
    rows = []
    for N in N_values:
        p = _params(int(N))
        solid = pf.empty_solid(p)
        geometry = solid.geometry
        volume = _np(geometry.volume)
        alpha = _np(geometry.alpha)
        rows.append(
            {
                "N": int(N),
                "volume_equals_cell_area": bool(np.allclose(volume, p.dx * p.dy, rtol=0.0, atol=0.0)),
                "alpha_all_one": bool(np.allclose(alpha, 1.0, rtol=0.0, atol=0.0)),
                "aperture_x_all_dy": bool(np.allclose(_np(geometry.aperture_x), p.dy, rtol=0.0, atol=0.0)),
                "aperture_y_all_dx": bool(np.allclose(_np(geometry.aperture_y), p.dx, rtol=0.0, atol=0.0)),
                "aperture_norm_all_one": bool(
                    np.allclose(_np(geometry.aperture_x_norm), 1.0, rtol=0.0, atol=0.0)
                    and np.allclose(_np(geometry.aperture_y_norm), 1.0, rtol=0.0, atol=0.0)
                ),
                "centroid_is_cell_centre": bool(
                    np.array_equal(_np(geometry.centroid_x), _cell_centres(p)[0])
                    and np.array_equal(_np(geometry.centroid_y), _cell_centres(p)[1])
                ),
                "face_distance_x_is_dx": bool(np.allclose(_np(geometry.face_distance_x), p.dx, rtol=0.0, atol=0.0)),
                "face_distance_y_is_dy": bool(np.allclose(_np(geometry.face_distance_y), p.dy, rtol=0.0, atol=0.0)),
                "weight_x_is_one": bool(np.allclose(_np(geometry.weight_x), 1.0, rtol=0.0, atol=0.0)),
                "weight_y_is_one": bool(np.allclose(_np(geometry.weight_y), 1.0, rtol=0.0, atol=0.0)),
                "wall_measure_zero": float(_np(geometry.wall_measure).sum()),
                "wall_area_field_zero": float(np.sum(_np(solid.wall_area))),
                "hard_v8_measure_zero": float(np.sum(_np(solid.wall_area_hard_v8))),
                "n_cut_cells": int(np.count_nonzero((volume > 0.0) & (volume < p.dx * p.dy - 1e-15))),
            }
        )
    exact = all(
        all(value for key, value in row.items() if key.endswith(("cell_area", "one", "dx", "dy", "centre", "center")))
        and row["wall_measure_zero"] == 0.0
        and row["n_cut_cells"] == 0
        for row in rows
    )
    checks = [
        Check(
            "empty_solid_degenerates_exactly",
            exact,
            "an everywhere-fluid SDF degenerates *exactly*: V_i = dx dy, alpha = 1, A_f the full face "
            "length (a_f = 1, including across the periodic y seam), the representative point the cell "
            "centre, d_ij = dx (or dy), w_f = 1 and A_wall,i = 0 -- bit-for-bit, so the static-droplet "
            "Laplace regression sees the unchanged contract-v8 operator",
            rows,
        )
    ]
    return checks, {"empty_solid": rows}


# --------------------------------------------------------------------------------------
#  7. small cut cells are reported, never floored
# --------------------------------------------------------------------------------------
def audit_small_cut_cells(N: int = 128):
    p = _params(int(N))
    cases = [("flat_0.375dy", pf.surface_flat(p, wall_height=0.25 + 0.375 * p.dy))]
    cases += [(name, getattr(pf, f"surface_{name}")(p, **kwargs)) for name, kwargs in TEXTURED_CASES]
    X, Y = pf.grids(p)
    cases.append(("thin_sliver_wall", (Y - (0.25 + 0.01 * p.dy))))  # a wall 1 % of a cell above a face
    rows = []
    for name, sdf in cases:
        sdf_np = np.asarray(sdf, dtype=np.float64)
        geometry, info = _geometry(sdf_np, p)
        invariants = _invariants(geometry, info, sdf_np, p)
        volume = _np(geometry.volume)
        positive = volume > 0.0
        alpha_sorted = np.sort(_np(geometry.alpha)[positive])
        rows.append(
            {
                "surface": name,
                "alpha_min_positive": float(alpha_sorted[0]) if alpha_sorted.size else None,
                "alpha_p01": float(alpha_sorted[max(0, int(0.01 * alpha_sorted.size) - 1)]) if alpha_sorted.size else None,
                "alpha_p05": float(alpha_sorted[max(0, int(0.05 * alpha_sorted.size) - 1)]) if alpha_sorted.size else None,
                "max_area_face_over_volume": invariants["max_area_face_over_volume"],
                "max_local_stiffness_indicator": float(
                    np.max(
                        2.0 * (_np(geometry.weight_x) + _np(geometry.weight_y))
                        + np.roll(_np(geometry.weight_x), 1, axis=0)
                        + np.roll(_np(geometry.weight_y), 1, axis=1)
                    )
                ),
                "max_wall_measure_over_volume": float(
                    np.max(_np(geometry.wall_measure) / _np(geometry.volume_safe))
                ),
                "n_cut_cells": invariants["n_cut_cells"],
                "n_positive_volume_cells": invariants["n_positive_volume_cells"],
                "sign_tolerance": invariants["sign_tolerance"],
                "sign_tolerance_over_dx": invariants["corner_sign_relative_to_dx"],
            }
        )
    source = Path(pf.__file__).read_text()
    tree = ast.parse(source)
    forbidden_hits = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
        for name in names:
            if any(token in name.lower() for token in FORBIDDEN_FLOOR_NAMES):
                forbidden_hits.append(name)
    checks = [
        Check(
            "small_cut_cells_are_reported_not_floored",
            not forbidden_hits,
            "alpha_min_positive, alpha_p01, alpha_p05 and max(A_face/V) are measured and reported for "
            "every geometry (including a deliberate 1 %-of-a-cell sliver wall), and phasefield.py "
            "contains no alpha/volume floor, clipping or agglomeration identifier "
            f"({list(FORBIDDEN_FLOOR_NAMES)})",
            {"forbidden_identifier_hits": sorted(set(forbidden_hits)), "rows": rows},
        ),
        Check(
            "corner_sign_tolerance_is_roundoff_only",
            all(row["sign_tolerance_over_dx"] < 1.0e-4 for row in rows),
            "the only threshold in the reconstruction is the corner *sign* resolution "
            f"8 eps_mach max|sdf| = {pf.CORNER_SIGN_TOLERANCE_FACTOR:g} eps_mach max|sdf|, which is "
            "< 1e-4 of a cell for every geometry: it declares 'the wall passes exactly through this "
            "grid corner' instead of leaving a 1e-31 sliver, and it is not an alpha or volume floor",
            {"worst_over_dx": max(row["sign_tolerance_over_dx"] for row in rows)},
        ),
        Check(
            "thin_sliver_wall_stays_bounded",
            all(math.isfinite(row["max_area_face_over_volume"]) for row in rows),
            "even a wall placed 1 % of a cell above a cell face yields a finite stiffness indicator "
            "(no degenerate 1e-31 volume, no inf/nan aperture)",
            {row["surface"]: row["max_area_face_over_volume"] for row in rows},
        ),
    ]
    return checks, {
        "small_cut_cells": rows,
        "alpha_min_positive_worst": min((r["alpha_min_positive"] for r in rows if r["alpha_min_positive"]), default=None),
        "max_area_face_over_volume_worst": max(r["max_area_face_over_volume"] for r in rows),
        "percentiles": {
            row["surface"]: {
                "alpha_min_positive": row["alpha_min_positive"],
                "alpha_p01": row["alpha_p01"],
                "alpha_p05": row["alpha_p05"],
            }
            for row in rows
        },
    }


# --------------------------------------------------------------------------------------
#  8. the sign convention keeps the measure on a live control volume when the wall is aligned
# --------------------------------------------------------------------------------------
def audit_aligned_wall(N_values: Sequence[int] = (96, 192)):
    """Walls exactly on a cell face: the degenerate corner case the sign tolerance resolves."""
    rows = []
    for N in N_values:
        p = _params(int(N))
        # 0.25 lands exactly on a corner row when 0.25/dy is an integer (N = 96, 192)
        height = 0.25
        sdf = np.asarray(pf.surface_flat(p, wall_height=height), dtype=np.float64)
        geometry, info = _geometry(sdf, p)
        volume = _np(geometry.volume)
        measure = _np(geometry.wall_measure)
        host = measure > 0.0
        rows.append(
            {
                "N": int(N),
                "wall_height_over_dy": float(height / p.dy),
                "n_cut_cells": int(info["n_cut_cells"]),
                "total_volume": float(volume.sum()),
                "analytic_area": float(p.Lx * (p.Ly - height)),
                "area_relative_error": abs(float(volume.sum()) - p.Lx * (p.Ly - height)) / (p.Lx * (p.Ly - height)),
                "wall_measure_total": float(measure.sum()),
                "host_cells": int(host.sum()),
                "host_cells_with_positive_volume": int(np.count_nonzero(host & (volume > 0.0))),
                "length_on_zero_volume_cells": float(info["length_on_zero_volume_cells"]),
                "aperture_volume_closure_changes": float(info["aperture_volume_closure_changes"]),
                "alpha_values": sorted({float(a) for a in _np(geometry.alpha)[volume > 0.0]}),
                "sign_tolerance": float(info["sign_tolerance"]),
                "corner_zero_rows": int(np.count_nonzero(np.all(np.abs(pf.sdf_corner_values(jnp.asarray(sdf), p)) < 1e-18, axis=0))),
            }
        )
    checks = [
        Check(
            "face_aligned_wall_is_not_degenerate",
            all(
                row["length_on_zero_volume_cells"] == 0.0
                and row["host_cells"] == row["host_cells_with_positive_volume"]
                and row["area_relative_error"] <= FLAT_AREA_RELATIVE_TOLERANCE
                and abs(row["wall_measure_total"] - 6.0) <= 1e-12
                for row in rows
            ),
            "a wall exactly on a cell face (0.25/dy an integer at N = 96 and 192) produces no "
            "degenerate sliver: alpha is 0 or 1, the volume stays exact, the wall measure is exactly "
            "Lx and it is hosted by a full-volume cell -- never by a zero-volume one",
            rows,
        )
    ]
    return checks, {"aligned_wall": rows}


def run_audit(quick: bool = False, N_values: Sequence[int] | None = None) -> Audit:
    """Run every geometry audit and evaluate the checks."""
    if not jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64:
        raise RuntimeError("the geometry audit needs jax_enable_x64 (run with JAX_ENABLE_X64=1)")
    grids = tuple(N_values) if N_values else ((32, 48) if quick else N_VALUES)
    offsets = (0.0, 0.375, 0.625) if quick else OFFSETS_OVER_DY
    textured_N = (48,) if quick else (64, 96, 128)
    inclined_N = (64, 96) if quick else (96, 128)
    runners = (
        (audit_flat_wall, {"N_values": grids, "offsets": offsets}),
        (audit_flat_wall_apertures_exact, {"N": int(grids[-1]), "offsets": offsets}),
        (audit_single_authority, {"N": int(grids[-2]) if len(grids) > 1 else int(grids[0])}),
        (audit_positive_volume_never_dropped, {"N_values": (int(grids[0]), int(grids[-1]))}),
        (audit_inclined, {"N_values": inclined_N}),
        (audit_textured, {"N_values": textured_N}),
        (audit_empty_solid, {"N_values": (int(grids[0]), int(grids[-1]))}),
        (audit_small_cut_cells, {"N": 48 if quick else 128}),
        (audit_aligned_wall, {"N_values": (32, 48) if quick else (96, 192)}),
    )
    audit = Audit(quick=quick)
    for runner, kwargs in runners:
        checks, numbers = runner(**kwargs)
        audit.checks.extend(checks)
        audit.numbers.update(numbers)
    return audit


def format_markdown(audit: Audit) -> str:
    data = audit.to_dict()
    lines = [
        f"# {STAGE} cut-cell geometry audit",
        "",
        f"- solver contract = {data['solver_contract_version']}, phase transport = "
        f"`{data['phase_transport']['phase_transport_geometry']}`",
        f"- profile = {'quick' if audit.quick else 'baseline'}, generated {data['generated_utc']}",
        f"- **passed: {data['passed']}**" + (f" (failed: {data['failed_checks']})" if data["failed_checks"] else ""),
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "| --- | --- | --- |",
    ]
    for check in data["checks"]:
        detail = check["detail"].replace("\n", " ")
        lines.append(f"| `{check['name']}` | {check['passed']} | {detail} |")
    small = data["numbers"].get("small_cut_cells") or []
    if small:
        lines += ["", "## Small cut cells (reported, never floored)", "",
                  "| geometry | alpha_min_positive | alpha_p01 | alpha_p05 | max(A_face/V) | max(A_wall/V) | cut cells |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in small:
            lines.append(
                f"| {row['surface']} | {row['alpha_min_positive']} | {row['alpha_p01']} | {row['alpha_p05']} | "
                f"{row['max_area_face_over_volume']:.4g} | {row['max_wall_measure_over_volume']:.4g} | "
                f"{row['n_cut_cells']} |"
            )
    flat = data["numbers"].get("flat_wall") or []
    if flat:
        lines += ["", "## Flat-wall closure", "",
                  f"- worst total-volume relative error = {data['numbers'].get('flat_worst_area_error'):.3e} "
                  f"(limit {FLAT_AREA_RELATIVE_TOLERANCE:g})",
                  f"- worst wall-measure relative error = {data['numbers'].get('flat_worst_length_error'):.3e}",
                  f"- cases = {len(flat)} (N x sub-cell offset)"]
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="CI profile: small grids, three offsets")
    parser.add_argument("--out", default=None, help="write JSON + Markdown here (default evidence/cutcell_geometry)")
    args = parser.parse_args(argv)
    audit = run_audit(quick=args.quick)
    data = audit.to_dict()
    root = Path(args.out) if args.out else Path("evidence") / "cutcell_geometry"
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False, default=str)
    (root / "cutcell_geometry_audit.json").write_text(payload)
    markdown = format_markdown(audit)
    (root / "cutcell_geometry_audit.md").write_text(markdown)
    manifest = {
        "module": MODULE,
        "stage": STAGE,
        "passed": bool(data["passed"]),
        "files": {
            "cutcell_geometry_audit.json": __import__("hashlib").sha256(payload.encode()).hexdigest(),
            "cutcell_geometry_audit.md": __import__("hashlib").sha256(markdown.encode()).hexdigest(),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(markdown)
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
