"""L1A-2y phase 1: the reconstruction-stable ``phi = 0.5`` contour-branch readout.

The readout answers one question for a given phase field: *where is the tracked interface branch
crossing the fixed ray that was pinned once at ``t = 0``?* It never runs a solver and never inspects a
field other than the one it is given, so it can be designed, prespecified and adversarially tested
before the single sanctioned window is ever advanced.

Why a new readout is needed
---------------------------
L1A-2x pinned the *grid edge* that carried the ``t = 0`` crossing and re-evaluated the frozen linear
crossing on that same edge. The reconstructed contour of a diffuse interface does not respect grid
edges: at macro step 25 of that window the crossing passed a shared grid node and the following
segment took over, so the pinned-edge measurement disappeared and the stage reported ``UNRESOLVED``.
The physics did not change at that step, only the support of the measurement.

This readout keeps the *reconstruction convention* identical to the frozen one (the frozen
bilinear/periodic sampler and the frozen linear ``phi = 0.5`` edge crossings) and changes only the
*support rule*: the branch is followed across adjacent grid edges, in the sense that the tracked root
is the continuous root of the same reconstructed curve, matched to the previous root by continuation.
Crossing a shared grid node is therefore an ordinary recorded event; a genuine ambiguity (two roots
the reconstruction cannot separate) still is a failure.

Prespecified rules (frozen before the window is measured)
--------------------------------------------------------
``branch_definition``
    Everything that selects *which* branch is tracked is computed from the ``t = 0`` state alone: the
    frozen global ``phi = 0.5`` minimum-gap support point ``O``, the embedded-wall normal ``n_w``, the
    ``t = 0`` discrete interface normal ``n_phi``, and the ``t = 0`` root of the frozen reconstruction
    on the fixed wall-normal ray through ``O``. No later field, no target direction and no outcome
    enters the definition.
``continuity``
    The tracked root must move by at most ``ROOT_MATCH_MAX_STEP_DX`` between consecutive measurements. A
    crossing that changes the grid edge it lives on is *allowed* and recorded (``segment_index``
    change); a crossing that changes the connected contour branch is a failure.
``root_matching``
    Among the reconstructed roots inside the continuation window ``+/- RAY_SEARCH_HALF_WINDOW_DX`` the
    tracked root is the one nearest the previous root. Matching is by proximity to the previous root
    only - never by direction of motion, magnitude of motion, or any consideration of what verdict the
    implied displacement would support.
``ambiguity``
    Two roots the reconstruction cannot separate (closer than ``ROOT_AMBIGUITY_SEPARATION_DX``, or
    coincident within ``NODE_DEGENERACY_TOLERANCE_DX``), a matched cell whose four corners demand an
    asymptotic decider with a saddle value within ``SADDLE_AMBIGUITY_TOLERANCE`` of the threshold, or a
    traced arc that meets a junction of more than two segments make the step ambiguous. Ambiguity is
    recorded and withholds a verdict; it is never resolved by preferring one candidate.
``failure``
    No root inside the continuation window, a root step beyond the continuity bound, or an arc that is
    not a simple open path through the matched point are failures. A failure withholds a verdict.
``no_favourable_selection``
    The readout is direction-blind: mirroring the wall normal mirrors the reported displacement exactly,
    and a nearer root is always preferred to a farther one even when the farther one would move the
    interface toward the solid.
``cahn_hilliard_pairing``
    The explicit chemical-potential flux and the implicit solve increment are two halves of one
    Cahn-Hilliard update. The readout reports them separately as ingredients, and every physical
    statement is made about their sum; they are never presented as two independent mechanisms.
"""

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

READOUT_VERSION = "contour_branch_v1"
THRESHOLD = 0.5

# --- prespecified constants ---------------------------------------------------------------------
RAY_SEARCH_HALF_WINDOW_DX = 1.5
RAY_SCAN_STEP_DX = 0.05
RAY_ROOT_TOLERANCE_DX = 1.0e-9
ROOT_MATCH_MAX_STEP_DX = 0.25
ROOT_AMBIGUITY_SEPARATION_DX = 0.5
NODE_DEGENERACY_TOLERANCE_DX = 1.0e-9
ARC_WINDOW_RADIUS_DX = 3.0
ARC_MAX_SEGMENTS = 400
ARC_ROOT_MATCH_TOLERANCE_DX = 0.05
ARC_BOUNDARY_TOLERANCE_DX = 1.0e-3
SADDLE_AMBIGUITY_TOLERANCE = 1.0e-3
CROSS_RAY_TOLERANCE_DX = 5.0e-3
GRADIENT_FLOOR_RATIO = 0.5
PROBE_BAND = 0.10
MAX_SUBSTEP_DISPLACEMENT_OVER_EPS = 0.25

PRESPECIFIED_RULES: dict[str, str] = {
    "branch_definition": (
        "the tracked branch is pinned at t = 0 from the frozen state alone: the frozen global phi = 0.5 "
        "minimum-gap support point O, the embedded-wall normal n_w into the solid, the t = 0 discrete "
        "interface normal n_phi, and the t = 0 root of the frozen linear reconstruction on the fixed "
        "wall-normal ray through O. No later field, no target direction and no outcome enters it"
    ),
    "continuity": (
        f"the tracked root may move by at most {ROOT_MATCH_MAX_STEP_DX} dx between consecutive "
        "measurements; a change of the grid edge that carries the crossing is an allowed, recorded event "
        "(segment_index change), while a change of the connected contour branch is a failure"
    ),
    "root_matching": (
        f"inside the continuation window of +/- {RAY_SEARCH_HALF_WINDOW_DX} dx around the previous root "
        "the tracked root is the one nearest that previous root; matching uses proximity only and never "
        "the direction or the magnitude of the implied motion"
    ),
    "ambiguity": (
        f"two roots closer than {ROOT_AMBIGUITY_SEPARATION_DX} dx, a saddle cell at the matched point "
        f"whose asymptotic decider value is within {SADDLE_AMBIGUITY_TOLERANCE} of the threshold, or a "
        "junction where more than two traced segments meet make the step ambiguous; ambiguity is a "
        "property of the reconstruction and is tested before the continuity bound, so an inseparable pair "
        "is reported as ambiguous rather than as a step"
    ),
    "failure": (
        f"no root inside the continuation window, a root step beyond {ROOT_MATCH_MAX_STEP_DX} dx, an arc "
        "that is not a simple open path through the matched point, or an arc segment farther than "
        f"{ARC_ROOT_MATCH_TOLERANCE_DX} dx from the matched root are failures; ambiguity and failure both "
        "withhold a verdict and are never resolved by preferring a candidate"
    ),
    "no_favourable_selection": (
        "the readout is direction-blind: mirroring the wall normal mirrors the reported displacement "
        "exactly, and a nearer root is always preferred to a farther one even when the farther root would "
        "move the interface toward the solid"
    ),
    "cahn_hilliard_pairing": (
        "the explicit chemical-potential flux and the implicit solve increment are two halves of one "
        "Cahn-Hilliard update; they are reported separately as ingredients and every physical statement is "
        "made about their sum, never about either half as an independent mechanism"
    ),
    "why_branch_following": (
        "the readout follows the frozen linear reconstruction across adjacent grid edges instead of "
        "pinning one grid edge, because the frozen event rule reconstructs the contour as linear crossings "
        "on grid edges and a physical interface is not attached to a grid edge"
    ),
}


def grid_spec(N: int = 64, L: float = 6.0, *, eps_over_dx: float = 2.0) -> SimpleNamespace:
    """A minimal grid specification for synthetic studies (no solver, no phasefield import)."""
    spacing = float(L) / int(N)
    return SimpleNamespace(
        Nx=int(N),
        Ny=int(N),
        dx=spacing,
        dy=spacing,
        Lx=float(L),
        Ly=float(L),
        eps=eps_over_dx * spacing,
    )


# ---------------------------------------------------------------------------
# frozen reconstruction convention (re-implemented, then compared against the frozen module)
# ---------------------------------------------------------------------------
def bilinear_sample(field: Any, x: Any, y: Any, p: Any) -> np.ndarray:
    """Periodic bilinear sample of a cell-centred field, identical in convention to the frozen one."""
    values = np.asarray(field, dtype=np.float64)
    Nx, Ny = int(p.Nx), int(p.Ny)
    fx = np.asarray(x, dtype=np.float64) / float(p.dx) - 0.5
    fy = np.asarray(y, dtype=np.float64) / float(p.dy) - 0.5
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    tx = fx - i0
    ty = fy - j0
    i0 %= Nx
    j0 %= Ny
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    return (
        values[i0, j0] * (1.0 - tx) * (1.0 - ty)
        + values[i1, j0] * tx * (1.0 - ty)
        + values[i0, j1] * (1.0 - tx) * ty
        + values[i1, j1] * tx * ty
    )


def scalar_sample(field: Any, x: float, y: float, p: Any) -> float:
    """Single-point bilinear sample as a Python float."""
    return float(np.asarray(bilinear_sample(field, x, y, p), dtype=np.float64).reshape(-1)[0])


def frozen_column_crossings(phi: Any, i: int, p: Any, threshold: float = THRESHOLD) -> list[dict[str, Any]]:
    """The frozen linear ``phi = threshold`` crossings on the vertical segments of one column.

    A vertical segment joins the cells ``(i, j)`` and ``(i, j + 1)`` at the fixed abscissa
    ``(i + 0.5) dx``, which is exactly the frozen reconstruction restricted to that column. A crossing
    that lands on a shared cell centre (``phi == threshold`` exactly) is reported once and flagged.
    """
    values = np.asarray(phi, dtype=np.float64)[int(i), :]
    Ny = int(p.Ny)
    roots: list[dict[str, Any]] = []
    for j in range(Ny):
        jn = (j + 1) % Ny
        a = float(values[j])
        b = float(values[jn])
        if a == threshold:
            roots.append(
                {
                    "y_over_dx": (j + 0.5),
                    "segment_index": j,
                    "fraction": 0.0,
                    "at_shared_node": True,
                }
            )
            continue
        if (a - threshold) * (b - threshold) < 0.0:
            fraction = (threshold - a) / (b - a)
            roots.append(
                {
                    "y_over_dx": ((j + 0.5 + fraction) % float(Ny)),
                    "segment_index": j,
                    "fraction": fraction,
                    "at_shared_node": False,
                }
            )
    roots.sort(key=lambda item: item["y_over_dx"])
    return roots


def scan_ray_crossings(
    phi: Any,
    origin: tuple[float, float],
    direction: tuple[float, float],
    p: Any,
    *,
    center: float,
    half_window: float = RAY_SEARCH_HALF_WINDOW_DX,
    scan_step: float = RAY_SCAN_STEP_DX,
    threshold: float = THRESHOLD,
) -> list[dict[str, Any]]:
    """Bilinear roots of ``phi = threshold`` along a general straight ray, restricted to a window.

    ``origin`` is a point in length units and ``direction`` a unit vector; ``center`` and ``half_window``
    are offsets along the ray in ``dx`` units. Sign changes are bisected to ``RAY_ROOT_TOLERANCE_DX``; an
    exact zero sample is reported as a shared-node root.
    """
    dx = float(p.dx)
    step = float(scan_step) * dx
    lo = (float(center) - float(half_window)) * dx
    hi = (float(center) + float(half_window)) * dx
    count = max(int(round((hi - lo) / step)), 1)
    offsets = np.linspace(lo, hi, count + 1)
    samples = np.array(
        [
            scalar_sample(phi, origin[0] + off * direction[0], origin[1] + off * direction[1], p) - threshold
            for off in offsets
        ]
    )
    roots: list[dict[str, Any]] = []
    for index in np.where(samples == 0.0)[0]:
        roots.append(
            {
                "s_over_dx": float(offsets[index]) / dx,
                "at_shared_node": True,
                "bracket_over_dx": [float(offsets[index]) / dx, float(offsets[index]) / dx],
            }
        )
    for index in np.where((samples[:-1] * samples[1:]) < 0.0)[0]:
        left = float(offsets[index])
        right = float(offsets[index + 1])
        f_left = float(samples[index])
        for _ in range(200):
            mid = 0.5 * (left + right)
            f_mid = scalar_sample(phi, origin[0] + mid * direction[0], origin[1] + mid * direction[1], p) - threshold
            if (f_left * f_mid) <= 0.0:
                right = mid
            else:
                left, f_left = mid, f_mid
            if (right - left) <= RAY_ROOT_TOLERANCE_DX * dx:
                break
        roots.append(
            {
                "s_over_dx": 0.5 * (left + right) / dx,
                "at_shared_node": False,
                "bracket_over_dx": [offsets[index] / dx, offsets[index + 1] / dx],
            }
        )
    roots.sort(key=lambda item: item["s_over_dx"])
    merged: list[dict[str, Any]] = []
    for root in roots:
        if merged and abs(root["s_over_dx"] - merged[-1]["s_over_dx"]) <= NODE_DEGENERACY_TOLERANCE_DX:
            merged[-1]["at_shared_node"] = bool(merged[-1]["at_shared_node"] and root["at_shared_node"])
            continue
        merged.append(root)
    return merged


def match_branch_root(candidates: list[dict[str, Any]], previous: float) -> dict[str, Any]:
    """Pick the tracked root by proximity to the previous root and report every ambiguity flag."""
    if not candidates:
        return {"status": "NO_ROOT_IN_WINDOW", "matched": None, "reason": "no reconstructed root inside the window"}
    ordered = sorted(candidates, key=lambda item: abs(float(item["s_over_dx"]) - float(previous)))
    matched = ordered[0]
    step = abs(float(matched["s_over_dx"]) - float(previous))
    second = ordered[1] if len(ordered) > 1 else None
    separation = None if second is None else abs(float(second["s_over_dx"]) - float(matched["s_over_dx"]))
    if separation is not None and separation < ROOT_AMBIGUITY_SEPARATION_DX:
        return {
            "status": "AMBIGUOUS_MULTIPLE_ROOTS",
            "matched": matched,
            "second": second,
            "separation_over_dx": separation,
            "reason": (
                f"the second reconstructed root is only {separation} dx away, below the ambiguity "
                f"separation {ROOT_AMBIGUITY_SEPARATION_DX} dx"
            ),
        }
    if step > ROOT_MATCH_MAX_STEP_DX:
        return {
            "status": "BRANCH_STEP_TOO_LARGE",
            "matched": matched,
            "step_over_dx": step,
            "reason": f"the nearest root moved {step} dx, beyond the continuity bound {ROOT_MATCH_MAX_STEP_DX} dx",
        }
    return {
        "status": "OK",
        "matched": matched,
        "step_over_dx": step,
        "separation_over_dx": separation,
        "candidates_over_dx": [float(item["s_over_dx"]) for item in ordered],
        "at_shared_node": bool(matched.get("at_shared_node", False)),
    }


# ---------------------------------------------------------------------------
# arc tracing on the same reconstruction
# ---------------------------------------------------------------------------
def _edge_crossing(
    value_a: float,
    value_b: float,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    threshold: float,
) -> tuple[float, float] | None:
    if value_a == threshold:
        return point_a
    if value_b == threshold:
        return point_b
    if (value_a - threshold) * (value_b - threshold) < 0.0:
        fraction = (threshold - value_a) / (value_b - value_a)
        return (
            point_a[0] + fraction * (point_b[0] - point_a[0]),
            point_a[1] + fraction * (point_b[1] - point_a[1]),
        )
    return None


def cell_segments(phi: Any, i: int, j: int, p: Any, threshold: float = THRESHOLD) -> dict[str, Any]:
    """The linear contour segment(s) of one reconstruction cell, with its saddle information.

    The cell spans four cell centres - ``(i, j)``, ``(i + 1, j)``, ``(i, j + 1)``, ``(i + 1, j + 1)`` - so
    its corners sit at ``(i + 0.5, j + 0.5) dx`` and so on. A cell with four crossings is ambiguous and is
    resolved by the asymptotic decider (the bilinear value at the cell centre), which is recorded.
    """
    values = np.asarray(phi, dtype=np.float64)
    Nx, Ny = int(p.Nx), int(p.Ny)
    i_next = (int(i) + 1) % Nx
    j_next = (int(j) + 1) % Ny
    x0 = (int(i) + 0.5) * float(p.dx)
    x1 = x0 + float(p.dx)
    y0 = (int(j) + 0.5) * float(p.dy)
    y1 = y0 + float(p.dy)
    corners = {
        "BL": ((x0, y0), float(values[int(i), int(j)])),
        "BR": ((x1, y0), float(values[i_next, int(j)])),
        "TL": ((x0, y1), float(values[int(i), j_next])),
        "TR": ((x1, y1), float(values[i_next, j_next])),
    }
    crossings: dict[str, tuple[float, float]] = {}
    for name, first, second in (
        ("BC", "BL", "BR"),
        ("TC", "TL", "TR"),
        ("LC", "BL", "TL"),
        ("RC", "BR", "TR"),
    ):
        point = _edge_crossing(corners[first][1], corners[second][1], corners[first][0], corners[second][0], threshold)
        if point is not None:
            crossings[name] = point
    center_value = 0.25 * sum(corner[1] for corner in corners.values())
    saddle = len(crossings) == 4
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if len(crossings) == 2:
        names = sorted(crossings)
        segments.append((crossings[names[0]], crossings[names[1]]))
    elif len(crossings) == 4:
        low_corners = tuple(sorted(name for name, corner in corners.items() if corner[1] < threshold))
        center_high = center_value > threshold
        if low_corners == ("BL", "TR"):
            if center_high:
                segments.extend([(crossings["BC"], crossings["LC"]), (crossings["TC"], crossings["RC"])])
            else:
                segments.extend([(crossings["LC"], crossings["TC"]), (crossings["BC"], crossings["RC"])])
        elif low_corners == ("BR", "TL"):
            if center_high:
                segments.extend([(crossings["BC"], crossings["RC"]), (crossings["LC"], crossings["TC"])])
            else:
                segments.extend([(crossings["TC"], crossings["RC"]), (crossings["BC"], crossings["LC"])])
        else:  # pragma: no cover - defensive: four crossings always split into diagonal pairs
            names = sorted(crossings)
            segments.extend([(crossings[names[0]], crossings[names[1]]), (crossings[names[2]], crossings[names[3]])])
    return {
        "cell": (int(i), int(j)),
        "segments": segments,
        "crossing_count": len(crossings),
        "saddle": saddle,
        "saddle_value": center_value,
        "saddle_ambiguous": bool(saddle and abs(center_value - threshold) <= SADDLE_AMBIGUITY_TOLERANCE),
    }


def window_segments(phi: Any, p: Any, seed: tuple[float, float], radius_dx: float) -> dict[str, Any]:
    """Every reconstruction segment inside a square window of ``+/- radius_dx`` around the seed point."""
    dx = float(p.dx)
    dy = float(p.dy)
    i_center = int(math.floor(seed[0] / dx - 0.5))
    j_center = int(math.floor(seed[1] / dy - 0.5))
    span = int(math.ceil(radius_dx)) + 1
    segments: list[dict[str, Any]] = []
    saddles: list[dict[str, Any]] = []
    for di in range(-span, span + 2):
        for dj in range(-span, span + 2):
            record = cell_segments(phi, i_center + di, j_center + dj, p)
            if record["saddle"]:
                saddles.append(
                    {
                        "cell": record["cell"],
                        "saddle_value": record["saddle_value"],
                        "saddle_ambiguous": record["saddle_ambiguous"],
                    }
                )
            for endpoints in record["segments"]:
                segments.append({"cell": record["cell"], "endpoints": endpoints})
    return {
        "segments": segments,
        "saddle_cells": saddles,
        "seed_cell": (int(i_center), int(j_center)),
        "window_over_dx": {
            "x": [seed[0] / dx - radius_dx, seed[0] / dx + radius_dx],
            "y": [seed[1] / dy - radius_dx, seed[1] / dy + radius_dx],
        },
    }


def _key(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0] * 1.0e9)), int(round(point[1] * 1.0e9)))


def trace_arc(
    phi: Any,
    p: Any,
    seed: tuple[float, float],
    *,
    radius_dx: float = ARC_WINDOW_RADIUS_DX,
    max_segments: int = ARC_MAX_SEGMENTS,
) -> dict[str, Any]:
    """Follow the reconstructed contour arc through the seed point, in both directions.

    The walk starts at the segment nearest the seed point and continues through shared endpoints. It
    reports whether the arc is a simple open path whose two ends leave the window, whether it revisited
    a segment (a loop), how long it is, and whether any point joins more than two segments.
    """
    dx = float(p.dx)
    dy = float(p.dy)
    window = window_segments(phi, p, seed, radius_dx)
    segments = window["segments"]
    if not segments:
        return {
            "status": "NO_ARC",
            "reason": "the window contains no reconstructed contour segment",
            "saddles": window["saddle_cells"],
            "seed_cell": window["seed_cell"],
        }
    nearest = min(
        range(len(segments)),
        key=lambda index: min(
            math.hypot(point[0] - seed[0], point[1] - seed[1]) for point in segments[index]["endpoints"]
        ),
    )
    seed_distance = min(math.hypot(point[0] - seed[0], point[1] - seed[1]) for point in segments[nearest]["endpoints"])
    adjacency: dict[tuple[int, int], list[int]] = {}
    for index, segment in enumerate(segments):
        for point in segment["endpoints"]:
            adjacency.setdefault(_key(point), []).append(index)
    x_lo = seed[0] / dx - radius_dx
    x_hi = seed[0] / dx + radius_dx
    y_lo = seed[1] / dy - radius_dx
    y_hi = seed[1] / dy + radius_dx

    def outside(point: tuple[float, float]) -> bool:
        x = point[0] / dx
        y = point[1] / dy
        tol = ARC_BOUNDARY_TOLERANCE_DX
        return x <= x_lo + tol or x >= x_hi - tol or y <= y_lo + tol or y >= y_hi - tol

    visited = {nearest}
    chain = [nearest]
    ends: list[dict[str, Any]] = []
    revisit = False
    truncated = False
    junctions: list[tuple[int, int]] = []
    for start_endpoint in segments[nearest]["endpoints"]:
        current_segment = nearest
        current_point = start_endpoint
        while True:
            partners = [idx for idx in adjacency.get(_key(current_point), []) if idx != current_segment]
            if len(partners) > 1:
                junctions.append(_key(current_point))
            if not partners:
                ends.append({"reason": "arc_end", "at": current_point})
                break
            partner = partners[0]
            if partner in visited:
                revisit = True
                ends.append({"reason": "revisit", "at": current_point})
                break
            if len(chain) >= max_segments:
                truncated = True
                ends.append({"reason": "max_segments", "at": current_point})
                break
            visited.add(partner)
            chain.append(partner)
            next_point = (
                segments[partner]["endpoints"][1]
                if _key(segments[partner]["endpoints"][0]) == _key(current_point)
                else segments[partner]["endpoints"][0]
            )
            if outside(next_point):
                ends.append({"reason": "window_boundary", "at": next_point})
                break
            current_segment, current_point = partner, next_point
    length = 0.0
    for index in chain:
        first, second = segments[index]["endpoints"]
        length += math.hypot(second[0] - first[0], second[1] - first[1])
    closed_loop = bool(revisit and len(ends) >= 1 and all(end["reason"] == "revisit" for end in ends))
    if closed_loop:
        status = "CLOSED_LOOP"
    elif revisit:
        status = "REVISIT"
    elif truncated:
        status = "TRUNCATED"
    elif junctions:
        status = "AMBIGUOUS_JUNCTION"
    elif len(ends) == 2 and len(chain) >= 1:
        status = "OPEN_PATH"
    else:
        status = "ENDED_INSIDE"
    return {
        "status": status,
        "seed_segment_distance_over_dx": seed_distance / dx,
        "segments_traced": len(chain),
        "arc_length_over_dx": length / dx,
        "ends": ends,
        "revisit": revisit,
        "truncated": truncated,
        "junctions": [list(key) for key in junctions],
        "open_simple_path": status == "OPEN_PATH",
        "saddles": window["saddle_cells"],
        "seed_cell": window["seed_cell"],
    }


# ---------------------------------------------------------------------------
# the readout itself
# ---------------------------------------------------------------------------
def branch_root(
    phi: Any,
    spec: dict[str, Any],
    previous_root: float,
    *,
    ray: str = "wall_normal",
    with_arc: bool = True,
) -> dict[str, Any]:
    """Locate the tracked branch root of one field and audit continuity, ambiguity and arc agreement.

    ``spec`` is the pinned branch definition from :func:`branch_definition`. ``previous_root`` is the root
    of the *base* field of the comparison, in ``dx`` units along the ray.
    """
    grid = spec["grid"]
    if ray == "wall_normal":
        candidates = [
            {"s_over_dx": root["y_over_dx"], **root}
            for root in frozen_column_crossings(phi, spec["column_index"], grid)
            if abs(root["y_over_dx"] - previous_root) <= RAY_SEARCH_HALF_WINDOW_DX
        ]
        matched_record = match_branch_root(candidates, previous_root)
        coordinates = None
        if matched_record["matched"] is not None:
            x_over_dx = float(spec["column_x_over_dx"])
            y_over_dx = float(matched_record["matched"]["s_over_dx"])
            coordinates = (x_over_dx * float(grid.dx), y_over_dx * float(grid.dy))
            matched_record["matched"] = {
                "s_over_dx": y_over_dx,
                "segment_index": matched_record["matched"].get("segment_index"),
                "fraction": matched_record["matched"].get("fraction"),
                "at_shared_node": matched_record["matched"].get("at_shared_node"),
                "x_over_dx": x_over_dx,
                "y_over_dx": y_over_dx,
            }
    elif ray == "interface_normal":
        origin = spec["origin"]
        direction = spec["interface_normal"]
        matched_record = match_branch_root(
            scan_ray_crossings(phi, origin, direction, grid, center=previous_root), previous_root
        )
        coordinates = None
        if matched_record["matched"] is not None:
            offset = float(matched_record["matched"]["s_over_dx"]) * float(grid.dx)
            coordinates = (
                float(origin[0]) + offset * float(direction[0]),
                float(origin[1]) + offset * float(direction[1]),
            )
            matched_record["matched"] = {
                "s_over_dx": float(matched_record["matched"]["s_over_dx"]),
                "segment_index": None,
                "fraction": None,
                "at_shared_node": bool(matched_record["matched"].get("at_shared_node", False)),
                "x_over_dx": coordinates[0] / float(grid.dx),
                "y_over_dx": coordinates[1] / float(grid.dy),
            }
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown ray {ray!r}")
    record: dict[str, Any] = {
        "ray": ray,
        "status": matched_record["status"],
        "matched": matched_record.get("matched"),
        "step_over_dx": matched_record.get("step_over_dx"),
        "separation_over_dx": matched_record.get("separation_over_dx"),
        "candidates_over_dx": matched_record.get("candidates_over_dx"),
        "reason": matched_record.get("reason"),
        "coordinates": coordinates,
    }
    if coordinates is not None:
        cell_i = int(math.floor(coordinates[0] / float(grid.dx) - 0.5))
        cell_j = int(math.floor(coordinates[1] / float(grid.dy) - 0.5))
        matched_cell = cell_segments(phi, cell_i, cell_j, grid)
        record["matched_cell"] = {
            "cell": [cell_i, cell_j],
            "saddle": matched_cell["saddle"],
            "saddle_value": matched_cell["saddle_value"],
            "saddle_ambiguous": matched_cell["saddle_ambiguous"],
        }
        if record["status"] == "OK" and matched_cell["saddle_ambiguous"]:
            record["status"] = "SADDLE_AMBIGUOUS"
            record["reason"] = "the matched cell demands an asymptotic decider within the ambiguity tolerance"
    if with_arc and coordinates is not None:
        arc = trace_arc(phi, grid, coordinates)
        record["arc"] = arc
        if record["status"] == "OK" and not arc.get("open_simple_path", False):
            record["status"] = "ARC_NOT_OPEN"
            record["reason"] = f"the traced arc is {arc['status']}, not a simple open path"
        elif record["status"] == "OK" and arc["seed_segment_distance_over_dx"] > ARC_ROOT_MATCH_TOLERANCE_DX:
            record["status"] = "ARC_ROOT_MISMATCH"
            record["reason"] = (
                f"the nearest arc segment is {arc['seed_segment_distance_over_dx']} dx from the matched "
                f"root, beyond {ARC_ROOT_MATCH_TOLERANCE_DX} dx"
            )
    return record


def branch_definition(p: Any, phi0: Any, support: dict[str, Any]) -> dict[str, Any]:
    """Pin the tracked branch once, from the ``t = 0`` state and the frozen event support only.

    ``p`` is the grid specification, ``phi0`` the ``t = 0`` phase field and ``support`` the frozen global
    ``phi = 0.5`` minimum-gap support record (point, wall normal, interface normal, gradient norm).
    """
    origin = tuple(float(value) for value in support["point"])
    wall_normal = tuple(float(value) for value in support["wall_normal_into_solid"])
    interface_normal = tuple(float(value) for value in support["interface_normal"])
    column_index = int(math.floor(origin[0] / float(p.dx)))
    previous = origin[1] / float(p.dy)
    candidates = [
        {"s_over_dx": root["y_over_dx"], **root}
        for root in frozen_column_crossings(phi0, column_index, p)
        if abs(root["y_over_dx"] - previous) <= RAY_SEARCH_HALF_WINDOW_DX
    ]
    root_record = match_branch_root(candidates, previous)
    if root_record["status"] != "OK":
        raise ValueError(f"the t = 0 branch definition is not unique: {root_record['status']}")
    return {
        "readout_version": READOUT_VERSION,
        "origin": origin,
        "origin_over_dx": [origin[0] / float(p.dx), origin[1] / float(p.dy)],
        "wall_normal_into_solid": wall_normal,
        "interface_normal": interface_normal,
        "column_index": column_index,
        "column_x_over_dx": column_index + 0.5,
        "t0_root_over_dx": float(root_record["matched"]["s_over_dx"]),
        "t0_root_segment_index": int(root_record["matched"]["segment_index"]),
        "t0_root_at_shared_node": bool(root_record["matched"]["at_shared_node"]),
        "t0_root_separation_over_dx": root_record["separation_over_dx"],
        "gradient_norm_at_origin": float(support["gradient_norm"]),
        "grid": p,
        "rules": PRESPECIFIED_RULES,
        "constants": {
            "RAY_SEARCH_HALF_WINDOW_DX": RAY_SEARCH_HALF_WINDOW_DX,
            "RAY_SCAN_STEP_DX": RAY_SCAN_STEP_DX,
            "RAY_ROOT_TOLERANCE_DX": RAY_ROOT_TOLERANCE_DX,
            "ROOT_MATCH_MAX_STEP_DX": ROOT_MATCH_MAX_STEP_DX,
            "ROOT_AMBIGUITY_SEPARATION_DX": ROOT_AMBIGUITY_SEPARATION_DX,
            "NODE_DEGENERACY_TOLERANCE_DX": NODE_DEGENERACY_TOLERANCE_DX,
            "ARC_WINDOW_RADIUS_DX": ARC_WINDOW_RADIUS_DX,
            "ARC_MAX_SEGMENTS": ARC_MAX_SEGMENTS,
            "ARC_ROOT_MATCH_TOLERANCE_DX": ARC_ROOT_MATCH_TOLERANCE_DX,
            "ARC_BOUNDARY_TOLERANCE_DX": ARC_BOUNDARY_TOLERANCE_DX,
            "SADDLE_AMBIGUITY_TOLERANCE": SADDLE_AMBIGUITY_TOLERANCE,
            "CROSS_RAY_TOLERANCE_DX": CROSS_RAY_TOLERANCE_DX,
            "GRADIENT_FLOOR_RATIO": GRADIENT_FLOOR_RATIO,
            "PROBE_BAND": PROBE_BAND,
            "MAX_SUBSTEP_DISPLACEMENT_OVER_EPS": MAX_SUBSTEP_DISPLACEMENT_OVER_EPS,
        },
    }


def displacement_over_dx(
    root_from: tuple[float, float] | None,
    root_to: tuple[float, float] | None,
    wall_normal: tuple[float, float],
    p: Any,
) -> float | None:
    """Signed displacement of the interface along the wall normal, in ``dx`` units.

    The two roots are given in length units, matching the ``coordinates`` field of a readout record.
    Positive means the interface moved toward the solid, because ``wall_normal`` points into the solid.
    """
    if root_from is None or root_to is None:
        return None
    delta_x = (float(root_to[0]) - float(root_from[0])) / float(p.dx)
    delta_y = (float(root_to[1]) - float(root_from[1])) / float(p.dy)
    return delta_x * float(wall_normal[0]) + delta_y * float(wall_normal[1])


def displacement_over_dx_from_heights(
    root_from_over_dx: float,
    root_to_over_dx: float,
    wall_normal: tuple[float, float],
) -> float:
    """Signed wall-normal displacement in ``dx`` units for two roots quoted as ray offsets in ``dx``."""
    return (float(root_to_over_dx) - float(root_from_over_dx)) * float(wall_normal[1])


def spec_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """Serialisable description of the pinned branch, without the in-memory grid object."""
    return {key: value for key, value in spec.items() if key not in {"grid", "rules", "constants"}} | {
        "rules": dict(spec["rules"]),
        "constants": dict(spec["constants"]),
    }


def design_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """The design record that is hashed into the evidence before any window measurement."""
    payload = {
        "readout_version": READOUT_VERSION,
        "rules": dict(PRESPECIFIED_RULES),
        "constants": dict(spec["constants"]),
        "pinned_branch": {
            key: value
            for key, value in spec_payload(spec).items()
            if key
            in {
                "origin_over_dx",
                "wall_normal_into_solid",
                "interface_normal",
                "column_index",
                "column_x_over_dx",
                "t0_root_over_dx",
                "t0_root_segment_index",
                "t0_root_at_shared_node",
                "t0_root_separation_over_dx",
                "gradient_norm_at_origin",
            }
        },
        "measured_quantity": (
            "the displacement of the tracked phi = 0.5 branch along the fixed wall-normal ray, in dx "
            "units, positive toward the solid"
        ),
        "not_used": [
            "any field later than the one being measured",
            "any target direction, verdict or dominance consideration",
            "any rescaling or normalisation of the displacement by its own magnitude",
        ],
    }
    payload["fingerprint_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# synthetic fields and the phase-1 adversarial validation
# ---------------------------------------------------------------------------
def _grid(phi_shape: tuple[int, int], p: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(
        (np.arange(phi_shape[0]) + 0.5) * float(p.dx), (np.arange(phi_shape[1]) + 0.5) * float(p.dy), indexing="ij"
    )


def diffuse_column_field(p: Any, *, crossing_over_dx: float, eps_over_dx: float = 2.0, tilt: float = 0.0) -> np.ndarray:
    """A diffuse tanh interface crossing the column exactly once (optionally tilted in ``x``)."""
    grid_x, grid_y = _grid((p.Nx, p.Ny), p)
    eps = eps_over_dx * float(p.dx)
    argument = (grid_y - crossing_over_dx * float(p.dy) - tilt * (grid_x - p.Lx / 2.0)) / (2.0 * eps)
    return 0.5 * (1.0 + np.tanh(argument))


def two_root_field(p: Any, *, roots_over_dx: tuple[float, float], steepness: float = 6.0) -> np.ndarray:
    """A field whose column has exactly two ``phi = 0.5`` crossings at the requested heights."""
    grid_y = (np.arange(p.Ny) + 0.5) * float(p.dy)
    first, second = roots_over_dx
    quadratic = (grid_y / float(p.dy) - first) * (grid_y / float(p.dy) - second)
    profile = 0.5 + 0.5 * np.tanh(steepness * quadratic)
    return np.repeat(profile[None, :], int(p.Nx), axis=0)


def piecewise_column_field(p: Any, values_by_index: dict[int, float], *, background: float = 0.9) -> np.ndarray:
    """A column whose cell-centre values are prescribed exactly (for root-rule studies)."""
    profile = np.full((p.Ny,), float(background))
    for index, value in values_by_index.items():
        profile[int(index) % int(p.Ny)] = float(value)
    return np.repeat(profile[None, :], int(p.Nx), axis=0)


def symmetric_droplet_field(
    p: Any, *, center_x: float, center_y: float, radius: float, eps_over_dx: float = 1.5
) -> np.ndarray:
    """A circular diffuse contour concentric with the given centre (mirror-symmetric in ``x``)."""
    grid_x, grid_y = _grid((p.Nx, p.Ny), p)
    radius_field = np.hypot(grid_x - center_x, grid_y - center_y)
    return 0.5 * (1.0 - np.tanh((radius_field - radius) / (2.0 * eps_over_dx * float(p.dx))))


def synthetic_directional_bias_check(p: Any) -> dict[str, Any]:
    """The nearest root must win even when the farther root is the one moving toward the solid."""
    spec = {
        "grid": p,
        "column_index": 8,
        "column_x_over_dx": 8.5,
        "wall_normal_into_solid": (0.0, -1.0),
        "origin": (8.5 * float(p.dx), 16.0 * float(p.dy)),
        "interface_normal": (0.0, 1.0),
    }
    # roots at 16.10 dx (0.15 away from the previous root, moving away from the solid) and at 15.30 dx
    # (0.65 away, moving toward the solid - the branch a verdict-favouring rule would prefer)
    field = two_root_field(p, roots_over_dx=(15.30, 16.10))
    record = branch_root(field, spec, 15.95, with_arc=False)
    matched_y = record["matched"]["y_over_dx"] if record["matched"] is not None else None
    mirrored = dict(spec)
    mirrored["wall_normal_into_solid"] = (0.0, 1.0)
    mirrored_record = branch_root(field, mirrored, 15.95, with_arc=False)
    mirrored_y = mirrored_record["matched"]["y_over_dx"] if mirrored_record["matched"] is not None else None
    return {
        "candidate_roots_over_dx": [15.30, 16.10],
        "previous_root_over_dx": 15.95,
        "matched_root_over_dx": matched_y,
        "prefers_the_nearer_root": bool(matched_y is not None and abs(matched_y - 16.10) < abs(matched_y - 15.30)),
        "nearer_root_moves_away_from_the_solid": bool(matched_y is not None and matched_y > 15.95),
        "mirrored_root_over_dx": mirrored_y,
        "mirrored_wall_normal_gives_the_same_root": bool(mirrored_y is not None and mirrored_y == matched_y),
        "status": record["status"],
    }


def synthetic_validation_report() -> dict[str, Any]:
    """The phase-1 adversarial suite: every prespecified rule is exercised on synthetic fields.

    No solver is called and no frozen evidence is read; the fields are analytic constructions on a small
    synthetic grid, so the suite is cheap enough to run inside the test file and again inside the stage.
    """
    p = grid_spec(N=64, L=6.0)
    spec = {
        "grid": p,
        "column_index": 8,
        "column_x_over_dx": 8.5,
        "wall_normal_into_solid": (0.0, -1.0),
        "origin": (8.5 * float(p.dx), 16.0 * float(p.dy)),
        "interface_normal": (0.0, 1.0),
    }
    base = 16.0
    checks: dict[str, Any] = {}

    # 1. known translation, both directions, plus the sign convention in both orientations
    translation_rows = []
    for offset in (-0.1, 0.1, -0.4, 0.4):
        steps = 4 if abs(offset) > 0.11 else 1
        previous_root = base
        measured_total = 0.0
        statuses = []
        for _ in range(steps):
            applied_step = offset / steps
            previous_root += applied_step
            field = diffuse_column_field(p, crossing_over_dx=previous_root)
            record = branch_root(field, spec, previous_root - applied_step, with_arc=False)
            statuses.append(record["status"])
            if record["matched"] is not None:
                measured_total += float(record["matched"]["y_over_dx"]) - (previous_root - applied_step)
        field = diffuse_column_field(p, crossing_over_dx=base + offset)
        record = branch_root(field, spec, base + offset - offset, with_arc=False)
        recovered = None if record["matched"] is None else float(record["matched"]["y_over_dx"]) - base
        mirrored = dict(spec)
        mirrored["wall_normal_into_solid"] = (0.0, 1.0)
        mirrored_record = branch_root(field, mirrored, base, with_arc=False)
        mirrored_displacement = displacement_over_dx(
            (8.5 * float(p.dx), base * float(p.dy)),
            None
            if mirrored_record["matched"] is None
            else (8.5 * float(p.dx), mirrored_record["matched"]["y_over_dx"] * float(p.dy)),
            mirrored["wall_normal_into_solid"],
            p,
        )
        forward_displacement = displacement_over_dx(
            (8.5 * float(p.dx), base * float(p.dy)),
            None if record["matched"] is None else (8.5 * float(p.dx), record["matched"]["y_over_dx"] * float(p.dy)),
            spec["wall_normal_into_solid"],
            p,
        )
        translation_rows.append(
            {
                "applied_offset_over_dx": offset,
                "steps_walked": steps,
                "walk_statuses": statuses,
                "walked_displacement_over_dx": measured_total,
                "walk_error_over_dx": abs(measured_total - offset),
                "recovered_offset_over_dx": recovered,
                "error_over_dx": None if recovered is None else abs(recovered - offset),
                "displacement_toward_solid_over_dx": forward_displacement,
                "mirrored_displacement_over_dx": mirrored_displacement,
                "mirror_is_exact": (
                    forward_displacement is not None
                    and mirrored_displacement is not None
                    and abs(forward_displacement + mirrored_displacement) <= 1.0e-12
                ),
                "status": record["status"],
            }
        )
    checks["linear_translation_is_recovered"] = {
        "rows": translation_rows,
        "max_abs_error_over_dx": max(row["error_over_dx"] for row in translation_rows),
        "note": (
            "the frozen linear reconstruction locates a smooth diffuse crossing with a small "
            "position-dependent bias of order (dx / eps)^2; the study below measures that bias for the "
            "diffuseness of this stage instead of assuming it"
        ),
        "max_walk_error_over_dx": max(row["walk_error_over_dx"] for row in translation_rows),
        "holds": all(
            row["recovered_offset_over_dx"] is not None
            and row["error_over_dx"] <= 3.0e-3
            and row["walk_error_over_dx"] <= 1.0e-2
            and row["mirror_is_exact"]
            and all(status == "OK" for status in row["walk_statuses"])
            for row in translation_rows
        ),
    }

    # 1b. reconstruction bias study at the diffuseness of this stage, swept over the expected motion
    bias_rows = []
    for index in range(0, 41):
        applied = base + 0.002 * index
        field = diffuse_column_field(p, crossing_over_dx=applied, eps_over_dx=1.5)
        record = branch_root(field, spec, applied, with_arc=False)
        if record["matched"] is not None:
            bias_rows.append(
                {
                    "applied_over_dx": applied,
                    "bias_over_dx": float(record["matched"]["y_over_dx"]) - applied,
                }
            )
    biases = [row["bias_over_dx"] for row in bias_rows]
    checks["reconstruction_bias_is_measured_not_assumed"] = {
        "diffuseness_eps_over_dx": 1.5,
        "sweep_over_dx": 0.08,
        "max_abs_bias_over_dx": max(abs(value) for value in biases),
        "bias_variation_over_the_sweep_over_dx": max(biases) - min(biases),
        "rows": bias_rows,
        "holds": bool(biases)
        and (max(biases) - min(biases)) <= 5.0e-3
        and max(abs(value) for value in biases) <= 5.0e-3,
    }

    # 2. crossing a shared grid node: continuous, recorded, and the supporting segment switches
    frames = [16.35 + 0.05 * index for index in range(7)]  # straddles the shared node at 16.5 dx
    node_rows = []
    previous = frames[0]
    segment_indices = []
    for crossing in frames[1:]:
        field = diffuse_column_field(p, crossing_over_dx=crossing)
        record = branch_root(field, spec, previous, with_arc=False)
        if record["matched"] is not None:
            node_rows.append(
                {
                    "crossing_over_dx": crossing,
                    "matched_y_over_dx": record["matched"]["y_over_dx"],
                    "step_over_dx": float(record["matched"]["y_over_dx"]) - previous,
                    "segment_index": record["matched"]["segment_index"],
                    "at_shared_node": record["matched"]["at_shared_node"],
                    "status": record["status"],
                }
            )
            segment_indices.append(int(record["matched"]["segment_index"]))
            previous = float(record["matched"]["y_over_dx"])
    steps = [abs(row["step_over_dx"]) for row in node_rows]
    checks["continuity_across_a_shared_grid_node"] = {
        "rows": node_rows,
        "segment_indices": segment_indices,
        "support_switch_observed": len(set(segment_indices)) > 1,
        "max_step_over_dx": max(steps) if steps else None,
        "holds": bool(steps)
        and all(row["status"] == "OK" for row in node_rows)
        and max(steps) <= 0.06
        and len(set(segment_indices)) > 1,
    }

    # 3. two roots: refused when the reconstruction cannot separate them, tracked when it can
    close_field = piecewise_column_field(p, {15: 0.9, 16: 0.49, 17: 0.9})
    close_record = branch_root(close_field, spec, base, with_arc=False)
    far_field = piecewise_column_field(p, {15: 0.9, 16: 0.2, 17: 0.9})
    far_record = branch_root(far_field, spec, base + 0.07, with_arc=False)
    checks["double_roots_are_refused_or_separated"] = {
        "close_status": close_record["status"],
        "close_separation_over_dx": close_record.get("separation_over_dx"),
        "far_status": far_record["status"],
        "far_separation_over_dx": far_record.get("separation_over_dx"),
        "refuses_when_close": close_record["status"] == "AMBIGUOUS_MULTIPLE_ROOTS",
        "separates_when_far": far_record["status"] == "OK",
    }
    checks["double_roots_are_refused_or_separated"]["holds"] = bool(
        checks["double_roots_are_refused_or_separated"]["refuses_when_close"]
        and checks["double_roots_are_refused_or_separated"]["separates_when_far"]
    )

    # 4. symmetric contour: the root sits on the mirror axis, both orientations agree, arc is open
    droplet_spec = {
        "grid": p,
        "column_index": 16,
        "column_x_over_dx": 16.5,
        "wall_normal_into_solid": (0.0, -1.0),
        "origin": (16.5 * float(p.dx), 38.0 * float(p.dy)),
        "interface_normal": (0.0, 1.0),
    }
    droplet = symmetric_droplet_field(
        p, center_x=16.5 * float(p.dx), center_y=38.0 * float(p.dy), radius=3.0 * float(p.dx)
    )
    droplet_record = branch_root(droplet, droplet_spec, 35.0, with_arc=True)
    mirrored_spec = dict(droplet_spec)
    mirrored_spec["wall_normal_into_solid"] = (0.0, 1.0)
    mirrored_record = branch_root(droplet, mirrored_spec, 35.0, with_arc=True)
    droplet_root = None if droplet_record["matched"] is None else droplet_record["matched"]["y_over_dx"]
    mirrored_root = None if mirrored_record["matched"] is None else mirrored_record["matched"]["y_over_dx"]
    checks["symmetric_contour_mirrors_exactly"] = {
        "root_over_dx": droplet_root,
        "expected_root_over_dx": 35.0,
        "mirrored_root_over_dx": mirrored_root,
        "arc_status": None if droplet_record.get("arc") is None else droplet_record["arc"]["status"],
        "mirror_identical": bool(droplet_root is not None and droplet_root == mirrored_root),
        "root_on_the_axis": bool(droplet_root is not None and abs(droplet_root - 35.0) <= 1.0e-9),
        "arc_is_an_open_path": bool(droplet_record.get("arc", {}).get("open_simple_path", False)),
    }
    checks["symmetric_contour_mirrors_exactly"]["holds"] = bool(
        checks["symmetric_contour_mirrors_exactly"]["mirror_identical"]
        and checks["symmetric_contour_mirrors_exactly"]["root_on_the_axis"]
        and checks["symmetric_contour_mirrors_exactly"]["arc_is_an_open_path"]
    )

    # 5. branch disappearance, the window bound, and a root beyond the continuity bound
    monotone = np.full((p.Nx, p.Ny), 1.0)
    empty_record = branch_root(monotone, spec, base, with_arc=False)
    near_field = diffuse_column_field(p, crossing_over_dx=base + 0.2)
    near_record = branch_root(near_field, spec, base, with_arc=False)
    jump_field = diffuse_column_field(p, crossing_over_dx=base + 0.3)
    jump_record = branch_root(jump_field, spec, base, with_arc=False)
    outside_field = diffuse_column_field(p, crossing_over_dx=base + 1.6)
    outside_record = branch_root(outside_field, spec, base, with_arc=False)
    outside_candidates = scan_ray_crossings(outside_field, spec["origin"], spec["interface_normal"], p, center=base)
    checks["branch_disappearance_is_a_failure"] = {
        "monotone_status": empty_record["status"],
        "near_status": near_record["status"],
        "jump_status": jump_record["status"],
        "outside_status": outside_record["status"],
        "candidates_outside_the_window": len(outside_candidates),
        "refuses_a_vanished_branch": empty_record["status"] == "NO_ROOT_IN_WINDOW",
        "accepts_a_root_within_the_continuity_bound": near_record["status"] == "OK",
        "refuses_a_root_beyond_the_continuity_bound": jump_record["status"] == "BRANCH_STEP_TOO_LARGE",
        "refuses_a_root_outside_the_search_window": outside_record["status"] == "NO_ROOT_IN_WINDOW",
    }
    checks["branch_disappearance_is_a_failure"]["holds"] = bool(
        checks["branch_disappearance_is_a_failure"]["refuses_a_vanished_branch"]
        and checks["branch_disappearance_is_a_failure"]["accepts_a_root_within_the_continuity_bound"]
        and checks["branch_disappearance_is_a_failure"]["refuses_a_root_beyond_the_continuity_bound"]
        and checks["branch_disappearance_is_a_failure"]["refuses_a_root_outside_the_search_window"]
    )

    # 6. directional blindness
    bias = synthetic_directional_bias_check(p)
    checks["no_favourable_selection"] = {
        **bias,
        "holds": bool(bias["prefers_the_nearer_root"] and bias["mirrored_wall_normal_gives_the_same_root"]),
    }

    # 7. saddle handling: a four-crossing cell is detected, resolved and reported
    saddle_field = np.zeros((p.Nx, p.Ny))
    grid_x, grid_y = _grid((p.Nx, p.Ny), p)
    saddle_field[:, :] = 0.5 + 2.0 * (grid_x - 8.5 * float(p.dx)) * (grid_y - 16.5 * float(p.dy)) / (float(p.dx) ** 2)
    saddle_record = cell_segments(saddle_field, 8, 16, p)
    checks["saddle_cells_are_detected"] = {
        "crossing_count": saddle_record["crossing_count"],
        "saddle": saddle_record["saddle"],
        "segment_count": len(saddle_record["segments"]),
        "saddle_value": saddle_record["saddle_value"],
        "holds": bool(saddle_record["saddle"] and len(saddle_record["segments"]) == 2),
    }

    # 8. the reconstruction is the frozen one: a column root must equal the frozen edge crossing
    frozen_column = np.zeros((p.Nx, p.Ny))
    frozen_column[:, :] = diffuse_column_field(p, crossing_over_dx=base + 0.37)
    roots = [
        root
        for root in frozen_column_crossings(frozen_column, 8, p)
        if abs(root["y_over_dx"] - (base + 0.37)) <= RAY_SEARCH_HALF_WINDOW_DX
    ]
    checks["column_root_is_a_frozen_segment_crossing"] = {
        "roots_in_window_over_dx": [round(root["y_over_dx"], 9) for root in roots],
        "fractions": [root["fraction"] for root in roots],
        "segment_indices": [root["segment_index"] for root in roots],
        "applied_over_dx": base + 0.37,
        "holds": len(roots) == 1
        and abs(roots[0]["y_over_dx"] - (base + 0.37)) <= 3.0e-3
        and roots[0]["segment_index"] == 15
        and roots[0]["at_shared_node"] is False,
    }

    return {
        "readout_version": READOUT_VERSION,
        "grid": {"Nx": int(p.Nx), "Ny": int(p.Ny), "dx": float(p.dx), "Lx": float(p.Lx)},
        "cases": checks,
        "all_hold": all(bool(check.get("holds", False)) for check in checks.values()),
    }
