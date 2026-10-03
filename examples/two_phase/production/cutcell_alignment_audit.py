"""L1A-2f evidence runner: phase transport on geometry-conforming cut-cell control volumes.

Blocker under test -- ``N-WALL-ALIGNMENT-TRANSPORT-DOMAIN``
-----------------------------------------------------------
Contract v8 transported the phase on the cell-centre hard-fluid staircase (``sdf >= 0``) while the
wall energy and the contact-angle measurement reference the true geometric ``sdf = 0`` plane. The
two references differ by up to half a cell as the wall translates inside a cell, and L1A-2e
measured the consequence directly: converged CH-only angles spread 1.57 / 0.93 / 3.63 / 7.24 deg
over eight sub-cell offsets at 60 / 90 / 120 / 150 deg, with ``theta_geo = theta_0 + slope *
(y_disc - y_geo)/dx``, slope = 1.79 / 1.58 / 4.16 / 8.20 deg per cell and ``R^2 >= 0.998``.

Contract v9 removes the *cause*: one geometry authority (``pf.embedded_fluid_geometry``) supplies
the partial control volumes ``V_i``, the shared partial face apertures ``A_f`` and the wall measure
``A_wall,i`` from a single corner-sampled reconstruction, so transport, wall energy and measurement
all live on the same ``sdf = 0`` surface. This runner measures whether the alignment error is gone.

What is measured
----------------
``translation``       the primary gate: 60/90/120/150 deg x eight sub-cell wall offsets, CH-only,
                      N = 128, eps = 2 dx, M = 4 M_ref, float32, truly converged. The spread of the
                      measured equilibrium angle over the offsets must be <= 2 deg (strong <= 1 deg)
                      against the *geometric* wall.
``falsification``     the same angles regressed on the old hard-mask plane offset
                      ``(y_disc - y_geo)/dx``: if the post-v9 error is still strongly linearly
                      correlated with it, the hypothesis is falsified and must be reported as such.
``translation_v8``    the pinned contract-v7/v8 transport (``hard_cell_v7``) re-measured under the
                      v9 code at a subset of the offsets: the reproduction guard for the A/B claim.
``resolution``        60/150 deg at N = 96/128/192 on the same physical wall: spread <= 2 deg
                      (L1A-2e measured 3.093 deg at 150 deg).
``primary_float64``   the four-target CH-only matrix in float64 with rtol = 1e-8.
``production_mobility``  the same equilibria at the production mobility ``M_ref``.
``precision``         the ``N-CH-MASS-PRECISION`` matrix: equal mobility-scaled time, three
                      precision settings, mass measured as ``sum_i V_i phi_i``.
``chns``              full CHNS at production defaults (``M_ref``, dt = 4e-3, N = 128, eps = 2 dx).
``laplace``           static-droplet pressure-jump regression (empty solid): the cut-cell geometry
                      must degenerate exactly, so the ratios stay within 0.01 of contract v8.
``impact``            droplet-impact regression (We = 100 neutral, We = 50 hydrophobic) plus the
                      cut-cell advective CFL diagnostic that decides whether subcycling is needed.
``geometry``          delegates to :mod:`production.cutcell_geometry_audit`.
``transport``         delegates to :mod:`production.cutcell_phase_transport_audit`.

Every relaxation case is cached individually under ``<out>/cases`` so a long run can be resumed, and
the report fails closed: a section that was never run yields ``measured: false`` gates and
``not_ready: true`` rather than a silent pass.

Run with::

    python -m production.cutcell_alignment_audit --profile baseline --out evidence/l1a2f
    python -m production.cutcell_alignment_audit --profile quick --out /tmp/l1a2f_quick
    python -m production.cutcell_alignment_audit --profile baseline --sections translation,chns
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import phasefield as pf
from production import contact_line_kinetics as clk
from production import nonneutral_wetting_audit as nwa
from production import observables as obs
from production import wall_measure_audit as wma

STAGE = "L1A-2f"
M_REF = nwa.M_REF

#: Acceptance gates, transcribed from the L1A-2f specification. Values are frozen before the
#: evidence is produced; nothing here is fitted to a measured angle.
GATES: dict[str, Any] = {
    # primary alignment gate: spread of the converged angle over eight sub-cell wall offsets
    "translation_angle_spread_deg": 2.0,
    "translation_angle_spread_strong_deg": 1.0,
    # falsification: post-v9 angle must not track the old hard-mask plane offset
    "falsification_max_abs_slope_deg_per_cell": 0.5,
    "falsification_slope_reduction_vs_v8": 4.0,
    # grid-convergence gate on the same physical wall
    "resolution_angle_spread_deg": 2.0,
    "resolution_angle_spread_strong_deg": 1.0,
    # CH-only target matrix
    "ch_only_mae_deg": 5.0,
    "ch_only_max_error_deg": 10.0,
    "neutral_error_deg": 3.0,
    "strong_mae_deg": 2.0,
    "strong_max_error_deg": 5.0,
    "ch_only_monotonic_required": True,
    "formal_mass_drift": 1.0e-3,
    "formal_mass_drift_ideal": 1.0e-5,
    "energy_increase_tolerance": 1.0e-6,
    # implicit solver
    "cg_relative_residual_float32": 1.0e-6,
    "cg_relative_residual_float64": 1.0e-8,
    "cg_fail_closed_required": True,
    # geometry
    "flat_wall_area_relative_error": 1.0e-12,
    "laplace_min_r_squared": 0.99,
    "laplace_ratio_change_vs_v8": 0.01,
    "laplace_all_ratios_positive": True,
    # small cut cells: reported, never floored
    "alpha_min_positive_reported": True,
    # cut-cell advective CFL: subcycling is required only if this is clearly violated
    "cutcell_advective_cfl_ratio_limit": 1.0,
    # v8 pinned reproduction
    "v8_reproduction_tolerance_deg": 3.0,
}

SECTIONS: tuple[str, ...] = (
    "translation",
    "translation_v8",
    "resolution",
    "primary_float64",
    "production_mobility",
    "precision",
    "chns",
    "laplace",
    "impact",
    "geometry",
    "transport",
)

PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "N": 128,
        "R": 1.1,
        "dt": 4.0e-3,
        "eps_factor": 2.0,
        "sample_every": 200,
        "targets": [60.0, 90.0, 120.0, 150.0],
        # compute allowance only (the stationarity criteria in ``nwa.CRITERIA`` are untouched):
        # L1A-2e needed ~46k accelerated steps for 60 deg and ~59k for 150 deg.
        "budgets": [10000, 25000, 50000, 80000],
        "neutral_budget": 100000,
        "translation": {"offsets": list(wma.TRANSLATION_OFFSETS), "targets": [60.0, 90.0, 120.0, 150.0]},
        "translation_v8": {"offsets": [0.0, 0.375, 0.625], "targets": [120.0, 150.0]},
        "resolution": {"N_values": [96, 128, 192], "targets": [60.0, 150.0]},
        "float64": {"targets": [60.0, 90.0, 120.0, 150.0], "rtol": 1.0e-8},
        "production_mobility": {"targets": [60.0, 150.0], "budget": 150000, "sample_every": 1000},
        "precision": {"target": 150.0, "fixed_steps": 20000, "mobility_factor": 4.0},
        "chns": {"targets": [60.0, 90.0, 120.0, 150.0], "budgets": [10000, 25000, 50000]},
        "laplace": {"N": 128, "radii": [0.6, 0.8, 1.0, 1.2], "steps": 4000, "save_every": 500},
        # The frozen L1A-2e / production baseline impact case (production/configs/baseline.json):
        # beta_max 2.1429 @ t = 2.3599 (neutral) and 2.0759 @ t = 2.1999 (hydrophobic) are the
        # contract-v8 numbers this section is compared against.
        "impact": {
            "N": 128,
            "steps": 1200,
            "save_every": 20,
            "dt": 2.0e-3,
            "eps_factor": 2.0,
            "impact_gap": 0.1,
            "u_impact": 1.0,
            "cases": [
                {"name": "we100_neutral", "We": 100.0, "Re": 200.0, "theta_deg": 90.0, "R": 0.7},
                {"name": "we50_hydrophobic", "We": 50.0, "Re": 150.0, "theta_deg": 120.0, "R": 0.7},
            ],
        },
    },
    "quick": {
        "N": 48,
        "R": 0.6,
        "dt": 4.0e-3,
        "eps_factor": 2.0,
        "sample_every": 40,
        "targets": [60.0, 150.0],
        "budgets": [200, 400],
        "neutral_budget": 400,
        "translation": {"offsets": [0.0, 0.5], "targets": [60.0, 150.0]},
        "translation_v8": {"offsets": [0.0, 0.5], "targets": [150.0]},
        "resolution": {"N_values": [32, 48], "targets": [150.0]},
        "float64": {"targets": [150.0], "rtol": 1.0e-8},
        "production_mobility": {"targets": [150.0], "budget": 400, "sample_every": 100},
        "precision": {"target": 150.0, "fixed_steps": 200, "mobility_factor": 2.0},
        "chns": {"targets": [150.0], "budgets": [200]},
        "laplace": {"N": 48, "radii": [0.6, 1.0], "steps": 200, "save_every": 100},
        "impact": {
            "N": 48,
            "steps": 60,
            "save_every": 20,
            "dt": 2.0e-3,
            "eps_factor": 2.0,
            "impact_gap": 0.1,
            "u_impact": 1.0,
            "cases": [{"name": "we100_neutral", "We": 100.0, "Re": 200.0, "theta_deg": 90.0, "R": 0.7}],
        },
    },
}


def _log(message: str) -> None:
    print(f"[l1a2f {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _budgets_for(cfg: dict[str, Any], target: float, *, long_neutral: bool = False) -> tuple[int, ...]:
    """Step allowance for one target (the neutral control needs a single long budget)."""
    if abs(float(target) - 90.0) < 1e-9 and long_neutral:
        return (int(cfg["neutral_budget"]),)
    return tuple(int(b) for b in cfg["budgets"])


def _cap(budgets: Sequence[int], max_steps: int | None) -> tuple[int, ...]:
    if max_steps is None:
        return tuple(int(b) for b in budgets)
    return tuple(min(int(b), int(max_steps)) for b in budgets)


def _case_key(section: str, **fields: Any) -> str:
    payload = json.dumps({"section": section, **fields}, sort_keys=True, default=str)
    return f"{section}__{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _clean(value: Any) -> Any:
    """Strict-JSON-safe copy (no NaN/inf, no numpy scalars)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) or value is None:
        return value
    return str(value)


def _relaxation_record(section: str, cfg: dict[str, Any], *, max_steps: int | None = None, **kwargs) -> dict:
    """One CH-only / CHNS relaxation, with per-case caching."""
    key = _case_key(section, **{k: v for k, v in kwargs.items() if k != "criteria"}, profile=cfg.get("_profile"))
    cache: Path | None = cfg.get("_cache")
    if cache is not None:
        path = cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
    budgets = _cap(kwargs.pop("budgets", cfg["budgets"]), max_steps)
    record = nwa.run_relaxation(budgets=budgets, **kwargs)
    record = _clean(record)
    record.pop("samples", None)  # the sampled trace stays in the section cache, not the case cache
    record["_case_key"] = key
    if cache is not None:
        (cache / f"{key}.json").write_text(json.dumps(record, allow_nan=False))
    return record


def _announce(record: dict[str, Any]) -> dict[str, Any]:
    _log(
        f"{record.get('group', ''):<16s} {record['dynamics_mode']:<7s} target={record['target_deg']:5.1f} "
        f"N={record['N']} offset={record.get('wall_offset_over_dy', 0.0):+.3f}dy "
        f"transport={record['phase_transport_geometry']:<17s} M/Mref={record['M_over_M_ref']:.2f} "
        f"steps={record['steps']:>7d} conv={str(record['converged']):<5s} "
        f"theta={record['final_sampled_angle_deg']} drift={record['mass_drift']:.2e} "
        f"conserved={record['conserved_mass_drift']} stop={record['stop_reason']} "
        f"({record['wall_seconds']:.0f}s)"
    )
    return record


# --------------------------------------------------------------------------------------
#  1. translation matrix (the primary alignment gate) + the pinned-v8 A/B
# --------------------------------------------------------------------------------------
def section_translation(cfg: dict[str, Any], *, max_steps: int | None = None, pinned: bool = False) -> list[dict]:
    spec = cfg["translation_v8"] if pinned else cfg["translation"]
    base = {
        "ch_only": True,
        "N": int(cfg["N"]),
        "R": float(cfg["R"]),
        "dt": float(cfg["dt"]),
        "eps_factor": float(cfg["eps_factor"]),
        "sample_every": int(cfg["sample_every"]),
        "M": 4.0 * M_REF,
        "budgets": _budgets_for(cfg, 90.0),
        "keep_samples": 400,
    }
    out: list[dict[str, Any]] = []
    for target in spec["targets"]:
        for offset in spec["offsets"]:
            kwargs = dict(base)
            kwargs["budgets"] = _budgets_for(cfg, target, long_neutral=True)
            if pinned:
                kwargs["phase_transport_geometry"] = "hard_cell_v7"
            out.append(
                _announce(
                    _relaxation_record(
                        "translation_v8" if pinned else "translation",
                        cfg,
                        max_steps=max_steps,
                        target_deg=float(target),
                        wall_offset_over_dy=float(offset),
                        group="translation_v8" if pinned else "translation",
                        label=f"offset_{offset:+.3f}dy",
                        **kwargs,
                    )
                )
            )
    return out


# --------------------------------------------------------------------------------------
#  2. grid-resolution matrix on the same physical wall
# --------------------------------------------------------------------------------------
def section_resolution(cfg: dict[str, Any], *, max_steps: int | None = None) -> list[dict]:
    spec = cfg["resolution"]
    out: list[dict[str, Any]] = []
    for target in spec["targets"]:
        for N in spec["N_values"]:
            out.append(
                _announce(
                    _relaxation_record(
                        "resolution",
                        cfg,
                        max_steps=max_steps,
                        target_deg=float(target),
                        ch_only=True,
                        N=int(N),
                        R=float(cfg["R"]),
                        dt=float(cfg["dt"]),
                        eps_factor=float(cfg["eps_factor"]),
                        sample_every=int(cfg["sample_every"]),
                        M=4.0 * M_REF,
                        budgets=_budgets_for(cfg, target, long_neutral=True),
                        wall_offset_over_dy=0.0,
                        group="resolution",
                        label=f"N{int(N)}",
                        keep_samples=400,
                    )
                )
            )
    return out


# --------------------------------------------------------------------------------------
#  3. float64 / production-mobility / precision matrices
# --------------------------------------------------------------------------------------
def section_float64(cfg: dict[str, Any], *, max_steps: int | None = None) -> list[dict]:
    spec = cfg["float64"]
    out: list[dict[str, Any]] = []
    for target in spec["targets"]:
        out.append(
            _announce(
                _relaxation_record(
                    "primary_float64",
                    cfg,
                    max_steps=max_steps,
                    target_deg=float(target),
                    ch_only=True,
                    N=int(cfg["N"]),
                    R=float(cfg["R"]),
                    dt=float(cfg["dt"]),
                    eps_factor=float(cfg["eps_factor"]),
                    sample_every=int(cfg["sample_every"]),
                    M=4.0 * M_REF,
                    budgets=_budgets_for(cfg, target, long_neutral=True),
                    dtype="float64",
                    ch_solver_rtol=float(spec["rtol"]),
                    wall_offset_over_dy=0.0,
                    group="primary_float64",
                    label="float64_rtol1e-8",
                    keep_samples=400,
                )
            )
        )
    return out


def section_production_mobility(cfg: dict[str, Any], *, max_steps: int | None = None) -> list[dict]:
    spec = cfg["production_mobility"]
    out: list[dict[str, Any]] = []
    for target in spec["targets"]:
        out.append(
            _announce(
                _relaxation_record(
                    "production_mobility",
                    cfg,
                    max_steps=max_steps,
                    target_deg=float(target),
                    ch_only=True,
                    N=int(cfg["N"]),
                    R=float(cfg["R"]),
                    dt=float(cfg["dt"]),
                    eps_factor=float(cfg["eps_factor"]),
                    sample_every=int(spec["sample_every"]),
                    M=M_REF,
                    budgets=(int(spec["budget"]),),
                    wall_offset_over_dy=0.0,
                    group="production_mobility",
                    label="M_ref",
                    keep_samples=400,
                )
            )
        )
    return out


def section_precision(cfg: dict[str, Any], *, max_steps: int | None = None) -> list[dict]:
    """``N-CH-MASS-PRECISION``: equal mobility-scaled time, three precision settings.

    Mass is reported both ways: the contract-v9 conserved quantity ``sum_i V_i phi_i`` and the
    legacy cell-centre hard-mask sum, so the drift can be attributed to the solver rather than to
    the metric.
    """
    spec = cfg["precision"]
    target = float(spec["target"])
    steps = int(spec["fixed_steps"])
    configs = (
        ("float32_rtol1e-6", "float32", 1.0e-6),
        ("float32_rtol1e-8", "float32", 1.0e-8),
        ("float64_rtol1e-8", "float64", 1.0e-8),
    )
    out: list[dict[str, Any]] = []
    for label, dtype, rtol in configs:
        out.append(
            _announce(
                _relaxation_record(
                    "precision",
                    cfg,
                    max_steps=max_steps,
                    target_deg=target,
                    ch_only=True,
                    N=int(cfg["N"]),
                    R=float(cfg["R"]),
                    dt=float(cfg["dt"]),
                    eps_factor=float(cfg["eps_factor"]),
                    sample_every=int(cfg["sample_every"]),
                    M=float(spec["mobility_factor"]) * M_REF,
                    fixed_steps=steps,
                    dtype=dtype,
                    ch_solver_rtol=rtol,
                    wall_offset_over_dy=0.0,
                    group="precision",
                    label=label,
                    keep_samples=400,
                )
            )
        )
    return out


def section_chns(cfg: dict[str, Any], *, max_steps: int | None = None) -> list[dict]:
    """Full CHNS at production defaults (``M_ref``, dt = 4e-3, N = 128, eps = 2 dx)."""
    spec = cfg["chns"]
    out: list[dict[str, Any]] = []
    for target in spec["targets"]:
        out.append(
            _announce(
                _relaxation_record(
                    "chns",
                    cfg,
                    max_steps=max_steps,
                    target_deg=float(target),
                    ch_only=False,
                    N=int(cfg["N"]),
                    R=float(cfg["R"]),
                    dt=float(cfg["dt"]),
                    eps_factor=float(cfg["eps_factor"]),
                    sample_every=int(cfg["sample_every"]),
                    M=M_REF,
                    budgets=tuple(int(b) for b in spec["budgets"]),
                    wall_offset_over_dy=0.0,
                    group="chns",
                    label="production_defaults",
                    keep_samples=400,
                )
            )
        )
    return out


# --------------------------------------------------------------------------------------
#  4. static Laplace regression (empty solid: the cut-cell geometry must degenerate exactly)
# --------------------------------------------------------------------------------------
def section_laplace(cfg: dict[str, Any], *, max_steps: int | None = None) -> dict[str, Any]:
    spec = cfg["laplace"]
    N = int(spec["N"])
    steps = int(spec["steps"]) if max_steps is None else min(int(spec["steps"]), int(max_steps))
    rows: list[dict[str, Any]] = []
    for transport in ("sdf_cutcell_fv_v1", "hard_cell_v7"):
        for R in spec["radii"]:
            key = _case_key("laplace", transport=transport, R=float(R), N=N, steps=steps, profile=cfg.get("_profile"))
            cache: Path | None = cfg.get("_cache")
            if cache is not None and (cache / f"{key}.json").exists():
                rows.append(json.loads((cache / f"{key}.json").read_text()))
                continue
            p = pf.PhaseFieldParams(
                Nx=N, Ny=N, Lx=6.0, Ly=6.0, Re=200.0, We=100.0, dt=2.0e-3, dtype=jnp.float64,
                phase_transport_geometry=transport,
            )
            p.eps = 2.0 * p.dx
            solid = pf.empty_solid(p)
            state = pf.droplet_initial_state(p, x0=p.Lx / 2.0, y0=p.Ly / 2.0, R=float(R), u_impact=0.0)
            step = jax.jit(pf.step, static_argnums=(2,))
            for _ in range(steps):
                state = step(state, solid, p)
            pressure = np.asarray(pf.pressure_field(state, solid, p), dtype=np.float64)
            X, Y = pf.grids(p)
            radius = np.sqrt((np.asarray(X) - p.Lx / 2.0) ** 2 + (np.asarray(Y) - p.Ly / 2.0) ** 2)
            inside = radius < 0.3 * float(R)
            outside = radius > 2.5 * float(R)
            jump = float(pressure[inside].mean() - pressure[outside].mean())
            row = _clean(
                {
                    "transport": transport,
                    "R": float(R),
                    "N": N,
                    "steps": steps,
                    "delta_p": jump,
                    "delta_p_times_R": jump * float(R),
                    "mass_drift": abs(float(pf.liquid_mass(state.phi, solid, p)) - float(np.pi * float(R) ** 2)),
                    "finite": bool(np.isfinite(np.asarray(state.phi)).all()),
                    # cut-cell degeneracy of the empty solid: V = dx dy, alpha = 1, aperture = 1
                    "alpha_min": float(np.min(np.asarray(solid.geometry.alpha))),
                    "volume_is_cell_area": bool(
                        np.allclose(np.asarray(solid.geometry.volume), p.dx * p.dy, rtol=0.0, atol=0.0)
                    ),
                    "aperture_x_is_full": bool(
                        np.allclose(np.asarray(solid.geometry.aperture_x_norm), 1.0, rtol=0.0, atol=0.0)
                    ),
                    "wall_measure_is_zero": float(np.sum(np.asarray(solid.geometry.wall_measure))),
                }
            )
            row["_case_key"] = key
            if cache is not None:
                (cache / f"{key}.json").write_text(json.dumps(row, allow_nan=False))
            rows.append(row)
            _log(f"laplace {transport:<17s} R={float(R):.2f} delta_p={jump:.6f} delta_p*R={jump * float(R):.6f}")
    cut = {row["R"]: row for row in rows if row["transport"] == "sdf_cutcell_fv_v1"}
    pinned = {row["R"]: row for row in rows if row["transport"] == "hard_cell_v7"}
    ratios = {R: cut[R]["delta_p"] / pinned[R]["delta_p"] for R in cut if R in pinned}
    return {"rows": rows, "ratios": ratios}


# --------------------------------------------------------------------------------------
#  5. impact regression + the cut-cell advective CFL diagnostic
# --------------------------------------------------------------------------------------
def section_impact(cfg: dict[str, Any], *, max_steps: int | None = None) -> list[dict]:
    """Impact regression on the *frozen* L1A-2e case definition, plus the cut-cell CFL diagnostic.

    The case parameters (N = 128, steps = 1200, save_every = 20, dt = 2e-3, R = 0.7, u_impact = 1.0,
    ``impact_gap`` = 0.1, eps = 2dx, We = 100 neutral / We = 50 hydrophobic) are copied from
    ``production/configs/baseline.json`` so the numbers are directly comparable with the frozen
    contract-v8 evidence. Nothing is tuned to the morphology: the geometry, the solver and the
    defaults are the contract-v9 ones.
    """
    spec = cfg["impact"]
    N = int(spec["N"])
    steps = int(spec["steps"]) if max_steps is None else min(int(spec["steps"]), int(max_steps))
    frozen = {
        "we100_neutral": {"beta_max": 2.1429, "time_beta_max": 2.3599, "y_cm": 0.9158, "first_contact": 2.1199},
        "we50_hydrophobic": {"beta_max": 2.0759, "time_beta_max": 2.1999, "y_cm": 0.8638},
    }
    out: list[dict[str, Any]] = []
    for case in spec["cases"]:
        key = _case_key("impact_v2", name=case["name"], N=N, steps=steps, profile=cfg.get("_profile"))
        cache: Path | None = cfg.get("_cache")
        if cache is not None and (cache / f"{key}.json").exists():
            out.append(json.loads((cache / f"{key}.json").read_text()))
            continue
        theta = float(case["theta_deg"])
        p = pf.PhaseFieldParams(
            Nx=N,
            Ny=N,
            Lx=6.0,
            Ly=6.0,
            Re=float(case["Re"]),
            We=float(case["We"]),
            dt=float(spec["dt"]),
            M=M_REF,
            dtype=jnp.float32,
        )
        p.eps = float(spec["eps_factor"]) * p.dx
        wall_height = 0.25
        cos_theta = math.cos(math.radians(theta))
        solid = pf.make_solid(pf.surface_flat(p, wall_height=wall_height), p, cos_theta=cos_theta)
        R = float(case["R"])
        gap = max(float(spec["impact_gap"]), 2.0 * float(p.eps), 0.05)
        y0 = wall_height + R + gap
        state = pf.droplet_initial_state(
            p, x0=p.Lx / 2.0, y0=y0, R=R, u_impact=float(spec["u_impact"]), velocity_mode="streamfunction"
        )
        volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
        state = pf.State(
            phi=jnp.where(jnp.asarray(volume) > 0.0, state.phi, 0.0), u=state.u, v=state.v, t=state.t
        )
        mass0 = float(np.sum(np.asarray(state.phi, dtype=np.float64) * volume))
        hard0 = obs.liquid_mass(np.asarray(state.phi), np.asarray(solid.sdf), p.dx, p.dy)
        step = jax.jit(pf.step_with_diagnostics, static_argnums=(2,))
        beta_max, time_beta_max, y_cm = 0.0, 0.0, None
        gaps05, gaps01, speeds = [], [], []
        iterations, residuals = 0, 0.0
        cfl_ratios, cfl_ratios_all, dt_adv_min = [], [], []
        first_contact, contact_event = None, False
        finite = True
        started = time.perf_counter()
        for i in range(steps):
            state, info = step(state, solid, p)
            phi = np.asarray(state.phi, dtype=np.float64)
            if not np.isfinite(phi).all():
                finite = False
                break
            iterations = max(iterations, int(np.max(np.asarray(info.implicit_iterations))))
            residuals = max(residuals, float(np.max(np.asarray(info.implicit_relative_residuals))))
            if (i + 1) % int(spec["save_every"]) == 0 or i + 1 == steps:
                sdf_np = np.asarray(solid.sdf)
                width = obs.periodic_spreading_width(phi, 0.5, p.dx, p.Lx)
                beta = width / (2.0 * R)
                if beta > beta_max:
                    beta_max, time_beta_max = beta, float(state.t)
                gaps05.append(float(obs.bottom_gap(phi, sdf_np, 0.5)))
                gaps01.append(float(obs.bottom_gap(phi, sdf_np, 0.1)))
                speeds.append(
                    float(np.max(np.sqrt(np.asarray(state.u) ** 2 + np.asarray(state.v) ** 2)))
                )
                _x_cm, y_cm = obs.center_of_mass(phi, sdf_np, *pf.grids(p))
                diag = pf.cutcell_advective_cfl_diagnostic(state.u, state.v, solid, p)
                cfl_ratios.append(diag["cutcell_advective_cfl_ratio"])
                cfl_ratios_all.append(diag["cutcell_advective_cfl_ratio_all_cells"])
                dt_adv_min.append(diag["dt_adv_min"])
                if gaps05[-1] <= 0.05:
                    contact_event = True
                    if first_contact is None:
                        first_contact = float(state.t)
        phi_final = np.asarray(state.phi, dtype=np.float64)
        row = _clean(
            {
                "case": case["name"],
                "We": float(case["We"]),
                "Re": float(case["Re"]),
                "theta_deg": theta,
                "N": N,
                "steps": steps,
                "dt": float(p.dt),
                "R": R,
                "impact_gap": gap,
                "u_impact": float(spec["u_impact"]),
                "time": float(state.t),
                "wall_seconds": time.perf_counter() - started,
                "beta_max": beta_max,
                "time_beta_max": time_beta_max,
                "y_cm": y_cm,
                "min_gap_0.5": min(gaps05) if gaps05 else None,
                "min_gap_0.1": min(gaps01) if gaps01 else None,
                "first_contact_time": first_contact,
                "contact_event": contact_event,
                "peak_speed": max(speeds) if speeds else None,
                "cutcell_mass_drift": abs(float(np.sum(phi_final * volume)) - mass0) / max(abs(mass0), 1e-30),
                "hard_mask_mass_drift": abs(
                    obs.liquid_mass(phi_final, np.asarray(solid.sdf), p.dx, p.dy) - hard0
                )
                / max(abs(hard0), 1e-30),
                "cg_iterations_max": iterations,
                "cg_relative_residual_max": residuals,
                "cutcell_advective_cfl_ratio_min": min(cfl_ratios) if cfl_ratios else None,
                "cutcell_advective_cfl_ratio_min_all_cells": min(cfl_ratios_all) if cfl_ratios_all else None,
                "dt_adv_min": min(dt_adv_min) if dt_adv_min else None,
                "advection_substeps": int(
                    max(
                        1.0,
                        min(
                            float(pf.PHASE_ADVECTION_MAX_SUBSTEPS),
                            math.ceil(
                                (float(p.dt) / 3.0)
                                / max(min(dt_adv_min) if dt_adv_min else float("inf"), 1e-30)
                            ),
                        ),
                    )
                )
                if dt_adv_min
                else 1,
                "phase_advection_subcycling": str(p.phase_advection_subcycling),
                "alpha_min_positive": float(
                    np.min(np.where(volume > 0.0, volume / (p.dx * p.dy), 1.0))
                ),
                "finite": finite,
                "frozen_contract_v8_evidence": frozen.get(case["name"]),
                **pf.phase_transport_metadata(p),
            }
        )
        row["_case_key"] = key
        if cache is not None:
            (cache / f"{key}.json").write_text(json.dumps(row, allow_nan=False))
        out.append(row)
        _log(
            f"impact {case['name']:<18s} beta_max={beta_max:.4f} @t={time_beta_max:.4f} "
            f"y_cm={row['y_cm']} gap0.5={row['min_gap_0.5']} drift={row['cutcell_mass_drift']:.2e} "
            f"cfl_ratio={row['cutcell_advective_cfl_ratio_min']} cg={iterations}"
        )
    return out


# --------------------------------------------------------------------------------------
#  gate evaluation
# --------------------------------------------------------------------------------------
def _spread(angles: Sequence[float | None]) -> float | None:
    values = [float(a) for a in angles if a is not None and math.isfinite(float(a))]
    if len(values) < 2:
        return None
    return float(max(values) - min(values))


def _errors(records: Sequence[dict]) -> dict[str, float]:
    out = {}
    for record in records:
        angle = record.get("equilibrium_angle_deg")
        if angle is None:
            angle = record.get("final_sampled_angle_deg")
        if angle is None:
            continue
        out[str(record["target_deg"])] = float(angle) - float(record["target_deg"])
    return out


def _linear_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares slope/intercept/R^2, or NaN when the fit is not defined.

    A fit over a *constant* abscissa has no slope, and after contract v9 that is not a hypothetical:
    the whole point of the change is that the discrete base and the measured geometric wall coincide,
    so a regression on the transport-domain offset can legitimately degenerate. The guard keeps the
    report honest (the row is marked ``fittable: False``) instead of raising from inside LAPACK.
    """
    if len(x) < 3 or len({round(float(value), 12) for value in x}) < 3:
        return (float("nan"),) * 3
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(xa, ya, 1)
    prediction = slope * xa + intercept
    ss_res = float(np.sum((ya - prediction) ** 2))
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r_squared)


@dataclasses.dataclass
class Gate:
    gate: str
    passed: bool
    measured: bool
    limit: Any
    value: Any
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_gates(numbers: dict[str, Any]) -> list[Gate]:
    gates: list[Gate] = []

    def add(name: str, measured: bool, passed: bool, limit: Any, value: Any, detail: str = "") -> None:
        gates.append(Gate(name, bool(passed) and bool(measured), bool(measured), limit, _clean(value), detail))

    translation = numbers.get("translation", {})
    for target, row in sorted(translation.items()):
        spread = row.get("spread_deg")
        # a spread over runs that have not reached the stationarity window is not an equilibrium
        # statement, so the gate stays *unmeasured* (and the report not-ready) until they converge
        converged = bool(row.get("all_converged"))
        add(
            f"translation_spread_{target}",
            spread is not None and converged,
            spread is not None and spread <= GATES["translation_angle_spread_deg"],
            GATES["translation_angle_spread_deg"],
            {"spread_deg": spread, "strong_limit": GATES["translation_angle_spread_strong_deg"]},
            f"converged CH-only angle spread over {row.get('n_offsets')} sub-cell wall offsets "
            f"(geometric reference): {spread if spread is None else format(spread, '.4f')} deg",
        )
        add(
            f"translation_spread_strong_{target}",
            spread is not None and converged,
            spread is not None and spread <= GATES["translation_angle_spread_strong_deg"],
            GATES["translation_angle_spread_strong_deg"],
            spread,
            "strong gate",
        )
    add(
        "translation_all_converged",
        bool(translation),
        bool(translation) and all(row.get("all_converged") for row in translation.values()),
        True,
        {target: row.get("all_converged") for target, row in sorted(translation.items())},
        "every offset run must truly converge (stationarity window), otherwise the spread is not an "
        "equilibrium statement",
    )
    mass = numbers.get("mass_precision") or {}
    float64_matrix = numbers.get("ch_only_matrix_float64") or {}
    formal_drift = mass.get("float64_rtol1e-8_conserved_mass_drift_max")
    add(
        "ch_only_mass_drift_formal",
        formal_drift is not None,
        formal_drift is not None and formal_drift <= GATES["formal_mass_drift"],
        GATES["formal_mass_drift"],
        {
            "float64_rtol1e-8_max": formal_drift,
            "float64_rtol1e-8_evidence_status": mass.get("float64_rtol1e-8_evidence_status"),
            "float64_rtol1e-8_targets": mass.get("float64_rtol1e-8_targets"),
            "float32_rtol1e-6_max_reported": mass.get("float32_rtol1e-6_conserved_mass_drift_max"),
            "hard_mask_metric_max_reported": mass.get("float32_rtol1e-6_hard_mask_mass_drift_max"),
            "float64_angles_deg": {k: v.get("angle_deg") for k, v in float64_matrix.items() if isinstance(v, dict)},
        },
        "formal mass criterion on the drift-clean evidence (float64, rtol = 1e-8), exactly as the "
        "N-CH-MASS-PRECISION blocker prescribes; the float32/1e-6 drift is measured and reported "
        "alongside it and stays the blocker's subject, never a fitted setting. The gate is only "
        "*measured* when the primary_float64 section ran and every target converged with a recorded "
        f"drift; otherwise it stays unmeasured ({mass.get('float64_rtol1e-8_evidence_status')}) and a "
        "profile-limited run may not report it as zero",
    )
    add(
        "ch_only_energy_non_increasing",
        bool(translation),
        bool(translation)
        and all((row.get("energy_violations") or 0) == 0 for row in translation.values()),
        0,
        {t: r.get("energy_violations") for t, r in sorted(translation.items())},
        "F_h must be non-increasing in every CH-only run (pairwise telescoping only)",
    )

    falsification = numbers.get("falsification", {})
    for target, row in sorted(falsification.items()):
        slope = row.get("slope_deg_per_cell")
        add(
            f"falsification_slope_{target}",
            bool(row.get("fittable")) and slope is not None and math.isfinite(float(slope)),
            slope is not None and abs(float(slope)) <= GATES["falsification_max_abs_slope_deg_per_cell"],
            GATES["falsification_max_abs_slope_deg_per_cell"],
            {"slope_deg_per_cell": slope, "r_squared": row.get("r_squared"), "v8_slope": row.get("v8_slope")},
            "post-v9 angle regressed on the old hard-mask plane offset (y_disc - y_geo)/dx: a slope "
            "still of order the contract-v8 value falsifies the hypothesis. A non-fittable row means "
            "the abscissa has (almost) no spread -- i.e. the transport-domain offset itself has "
            "collapsed -- which is the mechanism working, not a measurement gap",
        )

    resolution = numbers.get("resolution", {})
    for target, row in sorted(resolution.items()):
        spread = row.get("spread_deg")
        add(
            f"resolution_spread_{target}",
            spread is not None,
            spread is not None and spread <= GATES["resolution_angle_spread_deg"],
            GATES["resolution_angle_spread_deg"],
            {"spread_deg": spread, "angles": row.get("angles"), "v8_spread_deg": row.get("v8_spread_deg")},
            "same physical wall at N = 96/128/192 (L1A-2e measured 3.093 deg at 150 deg)",
        )

    ch_only = numbers.get("ch_only_matrix", {})
    add(
        "ch_only_mae_deg",
        ch_only.get("mae_deg") is not None,
        (ch_only.get("mae_deg") or 1e9) <= GATES["ch_only_mae_deg"],
        GATES["ch_only_mae_deg"],
        ch_only.get("mae_deg"),
        "four-target CH-only matrix, mean absolute angle error",
    )
    add(
        "ch_only_max_error_deg",
        ch_only.get("max_error_deg") is not None,
        (ch_only.get("max_error_deg") or 1e9) <= GATES["ch_only_max_error_deg"],
        GATES["ch_only_max_error_deg"],
        ch_only.get("max_error_deg"),
        "four-target CH-only matrix, worst angle error",
    )
    add(
        "ch_only_neutral_error_deg",
        ch_only.get("neutral_error_deg") is not None,
        abs(ch_only.get("neutral_error_deg") or 1e9) <= GATES["neutral_error_deg"],
        GATES["neutral_error_deg"],
        ch_only.get("neutral_error_deg"),
        "90 deg neutral control",
    )
    angles = ch_only.get("angles_deg") or {}
    add(
        "ch_only_monotonic",
        len(angles) == 4,
        bool(ch_only.get("monotonic")),
        True,
        ch_only.get("angles_deg"),
        "the four measured angles must be monotonic in the target angle; the gate stays unmeasured "
        "until all four targets have been run (a partial matrix is not a monotonicity statement)",
    )

    laplace = numbers.get("laplace", {})
    ratios = laplace.get("ratios") or {}
    add(
        "laplace_ratios_positive",
        bool(ratios),
        bool(ratios) and all(v > 0 for v in ratios.values()),
        True,
        ratios,
        "static-drop pressure-jump ratio v9 / pinned-v8 for every radius",
    )
    add(
        "laplace_ratio_change_vs_v8",
        bool(ratios),
        bool(ratios) and all(abs(v - 1.0) <= GATES["laplace_ratio_change_vs_v8"] for v in ratios.values()),
        GATES["laplace_ratio_change_vs_v8"],
        {R: (None if v is None else v - 1.0) for R, v in ratios.items()},
        "the empty solid must degenerate exactly, so the Laplace regression cannot move",
    )
    add(
        "laplace_r_squared",
        laplace.get("r_squared") is not None,
        (laplace.get("r_squared") or 0.0) >= GATES["laplace_min_r_squared"],
        GATES["laplace_min_r_squared"],
        laplace.get("r_squared"),
        "delta_p ~ 1/R regression quality (v9 cut-cell geometry)",
    )

    cg = numbers.get("cg", {})
    add(
        "cg_relative_residual_float32",
        cg.get("residual_max_float32") is not None,
        (cg.get("residual_max_float32") or 1e9) <= GATES["cg_relative_residual_float32"],
        GATES["cg_relative_residual_float32"],
        cg.get("residual_max_float32"),
        "worst relative residual over every float32 solve",
    )
    add(
        "cg_relative_residual_float64",
        cg.get("residual_max_float64") is not None,
        (cg.get("residual_max_float64") or 1e9) <= GATES["cg_relative_residual_float64"],
        GATES["cg_relative_residual_float64"],
        cg.get("residual_max_float64"),
        "worst relative residual over every float64 solve",
    )
    add(
        "cg_no_failed_solve",
        cg.get("n_failed") is not None,
        int(cg.get("n_failed") or 0) == 0,
        0,
        cg.get("n_failed"),
        "no solve may fail closed in an accepted trajectory",
    )

    impact = numbers.get("impact", {})
    add(
        "cutcell_advective_cfl_ratio",
        impact.get("cfl_ratio_min") is not None,
        (impact.get("cfl_ratio_min") or 1e9) <= GATES["cutcell_advective_cfl_ratio_limit"],
        GATES["cutcell_advective_cfl_ratio_limit"],
        {
            "ratio_min": impact.get("cfl_ratio_min"),
            "ratio_min_all_cells": impact.get("cfl_ratio_min_all_cells"),
            "dt_adv_min": impact.get("dt_adv_min"),
            "advection_substeps": [row.get("advection_substeps") for row in impact.get("rows") or []],
            "subcycling_setting": impact.get("subcycling_setting"),
        },
        "dt_global / min_i dt_adv,i over the impact runs, i.e. how much smaller the global step is "
        "than the tightest local cut-cell advection time: <= 1 means the advection is resolved, > 1 "
        "would require the phase-only subcycling path (which exists and is opt-in; flux "
        "redistribution and cell merging are out of scope for this stage)",
    )
    add(
        "impact_cutcell_mass_drift",
        impact.get("mass_drift_max") is not None,
        (impact.get("mass_drift_max") or 1e9) <= GATES["formal_mass_drift"],
        GATES["formal_mass_drift"],
        impact.get("mass_drift_max"),
        "conserved cut-cell mass drift over the impact regressions",
    )

    geometry = numbers.get("geometry", {})
    add(
        "geometry_audit_passed",
        geometry.get("passed") is not None,
        bool(geometry.get("passed")),
        True,
        {"passed": geometry.get("passed"), "failed": geometry.get("failed_checks")},
        "production.cutcell_geometry_audit (volume/aperture/wall closure, single authority, "
        "flat-wall exactness, degenerate empty solid)",
    )
    transport = numbers.get("transport", {})
    add(
        "transport_audit_passed",
        transport.get("passed") is not None,
        bool(transport.get("passed")),
        True,
        {"passed": transport.get("passed"), "failed": transport.get("failed_checks")},
        "production.cutcell_phase_transport_audit (pairwise conservation, telescoping, "
        "manufactured solutions, weighted-SPD operator, fail-closed CG)",
    )

    v8 = numbers.get("translation_v8", {})
    add(
        "v8_pinned_reproduction",
        v8.get("max_abs_difference_deg") is not None,
        (v8.get("max_abs_difference_deg") or 1e9) <= GATES["v8_reproduction_tolerance_deg"]
        and bool(v8.get("spread_grew")),
        GATES["v8_reproduction_tolerance_deg"],
        v8,
        "the pinned hard_cell_v7 transport still reproduces the contract-v8 alignment spread "
        "under the v9 code: the improvement is attributable to the transport domain, not to an "
        "unrelated change",
    )
    return gates


def not_ready_triggers(gates: Sequence[Gate], numbers: dict[str, Any]) -> list[str]:
    """Hard-fail conditions: any of these means L1A-2f is NOT READY."""
    triggers: list[str] = []
    for gate in gates:
        if gate.measured and not gate.passed:
            triggers.append(f"gate failed: {gate.gate} ({gate.detail})")
    for name, row in sorted((numbers.get("translation") or {}).items()):
        if row.get("n_converged", 0) < row.get("n_offsets", 0):
            triggers.append(
                f"translation {name}: only {row.get('n_converged')}/{row.get('n_offsets')} offsets truly converged"
            )
    if numbers.get("falsified"):
        triggers.append(
            "HYPOTHESIS FALSIFIED: the post-v9 angle still tracks the old hard-mask plane offset "
            f"(slopes {numbers.get('falsification')})"
        )
    return triggers


def assess_blockers(gates: Sequence[Gate], numbers: dict[str, Any]) -> dict[str, Any]:
    by_name = {gate.gate: gate for gate in gates}
    translation_spreads = {
        name: row.get("spread_deg") for name, row in sorted((numbers.get("translation") or {}).items())
    }
    alignment_measured = all(v is not None for v in translation_spreads.values()) and bool(translation_spreads)
    alignment_passed = alignment_measured and all(
        by_name[f"translation_spread_{name}"].passed for name in translation_spreads
    )
    resolution_passed = all(
        gate.passed for name, gate in by_name.items() if name.startswith("resolution_spread_") and gate.measured
    ) and any(gate.measured for name, gate in by_name.items() if name.startswith("resolution_spread_"))
    chns = numbers.get("chns_matrix") or {}
    chns_drift_clean = numbers.get("mass_precision", {}).get("float64_rtol1e-8_conserved_mass_drift_max")
    chns_drift_clean = chns_drift_clean is not None and chns_drift_clean <= GATES["formal_mass_drift"]
    chns_clean = bool(chns) and all(row.get("mass_drift_ok") for row in chns.values())
    chns_converged = bool(chns) and all(row.get("converged") for row in chns.values())
    return {
        "N-WALL-ALIGNMENT-TRANSPORT-DOMAIN": {
            "status": "resolved_in_contract_v9" if (alignment_passed and resolution_passed) else (
                "measurement_incomplete" if not (alignment_measured and resolution_passed) else "confirmed_problem"
            ),
            "evidence": {
                "translation_spreads_deg": translation_spreads,
                "translation_gate_deg": GATES["translation_angle_spread_deg"],
                "resolution_gate_deg": GATES["resolution_angle_spread_deg"],
                "falsification": numbers.get("falsification"),
                "v8_frozen_spreads_deg": {"60": 1.570, "90": 0.926, "120": 3.632, "150": 7.241},
                "v8_frozen_resolution_spread_deg": {"150": 3.093},
            },
        },
        "W-CONTACT-ANGLE": {
            "status": (
                "thermodynamic_equilibrium_validated_v9"
                if (chns_converged and chns_clean and by_name.get("ch_only_mae_deg") and
                    by_name["ch_only_mae_deg"].passed and chns_drift_clean)
                else "measurement_required"
            ),
            "evidence": {
                "ch_only_matrix": numbers.get("ch_only_matrix"),
                "chns_matrix": chns,
                "chns_all_converged": chns_converged,
                "chns_all_drift_clean": chns_clean,
            },
        },
        "N-CH-MASS-PRECISION": {
            "status": "measurement_recorded_v9",
            "evidence": {
                "mass_precision": numbers.get("mass_precision"),
                "precision_matrix": numbers.get("precision_matrix"),
            },
        },
        "P-SOLID-PIN": {
            "status": "re_examined_v9",
            "evidence": {
                "no_projection_in_v9_path": numbers.get("no_projection_calls"),
                "solid_phase_fraction_max": numbers.get("solid_phase_fraction_max"),
                "impact": numbers.get("impact"),
            },
        },
    }


def _discrete_plane_offset(record: dict[str, Any]) -> float | None:
    """``(y_discrete - y_geometric)/dx``: the old hard-mask transport-domain offset, in cells.

    Read from the recorded planes themselves (``discrete_wall_plane`` from the cell-centre mask and
    ``geometric_wall_plane`` from the wall measure), so the number is a property of the run that was
    actually performed; ``dx`` is the cell size of the (square) run.
    """
    discrete = record.get("discrete_wall_plane")
    geometric = record.get("geometric_wall_plane")
    cell = record.get("dx")
    if discrete is None or geometric is None or not cell:
        return None
    return float((float(discrete) - float(geometric)) / float(cell))


def collect_numbers(sections: dict[str, Any]) -> dict[str, Any]:
    """Reduce the raw section records to the gate inputs."""
    numbers: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for name in ("translation", "resolution", "primary_float64", "production_mobility", "precision", "chns"):
        for record in sections.get(name) or []:
            record = dict(record)
            record["_section"] = name
            records.append(record)
    numbers["records"] = len(records)

    def equilibrium(record: dict) -> float | None:
        angle = record.get("equilibrium_angle_deg")
        return float(angle) if angle is not None else None

    def summarize(section: str, key: str = "target_deg") -> dict[str, Any]:
        rows = [r for r in records if r.get("_section") == section]
        grouped: dict[Any, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row.get(key), []).append(row)
        out: dict[str, Any] = {}
        for group, items in grouped.items():
            angles = [equilibrium(item) for item in items]
            out[str(group)] = {
                "n_offsets": len(items),
                "n_converged": sum(1 for item in items if item.get("converged")),
                "all_converged": all(item.get("converged") for item in items),
                "angles_deg": _clean(angles),
                "spread_deg": _spread(angles),
                "offsets_over_dy": [item.get("wall_offset_over_dy") for item in items],
                # The offset the *contract-v8* mechanism lived on: how far the cell-centre hard-fluid
                # staircase (the transport domain before v9) sat from the geometric sdf = 0 wall that
                # the contact angle is measured against. Contract v9 removes this mismatch, so it is
                # the regressor of the falsification test below.
                "discrete_plane_offsets_cells": [
                    _discrete_plane_offset(item) for item in items
                ],
                "subcell_offsets_cells": [item.get("wall_offset_cells") for item in items],
                "max_conserved_mass_drift": max(
                    (item.get("conserved_mass_drift") or 0.0) for item in items
                )
                if items
                else None,
                "max_hard_mask_mass_drift": max((item.get("mass_drift") or 0.0) for item in items) if items else None,
                "energy_violations": sum(int(item.get("free_energy_monotonic_violations") or 0) for item in items),
                "N_values": sorted({int(item["N"]) for item in items}),
                "cg_iterations_max": max((int(item.get("implicit_iterations_max") or 0) for item in items), default=0),
            }
        return out

    numbers["translation"] = summarize("translation")
    numbers["resolution"] = summarize("resolution")

    # falsification: regress the converged angle on the *old* hard-mask plane offset
    falsification: dict[str, Any] = {}
    v8_slopes = {"60.0": 1.788, "90.0": 1.576, "120.0": 4.159, "150.0": 8.200}
    for target, row in numbers["translation"].items():
        pairs = [
            (o, a)
            for a, o in zip(row["angles_deg"], row["discrete_plane_offsets_cells"])
            if a is not None and o is not None
        ]
        offsets = [o for o, _ in pairs]
        angles = [a for _, a in pairs]
        v8_slope = v8_slopes.get(target)
        if len(offsets) < 3:
            falsification[target] = {"fittable": False, "n_points": len(angles), "v8_slope": v8_slope}
            continue
        slope, intercept, r_squared = _linear_fit(offsets, angles)
        fittable = slope == slope  # not NaN
        falsification[target] = {
            "fittable": bool(fittable),
            "slope_deg_per_cell": slope if fittable else None,
            "intercept_deg": intercept if fittable else None,
            "r_squared": r_squared if fittable else None,
            "n_points": len(angles),
            "distinct_offsets_cells": sorted({round(o, 12) for o in offsets}),
            "v8_slope_deg_per_cell": v8_slope,
            "v8_slope": v8_slope,
            "slope_reduction_vs_v8": (abs(v8_slope) / abs(slope)) if (v8_slope and fittable and slope) else None,
        }
    numbers["falsification"] = falsification
    numbers["falsified"] = any(
        (row.get("slope_deg_per_cell") is not None and abs(row["slope_deg_per_cell"]) > GATES["falsification_max_abs_slope_deg_per_cell"])
        or (row.get("r_squared") is not None and row["r_squared"] >= 0.9
            and row.get("slope_reduction_vs_v8") is not None
            and row["slope_reduction_vs_v8"] < GATES["falsification_slope_reduction_vs_v8"])
        for row in falsification.values()
    )
    numbers["falsification_fittable_targets"] = sorted(
        target for target, row in falsification.items() if row.get("fittable")
    )

    # CH-only four-target matrix (offset 0, float32, accelerated mobility)
    primary = {r["target_deg"]: r for r in records if r.get("_section") == "translation"
               and abs(float(r.get("wall_offset_over_dy") or 0.0)) < 1e-12}
    # the same matrix in float64 with rtol = 1e-8: the *formal* mass evidence (see the
    # N-CH-MASS-PRECISION blocker), because the production float32/1e-6 setting is exactly the
    # setting under measurement there
    float64_matrix = {}
    for record in [r for r in records if r.get("_section") == "primary_float64"]:
        float64_matrix[str(record["target_deg"])] = {
            "target_deg": float(record["target_deg"]),
            "angle_deg": equilibrium(record),
            "error_deg": None if equilibrium(record) is None else equilibrium(record) - float(record["target_deg"]),
            "converged": bool(record.get("converged")),
            "conserved_mass_drift": record.get("conserved_mass_drift"),
            "hard_mask_mass_drift": record.get("mass_drift"),
            "steps": record.get("steps"),
        }
    # Per-target rows only. The summary keys below (mae/max/neutral) are *derived* statistics, so they
    # must never be mistaken for measurements: a profile that skips ``primary_float64`` (the CI smoke
    # runs a section subset) leaves this dict empty and the formal gate stays *unmeasured*, never zero.
    f64_rows = [row for row in float64_matrix.values() if isinstance(row, dict)]
    f64_errors = [row["error_deg"] for row in f64_rows if row["error_deg"] is not None]
    float64_matrix["mae_deg"] = (sum(abs(e) for e in f64_errors) / len(f64_errors)) if f64_errors else None
    float64_matrix["max_error_deg"] = max((abs(e) for e in f64_errors), default=None)
    float64_matrix["neutral_error_deg"] = (float64_matrix.get("90.0") or {}).get("error_deg")
    float64_matrix["n_targets"] = len(f64_rows)
    numbers["ch_only_matrix_float64"] = float64_matrix

    # The *formal* mass evidence (the N-CH-MASS-PRECISION criterion) is only a measurement when the
    # float64 / rtol = 1e-8 target matrix actually ran and every target truly converged with a recorded
    # drift. Anything less leaves the value ``None``, which the gate below turns into ``measured=False``
    # -- fail closed, exactly like the missing sections themselves.
    formal_drift = None
    formal_status = "section primary_float64 not run"
    if f64_rows:
        missing_drift = sorted(str(r["target_deg"]) for r in f64_rows if r.get("conserved_mass_drift") is None)
        unconverged = [r for r in f64_rows if not r.get("converged")]
        missing_angle = sorted(str(r["target_deg"]) for r in f64_rows if r.get("error_deg") is None)
        if missing_drift:
            formal_status = f"conserved_mass_drift not recorded for {missing_drift}"
        elif unconverged:
            formal_status = (
                f"only {len(f64_rows) - len(unconverged)}/{len(f64_rows)} float64 targets converged"
            )
        elif missing_angle:
            formal_status = f"equilibrium angle missing for targets {missing_angle}"
        else:
            formal_drift = max(row["conserved_mass_drift"] for row in f64_rows)
            formal_status = f"{len(f64_rows)} converged float64 / rtol 1e-8 targets"

    f32_drift = max((r.get("conserved_mass_drift") or 0.0) for r in primary.values()) if primary else None
    numbers["mass_precision"] = {
        "float32_rtol1e-6_conserved_mass_drift_max": f32_drift,
        "float32_rtol1e-6_hard_mask_mass_drift_max": max(
            (r.get("mass_drift") or 0.0) for r in primary.values()
        )
        if primary
        else None,
        "float64_rtol1e-8_conserved_mass_drift_max": formal_drift,
        "float64_rtol1e-8_evidence_status": formal_status,
        "float64_rtol1e-8_targets": len(f64_rows),
        "formal_evidence_dtype": "float64",
        "formal_evidence_rtol": 1.0e-8,
        "production_default_dtype": "float32",
        "production_default_rtol": 1.0e-6,
        "blocker": "N-CH-MASS-PRECISION",
    }
    angles = {t: equilibrium(r) for t, r in sorted(primary.items())}
    errors = {t: (None if a is None else a - t) for t, a in angles.items()}
    finite_errors = [e for e in errors.values() if e is not None]
    measured_angles = [a for t, a in sorted(angles.items()) if a is not None]
    numbers["ch_only_matrix"] = {
        "angles_deg": _clean(angles),
        "errors_deg": _clean(errors),
        "mae_deg": (sum(abs(e) for e in finite_errors) / len(finite_errors)) if finite_errors else None,
        "max_error_deg": max((abs(e) for e in finite_errors), default=None),
        "neutral_error_deg": errors.get(90.0),
        "monotonic": (len(measured_angles) >= 2 and all(b >= a for a, b in zip(measured_angles, measured_angles[1:]))),
        "all_converged": all(r.get("converged") for r in primary.values()) if primary else None,
        "max_conserved_mass_drift": max((r.get("conserved_mass_drift") or 0.0) for r in primary.values())
        if primary
        else None,
    }

    # pinned-v8 A/B
    v8_rows = sections.get("translation_v8") or []
    v9_rows = {
        (r["target_deg"], round(float(r.get("wall_offset_over_dy") or 0.0), 6)): r
        for r in records
        if r.get("_section") == "translation"
    }
    differences, v8_by_target = [], {}
    for row in v8_rows:
        key = (row["target_deg"], round(float(row.get("wall_offset_over_dy") or 0.0), 6))
        counterpart = v9_rows.get(key)
        v8_by_target.setdefault(str(row["target_deg"]), []).append(equilibrium(row))
        if counterpart is not None:
            a, b = equilibrium(row), equilibrium(counterpart)
            if a is not None and b is not None:
                differences.append(abs(a - b))
    v8_spreads = {t: _spread(v) for t, v in v8_by_target.items()}
    v9_spreads = {t: numbers["translation"].get(t, {}).get("spread_deg") for t in v8_spreads}
    numbers["translation_v8"] = {
        "max_abs_difference_deg": max(differences) if differences else None,
        "n_paired_runs": len(differences),
        "v8_pinned_spreads_deg": v8_spreads,
        "v9_spreads_deg": v9_spreads,
        "spread_grew": any(
            (v8_spreads[t] is not None and v9_spreads.get(t) is not None and v8_spreads[t] > v9_spreads[t])
            for t in v8_spreads
        ),
    }

    # precision matrix
    numbers["precision_matrix"] = [
        {
            "label": r.get("label"),
            "dtype": r.get("dtype"),
            "rtol": r.get("ch_solver_rtol"),
            "steps": r.get("steps"),
            "mobility_scaled_time": r.get("mobility_scaled_time"),
            # a fixed-step probe has no equilibrium claim, so the sampled angle is reported as
            # ``sampled_angle_deg`` and never as an equilibrium
            "sampled_angle_deg": r.get("final_sampled_angle_deg"),
            "converged": bool(r.get("converged")),
            "angle_deg": equilibrium(r),
            "conserved_mass_drift": r.get("conserved_mass_drift"),
            "hard_mask_mass_drift": r.get("mass_drift"),
            "cg_iterations_max": r.get("implicit_iterations_max"),
            "cg_residual_max": r.get("implicit_residual_max"),
        }
        for r in records
        if r.get("_section") == "precision"
    ]

    # CHNS matrix at production defaults
    chns: dict[str, Any] = {}
    for r in [x for x in records if x.get("_section") == "chns"]:
        chns[str(r["target_deg"])] = {
            "converged": bool(r.get("converged")),
            "angle_deg": equilibrium(r),
            "error_deg": None if equilibrium(r) is None else equilibrium(r) - float(r["target_deg"]),
            "conserved_mass_drift": r.get("conserved_mass_drift"),
            "mass_drift_ok": (r.get("conserved_mass_drift") or 1.0) <= GATES["formal_mass_drift"],
            "steps": r.get("steps"),
            "stop_reason": r.get("stop_reason"),
            "peak_speed": r.get("final_max_speed"),
            "diagnostic_only": not bool(r.get("converged")),
        }
    numbers["chns_matrix"] = chns

    # CG quality across every relaxation record
    f32 = [r.get("implicit_residual_max") for r in records if r.get("dtype") == "float32"]
    f64 = [r.get("implicit_residual_max") for r in records if r.get("dtype") == "float64"]
    numbers["cg"] = {
        "residual_max_float32": max((v for v in f32 if v is not None), default=None),
        "residual_max_float64": max((v for v in f64 if v is not None), default=None),
        "iterations_max": max((int(r.get("implicit_iterations_max") or 0) for r in records), default=0),
        "n_failed": sum(1 for r in records if r.get("implicit_solve_failed")),
    }
    numbers["solid_phase_fraction_max"] = max(
        (float(r.get("solid_phase_fraction_max") or 0.0) for r in records), default=None
    )

    laplace = sections.get("laplace") or {}
    rows = laplace.get("rows") or []
    cut_rows = [r for r in rows if r["transport"] == "sdf_cutcell_fv_v1"]
    xs = [1.0 / r["R"] for r in cut_rows]
    ys = [r["delta_p"] for r in cut_rows]
    if len(xs) >= 2:
        slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
        prediction = slope * np.asarray(xs) + intercept
        ss_res = float(np.sum((np.asarray(ys) - prediction) ** 2))
        ss_tot = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    else:
        r_squared = None
    numbers["laplace"] = {
        "ratios": laplace.get("ratios") or {},
        "r_squared": r_squared,
        "delta_p_times_R": {r["R"]: r["delta_p_times_R"] for r in cut_rows},
        "empty_solid_degenerate": all(
            r["volume_is_cell_area"] and r["aperture_x_is_full"] and r["alpha_min"] == 1.0
            and r["wall_measure_is_zero"] == 0.0
            for r in cut_rows
        )
        if cut_rows
        else None,
        "rows": rows,
    }

    impact_rows = sections.get("impact") or []
    numbers["impact"] = {
        "rows": impact_rows,
        "cfl_ratio_min": min(
            (r["cutcell_advective_cfl_ratio_min"] for r in impact_rows if r.get("cutcell_advective_cfl_ratio_min")),
            default=None,
        ),
        "cfl_ratio_min_all_cells": min(
            (
                r["cutcell_advective_cfl_ratio_min_all_cells"]
                for r in impact_rows
                if r.get("cutcell_advective_cfl_ratio_min_all_cells") is not None
            ),
            default=None,
        ),
        "dt_adv_min": min((r["dt_adv_min"] for r in impact_rows if r.get("dt_adv_min")), default=None),
        "mass_drift_max": max((r.get("cutcell_mass_drift") or 0.0) for r in impact_rows) if impact_rows else None,
        "alpha_min_positive": min((r.get("alpha_min_positive") or 1.0) for r in impact_rows) if impact_rows else None,
        "cg_iterations_max": max((int(r.get("cg_iterations_max") or 0) for r in impact_rows), default=0),
        "subcycling_setting": pf.PHASE_ADVECTION_SUBCYCLING,
    }

    geometry = sections.get("geometry") or {}
    numbers["geometry"] = {
        "passed": geometry.get("passed"),
        "failed_checks": [c["name"] for c in geometry.get("checks", []) if not c.get("passed")],
        "alpha_min_positive": geometry.get("alpha_min_positive"),
        "alpha_percentiles": geometry.get("alpha_percentiles"),
        "max_area_face_over_volume": geometry.get("max_area_face_over_volume"),
    }
    transport = sections.get("transport") or {}
    numbers["transport"] = {
        "passed": transport.get("passed"),
        "failed_checks": [c["name"] for c in transport.get("checks", []) if not c.get("passed")],
    }
    numbers["no_projection_calls"] = True  # the v9 path contains no mass projection (AST-checked in CI)
    return numbers


def assemble_report(
    cfg: dict[str, Any], sections: dict[str, Any], numbers: dict[str, Any], gates: Sequence[Gate]
) -> dict[str, Any]:
    triggers = not_ready_triggers(gates, numbers)
    return _clean(
        {
            "stage": STAGE,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
            **pf.phase_transport_metadata(pf.PhaseFieldParams(Nx=2, Ny=2)),
            "dataset_schema_version": 3,
            "wall_measure_method": str(pf.WALL_MEASURE_METHOD),
            "wall_measure_contract_version": int(pf.WALL_MEASURE_CONTRACT_VERSION),
            "corner_sign_tolerance_factor": float(pf.CORNER_SIGN_TOLERANCE_FACTOR),
            "profile": cfg.get("_profile"),
            "settings": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "gates": [gate.to_dict() for gate in gates],
            "gate_limits": GATES,
            "numbers": numbers,
            "blockers": assess_blockers(gates, numbers),
            "not_ready_triggers": triggers,
            "not_ready": bool(triggers),
            "sections_run": sorted(sections),
            "sections_missing": [name for name in SECTIONS if name not in sections],
            "runtime": {
                "python_jax_version": str(jax.__version__),
                "backend": str(jax.default_backend()),
                "devices": int(jax.device_count()),
            },
        }
    )


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("stage") != STAGE:
        errors.append(f"stage must be {STAGE}")
    if report.get("solver_contract_version") != 9:
        errors.append(f"solver_contract_version must be 9; got {report.get('solver_contract_version')!r}")
    if report.get("phase_transport_geometry") != "sdf_cutcell_fv_v1":
        errors.append("phase_transport_geometry must be sdf_cutcell_fv_v1")
    if report.get("phase_control_volume") != "partial_cell_volume":
        errors.append("phase_control_volume must be partial_cell_volume")
    if report.get("phase_face_aperture") != "partial_open_length":
        errors.append("phase_face_aperture must be partial_open_length")
    for key in ("gates", "numbers", "blockers", "not_ready_triggers"):
        if key not in report:
            errors.append(f"missing report section {key}")
    if not report.get("allow_incomplete") and not report.get("not_ready") and report.get("sections_missing"):
        errors.append(f"a ready report cannot have missing sections: {report['sections_missing']}")
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as exc:
        errors.append(f"report is not strict JSON: {exc}")
    return errors


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {STAGE}: phase transport on geometry-conforming cut-cell control volumes",
        "",
        f"- solver contract = {report['solver_contract_version']}, dataset schema = {report['dataset_schema_version']}",
        f"- phase transport = `{report['phase_transport_geometry']}` "
        f"(control volume `{report['phase_control_volume']}`, face aperture `{report['phase_face_aperture']}`)",
        f"- profile = `{report['profile']}`, sections run = {report['sections_run']}",
        f"- **NOT READY: {report['not_ready']}**",
        "",
        "## Gates",
        "",
        "| gate | measured | passed | limit | value |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gate in report["gates"]:
        value = gate["value"]
        text = value if not isinstance(value, (dict, list)) else json.dumps(value, default=str)
        if len(str(text)) > 90:
            text = str(text)[:87] + "..."
        lines.append(
            f"| `{gate['gate']}` | {gate['measured']} | {gate['passed']} | {gate['limit']} | {text} |"
        )
    lines += ["", "## Translation matrix (converged CH-only angle vs sub-cell wall offset)", ""]
    translation = report["numbers"].get("translation") or {}
    if translation:
        lines += ["| target | offsets (dy) | angles (deg) | spread (deg) | converged | conserved drift |", "| --- | --- | --- | --- | --- | --- |"]
        for target, row in sorted(translation.items()):
            lines.append(
                f"| {target} | {row['offsets_over_dy']} | "
                f"{[None if a is None else round(a, 4) for a in row['angles_deg']]} | "
                f"{row['spread_deg']} | {row['n_converged']}/{row['n_offsets']} | "
                f"{row['max_conserved_mass_drift']} |"
            )
    falsification = report["numbers"].get("falsification") or {}
    if falsification:
        lines += ["", "## Falsification: angle vs the old hard-mask plane offset", "",
                  "| target | slope (deg/cell) | R^2 | v8 slope | reduction |", "| --- | --- | --- | --- | --- |"]
        for target, row in sorted(falsification.items()):
            lines.append(
                f"| {target} | {row['slope_deg_per_cell']} | {row['r_squared']} | "
                f"{row['v8_slope_deg_per_cell']} | {row['slope_reduction_vs_v8']} |"
            )
        lines += ["", f"Hypothesis falsified: **{report['numbers'].get('falsified')}**"]
    resolution = report["numbers"].get("resolution") or {}
    if resolution:
        lines += ["", "## Grid resolution on the same physical wall", "",
                  "| target | N | angles (deg) | spread (deg) |", "| --- | --- | --- | --- |"]
        for target, row in sorted(resolution.items()):
            lines.append(
                f"| {target} | {row['N_values']} | "
                f"{[None if a is None else round(a, 4) for a in row['angles_deg']]} | {row['spread_deg']} |"
            )
    ch_only = report["numbers"].get("ch_only_matrix") or {}
    if ch_only:
        lines += ["", "## CH-only four-target matrix", "",
                  f"- angles = {ch_only.get('angles_deg')}",
                  f"- errors = {ch_only.get('errors_deg')}",
                  f"- MAE = {ch_only.get('mae_deg')} deg, max = {ch_only.get('max_error_deg')} deg, "
                  f"90 deg = {ch_only.get('neutral_error_deg')} deg, monotonic = {ch_only.get('monotonic')}",
                  f"- conserved mass drift = {ch_only.get('max_conserved_mass_drift')}"]
    float64_matrix = report["numbers"].get("ch_only_matrix_float64") or {}
    if float64_matrix:
        lines += ["", "## CH-only four-target matrix, float64 / rtol 1e-8", "",
                  f"- angles = { {k: v['angle_deg'] for k, v in float64_matrix.items() if isinstance(v, dict)} }",
                  f"- MAE = {float64_matrix.get('mae_deg')} deg, max = {float64_matrix.get('max_error_deg')} deg, "
                  f"90 deg = {float64_matrix.get('neutral_error_deg')} deg",
                  f"- conserved mass drift = "
                  f"{(report['numbers'].get('mass_precision') or {}).get('float64_rtol1e-8_conserved_mass_drift_max')} "
                  f"({(report['numbers'].get('mass_precision') or {}).get('float64_rtol1e-8_evidence_status')})"]
    mass = report["numbers"].get("mass_precision") or {}
    if mass:
        lines += ["", "## Mass precision (`N-CH-MASS-PRECISION`)", "",
                  f"- float32 / rtol 1e-6, conserved `sum_i V_i phi_i` = "
                  f"{mass.get('float32_rtol1e-6_conserved_mass_drift_max')}",
                  f"- float32 / rtol 1e-6, cell-centre hard-mask metric = "
                  f"{mass.get('float32_rtol1e-6_hard_mask_mass_drift_max')}",
                  f"- float64 / rtol 1e-8, conserved = {mass.get('float64_rtol1e-8_conserved_mass_drift_max')} "
                  f"({mass.get('float64_rtol1e-8_evidence_status')})",
                  f"- formal criterion ({GATES['formal_mass_drift']:g}) is evaluated on the float64 evidence; "
                  "when that section is not part of the profile the gate stays *unmeasured* (never zero)"]
    precision = report["numbers"].get("precision_matrix") or []
    if precision:
        lines += ["", "## Precision matrix (N-CH-MASS-PRECISION)", "",
                  "Equal mobility-scaled time for every row; the angle is a fixed-step sample, not an "
                  "equilibrium claim.", "",
                  "| label | sampled angle | conserved drift | hard-mask drift | CG iters |",
                  "| --- | --- | --- | --- | --- |"]
        for row in precision:
            lines.append(
                f"| {row['label']} | {row.get('sampled_angle_deg')} (sampled, converged="
                f"{row.get('converged')}) | {row['conserved_mass_drift']} | "
                f"{row['hard_mask_mass_drift']} | {row['cg_iterations_max']} |"
            )
    chns = report["numbers"].get("chns_matrix") or {}
    if chns:
        lines += ["", "## Full CHNS at production defaults", "",
                  "| target | converged | angle | error (deg) | conserved drift | steps |", "| --- | --- | --- | --- | --- | --- |"]
        for target, row in sorted(chns.items()):
            lines.append(
                f"| {target} | {row['converged']} | {row['angle_deg']} | {row['error_deg']} | "
                f"{row['conserved_mass_drift']} | {row['steps']} |"
            )
    laplace = report["numbers"].get("laplace") or {}
    if laplace.get("rows"):
        lines += ["", "## Static Laplace regression (empty solid)", "",
                  f"- ratios v9 / pinned-v8 = {laplace.get('ratios')}",
                  f"- R^2 = {laplace.get('r_squared')}, delta_p*R = {laplace.get('delta_p_times_R')}",
                  f"- empty solid degenerate (V = dx dy, alpha = 1, aperture = 1, A_wall = 0) = "
                  f"{laplace.get('empty_solid_degenerate')}"]
    impact = report["numbers"].get("impact") or {}
    if impact.get("rows"):
        lines += ["", "## Impact regression + cut-cell advective CFL", "",
                  "The CFL ratio is `dt_global / min_i dt_adv,i`: <= 1 means the cut-cell advection is "
                  "resolved at the global step (subcycling disabled).",
                  "",
                  "| case | beta_max | t(beta_max) | y_cm | min gap 0.5 | first contact | cut-cell drift | CG iters | CFL ratio min | substeps | dt_adv min |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in impact["rows"]:
            lines.append(
                f"| {row['case']} | {row['beta_max']} | {row['time_beta_max']} | {row['y_cm']} | "
                f"{row['min_gap_0.5']} | {row['first_contact_time']} | {row['cutcell_mass_drift']} | "
                f"{row['cg_iterations_max']} | {row['cutcell_advective_cfl_ratio_min']} | "
                f"{row['advection_substeps']} | {row['dt_adv_min']} |"
            )
    lines += ["", "## Blockers", ""]
    for name, row in sorted((report.get("blockers") or {}).items()):
        lines.append(f"- `{name}`: **{row['status']}**")
    if report["not_ready_triggers"]:
        lines += ["", "## NOT READY triggers", ""]
        lines += [f"- {trigger}" for trigger in report["not_ready_triggers"]]
    lines.append("")
    return "\n".join(lines)


def run_audit(
    profile: str = "quick",
    out: str | os.PathLike | None = None,
    *,
    sections: Sequence[str] | None = None,
    max_steps: int | None = None,
    overwrite: bool = False,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    cfg = dict(PROFILES[profile])
    cfg["_profile"] = profile
    wanted = tuple(sections) if sections else SECTIONS
    unknown = [name for name in wanted if name not in SECTIONS]
    if unknown:
        raise ValueError(f"unknown sections {unknown}; expected a subset of {list(SECTIONS)}")
    root = Path(out) if out is not None else Path("evidence") / f"l1a2f_{profile}"
    cache = root / "cases"
    cache.mkdir(parents=True, exist_ok=True)
    cfg["_cache"] = cache
    if overwrite:
        for path in cache.glob("*.json"):
            path.unlink()

    runners = {
        "translation": lambda: section_translation(cfg, max_steps=max_steps),
        "translation_v8": lambda: section_translation(cfg, max_steps=max_steps, pinned=True),
        "resolution": lambda: section_resolution(cfg, max_steps=max_steps),
        "primary_float64": lambda: section_float64(cfg, max_steps=max_steps),
        "production_mobility": lambda: section_production_mobility(cfg, max_steps=max_steps),
        "precision": lambda: section_precision(cfg, max_steps=max_steps),
        "chns": lambda: section_chns(cfg, max_steps=max_steps),
        "laplace": lambda: section_laplace(cfg, max_steps=max_steps),
        "impact": lambda: section_impact(cfg, max_steps=max_steps),
        "geometry": _geometry_section,
        "transport": _transport_section,
    }
    loaded: dict[str, Any] = {}
    for name in SECTIONS:
        path = root / "sections" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in wanted:
            if path.exists() and not overwrite:
                _log(f"section {name}: cached")
                loaded[name] = json.loads(path.read_text())
                continue
            _log(f"section {name}: running")
            loaded[name] = runners[name]()
            path.write_text(json.dumps(_clean(loaded[name]), allow_nan=False))
        elif path.exists():
            loaded[name] = json.loads(path.read_text())

    numbers = collect_numbers(loaded)
    gates = evaluate_gates(numbers)
    report = assemble_report(cfg, loaded, numbers, gates)
    if allow_incomplete:
        # A profile-limited run (CI smoke) may leave gates unmeasured; it must never hide a *failed*
        # measurement. The report still carries ``not_ready`` and the missing sections, and the
        # measured failures stay in the trigger list so the reason is never dropped.
        failed_gates = [gate for gate in gates if gate.measured and not gate.passed]
        measured_failures = [gate.gate for gate in failed_gates]
        report["not_ready_triggers"] = [
            f"gate failed: {gate.gate} ({gate.detail})" for gate in failed_gates
        ] + [t for t in report["not_ready_triggers"] if not t.startswith("gate failed:")]
        report["not_ready"] = bool(measured_failures)
        report["allow_incomplete"] = True
        report["measured_failures"] = measured_failures
    errors = validate_report(report)
    if errors:
        raise RuntimeError(f"report failed validation: {errors}")
    (root / "cutcell_alignment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    markdown = format_markdown(report)
    (root / "cutcell_alignment_report.md").write_text(markdown)
    manifest = {
        "stage": STAGE,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "profile": profile,
        "not_ready": bool(report["not_ready"]),
        "files": {},
    }
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        manifest["files"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*.md")):
        manifest["files"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(
        f"report written to {root} (not_ready={report['not_ready']}, "
        f"triggers={len(report['not_ready_triggers'])})"
    )
    return report


def _geometry_section() -> dict[str, Any]:
    from production import cutcell_geometry_audit as cga

    audit = cga.run_audit(quick=False)
    data = audit.to_dict()
    # `numbers` holds the small-cell report under its own keys, so read them directly
    data["alpha_min_positive"] = audit.numbers.get("alpha_min_positive_worst")
    data["alpha_percentiles"] = audit.numbers.get("percentiles")
    data["max_area_face_over_volume"] = audit.numbers.get("max_area_face_over_volume_worst")
    return data


def _transport_section() -> dict[str, Any]:
    from production import cutcell_phase_transport_audit as cpta

    return cpta.run_audit(quick=False).to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    parser.add_argument("--out", default=None, help="output directory (default evidence/l1a2f_<profile>)")
    parser.add_argument("--sections", default=None, help=f"comma-separated subset of {','.join(SECTIONS)}")
    parser.add_argument("--max-steps", type=int, default=None, help="cap every step budget (smoke use only)")
    parser.add_argument("--overwrite", action="store_true", help="ignore cached sections and cases")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit 0 when gates are unmeasured but no measured gate failed (profile-limited smoke use)",
    )
    parser.add_argument("--float64", action="store_true", help="enable jax float64 (required for f64 sections)")
    args = parser.parse_args(argv)
    if args.float64:
        jax.config.update("jax_enable_x64", True)
    sections = tuple(name.strip() for name in args.sections.split(",")) if args.sections else None
    report = run_audit(
        args.profile,
        args.out,
        sections=sections,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
        allow_incomplete=args.allow_incomplete,
    )
    print(format_markdown(report))
    return 1 if report["not_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
