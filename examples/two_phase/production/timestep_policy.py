"""Canonical deterministic timestep policies for the two-phase dataset generator.

Each policy is a pure, case-static function

    (case, N, requested_dt) -> effective_dt

with a recorded name/version and the limiting criterion/value that produced the
effective step (L1A-2p sections 12-14/22).  Policies never relax the existing
``pf.stable_dt`` cap and never exceed the requested dt.  The promoted default
for the production trajectory path is :data:`DEFAULT_POLICY_NAME`.

History
-------
``legacy_requested_v0``
    The contract-11 behaviour: ``dt = min(requested, stable_dt(N, u_max=2))``.
    Kept as the reproduction/reference policy; it reproduces the L1A-2o
    rejected trajectories exactly (same effective schedule, same fingerprints
    up to the recorded policy identity).
``fixed_cap_002_v1``
    Candidate A: ``dt <= 0.002`` regardless of resolution (control candidate;
    measured good on the flat impact matrix at N=192).
``impact_phase_cap_dx2_v1``
    Candidate B / promoted default: ``dt <= C * dx**2`` with the empirical
    coefficient ``C = 2.048`` calibrated on the L1A-2p measurements (the
    measured-good dt=0.002 at the failing resolution N=192, dx=1/32).  At
    N=192 the cap yields exactly dt=0.002; at N=144 it yields 0.00356
    (below the measured-passing 0.004, hence safe a fortiori).  The dx**2
    family is the empirically supported interpolation between the measured
    resolution pair (alpha=1 is falsified by the N=144 PASS at dt=0.004,
    alpha>=3 is contradicted by the measured dt=0.002 success at N=192);
    the criterion is an *empirical* impact/phase-robustness indicator, not a
    derived stability formula (L1A-2p section 7).
``cfl_multicriterion_v1``
    Candidate C: ``dt = min(requested, stable_dt, cutcell CFL cap, dx**2
    impact cap)``.  In the audited matrix the cut-cell advective CFL never
    binds (ratio 0.137 at the pre-failure impact state), so this collapses
    exactly onto ``impact_phase_cap_dx2_v1``; it is kept as the explicit
    multi-criterion form.
"""

from __future__ import annotations

from typing import Any

import phasefield as pf

POLICY_REGISTRY_VERSION = 1

#: empirical coefficient of the promoted resolution-scaled impact cap
#: (calibrated: C * dx(N=192)**2 == 0.002, the measured-good production step).
IMPACT_PHASE_DX2_COEFFICIENT = 2.048

#: Candidate A control cap (L1A-2p section 12).
FIXED_CAP_DT = 0.002

DEFAULT_POLICY_NAME = "impact_phase_cap_dx2_v1"


def policy_identity(name: str) -> dict[str, Any]:
    """Name/version record for lineage and fingerprints (section 22)."""

    versions = {
        "legacy_requested_v0": 0,
        "fixed_cap_002_v1": 1,
        "impact_phase_cap_dx2_v1": 1,
        "cfl_multicriterion_v1": 1,
    }
    if name not in versions:
        raise ValueError(f"unknown timestep policy {name!r}; expected one of {sorted(versions)}")
    return {"time_step_policy_name": name, "time_step_policy_version": versions[name]}


def effective_dt_for_case(case: dict, N: int, requested_dt: float, policy_name: str | None = None) -> dict[str, Any]:
    """Resolve the deterministic case-static effective dt and its limiting criterion."""

    name = policy_name or DEFAULT_POLICY_NAME
    identity = policy_identity(name)
    p0 = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dt=float(requested_dt))
    stable = float(pf.stable_dt(p0, u_max=2.0))
    dx = 6.0 / N
    requested = float(requested_dt)

    if name == "legacy_requested_v0":
        effective = min(requested, stable)
        criterion = "requested_dt" if requested <= stable else "stable_dt_cap"
        limiting_value = requested if requested <= stable else stable
        cap_value = None
    elif name == "fixed_cap_002_v1":
        effective = min(requested, stable, FIXED_CAP_DT)
        criterion, limiting_value = min(
            (("requested_dt", requested), ("stable_dt_cap", stable), ("fixed_cap_dt", FIXED_CAP_DT)),
            key=lambda item: item[1],
        )
        cap_value = FIXED_CAP_DT
    elif name == "impact_phase_cap_dx2_v1":
        cap_value = IMPACT_PHASE_DX2_COEFFICIENT * dx**2
        effective = min(requested, stable, cap_value)
        criterion, limiting_value = min(
            (("requested_dt", requested), ("stable_dt_cap", stable), ("impact_phase_dx2_cap", cap_value)),
            key=lambda item: item[1],
        )
    elif name == "cfl_multicriterion_v1":
        cap_value = IMPACT_PHASE_DX2_COEFFICIENT * dx**2
        effective = min(requested, stable, cap_value)
        criterion, limiting_value = min(
            (("requested_dt", requested), ("stable_dt_cap", stable), ("impact_phase_dx2_cap", cap_value)),
            key=lambda item: item[1],
        )
    else:  # pragma: no cover - policy_identity already validated the name
        raise ValueError(name)
    return {
        **identity,
        "requested_dt": requested,
        "effective_dt": float(effective),
        "limiting_criterion": criterion,
        "limiting_value": float(limiting_value),
        "stable_dt_cap": stable,
        "dx": dx,
        "policy_cap_value": cap_value,
        "registry_version": POLICY_REGISTRY_VERSION,
    }
