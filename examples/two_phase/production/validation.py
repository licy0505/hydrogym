"""Physics benchmarks for the existing JAX phase-field solver (no solver edits)."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import phasefield as pf
from production import observables as obs

KNOWN_SOLVER_BLOCKERS = [
    {
        "id": "P-VARDENS-PROJ",
        "severity": "high",
        "status": "open",
        "description": "Pressure projection is constant-coefficient despite variable density.",
    },
    {
        "id": "P-CAP-RHO",
        "severity": "high",
        "status": "open",
        "description": "Capillary acceleration is divided by rho_l instead of local rho(phi).",
    },
    {
        "id": "N-DT",
        "severity": "high",
        "status": "open",
        "description": "stable_dt currently enforces only advective CFL.",
    },
    {
        "id": "W-CONTACT-ANGLE",
        "severity": "high",
        "status": "measurement_required",
        "description": "Current wall-affinity model must be calibrated against measured apparent contact angle.",
    },
    {
        "id": "I-CONTACT-GAP",
        "severity": "high",
        "status": "measurement_required",
        "description": (
            "Impact drops stop at a finite gas-film gap (min g_0.5 ~ 0.148, min g_0.1 ~ 0.055 at We=100, "
            "gap ~ 0.15). L1A-2b only adds the one-factor solid/gas-film ablation evidence "
            "(production/solid_gas_film_audit.py); it does not resolve the gap."
        ),
    },
    {
        "id": "P-SOLID-PIN",
        "severity": "high",
        "status": "confirmed_problem",
        "description": (
            "The Cahn-Hilliard phase field uses a spatially constant mobility, so the solid is not "
            "impermeable to the phase flux: without the projection the drop leaks into the solid "
            "(~7 % mean liquid fraction in the geometric solid), and with enforce_solid_phi the "
            "mass-conserving projection re-injects the leaked liquid at the wall, which pumps the "
            "contact line. A neutral wall (cos_theta = 0, g_w == 0) still drifts from 90 deg to "
            ">120 deg and eventually detaches; the drift is unchanged for dt in [1e-3, 4e-3], "
            "eps in [2, 6] dx and N in [96, 192] and is independent of the wetting model "
            "(none / legacy_affinity / surface_energy). Sessile equilibrium angles cannot be measured "
            "until the wall carries a no-flux phase boundary condition; that is L1A-2c, not this PR."
        ),
    },
    {
        "id": "P-LAPLACE-SIGN",
        "severity": "high",
        "status": "measurement_required",
        "description": (
            "Static projection-pressure jump p_liquid - p_gas must be positive and scale as 1/R "
            "(phi=1 liquid). Corrected in solver contract v5; reported as resolved_in_contract_v5 only "
            "when the multi-radius baseline measures it (see LAPLACE_SIGN_CLOSURE_CRITERIA)."
        ),
    },
    {
        "id": "P-VARVISC",
        "severity": "medium",
        "status": "open",
        "description": "Viscous acceleration uses nu(phi)*lap(u) rather than divergence of variable-viscosity stress.",
    },
    {
        "id": "BC-Y-PERIODIC",
        "severity": "medium",
        "status": "open",
        "description": "The computational operators remain periodic in y.",
    },
]

# Evidence needed before P-LAPLACE-SIGN may be reported as ``resolved_in_contract_v5``.  These gate the
# *sign and 1/R scaling* only; the 5 % magnitude goal stays a provisional target (small radii are
# interface-resolution sensitive and are handled by the convergence study, not by a calibration factor).
LAPLACE_SIGN_CLOSURE_CRITERIA = {
    "require_all_ratios_positive": True,
    "min_radii": 3,  # a 1/R fit over fewer radii cannot establish scaling
    "min_r_squared": 0.99,  # delta_p vs 1/R
    "require_positive_slope": True,
}

PROVISIONAL_READINESS_TARGETS = {
    "mass_relative_drift": 0.001,
    "laplace_relative_error": 0.05,
    "contact_angle_mae_deg": 5.0,
    "contact_angle_max_error_deg": 10.0,
    "key_observable_refinement_change": 0.03,
}


def _dtype(name: str):
    if name == "float64":
        return jnp.float64
    if name != "float32":
        raise ValueError(f"Unsupported dtype {name!r}")
    return jnp.float32


def _params(
    N: int,
    *,
    Re: float,
    We: float,
    eps_factor: float = 1.5,
    eps: float | None = None,
    dt: float = 2.0e-3,
    dtype: str = "float32",
    wall_energy_amp: float = 5.0,
    enforce_solid_phi: bool = False,
    eta_pen_over_dt: float | None = None,
    nu_g_over_nu_l: float | None = None,
    wetting_model: str | None = None,
) -> pf.PhaseFieldParams:
    """Solver parameters for a benchmark.

    ``eta_pen_over_dt``, ``nu_g_over_nu_l`` and ``wetting_model`` are *diagnostic
    overrides* used by the ablation harnesses; leaving them as ``None`` keeps the
    production defaults bit-for-bit (``eta_pen = 2 dt``, ``nu_g = 10 nu_l``).
    """
    p = pf.PhaseFieldParams(
        Nx=N,
        Ny=N,
        Lx=6.0,
        Ly=6.0,
        Re=Re,
        We=We,
        dt=dt,
        wall_energy_amp=wall_energy_amp,
        enforce_solid_phi=enforce_solid_phi,
        dtype=_dtype(dtype),
    )
    p.eps = float(eps) if eps is not None else float(eps_factor) * p.dx
    p.dt = min(float(p.dt), float(pf.stable_dt(p, u_max=2.0)))
    if eta_pen_over_dt is not None:
        p.eta_pen = float(eta_pen_over_dt) * float(p.dt)
    if nu_g_over_nu_l is not None:
        p.nu_g = float(nu_g_over_nu_l) * float(p.nu_l)
    if wetting_model is not None:
        model = str(wetting_model)
        if model not in {"legacy_affinity", "surface_energy", "none"}:
            raise ValueError(f"unknown wetting_model {model!r}")
        if hasattr(p, "wetting_model"):
            p.wetting_model = model
        elif model == "legacy_affinity":
            pass  # the contract-v5 default semantics
        elif model == "none":
            p.wall_energy_amp = 0.0
        else:
            raise ValueError(
                "wetting_model='surface_energy' requires the surface-energy solver "
                "(SOLVER_CONTRACT_VERSION >= 6); none is available in this build"
            )
    return p


def _assert_state_finite(state: pf.State, N: int) -> None:
    for name in ("phi", "u", "v"):
        value = np.asarray(getattr(state, name))
        if value.shape != (N, N):
            raise ValueError(f"state.{name} has shape {value.shape}, expected {(N, N)}")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"state.{name} contains non-finite values")
    if not math.isfinite(float(state.t)):
        raise FloatingPointError("state.t is not finite")


def _max_speed(state: pf.State) -> float:
    return float(jax.device_get(jnp.max(jnp.sqrt(state.u * state.u + state.v * state.v))))


def _kinetic_energy(state: pf.State, p: pf.PhaseFieldParams) -> float:
    energy = jnp.sum(0.5 * pf.rho_of(state.phi, p) * (state.u**2 + state.v**2)) * p.dx * p.dy
    return float(jax.device_get(energy))


def _runtime_record(start: float, steps: int, N: int, dtype: str) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - start, 1e-12)
    return {
        "wall_seconds": float(elapsed),
        "steps": int(steps),
        "steps_per_second": float(steps / elapsed),
        "N": int(N),
        "dtype": dtype,
    }


def _record_diagnostics(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams) -> tuple[float, float, float]:
    _assert_state_finite(state, p.Nx)
    max_speed = _max_speed(state)
    kinetic_energy = _kinetic_energy(state, p)
    liquid = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    if not all(math.isfinite(v) for v in (max_speed, kinetic_energy, liquid)):
        raise FloatingPointError("non-finite static-droplet observable")
    return float(state.t), max_speed, kinetic_energy


@dataclass
class StaticDropletCase:
    R: float
    N: int
    dx: float
    eps: float
    eps_over_dx: float
    Cn: float
    steps: int
    save_every: int
    mass_initial: float
    mass_final: float
    mass_relative_drift: float
    max_speed_final: float
    max_speed_peak: float
    kinetic_energy_final: float
    pressure_diagnostic_method: str
    p_inside: float
    p_outside: float
    delta_p: float
    laplace_ratio: float
    finite: bool
    time_series: dict[str, list[float]]
    runtime: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_static_droplet_case(
    R: float,
    N: int = 192,
    steps: int = 1000,
    We: float = 100.0,
    Re: float = 200.0,
    eps_factor: float = 1.5,
    save_every: int = 100,
    dt: float = 2.0e-3,
    dtype: str = "float32",
    eps: float | None = None,
) -> StaticDropletCase:
    """Static circular drop, reconstructed projection-pressure Laplace diagnostic, and currents."""
    if R <= 0 or not math.isfinite(R) or N < 8 or steps <= 0 or save_every <= 0:
        raise ValueError("R, N, steps, and save_every must be positive and valid")
    p = _params(N, Re=Re, We=We, eps_factor=eps_factor, eps=eps, dt=dt, dtype=dtype)
    solid = pf.empty_solid(p)
    state = pf.droplet_initial_state(p, x0=p.Lx / 2.0, y0=p.Ly / 2.0, R=R, u_impact=0.0)
    _assert_state_finite(state, N)
    mass_initial = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    if mass_initial <= 0.0:
        raise ValueError("static droplet initial fluid-region mass must be positive")

    series = {"time": [], "max_speed": [], "kinetic_energy": []}
    t0 = time.perf_counter()

    def sample(current: pf.State) -> tuple[float, float, float]:
        t, speed, energy = _record_diagnostics(current, solid, p)
        series["time"].append(t)
        series["max_speed"].append(speed)
        series["kinetic_energy"].append(energy)
        return t, speed, energy

    _, max_speed_peak, _ = sample(state)
    step_fn = jax.jit(pf.step, static_argnums=(2,))
    for i in range(steps):
        state = step_fn(state, solid, p)
        # Track the speed peak at every solver step; retain the configured cadence
        # for the JSON time series and kinetic-energy diagnostics.
        _assert_state_finite(state, N)
        speed = _max_speed(state)
        max_speed_peak = max(max_speed_peak, speed)
        if (i + 1) % save_every == 0 or i + 1 == steps:
            _t, _speed, _energy = sample(state)
            max_speed_peak = max(max_speed_peak, _speed)

    mass_final = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    pressure = np.asarray(pf.pressure_field(state, solid, p), dtype=np.float64)
    X, Y = pf.grids(p)
    radius_field = np.sqrt((np.asarray(X) - p.Lx / 2.0) ** 2 + (np.asarray(Y) - p.Ly / 2.0) ** 2)
    inside_mask = radius_field < 0.3 * R
    outside_mask = radius_field > 2.5 * R
    if not inside_mask.any() or not outside_mask.any():
        raise ValueError("pressure probe regions are empty; choose a resolved radius away from domain boundaries")
    # Sign convention (contract v5, see phasefield.py and production/capillary_audit.py): phi = 1 is liquid,
    # so "inside" (r < 0.3 R) is the liquid core and "outside" (r > 2.5 R) is far gas.  delta_p is
    # P_liquid - P_gas and is expected to be +1/(We R) for a 2-D circle.  It is never sign-flipped here.
    p_inside = float(pressure[inside_mask].mean())
    p_outside = float(pressure[outside_mask].mean())
    delta_p = p_inside - p_outside
    laplace_ratio = delta_p * R * p.We
    max_speed_final = _max_speed(state)
    kinetic_energy_final = _kinetic_energy(state, p)
    relative_drift = abs(mass_final - mass_initial) / max(abs(mass_initial), 1e-12)
    runtime = _runtime_record(t0, steps, N, dtype)
    numeric = (
        R,
        p.dx,
        p.eps,
        p.eps / p.dx,
        p.eps / (2.0 * R),
        mass_initial,
        mass_final,
        relative_drift,
        max_speed_final,
        max_speed_peak,
        kinetic_energy_final,
        p_inside,
        p_outside,
        delta_p,
        laplace_ratio,
        *series["time"],
        *series["max_speed"],
        *series["kinetic_energy"],
    )
    finite = all(math.isfinite(float(v)) for v in numeric)
    if not finite:
        raise FloatingPointError("static-droplet benchmark produced non-finite metrics")
    return StaticDropletCase(
        R=float(R),
        N=int(N),
        dx=float(p.dx),
        eps=float(p.eps),
        eps_over_dx=float(p.eps / p.dx),
        Cn=float(p.eps / (2.0 * R)),
        steps=int(steps),
        save_every=int(save_every),
        mass_initial=mass_initial,
        mass_final=mass_final,
        mass_relative_drift=relative_drift,
        max_speed_final=max_speed_final,
        max_speed_peak=max_speed_peak,
        kinetic_energy_final=kinetic_energy_final,
        pressure_diagnostic_method="projection_reconstructed",
        p_inside=p_inside,
        p_outside=p_outside,
        delta_p=delta_p,
        laplace_ratio=laplace_ratio,
        finite=finite,
        time_series=series,
        runtime=runtime,
    )


def summarize_static_droplets(cases: list[StaticDropletCase]) -> dict[str, Any]:
    """Multi-radius error and least-squares fit delta_p = a/R + b."""
    if not cases:
        raise ValueError("static droplet summary requires at least one case")
    ratios = np.asarray([case.laplace_ratio for case in cases], dtype=np.float64)
    x = 1.0 / np.asarray([case.R for case in cases], dtype=np.float64)
    y = np.asarray([case.delta_p for case in cases], dtype=np.float64)
    if len(cases) >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, deg=1)
        residual = y - (slope * x + intercept)
        total = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 1e-24 else 1.0
    else:
        slope, intercept, r_squared = 0.0, float(y[0]), None
    errors = np.abs(ratios - 1.0)
    return {
        "mean_absolute_relative_error_from_ratio_1": float(errors.mean()),
        "max_absolute_relative_error_from_ratio_1": float(errors.max()),
        "slope_delta_p_vs_inv_R": float(slope),
        "intercept_delta_p_vs_inv_R": float(intercept),
        "r_squared": None if r_squared is None else float(r_squared),
        "fit_equation": "delta_p = slope * (1/R) + intercept",
        "fit_n_radii": len(cases),
    }


def run_static_droplet_suite(**kwargs: Any) -> dict[str, Any]:
    """Run all configured radii; preserve an error record for any failed case."""
    radii = kwargs.pop("radii")
    records: list[dict[str, Any]] = []
    successful: list[StaticDropletCase] = []
    for radius in radii:
        try:
            case = run_static_droplet_case(R=float(radius), **kwargs)
            records.append(case.to_dict())
            successful.append(case)
        except Exception as exc:  # retain a machine-readable fail-closed record
            records.append({"R": float(radius), "finite": False, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "summary": summarize_static_droplets(successful) if successful else None,
        "cases": records,
    }


@dataclass
class ContactAngleCase:
    """One sessile relaxation.

    ``measured_deg`` (and the signed/absolute error) is only set when the run met
    the equilibrium criterion; a non-converged value is *not* an equilibrium
    contact angle and is reported as ``None``.  ``final_sampled_deg`` keeps the
    last sampled angle for diagnostics, and ``samples`` records the time,
    measured angle, maximum speed, total mass and fluid-region mass of every
    sample.
    """

    target_deg: float
    measured_deg: float | None
    signed_error_deg: float | None
    absolute_error_deg: float | None
    converged: bool
    converged_step: int | None
    converged_time: float | None
    final_sampled_deg: float | None
    mass_initial: float
    mass_final: float
    mass_relative_drift: float
    total_mass_initial: float
    total_mass_final: float
    total_mass_relative_drift: float
    initial_solid_liquid_fraction: float
    wetting_model: str
    enforce_solid_phi: bool
    relaxation_steps: int
    relaxation_time: float
    samples: list[dict[str, Any]]
    finite: bool
    runtime: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_contact_angle_case(
    target_deg: float,
    N: int = 192,
    relaxation_steps: int = 3000,
    Re: float = 200.0,
    We: float = 100.0,
    eps_factor: float = 1.5,
    wall_energy_amp: float = 5.0,
    R: float = 1.1,
    dt: float = 4.0e-3,
    dtype: str = "float32",
    eps: float | None = None,
    *,
    wetting_model: str = "surface_energy",
    enforce_solid_phi: bool = True,
    sample_every: int = 200,
    angle_tol_deg: float = 0.25,
    speed_tol: float = 5e-4,
    windows: int = 3,
    max_steps: int | None = None,
) -> ContactAngleCase:
    """Relax a clean sessile drop on a flat wall and measure the apparent angle.

    The initial state is target-independent (:func:`phasefield.sessile_initial_state`,
    geometric 90 deg cap with no liquid in the geometric solid) and ``u = v = 0``.
    The run stops as soon as ``windows`` consecutive samples have both
    ``|d(theta)| <= angle_tol_deg`` and ``max_speed <= speed_tol``; only then is the
    angle reported as an equilibrium contact angle.  ``wall_energy_amp`` is a
    legacy-only parameter (``wetting_model='legacy_affinity'``) and has no effect
    in the default ``surface_energy`` model.
    """
    if not (0.0 <= target_deg <= 180.0) or not math.isfinite(target_deg):
        raise ValueError("target_deg must lie in [0, 180]")
    budget = int(relaxation_steps if max_steps is None else max_steps)
    if budget < 0 or N < 8 or R <= 0:
        raise ValueError("max_steps (or relaxation_steps) must be non-negative; N and R must be positive")
    if int(windows) < 1 or int(sample_every) < 1:
        raise ValueError("windows and sample_every must be positive")
    if not (float(angle_tol_deg) > 0.0 and float(speed_tol) > 0.0):
        raise ValueError("angle_tol_deg and speed_tol must be positive")
    benchmark_started = time.perf_counter()
    p = _params(
        N,
        Re=Re,
        We=We,
        eps_factor=eps_factor,
        eps=eps,
        dt=dt,
        dtype=dtype,
        wall_energy_amp=wall_energy_amp,
        enforce_solid_phi=bool(enforce_solid_phi),
        wetting_model=wetting_model,
    )
    wall_height = 0.25
    sdf = pf.surface_flat(p, wall_height=wall_height)
    solid = pf.make_solid(sdf, p, cos_theta=math.cos(math.radians(target_deg)))
    # Clean, target-independent sessile seed (contract v6): geometric 90 deg cap with
    # its centre on the wall plane, phi only in sdf >= 0, u = v = 0.
    state = pf.sessile_initial_state(p, solid, R=R, wall_height=wall_height)
    solid_cells = jnp.asarray(solid.sdf < 0.0, dtype=p.dtype)
    solid_liquid = float(jnp.sum(jnp.where(solid_cells > 0.0, state.phi, 0.0)) / max(float(jnp.sum(solid_cells)), 1.0))
    _assert_state_finite(state, N)
    mass_initial = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    total_initial = float(jnp.sum(state.phi) * p.dx * p.dy)
    if mass_initial <= 0:
        raise ValueError("contact-angle initial fluid-region mass must be positive")
    step_fn = jax.jit(pf.step, static_argnums=(2,))
    every = max(1, min(int(sample_every), max(budget, 1)))
    relaxation_started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    converged = False
    converged_step = None
    applied = 0
    while applied < budget:
        state = step_fn(state, solid, p)
        applied += 1
        if applied % every:
            continue
        angle = float(pf.measure_contact_angle(state.phi, solid, p))
        max_speed = float(jnp.sqrt(jnp.max(state.u**2 + state.v**2)))
        total_mass = float(jnp.sum(state.phi) * p.dx * p.dy)
        fluid_mass = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
        samples.append(
            {
                "step": int(applied),
                "time": float(applied * p.dt),
                "measured_angle_deg": angle,
                "max_speed": max_speed,
                "total_mass": total_mass,
                "fluid_mass": fluid_mass,
            }
        )
        if len(samples) >= int(windows):
            window = samples[-int(windows) :]
            angles = [float(row["measured_angle_deg"]) for row in window]
            speeds = [float(row["max_speed"]) for row in window]
            calm = all(abs(a - b) <= float(angle_tol_deg) for a, b in zip(angles, angles[1:]))
            slow = all(speed <= float(speed_tol) for speed in speeds)
            if calm and slow:
                converged = True
                converged_step = int(applied)
                break
        if not math.isfinite(angle):
            break
    elapsed = time.perf_counter() - relaxation_started
    _assert_state_finite(state, N)
    mass_final = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    total_final = float(jnp.sum(state.phi) * p.dx * p.dy)
    final_angle = float(samples[-1]["measured_angle_deg"]) if samples else None
    wall_seconds = time.perf_counter() - benchmark_started
    measured = float(samples[-1]["measured_angle_deg"]) if converged and samples else None
    finite = final_angle is not None and math.isfinite(final_angle) and 0.0 <= final_angle <= 180.0
    drift = abs(mass_final - mass_initial) / max(abs(mass_initial), 1e-12)
    total_drift = abs(total_final - total_initial) / max(abs(total_initial), 1e-12)
    runtime = {
        "wall_seconds": float(wall_seconds),
        "steps": int(applied),
        "steps_per_second": float(applied / max(elapsed, 1e-12)),
        "N": int(N),
        "dtype": dtype,
        "sample_every": int(every),
        "max_steps": int(budget),
        "convergence": {
            "angle_tol_deg": float(angle_tol_deg),
            "speed_tol": float(speed_tol),
            "windows": int(windows),
        },
    }
    return ContactAngleCase(
        target_deg=float(target_deg),
        measured_deg=measured,
        signed_error_deg=(measured - target_deg) if measured is not None else None,
        absolute_error_deg=abs(measured - target_deg) if measured is not None else None,
        converged=bool(converged),
        converged_step=converged_step,
        converged_time=(converged_step * float(p.dt)) if converged_step is not None else None,
        final_sampled_deg=final_angle,
        mass_initial=float(mass_initial),
        mass_final=float(mass_final),
        mass_relative_drift=float(drift),
        total_mass_initial=float(total_initial),
        total_mass_final=float(total_final),
        total_mass_relative_drift=float(total_drift),
        initial_solid_liquid_fraction=float(solid_liquid),
        wetting_model=str(p.wetting_model),
        enforce_solid_phi=bool(p.enforce_solid_phi),
        relaxation_steps=int(applied),
        relaxation_time=float(elapsed),
        samples=samples,
        finite=bool(finite),
        runtime=runtime,
    )


def summarize_contact_angles(cases: list[ContactAngleCase]) -> dict[str, Any]:
    """Statistics over converged cases only.

    The MAE/RMSE/max error are computed from equilibrium angles, i.e. runs that met
    the convergence criterion.  Non-converged runs are counted and their last
    sampled angles are reported separately; they must never enter the error
    statistics (a drifting angle is not an equilibrium contact angle).
    """
    valid = sorted((case for case in cases if case.finite), key=lambda case: case.target_deg)
    converged = [case for case in valid if case.converged and case.measured_deg is not None]
    errors = np.asarray([case.absolute_error_deg for case in converged], dtype=np.float64)
    measurements = [float(case.measured_deg) for case in converged]
    monotonic = all(b >= a for a, b in zip(measurements, measurements[1:]))
    return {
        "mae_deg": float(errors.mean()) if errors.size else None,
        "rmse_deg": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
        "max_absolute_error_deg": float(errors.max()) if errors.size else None,
        "monotonic_target_to_measured": bool(monotonic) if len(converged) == len(cases) else False,
        "valid_case_count": len(valid),
        "case_count": len(cases),
        "converged_case_count": len(converged),
        "converged_targets": [float(case.target_deg) for case in converged],
        "non_converged_targets": [float(case.target_deg) for case in valid if not case.converged],
        "final_angles_deg": [
            {"target_deg": float(case.target_deg), "final_sampled_deg": case.final_sampled_deg} for case in valid
        ],
    }


def run_contact_angle_suite(**kwargs: Any) -> dict[str, Any]:
    targets = kwargs.pop("targets")
    records: list[dict[str, Any]] = []
    successful: list[ContactAngleCase] = []
    for target in targets:
        try:
            case = run_contact_angle_case(target_deg=float(target), **kwargs)
            records.append(case.to_dict())
            successful.append(case)
        except Exception as exc:
            records.append({"target_deg": float(target), "finite": False, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "summary": summarize_contact_angles(successful)
        if successful
        else {
            "mae_deg": None,
            "rmse_deg": None,
            "max_absolute_error_deg": None,
            "monotonic_target_to_measured": False,
            "valid_case_count": 0,
            "case_count": 0,
        },
        "cases": records,
    }


def _impact_params(
    case: dict[str, Any],
    N: int,
    eps_factor: float,
    dt: float,
    dtype: str,
    eps: float | None = None,
) -> tuple[pf.PhaseFieldParams, pf.Solid, pf.State, float]:
    Re = float(case.get("Re", 200.0))
    We = float(case.get("We", 100.0))
    R = float(case.get("R", 0.7))
    if min(Re, We, R) <= 0:
        raise ValueError("impact Re, We, and R must be positive")
    p = _params(
        N,
        Re=Re,
        We=We,
        eps_factor=eps_factor,
        eps=eps,
        dt=dt,
        dtype=dtype,
        enforce_solid_phi=bool(case.get("enforce_solid_phi", True)),
        eta_pen_over_dt=case.get("eta_pen_over_dt"),
        nu_g_over_nu_l=case.get("nu_g_over_nu_l"),
        wetting_model=case.get("wetting_model"),
    )
    wall_height = float(case.get("wall_height", 0.25))
    sdf = pf.surface_flat(p, wall_height=wall_height)
    cos_theta = float(case.get("cos_theta", 0.0))
    solid = pf.make_solid(sdf, p, cos_theta=cos_theta)
    x0 = float(case.get("x0", p.Lx / 2.0))
    u_impact = float(case.get("u_impact", 1.0))
    requested_gap = float(case.get("impact_gap", 0.1))
    # Preserve the existing solver's diffuse-interface clearance convention.
    gap = max(requested_gap, 2.0 * float(p.eps), 0.05)
    y0 = wall_height + R + gap
    velocity_mode = str(case.get("velocity_mode", "streamfunction"))
    state = pf.droplet_initial_state(
        p,
        x0=x0,
        y0=y0,
        R=R,
        u_impact=u_impact,
        velocity_mode=velocity_mode,
    )
    return p, solid, state, cos_theta


def run_impact_case(
    case_cfg: dict[str, Any],
    *,
    N: int = 192,
    eps_factor: float = 1.5,
    steps: int = 1000,
    save_every: int = 10,
    dt: float = 2.0e-3,
    dtype: str = "float32",
    eps: float | None = None,
) -> dict[str, Any]:
    """Run a flat-wall trajectory and record periodic-safe observables at saved frames."""
    if steps <= 0 or save_every <= 0:
        raise ValueError("steps and save_every must be positive")
    p, solid, state, cos_theta = _impact_params(case_cfg, N, eps_factor, dt, dtype, eps=eps)
    _assert_state_finite(state, N)
    R = float(case_cfg.get("R", 0.7))
    D0 = 2.0 * R
    mass_initial = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    if mass_initial <= 0:
        raise ValueError("impact initial fluid-region mass must be positive")
    X, Y = pf.grids(p)
    X_np, Y_np, sdf_np = np.asarray(X), np.asarray(Y), np.asarray(solid.sdf)

    names = (
        "time",
        "mass",
        "total_phase_mass",
        "x_cm",
        "y_cm",
        "width",
        "beta",
        "g_0.5",
        "g_0.1",
        "top_height",
        "bottom_height",
        "max_speed",
        "contact_signal",
    )
    series: dict[str, list[Any]] = {name: [] for name in names}

    def sample(current: pf.State) -> None:
        _assert_state_finite(current, N)
        phi_np = np.asarray(current.phi)
        mass = obs.fluid_phase_mass(phi_np, sdf_np, p.dx, p.dy)
        total_mass = obs.total_phase_mass(phi_np, p.dx, p.dy)
        x_cm, y_cm = obs.center_of_mass(phi_np, sdf_np, X_np, Y_np)
        width = obs.periodic_spreading_width(phi_np, 0.5, p.dx, p.Lx)
        gap05 = obs.bottom_gap(phi_np, sdf_np, 0.5)
        gap01 = obs.bottom_gap(phi_np, sdf_np, 0.1)
        top = obs.top_height(phi_np, sdf_np, Y_np[0, :], 0.5)
        bottom = obs.bottom_height(phi_np, sdf_np, Y_np[0, :], 0.5)
        contact = bool(gap05 <= 1.5 * p.dx)
        values = {
            "time": float(current.t),
            "mass": float(mass),
            "total_phase_mass": float(total_mass),
            "x_cm": float(x_cm),
            "y_cm": float(y_cm),
            "width": float(width),
            "beta": obs.beta_from_width(width, R),
            "g_0.5": float(gap05),
            "g_0.1": float(gap01),
            "top_height": float(top),
            "bottom_height": float(bottom),
            "max_speed": _max_speed(current),
            "contact_signal": contact,
        }
        if not all(math.isfinite(value) for key, value in values.items() if key != "contact_signal"):
            raise FloatingPointError("impact observable contains non-finite value")
        for name, value in values.items():
            series[name].append(value)

    started = time.perf_counter()
    sample(state)
    step_fn = jax.jit(pf.step, static_argnums=(2,))
    for i in range(steps):
        state = step_fn(state, solid, p)
        if (i + 1) % save_every == 0 or i + 1 == steps:
            sample(state)
    runtime = _runtime_record(started, steps, N, dtype)
    mass_final = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    beta_values = series["beta"]
    max_index = int(np.argmax(beta_values))
    contact_times = [float(t) for t, contacted in zip(series["time"], series["contact_signal"]) if contacted]
    first_contact = contact_times[0] if contact_times else None
    last_contact = contact_times[-1] if contact_times else None
    detachment = False
    if contact_times:
        last_index = max(i for i, value in enumerate(series["contact_signal"]) if value)
        detachment = any(not value for value in series["contact_signal"][last_index + 1 :])
    return {
        "case_name": str(case_cfg.get("name", "impact")),
        "We": float(case_cfg.get("We", 100.0)),
        "Re": float(case_cfg.get("Re", 200.0)),
        "Oh_derived": float(math.sqrt(float(case_cfg.get("We", 100.0))) / float(case_cfg.get("Re", 200.0))),
        "R": R,
        "D0": D0,
        "theta_deg": float(math.degrees(math.acos(cos_theta))),
        "cos_theta": float(cos_theta),
        "surface": "flat",
        "velocity_mode": str(case_cfg.get("velocity_mode", "streamfunction")),
        "eps": float(p.eps),
        "eps_over_dx": float(p.eps / p.dx),
        "Cn": float(p.eps / D0),
        "M": float(p.M),
        "rho_ratio": float(p.rho_l / p.rho_g),
        "nu_ratio": float(p.nu_l / p.nu_g),
        "dt": float(p.dt),
        "N": int(N),
        "Nx": int(p.Nx),
        "Ny": int(p.Ny),
        "mass_initial": float(mass_initial),
        "mass_final": float(mass_final),
        "mass_drift": float(abs(mass_final - mass_initial) / max(abs(mass_initial), 1e-12)),
        "beta_max": float(beta_values[max_index]),
        "time_to_beta_max": float(series["time"][max_index]),
        "first_contact_time": first_contact,
        "last_contact_time_if_observed": last_contact,
        "detachment_observed": bool(detachment),
        "minimum_gap": float(min(series["g_0.5"])),
        "minimum_gap_0.5": float(min(series["g_0.5"])),
        "minimum_gap_0.1": float(min(series["g_0.1"])),
        "peak_speed": float(max(series["max_speed"])),
        "eta_pen_over_dt": float(p.eta_pen / p.dt),
        "nu_g_over_nu_l": float(p.nu_g / p.nu_l),
        "wetting_model": str(getattr(p, "wetting_model", "legacy_affinity")),
        "enforce_solid_phi": bool(p.enforce_solid_phi),
        "final_y_cm": float(series["y_cm"][-1]),
        "time_series": series,
        "runtime": runtime,
        "finite": True,
    }


def run_impact_suite(
    cases: list[dict[str, Any]],
    *,
    N: int = 192,
    eps_factor: float = 1.5,
    steps: int = 1000,
    save_every: int = 10,
    dt: float = 2.0e-3,
    dtype: str = "float32",
) -> dict[str, Any]:
    records = []
    for case in cases:
        try:
            records.append(
                run_impact_case(
                    case,
                    N=N,
                    eps_factor=eps_factor,
                    steps=steps,
                    save_every=save_every,
                    dt=dt,
                    dtype=dtype,
                )
            )
        except Exception as exc:
            records.append(
                {
                    "case_name": str(case.get("name", "impact")),
                    "finite": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {"cases": records}


def run_convergence_point(case: dict[str, Any], *, dtype: str = "float32") -> dict[str, Any]:
    """Run static, optional sessile-angle, and optional impact metrics at one resolution."""
    started = time.perf_counter()
    N = int(case["N"])
    eps = case.get("eps")
    eps_factor = float(case.get("eps_factor", 1.5))
    static = run_static_droplet_case(
        R=float(case.get("R", 0.7)),
        N=N,
        steps=int(case.get("steps", 300)),
        We=float(case.get("We", 100.0)),
        Re=float(case.get("Re", 200.0)),
        eps_factor=eps_factor,
        eps=eps,
        save_every=int(case.get("save_every", 50)),
        dt=float(case.get("dt", 2.0e-3)),
        dtype=dtype,
    )
    metrics: dict[str, Any] = {
        "laplace_ratio": static.laplace_ratio,
        "mass_drift_static": static.mass_relative_drift,
        "spurious_current_peak": static.max_speed_peak,
        "spurious_current_final": static.max_speed_final,
        "static_wall_seconds": static.runtime["wall_seconds"],
        "mass_drift": static.mass_relative_drift,
    }
    if "target_deg" in case:
        contact = run_contact_angle_case(
            target_deg=float(case["target_deg"]),
            N=N,
            relaxation_steps=int(case.get("relaxation_steps", 300)),
            Re=float(case.get("Re", 200.0)),
            We=float(case.get("We", 100.0)),
            eps_factor=eps_factor,
            eps=eps,
            R=float(case.get("R", 0.7)),
            dt=float(case.get("dt", 2.0e-3)),
            dtype=dtype,
        )
        if not contact.finite:
            raise FloatingPointError("convergence contact-angle measurement is non-finite or out of range")
        metrics["contact_angle_deg"] = contact.measured_deg
        metrics["contact_angle_mass_drift"] = contact.mass_relative_drift
        metrics["contact_angle_wall_seconds"] = contact.relaxation_time
    if "impact_steps" in case:
        impact = run_impact_case(
            {
                "name": "convergence-impact",
                "We": float(case.get("We", 100.0)),
                "Re": float(case.get("Re", 200.0)),
                "R": float(case.get("R", 0.7)),
                "cos_theta": float(case.get("cos_theta", 0.0)),
                "impact_gap": float(case.get("impact_gap", 0.1)),
                "velocity_mode": str(case.get("velocity_mode", "streamfunction")),
            },
            N=N,
            eps_factor=eps_factor,
            eps=eps,
            steps=int(case["impact_steps"]),
            save_every=int(case.get("save_every", 50)),
            dt=float(case.get("dt", 2.0e-3)),
            dtype=dtype,
        )
        metrics["beta_max"] = impact["beta_max"]
        metrics["impact_mass_drift"] = impact["mass_drift"]
        metrics["impact_wall_seconds"] = impact["runtime"]["wall_seconds"]
    metrics["wall_seconds"] = float(time.perf_counter() - started)
    metrics["dx"] = static.dx
    metrics["eps"] = static.eps
    metrics["eps_over_dx"] = static.eps_over_dx
    metrics["Cn"] = static.Cn
    return metrics
