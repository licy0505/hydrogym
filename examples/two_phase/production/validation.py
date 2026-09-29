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
        "id": "P-LAPLACE-SIGN",
        "severity": "high",
        "status": "measurement_required",
        "description": "Static projection-pressure Laplace response must have the expected positive pressure jump.",
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
) -> pf.PhaseFieldParams:
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
    target_deg: float
    measured_deg: float | None
    signed_error_deg: float | None
    absolute_error_deg: float | None
    mass_initial: float
    mass_final: float
    mass_relative_drift: float
    relaxation_steps: int
    relaxation_time: float
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
) -> ContactAngleCase:
    """Relax a sessile drop on a flat affinity wall and measure apparent angle."""
    if not (0.0 <= target_deg <= 180.0) or not math.isfinite(target_deg):
        raise ValueError("target_deg must lie in [0, 180]")
    if relaxation_steps < 0 or N < 8 or R <= 0:
        raise ValueError("relaxation_steps must be non-negative; N and R must be positive")
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
        enforce_solid_phi=False,
    )
    wall_height = 0.25
    sdf = pf.surface_flat(p, wall_height=wall_height)
    solid = pf.make_solid(sdf, p, cos_theta=math.cos(math.radians(target_deg)))
    # Seed a sessile drop with a small geometric overlap, matching the legacy
    # manual test; no wetting parameter is tuned by this benchmark.
    y0 = wall_height + R - min(0.15, 0.14 * R)
    state = pf.droplet_initial_state(p, x0=p.Lx / 2.0, y0=y0, R=R, u_impact=0.0)
    _assert_state_finite(state, N)
    mass_initial = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    if mass_initial <= 0:
        raise ValueError("contact-angle initial fluid-region mass must be positive")
    step_fn = jax.jit(pf.step, static_argnums=(2,))
    relaxation_started = time.perf_counter()
    for _ in range(relaxation_steps):
        state = step_fn(state, solid, p)
    _assert_state_finite(state, N)
    elapsed = time.perf_counter() - relaxation_started
    mass_final = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    measured = float(pf.measure_contact_angle(state.phi, solid, p))
    wall_seconds = time.perf_counter() - benchmark_started
    finite = math.isfinite(measured) and 0.0 <= measured <= 180.0
    drift = abs(mass_final - mass_initial) / max(abs(mass_initial), 1e-12)
    runtime = {
        "wall_seconds": float(wall_seconds),
        "steps": int(relaxation_steps),
        "steps_per_second": float(relaxation_steps / max(wall_seconds, 1e-12)),
        "N": int(N),
        "dtype": dtype,
    }
    return ContactAngleCase(
        target_deg=float(target_deg),
        measured_deg=measured if finite else None,
        signed_error_deg=(measured - target_deg) if finite else None,
        absolute_error_deg=abs(measured - target_deg) if finite else None,
        mass_initial=float(mass_initial),
        mass_final=float(mass_final),
        mass_relative_drift=float(drift),
        relaxation_steps=int(relaxation_steps),
        relaxation_time=float(elapsed),
        finite=bool(finite),
        runtime=runtime,
    )


def summarize_contact_angles(cases: list[ContactAngleCase]) -> dict[str, Any]:
    valid = sorted((case for case in cases if case.finite), key=lambda case: case.target_deg)
    errors = np.asarray([case.absolute_error_deg for case in valid], dtype=np.float64)
    measurements = [float(case.measured_deg) for case in valid]
    monotonic = all(b >= a for a, b in zip(measurements, measurements[1:]))
    return {
        "mae_deg": float(errors.mean()) if errors.size else None,
        "rmse_deg": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
        "max_absolute_error_deg": float(errors.max()) if errors.size else None,
        "monotonic_target_to_measured": bool(monotonic) if len(valid) == len(cases) else False,
        "valid_case_count": len(valid),
        "case_count": len(cases),
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
        enforce_solid_phi=True,
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
