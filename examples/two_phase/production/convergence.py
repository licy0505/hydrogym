"""Reusable grid and interface-thickness sweep runners and convergence summaries."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable


MetricFunction = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ConvergencePoint:
    label: str
    parameters: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConvergenceStudy:
    study_type: str
    description: str
    points: list[ConvergencePoint]
    relative_changes: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_type": self.study_type,
            "description": self.description,
            "points": [point.to_dict() for point in self.points],
            "relative_changes": self.relative_changes,
        }


def adjacent_relative_changes(values: list[float], denominator_epsilon: float = 1e-12) -> list[float]:
    """Return |Q_fine-Q_coarse| / max(|Q_fine|, epsilon) for adjacent points."""
    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError("denominator_epsilon must be finite and positive")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("convergence values must be finite")
    return [
        abs(float(fine) - float(coarse)) / max(abs(float(fine)), denominator_epsilon)
        for coarse, fine in zip(values, values[1:])
    ]


def _with_metrics(points: list[ConvergencePoint]) -> dict[str, list[float]]:
    if len(points) < 2:
        return {}
    metric_names = set.intersection(*(set(point.metrics) for point in points))
    changes: dict[str, list[float]] = {}
    for name in sorted(metric_names):
        values = [point.metrics[name] for point in points]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            numeric_values = [float(value) for value in values]
            if all(math.isfinite(value) for value in numeric_values):
                changes[name] = adjacent_relative_changes(numeric_values)
    return changes


def _resolution(case: dict[str, Any], N: int) -> dict[str, Any]:
    Lx = float(case.get("Lx", 6.0))
    radius = float(case.get("R", 0.7))
    if "eps" in case:
        eps = float(case["eps"])
    else:
        eps = float(case.get("eps_factor", 1.5)) * Lx / N
    return {
        "N": N,
        "dx": Lx / N,
        "eps": eps,
        "eps_over_dx": eps / (Lx / N),
        "Cn": eps / (2.0 * radius),
    }


def run_grid_refinement_study(
    base_case: dict[str, Any],
    N_values: list[int],
    benchmark_fn: MetricFunction,
    *,
    eps_factor: float | None = None,
    eps_mode: str = "coupled",
    fixed_eps: float | None = None,
) -> ConvergenceStudy:
    """Run a resolution sweep.

    ``coupled`` holds eps/dx (eps_factor) fixed, so it couples grid refinement
    with a changing physical interface thickness. ``fixed_physical`` holds eps
    fixed and therefore provides a fixed-Cn grid study for fixed droplet diameter.
    """
    if len(N_values) < 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in N_values):
        raise ValueError("N_values must contain at least two positive integer resolutions")
    if N_values != sorted(set(N_values)):
        raise ValueError("N_values must be strictly increasing")
    if eps_mode not in {"coupled", "fixed_physical"}:
        raise ValueError("eps_mode must be 'coupled' or 'fixed_physical'")
    if eps_mode == "coupled":
        factor = float(eps_factor if eps_factor is not None else base_case.get("eps_factor", 1.5))
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("eps_factor must be finite and positive")
    else:
        if fixed_eps is None or not math.isfinite(float(fixed_eps)) or float(fixed_eps) <= 0:
            raise ValueError("fixed_eps must be finite and positive for a fixed-physical-eps study")

    points = []
    for N in N_values:
        case = dict(base_case)
        case["N"] = N
        if eps_mode == "coupled":
            case.pop("eps", None)
            case["eps_factor"] = factor
        else:
            case.pop("eps_factor", None)
            case["eps"] = float(fixed_eps)
        resolution = _resolution(case, N)
        metrics = dict(benchmark_fn(case))
        points.append(ConvergencePoint(f"N={N}", {**case, **resolution}, metrics))

    if eps_mode == "coupled":
        study_type = "coupled_grid_interface_refinement"
        description = (
            f"Coupled grid/interface refinement: eps/dx={factor:g} is fixed, so physical eps "
            "and Cn decrease as N increases."
        )
    else:
        study_type = "fixed_physical_eps_grid_refinement"
        description = (
            f"Grid refinement at fixed physical eps={float(fixed_eps):g} "
            "(fixed Cn for fixed droplet diameter); eps/dx increases with N."
        )
    return ConvergenceStudy(study_type, description, points, _with_metrics(points))


def run_interface_thickness_study(
    base_case: dict[str, Any],
    eps_factors: list[float],
    benchmark_fn: MetricFunction,
    *,
    N: int,
) -> ConvergenceStudy:
    """Vary eps/dx at fixed grid resolution, keeping grid spacing constant."""
    if len(eps_factors) < 2 or any(not math.isfinite(float(value)) or float(value) <= 0 for value in eps_factors):
        raise ValueError("eps_factors must contain at least two finite positive values")
    if N <= 0:
        raise ValueError("N must be positive")
    if list(eps_factors) != sorted(set(eps_factors)):
        raise ValueError("eps_factors must be strictly increasing")

    points = []
    for factor in eps_factors:
        case = dict(base_case)
        case["N"] = N
        case.pop("eps", None)
        case["eps_factor"] = float(factor)
        resolution = _resolution(case, N)
        metrics = dict(benchmark_fn(case))
        points.append(ConvergencePoint(f"eps_factor={factor:g}", {**case, **resolution}, metrics))
    description = (
        f"Interface-thickness sensitivity at fixed N={N}; physical eps and eps/dx vary, grid spacing is fixed."
    )
    return ConvergenceStudy("interface_thickness_sensitivity", description, points, _with_metrics(points))
