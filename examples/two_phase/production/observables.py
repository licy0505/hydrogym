"""Pure NumPy observables shared by validation, datasets, and later model evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _field(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional field; got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _matching_fields(phi: Any, sdf: Any) -> tuple[np.ndarray, np.ndarray]:
    phase = _field(phi, "phi")
    distance = _field(sdf, "sdf")
    if phase.shape != distance.shape:
        raise ValueError(f"phi and sdf shapes differ: {phase.shape} != {distance.shape}")
    return phase, distance


def total_phase_mass(phi: Any, dx: float, dy: float) -> float:
    """Integral of ``phi`` over every cell, including cells geometrically in solid."""
    phase = _field(phi, "phi")
    if not (math.isfinite(dx) and math.isfinite(dy) and dx > 0 and dy > 0):
        raise ValueError("dx and dy must be finite and positive")
    return float(np.sum(phase, dtype=np.float64) * dx * dy)


def fluid_phase_mass(phi: Any, sdf: Any, dx: float, dy: float) -> float:
    """Integral of ``phi`` over geometric fluid cells (``sdf >= 0``)."""
    phase, distance = _matching_fields(phi, sdf)
    if not (math.isfinite(dx) and math.isfinite(dy) and dx > 0 and dy > 0):
        raise ValueError("dx and dy must be finite and positive")
    return float(np.sum(phase[distance >= 0.0], dtype=np.float64) * dx * dy)


def liquid_mass(phi: Any, sdf: Any, dx: float, dy: float) -> float:
    """Compatibility name for fluid-region phase mass; not total phase mass."""
    return fluid_phase_mass(phi, sdf, dx, dy)


def solid_phase_mass(phi: Any, sdf: Any, dx: float, dy: float) -> float:
    """Integral of ``phi`` over geometric solid cells (``sdf < 0``)."""
    phase, distance = _matching_fields(phi, sdf)
    if not (math.isfinite(dx) and math.isfinite(dy) and dx > 0 and dy > 0):
        raise ValueError("dx and dy must be finite and positive")
    return float(np.sum(phase[distance < 0.0], dtype=np.float64) * dx * dy)


def _coordinate_axes(x: Any, y: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    x_array, y_array = np.asarray(x), np.asarray(y)
    nx, ny = shape
    if x_array.ndim == 1 and x_array.shape == (nx,):
        x_axis = x_array
    elif x_array.ndim == 2 and x_array.shape == shape:
        x_axis = x_array[:, 0]
    else:
        raise ValueError(f"x coordinates must have shape ({nx},) or {shape}; got {x_array.shape}")
    if y_array.ndim == 1 and y_array.shape == (ny,):
        y_axis = y_array
    elif y_array.ndim == 2 and y_array.shape == shape:
        y_axis = y_array[0, :]
    else:
        raise ValueError(f"y coordinates must have shape ({ny},) or {shape}; got {y_array.shape}")
    if not np.isfinite(x_axis).all() or not np.isfinite(y_axis).all():
        raise ValueError("coordinates must be finite")
    if nx < 2 or ny < 1:
        raise ValueError("grid must contain at least two x cells and one y cell")
    return x_axis.astype(np.float64), y_axis.astype(np.float64)


def center_of_mass(phi: Any, sdf: Any, x: Any, y: Any) -> tuple[float, float]:
    """Fluid-region COM with a circular mean in periodic x and arithmetic mean in y.

    ``x`` and ``y`` may be coordinate vectors or full meshgrid arrays. Coordinates
    are cell centers; the periodic length is inferred from the uniform x spacing.
    """
    phase, distance = _matching_fields(phi, sdf)
    x_axis, y_axis = _coordinate_axes(x, y, phase.shape)
    dx_values = np.diff(x_axis)
    if not np.allclose(dx_values, dx_values[0], rtol=1e-6, atol=1e-12) or dx_values[0] <= 0:
        raise ValueError("x coordinates must be uniformly spaced and increasing")
    period = float(dx_values[0] * phase.shape[0])
    weights = np.where(distance >= 0.0, phase, 0.0).astype(np.float64)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("center of mass is undefined for zero/non-positive fluid phase mass")

    y_cm = float(np.sum(weights * y_axis[None, :], dtype=np.float64) / total)
    angles = 2.0 * math.pi * (x_axis - x_axis[0]) / period
    x_weights = weights.sum(axis=1)
    cos_mean = float(np.sum(x_weights * np.cos(angles), dtype=np.float64) / total)
    sin_mean = float(np.sum(x_weights * np.sin(angles), dtype=np.float64) / total)
    if math.hypot(cos_mean, sin_mean) <= 1e-12:
        raise ValueError("periodic x center is undefined for a domain-uniform distribution")
    mean_angle = math.atan2(sin_mean, cos_mean) % (2.0 * math.pi)
    x_cm = float(x_axis[0] + period * mean_angle / (2.0 * math.pi))
    # Normalize to the domain interval [x0, x0 + Lx); seam values may be near either end.
    x_cm = float(x_axis[0] + ((x_cm - x_axis[0]) % period))
    return x_cm, y_cm


def periodic_spreading_width(phi: Any, threshold: float = 0.5, dx: float = 1.0, Lx: float | None = None) -> float:
    """Minimum periodic x-arc spanning all occupied columns, using cell-edge width.

    The largest *empty-column count* is removed from the ring. Thus columns
    ``[Nx-2, Nx-1, 0, 1]`` cover four cells, not almost the entire periodic domain.
    A single occupied column has width ``dx``; a fully occupied domain has width ``Lx``.
    """
    phase = _field(phi, "phi")
    nx = phase.shape[0]
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive")
    if Lx is None:
        Lx = nx * dx
    if not math.isfinite(Lx) or Lx <= 0 or not math.isclose(Lx, nx * dx, rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError("Lx must be finite, positive, and equal to Nx * dx")

    occupied = np.flatnonzero(np.any(phase >= threshold, axis=1))
    count = int(occupied.size)
    if count == 0:
        return 0.0
    if count == nx:
        return float(Lx)
    gaps_empty = (np.roll(occupied, -1) - occupied - 1) % nx
    largest_empty_gap = int(gaps_empty.max())
    covered_cells = nx - largest_empty_gap
    return float(covered_cells * dx)


def beta_from_width(width: float, R: float) -> float:
    """Spreading ratio ``beta = width / D0`` with ``D0 = 2 R``."""
    if not math.isfinite(width) or width < 0:
        raise ValueError("width must be finite and non-negative")
    if not math.isfinite(R) or R <= 0:
        raise ValueError("R must be finite and positive")
    return float(width / (2.0 * R))


def spreading_beta(phi: Any, R: float, threshold: float = 0.5, dx: float = 1.0, Lx: float | None = None) -> float:
    """Compute periodic spreading ratio from a phase field and initial radius."""
    return beta_from_width(periodic_spreading_width(phi, threshold, dx, Lx), R)


def bottom_gap(phi: Any, sdf: Any, threshold: float = 0.5) -> float:
    """Minimum fluid-side SDF beneath cells with ``phi >= threshold``.

    Raises ``ValueError`` when no selected liquid cell exists in the geometric
    fluid region; this is an undefined observable, not an infinite physical gap.
    """
    phase, distance = _matching_fields(phi, sdf)
    mask = (phase >= threshold) & (distance >= 0.0)
    if not mask.any():
        raise ValueError("bottom gap is undefined: no thresholded liquid in fluid region")
    return float(np.min(distance[mask]))


def contact_signal(phi: Any, sdf: Any, dx: float, phi_threshold: float = 0.5, gap_cells: float = 1.5) -> bool:
    """Diagnostic contact event: ``bottom_gap <= gap_cells * dx`` (not unique physics)."""
    if not math.isfinite(dx) or dx <= 0 or not math.isfinite(gap_cells) or gap_cells < 0:
        raise ValueError("dx must be positive and gap_cells non-negative")
    return bottom_gap(phi, sdf, phi_threshold) <= gap_cells * dx


def _height_mask(phi: Any, sdf: Any, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    phase, distance = _matching_fields(phi, sdf)
    mask = (phase >= threshold) & (distance >= 0.0)
    if not mask.any():
        raise ValueError("height is undefined: no thresholded liquid in fluid region")
    return phase, mask


def top_height(phi: Any, sdf: Any, y: Any, threshold: float = 0.5) -> float:
    """Highest y cell-center containing thresholded liquid in the fluid region."""
    phase, mask = _height_mask(phi, sdf, threshold)
    _, y_axis = _coordinate_axes(np.arange(phase.shape[0]), y, phase.shape)
    return float(y_axis[np.any(mask, axis=0)].max())


def bottom_height(phi: Any, sdf: Any, y: Any, threshold: float = 0.5) -> float:
    """Lowest y cell-center containing thresholded liquid in the fluid region."""
    phase, mask = _height_mask(phi, sdf, threshold)
    _, y_axis = _coordinate_axes(np.arange(phase.shape[0]), y, phase.shape)
    return float(y_axis[np.any(mask, axis=0)].min())


def vertical_extent(phi: Any, sdf: Any, y: Any, threshold: float = 0.5) -> float:
    """Vertical height of thresholded liquid in the geometric fluid region."""
    return top_height(phi, sdf, y, threshold) - bottom_height(phi, sdf, y, threshold)
