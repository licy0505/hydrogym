"""Conserved-mode algebra and exact mass reductions for the contract-v9 phase transport.

L1A-2g forensic support module.  It contains **no production numerics**: every function here either
* measures the shipped :mod:`phasefield` operators, or
* is a diagnostic helper that the audit driver calls.

The single authoritative conserved quantity of contract v9 is the cut-cell fluid mass

.. math::

    M_\\phi = \\sum_i V_i \\phi_i,

with ``V_i`` the partial fluid control volume of :func:`phasefield.phase_control_volumes`.  The
weighted implicit solve is posed in ``y = sqrt(V) phi``, so with

.. math::

    c = \\sqrt{V}\\,\\mathbf 1, \\qquad
    S = V^{-1/2} K V^{-1/2}, \\qquad
    A = I + \\alpha S^2, \\quad \\alpha = \\Delta t\\, M\\, \\epsilon

the physical mass is the ``c``-mode of the implicit variable, ``M_\\phi = c^T y``, and the shipped
operator satisfies ``S c = 0`` and ``A c = c``.  An exact solve therefore preserves ``c^T y = c^T b``.

Two independent facts make the measurements here authoritative:

1.  **Exact reduction.**  ``V_i`` and ``phi_i`` are stored in the working dtype (production
    ``float32``).  The product of two ``float32`` numbers has at most 48 significant bits, so it is
    *represented exactly* in ``float64``; :func:`exact_conserved_mass` therefore returns the exact
    mass of the stored state (``math.fsum`` over exactly represented products).  Any drift it
    reports is real state drift, not reduction noise.
2.  **Separate measurement channels.**  :func:`mass_reductions` reports the same mass computed with
    the working-dtype device reduction, a ``float64`` pairwise reduction and the exact compensated
    one, so ``REDUCTION_MEASUREMENT_ONLY`` can be distinguished from a genuine state change.

The module is import-light on purpose: it never mutates a solver array and never touches the phase
state.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import phasefield as pf

#: Metadata string for the L1A-2g conserved-mode convention (componentwise cut-cell volume).
CONSERVED_MODE_METHOD = "componentwise_cutcell_volume"
#: Lower bound below which a partial face aperture counts as *closed* when the fluid graph is built.
#: ``phi`` values are the model's own length/volume scales; apertures are geometric lengths and a
#: truly closed face has exact machine zero (see :func:`phasefield.phase_face_apertures`), so the
#: connectivity of the transported graph must not be decided by a loose tolerance.
APERTURE_OPEN_TOLERANCE = 0.0


# ----------------------------------------------------------------------------------------------
# mass reductions: three independent channels
# ----------------------------------------------------------------------------------------------
def _as_float64(array) -> np.ndarray:
    """Host ``float64`` copy of an array of any supported dtype (widening is exact)."""
    return np.asarray(array).astype(np.float64, copy=False)


def exact_conserved_mass(volume, phi) -> float:
    """Exact value of ``sum_i V_i phi_i`` for the *stored* arrays.

    ``float32`` products are exact in ``float64`` (24+24 <= 53 significant bits) and ``math.fsum``
    is correctly rounded, so this is the exact mass of the state as it exists in memory.  It is the
    forensic gate: a change reported here is a real change of ``M_phi``.
    """
    products = (_as_float64(volume) * _as_float64(phi)).ravel()
    return float(math.fsum(products.tolist()))


def reference_mass_f64(volume, phi) -> float:
    """``float64`` pairwise reduction of the same products (numpy device-style reduction)."""
    return float(np.sum(_as_float64(volume) * _as_float64(phi), dtype=np.float64))


def working_dtype_mass(volume, phi) -> float:
    """Device reduction in the *working* dtype, exactly as a production diagnostic would do it."""
    volume_arr = jnp.asarray(volume)
    phi_arr = jnp.asarray(phi)
    dtype = jnp.result_type(volume_arr, phi_arr)
    return float(jnp.sum(volume_arr.astype(dtype) * phi_arr.astype(dtype)))


def mass_reductions(volume, phi) -> dict[str, float]:
    """The three independent measurements of §7, side by side."""
    return {
        "M_device_work": working_dtype_mass(volume, phi),
        "M_reference_f64": reference_mass_f64(volume, phi),
        "M_reference_compensated": exact_conserved_mass(volume, phi),
    }


def roundoff_scales(volume, phi, *, gamma_count: int | None = None) -> dict[str, float]:
    """Round-off normalisation scales of §10 for one state.

    ``E_round = eps_mach * sum_i |V_i phi_i|`` is the unit-in-the-last-place scale of the
    *measurement*; ``E_sum = gamma_N * sum_i |V_i phi_i|`` with ``gamma_N = N eps / (1 - N eps)`` is
    the conservative accumulation bound of a length-``N`` reduction.  They are used to separate
    ``O(1)`` round-off from a systematic loss; they are not claimed to be strict error bounds.
    """
    magnitude = float(np.sum(np.abs(_as_float64(volume) * _as_float64(phi)), dtype=np.float64))
    dtype = jnp.result_type(jnp.asarray(volume), jnp.asarray(phi))
    eps = float(jnp.finfo(dtype).eps)
    count = int(np.size(_as_float64(volume))) if gamma_count is None else int(gamma_count)
    product = count * eps
    gamma_n = product / (1.0 - product) if product < 1.0 else float("inf")
    return {
        "eps_machine": eps,
        "sum_abs_mass_density": magnitude,
        "E_round": eps * magnitude,
        "n_terms": count,
        "gamma_n": gamma_n,
        "E_sum": gamma_n * magnitude,
    }


# ----------------------------------------------------------------------------------------------
# conserved mode and weighted operator
# ----------------------------------------------------------------------------------------------
def conserved_mode(sqrt_volume) -> jnp.ndarray:
    """``c = sqrt(V) * 1``: the constant mode of the Euclidean-SPD weighted operator."""
    return sqrt_volume * jnp.ones_like(sqrt_volume)


def operator_arrays(solid: pf.Solid, p: pf.PhaseFieldParams) -> pf.PhaseTransportOperator:
    """The shipped transport operator (never rebuilt, never modified)."""
    return pf.phase_transport_operator(solid, p)


def weighted_stiffness(value, inverse_sqrt_volume, weight_x, weight_y):
    """``S x = V^-1/2 K V^-1/2 x`` using the shipped kernel."""
    return pf.weighted_symmetric_operator(value, inverse_sqrt_volume, weight_x, weight_y)


def implicit_operator(value, inverse_sqrt_volume, weight_x, weight_y, alpha):
    """``A x = (I + alpha S^2) x``, written exactly like the shipped CG operator."""
    stiff = weighted_stiffness(value, inverse_sqrt_volume, weight_x, weight_y)
    return value + alpha * weighted_stiffness(stiff, inverse_sqrt_volume, weight_x, weight_y)


def implied_alpha(solid: pf.Solid, p: pf.PhaseFieldParams, dt: float) -> jnp.ndarray:
    """``alpha = dt * M * eps`` in the working dtype, as :func:`phasefield.solve_ch_implicit` uses it."""
    operator = operator_arrays(solid, p)
    return jnp.asarray(float(dt) * float(p.M) * float(p.eps), dtype=operator.sqrt_volume.dtype)


def _relative_norm(value, reference) -> float:
    numerator = jnp.linalg.norm(jnp.reshape(value, (-1,)))
    denominator = jnp.linalg.norm(jnp.reshape(reference, (-1,)))
    denominator = jnp.maximum(denominator, jnp.asarray(1.0e-300, dtype=denominator.dtype))
    return float(numerator / denominator)


def null_mode_identities(solid: pf.Solid, p: pf.PhaseFieldParams, dt: float) -> dict[str, Any]:
    """Direct measurement of ``S c = 0`` and ``A c = c`` in the working dtype.

    ``K 1`` is machine-exact zero (every face term is ``w_f (1 - 1)``), so ``S c`` is *expected* to
    be at the round-off floor; the size of that floor is what decides whether the invariant ``A c =
    c`` can be relied on in the working dtype.
    """
    operator = operator_arrays(solid, p)
    alpha = implied_alpha(solid, p, dt)
    sqrt_volume = operator.sqrt_volume
    c = conserved_mode(sqrt_volume)
    scaled_inverse = operator.inverse_sqrt_volume * c  # V^-1/2 c, mathematically 1
    stiffness = pf.graph_stiffness_apply(scaled_inverse, operator.weight_x, operator.weight_y)
    s_c = operator.inverse_sqrt_volume * stiffness
    a_c = implicit_operator(c, operator.inverse_sqrt_volume, operator.weight_x, operator.weight_y, alpha)
    defect = a_c - c
    dtype = sqrt_volume.dtype
    return {
        "dt": float(dt),
        "alpha": float(alpha),
        "dtype": str(dtype),
        "norm_c": float(jnp.linalg.norm(jnp.reshape(c, (-1,)))),
        "norm_Sc_over_norm_c": _relative_norm(s_c, c),
        "norm_Sc_abs": float(jnp.max(jnp.abs(s_c))),
        "norm_Ac_minus_c_over_norm_c": _relative_norm(defect, c),
        "norm_Ac_minus_c_abs": float(jnp.max(jnp.abs(defect))),
        "scaling_residual_max": float(jnp.max(jnp.abs(scaled_inverse - 1.0))),
        "alpha_times_norm_sc": float(alpha) * _relative_norm(s_c, c),
    }


def conserved_mode_products(y, b, sqrt_volume) -> dict[str, float]:
    """``c^T b`` and ``c^T y`` of one weighted solve, in the working dtype and in exact arithmetic."""
    c = sqrt_volume
    return {
        "cTb_work": float(jnp.sum(c * b)),
        "cTy_work": float(jnp.sum(c * y)),
        "cTb_f64": float(np.sum(_as_float64(c) * _as_float64(b), dtype=np.float64)),
        "cTy_f64": float(np.sum(_as_float64(c) * _as_float64(y), dtype=np.float64)),
        "cTb_exact": float(math.fsum((_as_float64(c) * _as_float64(b)).ravel().tolist())),
        "cTy_exact": float(math.fsum((_as_float64(c) * _as_float64(y)).ravel().tolist())),
    }


# ----------------------------------------------------------------------------------------------
# fluid connectivity / conserved component modes
# ----------------------------------------------------------------------------------------------
def _find(parent: list[int], index: int) -> int:
    root = index
    while parent[root] != root:
        root = parent[root]
    while parent[index] != root:
        parent[index], index = root, parent[index]
    return root


def _union(parent: list[int], rank: list[int], first: int, second: int) -> None:
    root_a, root_b = _find(parent, first), _find(parent, second)
    if root_a == root_b:
        return
    if rank[root_a] < rank[root_b]:
        root_a, root_b = root_b, root_a
    parent[root_b] = root_a
    if rank[root_a] == rank[root_b]:
        rank[root_a] += 1


def fluid_connectivity(aperture_x, aperture_y, volume) -> dict[str, Any]:
    """Connected components of the *transported* fluid graph defined by ``A_f > 0`` (host, exact).

    Open faces are read from the shipped ``A_f`` arrays, so the graph is the one the phase fluxes
    actually live on and not a cell-centre mask.  A single global ``c = sqrt(V) 1`` is only the
    complete constant mode of a **connected** fluid domain; a disconnected graph needs one mode per
    component, otherwise a global conservation gate would allow mass to move between components.
    """
    open_x = np.asarray(aperture_x, dtype=np.float64) > APERTURE_OPEN_TOLERANCE
    open_y = np.asarray(aperture_y, dtype=np.float64) > APERTURE_OPEN_TOLERANCE
    volume_host = _as_float64(volume)
    fluid = volume_host > 0.0
    nx, ny = fluid.shape
    count = int(fluid.sum())
    parent = list(range(nx * ny))
    rank = [0] * (nx * ny)

    def index(i: int, j: int) -> int:
        return (i % nx) + nx * (j % ny)

    for i in range(nx):
        for j in range(ny):
            if not fluid[i, j]:
                continue
            here = index(i, j)
            right = index(i + 1, j)
            if open_x[i, j] and fluid[(i + 1) % nx, j]:
                _union(parent, rank, here, right)
            up = index(i, j + 1)
            if open_y[i, j] and fluid[i, (j + 1) % ny]:
                _union(parent, rank, here, up)

    labels = np.full((nx, ny), -1, dtype=np.int64)
    roots: dict[int, int] = {}
    for i in range(nx):
        for j in range(ny):
            if not fluid[i, j]:
                continue
            root = _find(parent, index(i, j))
            if root not in roots:
                roots[root] = len(roots)
            labels[i, j] = roots[root]

    n_components = len(roots)
    component_volumes = [float(np.sum(volume_host[labels == k])) for k in range(n_components)]
    component_cells = [int(np.sum(labels == k)) for k in range(n_components)]
    order = np.argsort(np.asarray(component_volumes))[::-1]
    return {
        "n_fluid_components": int(n_components),
        "n_fluid_cells": count,
        "n_solid_cells": int(nx * ny - count),
        "component_volumes": [component_volumes[int(k)] for k in order],
        "component_cells": [component_cells[int(k)] for k in order],
        "labels": labels,
        "component_volume_sum": float(sum(component_volumes)),
        "total_volume": float(np.sum(volume_host)),
    }


def component_basis(labels, sqrt_volume, n_components: int) -> list[jnp.ndarray]:
    """Orthonormalised component modes ``q_k = sqrt(V) 1_{Omega_k} / ||sqrt(V) 1_{Omega_k}||``."""
    basis: list[jnp.ndarray] = []
    labels_host = np.asarray(labels)
    sqrt_host = _as_float64(sqrt_volume)
    for k in range(n_components):
        mask = (labels_host == k).astype(np.float64)
        vector = mask * sqrt_host
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            continue
        basis.append(jnp.asarray((vector / norm).astype(np.asarray(sqrt_volume).dtype)))
    return basis


def component_masses(volume, phi, labels, n_components: int) -> list[float]:
    """Exact mass carried by each fluid component (sorted like :func:`fluid_connectivity`)."""
    labels_host = np.asarray(labels)
    volume_host = _as_float64(volume)
    phi_host = _as_float64(phi)
    masses = []
    for k in range(n_components):
        mask = labels_host == k
        masses.append(float(math.fsum((volume_host[mask] * phi_host[mask]).tolist())))
    return sorted(masses, reverse=True)


# ----------------------------------------------------------------------------------------------
# audit-only CG with a per-iteration conserved-mode trace
# ----------------------------------------------------------------------------------------------
def cg_iteration_trace(
    rhs_field,
    sqrt_volume,
    inverse_sqrt_volume,
    weight_x,
    weight_y,
    alpha,
    rtol,
    max_iterations,
    *,
    max_traced: int = 64,
) -> dict[str, Any]:
    """Replicate :func:`phasefield._cg_solve_impl` step by step and trace the conserved mode.

    The arithmetic of the ``x``/``r``/``d`` recurrences is copied verbatim from the shipped solver so
    the traced iterate is the shipped iterate (checked bit-for-bit against
    :func:`phasefield.solve_ch_implicit` by the audit).  Only *independent* measurements are added:

    * ``||r_k||`` (recursive residual) and ``||b - A x_k||`` (recomputed),
    * ``c^T r_k``, ``c^T p_k``, ``c^T x_k``, ``c^T (b - A x_k)``,
    * ``alpha_k`` and ``beta_k``.

    Because ``A^-1 c = c``, the exact identity ``c^T y_k = c^T b - c^T r_k`` holds for any iterate;
    a growing ``c^T r_k`` is therefore exactly the conserved-mode pollution of the Krylov
    recurrence, and ``c^T(b - A x_k)`` is the same quantity measured on the true residual.
    """

    def operator(value):
        stiff = weighted_stiffness(value, inverse_sqrt_volume, weight_x, weight_y)
        return value + alpha * weighted_stiffness(stiff, inverse_sqrt_volume, weight_x, weight_y)

    c = sqrt_volume
    x = jnp.zeros_like(rhs_field)
    residual = rhs_field - operator(x)
    direction = residual
    residual_sq = jnp.vdot(residual, residual).real
    rhs_norm = jnp.sqrt(jnp.vdot(rhs_field, rhs_field).real)
    scale = jnp.maximum(rhs_norm, jnp.asarray(1.0e-30, dtype=rhs_field.dtype))
    rel = jnp.sqrt(residual_sq) / scale

    def record(index, x_value, r_value, p_value, rel_value, a_step, b_step):
        true_residual = rhs_field - operator(x_value)
        return {
            "iteration": int(index),
            "residual_norm": float(jnp.sqrt(jnp.vdot(r_value, r_value).real)),
            "recomputed_residual_norm": float(jnp.sqrt(jnp.vdot(true_residual, true_residual).real)),
            "relative_residual": float(rel_value),
            "cTr": float(jnp.sum(c * r_value)),
            "cTp": float(jnp.sum(c * p_value)),
            "cTx": float(jnp.sum(c * x_value)),
            "cT_true_residual": float(jnp.sum(c * true_residual)),
            "alpha_step": float(a_step),
            "beta_step": float(b_step),
        }

    trace = [record(0, x, residual, direction, rel, jnp.asarray(0.0), jnp.asarray(0.0))]
    iteration = 0
    while iteration < int(max_iterations) and bool(jnp.isfinite(rel)) and float(rel) > float(rtol):
        image = operator(direction)
        denominator = jnp.vdot(direction, image).real
        valid_denominator = jnp.isfinite(denominator) & (denominator > 0.0)
        safe_denominator = jnp.where(valid_denominator, denominator, 1.0)
        step_length = residual_sq / safe_denominator
        x_new = x + step_length * direction
        r_candidate = residual - step_length * image
        r_new = jnp.where(valid_denominator, r_candidate, jnp.full_like(r_candidate, jnp.nan))
        residual_sq_new = jnp.vdot(r_new, r_new).real
        rel_new = jnp.sqrt(residual_sq_new) / scale
        safe_rr = jnp.maximum(residual_sq, jnp.asarray(1.0e-30, dtype=rhs_field.dtype))
        beta = residual_sq_new / safe_rr
        direction_new = r_new + beta * direction
        x, residual, direction = x_new, r_new, direction_new
        residual_sq = residual_sq_new
        rel = rel_new
        iteration += 1
        if len(trace) <= int(max_traced):
            trace.append(record(iteration, x, residual, direction, rel, step_length, beta))
    converged = bool(jnp.isfinite(rel) & (rel <= rtol))
    solution = x if converged else jnp.full_like(x, jnp.nan)
    return {
        "y": solution,
        "iterations": iteration,
        "relative_residual": float(rel),
        "converged": converged,
        "trace": trace,
    }


def cg_solution_no_trace(rhs_field, sqrt_volume, inverse_sqrt_volume, weight_x, weight_y, alpha, rtol, max_iterations):
    """Untraced twin of :func:`cg_iteration_trace`, used for the bit-identity check."""
    result = cg_iteration_trace(
        rhs_field,
        sqrt_volume,
        inverse_sqrt_volume,
        weight_x,
        weight_y,
        alpha,
        rtol,
        max_iterations,
        max_traced=0,
    )
    return result["y"], result["iterations"], result["relative_residual"], result["converged"]


__all__ = [
    "APERTURE_OPEN_TOLERANCE",
    "CONSERVED_MODE_METHOD",
    "cg_iteration_trace",
    "cg_solution_no_trace",
    "component_basis",
    "component_masses",
    "conserved_mode",
    "conserved_mode_products",
    "exact_conserved_mass",
    "fluid_connectivity",
    "implicit_operator",
    "implied_alpha",
    "mass_reductions",
    "null_mode_identities",
    "operator_arrays",
    "reference_mass_f64",
    "roundoff_scales",
    "weighted_stiffness",
    "working_dtype_mass",
]
