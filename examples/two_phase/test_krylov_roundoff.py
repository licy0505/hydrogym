"""L1A-2h: float32 Krylov roundoff, the residual mass walk, and how it must not be "fixed".

The audit module measures where the residual contract-v10 mass drift comes from. These tests pin the
*harness* (so the measurement is trustworthy) and pin the *anti-cheating* properties the stage
requires: the solver must not learn a mass target, must not rescale the field, must not redistribute
mass, and must not be "fixed" by a post-step correction of any kind. They also pin the finding that
no candidate rule removes the drift, so a later change cannot quietly ship one and claim it did.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import phasefield as pf  # noqa: E402
from production import krylov_roundoff_audit as kra  # noqa: E402
from production import mass_precision_audit as mpa  # noqa: E402

MODULE_PATH = Path(kra.__file__)
PHASEFIELD_PATH = Path(pf.__file__)


# --------------------------------------------------------------------------- harness integrity
def test_no_duplicate_top_level_definitions():
    """A duplicated definition silently shadows the newer one (the bug class this stage hit)."""
    names = [node.name for node in ast.parse(MODULE_PATH.read_text()).body if isinstance(node, ast.FunctionDef)]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == [], f"duplicate top-level definitions shadow earlier ones: {duplicates}"


def test_variant_names_fail_closed():
    """An unknown variant must raise, never fall back to production arithmetic."""
    with pytest.raises(KeyError):
        kra.variant_flags("not_a_variant")
    assert kra.variant_flags("shipped") == dict.fromkeys(kra.VARIANT_FLAG_NAMES, False)
    for name in kra.SUBTEP_VARIANTS:
        flags = kra.variant_flags(name)
        assert set(flags) == set(kra.VARIANT_FLAG_NAMES)


def test_shipped_variant_is_bit_identical_to_production_substep():
    """`variant_substep('shipped')` reproduces `step_with_diagnostics` exactly (substep level)."""
    p, solid, state = mpa.build_case(48, target_deg=150.0)
    operator = pf.phase_transport_operator(solid, p)
    dt = p.dt / 3.0

    def production_substep(state_in):
        phi, u, v, t = state_in.phi, state_in.u, state_in.v, state_in.t
        phi_rhs, u_rhs, v_rhs, _mu, mu_expl = pf.rhs(pf.State(phi, u, v, t), solid, p)
        phi_new, _info = pf._phase_update(phi, u, v, solid, p, dt, phi_rhs, mu_expl)
        damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
        u_new = (u + dt * u_rhs) * damp
        v_new = (v + dt * v_rhs) * damp
        divergence = pf._ddx(u_new, p.dx) + pf._ddy(v_new, p.dy)
        pressure = pf.poisson_solve(divergence / dt, p.m2_proj)
        u_new = u_new - dt * pf._ddx(pressure, p.dx)
        v_new = v_new - dt * pf._ddy(pressure, p.dy)
        return pf.State(phi=phi_new, u=u_new, v=v_new, t=t + dt)

    mine = state
    theirs = state
    for _ in range(3):
        mine, _info, _stages = kra.variant_substep(mine, solid, p, operator, dt, "shipped")
        theirs = production_substep(theirs)
    assert float(jnp.max(jnp.abs(mine.phi - theirs.phi))) == 0.0
    assert float(jnp.max(jnp.abs(mine.u - theirs.u))) == 0.0
    assert float(jnp.max(jnp.abs(mine.v - theirs.v))) == 0.0


def test_advective_rate_follows_the_production_subcycling_policy():
    """The mirror must use whatever rate production uses for the current policy, and it must be able
    to tell the two apart: with subcycling on, the subcycled rate is *not* the single-step rate."""
    p, solid, state = mpa.build_case(48, target_deg=150.0)
    dt = p.dt / 3.0
    single = pf.rhs(state, solid, p)[0]
    rate = kra.production_advective_rate(state.phi, state.u, state.v, solid, p, dt, single)
    if pf.phase_advection_subcycles(p):
        subcycled = pf.advective_phase_source(state.phi, state.u, state.v, solid, p, dt)
        assert float(jnp.max(jnp.abs(rate - subcycled))) == 0.0
        assert float(jnp.max(jnp.abs(subcycled - single))) > 0.0
    else:
        assert float(jnp.max(jnp.abs(rate - single))) == 0.0
    # and the two policies really are different code paths, not a rename
    probe = pf.PhaseFieldParams(
        **{
            **{k: getattr(p, k) for k in ("Nx", "Ny", "Lx", "Ly", "Re", "We", "dt", "M", "eps")},
            "phase_advection_subcycling": "phase_only_fixed_substeps",
        }
    )
    assert pf.phase_advection_subcycles(probe) is True
    assert pf.phase_advection_subcycles(p) is False


def test_cg_variant_baseline_is_bit_identical_to_shipped_solve():
    payload = kra.baseline_identity_check()
    assert payload["bit_identical"] is True
    assert payload["baseline_max_abs_difference"] == 0.0


def test_weighted_dot_audit_float64_reference_agrees():
    dots = kra.dot_product_audit(48)
    assert abs(dots["rVr"]["D2_minus_reference_ulp"]) < 1.0
    assert abs(dots["pVAp"]["D2_minus_reference_ulp"]) < 1.0
    assert math.isfinite(dots["rVr"]["D1_minus_reference_ulp"])


def test_constant_mode_diagnostics_exist():
    trace = kra.constant_mode_trace(48)
    assert trace["iterations"] >= 1
    for key in ("projection_of_r", "projection_of_p", "projection_of_d", "projection_of_Ap"):
        assert len(trace["per_iteration"][key]) == trace["iterations"]
    assert trace["recursive_over_true_final"] is not None


def test_rule_table_parts_sum_to_total():
    """The decomposition is an identity, not a story: parts must add up to the measured total."""
    window = kra.decomposition_window(48, target_deg=150.0, warmup=60, steps=4)
    parts = (
        window["assembly"]["mean_over_E_round"]
        + window["correction_f32"]["mean_over_E_round"]
        + window["storage_residue"]["mean_over_E_round"]
    )
    assert abs(parts - window["sum_of_parts_over_E_round"]) <= 1e-9
    # the parts are measured against the assembled rhs, the total against the incoming field; the
    # closing term is the mass the correctly rounded rhs carries relative to phi, and it is reported
    closing = window["rhs_vs_phi_mass_gap_over_E_round"]
    assert abs(parts + closing - window["total"]["mean_over_E_round"]) < 1e-9


# --------------------------------------------------------------------------- the finding
def test_exact_arithmetic_defect_is_orders_below_every_float32_rule():
    """The f64 control is at the floor: the drift is arithmetic, and arithmetic cannot remove it."""
    table = kra.update_rule_defects(48, target_deg=150.0, warmup=60, substeps=4)
    floor = abs(table["f64_state"]["mean_over_E_round"])
    float32_rules = (
        table["production"]["mean_over_E_round"],
        table["assembly_f64"]["mean_over_E_round"],
        table["solve_f64"]["mean_over_E_round"],
        table["single_rounding"]["mean_over_E_round"],
        table["single_rounding_f32x"]["mean_over_E_round"],
    )
    assert floor < 1e-3
    assert min(abs(value) for value in float32_rules) > 100 * floor


def test_float64_flux_assembly_does_not_move_the_rule_table():
    """Candidate B/C style precision changes are not the mechanism (measured, not assumed)."""
    table = kra.update_rule_defects(48, target_deg=150.0, warmup=60, substeps=4)
    production = table["production"]["mean_over_E_round"]
    rounded_rhs = table["assembly_f64"]["mean_over_E_round"]
    assert abs(rounded_rhs - production) < 0.05 * max(abs(production), 1e-6)


def test_single_rounding_rules_disagree_with_production():
    """The update-form candidates really do change the arithmetic (not a silent no-op)."""
    p, solid, state = mpa.build_case(48, target_deg=150.0)
    operator = pf.phase_transport_operator(solid, p)
    dt = p.dt / 3.0
    reference, _info, _stages = kra.variant_substep(state, solid, p, operator, dt, "shipped")
    for variant in (
        "update_single_rounding",
        "krylov_f64+single_rounding+f64increment",
        "krylov_f64",
    ):
        candidate, _info, _stages = kra.variant_substep(state, solid, p, operator, dt, variant)
        assert float(jnp.max(jnp.abs(candidate.phi - reference.phi))) > 0.0, variant


def test_series_classification_flags_a_linear_bias():
    rng = np.random.default_rng(0)
    walk = 2.0 + np.cumsum(rng.normal(0.0, 1.0, 400))
    bias = 2.0 + np.arange(400) * 2.0 + rng.normal(0.0, 1.0, 400)
    walk_series = kra.series_classification(walk, 1.0, 1)
    bias_series = kra.series_classification(bias, 1.0, 1)
    assert walk_series["verdict"] == "RANDOM_WALK_DOMINANT"
    assert bias_series["verdict"] == "SYSTEMATIC_BIAS_DOMINANT"
    assert abs(bias_series["slope_t_statistic"]) > 10.0


# --------------------------------------------------------------------------- anti-cheating
def _strip_comments_and_strings(source: str) -> str:
    """Keep only code tokens: the anti-cheating search is about identifiers, not prose."""
    import io
    import tokenize

    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        pieces.append(token.string)
    return " ".join(pieces)


def _strip_comments_and_docstrings(source: str) -> str:
    """Keep code tokens only: the anti-cheating search is about identifiers, not prose."""
    import io
    import tokenize

    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        pieces.append(token.string)
    return " ".join(pieces)


def _solver_sources() -> str:
    """Source of everything on the phase-transport solve path, imports included."""
    names = (
        "phase_transport_operator",
        "volume_weighted_operator",
        "solve_ch_implicit",
        "_cg_solve_volume_weighted",
        "_phase_update",
        "step_with_diagnostics",
        "phase_only_step_with_diagnostics",
        "phase_transport_step",
        "_variant_solve",
        "variant_correction",
        "variant_substep",
    )
    chunks = []
    module_src = PHASEFIELD_PATH.read_text()
    for name in names:
        if name in ("_variant_solve", "variant_correction", "variant_substep"):
            chunks.append(inspect.getsource(getattr(kra, name)))
        else:
            node = next(node for node in ast.parse(module_src).body if getattr(node, "name", None) == name)
            chunks.append(ast.get_source_segment(module_src, node) or "")
    return _strip_comments_and_strings("\n".join(chunks))


def test_solver_never_reads_a_mass_target():
    source = _solver_sources()
    for forbidden in ("mass0", "initial_mass", "target_mass", "mass_target", "previous_mass", "m_ref"):
        assert forbidden not in source, f"solver path references {forbidden!r}"


def test_solver_never_rescales_or_offsets_the_field():
    """No post-step mass correction, no global rescale, no constant offset of phi."""
    source = _solver_sources()
    for forbidden in ("rescale", "renormalis", "renormaliz", "mass_fix", "correct_mass", "project_mass"):
        assert forbidden not in source, f"solver path contains {forbidden!r}"
    # a solve that multiplies or divides the whole field by a mass ratio would need these names
    assert "phi_mean" not in source
    assert "phi + offset" not in source


def test_phasefield_has_no_mass_redistribution():
    """The historical mass redistribution must stay unreachable on the impermeable path."""
    source = PHASEFIELD_PATH.read_text()
    assert "redistribute" not in source.lower()


def test_shipped_contract_metadata_unchanged_by_this_stage():
    """L1A-2h remains a historical audit while production ships contract 11/A1."""
    assert pf.SOLVER_CONTRACT_VERSION == 11
    assert pf.IMPLICIT_PHASE_SOLVER == "weighted_spd_nullspace_preserving_v1"
    assert pf.PHASE_MASS_INVARIANT == "componentwise_cutcell_volume"
    assert kra.CONTRACT_VERSION == pf.SOLVER_CONTRACT_VERSION


def test_audit_report_never_claims_all_ci_passed():
    """The report may not claim a global green: gates and verdicts are reported separately."""
    audit = kra.run_audit(profile="quick") if False else None  # never run the full audit here
    assert audit is None
    source = MODULE_PATH.read_text().lower()
    assert "all ci passed" not in source
    assert "everything passed" not in source
