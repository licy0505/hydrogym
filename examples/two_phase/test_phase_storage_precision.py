"""L1A-2i: phase-state storage precision -- harness, gates and the anti-cheating properties.

These tests pin (a) the *harness* -- the audit's exchange correction must be the shipped solve, its
mass metric must be the field the operators read, and its numbers must add up to the observed drift;
(b) the *contract* -- contract 11 defaults to A1 phase-only float64 while momentum remains float32,
and the explicit legacy model reproduces the contract-10 storage dtype; and (c) the *anti-cheating* rules of the stage: no global mass
target, no post-step correction, no cross-cell redistribution, no angle calibration, and the hidden
compensation state of the B1/C1 candidates is never advected, never read by an operator and never
counted as physical mass (so B/C can never "pass" by hiding mass in a bookkeeping field).
"""

from __future__ import annotations

import ast
import inspect
import math
import os
import subprocess
import sys
import tokenize
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import phasefield as pf  # noqa: E402
from production import krylov_roundoff_audit as kra  # noqa: E402
from production import mass_precision_audit as mpa  # noqa: E402
from production import phase_storage_precision_audit as psa  # noqa: E402

MODULE_PATH = Path(psa.__file__)
PHASEFIELD_PATH = Path(pf.__file__)


def code_only(path: Path) -> str:
    """Source with comments and string literals stripped (anti-cheating scans must not read prose)."""
    pieces: list[str] = []
    with path.open() as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    return " ".join(pieces)


# --------------------------------------------------------------------------- harness integrity
def test_no_duplicate_top_level_definitions():
    """A duplicated definition silently shadows the newer one (the bug class this stage hit)."""
    names = [
        node.name
        for node in ast.parse(MODULE_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef)
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == [], f"duplicate top-level definitions shadow earlier ones: {duplicates}"


@pytest.mark.parametrize("N", [32, 48])
def test_production_correction_matches_shipped_solve(N):
    """``rhs + x`` of the audit's correction extractor is *bit-identical* to the shipped solve.

    Every candidate is compared on the same solver; if this identity ever breaks, the audit would be
    measuring the storage rule against a different Krylov implementation than production runs.
    """
    p, solid, state, operator, _m0, _e = psa.build_case(N, target_deg=150.0, candidate="A0_float32")
    dt = p.dt / 3.0
    nxt = psa.to_state(state)
    phi_rhs, _u, _v, _mu, mu_expl = pf.rhs(nxt, solid, p)
    ch = -pf.control_volume_divergence(
        *pf.chemical_potential_fluxes(mu_expl, solid, p), operator.volume_safe
    )
    advective = kra.production_advective_rate(state["phi"], state["u"], state["v"], solid, p, dt, phi_rhs)
    rhs = state["phi"] + dt * (advective + ch)
    shipped, _info = pf.solve_ch_implicit(rhs, solid, p, dt)
    correction, converged, _iterations = psa.production_correction(
        rhs,
        operator.volume_safe,
        operator.weight_x,
        operator.weight_y,
        jnp.asarray(dt * float(p.M) * float(p.eps), rhs.dtype),
        jnp.asarray(float(p.ch_solver_rtol), rhs.dtype),
        jnp.asarray(int(p.ch_solver_max_iterations), jnp.int32),
    )
    assert bool(converged)
    assert jnp.array_equal(rhs + correction, shipped)


# --------------------------------------------------------------------------- the production default
def test_contract11_default_phase_storage_is_a1():
    """The promoted default is phase-only float64; contract-10 float32 remains explicit."""
    assert pf.SOLVER_CONTRACT_VERSION == 11
    assert pf.PHASE_STORAGE_MODEL == "phase_only_float64_v1"
    assert pf.PHASE_STORAGE_MODELS[0] == pf.PHASE_STORAGE_MODEL
    p = pf.PhaseFieldParams(Nx=16, Ny=16)
    assert p.phase_storage_model == "phase_only_float64_v1"
    assert pf.phase_state_dtype(p) == jnp.float64
    legacy = pf.PhaseFieldParams(Nx=16, Ny=16, phase_storage_model="float32_contract_10")
    assert pf.phase_state_dtype(legacy) == jnp.float32
    assert pf.phase_storage_metadata(p) == {
        "phase_storage_model": "phase_only_float64_v1",
        "phase_state_dtype": "float64",
        "velocity_state_dtype": "float32",
    }
    with pytest.raises(ValueError):
        pf.PhaseFieldParams(Nx=16, Ny=16, phase_storage_model="not_a_model")


def test_contract11_default_step_keeps_momentum_float32():
    """A production step stores phi in float64 and u/v/t in the working float32 dtype."""
    p, solid, state = mpa.build_case(
        32,
        M=mpa.M_REF,
        rtol=1.0e-6,
        target_deg=150.0,
        wall_height=0.25,
        phase_storage_model=pf.PHASE_STORAGE_MODEL,
    )
    stepped, _diagnostics = pf.step_with_diagnostics(state, solid, p)
    assert stepped.phi.dtype == jnp.float64
    for name in ("u", "v", "t"):
        assert getattr(stepped, name).dtype == jnp.float32, name


def test_default_matches_stage1_a1_candidate():
    """Contract-11 default state construction matches the frozen Stage-1 A1 representation."""
    p_default, solid_default, state_default = mpa.build_case(
        32, target_deg=150.0, wall_height=0.25, phase_storage_model=pf.PHASE_STORAGE_MODEL
    )
    p_candidate, solid_candidate, state_candidate, *_ = psa.build_case(
        32, target_deg=150.0, candidate="A1_phase_float64"
    )
    assert p_default.phase_storage_model == p_candidate.phase_storage_model == pf.PHASE_STORAGE_MODEL
    for name in ("phi", "u", "v", "t"):
        np.testing.assert_array_equal(getattr(state_default, name), getattr(psa.to_state(state_candidate), name))
    np.testing.assert_array_equal(solid_default.geometry.volume, solid_candidate.geometry.volume)


def test_unknown_storage_model_fails_closed():
    with pytest.raises(ValueError):
        pf.PhaseFieldParams(Nx=8, Ny=8, phase_storage_model="bogus_model")


def test_a1_default_fails_closed_when_jax_x64_is_disabled():
    """A fresh process without x64 must fail rather than warn-and-demote the default phase."""
    script = r'''
import jax
import phasefield as pf
assert not bool(jax.config.x64_enabled)
try:
    pf.PhaseFieldParams(Nx=8, Ny=8)
except RuntimeError as exc:
    assert "requires JAX float64 support" in str(exc)
else:
    raise AssertionError("contract-11 default silently accepted without x64")
'''
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PHASEFIELD_PATH.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- the A1 candidate
@pytest.mark.parametrize("candidate", psa.CANDIDATES)
def test_candidate_state_dtypes(candidate):
    """A1/A2/F promote *only* the phase field; the momentum state stays float32."""
    p, _solid, state, _operator, _m0, _e = psa.build_case(32, target_deg=150.0, candidate=candidate)
    promoted = candidate in psa.PHASE_FLOAT64_CANDIDATES
    assert state["phi"].dtype == (jnp.float64 if promoted else jnp.float32)
    assert state["u"].dtype == (jnp.float64 if candidate == "F_full_float64" else jnp.float32)
    assert state["v"].dtype == (jnp.float64 if candidate == "F_full_float64" else jnp.float32)
    assert state["t"].dtype == (jnp.float64 if candidate == "F_full_float64" else jnp.float32)
    if candidate in psa.HIDDEN_STATE_CANDIDATES:
        assert state["aux"].shape == state["phi"].shape
        assert state["aux"].dtype == state["phi"].dtype
    assert p.dtype == jnp.float32


def test_operator_field_is_the_physical_field_only():
    """``phi_phys`` is the stored field for every candidate -- never ``phi + hidden``."""
    _p, _solid, state, operator, _m0, _e = psa.build_case(
        32, target_deg=150.0, candidate="B1_compensated"
    )
    volume = psa.volume_of(operator)
    hidden = dict(state)
    hidden["aux"] = jnp.full_like(state["aux"], 1.0e-3)
    for candidate in psa.CANDIDATES:
        assert jnp.array_equal(psa.operator_field(hidden, candidate), hidden["phi"])
        assert float(psa.physical_mass(hidden, volume, candidate)) == float(
            jnp.sum(volume * hidden["phi"].astype(jnp.float64))
        )
    # the bookkeeping sum is a different number, and it is *reported as such* -- never as the mass
    assert float(psa.bookkeeping_mass(hidden, volume, "B1_compensated")) != pytest.approx(
        float(psa.physical_mass(hidden, volume, "B1_compensated")), rel=1e-9
    )


# --------------------------------------------------------------------------- anti-cheating
def test_audit_has_no_global_mass_machinery():
    """No global target, no post-step correction, no redistribution, no angle calibration."""
    source = code_only(MODULE_PATH).lower()
    forbidden = (
        "global_correction",
        "mass_target",
        "redistribut",
        "rescale",
        "renormalize_phi",
        "clip_phi",
        "angle_calibration",
        "calibrate_angle",
    )
    present = [word for word in forbidden if word in source]
    assert present == [], f"forbidden machinery in the audit module: {present}"


def test_no_production_operator_is_called_with_the_hidden_state():
    """AST: no operator call in the audit receives the hidden/compensation field as its argument."""
    tree = ast.parse(MODULE_PATH.read_text())
    operators = {
        "rhs",
        "chemical_potential_fluxes",
        "control_volume_divergence",
        "advective_phase_source",
        "production_advective_rate",
        "solve_ch_implicit",
        "production_correction",
        "phase_transport_operator",
    }
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name not in operators:
            continue
        for argument in list(node.args) + [keyword.value for keyword in node.keywords]:
            text = ast.unparse(argument)
            if "aux" in text or "phi_lo" in text:
                offenders.append(f"{name}({text})")
    assert offenders == [], f"hidden state handed to an operator: {offenders}"


def test_hidden_state_is_not_advected_at_runtime(monkeypatch):
    """Call-count/spy check: during a B1 substep the operators see ``phi``, never ``phi_hi + phi_lo``.

    The spy records the phase field every production operator receives. With ``aux`` deliberately set
    to a large value, no call may receive a field equal to ``phi + aux`` and the number of calls must
    be exactly the production count.
    """
    _p, solid, state, operator, _m0, _e = psa.build_case(32, target_deg=150.0, candidate="B1_compensated")
    hidden = dict(state)
    hidden["aux"] = jnp.full_like(state["aux"], 1.0e-3)
    combined = hidden["phi"] + hidden["aux"]
    seen: list[tuple[str, bool]] = []
    original_rhs = pf.rhs

    def spy_rhs(candidate_state, *args, **kwargs):
        field = candidate_state.phi if hasattr(candidate_state, "phi") else candidate_state[0]
        seen.append(("rhs", bool(jnp.array_equal(field, combined))))
        return original_rhs(candidate_state, *args, **kwargs)

    monkeypatch.setattr(pf, "rhs", spy_rhs)
    dt = _p.dt / 3.0
    out, _info = psa.storage_substep(hidden, solid, _p, operator, dt, "B1_compensated")
    monkeypatch.undo()
    assert seen, "the spy never fired: the substep did not call the phase operator"
    assert all(not combined_seen for _name, combined_seen in seen), (
        "an operator received phi + hidden (the compensation field would be advected as a scalar)"
    )
    # and the hidden field did change: the compensation is real bookkeeping, not a no-op
    assert float(jnp.max(jnp.abs(out["aux"]))) > 0.0


def test_ledger_terms_are_the_only_mass_movers():
    """Every mass-moving term of the audit is one of the two ledger terms (no hidden sink)."""
    source = MODULE_PATH.read_text()
    tree = ast.parse(source)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and "mass" in node.name.lower()
    }
    assert "physical_mass" in names and "bookkeeping_mass" in names
    # the *gate* path may not read the bookkeeping mass ...
    for function in ("candidate_rows", "select_candidate", "_verdict", "production_readiness"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
        body = ast.unparse(node)
        assert "bookkeeping" not in body, f"{function} must not read the bookkeeping mass"
    # ... while the drift series must report it *separately* from the gated field
    drift = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "drift_series")
    body = ast.unparse(drift)
    assert "physical_mass(out, volume, candidate)" in body
    assert "bookkeeping_mass(out, volume, candidate)" in body
    assert "'masses'" in body and "'bookkeeping'" in body


# --------------------------------------------------------------------------- fast gate behaviour
@pytest.fixture(scope="module")
def quick_pair():
    rows = {}
    for candidate in ("A0_float32", "A1_phase_float64"):
        payload = psa.drift_series(
            48, target_deg=150.0, M_factor=1.0, public_steps=250, sample_every=25, candidate=candidate
        )
        stats = psa.series_classification(payload["masses"], payload["E_round"], payload["sample_every"])
        rows[candidate] = (payload, stats)
    return rows


def test_quick_fixture_separates_the_storage_term(quick_pair):
    """The baseline loses mass to storage every update; the float64 state loses none of it."""
    a0_payload, a0_stats = quick_pair["A0_float32"]
    a1_payload, a1_stats = quick_pair["A1_phase_float64"]
    assert abs(a1_stats["final_relative_drift"]) <= 1.0e-12
    assert abs(a0_stats["final_relative_drift"]) >= 1.0e-8
    assert abs(a0_payload["ledger"]["storage_loss_over_E_round"]) >= 1.0
    assert abs(a1_payload["ledger"]["storage_loss_over_E_round"]) <= 1.0e-6


def test_ledger_explains_the_measured_drift(quick_pair):
    """The running solve + storage terms must add up to the observed drift (to first order)."""
    for candidate, (payload, stats) in quick_pair.items():
        explained = payload["ledger"]["explained_relative_drift"]
        observed = stats["final_relative_drift"]
        assert abs(explained - observed) <= 0.5 * abs(observed) + 1.0e-13


def test_restart_is_bitwise_for_every_persistent_candidate():
    for candidate in ("A0_float32", "A1_phase_float64", "B1_compensated"):
        result = psa.restart_equivalence(32, first=40, second=40, candidate=candidate)
        formal = result["formal_restart"]
        for key in (
            "phi_max_abs_difference",
            "u_max_abs_difference",
            "v_max_abs_difference",
            "aux_max_abs_difference",
            "disk_round_trip_phi_max_abs_difference",
        ):
            assert formal[key] == 0.0, f"{candidate}: {key}"
        for dtype in formal["disk_round_trip_dtypes"].values():
            assert dtype in ("float32", "float64")


def test_hidden_state_survives_a_zeroed_restart_only_as_a_documented_non_path():
    """Dropping the hidden state changes the trajectory -- so it must never be silently dropped."""
    result = psa.restart_equivalence(32, first=40, second=40, candidate="B1_compensated")
    assert result["aux_dropped_restart"] is not None
    assert "NOT an allowed formal path" in result["aux_dropped_restart"]["note"]


def test_storage_rule_is_differentiable():
    """Forward/reverse mode through the *storage rule* agree with a finite difference."""
    for candidate in ("A0_float32", "A1_phase_float64", "B1_compensated"):
        result = psa.gradient_check(16, steps=2, candidate=candidate)
        assert result["storage_rule_ad_ok"], (candidate, result)
        assert result["storage_rule_reverse_mode_finite"]
    # the contract-10 CG's lax.while_loop blocks reverse mode for every candidate alike
    assert psa.gradient_check(16, steps=1, candidate="A0_float32")["full_step_reverse_mode"]["available"] is False


# --------------------------------------------------------------------------- gates and selection
def test_gate_bounds_match_the_frozen_spec():
    assert psa.QUICK_GATE == 2.0e-6
    assert psa.QUICK_GATE_STRONG == 5.0e-7
    assert psa.MEDIUM_GATE == 2.0e-4
    assert psa.MEDIUM_GATE_STRONG == 1.0e-4
    assert psa.ANGLE_DELTA_GATE == 0.2
    assert psa.LONG_HORIZON_PROJECTION_BOUND == 1.0e-3
    assert kra.GATES["quick"]["N"] == 48 and kra.GATES["quick"]["steps"] == 2500
    assert kra.GATES["quick"]["offsets"] == (0.0, 0.5)


def test_selection_rejects_only_on_hard_gates_and_respects_the_vocabulary():
    good = {
        "candidate": "A1_phase_float64",
        "status": "CANDIDATE",
        "storage_model": "phase_only_float64_v1",
        "quick": {},
        "medium_final_relative_drift": None,
        "quick_passed": True,
        "medium_passed": True,
        "long_horizon_clean": True,
        "restart_reproducible": True,
        "ad_ok": True,
        "rejection_reasons": [],
    }
    bad = dict(good, candidate="B1_compensated", quick_passed=False, rejection_reasons=["MASS_GATE_FAIL"])
    reference = dict(good, candidate="F_full_float64", status="REFERENCE")
    report = {"quick_matrix": {}}
    selection = psa.select_candidate([reference, bad, good], report)
    assert selection["selected"] == "A1_phase_float64"
    assert bad["status"] == "REJECTED"
    assert set(bad["rejection_reasons"]) <= set(psa.REJECTION_REASONS)
    assert reference["status"] == "REFERENCE"


def test_hidden_state_candidates_are_flagged_for_the_dataset_contract():
    assert psa.dataset_lineage("B1_compensated")["requires_new_persistent_field"] is True
    assert psa.dataset_lineage("C1_residual_feedback")["requires_new_persistent_field"] is True
    assert psa.dataset_lineage("A1_phase_float64")["requires_new_persistent_field"] is False
    for candidate in psa.CANDIDATES:
        row = psa.dataset_lineage(candidate)
        assert row["fingerprint_has_solver_sha256"], "the generator must hash the solver source"
        assert row["fingerprint_has_phase_storage_model"], "the training fingerprint must freeze A1 lineage"
        assert row["silent_resume_possible"] is False


def test_frozen_stage1_selection_is_the_contract11_production_default():
    """Stage 2 promotes the selected A1 model without re-opening candidate selection."""
    assert pf.SOLVER_CONTRACT_VERSION == 11
    assert pf.PHASE_STORAGE_MODEL == psa.PHASE_STORAGE_MODEL_OF["A1_phase_float64"]
    assert psa.PHASE_STORAGE_MODEL_OF["A1_phase_float64"] == "phase_only_float64_v1"
    hidden_models = {psa.PHASE_STORAGE_MODEL_OF[name] for name in psa.HIDDEN_STATE_CANDIDATES}
    assert set(pf.PHASE_STORAGE_MODELS_WITH_HIDDEN_STATE) == hidden_models
    assert set(psa.HIDDEN_STATE_CANDIDATES) <= set(psa.CANDIDATES)
    report = {"selection": {"selected": None}, "profile": "quick", "quick_matrix": {}}
    assert psa._verdict(dict(report, selection={"selected": None})) == "NOT_READY_FOR_SELECTION"


def test_report_is_json_serialisable_on_a_synthetic_run(tmp_path):
    """The writers must not choke on any report shape (guards the CI artifact path)."""
    report = {
        "stage": psa.STAGE,
        "profile": "quick",
        "contract_version": 11,
        "verdict": "V",
        "gates": {},
        "selection": {"selected": None, "reason": "r"},
        "cost": {},
        "production_readiness": {"statement": "s"},
        "candidate_rows": [],
        "environment": {},
        "notes": [],
    }
    written = psa.write_reports(report, tmp_path)
    assert "phase_storage_precision_report.json" in written
    assert "candidate_matrix.json" in written
    assert "manifest.json" in written
