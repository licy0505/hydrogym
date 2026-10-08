"""L1A-2p tests: impact-phase robustness audit and timestep policy (section 33).

The heavy production-spec measurements live in the forensic profile; these tests
exercise the audit machinery on tiny configurations and pin the policy algebra
that the promotion depends on.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import cases as cases_module
import generate_dataset as generator
import phasefield as pf
from production import impact_phase_robustness_audit as audit
from production import timestep_policy

JAX_ENABLE_X64_REQUIRED = "these tests require JAX_ENABLE_X64=1 (production storage contract)"


def _primary_case() -> dict:
    return audit.baseline_case_matrix()[audit.PRIMARY_LABEL]


def _production_args() -> audit.RunArgs:
    return audit.RunArgs()


# ---------------------------------------------------------------------------
# policy algebra (sections 12-14/21-22)
# ---------------------------------------------------------------------------


def test_stable_dt_exact_reproduction() -> None:
    """The policy's stable-dt cap is exactly the production ``pf.stable_dt`` value."""

    p192 = pf.PhaseFieldParams(Nx=192, Ny=192, Lx=6.0, Ly=6.0, dt=1.0)
    expected = float(pf.stable_dt(p192, u_max=2.0))
    assert expected == pytest.approx(0.00625)
    record = timestep_policy.effective_dt_for_case(_primary_case(), 192, 0.004, "legacy_requested_v0")
    assert record["stable_dt_cap"] == expected


def test_policy_dt_never_exceeds_requested_dt() -> None:
    case = _primary_case()
    policies = ("legacy_requested_v0", "fixed_cap_002_v1", "impact_phase_cap_dx2_v1", "cfl_multicriterion_v1")
    for policy in policies:
        for requested in (0.004, 0.003, 0.002):
            record = timestep_policy.effective_dt_for_case(case, 192, requested, policy)
            assert record["effective_dt"] <= requested + 1e-15
    legacy = timestep_policy.effective_dt_for_case(case, 192, 0.004, "legacy_requested_v0")
    assert legacy["effective_dt"] == pytest.approx(0.004)


def test_policy_limiting_criterion_recorded() -> None:
    case = _primary_case()
    record = timestep_policy.effective_dt_for_case(case, 192, 0.004, "impact_phase_cap_dx2_v1")
    assert record["limiting_criterion"] == "impact_phase_dx2_cap"
    assert np.isfinite(record["limiting_value"])
    assert record["limiting_value"] == pytest.approx(timestep_policy.IMPACT_PHASE_DX2_COEFFICIENT * (6.0 / 192) ** 2)
    assert record["effective_dt"] == pytest.approx(0.002)  # the measured-good dt at the failing resolution
    legacy = timestep_policy.effective_dt_for_case(case, 192, 0.004, "legacy_requested_v0")
    assert legacy["limiting_criterion"] == "requested_dt"


def test_candidate_resolution_is_deterministic() -> None:
    case = _primary_case()
    a = timestep_policy.effective_dt_for_case(case, 192, 0.004, "impact_phase_cap_dx2_v1")
    b = timestep_policy.effective_dt_for_case(case, 192, 0.004, "impact_phase_cap_dx2_v1")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    n144 = timestep_policy.effective_dt_for_case(case, 144, 0.004, "impact_phase_cap_dx2_v1")
    assert n144["effective_dt"] == pytest.approx(timestep_policy.IMPACT_PHASE_DX2_COEFFICIENT * (6.0 / 144) ** 2)
    assert n144["effective_dt"] < 0.004  # the N=144 cap binds below the requested step


def test_horizon_and_frame_cadence_preserved_under_policy() -> None:
    """A policy change re-scales steps, never the requested physical schedule (section 21)."""

    args = _production_args()
    case = _primary_case()
    legacy = audit.effective_schedule(
        case, dataclasses.replace(args, timestep_policy="legacy_requested_v0").namespace()
    )
    policy = audit.effective_schedule(
        case, dataclasses.replace(args, timestep_policy="impact_phase_cap_dx2_v1").namespace()
    )
    for schedule in (legacy, policy):
        assert schedule["horizon"] == pytest.approx(8.0)
        assert schedule["frame_dt"] == pytest.approx(0.08)
        assert schedule["n_frames"] == 100
    assert policy["effective_dt"] == pytest.approx(0.002)
    assert policy["nsteps"] == 4000
    assert policy["save_every"] == 40
    assert legacy["effective_dt"] == pytest.approx(0.004)
    assert legacy["nsteps"] == 2000


def test_policy_enters_trajectory_identity_and_fingerprint() -> None:
    args_legacy = dataclasses.replace(_production_args(), timestep_policy="legacy_requested_v0")
    args_policy = dataclasses.replace(_production_args(), timestep_policy="impact_phase_cap_dx2_v1")
    case = _primary_case()
    schedule_l = audit.effective_schedule(case, args_legacy.namespace())
    schedule_p = audit.effective_schedule(case, args_policy.namespace())
    fp_legacy = generator._dataset_fingerprint(
        case, args_legacy.namespace(), schedule_l["effective_dt"], schedule_l["nsteps"], schedule_l["save_every"]
    )
    fp_policy = generator._dataset_fingerprint(
        case, args_policy.namespace(), schedule_p["effective_dt"], schedule_p["nsteps"], schedule_p["save_every"]
    )
    assert fp_legacy != fp_policy  # different policies -> different trajectories/fingerprints
    fp_again = generator._dataset_fingerprint(
        case, args_policy.namespace(), schedule_p["effective_dt"], schedule_p["nsteps"], schedule_p["save_every"]
    )
    assert fp_policy == fp_again  # deterministic identity
    record = generator._time_step_policy_record(case, args_policy.namespace(), schedule_p["effective_dt"])
    assert record["time_step_policy_name"] == "impact_phase_cap_dx2_v1"
    assert record["effective_dt"] == pytest.approx(schedule_p["effective_dt"])


def test_subcycling_is_not_enabled() -> None:
    """The cut-cell advective CFL never binds (measured 0.137): subcycling stays off (section 14)."""

    p = pf.PhaseFieldParams(Nx=48, Ny=48, Lx=6.0, Ly=6.0, dt=2e-3)
    assert pf.phase_advection_subcycles(p) is False
    assert pf.PHASE_ADVECTION_SUBCYCLING == "disabled"


# ---------------------------------------------------------------------------
# generator gates unrelaxed (section 15)
# ---------------------------------------------------------------------------


def test_generator_thresholds_unrelaxed() -> None:
    args = _production_args()
    assert args.max_phi_overshoot == 0.02
    assert args.max_solid_leak == 5e-4
    assert args.min_total_mass_ratio == 0.995
    assert args.max_total_mass_ratio == 1.005
    assert args.max_speed == 5.0
    assert args.min_feature_cells == 2.0
    # the audit module's own gate constant matches the generator
    assert audit.OVERSHOOT_GATE == 0.02
    assert audit.CROSSING_THRESHOLD == audit.OVERSHOOT_GATE


def test_no_forbidden_repair_in_policy_or_audit_sources() -> None:
    for path in (Path(timestep_policy.__file__), Path(audit.__file__)):
        text = path.read_text()
        for forbidden in ("jnp.clip(", "np.clip(phi", "mass_redistribut", "enforce_solid_phi"):
            assert forbidden not in text, f"forbidden repair {forbidden!r} in {path}"


# ---------------------------------------------------------------------------
# promotion preconditions (sections 16/23)
# ---------------------------------------------------------------------------


def test_contract_bump_refused_without_full_canary_acceptance() -> None:
    validation = {"all_mandatory_accepted": False, "n_accepted": 3, "n_total": 8}
    with pytest.raises(audit.AuditValidationError):
        audit.apply_contract_bump(validation)


def test_contract_bump_dry_run_edits_nothing_and_lists_metadata_only_edits(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    validation = {
        "all_mandatory_accepted": True,
        "n_accepted": 8,
        "n_total": 8,
        "policy": timestep_policy.policy_identity("impact_phase_cap_dx2_v1"),
    }
    report = audit.apply_contract_bump(validation, dry_run=True)
    assert report["promotion"] == "DRY_RUN"
    assert report["edits"] == []
    # every planned edit is a metadata-only version/policy-default change: the
    # non-digit skeleton is unchanged, or the line is a version membership/list
    # update (its non-version words may only re-order around the version digits)
    for _path, replacements in audit.contract_bump_edits().items():
        for old, new, _count in replacements:
            assert old != new
            skeleton_ok = "".join(ch for ch in old if not ch.isdigit()) == "".join(ch for ch in new if not ch.isdigit())
            version_list_ok = (
                "solver_contract" in old
                or "resolved_in_contract" in old
                or "in (" in new
                or "effective_dt_rule" in old
                or "contract 11" in old
            )
            assert skeleton_ok or version_list_ok, (old, new)


def test_solver_contract_bump_is_metadata_only_in_phasefield(tmp_path, monkeypatch) -> None:
    """The phasefield diff under promotion touches only the version constant."""

    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    text = Path(pf.__file__).read_text()
    assert "SOLVER_CONTRACT_VERSION = " in text
    # simulate: the promotion edit exists and replaces exactly one line
    old = "SOLVER_CONTRACT_VERSION = 11"
    new = "SOLVER_CONTRACT_VERSION = 12"
    if f"\n{old}" in text:
        edits = audit._rewrite_exact(Path(pf.__file__), [(old, new, 1)])
        updated = Path(pf.__file__).read_text()
        assert new in updated and old not in updated
        assert len(edits) == 1
        # restore
        audit._rewrite_exact(Path(pf.__file__), [(new, old, 1)])
    else:
        assert f"\n{new}" in text  # already bumped (post-promotion checkout)


# ---------------------------------------------------------------------------
# audit machinery on tiny configurations (sections 2-11)
# ---------------------------------------------------------------------------


def test_first_crossing_is_exact_on_a_tiny_case(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "CACHE_DIR", tmp_path)
    case = _primary_case()
    args = audit.RunArgs(N=96, ds=2, dt=4e-3, nsteps=160, save_every=80)
    p, solid, initial = pf.build_case(case, N=args.N, dt=args.dt)
    _final, phi, _u, _v = pf.rollout(initial, solid, p, 160, save_every=1)
    phi_np = np.asarray(phi)
    per_step = np.maximum(phi_np - 1.0, -phi_np).reshape(phi_np.shape[0], -1).max(axis=1)
    assert per_step.max() > 0.0, "test window must contain the wall impact"
    threshold = float(per_step.max()) / 2.0
    record = audit.find_first_crossing(case, args, max_steps=160, threshold=threshold)
    expected = int(np.flatnonzero(per_step > threshold)[0])
    if record["crossed"]:
        assert record["first_crossing_public_step"] == expected
        assert record["prefailure_public_step"] == expected - 1
    else:
        assert record["fallback_peak_state"] is True
        assert int(np.argmax(per_step)) == record["prefailure_public_step"]


def test_replay_is_noop_and_reconstruction_is_bitwise(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "CACHE_DIR", tmp_path)
    case = _primary_case()
    args = audit.RunArgs(N=48, ds=2, dt=2e-3, nsteps=30, save_every=15)
    p, solid, initial = pf.build_case(case, N=args.N, dt=args.dt)
    _final, phi, u, v = pf.rollout(initial, solid, p, 10, save_every=1)
    state = pf.State(
        phi=jnp.asarray(np.asarray(phi)[5]),
        u=jnp.asarray(np.asarray(u)[5]),
        v=jnp.asarray(np.asarray(v)[5]),
        t=jnp.asarray(6 * p.dt),
    )
    noop = audit.verify_replay_is_noop(state, solid, p)
    assert noop["bitwise_identical"] is True
    decomposition = audit.decompose_phase_update(state, solid, p)
    assert decomposition["capture_matches_production_step_bitwise"] is True
    assert decomposition["reconstruction"]["candidate_matches_explicit_increment_bitwise"] is True
    assert decomposition["reconstruction"]["no_surrogate_decomposition"]


def test_public_vs_substep_cfl_distinction(tmp_path) -> None:
    """The cut-cell CFL ratio is a property of the velocity field: it scales with dt (section 6)."""

    case = _primary_case()
    args = audit.RunArgs(N=48, ds=2, dt=3e-3, nsteps=30, save_every=15)
    p, solid, initial = pf.build_case(case, N=args.N, dt=args.dt)
    _final, phi, u, v = pf.rollout(initial, solid, p, 10, save_every=1)
    state = pf.State(
        phi=jnp.asarray(np.asarray(phi)[5]),
        u=jnp.asarray(np.asarray(u)[5]),
        v=jnp.asarray(np.asarray(v)[5]),
        t=jnp.asarray(6 * p.dt),
    )
    public = pf.cutcell_advective_cfl_diagnostic(state.u, state.v, solid, p)
    sub = pf.cutcell_advective_cfl_diagnostic(state.u, state.v, solid, dataclasses.replace(p, dt=p.dt / 3.0))
    assert public["dt_adv_min"] == pytest.approx(sub["dt_adv_min"])  # field property, dt-independent
    assert public["cutcell_advective_cfl_ratio"] == pytest.approx(3.0 * sub["cutcell_advective_cfl_ratio"])


def test_equal_physical_time_schedules_across_dt(tmp_path) -> None:
    """dt and dt/2 cover the same horizon at the same frame cadence (section 10/18)."""

    args = _production_args()
    case = _primary_case()
    full = audit.effective_schedule(case, args.namespace())
    # the half point derives from the production EFFECTIVE dt (under a policy
    # whose cap equals requested dt/2, halving the requested dt is vacuous)
    half_dt = full["effective_dt"] / 2.0
    half = audit.effective_schedule(
        case,
        dataclasses.replace(
            args,
            dt=half_dt,
            nsteps=int(round(full["horizon"] / half_dt)),
            save_every=int(round(full["frame_dt"] / half_dt)),
            timestep_policy="legacy_requested_v0",
        ).namespace(),
    )
    assert full["horizon"] == pytest.approx(half["horizon"])
    assert full["frame_dt"] == pytest.approx(half["frame_dt"])
    assert half["nsteps"] == 2 * full["nsteps"]
    assert half["effective_dt"] != full["effective_dt"]


def test_complex_probe_classification_schema(tmp_path, monkeypatch) -> None:
    """The complex probe emits a valid COMPLEX_DIVERGENCE_SHARED_DT_CAUSE classification."""

    monkeypatch.setattr(audit, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(audit, "COMPLEX_PROBE_MAX_STEPS", 3)
    case = audit.baseline_case_matrix()["complex_heldout"]
    args = audit.RunArgs(N=32, ds=2, dt=4e-3, nsteps=20, save_every=10)
    report = audit.complex_divergence_probe(case, args)
    assert report["classification"] in {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"}
    assert set(report["entries"]) == {"production_dt", "dt_half"}


# ---------------------------------------------------------------------------
# lineage/reader (section 20/33)
# ---------------------------------------------------------------------------


def test_reader_accepts_policy_lineage_sample(tmp_path) -> None:
    import production.l1a_data_readiness_exit_audit as l1a2o

    case = _primary_case()
    args = audit.RunArgs(N=32, ds=2, dt=4e-3, nsteps=30, save_every=15)
    schedule = audit.effective_schedule(case, args.namespace())
    policy_record = generator._time_step_policy_record(case, args.namespace(), schedule["effective_dt"])
    fingerprint = generator._dataset_fingerprint(
        case, args.namespace(), schedule["effective_dt"], schedule["nsteps"], schedule["save_every"]
    )
    p, solid, initial = pf.build_case(case, N=args.N, dt=schedule["effective_dt"])
    _final, phi, u, v = pf.rollout(initial, solid, p, schedule["nsteps"], save_every=schedule["save_every"])
    ok, diagnostics = generator._diagnose(
        initial,
        np.asarray(phi),
        np.asarray(u),
        np.asarray(v),
        solid,
        p,
        max_phi_overshoot=args.max_phi_overshoot,
        max_solid_leak=args.max_solid_leak,
        min_total_mass_ratio=args.min_total_mass_ratio,
        max_total_mass_ratio=args.max_total_mass_ratio,
        max_speed=args.max_speed,
    )
    path = tmp_path / "policy_lineage_sample.npz"
    generator._save_case(
        path,
        case,
        p,
        solid,
        np.asarray(phi),
        np.asarray(u),
        np.asarray(v),
        schedule["save_every"],
        args.ds,
        diagnostics,
        fingerprint,
        float("inf"),
        {"audit_role": "unit_test"},
        policy_record,
    )
    assert path.is_file()
    reader = l1a2o.reader_compatibility(path, str(case["split"]))
    assert reader.get("shapes_match_current_model_contract") is True
    with np.load(path, allow_pickle=True) as data:
        case_meta = json.loads(str(data["case"]))
    policy_meta = case_meta.get("time_step_policy")
    assert policy_meta is not None, "the sample must carry the time-step policy record"
    assert policy_meta["time_step_policy_name"] == policy_record["time_step_policy_name"]


def test_baseline_case_matrix_matches_the_l1a2o_freeze() -> None:
    matrix = audit.baseline_case_matrix()
    assert len(matrix) == 8
    flat_labels = [label for label in matrix if label.startswith("flat_")]
    assert len(flat_labels) == 6
    assert "pillar_training" in matrix and "complex_heldout" in matrix
    seeds = {label: spec[3] for label, spec in zip(matrix, audit.BASELINE_CASE_SPECS)}
    assert seeds["flat_we100_ct050"] == 6
    assert seeds["complex_heldout"] == 100
