"""L1A-2n inactive phase-state coupling audit tests (sections 40-41)."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phasefield as pf
from production import chns_nonstationarity_audit as chns
from production import inactive_phase_coupling_audit as audit
from production import inactive_phase_ghost_prototypes as ghosts
from production import stationarity_metric_domain_audit as l1a2m

# ---------------------------------------------------------------------------
# shared small fixtures (N = 48 quick-case geometry; hermetic, no checkpoints)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_case():
    p, solid, seed, config = chns._make_case(60.0, N_value=48, dt=chns.DT, M=chns.M_REF)
    return p, solid, seed, config


@pytest.fixture(scope="module")
def small_context(small_case):
    p, solid, seed, config = small_case
    state = pf.advance(seed, solid, p, 4)
    return audit.CaseContext({"p": p, "solid": solid, "state": state, "config": config, "step": 4})


# ---------------------------------------------------------------------------
# provenance / partition (sections 5-6)
# ---------------------------------------------------------------------------


def test_l1a2n_accepts_only_frozen_provenance_states(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "L1A2M_ARTIFACT_ROOT", tmp_path)
    # fail-closed on a missing upstream anchor and on a missing checkpoint payload
    with pytest.raises(Exception, match="lacks|required|missing"):
        audit._accept_chns_state("authority_060", {"cases": {}})
    with pytest.raises(Exception, match="lacks|required|missing"):
        audit._accept_ch_only_state({"phase_only_comparison": {}})
    # a malformed npz payload must also be rejected, never silently accepted
    (tmp_path / "checkpoints").mkdir()
    path = tmp_path / "checkpoints" / "authority_060_production_step_050000.npz"
    path.write_bytes(b"not an npz payload")
    with pytest.raises(Exception):
        audit._accept_chns_state("authority_060", {"cases": {}})


def test_inactive_and_physical_partitions_are_disjoint_complete(small_context):
    partition = small_context.partition
    shells = partition["shells"]
    physical = shells["P"]
    inactive = shells["I0_all"]
    assert not np.any(physical & inactive)
    assert int((physical | inactive).sum()) == small_context.p.Nx * small_context.p.Ny
    assert partition["counts"]["I0_all"] == (
        partition["counts"]["I0_1"] + partition["counts"]["I0_2"] + partition["counts"]["I0_deep"]
    )
    assert partition["open_faces_P_to_I0"] == {"x_faces": 0, "y_faces": 0}


def test_stencil_shells_are_deterministic(small_context):
    volume = small_context.partition["volume"]
    first = audit._shell_partition(volume)
    second = audit._shell_partition(volume)
    for key in ("I0_1", "I0_2", "I0_deep"):
        assert np.array_equal(first[key], second[key])
        assert audit._mask_hash(first[key]) == audit._mask_hash(second[key])


def test_contact_line_inactive_masks_are_deterministic(small_context):
    context = small_context
    args = (
        context.partition["shells"]["I0_all"],
        np.asarray(context.solid.sdf, dtype=np.float64),
        context.partition["cell_x"],
        float(context.p.dx),
        context.partition["contact_lines"]["left_contact_x"],
        context.partition["contact_lines"]["right_contact_x"],
        context.state_arrays["phi"],
    )
    first = audit._region_masks(*args)
    second = audit._region_masks(*args)
    for key in first:
        assert np.array_equal(first[key], second[key])
        assert audit._mask_hash(first[key]) == audit._mask_hash(second[key])


# ---------------------------------------------------------------------------
# perturbation admissibility (sections 9-10)
# ---------------------------------------------------------------------------


def test_inactive_perturbation_is_nonvacuous(small_context):
    mask = small_context.partition["shells"]["I0_1"]
    perturbed, delta = audit._perturb(small_context.state_arrays["phi"], mask, 1.0e-4)
    assert float(np.max(np.abs(delta[mask]))) == 1.0e-4
    assert int(np.count_nonzero(delta)) == int(mask.sum())


def test_inactive_perturbation_changes_only_selected_I0(small_context):
    context = small_context
    mask = context.partition["regions"]["I0_CL_left_2dx"]
    assert mask.any()
    perturbed, delta = audit._perturb(context.state_arrays["phi"], mask, 1.0e-4)
    assert not np.any(delta[~mask])
    record = audit._admissibility(
        context.state_arrays, perturbed, mask, context.solid, context.p, {}, context.geometry_hashes
    )
    assert record["perturbation_nonzero_only_on_selected_I0"]


def test_physical_phi_bitwise_unchanged_before_evaluation(small_context):
    context = small_context
    mask = context.partition["shells"]["I0_1"]
    perturbed, _delta = audit._perturb(context.state_arrays["phi"], mask, 1.0e-2)
    physical = context.partition["shells"]["P"]
    assert np.array_equal(perturbed[physical], context.state_arrays["phi"][physical])
    record = audit._admissibility(
        context.state_arrays, perturbed, mask, context.solid, context.p, {}, context.geometry_hashes
    )
    assert record["phi_P_bitwise_unchanged"] and record["u_bitwise_unchanged"] and record["v_bitwise_unchanged"]


def test_geometry_and_volume_hashes_unchanged(small_context):
    context = small_context
    mask = context.partition["shells"]["I0_2"]
    perturbed, _delta = audit._perturb(context.state_arrays["phi"], mask, 1.0e-4)
    audit._admissibility(context.state_arrays, perturbed, mask, context.solid, context.p, {}, context.geometry_hashes)
    assert audit._geometry_hashes(context.solid, context.p) == context.geometry_hashes
    assert "volume" in context.geometry_hashes and "chi_hard_solid_mask" in context.geometry_hashes


def test_formal_mass_unchanged_before_step(small_context):
    context = small_context
    mask = context.partition["shells"]["I0_1"]
    perturbed, _delta = audit._perturb(context.state_arrays["phi"], mask, 1.0e-2)
    mass_b = audit._formal_mass(context.state_arrays["phi"], context.volume)
    mass_p = audit._formal_mass(perturbed, context.volume)
    assert mass_b == mass_p


def test_authority_checkpoint_not_mutated(tmp_path, small_context):
    payload = np.arange(16, dtype=np.float64)
    path = tmp_path / "authority_checkpoint.npz"
    np.savez(path, phi=payload)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    mask = small_context.partition["shells"]["I0_1"]
    perturbed, _delta = audit._perturb(small_context.state_arrays["phi"], mask, 1.0e-4)
    audit._admissibility(
        small_context.state_arrays,
        perturbed,
        mask,
        small_context.solid,
        small_context.p,
        {},
        small_context.geometry_hashes,
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# dependency tracing (sections 8, 11, 14, 15)
# ---------------------------------------------------------------------------


def test_noop_dependency_trace_reports_noise_floor(small_context):
    noise = small_context.noise
    assert set(noise) == set(audit.LADDER_STAGES)
    for stage in noise.values():
        for field, item in stage.items():
            assert item["bitwise_zero"], (stage, field, item)


def test_operator_delta_is_measured_only_on_physical_domain(small_context):
    context = small_context
    physical = context.physical
    # a delta confined to I0 must contribute nothing to the P-domain measurement
    array_base = context.state_arrays["phi"].copy()
    array_pert = array_base.copy()
    array_pert[~physical] = 1.0e6
    record = audit._field_delta_on_P(array_base, array_pert, physical)
    assert record["linf"] == 0.0 and record["l2"] == 0.0 and not record["changed"]
    # the same physical delta on P must register
    array_pert2 = array_base.copy()
    array_pert2[physical] += 1.0e-3
    record2 = audit._field_delta_on_P(array_base, array_pert2, physical)
    assert record2["changed"] and record2["linf"] == pytest.approx(1.0e-3)


def test_first_changed_operator_classifier_uses_production_order():
    shape = (8, 8)
    physical = np.ones(shape, dtype=bool)
    physical[:, 0] = False
    zero_stage = lambda field: {"value": np.zeros(shape)}  # noqa: E731
    baseline = {stage: zero_stage(stage) for stage in audit.LADDER_STAGES}

    def perturbed_with(changes: dict[str, float]) -> dict:
        out = {stage: zero_stage(stage) for stage in audit.LADDER_STAGES}
        for stage, value in changes.items():
            out[stage] = {"value": np.full(shape, value)}
        return out

    noise = {}
    later = audit._first_changed_operator(
        baseline, perturbed_with({"5_capillary_force": 1e-9, "8_next_physical_state": 1e-3}), physical, noise
    )
    assert later["first_changed_stage"] == "5_capillary_force"
    early = audit._first_changed_operator(
        baseline,
        perturbed_with({"2_chemical_potential_and_wall_terms": 1e-9, "8_next_physical_state": 1e-3}),
        physical,
        noise,
    )
    assert early["first_changed_stage"] == "2_chemical_potential_and_wall_terms"
    none = audit._first_changed_operator(baseline, perturbed_with({}), physical, noise)
    assert none["first_changed_stage"] is None


def test_normalized_sensitivity_scales_on_linear_fixture(small_context):
    context = small_context
    mask = context.partition["shells"]["I0_1"]
    sensitivities = []
    for amplitude in (1.0e-6, 1.0e-4):
        run = context.experiment("I0_1", mask, amplitude)
        stage = run["first_changed_stage"]
        assert stage is not None
        sensitivities.append(max(item["S_l2"] for item in run["sensitivity_at_first_changed"].values()))
    ratio = sensitivities[0] / sensitivities[1]
    assert abs(ratio - 1.0) < 1.0e-6


# ---------------------------------------------------------------------------
# synthetic diagnostic fixtures (section 41) through the real classifier
# ---------------------------------------------------------------------------


_SYNTHETIC_STAGES = (
    ("1_properties_pointwise", ("rho", "nu")),
    ("2_chemical_potential_and_wall_terms", ("mu",)),
    ("3_phase_gradients_and_ch_ingredients", ("grad_phi_y",)),
    ("4_ch_face_flux", ("ch_flux_x",)),
    ("5_capillary_force", ("cap_y",)),
    ("6_momentum_predictor_inputs", ("u_rhs",)),
    ("7_pressure_rhs_and_projected_velocity", ("u_projected",)),
    ("8_next_physical_state", ("phi_next",)),
    ("9_contact_line_and_angle_observables", ("measured_angle_deg",)),
)


def _synthetic_ladder(values: dict[str, dict[str, np.ndarray]], shape) -> dict:
    out = {}
    for stage, fields in _SYNTHETIC_STAGES:
        stage_values = values.get(stage, {})
        out[stage] = {name: (stage_values[name] if name in stage_values else np.zeros(shape)) for name in fields}
    return out


def _synthetic_case():
    shape = (10, 12)
    physical = np.ones(shape, dtype=bool)
    physical[:, :2] = False
    return shape, physical


def test_fixture_pointwise_operator_has_no_inactive_coupling():
    shape, physical = _synthetic_case()
    baseline = _synthetic_ladder({}, shape)
    perturbed = _synthetic_ladder({}, shape)
    classifier = audit._first_changed_operator(baseline, perturbed, physical, {})
    assert classifier["first_changed_stage"] is None


def test_fixture_one_shell_stencil_localizes_to_I0_1():
    shape, physical = _synthetic_case()
    inactive = ~physical
    shells = audit._shell_partition(np.where(physical, 1.0, 0.0))
    assert shells["I0_1"][:, 1].any() and not shells["I0_2"][:, 1].any()
    # a one-shell stencil: physical cells adjacent to the perturbed shell respond
    response = np.zeros(shape)
    touching = np.zeros(shape, dtype=bool)
    for shifted in audit._neighbour_indices(shells["I0_1"], *shape):
        touching |= shifted
    response[touching & physical] = 1.0e-6
    baseline = _synthetic_ladder({"3_phase_gradients_and_ch_ingredients": {"grad_phi_y": np.zeros(shape)}}, shape)
    perturbed = _synthetic_ladder({"3_phase_gradients_and_ch_ingredients": {"grad_phi_y": response}}, shape)
    classifier = audit._first_changed_operator(baseline, perturbed, physical, {})
    assert classifier["first_changed_stage"] == "3_phase_gradients_and_ch_ingredients"
    assert inactive.any()


def test_fixture_two_shell_stencil_responds_at_I0_2():
    shape, physical = _synthetic_case()
    shells = audit._shell_partition(np.where(physical, 1.0, 0.0))
    # a two-shell stencil reaches THROUGH shell 1: physical cells within graph distance 2
    # of the perturbed I0_2 cells respond
    response = np.zeros(shape)
    touching = shells["I0_2"].copy()
    for _ in range(2):
        nxt = touching.copy()
        for shifted in audit._neighbour_indices(touching, *shape):
            nxt |= shifted
        touching = nxt
    response[touching & physical] = 1.0e-6
    baseline = _synthetic_ladder({"3_phase_gradients_and_ch_ingredients": {"grad_phi_y": np.zeros(shape)}}, shape)
    perturbed = _synthetic_ladder({"3_phase_gradients_and_ch_ingredients": {"grad_phi_y": response}}, shape)
    classifier = audit._first_changed_operator(baseline, perturbed, physical, {})
    assert classifier["first_changed_stage"] == "3_phase_gradients_and_ch_ingredients"
    assert int(shells["I0_2"].sum()) > 0


def test_fixture_contaminated_input_with_output_masking_is_not_coupling():
    shape, physical = _synthetic_case()
    # the operator reads the wild inactive storage internally but masks its output to P:
    # on the physical domain the result must be bitwise identical, so no coupling is reported
    wild = np.zeros(shape)
    wild[:, 0] = 1.0e3
    masked_out = np.where(physical, 0.0, wild)
    baseline = _synthetic_ladder({"5_capillary_force": {"cap_y": masked_out}}, shape)
    perturbed = _synthetic_ladder({"5_capillary_force": {"cap_y": masked_out}}, shape)
    classifier = audit._first_changed_operator(baseline, perturbed, physical, {})
    assert classifier["first_changed_stage"] is None


def test_fixture_boundary_consistent_ghost_reconstruction_removes_coupling(small_case):
    p, solid, seed, _config = small_case
    state = pf.advance(seed, solid, p, 4)
    phi = np.asarray(state.phi, dtype=np.float64)
    volume = np.asarray(pf.phase_transport_operator(solid, p).volume, dtype=np.float64)
    inactive = volume == 0.0
    phi_alt = phi.copy()
    phi_alt[inactive] = 0.75
    ghost_a, meta_a = ghosts.ghost_values(phi, solid, p, "boundary_consistent_ghost_v1")
    ghost_b, meta_b = ghosts.ghost_values(phi_alt, solid, p, "boundary_consistent_ghost_v1")
    assert meta_a["changed_inactive_cells"] > 0 and meta_b["changed_inactive_cells"] > 0
    # the ghost construction must be bitwise independent of the raw inactive storage
    assert np.array_equal(ghost_a[inactive], ghost_b[inactive])
    assert np.array_equal(ghost_a[~inactive], phi[~inactive])


# ---------------------------------------------------------------------------
# diagnostic prototypes (sections 26-29)
# ---------------------------------------------------------------------------


def test_ghost_prototype_not_reachable_from_production_default(small_case):
    source = Path(pf.__file__).read_text()
    assert "inactive_phase_ghost_prototypes" not in source
    # importing the prototypes module must not alter any production operator output
    p, solid, seed, _config = small_case
    before = np.asarray(pf.chemical_potential(seed.phi, solid, p), dtype=np.float64).copy()
    importlib.reload(ghosts)
    after = np.asarray(pf.chemical_potential(seed.phi, solid, p), dtype=np.float64)
    assert np.array_equal(before, after)


def test_ghost_prototype_ignores_raw_I0_storage(small_case):
    p, solid, seed, _config = small_case
    state = pf.advance(seed, solid, p, 4)
    phi = np.asarray(state.phi, dtype=np.float64)
    volume = np.asarray(pf.phase_transport_operator(solid, p).volume, dtype=np.float64)
    inactive = volume == 0.0
    physical = ~inactive
    phi_alt = phi.copy()
    phi_alt[inactive] += 0.5
    for closure in ("boundary_consistent_ghost_v1", "nearest_physical_extension_v1"):
        base = ghosts.capillary_acceleration_diagnostic(phi, None, None, solid, p, closure)
        alt = ghosts.capillary_acceleration_diagnostic(phi_alt, None, None, solid, p, closure)
        # invariance is required on the physical domain; the I0 cells themselves carry the
        # perturbed raw storage by construction
        assert np.array_equal(base["cap_y"][physical], alt["cap_y"][physical]), closure
        assert np.array_equal(base["cap_x"][physical], alt["cap_x"][physical]), closure


def test_operator_side_closure_removes_upstream_not_just_output_masking(small_case):
    p, solid, seed, _config = small_case
    state = pf.advance(seed, solid, p, 4)
    phi = np.asarray(state.phi, dtype=np.float64)
    volume = np.asarray(pf.phase_transport_operator(solid, p).volume, dtype=np.float64)
    inactive = volume == 0.0
    physical = ~inactive
    phi_pert = phi.copy()
    phi_pert[inactive] += 1.0e-3
    production_base = np.asarray(pf._ddy(jnp.asarray(phi), p.dy))
    production_pert = np.asarray(pf._ddy(jnp.asarray(phi_pert), p.dy))
    # production central difference at the wall-adjacent physical row reads the inactive row
    assert np.any((production_pert - production_base)[physical] != 0.0)
    closed_base = ghosts.one_sided_cap_closure_v1(phi, solid, p)["cap_y"]
    closed_pert = ghosts.one_sided_cap_closure_v1(phi_pert, solid, p)["cap_y"]
    assert not np.any((closed_pert - closed_base)[physical])
    # and the closure is not the production output masked: it differs from production on the
    # unperturbed state exactly at the wall-adjacent physical cells
    production_cap_y = (
        (pf.SIGMA_NORM / p.We)
        * np.asarray(pf.chemical_potential(jnp.asarray(phi), solid, p))
        * production_base
        / p.rho_l
    )
    difference = (closed_base - production_cap_y)[physical]
    assert np.any(difference != 0.0)


def test_candidate_does_not_change_contract11_default(small_case):
    assert pf.SOLVER_CONTRACT_VERSION == 12
    p, solid, seed, _config = small_case
    reference = pf.step(seed, solid, p)
    ghosts.ghost_values(np.asarray(seed.phi), solid, p, "boundary_consistent_ghost_v1")
    ghosts.one_sided_cap_closure_v1(np.asarray(seed.phi), solid, p)
    again = pf.step(seed, solid, p)
    assert np.array_equal(np.asarray(reference.phi), np.asarray(again.phi))
    assert np.array_equal(np.asarray(reference.u), np.asarray(again.u))


# ---------------------------------------------------------------------------
# anti-cheating (section 40, final block)
# ---------------------------------------------------------------------------


def test_l1a2n_keeps_contract11():
    assert audit.SOLVER_CONTRACT == 11
    assert pf.SOLVER_CONTRACT_VERSION == 12


FROZEN_SOURCE_HASHES = {
    "_ddx": "f0d4e8c3f2fd305c",
    "_ddy": "849125de5fd70299",
    "_lap": "488c2a5aa3dce9ae",
    "fluid_laplacian": "2a2ec3617c28c5a4",
    "graph_stiffness_apply": "497e6fcdcb028468",
    "chemical_potential": "1d7e134d592d0c0a",
    "_explicit_chemical_potential": "6ec023289eb4d0ab",
    "rhs": "15f98bdef39a4b08",
    "step_with_diagnostics": "21921d01ce8fcf3c",
    "phase_advective_fluxes": "efded9429908b3ab",
    "chemical_potential_fluxes": "b69c0fcf574c5d2a",
    "wall_energy_density": "ddd5840e830c1541",
    "wall_switch": "53f069b60927b2eb",
    "wall_switch_derivative": "b4302b6b2ed7de45",
    "natural_wall_normal_derivative": "59ace3e41a2e8dad",
    "rho_of": "d01b2a919bd42cd0",
    "nu_of": "31892097f41625de",
    "poisson_solve": "a2dcb5a1e0590d43",
    "solve_ch_implicit": "aef77e995468c200",
}

FROZEN_PHASEFIELD_FILE_SHA256_PREFIX = "4790c6235dd763db"
#: L1A-2p contract promotion (11 -> 12) re-sealed the file hash with its
#: sanctioned metadata-only edit (the SOLVER_CONTRACT_VERSION constant; see
#: evidence/l1a2p/contract12_promotion_report.json). The physics operators stay
#: sealed by test_frozen_production_operators_unchanged above.
PROMOTED_PHASEFIELD_FILE_SHA256_PREFIX = "ebb249a22fa2065f"
ACTIVE_PHASEFIELD_FILE_SHA256_PREFIXES = (
    FROZEN_PHASEFIELD_FILE_SHA256_PREFIX,
    PROMOTED_PHASEFIELD_FILE_SHA256_PREFIX,
)


def _source_hash(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(FROZEN_SOURCE_HASHES))
def test_frozen_production_operators_unchanged(name):
    assert _source_hash(getattr(pf, name)) == FROZEN_SOURCE_HASHES[name], name


def test_no_production_phi_semantics_change():
    digest = hashlib.sha256(Path(pf.__file__).read_bytes()).hexdigest()[:16]
    assert digest in ACTIVE_PHASEFIELD_FILE_SHA256_PREFIXES
    manifest = json.loads((Path(pf.__file__).resolve().parent / "evidence" / "l1a2l" / "manifest.json").read_text())
    # the frozen L1A-2l evidence keeps its original pre-promotion seal
    assert manifest["source_hashes"]["phasefield"].startswith(FROZEN_PHASEFIELD_FILE_SHA256_PREFIX)


def test_no_production_stencil_change():
    for name in ("_ddx", "_ddy", "_lap", "fluid_laplacian", "graph_stiffness_apply", "solve_ch_implicit"):
        assert _source_hash(getattr(pf, name)) == FROZEN_SOURCE_HASHES[name], name


def test_no_production_wetting_change():
    for name in ("wall_energy_density", "wall_switch", "wall_switch_derivative", "natural_wall_normal_derivative"):
        assert _source_hash(getattr(pf, name)) == FROZEN_SOURCE_HASHES[name], name


def test_no_production_capillary_change():
    for name in ("rhs", "chemical_potential_fluxes", "phase_advective_fluxes"):
        assert _source_hash(getattr(pf, name)) == FROZEN_SOURCE_HASHES[name], name


def test_no_production_rho_nu_change():
    for name in ("rho_of", "nu_of"):
        assert _source_hash(getattr(pf, name)) == FROZEN_SOURCE_HASHES[name], name


def test_no_threshold_change():
    assert nwa_criteria_phase_rate_tol() == pytest.approx(1.0e-3)
    assert nwa_criteria_angle_tol() == pytest.approx(0.10)
    assert chns.DT == pytest.approx(4.0e-3)
    assert chns.M_REF == pytest.approx(2.0e-3)
    assert audit.REL_TOLERANCE == 1.0e-12  # audit-side tolerance documented, production untouched


def nwa_criteria_phase_rate_tol():
    from production import nonneutral_wetting_audit as nwa

    return nwa.CRITERIA["phase_rate_l2_tol"]


def nwa_criteria_angle_tol():
    from production import nonneutral_wetting_audit as nwa

    return nwa.CRITERIA["angle_tol_deg"]


def test_w_contact_angle_remains_open():
    evidence = Path(pf.__file__).resolve().parent / "evidence" / "l1a2j" / "chns_nonstationarity_report.json"
    report = json.loads(evidence.read_text())
    acceptance = report["acceptance"]
    assert acceptance["production_60_degree_stationarity_gate_at_50k"] is False
    assert acceptance["status"] == "production_gate_not_passed_at_50k"
    assert audit.MERGED_MAIN_SHA == l1a2m_git_sha_of_merge()
    assert report is not None


def l1a2m_git_sha_of_merge() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "log", "--format=%H", "-1", "ee15098ab0600b2bdf450162a3c89ec7ce2950d1"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unavailable"
