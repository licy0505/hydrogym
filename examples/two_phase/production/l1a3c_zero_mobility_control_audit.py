"""L1A-3c: fail-closed admissibility and one fully coupled zero-mobility arm.

No solver expressions are edited. Target execution requires a committed source/test/protocol
boundary and a verified baseline replay. An exclusive execution receipt forbids repeating either arm.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import phasefield as pf
from production import frozen_chns_early_window_transport_attribution_audit as capture
from production import impact_initialization_compatibility_audit as initializer
from production import l1a2z_sequential_ownbase_attribution_audit as stage_z
from production import l1a3b_support_indexed_observable_audit as observable

ROOT = Path(__file__).resolve().parents[3]
TP = ROOT / "examples/two_phase"
EVIDENCE = TP / "evidence/l1a3c"
PRESPEC = TP / "l1a3c_prespecification.md"
MODULE = Path(__file__).resolve()
TEST = TP / "tests/test_l1a3c_zero_mobility_control_audit.py"
PUBLISHED_3B = "144d92f273eaaef68175baf72620ec3919275b31"
BRANCH = "arena/02b6c71b-hydrogym"
PHYSICAL_STATUS = "PHYSICAL_MECHANISM_NOT_ESTABLISHED"
ENDPOINTS = {"RAY_95": 0.0626486818930445, "RAY_96": 0.0626486836962457}
VERDICTS = (
    "ZERO_MOBILITY_ADMISSIBILITY_BLOCKED",
    "CONTROL_PROTOCOL_OR_BASELINE_BLOCKED",
    "CONTROL_OBSERVABLE_INVALID",
    "CONTROL_PAIRED_RESPONSE_RECORDED",
)
# No independently justified total uncertainty/sensitivity criterion exists. The stronger
# EARLY_WINDOW_MOBILITY_SENSITIVITY_RESOLVED verdict is intentionally unreachable in this freeze.
OWNED = (
    str(MODULE.relative_to(ROOT)),
    str(TEST.relative_to(ROOT)),
    str(PRESPEC.relative_to(ROOT)),
    "examples/two_phase/evidence/l1a3c/",
)


class Blocked(RuntimeError):
    """An intervention, lineage, evidence, or measurement gate failed."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=initializer._json_default) + "\n"
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require(holds: bool, message: str) -> None:
    if not holds:
        raise Blocked(message)


def hashes(state: Any) -> dict[str, str]:
    return initializer._state_hashes(state)


def sources() -> dict[str, str]:
    paths = {MODULE, TEST, PRESPEC, TP / "phasefield.py", Path(capture.__file__), Path(observable.__file__)}
    for stage in ("l1a2z", "l1a3b"):
        d = TP / "evidence" / stage
        m = load(d / "manifest.json")
        paths.add(d / "manifest.json")
        if stage == "l1a2z":
            paths.update(ROOT / v for v in load(d / "source_and_provenance.json")["source_paths"].values())
        else:
            paths.update(observable.SOURCE_PATHS.values())
        require(all(sha(d / name) == digest for name, digest in m["files"].items()), f"{stage} evidence tampered")
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def prerequisites() -> dict[str, Any]:
    z = stage_z.verify_evidence_manifest(TP / "evidence/l1a2z")
    m = load(TP / "evidence/l1a3b/manifest.json")
    require(m["verdict"] == "SUPPORT_INDEXED_OBSERVABLE_FROZEN", "observable is not frozen")
    require(not observable.verify_bound_files(TP / "evidence/l1a3b", m), "L1A-3b bound evidence mismatch")
    for key, path in observable.SOURCE_PATHS.items():
        require(sha(path) == m["source_hashes"][key], f"L1A-3b source mismatch {key}")
    hist = load(TP / "evidence/l1a3b/historical_validation.json")
    for ray in observable.RAY_IDS:
        require(
            abs(hist["endpoints"][ray]["toward_solid_displacement_dx"] - ENDPOINTS[ray]) <= 8e-15,
            f"sealed endpoint mismatch {ray}",
        )
    return {
        "l1a2z": z,
        "l1a3b_content_verified": True,
        "l1a3b_prior_head_consumer_verified": PUBLISHED_3B,
        "note": "Stage-owned live preflight of L1A-3b is not reused after a successor commit; content is verified.",
    }


def static_mobility_paths() -> dict[str, Any]:
    text = (TP / "phasefield.py").read_text()
    tree = ast.parse(text)
    records = []
    for func in tree.body:
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and node.attr == "M":
                    records.append(
                        {
                            "function": func.name,
                            "line": node.lineno,
                            "expression": text.splitlines()[node.lineno - 1].strip(),
                        }
                    )
    require(
        {r["function"] for r in records}
        == {"chemical_potential_fluxes", "solve_ch_implicit", "implicit_solve_diagnostics", "_phase_update"},
        "mobility source-path inventory changed",
    )
    return {
        "source_sha256": sha(TP / "phasefield.py"),
        "all_M_attribute_occurrences": records,
        "explicit": "J_CH,f=-M*w_f*(mu_exp,j-mu_exp,i); b=phi+h*(a-div(J_CH))",
        "implicit": "A=I+h*M*eps*L^2; L=V_safe^-1*K; solve A*phi_new=b in the V inner product",
        "zero_limit": "J_CH=0, A=I, correction_rhs=b-A*b=0; zero CG iterations, phi_new=b",
        "conditioning": "A=I has condition number 1; no inverse mobility or zero-mobility preconditioner",
        "chemical_potential": "mu=fprime(phi)/eps+wall_derivative-eps*fluid_laplacian(phi); independent of M",
        "momentum": "rhs retains +(SIGMA_NORM/We)*mu*grad(phi)/rho_l, viscosity, advection, gravity flag",
        "pressure_and_Brinkman": "unchanged damping then frozen central-D/G periodic pressure projection",
        "boundary": "shared apertures close wall/seam flux; no separate mobility wall flux",
        "legacy_unexecuted": "projection_legacy FFT denominator=1+h*M*eps*m2^2; at zero it is 1; not selected",
        "normalization": "CG floors/denominators depend on rhs/residual, not 1/M; zero residual skips body",
        "conservation": "sum(V*div(face_flux)) telescopes; implicit identity adds no exchange at M=0",
        "limitations": "Finite fields/geometry only; discrete advection persists; no material-velocity claim.",
    }


def synthetic_bundle() -> tuple[Any, Any]:
    p = pf.PhaseFieldParams(Nx=16, Ny=16, Lx=2.0, Ly=2.0, M=0.0, dt=0.002)
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.28), p, cos_theta=-0.5)
    x, y = pf.grids(p)
    phi = (0.35 + 0.12 * jnp.sin(2 * jnp.pi * x / p.Lx) * jnp.cos(2 * jnp.pi * y / p.Ly)).astype(jnp.float64)
    u = (0.02 * jnp.cos(2 * jnp.pi * y / p.Ly)).astype(p.dtype)
    v = (0.01 * jnp.sin(2 * jnp.pi * x / p.Lx)).astype(p.dtype)
    return SimpleNamespace(p=p, solid=solid), pf.State(phi, u, v, jnp.asarray(0.0, dtype=p.dtype))


def zero_checks(bundle: Any, state: Any, ingredient: dict[str, Any], new: Any) -> dict[str, Any]:
    p, solid = bundle.p, bundle.solid
    h = float(p.dt) / 3
    adv = pf.advective_phase_source(state.phi, state.u, state.v, solid, p, h)
    expected = np.asarray(state.phi + h * adv)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    mass0 = float(np.sum(volume * np.asarray(state.phi)))
    mass1 = float(np.sum(volume * np.asarray(new.phi)))
    # Arithmetic bound: 64 eps times weighted absolute input/output mass, not a solver-state uncertainty.
    bound = 64 * np.finfo(np.float64).eps * max(1.0, float(np.sum(volume * (abs(expected) + abs(state.phi)))))
    info = ingredient["solve_info"]
    checks = {
        "zero_ch_flux": all(np.count_nonzero(np.asarray(ingredient[k])) == 0 for k in ("ch_x", "ch_y")),
        "zero_implicit_exchange": np.array_equal(np.asarray(new.phi), np.asarray(ingredient["phi_hat"])),
        "advective_limit": np.array_equal(np.asarray(new.phi), expected),
        "cg_identity": bool(info.converged) and int(info.iterations) == 0 and float(info.relative_residual) == 0,
        "mass_conservation": abs(mass1 - mass0) <= bound,
    }
    return {
        "checks": checks,
        "holds": all(checks.values()),
        "mass_delta": mass1 - mass0,
        "arithmetic_mass_check_bound": bound,
    }


def admissibility() -> dict[str, Any]:
    static = static_mobility_paths()
    bundle, state = synthetic_bundle()
    result, ing = capture.substep_capture(*state, bundle)
    new = pf.State(*result)
    zero = zero_checks(bundle, state, ing, new)
    positive = replace(bundle.p, M=0.002)
    rhs0 = pf.rhs(state, bundle.solid, bundle.p)
    rhs1 = pf.rhs(state, bundle.solid, positive)
    expected_u = (
        -pf.div_upwind(state.u, state.v, state.u, bundle.p.dx, bundle.p.dy)
        + pf.nu_of(state.phi, bundle.p) * pf._lap(state.u, bundle.p.dx, bundle.p.dy)
        + (pf.SIGMA_NORM / bundle.p.We) * rhs0[3] * pf._ddx(state.phi, bundle.p.dx) / bundle.p.rho_l
    )
    coupling = np.array_equal(np.asarray(rhs0[1]), np.asarray(expected_u.astype(bundle.p.dtype)))
    independent = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(rhs0, rhs1))
    cap = np.asarray((pf.SIGMA_NORM / bundle.p.We) * rhs0[3] * pf._ddx(state.phi, bundle.p.dx) / bundle.p.rho_l)
    public = pf.step_with_diagnostics(state, bundle.solid, bundle.p)[0]
    carried = state
    for _ in range(3):
        carried = pf.State(*capture.substep_capture(*carried, bundle)[0])
    max_diff = {
        k: float(np.max(abs(np.asarray(getattr(carried, k)) - np.asarray(getattr(public, k)))))
        for k in ("phi", "u", "v")
    }
    positive_flux = pf.chemical_potential_fluxes(rhs1[4], bundle.solid, positive)
    checks = {
        **zero["checks"],
        "rhs_and_mu_independent_of_M": independent,
        "momentum_and_capillary_retained": coupling,
        "nonzero_capillary_probe": float(np.max(abs(cap))) > 0,
        "positive_mobility_discriminated": any(np.count_nonzero(np.asarray(v)) for v in positive_flux),
        "capture_public_equivalence": all(v <= capture.EQUIVALENCE_MAX_PHI_DIFF for v in max_diff.values()),
    }
    return {
        "holds": all(checks.values()),
        "checks": checks,
        "synthetic_grid": [16, 16],
        "target_trajectories_run": 0,
        "discrete_audit": static,
        "zero_check": zero,
        "public_capture_max_difference": max_diff,
    }


def configuration(p: Any) -> dict[str, Any]:
    return {k: str(v) if k == "dtype" else v for k, v in asdict(p).items()}


def intervention_gate(
    base_config: dict[str, Any],
    control_config: dict[str, Any],
    base_hashes: dict[str, str],
    control_hashes: dict[str, str],
    base_geometry: str,
    control_geometry: str,
) -> None:
    require(base_hashes == control_hashes and set(base_hashes) == {"phi", "u", "v"}, "initial state mismatch")
    require(base_geometry == control_geometry, "geometry mismatch")
    require(base_config.get("M") == 0.002 and control_config.get("M") == 0.0, "incorrect mobility")
    require(
        {k: v for k, v in base_config.items() if k != "M"} == {k: v for k, v in control_config.items() if k != "M"},
        "unrelated parameter changed",
    )


def quality_gate(record: dict[str, Any]) -> None:
    """Accept observations only with exact recorded historical failure identities."""
    require(record.get("focused_passed", 0) >= 44 and record.get("focused_failures") == [], "focused tests failed")
    require(
        record.get("ratchet") == "PASS_BASELINE_ONLY" and record.get("added_diagnostics") == 0, "quality ratchet failed"
    )
    prior = load(TP / "evidence/l1a3b/quality_and_blocker_matrix.json")
    expected = set(prior["frozen_adjacent_sweep"]["failing"]) | set(prior["legacy_lineage_suite"]["failing"])
    # The immutable L1A-3b consumer asserts stage-owned commits; on any successor stage this one
    # additional test fails at that preflight, before exercising its tamper probe. Preserve it.
    expected.add(
        "test_l1a3b_support_indexed_observable_audit.py::"
        "test_full_stage_evidence_verifier_rejects_a_mutated_sealed_historical_vector"
    )
    actual = {
        value.removeprefix("tests/").removeprefix("examples/two_phase/tests/")
        for value in record.get("adjacent_failures", [])
    }
    require(actual == {value.removeprefix("tests/") for value in expected}, "adjacent failure inventory differs")
    require(record.get("adjacent_passed", 0) >= 435, "adjacent suites incomplete")


def validate_effect(value: dict[str, Any]) -> None:
    """Reject sign inversion or scalar substitution, independently of reporting order."""
    if value.get("status", "VALID") != "VALID":
        require(value.get("E_dx") is None, "partial invalid vector")
        return
    effects = observable.validate_complete_vector(value["E_dx"])
    b = observable.validate_complete_vector(value["displacement_baseline_dx"])
    c = observable.validate_complete_vector(value["displacement_control_dx"])
    require(effects == {ray: c[ray] - b[ray] for ray in observable.RAY_IDS}, "effect sign or component altered")
    require(value["raw_signed_effect_range_dx"] == [min(effects.values()), max(effects.values())], "range altered")
    require(
        value["absolute_support_to_support_effect_disagreement_dx"] == abs(effects["RAY_95"] - effects["RAY_96"]),
        "support disagreement hidden",
    )


def freeze() -> dict[str, Any]:
    require(git("branch", "--show-current") == BRANCH, "wrong branch")
    require(git("rev-parse", "--is-shallow-repository") == "false", "shallow lineage")
    subprocess.run(["git", "merge-base", "--is-ancestor", PUBLISHED_3B, "HEAD"], cwd=ROOT, check=True)
    require(not git("status", "--porcelain", "--", str(MODULE), str(TEST), str(PRESPEC)), "uncommitted protocol source")
    boundary = git("log", "-1", "--format=%H", "--", str(PRESPEC))
    require(boundary != "", "missing committed prespecification")
    current = sources()
    for name, digest in current.items():
        blob = subprocess.check_output(["git", "show", f"{boundary}:{name}"], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == digest, f"source changed since freeze: {name}")
    require(git("rev-parse", "origin/main") == observable.VERIFIED_MAIN_SHA, "main moved; reseal required")
    for line in git("status", "--porcelain").splitlines():
        path = line[3:].strip().strip('"')
        require(any(path == owned or path.startswith(owned) for owned in OWNED), "foreign dirty path")
    changes = git("diff", "--name-only", PUBLISHED_3B, "HEAD").splitlines()
    require(
        all(any(path == owned or path.startswith(owned) for owned in OWNED) for path in changes),
        "prior-stage or unrelated source changed",
    )
    return {"prespecification_commit": boundary, "source_hashes": current, "holds": True}


@lru_cache(maxsize=1)
def rays() -> list[Any]:
    return observable.frozen_ray_identities(
        load(TP / "evidence/l1a2z/sequential_ownbase_ledger.json"),
        load(TP / "evidence/l1a2z/support_uniqueness_design.json"),
        load(TP / "evidence/l1a2z/initial_state_identity.json"),
        load(TP / "evidence/l1a3a/phase_a_audit.json"),
    )[0]


def read(phi: Any, bundle: Any, previous: dict[str, float], points: Any) -> dict[str, Any]:
    r = observable.frozen_support_indexed_readout(
        phi, bundle.p, rays(), previous, previous_branch_points=points, bundle=bundle
    )
    arithmetic = {}
    if r["status"] == "VALID":
        for ray in rays():
            rr = r["ray_records"][ray.ray_id]
            arithmetic[ray.ray_id] = observable.root_reconstruction_uncertainty(
                phi, ray, rr["selected_segment_index"], r["Q_dx"][ray.ray_id], bundle.p
            )
    r["reconstruction_arithmetic"] = arithmetic
    return r


def serial_read(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in record.items()}


def baseline_compare(records: list[dict[str, Any]], initial: dict[str, str], final: dict[str, str]) -> dict[str, Any]:
    sealed = load(TP / "evidence/l1a2z/sequential_ownbase_ledger.json")
    require(len(records) == 360, "baseline incomplete intermediate reconstruction")
    require(initial == load(TP / "evidence/l1a2z/initial_state_identity.json")["hashes"], "baseline initial identity")
    require(final == {k: sealed["final_state"][f"{k}_hash"] for k in ("phi", "u", "v")}, "baseline final-state hash")
    labels = {int(s["column_index"]): s["label"] for s in sealed["supports"]}
    diffs = []
    for rec in records:
        row = sealed["rows"][rec["substep_index"]]
        name = rec["state_name"]
        require(
            rec["phi_hash"] == row["state_hashes"][name], f"baseline reconstruction hash: {rec['substep_index']}:{name}"
        )
        require(rec["readout"]["status"] == "VALID", "baseline frozen observable invalid")
        for ray in rays():
            actual = rec["readout"]["Q_dx"][ray.ray_id]
            expected = row["per_support"][labels[ray.column_index]][f"q{name[-1]}_over_dx"]
            diffs.append(abs(actual - expected))
    require(max(diffs) <= stage_z.CONTRACT_LEG_TOLERANCE_DX, "baseline root replay discrepancy")
    return {
        "holds": True,
        "state_hashes_match_every_intermediate": True,
        "max_root_difference_dx": max(diffs),
        "full_contour_replay": True,
        "historical_endpoints_dx": ENDPOINTS,
    }


def execution_receipt(arm: str) -> None:
    path = EVIDENCE / f"{arm}_execution_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump({"arm": arm, "status": "STARTED_NO_RETRY_PERMITTED", "head": git("rev-parse", "HEAD")}, stream)


def execute(arm: str) -> dict[str, Any]:
    """Execute one arm; no duplicates, retries, public-step shadow trajectories or solver edits."""
    require(arm in ("baseline", "control"), "unknown arm")
    bound = freeze()
    pre = prerequisites()
    a = admissibility()
    require(a["holds"], "zero-mobility admissibility failed")
    approval = load(EVIDENCE / "precontrol_tests.json")
    quality_gate(approval)
    require(
        approval["passed"] is True and approval["source_hashes"] == bound["source_hashes"], "tests not frozen/passed"
    )
    baseline = None
    if arm == "control":
        baseline = load(EVIDENCE / "baseline.json")
        require(not baseline["failures"], "baseline numerical gates failed")
        require(baseline.get("integrity", {}).get("holds") is True, "baseline not verified")
        require(baseline["freeze"] == bound, "baseline source boundary differs")
        require(
            sha(EVIDENCE / "baseline.json") == load(EVIDENCE / "baseline_verified.json")["sha256"],
            "baseline replay tampered",
        )
    bundle, state = capture._bundle_and_state()
    sealed = load(TP / "evidence/l1a2z/initial_state_identity.json")
    require(hashes(state) == sealed["hashes"], "rebuilt target initial state mismatch")
    require(float(state.t) == 0.0, "initial time mismatch")
    identity = capture.initial_state_identity()
    require(identity["geometry_hash"] == sealed["geometry_hash"], "rebuilt geometry mismatch")
    require(identity["parameter_hash"] == sealed["parameter_hash"], "rebuilt baseline parameters mismatch")
    base_config = configuration(bundle.p)
    if arm == "control":
        bundle = replace(bundle, p=replace(bundle.p, M=0.0))
    config = configuration(bundle.p)
    if arm == "control":
        intervention_gate(
            base_config,
            config,
            baseline["initial_hashes"],
            hashes(state),
            baseline["geometry_hash"],
            identity["geometry_hash"],
        )
    initial_hashes = hashes(state)
    previous = {r.ray_id: r.initial_root_over_dx for r in rays()}
    points = None
    r0 = read(state.phi, bundle, previous, points)
    require(r0["status"] == "VALID", "target initial readout invalid")
    points = r0["branch_points_for_next_state"]
    volume = np.asarray(pf.phase_control_volumes(bundle.solid, bundle.p), dtype=np.float64)
    records, balances, failures = [], [], []
    contact = {"triggered": False}
    execution_receipt(arm)
    # Once a root is lost, never extrapolate or restart tracking from a newly preferred branch.
    tracking_alive = True
    for index in range(120):
        old = state
        result, ing = capture.substep_capture(*state, bundle)
        state = pf.State(*result)
        fields = {"s1": old.phi + ing["d_adv"], "s2": ing["phi_hat"], "s3": state.phi}
        for name, field in fields.items():
            if tracking_alive:
                rr = read(field, bundle, previous, points)
                if rr["status"] == "VALID":
                    previous = rr["Q_dx"]
                    points = rr["branch_points_for_next_state"]
                else:
                    tracking_alive = False
                    failures.extend(f"{index}:{name}:{v}" for v in rr["blocking_failures"])
            else:
                rr = read(field, bundle, previous, points)
                rr["status"] = "INVALID"
                rr["Q_dx"] = {r: None for r in observable.RAY_IDS}
                rr["blocking_failures"].append("TRACKING_ALREADY_LOST_NO_EXTRAPOLATION")
            records.append(
                {
                    "substep_index": index,
                    "macro_step": index // 3 + 1,
                    "state_name": name,
                    "t_before": float(old.t),
                    "t_after": float(state.t),
                    "phi_hash": initializer._array_sha(field),
                    "readout": serial_read(rr),
                }
            )
        info = ing["solve_info"]
        mass0, mass1 = capture._mass(volume, old.phi), capture._mass(volume, state.phi)
        increment_mass = {k: capture._mass(volume, ing[f"d_{k}"]) for k in ("adv", "ch", "imp", "full")}
        apertures = pf.phase_face_apertures(bundle.solid, bundle.p)
        closed = max(
            float(np.max(abs(np.asarray(ing[f"{kind}_{axis}"])[np.asarray(ap) == 0]), initial=0))
            for kind in ("adv", "ch")
            for axis, ap in zip(("x", "y"), apertures)
        )
        balance = {
            "substep_index": index,
            "mass_before": mass0,
            "mass_after": mass1,
            "mass_delta": mass1 - mass0,
            "increment_mass": increment_mass,
            "balance_residual": mass1 - mass0 - increment_mass["full"],
            "closed_face_flux_max": closed,
            "cg_converged": bool(info.converged),
            "cg_iterations": int(info.iterations),
            "cg_relative_residual": float(info.relative_residual),
        }
        balance["operator_recombination_linf"] = float(
            np.max(abs(np.asarray(ing["d_adv"] + ing["d_ch"] + ing["d_imp"] - ing["d_full"])))
        )
        if abs(balance["balance_residual"]) > 1e-12 or closed != 0 or balance["operator_recombination_linf"] > 1e-12:
            failures.append(f"{index}:FROZEN_CONSERVATION_GATE_FAILED")
        if arm == "control":
            balance["zero_checks"] = zero_checks(bundle, old, ing, state)
            if not balance["zero_checks"]["holds"]:
                failures.append(f"{index}:ZERO_LIMIT_NUMERICAL_CHECK_FAILED")
        balances.append(balance)
        if not bool(info.converged) or not all(
            np.isfinite(np.asarray(getattr(state, k))).all() for k in ("phi", "u", "v")
        ):
            failures.append(f"{index}:NONFINITE_OR_SOLVER_FAILURE")
            break
        global_gap = initializer._local_gap(np.asarray(state.phi), 0.5, bundle.solid, bundle.p)
        gap = None if global_gap.get("gap") is None else float(global_gap["gap"]) / float(bundle.p.dx)
        balance["contact_gap_over_dx"] = gap
        if gap is None:
            failures.append(f"{index}:GLOBAL_CONTACT_GUARD_UNAVAILABLE")
            break
        if gap <= stage_z.CONTACT_GAP_THRESHOLD_DX:
            contact = {"triggered": True, "substep_index": index, "gap_over_dx": gap}
            failures.append("CONTACT_GUARD_TRIGGERED")
            break
    result = {
        "arm": arm,
        "freeze": bound,
        "prerequisites": pre,
        "admissibility": a,
        "initial_hashes": initial_hashes,
        "final_hashes": hashes(state),
        "initial_readout": serial_read(r0),
        "geometry_hash": identity["geometry_hash"],
        "baseline_configuration": base_config,
        "configuration": config,
        "records": records,
        "balances": balances,
        "contact_guard": contact,
        "failures": failures,
        "substeps_executed": len(balances),
        "t_final": float(state.t),
        "actual_trajectory_executions": 1,
        "physical_status": PHYSICAL_STATUS,
    }
    if arm == "baseline":
        try:
            result["integrity"] = baseline_compare(records, initial_hashes, result["final_hashes"])
        except Blocked as exc:
            result["integrity"] = {"holds": False, "blocker": str(exc)}
    write(EVIDENCE / f"{arm}.json", result)
    if arm == "baseline" and result["integrity"]["holds"]:
        write(EVIDENCE / "baseline_verified.json", {"sha256": sha(EVIDENCE / "baseline.json")})
    return result


def paired(baseline: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    intervention_gate(
        baseline["configuration"],
        control["configuration"],
        baseline["initial_hashes"],
        control["initial_hashes"],
        baseline["geometry_hash"],
        control["geometry_hash"],
    )
    q0 = baseline["initial_readout"]["Q_dx"]
    require(q0 == control["initial_readout"]["Q_dx"], "initial root vector mismatch")
    require(len(baseline["records"]) == len(control["records"]), "paired window incomplete")
    records = []
    for b, c in zip(baseline["records"], control["records"]):
        require((b["substep_index"], b["state_name"]) == (c["substep_index"], c["state_name"]), "paired state mismatch")
        br, cr = b["readout"], dict(c["readout"])
        if br["status"] == cr["status"] == "VALID":
            correspondence = observable.geometric_branch_correspondence(
                np.asarray(br["branch_points_for_next_state"]),
                np.asarray(cr["branch_points_for_next_state"]),
                SimpleNamespace(dx=0.03125, Lx=6.0, Ly=6.0),
            )
            cr["paired_geometric_correspondence"] = correspondence
            if not correspondence["holds"]:
                cr["status"] = "INVALID"
                cr["blocking_failures"] = ["PAIRED_GEOMETRIC_BRANCH_CORRESPONDENCE_LOST"]
        value = observable.paired_effect_audit(
            q0,
            br,
            cr,
            baseline_initial_hashes=baseline["initial_hashes"],
            control_initial_hashes=control["initial_hashes"],
        )
        validate_effect(value)
        records.append(
            {
                "substep_index": b["substep_index"],
                "state_name": b["state_name"],
                "effect": value,
                "paired_correspondence": cr.get("paired_geometric_correspondence"),
            }
        )
    return {
        "records": records,
        "final": records[-1]["effect"],
        "all_valid": all(r["effect"]["status"] == "VALID" for r in records),
        "total_uncertainty": None,
        "uncertainty_resolution": "UNRESOLVED_NO_TOTAL_SOLVER_STATE_BOUND",
        "support_independence": "UNRESOLVED_NO_JUSTIFIED_BINARY_THRESHOLD",
        "physical_status": PHYSICAL_STATUS,
    }


def verdict(admissible: bool, protocol: bool, executed: bool, valid: bool) -> str:
    if not admissible:
        return VERDICTS[0]
    if not protocol or not executed:
        return VERDICTS[1]
    return VERDICTS[3] if valid else VERDICTS[2]


def validate_report(report: dict[str, Any]) -> None:
    require(report.get("physical_status") == PHYSICAL_STATUS, "physical mechanism promotion forbidden")
    require(report.get("verdict") == verdict(**report["decision_inputs"]), "promotion after failed gate")
    require(report.get("total_uncertainty") is None, "unsupported total uncertainty claim")
    require(report.get("support_independence") == "UNRESOLVED_NO_JUSTIFIED_BINARY_THRESHOLD", "unsupported invariance")
    require(report.get("scalar_effect") is None, "averaging or preferred component forbidden")


def seal(report: dict[str, Any], files: list[str]) -> None:
    validate_report(report)
    write(EVIDENCE / "l1a3c_final_report.json", report)
    names = sorted(set(files + ["l1a3c_final_report.json", "l1a3c_final_report.md"]))
    write(
        EVIDENCE / "manifest.json",
        {
            "stage": "L1A-3c",
            "files": {name: sha(EVIDENCE / name) for name in names},
            "source_hashes": sources(),
            "prespecification_commit": freeze()["prespecification_commit"],
            "verdict": report["verdict"],
            "control_trajectories_run": int(report["decision_inputs"]["executed"]),
            "baseline_trajectories_run": int((EVIDENCE / "baseline_execution_receipt.json").exists()),
            "physical_status": PHYSICAL_STATUS,
        },
    )


def verify(directory: Path = EVIDENCE) -> dict[str, Any]:
    m = load(directory / "manifest.json")
    require(m["stage"] == "L1A-3c", "wrong evidence stage")
    require(m["source_hashes"] == sources(), "source hash mismatch")
    for name, digest in m["files"].items():
        require(Path(name).name == name and sha(directory / name) == digest, f"tampered evidence {name}")
    report = load(directory / "l1a3c_final_report.json")
    validate_report(report)
    frozen_admissibility = load(directory / "admissibility.json")
    require(report["decision_inputs"]["admissible"] == frozen_admissibility["holds"], "admissibility decision altered")
    receipt = (directory / "control_execution_receipt.json").exists()
    require(report["decision_inputs"]["executed"] == receipt, "execution count altered")
    if receipt:
        control = load(directory / "control.json")
        baseline = load(directory / "baseline.json")
        comparison = paired(baseline, control)
        require(
            report["decision_inputs"]["valid"] == (comparison["all_valid"] and not control["failures"]),
            "invalid measurement promoted",
        )
        require(report["paired_result"] == comparison, "paired result altered")
    require(report["verdict"] == m["verdict"], "manifest verdict mismatch")
    require(m["control_trajectories_run"] <= 1, "multiple controls forbidden")
    require(m["prespecification_commit"] == freeze()["prespecification_commit"], "freeze mismatch")
    prerequisites()
    return {
        "verified": True,
        "bound_files": len(m["files"]),
        "source_hashes": len(m["source_hashes"]),
        "verdict": m["verdict"],
        "physical_status": PHYSICAL_STATUS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("admissibility", "baseline", "control", "verify"))
    args = parser.parse_args()
    if args.action == "admissibility":
        write(EVIDENCE / "admissibility.json", admissibility())
    elif args.action == "verify":
        print(json.dumps(verify(), indent=2))
    else:
        execute(args.action)


if __name__ == "__main__":
    main()
