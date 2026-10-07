"""Regression tests for the L1A-2f alignment evidence runner (``--allow-incomplete`` contract).

The hosted CI runs the alignment runner with a *section subset*
(``--profile quick --sections resolution,precision,laplace,impact,geometry,transport
--allow-incomplete``). Such a run may leave gates unmeasured -- it must never invent a value for a
measurement it did not make, and it must never hide a measurement that *failed*.

These tests pin the two CI regressions directly:

1.  a profile that skips ``primary_float64`` must not crash on ``max()`` of an empty iterable, and the
    formal ``N-CH-MASS-PRECISION`` gate must stay ``measured = False`` (never ``0.0`` / PASS) while
    the runner still exits 0 under ``--allow-incomplete``; and
2.  the cut-cell advective CFL audit must quote the *same* analytic value it measures -- for the
    manufactured uniform field ``(u, v) = (2, -1)`` the empty solid is
    ``cfl dx dy / (2 (dx |u| + dy |v|))``, not the unit-speed ``cfl dx dy / (2 (dx + dy))``.

The long translation / resolution / CHNS matrices stay in the evidence runner and are never part of
CI.
"""

from __future__ import annotations

import json

import pytest

jax = pytest.importorskip("jax")

from production import cutcell_alignment_audit as alignment  # noqa: E402
from production import cutcell_phase_transport_audit as cpta  # noqa: E402


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _gate_map(report_or_gates):
    """``{gate name: gate}`` for a report dict or a list of :class:`Gate` objects."""
    items = report_or_gates["gates"] if isinstance(report_or_gates, dict) else report_or_gates
    return {(item["gate"] if isinstance(item, dict) else item.gate): item for item in items}


def _f64_record(target_deg: float, angle_deg: float, drift: float | None, *, converged: bool = True) -> dict:
    """One ``primary_float64`` section record, as ``section_float64`` writes it."""
    return {
        "target_deg": float(target_deg),
        "equilibrium_angle_deg": float(angle_deg),
        "converged": bool(converged),
        "conserved_mass_drift": drift,
        "mass_drift": 1.0e-2,
        "steps": 10000,
        "dtype": "float64",
        "ch_solver_rtol": 1.0e-8,
    }


def _stub_geometry_section() -> dict:
    """A tiny stand-in for the (expensive) geometry section: the runner never inspects its records."""
    return {"stage": alignment.STAGE, "quick": True, "checks": [], "passed": True, "numbers": {}}


# --------------------------------------------------------------------------------------
#  1. the CI smoke profile: missing primary_float64 must not crash nor fake a measurement
# --------------------------------------------------------------------------------------
def test_cutcell_alignment_quick_allows_missing_primary_float64():
    """The exact CI failure: no ``primary_float64`` section, and no ``max()`` of an empty iterable."""
    numbers = alignment.collect_numbers({})  # the quick profile's section subset, reduced
    gates = alignment.evaluate_gates(numbers)
    formal = _gate_map(gates)["ch_only_mass_drift_formal"]

    assert formal.measured is False
    assert formal.passed is False
    assert formal.value["float64_rtol1e-8_max"] is None

    matrix = numbers["ch_only_matrix_float64"]
    assert matrix["n_targets"] == 0
    assert matrix["mae_deg"] is None
    assert matrix["max_error_deg"] is None
    assert matrix["neutral_error_deg"] is None
    assert numbers["mass_precision"]["float64_rtol1e-8_conserved_mass_drift_max"] is None
    assert "not run" in numbers["mass_precision"]["float64_rtol1e-8_evidence_status"]

    # the same dictionary must stay strict JSON (the report writer rejects NaN/Inf)
    json.dumps(alignment._clean(numbers), allow_nan=False)


def test_cutcell_alignment_quick_run_exits_zero_under_allow_incomplete(tmp_path, monkeypatch):
    """``--profile quick --sections <subset> --allow-incomplete``: report written, exit code 0."""
    monkeypatch.setattr(alignment, "_geometry_section", _stub_geometry_section)
    report = alignment.run_audit(
        profile="quick",
        out=tmp_path / "quick",
        sections=("geometry",),
        allow_incomplete=True,
    )

    assert report["allow_incomplete"] is True
    assert report["not_ready"] is False  # the CLI returns `1 if report["not_ready"] else 0`
    assert report["measured_failures"] == []
    assert report["not_ready_triggers"] == []
    assert "primary_float64" in report["sections_missing"]
    formal = _gate_map(report)["ch_only_mass_drift_formal"]
    assert formal["measured"] is False and formal["value"]["float64_rtol1e-8_max"] is None

    written = json.loads((tmp_path / "quick" / "cutcell_alignment_report.json").read_text())
    assert written["not_ready"] is False
    assert written["allow_incomplete"] is True
    assert _gate_map(written)["ch_only_mass_drift_formal"]["measured"] is False
    manifest = json.loads((tmp_path / "quick" / "manifest.json").read_text())
    assert manifest["files"], "the manifest must carry the SHA256 of the written artifacts"


def test_cutcell_alignment_strict_subset_run_fails_closed(tmp_path, monkeypatch):
    """Without ``--allow-incomplete`` an incomplete profile is refused instead of silently 'ready'."""
    monkeypatch.setattr(alignment, "_geometry_section", _stub_geometry_section)
    with pytest.raises(RuntimeError) as excinfo:
        alignment.run_audit(profile="quick", out=tmp_path / "strict", sections=("geometry",))
    assert "missing sections" in str(excinfo.value)
    assert not (tmp_path / "strict" / "cutcell_alignment_report.json").exists()


def test_cutcell_alignment_allow_incomplete_never_hides_a_measured_failure(tmp_path, monkeypatch):
    """A *measured* failing gate survives ``--allow-incomplete``: readiness stays False."""
    monkeypatch.setattr(alignment, "_geometry_section", _stub_geometry_section)
    original = alignment.evaluate_gates

    def with_injected_failure(numbers):
        return list(original(numbers)) + [
            alignment.Gate("synthetic_measured_failure", False, True, 0.0, 1.0, "injected by the test")
        ]

    monkeypatch.setattr(alignment, "evaluate_gates", with_injected_failure)
    report = alignment.run_audit(
        profile="quick",
        out=tmp_path / "injected",
        sections=("geometry",),
        allow_incomplete=True,
    )

    assert report["measured_failures"] == ["synthetic_measured_failure"]
    assert report["not_ready"] is True
    assert any("synthetic_measured_failure" in trigger for trigger in report["not_ready_triggers"])


# --------------------------------------------------------------------------------------
#  2. impact mass drift is a nonnegative measurement; exact zero must pass, not look missing
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mass_drift, measured, passed",
    [
        (0.0, True, True),
        (alignment.GATES["formal_mass_drift"], True, True),
        (alignment.GATES["formal_mass_drift"] + 1e-9, True, False),
        (None, False, False),
    ],
)
def test_cutcell_alignment_impact_mass_drift_gate_handles_zero_and_missing(mass_drift, measured, passed):
    gate = _gate_map(alignment.evaluate_gates({"impact": {"mass_drift_max": mass_drift}}))["impact_cutcell_mass_drift"]

    assert gate.value == mass_drift
    assert gate.measured is measured
    assert gate.passed is passed


# --------------------------------------------------------------------------------------
#  3. the formal float64 gate: measured exactly when the evidence exists
# --------------------------------------------------------------------------------------
def test_cutcell_alignment_formal_mass_gate_is_measured_on_complete_float64_matrix():
    sections = {
        "primary_float64": [
            _f64_record(60.0, 60.481, 3.1e-5),
            _f64_record(90.0, 89.760, 3.9e-5),
            _f64_record(120.0, 119.287, 2.4e-5),
            _f64_record(150.0, 149.052, 1.1e-5),
        ]
    }
    numbers = alignment.collect_numbers(sections)
    formal = _gate_map(alignment.evaluate_gates(numbers))["ch_only_mass_drift_formal"]

    assert formal.measured is True
    assert formal.passed is True
    assert formal.value["float64_rtol1e-8_max"] == pytest.approx(3.9e-5)
    assert numbers["ch_only_matrix_float64"]["n_targets"] == 4
    assert numbers["ch_only_matrix_float64"]["mae_deg"] == pytest.approx(
        (0.481 + 0.240 + 0.713 + 0.948) / 4.0, rel=1e-6
    )


@pytest.mark.parametrize(
    "rows, reason",
    [
        ([_f64_record(150.0, 149.0, None)], "conserved_mass_drift not recorded"),
        ([_f64_record(150.0, 149.0, 1.0e-5, converged=False)], "converged"),
        ([_f64_record(150.0, 149.0, 9.0e-3)], "gate failure is a measurement"),
    ],
)
def test_cutcell_alignment_formal_mass_gate_is_fail_closed(rows, reason):
    """Missing / unconverged drift is unmeasured; a converged drift above the limit is a failure."""
    numbers = alignment.collect_numbers({"primary_float64": rows})
    formal = _gate_map(alignment.evaluate_gates(numbers))["ch_only_mass_drift_formal"]
    if reason == "gate failure is a measurement":
        assert formal.measured is True and formal.passed is False
    else:
        assert formal.measured is False and formal.passed is False
        assert formal.value["float64_rtol1e-8_max"] is None
        assert reason in numbers["mass_precision"]["float64_rtol1e-8_evidence_status"]


# --------------------------------------------------------------------------------------
#  4. the cut-cell advective CFL audit: quoted analytic value == measured value
# --------------------------------------------------------------------------------------
def test_cutcell_cfl_diagnostic_matches_uniform_cell_closed_form(x64):
    """The audit's detail text and the measured empty-solid value must be the same number."""
    checks, numbers = cpta.audit_cutcell_cfl_diagnostic(N=24)
    by_name = {check.name: check for check in checks}
    assert by_name["cutcell_advective_cfl_diagnostic_is_well_defined"].passed

    empty = next(row for row in numbers["cfl_diagnostic"] if row["surface"] == "empty")
    closed_form = numbers["uniform_cell_dt_adv"]
    unit_speed = numbers["uniform_cell_dt_adv_unit_speed"]

    # the manufactured fixture is (u, v) = (2, -1), so the unit-speed formula is off by 1.5x
    assert (numbers["manufactured_u"], numbers["manufactured_v"]) == (2.0, 1.0)
    assert closed_form * 1.5 == pytest.approx(unit_speed, rel=1e-12)
    assert closed_form == pytest.approx(
        empty["cfl"] * empty["dx"] * empty["dy"] / (2.0 * (empty["dx"] * 2.0 + empty["dy"] * 1.0)),
        rel=1e-12,
    )
    assert numbers["uniform_cell_dt_adv_relative_error"] <= 1.0e-6
    assert numbers["uniform_cell_dt_adv_relative_error"] == pytest.approx(
        abs(empty["dt_adv_min"] - closed_form) / closed_form, rel=1e-9
    )

    # the reported text carries both numbers, and the measured one is the closed form
    detail = by_name["cutcell_advective_cfl_diagnostic_is_well_defined"].detail
    assert f"{closed_form:.10g}" in detail
    assert f"{empty['dt_adv_min']:.10g}" in detail


def test_cutcell_cfl_diagnostic_empty_solid_is_the_interior_cell_value(x64):
    """Every empty-solid cell is a full cell, so all its local time steps are the interior value."""
    checks, numbers = cpta.audit_cutcell_cfl_diagnostic(N=24)
    empty = next(row for row in numbers["cfl_diagnostic"] if row["surface"] == "empty")
    assert empty["n_active_cells"] == 24 * 24
    assert empty["dt_adv_min"] == pytest.approx(numbers["uniform_cell_dt_adv"], rel=1e-6)
    # the diagnostic stays a pure measurement: subcycling remains opt-in and disabled
    assert empty["subcycling"] == "disabled"
