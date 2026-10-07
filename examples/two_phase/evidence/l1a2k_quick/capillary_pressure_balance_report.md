# L1A-2k: capillary–pressure balance audit

- **Profile:** `quick`
- **Status:** `quick_complete`
- **Git SHA:** `f4b10e20a868960d075b5975080e9f821ee171e8`
- **Solver contract:** `11` (required 11; unchanged)
- **Scope:** Diagnostic/root-cause forensic only. No production force, pressure projection, phase evolution, or defaults were changed.
- **Decomposition:** production-operator decomposition via the exact centred periodic D/G and `m2_proj`; not called an exact orthogonal Hodge decomposition.

## Matched snapshot provenance

- Prior L1A-2j report available: `True`; raw artifact directory existed at start: `True`.
- Reuse decision: candidate artifacts will be accepted only by exact checkpoint loader checks (SHA, contract, source hashes, config, state hashes, step)
- Every accepted state is checked for contract, Git SHA, source hashes, exact config/fingerprint, state-array hashes, and step.

| Case | Step | State status | State hash (phi) | Residual/source L2 | Projectable/source L2 | D residual L2 | Poisson solve residual L2 |
|---|---:|---|---|---:|---:|---:|---:|
| authority_060 | — | unmeasured | — | — | — | — | — |
| control_090 | — | unmeasured | — | — | — | — | — |
| control_150 | — | unmeasured | — | — | — | — | — |

## Force-density and acceleration decomposition

## Regional residual and curl localization


## Mechanism matrix

| Candidate | Status | Scope / evidence note |
|---|---|---|
| `CAPILLARY_PRESSURE_IMBALANCE` | **NOT_TESTED** | support establishes the force-subspace property only; it does not establish that this component caused the 60-degree contact-angle nonstationarity |
| `PRESSURE_PROJECTION_LIMITED` | **NOT_TESTED** | the nonzero solenoidal residual is outside the projectable subspace by definition; it is not a Poisson convergence failure |
| `CAPILLARY_DENSITY_SCALING` | **NOT_TESTED** | does not clear the separate model-choice caveat about density-dependent inertia or projection; local-density scaling was not run |
| `VARIABLE_DENSITY_COUPLING` | **NOT_TESTED** | the one-step equal-density blend A/B is identical under the current gravity-off velocity formulation; no variable-density projection or inertia prototype was justified or run |
| `DISCRETE_PRODUCT_RULE_IN_CAPILLARY_FORCE` | **NOT_TESTED** | this is an exact algebraic decomposition, not evidence that Form B is a better model; no force replacement or variant trajectory was used |
| `WALL_ENERGY_MOMENTUM_CONSISTENCY` | **NOT_TESTED** | wall-energy-derived force is measured and localized; no wall-energy, momentum, or contact-angle formulation was modified |
| `BRINKMAN_WALL_COUPLING` | **NOT_TESTED** | spatial overlap is measured; no Brinkman retuning or on/off A/B was run without a separate evidence trigger |
| `CONTACT_LINE_PINNING` | **NOT_TESTED** | 2dx and 4dx neighborhoods/exclusions are measured; no force-threshold test was run |
| `Y_PERIODIC_TOPOLOGY_COUPLING` | **NOT_TESTED** | seam overlap is localized only; no boundary-topology ablation was run |
| `TIME_SPLITTING_OR_DT_SENSITIVITY` | **NOT_TESTED** | dt/2 was not run |
| `MOMENTUM_VISCOSITY_DISCRETIZATION` | **NOT_TESTED** | production nu(phi) and Laplacian remain unchanged |
| `PHASE_KINETICS_LIMITED` | **SUSPECTED** | no phase-mobility or phase-equation ablation was run |
| `MULTIPLE_CONTRIBUTORS` | **NOT_TESTED** | mechanism closure is limited to the measured production-force subspace property |

- **Final root cause of the measured capillary/pressure subspace property:** `INCONCLUSIVE`.
- **Causal root cause of the L1A-2j 60° nonstationarity:** `INCONCLUSIVE`.
- Scope: the measured structural non-projectable capillary-acceleration component only; not a causal explanation of the 60-degree angle-cycle/production-gate failure.

## CHNS-50k versus converged CH-only 60° phi

- **unmeasured:** {'status': 'unmeasured'}

## Static Laplace invariant

- Existing `production.capillary_audit` checks rerun without changing any formulation: all passed = `True`.
- `laplace_jump_sign_and_scale`: passed `True`, observed `{'laplace_ratio': 0.9850772536075545, 'phi_mean_inside_probe': 0.9999174751504804, 'phi_mean_outside_probe': 5.052422088479835e-10}`.

## Unmeasured sections and scope limits

- `local_density_scaling_ab`: **unmeasured_not_triggered_by_constant_rho_l_or_equal_density_one_step_ab**
- `variable_density_projection_or_inertia_prototype`: **unmeasured_not_justified_by_equal_density_ab**
- `brinkman_on_off_ab`: **unmeasured_pending_spatial_overlap_evidence_review**
- `dt_half_matched_physical_time`: **unmeasured_frozen_state_diagnostics_not_insufficient**
- `contact_line_unpinning_threshold`: **unmeasured_no_threshold_test_requested_or_triggered**
- `production_repair_or_validation`: **not_in_scope**

## Artifacts and source provenance

- Field artifacts: `artifacts/l1a2k_quick/fields/` (diagnostic only).
- Machine-readable report: `capillary_pressure_balance_report.json`.
- Operator map: `operator_map.json`; source-function and whole-file SHA256 values are recorded there and in the manifest.
- Production contract 11, capillary denominator, pressure projection, wall energy, phase storage, viscosity, Brinkman defaults, and acceptance thresholds remain unchanged.
- No diagnostic variant is production validation; no authority continuation beyond step 50,000 was performed.
