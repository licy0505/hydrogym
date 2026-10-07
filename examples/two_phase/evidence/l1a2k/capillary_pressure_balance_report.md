# L1A-2k: capillary–pressure balance audit

- **Profile:** `forensic`
- **Status:** `complete`
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
| authority_060 | 50000 | measured | `1e76060a3c3db681…` | 0.14439738 | 0.98951979 | 3.31543e-07 | 3.31684e-07 |
| control_090 | 27200 | measured | `591e1870a894e67e…` | 0.15540653 | 0.98785059 | 5.25491e-07 | 5.25239e-07 |
| control_150 | 100000 | measured | `be58678ebea3c6b6…` | 0.1585175 | 0.98735613 | 6.7622e-07 | 6.75538e-07 |

## Force-density and acceleration decomposition

### authority_060

- Force-density L2: `0.0315783674`; raw acceleration L2: `0.0315783674`.
- Production-work-dtype acceleration residual fraction: `0.144397383`; projectable fraction: `0.989519791`.
- Residual divergence L2: `3.31543e-07` (tolerance `2.90746e-05`); reconstruction L∞: `1.16415e-10` (tolerance `7.34017e-08`).
- Constant `rho_l` residual-fraction difference, force density vs acceleration: `0`.
- Product-form identity reconstruction L∞: `2.77556e-17`; projected identity residual L2 error: `2.38896e-08`.
- Frozen-φ matched on/off velocity L2 ratio: `7.887394218790659`; one-step projected impulse: `1.63309654e-05`.

### control_090

- Force-density L2: `0.0459577135`; raw acceleration L2: `0.0459577135`.
- Production-work-dtype acceleration residual fraction: `0.155406526`; projectable fraction: `0.98785059`.
- Residual divergence L2: `5.25491e-07` (tolerance `4.23138e-05`); reconstruction L∞: `2.32831e-10` (tolerance `1.20089e-07`).
- Constant `rho_l` residual-fraction difference, force density vs acceleration: `0`.
- Product-form identity reconstruction L∞: `2.77556e-17`; projected identity residual L2 error: `3.42706e-08`.
- Frozen-φ matched on/off velocity L2 ratio: `4.731904787082291`; one-step projected impulse: `2.53473194e-05`.

### control_150

- Force-density L2: `0.0537680655`; raw acceleration L2: `0.0537680655`.
- Production-work-dtype acceleration residual fraction: `0.158517502`; projectable fraction: `0.987356131`.
- Residual divergence L2: `6.7622e-07` (tolerance `4.95049e-05`); reconstruction L∞: `4.65661e-10` (tolerance `1.71345e-07`).
- Constant `rho_l` residual-fraction difference, force density vs acceleration: `0`.
- Product-form identity reconstruction L∞: `5.55112e-17`; projected identity residual L2 error: `3.57211e-08`.
- Frozen-φ matched on/off velocity L2 ratio: `4.364379466140128`; one-step projected impulse: `2.99723965e-05`.

## Regional residual and curl localization

Masks are deterministic and intentionally overlapping. Norms use float64 reductions on the full uniform momentum-cell area `dx*dy`; residual-energy fractions therefore do not sum to one.

| Region/mask | 60° residual L2 | 90° residual L2 | 150° residual L2 | 60° residual/source | 60/90 L2 | 60/150 L2 | 60° curl L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_momentum_cells` | 0.00455983 | 0.00714213 | 0.00852318 | 0.144397 | 0.638442 | 0.534992 | 0.0283682 |
| `fluid_cell_centres` | 0.0039484 | 0.00599486 | 0.00660829 | 0.125035 | 0.658631 | 0.597492 | 0.00375955 |
| `interface_fluid_phi_005_095` | 0.00199366 | 0.00317266 | 0.00406508 | 0.104632 | 0.628387 | 0.490435 | 0.00279563 |
| `interface_all_phi_005_095` | 0.00199366 | 0.00317266 | 0.00406508 | 0.104632 | 0.628387 | 0.490435 | 0.00279563 |
| `liquid_phi_ge_095_fluid` | 0.00138147 | 0.00266658 | 0.00350122 | 0.0549419 | 0.518067 | 0.394568 | 0.00251283 |
| `gas_phi_le_005_fluid` | 0.00311557 | 0.00433151 | 0.00385825 | 2.25391 | 0.71928 | 0.807508 | 6.59905e-05 |
| `near_wall_fluid_0_2dx` | 0.00147166 | 0.00255592 | 0.00363856 | 0.0552597 | 0.575785 | 0.404462 | 0.00373296 |
| `near_wall_fluid_0_4dx` | 0.00187977 | 0.00321131 | 0.00446476 | 0.0695953 | 0.58536 | 0.421024 | 0.00374508 |
| `solid_cell_centres` | 0.00228083 | 0.00388222 | 0.00538285 | unmeasured | 0.587507 | 0.423722 | 0.028118 |
| `solid_neighborhood_inside_0_2dx` | 0.00170584 | 0.00296247 | 0.00415927 | unmeasured | 0.575816 | 0.410129 | 0.028118 |
| `solid_core_deeper_than_2dx` | 0.00151403 | 0.00250906 | 0.00341694 | unmeasured | 0.603428 | 0.443097 | 8.84931e-08 |
| `brinkman_chi_gt_050` | 0.00228083 | 0.00388222 | 0.00538285 | unmeasured | 0.587507 | 0.423722 | 0.028118 |
| `brinkman_chi_gt_010` | 0.0027144 | 0.00464804 | 0.00649724 | 0.101924 | 0.583988 | 0.417777 | 0.0283647 |
| `y_periodic_seam_2cells` | 0.00150907 | 0.00246941 | 0.00322166 | 50.679 | 0.611103 | 0.468413 | 1.23913e-07 |
| `y_periodic_seam_4cells` | 0.00226399 | 0.00374996 | 0.00499188 | 76.0315 | 0.603738 | 0.453535 | 1.23913e-07 |
| `wall_distance_fluid_0_1dx` | 0.00109946 | 0.0019969 | 0.00291089 | 0.041558 | 0.550583 | 0.377706 | 0.00372701 |
| `wall_distance_solid_0_1dx` | 0.00131484 | 0.00229405 | 0.00319696 | unmeasured | 0.573155 | 0.411279 | 0.028118 |
| `wall_distance_fluid_1_2dx` | 0.000978249 | 0.00159534 | 0.00218308 | 0.320328 | 0.613192 | 0.448104 | 0.000210612 |
| `wall_distance_solid_1_2dx` | 0.00108677 | 0.00187446 | 0.00266064 | unmeasured | 0.579778 | 0.408462 | 0 |
| `wall_distance_fluid_2_4dx` | 0.00116951 | 0.00194417 | 0.00258747 | 0.259605 | 0.601549 | 0.451992 | 0.000301151 |
| `wall_distance_solid_2_4dx` | 0.00128716 | 0.00215166 | 0.00297401 | unmeasured | 0.598214 | 0.432801 | 0 |
| `wall_distance_fluid_4_8dx` | 0.00129256 | 0.00206275 | 0.00242678 | 0.195623 | 0.626619 | 0.532624 | 0.000313522 |
| `wall_distance_solid_4_8dx` | 0.000797202 | 0.00129062 | 0.00168248 | unmeasured | 0.617689 | 0.473826 | 8.84931e-08 |
| `wall_distance_fluid_8_infdx` | 0.00322268 | 0.00462287 | 0.00422447 | 0.215327 | 0.697116 | 0.762858 | 0.000101423 |
| `wall_distance_solid_8_infdx` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `contact_line_left_within_2dx` | 0.000708131 | 0.00133092 | 0.00208483 | 0.191479 | 0.532062 | 0.339658 | 0.015287 |
| `contact_line_left_within_4dx` | 0.00131257 | 0.00240829 | 0.00344583 | 0.215393 | 0.54502 | 0.380914 | 0.0192427 |
| `contact_line_right_within_2dx` | 0.000708131 | 0.00133092 | 0.00208483 | 0.191479 | 0.532062 | 0.339658 | 0.015287 |
| `contact_line_right_within_4dx` | 0.00131257 | 0.00240829 | 0.00344583 | 0.215393 | 0.54502 | 0.380914 | 0.0192427 |
| `contact_line_union_within_2dx` | 0.00100145 | 0.0018822 | 0.0029484 | 0.191479 | 0.532062 | 0.339658 | 0.0216191 |
| `outside_contact_line_union_2dx` | 0.0044485 | 0.00688965 | 0.00799697 | 0.142845 | 0.645679 | 0.556274 | 0.0183676 |
| `contact_line_union_within_4dx` | 0.00185625 | 0.00340584 | 0.00487315 | 0.215393 | 0.54502 | 0.380914 | 0.0272133 |
| `outside_contact_line_union_4dx` | 0.0041649 | 0.00627776 | 0.00699264 | 0.137095 | 0.663438 | 0.595612 | 0.00801202 |

### Global cross-case effect sizes

| Metric | 60° | 90° | 150° | 60−90 | 60/90 | 60−150 | 60/150 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `source_acceleration_l2` | 0.0315784 | 0.0459577 | 0.0537681 | -0.0143793 | 0.687118 | -0.0221897 | 0.587307 |
| `projectable_gradient_l2` | 0.0312474 | 0.0453994 | 0.0530882 | -0.0141519 | 0.688279 | -0.0218408 | 0.588594 |
| `nonprojectable_residual_l2` | 0.00455983 | 0.00714213 | 0.00852318 | -0.00258229 | 0.638442 | -0.00396335 | 0.534992 |
| `projectable_fraction_of_source_l2` | 0.98952 | 0.987851 | 0.987356 | 0.0016692 | 1.00169 | 0.00216366 | 1.00219 |
| `residual_fraction_of_source_l2` | 0.144397 | 0.155407 | 0.158518 | -0.0110091 | 0.929159 | -0.0141201 | 0.910924 |
| `residual_over_projectable_l2` | 0.145927 | 0.157318 | 0.160547 | -0.0113911 | 0.927592 | -0.0146207 | 0.908932 |
| `residual_divergence_l2` | 3.31543e-07 | 5.25491e-07 | 6.7622e-07 | -1.93948e-07 | 0.630921 | -3.44677e-07 | 0.490289 |
| `poisson_solve_residual_l2` | 3.31684e-07 | 5.25239e-07 | 6.75538e-07 | -1.93554e-07 | 0.631492 | -3.43854e-07 | 0.490992 |

## Mechanism matrix

| Candidate | Status | Scope / evidence note |
|---|---|---|
| `CAPILLARY_PRESSURE_IMBALANCE` | **SUPPORTED** | support establishes the force-subspace property only; it does not establish that this component caused the 60-degree contact-angle nonstationarity |
| `PRESSURE_PROJECTION_LIMITED` | **FALSIFIED** | the nonzero solenoidal residual is outside the projectable subspace by definition; it is not a Poisson convergence failure |
| `CAPILLARY_DENSITY_SCALING` | **FALSIFIED** | does not clear the separate model-choice caveat about density-dependent inertia or projection; local-density scaling was not run |
| `VARIABLE_DENSITY_COUPLING` | **NOT_TESTED** | the one-step equal-density blend A/B is identical under the current gravity-off velocity formulation; no variable-density projection or inertia prototype was justified or run |
| `DISCRETE_PRODUCT_RULE_IN_CAPILLARY_FORCE` | **SUPPORTED** | this is an exact algebraic decomposition, not evidence that Form B is a better model; no force replacement or variant trajectory was used |
| `WALL_ENERGY_MOMENTUM_CONSISTENCY` | **NOT_TESTED** | wall-energy-derived force is measured and localized; no wall-energy, momentum, or contact-angle formulation was modified |
| `BRINKMAN_WALL_COUPLING` | **SUPPORTED** | isolated frozen-phi A/B changes the short momentum response; not evidence that Brinkman coupling caused 60-degree nonstationarity |
| `CONTACT_LINE_PINNING` | **NOT_TESTED** | 2dx and 4dx neighborhoods/exclusions are measured; no force-threshold test was run |
| `Y_PERIODIC_TOPOLOGY_COUPLING` | **NOT_TESTED** | seam overlap is localized only; no boundary-topology ablation was run |
| `TIME_SPLITTING_OR_DT_SENSITIVITY` | **NOT_TESTED** | dt/2 was not run |
| `MOMENTUM_VISCOSITY_DISCRETIZATION` | **NOT_TESTED** | production nu(phi) and Laplacian remain unchanged |
| `PHASE_KINETICS_LIMITED` | **SUSPECTED** | no phase-mobility or phase-equation ablation was run |
| `MULTIPLE_CONTRIBUTORS` | **NOT_TESTED** | mechanism closure is limited to the measured production-force subspace property |

- **Final root cause of the measured capillary/pressure subspace property:** `CAPILLARY_PRESSURE_IMBALANCE`.
- **Causal root cause of the L1A-2j 60° nonstationarity:** `INCONCLUSIVE`.
- Scope: the measured structural non-projectable capillary-acceleration component only; not a causal explanation of the 60-degree angle-cycle/production-gate failure.

## Evidence-triggered Brinkman damping A/B (diagnostic only)

The residual-energy overlap in the fixed `chi>0.50` mask (25.0%, 29.5%, 39.9% for 60°, 90°, 150°) triggered this short diagnostic. Each branch starts from the exact matched snapshot, holds phi fixed, and advances 100 steps. Only `solid.chi` in the Brinkman damping term changes. Geometry, wall energy, wetting, dt, M, capillary force, density, viscosity, and projection are otherwise unchanged. This is not a production trajectory, retuning, or validation.

| Case | Snapshot step | Steps | chi>0.50 residual-energy fraction | Velocity delta L2 (off−on) | delta/on L2 | Off/on velocity L2 | delta L2 inside chi>0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `authority_060` | 50000 | 100 | 0.2502 | 0.00092062935 | 2.7523101 | 3.6152093 | 0.00036396906 |
| `control_090` | 27200 | 100 | 0.295464 | 0.0013321799 | 3.1167389 | 3.9348835 | 0.00056944731 |
| `control_150` | 100000 | 100 | 0.39886 | 0.0015421714 | 4.6286485 | 5.3772815 | 0.00080849383 |

- The chemical potential and applied capillary force were bitwise equal between branches; phi remained bitwise fixed. The `chi=0` branch has zero Brinkman term by construction.
- Full per-case norms, regional differences, projection observations, strict snapshot provenance, field arrays, and supplemental source/function hashes are in `capillary_pressure_balance_report.json`, `mechanism_matrix.json`, `operator_map.json`, `artifacts/l1a2k/diagnostics/brinkman_on_off_ab.json`, and `artifacts/l1a2k/fields/`.
- `BRINKMAN_WALL_COUPLING` is `SUPPORTED` only for this short frozen-phi momentum response. Causal attribution of the 60° angle nonstationarity remains `INCONCLUSIVE`; no Brinkman retuning or production change is recommended by this audit.

## CHNS-50k versus converged CH-only 60° phi

- CHNS state: step `50000`; converged CH-only endpoint: step `180000`.
- `phi_CHNS − phi_CH-only`: L2(dxdy) `0.0747239567`, L∞ `0.139078849`, relative L2 vs CHNS `0.05872628463795332`.
- Formal `sum_i(V_i*phi_i)` mass: CHNS `1.92337030287`, CH-only `1.92337030287`, absolute difference `7.54952e-15` (comparison only, not an acceptance gate).
- Full-grid `dx*dy*sum(phi)` is retained as a diagnostic only and is not used as formal mass or an acceptance gate.
- Convergence gate: `{'converged': True, 'angle_ok': True, 'energy_ok': True, 'rate_ok': True, 'speed_ok': True, 'angle_window_spread_deg': 0.09537505932090085, 'window_samples_used': 7, 'window_mobility_time': 0.05, 'energy_window_max_rel_change': 3.8545370993903205e-06, 'energy_stationary_strict': False}`.

## Static Laplace invariant

- Existing `production.capillary_audit` checks rerun without changing any formulation: all passed = `True`.
- `laplace_jump_sign_and_scale`: passed `True`, observed `{'laplace_ratio': 0.9850772536075545, 'phi_mean_inside_probe': 0.9999174751504804, 'phi_mean_outside_probe': 5.052422088479835e-10}`.

## Unmeasured sections and scope limits

- `local_density_scaling_ab`: **unmeasured_not_triggered_by_constant_rho_l_or_equal_density_one_step_ab**
- `variable_density_projection_or_inertia_prototype`: **unmeasured_not_justified_by_equal_density_ab**
- `brinkman_on_off_ab`: **measured_100_step_diagnostic_ab_not_production_validation**
- `dt_half_matched_physical_time`: **unmeasured_frozen_state_diagnostics_not_insufficient**
- `contact_line_unpinning_threshold`: **unmeasured_no_threshold_test_requested_or_triggered**
- `production_repair_or_validation`: **not_in_scope**

## Artifacts and source provenance

- Field artifacts: `artifacts/l1a2k/fields/` (diagnostic only).
- Machine-readable report: `capillary_pressure_balance_report.json`.
- Operator map: `operator_map.json`; source-function and whole-file SHA256 values are recorded there and in the manifest.
- Production contract 11, capillary denominator, pressure projection, wall energy, phase storage, viscosity, Brinkman defaults, and acceptance thresholds remain unchanged.
- No diagnostic variant is production validation; no authority continuation beyond step 50,000 was performed.
