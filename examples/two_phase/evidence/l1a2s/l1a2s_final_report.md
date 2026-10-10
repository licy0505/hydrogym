# L1A-2s — divergence-free, solid-compatible impact initialization re-audit

Forensic diagnostic report · stage `l1a2s_v1` · base/main `002120f3a051e638a7e85f9db4022107dbe45780` · generated from the refreshed contour-gap event metric.

## A. Scope and authority

This stage measures C0/C1/C2 initial compatibility, first-substep causality, local-support impact authentication on the four frozen L1A-2r canaries, temporal/spatial sensitivity, and T=8 data-readiness. It is diagnostic only: **no production physics/default change, no candidate promotion, no contract-13 promotion, no official dataset, and no bulk L1B generation**.

## B. Base, prerequisite, and branch

PR #21 is merged at `002120f3a051e638a7e85f9db4022107dbe45780`; the preflight confirms current `main` equals that merge. Work remained on the Arena-bound branch `arena/3d47375a-hydrogym`. No branch switch or stacked unmerged prerequisite was used.

## C. Production freeze

`SOLVER_CONTRACT_VERSION=12`; `impact_phase_cap_dx2_v1`; production uniform initializer unchanged. `phasefield.step*`, `rhs`, Poisson solve, solid construction, CH/Young wall, geometry, timestep policy, dtypes/defaults, and all acceptance thresholds remain unmodified. Only local diagnostic candidate states are built. The reused projection audit adds measurement/capture fields only; it does not change solver arithmetic.

## D. Frozen canaries and case-specific inputs

All four exact canaries were built from their current case/SDF/placement and physics; no drop or geometry was moved to manufacture contact. N=192, domain 6×6, R=0.7, effective `dt=0.002`, requested `dt=0.004` for the primary full run, `M=0.002`, `eta_pen=0.004`, `eps/dx=1.5`:

| Canary | Actual geometry / seed / split | Nominal We / Re / cosθ | Actual x₀, y₀, surface top, initial gap | ε / ε÷dx | C2 initial D Linf |
|---|---|---:|---:|---:|---:|
| `flat_we100_ct050` | flat surface / seed 6 / `train` | We=100, Re=200, cosθ=0.5 | x₀=3, y₀=1.02812, top=0.234375, gap₀=0.09375 | 0.046875 / 1.5 | 2.921e-06 |
| `flat_we200_ct000` | flat surface / seed 5 / `train` | We=200, Re=200, cosθ=0 | x₀=3, y₀=1.02812, top=0.234375, gap₀=0.09375 | 0.046875 / 1.5 | 2.921e-06 |
| `pillar_training` | 4 pillars, width 0.3, height 0.4 / seed 8 / `train` | We=100, Re=200, cosθ=-0.5 | x₀=3, y₀=1.43437, top=0.640625, gap₀=0.09375 | 0.046875 / 1.5 | 2.861e-06 |
| `complex_heldout` | 7 random pillars, width range [0.2,0.4] / seed 100 / `test` | We=100, Re=200, cosθ=-0.5 | x₀=3, y₀=1.71562, top=0.921875, gap₀=0.09375 | 0.046875 / 1.5 | 3.338e-06 |

## E. Candidate family and early-stop gate

C0=`UNIFORM_ALL_DOMAIN` is the production-style negative control; C1=`STREAMFUNCTION_LOCALIZED_V0` has low central-D divergence but fails the declared wall/FV tests and is **not solid-compatible**; C2=`SDF_TAPERED_STREAMFUNCTION_V1` is an isolated diagnostic streamfunction with the actual SDF taper, periodic seam taper, and case-specific amplitude. C2 passed its predeclared N=192 divergence/wall gate on all four canaries while retaining liquid-core mean `v≈−0.5`; therefore the specified early stop did not trigger. C3 was **NOT_RUN**: C2 did not fail a declared initial compatibility property, so an optional constrained solve was not justified.

## F. Discrete incompressibility

The measured periodic central-D divergence is kept distinct from the open-face finite-volume reconstruction. C2 values and their dimensionless scalings are:

| Canary | central-D Linf | open-face FV Linf | dx·D Linf/uimpact | dx·FV Linf/uimpact | deep-solid speed | near-wall χ speed | wall-normal speed | Initial gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `flat_we100_ct050` | 2.921e-06 | 9.179e-06 | 1.825e-07 | 5.737e-07 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| `flat_we200_ct000` | 2.921e-06 | 9.179e-06 | 1.825e-07 | 5.737e-07 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS |
| `pillar_training` | 2.861e-06 | 5.722e-06 | 1.788e-07 | 3.576e-07 | 0.000e+00 | 7.807e-06 | 0.000e+00 | PASS |
| `complex_heldout` | 3.338e-06 | 8.821e-06 | 2.086e-07 | 5.513e-07 | 0.000e+00 | 5.295e-04 | 0.000e+00 | PASS |

Predeclared bounds: central D `≤ 7.629e-06` (`64·float32 eps`); open-face FV `≤0.01` dimensionless. The values are actual discrete measurements, not continuum claims.

## G. Cut-cell/open-face flux and embedded-wall compatibility

The FV/cut-cell diagnostic is **MEASURED_OPEN_FACE_RECONSTRUCTION**: arithmetic adjacent-cell face velocities, shared contract-9 open apertures `A_f`, and division by `V_i`; this is not mislabeled as the central-D operator. Embedded-wall flux is separately measured by bilinear cell-velocity samples at wall centroids weighted by `wall_measure`. C2 has zero measured deep-solid velocity and zero measured embedded-wall normal velocity on all four cases (declared normalized bounds: deep-solid `≤3.815e-6`, near-wall χ and wall-normal `≤0.02`). Its largest near-wall `chi≥0.01` speed is 5.295e-04 (ratio to `u_impact` 1.059e-03).

Controls remain distinct: C0 has central D Linf `0` but uniform `0.5` velocity in deep solid and maximum open-face FV Linf `1.140e+04`. C1 has central-D Linf `≈3.10e-6` but maximum deep-solid speed `1.496`, near-wall speed `1.898`, and embedded-wall normal speed `1.326`. Neither is called a solid-compatible physical seed.

## H. Initial region budget and motion

C2's V-weighted core target is approximately `−0.5` in all cases; liquid means span `-0.3176` to `-0.3105`. Far-field gas is measured separately from local return flow. The return-flow peak and full region budgets are in `initial_velocity_region_budget.json`; no gas/solid mask is silently applied after streamfunction differentiation.

## I. Causal first-substep ledger

The JIT ledger matches `pf.step` within declared dtype tolerances; bitwise RHS recomposition is verified. For flat C2 on the first `h=dt/3` substep, liquid-weighted Δv is RHS `-9.431e-05`, Brinkman `9.885e-08`, and periodic projection `-3.648e-06`. Their full-grid-mean changes are `-5.192e-07`, `9.700e-09`, and `-1.563e-11` respectively. The pressure-gradient correction has Linf `5.072e-03` but mean `-5.826e-14`; it is not conflated with Brinkman damping. C0’s full-grid mean changes from `−0.5` to `−0.496981` at Brinkman, while its periodic projection correction mean remains approximately zero.

## J. Contact measurement and focused tests

The diagnostic gap now samples bilinear SDF values on the **linearly reconstructed periodic `phi=0.5` contour**, not a nearest thresholded-liquid cell. Periodic nearest-wall lookup uses actual wall-measure centroids; approach velocity is bilinearly sampled at that contour support and projected onto the local normal into solid. Contact remains gap `≤1.5 dx`; the inherited meaningful-approach floor remains `0.2·u_impact=0.1`. Two synthetic tests guard subcell contour motion and bilinear velocity sampling; their translated contour is test-only and is not a canary/evidence trajectory.

## K. T=8 four-canary event authentication

Full runs use physical `T=8`, N=192, and exact saved times every `0.08` (101 frames; no time interpolation). Event and postcontact rows are sampled every public step; the same screen run is continued to T=8. All four cases now have decreasing contour-gap samples and verified postcontact response. Only the held-out complex case clears the inherited local-normal approach floor:

| Canary | Event verdict | Interpolated contact time | Precontact local normal approach / floor | Several decreasing gaps | Postcontact dynamics |
|---|---|---:|---:|---|---|
| `flat_we100_ct050` | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.208645 | 0.04492 / 0.10 | True | True |
| `flat_we200_ct000` | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.286762 | 0.04549 / 0.10 | True | True |
| `pillar_training` | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.353642 | 0.06940 / 0.10 | True | True |
| `complex_heldout` | IMPACT_AUTHENTICATED | 0.842387 | 0.16234 / 0.10 | True | True |

`complex_heldout` authenticates at `t=0.842387` with approach `0.16234` and observed postcontact window `[0.844,1.084]`. The three other events are `CONTACT_WITHOUT_MEANINGFUL_IMPACT`, not contact-detection failures: their gaps decrease and postcontact dynamics exist, but their local-normal approach is below 0.1. Fleet verdict is **`PARTIAL_IMPACT_AUTHENTICATION`**, not all-canary authentication.

## L. Temporal dt sensitivity

N=192 comparisons use dt `0.002/0.001/0.0005` at equal physical times `[0,0.08,0.16,0.24]`, with no interpolation. Event screen results:

| Canary | dt | Verdict | Contact time (if observed) | Precontact approach (if observed) |
|---|---:|---|---:|---:|
| `flat_we100_ct050` | 0.002 | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.208645 | 0.04492 |
| `flat_we100_ct050` | 0.001 | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.230024 | 0.02356 |
| `flat_we100_ct050` | 0.0005 | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.252528 | 0.01177 |
| `flat_we200_ct000` | 0.002 | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.286762 | 0.04549 |
| `flat_we200_ct000` | 0.001 | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.339843 | 0.02315 |
| `flat_we200_ct000` | 0.0005 | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.409800 | 0.01099 |
| `pillar_training` | 0.002 | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.353642 | 0.06940 |
| `pillar_training` | 0.001 | CONTACT_WITHOUT_MEANINGFUL_IMPACT | 0.426889 | 0.03815 |
| `pillar_training` | 0.0005 | NO_CONTACT_WITHIN_OBSERVED_WINDOW | — | — |

The primary flat events are unauthenticated at all dt levels, and pillar has no contact in the 0.48 screen at dt=0.0005. Accordingly `TEMPORAL_REFINEMENT=UNMEASURED` for authenticated impact observables; exact-time retention/equal-time fields are still measured in `temporal_retention_matrix.json`. The historical 3% target is not repurposed as an initialization or contact-speed threshold.

## M. Spatial comparison and ε/dx scaling

The bounded N=144/N=192 diagnostic compares the fixed-ε/dx production family and a fixed-physical-ε family through `T=0.48` at 0.08 cadence. The common-grid field L2 values below are descriptive, **not** a redefinition of the 3% key-observable gate:

| Family | N=144 ε / ε÷dx | N=192 ε / ε÷dx | initial gap N144 / N192 | max relative L2 φ/u/v over saved times |
|---|---:|---:|---:|---:|
| fixed ε/dx production family | 0.0625 / 1.500 | 0.046875 / 1.500 | 0.125 / 0.09375 | 2.846% / 29.648% / 17.544% |
| fixed physical ε diagnostic family | 0.046875 / 1.125 | 0.046875 / 1.500 | 0.0989583 / 0.09375 | 2.715% / 31.166% / 11.147% |

The fixed-ε/dx production family also changes physical interface thickness and case-derived clearance; the fixed-physical-ε diagnostic holds physical ε and common `y0` but changes ε/dx with resolution. No family promotes the historical formal result: `SPATIAL_REFINEMENT=FAIL` remains unchanged because this candidate audit is descriptive and does not pass the original formal exit tests.

## N. Generator quality and T=8 data-readiness

Canonical read-only generator diagnostics were run on diagnostic-only trajectories; no official/training data were written. Historical thresholds were re-read unchanged:

| Canary | max φ overshoot / 0.02 | solid leak / 5e-4 | total-mass ratio min–max / 0.995–1.005 | max speed / 5 | Gate |
|---|---:|---:|---:|---:|---|
| `flat_we100_ct050` | 0.00715 / 0.02000 | 0.000e+00 / 5.0e-4 | 1.000000–1.000000 / 0.995–1.005 | 2.1299 / 5.0 | PASS |
| `flat_we200_ct000` | 0.00668 / 0.02000 | 0.000e+00 / 5.0e-4 | 1.000000–1.000000 / 0.995–1.005 | 2.1298 / 5.0 | PASS |
| `pillar_training` | 0.03697 / 0.02000 | 0.000e+00 / 5.0e-4 | 1.000173–1.005185 / 0.995–1.005 | 2.0865 / 5.0 | FAIL |
| `complex_heldout` | 0.00752 / 0.02000 | 0.000e+00 / 5.0e-4 | 1.000001–1.001714 / 0.995–1.005 | 2.0835 / 5.0 | PASS |

`pillar_training` fails overshoot (`0.03697`) and maximum total-mass ratio (`1.005185`); therefore `GENERATOR_QUALITY_GATES=FAIL`. Every artifact is marked `diagnostic_only=true` and `production_lineage_eligible=false`.

## O. Direct-field labels and export representation

Direct solver-grid/export errors are finite and measured; export dtypes/shapes match the current generator (`phi=float32`, `u/v=float16`, ds=3). No approved universal direct-field tolerance exists, so `DIRECT_FIELD_LABELS=UNMEASURED`; morphology or a single authenticated canary does not waive this. See `direct_field_quality_matrix.json` for solver-grid and quantized export errors at exact frames.

## P. Nominal We/Re and contact-speed semantics

| Canary | nominal input We / Re | generator auxiliary kinematic We / Re | `u_impact` argument | measured local `U_contact` | `U_contact/u_impact` |
|---|---:|---:|---:|---:|---:|
| `flat_we100_ct050` | 100 / 200 | 25 / 100 | 0.5 | 0.04338 | 0.087 |
| `flat_we200_ct000` | 200 / 200 | 50 / 100 | 0.5 | 0.04506 | 0.090 |
| `pillar_training` | 100 / 200 | 25 / 100 | 0.5 | 0.06912 | 0.138 |
| `complex_heldout` | 100 / 200 | 25 / 100 | 0.5 | 0.16173 | 0.323 |

`impact_parameter_semantics.json` remains **`UNRESOLVED`**: `PhaseFieldParams` documents unit reference U and `D=1`, while the generator separately records `We·u_impact²` and `Re·|u_impact|`; measured contact speed is local normal speed, not a relabeling of nominal parameters. No PDE parameter or sample label is changed, and the 3% gate is not applied to `U_contact/u_impact`.

## Q. Historical blockers and formal-exit discipline

Unchanged: `L1A_STATUS=BLOCKED`, `L1B_DATA=L1B_DATA_NOT_READY`, `N_DT=TARGET_CRITICAL`, official `SPATIAL_REFINEMENT=FAIL`, `W_CONTACT_ANGLE=OPEN`, `P_VARDENS_PROJ=OPEN`, `D_FRESH_TRAIN_CONTRACT=OPEN`. This diagnostic does not satisfy or waive any original formal exit test.

## R. Readiness matrix

| Gate | Exact status |
|---|---|
| `TEMPORAL_REFINEMENT` | `UNMEASURED` |
| `SPATIAL_REFINEMENT` | `FAIL` |
| `DIRECT_FIELD_LABELS` | `UNMEASURED` |
| `GENERATOR_QUALITY_GATES` | `FAIL` |
| initializer | `INITIALIZER_COMPATIBLE` |
| fleet | `PARTIAL_IMPACT_AUTHENTICATION` |
| stage decision | `B_INITIALIZATION_HELPFUL_BUT_REFINEMENT_OR_GEOMETRY_BLOCKS` |

## S. Primary blocker and exactly one next action

Primary blocker enum remains `REQUIRED_PILLAR_OR_COMPLEX_IMPACT_AUTHENTICATION`. After this rerun, `complex_heldout` passes; the pillar branch of that blocker is the unresolved named case (`pillar_training`: `0.06940 < 0.1`). The three unauthenticated cases remain explicit in the event matrix. Exactly one next action: **diagnose the below-floor local-normal approach on pillar_training under the frozen geometry, drop placement, and inherited 0.2*u_impact threshold; do not tune the threshold or promote data**.

## T. Diagnostic lineage and anti-promotion controls

The T=8 NPZs live only under ignored `examples/two_phase/artifacts/l1a2s/`; committed evidence is under `evidence/l1a2s/`. Fingerprints bind contract 12, exact geometry/initial state, policy and source hashes; diagnostic artifacts fail closed for production lineage. Candidate promotion, official data release, contract-13 promotion, bulk dataset generation, and model training remain false.

## U. Local validation

- Focused L1A-2s test module: **49 passed**.
- Correct-root non-slow two-phase suite: **223 passed, 5 failed**. The same five frozen-source/merge-fixture failures previously observed remain a merge blocker: `test_no_production_phi_semantics_change`, `test_w_contact_angle_remains_open`, `test_phasefield_source_unchanged`, `test_no_production_phase_rate_change`, and `test_no_cutcell_geometry_change`. None are concealed as passes.
- `ruff check`: **PASS** for all changed Python. `ruff format --check`: **PASS** for the new audit and tests. The reused projection-audit file already fails whole-file formatting at HEAD; no unrelated whole-file reformat was added.
- `py_compile`: **PASS** for all three changed Python files. `git diff --check`: **PASS**.

## V. Hosted CI and merge status

Hosted CI is **`PENDING_PR`** at this report's generation point. The PR must remain unmerged. Any hosted failure other than a verified identical pre-existing locked-dependency audit is a merge blocker; local frozen-source/merge-fixture failures are also unresolved blockers.

## W. Reproduction and evidence map

```bash
cd examples/two_phase
JAX_ENABLE_X64=1 PYTHONPATH=. ../../.venv/bin/python -u production/impact_initialization_compatibility_audit.py --profile forensic --stages all
JAX_ENABLE_X64=1 PYTHONPATH=. ../../.venv/bin/pytest -q tests/test_impact_initialization_compatibility_audit.py
JAX_ENABLE_X64=1 PYTHONPATH=. ../../.venv/bin/pytest -q tests -m 'not slow'
```

Evidence JSONs: `initializer_constraint_matrix.json`, `initial_velocity_region_budget.json`, `projection_brinkman_startup_ledger.json`, `control_matrix.json`, `temporal_retention_matrix.json`, `impact_event_matrix.json`, `bounded_full_horizon_matrix.json`, `impact_parameter_semantics.json`, `spatial_temporal_reaudit.json`, `direct_field_quality_matrix.json`, `gate_and_blocker_matrix.json`, and `manifest.json`. Forensic run elapsed `2589.6s`; the T=8 matrix alone took `2187.3s`.

## X. Exact status lines

```text
STAGE=L1A-2s
BASE_MAIN_SHA=002120f3a051e638a7e85f9db4022107dbe45780
BRANCH=arena/3d47375a-hydrogym
SOLVER_CONTRACT_VERSION=12
PRODUCTION_TIMESTEP_POLICY=impact_phase_cap_dx2_v1
INITIALIZER_VERDICT=INITIALIZER_COMPATIBLE
FLEET_IMPACT_VERDICT=PARTIAL_IMPACT_AUTHENTICATION
STAGE_DECISION=B_INITIALIZATION_HELPFUL_BUT_REFINEMENT_OR_GEOMETRY_BLOCKS
TEMPORAL_REFINEMENT=UNMEASURED
SPATIAL_REFINEMENT=FAIL
DIRECT_FIELD_LABELS=UNMEASURED
GENERATOR_QUALITY_GATES=FAIL
IMPACT_PARAMETER_SEMANTICS=UNRESOLVED
L1A_STATUS=BLOCKED
L1B_DATA=L1B_DATA_NOT_READY
N_DT=TARGET_CRITICAL
W_CONTACT_ANGLE=OPEN
P_VARDENS_PROJ=OPEN
D_FRESH_TRAIN_CONTRACT=OPEN
PRIMARY_BLOCKER=REQUIRED_PILLAR_OR_COMPLEX_IMPACT_AUTHENTICATION
NEXT_ACTION=diagnose the below-floor local-normal approach on pillar_training under the frozen geometry, drop placement, and inherited 0.2*u_impact threshold; do not tune the threshold or promote data
CANDIDATE_PROMOTED=false
CONTRACT13_PROMOTED=false
OFFICIAL_DATASET_WRITTEN=false
BULK_TRAINING_DATA_GENERATED=false
HOSTED_CI=PENDING_PR
MERGE=DO_NOT_MERGE
```
