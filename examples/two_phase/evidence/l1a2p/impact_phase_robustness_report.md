# L1A-2p -- impact-window phase robustness and production time-step policy closure

- profile: **forensic** (forensic profile: the full production-spec evidence (section 32))
- git sha: `0a0e7dde4dcc70fb221aad9e64ab59208567803e`
- solver contract version: **12**
- default timestep policy: `impact_phase_cap_dx2_v1`

## Frozen L1A-2o baseline (unchanged inputs)

The six flat impact canaries were rejected at the requested dt=0.004 (overshoot 0.0596-0.0934 against the 0.02 gate); the complex held-out canary diverged (non-finite phi at t~1.33); N-DT was TARGET_CRITICAL. Thresholds here are unrelaxed (section 1).

## Failure mechanism (sections 3-5)

- first public-step crossing of the 0.02 overshoot gate: step 54
- bitwise replay before decomposition: `{'bitwise_identical': True, 'per_field': {'phi': True, 'u': True, 'v': True}, 'reference': 'phasefield.step (production, unmodified)'}`
- causal classification: **ADVECTIVE_PHASE_CFL**

| intervention | overshoot after one public step |
|---|---|
| `prefailure_overshoot` | 0.0194 |
| `A_phase_advection_only_implicit_filtered` | 0.0327 |
| `A2_raw_advective_no_implicit_filter_control` | 0.0429 |
| `B_ch_only_u_v_zero_production_operator` | 0.0063 |
| `C_full_phase_update_frozen_velocity` | 0.0223 |
| `D_fully_coupled_production_step` | 0.0222 |

## Stability indicators at the pre-failure state (sections 6-8)

- global advective CFL (public dt): 0.025345258712768555
- cut-cell advective CFL ratio: 0.1373538225889206 (dt_adv_min 0.029121870175004005 vs public dt) -> subcycling not indicated
- dt/stable_dt: 0.64
- empirical dt/dx^2 indicator: 4.096 (labelled empirical)

## Resolution scaling and dt sweep (sections 9-10)

| dt (N192) | window peak overshoot | passes 0.02 |
|---|---|---|
| 0.004 | 0.0598 | FAIL |
| 0.003 | 0.0080 | PASS |
| 0.002 | 0.0042 | PASS |

- alpha family selected: **2** (EMPIRICAL (physically supported family; not a derived stability law))

## Complex-surface divergence (section 11)

- classification: **SUPPORTED** (COMPLEX_DIVERGENCE_SHARED_DT_CAUSE)

## Candidate policies (section 12)

- selected: **impact_phase_cap_dx2_v1**

| candidate | effective dt @N192 | limiting criterion |
|---|---|---|
| `legacy_requested_v0` | 0.004 | requested_dt |
| `fixed_cap_002_v1` | 0.002 | fixed_cap_dt |
| `impact_phase_cap_dx2_v1` | 0.002 | impact_phase_dx2_cap |
| `cfl_multicriterion_v1` | 0.002 | impact_phase_dx2_cap |

## Mandatory canary matrix under the candidate policy (sections 16/20)

- accepted: **8/8** (all accepted: `True`)

| canary | accepted | overshoot | policy dt |
|---|---|---|---|
| flat_we100_ct050 | True | 0.0115 | 0.002 |
| flat_we100_ct000 | True | 0.0067 | 0.002 |
| flat_we100_ctm050 | True | 0.0072 | 0.002 |
| flat_we200_ct050 | True | 0.0121 | 0.002 |
| flat_we200_ct000 | True | 0.0068 | 0.002 |
| flat_we200_ctm050 | True | 0.0072 | 0.002 |
| pillar_training | True | 0.0077 | 0.002 |
| complex_heldout | True | 0.0081 | 0.002 |

## Verdicts (sections 27-30)

- repair label: **DT_POLICY_PARTIALLY_SUFFICIENT**
- exit verdict: **L1B_DATA_NOT_READY**
- single remaining target-critical blocker: TEMPORAL_REFINEMENT

## Contract promotion (sections 23-24)

- promotion: **ALREADY_APPLIED**
- edits applied: 0
- production semantics changed: True

## Provenance

- thresholds: every generator gate runs unmodified (overshoot 0.02, leak 5e-4, mass 0.995/1.005, speed 5.0, feature cells 2); no phi clipping, bounded projection, mass redistribution or threshold relaxation enters production (section 15)
- the audit trajectories are diagnostic evidence, never the acceptance authority (section 16/20)
- empirical indicators are labelled empirical; no manufactured stability formula (section 7)
