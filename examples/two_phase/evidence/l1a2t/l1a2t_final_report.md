# L1A-2t — Near-Wall Taper, Interface Transport and Contact-Side Impact Causal Audit

Profile: `forensic` | evidence class: `FORENSIC_N192` | diagnostic only: no promotion, no production change, no contract change.

## A. Prerequisite, PR and CI status
- verified merged main SHA: `09e2993b8da478d79e768054194201af57121784`; verified parents ['039f6a0d9a14ff601c76a58f5a66ce0900677de5', '865317d438eee5c8d5f1e942f0ad30c943010818']
- PR #22 `MERGED` (head `865317d438eee5c8d5f1e942f0ad30c943010818`, merged 2026-10-11T07:37:01Z); hosted Python CI run 38120327651: observed_pass_8_of_8_jobs_at_preflight_time (engineering-only, reuse of the frozen freeze)
- branch `arena/02b6c71b-hydrogym`, merge-base `09e2993b8da478d79e768054194201af57121784`, worktree clean outside stage deliverables: True
- stage commits on top of the verified main: ['2dd0f16 L1A-2t: harden preflight provenance guards (branch ancestor, porcelain parsing)', 'c52f049 L1A-2t: near-wall taper, interface transport and contact-side impact audit (diagnostic)']
- L1A-2t hosted CI is not claimed here; no CI result is asserted in advance

## B. Frozen source, policy and prohibitions
- solver contract 12; timestep policy `impact_phase_cap_dx2_v1`
- frozen source hashes match the frozen L1A-2s manifest: {'phasefield': True, 'cases': True, 'generate_dataset': True, 'timestep_policy': True}
- untouched: phasefield solver, production initializer default, thresholds, historical evidence, mass baselines, approved SHA migrations, `quality/legacy_two_phase_debt_v1.json`
- no T=8 recompute, no L1B generation, no ML training, no contract 13, no candidate promotion

## C. Inherited L1A-2s measurements and replication
- inherited frozen table (dt refinement, flat/pillar): approach 0.04492/0.02356/0.01177 (flat100), 0.04549/0.02315/0.01099 (flat200), 0.06940/0.03815/none (pillar)
- this stage re-measures them from the reproduced C2 state instead of copying them; the C2 t=0 fields
  are hash-compared with the frozen L1A-2s evidence and any mismatch raises a fail-closed error
- C2 t=0 identity reproduction: {"complex_heldout": {"amplitude_scale_l1a2s_builder": 2.1183961728581067, "amplitude_scale_relative_delta": 0.0, "amplitude_scale_taper_family": 2.1183961728581067, "case_name": "complex_heldout", "comparison_rule": "sha256 of the exact solver arrays plus bitwise array equality of the two builders", "hash_checks": {"geometry": true, "phi": true, "u": true, "v": true}, "hashes": {"geometry": "a7fec

## D. Taper mechanism hypothesis
- H-TAPER: the C2 wall taper onset (a=4 dx) sits above the initial contour support (sdf~2.5 dx), so the
  incident front fluid velocity there is exactly zero; opening the taper should restore it.
- competing explanations measured separately: H-CH, H-BP, H-GEOMETRY (diffuse-indicator gate domain), H-MULTI
- quarantined taper-free control: front approach 0.324973 vs 0 with the taper present

## E. Prespecified candidate equations
- `C2_BASE` a=4 b=8 (role: exact L1A-2s C2 reproduction)
- `T1_NEAR_NARROW` a=1 b=4 (role: shortened near-wall taper)
- `T2_NEAR_WIDE` a=1 b=8 (role: onset-only comparison vs C2)
- `T3_FAR_NARROW` a=4 b=4 (role: width-only comparison vs C2)
- quarantined `NC_NO_WALL_TAPER`: quarantined diagnostic counterfactual; not a candidate, never eligible, never promoted
- identical radial envelope, periodic seam taper, differentiation, x0/y0/R, core-mean normalization

## F. All-candidate t=0 compatibility (frozen gates)

| case | candidate | gate | failing gate | near-wall chi ratio (limit 0.02) | front approach | front floor 0.1 |
|---|---|---|---|---:|---:|---|
| `flat_we100_ct050` | `C2_BASE` | PASS | - | 0 | 0 | False |
| `flat_we100_ct050` | `T1_NEAR_NARROW` | FAIL | nearwall_chi_velocity | 1.776 | 0.09050177 | False |
| `flat_we100_ct050` | `T2_NEAR_WIDE` | FAIL | nearwall_chi_velocity | 0.4465 | 0.01626047 | False |
| `flat_we100_ct050` | `T3_FAR_NARROW` | PASS | - | 0 | 0 | False |
| `flat_we200_ct000` | `C2_BASE` | PASS | - | 0 | 0 | False |
| `flat_we200_ct000` | `T1_NEAR_NARROW` | FAIL | nearwall_chi_velocity | 1.776 | 0.09050177 | False |
| `flat_we200_ct000` | `T2_NEAR_WIDE` | FAIL | nearwall_chi_velocity | 0.4465 | 0.01626047 | False |
| `flat_we200_ct000` | `T3_FAR_NARROW` | PASS | - | 0 | 0 | False |
| `pillar_training` | `C2_BASE` | PASS | - | 1.561e-05 | 0 | False |
| `pillar_training` | `T1_NEAR_NARROW` | FAIL | nearwall_chi_velocity | 1.889 | 0.118514 | True |
| `pillar_training` | `T2_NEAR_WIDE` | FAIL | nearwall_chi_velocity | 0.6544 | 0.0225162 | False |
| `pillar_training` | `T3_FAR_NARROW` | PASS | - | 0.0001215 | 0 | False |
| `complex_heldout` | `C2_BASE` | PASS | - | 0.001059 | -0.007478711 | False |
| `complex_heldout` | `T1_NEAR_NARROW` | FAIL | nearwall_chi_velocity | 2.463 | 0.2183294 | True |
| `complex_heldout` | `T2_NEAR_WIDE` | FAIL | nearwall_chi_velocity | 0.8717 | 0.05100722 | False |
| `complex_heldout` | `T3_FAR_NARROW` | PASS | - | 0.007755 | -0.0667625 | False |

Quarantined taper-free negative control (not a candidate):
- `flat_we100_ct050`: front approach 0.3249727, near-wall chi ratio 0.9662, gate FAIL
- `flat_we200_ct000`: front approach 0.3249727, near-wall chi ratio 0.9662, gate FAIL
- `pillar_training`: front approach 0.3249727, near-wall chi ratio 0.8276, gate FAIL
- `complex_heldout`: front approach 0.2137239, near-wall chi ratio 1.104, gate FAIL

## G. Contour and velocity distance bands
- field layout `[x, y]`; window: open control volumes with V_i>0 within +/-8 cells (periodic x) of the frozen reconstructed phi=0.5 contour support
- `flat_we100_ct050` `C2_BASE`: 0_1: n=17 taper=0 max speed/U=0; 1_2: n=17 taper=0 max speed/U=0; 2_3: n=17 taper=0 max speed/U=0; 3_4: n=17 taper=0 max speed/U=0.006837; 4_6: n=34 taper=0.0255 max speed/U=0.7753; 6_12: n=102 taper=0.658 max speed/U=3.446
- `flat_we100_ct050` `T1_NEAR_NARROW`: 0_1: n=17 taper=0 max speed/U=0.0246; 1_2: n=17 taper=0.0161 max speed/U=0.5371; 2_3: n=17 taper=0.275 max speed/U=1.736; 3_4: n=17 taper=0.725 max speed/U=2.378; 4_6: n=34 taper=0.992 max speed/U=1.797; 6_12: n=102 taper=1 max speed/U=2.348
- frozen-gate argmax location (flat_we100_ct050, T1): cell [86, 10] sdf/dx=2.5 chi=0.03445 V/Vcell=1 taper=0.275 speed/U=1.776 (open-volume share of gate cells 0.273)

## H. Phase advection / explicit CH / implicit CH ledger
- `flat_we100_ct050` `C2_BASE`: additive identity max|residual| 7.54e-17 (active cells), phase update bitwise True, blocked-face flux 0/0, telescoped flux sum 8.67e-17, V-weighted mass closure 1.56e-18, first-substep components adv/CH-expl/CH-impl 0.0048/0.00884/0.00581, implicit CG converged True (iters 9, rel. residual 4.42e-07)
- `flat_we100_ct050` `T3_FAR_NARROW`: additive identity max|residual| 5.55e-17 (active cells), phase update bitwise True, blocked-face flux 0/0, telescoped flux sum 6.11e-17, V-weighted mass closure 1.89e-18, first-substep components adv/CH-expl/CH-impl 0.0052/0.00884/0.00571, implicit CG converged True (iters 9, rel. residual 4.43e-07)
- `flat_we200_ct000` `C2_BASE`: additive identity max|residual| 1.04e-16 (active cells), phase update bitwise True, blocked-face flux 0/0, telescoped flux sum 4.76e-17, V-weighted mass closure 1.18e-18, first-substep components adv/CH-expl/CH-impl 0.0048/0.00603/0.00581, implicit CG converged True (iters 9, rel. residual 4.44e-07)
- `flat_we200_ct000` `T3_FAR_NARROW`: additive identity max|residual| 5.55e-17 (active cells), phase update bitwise True, blocked-face flux 0/0, telescoped flux sum 4.85e-17, V-weighted mass closure 3.59e-19, first-substep components adv/CH-expl/CH-impl 0.0052/0.00603/0.00571, implicit CG converged True (iters 9, rel. residual 4.44e-07)
- `pillar_training` `C2_BASE`: additive identity max|residual| 6.64e-17 (active cells), phase update bitwise True, blocked-face flux 0/0, telescoped flux sum 1.02e-16, V-weighted mass closure 4.87e-18, first-substep components adv/CH-expl/CH-impl 0.00447/0.0497/0.0888, implicit CG converged True (iters 18, rel. residual 5.17e-07)
- `pillar_training` `T3_FAR_NARROW`: additive identity max|residual| 9.23e-17 (active cells), phase update bitwise True, blocked-face flux 0/0, telescoped flux sum 4.19e-17, V-weighted mass closure 3.26e-18, first-substep components adv/CH-expl/CH-impl 0.00517/0.0497/0.0888, implicit CG converged True (iters 18, rel. residual 5.17e-07)
- `complex_heldout` `C2_BASE`: additive identity max|residual| 1.03e-16 (active cells), phase update bitwise True, blocked-face flux 0/0, telescoped flux sum 3.6e-17, V-weighted mass closure 6.71e-19, first-substep components adv/CH-expl/CH-impl 0.00475/0.0162/0.0176, implicit CG converged True (iters 39, rel. residual 9.37e-07)
- `complex_heldout` `T3_FAR_NARROW`: additive identity max|residual| 1.1e-16 (active cells), phase update bitwise True, blocked-face flux 0/0, telescoped flux sum 4.08e-17, V-weighted mass closure 1.5e-18, first-substep components adv/CH-expl/CH-impl 0.00548/0.0162/0.0176, implicit CG converged True (iters 39, rel. residual 9.49e-07)
- identity: `delta_phi_total = delta_phi_adv + delta_phi_CH_explicit + delta_phi_CH_implicit` per cell; the implicit term is an implicit-solve increment and is never double counted
- subcycling status: {"complex_heldout": false, "flat_we100_ct050": false, "flat_we200_ct000": false, "pillar_training": false}

## I. Fluid velocity versus contour velocity
- `flat_we100_ct050`: matched contour displacement median 0.006648 dx, max 0.02732 dx, normal projection mean 0.146 (per h), usable match fraction 1, support switched False
- `flat_we200_ct000`: matched contour displacement median 0.005789 dx, max 0.02732 dx, normal projection mean 0.151 (per h), usable match fraction 1, support switched False
- `pillar_training`: matched contour displacement median 0.005947 dx, max 0.03496 dx, normal projection mean 0.1529 (per h), usable match fraction 1, support switched True
- `complex_heldout`: matched contour displacement median 0.005982 dx, max 0.02685 dx, normal projection mean 0.15 (per h), usable match fraction 1, support switched False
- independent contour reconstruction vs the frozen official rule: matched fraction 1, max match distance 4.55e-13 dx; linearized normal advance -0.000223 with conditioning 1 — explicitly `not_exact_contour_motion=True`

## J. First-substep RHS/Brinkman/projection
- `flat_we100_ct050` `C2_BASE` first substep: explicit RHS -0.0001, Brinkman 5.73e-07, periodic projection 0.00106 (approach-speed deltas); projection plain-grid mean -1.44e-14, CG converged True
- `flat_we100_ct050` `T3_FAR_NARROW` first substep: explicit RHS -9.99e-05, Brinkman 5.72e-07, periodic projection 0.00123 (approach-speed deltas); projection plain-grid mean -2.59e-14, CG converged True
- `flat_we200_ct000` `C2_BASE` first substep: explicit RHS -5.02e-05, Brinkman 2.87e-07, periodic projection 0.000949 (approach-speed deltas); projection plain-grid mean -4.99e-14, CG converged True
- `flat_we200_ct000` `T3_FAR_NARROW` first substep: explicit RHS -4.97e-05, Brinkman 2.86e-07, periodic projection 0.00112 (approach-speed deltas); projection plain-grid mean 4.41e-14, CG converged True
- `pillar_training` `C2_BASE` first substep: explicit RHS -0.0001, Brinkman 4.43e-07, periodic projection 0.000668 (approach-speed deltas); projection plain-grid mean 1.39e-14, CG converged True
- `pillar_training` `T3_FAR_NARROW` first substep: explicit RHS -9.91e-05, Brinkman 4.42e-07, periodic projection 0.000921 (approach-speed deltas); projection plain-grid mean -2.36e-14, CG converged True
- `complex_heldout` `C2_BASE` first substep: explicit RHS -0.000349, Brinkman 1.38e-06, periodic projection -0.00101 (approach-speed deltas); projection plain-grid mean 5.08e-14, CG converged True
- `complex_heldout` `T3_FAR_NARROW` first substep: explicit RHS -0.00287, Brinkman 1.15e-05, periodic projection 0.00302 (approach-speed deltas); projection plain-grid mean -4.33e-14, CG converged True

## K. Bounded event and real postcontact evidence
- eligible candidates per case (frozen t=0 gates): {"complex_heldout": ["C2_BASE", "T3_FAR_NARROW"], "flat_we100_ct050": ["C2_BASE", "T3_FAR_NARROW"], "flat_we200_ct000": ["C2_BASE", "T3_FAR_NARROW"], "pillar_training": ["C2_BASE", "T3_FAR_NARROW"]}
- ineligible candidates appear only as one-substep quarantined controls: ['complex_heldout/T1_NEAR_NARROW', 'complex_heldout/T2_NEAR_WIDE', 'flat_we100_ct050/T1_NEAR_NARROW', 'flat_we100_ct050/T2_NEAR_WIDE', 'flat_we200_ct000/T1_NEAR_NARROW', 'flat_we200_ct000/T2_NEAR_WIDE', 'pillar_training/T1_NEAR_NARROW', 'pillar_training/T2_NEAR_WIDE'] (never evolved to events, never promoted)

| case | candidate | stop | contact time | precontact approach | verdict | postcontact verified | max overshoot | max mass ratio |
|---|---|---|---:|---:|---|---|---:|---:|
| `flat_we100_ct050` | `C2_BASE` | contact_detected | 0.208645 | 0.04492073 | `CONTACT_WITHOUT_MEANINGFUL_IMPACT` | True | 0.00641 | 1 |
| `flat_we100_ct050` | `T3_FAR_NARROW` | contact_detected | 0.1860339 | 0.04991057 | `CONTACT_WITHOUT_MEANINGFUL_IMPACT` | True | 0.00639 | 1 |
| `flat_we200_ct000` | `C2_BASE` | contact_detected | 0.286762 | 0.04548784 | `CONTACT_WITHOUT_MEANINGFUL_IMPACT` | True | 0.00668 | 1 |
| `flat_we200_ct000` | `T3_FAR_NARROW` | contact_detected | 0.2458556 | 0.04793813 | `CONTACT_WITHOUT_MEANINGFUL_IMPACT` | True | 0.00665 | 1 |
| `pillar_training` | `C2_BASE` | contact_detected | 0.3536423 | 0.06940081 | `CONTACT_WITHOUT_MEANINGFUL_IMPACT` | True | 0.00586 | 1 |
| `pillar_training` | `T3_FAR_NARROW` | contact_detected | 0.2915779 | 0.07231914 | `CONTACT_WITHOUT_MEANINGFUL_IMPACT` | True | 0.00587 | 1 |
| `complex_heldout` | `C2_BASE` (extended horizon) | horizon_reached | 0.842387 | 0.1623392 | `IMPACT_AUTHENTICATED` | True | 0.00754 | 1 |
| `complex_heldout` | `T3_FAR_NARROW` (extended horizon) | horizon_reached | 0.8167565 | 0.1560864 | `IMPACT_AUTHENTICATED` | True | 0.0073 | 1 |
- `complex_heldout`/`C2_BASE`: the primary 0.8 horizon returned `NO_CONTACT_WITHIN_OBSERVED_WINDOW`; the single declared targeted continuation to t<=1.2 supplied the authoritative label and its postcontact window
- `complex_heldout`/`T3_FAR_NARROW`: the primary 0.8 horizon returned `NO_CONTACT_WITHIN_OBSERVED_WINDOW`; the single declared targeted continuation to t<=1.2 supplied the authoritative label and its postcontact window

## L. Time-step sensitivity
- matched physical times at fixed N; the official 3% refinement gate is NOT executed here

| case | candidate | approach dt=0.002 | dt=0.001 | dt=0.0005 | finest/coarsest | label |
|---|---|---:|---:|---:|---:|---|
| `flat_we100_ct050` | `C2_BASE` | 0.0449207 | 0.0235578 | 0.0117695 | 0.262 | `DT_SENSITIVE` |
| `flat_we100_ct050` | `T3_FAR_NARROW` | 0.0499106 | 0.0250108 | 0.0117368 | 0.235 | `DT_SENSITIVE` |
| `flat_we200_ct000` | `C2_BASE` | 0.0454878 | 0.0231463 | 0.0109935 | 0.242 | `DT_SENSITIVE` |
| `flat_we200_ct000` | `T3_FAR_NARROW` | 0.0479381 | 0.0225296 | 0.0103007 | 0.215 | `DT_SENSITIVE` |
| `pillar_training` | `C2_BASE` | 0.0694008 | 0.0381479 | UNMEASURED | 0.55 | `DT_SENSITIVE` |
| `pillar_training` | `T3_FAR_NARROW` | 0.0723191 | 0.0493145 | 0.0235595 | 0.326 | `DT_SENSITIVE` |
- consequence: the coarse-dt front-speed difference between C2 and an opened taper does not persist at the finest dt, so no dt-independent front-speed improvement is claimed

## M. Negative controls and falsification
- competing mechanism labels: {"H_BP": "TESTED_SEPARATELY", "H_CH": "TESTED_SEPARATELY", "H_GEOMETRY": "TESTED_SEPARATELY", "H_MULTI": "CONSISTENT_WITH_MEASUREMENTS", "H_TAPER": "SUPPORTED_IN_DIRECTION_BUT_COMPATIBILITY_BLOCKED"}
- CH-only frozen-velocity control `flat_we100_ct050`: contact reached by pure relaxation True (gap 0.07831 -> 0.008952 over t=0.48); classified `DIAGNOSTIC_COUNTERFACTUAL_CH_ONLY` and never counted as an authenticated impact
- CH-only frozen-velocity control `flat_we200_ct000`: contact reached by pure relaxation False (gap 0.07831 -> 0.07334 over t=0.48); classified `DIAGNOSTIC_COUNTERFACTUAL_CH_ONLY` and never counted as an authenticated impact
- CH-only frozen-velocity control `pillar_training`: contact reached by pure relaxation False (gap 0.08456 -> 0.08602 over t=0.48); classified `DIAGNOSTIC_COUNTERFACTUAL_CH_ONLY` and never counted as an authenticated impact
- falsification A: T1 raises the t=0 front approach to 0.0905018 but fails the frozen gate (nearwall_chi_velocity); C2 remains at 0
- projection negative controls: uniform field preserved True, max |delta v| full step 5.9e-06; projection-only 0

## N. Generator quality and inherited blockers
- `L1B_DATA_NOT_READY`, `SPATIAL_REFINEMENT=FAIL`, `TEMPORAL_REFINEMENT=UNMEASURED`, `DIRECT_FIELD_LABELS=UNMEASURED` unchanged by this stage
- inherited pillar overshoot 0.03697 > 0.02 and max mass ratio 1.005185 > 1.005 remain unresolved blockers; `PRIMARY_BLOCKER=REQUIRED_PILLAR_OR_COMPLEX_IMPACT_AUTHENTICATION` remains the owner of the next action

## O. Tests, CI and compute cost
- compute: {"N": 192, "dt_levels": [0.002, 0.001, 0.0005], "elapsed_seconds": 1035.1990078689996, "missing_or_unmeasured": [], "profile": "forensic", "stages": ["preflight", "initial", "operator", "budget", "events", "temporal", "assemble"]}
- local validation recorded at this source revision: {"focused_test_module": {"command": "JAX_ENABLE_X64=1 PYTHONPATH=. ../../.venv/bin/pytest -q tests/test_nearwall_taper_interface_transport_audit.py", "duration_seconds": 61.83, "note": "N=48 module-scope fixtures plus the N=192 saved-C2-hash gate; re-run at the sealed source revision after the final report-label edit", "result": "47 passed; 0 failed", "slow_marker_included": true}, "not_run": ["hosted CI (no PR opened for this stage revision)", "T=8 recompute", "L1B data generation", "ML training"], "quality_gate": {"codespell": "PASS_BASELINE_ONLY NEW_FINDINGS=0 (historical 3)", "command": "export PATH=\"$PWD/.venv/bin:$PATH\"; python scripts/quality_debt_gate.py --tool {ruff,format,isort,codespell}", "format": "PASS_BASELINE_ONLY NEW_FINDINGS=0 (historical 27)", "isort": "PASS_BASELINE_ONLY NEW_FINDINGS=0 (historical 28)", "note": "new L1A-2t code adds zero diagnostics; no frozen module was reformatted", "ruff": "PASS_BASELINE_ONLY NEW_FINDINGS=0 (historical 171)"}, "recorded_at_utc": "2026-10-11T09:20:34Z", "related_frozen_modules": {"command": "JAX_ENABLE_X64=1 PYTHONPATH=. ../../.venv/bin/pytest -q tests/test_impact_initialization_compatibility_audit.py tests/test_impact_impulse_projection_audit.py", "failure": "test_l1a2r_merged_base_verified_or_fail_closed - the frozen L1A-2s source gate rejects the current main 09e2993 (PR #22 merge) because it is later than the L1A-2s accepted lineage; pre-existing, unrelated to L1A-2t, reported not concealed", "result": "69 passed; 1 failed"}, "stage": "L1A-2t", "stage_source_sha256": "f8c374892fc87a873b7ee2de8457b97b8418fe5d9520c3a2aadff8777dd01e56", "stage_version": "l1a2t_v1"}
- hosted CI: not run for this stage revision at report time; no hosted result is claimed

## P. Mechanism matrix
- taper mechanism verdict: `TAPER_COMPATIBILITY_TRADEOFF`
- candidate outcome: `COMPATIBLE_IMPACT_BUT_QUALITY_OR_REFINEMENT_FAIL`
- engineering stage decision: `DIAGNOSTIC_COMPLETE_NO_PROMOTION`
- compatible candidates per case: {"complex_heldout": ["C2_BASE", "T3_FAR_NARROW"], "flat_we100_ct050": ["C2_BASE", "T3_FAR_NARROW"], "flat_we200_ct000": ["C2_BASE", "T3_FAR_NARROW"], "pillar_training": ["C2_BASE", "T3_FAR_NARROW"]}
- front-speed-raising and compatible: []
- temporal dt-sensitive: True
- phase-budget identity holds: True
- physical verdict valid: True

## Q. Exactly one next action
- Design and audit a boundary-consistent incident velocity whose advective forcing is confined to the region sdf > 3 dx while preserving the frozen chi>=0.01 near-wall gate: rebuild the seed from a single smooth streamfunction whose *radial envelope* (not the wall taper) supplies the near-wall roll-off, then re-measure the same t=0 matrix before any trajectory work.
- bounded scope: initializer design + same-state matrix only; no trajectory, no generator, no ML; not started in this stage

## R. Reproducibility manifest and raw statuses
- every evidence file is bound by SHA256 in `manifest.json`; `verify_evidence_manifest()` fails closed on a
  missing, tampered or unbound file

```text
STAGE=L1A-2t
STAGE_VERSION=l1a2t_v1
PROFILE=forensic
EVIDENCE_CLASS=FORENSIC_N192
VERIFIED_MAIN_SHA=09e2993b8da478d79e768054194201af57121784
BRANCH=arena/02b6c71b-hydrogym
SOLVER_CONTRACT_VERSION=12
TIMESTEP_POLICY=impact_phase_cap_dx2_v1
TAPER_MECHANISM_VERDICT=TAPER_COMPATIBILITY_TRADEOFF
CANDIDATE_OUTCOME=COMPATIBLE_IMPACT_BUT_QUALITY_OR_REFINEMENT_FAIL
STAGE_DECISION=DIAGNOSTIC_COMPLETE_NO_PROMOTION
TEMPORAL_DT_SENSITIVE=True
CANDIDATE_PROMOTED=false
CONTRACT13_PROMOTED=false
OFFICIAL_DATASET_WRITTEN=false
BULK_TRAINING_DATA_GENERATED=false
L1B_DATA=L1B_DATA_NOT_READY
```
