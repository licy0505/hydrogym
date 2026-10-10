# Focused `pillar_training` versus flat-canary precontact diagnosis

**Status: `diagnostic_only`; `production_lineage_eligible=false`.** This is a narrow L1A-2s follow-up to the saved T=8 evidence. It compares only the requested `pillar_training`, `flat_we100_ct050`, and `flat_we200_ct000` canaries. It does not change production source, contract 12, the `impact_phase_cap_dx2_v1` timestep policy, initializer defaults, geometry, placement, contact threshold, or readiness status. No official data were generated.

## Scope and identity checks

- The compared seed is `SDF_TAPERED_STREAMFUNCTION_V1` (C2), an **isolated diagnostic initializer**, not the production default.
- Reused the saved `impact_event_matrix.json` contact times, local event velocities, and postcontact verification. The new measurement advanced from the exact saved initial states only to their saved precontact times: 104, 143, and 176 public steps, respectively. Effective `dt=0.002`, `h=dt/3`, `N=192`, and `eps/dx=1.5`. The targeted run took 39.7 seconds on the local CPU environment; the approximately 43-minute T=8 calculation was **not** repeated.
- Exact geometry, phase, and candidate velocity hashes for all three cases match `initializer_constraint_matrix.json`; initial operator-ledger values match the saved startup ledger. At precontact, the rerun reproduces the saved `phi=0.5` gap and normal approach speed within `2e-6`. The temporary stage-capture kernel agrees with the audited fast step to at most `1.2e-7` in state fields.
- The event definition remains the bilinear actual-SDF gap on the periodic reconstructed `phi=0.5` contour, with contact at `gap <= 1.5 dx`. The inherited meaningful-approach floor remains `0.2*u_impact = 0.1`. No geometry or contact was manufactured.
- Saved T=8 postcontact windows are reused; this short run ends **before** first contact and makes no new postcontact claim. `complex_heldout` was not rerun or replaced.

Reproduction script: `run_short_operator_comparison.py`. Machine-readable results and hashes: `pillar_flat_operator_comparison.json` and `comparison_manifest.json`.

## 1. Initialized velocity: the core starts fast, but the contact-side sample does not

At initialization, the liquid-core vertical mean is effectively `-0.5` in every C2 case. However, at the actual nearest-wall support of the reconstructed `phi=0.5` contour, the measured normal approach is **zero** in all three. Thus the incoming-speed issue is not a missing bulk/core velocity; the requested core speed is present but is not initially delivered at the local contact-side sample.

| Canary | C2 core `v` (`phi>=0.9`) | C2 liquid-weighted `v` | Initial local normal approach | Central-D Linf | Open-face FV Linf |
|---|---:|---:|---:|---:|---:|
| `flat_we100_ct050` | -0.500000 | -0.310452 | 0.000000 | `2.921e-6` | `9.179e-6` |
| `flat_we200_ct000` | -0.500000 | -0.310452 | 0.000000 | `2.921e-6` | `9.179e-6` |
| `pillar_training` | -0.500000 | -0.317647 | 0.000000 | `2.861e-6` | `5.722e-6` |

The central-D and open-face finite-volume values are separate discrete measurements, not interchangeable divergence claims. The open-face result uses the declared adjacent-cell velocity reconstruction, contract-9 apertures, and division by `V_i` (`MEASURED_OPEN_FACE_RECONSTRUCTION`). Embedded-wall flux is a separate bilinear wall-centroid measurement. Exact initial metrics, statuses, and hashes are in the JSON.

## 2. Saved event comparison: the low-speed result is shared, and the pillar is not the slowest case

| Canary | Saved precontact `t` | Interpolated contact `t` | `phi=0.5` gap / `dx` at precontact | Local normal approach | Fraction of 0.1 floor | Core `v` at precontact | Saved postcontact dynamics |
|---|---:|---:|---:|---:|---:|---:|---|
| `flat_we100_ct050` | 0.208 | 0.208645 | 1.5229 | 0.04492 | 44.9% | -0.49827 | verified |
| `flat_we200_ct000` | 0.286 | 0.286762 | 1.5067 | 0.04549 | 45.5% | -0.47782 | verified |
| `pillar_training` | 0.352 | 0.353642 | 1.5102 | 0.06940 | 69.4% | -0.49570 | verified |

All three remain below the unchanged floor. The pillar approach is about **1.53–1.55× higher** than either flat approach, although it also fails the same floor. At the saved contact locations, all three nearest-wall normals are `[0, -1]`: the pillar event is on its actual horizontal top (`wall y=0.65`), while the flat events are on `wall y=0.25`. This rules out an oblique-wall-normal projection as the explanation for the pillar's low value. The core velocities also remain near `-0.5`, so the shortfall is localized to the contact-side flow rather than a comparable loss of bulk speed.

## 3. First internal substep: pressure projection creates the initial local approach

Positive local normal speed points **toward** the solid. Starting from zero at the C2 `phi=0.5` support, the first RHS stage gives a small away-from-solid value; Brinkman changes it only slightly; the periodic pressure projection then produces the larger positive local response:

| Canary | After explicit RHS | After Brinkman | After pressure projection | Projection's local-normal increment |
|---|---:|---:|---:|---:|
| `flat_we100_ct050` | -0.0001004 | -0.00009983 | +0.0009562 | +0.0010560 |
| `flat_we200_ct000` | -0.00005016 | -0.00004988 | +0.0008986 | +0.0009485 |
| `pillar_training` | -0.0001003 | -0.00009983 | +0.0005682 | +0.0006681 |

These are the actual-dtype, same-support stage samples from the first `h=0.0006667` internal substep. They show the same qualitative mechanism on both flat surfaces and the pillar: projection redistributes velocity into the local approach direction; initial Brinkman response at this support is only about `3e-7` to `6e-7`.

## 4. Precontact operator response: local correction and domain mean are different quantities

For each final public step ending at the saved precontact time, the local support and its nearest actual wall-measure normal were recomputed at each of the three internal substeps. The following are mean local-normal increments per internal substep; positive is toward the solid. The net is the sum of the three displayed stage increments.

| Canary | Explicit RHS | Brinkman | Periodic pressure projection | Net local-normal change |
|---|---:|---:|---:|---:|
| `flat_we100_ct050` | +0.000762 | -0.000854 | +0.000145 | +0.000053 |
| `flat_we200_ct000` | +0.000668 | -0.000886 | +0.000239 | +0.000021 |
| `pillar_training` | +0.000909 | -0.001315 | +0.000452 | +0.000047 |

This last-step pattern is also shared: Brinkman damping locally opposes downward approach, while projection adds a smaller positive local-normal correction; explicit RHS offsets much of Brinkman's local reduction. The net approach changes only slightly per substep. The pillar's local damping and projection increments are larger in magnitude, but their signs are not a pillar-only mechanism.

Across the **last 10 public steps** (30 internal substeps), the global liquid-weighted `Delta v` differs from the contact-point normal response:

| Canary | RHS `Delta v_liquid` | Brinkman `Delta v_liquid` | Projection `Delta v_liquid` | Brinkman full-grid mean `Delta v` | Projection full-grid mean `Delta v` | Projection vertical correction Linf |
|---|---:|---:|---:|---:|---:|---:|
| `flat_we100_ct050` | -1.231e-4 | +1.000e-5 | +1.585e-4 | +1.812e-7 | +4.93e-12 | 3.496e-3 |
| `flat_we200_ct000` | -1.306e-4 | +1.040e-5 | +1.804e-4 | -2.872e-7 | +7.54e-12 | 3.518e-3 |
| `pillar_training` | -5.523e-5 | +2.260e-5 | +8.208e-5 | +1.736e-6 | -2.61e-12 | 2.959e-3 |

All deltas are per internal substep. `Delta v_liquid` uses the audited `V_i*phi_i` weights frozen at the substep input; positive lab-frame `Delta v` is upward. At the contact support, by contrast, positive normal speed is downward into the wall. Therefore the positive liquid-weighted projection increments above do **not** mean that the local contact-side projection increment points away from the wall: the captured local increments in the preceding table are positive. This is spatial redistribution, not uniform damping.

The periodic pressure-gradient correction's full-grid mean remains at roundoff scale (maximum absolute mean over the window: `5.09e-11`, `6.41e-11`, and `4.59e-11`, respectively), while its vertical correction Linf is about `3e-3`. Brinkman's full-grid mean is separately nonzero and can change sign with the case/time window. These quantities are not conflated. All 30 projection solves per case converged; post-projection central-D Linf remained about `2e-6` to `3e-6` in the final window.

## 5. Discrete divergence and cut-cell measurements at precontact

The saved event matrix had precontact open-face/wall measurements for `flat_we100_ct050`; those fields were null for the later `flat_we200_ct000` and `pillar_training` rows. The short rerun measured all three at their exact saved precontact states. Central-D and finite-volume/open-face divergence remain separate; the table gives each with its own declared reconstruction. Embedded-wall flux uses the bilinear wall-centroid reconstruction and is listed separately.

| Canary | Central-D Linf | Central-D fluid-volume L2 RMS | Open-face FV Linf | Open-face FV fluid-volume L2 RMS | Embedded-wall signed / absolute flux | Max bilinear wall-normal speed |
|---|---:|---:|---:|---:|---:|---:|
| `flat_we100_ct050` | 2.384e-6 | 1.440e-7 | 0.8007 | 0.01664 | +0.0001624 / 0.01997 | 0.02502 |
| `flat_we200_ct000` | 2.205e-6 | 1.412e-7 | 0.7953 | 0.01708 | -0.0003120 / 0.02093 | 0.02485 |
| `pillar_training` | 1.907e-6 | 1.285e-7 | 5.5669 | 0.04569 | -0.002746 / 0.02234 | 0.03730 |

Open-face entries are `MEASURED_OPEN_FACE_RECONSTRUCTION`; wall-flux entries are `MEASURED_WITH_DECLARED_BILINEAR_CELL_VELOCITY_RECONSTRUCTION`. The pillar's open-face reconstruction is much larger than the flat values even though central-D divergence remains small; this is a separate cut-cell/open-face diagnostic, not a central-D residual and not by itself a claim of physical wall penetration. The measurement choices and formulas are preserved in the JSON. No unmeasured value is treated as zero.

## 6. Diagnosis and preserved status

**Supported diagnosis:** C2 initializes a fast core but zero approach at the near-wall liquid contour. As the cases evolve, both flat canaries and the pillar develop a local approach far below `u_impact`; the pillar is faster than the flats but still below the inherited floor. Near the event, Brinkman damping reduces local approach and pressure projection redistributes velocity back toward the local wall-normal direction, while the liquid-weighted and full-grid means tell different stories. The evidence supports a shared contact-side initialization/evolution bottleneck, not a pillar-only low-speed mechanism. It does not establish that any single operator is the sole cause.

The saved postcontact windows are retained as the required event evidence; the new run did not advance through contact. The 3% refinement gate is not reinterpreted. Historical statuses remain unchanged:

| Status | Preserved value |
|---|---|
| Initializer | `INITIALIZER_COMPATIBLE` |
| Fleet event verdict | `PARTIAL_IMPACT_AUTHENTICATION` |
| Primary blocker | `REQUIRED_PILLAR_OR_COMPLEX_IMPACT_AUTHENTICATION` |
| Stage decision | `B_INITIALIZATION_HELPFUL_BUT_REFINEMENT_OR_GEOMETRY_BLOCKS` |
| L1A | `BLOCKED` |
| L1B data | `L1B_DATA_NOT_READY` |
| Generator quality gates | `FAIL` |
| Spatial refinement | `FAIL` |
| Temporal refinement | `UNMEASURED` |
| Direct-field labels | `UNMEASURED` |
| `P_VARDENS_PROJ` / `W_CONTACT_ANGLE` | `OPEN` / `OPEN` |
| `N_DT` | `TARGET_CRITICAL` |
| `D_FRESH_TRAIN_CONTRACT` | `OPEN` |

No candidate, threshold, contract, timestep, geometry, official data, training data, or production lineage status was promoted or changed.
