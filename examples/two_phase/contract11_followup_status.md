# Contract 11 follow-up — status and measured results

Scope: follow-up audits only. No contract bump or production-physics/default change was made. In particular, accelerated CH-only mobility (`4*M_ref`) is confined to equilibrium-geometry audits.

## Phase-mass gate definition

The formal cut-cell invariant is `sum_i(V_i * phi_i)` (`pf.liquid_mass`). The historical `dx*dy*sum(phi)` / `total_mass` full-grid reconstruction is retained for diagnostics only; it is not conserved by the cut-cell transport and is explicitly excluded from formal acceptance gates. Tests exercise both its nonconservation and the rule that a large diagnostic drift cannot reject a case whose formal mass drift passes.

## Formal CH-only thermodynamic matrix

Configuration: contract 11, float64 phase state, `M=4*M_ref`, `N=128`, targets 60/90/120/150 degrees. All four cases genuinely converged.

| Target | Steps | Converged angle | Error | Conserved-mass drift | Full-grid diagnostic drift |
|---:|---:|---:|---:|---:|---:|
| 60° | 44,850 | 60.4815730643° | +0.4815730643° | 1.27e-15 | 5.43e-3 |
| 90° | 6,900 | 89.7599339605° | -0.2400660395° | 4.62e-16 | 3.34e-4 |
| 120° | 42,750 | 119.2869887080° | -0.7130112920° | 8.08e-16 | 6.30e-3 |
| 150° | 58,950 | 149.0517364238° | -0.9482635762° | 2.19e-15 | 1.09e-2 |

Matrix MAE is 0.5957284930°, maximum absolute error 0.9482635762°, and the neutral-angle error is 0.2400660395°. The specific four-target CH-only and formal-mass gates pass. The encompassing audit report remains fail-closed for other sections that were intentionally not run in this focused invocation.

## Alignment/grid regression

Completed previously: 60° at `N=128`, offsets 0 and 0.5 `dy`, both converged:

| Target | Offset | Angle | Formal conserved-mass drift | Full-grid/hard-mask diagnostic drift |
|---:|---:|---:|---:|---:|
| 60° | 0 `dy` | 60.4796058075° | 1.500792e-15 | 5.43e-3 |
| 60° | 0.5 `dy` | 60.4709884564° | 3.347919e-15 | 2.88e-3 |
| 150° | 0 `dy` | 149.0593767946° | 2.424357e-15 | 1.09e-2 |
| 150° | 0.5 `dy` | 149.1818066843° | 2.655246e-15 | 5.78e-3 |

The two-offset spreads are 0.0086173511° at 60° and 0.1224298897° at 150°. Both are well within the 2° alignment threshold. The large diagnostic drifts above are not formal mass failures.

Grid-resolution cases (fixed offset 0 `dy`):

| Target | N | Steps | Converged angle | Conserved-mass drift | Full-grid diagnostic drift |
|---:|---:|---:|---:|---:|---:|
| 60° | 96 | 42,950 | 60.3720138332° | 3.088662e-15 | 3.09e-15 |
| 60° | 128 | 44,850 | 60.4796058075° | 1.500793e-15 | 5.43e-3 |
| 60° | 192 | 47,450 | 60.6221008922° | 1.394492e-15 | 1.51e-15 |
| 150° | 96 | 61,050 | 149.5571025591° | 3.889426e-15 | 3.89e-15 |
| 150° | 128 | 58,950 | 149.0593767946° | 2.424357e-15 | 1.09e-2 |
| 150° | 192 | 64,250 | 149.1401764864° | 1.394492e-15 | 1.39e-15 |

The 60° grid spread is 0.2500870590° and the 150° grid spread is 0.4977257645°; both are within the 2° resolution threshold. All six grid cases converged and passed their formal mass-drift checks.

## Staged production CHNS closure

All four targets completed 50,000 steps before any extension decision. Extension policy required complete, drift-clean, nonstationary cases that the existing classifier still identified as relaxing.

| Target | Final milestone | Angle | Stationary | Still relaxing at decision | Conserved-mass drift | Full-grid diagnostic drift | Decision |
|---:|---:|---:|:---:|:---:|---:|---:|---|
| 60° | 50,000 | 59.6426515162° (sampled; not converged) | No | No | 1.962575e-15 | 6.132335e-3 | Stop: classifier said no longer relaxing |
| 90° | 50,000 | 89.4016117734° | Yes | No | 9.235647e-16 | 4.942136e-4 | Stop: stationary window passed |
| 120° | 50,000 | 118.8443099821° | Yes | No | 5.772279e-16 | 5.624030e-3 | Stop: stationary window passed |
| 150° | 100,000 | 147.7536522852° | Yes | No | 2.655248e-15 | 1.046164e-2 | Extend once from 50k; stop at stationary window |

The staged protocol completed, but **four-angle production CHNS closure did not pass** because 60° had no converged equilibrium angle. It was not extended because the specified relaxation classifier reported `still_relaxing=false`. No 150k/200k stage was warranted for the other cases.

## Tests and checks

- Alignment audit unit tests: 10 passed in 10.27s with `JAX_ENABLE_X64=1`.
- Focused mass/non-gating, CHNS extension-policy, and report-schema tests: 4 passed in 30.38s with `JAX_ENABLE_X64=1`.
- `py_compile` and `git diff --check` passed for the changed audit/test files.
- Full `test_production_validation.py` timed out after 600 seconds during earlier concurrent long audits; no final test summary was produced. This is not counted as a pass.

## Final scope/readiness statement

The mass-definition correction, four-target CH-only matrix, and requested 60°/150° alignment and three-grid equilibrium regressions are complete; their specific gates pass. The CHNS staged protocol also completed exactly under the requested extension rule, but **the four-angle production CHNS closure gate remains unmet**: the 60° case did not converge and the classifier did not label it as still relaxing at 50k, so extending it would violate the stated policy. The focused alignment runner did not execute the other L1A-2f sections; unmeasured gates in that partial report are not treated as passes, and its `NOT READY: false` trigger summary is not a general production-readiness claim.

No contract bump or production physics/default change was made.
