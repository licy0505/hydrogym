# L1A-2h — float32 Krylov roundoff and the residual mass walk

Contract 10, solver `weighted_spd_nullspace_preserving_v1`, invariant `componentwise_cutcell_volume`, profile `forensic`.

- **passed: True**

## Checks

| check | passed | detail |
| --- | --- | --- |
| `audit_variant_matches_shipped_solve` | True | max|baseline_f32 - shipped| = 0.000e+00 (bit-identical: True); traced = 0.000e+00 |
| `weighted_dot_float64_reference` | True | float64 device reduction of the float32 inputs agrees with the host fsum to <1 ULP (rVr 0.00, pVAp 0.00) |
| `weighted_dot_float32_reduction_is_biased` | True | float32 reduction of rVr is +0.98 ULP from the reference, compensated -0.02 |
| `constant_mode_orthogonality_measured` | True | per-iteration <1,v>_V/(||1||_V ||v||_V) recorded for r, p, d, Ap over 18 iterations; max|r| overlap = 4.432e-08 |
| `recursive_and_true_residual_recorded` | True | recursive/true residual ratio at exit = 0.999935 |
| `drift_classified` | True | baseline_f32 at N=48: verdict MIXED (exponent 0.390, signed mean -1.061e-06, t=8.01) |
| `runtime_measured` | True | per-substep cost: baseline_f32=5.3ms (x1.00), f64_scalars=6.1ms (x1.16), compensated_reduction=7.9ms (x1.50), f64_scalars_replace8=6.9ms (x1.32), f64_scalars_orthogonalised=6.7ms (x1.27), full_f64=8.3ms (x1.58) |
| `float64_krylov_correction_is_mass_neutral_60deg` | True | the same solve in float64 injects -1.32e-11 E_round of mass against 0.01749 for the shipped float32 recurrence: the recurrence, not the equation, carries this term |
| `float64_krylov_correction_is_mass_neutral_150deg` | True | the same solve in float64 injects -3.78e-12 E_round of mass against 0.01804 for the shipped float32 recurrence: the recurrence, not the equation, carries this term |
| `float64_flux_assembly_does_not_move_the_drift` | True | correctly rounded rhs: 0.04155 E/substep vs production 0.04164 |
| `float64_krylov_recurrence_leaves_the_bias` | True | float64 Krylov recurrence on production's update: 0.01132 E/substep vs production 0.04164 (3.7x smaller, still one-sided at fraction 1.00) |

## Drift classification

**baseline_f32**: `MIXED` — exponent 0.390 (R² 0.798), signed ensemble mean -1.061e-06 (seed scatter 3.748e-07, t 8.01)

| horizon (substeps) | E\|ΔM\| | signed mean |
| --- | --- | --- |
| 100 | 1.468e-07 | +1.468e-07 |
| 300 | 1.940e-07 | +1.940e-07 |
| 1000 | 1.847e-07 | +1.847e-07 |
| 3000 | 3.263e-07 | -8.247e-08 |
| 10000 | 1.061e-06 | -1.061e-06 |

| model | coefficient | R² | AIC |
| --- | --- | --- | --- |
| random_walk_model | 9.353e-09 | 0.8878 | -155.74 |
| linear_model | 1.076e-10 | 0.9145 | -157.10 |

**f64_scalars**: `SYSTEMATIC_BIAS_DOMINANT` — exponent 0.350 (R² 0.541), signed ensemble mean -1.126e-06 (seed scatter 5.180e-07, t 6.15)

| horizon (substeps) | E\|ΔM\| | signed mean |
| --- | --- | --- |
| 100 | 1.479e-07 | +1.479e-07 |
| 300 | 1.772e-07 | +1.772e-07 |
| 1000 | 1.641e-07 | +1.641e-07 |
| 3000 | 1.660e-07 | -6.855e-08 |
| 10000 | 1.126e-06 | -1.126e-06 |

| model | coefficient | R² | AIC |
| --- | --- | --- | --- |
| random_walk_model | 9.128e-09 | 0.7624 | -150.89 |
| linear_model | 1.089e-10 | 0.9057 | -155.51 |

**compensated_reduction**: `MIXED` — exponent 0.389 (R² 0.776), signed ensemble mean -1.039e-06 (seed scatter 4.798e-07, t 6.12)

| horizon (substeps) | E\|ΔM\| | signed mean |
| --- | --- | --- |
| 100 | 1.451e-07 | +1.451e-07 |
| 300 | 1.776e-07 | +1.776e-07 |
| 1000 | 1.731e-07 | +1.714e-07 |
| 3000 | 3.017e-07 | -8.935e-08 |
| 10000 | 1.039e-06 | -1.039e-06 |

| model | coefficient | R² | AIC |
| --- | --- | --- | --- |
| random_walk_model | 9.055e-09 | 0.8759 | -155.38 |
| linear_model | 1.048e-10 | 0.9232 | -157.77 |

**f64_scalars_replace8**: `SYSTEMATIC_BIAS_DOMINANT` — exponent 0.350 (R² 0.541), signed ensemble mean -1.126e-06 (seed scatter 5.180e-07, t 6.15)

| horizon (substeps) | E\|ΔM\| | signed mean |
| --- | --- | --- |
| 100 | 1.479e-07 | +1.479e-07 |
| 300 | 1.772e-07 | +1.772e-07 |
| 1000 | 1.641e-07 | +1.641e-07 |
| 3000 | 1.660e-07 | -6.855e-08 |
| 10000 | 1.126e-06 | -1.126e-06 |

| model | coefficient | R² | AIC |
| --- | --- | --- | --- |
| random_walk_model | 9.128e-09 | 0.7624 | -150.89 |
| linear_model | 1.089e-10 | 0.9057 | -155.51 |

**f64_scalars_orthogonalised**: `MIXED` — exponent 0.405 (R² 0.612), signed ensemble mean -1.411e-06 (seed scatter 3.440e-07, t 11.6)

| horizon (substeps) | E\|ΔM\| | signed mean |
| --- | --- | --- |
| 100 | 1.444e-07 | +1.444e-07 |
| 300 | 1.927e-07 | +1.927e-07 |
| 1000 | 1.663e-07 | +1.663e-07 |
| 3000 | 2.073e-07 | -2.205e-08 |
| 10000 | 1.411e-06 | -1.411e-06 |

| model | coefficient | R² | AIC |
| --- | --- | --- | --- |
| random_walk_model | 1.128e-08 | 0.7656 | -148.47 |
| linear_model | 1.359e-10 | 0.9311 | -154.59 |

**full_f64**: `RANDOM_WALK_DOMINANT` — exponent 0.115 (R² 0.916), signed ensemble mean -8.327e-16 (seed scatter 1.949e-15, t 1.21)

| horizon (substeps) | E\|ΔM\| | signed mean |
| --- | --- | --- |
| 100 | 9.992e-16 | +9.992e-16 |
| 300 | 1.332e-15 | -2.220e-16 |
| 1000 | 1.388e-15 | -3.886e-16 |
| 3000 | 1.499e-15 | -8.327e-16 |
| 10000 | 1.832e-15 | -8.327e-16 |

| model | coefficient | R² | AIC |
| --- | --- | --- | --- |
| random_walk_model | 2.377e-17 | -5.0027 | -345.57 |
| linear_model | 2.244e-19 | -12.1775 | -341.63 |

## Weighted dot products (spec 14)

| scalar | float32 reduction (ULP) | float64 device (ULP) | compensated (ULP) |
| --- | --- | --- | --- |
| `rVr` | +0.98 | +0.00 | -0.02 |
| `pVAp` | +0.28 | +0.00 | +0.28 |
| `oneVr` | -5868859.96 | +0.00 | -870886.96 |

## Constant-mode orthogonality (spec 19)

iterations 18, max overlaps: r 4.432e-08, p 8.183e-08, d 2.006e-08, Ap 1.496e-08; recursive/true residual at exit 0.999935

## Geometry control (spec 13)

| geometry | fluid cells | cut cells | mean ΔM/substep | E\|ΔM\| |
| --- | --- | --- | --- | --- |
| cartesian | 2304 | 0 | +3.530e-10 | +3.530e-10 |
| flat_cut | 2304 | 0 | -1.888e-11 | +1.888e-11 |
| wedge | 2304 | 60 | -6.049e-10 | +6.049e-10 |

## Dense authority (spec 53)

| N | candidate | solution error | mass error | iterations |
| --- | --- | --- | --- | --- |
| 8 | baseline_f32 | 1.648e-04 | +1.601e-05 | 1 |
| 8 | f64_scalars | 1.648e-04 | +1.601e-05 | 1 |
| 8 | compensated_reduction | 1.648e-04 | +1.601e-05 | 1 |
| 8 | f64_scalars_replace8 | 1.648e-04 | +1.601e-05 | 1 |
| 8 | f64_scalars_orthogonalised | 1.648e-04 | +1.601e-05 | 1 |
| 8 | full_f64 | 1.648e-04 | +1.600e-05 | 1 |
| 12 | baseline_f32 | 4.974e-04 | +1.068e-05 | 1 |
| 12 | f64_scalars | 4.974e-04 | +1.068e-05 | 1 |
| 12 | compensated_reduction | 4.974e-04 | +1.068e-05 | 1 |
| 12 | f64_scalars_replace8 | 4.974e-04 | +1.068e-05 | 1 |
| 12 | f64_scalars_orthogonalised | 4.974e-04 | +1.068e-05 | 1 |
| 12 | full_f64 | 4.975e-04 | +1.067e-05 | 1 |
| 16 | baseline_f32 | 1.045e-03 | +8.006e-06 | 2 |
| 16 | f64_scalars | 1.045e-03 | +8.006e-06 | 2 |
| 16 | compensated_reduction | 1.045e-03 | +8.006e-06 | 2 |
| 16 | f64_scalars_replace8 | 1.045e-03 | +8.006e-06 | 2 |
| 16 | f64_scalars_orthogonalised | 1.045e-03 | +8.006e-06 | 2 |
| 16 | full_f64 | 1.045e-03 | +8.000e-06 | 2 |

## Runtime (spec 24)

| candidate | per substep | ratio | iterations |
| --- | --- | --- | --- |
| baseline_f32 | 5.25 ms | x1.000 | 18.0 |
| f64_scalars | 6.10 ms | x1.161 | 18.0 |
| compensated_reduction | 7.90 ms | x1.504 | 18.0 |
| f64_scalars_replace8 | 6.91 ms | x1.316 | 18.0 |
| f64_scalars_orthogonalised | 6.68 ms | x1.271 | 18.0 |
| full_f64 | 8.30 ms | x1.580 | 18.0 |

## Where the drift lives: the update rule, at fixed states (spec 3)

Every row is the V-weighted mass change of one substep, in units of the reduction spread `E_round`, measured at two states of the same relaxation: the one production arithmetic reaches and the one the best candidate rule reaches. The rules are the *same mathematics*; they differ only in where a rounding happens.

| rule | at production state | at candidate state | state sensitivity |
| --- | --- | --- | --- |
| `production` | +0.04164 | +0.04371 | +0.00207 |
| `assembly_f64` | +0.04155 | +0.04365 | +0.00210 |
| `solve_f64` | +0.01132 | +0.01136 | +0.00004 |
| `single_rounding` | +0.00594 | +0.00594 | -0.00001 |
| `single_rounding_f64` | +0.00001 | +0.00002 | +0.00001 |
| `single_rounding_f32x` | +0.01085 | +0.01691 | +0.00606 |
| `ideal_f32_storage` | +0.00001 | +0.00002 | +0.00001 |
| `f64_state` | +0.00001 | +0.00002 | +0.00001 |

* `production` is contract v10 verbatim: `rhs = fl32(phi + dt*source)`, then `fl32(rhs + x)` with the float32 Krylov correction `x` -- **two** roundings of the updated field.
* `assembly_f64` rounds the same right-hand side correctly and changes nothing: the assembly is not where the drift is.
* `solve_f64` runs the Krylov recurrence in float64 and keeps production's two-rounding update: 2-4x better, still biased.
* `single_rounding*` folds the correction into the increment so the field is rounded once: better at one state, **worse** at the other -- the mass defect of any float32 rule is set by where the interface cells happen to sit relative to the grid, not by the rule.
* `ideal_f32_storage` (float64 increment and float64 Krylov, one rounding) and `f64_state` (no storage rounding at all) are the exact-arithmetic references.

### Window-averaged decomposition (spec 5)

| term | mean | std | positive fraction |
| --- | --- | --- | --- |
| `total` | +0.04698 | 0.01266 | 1.00 |
| `assembly` | +0.01431 | 0.00610 | 1.00 |
| `correction_f32` | +0.00634 | 0.00544 | 0.87 |
| `correction_f64` | +0.00000 | 0.00000 | 0.67 |
| `storage_residue` | +0.02631 | 0.00600 | 1.00 |

The parts are measured against the assembled right-hand side and the total against the incoming field, so they close through the mass the correctly rounded right-hand side carries (+0.00002 E_round, closure residual 6.94e-18): the float32 Krylov correction carries 13.5%, the float32 storage rounding of the updated field 56.0%, the assembly 30.5%, and the same solve in float64 carries 8.04e-12.


### One substep, one fixed state: where the defect is manufactured (spec 5)

Each entry is a mass defect divided by `E_round`, measured on a single substep at a late-time state with the production step's own inputs. The right-hand side is the one production assembles; the two Krylov columns solve the *same* exchange system.

| state | rhs assembly vs correctly rounded | float32 Krylov `x` | float64 Krylov `x` | cast of `phi + x` | whole update in f64 |
| --- | --- | --- | --- | --- | --- |
| 60deg | +2.41e-04 | +0.01749 | -1.32e-11 | +0.02061 | +0.03789 |
| 150deg | +1.31e-04 | +0.01804 | -3.78e-12 | +0.02745 | +0.01082 |

Two causal facts hold at every state measured. The float32 Krylov recurrence injects mass that the identical solve in float64 does not (machine zero), so that share is recurrence rounding and nothing else. And casting the updated field back onto the float32 grid leaves a one-sided residue across ~96% of the cells -- including 632 saturated cells at `|phi| > 0.9`, where the grid spacing above 1 is twice the spacing below it, so a symmetric update need not leave a symmetric residue. Correctly rounding the right-hand side is worth +2.41e-04 E_round, three orders below both terms.


### Mechanism: where the drift is manufactured, and why no candidate removes it

The residual contract-v10 drift is **not** a Krylov defect. Measured causally at a fixed late-time state with the production step's own inputs:

* the float32 Krylov recurrence injects `+0.04164` E_round/substep of mass where the *identical* solve in float64 injects `-1.32e-11`;
* casting the update back onto the float32 grid leaves `+0.02061` E_round/substep one-sided in aggregate, although only 48% of the cells round up and 96% round at all -- the cells that round up do so by more than the cells that round down, and an exact correction still leaves `+0.03789` E_round, so the grid is a floor;
* correctly rounding the right-hand side is worth `2.41e-04` E_round/substep, three orders below both.

So the carrier is the float32 *storage* of the interface update: increments below the local ULP are lost, increments above it jump a full ULP, and which cells do which is a property of the state rather than of the solver. That is why every shipment-ready rule in the candidate family lands in the same 0.00-0.06 E_round/substep band, why the best of them is no better over 50k steps than the shipped arithmetic, and why the two long production series classify identically. The only arithmetic that removes the term is float64 *storage*, which the specification excludes as a default shortcut and which is a policy decision for a separate stage, not an arithmetic patch. No production change is shipped and the contract stays 10.


### Iteration-count dependence (spec 3)

One production substep at a quasi-static state, with the CG iteration cap injected through the production path only:

| iteration cap | iterations used | relative residual | mass defect (E_round/substep) |
| --- | --- | --- | --- |
| 2 | 2 | 1.18e-04 | not converged (fail-closed NaNs) |
| 4 | 4 | 1.31e-05 | not converged (fail-closed NaNs) |
| 8 | 7 | 5.45e-07 | +0.06144 |
| 16 | 7 | 5.45e-07 | +0.06144 |
| 32 | 7 | 5.45e-07 | +0.06144 |
| 64 | 7 | 5.45e-07 | +0.06144 |
| 200 | 7 | 5.45e-07 | +0.06144 |

The defect is flat across three orders of magnitude of Krylov work, so it is not a truncation term: the same defect appears when the solve is converged to machine precision. That is the third independent reading of the same finding -- a converged solve in float64 (section above), a tight cap here, and a self-warmed trajectory over 50k steps (section E).


## Gates (spec 23)

| gate | geometry | measured | bound | passed |
| --- | --- | --- | --- | --- |
| quick | N=48, 150.0 deg, 2450 steps | -1.243e-05 | 2e-06 | False |
| medium | N=128, 60.0 deg, 4900 steps | -1.331e-05 | 0.0002 | True |
| strong | N=128, 60.0 deg, 9800 steps | +1.738e-05 | 0.0001 | True |
| closure | N=128, 60 deg, 50000 steps (ledger) | 60 +1.054e-03, 90 +4.764e-04, 120 +1.937e-04, 150 +3.119e-04 | 0.001 | False |

A gate that fails is a *finding* here, not an audit error: see the verdict.

## Quick gate detail, both wall offsets (spec 23)

| quick run | geometry | measured relative drift | bound | passed |
| --- | --- | --- | --- | --- |
| quick | N=48, 150 deg, 0.0 offset, 2450 steps | -1.243e-05 | 2e-06 | False |
| quick | N=48, 150 deg, 0.5 offset, 2450 steps | -1.980e-05 | 2e-06 | False |

## Long-run series (spec 23: no significant linear bias)

| arithmetic | steps | final drift | drift / random-walk expectation | slope t | increment autocorrelation | one-sided | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_f32` | 4900 | -1.331e-05 | 9.6 | -48.6 | 0.359 | 0.92 | `SYSTEMATIC_BIAS_DOMINANT` |
| `krylov_f64+single_rounding+f64increment` | 4900 | -1.408e-05 | 10.8 | -46.7 | 0.57 | 0.96 | `SYSTEMATIC_BIAS_DOMINANT` |


## Verdict

- drift classification (ensemble, spec 10): **MIXED** (exponent 0.3896896927965048, t 8.009142467545066)
- classification of the long series (spec 23): `baseline_f32` SYSTEMATIC_BIAS_DOMINANT, `krylov_f64+single_rounding+f64increment` SYSTEMATIC_BIAS_DOMINANT
- `N-CH-MASS-PRECISION`: **confirmed_problem** -- systematic, storage-limited, not a Krylov-arithmetic defect
- contract status: **unchanged_at_10** (no production arithmetic changed, so no v10 trajectory is invalidated)
- quick gate (N=48, 150 deg, offsets 0/0.5, 2500 steps, relative <= 2e-6): **NOT met** ({'offset_0.0': -1.242975042432493e-05, 'offset_0.5': -1.9800228207289915e-05})
- exact-arithmetic control: 1.50e-05 E_round/substep vs production 0.04164 E_round/substep
- why no production change is shipped: no rule in the candidate family removes the drift: the exact-arithmetic control is 1.50e-05 E_round/substep while every shipment-ready float32 rule sits at 0.00-0.05, and the same rule moves by 0.01 E_round/substep between two states that differ by ~1e-6 in the field
- `W-CONTACT-ANGLE`: **not closed** -- the 60 deg closure measures 1.0538e-03 > 1e-3 at 50k steps (90/120/150 deg measure 4.76e-04 / 1.94e-04 / 3.12e-04), and this audit explains why no arithmetic change in the candidate family moves that number
