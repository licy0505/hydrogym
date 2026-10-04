# L1A-2h — float32 Krylov round-off and the drift-clean CHNS closure: final report

Stage L1A-2h, stacked on the L1A-2g head (`70a1932`, contract **10**). `STACKED_PR=YES`,
`BASE_IS_L1A2G_HEAD=YES`. Everything below is measured in this tree with the commands in section Q.

**Verdict in one line: the residual contract-v10 mass drift is a systematic, state-driven float32
*storage* bias (interface pinning), not a Krylov round-off defect; no candidate in the specified
family A–F removes it, so no production arithmetic is changed, the contract stays at 10, and
`W-CONTACT-ANGLE` stays open.**

---

## A. Scope and authority

The stage studies exactly two things:

1. the residual mass drift of contract v10's phase transport, classified as a random walk or as a
   systematic bias, using at least eight deterministic zero-mass perturbation seeds *and* long
   production-default series;
2. whether a drift-clean CHNS contact-angle closure at 50k/100k/150k steps is reachable with a
   minimum-cost change to the float32 Krylov arithmetic.

Everything else is frozen and verified untouched: Young wall energy, cut-cell geometry, wall
measure, the contact-angle mapping, the capillary density denominator, variable-viscosity stress,
Brinkman penalisation, gas properties, the mass matrix `M`, the timestep policy and the pressure
projection. `N-DT` is not used as a lever and is unchanged; `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN` stays
`resolved_in_contract_v9`; `P-SOLID-PIN` and `I-CONTACT-GAP` stay independent. No block or
multi-component scheme is introduced: the exchange solve keeps its fail-closed single-component
precondition and stops if the fluid region is not one connected component.

## B. Base, contract, and what changed in the tree

| item | value |
| --- | --- |
| base commit | `70a1932` (contract 10, `weighted_spd_nullspace_preserving_v1`) |
| solver contract after this stage | **10** — unchanged |
| `DATASET_SCHEMA_VERSION` | 3 — unchanged |
| production arithmetic changed | **no** |
| new files | `production/krylov_roundoff_audit.py`, `test_krylov_roundoff.py`, `evidence/l1a2h/` |
| changed files | `.github/workflows/python-ci.yml` (test + quick audit step) |
| `phasefield.py` | **not touched** |

Because no production arithmetic changes, the spec's contract bump 10 → 11 does not apply: the
metadata block (`krylov_vector_precision`, `krylov_scalar_precision`, `krylov_weighted_reduction`,
`cg_residual_replacement_period`, `constant_mode_maintenance`) is *reported* for every candidate in
the audit report but never promoted to a shipped configuration, and v10 trajectories stay valid.

## C. Harness and fidelity

The audit drives the production step itself (`phasefield.step_with_diagnostics`) for every long
series, and mirrors the substep only where a candidate arithmetic has to be injected. Two fidelity
facts are asserted by tests rather than assumed:

* `variant_substep(..., "shipped")` reproduces one production substep **bit for bit**
  (`max|Δφ| = 0.0`, `max|Δu| = 0.0`, `max|Δv| = 0.0`) at a late-time CHNS state;
* the 50 000-step series measured through the variant harness reproduces the production scan to
  0.1 % (`+1.0551e-03` vs `+1.0560e-03`), i.e. jit-program reassociation, not arithmetic.

One harness defect was found and fixed *before* any of the numbers below were taken: an earlier
draft staged the **unsubcycled** advective rate while `phasefield._phase_update` recomputes the
advected rate with `advective_phase_source` whenever phase-advection subcycling is enabled. The
staged masses of that draft described a different equation and overstated the per-substep defect by
an order of magnitude. `production_advective_rate` now selects the rate production selects, and the
substep identity test above is what guards it. The production default is
`PHASE_ADVECTION_SUBCYCLING = "disabled"`, so the audit's fixtures take the same path production
takes; the guard matters for configurations that enable subcycling.

Cross-harness comparisons are quoted only where the same compiled program produces both numbers;
absolute per-substep defects are never compared across harnesses (they differ by ~30 % at nominally
identical states, which is itself part of the finding in section G).

## D. Task 1 — classification: systematic bias, not a random walk

The decisive series are the production-default CHNS runs at `N = 128`, 60° contact angle,
`M = M_ref`, `rtol = 1e-6`, three `dt/3` substeps per step, sampled every 100 steps
(`evidence/l1a2h/long_run_series.json`; reproducible with
`production.krylov_roundoff_audit.variant_drift_series`).

| statistic | contract-10 baseline | best candidate rule (section E) |
| --- | --- | --- |
| cumulative relative drift after 50 000 steps | `+1.0560e-03` | `+1.2167e-03` |
| cumulative drift in `E_round` units | `+4605.7` | `+5306.7` |
| linear-slope t-statistic of the cumulative drift | `+96.7` | `+90.6` |
| lag-1 autocorrelation of the per-100-step increments | `+0.971` | `+0.997` |
| fraction of increments with the drift's sign | `0.898` | `0.904` |
| cumulative drift / random-walk expectation `σ√N` | `38.6×` | `38.3×` |
| second-half drift / first-half drift | `2.78×` | `2.99×` |
| verdict | **SYSTEMATIC_BIAS_DOMINANT** | **SYSTEMATIC_BIAS_DOMINANT** |

A random walk of independent roundings would give a cumulative of order `σ√N` (ratio ≈ 1), no
lag-1 correlation, an increment sign split near 1/2 and a second-half/first-half ratio near 1/3. All
four indicators fail together, in both series, and the block means rise monotonically after the
early transient (`+0.7, −2.6, …, +12.2, +12.6, +14.4, +14.3, +14.5, +14.1, +13.5, +12.2, +13.4`).
The drift is therefore *not* a walk: it is a one-sided bias whose rate roughly doubles between the
first and second halves of the horizon.

### The four task-1 audits

| audit | measurement | reading |
| --- | --- | --- |
| V-weighted dot products (D1 float32 reduction, D2 float64 device, D3 host `fsum` reference, D4 compensated device) | `rVr` (the CG denominator) is `+0.98 ULP` high in float32 where D2 is exact; `pVAp` `+0.28 ULP`; the compensated reduction lands at `−0.015 ULP`. For the near-cancelling constant-mode overlap `⟨1, r⟩_V` the float32 reduction is `58 %` off in relative terms and compensation cannot recover it (`−8.7 %`: the *products* are already rounded), while D2 is `0.004 ULP` | the reductions are biased by ~1 ULP and biased reductions *can* be fixed — but fixing them does not move the trajectory (section E) |
| recursive vs true residual | recursive/true at exit `0.999935` (constant mode), `1.0000062` (production step); `true_residual_max ≈ 1.1e-05` | the recurrence does not drift from the true residual, so accurate-residual replacement has nothing to fix (candidate D: no effect) |
| constant-mode orthogonality `⟨1, r⟩_V, ⟨1, p⟩_V, ⟨1, d⟩_V, ⟨1, Ap⟩_V`, normalised by `‖1‖_V‖v‖_V` | recorded per iteration over 18 iterations; `max|proj r| = 4.43e-08` | the Krylov iterates carry a constant-mode component far below the defect scale; maintaining orthogonality (candidate E) buys 26 % at a fixed state and nothing over a horizon |
| iteration-count dependence | caps 2 and 4 do not converge (fail-closed NaNs, as contract 10 requires); from cap 8 to cap 200 the solve stops at 7 iterations and the substep defect is `+0.06144` E_round every time | the defect is not a truncation term — it survives a converged solve unchanged |

The eight-seed deterministic ensemble (zero-mass perturbation seeds, horizons 100 … 10 000 steps,
all candidates) is reported alongside these in the JSON as the spec's task-1 measurement. In the
ensemble the *short-horizon* classification is `MIXED`/walk-like (exponent ≈ 0.39–0.72, one-sided
mean with t ≈ 3.8–8.0) — short horizons cannot separate a bias from a walk when the walk noise is
larger than the bias accumulated so far. The long series, which can, say bias; the two readings are
consistent and the long series is the decider.

## E. Task 2 — the candidate matrix

Fixed-state comparison (`update_rule_defects`, `N = 128`, 60°, production-step state after a
20 000-step warmup, 12-substep windows; units of `E_round` per substep). Every rule integrates the
*same* equation with the *same* fluxes and the *same* timestep — only the arithmetic differs.

| rule (spec candidate) | defect at the fixed state | trajectory: 50 000 steps at 60° |
| --- | --- | --- |
| A `production` (shipped contract 10) | `+0.04164` | `+1.0560e-03` |
| `assembly_f64` (float64 flux/rhs assembly) | `+0.04155` | not run — no fixed-state effect |
| B `solve_f64` (float64 Krylov scalars *and* vectors) | `+0.01132` | — |
| C `compensated_reduction` (compensated V-weighted dot products) | no fixed-state change (rVr bias `+0.98 ULP` → `−0.015 ULP`, drift unchanged) | — |
| D `f64_scalars_replace8` (periodic accurate-residual replacement, period 8; 4/16/32 in the matrix) | recursive/true residual `1.0000062` — replacement has nothing to fix | — |
| E `f64_scalars_orthogonalised` (constant-mode orthogonality maintained, own coefficient) | `+0.0330` (26 % better, still one-sided) | — |
| `single_rounding_f32x` (single rounding, float32 Krylov) | `+0.01085` | ~`−10 %` in the exploratory run; superseded by the stronger candidate below |
| `single_rounding` (single rounding, float64 increment, float32 Krylov) | `+0.00594` | ~`−10 %` in the exploratory run; superseded |
| **`single_rounding_f64` = `ideal_f32_storage`** (float64 Krylov **and** float64 increment, one field rounding) | **`+0.00001`** (4000× better) | **`+1.2167e-03` (+15 % *worse*)** |
| F `f64_state` (full float64 reference/control) | `+0.00001` | — (out of scope as a default) |

The two ~10 % rows are the exploratory 50 000-step runs that motivated the stronger candidate;
the committed series file carries the two decisive series (shipped and the best rule) so the
comparison can be re-checked without trusting a number whose artifact is not in the repository.

Two readings matter:

1. **No shipment-ready rule removes the drift.** The best fixed-state rule sits at the float64 floor
   (`+0.00001` vs `+0.04164`, a factor 4000) and yet over 50 000 steps drifts *more* than the shipped
   arithmetic (`+1.2167e-03` vs `+1.0560e-03`, factor 1.15). A 12-substep window therefore cannot be
   used to predict a long horizon: this stage measured that failure directly instead of trusting it.
2. **The candidate family covers the space.** Compensated reductions, residual replacement and
   constant-mode orthogonality — the three defect *mechanisms* the spec asked about — change the
   fixed-state defect by at most 26 % and change the drift not at all. The float64 *scalars* variant
   costs `1.16×` runtime and buys 3.7× at the fixed state; the float64 *recurrence* on the
   production update costs `1.42×` for the whole substep and buys 2200× at the fixed state; neither
   survives the trajectory.

Runtime (per production substep, `N = 128`, same machine, ratio to the shipped solve, measured in
the report JSON): `f64_scalars 1.16×`, `f64_scalars_replace8 1.32×`, `f64_scalars_orthogonalised`
`1.27×`, `compensated_reduction 1.50×`, `full_f64 1.58×`; the best *update* rule
(`krylov_f64+single_rounding+f64increment`) costs `1.42×` of the whole substep. These are CPU wall
clocks and are quoted as ratios, not absolutes; between two consecutive runs on the same machine the
ratios move by ~10 % (e.g. the best rule measured `1.42×` and `1.71×` on two runs of the same
fixture), so the ordering is meaningful and the third digit is not. On a GPU the float64 recurrence is worse than this —
consumer parts run float64 at 1/32 of the float32 rate and the doubled Krylov vectors double the
memory traffic — so the price of the only rule that touches the fixed-state defect is higher than
the CPU number. That cost buys no measurable drift reduction, which is why nothing is shipped.

## F. Mechanism — where the defect is actually manufactured

Fixed-state causal decomposition (`solve_drift_decomposition`, `N = 128`, warmup 20 000 steps, one
substep, production's own inputs; `E_round` per substep):

| term | 60° | 150° |
| --- | --- | --- |
| right-hand side assembly vs correctly rounded assembly | `+2.41e-04` | `+1.31e-04` |
| float32 Krylov correction `x` (mass it injects) | `+1.749e-02` | `+1.804e-02` |
| **float64 Krylov correction `x` (same system)** | **`0` (exactly mass-neutral)** | **`0` (exactly mass-neutral)** |
| cast of `phi + x` back onto the float32 grid | `+2.061e-02` | `+2.745e-02` |
| whole update assembled and solved in float64, cast once | `+3.789e-02` | `+1.082e-02` |
| exact-arithmetic control (`f64_state`, `update_rule_defects`) | `+1.5e-05` | `+1.5e-05` |

The float64 solve of the *same* float32 right-hand side injects **exactly zero** mass in these
fixtures (at other states it is `~1e-11`, still machine zero), so the Krylov recurrence's share is
pure recurrence rounding and is removable in principle. The storage cast is larger, touches ~96 % of
the cells (`632` saturated cells at `|phi| > 0.9`, where the float32 spacing above 1 is twice the
spacing below it), and is **state driven**: increments below the local ULP are lost, increments above
it jump a whole ULP, and which cells do which is a property of where the interface sits in the grid.
Note the shape of the residue: only `48 %` of the cells round up, yet the mean residue is positive —
the cells that round up do so by more than the cells that round down, so the bias is not "most cells
drift the same way" but "the same way on average". Correctly rounding the right-hand side, by
contrast, is worth `2.4e-04` — two orders below both terms.

That state dependence is measured, not asserted. The same rule, evaluated at two states that differ
by ~1e-6 in `phi` (both reachable on the same trajectory), moves by `0.0165` (quick profile) to
`0.0061` (forensic profile) `E_round`/substep — the same order as the whole defect. The immediate
consequence is that any candidate that wins a 12-substep window can lose over a horizon, which is
exactly what the `single_rounding_f64` rule does in section E.

## G. Gates

| gate | bound | measured | status |
| --- | --- | --- | --- |
| quick: `N = 48`, 150°, 2500 steps, wall offsets 0.0 / 0.5 | `≤ 2e-6` | `−1.243e-05` / `−1.980e-05` | **not met** |
| medium: `N = 128`, 60°, 5000 steps | `≤ 2e-04` | `−1.331e-05` | **met** |
| strong: `N = 128`, 60°, 10 000 steps | `≤ 1e-04` | `+1.738e-05` | **met** |
| long run 5k/10k/25k/50k/100k: no significant linear bias | — | slope `t = +96.7` at 50k, one-sided | **not met** |
| closure: `W-CONTACT-ANGLE` closes only when all four angles are drift clean at `1e-3` | `≤ 1e-3` | 60° `1.0538e-03` (fails); 90° `4.764e-04`, 120° `1.937e-04`, 150° `3.119e-04` (pass) | **not met** |

The quick gate fails at both wall offsets with the same sign, which is the same bias seen from a
different angle. Because the production-default 50k horizon already leaves the 60° case above the
closure's `1e-3` drift bound while 90°/120°/150° pass it, and the closure requires all four angles
drift clean, the spec's extension rule stops the staged horizon at 50k: running 100k/150k/200k cannot
change the closure decision. The staged
series are kept in `long_run_series.json` as the evidence for that call rather than being extrapolated.

## H. What this means for the closure

`W-CONTACT-ANGLE` remains **not closed**: over 50 000 steps the 60° case drifts `1.0538e-03`
against a `1e-3` gate, while 90° (`4.764e-04`), 120° (`1.937e-04`) and 150° (`3.119e-04`) are drift
clean. The blocked angle is therefore 60° alone, and it is blocked by `5.4e-05` of relative drift --
which is why the arithmetic question was worth asking, and why the answer (section F) matters. This is not a statement about the angles themselves — all
four are close to their targets — it is the conserved-drift gate the spec sets for calling the
closure done. Since the drift is a float32 *storage* effect (section F), it cannot be removed by any
change to the Krylov arithmetic, which is the second thing this stage establishes.

The honest options are therefore *policy* options, not arithmetic patches:

* keep contract 10 and the documented drift budget, and treat the 60° drift as a known
  float32-storage limitation of the closure (this stage's default);
* or promote float64 *storage* for the phase field in a separate, explicitly scoped stage, with its
  own contract bump and its own physics gates — the spec excludes "full float64 as a default" as a
  shortcut here, and this stage does not take it.

## I. Anti-cheating evidence

The audit ships tests that would fail if the drift were made to vanish by cheating rather than by
arithmetic:

* the solver path is scanned for post-step mass correction, global `phi` offset/rescale, mass
  redistribution projection, and any comparison against a previous/initial mass target;
* the shipped solver contract constants (`SOLVER_CONTRACT_VERSION == 10`,
  `IMPLICIT_PHASE_SOLVER`, `PHASE_MASS_INVARIANT`) are asserted unchanged;
* `variant_flags` fails closed on unknown names (no silent fallback to production arithmetic);
* `variant_substep(..., "shipped")` is asserted bit-identical to the production substep, so a
  "candidate" cannot be reported as production;
* constant-mode projections, where they are evaluated at all, take their coefficient from the
  current Krylov vector (`<1, v>_V / <1, 1>_V`) and never from a stored mass or a target.

Forbidden and not used anywhere in this stage: post-step mass correction, global phi offset or
rescale, mass-redistribution projection, float64-as-default, any change to `M`, `dt`, Brinkman,
gas properties, wall measure, cut-cell geometry or the pressure projection.

## J. Independence and untouched blockers

`N-DT` is evidence-only and untouched by this stage; nothing here changes or closes it. The
single-component fail-closed precondition of the exchange solve is unchanged, and still
verified: `test_mass_precision.py::test_single_fluid_component_and_fail_closed_on_two` shows
the fixture is one component and a split chamber is detected as two, and this stage's tests
assert the solver path never learns a mass target or rescales the field. `P-SOLID-PIN` and
`I-CONTACT-GAP` are untouched and independent. `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN` stays
`resolved_in_contract_v9`. `N-CH-MASS-PRECISION` resolves to **`confirmed_problem`**: the drift is a
real, reproducible, systematic bias of the contract-10 arithmetic — it is *not* walk-limited, and it
is *not* resolved in contract v10 (or v11, since nothing was shipped).

## K. CI

`.github/workflows/python-ci.yml` gains the new test file in the two-phase test step and a
`--profile quick` run of the audit. The audit's exit code covers its own integrity checks only; the
mass gates are recorded as *findings* (`--require-gates` promotes them to a failure for callers that
want that), because "the drift is systematic" is this stage's result, not a broken run. No job claims
that all CI passed, and the pre-existing `Audit locked dependencies` failure in the Quality workflow
is unchanged and unrelated to this stage. Because that step fails first, the Quality job's
lint/format/import-order/spelling steps are *skipped* in CI (as they already are on `main`); the new
files were therefore checked locally with the pinned tool versions — `ruff 0.15.22` (`ruff check`
and `ruff format --check`), `isort` with the repository's `black` profile, and `codespell` — and pass
all four, so this stage adds no lint debt.

## L. How to reproduce

```bash
cd examples/two_phase
JAX_ENABLE_X64=1 python -m pytest -q test_krylov_roundoff.py
JAX_ENABLE_X64=1 python -m production.krylov_roundoff_audit --profile quick     --out /tmp/l1a2h_quick
JAX_ENABLE_X64=1 python -m production.krylov_roundoff_audit --profile forensic  --out evidence/l1a2h
JAX_ENABLE_X64=1 python -m production.krylov_roundoff_audit --profile forensic --require-gates \
    --out /tmp/l1a2h_forensic_gates   # exits non-zero: the gates are not met
```

The 50 000-step series in `evidence/l1a2h/long_run_series.json` come from
`variant_drift_series(128, target_deg=60.0, steps=50000, sample_every=100, variant=...)` for
`"shipped"` and for `"krylov_f64+single_rounding+f64increment"`.

## M. Artifacts

| artifact | content |
| --- | --- |
| `evidence/l1a2h/krylov_roundoff_report.json` | every number, check and verdict of the forensic profile, including the gate matrix, the fixed-state decomposition, the iteration-cap table and both runtime matrices |
| `evidence/l1a2h/krylov_roundoff_report.md` | the same report rendered, with the mechanism section |
| `evidence/l1a2h/manifest.json` | provenance: stage, module, contract, profile, and the hashes of the two report files |
| `evidence/l1a2h/long_run_series.json` | the two 50 000-step closure series and their classification |
| `evidence/l1a2h/L1A2H_FINAL_REPORT.md` | this document |
| `production/krylov_roundoff_audit.py` | the audit itself (profiles quick / baseline / forensic) |
| `test_krylov_roundoff.py` | integrity, fidelity, classification and anti-cheating tests |

## N. Open items handed forward

1. **Storage-level precision policy** (the only lever that removes the bias): a separate stage, with
   its own spec, contract bump and physics gates — the phase field in float64 (or a compensated
   sub-grid accumulator for the interface update) and the closure re-run.
2. **Interface-pinning characterisation**: the per-substep defect is a slowly varying, state-driven
   quantized residue; quantifying the pinning length scale would let the closure gate be stated as a
   drift *budget* instead of an absolute number.
3. **60° closure drift**: documented, not closed. Do not read the 60° drift as a physics error; it
   is the float32 grid, and section E shows no arithmetic candidate fixes it.

## O. Compliance summary against the spec's hard-fail list

| hard-fail condition | status |
| --- | --- |
| production change before the classification | not applicable — no production change at all |
| state-mass correction / offset / redistribution / full-float64 default | none present (section I) |
| an unconverged angle called equilibrium | none claimed |
| drift `> 1e-3` after a claimed closure | no closure claimed; `W-CONTACT-ANGLE` stays open |
| dense/VJP/L0 failures | audit reports them; see the JSON checks |
| physics or geometry edits | none (`phasefield.py` untouched) |
| equilibrium shift `> 0.5°` or spread `> 2°` | none; the closure rows are quoted from contract-10 evidence |
| "all CI passed" claim | not made |

## P. Status lines

```
N-CH-MASS-PRECISION                 confirmed_problem
N-DT                                evidence_only_untouched
N-WALL-ALIGNMENT-TRANSPORT-DOMAIN   resolved_in_contract_v9
P-SOLID-PIN                         independent
I-CONTACT-GAP                       independent
W-CONTACT-ANGLE                     not_closed
SOLVER_CONTRACT_VERSION             10  (unchanged: no production arithmetic changed)
DATASET_SCHEMA_VERSION              3   (unchanged)
STACKED_PR                          YES (base = L1A-2g head 70a1932)
```

## Q. Reproduction and provenance

All numbers are produced by the module in this tree at the commit that contains this report; the
JSON report records the profile, the fixtures (`N`, `M_factor`, `rtol`, contact angle, warmup) and
every intermediate quantity, so each table cell above can be traced to a key rather than to a
sentence. The two long series are stored in full (501 samples each) so the classification statistics
can be recomputed without re-running the 50 000 steps.
