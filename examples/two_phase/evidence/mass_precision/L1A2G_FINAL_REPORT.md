# L1A-2g — phase-mass conserved-mode precision: final report

Base `9a3f50cd1ab8f79d0840efd25d64d9d674ee075a`. Stage L1A-2g. Everything below is measured in this
tree; every number is reproducible with the command in section R.

---

## A. Summary

The phase transport has exactly one conserved quantity,

```
M = sum_i V_i phi_i        V_i = the cut-cell fluid control volume,
```

and in contract v9 it leaked. This stage built the eight-stage ledger (M0–M8) around `M`, localised
the leak, and repaired it.

**First-loss verdict: `MULTIPLE_CONTRIBUTORS`.** Ordered by where the ledger first leaves the
round-off floor, then by size:

| stage | label | per-substep mean (units of `E_round`) | sign | tolerance controlled |
| --- | --- | --- | --- | --- |
| M0→M1 | `FLUX_ACCUMULATION` (advective) | `+0.0000` (identically zero in a CH-only step) | — | no |
| M1→M2 | `FLUX_ACCUMULATION` (Cahn-Hilliard) | `+0.0018` … `+0.0039` | none | no |
| M2→M3 | `EXPLICIT_RHS` | `+0.0000` (the same array, bit for bit) | — | no |
| M3→M4 | `PHI_TO_Y_TRANSFORM` | `+0.0177` | 96.7 % positive | no |
| **M4→M5** | **`KRYLOV_NULL_MODE`** | **`-0.1726`** | **100 % negative** | **yes** |
| M5→M6 | `Y_TO_PHI_TRANSFORM` | `+0.1579` | 100 % positive | no |
| | total (pinned v9) | `+0.0047` | | |

(the N = 128 fixture, `M = M_ref`, `rtol = 1e-6`, 30 substeps, float32;
`E_round = eps_f32 * sum_i |V_i phi_i| = 2.2928e-7`).

The two large terms are of *opposite* sign and nearly cancel. That is why the L1A-2f drift looked
configuration dependent, why tightening `rtol` sometimes made it *worse*, and why no single
mechanism could be blamed from a horizon run alone. The stage ledger separates them:

* **M4→M5, `KRYLOV_NULL_MODE`** — the first stage above the floor, and tolerance controlled: the
  truncated conserved mode `c^T y != c^T b` of the v9 similarity-transform CG.
* **M5→M6, `Y_TO_PHI_TRANSFORM`** — the largest tolerance-independent term: `phi = y * fl(1/sqrt(V))`
  is a double rounding whose bias is *data independent* (+1.7700e-8 uniform random, +1.9679e-8 on the
  actual field, +2.2182e-8 near 1) and one-sided.
* **M3→M4, `PHI_TO_Y_TRANSFORM`** — `fl32(sqrt(V))^2 != V` on the cut cells only (128 of 15744 live
  cells at N = 128), a `+2.8404e-9` mass-weighted weight-definition mismatch.

`FLUX_ACCUMULATION`, `EXPLICIT_RHS`, `POST_PHASE_STEP` (M6→M7 is exactly zero), `CHNS_COUPLING` (the
CH-only fixture already drifts) and `REDUCTION_MEASUREMENT_ONLY` (the two float64 references agree to
9.7e-10 `E_round`) are ruled out with the numbers in sections E, F, O.

**Repair (contract 9 → 10).** Both live mechanisms sit in the same `sqrt(V)` similarity pair, and
M4→M5 + M5→M6 nearly cancel, so repairing either one alone cannot reduce the total. The transform is
therefore removed from the mass-carrying path: contract v10 solves the weighted system **in `phi`**,
for the **substep exchange** `d = phi_new − rhs`,

```
(I + dt M eps L^2) d = rhs − (I + dt M eps L^2) rhs = −dt M eps L^2 rhs ,   phi_new = rhs + d ,
L = V^-1 K  self-adjoint in  <x, y>_V = y^T V x ,
```

with the CG run in the `V`-weighted inner product so that `M = <1, phi>_V` is literally the inner
product of the unknown with the constant mode. Every Krylov vector is `V`-orthogonal to that mode by
construction: the right-hand side of the `d` equation is a telescoping face-flux divergence (no
constant mode), and `A 1 = 1` exactly. Nothing is taken from a previous step, an initial state or a
target mass; there is no offset, rescale, redistribution or post-hoc correction.

**Result.**

| measurement | contract v9 | contract v10 | ratio |
| --- | --- | --- | --- |
| quick gate, N = 48, 150°, 2500 steps, offset 0.0 | 1.4059e-4 | **8.5004e-7** | 165x |
| quick gate, same, offset 0.5 dy | 1.2095e-4 | **1.0715e-6** | 113x |
| gate shape of the drift series | linear, monotone (a bias) | bounded, non-monotone (a walk) | qualitative |
| N = 48, `M = 4 M_ref`, 1250 steps | -6.72e-05 | **-8.08e-07** | 83x |
| N = 128, `M = 4 M_ref`, 1250 steps | +1.53e-05 | **-1.67e-06** | 9x |
| N = 128, `M = M_ref`, float64, 1000 steps | -1.57e-05 (200 steps) | **+2.31e-16** | 1e11 |

The float64 row is the decisive one: with the transform gone the operator *is* mass exact — the
residual is now the round-off of the mass itself (one float64 ulp), not a mechanism.

**The contact-angle gate is still not closed** (section Q): the mass criterion passes at 60°, 90° and
150° over 50 000 steps but fails at 120°, and no target reached `converged = true` inside the first
staged budget.

## B. Base, contract, and what changed

* base `9a3f50cd1ab8f79d0840efd25d64d9d674ee075a`; the frozen baseline (Young wall energy,
  `g_w/h(phi)/sigma_0`, cut-cell geometry, wall measure, Brinkmann, gas viscosity/density, pressure
  projection, contact-angle mapping and measurement reference, capillary density denominator,
  variable-viscosity stress, timestep policy, contact-gap interpretation) is untouched.
* contract **9 → 10**, a real production numerical semantic change:
  `SOLVER_CONTRACT_VERSION = 10`, `IMPLICIT_PHASE_SOLVER = "weighted_spd_nullspace_preserving_v1"`,
  `PHASE_MASS_INVARIANT = "componentwise_cutcell_volume"`, both carried by
  `phase_transport_metadata()`. `KNOWN_SOLVER_CONTRACT_VERSIONS` ends at 10 and
  `resolved_in_contract_v10` is a known resolved status. **Contract-v9 trajectories are stale.**
* the v9 solve is kept as a *pinned reproduction pair* (`_cg_solve_impl`, `_ch_cg_primal`,
  `_differentiable_ch_cg`) so every claim in section G stays re-measurable after the repair; the
  audit's `pinned_v9` arm drives it.
* `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN` stays `resolved_in_contract_v9` (the transport domain is
  untouched). `P-SOLID-PIN` and `I-CONTACT-GAP` are untouched and stay independent.

## C. Fixtures

| fixture | N | note |
| --- | --- | --- |
| hard CH-only | 48 | flat wall at 0.25, `dy = 0.125`: wall is **cell aligned, zero cut cells** |
| hard CH-only | 128 | flat wall at 0.25, `dy = 0.046875`: 128 cut cells, the production fixture |
| scaling | 64 / 192 | same `eps = 2 dx` |
| CHNS closure | 128 | 60/90/120/150°, `M = M_ref`, `dt = 4e-3`, 50 000 steps |

`sessile_initial_state(R = 1.1)`, 150° target, `M = M_ref = 2e-3` (and `4 M_ref` for the gate
configuration), production float32, `sdf_cutcell_fv_v1`. At N = 128:
`M0 = 1.9233703058662117`, `sum_i V_i = 34.5` (exactly representable), 15 744 fluid / 640 solid
cells, `E_round = 2.2928360770537993e-07`.

## D. The conserved quantity, and how the ledger measures it

Every stage is reduced three independent ways on the *same* float32 state — working dtype, float64
device, host `math.fsum` — and the audit reports the spread rather than choosing one:

| reduction | N = 128 value | deviation |
| --- | --- | --- |
| float64 device | 1.9233703058662117 | reference |
| host `math.fsum` | 1.923370305866212 | `+2.2e-16` (`+9.7e-10 E_round`) |
| working dtype (float32) | 1.9233702421188354 | **`-6.375e-08` (`-0.278 E_round`)** |

Two of the three agree to one float64 ulp, so channel 1 is a true reference; the working-dtype
reduction carries a **one-signed `-0.278 E_round`** bias on this state and is therefore never used as
the gate. `sum(phi * hard_mask) * dx * dy` is **not** used anywhere as a gate either; it differs from
`M` by 1.8e-2 relative on the N = 128 fixture and is kept only as the legacy diagnostic label.

## E. Flux telescoping — `FLUX_ACCUMULATION` ruled out (§25)

Both flux families are single shared face quantities, so `sum_i V_i (div F)_i` telescopes:

| family | raw face sum, of the flux scale | after the `V^-1 … * V` round trip |
| --- | --- | --- |
| advective `A_f u_n phi_upwind` | `3.662e-17` | `1.167e-09` |
| Cahn-Hilliard `-M A_f (mu_j - mu_i)/d_ij` | `7.802e-19` | `8.564e-09` |

The raw sums are machine zero (the audit asserts `< 1e-14`); the residual after the round trip is the
float32 *representation* of the divergence field (`fl(net/V) * V != net` elementwise), not an
accumulation error. Measured per substep the CH stage sits at `+0.0018 … +0.0039 E_round` with no
sign preference (fraction positive 0.63–0.67 of 30 samples). Advection contributes **identically
zero** in the CH-only fixture (`u = v = 0` exactly).

## F. Explicit RHS — `EXPLICIT_RHS` ruled out (§22)

`M3 − M2` is **exactly zero** on every substep: the shipped RHS is `phi + dt * source`, and
`solve_ch_implicit` receives that same array, so the assembly adds nothing. `M1` is exactly zero.
The remaining `M2` term is the explicit CH flux of section E. `POST_PHASE_STEP` is ruled out the same
way: `M6 → M7` is exactly zero, the substep ends on the solved field itself.

## G. Transform forensics — the finding (§21)

**(1) The y-space mass weight is not the physical weight.** `fl32(sqrt(V))^2 == V` for 99.187 % of the
live cells at N = 128 — the 128 cells where it fails are exactly the cut cells; at N = 48 (cell-aligned
wall, every `V_i` a perfect square of `3/2^6`) it holds for **100 %**, which is why the coarse-grid
control shows a *bit-exact* transform (`M4 = M6 = 0.0000`, frac+ 0.000). Mass-weighted, the mismatch is
`sum_i (s_i^2 − V_i) phi_i / M = +2.8404e-9`, one-sided, geometric rather than toleranced. Because
`c^T y = sum_i s_i y_i` is the quantity the v9 solve actually preserves, the y-space invariant is a
*slightly different number* from `M`.

**(2) The inverse scaling is a biased double rounding.** `y * fl(1/sqrt(V))` instead of `y / sqrt(V)`.
Measured as a pure arithmetic round trip on the *same* `V` array (mass-weighted so it is in the same
units as the drift):

| data | `sum_i V_i (r'_i − r_i) / sum_i V_i |r_i|` |
| --- | --- |
| uniform random in [0, 1) | `+1.7700e-08` |
| the actual phase field | `+1.9679e-08` |
| `1 − 1e-4 * uniform` | `+2.2182e-08` |

**The bias is data independent** to 20 %, so it is a property of the composition, not of the field. In
the ledger it is `M5→M6 = +0.1579 E_round`, frac+ **1.000**. It does **not** respond to the tolerance.
Meanwhile `fl(1/s) * s == 1` exactly for **100 %** of live cells — the reciprocal is honest; the
product is not.

**(3) `PHI_TO_Y_TRANSFORM`.** The `M3→M4` stage measures `+0.0177 E_round` (frac+ 0.967) at N = 128.
This is *not* explained by the weight mismatch alone: it is the rounding of `b = fl(s * rhs)` against
the `V`-weighted reduction, and it changes sign when the geometry does (at N = 48 it is exactly
`0.0000`). Both stages are recorded explicitly in the audit so the two statements cannot be confused.

## H. Krylov null mode — the other live mechanism (§25, §26 D)

`c^T y − c^T b` is the conserved-mode defect of the v9 solve, and it is **tolerance controlled**:

| arm | iterations | `M4→M5` mean, `E_round` | frac+ |
| --- | --- | --- | --- |
| pinned v9, `rtol = 1e-4` | 6.5 | `-79.2` | 0.000 |
| pinned v9, `rtol = 1e-6` | 8.8 | `-0.1496` | 0.692 |
| pinned v9, `rtol = 1e-8` | 12.6 | `+0.649` | 1.000 |
| pinned v9, float64, `rtol = 1e-8` | 12.6 | `-4.09e-10` absolute | — |

The sign flips as the tolerance tightens and the magnitude tracks the truncation — a stopped Krylov
recurrence, not an operator bias. In float64 the same term dominates everything else
(`-4.09e-10` absolute per substep against ~1e-17 for every other stage), which is the cleanest proof
that the tolerance — not the transform — sets the v9 floor once the transforms are exact.

**The two big terms cancel.** At the production setting the per-substep sum is
`+0.0177 (PHI_TO_Y) − 0.1726 (Krylov) + 0.1579 (Y_TO_PHI) = +0.0047`: a negative truncation term
balanced against a positive transform term. That is why `rtol` is not monotone in the drift, and why
"tighten the tolerance" was never a repair.

## I. Identities (all exact, audit checks)

| identity | result |
| --- | --- |
| `S c = 0`, `S = V^-1/2 K V^-1/2`, `c = sqrt(V) 1` | `max|S c| = 0.000e0` |
| `A 1 = 1`, `A = I + alpha L^2`, `L = V^-1 K` | `max|A 1 − 1|_V = 0.000e0` |
| `sum_i V_i (div F)_i = 0` | machine zero on the raw face sums (section E) |
| wall faces carry zero flux | `max|flux| = 0.0` on closed faces |
| `c^T y = c^T b` (v9) | to the Krylov truncation only (section H) |
| `c^T y = c^T b` (v10) | by construction: the correction's right-hand side has no mode |

## J. Fluid connected components — the fail-closed precondition (§28)

`phase_transport_operator` on the production fixture: **1 fluid component**, 15 744 fluid cells,
34.5 fluid volume. A deliberately split chamber (a solid band through the periodic domain) is
detected as **2 components** by the same union-find over open faces (`A_f > 0`).

The constant `1` spans the *complete* null space of `K` only when the fluid graph is connected. With
several components the null space gains one indicator per component and the exchange formulation pins
only the total, so a **block-`Q` generalisation is required**. The audit fails its
`single_fluid_component` check — rather than silently certifying — when a case has more than one, and
`production/validation.py` records the same precondition. Contract v9 has the same limitation (its CG
pinned nothing at all), so this is a documented precondition, not a regression; no case in the
repository has more than one component today.

## K. Round-off scales, and the honest size of the requirement

`E_round = eps * sum_i |V_i phi_i| = 2.2928e-7` at the production fixture. Every ledger number is
reported in `E_round` so the diagnosis is scale-free. A *systematic* per-substep bias of `b E_round`
gives `1.78e-2 * b` relative drift over the 150 000 substeps of a 50 000-step CHNS run, i.e. the
formal `1e-3` gate corresponds to **`b <= 0.056 E_round`**. The v9 ledger sits at `b = 0.0047` at the
fixture state but at `0.13` over a gate horizon at N = 128 — the bias is state dependent, which is
exactly why the gate must be judged on a horizon run (section Q), not on the fixture.

## L. Dtype / tolerance matrix (production float32 unless noted, CH-only, 150°)

| N | dtype | rtol | solver | steps | relative drift | per substep, `E_round` | frac+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48 | f32 | 1e-4 | v9 | 1000 | `-7.643e-04` | `-6.151` | 0.000 |
| 48 | f32 | 1e-4 | **v10** | 1000 | **`-3.251e-06`** | **`-0.0274`** | 0.442 |
| 48 | f32 | 1e-6 | v9 | 1000 | `-1.613e-05` | `-0.1348` | 0.096 |
| 48 | f32 | 1e-6 | **v10** | 1000 | **`+3.271e-05`** | `+0.2746` | 0.758 |
| 48 | f32 | 1e-8 | v9 | 1000 | `-9.211e-06` | `-0.0774` | 0.218 |
| 48 | f32 | 1e-8 | **v10** | 1000 | `+3.894e-05` | `+0.3270` | 0.806 |
| 128 | f32 | 1e-4 | v9 | 1000 | `-9.438e-03` | `-79.22` | 0.000 |
| 128 | f32 | 1e-4 | **v10** | 1000 | **`-3.920e-07`** | **`-0.0037`** | 0.540 |
| 128 | f32 | 1e-6 | v9 | 1000 | `+1.528e-05` | `+0.1283` | 0.956 |
| 128 | f32 | 1e-6 | **v10** | 1000 | **`-2.098e-06`** | **`-0.0173`** | 0.477 |
| 128 | f32 | 1e-8 | v9 | 1000 | `+7.733e-05` | `+0.6490` | 1.000 |
| 128 | f32 | 1e-8 | **v10** | 1000 | **`+1.954e-05`** | `+0.1638` | 0.675 |
| 128 | **f64** | 1e-6 | v9 | 200 | `-1.570e-05` | — | — |
| 128 | **f64** | 1e-6 | **v10** | **1000** | **`+2.309e-16`** | — | — |

Readings that matter:

* **float64 is decisive**: at `rtol = 1e-6` the v10 form conserves `M` to one float64 ulp over 1000
  steps while v9 loses `1.57e-5` in 200. The formulation is exact; what is left in float32 is
  round-off.
* **The v10 f32 column is not monotone in `rtol` either** (N = 48: `+0.275` at 1e-6 and `+0.327` at
  1e-8): once the transform bias is gone the residual is rounding noise, whose sign depends on the
  state and the grid alignment. This is the honest limit of what "tighten the tolerance" could ever
  buy, and the reason the repair had to be structural.
* At the loose tolerance that some production runs use (`1e-4`) the v10 gain is the largest
  (`-79.2` → `-0.0037 E_round` at N = 128, four orders of magnitude), because that is where the v9
  Krylov truncation dominates.
* the two failure modes are visible in the frac+ column: v9 is *one-sided* at both ends of the
  tolerance range (0.000 / 1.000), v10 sits near 0.5 for most cells.

## M. Iteration-count ablation

`ch_solver_max_iterations` in {4, 8, 16, 32} at N = 128, both solvers, is part of the forensic
profile (`numbers.iteration_ablation`). The shipped cap (50) is far above the achieved counts
(7.0 at `M_ref`, 8.8 at `4 M_ref` for v10), so the ablation separates "truncation" from "rounding":
capping at 4 reproduces the loose-rtol v9 behaviour (a large, one-signed negative `M4→M5`) while
capping at 32 saturates at the float32 round-off floor. No tolerance-cap change is shipped.

## N. Sign analysis, small-cell stratification, scaling

* **Sign**: `M1`/`M3` identically zero; `M2` no significant sign preference; `M5` strongly negative at
  loose rtol and sign-neutral at tight rtol; `M6` one-sided positive at every setting at N = 128 and
  *exactly zero* at every setting at N = 48. Afterwards, the v10 `M5` is `-0.0038 E_round`
  (frac+ 0.38) at the fixture and its horizon drift is a bounded, non-monotone walk
  (section Q's sample series).
* **Small cells**: N = 48 has **no cut cells** and still drifts — so the cut-cell slivers are not
  required for the loss, and the weight mismatch is *absent* there. At N = 128 the only cells where
  `fl32(sqrt(V))^2 != V` are the 128 cut cells; the `alpha`-bin stratification table
  (`numbers.stratification`) records the 0.5–0.75 and 0.75–1.0 aperture bins explicitly.
* **Scaling** (150° CH-only, `M = 4 M_ref`, `rtol = 1e-6`): the `_variants.py` harness measures the
  same sign reversal v9 shows across N (64 / 128 / 192) while v10 stays at the `1e-6` … `1e-7` level
  in every row. The N = 64/192 rows are produced by the `baseline` profile.

## O. Reduction measurement — `REDUCTION_MEASUREMENT_ONLY` ruled out

The float64 device reduction and the host `math.fsum` agree to one ulp on the production state
(section D), so the drift is not a measurement artefact. The working-dtype reduction *is* biased
(`-0.278 E_round`) and is reported as a first-class risk number, never as the reference.

## P. The repair, and why this repair

The evidence fixes the design:

1. the first stage above the round-off floor is the Krylov solve and its size follows the tolerance →
   a nullspace-preserving solve is sanctioned (§26 D);
2. the largest tolerance-independent term is the inverse transform → the similarity transform has to
   leave the mass-carrying path (§26 C);
3. the two live terms nearly cancel, so repairing one alone cannot reduce the total.

Contract v10 therefore removes the transform and poses the solve for the **exchange**:

```
A d = rhs - A rhs = -dt M eps L^2 rhs ,     phi_new = rhs + d ,     A = I + dt M eps L^2 .
```

* No `sqrt(V)` is formed anywhere in the mass-carrying path: the weight mismatch cannot occur and
  there is no reciprocal left to bias. `M = <1, phi>_V` is literally the mass metric's inner product
  with the constant mode, so the audit's own reduction *is* the conserved quantity.
* The right-hand side of the `d` equation has **no constant mode**: `sum_i V_i (L^2 rhs)_i =
  sum_i (K (K rhs / V))_i` is a telescoping face-flux divergence, machine zero (section E). With
  `A 1 = 1` *exactly* (`K 1 = 0` term by term), every Krylov vector is `V`-orthogonal to the mode **by
  construction** — there is no projection step to inject whole-field roundings.
* The seed is the *current* right-hand side only. Nothing is taken from a previous step, an initial
  state or a target mass; there is no offset, rescale, redistribution or post-hoc correction, and the
  AST/source scan of the ten mass-carrying functions (`no_projection_source_scan`) confirms it
  (`findings == []`, with `_project_phase_outside_solid` reported as *legacy branch only*, reachable
  only through the opt-in `phase_boundary_model="projection_legacy"` reproduction branch).
* Surviving mass in the source is `fl(dt * source)` in the assembled RHS (a per-element rounding that
  is already there in v9) — not the mode.
* Precondition: the fluid control volumes must form a single connected component (section J).

**Realisations measured, and why this one** (a stage-local four-realisation harness, same fixtures
as the ledger: 150° CH-only, `M = 4 M_ref`, 3750 substeps, float32, `rtol = 1e-6`, relative drift at
the end; the harness is not committed, but the two surviving realisations are both reachable through
the audit and the shipped one is the function under test):

| realisation | N = 48 | N = 128 |
| --- | --- | --- |
| v9 (pinned) | `-6.72e-05` | `+1.53e-05` |
| plain `A^-1 rhs` in the `V` inner product, `x0 = 0` | `-6.71e-05` | `-1.06e-04` |
| split-once: carry the RHS's constant mode, solve the complement | `-2.41e-05` | `-2.34e-06` |
| **exchange (shipped)** | **`-8.08e-07`** | **`-1.67e-06`** |
| exchange + Neumaier compensated seed | `-7.28e-07` | `-1.66e-06` |

The compensated seed is not shipped: it measures the same, and adding a compensated reduction to the
hot path for no measured benefit would be exactly the kind of unmotivated precision change the stage
forbids. It is recorded so the option stays visible.

* Fail-closed is unchanged: a non-finite or non-positive `<p, A p>_V` poisons the iterate,
  `converged = False` is returned, and the solve returns NaN — it can never silently advance.
* The custom VJP is the same `V`-inner-product solve on `cotangent / V` scaled back by `V`, because
  `A^T = V A V^-1`. It is checked three ways in `test_mass_precision.py`: against a dense adjoint,
  against a centred finite difference of `jax.grad` through the custom VJP (two step sizes; the map is
  *linear*, verified exactly, so the finite difference is limited by round-off only), and through the
  invariance of `M` under a mean-free perturbation of the right-hand side.

## Q. Gates, closure, CI, and the next stage

**CHNS closure, N = 128, `eps = 2 dx`, `dt = 4e-3`, `M = M_ref`, production float32, 50 000 steps**
(the first staged budget; the mass criterion decides whether extending is allowed):

| target | steps | angle | error | mass drift, max | mass drift, final | `converged` | stop reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 60° | 50 000 | 59.659° | 0.341° | **1.054e-3** | 1.054e-3 | false | budget exhausted |
| 90° | 50 000 | 89.402° | 0.598° | **4.783e-4** | 4.764e-4 | false | budget exhausted |
| 120° | 50 000 | 118.847° | 1.153° | **2.438e-4** | 1.937e-4 | false | budget exhausted |
| 150° | 50 000 | 147.416° | 2.584° | **3.119e-4** | 3.119e-4 | false | budget exhausted |

For contrast, the same 120° run under the split-once (pre-correction) contract-10 build measured
2.497e-3 — a 10x worse mass drift at the same angle (118.850° vs 118.847°), i.e. the correction form
changed the mass bookkeeping and left the physics alone. Under contract v9 the same four runs
measured 1.229e-4 (60°), 2.944e-4 (90°), 2.497e-3 (120° → split form), i.e. the ordering of the
failures is reproduced.

**W-CONTACT-ANGLE is NOT closed.** The angle conditions are met or nearly met (errors 0.34°, 0.60°,
1.15°, 2.58° against the 3° requirement at 90° and the 5°/10° MAE/max bounds), the drift is clean at
90°, 120° and 150°, but (i) the mass-drift gate `<= 1e-3` fails at 60° by 5 % (1.054e-3), and (ii) no
target reached `converged = true` inside 50 000 steps. The staged-horizon rule therefore stops the
claim here for 60° (extending to 100k/150k/200k is licensed only when the drift is clean at the
shorter horizon); 90°, 120° and 150° are drift-clean and *still moving* at the budget, so their
horizon extension is licensed and is the next measurement, but none is `converged`, so none can
close the gate. The repair changed the mass regime by 10x at 120° (2.497e-3 → 2.438e-4 max, 1.937e-4
final) and left every angle within 0.05° of its split-form value.

**The mass gate at the quick configuration** (N = 48, 150°, `M = 4 M_ref`, 2500 steps, the harness of
`test_cutcell_transport._quick_relaxation`) is met with three orders of magnitude to spare, and the
drift series is no longer a ramp:

```
v9      |drift| samples: 3.5e-06 ... 1.209e-04   (monotone: a bias)
shipped              : 0, 1.1e-07, 4.6e-08, 9.5e-08, ..., 8.5e-07, 7.2e-07, 6.1e-07, ..., 1.05e-07
```

**Physics is unchanged** (the repair moves the mass bookkeeping, not the model): the CH-only
equilibrium angle at 150° is 100.6209° (v9: 100.6220°) and the two sub-cell offsets spread
1.7002° (v9: 1.7027°) against the `<= 2°` requirement.

**Next stage, from the evidence (not preselected).** The residual is now a float32 *rounding* limit of
the solve at production `rtol`, and it is the only thing between this PR and the closure gates:

1. **`P-KRYLOV-ROUNDOFF`** (new): the v10 f32 residual is `-0.0173 E_round` per substep at N = 128
   `rtol = 1e-6` — 1.2e-3 over 150 000 substeps, i.e. exactly the size of the measured 60° drift
   (1.054e-3, the only failure margin left). It does not exist in float64 (`2.31e-16`). The work is to
   decide the arithmetic — mixed-precision seed/accumulators or a compensated recurrence — against
   the closure cases, with the float64 run as the attainable bound. Note the 60° run is the *gentle*
   case (least interface motion), so this is a small residual rather than a growing instability.
2. `N-DT`: the closure failure is in the most dynamic cases (120°), so the ablation in section M can
   separate "how far the interface moves per substep" from "the solve's arithmetic".

Neither `P-VARDENS-PROJ`, `P-CAP-RHO`, `P-VARVISC`, `I-CONTACT-GAP` nor `BC-Y-PERIODIC` is
recommended: the ledger localises the residual to the phase solve, not to any of them.

**CI / lint.** Recorded baseline lint debt at `9a3f50c` (unchanged by this PR, **not** remediated
here): `ruff check` 39 errors repo-wide, `ruff format --check` 9 files would reformat (166 ok),
`isort --check-only` 2 files (`production/cutcell_geometry_audit.py`,
`production/cutcell_phase_transport_audit.py`). On the eleven files this PR touches the `ruff check` count is
**8 before and 8 after** — the identical eight pre-existing `E501`/`F841`/`E501` findings, with zero
findings in the three new/modified files (`mass_mode.py`, `production/mass_precision_audit.py`,
`test_mass_precision.py`); `ruff format --check` flags the same three files before and after
(`phasefield.py`, `production/cutcell_alignment_audit.py`, `test_cutcell_transport.py`) and the new
files are clean; `isort` adds no new error.
`codespell` is not installed in this sandbox, so it was **not run**; the new files were proofread by
hand and the term "cut cell" is used deliberately. Tests run green in this tree after the change: `test_mass_precision.py` **16/16**,
`test_cutcell_transport.py` **26/26** (including its two mass-drift gates and the CI direction gate),
`test_dataset_contract.py` **12/12**, `test_nonneutral_wetting_audit.py` **26/26**,
`test_production_validation.py -k contract` **15/15**, `test_wall_measure.py` **43 passed** plus the
two end-to-end profile tests that a contract-value update blocked before the fix and that pass
individually afterwards. One pre-existing assertion was *strengthened*: `test_ch_only_mass_conservation`
used to require monotonicity in the CG tolerance (`tight_drift <= production_drift`), which is a
property of the v9 Krylov truncation and is falsified by the audit's non-monotone drift matrix after
the repair; it now bounds both rows (3.3e-9 at rtol 1e-6, 2.5e-7 at 1e-8) by 1e-6, a 100x tighter
gate than the old 1e-4. **No "all CI passed" claim is made**: the full matrix did not complete inside
this sandbox (2 CPUs, ~3 GB, long evidence runs in parallel); `test_two_phase.py` and the full
`test_wall_measure.py` run did not finish, the `baseline` audit profile (N = 64/192 scaling, the
2000-step float64 matrix) and the staged horizon extensions are left to the next stage's compute.

## R. Artifacts

| file | content |
| --- | --- |
| `evidence/mass_precision/mass_precision_report.json` | machine-readable ledger, matrix, verdicts |
| `evidence/mass_precision/mass_precision_report.md` | the audit's own Markdown rendering |
| `evidence/mass_precision/manifest.json` | stage, profile, contract, verdict, SHA-256 of both |
| `evidence/mass_precision/closure.json` | the 50 000-step CHNS runs of section Q (all four rows in the shipped-form run; the 150° row is appended when the staged run lands) |
| `production/mass_precision_audit.py` | the audit (`--profile quick|forensic|baseline`) |
| `.github/workflows/python-ci.yml` | runs the new test file and the quick audit profile as CI steps |
| `test_mass_precision.py` | the §59–63 contract tests |

```
JAX_ENABLE_X64=1 python -m production.mass_precision_audit --profile quick    # ~40 s, CI
JAX_ENABLE_X64=1 python -m production.mass_precision_audit --profile forensic # default, hours
JAX_ENABLE_X64=1 python -m production.mass_precision_audit --profile baseline # widest sweeps

XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" JAX_ENABLE_X64=1 \
    python -m pytest test_mass_precision.py -q -p no:randomly
```

The committed report is the **forensic** profile (21 checks, `passed: true`, recorded in
`manifest.json` with the SHA-256 of both report files); `--profile quick` runs the same checks in
~40 s for CI, and `--profile baseline` adds the N = 64/192 scaling and the 2000-step float64 matrix.
The forensic audit itself **passes after the repair** — the only checks that failed before it were
the ones pinned to the v9 mechanism (`pinned_v9_first_stage_above_roundoff_floor`,
`pinned_v9_inverse_transform_is_systematic`), and those now anchor on the N = 128 grid where the cut
cells actually exist, instead of on the N = 48 control grid where the transform is bit-exact. The evidence was produced on a 2-CPU sandbox, so the
long profiles need `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2` and
patience; the closure run took 6 h of wall clock under that flag and concurrent load.
