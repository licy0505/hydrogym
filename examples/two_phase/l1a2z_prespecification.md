# L1A-2z prespecification — own-base attribution and a unique-support measurement

Stage: **L1A-2z** (`l1a2z_v1`) · branch `arena/02b6c71b-hydrogym` · measured from the published L1A-2y
evidence head `cd20a7838fb616f3857092c580f3c24802c6115a` · verified main
`09e2993b8da478d79e768054194201af57121784` · PR #24 stays Draft.

This document is the frozen measurement protocol. It was written and committed, together with the
implementation and the adversarial tests it names, **before** any target-window measurement of this stage
was looked at. The commit that introduces this file is the freeze boundary: the stage refuses to measure if
the module or the test file differs from the hashes recorded below, or if the measurement code changed after
that commit.

The stage corrects a measurement. It does not touch physics, the trajectory, the acceptance criteria or the
prior evidence, and it claims no physical mechanism in any outcome.

## 1. What was wrong (audited, not assumed)

L1A-2y reported `EARLY_WINDOW_ATTRIBUTION_UNRESOLVED` and withheld every dominance claim for two
independent reasons. Both are re-derived here from its sealed evidence.

**Failure F1 — the primary attribution rule could not close.** The frozen L1A-2x capture of one substep is
`S1 = phi + d_adv`, `S2 = phi_hat = S1 + d_ch`, `S3 = phi_new = S2 + d_imp`, with
`d_adv = dt*phi_rhs`, `d_ch = -dt*div_ch` and `d_imp` the implicit Cahn–Hilliard solve increment, so
`d_adv + d_ch + d_imp = phi_new - phi` exactly. L1A-2y's primary rule applied each ingredient to the *same*
substep base and then chained each ingredient's own increments across the whole window. Such a chain
telescopes to a difference between crossings of two *partial* fields of different substeps. It is not the
ingredient's contribution to the actual motion, and its legs cannot sum to the measured total: the sealed
residual is `R_attribution_closure = -0.12231966 dx`, 12.2 times the frozen `0.01 dx` band.

**Failure F2 — the frozen global contour match was non-unique on all 40 macro steps.** The frozen L1A-2x
diagnostic minimises the wall distance over every reconstructed `phi = 0.5` crossing and reports
`NON_UNIQUE` whenever a second candidate lies within `CONTOUR_UNIQUENESS_MARGIN_DX = 0.01 dx` of the
minimum, excluding candidates closer than `0.5 dx` to the best one as "not distinct". At `t = 0` the two
lowest-gap crossings are the mirror pair `x = 2.984375` and `x = 3.015625` at the *same* height
`y = 0.7345619198479572`: the frozen state is exactly mirror-symmetric there and the frozen SDF is exactly
linear in the normal coordinate, so the two wall distances agree to machine precision, and the frozen readout
traces both crossings on **one** open contour arc. The pair is separated by exactly `1.0 dx`, which is larger
than the `0.5 dx` non-distinctness exclusion, so the rule counts the plateau partner as a competing candidate
with zero margin: `margin_over_dx` between `2.26e-11` and `1.80e-9` on the 40 macro steps. The degeneracy is
structural in the point question, and it is a same-branch multiplicity in the branch question.

Both failures are measurement failures. The frozen verdict stands unchanged and is not reinterpreted.

## 2. The corrected rule

For every frozen substep `k` and every admissible support, the frozen observable `Q` (the tracked
`phi = 0.5` branch root on the fixed wall-normal ray, read by the frozen `l1a2y_contour_branch_readout`) is
evaluated on the *actual* consecutive states:

```
D_adv = sum_k [ Q(S1_k) - Q(S0_k) ]
D_ch  = sum_k [ Q(S2_k) - Q(S1_k) ]
D_imp = sum_k [ Q(S3_k) - Q(S2_k) ]
D_full = Q(S3_K) - Q(S0_1)
```

* Each contribution uses its own *pre-operator* state as the base: the base of `D_ch` is `S1`, the base of
  `D_imp` is `S2`. It is never the substep base `S0` and never a field rebuilt by summing increments.
* The executed operator order is the implementation's: advection and the explicit chemical-potential flux
  are folded into `phi_hat`, then the implicit solve. The order is not permuted.
* The telescoping identity `D_adv + D_ch + D_imp = D_full` is an algebraic consequence of the definition. It
  is reported with its floating-point residual as a *closure* check. It is not evidence of a mechanism, and
  no verdict of this stage depends on the fact that it holds.
* `R_sequential_telescoping_total`, `R_sequential_telescoping_max_substep` and the marginal/sequential
  interaction term are judged against the frozen `0.01 dx` band, unchanged.
* The two Cahn–Hilliard halves are reported as **one** update with two increments. No verdict is attached to
  either half separately and they are never presented as two physical mechanisms.
* The L1A-2y primary rule is re-reported from its sealed payload as a historical comparator. It is never
  recomputed into a success and its files are never modified.

Every displacement is a raw displacement in `dx` along the frozen wall normal, positive toward the solid.
Nothing is normalised by the quantity it measures and no post hoc band is introduced.

## 3. The support-selection algorithm (prespecified)

**Candidates.** Every reconstructed `phi = 0.5` crossing of the frozen periodic/linear reconstruction, with
its grid edge and its bilinear SDF wall distance. The enumeration must reproduce
`impact_initialization_compatibility_audit._threshold_contour_edge_crossings` point for point; a mismatch
blocks the measurement.

**Components.** The connected components of the same reconstruction, built from the *same* per-cell linear
segments the frozen readout uses (including the frozen asymptotic-decider pairing at a four-crossing saddle
cell) by union-find over crossing edges. The construction is validated cell by cell against
`l1a2y_contour_branch_readout.cell_segments`. Union-find labels are reported for information only: they are
not identities.

**Minimum and competition.** `g*` is the minimum wall distance over the candidates. A candidate is *tied* at
the minimum if its distance is within `TIE_TOLERANCE_DX = 1e-6 dx` of `g*`, and it *competes* if it is within
`COMPETITION_TOLERANCE_DX = 1e-2 dx` of `g*`.

**Uniqueness rule.** The measurement of `Q` is admissible at one state only if

1. every competing candidate lies on **one** connected component;
2. no grid node value equals the threshold exactly (`phi == 0.5`), which would make the pairing ambiguous;
3. the frozen enumeration matches; and
4. every candidate count and distance is reproduced under a permutation of the enumeration.

**Support selection.** All tied minimum-gap candidates (both `t = 0` supports, in this window) are pinned
independently from the `t = 0` state alone, and the whole attribution is measured on **every** one of them.
The primary reporting support is the tied candidate with the smallest `x`: that is a deterministic
tie-break for reporting, it is explicitly *not* evidence of uniqueness, and no region is shrunk to leave one
candidate. The difference between the tied supports' margins is reported as the support-tie term of the
comparison error. The frozen minimum-gap rule is reproduced exactly from the frozen module and is retained
verbatim as a historical comparator; its `NON_UNIQUE` verdict answers a point question that this stage
reports but does not gate on.

**Local tie separation.** The tied pair must also be one *local* piece of the reconstruction: both
candidates are projected onto the segments of the frozen reconstruction inside the pinned
`ARC_WINDOW_RADIUS_DX = 3.0 dx` window, and the shortest path between their nearest segments through the
frozen segment graph is measured. No path inside the window means the tie is a multiplicity across separated
branches and the gate blocks. The path length and the separation to the nearest non-competing candidate are
reported as the quantitative separation of the tie.

**Branch identity.** Identity is geometric, never a union-find label: between two consecutive measured states
the tracked root's component must have every crossing within `BRANCH_IDENTITY_TOLERANCE_DX = 0.375 dx` of
some crossing of its component in the previous state, and the other way round. The tolerance is derived from
the frozen readout rule (`MAX_SUBSTEP_DISPLACEMENT_OVER_EPS * eps_over_dx = 0.25 * 1.5 dx`) and never from
the data. Containment in a larger component is a recorded *merge*, containment of a smaller one a recorded
*split*; both are ordinary events. A disjoint relation is a reassignment, an ambiguous pair blocks, and both deny
the verdict.

**Every intermediate state.** The rule is applied to every required intermediate state `S1, S2, S3` of every
substep, not to macro boundaries only. A support switch recorded by the frozen readout (the reconstructed
crossing moving to the next grid segment) is an explicit recorded event; it is never resolved silently and it
is never used to justify shrinking the search region.

**No uniqueness is manufactured.** If no support satisfies the rule at every required state, the stage reports
the limitation and keeps `EARLY_WINDOW_ATTRIBUTION_UNRESOLVED`.

## 4. Measurement conventions (all four prespecified, all reported)

1. `SEQUENTIAL_OWN_BASE` — the corrected rule of §2. Primary.
2. `MARGINAL_OWN_BASE` — each increment applied alone to its own substep base (the L1A-2y `RULE_B`
   anchoring), reported as an alternative convention, never as the primary rule.
3. `LINEARISED_POINT_T0` — the frozen L1A-2x linearisation coefficient rebuilt from the same frozen `t = 0`
   normal, gradient and support point, applied to the per-support field increments sampled at the support
   point.
4. `LINEARISED_SUPPORT_CELL_T0` — the same coefficient applied to the field increments at the support cell
   centre.

The comparison error is the maximum of the prespecified floor (`1e-3 dx`), the spread between (1) and the
other conventions, the support-tie spread and the root-matching noise. The dominance floor is
`3 x` the comparison error, carried over unchanged.

## 5. Acceptance criteria (frozen, carried over unless stated)

| quantity | value | carried over |
|---|---|---|
| attribution closure band | `0.01 dx` | yes (L1A-2y) |
| total-displacement closure band | `0.01 dx` | yes |
| dominance factor | `3.0` | yes |
| walk-noise floor | `1e-3 dx` | yes |
| capture-equivalence maxima | `1e-6` (phi), `1e-4 dx` (gap) | yes |
| identical-window tolerance | `0.05 dx` | yes |
| contact gap threshold (stop guard only) | `1.5 dx` | yes |
| window | one case, N=192, `dt = 0.002`, ≤ 40 macro steps, 120 substeps, `t <= 0.08` | yes |
| tie tolerance | `1e-6 dx` | new |
| competition tolerance | `1e-2 dx` | new |
| branch-identity tolerance | `0.375 dx` | new |
| tie-arc tolerance | `1e-6 dx` | new |
| exact-threshold rule | `phi == 0.5` exactly is a blocking degeneracy | new |

## 6. Failure conditions (each one withholds the verdict; none can grant it)

1. A readout status other than `OK` at any required state.
2. Any required intermediate state failing the uniqueness rule (multi-component competition, an exact
   threshold node, an enumeration mismatch, a permutation-dependent answer).
3. A branch reassignment or an ambiguous branch relation.
4. A support-tie spread larger than the closure band.
5. `|R_sequential_telescoping_total|`, `|R_sequential_telescoping_max_substep|` or
   `|R_marginal_interaction|` above `0.01 dx`.
6. A capture-equivalence or conservation/convergence failure.
7. A trajectory that does not reproduce the sealed L1A-2y window inside the frozen tolerance: that is a
   separate regression, not a resolved attribution.
8. A protocol, provenance, source-integrity, shallow-clone or unexpected-dirty-worktree condition, or a
   prespecification-hash mismatch.

A failed condition is reported, never repaired in place, and never reclassified as a pass.

## 7. Adversarial and falsification tests (frozen with this document)

1. exact telescoping identity on a synthetic operator sequence, with a discriminating wrong sum;
2. a wrong base or wrong pairing is detected;
3. a mixed-family sum is inadmissible even when it lands inside the band (and it is shown to be able to);
4. operator order is recorded and the swap is measured against the executed order;
5. nonlinear contour response is measured, not assumed;
6. the frozen closure gate can fail (the sealed L1A-2y residual misses the same band);
7. a rule that improves the number by rescaling one leg to the total is rejected;
8. the frozen L1A-2x diagnostic is retained and not redefined;
9. the prior evidence is unmodified by the run;
10. candidate permutation and enumeration order do not move the answer;
11. the tied second candidate lies on the tracked arc;
12. a broken or absent observable is caught by the gate;
13. two distinct branches near the wall are reported non-unique;
14. an exact threshold grid node is flagged;
15. the recorded support switch stays on one branch;
16. the tie-break is never used as evidence of uniqueness;
17. the branch-identity relation table is discriminating (persistent / merged / split / reassigned);
18. the identical-window tolerance can reject;
19. evidence mutation is rejected by the independent verifier.

Every test must be able to fail. A discovered implementation error requires an explicit record and a new
auditable prespecification cycle; the tests, their thresholds and this document are never changed after a
target-window result has been inspected.

## 8. Verdict discipline

Three questions are kept strictly apart.

* **A — measurement validity.** Is the tracked observable uniquely and reproducibly attached to one interface
  branch at every required state? This stage answers it.
* **B — mathematical attribution.** Does the corrected own-base decomposition close inside the frozen
  `0.01 dx` band? This stage answers it.
* **C — physical mechanism.** Does the experiment establish *why* the interface moved? This stage never
  answers it. There is one trajectory and no control arm, so no operator can be isolated; and the telescoping
  identity is algebra, so a closed decomposition is not a causal statement.

Passing A and B yields `EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED`, which is explicitly a statement about
a prespecified measurement convention, not a mechanism. Failing any of A, B or a required gate yields
`EARLY_WINDOW_ATTRIBUTION_UNRESOLVED`, which is an acceptable result.

## 9. Frozen hashes

<!-- L1A2Z-FROZEN-HASHES
protocol_version l1a2z_v1
module_sha256 1c1815991e929860d377855a2d4c46d890040da0d0ee1733ff9f675784f5e858
test_sha256 0dc6945baf841bc27a506e3c4f779da11fcb81da4f00c8e322ee3d7fbd9f45cc
sync_count 1
-->
