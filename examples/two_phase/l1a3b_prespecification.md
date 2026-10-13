# L1A-3b — Tie-break-free support-indexed observable contract

**Stage:** L1A-3b  
**Protocol version:** `l1a3b_v1`  
**Authorized branch:** `arena/02b6c71b-hydrogym`  
**Baseline head:** `29dd981dd0b162af05ca9bba00aa90f7dc5e8fea`  
**Source L1A-2z head:** `d774e38209d7bd1c13810aaa3c2b6434209291f9`  
**Verified main:** `09e2993b8da478d79e768054194201af57121784`

This document freezes the **observable contract only**. It does not authorize a mobility intervention, a zero-mobility (`M=0`) control, a new solver trajectory, contact-stage experiment, or any physical-mechanism conclusion. L1A-2x/2y/2z and L1A-3a evidence, verdicts, code, solver behavior, physics thresholds, and initial conditions remain read-only.

## 1. Contract objective and scope

L1A-3a stopped before a control because its scalar `Q` selected one of two co-minimal support points by spatial order. The authorized resolution is the symmetric vector over **both** frozen supports. There is no primary ray, no dynamic support replacement, and no scalar reduction of the two components.

The observable describes tracked roots on two fixed rays. A future paired experiment may use this contract only after its own separate approval and prespecification. Numerical symmetry in the historical baseline is not evidence of support independence under an intervention.

## 2. Frozen ray geometry and identities

The sealed L1A-2z grid is `N=192`, `Lx=Ly=6`, `dx=dy=0.03125`. Both wall-normal rays point downward, `n_wall=(0,-1)`, toward the solid. Ray identity is bound solely to the frozen cell-centred column and its `x` coordinate:

| Immutable identity | Column index | `x` | Initial support `y` | Initial root `q(0)` (`dx`) |
|---|---:|---:|---:|---:|
| `RAY_95` | 95 | 2.984375 | 0.7345619198479572 | 23.50598143513463 |
| `RAY_96` | 96 | 3.015625 | 0.7345619198479572 | 23.50598143513463 |

The coordinate identity rule is `x=(column_index+0.5)*dx`. L1A-2z’s legacy support labels are mapped to `RAY_95` / `RAY_96` by column index only; those labels do not select or rank a ray. Both points are co-minimal on the same t=0 contour component in the sealed geometry. Later component-label integers are state-local and are not compared across time or independent trajectories.

## 3. Mathematical observable and sign convention

For each measured state `t`,

\[
Q(t)=[q_{95}(t),q_{96}(t)],
\]

where `q_j(t)` is the frozen linear `phi=0.5` root coordinate `y/dx` on the fixed vertical ray with identity `RAY_j`. The original signed coordinate convention is retained; because positive displacement is toward the solid on these downward rays,

\[
d_j(t;M)=q_j(0;M)-q_j(t;M).
\]

For a future paired comparison with identical initial-state hashes,

\[
E_j(t)=d_j(t;M_{control})-d_j(t;M_{baseline})
      =q_j(t;M_{baseline})-q_j(t;M_{control}).
\]

The implementation reports both `d_95,d_96` and `E_95,E_96`, checks the algebraic root-coordinate identity, and rejects unequal initial-state hashes. It never averages supports or chooses the larger/smaller component as the result.

## 4. Frozen reconstruction, enumeration, and branch correspondence

At every readout state the contract:

1. Enumerates **all** frozen linear `phi=0.5` crossings on each fixed ray, not just a preferred candidate.
2. Maps each ray crossing to the edge/component of the frozen global contour reconstruction. It independently checks crossing enumeration and cell-segment pairing against the frozen implementations.
3. Keeps every candidate and its ray identity, root coordinate, segment index, shared-node flag, and within-state contour-component association in the readout record.
4. Tracks each ray from its own preceding root and matches the two roots by immutable ray identity. The selected roots must lie on one current connected component.
5. Checks cross-state component correspondence geometrically with the frozen L1A-2z component-identity tolerance. Raw component-label integers are not correspondence criteria.
6. When a geometry bundle is provided, recomputes the original L1A-2x global minimum-gap diagnostic without using it to select either ray. A production-contract readout requires this diagnostic bundle; global `NON_UNIQUE` is preserved and is not used to substitute a support.

A segment-index change is a readout-grid event, not a ray-identity change. Branch split/merge relations are recorded as topology events. If either fixed-ray root loses geometric correspondence, becomes multiply admissible, or the supports cease to share the tracked current component, the paired readout is invalid rather than favorably reassigned.

## 5. Frozen validity thresholds and fail-closed states

The root and component limits below are inherited from the frozen L1A-2y/L1A-2z readout; none is outcome-tuned for L1A-3b.

| Gate | Frozen rule | Failure state |
|---|---|---|
| Contour threshold | `phi=0.5`, original linear edge interpolation and periodic contour rule | frozen enumeration or cell-reconstruction mismatch |
| Exact threshold nodes | No grid node may equal `0.5` in an evaluated state | `EXACT_THRESHOLD_NODE_DEGENERACY` |
| Root continuation window | Exactly one candidate within `±1.5 dx` of the preceding ray root | `NO_ROOT_IN_WINDOW` or `MULTIPLE_ADMISSIBLE_ROOTS` |
| Root movement | Selected root moves by at most `0.25 dx` per readout | `BRANCH_STEP_TOO_LARGE` |
| Root separation | A competing root closer than `0.5 dx` to the selected candidate is ambiguous | `ROOT_SEPARATION_BELOW_FROZEN_AMBIGUITY_LIMIT` |
| Component correspondence | Frozen geometric component tolerance `0.375 dx` | `CORRESPONDENCE_LOST` / `GEOMETRIC_BRANCH_CORRESPONDENCE_LOST` |
| Two-ray membership | Both selected roots must lie on one current component | `THE_TWO_FIXED_RAYS_NO_LONGER_BELONG_TO_ONE_CURRENT_COMPONENT` |
| Global reconstruction cross-check | Existing frozen crossing/segment comparison tolerance `1e-12` in coordinates | `FROZEN_GLOBAL_CONTOUR_ENUMERATION_MISMATCH` or `FROZEN_CELL_RECONSTRUCTION_MISMATCH` |

Enumeration order, ray input order, or proximity ranking never resolves a multiple/ambiguous case. A state or paired effect with a validity failure is explicitly invalid/inconclusive; no one-ray or scalar fallback is permitted. The original L1A-2x minimum-gap diagnostic remains a distinct global-contour diagnostic, not a root-validity pass/fail replacement.

## 6. Symmetric response reporting

For any valid future time point, report at least:

- `Q_dx={RAY_95:q95,RAY_96:q96}`;
- `d_dx={RAY_95:d95,RAY_96:d96}` and, only for a valid paired comparison, `E_dx={RAY_95:E95,RAY_96:E96}`;
- the raw signed range `[min(E95,E96), max(E95,E96)]`, `E95-E96`, and `abs(E95-E96)`;
- each component’s sign and whether signs are discordant/opposite;
- per-ray root-reconstruction arithmetic differences, any justified total numerical uncertainty, and all validity failures;
- initial-state hash equality and geometric correspondence results for the two trajectories.

A scalar mean or selection of either support is not a permissible summary. If either trajectory is invalid, return its failure inventory and no partial effect vector.

## 7. Numerical uncertainty and support-agreement policy

For each future root, the readout can recompute the linear intersection from the exact stored binary64 edge values using high-precision `Decimal` and report the raw arithmetic difference in `dx`. This is an interpolation-arithmetic comparison for the same stored field; it is **not** a solver-state, discretization, or physical uncertainty bound.

The sealed baseline has a maximum two-ray root spread of `1.8504060506074893e-9 dx` over the 360 stored intermediate states and an endpoint spread of `1.803201143957267e-9 dx`. These are measured baseline variations, not a noise floor or tolerance. The historical eager/public-step `phi`/gap checks and the zero root difference in the identical-window check are also not calibrated as per-ray intervention-response uncertainty bounds.

No defensible binary threshold for support invariance is established here. The support-agreement conclusion is therefore **UNRESOLVED**; a future study must report raw signed effects and support disagreement without using baseline agreement to infer intervention invariance. The two-vector observable remains valid even if its components differ.

## 8. Historical L1A-2z Phase-C comparison

Only the sealed L1A-2z records are used; no solver replay or new physical trajectory is performed. The available record check covers all 360 stored S1/S2/S3 states for both rays, all 120 substeps, and all available carried/re-measured q0 and direct q3 readbacks:

- q1/q2/q3 readout statuses: `720 OK`;
- q0 re-measure and direct-q3 statuses: `480 OK`;
- L1A-2z tracked-branch status: `360 UNIQUE_BRANCH`;
- original L1A-2x global contour status: `360 NON_UNIQUE`, retained unchanged;
- both ray roots remain on one current component in every stored state;
- component relations across both supports: `716 PERSISTENT`, `2 FRAGMENTED_OR_SPLIT`, `2 GREW_OR_MERGED`; no stored reassignment;
- competition-set sizes: `333` states with size 2 and `27` with size 4;
- segment-index changes are recorded independently on both fixed rays (macro 25, substep 73 S2/S3 and substep 74 S1); no ray identity changes.

| Identity | Initial `q` (`dx`) | Final `q` (`dx`) | `d(T)=q(0)-q(T)` (`dx`) |
|---|---:|---:|---:|
| `RAY_95` | 23.50598143513463 | 23.443332753241584 | +0.06264868189304451 |
| `RAY_96` | 23.50598143513463 | 23.443332751438383 | +0.06264868369624566 |

The sealed ledger does not contain intermediate phase arrays or complete per-ray crossing candidate lists. L1A-3b verifies every stored root/status/segment/component/readback and uniqueness row, but cannot independently reconstruct omitted candidate lists without leaving the authorized stage scope. This limitation is reported, not filled by a solver rerun.

## 9. Adversarial and falsification test inventory

The implementation and focused suite cover 17 named adversarial classes: two co-minimal supports on one branch; genuinely competing branches; candidate enumeration permutation; ray-order permutation; selective one-support rejection; distinct/equal/opposite support effects; missing roots; multiple roots; branch fragmentation/correspondence loss; historical segment switches; near-limit root perturbations; displacement-sign convention; swapped ray identity; misleading scalar means; and evidence payload mutation detection. Negative cases assert explicit invalidity or manifest rejection. The independent full-manifest test also mutates a sealed historical vector and must fail hash verification.

Focused stage tests additionally verify frozen ray coordinates, every available historical state/readback, endpoint units and signs, unchanged global non-uniqueness, exact-node rejection, initial-state hash mismatch, Decimal interpolation comparison, and the production entry’s requirement to retain the global diagnostic.

## 10. Evidence verification and immutable source inputs

Required verification before a future control measurement:

1. Verify the original L1A-2z manifest, all bound files, and all recorded source hashes with its frozen verifier.
2. Verify the two-file L1A-3a blocker manifest, its `EARLY_WINDOW_CONTROL_INCONCLUSIVE` verdict, and `control_trajectory_run=false`.
3. Verify this L1A-3b manifest’s hash for every required evidence file and source, including the two fixed-ray module/test, the prespecification, and prior-stage manifests.
4. Verify the frozen hash block against the worktree; require the boundary commit to be an ancestor and the declared `sync_count` to equal the reviewed post-boundary synchronization count.
5. Verify the authorized branch, verified `origin/main`, non-shallow history, stage-owned changes only, and explicit no-control/no-solver-edit invariants.
6. Keep the machine-readable verdict and report bound by the same evidence manifest; the report must contain all 13 required items.

No prior-stage evidence, solver, initialization, physics threshold, quality baseline, or verdict may be rewritten to satisfy this contract.

## 11. Verdict rule

Return `SUPPORT_INDEXED_OBSERVABLE_FROZEN` only if the fixed geometry, readout implementation, adversarial gates, all available historical-record checks, prior evidence manifests, frozen hash boundary, evidence seal, and required local test/quality observations pass. Otherwise return `SUPPORT_INDEXED_OBSERVABLE_UNRESOLVED` and name every exact blocker.

An unresolved support-agreement conclusion is not converted to agreement and does not justify a scalar result. A frozen verdict certifies only the two-ray observable contract; it is not a mobility-sensitivity, support-invariance, or physical-mechanism verdict.

## 12. Next action and stop boundary

Next action is **NOT STARTED**. A separate owner approval and a new prespecified protocol are required before any paired intervention, control trajectory, or physical-mechanism study. This authorization does not permit `M=0` or any new trajectory.

## Implementation corrections included before the freeze boundary

The frozen module/test already include two audit-only corrections before this prespecification boundary: normalize pytest failure identifiers by the optional leading `tests/` prefix only for comparison against the immutable L1A-2z failure inventory, while preserving full observed identifiers in the report; and render machine-readable hashes, suite IDs, and publication fields as indented JSON blocks so Markdown wrapping cannot split identifiers and create false codespell findings. Focused tests cover the path forms, and the final evidence/report verifier checks all 13 report items. These do not change ray geometry, scientific thresholds, equations, historical evidence, or verdict rules. The hash block seals these corrected files at the boundary; the post-boundary synchronization count is zero.

<!-- L1A3B-FROZEN-HASHES
protocol_version l1a3b_v1
module_sha256 64ea61b912630cd35b8ed3dd1bfcf51a733b12373325340e4983b9814625a104
test_sha256 932ce4f2d5fc7ab37d05eb71e962833a9e75708786a728e1d955db2c9993b686
sync_count 0
-->
