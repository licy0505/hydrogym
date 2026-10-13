# L1A-3a Phase A — pre-control observable audit (blocked)

**Result:** `EARLY_WINDOW_CONTROL_INCONCLUSIVE` — Phase A stopped the intervention before protocol freeze or any control run.

**Scope:** this is a Phase-A blocker record only. It is not an L1A-3a prespecification, solver-admissibility finding, control measurement, or physical-mechanism result. The proposed `M=0` trajectory was **not** frozen or executed. No L1A-2x/2y/2z evidence was changed.

## 1. Independent verification of the published L1A-2z baseline

- Published branch head and `origin/arena/02b6c71b-hydrogym`: `d774e38209d7bd1c13810aaa3c2b6434209291f9`; repository is non-shallow and was clean before this audit record was added.
- Main: `09e2993b8da478d79e768054194201af57121784`.
- PR #24 remains **OPEN / Draft**, based on `main`.
- L1A-2z evidence verification returned `verified=true`, `problems=[]`, 21 bound evidence files and all 8 recorded source hashes verified against the worktree.
- `prespecification_freeze()` holds: boundary `7a9fdf0eea48f479edd05de2d4e5cafc501a34f2`; declared/observed hash-block syncs `2/2`; module hash `1c1815991e929860d377855a2d4c46d890040da0d0ee1733ff9f675784f5e858`; focused-test hash `0f329a0fd0b62f9de411e5f3de544b53e60378e964887b49878cb99037ec4a70`.
- Focused verification command: `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. /home/user/hydrogym/.venv/bin/python -m pytest -q tests/test_l1a2z_sequential_ownbase_attribution_audit.py` → **70 passed in 3.21s**.
- Frozen L1A-2z verdict remains `EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED`; L1A-2y remains `EARLY_WINDOW_ATTRIBUTION_UNRESOLVED`.

## 2. Exact observable audited

The frozen constant is `RULE_S_SEQUENTIAL_OWN_BASE_ACTUAL_INTERMEDIATE_STATES`. For each substep, the recorded phase states are `S0=phi`, `S1=S0+d_adv`, `S2=phi_hat=S1+d_ch`, and `S3=phi_new=S2+d_imp`; the sequential legs are `Q(S1)-Q(S0)`, `Q(S2)-Q(S1)`, and `Q(S3)-Q(S2)`. This is actual-state algebraic attribution, not a causal or physical-mechanism claim.

Here `Q` is **not** the global minimum-gap value. It is the frozen `phi=0.5` root on a fixed wall-normal ray whose column and origin are pinned through one selected `t=0` support point. The source contract is visible in `l1a2z_sequential_ownbase_attribution_audit.py` (`measure_root`, `support_design`, `run_window`, `attribution`) and `l1a2y_contour_branch_readout.py` (`branch_definition`, `branch_root`). The latter computes a column from the support point's x-coordinate and follows the root on that fixed column.

## 3. Three different uniqueness questions

### Connected component — unique under the frozen branch rule

At `t=0`, the two competing minimum-gap crossings both belong to component 76 of the same reconstructed contour. The audit has 176 crossings in total, one component, one competing component, no exact-threshold nodes, and 360/360 intermediate states classified `UNIQUE_BRANCH`. The branch-identity record has 716 persistent relations, two recorded split relations, two recorded merge relations, and no disjoint reassignment. This is evidence about component identity only; it is **not** unique geometric support.

### Scalar minimum-gap value — defined

The scalar `g*=min wall-SDF distance` is well-defined even when the argmin is a set. At `t=0`, both points have the same recorded value, `2.705981482818345 dx`. That scalar does not require selecting one of the minimizers. But using `g*` as the intervention response would replace the L1A-2z fixed-ray root-displacement observable with a different observable; this audit does not make that substitution.

### Spatial argmin — non-unique

The two co-minimizers are:

- `(x,y)=(2.984375, 0.7345619198479572)`, `2.705981482818345 dx`, component 76, edge `[1,95,23]`;
- `(x,y)=(3.015625, 0.7345619198479572)`, `2.705981482818345 dx`, component 76, edge `[1,96,23]`.

They lie on the same open contour arc, separated by `1.0 dx` along that arc. The nearest non-competing candidate is at `(2.953125, 0.7360437254866143)`, with a gap increase of `0.04741778043702993 dx`. The frozen global L1A-2x diagnostic is retained unchanged: `NON_UNIQUE` at all 360 L1A-2z intermediate states (margin range `0` to `1.8504060506074893e-9 dx`). This is a spatial-argmin non-uniqueness, not a claim that the connected component is ambiguous.

## 4. Why the L1A-3a scalar response is blocked

L1A-2z does measure both co-minimal supports, but its reporting primary is explicitly selected as the tied support with the **smallest x**. The source calls this a deterministic reporting tie-break and correctly says it is not evidence of uniqueness. The primary response is then read along that support's fixed ray. The second support defines a different fixed ray in column 96 rather than column 95.

In the sealed original baseline, the support-specific full-window displacements are nearly equal:

| support | `Q(0)` root (`dx` coordinate) | `Q(0.08)` root (`dx` coordinate) | full displacement (raw `dx`) |
|---|---:|---:|---:|
| `PRIMARY` (smaller x) | 23.50598143513463 | 23.443332753241584 | +0.06264868189304451 |
| `TIE_1` | 23.50598143513463 | 23.443332751438383 | +0.06264868369624566 |

Their displacement spread is `1.803201143957267e-9 dx` in this baseline. This is a baseline observation, not proof that the two ray observables remain identical under a mobility intervention. The existing L1A-2z primary scalar is still tied to the smaller-x support; the small observed baseline spread does not remove the spatial choice or establish support-invariance for the control.

Therefore, under the L1A-3a requirement to retain the same L1A-2z `Q` and not resolve a spatial degeneracy by an arbitrary tie-break, the single scalar primary response is not established reproducibly for the proposed intervention. A vector of both support-specific responses, or a branch-level minimum-gap response, may be considered only under a new, explicitly approved protocol; neither is silently substituted here.

## 5. Phases not entered

- Proposed intervention only: `M_baseline=0.002`, `M_control=0.0`, with the requested case/window/configuration. No L1A-3a prespecification commit or source-hash boundary was created because the Phase-A gate failed first.
- The actual CH implementation and the discrete zero-mobility limit were **not assessed**. No claim is made that `M=0` is admissible or inadmissible.
- No L1A-3a baseline/control paired comparison, control Q(t), paired `Delta_Q`, control conservation/convergence result, or instability result exists.
- Baseline-only values below are historical L1A-2z reference data, not L1A-3a diagnostics: mass `1.5500927792312642` initially and finally; drift `0`; worst substep mass residual `4.470435646983889e-16`; worst closed-face flux `0`; 120 CG solves, all converged in 18 iterations, worst relative residual `5.718904648233452e-07`; contact guard not triggered.
- Existing L1A-2z adversarial checks: `19` run, `0` failed. They are not L1A-3a falsification tests. No L1A-3a tests were frozen or run, and no control trajectory was run.
- No causal physical-mechanism conclusion is made.

## 6. Next action (not started; approval required)

Obtain fresh owner approval for an ambiguity-safe observable before any new target-window measurement: either (a) use the scalar minimum gap of the tracked component and explicitly acknowledge that it differs from L1A-2z's fixed-ray `Q`, or (b) retain both co-minimal fixed-ray observables as a support-indexed vector/set with frozen adjudication and no primary tie-break. Then freeze the selected definition, tests, gates, and hashes before any control run. No such design or run has started.

Machine-readable details and exact input hashes are in `phase_a_audit.json`; this directory's manifest seals the two report files. L1A-2x/2y/2z files were read-only throughout.
