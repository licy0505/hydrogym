# L1A-2i — phase-state storage precision and the drift-clean CHNS closure: final report

Stage L1A-2i, on `main` at `20b3432` (the L1A-2h merge, contract **10**,
`IMPLICIT_PHASE_SOLVER = weighted_spd_nullspace_preserving_v1`). Everything below is measured in this
tree with the commands in section Q.

**Verdict in one line: the frozen diagnosis is confirmed and quantified — the residual contract-v10
phase-mass drift is manufactured by two float32 terms, a dominant *phase-state storage bias* and a
smaller *float32 Krylov mass defect*; the only measured state representation that removes both is
phase-only float64 (A1), which is selected as the least invasive passing candidate; the contract is
not bumped in this stage, and A1's measured cost is above the preferred envelope on CPU with the GPU
cost unverified, so production readiness is reported as NOT READY rather than accepted silently.**

---

## A. Scope and authority

The stage studies exactly two questions:

1. whether the residual contract-v10 phase-mass drift is a float32 **phase-state storage** systematic
   bias (the frozen L1A-2h diagnosis) and how much of it a change of the *stored state* removes;
2. which state representation — A1 phase-only float64, B1 compensated local accumulation, C1 local
   residual feedback (plus the A2 mixed-precision variant and the F full-float64 reference measured
   here) — is the least invasive one that clears every conservation, physics, restart, AD and
   cost-report gate.

Everything else is frozen and verified untouched: the Krylov arithmetic (contract 10's
`_cg_solve_volume_weighted` is used *verbatim* by every candidate through
`production_correction`), the Young wall energy, the cut-cell geometry and wall measure, the mass
matrix `M`, `dt`, Brinkman penalisation, the gas properties, the pressure projection, the capillary
density denominator and the variable-viscosity stress. No global mass target, no post-step global
correction, no cross-cell redistribution and no angle calibration exists anywhere in the new module
(tests in section M enforce this on the source, not by assertion of intent).

## B. Base, contract, and what changed in the tree

| item | value |
| --- | --- |
| base commit | `20b3432` (`origin/main`, L1A-2h merged) |
| solver contract after this stage | **10** — unchanged (selection happens first, the bump is stage 2) |
| `DATASET_SCHEMA_VERSION` | 3 — unchanged |
| production arithmetic changed | **no default change**; one opt-in parameter was added |
| new files | `production/phase_storage_precision_audit.py`, `test_phase_storage_precision.py`, `evidence/l1a2i/`, `evidence/l1a2i_baseline/` |
| changed files | `phasefield.py` (opt-in storage model only), `.github/workflows/python-ci.yml` |
| dataset lineage | v10 datasets cannot resume silently: the generator fingerprint carries the solver contract *and* a sha256 of `phasefield.py`, both of which change when the storage model ships |

The `phasefield.py` delta is deliberately minimal and opt-in: `PHASE_STORAGE_MODEL`
(default `"float32_contract_10"`), `PhaseFieldParams.phase_storage_model`,
`phase_state_dtype(p)`, a float32 cast of the momentum right-hand sides so a float64 phase state
cannot promote the velocity state, and a dtype-following `solve_ch_implicit`. The default path is
bit-identical (section C).

## C. Harness and fidelity

Two fidelity facts are asserted by tests, not assumed:

* `production_correction` — the audit's correction extractor, a line-for-line copy of
  `phasefield._cg_solve_volume_weighted` that returns the exchange vector `x` instead of `rhs + x` —
  satisfies `rhs + x == pf.solve_ch_implicit(rhs)` **bit for bit** (`jnp.array_equal`) across
  fixtures and iteration counts. Every candidate is therefore compared on the *shipped* solver, and
  the A0 row *is* production.
* the audit's A0 substep chain reproduces the L1A-2h closure ledger's CH-only fixture to
  `3.0e-07` (offset 0) and `3.7e-07` (offset 0.5) in relative mass drift at the frozen quick fixture
  (N = 48, 150°, M = 1·M_ref, 2500 public steps). The residual difference is the phase advection the
  audit's momentum path adds; the frozen 2500-step quick gate value is `-1.242975e-05` (ledger) vs
  `-1.212479e-05` (this audit), and the classification agrees (systematic bias, lag-1 increment
  autocorrelation 0.58 vs 0.61).

Two harness facts were re-derived rather than trusted, and both changed a number:

* the audit's quick gate is run at **M = 1·M_ref** (the frozen `kra.GATES` fixture), not `4·M_ref`;
  at `4·M_ref` every candidate's drift is roughly an order of magnitude smaller and the gate would
  have looked easier than the row the incumbent actually fails;
* the "2500 steps" of the frozen ledger are 2500 **public** steps (7500 substeps), not 2500
  substeps. Both conventions were run during development; the report and the audit use public steps.

## D. Frozen diagnosis and what the audit does with it

The L1A-2h conclusion — the drift is a float32 *storage* bias, not a Krylov defect — is treated as a
hypothesis to be tested on the production path, and it is confirmed *after* a correction that matters:
the earlier L1A-2h finding that a correction recovered as `solved - rhs` is *exact* (`x = fl(phi_new) - rhs`
reproduces `x` exactly when `phi_new = fl(rhs + x)`) is why the audit extracts `x` from the solver
itself. Without that, the B1/C1 candidates are indistinguishable from A0 (measured during
development: bit-identical drift series), i.e. the compensation would have been measured on an
already-rounded quantity.

## E. Candidate matrix

Quick gate: **N = 48, 150°, M = 1·M_ref, 2500 public steps, wall offsets 0 and 0.5 dy, bound
2e-6 (strong 5e-7)**. Medium gate: **N = 128, M = 4·M_ref, 5000 public steps, bound 2e-4 (strong
1e-4)** plus the measured contact angle. All values are relative physical-mass drift of the field the
production operators read, `M = Σ V_i φ_phys,i`.

| candidate | quick offset 0.0 | quick offset 0.5 | medium 150° | medium 60° | verdict |
| --- | --- | --- | --- | --- | --- |
| `A0_float32` (incumbent) | -1.212e-05 | -1.944e-05 | -1.154e-06 | +9.443e-07 | **fails the quick gate** |
| `A1_phase_float64` | **+1.938e-15** | **+1.723e-15** | -3.463e-16 | +9.236e-16 | **passes, strong** |
| `A2_float64_storage_f32_krylov` | -1.188e-06 | -3.989e-06 | +1.368e-07 | +1.456e-07 | fails the quick gate at offset 0.5 |
| `B1_compensated` | -3.815e-06 | -8.405e-06 | +3.809e-07 | -3.970e-07 | fails the quick gate |
| `C1_residual_feedback` | -3.746e-06 | -8.306e-06 | +3.540e-07 | -5.117e-07 | fails the quick gate |
| `F_full_float64` (reference) | +2.154e-16 | +8.615e-16 | -2.309e-16 | +3.463e-16 | passes, strong (not selectable: global promotion) |

The contact angle is identical to three decimals for **every** candidate at the medium fixture
(129.089° at 150°, 69.388° at 60°), so no storage model shifts the measured physics at this horizon;
the angle gate (Δ ≤ 0.2°) is not the binding constraint anywhere in this stage.

Long horizon (N = 48, M = 4·M_ref, 5k/10k/25k/50k public steps, sampled every 1/60 of the horizon),
with the projection of the fitted per-step slope to the 200k CHNS cap:

| candidate | 5k | 10k | 25k | 50k | projection @200k | class |
| --- | --- | --- | --- | --- | --- | --- |
| `A0_float32` | +2.87e-06 | -7.52e-06 | -5.47e-05 | +1.68e-05 | 6.4e-05 … 4.9e-04 | systematic at 25k |
| `A1_phase_float64` | +1.08e-15 | +1.94e-15 | +8.61e-16 | +1.08e-15 | ≤ 1.6e-14 | random walk at the float64 floor |
| `A2_..._f32_krylov` | +9.60e-06 | +1.49e-05 | +1.05e-05 | +1.25e-05 | 3.4e-05 … 4.5e-04 | systematic at 5k/10k |
| `B1_compensated` | +9.19e-08 | -8.69e-06 | -2.75e-05 | +1.95e-06 | 3.9e-05 … 2.8e-04 | systematic at 25k |
| `C1_residual_feedback` | +3.08e-07 | -8.97e-06 | -2.77e-05 | +1.79e-06 | 5.5e-05 … 2.8e-04 | systematic at 25k |

## F. Mechanism — the two-term mass ledger

Every update moves mass through exactly two computed terms, and the audit accumulates both per
substep (`ledger` in the quick/medium rows; `fixed_state_causal` for the single-update version):

* **solve defect** `Σ_i V_i x_i` — the mass of the exchange correction the Krylov solve returns. In
  exact arithmetic the exchange formulation makes this zero *by construction* (the right-hand side is
  a face-flux divergence); in float32 it is not, because each Krylov combination and each division by
  `V_i` re-rounds the telescoping identity.
* **storage loss** `Σ_i V_i (stored_i - (rhs_i + x_i))` — the mass the storage rule drops relative to
  the float64 evaluation of its own two terms.

Accumulated over the frozen quick gate (offset 0, in units of the mass-reduction rounding
`E_round`), against the measured drift:

| candidate | solve defect | storage loss | explained drift | measured drift |
| --- | --- | --- | --- | --- |
| `A0_float32` | -18.8 | **-73.1** | -1.095e-05 | -1.213e-05 |
| `A1_phase_float64` | -0.0 | +0.0 | -1.26e-16 | +1.94e-15 |
| `A2_..._f32_krylov` | -8.1 | +0.0 | -9.69e-07 | -1.188e-06 |
| `B1_compensated` | -20.1 | -0.0 | -2.41e-06 | -3.815e-06 |
| `C1_residual_feedback` | -19.6 | +0.0 | — | -3.746e-06 |

This is the frozen diagnosis, quantified: **for the incumbent the storage term dominates (-73 of
-92 E_round)**, exactly as L1A-2h concluded, and the ledger explains the observed drift to ~10 %
(the remainder is the trajectory change the mass terms themselves induce). It also explains why
each alternative behaves the way it does:

* `B1`/`C1` eliminate the storage term completely (a TwoSum pair / a fed-back residual makes the
  operator-visible field the *correctly rounded* update, so its residual is unbiased), but they
  cannot touch the solve term and the operator-visible field is still float32 — hence a 3×
  improvement and a clean fail.
* `A2` removes the storage term without paying for a float64 Krylov recurrence, but keeps the float32
  solve defect — hence a 10× improvement and a fail at offset 0.5 (`-3.99e-06` vs the 2e-6 bound).
* `A1` removes **both**: the float64 state means there is no storage loss at all, and the float64
  recurrence makes the telescoping identity hold to 1e-16 relative.

## G. Fixed-state causal test and cell-population decomposition

`fixed_state_causal` (spec 26) evaluates the shared production inputs once and reports, per candidate,
the solve defect, the storage loss and the *input shift* (the difference of the candidate's exact
update from the baseline's — a change of the equation, reported separately so it can never be
confused with a storage metric). At N = 128, M = 4·M_ref the ordering is the same as in section F.

`population_series` (spec 27) accumulates the baseline's storage residual over consecutive updates at
one fixed population assignment (N = 128, warmup 600): total `-0.497 E_round`, of which the *liquid
bulk* carries `-0.52`, the interface `+0.03`, and the gas and the cut/zero-volume cells `0.00`. The
single-update decomposition at the same state shows the same structure. The rounding is not
manufactured in the cut cells or in the gas; it is manufactured wherever the phase field itself is
stored, which is what makes a *state* representation (and not a flux or geometry change) the right
lever.

## H. Restart, AD and performance audits

**Restart (spec 24).** For every candidate the formal restart is bit-identical to the uninterrupted
run (`max|Δφ| = max|Δu| = max|Δv| = max|Δaux| = 0.0`), including a real `np.savez_compressed`
round trip that preserves dtype and shape. Dropping a hidden state (`aux = 0`) *does* change the
trajectory, and the audit reports that as a documented non-path rather than a legal restart: a
B1/C1 production candidate whose checkpoint discards its compensation would be a silent physical
change, which is why the dataset lineage section requires a restart-state version bump for them.

**AD (spec 44).** With the exchange increment frozen (so the contract-10 CG's `lax.while_loop` is out
of the differentiated path), reverse and forward mode through the storage rule agree with a central
finite difference (mismatch ≤ 0.2 %) for every candidate, and all gradients are finite — the storage
rules (TwoSum, residual feedback, dtype promotion) are differentiable. Reverse mode through the
*whole* step is unavailable for every candidate including the shipped A0 (the `while_loop` rejects
it); that is a property of contract 10, identical before and after, and is reported as such rather
than presented as a candidate failure.

**Performance (spec 39–42, CPU-only).** Per-substep wall clock after a warm compile, at N = 128
(30 substeps, **minimum of three** timed repetitions; `performance_n128_min3.json`) and at N = 48
(the quick profile, same method). Persistent bytes are exact, not measured. The single-repetition
timings recorded inside the baseline report's `performance.json` (A1 1.66×) and in the first
baseline draw (2.56×) bracket the min-of-three figure below; the conclusion is robust to the noise
because the noise band is ~±5 % within a min-of-three run and ~±15 % single-shot.

| candidate | N = 128 s/substep (min of 3) | ratio vs A0 | N = 48 ratio (min of 3) | persistent bytes ratio | preferred budget (≤ +15 % runtime, ≤ +20 % memory) |
| --- | --- | --- | --- | --- | --- |
| `A0_float32` | 0.002639 | 1.000 | 1.000 | 1.00 | — |
| `A1_phase_float64` | 0.005560 | **2.107** | 1.203 | **1.333** | exceeded |
| `A2_..._f32_krylov` | 0.002732 | 1.035 | 1.137 | 1.333 | runtime within, memory exceeded |
| `B1_compensated` | 0.002881 | 1.092 | 1.016 | 1.333 | runtime within, memory exceeded |
| `C1_residual_feedback` | 0.002582 | 0.978 (in the noise) | 1.006 | 1.333 | runtime within, memory exceeded |
| `F_full_float64` | 0.005327 | 2.019 | 1.291 | 2.000 | exceeded |

The structure of the cost is exactly what the arithmetic predicts: A1 and F pay for a float64 Krylov
recurrence, whose weight grows with the CG iteration count (2–3 iterations at N = 48 → 1.2×; 18 at
N = 128 → 2.1×), while A2/B1/C1 keep the float32 recurrence and pay only for their extra per-cell
work (≤ 1.1×).

No GPU device exists in this environment, so the GPU cost of any candidate — in particular the fp64
throughput that decides whether A1 is affordable in production — is **unverified**, and the report
says so instead of implying it was measured.

## I. Candidate selection and rejection reasons

Selection is by invasive rank among the candidates that clear every *measured* gate, and the rank is
`A2 (1) < A1 (2) < B1 (3) < C1 (4)`; `A0` and `F` are references and can never be selected (F's global
float64 promotion is forbidden without cost evidence).

* `A2` — **REJECTED**: `MASS_GATE_FAIL` (quick offset 0.5, `-3.99e-06` > 2e-6).
* `B1` — **REJECTED**: `MASS_GATE_FAIL` (both quick offsets), `DATASET_STATE_AMBIGUOUS` (new
  persistent field).
* `C1` — **REJECTED**: `MASS_GATE_FAIL` (both quick offsets), `DATASET_STATE_AMBIGUOUS`.
* `A1` — **SELECTED**: least invasive candidate clearing the quick (both offsets, strong bound), the
  medium (150° and 60°), the long-horizon (no systematic bias; projection ≤ 1.6e-14), the restart and
  the AD gates, with `phase_only_float64_v1` (production model name).

Every rejection reason is drawn from the machine-readable vocabulary in
`phase_storage_precision_audit.REJECTION_REASONS` and written to `candidate_matrix.json` per
candidate, together with the measured value that produced it.

## J. Cost and production readiness

The selected candidate is selected on *physics*, and its cost is reported, not hidden: on this 2-core
CPU sandbox A1 costs **2.11× the incumbent's runtime** at N = 128 (and 1.20× at N = 48, min-of-three
timings) and **1.33× the persistent phase state**, both above the spec's *preferred* envelope
(≤ +15 % / ≤ +20 %). The audit's own verdict
string is therefore

```
PHASE_STORAGE_MODEL_SELECTED_COST_ABOVE_PREFERRED_BUDGET
```

and its readiness statement is

```
NOT READY FOR L1B PRODUCTION SCALE (runtime/memory above the preferred envelope on the measured
device; GPU cost unverified (no GPU in this environment))
```

That is the honest reading of the evidence: the physics gate is closed by A1 and only by A1, while the
cost envelope is *not* met on the measured device and is unmeasured on the production one. The
cheaper mixed-precision variant that would have met the runtime envelope (A2, +7 % runtime) does not
clear the mass gate, so the choice is between paying A1's cost and leaving `N-CH-MASS-PRECISION` open.

## K. Dataset and restart lineage (spec 52)

`dataset_lineage` reads the generator rather than trusting a comment: the trajectory fingerprint
carries `schema`, `solver_contract` **and** `solver_sha256 = sha256(phasefield.py)` plus
`phase_transport_metadata`, and `_saved_case_is_current` returns False when either mismatches.
Consequences:

* **A1/A2/F** need no new dataset field and no checkpoint-format change: the promoted phase array has
  the same name and shape, so a stored float64 array is written/read by the existing `np.savez`
  path, and every v10 dataset is invalidated by the solver hash + contract change (no silent resume).
* **B1/C1** would need a second persistent array, a restart-state version bump and an explicit
  fingerprint key; the audit records `requires_new_persistent_field: true`,
  `dataset_state_ambiguous` and the fail-closed rule for them. Both are rejected on the mass gate
  anyway, so no dataset surgery is required by this stage.
* The v11 metadata block to add when the storage model ships (`phase_storage_model`,
  `phase_storage_dtype`, `phase_compensation_model`, `phase_restart_state_version`) is listed in the
  audit report for the stage-2 patch; nothing is bumped now.

## L. Gates

| gate | bound | A1 (selected) | A0 (incumbent) |
| --- | --- | --- | --- |
| quick, offsets 0 / 0.5 dy | ≤ 2e-6 (strong 5e-7) | 1.9e-15 / 1.7e-15 **pass (strong)** | 1.2e-05 / 1.9e-05 **fail** |
| medium N = 128, M = 4 M_ref, 150°/60° | ≤ 2e-4 (strong 1e-4) | 3.5e-16 / 9.2e-16 **pass (strong)** | pass |
| angle delta (candidate vs A0) | ≤ 0.2° | 0.000° **pass** | — |
| long horizon, no systematic bias | projection @200k < 1e-3 | ≤ 1.6e-14 **pass** | systematic at 25k |
| restart equivalence | bit-identical | **pass** | pass |
| AD (storage rule) | forward/reverse vs FD ≤ 5 % | 0.2 % **pass** | pass |
| performance (preferred) | ≤ +15 % runtime, ≤ +20 % memory | **2.11× / 1.33× — exceeded** | — |
| GPU cost | must be measured or declared | **not measured (no GPU)** | not measured |

## M. Anti-cheating evidence

Every claim below is a test in `test_phase_storage_precision.py` that fails the build rather than a
statement of intent:

* the formal gate and the ledger are built from `physical_mass` / `operator_field` only; the
  bookkeeping sum is computed in `bookkeeping_mass` and *reported separately*, and the gate path
  (`candidate_rows`, `select_candidate`, `_verdict`, `production_readiness`) is asserted by AST to
  never read it, so `Σ V(φ + hidden)` can never be passed off as physical mass;
* `operator_field(state, candidate) == state["phi"]` bitwise for every candidate, and for B1 with
  `aux = 1e-3` the reported physical mass is exactly `Σ V φ`, not the compensated sum;
* an AST scan fails if any production operator call (`rhs`, `chemical_potential_fluxes`,
  `control_volume_divergence`, `advective_phase_source`, `production_advective_rate`,
  `solve_ch_implicit`, `production_correction`, `phase_transport_operator`) is handed the hidden
  field, and a runtime spy over `pf.rhs` fails if the compensation field is ever passed as the phase
  state — i.e. the compensation is never advected as a physical scalar (spec 30/31);
* a source scan (comments and docstrings stripped by `tokenize`) fails on any global-correction,
  redistribution, rescaling, clipping or angle-calibration machinery in the audit module;
* the production default is asserted to be `float32_contract_10` with `phase_state_dtype == float32`,
  a whole default step is asserted to keep φ/u/v/t in float32 (no silent global promotion), and an
  unknown model fails closed;
* the contract version is asserted to still be 10 in this stage (the bump is stage 2).

## N. Untouched blockers and independence

| item | status in this stage |
| --- | --- |
| `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN` | `resolved_in_contract_v9` — untouched |
| `P-SOLID-PIN`, `I-CONTACT-GAP`, `N-DT`, `P-VARDENS-PROJ`, `P-CAP-RHO`, `P-VARVISC`, `BC-Y-PERIODIC` | independent, untouched |
| `N-CH-MASS-PRECISION` | **still open** — the candidate is selected, the production change has not shipped, and the staged CHNS closure has not been rerun on A1 |
| `W-CONTACT-ANGLE` | **still open** — same reason; the 150° 50k row of the contract-10 ledger was `+3.118841e-04` with a 2.58° angle error, and no closure claim is made here |
| Krylov / Young wetting / cut-cell geometry / wall measure / M / dt / Brinkman / gas properties / pressure projection | **not touched by any candidate** (the audit uses the shipped solver verbatim) |

## O. Staged CHNS closure handoff (what this stage does *not* do)

The contract-10 closure ledger (`evidence/mass_precision/closure.json`, authoritative, 50 000 steps
each) records 60° `1.0538e-03` (**above the 1e-3 bound**), 90° `4.784e-04`, 120° `2.44e-04`, 150°
`3.12e-04`. With A1 selected, the staged rerun (60° → 90° → 120° → 150° at 50k, then extend to
100k/150k/optional 200k only while the drift is clean or relaxing, then the four-offset alignment
family and the Laplace/fit checks) is the **next** stage's work, together with the contract 10 → 11
bump and the metadata block. This report deliberately makes no convergence claim for it. Given A1's
measured CPU cost, the staged rerun is expected to be, at a minimum, ~2.5× the contract-10 wall clock
on the same hardware.

## P. CI

`.github/workflows/python-ci.yml` gains the new test file in the two-phase unit-test step and a
quick-profile audit step that writes `artifacts/phase_storage_precision_quick`. Hosted CI runs **only**
quick-profile items (spec 73): the medium matrix, the long-horizon series, the performance matrix and
the 100k horizons are not run in CI. The audit's CLI exits 0 when the *harness* integrity holds; a
gate that is not met is a reported finding, and no step or artifact of this stage claims "all CI
passed".

## Q. How to reproduce

```bash
cd examples/two_phase
export JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"

# CI-sized evidence (quick profile): evidence/l1a2i/
python -m production.phase_storage_precision_audit --profile quick --out evidence/l1a2i

# the decision evidence (medium matrix + long horizons): evidence/l1a2i_baseline/
python -m production.phase_storage_precision_audit --profile baseline --out evidence/l1a2i_baseline

# the forensic profile adds the 100k horizon and every candidate in the restart/performance matrices
python -m production.phase_storage_precision_audit --profile forensic --out evidence/l1a2i_forensic

python -m pytest -q test_phase_storage_precision.py
```

Artifacts of the baseline run: `phase_storage_precision_report.json/.md`, `candidate_matrix.json`,
`manifest.json`, `performance.json`, `restart_equivalence.json`, `long_run_series.json` (the raw
sampled mass series of every long-horizon run, so no number in this report has to be taken on trust),
plus `performance_n128_min3.json` (the min-of-three N = 128 timing quoted in section H, written by
`phase_storage_precision_audit.performance` directly).

## R. Compliance against the spec's hard-fail list

| hard fail (spec 76) | status |
| --- | --- |
| global mass correction | absent (AST scan + no such call path) |
| hidden residual counted as physical mass | absent (gate built from `physical_mass`; AST-enforced) |
| restart discarding compensation state | impossible by construction; documented and tested as a non-path |
| contact-angle shift > 0.5° unexplained | 0.000° shift measured at the medium fixture |
| alignment spread > 2° | not measured in this stage (no geometry change); no claim is made |
| energy monotonicity break | not measured in this stage; no claim is made |
| M or dt changed to reduce drift | unchanged — the audit inherits `mpa.build_case`; no candidate touches them |
| wetting or geometry modified | unchanged (Young energy, wall measure, cut-cell arrays untouched) |
| global float64 promotion without cost evidence | F is measured and *rejected as a selection*; A1's cost is reported in full, including the unverified GPU part |
| dataset lineage ambiguity | A1 needs no new field; B/C flagged and rejected; v10 datasets cannot resume silently |
| old contract data resuming silently | impossible: fingerprint carries the solver hash and the contract |
| L0 smoke failure | not run in this stage's evidence; the two-phase job's existing smoke steps are untouched by the CI delta |

## S. Status lines and open items

```
STAGE=L1A-2i
SELECTED_CANDIDATE=A1_phase_float64 (phase_only_float64_v1)
CONTRACT_VERSION=10 (unchanged; stage 2 owns the 10 -> 11 bump)
VERDICT=PHASE_STORAGE_MODEL_SELECTED_COST_ABOVE_PREFERRED_BUDGET
PRODUCTION_READINESS=NOT READY FOR L1B PRODUCTION SCALE (cost above the preferred envelope on CPU;
   GPU cost unverified)
QUICK_GATE=A1 pass (strong), A0 fail
MEDIUM_GATE=A1 pass (strong); all candidates pass; angle shift 0.000 deg
LONG_HORIZON=A1 random walk at the float64 floor; A0/A2/B1/C1 systematic
N-CH-MASS-PRECISION=OPEN (production change not shipped; staged CHNS closure not rerun)
W-CONTACT-ANGLE=OPEN (unchanged)
NEXT=contract 10 -> 11 with phase_only_float64_v1 + metadata, then the staged CHNS closure
      (50k -> 100k -> 150k -> optional 200k), then the alignment/Laplace family
```

Open items handed forward, in order:

1. ship `phase_only_float64_v1` in production (contract 11) — including the fingerprint metadata and
   the `phase_storage_dtype`/`phase_storage_model` keys — or decide against it on cost grounds;
2. rerun the staged CHNS closure at 60/90/120/150° (50k → 100k → 150k → optional 200k), extending a
   horizon only while it stays drift-clean or relaxing, and measure the angle deltas against
   contract 10 at each stage;
3. only if all four angles truly converge with mass drift ≤ 1e-3, close `N-CH-MASS-PRECISION`; then
   close `W-CONTACT-ANGLE` if the angle error also clears its own bar;
4. measure the GPU cost of A1 (the one number this environment could not produce) before scaling to
   L1B.
