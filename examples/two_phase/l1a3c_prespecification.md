# L1A-3c prespecification — zero-mobility admissibility and one control

Authority: owner approval for L1A-3c, following published L1A-3b head
`144d92f273eaaef68175baf72620ec3919275b31`, observable freeze
`6849cc0dd77383a728ffdaf8c8c23df10c9a8e15`. PR #24 remains OPEN/Draft.
This protocol, implementation and adversarial tests must be committed before either target arm.
Their SHA-256 values and the commit are recorded in `prespecification_freeze.json` before measurement.
No method, test or decision-rule editing after measurement; a failure stops rather than being repaired
with favorable thresholds. Prior evidence and source, solver contract 12, geometry and quality debt are immutable.

## A. Discrete intervention and admissibility

Only `PhaseFieldParams.M` changes from 0.002 to exactly 0.0. No solver code is changed.
On every frozen substep h=dt/3, define K from authoritative shared face weights w=A/d and
L=V_safe^-1 K (positive stiffness sign). The executed phase update is

- `mu_exp=fprime(phi)/eps+wall_energy_derivative(phi)*wall_measure_density`;
- `mu=mu_exp+eps*L(phi)`;
- `a=-div(F_adv)` with unchanged upwind flux `F_adv=A*u_face*phi_upwind`;
- `J_CH=-M*w*(mu_exp,j-mu_exp,i)`;
- `b=phi+h*(a-div(J_CH))`;
- `(I+h*M*eps*L^2)*phi_new=b`, solved by the existing volume-weighted exchange CG.

For M=0 and finite geometry/fields, J_CH=0 and A=I, whose condition number is 1. The correction
rhs `b-A*b` is zero, the initial CG residual is zero, its while-loop does not enter a 0/0 recurrence,
and phi_new=b with zero iterations, zero reported relative residual and converged=True. The linear
operator call remains; this is not bypassing the CH solver, clipping phi or deleting capillary physics.
There is no inverse mobility normalization, mobility preconditioner or mobility-based stability test
on this path. CG floors `1e-30`, guarded inner-product denominator, V_safe and eps are unchanged.
The unused `projection_legacy` FFT denominator tends to 1, but that branch is never selected.
`implicit_solve_diagnostics` reports h*M*eps and its spectral estimate, both zero; geometry diagnostics
remain defined. The adjoint uses the same identity solve (no optimization/training performed).

Full chemical potential, wall terms, capillary acceleration `+(SIGMA_NORM/We)*mu*grad(phi)/rho_l`,
viscosity, advection, Brinkman damping, pressure projection and existing gravity flag remain active.
Velocity RHSs are independent of M at identical inputs, but subsequent flow may differ through phi.
The conservative face divergence telescopes in sum(V*phi), including zero flux on closed faces and
the closed y seam. At zero mobility the implicit exchange is exactly zero; advection conserves mass
up to floating-point arithmetic. This does not prove material advection accuracy or a physical mechanism.

Static AST inventory plus synthetic 16x16 flat-wall finite phase/velocity probes must pass before target
execution: exact zero CH flux, exact implicit identity, exact advective limiting update, CG identity,
weighted mass arithmetic check, nonzero capillary probe, unchanged and correctly retained momentum RHS,
positive-M flux discriminator, and three captured synthetic substeps versus the public step within the
inherited 1e-6 field equivalence gate. These synthetic runs are not pillar_training trajectories.
Arithmetic conservation probe bound is 64 float64 eps times max(1, weighted absolute input/output mass).
It is not a root/solver-state uncertainty estimate. Failure => ZERO_MOBILITY_ADMISSIBILITY_BLOCKED.

## B. Frozen paired observable and reporting

Import the exact L1A-3b implementation, never edit it. Q=[q95,q96] on ray 95 at x=2.984375 and
ray 96 at x=3.015625. Both use the fixed downward wall-normal direction and original signed y/dx root
coordinate. Keep all crossings, frozen global contour/cell equivalence, component topology and historical
NON_UNIQUE minimum-gap diagnostics. Neither a primary support nor any scalar aggregation is permitted.

Retain ±1.5 dx root continuation window, unique candidate, ≤0.25 dx root step, ≥0.5 dx competing-root
separation, node-degeneracy rule and 0.375 dx geometric correspondence gate. Both roots must belong to
one current contour component. Track components geometrically; never compare integer labels across
states or trajectories. Same-time paired branch correspondence uses the frozen geometric relation
and tolerance; record split/merge relations rather than rewriting them. An ambiguous/disjoint/no-component
relation invalidates the pair. Topology change itself is not automatically failure when frozen correspondence
holds. Candidate and ray enumeration order cannot determine a tie-break.

For each j: d_j(t;M)=q_j(0;M)-q_j(t;M), positive toward the solid.
E_j=d_j(t;0)-d_j(t;0.002)=q_j(t;0.002)-q_j(t;0), given identical initial roots.
Positive E means the zero-M trajectory moved farther toward the solid on that ray.
Record both components, each sign, raw signed range, signed E95-E96, absolute support disagreement,
sign discordance, identity residual and validity failures at every S1/S2/S3, all 120 substeps and all
40 macro boundaries (S3 at substep indices 2,5,...119), plus initial Q and state hashes.
An invalid ray invalidates the required complete pair: no partial effect vector, extrapolation or root
reselection. Continue collecting raw contour/candidate diagnostics after tracking loss, explicitly marking
the readout unavailable. Numerical nonfinite/solver failure or contact guard stops evolution.
No comparison of E with the old operator-attribution CH leg as though they measure the same quantity.

## C. Baseline integrity and execution order

Baseline is frozen diagnostic C2 (`SDF_TAPERED_STREAMFUNCTION_V1`), not production uniform or C0/C1.
Case pillar_training; Nx=Ny=192, Lx=Ly=6, dx=dy=0.03125, dt=0.002, eps=0.046875, M=0.002,
40 macro steps, 120 substeps, t<=0.08. All physical/numerical dataclass fields, dtypes, initial phi/u/v,
initial time, geometry and boundary settings must match the baseline. Control differs only in M.
C2 remains diagnostic, never promoted.

Verify sealed L1A-2z/3b bound files, all source hashes, historical endpoints
`d95=+0.0626486818930445 dx`, `d96=+0.0626486836962457 dx`.
L1A-3b consumer was independently verified at its published head before this stage. Its stage-owned
live preflight cannot be reused on successor commits; reverify its immutable contents and source hashes.
The L1A-2z consumer remains authoritative. Never edit earlier-stage evidence to make a successor preflight pass.

Sealed records lack phi arrays and complete candidate lists, so ONE baseline replay is necessary and
explicitly authorized here before control. It is an actual trajectory execution, not an intervention arm.
Use frozen `substep_capture`, reusing all production kernels and momentum/pressure algorithms, without
public-step shadow target trajectories. Reconstruct every intermediate global contour and fixed-ray
candidate list. Require all 360 intermediate phi hashes and final phi/u/v hashes EXACTLY equal to the
sealed baseline, all baseline readouts VALID, same initial hashes, parameter/geometry fingerprints, and
maximum root discrepancy ≤the inherited CONTRACT_LEG_TOLERANCE_DX=1e-9. Do not relax byte identity
on a reset environment. Failure => CONTROL_PROTOCOL_OR_BASELINE_BLOCKED; zero controls executed.

Run order: restore source/environment; immutable prerequisite verification; static and synthetic audit;
adversarial/focused and adjacent/lineage suites and quality ratchet; commit freeze; record exact passing
precontrol tests bound to frozen source hashes; execute one baseline replay; verify it independently;
only then execute exactly one M=0 trajectory. Exclusive receipt files are written BEFORE each arm's
first evolution call. A started receipt prohibits repeats, including after interruption or numerical failure.
No other mobility, arm, target shadow replay, mesh, time step or initializer is permitted.

Conservation gates remain those from L1A-2z: substep mass-balance residual ≤1e-12, exact closed-face
flux zero, algebraic increment recombination Linf ≤1e-12, every CG solve converged. Report raw mass
drift, per-leg mass, residuals, CG residuals/iterations and contact gaps; zero-arm exact identity/flux
checks are additional limiting-equation tests, not changed physics tolerances. Stop if frozen contact
gap ≤1.5 dx on any substep; no contact scoring. Frozen contour readout/reconstruction is checked against
original enumeration and cell rules on each stored field. Synthetic capture/public equivalence and sealed
baseline hash replay verify the executed algorithm; no extra control trajectory is used to test fusion.

## D. Uncertainty, falsification and verdict

Measure float interpolation versus 60-digit Decimal reconstruction of the same stored field on each
selected root. This is arithmetic discrepancy only, not an upper bound on solver-state, discretization
or physical error. Record baseline replay discrepancies separately. Historical ~1.8e-9 dx ray agreement
is not a noise floor, intervention threshold or uncertainty estimate. No independently justified total
solver-state uncertainty bound or binary support-invariance threshold is established in this freeze.
They remain null/unresolved. Large valid disagreements and opposite signs are retained, not invalidated
solely for disagreement. No claim of support-independent sensitivity or physical significance.

Premeasurement tests must discriminate: zero limiting equation/conservation; incorrect explicit/implicit
mobility; mass/CG failure; deleted capillary/momentum; different initial state/geometry; changed unrelated
parameters; missing/multiple/ambiguous roots, correspondence loss, identity swapping, enumeration/ray-order
permutations; sign inversion/opposite signs, large disagreements, averaging; invalid uncertainty/invariance
claims; reconstruction/initial/final baseline failures; source/evidence tampering; failed-gate promotion;
and exclusive execution receipts. Reuse frozen L1A-3b adversarial tests without edits.

Verdicts, in order: failed A -> ZERO_MOBILITY_ADMISSIBILITY_BLOCKED; failed protocol/baseline before
comparison -> CONTROL_PROTOCOL_OR_BASELINE_BLOCKED; executed but invalid required pair ->
CONTROL_OBSERVABLE_INVALID; valid complete raw effect vector -> CONTROL_PAIRED_RESPONSE_RECORDED.
EARLY_WINDOW_MOBILITY_SENSITIVITY_RESOLVED is unreachable without a new independently justified,
owner-approved uncertainty/sensitivity freeze. Every outcome says PHYSICAL_MECHANISM_NOT_ESTABLISHED.

## E. Repository and publication

Keep PR #24 OPEN/Draft, normal pushes only on arena/02b6c71b-hydrogym. No prior evidence, physics,
legacy failure inventory or `quality/legacy_two_phase_debt_v1.json` changes. Preserve
PASS_BASELINE_ONLY (171 Ruff / 27 formatter / 28 isort / 3 codespell), zero added diagnostics.
Run focused L1A-3c, L1A-3b/2z/2y, frozen adjacent 2t/u/v/w/x and legacy-lineage 2s suites; preserve and
report inherited failures verbatim, block on unexpected failures. The prior L1A-3b stage-owned preflight
is expected to fail on successor commits and must not be edited or counted as a pass.
Seal l1a3c evidence with bound-file inventory and source/test/prespec hashes; independent consumer
must reject mutations and invalid promotion. A 17-item final report records all scientific limitations,
execution counts, exact commands/results, hashes, CI and next action. Hosted CI must be observed for
the actual final pushed head; record that observation in PR #24 without repeatedly changing sealed
scientific evidence merely to chase an evidence-only commit. Next action NOT_STARTED, separate approval.
