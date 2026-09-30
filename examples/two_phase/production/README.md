# L1A-1 production physics validation

## A. Purpose

This package establishes a traceable, repeatable **production validation framework** for the existing JAX two-phase prototype. It is separate from the L0 smoke pipeline and is **not** a production-dataset generator. The suite measures static-droplet Laplace response and parasitic currents, apparent contact-angle calibration, flat-wall impact observables, and optional grid/interface-thickness sensitivity.

The framework is deliberately diagnostic. It does not rewrite the numerical physics core, fit wetting parameters, or claim that the current solver is production validated. Every report is marked `physics_status = BASELINE_ONLY`.

## B. Current solver baseline limitations

These are known open blockers captured in every report; L1A-1 measures and documents them but does not fix them:

- Pressure projection is constant-coefficient, not the variable-density operator `div((1/rho) grad(p))`.
- Viscous acceleration uses `nu(phi) * lap(u)` rather than the full divergence of variable-viscosity stress.
- Capillary acceleration is divided by `rho_l`, rather than local `rho(phi)`.
- `stable_dt()` enforces advective CFL only; it has no explicit viscous, capillary, or phase-field bounds.
- Operators are periodic in both x and y. The embedded bottom solid therefore remains periodically connected to the top boundary.
- Wetting is the existing wall-affinity model. The measured apparent-angle calibration has not yet been accepted as adequate for L1 production.
- `pressure_field()` reconstructs the projection pressure from the current state. It is a **projection-reconstructed pressure diagnostic**, not an independently advanced thermodynamic pressure.

`SOLVER_CONTRACT_VERSION` is 5. Contract 4 -> 5 (L1A-2a, section E) changed exactly one thing in the solver, the sign of the capillary (Korteweg) force; the validation package itself still only reads solver states and does not change trajectory semantics. The legacy `validate_physics.py` remains available as a manual script; this machine-readable runner lives under `production/`.

## C. Commands

Run commands from `examples/two_phase`:

```bash
# Fast, CPU-sized framework/contract run (not an accuracy gate)
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONHASHSEED=0 \
python -m production.run_validation \
  --config production/configs/ci.json \
  --out artifacts/production_validation/ci \
  --overwrite

# Main physics baseline (larger and intentionally not part of normal CI)
python -m production.run_validation \
  --config production/configs/baseline.json \
  --out artifacts/production_validation/baseline \
  --overwrite

# Optional grid and interface-thickness sweeps; can be expensive
python -m production.run_validation \
  --config production/configs/convergence.example.json \
  --out artifacts/production_validation/convergence \
  --overwrite

# Analytic capillary / pressure sign audit (no time stepping, a few seconds; exit 1 = inconsistent)
python -m production.capillary_audit

# Strict before -> after comparison of two reports (exit 1 = a guard failed: STOP and investigate)
python -m production.compare_reports \\
  --before artifacts/production_validation/<old>/report.json \\
  --after artifacts/production_validation/<new>/report.json

# Opt in to provisional physics-readiness target checks (not enabled in CI)
python -m production.run_validation \
  --config production/configs/baseline.json \
  --out artifacts/production_validation/baseline-strict \
  --strict-targets
```

The output is `report.json`. It is written atomically as strict UTF-8 JSON. Existing output directories fail closed unless `--overwrite` is supplied. Generated reports are kept under the already-ignored `examples/two_phase/artifacts/` root.

Configuration is schema-checked before solver execution; unknown keys and invalid dimensions/values are rejected. The canonical sorted-key JSON SHA-256, git SHA (or explicit fallback), `phasefield.py` SHA-256, validation-source SHA-256, runtime versions/backend/devices, and all run parameters are recorded. The source fingerprint covers the sorted set `config.py`, `observables.py`, `validation.py`, `convergence.py`, `report.py`, and `run_validation.py`.

### Observable definitions

- `total_phase_mass` integrates over all cells; `fluid_phase_mass` (also exposed as `liquid_mass`) integrates only where `sdf >= 0`.
- x-COM uses a periodic circular mean; y-COM uses a regular weighted mean.
- Periodic width is the minimum covering arc from thresholded occupied x-columns; `beta = width / (2R)`. Empty and full-domain cases are defined.
- `g_0.5` and `g_0.1` are minimum nonnegative SDF values at liquid thresholds 0.5 and 0.1. The event flag is the diagnostic rule `g_0.5 <= 1.5*dx`, not a unique physical definition of contact.
- Static `laplace_ratio = (p_inside - p_outside) * R * We` follows this solver's nondimensional convention: `phi = 1` is liquid, so *inside* (`r < 0.3 R`) is the liquid core, *outside* (`r > 2.5 R`) is far gas, and `delta_p = P_liquid - P_gas` is expected to be `+1/(We R)` (ratio `+1`) for a 2-D circle. Contract 4 measured about `-1` because of the capillary sign (section E); the benchmark is never sign-flipped. The pressure method is always named `projection_reconstructed`.
- `Oh_derived = sqrt(We) / Re` is metadata only; the solver continues to accept `We` and `Re`.
- Relative sweep changes use `abs(Q_fine-Q_coarse) / max(abs(Q_fine), epsilon)`. A fixed `eps_factor` resolution sweep is explicitly named **coupled grid/interface refinement** because physical epsilon and Cahn number decrease as N increases. Fixed-physical-epsilon sweeps are also supported.

## D. Interpretation

`contract_status=PASS` means the configured run completed with complete records, valid shapes/ranges, finite metrics, valid config lineage, and a schema-valid report. It **does not imply** `physics_status=VALIDATED`, that any provisional accuracy target is met, or that the solver is `PRODUCTION_READY`. The CI validation profile checks the framework contract and numerical finiteness only; it does not gate on Laplace error, contact-angle error, spurious-current magnitude, or refinement sensitivity. The Laplace **sign** (and `1/R` ordering) is gated separately by fast unit tests and by `production.capillary_audit`. `P-LAPLACE-SIGN` is reported as `resolved_in_contract_v5` only when a multi-radius baseline shows all ratios positive, `R^2 >= 0.99` and a positive slope; this does not validate the capillary model or the solver.

The provisional targets in the report (mass drift 0.1%, Laplace error 5%, contact-angle MAE 5 degrees / maximum 10 degrees, key-observable refinement change 3%) are held fixed as next-stage engineering goals, not thresholds tuned to the existing results. Use `--strict-targets` only when a nonzero exit on missed readiness targets is desired.

## E. L1A-2a: capillary / pressure sign consistency (solver contract 5)

**Finding.** The L1A-1 static-droplet baseline followed the expected `1/R` law (`delta_p` vs `1/R`: R^2 = 0.99964) but gave `laplace_ratio = delta_p R We` of about `-1` instead of `+1`. Magnitude and scaling were right; the sign was not. `python -m production.capillary_audit` localised the defect to the **force** convention: the pressure projection and `pressure_field()` were already consistent with each other, but the Korteweg force pointed away from the liquid and the flow it drives *raised* the interfacial free energy (a negative surface tension). Contract 5 flips that one sign. Unchanged: `SIGMA_NORM`, `poisson_solve`/`m2_proj`, the `/rho_l` denominator, viscosity, wetting, `dt`, boundaries.

**Convention** (derived from the free energy, not fitted to the benchmark; also written in the `phasefield.py` module docstring):

- Orientation: `phi = 1` liquid, `phi = 0` gas, `phi(r) = 0.5 (1 - tanh((r - R)/(sqrt(2) eps)))`, so `dphi/dr < 0` and `grad(phi)` points gas -> liquid. The outward normal (liquid -> gas) is `n_out = -grad(phi)/norm(grad(phi))`.
- Chemical potential: `mu = dF/dphi = f'(phi)/eps - eps lap(phi)` (+ wetting). On a convex liquid interface `mu = +eps norm(dphi/dr) / r > 0`.
- Force: Cahn-Hilliard advection changes the free energy at the rate `dF/dt = -int mu u.grad(phi)`, so a force that creates no free energy is `+mu grad(phi)`, equivalently `-phi grad(mu)` (they differ by the pure gradient `grad(mu phi)`). Implemented: `F = +(SIGMA_NORM/We) mu grad(phi) / rho_l`, directed toward the liquid (`F . n_out < 0`). Contract 4 had the opposite sign.
- Pressure: `du/dt = ... - grad(P) + F`, `u+ = u* - dt grad(P)`, `lap(P) = div(u*)/dt`; a static drop has `grad(P) = F`, so `P_liquid - P_gas = -int F_r dr`.
- Laplace jump: `delta_p = P_liquid - P_gas = +1/(We R)` (2-D), `laplace_ratio -> +1`.
- Form A (`+mu grad(phi)`) rather than Form B (`-phi grad(mu)`): both are in the energy-consistent class, so energetics fix the *sign*, not the form. Form A keeps the change to one sign, and its projection pressure is the mechanical pressure whose jump is the Young-Laplace value for any admissible `mu`; Form B's reduced pressure `P_A - (SIGMA_NORM/We) mu phi` loses the jump once `mu` relaxes to a constant (audit check `relaxed_mu_form_choice`). For variable density the two forms are no longer equivalent; that choice belongs to L1A-2c, not here.

**Audit** (`python -m production.capillary_audit`, N=128, R=0.8, no time stepping). Checks that fail on contract 4 and pass on contract 5: `korteweg_force_direction` (net `F . n_out` = +0.0123 -> -0.0123), `laplace_jump_sign_and_scale` (-0.985 -> +0.985), `free_energy_consistency` (the driven flow raised `F` -> lowers it), `force_form_equivalence_class` (solver force vs Form A: cosine -1 -> +1). Checks that pass on both: phase orientation, `mu = +dF/dphi`, projection convention (`P = +psi` for `F = +grad(psi)`), `pressure_field()` = projection pressure, and force-integral jump = projection jump (`-0.985064` vs `-0.985077`; the diagnostic hides nothing).

**Before / after** (unchanged `configs/baseline.json`; N=128, We=100, float32, 1000 steps; before = `85c2a47` contract 4, after = `efdb4d1` contract 5, produced with `production.compare_reports`):

| R | Before ratio | After ratio | abs(After-1) |
|---|---:|---:|---:|
| 0.6 | -1.0459 | +1.0459 | 0.0459 |
| 0.8 | -1.0193 | +1.0194 | 0.0194 |
| 1.0 | -1.0083 | +1.0083 | 0.0083 |
| 1.2 | -1.0031 | +1.0032 | 0.0032 |

- `delta_p` vs `1/R`: slope -0.010905 -> +0.010905, R^2 0.99964 -> 0.99964; after mean abs error 0.0192, max 0.0459 (R = 0.6; the resolved radii R >= 0.8 are within 2 %). No calibration factor is applied for small radii; their interface sensitivity is left to the convergence study.
- Parasitic current (peak over the run) fell at every radius, 1.9-2.6e-4 -> 1.5-1.8e-4, and final kinetic energy fell by a factor 1.3-6.5. Static mass drift stays at 3e-8 to 3.5e-7.
- Flat-wall impact (both baseline cases): finite, mass drift unchanged (1.0e-3), peak speed unchanged (it is the initial condition, 2.31); `beta_max` 2.21 -> 2.14 (We = 100, neutral) and 2.14 -> 2.08 (We = 50, hydrophobic); the time of `beta_max` moves from 2.40 (the last sample, still increasing) to 2.28 for the neutral case and from 2.16 to 2.12 for the hydrophobic one.
- Sessile contact angle is **not fixed and not tuned here**; it got worse in aggregate: measured 114.1/117.0/119.9/122.8 deg -> 119.8/120.0/120.1/117.3 deg for targets 60/90/120/150 deg (MAE 27.07 -> 30.63 deg, non-monotonic at 150 deg). Its fluid-region mass drift (0.6-0.7 % -> 0.2-1.6 %) is liquid seeded inside the solid slab moving into the fluid region; total phase mass is conserved to about 3e-7.
- Pre-contact gap: `min g0.5 = 0.1484` and `min g0.1 = 0.0547` are unchanged. A diagnostic-only ablation (not committed) showed they do not move with the capillary sign, with `We -> 1e6`, with `wall_energy_amp = 0` or with `wet_band = 0.05`, but they do move with the Brinkman strength (`min g0.5` 0.148 -> 0.055), `enforce_solid_phi` (-> 0.102) and the gas viscosity (`min g0.1` -> 0.008); the coincidence with `wet_band = 0.15` is accidental. The no-contact behaviour therefore points to the penalised solid / gas film, not to the wetting band.

**Status.** `physics_status` stays `BASELINE_ONLY`. Open: `P-VARDENS-PROJ`, `P-CAP-RHO`, `N-DT`, `P-VARVISC`, `BC-Y-PERIODIC`, and `W-CONTACT-ANGLE` (confirmed problem). Only `P-LAPLACE-SIGN` is `resolved_in_contract_v5`; this is a sign/consistency fix, not a validation of the capillary model. Old contract-4 datasets are stale by fingerprint (`SOLVER_CONTRACT_VERSION` is part of `generate_dataset._dataset_fingerprint`) and are regenerated by `generate_dataset.py`.

