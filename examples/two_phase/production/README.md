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

`SOLVER_CONTRACT_VERSION` remains 4. This package only reads solver states and does not change trajectory semantics. The legacy `validate_physics.py` remains available as a manual script; this machine-readable runner lives under `production/`.

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
- Static `laplace_ratio = (p_inside - p_outside) * R * We` follows this solver's current nondimensional convention. The pressure method is always named `projection_reconstructed`.
- `Oh_derived = sqrt(We) / Re` is metadata only; the solver continues to accept `We` and `Re`.
- Relative sweep changes use `abs(Q_fine-Q_coarse) / max(abs(Q_fine), epsilon)`. A fixed `eps_factor` resolution sweep is explicitly named **coupled grid/interface refinement** because physical epsilon and Cahn number decrease as N increases. Fixed-physical-epsilon sweeps are also supported.

## D. Interpretation

`contract_status=PASS` means the configured run completed with complete records, valid shapes/ranges, finite metrics, valid config lineage, and a schema-valid report. It **does not imply** `physics_status=VALIDATED`, that any provisional accuracy target is met, or that the solver is `PRODUCTION_READY`. CI checks the framework contract and numerical finiteness only; it does not gate on Laplace error, contact-angle error, spurious-current magnitude, or refinement sensitivity.

The provisional targets in the report (mass drift 0.1%, Laplace error 5%, contact-angle MAE 5 degrees / maximum 10 degrees, key-observable refinement change 3%) are held fixed as next-stage engineering goals, not thresholds tuned to the existing results. Use `--strict-targets` only when a nonzero exit on missed readiness targets is desired.
