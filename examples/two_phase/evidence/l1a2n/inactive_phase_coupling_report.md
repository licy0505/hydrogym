# L1A-2n inactive phase-state coupling root-cause audit

- **Status:** `complete`
- **Profile:** `forensic`
- **Final root cause:** `INACTIVE_COUPLING_BACKGROUND_ONLY`
- **Mechanism stage:** `CAPILLARY_STENCIL_LEAKAGE`
- **60-degree specific:** `False`
- **Solver contract:** `11` (unchanged)
- **Production semantics changed:** `False`
- **Inherited mechanism:** `INACTIVE_STATE_COUPLING` (confirmed in L1A-2m)

## Matched-control sensitivity

- first-stage read (storage read stencil): S_60 = `1.066667e+01`, S_90 = `1.066667e+01`, S_150 = `1.066667e+01`
- capillary force: S_60 = `6.400460e-02`, S_90 = `9.561766e-02`, S_150 = `1.469493e-01`, S_60/S_90 = `0.6693804694902064`, S_60/S_150 = `0.4355555349681333`
- one-step contact-line response: `{'authority_060': 1.534e-10, 'control_090': 1.235e-10, 'control_150': 1.362e-10, 'ch_only_equilibrium_060': 0.0}`

## Repair candidates

- `boundary_consistent_ghost_v1`: sensitivity_before = 1.066667e+01, suppressed = True
- `nearest_physical_extension_v1`: sensitivity_before = 1.066667e+01, suppressed = True
- `one_sided_cap_closure_v1`: sensitivity_before = 1.066667e+01, suppressed = True

## Mechanism matrix

| Candidate | Status |
|---|---|
| `CHEMICAL_POTENTIAL_STENCIL_LEAKAGE` | `FALSIFIED` |
| `WALL_WETTING_GHOST_COUPLING` | `FALSIFIED` |
| `CAPILLARY_STENCIL_LEAKAGE` | `SUPPORTED` |
| `PHASE_TRANSPORT_GHOST_COUPLING` | `FALSIFIED` |
| `PROPERTY_INTERPOLATION_LEAKAGE` | `FALSIFIED` |
| `BRINKMAN_AMPLIFIED_INACTIVE_COUPLING` | `SUSPECTED` |
| `PERIODIC_SEAM_INACTIVE_COUPLING` | `FALSIFIED` |
| `MULTIPLE_OPERATOR_LEAKAGE` | `FALSIFIED` |
| `INACTIVE_COUPLING_BACKGROUND_ONLY` | `SUPPORTED` |

## Blockers

- `N-INACTIVE-PHASE-STATE-COUPLING` = {'status': 'confirmed_problem_in_contract_v11', 'created_in': 'L1A-2n', 'resolved_in_this_stage': False, 'note': 'L1A-2m discovery transferred here; the first causal operator is measured in this stage'}
- `N-STATIONARITY-METRIC-DOMAIN` = {'status': 're_examined_in_contract_v11', 'zero_volume_rate_inflation': 'falsified', 'transferred_to': 'N-INACTIVE-PHASE-STATE-COUPLING'}
- `N-CH-MASS-PRECISION` = resolved_in_contract_v11
- `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN` = resolved_in_contract_v9 (contract-11 regression confirmed)
- `N-CAPILLARY-PRESSURE-BALANCE` = structural background
- `W-CONTACT-ANGLE` = open
- `final_root_cause_recorded` = INACTIVE_COUPLING_BACKGROUND_ONLY

## Unmeasured sections

- `short_diagnostic_continuation`: {'measured': False, 'reason': 'DIAGNOSTIC_SHORT_CONTINUATION is only run on a plausible 60-degree-specific pathway'}
- `production_repair`: {'measured': False, 'reason': 'a production repair is a separate later stage (section 50)'}
- `contract_bump_decision`: {'measured': False, 'reason': 'explicitly deferred to the later repair stage (section 52)'}
