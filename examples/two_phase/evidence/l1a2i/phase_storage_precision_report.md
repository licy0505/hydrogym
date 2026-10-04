# L1A-2i phase-state storage precision report (quick profile)

- stage: `L1A-2i` (spec `TWO_PHASE_L1A2I_PHASE_STATE_STORAGE_PRECISION_AGENT_SPEC.md`)
- contract version: `10` (unchanged by this audit)
- verdict: **NOT_READY_FOR_SELECTION**
- selection: `None` -- gates not yet measured: long_horizon, medium
- production readiness: NOT READY FOR L1B PRODUCTION SCALE (no candidate cleared every measured gate; GPU cost unverified (no GPU in this environment))
- cost: runtime ratio None (preferred +0.15), memory ratio None -- NOT_MEASURED

## Gates

| gate | value |
|---|---|
| quick_bound | `2e-06` |
| quick_strong_bound | `5e-07` |
| medium_bound | `0.0002` |
| medium_strong_bound | `0.0001` |
| angle_delta_bound_deg | `0.2` |
| long_horizon_projection_bound | `0.001` |
| runtime_preferred_fraction | `0.15` |
| memory_preferred_fraction | `0.2` |
| closure_horizon | `200000` |

## Drift ledger (accumulated over the quick gate)

| candidate | solve defect / E_round | storage loss / E_round | explained drift | measured drift |
|---|---|---|---|---|
| `A0_float32` | -18.8 | -73.1 | -1.095e-05 | -1.212e-05 |
| `A1_phase_float64` | -0.0 | +0.0 | -1.262e-16 | +1.938e-15 |
| `A2_float64_storage_f32_krylov` | -8.1 | +0.0 | -9.690e-07 | -1.188e-06 |
| `B1_compensated` | -20.1 | -0.0 | -2.407e-06 | -3.815e-06 |
| `C1_residual_feedback` | -19.6 | +0.0 | -2.335e-06 | -3.746e-06 |
| `F_full_float64` | +0.0 | +0.0 | +5.204e-16 | +2.154e-16 |

## Candidate rows

| candidate | quick (off 0.0 / 0.5) | medium | angle delta | long horizon | restart | AD | runtime ratio | status |
|---|---|---|---|---|---|---|---|---|
| `A0_float32` | -1.212e-05 / -1.944e-05 | n/a | n/a | n/a | True | True | 1.000 | REFERENCE |
| `A1_phase_float64` | +1.938e-15 / +1.723e-15 | n/a | n/a | n/a | True | True | 1.203 | CANDIDATE |
| `A2_float64_storage_f32_krylov` | -1.188e-06 / -3.989e-06 | n/a | n/a | n/a | True | True | 1.137 | REJECTED |
| `B1_compensated` | -3.815e-06 / -8.405e-06 | n/a | n/a | n/a | True | True | 1.016 | REJECTED |
| `C1_residual_feedback` | -3.746e-06 / -8.306e-06 | n/a | n/a | n/a | None | True | 1.006 | REJECTED |
| `F_full_float64` | +2.154e-16 / +8.615e-16 | n/a | n/a | n/a | None | True | 1.291 | REFERENCE |

## Fixed-state causal test (spec 26)

- N = 48, target = 150 deg, M = 4.0x M_ref, warmup = 300 substeps, `E_round` = 2.4580e-07
- baseline storage residual -0.0001 E_round per update, 0.957 of the updates rounded

| candidate | working dtype | Krylov dtype | solve defect / E_round | storage loss / E_round | input shift / E_round | nonzero |
|---|---|---|---|---|---|---|
| `A0_float32` | float32 | float32 | -0.0141 | -0.0001 | +0.0000 | 0.957 |
| `A1_phase_float64` | float64 | float64 | -0.0000 | +0.0000 | +0.0408 | 0.000 |
| `A2_float64_storage_f32_krylov` | float64 | float32 | -0.0141 | +0.0000 | +0.0267 | 0.000 |
| `B1_compensated` | float32 | float32 | -0.0141 | -0.0001 | +0.0000 | 0.957 |
| `C1_residual_feedback` | float32 | float32 | -0.0141 | -0.0001 | +0.0000 | 0.957 |
| `F_full_float64` | float64 | float64 | -0.0000 | +0.0000 | +0.0408 | 0.000 |

Baseline storage residual by cell population (E_round):

- `gas`: +0.0001
- `near_gas`: +0.0013
- `interface`: +0.0083
- `near_liquid`: +0.0056
- `liquid`: -0.0155
- `cut`: -0.0000
- `full`: -0.0001
- `fluid`: -0.0001

## Dataset and restart lineage (spec 52)

- `A0_float32`: no new dataset field; the solver source hash and the contract-11 metadata already invalidate every contract-10 dataset, and the checkpoint layout is unchanged
- `A1_phase_float64`: no new dataset field; the solver source hash and the contract-11 metadata already invalidate every contract-10 dataset, and the checkpoint layout is unchanged
- `A2_float64_storage_f32_krylov`: no new dataset field; the solver source hash and the contract-11 metadata already invalidate every contract-10 dataset, and the checkpoint layout is unchanged
- `B1_compensated`: storage model must be added to the trajectory fingerprint and to phase_transport_metadata; the restart state version must be bumped (old checkpoints cannot be resumed)
- `C1_residual_feedback`: storage model must be added to the trajectory fingerprint and to phase_transport_metadata; the restart state version must be bumped (old checkpoints cannot be resumed)
- `F_full_float64`: no new dataset field; the solver source hash and the contract-11 metadata already invalidate every contract-10 dataset, and the checkpoint layout is unchanged

## Notes

- CPU-only environment: GPU cost is not verified (no GPU device present). The performance section is a CPU measurement and is reported as such.
- numpy.polyfit's RankWarning about its step-axis conditioning is silenced in series_classification (the coefficients are unchanged, the warning is about the x scale).
- The audit's A0 row uses the full CHNS substep chain (momentum, Brinkmann damping and pressure projection on, as in step_with_diagnostics); the L1A-2h closure ledger's CH-only fixture is reproduced by the same chain to within 3.1e-07 relative drift at 2500 public steps, the difference being the phase advection the momentum path adds.

