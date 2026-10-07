# L1A-2l phase-coupling and relaxation forensics

- **Status:** `complete`
- **Profile:** `forensic`
- **Final root-cause label:** `INCONCLUSIVE`
- **Solver contract:** `11` (unchanged)
- **Production semantics changed:** `False`
- **Diagnostic-only:** yes; none of these measurements are production acceptance evidence.

## Frozen production conditions

`M=0.002`, `dt=0.004`, phase-only float64 storage, Young surface energy, cut-cell geometry, momentum/projection, Brinkman defaults, and all acceptance thresholds remain unchanged.

## Required state provenance

The raw L1A-2j/L1A-2k state arrays were absent at execution. Production trajectories were rehydrated from their exact seeds and are accepted only when all four per-field hashes match the saved L1A-2k report. Any mismatch rejects that state for causal comparisons.

### authority_060

- State acceptance: `accepted_exact_rehydration` at step `50000`.
- Per-field hash match to frozen L1A-2k: `True`.
- Configuration fingerprint: `2563f20daad7bcdcbb11150fc54daa6a66ef3ff04326b7434d01c0560d5c495f`.

### control_090

- State acceptance: `accepted_exact_rehydration` at step `27200`.
- Per-field hash match to frozen L1A-2k: `True`.
- Configuration fingerprint: `c7857ca99258032c79ba00bfb9de8d24e5e5ef905fd551c854625a81d9ecdea0`.

### control_150

- State acceptance: `accepted_exact_rehydration` at step `100000`.
- Per-field hash match to frozen L1A-2k: `True`.
- Configuration fingerprint: `82a53cc0e10d6ec8000660b6f4e4632bf69c69c61c566ee43eedb8f15677c730`.

## Production-gate normalization

The exact phase-rate field is `nwa._sample.phase_rate_l2`: `sqrt(sum((phi_final - phi_before_last_public_step)^2 / dt^2) * dx * dy)`.
Normalization is `cell area dx*dy (not V_i); rate uses the exact production public-step delta and dt`; mask: `none; every grid cell including solid/zero-volume cells participates exactly as in nwa._sample`; `dt=0.004`; M-ref threshold `0.001`.
Cut-cell-volume-weighted forensic norms are reported separately and never replace this gate.

## Decomposed windows

| Window | Exact production endpoint | Samples | Whole-fluid C_dir (last) | C_mag (last) | Advective L2 (last) | CH L2 (last) | Net L2 (last) |
|---|---:|---:|---:|---:|---:|---:|---:|
| burst_10_step_48k_50k | True | 200 | -0.9572474720011884 | 0.14653876131589144 | 0.0011083800909562694 | 0.0010964356308540385 | 0.00026648794258370595 |
| control_090_late_stationary_1000_step_samples | True | 10 | -0.8610556301451253 | 0.2645884148507427 | 0.001487085100627046 | 0.0015464817913849694 | 0.0006695464870462916 |
| control_150_late_stationary_1000_step_samples | True | 10 | -0.9391102797020134 | 0.17570630669657544 | 0.0013441939550241499 | 0.0013876916366871803 | 0.0003972954354268629 |
| one_step_50000 | True | 1 | -0.9572552112923786 | 0.14652558250753975 | 0.001108381839551122 | 0.0010964371811421126 | 0.00026646480370739984 |
| window_1000_step_40k_50k | True | 10 | -0.9565372954503907 | 0.1477432304609049 | 0.001108226654235975 | 0.001096295685620454 | 0.00026860303047184024 |

## Equilibrium and counterfactuals

- CH-only 60° reference: `accepted_provenance_checked_l1a2k_reference`; step `180000`; gate `{'angle_ok': True, 'angle_window_spread_deg': 0.09537505932090085, 'converged': True, 'energy_ok': True, 'energy_stationary_strict': False, 'energy_window_max_rel_change': 3.8545370993903205e-06, 'rate_ok': True, 'speed_ok': True, 'window_mobility_time': 0.05, 'window_samples_used': 7}`.
- Raw, unaligned authority-to-equilibrium distance: `0.060347868156744186`.
- Freeze-u M_ref continuation: `converged`; steps `7000`; gate `{'angle_ok': True, 'angle_window_spread_deg': 0.011577805653850248, 'converged': True, 'energy_ok': True, 'energy_stationary_strict': False, 'energy_window_max_rel_change': 8.208367574163988e-05, 'rate_ok': True, 'speed_ok': True, 'window_mobility_time': 0.05, 'window_samples_used': 7}`.
- Matched production-velocity vs zero-velocity replay: `measured`; steps `100`.

## Candidate matrix

| Candidate | Status | Scope |
|---|---|---|
| `STATE_PROVENANCE_MISMATCH` | `FALSIFIED` | evidence admissibility blocker only; not a physics root-cause claim |
| `ADVECTIVE_CH_NEAR_CANCELLATION` | `SUPPORTED` | 60-degree differential mechanism candidate; C_dir is a cut-cell-volume cosine and C_mag is the cut-cell-volume norm ratio |
| `HYDRODYNAMICALLY_DRIVEN_PHASE_NONEQUILIBRIUM` | `FALSIFIED` | causal one-step plus matched 100-step production-velocity/zero-velocity counterfactual |
| `MREF_INTRINSIC_CH_RELAXATION_LIMITED` | `FALSIFIED` | matched freeze-u, M_ref continuation only; diagnostic and not production closure |
| `CHEMICAL_POTENTIAL_UNIFORMITY_LIMITATION` | `NOT_TESTED` | no cross-component averaging |
| `N-CAPILLARY-PRESSURE-BALANCE` | `SUPPORTED` | structural background only; explicitly not the 60-degree root cause |
| `N-CH-MASS-PRECISION` | `FALSIFIED` | historical resolved constraint, not reopened |
| `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN` | `FALSIFIED` | historical resolved constraint, not reopened |
| `W-CONTACT-ANGLE` | `NOT_TESTED` | open condition; do not upgrade angle-cycle classification |
| `MULTIPLE_CONTRIBUTORS` | `NOT_TESTED` | composite label is not a way to bypass candidate-level evidence |

**Final root-cause label:** `INCONCLUSIVE` — No sole 60-degree root cause is assigned without provenance-checked, differential causal evidence; the L1A-2k capillary result remains a structural background only.

## Unmeasured sections

- `M_4x_closure`
- `dt_half`

## Quality and provenance

- Runtime: `{'jax': '0.10.2', 'jaxlib': '0.10.2', 'numpy': '2.4.6', 'python': '3.11.2', 'scipy': '1.17.1'}`.
- Source hashes: `{'audit_runner': '1e38222b98943766f9bdd431ef1d2c5258c10546977f6f664f80e9138cf2e64d', 'capillary_audit': '83340302e5d3940cd0b8b5c9a8ba40cceab04a6835e5ce9845954a5ad2d1ea4b', 'capillary_pressure_balance_audit': '6e6b45ce72131a98f156695ee29261e89eb9b4a0a832f837af465b17e5b121ae', 'chns_nonstationarity_audit': '4e74208d191273d617446297d431a813abede74b8b2924043852bfe9b3ee9b1a', 'contact_line_kinetics': 'fca52869eb8fc6c8c327661ecbc485e838b42939b92c297636bbd56a5b1be5c5', 'nonneutral_wetting_audit': 'c1f7570fa06f5ce113cbd7b52fe5ba50afd82f0addf2d0ff695ac9cf1e5139b7', 'observables': '6653e87f2043b9e24b171b4ad50446251b088609404c4f049b1cbde29670b68a', 'phasefield': '4790c6235dd763dbaad5aa838953cda3e39fe95d3e37844b9c4433b2547b3f0e', 'upstream_l1a2j_report': '7734344d07ede28af0142e980a82d551dea43817392ae267645afc750d3ef87f', 'upstream_l1a2k_report': '6254206d2afc33284632af11953537e59fcae48dbc4cb404b6f9dbb1a724a88f'}`.
- Quality status: `{'pytest': {'exit_code': 0, 'stdout_tail': '........                                                                 [100%]\n8 passed in 23.80s\n', 'stderr_tail': '', 'passed': True}, 'ruff': {'exit_code': 0, 'passed': True, 'stdout_tail': 'All checks passed!\n', 'stderr_tail': ''}, 'production_solver_source_integrity': {'hashes_at_audit_import': {'phasefield': '4790c6235dd763dbaad5aa838953cda3e39fe95d3e37844b9c4433b2547b3f0e', 'chns_nonstationarity_audit': '4e74208d191273d617446297d431a813abede74b8b2924043852bfe9b3ee9b1a', 'nonneutral_wetting_audit': 'c1f7570fa06f5ce113cbd7b52fe5ba50afd82f0addf2d0ff695ac9cf1e5139b7'}, 'hashes_after_audit': {'phasefield': '4790c6235dd763dbaad5aa838953cda3e39fe95d3e37844b9c4433b2547b3f0e', 'chns_nonstationarity_audit': '4e74208d191273d617446297d431a813abede74b8b2924043852bfe9b3ee9b1a', 'nonneutral_wetting_audit': 'c1f7570fa06f5ce113cbd7b52fe5ba50afd82f0addf2d0ff695ac9cf1e5139b7'}, 'working_tree_modifications_vs_git_head': ['examples/two_phase/phasefield.py', 'examples/two_phase/production/nonneutral_wetting_audit.py'], 'changed_during_l1a2l': [], 'passed': True, 'interpretation': 'Git-relative modifications may predate L1A-2l; the before/after SHA-256 guard verifies this diagnostic stage did not modify production sources.'}, 'contract_11': 'unchanged', 'production_semantics_changed': False, 'diagnostic_only': True, 'py_compile': {'passed': True, 'scope': 'phase_coupling_relaxation_audit.py and test_phase_coupling_relaxation_audit.py'}}`.
- `N-CH-MASS-PRECISION=resolved_in_contract_v11`, `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN=resolved_in_contract_v9`, `N-CAPILLARY-PRESSURE-BALANCE` as a structural background, and `W-CONTACT-ANGLE=open` are preserved.
- No generic L1A-2k residual is used as a causal 60° explanation; the apparent angle-cycle classification is not upgraded by this report.


### Post-run classifier refresh

- Scope: classifier effect-size field correction and workspace-artifact availability annotation only
- Measurement-run audit source SHA-256: `1e38222b98943766f9bdd431ef1d2c5258c10546977f6f664f80e9138cf2e64d`.
- Post-processing audit source SHA-256: `9942fbb810447f8c64d3fa9d52a9d063464615cbb8ac380562f62c54a2a66841`.
- No solver replay or production-state measurement was rerun or altered by this summary correction.