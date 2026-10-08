# L1A-2r — impact-impulse retention and projection-Brinkman causal audit

- git sha: `21ec435d15f21260f4073b9d47d7c95a7bbab323` · solver contract: **12** · policy: `impact_phase_cap_dx2_v1`
- diagnostic only: contract 12 frozen, no production change, no threshold relaxed

## Verdicts (sections 22/23)

- root cause: **BRINKMAN_PROJECTION_COUPLING**
- impact event: **INCONCLUSIVE**
- next stage: **B**

## Ledger identity checks (sections 6/8)

- ledger reproduces the production public step within dtype tolerance: `True` (max |du| 9.12e-08)
- periodic projection-mean identity: cancels=`True` (max |mean(-h·G(P))| = 1.779e-11; Brinkman-driven plain-grid mean change per substep = 3.019e-03, section 16 accounting)
- Poisson residual (actual D/G/m2_proj): max 4.552e-03

## Negative controls (section 12)

- A_empty_solid_uniform: {"projection_only_max_abs_change_v": 0.0, "projection_preserves_uniform": true, "full_step_max_abs_change_v": 5.900859832763672e-06, "full_step_preserves_uniform": true, "divergence_initial": 0.0}
- B_projection_uniform_with_solid_present: {"max_abs_change_v": 0.0, "projection_alone_preserves_uniform": true}
- C_zero_velocity_force_free: {"max_abs_v_over_0_08": 0.00020517582015600055, "spontaneous_fraction_of_u_impact": 0.0004103516403120011, "no_spontaneous_startup": true}
- D_pure_gradient_field: {"div_lininf_before": 233.16713905334473, "div_lininf_after_projection": 7.05718994140625e-05, "reduced": true}

## Initialization variants (section 14)

- UNIFORM_ALL_DOMAIN: div=0.000e+00, v_liquid=-0.5000, v_core=-0.5000, |v|_solid=0.5000
- STREAMFUNCTION_LOCALIZED: div=3.815e-06, v_liquid=-0.4990, v_core=-0.4999, |v|_solid=1.0000
- GAS_AT_REST_CONTROL: div=8.000e+00, v_liquid=-0.4647, v_core=-0.5000, |v|_solid=0.0000
- SOLID_COMPATIBLE_INITIAL: div=8.000e+00, v_liquid=-0.5000, v_core=-0.5000, |v|_solid=0.0000

## Impact authentication (section 23)

- flat_we100_ct050: **IMPACT_AUTHENTICATED** contact_t=0.17400000000000002 v_liquid_precontact=-0.300847070329903 retention_at_contact=0.5944755277202775
- flat_we200_ct000: **IMPACT_AUTHENTICATED** contact_t=0.186 v_liquid_precontact=-0.28997363695536366 retention_at_contact=0.5729735846495383
- pillar_training: **NO_CONTACT_OR_HOVER** contact_t=None v_liquid_precontact=None retention_at_contact=0.20479536401723591
- complex_heldout: **NO_CONTACT_OR_HOVER** contact_t=None v_liquid_precontact=None retention_at_contact=0.2020809365247083

## Section 3 answers

1. **Real approach state attained?** At N=192 the two flat canaries reach contact with a signed downward precontact velocity of 57-60% of u_impact (we100 t=0.174, we200 t=0.186): an approach state is attained but 40-43% of the requested impulse is lost before contact. pillar/complex never contact within the diagnostic horizon (retention decays to ~20%: hover).
2. **Which operator changes impulse first?** The momentum rhs and capillary branches change v at O(1e-5) per substep; the first O(1e-3) change is the Brinkman damping of the solid-region velocity (F_NO_MOMENTUM_RHS: grid l2 3.0e-3 per substep), and the projection then redistributes that change globally on the periodic grid (liquid-weighted v shift -0.49698 vs -0.49992 with projection disabled).
3. **Brinkman -> divergence -> projection redistribution?** Measured directly: the Brinkman factor damps only where chi>0 (damp_min 0.857 at N=192), creating a divergence field whose constant mode the Poisson solve removes from the non-null spectrum; the correction -h·G(P) has exact zero periodic mean (2.2e-10) and therefore does NOT destroy global grid momentum - it redistributes locally between solid-adjacent and interior cells (section 8 distinction).
4. **dt-sensitivity origin?** With NO solid the uniform impulse is preserved exactly (-0.5000 for all dt through t=0.24): the sink requires the solid. With the solid present the retained impulse at t=0.08 is dt-dependent (see table below): refinement REMOVES more impulse. A solid-compatible (streamfunction) seed cuts the spread from 35% to 4.3% across dt=0.002 -> 0.0005: the incompatibility of the uniform all-domain seed with the no-slip/Brinkman solid is the dominant dt-sensitivity source, not the momentum time integrator alone.
5. **ONE next action (decision B, diagnostic)**: adopt the streamfunction-localized (solid-compatible, return-flow-aware) initial velocity as the impact-case initialization in a FOLLOW-UP diagnostic contract (not a silent production change), and re-run this audit: if retention at first contact becomes dt-insensitive within the 3% gate, the generator can simulate real impacts; if not, the formulation itself is unsuitable for impact data.

## Temporal dt-sensitivity (section 19, v_liquid weighted, N=192)

- UNIFORM_ALL_DOMAIN: dt=0.0005: v(0.08)=-0.1899, v(0.24)=-0.0275; dt=0.001: v(0.08)=-0.3081, v(0.24)=-0.1171; dt=0.002: v(0.08)=-0.3925, v(0.24)=-0.2421
- EMPTY_SOLID_UNIFORM: dt=0.0005: v(0.08)=-0.5000, v(0.24)=-0.5000; dt=0.001: v(0.08)=-0.5000, v(0.24)=-0.5000; dt=0.002: v(0.08)=-0.5000, v(0.24)=-0.5000
- STREAMFUNCTION_LOCALIZED: dt=0.0005: v(0.08)=-0.3857, v(0.24)=-0.3555; dt=0.001: v(0.08)=-0.4006, v(0.24)=-0.3648; dt=0.002: v(0.08)=-0.4199, v(0.24)=-0.3776
- GAS_AT_REST_CONTROL: dt=0.0005: v(0.08)=-0.1745, v(0.24)=-0.1511; dt=0.001: v(0.08)=-0.1893, v(0.24)=-0.1594; dt=0.002: v(0.08)=-0.2048, v(0.24)=-0.1725

## Mechanism matrix (section 21)

- `INITIAL_ALL_DOMAIN_VELOCITY_INCOMPATIBILITY`: **SUSPECTED**
- `INITIAL_GAS_LIQUID_RELATIVE_VELOCITY_MISMATCH`: **FALSIFIED**
- `BRINKMAN_LOCAL_DAMPING`: **SUPPORTED**
- `BRINKMAN_PROJECTION_COUPLING`: **SUPPORTED**
- `PROJECTION_DIRECT_GLOBAL_MOMENTUM_LOSS`: **FALSIFIED**
- `PROJECTION_LOCAL_IMPULSE_REDISTRIBUTION`: **SUPPORTED**
- `PROJECTION_DISCRETE_OPERATOR_MISMATCH`: **FALSIFIED**
- `PERIODIC_Y_TOPOLOGY_COUPLING`: **NOT_TESTED**
- `CONSTANT_DENSITY_PROJECTION_LIMITATION`: **NOT_TESTED**
- `CAPILLARY_STARTUP_RESPONSE`: **SUSPECTED**
- `MOMENTUM_TIME_INTEGRATION_LIMITATION`: **SUSPECTED**
- `MAX_SPEED_METRIC_DOMAIN_MISMATCH`: **SUPPORTED**
- `MULTIPLE_CONTRIBUTORS`: **NOT_TESTED**

## Blockers (unchanged, section 25)

- N-DT TARGET_CRITICAL · SPATIAL_REFINEMENT FAIL · L1B_DATA_NOT_READY · L1A_STATUS BLOCKED
