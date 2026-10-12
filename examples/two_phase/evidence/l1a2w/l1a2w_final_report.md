# L1A-2w final report - boundary-compatible incident-momentum reachability bound at t=0

## A. Scope, authority and baseline
- baseline: PR #24 head `73c230b93f7e2d55fc7142a9b481ae4a45cc43e2`; run head `af291af988f6f2ef39338e559cd22361eaafc146`
- verified main `09e2993b8da478d79e768054194201af57121784`; branch `arena/02b6c71b-hydrogym`; contract 12; policy `impact_phase_cap_dx2_v1`
- profile N=192, dt=0.002, cases ['flat_we100_ct050', 'flat_we200_ct000', 'pillar_training', 'complex_heldout']
- trajectories 0, contact events 0, refinements 0

## B. Frozen contract and prohibitions
- band speed cap 0.02, envelope 4.905526491646905 (both frozen)
- envelope definition: the maximum peak open-cell seed speed the frozen reference family (C0/C1/C2) itself produces in the same cases, sealed in L1A-2u and re-read through the L1A-2v seal; it is a reference-calibrated cap, not a physical threshold
- radial family untouched: re_tuned=False
- prohibitions: {"trajectories": "none; the frozen substep kernel is never constructed", "contact_events": "none", "temporal_refinement": "none", "spatial_refinement": "none", "t8_recomputation": "none", "l1b_data_generation": false, "ml_training": false, "contract_13": false, "historical_evidence_edits": "none", "production_initializer_promotion": "none"}

## C. Sealed evidence verification
- l1a2t: verified=true, bound files 14
- l1a2u: verified=true, bound files 13
- l1a2v: verified=true, bound files 14
- envelope read back from sealed evidence: 4.90553 (match true)

## D. Prespecified bound contract
- `B1_contour_cell_normal_speed`: |v_n| at the contour support cell <= min(envelope, band cap if that cell lies in the chi >= 0.01 band); for any admissible state this is implied by the frozen gate set itself
- `B1b_contour_point_normal_speed`: the bilinear value at the contour crossing point is a convex combination of its four supporting cells, so |v_n(point)| <= sum_c w_c * cap_c with cap_c the per-cell cap; this is the tightest statement available without assuming the point lies in one regime
- `B2_placeable_downward_momentum`: P_down <= v_max * sum_{open, chi<0.01} V_i phi_i : pointwise speed cap times the admissible liquid volume, valid for any admissible field
- `B3_in_band_phase_flux`: downward advective phase flux over in-band faces <= v_band * sum_f A_f phi_upwind,f, because every in-band cell speed is <= v_band so every in-band face speed is too
- `B4_in_band_transport_time`: crossing an in-band normal distance ds takes at least ds / v_band while the initializer constraint set holds
- `B5_shape_ratio_criterion`: for a seed built as psi = sx * envelope * seam * T(sdf) and renormalised so the mandated phi>=0.9 core mean equals -u_impact, the ratio max_band|v| / |core mean v| is a shape invariant independent of the amplitude. A shape whose ratio exceeds the band gate can never be admissible under the mandated core target, whatever its amplitude; this is a necessary rejection that costs no construction
- separation instantaneous vs finite-time: {"instantaneous": "what the t=0 state does at the frozen phi=0.5 contour right now: the normal speed at the contour support cell and at the contour crossing point, plus the frozen face flux there", "finite_time": "what can reach the contour in finite time from the t=0 state. The six gates constrain the initializer, not the evolved solver, so B3/B4 describe the initial configuration's reachability only; they are not a bound on the subsequent trajectory, whose transport is governed by the frozen solver's advection, Brinkman penalization and Cahn-Hilliard dynamics"}

## E. Constraint geometry
- `flat_we100_ct050`: band cells 2112, band sdf range [-7.5, 2.5], contour support cell [95, 10] at sdf/dx=2.5 chi=0.034445 inside band=true, sample-count in band=2
    - liquid volume: open=1.54951, admissible=1.53657, in band=0.0129392; core volume=1.24414 (80.3% of the liquid)
- `flat_we200_ct000`: band cells 2112, band sdf range [-7.5, 2.5], contour support cell [95, 10] at sdf/dx=2.5 chi=0.034445 inside band=true, sample-count in band=2
    - liquid volume: open=1.54951, admissible=1.53657, in band=0.0129392; core volume=1.24414 (80.3% of the liquid)
- `pillar_training`: band cells 2936, band sdf range [-7.5, 3.1906111240386963], contour support cell [95, 23] at sdf/dx=2.7 chi=0.026597 inside band=true, sample-count in band=2
    - liquid volume: open=1.55009, admissible=1.53917, in band=0.0109202; core volume=1.24414 (80.3% of the liquid)
- `complex_heldout`: band cells 4188, band sdf range [-7.5, 3.445622444152832], contour support cell [106, 35] at sdf/dx=5.682 chi=0.00051239 inside band=false, sample-count in band=0
    - liquid volume: open=1.55072, admissible=1.54996, in band=0.000761994; core volume=1.24414 (80.2% of the liquid)

## F. Necessary bounds
- `flat_we100_ct050` [INSIDE_BAND]: B1 |v_n| cell cap=0.01 (0.02 u_impact, band cap), B1b point cap=2.492 u_impact; B2 placeable downward momentum<=3.76884 (core mandates 0.62207); B4 in-band transport time>=UNMEASURED (UNMEASURED_NO_IN_BAND_LEG)
    - B3 in-band budget: 384 faces, area*phi_upwind=0.00035004, flux bound=3.5004e-06
- `flat_we200_ct000` [INSIDE_BAND]: B1 |v_n| cell cap=0.01 (0.02 u_impact, band cap), B1b point cap=2.492 u_impact; B2 placeable downward momentum<=3.76884 (core mandates 0.62207); B4 in-band transport time>=UNMEASURED (UNMEASURED_NO_IN_BAND_LEG)
    - B3 in-band budget: 384 faces, area*phi_upwind=0.00035004, flux bound=3.5004e-06
- `pillar_training` [INSIDE_BAND]: B1 |v_n| cell cap=0.01 (0.02 u_impact, band cap), B1b point cap=2.492 u_impact; B2 placeable downward momentum<=3.77523 (core mandates 0.62207); B4 in-band transport time>=1.53316 (ds / v_band over the in-band leg between the contour cell and the band top)
    - B3 in-band budget: 824 faces, area*phi_upwind=0.00033753, flux bound=3.3753e-06
- `complex_heldout` [OUTSIDE_BAND]: B1 |v_n| cell cap=2.45276 (4.906 u_impact, reference envelope), B1b point cap=4.906 u_impact; B2 placeable downward momentum<=3.80168 (core mandates 0.62207); B4 in-band transport time>=UNMEASURED (UNMEASURED_NO_IN_BAND_LEG)
    - B3 in-band budget: 1372 faces, area*phi_upwind=2.35075e-05, flux bound=2.35075e-07

## G. Measurements on the frozen candidates (t=0)
- `flat_we100_ct050` `C2_BASE`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=4.316, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=0.849781
- `flat_we100_ct050` `R0_WEAK`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=30.97, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=1.57074
- `flat_we100_ct050` `R1_MEDIUM`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=89.77, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=4.4334
- `flat_we100_ct050` `R2_SHARP`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=36.33, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=1.60981
- `flat_we200_ct000` `C2_BASE`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=4.316, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=0.849781
- `flat_we200_ct000` `R0_WEAK`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=30.97, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=1.57074
- `flat_we200_ct000` `R1_MEDIUM`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=89.77, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=4.4334
- `flat_we200_ct000` `R2_SHARP`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=36.33, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=1.60981
- `pillar_training` `C2_BASE`: contour cell v_n/U=-0 (point -0), band speed/U=1.5613e-05, band/core ratio=1.56e-05, peak/U=4.225, in-band flux=1.94617e-11 (face max/U=5.583e-06), gates=true, core v/U=-1, P_down=0.855754
- `pillar_training` `R0_WEAK`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=30.97, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=1.57074
- `pillar_training` `R1_MEDIUM`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=89.77, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=4.4334
- `pillar_training` `R2_SHARP`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=36.33, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=1.60981
- `complex_heldout` `C2_BASE`: contour cell v_n/U=0.33539 (point 0.24841), band speed/U=0.00105894, band/core ratio=0.00106, peak/U=4.213, in-band flux=1.04621e-15 (face max/U=0.00091123), gates=true, core v/U=-1, P_down=0.85706
- `complex_heldout` `R0_WEAK`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=30.97, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=1.57074
- `complex_heldout` `R1_MEDIUM`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=89.77, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=4.4334
- `complex_heldout` `R2_SHARP`: contour cell v_n/U=-0 (point -0), band speed/U=0, band/core ratio=0, peak/U=36.33, in-band flux=0 (face max/U=0), gates=true, core v/U=-1, P_down=1.60981

## H. Discrete verification of every bound
- holds on every admissible state: true; rows 22 (10 admissible), inequalities 60; violations []
    - exempted as inadmissible (hypothesis fails, not a violation): complex_heldout/R0_WEAK - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): complex_heldout/R1_MEDIUM - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): complex_heldout/R2_SHARP - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): flat_we100_ct050/R0_WEAK - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): flat_we100_ct050/R1_MEDIUM - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): flat_we100_ct050/R2_SHARP - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): flat_we200_ct000/R0_WEAK - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): flat_we200_ct000/R1_MEDIUM - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): flat_we200_ct000/R2_SHARP - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): pillar_training/R0_WEAK - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): pillar_training/R1_MEDIUM - global reference envelope exceeded
    - exempted as inadmissible (hypothesis fails, not a violation): pillar_training/R2_SHARP - global reference envelope exceeded
    - closest approach to a bound: R2_SHARP -> 0.2267 of the bound
    - closest approach to a bound: R2_SHARP -> 0.2255 of the bound
    - closest approach to a bound: R2_SHARP -> 0.2255 of the bound
    - closest approach to a bound: R2_SHARP -> 0.2254 of the bound
    - closest approach to a bound: R2_SHARP -> 0.06837 of the bound
    - closest approach to a bound: R2_SHARP -> 0.05295 of the bound

## I. Attainability: prespecified ramp ladder and exploratory appendix
- criterion: band_core_ratio > band cap => the shape can never be admissible at the mandated core target, whatever the amplitude, so it is rejected analytically without constructing a state
- arms: ['PRESPECIFIED_PRIMARY', 'EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS']
- declared before measurement: ['PRESPECIFIED_PRIMARY']
- added after seeing the primary arm: ['EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS']
- `flat_we100_ct050` (u_impact=0.5): cell cap=0.02 u_impact, point cap=2.492 u_impact; analytically rejected 19 shapes; admissible constructions ['RAMP_A3.5_W3', 'RAMP_A4_W8']; best `None` (arm None) = cell 0 (0% of the cell cap), point 0 (0% of the point cap)
    - arm `PRESPECIFIED_PRIMARY`: screened 10, rejected analytically 10, constructed 0, admissible 0, min band/core ratio among constructed=UNMEASURED (gate 0.02)
    - arm `EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS`: screened 11, rejected analytically 9, constructed 2, admissible 2, min band/core ratio among constructed=0 (gate 0.02)
    - `RAMP_A0_W4` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=2.023 > cap 0.02
    - `RAMP_A0_W6` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=1.464 > cap 0.02
    - `RAMP_A0_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.881 > cap 0.02
    - `RAMP_A0_W10` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.5479 > cap 0.02
    - `RAMP_A0_W12` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.3603 > cap 0.02
    - `RAMP_A0_W16` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.1801 > cap 0.02
    - `RAMP_A0_W24` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.06689 > cap 0.02
    - `RAMP_A0_W32` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.03426 > cap 0.02
    - `RAMP_A0.5_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.6614 > cap 0.02
    - `RAMP_A1_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.4465 > cap 0.02
    - `RAMP_A2_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=2.227 > cap 0.02
    - `RAMP_A2_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.242 > cap 0.02
    - `RAMP_A2_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.6838 > cap 0.02
    - `RAMP_A2.5_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.242 > cap 0.02
    - `RAMP_A2.5_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.5214 > cap 0.02
    - `RAMP_A2.5_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.2574 > cap 0.02
    - `RAMP_A3_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.2572 > cap 0.02
    - `RAMP_A3_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.08821 > cap 0.02
    - `RAMP_A3_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.03998 > cap 0.02
    - `RAMP_A3.5_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: contour cell v_n/U=0, point v_n/U=-0, band/U=0, peak/U=4.151, gates=true, envelope_ok=true, admissible=true
    - `RAMP_A4_W8` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: contour cell v_n/U=0, point v_n/U=-0, band/U=0, peak/U=4.316, gates=true, envelope_ok=true, admissible=true
- `flat_we200_ct000` (u_impact=0.5): cell cap=0.02 u_impact, point cap=2.492 u_impact; analytically rejected 19 shapes; admissible constructions ['RAMP_A3.5_W3', 'RAMP_A4_W8']; best `None` (arm None) = cell 0 (0% of the cell cap), point 0 (0% of the point cap)
    - arm `PRESPECIFIED_PRIMARY`: screened 10, rejected analytically 10, constructed 0, admissible 0, min band/core ratio among constructed=UNMEASURED (gate 0.02)
    - arm `EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS`: screened 11, rejected analytically 9, constructed 2, admissible 2, min band/core ratio among constructed=0 (gate 0.02)
    - `RAMP_A0_W4` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=2.023 > cap 0.02
    - `RAMP_A0_W6` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=1.464 > cap 0.02
    - `RAMP_A0_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.881 > cap 0.02
    - `RAMP_A0_W10` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.5479 > cap 0.02
    - `RAMP_A0_W12` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.3603 > cap 0.02
    - `RAMP_A0_W16` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.1801 > cap 0.02
    - `RAMP_A0_W24` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.06689 > cap 0.02
    - `RAMP_A0_W32` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.03426 > cap 0.02
    - `RAMP_A0.5_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.6614 > cap 0.02
    - `RAMP_A1_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.4465 > cap 0.02
    - `RAMP_A2_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=2.227 > cap 0.02
    - `RAMP_A2_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.242 > cap 0.02
    - `RAMP_A2_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.6838 > cap 0.02
    - `RAMP_A2.5_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.242 > cap 0.02
    - `RAMP_A2.5_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.5214 > cap 0.02
    - `RAMP_A2.5_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.2574 > cap 0.02
    - `RAMP_A3_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.2572 > cap 0.02
    - `RAMP_A3_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.08821 > cap 0.02
    - `RAMP_A3_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.03998 > cap 0.02
    - `RAMP_A3.5_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: contour cell v_n/U=0, point v_n/U=-0, band/U=0, peak/U=4.151, gates=true, envelope_ok=true, admissible=true
    - `RAMP_A4_W8` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: contour cell v_n/U=0, point v_n/U=-0, band/U=0, peak/U=4.316, gates=true, envelope_ok=true, admissible=true
- `pillar_training` (u_impact=0.5): cell cap=0.02 u_impact, point cap=2.492 u_impact; analytically rejected 20 shapes; admissible constructions ['RAMP_A4_W8']; best `None` (arm None) = cell 0 (0% of the cell cap), point 0 (0% of the point cap)
    - arm `PRESPECIFIED_PRIMARY`: screened 10, rejected analytically 10, constructed 0, admissible 0, min band/core ratio among constructed=UNMEASURED (gate 0.02)
    - arm `EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS`: screened 11, rejected analytically 10, constructed 1, admissible 1, min band/core ratio among constructed=1.5613e-05 (gate 0.02)
    - `RAMP_A0_W4` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=1.496 > cap 0.02
    - `RAMP_A0_W6` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=1.475 > cap 0.02
    - `RAMP_A0_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=1.011 > cap 0.02
    - `RAMP_A0_W10` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.665 > cap 0.02
    - `RAMP_A0_W12` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.4496 > cap 0.02
    - `RAMP_A0_W16` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.2291 > cap 0.02
    - `RAMP_A0_W24` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.08379 > cap 0.02
    - `RAMP_A0_W32` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.04147 > cap 0.02
    - `RAMP_A0.5_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.8435 > cap 0.02
    - `RAMP_A1_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.6544 > cap 0.02
    - `RAMP_A2_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=2.478 > cap 0.02
    - `RAMP_A2_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=2.063 > cap 0.02
    - `RAMP_A2_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.324 > cap 0.02
    - `RAMP_A2.5_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=2.446 > cap 0.02
    - `RAMP_A2.5_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.367 > cap 0.02
    - `RAMP_A2.5_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.7551 > cap 0.02
    - `RAMP_A3_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.365 > cap 0.02
    - `RAMP_A3_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.5807 > cap 0.02
    - `RAMP_A3_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.2886 > cap 0.02
    - `RAMP_A3.5_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.1109 > cap 0.02
    - `RAMP_A4_W8` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: contour cell v_n/U=0, point v_n/U=-0, band/U=1.5613e-05, peak/U=4.225, gates=true, envelope_ok=true, admissible=true
- `complex_heldout` (u_impact=0.5): cell cap=4.906 u_impact, point cap=4.906 u_impact; analytically rejected 20 shapes; admissible constructions ['RAMP_A4_W8']; best `RAMP_A4_W8` (arm EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS) = cell 0.33539 (6.84% of the cell cap), point 0.24841 (5.06% of the point cap)
    - arm `PRESPECIFIED_PRIMARY`: screened 10, rejected analytically 10, constructed 0, admissible 0, min band/core ratio among constructed=UNMEASURED (gate 0.02)
    - arm `EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS`: screened 11, rejected analytically 10, constructed 1, admissible 1, min band/core ratio among constructed=0.0010589 (gate 0.02)
    - `RAMP_A0_W4` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=2 > cap 0.02
    - `RAMP_A0_W6` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=1.967 > cap 0.02
    - `RAMP_A0_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=1.352 > cap 0.02
    - `RAMP_A0_W10` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.8902 > cap 0.02
    - `RAMP_A0_W12` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.603 > cap 0.02
    - `RAMP_A0_W16` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.3088 > cap 0.02
    - `RAMP_A0_W24` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.1132 > cap 0.02
    - `RAMP_A0_W32` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.05553 > cap 0.02
    - `RAMP_A0.5_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=1.125 > cap 0.02
    - `RAMP_A1_W8` [PRESPECIFIED_PRIMARY]: REJECTED analytically, band/core ratio=0.8717 > cap 0.02
    - `RAMP_A2_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=2.959 > cap 0.02
    - `RAMP_A2_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=2.564 > cap 0.02
    - `RAMP_A2_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.716 > cap 0.02
    - `RAMP_A2.5_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=2.876 > cap 0.02
    - `RAMP_A2.5_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.797 > cap 0.02
    - `RAMP_A2.5_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.034 > cap 0.02
    - `RAMP_A3_W2` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=1.941 > cap 0.02
    - `RAMP_A3_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.8929 > cap 0.02
    - `RAMP_A3_W4` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.4569 > cap 0.02
    - `RAMP_A3.5_W3` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: REJECTED analytically, band/core ratio=0.2357 > cap 0.02
    - `RAMP_A4_W8` [EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS]: contour cell v_n/U=0.33539, point v_n/U=0.24841, band/U=0.00105893, peak/U=4.213, gates=true, envelope_ok=true, admissible=true

## J. Case classification
- contour inside the chi band: ['flat_we100_ct050', 'flat_we200_ct000', 'pillar_training']
- contour outside the chi band: ['complex_heldout']
    - `flat_we100_ct050`: CONTOUR_INSIDE_CHI_GATE_BAND, binding constraint at the contour `band_speed_cap`, contour sdf/dx=2.5, band top sdf/dx=2.5, point cap=2.492 u_impact
    - `flat_we200_ct000`: CONTOUR_INSIDE_CHI_GATE_BAND, binding constraint at the contour `band_speed_cap`, contour sdf/dx=2.5, band top sdf/dx=2.5, point cap=2.492 u_impact
    - `pillar_training`: CONTOUR_INSIDE_CHI_GATE_BAND, binding constraint at the contour `band_speed_cap`, contour sdf/dx=2.7, band top sdf/dx=3.191, point cap=2.492 u_impact
    - `complex_heldout`: CONTOUR_OUTSIDE_CHI_GATE_BAND, binding constraint at the contour `global_speed_envelope`, contour sdf/dx=5.682, band top sdf/dx=3.446, point cap=4.906 u_impact

## K. Decision
- verdict `BOUND_PROVEN_NECESSARY_ONLY_CONSTRUCTION_BELOW_THE_CAP`
- headline: At t=0 the frozen initializers place exactly zero normal fluid velocity at the frozen phi=0.5 contact contour in 3 of the 3 in-band cases (contour support cell |v_n| = 0.000000 u_impact; in-band downward phase flux at or below 2e-11), while the frozen band gate would permit up to 0.02 u_impact there. Of the 21 shapes screened in the frozen design language (84 shape-case evaluations), none is admissible while delivering normal flow to that cell: the shapes whose in-band speed ratio comes closest to the gate are rejected analytically at ratio 0.03426 > 0.02 (and those already exceed the reference envelope), while the admissible shapes keep the band quiet (band/core ratio 0) and place |v_n| <= 0 u_impact at the contour cell. That is a measured tension between the frozen band gate and the reference envelope inside this design language, not a proof of global infeasibility
    - `flat_we100_ct050` [INSIDE_BAND]: cell cap=0.02 u_impact, point cap=2.492 u_impact; frozen best contour cell |v_n|=0, frozen best point |v_n|=0 (0% of the point cap); constructed best cell=0 (0% of the cell cap), constructed best point=0 (0% of the point cap)
    - `flat_we200_ct000` [INSIDE_BAND]: cell cap=0.02 u_impact, point cap=2.492 u_impact; frozen best contour cell |v_n|=0, frozen best point |v_n|=0 (0% of the point cap); constructed best cell=0 (0% of the cell cap), constructed best point=0 (0% of the point cap)
    - `pillar_training` [INSIDE_BAND]: cell cap=0.02 u_impact, point cap=2.492 u_impact; frozen best contour cell |v_n|=0, frozen best point |v_n|=0 (0% of the point cap); constructed best cell=0 (0% of the cell cap), constructed best point=0 (0% of the point cap)
    - `complex_heldout` [OUTSIDE_BAND]: cell cap=4.906 u_impact, point cap=4.906 u_impact; frozen best contour cell |v_n|=0.33539, frozen best point |v_n|=0.24841 (5.06% of the point cap); constructed best cell=0.33539 (6.84% of the cell cap), constructed best point=0.24841 (5.06% of the point cap)
- instantaneous vs finite time: {"instantaneous": "B1/B1b are evaluated on the t=0 state at the frozen phi=0.5 contour cell and crossing point, and are checked there for every admissible state. In the three in-band cases the frozen families place exactly zero normal velocity at that cell, so the initial state exerts no normal advection on the contact contour at all; the in-band downward phase flux is likewise zero to machine level", "finite_time": "B3/B4 describe what the initial configuration could deliver while the initializer constraint set holds. They are not a bound on the evolved solver: after t=0 the frozen advection, Brinkman penalization and Cahn-Hilliard dynamics transport the interface, and nothing in the six initializer gates constrains that transport"}
- constructibility rule: {"attained": ["flat_we100_ct050", "flat_we200_ct000", "pillar_training", "complex_heldout"], "not_found": [], "proven_infeasible": [], "statement": "shapes rejected analytically are rejected only for the mandated core target and the frozen band gate, not in general; a failed search is NOT_FOUND_NOT_PROVEN_INFEASIBLE; this stage holds no mathematical proof of infeasibility for any case, and no construction is promoted", "in_band_cases_attaining_at_least_90pct_of_the_cell_cap": [], "attainment_is_judged_on_the_tight_cell_cap": "for the in-band cases the contour support cell itself is inside the gate band, so its cap 0.02 u_impact is tight; the bilinear point cap is weaker because the crossing point straddles the band edge, and both are reported so that no looser number is used where a tighter one applies", "exploratory_arm_provenance": "the exploratory appendix arm was added after the primary arm showed every wall-onset ramp to be analytically rejected; it carries no promotion rights and is labelled as post hoc in every artefact"}

## L. Adversarial and falsification checks
- `amplified_state_is_refused_and_exempted_correctly`: {"band_max_speed_over_u_impact": 0.0, "band_cap": 0.02, "violates_band_cap": false, "band_gate_passed": true, "peak_over_u_impact": 863.1201122833244, "envelope": 4.905526491646905, "violates_envelope": true, "classified_inadmissible": true, "reason": "frozen six gates failed", "note": "the caps are not vacuous (a 200x
- `band_geometry_is_non_trivial`: {"band_cell_count": 2112, "admissible_open_cell_count": 34752, "contour_interpolation_support": {"95,10": {"weight": 0.2470092824326855, "chi": 0.03444519639015198, "sdf_over_dx": 2.5, "inside_gate_band": true, "open": true, "cell_cap_over_u_impact": 0.02}, "95,11": {"weight": 0.2529907175673145, "chi": 0.0093159591779
- `shape_criterion_rejects_a_gate_violating_shape`: {"band_core_ratio": 1.8650827712843494, "band_cap": 0.02, "rejected": true, "holds": true}
- `checker_refuses_a_fabricated_violation`: {"violations": ["B1b_contour_point_normal_speed[flat_we100_ct050/FABRICATED]", "B2_placeable_downward_momentum[flat_we100_ct050/FABRICATED]", "B3_in_band_flux[flat_we100_ct050/FABRICATED]"], "discriminates": true, "note": "a checker that accepted these would make the whole bound verification vacuous"}
- `no_trajectory_material_in_the_evidence`: {"trajectories_run": 0, "contact_events_measured": 0, "refinement_studies_run": 0, "holds": true}
- infeasibility rule: constructions that were not found are labelled NOT_FOUND_NOT_PROVEN_INFEASIBLE, and shapes rejected by the band-gate ratio are rejected only for the mandated core target; the only impossibility claims in this stage are the derived bounds themselves

## M. Quality, tests and hosted CI
- quality baseline `52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b`, declared new diagnostics 0, expectation `PASS_BASELINE_ONLY`
- frozen adjacent failure `test_l1a2r_merged_base_verified_or_fail_closed`: FROZEN_LINEAGE_DRIFT_CONFIRMED - pre-existing; reproduced on a pristine worktree at the verified main; not edited
- frozen adjacent failure `tests/test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight`: SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION - the frozen L1A-2t preflight forbids any file or commit from a later stage on the same branch, which cannot hold once L1A-2u/2v/2w exist; not edited
- frozen adjacent failure `tests/test_interior_speed_admissibility_screen_audit.py (5 tests)`: SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION - the frozen L1A-2v preflight requires every commit above its own baseline to touch only L1A-2v paths and the worktree to be clean outside them; the L1A-2w stage commit necessarily breaks that rule, so test_required_evidence_files_are_declared_and_complete, test_manifest_self_verification_and_tamper_detection, test_evidence_verification_rejects_a_moved_envelope, test_reseal_records_the_observed_ci_status_without_touching_physics and test_report_carries_every_required_section fail on the stage commit; not edited
- frozen adjacent suite run (the L1A-2w stage commit, with the stage test file, module and evidence committed and the worktree clean outside the stage paths): 6 failed, and every other collected test passed; the pass count is not recorded because it moves with the stage test count, while the failing set is structural (suite failures L1A-2t 1, L1A-2u 0, L1A-2v 5, L1A-2w 0); reconfirmed at the re-seal head: UNMEASURED; cause: frozen later-stage lineage rules, not a physics or protocol difference; action: recorded; no frozen test edited
    - failing: tests/test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section
- not run: ['trajectories', 'contact events', 'temporal refinement', 'spatial refinement', 'T=8 recompute', 'L1B data generation', 'ML training', 'contract 13']

## N. Exactly one next action
- Resolve the one question this t=0 bound raises and cannot answer: the frozen seeds place exactly no normal fluid motion at the contact contour in the in-band cases, the frozen band gate would permit up to the band cap there, and none of the screened same-language shapes is admissible while delivering that flow. Measure, in a single short window, whether the *interface* transport (frozen Cahn-Hilliard flux and the resulting contour motion) or fluid advection produces the observed L1A-2s contact signature. Run exactly one frozen-solver trajectory for the frozen pillar_training case at N=192, dt=0.002 over t <= 0.08 only, and report the per-cell split of the contour's normal motion into advective and Cahn-Hilliard contributions with the frozen event rule unchanged; no refinement, no new initializer, no generator.
- bounded scope: one case (pillar_training), one dt (0.002), one window (t <= 0.08), frozen solver and frozen event rule, attribution readout only; no refinement, no generator, no ML
- must not be started in this stage: true

