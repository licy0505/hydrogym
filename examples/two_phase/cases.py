"""
Case definitions for two-phase droplet-impact data generation.

The *transfer* question this example addresses: can a surrogate trained on

* **simple** cases -- a droplet falling on a *flat* wall, and on *one or two*
  periodic pillar arrays --
* generalise to **complex** unseen surfaces (random / hierarchical pillars,
  grooves, inclined walls)?

So the train set deliberately contains only flat + a couple of pillar configs,
and the test set contains richer geometries.
"""

from __future__ import annotations

import copy

import numpy as np

# Each case is a dict understood by :mod:`phasefield`.  ``split`` says where the
# case belongs.  The train set is intentionally simple.
CASES = [
    # ------------------------------------------------------------------ TRAIN
    # Flat wall, sweep of Weber number and wettability
    dict(split="train", surface="flat", We=50.0, cos_theta=-0.5, seed=0),
    dict(split="train", surface="flat", We=100.0, cos_theta=-0.5, seed=1),
    dict(split="train", surface="flat", We=200.0, cos_theta=-0.5, seed=2),
    dict(split="train", surface="flat", We=300.0, cos_theta=-0.5, seed=3),
    dict(split="train", surface="flat", We=100.0, cos_theta=0.0, seed=4),
    dict(split="train", surface="flat", We=200.0, cos_theta=0.0, seed=5),
    dict(split="train", surface="flat", We=100.0, cos_theta=0.5, seed=6),
    dict(split="train", surface="flat", We=200.0, cos_theta=0.5, seed=7),
    # A couple of *simple* periodic pillar arrays
    dict(split="train", surface="pillars", n_pillars=4, width=0.3, height=0.4, We=100.0, cos_theta=-0.5, seed=8),
    dict(split="train", surface="pillars", n_pillars=4, width=0.3, height=0.4, We=200.0, cos_theta=-0.5, seed=9),
    dict(split="train", surface="pillars", n_pillars=6, width=0.2, height=0.3, We=100.0, cos_theta=0.0, seed=10),
    dict(split="train", surface="pillars", n_pillars=6, width=0.2, height=0.3, We=200.0, cos_theta=0.0, seed=11),
    dict(split="train", surface="pillars", n_pillars=4, width=0.3, height=0.4, We=100.0, cos_theta=0.5, seed=12),
    # ------------------------------------------------------------------- TEST
    # Unseen random pillar fields (the "complex" surfaces)
    dict(split="test", surface="random_pillars", n_pillars=7, seed=100, We=100.0, cos_theta=-0.5),
    dict(split="test", surface="random_pillars", n_pillars=7, seed=101, We=200.0, cos_theta=-0.5),
    dict(split="test", surface="random_pillars", n_pillars=6, seed=102, We=100.0, cos_theta=0.0),
    dict(split="test", surface="random_pillars", n_pillars=8, seed=103, We=200.0, cos_theta=0.5),
    # Hierarchical two-scale pillars
    dict(split="test", surface="hierarchical", seed=104, We=100.0, cos_theta=-0.5),
    dict(split="test", surface="hierarchical", seed=105, We=200.0, cos_theta=0.0),
    # Grooved surfaces
    dict(split="test", surface="grooves", seed=106, We=100.0, cos_theta=-0.5),
    dict(split="test", surface="grooves", seed=107, We=200.0, cos_theta=0.0),
    # Inclined wall
    dict(split="test", surface="wedge", seed=108, We=100.0, cos_theta=0.0),
    dict(split="test", surface="wedge", seed=109, We=200.0, cos_theta=-0.5),
]


def train_cases():
    return [copy.deepcopy(c) for c in CASES if c["split"] == "train"]


def test_cases():
    return [copy.deepcopy(c) for c in CASES if c["split"] == "test"]


def case_label(c: dict) -> str:
    extra = {k: v for k, v in c.items() if k not in ("split", "surface", "seed")}
    bits = [c["surface"]] + [f"{k}={v}" for k, v in sorted(extra.items())]
    return "_".join(bits)


# -------------------------------------------------------------------------------------
#  Large procedurally-sampled training sets
# -------------------------------------------------------------------------------------
#
# ``large_simple_cases`` keeps the *transfer protocol* intact: it only samples the two
# simple families (flat wall + periodic pillar arrays), but with continuous random
# Weber number, wettability, droplet radius and pillar geometry -- i.e. a much denser
# and wider coverage of the *simple* distribution.
#
# ``augmented_cases`` is the geometry-augmentation experiment (README route (a)): it
# additionally samples random / hierarchical pillars, grooves and wedges, but with
# seeds and parameters that are disjoint from the fixed test cases above (seeds
# >= 1000, test uses 100..109), so the test surfaces are still never seen.

SIMPLE_FAMILIES = ("flat", "pillars")
COMPLEX_FAMILIES = ("random_pillars", "hierarchical", "grooves", "wedge")


def _common(rng):
    return dict(
        We=float(np.round(rng.uniform(40.0, 320.0), 1)),
        cos_theta=float(np.round(rng.uniform(-0.7, 0.7), 2)),
        R=float(np.round(rng.uniform(0.55, 0.8), 3)),
    )


def _sample_simple(rng, i):
    if rng.uniform() < 0.4:
        return dict(surface="flat", **_common(rng))
    return dict(
        surface="pillars",
        n_pillars=int(rng.integers(3, 9)),
        width=float(np.round(rng.uniform(0.15, 0.4), 3)),
        height=float(np.round(rng.uniform(0.2, 0.55), 3)),
        center=float(np.round(3.0 + rng.uniform(-0.5, 0.5), 3)),
        **_common(rng),
    )


def _sample_complex(rng, i):
    fam = COMPLEX_FAMILIES[i % len(COMPLEX_FAMILIES)]
    c = dict(surface=fam, seed=1000 + i, **_common(rng))
    if fam == "random_pillars":
        c["n_pillars"] = int(rng.integers(4, 10))
    elif fam == "hierarchical":
        c.update(
            n_pillars=int(rng.integers(2, 5)),
            width=float(np.round(rng.uniform(0.4, 0.8), 3)),
            height=float(np.round(rng.uniform(0.3, 0.6), 3)),
            n_sub=int(rng.integers(2, 5)),
            sub_width=float(np.round(rng.uniform(0.06, 0.14), 3)),
            sub_height=float(np.round(rng.uniform(0.08, 0.2), 3)),
        )
    elif fam == "grooves":
        c.update(
            n_grooves=int(rng.integers(3, 10)),
            width=float(np.round(rng.uniform(0.15, 0.4), 3)),
            depth=float(np.round(rng.uniform(0.15, 0.4), 3)),
        )
    else:  # wedge (same wall anchoring as the test wedge)
        c["slope"] = float(np.round(rng.choice([-1, 1]) * rng.uniform(0.15, 0.7), 3))
    return c


def large_simple_cases(n: int = 120, seed: int = 2026):
    rng = np.random.default_rng(seed)
    return [dict(split="train", family="simple", **_sample_simple(rng, i)) for i in range(n)]


def augmented_cases(n: int = 48, seed: int = 4242):
    rng = np.random.default_rng(seed)
    return [dict(split="train", family="complex_aug", **_sample_complex(rng, i)) for i in range(n)]


def _low_common(rng):
    # Low-We regime: gentler impact, higher surface tension.  The original
    # cases kept Re=200 while reducing We, which made the coarse diffuse
    # interface develop large spurious currents before impact.  Re is matched
    # to the sampled impact speed and the interface is deliberately resolved
    # with three grid cells for this stress-test regime.
    We = float(np.round(rng.uniform(12.0, 60.0), 1))
    u_impact = float(np.round(rng.uniform(0.15, 0.30), 3))
    Re = float(np.round(np.clip(200.0 * u_impact / 0.5, 60.0, 120.0), 1))
    # dt 1e-3 for We<20, 2e-3 otherwise (CFL for capillary)
    dt = 1e-3 if We < 18 else 2e-3
    return dict(
        We=We,
        Re=Re,
        cos_theta=float(np.round(rng.uniform(-0.5, 0.6), 2)),
        R=float(np.round(rng.uniform(0.55, 0.75), 3)),
        u_impact=u_impact,
        eps_factor=3.0,
        wall_energy_amp=0.5,
        wet_band=0.08,
        # Keep the initial interface outside the wall but close enough that
        # the localized divergence-free impact field reaches it within the
        # low-We training horizon.
        impact_gap=0.03,
        velocity_mode="streamfunction",
        dt=dt,
    )

def _sample_low(rng, i):
    # 60% flat, 40% pillars, all low-We
    if rng.uniform() < 0.6:
        return dict(surface="flat", **_low_common(rng))
    return dict(
        surface="pillars",
        n_pillars=int(rng.integers(3, 6)),
        width=float(np.round(rng.uniform(0.18, 0.35), 3)),
        height=float(np.round(rng.uniform(0.15, 0.35), 3)),
        center=float(np.round(3.0 + rng.uniform(-0.4, 0.4), 3)),
        **_low_common(rng),
    )

def lowwe_cases(n: int = 48, seed: int = 2027):
    rng = np.random.default_rng(seed)
    return [dict(split="train", family="lowWe", **_sample_low(rng, i)) for i in range(n)]

def lowwe_test_cases(n: int = 12, seed: int = 3027):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fam = COMPLEX_FAMILIES[i % len(COMPLEX_FAMILIES)]
        c = dict(split="test", family="lowWe_test", surface=fam, seed=2000+i, **_low_common(rng))
        if fam == "random_pillars":
            c["n_pillars"] = int(rng.integers(4, 8))
        elif fam == "hierarchical":
            c.update(n_pillars=int(rng.integers(2,4)), width=float(np.round(rng.uniform(0.4,0.7),3)),
                     height=float(np.round(rng.uniform(0.3,0.55),3)), n_sub=int(rng.integers(2,4)),
                     sub_width=float(np.round(rng.uniform(0.07,0.12),3)), sub_height=float(np.round(rng.uniform(0.08,0.16),3)))
        elif fam == "grooves":
            c.update(n_grooves=int(rng.integers(4,8)), width=float(np.round(rng.uniform(0.18,0.32),3)), depth=float(np.round(rng.uniform(0.15,0.3),3)))
        else:
            c["slope"] = float(np.round(rng.choice([-1,1])*rng.uniform(0.15,0.5),3))
        out.append(c)
    return out

CASE_SETS = {
    "base": lambda: [copy.deepcopy(c) for c in CASES],
    "large": large_simple_cases,
    "aug": augmented_cases,
    "lowWe": lowwe_cases,
    "lowWe_test": lowwe_test_cases,
    "lowWe_all": lambda: lowwe_cases(48, 2027) + lowwe_test_cases(12, 3027),
}
