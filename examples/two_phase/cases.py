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


CASE_SETS = {
    "base": lambda: [copy.deepcopy(c) for c in CASES],
    "large": large_simple_cases,
    "aug": augmented_cases,
}
