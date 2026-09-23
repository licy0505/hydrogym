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
