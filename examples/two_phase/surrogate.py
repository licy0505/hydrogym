"""
A small conditional U-Net surrogate for the phase distribution.

Given the current two-phase state (phi, u, v), the solid indicator chi and a few
scalars (We, Re, cos_theta), it predicts the state one save-interval (20 solver
steps) ahead.  Trained on *simple* cases it is rolled out autoregressively to
predict impacts on *complex* surfaces -- the transfer question.

The architecture is a compact U-Net with FiLM conditioning on the scalars.  It is
deliberately small so that it trains in minutes on CPU.  For production one would
swap in a Fourier Neural Operator (FNO) or a larger U-FNO; the data pipeline and
transfer protocol here are identical.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class Film(nn.Module):
    """Feature-wise linear modulation from the scalar conditioning vector."""

    channels: int

    @nn.compact
    def __call__(self, x, cond):
        # cond: (B, S) -> per-channel (gamma, beta)
        h = nn.Dense(2 * self.channels)(cond)
        gamma, beta = jnp.split(h, 2, axis=-1)  # (B, C)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return gamma * x + beta


class ConvBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x, cond):
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = Film(self.channels)(x, cond)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = Film(self.channels)(x, cond)
        x = nn.relu(x)
        return x


class UNet(nn.Module):
    base: int = 32
    levels: int = 3
    out_channels: int = 3

    @nn.compact
    def __call__(self, x, cond):
        # x: (B, H, W, 4) [phi,u,v,chi]; cond: (B, S)
        skips = []
        c = self.base
        for i in range(self.levels):
            x = ConvBlock(c)(x, cond)
            skips.append(x)
            x = nn.max_pool(x, (2, 2), (2, 2))
            c *= 2
        x = ConvBlock(c)(x, cond)
        for i in range(self.levels):
            c //= 2
            x = nn.ConvTranspose(c, (2, 2), strides=(2, 2))(x)
            x = jnp.concatenate([x, skips[self.levels - 1 - i]], axis=-1)
            x = ConvBlock(c)(x, cond)
        return nn.Conv(self.out_channels, (3, 3), padding="SAME")(x)


def load_arrays(data_dir, split):
    """Load (X, Y, cond, chi, meta) for a split.  Returns stacked arrays.

    X[t] = (phi,u,v,chi) at frame i, Y[t] = (phi,u,v) at frame i+1.
    """
    import glob
    import os

    Xs, Ys, Cs, metas = [], [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if str(d["split"]) != split:
            continue
        phi = d["phi"].astype(np.float32)
        u = d["u"].astype(np.float32)
        v = d["v"].astype(np.float32)
        chi = d["chi"].astype(np.float32)
        scal = d["scalars"].astype(np.float32)
        T = phi.shape[0] - 1
        chi3 = np.repeat(chi[None], T, axis=0)
        X = np.stack([phi[:-1], u[:-1], v[:-1], chi3], axis=-1)
        Y = np.stack([phi[1:], u[1:], v[1:]], axis=-1)
        cond = np.repeat(scal[None], T, axis=0)
        Xs.append(X)
        Ys.append(Y)
        Cs.append(cond)
        metas.append(dict(file=f, surface=str(d["surface"]), scal=scal, chi=chi, T=T))
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    C = np.concatenate(Cs)
    return X, Y, C, metas


def load_full(data_dir, split):
    """Per-case full trajectories for rollout evaluation."""
    import glob
    import os

    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if str(d["split"]) != split:
            continue
        out.append(
            dict(
                phi=d["phi"].astype(np.float32),
                u=d["u"].astype(np.float32),
                v=d["v"].astype(np.float32),
                chi=d["chi"].astype(np.float32),
                scalars=d["scalars"].astype(np.float32),
                surface=str(d["surface"]),
                file=f,
            )
        )
    return out


def load_windows(data_dir, split, K):
    """Return windows of K+1 consecutive frames for unrolled training.

    Each window is a dict with arrays ``frames`` (K+1, H, W, 3) of (phi,u,v),
    ``chi`` (H,W) and ``scalars`` (S,).  Windows never cross case boundaries.
    """
    import glob
    import os

    wins = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if str(d["split"]) != split:
            continue
        phi = d["phi"].astype(np.float32)
        u = d["u"].astype(np.float32)
        v = d["v"].astype(np.float32)
        chi = d["chi"].astype(np.float32)
        scal = d["scalars"].astype(np.float32)
        T = phi.shape[0]
        for t in range(0, T - K):
            fr = np.stack([phi[t : t + K + 1], u[t : t + K + 1], v[t : t + K + 1]], axis=-1)
            wins.append(dict(frames=fr, chi=chi, scalars=scal))
    return wins
