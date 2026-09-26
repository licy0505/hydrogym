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

DATASET_SCHEMA_VERSION = 2


def _require_current_dataset(d, path):
    if "dataset_schema_version" not in d.files:
        raise RuntimeError(f"stale two_phase dataset: {path} has no schema marker; regenerate with generate_dataset.py")
    version = int(np.asarray(d["dataset_schema_version"]).item())
    if version != DATASET_SCHEMA_VERSION:
        raise RuntimeError(
            f"stale two_phase dataset: {path} schema={version}, expected={DATASET_SCHEMA_VERSION}; regenerate"
        )


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
        _require_current_dataset(d, f)
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
        _require_current_dataset(d, f)
        out.append(
            dict(
                phi=d["phi"].astype(np.float32),
                u=d["u"].astype(np.float32),
                v=d["v"].astype(np.float32),
                chi=d["chi"].astype(np.float32),
                sdf=d["sdf"].astype(np.float32) if "sdf" in d else None,
                time=d["time"].astype(np.float32) if "time" in d else np.arange(d["phi"].shape[0], dtype=np.float32),
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
        _require_current_dataset(d, f)
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


#######################################################################################
#                                                                                     #
#          Fourier Neural Operator + SDF geometry encoding + conservative head        #
#                                                                                     #
#######################################################################################
#
# The U-Net above sees the geometry only through the (smoothed) solid indicator chi,
# which is ~0 everywhere except inside the solid: a droplet one cell above a pillar
# "sees" nothing.  Two upgrades, both geometry-agnostic so they transfer to unseen
# surfaces:
#
# * **SDF multi-scale encoding** -- the solid signed-distance field is expanded into
#   tanh(sdf / l) at several length scales l (near-wall band, pillar scale, droplet
#   scale) plus the unit wall normal grad(sdf).  Every fluid cell now knows how far
#   the nearest wall is and in which direction.
# * **FNO** -- spectral convolutions give every layer a global receptive field (the
#   solver itself is spectral / periodic: FFT pressure projection), plus a local 3x3
#   path for sharp interfaces ("U-FNO-lite").  FiLM injects (We, Re, cos theta).
# * **Conservative head** -- the predicted phase field is corrected so that the
#   total liquid mass exactly equals the input mass (the solver conserves it to
#   <0.002 %).  The correction is distributed over the interface
#   band 4 phi (1-phi), is differentiable and is used in training *and* rollout.

SDF_SCALES = (0.1, 0.4, 1.6)


def geometry_features(chi: np.ndarray, sdf: np.ndarray | None, dx: float, mode: str = "sdf") -> np.ndarray:
    """Static geometry channels (H, W, G).

    mode="chi" -> [chi]                       (baseline, as the original U-Net)
    mode="sdf" -> [chi, tanh(sdf/l)..., nx, ny]
    """
    chi = np.asarray(chi, np.float32)
    if mode == "chi" or sdf is None:
        if mode != "chi":
            raise ValueError("dataset has no 'sdf' array -- regenerate with generate_dataset.py")
        return chi[..., None]
    sdf = np.asarray(sdf, np.float32)
    feats = [chi] + [np.tanh(sdf / scale) for scale in SDF_SCALES]
    gx = (np.roll(sdf, -1, 0) - np.roll(sdf, 1, 0)) / (2 * dx)
    gy = (np.roll(sdf, -1, 1) - np.roll(sdf, 1, 1)) / (2 * dx)
    nrm = np.sqrt(gx**2 + gy**2) + 1e-6
    feats += [gx / nrm, gy / nrm]
    return np.stack(feats, axis=-1).astype(np.float32)


def n_geom(mode: str) -> int:
    return 1 if mode == "chi" else 1 + len(SDF_SCALES) + 2


class SpectralConv2d(nn.Module):
    out_channels: int
    modes1: int
    modes2: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        m1, m2 = min(self.modes1, H // 2), min(self.modes2, W // 2 + 1)
        scale = 1.0 / (C * self.out_channels)
        init = lambda k, s: scale * jax.random.uniform(k, s)
        w1 = self.param("w1", init, (m1, m2, C, self.out_channels, 2))
        w2 = self.param("w2", init, (m1, m2, C, self.out_channels, 2))
        w1 = w1[..., 0] + 1j * w1[..., 1]
        w2 = w2[..., 0] + 1j * w2[..., 1]
        xf = jnp.fft.rfft2(x, axes=(1, 2))
        top = jnp.einsum("bxyi,xyio->bxyo", xf[:, :m1, :m2], w1)
        bot = jnp.einsum("bxyi,xyio->bxyo", xf[:, H - m1 :, :m2], w2)
        out = jnp.zeros((B, H, W // 2 + 1, self.out_channels), dtype=xf.dtype)
        out = out.at[:, :m1, :m2].set(top).at[:, H - m1 :, :m2].set(bot)
        return jnp.fft.irfft2(out, s=(H, W), axes=(1, 2)).astype(x.dtype)


class FNOBlock(nn.Module):
    width: int
    modes: int
    local: bool = True

    @nn.compact
    def __call__(self, x, cond):
        y = SpectralConv2d(self.width, self.modes, self.modes)(x)
        y = y + nn.Conv(self.width, (1, 1))(x)
        if self.local:
            y = y + nn.Conv(self.width, (3, 3), padding="CIRCULAR")(x)
        y = Film(self.width)(y, cond)
        return x + nn.gelu(y)  # residual block


def mass_project(phi_pred, phi_in, chi=None):
    """Bounded, fluid-only, mass-conservative projection.

    The previous one-shot correction could push phi outside [0,1]; the caller
    then clipped it and silently broke the claimed exact conservation.  Solve
    for a scalar correction by bisection *with the bounds inside the solve*.
    """
    phi0 = jnp.clip(phi_pred, 0.0, 1.0)
    if chi is None:
        fluid = jnp.ones_like(phi0)
    else:
        fluid = (chi < 0.5).astype(phi0.dtype)

    phi0 = phi0 * fluid
    target = jnp.sum(jnp.clip(phi_in, 0.0, 1.0) * fluid, axis=(1, 2), keepdims=True)
    capacity = jnp.sum(fluid, axis=(1, 2), keepdims=True)
    target = jnp.clip(target, 0.0, capacity)
    weight = fluid * (4.0 * phi0 * (1.0 - phi0) + 5.0e-2)

    lo = -32.0 * jnp.ones_like(target)
    hi = 32.0 * jnp.ones_like(target)

    def body(_i, bounds):
        lo_, hi_ = bounds
        mid = 0.5 * (lo_ + hi_)
        cand = jnp.clip(phi0 + mid * weight, 0.0, 1.0) * fluid
        mass = jnp.sum(cand, axis=(1, 2), keepdims=True)
        lo_ = jnp.where(mass < target, mid, lo_)
        hi_ = jnp.where(mass < target, hi_, mid)
        return lo_, hi_

    lo, hi = jax.lax.fori_loop(0, 36, body, (lo, hi))
    lam = 0.5 * (lo + hi)
    return jnp.clip(phi0 + lam * weight, 0.0, 1.0) * fluid


class FNO(nn.Module):
    width: int = 32
    modes: int = 12
    layers: int = 4
    local: bool = True
    out_channels: int = 3

    @nn.compact
    def __call__(self, x, cond):
        h = nn.Dense(self.width)(x)
        for _ in range(self.layers):
            h = FNOBlock(self.width, self.modes, self.local)(h, cond)
        h = nn.gelu(nn.Dense(2 * self.width)(h))
        return nn.Dense(self.out_channels, kernel_init=nn.initializers.zeros)(h)


class Surrogate(nn.Module):
    """State-to-state wrapper: x=(phi,u,v,geom...) -> next (phi,u,v).

    ``residual`` predicts the increment; ``conservative`` applies :func:`mass_project`.
    """

    arch: str = "fno"
    residual: bool = True
    conservative: bool = True
    width: int = 32
    modes: int = 12
    layers: int = 4
    base: int = 16
    levels: int = 3

    @nn.compact
    def __call__(self, x, cond):
        if self.arch == "fno":
            out = FNO(self.width, self.modes, self.layers)(x, cond)
        else:
            out = UNet(self.base, self.levels, 3)(x, cond)
        if self.residual:
            out = out + x[..., :3]
        if self.conservative:
            phi = mass_project(out[..., 0], x[..., 0], x[..., 3])
            out = jnp.concatenate([phi[..., None], out[..., 1:]], axis=-1)
        return out


def build_model(cfg: dict) -> Surrogate:
    keys = ("arch", "residual", "conservative", "width", "modes", "layers", "base", "levels")
    return Surrogate(**{k: cfg[k] for k in keys if k in cfg})


def load_checkpoint(path):
    """Return (model, params, cfg).  Handles legacy U-Net checkpoints."""
    import pickle

    with open(path, "rb") as fh:
        ck = pickle.load(fh)
    cfg = ck.get("cfg")
    if cfg is None:  # legacy: plain UNet, absolute prediction, chi only
        cfg = dict(
            arch="unet",
            residual=False,
            conservative=False,
            base=ck.get("base", 16),
            levels=ck.get("levels", 3),
            geom="chi",
            uv_scale=ck["uv_scale"],
            legacy=True,
        )
        return UNet(base=cfg["base"], levels=cfg["levels"], out_channels=3), ck["params"], cfg
    version = int(cfg.get("dataset_schema_version", 0))
    if version != DATASET_SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint {path} was trained on stale two_phase data schema={version}; "
            f"expected={DATASET_SCHEMA_VERSION}. Regenerate data and retrain."
        )
    return build_model(cfg), ck["params"], cfg


class TrajectoryStore:
    """Memory-lean container of many trajectories (float16 states, per-case geometry).

    Batches are windows of ``K+1`` consecutive frames that never cross case
    boundaries, so the same store serves teacher-forced (K=1) and unrolled training.
    """

    def __init__(self, dirs, split="train", geom="sdf", families=None):
        import glob
        import os

        if isinstance(dirs, str):
            dirs = [d for d in dirs.split(",") if d]
        files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.npz")))
        states, geoms, scals, self.meta = [], [], [], []
        for f in files:
            d = np.load(f, allow_pickle=True)
            if str(d["split"]) != split:
                continue
            _require_current_dataset(d, f)
            fam = "simple" if str(d["surface"]) in ("flat", "pillars") else "complex"
            if families and fam not in families:
                continue
            states.append(np.stack([d["phi"], d["u"], d["v"]], axis=-1).astype(np.float16))
            scal = d["scalars"].astype(np.float32)
            geoms.append(geometry_features(d["chi"], d["sdf"] if "sdf" in d else None, float(scal[3]), geom))
            scals.append(scal)
            self.meta.append(dict(file=f, surface=str(d["surface"]), family=fam))
        if not states:
            raise RuntimeError(f"no '{split}' trajectories found in {dirs}")
        T = min(s.shape[0] for s in states)
        self.states = np.stack([s[:T] for s in states])  # (N, T, H, W, 3) float16
        self.geom = np.stack(geoms)  # (N, H, W, G)
        self.scal = np.stack(scals)  # (N, S)
        self.T = T
        self.uv_scale = float(max(np.abs(self.states[..., 1:3].astype(np.float32)).max(), 1e-3))

    def __len__(self):
        return self.states.shape[0]

    def windows(self, K):
        n, T = len(self), self.T
        return np.array([(c, t) for c in range(n) for t in range(T - K)], dtype=np.int64)

    def batch(self, idx, K, uv):
        c, t = idx[:, 0], idx[:, 1]
        fr = np.stack([self.states[c, t + k] for k in range(K + 1)], axis=1).astype(np.float32)
        fr[..., 1:3] /= uv
        return fr, self.geom[c], self.scal[c]
