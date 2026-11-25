"""Pure-PyTorch implementation of the SO3LR-Tauri trunk and partial-charge head.

Reproduces ``so3lr-tauri-{s,m,l}-2606`` node representations ``(x, ev)`` and
charge-conserving partial charges ``q`` (``PartialChargesSparse``) from atomic
numbers, coordinates, and short-range intramonomer edges (``<= 5.0 A``), with
zero JAX/Flax/``so3lr`` runtime dependencies and zero long-range neighbor list.

The forward matches JAX ``so3lr`` in float64 to machine precision (``< 1e-12``)
across all three 2606 flavors (``T-S``, ``T-M``, ``T-L``) and compiles as a
single graph under ``torch.compile(fullgraph=True, dynamic=True)``.

Weights can be imported from a SO3LR checkpoint directory containing
``params.pkl`` (and optionally ``hyperparameters.yaml``) via
:meth:`SO3LRTrunk.from_so3lr_dir`, or stored directly inside an
:class:`~apnet_pt.ff2.FF2Model` checkpoint via ``embed_so3lr_trunk=True`` /
:meth:`~apnet_pt.ff2.FF2Model.attach_so3lr_trunk`.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..ff1.trunk import scatter_sum
from .trunk import real_sph

# 1 / (2 * Bohr_in_Angstrom): so3lr Bernstein radial basis exponent (A^-1)
BERNSTEIN_GAMMA = 0.5 / 0.529177210903
PAIR_FLOOR_WIDTH = 0.05
RMS_EPS = 1e-6

SO3LR_PRESETS: dict[str, dict[str, Any]] = {
    "so3lr-tauri-s-2606": {
        "cutoff": 4.5,
        "num_layers": 2,
        "num_features": 128,
        "num_heads": 4,
        "degrees": (1, 2, 3, 4),
        "num_rbf": 32,
        "avg_num_neighbors": 16.0,
        "energy_regression_dim": 128,
        "zmax": 118,
        "gamma": BERNSTEIN_GAMMA,
    },
    "so3lr-tauri-m-2606": {
        "cutoff": 4.5,
        "num_layers": 3,
        "num_features": 128,
        "num_heads": 4,
        "degrees": (1, 2, 3, 4),
        "num_rbf": 32,
        "avg_num_neighbors": 16.0,
        "energy_regression_dim": 128,
        "zmax": 118,
        "gamma": BERNSTEIN_GAMMA,
    },
    "so3lr-tauri-l-2606": {
        "cutoff": 5.0,
        "num_layers": 3,
        "num_features": 256,
        "num_heads": 8,
        "degrees": (1, 1, 2, 2, 3, 3, 4, 4),
        "num_rbf": 64,
        "avg_num_neighbors": 22.0,
        "energy_regression_dim": 128,
        "zmax": 118,
        "gamma": BERNSTEIN_GAMMA,
    },
}

PRESET_ALIASES: dict[str, str] = {
    "ts": "so3lr-tauri-s-2606",
    "tm": "so3lr-tauri-m-2606",
    "tl": "so3lr-tauri-l-2606",
    "so3lr-t-s": "so3lr-tauri-s-2606",
    "so3lr-t-m": "so3lr-tauri-m-2606",
    "so3lr-t-l": "so3lr-tauri-l-2606",
}

_SPH_DEGREE_SLICE = {1: (0, 3), 2: (3, 8), 3: (8, 15), 4: (15, 24)}
_RMIN_CACHE: np.ndarray | None = None


def resolve_preset_name(name: str) -> str:
    key = name.strip().lower()
    key = PRESET_ALIASES.get(key, key)
    if key not in SO3LR_PRESETS:
        raise ValueError(
            f"unknown SO3LR preset {name!r}; expected one of "
            f"{tuple(SO3LR_PRESETS)} or aliases {tuple(PRESET_ALIASES)}"
        )
    return key


def load_nlh_rmin_table() -> np.ndarray:
    """Return the (92, 92) float64 NLH 5 eV reference-distance table (A)."""
    global _RMIN_CACHE
    if _RMIN_CACHE is None:
        ref = importlib_resources.files("apnet_pt").joinpath(
            "data", "so3lr_nlh_rmin_5ev.npz"
        )
        with importlib_resources.as_file(ref) as path:
            with np.load(path) as data:
                _RMIN_CACHE = np.array(data["rmin_table"], dtype=np.float64)
    return _RMIN_CACHE


def phys_cutoff(d: Tensor, cutoff: float) -> Tensor:
    """SO3LR polynomial physical cutoff ``1 - 6u^5 + 15u^4 - 10u^3`` on ``(0, cutoff)``."""
    inside = (d > 0.0) & (d < cutoff)
    u = torch.where(inside, d / cutoff, torch.ones_like(d))
    u2 = u * u
    u3 = u2 * u
    val = 1.0 - 6.0 * u3 * u2 + 15.0 * u2 * u2 - 10.0 * u3
    return torch.where(inside, val, torch.zeros_like(d))


def floor_descriptor_distance(
    d: Tensor,
    z: Tensor,
    src: Tensor,
    dst: Tensor,
    rmin_table: Tensor,
    width: float = PAIR_FLOOR_WIDTH,
) -> Tensor:
    """C1 per-element-pair distance floor handed to the radial basis only."""
    max_idx = rmin_table.shape[0] - 1
    zi = (z[src] - 1).clamp(0, max_idx)
    zj = (z[dst] - 1).clamp(0, max_idx)
    rmin = rmin_table[torch.minimum(zi, zj), torch.maximum(zi, zj)].to(d.dtype)
    d_off = rmin + width
    u = (d_off - d) / width
    return torch.where(d >= d_off, d, d_off - width * torch.tanh(u))


def rms_norm(x: Tensor, eps: float = RMS_EPS) -> Tensor:
    """Parameter-free RMSNorm along the last axis."""
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)


def real_sph_degrees(unit_r: Tensor, degrees: tuple[int, ...]) -> Tensor:
    """Concatenated orthonormal real spherical harmonics for ``degrees`` (each in 1..4)."""
    y24 = real_sph(unit_r)
    if degrees == (1, 2, 3, 4):
        return y24
    parts = []
    for deg in degrees:
        lo, hi = _SPH_DEGREE_SLICE[deg]
        parts.append(y24[:, lo:hi])
    return torch.cat(parts, dim=-1)


class BernsteinRBF(nn.Module):
    """SO3LR exponential Bernstein radial basis ``B_{k,K-1}(exp(-gamma d))``."""

    def __init__(self, num_rbf: int, gamma: float = BERNSTEIN_GAMMA):
        super().__init__()
        self.num_rbf = int(num_rbf)
        self.gamma = float(gamma)
        k = torch.arange(self.num_rbf, dtype=torch.float64)
        n = float(self.num_rbf - 1)
        log_binom = (
            torch.lgamma(torch.tensor(n + 1.0, dtype=torch.float64))
            - torch.lgamma(k + 1.0)
            - torch.lgamma(n - k + 1.0)
        )
        self.register_buffer("k", k, persistent=False)
        self.register_buffer("n_minus_k", n - k, persistent=False)
        self.register_buffer("log_binom", log_binom, persistent=False)

    def forward(self, d: Tensor) -> Tensor:
        dt = d.dtype
        exp_r = torch.exp(-self.gamma * d)
        valid = (exp_r > 0.0) & (exp_r < 1.0)
        er = torch.where(valid, exp_r, torch.full_like(exp_r, 0.5))
        log_er = torch.log(er)[:, None]
        log_1m = torch.log1p(-er)[:, None]
        arg = (
            self.log_binom.to(dt)
            + self.k.to(dt) * log_er
            + self.n_minus_k.to(dt) * log_1m
        )
        out = torch.exp(arg)
        return torch.where(valid[:, None], out, torch.zeros_like(out))


class ChargeSpinEmbed(nn.Module):
    """SpookyNet / SO3LR per-monomer total-charge or unpaired-spin embedding."""

    def __init__(self, num_features: int, zmax: int = 118):
        super().__init__()
        self.num_features = int(num_features)
        self.atom_embed = nn.Embedding(zmax + 1, self.num_features)
        self.key_embed = nn.Embedding(2, self.num_features)
        self.val_embed = nn.Embedding(2, self.num_features)
        self.res1 = nn.Linear(self.num_features, self.num_features, bias=False)
        self.res2 = nn.Linear(self.num_features, self.num_features, bias=False)

    def forward(self, z: Tensor, psi: Tensor, monomer: Tensor) -> Tensor:
        sign = (psi < 0).long()
        q = self.atom_embed(z)
        k = self.key_embed(sign)[monomer]
        v = self.val_embed(sign)[monomer]
        y = F.softplus((q * k).sum(dim=-1) / math.sqrt(self.num_features))
        den = y.new_zeros(psi.shape).index_add(0, monomer, y).clamp_min(1e-30)
        a = psi.to(y.dtype)[monomer] * y / den[monomer]
        x = a[:, None] * v
        return x + self.res2(F.silu(self.res1(F.silu(x))))


class SO3kratesLayer(nn.Module):
    """One SO3LR-Tauri message-passing layer (Attention + RMSNorm + ResMLP + Exchange)."""

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        degrees: tuple[int, ...],
        num_rbf: int,
        avg_num_neighbors: float,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.num_heads = int(num_heads)
        self.degrees = tuple(int(d) for d in degrees)
        self.num_degrees = len(self.degrees)
        if self.num_features % self.num_heads != 0:
            raise ValueError(
                f"num_features {num_features} not divisible by num_heads {num_heads}"
            )
        if self.num_features % self.num_degrees != 0:
            raise ValueError(
                f"num_features {num_features} not divisible by len(degrees) {self.num_degrees}"
            )
        self.d_head1 = self.num_features // self.num_heads
        self.d_head2 = self.num_features // self.num_degrees
        self.avg_num_neighbors = float(avg_num_neighbors)

        slices = []
        lo = 0
        for d in self.degrees:
            hi = lo + 2 * d + 1
            slices.append((lo, hi))
            lo = hi
        self.degree_slices = tuple(slices)

        f = self.num_features
        h = self.num_heads
        nd = self.num_degrees
        sf_hid = f // 4

        # Radial & spherical attention filters
        self.rf1_1 = nn.Linear(num_rbf, f)
        self.rf1_2 = nn.Linear(f, f)
        self.rf2_1 = nn.Linear(num_rbf, f)
        self.rf2_2 = nn.Linear(f, f)
        self.sf1_1 = nn.Linear(nd, sf_hid)
        self.sf1_2 = nn.Linear(sf_hid, f)
        self.sf2_1 = nn.Linear(nd, sf_hid)
        self.sf2_2 = nn.Linear(sf_hid, f)

        # Multi-head Q/K/V projections (exact so3lr shapes)
        self.Wq1 = nn.Parameter(torch.empty(h, self.d_head1, self.d_head1))
        self.Wk1 = nn.Parameter(torch.empty(h, self.d_head1, self.d_head1))
        self.Wv1 = nn.Parameter(torch.empty(h, self.d_head1, self.d_head1))
        self.Wq2 = nn.Parameter(torch.empty(nd, self.d_head2, self.d_head2))
        self.Wk2 = nn.Parameter(torch.empty(nd, self.d_head2, self.d_head2))

        # Residual MLP 1 (post layer_normalization_1)
        self.res1 = nn.Linear(f, f)
        self.res2 = nn.Linear(f, f)

        # Exchange block (linear mixing of [x, contract(ev)])
        self.exch = nn.Linear(f + nd, f + nd)

        self._reset_qkv()

    def _reset_qkv(self) -> None:
        for p in (self.Wq1, self.Wk1, self.Wv1, self.Wq2, self.Wk2):
            nn.init.normal_(p, std=1.0 / math.sqrt(p.shape[-1]))

    def _contract(self, ev: Tensor) -> Tensor:
        return torch.stack(
            [
                (ev[..., lo:hi] * ev[..., lo:hi]).sum(dim=-1) / math.sqrt(hi - lo)
                for lo, hi in self.degree_slices
            ],
            dim=-1,
        )

    def _degree_repeat(self, a: Tensor) -> Tensor:
        return torch.cat(
            [
                a[:, k : k + 1].expand(-1, hi - lo)
                for k, (lo, hi) in enumerate(self.degree_slices)
            ],
            dim=-1,
        )

    def forward(
        self,
        x: Tensor,
        ev: Tensor,
        rbf_ij: Tensor,
        ylm_ij: Tensor,
        cut: Tensor,
        src: Tensor,
        dst: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n = x.shape[0]
        f = self.num_features
        h = self.num_heads
        nd = self.num_degrees

        # Filters on edges
        dev_inv = self._contract(ev[dst] - ev[src])
        w1 = (
            self.rf1_2(F.silu(self.rf1_1(rbf_ij)))
            + self.sf1_2(F.silu(self.sf1_1(dev_inv)))
        ).view(-1, h, self.d_head1)
        w2 = (
            self.rf2_2(F.silu(self.rf2_1(rbf_ij)))
            + self.sf2_2(F.silu(self.sf2_1(dev_inv)))
        ).view(-1, nd, self.d_head2)

        # Node-level Q, K, V projections + QK RMSNorm (identical to edge-level, faster)
        x1_h = x.view(n, h, self.d_head1)
        x2_h = x.view(n, nd, self.d_head2)
        q1 = rms_norm(torch.einsum("hij,nhj->nhi", self.Wq1, x1_h))
        k1 = rms_norm(torch.einsum("hij,nhj->nhi", self.Wk1, x1_h))
        v1 = torch.einsum("hij,nhj->nhi", self.Wv1, x1_h)
        q2 = rms_norm(torch.einsum("hij,nhj->nhi", self.Wq2, x2_h))
        k2 = rms_norm(torch.einsum("hij,nhj->nhi", self.Wk2, x2_h))

        inv_nbar = 1.0 / self.avg_num_neighbors
        alpha1 = (q1[src] * w1 * k1[dst]).sum(dim=-1) * inv_nbar * cut[:, None]
        alpha2 = (q2[src] * w2 * k2[dst]).sum(dim=-1) * inv_nbar * cut[:, None]

        # Message aggregation
        msg_x = (alpha1[:, :, None] * v1[dst]).reshape(-1, f)
        x = x + scatter_sum(msg_x, src, n)
        ev = ev + scatter_sum(self._degree_repeat(alpha2) * ylm_ij, src, n)

        # Post-attention RMSNorm + Residual MLP 1
        x = rms_norm(x)
        x = x + self.res2(F.silu(self.res1(F.silu(x))))

        # Exchange block
        y = self.exch(torch.cat([x, self._contract(ev)], dim=-1))
        cx, cev = y[:, :f], y[:, f:]
        x = x + cx
        ev = ev + self._degree_repeat(cev) * ev
        return x, ev


class PartialChargesHead(nn.Module):
    """SO3LR ``PartialChargesSparse`` charge-conserving readout from ``x``."""

    def __init__(
        self,
        num_features: int,
        regression_dim: int = 128,
        max_z_bias: int = 100,
    ):
        super().__init__()
        self.elem_bias = nn.Embedding(max_z_bias, 1)
        self.dense1 = nn.Linear(num_features, regression_dim)
        self.dense2 = nn.Linear(regression_dim, 1)

    def forward(
        self,
        x: Tensor,
        z: Tensor,
        monomer: Tensor,
        monomer_charge: Tensor,
    ) -> Tensor:
        q_bias = self.elem_bias(z).squeeze(-1)
        q_nn = self.dense2(F.silu(self.dense1(x))).squeeze(-1)
        x_q = q_nn + q_bias
        q_sum = x_q.new_zeros(monomer_charge.shape).index_add(0, monomer, x_q)
        counts = (
            x_q.new_zeros(monomer_charge.shape)
            .index_add(0, monomer, torch.ones_like(x_q))
            .clamp_min(1.0)
        )
        shift = (monomer_charge.to(x_q.dtype) - q_sum) / counts
        return x_q + shift[monomer]


class SO3LRTrunk(nn.Module):
    """Pure-PyTorch SO3LR-Tauri trunk returning ``(x, ev, partial_charges)``.

    Parameters
    ----------
    cutoff : float
        Short-range cutoff in Angstrom (``4.5`` for T-S/T-M, ``5.0`` for T-L).
        Edges beyond ``cutoff`` receive exact zero cutoff weight, so a caller's
        ``5.0 A`` edge list can be shared directly with all three flavors.
    num_layers, num_features, num_heads, degrees, num_rbf, avg_num_neighbors
        SO3krates architecture hyperparameters.
    energy_regression_dim : int
        Hidden width of ``PartialChargesHead`` (128 for all 2606 models).
    zmax : int
        Maximum atomic number for element embeddings (118).
    gamma : float
        Bernstein exponential radial basis decay constant (A^-1).
    preset : str, optional
        Canonical preset name recorded in :meth:`config`.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        num_layers: int = 3,
        num_features: int = 256,
        num_heads: int = 8,
        degrees: tuple[int, ...] = (1, 1, 2, 2, 3, 3, 4, 4),
        num_rbf: int = 64,
        avg_num_neighbors: float = 22.0,
        energy_regression_dim: int = 128,
        zmax: int = 118,
        gamma: float = BERNSTEIN_GAMMA,
        preset: str | None = None,
    ):
        super().__init__()
        self.preset = preset
        self.cutoff = float(cutoff)
        self.num_layers = int(num_layers)
        self.num_features = int(num_features)
        self.num_heads = int(num_heads)
        self.degrees = tuple(int(d) for d in degrees)
        self.n_ev = sum(2 * d + 1 for d in self.degrees)
        self.num_rbf = int(num_rbf)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.energy_regression_dim = int(energy_regression_dim)
        self.zmax = int(zmax)
        self.gamma = float(gamma)

        self.register_buffer(
            "rmin_table",
            torch.from_numpy(load_nlh_rmin_table().copy()),
            persistent=True,
        )
        self.rbf = BernsteinRBF(self.num_rbf, self.gamma)
        self.atom_embed = nn.Embedding(self.zmax + 1, self.num_features)
        self.charge_embed = ChargeSpinEmbed(self.num_features, self.zmax)
        self.spin_embed = ChargeSpinEmbed(self.num_features, self.zmax)
        self.resid_lambdas = nn.Parameter(torch.ones(self.num_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(self.num_layers))
        self.layers = nn.ModuleList(
            [
                SO3kratesLayer(
                    self.num_features,
                    self.num_heads,
                    self.degrees,
                    self.num_rbf,
                    self.avg_num_neighbors,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.partial_charges = PartialChargesHead(
            self.num_features, self.energy_regression_dim
        )

    def config(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cutoff": self.cutoff,
            "num_layers": self.num_layers,
            "num_features": self.num_features,
            "num_heads": self.num_heads,
            "degrees": list(self.degrees),
            "num_rbf": self.num_rbf,
            "avg_num_neighbors": self.avg_num_neighbors,
            "energy_regression_dim": self.energy_regression_dim,
            "zmax": self.zmax,
            "gamma": self.gamma,
        }
        if self.preset is not None:
            out["preset"] = self.preset
        return out

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "SO3LRTrunk":
        canon = resolve_preset_name(name)
        cfg = {**SO3LR_PRESETS[canon], **overrides, "preset": canon}
        return cls(**cfg)

    def forward(
        self,
        z: Tensor,
        R: Tensor,
        edges: Tensor,
        monomer: Tensor,
        monomer_charge: Tensor,
        num_unpaired_electrons: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute ``(x, ev, q)`` for a batch of monomers.

        Parameters
        ----------
        z : (N,) int tensor
            Atomic numbers.
        R : (N, 3) float tensor
            Atomic positions in Angstrom.
        edges : (2, E) int tensor
            Directed intramonomer edge indices ``(src, dst)`` (``idx_i, idx_j``).
        monomer : (N,) int tensor
            Monomer index ``0..M-1`` of each atom.
        monomer_charge : (M,) float/int tensor
            Formal total charge of each monomer.
        num_unpaired_electrons : (M,) tensor, optional
            Unpaired electrons per monomer (``multiplicity - 1``).  When
            omitted, defaults to the lowest-multiplicity state
            ``(sum_Z - round(Q)) % 2``.
        """
        src, dst = edges[0], edges[1]
        r_ij = R[dst] - R[src]
        r2 = (r_ij * r_ij).sum(dim=-1)
        nonzero = r2 > 0.0
        d_safe = torch.sqrt(torch.where(nonzero, r2, torch.ones_like(r2)))
        d_ij = torch.where(nonzero, d_safe, torch.zeros_like(r2))
        unit_r_ij = torch.where(
            nonzero[:, None],
            r_ij / d_safe[:, None],
            torch.zeros_like(r_ij),
        )

        cut = phys_cutoff(d_ij, self.cutoff)
        d_basis = floor_descriptor_distance(d_ij, z, src, dst, self.rmin_table)
        rbf_ij = self.rbf(d_basis)
        ylm_ij = real_sph_degrees(unit_r_ij, self.degrees)

        if num_unpaired_electrons is None:
            z_sum = monomer.new_zeros(monomer_charge.shape, dtype=torch.long).index_add(
                0, monomer, z
            )
            spin = ((z_sum - monomer_charge.round().long()) & 1).to(R.dtype)
        else:
            spin = num_unpaired_electrons.to(R.dtype)

        e0 = self.atom_embed(z)
        e1 = self.charge_embed(z, monomer_charge.to(e0.dtype), monomer)
        e2 = self.spin_embed(z, spin.to(e0.dtype), monomer)
        x = (e0 + e1 + e2) * (1.0 / math.sqrt(3.0))
        x0 = x
        ev = x.new_zeros((z.shape[0], self.n_ev))

        for n, layer in enumerate(self.layers):
            x = self.resid_lambdas[n] * x + self.x0_lambdas[n] * x0
            x, ev = layer(x, ev, rbf_ij, ylm_ij, cut, src, dst)

        q = self.partial_charges(x, z, monomer, monomer_charge)
        return x, ev, q

    # -- Flax params.pkl import ----------------------------------------------

    def load_so3lr_params(self, raw_params: dict[str, Any]) -> None:
        """Load weights from a Flax SO3LR ``params.pkl`` mapping in place."""
        p = raw_params["params"] if "params" in raw_params else raw_params

        def copy_tensor(target: Tensor, src: Any, transpose: bool = False) -> None:
            arr = np.asarray(src)
            if transpose:
                arr = arr.T
            if tuple(arr.shape) != tuple(target.shape):
                raise ValueError(
                    f"SO3LR weight shape mismatch: got {tuple(arr.shape)}, "
                    f"expected {tuple(target.shape)}"
                )
            target.copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(target.dtype))

        def copy_linear(mod: nn.Linear, node: dict[str, Any]) -> None:
            copy_tensor(mod.weight, node["kernel"], transpose=True)
            if mod.bias is not None:
                copy_tensor(mod.bias, node["bias"])

        def copy_charge_spin(mod: ChargeSpinEmbed, node: dict[str, Any]) -> None:
            cs = node["ChargeSpinEmbedSparse_0"]
            copy_tensor(mod.atom_embed.weight, cs["Embed_0"]["embedding"])
            copy_tensor(mod.key_embed.weight, cs["Embed_1"]["embedding"])
            copy_tensor(mod.val_embed.weight, cs["Embed_2"]["embedding"])
            res = cs["Residual_0"]
            copy_linear(mod.res1, res["layers_0"])
            copy_linear(mod.res2, res["layers_1"])

        with torch.no_grad():
            copy_tensor(
                self.atom_embed.weight,
                p["feature_embeddings_0"]["Embed_0"]["embedding"],
            )
            copy_charge_spin(self.charge_embed, p["feature_embeddings_1"])
            copy_charge_spin(self.spin_embed, p["feature_embeddings_2"])
            copy_tensor(self.resid_lambdas, p["resid_lambdas"])
            copy_tensor(self.x0_lambdas, p["x0_lambdas"])

            for i, layer in enumerate(self.layers):
                lp = p[f"layers_{i}"]
                att = lp["attention_block"]
                copy_linear(layer.rf1_1, att["radial_filter1_layer_1"])
                copy_linear(layer.rf1_2, att["radial_filter1_layer_2"])
                copy_linear(layer.rf2_1, att["radial_filter2_layer_1"])
                copy_linear(layer.rf2_2, att["radial_filter2_layer_2"])
                copy_linear(layer.sf1_1, att["spherical_filter1_layer_1"])
                copy_linear(layer.sf1_2, att["spherical_filter1_layer_2"])
                copy_linear(layer.sf2_1, att["spherical_filter2_layer_1"])
                copy_linear(layer.sf2_2, att["spherical_filter2_layer_2"])
                copy_tensor(layer.Wq1, att["Wq1"])
                copy_tensor(layer.Wk1, att["Wk1"])
                copy_tensor(layer.Wv1, att["Wv1"])
                copy_tensor(layer.Wq2, att["Wq2"])
                copy_tensor(layer.Wk2, att["Wk2"])
                copy_linear(layer.res1, lp["res_mlp_1_layer_1"])
                copy_linear(layer.res2, lp["res_mlp_1_layer_2"])
                copy_linear(layer.exch, lp["exchange_block"]["mlp_layer_2"])

            pc = p["observables_0"]["electrostatic_energy"]["partial_charges"]
            copy_tensor(self.partial_charges.elem_bias.weight, pc["Embed_0"]["embedding"])
            copy_linear(self.partial_charges.dense1, pc["charge_dense_regression_vec"])
            copy_linear(self.partial_charges.dense2, pc["charge_dense_final_vec"])

    @classmethod
    def from_so3lr_dir(
        cls,
        model_dir: str | Path,
        *,
        preset: str | None = None,
        dtype: torch.dtype = torch.float32,
        freeze: bool = True,
    ) -> "SO3LRTrunk":
        """Build a :class:`SO3LRTrunk` and load weights from a SO3LR model directory.

        Reads ``params.pkl`` and (if ``preset`` is not given) infers
        hyperparameters from ``hyperparameters.yaml`` / ``hyperparameters.json``
        or the directory name.
        """
        root = Path(model_dir)
        params_path = root / "params.pkl"
        if not params_path.is_file():
            raise FileNotFoundError(f"SO3LR params.pkl not found in {root}")
        with open(params_path, "rb") as fh:
            raw_params = pickle.load(fh)

        if preset is not None:
            trunk = cls.from_preset(preset)
        else:
            cfg = _load_so3lr_dir_config(root, raw_params)
            trunk = cls(**cfg)

        trunk.to(dtype=dtype)
        trunk.load_so3lr_params(raw_params)
        if freeze:
            trunk.requires_grad_(False)
            trunk.eval()
        return trunk


def _parse_yaml_scalar(val: str) -> Any:
    v = val.strip().strip("'\"")
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("null", "none", "~"):
        return None
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_parse_yaml_scalar(x) for x in inner.split(",")] if inner else []
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _parse_so3lr_yaml(text: str) -> dict[str, Any]:
    """Minimal dependency-free parser for the ``data`` and ``model`` blocks of
    SO3LR's ``hyperparameters.yaml`` (used when PyYAML is not installed)."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        pass

    out: dict[str, dict[str, Any]] = {}
    section: str | None = None
    list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1].strip()
            out.setdefault(section, {})
            list_key = None
            continue
        if section not in ("data", "model"):
            continue
        if indent == 2 and ":" in stripped and not stripped.startswith("-"):
            k, _, rest = stripped.partition(":")
            k = k.strip()
            rest = rest.strip()
            if rest:
                out[section][k] = _parse_yaml_scalar(rest)
                list_key = None
            else:
                out[section][k] = []
                list_key = k
        elif list_key is not None and stripped.startswith("- "):
            item = stripped[2:].strip()
            if ":" not in item:
                out[section][list_key].append(_parse_yaml_scalar(item))
    return out


def _load_so3lr_dir_config(root: Path, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Extract :class:`SO3LRTrunk` kwargs from a SO3LR checkpoint directory."""
    import json

    dir_name = root.name.lower()
    preset = dir_name if dir_name in SO3LR_PRESETS else None

    hp_yaml = root / "hyperparameters.yaml"
    hp_json = root / "hyperparameters.json"
    cfg_doc: dict[str, Any] | None = None
    if hp_yaml.is_file():
        cfg_doc = _parse_so3lr_yaml(hp_yaml.read_text())
    elif hp_json.is_file():
        cfg_doc = json.loads(hp_json.read_text())

    if cfg_doc is not None and "model" in cfg_doc:
        m = cfg_doc["model"]
        d = cfg_doc.get("data", {})
        avg_nb = m.get("avg_num_neighbors", d.get("avg_num_neighbors"))
        if avg_nb is None:
            raise ValueError(f"{root}: avg_num_neighbors missing from hyperparameters")
        return {
            "cutoff": float(m["cutoff"]),
            "num_layers": int(m["num_layers"]),
            "num_features": int(m["num_features"]),
            "num_heads": int(m["num_heads"]),
            "degrees": tuple(int(x) for x in m["degrees"]),
            "num_rbf": int(m["num_radial_basis_fn"]),
            "avg_num_neighbors": float(avg_nb),
            "energy_regression_dim": int(m.get("energy_regression_dim", 128)),
            "zmax": int(m.get("zmax", 118)),
            "gamma": float(m.get("gamma", BERNSTEIN_GAMMA)),
            "preset": preset,
        }

    if preset is not None:
        return {**SO3LR_PRESETS[preset], "preset": preset}

    # Last-resort inference from params shapes (only if matches a known preset)
    p = raw_params["params"] if "params" in raw_params else raw_params
    num_layers = int(np.asarray(p["resid_lambdas"]).shape[0])
    num_features = int(
        np.asarray(p["feature_embeddings_0"]["Embed_0"]["embedding"]).shape[1]
    )
    for name, spec in SO3LR_PRESETS.items():
        if spec["num_layers"] == num_layers and spec["num_features"] == num_features:
            return {**spec, "preset": name}
    raise ValueError(
        f"cannot infer SO3LR hyperparameters from {root}; pass preset=... explicitly"
    )


__all__ = [
    "BERNSTEIN_GAMMA",
    "PAIR_FLOOR_WIDTH",
    "PRESET_ALIASES",
    "SO3LR_PRESETS",
    "BernsteinRBF",
    "ChargeSpinEmbed",
    "PartialChargesHead",
    "SO3LRTrunk",
    "SO3kratesLayer",
    "floor_descriptor_distance",
    "load_nlh_rmin_table",
    "phys_cutoff",
    "real_sph_degrees",
    "resolve_preset_name",
    "rms_norm",
]
