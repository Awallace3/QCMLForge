"""Frozen monomer parameters for FF-2 MD.

FF-2 normally recomputes every atomic parameter (charges, multipoles,
polarizabilities, exchange/dispersion exponents) from each monomer's current
geometry.  Far outside the training geometries this response can run away: in
an [EMIM][TFA]/water cluster a thermally stretched EMIM C-H drives FF-2
induction to ~-1e3 kcal/mol and pulls the proton onto a TFA oxygen within
0.1 ps.  :class:`FrozenAtoms` instead evaluates the parameters once, at a
reference geometry, stores the oriented ones in per-atom local frames, and
rotates them with each atom's frame at every step (the MASTIFF/AMOEBA
convention).  For rigid-body motion of every monomer it reproduces the live
parameters exactly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..ff1.trunk import ANG2BOHR

SCALARS = ("q", "n_act", "beta", "shell_b", "A", "B", "D", "C", "eta_B",
           "alpha_iso", "eta", "beta_pol")
# component axes that rotate: vectors (1,), rank-2 tensors (1, 2), shells offset
ORIENTED = {"mu": (1,), "th": (1, 2), "shell_M": (2,), "shell_T": (2, 3),
            "vecs": (1,), "tens": (1, 2), "sqrt_alpha": (1, 2)}


def frame_indices(z: Sequence[int], bonded: np.ndarray):
    """Z-then-X local frame of every atom as indices.

    The z axis points at the heaviest bonded neighbour ``kz`` (lowest index on
    ties); the x axis is ``R[kx] - R[ko]`` orthogonalised against z, from the
    atom's next neighbour, or else a neighbour of ``kz``.
    """
    z = np.asarray(z)
    n = len(z)
    nb = [sorted(np.flatnonzero(bonded[a]), key=lambda k: (-z[k], k)) for a in range(n)]
    kz, kx, ko = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    for a in range(n):
        if not nb[a]:
            raise ValueError(f"atom {a} has no bonded neighbour")
        kz[a] = nb[a][0]
        cand = [(k, a) for k in nb[a][1:]] + [(k, kz[a]) for k in nb[kz[a]] if k != a]
        if not cand:
            raise ValueError(f"atom {a}: no x-axis atom (diatomic)")
        kx[a], ko[a] = cand[0]
    return kz, kx, ko


def frames(R: Tensor, kz: Tensor, kx: Tensor, ko: Tensor) -> Tensor:
    """(N, 3, 3) rows x, y, z of each atom's frame in lab components."""
    ez = R[kz] - R
    ez = ez / ez.norm(dim=-1, keepdim=True)
    v = R[kx] - R[ko]
    ex = v - (v * ez).sum(-1, keepdim=True) * ez
    ex = ex / ex.norm(dim=-1, keepdim=True)
    return torch.stack([ex, torch.cross(ez, ex, dim=-1), ez], 1)


def rotate(x: Tensor, M: Tensor, axes) -> Tensor:
    """Apply the per-atom 3x3 ``M`` to each listed component axis of ``x``."""
    for ax in axes:
        x = torch.movedim(x, ax, -1)
        x = (M.reshape(M.shape[0], *([1] * (x.dim() - 2)), 3, 3)
             * x[..., None, :]).sum(-1)
        x = torch.movedim(x, -1, ax)
    return x


class FrozenAtoms(nn.Module):
    """FF-2 atomic parameters frozen at a reference geometry.

    Parameters
    ----------
    model : FF2Model
        Evaluated once, at ``R_ref``; not stored.
    z, monomer, monomer_charge
        Topology, as for :class:`apnet_pt.ff2.pme.FF2LongRangeEnergy`.
    R_ref : (N, 3) reference positions in Angstrom (each monomer whole).
    bonds : (B, 2) covalent bonds (atom indices); they define the frames.

    ``forward(R_ang)`` returns ``(R_bohr, atoms)`` with the same contract as
    ``FF2Model._forward_atoms``: scalars unchanged, oriented quantities rotated
    from the reference frames into the current ones.
    """

    def __init__(self, model, z: Sequence[int], monomer: Sequence[int],
                 monomer_charge: Sequence[float], R_ref, bonds,
                 extra_batch: dict | None = None):
        super().__init__()
        from .md import cluster_batch

        n = len(z)
        bonds = np.asarray(bonds, dtype=int).reshape(-1, 2)
        bonded = np.zeros((n, n), dtype=bool)
        bonded[bonds[:, 0], bonds[:, 1]] = True
        bonded |= bonded.T
        mon = np.asarray(monomer)
        if (mon[bonds[:, 0]] != mon[bonds[:, 1]]).any():
            raise ValueError("a bond joins two monomers")
        dev = next(model.parameters()).device
        kz, kx, ko = (torch.as_tensor(a, dtype=torch.long, device=dev)
                      for a in frame_indices(z, bonded))
        self.register_buffer("kz", kz)
        self.register_buffer("kx", kx)
        self.register_buffer("ko", ko)
        self.physics_dtype = model.physics_dtype
        b = cluster_batch(z, monomer, monomer_charge, device=dev)
        Rr = torch.as_tensor(np.asarray(R_ref), dtype=torch.float64, device=dev)
        mono_in = {"z": b["z"], "R": Rr, "edges": b["edges"],
                   "monomer": b["monomer"],
                   "monomer_charge": b["monomer_charge"]}
        if extra_batch is not None:
            for k, v in extra_batch.items():
                mono_in[k] = torch.as_tensor(v, device=dev)
        with torch.no_grad():
            _, at = model._forward_atoms(mono_in)
        F = frames(Rr, kz, kx, ko)                       # lab -> local
        for key in SCALARS:
            self.register_buffer(f"at_{key}", at[key].detach().clone())
        for key, axes in ORIENTED.items():
            self.register_buffer(f"at_{key}", rotate(at[key].detach().to(F.dtype), F, axes)
                                 .to(at[key].dtype))

    def forward(self, R_ang: Tensor) -> tuple[Tensor, dict]:
        F = frames(R_ang, self.kz, self.kx, self.ko).transpose(1, 2)   # local -> lab
        at = {k: getattr(self, f"at_{k}") for k in SCALARS}
        at.update({k: rotate(getattr(self, f"at_{k}").to(F.dtype), F, axes)
                   for k, axes in ORIENTED.items()})
        return R_ang.to(self.physics_dtype) * ANG2BOHR, at


__all__ = ["FrozenAtoms", "frame_indices", "frames", "rotate"]
