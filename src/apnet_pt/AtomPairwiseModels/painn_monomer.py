"""PaiNN message passing for the AP3-D3 pair model's monomer featurization.

An equivariant alternative to ``APNet3D3_AtomType_MPNN``'s intramonomer
update/directional layers (``monomer_mpnn="painn"``).  PaiNN (Schuett, Unke
and Gastegger, ICML 2021) keeps a scalar channel ``s`` [natom, F] and a vector
channel ``v`` [natom, 3, F] and updates both with rotation-equivariant
interaction and update blocks.  ``s`` is invariant and ``v`` rotates with the
molecule, so they slot in for the pair model's invariant hidden state ``h``
and directional state ``h_dir``, which the pair readout projects onto each
intermolecular axis.

Edge convention is the pair model's: each atom ``e_source`` gathers from its
neighbour ``e_target``, and edges never cross monomers.
"""

import math

import torch
import torch.nn as nn

from ..util import scatter_sum_compile


def cosine_cutoff(r, r_cut):
    """0.5 (cos(pi r / r_cut) + 1) inside r_cut, 0 outside."""
    return torch.where(
        r < r_cut,
        0.5 * (torch.cos(math.pi * r / r_cut) + 1.0),
        torch.zeros_like(r),
    )


class SineBasis(nn.Module):
    """sin(n pi r / r_cut) / r, n = 1 .. n_rbf (PaiNN's radial basis)."""

    def __init__(self, n_rbf, r_cut):
        super().__init__()
        self.r_cut = float(r_cut)
        self.register_buffer(
            "n", torch.arange(1, n_rbf + 1, dtype=torch.float32) * math.pi
        )

    def forward(self, r):
        r = r.unsqueeze(-1)
        return torch.sin(self.n.to(r.dtype) * r / self.r_cut) / r


class PaiNNMonomer(nn.Module):
    """``n_interactions`` PaiNN message + update blocks on one monomer graph.

    ``forward(Z, R, e_source, e_target) -> (s, v)`` with ``s`` [natom, width]
    and ``v`` [natom, 3, width].  ``layer_norm`` normalises the scalar channel
    after every interaction (an unnormalised PaiNN diverged in an exchange
    trunk when one molecule's scalars grew without bound).
    """

    def __init__(self, width=128, n_interactions=3, n_rbf=8, r_cut=5.0,
                 layer_norm=True, max_z=118):
        super().__init__()
        F = self.width = int(width)
        self.n_interactions, self.n_rbf = int(n_interactions), int(n_rbf)
        self.r_cut, self.layer_norm = float(r_cut), bool(layer_norm)
        self.embed = nn.Embedding(max_z + 1, F)
        self.rbf = SineBasis(n_rbf, r_cut)
        self.phi = nn.ModuleList(
            nn.Sequential(nn.Linear(F, F), nn.SiLU(), nn.Linear(F, 3 * F))
            for _ in range(n_interactions))
        self.filters = nn.ModuleList(
            nn.Linear(n_rbf, 3 * F) for _ in range(n_interactions))
        self.U = nn.ModuleList(
            nn.Linear(F, F, bias=False) for _ in range(n_interactions))
        self.V = nn.ModuleList(
            nn.Linear(F, F, bias=False) for _ in range(n_interactions))
        self.update = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * F, F), nn.SiLU(), nn.Linear(F, 3 * F))
            for _ in range(n_interactions))
        self.norms = nn.ModuleList(
            nn.LayerNorm(F) if layer_norm else nn.Identity()
            for _ in range(n_interactions))

    def get_config(self):
        return {"width": self.width, "n_interactions": self.n_interactions,
                "n_rbf": self.n_rbf, "r_cut": self.r_cut,
                "layer_norm": self.layer_norm}

    def forward(self, Z, R, e_source, e_target):
        natom, F = Z.size(0), self.width
        s = self.embed(Z.long())
        v = s.new_zeros((natom, 3, F))
        vec = R[e_target] - R[e_source]
        r = torch.sqrt((vec * vec).sum(-1).clamp_min(1e-10))
        unit = vec / r.unsqueeze(-1)
        cut = cosine_cutoff(r, self.r_cut).unsqueeze(-1)
        rbf = self.rbf(r)
        for phi, W, U, V, update, norm in zip(self.phi, self.filters, self.U,
                                              self.V, self.update, self.norms):
            # message: neighbour e_target -> atom e_source
            x = phi(s)[e_target] * W(rbf) * cut
            ds, dvv, dvs = x.split(F, dim=-1)
            dv = v[e_target] * dvv.unsqueeze(1) + dvs.unsqueeze(1) * unit.unsqueeze(-1)
            s = s + scatter_sum_compile(ds, e_source, natom)
            v = v + scatter_sum_compile(dv, e_source, natom)
            # update: mix channels per atom
            Uv, Vv = U(v), V(v)
            v_norm = torch.sqrt((Vv * Vv).sum(1) + 1e-8)
            avv, asv, ass = update(torch.cat([s, v_norm], -1)).split(F, dim=-1)
            v = v + avv.unsqueeze(1) * Uv
            s = norm(s + asv * (Uv * Vv).sum(1) + ass)
        return s, v
