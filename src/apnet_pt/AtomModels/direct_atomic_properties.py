"""Absolute, energy-trained atomic multipoles and effective volume ratios.

No pretrained model or atomic-property labels are needed. R is in Angstrom;
q, mu and symmetric-traceless quadrupoles use the package's atomic units.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class DirectAtomicProperties(nn.Module):
    """Predict monomer-conditioned properties with exact charge conservation.

    Parameters
    ----------
    hfvr_min, hfvr_max : float
        Strict positive bounds on effective HFVR, not a stability guarantee.
    hfvr_init : float
        Initial reference ratio (the small readout perturbation is nonzero).
    readout_init_scale : float
        Small, nonzero initialization avoids an all-zero neutral field.

    Notes
    -----
    The graph needs x (atomic numbers), R, directed intramonomer edge_index,
    molecule_ind, and total_charge (one value per monomer). Charge conditioning
    affects all heads, not just the final charge projection. Dipoles and
    quadrupoles are neighbor-vector contractions, including for empty graphs.
    """

    def __init__(
        self,
        n_message: int = 3,
        n_rbf: int = 8,
        n_neuron: int = 128,
        n_embed: int = 8,
        r_cut: float = 5.0,
        hfvr_min: float = 0.001,
        hfvr_max: float = 2.0,
        hfvr_init: float = 1.0,
        readout_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0 < hfvr_min < hfvr_init < hfvr_max:
            raise ValueError("HFVR requires 0 < min < init < max")
        if n_message < 1 or min(n_rbf, n_neuron, n_embed) < 1 or r_cut <= 0:
            raise ValueError("network dimensions and cutoff must be positive")
        if not 0 < readout_init_scale <= 1:
            raise ValueError("readout_init_scale must be in (0, 1]")
        self.hfvr_min, self.hfvr_max = float(hfvr_min), float(hfvr_max)
        self.r_cut = float(r_cut)
        self.embed = nn.Embedding(119, n_embed)
        self.charge_embed = nn.Linear(2, n_embed)
        self.register_buffer("rbf_centers", torch.linspace(0.0, r_cut, n_rbf))
        self.rbf_width = max(r_cut / n_rbf, 0.1)
        self.messages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * n_embed + n_rbf, n_neuron),
                    nn.SiLU(),
                    nn.Linear(n_neuron, n_embed),
                )
                for _ in range(n_message)
            ]
        )
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * n_embed, n_neuron),
                    nn.SiLU(),
                    nn.Linear(n_neuron, n_embed),
                )
                for _ in range(n_message)
            ]
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(n_embed, n_neuron), nn.SiLU(), nn.Linear(n_neuron, 2)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * n_embed + n_rbf, n_neuron), nn.SiLU(), nn.Linear(n_neuron, 2)
        )
        fraction = (hfvr_init - hfvr_min) / (hfvr_max - hfvr_min)
        self.hfvr_logit_init = math.log(fraction / (1 - fraction))
        for head in (self.scalar_head, self.edge_head):
            with torch.no_grad():
                head[-1].weight.mul_(readout_init_scale)
                head[-1].bias.zero_()

    def forward(self, graph: Any) -> dict[str, torch.Tensor]:
        """Return absolute q, mu, quad and hfvr; never return MBIS corrections."""
        z, mol, R = graph.x.long(), graph.molecule_ind.long(), graph.R
        total = graph.total_charge.to(R).reshape(-1)
        n_atoms, n_mol = len(z), total.numel()
        counts = R.new_zeros(n_mol).index_add(0, mol, R.new_ones(n_atoms))
        local_charge = torch.stack((total, total / counts.clamp_min(1)), -1)[mol]
        h = self.embed(z) + self.charge_embed(local_charge)
        src, tgt = graph.edge_index
        vec = R[tgt] - R[src]
        dist = vec.norm(dim=-1)
        # Zero-length/self edges are invalid input; validate at dataset boundary.
        u = vec / dist[:, None].clamp_min(torch.finfo(R.dtype).eps)
        envelope = (
            0.5 * (1 + torch.cos(math.pi * dist / self.r_cut)) * (dist < self.r_cut)
        )
        rbf = torch.exp(-(((dist[:, None] - self.rbf_centers) / self.rbf_width) ** 2))
        rbf = rbf * envelope[:, None]
        for message, update in zip(self.messages, self.updates):
            edge = torch.cat((h[src], h[tgt], rbf), -1)
            msg = message(edge) * envelope[:, None]
            aggregate = h.new_zeros(h.shape).index_add(0, src, msg)
            h = h + update(torch.cat((h, aggregate), -1))
        scalar = self.scalar_head(h)
        sums = scalar.new_zeros(n_mol).index_add(0, mol, scalar[:, 0])
        q = scalar[:, 0] + ((total - sums) / counts.clamp_min(1))[mol]
        weights = self.edge_head(torch.cat((h[src], h[tgt], rbf), -1))
        weights = weights * envelope[:, None]
        mu = R.new_zeros((n_atoms, 3)).index_add(0, src, weights[:, :1] * u)
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        shape = 1.5 * u[:, :, None] * u[:, None, :] - 0.5 * eye
        quad = R.new_zeros((n_atoms, 3, 3)).index_add(
            0, src, weights[:, 1, None, None] * shape
        )
        hfvr = self.hfvr_min + (self.hfvr_max - self.hfvr_min) * torch.sigmoid(
            self.hfvr_logit_init + scalar[:, 1]
        )
        return {"q": q, "mu": mu, "quad": quad, "hfvr": hfvr}
