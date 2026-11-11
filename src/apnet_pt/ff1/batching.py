"""Build FF-1 batches from dimer geometries or from mastiff-classical stores."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def intramonomer_edges(R: Tensor, monomer: Tensor, r_cut: float) -> Tensor:
    """Directed (i, k), i != k, same monomer, |R_k - R_i| < r_cut (Angstrom)."""
    same = monomer[:, None] == monomer[None, :]
    d = torch.cdist(R.detach(), R.detach())
    mask = same & (d < r_cut) & ~torch.eye(len(R), dtype=torch.bool, device=R.device)
    return torch.stack(torch.nonzero(mask, as_tuple=True))


def batch_from_dimers(dimers: Sequence[dict], r_cut: float = 5.0,
                      dtype: torch.dtype = torch.float64, device="cpu") -> dict:
    """``dimers``: dicts with ``zA``, ``RA``, ``zB``, ``RB`` (Angstrom) and
    optional ``qA``, ``qB`` (formal charges)."""
    z, R, mon, mq, adim, ai, aj, pdim = [], [], [], [], [], [], [], []
    offset = 0
    for d, dimer in enumerate(dimers):
        za, zb = list(dimer["zA"]), list(dimer["zB"])
        na, nb = len(za), len(zb)
        z += za + zb
        R.append(torch.as_tensor(dimer["RA"], dtype=dtype).reshape(na, 3))
        R.append(torch.as_tensor(dimer["RB"], dtype=dtype).reshape(nb, 3))
        mon += [2 * d] * na + [2 * d + 1] * nb
        mq += [float(dimer.get("qA", 0)), float(dimer.get("qB", 0))]
        adim += [d] * (na + nb)
        ia = torch.arange(na).repeat_interleave(nb) + offset
        jb = torch.arange(nb).repeat(na) + offset + na
        ai.append(ia)
        aj.append(jb)
        pdim.append(torch.full((na * nb,), d))
        offset += na + nb
    R = torch.cat(R).to(device)
    monomer = torch.tensor(mon, device=device)
    return {"z": torch.tensor(z, device=device), "R": R,
            "edges": intramonomer_edges(R, monomer, r_cut), "monomer": monomer,
            "monomer_charge": torch.tensor(mq, dtype=torch.float64, device=device),
            "atom_dimer": torch.tensor(adim, device=device),
            "ai": torch.cat(ai).to(device), "aj": torch.cat(aj).to(device),
            "pair_dimer": torch.cat(pdim).to(device), "n": len(dimers)}


def batch_from_store(batch: dict) -> dict:
    """A mastiff-classical ``ClassicalStore.batch`` in FF-1's format."""
    g = batch["graph"]
    return {"z": batch["atoms"]["z"].long(), "R": batch["atoms"]["R"],
            "edges": g.edge_index, "monomer": g.molecule_ind.long(),
            "monomer_charge": g.total_charge,
            "atom_dimer": g.molecule_ind.long() // 2,
            "ai": batch["ai"], "aj": batch["aj"], "pair_dimer": batch["pair_dimer"],
            "n": batch["n"]}


def with_coordinates(batch: dict, R: Tensor) -> dict:
    """Same batch (topology, edges) at new coordinates."""
    return {**batch, "R": R}
