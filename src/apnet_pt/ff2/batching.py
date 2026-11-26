"""FF-2 batches: FF-1's batch dict plus every data-dependent index, built once
on the host side so the model forward is shape-polymorphic and sync-free.

``prepare(batch)`` adds

==================  ====================================================
``pol_k``, ``pol_l``  (Q,) atom pairs k < l of the same dimer (both
                      monomers): the polarization interaction list
``pol_dimer``         (Q,) their dimer
``pol_lk``, ``pol_ll`` (Q,) their local slots inside the dimer
``atom_local``        (N,) each atom's slot inside its dimer
``slots``             (n, m) bool: slot occupied (m = largest dimer)
``n_dimer``           (n,) zeros; carries the dimer count as a shape
==================  ====================================================

and drops the FF-1 keys that only the covalent-fragment rule reads.  The
dimer and slot counts travel as tensor *shapes*, never Python ints, so a
compiled forward sees them as symbolic sizes.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ..ff1 import batching as ff1_batching
from ..ff1 import charges as ff1_charges
from ..ff1 import elements as ff1_elements

with_coordinates = ff1_batching.with_coordinates

_R_COV_CACHE: dict[torch.device, Tensor] = {}


def _r_cov_table(device: torch.device) -> Tensor:
    t = _R_COV_CACHE.get(device)
    if t is None:
        t = torch.tensor(ff1_elements.table("r_cov"), dtype=torch.float64, device=device)
        _R_COV_CACHE[device] = t
    return t


@torch.no_grad()
def covalent_fragments(z: Tensor, R: Tensor, edges: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Covalent connected components of the atoms in a batch.

    Returns ``(edge_same_frag, atom_frag, n_frag)`` where ``edge_same_frag``
    is ``(E,)`` bool, ``atom_frag`` is ``(N,)`` long in ``0 .. nf - 1``, and
    ``n_frag`` is ``(nf,)`` zeros carrying ``nf`` as a tensor shape for
    ``torch.compile``.
    """
    n = len(z)
    if edges.numel() == 0:
        frag = torch.arange(n, dtype=torch.long, device=z.device)
        return (torch.ones((0,), dtype=torch.bool, device=z.device),
                frag, torch.zeros(n, dtype=torch.long, device=z.device))
    src, dst = edges
    r_cov = _r_cov_table(z.device)
    r = (R[dst].double() - R[src].double()).norm(dim=-1)
    bonded = r < 1.25 * (r_cov[z[src]] + r_cov[z[dst]])
    frag = ff1_charges.connected_components(n, edges[:, bonded])
    nf = int(frag.max()) + 1 if n > 0 else 0
    return frag[src] == frag[dst], frag, torch.zeros(nf, dtype=torch.long, device=z.device)


@torch.no_grad()
def edge_same_fragment(z: Tensor, R: Tensor, edges: Tensor) -> Tensor:
    """Bool mask (E,) true iff both endpoints lie in the same covalent connected component."""
    return covalent_fragments(z, R, edges)[0]


def dimer_layout(atom_dimer: Tensor, n_dimer: int) -> dict:
    """Slot table and k < l polarization pairs (one host sync for the max)."""
    dev = atom_dimer.device
    counts = torch.bincount(atom_dimer, minlength=n_dimer)
    offsets = torch.cumsum(counts, 0) - counts
    idx = torch.arange(len(atom_dimer), device=dev)
    local = idx - offsets[atom_dimer]
    m = int(counts.max()) if n_dimer else 0
    table = torch.full((n_dimer, m), -1, dtype=torch.long, device=dev)
    table[atom_dimer, local] = idx
    slots = table >= 0
    upper = torch.triu(torch.ones(m, m, dtype=torch.bool, device=dev), diagonal=1)
    grid = slots[:, :, None] & slots[:, None, :] & upper[None]
    d, lk, ll = torch.nonzero(grid, as_tuple=True)
    return {"pol_k": table[d, lk], "pol_l": table[d, ll], "pol_dimer": d,
            "pol_lk": lk, "pol_ll": ll, "atom_local": local, "slots": slots,
            "n_dimer": torch.zeros(n_dimer, dtype=torch.long, device=dev)}


def prepare(batch: dict) -> dict:
    """An FF-1 batch (``batch_from_dimers``/``batch_from_store`` format) in
    FF-2's format; idempotent."""
    if "slots" in batch:
        if "R" in batch and ("edge_same_frag" not in batch or "atom_frag" not in batch
                             or "n_frag" not in batch):
            same, frag, n_frag = covalent_fragments(
                batch["z"].long(), batch["R"], batch["edges"])
            return {**batch, "edge_same_frag": same, "atom_frag": frag,
                    "n_frag": n_frag}
        return batch
    out = {k: v for k, v in batch.items()
           if k not in ("fragment", "fragment_charge")}
    out.update(dimer_layout(batch["atom_dimer"], int(batch["n"])))
    if "R" in batch:
        same, frag, n_frag = covalent_fragments(
            batch["z"].long(), batch["R"], batch["edges"])
        out["edge_same_frag"] = same
        out["atom_frag"] = frag
        out["n_frag"] = n_frag
    return out


def batch_from_dimers(dimers: Sequence[dict], r_cut: float = 5.0,
                      dtype: torch.dtype = torch.float64, device="cpu") -> dict:
    """``ff1.batching.batch_from_dimers`` followed by :func:`prepare`."""
    return prepare(ff1_batching.batch_from_dimers(dimers, r_cut, dtype, device))


def batch_from_store(batch: dict) -> dict:
    """A mastiff-classical ``ClassicalStore.batch`` in FF-2's format."""
    return prepare(ff1_batching.batch_from_store(batch))


def monomer_batch(batch: dict) -> dict:
    """The FF-1 batch keys FF-1 reads in one-mu (monomer) mode: every monomer
    is one fragment carrying the monomer charge (the parity reference)."""
    out = {k: v for k, v in batch.items()
           if k not in ("pol_k", "pol_l", "pol_dimer", "pol_lk", "pol_ll",
                        "atom_local", "slots", "n_dimer")}
    out["fragment"] = batch["monomer"].clone()
    out["fragment_charge"] = batch["monomer_charge"].to(torch.float64)
    return out
