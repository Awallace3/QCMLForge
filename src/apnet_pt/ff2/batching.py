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

with_coordinates = ff1_batching.with_coordinates


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
        return batch
    out = {k: v for k, v in batch.items()
           if k not in ("fragment", "fragment_charge")}
    out.update(dimer_layout(batch["atom_dimer"], int(batch["n"])))
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
