"""FF-2 as a fixed-topology potential for molecular dynamics (OpenMM TorchForce).

:class:`FF2Energy` maps positions (nm, OpenMM's unit) to FF-2's total
intermolecular energy (kJ/mol), re-running the whole FF-2 forward each call:
monomer parameters from each monomer's current geometry, the one-mu charge
solve, Cholesky polarization, CT and BJ dispersion.  Forces come from autograd.

The topology is fixed once, at construction:

* intramonomer edges are **every** same-monomer atom pair, not the pairs
  inside ``r_cut``.  FF-1's trunk and bond-flow charges weight each edge by
  the C^3 cutoff ``f_cut``, which is exactly zero (value and gradient) past
  ``r_cut``, so the result equals FF-2 on a batch rebuilt at every geometry,
  while the edge list never changes as the molecule flexes;
* every monomer sits in one cluster (FF-2's "dimer"): pair terms run over
  every pair of atoms on different monomers, the permanent field on every atom
  sums over the other monomers, and polarization couples every atom pair.
  For two monomers this is exactly FF-2's dimer energy.

openmm-torch (1.5) loads TorchScript only.  ``FF2Model.forward`` does not
script (dict expansion, closures, heterogeneous kernel-spec tuples,
``getattr`` by name), but it has no host syncs and no data-dependent branches,
so :func:`trace_energy` records it with ``torch.jit.trace`` as one TorchScript
graph that is exact for this topology.  The model's source is unchanged and
still compiles with ``torch.compile``.
"""

from __future__ import annotations

import copy
import warnings
from typing import Sequence

import torch
from torch import Tensor, nn

from .batching import prepare
from .model import FF2Model

NM2ANG = 10.0
KCAL2KJ = 4.184
# keys of the static batch; R is the only input
STATIC_KEYS = ("z", "edges", "monomer", "monomer_charge", "atom_dimer", "ai",
               "aj", "pair_dimer", "pol_k", "pol_l", "pol_dimer", "pol_lk",
               "pol_ll", "atom_local", "slots", "n_dimer")


def cluster_batch(z: Sequence[int], monomer: Sequence[int],
                  monomer_charge: Sequence[float], device="cpu") -> dict:
    """Static FF-2 batch (no ``R``) of one cluster of monomers.

    Parameters
    ----------
    z : (N,) atomic numbers.
    monomer : (N,) monomer index of each atom, ``0 .. M-1``.
    monomer_charge : (M,) formal charge of each monomer.

    Returns
    -------
    dict
        The :func:`apnet_pt.ff2.batching.prepare` keys with every same-monomer
        pair as an edge and every different-monomer pair ``i < j`` as an
        intermolecular pair.
    """
    z = torch.as_tensor(z, dtype=torch.long, device=device)
    mon = torch.as_tensor(monomer, dtype=torch.long, device=device)
    q = torch.as_tensor(monomer_charge, dtype=torch.float64, device=device)
    n = z.shape[0]
    if mon.shape != (n,) or q.shape[0] != int(mon.max()) + 1:
        raise ValueError("monomer must label N atoms with 0 .. M-1, and "
                         "monomer_charge must have M entries")
    same = mon[:, None] == mon[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=device)
    edges = torch.stack(torch.nonzero(same & ~eye, as_tuple=True))
    ai, aj = torch.nonzero(torch.triu(~same, diagonal=1), as_tuple=True)
    zeros = torch.zeros(n, dtype=torch.long, device=device)
    batch = {"z": z, "edges": edges, "monomer": mon, "monomer_charge": q,
             "atom_dimer": zeros, "ai": ai, "aj": aj,
             "pair_dimer": torch.zeros_like(ai), "n": 1}
    out = prepare(batch)
    return {k: out[k] for k in STATIC_KEYS}


class FF2Energy(nn.Module):
    """FF-2 interaction energy of a fixed cluster: positions (N, 3) in nm to
    kJ/mol (:meth:`forward`) or ``[elst, exch, ind, disp]`` in kcal/mol
    (:meth:`components`).

    Parameters
    ----------
    model : FF2Model
        Used as is (share or copy it yourself; :func:`trace_energy` copies).
    z, monomer, monomer_charge
        See :func:`cluster_batch`.
    """

    def __init__(self, model: FF2Model, z: Sequence[int], monomer: Sequence[int],
                 monomer_charge: Sequence[float]):
        super().__init__()
        self.model = model
        device = next(model.parameters()).device
        for key, value in cluster_batch(z, monomer, monomer_charge, device).items():
            self.register_buffer(f"batch_{key}", value)

    def batch(self, positions: Tensor) -> dict:
        """The FF-2 batch at ``positions`` (nm)."""
        out = {k: getattr(self, f"batch_{k}") for k in STATIC_KEYS}
        out["R"] = positions.to(self.model.physics_dtype) * NM2ANG
        return out

    def components(self, positions: Tensor) -> Tensor:
        """(4,) ``[elst, exch, ind, disp]`` in kcal/mol."""
        pred, _ = self.model(self.batch(positions))
        return pred[0]

    def forward(self, positions: Tensor) -> Tensor:
        return self.components(positions).sum() * KCAL2KJ


def trace_energy(model: FF2Model, z: Sequence[int], monomer: Sequence[int],
                 monomer_charge: Sequence[float],
                 positions: Tensor) -> torch.jit.ScriptModule:
    """TorchScript :class:`FF2Energy` for OpenMM's ``TorchForce``.

    The model is deep-copied, put in eval mode and frozen (no parameter
    gradients), then traced at ``positions`` (nm, (N, 3)).  The trace is
    valid at every geometry of this topology; save it with
    ``torch.jit.save`` and pass the file to ``openmmtorch.TorchForce``.
    """
    frozen = copy.deepcopy(model).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    module = FF2Energy(frozen, z, monomer, monomer_charge)
    with torch.no_grad():  # build kernel tables now, so the trace holds them
        module(positions.detach())  # as constants instead of recording the build
    with warnings.catch_warnings():
        # constant torch.tensor()s (quadrature nodes, bounds) and fixed-length
        # unpacking are exact for a fixed topology
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", message=".*torch.jit.trace.*deprecated")
        return torch.jit.trace(module, positions.detach(), check_trace=False)


class CompiledEnergy(nn.Module):
    """:class:`FF2Energy` compiled by Inductor as one graph (forward and, via
    AOTAutograd, its backward), for OpenMM's ``PythonForce`` or any Python
    MD loop.  See :func:`compile_energy`."""

    def __init__(self, module: FF2Energy, **options):
        super().__init__()
        self.module = module
        # Only the four components are compiled.  With frozen parameters,
        # table kernels and a grad-requiring input, torch 2.10's Inductor
        # (CPU) returns a wrong ``components.sum() * KCAL2KJ`` (one 3-monomer
        # case: -373.67 against -277.58 kJ/mol) while the components
        # themselves are exact; the scalar epilogue stays eager.
        self._compiled = torch.compile(module.components,
                                       **{"fullgraph": True, "dynamic": False,
                                          **options})

    def forward(self, positions: Tensor) -> Tensor:
        # the point_x split gathers a data-dependent number of near pairs
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            return self._compiled(positions).sum() * KCAL2KJ

    def energy_forces(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """(kJ/mol, kJ/mol/nm (N, 3)) at ``positions`` (nm)."""
        x = positions.detach().to(self.module.model.physics_dtype).requires_grad_(True)
        e = self(x)
        (g,) = torch.autograd.grad(e, x)
        return e.detach(), -g


def compile_energy(model: FF2Model, z: Sequence[int], monomer: Sequence[int],
                   monomer_charge: Sequence[float], positions: Tensor,
                   **options) -> CompiledEnergy:
    """Inductor-compiled FF-2 energy of a fixed cluster.

    Like :func:`trace_energy`, the model is deep-copied, put in eval mode and
    frozen.  One eager call at ``positions`` (nm) first builds the kernel
    tables (``kernel_mode = "table"``), which are module-level caches that
    would otherwise break the graph.  The first compiled call compiles (about
    two minutes on one CPU core).  ``options`` go to ``torch.compile``.
    """
    frozen = copy.deepcopy(model).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    module = FF2Energy(frozen, z, monomer, monomer_charge)
    with torch.no_grad():
        module(positions.detach())
    return CompiledEnergy(module, **options)


__all__ = ["CompiledEnergy", "FF2Energy", "KCAL2KJ", "NM2ANG", "cluster_batch",
           "compile_energy", "trace_energy"]
