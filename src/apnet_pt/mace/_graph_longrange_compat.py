"""Size the ``graph_longrange`` real-space scatters explicitly.

``graph_longrange`` 0.4.0 reduces its real-space electrostatic edge energies
with ``scatter_sum(src=edge_energy, index=receiver)`` and no ``dim_size``.
``mace.tools.scatter.scatter_sum`` then sizes the output from ``index.max()``,
so the per-node energy vector is truncated whenever the *trailing* nodes of the
batch receive no edges, and the following graph-level scatter fails with

    The expanded size of the tensor (812) must match the existing size (816)

A monatomic monomer produces exactly that: its duplicates are all excluded from
the complete graph by the same-original-atom mask, so it contributes no edges.
The failure therefore depends on collation order rather than on the batch
containing a monatomic monomer at all, and it costs a whole training task.

The sibling ``charges_features_from_graph`` in the same upstream module already
passes ``dim_size=num_nodes``, which is why this is read as an omission rather
than a deliberate contract.  Patching restores the intended output length and
is a no-op for every batch whose last graph has at least one edge.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

_PATCH_MARK = "_apnet_pt_dim_size_patched"

# sha256 of ``inspect.getsource(charges_energy_from_graph)`` for the defective
# graph_longrange 0.4.0 body this module is written against.
_KNOWN_DEFECTIVE_DIGEST = (
    "7e68a8ac3842d93d21637f69223cdf2fd4a589e3f740be1e96f07a9e3032e684"
)


def patch_realspace_scatter_dim_size() -> bool:
    """Replace ``charges_energy_from_graph`` with a correctly sized version.

    Returns ``True`` when the patch was installed by this call, ``False`` when
    it was already installed or the upstream implementation no longer needs it.
    Raises ``RuntimeError`` for an unrecognised upstream body, because silently
    substituting our copy of the physics for a body we have not read would turn
    a crash into wrong numbers.
    """

    from graph_longrange import realspace_electrostatics as rse

    original = rse.charges_energy_from_graph
    if getattr(original, _PATCH_MARK, False):
        return False

    source = inspect.getsource(original)
    if "dim_size" in source:
        # Upstream now sizes the reduction itself; leave it alone.
        return False
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != _KNOWN_DEFECTIVE_DIGEST:
        raise RuntimeError(
            "graph_longrange.charges_energy_from_graph does not match the "
            "version this compatibility patch was written against "
            f"(sha256 {digest}); re-read the upstream body before patching."
        )

    import torch
    from mace.tools.scatter import scatter_sum

    field_constant = rse.FIELD_CONSTANT
    pi = rse.pi

    def charges_energy_from_graph(
        charges: Any,
        positions: Any,
        edge_index: Any,
        batch: Any,
        density_smearing_width: Any,
    ):
        sender, receiver = edge_index

        R_ij = positions[receiver] - positions[sender]
        d_ij = torch.linalg.norm(R_ij, dim=-1)
        smooth_reciprocal = torch.erf(d_ij * 0.5 / density_smearing_width) / (
            torch.abs(d_ij) + 1e-6
        )

        edge_energy = (
            0.5
            * field_constant
            * smooth_reciprocal
            * charges[sender]
            * charges[receiver]
            / (4 * pi)
        )
        n_graphs = int(batch.max()) + 1
        if edge_energy.numel() == 0:
            return torch.zeros(
                (n_graphs,), dtype=charges.dtype, device=charges.device
            )
        node_energies = scatter_sum(
            src=edge_energy.squeeze(-1),
            index=receiver,
            dim=-1,
            dim_size=charges.shape[0],
        )
        return scatter_sum(
            src=node_energies, index=batch, dim=-1, dim_size=n_graphs
        )

    charges_energy_from_graph.__doc__ = original.__doc__
    charges_energy_from_graph._apnet_pt_original = original
    setattr(charges_energy_from_graph, _PATCH_MARK, True)
    rse.charges_energy_from_graph = charges_energy_from_graph
    return True
