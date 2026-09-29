"""Regression tests for the graph_longrange real-space scatter sizing patch.

A monatomic monomer contributes no real-space edges, so when it lands last in a
collated batch the upstream per-node scatter is short by that monomer's nodes
and the graph-level scatter raises.  The bug is collation-order dependent, which
is why an ordinary batch of dimers passes for a long time and then does not.
"""

import pytest
import torch

rse = pytest.importorskip("graph_longrange.realspace_electrostatics")

from apnet_pt.mace._graph_longrange_compat import (
    patch_realspace_scatter_dim_size,
)


@pytest.fixture(autouse=True)
def _restore_upstream():
    original = rse.charges_energy_from_graph
    yield
    rse.charges_energy_from_graph = original


def _batch(sizes):
    torch.manual_seed(1)
    batch = torch.cat(
        [torch.full((n,), g, dtype=torch.long) for g, n in enumerate(sizes)]
    )
    n = int(batch.numel())
    return torch.randn(n, 4) * 0.1, torch.randn(n, 3) * 3.0, batch


def _energy(max_l, sizes):
    module = rse.RealSpaceFiniteDiffereneEnergy(
        density_max_l=max_l, density_smearing_width=1.0
    )
    feats, positions, batch = _batch(sizes)
    if max_l == 0:
        feats = feats[:, :1]
    return module(feats, positions, batch)


@pytest.mark.parametrize("max_l", [0, 1])
def test_trailing_monatomic_graph_raises_without_the_patch(max_l):
    with pytest.raises(RuntimeError, match="expanded size of the tensor"):
        _energy(max_l, [5, 4, 1])


@pytest.mark.parametrize("max_l", [0, 1])
@pytest.mark.parametrize("sizes", [[5, 4, 1], [5, 4, 1, 1]])
def test_patch_admits_a_trailing_monatomic_graph(max_l, sizes):
    patch_realspace_scatter_dim_size()
    energy = _energy(max_l, sizes)
    assert energy.shape == (len(sizes),)
    # A lone atom has no distinct-atom pairs, so its real-space energy is zero.
    assert float(energy[-1]) == 0.0


@pytest.mark.parametrize("sizes", [[5, 4, 3], [5, 1, 3], [1, 5, 3], [1]])
def test_patch_is_bit_identical_where_upstream_already_worked(sizes):
    before = _energy(1, sizes)
    patch_realspace_scatter_dim_size()
    assert torch.equal(_energy(1, sizes), before)


def test_energy_does_not_depend_on_collation_order():
    patch_realspace_scatter_dim_size()
    module = rse.RealSpaceFiniteDiffereneEnergy(
        density_max_l=1, density_smearing_width=1.0
    )
    torch.manual_seed(3)
    sizes = [5, 4, 1]
    feats = [torch.randn(n, 4) * 0.1 for n in sizes]
    positions = [torch.randn(n, 3) * 3.0 for n in sizes]

    def collate(order):
        batch = torch.cat(
            [
                torch.full((sizes[g],), slot, dtype=torch.long)
                for slot, g in enumerate(order)
            ]
        )
        return module(
            torch.cat([feats[g] for g in order]),
            torch.cat([positions[g] for g in order]),
            batch,
        )

    monatomic_last = collate([0, 1, 2])
    monatomic_middle = collate([0, 2, 1])
    assert torch.equal(monatomic_last[0], monatomic_middle[0])
    assert torch.equal(monatomic_last[1], monatomic_middle[2])
    assert float(monatomic_last[2]) == 0.0
    assert float(monatomic_middle[1]) == 0.0


def test_patch_is_idempotent_and_self_retiring():
    assert patch_realspace_scatter_dim_size() is True
    assert patch_realspace_scatter_dim_size() is False


def test_patch_refuses_an_unrecognised_upstream_body(monkeypatch):
    def impostor(charges, positions, edge_index, batch, density_smearing_width):
        raise AssertionError("never called")

    monkeypatch.setattr(rse, "charges_energy_from_graph", impostor)
    with pytest.raises(RuntimeError, match="does not match the version"):
        patch_realspace_scatter_dim_size()


# --------------------------------------------------------------------------
# Non-periodic Ewald k-grid elision
# --------------------------------------------------------------------------

extensions = pytest.importorskip("mace.modules.extensions")
gl_features = pytest.importorskip("graph_longrange.features")

from apnet_pt.mace._graph_longrange_compat import (
    nonperiodic_kspace_elision,
    patch_kspace_grid_for_nonperiodic,
)


@pytest.fixture(autouse=True)
def _restore_kspace():
    original = extensions.compute_k_vectors_flat
    yield
    extensions.compute_k_vectors_flat = original


def _cube(edge=10.0):
    cell = torch.eye(3) * edge
    rcell = 2 * torch.pi * torch.linalg.inv(cell.mT)
    return cell.view(1, 3, 3), rcell.view(1, 3, 3)


def test_kspace_patch_is_idempotent():
    assert patch_kspace_grid_for_nonperiodic() is True
    assert patch_kspace_grid_for_nonperiodic() is False


def test_kspace_grid_is_built_outside_the_elision_context():
    patch_kspace_grid_for_nonperiodic()
    cell, rcell = _cube()
    kvectors, *_ = extensions.compute_k_vectors_flat(1.0, cell, rcell)
    assert kvectors.shape[0] > 0


def test_kspace_grid_is_elided_inside_the_context():
    patch_kspace_grid_for_nonperiodic()
    cell, rcell = _cube()
    real = extensions.compute_k_vectors_flat(1.0, cell, rcell)
    with nonperiodic_kspace_elision():
        elided = extensions.compute_k_vectors_flat(1.0, cell, rcell)
    # Same arity and dtypes as upstream, but no grid.  A caller that reads the
    # k-vectors gets an empty batch rather than a wrong one.
    assert len(elided) == len(real)
    assert [t.shape[0] for t in elided] == [0, 0, 0, 0]
    assert [t.dtype for t in elided] == [t.dtype for t in real]
    assert elided[0].shape[1:] == real[0].shape[1:]
    # And the context does not leak.
    assert extensions.compute_k_vectors_flat(1.0, cell, rcell)[0].shape[0] > 0


def test_elision_context_is_a_no_op_when_disabled():
    patch_kspace_grid_for_nonperiodic()
    cell, rcell = _cube()
    with nonperiodic_kspace_elision(enabled=False):
        kvectors, *_ = extensions.compute_k_vectors_flat(1.0, cell, rcell)
    assert kvectors.shape[0] > 0


def test_elision_context_restores_an_enclosing_context():
    patch_kspace_grid_for_nonperiodic()
    cell, rcell = _cube()
    with nonperiodic_kspace_elision():
        with nonperiodic_kspace_elision(enabled=False):
            pass
        # The inner no-op must not cancel the outer elision.
        assert extensions.compute_k_vectors_flat(1.0, cell, rcell)[0].shape[0] == 0


def test_kspace_patch_refuses_an_upstream_that_ignores_pbc(monkeypatch):
    def precompute_geometry(self, *args, **kwargs):
        raise AssertionError("never called")

    monkeypatch.setattr(
        gl_features.GTOElectrostaticFeatures,
        "precompute_geometry",
        precompute_geometry,
    )
    with pytest.raises(RuntimeError, match="no longer dispatches on pbc"):
        patch_kspace_grid_for_nonperiodic()
