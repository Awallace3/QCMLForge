"""Every registered MACE-fused route forwards, trains, and respects symmetry.

These run against ``tests.mace_stub_harness``, so they exercise the fusion
seams -- pair core, ledger accounting, classical injection -- without the
licensed foundation checkpoint.  A route that reaches the registry but not the
pair core, or that loses the geometry's symmetry while slicing equivariant
features, fails here rather than after a training run.
"""

from copy import deepcopy

import pytest
import torch

from apnet_pt.mace.model import MACEAP3D3Model
from apnet_pt.mace.pair import MACE_E3NN_AXIS_PERMUTATION

from tests.mace_stub_harness import ROUTES, _augment_batch, _batch, _make_model


def _permute_batch(batch, order_a, order_b):
    inverse_a = torch.argsort(order_a)
    inverse_b = torch.argsort(order_b)
    permuted = deepcopy(batch)
    permuted.ZA, permuted.RA = batch.ZA[order_a], batch.RA[order_a]
    permuted.ZB, permuted.RB = batch.ZB[order_b], batch.RB[order_b]
    permuted.molecule_ind_A = batch.molecule_ind_A[order_a]
    permuted.molecule_ind_B = batch.molecule_ind_B[order_b]
    for prefix, inverse in (("e_AA", inverse_a), ("e_BB", inverse_b)):
        setattr(permuted, f"{prefix}_source", inverse[getattr(batch, f"{prefix}_source")])
        setattr(permuted, f"{prefix}_target", inverse[getattr(batch, f"{prefix}_target")])
    for prefix in ("e_ABsr", "e_ABlr", "e_ABfull"):
        setattr(permuted, f"{prefix}_source", inverse_a[getattr(batch, f"{prefix}_source")])
        setattr(permuted, f"{prefix}_target", inverse_b[getattr(batch, f"{prefix}_target")])
    return permuted


@pytest.mark.parametrize("route", ROUTES)
def test_route_forward_backward_and_component_ledgers(route):
    torch.manual_seed(41)
    model, provider, long_range = _make_model(route)
    batch = _augment_batch(_batch())
    labels_before = batch.y.clone()
    details = model(batch, return_details=True)

    assert details.components.shape == (1, 4)
    assert torch.isfinite(details.components).all()
    assert tuple(details.component_ledger) == ("elst", "exch", "indu", "disp")
    assert tuple(details.residual_ledger) == ("elst", "exch", "indu", "disp")
    assert set(details.classical_ledger) == {"elst", "indu", "disp"}
    assert details.induction_diagnostics.converged
    # The classical bundle is added to the neural residual exactly once, and
    # never to the exchange column, which has no classical counterpart.
    expected = details.residual.clone()
    expected[:, 0] += 0.1
    expected[:, 2] += 0.2
    expected[:, 3] += 0.3
    assert torch.allclose(details.components, expected)
    assert long_range.calls == long_range.dispersion_calls == 1
    assert not model.featurizer.backbone.training
    assert all(
        not parameter.requires_grad
        for parameter in model.featurizer.backbone.parameters()
    )
    if route == "direct-polar":
        assert provider.direct_calls == 2

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    harness = MACEAP3D3Model(model)
    before = model.pair_core.h0_projection.weight.detach().clone()
    loss = harness.train_step(batch, optimizer)
    assert torch.isfinite(loss)
    assert set(harness.last_component_losses) == {"elst", "exch", "indu", "disp"}
    assert not torch.equal(before, model.pair_core.h0_projection.weight)
    assert torch.equal(batch.y, labels_before)
    assert harness.predict_batch(batch).shape == (1, 4)


def test_no_disp_neural_route_retains_d3_exactly_once():
    model, _, long_range = _make_model("hybrid-h2", no_disp=True)
    details = model(_augment_batch(_batch()), return_details=True)
    assert torch.equal(details.residual[:, 3], torch.zeros_like(details.residual[:, 3]))
    assert torch.equal(details.components[:, 3], torch.full((1,), 0.3))
    assert long_range.calls == long_range.dispersion_calls == 1


@pytest.mark.parametrize("route", ROUTES)
def test_route_rotation_translation_and_permutation_equivalence(route):
    torch.manual_seed(47)
    model, _, _ = _make_model(route)
    batch = _augment_batch(_batch())
    reference = model(batch)

    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = torch.tensor([1.2, -0.8, 0.4])
    transformed = deepcopy(batch)
    transformed.RA = batch.RA @ rotation.T + shift
    transformed.RB = batch.RB @ rotation.T + shift
    assert torch.allclose(model(transformed), reference, atol=3.0e-6)

    permuted = _permute_batch(batch, torch.tensor([1, 0]), torch.tensor([1, 0]))
    assert torch.allclose(model(permuted), reference, atol=3.0e-6)


def test_frame_permutation_matches_mace():
    """The frame constant is only safe while MACE agrees with it.

    ``MACE_E3NN_AXIS_PERMUTATION`` is restated in ``pair.py`` because the
    directional routes contract against MACE's equivariant output and a
    disagreement here is not an error -- it is a rotationally non-invariant
    contraction that still trains.
    """

    extensions = pytest.importorskip("mace.modules.extensions")
    probe = torch.tensor([[0.0, 1.0, 2.0]])
    permuted = extensions._permute_to_e3nn_convention(probe)
    assert torch.equal(permuted, probe[:, list(MACE_E3NN_AXIS_PERMUTATION)])
