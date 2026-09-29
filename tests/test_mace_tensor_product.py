"""Public contracts for two-centre MACE angular correlations."""

import io
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from apnet_pt.mace.pair import (
    MACE_E3NN_AXIS_PERMUTATION,
    pair_tensor_invariants,
    real_spherical_harmonics,
)


def test_transverse_vectors_distinguish_twist_invisible_to_axis_projections():
    # In an l=1 Cartesian basis, both x and y are perpendicular to the
    # interatomic z axis: all one-centre axial contractions are zero.
    a = torch.tensor([[[1.0, 0.0, 0.0]]])
    parallel = a.clone()
    perpendicular = torch.tensor([[[0.0, 1.0, 0.0]]])
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.equal((a * axis[:, None]).sum(-1), torch.zeros(1, 1))
    assert torch.equal((perpendicular * axis[:, None]).sum(-1), torch.zeros(1, 1))

    torch.testing.assert_close(pair_tensor_invariants(a, parallel), torch.ones(1, 1))
    torch.testing.assert_close(
        pair_tensor_invariants(a, perpendicular), torch.zeros(1, 1)
    )


def _twist_inputs():
    from tests.test_mace_h3_pair import _equivariant_features, _h3_fixture

    _, batch, fa, fb, pa, pb = _h3_fixture("h3l3")
    # One observed atom pair, with both monomer descriptors carrying the
    # orientation of a transverse local environment. Other atom pairs can
    # encode twist through their distances; this is an edge-level probe.
    batch.e_ABsr_source = torch.tensor([0])
    batch.e_ABsr_target = torch.tensor([0])
    batch.dimer_ind = torch.tensor([0])
    batch.RA[0] = torch.tensor([0.0, 0.0, 0.0])
    batch.RB[0] = torch.tensor([0.0, 0.0, 3.0])

    def features(numbers, direction):
        vector = torch.tensor([direction])[:, list(MACE_E3NN_AXIS_PERMUTATION)]
        y = real_spherical_harmonics(3, vector)
        block = y.expand(numbers.numel(), 4, 7).reshape(numbers.numel(), -1)
        equivariant = torch.cat([torch.zeros(numbers.numel(), 36), block], -1)
        return _equivariant_features(
            numbers, torch.zeros(numbers.numel(), 16), equivariant
        )

    fa = features(batch.ZA, [1.0, 0.0, 0.0])
    fb = features(batch.ZB, [1.0, 0.0, 0.0])
    twisted = features(batch.ZB, [0.0, 1.0, 0.0])
    return batch, fa, fb, twisted, pa, pb


def test_tensor_route_resolves_twist_that_axial_control_cannot():
    from tests.test_mace_h3_pair import _h3_fixture

    batch, fa, fb, twisted, pa, pb = _twist_inputs()
    control = _h3_fixture("h3l3")[0]
    torch.testing.assert_close(
        control(batch, fa, fb, pa, pb),
        control(batch, fa, twisted, pa, pb),
        atol=1e-8,
        rtol=0,
    )
    tensor = _h3_fixture("h3l3", architecture_id="hybrid-h3l3t")[0]
    reference = tensor(batch, fa, fb, pa, pb)
    changed = tensor(batch, fa, twisted, pa, pb)
    assert (changed - reference).abs().max() > 1e-6


def test_tensor_route_identity_and_checkpoint_roundtrip():
    from apnet_pt.training.mace_ap3d3_factory import resolve_mace_option

    from tests.test_mace_h3_pair import _h3_fixture
    from tests.test_mace_model_harness import _mount

    resolved = resolve_mace_option("MACE-AP3D3-H3L3T")
    assert resolved.internal_architecture == "hybrid-h3l3t"
    assert resolved.pair_mode == "h3l3"
    core, batch, fa, fb, pa, pb = _h3_fixture(
        "h3l3", architecture_id=resolved.internal_architecture
    )
    _mount(resolved.internal_architecture, core)
    expected = core(batch, fa, fb, pa, pb)
    buffer = io.BytesIO()
    torch.save(core.state_dict(), buffer)
    buffer.seek(0)
    fresh = _h3_fixture("h3l3", architecture_id=resolved.internal_architecture)[0]
    fresh.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(fresh(batch, fa, fb, pa, pb), expected, atol=0, rtol=0)

    control = _h3_fixture("h3l3")[0]
    assert "pair_tensor_product" not in control.get_config()
    with pytest.raises(RuntimeError, match="does not match state_dict"):
        control.load_state_dict(core.state_dict())
    with pytest.raises(RuntimeError, match="does not match state_dict"):
        core.load_state_dict(control.state_dict())
    with pytest.raises(ValueError, match="pair_tensor_product"):
        _mount(resolved.internal_architecture, control)
    with pytest.raises(ValueError):
        _mount("hybrid-h3l3", fresh)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("reflection", [False, True])
def test_tensor_features_are_o3_scalars_and_swap_symmetric(degree, reflection):
    from apnet_pt.mace.encoder import _e3nn_o3

    torch.manual_seed(82)
    o3 = _e3nn_o3()
    transform = o3.rand_matrix(dtype=torch.float64)
    if reflection:
        transform = -transform
    wigner = o3.Irrep(degree, (-1) ** degree).D_from_matrix(transform)
    a = torch.randn(5, 4, 2 * degree + 1, dtype=torch.float64)
    b = torch.randn_like(a)
    expected = pair_tensor_invariants(a, b)
    torch.testing.assert_close(pair_tensor_invariants(b, a), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        pair_tensor_invariants(a @ wigner.T, b @ wigner.T),
        expected,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("reflection", [False, True])
def test_tensor_pair_energy_is_o3_invariant(reflection):
    from apnet_pt.mace.encoder import _e3nn_o3

    from tests.test_mace_h3_pair import _h3_fixture

    core, batch, fa, fb, pa, pb = _h3_fixture("h3l3", architecture_id="hybrid-h3l3t")
    expected = core(batch, fa, fb, pa, pb)
    o3 = _e3nn_o3()
    transform = o3.rand_matrix()
    if reflection:
        transform = -transform
    permutation = torch.eye(3)[list(MACE_E3NN_AXIS_PERMUTATION)]
    internal = permutation @ transform @ permutation.T

    def rotate(features):
        blocks = [
            (
                features.equivariant_degree(degree)
                @ o3.Irrep(degree, (-1) ** degree).D_from_matrix(internal).T
            ).reshape(features.natom, -1)
            for degree in range(4)
        ]
        return replace(features, equivariant=torch.cat(blocks, dim=1))

    moved = deepcopy(batch)
    moved.RA = batch.RA @ transform.T + 4.0
    moved.RB = batch.RB @ transform.T + 4.0
    torch.testing.assert_close(
        core(moved, rotate(fa), rotate(fb), pa, pb),
        expected,
        atol=2e-6,
        rtol=2e-5,
    )


def test_tensor_pair_coordinate_and_feature_gradients_and_cutoff():
    from tests.test_mace_h3_pair import _h3_fixture

    core, batch, fa, fb, pa, pb = _h3_fixture("h3l3", architecture_id="hybrid-h3l3t")
    batch.RA.requires_grad_()
    batch.RB.requires_grad_()
    fa = replace(fa, equivariant=fa.equivariant.detach().requires_grad_())
    fb = replace(fb, equivariant=fb.equivariant.detach().requires_grad_())
    energy = core(batch, fa, fb, pa, pb)
    grads = torch.autograd.grad(
        energy.sum(),
        (
            batch.RA,
            batch.RB,
            fa.equivariant,
            fb.equivariant,
            core.directional_projection.weight,
        ),
    )
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads)
    torch.testing.assert_close(
        grads[0].sum(0) + grads[1].sum(0), torch.zeros(3), atol=1e-7, rtol=0
    )
    with torch.no_grad():
        batch.RB.add_(20.0)
    far = core(batch, fa, fb, pa, pb)
    torch.testing.assert_close(far, torch.zeros_like(far), atol=0, rtol=0)
    force = torch.autograd.grad(far.sum(), batch.RB)[0]
    torch.testing.assert_close(force, torch.zeros_like(force), atol=0, rtol=0)


def test_tensor_feature_gradcheck_compile_and_empty_edges():
    torch.manual_seed(91)
    a = torch.randn(3, 2, 7, dtype=torch.float64, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    assert torch.autograd.gradcheck(pair_tensor_invariants, (a, b))
    compiled = torch.compile(pair_tensor_invariants, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(a, b), pair_tensor_invariants(a, b))
    assert compiled(a[:0], b[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="matching"):
        pair_tensor_invariants(a, b[:, :1])


def test_tensor_route_empty_pair_graph():
    from tests.test_mace_h3_pair import _h3_fixture

    core, batch, fa, fb, pa, pb = _h3_fixture("h3l3", architecture_id="hybrid-h3l3t")
    batch.e_ABsr_source = torch.empty(0, dtype=torch.long)
    batch.e_ABsr_target = torch.empty(0, dtype=torch.long)
    batch.dimer_ind = torch.empty(0, dtype=torch.long)
    torch.testing.assert_close(
        core(batch, fa, fb, pa, pb), torch.zeros(1, 4), atol=0, rtol=0
    )


def test_tensor_route_rejects_incompatible_irrep_layouts():
    from tests.test_mace_h3_pair import _h3_fixture

    core, batch, fa, fb, pa, pb = _h3_fixture("h3l3", architecture_id="hybrid-h3l3t")
    wrong = replace(fb, feature_schema=fb.feature_schema.replace("3o", "3e"))
    with pytest.raises(ValueError, match="same irrep"):
        core(batch, fa, wrong, pa, pb)


def test_tensor_route_rejects_unnatural_parity_on_both_monomers():
    from tests.test_mace_h3_pair import _h3_fixture

    core, batch, fa, fb, pa, pb = _h3_fixture("h3l3", architecture_id="hybrid-h3l3t")
    wrong_a = replace(fa, feature_schema=fa.feature_schema.replace("3o", "3e"))
    wrong_b = replace(fb, feature_schema=fb.feature_schema.replace("3o", "3e"))
    with pytest.raises(ValueError, match="natural parity"):
        core(batch, wrong_a, wrong_b, pa, pb)


def test_default_pair_factory_constructs_tensor_route(tmp_path):
    from apnet_pt.training.mace_ap3d3_factory import (
        _default_factory_dependencies,
        validate_mace_cli_args,
    )

    from tests.test_mace_ap3d3_cli import _base_cli

    args, _ = _base_cli(tmp_path, "MACE-AP3D3-H3L3T")
    plan = validate_mace_cli_args(args)
    core = _default_factory_dependencies(plan).pair_core_builder(plan)
    assert core.architecture_id == "hybrid-h3l3t"
    assert core.mace_feature_dim == 2560
    assert core.mace_equivariant_dim == 512
    assert core.get_config()["pair_tensor_product"] == "shared-raw-inner-product-v1"
    for component in ("elst", "exch", "indu", "disp"):
        assert (
            getattr(core.ap3_core, f"readout_layer_{component}")[0].in_features == 150
        )


def test_tensor_route_full_v3_external_reconstruction(tmp_path):
    from apnet_pt.mace.model import MACEAP3D3

    from tests.test_mace_checkpoint_v3 import (
        _config,
        _external_loader,
        _external_metadata,
        _factory,
        _initialized_model,
    )

    route = "hybrid-h3l3t"
    model, batch, expected, artifact, digest = _initialized_model(tmp_path, route)
    checkpoint_path = tmp_path / "tensor-v3.pt"
    model.save_checkpoint_v3(
        checkpoint_path,
        config=_config(digest, route),
        external_mace=_external_metadata(digest),
    )
    calls = []
    restored = MACEAP3D3.load_checkpoint_v3(
        checkpoint_path,
        mace_artifact_path=artifact,
        model_factory=_factory,
        backbone_loader=_external_loader(calls),
        semantic_expectations={"architecture": route},
    )
    assert calls == [artifact]
    assert restored.architecture == route
    assert restored.pair_core.get_config()["pair_tensor_product"] == (
        "shared-raw-inner-product-v1"
    )
    torch.testing.assert_close(restored(batch), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="architecture"):
        MACEAP3D3.load_checkpoint_v3(
            checkpoint_path,
            mace_artifact_path=artifact,
            model_factory=_factory,
            backbone_loader=_external_loader([]),
            semantic_expectations={"architecture": "hybrid-h3l3"},
        )
