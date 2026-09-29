"""Public feature, exchange-energy and checkpoint contracts for MACE eq-l3."""

import io
import math
from dataclasses import replace

import pytest
import torch

from apnet_pt.mace.schema import MACEAtomicFeatures


def _fixture(**overrides):
    from apnet_pt.mace import MACEMASTIFFExchange

    torch.manual_seed(17)
    schema = "test-mace:irreps=4x0e+4x1o+4x2e+4x3o"

    def features():
        return MACEAtomicFeatures(
            invariant=torch.randn(2, 8, dtype=torch.float64),
            equivariant=torch.randn(2, 64, dtype=torch.float64),
            atomic_numbers=torch.tensor([1, 8]),
            batch=torch.zeros(2, dtype=torch.long),
            total_charge=torch.zeros(1, dtype=torch.float64),
            total_spin=torch.ones(1, dtype=torch.float64),
            feature_schema=schema,
        )

    config = {
        "mace_feature_dim": 8,
        "feature_schema": schema,
        "mace_checkpoint_sha256": "a" * 64,
        "element_baselines": {1: (2.0, 1.0), 8: (3.0, 1.0)},
        "readout_init_scale": 0.0,
    }
    config.update(overrides)
    model = MACEMASTIFFExchange(**config).double()
    ra = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    rb = ra + torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    edges = torch.tensor([[0, 1], [0, 1]])
    return model, features(), features(), ra, rb, edges


def test_zero_readout_matches_element_baseline_and_positive_overlap():
    model, fa, fb, ra, rb, edges = _fixture()
    # Two length-2 pairs: amplitudes 2*2 and 3*3, x=2 for both.
    expected = torch.tensor([13 * (13 / 3) * math.exp(-2)], dtype=torch.float64)
    torch.testing.assert_close(model(fa, fb, ra, rb, edges), expected)
    a, b, coefficients = model.atom_quantities(fa)
    torch.testing.assert_close(a, torch.tensor([2.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(b, torch.ones(2, dtype=torch.float64))
    assert [c.shape[-1] for c in coefficients] == [3, 5, 7]
    assert all(torch.count_nonzero(c) == 0 for c in coefficients)


def test_head_learns_positive_exchange_without_feature_gradients():
    model, fa, fb, ra, rb, edges = _fixture()
    fa = replace(
        fa,
        invariant=fa.invariant.requires_grad_(),
        equivariant=fa.equivariant.requires_grad_(),
    )
    initial = model(fa, fb, ra, rb, edges).detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = initial * 0.8
    for _ in range(12):
        optimizer.zero_grad()
        energy = model(fa, fb, ra, rb, edges)
        (energy - target).square().mean().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        optimizer.step()
    actual = model(fa, fb, ra, rb, edges)
    assert ((actual - target).abs() < (initial - target).abs()).all()
    assert (model.pair_energies(fa, fb, ra, rb, edges) > 0).all()
    assert fa.invariant.grad is None
    assert fa.equivariant.grad is None
    assert all(c.norm() > 0 for c in model.atom_quantities(fa)[2])


@pytest.mark.parametrize("reflection", [False, True])
def test_nonzero_coefficients_are_joint_o3_and_swap_invariant(reflection):
    from apnet_pt.mace.encoder import _e3nn_o3
    from apnet_pt.mace.pair import MACE_E3NN_AXIS_PERMUTATION

    o3 = _e3nn_o3()
    model, fa, fb, ra, rb, edges = _fixture(readout_init_scale=0.2)
    transform = o3.rand_matrix(dtype=torch.float64)
    if reflection:
        transform = -transform
    p = torch.eye(3, dtype=torch.float64)[list(MACE_E3NN_AXIS_PERMUTATION)]
    internal = p @ transform @ p.T

    def rotate(features):
        blocks = [
            (
                features.equivariant_degree(degree)
                @ o3.Irrep(degree, (-1) ** degree).D_from_matrix(internal).T
            ).reshape(features.natom, -1)
            for degree in range(4)
        ]
        return replace(features, equivariant=torch.cat(blocks, -1))

    expected = model(fa, fb, ra, rb, edges)
    torch.testing.assert_close(
        model(
            rotate(fa), rotate(fb), ra @ transform.T + 5, rb @ transform.T + 5, edges
        ),
        expected,
        atol=2e-7,
        rtol=2e-7,
    )
    torch.testing.assert_close(
        model(fb, fa, rb, ra, edges.flip(0)),
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_checkpoint_roundtrip_binds_feature_artifact_and_kernel_choices():
    from apnet_pt.mace import MACEMASTIFFExchange

    model, fa, fb, ra, rb, edges = _fixture(readout_init_scale=0.2)
    expected = model(fa, fb, ra, rb, edges)
    buffer = io.BytesIO()
    torch.save({"config": model.get_config(), "state": model.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = MACEMASTIFFExchange(**saved["config"]).double()
    restored.load_state_dict(saved["state"])
    assert torch.equal(restored(fa, fb, ra, rb, edges), expected)
    for changes in (
        {"rho": 0.3},
        {"mace_checkpoint_sha256": "b" * 64},
        {"feature_schema": fa.feature_schema.replace("test", "wrong")},
        {"element_baselines": {1: (4.0, 1.0), 8: (3.0, 1.0)}},
        {"anisotropy": False},
    ):
        wrong = MACEMASTIFFExchange(**(model.get_config() | changes)).double()
        with pytest.raises(RuntimeError, match="contract mismatch"):
            wrong.load_state_dict(saved["state"])


def test_degree_bounds_empty_pairs_and_zero_equivariants_are_finite():
    model, fa, fb, ra, rb, edges = _fixture(readout_init_scale=0.2)
    huge = replace(fa, equivariant=fa.equivariant * 1e200)
    coefficients = model.atom_quantities(huge)[2]
    assert all(torch.isfinite(c).all() for c in coefficients)
    assert all((c.norm(dim=-1) <= 0.4 + 1e-15).all() for c in coefficients)
    empty = edges[:, :0]
    assert torch.equal(model(fa, fb, ra, rb, empty), torch.zeros(1).double())
    zero = replace(fa, equivariant=torch.zeros_like(fa.equivariant))
    assert all(torch.count_nonzero(c) == 0 for c in model.atom_quantities(zero)[2])
    model(zero, fb, ra, rb, edges).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_isotropic_control_is_orientation_blind_and_has_no_dead_angular_heads():
    model, fa, fb, ra, rb, edges = _fixture(
        anisotropy=False,
        readout_init_scale=0.2,
    )
    changed = replace(fa, equivariant=-fa.equivariant * 5)
    assert torch.equal(model(changed, fb, ra, rb, edges), model(fa, fb, ra, rb, edges))
    assert all(torch.count_nonzero(c) == 0 for c in model.atom_quantities(fa)[2])
    model(fa, fb, ra, rb, edges).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_mixed_dimers_sum_independently_and_reject_cross_dimer_edges():
    model, fa, fb, ra, rb, edges = _fixture(readout_init_scale=0.2)
    single = model(fa, fb, ra, rb, edges)

    def duplicate(f):
        return replace(
            f,
            invariant=f.invariant.repeat(2, 1),
            equivariant=f.equivariant.repeat(2, 1),
            atomic_numbers=f.atomic_numbers.repeat(2),
            batch=torch.tensor([0, 0, 1, 1]),
            total_charge=f.total_charge.repeat(2),
            total_spin=f.total_spin.repeat(2),
        )

    fa2, fb2 = duplicate(fa), duplicate(fb)
    pairs = torch.cat((edges, edges + 2), 1)
    actual = model(fa2, fb2, ra.repeat(2, 1), rb.repeat(2, 1), pairs)
    torch.testing.assert_close(actual, single.repeat(2))
    bad = pairs.clone()
    bad[1, 0] = 2
    with pytest.raises(ValueError, match="dimer"):
        model(fa2, fb2, ra.repeat(2, 1), rb.repeat(2, 1), bad)


@pytest.mark.parametrize("change", ["schema", "width", "parity", "nan", "element"])
def test_invalid_feature_contract_is_rejected(change):
    model, fa, fb, ra, rb, edges = _fixture()
    if change == "schema":
        fa = replace(fa, feature_schema=fa.feature_schema.replace("test", "bad"))
    elif change == "width":
        fa = replace(fa, invariant=fa.invariant[:, :2])
    elif change == "parity":
        with pytest.raises(ValueError, match="natural parity"):
            _fixture(feature_schema=fa.feature_schema.replace("1o", "1e"))
        return
    elif change == "nan":
        # The feature container itself may reject this before the head.
        with pytest.raises(ValueError):
            fa = replace(fa, invariant=fa.invariant * float("nan"))
            model(fa, fb, ra, rb, edges)
        return
    else:
        fa = replace(fa, atomic_numbers=torch.tensor([1, 17]))
    with pytest.raises(ValueError):
        model(fa, fb, ra, rb, edges)


def test_invalid_geometry_and_edge_indices_are_rejected():
    model, fa, fb, ra, rb, edges = _fixture()
    for bad in (edges.float(), edges + 10, edges - 1, edges.flatten()):
        with pytest.raises(ValueError):
            model(fa, fb, ra, rb, bad)
    with pytest.raises(ValueError, match="positive"):
        model(fa, fb, ra, ra, edges)
    with pytest.raises(ValueError, match="finite"):
        model(fa, fb, ra, rb * float("nan"), edges)


def test_nonfinite_learned_parameters_fail_instead_of_emitting_nan():
    model, fa, fb, ra, rb, edges = _fixture(readout_init_scale=0.2)
    # Finite but out-of-range descriptors must not silently yield NaN energies.
    huge = replace(fa, invariant=fa.invariant * 1e200)
    with pytest.raises(ValueError, match="radial"):
        model(huge, fb, ra, rb, edges)


def _float_fixture(**overrides):
    model, fa, fb, ra, rb, edges = _fixture(**overrides)

    def cast(f):
        return replace(
            f,
            invariant=f.invariant.float(),
            equivariant=f.equivariant.float(),
            total_charge=f.total_charge.float(),
            total_spin=f.total_spin.float(),
        )

    return model.float(), cast(fa), cast(fb), ra.float(), rb.float(), edges


def test_finite_float32_pairs_cannot_silently_overflow_dimer_sum():
    model, fa, fb, ra, rb, _ = _float_fixture(
        element_baselines={1: (1.2e19, 0.005), 8: (1.2e19, 0.005)},
    )
    edges = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    assert torch.isfinite(model.pair_energies(fa, fb, ra, rb, edges)).all()
    with pytest.raises(ValueError, match="dimer"):
        model(fa, fb, ra, rb, edges)


def test_float32_optimizer_step_and_dtype_preserving_checkpoint():
    from apnet_pt.mace import MACEMASTIFFExchange

    model, fa, fb, ra, rb, edges = _float_fixture(readout_init_scale=0.1)
    before = model(fa, fb, ra, rb, edges).detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    (model(fa, fb, ra, rb, edges) - before * 0.9).square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    after = model(fa, fb, ra, rb, edges)
    assert torch.isfinite(after).all() and not torch.equal(after, before)
    restored = MACEMASTIFFExchange(**model.get_config()).float()
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored(fa, fb, ra, rb, edges), after)


def test_unequal_monomers_all_pairs_atom_permutations_and_edge_order():
    model, fa, fb, ra, rb, _ = _fixture(readout_init_scale=0.2)
    fb = replace(
        fb,
        invariant=torch.cat((fb.invariant, fb.invariant[:1] * 1.7)),
        equivariant=torch.cat((fb.equivariant, fb.equivariant[:1] * -0.3)),
        atomic_numbers=torch.tensor([1, 8, 1]),
        batch=torch.zeros(3).long(),
    )
    rb = torch.cat((rb, rb[:1] + torch.tensor([0.3, -0.5, 0.1]).double()))
    source, target = torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij")
    edges = torch.stack((source.flatten(), target.flatten()))
    expected = model(fa, fb, ra, rb, edges)
    torch.testing.assert_close(
        model(fb, fa, rb, ra, edges.flip(0)),
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
    pa, pb = torch.tensor([1, 0]), torch.tensor([2, 0, 1])

    def permute(f, order):
        return replace(
            f,
            invariant=f.invariant[order],
            equivariant=f.equivariant[order],
            atomic_numbers=f.atomic_numbers[order],
            batch=f.batch[order],
        )

    remapped = torch.stack((pa.argsort()[edges[0]], pb.argsort()[edges[1]]))
    remapped = remapped[:, torch.tensor([4, 0, 3, 5, 1, 2])]
    actual = model(permute(fa, pa), permute(fb, pb), ra[pa], rb[pb], remapped)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_each_equivariant_degree_independently_changes_exchange(degree):
    from apnet_pt.mace.schema import irreps_degree_slice

    model, fa, fb, ra, rb, edges = _fixture(readout_init_scale=0.2)
    start, stop, _ = irreps_degree_slice(fa.equivariant_irreps, degree)
    selected = torch.zeros_like(fa.equivariant)
    selected[:, start:stop] = fa.equivariant[:, start:stop]
    zero_a = replace(fa, equivariant=torch.zeros_like(fa.equivariant))
    zero_b = replace(fb, equivariant=torch.zeros_like(fb.equivariant))
    baseline = model(zero_a, zero_b, ra, rb, edges)
    actual = model(replace(fa, equivariant=selected), zero_b, ra, rb, edges)
    assert (actual - baseline).abs().max() > 1e-5


@pytest.mark.parametrize("corruption", ["metadata", "version", "missing", "extra"])
def test_strict_checkpoint_rejects_incomplete_or_foreign_state(corruption):
    from apnet_pt.mace import MACEMASTIFFExchange

    model, *_ = _fixture()
    state = model.state_dict()
    if corruption == "metadata":
        del state["_extra_state"]
    elif corruption == "version":
        state["_extra_state"]["architecture"] = "wrong-version"
    elif corruption == "missing":
        del state[next(k for k in state if k.endswith(".weight"))]
    else:
        state["featurizer.backbone.foreign_tensor"] = torch.ones(1)
    restored = MACEMASTIFFExchange(**model.get_config()).double()
    with pytest.raises(RuntimeError):
        restored.load_state_dict(state)
