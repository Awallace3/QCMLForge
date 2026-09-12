"""H3: MACE equivariant features contracted into the AP3 directional slot.

H2 bypasses the AP3 intramonomer stack *and* zeroes the directional slot, and
``all-scalars+norms`` has already reduced MACE's equivariant channels to norms,
so H2 carries no atomic anisotropy at all. H3 restores it -- ``h3`` from the
l=2 block, ``h3l1`` from l=1 -- leaving the bypass as the only difference from
H2 and the anisotropy source as the only difference between the two H3 arms.

The contraction is only correct if it pairs MACE's equivariant components with
spherical harmonics of the interatomic axis *in MACE's own permuted frame*. A
mismatch there yields a model that is not rotationally invariant but trains
perfectly well, so both halves are asserted here: the frame constant against
MACE itself, and the contraction algebra against an explicit Wigner rotation.
"""

import pytest
import torch

from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_MPNN
from apnet_pt.mace.encoder import _e3nn_o3
from apnet_pt.mace.pair import (
    CANONICAL_AP3D3_DIMENSIONS,
    DIRECTIONAL_DEGREES,
    MACE_E3NN_AXIS_PERMUTATION,
    MACEPairResidualCore,
)
from apnet_pt.mace.schema import MACEAtomicFeatures
from tests.test_mace_h1_pair import _batch, _properties


TEST_IRREPS = "4x0e+4x1o+4x2e"
TEST_CHANNELS = 4
DIRECTIONAL_WIDTH = (
    CANONICAL_AP3D3_DIMENSIONS["n_message"] * CANONICAL_AP3D3_DIMENSIONS["n_embed"]
)


def _equivariant_features(numbers, invariant, equivariant):
    natom = numbers.numel()
    batch = torch.zeros(natom, dtype=torch.long, device=numbers.device)
    return MACEAtomicFeatures(
        invariant=invariant,
        equivariant=equivariant,
        batch=batch,
        atomic_numbers=numbers,
        total_charge=invariant.new_zeros(1),
        total_spin=invariant.new_ones(1),
        feature_schema=(
            "polar-1-s:mace=0.3.16:mode=all-scalars+norms:adapter=stub:"
            f"inv={invariant.shape[1]}:equiv={equivariant.shape[1]}:layers=1:"
            f"private=stub:irreps={TEST_IRREPS}"
        ),
    )


def _h3_fixture(pair_mode="h3", *, seed=31, feature_dim=16):
    torch.manual_seed(seed)
    batch = _batch()
    width = sum(
        TEST_CHANNELS * (2 * degree + 1) for degree in (0, 1, 2)
    )
    features = []
    for numbers in (batch.ZA, batch.ZB):
        features.append(
            _equivariant_features(
                numbers,
                torch.randn(numbers.numel(), feature_dim),
                torch.randn(numbers.numel(), width),
            )
        )
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None,
        use_precomputed_classical=True,
    )
    core = MACEPairResidualCore(
        ap3,
        mace_feature_dim=feature_dim,
        pair_mode=pair_mode,
        feature_mode="all-scalars+norms",
        mace_equivariant_dim=TEST_CHANNELS,
    )
    props_a = _properties(batch.ZA)
    props_b = _properties(batch.ZB, 0.1)
    return core, batch, features[0], features[1], props_a, props_b


def test_frame_permutation_matches_mace():
    """The frame constant is only safe while MACE agrees with it."""

    extensions = pytest.importorskip("mace.modules.extensions")
    probe = torch.tensor([[0.0, 1.0, 2.0]])
    permuted = extensions._permute_to_e3nn_convention(probe)
    expected = probe[:, list(MACE_E3NN_AXIS_PERMUTATION)]
    assert torch.equal(permuted, expected)


@pytest.mark.parametrize("pair_mode,degree", [("h3", 2), ("h3l1", 1)])
def test_h3_registers_its_directional_degree(pair_mode, degree):
    core, *_ = _h3_fixture(pair_mode)
    assert DIRECTIONAL_DEGREES[pair_mode] == degree
    assert core.directional_degree == degree
    assert core.bypass_intra_updates
    assert core.directional_projection.bias is None
    assert core.directional_projection.out_features == DIRECTIONAL_WIDTH
    assert core.get_config()["directional_degree"] == degree


@pytest.mark.parametrize("pair_mode", ["h3", "h3l1"])
def test_h3_pair_width_matches_h1_and_h2(pair_mode):
    core, batch, features_a, features_b, props_a, props_b = _h3_fixture(pair_mode)
    core(batch, features_a, features_b, props_a, props_b)
    expected = (
        2 * (core.ap3_core.n_message + 1) * core.ap3_core.n_embed
        + 6
        + core.ap3_core.n_rbf
        + 2 * core.ap3_core.n_message * core.ap3_core.n_embed
    )
    assert core.last_h_ab.shape[1] == expected == 126
    assert core.last_h_ba.shape[1] == expected


@pytest.mark.parametrize("pair_mode,degree", [("h3", 2), ("h3l1", 1)])
def test_h3_residual_is_rotationally_invariant(pair_mode, degree):
    """Rotate the dimer and the features together; the energy must not move.

    The features are rotated by the Wigner matrix of the *permuted-frame*
    rotation, which is exactly how real MACE features respond, so this checks
    the contraction algebra without assuming anything about how the block was
    produced.
    """

    o3 = _e3nn_o3()
    core, batch, features_a, features_b, props_a, props_b = _h3_fixture(pair_mode)
    reference = core(batch, features_a, features_b, props_a, props_b)

    torch.manual_seed(7)
    rotation = o3.rand_matrix().to(batch.RA.dtype)
    permutation = torch.eye(3, dtype=rotation.dtype)[
        list(MACE_E3NN_AXIS_PERMUTATION)
    ]
    # perm(R r) = (P R P^T) perm(r): the harmonics see this rotation, so the
    # equivariant block has to be rotated by the same one.
    permuted_rotation = permutation @ rotation @ permutation.T

    rotated_batch = batch
    rotated_batch.RA = batch.RA @ rotation.T
    rotated_batch.RB = batch.RB @ rotation.T

    wigner = {
        block_degree: o3.Irrep(
            block_degree, (-1) ** block_degree
        ).D_from_matrix(permuted_rotation)
        for block_degree in (0, 1, 2)
    }
    rotated = []
    for features in (features_a, features_b):
        blocks = []
        for block_degree in (0, 1, 2):
            block = features.equivariant_degree(block_degree)
            blocks.append(
                torch.einsum("acm,nm->acn", block, wigner[block_degree]).reshape(
                    block.shape[0], -1
                )
            )
        rotated.append(
            _equivariant_features(
                features.atomic_numbers,
                features.invariant,
                torch.cat(blocks, dim=1),
            )
        )
    rotated_residual = core(
        rotated_batch, rotated[0], rotated[1], props_a, props_b
    )

    # Scale the tolerance by the residual magnitude rather than fixing it:
    # a fixed atol silently tightens or loosens as the batch or the head
    # changes scale.
    scale = float(reference.detach().abs().max().clamp_min(1.0))
    deviation = float((rotated_residual - reference).detach().abs().max())
    assert deviation <= 2.0e-5 * scale, f"max deviation {deviation:.3e}"


@pytest.mark.parametrize("pair_mode", ["h3", "h3l1"])
def test_h3_actually_uses_the_equivariant_block(pair_mode):
    """Zeroing the consumed degree must change the answer; H2 would not."""

    core, batch, features_a, features_b, props_a, props_b = _h3_fixture(pair_mode)
    reference = core(batch, features_a, features_b, props_a, props_b)

    start, stop, _ = __import__(
        "apnet_pt.mace.schema", fromlist=["irreps_degree_slice"]
    ).irreps_degree_slice(TEST_IRREPS, core.directional_degree)
    zeroed = []
    for features in (features_a, features_b):
        equivariant = features.equivariant.clone()
        equivariant[:, start:stop] = 0.0
        zeroed.append(
            _equivariant_features(
                features.atomic_numbers, features.invariant, equivariant
            )
        )
    ablated = core(batch, zeroed[0], zeroed[1], props_a, props_b)
    assert not torch.allclose(ablated, reference, atol=1.0e-8)


def test_h3_rejects_features_without_an_irrep_layout():
    core, batch, features_a, features_b, props_a, props_b = _h3_fixture()
    stripped = MACEAtomicFeatures(
        invariant=features_a.invariant,
        equivariant=features_a.equivariant,
        batch=features_a.batch,
        atomic_numbers=features_a.atomic_numbers,
        total_charge=features_a.total_charge,
        total_spin=features_a.total_spin,
        feature_schema=(
            "polar-1-s:mace=0.3.16:mode=all-scalars+norms:adapter=stub:"
            f"inv={features_a.invariant.shape[1]}:equiv=0:layers=1:public"
        ),
    )
    with pytest.raises(ValueError, match="no equivariant irrep layout"):
        core(batch, stripped, features_b, props_a, props_b)


def test_h3_rejects_a_channel_count_mismatch():
    with pytest.raises(ValueError, match="channels, expected"):
        core, batch, features_a, features_b, props_a, props_b = _h3_fixture()
        core.mace_equivariant_dim = TEST_CHANNELS + 1
        core(batch, features_a, features_b, props_a, props_b)


def test_non_directional_modes_reject_an_equivariant_dim():
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None, use_precomputed_classical=True
    )
    with pytest.raises(ValueError, match="takes no mace_equivariant_dim"):
        MACEPairResidualCore(
            ap3, mace_feature_dim=16, pair_mode="h2", mace_equivariant_dim=4
        )


def test_h3_requires_an_equivariant_dim():
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None, use_precomputed_classical=True
    )
    with pytest.raises(ValueError, match="requires a positive mace_equivariant_dim"):
        MACEPairResidualCore(ap3, mace_feature_dim=16, pair_mode="h3")


def test_injected_directional_requires_the_bypass():
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None, use_precomputed_classical=True
    )
    batch = _batch()
    stub = torch.zeros(batch.e_ABsr_source.numel(), DIRECTIONAL_WIDTH)
    with pytest.raises(ValueError, match="requires bypass_intra_updates"):
        ap3(
            batch,
            initial_atom_states=(
                torch.zeros(batch.ZA.numel(), CANONICAL_AP3D3_DIMENSIONS["n_embed"]),
                torch.zeros(batch.ZB.numel(), CANONICAL_AP3D3_DIMENSIONS["n_embed"]),
            ),
            atomic_properties=(_properties(batch.ZA), _properties(batch.ZB)),
            residual_only=True,
            injected_pair_directional=(stub, stub),
        )
