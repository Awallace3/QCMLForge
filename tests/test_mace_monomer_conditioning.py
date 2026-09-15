"""H3L3Q: monomer charge and spin conditioning on the AP3 pair readout.

The AP3 pair feature already carries the *predicted* per-atom monopole of both
edge atoms, but nothing that says what the monomer those atoms belong to
actually is -- a -2 anion and a neutral can present the same local charge. The
PLA15 control measured what that costs an MLIP: MACE-POLAR-1-S scores 3.229
MAPE with formal fragment charges supplied and 10.702 with them withheld, and
every formal-charge-0 fragment moved by exactly 0.0000 eV.

``hybrid-h3l3q`` is ``hybrid-h3l3`` plus six per-edge scalars and nothing else,
so the two are a one-lever comparison. What is asserted here is that the lever
is real (the readout is wider and the charge reaches it unmodified), that it is
symmetric (hAB and hBA stay mirrors, so the residual is swap-symmetric), that
it cannot be switched on by accident or off by accident, and that turning it
off leaves every pre-existing route byte-for-byte where it was.
"""

from types import SimpleNamespace

import pytest
import torch

from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_MPNN
from apnet_pt.mace.pair import (
    MONOMER_CONDITIONING_ARCHITECTURES,
    MONOMER_CONDITIONING_SCALARS,
    MONOMER_CONDITIONING_WIDTH,
    MACEPairResidualCore,
)
from tests.test_mace_h3_pair import _h3_fixture

CONTROL_PAIR_WIDTH = 126


def _fixture(*, conditioned, charge_a=0.0, charge_b=0.0, spin_a=1.0, spin_b=1.0):
    """An H3L3 fixture, optionally with the conditioning block enabled.

    ``_h3_fixture`` builds the single-dimer batch every H3 test uses; it
    predates the monomer metadata, so the two indices and the two spins the
    conditioning block reads are attached here rather than there.
    """

    core, batch, features_a, features_b, props_a, props_b = _h3_fixture(
        "h3l3", architecture_id=("hybrid-h3l3q" if conditioned else None)
    )
    batch.molecule_ind_A = torch.zeros(batch.ZA.numel(), dtype=torch.long)
    batch.molecule_ind_B = torch.zeros(batch.ZB.numel(), dtype=torch.long)
    batch.total_charge_A = torch.tensor([charge_a])
    batch.total_charge_B = torch.tensor([charge_b])
    batch.total_spin_A = torch.tensor([spin_a])
    batch.total_spin_B = torch.tensor([spin_b])
    return core, batch, features_a, features_b, props_a, props_b


def test_route_table_names_the_conditioned_architecture():
    assert MONOMER_CONDITIONING_ARCHITECTURES == frozenset({"hybrid-h3l3q"})
    assert MONOMER_CONDITIONING_WIDTH == 2 * len(MONOMER_CONDITIONING_SCALARS) == 6


def test_conditioning_widens_the_pair_readout_and_control_is_unchanged():
    core, *inputs = _fixture(conditioned=True)
    core(*inputs)
    assert core.last_h_ab.shape[1] == CONTROL_PAIR_WIDTH + MONOMER_CONDITIONING_WIDTH
    assert core.last_h_ba.shape[1] == core.last_h_ab.shape[1]
    # All four component heads have to widen together, not just electrostatics:
    # the charge state is as much an induction and exchange fact as a
    # multipole one, and a single narrow head would fail only at runtime.
    for component in ("elst", "exch", "indu", "disp"):
        head = getattr(core.ap3_core, f"readout_layer_{component}")
        assert head[0].in_features == core.last_h_ab.shape[1]

    control, *control_inputs = _fixture(conditioned=False)
    control(*control_inputs)
    assert control.last_h_ab.shape[1] == CONTROL_PAIR_WIDTH


def test_the_appended_block_is_the_monomer_charge_and_spin():
    core, batch, *rest = _fixture(
        conditioned=True, charge_a=-2.0, charge_b=1.0, spin_a=3.0, spin_b=1.0
    )
    core(batch, *rest)
    natom_a = float(batch.ZA.numel())
    natom_b = float(batch.ZB.numel())
    expected_a = torch.tensor([-2.0, -2.0 / natom_a, 2.0])
    expected_b = torch.tensor([1.0, 1.0 / natom_b, 0.0])

    block_ab = core.last_h_ab[:, -MONOMER_CONDITIONING_WIDTH:]
    block_ba = core.last_h_ba[:, -MONOMER_CONDITIONING_WIDTH:]
    for edge in range(block_ab.shape[0]):
        assert torch.allclose(block_ab[edge, :3], expected_a)
        assert torch.allclose(block_ab[edge, 3:], expected_b)
    # hBA is the B-then-A view of the same block: the mirror AP3 applies to
    # every other pair feature, so the residual stays swap-symmetric.
    assert torch.allclose(block_ba, block_ab[:, [3, 4, 5, 0, 1, 2]])


def test_a_closed_shell_monomer_contributes_an_exact_zero_spin_channel():
    """Spin enters as ``multiplicity - 1``.

    Every dimer in the current training data is a pair of singlets, so this
    channel has to be identically zero today -- a constant would merely
    duplicate the readout bias -- while still carrying signal the moment
    open-shell records appear, with no second architecture.
    """

    core, batch, *rest = _fixture(conditioned=True, charge_a=1.0, charge_b=-1.0)
    core(batch, *rest)
    block = core.last_h_ab[:, -MONOMER_CONDITIONING_WIDTH:]
    assert torch.equal(block[:, 2], torch.zeros_like(block[:, 2]))
    assert torch.equal(block[:, 5], torch.zeros_like(block[:, 5]))


def test_charge_moves_the_conditioned_residual_and_not_the_control():
    for conditioned, expected_move in ((True, True), (False, False)):
        torch.manual_seed(7)
        core, *inputs = _fixture(conditioned=conditioned, charge_a=0.0)
        neutral = core(*inputs)[0].clone()
        core, *inputs = _fixture(conditioned=conditioned, charge_a=-1.0)
        # The property bundle is a stub, so the monomer charge reaches this
        # readout through the conditioning block or not at all.
        anion = core(*inputs)[0]
        assert bool(not torch.allclose(neutral, anion)) is expected_move


def test_missing_charge_or_spin_is_refused_rather_than_defaulted():
    for field in ("total_charge_A", "total_charge_B", "total_spin_A", "total_spin_B"):
        core, batch, *rest = _fixture(conditioned=True)
        setattr(batch, field, None)
        with pytest.raises(ValueError, match=field):
            core(batch, *rest)


def test_per_monomer_scalars_track_each_dimer_in_a_mixed_batch():
    core, *_ = _fixture(conditioned=True)
    batch = SimpleNamespace(
        RA=torch.zeros(1, 3),
        total_charge_A=torch.tensor([-2.0, 0.0, 1.0]),
        total_charge_B=torch.tensor([0.0, -1.0, 0.0]),
        total_spin_A=torch.tensor([1.0, 2.0, 1.0]),
        total_spin_B=torch.tensor([1.0, 1.0, 1.0]),
        molecule_ind_A=torch.tensor([0, 0, 1, 1, 1, 2]),
        molecule_ind_B=torch.tensor([0, 1, 1, 2, 2, 2]),
        dimer_ind=torch.tensor([0, 1, 1, 2]),
    )
    scalars_ab, scalars_ba = core._pair_monomer_scalars(batch)
    expected = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, -0.5, 0.0],
            [0.0, 0.0, 1.0, -1.0, -0.5, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(scalars_ab, expected)
    assert torch.allclose(scalars_ba, expected[:, [3, 4, 5, 0, 1, 2]])


def test_conditioning_is_a_property_of_the_route_not_a_free_flag():
    ap3 = APNet3D3_AtomType_MPNN(dimer_prop_model=None, use_precomputed_classical=True)
    with pytest.raises(ValueError, match="monomer_conditioning"):
        MACEPairResidualCore(
            ap3,
            mace_feature_dim=16,
            pair_mode="h3l3",
            feature_mode="all-scalars+norms",
            mace_equivariant_dim=4,
            monomer_conditioning=True,
        )
    with pytest.raises(ValueError, match="monomer_conditioning"):
        MACEPairResidualCore(
            ap3,
            mace_feature_dim=16,
            pair_mode="h3l3",
            feature_mode="all-scalars+norms",
            architecture_id="hybrid-h3l3q",
            mace_equivariant_dim=4,
            monomer_conditioning=False,
        )


def test_the_config_key_appears_only_when_the_block_does():
    control, *_ = _fixture(conditioned=False)
    conditioned, *_ = _fixture(conditioned=True)
    # Absent, not False: ``set_extra_state`` compares configs for equality, so
    # a key present on every route would fail every checkpoint written before
    # the route existed.
    assert "monomer_conditioning" not in control.get_config()
    assert conditioned.get_config()["monomer_conditioning"] is True
    assert control.get_config()["architecture_id"] == "MACE-AP3D3-H3L3"
    assert conditioned.get_config()["architecture_id"] == "hybrid-h3l3q"

    control.set_extra_state(control.get_config())
    with pytest.raises(RuntimeError, match="does not match state_dict"):
        control.set_extra_state(conditioned.get_config())
    with pytest.raises(RuntimeError, match="does not match state_dict"):
        conditioned.set_extra_state(control.get_config())
