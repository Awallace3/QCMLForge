"""End-to-end wiring for the `mastiff-lm` arm: head, dimer, optimizer.

`tests/test_mastiff_lm_exchange.py` pins the physics of the angular factor in
isolation -- the harmonics, the frame, and `cliff_exchange` called directly.
What is left, and what breaks silently rather than loudly, is the plumbing
between them: the head packs the coefficients, the frame, and its validity
flags into trailing columns of one parameter tensor, `DimerProp.forward`
unpacks that packing, and `_optimizer_parameter_groups` has to account for
every parameter the head owns or refuse to build an optimizer at all.

Two invariants carry the arm's credibility and are asserted here directly:

* enabling the mode is a **numerical no-op at initialization**, so any number
  the campaign reports is attributable to training rather than to the
  reparameterization; and
* the learned frame lands in the anisotropy learning-rate group, not in
  `base`, which is the same failure `atom_model_lr` exists to prevent.

The two new rate knobs (`exch_param_lr` for MASTIFF's `A_iso`, and
`valence_width_lr` for the overlap exponent) are tested here too, because they
are the other half of the same campaign: MASTIFF refits `A_iso` and `B`
alongside `a_lm`, and the closed null on the pinned-basis arm trained neither.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apnet_pt import local_frame  # noqa: E402
from apnet_pt.AtomPairwiseModels import mtp_mtp  # noqa: E402

HEAD_KWARGS = dict(n_message=1, n_neuron=8, n_embed=4)
FRAME_WIDTH = mtp_mtp._CLIFF_MASTIFF_FRAME_WIDTH
N_PARAMS = len(mtp_mtp.CLIFF_CLASSICAL_PARAMETER_NAMES)


def _harness(nested, **overrides):
    return mtp_mtp.CliffClassicalOverlapModel(
        atom_model=copy.deepcopy(nested),
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        **{**HEAD_KWARGS, **overrides},
    )


def _dimer(head):
    return mtp_mtp.DimerProp(
        ATParam=head,
        dimer_eval="cliff_classical_overlap",
        freeze_atom_model=True,
    )


def _unfreeze(harness):
    trunk = list(harness.model.atom_model.parameters())
    for parameter in trunk:
        parameter.requires_grad_(True)
    assert trunk, "fixture has no trunk parameters to unfreeze"
    return trunk


def _nudge_coefficients(head, value=0.05):
    """Move the zero-init angular readouts off zero.

    At `xi = 0` the frame is genuinely irrelevant to the energy, so any test
    about the frame mattering has to leave the initialization first.
    """
    for stack in head.anisotropy_readout_layers:
        for readout in stack:
            output_layer = [
                module
                for module in readout.modules()
                if isinstance(module, torch.nn.Linear)
            ][-1]
            torch.nn.init.constant_(output_layer.bias, value)


# --------------------------------------------------------------------------
# What the head emits
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parity,n_coefficients",
    [
        ("even", len(local_frame.PARITY_EVEN_LABELS)),
        ("all", len(local_frame.RACAH_L1L2_LABELS)),
    ],
)
def test_head_packs_coefficients_then_frame_then_flags(
    nested_hfvr_vw_model, atomic_batch, parity, n_coefficients
):
    """The trailing block is `[a_lm ...] + [9 rotation entries] + [2 flags]`.

    `DimerProp.forward` recovers the coefficient count by subtracting the
    fixed frame width rather than reading a stored field, so a checkpoint
    trained under either parity unpacks correctly.  That only holds if the
    width really is fixed and the coefficients really do come first.
    """
    head = _harness(nested_hfvr_vw_model).model
    head.enable_multipole_anisotropy("mastiff-lm", parity=parity)
    assert len(head.anisotropy_channels) == n_coefficients

    parameters = head(atomic_batch)[-1]
    assert parameters.size(1) == N_PARAMS + n_coefficients + FRAME_WIDTH

    frame = parameters[:, N_PARAMS + n_coefficients:][:, :9].reshape(-1, 3, 3)
    identity = torch.eye(3, dtype=frame.dtype).expand_as(frame)
    assert torch.allclose(frame @ frame.transpose(-1, -2), identity, atol=1e-5)
    assert torch.allclose(
        torch.det(frame),
        torch.ones(frame.size(0), dtype=frame.dtype),
        atol=1e-5,
    )
    # The flags describe the *learned* axes, not the geometry.  Water gives
    # every atom neighbours, so the polar axis exists; the azimuth need not,
    # and at random initialization the frame MLP has no reason to emit two
    # independent directions.  What must hold is that the flags are boolean
    # and nested -- no azimuth without a polar axis to measure it from.
    z_valid, x_valid = parameters[:, -2], parameters[:, -1]
    assert torch.equal(z_valid, torch.ones_like(z_valid))
    assert set(x_valid.tolist()) <= {0.0, 1.0}
    assert torch.all(x_valid <= z_valid)


def test_head_zero_initializes_every_coefficient(
    nested_hfvr_vw_model, atomic_batch
):
    """`xi = 0` at init is the whole no-op argument; assert it exactly."""
    head = _harness(nested_hfvr_vw_model).model
    head.enable_multipole_anisotropy("mastiff-lm")
    n_coefficients = len(head.anisotropy_channels)
    coefficients = head(atomic_batch)[-1][
        :, N_PARAMS:N_PARAMS + n_coefficients
    ]
    assert torch.equal(coefficients, torch.zeros_like(coefficients))


def test_enabling_mastiff_twice_under_a_different_parity_fails_closed(
    nested_hfvr_vw_model,
):
    """Five readouts cannot serve eight channels.

    Silently reusing them would reinterpret `a_21c` as `a_21s` on reload.
    """
    head = _harness(nested_hfvr_vw_model).model
    head.enable_multipole_anisotropy("mastiff-lm", parity="even")
    with pytest.raises(ValueError, match="needs 8 readouts"):
        head.enable_multipole_anisotropy("mastiff-lm", parity="all")


def test_parity_and_frame_cutoff_survive_a_config_round_trip(
    nested_hfvr_vw_model,
):
    """`get_config` is what a checkpoint replays.

    A key missing there comes back with the setting switched off, no error.
    """
    head = _harness(nested_hfvr_vw_model).model
    head.enable_multipole_anisotropy(
        "mastiff-lm", parity="all", frame_r_cut=4.25
    )
    config = head.get_config()
    assert config["anisotropy_mode"] == "mastiff-lm"
    assert config["anisotropy_parity"] == "all"
    assert config["anisotropy_frame_r_cut"] == 4.25


# --------------------------------------------------------------------------
# Through the dimer
# --------------------------------------------------------------------------


def test_mastiff_dimer_energy_is_unchanged_at_initialization(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The arm's credibility rests on this.

    Enabling the mode changes nothing until something is trained.
    Bit-identical, not merely close.
    """
    torch.manual_seed(0)
    isotropic = _harness(nested_hfvr_vw_model).model
    anisotropic = copy.deepcopy(isotropic)
    anisotropic.enable_multipole_anisotropy("mastiff-lm")

    reference, _, _ = _dimer(isotropic)(synthetic_dimer_batch)
    energy, _, _ = _dimer(anisotropic)(synthetic_dimer_batch)
    assert torch.equal(energy, reference)


def test_mastiff_dimer_energy_moves_once_the_coefficients_are_nonzero(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The complement of the test above: the packed columns are consumed.

    A packing bug that dropped them would pass the no-op test forever,
    because zero is what a dropped column looks like.
    """
    torch.manual_seed(0)
    head = _harness(nested_hfvr_vw_model).model
    head.enable_multipole_anisotropy("mastiff-lm")
    reference, _, _ = _dimer(copy.deepcopy(head))(synthetic_dimer_batch)

    _nudge_coefficients(head)
    energy, _, _ = _dimer(head)(synthetic_dimer_batch)
    assert torch.isfinite(energy).all()
    assert (energy - reference).abs().max() > 1e-8


def test_mastiff_dimer_gradients_reach_the_frame(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """A frame with no gradient path is a random rotation the run cannot fix."""
    torch.manual_seed(0)
    head = _harness(nested_hfvr_vw_model).model
    head.enable_multipole_anisotropy("mastiff-lm")
    _nudge_coefficients(head)

    energy, _, _ = _dimer(head)(synthetic_dimer_batch)
    energy.sum().backward()
    gradients = [
        parameter.grad
        for parameter in head.anisotropy_frame.parameters()
        if parameter.grad is not None
    ]
    assert gradients, "the learned frame received no gradient at all"
    assert max(g.abs().max().item() for g in gradients) > 0.0


# --------------------------------------------------------------------------
# Optimizer groups
# --------------------------------------------------------------------------


def test_mastiff_frame_lands_in_the_anisotropy_group(nested_hfvr_vw_model):
    """Not in `base`.

    This is the `atom_model_lr` failure in another costume: a second parameter
    set swept into the head's rate without saying so.
    """
    harness = _harness(nested_hfvr_vw_model)
    harness.model.enable_multipole_anisotropy("mastiff-lm")
    groups = harness._optimizer_parameter_groups(
        5e-4, 2.5e-5, anisotropy_lr=1e-6
    )
    assert [group["group_name"] for group in groups] == [
        "base",
        "thole",
        "anisotropy",
    ]
    frame_ids = {id(p) for p in harness.model.anisotropy_frame.parameters()}
    assert frame_ids
    assert frame_ids <= {id(p) for p in groups[-1]["params"]}
    assert not frame_ids & {id(p) for p in groups[0]["params"]}


def test_exchange_prefactor_gets_its_own_group(nested_hfvr_vw_model):
    """`A_iso` at its own rate is the point of the unfreeze arm.

    It has to be identifiable against a zero-init angular head, which needs
    the two to move at different speeds.
    """
    harness = _harness(nested_hfvr_vw_model)
    groups = harness._optimizer_parameter_groups(
        5e-4, 2.5e-5, exch_param_lr=1e-6
    )
    assert [group["group_name"] for group in groups] == [
        "base",
        "thole",
        "exchange_param",
    ]
    assert groups[-1]["lr"] == 1e-6
    head = harness.model
    expected = {
        id(p)
        for layer in (
            head.guess_layer[mtp_mtp.CLIFF_CLASSICAL_EXCH_INDEX],
            head.param_readout_layers[mtp_mtp.CLIFF_CLASSICAL_EXCH_INDEX],
        )
        for p in layer.parameters()
    }
    assert {id(p) for p in groups[-1]["params"]} == expected
    assert not expected & {id(p) for p in groups[0]["params"]}


def test_valence_widths_split_out_of_the_trunk(nested_hfvr_vw_model):
    """The valence width IS the overlap exponent.

    The arm needs to move it while the 1.89M-parameter message-passing trunk
    underneath stays put.
    """
    harness = _harness(nested_hfvr_vw_model)
    _unfreeze(harness)
    groups = harness._optimizer_parameter_groups(
        5e-4, 2.5e-5, None, 0.0, valence_width_lr=1e-6
    )
    assert [group["group_name"] for group in groups] == [
        "base",
        "thole",
        "valence_width",
        "atom_model",
    ]
    widths, trunk = groups[2], groups[3]
    assert widths["lr"] == 1e-6
    assert trunk["lr"] == 0.0
    nested_ids = {
        id(p)
        for p in mtp_mtp._innermost_atom_mpnn(
            harness.model.atom_model
        ).parameters()
    }
    assert {id(p) for p in trunk["params"]} == nested_ids
    assert not {id(p) for p in widths["params"]} & nested_ids
    assert widths["params"], "no AtomTypeParamNN readout parameters were found"


def test_valence_width_lr_on_a_frozen_trunk_fails_closed(nested_hfvr_vw_model):
    """Silently emitting an empty group would report a rate that trains
    nothing, indistinguishable in the logs from one that works."""
    harness = _harness(nested_hfvr_vw_model)
    with pytest.raises(ValueError, match="unfreeze_atom_model"):
        harness._optimizer_parameter_groups(
            5e-4, 2.5e-5, None, 0.0, valence_width_lr=1e-6
        )


def test_the_three_new_knobs_compose(nested_hfvr_vw_model):
    """The campaign runs them together.

    The coverage assertion inside `_optimizer_parameter_groups` is what makes
    that safe to do.
    """
    harness = _harness(nested_hfvr_vw_model)
    harness.model.enable_multipole_anisotropy("mastiff-lm")
    _unfreeze(harness)
    groups = harness._optimizer_parameter_groups(
        5e-4,
        2.5e-5,
        None,
        0.0,
        anisotropy_lr=1e-6,
        exch_param_lr=1e-7,
        valence_width_lr=1e-8,
    )
    assert [group["group_name"] for group in groups] == [
        "base",
        "thole",
        "exchange_param",
        "valence_width",
        "atom_model",
        "anisotropy",
    ]
    seen = [id(p) for group in groups for p in group["params"]]
    assert len(seen) == len(set(seen))
    trainable = {id(p) for p in harness.model.parameters() if p.requires_grad}
    assert set(seen) == trainable


def test_omitting_every_new_knob_keeps_the_legacy_iterator(
    nested_hfvr_vw_model,
):
    """Historical runs must reproduce byte for byte.

    The short-circuit is what guarantees that, so it has to survive the two
    added parameters.
    """
    harness = _harness(nested_hfvr_vw_model)
    assert not isinstance(
        harness._optimizer_parameter_groups(5e-4, None), list
    )
