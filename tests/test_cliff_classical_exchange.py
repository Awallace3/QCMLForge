"""Task-A tests for the shared CLIFF overlap helper and exchange kernel.

Covers:
  * the golden water-dimer overlap / exchange values,
  * property tests for ``mtp_mtp.atomic_overlap_S_ij``,
  * a pre-refactor regression pin for the three induction-overlap call sites,
  * a legacy pin for ``apnet3.APNet3_MPNN.valence_width_exch``,
  * the multiplicative (not geometric-mean) exchange pair rule,
  * sign / magnitude behavior on a real close-contact water dimer.

Task-B tests for the generalized positive-parameter contract:

  * ``mtp_mtp._validate_positive_initialization`` and the byte-for-byte
    preservation of its ``_validate_rackers_initialization`` wrapper's messages,
  * ``CliffExchangeNN`` / ``CliffClassicalNN`` output shape, positivity,
    initialization, wrapped-output preservation, gradients, and ``get_config``.

Task-C tests for the four new ``DimerProp`` modes:

  * ``FULL_EDGE_DIMER_EVAL_MODES`` membership versus the forwards that actually
    read ``e_ABfull_*``, and ``_dimer_index_for_output`` agreement,
  * exclusive physics routing of all five ``CliffClassicalNN`` columns,
  * ``cliff_exch`` running neither electrostatics nor induction and needing no
    polarizability table,
  * one intermolecular distance reduction per forward pass, and the
    Angstrom-to-bohr conversion on the shared-distance path,
  * joint-forward shapes, scatter aggregation, per-mode energy-active gradient
    heads, and post-optimizer-step positivity.

Task-D tests for the training harnesses, loss weighting, and checkpoints:

  * ``CliffExchangeModel`` / ``CliffClassicalModel`` /
    ``CliffClassicalOverlapModel`` contract (fixed model type, dimer mode,
    parameter count and ordering; no ``n_params`` / ``model_type`` /
    ``dimer_eval_type`` in the public constructor),
  * the ``y_ind`` / ``term`` dispatch (``1`` for exchange, ``[0, 1, 2]`` for the
    classical routes) and the width-agnostic MAE report header,
  * CLIFF Eq. (23) component/total loss weighting, including a bitwise
    comparison of the ``component_gamma == 1.0`` default against the
    pre-Task-D loss expression,
  * checkpoint round trips for both new model types and rejection of absent,
    reordered, or foreign ``parameter_names``.

Task-E tests for the ``train_models.py`` dispatch and CLI:

  * the three new ``--train_apnet`` identifiers selecting their harness, dimer
    mode, default initialization tuple, and target columns,
  * the two-stage HFVR/valence-width construction, forwarded checkpoint paths,
    forced ``world_size = 1``, absent legacy checkpoint default, and the
    ``--unfreeze_atom_model`` freeze contract,
  * rejection of scalar broadcasting and wrong-length parameter overrides,
  * ``--component_gamma`` / ``--total_includes_d3`` forwarding (including the
    ``None`` default surviving the ``inspect.signature`` filter), the
    ``--include_total_mse`` reinterpretation, and rejection on every route
    without a total/component split,
  * ``python train_models.py --help`` exiting 0 and advertising all of it.

Unit convention
---------------
``atomic_overlap_S_ij`` takes ``dR_AB`` in **bohr**, matching every in-repo
caller: both ``_rackers_distance_tensors`` and ``distance_tensors`` divide by
``constants.au2ang`` before returning.  ``cliff_exchange`` is the only entry
point that accepts Angstrom coordinates, and it converts internally.
"""

import copy
import inspect
import math
import os
import pathlib
import re
import subprocess
import sys
import types

import numpy as np
import pytest
import qcelemental as qcel
import torch
import torch.nn.functional as F

from apnet_pt import constants, model_io
from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomPairwiseModels import mtp_mtp
from apnet_pt.AtomPairwiseModels.apnet3 import APNet3_MPNN
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    CLIFF_CLASSICAL_ELST_INDEX,
    CLIFF_CLASSICAL_EXCH_INDEX,
    CLIFF_CLASSICAL_IND_OVERLAP_INDEX,
    CLIFF_CLASSICAL_INITIAL_STDS,
    CLIFF_CLASSICAL_INITIAL_VALUES,
    CLIFF_CLASSICAL_PARAMETER_NAMES,
    CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
    CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
    CLIFF_EXCH_INDEX,
    CLIFF_EXCH_INITIAL_STDS,
    CLIFF_EXCH_INITIAL_VALUES,
    CLIFF_EXCH_PARAMETER_NAMES,
    COMBINED_CLIFF_DIMER_EVAL_MODES,
    OVERLAP_WIDTH_FLOOR,
    POSITIVE_PARAMETER_CONTRACTS,
    RACKERS_INITIAL_STDS,
    RACKERS_INITIAL_VALUES,
    RACKERS_PARAMETER_NAMES,
    AM_DimerParam_Model,
    AtomTypeParamNN,
    CliffClassicalModel,
    CliffClassicalNN,
    CliffClassicalOverlapModel,
    CliffExchangeModel,
    CliffExchangeNN,
    RackersTholeDampingModel,
    RackersTholeDampingNN,
    _mae_report_header,
    _rebuild_nested_atom_model,
    _validate_positive_initialization,
    _validate_rackers_initialization,
    atomic_overlap_S_ij,
    cliff_exchange,
    geometric_mean_edge_values,
)
from apnet_pt.util import scatter_sum_compile

import train_models

from .conftest import _make_collate_item
from .test_rackers_thole_damping import _FakeAtomTypeParamModel

# CLIFF Table I / Fig. 5 water-dimer hydrogen bond.
K_EXCH_O2 = 5.8538
K_EXCH_HO = 0.5996
SIGMA_O = 0.39
SIGMA_H = 0.36
R_HBOND_ANG = 1.95

GOLDEN_B_IJ = 2.6688
GOLDEN_S_IJ = 2.30760e-3
GOLDEN_E_EXCH_KCAL = 5.0825

# The four `dimer_eval` modes Task C adds.
CLIFF_DIMER_EVAL_MODES = frozenset(
    {
        "cliff_exch",
        "cliff_classical",
        "cliff_classical_overlap",
        "cliff_classical_d3",
    }
)


def _single_edge():
    return (
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    )


# --------------------------------------------------------------------------
# Golden overlap value test
# --------------------------------------------------------------------------


def test_golden_overlap_value():
    """Pin B_ij = 1/sqrt(sigma_i sigma_j), S_ij, and the h2kcalmol placement."""
    e_src, e_tgt = _single_edge()
    r_bohr = torch.tensor(
        [R_HBOND_ANG / constants.au2ang], dtype=torch.float64
    )

    # B_ij is not returned by the helper, so assert it independently and then
    # confirm the helper's S_ij is the closed form evaluated at that B_ij.
    B_ij = 1.0 / math.sqrt(SIGMA_O * SIGMA_H)
    assert B_ij == pytest.approx(GOLDEN_B_IJ, rel=1e-4)

    S_ij = atomic_overlap_S_ij(
        torch.tensor([SIGMA_O], dtype=torch.float64),
        torch.tensor([SIGMA_H], dtype=torch.float64),
        e_src,
        e_tgt,
        r_bohr,
    )
    assert S_ij.item() == pytest.approx(GOLDEN_S_IJ, rel=1e-4)

    # The golden widths (0.39, 0.36) both sit above OVERLAP_WIDTH_FLOOR, so the
    # default floor is inert here and the value pins the physics, not the clamp.
    assert SIGMA_H > OVERLAP_WIDTH_FLOOR

    E = cliff_exchange(
        RA=None,
        RB=None,
        e_AB_source=e_src,
        e_AB_target=e_tgt,
        valence_widths_A=torch.tensor([SIGMA_O], dtype=torch.float64),
        valence_widths_B=torch.tensor([SIGMA_H], dtype=torch.float64),
        K_exch_A=torch.tensor([K_EXCH_O2], dtype=torch.float64),
        K_exch_B=torch.tensor([K_EXCH_HO], dtype=torch.float64),
        dR_AB=r_bohr,
    )
    assert E.item() == pytest.approx(GOLDEN_E_EXCH_KCAL, rel=1e-3)


def test_shipped_form_is_not_the_literal_cliff_eq_10():
    """The literal 1/(sigma_i sigma_j) reading is not what ships.

    ``apnet3.valence_width_exch`` uses that literal form; it underpredicts a
    water-dimer hydrogen-bond overlap by more than six orders of magnitude.
    """
    e_src, e_tgt = _single_edge()
    r_bohr = torch.tensor(
        [R_HBOND_ANG / constants.au2ang], dtype=torch.float64
    )
    S_shipped = atomic_overlap_S_ij(
        torch.tensor([SIGMA_O], dtype=torch.float64),
        torch.tensor([SIGMA_H], dtype=torch.float64),
        e_src,
        e_tgt,
        r_bohr,
    ).item()

    B_literal = 1.0 / (SIGMA_O * SIGMA_H)
    x = B_literal * r_bohr.item()
    S_literal = (x * x / 3.0 + x + 1.0) * math.exp(-x)

    assert S_shipped / S_literal > 1e6


def test_cliff_exchange_angstrom_path_matches_hand_converted_bohr():
    """``dR_AB=None`` must convert Angstrom -> bohr; a supplied dR_AB is bohr.

    This is the single place a unit slip would produce a wrong-but-plausible
    exchange energy, so pin both paths against each other.
    """
    RA = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    RB = torch.tensor([[R_HBOND_ANG, 0.0, 0.0]], dtype=torch.float64)
    e_src, e_tgt = _single_edge()
    vwA = torch.tensor([SIGMA_O], dtype=torch.float64)
    vwB = torch.tensor([SIGMA_H], dtype=torch.float64)
    KA = torch.tensor([K_EXCH_O2], dtype=torch.float64)
    KB = torch.tensor([K_EXCH_HO], dtype=torch.float64)

    from_coords = cliff_exchange(RA, RB, e_src, e_tgt, vwA, vwB, KA, KB)
    from_bohr = cliff_exchange(
        RA,
        RB,
        e_src,
        e_tgt,
        vwA,
        vwB,
        KA,
        KB,
        dR_AB=torch.tensor(
            [R_HBOND_ANG / constants.au2ang], dtype=torch.float64
        ),
    )
    assert from_coords.item() == pytest.approx(from_bohr.item(), rel=1e-12)
    assert from_coords.item() == pytest.approx(GOLDEN_E_EXCH_KCAL, rel=1e-3)

    # A caller that forgot the conversion (Angstrom distances against bohr^-1
    # widths) would land on a visibly different number, so the test above is
    # not vacuous.
    wrong = cliff_exchange(
        RA,
        RB,
        e_src,
        e_tgt,
        vwA,
        vwB,
        KA,
        KB,
        dR_AB=torch.tensor([R_HBOND_ANG], dtype=torch.float64),
    )
    assert wrong.item() > 10.0 * from_coords.item()


def test_cliff_exchange_rejects_mismatched_distance_length():
    with pytest.raises(ValueError, match="dR_AB"):
        cliff_exchange(
            None,
            None,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0.39, 0.36]),
            torch.tensor([0.41]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0]),
            dR_AB=torch.tensor([3.0]),
        )


# --------------------------------------------------------------------------
# Overlap helper property tests
# --------------------------------------------------------------------------


def test_overlap_symmetry_under_monomer_exchange():
    vwA = torch.tensor([0.39, 0.52], dtype=torch.float64)
    vwB = torch.tensor([0.36, 0.44, 0.61], dtype=torch.float64)
    e_src = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    e_tgt = torch.tensor([0, 2, 1, 0], dtype=torch.long)
    dR = torch.tensor([3.1, 4.7, 5.2, 3.9], dtype=torch.float64)

    forward = atomic_overlap_S_ij(vwA, vwB, e_src, e_tgt, dR)
    swapped = atomic_overlap_S_ij(vwB, vwA, e_tgt, e_src, dR)
    torch.testing.assert_close(forward, swapped, rtol=0, atol=0)


def test_overlap_decays_monotonically_in_r():
    vwA = torch.full((1,), 0.39, dtype=torch.float64)
    vwB = torch.full((1,), 0.36, dtype=torch.float64)
    r = torch.linspace(0.5, 12.0, 40, dtype=torch.float64)
    e_src = torch.zeros(40, dtype=torch.long)
    S = atomic_overlap_S_ij(vwA, vwB, e_src, e_src, r)
    assert torch.all(S[1:] < S[:-1])
    assert torch.all(S > 0)


def test_overlap_decays_monotonically_in_B_ij():
    """Larger B_ij (i.e. smaller widths) must give a smaller overlap."""
    n = 30
    sigma = torch.linspace(0.15, 1.2, n, dtype=torch.float64)  # B_ij decreasing
    e_src = torch.arange(n, dtype=torch.long)
    dR = torch.full((n,), 4.0, dtype=torch.float64)
    S = atomic_overlap_S_ij(sigma, sigma, e_src, e_src, dR, width_floor=0.0)
    B_ij = torch.rsqrt(sigma * sigma)
    assert torch.all(B_ij[1:] < B_ij[:-1])
    assert torch.all(S[1:] > S[:-1])


def test_overlap_tends_to_one_at_zero_separation():
    e_src, e_tgt = _single_edge()
    S = atomic_overlap_S_ij(
        torch.tensor([0.39], dtype=torch.float64),
        torch.tensor([0.36], dtype=torch.float64),
        e_src,
        e_tgt,
        torch.tensor([0.0], dtype=torch.float64),
    )
    assert S.item() == pytest.approx(1.0, abs=1e-14)

    S_small = atomic_overlap_S_ij(
        torch.tensor([0.39], dtype=torch.float64),
        torch.tensor([0.36], dtype=torch.float64),
        e_src,
        e_tgt,
        torch.tensor([1e-8], dtype=torch.float64),
    )
    assert S_small.item() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_overlap_preserves_dtype_and_device(dtype):
    vwA = torch.tensor([0.39, 0.41], dtype=dtype)
    vwB = torch.tensor([0.36], dtype=dtype)
    e_src = torch.tensor([0, 1], dtype=torch.long)
    e_tgt = torch.tensor([0, 0], dtype=torch.long)
    dR = torch.tensor([3.5, 4.5], dtype=dtype)
    S = atomic_overlap_S_ij(vwA, vwB, e_src, e_tgt, dR)
    assert S.dtype == dtype
    assert S.device == vwA.device
    assert torch.isfinite(S).all()


def test_overlap_no_unit_conversion_applied():
    """The helper is dimensionless: no h2kcalmol, no au2ang."""
    e_src, e_tgt = _single_edge()
    S = atomic_overlap_S_ij(
        torch.tensor([0.39], dtype=torch.float64),
        torch.tensor([0.36], dtype=torch.float64),
        e_src,
        e_tgt,
        torch.tensor([0.0], dtype=torch.float64),
    )
    # S(0) == 1 exactly rules out any hidden multiplicative factor.
    assert S.item() == 1.0


def test_width_floor_engages_below_the_floor():
    e_src, e_tgt = _single_edge()
    dR = torch.tensor([4.0], dtype=torch.float64)
    tiny = torch.tensor([1e-4], dtype=torch.float64)  # AtomHirshfeldMPNN floor
    ok = torch.tensor([0.36], dtype=torch.float64)

    floored = atomic_overlap_S_ij(tiny, ok, e_src, e_tgt, dR)
    unfloored = atomic_overlap_S_ij(
        tiny, ok, e_src, e_tgt, dR, width_floor=0.0
    )
    clamped_ref = atomic_overlap_S_ij(
        torch.tensor([OVERLAP_WIDTH_FLOOR], dtype=torch.float64),
        ok,
        e_src,
        e_tgt,
        dR,
        width_floor=0.0,
    )
    torch.testing.assert_close(floored, clamped_ref, rtol=0, atol=0)
    assert floored.item() > unfloored.item()
    assert unfloored.item() < 1e-20  # unfloored B_ij ~ 167 bohr^-1


def test_width_floor_zero_is_a_no_op_above_the_floor():
    vwA = torch.tensor([0.39, 0.52], dtype=torch.float64)
    vwB = torch.tensor([0.36], dtype=torch.float64)
    e_src = torch.tensor([0, 1], dtype=torch.long)
    e_tgt = torch.tensor([0, 0], dtype=torch.long)
    dR = torch.tensor([3.0, 5.0], dtype=torch.float64)
    torch.testing.assert_close(
        atomic_overlap_S_ij(vwA, vwB, e_src, e_tgt, dR, width_floor=0.0),
        atomic_overlap_S_ij(vwA, vwB, e_src, e_tgt, dR),
        rtol=0,
        atol=0,
    )


def test_overlap_never_materializes_an_nA_by_nB_intermediate():
    """Output is per-edge, not [n_A, n_B]: 3 * 4 = 12 != 5 edges."""
    vwA = torch.tensor([0.39, 0.41, 0.44], dtype=torch.float64)
    vwB = torch.tensor([0.36, 0.38, 0.40, 0.42], dtype=torch.float64)
    e_src = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    e_tgt = torch.tensor([0, 3, 1, 0, 2], dtype=torch.long)
    dR = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
    S = atomic_overlap_S_ij(vwA, vwB, e_src, e_tgt, dR)
    assert S.shape == (5,)
    assert vwA.numel() * vwB.numel() == 12


def test_overlap_is_differentiable_and_finite():
    vwA = torch.tensor([0.39, 0.41], dtype=torch.float64, requires_grad=True)
    vwB = torch.tensor([0.36], dtype=torch.float64, requires_grad=True)
    e_src = torch.tensor([0, 1], dtype=torch.long)
    e_tgt = torch.tensor([0, 0], dtype=torch.long)
    dR = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    atomic_overlap_S_ij(vwA, vwB, e_src, e_tgt, dR).sum().backward()
    for t in (vwA, vwB, dR):
        assert torch.isfinite(t.grad).all()
        assert torch.any(t.grad != 0)


# --------------------------------------------------------------------------
# Refactor equivalence test
# --------------------------------------------------------------------------
#
# The literals below were captured by running the three induction routines on
# the fixture inputs BEFORE mtp_mtp.py:2470-2476 / 2592-2598 / 3051-3054 were
# refactored to delegate to atomic_overlap_S_ij.  They are the safety gate for
# that refactor, and in particular they lock in that the three sites do NOT
# apply the 0.1 width floor: the fixture deliberately includes sub-0.1 valence
# widths (0.04, 0.07) that AtomHirshfeldMPNN's relu(...) + 1e-4 head can
# really emit, so clamping there would move these numbers substantially.

# Pins for the LEGACY path (`intramolecular_permanent_field=True`). They
# still do the job they were written for -- guarding a refactor against
# numerical drift -- but they are not physically meaningful: three of the
# six no-overlap edges are positive and the sum is repulsive. Kept so a
# checkpoint trained before the fix can be reproduced exactly.
_PRE_REFACTOR_RACKERS_OVERLAP = [
    -0.2079548809718,
    -1.444842017749,
    -1.526587405973,
    0.7417372056996,
    -0.4998319728404,
    0.6196945940438,
]
_PRE_REFACTOR_RACKERS_NO_OVERLAP = [
    -0.05342854313832,
    -1.444842017749,
    1.838771066149,
    0.7417372079319,
    -0.4998319728404,
    0.6196945940438,
]
_PRE_REFACTOR_INDUCED_DIPOLE = [
    0.4109346216706,
    0.288751944437,
    -7.5915627379,
    0.6356176190447,
    -0.8057218666114,
    0.0829153364201,
]


@pytest.fixture
def overlap_regression_inputs():
    torch.manual_seed(20260820)
    nA, nB = 3, 2
    return dict(
        ZA=torch.tensor([8, 1, 1], dtype=torch.long),
        RA=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.9584, 0.0, 0.0],
                [-0.2396, 0.9281, 0.0],
            ],
            dtype=torch.float64,
        ),
        qA=torch.tensor([-0.8, 0.4, 0.4], dtype=torch.float64),
        muA=torch.randn(nA, 3, dtype=torch.float64) * 0.1,
        quadA=torch.zeros(nA, 3, 3, dtype=torch.float64),
        ZB=torch.tensor([8, 1], dtype=torch.long),
        RB=torch.tensor(
            [[2.9, 0.3, 0.1], [3.5, 1.0, -0.2]], dtype=torch.float64
        ),
        qB=torch.tensor([-0.8, 0.4], dtype=torch.float64),
        muB=torch.randn(nB, 3, dtype=torch.float64) * 0.1,
        quadB=torch.zeros(nB, 3, 3, dtype=torch.float64),
        e_AB_source=torch.arange(nA).repeat_interleave(nB),
        e_AB_target=torch.arange(nB).repeat(nA),
        e_AA_source=torch.tensor([0, 0, 1, 1, 2, 2]),
        e_AA_target=torch.tensor([1, 2, 0, 2, 0, 1]),
        e_BB_source=torch.tensor([0, 1]),
        e_BB_target=torch.tensor([1, 0]),
        hfvr_A=torch.tensor([1.05, 0.85, 0.9], dtype=torch.float64),
        hfvr_B=torch.tensor([1.02, 0.88], dtype=torch.float64),
        # sub-0.1 widths on purpose -- see comment above.
        vw_A=torch.tensor([0.39, 0.36, 0.04], dtype=torch.float64),
        vw_B=torch.tensor([0.41, 0.07], dtype=torch.float64),
        thole_direct_A=torch.tensor([0.34, 0.30, 0.32], dtype=torch.float64),
        thole_direct_B=torch.tensor([0.35, 0.31], dtype=torch.float64),
        thole_mutual_A=torch.tensor([0.39, 0.37, 0.38], dtype=torch.float64),
        thole_mutual_B=torch.tensor([0.40, 0.36], dtype=torch.float64),
        K_A=torch.tensor([1.8, 1.2, 0.9], dtype=torch.float64),
        K_B=torch.tensor([1.7, 1.1], dtype=torch.float64),
        table=constants.polarizability_table.to(torch.float64),
    )


def _run_rackers(d, include_overlap, legacy=False):
    return mtp_mtp.rackers_thole_induction(
        d["ZA"], d["RA"], d["qA"], d["muA"], d["quadA"],
        d["ZB"], d["RB"], d["qB"], d["muB"], d["quadB"],
        d["e_AB_source"], d["e_AB_target"],
        d["e_AA_source"], d["e_BB_source"],
        d["e_AA_target"], d["e_BB_target"],
        d["hfvr_A"], d["hfvr_B"], d["vw_A"], d["vw_B"],
        d["thole_direct_A"], d["thole_direct_B"],
        d["thole_mutual_A"], d["thole_mutual_B"],
        d["K_A"], d["K_B"],
        include_overlap=include_overlap,
        intramolecular_permanent_field=legacy,
        polarizability_table=d["table"],
    )


def _run_induced_dipole(d, fn):
    return fn(
        d["ZA"], d["RA"], d["qA"], d["muA"], d["quadA"],
        d["ZB"], d["RB"], d["qB"], d["muB"], d["quadB"],
        d["e_AB_source"], d["e_AB_target"],
        d["e_AA_source"], d["e_BB_source"],
        d["e_AA_target"], d["e_BB_target"],
        d["hfvr_A"], d["hfvr_B"], d["vw_A"], d["vw_B"],
        d["K_A"], d["K_B"],
        polarizability_table=d["table"],
    )


def test_rackers_thole_induction_overlap_matches_pre_refactor(
    overlap_regression_inputs,
):
    got = _run_rackers(
        overlap_regression_inputs, include_overlap=True, legacy=True
    )
    torch.testing.assert_close(
        got,
        torch.tensor(_PRE_REFACTOR_RACKERS_OVERLAP, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )


def test_rackers_thole_induction_no_overlap_matches_pre_refactor(
    overlap_regression_inputs,
):
    got = _run_rackers(
        overlap_regression_inputs, include_overlap=False, legacy=True
    )
    torch.testing.assert_close(
        got,
        torch.tensor(_PRE_REFACTOR_RACKERS_NO_OVERLAP, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    # The overlap branch must actually be doing something on this fixture,
    # otherwise the regression pin above would be vacuous.
    assert not torch.allclose(
        got,
        _run_rackers(
            overlap_regression_inputs, include_overlap=True, legacy=True
        ),
    )


# The corrected path. Per-edge values are deliberately not sign-constrained --
# induction is many-body and a single intermolecular pair's share of it can be
# either sign -- so the invariant is on the sum, which is the quantity that is
# trained on, evaluated, and gated. On this fixture the sum goes from +1.202
# (repulsive, and impossible) to -1.450.
_CORRECTED_RACKERS_NO_OVERLAP = [
    0.089248862996,
    0.227372857839,
    -1.663607802379,
    0.310104689048,
    -0.472203538697,
    0.059355193314,
]
_CORRECTED_RACKERS_OVERLAP = [
    -0.065277474837,
    0.227372857839,
    -5.028966274500,
    0.310104686816,
    -0.472203538697,
    0.059355193314,
]


@pytest.mark.parametrize(
    "include_overlap,expected",
    [
        (False, _CORRECTED_RACKERS_NO_OVERLAP),
        (True, _CORRECTED_RACKERS_OVERLAP),
    ],
)
def test_rackers_thole_induction_corrected_values(
    overlap_regression_inputs, include_overlap, expected
):
    got = _run_rackers(overlap_regression_inputs, include_overlap=include_overlap)
    torch.testing.assert_close(
        got,
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("include_overlap", [False, True])
def test_the_corrected_path_is_attractive_on_this_fixture(
    overlap_regression_inputs, include_overlap
):
    """The sign invariant, with and without the overlap term."""
    corrected = _run_rackers(
        overlap_regression_inputs, include_overlap=include_overlap
    ).sum()
    assert corrected.item() < 0, corrected.item()


def test_the_legacy_polarization_alone_was_repulsive_here():
    """Why the overlap term looked healthier than it was.

    The legacy polarization energy sums to +1.202 on this fixture -- repulsive,
    which induction cannot be. Adding the overlap term drags the total back to
    -2.318, so the *total* looked fine while the term underneath it had the
    wrong sign. That is the same masking seen on S66x8, where `cliff2_ind_ipd`
    was positive on 421 of 528 geometries but `cliff2_ind` on only 341.
    """
    legacy_polarization = sum(_PRE_REFACTOR_RACKERS_NO_OVERLAP)
    legacy_total = sum(_PRE_REFACTOR_RACKERS_OVERLAP)
    assert legacy_polarization > 0
    assert legacy_total < 0


@pytest.mark.parametrize(
    "fn",
    [
        mtp_mtp.induced_dipole_induction,
        mtp_mtp.induced_dipole_induction_optimized,
    ],
    ids=["induced_dipole_induction", "induced_dipole_induction_optimized"],
)
def test_induced_dipole_induction_matches_pre_refactor(
    fn, overlap_regression_inputs
):
    got = _run_induced_dipole(overlap_regression_inputs, fn)
    torch.testing.assert_close(
        got,
        torch.tensor(_PRE_REFACTOR_INDUCED_DIPOLE, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )


def test_refactored_sites_use_the_shared_helper(overlap_regression_inputs):
    """The three call sites really delegate; patching the helper moves them."""
    d = overlap_regression_inputs
    calls = []
    real = mtp_mtp.atomic_overlap_S_ij

    def spy(*args, **kwargs):
        calls.append(kwargs.get("width_floor", OVERLAP_WIDTH_FLOOR))
        return real(*args, **kwargs)

    mtp_mtp.atomic_overlap_S_ij = spy
    try:
        _run_rackers(d, include_overlap=True)
        _run_induced_dipole(d, mtp_mtp.induced_dipole_induction)
    finally:
        mtp_mtp.atomic_overlap_S_ij = real

    assert len(calls) == 2
    # Both legacy sites must opt out of the width floor to stay numerically
    # identical to their pre-refactor behavior.
    assert calls == [0.0, 0.0]


# --------------------------------------------------------------------------
# apnet3 legacy pin test
# --------------------------------------------------------------------------


def test_apnet3_valence_width_exch_legacy_pin():
    """Pin apnet3's deliberately non-physical S_ij.

    ``valence_width_exch`` uses B_ij = 1/(sigma_i sigma_j) -- CLIFF Eq. (11)
    without its square root -- and folds ``hartree2kcal`` into the returned
    value.  That is *not* a bug to fix: the result is multiplied by the learned
    ``readout_layer_exch_quotient``, so the missing square root has been
    absorbed into fitted weights.  "Correcting" B_ij here would silently
    invalidate every trained AP3 checkpoint.  New physics must call
    ``mtp_mtp.atomic_overlap_S_ij`` instead.
    """
    e_source = torch.tensor([0, 0, 1], dtype=torch.long)
    e_target = torch.tensor([0, 1, 0], dtype=torch.long)
    vwA = torch.tensor([1.20, 0.05], dtype=torch.float64)
    vwB = torch.tensor([1.10, 0.90], dtype=torch.float64)
    r_ij = torch.tensor([2.0, 3.0, 2.5], dtype=torch.float64)

    got = APNet3_MPNN.valence_width_exch(
        None, e_source, e_target, vwA, vwB, r_ij
    )

    # Hand-evaluated legacy form, including the 0.1 where() floor, the literal
    # (un-square-rooted) B_ij, and the folded hartree->kcal/mol factor.
    h2kcal = qcel.constants.conversion_factor("hartree", "kcal/mol")
    expected = []
    for i, j, r in zip(
        e_source.tolist(), e_target.tolist(), r_ij.tolist()
    ):
        sa = max(vwA[i].item(), 0.1)
        sb = max(vwB[j].item(), 0.1)
        B = 1.0 / (sa * sb)
        x = B * r
        expected.append((x * x / 3.0 + x + 1.0) * math.exp(-x) * h2kcal)
    torch.testing.assert_close(
        got,
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-12,
        atol=0,
    )

    # And it is emphatically NOT the physical helper.
    physical = atomic_overlap_S_ij(
        vwA, vwB, e_source, e_target, r_ij, width_floor=0.1
    )
    assert not torch.allclose(got / h2kcal, physical, rtol=1e-3)


# --------------------------------------------------------------------------
# Combination-rule test
# --------------------------------------------------------------------------


def test_exchange_uses_product_not_geometric_mean():
    """K_i * K_j, not sqrt(K_i * K_j).

    Values are chosen so the two rules differ by a wide margin: for
    K_i = 5.8538 and K_j = 0.5996 the product is 3.510 while the geometric
    mean is 1.873.
    """
    e_src, e_tgt = _single_edge()
    r_bohr = torch.tensor([3.6849659, 3.6849659], dtype=torch.float64)[:1]
    vwA = torch.tensor([SIGMA_O], dtype=torch.float64)
    vwB = torch.tensor([SIGMA_H], dtype=torch.float64)
    KA = torch.tensor([K_EXCH_O2], dtype=torch.float64)
    KB = torch.tensor([K_EXCH_HO], dtype=torch.float64)

    product = (KA[0] * KB[0]).item()
    geo_mean = geometric_mean_edge_values(KA, KB, e_src, e_tgt)[0].item()
    assert product == pytest.approx(3.5099, rel=1e-3)
    assert geo_mean == pytest.approx(1.8734, rel=1e-3)
    assert abs(product - geo_mean) > 1.0

    S_ij = atomic_overlap_S_ij(vwA, vwB, e_src, e_tgt, r_bohr).item()
    E = cliff_exchange(
        None, None, e_src, e_tgt, vwA, vwB, KA, KB, dR_AB=r_bohr
    ).item()

    assert E == pytest.approx(product * S_ij * constants.h2kcalmol, rel=1e-12)
    assert E != pytest.approx(
        geo_mean * S_ij * constants.h2kcalmol, rel=1e-3
    )


def test_exchange_path_does_not_call_geometric_mean(monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError(
            "cliff_exchange must not use geometric_mean_edge_values"
        )

    monkeypatch.setattr(mtp_mtp, "geometric_mean_edge_values", boom)
    e_src, e_tgt = _single_edge()
    out = cliff_exchange(
        None,
        None,
        e_src,
        e_tgt,
        torch.tensor([SIGMA_O], dtype=torch.float64),
        torch.tensor([SIGMA_H], dtype=torch.float64),
        torch.tensor([K_EXCH_O2], dtype=torch.float64),
        torch.tensor([K_EXCH_HO], dtype=torch.float64),
        dR_AB=torch.tensor([3.6849659], dtype=torch.float64),
    )
    assert torch.isfinite(out).all()


def test_exchange_is_bilinear_in_K():
    """Product rule => scaling K_i alone scales the edge energy linearly."""
    e_src, e_tgt = _single_edge()
    dR = torch.tensor([3.5], dtype=torch.float64)
    args = (
        torch.tensor([SIGMA_O], dtype=torch.float64),
        torch.tensor([SIGMA_H], dtype=torch.float64),
    )
    base = cliff_exchange(
        None, None, e_src, e_tgt, *args,
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([3.0], dtype=torch.float64),
        dR_AB=dR,
    ).item()
    doubled = cliff_exchange(
        None, None, e_src, e_tgt, *args,
        torch.tensor([4.0], dtype=torch.float64),
        torch.tensor([3.0], dtype=torch.float64),
        dR_AB=dR,
    ).item()
    # sqrt-mean would give a factor of sqrt(2) instead of 2.
    assert doubled / base == pytest.approx(2.0, rel=1e-12)


# --------------------------------------------------------------------------
# Sign and magnitude test
# --------------------------------------------------------------------------

_WATER_CLOSE = (
    "tests/test_data_path/test_geoms/many_geom/mol_cliff_water_close.dat"
)


@pytest.fixture
def cliff_water_close_dimer():
    """Real close-contact water dimer (stored in bohr, no_com/no_reorient)."""
    with open(_WATER_CLOSE) as fh:
        mol = qcel.models.Molecule.from_data(fh.read())
    frag_a, frag_b = mol.get_fragment(0), mol.get_fragment(1)
    # qcelemental stores geometry in bohr; cliff_exchange wants Angstrom.
    RA = torch.tensor(
        frag_a.geometry * constants.au2ang, dtype=torch.float64
    )
    RB = torch.tensor(
        frag_b.geometry * constants.au2ang, dtype=torch.float64
    )
    return RA, RB


def _full_edges(nA, nB):
    return (
        torch.arange(nA).repeat_interleave(nB),
        torch.arange(nB).repeat(nA),
    )


def _water_widths_and_K(n):
    # O then H, H -- physical CLIFF-scale values, not fitted.
    sigma = torch.tensor([SIGMA_O, SIGMA_H, SIGMA_H], dtype=torch.float64)
    K = torch.tensor([K_EXCH_O2, K_EXCH_HO, K_EXCH_HO], dtype=torch.float64)
    return sigma[:n], K[:n]


def test_exchange_strictly_positive_on_real_dimer(cliff_water_close_dimer):
    RA, RB = cliff_water_close_dimer
    e_src, e_tgt = _full_edges(RA.shape[0], RB.shape[0])
    vwA, KA = _water_widths_and_K(RA.shape[0])
    vwB, KB = _water_widths_and_K(RB.shape[0])
    E = cliff_exchange(RA, RB, e_src, e_tgt, vwA, vwB, KA, KB)
    assert E.shape == (RA.shape[0] * RB.shape[0],)
    assert torch.all(E > 0.0)
    assert torch.isfinite(E).all()
    # Close contact => a chemically sensible total, order 1-100 kcal/mol.
    assert 1e-2 < E.sum().item() < 1e3


def test_exchange_decays_monotonically_with_separation(
    cliff_water_close_dimer,
):
    RA, RB = cliff_water_close_dimer
    e_src, e_tgt = _full_edges(RA.shape[0], RB.shape[0])
    vwA, KA = _water_widths_and_K(RA.shape[0])
    vwB, KB = _water_widths_and_K(RB.shape[0])
    # Push B away from A along the centroid-separation direction.
    axis = RB.mean(0) - RA.mean(0)
    axis = axis / axis.norm()

    totals = []
    for shift in torch.linspace(0.0, 4.0, 9, dtype=torch.float64):
        E = cliff_exchange(
            RA, RB + shift * axis, e_src, e_tgt, vwA, vwB, KA, KB
        )
        assert torch.all(E > 0.0)
        totals.append(E.sum().item())

    for prev, nxt in zip(totals[:-1], totals[1:]):
        assert nxt < prev
    assert totals[-1] < 1e-3 * totals[0]


# ==========================================================================
# Generalized positive-parameter contract (_validate_positive_initialization)
# ==========================================================================

# The Rackers wrapper's error text is a hard compatibility surface: it is
# asserted on by tests/test_rackers_thole_damping.py and surfaced verbatim to
# users by train_models.py, which calls _validate_rackers_initialization
# directly to validate CLI overrides.  Generalizing the validator must not
# change a single byte of any of these.
def _rackers_dtype_messages():
    dtype = torch.get_default_dtype()
    return (
        "transformed param_start_mean values must be finite and "
        f"representable in the {dtype} embedding dtype",
        "param_start_std values must be representable in the "
        f"{dtype} embedding dtype",
    )


def test_validate_positive_initialization_returns_softplus_preimages():
    positive_means, raw_stds, epsilon, raw_means = (
        _validate_positive_initialization(
            CLIFF_CLASSICAL_PARAMETER_NAMES,
            CLIFF_CLASSICAL_INITIAL_VALUES,
            CLIFF_CLASSICAL_INITIAL_STDS,
            1e-8,
        )
    )
    assert positive_means == list(CLIFF_CLASSICAL_INITIAL_VALUES)
    assert raw_stds == list(CLIFF_CLASSICAL_INITIAL_STDS)
    assert epsilon == 1e-8
    # `raw_means` must be the inverse-softplus pre-images, so a zeroed
    # correction head reproduces the requested positive values exactly.
    recovered = F.softplus(torch.tensor(raw_means, dtype=torch.float64))
    recovered = recovered + epsilon
    assert torch.allclose(
        recovered,
        torch.tensor(CLIFF_CLASSICAL_INITIAL_VALUES, dtype=torch.float64),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "parameter_names,expected",
    [
        (CLIFF_EXCH_PARAMETER_NAMES, "exactly one value"),
        (CLIFF_CLASSICAL_PARAMETER_NAMES, "exactly five values"),
        (RACKERS_PARAMETER_NAMES, "exactly four values"),
    ],
)
@pytest.mark.parametrize("field", ["param_start_mean", "param_start_std"])
def test_validate_positive_initialization_rejects_wrong_length(
    parameter_names, expected, field
):
    """Wrong-length lists raise, reporting len(parameter_names)."""
    kwargs = {
        "param_start_mean": [1.0] * len(parameter_names),
        "param_start_std": [0.0] * len(parameter_names),
    }
    kwargs[field] = [1.0] * (len(parameter_names) + 1)
    with pytest.raises(ValueError) as excinfo:
        _validate_positive_initialization(
            parameter_names,
            kwargs["param_start_mean"],
            kwargs["param_start_std"],
            1e-8,
        )
    assert str(excinfo.value) == f"{field} must contain {expected}"


@pytest.mark.parametrize("field", ["param_start_mean", "param_start_std"])
def test_validate_positive_initialization_rejects_non_sequence(field):
    """A bare scalar is not silently broadcast to n_params."""
    kwargs = {
        "param_start_mean": list(CLIFF_EXCH_INITIAL_VALUES),
        "param_start_std": list(CLIFF_EXCH_INITIAL_STDS),
    }
    kwargs[field] = 2.5
    with pytest.raises(ValueError, match="exactly one value"):
        _validate_positive_initialization(
            CLIFF_EXCH_PARAMETER_NAMES,
            kwargs["param_start_mean"],
            kwargs["param_start_std"],
            1e-8,
        )


def test_rackers_wrapper_length_messages_are_byte_identical():
    with pytest.raises(ValueError) as mean_exc:
        _validate_rackers_initialization([0.1, 0.2, 0.3], list(RACKERS_INITIAL_STDS), 1e-8)
    assert str(mean_exc.value) == (
        "param_start_mean must contain exactly four values"
    )

    with pytest.raises(ValueError) as std_exc:
        _validate_rackers_initialization(list(RACKERS_INITIAL_VALUES), [0.01], 1e-8)
    assert str(std_exc.value) == (
        "param_start_std must contain exactly four values"
    )


@pytest.mark.parametrize(
    "mean,std,epsilon,expected",
    [
        (
            RACKERS_INITIAL_VALUES,
            RACKERS_INITIAL_STDS,
            0.0,
            "positivity_epsilon must be finite and strictly greater than zero",
        ),
        (
            RACKERS_INITIAL_VALUES,
            RACKERS_INITIAL_STDS,
            float("nan"),
            "positivity_epsilon must be finite and strictly greater than zero",
        ),
        (
            (1.8, 0.0, 0.39, 1.8),
            RACKERS_INITIAL_STDS,
            1e-8,
            "param_start_mean values must be finite and strictly greater than "
            "positivity_epsilon",
        ),
        (
            RACKERS_INITIAL_VALUES,
            (0.01, -0.1, 0.01, 0.01),
            1e-8,
            "param_start_std values must be finite and greater than or equal "
            "to zero",
        ),
    ],
)
def test_rackers_wrapper_domain_messages_are_byte_identical(
    mean, std, epsilon, expected
):
    with pytest.raises(ValueError) as excinfo:
        _validate_rackers_initialization(list(mean), list(std), epsilon)
    assert str(excinfo.value) == expected


@pytest.mark.parametrize("field", ["param_start_mean", "param_start_std"])
def test_rackers_wrapper_dtype_messages_are_byte_identical(field):
    mean_message, std_message = _rackers_dtype_messages()
    kwargs = {
        "param_start_mean": list(RACKERS_INITIAL_VALUES),
        "param_start_std": list(RACKERS_INITIAL_STDS),
    }
    if field == "param_start_mean":
        kwargs[field] = [1e39, 1.0, 1.0, 1.0]
        expected = mean_message
    else:
        kwargs[field] = [1e39, 0.0, 0.0, 0.0]
        expected = std_message
    with pytest.raises(ValueError) as excinfo:
        _validate_rackers_initialization(
            kwargs["param_start_mean"], kwargs["param_start_std"], 1e-8
        )
    assert str(excinfo.value) == expected


def test_rackers_wrapper_delegates_to_generalized_validator(monkeypatch):
    """The wrapper binds RACKERS_PARAMETER_NAMES and forwards everything else."""
    seen = {}

    def spy(parameter_names, mean, std, epsilon):
        seen["names"] = parameter_names
        seen["args"] = (mean, std, epsilon)
        return ([], [], epsilon, [])

    monkeypatch.setattr(
        mtp_mtp, "_validate_positive_initialization", spy
    )
    mtp_mtp._validate_rackers_initialization([1.0] * 4, [0.0] * 4, 1e-7)
    assert tuple(seen["names"]) == RACKERS_PARAMETER_NAMES
    assert seen["args"] == ([1.0] * 4, [0.0] * 4, 1e-7)


# ==========================================================================
# Parameter-head tests: CliffExchangeNN / CliffClassicalNN
# ==========================================================================

_HEAD_CASES = [
    (
        CliffExchangeNN,
        "CliffExchangeNN",
        CLIFF_EXCH_PARAMETER_NAMES,
        CLIFF_EXCH_INITIAL_VALUES,
        CLIFF_EXCH_INITIAL_STDS,
    ),
    (
        CliffClassicalNN,
        "CliffClassicalNN",
        CLIFF_CLASSICAL_PARAMETER_NAMES,
        CLIFF_CLASSICAL_INITIAL_VALUES,
        CLIFF_CLASSICAL_INITIAL_STDS,
    ),
]
_HEAD_IDS = [case[1] for case in _HEAD_CASES]


def _build_head(model_type, nested, **kwargs):
    kwargs.setdefault("n_message", 1)
    kwargs.setdefault("n_neuron", 8)
    kwargs.setdefault("n_embed", 4)
    return model_type(atom_model=nested, **kwargs)


def _zero_readout_heads(model):
    """Zero every correction MLP so only the guess embedding survives."""
    with torch.no_grad():
        for head in model.param_readout_layers:
            for readout in head:
                for parameter in readout.parameters():
                    parameter.zero_()


@pytest.mark.parametrize(
    "model_type,_name,parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_output_shape(
    model_type, _name, parameter_names, _values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    """[n_atoms, 1] for exchange and [n_atoms, 5] for classical."""
    model = _build_head(model_type, nested_hfvr_vw_model)
    parameters = model(atomic_batch)[-1]
    n_atoms = atomic_batch.x.numel()
    assert parameters.dim() == 2
    assert parameters.shape == (n_atoms, len(parameter_names))
    assert model.n_params == len(parameter_names)


def test_cliff_exchange_head_undoes_the_n_params_one_squeeze(
    atomic_batch, nested_hfvr_vw_model
):
    """The single-parameter head must still return [n_atoms, 1].

    `AtomTypeParamNN.forward` returns `K.squeeze(-1)` when `n_params == 1`, so
    a naive subclass would hand back `[n_atoms]` and every
    `parameters[:, CLIFF_EXCH_INDEX]` read downstream would raise (or, worse,
    index atoms instead of columns).  Pin both halves: the base class really
    does squeeze, and `CliffExchangeNN` really does restore the column axis.
    """
    n_atoms = atomic_batch.x.numel()

    squeezing_base = AtomTypeParamNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        n_message=1,
        n_neuron=8,
        n_embed=4,
        param_start_mean=[1.0],
        param_start_std=[0.0],
        n_params=1,
    )
    base_parameters = squeezing_base(atomic_batch)[-1]
    assert base_parameters.shape == (n_atoms,)

    model = _build_head(CliffExchangeNN, nested_hfvr_vw_model)
    parameters = model(atomic_batch)[-1]
    assert parameters.shape == (n_atoms, 1)
    # Column indexing must work identically for either CLIFF head.
    assert parameters[:, CLIFF_EXCH_INDEX].shape == (n_atoms,)

    classical = _build_head(
        CliffClassicalNN, copy.deepcopy(nested_hfvr_vw_model)
    )
    classical_parameters = classical(atomic_batch)[-1]
    assert classical_parameters.shape == (n_atoms, 5)
    assert (
        classical_parameters[:, CLIFF_CLASSICAL_EXCH_INDEX].shape
        == parameters[:, CLIFF_EXCH_INDEX].shape
    )


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_outputs_finite_and_strictly_positive(
    model_type, _name, _parameter_names, _values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    model = _build_head(model_type, nested_hfvr_vw_model)
    parameters = model(atomic_batch)[-1]
    assert torch.isfinite(parameters).all()
    assert torch.all(parameters > 0.0)
    # Positivity comes from softplus, so it must survive a hostile correction
    # head too -- no abs / clamp is allowed to be doing the work.
    with torch.no_grad():
        for head in model.param_readout_layers:
            for readout in head:
                for parameter in readout.parameters():
                    parameter.fill_(-25.0)
    hostile = model(atomic_batch)[-1]
    assert torch.isfinite(hostile).all()
    assert torch.all(hostile > 0.0)
    assert torch.all(hostile >= model.positivity_epsilon)


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_zeroed_corrections_recover_initial_values(
    model_type, _name, _parameter_names, values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    """Zeroed correction heads recover the *scalar* seeds when no per-Z table.

    ``param_start_mean_by_Z={}`` opts out of the per-element seeding, which is
    the only configuration in which every atom shares a column's scalar seed.
    The default configuration seeds ``exch`` per element and is pinned by
    :func:`test_cliff_head_zeroed_corrections_recover_per_element_values`.
    """
    model = _build_head(
        model_type, nested_hfvr_vw_model, param_start_mean_by_Z={}
    )
    _zero_readout_heads(model)
    parameters = model(atomic_batch)[-1]
    expected = torch.tensor(values, dtype=parameters.dtype)
    # The tolerance is derived from the configured initialization spread rather
    # than hard-coded: `param_start_std` is a *raw*-space standard deviation and
    # `dK/draw = sigmoid(raw) <= 1`, so the emitted per-atom noise is at most
    # `std`, and the mean over `n_atoms` has standard error at most
    # `std / sqrt(n_atoms)`.  Four of those is a ~4-sigma band.  Hard-coding
    # 0.05 silently pinned this to `std = 0.01` and broke the moment the
    # exchange column moved to 0.25; the exact contract is asserted below with
    # zero noise.
    n_atoms = parameters.shape[0]
    atol = 4.0 * max(model.param_start_std) / math.sqrt(n_atoms)
    assert torch.allclose(parameters.mean(dim=0), expected, atol=atol)

    # With zero initialization noise the recovery is exact, not merely close.
    exact = _build_head(
        model_type,
        copy.deepcopy(nested_hfvr_vw_model),
        param_start_std=[0.0] * len(values),
        param_start_mean_by_Z={},
    )
    _zero_readout_heads(exact)
    exact_parameters = exact(atomic_batch)[-1]
    assert torch.allclose(
        exact_parameters,
        expected.expand_as(exact_parameters),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "model_type,_name,parameter_names,values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_zeroed_corrections_recover_per_element_values(
    model_type, _name, parameter_names, values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    """By default each atom recovers *its own element's* ``K_exch`` seed.

    This is the load-bearing half of the initialization fix: a uniform
    ``K_exch`` makes an H-H pair an order of magnitude too repulsive, because
    Eq. (8) combines the two multiplicatively.  The batch is water, so oxygen
    must come back at 5.6 and both hydrogens at 0.77 -- not all three at 2.5.
    Columns without a per-element table keep their scalar seed.
    """
    exact = _build_head(
        model_type,
        nested_hfvr_vw_model,
        param_start_std=[0.0] * len(values),
    )
    _zero_readout_heads(exact)
    parameters = exact(atomic_batch)[-1]

    by_Z = exact.param_start_mean_by_Z
    assert by_Z == {"exch": dict(mtp_mtp.CLIFF_EXCH_INITIAL_VALUES_BY_Z)}

    expected = torch.tensor(values, dtype=parameters.dtype).expand(
        atomic_batch.x.shape[0], len(values)
    ).clone()
    for name, table in by_Z.items():
        column = parameter_names.index(name)
        for row, z in enumerate(atomic_batch.x.tolist()):
            if z in table:
                expected[row, column] = table[z]
    assert torch.allclose(parameters, expected, atol=1e-5)

    # Water is O/H/H, so the exchange column must not be constant.
    exch_column = parameter_names.index("exch")
    assert parameters[:, exch_column].std() > 1.0


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_preserves_wrapped_outputs(
    model_type, _name, _parameter_names, _values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    """Multipoles and the HFVR / valence-width columns must pass through.

    The physics paths read Hirshfeld volume ratios as ``abs(output[-2][:, 0])``
    and valence widths as ``output[-2][:, 1])``; appending a parameter column
    must not shift either of them.
    """
    model = _build_head(model_type, nested_hfvr_vw_model)
    nested_output = nested_hfvr_vw_model(atomic_batch)
    output = model(atomic_batch)

    assert len(output) == len(nested_output) + 1
    for wrapped, expected in zip(output[:-1], nested_output):
        assert torch.allclose(wrapped, expected)

    nested_parameters = nested_output[-1]
    assert nested_parameters.shape == (atomic_batch.x.numel(), 2)
    assert output[-2] is not output[-1]
    assert torch.allclose(output[-2], nested_parameters)
    # HFVR (column 0) and valence width (column 1) land where the physics
    # paths expect them.
    assert torch.allclose(output[-2][:, 0], nested_parameters[:, 0])
    assert torch.allclose(output[-2][:, 1], nested_parameters[:, 1])
    assert torch.isfinite(output[-2]).all()


@pytest.mark.parametrize(
    "model_type,_name,parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_parameters_are_mutually_independent(
    model_type, _name, parameter_names, _values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    """Each column must be its own learned function of the shared features.

    These are per-component atom-type parameters: the electrostatic damping K
    and the Thole damping parameters describe different physics and must not
    share a learned parameter. Nothing in the architecture forces that -- one
    reused ``param_readout_layers`` index would couple two components silently,
    and every other test here would still pass -- so it is pinned directly.
    """
    head = _build_head(model_type, nested_hfvr_vw_model)
    for column in range(len(parameter_names)):
        head.zero_grad(set_to_none=True)
        head(atomic_batch)[-1][:, column].sum().backward()
        for p, name in enumerate(parameter_names):
            embedding_grad = head.guess_layer[p].weight.grad
            touched = embedding_grad is not None and bool(
                embedding_grad.abs().sum() > 0
            )
            touched = touched or any(
                q.grad is not None and bool(q.grad.abs().sum() > 0)
                for readout in head.param_readout_layers[p]
                for q in readout.parameters()
            )
            assert touched is (p == column), (
                f"d(K[:, {column}])/d({name} head) should be "
                f"{'nonzero' if p == column else 'zero'}"
            )


@pytest.mark.parametrize(
    "model_type,_name,parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_gradients_finite_at_every_readout(
    model_type, _name, parameter_names, _values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    model = _build_head(model_type, nested_hfvr_vw_model)
    _zero_readout_heads(model)
    parameters = model(atomic_batch)[-1]
    parameters.sum().backward()

    assert len(model.param_readout_layers) == len(parameter_names)
    for head in model.param_readout_layers:
        assert len(head) == model.n_message + 1
        for readout in head:
            gradients = [p.grad for p in readout.parameters()]
            assert all(g is not None for g in gradients)
            assert all(torch.isfinite(g).all() for g in gradients)
            # The final Linear's bias sees dL/dK directly, so a nonzero
            # gradient there proves the head is genuinely energy-active
            # rather than merely allocated.
            final_bias_grad = readout[-1].bias.grad
            assert torch.all(final_bias_grad != 0.0)

    # Frozen by default; unfreezing propagates to the nested model.
    assert all(
        not parameter.requires_grad
        for parameter in model.atom_model.parameters()
    )
    unfrozen = _build_head(
        model_type,
        copy.deepcopy(nested_hfvr_vw_model),
        freeze_atom_model=False,
    )
    assert all(
        parameter.requires_grad
        for parameter in unfrozen.atom_model.parameters()
    )


@pytest.mark.parametrize(
    "model_type,_name,parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_rejects_n_params(
    model_type, _name, parameter_names, _values, _stds,
    nested_hfvr_vw_model,
):
    """The output count is fixed; `n_params` must not be configurable."""
    signature = inspect.signature(model_type.__init__)
    assert "n_params" not in signature.parameters
    with pytest.raises(TypeError):
        model_type(
            atom_model=nested_hfvr_vw_model,
            n_params=len(parameter_names) + 1,
        )


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_rejects_non_atomtypeparamnn_nested_model(
    model_type, _name, _parameter_names, _values, _stds,
    nested_hfvr_vw_model,
):
    """Only an exact AtomTypeParamNN supplies the HFVR / valence-width pair."""
    with pytest.raises(ValueError, match="AtomTypeParamNN"):
        model_type(
            atom_model=AtomMPNN(
                n_message=1,
                n_rbf=2,
                n_neuron=8,
                n_embed=4,
            )
        )
    # A *subclass* is also rejected: `type(...) is not AtomTypeParamNN`,
    # matching RackersTholeDampingNN.
    subclass_instance = RackersTholeDampingNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    with pytest.raises(ValueError, match="AtomTypeParamNN"):
        model_type(atom_model=subclass_instance)


@pytest.mark.parametrize(
    "model_type,_name,parameter_names,values,stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_initialization_length_errors_report_true_count(
    model_type, _name, parameter_names, values, stds,
    nested_hfvr_vw_model,
):
    expected = "exactly one value" if len(values) == 1 else "exactly five values"
    with pytest.raises(ValueError) as mean_exc:
        _build_head(
            model_type,
            nested_hfvr_vw_model,
            param_start_mean=list(values) + [1.0],
        )
    assert str(mean_exc.value) == f"param_start_mean must contain {expected}"
    with pytest.raises(ValueError) as std_exc:
        _build_head(
            model_type,
            nested_hfvr_vw_model,
            param_start_std=list(stds) + [0.0],
        )
    assert str(std_exc.value) == f"param_start_std must contain {expected}"


@pytest.mark.parametrize(
    "model_type,name,parameter_names,values,stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_get_config_round_trips_parameter_contract(
    model_type, name, parameter_names, values, stds,
    atomic_batch, nested_hfvr_vw_model,
):
    model = _build_head(
        model_type,
        nested_hfvr_vw_model,
        positivity_epsilon=1e-7,
        width_floor=0.25,
    )
    config = model.get_config()

    assert config["model_type"] == name
    # Exact order matters: the column index constants depend on it.
    assert config["parameter_names"] == list(parameter_names)
    assert config["param_start_mean"] == list(values)
    assert config["param_start_std"] == list(stds)
    assert config["positivity_epsilon"] == 1e-7
    assert config["width_floor"] == 0.25
    assert config["n_message"] == 1
    assert config["n_neuron"] == 8
    assert config["n_embed"] == 4
    assert config["nested_atom_model"]["model_type"] == "AtomTypeParamNN"
    assert (
        config["nested_atom_model"]["atom_model"]["model_type"] == "AtomMPNN"
    )

    rebuilt = model_type(
        atom_model=_rebuild_nested_atom_model(
            config["nested_atom_model"], freeze_atom_model=True
        ),
        n_message=config["n_message"],
        n_neuron=config["n_neuron"],
        n_embed=config["n_embed"],
        param_start_mean=config["param_start_mean"],
        param_start_std=config["param_start_std"],
        positivity_epsilon=config["positivity_epsilon"],
        width_floor=config["width_floor"],
    )
    assert rebuilt.get_config() == config
    rebuilt.load_state_dict(model.state_dict())
    assert torch.allclose(
        rebuilt(atomic_batch)[-1], model(atomic_batch)[-1]
    )


def test_cliff_classical_column_indices_match_parameter_names():
    """The named index constants are the only sanctioned column accessors."""
    assert CLIFF_EXCH_PARAMETER_NAMES[CLIFF_EXCH_INDEX] == "exch"
    for index, name in (
        (CLIFF_CLASSICAL_ELST_INDEX, "elst"),
        (CLIFF_CLASSICAL_THOLE_DIRECT_INDEX, "thole_direct"),
        (CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX, "thole_mutual"),
        (CLIFF_CLASSICAL_IND_OVERLAP_INDEX, "ind_overlap"),
        (CLIFF_CLASSICAL_EXCH_INDEX, "exch"),
    ):
        assert CLIFF_CLASSICAL_PARAMETER_NAMES[index] == name
    # Columns 0-3 deliberately mirror the Rackers ordering so the existing
    # electrostatics / induction paths are reused unchanged.
    assert (
        CLIFF_CLASSICAL_PARAMETER_NAMES[:4] == RACKERS_PARAMETER_NAMES
    )


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
@pytest.mark.parametrize(
    "invalid_width_floor",
    [0.0, -0.1, float("nan"), float("inf"), -float("inf"), None],
)
def test_cliff_head_rejects_invalid_width_floor(
    model_type, _name, _parameter_names, _values, _stds,
    nested_hfvr_vw_model, invalid_width_floor,
):
    """A *model config* width_floor must be > 0 and finite.

    This is stricter than the `atomic_overlap_S_ij` argument, which accepts
    `0.0` as the documented legacy-parity bypass for the three pre-existing
    induction-overlap call sites (see
    `test_width_floor_zero_is_a_no_op_above_the_floor`).  A config value of
    `0.0` would instead silently remove the guard protecting `rsqrt` from a
    degenerate predicted valence width.
    """
    with pytest.raises(
        ValueError,
        match="width_floor must be finite and strictly greater than zero",
    ):
        _build_head(
            model_type,
            nested_hfvr_vw_model,
            width_floor=invalid_width_floor,
        )


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_width_floor_defaults_to_module_constant(
    model_type, _name, _parameter_names, _values, _stds,
    nested_hfvr_vw_model,
):
    model = _build_head(model_type, nested_hfvr_vw_model)
    assert model.width_floor == OVERLAP_WIDTH_FLOOR
    assert model.get_config()["width_floor"] == OVERLAP_WIDTH_FLOOR


# --------------------------------------------------------------------------
# Task C: FULL_EDGE_DIMER_EVAL_MODES
# --------------------------------------------------------------------------


def _accepted_dimer_eval_modes():
    """Every string literal `DimerProp.set_forward` dispatches on."""
    source = inspect.getsource(mtp_mtp.DimerProp.set_forward)
    body = source.split('"""', 2)[-1]
    return tuple(re.findall(r'dimer_eval == "([^"]+)"', body))


def _forward_source_for_mode(mode):
    """Source of the forward selected for `mode`, plus any delegate it calls.

    The per-mode forwards for the Rackers and CLIFF families are thin wrappers
    around a shared `*_common_forward`, so the edge domain only becomes visible
    once the delegate is followed.
    """
    dimer = mtp_mtp.DimerProp.__new__(mtp_mtp.DimerProp)
    torch.nn.Module.__init__(dimer)
    dimer.AtomTypeParam = _ControlledCliffClassicalAtomParam()
    dimer.elst_damping_type = "CLIFF"
    dimer.set_d3_damping_parameters(None)
    dimer.set_forward(mode)
    source = inspect.getsource(dimer.forward)
    for delegate in re.findall(r"self\.(_\w*common_forward)\(", source):
        source += inspect.getsource(getattr(dimer, delegate))
    return source


# `dimer_eval` modes whose per-edge output is aggregated by a model *other*
# than `AM_DimerParam_Model`: the `apnet3_fused` family calls
# `DimerProp.set_forward("ap3_...")` directly (apnet3_d3_fused.py:949) and owns
# its own scatter, and `"disp"` is a bare `d3` probe that
# `AM_DimerParam_Model.train` rejects outright in its `y_ind` dispatch.  Their
# edge domain therefore is not, and never was, governed by
# `_dimer_index_for_output`.  Task C does not change them; they are listed here
# so the equality assertion below stays exact instead of being weakened to a
# subset check.
_NOT_AM_DIMERPARAM_OWNED_MODES = frozenset(
    {
        "ap3_elst_damping__induced_dipole",
        "ap3_elst_damping__induced_dipole__disp",
        "ap3_atomMPNN",
        "disp",
    }
)


def test_full_edge_mode_set_matches_the_forwards_using_e_abfull():
    """The set must list exactly the modes whose forwards use `e_ABfull_*`.

    Every new CLIFF mode evaluates its kernels over `e_ABfull_*`, so all four
    belong here.  Omitting one would scatter full-edge energies with the
    short-range `batch.dimer_ind`, silently attributing long-range edges to the
    wrong dimer and dropping the long-range tail from every dimer total -- a
    wrong answer with no error.
    """
    assert isinstance(mtp_mtp.FULL_EDGE_DIMER_EVAL_MODES, frozenset)

    full_edge_modes = set()
    for mode in _accepted_dimer_eval_modes():
        try:
            source = _forward_source_for_mode(mode)
        except AttributeError:
            # "elst_damping_AMOEBA" maps to a method name that does not exist
            # (`_elst_damping_AMOEBA_forward` vs the defined
            # `_elst_damping_forward_AMOEBA`).  Pre-existing and out of scope.
            continue
        if "e_ABfull_" in source or "d3(batch" in source:
            full_edge_modes.add(mode)

    assert full_edge_modes - _NOT_AM_DIMERPARAM_OWNED_MODES == set(
        mtp_mtp.FULL_EDGE_DIMER_EVAL_MODES
    )
    assert not (
        mtp_mtp.FULL_EDGE_DIMER_EVAL_MODES
        & _NOT_AM_DIMERPARAM_OWNED_MODES
    )
    # And the converse direction, spelled out: no listed mode may quietly read
    # the short-range edge lists.
    for mode in mtp_mtp.FULL_EDGE_DIMER_EVAL_MODES:
        source = _forward_source_for_mode(mode)
        assert "e_ABfull_source" in source
        assert "e_ABsr_" not in source


def test_all_four_cliff_modes_are_full_edge_modes():
    assert CLIFF_DIMER_EVAL_MODES <= mtp_mtp.FULL_EDGE_DIMER_EVAL_MODES
    assert {"rackers_thole", "rackers_thole_overlap"} <= (
        mtp_mtp.FULL_EDGE_DIMER_EVAL_MODES
    )
    assert len(mtp_mtp.FULL_EDGE_DIMER_EVAL_MODES) == 6


@pytest.mark.parametrize(
    "mode,expected_index",
    [
        ("cliff_exch", "dimer_ind_full"),
        ("cliff_classical", "dimer_ind_full"),
        ("cliff_classical_overlap", "dimer_ind_full"),
        ("cliff_classical_d3", "dimer_ind_full"),
        ("elst_damping", "dimer_ind"),
    ],
)
def test_dimer_index_for_output_uses_the_full_edge_set(
    mode, expected_index, synthetic_dimer_batch
):
    harness = AM_DimerParam_Model.__new__(AM_DimerParam_Model)
    harness.dimer_eval_type = mode
    selected = harness._dimer_index_for_output(synthetic_dimer_batch)
    assert selected is getattr(synthetic_dimer_batch, expected_index)


# --------------------------------------------------------------------------
# Task C: physics routing
# --------------------------------------------------------------------------


class _ControlledCliffClassicalAtomParam(torch.nn.Module):
    """Parameter head stub whose columns are pairwise distinct tensors.

    Column ``k`` is ``0.37 * (k + 1) + 0.11 * atom_index``, offset by the call
    index so monomer A's columns are distinct from monomer B's.  A forward that
    read column 1 where it should read column 2, or monomer B's column where it
    should read monomer A's, therefore hands a kernel a tensor that compares
    unequal to the expected column.  Mirrors ``_ControlledRackersAtomParam`` in
    ``tests/test_rackers_thole_damping.py`` but with five columns and the
    per-monomer offset the exclusivity sweep needs.
    """

    def __init__(self, n_params=5):
        super().__init__()
        self.atom_model = torch.nn.Identity()
        self.batch_calls = []
        self.n_params = n_params

    def forward(self, batch):
        self.batch_calls.append(batch)
        monomer_offset = 0.017 * (len(self.batch_calls) - 1)
        atom_index = torch.arange(
            batch.x.numel(), dtype=batch.R.dtype, device=batch.x.device
        )
        charge = 0.1 + 0.05 * atom_index
        dipole = torch.stack((charge, charge + 0.1, charge + 0.2), dim=1)
        quadrupole = torch.zeros(
            (batch.x.numel(), 3, 3),
            dtype=batch.R.dtype,
            device=batch.x.device,
        )
        # Column 0 negative on purpose: the forward must take `abs` for the
        # Hirshfeld volume ratio and pass the valence width (column 1) through.
        hfvr_vw = torch.stack(
            (-(0.8 + 0.1 * atom_index), 0.4 + 0.05 * atom_index), dim=1
        )
        parameters = torch.stack(
            [
                0.37 * (column + 1) + 0.11 * atom_index + monomer_offset
                for column in range(self.n_params)
            ],
            dim=1,
        )
        return charge, dipole, quadrupole, hfvr_vw, parameters


def _record_call_kwargs(kwargs):
    """Snapshot a kernel call's kwargs, cloning tensors so a later in-place
    mutation of the originals cannot rewrite the recorded evidence."""
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
    }


@pytest.mark.parametrize(
    "mode,include_overlap,n_columns",
    [
        ("cliff_classical", False, 3),
        ("cliff_classical_overlap", True, 3),
        ("cliff_classical_d3", True, 4),
    ],
)
@pytest.mark.parametrize("elst_damping_type", ["CLIFF", "AMOEBA"])
def test_cliff_classical_forward_routes_every_column_exclusively(
    mode,
    include_overlap,
    n_columns,
    elst_damping_type,
    synthetic_dimer_batch,
    monkeypatch,
):
    """Each predicted column must reach exactly one physics term.

    Rather than only asserting the expected consumer of each column, this
    collects every per-atom tensor handed to the electrostatics, exchange, and
    induction kernels and asserts the *complete* set of consumers per column.
    A column leaking into a second term therefore fails, not just a column
    arriving at the wrong one.
    """
    electrostatic_calls = {"CLIFF": [], "AMOEBA": []}
    exchange_calls = []
    induction_calls = []

    def electrostatic_stub(damping_type):
        def evaluate(**kwargs):
            electrostatic_calls[damping_type].append(
                _record_call_kwargs(kwargs)
            )
            # Reproduce the documented in-place mutation of `mtp_elst`
            # (mtp_mtp.py:1671) so a forward that forgot to clone the charges
            # would corrupt the induction inputs asserted below.
            kwargs["qA_0"].add_(100.0)
            kwargs["qB_0"].sub_(100.0)
            return torch.ones_like(
                kwargs["e_AB_source"], dtype=kwargs["RA"].dtype
            )

        return evaluate

    def exchange_stub(**kwargs):
        exchange_calls.append(_record_call_kwargs(kwargs))
        return torch.full_like(
            kwargs["e_AB_source"], 3.0, dtype=kwargs["RA"].dtype
        )

    def induction_stub(**kwargs):
        induction_calls.append(_record_call_kwargs(kwargs))
        return torch.full_like(
            kwargs["e_AB_source"], 2.0, dtype=kwargs["RA"].dtype
        )

    monkeypatch.setattr(
        mtp_mtp, "mtp_elst_damping", electrostatic_stub("CLIFF")
    )
    monkeypatch.setattr(
        mtp_mtp, "mtp_elst_damping_AMOEBA", electrostatic_stub("AMOEBA")
    )
    monkeypatch.setattr(mtp_mtp, "cliff_exchange", exchange_stub)
    monkeypatch.setattr(mtp_mtp, "rackers_thole_induction", induction_stub)

    atom_parameters = _ControlledCliffClassicalAtomParam()
    dimer = mtp_mtp.DimerProp(
        ATParam=atom_parameters,
        dimer_eval=mode,
        elst_damping_type=elst_damping_type,
    )
    edge_energy, output_A, output_B = dimer(synthetic_dimer_batch)

    # The parameter model is evaluated once per monomer.
    assert len(atom_parameters.batch_calls) == 2
    assert (
        sum(
            call is synthetic_dimer_batch.batch_atomic_A
            for call in atom_parameters.batch_calls
        )
        == 1
    )
    assert (
        sum(
            call is synthetic_dimer_batch.batch_atomic_B
            for call in atom_parameters.batch_calls
        )
        == 1
    )

    other = "AMOEBA" if elst_damping_type == "CLIFF" else "CLIFF"
    assert len(electrostatic_calls[elst_damping_type]) == 1
    assert electrostatic_calls[other] == []
    assert len(exchange_calls) == 1
    assert len(induction_calls) == 1
    electrostatic = electrostatic_calls[elst_damping_type][0]
    exchange = exchange_calls[0]
    induction = induction_calls[0]

    labelled = {
        f"elst:{key}": value for key, value in electrostatic.items()
    }
    labelled.update(
        {f"exch:{key}": value for key, value in exchange.items()}
    )
    labelled.update(
        {f"indu:{key}": value for key, value in induction.items()}
    )

    expected_consumers = {
        "A": {
            CLIFF_CLASSICAL_ELST_INDEX: {"elst:Ka"},
            CLIFF_CLASSICAL_THOLE_DIRECT_INDEX: {"indu:thole_direct_A"},
            CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX: {"indu:thole_mutual_A"},
            CLIFF_CLASSICAL_IND_OVERLAP_INDEX: {"indu:ind_overlap_A"},
            CLIFF_CLASSICAL_EXCH_INDEX: {"exch:K_exch_A"},
        },
        "B": {
            CLIFF_CLASSICAL_ELST_INDEX: {"elst:Kb"},
            CLIFF_CLASSICAL_THOLE_DIRECT_INDEX: {"indu:thole_direct_B"},
            CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX: {"indu:thole_mutual_B"},
            CLIFF_CLASSICAL_IND_OVERLAP_INDEX: {"indu:ind_overlap_B"},
            CLIFF_CLASSICAL_EXCH_INDEX: {"exch:K_exch_B"},
        },
    }
    for monomer, output in (("A", output_A), ("B", output_B)):
        parameters = output[-1]
        assert parameters.shape == (output[0].numel(), 5)
        # Columns must actually be distinguishable for the exclusivity check
        # below to mean anything.
        for left in range(5):
            for right in range(left + 1, 5):
                assert not torch.equal(
                    parameters[:, left], parameters[:, right]
                )
        for column, expected_keys in expected_consumers[monomer].items():
            found = {
                key
                for key, value in labelled.items()
                if isinstance(value, torch.Tensor)
                and value.dim() == 1
                and value.shape == parameters[:, column].shape
                and torch.equal(value, parameters[:, column])
            }
            assert found == expected_keys, (
                monomer,
                column,
                sorted(found),
                sorted(expected_keys),
            )

    # Nested HFVR / valence-width contract.
    assert torch.equal(
        induction["hirshfeld_volume_ratio_A"], output_A[-2][:, 0].abs()
    )
    assert torch.equal(
        induction["hirshfeld_volume_ratio_B"], output_B[-2][:, 0].abs()
    )
    for call in (exchange, induction):
        assert torch.equal(call["valence_widths_A"], output_A[-2][:, 1])
        assert torch.equal(call["valence_widths_B"], output_B[-2][:, 1])

    # Column 3 only ever reaches the overlap route's *energy*; the kwarg is
    # always forwarded, and `include_overlap` is what gates it.
    assert induction["include_overlap"] is include_overlap

    # Electrostatics ran on cloned charges, so induction saw the originals.
    assert torch.equal(induction["qA"], output_A[0])
    assert torch.equal(induction["qB"], output_B[0])
    assert torch.equal(induction["qA"], electrostatic["qA_0"])
    assert torch.equal(induction["qB"], electrostatic["qB_0"])

    # Exchange and electrostatics consume the same full AB edge domain.
    for call in (electrostatic, exchange, induction):
        assert torch.equal(
            call["e_AB_source"], synthetic_dimer_batch.e_ABfull_source
        )
        assert torch.equal(
            call["e_AB_target"], synthetic_dimer_batch.e_ABfull_target
        )
    assert torch.equal(
        exchange["e_AB_source"], electrostatic["e_AB_source"]
    )
    assert torch.equal(
        exchange["e_AB_target"], electrostatic["e_AB_target"]
    )

    # Column order is (Elst, Exch, Indu[, Disp]).
    n_edges = synthetic_dimer_batch.e_ABfull_source.numel()
    assert edge_energy.shape == (n_edges, n_columns)
    assert torch.equal(edge_energy[:, 0], torch.ones(n_edges))
    assert torch.equal(edge_energy[:, 1], torch.full((n_edges,), 3.0))
    assert torch.equal(edge_energy[:, 2], torch.full((n_edges,), 2.0))
    if n_columns == 4:
        assert torch.isfinite(edge_energy[:, 3]).all()
        assert not torch.equal(
            edge_energy[:, 3], torch.zeros(n_edges)
        )


def test_cliff_classical_forward_rejects_unknown_elst_damping(
    synthetic_dimer_batch,
):
    dimer = mtp_mtp.DimerProp(
        ATParam=_ControlledCliffClassicalAtomParam(),
        dimer_eval="cliff_classical",
        elst_damping_type="unsupported",
    )
    with pytest.raises(ValueError, match="Unsupported elst_damping_type"):
        dimer(synthetic_dimer_batch)


def test_cliff_exch_needs_no_polarizability_and_skips_induction(
    synthetic_dimer_batch, monkeypatch
):
    """`cliff_exch` is exchange-only: no polarizability table, no induction."""

    def forbidden(name):
        def call(*args, **kwargs):
            raise AssertionError(
                f"cliff_exch must not call {name}"
            )

        return call

    for name in (
        "rackers_thole_induction",
        "mtp_elst_damping",
        "mtp_elst_damping_AMOEBA",
        "induced_dipole_induction",
        "induced_dipole_induction_optimized",
    ):
        monkeypatch.setattr(mtp_mtp, name, forbidden(name))

    exchange_calls = []
    original_exchange = mtp_mtp.cliff_exchange

    def record_exchange(**kwargs):
        exchange_calls.append(_record_call_kwargs(kwargs))
        return original_exchange(**kwargs)

    monkeypatch.setattr(mtp_mtp, "cliff_exchange", record_exchange)

    atom_parameters = _ControlledCliffClassicalAtomParam(n_params=1)
    dimer = mtp_mtp.DimerProp(
        ATParam=atom_parameters, dimer_eval="cliff_exch"
    )
    assert not hasattr(dimer, "polarizability_table")

    edge_energy, output_A, output_B = dimer(synthetic_dimer_batch)

    n_edges = synthetic_dimer_batch.e_ABfull_source.numel()
    assert edge_energy.shape == (n_edges,)
    assert torch.isfinite(edge_energy).all()
    assert torch.all(edge_energy > 0)

    assert len(exchange_calls) == 1
    exchange = exchange_calls[0]
    assert torch.equal(
        exchange["K_exch_A"], output_A[-1][:, CLIFF_EXCH_INDEX]
    )
    assert torch.equal(
        exchange["K_exch_B"], output_B[-1][:, CLIFF_EXCH_INDEX]
    )
    assert torch.equal(
        exchange["e_AB_source"], synthetic_dimer_batch.e_ABfull_source
    )
    assert torch.equal(
        exchange["e_AB_target"], synthetic_dimer_batch.e_ABfull_target
    )


@pytest.mark.parametrize(
    "mode,expected_distance_calls",
    [
        # `_cliff_exch_forward` leaves the single reduction to
        # `cliff_exchange`; the classical forwards do it themselves and hand
        # the result down, so exchange adds no second call.
        ("cliff_exch", 1),
        ("cliff_classical", 1),
        ("cliff_classical_overlap", 1),
    ],
)
def test_intermolecular_distances_are_computed_once_per_forward(
    mode, expected_distance_calls, synthetic_dimer_batch, monkeypatch
):
    """Stub the kernels that own their own reductions, then count.

    `mtp_elst_damping` and `rackers_thole_induction` each call
    `get_distances` internally on their own edge domains, so they are replaced
    with stubs here; what remains is the forward's own reduction plus anything
    `cliff_exchange` adds.
    """
    original_get_distances = mtp_mtp.get_distances
    distance_calls = []

    def counting_get_distances(RA, RB, e_source, e_target):
        distance_calls.append(e_source)
        return original_get_distances(RA, RB, e_source, e_target)

    monkeypatch.setattr(mtp_mtp, "get_distances", counting_get_distances)
    monkeypatch.setattr(
        mtp_mtp,
        "mtp_elst_damping",
        lambda **kwargs: torch.zeros_like(
            kwargs["e_AB_source"], dtype=kwargs["RA"].dtype
        ),
    )
    monkeypatch.setattr(
        mtp_mtp,
        "rackers_thole_induction",
        lambda **kwargs: torch.zeros_like(
            kwargs["e_AB_source"], dtype=kwargs["RA"].dtype
        ),
    )

    n_params = 1 if mode == "cliff_exch" else 5
    dimer = mtp_mtp.DimerProp(
        ATParam=_ControlledCliffClassicalAtomParam(n_params=n_params),
        dimer_eval=mode,
    )
    dimer(synthetic_dimer_batch)

    assert len(distance_calls) == expected_distance_calls
    for e_source in distance_calls:
        assert torch.equal(
            e_source, synthetic_dimer_batch.e_ABfull_source
        )


def test_dimer_prop_forwards_configured_scf_controls(
    synthetic_dimer_batch, monkeypatch
):
    calls = []
    original = mtp_mtp.rackers_thole_induction

    def record_induction(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(mtp_mtp, "rackers_thole_induction", record_induction)
    dimer = mtp_mtp.DimerProp(
        ATParam=_ControlledCliffClassicalAtomParam(),
        dimer_eval="cliff_classical_overlap",
        induction_convergence_threshold=1e-6,
        induction_max_iterations=50,
    )
    dimer(synthetic_dimer_batch)
    assert len(calls) == 1
    assert calls[0]["convergence_threshold"] == 1e-6
    assert calls[0]["max_iterations"] == 50


def test_shared_distances_are_converted_to_bohr_for_exchange(
    synthetic_dimer_batch, monkeypatch
):
    """The shared-distance path must equal the self-computed-distance path.

    `get_distances` returns ANGSTROM while `atomic_overlap_S_ij` needs BOHR, so
    a shared distance handed to `cliff_exchange` without `/ constants.au2ang`
    would produce a wrong-but-finite exchange energy.  This asserts the two
    paths agree, and that the un-converted distance really would differ.
    """
    exchange_calls = []
    original_exchange = mtp_mtp.cliff_exchange

    def record_exchange(**kwargs):
        exchange_calls.append(_record_call_kwargs(kwargs))
        return original_exchange(**kwargs)

    monkeypatch.setattr(mtp_mtp, "cliff_exchange", record_exchange)

    dimer = mtp_mtp.DimerProp(
        ATParam=_ControlledCliffClassicalAtomParam(),
        dimer_eval="cliff_classical",
    )
    edge_energy, output_A, output_B = dimer(synthetic_dimer_batch)

    assert len(exchange_calls) == 1
    shared = exchange_calls[0]["dR_AB"]
    assert shared is not None

    dR_ang, _ = mtp_mtp.get_distances(
        synthetic_dimer_batch.RA,
        synthetic_dimer_batch.RB,
        synthetic_dimer_batch.e_ABfull_source,
        synthetic_dimer_batch.e_ABfull_target,
    )
    assert torch.allclose(shared, dR_ang / constants.au2ang)
    assert not torch.allclose(shared, dR_ang)

    kernel_inputs = dict(
        RA=synthetic_dimer_batch.RA,
        RB=synthetic_dimer_batch.RB,
        e_AB_source=synthetic_dimer_batch.e_ABfull_source,
        e_AB_target=synthetic_dimer_batch.e_ABfull_target,
        valence_widths_A=output_A[-2][:, 1],
        valence_widths_B=output_B[-2][:, 1],
        K_exch_A=output_A[-1][:, CLIFF_CLASSICAL_EXCH_INDEX],
        K_exch_B=output_B[-1][:, CLIFF_CLASSICAL_EXCH_INDEX],
    )
    self_computed = original_exchange(**kernel_inputs, dR_AB=None)
    with_shared = original_exchange(**kernel_inputs, dR_AB=shared)
    assert torch.allclose(self_computed, with_shared)
    # The forward's exchange column is exactly the self-computed-distance
    # answer, so the conversion in the shared path is what ships.
    assert torch.allclose(
        edge_energy[:, 1], self_computed, rtol=1e-6, atol=1e-12
    )

    # Sanity: forgetting the conversion is not a harmless no-op.
    un_converted = original_exchange(**kernel_inputs, dR_AB=dR_ang)
    assert not torch.allclose(un_converted, self_computed)


# --------------------------------------------------------------------------
# Task C: joint forward, scatter aggregation, gradients
# --------------------------------------------------------------------------


def _cliff_dimer_model(mode, nested_model, seed=0):
    head_type = (
        CliffExchangeNN if mode == "cliff_exch" else CliffClassicalNN
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = head_type(
            atom_model=copy.deepcopy(nested_model),
            n_message=1,
            n_neuron=8,
            n_embed=4,
            freeze_atom_model=True,
        )
    dimer = mtp_mtp.DimerProp(
        ATParam=model, dimer_eval=mode, freeze_atom_model=True
    )
    return model, dimer


@pytest.mark.parametrize(
    "mode,n_columns",
    [
        ("cliff_exch", None),
        ("cliff_classical", 3),
        ("cliff_classical_overlap", 3),
        ("cliff_classical_d3", 4),
    ],
)
def test_cliff_forward_shapes_and_scatter_aggregation(
    mode, n_columns, nested_hfvr_vw_model, synthetic_dimer_batch
):
    _, dimer = _cliff_dimer_model(mode, nested_hfvr_vw_model)
    edge_energy, _, _ = dimer(synthetic_dimer_batch)

    n_edges = synthetic_dimer_batch.e_ABfull_source.numel()
    expected_shape = (
        (n_edges,) if n_columns is None else (n_edges, n_columns)
    )
    assert edge_energy.shape == expected_shape
    assert torch.isfinite(edge_energy).all()

    harness = AM_DimerParam_Model.__new__(AM_DimerParam_Model)
    harness.dimer_eval_type = mode
    dimer_index = harness._dimer_index_for_output(synthetic_dimer_batch)
    assert dimer_index is synthetic_dimer_batch.dimer_ind_full
    assert dimer_index.numel() == n_edges

    batch_size = synthetic_dimer_batch.total_charge_A.size(0)
    dimer_energy = scatter_sum_compile(
        edge_energy, dimer_index, dim_size=batch_size
    )
    assert dimer_energy.shape == (
        (batch_size,) if n_columns is None else (batch_size, n_columns)
    )
    assert torch.isfinite(dimer_energy).all()


@pytest.mark.parametrize(
    "mode,n_heads,expected_active_heads",
    [
        # `cliff_exch` has a single head and it must be active.
        ("cliff_exch", 1, {CLIFF_EXCH_INDEX}),
        # No overlap route: the `ind_overlap` column (3) is forwarded but
        # contributes no energy, so four of the five heads carry gradient.
        (
            "cliff_classical",
            5,
            {
                CLIFF_CLASSICAL_ELST_INDEX,
                CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
                CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
                CLIFF_CLASSICAL_EXCH_INDEX,
            },
        ),
        # Overlap route: all five heads are energy-active.
        (
            "cliff_classical_overlap",
            5,
            {
                CLIFF_CLASSICAL_ELST_INDEX,
                CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
                CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
                CLIFF_CLASSICAL_IND_OVERLAP_INDEX,
                CLIFF_CLASSICAL_EXCH_INDEX,
            },
        ),
        (
            "cliff_classical_d3",
            5,
            {
                CLIFF_CLASSICAL_ELST_INDEX,
                CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
                CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
                CLIFF_CLASSICAL_IND_OVERLAP_INDEX,
                CLIFF_CLASSICAL_EXCH_INDEX,
            },
        ),
    ],
)
def test_cliff_joint_forward_gradients_and_positive_step(
    mode,
    n_heads,
    expected_active_heads,
    nested_hfvr_vw_model,
    synthetic_dimer_batch,
):
    """Backward must reach exactly the energy-active heads.

    Gradient activity is asserted at each raw guess-embedding output rather
    than at every readout parameter: a randomly initialized ReLU readout may be
    entirely dead and legitimately carry zero gradient, which would make a
    per-parameter assertion flaky rather than meaningful.
    """
    model, dimer = _cliff_dimer_model(mode, nested_hfvr_vw_model)
    assert len(model.guess_layer) == n_heads

    guess_outputs = [[] for _ in model.guess_layer]
    hook_handles = []
    for index, guess_layer in enumerate(model.guess_layer):

        def capture_guess_output(module, inputs, output, head_index=index):
            output.retain_grad()
            guess_outputs[head_index].append(output)

        hook_handles.append(
            guess_layer.register_forward_hook(capture_guess_output)
        )

    edge_energy, _, _ = dimer(synthetic_dimer_batch)
    assert torch.isfinite(edge_energy).all()

    dimer_energy = scatter_sum_compile(
        edge_energy,
        synthetic_dimer_batch.dimer_ind_full,
        dim_size=synthetic_dimer_batch.total_charge_A.size(0),
    )
    dimer_energy.square().mean().backward()
    for handle in hook_handles:
        handle.remove()

    active_heads = set()
    for index, outputs in enumerate(guess_outputs):
        assert len(outputs) == 2  # one monomer A call, one monomer B call
        assert all(output.grad is not None for output in outputs)
        assert all(
            torch.isfinite(output.grad).all() for output in outputs
        )
        if any(torch.count_nonzero(output.grad) > 0 for output in outputs):
            active_heads.add(index)
    assert active_heads == expected_active_heads
    assert len(active_heads) == len(expected_active_heads)

    for head in model.param_readout_layers:
        for readout in head:
            for parameter in readout.parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    updated_parameters = model(synthetic_dimer_batch.batch_atomic_A)[-1]
    assert updated_parameters.shape == (
        synthetic_dimer_batch.batch_atomic_A.x.numel(),
        n_heads,
    )
    assert torch.isfinite(updated_parameters).all()
    assert torch.all(updated_parameters > 0)


# ---------------------------------------------------------------------------
# Task D: training harnesses
# ---------------------------------------------------------------------------

CLIFF_HARNESSES = (
    (CliffExchangeModel, "cliff_exch", "CliffExchangeNN",
     CLIFF_EXCH_PARAMETER_NAMES, CLIFF_EXCH_INITIAL_VALUES,
     CLIFF_EXCH_INITIAL_STDS),
    (CliffClassicalModel, "cliff_classical", "CliffClassicalNN",
     CLIFF_CLASSICAL_PARAMETER_NAMES, CLIFF_CLASSICAL_INITIAL_VALUES,
     CLIFF_CLASSICAL_INITIAL_STDS),
    (CliffClassicalOverlapModel, "cliff_classical_overlap",
     "CliffClassicalNN", CLIFF_CLASSICAL_PARAMETER_NAMES,
     CLIFF_CLASSICAL_INITIAL_VALUES, CLIFF_CLASSICAL_INITIAL_STDS),
)

CLIFF_HARNESS_IDS = [entry[1] for entry in CLIFF_HARNESSES]


def _build_cliff_harness(harness_type, nested_model, **kwargs):
    """Construct a CLIFF harness with no dataset and no GPU.

    ``ignore_database_null=True`` is the same escape hatch the Rackers harness
    tests use: it keeps ``AM_DimerParam_Model.__init__`` from building an
    on-disk pairwise dataset, so nothing here touches real data.
    """
    options = dict(
        atom_model=copy.deepcopy(nested_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    options.update(kwargs)
    return harness_type(**options)


@pytest.mark.parametrize(
    "harness_type,expected_mode,expected_model_type,parameter_names,"
    "initial_values,initial_stds",
    CLIFF_HARNESSES,
    ids=CLIFF_HARNESS_IDS,
)
def test_cliff_harness_contract(
    harness_type,
    expected_mode,
    expected_model_type,
    parameter_names,
    initial_values,
    initial_stds,
    nested_hfvr_vw_model,
):
    harness = _build_cliff_harness(harness_type, nested_hfvr_vw_model)

    assert type(harness.model).__name__ == expected_model_type
    assert harness.dimer_eval_type == expected_mode
    assert harness_type.DIMER_EVAL == expected_mode
    assert harness_type.MODEL_TYPE == expected_model_type
    # Ordering, not just membership: a reordered contract silently reassigns
    # physical meaning to every column.
    assert harness_type.PARAMETER_NAMES == parameter_names
    assert list(harness.model.get_config()["parameter_names"]) == list(
        parameter_names
    )
    assert harness.n_params == len(parameter_names)
    assert harness.model.n_params == len(parameter_names)
    assert harness.model.param_start_mean == list(initial_values)
    assert harness.model.param_start_std == list(initial_stds)
    assert harness.model.width_floor == OVERLAP_WIDTH_FLOOR
    assert harness.width_floor == OVERLAP_WIDTH_FLOOR
    assert all(
        not parameter.requires_grad
        for parameter in harness.model.atom_model.parameters()
    )


@pytest.mark.parametrize(
    "harness_type", [entry[0] for entry in CLIFF_HARNESSES],
    ids=CLIFF_HARNESS_IDS,
)
@pytest.mark.parametrize(
    "forbidden", ["n_params", "model_type", "dimer_eval_type"]
)
def test_cliff_harness_hides_fixed_configuration(
    harness_type, forbidden, nested_hfvr_vw_model
):
    """None of the fixed configuration is a public constructor argument."""
    parameters = inspect.signature(harness_type.__init__).parameters
    assert forbidden not in parameters
    # `**dataset_kwargs` would otherwise swallow it and forward a duplicate.
    with pytest.raises(TypeError, match=forbidden):
        _build_cliff_harness(
            harness_type, nested_hfvr_vw_model, **{forbidden: "ignored"}
        )


@pytest.mark.parametrize(
    "harness_type,parameter_names",
    [(entry[0], entry[3]) for entry in CLIFF_HARNESSES],
    ids=CLIFF_HARNESS_IDS,
)
@pytest.mark.parametrize("field", ["param_start_mean", "param_start_std"])
def test_cliff_harness_initialization_length_errors(
    harness_type, parameter_names, field, nested_hfvr_vw_model
):
    """Initialization routes through ``_validate_positive_initialization``.

    Asserted through the reported count word, which the shared validator
    derives from ``len(parameter_names)``.
    """
    wrong_length = [0.5] * (len(parameter_names) + 1)
    # The validator reports the *expected* count, derived from the harness's
    # own PARAMETER_NAMES, so one-parameter and five-parameter routes cannot
    # accidentally share the four-value Rackers message.
    expected = "exactly one value" if len(parameter_names) == 1 else (
        "exactly five values"
    )
    with pytest.raises(ValueError, match=expected):
        _build_cliff_harness(
            harness_type, nested_hfvr_vw_model, **{field: wrong_length}
        )


def test_positive_parameter_contracts_cover_every_head():
    assert POSITIVE_PARAMETER_CONTRACTS == {
        "RackersTholeDampingNN": RACKERS_PARAMETER_NAMES,
        "CliffExchangeNN": CLIFF_EXCH_PARAMETER_NAMES,
        "CliffClassicalNN": CLIFF_CLASSICAL_PARAMETER_NAMES,
        # Same five-parameter contract as CliffClassicalNN, different
        # featurizer; the contract is what the physics reads, so it must match
        # exactly rather than merely have the same length.
        "CliffClassicalMPNN": CLIFF_CLASSICAL_PARAMETER_NAMES,
    }


def test_combined_cliff_modes_are_the_trainable_multi_component_routes():
    assert COMBINED_CLIFF_DIMER_EVAL_MODES == frozenset(
        {"cliff_classical", "cliff_classical_overlap"}
    )
    # `cliff_classical_d3` is inference-only and `cliff_exch` has a single
    # component, so neither may carry a total/component split.
    assert "cliff_classical_d3" not in COMBINED_CLIFF_DIMER_EVAL_MODES
    assert "cliff_exch" not in COMBINED_CLIFF_DIMER_EVAL_MODES


# ---------------------------------------------------------------------------
# Task D: y_ind / term dispatch
# ---------------------------------------------------------------------------


def test_mae_report_header_is_width_agnostic():
    """The header is generated, so widening the selection is data, not code.

    The first two expectations are the literals the pre-Task-D dispatch used
    for its one- and two-column selections; keeping them byte-identical is what
    proves the generated form did not change any existing report.
    """
    assert _mae_report_header(("Elst",)) == "Elst"
    assert _mae_report_header(("Elst", "Ind")) == "Elst      Ind"
    assert (
        _mae_report_header(("Elst", "Exch", "Ind")) == "Elst      Exch      Ind"
    )
    # Any width, without a per-width branch.
    assert _mae_report_header(()) == ""
    assert _mae_report_header(("A", "B", "C", "D")) == (
        "A" + " " * 9 + "B" + " " * 9 + "C" + " " * 9 + "D"
    )


@pytest.mark.parametrize(
    "harness_type,expected_y_ind,expected_term",
    [
        (CliffExchangeModel, 1, "Exch"),
        (
            CliffClassicalModel,
            torch.tensor([0, 1, 2]),
            "Elst      Exch      Ind",
        ),
        (
            CliffClassicalOverlapModel,
            torch.tensor([0, 1, 2]),
            "Elst      Exch      Ind",
        ),
    ],
    ids=CLIFF_HARNESS_IDS,
)
def test_cliff_training_dispatch_selects_targets(
    harness_type,
    expected_y_ind,
    expected_term,
    nested_hfvr_vw_model,
    synthetic_dimer_batch,
    monkeypatch,
    capsys,
    tmp_path,
):
    """Drive one real epoch and capture the selected target columns.

    ``cliff_exch`` selects SAPT ``Exch`` (scalar column ``1``) and the classical
    routes select ``[0, 1, 2]``.  Running the loop rather than reading the
    dispatch also proves ``cliff_exch`` needs no polarizability table: the mode
    never creates one, so any induction device move in that branch would raise
    ``AttributeError`` here.
    """
    harness = _build_cliff_harness(harness_type, nested_hfvr_vw_model)
    if harness_type is CliffExchangeModel:
        assert not hasattr(harness.dimer_model, "polarizability_table")
    harness.example_input = lambda: synthetic_dimer_batch.batch_atomic_A
    harness.compile_model = lambda: None

    selected = []
    original = harness._AM_DimerParam_Model__train_batches_single_proc

    def record(*args, **kwargs):
        selected.append(kwargs["y_ind"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        harness, "_AM_DimerParam_Model__train_batches_single_proc", record
    )
    harness.model_save_path = tmp_path / f"{harness_type.DIMER_EVAL}.pt"
    harness.single_proc_train(
        train_dataset=[_make_collate_item(1.0)],
        test_dataset=[_make_collate_item(1.1)],
        n_epochs=1,
        batch_size=1,
        lr=1e-5,
        pin_memory=False,
        num_workers=0,
    )

    assert len(selected) == 1
    if isinstance(expected_y_ind, torch.Tensor):
        assert isinstance(selected[0], torch.Tensor)
        assert torch.equal(selected[0], expected_y_ind)
    else:
        # A plain int, not a one-element tensor: the scalar path reports a
        # single MAE column rather than a per-column row.
        assert not isinstance(selected[0], torch.Tensor)
        assert selected[0] == expected_y_ind
    header = f"                                       {expected_term}"
    assert header in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Task D: component/total loss weighting
# ---------------------------------------------------------------------------


def _legacy_batch_loss(preds, ref, comp_errors, loss_fn):
    """The pre-Task-D per-batch loss expression, copied verbatim.

    Kept as an independent reference so ``component_gamma == 1.0`` is checked
    against the *old* code path rather than against a restatement of the new
    one.
    """
    return (
        torch.mean(torch.square(comp_errors))
        if (loss_fn is None)
        else loss_fn(preds, ref)
    )


def _loss_probe_harness(
    dimer_eval_type="cliff_classical",
    component_gamma=None,
    total_includes_d3=False,
    d3_damping_parameters=None,
):
    """A bare harness carrying only what ``_batch_loss`` reads."""
    harness = AM_DimerParam_Model.__new__(AM_DimerParam_Model)
    harness.dimer_eval_type = dimer_eval_type
    harness.component_gamma = component_gamma
    harness.total_includes_d3 = total_includes_d3
    harness.dimer_model = types.SimpleNamespace(
        d3_damping_parameters=d3_damping_parameters
    )
    return harness


@pytest.mark.parametrize("loss_fn_factory", [lambda: None, torch.nn.MSELoss])
@pytest.mark.parametrize("n_columns", [1, 2, 3, 4])
def test_component_gamma_none_reproduces_plain_mse_bitwise(
    loss_fn_factory, n_columns
):
    """``component_gamma is None`` is the legacy loss, bitwise.

    The sentinel is deliberately *not* ``1.0``: under CLIFF Eq. (23),
    ``gamma == 1.0`` means ``sum_C MSE(E_C)``, which is ``k`` times the plain
    mean over ``k`` columns.  Overloading one value to mean both would put a
    factor-of-``k`` step at the endpoint of the Fig. 3 gamma sweep.
    """
    loss_fn = loss_fn_factory()
    generator = torch.Generator().manual_seed(11)
    shape = (7,) if n_columns == 1 else (7, n_columns)
    preds = torch.randn(shape, generator=generator, dtype=torch.float64)
    ref = torch.randn(shape, generator=generator, dtype=torch.float64)
    comp_errors = preds - ref

    harness = _loss_probe_harness(component_gamma=None)
    actual = harness._batch_loss(preds, ref, comp_errors, None, loss_fn)
    expected = _legacy_batch_loss(preds, ref, comp_errors, loss_fn)
    assert torch.equal(actual, expected)


def test_default_harness_component_gamma_is_the_neutral_default(
    nested_hfvr_vw_model,
):
    harness = _build_cliff_harness(CliffClassicalModel, nested_hfvr_vw_model)
    assert harness.component_gamma is None
    assert harness.total_includes_d3 is False


def test_component_loss_weighting_defaults_without_attributes():
    """A ``__new__``-constructed harness keeps the historical behavior."""
    harness = AM_DimerParam_Model.__new__(AM_DimerParam_Model)
    assert harness._component_loss_weighting() == (None, False)


class _FullEdgeCliffTrainModel(torch.nn.Module):
    """Three-column per-edge stub standing in for a CLIFF classical forward."""

    def __init__(self, n_columns=3):
        super().__init__()
        self.n_columns = n_columns
        self.scale = torch.nn.Parameter(torch.tensor(1.25))

    def forward(self, batch):
        base = torch.arange(
            1, batch.dimer_ind_full.numel() * self.n_columns + 1,
            dtype=self.scale.dtype,
        ).reshape(batch.dimer_ind_full.numel(), self.n_columns)
        return (self.scale * base,)


def test_training_loop_component_gamma_none_matches_legacy_path_bitwise(
    synthetic_dimer_batch,
):
    """Same assertion, but through the real training/eval loops.

    The second run replaces ``_batch_loss`` with the verbatim pre-Task-D
    expression, so the comparison is against the old code path executing inside
    the same loop, not against a hand-written formula.
    """
    y_ind = torch.tensor([0, 1, 2])

    def run(use_legacy):
        torch.manual_seed(3)
        harness = AM_DimerParam_Model.__new__(AM_DimerParam_Model)
        harness.dimer_eval_type = "cliff_classical"
        harness.component_gamma = None
        harness.total_includes_d3 = False
        harness.model = _FullEdgeCliffTrainModel()
        harness.dimer_model = harness.model
        if use_legacy:
            harness._batch_loss = (
                lambda preds, ref, comp_errors, batch, loss_fn:
                _legacy_batch_loss(preds, ref, comp_errors, loss_fn)
            )
        optimizer = torch.optim.SGD(harness.model.parameters(), lr=0.01)
        train = harness._AM_DimerParam_Model__train_batches_single_proc(
            [synthetic_dimer_batch],
            loss_fn=torch.nn.MSELoss(),
            optimizer=optimizer,
            rank_device=torch.device("cpu"),
            scheduler=None,
            y_ind=y_ind,
        )
        evaluate = harness._AM_DimerParam_Model__evaluate_batches_single_proc(
            [synthetic_dimer_batch],
            loss_fn=torch.nn.MSELoss(),
            rank_device=torch.device("cpu"),
            y_ind=y_ind,
        )
        return train, evaluate

    (new_train, new_eval) = run(use_legacy=False)
    (old_train, old_eval) = run(use_legacy=True)

    assert new_train[0] == old_train[0]
    assert torch.equal(new_train[1], old_train[1])
    assert new_eval[0] == old_eval[0]
    assert torch.equal(new_eval[1], old_eval[1])
    # Three columns reported, not a hardcoded two.
    assert new_train[1].shape == (3,)


def test_component_gamma_zero_depends_only_on_the_summed_total():
    ref = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    preds = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    # Same row sums, completely different column split.
    reshuffled = torch.tensor([[3.0, 1.0, 2.0], [0.5, 0.5, -1.0]])
    assert torch.allclose(preds.sum(-1), reshuffled.sum(-1))

    harness = _loss_probe_harness(component_gamma=0.0)
    first = harness._batch_loss(preds, ref, preds - ref, None, None)
    second = harness._batch_loss(
        reshuffled, ref, reshuffled - ref, None, None
    )
    assert torch.equal(first, second)

    # (1 - 0) * MSE(total): total error is [4.0, -1.0].
    assert first.item() == pytest.approx((16.0 + 1.0) / 2.0)

    # ... and it is genuinely different from the legacy plain MSE.
    plain = _loss_probe_harness(component_gamma=None)._batch_loss(
        preds, ref, preds - ref, None, None
    )
    assert not torch.allclose(first, plain)


def test_component_gamma_matches_hand_computed_value():
    """CLIFF's fitted gamma = 0.4, computed by hand.

    errors        = [[1, 1, 2], [-1, -1, 1]]
    per-column MSE = 1.0, 1.0, 2.5  -> sum_C MSE(E_C) = 4.5
    predicted total = [6, 0]; reference total = [2, 1]
    total error    = [4, -1] -> MSE(E_total) = 8.5
    L = 0.6 * 8.5 + 0.4 * 4.5 = 5.1 + 1.8 = 6.9
    """
    ref = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    preds = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    harness = _loss_probe_harness(component_gamma=0.4)
    loss = harness._batch_loss(preds, ref, preds - ref, None, None)
    assert loss.item() == pytest.approx(6.9, rel=1e-6)


def test_cliff_component_loss_is_continuous_across_the_gamma_sweep():
    """CLIFF Fig. 3 sweeps gamma over [0, 1]; the loss must not step.

    This is the regression guard for the earlier design in which
    ``component_gamma = 1.0`` doubled as the legacy plain-MSE default: with
    ``k`` columns, ``sum_C MSE(E_C) == k * mean MSE``, so the endpoint jumped by
    a factor of ``k`` and the effective learning rate jumped with it.  The
    legacy loss now has its own sentinel (``None``), leaving the Eq. (23)
    family continuous.
    """
    ref = torch.tensor(
        [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    preds = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=torch.float64
    )

    def loss_at(gamma):
        harness = _loss_probe_harness(component_gamma=gamma)
        return harness._batch_loss(
            preds, ref, preds - ref, None, None
        ).item()

    # Continuity at the endpoint that used to jump.  L is affine in gamma with
    # slope (component_term - total_term), so a 1e-4 step may move it by at
    # most 1e-4 * |slope|; the old design moved it by a factor of k instead.
    slope = abs(loss_at(1.0) - loss_at(0.0))
    assert abs(loss_at(1.0) - loss_at(0.9999)) <= 1e-4 * slope * 1.001
    assert abs(loss_at(0.0) - loss_at(0.0001)) <= 1e-4 * slope * 1.001
    assert loss_at(1.0) == pytest.approx(loss_at(0.9999), rel=1e-3)
    assert loss_at(0.0) == pytest.approx(loss_at(0.0001), rel=1e-3)

    # gamma = 1.0 is exactly the unnormalized component sum, with zero weight
    # on the total -- the honest Eq. (23) endpoint.
    component_sum = torch.square(preds - ref).mean(dim=0).sum()
    assert loss_at(1.0) == pytest.approx(component_sum.item(), rel=1e-12)

    # ... which is k times the legacy plain MSE, hence a different functional.
    legacy = _loss_probe_harness(component_gamma=None)._batch_loss(
        preds, ref, preds - ref, None, None
    )
    assert loss_at(1.0) == pytest.approx(
        preds.shape[1] * legacy.item(), rel=1e-12
    )

    # Monotone and smooth across the whole sweep: no step anywhere.
    sweep = [round(0.05 * i, 2) for i in range(21)]
    values = [loss_at(gamma) for gamma in sweep]
    steps = [b - a for a, b in zip(values, values[1:])]
    assert max(steps) - min(steps) < 1e-9  # linear in gamma, so equal steps


def test_component_gamma_is_linear_in_gamma():
    """L is affine in gamma, so two anchors determine the whole sweep."""
    ref = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    preds = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

    def loss_at(gamma):
        return _loss_probe_harness(component_gamma=gamma)._batch_loss(
            preds, ref, preds - ref, None, None
        ).item()

    total_only, component_only = loss_at(0.0), loss_at(1.0)
    for gamma in (0.1, 0.4, 0.75):
        expected = (1.0 - gamma) * total_only + gamma * component_only
        assert loss_at(gamma) == pytest.approx(expected, rel=1e-6)


def test_component_gamma_rejects_a_single_component_prediction():
    harness = _loss_probe_harness(component_gamma=0.4)
    preds = torch.tensor([1.0, 2.0])
    ref = torch.tensor([0.0, 0.0])
    with pytest.raises(ValueError, match="multi-component"):
        harness._batch_loss(preds, ref, preds - ref, None, None)


def test_total_includes_d3_changes_the_total_and_carries_no_gradient(
    synthetic_dimer_batch, monkeypatch
):
    """D3 shifts the predicted total but must be gradient-free.

    ``mtp_mtp.d3`` is replaced by a per-edge tensor built from a leaf that
    *does* require grad, so a missing ``detach()`` would show up as a populated
    ``leaf.grad`` after ``backward()``.
    """
    batch = synthetic_dimer_batch
    n_edges = batch.dimer_ind_full.numel()
    leaf = torch.nn.Parameter(torch.tensor(0.5))

    def fake_d3(passed_batch, params=None):
        assert passed_batch is batch
        assert params == {"a1": 1.0}
        return leaf * torch.ones(n_edges)

    monkeypatch.setattr(mtp_mtp, "d3", fake_d3)

    trainable = torch.nn.Parameter(torch.ones(1))
    preds = trainable * torch.arange(
        1.0, batch.total_charge_A.numel() * 3 + 1.0
    ).reshape(batch.total_charge_A.numel(), 3)
    ref = batch.y[:, torch.tensor([0, 1, 2])]

    without_d3 = _loss_probe_harness(
        component_gamma=0.4, d3_damping_parameters={"a1": 1.0}
    )._batch_loss(preds, ref, preds - ref, batch, None)
    with_d3 = _loss_probe_harness(
        component_gamma=0.4,
        total_includes_d3=True,
        d3_damping_parameters={"a1": 1.0},
    )._batch_loss(preds, ref, preds - ref, batch, None)

    assert not torch.allclose(without_d3, with_d3)
    with_d3.backward()
    assert leaf.grad is None
    assert trainable.grad is not None
    assert torch.isfinite(trainable.grad).all()


def test_total_includes_d3_compares_against_all_four_sapt_columns(
    synthetic_dimer_batch, monkeypatch
):
    batch = synthetic_dimer_batch
    n_edges = batch.dimer_ind_full.numel()
    monkeypatch.setattr(
        mtp_mtp, "d3", lambda b, params=None: torch.zeros(n_edges)
    )
    # The shared fixture's y happens to satisfy |sum of 4| == |sum of 3|, so
    # pick columns where the three- and four-column totals genuinely differ.
    batch.y = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )
    preds = torch.zeros((batch.total_charge_A.numel(), 3))
    ref = batch.y[:, torch.tensor([0, 1, 2])]

    harness = _loss_probe_harness(
        component_gamma=0.0, total_includes_d3=True
    )
    loss = harness._batch_loss(preds, ref, preds - ref, batch, None)
    # With zero predictions and zero dispersion the total term is the MSE of
    # the four-column reference sum, not the three-column one.
    expected = torch.mean(torch.square(batch.y.sum(dim=-1)))
    assert torch.allclose(loss, expected)
    three_column = torch.mean(torch.square(ref.sum(dim=-1)))
    assert not torch.allclose(loss, three_column)


@pytest.mark.parametrize("gamma", [-0.1, 1.0001, 2.0, float("nan")])
def test_component_gamma_out_of_range_raises(gamma, nested_hfvr_vw_model):
    harness = _build_cliff_harness(CliffClassicalModel, nested_hfvr_vw_model)
    with pytest.raises(ValueError, match=r"component_gamma must be in"):
        harness.train(component_gamma=gamma)


def test_component_gamma_rejected_for_the_exchange_route(
    nested_hfvr_vw_model,
):
    harness = _build_cliff_harness(CliffExchangeModel, nested_hfvr_vw_model)
    with pytest.raises(ValueError, match="only supported for the combined"):
        harness.train(component_gamma=0.4)
    with pytest.raises(ValueError, match="only supported for the combined"):
        harness.train(total_includes_d3=True)
    # Even the Eq. (23) endpoint is rejected: the sentinel for "leave the loss
    # alone" is `None`, not `1.0`.
    with pytest.raises(ValueError, match="only supported for the combined"):
        harness.train(component_gamma=1.0)
    # The neutral default is accepted; training then fails for the ordinary
    # reason that no dataset was supplied.
    with pytest.raises(ValueError, match="No dataset provided"):
        harness.train()


def test_component_gamma_rejected_for_pre_existing_routes(
    nested_hfvr_vw_model,
):
    harness = RackersTholeDampingModel(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    with pytest.raises(ValueError, match="only supported for the combined"):
        harness.train(component_gamma=0.4)


@pytest.mark.parametrize(
    "harness_type",
    [CliffClassicalModel, CliffClassicalOverlapModel],
)
def test_component_gamma_is_recorded_on_the_harness_by_train(
    harness_type, nested_hfvr_vw_model
):
    harness = _build_cliff_harness(harness_type, nested_hfvr_vw_model)
    with pytest.raises(ValueError, match="No dataset provided"):
        harness.train(component_gamma=0.4, total_includes_d3=True)
    assert harness.component_gamma == 0.4
    assert harness.total_includes_d3 is True


class _StubTrainDataset:
    """Minimal stand-in for a prebatched pairwise dataset."""

    training_batch_size = 1

    def __len__(self):
        return 2

    def __getitem__(self, indices):
        return self


def test_total_includes_d3_requires_an_explicit_gamma(nested_hfvr_vw_model):
    """The default loss has no total term, so the flag would be a no-op."""
    harness = _build_cliff_harness(CliffClassicalModel, nested_hfvr_vw_model)
    with pytest.raises(
        ValueError,
        match="total_includes_d3 requires an explicit component_gamma",
    ):
        harness.train(total_includes_d3=True)
    # Accepted for any explicit gamma, including the 1.0 endpoint, which is a
    # legitimate Eq. (23) setting rather than the legacy default.
    for gamma in (0.4, 1.0):
        with pytest.raises(ValueError, match="No dataset provided"):
            harness.train(component_gamma=gamma, total_includes_d3=True)


def test_multi_process_training_is_no_longer_rejected():
    """The CLIFF routes now have a DDP path, and it is one loop, not two.

    This test used to assert the opposite -- that ``train(world_size=2)``
    raised ``NotImplementedError("Multi-process training ...")``. It does not
    any more. What is worth pinning instead is that the distributed path did
    not arrive as a *second* epoch loop: ``ddp_train`` delegates to
    ``single_proc_train``, so the golden source-introspection contracts in
    ``tests/test_cliff_induction_golden.py`` still cover the loop that
    actually runs under DDP. A real two-rank gloo run lives in
    ``tests/test_cliff_induction_ddp.py``.
    """
    train_source = inspect.getsource(AM_DimerParam_Model.train)
    assert "NotImplementedError" not in train_source
    assert "mp.spawn(" in train_source

    ddp_source = inspect.getsource(AM_DimerParam_Model.ddp_train)
    assert "self.single_proc_train(" in ddp_source
    assert "for epoch in range(" not in ddp_source


def test_component_gamma_survives_the_train_models_signature_filter():
    """``train_models.py`` drops any kwarg absent from ``train``'s signature.

    The shared pairwise tail filters ``train_kwargs`` through
    ``inspect.signature(apnet.train).parameters``, so these must be *named*
    parameters rather than ``**kwargs`` to be forwarded at all.
    """
    parameters = inspect.signature(AM_DimerParam_Model.train).parameters
    assert "component_gamma" in parameters
    assert "total_includes_d3" in parameters
    # `None` is the default, and it must survive the filter as a real value:
    # the filter keys on the parameter *name*, so a `None` default is forwarded
    # like any other, and the harness -- not the CLI -- owns its meaning.
    assert parameters["component_gamma"].default is None
    assert parameters["total_includes_d3"].default is False
    assert parameters["component_gamma"].kind is not (
        inspect.Parameter.VAR_KEYWORD
    )
    for harness_type in (
        CliffExchangeModel,
        CliffClassicalModel,
        CliffClassicalOverlapModel,
    ):
        harness_parameters = inspect.signature(harness_type.train).parameters
        assert "component_gamma" in harness_parameters
        assert "total_includes_d3" in harness_parameters


# ---------------------------------------------------------------------------
# Task D: checkpoint round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "harness_type,expected_mode,expected_model_type,parameter_names,_v,_s",
    CLIFF_HARNESSES,
    ids=CLIFF_HARNESS_IDS,
)
def test_cliff_checkpoint_round_trip(
    tmp_path,
    harness_type,
    expected_mode,
    expected_model_type,
    parameter_names,
    _v,
    _s,
    nested_hfvr_vw_model,
    synthetic_qcel_dimers,
):
    harness = _build_cliff_harness(harness_type, nested_hfvr_vw_model)
    before = harness.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )

    path = tmp_path / f"{expected_mode}.pt"
    harness.save_model(path)
    checkpoint = model_io.load_checkpoint(path)

    assert checkpoint["model_type"] == expected_model_type
    config = checkpoint["config"]
    assert config["model_type"] == expected_model_type
    assert config["parameter_names"] == list(parameter_names)
    assert config["dimer_eval"] == expected_mode
    for key in (
        "param_start_mean",
        "param_start_std",
        "positivity_epsilon",
        "width_floor",
        "elst_damping_type",
        "d3_damping_parameters",
        "nested_atom_model",
    ):
        assert key in config, key
    if expected_mode in COMBINED_CLIFF_DIMER_EVAL_MODES:
        assert "component_gamma" in config
        assert config["component_gamma"] is None
        assert config["total_includes_d3"] is False
    else:
        assert "component_gamma" not in config
    if expected_mode in mtp_mtp.INDUCTION_DIMER_EVAL_MODES:
        assert config["induction_convergence_threshold"] == 1e-8
        assert config["induction_max_iterations"] == 200

    # Nested state is restored from the checkpoint alone.
    loaded = harness_type(
        pre_trained_model_path=path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert loaded.atom_model is loaded.model.atom_model
    assert loaded.dimer_model.AtomTypeParam is loaded.model
    assert (
        loaded.model.n_message,
        loaded.model.n_neuron,
        loaded.model.n_embed,
    ) == (1, 8, 4)
    assert loaded.model.get_config() == harness.model.get_config()
    after = loaded.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    assert np.allclose(before, after, atol=1e-6)

    second_path = tmp_path / f"{expected_mode}-second.pt"
    loaded.save_model(second_path)
    reloaded = harness_type(
        pre_trained_model_path=second_path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert reloaded.model.get_config() == loaded.model.get_config()
    assert np.allclose(
        after,
        reloaded.predict_qcel_mols_dimer(synthetic_qcel_dimers, batch_size=2),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "harness_type,parameter_names",
    [(entry[0], entry[3]) for entry in CLIFF_HARNESSES],
    ids=CLIFF_HARNESS_IDS,
)
@pytest.mark.parametrize(
    "tamper_name",
    ["reordered", "missing", "foreign", "wrong_model_type", "bad_version",
     "missing_nested"],
)
def test_cliff_checkpoint_rejects_invalid_parameter_metadata(
    tmp_path,
    harness_type,
    parameter_names,
    tamper_name,
    nested_hfvr_vw_model,
):
    harness = _build_cliff_harness(harness_type, nested_hfvr_vw_model)
    checkpoint = harness._create_checkpoint()
    if tamper_name == "reordered":
        if len(parameter_names) == 1:
            pytest.skip("a one-element contract cannot be reordered")
        checkpoint["config"]["parameter_names"] = list(
            reversed(parameter_names)
        )
        match = "parameter_names"
    elif tamper_name == "missing":
        checkpoint["config"].pop("parameter_names")
        match = "parameter_names"
    elif tamper_name == "foreign":
        checkpoint["config"]["parameter_names"] = list(RACKERS_PARAMETER_NAMES)
        match = "parameter_names"
    elif tamper_name == "wrong_model_type":
        checkpoint["model_type"] = "AtomTypeParamNN"
        match = "model_type"
    elif tamper_name == "bad_version":
        checkpoint["checkpoint_version"] = 1
        match = "checkpoint_version"
    else:
        checkpoint["config"].pop("nested_atom_model")
        match = "nested_atom_model"

    path = tmp_path / "tampered.pt"
    model_io.save_checkpoint(checkpoint, path)
    with pytest.raises(ValueError, match=match):
        harness_type(
            pre_trained_model_path=path,
            atom_model=None,
            dataset=None,
            ignore_database_null=True,
            use_GPU=False,
        )


def test_cliff_checkpoint_error_messages_name_their_own_contract(
    tmp_path, nested_hfvr_vw_model
):
    """Generalized messages name the declaring model type, not "Rackers"."""
    harness = _build_cliff_harness(CliffClassicalModel, nested_hfvr_vw_model)
    checkpoint = harness._create_checkpoint()
    checkpoint["config"]["parameter_names"] = list(RACKERS_PARAMETER_NAMES)
    path = tmp_path / "labelled.pt"
    model_io.save_checkpoint(checkpoint, path)
    with pytest.raises(ValueError) as excinfo:
        CliffClassicalModel(
            pre_trained_model_path=path,
            atom_model=None,
            dataset=None,
            ignore_database_null=True,
            use_GPU=False,
        )
    assert str(excinfo.value) == (
        "CliffClassicalNN checkpoint parameter_names must exactly match "
        f"{list(CLIFF_CLASSICAL_PARAMETER_NAMES)}"
    )


def test_cliff_checkpoint_preserves_nondefault_scf_controls(
    tmp_path, nested_hfvr_vw_model
):
    harness = _build_cliff_harness(
        CliffClassicalOverlapModel,
        nested_hfvr_vw_model,
        induction_convergence_threshold=1e-6,
        induction_max_iterations=50,
    )
    path = tmp_path / "scf50.pt"
    harness.save_model(path)
    config = model_io.load_checkpoint(path)["config"]
    assert config["induction_convergence_threshold"] == 1e-6
    assert config["induction_max_iterations"] == 50

    loaded = CliffClassicalOverlapModel(
        pre_trained_model_path=path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert loaded.dimer_model.induction_convergence_threshold == 1e-6
    assert loaded.dimer_model.induction_max_iterations == 50


def test_legacy_cliff_checkpoint_uses_historical_scf_defaults(
    tmp_path, nested_hfvr_vw_model
):
    harness = _build_cliff_harness(
        CliffClassicalOverlapModel, nested_hfvr_vw_model
    )
    checkpoint = harness._create_checkpoint()
    checkpoint["config"].pop("induction_convergence_threshold")
    checkpoint["config"].pop("induction_max_iterations")
    path = tmp_path / "legacy-scf.pt"
    model_io.save_checkpoint(checkpoint, path)
    loaded = CliffClassicalOverlapModel(
        pre_trained_model_path=path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert loaded.dimer_model.induction_convergence_threshold == 1e-8
    assert loaded.dimer_model.induction_max_iterations == 200


def test_cliff_checkpoint_rejects_the_wrong_harness_mode(
    tmp_path, nested_hfvr_vw_model
):
    harness = _build_cliff_harness(CliffClassicalModel, nested_hfvr_vw_model)
    path = tmp_path / "wrong-mode.pt"
    harness.save_model(path)
    with pytest.raises(ValueError, match="dimer_eval"):
        CliffClassicalOverlapModel(
            pre_trained_model_path=path,
            atom_model=None,
            dataset=None,
            ignore_database_null=True,
            use_GPU=False,
        )


@pytest.mark.parametrize(
    "gamma,includes_d3",
    [(0.4, True), (1.0, False), (0.0, False), (None, False)],
    ids=["0.4-d3", "endpoint-1.0", "endpoint-0.0", "legacy-None"],
)
def test_cliff_combined_checkpoint_preserves_loss_weighting(
    tmp_path, gamma, includes_d3, nested_hfvr_vw_model
):
    """The recorded weighting round-trips, ``None`` included.

    ``None`` must not be coerced to ``1.0`` on reload: under Eq. (23) that
    value is the component-only endpoint, ``k`` times the legacy loss.
    """
    harness = _build_cliff_harness(
        CliffClassicalOverlapModel, nested_hfvr_vw_model
    )
    harness.component_gamma = gamma
    harness.total_includes_d3 = includes_d3
    path = tmp_path / "weighted.pt"
    harness.save_model(path)
    assert (
        model_io.load_checkpoint(path)["config"]["component_gamma"] == gamma
        if gamma is not None
        else model_io.load_checkpoint(path)["config"]["component_gamma"]
        is None
    )

    loaded = CliffClassicalOverlapModel(
        pre_trained_model_path=path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    if gamma is None:
        assert loaded.component_gamma is None
    else:
        assert loaded.component_gamma == gamma
    assert loaded.total_includes_d3 is includes_d3


@pytest.mark.parametrize("width_floor", [0.05, 0.25])
def test_cliff_checkpoint_preserves_width_floor(
    tmp_path, width_floor, nested_hfvr_vw_model
):
    harness = _build_cliff_harness(
        CliffExchangeModel, nested_hfvr_vw_model, width_floor=width_floor
    )
    assert harness.model.width_floor == width_floor
    path = tmp_path / "floor.pt"
    harness.save_model(path)
    loaded = CliffExchangeModel(
        pre_trained_model_path=path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert loaded.model.width_floor == width_floor
    assert loaded.width_floor == width_floor
    assert loaded.dimer_model._overlap_width_floor() == width_floor


RACKERS_CHECKPOINT_CONTRACT_MESSAGES = (
    (
        "version-missing",
        lambda checkpoint: checkpoint.pop("checkpoint_version"),
        "Rackers checkpoint_version mismatch: expected 2, got None",
    ),
    (
        "version-1",
        lambda checkpoint: checkpoint.__setitem__("checkpoint_version", 1),
        "Rackers checkpoint_version mismatch: expected 2, got 1",
    ),
    (
        "model-type",
        lambda checkpoint: checkpoint.__setitem__(
            "model_type", "AtomTypeParamNN"
        ),
        "Rackers checkpoint model_type mismatch: expected "
        "RackersTholeDampingNN, got 'AtomTypeParamNN'",
    ),
    (
        "parameter-names-reordered",
        lambda checkpoint: checkpoint["config"].__setitem__(
            "parameter_names", list(reversed(RACKERS_PARAMETER_NAMES))
        ),
        "Rackers checkpoint parameter_names must exactly match "
        "['elst', 'thole_direct', 'thole_mutual', 'ind_overlap']",
    ),
    (
        "parameter-names-missing",
        lambda checkpoint: checkpoint["config"].pop("parameter_names"),
        "Rackers checkpoint parameter_names must exactly match "
        "['elst', 'thole_direct', 'thole_mutual', 'ind_overlap']",
    ),
    (
        "nested-missing",
        lambda checkpoint: checkpoint["config"].pop("nested_atom_model"),
        "Rackers checkpoint missing nested_atom_model metadata",
    ),
)


@pytest.mark.parametrize(
    "tamper,expected",
    [entry[1:] for entry in RACKERS_CHECKPOINT_CONTRACT_MESSAGES],
    ids=[entry[0] for entry in RACKERS_CHECKPOINT_CONTRACT_MESSAGES],
)
def test_rackers_checkpoint_messages_are_byte_identical(
    tmp_path, tamper, expected, nested_hfvr_vw_model
):
    """Pin the pre-generalization Rackers checkpoint-contract error text.

    ``AM_DimerParam_Model.__init__`` now looks its expected contract up in
    ``POSITIVE_PARAMETER_CONTRACTS`` instead of hard-coding
    ``RackersTholeDampingNN``.  The Rackers strings predate that change, so
    they are asserted here with ``==`` rather than a substring match; the
    "Rackers" prefix in particular comes from a label mapping and would
    silently become "RackersTholeDampingNN" if that mapping were dropped.
    """
    harness = RackersTholeDampingModel(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    checkpoint = harness._create_checkpoint()
    tamper(checkpoint)
    path = tmp_path / "tampered.pt"
    model_io.save_checkpoint(checkpoint, path)

    with pytest.raises(ValueError) as excinfo:
        RackersTholeDampingModel(
            pre_trained_model_path=path,
            atom_model=None,
            dataset=None,
            ignore_database_null=True,
            use_GPU=False,
        )
    assert str(excinfo.value) == expected


def test_rackers_checkpoint_dimer_eval_message_is_byte_identical(
    tmp_path, nested_hfvr_vw_model
):
    harness = RackersTholeDampingModel(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    path = tmp_path / "mode.pt"
    harness.save_model(path)
    with pytest.raises(ValueError) as excinfo:
        mtp_mtp.RackersTholeDampingOverlapModel(
            pre_trained_model_path=path,
            atom_model=None,
            dataset=None,
            ignore_database_null=True,
            use_GPU=False,
        )
    assert str(excinfo.value) == (
        "Rackers checkpoint dimer_eval mismatch: expected "
        "rackers_thole_overlap, got 'rackers_thole'"
    )


# ---------------------------------------------------------------------------
# Task E: train_models.py dispatch and CLI
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _FakeCliffHarnessBase:
    """Stand-in for a CLIFF harness, mirroring ``_FakeRackersHarnessBase``.

    ``train`` declares ``component_gamma`` / ``total_includes_d3`` as *named*
    parameters exactly as ``AM_DimerParam_Model.train`` does, because
    ``train_models.py`` filters ``train_kwargs`` through
    ``inspect.signature(apnet.train).parameters``: a ``**kwargs`` fake would
    make the forwarding assertions vacuous.
    """

    calls = []
    DIMER_EVAL = None
    MODEL_TYPE = None
    PARAMETER_NAMES = ()

    def __init__(self, atom_model, **kwargs):
        self.kwargs = {"atom_model": atom_model, **kwargs}
        self.model = atom_model
        self.model.requires_grad_(not kwargs["freeze_atom_model"])
        self.dataset = object()
        self.train_calls = []
        type(self).calls.append(self)

    def train(
        self,
        model_path=None,
        n_epochs=50,
        world_size=1,
        omp_num_threads_per_process=6,
        lr=5e-4,
        dataloader_num_workers=4,
        random_seed=42,
        lr_decay=None,
        component_gamma=None,
        total_includes_d3=False,
    ):
        self.train_calls.append(
            {
                "model_path": model_path,
                "n_epochs": n_epochs,
                "world_size": world_size,
                "omp_num_threads_per_process": omp_num_threads_per_process,
                "lr": lr,
                "dataloader_num_workers": dataloader_num_workers,
                "random_seed": random_seed,
                "lr_decay": lr_decay,
                "component_gamma": component_gamma,
                "total_includes_d3": total_includes_d3,
            }
        )


class _FakeCliffExchangeModel(_FakeCliffHarnessBase):
    calls = []
    DIMER_EVAL = "cliff_exch"
    MODEL_TYPE = "CliffExchangeNN"
    PARAMETER_NAMES = CLIFF_EXCH_PARAMETER_NAMES


class _FakeCliffClassicalModel(_FakeCliffHarnessBase):
    calls = []
    DIMER_EVAL = "cliff_classical"
    MODEL_TYPE = "CliffClassicalNN"
    PARAMETER_NAMES = CLIFF_CLASSICAL_PARAMETER_NAMES


class _FakeCliffClassicalOverlapModel(_FakeCliffHarnessBase):
    calls = []
    DIMER_EVAL = "cliff_classical_overlap"
    MODEL_TYPE = "CliffClassicalNN"
    PARAMETER_NAMES = CLIFF_CLASSICAL_PARAMETER_NAMES


_CLIFF_DISPATCH_FAKES = {
    "CliffExchangeModel": _FakeCliffExchangeModel,
    "CliffClassicalModel": _FakeCliffClassicalModel,
    "CliffClassicalOverlapModel": _FakeCliffClassicalOverlapModel,
}


def _patch_cliff_dispatch_fakes(monkeypatch):
    """Replace the heavyweight construction on every CLIFF dispatch path.

    ``AtomTypeParamModel`` is the shared fake from the Rackers dispatch tests,
    so the two suites cannot disagree about the wrapper's contract.
    """
    _FakeAtomTypeParamModel.calls.clear()
    for fake in _CLIFF_DISPATCH_FAKES.values():
        fake.calls.clear()
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "AtomTypeParamModel",
        _FakeAtomTypeParamModel,
    )
    for identifier, fake in _CLIFF_DISPATCH_FAKES.items():
        monkeypatch.setattr(
            train_models.AtomPairwiseModels.mtp_mtp, identifier, fake
        )


# (identifier, real harness, dimer mode, initial values, initial stds)
CLIFF_CLI_ROUTES = (
    (
        "CliffExchangeModel",
        CliffExchangeModel,
        "cliff_exch",
        CLIFF_EXCH_INITIAL_VALUES,
        CLIFF_EXCH_INITIAL_STDS,
    ),
    (
        "CliffClassicalModel",
        CliffClassicalModel,
        "cliff_classical",
        CLIFF_CLASSICAL_INITIAL_VALUES,
        CLIFF_CLASSICAL_INITIAL_STDS,
    ),
    (
        "CliffClassicalOverlapModel",
        CliffClassicalOverlapModel,
        "cliff_classical_overlap",
        CLIFF_CLASSICAL_INITIAL_VALUES,
        CLIFF_CLASSICAL_INITIAL_STDS,
    ),
)
CLIFF_CLI_IDS = [route[0] for route in CLIFF_CLI_ROUTES]
COMBINED_CLIFF_CLI_ROUTES = [
    route for route in CLIFF_CLI_ROUTES if route[2] != "cliff_exch"
]
COMBINED_CLIFF_CLI_IDS = [route[0] for route in COMBINED_CLIFF_CLI_ROUTES]


def test_cliff_cli_route_sets_are_exactly_the_declared_identifiers():
    assert train_models.CLIFF_MODEL_TYPES == {
        "CliffExchangeModel",
        "CliffClassicalModel",
        "CliffClassicalOverlapModel",
        "CliffClassicalOverlapMPNNModel",
    }
    # The exchange route has no total/component split, so it is deliberately
    # absent from the combined set even though it is a CLIFF route.
    assert train_models.COMBINED_CLIFF_MODEL_TYPES == {
        "CliffClassicalModel",
        "CliffClassicalOverlapModel",
        "CliffClassicalOverlapMPNNModel",
    }
    # Only the message-passing head has an architecture to size.
    assert train_models.CLIFF_MPNN_MODEL_TYPES == {
        "CliffClassicalOverlapMPNNModel",
    }
    assert train_models.CLIFF_MPNN_MODEL_TYPES <= train_models.CLIFF_MODEL_TYPES
    assert train_models.POSITIVE_PARAM_MODEL_TYPES == (
        train_models.RACKERS_MODEL_TYPES | train_models.CLIFF_MODEL_TYPES
    )


@pytest.mark.parametrize(
    "identifier,harness_type,expected_mode,initial_values,initial_stds",
    CLIFF_CLI_ROUTES,
    ids=CLIFF_CLI_IDS,
)
def test_cliff_cli_identifier_resolves_to_its_harness_and_mode(
    identifier, harness_type, expected_mode, initial_values, initial_stds
):
    """The identifier is the harness class name; the class fixes the mode."""
    assert getattr(mtp_mtp, identifier) is harness_type
    assert harness_type.DIMER_EVAL == expected_mode
    # `_cliff_parameter_contract` reads the defaults out of `mtp_mtp` rather
    # than restating the numbers, so the CLI cannot drift from the harness.
    names, means, stds = train_models._cliff_parameter_contract(identifier)
    assert names is harness_type.PARAMETER_NAMES
    assert means is initial_values
    assert stds is initial_stds


@pytest.mark.parametrize(
    "identifier,harness_type,expected_mode,initial_values,initial_stds",
    CLIFF_CLI_ROUTES,
    ids=CLIFF_CLI_IDS,
)
@pytest.mark.parametrize("freeze_atom_model", [True, False])
def test_cliff_dispatch_selects_harness_and_forwards_contract(
    tmp_path,
    monkeypatch,
    identifier,
    harness_type,
    expected_mode,
    initial_values,
    initial_stds,
    freeze_atom_model,
):
    _patch_cliff_dispatch_fakes(monkeypatch)

    model_out = tmp_path / "cliff-output.pt"
    train_models.train_pairwise_model(
        apnet_model_type=identifier,
        model_out=str(model_out),
        am_model_path="multipole-checkpoint.pt",
        atom_type_param_model_path="hfvr-vw-checkpoint.pt",
        data_dir="cliff-data",
        n_epochs=5,
        lr=2e-4,
        lr_decay=0.5,
        random_seed=13,
        spec_type=11,
        r_cut=6.0,
        n_rbf=6,
        n_neuron=48,
        n_embed=12,
        n_params=99,
        pre_trained_model_path="cliff-checkpoint.pt",
        elst_damping_type="AMOEBA",
        ds_in_memory=True,
        freeze_atom_model=freeze_atom_model,
        omp_num_threads=17,
    )

    # Stage one: the HFVR/valence-width wrapper built from the two checkpoints.
    assert len(_FakeAtomTypeParamModel.calls) == 1
    hfvr_wrapper = _FakeAtomTypeParamModel.calls[0]
    assert hfvr_wrapper.kwargs == {
        "ds_root": None,
        "use_GPU": False,
        "ignore_database_null": True,
        "atom_model_pre_trained_path": "multipole-checkpoint.pt",
        "pre_trained_model_path": "hfvr-vw-checkpoint.pt",
        "freeze_atom_model": freeze_atom_model,
    }

    # Stage two: exactly the intended harness, wrapping the wrapper's `.model`.
    selected = _CLIFF_DISPATCH_FAKES[identifier]
    assert len(selected.calls) == 1
    for other_identifier, other in _CLIFF_DISPATCH_FAKES.items():
        if other_identifier != identifier:
            assert other.calls == []
    assert selected.DIMER_EVAL == expected_mode
    harness = selected.calls[0]
    assert harness.kwargs["atom_model"] is hfvr_wrapper.model
    assert harness.kwargs["pre_trained_model_path"] == "cliff-checkpoint.pt"
    assert harness.kwargs["param_start_mean"] == list(initial_values)
    assert harness.kwargs["param_start_std"] == list(initial_stds)
    assert harness.kwargs["freeze_atom_model"] is freeze_atom_model
    assert harness.kwargs["elst_damping_type"] == "AMOEBA"
    assert harness.kwargs["n_rbf"] == 6
    assert harness.kwargs["n_neuron"] == 48
    assert harness.kwargs["n_embed"] == 12
    assert harness.kwargs["r_cut"] == 6.0
    assert harness.kwargs["ds_spec_type"] == 11
    assert harness.kwargs["ds_root"] == "cliff-data"
    assert harness.kwargs["ds_random_seed"] == 13
    assert harness.kwargs["ds_in_memory"] is True
    # Every CLIFF harness fixes its own parameter count; `--n_params 99` above
    # must not reach it.
    assert "n_params" not in harness.kwargs
    assert all(
        parameter.requires_grad is not freeze_atom_model
        for parameter in harness.model.parameters()
    )
    assert harness.train_calls == [
        {
            "model_path": str(model_out),
            "n_epochs": 5,
            # No DDP path exists for these harnesses.
            "world_size": 1,
            "omp_num_threads_per_process": 17,
            "lr": 2e-4,
            "dataloader_num_workers": 4,
            "random_seed": 13,
            "lr_decay": 0.5,
            # `None`, not `1.0`: the sentinel for the legacy plain MSE.
            "component_gamma": None,
            "total_includes_d3": False,
        }
    ]


@pytest.mark.parametrize("identifier", CLIFF_CLI_IDS)
def test_cliff_dispatch_omits_the_legacy_pretrained_checkpoint(
    tmp_path, monkeypatch, identifier
):
    """An unset ``pre_trained_model_path`` resolves to ``None``, not dAPNet2."""
    _patch_cliff_dispatch_fakes(monkeypatch)
    train_models.train_pairwise_model(
        apnet_model_type=identifier,
        model_out=str(tmp_path / "missing-output.pt"),
    )
    harness = _CLIFF_DISPATCH_FAKES[identifier].calls[0]
    assert harness.kwargs["pre_trained_model_path"] is None
    assert (
        harness.kwargs["pre_trained_model_path"]
        != train_models.LEGACY_PAIRWISE_PRETRAINED_MODEL_PATH
    )


@pytest.mark.parametrize("identifier", CLIFF_CLI_IDS)
@pytest.mark.parametrize("field", ["param_start_mean", "param_start_std"])
def test_cliff_dispatch_rejects_scalar_broadcasting(
    monkeypatch, identifier, field
):
    """A bare scalar is ambiguous on a fixed-contract route, so it is rejected.

    The five-parameter contract mixes electrostatic, Thole, overlap, and
    exchange scales; broadcasting one number across them would silently invent
    an initialization no caller asked for.
    """
    _patch_cliff_dispatch_fakes(monkeypatch)
    with pytest.raises(ValueError, match="exactly"):
        train_models.train_pairwise_model(
            apnet_model_type=identifier,
            pre_trained_model_path=None,
            **{field: 1.8},
        )
    assert _FakeAtomTypeParamModel.calls == []
    assert _CLIFF_DISPATCH_FAKES[identifier].calls == []


@pytest.mark.parametrize(
    "identifier,harness_type,expected_mode,initial_values,initial_stds",
    CLIFF_CLI_ROUTES,
    ids=CLIFF_CLI_IDS,
)
@pytest.mark.parametrize("field", ["param_start_mean", "param_start_std"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_cliff_dispatch_rejects_wrong_length_overrides(
    monkeypatch,
    identifier,
    harness_type,
    expected_mode,
    initial_values,
    initial_stds,
    field,
    delta,
):
    _patch_cliff_dispatch_fakes(monkeypatch)
    n_params = len(harness_type.PARAMETER_NAMES)
    length = n_params + delta
    if length < 0:
        pytest.skip("no shorter list exists for a one-parameter contract")
    expected = "exactly one value" if n_params == 1 else "exactly five values"
    with pytest.raises(ValueError, match=expected):
        train_models.train_pairwise_model(
            apnet_model_type=identifier,
            pre_trained_model_path=None,
            **{field: [0.5] * length},
        )
    assert _CLIFF_DISPATCH_FAKES[identifier].calls == []


@pytest.mark.parametrize(
    "identifier,harness_type,expected_mode,initial_values,initial_stds",
    CLIFF_CLI_ROUTES,
    ids=CLIFF_CLI_IDS,
)
def test_cliff_dispatch_accepts_a_correct_length_override(
    tmp_path,
    monkeypatch,
    identifier,
    harness_type,
    expected_mode,
    initial_values,
    initial_stds,
):
    _patch_cliff_dispatch_fakes(monkeypatch)
    means = [1.25 + index for index in range(len(harness_type.PARAMETER_NAMES))]
    stds = [0.02] * len(harness_type.PARAMETER_NAMES)
    train_models.train_pairwise_model(
        apnet_model_type=identifier,
        model_out=str(tmp_path / "override.pt"),
        pre_trained_model_path=None,
        param_start_mean=means,
        param_start_std=stds,
    )
    harness = _CLIFF_DISPATCH_FAKES[identifier].calls[0]
    assert harness.kwargs["param_start_mean"] == means
    assert harness.kwargs["param_start_std"] == stds


@pytest.mark.parametrize("identifier", CLIFF_CLI_IDS)
@pytest.mark.parametrize(
    "field,value,match",
    [
        ("param_start_mean", 0.0, "strictly greater"),
        ("param_start_mean", float("nan"), "strictly greater"),
        ("param_start_std", -0.1, "greater than or equal"),
    ],
)
def test_cliff_dispatch_routes_through_the_shared_validator(
    monkeypatch, identifier, field, value, match
):
    """Domain errors come from ``_validate_positive_initialization`` itself."""
    _patch_cliff_dispatch_fakes(monkeypatch)
    _, means, stds = train_models._cliff_parameter_contract(identifier)
    overrides = {
        "param_start_mean": list(means),
        "param_start_std": list(stds),
    }
    overrides[field][0] = value
    with pytest.raises(ValueError, match=match):
        train_models.train_pairwise_model(
            apnet_model_type=identifier,
            pre_trained_model_path=None,
            **overrides,
        )
    assert _CLIFF_DISPATCH_FAKES[identifier].calls == []


# --- component_gamma / total_includes_d3 -----------------------------------


def test_component_gamma_survives_the_dispatch_signature_filter():
    """The filter train_models.py applies must keep both new kwargs.

    Asserted against the *real* harnesses, since the dispatch tests above use
    fakes: `train_kwargs` is filtered by
    ``inspect.signature(apnet.train).parameters``, so a kwarg missing from that
    mapping is dropped silently rather than raising.
    """
    for harness_type in (
        CliffExchangeModel,
        CliffClassicalModel,
        CliffClassicalOverlapModel,
    ):
        supported = inspect.signature(harness_type.train).parameters
        assert "component_gamma" in supported
        assert "total_includes_d3" in supported
        assert supported["component_gamma"].default is None


@pytest.mark.parametrize(
    "identifier,harness_type,expected_mode,initial_values,initial_stds",
    COMBINED_CLIFF_CLI_ROUTES,
    ids=COMBINED_CLIFF_CLI_IDS,
)
@pytest.mark.parametrize("gamma", [0.0, 0.4, 1.0])
def test_component_gamma_is_forwarded_verbatim(
    tmp_path,
    monkeypatch,
    capsys,
    identifier,
    harness_type,
    expected_mode,
    initial_values,
    initial_stds,
    gamma,
):
    _patch_cliff_dispatch_fakes(monkeypatch)
    train_models.train_pairwise_model(
        apnet_model_type=identifier,
        model_out=str(tmp_path / "gamma.pt"),
        pre_trained_model_path=None,
        component_gamma=gamma,
        total_includes_d3=True,
    )
    harness = _CLIFF_DISPATCH_FAKES[identifier].calls[0]
    assert harness.train_calls[0]["component_gamma"] == pytest.approx(gamma)
    assert harness.train_calls[0]["total_includes_d3"] is True
    # Neither kwarg may show up in the "unsupported train() kwargs" notice.
    captured = capsys.readouterr().out
    assert "component_gamma" not in captured
    assert "total_includes_d3" not in captured


@pytest.mark.parametrize(
    "identifier,harness_type,expected_mode,initial_values,initial_stds",
    CLIFF_CLI_ROUTES,
    ids=CLIFF_CLI_IDS,
)
def test_component_gamma_default_reaching_the_harness_is_none(
    tmp_path,
    monkeypatch,
    identifier,
    harness_type,
    expected_mode,
    initial_values,
    initial_stds,
):
    """``None``, not ``1.0``: ``1.0`` would be ``k`` times the legacy loss."""
    _patch_cliff_dispatch_fakes(monkeypatch)
    train_models.train_pairwise_model(
        apnet_model_type=identifier,
        model_out=str(tmp_path / "default-gamma.pt"),
        pre_trained_model_path=None,
    )
    forwarded = _CLIFF_DISPATCH_FAKES[identifier].calls[0].train_calls[0]
    assert forwarded["component_gamma"] is None
    assert forwarded["total_includes_d3"] is False
    assert (
        inspect.signature(train_models.train_pairwise_model)
        .parameters["component_gamma"]
        .default
        is None
    )


@pytest.mark.parametrize(
    "identifier,harness_type,expected_mode,initial_values,initial_stds",
    COMBINED_CLIFF_CLI_ROUTES,
    ids=COMBINED_CLIFF_CLI_IDS,
)
def test_include_total_mse_becomes_gamma_one_half_on_cliff_routes(
    tmp_path,
    monkeypatch,
    identifier,
    harness_type,
    expected_mode,
    initial_values,
    initial_stds,
):
    """``--include_total_mse`` is the pre-CLIFF spelling of "fit the total too".

    ``AM_DimerParam_Model.train`` never accepted it, so on a CLIFF route it is
    reinterpreted as ``component_gamma = 0.5`` rather than filtered away.
    """
    _patch_cliff_dispatch_fakes(monkeypatch)
    train_models.train_pairwise_model(
        apnet_model_type=identifier,
        model_out=str(tmp_path / "include-total.pt"),
        pre_trained_model_path=None,
        include_total_mse=True,
    )
    forwarded = _CLIFF_DISPATCH_FAKES[identifier].calls[0].train_calls[0]
    assert forwarded["component_gamma"] == pytest.approx(
        train_models.CLIFF_INCLUDE_TOTAL_MSE_GAMMA
    )
    assert forwarded["component_gamma"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "identifier,harness_type,expected_mode,initial_values,initial_stds",
    COMBINED_CLIFF_CLI_ROUTES,
    ids=COMBINED_CLIFF_CLI_IDS,
)
def test_explicit_component_gamma_wins_nothing_over_include_total_mse(
    monkeypatch,
    identifier,
    harness_type,
    expected_mode,
    initial_values,
    initial_stds,
):
    """Supplying both spellings is an error, not a precedence rule."""
    _patch_cliff_dispatch_fakes(monkeypatch)
    with pytest.raises(
        ValueError, match="include_total_mse and component_gamma"
    ):
        train_models.train_pairwise_model(
            apnet_model_type=identifier,
            pre_trained_model_path=None,
            include_total_mse=True,
            component_gamma=0.4,
        )
    assert _CLIFF_DISPATCH_FAKES[identifier].calls == []


def test_include_total_mse_on_the_exchange_route_raises(monkeypatch):
    """The shorthand resolves to a gamma, which the exchange route rejects."""
    _patch_cliff_dispatch_fakes(monkeypatch)
    with pytest.raises(ValueError, match="component_gamma is only supported"):
        train_models.train_pairwise_model(
            apnet_model_type="CliffExchangeModel",
            pre_trained_model_path=None,
            include_total_mse=True,
        )
    assert _CLIFF_DISPATCH_FAKES["CliffExchangeModel"].calls == []


@pytest.mark.parametrize(
    "identifier",
    [
        "CliffExchangeModel",
        "RackersTholeDampingModel",
        "RackersTholeDampingOverlapModel",
        "AM-DimerParam",
        "APNet2",
        "APNet3-fused",
    ],
)
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"component_gamma": 0.4}, "component_gamma is only supported"),
        ({"component_gamma": 0.0}, "component_gamma is only supported"),
        ({"total_includes_d3": True}, "total_includes_d3 is only supported"),
    ],
)
def test_component_gamma_rejected_on_every_route_without_a_split(
    monkeypatch, identifier, kwargs, match
):
    """A clear error, never a silent drop.

    The signature filter would otherwise discard these for any route whose
    ``train`` does not name them, and forward a meaningless ``0.4`` to the
    ones that do.  The check runs before any dataset or model construction, so
    the heavyweight legacy routes are safe to exercise here.
    """
    _patch_cliff_dispatch_fakes(monkeypatch)
    with pytest.raises(ValueError, match=match):
        train_models.train_pairwise_model(
            apnet_model_type=identifier,
            pre_trained_model_path=None,
            **kwargs,
        )
    assert _FakeAtomTypeParamModel.calls == []
    for fake in _CLIFF_DISPATCH_FAKES.values():
        assert fake.calls == []


def test_include_total_mse_behavior_is_unchanged_off_the_cliff_routes(
    tmp_path, monkeypatch
):
    """A pre-existing route still merely forwards (and then filters) the flag."""
    from .test_rackers_thole_damping import (
        _FakeRackersTholeDampingModel,
        _patch_rackers_dispatch_fakes,
    )

    _patch_rackers_dispatch_fakes(monkeypatch)
    train_models.train_pairwise_model(
        apnet_model_type="RackersTholeDampingModel",
        model_out=str(tmp_path / "rackers-include-total.pt"),
        pre_trained_model_path=None,
        include_total_mse=True,
    )
    forwarded = _FakeRackersTholeDampingModel.calls[0].train_calls[0]
    # No component_gamma is invented for a non-CLIFF route, and the flag is
    # filtered out by signature exactly as it always was.
    assert "component_gamma" not in forwarded
    assert "include_total_mse" not in forwarded


# --- target-column dispatch -------------------------------------------------


@pytest.mark.parametrize(
    "identifier,expected_y_ind",
    [
        ("CliffExchangeModel", 1),
        ("CliffClassicalModel", torch.tensor([0, 1, 2])),
        ("CliffClassicalOverlapModel", torch.tensor([0, 1, 2])),
    ],
    ids=CLIFF_CLI_IDS,
)
def test_cliff_cli_identifier_selects_expected_target_columns(
    identifier,
    expected_y_ind,
    nested_hfvr_vw_model,
    synthetic_dimer_batch,
    monkeypatch,
    tmp_path,
):
    """Resolve the identifier the way the dispatch does, then run one epoch.

    ``getattr(mtp_mtp, identifier)`` is exactly what ``train_pairwise_model``
    evaluates, so this ties the CLI string to the SAPT columns actually fitted:
    ``Exch`` alone (scalar ``1``) for exchange and ``[Elst, Exch, Ind]`` for the
    classical routes.
    """
    harness_type = getattr(train_models.AtomPairwiseModels.mtp_mtp, identifier)
    harness = _build_cliff_harness(harness_type, nested_hfvr_vw_model)
    harness.example_input = lambda: synthetic_dimer_batch.batch_atomic_A
    harness.compile_model = lambda: None

    selected = []
    original = harness._AM_DimerParam_Model__train_batches_single_proc

    def record(*args, **kwargs):
        selected.append(kwargs["y_ind"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        harness, "_AM_DimerParam_Model__train_batches_single_proc", record
    )
    harness.model_save_path = tmp_path / f"{identifier}.pt"
    harness.single_proc_train(
        train_dataset=[_make_collate_item(1.0)],
        test_dataset=[_make_collate_item(1.1)],
        n_epochs=1,
        batch_size=1,
        lr=1e-5,
        pin_memory=False,
        num_workers=0,
    )

    assert len(selected) == 1
    if isinstance(expected_y_ind, torch.Tensor):
        assert isinstance(selected[0], torch.Tensor)
        assert torch.equal(selected[0], expected_y_ind)
    else:
        assert not isinstance(selected[0], torch.Tensor)
        assert selected[0] == expected_y_ind


# --- CLI surface ------------------------------------------------------------


def test_train_models_help_exits_zero_and_advertises_the_cliff_routes():
    # `./src` is prepended so the subprocess imports this worktree's
    # `apnet_pt`, the way `tests/conftest.py` does for the pytest process
    # itself; the editable install in the environment points elsewhere.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "train_models.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    help_text = result.stdout
    for identifier in (
        "CliffExchangeModel",
        "CliffClassicalModel",
        "CliffClassicalOverlapModel",
    ):
        assert identifier in help_text
    # The pre-existing Rackers mention must survive alongside them.
    assert "RackersTholeDampingModel" in help_text
    assert "RackersTholeDampingOverlapModel" in help_text
    assert "--component_gamma" in help_text
    assert "--total_includes_d3" in help_text
    assert "--grad_clip_norm" in help_text
    assert "--grad_clip_mode" in help_text
    assert "--thole_lr" in help_text
    assert "--induction_diagnostics" in help_text
    assert "--induction_convergence_threshold" in help_text
    assert "--induction_max_iterations" in help_text


# ---------------------------------------------------------------------------
# Raw-parameter bounds and gradient survival
#
# The first 100-epoch CLIFF run collapsed four of the five parameter columns to
# `positivity_epsilon` and inflated the fifth to ~90x its seed.  The mechanism
# is that `K = softplus(raw) + eps` has `dK/draw = sigmoid(raw)`, so a column
# driven toward zero loses the very gradient that would bring it back.  These
# tests pin the two properties that fix it: the bound is enforced on the value,
# and the gradient survives being outside it.
# ---------------------------------------------------------------------------


def test_ste_clamp_bounds_value_and_passes_gradient():
    lower = torch.tensor([[-2.0]])
    upper = torch.tensor([[3.0]])
    raw = torch.tensor(
        [[-40.0], [-2.5], [-2.0], [0.0], [3.0], [3.5], [40.0]],
        requires_grad=True,
    )
    clamped = mtp_mtp._ste_clamp(raw, lower, upper)
    assert torch.all(clamped >= lower)
    assert torch.all(clamped <= upper)
    # Inside the interval the clamp is the identity, not merely close to it.
    inside = (raw.detach() >= lower) & (raw.detach() <= upper)
    assert torch.equal(clamped.detach()[inside], raw.detach()[inside])

    clamped.sum().backward()
    # The whole point: an out-of-range parameter still receives gradient 1, so
    # it can climb back in.  A plain `clamp` would give 0 here and freeze it.
    assert torch.equal(raw.grad, torch.ones_like(raw))


def test_ste_clamp_accepts_one_sided_bounds():
    raw = torch.tensor([[-5.0], [5.0]], requires_grad=True)
    lower_only = mtp_mtp._ste_clamp(raw, torch.tensor([[0.0]]), None)
    assert lower_only.detach().tolist() == [[0.0], [5.0]]
    upper_only = mtp_mtp._ste_clamp(raw, None, torch.tensor([[0.0]]))
    assert upper_only.detach().tolist() == [[-5.0], [0.0]]
    assert torch.equal(
        mtp_mtp._ste_clamp(raw, None, None).detach(), raw.detach()
    )


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_bounds_are_the_configured_multiples_of_the_seed(
    model_type, _name, _parameter_names, values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    """The raw buffers must map back to `fraction * seed` and `multiple * seed`."""
    model = _build_head(model_type, nested_hfvr_vw_model)
    seeds = torch.tensor(values, dtype=torch.get_default_dtype())

    floor = F.softplus(model.raw_parameter_floor) + model.positivity_epsilon
    ceiling = F.softplus(model.raw_parameter_ceiling) + model.positivity_epsilon
    # The floor is per column on the five-parameter contract: the induction
    # columns are held at 0.5x their seed because near-zero Thole damping makes
    # the mutual polarization solve diverge, while `exch` keeps the loose 0.05x
    # it needs for hydrogen's Table I value at 0.31x the seed.
    expected_floor = torch.tensor(
        mtp_mtp._broadcast_bound_scale(
            model.param_floor_fraction, seeds.numel()
        ),
        dtype=seeds.dtype,
    )
    assert torch.allclose(floor.reshape(-1), expected_floor * seeds, atol=1e-5)
    expected_ceiling = torch.tensor(
        mtp_mtp._broadcast_bound_scale(
            model.param_ceiling_multiple, seeds.numel()
        ),
        dtype=seeds.dtype,
    )
    assert torch.allclose(
        ceiling.reshape(-1), expected_ceiling * seeds, atol=1e-4
    )
    # Config-derived, so deliberately absent from the checkpoint: a build that
    # predates the bound must still be able to load a checkpoint written now.
    assert "raw_parameter_floor" not in model.state_dict()
    assert "raw_parameter_ceiling" not in model.state_dict()


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_survives_saturating_readout_with_live_gradient(
    model_type, _name, _parameter_names, values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    """A hostile readout must park the head at the floor, still trainable.

    This is the exact failure that killed the first run, reproduced in one
    forward pass: drive the correction MLP hard negative and check both that
    the emitted parameter stops at the floor and that gradient still flows back
    into the readout weights.  Without the bound the parameter reaches
    ``positivity_epsilon`` and the gradient underflows to ~1e-13.
    """
    bounded = _build_head(model_type, nested_hfvr_vw_model)
    unbounded = _build_head(
        model_type,
        copy.deepcopy(nested_hfvr_vw_model),
        param_floor_fraction=None,
        param_ceiling_multiple=None,
    )
    floor = (
        F.softplus(bounded.raw_parameter_floor) + bounded.positivity_epsilon
    )

    grads = {}
    for label, model in (("bounded", bounded), ("unbounded", unbounded)):
        with torch.no_grad():
            for head in model.param_readout_layers:
                for readout in head:
                    for parameter in readout.parameters():
                        parameter.fill_(-25.0)
        parameters = model(atomic_batch)[-1]
        assert torch.isfinite(parameters).all()
        parameters.sum().backward()
        grads[label] = max(
            parameter.grad.abs().max().item()
            for head in model.param_readout_layers
            for readout in head
            for parameter in readout.parameters()
            if parameter.grad is not None
        )
        if label == "bounded":
            assert torch.allclose(
                parameters, floor.expand_as(parameters), atol=1e-5
            )
        else:
            assert torch.all(parameters < 1e-3)

    assert grads["unbounded"] < 1e-6, grads["unbounded"]
    assert grads["bounded"] > 1e-3 * (1.0 + grads["unbounded"]), grads
    assert grads["bounded"] > 1e6 * grads["unbounded"]


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_cliff_head_ceiling_caps_a_runaway_readout(
    model_type, _name, _parameter_names, values, _stds,
    atomic_batch, nested_hfvr_vw_model,
):
    model = _build_head(model_type, nested_hfvr_vw_model)
    with torch.no_grad():
        for head in model.param_readout_layers:
            for readout in head:
                for parameter in readout.parameters():
                    parameter.fill_(25.0)
    parameters = model(atomic_batch)[-1]
    ceiling = F.softplus(model.raw_parameter_ceiling) + model.positivity_epsilon
    assert torch.isfinite(parameters).all()
    assert torch.all(parameters <= ceiling + 1e-4)


@pytest.mark.parametrize(
    "bad",
    [0.0, -1.0, float("nan"), float("inf"), "x", object()],
)
@pytest.mark.parametrize(
    "field", ["param_floor_fraction", "param_ceiling_multiple"]
)
def test_cliff_head_rejects_invalid_bound_scale(
    bad, field, nested_hfvr_vw_model
):
    with pytest.raises(ValueError, match=field):
        _build_head(
            mtp_mtp.CliffExchangeNN, nested_hfvr_vw_model, **{field: bad}
        )


def test_cliff_head_rejects_inverted_bounds(nested_hfvr_vw_model):
    with pytest.raises(ValueError, match="strictly less than"):
        _build_head(
            mtp_mtp.CliffExchangeNN,
            nested_hfvr_vw_model,
            param_floor_fraction=2.0,
            param_ceiling_multiple=1.0,
        )


def test_cliff_head_accepts_disabled_bounds(nested_hfvr_vw_model, atomic_batch):
    """`None`/`None` reproduces the pre-bound forward exactly."""
    model = _build_head(
        mtp_mtp.CliffExchangeNN,
        nested_hfvr_vw_model,
        param_floor_fraction=None,
        param_ceiling_multiple=None,
    )
    assert model.raw_parameter_floor is None
    assert model.raw_parameter_ceiling is None
    config = model.get_config()
    assert config["param_floor_fraction"] is None
    assert config["param_ceiling_multiple"] is None
    assert torch.all(model(atomic_batch)[-1] > 0.0)


# ---------------------------------------------------------------------------
# Per-element seeding
# ---------------------------------------------------------------------------


def test_cliff_exch_per_element_table_covers_the_dataset_elements():
    table = mtp_mtp.CLIFF_EXCH_INITIAL_VALUES_BY_Z
    # H/C/N/O/F/S/Cl/Br are the elements the SAPT training sets contain.
    assert set(table) == {1, 6, 7, 8, 9, 16, 17, 35}
    assert all(value > 0.0 for value in table.values())
    # Hydrogen must be well below the scalar seed -- that asymmetry is the
    # entire point, since `K_i K_j` squares it on the most common pair.
    assert table[1] < mtp_mtp.CLIFF_EXCH_INITIAL_VALUES[0]
    assert table[1] < table[6] < table[7] < table[8] < table[9]


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"nope": {1: 1.0}}, "not one of"),
        ({"exch": [1.0]}, "mapping of Z to value"),
        ({"exch": {1: 0.0}}, "strictly positive"),
        ({"exch": {1: -1.0}}, "strictly positive"),
        ({"exch": {1: float("nan")}}, "strictly positive"),
        ({"exch": {200: 1.0}}, "outside"),
        ({"exch": {-1: 1.0}}, "outside"),
        ({"exch": {"h": 1.0}}, "atomic numbers"),
        ([("exch", {})], "mapping of parameter name"),
    ],
)
def test_cliff_head_rejects_invalid_per_element_table(
    bad, match, nested_hfvr_vw_model
):
    with pytest.raises(ValueError, match=match):
        _build_head(
            mtp_mtp.CliffExchangeNN,
            nested_hfvr_vw_model,
            param_start_mean_by_Z=bad,
        )


def test_cliff_head_per_element_table_survives_string_keys(
    nested_hfvr_vw_model, atomic_batch
):
    """A config round-trip through JSON stringifies integer keys."""
    model = _build_head(
        mtp_mtp.CliffExchangeNN,
        nested_hfvr_vw_model,
        param_start_std=[0.0],
        param_start_mean_by_Z={"exch": {"1": 0.5, "8": 4.0}},
    )
    assert model.param_start_mean_by_Z == {"exch": {1: 0.5, 8: 4.0}}
    _zero_readout_heads(model)
    parameters = model(atomic_batch)[-1]
    assert torch.allclose(
        parameters.reshape(-1),
        torch.tensor([4.0, 0.5, 0.5], dtype=parameters.dtype),
        atol=1e-5,
    )


def test_cliff_head_per_element_seeds_do_not_leak_across_columns(
    nested_hfvr_vw_model, atomic_batch
):
    """Seeding `exch` must leave the four Rackers columns on their scalars."""
    model = _build_head(
        mtp_mtp.CliffClassicalNN,
        nested_hfvr_vw_model,
        param_start_std=[0.0] * 5,
    )
    _zero_readout_heads(model)
    parameters = model(atomic_batch)[-1]
    rackers = parameters[:, : mtp_mtp.CLIFF_CLASSICAL_EXCH_INDEX]
    expected = torch.tensor(
        mtp_mtp.CLIFF_CLASSICAL_INITIAL_VALUES[
            : mtp_mtp.CLIFF_CLASSICAL_EXCH_INDEX
        ],
        dtype=parameters.dtype,
    )
    assert torch.allclose(rackers, expected.expand_as(rackers), atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient clipping plumbing
# ---------------------------------------------------------------------------


def test_train_declares_grad_clip_configuration():
    """`train_models.py` drops kwargs missing from the signature silently."""
    signature = inspect.signature(mtp_mtp.AM_DimerParam_Model.train)
    assert signature.parameters["grad_clip_norm"].default is None
    assert signature.parameters["grad_clip_mode"].default == "global"
    assert signature.parameters["thole_lr"].default is None
    assert signature.parameters["induction_diagnostics"].default is False
    assert signature.parameters["induction_convergence_threshold"].default is None
    assert signature.parameters["induction_max_iterations"].default is None
    inner = inspect.signature(mtp_mtp.AM_DimerParam_Model.single_proc_train)
    assert inner.parameters["grad_clip_norm"].default is None
    assert inner.parameters["grad_clip_mode"].default == "global"
    assert inner.parameters["thole_lr"].default is None
    assert inner.parameters["induction_diagnostics"].default is False


@pytest.mark.parametrize(
    "threshold,max_iterations",
    [(0.0, 50), (-1e-6, 50), (float("nan"), 50), (1e-6, 0),
     (1e-6, -1), (1e-6, 2.5), (1e-6, True)],
)
def test_validate_induction_solver_controls_rejects_invalid(
    threshold, max_iterations
):
    with pytest.raises(ValueError, match="induction_"):
        mtp_mtp._validate_induction_solver_controls(
            threshold, max_iterations
        )


def test_exchange_train_rejects_induction_solver_controls(
    nested_hfvr_vw_model
):
    harness = _build_cliff_harness(CliffExchangeModel, nested_hfvr_vw_model)
    with pytest.raises(ValueError, match="induction solver controls"):
        harness.train(
            induction_convergence_threshold=1e-6,
            induction_max_iterations=50,
        )


def test_validate_induction_solver_controls_accepts_profiled_values():
    assert mtp_mtp._validate_induction_solver_controls(1e-6, 50) == (
        1e-6,
        50,
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), "x"])
def test_validate_bound_scale_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="grad_clip_norm"):
        mtp_mtp._validate_bound_scale(bad, "grad_clip_norm")


def test_validate_bound_scale_allows_none_and_coerces_ints():
    assert mtp_mtp._validate_bound_scale(None, "grad_clip_norm") is None
    assert mtp_mtp._validate_bound_scale(2, "grad_clip_norm") == 2.0
    with pytest.raises(ValueError, match="grad_clip_norm"):
        mtp_mtp._validate_bound_scale(
            None, "grad_clip_norm", allow_none=False
        )


def test_ste_clamp_is_exact_at_large_magnitudes():
    """Regression: the bound must hold when `|x|` dwarfs it.

    The first implementation used `x - (x - upper).clamp_min(0).detach()`,
    which is algebraically correct but cancels: at `x = 3e7` in float32,
    `x - upper` rounds back toward `x` and the "clamped" result came out at 32
    instead of 25.  `test_cliff_head_ceiling_caps_a_runaway_readout` is what
    caught it; this pins the helper directly.
    """
    lower = torch.tensor([[-25.0]])
    upper = torch.tensor([[25.0]])
    raw = torch.tensor(
        [[-3.0e7], [-1.0e10], [3.0e7], [1.0e10], [3.4e7]], requires_grad=True
    )
    clamped = mtp_mtp._ste_clamp(raw, lower, upper)
    expected = raw.detach().clamp(min=-25.0, max=25.0)
    assert torch.equal(clamped.detach(), expected)
    clamped.sum().backward()
    assert torch.equal(raw.grad, torch.ones_like(raw))


# ---------------------------------------------------------------------------
# Readout initialization scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,_name,_parameter_names,_values,_stds",
    _HEAD_CASES,
    ids=_HEAD_IDS,
)
def test_readout_init_scale_shrinks_only_the_output_layer(
    model_type, _name, _parameter_names, _values, _stds, nested_hfvr_vw_model
):
    """Scaling must be linear in the knob, so only the output layer is touched.

    Scaling every layer of the four-deep readout stack would compound as
    ``s ** 4`` and make the configured number unreadable.
    """
    scale = 0.25
    # Both heads must draw the *same* random initialization, or nothing about
    # their weights is comparable.
    torch.manual_seed(0)
    unscaled = _build_head(
        model_type, copy.deepcopy(nested_hfvr_vw_model), readout_init_scale=None
    )
    torch.manual_seed(0)
    scaled = _build_head(
        model_type,
        copy.deepcopy(nested_hfvr_vw_model),
        readout_init_scale=scale,
    )
    assert scaled.readout_init_scale == scale
    assert unscaled.readout_init_scale is None

    checked_outputs = 0
    for head_u, head_s in zip(
        unscaled.param_readout_layers, scaled.param_readout_layers
    ):
        for readout_u, readout_s in zip(head_u, head_s):
            linears_u = [
                m for m in readout_u.modules() if isinstance(m, torch.nn.Linear)
            ]
            linears_s = [
                m for m in readout_s.modules() if isinstance(m, torch.nn.Linear)
            ]
            assert len(linears_u) == len(linears_s) > 1
            # Every layer but the last is untouched, bit for bit.
            for layer_u, layer_s in zip(linears_u[:-1], linears_s[:-1]):
                assert torch.equal(layer_u.weight, layer_s.weight)
                assert torch.equal(layer_u.bias, layer_s.bias)
            # The output layer is scaled exactly, weight and bias alike.
            assert torch.allclose(
                linears_s[-1].weight, scale * linears_u[-1].weight, atol=1e-7
            )
            assert torch.allclose(
                linears_s[-1].bias, scale * linears_u[-1].bias, atol=1e-7
            )
            assert linears_u[-1].weight.abs().max() > 0.0
            checked_outputs += 1
    assert checked_outputs == len(unscaled.PARAMETER_NAMES) * (
        unscaled.n_message + 1
    )


def test_readout_init_scale_shrinks_the_correction(
    nested_hfvr_vw_model, atomic_batch
):
    """A scaled readout must leave the emitted parameter nearer its seed."""
    seeds = torch.tensor(
        [mtp_mtp.CLIFF_EXCH_INITIAL_VALUES_BY_Z[int(z)] for z in atomic_batch.x],
        dtype=torch.get_default_dtype(),
    ).reshape(-1, 1)

    def deviation(scale):
        torch.manual_seed(3)
        model = _build_head(
            mtp_mtp.CliffExchangeNN,
            copy.deepcopy(nested_hfvr_vw_model),
            param_start_std=[0.0],
            readout_init_scale=scale,
            # The bounds would clamp both variants to the same ceiling and make
            # the comparison vacuous; this test is about the correction's size.
            param_floor_fraction=None,
            param_ceiling_multiple=None,
        )
        # Nudge the readout off its draw so there is a correction to shrink at
        # all, small enough to stay off the softplus tails.
        with torch.no_grad():
            for head in model.param_readout_layers:
                for readout in head:
                    for parameter in readout.parameters():
                        parameter.add_(0.02)
        with torch.no_grad():
            return (model(atomic_batch)[-1] - seeds).abs().mean().item()

    full = deviation(None)
    shrunk = deviation(0.1)
    assert full > 0.0
    assert shrunk < full, (shrunk, full)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), "x"])
def test_readout_init_scale_rejects_non_positive(bad, nested_hfvr_vw_model):
    with pytest.raises(ValueError, match="readout_init_scale"):
        _build_head(
            mtp_mtp.CliffExchangeNN,
            nested_hfvr_vw_model,
            readout_init_scale=bad,
        )


def test_cliff_head_overrides_distinguishes_none_from_unset():
    """`None` disables a feature; the sentinel means "use the head default"."""
    sentinel = mtp_mtp._CLIFF_HEAD_DEFAULT
    assert mtp_mtp._cliff_head_overrides(
        param_floor_fraction=sentinel, readout_init_scale=None
    ) == {"readout_init_scale": None}
    assert mtp_mtp._cliff_head_overrides(param_floor_fraction=sentinel) == {}
    assert mtp_mtp._cliff_head_overrides(param_floor_fraction=0.5) == {
        "param_floor_fraction": 0.5
    }


# ---------------------------------------------------------------------------
# Valence-width ceiling
#
# `S_ij` decays as `exp(-r / sqrt(sigma_i sigma_j))`, so an over-large width
# flattens the exponential and the pair energy explodes.  The frozen atom model
# emits sigma = 1.8952 for every Na atom in the training data (identical to four
# decimals, i.e. an untrained embedding bias, not a prediction) against 0.40-0.52
# for C/N/O, which produced single-dimer exchange predictions three orders of
# magnitude above reference.  The floor guards `B_ij -> inf`; these pin the
# guard on the opposite, far more damaging direction.
# ---------------------------------------------------------------------------


def test_overlap_ceiling_defaults_off_in_the_helper_and_on_in_exchange():
    """Legacy induction-overlap sites must keep their exact numerics."""
    assert (
        inspect.signature(mtp_mtp.atomic_overlap_S_ij)
        .parameters["width_ceiling"]
        .default
        is None
    )
    assert (
        inspect.signature(mtp_mtp.cliff_exchange)
        .parameters["width_ceiling"]
        .default
        == mtp_mtp.OVERLAP_WIDTH_CEILING
    )
    # The ceiling must sit above every width the atom model legitimately emits
    # and below the out-of-domain ones it invents.
    assert mtp_mtp.OVERLAP_WIDTH_FLOOR < mtp_mtp.OVERLAP_WIDTH_CEILING
    assert 0.75 < mtp_mtp.OVERLAP_WIDTH_CEILING < 1.8952


def test_overlap_ceiling_clamps_only_widths_above_it():
    src = torch.arange(4)
    tgt = torch.arange(4)
    dR = torch.full((4,), 6.0)
    # 1.8952 is the Na width the atom model actually emits; 0.70 is the largest
    # legitimate one observed (phosphorus).
    wide = torch.tensor([1.8952, 1.8952, 0.70, 0.45])
    narrow = torch.tensor([1.8952, 0.45, 0.70, 0.45])
    ceiling = mtp_mtp.OVERLAP_WIDTH_CEILING

    unbounded = mtp_mtp.atomic_overlap_S_ij(wide, narrow, src, tgt, dR)
    bounded = mtp_mtp.atomic_overlap_S_ij(
        wide, narrow, src, tgt, dR, width_ceiling=ceiling
    )
    reference = mtp_mtp.atomic_overlap_S_ij(
        wide.clamp_max(ceiling), narrow.clamp_max(ceiling), src, tgt, dR
    )
    assert torch.allclose(bounded, reference)
    # Edges whose widths are both already under the ceiling are untouched.
    assert torch.equal(bounded[2:], unbounded[2:])
    # The Na-Na edge is strongly suppressed.  At this separation (6 bohr) the
    # factor is ~6.7x; because the clamp acts inside the exponent it grows
    # without bound with distance, which the sweep below pins.
    assert bounded[0] < unbounded[0] / 5.0
    ratios = []
    for r in (6.0, 9.0, 12.0):
        dRr = torch.full((4,), r)
        u = mtp_mtp.atomic_overlap_S_ij(wide, narrow, src, tgt, dRr)
        g = mtp_mtp.atomic_overlap_S_ij(
            wide, narrow, src, tgt, dRr, width_ceiling=ceiling
        )
        ratios.append((u[0] / g[0]).item())
    assert ratios[0] < ratios[1] < ratios[2], ratios


def test_overlap_ceiling_suppresses_the_na_blowup_in_exchange():
    """A Na-like width must not produce a runaway per-edge exchange energy."""
    RA = torch.tensor([[0.0, 0.0, 0.0]])
    RB = torch.tensor([[3.2, 0.0, 0.0]])   # Angstrom, a close contact
    src = torch.zeros(1, dtype=torch.long)
    tgt = torch.zeros(1, dtype=torch.long)
    na_width = torch.tensor([1.8952])
    c_width = torch.tensor([0.5])
    K = torch.tensor([2.5])

    def exchange(width_ceiling):
        return mtp_mtp.cliff_exchange(
            RA=RA, RB=RB, e_AB_source=src, e_AB_target=tgt,
            valence_widths_A=na_width, valence_widths_B=c_width,
            K_exch_A=K, K_exch_B=K, width_ceiling=width_ceiling,
        ).item()

    unguarded = exchange(None)
    guarded = exchange(mtp_mtp.OVERLAP_WIDTH_CEILING)
    # Unguarded, a single Na-C edge at 3.2 A is already worth ~158 kcal/mol,
    # which is why a handful of Na dimers could carry 95% of the squared error.
    assert unguarded > 100.0, unguarded
    assert guarded < unguarded / 5.0, (guarded, unguarded)
    # Still strictly positive: the guard must not flip or zero the term.
    assert guarded > 0.0


def test_overlap_ceiling_leaves_covered_elements_untouched():
    """Widths for the elements CLIFF covers must pass through unchanged."""
    # Largest per-element mean widths the frozen model emits for Table I
    # elements, from the training data: S 0.591, C 0.521, N 0.453, O 0.404,
    # H 0.373, F 0.346.
    widths = torch.tensor([0.591, 0.521, 0.453, 0.404, 0.373, 0.346])
    assert torch.all(widths < mtp_mtp.OVERLAP_WIDTH_CEILING)
    src = torch.arange(widths.numel())
    tgt = torch.arange(widths.numel())
    dR = torch.full((widths.numel(),), 6.0)
    assert torch.equal(
        mtp_mtp.atomic_overlap_S_ij(widths, widths, src, tgt, dR),
        mtp_mtp.atomic_overlap_S_ij(
            widths, widths, src, tgt, dR,
            width_ceiling=mtp_mtp.OVERLAP_WIDTH_CEILING,
        ),
    )


# ---------------------------------------------------------------------------
# CLIFF Table I fidelity
# ---------------------------------------------------------------------------


def test_per_element_seeds_are_the_table_i_means():
    """Each seed must be the mean of that element's CLIFF Table I atom types."""
    table_i_atom_types = {
        1: (0.9890, 0.6910, 0.5996, 0.7909),   # HC, HN, HO, HS
        6: (2.2649, 2.4566, 2.8023),           # C4, C3, C2
        7: (4.4660, 4.6251, 3.4896),           # N3, N2, N1
        8: (5.8538, 5.3435),                   # O2, O1
        9: (7.6036,),                          # F
        16: (3.2842, 3.1773),                  # S2, S1
        17: (3.8152,),                         # Cl
        35: (4.1008,),                         # Br
    }
    seeds = mtp_mtp.CLIFF_EXCH_INITIAL_VALUES_BY_Z
    assert set(seeds) == set(table_i_atom_types)
    for z, types in table_i_atom_types.items():
        assert seeds[z] == pytest.approx(sum(types) / len(types), abs=1e-4), z
    assert mtp_mtp.CLIFF_TABLE_I_ELEMENTS == frozenset(seeds)
    # Na (11) and P (15) appear in the training data and are *not* covered.
    assert 11 not in mtp_mtp.CLIFF_TABLE_I_ELEMENTS
    assert 15 not in mtp_mtp.CLIFF_TABLE_I_ELEMENTS


def test_exchange_initialization_std_is_wide_enough_to_separate_atom_types():
    """`K_exch` spans 0.60-7.60, so a 0.01 raw std would be a delta function."""
    assert mtp_mtp.CLIFF_EXCH_INITIAL_STDS == (0.25,)
    assert (
        mtp_mtp.CLIFF_CLASSICAL_INITIAL_STDS[mtp_mtp.CLIFF_CLASSICAL_EXCH_INDEX]
        == 0.25
    )
    # The four Rackers columns keep their original, much tighter spread.
    assert mtp_mtp.CLIFF_CLASSICAL_INITIAL_STDS[
        : mtp_mtp.CLIFF_CLASSICAL_EXCH_INDEX
    ] == mtp_mtp.RACKERS_INITIAL_STDS


# ---------------------------------------------------------------------------
# Task-F: dataset element exclusion
#
# The frozen valence-width model cannot predict a monatomic ion's width -- a
# one-atom monomer has no intramolecular edges, so message passing contributes
# nothing -- and exchange goes as exp(-r / sqrt(sigma_i sigma_j)), so those
# atoms dominate the loss. `OVERLAP_WIDTH_CEILING` bounds the damage;
# `ds_exclude_elements` lets a run remove the elements outright instead.
# ---------------------------------------------------------------------------


class _FakeDimer:
    def __init__(self, za, zb):
        self.ZA = torch.tensor(za, dtype=torch.long)
        self.ZB = torch.tensor(zb, dtype=torch.long)


class _FakeDimerDataset:
    """Minimal stand-in exposing only what the filter is allowed to touch.

    Deliberately does NOT implement ``__getitem__``: the filter must go through
    ``get`` so it never triggers this dataset family's ``len()``, which globs
    the whole processed directory on every call.
    """

    def __init__(self, dimers):
        self._dimers = list(dimers)
        self.n_len_calls = 0
        self.n_get_calls = 0

    def len(self):
        self.n_len_calls += 1
        return len(self._dimers)

    def get(self, idx):
        self.n_get_calls += 1
        return self._dimers[idx]


def _mixed_dataset():
    #        0        1         2         3         4         5
    return _FakeDimerDataset([
        _FakeDimer([1, 6], [8]),        # clean
        _FakeDimer([11], [8, 1]),       # Na on side A
        _FakeDimer([6, 6], [17]),       # Cl on side B
        _FakeDimer([7, 1], [6, 8]),     # clean
        _FakeDimer([17, 11], [1]),      # both
        _FakeDimer([16, 1], [1, 1]),    # clean
    ])


def test_normalize_excluded_elements_accepts_none_scalar_and_iterable():
    assert mtp_mtp.normalize_excluded_elements(None) == frozenset()
    assert mtp_mtp.normalize_excluded_elements(11) == frozenset({11})
    assert mtp_mtp.normalize_excluded_elements([11, 17, 11]) == frozenset(
        {11, 17}
    )
    assert mtp_mtp.normalize_excluded_elements(
        np.array([11, 17])
    ) == frozenset({11, 17})
    assert mtp_mtp.normalize_excluded_elements(()) == frozenset()


def test_normalize_excluded_elements_rejects_symbols_and_bad_values():
    # Element symbols are the tempting spelling and the dangerous one: mapping
    # them here would let a typo become an empty exclusion set, which trains on
    # exactly the data the caller asked to drop.
    with pytest.raises(TypeError, match="not element symbols"):
        mtp_mtp.normalize_excluded_elements("Cl")
    with pytest.raises(TypeError, match="atomic numbers"):
        mtp_mtp.normalize_excluded_elements(["Cl"])
    with pytest.raises(TypeError, match="atomic numbers"):
        mtp_mtp.normalize_excluded_elements([11.0])
    with pytest.raises(TypeError, match="atomic numbers"):
        mtp_mtp.normalize_excluded_elements(True)
    with pytest.raises(ValueError, match=">= 1"):
        mtp_mtp.normalize_excluded_elements([0])
    with pytest.raises(ValueError, match=">= 1"):
        mtp_mtp.normalize_excluded_elements([-6])


def test_dimer_indices_excluding_elements_drops_either_monomer():
    ds = _mixed_dataset()
    keep = mtp_mtp.dimer_indices_excluding_elements(ds, [11, 17], print_level=0)
    assert keep == [0, 3, 5]


def test_dimer_indices_excluding_elements_single_element():
    ds = _mixed_dataset()
    assert mtp_mtp.dimer_indices_excluding_elements(
        ds, [11], print_level=0
    ) == [0, 2, 3, 5]
    assert mtp_mtp.dimer_indices_excluding_elements(
        ds, [17], print_level=0
    ) == [0, 1, 3, 5]


def test_dimer_indices_excluding_elements_empty_spec_is_plain_truncation():
    ds = _mixed_dataset()
    assert mtp_mtp.dimer_indices_excluding_elements(
        ds, None, print_level=0
    ) == [0, 1, 2, 3, 4, 5]
    assert mtp_mtp.dimer_indices_excluding_elements(
        ds, [], max_size=3, print_level=0
    ) == [0, 1, 2]
    # No scan at all when nothing is excluded.
    assert ds.n_get_calls == 0


def test_dimer_indices_excluding_elements_stops_at_max_size():
    ds = _mixed_dataset()
    keep = mtp_mtp.dimer_indices_excluding_elements(
        ds, [11, 17], max_size=2, print_level=0
    )
    assert keep == [0, 3]
    # Index 3 is the second survivor, so the scan must stop there rather than
    # walking the rest of the store. This is what makes a filtered subset cost
    # a scan proportional to the subset, not to the 1.5M-dimer set.
    assert ds.n_get_calls == 4


def test_dimer_indices_excluding_elements_warns_when_exhausted():
    ds = _mixed_dataset()
    with pytest.warns(RuntimeWarning, match="exhausted the dataset"):
        keep = mtp_mtp.dimer_indices_excluding_elements(
            ds, [11, 17], max_size=99, print_level=0
        )
    assert keep == [0, 3, 5]


def test_dimer_indices_excluding_elements_avoids_len_per_item():
    ds = _mixed_dataset()
    mtp_mtp.dimer_indices_excluding_elements(ds, [11, 17], print_level=0)
    # One len() for the loop bound; anything proportional to the item count
    # means the scan is routing through Dataset.__getitem__/indices() again.
    assert ds.n_len_calls == 1
    assert ds.n_get_calls == 6


def test_dimer_indices_excluding_elements_requires_za_zb():
    class _NoZ:
        def len(self):
            return 1

        def get(self, idx):
            return object()

    with pytest.raises(TypeError, match="ZA/ZB"):
        mtp_mtp.dimer_indices_excluding_elements(_NoZ(), [11], print_level=0)


def test_dimer_indices_excluding_elements_requires_get():
    class _NoGet:
        def len(self):
            return 0

    # An empty exclusion set never scans, so it must not need get().
    assert mtp_mtp.dimer_indices_excluding_elements(
        _NoGet(), None, print_level=0
    ) == []
    with pytest.raises(TypeError, match="no get\\(\\) method"):
        mtp_mtp.dimer_indices_excluding_elements(_NoGet(), [11], print_level=0)


def test_ds_exclude_elements_is_declared_on_the_model_constructor():
    sig = inspect.signature(mtp_mtp.AM_DimerParam_Model.__init__)
    assert "ds_exclude_elements" in sig.parameters
    assert sig.parameters["ds_exclude_elements"].default is None
    # The CLIFF routes reach it through **dataset_kwargs, so they must not
    # shadow it with a positional-only or differently-named parameter.
    for cls in (
        mtp_mtp.CliffExchangeModel,
        mtp_mtp.CliffClassicalModel,
        mtp_mtp.CliffClassicalOverlapModel,
    ):
        params = inspect.signature(cls.__init__).parameters
        assert any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ), cls.__name__
        assert "ds_exclude_elements" not in params, cls.__name__


def test_excluded_elements_are_recorded_for_tracking():
    """A filtered run must be distinguishable from a full one on the dashboard.

    The exclusion changes what the model is fit to, so a run config that omits
    it is not a record of the experiment.
    """
    src = inspect.getsource(mtp_mtp.AM_DimerParam_Model.train)
    assert '"data/excluded_elements"' in src
    assert '"training/grad_clip_norm"' in src
    assert '"training/grad_clip_mode"' in src
    assert '"training/thole_learning_rate"' in src
    assert '"training/induction_diagnostics"' in src
    assert '"training/component_gamma"' in src
    assert '"training/total_includes_d3"' in src
    # And the attribute the tracking config reads must be set by __init__ for
    # every route, not only when something is excluded.
    init_src = inspect.getsource(mtp_mtp.AM_DimerParam_Model.__init__)
    assert "self.ds_excluded_elements" in init_src


def test_exclude_scan_multiple_bounds_the_raw_dataset_cap():
    """Element exclusion must not turn into a full-store processing job.

    `max_size` on the fused dataset bounds not just the file list but how much
    of the raw pickle gets *processed* on first use. An earlier version passed
    None here so the scan could reach `ds_max_size` survivors, which on any
    machine whose processed store is not already built silently turns "give me
    5000 filtered dimers" into processing all 1.6M.
    """
    sig = inspect.signature(mtp_mtp.AM_DimerParam_Model.__init__)
    assert "ds_exclude_scan_multiple" in sig.parameters
    assert sig.parameters["ds_exclude_scan_multiple"].default == 2.0

    src = inspect.getsource(mtp_mtp.AM_DimerParam_Model.__init__)
    # One helper now derives the raw cap for each split, so the train and the
    # validation store are bounded by the same rule.
    assert "def _raw_cap(cap):" in src
    assert "math.ceil(cap * float(ds_exclude_scan_multiple))" in src
    assert "ds_raw_max_size = _raw_cap(ds_max_size)" in src
    assert "ds_raw_max_size_test = _raw_cap(ds_max_size_test)" in src
    # The reason has to survive in the source, or the next person "simplifies"
    # it back to None.
    assert "processing job" in src


def test_exclude_scan_multiple_rejects_bad_values():
    import math as _math

    bad_values = [0.99, 0.0, -1.0, float("nan"), float("inf")]
    for value in bad_values:
        with pytest.raises((ValueError,)):
            mtp_mtp._validate_scan_multiple_for_test(value)
    for value in ["2", None, True]:
        with pytest.raises(TypeError):
            mtp_mtp._validate_scan_multiple_for_test(value)
    # Valid values pass through.
    for value in (1, 1.0, 2.0, 10):
        assert mtp_mtp._validate_scan_multiple_for_test(value) == float(value)
    assert _math.isfinite(1.0)
