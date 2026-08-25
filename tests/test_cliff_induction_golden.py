"""A golden value for the induction energy, computed by hand.

Exchange reached parity with CLIFF behind roughly twenty physics tests, one of
which -- ``test_golden_overlap_value`` -- pins its energy to a number derived
independently of the implementation. Induction had no equivalent: every test
either pinned current behaviour against its own past (``*_matches_pre_refactor``)
or asserted something a wrong answer also satisfies, like "finite and distinct".
That is how induction positive on 16 of 32 S66x8 geometries survived a green
suite, and it is why a sign error or a wrong prefactor could persist here
indefinitely.

The system below is the smallest one whose induced dipoles have a closed form:
one atom per monomer on the z axis, a charge on A, nothing permanent on B, and
no intramolecular edges. The self-consistent field is then a 2x2 linear system,
solved here from the published Thole forms rather than by calling the module,
so agreement is evidence and not a tautology.

Reference for the damping: AMOEBA+ (Liu, Piquemal, Ren, JCTC 2019) damps the
permanent field with ``u**1.5`` and leaves the mutual part at AMOEBA's ``u**3``.
"""

import math

import pytest
import torch

from apnet_pt import constants
from apnet_pt.AtomPairwiseModels import mtp_mtp
from apnet_pt.AtomPairwiseModels.mtp_mtp import atomic_overlap_S_ij

D = torch.float64

R_BOHR = 5.0
Q_A = 0.7
A_THOLE = mtp_mtp.CLIFF_THOLE_SMEARING
Z = 8  # oxygen
SIGMA = 0.39
K_IND = mtp_mtp.CLIFF_IND_OVERLAP_SEED

# Hand-computed below in `_closed_form`, at Hirshfeld volume ratio 1 so that
# alpha is the table value exactly. Attractive, as induction must be.
GOLDEN_E_IND_KCAL = -0.928123137929


def _inputs(r_bohr=R_BOHR, q_A=Q_A, **overrides):
    empty = torch.tensor([], dtype=torch.long)
    zeros3 = torch.zeros((1, 3), dtype=D)
    one = torch.ones(1, dtype=D)
    kwargs = dict(
        ZA=torch.tensor([Z]),
        RA=torch.tensor([[0.0, 0.0, 0.0]], dtype=D),
        qA=torch.tensor([q_A], dtype=D),
        muA=zeros3,
        quadA=torch.zeros((1, 3, 3), dtype=D),
        ZB=torch.tensor([Z]),
        RB=torch.tensor([[0.0, 0.0, r_bohr * constants.au2ang]], dtype=D),
        qB=torch.tensor([0.0], dtype=D),
        muB=zeros3,
        quadB=torch.zeros((1, 3, 3), dtype=D),
        e_AB_source=torch.tensor([0]),
        e_AB_target=torch.tensor([0]),
        e_AA_source=empty,
        e_BB_source=empty,
        e_AA_target=empty,
        e_BB_target=empty,
        hirshfeld_volume_ratio_A=one,
        hirshfeld_volume_ratio_B=one,
        valence_widths_A=torch.tensor([SIGMA], dtype=D),
        valence_widths_B=torch.tensor([SIGMA], dtype=D),
        thole_direct_A=torch.tensor([A_THOLE], dtype=D),
        thole_direct_B=torch.tensor([A_THOLE], dtype=D),
        thole_mutual_A=torch.tensor([A_THOLE], dtype=D),
        thole_mutual_B=torch.tensor([A_THOLE], dtype=D),
        ind_overlap_A=torch.tensor([K_IND], dtype=D),
        ind_overlap_B=torch.tensor([K_IND], dtype=D),
        # Tight, because this test compares against an exact fixed point rather
        # than against another approximate solve.
        convergence_threshold=1e-14,
        max_iterations=500,
    )
    kwargs.update(overrides)
    return kwargs


def _thole(u, n):
    """Thole damping from rho(u) ~ exp(-a u**n). Written from the papers."""
    au = A_THOLE * u**n
    lam3 = 1 - math.exp(-au)
    lam5 = 1 - (1 + (n / 3.0) * au) * math.exp(-au)
    return lam3, lam5


def _closed_form(r_bohr=R_BOHR, q_A=Q_A):
    """E_ind for one charge and one polarizable atom, solved exactly.

    Along the axis both induced dipoles are parallel to z, so the SCF reduces
    to two scalar equations:

        mu_A = alpha_A * C * mu_B
        mu_B = alpha_B * lam3_direct * q_A / r**2  +  alpha_B * C * mu_A

    with ``C = T2_zz = (3 lam5_mutual - lam3_mutual) / r**3`` the axial
    component of the mutual dipole-dipole tensor. Eliminating mu_A gives the
    geometric series below in closed form. The energy is the charge-induced
    dipole contraction, halved.
    """
    alpha = constants.polarizability_table[Z].item()
    u = r_bohr / ((alpha * alpha) ** (1.0 / 6.0))
    lam3_direct, _ = _thole(u, 1.5)
    lam3_mutual, lam5_mutual = _thole(u, 3.0)
    C = (3 * lam5_mutual - lam3_mutual) / r_bohr**3
    mu_B = (alpha * lam3_direct * q_A / r_bohr**2) / (1 - alpha * alpha * C**2)
    E_qu = -(lam3_direct / r_bohr**2) * q_A * mu_B * constants.h2kcalmol
    return E_qu / 2.0


def test_golden_induction_value():
    """The number, and the derivation that produced it, must both hold.

    Pinning only the literal would let a future refactor of the closed form
    drift; pinning only the closed form would let both drift together.
    """
    assert _closed_form() == pytest.approx(GOLDEN_E_IND_KCAL, rel=1e-9)
    E = mtp_mtp.rackers_thole_induction(**_inputs())
    assert E.item() == pytest.approx(GOLDEN_E_IND_KCAL, rel=1e-9)


def test_the_golden_value_is_attractive_and_of_a_physical_size():
    """Guards against a sign flip that a `pytest.approx` refresh would absorb."""
    assert GOLDEN_E_IND_KCAL < 0
    assert 0.1 < abs(GOLDEN_E_IND_KCAL) < 10.0


def test_dropping_the_mutual_coupling_recovers_minus_half_alpha_e_squared():
    """The textbook limit, as an independent check on the closed form itself.

    With the induced-induced term switched off the answer must be exactly
    `-1/2 alpha |E|^2`, which involves none of this module's machinery. The
    mutual coupling is worth about 0.7% here, so the two agree closely without
    being the same expression -- a sign or factor-of-two error would not.
    """
    alpha = constants.polarizability_table[Z].item()
    u = R_BOHR / ((alpha * alpha) ** (1.0 / 6.0))
    lam3_direct, _ = _thole(u, 1.5)
    field = lam3_direct * Q_A / R_BOHR**2
    textbook = -0.5 * alpha * field**2 * constants.h2kcalmol
    assert textbook == pytest.approx(GOLDEN_E_IND_KCAL, rel=0.01)
    assert textbook > GOLDEN_E_IND_KCAL  # mutual coupling deepens the well


def test_the_half_factor_is_exactly_a_factor_of_two():
    """CLIFF Eq. (19) carries no half. The flag must change only that."""
    halved = mtp_mtp.rackers_thole_induction(**_inputs()).item()
    whole = mtp_mtp.rackers_thole_induction(
        **_inputs(energy_half_factor=False)
    ).item()
    assert whole == pytest.approx(2.0 * halved, rel=1e-12)


def test_induction_deepens_monotonically_as_the_monomers_approach():
    """The physical acceptance criterion, on a system with no confounders.

    IPD + overlap is not expected to reproduce SAPT0 induction, so the S66x8
    gate asks only that induction be attractive everywhere and deepen as the
    monomers close. Here that must hold exactly: one charge polarizing one
    atom, with no basis-set or geometry effects to appeal to.
    """
    distances = [3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
    energies = [
        mtp_mtp.rackers_thole_induction(**_inputs(r_bohr=r)).item()
        for r in distances
    ]
    assert all(e < 0 for e in energies), energies
    assert all(
        a < b for a, b in zip(energies, energies[1:])
    ), energies  # deeper at short range


def test_the_overlap_term_is_attractive_and_converted_once_to_kcal_mol():
    """`- K_i S_ij K_j` in hartree, times h2kcalmol, and nothing else.

    Asserted as the *difference* the flag makes, so it isolates the overlap
    term from the polarization energy it is added to. `atomic_overlap_S_ij` is
    already pinned to a hand computation by the exchange suite, so using it
    here anchors this term's sign, its bilinearity in K, and the single unit
    conversion -- the three things that would let a correct-looking overlap
    contribute nothing at short range.
    """
    without = mtp_mtp.rackers_thole_induction(**_inputs()).item()
    with_overlap = mtp_mtp.rackers_thole_induction(
        **_inputs(include_overlap=True)
    ).item()
    S_ij = atomic_overlap_S_ij(
        torch.tensor([SIGMA], dtype=D),
        torch.tensor([SIGMA], dtype=D),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([R_BOHR], dtype=D),
        width_floor=0.0,
    ).item()
    expected = -K_IND * S_ij * K_IND * constants.h2kcalmol
    assert with_overlap - without == pytest.approx(expected, rel=1e-12)
    assert expected < 0  # attractive, like the polarization it adds to
    # Bilinear in K: doubling both ends quadruples the term.
    doubled = mtp_mtp.rackers_thole_induction(
        **_inputs(
            include_overlap=True,
            ind_overlap_A=torch.tensor([2 * K_IND], dtype=D),
            ind_overlap_B=torch.tensor([2 * K_IND], dtype=D),
        )
    ).item()
    assert doubled - without == pytest.approx(4.0 * expected, rel=1e-12)


def test_the_overlap_terms_share_of_induction_grows_as_the_atoms_approach():
    """Exponential against a power law: the ratio must rise monotonically.

    This is the derivable statement. Polarization falls off as r^-4 and the
    overlap term falls off exponentially, so the overlap's share of induction
    has to increase as the atoms close, whatever K happens to be. Nothing here
    pins *how large* that share is -- that is the next test, and it is a
    property of the seed rather than of the functional form.
    """
    def share(r):
        pol = mtp_mtp.rackers_thole_induction(**_inputs(r_bohr=r)).item()
        total = mtp_mtp.rackers_thole_induction(
            **_inputs(r_bohr=r, include_overlap=True)
        ).item()
        return abs(total - pol) / abs(pol)

    shares = [share(r) for r in (6.0, 5.0, 4.0, 3.0, 2.0, 1.5)]
    assert all(a < b for a, b in zip(shares, shares[1:])), shares


@pytest.mark.parametrize(
    "r_bohr,expected_share",
    [(5.0, 0.006), (4.0, 0.023), (3.0, 0.093), (2.0, 0.306), (1.5, 0.444)],
)
def test_at_the_seed_the_overlap_term_is_a_small_correction(
    r_bohr, expected_share
):
    """Measured, not asserted from intuition -- and worth knowing.

    `CLIFF_IND_OVERLAP_SEED` was lowered 1.8 -> 0.2 because 1.8 over-polarizes:
    at 1.8 this same pair gets -425 kcal/mol of overlap at 1.5 bohr against
    -11.8 of polarization, which is not a correction to anything. But 0.2 lands
    on the other side. On one O-O pair the overlap term is under 1% of induction
    at 5 bohr and under 10% at 3 bohr, so a run that fits the overlap alone
    starts with a term that is nearly inert at contact distance and has to move
    K a long way before it can matter.

    Pinned so that a future seed change has to acknowledge this, rather than
    silently making the term inert again. These are shares of a single pair;
    a real dimer sums many, and K is fitted rather than fixed.
    """
    pol = mtp_mtp.rackers_thole_induction(**_inputs(r_bohr=r_bohr)).item()
    total = mtp_mtp.rackers_thole_induction(
        **_inputs(r_bohr=r_bohr, include_overlap=True)
    ).item()
    assert abs(total - pol) / abs(pol) == pytest.approx(
        expected_share, abs=0.001
    )


# ---------------------------------------------------------------------------
# Why induction comes out repulsive on most of S66x8
#
# The S66x8 gate on the two finished 50-epoch arms put `cliff2_ind_ipd` -- the
# polarization energy with the overlap term switched off -- positive on 421 of
# 528 geometries, maximum +4.36 kcal/mol, while `cliff2_ind_overlap` was
# non-positive on all 528. So the overlap term is not the problem; the IPD
# energy is. The same panel puts `ap3d3_classical_ind`, this repository's other
# induced-point-dipole path, non-positive on all 528.
#
# The two paths differ in one place. `dimer_induced_dipole_torch` (AP3-D3)
# builds `mu_induced_0` from the intermolecular edges alone.
# `_rackers_initial_permanent_fields` (CLIFF2) additionally accumulates the AA
# and BB permanent fields, so its induced dipoles include the polarization each
# monomer already had in isolation -- and then the energy contracts those
# dipoles over intermolecular edges only. The monomer-intrinsic part of mu
# contracted with the other monomer's field is not an induction energy and
# carries no sign constraint.
#
# The pair below is the minimal demonstration: identical geometry, identical
# parameters, differing only in whether the intramolecular edges are supplied.
# ---------------------------------------------------------------------------

_INTRA_Q = (0.9, -0.9)
# Bohr, like every other length in this file; positions are converted to the
# Angstrom the function expects exactly once, at the boundary.
_INTRA_SEP_BOHR = 1.2
_INTRA_R_BOHR = 4.5


def _two_atom_monomer_inputs(*, with_intramolecular):
    empty = torch.tensor([], dtype=torch.long)
    intra_source = torch.tensor([0, 1])
    intra_target = torch.tensor([1, 0])
    return dict(
        ZA=torch.tensor([Z, Z]),
        RA=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, _INTRA_SEP_BOHR * constants.au2ang]],
            dtype=D,
        ),
        qA=torch.tensor(_INTRA_Q, dtype=D),
        muA=torch.zeros((2, 3), dtype=D),
        quadA=torch.zeros((2, 3, 3), dtype=D),
        ZB=torch.tensor([Z]),
        RB=torch.tensor(
            [[0.0, 0.0, _INTRA_R_BOHR * constants.au2ang]], dtype=D
        ),
        qB=torch.tensor([0.0], dtype=D),
        muB=torch.zeros((1, 3), dtype=D),
        quadB=torch.zeros((1, 3, 3), dtype=D),
        e_AB_source=torch.tensor([0, 1]),
        e_AB_target=torch.tensor([0, 0]),
        e_AA_source=intra_source if with_intramolecular else empty,
        e_AA_target=intra_target if with_intramolecular else empty,
        e_BB_source=empty,
        e_BB_target=empty,
        hirshfeld_volume_ratio_A=torch.ones(2, dtype=D),
        hirshfeld_volume_ratio_B=torch.ones(1, dtype=D),
        valence_widths_A=torch.full((2,), SIGMA, dtype=D),
        valence_widths_B=torch.full((1,), SIGMA, dtype=D),
        thole_direct_A=torch.full((2,), A_THOLE, dtype=D),
        thole_direct_B=torch.full((1,), A_THOLE, dtype=D),
        thole_mutual_A=torch.full((2,), A_THOLE, dtype=D),
        thole_mutual_B=torch.full((1,), A_THOLE, dtype=D),
        ind_overlap_A=torch.full((2,), K_IND, dtype=D),
        ind_overlap_B=torch.full((1,), K_IND, dtype=D),
        convergence_threshold=1e-13,
        max_iterations=500,
    )


def test_the_intermolecular_field_alone_gives_attractive_induction():
    """The AP3-D3 construction, on the geometry that breaks the other one."""
    E = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=False)
    ).sum()
    assert E.item() == pytest.approx(-0.498250, rel=1e-4)
    assert E.item() < 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect: mu_induced_0 accumulates the AA/BB permanent fields "
        "while the energy contracts only AB edges, so the monomers' intrinsic "
        "polarization enters an energy that has no sign constraint. This is "
        "the mechanism behind cliff2_ind_ipd being positive on 421 of 528 "
        "S66x8 geometries. Remove this marker with the fix -- strict=True "
        "means it fails loudly once induction here becomes attractive."
    ),
)
def test_induction_must_be_attractive_with_intramolecular_fields_too():
    """Same geometry and parameters; only the AA edges are added.

    Attractive without them (-0.498 kcal/mol), repulsive with them (+0.346).
    Nothing about adding a monomer's own internal field to its own dipole solve
    should be able to turn an interaction induction repulsive.
    """
    E = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=True)
    ).sum()
    assert E.item() < 0, f"induction is repulsive: {E.item():+.6f} kcal/mol"


def test_the_intramolecular_contribution_is_repulsive_and_this_is_its_size():
    """Pinned so the fix has something quantitative to move.

    Not a tolerance on a physical quantity -- a record of how large the
    unconstrained term is on a deliberately unfavourable geometry, so a change
    to the functional can be checked against it rather than against intuition.
    """
    without = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=False)
    ).sum().item()
    with_intra = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=True)
    ).sum().item()
    assert with_intra - without == pytest.approx(+0.844, abs=0.002)
    assert with_intra > 0 > without
