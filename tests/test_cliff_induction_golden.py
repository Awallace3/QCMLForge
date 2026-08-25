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


def test_intramolecular_edges_can_no_longer_flip_the_sign():
    """The regression this file was written to catch, now fixed.

    Before the fix, adding the AA edges turned this geometry from -0.498 to
    +0.346 kcal/mol, because they fed the monomer's own permanent field into
    `mu_induced_0`. The permanent driving field is now intermolecular by
    construction, so the AA edges reach only the induced-induced relay.

    That relay is *not* inert and should not be: a dipole induced on atom 0 by
    B propagates to atom 1 through `T2_AA`, which is real many-body
    polarization. So the energy still moves -- by 0.9% here, and deeper rather
    than shallower. What it can no longer do is change sign.
    """
    with_intra = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=True)
    ).sum().item()
    without = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=False)
    ).sum().item()
    assert with_intra < 0 and without < 0
    assert abs(with_intra - without) / abs(without) < 0.02
    assert with_intra == pytest.approx(-0.493529, rel=1e-5)


def test_the_pre_fix_construction_is_still_reachable_and_still_repulsive():
    """Kept so pre-fix checkpoints can be reproduced, not because it is right.

    Pinning the old number is what makes "the tainted runs used this" a
    checkable statement rather than a note in a log.
    """
    E = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=True),
        intramolecular_permanent_field=True,
    ).sum()
    assert E.item() == pytest.approx(+0.345847, rel=1e-4)


def test_the_size_of_the_defect_that_was_removed():
    """How large the unconstrained term was, on one unfavourable geometry.

    Recorded because it sets the scale of what changes in every pre-fix
    number: +0.844 kcal/mol on a single three-atom system, against an
    attractive -0.494 -- enough to reverse the sign on its own.
    """
    corrected = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=True)
    ).sum().item()
    pre_fix = mtp_mtp.rackers_thole_induction(
        **_two_atom_monomer_inputs(with_intramolecular=True),
        intramolecular_permanent_field=True,
    ).sum().item()
    assert pre_fix - corrected == pytest.approx(+0.839, abs=0.002)
    assert pre_fix > 0 > corrected


# ---------------------------------------------------------------------------
# Equivalence with the AP3-D3 path
#
# `dimer_induced_dipole_torch` is the induced-point-dipole implementation that
# is non-positive on all 528 S66x8 geometries. It uses one damping form
# (`thole_damping_torch`, the mutual `u**3`) for both the permanent and the
# induced tensors, so CLIFF2 must reproduce it exactly once the direct and
# mutual columns are equal and the direct exponent is 3 -- which is only a
# meaningful comparison because the `lambda_5` coefficient now follows the
# exponent. Anything left over is a difference between the two routes that
# nobody chose.
# ---------------------------------------------------------------------------


def _matched_dimer():
    """A water-dimer-like system with full intramolecular edge sets."""
    n_a, n_b = 3, 2
    torch.manual_seed(11)

    def intra(n):
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        source, target = zip(*pairs)
        return torch.tensor(source), torch.tensor(target)

    e_aa_source, e_aa_target = intra(n_a)
    e_bb_source, e_bb_target = intra(n_b)
    return n_a, n_b, dict(
        ZA=torch.tensor([8, 1, 1]),
        RA=torch.tensor(
            [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.3, 0.9, 0.0]], dtype=D
        ),
        qA=torch.tensor([-0.8, 0.4, 0.4], dtype=D),
        muA=torch.randn(n_a, 3, dtype=D) * 0.1,
        quadA=torch.zeros(n_a, 3, 3, dtype=D),
        ZB=torch.tensor([8, 1]),
        RB=torch.tensor([[0.2, 0.4, 3.1], [0.9, 1.2, 3.6]], dtype=D),
        qB=torch.tensor([-0.5, 0.5], dtype=D),
        muB=torch.randn(n_b, 3, dtype=D) * 0.1,
        quadB=torch.zeros(n_b, 3, 3, dtype=D),
        e_AB_source=torch.arange(n_a).repeat_interleave(n_b),
        e_AB_target=torch.arange(n_b).repeat(n_a),
        e_AA_source=e_aa_source,
        e_BB_source=e_bb_source,
        e_AA_target=e_aa_target,
        e_BB_target=e_bb_target,
        hirshfeld_volume_ratio_A=torch.tensor([0.9, 0.95, 0.95], dtype=D),
        hirshfeld_volume_ratio_B=torch.tensor([0.92, 0.97], dtype=D),
        valence_widths_A=torch.full((n_a,), SIGMA, dtype=D),
        valence_widths_B=torch.full((n_b,), SIGMA, dtype=D),
    )


def _cliff2_on(shared, n_a, n_b, **overrides):
    return mtp_mtp.rackers_thole_induction(
        **shared,
        thole_direct_A=torch.full((n_a,), A_THOLE, dtype=D),
        thole_direct_B=torch.full((n_b,), A_THOLE, dtype=D),
        thole_mutual_A=torch.full((n_a,), A_THOLE, dtype=D),
        thole_mutual_B=torch.full((n_b,), A_THOLE, dtype=D),
        ind_overlap_A=torch.full((n_a,), K_IND, dtype=D),
        ind_overlap_B=torch.full((n_b,), K_IND, dtype=D),
        thole_direct_exponent=3.0,
        convergence_threshold=1e-13,
        max_iterations=1000,
        **overrides,
    ).sum().item()


def _ap3d3_on(shared):
    import contextlib
    import io

    from apnet_pt import multipole

    # The function prints its intermediates unconditionally.
    with contextlib.redirect_stdout(io.StringIO()):
        return multipole.dimer_induced_dipole_torch(
            **shared, thole_damping_param=A_THOLE
        ).sum().item()


def test_corrected_cliff2_reproduces_the_ap3d3_induction_path():
    """Same physics, same numbers -- to the SCF's own convergence."""
    n_a, n_b, shared = _matched_dimer()
    assert _cliff2_on(shared, n_a, n_b) == pytest.approx(
        _ap3d3_on(shared), abs=1e-8
    )


def test_the_pre_fix_construction_disagreed_with_ap3d3_by_a_kcal_per_mol():
    """The size of the defect on a realistic system, not a toy.

    1.10 kcal/mol on one water-dimer-like pair, against an AP3-D3 answer of
    -0.117. Every induction number produced before the fix carries an error of
    this order, which is why they are all tainted rather than merely noisy.
    """
    n_a, n_b, shared = _matched_dimer()
    reference = _ap3d3_on(shared)
    pre_fix = _cliff2_on(
        shared, n_a, n_b, intramolecular_permanent_field=True
    )
    assert abs(pre_fix - reference) == pytest.approx(1.104, abs=0.01)
