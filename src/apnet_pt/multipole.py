"""
Functions for evaluating electrostatics between atom-centered multipoles
"""

from importlib import resources
import pandas as pd
import numpy as np
from . import constants
import torch
from typing import Tuple
import qcelemental as qcel
from .util import scatter_sum_compile
# from .AtomPairwiseModels.mtp_mtp import get_distances


def proc_molden(name):
    """Get coordinates (a.u.) and atom types from a molden.
    Accounts for ghost atoms"""

    with open(name, "r") as fp:
        data = fp.read().split("[GTO]")[0].strip()
    data = data.split("\n")[2:]
    data = [line.strip().split() for line in data]
    Z = [line[0] for line in data]

    try:
        Z = [constants.elem_to_z[elem] for elem in Z]
    except:
        print(name)
        return 0, 0
    Z = np.array(Z, dtype=np.int64)

    R = [line[3:] for line in data]
    R = [[float(xyz) for xyz in line] for line in R]
    R = np.array(R, dtype=np.float64)

    mask = [line[2] for line in data]
    mask = [(d != "0") for d in mask]

    return R[mask], Z[mask]


def make_quad_np(flat_quad):
    natom = flat_quad.shape[0]
    full_quad = np.zeros((natom, 3, 3))
    full_quad[:, 0, 0] = flat_quad[:, 0]  # xx
    full_quad[:, 0, 1] = flat_quad[:, 1]  # xy
    full_quad[:, 1, 0] = flat_quad[:, 1]  # xy
    full_quad[:, 0, 2] = flat_quad[:, 2]  # xz
    full_quad[:, 2, 0] = flat_quad[:, 2]  # xz
    full_quad[:, 1, 1] = flat_quad[:, 3]  # yy
    full_quad[:, 1, 2] = flat_quad[:, 4]  # yz
    full_quad[:, 2, 1] = flat_quad[:, 4]  # yz
    full_quad[:, 2, 2] = flat_quad[:, 5]  # zz

    trace = full_quad[:, 0, 0] + full_quad[:, 1, 1] + full_quad[:, 2, 2]

    full_quad[:, 0, 0] -= trace / 3.0
    full_quad[:, 1, 1] -= trace / 3.0
    full_quad[:, 2, 2] -= trace / 3.0

    return full_quad


def qpole_redundant(unique):
    assert len(unique) == 6
    redundant = np.zeros((3, 3))

    redundant[0, 0] = unique[0]
    redundant[0, 1] = unique[1]
    redundant[1, 0] = unique[1]
    redundant[0, 2] = unique[2]
    redundant[2, 0] = unique[2]
    redundant[1, 1] = unique[3]
    redundant[1, 2] = unique[4]
    redundant[2, 1] = unique[4]
    redundant[2, 2] = unique[5]
    return redundant


def ensure_traceless_qpole(qpole):
    # get device of qpole
    qpole_mask = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=qpole.dtype,
        device=qpole.device,
    )
    trace = qpole * qpole_mask
    trace = torch.sum(trace, dim=[1, 2], keepdim=True) / 3.0
    trace = qpole_mask * trace
    qpole = qpole - trace
    return qpole


def qpole_expand_and_traceless(qpole):
    qpole = torch.tensor(qpole_redundant(qpole))
    qpole = ensure_traceless_qpole(qpole)
    return qpole


def ensure_traceless_qpole_torch(qpole):
    qpoles = []
    for i in range(qpole.shape[0]):
        qpoles.append(qpole_expand_and_traceless(qpole[i]).view(3, 3))
    qpole = torch.stack(qpoles)
    qpole = torch.cat((qpole[:, 0, :], qpole[:, 1, 1:], qpole[:, 2, 2:]), dim=1)
    return qpole


def compact_multipoles_to_charge_dipole_qpoles(multipoles):
    charges = multipoles[:, 0]
    dipoles = multipoles[:, 1:4]
    qpoles = multipoles[:, 4:10]
    qpoles_out = []
    for i in range(qpoles.shape[0]):
        # qpoles_out.append(qpole_redundant(qpoles[i]))
        qpoles_out.append(qpole_expand_and_traceless(qpoles[i]))
    qpoles = np.array(qpoles_out)
    return charges, dipoles, qpoles


def charge_dipole_qpoles_to_compact_multipoles(charges, dipoles, qpoles):
    multipoles = np.zeros((charges.shape[0], 10))
    multipoles[:, 0] = charges
    multipoles[:, 1:4] = dipoles
    for i in range(qpoles.shape[0]):
        multipoles[i, 4:10] = qpoles[i].flatten()[[0, 1, 2, 4, 5, 8]]
    return multipoles


def T_cart_Z_MTP(RA, RB, alpha_j=None):
    lam_1, lam_3, lam_5 = (1.0, 1.0, 1.0)

    dR = RB - RA
    R = np.linalg.norm(dR)

    if alpha_j is not None:
        lam_1, lam_3, lam_5 = elst_damping_z_mtp(alpha_j, R)

    delta = np.identity(3)
    T0 = R**-1 * lam_1
    T1 = (R**-3) * (-1.0 * dR) * lam_3
    T2 = (R**-5) * (lam_5 * 3 * np.outer(dR, dR) - lam_3 * R * R * delta)
    return T0, T1, T2


def T_cart(RA, RB, alpha_i=None, alpha_j=None):
    lam_1, lam_3, lam_5, lam_7, lam_9 = (1.0, 1.0, 1.0, 1.0, 1.0)

    dR = RB - RA
    R = np.linalg.norm(dR)
    if alpha_i is not None and alpha_j is not None:
        lam_1, lam_3, lam_5, lam_7, lam_9 = elst_damping_mtp_mtp(alpha_i, alpha_j, R)
    delta = np.identity(3)
    # E_qq
    T0 = R**-1 * lam_1
    # E_qu
    T1 = (R**-3) * (-1.0 * dR) * lam_3
    T2 = (R**-5) * (lam_5 * 3 * np.outer(dR, dR) - lam_3 * R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    lam_5_const = np.ones((3, 3, 3)) * 3
    lam_5_const *= lam_5
    T3 = (R**-7) * (
        -15 * lam_7 * np.multiply.outer(np.outer(dR, dR), dR)
        + lam_5_const * R * R * (Rdd + Rdd.transpose(1, 0, 2) + Rdd.transpose(2, 0, 1))
    )
    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    # Used for E_QQ
    T4 = (R**-9) * (
        105 * lam_9 * np.multiply.outer(np.outer(dR, dR), np.outer(dR, dR))
        - 15
        * lam_7
        * R
        * R
        * (
            RRdd
            + RRdd.transpose(0, 2, 1, 3)
            + RRdd.transpose(0, 3, 2, 1)
            + RRdd.transpose(2, 1, 0, 3)
            + RRdd.transpose(3, 1, 2, 0)
            + RRdd.transpose(2, 3, 0, 1)
        )
        + 3
        * lam_5
        * (R**4)
        * (dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1))
    )

    return T0, T1, T2, T3, T4


def thole_damping_direct(r_ij, alpha_i, alpha_j, a):
    """
    Apply Thole damping to interaction tensor from AMOEBA+
    https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.7b00225/suppl_file/ct7b00225_si_001.pdf
    """
    # Compute damping factor
    u = r_ij / ((alpha_i * alpha_j) ** (1.0 / 6.0))
    au3 = a * (u**(3/2))
    l3 = 1 - np.exp(-au3)
    l5 = 1 - (1 + 0.5 * au3) * np.exp(-au3)
    l7 = 1 - (1.0 + 39/60 * au3 + 9/60 * au3**2) * np.exp(-au3)
    l9 = 1 - (1 + 609 / 840 * au3 + (189 / 840 * au3**2 + 27/840 * au3**3)) * np.exp(-au3)
    return au3, l3, l5, l7, l9


def thole_damping_mutual(r_ij, alpha_i, alpha_j, a):
    """Apply Thole damping to interaction tensor"""
    # Compute damping factor
    u = r_ij / ((alpha_i * alpha_j) ** (1.0 / 6.0))
    # print(f"{u=:.2f}, {r_ij:.2f}, {alpha_i:.2f}, {alpha_i:.2f}")
    au3 = a * (u**3)
    l3 = 1 - np.exp(-au3)
    l5 = 1 - (1 + au3) * np.exp(-au3)
    l7 = 1 - (1.0 + au3 + 0.6 * au3**2) * np.exp(-au3)
    l9 = 1 - (1 + au3 + (18 * au3**2 + 9 * au3**3) / 35) * np.exp(-au3)
    return au3, l3, l5, l7, l9


def elst_damping_mtp_mtp(alpha_i, alpha_j, r):
    """
    # MTP-MTP interaction from CLIFF
    """
    r2 = r**2
    r3 = r2 * r
    r4 = r2**2
    r5 = r4 * r
    a1_2 = alpha_i * alpha_i
    a2_2 = alpha_j * alpha_j
    a1_3 = a1_2 * alpha_i
    a2_3 = a2_2 * alpha_j
    a1_4 = a1_3 * alpha_i
    a2_4 = a2_3 * alpha_j
    e1r = np.exp(-1.0 * alpha_i * r)
    e2r = np.exp(-1.0 * alpha_j * r)
    lam1, lam3, lam5, lam7, lam9 = (1.0, 1.0, 1.0, 1.0, 1.0)
    if abs(alpha_i - alpha_j) > 1e-6:
        A = a2_2 / (a2_2 - a1_2)
        B = a1_2 / (a1_2 - a2_2)
        lam1 -= A * e1r
        lam1 -= B * e2r
        lam3 -= (1.0 + alpha_i * r) * A * e1r
        lam3 -= (1.0 + alpha_j * r) * B * e2r

        lam5 -= (1.0 + alpha_i * r + (1.0 / 3.0) * a1_2 * r2) * A * e1r
        lam5 -= (1.0 + alpha_j * r + (1.0 / 3.0) * a2_2 * r2) * B * e2r

        lam7 -= (
            (1.0 + alpha_i * r + (2.0 / 5.0) * a1_2 * r2 + (1.0 / 15.0) * a1_3 * r3)
            * A
            * e1r
        )
        lam7 -= (
            (1.0 + alpha_j * r + (2.0 / 5.0) * a2_2 * r2 + (1.0 / 15.0) * a2_3 * r3)
            * B
            * e2r
        )

        lam9 -= (
            (
                1.0
                + alpha_i * r
                + (3.0 / 7.0) * a1_2 * r2
                + (2.0 / 21.0) * a1_3 * r3
                + (1.0 / 105.0) * a1_4 * r4
            )
            * A
            * e1r
        )
        lam9 -= (
            (
                1.0
                + alpha_j * r
                + (3.0 / 7.0) * a2_2 * r2
                + (2.0 / 21.0) * a2_3 * r3
                + (1.0 / 105.0) * a2_4 * r4
            )
            * B
            * e2r
        )

    else:
        lam1 -= (1.0 + 0.5 * alpha_i * r) * e1r
        lam3 -= (1.0 + alpha_i * r + 0.5 * a1_2 * r2) * e1r
        lam5 -= (1.0 + alpha_i * r + 0.5 * a1_2 * r2 + (1.0 / 6.0) * a1_3 * r3) * e1r
        lam7 -= (
            1.0
            + alpha_i * r
            + 0.5 * a1_2 * r2
            + (1.0 / 6.0) * a1_3 * r3
            + (1.0 / 30.0) * a1_4 * r4
        ) * e1r
        lam9 -= (
            1.0
            + alpha_i * r
            + 0.5 * a1_2 * r2
            + (1.0 / 6.0) * a1_3 * r3
            + (4.0 / 105.0) * a1_4 * r4
            + (1.0 / 210.0) * a1_4 * alpha_i * r5
        ) * e1r
    return lam1, lam3, lam5, lam7, lam9


def elst_damping_z_mtp(alpha_j, r):
    """
    # Z-MTP interaction from CLIFF
    lam_1 = 1.0 - np.exp(-1.0 * np.multiply(alpha2,r))
    lam_3 = 1.0 - (1.0 + np.multiply(alpha2,r)) * np.exp(-1.0*np.multiply(alpha2,r))
    lam_5 = 1.0 - (1.0 + np.multiply(alpha2,r) + (1.0/3.0)*np.multiply(np.square(alpha2),r2)) * np.exp(-1.0*np.multiply(alpha2,r))
    # TODO: remove the damping by setting all lam* = 1.0
    if damping == False:
        lam_1 = 1.0
        lam_3 = 1.0
        lam_5 = 1.0
    """
    # Z-MTP interaction from CLIFF
    lam_1 = 1.0 - np.exp(-1.0 * np.multiply(alpha_j, r))
    lam_3 = 1.0 - (1.0 + np.multiply(alpha_j, r)) * np.exp(
        -1.0 * np.multiply(alpha_j, r)
    )
    lam_5 = 1.0 - (
        1.0
        + np.multiply(alpha_j, r)
        + (1.0 / 3.0) * np.multiply(np.square(alpha_j), r**2)
    ) * np.exp(-1.0 * np.multiply(alpha_j, r))
    return lam_1, lam_3, lam_5


def T_cart_Thole_damping(RA, RB, alpha_i, alpha_j, a, damping_term="mutual"):
    dR = RB - RA
    R = np.linalg.norm(dR)

    delta = np.identity(3)

    if damping_term == "direct":
        au3, l3, l5, l7, l9 = thole_damping_direct(R, alpha_i, alpha_j, a)
    elif damping_term == "mutual":
        au3, l3, l5, l7, l9 = thole_damping_mutual(R, alpha_i, alpha_j, a)
    else:
        raise ValueError("damping_term must be 'direct' or 'mutual'")

    # l3, l5, l7, l9 = (1.0, 1.0, 1.0, 1.0)  # Turn off Thole damping
    # print(f"   {l3 = }")

    # l3 = np.ones_like(l3)
    # l5 = np.ones_like(l5)
    # print(f"{alpha_i:.2f}-{alpha_j:.2f} {l3 = }, {l5 = }")
    T0 = R**-1
    # print(f"{damping_term}, {l3=:.2f}")
    T1 = l3 * (R**-3) * (-1.0 * dR)
    T2 = (R**-5) * (l5 * 3 * np.outer(dR, dR) - l3 * R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    T3 = (
        (R**-7)
        * -1.0
        * (
            l7 * 15 * np.multiply.outer(np.outer(dR, dR), dR)
            - l5 * 3 * R * R * (Rdd + Rdd.transpose(1, 0, 2) + Rdd.transpose(2, 0, 1))
        )
    )

    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    T4 = (R**-9) * (
        l9 * 105 * np.multiply.outer(np.outer(dR, dR), np.outer(dR, dR))
        - l7
        * 15
        * R
        * R
        * (
            RRdd
            + RRdd.transpose(0, 2, 1, 3)
            + RRdd.transpose(0, 3, 2, 1)
            + RRdd.transpose(2, 1, 0, 3)
            + RRdd.transpose(3, 1, 2, 0)
            + RRdd.transpose(2, 3, 0, 1)
        )
        + l5
        * 3
        * (R**4)
        * (dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1))
    )
    return T0, T1, T2, T3, T4


def eval_qcel_dimer(mol_dimer, qA, muA, thetaA, qB, muB, thetaB):
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.
    """
    total_energy = 0.0
    RA = mol_dimer.get_fragment(0).geometry
    RB = mol_dimer.get_fragment(1).geometry
    ZA = mol_dimer.get_fragment(0).atomic_numbers
    ZB = mol_dimer.get_fragment(1).atomic_numbers
    for i in range(len(ZA)):
        for j in range(len(ZB)):
            rA = RA[i]
            qA_i = qA[i]
            muA_i = muA[i]
            thetaA_i = thetaA[i]

            rB = RB[j]
            qB_j = qB[j]
            muB_j = muB[j]
            thetaB_j = thetaB[j]

            pair_energy = eval_interaction(
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j
            )
            total_energy += pair_energy
    return total_energy * constants.h2kcalmol


def eval_qcel_dimer_individual(
    mol_dimer, qA, muA, thetaA, qB, muB, thetaB, match_cliff=False
) -> float:
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.
    """
    total_energy = np.zeros(3)
    RA = mol_dimer.get_fragment(0).geometry
    RB = mol_dimer.get_fragment(1).geometry
    ZA = mol_dimer.get_fragment(0).atomic_numbers
    ZB = mol_dimer.get_fragment(1).atomic_numbers
    for i in range(len(ZA)):
        for j in range(len(ZB)):
            rA = RA[i]
            qA_i = qA[i]
            muA_i = muA[i]
            thetaA_i = thetaA[i]

            rB = RB[j]
            qB_j = qB[j]
            muB_j = muB[j]
            thetaB_j = thetaB[j]

            E_q, E_dp, E_qpole = eval_interaction_individual(
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j
            )
            total_energy[0] += E_q
            total_energy[1] += E_dp
            total_energy[2] += E_qpole
    return total_energy * constants.h2kcalmol


def eval_qcel_dimer_individual_pairs(
    mol_dimer, qA, muA, thetaA, qB, muB, thetaB, match_cliff=False
) -> float:
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.
    """
    total_energy = np.zeros(3)
    RA = mol_dimer.get_fragment(0).geometry
    RB = mol_dimer.get_fragment(1).geometry
    ZA = mol_dimer.get_fragment(0).atomic_numbers
    ZB = mol_dimer.get_fragment(1).atomic_numbers
    elst_pairs = np.zeros((len(ZA), len(ZB)))
    for i in range(len(ZA)):
        for j in range(len(ZB)):
            rA = RA[i]
            qA_i = qA[i]
            muA_i = muA[i]
            thetaA_i = thetaA[i]

            rB = RB[j]
            qB_j = qB[j]
            muB_j = muB[j]
            thetaB_j = thetaB[j]

            E_q, E_dp, E_qpole = eval_interaction_individual(
                rA, qA_i, muA_i, thetaA_i, rB, qB_j, muB_j, thetaB_j
            )
            elst_pairs[i, j] += E_q
            elst_pairs[i, j] += E_dp
            elst_pairs[i, j] += E_qpole
    return elst_pairs * constants.h2kcalmol

def eval_qcel_dimer_individual_components(
    mol_dimer,
    qA,
    muA,
    thetaA,
    qB,
    muB,
    thetaB,
    alphaA=None,
    alphaB=None,
    traceless=True,
    amoeba_eq=False,
    match_cliff=True,
) -> Tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Evaluate the electrostatic interaction energy between two molecules using
    their multipole moments. Dimensionalities of qA should be [N], muA should
    be [N, 3], and thetaA should be [N, 3, 3]. Same for qB, muB, and thetaB.

    NOTE: the mu-Q and Q-Q damping terms do not agree with the CLIFF
    implementation. If damping is disabled or only care about the q-q, q-u,
    u-u, and q-Q terms then this function can be used. AP2 only uses these q-q,
    q-u, u-u, and q-Q terms, so this implementation is sufficient for AP2.
    """
    RA = mol_dimer.get_fragment(0).geometry
    RB = mol_dimer.get_fragment(1).geometry
    ZA = mol_dimer.get_fragment(0).atomic_numbers
    ZB = mol_dimer.get_fragment(1).atomic_numbers
    t = np.zeros((len(ZA), len(ZB)))
    E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs = (
        t.copy(),
        t.copy(),
        t.copy(),
        t.copy(),
        t.copy(),
        t.copy(),
    )
    E_ZA_MBs, E_ZB_MAs, E_ZA_ZBs = (
        t.copy(),
        t.copy(),
        t.copy(),
    )
    for i in range(len(ZA)):
        for j in range(len(ZB)):
            rA = RA[i]
            qA_i = qA[i]
            muA_i = muA[i]
            thetaA_i = thetaA[i]

            rB = RB[j]
            qB_j = qB[j]
            muB_j = muB[j]
            thetaB_j = thetaB[j]
            a_i = alphaA[i] if alphaA is not None and alphaB is not None else None
            a_j = alphaB[j] if alphaA is not None and alphaB is not None else None
            za_i = ZA[i] if amoeba_eq else None
            zb_j = ZB[j] if amoeba_eq else None
            E_qq, E_qu, E_uu, E_qQ, E_uQ, E_QQ, E_ZA_ZB, E_ZA_MB, E_ZB_MA = (
                eval_interaction_individual_components(
                    rA,
                    qA_i,
                    muA_i,
                    thetaA_i,
                    rB,
                    qB_j,
                    muB_j,
                    thetaB_j,
                    ZA=za_i,
                    ZB=zb_j,
                    alpha_i=a_i,
                    alpha_j=a_j,
                    traceless=traceless,
                    match_cliff=match_cliff,
                )
            )
            E_qqs[i, j] = E_qq
            E_qus[i, j] = E_qu
            E_uus[i, j] = E_uu
            E_qQs[i, j] = E_qQ
            E_uQs[i, j] = E_uQ
            E_QQs[i, j] = E_QQ
            if amoeba_eq:
                E_ZA_ZBs[i, j] = E_ZA_ZB
                E_ZA_MBs[i, j] = E_ZA_MB
                E_ZB_MAs[i, j] = E_ZB_MA
    total_energy = (
        np.sum(E_qqs)
        + np.sum(E_qus)
        + np.sum(E_uus)
        + np.sum(E_qQs)
        + np.sum(E_uQs)
        + np.sum(E_QQs)
        + np.sum(E_ZA_MBs)
        + np.sum(E_ZB_MAs)
        + np.sum(E_ZA_ZBs)
    )
    total_energy *= constants.h2kcalmol
    E_qqs *= constants.h2kcalmol
    E_qus *= constants.h2kcalmol
    E_uus *= constants.h2kcalmol
    E_qQs *= constants.h2kcalmol
    E_uQs *= constants.h2kcalmol
    E_QQs *= constants.h2kcalmol
    E_ZA_MBs *= constants.h2kcalmol
    E_ZB_MAs *= constants.h2kcalmol
    E_ZA_ZBs *= constants.h2kcalmol
    return (
        total_energy,
        E_qqs,
        E_qus,
        E_uus,
        E_qQs,
        E_uQs,
        E_QQs,
        E_ZA_ZBs,
        E_ZA_MBs,
        E_ZB_MAs,
    )


def eval_interaction_individual(
    RA, qA, muA, thetaA, RB, qB, muB, thetaB, traceless=False
):
    T0, T1, T2, T3, T4 = T_cart(RA, RB)

    # Most inputs will already be traceless, but we can ensure this is the case
    if not traceless:
        # divide by 3 because of traceless Buckingham quadrupoles have T_ij(2)
        # annihalite the trace
        traceA = np.trace(thetaA)
        thetaA[0, 0] -= traceA / 3.0
        thetaA[1, 1] -= traceA / 3.0
        thetaA[2, 2] -= traceA / 3.0
        traceB = np.trace(thetaB)
        thetaB[0, 0] -= traceB / 3.0
        thetaB[1, 1] -= traceB / 3.0
        thetaB[2, 2] -= traceB / 3.0

    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * (1.0 / 3.0)

    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = np.sum(
        T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA))
    ) * (-1.0 / 3.0)

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * (1.0 / 9.0)
    # partial-charge electrostatic energy
    E_q = E_qq
    # dipole correction
    E_u = E_qu + E_uu
    # quadrupole correction
    E_Q = E_qQ + E_uQ + E_QQ
    return E_q, E_u, E_Q


def eval_interaction_individual_components(
    RA,
    qA,
    muA,
    thetaA,
    RB,
    qB,
    muB,
    thetaB,
    ZA=None,
    ZB=None,
    alpha_i=None,
    alpha_j=None,
    traceless=False,
    match_cliff=False,
):
    """
    if alpha_i and alpha_j are provided, Thole damping is applied.

    if amoeba_eq is True, evaulate 4 elst terms instead of just MTP-MTP. Note
    the charge term has Z subtracted from qA and qB, so it is not the same as
    the MTP-MTP charge term, but these values will ultimately agree if damping
    is disabled. Need extra flexibility for Z-MTP and MTP-MTP terms when
    damping.
    """
    if not match_cliff:
        c_qQ, c_uQ, c_QQ = (1.0 / 3.0), (1.0 / 3.0), (1.0 / 9.0)
    else:
        c_qQ, c_uQ, c_QQ = 1.0, 1.0, 1.0
    T0, T1, T2, T3, T4 = T_cart(RA, RB, alpha_i, alpha_j)
    E_ZA_MB = None
    E_ZB_MA = None
    E_ZA_ZB = None
    if ZA is not None and ZB is not None:
        qA -= ZA
        qB -= ZB

    # Most inputs will already be traceless, but we can ensure this is the case
    if not traceless:
        thetaA = 0.5 * (thetaA + np.swapaxes(thetaA, -1, -2))
        traceA = np.trace(thetaA)
        thetaA[0, 0] -= traceA / 3.0
        thetaA[1, 1] -= traceA / 3.0
        thetaA[2, 2] -= traceA / 3.0
        thetaB = 0.5 * (thetaB + np.swapaxes(thetaB, -1, -2))
        traceB = np.trace(thetaB)
        thetaB[0, 0] -= traceB / 3.0
        thetaB[1, 1] -= traceB / 3.0
        thetaB[2, 2] -= traceB / 3.0

    # AP2 code had factors of 1/3, -1/3, 1/9 in qQ, uQ, QQ terms; however,
    # these make the energies disagree with CLIFF. CLIFF achieves better
    # agreement with SAPT0 elst, so which is right?
    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * c_qQ  # * (1.0 / 3.0)

    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = (
        np.sum(T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA)))
        * -1.0
        * c_uQ
    )  # * (-1.0 / 3.0)

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * c_QQ  # * (1.0 / 9.0)
    if ZA is not None and ZB is not None:
        # Nuclear attraction terms
        T0, _, _ = T_cart_Z_MTP(RA, RB, None)
        E_ZA_ZB = T0 * ZA * ZB
        # Only update to specific T's if damping
        if alpha_i is not None and alpha_j is not None:
            T0, T1, T2 = T_cart_Z_MTP(RA, RB, alpha_j)
        # A: Nuclear - charge, Nuclear - dipole, Nuclear - theta
        E_ZA_qB = T0 * ZA * qB
        E_ZA_uB = np.sum(T1 * ZA * muB)
        E_ZA_QB = np.sum(T2 * ZA * thetaB * c_qQ)
        E_ZA_MB = E_ZA_qB + E_ZA_uB + E_ZA_QB
        if alpha_i is not None:
            T0, T1, T2 = T_cart_Z_MTP(RA, RB, alpha_i)
        # B: Nuclear - charge, Nuclear - dipole, Nuclear - theta
        E_ZB_qA = T0 * ZB * qA
        E_ZB_uA = np.sum(-T1 * ZB * muA)
        E_ZB_QA = np.sum(T2 * ZB * thetaA * c_qQ)
        E_ZB_MA = E_ZB_qA + E_ZB_uA + E_ZB_QA

    return E_qq, E_qu, E_uu, E_qQ, E_uQ, E_QQ, E_ZA_ZB, E_ZA_MB, E_ZB_MA


def interaction_tensor(coord1, coord2, cell=None):
    """Return interaction tensor up to quadrupoles between two atom coordinates"""
    # Indices for MTP moments:
    # 00  01  02  03  04  05  06  07  08  09  10  11  12
    #  .,  x,  y,  z, xx, xy, xz, yx, yy, yz, zx, zy, zz
    if cell:
        vec = cell.pbc_distance(coord1, coord2)
    else:
        vec = coord2 - coord1
    r = np.linalg.norm(vec)
    r2 = r**2
    r4 = r2**2
    ri = 1.0 / r
    ri2 = ri**2
    ri3 = ri**3
    ri5 = ri**5
    ri7 = ri**7
    ri9 = ri**9
    x = vec[0]
    y = vec[1]
    z = vec[2]
    x2 = x**2
    y2 = y**2
    z2 = z**2
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2
    it = np.zeros((13, 13))
    # Charge charge
    it[0, 0] = ri
    # Charge dipole
    it[0, 1] = -x * ri3
    it[0, 2] = -y * ri3
    it[0, 3] = -z * ri3
    # Charge quadrupole
    it[0, 4] = (3 * x2 - r2) * ri5  # xx
    it[0, 5] = 3 * x * y * ri5  # xy
    it[0, 6] = 3 * x * z * ri5  # xz
    it[0, 7] = it[0, 5]  # yx
    it[0, 8] = (3 * y2 - r2) * ri5  # yy
    it[0, 9] = 3 * y * z * ri5  # yz
    it[0, 10] = it[0, 6]  # zx
    it[0, 11] = it[0, 9]  # zy
    it[0, 12] = -it[0, 4] - it[0, 8]  # zz
    # Dipole dipole
    it[1, 1] = -it[0, 4]  # xx
    it[1, 2] = -it[0, 5]  # xy
    it[1, 3] = -it[0, 6]  # xz
    it[2, 2] = -it[0, 8]  # yy
    it[2, 3] = -it[0, 9]  # yz
    it[3, 3] = -it[1, 1] - it[2, 2]  # zz
    # Dipole quadrupole
    it[1, 4] = -3 * x * (3 * r2 - 5 * x2) * ri7  # xxx
    it[1, 5] = it[1, 7] = it[2, 4] = -3 * y * (r2 - 5 * x2) * ri7  # xxy xyx yxx
    it[1, 6] = it[1, 10] = it[3, 4] = -3 * z * (r2 - 5 * x2) * ri7  # xxz xzx zxx
    it[1, 8] = it[2, 5] = it[2, 7] = -3 * x * (r2 - 5 * y2) * ri7  # xyy yxy yyx
    it[1, 9] = it[1, 11] = it[2, 6] = it[2, 10] = it[3, 5] = it[3, 7] = (
        15 * x * y * z * ri7
    )  # xyz xzy yxz yzx zxy zyx
    it[1, 12] = it[3, 6] = it[3, 10] = -it[1, 4] - it[1, 8]  # xzz zxz zzx
    it[2, 8] = -3 * y * (3 * r2 - 5 * y2) * ri7  # yyy
    it[2, 9] = it[2, 11] = it[3, 8] = -3 * z * (r2 - 5 * y2) * ri7  # yyz yzy zyy
    it[2, 12] = it[3, 9] = it[3, 11] = -it[1, 5] - it[2, 8]  # yzz zyz zzy
    it[3, 12] = -it[1, 6] - it[2, 9]  # zzz
    # Quadrupole quadrupole
    it[4, 4] = (105 * x4 - 90 * x2 * r2 + 9 * r4) * ri9  # xxxx
    it[4, 5] = it[4, 7] = 15 * x * y * (7 * x2 - 3 * r2) * ri9  # xxxy xxyx
    it[4, 6] = it[4, 10] = 15 * x * z * (7 * x2 - 3 * r2) * ri9  # xxxz xxzx
    it[4, 8] = it[5, 5] = it[5, 7] = it[7, 7] = (
        105 * x2 * y2 + 15 * z2 * r2 - 12 * r4
    ) * ri9  # xxyy xyxy xyyx yxyx
    it[4, 9] = it[4, 11] = it[5, 6] = it[5, 10] = it[6, 7] = it[7, 10] = (
        15 * y * z * (7 * x2 - 1 * r2) * ri9
    )  # xxyz xxzy xyxz xyzx xzyx yxzx
    it[4, 12] = it[6, 6] = it[6, 10] = it[10, 10] = (
        -it[4, 4] - it[4, 8]
    )  # xxzz xzxz xzzx zxzx
    it[5, 8] = it[7, 8] = 15 * x * y * (7 * y2 - 3 * r2) * ri9  # xyyy yxyy
    it[5, 9] = it[5, 11] = it[6, 8] = it[7, 9] = it[7, 11] = it[8, 10] = (
        15 * x * z * (7 * y2 - r2) * ri9
    )  # xyyz xyzy xzyy yxyz yxzy yyzx
    it[5, 12] = it[6, 9] = it[6, 11] = it[7, 12] = it[9, 10] = it[10, 11] = (
        -it[4, 5] - it[5, 8]
    )  # xyzz xzyz xzzy yxzz yzzx zxzy
    it[6, 12] = it[10, 12] = -it[4, 6] - it[5, 9]  # xzzz zxzz
    it[8, 8] = (105 * y4 - 90 * y2 * r2 + 9 * r4) * ri9  # yyyy
    it[8, 9] = it[8, 11] = 15 * y * z * (7 * y2 - 3 * r2) * ri9  # yyyz yyzy
    it[8, 12] = it[9, 9] = it[9, 11] = it[11, 11] = (
        -it[4, 8] - it[8, 8]
    )  # yyzz yzyz yzzy zyzy
    it[9, 12] = it[11, 12] = -it[4, 9] - it[8, 9]  # yzzz zyzz
    it[12, 12] = -it[4, 12] - it[8, 12]  # zzzz
    # Symmetrize
    it = it + it.T - np.diag(it.diagonal())
    # Some coefficients need to be multiplied by -1
    for i in range(1, 4):
        for j in range(0, 1):
            it[i, j] *= -1.0
    for i in range(4, 13):
        for j in range(1, 4):
            it[i, j] *= -1.0
    return it


def eval_interaction(RA, qA, muA, thetaA, RB, qB, muB, thetaB, traceless=False):
    T0, T1, T2, T3, T4 = T_cart(RA, RB)

    # Most inputs will already be traceless, but we can ensure this is the case
    if not traceless:
        traceA = np.trace(thetaA)
        thetaA[0, 0] -= traceA / 3.0
        thetaA[1, 1] -= traceA / 3.0
        thetaA[2, 2] -= traceA / 3.0
        traceB = np.trace(thetaB)
        thetaB[0, 0] -= traceB / 3.0
        thetaB[1, 1] -= traceB / 3.0
        thetaB[2, 2] -= traceB / 3.0

    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * (1.0 / 3.0)

    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = np.sum(
        T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA))
    ) * (-1.0 / 3.0)

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * (1.0 / 9.0)

    # partial-charge electrostatic energy
    E_q = E_qq

    # dipole correction
    E_u = E_qu + E_uu

    # quadrupole correction
    E_Q = E_qQ + E_uQ + E_QQ

    return E_q + E_u + E_Q


def eval_dimer2(RA, RB, ZA, ZB, QA, QB):
    maskA = ZA >= 1
    maskB = ZB >= 1

    # Keep R in a.u. (molden convention)
    RA_temp = RA[maskA] * 1.88973
    RB_temp = RB[maskB] * 1.88973
    ZA_temp = ZA[maskA]
    ZB_temp = ZB[maskB]
    QA_temp = QA[maskA]
    QB_temp = QB[maskB]

    quadrupole_A = []
    quadrupole_B = []

    for ia in range(len(RA_temp)):
        # QA_temp[ia][4:10] = (3.0/2.0) * qpole_redundant(QA_temp[ia][4:10])
        quadrupole_A.append((3.0 / 2.0) * qpole_redundant(QA_temp[ia][4:10]))

    for ib in range(len(RB_temp)):
        # QB_temp[ib][4:10] = (3.0/2.0) * qpole_redundant(QB_temp[ib][4:10])
        quadrupole_B.append((3.0 / 2.0) * qpole_redundant(QB_temp[ib][4:10]))

    total_energy = 0.0

    # calculate multipole electrostatics for each atom pair
    for ia in range(len(RA_temp)):
        for ib in range(len(RB_temp)):
            rA = RA_temp[ia]
            qA = QA_temp[ia]

            rB = RB_temp[ib]
            qB = QB_temp[ib]

            pair_energy = eval_interaction(
                rA,
                qA[0],
                qA[1:4],
                # qA[4:10],
                quadrupole_A[ia],
                rB,
                qB[0],
                qB[1:4],
                # qB[4:10])
                quadrupole_B[ib],
            )

            total_energy += pair_energy

    Har2Kcalmol = 627.5094737775374055927342256

    return total_energy * Har2Kcalmol


def eval_dimer(RA, RB, ZA, ZB, QA, QB):
    # Keep R in a.u. (molden convention)
    RA_temp = RA * 1.88973
    RB_temp = RB * 1.88973

    total_energy = 0.0

    maskA = ZA >= 1
    maskB = ZB >= 1

    pair_mat = np.zeros((int(np.sum(maskA, axis=0)), int(np.sum(maskB, axis=0))))

    # calculate multipole electrostatics for each atom pair
    for ia in range(len(RA_temp)):
        for ib in range(len(RB_temp)):
            rA = RA_temp[ia]
            zA = ZA[ia]
            qA = QA[ia]

            rB = RB_temp[ib]
            zB = ZB[ib]
            qB = QB[ib]

            if (zA == 0) or (zB == 0):
                continue

            pair_energy = eval_interaction(
                rA,
                qA[0],
                qA[1:4],
                (3.0 / 2.0) * qpole_redundant(qA[4:10]),
                rB,
                qB[0],
                qB[1:4],
                (3.0 / 2.0) * qpole_redundant(qB[4:10]),
            )
            total_energy += pair_energy
            pair_mat[ia][ib] = pair_energy

    Har2Kcalmol = 627.5094737775374055927342256

    return total_energy * Har2Kcalmol, pair_mat * Har2Kcalmol


libmbd_vwd_params = pd.read_csv(
    # osp.join(current_file_path, "data", "vdw-params.csv"),
    resources.files(
        "apnet_pt",
    ).joinpath("data", "vdw-params.csv"),
    header=0,
    index_col=0,
    sep=",",
    nrows=102,
)
free_atom_polarizabilities = {
    # el: v for el, v in zip(libmbd_vwd_params['Z'], libmbd_vwd_params['alpha_0(TS)'])
    el: v
    for el, v in zip(libmbd_vwd_params["Z"], libmbd_vwd_params["alpha_0(BG)"])
}


def dimer_induced_dipole(
    qcel_dimer: qcel.models.Molecule,
    qA: np.ndarray,
    muA: np.ndarray,
    thetaA: np.ndarray,
    qB: np.ndarray,
    muB: np.ndarray,
    thetaB: np.ndarray,
    hirshfeld_volume_ratio_A: np.ndarray,
    hirshfeld_volume_ratio_B: np.ndarray,
    valence_widths_A: np.ndarray,
    valence_widths_B: np.ndarray,
    atom_polarizabilities_A: np.ndarray = None,
    atom_polarizabilities_B: np.ndarray = None,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    # CLIFF omega parameter = 0.7 for fewer iterations
    omega: float = 0.7,
    # CLIFF Thole damping parameter = 0.39
    thole_damping_param: float = 0.39,
) -> float:
    """
    Calculate the induced dipole interaction energy between two molecules using
    their multipole moments and Hirshfeld volume ratios. Follow classical
    induction model from this paper:
    https://pubs.aip.org/aip/jcp/article/154/18/184110/200216/CLIFF-A-component-based-machine-learned
    """

    # Get molecular fragments
    molA = qcel_dimer.get_fragment(0)
    molB = qcel_dimer.get_fragment(1)
    alpha_0_A = np.array([free_atom_polarizabilities[i] for i in molA.atomic_numbers])
    alpha_0_B = np.array([free_atom_polarizabilities[i] for i in molB.atomic_numbers])
    hirshfeld_volume_ratio_A = hirshfeld_volume_ratio_A.flatten()
    hirshfeld_volume_ratio_B = hirshfeld_volume_ratio_B.flatten()

    # Get atomic positions (in bohr)
    RA = molA.geometry
    RB = molB.geometry
    ZA = molA.atomic_numbers
    ZB = molB.atomic_numbers

    # Calculate atomic polarizabilities using Hirshfeld volume ratios
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B

    if atom_polarizabilities_A is not None and atom_polarizabilities_B is not None:
        alpha_A = atom_polarizabilities_A.flatten()
        alpha_B = atom_polarizabilities_B.flatten()

    # Combine all atoms and properties
    R_all = np.vstack([RA, RB])
    alpha_all = np.vstack([alpha_A.reshape(-1, 1), alpha_B.reshape(-1, 1)]).flatten()
    q_all = np.concatenate([qA, qB]).flatten()
    mu_all = np.concatenate([muA, muB])
    theta_all = np.concatenate([thetaA, thetaB])

    n_atoms_A = len(RA)
    n_atoms_B = len(RB)
    n_atoms_total = n_atoms_A + n_atoms_B

    # Interaction tensor between M_i*T_ij*M_j
    T_abij = np.zeros((n_atoms_total, n_atoms_total, 13, 13))
    M = np.zeros((n_atoms_total, 13))
    M[:, 0] = q_all  # Charge
    M[:, 1:4] = mu_all  # Dipole
    M[:, 4:13] = theta_all.reshape(n_atoms_total, 9)  # Quadrupole (flattened)

    # Multipoles on molecule A
    M_A = M[:n_atoms_A, :]
    # Multipoles on molecule B
    M_B = M[n_atoms_A:, :]

    # Initialize interaction tensors
    for i in range(n_atoms_total):
        for j in range(n_atoms_total):
            if i == j:
                T_abij[i, j, :, :] = np.zeros((13, 13))
                continue
            # print(f"{i}::{j}")
            T0, T1, T2, T3, T4 = T_cart_Thole_damping(
                R_all[i], R_all[j], alpha_all[i], alpha_all[j], thole_damping_param
            )
            T_abij[i, j, 0, 0] = T0
            T_abij[i, j, 0, 1:4] = T1
            T_abij[i, j, 1:4, 0] = T1
            T_abij[i, j, 1:4, 1:4] = T2
            T_abij[i, j, 1:4, 4:13] = T3.reshape(3, 9)
            T_abij[i, j, 4:13, 1:4] = T3.T.reshape(9, 3)
            T_abij[i, j, 4:13, 4:13] = T4.reshape(9, 9)
            T_abij[i, j, 0, 4:13] = T2.reshape(9)
            T_abij[i, j, 4:13, 0] = T2.reshape(9)

    mu_induced_0 = np.zeros((n_atoms_total, 3))
    mu_induced_0_A = mu_induced_0[:n_atoms_A, :]
    mu_induced_0_B = mu_induced_0[n_atoms_A:, :]
    # print(f"{alpha_A=}")
    # mu_induced_0_A[:, :] = np.einsum(
    #     "a,abij,bj->ai", alpha_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, :], M_B
    # )
    mu_induced_0_A[:, :] = np.einsum(
        "a,abi,b->ai", alpha_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, 0], M_B[:, 0]
    )
    # print(f"{T_abij[:n_atoms_A, n_atoms_A:, 1:4, 0]=}")
    # print(f"{T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4]=}")
    print(f"{mu_induced_0_A=}")
    mu_induced_0_A[:, :] += np.einsum(
        "a,abij,bj->ai", alpha_A, T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4], M_B[:, 1:4]
    )
    print(f"{mu_induced_0_A=}")
    # mu_induced_0_B[:, :] = np.einsum(
    #     "b,baij,ai->bj", alpha_B, T_abij[n_atoms_A:, :n_atoms_A, :, 1:4], M_A
    # )
    mu_induced_0_B[:, :] = np.einsum(
        "b,baj,a->bj", alpha_B, T_abij[n_atoms_A:, :n_atoms_A, 0, 1:4], M_A[:, 0]
    )
    print(f"{mu_induced_0_B=}")
    mu_induced_0_B[:, :] += np.einsum(
        "b,baij,ai->bj", alpha_B, T_abij[n_atoms_A:, :n_atoms_A, 1:4, 1:4], M_A[:, 1:4]
    )
    print(f"{mu_induced_0_B=}")
    # Self-consistent induced dipole iterations
    mu_induced = mu_induced_0.copy()
    print("mu(0):")
    print(mu_induced_0)
    """
mtp1 = array([[-0.90283, -0.12100, -0.19035,  0.00497,  0.00000,  0.00000,
         0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
         0.00000],
       [ 0.45219, -0.02343,  0.01783, -0.00038,  0.00000,  0.00000,
         0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
         0.00000],
       [ 0.45063,  0.02653, -0.01379,  0.00028,  0.00000,  0.00000,
         0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
         0.00000]])
O [ 0.06873  0.02041 -0.00066]
H [ 0.00287 -0.00024 -0.00000]
H [ 0.01425  0.00205 -0.00008]
O [ 0.14175 -0.02421  0.00023]
H [ 0.00287 -0.00183  0.00173]
H [ 0.00286 -0.00192 -0.00165]
mtp1 = array([[-0.90516, -0.14261,  0.17417, -0.00395,  0.00000,  0.00000,
         0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
         0.00000],
       [ 0.45258,  0.00157, -0.00108,  0.02948,  0.00000,  0.00000,
         0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
         0.00000],
       [ 0.45258,  0.00140, -0.00256, -0.02940,  0.00000,  0.00000,
         0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,
         0.00000]])
mu_prev:
 [[[ 0.00000  0.00000  0.00000]
  [ 0.00000  0.00000  0.00000]
  [ 0.00000  0.00000  0.00000]]

 [[ 0.00000  0.00000  0.00000]
  [ 0.00000  0.00000  0.00000]
  [ 0.00000  0.00000  0.00000]]]
mu_next:
 [[[ 0.06873  0.02041 -0.00066]
  [ 0.00287 -0.00024 -0.00000]
  [ 0.01425  0.00205 -0.00008]]

 [[ 0.14175 -0.02421  0.00023]
  [ 0.00287 -0.00183  0.00173]
  [ 0.00286 -0.00192 -0.00165]]]
mu_prev:
 [[[ 0.09856  0.02394 -0.00082]
  [-0.00114 -0.00314  0.00008]
  [ 0.02094  0.00186 -0.00010]]

 [[ 0.17470 -0.02548  0.00017]
  [-0.00377 -0.00336  0.00662]
  [-0.00381 -0.00368 -0.00642]]]
mu_next:
 [[[ 0.09856  0.02394 -0.00082]
  [-0.00114 -0.00314  0.00008]
  [ 0.02094  0.00186 -0.00010]]

 [[ 0.17470 -0.02548  0.00017]
  [-0.00377 -0.00336  0.00662]
  [-0.00381 -0.00368 -0.00642]]]
    """

    M_induced_0 = M.copy()
    M_induced_0[:n_atoms_A, 1:4] = mu_induced_0_A
    M_induced_0[n_atoms_A:, 1:4] = mu_induced_0_B
    M_A_induced_0 = M_induced_0[:n_atoms_A, :]
    M_B_induced_0 = M_induced_0[n_atoms_A:, :]
    M_induced = M.copy()
    for iteration in range(max_iterations):
        mu_induced_old = mu_induced.copy()
        mu_sum = np.zeros_like(mu_induced)
        for i in range(n_atoms_total):
            # CLIFF Eq. 20 is wrong, should be Eq. 6 from
            # https://pubs.acs.org/doi/10.1021/ct200304d where update is
            # restricted to induced_dipoles only (not permanent multipoles)
            mu_sum[i] = alpha_all[i] * np.einsum(
                "nij,nj->i",
                T_abij[i, :, 1:4, 1:4],
                M_induced[:, 1:4],
            )
        # print(f"{mu_sum[0, :] = }")
        mu_sum += mu_induced_0
        mu_induced = (1 - omega) * mu_induced_old + omega * (mu_sum)
        M_induced[:, 1:4] = mu_induced
        # print(f"{mu_induced[0, :] = }")
        # break
        # Check convergence
        delta = np.linalg.norm(mu_induced - mu_induced_old)
        if delta < convergence_threshold:
            print(f"   Converged after {iteration + 1} iterations.")
            break
    mu_induced_A = mu_induced[:n_atoms_A, :]
    mu_induced_B = mu_induced[n_atoms_A:, :]
    # print("mu:")
    # print(mu_all)
    # print("mu(0):")
    # print(mu_induced_0)
    print("mu(n):")
    print(mu_induced)
    # diff
    # print(
    #     f"Converged in {iteration + 1} steps with diff: {delta:.5e}, target: {
    #         convergence_threshold:.5e}"
    # )
    """
    Converged in 14 steps with diff: 6.97042e-06, target: 1.00000e-05
Pair: O-O  E_ind:  1.8889 kcal/mol
  mu_ind1: [ 0.09857  0.02394 -0.00082]  mu_ind2: [ 0.17471 -0.02548  0.00017]
  M_A: [-0.90283 -0.12100 -0.19035  0.00497]  M_B: [-0.90516 -0.14261  0.17417 -0.00395]
  T[:4, :4] = array([[ 0.19095, -0.03645, -0.00110,  0.00012],
       [ 0.03645, -0.01391, -0.00063,  0.00007],
       [ 0.00110, -0.00063,  0.00694,  0.00000],
       [-0.00012,  0.00007,  0.00000,  0.00696]])
Pair: O-H  E_ind:  0.6634 kcal/mol
  mu_ind1: [ 0.09857  0.02394 -0.00082]  mu_ind2: [-0.00377 -0.00336  0.00662]
  M_A: [-0.90283 -0.12100 -0.19035  0.00497]  M_B: [ 0.45258  0.00157 -0.00108  0.02948]
  T[:4, :4] = array([[ 0.16254, -0.02553,  0.00288, -0.00614],
       [ 0.02553, -0.00774,  0.00136, -0.00290],
       [-0.00288,  0.00136,  0.00414,  0.00033],
       [ 0.00614, -0.00290,  0.00033,  0.00360]])
    """
    E_0_ind = (
        float(
            np.einsum(
                "abji,bi,aj->",
                T_abij[:n_atoms_A, n_atoms_A:, :, 1:4],
                mu_induced_0_B,
                M_A_induced_0,
            )
            - np.einsum(
                "abij,ai,bj->",
                T_abij[:n_atoms_A, n_atoms_A:, 1:4, :],
                mu_induced_0_A,
                M_B_induced_0,
            )
        )
        * constants.h2kcalmol
    )
    # Calculate induction energy
    E_ind = 0.0
    en1 = (
        -np.einsum(
            "ai,abi,b->ab",
            mu_induced_A,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 0],
            M_B[:, 0],
        )
        - np.einsum(
            "ai,abij,bj->ab",
            mu_induced_A,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
            M_B[:, 1:4],
        )
    ) * constants.h2kcalmol
    en2 = (
        np.einsum(
            "bj,abj,a->ab",
            mu_induced_B,
            T_abij[:n_atoms_A, n_atoms_A:, 0, 1:4],
            M_A[:, 0],
        )
        - np.einsum(
            "bj,abij,ai->ab",
            mu_induced_B,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
            M_A[:, 1:4],
        )
    ) * constants.h2kcalmol
    E_qu = (
        -np.einsum(
            "ai,abi,b->ab",
            mu_induced_A,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 0],
            M_B[:, 0],
        ) +
        np.einsum(
            "bj,abj,a->ab",
            mu_induced_B,
            T_abij[:n_atoms_A, n_atoms_A:, 0, 1:4],
            M_A[:, 0],
        )
            ) * constants.h2kcalmol
    E_uu = (
        - np.einsum(
            "ai,abij,bj->ab",
            mu_induced_A,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
            M_B[:, 1:4],
        )
        - np.einsum(
            "bj,abij,ai->ab",
            mu_induced_B,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
            M_A[:, 1:4],
        )
            ) * constants.h2kcalmol

    print(f"{E_qu=}")
    print(f"{E_uu=}")
    print(f"{np.sum(E_qu)=}")
    print(f"{np.sum(E_uu)=}")
    print(
     (
        - np.einsum(
            "ai,abij,bj->ab",
            mu_induced_A,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
            M_B[:, 1:4],
        )
            ) * constants.h2kcalmol,
        - np.einsum(
            "bj,abij,ai->ab",
            mu_induced_B,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
            M_A[:, 1:4],
        )* constants.h2kcalmol
    )
    print(
     np.sum(
        - np.einsum(
            "ai,abij,bj->ab",
            mu_induced_A,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
            M_B[:, 1:4],
        )
            ) * constants.h2kcalmol,
        -np.sum(np.einsum(
            "bj,abij,ai->ab",
            mu_induced_B,
            T_abij[:n_atoms_A, n_atoms_A:, 1:4, 1:4],
            M_A[:, 1:4],
        ))* constants.h2kcalmol
    )

    print(f"{en1=}, {en2=}")
    print(f"{np.sum(en1)=:.2f}, {np.sum(en2)=:.2f}")
    E_ind = en1 + en2
    E_ind = np.sum(E_ind) / 2
    return E_ind


def thole_damping_torch(r_ij, alpha_i, alpha_j, a):
    """Apply Thole damping to interaction tensor"""
    # Compute damping factor
    u = r_ij / ((alpha_i * alpha_j) ** (1.0 / 6.0))
    print(f"{u = }")
    au3 = a * (u**3)
    l3 = 1 - torch.exp(-au3)
    l5 = 1 - (1 + au3) * torch.exp(-au3)
    return au3, l3, l5


def dimer_induced_dipole_torch(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    ZB,
    RB,
    qB,
    muB,
    quadB,
    e_AB_source,
    e_AB_target,
    e_AA_source,
    e_BB_source,
    e_AA_target,
    e_BB_target,
    hirshfeld_volume_ratio_A: torch.tensor,
    hirshfeld_volume_ratio_B: torch.tensor,
    valence_widths_A: torch.tensor,
    valence_widths_B: torch.tensor,
    atom_polarizabilities_A: torch.tensor = None,
    atom_polarizabilities_B: torch.tensor = None,
    K_A: torch.tensor = None,
    K_B: torch.tensor = None,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param: float = 0.39,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
) -> float:
    """
    Calculate the induced dipole interaction energy between two molecules using
    their multipole moments and Hirshfeld volume ratios. Follow classical
    induction model from this paper:
    https://pubs.aip.org/aip/jcp/article/154/18/184110/200216/CLIFF-A-component-based-machine-learned

    vwA = torch.where(vwA > 0.1, vwA, 0.1)
    vwB = torch.where(vwB > 0.1, vwB, 0.1)
    sigma_A_source = vwA.index_select(0, e_source)
    sigma_B_target = vwB.index_select(0, e_target)
    sigma_ij = sigma_A_source * sigma_B_target
    B_ij = (1.0 / sigma_ij).squeeze()
    # print(f"{B_ij = }")
    # print(f"{r_ij = }")
    S_ij = (
        (1.0 / 3.0 * (B_ij * r_ij) ** 2 + B_ij * r_ij + 1.0)
        * torch.exp(-B_ij * r_ij)
        * hartree2kcal
    )
    """
    from apnet_pt.AtomPairwiseModels.mtp_mtp import get_distances

    delta = torch.eye(3, device=qA.device)
    h2kcalmol = constants.h2kcalmol  # Hartree to kcal/mol conversion factor

    print(f"{hirshfeld_volume_ratio_A=}")
    print(f"{hirshfeld_volume_ratio_B=}")


    if atom_polarizabilities_A is not None and atom_polarizabilities_B is not None:
        alpha_A = atom_polarizabilities_A.squeeze(-1)
        alpha_B = atom_polarizabilities_B.squeeze(-1)
    else:
        alpha_0_A = torch.tensor([free_atom_polarizabilities[int(i)] for i in ZA], dtype=hirshfeld_volume_ratio_A.dtype, device=hirshfeld_volume_ratio_A.device)
        alpha_0_B = torch.tensor([free_atom_polarizabilities[int(i)] for i in ZB], dtype=hirshfeld_volume_ratio_A.dtype, device=hirshfeld_volume_ratio_A.device)
        alpha_A = alpha_0_A * hirshfeld_volume_ratio_A **(4/3.)
        alpha_B = alpha_0_B * hirshfeld_volume_ratio_B **(4/3.)

    # Note: need to include Thole damping here...
    def distance_tensors(Ri, Rj, e_source, e_target, alpha_A=None, alpha_B=None):
        dR_ang, dR_xyz_ang = get_distances(Ri, Rj, e_source, e_target)
        dR_xyz = dR_xyz_ang / constants.au2ang
        dR = dR_ang / constants.au2ang
        alpha_i = alpha_A.index_select(0, e_source)
        alpha_j = alpha_B.index_select(0, e_target)
        au3, lam_3, lam_5 = thole_damping_torch(dR, alpha_i, alpha_j, thole_damping_param)
        print(dR, alpha_i, alpha_j, sep='\n')
        # lam_5 = torch.ones_like(lam_5)
        # print(f"{lam_3=}, {lam_5=}")
        delta = torch.eye(3, device=dR.device)
        oodR = 1.0 / dR
        T1 = torch.einsum("x,xy,x->xy", oodR**3, -1.0 * dR_xyz, lam_3)
        T2 = 3 * torch.einsum("xy,xz,x->xyz", dR_xyz, dR_xyz, lam_5) - torch.einsum(
            "x,x,yz,x->xyz", dR, dR, delta, lam_3
        )
        T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)
        return dR, dR_xyz, oodR, T1, T2

    # Calculate interaction tensors between atoms
    dR_AB, dR_AB_xyz, T0_AB, T1_AB, T2_AB = distance_tensors(RA, RB, e_AB_source, e_AB_target, alpha_A, alpha_B)
    # print(f"{T0_AB=}")
    # print(f"{T1_AB=}")
    # print(f"{T2_AB=}")
    dR_AA, dR_AA_xyz, T0_AA, T1_AA, T2_AA = distance_tensors(RA, RA, e_AA_source, e_AA_target, alpha_A, alpha_A)
    dR_BB, dR_BB_xyz, T0_BB, T1_BB, T2_BB = distance_tensors(RB, RB, e_BB_source, e_BB_target, alpha_B, alpha_B)

    # Select relevant tensors for atom pairs
    alpha_A_source = alpha_A.index_select(0, e_AB_source)
    alpha_B_target = alpha_B.index_select(0, e_AB_target)

    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    alpha_BB_target = alpha_B.index_select(0, e_BB_target)

    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_target = qB.squeeze(-1).index_select(0, e_AB_target)

    muA_source = muA.index_select(0, e_AB_source)
    muB_target = muB.index_select(0, e_AB_target)

    # Initialize tensors for induced dipoles
    n_atoms_A = RA.shape[0]
    n_atoms_B = RB.shape[0]

    if K_A is not None and K_B is not None:
        """
        v_widths[s1] = [0.4111834223806629, 0.3502946586498706, 0.35229699276619997]
        v_widths[s2] = [0.41117481494233643, 0.35060148140776876, 0.35060415277704976]
        r = array([[ 5.23691,  6.15248,  6.15150],
       [ 6.04079,  7.12206,  7.12139],
       [ 3.42099,  4.45900,  4.45825]])
ovp = array([[ 0.00020,  0.00001,  0.00001],
       [ 0.00001,  0.00000,  0.00000],
       [ 0.00461,  0.00021,  0.00021]])
ind_params[s1] = [1.14769962, 0.685558974, 0.685558974]
ind_params[s2] = [1.14769962, 0.685558974, 0.685558974]

        """
        K_A_source = K_A.index_select(0, e_AB_source)
        K_B_target = K_B.index_select(0, e_AB_target)
        sigma_A_source = valence_widths_A.index_select(0, e_AB_source)
        sigma_B_target = valence_widths_B.index_select(0, e_AB_target)
        print(f"{sigma_A_source=}")
        print(f"{sigma_B_target=}")
        print(f"{dR_AB=}")
        B_ij = torch.sqrt(1.0 / (sigma_A_source * sigma_B_target))
        print(f"{B_ij=}")
        S_ij = (1.0 / 3.0 * (B_ij * dR_AB) ** 2 + B_ij * dR_AB + 1.0) * torch.exp(-B_ij * dR_AB)
        
        print(f"{K_A_source=}")
        print(f"{K_B_target=}")
        print(f"{S_ij=}")
        E_ind_overlap = K_A_source * S_ij * K_B_target * h2kcalmol
        print(f"{E_ind_overlap=}")
        print(f"Sum E_ind_overlap: {torch.sum(E_ind_overlap)=}")


    # Calculate initial induced dipoles (order-0)
    # A: Induced by B's multipoles
    mu_induced_0_A = torch.zeros((n_atoms_A, 3), device=qA.device)
    mu_induced_0_B = torch.zeros((n_atoms_B, 3), device=qB.device)

    # Calculate initial induced dipoles from molecule B's multipoles on molecule A
    # Contribution from charges
    mu_charge_A = torch.einsum("a,ai,a->ai", alpha_A_source, T1_AB, qB_target)
    mu_induced_0_A = scatter_sum_compile(mu_charge_A, e_AB_source, n_atoms_A)
    mu_dipole_A = torch.einsum("a,aij,aj->ai", alpha_A_source, T2_AB, muB_target)
    mu_induced_0_A += scatter_sum_compile(mu_dipole_A, e_AB_source, n_atoms_A)
    print(f"{mu_induced_0_A=}")

    mu_charge_B = torch.einsum("a,ai,a->ai", alpha_B_target, -T1_AB, qA_source)
    mu_induced_0_B = scatter_sum_compile(mu_charge_B, e_AB_target, n_atoms_B)
    mu_dipole_B = torch.einsum("a,aij,aj->ai", alpha_B_target, T2_AB, muA_source)
    mu_induced_0_B += scatter_sum_compile(mu_dipole_B, e_AB_target, n_atoms_B)
    print(f"{mu_induced_0_B=}")

    # Self-consistent induced dipole iterations
    mu_induced_A = mu_induced_0_A.clone()
    mu_induced_B = mu_induced_0_B.clone()

    # Iterative SCF procedure to converge induced dipoles
    for iteration in range(max_iterations):
        mu_induced_A_old = mu_induced_A.clone()
        mu_induced_B_old = mu_induced_B.clone()

        ####### (A) INDUCED DIPOLES ########
        # Induced dipoles on A due to induced dipoles on B
        mu_induced_A_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_A_source, T2_AB, mu_induced_B.index_select(0, e_AB_target)
        )
        mu_induced_A_new = scatter_sum_compile(mu_induced_A_due_B, e_AB_source, n_atoms_A)
        # Induced dipoles on A due to induced dipoles on A
        mu_induced_A_due_A = torch.einsum(
                "a,aij,aj->ai", alpha_AA_target, T2_AA, mu_induced_A.index_select(0, e_AA_source)
        )
        mu_induced_A_new += scatter_sum_compile(mu_induced_A_due_A, e_AA_target, n_atoms_A)
        mu_induced_A_new += mu_induced_0_A

        ####### (B) INDUCED DIPOLES ########
        # Induced dipoles on B due to induced dipoles on A
        mu_induced_B_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_B_target, T2_AB, mu_induced_A.index_select(0, e_AB_source)
        )
        mu_induced_B_new = scatter_sum_compile(mu_induced_B_due_A, e_AB_target, n_atoms_B)
        # Induced dipoles on B due to induced dipoles on B
        mu_induced_B_due_B = torch.einsum(
                "a,aij,aj->ai", alpha_BB_target, T2_BB, mu_induced_B.index_select(0, e_BB_source)
        )
        mu_induced_B_new += scatter_sum_compile(mu_induced_B_due_B, e_BB_target, n_atoms_B)
        mu_induced_B_new += mu_induced_0_B

        # Apply mixing
        mu_induced_A = (1 - omega) * mu_induced_A_old + omega * mu_induced_A_new
        mu_induced_B = (1 - omega) * mu_induced_B_old + omega * mu_induced_B_new

        # Check convergence
        delta_A = torch.norm(mu_induced_A - mu_induced_A_old)
        delta_B = torch.norm(mu_induced_B - mu_induced_B_old)
        delta = max(delta_A, delta_B)
        if delta < convergence_threshold:
            print(f"   Converged after {iteration + 1} iterations.")
            break
    print(f"{mu_induced_A=}")
    print(f"{mu_induced_B=}")
    muA_induced_source = mu_induced_A.index_select(0, e_AB_source)
    muB_induced_target = mu_induced_B.index_select(0, e_AB_target)
    qu = torch.einsum("x,xy->xy", qA_source, muB_induced_target) - torch.einsum(
        "x,xy->xy", qB_target, muA_induced_source
    )
    E_qu = torch.einsum("xy,xy->x", T1_AB, qu) * h2kcalmol
    E_uu = -1.0 * (
        torch.einsum("xy,xz,xyz->x", muA_induced_source, muB_target, T2_AB) +
        torch.einsum("xy,xz,xyz->x", muA_source, muB_induced_target, T2_AB)
    ) * h2kcalmol
    # print(f"{E_qu=}")
    # print(f"{E_uu=}")
    # print(f"{E_qu.sum()=}")
    # print(f"{E_uu.sum()=}")
    E_ind = (E_qu + E_uu) / 2.0
    if K_A is not None and K_B is not None:
        E_ind -= E_ind_overlap
    return E_ind


def intramolecular_induced_dipole(
    qcel_mol: qcel.models.Molecule,
    q: np.ndarray,
    mu: np.ndarray,
    theta: np.ndarray,
    hirshfeld_volume_ratio: np.ndarray = None,
    valence_widths: np.ndarray = None,
    atom_polarizabilities: np.ndarray = None,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param_mutual: float = 0.390,
    thole_damping_param_direct: float = 0.340,
    zero_dipoles: bool = False,
    zero_quadrupoles: bool = False,
    screening: bool = True,
    screening_distance: float = 1.8,
    heavy_atoms_only: bool = True,
    compute_energies: bool = False,
    verbose: int = 0,
) -> tuple:
    """
    Calculate intramolecular induced dipoles for a single molecule using
    its multipole moments and Hirshfeld volume ratios. Follow classical
    induction model from CLIFF paper:
    https://pubs.aip.org/aip/jcp/article/154/18/184110/200216/CLIFF-A-component-based-machine-learned
    
    Parameters
    ----------
    qcel_mol : qcelemental.models.Molecule
        The molecule object
    q : np.ndarray
        Atomic charges (n_atoms,)
    mu : np.ndarray
        Atomic dipole moments (n_atoms, 3)
    theta : np.ndarray
        Atomic quadrupole moments (n_atoms, 3, 3)
    hirshfeld_volume_ratio : np.ndarray
        Hirshfeld volume ratios for polarizability scaling (n_atoms,)
    valence_widths : np.ndarray
        Valence widths for each atom (n_atoms,)
    atom_polarizabilities : np.ndarray, optional
        Explicit atomic polarizabilities. If None, calculated from Hirshfeld ratios
    max_iterations : int
        Maximum number of SCF iterations
    convergence_threshold : float
        Convergence threshold for induced dipoles
    omega : float
        Damping parameter for SCF convergence (0.7 recommended)
    thole_damping_param : float
        Thole damping parameter (0.39 recommended)
    zero_dipoles : bool
        If True, set multipole dipoles to zero to see if induced dipoles recover them.
    
    Returns
    -------
    tuple
        (charges, induced_dipoles, quadrupoles) as numpy arrays
        - charges: original charges (n_atoms,)
        - induced_dipoles: converged induced dipole moments (n_atoms, 3)
        - quadrupoles: original quadrupoles (n_atoms, 3, 3)
    """
    
    R = qcel_mol.geometry
    # dist = np.linalg.norm(R[:, np.newaxis, :] - R[np.newaxis, :, :], axis=-1)
    Z = qcel_mol.atomic_numbers
    # print(f"{R=}")
    # print(f"{Z=}")
    
    if atom_polarizabilities is not None:
        alpha = atom_polarizabilities.flatten()
    else:
        alpha_0 = np.array([free_atom_polarizabilities[i] for i in Z])
        hirshfeld_volume_ratio = hirshfeld_volume_ratio.flatten()
        alpha = alpha_0 * hirshfeld_volume_ratio ** (4 / 3.0)
    if verbose > 0:
        print(f"{alpha=}")
        print(f"{thole_damping_param_mutual=}")
    
    n_atoms = len(R)
    q_flat = q.flatten()
    
    T_abij_mutual = np.zeros((n_atoms, n_atoms, 13, 13))
    T_abij_direct = np.zeros((n_atoms, n_atoms, 13, 13))
    M = np.zeros((n_atoms, 13))
    M[:, 0] = q_flat
    M[:, 1:4] = mu
    M[:, 4:13] = theta.reshape(n_atoms, 9)
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                T_abij_mutual[i, j, :, :] = np.zeros((13, 13))
                continue
            T0, T1, T2, T3, T4 = T_cart_Thole_damping(
                R[i], R[j], alpha[i], alpha[j], thole_damping_param_mutual, damping_term="mutual"
            )
            T_abij_mutual[i, j, 0, 0] = T0
            T_abij_mutual[i, j, 0, 1:4] = T1
            T_abij_mutual[i, j, 1:4, 0] = T1
            T_abij_mutual[i, j, 1:4, 1:4] = T2
            T_abij_mutual[i, j, 1:4, 4:13] = T3.reshape(3, 9)
            T_abij_mutual[i, j, 4:13, 1:4] = T3.T.reshape(9, 3)
            T_abij_mutual[i, j, 4:13, 4:13] = T4.reshape(9, 9)
            T_abij_mutual[i, j, 0, 4:13] = T2.reshape(9)
            T_abij_mutual[i, j, 4:13, 0] = T2.reshape(9)

            T0, T1, T2, T3, T4 = T_cart_Thole_damping(
                R[i], R[j], alpha[i], alpha[j], thole_damping_param_direct, damping_term="direct"
            )
            if screening and T0 ** -1 < screening_distance / constants.au2ang:
                # screening out 1-2 and 1-3 type interactions crudely
                # print(f"  Screening direct {i}-{j} at distance {T0**-1:.2f} < {1.8 / constants.au2ang} bohr")
                T0 *= 0
                T1 *= 0
                T2 *= 0
                T3 *= 0
                T4 *= 0
            T_abij_direct[i, j, 0, 0] = T0
            T_abij_direct[i, j, 0, 1:4] = T1
            T_abij_direct[i, j, 1:4, 0] = T1
            T_abij_direct[i, j, 1:4, 1:4] = T2
            T_abij_direct[i, j, 1:4, 4:13] = T3.reshape(3, 9)
            T_abij_direct[i, j, 4:13, 1:4] = T3.T.reshape(9, 3)
            T_abij_direct[i, j, 4:13, 4:13] = T4.reshape(9, 9)
            T_abij_direct[i, j, 0, 4:13] = T2.reshape(9)
            T_abij_direct[i, j, 4:13, 0] = T2.reshape(9)
    
    mu_induced_0 = np.zeros((n_atoms, 3))
    if zero_dipoles:
        M[:, 1:4] = 0.0

    # if zero_quadrupoles:
    #     M[:, 4:13] = 0.0

    mu_induced_0[:, :] = np.einsum(
        "a,abi,b->ai", alpha, T_abij_direct[:, :, 1:4, 0], M[:, 0]
    )
    mu_induced_0[:, :] += np.einsum(
        "a,abij,bj->ai", alpha, T_abij_direct[:, :, 1:4, 1:4], M[:, 1:4]
    )
    if heavy_atoms_only:
        h_inds = np.where(Z == 1)[0]
        mu_induced_0[h_inds, :] *= 0
        T_abij_mutual[h_inds, :, :, :] *= 0

    # mu_induced_0[:, :] += np.einsum(
    #     "a,abik,bk->ai", alpha, T_abij[:, :, 1:4, 4:13], M[:, 4:13]
    # )
    
    mu_induced = mu_induced_0.copy()
    
    for iteration in range(max_iterations):
        mu_induced_old = mu_induced.copy()
        mu_sum = np.zeros_like(mu_induced)
        for i in range(n_atoms):
            mu_sum[i] = alpha[i] * np.einsum(
                "nij,nj->i",
                T_abij_mutual[i, :, 1:4, 1:4],
                mu_induced,
            )
        mu_sum += mu_induced_0
        mu_induced = (1 - omega) * mu_induced_old + omega * mu_sum
        
        delta = np.linalg.norm(mu_induced - mu_induced_old)
        if delta < convergence_threshold:
            break

    # Energies
    if compute_energies:

        E_ind_pairs = (
            np.einsum(
                "ai,abi,b->ab",
                mu_induced,
                T_abij_direct[:, :, 1:4, 0],
                M[:, 0],
            )
            + np.einsum(
                "ai,abij,bj->ab",
                mu_induced,
                T_abij_direct[:, :, 1:4, 1:4],
                M[:, 1:4],
            )
        ) * constants.h2kcalmol
        E_ind = np.sum(E_ind_pairs) / -2
        print(f"Total intramolecular E_ind: {E_ind:.4f} kcal/mol")
        # Electrostatics energies using all permanent multipoles only
        E_elst_pairs = (
            # q-q
            np.einsum(
                "a,ab,b->ab",
                M[:, 0],
                T_abij_direct[:, :, 0, 0],
                M[:, 0],
            )
            # q-mu
            + np.einsum(
                "ai,abi,b->ab",
                M[:, 1:4],
                T_abij_direct[:, :, 1:4, 0],
                M[:, 0],
            )
            # q-Q
            + np.einsum(
                "ai,abi,b->ab",
                M[:, 4:13],
                T_abij_direct[:, :, 4:13, 0],
                M[:, 0],
            )
            # mu-mu
            + np.einsum(
                "ai,abij,bj->ab",
                M[:, 1:4],
                T_abij_direct[:, :, 1:4, 1:4],
                M[:, 1:4],
            )
            # mu-Q
            + np.einsum(
                "ai,abik,bl->ab",
                M[:, 1:4],
                T_abij_direct[:, :, 1:4, 4:13],
                M[:, 4:13],
            )
            # Q-Q
            + np.einsum(
                "ak,abkl,bl->ab",
                M[:, 4:13],
                T_abij_direct[:, :, 4:13, 4:13],
                M[:, 4:13],
                )
        ) * constants.h2kcalmol
        # Now that you don't have an A and B, you have to divide by 2 to avoid double counting
        E_elst = np.sum(E_elst_pairs) / -2
        print(f"Total intramolecular E_elst: {E_elst:.4f} kcal/mol")
        # difference of pairs
        if verbose > 1:
            E_diff_pairs = E_elst_pairs - E_ind_pairs
            print(f"{E_ind_pairs =}")
            print(f"{E_elst_pairs =}")
            print(f"{E_diff_pairs =}")
    if verbose > 0:
        mu_diff = mu_induced - mu.reshape(-1, 3)
        print(f"Original   dipoles:\n{mu}")
        print(f"Induced    dipoles:\n{mu_induced}")
        print("mu(0)=direct induced-dipoles:")
        print(mu_induced_0)
        # get magnitudes of dipoles
        mu_magnitudes = np.linalg.norm(mu.reshape(-1, 3), axis=1)
        mu_induced_magnitudes = np.linalg.norm(mu_induced, axis=1)
        mu_diff_magnitudes = np.linalg.norm(mu_diff, axis=1)
        print(f"Original   dipole magnitudes: {mu_magnitudes}")
        print(f"Induced    dipole magnitudes: {mu_induced_magnitudes}")
        print(f"Difference dipole magnitudes: {mu_diff_magnitudes}")
    return q_flat, mu_induced, theta




if __name__ == "__main__":
    T_cart()
