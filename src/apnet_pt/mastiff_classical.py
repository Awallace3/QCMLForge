"""MASTIFF shared-exponent short-range terms for electrostatics and induction.

In MASTIFF (Van Vleet et al., JCTC 2016, 12, 3851; JCTC 2018, 14, 739) each atom
carries ONE exponent B, shared by every short-range component, and each
component its own prefactor A:

    E_sr = sign * A_i A_j P(x) exp(-x),   P(x) = 1 + x + x^2/3,
    x = sqrt(B_i(W) B_j(W)) r.

Exchange is the positive member (`apnet_pt.mastiff.MastiffExchange`).  Charge
penetration and short-range induction are attractive, so here the sign is fixed
at -1 and A >= 0 is the caller's responsibility (exp or softplus readouts).
With that, the term is <= 0 on every pair by construction, not by a clamp.

Anisotropy follows the frame-free exponent placement of the `eq-l3` exchange:
per-atom global-frame coefficients c_i^{lm}, l = 1..3, contracted with the
Racah harmonics of the pair direction,

    B_i(W) = B_i exp(sum_lm c_i^{lm} C_lm(u_ij)),

and the j end sees -u_ij, i.e. a (-1)^l parity.  The same contraction applied
to a prefactor, A_i(W) = A_i exp(s_i), keeps A_i(W) > 0 for any coefficients.

Units are the caller's: B and r in reciprocal units of one another, A_i A_j in
the energy unit.  Nothing here converts Angstrom to bohr or hartree to kcal/mol.
"""

import math

import torch
from torch import Tensor

SQRT3, SQRT6, SQRT10, SQRT15 = (math.sqrt(n) for n in (3, 6, 10, 15))
RACAH_L3_LABELS = (
    "10", "11c", "11s",
    "20", "21c", "21s", "22c", "22s",
    "30", "31c", "31s", "32c", "32s", "33c", "33s",
)
RACAH_L3_DEGREES = (1,) * 3 + (2,) * 5 + (3,) * 7


def racah_l3(unit: Tensor) -> Tensor:
    """Real Racah-normalized harmonics C_lm, l = 1..3, on (..., 3) unit vectors.

    Slot order is `RACAH_L3_LABELS`; its first eight slots equal
    `apnet_pt.mastiff.RacahHarmonics`.  Each degree satisfies
    sum_m C_lm(u)^2 = 1 for a unit vector u.
    """
    x, y, z = unit.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack(
        [
            z, x, y,
            (3 * zz - 1) / 2,
            SQRT3 * x * z,
            SQRT3 * y * z,
            SQRT3 / 2 * (xx - yy),
            SQRT3 * x * y,
            z * (5 * zz - 3) / 2,
            SQRT6 / 4 * x * (5 * zz - 1),
            SQRT6 / 4 * y * (5 * zz - 1),
            SQRT15 / 2 * z * (xx - yy),
            SQRT15 * x * y * z,
            SQRT10 / 4 * x * (xx - 3 * yy),
            SQRT10 / 4 * y * (3 * xx - yy),
        ],
        dim=-1,
    )


def racah_parity(width: int, dtype=None, device=None) -> Tensor:
    """(-1)^l for the first `width` slots: C_lm(-u) = (-1)^l C_lm(u)."""
    degrees = torch.tensor(RACAH_L3_DEGREES[:width], device=device)
    return (-1.0) ** degrees.to(dtype or torch.get_default_dtype())


def anisotropic_log_scale(
    coefficients_i: Tensor,
    coefficients_j: Tensor,
    displacement: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return (s_i, s_j) for per-edge coefficient rows and R_j - R_i.

    `coefficients_*` are (..., w) with w <= 15 in `RACAH_L3_LABELS` order,
    already gathered onto the edge.  Multiply a quantity by exp(s) to make it
    anisotropic; the i end sees +u and the j end -u.
    """
    width = coefficients_i.shape[-1]
    if coefficients_j.shape[-1] != width or width > len(RACAH_L3_LABELS):
        raise ValueError("coefficient widths disagree or exceed l = 3")
    unit = displacement / torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    harmonics = racah_l3(unit)[..., :width].to(coefficients_i.dtype)
    parity = racah_parity(width, coefficients_i.dtype, coefficients_i.device)
    s_i = (coefficients_i * harmonics).sum(-1)
    s_j = (coefficients_j * harmonics * parity).sum(-1)
    return s_i, s_j


def slater_overlap(x: Tensor) -> Tensor:
    """P(x) exp(-x) with P(x) = 1 + x + x^2/3; 1 at x = 0, monotone decreasing."""
    return (1.0 + x + x.square() / 3.0) * torch.exp(-x)


def mastiff_short_range(
    a_i: Tensor,
    a_j: Tensor,
    b_i: Tensor,
    b_j: Tensor,
    distance: Tensor,
    log_width_i: Tensor | None = None,
    log_width_j: Tensor | None = None,
    log_prefactor_i: Tensor | None = None,
    log_prefactor_j: Tensor | None = None,
) -> Tensor:
    """Attractive MASTIFF short-range pair energy, -A_i A_j P(x) exp(-x).

    All inputs are aligned per edge.  `a_*` must be >= 0 and `b_*` > 0; the
    optional `log_width_*` make the exponent anisotropic (B(W) = B exp(s)) and
    `log_prefactor_*` the prefactor (A(W) = A exp(s)).  Neither can change the
    sign, so the returned energy is <= 0 whenever a_i, a_j >= 0.
    """
    width = b_i * b_j
    if log_width_i is not None:
        width = width * torch.exp(log_width_i + log_width_j)
    prefactor = a_i * a_j
    if log_prefactor_i is not None:
        prefactor = prefactor * torch.exp(log_prefactor_i + log_prefactor_j)
    return -prefactor * slater_overlap(torch.sqrt(width) * distance)


def mastiff_donor_acceptor_short_range(
    donor_i: Tensor,
    acceptor_i: Tensor,
    donor_j: Tensor,
    acceptor_j: Tensor,
    b_i: Tensor,
    b_j: Tensor,
    distance: Tensor,
    log_width_i: Tensor | None = None,
    log_width_j: Tensor | None = None,
    log_donor_i: Tensor | None = None,
    log_acceptor_i: Tensor | None = None,
    log_donor_j: Tensor | None = None,
    log_acceptor_j: Tensor | None = None,
) -> Tensor:
    """Complementary donor-acceptor short-range pair energy,
    -(D_i C_j + C_i D_j) P(x) exp(-x) <= 0.

    All inputs are aligned per edge with ``donor_* >= 0``, ``acceptor_* >= 0``,
    and ``b_* > 0``. Permutation symmetric in ``i <-> j`` while breaking
    like-pair homodimer false attraction (e.g. cation-cation has ``D = 0``).
    """
    width = b_i * b_j
    if log_width_i is not None:
        width = width * torch.exp(log_width_i + log_width_j)
    di, ci = donor_i, acceptor_i
    dj, cj = donor_j, acceptor_j
    if log_donor_i is not None:
        di = di * torch.exp(log_donor_i)
    if log_acceptor_i is not None:
        ci = ci * torch.exp(log_acceptor_i)
    if log_donor_j is not None:
        dj = dj * torch.exp(log_donor_j)
    if log_acceptor_j is not None:
        cj = cj * torch.exp(log_acceptor_j)
    prefactor = di * cj + ci * dj
    return -prefactor * slater_overlap(torch.sqrt(width) * distance)


def undamped_multipole_elst(
    ZA: Tensor,
    RA: Tensor,
    qA: Tensor,
    muA: Tensor,
    quadA: Tensor,
    ZB: Tensor,
    RB: Tensor,
    qB: Tensor,
    muB: Tensor,
    quadB: Tensor,
    e_AB_source: Tensor,
    e_AB_target: Tensor,
    Q_const: float = 3.0,
    include_uq_qq: bool = False,
) -> Tensor:
    """Undamped per-edge multipole electrostatics in kcal/mol.

    With ``include_uq_qq=False`` (default), evaluates CLIFF2's truncated term
    set (q-q, q-mu, mu-mu, q-Theta). With ``include_uq_qq=True``, adds exact
    dipole-quadrupole (u-Q) and quadrupole-quadrupole (Q-Q) interactions.
    """
    from apnet_pt import constants
    from apnet_pt.AtomPairwiseModels.mtp_mtp import _contract_uq_qq_damped

    dtype = qA.dtype
    RA, RB = RA.to(dtype), RB.to(dtype)
    del ZA, ZB  # the nuclear split cancels exactly when nothing is damped
    d_xyz = (RB.index_select(0, e_AB_target) - RA.index_select(0, e_AB_source))
    d_xyz = d_xyz / constants.au2ang
    d = torch.linalg.vector_norm(d_xyz, dim=-1)
    ood = 1.0 / d
    qa = qA.reshape(-1).index_select(0, e_AB_source)
    qb = qB.reshape(-1).index_select(0, e_AB_target)
    mua = muA.to(dtype).index_select(0, e_AB_source)
    mub = muB.to(dtype).index_select(0, e_AB_target)
    Qa = quadA.to(dtype).index_select(0, e_AB_source)
    Qb = quadB.to(dtype).index_select(0, e_AB_target)
    e_qq = qa * qb * ood
    T1 = -d_xyz * ood[:, None] ** 3
    e_qu = (T1 * (qa[:, None] * mub - qb[:, None] * mua)).sum(-1)
    eye = torch.eye(3, dtype=dtype, device=d.device)
    T2 = (3.0 * d_xyz[:, :, None] * d_xyz[:, None, :]
          - (d * d)[:, None, None] * eye) * ood[:, None, None] ** 5
    e_uu = -torch.einsum("xy,xz,xyz->x", mua, mub, T2)
    e_qQ = (T2 * (qa[:, None, None] * Qb + qb[:, None, None] * Qa)).sum((-1, -2))
    total = e_qq + e_qu + e_uu + e_qQ / Q_const
    if include_uq_qq:
        ones = torch.ones_like(ood)
        e_uQ, e_QQ = _contract_uq_qq_damped(
            d_xyz, ood, mua, Qa, mub, Qb, ones, ones, ones, Q_const=Q_const,
        )
        total = total + e_uQ + e_QQ
    return 627.509 * total


def rackers_elst_pair_exponents(
    ZA: Tensor,
    RA: Tensor,
    qA: Tensor,
    muA: Tensor,
    quadA: Tensor,
    ZB: Tensor,
    RB: Tensor,
    qB: Tensor,
    muB: Tensor,
    quadB: Tensor,
    e_AB_source: Tensor,
    e_AB_target: Tensor,
    alpha_source: Tensor,
    alpha_target: Tensor,
    Q_const: float = 3.0,
    damping: str = "cliff",
    include_uq_qq: bool = False,
) -> Tensor:
    """Damped multipole electrostatics with one damping exponent per pair END.

    ``damping="cliff"`` (default, ``include_uq_qq=False``) evaluates CLIFF2's
    ``mtp_elst_damping``; ``damping="gordon1"`` evaluates Poisson-consistent
    1-center & 2-center 1s Slater charge penetration
    (:func:`mtp_elst_damping_gordon1_full`), optionally including ``u-Q`` and
    ``Q-Q`` terms.
    """
    from apnet_pt.AtomPairwiseModels.mtp_mtp import (
        mtp_elst_damping,
        mtp_elst_damping_gordon1_full,
    )

    dtype = qA.dtype
    pair = torch.arange(len(e_AB_source), device=e_AB_source.device)

    def ends(x, index):
        return x.to(dtype).index_select(0, index)

    if damping == "cliff":
        if include_uq_qq:
            raise ValueError(
                "include_uq_qq=True requires damping='gordon1' (CLIFF 1-center "
                "damping has an r^3 cusp whose T_4 diverges as 1/r at r -> 0)"
            )
        return mtp_elst_damping(
            ends(ZA, e_AB_source), ends(RA, e_AB_source),
            ends(qA.reshape(-1), e_AB_source), ends(muA, e_AB_source),
            ends(quadA, e_AB_source), alpha_source.to(dtype),
            ends(ZB, e_AB_target), ends(RB, e_AB_target),
            ends(qB.reshape(-1), e_AB_target), ends(muB, e_AB_target),
            ends(quadB, e_AB_target), alpha_target.to(dtype),
            pair, pair, Q_const=Q_const,
        )
    if damping == "gordon1":
        return mtp_elst_damping_gordon1_full(
            ends(ZA, e_AB_source), ends(RA, e_AB_source),
            ends(qA.reshape(-1), e_AB_source), ends(muA, e_AB_source),
            ends(quadA, e_AB_source), alpha_source.to(dtype),
            ends(ZB, e_AB_target), ends(RB, e_AB_target),
            ends(qB.reshape(-1), e_AB_target), ends(muB, e_AB_target),
            ends(quadB, e_AB_target), alpha_target.to(dtype),
            pair, pair, Q_const=Q_const, include_uq_qq=include_uq_qq,
        )
    raise ValueError(f"Unknown electrostatic damping mode: {damping!r}")
