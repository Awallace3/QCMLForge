"""MASTIFF shared-exponent electrostatics/induction terms, independent of weights."""

import pytest
import torch

from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    _rackers_converge_dipoles,
    _rackers_distance_tensors,
    mtp_elst,
    mtp_elst_damping,
)
from apnet_pt.mastiff import RacahHarmonics
from apnet_pt.mastiff_classical import (
    RACAH_L3_DEGREES,
    anisotropic_log_scale,
    mastiff_donor_acceptor_short_range,
    mastiff_short_range,
    racah_l3,
    racah_parity,
    rackers_elst_pair_exponents,
    slater_overlap,
    undamped_multipole_elst,
)

DTYPE = torch.float64
EPS32 = torch.finfo(torch.float32).eps


def unit_vectors(n, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g, dtype=DTYPE)
    return v / v.norm(dim=-1, keepdim=True)


def test_racah_l3_extends_the_l2_basis():
    u = unit_vectors(64, 1)
    torch.testing.assert_close(racah_l3(u)[:, :8], RacahHarmonics()(u))


def test_racah_l3_is_unit_norm_per_degree_and_has_parity():
    u = unit_vectors(64, 2)
    c = racah_l3(u)
    degrees = torch.tensor(RACAH_L3_DEGREES)
    for deg in (1, 2, 3):
        norm = c[:, degrees == deg].square().sum(-1)
        torch.testing.assert_close(norm, torch.ones_like(norm))
    torch.testing.assert_close(racah_l3(-u), c * racah_parity(15, DTYPE))


def test_short_range_is_nonpositive_and_decays():
    g = torch.Generator().manual_seed(3)
    n = 4096
    a_i, a_j = torch.rand(n, generator=g, dtype=DTYPE) * 50, torch.rand(
        n, generator=g, dtype=DTYPE) * 50
    b_i, b_j = 1 + 4 * torch.rand(n, generator=g, dtype=DTYPE), 1 + 4 * torch.rand(
        n, generator=g, dtype=DTYPE)
    r = 0.5 + 10 * torch.rand(n, generator=g, dtype=DTYPE)
    s = 2 * torch.randn(n, generator=g, dtype=DTYPE)
    e = mastiff_short_range(a_i, a_j, b_i, b_j, r, s, -s, s, s)
    assert (e <= 0).all()
    far = mastiff_short_range(a_i, a_j, b_i, b_j, r + 100.0)
    assert far.abs().max() < 1e-30
    assert slater_overlap(torch.zeros(1, dtype=DTYPE)).item() == 1.0


def test_short_range_is_minus_the_isotropic_mastiff_exchange_form():
    a_i, a_j = torch.tensor([3.0, 7.0], dtype=DTYPE), torch.tensor([2.0, 5.0], dtype=DTYPE)
    b_i, b_j = torch.tensor([4.0, 3.5], dtype=DTYPE), torch.tensor([4.5, 3.7], dtype=DTYPE)
    r = torch.tensor([2.5, 3.1], dtype=DTYPE)
    x = torch.sqrt(b_i * b_j) * r
    expected = -a_i * a_j * (1 + x + x * x / 3) * torch.exp(-x)
    torch.testing.assert_close(mastiff_short_range(a_i, a_j, b_i, b_j, r), expected)


def test_exponent_anisotropy_matches_the_eq_l3_contraction():
    # eq-l3: x = sqrt(B_i B_j exp(s_i) exp(s_j)) r with s_j read along -u.
    g = torch.Generator().manual_seed(4)
    d = torch.randn(32, 3, generator=g, dtype=DTYPE) * 3
    ci, cj = (0.1 * torch.randn(32, 15, generator=g, dtype=DTYPE) for _ in range(2))
    s_i, s_j = anisotropic_log_scale(ci, cj, d)
    u = d / d.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(s_i, (ci * racah_l3(u)).sum(-1))
    torch.testing.assert_close(s_j, (cj * racah_l3(-u)).sum(-1))
    # Swapping the ends of an edge swaps the roles exactly.
    t_j, t_i = anisotropic_log_scale(cj, ci, -d)
    torch.testing.assert_close((t_i, t_j), (s_i, s_j))


def random_dimer(seed, separation, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    na, nb = 5, 4
    RA = torch.randn(na, 3, generator=g).to(dtype)
    RB = torch.randn(nb, 3, generator=g).to(dtype) + torch.tensor([separation, 0, 0])
    ZA = torch.tensor([8, 1, 1, 6, 7], dtype=dtype)
    ZB = torch.tensor([6, 1, 1, 8], dtype=dtype)

    def multipoles(n):
        q = 0.4 * torch.randn(n, generator=g).to(dtype)
        mu = 0.3 * torch.randn(n, 3, generator=g).to(dtype)
        Q = 0.3 * torch.randn(n, 3, 3, generator=g).to(dtype)
        Q = 0.5 * (Q + Q.transpose(1, 2))
        Q = Q - torch.eye(3, dtype=dtype) * Q.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
        return q, mu, Q

    qA, muA, QA = multipoles(na)
    qB, muB, QB = multipoles(nb)
    src = torch.arange(na).repeat_interleave(nb)
    tgt = torch.arange(nb).repeat(na)
    return ZA, RA, qA, muA, QA, ZB, RB, qB, muB, QB, src, tgt


def nuclear_scale(ZA, RA, ZB, RB, src, tgt):
    """kcal/mol magnitude of the Z_a Z_b / r terms a nucleus/cloud split carries.

    `mtp_elst` and `mtp_elst_damping` form every edge as a difference of
    these, so their float32 rounding lives at this scale, not at sum |e|.
    """
    d = (RB[tgt] - RA[src]).double().norm(dim=-1)
    return 627.509 * (ZA[src] * ZB[tgt]).double().abs().mul(0.529177).div(d).sum()


def test_undamped_equals_mtp_elst_and_does_not_mutate():
    args = random_dimer(5, 4.0)
    before = (args[2].clone(), args[7].clone())
    ours32 = undamped_multipole_elst(*args)
    torch.testing.assert_close((args[2], args[7]), before)
    ours = undamped_multipole_elst(*random_dimer(5, 4.0, dtype=DTYPE))
    theirs = mtp_elst(*[a.clone() for a in args]).double()
    ZA, RA, _, _, _, ZB, RB, _, _, _, src, tgt = args
    bound = 64 * EPS32 * nuclear_scale(ZA, RA, ZB, RB, src, tgt)
    assert (ours - theirs).abs().sum() <= bound
    # Without the split, our own float32 path rounds at the far smaller sum |e|.
    assert (ours32.double() - ours).abs().sum() <= 64 * EPS32 * ours.abs().sum()


def test_undamped_float64_matches_float32_to_the_cancelling_scale():
    args32 = random_dimer(6, 3.5)
    args64 = random_dimer(6, 3.5, dtype=DTYPE)
    e32 = undamped_multipole_elst(*args32).double()
    e64 = undamped_multipole_elst(*args64)
    bound = 64 * EPS32 * e64.abs().sum()
    assert abs(e32.sum() - e64.sum()) <= bound


@pytest.mark.parametrize("separation", [25.0, 40.0])
def test_rackers_damping_vanishes_at_long_range(separation):
    ZA, RA, qA, muA, QA, ZB, RB, qB, muB, QB, src, tgt = random_dimer(7, separation)
    Ka = torch.full_like(qA, 1.8)
    Kb = torch.full_like(qB, 1.9)
    damped = mtp_elst_damping(ZA, RA, qA.clone(), muA, QA, Ka,
                              ZB, RB, qB.clone(), muB, QB, Kb, src, tgt)
    undamped = undamped_multipole_elst(ZA, RA, qA, muA, QA, ZB, RB, qB, muB, QB, src, tgt)
    # The nuclear split makes each damped edge a difference of O(Z^2/r)
    # terms, so the rounding scale is the uncancelled nuclear magnitude.
    d = (RB[tgt] - RA[src]).norm(dim=-1)
    scale = 627.509 * (ZA[src] * ZB[tgt] * 0.529177 / d).abs().sum()
    assert (damped.sum() - undamped.sum()).abs() <= 64 * EPS32 * scale


def test_rackers_damping_differs_at_short_range():
    args = random_dimer(8, 2.0)
    ZA, RA, qA, muA, QA, ZB, RB, qB, muB, QB, src, tgt = args
    damped = mtp_elst_damping(ZA, RA, qA.clone(), muA, QA, torch.full_like(qA, 1.8),
                              ZB, RB, qB.clone(), muB, QB, torch.full_like(qB, 1.8),
                              src, tgt)
    undamped = undamped_multipole_elst(*args)
    assert (damped.sum() - undamped.sum()).abs() > 1e-2


def test_pair_exponent_rackers_equals_per_atom_rackers():
    args = random_dimer(9, 2.5)
    ZA, RA, qA, muA, QA, ZB, RB, qB, muB, QB, src, tgt = args
    g = torch.Generator().manual_seed(9)
    Ka = 1.5 + torch.rand(len(qA), generator=g)
    Kb = 1.5 + torch.rand(len(qB), generator=g)
    before = (qA.clone(), qB.clone())
    theirs = mtp_elst_damping(ZA, RA, qA.clone(), muA, QA, Ka,
                              ZB, RB, qB.clone(), muB, QB, Kb, src, tgt)
    ours = rackers_elst_pair_exponents(*args, Ka[src], Kb[tgt])
    torch.testing.assert_close((qA, qB), before)
    assert ours.dtype == theirs.dtype
    bound = 64 * EPS32 * nuclear_scale(ZA, RA, ZB, RB, src, tgt)
    assert (ours.double() - theirs.double()).abs().sum() <= bound


def test_pair_exponent_rackers_runs_in_float64_and_differentiates():
    args = random_dimer(10, 2.5, dtype=DTYPE)
    ZA, RA, qA, muA, QA, ZB, RB, qB, muB, QB, src, tgt = args
    scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    alpha_i = scale * torch.full((len(src),), 1.8, dtype=DTYPE)
    alpha_j = scale * torch.linspace(1.6, 2.0, len(src), dtype=DTYPE)
    e = rackers_elst_pair_exponents(*args, alpha_i, alpha_j)
    assert e.dtype == DTYPE and torch.get_default_dtype() == torch.float32
    e.sum().backward()
    assert torch.isfinite(scale.grad) and scale.grad != 0
    # A per-pair exponent is a real degree of freedom: changing one pair's
    # exponent changes that pair's energy only.
    bumped = alpha_j.detach().clone()
    bumped[3] *= 1.2
    e2 = rackers_elst_pair_exponents(*args, alpha_i.detach(), bumped)
    changed = (e2 - e.detach()).abs() > 0
    assert changed.tolist() == [k == 3 for k in range(len(src))]


def _axial_quadrupole_stencil(center, axis, q_tip, s, dtype=DTYPE):
    """Exact 3-site point-charge stencil (+q_tip at +-s*axis, -2*q_tip at center).

    `center` and `s` are in Angstrom; Buckingham's quadrupole tensor `theta` is
    returned in atomic units (`e * bohr^2`) to match `AtomMPNN` / `mtp_elst`:
    Theta = q_tip * s_bohr^2 * (3 axis (x) axis - I), with O(s^4) hexadecapole error.
    """
    from apnet_pt import constants

    axis = axis / axis.norm()
    coords = torch.stack([center + s * axis, center - s * axis, center], dim=0)
    charges = torch.tensor([q_tip, q_tip, -2.0 * q_tip], dtype=dtype)
    s_bohr = s / constants.au2ang
    theta = q_tip * (s_bohr * s_bohr) * (3.0 * torch.outer(axis, axis) - torch.eye(3, dtype=dtype))
    return coords, charges, theta


def test_equal_exponent_damping_is_symmetric_and_smooth_in_float32():
    """Both cliff and gordon1 damping must give identical dE/d(alpha_i) and
    dE/d(alpha_j) on a symmetric homodimer at alpha_i == alpha_j, and vary
    smoothly across alpha_i -> alpha_j in float32 without 1/(a2^2 - a1^2) spikes."""
    for damping, uq_qq in [("cliff", False), ("gordon1", True)]:
        ZA = torch.tensor([6.0], dtype=DTYPE)
        RA = torch.tensor([[0.0, 0.0, 0.0]], dtype=DTYPE)
        RB = torch.tensor([[2.2, 0.0, 0.0]], dtype=DTYPE)
        q = torch.tensor([-0.15], dtype=DTYPE)
        mu = torch.tensor([[0.25, 0.1, -0.05]], dtype=DTYPE)
        mu_mirror = torch.tensor([[-0.25, 0.1, -0.05]], dtype=DTYPE)
        Q = torch.tensor([[[0.4, 0.05, 0.0], [0.05, -0.2, 0.0], [0.0, 0.0, -0.2]]], dtype=DTYPE)
        src = torch.tensor([0])
        tgt = torch.tensor([0])

        ai = torch.tensor([1.85], dtype=DTYPE, requires_grad=True)
        aj = torch.tensor([1.85], dtype=DTYPE, requires_grad=True)
        e = rackers_elst_pair_exponents(
            ZA, RA, q, mu, Q, ZA, RB, q, mu_mirror, Q, src, tgt, ai, aj,
            damping=damping, include_uq_qq=uq_qq,
        )
        e.sum().backward()
        torch.testing.assert_close(ai.grad, aj.grad, rtol=1e-12, atol=1e-12)

        # Float32 scan across u = (ai - aj)/(ai + aj) in [-0.15, 0.15]: compare to float64.
        deltas = torch.logspace(-7, -0.8, 64, dtype=DTYPE)
        deltas = torch.cat([-deltas.flip(0), torch.zeros(1, dtype=DTYPE), deltas])
        ai64 = 1.85 * (1.0 + deltas)
        aj64 = 1.85 * (1.0 - deltas)
        n = len(deltas)
        rep = lambda t: t.expand(n, *t.shape[1:]).contiguous()  # noqa: E731
        idx = torch.arange(n)
        ref64 = rackers_elst_pair_exponents(
            rep(ZA), rep(RA), rep(q), rep(mu), rep(Q),
            rep(ZA), rep(RB), rep(q), rep(mu_mirror), rep(Q),
            idx, idx, ai64, aj64, damping=damping, include_uq_qq=uq_qq,
        )
        got32 = rackers_elst_pair_exponents(
            rep(ZA.float()), rep(RA.float()), rep(q.float()), rep(mu.float()), rep(Q.float()),
            rep(ZA.float()), rep(RB.float()), rep(q.float()), rep(mu_mirror.float()), rep(Q.float()),
            idx, idx, ai64.float(), aj64.float(), damping=damping, include_uq_qq=uq_qq,
        )
        assert torch.isfinite(got32).all()
        assert (got32.double() - ref64).abs().max() < 5e-3


def test_full_multipole_uq_qq_and_ipd_quad_field_match_point_charge_stencils():
    """Verify signs and Q_const=3.0 prefactors of E_uQ, E_QQ, and IPD's
    permanent quadrupole field against exact finite-size point-charge stencils."""
    s = 1e-2  # Angstrom stencil arm -> O(10 * s^2/R^2) ~ 5e-5 relative hexadecapole error
    RA = torch.tensor([[0.0, 0.0, 0.0]], dtype=DTYPE)
    RB = torch.tensor([[4.2, 0.7, -0.5]], dtype=DTYPE)
    axis_a = torch.tensor([0.6, -0.7, 0.3872983346], dtype=DTYPE)
    axis_b = torch.tensor([-0.3, 0.5, 0.8124038405], dtype=DTYPE)
    pts_a, q_a, theta_a = _axial_quadrupole_stencil(RA[0], axis_a, 500.0, s)
    pts_b, q_b, theta_b = _axial_quadrupole_stencil(RB[0], axis_b, -360.0, s)
    zero1 = torch.zeros(1, dtype=DTYPE)
    zero3 = torch.zeros(1, 3, dtype=DTYPE)
    zero33 = torch.zeros(1, 3, 3, dtype=DTYPE)
    z_dummy = torch.ones(1, dtype=DTYPE)
    src0, tgt0 = torch.tensor([0]), torch.tensor([0])

    # 1. Quadrupole-Quadrupole (E_QQ): compare analytical vs 3x3 point-charge Coulomb.
    s9 = torch.arange(3).repeat_interleave(3)
    t9 = torch.arange(3).repeat(3)
    e_qq_stencil = undamped_multipole_elst(
        torch.ones(3, dtype=DTYPE), pts_a, q_a, torch.zeros(3, 3, dtype=DTYPE),
        torch.zeros(3, 3, 3, dtype=DTYPE),
        torch.ones(3, dtype=DTYPE), pts_b, q_b, torch.zeros(3, 3, dtype=DTYPE),
        torch.zeros(3, 3, 3, dtype=DTYPE), s9, t9,
    ).sum()
    e_qq_analytic = undamped_multipole_elst(
        z_dummy, RA, zero1, zero3, theta_a[None],
        z_dummy, RB, zero1, zero3, theta_b[None], src0, tgt0,
        include_uq_qq=True,
    ).sum()
    assert abs(float(e_qq_analytic)) > 1e-4
    torch.testing.assert_close(e_qq_analytic, e_qq_stencil, rtol=1e-4, atol=1e-9)

    # 2. Dipole-Quadrupole (E_uQ): dipole on A vs quadrupole stencil on B, and vice versa.
    delta = 1e-2
    u_dir = torch.tensor([0.4, 0.8, -0.4472135955], dtype=DTYPE)
    # Note: dipole in apnet_pt is in e*bohr, while coordinates are in Angstrom!
    from apnet_pt import constants
    q_dip = 50.0
    mu_vec = (q_dip * (delta / constants.au2ang) * u_dir)[None]
    dip_a_pts = torch.stack([RA[0] + 0.5 * delta * u_dir, RA[0] - 0.5 * delta * u_dir])
    dip_q = torch.tensor([q_dip, -q_dip], dtype=DTYPE)
    s6 = torch.arange(2).repeat_interleave(3)
    t6 = torch.arange(3).repeat(2)
    e_uq_ab_stencil = undamped_multipole_elst(
        torch.ones(2, dtype=DTYPE), dip_a_pts, dip_q, torch.zeros(2, 3, dtype=DTYPE),
        torch.zeros(2, 3, 3, dtype=DTYPE),
        torch.ones(3, dtype=DTYPE), pts_b, q_b, torch.zeros(3, 3, dtype=DTYPE),
        torch.zeros(3, 3, 3, dtype=DTYPE), s6, t6,
    ).sum()
    e_uq_ab_analytic = undamped_multipole_elst(
        z_dummy, RA, zero1, mu_vec, zero33,
        z_dummy, RB, zero1, zero3, theta_b[None], src0, tgt0,
        include_uq_qq=True,
    ).sum()
    assert abs(float(e_uq_ab_analytic)) > 1e-3
    torch.testing.assert_close(e_uq_ab_analytic, e_uq_ab_stencil, rtol=1e-4, atol=1e-9)

    # Swap: quadrupole stencil on A, dipole on B.
    dip_b_pts = torch.stack([RB[0] + 0.5 * delta * u_dir, RB[0] - 0.5 * delta * u_dir])
    e_uq_ba_stencil = undamped_multipole_elst(
        torch.ones(3, dtype=DTYPE), pts_a, q_a, torch.zeros(3, 3, dtype=DTYPE),
        torch.zeros(3, 3, 3, dtype=DTYPE),
        torch.ones(2, dtype=DTYPE), dip_b_pts, dip_q, torch.zeros(2, 3, dtype=DTYPE),
        torch.zeros(2, 3, 3, dtype=DTYPE), t6, s6,
    ).sum()
    e_uq_ba_analytic = undamped_multipole_elst(
        z_dummy, RA, zero1, zero3, theta_a[None],
        z_dummy, RB, zero1, mu_vec, zero33, src0, tgt0,
        include_uq_qq=True,
    ).sum()
    torch.testing.assert_close(e_uq_ba_analytic, e_uq_ba_stencil, rtol=1e-4, atol=1e-9)

    # 3. Gordon1 damped full multipoles converge to undamped full multipoles at long range.
    args_lr = random_dimer(21, 30.0, dtype=DTYPE)
    ZA, RA_lr, qA, muA, QA, ZB, RB_lr, qB, muB, QB, src, tgt = args_lr
    ai = torch.full((len(src),), 2.1, dtype=DTYPE)
    aj = torch.full((len(src),), 1.9, dtype=DTYPE)
    damped_lr = rackers_elst_pair_exponents(
        *args_lr, ai, aj, damping="gordon1", include_uq_qq=True,
    )
    undamped_lr = undamped_multipole_elst(*args_lr, include_uq_qq=True)
    torch.testing.assert_close(damped_lr, undamped_lr, rtol=1e-9, atol=1e-9)

    # 4. Permanent quadrupole field in IPD matches 3-point-charge stencil on both sides.
    alpha1 = torch.tensor([4.5], dtype=DTYPE)
    alpha3 = torch.tensor([4.5, 0.0, 0.0], dtype=DTYPE)  # polarize site 0 only
    empty = torch.zeros(0, dtype=torch.long)
    thole_big = torch.tensor([1e6], dtype=DTYPE)  # undamped limit
    d_ab = _rackers_distance_tensors(
        RA, RB, src0, tgt0, alpha1, alpha1, thole_big, "direct",
        direct_exponent=3.0, include_t3=True,
    )
    m_ab = _rackers_distance_tensors(
        RA, RB, src0, tgt0, alpha1, alpha1, thole_big, "mutual",
    )
    empty_t2 = torch.zeros((0, 3, 3), dtype=DTYPE)
    _, _, mu0_a, mu0_b, *_ = _rackers_converge_dipoles(
        alpha1, alpha1, zero1, zero3, zero1, zero3,
        src0, tgt0, empty, empty, empty, empty,
        d_ab[3], d_ab[4], None, None, None, None,
        m_ab[4], empty_t2, empty_t2,
        50, 1e-12, 0.7,
        quadA=theta_a[None], quadB=theta_b[None],
        direct_T3_AB=(d_ab[1], d_ab[2], d_ab[5], d_ab[6]),
    )
    # Stencil field at RA due to B's 3 point charges:
    s13, t13 = torch.zeros(3, dtype=torch.long), torch.arange(3)
    d_st_b = _rackers_distance_tensors(
        RA, pts_b, s13, t13, alpha1, torch.ones(3, dtype=DTYPE),
        torch.full((3,), 1e6, dtype=DTYPE), "direct", direct_exponent=3.0,
    )
    _, _, mu0_a_st, _, *_ = _rackers_converge_dipoles(
        alpha1, torch.ones(3, dtype=DTYPE), zero1, zero3, q_b, torch.zeros(3, 3, dtype=DTYPE),
        s13, t13, empty, empty, empty, empty,
        d_st_b[3], d_st_b[4], None, None, None, None,
        d_st_b[4], empty_t2, empty_t2, 1, 1e-12, 0.7,
    )
    # Stencil field at RB due to A's 3 point charges:
    d_st_a = _rackers_distance_tensors(
        pts_a, RB, t13, s13, torch.ones(3, dtype=DTYPE), alpha1,
        torch.full((3,), 1e6, dtype=DTYPE), "direct", direct_exponent=3.0,
    )
    _, _, _, mu0_b_st, *_ = _rackers_converge_dipoles(
        torch.ones(3, dtype=DTYPE), alpha1, q_a, torch.zeros(3, 3, dtype=DTYPE), zero1, zero3,
        t13, s13, empty, empty, empty, empty,
        d_st_a[3], d_st_a[4], None, None, None, None,
        d_st_a[4], empty_t2, empty_t2, 1, 1e-12, 0.7,
    )
    del alpha3
    torch.testing.assert_close(mu0_a, mu0_a_st, rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(mu0_b, mu0_b_st, rtol=1e-5, atol=1e-9)


def test_gordon1_full_multipoles_have_pure_repulsive_nuclear_wall_at_short_range():
    """With Gordon1 1-center & 2-center Slater damping, all cloud-cloud (including
    Theta-Theta) and core-cloud (including Z-Theta) terms stay O(1) as r -> 0,
    so r * E_elst(r) -> +627.509 * Z_A * Z_B (in bohr) even for huge hostile quadrupoles."""
    from apnet_pt import constants

    ZA = torch.tensor([6.0], dtype=DTYPE)
    ZB = torch.tensor([8.0], dtype=DTYPE)
    qA = torch.tensor([-0.4], dtype=DTYPE)
    qB = torch.tensor([-0.6], dtype=DTYPE)
    muA = torch.tensor([[1.5, -0.8, 0.4]], dtype=DTYPE)
    muB = torch.tensor([[-1.2, 0.9, -0.3]], dtype=DTYPE)
    # Hostile large quadrupoles that would diverge to -infinity as -1/r under CLIFF T4!
    QA = torch.tensor([[[4.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -2.0]]], dtype=DTYPE)
    QB = torch.tensor([[[-4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]], dtype=DTYPE)
    RA = torch.zeros(1, 3, dtype=DTYPE)
    src0, tgt0 = torch.tensor([0]), torch.tensor([0])
    ai = torch.tensor([2.3], dtype=DTYPE)
    aj = torch.tensor([2.7], dtype=DTYPE)

    for r_bohr in (0.05, 0.01, 1e-3):
        RB = torch.tensor([[r_bohr * constants.au2ang, 0.0, 0.0]], dtype=DTYPE)
        e = rackers_elst_pair_exponents(
            ZA, RA, qA, muA, QA, ZB, RB, qB, muB, QB,
            src0, tgt0, ai, aj, damping="gordon1", include_uq_qq=True,
        )
        assert float(e) > 0.0
    # Check asymptotic limit lim_{r->0} r_bohr * E_elst(r) = 627.509 * ZA * ZB.
    r_tiny = 1e-4
    RB_tiny = torch.tensor([[r_tiny * constants.au2ang, 0.0, 0.0]], dtype=DTYPE)
    e_tiny = rackers_elst_pair_exponents(
        ZA, RA, qA, muA, QA, ZB, RB_tiny, qB, muB, QB,
        src0, tgt0, ai, aj, damping="gordon1", include_uq_qq=True,
    )
    expected_z_wall = 627.509 * float(ZA * ZB)
    assert abs(float(e_tiny) * r_tiny - expected_z_wall) / expected_z_wall < 1e-3


def test_donor_acceptor_short_range_cancels_like_pairs_and_keeps_attraction():
    """-(D_i C_j + C_i D_j) P(x) exp(-x) is <= 0, symmetric under i <-> j, and
    strictly zero for cation-cation (D=0, C>0) while attractive for cation-donor."""
    b = torch.tensor([3.5, 3.5], dtype=DTYPE)
    r = torch.tensor([2.2, 2.2], dtype=DTYPE)
    # Edge 0: Na+..Na+ (D=0, C=4 -> zero!). Edge 1: Na+..OH2 (D_i=0, C_i=4; D_j=3, C_j=0.5 -> < 0!).
    d_i = torch.tensor([0.0, 0.0], dtype=DTYPE)
    c_i = torch.tensor([4.0, 4.0], dtype=DTYPE)
    d_j = torch.tensor([0.0, 3.0], dtype=DTYPE)
    c_j = torch.tensor([4.0, 0.5], dtype=DTYPE)
    e = mastiff_donor_acceptor_short_range(d_i, c_i, d_j, c_j, b, b, r)
    assert float(e[0]) == 0.0
    assert float(e[1]) < 0.0
    e_swap = mastiff_donor_acceptor_short_range(d_j, c_j, d_i, c_i, b, b, r)
    torch.testing.assert_close(e, e_swap)

