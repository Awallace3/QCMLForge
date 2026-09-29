"""MASTIFF shared-exponent electrostatics/induction terms, independent of weights."""

import pytest
import torch

from apnet_pt.AtomPairwiseModels.mtp_mtp import mtp_elst, mtp_elst_damping
from apnet_pt.mastiff import RacahHarmonics
from apnet_pt.mastiff_classical import (
    RACAH_L3_DEGREES,
    anisotropic_log_scale,
    mastiff_short_range,
    racah_l3,
    racah_parity,
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
    for l in (1, 2, 3):
        norm = c[:, degrees == l].square().sum(-1)
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
