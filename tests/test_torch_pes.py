"""Tests for torch_pes (forked torch-pme + multipolar SPME + dispersion LJPME
+ cell-list neighbor list) and its FF-2 long-range MD integration."""

from __future__ import annotations

import math

import pytest
import torch
from torch._dynamo.utils import counters

import torch_pes
from apnet_pt.ff1 import FF1Model
from apnet_pt.ff2 import (
    FF2LongRangeEnergy,
    FF2Model,
    compile_long_range_energy,
    openmm_python_force,
    physics,
)
from apnet_pt.ff2.md import FF2Energy
from torch_pes.calculators.dispersion_pme import (
    DispersionEwald,
    DispersionPME,
    homogeneous_tail_correction,
)
from torch_pes.calculators.multipole_ewald import MultipoleEwald
from torch_pes.calculators.multipole_pme import MultipolePME
from torch_pes.calculators.neighbor_list import (
    NeighborList,
    _build_allpairs_nonperiodic,
    _build_allpairs_periodic,
)
from torch_pes.lib.boys import (
    dipole_tensor_coeffs,
    dispersion_r6_erf,
    dispersion_r6_erfc,
    ewald_boys_erf,
    ewald_boys_erfc,
    ewald_boys_point,
    multipole_pair_energy,
    multipole_pair_field,
)
from torch_pes.lib.bsplines import (
    bspline_modulus_sq_1d,
    bspline_modulus_sq_3d,
    bspline_weights,
    next_smooth_size,
)
from torch_pes.tuning import tune_dispersion_pme, tune_multipole_pme

F64 = torch.float64

WATER = [[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]]
NH3 = [
    [0.0, 0.0, 0.0],
    [0.94, 0.0, -0.33],
    [-0.47, 0.81, -0.33],
    [-0.47, -0.81, -0.33],
]


def _rand_traceless(n: int, gen: torch.Generator) -> torch.Tensor:
    t = torch.randn(n, 3, 3, generator=gen, dtype=F64)
    t = 0.5 * (t + t.transpose(-1, -2))
    tr = (t[:, 0, 0] + t[:, 1, 1] + t[:, 2, 2]) / 3.0
    return t - tr[:, None, None] * torch.eye(3, dtype=F64)


def _small_ff2(seed: int = 0, scale: float = 0.10) -> FF2Model:
    torch.manual_seed(seed)
    ff1 = FF1Model(
        n_feat=32,
        n_chan=4,
        n_layers=2,
        n_rbf=8,
        trunk_dtype=F64,
        physics_dtype=F64,
        g0_init=-2.5,
        elst_beta_def=True,
        elst_sr_shell=True,
    )
    with torch.no_grad():
        for p in ff1.parameters():
            p.add_(scale * torch.randn_like(p))
    return FF2Model.from_ff1(ff1).eval()


def _periodic_water_nh3_box():
    """8-monomer (4 water + 4 NH3, 28 atoms) triclinic periodic cell."""
    box_nm = torch.tensor(
        [[1.6, 0.05, 0.0], [0.02, 1.55, 0.03], [0.0, -0.02, 1.65]], dtype=F64
    )
    z, mon, q, pos = [], [], [], []
    m_id = 0
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                base = [ix * 0.78 + 0.15, iy * 0.75 + 0.15, iz * 0.80 + 0.15]
                if (ix + iy + iz) % 2 == 0:
                    for p in WATER:
                        pos.append(
                            [base[0] + p[0] / 10, base[1] + p[1] / 10, base[2] + p[2] / 10]
                        )
                    z += [8, 1, 1]
                    mon += [m_id] * 3
                    q.append(0.0)
                else:
                    for p in NH3:
                        pos.append(
                            [base[0] + p[0] / 10, base[1] + p[1] / 10, base[2] + p[2] / 10]
                        )
                    z += [7, 1, 1, 1]
                    mon += [m_id] * 4
                    q.append(0.0)
                m_id += 1
    return z, mon, q, torch.tensor(pos, dtype=F64), box_nm


# --- 1. Forked torch-pme sanity check ----------------------------------------


def test_forked_torch_pme_ewald_and_pme_agree():
    """Core forked lab-cosmo/torch-pme calculators remain functional."""
    cell = torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=F64))
    pos = torch.tensor(
        [[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [2.5, 8.0, 1.5], [7.5, 3.0, 6.5]],
        dtype=F64,
    )
    charges = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]], dtype=F64)
    ii, jj = torch.triu_indices(4, 4, offset=1)
    disp = pos[jj] - pos[ii]
    disp = disp - torch.round(disp / 10.0) * 10.0
    dists = torch.linalg.norm(disp, dim=-1)
    idx = torch.stack([ii, jj], dim=-1)

    pot = torch_pes.CoulombPotential(smearing=1.2)
    ew = torch_pes.EwaldCalculator(pot, lr_wavelength=1.0)
    pme = torch_pes.PMECalculator(pot, mesh_spacing=0.6, interpolation_nodes=5)

    v_ew = ew(charges, cell, pos, idx, dists)
    v_pme = pme(charges, cell, pos, idx, dists)
    assert torch.allclose(v_ew, v_pme, atol=1e-4, rtol=1e-4)


# --- 2. Cardinal B-splines ---------------------------------------------------


@pytest.mark.parametrize("order", [4, 5, 6, 7])
def test_bspline_partition_of_unity_and_exact_derivatives(order):
    delta = torch.linspace(0.01, 0.99, 50, dtype=F64, requires_grad=True)
    w0, w1, w2 = bspline_weights(delta, order=order, deriv_order=2)
    assert torch.allclose(w0.sum(-1), torch.ones_like(delta), atol=1e-14)
    assert torch.allclose(w1.sum(-1), torch.zeros_like(delta), atol=1e-14)
    assert torch.allclose(w2.sum(-1), torch.zeros_like(delta), atol=1e-14)

    # Autograd derivatives of w0 and w1 match exact recurrence w1 and w2
    for k in range(order):
        (dw0,) = torch.autograd.grad(w0[:, k].sum(), delta, create_graph=True)
        assert torch.allclose(dw0, w1[:, k], atol=1e-14)
        (dw1,) = torch.autograd.grad(w1[:, k].sum(), delta, retain_graph=True)
        assert torch.allclose(dw1, w2[:, k], atol=1e-14)


@pytest.mark.parametrize("order", [4, 5, 6])
def test_bspline_modulus_finite_and_symmetric(order):
    m1 = bspline_modulus_sq_1d(16, order=order, rfft=False)
    m3 = bspline_modulus_sq_3d((16, 18, 20), order=order)
    assert bool(torch.isfinite(m1).all())
    assert bool(torch.isfinite(m3).all())
    assert float(m1[0]) == pytest.approx(1.0, abs=1e-14)
    assert m3.shape == (16, 18, 11)
    assert next_smooth_size(25) == 30
    assert next_smooth_size(31) == 32


# --- 3. Boys functions & FF-2 point-multipole parity -------------------------


def test_boys_and_dispersion_kernels_exact_split_and_continuity():
    alpha = 0.35
    r = torch.cat(
        [
            torch.linspace(1e-3, 15.0, 200, dtype=F64),
            torch.tensor(
                [1.0 / alpha - 1e-8, 1.0 / alpha, 1.0 / alpha + 1e-8], dtype=F64
            ),
        ]
    ).requires_grad_(True)
    b_erf = ewald_boys_erf(r, alpha)
    b_erfc = ewald_boys_erfc(r, alpha)
    b_pt = ewald_boys_point(r)
    for ell in range(5):
        rel = (b_erf[ell] + b_erfc[ell] - b_pt[ell]).abs() / b_pt[ell].abs()
        assert float(rel.detach().max()) < 1e-13
        # Check d B_l / dr = -r B_{l+1} for l < 4
        if ell < 4:
            (db,) = torch.autograd.grad(b_erf[ell].sum(), r, retain_graph=True)
            assert torch.allclose(db, -r * b_erf[ell + 1], rtol=1e-11, atol=1e-12)

    d_erf = dispersion_r6_erf(r, alpha)
    d_erfc = dispersion_r6_erfc(r, alpha)
    assert float(((d_erf + d_erfc - r**-6).abs() * r**6).detach().max()) < 1e-13
    # Finite at r = 0
    r0 = torch.tensor([0.0], dtype=F64)
    assert float(dispersion_r6_erf(r0, alpha)[0]) == pytest.approx(
        alpha**6 / 6.0, rel=1e-14
    )


def test_multipole_pair_kernels_match_ff2_point_limit():
    gen = torch.Generator().manual_seed(42)
    p = 40
    ri = torch.randn(p, 3, generator=gen, dtype=F64)
    rj = ri + 3.0 + torch.randn(p, 3, generator=gen, dtype=F64)
    dvec = rj - ri
    dist = torch.linalg.norm(dvec, dim=-1)
    rhat = dvec / dist[:, None]
    qi = torch.randn(p, generator=gen, dtype=F64)
    qj = torch.randn(p, generator=gen, dtype=F64)
    mui = torch.randn(p, 3, generator=gen, dtype=F64)
    muj = torch.randn(p, 3, generator=gen, dtype=F64)
    thi, thj = _rand_traceless(p, gen), _rand_traceless(p, gen)
    dsr_i = torch.randn(p, 3, generator=gen, dtype=F64)
    dsr_j = torch.randn(p, 3, generator=gen, dtype=F64)
    qsr_i, qsr_j = _rand_traceless(p, gen), _rand_traceless(p, gen)

    def mk_side(q, mu, th, dsr, qsr):
        z = torch.full_like(q, 8.0)
        nc = torch.full_like(q, 2.0)
        b = torch.full_like(q, 2.0)
        return {
            "z": z,
            "nc": nc,
            "bc": b,
            "n": z - nc - q,
            "b": b,
            "q": q,
            "M": torch.stack([mu + dsr, -dsr], 1),
            "T": torch.stack([th + qsr, -qsr], 1),
            "bs": torch.stack([b, b], 1),
        }

    si, sj = mk_side(qi, mui, thi, dsr_i, qsr_i), mk_side(qj, muj, thj, dsr_j, qsr_j)
    boys_pt = ewald_boys_point(dist)
    e_ff2 = physics.elst_pairs(dist, rhat, si, sj, mode="point")
    e_pes = multipole_pair_energy(dist, rhat, qi, mui, thi, qj, muj, thj, boys_pt)
    assert torch.allclose(e_ff2, e_pes, atol=1e-14, rtol=1e-14)

    f_ff2 = physics.permanent_field(
        dist, rhat, torch.full_like(dist, 2.0), si, mode="point"
    )
    f_pes = multipole_pair_field(dist, rhat, qi, mui, thi, boys_pt)
    assert torch.allclose(f_ff2, f_pes, atol=1e-14, rtol=1e-14)

    l33, l53 = dipole_tensor_coeffs(dist, boys_pt)
    assert torch.allclose(l33, dist**-3, atol=1e-14)
    assert torch.allclose(l53, dist**-3, atol=1e-14)


# --- 4. Multipolar Ewald & SPME (triclinic) ----------------------------------


def test_multipole_ewald_alpha_invariance_and_spme_parity_triclinic():
    gen = torch.Generator().manual_seed(9)
    n = 12
    cell = torch.tensor(
        [[16.0, 0.5, 0.0], [0.4, 15.0, 0.3], [0.2, -0.3, 17.0]], dtype=F64
    )
    pos = torch.rand(n, 3, generator=gen, dtype=F64) @ cell
    q = torch.randn(n, generator=gen, dtype=F64)
    q = q - q.mean()
    mu = torch.randn(n, 3, generator=gen, dtype=F64) * 0.5
    th = _rand_traceless(n, gen) * 0.4

    ew1 = MultipoleEwald(alpha=0.25, k_max=10)
    ed1, fd1 = ew1.direct_energy_and_field(pos, cell, q, mu, th, r_cut=24.0)
    er1, fr1 = ew1.reciprocal_energy_and_field(pos, cell, q, mu, th)

    ew2 = MultipoleEwald(alpha=0.35, k_max=12)
    ed2, fd2 = ew2.direct_energy_and_field(pos, cell, q, mu, th, r_cut=20.0)
    er2, fr2 = ew2.reciprocal_energy_and_field(pos, cell, q, mu, th)

    assert abs(float((ed1 + er1) - (ed2 + er2))) < 1e-12
    assert float(((fd1 + fr1) - (fd2 + fr2)).abs().max()) < 1e-12

    # SPME vs direct Ewald for energy, field, forces, multipole gradients, dipole matvec
    pme = MultipolePME(alpha=0.25, ns_mesh=(40, 40, 40), order=6)
    pos_e, q_e, mu_e, th_e = [t.clone().requires_grad_(True) for t in (pos, q, mu, th)]
    pos_p, q_p, mu_p, th_p = [t.clone().requires_grad_(True) for t in (pos, q, mu, th)]
    e_ew, f_ew = ew1.reciprocal_energy_and_field(pos_e, cell, q_e, mu_e, th_e)
    e_pm, f_pm, st = pme.reciprocal_energy_and_field(pos_p, cell, q_p, mu_p, th_p)

    assert abs(float((e_ew - e_pm).detach())) < 1e-7
    assert float((f_ew - f_pm).detach().abs().max()) < 1e-7

    g_ew = torch.autograd.grad(e_ew, (pos_e, q_e, mu_e, th_e))
    g_pm = torch.autograd.grad(e_pm, (pos_p, q_p, mu_p, th_p))
    for a, b in zip(g_ew, g_pm):
        assert float((a - b).abs().max()) < 1e-6

    p_ind = torch.randn(n, 3, generator=gen, dtype=F64)
    mv_ew = ew1.reciprocal_dipole_matvec(pos, cell, p_ind)
    mv_pm = pme.reciprocal_dipole_matvec(p_ind, st)
    assert float((mv_ew - mv_pm).detach().abs().max()) < 1e-7

    # Reciprocal dipole energy matches 0.5 * p . (T_rec @ p)
    e_dip = pme.reciprocal_dipole_energy(p_ind, st)
    assert abs(float((e_dip - 0.5 * (p_ind * mv_pm).sum()).detach())) < 1e-14


# --- 5. Dispersion LJPME & Casimir-Polder low-rank factorization -------------


def test_dispersion_ewald_and_pme_and_casimir_polder_convergence():
    gen = torch.Generator().manual_seed(17)
    cell = torch.tensor(
        [[18.0, 0.6, 0.0], [0.3, 17.0, 0.4], [0.2, -0.2, 19.0]], dtype=F64
    )
    n = 16
    pos = torch.rand(n, 3, generator=gen, dtype=F64) @ cell
    alpha_iso = 2.0 + 10.0 * torch.rand(n, generator=gen, dtype=F64)
    eta = 0.1 + 1.8 * torch.rand(n, generator=gen, dtype=F64)

    # Casimir-Polder exact matrix vs Chebyshev-Nystrom ranks 2, 4, 6
    c6_exact = (
        alpha_iso[:, None]
        * alpha_iso[None, :]
        * 2.0
        * eta[:, None]
        * eta[None, :]
        / (eta[:, None] + eta[None, :])
    )
    errs = []
    for rank in (2, 4, 6):
        dp = DispersionPME(alpha=0.2, ns_mesh=(24, 24, 24), rank=rank)
        fac = dp.factors(alpha_iso, eta)
        c6_approx = fac @ fac.T
        rel_err = float(((c6_approx - c6_exact).abs() / c6_exact).max())
        errs.append(rel_err)
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] < 3e-3
    assert errs[2] < 1e-4

    # Alpha-invariance of DispersionEwald & parity of DispersionPME
    dp = DispersionPME(alpha=0.20, ns_mesh=(30, 30, 30), order=4, rank=4)
    fac = dp.factors(alpha_iso, eta)
    dew1 = DispersionEwald(alpha=0.20, k_max=8)
    dew2 = DispersionEwald(alpha=0.28, k_max=10)
    e1 = dew1.direct_energy(pos, cell, fac, r_cut=24.0) + dew1.reciprocal_energy(
        pos, cell, fac
    )
    e2 = dew2.direct_energy(pos, cell, fac, r_cut=20.0) + dew2.reciprocal_energy(
        pos, cell, fac
    )
    assert abs(float(e1 - e2)) < 1e-12

    pos_e = pos.clone().requires_grad_(True)
    pos_p = pos.clone().requires_grad_(True)
    er_ew = dew1.reciprocal_energy(pos_e, cell, fac)
    er_pm = dp.reciprocal_energy(pos_p, cell, fac)
    (ge,) = torch.autograd.grad(er_ew, pos_e)
    (gp,) = torch.autograd.grad(er_pm, pos_p)
    assert abs(float((er_ew - er_pm).detach())) < 1e-6
    assert float((ge - gp).abs().max()) < 1e-6


# --- 6. Cell-linked NeighborList & Tuners ------------------------------------


def test_neighbor_list_cell_matches_brute_force_and_skin_caching():
    gen = torch.Generator().manual_seed(23)
    cell = torch.tensor(
        [[35.0, 1.5, 0.0], [0.8, 33.0, 1.0], [0.5, -0.7, 36.0]], dtype=F64
    )
    pos = (torch.rand(120, 3, generator=gen, dtype=F64) * 2.5 - 0.5) @ cell
    mon = torch.arange(120) // 3

    nl = NeighborList(r_cut=8.0, skin=1.5, monomer=mon)
    inter, all_b = nl.update(pos, cell)
    assert nl.n_rebuilds == 1

    m = all_b.mask
    d_cell = (
        torch.linalg.norm(
            pos[all_b.aj[m]] - pos[all_b.ai[m]] + all_b.shift[m] @ cell, dim=-1
        )
        .sort()
        .values
    )
    ai_r, aj_r, sh_r = _build_allpairs_periodic(pos, cell, 9.5)
    d_ref = (
        torch.linalg.norm(pos[aj_r] - pos[ai_r] + sh_r @ cell, dim=-1).sort().values
    )
    assert d_cell.shape == d_ref.shape
    assert torch.allclose(d_cell, d_ref, atol=1e-12)

    # Small move (< skin/2) skips rebuild; large move triggers rebuild
    pos_small = pos + 0.1 * torch.randn(120, 3, generator=gen, dtype=F64)
    nl.update(pos_small, cell)
    assert nl.n_rebuilds == 1
    pos_large = pos + 1.0 * torch.randn(120, 3, generator=gen, dtype=F64)
    nl.update(pos_large, cell)
    assert nl.n_rebuilds == 2

    # Non-periodic 500 atoms exercises cell-list branch
    pos_np = torch.randn(500, 3, generator=gen, dtype=F64) * 18.0
    nl_np = NeighborList(r_cut=7.5, skin=1.0)
    _, buf_np = nl_np.update(pos_np, None)
    m_np = buf_np.mask
    d_np = torch.linalg.norm(pos_np[buf_np.aj[m_np]] - pos_np[buf_np.ai[m_np]], dim=-1).sort().values
    ai2, aj2, _ = _build_allpairs_nonperiodic(pos_np, 8.5)
    d_np_ref = torch.linalg.norm(pos_np[aj2] - pos_np[ai2], dim=-1).sort().values
    assert torch.allclose(d_np, d_np_ref, atol=1e-12)

    # Tuners return smooth grids
    mp = tune_multipole_pme(cell, cutoff=18.9, accuracy=1e-6)
    dp = tune_dispersion_pme(cell, cutoff=18.9, accuracy=1e-5)
    assert mp.alpha > 0 and all(k >= 8 and k % 2 == 0 for k in mp.ns_mesh)
    assert dp.alpha > 0 and all(k >= 8 and k % 2 == 0 for k in dp.ns_mesh)


# --- 7. FF-2 Long-Range MD Integration (non-periodic & periodic) -------------


def test_ff2_long_range_cluster_parity_with_exact_ff2():
    """Non-periodic FF2LongRangeEnergy with large cutoff reproduces all-pairs
    FF2Energy to <= 1e-9 kJ/mol in energy and forces."""
    model = _small_ff2(seed=1)
    z = [8, 1, 1, 7, 1, 1, 1, 8, 1, 1]
    mon = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    q = [0.0, 0.0, 0.0]
    coords_ang = (
        WATER
        + [[x + 0.4, y + 0.5, z_ + 3.2] for x, y, z_ in NH3]
        + [[x - 2.5, y + 2.2, z_ + 1.8] for x, y, z_ in WATER]
    )
    x_nm = torch.tensor(coords_ang, dtype=F64) / 10.0

    ref = FF2Energy(model, z, mon, q)
    xr = x_nm.clone().requires_grad_(True)
    er = ref(xr)
    (gr,) = torch.autograd.grad(er, xr)

    lr = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=50.0,
        skin=1.0,
        pbc=False,
        cutoff_switch="none",
        pol_cg_iters=25,
    )
    e_lr, f_lr = lr.energy_forces(x_nm)
    assert abs(float((e_lr - er).detach())) < 1e-9
    assert float((f_lr + gr).abs().max()) < 1e-8


@pytest.mark.parametrize("cutoff_switch", ["smooth", "group"])
def test_ff2_nonperiodic_smooth_cutoff_autograd_matches_fd(cutoff_switch):
    """Smooth quintic cutoff (atom-based and monomer-COM group-based) in
    non-periodic mode gives exact autograd forces across the switching region."""
    model = _small_ff2(seed=2)
    z = [8, 1, 1, 8, 1, 1]
    mon = [0, 0, 0, 1, 1, 1]
    q = [0.0, 0.0]
    # Place second water right inside the switching window [4.0, 6.0] A
    coords_ang = WATER + [[x, y, z_ + 5.0] for x, y, z_ in WATER]
    x_nm = torch.tensor(coords_ang, dtype=F64) / 10.0

    lr = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=6.0,
        skin=1.0,
        pbc=False,
        cutoff_switch=cutoff_switch,
        switch_width=2.0,
        pol_cg_iters=25,
    )
    _, f = lr.energy_forces(x_nm)
    eps = 1e-6
    for i in (0, 3):
        for d in range(3):
            xp = x_nm.clone()
            xm = x_nm.clone()
            xp[i, d] += eps
            xm[i, d] -= eps
            f_fd = -(float(lr(xp)) - float(lr(xm))) / (2.0 * eps)
            assert abs(float(f[i, d]) - f_fd) < 1e-4


def test_ff2_periodic_spme_alpha_wrapping_and_fd_forces():
    """Periodic FF2LongRangeEnergy satisfies alpha-invariance, atom-wrapping
    invariance, and exact finite-difference forces (including dq/dR and CG
    envelope theorem)."""
    model = _small_ff2(seed=0)
    z, mon, q, x_p, box_nm = _periodic_water_nh3_box()

    pme_a = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=12.0,
        skin=1.0,
        cell=box_nm,
        elst_alpha=0.19,
        elst_mesh=(48, 48, 48),
        elst_order=6,
        disp_mode="pme_lowrank",
        disp_rank=4,
        disp_alpha=0.16,
        disp_mesh=(36, 36, 36),
        disp_order=4,
        pol_cg_iters=25,
    )
    pme_b = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=12.0,
        skin=1.0,
        cell=box_nm,
        elst_alpha=0.23,
        elst_mesh=(54, 54, 54),
        elst_order=6,
        disp_mode="pme_lowrank",
        disp_rank=4,
        disp_alpha=0.20,
        disp_mesh=(48, 48, 48),
        disp_order=4,
        pol_cg_iters=25,
    )
    ea, fa = pme_a.energy_forces(x_p, box_nm)
    eb, fb = pme_b.energy_forces(x_p, box_nm)
    assert abs(float(ea - eb)) < 1e-3
    assert float((fa - fb).abs().max()) < 1e-2

    # Atom-level random lattice wrapping leaves energy & forces unchanged
    gen = torch.Generator().manual_seed(11)
    shifts = torch.randint(-2, 3, (len(z), 3), generator=gen, dtype=F64)
    ew, fw = pme_a.energy_forces(x_p + shifts @ box_nm, box_nm)
    assert abs(float(ea - ew)) < 1e-10
    assert float((fa - fw).abs().max()) < 1e-9

    # Finite-difference forces match autograd
    eps = 1e-6
    for i in (0, 4):
        for d in range(3):
            xp = x_p.clone()
            xm = x_p.clone()
            xp[i, d] += eps
            xm[i, d] -= eps
            f_fd = -(float(pme_a(xp, box_nm)) - float(pme_a(xm, box_nm))) / (2.0 * eps)
            assert abs(float(fa[i, d]) - f_fd) < 1e-4


@pytest.mark.parametrize("disp_mode", ["pme_lowrank", "pme_geom", "tail", "cutoff"])
def test_ff2_periodic_dispersion_modes(disp_mode):
    model = _small_ff2(seed=3)
    z, mon, q, x_p, box_nm = _periodic_water_nh3_box()
    lr = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=8.0,
        skin=1.0,
        cell=box_nm,
        accuracy=1e-5,
        disp_mode=disp_mode,
        pol_cg_iters=15,
    )
    e, f = lr.energy_forces(x_p, box_nm)
    assert bool(torch.isfinite(e))
    assert bool(torch.isfinite(f).all())


def test_openmm_periodic_nve_energy_conservation():
    """Periodic 8-water box in OpenMM 8.5 via PythonForce conserves NVE energy."""
    mm = pytest.importorskip("openmm")
    import numpy as np

    model = _small_ff2(seed=0, scale=0.05)
    L_nm = 1.4
    box_nm = torch.diag(torch.tensor([L_nm, L_nm, L_nm], dtype=F64))
    z_w, mon_w, q_w, pos_w = [], [], [], []
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(L_nm, 0, 0), mm.Vec3(0, L_nm, 0), mm.Vec3(0, 0, L_nm)
    )
    bonds = mm.HarmonicBondForce()
    angles = mm.HarmonicAngleForce()
    # Equilibrium water geometry matching harmonic force field (r0=0.09572 nm)
    r0, th0 = 0.09572, 1.8242
    water_eq = [
        [0.0, 0.0, 0.0],
        [r0 * math.sin(th0 / 2.0), 0.0, r0 * math.cos(th0 / 2.0)],
        [-r0 * math.sin(th0 / 2.0), 0.0, r0 * math.cos(th0 / 2.0)],
    ]
    m_id = 0
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                base = [ix * 0.70 + 0.2, iy * 0.70 + 0.2, iz * 0.70 + 0.2]
                o_idx = len(z_w)
                for k, p in enumerate(water_eq):
                    pos_w.append([base[0] + p[0], base[1] + p[1], base[2] + p[2]])
                    system.addParticle(15.999 if k == 0 else 1.008)
                bonds.addBond(o_idx, o_idx + 1, r0, 462750.4)
                bonds.addBond(o_idx, o_idx + 2, r0, 462750.4)
                angles.addAngle(o_idx + 1, o_idx, o_idx + 2, th0, 836.8)
                z_w += [8, 1, 1]
                mon_w += [m_id] * 3
                q_w.append(0.0)
                m_id += 1
    system.addForce(bonds)
    system.addForce(angles)

    ff2_pme = FF2LongRangeEnergy(
        model,
        z_w,
        mon_w,
        q_w,
        r_cut=6.5,
        skin=1.0,
        cell=box_nm,
        accuracy=1e-6,
        disp_mode="pme_lowrank",
        disp_rank=4,
        pol_cg_iters=20,
    )
    system.addForce(openmm_python_force(ff2_pme, force_group=1))
    # Verify O(dt^2) symplectic NVE conservation at 0.5 fs and 0.25 fs
    stds = []
    for dt in (0.0005, 0.00025):
        integ = mm.VerletIntegrator(dt)
        ctx = mm.Context(
            system, integ, mm.Platform.getPlatformByName("CPU"), {"Threads": "1"}
        )
        ctx.setPositions(np.array(pos_w))
        ctx.setVelocitiesToTemperature(150.0, 42)
        n_steps = int(round(0.015 / dt))
        etots = []
        for _ in range(n_steps):
            st = ctx.getState(getEnergy=True)
            etots.append(
                st.getPotentialEnergy()._value + st.getKineticEnergy()._value
            )
            integ.step(1)
        stds.append(float(np.std(etots)))
    assert stds[0] < 0.2
    assert stds[1] < 0.05
    # Shadow Hamiltonian fluctuation scales as dt^2 (ratio ~ 4)
    assert stds[0] / stds[1] > 3.0


@pytest.mark.slow
@pytest.mark.parametrize("pbc", [False, True])
def test_compiled_long_range_matches_eager_and_no_recompile_on_rebuild(pbc):
    """compile_long_range_energy compiles to one graph and does not recompile
    across neighbor-list rebuilds of the same static capacity."""
    model = _small_ff2(seed=4)
    model.kernel_mode = "table"
    z, mon, q, x_p, box_nm = _periodic_water_nh3_box()
    cell = box_nm if pbc else None

    counters.clear()
    eager = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=10.0,
        skin=1.5,
        cell=cell,
        pbc=pbc,
        accuracy=1e-6,
        pol_cg_iters=15,
    )
    comp = compile_long_range_energy(
        model,
        z,
        mon,
        q,
        x_p,
        cell_nm=cell,
        r_cut=10.0,
        skin=1.5,
        pbc=pbc,
        accuracy=1e-6,
        pol_cg_iters=15,
    )
    e1, f1 = comp.energy_forces(x_p, cell)
    frames_1 = counters["frames"]["total"]
    er1, fr1 = eager.energy_forces(x_p, cell)
    assert abs(float(e1 - er1)) < 1e-8
    assert float((f1 - fr1).abs().max()) < 1e-7

    # Perturb enough to trigger NL rebuild without overflowing capacity
    gen = torch.Generator().manual_seed(99)
    x_p2 = x_p + 0.09 * torch.randn(x_p.shape, generator=gen, dtype=F64)
    e2, f2 = comp.energy_forces(x_p2, cell)
    frames_2 = counters["frames"]["total"]
    er2, fr2 = eager.energy_forces(x_p2, cell)
    assert comp.module.neighbor_list.n_rebuilds >= 2
    assert frames_2 == frames_1
    assert abs(float(e2 - er2)) < 1e-8
    assert float((f2 - fr2).abs().max()) < 1e-7


@pytest.mark.parametrize("pbc", [False, True])
def test_pair_chunk_size_matches_unchunked(pbc):
    """Gradient-checkpointed real-space pair chunking reproduces unchunked
    energy and forces to machine precision."""
    model = _small_ff2(seed=7)
    model.kernel_mode = "table"
    z, mon, q, x_p, box_nm = _periodic_water_nh3_box()
    cell = box_nm if pbc else None

    ref = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=10.0,
        skin=1.0,
        cell=cell,
        pbc=pbc,
        accuracy=1e-6,
        pol_cg_iters=15,
        pair_chunk_size=None,
    )
    chunked = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=10.0,
        skin=1.0,
        cell=cell,
        pbc=pbc,
        accuracy=1e-6,
        pol_cg_iters=15,
        pair_chunk_size=64,
    )
    e_ref, f_ref = ref.energy_forces(x_p, cell)
    e_ch, f_ch = chunked.energy_forces(x_p, cell)
    assert abs(float(e_ch - e_ref)) < 1e-11
    assert float((f_ch - f_ref).abs().max()) < 1e-10


@pytest.mark.slow
def test_compiled_pair_chunk_size_matches_eager_and_no_recompile():
    """compile_long_range_energy(..., pair_chunk_size=C) compiles ONE static
    chunk kernel with zero recompiles across chunks and NL rebuilds."""
    model = _small_ff2(seed=11)
    model.kernel_mode = "table"
    z, mon, q, x_p, box_nm = _periodic_water_nh3_box()

    counters.clear()
    eager = FF2LongRangeEnergy(
        model,
        z,
        mon,
        q,
        r_cut=10.0,
        skin=1.5,
        cell=box_nm,
        pbc=True,
        accuracy=1e-6,
        pol_cg_iters=15,
    )
    comp_ch = compile_long_range_energy(
        model,
        z,
        mon,
        q,
        x_p,
        cell_nm=box_nm,
        r_cut=10.0,
        skin=1.5,
        pbc=True,
        accuracy=1e-6,
        pol_cg_iters=15,
        pair_chunk_size=64,
    )
    e1, f1 = comp_ch.energy_forces(x_p, box_nm)
    frames_1 = counters["frames"]["total"]
    er1, fr1 = eager.energy_forces(x_p, box_nm)
    assert abs(float(e1 - er1)) < 1e-8
    assert float((f1 - fr1).abs().max()) < 1e-7

    gen = torch.Generator().manual_seed(123)
    x_p2 = x_p + 0.09 * torch.randn(x_p.shape, generator=gen, dtype=F64)
    e2, f2 = comp_ch.energy_forces(x_p2, box_nm)
    frames_2 = counters["frames"]["total"]
    er2, fr2 = eager.energy_forces(x_p2, box_nm)
    assert comp_ch.module.neighbor_list.n_rebuilds >= 2
    assert frames_2 == frames_1
    assert abs(float(e2 - er2)) < 1e-8
    assert float((f2 - fr2).abs().max()) < 1e-7


def test_ff2_mixed_precision_accuracy_and_nve_drift():
    """Mixed precision (fp64 state/PME/CG/sums + fp32 pair kernels) produces
    float64 conservative forces with tight fp64 parity and O(dt^2) NVE stability."""
    openmm = pytest.importorskip("openmm")
    import numpy as np
    import openmm as mm

    model = _small_ff2(seed=13)
    model.kernel_mode = "table"
    z, mon, q, x_p, box_nm = _periodic_water_nh3_box()

    lr64 = FF2LongRangeEnergy(
        model, z, mon, q, r_cut=10.0, skin=1.0, cell=box_nm, pbc=True,
        accuracy=1e-6, pol_cg_iters=15, precision="fp64",
    )
    lr_mix = FF2LongRangeEnergy(
        model, z, mon, q, r_cut=10.0, skin=1.0, cell=box_nm, pbc=True,
        accuracy=1e-6, pol_cg_iters=15, precision="mixed",
    )
    lr32 = FF2LongRangeEnergy(
        model, z, mon, q, r_cut=10.0, skin=1.0, cell=box_nm, pbc=True,
        accuracy=1e-6, pol_cg_iters=15, precision="fp32",
    )

    e64, f64 = lr64.energy_forces(x_p, box_nm)
    e_mix, f_mix = lr_mix.energy_forces(x_p, box_nm)
    e32, f32 = lr32.energy_forces(x_p, box_nm)
    assert e_mix.dtype == F64 and f_mix.dtype == F64
    assert abs(float(e_mix - e64)) < 1e-3
    assert float((f_mix - f64).abs().max()) < 5e-2
    # Mixed precision should be at least as accurate as pure fp32 on energy
    assert abs(float(e_mix - e64)) <= abs(float(e32 - e64)) + 1e-4

    # OpenMM periodic NVE conservation check for mixed precision
    r0, th0 = 0.09572, 1.8242
    water_eq = np.array(
        [
            [0.0, 0.0, 0.0],
            [r0 * math.sin(th0 / 2.0), 0.0, r0 * math.cos(th0 / 2.0)],
            [-r0 * math.sin(th0 / 2.0), 0.0, r0 * math.cos(th0 / 2.0)],
        ]
    )
    z_w, mon_w, q_w, pos_w = [], [], [], []
    shifts = [[0.2, 0.2, 0.2], [0.2, 0.85, 0.85], [0.85, 0.2, 0.85], [0.85, 0.85, 0.2]]
    for m_idx, sh in enumerate(shifts):
        z_w.extend([8, 1, 1])
        mon_w.extend([m_idx] * 3)
        q_w.append(0.0)
        pos_w.extend((water_eq + np.array(sh)).tolist())
    box_w = torch.diag(torch.tensor([1.3, 1.3, 1.3], dtype=F64))

    lr_nve = FF2LongRangeEnergy(
        model, z_w, mon_w, q_w, r_cut=5.5, skin=1.0, cell=box_w, pbc=True,
        accuracy=1e-6, pol_cg_iters=15, precision="mixed",
    )
    system = mm.System()
    system.setDefaultPeriodicBoxVectors([1.3, 0, 0], [0, 1.3, 0], [0, 0, 1.3])
    for atomic_num in z_w:
        system.addParticle(15.999 if atomic_num == 8 else 1.008)
    hb = mm.HarmonicBondForce()
    hb.setUsesPeriodicBoundaryConditions(True)
    ha = mm.HarmonicAngleForce()
    ha.setUsesPeriodicBoundaryConditions(True)
    for m_idx in range(4):
        o = 3 * m_idx
        hb.addBond(o, o + 1, 0.09572, 462750.4)
        hb.addBond(o, o + 2, 0.09572, 462750.4)
        ha.addAngle(o + 1, o, o + 2, 1.824218, 836.8)
    system.addForce(hb)
    system.addForce(ha)
    system.addForce(openmm_python_force(lr_nve, force_group=1))

    stds = []
    for dt in (0.0005, 0.00025):
        integ = mm.VerletIntegrator(dt)
        ctx = mm.Context(
            system, integ, mm.Platform.getPlatformByName("CPU"), {"Threads": "1"}
        )
        ctx.setPositions(np.array(pos_w))
        ctx.setVelocitiesToTemperature(150.0, 42)
        n_steps = int(round(0.015 / dt))
        etots = []
        for _ in range(n_steps):
            st = ctx.getState(getEnergy=True)
            etots.append(st.getPotentialEnergy()._value + st.getKineticEnergy()._value)
            integ.step(1)
        stds.append(float(np.std(etots)))
    assert stds[0] < 0.2
    assert stds[1] < 0.05
    assert stds[0] / stds[1] > 3.0


@pytest.mark.slow
@pytest.mark.parametrize("chunk", [None, 64])
def test_ff2_mixed_precision_compiled(chunk):
    """compile_long_range_energy(..., precision='mixed') compiles cleanly with
    zero recompiles for both unchunked and static-chunked paths."""
    model = _small_ff2(seed=14)
    model.kernel_mode = "table"
    z, mon, q, x_p, box_nm = _periodic_water_nh3_box()

    counters.clear()
    eager = FF2LongRangeEnergy(
        model, z, mon, q, r_cut=10.0, skin=1.5, cell=box_nm, pbc=True,
        accuracy=1e-6, pol_cg_iters=15, precision="mixed",
    )
    comp = compile_long_range_energy(
        model, z, mon, q, x_p, cell_nm=box_nm, r_cut=10.0, skin=1.5, pbc=True,
        accuracy=1e-6, pol_cg_iters=15, precision="mixed", pair_chunk_size=chunk,
    )
    e1, f1 = comp.energy_forces(x_p, box_nm)
    frames_1 = counters["frames"]["total"]
    er1, fr1 = eager.energy_forces(x_p, box_nm)
    assert e1.dtype == F64 and f1.dtype == F64
    assert abs(float(e1 - er1)) < 1e-4
    assert float((f1 - fr1).abs().max()) < 1e-3

    gen = torch.Generator().manual_seed(202)
    x_p2 = x_p + 0.09 * torch.randn(x_p.shape, generator=gen, dtype=F64)
    e2, f2 = comp.energy_forces(x_p2, box_nm)
    frames_2 = counters["frames"]["total"]
    er2, fr2 = eager.energy_forces(x_p2, box_nm)
    assert comp.module.neighbor_list.n_rebuilds >= 2
    assert frames_2 == frames_1
    assert abs(float(e2 - er2)) < 1e-4
    assert float((f2 - fr2).abs().max()) < 1e-3



