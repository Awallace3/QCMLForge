"""rackers_thole_induction(solver="exact"): the dense per-dimer solve of the
same SCF equations the omega-damped loop relaxes."""

import pytest
import torch

from apnet_pt import constants
from apnet_pt.AtomPairwiseModels.mtp_mtp import rackers_thole_induction


def _all_pairs(n_a, n_b, off_a=0, off_b=0):
    s, t = torch.meshgrid(torch.arange(n_a), torch.arange(n_b), indexing="ij")
    return s.flatten() + off_a, t.flatten() + off_b


def _intra(n, off=0):
    s, t = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = s != t
    return s[keep] + off, t[keep] + off


def batch(seed=0, sizes=((3, 2), (2, 4), (1, 3)), separation=4.0, hfvr=1.0):
    """Several dimers in one batch, per-dimer edges only."""
    g = torch.Generator().manual_seed(seed)
    dt = torch.float64
    RA, RB, ZA, ZB, ia, ib = [], [], [], [], [], []
    ab_s, ab_t, aa_s, aa_t, bb_s, bb_t = ([] for _ in range(6))
    na = nb = 0
    for d, (n_a, n_b) in enumerate(sizes):
        RA.append(torch.rand(n_a, 3, generator=g, dtype=dt) * 2.0)
        RB.append(torch.rand(n_b, 3, generator=g, dtype=dt) * 2.0
                  + torch.tensor([separation, 0.0, 0.0], dtype=dt))
        ZA.append(torch.randint(1, 9, (n_a,), generator=g))
        ZB.append(torch.randint(1, 9, (n_b,), generator=g))
        ia.append(torch.full((n_a,), d))
        ib.append(torch.full((n_b,), d))
        s, t = _all_pairs(n_a, n_b, na, nb)
        ab_s.append(s), ab_t.append(t)
        s, t = _intra(n_a, na)
        aa_s.append(s), aa_t.append(t)
        s, t = _intra(n_b, nb)
        bb_s.append(s), bb_t.append(t)
        na, nb = na + n_a, nb + n_b
    cat = torch.cat
    RA, RB, ZA, ZB = cat(RA), cat(RB), cat(ZA), cat(ZB)
    q = lambda n: (torch.rand(n, generator=g, dtype=dt) - 0.5)  # noqa: E731
    mu = lambda n: (torch.rand(n, 3, generator=g, dtype=dt) - 0.5) * 0.1  # noqa: E731
    return dict(
        ZA=ZA, RA=RA, qA=q(na), muA=mu(na), quadA=torch.zeros(na, 3, 3, dtype=dt),
        ZB=ZB, RB=RB, qB=q(nb), muB=mu(nb), quadB=torch.zeros(nb, 3, 3, dtype=dt),
        e_AB_source=cat(ab_s), e_AB_target=cat(ab_t),
        e_AA_source=cat(aa_s), e_BB_source=cat(bb_s),
        e_AA_target=cat(aa_t), e_BB_target=cat(bb_t),
        hirshfeld_volume_ratio_A=torch.full((na,), hfvr, dtype=dt),
        hirshfeld_volume_ratio_B=torch.full((nb,), hfvr, dtype=dt),
        valence_widths_A=torch.ones(na, dtype=dt), valence_widths_B=torch.ones(nb, dtype=dt),
        thole_direct_A=torch.full((na,), 0.38539, dtype=dt),
        thole_direct_B=torch.full((nb,), 0.38539, dtype=dt),
        thole_mutual_A=torch.full((na,), 0.38539, dtype=dt),
        thole_mutual_B=torch.full((nb,), 0.38539, dtype=dt),
        ind_overlap_A=torch.zeros(na, dtype=dt), ind_overlap_B=torch.zeros(nb, dtype=dt),
        polarizability_table=constants.polarizability_table.double(),
        molecule_ind_A=cat(ia), molecule_ind_B=cat(ib),
        convergence_threshold=1e-12, max_iterations=2000, convergence_norm="max",
    )


@pytest.mark.parametrize("variational", [False, True])
def test_exact_matches_converged_iteration(variational):
    p = batch()
    it, d_it = rackers_thole_induction(**p, variational_energy=variational,
                                       return_diagnostics=True)
    ex, d_ex = rackers_thole_induction(**p, variational_energy=variational,
                                       return_diagnostics=True, solver="exact")
    assert d_it["scf_converged"] and d_ex["scf_converged"]
    assert d_ex["scf_iterations"] == 0 and float(d_ex["scf_residual"]) < 1e-10
    torch.testing.assert_close(ex, it, rtol=1e-9, atol=1e-11)


def test_iterative_default_is_unchanged():
    p = batch(seed=3)
    torch.testing.assert_close(rackers_thole_induction(**p),
                               rackers_thole_induction(**p, solver="iterative"),
                               rtol=0, atol=0)


def test_exact_gradients_match_iteration():
    grads = []
    for solver in ("iterative", "exact"):
        p = batch(seed=1)
        p["muA"].requires_grad_(True)
        p["hirshfeld_volume_ratio_B"].requires_grad_(True)
        rackers_thole_induction(**p, solver=solver).sum().backward()
        grads.append((p["muA"].grad, p["hirshfeld_volume_ratio_B"].grad))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-10)


def test_exact_solves_where_damped_loop_diverges():
    # strongly coupled but positive definite: the omega=0.7 loop does not
    # converge in 200 steps, the exact solve satisfies the same equations
    p = batch(seed=0, sizes=((3, 3),), separation=0.8, hfvr=3.0)
    p.update(max_iterations=200, convergence_threshold=1e-8)
    _, d_it = rackers_thole_induction(**p, return_diagnostics=True)
    ex, d_ex = rackers_thole_induction(**p, solver="exact", return_diagnostics=True)
    assert not d_it["scf_converged"]
    assert torch.isfinite(ex).all() and d_ex["scf_converged"]
    assert float(d_ex["scf_residual"]) < 1e-12


def test_catastrophe_raises():
    # weak Thole damping (large a) at short range: no stable response exists
    p = batch(seed=0, sizes=((3, 3),), separation=0.5, hfvr=5.0)
    for k in ("thole_direct_A", "thole_direct_B", "thole_mutual_A", "thole_mutual_B"):
        p[k] = torch.full_like(p[k], 2.0)
    with pytest.raises(RuntimeError, match="polarization catastrophe"):
        rackers_thole_induction(**p, solver="exact")


def test_exact_needs_molecule_indices_and_known_solver():
    p = batch()
    p.pop("molecule_ind_A")
    with pytest.raises(ValueError, match="molecule_ind"):
        rackers_thole_induction(**p, solver="exact")
    with pytest.raises(ValueError, match="solver"):
        rackers_thole_induction(**batch(), solver="cg")
