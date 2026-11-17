"""FF-2: parity with FF-1 (one-mu mode), symmetry, conservation, compile."""

import math

import pytest
import torch

from apnet_pt.ff1 import FF1Model, gordon1
from apnet_pt.ff2 import FF2Model, compile_ff2, ff2_loss
from apnet_pt.ff2 import charges, kernels, physics
from apnet_pt.ff2.batching import batch_from_dimers, monomer_batch, with_coordinates

F64 = torch.float64
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]

WATER = [[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]]
NH3 = [[0.0, 0.0, 0.0], [0.94, 0.0, -0.33], [-0.47, 0.81, -0.33],
       [-0.47, -0.81, -0.33]]
ACETATE = ([6, 6, 8, 8, 1, 1, 1],
           [[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.15, 1.07, 0.0],
            [2.15, -1.07, 0.0], [-0.39, 1.02, 0.0], [-0.39, -0.51, 0.88],
            [-0.39, -0.51, -0.88]])
GLYCINE_ZW = ([7, 1, 1, 1, 6, 1, 1, 6, 8, 8],
              [[-1.90, 0.10, 0.0], [-2.30, 1.03, 0.0], [-2.30, -0.38, 0.82],
               [-2.30, -0.38, -0.82], [-0.42, 0.12, 0.0], [-0.07, 0.65, 0.89],
               [-0.07, 0.65, -0.89], [0.17, -1.29, 0.0], [1.42, -1.33, 0.0],
               [-0.57, -2.28, 0.0]])
# two waters as ONE monomer: FF-1's covalent rule would split it, one-mu not
WATER2 = WATER + [[x + 2.9, y, z] for x, y, z in WATER]


def shift(coords, dx=0.0, dy=0.0, dz=0.0):
    return [[x + dx, y + dy, z + dz] for x, y, z in coords]


def dimers():
    return [
        {"zA": [8, 1, 1], "RA": WATER, "zB": [7, 1, 1, 1],
         "RB": shift(NH3, 0.3, 0.4, 3.0)},
        {"zA": ACETATE[0], "RA": ACETATE[1], "qA": -1, "zB": [8, 1, 1],
         "RB": shift(WATER, 0.0, 0.0, 3.2)},
        {"zA": GLYCINE_ZW[0], "RA": GLYCINE_ZW[1], "zB": [7, 1, 1, 1],
         "RB": shift(NH3, 1.0, 0.5, 3.5)},
        {"zA": [8, 1, 1, 8, 1, 1], "RA": WATER2, "qA": 1, "zB": [7, 1, 1, 1],
         "RB": shift(NH3, 0.0, 1.0, -3.1)},
    ]


def make_ff1(seed=0, device="cpu", scale=0.15, **options):
    torch.manual_seed(seed)
    opts = {"g0_init": -2.5, "elst_beta_def": True, "elst_sr_shell": True,
            **options}
    model = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8,
                     trunk_dtype=F64, physics_dtype=F64, **opts).to(device)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn_like(p))
    return model


def random_orthogonal(det, seed=1):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=F64))
    if torch.det(q) * det < 0:
        q[:, 0] = -q[:, 0]
    return q


def grads(model, value):
    out = torch.autograd.grad(value, list(model.parameters()), allow_unused=True)
    return {n: g for (n, _), g in zip(model.named_parameters(), out)}


def assert_parity(ff1, ff2, batch, e_tol=1e-9, g_tol=1e-9):
    p1, x1 = ff1(monomer_batch(batch))
    p2, x2 = ff2(batch)
    assert torch.allclose(p1, p2, rtol=0, atol=e_tol), (p1 - p2).abs().max()
    for key in ("q", "mu", "th", "alpha_iso", "beta", "induced_dipole",
                "e_pol", "e_ct", "exch_pair", "disp_pair", "ct_pair"):
        assert torch.allclose(x1[key], x2[key], rtol=1e-9, atol=1e-11), key
    w = torch.arange(1, 5, dtype=p1.dtype, device=p1.device)
    g1, g2 = grads(ff1, (p1 * w).sum()), grads(ff2, (p2 * w).sum())
    for name, a in g1.items():
        b = g2[name]
        if a is None:
            assert b is None or float(b.abs().max()) == 0.0, name
            continue
        scale = float(a.abs().max()) + 1e-12
        assert float((a - b).abs().max()) <= g_tol * scale, name


# --- FF-1 parity -------------------------------------------------------------

def test_state_dict_is_ff1_v5():
    ff1 = make_ff1()
    ff2 = FF2Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8)
    assert list(ff2.state_dict()) == list(ff1.state_dict())
    assert ff2.config()["g0_init"] == -2.5
    assert ff2.elst_beta_def and ff2.elst_sr_shell


@pytest.mark.parametrize("device", DEVICES)
def test_parity_with_ff1_monomer_mode(device):
    ff1 = make_ff1(0, device)
    ff2 = FF2Model.from_ff1(ff1)
    assert_parity(ff1, ff2, batch_from_dimers(dimers(), device=device))


@pytest.mark.parametrize("options", [{}, {"elst_sr_shell": False},
                                     {"elst_beta_def": False, "elst_sr_shell": False,
                                      "g0_init": -6.0}])
def test_from_ff1_reproduces_reduced_configurations(options):
    """FF-1 models without beta_def/sr shell load into FF-2 exactly."""
    ff1 = make_ff1(2, **options)
    ff2 = FF2Model.from_ff1(ff1)
    assert ff2.g0_init == ff1.g0_init
    assert_parity(ff1, ff2, batch_from_dimers(dimers()))


def test_parity_all_float64_and_independent_weights():
    """An FF-1 cast to float64 throughout (element tables included) loads with
    its dtypes, and the copy does not alias FF-1's parameters."""
    ff1 = make_ff1(10).double()
    ff2 = FF2Model.from_ff1(ff1)
    assert all(p.dtype == F64 for p in ff2.parameters())
    assert_parity(ff1, ff2, batch_from_dimers(dimers()), e_tol=1e-10, g_tol=1e-10)
    with torch.no_grad():
        ff2.gamma_raw.add_(1.0)
    assert not torch.equal(ff2.gamma_raw, ff1.gamma_raw)


def test_parity_with_s_cut():
    ff1 = make_ff1(4, s_cut=(3.0, 4.0))
    ff2 = FF2Model.from_ff1(ff1)
    assert_parity(ff1, ff2, batch_from_dimers(dimers()))


def test_forward_monomers_matches_ff1():
    ff1 = make_ff1(5)
    ff2 = FF2Model.from_ff1(ff1)
    b = batch_from_dimers(dimers())
    _, a1 = ff1.forward_monomers(monomer_batch(b))
    _, a2 = ff2.forward_monomers(b)
    for key in ("q", "mu", "th", "beta", "beta_def", "d_sr", "Q_sr", "sqrt_alpha"):
        assert torch.allclose(a1[key], a2[key], rtol=1e-10, atol=1e-12), key


# --- one-mu charges ----------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_monomer_charge_is_conserved_and_solve_converges(device):
    model = FF2Model.from_ff1(make_ff1(3, device, scale=0.5))
    b = batch_from_dimers(dimers(), device=device)
    _, x = model(b)
    q_mon = torch.zeros(len(b["monomer_charge"]), dtype=F64, device=device)
    q_mon.index_add_(0, b["monomer"], x["q"])
    assert torch.allclose(q_mon, b["monomer_charge"], atol=1e-10)
    assert float(x["mu_residual"].max()) < 1e-12


def test_solve_is_a_fixed_point_once_converged():
    """FF-1 breaks out of Newton at convergence; FF-2's fixed iteration count
    must not walk away from an exact root (f == 0 used to bisect off it)."""
    g = torch.Generator().manual_seed(0)
    mon = torch.tensor([0, 0, 0, 1, 1])
    log_n0 = torch.randn(5, generator=g, dtype=F64)
    s_pos = 0.05 + torch.rand(5, generator=g, dtype=F64)
    s_neg = 0.05 + torch.rand(5, generator=g, dtype=F64)
    log_t = torch.tensor([math.log(5.0), math.log(1.5)], dtype=F64)
    mu5, r5 = charges.solve_mu(log_n0, s_pos, s_neg, mon, log_t, 5)
    mu80, r80 = charges.solve_mu(log_n0, s_pos, s_neg, mon, log_t, 80)
    assert float(r80.max()) < 1e-12
    assert torch.equal(mu5, mu80) or float((mu5 - mu80).abs().max()) < 1e-13


def test_one_mu_gradient_is_the_ift_gradient():
    g = torch.Generator().manual_seed(1)
    mon = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    n0 = (0.5 + torch.rand(7, generator=g, dtype=F64)).requires_grad_(True)
    sp = (0.1 + torch.rand(7, generator=g, dtype=F64)).requires_grad_(True)
    sn = (0.1 + torch.rand(7, generator=g, dtype=F64)).requires_grad_(True)
    target = torch.tensor([2.2, 5.5], dtype=F64)
    w = torch.randn(7, generator=g, dtype=F64)

    def f(a, b, c):
        return (charges.monomer_populations(a, b, c, mon, target)[0] * w).sum()

    assert torch.autograd.gradcheck(f, (n0, sp, sn), atol=1e-7, rtol=1e-6)


def test_bare_proton_monomer():
    model = FF2Model.from_ff1(make_ff1(6))
    b = batch_from_dimers([{"zA": [8, 1, 1], "RA": WATER, "zB": [1],
                            "RB": [[0.0, 0.0, -1.6]], "qB": 1}])
    R = b["R"].clone().requires_grad_(True)
    pred, x = model(with_coordinates(b, R))
    assert torch.isfinite(pred).all()
    assert float(x["q"][3]) == pytest.approx(1.0, abs=1e-14)
    gs = torch.autograd.grad(pred.sum(), [R, *model.parameters()], allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in gs)


# --- symmetry and forces -----------------------------------------------------

@pytest.mark.parametrize("det", [1.0, -1.0])
def test_o3_invariance_and_swap_symmetry(det):
    model = FF2Model.from_ff1(make_ff1(1))
    d = dimers()[0]
    b = batch_from_dimers([d])
    O = random_orthogonal(det)
    t = torch.tensor([0.3, -1.1, 2.0], dtype=F64)
    p0, x0 = model(b)
    p1, x1 = model(with_coordinates(b, b["R"] @ O.T + t))
    assert torch.allclose(p0, p1, atol=1e-9, rtol=1e-10)
    assert torch.allclose(x0["induced_dipole"] @ O.T, x1["induced_dipole"], atol=1e-9)
    swapped = {"zA": d["zB"], "RA": d["RB"], "zB": d["zA"], "RB": d["RA"]}
    p2, _ = model(batch_from_dimers([swapped]))
    assert torch.allclose(p0, p2, atol=1e-9, rtol=1e-10)


def test_forces_match_finite_differences():
    model = FF2Model.from_ff1(make_ff1(9))
    b = batch_from_dimers([dimers()[0]])
    R0 = b["R"].clone().requires_grad_(True)
    pred, _ = model(with_coordinates(b, R0))
    (grad,) = torch.autograd.grad(pred.sum(), R0)
    h = 1e-4
    fd = torch.zeros_like(R0)
    with torch.no_grad():
        for a in range(R0.shape[0]):
            for x in range(3):
                vals = []
                for k in (-2, -1, 1, 2):
                    R = R0.detach().clone()
                    R[a, x] += k * h
                    vals.append(float(model(with_coordinates(b, R))[0].sum()))
                fd[a, x] = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
    assert torch.allclose(grad, fd, atol=1e-6, rtol=1e-6)


# --- kernels -----------------------------------------------------------------

def test_batched_kernels_match_ff1():
    g = torch.Generator().manual_seed(2)
    r = torch.tensor([0.0, 1e-6, 0.4, 1.3, 7.5, 8.0, 12.0], dtype=F64)
    bi = 0.5 + 3 * torch.rand(7, generator=g, dtype=F64)
    bj = 0.5 + 3 * torch.rand(7, generator=g, dtype=F64)
    specs = gordon1.ELST_AA_SPECS
    got = kernels.two_center(bi, bj, r, specs)
    ref = gordon1.two_center(bi, bj, r, specs)
    for s, key in enumerate(specs):
        assert torch.allclose(got[:, s], ref[key], rtol=1e-13, atol=1e-15), key
    got = kernels.one_center(bi, r, gordon1.CHARGE_SPECS)
    ref = gordon1.one_center(bi, r, gordon1.CHARGE_SPECS)
    for s, key in enumerate(gordon1.CHARGE_SPECS):
        assert torch.allclose(got[:, s], ref[key], rtol=1e-13, atol=1e-15), key


def test_expm_sym_matches_matrix_exp():
    g = torch.Generator().manual_seed(3)
    A = torch.randn(64, 3, 3, generator=g, dtype=F64)
    A = 0.5 * (A + A.transpose(1, 2))
    A = A * (0.2 / torch.linalg.matrix_norm(A))[:, None, None]
    assert torch.allclose(physics.expm_sym(A), torch.linalg.matrix_exp(A),
                          rtol=0, atol=1e-15)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_physics_tracks_float64(device):
    ff1 = make_ff1(7, device)
    ff2 = FF2Model.from_ff1(ff1)
    f32 = FF2Model.from_ff1(ff1, physics_dtype=torch.float32)
    b = batch_from_dimers(dimers(), device=device)
    p64, _ = ff2(b)
    p32, _ = f32(b)
    assert p32.dtype == torch.float32
    rel = (p32.double() - p64).abs() / (p64.abs() + 1.0)
    assert float(rel.max()) < 1e-4
    g = torch.autograd.grad(ff2_loss(p32, p64.detach())[0], list(f32.parameters()),
                            allow_unused=True)
    assert all(x is None or torch.isfinite(x).all() for x in g)


# --- torch.compile -----------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
def test_compiled_fullgraph_matches_eager(device):
    torch._dynamo.reset()
    ff1 = make_ff1(8, device)
    eager = FF2Model.from_ff1(ff1)
    compiled = compile_ff2(eager)
    # 3 dimers: a size equal to the coordinate axis must not specialize
    for batch in (batch_from_dimers(dimers(), device=device),
                  batch_from_dimers(dimers()[:3], device=device),
                  batch_from_dimers(dimers()[1:2], device=device),  # m == N, n == 1
                  batch_from_dimers(dimers()[:3] + dimers()[:2], device=device)):
        pe, _ = eager(batch)
        pc, xc = compiled(batch)
        assert torch.allclose(pe, pc, rtol=0, atol=1e-9)
        ge = grads(eager, pe.sum())
        gc = grads(eager, pc.sum())
        for name, a in ge.items():
            if a is not None:  # FF-1's element tables are float32 parameters
                tol = 1e-8 if a.dtype == F64 else 1e-5
                assert torch.allclose(a, gc[name], rtol=tol,
                                      atol=tol * 1e-2 * float(a.abs().max() + 1)), name
    from torch._dynamo.utils import counters
    assert counters["stats"]["unique_graphs"] == 1
