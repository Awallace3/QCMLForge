"""FF-2: parity with FF-1 (one-mu mode), symmetry, conservation, compile."""

import math

import pytest
import torch
from torch._dynamo.utils import counters

from apnet_pt.ff1 import FF1Model, gordon1
from apnet_pt.ff2 import FF2Model, compile_ff2, ff2_loss
from apnet_pt.ff2 import charges, kernels, physics
from apnet_pt.ff2.batching import batch_from_dimers, monomer_batch, with_coordinates
from apnet_pt.ff2.md import (KCAL2KJ, FF2Energy, cluster_batch, compile_energy,
                             trace_energy)

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
    counters.clear()
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
    assert counters["stats"]["unique_graphs"] == 1


@pytest.mark.skipif(not torch.backends.opt_einsum.is_available(),
                    reason="opt_einsum not installed: nothing to specialize on")
def test_compile_ignores_opt_einsum_paths():
    # with opt_einsum on, FF-1's three-operand trunk einsums choose their path
    # from the concrete atom count, guarding it; Phoenix job 15791285 then
    # recompiled every chunk.  Dynamo's own backend traces the same guards.
    torch._dynamo.reset()
    counters.clear()
    compiled = compile_ff2(FF2Model.from_ff1(make_ff1(9)), backend="eager")
    with torch.backends.opt_einsum.flags(enabled=True):
        for batch in (batch_from_dimers(dimers()),
                      batch_from_dimers(dimers()[:3]),
                      batch_from_dimers(dimers()[2:3] + dimers()[:1])):
            compiled(batch)
    assert counters["stats"]["unique_graphs"] == 1


# --- delayed charge transfer -------------------------------------------------

def ct_masks(model):
    """{parameter name: bool mask} of the CT-only entries."""
    params = dict(model.named_parameters())
    names = {id(p): n for n, p in params.items()}
    out = {}
    for p, dim, idx in model.ct_parameter_slices():
        mask = out.setdefault(names[id(p)], torch.zeros_like(p, dtype=torch.bool))
        mask.index_fill_(dim, torch.as_tensor(idx, device=p.device), True)
    return out


def test_ct_slices_reach_only_induction():
    model = FF2Model.from_ff1(make_ff1(11))
    batch = batch_from_dimers(dimers())
    p0, x0 = model(batch)
    masks = ct_masks(model)
    assert set(masks) == {"guess.weight", "prop_head.2.weight", "prop_head.2.bias",
                          "vec_readout.weight", "ten_readout.weight", "kappa_ct_raw"}
    torch.manual_seed(3)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.add_(0.3 * torch.randn_like(p) * masks[name])
    p1, x1 = model(batch)
    assert torch.equal(p0[:, [0, 1, 3]], p1[:, [0, 1, 3]])
    assert torch.equal(x0["e_pol"], x1["e_pol"])
    assert float((x1["e_ct"] - x0["e_ct"]).abs().max()) > 1e-6


def test_ct_grad_scale_gates_only_the_ct_gradient():
    model = FF2Model.from_ff1(make_ff1(12))
    batch = batch_from_dimers(dimers())
    w = torch.arange(1, 5, dtype=F64)
    p1, x1 = model(batch)
    g1 = grads(model, (p1 * w).sum())
    model.ct_grad_scale.fill_(0.0)
    p0, x0 = model(batch)
    assert torch.equal(p0, p1)
    g0 = grads(model, (p0 * w).sum())
    masks = ct_masks(model)
    moved = 0
    for name, g in g0.items():
        if g is None:
            continue
        if name in masks:
            assert float(g[masks[name]].abs().max()) == 0.0, name
            moved += int(float(g1[name][masks[name]].abs().max()) > 0)
    assert moved == len(masks)
    # the shared trunk loses CT's share of its gradient
    trunk = [n for n in g0 if n.startswith("trunk.") and g0[n] is not None]
    assert any(not torch.allclose(g0[n], g1[n]) for n in trunk)
    assert "ct_grad_scale" not in model.state_dict()


@pytest.mark.slow
def test_compiled_ct_grad_scale_needs_no_recompile():
    torch._dynamo.reset()
    counters.clear()
    model = FF2Model.from_ff1(make_ff1(13))
    compiled = compile_ff2(model)
    batch = batch_from_dimers(dimers())
    masks = ct_masks(model)
    for s in (1.0, 0.0, 0.5):
        model.ct_grad_scale.fill_(s)
        pc, _ = compiled(batch)
        pe, _ = model(batch)
        gc, ge = grads(model, pc.sum()), grads(model, pe.sum())
        for name, m in masks.items():
            assert torch.allclose(gc[name][m], ge[name][m], rtol=1e-6, atol=1e-10), (s, name)
            if s == 0.0:
                assert float(gc[name][m].abs().max()) == 0.0, name
    assert counters["stats"]["unique_graphs"] == 1


# --- fixed-topology MD potential (OpenMM TorchForce) --------------------------


def md_model(seed=9, **options):
    return FF2Model.from_ff1(make_ff1(seed, **options)).eval()


def layout(dimer):
    na, nb = len(dimer["zA"]), len(dimer["zB"])
    z = list(dimer["zA"]) + list(dimer["zB"])
    x = torch.cat([torch.as_tensor(dimer["RA"], dtype=F64),
                   torch.as_tensor(dimer["RB"], dtype=F64)]) / 10.0
    return z, [0] * na + [1] * nb, [dimer.get("qA", 0), dimer.get("qB", 0)], x


def reference(model, dimer, x):
    """kJ/mol and forces from FF-2 on a batch built at ``x`` (nm)."""
    na = len(dimer["zA"])
    xr = x.clone().requires_grad_(True)
    R = xr * 10.0
    b = batch_from_dimers([{**dimer, "RA": R[:na].detach(), "RB": R[na:].detach()}],
                          r_cut=model.trunk.r_cut)
    b["R"] = R
    e = model(b)[0].sum() * KCAL2KJ
    (g,) = torch.autograd.grad(e, xr)
    return e.detach(), -g


def evaluate(module, x):
    xr = x.clone().requires_grad_(True)
    e = module(xr)
    (g,) = torch.autograd.grad(e, xr)
    return e.detach(), -g


@pytest.mark.parametrize("k", range(4))
def test_traced_energy_and_forces_match_ff2(k):
    model, dimer = md_model(), dimers()[k]
    z, mon, q, x0 = layout(dimer)
    traced = trace_energy(model, z, mon, q, x0)
    gen = torch.Generator().manual_seed(k)
    for step in range(3):
        x = x0 + 0.01 * step * torch.randn(x0.shape, generator=gen, dtype=F64)
        e_ref, f_ref = reference(model, dimer, x)
        e, f = evaluate(traced, x)
        assert torch.allclose(e, e_ref, rtol=0, atol=1e-9 * (1 + e_ref.abs()))
        assert torch.allclose(f, f_ref, rtol=0, atol=1e-8 * (1 + f_ref.abs().max()))


def test_intramonomer_pair_crossing_r_cut_is_exact():
    """A 2.5 Angstrom cutoff: stretch water's O-H through it."""
    model = md_model(r_cut=2.5)
    dimer = {"zA": [8, 1, 1], "RA": WATER, "zB": [7, 1, 1, 1],
             "RB": shift(NH3, 0.3, 0.4, 3.0)}
    z, mon, q, x0 = layout(dimer)
    traced = trace_energy(model, z, mon, q, x0)
    for h_x in (0.20, 0.249, 0.251, 0.28):   # nm: H1 in, at and past r_cut
        x = x0.clone()
        x[1, 0], x[1, 2] = h_x, 0.0
        e_ref, f_ref = reference(model, dimer, x)
        e, f = evaluate(traced, x)
        assert torch.allclose(e, e_ref, rtol=0, atol=1e-9 * (1 + e_ref.abs()))
        assert torch.allclose(f, f_ref, rtol=0, atol=1e-8 * (1 + f_ref.abs().max()))


def test_traced_forces_match_finite_differences():
    model, dimer = md_model(), dimers()[1]
    z, mon, q, x0 = layout(dimer)
    traced = trace_energy(model, z, mon, q, x0)
    _, f = evaluate(traced, x0)
    h = 1e-6
    fd = torch.zeros_like(x0)
    with torch.no_grad():
        for a in range(x0.shape[0]):
            for c in range(3):
                vals = []
                for s in (-2, -1, 1, 2):
                    x = x0.clone()
                    x[a, c] += s * h
                    vals.append(traced(x))
                fd[a, c] = -(vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
    assert torch.allclose(f, fd, rtol=1e-6, atol=1e-5)


def test_saved_trace_round_trips(tmp_path):
    model, dimer = md_model(), dimers()[2]
    z, mon, q, x0 = layout(dimer)
    path = tmp_path / "ff2.pt"
    torch.jit.save(trace_energy(model, z, mon, q, x0), str(path))
    loaded = torch.jit.load(str(path))
    x = x0 + 0.005
    assert torch.equal(loaded(x), trace_energy(model, z, mon, q, x0)(x))


def test_float32_positions_are_promoted():
    model, dimer = md_model(), dimers()[0]
    z, mon, q, x0 = layout(dimer)
    traced = trace_energy(model, z, mon, q, x0)
    e64 = traced(x0.float().double())
    e32 = traced(x0.float())
    assert e32.dtype == F64 and torch.equal(e32, e64)


def test_cluster_pair_terms_add_over_dimers():
    """Three monomers: elst, exch, CT and dispersion are pairwise, so they sum
    over the three dimers; only polarization is many-body."""
    model = md_model()
    monos = [([8, 1, 1], WATER), ([7, 1, 1, 1], shift(NH3, 0.3, 0.4, 3.0)),
             ([8, 1, 1], shift(WATER, 3.1, -0.2, 1.0))]
    z = sum((m[0] for m in monos), [])
    mon = sum(([k] * len(m[0]) for k, m in enumerate(monos)), [])
    x = torch.cat([torch.as_tensor(m[1], dtype=F64) for m in monos]) / 10.0
    energy = FF2Energy(model, z, mon, [0, 0, 0])
    with torch.no_grad():
        pred, extras = model(energy.batch(x))
        total = pred[0] - torch.tensor([0, 0, 1, 0], dtype=F64) * extras["e_pol"][0]
        pairs = torch.zeros(4, dtype=F64)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            d = {"zA": monos[a][0], "RA": monos[a][1],
                 "zB": monos[b][0], "RB": monos[b][1]}
            p, xd = model(batch_from_dimers([d]))
            pairs += p[0] - torch.tensor([0, 0, 1, 0], dtype=F64) * xd["e_pol"][0]
    assert torch.allclose(total, pairs, rtol=0, atol=1e-9)


def test_cluster_batch_validates_monomer_labels():
    with pytest.raises(ValueError):
        cluster_batch([8, 1, 1], [0, 0, 2], [0.0, 0.0])
    b = cluster_batch([8, 1, 1, 7], [0, 0, 0, 1], [0.0, 0.0])
    assert b["edges"].shape == (2, 6) and b["ai"].numel() == 3


# --- speed paths: point-multipole far field ----------------------------------


def test_point_kernel_is_the_large_separation_limit():
    bi = torch.tensor([1.3, 2.0], dtype=F64)[:, None]
    bj = torch.tensor([2.1, 1.1], dtype=F64)[:, None]
    for x in (40.0, 60.0):
        r = torch.full((2, 1), x / 1.1, dtype=F64)
        for specs in (kernels.MM, kernels.CHARGE_SHELL, kernels.DIPOLE_T, kernels.BAR1):
            exact = kernels.two_center(bi, bj, r, specs)
            pt = kernels.point(r, exact.shape[:-1], specs)
            tol = 1e-11 * pt.abs().max() + (1e-16 if specs == kernels.BAR1 else 0.0)
            assert float((exact - pt).abs().max()) <= tol, (x, specs)


def three_monomers(far=12.0):
    """Water-ammonia close contact plus a second water ``far`` Angstrom off."""
    z = [8, 1, 1, 7, 1, 1, 1, 8, 1, 1]
    R = WATER + shift(NH3, 0.3, 0.4, 3.0) + shift(WATER, far, 0.5, 1.0)
    return z, [0] * 3 + [1] * 4 + [2] * 3, [0, 0, 0], torch.tensor(R, dtype=F64) / 10


@pytest.mark.parametrize("far", [6.0, 12.0, 25.0, 80.0])
def test_point_x_split_matches_every_pair_damped(far):
    model = md_model()
    z, mon, q, x = three_monomers(far)
    e_ref, f_ref = evaluate(FF2Energy(model, z, mon, q), x)
    model.point_x = 40.0
    e, f = evaluate(FF2Energy(model, z, mon, q), x)
    if far == 80.0:  # pairs with the third water reach the point limit
        b = cluster_batch(z, mon, q)
        b["R"] = x * 10
        with torch.no_grad():
            R_b, at = model._forward_atoms(b)
            i, j = b["ai"], b["aj"]
            r = (R_b[j] - R_b[i]).norm(dim=-1)
            zf = b["z"].to(F64)
            xm = r * torch.minimum(model._beta_min(model._side(i, zf, at)),
                                   model._beta_min(model._side(j, zf, at)))
        assert 0 < int((xm > 40.0).sum()) < len(xm)  # both branches run
    assert abs(float(e - e_ref)) <= 1e-9 * (1 + abs(float(e_ref)))
    assert float((f - f_ref).abs().max()) <= 1e-8 * (1 + float(f_ref.abs().max()))


def test_point_x_trace_follows_a_changing_partition():
    """Traced with the third water far away, evaluated with it close."""
    model = md_model()
    model.point_x = 40.0
    z, mon, q, x_far = three_monomers(25.0)
    traced = trace_energy(model, z, mon, q, x_far)
    exact = md_model()
    for far in (25.0, 6.0):
        x = three_monomers(far)[3]
        e_ref, f_ref = evaluate(FF2Energy(exact, z, mon, q), x)
        e, f = evaluate(traced, x)
        assert abs(float(e - e_ref)) <= 1e-9 * (1 + abs(float(e_ref))), far
        assert float((f - f_ref).abs().max()) <= 1e-8 * (1 + float(f_ref.abs().max())), far


# --- speed paths: tabulated reduced functions --------------------------------


@pytest.mark.parametrize("specs", [kernels.MM, kernels.CHARGE_SHELL, kernels.DIPOLE_T,
                                   kernels.SHELL_FIELD, kernels.BAR1])
def test_tabulated_two_center_matches_series(specs):
    g = torch.Generator().manual_seed(3)
    bi = 1.0 + 14.0 * torch.rand(400, 1, generator=g, dtype=F64)
    bj = 1.0 + 3.0 * torch.rand(400, 1, generator=g, dtype=F64)
    r = 1.5 + 30.0 * torch.rand(400, 1, generator=g, dtype=F64)
    exact = kernels.two_center(bi, bj, r, specs)
    tab = kernels.two_center_table(bi, bj, r, specs)
    scale = r[..., None] ** -torch.tensor([float(p) for _, p in specs], dtype=F64)
    assert float(((tab - exact) / scale).abs().max()) < 1e-10


@pytest.mark.parametrize("two_center", [True, False])
def test_tabulated_reduced_functions_across_the_series_switch(two_center):
    """A table node sits exactly on X_SWITCH, where _reduced changes branch;
    its autograd slope must come from the branch that is evaluated there
    (torch >= 2.15 gives clamp a zero gradient exactly at its bound)."""
    specs = kernels.MM if two_center else kernels.CHARGE_SHELL
    x = torch.linspace(kernels.X_SWITCH - 0.05, kernels.X_SWITCH + 0.05, 2001, dtype=F64)
    ps = torch.tensor([float(p) for _, p in specs], dtype=F64)
    d = (kernels._reduced_table(x, specs, two_center)
         - kernels._reduced(x, specs, two_center)) * x[:, None] ** ps
    assert float(d.abs().max()) < 1e-10


@pytest.mark.parametrize("specs", [kernels.CHARGE_SHELL, kernels.CHARGE_FIELD,
                                   kernels.BAR1])
def test_tabulated_one_center_matches_series(specs):
    b = torch.linspace(0.8, 16.0, 300, dtype=F64)[:, None]
    r = torch.linspace(1.5, 30.0, 300, dtype=F64)[:, None]
    exact = kernels.one_center(b, r, specs)
    tab = kernels.one_center_table(b, r, specs)
    scale = r[..., None] ** -torch.tensor([float(p) for _, p in specs], dtype=F64)
    assert float(((tab - exact) / scale).abs().max()) < 1e-10


@pytest.mark.parametrize("point_x", [None, 40.0])
def test_table_mode_matches_exact_energy_and_forces(point_x):
    exact = md_model()
    fast = md_model()
    fast.kernel_mode, fast.point_x = "table", point_x
    for far in (6.0, 80.0):
        z, mon, q, x = three_monomers(far)
        e_ref, f_ref = evaluate(FF2Energy(exact, z, mon, q), x)
        e, f = evaluate(FF2Energy(fast, z, mon, q), x)
        assert abs(float(e - e_ref)) <= 1e-8 * (1 + abs(float(e_ref))), far
        assert float((f - f_ref).abs().max()) <= 1e-7 * (1 + float(f_ref.abs().max())), far


@pytest.mark.parametrize("specs", [kernels.MM, kernels.CHARGE_SHELL, kernels.DIPOLE_T,
                                   kernels.SHELL_FIELD, kernels.BAR1])
def test_2d_table_two_center_matches_quadrature(specs):
    """Exponent pairs in either order, ratios down to kernels.RHO_TAB, and
    separations through the x = 40 point-limit switch."""
    g = torch.Generator().manual_seed(4)
    bi = 1.0 + 14.0 * torch.rand(600, 1, generator=g, dtype=F64)
    bj = bi * torch.exp(torch.empty(600, 1, dtype=F64).uniform_(
        math.log(kernels.RHO_TAB), -math.log(kernels.RHO_TAB), generator=g))
    r = 0.1 + 40.0 * torch.rand(600, 1, generator=g, dtype=F64) ** 2
    exact = kernels.two_center(bi, bj, r, specs)
    tab = kernels.two_center_table2d(bi, bj, r, specs)
    scale = r[..., None] ** -torch.tensor([float(p) for _, p in specs], dtype=F64)
    assert float(((tab - exact) / scale).abs().max()) < 1e-8


def test_2d_table_gradients_match_quadrature():
    g = torch.Generator().manual_seed(6)
    args = [(1.0 + 6.0 * torch.rand(300, 1, generator=g, dtype=F64)).requires_grad_(True),
            (1.0 + 14.0 * torch.rand(300, 1, generator=g, dtype=F64)).requires_grad_(True),
            (2.0 + 20.0 * torch.rand(300, 1, generator=g, dtype=F64)).requires_grad_(True)]
    w = torch.randn(300, 1, len(kernels.MM), generator=g, dtype=F64)
    ga = torch.autograd.grad((kernels.two_center(*args, kernels.MM) * w).sum(), args)
    gb = torch.autograd.grad((kernels.two_center_table2d(*args, kernels.MM) * w).sum(), args)
    for a, b in zip(ga, gb):
        assert float((a - b).abs().max()) < 1e-7 * (1 + float(a.abs().max()))


def test_2d_tables_build_under_no_grad():
    kernels._TABLES.clear()
    with torch.no_grad():
        out = kernels.two_center_table2d(torch.tensor([[1.2]], dtype=F64),
                                         torch.tensor([[3.0]], dtype=F64),
                                         torch.tensor([[3.0]], dtype=F64), kernels.BAR1)
    assert bool(torch.isfinite(out).all())


@pytest.mark.parametrize("point_x", [None, 40.0])
def test_table2d_mode_matches_exact_energy_and_forces(point_x):
    exact = md_model()
    fast = md_model()
    fast.kernel_mode, fast.point_x = "table2d", point_x
    for far in (6.0, 80.0):
        z, mon, q, x = three_monomers(far)
        e_ref, f_ref = evaluate(FF2Energy(exact, z, mon, q), x)
        e, f = evaluate(FF2Energy(fast, z, mon, q), x)
        assert abs(float(e - e_ref)) <= 1e-7 * (1 + abs(float(e_ref))), far
        assert float((f - f_ref).abs().max()) <= 1e-6 * (1 + float(f_ref.abs().max())), far


@pytest.mark.parametrize("mode", ["table", "table2d"])
def test_trace_holds_tables_as_constants(mode):
    """A trace started with empty table caches must not record the table
    build: OpenMM would rerun it every step."""
    model = md_model()
    model.kernel_mode = mode
    z, mon, q, x = three_monomers(6.0)
    kernels._TABLES.clear()
    traced = trace_energy(model, z, mon, q, x)
    assert "linspace" not in str(traced.inlined_graph)
    e_ref, f_ref = evaluate(FF2Energy(model, z, mon, q), x)
    e, f = evaluate(traced, x)
    assert abs(float(e - e_ref)) <= 1e-10 * (1 + abs(float(e_ref)))
    assert float((f - f_ref).abs().max()) <= 1e-9 * (1 + float(f_ref.abs().max()))


def test_unknown_kernel_mode_is_rejected():
    model = md_model()
    model.kernel_mode = "fast"
    z, mon, q, x = three_monomers()
    with pytest.raises(ValueError, match="kernel mode"):
        FF2Energy(model, z, mon, q)(x)


# --- speed paths: Inductor-compiled energy -----------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("mode, point_x", [("table", None), ("table", 40.0),
                                           ("table2d", 40.0)])
def test_compiled_energy_matches_eager(mode, point_x):
    model = md_model()
    model.kernel_mode, model.point_x = mode, point_x
    z, mon, q, x = three_monomers(80.0)
    compiled = compile_energy(model, z, mon, q, x)
    for far in (80.0, 6.0):  # 6.0 changes the near/far partition
        x = three_monomers(far)[3]
        e_ref, f_ref = evaluate(FF2Energy(model, z, mon, q), x)
        e, f = compiled.energy_forces(x)
        assert abs(float(e - e_ref)) <= 1e-10 * (1 + abs(float(e_ref))), far
        assert float((f - f_ref).abs().max()) <= 1e-9 * (1 + float(f_ref.abs().max())), far


def test_tables_build_under_no_grad():
    kernels._TABLES.clear()
    with torch.no_grad():
        out = kernels.one_center_table(torch.tensor([[1.2]], dtype=F64),
                                       torch.tensor([[3.0]], dtype=F64), kernels.BAR1)
    assert bool(torch.isfinite(out).all())


# --- speed paths: iterative polarization -------------------------------------


def test_cg_solve_matches_cholesky_on_an_spd_batch():
    g = torch.Generator().manual_seed(5)
    M = torch.randn(2, 30, 30, generator=g, dtype=F64)
    A = torch.eye(30, dtype=F64) + 0.02 * M @ M.transpose(1, 2)
    rhs = torch.randn(2, 30, 1, generator=g, dtype=F64)
    exact = torch.cholesky_solve(rhs, torch.linalg.cholesky(A))
    x = physics.cg_solve(A, rhs, 30)
    assert float((x - exact).norm() / exact.norm()) < 1e-12


def test_cg_polarization_matches_cholesky_energy_and_forces():
    exact, cg = md_model(), md_model()
    cg.pol_cg_iters = 20
    for far in (6.0, 12.0):
        z, mon, q, x = three_monomers(far)
        e_ref, f_ref = evaluate(FF2Energy(exact, z, mon, q), x)
        e, f = evaluate(FF2Energy(cg, z, mon, q), x)
        assert abs(float(e - e_ref)) <= 1e-9 * (1 + abs(float(e_ref))), far
        assert float((f - f_ref).abs().max()) <= 1e-8 * (1 + float(f_ref.abs().max())), far
