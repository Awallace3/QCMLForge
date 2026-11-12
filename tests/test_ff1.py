"""FF-1 (l <= 2) force field: symmetry, physics and conservation tests."""

import math

import pytest
import torch

from apnet_pt.ff1 import FF1Model, ff1_loss
from apnet_pt.ff1 import gordon1, physics
from apnet_pt.ff1.batching import batch_from_dimers, with_coordinates
from apnet_pt.mastiff_classical import undamped_multipole_elst

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
F64 = torch.float64

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


def shift(coords, dx=0.0, dy=0.0, dz=0.0):
    return [[x + dx, y + dy, z + dz] for x, y, z in coords]


def water_ammonia():
    return {"zA": [8, 1, 1], "RA": WATER, "zB": [7, 1, 1, 1],
            "RB": shift(NH3, 0.3, 0.4, 3.0)}


def make_model(seed=0, device="cpu", scale=0.15):
    torch.manual_seed(seed)
    model = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8,
                     trunk_dtype=F64, physics_dtype=F64).to(device)
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


def energy_and_forces(model, batch, column=None):
    R = batch["R"].clone().requires_grad_(True)
    pred, extra = model(with_coordinates(batch, R))
    e = pred.sum() if column is None else pred[:, column].sum()
    (grad,) = torch.autograd.grad(e, R)
    return pred.detach(), -grad, extra


# --- Gordon1 kernels -------------------------------------------------------

def test_gordon1_branch_free_continuity_and_r0():
    r = torch.tensor([0.0, 1e-6, 0.3, 1.3, 7.0], dtype=F64)
    specs = [k for k in gordon1.ELST_AA_SPECS if k[0] != "bar1"]
    b = torch.full_like(r, 2.1)
    ref = gordon1.equal_center(b, r, specs)
    for rel in (-1e-9, 0.0, 1e-9):
        got = gordon1.two_center(b, b * (1 + rel), r, specs)
        for key in specs:
            assert torch.allclose(got[key], ref[key], rtol=1e-8, atol=1e-14), key
    bi, bj = torch.tensor([1.7], dtype=F64), torch.tensor([2.3], dtype=F64)
    zero = torch.zeros(1, dtype=F64)
    lam = gordon1.two_center(bi, bj, zero, ((5, 5), (7, 5), (9, 5)))
    expect = bi**4 * bj**4 / (135.0 * (bi + bj) ** 3)
    assert torch.allclose(2.0 / 3.0 * lam[(5, 5)], expect, rtol=1e-12)
    assert float(lam[(7, 5)]) == 0.0 and float(lam[(9, 5)]) == 0.0


def test_gordon1_qq_energy_finite_at_contact():
    th = torch.tensor([[[0.4, 0.1, 0.0], [0.1, -0.1, 0.2], [0.0, 0.2, -0.3]]],
                      dtype=F64)
    side = {"z": torch.zeros(1, dtype=F64), "nc": torch.zeros(1, dtype=F64),
            "bc": torch.ones(1, dtype=F64), "n": torch.zeros(1, dtype=F64),
            "q": torch.zeros(1, dtype=F64), "mu": torch.zeros(1, 3, dtype=F64)}
    i = {**side, "b": torch.tensor([1.7], dtype=F64), "th": th}
    j = {**side, "b": torch.tensor([2.3], dtype=F64), "th": th}
    vals = []
    for r in (0.0, 1e-4, 1e-3):
        rv = torch.tensor([[0.3, -0.2, 0.9]], dtype=F64) * r
        rr, rh = gordon1.safe_unit(rv)
        rh = torch.where(rr[:, None] > 0, rh, torch.tensor([[0.0, 0.0, 1.0]],
                                                           dtype=F64))
        vals.append(float(gordon1.elst_pairs(rr, rh, i, j)))
    expect = 1.7**4 * 2.3**4 / (135 * 4.0**3) * float((th * th).sum())
    assert vals[0] == pytest.approx(expect, rel=1e-12)
    assert all(math.isfinite(v) for v in vals)
    assert vals[1] == pytest.approx(expect, rel=1e-3)


def test_long_range_matches_point_multipoles():
    """Far apart, Gordon1 reduces to apnet_pt's undamped Buckingham terms."""
    model = make_model(3)
    batch = batch_from_dimers([{"zA": [8, 1, 1], "RA": WATER, "zB": [7, 1, 1, 1],
                                "RB": shift(NH3, 0, 0, 40.0)}])
    pred, extra = model(batch)
    z, R = batch["z"].to(F64), batch["R"]
    ia, ib = batch["ai"], batch["aj"]
    ref = undamped_multipole_elst(z, R, extra["q"], extra["mu"], extra["th"],
                                  z, R, extra["q"], extra["mu"], extra["th"],
                                  ia, ib, include_uq_qq=True)
    assert float(pred[0, 0]) == pytest.approx(float(ref.sum()), rel=1e-6, abs=1e-9)


# --- symmetry ----------------------------------------------------------------

@pytest.mark.parametrize("det", [1.0, -1.0])
@pytest.mark.parametrize("device", DEVICES)
def test_o3_invariance_and_equivariance(det, device):
    model = make_model(1, device)
    batch = batch_from_dimers([water_ammonia()], device=device)
    O = random_orthogonal(det).to(device)
    e0, f0, x0 = energy_and_forces(model, batch)
    t = torch.tensor([0.3, -1.1, 2.0], dtype=F64, device=device)
    e1, f1, x1 = energy_and_forces(model, with_coordinates(batch, batch["R"] @ O.T + t))
    assert torch.allclose(e0, e1, atol=1e-9, rtol=1e-10)
    assert torch.allclose(f0 @ O.T, f1, atol=1e-8)
    assert torch.allclose(x0["mu"] @ O.T, x1["mu"], atol=1e-10)
    assert torch.allclose(O @ x0["th"] @ O.T, x1["th"], atol=1e-10)
    assert torch.allclose(x0["induced_dipole"] @ O.T, x1["induced_dipole"], atol=1e-9)
    out0 = model._forward_atoms(batch)[1]
    out1 = model._forward_atoms(with_coordinates(batch, batch["R"] @ O.T + t))[1]
    assert torch.allclose(O @ out0["sqrt_alpha"] @ O.T, out1["sqrt_alpha"], atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_monomer_swap_symmetry(device):
    model = make_model(2, device)
    d = water_ammonia()
    swapped = {"zA": d["zB"], "RA": d["RB"], "zB": d["zA"], "RB": d["RA"]}
    hetero = {"zA": ACETATE[0], "RA": ACETATE[1], "qA": -1, "zB": [8, 1, 1],
              "RB": shift(WATER, 0.2, 0.1, 3.4)}
    hetero_s = {"zA": [8, 1, 1], "RA": hetero["RB"], "zB": ACETATE[0],
                "RB": ACETATE[1], "qB": -1}
    p0, _ = model(batch_from_dimers([d, hetero], device=device))
    p1, _ = model(batch_from_dimers([swapped, hetero_s], device=device))
    assert torch.allclose(p0, p1, atol=1e-9, rtol=1e-10)


def test_exchange_coaxial_dihedral_sensitivity():
    """Rotating j's l <= 2 tensors about the shared axis moves exchange with
    both cos(phi) and cos(2 phi) components (single-axis models are flat)."""
    rhat = torch.tensor([[0.0, 0.0, 1.0]], dtype=F64)
    u = torch.tensor([[0.2, 0.05, 0.1]], dtype=F64)
    Q = torch.tensor([[[0.2, 0.05, 0.0], [0.05, -0.15, 0.03], [0.0, 0.03, -0.05]]],
                     dtype=F64)
    phis = torch.linspace(0, 2 * math.pi, 33, dtype=F64)[:-1]
    kappa = torch.tensor([0.8, -0.6, 0.7, 0.9], dtype=F64)
    energies, flat = [], []
    one = torch.ones(1, dtype=F64)
    r = torch.tensor([6.0], dtype=F64)
    for phi in phis:
        c, s = math.cos(phi), math.sin(phi)
        rot = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=F64)
        uj, Qj = u @ rot.T, rot @ Q @ rot.T
        omega = physics.transverse_invariants(rhat, u, Q, uj, Qj)
        args = (r, rhat, one, one, 2 * one, 2 * one, 0 * one, 0 * one, omega)
        energies.append(float(physics.exchange_pairs(*args, kappa)))
        flat.append(float(physics.exchange_pairs(*args, 0 * kappa)))
    e = torch.tensor(energies)
    e = e - e.mean()
    c1 = float((e * torch.cos(phis.float())).mean())
    c2 = float((e * torch.cos(2 * phis.float())).mean())
    assert abs(c1) > 1e-3 * float(torch.tensor(energies).abs().mean())
    assert abs(c2) > 1e-3 * float(torch.tensor(energies).abs().mean())
    assert max(flat) - min(flat) < 1e-15


# --- charges -----------------------------------------------------------------

def test_zero_spectator_leakage_and_conservation():
    model = make_model(4, scale=0.3)
    za, ra = ACETATE
    for n_water in (0, 4, 10):
        z, R = list(za), [list(x) for x in ra]
        for k in range(n_water):
            ang = 2 * math.pi * k / max(n_water, 1)
            z += [8, 1, 1]
            R += shift(WATER, 4.0 * math.cos(ang), 4.0 * math.sin(ang), 1.5 * (k % 3))
        batch = batch_from_dimers([{"zA": z, "RA": R, "qA": -1, "zB": [8, 1, 1],
                                    "RB": shift(WATER, 0, 0, 9.0)}])
        _, extra = model(batch)
        q = extra["q"]
        assert float(q[:7].sum()) == pytest.approx(-1.0, abs=1e-12)
        for k in range(n_water):
            lo = 7 + 3 * k
            assert abs(float(q[lo:lo + 3].sum())) < 1e-12
        assert abs(float(q[len(z):].sum())) < 1e-12


def test_zwitterion_has_dipole_at_zero_fragment_charge():
    model = make_model(5, scale=0.3)
    z, R = GLYCINE_ZW
    batch = batch_from_dimers([{"zA": z, "RA": R, "zB": [8, 1, 1],
                                "RB": shift(WATER, 0, 0, 6.0)}])
    _, extra = model(batch)
    q = extra["q"][:len(z)]
    assert abs(float(q.sum())) < 1e-12
    dipole = (q[:, None] * batch["R"][:len(z)]).sum(0).norm()
    assert float(dipole) > 1e-2
    assert float(q.abs().max()) > 1e-2


def test_fragment_charge_above_active_charge_raises():
    model = make_model(0)
    batch = batch_from_dimers([{"zA": [1], "RA": [[0, 0, 0]], "qA": 2,
                                "zB": [8, 1, 1], "RB": shift(WATER, 0, 0, 3.0)}])
    with pytest.raises(ValueError):
        model(batch)


# --- bare proton -------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_bare_proton_safety(device):
    model = make_model(6, device)
    batch = batch_from_dimers([{"zA": [8, 1, 1], "RA": WATER, "zB": [1],
                                "RB": [[0.0, 0.0, -1.6]], "qB": 1}], device=device)
    R = batch["R"].clone().requires_grad_(True)
    pred, extra = model(with_coordinates(batch, R))
    assert torch.isfinite(pred).all()
    assert float(extra["q"][3]) == pytest.approx(1.0, abs=1e-14)
    assert float(extra["alpha_iso"][3]) == 0.0
    assert float(extra["c6_pair"].abs().max()) == 0.0
    assert float(extra["ct_pair"].abs().sum()) > 0.0
    for c in range(4):
        grads = torch.autograd.grad(pred[0, c], [R, *model.parameters()],
                                    retain_graph=True, allow_unused=True)
        for g in grads:
            assert g is None or torch.isfinite(g).all()


# --- polarization stability and short-range wall -----------------------------

def test_polarization_matrix_spd_on_compressed_cluster():
    model = make_model(7, scale=0.5)
    g = torch.Generator().manual_seed(3)
    pts = torch.stack(torch.meshgrid(*[torch.arange(3, dtype=F64)] * 3,
                                     indexing="ij"), -1).reshape(-1, 3) * 0.4
    pts = pts + 0.05 * torch.randn(pts.shape, generator=g, dtype=F64)
    z = [6, 7, 8, 1, 6, 8, 1, 7, 6] * 3
    batch = batch_from_dimers([{"zA": z[:13], "RA": pts[:13].tolist(),
                                "zB": z[13:], "RB": pts[13:].tolist()}])
    _, extra = model(batch, return_matrix=True)
    lam = torch.linalg.eigvalsh(extra["pol_matrix"]).min()
    assert float(lam) >= 1 - physics.A_POL * math.exp(math.sqrt(2 / 3) * 0.35) - 1e-10


@pytest.mark.parametrize("pair", ["water", "methane-ne"])
def test_short_range_wall_is_repulsive(pair):
    model = make_model(8)
    if pair == "water":
        d = {"zA": [8, 1, 1], "RA": WATER, "zB": [8, 1, 1],
             "RB": shift(WATER, 0.0, 0.0, 0.3)}
    else:
        d = {"zA": [6, 1, 1, 1, 1],
             "RA": [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                    [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]],
             "zB": [10], "RB": [[0.0, 0.0, 0.3]]}
    pred, _ = model(batch_from_dimers([d]))
    assert float(pred[0, :3].sum()) > 0.0


# --- conservative forces -----------------------------------------------------

@pytest.mark.parametrize("mode", ["train", "eval"])
def test_forces_match_finite_differences(mode):
    model = make_model(9)
    model.train(mode == "train")
    batch = batch_from_dimers([water_ammonia()])
    R0 = batch["R"].clone()
    h = 1e-4
    for column in (0, 1, 2, 3, None):
        _, forces, _ = energy_and_forces(model, batch, column)

        def energy(R):
            with torch.no_grad():
                p, _ = model(with_coordinates(batch, R))
            return float(p.sum() if column is None else p[:, column].sum())

        fd = torch.zeros_like(R0)
        for a in range(R0.shape[0]):
            for x in range(3):
                vals = []
                for k in (-2, -1, 1, 2):
                    R = R0.clone()
                    R[a, x] += k * h
                    vals.append(energy(R))
                fd[a, x] = -(vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
        err = float((fd - forces).abs().max())
        assert err < 1e-7, (column, err)


def test_loss_terms_and_gradients():
    model = make_model(10)
    batch = batch_from_dimers([water_ammonia()])
    pred, extra = model(batch)
    target = torch.tensor([[-3.0, 2.0, -1.0, -1.5]], dtype=F64)
    loss, parts = ff1_loss(pred, target, nu=extra["nu"])
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["nu_l2"] > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


# --- float32 physics --------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_float32_kernel_gradients_are_finite(device):
    """The tail series' backward pass stays in range in float32 over
    contact-to-long-range distances and core-to-diffuse exponents (a Horner
    loop overflowed at beta R ~ 8)."""
    f32 = torch.float32
    b = torch.tensor([0.5, 1, 2, 4, 8, 15, 30], dtype=f32, device=device)
    b1, b2, r = torch.meshgrid(b, b, torch.linspace(0.3, 40, 300, dtype=f32,
                                                    device=device), indexing="ij")
    b1, b2, r = (t.reshape(-1).clone().requires_grad_(True) for t in (b1, b2, r))
    for key, val in gordon1.two_center(b1, b2, r, gordon1.ELST_AA_SPECS).items():
        weight = torch.rand_like(val) * 1e4  # large upstream gradients
        for g in torch.autograd.grad((weight * val).sum(), (b1, b2, r),
                                     retain_graph=True):
            assert torch.isfinite(g).all(), key


@pytest.mark.parametrize("device", DEVICES)
def test_float32_psi_is_finite_for_large_gamma(device):
    """psi(f, gamma) in float32: finite values and gradients up to gamma = 10
    (expm1 overflowed past gamma ~ 6, which the element prefit reached on the
    full stores), within 1e-5 of float64, and exactly 0 at f = 0."""
    from apnet_pt.ff1.model import psi

    x64 = torch.tensor([0.0, 1e-9, 1e-6, 1e-4, 0.01, 0.3, 1.0, 1.7, 3.0],
                       dtype=F64, device=device)
    g64 = torch.tensor([0.5, 1.0, 1.85, 3.0, 6.0, 6.5, 8.0, 10.0], dtype=F64,
                       device=device)
    x64, g64 = (t.reshape(-1) for t in torch.meshgrid(x64, g64, indexing="ij"))
    ref = psi(x64, g64)
    x32 = x64.float().requires_grad_(True)
    g32 = g64.float().requires_grad_(True)
    val = psi(x32, g32)
    assert torch.isfinite(val).all()
    assert (val[x64 == 0] == 0).all()
    assert torch.allclose(val.double(), ref, rtol=1e-5, atol=1e-30)
    for grad in torch.autograd.grad(val.sum(), (x32, g32)):
        assert torch.isfinite(grad).all()


def _f32_pair(seed, scale, device):
    m64 = make_model(seed, device, scale)
    m32 = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8, trunk_dtype=F64,
                   physics_dtype=torch.float32).to(device)
    m32.load_state_dict(m64.state_dict())
    return m64, m32


def _f32_dimers():
    out = [{"zA": [8, 1, 1], "RA": WATER, "zB": [7, 1, 1, 1],
            "RB": shift(NH3, 0.3, 0.4, dz)} for dz in (1.6, 2.5, 4.0, 10.0)]
    out += [{"zA": ACETATE[0], "RA": ACETATE[1], "qA": -1, "zB": GLYCINE_ZW[0],
             "RB": shift(GLYCINE_ZW[1], 0, 0, dz)} for dz in (2.2, 5.0)]
    out.append({"zA": [8, 1, 1], "RA": WATER, "zB": [11], "RB": [[0, 0, -2.3]],
                "qB": 1})
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed,scale", [(0, 0.15), (3, 0.5)])
def test_float32_physics_tracks_float64(device, seed, scale):
    m64, m32 = _f32_pair(seed, scale, device)
    batch = batch_from_dimers(_f32_dimers(), device=device)
    p64, f64, _ = energy_and_forces(m64, batch)
    p32, f32, x32 = energy_and_forces(m32, batch)
    assert p32.dtype == torch.float32 and torch.isfinite(f32).all()
    tol = 1e-4 * p64.abs().clamp_min(1.0)
    assert ((p32.double() - p64).abs() <= tol).all()
    assert (f32.double() - f64).abs().max() <= 1e-4 * f64.abs().max()
    # charges are solved in float64 in either mode: exactly conserved
    assert float(x32["q"][:3].sum()) == pytest.approx(0.0, abs=1e-12)
    pred, _ = m32(batch)
    grads = torch.autograd.grad(pred.sum(), list(m32.parameters()),
                                allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


# --- configurable bond-flow prior, deformation shells (ff1-v4) ---------------

from apnet_pt.ff1 import model as ff1_model  # noqa: E402


def _sites(r_dtype=F64, device="cpu", seed=0, sr=True):
    """Two realistic Eq. 7 sites (rot_m-like i, H-like j) with deformation shells."""
    g = torch.Generator().manual_seed(seed)

    def sym0(a):
        a = 0.5 * (a + a.T)
        return a - torch.trace(a) / 3 * torch.eye(3, dtype=a.dtype)

    def site(z, nc, bc, n, b, bdef):
        s = {"z": torch.tensor([z]), "nc": torch.tensor([nc]),
             "bc": torch.tensor([bc]), "n": torch.tensor([n]),
             "b": torch.tensor([b]), "q": torch.tensor([z - nc - n]),
             "mu": 0.3 * torch.randn(1, 3, generator=g, dtype=F64),
             "th": sym0(0.4 * torch.randn(3, 3, generator=g, dtype=F64))[None],
             "b_def": torch.tensor([bdef])}
        if sr:
            s["d_sr"] = 0.2 * torch.randn(1, 3, generator=g, dtype=F64)
            s["Q_sr"] = sym0(0.25 * torch.randn(3, 3, generator=g, dtype=F64))[None]
        return {k: v.to(dtype=r_dtype, device=device) for k, v in s.items()}

    return (site(8.0, 2.0, 13.0, 6.5, 2.2, 1.6), site(1.0, 0.0, 1.0, 0.6, 2.5, 3.1))


def _place(r, direction=(0.3, -0.5, 0.8), dtype=F64, device="cpu"):
    v = torch.tensor([direction], dtype=dtype, device=device)
    v = v / v.norm() * r
    return gordon1.safe_unit(v)


def test_default_model_has_no_new_parameters_and_same_shift():
    m = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8)
    keys = set(m.state_dict())
    assert not any(k.startswith(("beta_def", "sr_")) for k in keys)
    assert m.g_shift == ff1_model.G_SHIFT  # bit-identical default
    assert m.config() == {"g0_init": -6.0, "elst_beta_def": False,
                          "elst_sr_shell": False}
    torch.manual_seed(5)
    a = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8)
    torch.manual_seed(5)
    b = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8, elst_beta_def=True,
                 elst_sr_shell=True)
    sb = b.state_dict()
    for k, v in a.state_dict().items():  # same init stream for every old tensor
        assert torch.equal(v, sb[k]), k
    with pytest.raises(ValueError):
        FF1Model(elst_sr_shell=True)
    with pytest.raises(ValueError):
        FF1Model(g0_init=-8.5)


@pytest.mark.parametrize("g0", [-6.0, -2.5, 1.0])
def test_g0_init_is_the_logit_at_zero_readout(g0):
    s = ff1_model.g_shift(g0)
    raw = torch.zeros(3, dtype=F64)
    g = ff1_model.G_CENTER + ff1_model.G_HALF * torch.tanh(
        (raw - s) / ff1_model.G_HALF)
    assert torch.allclose(g, torch.full_like(g, g0), atol=1e-14)


def test_unbottlenecked_prior_moves_charge_at_init():
    batch = batch_from_dimers([water_ammonia()])
    q = {}
    for g0 in (-6.0, -2.5):
        torch.manual_seed(0)
        m = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8, trunk_dtype=F64,
                     g0_init=g0)
        q[g0] = m(batch)[1]["q"].abs().mean()
    assert float(q[-2.5]) > 5 * float(q[-6.0])


def test_forward_monomers_matches_dimer_atoms():
    model = make_model(11)
    batch = batch_from_dimers([water_ammonia()])
    mono = {k: batch[k] for k in ("z", "R", "edges", "monomer", "monomer_charge")}
    R_b, atoms = model.forward_monomers(mono)
    _, extra = model(batch)
    for k in ("q", "mu", "th", "beta", "alpha_iso", "nu", "n_act"):
        assert torch.equal(atoms[k], extra[k]), k
    assert torch.allclose(R_b, batch["R"] * ff1_model.ANG2BOHR)


def _with_extensions(seed=0, sr=False, device="cpu"):
    base = make_model(seed, device)
    ext = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8, trunk_dtype=F64,
                   physics_dtype=F64, elst_beta_def=True, elst_sr_shell=sr).to(device)
    missing, unexpected = ext.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith(("beta_def", "sr_")) for k in missing)
    return base, ext


@pytest.mark.parametrize("sr", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_deformation_shell_at_init_is_the_default_model(sr, device):
    base, ext = _with_extensions(12, sr, device)
    batch = batch_from_dimers(_f32_dimers(), device=device)
    p0, x0 = base(batch)
    p1, x1 = ext(batch)
    assert torch.equal(x1["beta_def"], x1["beta"])
    assert torch.allclose(p0, p1, atol=1e-11, rtol=1e-12)
    assert torch.allclose(x0["induced_dipole"], x1["induced_dipole"], atol=1e-12)


def _split(side, part):
    """One piece of a site for the bilinear decomposition, as a single-exponent
    site: 'charges' (z, nc/bc, n/b, q) or each multipole shell on its exponent."""
    zero = torch.zeros_like(side["z"])
    if part == "charges":
        return {**{k: side[k] for k in ("z", "nc", "bc", "n", "b", "q")},
                "mu": torch.zeros_like(side["mu"]), "th": torch.zeros_like(side["th"])}
    mu, th, b = gordon1.multipole_shells(side)[part]
    return {"z": zero, "nc": zero, "bc": side["bc"], "n": zero, "b": b, "q": zero,
            "mu": mu, "th": th}


@pytest.mark.parametrize("sr", [False, True])
@pytest.mark.parametrize("r", [0.0, 0.7, 2.5, 6.0])
def test_shell_energy_is_the_bilinear_sum_of_single_exponent_pieces(sr, r):
    """The two-exponent energy equals the old (Fourier-verified) single-exponent
    kernel applied to each (charge | multipole shell) x (charge | shell) piece."""
    i, j = _sites(sr=sr)
    rr, rh = _place(r)
    if r == 0.0:
        rh = torch.tensor([[0.0, 0.0, 1.0]], dtype=F64)
    got = gordon1.elst_pairs(rr, rh, i, j)
    parts = ["charges"] + list(range(len(gordon1.multipole_shells(i))))
    ref = sum(gordon1.elst_pairs(rr, rh, _split(i, a), _split(j, b))
              for a in parts for b in parts)
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-13)
    f_got = gordon1.permanent_field(rr, rh, torch.tensor([1.9], dtype=F64), j)
    f_ref = sum(gordon1.permanent_field(rr, rh, torch.tensor([1.9], dtype=F64),
                                        _split(j, b)) for b in parts)
    assert torch.allclose(f_got, f_ref, rtol=1e-12, atol=1e-13)


def _fourier_energy(i, j, R, n_k=400001, k_max=4000.0):
    """Exact Coulomb energy of two axial Eq. 7 two-exponent densities (z axis,
    j at +R z) by quadrature of E = int d^3k/(2pi)^3 4pi/k^2 rho_i(-k)
    rho_j(k) e^{-ik.R}.  The angular integral is analytic: the integrand is a
    degree-4 polynomial in c = cos(k, z), expanded in Legendre P_l with
    int P_l(c) e^{-ixc} dc = 2 (-i)^l j_l(x); nucleus-nucleus is z_i z_j / R."""
    import numpy as np
    from scipy.special import spherical_jn

    k = np.linspace(1e-9, k_max, n_k)

    def slater(b):
        return b**4 / (b * b + k * k) ** 2

    def poly(s, sign):  # rho(sign k) as monomial coefficients in c (c^0..c^2)
        fd = slater(s["b_def"])
        a = s["z"] - s["nc"] * slater(s["bc"]) - s["n"] * slater(s["b"]) + 0j
        c2 = -s["t"] * k * k * fd / 3.0  # times P2 = (3c^2 - 1)/2
        return [a - 0.5 * c2, -1j * sign * s["mu_z"] * k * fd, 1.5 * c2]

    pi_, pj = poly(i, -1), poly(j, 1)
    prod = [0j * k for _ in range(5)]
    for m, x in enumerate(pi_):
        for n, y in enumerate(pj):
            prod[m + n] = prod[m + n] + x * y
    prod[0] = prod[0] - i["z"] * j["z"]
    to_leg = np.zeros((5, 5))  # (monomial, legendre)
    for m in range(5):
        c = np.polynomial.legendre.poly2leg(np.eye(5)[m])
        to_leg[m, :len(c)] = c
    leg = np.einsum("mk,ml->lk", np.stack(prod), to_leg)
    ang = sum(leg[ell] * 2 * (-1j) ** ell * spherical_jn(ell, k * R) for ell in range(5))
    e = np.trapezoid(ang.real, k) / np.pi
    return e + i["z"] * j["z"] / R


@pytest.mark.parametrize("R", [1.2, 2.5, 4.0])
def test_two_exponent_density_matches_fourier_quadrature(R):
    def axial(t):
        return torch.tensor([[[-t / 2, 0, 0], [0, -t / 2, 0], [0, 0, t]]], dtype=F64)

    spec = [dict(z=8.0, nc=2.0, bc=13.0, n=6.5, b=2.2, b_def=1.6, mu_z=0.3, t=0.4),
            dict(z=1.0, nc=0.0, bc=1.0, n=0.6, b=2.5, b_def=3.1, mu_z=-0.2, t=0.15)]
    sides = []
    for s in spec:
        t = {k: torch.tensor([v], dtype=F64) for k, v in s.items()
             if k not in ("mu_z", "t")}
        t["q"] = t["z"] - t["nc"] - t["n"]
        t["mu"] = torch.tensor([[0.0, 0.0, s["mu_z"]]], dtype=F64)
        t["th"] = axial(s["t"])
        sides.append(t)
    rr = torch.tensor([R], dtype=F64)
    rh = torch.tensor([[0.0, 0.0, 1.0]], dtype=F64)
    got = float(gordon1.elst_pairs(rr, rh, *sides))
    ref = _fourier_energy(spec[0], spec[1], R)
    assert got == pytest.approx(ref, abs=2e-7, rel=1e-6)


@pytest.mark.parametrize("sr", [False, True])
def test_shells_keep_the_point_multipole_tail(sr):
    """At 25 bohr the two-exponent energy is the undamped Buckingham one."""
    from apnet_pt.mastiff_classical import undamped_multipole_elst

    i, j = _sites(sr=sr)
    rr, rh = _place(25.0)
    got = float(gordon1.elst_pairs(rr, rh, i, j)) * ff1_model.H2KCAL
    R = torch.cat([torch.zeros(1, 3, dtype=F64), rh * rr[:, None]]) / ff1_model.ANG2BOHR
    z = torch.tensor([8.0, 1.0], dtype=F64)
    q = torch.cat([i["q"], j["q"]])
    mu = torch.cat([i["mu"], j["mu"]])
    th = torch.cat([i["th"], j["th"]])
    ref = undamped_multipole_elst(z, R, q, mu, th, z, R, q, mu, th,
                                  torch.tensor([0]), torch.tensor([1]),
                                  include_uq_qq=True)
    assert got == pytest.approx(float(ref.sum()), rel=1e-6, abs=1e-9)


def test_sr_shell_gradcheck_energy_and_field():
    i, j = _sites(seed=3)
    v = torch.tensor([[0.4, -0.9, 1.7]], dtype=F64, requires_grad=True)

    def energy(vec):
        r, rh = gordon1.safe_unit(vec)
        return gordon1.elst_pairs(r, rh, i, j)

    assert torch.autograd.gradcheck(energy, (v,), eps=1e-6, atol=1e-8)

    def field(vec):
        r, rh = gordon1.safe_unit(vec)
        return gordon1.permanent_field(r, rh, torch.tensor([1.9], dtype=F64), j)

    assert torch.autograd.gradcheck(field, (v,), eps=1e-6, atol=1e-8)
    # the field is the energy gradient seen by a pure probe dipole on
    # Slater(beta_probe): E = -mu_p . F
    mu_p = torch.tensor([[0.3, -0.1, 0.25]], dtype=F64)
    zero = torch.zeros(1, dtype=F64)
    probe = {"z": zero, "nc": zero, "bc": torch.ones(1, dtype=F64), "n": zero,
             "b": torch.tensor([1.9], dtype=F64), "q": zero, "mu": mu_p,
             "th": torch.zeros(1, 3, 3, dtype=F64)}
    for r_ in (0.8, 2.0, 5.0):
        rr, rh = _place(r_)
        # rhat points from source j to the probe; elst_pairs' from i (probe) to j
        f = gordon1.permanent_field(rr, rh, torch.tensor([1.9], dtype=F64), j)
        e = gordon1.elst_pairs(rr, -rh, probe, j)
        assert torch.allclose(e, -(mu_p * f).sum(-1), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("det", [1.0, -1.0])
def test_sr_shell_is_o3_invariant(det):
    rot_m = random_orthogonal(det, seed=4)
    i, j = _sites(seed=4)
    rr, rh = _place(1.7)

    def rot(s):
        out = dict(s)
        for k in ("mu", "d_sr"):
            out[k] = s[k] @ rot_m.T
        for k in ("th", "Q_sr"):
            out[k] = rot_m @ s[k] @ rot_m.T
        return out

    e0 = gordon1.elst_pairs(rr, rh, i, j)
    e1 = gordon1.elst_pairs(rr, rh @ rot_m.T, rot(i), rot(j))
    assert torch.allclose(e0, e1, rtol=1e-13, atol=1e-14)
    bp = torch.tensor([1.9], dtype=F64)
    f0 = gordon1.permanent_field(rr, rh, bp, j)
    f1 = gordon1.permanent_field(rr, rh @ rot_m.T, bp, rot(j))
    assert torch.allclose(f0 @ rot_m.T, f1, rtol=1e-13, atol=1e-14)


def test_sr_shell_is_smooth_at_r0():
    i, j = _sites(seed=5)
    for r_ in (0.0, 1e-6, 1e-3):
        v = (torch.tensor([[0.3, -0.2, 0.9]], dtype=F64) * r_).requires_grad_(True)
        r, rh = gordon1.safe_unit(v)
        if r_ == 0.0:
            rh = torch.tensor([[0.0, 0.0, 1.0]], dtype=F64)
        e = gordon1.elst_pairs(r, rh, i, j)
        f = gordon1.permanent_field(r, rh, torch.tensor([1.9], dtype=F64), j)
        assert torch.isfinite(e).all() and torch.isfinite(f).all()
        if r_ > 0:
            (g,) = torch.autograd.grad(e.sum(), v)
            assert torch.isfinite(g).all()


@pytest.mark.parametrize("dtype,tail", [(F64, 1e-13), (torch.float32, 1e-7)])
@pytest.mark.parametrize("device", DEVICES)
def test_sr_shell_has_no_multipole_tail(dtype, tail, device):
    """The zero-moment shell moves contact electrostatics and decays
    exponentially: no 1/R^k leakage, in float64 and float32."""
    i, j = _sites(dtype, device, seed=6)
    off_i = {**i, "d_sr": torch.zeros_like(i["d_sr"]),
             "Q_sr": torch.zeros_like(i["Q_sr"])}
    off_j = {**j, "d_sr": torch.zeros_like(j["d_sr"]),
             "Q_sr": torch.zeros_like(j["Q_sr"])}
    bp = torch.tensor([1.9], dtype=dtype, device=device)

    def diff(r_):
        rr, rh = _place(r_, dtype=dtype, device=device)
        de = gordon1.elst_pairs(rr, rh, i, j) - gordon1.elst_pairs(rr, rh, off_i, off_j)
        df = (gordon1.permanent_field(rr, rh, bp, j)
              - gordon1.permanent_field(rr, rh, bp, off_j)).abs().max()
        return float(de.abs()), float(df)

    near, _ = diff(4.0)
    assert near > 1e-4
    e25, f25 = diff(25.0)
    assert e25 < tail and f25 < tail
    if dtype == F64:
        beta_min = 1.6
        e8, e16 = diff(8.0)[0], diff(16.0)[0]
        assert e16 < e8 * math.exp(-0.5 * beta_min * 8.0)


@pytest.mark.parametrize("device", DEVICES)
def test_extended_model_symmetry_forces_and_gradients(device):
    _, ext = _with_extensions(13, True, device)
    with torch.no_grad():
        for mod in (ext.beta_def_head[-1], ext.sr_vec_readout, ext.sr_ten_readout):
            for p in mod.parameters():
                p.add_(0.3 * torch.randn_like(p))
        ext.beta_def_guess.weight.add_(0.2 * torch.randn_like(ext.beta_def_guess.weight))
    batch = batch_from_dimers([water_ammonia()], device=device)
    _, x = ext(batch)
    assert not torch.equal(x["beta_def"], x["beta"])
    assert float(x["d_sr"].abs().max()) > 0
    tr = torch.diagonal(x["Q_sr"], dim1=1, dim2=2).sum(-1)
    assert float(tr.abs().max()) < 1e-12
    assert torch.allclose(x["Q_sr"], x["Q_sr"].transpose(1, 2), atol=1e-14)
    rot_m = random_orthogonal(-1.0).to(device)
    e0, f0, _ = energy_and_forces(ext, batch)
    e1, f1, _ = energy_and_forces(ext, with_coordinates(batch, batch["R"] @ rot_m.T))
    assert torch.allclose(e0, e1, atol=1e-9, rtol=1e-10)
    assert torch.allclose(f0 @ rot_m.T, f1, atol=1e-8)
    R0 = batch["R"].clone()
    h = 1e-4
    a, x_ = 4, 2
    vals = []
    for kk in (-1, 1):
        R = R0.clone()
        R[a, x_] += kk * h
        with torch.no_grad():
            vals.append(float(ext(with_coordinates(batch, R))[0].sum()))
    assert -(vals[1] - vals[0]) / (2 * h) == pytest.approx(float(f0[a, x_]), abs=1e-6)
    pred, _ = ext(batch)
    grads = torch.autograd.grad(pred.sum(), list(ext.parameters()), allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert any(p is ext.beta_def_guess.weight for p in ext.element_parameters())
