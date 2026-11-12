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
