"""FF-3 backbone arms on FF-2 (apnet_pt.ff2.trunk): the SO3LR total-charge
embedding (E), the So3krates attention gate + Euclidean variables (A, A-ev)
and the near-identity capacity layer (CAP).

Set ``QCML_MBIS_M3`` to the MBIS M3 checkpoint (``best-esp.pt`` of the
ff2-lite lock) to run the identity checks on its real weights as well."""

import os

import pytest
import torch
from torch._dynamo.utils import counters

from apnet_pt.ff1.trunk import FF1MonomerTrunk
from apnet_pt.ff2 import FF2Model, compile_ff2
from apnet_pt.ff2.batching import batch_from_dimers, with_coordinates
from apnet_pt.ff2.trunk import FF3MonomerTrunk, degree_contract, real_sph

from .test_ff2 import (
    ACETATE,
    NH3,
    WATER,
    dimers,
    grads,
    make_ff1,
    random_orthogonal,
    shift,
)

F64 = torch.float64
ARMS = {"E": {"charge_embedding": True},
        "A": {"attention": True},
        "A-ev": {"attention": True, "euclidean_variables": True},
        "CAP": {"n_identity_layers": 1}}
EXACT_ARMS = ("E", "A", "A-ev")
MBIS_M3 = os.environ.get("QCML_MBIS_M3")


def arm_options(arm, n_layers):
    """The arm's FF2Model options on an ``n_layers`` parent (CAP adds one)."""
    return {**ARMS[arm], "n_layers": n_layers + (arm == "CAP")}


def edge_case_dimers():
    """A sodium cation, a chloride anion and a monomer with two coincident
    atoms, plus the charged acetate."""
    return [
        {"zA": [11], "RA": [[0.0, 0.0, 0.0]], "qA": 1, "zB": [8, 1, 1],
         "RB": shift(WATER, 0.2, 0.1, 2.4)},
        {"zA": [17], "RA": [[0.0, 0.0, 0.0]], "qA": -1, "zB": ACETATE[0],
         "RB": shift(ACETATE[1], 0.0, 0.0, 3.4), "qB": -1},
        {"zA": [8, 1, 1, 1], "RA": WATER + [WATER[1]], "qA": 1, "zB": [7, 1, 1, 1],
         "RB": shift(NH3, 0.3, 0.4, 3.0)},
    ]


def perturbed(model, seed=0, scale=0.2):
    """Every parameter moved off its init, so the gates and the identity
    layers are live."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def trunk_out(model, b):
    z = b["z"].long()
    extra = ((b["monomer"], b["monomer_charge"]) if model.charge_embedding else ())
    return model.trunk(z, b["R"], b["edges"], *extra)[:3]


# --- defaults ------------------------------------------------------------------

def test_default_config_is_ff2():
    torch.manual_seed(0)
    model = FF2Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8)
    assert type(model.trunk) is FF1MonomerTrunk
    assert model.backbone() == {"charge_embedding": False, "attention": False,
                                "euclidean_variables": False, "n_identity_layers": 0}
    assert model.identity_init_keys() == []
    assert list(model.state_dict()) == list(make_ff1().state_dict())


@pytest.mark.parametrize("arm", list(ARMS))
def test_arm_keeps_the_default_draw_of_every_old_weight(arm):
    """Same seed: every key of the default model with the same depth carries
    the same tensor (bar CAP's identity layer), so arms differ only by the
    identity-init keys."""
    opts = arm_options(arm, 2)
    torch.manual_seed(7)
    base = FF2Model(n_feat=32, n_chan=4, n_layers=opts["n_layers"], n_rbf=8)
    torch.manual_seed(7)
    model = FF2Model(n_feat=32, n_chan=4, n_rbf=8, **opts)
    new = set(model.identity_init_keys())
    old = base.state_dict()
    for key, value in model.state_dict().items():
        if key in new:
            continue
        assert torch.equal(value, old[key]), key
    assert set(old) | new == set(model.state_dict())


# --- O(3) equivariance ---------------------------------------------------------

@pytest.mark.parametrize("arm", [*ARMS, "all"])
def test_trunk_is_o3_equivariant_with_ions_and_coincident_atoms(arm):
    opts = ({**ARMS["E"], **ARMS["A-ev"], "n_identity_layers": 1, "n_layers": 3}
            if arm == "all" else arm_options(arm, 2))
    torch.manual_seed(1)
    model = perturbed(FF2Model(n_feat=32, n_chan=4, n_rbf=8, trunk_dtype=F64,
                               **opts))
    b = batch_from_dimers(edge_case_dimers() + dimers()[:2])
    O = random_orthogonal(-1.0, seed=3)
    t = torch.tensor([0.7, -0.4, 1.9], dtype=F64)
    h0, u0, Q0 = trunk_out(model, b)
    h1, u1, Q1 = trunk_out(model, with_coordinates(b, b["R"] @ O.T + t))
    u_rot = torch.einsum("ab,nbc->nac", O, u0)
    Q_rot = torch.einsum("ab,nbdc->nadc", O, Q0)
    Q_rot = torch.einsum("nadc,ed->naec", Q_rot, O)
    err = {"h": float((h1 - h0).abs().max()), "u": float((u1 - u_rot).abs().max()),
           "Q": float((Q1 - Q_rot).abs().max())}
    h0, u0, Q0 = h0.detach(), u0.detach(), Q0.detach()
    print(f"equivariance {arm}: {err}")
    assert max(err.values()) < 1e-10, err
    assert float(u0.abs().max()) > 1e-3 and float(Q0.abs().max()) > 1e-3
    p0, _ = model(b)
    p1, _ = model(with_coordinates(b, b["R"] @ O.T + t))
    assert torch.isfinite(p0).all()
    assert torch.allclose(p0, p1, rtol=1e-10, atol=1e-9)


def test_spherical_harmonics_are_orthonormal_and_homogeneous():
    O = random_orthogonal(-1.0, seed=5)
    r = torch.randn(64, 3, dtype=F64)
    assert torch.allclose(degree_contract(real_sph(r @ O.T)), degree_contract(real_sph(r)),
                          atol=1e-12)
    assert float(real_sph(torch.zeros(1, 3, dtype=F64)).abs().max()) == 0.0
    # |Y_l|^2 summed over m is (2l + 1) / (4 pi) on the unit sphere
    u = r / r.norm(dim=1, keepdim=True)
    k = degree_contract(real_sph(u))
    want = torch.tensor([(2 * l + 1) ** 0.5 / (4 * torch.pi) for l in (1, 2, 3, 4)],
                        dtype=F64)
    assert torch.allclose(k, want.expand_as(k), atol=1e-12)


# --- identity at init ----------------------------------------------------------

def max_diff(base, model, b):
    p0, x0 = base(b)
    p1, x1 = model(b)
    out = {"pred": float((p1 - p0).abs().max())}
    for k in ("q", "mu", "th", "alpha_iso"):
        out[k] = float((x1[k] - x0[k]).abs().max())
    return out, p0


@pytest.mark.parametrize("arm", EXACT_ARMS)
def test_exact_arms_reproduce_ff2_at_init(arm):
    ff1 = make_ff1(11).double()
    base = FF2Model.from_ff1(ff1)
    model = FF2Model.from_ff1(ff1, **ARMS[arm])
    b = batch_from_dimers(dimers() + edge_case_dimers())
    diff, _ = max_diff(base, model, b)
    assert max(diff.values()) <= 1e-10, diff


def test_cap_is_near_identity_at_init():
    ff1 = make_ff1(11).double()
    base = FF2Model.from_ff1(ff1)
    model = FF2Model.from_ff1(ff1, **arm_options("CAP", 2))
    b = batch_from_dimers(dimers() + edge_case_dimers())
    diff, p0 = max_diff(base, model, b)
    print(f"CAP step-0 diff (random FF-1 weights): {diff}, |pred| max "
          f"{float(p0.abs().max()):.3g}")
    assert 0.0 < diff["pred"] < 0.1 * float(p0.abs().max())


def mbis_model(**options):
    ckpt = torch.load(MBIS_M3, map_location="cpu", weights_only=False)
    cfg = {k: ckpt["model_config"][k] for k in ("n_feat", "n_chan", "n_layers",
                                                "n_rbf", "r_cut", "rho_tr", "g0_init")}
    cfg["n_layers"] += options.pop("extra_layers", 0)
    model = FF2Model(**cfg, trunk_dtype=F64, physics_dtype=F64,
                     sr_shell=False, transverse_invariants=False,
                     alpha_anisotropy=False, **options)
    state = {k: v.to(F64) if v.is_floating_point() else v
             for k, v in ckpt["model"].items()}
    missing = model.load_ff1_state_dict(state)
    return model.double(), missing


@pytest.mark.skipif(not MBIS_M3, reason="QCML_MBIS_M3 not set")
def test_mbis_m3_identity_at_init():
    """The real MBIS M3 weights (FF-2-lite switches, float64 throughout):
    E and A-ev reproduce FF-2-lite to 1e-10; CAP's difference is reported."""
    base, _ = mbis_model()
    b = batch_from_dimers(dimers() + edge_case_dimers())
    for arm in ("E", "A-ev", "CAP"):
        opts = dict(ARMS[arm])
        if arm == "CAP":
            opts = {"extra_layers": 1, "n_identity_layers": 1}
        model, missing = mbis_model(**opts)
        assert set(missing) - set(model.identity_init_keys()) <= {
            k for k in missing if k.startswith(("beta_def", "sr_"))}
        diff, _ = max_diff(base, model, b)
        h0, u0, Q0 = trunk_out(base, b)
        h1, u1, Q1 = trunk_out(model, b)
        rel = {"u": float((u1 - u0).norm() / u0.norm()),
               "Q": float((Q1 - Q0).norm() / Q0.norm()),
               "h": float((h1 - h0).norm() / h0.norm())}
        print(f"MBIS M3 step-0 {arm}: max|d| {diff}; trunk rel {rel}")
        if arm != "CAP":
            assert max(diff.values()) <= 1e-10, (arm, diff)


# --- gradients at step 0 -------------------------------------------------------

GATE_PARAMS = {
    "E": ["trunk.charge_embed.res2.weight"],
    "A": [f"trunk.layers.{t}.filt2.weight" for t in range(2)],
    "A-ev": [f"trunk.layers.{t}.{m}.weight" for t in range(2)
             for m in ("filt2", "w_ev")],
    "CAP": [f"trunk.layers.2.{m}" for m in ("phi.weight", "psi_h.2.weight",
                                             "w_uq.weight", "w_qlin.weight",
                                             "w_q1.weight")],
}


@pytest.mark.parametrize("arm", list(ARMS))
def test_new_parameters_get_gradients_at_step_0(arm):
    ff1 = make_ff1(13).double()
    model = FF2Model.from_ff1(ff1, **arm_options(arm, 2))
    pred, _ = model(batch_from_dimers(dimers() + edge_case_dimers()))
    g = grads(model, (pred * torch.arange(1, 5, dtype=F64)).sum())
    for name in GATE_PARAMS[arm]:
        assert g[name] is not None and float(g[name].abs().max()) > 1e-8, name


# --- torch.compile -------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("arm", list(ARMS))
def test_arm_compiles_to_one_graph(arm):
    """Inductor, fullgraph, dynamic: three batch shapes, 0 graph breaks,
    0 recompiles, compiled == eager."""
    torch._dynamo.reset()
    counters.clear()
    model = perturbed(FF2Model.from_ff1(make_ff1(17), **arm_options(arm, 2)),
                      seed=2, scale=0.05)
    compiled = compile_ff2(model)
    for batch in (batch_from_dimers(dimers() + edge_case_dimers()[:1]),
                  batch_from_dimers(dimers()[:3]),
                  batch_from_dimers(dimers()[1:2])):
        pe, _ = model(batch)
        pc, _ = compiled(batch)
        assert torch.allclose(pe, pc, rtol=0, atol=1e-6), (pe - pc).abs().max()
        gc = grads(model, pc.sum())
        assert all(v is None or torch.isfinite(v).all() for v in gc.values())
    assert counters["stats"]["unique_graphs"] == 1
    assert not counters["graph_break"], dict(counters["graph_break"])
    assert sum(counters["recompiles"].values()) == 0


def test_identity_layer_count_is_checked():
    with pytest.raises(ValueError):
        FF2Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8, n_identity_layers=3)


def test_ff3_trunk_without_options_is_ff1_trunk():
    torch.manual_seed(3)
    a = FF1MonomerTrunk(32, 4, 2, 8).double()
    b = FF3MonomerTrunk(32, 4, 2, 8).double()
    b.load_state_dict(a.state_dict())
    bt = batch_from_dimers(dimers())
    oa = a(bt["z"], bt["R"], bt["edges"])
    ob = b(bt["z"], bt["R"], bt["edges"])
    for x, y in zip(oa[:3], ob[:3]):
        assert torch.equal(x, y)
