import math

import pytest
import torch

from apnet_pt.atomic_c6_dispersion import (
    BOHR,
    H2KCAL,
    AtomicC6Dispersion,
    dispersion_objective,
    free_atom_c6,
    prefit,
    sqrt_q,
    tang_toennies,
)

OPTIONS = {"n_message": 2, "n_rbf": 4, "n_neuron": 16, "n_embed": 4, "r_cut": 5.0}
WATER = ([8, 1, 1], [[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]])
METHANE = (
    [6, 1, 1, 1, 1],
    [[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
     [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]],
)


def dimer_inputs(monomers, shifts):
    """Batch inputs for dimers given as ((A, B), (shift_A, shift_B)) pairs."""
    z, R, mol, edges, pi, pj, pd = [], [], [], [], [], [], []
    n_atom = n_mol = 0
    for d, ((a, b), (sa, sb)) in enumerate(zip(monomers, shifts)):
        offsets = []
        for (zm, rm), shift in ((a, sa), (b, sb)):
            idx = torch.arange(n_atom, n_atom + len(zm))
            offsets.append(idx)
            z.append(torch.tensor(zm))
            R.append(torch.tensor(rm, dtype=torch.float64)
                     + torch.tensor(shift, dtype=torch.float64))
            mol.append(torch.full((len(zm),), n_mol))
            ii, jj = torch.meshgrid(idx, idx, indexing="ij")
            keep = ii != jj
            edges.append(torch.stack([ii[keep], jj[keep]]))
            n_atom += len(zm)
            n_mol += 1
        ii, jj = torch.meshgrid(offsets[0], offsets[1], indexing="ij")
        pi.append(ii.reshape(-1))
        pj.append(jj.reshape(-1))
        pd.append(torch.full((ii.numel(),), d))
    return {
        "z": torch.cat(z), "R": torch.cat(R), "edge_index": torch.cat(edges, 1),
        "molecule_ind": torch.cat(mol), "pair_i": torch.cat(pi),
        "pair_j": torch.cat(pj), "pair_dimer": torch.cat(pd),
        "n_dimer": len(monomers),
    }


def water_methane(separations=(3.5, 4.5, 6.0)):
    return dimer_inputs([(WATER, METHANE)] * len(separations),
                        [((0, 0, 0), (s, 0.0, 0.0)) for s in separations])


def perturb(model, scale=0.05, seed=0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.mpnn.parameters():
            if p.requires_grad:
                p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return model


ARMS = [
    {"damping": "tt", "beta": "atom"},
    {"damping": "tt", "beta": "r0", "slater_isa": True},
    {"damping": "bj", "beta": None, "orders": (6, 8)},
    {"damping": "tt", "beta": "atom", "higher_order": "delta"},
    {"damping": "tt", "beta": "atom", "higher_order": "element",
         "c6_anchor": "free-atom-ratio"},
]


@pytest.mark.parametrize("arm", ARMS)
def test_untrained_model_is_its_element_model(arm):
    torch.manual_seed(0)
    model = AtomicC6Dispersion(mpnn_options=OPTIONS, **arm)
    x = water_methane()
    energy, extra = model(**x)
    element, _ = model(**x, use_mpnn=False)
    for k in ("d6", "d8", "d10", "dbeta"):
        assert float(extra["atom"][k].abs().max()) == 0.0
    assert torch.equal(energy, element)
    assert energy.dtype == torch.float64 and energy.shape == (3,)


@pytest.mark.parametrize("damping", ["tt", "bj"])
def test_one_pair_matches_the_closed_form(damping):
    """Ne...Ar at 3.4 A, element model, against the formulas by hand."""
    model = AtomicC6Dispersion(
        damping=damping, beta="atom" if damping == "tt" else None,
        mpnn=damping != "bj", mpnn_options=OPTIONS)
    x = dimer_inputs([(([10], [[0.0, 0.0, 0.0]]), ([18], [[0.0, 0.0, 0.0]]))],
                     [((0, 0, 0), (3.4, 0.0, 0.0))])
    energy, _ = model(**x)
    free, sq = free_atom_c6(), sqrt_q()
    g = {k: float(v) for k, v in model.globals().items()}
    r = 3.4 / BOHR
    c6 = math.sqrt(float(free[10]) * float(free[18]))
    qq = 3.0 * float(sq[10]) * float(sq[18])
    coef = {6: c6, 8: c6 * qq, 10: 49.0 / 40.0 * c6 * qq * qq}
    scale = {6: 1.0, 8: g["s8"], 10: g["s10"]}
    if damping == "bj":
        r0 = g["a1"] * math.sqrt(qq) + g["a2"]
        ref = sum(scale[n] * coef[n] / (r**n + r0**n) for n in (6, 8, 10))
    else:
        beta = math.sqrt(g["b"] / (math.sqrt(3) * float(sq[10]))
                         * g["b"] / (math.sqrt(3) * float(sq[18])))
        damp = {n: float(tang_toennies(n, torch.tensor(beta * r, dtype=torch.float64)))
                for n in coef}
        ref = sum(scale[n] * damp[n] * coef[n] / r**n for n in (6, 8, 10))
    assert float(energy[0]) == pytest.approx(-H2KCAL * ref, rel=1e-12)


@pytest.mark.parametrize("arm", ARMS[:3])
def test_long_range_is_the_undamped_series(arm):
    """At 40 A the damping is 1 to well below 1e-8 for every form."""
    model = perturb(AtomicC6Dispersion(mpnn_options=OPTIONS, **arm))
    x = water_methane(separations=(40.0,))
    energy, extra = model(**x)
    t, g = extra["atom"], model.globals()
    i, j = x["pair_i"], x["pair_j"]
    r = (x["R"][j] - x["R"][i]).norm(dim=-1) / BOHR
    c6 = torch.exp(0.5 * (t["log_c6"][i] + t["log_c6"][j]))
    sq = model.sqrt_q[x["z"]]
    qq = 3.0 * sq[i] * sq[j]
    ref = c6 / r**6 + g["s8"] * c6 * qq / r**8
    if 10 in model.orders:
        ref = ref + g["s10"] * 49.0 / 40.0 * c6 * qq * qq / r**10
    assert float(energy[0]) == pytest.approx(float(-H2KCAL * ref.sum()), rel=1e-8)


@pytest.mark.parametrize("arm", ARMS)
def test_pairs_attract_and_energy_is_invariant_to_rigid_motion(arm):
    model = perturb(AtomicC6Dispersion(mpnn_options=OPTIONS, **arm))
    x = water_methane()
    energy, extra = model(**x)
    assert float(extra["pair"].max()) <= 0.0
    assert float(extra["atom"]["d6"].abs().max()) > 0.0
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(1)))
    moved = {**x, "R": x["R"] @ q.T + torch.tensor([1.0, -2.0, 0.5])}
    assert torch.allclose(model(**moved)[0], energy, rtol=1e-10, atol=0.0)


def test_gradients_reach_the_mpnn_and_element_parameters():
    model = perturb(AtomicC6Dispersion(mpnn_options=OPTIONS))
    model(**water_methane())[0].sum().backward()
    assert any(float(p.grad.abs().max()) > 0 for p in model.mpnn.parameters()
               if p.grad is not None)
    for p in model.element_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_guess_layer_is_frozen_so_deltas_are_environment_only():
    model = perturb(AtomicC6Dispersion(higher_order="delta", mpnn_options=OPTIONS))
    guess = list(model.mpnn.guess_layer.parameters())
    assert guess and not any(p.requires_grad for p in guess)
    model(**water_methane())[0].sum().backward()
    assert all(p.grad is None for p in guess)
    argon = ([18], [[0.0, 0.0, 0.0]])
    x = dimer_inputs([(argon, argon)], [((0, 0, 0), (3.8, 0.0, 0.0))])
    _, extra = model(**x)
    for k in ("d6", "d8", "d10", "dbeta"):
        assert float(extra["atom"][k].abs().max()) == 0.0


def test_anchor_freezes_element_c6_and_learns_one_scale():
    free = AtomicC6Dispersion(mpnn_options=OPTIONS)
    anchored = AtomicC6Dispersion(c6_anchor="free-atom-ratio", mpnn_options=OPTIONS)
    assert free.log_c6.requires_grad and not anchored.log_c6.requires_grad
    assert all(p is not anchored.log_c6 for p in anchored.element_parameters())
    with torch.no_grad():
        anchored.log_globals["c6_scale"].fill_(math.log(1.5))
    ratio = (anchored.element_log_c6() - free.element_log_c6()).exp()
    known = ~torch.isnan(anchored.free_c6)
    assert torch.allclose(ratio[known], torch.full_like(ratio[known], 1.5))


def test_prefit_lowers_the_loss_and_loads_the_best_values():
    x = water_methane(separations=(3.2, 3.6, 4.2, 5.0, 7.0))
    teacher = AtomicC6Dispersion(mpnn_options=OPTIONS)
    with torch.no_grad():
        teacher.log_c6.add_(0.3)
        teacher.log_globals["b"].add_(-0.2)
        target, _ = teacher(**x)
    model = AtomicC6Dispersion(mpnn_options=OPTIONS)
    loss = {"name": "huber+relative", "delta": 1.0, "relative_weight": 4.0,
            "relative_floor": 0.05, "relative_delta": 0.05}
    info = prefit(model, x, target, loss, steps=60, lr=0.03)
    assert info["prefit_loss_best"] < info["prefit_loss_first"]
    assert info["param_l1_moved"] > 0
    with torch.no_grad():
        pred, extra = model(**x)
    # The loaded values are the ones that scored the best loss.
    assert float(dispersion_objective(pred, target, loss)) == pytest.approx(
        info["prefit_loss_best"], rel=1e-12)
    assert float(extra["atom"]["d6"].abs().max()) == 0.0


@pytest.mark.parametrize("steps", [0, -1])
def test_prefit_rejects_fewer_than_one_step(steps):
    model = AtomicC6Dispersion(mpnn_options=OPTIONS)
    with pytest.raises(ValueError, match="steps"):
        prefit(model, water_methane(), torch.zeros(3),
               {"name": "huber", "delta": 1.0}, steps=steps)


def test_config_round_trips_through_a_state_dict():
    model = perturb(AtomicC6Dispersion(higher_order="element",
                                       c6_anchor="free-atom-ratio",
                                       mpnn_options=OPTIONS))
    clone = AtomicC6Dispersion(**model.config())
    clone.load_state_dict(model.state_dict())
    x = water_methane()
    assert torch.equal(clone(**x)[0], model(**x)[0])


@pytest.mark.parametrize("kwargs, match", [
    ({"damping": "xx"}, "damping"),
    ({"orders": (6, 10)}, "orders"),
    ({"higher_order": "free"}, "higher_order"),
    ({"damping": "tt", "beta": "far"}, "beta"),
    ({"damping": "bj", "beta": "atom"}, "Tang-Toennies"),
    ({"c6_anchor": "d3"}, "c6_anchor"),
    ({"mpnn": False, "beta": "atom"}, "need the MPNN"),
])
def test_invalid_options_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AtomicC6Dispersion(**kwargs)


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="loss name"):
        dispersion_objective(torch.zeros(2), torch.zeros(2), {"name": "mse"})
