"""Explicit-property solver: no checkpoints, widths, or overlap parameters."""

import itertools

import pytest
import torch

from apnet_pt import constants
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    _rackers_exact_dipoles,
    explicit_thole_induction,
    rackers_thole_induction,
)


def inputs():
    dtype = torch.float64
    return dict(
        RA=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype),
        RB=torch.tensor([[5.0, 0.2, 0.0]], dtype=dtype),
        qA=torch.tensor([0.2, -0.2], dtype=dtype, requires_grad=True),
        qB=torch.tensor([1.0], dtype=dtype, requires_grad=True),
        muA=torch.tensor(
            [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=dtype, requires_grad=True
        ),
        muB=torch.zeros(1, 3, dtype=dtype, requires_grad=True),
        alpha_A=torch.tensor([2.0, 2.0], dtype=dtype, requires_grad=True),
        alpha_B=torch.tensor([3.0], dtype=dtype, requires_grad=True),
        edges_AB=torch.tensor([[0, 1], [0, 0]]),
        edges_AA=torch.tensor([[0, 1], [1, 0]]),
        edges_BB=torch.empty(2, 0, dtype=torch.long),
        molecule_ind_A=torch.zeros(2, dtype=torch.long),
        molecule_ind_B=torch.zeros(1, dtype=torch.long),
        n_dimers=1,
        convergence_threshold=1e-10,
    )


def test_matches_existing_no_overlap_energy_and_gradients():
    p = inputs()
    energy, diag = explicit_thole_induction(**p)
    assert diag["scf_converged"] and energy.item() < 0
    zA, zB = torch.ones(2, dtype=torch.long), torch.ones(1, dtype=torch.long)
    table = constants.polarizability_table.double()
    hA, hB = (p["alpha_A"] / table[1]) ** 0.75, (p["alpha_B"] / table[1]) ** 0.75
    ab, aa, bb = p["edges_AB"], p["edges_AA"], p["edges_BB"]
    old = rackers_thole_induction(
        zA,
        p["RA"],
        p["qA"],
        p["muA"],
        torch.zeros(2, 3, 3, dtype=torch.float64),
        zB,
        p["RB"],
        p["qB"],
        p["muB"],
        torch.zeros(1, 3, 3, dtype=torch.float64),
        ab[0],
        ab[1],
        aa[0],
        bb[0],
        aa[1],
        bb[1],
        hA,
        hB,
        torch.ones_like(hA),
        torch.ones_like(hB),
        torch.full_like(hA, 0.38539),
        torch.full_like(hB, 0.38539),
        torch.full_like(hA, 0.38539),
        torch.full_like(hB, 0.38539),
        torch.zeros_like(hA),
        torch.zeros_like(hB),
        include_overlap=False,
        convergence_threshold=1e-10,
        convergence_norm="rms",
        polarizability_table=table,
    )
    torch.testing.assert_close(energy.sum(), old.sum(), rtol=1e-8, atol=1e-10)
    energy.sum().backward()
    for key in ("qA", "qB", "muA", "muB", "alpha_A", "alpha_B"):
        assert torch.isfinite(p[key].grad).all()
        assert p[key].grad.abs().max() > 0


def test_batch_composition_does_not_dilute_convergence():
    p = inputs()
    alone, _ = explicit_thole_induction(**p)
    joined = {**p, "n_dimers": 2}
    for key in ("RA", "RB"):
        joined[key] = torch.cat((p[key], p[key] + 20.0))
    for key in ("qA", "qB", "muA", "muB", "alpha_A", "alpha_B"):
        joined[key] = torch.cat((p[key], p[key]))
    joined["alpha_B"] = torch.cat((p["alpha_B"], 4 * p["alpha_B"]))
    joined["edges_AB"] = torch.cat(
        (p["edges_AB"], p["edges_AB"] + torch.tensor([[2], [1]])), 1
    )
    joined["edges_AA"] = torch.cat((p["edges_AA"], p["edges_AA"] + 2), 1)
    joined["molecule_ind_A"] = torch.tensor([0, 0, 1, 1])
    joined["molecule_ind_B"] = torch.tensor([0, 1])
    batched, diag = explicit_thole_induction(**joined)
    assert diag["scf_converged"]
    torch.testing.assert_close(batched[:1], alone, rtol=1e-8, atol=1e-9)


def test_nonconvergence_is_not_silently_accepted():
    p = inputs()
    with pytest.raises(RuntimeError, match="converge"):
        explicit_thole_induction(
            **{**p, "max_iterations": 1, "convergence_threshold": 1e-16}
        )


def test_nonpositive_polarizability_is_rejected():
    p = inputs()
    with pytest.raises(ValueError, match="polarizabilities"):
        explicit_thole_induction(**{**p, "alpha_A": -p["alpha_A"]})


def all_pairs(n):
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = i != j
    return torch.stack((i[keep], j[keep]))


def cube(spacing, alpha, shift=0.0):
    """An eight-atom cube A next to a point charge B: one strongly coupled
    dimer whose response is stable but slow for the damped loop."""
    dtype = torch.float64
    R = torch.tensor(list(itertools.product(range(2), repeat=3)), dtype=dtype)
    n = len(R)
    return {
        "RA": R * spacing + shift,
        "RB": torch.tensor([[2 * spacing + 4.0, 0.3, 0.2]], dtype=dtype) + shift,
        "qA": torch.linspace(-0.1, 0.1, n, dtype=dtype),
        "qB": torch.tensor([1.0], dtype=dtype),
        "muA": torch.zeros(n, 3, dtype=dtype),
        "muB": torch.zeros(1, 3, dtype=dtype),
        "alpha_A": torch.full((n,), float(alpha), dtype=dtype),
        "alpha_B": torch.tensor([3.0], dtype=dtype),
        "edges_AB": torch.stack((torch.arange(n), torch.zeros(n, dtype=torch.long))),
        "edges_AA": all_pairs(n),
        "edges_BB": torch.empty(2, 0, dtype=torch.long),
        "molecule_ind_A": torch.zeros(n, dtype=torch.long),
        "molecule_ind_B": torch.zeros(1, dtype=torch.long),
        "n_dimers": 1,
        "convergence_threshold": 1e-11,
        "max_iterations": 20000,
    }


def join(a, b):
    """Batch two dimers of different sizes (b's indices offset after a's)."""
    nA, nB = len(a["RA"]), len(a["RB"])
    out = {k: b[k] for k in ("convergence_threshold", "max_iterations")}
    for key in ("RA", "RB", "qA", "qB", "muA", "muB", "alpha_A", "alpha_B"):
        out[key] = torch.cat((a[key], b[key]))
    out["edges_AB"] = torch.cat(
        (a["edges_AB"], b["edges_AB"] + torch.tensor([[nA], [nB]])), 1)
    out["edges_AA"] = torch.cat((a["edges_AA"], b["edges_AA"] + nA), 1)
    out["edges_BB"] = torch.cat((a["edges_BB"], b["edges_BB"] + nB), 1)
    out["molecule_ind_A"] = torch.cat((a["molecule_ind_A"], b["molecule_ind_A"] + 1))
    out["molecule_ind_B"] = torch.cat((a["molecule_ind_B"], b["molecule_ind_B"] + 1))
    out["n_dimers"] = 2
    return out


@pytest.mark.parametrize("make", [
    lambda: inputs(),
    lambda: cube(1.0, 12.0),                      # loop needs ~180 iterations
    lambda: join(inputs(), cube(1.4, 8.0, 30.0)),  # padding, unequal sizes
])
def test_exact_solver_matches_converged_loop_energy_and_gradients(make):
    keys = ("qA", "qB", "muA", "muB", "alpha_A", "alpha_B")
    results = []
    for solver in ("iterative", "exact"):
        p = make()
        for key in keys:
            p[key] = p[key].detach().clone().requires_grad_(True)
        energy, diag = explicit_thole_induction(**p, solver=solver)
        assert diag["scf_converged"]
        grads = torch.autograd.grad(energy.sum(), [p[k] for k in keys])
        results.append((energy.detach(), grads, diag))
    (e_it, g_it, _), (e_ex, g_ex, diag) = results
    torch.testing.assert_close(e_ex, e_it, rtol=1e-9, atol=1e-10)
    for a, b in zip(g_ex, g_it):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)
    assert diag["scf_iterations"] == 0 and diag["scf_residual"] < 1e-10


def test_exact_solver_handles_stable_dimer_the_loop_cannot():
    p = cube(1.0, 30.0)
    with pytest.raises(RuntimeError, match="converge"):
        explicit_thole_induction(**{**p, "max_iterations": 3000})
    energy, diag = explicit_thole_induction(**p, solver="exact")
    assert bool(torch.isfinite(energy).all()) and energy.item() < 0
    assert diag["scf_residual"] < 1e-10


def test_exact_solver_reports_polarization_catastrophe():
    one = torch.ones(1, dtype=torch.float64)
    T = 10.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0)
    with pytest.raises(RuntimeError, match="catastrophe.*\\[0\\]"):
        _rackers_exact_dipoles(
            one, one, torch.ones(1, 3, dtype=torch.float64),
            torch.zeros(1, 3, dtype=torch.float64),
            torch.tensor([[0], [0]]), torch.empty(2, 0, dtype=torch.long),
            torch.empty(2, 0, dtype=torch.long), T, T[:0], T[:0],
            torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long), 1)


def test_unknown_solver_is_rejected():
    with pytest.raises(ValueError, match="solver"):
        explicit_thole_induction(**inputs(), solver="cg")


@pytest.mark.parametrize(
    "override, match",
    [
        ({"convergence_norm": "maxx"}, "convergence_norm"),
        ({"convergence_norm": None}, "convergence_norm"),
        ({"solver": None}, "solver"),
        ({"max_iterations": 2.5}, "max_iterations"),
        ({"max_iterations": True}, "max_iterations"),
        ({"convergence_threshold": "1e-6"}, "convergence_threshold"),
        ({"convergence_threshold": 0.0}, "convergence_threshold"),
        ({"omega": "0.7"}, "omega"),
        ({"thole_direct": None}, "thole_direct"),
        ({"n_dimers": 1.0}, "n_dimers"),
    ],
)
def test_invalid_solver_controls_are_rejected(override, match):
    with pytest.raises(ValueError, match=match):
        explicit_thole_induction(**{**inputs(), **override})


def test_exact_solver_rejects_residual_above_threshold():
    """Cholesky reads one triangle only, so one-directional AA edges solve a
    different system without failing; the residual check must catch it."""
    p = cube(1.0, 12.0)
    aa = p["edges_AA"]
    p["edges_AA"] = aa[:, aa[0] < aa[1]]
    with pytest.raises(RuntimeError, match="converge"):
        explicit_thole_induction(**p, solver="exact")
