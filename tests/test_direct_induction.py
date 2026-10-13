"""Explicit-property solver: no checkpoints, widths, or overlap parameters."""

import pytest
import torch
from apnet_pt import constants
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
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
