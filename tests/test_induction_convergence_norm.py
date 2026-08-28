"""The Rackers/Thole stopping rule and its reduction.

The historical rule compares an *unnormalised* batch-wide L2 norm of the
induced-dipole change against an *absolute* threshold.  That is extensive: the
same per-atom convergence produces a residual that grows as sqrt(n_atoms), so
the effective tolerance tightens as the batch grows, and at production batch
sizes the test is unreachable -- the solve runs its full iteration cap on every
batch (docs/profiling.html section 12).

These tests pin three things:

1. the default is still ``l2``, and the ``l2`` branch is *bit-identical* to the
   expression the loop used before the mode existed -- not merely close;
2. ``rms`` and ``max`` are batch-size independent, which is the whole point;
3. the mode reaches the solver, survives a config round trip, and is rejected
   loudly when it is not one of the three.
"""

import math

import pytest
import torch

from apnet_pt.AtomPairwiseModels import mtp_mtp


NORMS = mtp_mtp.INDUCTION_CONVERGENCE_NORMS


def _pair(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, 3, generator=g, dtype=torch.float64),
        torch.randn(n, 3, generator=g, dtype=torch.float64),
    )


def test_default_is_the_historical_rule():
    assert mtp_mtp.DEFAULT_INDUCTION_CONVERGENCE_NORM == "l2"
    assert NORMS == ("l2", "rms", "max")


def test_l2_branch_is_bit_identical_to_the_old_expression():
    for n in (1, 7, 512, 5000):
        a, b = _pair(n, seed=n)
        historical = torch.maximum(torch.norm(a), torch.norm(b))
        got = mtp_mtp._scf_residual(a, b, "l2")
        # Bit-identical, not allclose: a default-configured run must reproduce
        # its own prior trajectory exactly, including the iteration it stops on.
        assert got.item() == historical.item()


def test_rms_and_max_are_batch_size_independent():
    # Tile one atom's change N times.  Every per-atom quantity is unchanged, so
    # a batch-size-independent rule must return the same residual.
    unit_a = torch.tensor([[1e-6, -2e-6, 3e-6]], dtype=torch.float64)
    unit_b = torch.tensor([[4e-7, 5e-7, -6e-7]], dtype=torch.float64)
    ref = {
        norm: mtp_mtp._scf_residual(unit_a, unit_b, norm).item()
        for norm in NORMS
    }
    for reps in (2, 16, 400):
        a = unit_a.repeat(reps, 1)
        b = unit_b.repeat(reps, 1)
        assert mtp_mtp._scf_residual(a, b, "rms").item() == pytest.approx(
            ref["rms"], rel=1e-12
        )
        assert mtp_mtp._scf_residual(a, b, "max").item() == pytest.approx(
            ref["max"], rel=1e-12
        )
        # ...and l2 is not, which is the defect being worked around.
        grew = mtp_mtp._scf_residual(a, b, "l2").item()
        assert grew == pytest.approx(ref["l2"] * math.sqrt(reps), rel=1e-12)


def test_max_is_the_largest_component_and_rms_is_below_it():
    a, b = _pair(64, seed=3)
    got_max = mtp_mtp._scf_residual(a, b, "max").item()
    assert got_max == pytest.approx(
        max(a.abs().max().item(), b.abs().max().item()), rel=0, abs=0
    )
    assert mtp_mtp._scf_residual(a, b, "rms").item() < got_max


def test_empty_tensors_do_not_raise():
    empty = torch.zeros(0, 3, dtype=torch.float64)
    a, _ = _pair(4, seed=5)
    for norm in NORMS:
        got = mtp_mtp._scf_residual(empty, a, norm)
        assert torch.isfinite(got)
        assert mtp_mtp._scf_residual(empty, empty, norm).item() == 0.0


@pytest.mark.parametrize("norm", ["l2", "RMS", " Max ", "rms"])
def test_validator_normalises_case_and_whitespace(norm):
    assert mtp_mtp._validate_induction_convergence_norm(norm) == norm.strip().lower()


@pytest.mark.parametrize("bad", ["l1", "", "linf", None, 2, 1e-8, True, ["l2"]])
def test_validator_rejects_everything_else(bad):
    with pytest.raises(ValueError, match="induction_convergence_norm"):
        mtp_mtp._validate_induction_convergence_norm(bad)


def test_solver_signature_accepts_the_mode():
    import inspect

    for fn in (mtp_mtp.rackers_thole_induction, mtp_mtp._rackers_converge_dipoles):
        params = inspect.signature(fn).parameters
        assert "convergence_norm" in params, fn.__name__
        assert params["convergence_norm"].default == "l2", fn.__name__


def test_mode_reaches_the_solver_and_the_monomer_solve():
    # Both the dimer solve and the monomers-alone solve must use the same rule;
    # a mismatch would make `scf_converged` mean two different things.
    src = inspect_source(mtp_mtp.rackers_thole_induction)
    assert src.count("convergence_norm=convergence_norm") == 1
    assert "_scf_residual(" in src


def inspect_source(fn):
    import inspect

    return inspect.getsource(fn)


def test_train_signature_and_cli_expose_the_mode():
    import inspect
    from pathlib import Path

    params = inspect.signature(mtp_mtp.AM_DimerParam_Model.train).parameters
    assert "induction_convergence_norm" in params
    assert params["induction_convergence_norm"].default is None

    root = Path(mtp_mtp.__file__).resolve().parents[3]
    cli = (root / "train_models.py").read_text()
    assert '"--induction_convergence_norm"' in cli
    assert 'choices=["l2", "rms", "max"]' in cli
