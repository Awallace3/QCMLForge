"""The real long-range provider, run on its default kernels.

Route tests substitute ``StubLongRangeProvider``, so they cannot see a default
backend whose signature or return type drifted from the provider's contract.
Dispersion is injected: only electrostatics and induction are under test.
"""

import pytest
import torch

from apnet_pt import constants
from apnet_pt.mace.long_range import LongRangeSAPTProvider
from apnet_pt.mace.schema import AtomicPropertyBundle, PhysicsConfig

from tests.mace_stub_harness import _augment_batch, _batch


def _props(numbers, seed):
    generator = torch.Generator().manual_seed(seed)
    n = numbers.numel()
    hfvr = torch.full((n, 1), 0.8)
    alpha = constants.polarizability_table.index_select(0, numbers).reshape(-1, 1)
    return AtomicPropertyBundle(
        q=0.1 * torch.randn(n, 1, generator=generator),
        mu=0.05 * torch.randn(n, 3, generator=generator),
        quadrupole=torch.zeros(n, 3, 3),
        hfvr=hfvr,
        valence_width=torch.full((n, 1), 0.5),
        alpha=alpha.to(hfvr) * hfvr.pow(4.0 / 3.0),
        damping=torch.full((n, 1), 2.0),
    )


def _provider(**physics):
    return LongRangeSAPTProvider(
        PhysicsConfig(**physics),
        dispersion_kernel=lambda batch, params=None: torch.zeros(
            batch.e_ABfull_source.numel()
        ),
    )


@pytest.mark.parametrize("mode", ["damped-cliff", "undamped"])
def test_default_kernels_run_and_report_convergence(mode):
    batch = _augment_batch(_batch())
    energies = _provider(electrostatics_mode=mode)(
        batch, _props(batch.ZA, 0), _props(batch.ZB, 1)
    )
    assert energies.pair_elst.shape == energies.pair_ind.shape == (4,)
    assert torch.isfinite(energies.dimer_elst).all()
    assert torch.isfinite(energies.dimer_ind).all()
    assert energies.induction_diagnostics.converged
    assert energies.induction_diagnostics.iterations >= 1


def test_split_thole_damping_reaches_the_induction_kernel():
    batch = _augment_batch(_batch())
    props = _props(batch.ZA, 0), _props(batch.ZB, 1)
    equal = _provider(thole_direct=0.39, thole_mutual=0.39)(batch, *props)
    split = _provider(thole_direct=0.2, thole_mutual=0.39)(batch, *props)
    assert not torch.allclose(equal.dimer_ind, split.dimer_ind)
