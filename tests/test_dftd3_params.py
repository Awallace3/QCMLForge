import pytest
import qcelemental as qcel
import torch

from qcml_dftd3.d3 import (
    D3_DAMPING_PARAMETER_SETS,
    d3,
    d3_pair_energies,
    d3_pair_terms,
    params_intermolecular_saptpbe0_d3i,
    resolve_d3_damping_parameters,
)
from apnet_pt.pt_datasets.ap3_fused_ds import qcel_dimer_to_fused_data

# The parameters every model route used before named sets existed.
SAPT_PBE0_D3I = {"s6": 1.0, "s8": 0.8614, "a1": 0.7171, "a2": 0.5375}

water_dimer = qcel.models.Molecule.from_data("""
0 1
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
--
0 1
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
units angstrom
""")


@pytest.fixture
def water_batch():
    return qcel_dimer_to_fused_data(water_dimer, dimer_ind=0)


def test_default_d3_damping_parameters_unchanged():
    assert resolve_d3_damping_parameters() == SAPT_PBE0_D3I
    assert resolve_d3_damping_parameters(None) == SAPT_PBE0_D3I
    assert params_intermolecular_saptpbe0_d3i == SAPT_PBE0_D3I
    assert d3.__defaults__ == (params_intermolecular_saptpbe0_d3i,)


def test_default_d3_energy_unchanged(water_batch):
    default = d3(water_batch).sum()
    explicit = d3(water_batch, params=SAPT_PBE0_D3I).sum()
    assert torch.equal(default, explicit)
    # value computed before named sets were added (commit 06d14752)
    assert default.item() == pytest.approx(-2.912623, abs=1e-5)


def test_default_set_is_selectable_by_name(water_batch):
    assert resolve_d3_damping_parameters("sapt-pbe0-d3i") == SAPT_PBE0_D3I
    named = d3(water_batch, params="sapt-pbe0-d3i").sum()
    assert torch.equal(named, d3(water_batch).sum())


def test_resolved_named_set_is_a_copy():
    resolved = resolve_d3_damping_parameters("sapt-pbe0-d3i")
    resolved["a1"] = -1.0
    assert D3_DAMPING_PARAMETER_SETS["sapt-pbe0-d3i"]["a1"] == 0.7171


def test_unknown_named_set_raises():
    with pytest.raises(ValueError, match="Unknown D3 damping parameter set"):
        resolve_d3_damping_parameters("sapt0")


def test_pair_terms_reproduce_d3(water_batch):
    terms = d3_pair_terms(water_batch)
    other = {"s6": 1.0, "s8": 0.5, "a1": 0.3, "a2": 4.0}
    for params in (SAPT_PBE0_D3I, other):
        assert torch.equal(d3_pair_energies(terms, params), d3(water_batch, params))


def test_pair_energies_differentiable_in_damping_parameters(water_batch):
    terms = d3_pair_terms(water_batch)
    params = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
              for k, v in SAPT_PBE0_D3I.items()}
    terms = {k: v.double() for k, v in terms.items()}
    d3_pair_energies(terms, params).sum().backward()
    for key in ("s8", "a1", "a2"):
        assert params[key].grad is not None and params[key].grad != 0
