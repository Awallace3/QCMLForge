"""Focused tests for apnet_pt.model_print."""

import io
import sys
from pathlib import Path
from typing import cast

import pytest
from torch.nn.parameter import UninitializedParameter

from apnet_pt import get_model_info, print_model_tree
from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomModels.ap2_hirshfeld_atom_model import AtomHirshfeldMPNN
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (
    APNet3D3_AtomType_Model,
    APNet3D3_AtomType_MPNN,
)
from apnet_pt.AtomPairwiseModels.mtp_mtp import AtomTypeParamNN, DimerProp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def hirshfeld():
    return AtomHirshfeldMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)


@pytest.fixture(scope="module")
def atpnn(hirshfeld):
    return AtomTypeParamNN(
        atom_model=hirshfeld,
        n_message=1,
        n_neuron=16,
        n_embed=4,
        n_params=1,
        freeze_atom_model=True,
    )


@pytest.fixture(scope="module")
def dimer_prop(atpnn):
    return DimerProp(ATParam=atpnn, freeze_atom_model=True)


@pytest.fixture(scope="module")
def ap3d3_mpnn(dimer_prop):
    return APNet3D3_AtomType_MPNN(
        dimer_prop_model=dimer_prop,
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )


@pytest.fixture(scope="module")
def ap3d3_model(dimer_prop):
    return APNet3D3_AtomType_Model(
        dimer_prop_model=dimer_prop,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )


def build_demo_ap3d3_model():
    hirshfeld_model = AtomHirshfeldMPNN(
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )
    atpnn_model = AtomTypeParamNN(
        atom_model=cast(AtomMPNN, hirshfeld_model),
        n_message=1,
        n_neuron=16,
        n_embed=4,
        n_params=1,
        freeze_atom_model=True,
    )
    dimer_prop_model = DimerProp(ATParam=atpnn_model, freeze_atom_model=True)
    return APNet3D3_AtomType_Model(
        dimer_prop_model=dimer_prop_model,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )


def count_params(module):
    return sum(
        0 if isinstance(p, UninitializedParameter) else p.numel()
        for p in module.parameters()
    )


def capture_tree(target, **kwargs):
    buf = io.StringIO()
    print_model_tree(target, file=buf, **kwargs)
    return buf.getvalue()


def test_get_model_info_delegates_to_model_method(hirshfeld):
    info_method = hirshfeld.get_model_info()
    info_func = get_model_info(hirshfeld)
    assert info_func.name == info_method.name
    assert info_func.n_calls == info_method.n_calls
    assert info_func.outputs == info_method.outputs


def test_atom_type_param_info_tracks_runtime_atom_model(atpnn):
    info = atpnn.get_model_info()
    assert info.name == "AtomTypeParamNN"
    assert info.n_calls == 1
    assert info.inputs == ["h_list [from AtomHirshfeldMPNN]"]
    child = next(c for c in info.children if c.name == "AtomHirshfeldMPNN")
    assert child.n_calls == 1
    assert child.frozen is True


def test_atom_type_param_info_uses_nested_model_name_when_wrapped():
    inner = AtomTypeParamNN(
        atom_model=AtomMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4),
        n_message=1,
        n_neuron=16,
        n_embed=4,
        n_params=1,
        freeze_atom_model=True,
    )
    outer = AtomTypeParamNN(
        atom_model=inner,
        n_message=1,
        n_neuron=16,
        n_embed=4,
        n_params=1,
        freeze_atom_model=True,
    )
    assert outer.get_model_info().inputs == ["h_list [from AtomTypeParamNN]"]


def test_ap3d3_mpnn_info_exposes_call_note_and_outputs(ap3d3_mpnn):
    info = ap3d3_mpnn.get_model_info()
    assert info.name == "APNet3D3_AtomType_MPNN"
    assert info.n_calls == 1
    assert info.outputs == [
        "ΔE_elst_NN",
        "E_exch_NN",
        "ΔE_ind_NN",
        "ΔE_disp_NN",
    ]
    note_lower = info.call_note.lower()
    assert any(word in note_lower for word in ("intra", "monomer", "shared"))
    assert info.n_params >= 0
    assert info.n_params_total >= 0


def test_ap3d3_model_info_lists_expected_top_level_children(ap3d3_model):
    info = ap3d3_model.get_model_info()
    child_names = [c.name for c in info.children]
    assert info.name == "APNet3D3_AtomType_Model"
    assert child_names == [
        "AtomHirshfeldMPNN",
        "AtomTypeParamNN",
        "Classical",
        "APNet3D3_AtomType_MPNN",
    ]


def test_ap3d3_model_info_has_expected_call_counts_and_groups(ap3d3_model):
    info = ap3d3_model.get_model_info()
    hirshfeld = next(c for c in info.children if c.name == "AtomHirshfeldMPNN")
    atpnn = next(c for c in info.children if c.name == "AtomTypeParamNN")
    classical = next(c for c in info.children if c.name == "Classical")
    mpnn = next(c for c in info.children if c.name == "APNet3D3_AtomType_MPNN")
    assert hirshfeld.frozen is True
    assert hirshfeld.n_calls == 2
    assert atpnn.n_calls == 2
    assert classical.is_group is True
    assert {c.name for c in classical.children} == {
        "DampedMTPElectrostatics",
        "PointInducedDipole",
        "DFTD3Dispersion",
    }
    assert mpnn.n_calls == 1
    assert mpnn.call_note


def test_ap3d3_model_info_uses_flat_parameter_accounting(ap3d3_model):
    info = ap3d3_model.get_model_info()
    hirshfeld = next(c for c in info.children if c.name == "AtomHirshfeldMPNN")
    atpnn = next(c for c in info.children if c.name == "AtomTypeParamNN")
    mpnn = next(c for c in info.children if c.name == "APNet3D3_AtomType_MPNN")
    atom_model = ap3d3_model.model.dimer_prop_model.AtomTypeParam.atom_model
    at_param = ap3d3_model.model.dimer_prop_model.AtomTypeParam
    assert info.n_params_total == count_params(ap3d3_model.model)
    assert hirshfeld.n_params_total == count_params(atom_model)
    assert atpnn.children == []
    assert atpnn.n_params_total == count_params(at_param) - count_params(atom_model)
    assert mpnn.n_params_total == (
        count_params(ap3d3_model.model)
        - count_params(ap3d3_model.model.dimer_prop_model)
    )


def test_print_model_tree_unicode_output_contains_structure(ap3d3_model):
    output = capture_tree(ap3d3_model)
    assert "APNet3D3_AtomType_Model" in output
    assert "AtomHirshfeldMPNN" in output
    assert "AtomTypeParamNN" in output
    assert "Classical" in output
    assert "APNet3D3_AtomType_MPNN" in output
    assert "×2" in output
    assert "note:" in output
    assert "Inputs" in output
    assert "Outputs" in output
    assert "├─" in output or "└─" in output


def test_print_model_tree_ascii_fallback_uses_ascii_connectors(ap3d3_model):
    output = capture_tree(ap3d3_model, unicode=False)
    assert "APNet3D3_AtomType_Model" in output
    assert "├─" not in output
    assert "+- " in output or "\\- " in output


def test_ap3d3_model_info_prints_unicode_tree(capsys):
    build_demo_ap3d3_model().info()
    captured = capsys.readouterr()
    assert "APNet3D3_AtomType_Model" in captured.out
    assert "├─" in captured.out or "└─" in captured.out
