import os

import numpy as np
import pytest
import qcelemental as qcel
import torch

import apnet_pt


pytest.importorskip("e3nn")

current_file_path = os.path.dirname(os.path.realpath(__file__))

mol_water = qcel.models.Molecule.from_data(
    """
0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
"""
)

mon_element = qcel.models.Molecule.from_data(
    """
1 1
11   -0.902196054   -0.106060256   0.009942262
"""
)


def set_weights_to_value(model, value=0.02):
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(value)


def _flatten_quad_components(quads):
    quads = quads.reshape(-1, 9)
    return np.array([quads[i].flatten()[[0, 1, 2, 4, 5, 8]] for i in range(len(quads))])


def test_am_e3_architecture():
    atom_model = apnet_pt.AtomModels.ap2_atom_e3_model.AtomE3Model(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    set_weights_to_value(atom_model.model, 0.02)

    v = atom_model.predict_qcel_mols([mol_water], batch_size=1)
    charges, dipoles, quads, hlist = v[0]

    charges = charges.detach().cpu().numpy()
    dipoles = dipoles.detach().cpu().numpy()
    quads = quads.detach().cpu().numpy()
    hlist = hlist.detach().cpu().numpy()

    assert charges.shape == (3,)
    assert dipoles.shape == (3, 3)
    assert quads.shape == (3, 3, 3)
    assert hlist.shape[0] == 3
    assert hlist.shape[1] == atom_model.model.n_message + 1
    assert hlist.shape[2] == atom_model.model.n_embed

    # Neutral molecule; enforce charge conservation behavior.
    assert np.allclose(np.sum(charges), 0.0, atol=1e-4)

    # Quadrupole should be traceless.
    traces = np.trace(quads, axis1=1, axis2=2)
    assert np.allclose(traces, 0.0, atol=1e-3)

    # Keep this formatted output for local debugging with python.
    print(f"{charges=}")
    print(f"{dipoles=}")
    print(f"{_flatten_quad_components(quads)=}")
    print(f"{hlist.shape=}")


def test_am_e3_element_and_water_batch():
    atom_model = apnet_pt.AtomModels.ap2_atom_e3_model.AtomE3Model(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    set_weights_to_value(atom_model.model, 0.02)

    qcel_mols = [mon_element, mon_element, mol_water, mol_water]
    output = atom_model.predict_qcel_mols(qcel_mols, batch_size=4)

    assert len(output) == 4
    assert output[0][0].shape[0] == 1
    assert output[2][0].shape[0] == 3

    # Charge conservation per molecule: Na+ (+1), water (0)
    assert torch.allclose(output[0][0].sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(output[1][0].sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(output[2][0].sum(), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(output[3][0].sum(), torch.tensor(0.0), atol=1e-4)


@pytest.mark.parametrize(
    "e3_kwargs",
    [
        {
            "e3_lmax": 1,
            "e3_contraction": "einsum",
            "e3_dipole_mode": "l1",
            "e3_qpole_mode": "legacy",
            "e3_message_mode": "none",
        },
        {
            "e3_lmax": 2,
            "e3_contraction": "tensor_product",
            "e3_dipole_mode": "multi_l",
            "e3_qpole_mode": "l2",
            "e3_message_mode": "concat_sh",
        },
        {
            "e3_lmax": 3,
            "e3_contraction": "fully_connected_tp",
            "e3_dipole_mode": "multi_l",
            "e3_qpole_mode": "l2",
            "e3_message_mode": "concat_sh",
        },
    ],
)
def test_am_e3_configurable_modes_smoke(e3_kwargs):
    atom_model = apnet_pt.AtomModels.ap2_atom_e3_model.AtomE3Model(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
        **e3_kwargs,
    )
    set_weights_to_value(atom_model.model, 0.02)
    out = atom_model.predict_qcel_mols([mol_water], batch_size=1)
    charges, dipoles, quads, _ = out[0]
    assert charges.shape == (3,)
    assert dipoles.shape == (3, 3)
    assert quads.shape == (3, 3, 3)
    cfg = atom_model.model.get_config()
    for key, value in e3_kwargs.items():
        assert cfg[key] == value


if __name__ == "__main__":
    test_am_e3_architecture()
