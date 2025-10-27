"""
Test converted TensorFlow APNet models (atom and pair) with water dimers.

This test verifies that the TensorFlow SavedModel weights were correctly
converted to PyTorch format and can be used for prediction.
"""

import os
import pytest
import torch
import numpy as np
import qcelemental as qcel
from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model
from pprint import pprint as pp

current_file_path = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(current_file_path)

sapt0_elst = -10.779293
sapt0_ind = -3.414543
sapt0_exch = 11.390991
sapt0_disp = -2.436026
ref_water = qcel.models.Molecule.from_data("""
0 1
--
0 1
O                    -1.326958220000    -0.105938540000     0.018788150000
H                    -1.931665230000     1.600174310000    -0.021710520000
H                     0.486644270000     0.079598100000     0.009862480000
--
0 1
O                     3.907523240000     0.052757410000     0.001850160000
H                     4.619234940000    -0.775660840000     1.449615410000
H                     4.611000850000    -0.847154680000    -1.406756420000
units bohr
no_com
no_reorient
""")


def test_tf_converted_atom_model_loads():
    """Test that converted TF atom models can be loaded."""
    am_pt = AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am_pt.set_pretrained_model(model_id=0)
    v_pt = am_pt.predict_qcel_mols(
        [ref_water.get_fragment(0)], batch_size=1)[0]
    pp(v_pt[0])
    pp(v_pt[1])

    for i in range(5):
        model_path = os.path.join(
            project_root, f"models/ap2_tf/atom_models/atom{i}.pt")

        # Check file exists
        assert os.path.exists(
            model_path), f"Atom model {i} not found at {model_path}"

        # Load checkpoint
        checkpoint = torch.load(model_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint

        # Create model and load weights
        atom_model = AtomModel(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
        )
        atom_model.set_pretrained_model(model_path=model_path)

        # Verify model is in eval mode
        atom_model.model.eval()

        print(f"✓ Successfully loaded atom model {i}")
        v = atom_model.predict_qcel_mols(
            [ref_water.get_fragment(0)], batch_size=1)[0]
        pp(v[0])
        pp(v[1])
        assert np.allclose(np.array(v_pt[0]), np.array(v[0]), atol=1e-1), (
            f"Should be close to pt model\n{v_pt[0] = }\n{v[0] = }"
        )


def test_tf_converted_pair_model_loads():
    """Test that converted TF pair models can be loaded."""

    for i in range(5):
        atom_model_path = os.path.join(
            project_root, f"models/ap2_tf/atom_models/atom{i}.pt"
        )
        # Create model and load weights
        pair_model_path = os.path.join(
            project_root, f"models/ap2_tf/pair_models/pair{i}.pt"
        )

        # Check file exists
        assert os.path.exists(pair_model_path), (
            f"Pair model {i} not found at {pair_model_path}"
        )

        # Load checkpoint
        checkpoint = torch.load(pair_model_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint

        # Create atom model first (required for pair model)
        # Create pair model and load weights
        pair_model = APNet2Model(
            atom_model=None,
            ignore_database_null=True,
            use_GPU=False,
        )
        pair_model.set_pretrained_model(
            ap2_model_path=pair_model_path, am_model_path=atom_model_path
        )

        # Verify model is in eval mode
        pair_model.model.eval()

        print(f"✓ Successfully loaded pair model {i}")


def test_tf_converted_predict_water_dimer_single():
    """Test prediction on a single water dimer using converted TF models."""
    # pt reference
    print(
        f"SAPT0 Energies:\nELST={sapt0_elst}\nEXCH={sapt0_exch}\nIND={sapt0_ind}\nDISP={sapt0_disp}"
    )
    pair_model = APNet2Model(
        atom_model=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    pair_model.set_pretrained_model(
        ap2_model_path=os.path.join(
            project_root, "models/ap2_ensemble/ap2_0.pt"),
        am_model_path=os.path.join(project_root, "models/am_ensemble/am_0.pt"),
    )
    output = pair_model.predict_qcel_mols([ref_water], batch_size=1)
    pt_elst, pt_exch, pt_ind, pt_disp = output[0]
    print(
        f"\nPT    Energies:\nELST={pt_elst}\nEXCH={pt_exch}\nIND={pt_ind}\nDISP={pt_disp}"
    )
    assert np.isclose(pt_elst, sapt0_elst, atol=1.5e0), (
        f"ELST should be close to SAPT0 reference: {pt_elst} vs {sapt0_elst}"
    )
    assert np.isclose(pt_exch, sapt0_exch, atol=1e0), (
        f"EXCH should be close to SAPT0 reference: {pt_exch} vs {sapt0_exch}"
    )
    assert np.isclose(pt_ind, sapt0_ind, atol=1e0), (
        f"IND should be close to SAPT0 reference: {pt_ind} vs {sapt0_ind}"
    )
    assert np.isclose(pt_disp, sapt0_disp, atol=1e0), (
        f"DISP should be close to SAPT0 reference: {pt_disp} vs {sapt0_disp}"
    )

    atom_model_path = os.path.join(
        project_root, "models/ap2_tf/atom_models/atom0.pt")
    pair_model_path = os.path.join(
        project_root, "models/ap2_tf/pair_models/pair0.pt")
    # Load models
    pair_model = APNet2Model(
        atom_model=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    pair_model.set_pretrained_model(
        ap2_model_path=pair_model_path, am_model_path=atom_model_path
    )

    # Run prediction
    output = pair_model.predict_qcel_mols([ref_water], batch_size=1)
    tf_elst, tf_exch, tf_ind, tf_disp = output[0]
    print(
        f"\nTF    Energies:\nELST={tf_elst}\nEXCH={tf_exch}\nIND={tf_ind}\nDISP={tf_disp}"
    )
    assert np.isclose(tf_elst, sapt0_elst, atol=1.5e0), (
        f"ELST should be close to SAPT0 reference: {tf_elst} vs {sapt0_elst}"
    )
    assert np.isclose(tf_exch, sapt0_exch, atol=1e0), (
        f"EXCH should be close to SAPT0 reference: {tf_exch} vs {sapt0_exch}"
    )
    assert np.isclose(tf_ind, sapt0_ind, atol=1e0), (
        f"IND should be close to SAPT0 reference: {tf_ind} vs {sapt0_ind}"
    )
    assert np.isclose(tf_disp, sapt0_disp, atol=1e0), (
        f"DISP should be close to SAPT0 reference: {tf_disp} vs {sapt0_disp}"
    )


def test_tf_converted_predict_water_dimer_single_README():
    """Test prediction on a single water dimer using converted TF models."""
    # pt reference
    energies = [-2.617089,   -3.5254788,   2.460659,   -0.5823244,  -0.96994525]
    ref_elst, ref_exch, ref_ind, ref_disp = energies[1:]
    dimer = qcel.models.Molecule.from_data("""
        0 1
        O 0.000000 0.000000  0.000000
        H 0.758602 0.000000  0.504284
        H 0.260455 0.000000 -0.872893
        --
        0 1
        O 3.000000 0.500000  0.000000
        H 3.758602 0.500000  0.504284
        H 3.260455 0.500000 -0.872893
        """)
    atom_model_path = os.path.join(
        project_root, "models/ap2_tf/atom_models/atom0.pt")
    pair_model_path = os.path.join(
        project_root, "models/ap2_tf/pair_models/pair0.pt")
    # Load models
    pair_model = APNet2Model(
        atom_model=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    pair_model.set_pretrained_model(
        ap2_model_path=pair_model_path, am_model_path=atom_model_path
    )

    # Run prediction
    output = pair_model.predict_qcel_mols([dimer], batch_size=1)
    tf_elst, tf_exch, tf_ind, tf_disp = output[0]
    print(
        f"\nTF    Energies:\nELST={tf_elst}\nEXCH={tf_exch}\nIND={tf_ind}\nDISP={tf_disp}"
    )
    assert np.isclose(tf_elst, ref_elst, atol=1.5e0), (
        f"ELST should be close to reference: {tf_elst} vs {ref_elst}"
    )
    assert np.isclose(tf_exch, ref_exch, atol=1e0), (
        f"EXCH should be close to reference: {tf_exch} vs {ref_exch}"
    )
    assert np.isclose(tf_ind, ref_ind, atol=1e0), (
            f"IND should be close to reference: {tf_ind} vs {ref_ind}"
    )
    assert np.isclose(tf_disp, ref_disp, atol=1e0), (
        f"DISP should be close to reference: {tf_disp} vs {ref_disp}"
    )


if __name__ == "__main__":
    # Run tests manually
    print("Testing converted TensorFlow models...")
    # test_tf_converted_atom_model_loads()
    # test_tf_converted_pair_model_loads()
    # test_tf_converted_predict_water_dimer_single()
    test_tf_converted_predict_water_dimer_single_README()
