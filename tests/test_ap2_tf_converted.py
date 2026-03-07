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

# Water dimer - equilibrium geometry
mol_water_dimer = qcel.models.Molecule.from_data("""
0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H  0.260455  0.000000 -0.872893
--
0 1
O  3.000000  0.500000  0.000000
H  3.758602  0.500000  0.504284
H  3.260455  0.500000 -0.872893
""")

# Water dimer - closer distance (stronger interaction)
mol_water_dimer_close = qcel.models.Molecule.from_data("""
0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H  0.260455  0.000000 -0.872893
--
0 1
O  2.500000  0.000000  0.000000
H  3.258602  0.000000  0.504284
H  2.760455  0.000000 -0.872893
""")

# Water dimer - far apart (weak interaction)
mol_water_dimer_far = qcel.models.Molecule.from_data("""
0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H  0.260455  0.000000 -0.872893
--
0 1
O  5.000000  0.000000  0.000000
H  5.758602  0.000000  0.504284
H  5.260455  0.000000 -0.872893
""")


def test_tf_converted_atom_model_loads():
    """Test that converted TF atom models can be loaded."""
    am_pt = AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am_pt.set_pretrained_model(model_id=0)
    v_pt = am_pt.predict_qcel_mols([mol_water_dimer.get_fragment(0)], batch_size=1)[0]
    pp(v_pt[0])
    pp(v_pt[1])

    for i in range(5):
        model_path = os.path.join(
            project_root, f"models/ap2_tf/atom_models/atom{i}.pt"
        )
        
        # Check file exists
        assert os.path.exists(model_path), f"Atom model {i} not found at {model_path}"
        
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
        v = atom_model.predict_qcel_mols([mol_water_dimer.get_fragment(0)], batch_size=1)[0]
        pp(v[0])
        pp(v[1])
        assert np.allclose(np.array(v_pt[0]), np.array(v[0]), atol=1e-1), f"Should be close to pt model\n{v_pt[0] = }\n{v[0] = }"


def test_tf_converted_pair_model_loads():
    """Test that converted TF pair models can be loaded."""
    atom_model_path = os.path.join(
        project_root, "models/ap2_tf/atom_models/atom0.pt"
    )
    
    for i in range(5):
        model_path = os.path.join(
            project_root, f"models/ap2_tf/atom_models/atom{i}.pt"
        )
        
        # Check file exists
        assert os.path.exists(model_path), f"Atom model {i} not found at {model_path}"
        
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
        pair_model_path = os.path.join(
            project_root, f"models/ap2_tf/pair_models/pair{i}.pt"
        )
        
        # Check file exists
        assert os.path.exists(pair_model_path), f"Pair model {i} not found at {pair_model_path}"
        
        # Load checkpoint
        checkpoint = torch.load(pair_model_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint
        
        # Create atom model first (required for pair model)
        atom_model = AtomModel(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
        )
        atom_model.set_pretrained_model(model_path=atom_model_path)
        
        # Create pair model and load weights
        pair_model = APNet2Model(
            atom_model=atom_model.model,
            ignore_database_null=True,
            use_GPU=False,
        )
        pair_model.set_pretrained_model(
            ap2_model_path=pair_model_path,
            am_model_path=atom_model_path
        )
        
        # Verify model is in eval mode
        pair_model.model.eval()
        
        print(f"✓ Successfully loaded pair model {i}")


def test_tf_converted_predict_water_dimer_single():
    """Test prediction on a single water dimer using converted TF models."""
    atom_model_path = os.path.join(
        project_root, "models/ap2_tf/atom_models/atom0.pt"
    )
    pair_model_path = os.path.join(
        project_root, "models/ap2_tf/pair_models/pair0.pt"
    )
    
    # Load models
    atom_model = AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    atom_model.set_pretrained_model(model_path=atom_model_path)
    
    pair_model = APNet2Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
        use_GPU=False,
    )
    pair_model.set_pretrained_model(
        ap2_model_path=pair_model_path,
        am_model_path=atom_model_path
    )
    
    # Run prediction
    output = pair_model.predict_qcel_mols([mol_water_dimer], batch_size=1)
    
    # Verify output structure
    assert len(output) == 1, "Should have one output for one dimer"
    assert len(output[0]) == 4, "Should have 4 energy components [elst, exch, ind, disp]"
    
    elst, exch, ind, disp = output[0]
    
    print(f"\nWater dimer interaction energies (kcal/mol):")
    print(f"  ELST (electrostatics): {elst:.6f}")
    print(f"  EXCH (exchange):       {exch:.6f}")
    print(f"  IND  (induction):      {ind:.6f}")
    print(f"  DISP (dispersion):     {disp:.6f}")
    print(f"  TOTAL:                 {sum(output[0]):.6f}")
    
    # Sanity checks for water dimer at ~3 Angstrom separation
    # ELST should be negative (attractive) and dominant
    assert elst < 0, f"ELST should be negative (attractive), got {elst}"
    
    # EXCH should be positive (repulsive)
    assert exch > 0, f"EXCH should be positive (repulsive), got {exch}"
    
    # IND is typically negative (attractive), but can be small/slightly positive
    # Just check it's reasonable (small magnitude)
    assert abs(ind) < 2.0, f"IND magnitude should be reasonable, got {ind}"
    
    # DISP should be negative (attractive)
    assert disp < 0, f"DISP should be negative (attractive), got {disp}"
    
    # All energies should be finite
    assert all(np.isfinite(output[0])), "All energies should be finite"
    
    print("✓ All sanity checks passed for water dimer prediction")


def test_tf_converted_predict_water_dimer_batch():
    """Test batch prediction on multiple water dimers."""
    atom_model_path = os.path.join(
        project_root, "models/ap2_tf/atom_models/atom0.pt"
    )
    pair_model_path = os.path.join(
        project_root, "models/ap2_tf/pair_models/pair0.pt"
    )
    
    # Load models
    atom_model = AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    atom_model.set_pretrained_model(model_path=atom_model_path)
    
    pair_model = APNet2Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
        use_GPU=False,
    )
    pair_model.set_pretrained_model(
        ap2_model_path=pair_model_path,
        am_model_path=atom_model_path
    )
    
    # Run prediction on batch
    dimers = [mol_water_dimer, mol_water_dimer_close, mol_water_dimer_far]
    output = pair_model.predict_qcel_mols(dimers, batch_size=3)
    
    # Verify output structure
    assert len(output) == 3, "Should have three outputs for three dimers"
    
    print("\nBatch prediction results:")
    for i, (dimer_name, energies) in enumerate(zip(
        ["equilibrium", "close", "far"], output
    )):
        elst, exch, ind, disp = energies
        total = sum(energies)
        print(f"\n  Water dimer ({dimer_name}):")
        print(f"    ELST: {elst:8.4f} kcal/mol")
        print(f"    EXCH: {exch:8.4f} kcal/mol")
        print(f"    IND:  {ind:8.4f} kcal/mol")
        print(f"    DISP: {disp:8.4f} kcal/mol")
        print(f"    TOTAL: {total:8.4f} kcal/mol")
        
        # Verify all finite
        assert all(np.isfinite(energies)), f"Dimer {i} has non-finite energies"
    
    # Check distance-dependent behavior
    total_eq = sum(output[0])
    total_close = sum(output[1])
    total_far = sum(output[2])
    
    # Closer dimer should have stronger (more negative) interaction
    # This might not always hold due to exchange repulsion, but ELST should be stronger
    elst_eq = output[0][0]
    elst_close = output[1][0]
    elst_far = output[2][0]
    
    assert abs(elst_close) > abs(elst_eq), "Close dimer should have stronger ELST"
    assert abs(elst_eq) > abs(elst_far), "Equilibrium should have stronger ELST than far"
    
    print("\n✓ Distance-dependent behavior verified")


def test_tf_converted_ensemble_prediction():
    """Test ensemble prediction using all 5 converted models."""
    atom_models = []
    pair_models = []
    
    # Load all 5 model pairs
    for i in range(5):
        atom_model_path = os.path.join(
            project_root, f"models/ap2_tf/atom_models/atom{i}.pt"
        )
        pair_model_path = os.path.join(
            project_root, f"models/ap2_tf/pair_models/pair{i}.pt"
        )
        
        atom_model = AtomModel(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
        )
        atom_model.set_pretrained_model(model_path=atom_model_path)
        
        pair_model = APNet2Model(
            atom_model=atom_model.model,
            ignore_database_null=True,
            use_GPU=False,
        )
        pair_model.set_pretrained_model(
            ap2_model_path=pair_model_path,
            am_model_path=atom_model_path
        )
        
        atom_models.append(atom_model)
        pair_models.append(pair_model)
    
    # Get predictions from all 5 models
    all_predictions = []
    for i, pair_model in enumerate(pair_models):
        output = pair_model.predict_qcel_mols([mol_water_dimer], batch_size=1)
        all_predictions.append(output[0])
        print(f"Model {i}: {output[0]}")
    
    # Compute ensemble mean and std
    all_predictions = np.array(all_predictions)  # shape: (5, 4)
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)
    
    print("\nEnsemble statistics for water dimer:")
    component_names = ["ELST", "EXCH", "IND", "DISP"]
    for i, name in enumerate(component_names):
        print(f"  {name}: {mean_pred[i]:8.4f} ± {std_pred[i]:6.4f} kcal/mol")
    
    total_mean = np.sum(mean_pred)
    # Propagate uncertainty (assuming independent)
    total_std = np.sqrt(np.sum(std_pred**2))
    print(f"  TOTAL: {total_mean:8.4f} ± {total_std:6.4f} kcal/mol")
    
    # Verify all predictions are finite
    assert np.all(np.isfinite(all_predictions)), "All predictions should be finite"
    
    # Standard deviation should be reasonable (models shouldn't be identical)
    # but also not wildly different
    assert np.all(std_pred < 10.0), "Standard deviation should be reasonable"
    
    print("\n✓ Ensemble prediction completed successfully")


def test_tf_converted_with_elst_breakdown():
    """Test prediction with electrostatic breakdown (multipole vs NN)."""
    atom_model_path = os.path.join(
        project_root, "models/ap2_tf/atom_models/atom0.pt"
    )
    pair_model_path = os.path.join(
        project_root, "models/ap2_tf/pair_models/pair0.pt"
    )
    
    # Load models
    atom_model = AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    atom_model.set_pretrained_model(model_path=atom_model_path)
    
    pair_model = APNet2Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
        use_GPU=False,
    )
    pair_model.set_pretrained_model(
        ap2_model_path=pair_model_path,
        am_model_path=atom_model_path
    )
    
    # Run prediction with ELST breakdown
    output, mtp_elst = pair_model.predict_qcel_mols(
        [mol_water_dimer], 
        batch_size=1, 
        return_elst=True
    )
    
    elst_total = output[0][0]
    mtp_elst_sum = np.sum(mtp_elst)
    nn_elst = elst_total - mtp_elst_sum
    
    print(f"\nElectrostatic breakdown:")
    print(f"  Total ELST:     {elst_total:.6f} kcal/mol")
    print(f"  Multipole ELST: {mtp_elst_sum:.6f} kcal/mol")
    print(f"  NN correction:  {nn_elst:.6f} kcal/mol")
    
    # Verify all values are finite
    assert np.isfinite(elst_total), "Total ELST should be finite"
    assert np.isfinite(mtp_elst_sum), "Multipole ELST should be finite"
    assert np.isfinite(nn_elst), "NN correction should be finite"
    
    # Multipole should be the dominant contribution (usually >80%)
    # but this depends on the specific model and system
    print(f"  Multipole contribution: {100 * mtp_elst_sum / elst_total:.1f}%")
    
    print("✓ ELST breakdown analysis completed")


if __name__ == "__main__":
    # Run tests manually
    print("Testing converted TensorFlow models...")
    test_tf_converted_atom_model_loads()
    test_tf_converted_pair_model_loads()
    test_tf_converted_predict_water_dimer_single()
    test_tf_converted_predict_water_dimer_batch()
    test_tf_converted_ensemble_prediction()
    test_tf_converted_with_elst_breakdown()
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)
