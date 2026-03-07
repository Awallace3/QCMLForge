import os
import tempfile

import qcelemental as qcel
import torch

from apnet_pt import AtomModels, atomic_datasets

torch.manual_seed(42)
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"

# Model paths for loading tests
am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
atp_path = f"{current_file_path}/test_models/ap3_ensemble_0/atp_mpnn_1.pt"

# Model paths for numerical inference tests
am_path_feb26 = f"{current_file_path}/test_models/am_3_feb26.pt"
idm_path_feb26 = f"{current_file_path}/test_models/idm_atp_am_1_feb26.pt"
atp_path_feb26 = f"{current_file_path}/test_models/atp_mpnn_1_feb26.pt"

mol_dimer = qcel.models.Molecule.from_data("""
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
""")

# Reference values from inference (computed with torch.manual_seed(42))
REF_E_ELST = -3.3664696902
REF_E_ELST_DIMER = -4.0914264821
REF_E_INDU = -0.7249567920


def test_aid_pretrained_loading():
    """
    Test loading a pre-trained InducedDipoleModel with frozen AtomMPNN.

    This test verifies loading an InducedDipoleModel configured as per:
        python train_models.py --train_am InducedDipoleModel \\
            --atom_mpnn_pretrained_path ./models/spice/am_3.pt

    The test:
    1. Creates an InducedDipoleModel with pretrained AtomMPNN
    2. Saves it to a temp file via training
    3. Loads it back via pre_trained_model_path
    4. Verifies the loaded model has correct architecture and weights
    """
    # Get test dataset
    ds = atomic_datasets.atomic_module_dataset(
        data_path, spec_type=6, testing=False, in_memory=True
    )

    # Load pretrained atomtype model for HFVR
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )

    # Create temporary file for saving/loading test
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        temp_model_path = tmp.name

    try:
        # Step 1: Create InducedDipoleModel with pretrained AtomMPNN
        # Note: use_nn_screening=False since test data lacks edge_index_full
        am_original = AtomModels.ap3_atom_model_frozen.InducedDipoleModel(
            atomtype_hfvr_model=atpm.model,
            atom_mpnn_pre_trained_path=am_path,
            use_nn_screening=False,
            precompute_hfvr=False,
            use_GPU=False,
            ignore_database_null=False,
            dataset=ds,
        )

        # Verify model has frozen AtomMPNN
        assert am_original.model.atom_mpnn_model is not None, (
            "Model should have atom_mpnn_model"
        )

        # Get sample weights from frozen AtomMPNN layers
        orig_weight = am_original.model.atom_mpnn_model.embed_layer.weight
        original_embed_weight = orig_weight.clone().detach()

        # Verify frozen layers are not trainable
        frozen_count = sum(
            1
            for p in am_original.model.atom_mpnn_model.parameters()
            if not p.requires_grad
        )
        assert frozen_count > 0, "AtomMPNN layers should be frozen"

        # Step 2: Save the model by training for 1 epoch
        am_original.train(
            n_epochs=1,
            batch_size=8,
            lr=5e-5,
            split_percent=0.5,
            model_path=temp_model_path,
            shuffle=False,
            skip_compile=True,
            dataloader_num_workers=0,
            world_size=1,
            omp_num_threads_per_process=4,
            random_seed=42,
        )

        # Verify checkpoint was saved with correct config
        checkpoint = torch.load(temp_model_path, weights_only=False)
        assert checkpoint["config"].get("has_pretrained_atom_mpnn", False), (
            "Checkpoint should have has_pretrained_atom_mpnn=True"
        )

        # Step 3: Load the model back via pre_trained_model_path
        am_loaded = AtomModels.ap3_atom_model_frozen.InducedDipoleModel(
            pre_trained_model_path=temp_model_path,
            precompute_hfvr=False,
            use_GPU=False,
            ignore_database_null=False,
            dataset=ds,
        )

        # Step 4: Verify loaded model architecture
        assert am_loaded.model.atom_mpnn_model is not None, (
            "Loaded model should have atom_mpnn_model"
        )

        # Verify weights match
        loaded_weight = am_loaded.model.atom_mpnn_model.embed_layer.weight
        loaded_embed_weight = loaded_weight.clone().detach()
        assert torch.allclose(original_embed_weight, loaded_embed_weight), (
            "Loaded AtomMPNN weights should match original"
        )

        print("test_aid_pretrained_loading passed")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)


def test_aid_numerical_inference():
    """
    Test numerical accuracy of InducedDipoleModel inference on a water dimer.

    This test loads pretrained models and verifies that inference produces
    expected electrostatic and induction energy values.
    """
    # Load atomtype model for HFVR computation
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path_feb26,
    )

    # Load InducedDipoleModel with atomtype_hfvr_model
    am = AtomModels.ap3_atom_model_frozen.InducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        pre_trained_model_path=idm_path_feb26,
        precompute_hfvr=False,
        use_GPU=False,
        ignore_database_null=True,
    )

    # Run inference on dimer
    E_elst, E_elst_dimer, E_indu = am.predict_elst_ind_dimer(
        [mol_dimer],
        batch_size=1,
    )

    # Verify numerical accuracy (tolerance for floating point)
    tol = 1e-5
    assert abs(E_elst[0] - REF_E_ELST) < tol, (
        f"E_elst mismatch: {E_elst[0]:.10f} vs {REF_E_ELST:.10f}"
    )
    assert abs(E_elst_dimer[0] - REF_E_ELST_DIMER) < tol, (
        f"E_elst_dimer mismatch: {E_elst_dimer[0]:.10f} vs {REF_E_ELST_DIMER:.10f}"
    )
    assert abs(E_indu[0] - REF_E_INDU) < tol, (
        f"E_indu mismatch: {E_indu[0]:.10f} vs {REF_E_INDU:.10f}"
    )

    print("test_aid_numerical_inference passed")
    print(f"  E_elst: {E_elst[0]:.10f} (ref: {REF_E_ELST:.10f})")
    print(f"  E_elst_dimer: {E_elst_dimer[0]:.10f} (ref: {REF_E_ELST_DIMER:.10f})")
    print(f"  E_indu: {E_indu[0]:.10f} (ref: {REF_E_INDU:.10f})")


if __name__ == "__main__":
    test_aid_pretrained_loading()
    test_aid_numerical_inference()
