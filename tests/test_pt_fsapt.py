import apnet_pt
from apnet_pt.pt_datasets.ap3_fused_fsapt_ds import (
    ap3_fused_fsapt_module_dataset_lmdb,
    ap3_fused_fsapt_collate_update,
)
from apnet_pt.AtomPairwiseModels.apnet3_fused import APNet3_AtomType_Model
import os
import numpy as np
import pytest
from glob import glob
import qcelemental as qcel
import torch
import pandas as pd
from pprint import pprint as pp
import shutil
import tempfile


torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../src/apnet_pt/models/am_ensemble/am_0.pt"

am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_h+1_3.pt"
at_elst_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
ap3_path = f"{current_file_path}/test_models/ap3_ensemble_0/ap3_.pt"
am_hf_path = f"{current_file_path}/test_models/am_hf_0.pt"


def test_ap3_fused_fsapt():
    # Objective: train models to predict correct FSAPT pairwise energies for
    # AP3 fused model. Dataset should contain Data() with labels scalar labels
    # of F-Electrostatics, F-Exchange, F-Dispersion, F-Total, F-Induction for
    # each fragment along with correct indices to sum to match the functional
    # group breakdown. Then, the training of AP3 fused model on this type of dataset
    # will use the Frag1_indices and Frag2_indices to sum the atomic contributions
    # to get the fragment energies for computing the loss during training.

    temp_dir = tempfile.mkdtemp()
    test_df_path = f"{data_path}/raw/fsapt_test_data.pkl"
    train_df_path = f"{data_path}/raw/fsapt_train_data.pkl"
    # mkdir raw under temp_dir and copy test_df there
    raw_dir = os.path.join(temp_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Copying test dataframe to {raw_dir}")
    shutil.copy(test_df_path, raw_dir)
    shutil.copy(train_df_path, raw_dir)
    # see if the dataframe copied correctly
    print(os.listdir(raw_dir))
    test_df = pd.read_pickle(test_df_path)
    print(
        test_df[
            ["F-Electrostatics", "F-Exchange", "F-Induction", "F-Dispersion", "F-Total"]
        ].head()
    )

    # Create dataset from first 5 rows
    dataset = ap3_fused_fsapt_module_dataset_lmdb(
        root=temp_dir,
        split="test",
        spec_type=5,
        r_cut=5.0,
        r_cut_im=8.0,
        force_reprocess=True,
        random_seed=None,
    )
    print(dataset)

    try:
        # Test dataset length
        assert len(dataset) == 308, f"Expected 159 data points, got {len(dataset)}"
        print(f"Dataset length: {len(dataset)}")

        # Test getting a single data point
        data = dataset[0]
        print(f"\nFirst data point:")
        print(f"  ZA shape: {data.ZA.shape}")
        print(f"  RA shape: {data.RA.shape}")
        print(f"  ZB shape: {data.ZB.shape}")
        print(f"  RB shape: {data.RB.shape}")
        print(f"  y (FSAPT labels) shape: {data.y.shape}")
        print(f"  y values: {data.y}")
        print(f"  frag1_ind: {data.frag1_ind}")
        print(f"  frag2_ind: {data.frag2_ind}")

        # Test that labels match expected FSAPT energies
        expected_labels = torch.tensor(
            [
                test_df.iloc[0]["F-Electrostatics"],
                test_df.iloc[0]["F-Exchange"],
                test_df.iloc[0]["F-Induction"],
                test_df.iloc[0]["F-Dispersion"],
                test_df.iloc[0]["F-Total"],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(data.y, expected_labels), (
            "Labels don't match expected values"
            f"\nExpected: {expected_labels}\nGot: {data.y}"
        )
        print(f"\nLabels match expected FSAPT energies!")

        # Test batch collation
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=ap3_fused_fsapt_collate_update,
        )

        batch = next(iter(dataloader))
        print(f"\nBatched data:")
        print(f"  Batch y shape: {batch.y.shape}")
        print(f"  Batch ZA shape: {batch.ZA.shape}")
        print(f"  Batch RA shape: {batch.RA.shape}")
        print(f"  Number of fragment 1 indices: {len(batch.frag1_ind)}")
        print(f"  Number of fragment 2 indices: {len(batch.frag2_ind)}")

        # Verify batch size
        assert batch.y.shape[0] == 2, f"Expected batch size 2, got {batch.y.shape[0]}"
        assert len(batch.frag1_ind) == 2, "Expected 2 fragment 1 index tensors"
        assert len(batch.frag2_ind) == 2, "Expected 2 fragment 2 index tensors"

        print("\nAll tests passed!")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


@pytest.mark.skip("incomplete functionality")
def test_ap3_fused_fsapt_energies():
    """Test training AP3 fused model on FSAPT fragment energy data"""
    df = pd.read_pickle(f"{data_path}/raw/fsapt_test_simple.pkl")
    # drop row that has frag2 == Tert-B because it has no indices
    df = df[df["Frag2"] != "T-Butyl_B"]
    df = df[df["Frag1"] != "All"]
    df = df[df["Frag2"] != "All"]
    pp(df.columns.tolist())
    df.to_pickle(f"{data_path}/raw/fsapt_test_simple_2.pkl")
    df.to_pickle(f"{data_path}/raw/fsapt_train_simple_2.pkl")
    print(
        df[
            [
                "Frag1",
                "Frag2",
                "F-Total",
                "F-Electrostatics",
                "F-Exchange",
                "F-Induction",
                "F-EDispersion",
            ]
        ].head()
    )
    print(df["Frag1_indices"])
    print(df["Frag2_indices"])

    print(df.columns.tolist())
    fAs = {}
    fBs = {}
    for f1, inds in zip(df["Frag1"], df["Frag1_indices"]):
        fAs[f1] = inds
    for f2, inds in zip(df["Frag2"], df["Frag2_indices"]):
        fBs[f2] = inds
    pred_IEs, pairwise_energies, df_out = (
        apnet_pt.pretrained_models.apnet3_model_predict_pairs(
            [df["qcel_molecule"].iloc[0]],
            fAs=[fAs],
            fBs=[fBs],
            compile=False,
        )
    )
    pp(fAs)
    pp(fBs)
    print(df_out)
    """
FISAPT0/aug-cc-pVDZ
       Frag1      Frag2   F-Total  F-Electrostatics  F-Exchange  F-Induction  F-Dispersion
0  Methyl1_A        All  0.408609          0.739977    0.071830    -0.011515     -0.391684
1  Methyl2_A        All -0.790990         -2.031429    4.245923    -0.501495     -2.503989
2        All  Peptide_B -0.330418         -0.149510    0.039503    -0.126168     -0.094244
3        All  T-Butyl_B -0.051963         -1.141942    4.278250    -0.386842     -2.801430
4        All        All -0.382381         -1.291452    4.317753    -0.513009     -2.895673
AP3-fused
                 fA-fB     total      elst      exch      indu      disp
0        Methyl1_A-All -0.445212 -0.209964  0.110707 -0.027262 -0.318692
3        Methyl2_A-All  0.151453 -1.095993  4.227958 -0.393157 -2.587355
7        All-Peptide_B -0.014877  0.045080  0.008084 -0.003050 -0.064991
8        All-T-Butyl_B -0.278882 -1.351036  4.330580 -0.417369 -2.841057
6              All-All -0.293759 -1.305957  4.338664 -0.420419 -2.906048
{'All': [1, 2, 7, 8, 3, 4, 5, 6],
 'Methyl1_A': [1, 2, 7, 8],
 'Methyl2_A': [3, 4, 5, 6]}
{'All': [9, 10, 11, 16, 26, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
 'Peptide_B': [9, 10, 11, 16, 26],
 'T-Butyl_B': [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25]}
    """
    return


@pytest.mark.skip("incomplete functionality")
def test_ap3_fused_fsapt_energies_mocking_test():
    """Test training AP3 fused model on FSAPT fragment energy data"""
    Na_methyl_sr = qcel.models.Molecule.from_data("""1 1
Na 0.00000000 0.00000000 0.00000000
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
no_com
    """)
    Na_methyl_lr = qcel.models.Molecule.from_data("""1 1
Na -10.00000000 0.00000000 0.00000000
--
0 1
C 6.44536662 -0.26509169 -0.00000000
H 7.53536662 -0.26509169 -0.00000000
H 6.08203329 0.57399070 0.59332085
H 6.08203329 -0.17080196 -1.02332709
H 6.08203329 -1.19846381 0.43000624
symmetry c1
no_reorient
no_com
    """)
    fAs = {
        "He": [1],
    }
    fBs = {
        "methyl": [2, 3, 4, 5, 6],
        "CH": [2, 3],
    }
    dimers = [Na_methyl_sr, Na_methyl_lr]
    pred_IEs, pairwise_energies, df_out = (
        apnet_pt.pretrained_models.apnet3_model_predict_pairs(
            dimers,
            fAs=[fAs for i in range(2)],
            fBs=[fBs for i in range(2)],
            compile=False,
            print_results=True,
        )
    )
    pp(fAs)
    pp(fBs)
    print(df_out)
    df_out = df_out.rename(
        columns={
            "total": "F-Total",
            "elst": "F-Electrostatics",
            "exch": "F-Exchange",
            "indu": "F-Induction",
            "disp": "F-Dispersion",
        }
    )
    df_out["frag1"] = df_out["fA-fB"].apply(lambda x: x.split("-")[0])
    df_out["frag2"] = df_out["fA-fB"].apply(lambda x: x.split("-")[1])
    print(df_out[["frag1", "frag2"]])
    df_out["Frag1_indices"] = df_out["frag1"].apply(lambda x: fAs[x])
    df_out["Frag2_indices"] = df_out["frag2"].apply(lambda x: fBs[x])
    print(df_out)
    df_out["qcel_molecule"] = [Na_methyl_sr, Na_methyl_sr, Na_methyl_lr, Na_methyl_lr]
    pp(df_out.columns.tolist())
    df_out.to_pickle(f"{data_path}/raw/fsapt_test_simple_2.pkl")
    df_out.to_pickle(f"{data_path}/raw/fsapt_train_simple_2.pkl")
    return


@pytest.mark.skip("incomplete functionality")
def test_ap3_fused_fsapt_training_mock():
    """
    Test training AP3 fused model on FSAPT fragment energy data on extremely
    simple mock example of Na-Methyl. Checks that eval correctly sums
    fragments.
    """
    temp_dir = tempfile.mkdtemp()
    test_df_path = f"{data_path}/raw/fsapt_test_simple_2.pkl"
    train_df_path = f"{data_path}/raw/fsapt_train_simple_2.pkl"
    # mkdir raw under temp_dir and copy test_df there
    raw_dir = os.path.join(temp_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Copying test dataframe to {raw_dir}")
    shutil.copy(test_df_path, raw_dir)
    shutil.copy(train_df_path, raw_dir)
    try:
        # Initialize atom models (required for AP3)
        atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_path,
            pre_trained_model_path=at_hf_vw_path,
        )
        atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=at_elst_path,
        )

        # Initialize AP3 model for FSAPT training

        ap3 = APNet3_AtomType_Model(
            # ds_root=temp_dir,
            ds_root=data_path,
            atom_type_model=atom_type_hf_vw_model.model,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            am_dimer_param_model=atom_type_elst_model,
            use_precomputed_classical=False,
            ignore_database_null=False,
            ds_spec_type=7,
            ds_type="fsapt_energies",  # Important: set ds_type for FSAPT
            pre_trained_model_path=ap3_path,
            ds_batch_size=2,
        )

        print("\nStarting FSAPT training...")

        # Train for a few epochs
        ap3.train(
            n_epochs=0,
            lr=5e-5,
            skip_compile=True,  # Skip compilation for faster testing
        )

        print("\nFSAPT training test completed successfully!")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    return


@pytest.mark.skip("incomplete functionality")
def test_ap3_fused_fsapt_training():
    """
    Test training AP3 fused model on FSAPT fragment energy data on
    simple system dataset to ensure training loop works.
    """
    temp_dir = tempfile.mkdtemp()
    test_df_path = f"{data_path}/raw/fsapt_test_simple.pkl"
    train_df_path = f"{data_path}/raw/fsapt_train_simple.pkl"
    # mkdir raw under temp_dir and copy test_df there
    raw_dir = os.path.join(temp_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Copying test dataframe to {raw_dir}")
    shutil.copy(test_df_path, raw_dir)
    shutil.copy(train_df_path, raw_dir)
    try:
        # Initialize atom models (required for AP3)
        atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_path,
            pre_trained_model_path=at_hf_vw_path,
        )
        atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=at_elst_path,
        )

        # Initialize AP3 model for FSAPT training

        ap3 = APNet3_AtomType_Model(
            # ds_root=temp_dir,
            ds_root=data_path,
            atom_type_model=atom_type_hf_vw_model.model,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            am_dimer_param_model=atom_type_elst_model,
            use_precomputed_classical=False,
            ignore_database_null=False,
            ds_spec_type=6,
            ds_type="fsapt_energies",  # Important: set ds_type for FSAPT
            pre_trained_model_path=ap3_path,
            ds_batch_size=16,
        )

        print("\nStarting FSAPT training...")

        # Train for a few epochs
        ap3.train(
            n_epochs=100,
            lr=5e-3,
            skip_compile=True,  # Skip compilation for faster testing
        )

        print("\nFSAPT training test completed successfully!")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    return


@pytest.mark.skip("data analysis")
def test_ap2_ap3_fused_fsapt_energies():
    """Test training AP3 fused model on FSAPT fragment energy data"""
    df = pd.read_pickle(f"{data_path}/raw/fsapt_full_data.pkl")
    df["fA-fB"] = df.apply(lambda r: f"{r['Frag1']}-{r['Frag2']}", axis=1)
    print(
        df[
            [
                "Frag1",
                "Frag2",
                "F-Total",
                "F-Electrostatics",
                "F-Exchange",
                "F-Induction",
                "F-EDispersion",
                "qcel_molecule",
            ]
        ]
    )
    fAs = []
    fBs = []
    for f1, inds in zip(df["Frag1"], df["Frag1_indices"]):
        fAs.append({f1: inds})
    for f2, inds in zip(df["Frag2"], df["Frag2_indices"]):
        fBs.append({f2: inds})
    print(fAs[0])
    print(fBs[0])
    pred_IEs, pairwise_energies, df_out_ap2 = (
        apnet_pt.pretrained_models.apnet2_model_predict_pairs(
            df["qcel_molecule"].tolist(),
            fAs=fAs,
            fBs=fBs,
            ap2_fused=True,
            compile=False,
        )
    )
    print(df_out_ap2)
    # df_out_ap2.to_pickle("fsapt_ap2_fused_predictions.pkl")
    df["total_ap2"] = df_out_ap2["total"]
    df["elst_ap2"] = df_out_ap2["elst"]
    df["exch_ap2"] = df_out_ap2["exch"]
    df["indu_ap2"] = df_out_ap2["indu"]
    df["disp_ap2"] = df_out_ap2["disp"]
    df["fA-fB_ap2"] = df_out_ap2["fA-fB"]
    pred_IEs, pairwise_energies, df_out_ap3 = (
        apnet_pt.pretrained_models.apnet3_model_predict_pairs(
            df["qcel_molecule"].tolist(),
            fAs=fAs,
            fBs=fBs,
            compile=False,
        )
    )
    print(df_out_ap3)
    # df_out_ap3.to_pickle("fsapt_ap3_fused_predictions.pkl")
    df["total_ap3"] = df_out_ap3["total"]
    df["elst_ap3"] = df_out_ap3["elst"]
    df["exch_ap3"] = df_out_ap3["exch"]
    df["indu_ap3"] = df_out_ap3["indu"]
    df["disp_ap3"] = df_out_ap3["disp"]
    df["fA-fB_ap3"] = df_out_ap3["fA-fB"]
    print(df)
    pp(df.columns.tolist())
    df["base_id"] = df["id"].str.replace(r"-[a-z][a-z]", "", regex=True)
    unique_ids = df["base_id"].unique()
    print(f"Unique base IDs: {unique_ids}")
    df.to_pickle("fsapt_ap2_ap3_fused_comparison.pkl")
    # Since dataframes are identical except for the energy predictions, just add ap3 energy cals to ap2
    return


if __name__ == "__main__":
    # Application tests
    # test_ap2_ap3_fused_fsapt_energies()

    # test_ap3_fused_fsapt()
    # test_ap3_fused_fsapt_energies()
    # test_ap3_fused_fsapt_energies_mocking_test()
    # test_ap3_fused_fsapt_training_mock()
    test_ap3_fused_fsapt_training()
