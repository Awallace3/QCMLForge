from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model
import apnet_pt
from apnet_pt.AtomPairwiseModels.dapnet2 import dAPNet2Model, APNet2_dAPNet2Model
from apnet_pt import AtomPairwiseModels
from apnet_pt import atomic_datasets
from apnet_pt import pairwise_datasets
from apnet_pt import AtomModels
from apnet_pt.pairwise_datasets import (
    apnet2_module_dataset,
    apnet2_collate_update,
    apnet2_collate_update_prebatched,
    APNet2_DataLoader,
    apnet3_module_dataset,
    apnet3_collate_update,
    apnet3_collate_update_prebatched,
)
from apnet_pt.pt_datasets.dapnet_ds import (
    dapnet2_module_dataset,
    dapnet2_module_dataset_apnetStored,
    dapnet2_collate_update_no_target,
)
import os
import numpy as np
import pytest
from glob import glob
import qcelemental as qcel
import torch
import pandas as pd
from pprint import pprint as pp


torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../src/apnet_pt/models/am_ensemble/am_0.pt"
am_hf_path = f"{current_file_path}/../src/apnet_pt/models/am_hf_ensemble/am_0.pt"


def test_ap3_train():
    df = pd.read_pickle(current_file_path + "/dataset_data/elst_damping_test.pkl")
    qcel_molecules = df["qcel_molecule"].to_list()
    for i in qcel_molecules:
        print(i.to_string("psi4"))
    energy_labels = (
        df[["SAPT0 ELST", "SAPT0 EXCH", "SAPT0 IND", "SAPT0 DISP"]].values
        * qcel.constants.hartree2kcalmol
    )
    print(energy_labels)
    atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=current_file_path
        + "/../models/am_ensemble/am_0.pt",
        pre_trained_model_path=current_file_path
        + "/../models/ap_atomTypeParamModel/am_h+1_0.pt",
    )
    atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path="/home/amwalla3/projects/qcmlforge_tests/water_elst/models/ap_dimerParamModel-elst_damping_0.pt",
    )
    # print(atom_type_elst_model.atom_model)
    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=8,
        ds_in_memory=False,
    )
    ap3.train(
        n_epochs=50,
        skip_compile=True,
        transfer_learning=False,
        lr=5e-4,
        dataloader_num_workers=4,
    )


if __name__ == "__main__":
    test_ap3_train()
