import os
import shutil
import tempfile
from glob import glob

import numpy as np
import pandas as pd
import pytest
import qcelemental as qcel
import torch

import apnet_pt
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_Model
from apnet_pt.pt_datasets.ap3_fused_ds import (
    ap3_fused_module_dataset,
    ap3_fused_module_dataset_lmdb,
)

torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"

am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_h+1_3.pt"
at_elst_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
ap3_path = f"{current_file_path}/test_models/ap3_ensemble_0/ap3_.pt"

mol_cliff_water_close = qcel.models.Molecule.from_data("""
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
lr_water_dimer = qcel.models.Molecule.from_data("""
0 1
--
0 1
O                    -1.326958230000    -0.105938530000     0.018788150000
H                    -1.931665240000     1.600174320000    -0.021710520000
H                     0.486644280000     0.079598090000     0.009862480000
--
0 1
O                     8.088671270000     0.019951580000    -0.007942850000
H                     8.800382980000    -0.808466680000     1.439822410000
H                     8.792148880000    -0.879960520000    -1.416549430000
units bohr
""")
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

mol_element = qcel.models.Molecule.from_data("""
1 1
11   -0.902196054   -0.106060256   0.009942262
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
""")

mol3 = qcel.models.Molecule.from_data(
    """
    1 1
    C       0.0545060001    -0.1631290019   -1.1141539812
    C       -0.9692260027   -1.0918780565   0.6940879822
    C       0.3839910030    0.5769280195    -0.0021170001
    C       1.3586950302    1.7358809710    0.0758149996
    N       -0.1661809981   -0.0093130004   1.0584640503
    N       -0.8175240159   -1.0993789434   -0.7090409994
    H       0.3965460062    -0.1201139987   -2.1653149128
    H       -1.5147459507   -1.6961929798   1.3000769615
    H       0.7564010024    2.6179349422    0.4376020133
    H       2.2080008984    1.5715960264    0.7005280256
    H       1.7567750216    2.0432629585    -0.9004560113
    H       -0.1571149975   0.2784340084    1.9974440336
    H       -1.2523859739   -1.9090379477   -1.2904200554
    --
    -1 1
    C       -5.6793351173   2.6897408962    7.4496979713
    C       -4.5188479424   3.5724110603    6.9706201553
    N       -6.1935510635   1.6698499918    6.8358440399
    N       -6.2523350716   2.9488639832    8.6100416183
    N       -7.1709971428   1.1798499823    7.7206158638
    N       -7.2111191750   1.9820170403    8.7515516281
    H       -4.9275932312   4.5184249878    6.4953727722
    H       -3.8300020695   3.8421258926    7.6719899178
    H       -4.1228170395   3.0444390774    6.1303391457
    units angstrom
                """
)


def set_weights_to_value(model, value=0.9):
    """Sets all weights and biases in the model to a specific value."""
    with torch.no_grad():  # Disable gradient tracking
        for param in model.parameters():
            param.fill_(value)  # Set all elements to the given value
    return


def test_ap3_fused_train_qcel_molecules_in_memory():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 6
    qcel_molecules = [mol_cliff_water_close] * 4
    energy_labels = [
        np.array(
            [
                -10.779292828139122,
                11.390991215401051,
                -3.414543432719425,
                -2.436025699701581,
            ]
        )
        for _ in range(len(qcel_molecules))
    ]
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
    ds = ap3_fused_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atomic_batch_size=atomic_batch_size,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        num_devices=1,
        skip_processed=True,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
        random_seed=None,
    )
    # print(atom_type_elst_model.atom_model)
    print(atom_type_elst_model.dimer_model.AtomTypeParam)
    ap3 = APNet3D3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        am_dimer_param_model=atom_type_elst_model,
        use_precomputed_classical=False,
    )
    print(ap3)
    ap3.train(
        ds,
        n_epochs=5,
        skip_compile=True,
        transfer_learning=False,
        lr=0.0005,
    )
    # This also tests to make sure only best model is returned
    v_0 = ap3.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    ap3.train(
        ds,
        n_epochs=1,
        skip_compile=True,
        transfer_learning=False,
        lr=0.05,
    )
    v = ap3.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    print(v_0, v)
    assert np.allclose(v_0, v, atol=1e-6)


@pytest.mark.xdist_group(name="io_operations")
def test_ap3_fused_train_qcel_molecules_in_memory_precompute():
    try:
        for i in glob(f"{data_path}/processed/dimer_ap3*"):
            os.remove(i)
    except:
        pass
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 6
    qcel_molecules = [mol_cliff_water_close] * 4
    energy_labels = [
        np.array(
            [
                -10.779292828139122,
                11.390991215401051,
                -3.414543432719425,
                -2.436025699701581,
            ]
        )
        for _ in range(len(qcel_molecules))
    ]
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
    ds = ap3_fused_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atomic_batch_size=atomic_batch_size,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
        random_seed=None,
    )
    # print(atom_type_elst_model.atom_model)
    print(atom_type_elst_model.dimer_model.AtomTypeParam)
    ap3 = APNet3D3_AtomType_Model(
        dataset=ds,
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        am_dimer_param_model=atom_type_elst_model,
        use_precomputed_classical=False,
        ds_class_type="pt",
    )
    print(ap3)
    ap3.train(
        ds,
        n_epochs=5,
        skip_compile=True,
        transfer_learning=False,
        lr=0.0005,
    )
    # This also tests to make sure only best model is returned
    v_0 = ap3.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    ap3.train(
        ds,
        n_epochs=1,
        skip_compile=True,
        transfer_learning=False,
        lr=0.05,
    )
    v = ap3.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    print(v_0, v)
    assert np.allclose(v_0, v, atol=1e-6)
    for i in glob(f"{data_path}/processed/dimer_ap3*"):
        os.remove(i)


@pytest.mark.xdist_group(name="io_operations")
def test_classical_ap3():
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    print(r["SAPT0 ELST ENERGY adz"])
    print(r["SAPT0 IND ENERGY adz"] * 627.5094740631)
    atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=data_path,
        use_GPU=False,
        n_neuron=64,
        n_params=1,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )
    ap3 = APNet3D3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        am_dimer_param_model=atom_type_elst_model,
    )
    monA_props, monB_props = atom_type_elst_model.predict_qcel_mols_monomer_props(
        [mol], model_type="model", am_type="ap3"
    )

    dimer_batch = apnet_pt.pt_datasets.ap3_fused_ds.ap3_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )

    dimer_batch.qA = torch.tensor(monA_props[0][0], dtype=torch.float32)
    dimer_batch.qB = torch.tensor(monB_props[0][0], dtype=torch.float32)
    dimer_batch.muA = torch.tensor(monA_props[0][1], dtype=torch.float32)
    dimer_batch.muB = torch.tensor(monB_props[0][1], dtype=torch.float32)
    dimer_batch.quadA = torch.tensor(monA_props[0][2], dtype=torch.float32)
    dimer_batch.quadB = torch.tensor(monB_props[0][2], dtype=torch.float32)
    dimer_batch.hlistA = torch.tensor(monA_props[0][3], dtype=torch.float32)
    dimer_batch.hlistB = torch.tensor(monB_props[0][3], dtype=torch.float32)
    dimer_batch.Ka = torch.tensor(monA_props[0][-1], dtype=torch.float32)
    dimer_batch.Kb = torch.tensor(monB_props[0][-1], dtype=torch.float32)
    dimer_batch.vw_A = torch.tensor(monA_props[0][-2], dtype=torch.float32)
    dimer_batch.vw_B = torch.tensor(monB_props[0][-2], dtype=torch.float32)
    dimer_batch.hfvr_A = torch.tensor(monA_props[0][-3], dtype=torch.float32)
    dimer_batch.hfvr_B = torch.tensor(monB_props[0][-3], dtype=torch.float32)
    print(f"dimer_batch.Ka: {dimer_batch.Ka}")
    print(f"dimer_batch.Kb: {dimer_batch.Kb}")
    print(f"dimer_batch.vw_A: {dimer_batch.vw_A}")
    print(f"dimer_batch.vw_B: {dimer_batch.vw_B}")
    print(f"dimer_batch.hfvr_A: {dimer_batch.hfvr_A}")
    print(f"dimer_batch.hfvr_B: {dimer_batch.hfvr_B}")

    torch_elst = apnet_pt.AtomPairwiseModels.mtp_mtp.mtp_elst_damping(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA_0=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        Ka=dimer_batch.Ka,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB_0=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        Kb=dimer_batch.Kb,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
    )
    # ref = -10.753819
    # print(f"Torch elst = {torch.sum(torch_elst):.6f} kcal/mol")
    # assert np.allclose(torch.sum(torch_elst).item(), ref, atol=1e-4)

    torch_ind = apnet_pt.AtomPairwiseModels.mtp_mtp.induced_dipole_induction_optimized_no_correction(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        hirshfeld_volume_ratio_A=dimer_batch.hfvr_A,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        hirshfeld_volume_ratio_B=dimer_batch.hfvr_B,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
        e_AA_source=dimer_batch.e_AA_source,
        e_BB_source=dimer_batch.e_BB_source,
        e_AA_target=dimer_batch.e_AA_target,
        e_BB_target=dimer_batch.e_BB_target,
    )
    # print(f"Torch ind = {torch.sum(torch_ind):.6f} kcal/mol")
    # ref = -1.264973
    # assert np.allclose(torch.sum(torch_ind).item(), ref, atol=1e-4)

    pred, pair_elst, pair_ind, pair_disp = ap3.predict_qcel_mols(
        [mol], batch_size=1, return_classical_pairs=True
    )
    print(f"AP3 elst = {pred[0][0]:.6f} kcal/mol")
    print(f"{torch_elst = }")
    print(f"{pair_elst  = }")
    assert np.allclose(torch_elst.cpu().numpy(), pair_elst[0].flatten(), atol=1e-4)
    print(f"AP3 ind = {pred[0][2]:.6f} kcal/mol")
    print(f"{torch_ind = }")
    print(f"{pair_ind  = }")
    assert np.allclose(torch_ind.cpu().numpy(), pair_ind[0].flatten(), atol=1e-4)
    pred, pair_elst, pair_ind, pair_disp = ap3.predict_qcel_mols(
        [mol, mol_element], batch_size=1, return_classical_pairs=True
    )
    assert np.allclose(torch_elst.cpu().numpy(), pair_elst[0].flatten(), atol=1e-4)
    assert np.allclose(torch_ind.cpu().numpy(), pair_ind[0].flatten(), atol=1e-4)
    pred, pair_ind, pair_ind, pair_disp = ap3.predict_qcel_mols(
        [mol, mol_element], batch_size=1, return_classical_pairs=True
    )
    assert np.allclose(torch_ind.cpu().numpy(), pair_ind[0].flatten(), atol=1e-4)
    return


def test_classical_ap3_long_range():
    mol = lr_water_dimer
    atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=data_path,
        use_GPU=False,
        n_neuron=64,
        n_params=1,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )
    ap3_d3 = APNet3D3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        am_dimer_param_model=atom_type_elst_model,
    )
    monA_props, monB_props = atom_type_elst_model.predict_qcel_mols_monomer_props(
        [mol], model_type="model", am_type="ap3"
    )
    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )
    dimer_batch.qA = torch.tensor(monA_props[0][0], dtype=torch.float32)
    dimer_batch.qB = torch.tensor(monB_props[0][0], dtype=torch.float32)
    dimer_batch.muA = torch.tensor(monA_props[0][1], dtype=torch.float32)
    dimer_batch.muB = torch.tensor(monB_props[0][1], dtype=torch.float32)
    dimer_batch.quadA = torch.tensor(monA_props[0][2], dtype=torch.float32)
    dimer_batch.quadB = torch.tensor(monB_props[0][2], dtype=torch.float32)

    dimer_batch.Ka = torch.tensor(monA_props[0][-1], dtype=torch.float32)
    dimer_batch.Kb = torch.tensor(monB_props[0][-1], dtype=torch.float32)
    dimer_batch.vw_A = torch.tensor(monA_props[0][-2], dtype=torch.float32)
    dimer_batch.vw_B = torch.tensor(monB_props[0][-2], dtype=torch.float32)
    dimer_batch.hfvr_A = torch.tensor(monA_props[0][-3], dtype=torch.float32)
    dimer_batch.hfvr_B = torch.tensor(monB_props[0][-3], dtype=torch.float32)
    print(f"dimer_batch.Ka: {dimer_batch.Ka}")
    print(f"dimer_batch.Kb: {dimer_batch.Kb}")
    print(f"dimer_batch.vw_A: {dimer_batch.vw_A}")
    print(f"dimer_batch.vw_B: {dimer_batch.vw_B}")
    print(f"dimer_batch.hfvr_A: {dimer_batch.hfvr_A}")
    print(f"dimer_batch.hfvr_B: {dimer_batch.hfvr_B}")

    torch_elst = apnet_pt.AtomPairwiseModels.mtp_mtp.mtp_elst_damping(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA_0=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        Ka=dimer_batch.Ka,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB_0=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        Kb=dimer_batch.Kb,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
    )
    # ref = -0.857894
    # print(f"Torch elst = {torch.sum(torch_elst):.6f} kcal/mol")
    # assert np.allclose(torch.sum(torch_elst).item(), ref, atol=1e-4)

    torch_ind = apnet_pt.AtomPairwiseModels.mtp_mtp.induced_dipole_induction_optimized_no_correction(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        hirshfeld_volume_ratio_A=dimer_batch.hfvr_A,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        hirshfeld_volume_ratio_B=dimer_batch.hfvr_B,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
        e_AA_source=dimer_batch.e_AA_source,
        e_BB_source=dimer_batch.e_BB_source,
        e_AA_target=dimer_batch.e_AA_target,
        e_BB_target=dimer_batch.e_BB_target,
    )
    print(f"Torch ind = {torch.sum(torch_ind):.6f} kcal/mol")
    # ref = -0.016318
    # assert np.allclose(torch.sum(torch_ind).item(), ref, atol=1e-4)

    pred, pair_elst, pair_ind, pair_disp = ap3_d3.predict_qcel_mols(
        [mol], batch_size=1, return_classical_pairs=True
    )
    print(f"AP3 elst = {pred[0][0]:.6f} kcal/mol")
    print(f"{torch_elst = }")
    print(f"{pair_elst  = }")
    assert np.allclose(torch_elst.cpu().numpy(), pair_elst[0].flatten(), atol=1e-4)
    print(f"AP3 ind = {pred[0][2]:.6f} kcal/mol")
    print(f"{torch_ind = }")
    print(f"{pair_ind  = }")
    assert np.allclose(torch_ind.cpu().numpy(), pair_ind[0].flatten(), atol=1e-4)
    pred, pair_elst, pair_ind, pair_disp = ap3_d3.predict_qcel_mols(
        [mol, mol_element], batch_size=1, return_classical_pairs=True
    )
    assert np.allclose(torch_elst.cpu().numpy(), pair_elst[0].flatten(), atol=1e-4)
    assert np.allclose(torch_ind.cpu().numpy(), pair_ind[0].flatten(), atol=1e-4)
    pred, pair_ind, pair_ind, pair_disp = ap3_d3.predict_qcel_mols(
        [mol, mol_element], batch_size=1, return_classical_pairs=True
    )
    assert np.allclose(torch_ind.cpu().numpy(), pair_ind[0].flatten(), atol=1e-4)
    return


def test_ap3_fused_lmdb_dataset():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 6
    qcel_molecules = [mol_cliff_water_close] * 4
    energy_labels = [
        np.array(
            [
                -10.779292828139122,
                11.390991215401051,
                -3.414543432719425,
                -2.436025699701581,
            ]
        )
        for _ in range(len(qcel_molecules))
    ]
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

    temp_dir = tempfile.mkdtemp()

    try:
        ds_lmdb = ap3_fused_module_dataset_lmdb(
            root=temp_dir,
            r_cut=5.0,
            r_cut_im=8.0,
            spec_type=None,
            qcel_molecules=qcel_molecules,
            energy_labels=energy_labels,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            cache_size=1000,
            lmdb_map_size=1024**3,
            print_level=2,
            atomic_batch_size=atomic_batch_size,
            datapoint_storage_n_objects=datapoint_storage_n_objects,
            batch_size=batch_size,
        )

        assert len(ds_lmdb) == len(qcel_molecules)

        item_0 = ds_lmdb[0]
        assert item_0 is not None
        assert hasattr(item_0, "y")
        assert hasattr(item_0, "RA")
        assert hasattr(item_0, "RB")

        del ds_lmdb

        ds_lmdb_reload = ap3_fused_module_dataset_lmdb(
            root=temp_dir,
            r_cut=5.0,
            r_cut_im=8.0,
            spec_type=None,
            cache_size=1000,
            print_level=2,
            lmdb_readonly=True,
            atomic_batch_size=atomic_batch_size,
            datapoint_storage_n_objects=datapoint_storage_n_objects,
            batch_size=batch_size,
        )

        assert len(ds_lmdb_reload) == len(qcel_molecules)

        item_0_reload = ds_lmdb_reload[0]
        assert torch.allclose(item_0.y, item_0_reload.y, atol=1e-6)
        assert torch.allclose(item_0.RA, item_0_reload.RA, atol=1e-6)
        assert torch.allclose(item_0.RB, item_0_reload.RB, atol=1e-6)

        item_1 = ds_lmdb_reload[1]
        assert item_1 is not None

        ds_orig = ap3_fused_module_dataset(
            root=data_path,
            r_cut=5.0,
            r_cut_im=8.0,
            spec_type=None,
            max_size=None,
            force_reprocess=True,
            atomic_batch_size=atomic_batch_size,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            datapoint_storage_n_objects=datapoint_storage_n_objects,
            batch_size=batch_size,
            num_devices=1,
            skip_processed=True,
            skip_compile=True,
            print_level=2,
            qcel_molecules=qcel_molecules,
            energy_labels=energy_labels,
            in_memory=True,
            random_seed=None,
        )

        item_0_orig = ds_orig[0]
        assert torch.allclose(item_0_reload.y, item_0_orig.y, atol=1e-6)

        print("All LMDB dataset tests passed!")

    finally:
        shutil.rmtree(temp_dir)


def test_ap3_fused_train_qcel_molecules_in_memory_precompute_lmdb():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 6
    qcel_molecules = [mol_cliff_water_close] * 4
    energy_labels = [
        np.array(
            [
                -10.779292828139122,
                11.390991215401051,
                -3.414543432719425,
                -2.436025699701581,
            ]
        )
        for _ in range(len(qcel_molecules))
    ]
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
    ds = ap3_fused_module_dataset_lmdb(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atomic_batch_size=atomic_batch_size,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
        random_seed=None,
    )
    # print(atom_type_elst_model.atom_model)
    print(atom_type_elst_model.dimer_model.AtomTypeParam)
    ap3 = APNet3D3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        am_dimer_param_model=atom_type_elst_model,
        use_precomputed_classical=False,
    )
    print(ap3)
    ap3.train(
        ds,
        n_epochs=5,
        skip_compile=True,
        transfer_learning=False,
        lr=0.0005,
    )
    # This also tests to make sure only best model is returned
    v_0 = ap3.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    ap3.train(
        ds,
        n_epochs=1,
        skip_compile=True,
        transfer_learning=False,
        lr=0.05,
    )
    v = ap3.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    print(v_0, v)
    assert np.allclose(v_0, v, atol=1e-6)
    # need to cleanup lmdb_ap3_fused_spec_None in data_path/processed
    shutil.rmtree(f"{data_path}/processed/lmdb_ap3_fused_spec_None")


def test_classical_ap3_dispersion():
    menh2_water_dimer = qcel.models.Molecule.from_data("""
    0 1
    --
    0 1
    N                    -1.008100800000    -0.528355160000     0.202192680000
    H                    -1.188923790000    -2.359180490000     0.723479130000
    H                    -2.121413420000    -0.313995840000    -1.337480310000
    C                    -1.921680310000     1.112077560000     2.243810640000
    H                    -1.724865800000     3.071847600000     1.662054110000
    H                    -3.882890690000     0.784391550000     2.793966870000
    H                    -0.727588730000     0.848110780000     3.893996460000
    --
    0 1
    O                     4.023106790000     1.759569490000     0.398270440000
    H                     2.478222790000     0.821750410000     0.071058880000
    H                     5.122741160000     1.271079320000    -0.954293650000
    units bohr
    no_com
    no_reorient
    """)

    water_water_dimer = qcel.models.Molecule.from_data("""
    0 1
    --
    0 1
    O                    -1.326958230000    -0.105938530000     0.018788150000
    H                    -1.931665240000     1.600174320000    -0.021710520000
    H                     0.486644280000     0.079598090000     0.009862480000
    --
    0 1
    O                     4.287563290000     0.049775580000     0.000960040000
    H                     4.999275000000    -0.778642690000     1.448725300000
    H                     4.991040900000    -0.850136520000    -1.407646550000
    units bohr
    no_com
    no_reorient
    """)

    benzene_pyridine_dimer = qcel.models.Molecule.from_data("""
    0 1
    --
    0 1
    C                     1.547207570000     1.633049050000     0.355809190000
    H                     2.770553190000     3.244031740000     0.651429050000
    C                     2.587029620000    -0.737983290000    -0.126041300000
    H                     4.616669650000    -0.967278880000    -0.208955410000
    C                     1.009829210000    -2.812844480000    -0.513793930000
    H                     1.815740040000    -4.651682210000    -0.898578930000
    C                    -1.604595960000    -2.514294490000    -0.415544040000
    H                    -2.829051200000    -4.123118380000    -0.717251620000
    C                    -2.644644230000    -0.143676260000     0.076409500000
    H                    -4.672700600000     0.084863400000     0.176466510000
    C                    -1.068247630000     1.930172610000     0.457841350000
    H                    -1.875660280000     3.767473860000     0.843305720000
    --
    0 1
    N                    -4.484042050000     0.282741320000     6.476783420000
    C                    -3.322076040000     2.470416420000     7.006028040000
    H                    -4.545870340000     4.082676810000     7.313131390000
    C                    -0.710068430000     2.736580070000     7.176426190000
    H                     0.106007450000     4.557614870000     7.613361660000
    C                     0.803510340000     0.625795100000     6.771620560000
    H                     2.839820940000     0.758802100000     6.875242770000
    C                    -0.369162810000    -1.658231910000     6.216684390000
    H                     0.718893330000    -3.350711980000     5.868833170000
    C                    -2.996853970000    -1.730113090000     6.093002910000
    H                    -3.958446630000    -3.484635690000     5.660552930000
    units bohr
    no_com
    no_reorient
    """)

    mols = [
        water_water_dimer,
        menh2_water_dimer,
        benzene_pyridine_dimer,
    ]

    # mols = [
    #     water_water_dimer,
    # ]

    batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut=5.0, dimer_ind=n, r_cut_im=torch.inf
            )
            for n, mol in enumerate(mols)
        ]
    )

    print(dir(batch))
    atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )

    atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        use_GPU=False,
        n_neuron=64,
        n_params=1,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        model_type="AtomTypeParamNN",
        # model_type="AtomTypeParamMPNN",
        # pre_trained_model_path=at_elst_path_mpnn,
        pre_trained_model_path=at_elst_path,
    )

    ap3 = APNet3D3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        am_dimer_param_model=atom_type_elst_model,
        # use_precomputed_classical=False,
    )
    print(f"{batch.dimer_ind = }")
    v = ap3.predict_qcel_mols(
        [
            water_water_dimer,
        ],
        batch_size=1,
        return_classical_pairs=True,
    )
    E_classical, mA, mB = ap3.dimer_prop_model(batch)
    ref_disp = 0
    print(f"{E_classical[:,2]= }")
    torch_disp = E_classical[:, 2]
    # print(f"{torch_disp = }")

    dimer_energies = torch.zeros(
        3,
    )

    dimer_energies.scatter_add_(0, batch.dimer_ind, torch_disp)
    print(f"{dimer_energies = }")
    ap3_disp = dimer_energies.cpu().numpy()
    # Energies are from simple dftd3
    
    simple_dftd3_energies = np.array([-1.6318158037336559, -3.095885350720171, -6.786625297216168])
    print(f"{simple_dftd3_energies = }")
    print(f"{ap3_disp = }")

    return


if __name__ == "__main__":
    # test_classical_ap3()
    # test_classical_ap3_long_range()
    # test_ap3_fused_train_qcel_molecules_in_memory()
    # test_ap3_fused_train_qcel_molecules_in_memory_precompute()
    # test_classical_ap3_induction()
    # test_ap3_fused_lmdb_dataset()
    # test_ap3_fused_train_qcel_molecules_in_memory_precompute_lmdb()
    # test_classical_ap3_dispersion()
    # test_ap3_fused_train_qcel_molecules_in_memory()
    test_classical_ap3_long_range()
