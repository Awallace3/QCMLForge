import pytest
import apnet_pt
import qcelemental as qcel
import os
import pandas as pd
from pprint import pprint
import numpy as np
import torch
from apnet_pt import atomic_datasets
from apnet_pt import constants
import torch


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

current_file_path = os.path.dirname(os.path.abspath(__file__))
data_path = f"{current_file_path}/test_data_path"
h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_h+1_3.pt"
at_elst_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
ap3_path = f"{current_file_path}/test_models/ap3_ensemble_0/ap3_.pt"


def test_elst_multipoles_AP2():
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols(
        [monA, monB, monA.copy(), monB.copy()], batch_size=3
    )
    assert len(multipoles) == 4, f"Expected 4 multipoles, got {len(multipoles)}"
    mtp_A = multipoles[0]
    mtp_B = multipoles[1]
    E_elst = apnet_pt.multipole.eval_qcel_dimer(
        mol_dimer=lr_water_dimer,
        qA=mtp_A[0].numpy(),
        muA=mtp_A[1].numpy(),
        thetaA=mtp_A[2].numpy(),
        qB=mtp_B[0].numpy(),
        muB=mtp_B[1].numpy(),
        thetaB=mtp_B[2].numpy(),
    )
    print(f"E_elst = {E_elst:.6f} kcal/mol")
    E_ref = -0.853646
    assert abs(E_elst - E_ref) < 1e-5, f"Expected {E_ref}, got {E_elst}"


def test_elst_multipoles_MTP_torch_no_damping():
    import torch

    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    r = df.iloc[0]
    # print(r['SAPT0 ELST ENERGY adz'])
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    alphaA = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    alphaB = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    (
        ref_elst_q,
        E_qqs_q,
        E_qus_q,
        E_uus_q,
        E_qQs_q,
        E_uQs_q,
        E_QQs_q,
        E_ZA_ZBs_q,
        E_ZA_MBs_q,
        E_ZB_MAs_q,
    ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
        mol_dimer=mol,
        qA=qA,
        muA=muA,
        qB=qB,
        muB=muB,
        thetaA=thetaA,
        thetaB=thetaB,
        # thetaA=np.zeros_like(thetaA),
        # thetaB=np.zeros_like(thetaB),
        alphaA=None,
        alphaB=None,
        traceless=False,
        amoeba_eq=True,
        match_cliff=False,
    )
    MTP_MTP = np.sum(E_qqs_q) + np.sum(E_qus_q) + np.sum(E_uus_q) + np.sum(E_qQs_q)
    E_ZA_ZB = E_ZA_ZBs_q.sum()
    E_ZA_MB = E_ZA_MBs_q.sum()
    E_ZB_MA = E_ZB_MAs_q.sum()
    ref_elst_q = MTP_MTP + E_ZA_ZB + E_ZA_MB + E_ZB_MA
    print(f"E_ZA_ZB = {E_ZA_ZB:.4f}")
    print(f"E_ZA_MB = {E_ZA_MB:.4f}")
    print(f"E_ZB_MA = {E_ZB_MA:.4f}")
    print(f"MTP_MTP = {MTP_MTP:.4f}")
    print(f"{ref_elst_q=:.6f} kcal/mol")

    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )
    dimer_batch.Ka = torch.tensor(alphaA, dtype=torch.float32)
    dimer_batch.Kb = torch.tensor(alphaB, dtype=torch.float32)
    RA = dimer_batch.RA
    RB = dimer_batch.RB
    dimer_batch.qA = torch.tensor(qA, dtype=torch.float32)
    dimer_batch.muA = torch.tensor(muA, dtype=torch.float32)
    dimer_batch.qB = torch.tensor(qB, dtype=torch.float32)
    dimer_batch.muB = torch.tensor(muB, dtype=torch.float32)

    dimer_batch.quadA = torch.zeros_like(torch.tensor(thetaA, dtype=torch.float32))
    dimer_batch.quadB = torch.zeros_like(torch.tensor(thetaB, dtype=torch.float32))
    dimer_batch.quadA = torch.tensor(thetaA, dtype=torch.float32)
    dimer_batch.quadB = torch.tensor(thetaB, dtype=torch.float32)

    torch_elst = apnet_pt.AtomPairwiseModels.mtp_mtp.mtp_elst(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
        # Q_const=1.0, # Agree with CLIFF
    )
    print(f"Torch elst = {torch.sum(torch_elst):.6f} kcal/mol")
    assert abs(ref_elst_q - torch.sum(torch_elst).item()) < 1e-2, (
        f"Expected {ref_elst_q}, got {torch.sum(torch_elst).item()}"
    )
    return


def test_elst_multipoles_MTP_torch_damping():
    import torch

    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    np.set_printoptions(precision=6)
    torch.set_printoptions(precision=6)
    alphaA = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    alphaB = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    (
        ref_elst_q,
        E_qqs_q,
        E_qus_q,
        E_uus_q,
        E_qQs_q,
        E_uQs_q,
        E_QQs_q,
        E_ZA_ZBs_q,
        E_ZA_MBs_q,
        E_ZB_MAs_q,
    ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
        mol_dimer=mol,
        qA=qA,
        qB=qB,
        muA=muA,
        muB=muB,
        # muA=np.zeros_like(muA),
        # muB=np.zeros_like(muB),
        thetaA=thetaA,
        thetaB=thetaB,
        # thetaA=np.zeros_like(thetaA),
        # thetaB=np.zeros_like(thetaB),
        alphaA=alphaA,
        alphaB=alphaB,
        traceless=False,
        amoeba_eq=True,
        match_cliff=False,
    )
    MTP_MTP = np.sum(E_qqs_q) + np.sum(E_qus_q) + np.sum(E_uus_q) + np.sum(E_qQs_q)
    E_ZA_ZB = E_ZA_ZBs_q.sum()
    E_ZA_MB = E_ZA_MBs_q.sum()
    E_ZB_MA = E_ZB_MAs_q.sum()
    ref_elst_q = MTP_MTP + E_ZA_ZB + E_ZA_MB + E_ZB_MA
    print(f"E_ZA_ZB = {E_ZA_ZB:.4f}")
    print(f"E_ZA_MB = {E_ZA_MB:.4f}")
    print(f"E_ZB_MA = {E_ZB_MA:.4f}")
    print(f"MTP_MTP = {MTP_MTP:.4f}")
    print(f"{ref_elst_q=:.6f} kcal/mol")

    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )
    dimer_batch.Ka = torch.tensor(alphaA, dtype=torch.float32)
    dimer_batch.Kb = torch.tensor(alphaB, dtype=torch.float32)
    RA = dimer_batch.RA
    RB = dimer_batch.RB
    dimer_batch.qA = torch.tensor(qA, dtype=torch.float32)
    dimer_batch.qB = torch.tensor(qB, dtype=torch.float32)

    dimer_batch.muA = torch.tensor(muA, dtype=torch.float32)
    dimer_batch.muB = torch.tensor(muB, dtype=torch.float32)
    # dimer_batch.muA = torch.zeros_like(torch.tensor(muA, dtype=torch.float32))
    # dimer_batch.muB = torch.zeros_like(torch.tensor(muB, dtype=torch.float32))

    # dimer_batch.quadA = torch.zeros_like(torch.tensor(thetaA, dtype=torch.float32))
    # dimer_batch.quadB = torch.zeros_like(torch.tensor(thetaB, dtype=torch.float32))
    dimer_batch.quadA = torch.tensor(thetaA, dtype=torch.float32)
    dimer_batch.quadB = torch.tensor(thetaB, dtype=torch.float32)

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
        # Q_const=1.0, # Agree with CLIFF
    )
    print(f"Torch elst = {torch.sum(torch_elst):.6f} kcal/mol")
    assert abs(ref_elst_q - torch.sum(torch_elst).item()) < 1e-2, (
        f"Expected {ref_elst_q}, got {torch.sum(torch_elst).item()}"
    )
    return


def test_elst_charge_dipole_qpole():
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols(
        [monA, monB, monA.copy(), monB.copy()], batch_size=3
    )
    assert len(multipoles) == 4, f"Expected 4 multipoles, got {len(multipoles)}"
    mtp_A = multipoles[0]
    mtp_B = multipoles[1]
    E_q, E_dp, E_qpole = apnet_pt.multipole.eval_qcel_dimer_individual(
        mol_dimer=lr_water_dimer,
        qA=mtp_A[0].numpy(),
        muA=mtp_A[1].numpy(),
        thetaA=mtp_A[2].numpy(),
        qB=mtp_B[0].numpy(),
        muB=mtp_B[1].numpy(),
        thetaB=mtp_B[2].numpy(),
    )
    print(f"E_q = {E_q:.6f} kcal/mol")
    print(f"E_dp = {E_dp:.6f} kcal/mol")
    print(f"E_qpole = {E_qpole:.6f} kcal/mol")
    E_q_ref = -1.239722
    E_dp_ref = 0.392898
    E_qpole_ref = -0.006823
    assert abs(E_q - E_q_ref) < 1e-5, f"Expected {E_q_ref}, got {E_q}"
    assert abs(E_dp - E_dp_ref) < 1e-5, f"Expected {E_dp_ref}, got {E_dp}"
    assert abs(E_qpole - E_qpole_ref) < 1e-5, f"Expected {E_qpole_ref}, got {E_qpole}"


def test_elst_charge_dipole_qpole_pairwise():
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    ).set_pretrained_model(model_id=0)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols(
        [monA, monB, monA.copy(), monB.copy()], batch_size=3
    )
    assert len(multipoles) == 4, f"Expected 4 multipoles, got {len(multipoles)}"
    mtp_A = multipoles[0]
    mtp_B = multipoles[1]
    total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs, _, _, _ = (
        apnet_pt.multipole.eval_qcel_dimer_individual_components(
            mol_dimer=lr_water_dimer,
            qA=mtp_A[0].numpy(),
            muA=mtp_A[1].numpy(),
            thetaA=mtp_A[2].numpy(),
            qB=mtp_B[0].numpy(),
            muB=mtp_B[1].numpy(),
            thetaB=mtp_B[2].numpy(),
        )
    )
    print(f"Total energy = {total_energy:.6f} kcal/mol")
    print(f"E_qqs = {E_qqs.sum():.6f} kcal/mol")
    print(f"E_qus = {E_qus.sum():.6f} kcal/mol")
    print(f"E_uus = {E_uus.sum():.6f} kcal/mol")
    print(f"E_qQs = {E_qQs.sum():.6f} kcal/mol")
    print(f"E_uQs = {E_uQs.sum():.6f} kcal/mol")
    print(f"E_QQs = {E_QQs.sum():.6f} kcal/mol")
    return


def test_elst_multipoles_am_hirshfeld():
    atom_model = apnet_pt.AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    atom_model.set_pretrained_model(
        current_file_path + "/../models/am_hf_ensemble/am_0.pt"
    )
    print(atom_model)
    monA = lr_water_dimer.get_fragment(0).copy()
    monB = lr_water_dimer.get_fragment(1).copy()
    multipoles = atom_model.predict_qcel_mols(
        [monA, monB, monA.copy(), monB.copy()], batch_size=3
    )
    assert len(multipoles) == 4, f"Expected 4 multipoles, got {len(multipoles)}"
    mtp_A = multipoles[0]
    mtp_B = multipoles[1]
    E_elst = apnet_pt.multipole.eval_qcel_dimer(
        mol_dimer=lr_water_dimer,
        qA=mtp_A[0].numpy(),
        muA=mtp_A[1].numpy(),
        thetaA=mtp_A[2].numpy(),
        qB=mtp_B[0].numpy(),
        muB=mtp_B[1].numpy(),
        thetaB=mtp_B[2].numpy(),
    )
    print(f"E_elst = {E_elst:.6f} kcal/mol")
    E_ref = -0.7430384309295008
    assert abs(E_elst - E_ref) < 1e-6, f"Expected {E_ref}, got {E_elst}"


def test_induced_dipole():
    # check here for CLIFF eval: /home/awallace43/projects/multipoles/cliff_tests
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    for n, r in df.iterrows():
        sapt0_ind = r["SAPT0 IND ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        cliff_ind = r["cliff_indu_q_mu"]
        mol = r["qcel_molecule"]
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(
            np.sum((monA.geometry[:, None] - monB.geometry) ** 2, axis=2)
        ).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r["q_A pbe0/atz"]
        muA = r["mu_A pbe0/atz"]
        thetaA = r["theta_A pbe0/atz"]
        thetaA = np.zeros_like(thetaA)
        qB = r["q_B pbe0/atz"]
        muB = r["mu_B pbe0/atz"]
        thetaB = r["theta_B pbe0/atz"]
        thetaB = np.zeros_like(thetaB)
        vrA = r["vol_ratios_A pbe0/atz"]
        vrB = r["vol_ratios_B pbe0/atz"]
        vwA = r["val_widths_A pbe0/atz"]
        vwB = r["val_widths_B pbe0/atz"]
        atom_alpha_iso = np.array(
            [
                [8.38374595553467, 0.4842211422539944, 0.4977805639070765],
                [8.388563748172823, 0.4855270362311864, 0.4855449542590184],
            ]
        )
        induction_energy = apnet_pt.multipole.dimer_induced_dipole(
            mol,
            qA=qA,
            muA=muA,
            thetaA=thetaA,
            qB=qB,
            muB=muB,
            thetaB=thetaB,
            hirshfeld_volume_ratio_A=vrA,
            hirshfeld_volume_ratio_B=vrB,
            valence_widths_A=vwA,
            valence_widths_B=vwB,
            atom_polarizabilities_A=atom_alpha_iso[0],
            atom_polarizabilities_B=atom_alpha_iso[1],
        )
        print(f"Distance between monomers: {dist * bohr2angstrom:.2f} A")
        print(f"SAPT induction   = {sapt0_ind:.6f} kcal/mol")
        print(f"Induction energy = {induction_energy:.6f} kcal/mol")
        print(f"CLIFF induction  = {cliff_ind:.6f}")


def test_induced_dipole_bz_meoh():
    df = pd.read_pickle(
        current_file_path + os.sep + os.path.join("dataset_data", "df_bz_meoh_mbis.pkl")
    )
    for n, r in df.iterrows():
        sapt0_ind = r["SAPT0 IND ENERGY adz"]
        sapt0_elst = r["SAPT0 ELST ENERGY adz"]
        mol = r["qcel_molecule"]
        # qm_tools_aw.molecular_visualization.visualize_molecule(
        #     mol,
        #    temp_filename=f"{n}_water_dimer_sapt0_ind.html",
        #                                                        )
        # Distance between monomers
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(
            np.sum((monA.geometry[:, None] - monB.geometry) ** 2, axis=2)
        ).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r["q_A pbe0/atz"]
        muA = r["mu_A pbe0/atz"]
        thetaA = r["theta_A pbe0/atz"]
        qB = r["q_B pbe0/atz"]
        muB = r["mu_B pbe0/atz"]
        thetaB = r["theta_B pbe0/atz"]
        vrA = r["vol_ratios_A pbe0/atz"]
        vrB = r["vol_ratios_B pbe0/atz"]
        vwA = r["val_widths_A pbe0/atz"]
        vwB = r["val_widths_B pbe0/atz"]
        total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs, _, _, _ = (
            apnet_pt.multipole.eval_qcel_dimer_individual_components(
                mol_dimer=mol,
                qA=qA,
                muA=muA,
                thetaA=thetaA,
                qB=qB,
                muB=muB,
                thetaB=thetaB,
            )
        )
        E_qq = E_qqs.sum()
        E_qu = E_qus.sum()
        E_uu = E_uus.sum()
        E_uQ = E_uQs.sum()
        E_qQ = E_qQs.sum()
        print(f"{total_energy=:.6f} kcal/mol")
        print(f"{E_qq=:.6f} kcal/mol")
        print(f"{E_qu=:.6f} kcal/mol")
        print(f"{E_uu=:.6f} kcal/mol")
        print(f"{E_qQ=:.6f} kcal/mol")
        print(f"{E_uQ=:.6f} kcal/mol")
        induction_energy = apnet_pt.multipole.dimer_induced_dipole(
            mol,
            qA=qA,
            muA=muA,
            thetaA=thetaA,
            qB=qB,
            muB=muB,
            thetaB=thetaB,
            hirshfeld_volume_ratio_A=vrA,
            hirshfeld_volume_ratio_B=vrB,
            valence_widths_A=vwA,
            valence_widths_B=vwB,
        )
        h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")
        print(f"Distance between monomers: {dist * bohr2angstrom:.2f} A")
        print(f"SAPT elst        = {sapt0_elst * h2kcalmol:.6f} kcal/mol")
        print(f"SAPT induction   = {sapt0_ind * h2kcalmol:.6f} kcal/mol")
        print(f"Induction energy = {induction_energy:.6f} kcal/mol")
        # assert abs(induction_energy - sapt0_ind) < 1e-6, f"Expected {sapt0_ind}, got {induction_energy}"


def test_classical_cliff():
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    # pprint(df.columns.to_list())
    ap_elst, ap_ind = [], []
    ap_elst_q, ap_elst_q_mu, ap_elst_mu, ap_elst_theta = [], [], [], []
    ap_elst_q_theta, ap_elst_mu_theta = [], []
    r = df.iloc[0]
    for n, r in df.iterrows():
        sapt0_ind = r["SAPT0 IND ENERGY adz"]
        sapt0_elst = r["SAPT0 ELST ENERGY adz"]
        mol = r["qcel_molecule"]
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(
            np.sum((monA.geometry[:, None] - monB.geometry) ** 2, axis=2)
        ).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r["q_A pbe0/atz"]
        muA = r["mu_A pbe0/atz"]
        thetaA = r["theta_A pbe0/atz"]
        qB = r["q_B pbe0/atz"]
        muB = r["mu_B pbe0/atz"]
        thetaB = r["theta_B pbe0/atz"]
        vrA = r["vol_ratios_A pbe0/atz"]
        vrB = r["vol_ratios_B pbe0/atz"]
        vwA = r["val_widths_A pbe0/atz"]
        vwB = r["val_widths_B pbe0/atz"]
        print(f"{qA=}")
        print(f"{muA=}")
        print(f"{thetaA=}")
        total_energy, E_qqs, E_qus, E_uus, E_qQs, E_uQs, E_QQs, _, _, _ = (
            apnet_pt.multipole.eval_qcel_dimer_individual_components(
                mol_dimer=mol,
                qA=qA,
                muA=muA,
                thetaA=thetaA,
                qB=qB,
                muB=muB,
                thetaB=thetaB,
                traceless=False,
            )
        )
        ap_elst.append(total_energy)
        E_qq = E_qqs.sum()
        E_qu = E_qus.sum()
        E_uu = E_uus.sum()
        E_QQ = E_QQs.sum()
        E_uQ = E_uQs.sum()
        E_qQ = E_qQs.sum()
        ap_elst_q.append(E_qq)
        ap_elst_q_mu.append(E_qq + E_qu + E_uu)
        ap_elst_mu.append(E_uu)
        ap_elst_theta.append(E_QQ)
        ap_elst_q_theta.append(E_qq + E_qQ + E_QQ)
        ap_elst_mu_theta.append(E_uu + E_uQ + E_QQ)
        # induction_energy = apnet_pt.multipole.dimer_induced_dipole(
        #     mol,
        #     qA=qA,
        #     muA=muA,
        #     thetaA=thetaA,
        #     qB=qB,
        #     muB=muB,
        #     thetaB=thetaB,
        #     hirshfeld_volume_ratio_A=vrA,
        #     hirshfeld_volume_ratio_B=vrB,
        #     valence_widths_A=vwA,
        #     valence_widths_B=vwB,
        # )
        # ap_ind.append(induction_energy)
        # break
    print(
        apnet_pt.multipole.charge_dipole_qpoles_to_compact_multipoles(
            charges=qA,
            dipoles=muA,
            qpoles=thetaA,
        )
    )
    print(ap_elst)
    # return
    df["ap_elst"] = ap_elst
    df["ap_elst_q"] = ap_elst_q
    df["ap_elst_q_mu"] = ap_elst_q_mu
    df["ap_elst_mu"] = ap_elst_mu
    df["ap_elst_theta"] = ap_elst_theta
    df["ap_elst_q_theta"] = ap_elst_q_theta
    df["ap_elst_mu_theta"] = ap_elst_mu_theta
    print(df[["cliff_elst_q_mu_theta", "ap_elst", "SAPT0 ELST ENERGY adz"]])
    print(df[["cliff_elst_q_mu_theta_noDamp_noZ", "ap_elst", "SAPT0 ELST ENERGY adz"]])
    print(df[["cliff_elst_q_noDamp_noZ", "ap_elst_q", "SAPT0 ELST ENERGY adz"]])
    print(df[["cliff_elst_mu_noDamp_noZ", "ap_elst_mu"]])
    print(df[["cliff_elst_theta_noDamp_noZ", "ap_elst_theta"]])

    print("\nCross terms\n")
    print(df[["cliff_elst_q_mu_noDamp_noZ", "ap_elst_q_mu"]])
    print(df[["cliff_elst_mu_theta_noDamp_noZ", "ap_elst_mu_theta"]])
    print(df[["cliff_elst_q_theta_noDamp_noZ", "ap_elst_q_theta"]])
    return


def test_elst_ameoba():
    """
    Validate AMOEBA-equivalent electrostatic energy components against CLIFF no-damping references for a water dimer.
    
    Loads the first entry from the water_dimer_pes3.pkl dataset and computes electrostatic energies (charge-only, charge+dipole, charge+dipole+quadrupole) using the APNET evaluation configured to match CLIFF and AMOEBA-equivalent settings. For each multipole expansion level the test compares the computed AP energy to the corresponding CLIFF no-damping reference and asserts agreement within 1e-4 kcal/mol.
    """
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    # q-q case
    (
        ap_q,
        E_qqs_q,
        E_qus_q,
        E_uus_q,
        E_qQs_q,
        E_uQs_q,
        E_QQs_q,
        E_ZA_ZBs_q,
        E_ZA_MBs_q,
        E_ZB_MAs_q,
    ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
        mol_dimer=mol,
        qA=qA,
        muA=np.zeros_like(muA),
        thetaA=np.zeros_like(thetaA),
        qB=qB,
        muB=np.zeros_like(muB),
        thetaB=np.zeros_like(thetaB),
        traceless=False,
        amoeba_eq=True,
        match_cliff=True,
    )
    E_ZA_ZB = E_ZA_ZBs_q.sum()
    E_ZA_MB = E_ZA_MBs_q.sum()
    E_ZB_MA = E_ZB_MAs_q.sum()
    cliff_type = "q_noDamp"
    print(f"Using cliff type: {cliff_type}\n")
    print(f"{E_ZA_ZB=:.6f}, {E_ZA_MB=:.6f}, {E_ZB_MA=:.6f}")
    print(f"{ap_q=:.6f} kcal/mol")
    cliff_elst_q = r[f"cliff_elst_{cliff_type}"]
    print(f"CLIFF q = {cliff_elst_q:.6f}, AP q = {ap_q:.6f}")
    assert abs(cliff_elst_q - ap_q) < 1e-4, f"Expected {cliff_elst_q}, got {ap_q}"
    (
        ap_q_mu,
        E_qqs_q_mu,
        E_qus_q_mu,
        E_uus_q_mu,
        E_qQs_q_mu,
        E_uQs_q_mu,
        E_QQs_q_mu,
        E_ZA_ZBs_q_mu,
        E_ZA_MBs_q_mu,
        E_ZB_MAs_q_mu,
    ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
        mol_dimer=mol,
        qA=qA,
        muA=muA,
        thetaA=np.zeros_like(thetaA),
        qB=qB,
        muB=muB,
        thetaB=np.zeros_like(thetaB),
        traceless=False,
        amoeba_eq=True,
        match_cliff=True,
    )
    E_ZA_ZB = E_ZA_ZBs_q_mu.sum()
    E_ZA_MB = E_ZA_MBs_q_mu.sum()
    E_ZB_MA = E_ZB_MAs_q_mu.sum()
    cliff_type = "q_mu_noDamp"
    print(f"Using cliff type: {cliff_type}\n")
    print(f"{E_ZA_ZB=:.6f}, {E_ZA_MB=:.6f}, {E_ZB_MA=:.6f}")
    print(f"{ap_q_mu=:.6f} kcal/mol")
    cliff_elst_q_mu = r[f"cliff_elst_{cliff_type}"]
    print(f"CLIFF q = {cliff_elst_q_mu:.6f}, AP q = {ap_q_mu:.6f}")
    assert abs(cliff_elst_q_mu - ap_q_mu) < 1e-4, (
        f"Expected {cliff_elst_q_mu}, got {ap_q_mu}"
    )
    (
        ap_q_mu_theta,
        E_qqs_q_mu_theta,
        E_qus_q_mu_theta,
        E_uus_q_mu_theta,
        E_qQs_q_mu_theta,
        E_uQs_q_mu_theta,
        E_QQs_q_mu_theta,
        E_ZA_ZBs_q_mu_theta,
        E_ZA_MBs_q_mu_theta,
        E_ZB_MAs_q_mu_theta,
    ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
        mol_dimer=mol,
        qA=qA,
        muA=muA,
        thetaA=thetaA,
        qB=qB,
        muB=muB,
        thetaB=thetaB,
        traceless=False,
        amoeba_eq=True,
        match_cliff=True,
    )
    E_ZA_ZB = E_ZA_ZBs_q_mu_theta.sum()
    E_ZA_MB = E_ZA_MBs_q_mu_theta.sum()
    E_ZB_MA = E_ZB_MAs_q_mu_theta.sum()
    cliff_type = "q_mu_theta_noDamp"
    print(f"Using cliff type: {cliff_type}\n")
    print(f"{E_ZA_ZB=:.6f}, {E_ZA_MB=:.6f}, {E_ZB_MA=:.6f}")
    print(f"{ap_q_mu_theta=:.6f} kcal/mol")
    cliff_elst_q_mu_theta = r[f"cliff_elst_{cliff_type}"]
    print(f"CLIFF q = {cliff_elst_q_mu_theta:.6f}, AP q = {ap_q_mu_theta:.6f}")
    assert abs(cliff_elst_q_mu_theta - ap_q_mu_theta) < 1e-4, (
        f"Expected {cliff_elst_q_mu_theta}, got {ap_q_mu_theta}"
    )
    return


def test_elst_damping_CLIFF():
    """
    Validate that AP electrostatic energies match CLIFF reference values for two damping configurations.
    
    Loads the first entry from the water dimer test dataset, computes electrostatic components with AMOEBA-equivalent damping and CLIFF matching for the charge-only ("q") and charge+dipole ("q_mu") cases, and asserts the AP-computed totals agree with the stored CLIFF reference values within 1e-4.
    """
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    alphaA = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    alphaB = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    # q-q case
    (
        ap_q,
        E_qqs_q,
        E_qus_q,
        E_uus_q,
        E_qQs_q,
        E_uQs_q,
        E_QQs_q,
        E_ZA_ZBs_q,
        E_ZA_MBs_q,
        E_ZB_MAs_q,
    ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
        mol_dimer=mol,
        qA=qA,
        muA=np.zeros_like(muA),
        thetaA=np.zeros_like(thetaA),
        qB=qB,
        muB=np.zeros_like(muB),
        thetaB=np.zeros_like(thetaB),
        alphaA=alphaA,
        alphaB=alphaB,
        traceless=False,
        amoeba_eq=True,
        match_cliff=True,
    )
    MTP_MTP = (
        np.sum(E_qqs_q)
        + np.sum(E_qus_q)
        + np.sum(E_uus_q)
        + np.sum(E_qQs_q)
        + np.sum(E_uQs_q)
        + np.sum(E_QQs_q)
    )
    E_ZA_ZB = E_ZA_ZBs_q.sum()
    E_ZA_MB = E_ZA_MBs_q.sum()
    E_ZB_MA = E_ZB_MAs_q.sum()
    # print(h2kcalmol)
    # print(a2b)
    # print(b2a)
    cliff_type = "q"
    print(f"Using cliff type: {cliff_type}\n")
    # print("Elst: 12056.938032 + -12237.127718 + -11859.847832 + 12026.462390 = -13.575127")
    print(f"{ap_q=:.6f} kcal/mol")
    cliff_elst_q = r[f"cliff_elst_{cliff_type}"]
    print(f"CLIFF q = {cliff_elst_q:.6f}, AP q = {ap_q:.6f}")
    assert abs(cliff_elst_q - ap_q) < 1e-4, f"Expected {cliff_elst_q}, got {ap_q}"
    (
        ap_q_mu,
        E_qqs_q_mu,
        E_qus_q_mu,
        E_uus_q_mu,
        E_qQs_q_mu,
        E_uQs_q_mu,
        E_QQs_q_mu,
        E_ZA_ZBs_q_mu,
        E_ZA_MBs_q_mu,
        E_ZB_MAs_q_mu,
    ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
        mol_dimer=mol,
        qA=qA,
        # muA=np.zeros_like(muA),
        muA=muA,
        thetaA=np.zeros_like(thetaA),
        qB=qB,
        # muB=np.zeros_like(muB),
        muB=muB,
        thetaB=np.zeros_like(thetaB),
        alphaA=alphaA,
        alphaB=alphaB,
        traceless=False,
        amoeba_eq=True,
        match_cliff=True,
    )
    MTP_MTP = (
        np.sum(E_qqs_q_mu)
        + np.sum(E_qus_q_mu)
        + np.sum(E_uus_q_mu)
        + np.sum(E_qQs_q_mu)
        + np.sum(E_uQs_q_mu)
        + np.sum(E_QQs_q_mu)
    )
    E_ZA_ZB = E_ZA_ZBs_q_mu.sum()
    E_ZA_MB = E_ZA_MBs_q_mu.sum()
    E_ZB_MA = E_ZB_MAs_q_mu.sum()
    cliff_type = "q_mu"
    print(f"Using cliff type: {cliff_type}\n")
    print(f"{E_ZA_ZB=:.6f}, {E_ZA_MB=:.6f}, {E_ZB_MA=:.6f}")
    print(f"{ap_q_mu=:.6f} kcal/mol")
    print(f"{E_ZA_ZB=:.6f} + {E_ZA_MB=:.6f} + {E_ZB_MA=:.6f} + {MTP_MTP:.6f}")
    print(
        f"Elst: {E_ZA_ZB: .6f} + {E_ZA_MB: .6f} + {E_ZB_MA: .6f} + "
        f"{MTP_MTP: .6f}={ap_q_mu: .6f}"
    )
    print(
        "Elst: 12056.938032 + -12204.355385 + -11877.736773 + 12014.622387 = -10.531739"
    )
    cliff_elst_q_mu = r[f"cliff_elst_{cliff_type}"]
    print(f"CLIFF q = {cliff_elst_q_mu:.6f}, AP q = {ap_q_mu:.6f}")
    assert abs(cliff_elst_q_mu - ap_q_mu) < 1e-4, (
        f"Expected {cliff_elst_q_mu}, got {ap_q_mu}"
    )
    return


def test_induced_dipole_no_damping():
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    print(df)
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    vrA = r["vol_ratios_A pbe0/atz"]
    vrB = r["vol_ratios_B pbe0/atz"]
    vwA = r["val_widths_A pbe0/atz"]
    vwB = r["val_widths_B pbe0/atz"]
    atom_alpha_iso = np.array(
        [
            [8.38374595553467, 0.4842211422539944, 0.4977805639070765],
            [8.388563748172823, 0.4855270362311864, 0.4855449542590184],
        ]
    )
    thetaA = np.zeros_like(thetaA)
    thetaB = np.zeros_like(thetaB)
    ap_q_mu_induction = apnet_pt.multipole.dimer_induced_dipole(
        mol,
        qA=qA,
        muA=muA,
        thetaA=thetaA,
        qB=qB,
        muB=muB,
        thetaB=thetaB,
        hirshfeld_volume_ratio_A=vrA,
        hirshfeld_volume_ratio_B=vrB,
        atom_polarizabilities_A=atom_alpha_iso[0],
        atom_polarizabilities_B=atom_alpha_iso[1],
        valence_widths_A=vwA,
        valence_widths_B=vwB,
    )
    cliff_type = "q_mu"
    print(f"Using cliff type: {cliff_type}\n")
    cliff_indu_q_mu = r[f"cliff_indu_{cliff_type}"]
    print(f"CLIFF q = {cliff_indu_q_mu:.6f}, AP q = {ap_q_mu_induction:.6f}")
    # not using CLIFF directly, because we are not implementing the short-range
    # exch-ind correction
    assert abs(-1.4232044527609915 - ap_q_mu_induction) < 1e-4, (
        f"Expected {cliff_indu_q_mu}, got {ap_q_mu_induction}"
    )
    return


def test_induced_dipole_torch():
    import torch

    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    vrA = r["vol_ratios_A pbe0/atz"]
    vrB = r["vol_ratios_B pbe0/atz"]
    vwA = r["val_widths_A pbe0/atz"]
    vwB = r["val_widths_B pbe0/atz"]
    atom_alpha_iso = np.array(
        [
            [8.38374595553467, 0.4842211422539944, 0.4977805639070765],
            [8.388563748172823, 0.4855270362311864, 0.4855449542590184],
        ]
    )
    thetaA = np.zeros_like(thetaA)
    thetaB = np.zeros_like(thetaB)
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    ap_q_mu_induction = apnet_pt.multipole.dimer_induced_dipole(
        mol,
        qA=qA,
        muA=muA,
        thetaA=thetaA,
        qB=qB,
        muB=muB,
        thetaB=thetaB,
        hirshfeld_volume_ratio_A=vrA,
        hirshfeld_volume_ratio_B=vrB,
        # Use computed polarizabilities
        atom_polarizabilities_A=atom_alpha_iso[0],
        # Use computed polarizabilities
        atom_polarizabilities_B=atom_alpha_iso[1],
        valence_widths_A=vwA,
        valence_widths_B=vwB,
    )
    ref_e = -1.4232045
    print(f"{ap_q_mu_induction = }")
    # assert abs(ref_e - ap_q_mu_induction) < 1e-4, (
    #     f"Expected {ref_e}, got {ap_q_mu_induction}"
    # )
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    alphaA = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    alphaB = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    vrA = r["vol_ratios_A pbe0/atz"]
    vrB = r["vol_ratios_B pbe0/atz"]
    vwA = r["val_widths_A pbe0/atz"]
    vwB = r["val_widths_B pbe0/atz"]
    atom_alpha_iso = torch.tensor(
        [
            [8.38374595553467, 0.4842211422539944, 0.4977805639070765],
            [8.388563748172823, 0.4855270362311864, 0.4855449542590184],
        ]
    )
    thetaA = np.zeros_like(thetaA)
    thetaB = np.zeros_like(thetaB)
    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )
    dimer_batch.Ka = torch.tensor(alphaA, dtype=torch.float32)
    dimer_batch.Kb = torch.tensor(alphaB, dtype=torch.float32)
    dimer_batch.qA = torch.tensor(qA, dtype=torch.float32)
    dimer_batch.qB = torch.tensor(qB, dtype=torch.float32)

    dimer_batch.muA = torch.tensor(muA, dtype=torch.float32)
    dimer_batch.muB = torch.tensor(muB, dtype=torch.float32)
    dimer_batch.quadA = torch.zeros_like(torch.tensor(thetaA, dtype=torch.float32))
    dimer_batch.quadB = torch.zeros_like(torch.tensor(thetaB, dtype=torch.float32))

    ap_q_mu_induction = apnet_pt.multipole.dimer_induced_dipole_torch(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        e_AA_source=dimer_batch.e_AA_source,
        e_BB_source=dimer_batch.e_BB_source,
        e_AA_target=dimer_batch.e_AA_target,
        e_BB_target=dimer_batch.e_BB_target,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
        hirshfeld_volume_ratio_A=torch.tensor(vrA),
        hirshfeld_volume_ratio_B=torch.tensor(vrB),
        valence_widths_A=torch.tensor(vwA),
        valence_widths_B=torch.tensor(vwB),
        atom_polarizabilities_A=atom_alpha_iso[0],
        atom_polarizabilities_B=atom_alpha_iso[1],
        # Q_const=1.0, # Agree with CLIFF
    )
    torch_ap_indu = ap_q_mu_induction.detach().numpy().sum()
    print(f"{torch_ap_indu = }")
    assert abs(ref_e - torch_ap_indu) < 1e-4, f"Expected {ref_e}, got {torch_ap_indu}"
    return


def test_induced_dipole_torch_alphas():
    import torch

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    vrA = r["vol_ratios_A pbe0/atz"]
    vrB = r["vol_ratios_B pbe0/atz"]
    vwA = r["val_widths_A pbe0/atz"]
    vwB = r["val_widths_B pbe0/atz"]
    Ks = [
        [1.14769962, 0.685558974, 0.685558974],
        [1.14769962, 0.685558974, 0.685558974],
    ]
    thetaA = np.zeros_like(thetaA)
    thetaB = np.zeros_like(thetaB)
    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )
    dimer_batch.qA = torch.tensor(qA, dtype=torch.float32)
    dimer_batch.qB = torch.tensor(qB, dtype=torch.float32)

    dimer_batch.muA = torch.tensor(muA, dtype=torch.float32)
    dimer_batch.muB = torch.tensor(muB, dtype=torch.float32)
    dimer_batch.quadA = torch.zeros_like(torch.tensor(thetaA, dtype=torch.float32))
    dimer_batch.quadB = torch.zeros_like(torch.tensor(thetaB, dtype=torch.float32))

    ap_q_mu_induction = apnet_pt.multipole.dimer_induced_dipole_torch(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        e_AA_source=dimer_batch.e_AA_source,
        e_BB_source=dimer_batch.e_BB_source,
        e_AA_target=dimer_batch.e_AA_target,
        e_BB_target=dimer_batch.e_BB_target,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
        hirshfeld_volume_ratio_A=torch.tensor(vrA, dtype=torch.float32),
        hirshfeld_volume_ratio_B=torch.tensor(vrB, dtype=torch.float32),
        valence_widths_A=torch.tensor(vwA),
        valence_widths_B=torch.tensor(vwB),
        K_A=torch.tensor(Ks[0], dtype=torch.float32),
        K_B=torch.tensor(Ks[1], dtype=torch.float32),
    )
    # using libmbd free polarizabilities which causes the shift from CLIFF
    # induction energies AND K_ij. Everything else has been verified to match CLIFF
    ref_e = -3.9513449668884277
    torch_ap_indu = ap_q_mu_induction.detach().numpy().sum()
    print(f"{torch_ap_indu = }")
    assert abs(ref_e - torch_ap_indu) < 1e-4, f"Expected {ref_e}, got {torch_ap_indu}"
    return


def test_induced_dipole_torch_alphas_dimer_eval():
    import torch

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    qB = r["q_B pbe0/atz"]
    muB = r["mu_B pbe0/atz"]
    thetaB = r["theta_B pbe0/atz"]
    vrA = r["vol_ratios_A pbe0/atz"]
    vrB = r["vol_ratios_B pbe0/atz"]
    vwA = r["val_widths_A pbe0/atz"]
    vwB = r["val_widths_B pbe0/atz"]
    Ks = [
        [1.14769962, 0.685558974, 0.685558974],
        [1.14769962, 0.685558974, 0.685558974],
    ]
    thetaA = np.zeros_like(thetaA)
    thetaB = np.zeros_like(thetaB)
    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )
    dimer_batch.qA = torch.tensor(qA, dtype=torch.float32)
    dimer_batch.qB = torch.tensor(qB, dtype=torch.float32)

    dimer_batch.muA = torch.tensor(muA, dtype=torch.float32)
    dimer_batch.muB = torch.tensor(muB, dtype=torch.float32)
    dimer_batch.quadA = torch.zeros_like(torch.tensor(thetaA, dtype=torch.float32))
    dimer_batch.quadB = torch.zeros_like(torch.tensor(thetaB, dtype=torch.float32))

    ap_q_mu_induction = apnet_pt.multipole.dimer_induced_dipole_torch(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        e_AA_source=dimer_batch.e_AA_source,
        e_BB_source=dimer_batch.e_BB_source,
        e_AA_target=dimer_batch.e_AA_target,
        e_BB_target=dimer_batch.e_BB_target,
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
        hirshfeld_volume_ratio_A=torch.tensor(vrA, dtype=torch.float32),
        hirshfeld_volume_ratio_B=torch.tensor(vrB, dtype=torch.float32),
        valence_widths_A=torch.tensor(vwA),
        valence_widths_B=torch.tensor(vwB),
        K_A=torch.tensor(Ks[0], dtype=torch.float32),
        K_B=torch.tensor(Ks[1], dtype=torch.float32),
    )
    # using libmbd free polarizabilities which causes the shift from CLIFF
    # induction energies AND K_ij. Everything else has been verified to match CLIFF
    ref_e = -3.9513449668884277
    torch_ap_indu = ap_q_mu_induction.detach().numpy().sum()
    print(f"{torch_ap_indu = }")
    assert abs(ref_e - torch_ap_indu) < 1e-4, f"Expected {ref_e}, got {torch_ap_indu}"
    return


def test_induced_dipole_torch_df():
    # check here for CLIFF eval: /home/awallace43/projects/multipoles/cliff_tests
    """
    vrA=array([1.3908595 , 0.18787692, 0.19180904]), vrB=array([1.39145891, 0.1882568 , 0.18826201]
    )
    vwA=array([0.41118342, 0.35029466, 0.35229699]), vwB=array([0.41117481, 0.35060148, 0.35060415]
    )
    Ks=[[1.14769962, 0.685558974, 0.685558974], [1.14769962, 0.685558974, 0.685558974]]
    hirshfeld_volume_ratio_A=tensor([1.3909, 0.1879, 0.1918])
    hirshfeld_volume_ratio_B=tensor([1.3915, 0.1883, 0.1883])
    """
    import torch

    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    Ks = [
        [1.14769962, 0.685558974, 0.685558974],
        [1.14769962, 0.685558974, 0.685558974],
    ]
    for n, r in df.iterrows():
        sapt0_ind = r["SAPT0 IND ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        cliff_ind = r["cliff_indu_q_mu"]
        mol = r["qcel_molecule"]
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(
            np.sum((monA.geometry[:, None] - monB.geometry) ** 2, axis=2)
        ).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r["q_A pbe0/atz"]
        muA = r["mu_A pbe0/atz"]
        thetaA = r["theta_A pbe0/atz"]
        thetaA = np.zeros_like(thetaA)
        qB = r["q_B pbe0/atz"]
        muB = r["mu_B pbe0/atz"]
        thetaB = r["theta_B pbe0/atz"]
        thetaB = np.zeros_like(thetaB)
        vrA = r["vol_ratios_A pbe0/atz"]
        vrB = r["vol_ratios_B pbe0/atz"]
        vwA = r["val_widths_A pbe0/atz"]
        vwB = r["val_widths_B pbe0/atz"]
        atom_alpha_iso = torch.tensor(
            [
                [8.38374595553467, 0.4842211422539944, 0.4977805639070765],
                [8.388563748172823, 0.4855270362311864, 0.4855449542590184],
            ]
        )
        thetaA = np.zeros_like(thetaA)
        thetaB = np.zeros_like(thetaB)
        dimer_batch = (
            apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
                [
                    apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                        mol, r_cut_im=99999.0, dimer_ind=0
                    )
                ]
            )
        )
        dimer_batch.qA = torch.tensor(qA, dtype=torch.float32)
        dimer_batch.qB = torch.tensor(qB, dtype=torch.float32)

        dimer_batch.muA = torch.tensor(muA, dtype=torch.float32)
        dimer_batch.muB = torch.tensor(muB, dtype=torch.float32)
        dimer_batch.quadA = torch.zeros_like(torch.tensor(thetaA, dtype=torch.float32))
        dimer_batch.quadB = torch.zeros_like(torch.tensor(thetaB, dtype=torch.float32))

        print(f"{vrA=}, {vrB=}")
        print(f"{vwA=}, {vwB=}")
        print(f"{Ks=}")

        ap_q_mu_induction = apnet_pt.multipole.dimer_induced_dipole_torch(
            ZA=dimer_batch.ZA,
            RA=dimer_batch.RA,
            qA=dimer_batch.qA,
            muA=dimer_batch.muA,
            quadA=dimer_batch.quadA,
            ZB=dimer_batch.ZB,
            RB=dimer_batch.RB,
            qB=dimer_batch.qB,
            muB=dimer_batch.muB,
            quadB=dimer_batch.quadB,
            e_AA_source=dimer_batch.e_AA_source,
            e_BB_source=dimer_batch.e_BB_source,
            e_AA_target=dimer_batch.e_AA_target,
            e_BB_target=dimer_batch.e_BB_target,
            e_AB_source=dimer_batch.e_ABsr_source,
            e_AB_target=dimer_batch.e_ABsr_target,
            hirshfeld_volume_ratio_A=torch.tensor(vrA, dtype=torch.float32),
            hirshfeld_volume_ratio_B=torch.tensor(vrB, dtype=torch.float32),
            valence_widths_A=torch.tensor(vwA),
            valence_widths_B=torch.tensor(vwB),
            atom_polarizabilities_A=atom_alpha_iso[0],
            atom_polarizabilities_B=atom_alpha_iso[1],
            K_A=torch.tensor(Ks[0], dtype=torch.float32),
            K_B=torch.tensor(Ks[1], dtype=torch.float32),
            # Q_const=1.0, # Agree with CLIFF
        )
        induction_energy = ap_q_mu_induction.detach().numpy().sum()
        print(f"Distance between monomers: {dist * bohr2angstrom:.2f} A")
        print(f"SAPT induction   = {sapt0_ind:.6f} kcal/mol")
        print(f"Induction energy = {induction_energy:.6f} kcal/mol")
        print(f"CLIFF induction  = {cliff_ind:.6f}")


def test_elst_damping_dipole_torch_df_CLIFF():
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
    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
    )

    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    Ks = [
        [1.14769962, 0.685558974, 0.685558974],
        [1.14769962, 0.685558974, 0.685558974],
    ]
    alphaA = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    alphaB = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    for n, r in df.iterrows():
        sapt0_elst = r["SAPT0 ELST ENERGY adz"]
        sapt0_ind = r["SAPT0 IND ENERGY adz"] * h2kcalmol
        mol = r["qcel_molecule"]
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(
            np.sum((monA.geometry[:, None] - monB.geometry) ** 2, axis=2)
        ).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r["q_A pbe0/atz"]
        muA = r["mu_A pbe0/atz"]
        # muA = np.zeros_like(muA)
        thetaA = r["theta_A pbe0/atz"]
        thetaA = np.zeros_like(thetaA)
        qB = r["q_B pbe0/atz"]
        muB = r["mu_B pbe0/atz"]
        # muB = np.zeros_like(muB)
        thetaB = r["theta_B pbe0/atz"]
        thetaB = np.zeros_like(thetaB)
        thetaA = np.zeros_like(thetaA)
        thetaB = np.zeros_like(thetaB)

        (
            ref_elst_q,
            E_qqs_q,
            E_qus_q,
            E_uus_q,
            E_qQs_q,
            E_uQs_q,
            E_QQs_q,
            E_ZA_ZBs_q,
            E_ZA_MBs_q,
            E_ZB_MAs_q,
        ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
            mol_dimer=mol,
            qA=qA,
            qB=qB,
            muA=muA,
            muB=muB,
            # muA=np.zeros_like(muA),
            # muB=np.zeros_like(muB),
            thetaA=thetaA,
            thetaB=thetaB,
            # thetaA=np.zeros_like(thetaA),
            # thetaB=np.zeros_like(thetaB),
            alphaA=alphaA,
            alphaB=alphaB,
            traceless=False,
            amoeba_eq=True,
            match_cliff=False,
        )
        elst = ref_elst_q
        pred, pair_elst, pair_ind = ap3.predict_qcel_mols(
            [mol], batch_size=1, return_classical_pairs=True
        )
        ap3_elst = np.sum(pair_elst[0])
        ap3_ind = np.sum(pair_ind[0])
        print(f"Distance between monomers: {dist * bohr2angstrom:.2f} A")
        print(f"SAPT ELST   = {sapt0_elst:.6f} kcal/mol")
        print(f"ELST Pred   = {elst:.6f} kcal/mol")
        print(f"AP3  ELST   = {ap3_elst:.6f} kcal/mol")


def test_elst_damping_dipole_torch_df_AMOEBA():
    """
    Compare AMOEBA-damped electrostatic and induction predictions to SAPT references and AP3 model outputs for water dimer entries, printing distances and energy comparisons.
    
    Builds AMOEBA atom-type and dimer parameter models, loads water dimer test data, computes reference AMOEBA-damped energies via eval_qcel_dimer_individual_components, obtains AP3 predictions, and prints SAPT, reference ELST, and AP3 ELST for each dimer entry.
    """
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
    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
    )
    print(atom_type_elst_model)

    df = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    Ks = [
        [1.14769962, 0.685558974, 0.685558974],
        [1.14769962, 0.685558974, 0.685558974],
    ]
    alphaA = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    alphaB = np.array([2.05109221104216, 1.65393856475232, 1.65393856475232])
    for n, r in df.iterrows():
        sapt0_elst = r["SAPT0 ELST ENERGY adz"]
        sapt0_ind = r["SAPT0 IND ENERGY adz"] * h2kcalmol
        mol = r["qcel_molecule"]
        monA = mol.get_fragment(0).copy()
        monB = mol.get_fragment(1).copy()
        dist = np.sqrt(
            np.sum((monA.geometry[:, None] - monB.geometry) ** 2, axis=2)
        ).min()
        bohr2angstrom = qcel.constants.conversion_factor("bohr", "angstrom")
        qA = r["q_A pbe0/atz"]
        muA = r["mu_A pbe0/atz"]
        # muA = np.zeros_like(muA)
        thetaA = r["theta_A pbe0/atz"]
        thetaA = np.zeros_like(thetaA)
        qB = r["q_B pbe0/atz"]
        muB = r["mu_B pbe0/atz"]
        # muB = np.zeros_like(muB)
        thetaB = r["theta_B pbe0/atz"]
        thetaB = np.zeros_like(thetaB)
        thetaA = np.zeros_like(thetaA)
        thetaB = np.zeros_like(thetaB)

        (
            ref_elst_q,
            E_qqs_q,
            E_qus_q,
            E_uus_q,
            E_qQs_q,
            E_uQs_q,
            E_QQs_q,
            E_ZA_ZBs_q,
            E_ZA_MBs_q,
            E_ZB_MAs_q,
        ) = apnet_pt.multipole.eval_qcel_dimer_individual_components(
            mol_dimer=mol,
            qA=qA,
            qB=qB,
            muA=muA,
            muB=muB,
            # muA=np.zeros_like(muA),
            # muB=np.zeros_like(muB),
            thetaA=thetaA,
            thetaB=thetaB,
            # thetaA=np.zeros_like(thetaA),
            # thetaB=np.zeros_like(thetaB),
            alphaA=alphaA,
            alphaB=alphaB,
            traceless=False,
            amoeba_eq=True,
            match_cliff=False,
        )
        elst = ref_elst_q
        pred, pair_elst, pair_ind = ap3.predict_qcel_mols(
            [mol], batch_size=1, return_classical_pairs=True
        )
        ap3_elst = np.sum(pair_elst[0])
        ap3_ind = np.sum(pair_ind[0])
        print(f"Distance between monomers: {dist * bohr2angstrom:.2f} A")
        print(f"SAPT ELST   = {sapt0_elst:.6f} kcal/mol")
        print(f"ELST Pred   = {elst:.6f} kcal/mol")
        print(f"AP3  ELST   = {ap3_elst:.6f} kcal/mol")


def test_elst_damping_AMOEBA_mtp_mtp_torch():
    """
    Run the AMOEBA (GORDON1) electrostatic damping test on a water dimer and compare results to reference data.
    
    Performs an AMOEBA (GORDON1) damped multipole electrostatics calculation for a water dimer using reference multipoles and damping parameters, evaluates low-level damping factors, and compares the computed AMOEBA-damped energy with CLIFF/AMOEBA and SAPT0 reference values. Also performs basic sanity checks to ensure the computed tensor contains no NaN or infinite values.
    """

    # Load reference data with AMOEBA values
    ref_data = pd.read_pickle(
        current_file_path
        + os.sep
        + os.path.join("dataset_data", "amoeba_water_dimer_ref.pkl")
    )
    mol = ref_data["qcel_molecule"]
    print(mol.to_string('xyz'))
    qA = ref_data["q_A pbe0/atz"]
    muA = ref_data["mu_A pbe0/atz"]
    thetaA = ref_data["theta_A pbe0/atz"]
    qB = ref_data["q_B pbe0/atz"]
    muB = ref_data["mu_B pbe0/atz"]
    thetaB = ref_data["theta_B pbe0/atz"]

    # AMOEBA damping parameters from reference data
    Ka = ref_data["alpha_A"]  # [4.7004, 4.7441, 4.7441] for O, H, H
    Kb = ref_data["alpha_B"]  # [4.7004, 4.7441, 4.7441] for O, H, H

    # Reference values for comparison
    ref_amoeba_elst = float(ref_data["amoeba_elst_hippo"])  # -7.2136 kcal/mol
    ref_sapt0_elst = float(ref_data["SAPT0 ELST kcalmol"])  # -7.1946 kcal/mol

    np.set_printoptions(precision=6)
    torch.set_printoptions(precision=6)

    # Create dimer batch
    dimer_batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut_im=99999.0, dimer_ind=0
            )
        ]
    )

    # Set up batch data
    dimer_batch.Ka = torch.tensor(Ka, dtype=torch.float32)
    dimer_batch.Kb = torch.tensor(Kb, dtype=torch.float32)
    dimer_batch.qA = torch.tensor(qA, dtype=torch.float32)
    dimer_batch.qB = torch.tensor(qB, dtype=torch.float32)
    dimer_batch.muA = torch.tensor(muA, dtype=torch.float32)
    dimer_batch.muB = torch.tensor(muB, dtype=torch.float32)
    dimer_batch.quadA = torch.tensor(thetaA, dtype=torch.float32)
    dimer_batch.quadB = torch.tensor(thetaB, dtype=torch.float32)

    # Call the AMOEBA damping function
    torch_elst_amoeba = apnet_pt.AtomPairwiseModels.mtp_mtp.mtp_elst_damping_AMOEBA(
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

    total_elst = torch.sum(torch_elst_amoeba).item()
    print(f"\n=== AMOEBA (GORDON1) Electrostatic Damping Test ===")
    print(f"Damping parameters: Ka = {Ka}, Kb = {Kb}")
    print(f"Total AMOEBA damped elst (torch) = {total_elst:.6f} kcal/mol")
    print(f"Reference AMOEBA HIPPO = {ref_amoeba_elst:.6f} kcal/mol")
    print(f"Reference SAPT0 ELST   = {ref_sapt0_elst:.6f} kcal/mol")

    # Also test the low-level damping function directly
    from apnet_pt.AtomPairwiseModels.mtp_mtp import (
        elst_damping_AMOEBA_mtp_mtp_torch,
        get_distances,
    )
    from apnet_pt import constants

    # Get distances for testing
    dR_ang, dR_xyz_ang = get_distances(
        dimer_batch.RA,
        dimer_batch.RB,
        dimer_batch.e_ABsr_source,
        dimer_batch.e_ABsr_target,
    )
    dR = dR_ang / constants.au2ang

    # Call the damping function directly
    lam1, lam3, lam5 = elst_damping_AMOEBA_mtp_mtp_torch(
        dimer_batch.Ka,
        dimer_batch.Kb,
        dR,
        dimer_batch.e_ABsr_source,
        dimer_batch.e_ABsr_target,
    )

    print(f"\nLow-level damping factors GORDON1:")
    print(f"  lam1: {lam1}")
    print(f"  lam3: {lam3}")
    print(f"  lam5: {lam5}")

    # Basic sanity checks
    assert not torch.isnan(torch_elst_amoeba).any(), "NaN values in AMOEBA elst output"
    assert not torch.isinf(torch_elst_amoeba).any(), "Inf values in AMOEBA elst output"

    # Compare with CLIFF damping for reference
    torch_elst_cliff = apnet_pt.AtomPairwiseModels.mtp_mtp.mtp_elst_damping(
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
    total_elst_cliff = torch.sum(torch_elst_cliff).item()
    print(f"\nComparison with CLIFF (GORDON2) damping:")
    print(f"  AMOEBA (GORDON1) elst = {total_elst:.6f} kcal/mol")
    print(f"  Difference = {abs(total_elst - ref_sapt0_elst):.6f} kcal/mol")
    print(f"  CLIFF  (GORDON2) elst = {total_elst_cliff:.6f} kcal/mol")
    print(f"  Difference = {abs(total_elst_cliff - ref_sapt0_elst):.6f} kcal/mol")

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
    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
    )
    print(ref_data)
    energies = ap3.predict_qcel_mols([mol], batch_size=1)
    print(energies)
    energies = atom_type_elst_model.predict_qcel_mols_dimer([mol], batch_size=1)
    print(energies)
    monA, monB = atom_type_elst_model.predict_qcel_mols_monomer_props([mol], batch_size=1)
    # Compare with CLIFF damping for reference
    torch_elst_cliff = apnet_pt.AtomPairwiseModels.mtp_mtp.mtp_elst_damping(
        ZA=dimer_batch.ZA,
        RA=dimer_batch.RA,
        qA_0=dimer_batch.qA,
        muA=dimer_batch.muA,
        quadA=dimer_batch.quadA,
        Ka=monA[0][-1],
        ZB=dimer_batch.ZB,
        RB=dimer_batch.RB,
        qB_0=dimer_batch.qB,
        muB=dimer_batch.muB,
        quadB=dimer_batch.quadB,
        Kb=monB[0][-1],
        e_AB_source=dimer_batch.e_ABsr_source,
        e_AB_target=dimer_batch.e_ABsr_target,
    )
    lam1, lam3, lam5 = apnet_pt.AtomPairwiseModels.mtp_mtp.elst_damping_mtp_mtp_torch(
        dimer_batch.Ka,
        dimer_batch.Kb,
        dR,
        dimer_batch.e_ABsr_source,
        dimer_batch.e_ABsr_target,
    )

    print(f"\nLow-level damping factors GORDON1:")
    print(f"  lam1: {lam1}")
    print(f"  lam3: {lam3}")
    print(f"  lam5: {lam5}")
    total_elst_cliff = torch.sum(torch_elst_cliff).item()
    print(f"\nComparison with CLIFF (GORDON1) damping with AP3-DimerParams:")
    print(f"  CLIFF  (GORDON2) elst = {total_elst_cliff:.6f} kcal/mol")
    print(f"  Difference = {abs(total_elst_cliff - ref_sapt0_elst):.6f} kcal/mol")

    return


def test_induced_dipole_torch_intramolecular():
    # Load the monomer data
    df = pd.read_pickle(
        current_file_path + os.sep + os.path.join("dataset_data", "df_bz_meoh_mbis.pkl")
        # current_file_path + os.sep + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    # df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    r = df.iloc[0]
    mol = r["qcel_molecule"].get_fragment(0)
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    vrA = r["vol_ratios_A pbe0/atz"].flatten()
    vwA = r["val_widths_A pbe0/atz"].flatten()
    thetaA = np.zeros_like(thetaA)
    # Create monomer batch with edge indices
    monomer_batch = apnet_pt.atomic_datasets.atomic_collate_update_no_target(
        [atomic_datasets.qcel_mon_to_pyg_data(mol, r_cut=99999.0)]
    )

    # Get atomic numbers and positions
    Z = torch.tensor(mol.atomic_numbers, dtype=torch.long)
    R = torch.tensor(mol.geometry, dtype=torch.float32) * constants.au2ang

    # Convert to torch tensors
    q_tensor = torch.tensor(qA, dtype=torch.float32)
    mu_tensor = torch.tensor(muA, dtype=torch.float32)
    quad_tensor = torch.zeros_like(torch.tensor(thetaA, dtype=torch.float32))
    vr_tensor = torch.tensor(vrA, dtype=torch.float32)
    vw_tensor = torch.tensor(vwA, dtype=torch.float32)

    # Get edge indices from the batch
    e_source = monomer_batch.edge_index[0]
    e_target = monomer_batch.edge_index[1]

    # Call the updated monomer_induced_dipole_torch function
    print("\n=== PyTorch Version ===")
    q_torch, mu_induced_torch, quad_torch = (
        apnet_pt.multipole.monomer_induced_dipole_torch(
            Z=Z,
            R=R,
            q=q_tensor,
            mu=mu_tensor,
            quad=quad_tensor,
            e_source=e_source,
            e_target=e_target,
            hirshfeld_volume_ratio=vr_tensor,
            valence_widths=vw_tensor,
            compute_energies=False,
            screening_distance=1.8,
            verbose=1,
        )
    )

    # Call the numpy version for comparison
    print("\n=== NumPy Version ===")
    q_numpy, mu_induced_numpy, quad_numpy = (
        apnet_pt.multipole.intramolecular_induced_dipole(
            qcel_mol=mol,
            q=qA,
            mu=muA,
            theta=thetaA,
            hirshfeld_volume_ratio=vrA,
            valence_widths=vwA,
            thole_damping_param_mutual=0.39,
            thole_damping_param_direct=0.34,
            heavy_atoms_only=True,
            screening_distance=1.8,
            compute_energies=False,
            verbose=1,
        )
    )

    # Compare results
    print("\n=== Comparison ===")
    q_torch_np = q_torch.detach().numpy()
    mu_torch_np = mu_induced_torch.detach().numpy()
    quad_torch_np = quad_torch.detach().numpy()

    print(f"Charges match: {np.allclose(q_torch_np, q_numpy, rtol=1e-4, atol=1e-6)}")
    print(f"Max charge difference: {np.abs(q_torch_np - q_numpy).max():.2e}")

    print(
        f"Induced dipoles match: {np.allclose(mu_torch_np, mu_induced_numpy, rtol=1e-4, atol=1e-6)}"
    )
    print(
        f"Max induced dipole difference: {np.abs(mu_torch_np - mu_induced_numpy).max():.2e}"
    )

    print(
        f"Quadrupoles match: {np.allclose(quad_torch_np, quad_numpy, rtol=1e-4, atol=1e-6)}"
    )
    print(f"Max quadrupole difference: {np.abs(quad_torch_np - quad_numpy).max():.2e}")

    # Assert they match within tolerance
    assert np.allclose(q_torch_np, q_numpy, rtol=1e-4, atol=1e-6), (
        "Charges don't match!"
    )
    assert np.allclose(mu_torch_np, mu_induced_numpy, rtol=1e-4, atol=1e-6), (
        "Induced dipoles don't match!"
    )
    assert np.allclose(quad_torch_np, quad_numpy, rtol=1e-4, atol=1e-6), (
        "Quadrupoles don't match!"
    )


if __name__ == "__main__":
    # test_induced_dipole_torch_intramolecular()
    # test_elst_damping_dipole_torch_df()
    # test_elst_multipoles_MTP_torch_damping()
    # test_elst_damping_dipole_torch_df()
    # test_elst_charge_dipole_qpole()
    # test_elst_multipoles()
    # test_classical_cliff()
    # test_elst_ameoba()
    # test_elst_damping()
    # test_elst_multipoles_MTP_torch()
    # test_elst_multipoles_MTP_torch_no_damping()
    # test_induced_dipole_bz_meoh()
    # test_induced_dipole_no_damping()
    # test_induced_dipole_no_damping()

    # test_induced_dipole_torch()
    # test_induced_dipole_torch_alphas()

    # test_induced_dipole()
    # test_induced_dipole_torch_df()
    test_elst_damping_AMOEBA_mtp_mtp_torch()
    # test_elst_multipoles_AP2()
