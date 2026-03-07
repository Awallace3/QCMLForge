import apnet_pt
import numpy as np
import qcelemental as qcel
import torch
import os
from apnet_pt.pt_datasets.ap2_fused_ds import (
    ap2_fused_module_dataset,
    ap2_fused_collate_update,
    APNet2_fused_DataLoader,
)
from apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_Model
from glob import glob

torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../src/apnet_pt/models/am_ensemble/am_0.pt"
am_hf_path = f"{current_file_path}/../src/apnet_pt/models/am_hf_ensemble/am_0.pt"

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
print(dict(mol_dimer))

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

mol_fsapt = qcel.models.Molecule.from_data("""
0 1
C   11.54100       27.68600       13.69600
H   12.45900       27.15000       13.44600
C   10.79000       27.96500       12.40600
H   10.55700       27.01400       11.92400
H   9.879000       28.51400       12.64300
H   11.44300       28.56800       11.76200
H   10.90337       27.06487       14.34224
H   11.78789       28.62476       14.21347
--
0 1
C   10.60200       24.81800       6.466000
O   10.95600       23.84000       7.103000
N   10.17800       25.94300       7.070000
C   10.09100       26.25600       8.476000
C   9.372000       27.59000       8.640000
C   11.44600       26.35600       9.091000
C   9.333000       25.25000       9.282000
H   9.874000       26.68900       6.497000
H   9.908000       28.37100       8.093000
H   8.364000       27.46400       8.233000
H   9.317000       27.84600       9.706000
H   9.807000       24.28200       9.160000
H   9.371000       25.57400       10.32900
H   8.328000       25.26700       8.900000
H   11.28800       26.57600       10.14400
H   11.97000       27.14900       8.585000
H   11.93200       25.39300       8.957000
H   10.61998       24.85900       5.366911
units angstrom

symmetry c1
no_reorient
no_com
""")


def set_weights_to_value(model, value=0.9):
    """Sets all weights and biases in the model to a specific value."""
    with torch.no_grad():  # Disable gradient tracking
        for param in model.parameters():
            param.fill_(value)  # Set all elements to the given value
    return


def test_ap2_fused_dataset_size():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    collate = ap2_fused_collate_update
    ds = ap2_fused_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
    )
    print()
    print(ds)

    train_loader = APNet2_fused_DataLoader(
        dataset=ds,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        print(i)
        cnt += i.y.shape[0]
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_8*.pt"):
        os.remove(i)
    assert ds_labels == cnt, f"Expected {len(ds)} points, but got {cnt} points"


def test_ap2_fused_train_qcel_molecules_in_memory():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 6
    qcel_molecules = [mol_element] * 16
    qcel_molecules.extend([mol_dimer] * 15)
    energy_labels = [[1.0] * 4 for _ in range(len(qcel_molecules))]
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel().set_pretrained_model(
        model_id=0
    )
    ap2 = APNet2_AM_Model()
    apnet_pt.torch_util.set_weights_to_value(atom_model.model, 0.01)
    ds = ap2_fused_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atom_model=atom_model,
        atomic_batch_size=atomic_batch_size,
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
    ap2.train(
        ds,
        n_epochs=1,
        skip_compile=True,
        transfer_learning=False,
        lr=0.005,
    )
    total_params = sum(param.numel() for param in ap2.model.parameters())
    print(f"ap2 total parameters: {total_params}")
    # This also tests to make sure only best model is returned
    v_0 = ap2.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    ap2.train(
        ds,
        n_epochs=1,
        skip_compile=True,
        transfer_learning=False,
        lr=0.05,
    )
    v = ap2.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    print(v_0, v)
    assert np.allclose(v_0, v, atol=1e-6)


def test_ap2_fused_architecture():
    target_energies = [
        -3.402202606201171875e01,  # ELST
        4.996978532290086150e-06,  # EXCH
        4.996978532290086150e-06,  # INDU
        4.996978532290086150e-06,  # DISP
    ]
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    set_weights_to_value(atom_model.model, 0.0001)
    pair_model = apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
        use_GPU=False,
    )
    output = pair_model.predict_qcel_mols([mol3], batch_size=1)
    set_weights_to_value(pair_model.model, 0.0001)
    output = pair_model.predict_qcel_mols([mol3], batch_size=1)
    print(target_energies)
    print(output[0])
    assert np.allclose(output[0], target_energies, atol=1e-6)


def test_ap2_fused_ensemble_water_dimer():
    import torch
    import pandas as pd

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
        sapt0_elst = r["SAPT0 ELST ENERGY adz"]
        sapt0_total = r["SAPT0 TOTAL ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        sapt0_exch = r["SAPT0 EXCH ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        sapt0_ind = r["SAPT0 IND ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        sapt0_disp = r["SAPT0 DISP ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        mol = r["qcel_molecule"]
        interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
            [mol],
            compile=False,
            batch_size=2,
            ap2_fused=True,
        )
        print(
            f"TOTAL = {sapt0_total:.6f}\n ELST = {sapt0_elst:.6f}\n EXCH = {
                sapt0_exch:.6f}\n DISP = {sapt0_disp:.6f}\n IND = {sapt0_ind:.6f}"
        )
        print(interaction_energies)


def test_ap2_fused_ensemble_water_dimer():
    import torch
    import pandas as pd

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
        sapt0_elst = r["SAPT0 ELST ENERGY adz"]
        sapt0_total = r["SAPT0 TOTAL ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        sapt0_exch = r["SAPT0 EXCH ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        sapt0_ind = r["SAPT0 IND ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        sapt0_disp = r["SAPT0 DISP ENERGY adz"] * qcel.constants.conversion_factor(
            "hartree", "kcal/mol"
        )
        mol = r["qcel_molecule"]
        interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
            [mol],
            compile=False,
            batch_size=2,
            ap2_fused=True,
        )
        print(
            f"TOTAL = {sapt0_total:.6f}\n ELST = {sapt0_elst:.6f}\n EXCH = {sapt0_exch:.6f}\n DISP = {sapt0_disp:.6f}\n IND = {sapt0_ind:.6f}"
        )
        print(interaction_energies)


if __name__ == "__main__":
    # test_ap2_fused_dataset_size()
    # test_ap2_fused_architecture()
    # test_ap2_fused_train_qcel_molecules_in_memory()
    # test_ap2_fused_dataset_size()
    # test_ap2_fused_ensemble_water_dimer()
    test_ap2_fused_train_qcel_molecules_in_memory()
