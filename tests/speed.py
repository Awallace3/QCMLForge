import torch
import timeit
import apnet_pt
from apnet_pt.torch_util import set_weights_to_value
import numpy as np
import qcelemental as qcel
import os
from apnet_pt.pt_datasets.ap2_fused_ds import (
    ap2_fused_module_dataset,
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


def main():
    # First check that architecture isn't broken
    test_ap2_fused_architecture()
    batch_size = 16
    atomic_batch_size = 16
    datapoint_storage_n_objects = 4
    qcel_molecules = [mol_dimer] * 32
    energy_labels = [[1.0] * 4 for _ in range(len(qcel_molecules))]
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel().set_pretrained_model(
        model_id=0
    )
    ap2 = APNet2_AM_Model()
    ap2.set_all_weights_to_value(0.01)
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

    def train():
        apnet_pt.torch_util.set_weights_to_value(ap2.model, 0.01)
        ap2.train(
            ds,
            n_epochs=5,
            skip_compile=True,
            transfer_learning=False,
            lr=0.0005,
        )

    v = timeit.timeit(train, number=3)
    print(f"\nSpeed test took {v:.2f} seconds")
    return


if __name__ == "__main__":
    main()
