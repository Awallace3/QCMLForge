import apnet_pt
import numpy as np
import qcelemental as qcel
import torch
import pytest

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
water = qcel.models.Molecule.from_data("""

0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
units angstrom
        """)


def set_weights_to_value(model, value=0.9):
    """Sets all weights and biases in the model to a specific value."""
    with torch.no_grad():  # Disable gradient tracking
        for param in model.parameters():
            param.fill_(value)  # Set all elements to the given value


@pytest.mark.skip(reason="Models still being updated... run only for development reasons.")
def test_ap3_architecture():
    target_energies = [
        -3.402202606201171875e01,  # ELST
        4.996978532290086150e-06,  # EXCH
        4.996978532290086150e-06,  # INDU
        4.996978532290086150e-06,  # DISP
    ]
    atom_model = apnet_pt.AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel(
        ds_root=None,
        ignore_database_null=True,
    )
    set_weights_to_value(atom_model.model, 0.0001)
    pair_model = apnet_pt.AtomPairwiseModels.apnet3.APNet3Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
    )
    output = pair_model.predict_qcel_mols([mol3], batch_size=1)
    set_weights_to_value(pair_model.model, 0.0001)
    output = pair_model.predict_qcel_mols([mol3], batch_size=1)
    print(target_energies)
    print(output[0])
    assert np.allclose(output[0], target_energies, atol=1e-6)


@pytest.mark.skip(reason="Models still being updated... run only for development reasons.")
def test_ap3_exch():
    atom_model = apnet_pt.AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel(
        ds_root=None,
        ignore_database_null=True,
        pre_trained_model_path="./models/am_hf_ensemble/am_4.pt",
    )
    pair_model = apnet_pt.AtomPairwiseModels.apnet3.APNet3Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
    )
    output = pair_model.predict_qcel_mols([water], batch_size=1)
    set_weights_to_value(pair_model.model, 0.01)
    output = pair_model.predict_qcel_mols([water, water], batch_size=2)
    ref_energies = np.array(
        [
            [5.14170970e-04],
            [5.14170213e-04],
        ]
    )
    assert np.allclose(output[:, 1], ref_energies, atol=1e-6)
    return


@pytest.mark.skip(reason="Models still being updated... run only for development reasons.")
def test_ap3_indu():
    atom_model = apnet_pt.AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel(
        ds_root=None,
        ignore_database_null=True,
        pre_trained_model_path="./models/am_hf_ensemble/am_4.pt",
    )
    pair_model = apnet_pt.AtomPairwiseModels.apnet3.APNet3Model(
        atom_model=atom_model.model,
        ignore_database_null=True,
    )
    au2ang = qcel.constants.conversion_factor("bohr", "angstrom")
    print("water geometry in angstrom")
    print(water.geometry * au2ang)
    print()
    output = pair_model.predict_qcel_mols([water], batch_size=1)
    set_weights_to_value(pair_model.model, 0.01)
    output = pair_model.predict_qcel_mols([water, water], batch_size=2)
    ref_energies = np.array(
        [
            [5.14170970e-04],
            [5.14170213e-04],
        ]
    )
    assert np.allclose(output[:, 1], ref_energies, atol=1e-6)
    return


if __name__ == "__main__":
    # test_ap3_architecture()
    # test_ap3_exch()
    test_ap3_indu()
