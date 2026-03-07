import apnet_pt
import qcelemental as qcel
import torch
import os
import numpy as np
import pytest

mol_mon = qcel.models.Molecule.from_data("""0 1
16  -0.8795  -2.0832  -0.5531
7   -0.2959  -1.8177   1.0312
7    0.5447  -0.7201   1.0401
6    0.7089  -0.1380  -0.1269
6    0.0093  -0.7249  -1.1722
1    1.3541   0.7291  -0.1989
1   -0.0341  -0.4523  -2.2196
units angstrom
""")

mol_dimer = qcel.models.Molecule.from_data("""
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


def test_am_ensemble():
    print("Testing AM ensemble...")
    ref = torch.load(
        os.path.join(os.path.dirname(__file__), "dataset_data/am_ensemble_test.pt"),
        weights_only=False,
    )

    mols = [mol_mon for _ in range(3)]
    multipoles = apnet_pt.pretrained_models.atom_model_predict(
        mols,
        compile=False,
        batch_size=2,
    )
    q_ref = ref[0]
    q = multipoles[0]
    assert np.allclose(q, q_ref, atol=1e-6)
    d_ref = ref[1]
    d = multipoles[1]
    assert np.allclose(d, d_ref, atol=1e-6)
    qp_ref = ref[2]
    qp = multipoles[2]
    assert np.allclose(qp, qp_ref, atol=1e-6)


@pytest.mark.skip(reason="ap2 ensemble models not available in download, huggingface shift")
def test_ap2_ensemble():
    print("Testing AP2 ensemble...")
    ref = torch.load(
        os.path.join(os.path.dirname(__file__), "dataset_data/ap2_ensemble_test.pt"),
        weights_only=False,
    )

    mols = [mol_dimer for _ in range(3)]
    interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
        mols,
        compile=False,
        batch_size=2,
    )
    torch.save(interaction_energies, os.path.join(os.path.dirname(
        __file__), "dataset_data/ap2_ensemble_test.pt"))
    print(interaction_energies)
    print(ref)
    assert np.allclose(interaction_energies, ref, atol=1e-3)


def test_am_ensemble_compile():
    print("Testing AM ensemble...")
    ref = torch.load(
        os.path.join(os.path.dirname(__file__), "dataset_data/am_ensemble_test.pt"),
        weights_only=False,

    )

    mols = [mol_mon for _ in range(3)]
    multipoles = apnet_pt.pretrained_models.atom_model_predict(
        mols,
        compile=True,
        batch_size=2,
    )
    q_ref = ref[0]
    q = multipoles[0]
    assert np.allclose(q, q_ref, atol=1e-6)
    d_ref = ref[1]
    d = multipoles[1]
    assert np.allclose(d, d_ref, atol=1e-6)
    qp_ref = ref[2]
    qp = multipoles[2]
    assert np.allclose(qp, qp_ref, atol=1e-6)


@pytest.mark.skip("Models no longer available vi PyPI")
def test_ap2_ensemble_compile():
    print("Testing AP2 ensemble...")
    ref = torch.load(
        os.path.join(os.path.dirname(__file__), "dataset_data/ap2_ensemble_test.pt"),
        weights_only=False,
    )

    mols = [mol_dimer for _ in range(3)]
    interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
        mols,
        compile=True,
        batch_size=2,
    )
    assert np.allclose(interaction_energies, ref, atol=1e-5)


@pytest.mark.skip(reason="ap2 ensemble models not available in download, huggingface shift")
def test_ap2_ensemble_predict_pairs():
    _, pairs, df = apnet_pt.pretrained_models.apnet2_model_predict_pairs(
        [
            mol_fsapt,
            mol_fsapt,
            mol_fsapt,
        ],
        compile=False,
        batch_size=2,
        fAs=[{
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        } for _ in range(3)],
        fBs=[{
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        } for _ in range(3)],
        print_results=True,
    )
    print(df)
    ref = {
        "Methyl1_A-Peptide_B": {
            'total':  0.140302,
            'elst': 0.160579,
            'exch': 0.004259,
            'indu': -0.007032,
            'disp': -0.017503,
        }
    }
    for k, v in ref.items():
        # get row where fA-fB equals 'Methyl1_A-Peptide_B'
        row = df[df['fA-fB'] == k]
        assert np.isclose(row['total'].values[0], v['total'], atol=1e-6)
        assert np.isclose(row['elst'].values[0], v['elst'], atol=1e-6)
        assert np.isclose(row['exch'].values[0], v['exch'], atol=1e-6)
        assert np.isclose(row['indu'].values[0], v['indu'], atol=1e-6)
        assert np.isclose(row['disp'].values[0], v['disp'], atol=1e-6)
    return


def test_ap2_fused_ensemble():
    print("Testing AP2 ensemble...")
    ref = torch.load(
        os.path.join(os.path.dirname(__file__), "dataset_data/ap2_fused_ensemble_test.pt"),
        weights_only=False,
    )

    mols = [mol_dimer for _ in range(3)]
    interaction_energies = apnet_pt.pretrained_models.apnet2_model_predict(
        mols,
        compile=False,
        batch_size=2,
        ap2_fused=True,
    )
    torch.save(interaction_energies, os.path.join(os.path.dirname(
        __file__), "dataset_data/ap2_fused_ensemble_test.pt"))
    assert np.allclose(interaction_energies, ref, atol=1e-5), f"{ref=}\n{interaction_energies = }"


def test_ap2_fused_ensemble_predict_pairs():
    _, pairs, df = apnet_pt.pretrained_models.apnet2_model_predict_pairs(
        [mol_fsapt],
        compile=False,
        batch_size=2,
        fAs=[{
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        }],
        fBs=[{
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        }],
        print_results=True,
        ap2_fused=True,
    )
    print(df)
    """
==> AP2-FSAPT <==
monA-monB full IE: [-0.539679   -1.29448098  4.26433879 -0.55210597 -2.95743084]

 Frag1      Frag2         Elst       Exch       Ind        Disp       Total
            
Methyl1_A  Peptide_B    0.168963  -0.003577   0.001630  -0.012463   0.154552
Methyl1_A  T-Butyl_B   -0.423034   0.111288  -0.030383  -0.334297  -0.676425
Methyl2_A  Peptide_B   -0.212438  -0.042306  -0.008193  -0.039332  -0.302270
Methyl2_A  T-Butyl_B   -0.827972   4.198934  -0.515159  -2.571339   0.284464
                 fA-fB     total      elst      exch      indu      disp
0  Methyl1_A-Peptide_B  0.154552  0.168963 -0.003577  0.001630 -0.012463
1  Methyl1_A-T-Butyl_B -0.676425 -0.423034  0.111288 -0.030383 -0.334297
2  Methyl2_A-Peptide_B -0.302270 -0.212438 -0.042306 -0.008193 -0.039332
3  Methyl2_A-T-Butyl_B  0.284464 -0.827972  4.198934 -0.515159 -2.571339
    """
    _, pairs, df = apnet_pt.pretrained_models.apnet2_model_predict_pairs(
        [
            mol_fsapt,
            mol_fsapt,
            mol_fsapt,
        ],
        compile=False,
        batch_size=2,
        fAs=[{
            "Methyl1_A": [1, 2, 7, 8],
            "Methyl2_A": [3, 4, 5, 6],
        } for _ in range(3)],
        fBs=[{
            "Peptide_B": [9, 10, 11, 16, 26],
            "T-Butyl_B": [12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        } for _ in range(3)],
        print_results=True,
        ap2_fused=True,
    )
    print(df)
    ref = {
        "Methyl1_A-Peptide_B": {
            'total': 0.154551,
            'elst': 0.168962,
            'exch': -0.003577,
            'indu': 0.001630,
            'disp': -0.012463,
        }
    }
    for k, v in ref.items():
        # get row where fA-fB equals 'Methyl1_A-Peptide_B'
        row = df[df['fA-fB'] == k]
        print(f"{row['total'].values[0]=}, {v['total']=}")
        assert np.isclose(row['total'].values[0], v['total'], atol=1e-5), f"{row['total'].values[0]=}, {v['total']=}"
        assert np.isclose(row['elst'].values[0], v['elst'], atol=1e-5), f"{row['elst'].values[0]=}, {v['elst']=}"
        assert np.isclose(row['exch'].values[0], v['exch'], atol=1e-5), f"{row['exch'].values[0]=}, {v['exch']=}"
        assert np.isclose(row['indu'].values[0], v['indu'], atol=1e-5), f"{row['indu'].values[0]=}, {v['indu']=}"
        assert np.isclose(row['disp'].values[0], v['disp'], atol=1e-5), f"{row['disp'].values[0]=}, {v['disp']=}"
    return


if __name__ == "__main__":
    # test_am_ensemble()
    # test_ap2_ensemble()
    # test_ap2_ensemble_predict_pairs()
    # test_ap2_fused_ensemble()
    # test_ap2_fused_ensemble_predict_pairs()
    test_ap2_fused_ensemble_predict_pairs()
