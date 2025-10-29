from . import AtomPairwiseModels
from . import AtomModels
from . import atomic_datasets
from qcelemental.models.molecule import Molecule
import os
import numpy as np
import copy
from importlib import resources
from apnet_pt.pt_datasets.dapnet_ds import clean_str_for_filename
import pandas as pd

# model_dir = os.path.dirname(os.path.realpath(__file__)) + "/models/"
model_dir = resources.files("apnet_pt").joinpath("models")


def atom_model_predict(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 3,
    return_mol_arrays: bool = True,
):
    num_models = 5
    am = AtomModels.ap2_atom_model.AtomModel(
        # pre_trained_model_path=f"{model_dir}am_ensemble/am_0.pt",
        pre_trained_model_path=resources.files("apnet_pt").joinpath(
            "models", "am_ensemble", "am_0.pt"
        ),
    )
    if compile:
        print("Compiling models...")
        am.compile_model()
    models = [copy.deepcopy(am) for _ in range(num_models)]
    for i in range(1, num_models):
        models[i].set_pretrained_model(
            # model_path=f"{model_dir}am_ensemble/am_{i}.pt",
            model_path=resources.files("apnet_pt").joinpath(
                "models", "am_ensemble", f"am_{i}.pt"
            ),
        )
    print("Processing mols...")
    data = [
        atomic_datasets.qcel_mon_to_pyg_data(mol, r_cut=am.model.r_cut) for mol in mols
    ]
    print(data)
    batched_data = [
        atomic_datasets.atomic_collate_update_no_target(data[i : i + batch_size])
        for i in range(0, len(data), batch_size)
    ]
    print(f"Number of batches: {len(batched_data)}")
    print(f"{batched_data = }")
    print(f"{batched_data[0].x = }")
    print("Predicting...")
    atom_count = sum([len(d.x) for d in data])
    pred_qs = np.zeros((atom_count))
    pred_ds = np.zeros((atom_count, 3))
    pred_qps = np.zeros((atom_count, 3, 3))
    atom_idx = 0
    mol_ids = []
    for batch in batched_data:
        # Intermediates which get averaged from num_models
        qs_t = np.zeros((len(batch.x)))
        ds_t = np.zeros((len(batch.x), 3))
        qps_t = np.zeros((len(batch.x), 3, 3))
        for i in range(num_models):
            q, d, qp, _ = models[i].predict_multipoles_batch(
                batch,
                isolate_predictions=False,
            )
            qs_t += q.numpy()
            ds_t += d.numpy()
            qps_t += qp.numpy()
        qs_t /= num_models
        ds_t /= num_models
        qps_t /= num_models
        pred_qs[atom_idx : atom_idx + len(batch.x)] = qs_t
        pred_ds[atom_idx : atom_idx + len(batch.x)] = ds_t
        pred_qps[atom_idx : atom_idx + len(batch.x)] = qps_t
        unique_values, repeats = np.unique(
            [batch.molecule_ind[i] for i in range(len(batch.molecule_ind))],
            return_counts=True,
        )
        mol_id_ranges = []
        for i in repeats:
            mol_id_ranges.append(int(i) + atom_idx)
            atom_idx += int(i)
        mol_ids.extend(mol_id_ranges)
    if return_mol_arrays:
        # Drop the final mol_ids because the split will take the full last
        # slice
        pred_qs = np.split(pred_qs, mol_ids[:-1])
        pred_ds = np.split(pred_ds, mol_ids[:-1])
        pred_qps = np.split(pred_qps, mol_ids[:-1])
        return pred_qs, pred_ds, pred_qps
    return pred_qs, pred_ds, pred_qps, mol_ids


def apnet2_model_predict(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 16,
    ensemble_model_dir: str = model_dir,
    ap2_fused: bool = False,
):
    if ap2_fused:
        num_models = 4
        additional_models_start = 2
        ap2 = AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
            pre_trained_model_path=resources.files("apnet_pt").joinpath(
                "models", "ap2-fused_ensemble", "ap2_1.pt"
            )
        )
    else:
        num_models = 5
        additional_models_start = 1
        ap2 = AtomPairwiseModels.apnet2.APNet2Model(
            pre_trained_model_path=resources.files("apnet_pt").joinpath(
                "models", "ap2_ensemble", "ap2_0.pt"
            ),
            atom_model_pre_trained_path=resources.files("apnet_pt").joinpath(
                "models", "am_ensemble", "am_0.pt"
            ),
        )
    if compile:
        print("Compiling models...")
        ap2.compile_model()
    models = [copy.deepcopy(ap2) for _ in range(num_models)]
    for i in range(additional_models_start, num_models):
        if ap2_fused:
            models[i].set_pretrained_model(
                ap2_model_path=resources.files("apnet_pt").joinpath(
                    "models", "ap2-fused_ensemble", f"ap2_{i}.pt"
                )
            )
        else:
            models[i].set_pretrained_model(
                ap2_model_path=resources.files("apnet_pt").joinpath(
                    "models", "ap2_ensemble", f"ap2_{i}.pt"
                ),
                am_model_path=resources.files("apnet_pt").joinpath(
                    "models", "am_ensemble", f"am_{i}.pt"
                ),
            )
    pred_IEs = np.zeros((len(mols), 5))
    print("Processing mols...")
    for i in range(num_models):
        IEs = models[i].predict_qcel_mols(
            mols,
            batch_size=batch_size,
        )
        pred_IEs[:, 1:] += IEs
        pred_IEs[:, 0] += np.sum(IEs, axis=1)
    pred_IEs /= num_models
    return pred_IEs


def apnet2_model_predict_pairs(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 16,
    ensemble_model_dir: str = model_dir,
    fAs: [{str: [int]}] = None,
    fBs: [{str: [int]}] = None,
    print_results: bool = False,
    ap2_fused: bool = True,
):
    """
    Predicts AP2 pairwise energies that correspond to an FSAPT calculation. fA
    and fB are LISTS of dictionaries that specify the atom indices for fragment
    A and B to sum their contributions. The syntax is identical to the Psi4
    FSAPT updates from https://github.com/psi4/psi4/pull/3222
    """
    assert fAs is not None, (
        "fAs must be provided. Example: [{'Methyl1_A': [1, 2, 7, 8], 'Methyl2_A': [3, 4, 5, 6]}...]"
    )
    assert fBs is not None, (
        "fBs must be provided, Example: [{'Peptide_B': [9, 10, 11, 16, 26], 'T-Butyl_B': [12, 13, 14, 15]}...]"
    )
    if ap2_fused:
        # Note: experimental, not finalized
        additional_models_start = 2
        num_models = 4
        ap2 = AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
            pre_trained_model_path=resources.files("apnet_pt").joinpath(
                "models", "ap2-fused_ensemble", "ap2_1.pt"
            )
        )
    else:
        additional_models_start = 2
        num_models = 5
        ap2 = AtomPairwiseModels.apnet2.APNet2Model(
            pre_trained_model_path=resources.files("apnet_pt").joinpath(
                "models", "ap2_ensemble", "ap2_0.pt"
            ),
            atom_model_pre_trained_path=resources.files("apnet_pt").joinpath(
                "models", "am_ensemble", "am_0.pt"
            ),
        )
    if compile:
        print("Compiling models...")
        ap2.compile_model()
    models = [copy.deepcopy(ap2) for _ in range(num_models)]
    for i in range(additional_models_start, num_models):
        if ap2_fused:
            models[i].set_pretrained_model(
                ap2_model_path=resources.files("apnet_pt").joinpath(
                    "models", "ap2-fused_ensemble", f"ap2_{i}.pt"
                )
            )
        else:
            models[i].set_pretrained_model(
                ap2_model_path=resources.files("apnet_pt").joinpath(
                    "models", "ap2_ensemble", f"ap2_{i}.pt"
                ),
                am_model_path=resources.files("apnet_pt").joinpath(
                    "models", "am_ensemble", f"am_{i}.pt"
                ),
            )
    pred_IEs = np.zeros((len(mols), 5))
    print("Processing mols...")
    IEs, pairwise_energies = models[0].predict_qcel_mols(
        mols,
        batch_size=batch_size,
        return_pairs=True,
    )
    pred_IEs[:, 1:] += IEs
    pred_IEs[:, 0] += np.sum(IEs, axis=1)
    for i in range(1, num_models):
        IEs, pairs = models[i].predict_qcel_mols(
            mols,
            batch_size=batch_size,
            return_pairs=True,
        )
        for i in range(len(pairs)):
            for j in range(len(pairs[i])):
                pairwise_energies[i][j] += pairs[i][j]
        pred_IEs[:, 1:] += IEs
        pred_IEs[:, 0] += np.sum(IEs, axis=1)
    for i in range(len(pairs)):
        for j in range(len(pairs[i])):
            pairwise_energies[i][j] /= num_models
    pred_IEs /= num_models

    data_dict = {
        # 'mol': [],
        "fA-fB": [],
        "total": [],
        "elst": [],
        "exch": [],
        "indu": [],
        "disp": [],
    }
    for i, mol in enumerate(mols):
        if print_results:
            # Analyze results
            print(f"")
            header = f"""==> AP2-FSAPT <==
monA-monB full IE: {pred_IEs[i]}

 Frag1      Frag2         Elst       Exch       Ind        Disp       Total
            """
            print(header)
        monA = mol.get_fragment([0])
        nA = len(monA.atomic_numbers)
        for kA, vA in fAs[i].items():
            for kB, vB in fBs[i].items():
                elst_sum = 0.0
                exch_sum = 0.0
                indu_sum = 0.0
                disp_sum = 0.0
                total_sum = 0.0
                for iA in vA:
                    for iB in vB:
                        # Subtract 1 to convert to 0-based indexing
                        elst_sum += pairwise_energies[i][0, iA - 1, iB - nA - 1]
                        exch_sum += pairwise_energies[i][1, iA - 1, iB - nA - 1]
                        indu_sum += pairwise_energies[i][2, iA - 1, iB - nA - 1]
                        disp_sum += pairwise_energies[i][3, iA - 1, iB - nA - 1]
                total_sum = elst_sum + exch_sum + indu_sum + disp_sum
                if print_results:
                    print(
                        f"{kA:10s} {kB:10s} {elst_sum:10.6f} {exch_sum:10.6f} "
                        f"{indu_sum:10.6f} {disp_sum:10.6f} {total_sum:10.6f}"
                    )
                # data_dict["mol"].append(mol)
                data_dict["fA-fB"].append(f"{kA}-{kB}")
                data_dict["elst"].append(elst_sum)
                data_dict["exch"].append(exch_sum)
                data_dict["indu"].append(indu_sum)
                data_dict["disp"].append(disp_sum)
                data_dict["total"].append(total_sum)
    df = pd.DataFrame(data_dict)
    return pred_IEs, pairwise_energies, df


def apnet3_model_predict_pairs(
    mols: [Molecule],
    compile: bool = True,
    batch_size: int = 16,
    ensemble_model_dir: str = model_dir,
    fAs: [{str: [int]}] = None,
    fBs: [{str: [int]}] = None,
    print_results: bool = False,
):
    """
    Predicts AP2 pairwise energies that correspond to an FSAPT calculation. fA
    and fB are LISTS of dictionaries that specify the atom indices for fragment
    A and B to sum their contributions. The syntax is identical to the Psi4
    FSAPT updates from https://github.com/psi4/psi4/pull/3222
    """

    current_file_path = os.path.dirname(os.path.realpath(__file__))
    am_path = f"{current_file_path}/../../tests/test_models/ap3_ensemble_0/am_3.pt"
    at_hf_vw_path = f"{current_file_path}/../../tests/test_models/ap3_ensemble_0/am_h+1_3.pt"
    at_elst_path = f"{current_file_path}/../../tests/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
    ap3_path = f"{current_file_path}/../../tests/test_models/ap3_ensemble_0/ap3_.pt"
    print("THIS MODEL IS EXPERIMENTAL AND MAY NOT YIELD ACCURATE RESULTS")
    atom_type_hf_vw_model = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    atom_type_elst_model = AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )

    # Initialize AP3 model for FSAPT training

    ap3 = AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        # ds_root=temp_dir,
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        am_dimer_param_model=atom_type_elst_model,
        use_precomputed_classical=False,
        ignore_database_null=True,
        ds_spec_type=6,  # NOTE spec_type 5 for FSAPT
        ds_type="fsapt_energies",  # Important: set ds_type for FSAPT
        pre_trained_model_path=ap3_path,
    )
    assert fAs is not None, (
        "fAs must be provided. Example: [{'Methyl1_A': [1, 2, 7, 8], 'Methyl2_A': [3, 4, 5, 6]}...]"
    )
    assert fBs is not None, (
        "fBs must be provided, Example: [{'Peptide_B': [9, 10, 11, 16, 26], 'T-Butyl_B': [12, 13, 14, 15]}...]"
    )
    pred_IEs = np.zeros((len(mols), 5))
    print("Processing mols...")
    IEs, pairwise_energies = ap3.predict_qcel_mols(
        mols,
        batch_size=batch_size,
        return_pairs=True,
    )
    pred_IEs[:, 1:] += IEs
    pred_IEs[:, 0] += np.sum(IEs, axis=1)

    data_dict = {
        "fA-fB": [],
        "total": [],
        "elst": [],
        "exch": [],
        "indu": [],
        "disp": [],
    }
    for i, mol in enumerate(mols):
        if print_results:
            # Analyze results
            print(f"")
            header = f"""==> AP2-FSAPT <==
monA-monB full IE: {pred_IEs[i]}

 Frag1      Frag2         Elst       Exch       Ind        Disp       Total
            """
            print(header)
        monA = mol.get_fragment([0])
        nA = len(monA.atomic_numbers)
        for kA, vA in fAs[i].items():
            for kB, vB in fBs[i].items():
                elst_sum = 0.0
                exch_sum = 0.0
                indu_sum = 0.0
                disp_sum = 0.0
                total_sum = 0.0
                for iA in vA:
                    for iB in vB:
                        # Subtract 1 to convert to 0-based indexing
                        elst_sum += pairwise_energies[i][0, iA - 1, iB - nA - 1]
                        exch_sum += pairwise_energies[i][1, iA - 1, iB - nA - 1]
                        indu_sum += pairwise_energies[i][2, iA - 1, iB - nA - 1]
                        disp_sum += pairwise_energies[i][3, iA - 1, iB - nA - 1]
                total_sum = elst_sum + exch_sum + indu_sum + disp_sum
                if print_results:
                    print(
                        f"{kA:10s} {kB:10s} {elst_sum:10.6f} {exch_sum:10.6f} "
                        f"{indu_sum:10.6f} {disp_sum:10.6f} {total_sum:10.6f}"
                    )
                # data_dict["mol"].append(mol)
                data_dict["fA-fB"].append(f"{kA}-{kB}")
                data_dict["elst"].append(elst_sum)
                data_dict["exch"].append(exch_sum)
                data_dict["indu"].append(indu_sum)
                data_dict["disp"].append(disp_sum)
                data_dict["total"].append(total_sum)
    df = pd.DataFrame(data_dict)
    return pred_IEs, pairwise_energies, df


def dapnet2_levels_of_theory_pretrained():
    """
    Returns a list of possible levels of theory with pretrained models to use
    in the dapnet2_model_predict(). Note that these pretrained models
    predicts E=(m1-CCSD(T)/CBS/CP)
    """
    return [
        "B3LYP-D3/aug-cc-pVTZ/unCP",
        "B2PLYP-D3/aug-cc-pVTZ/unCP",
        "wB97X-V/aug-cc-pVTZ/CP",
        "SAPT0/aug-cc-pVDZ/SA",
        "MP2/aug-cc-pVTZ/CP",
        "HF/aug-cc-pVDZ/CP",
    ]


def dapnet2_model_predict(
    mols: [Molecule],
    m1: str,
    m2: str,
    compile: bool = True,
    pre_trained_model_path: str = None,
    batch_size: int = 16,
    use_GPU: bool = None,
) -> np.ndarray:
    atom_model = AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=use_GPU,
    ).set_pretrained_model(model_id=0)
    apnet2 = AtomPairwiseModels.apnet2.APNet2Model(
        atom_model=atom_model.model,
        use_GPU=use_GPU,
    ).set_pretrained_model(model_id=0)
    apnet2.model.return_hidden_states = True
    if pre_trained_model_path is None:
        assert m1 in dapnet2_levels_of_theory_pretrained(), (
            f"Pretrained model for {m1} not found. "
            f"Please use one of the following: {dapnet2_levels_of_theory_pretrained()}"
        )
        assert m2 == "CCSD(T)/CBS/CP", (
            "Pretrained models only predict m2=CCSD(T)/CBS/CP"
        )
        pre_trained_model_path = resources.files("apnet_pt").joinpath(
            "models",
            "dapnet2",
            f"{clean_str_for_filename(m1)}_to_{clean_str_for_filename(m2)}_0.pt",
        )
    dapnet2 = AtomPairwiseModels.dapnet2.dAPNet2Model(
        atom_model=atom_model,
        apnet2_model=apnet2,
        pre_trained_model_path=pre_trained_model_path,
        use_GPU=use_GPU,
    )
    return dapnet2.predict_qcel_mols(mols, batch_size=batch_size)
