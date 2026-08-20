import copy
import logging
import os
import sys
from importlib import resources

import numpy as np
import pandas as pd
from qcelemental.models.molecule import Molecule

from apnet_pt.pt_datasets.dapnet_ds import clean_str_for_filename

from . import AtomModels
from . import AtomPairwiseModels
from . import atomic_datasets

# model_dir = os.path.dirname(os.path.realpath(__file__)) + "/models/"
model_dir = resources.files("apnet_pt").joinpath("models")
HF_REPO_ID = "awallace3/qcmlforge"
_DOWNLOAD_APPROVED = None
LOGGER = logging.getLogger(__name__)


def _hf_hub_download(rel_path: str, local_files_only: bool) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to load pretrained models. "
            "Install qcmlforge with dependencies or `pip install huggingface_hub`."
        ) from exc

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=rel_path,
        local_files_only=local_files_only,
    )


def _packaged_model_path(rel_path: str) -> str | None:
    model_path = resources.files("apnet_pt").joinpath("models", *rel_path.split("/"))
    return str(model_path) if model_path.is_file() else None


def _allow_model_download(missing_paths: list[str]) -> bool:
    global _DOWNLOAD_APPROVED

    if _DOWNLOAD_APPROVED is not None:
        return _DOWNLOAD_APPROVED

    env_value = os.getenv("QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED", "").strip().lower()
    if env_value in {"1", "true", "yes", "y"}:
        _DOWNLOAD_APPROVED = True
        LOGGER.info(
            "QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED enabled pretrained downloads from %s",
            HF_REPO_ID,
        )
        return True
    if env_value in {"0", "false", "no", "n"}:
        _DOWNLOAD_APPROVED = False
        LOGGER.info(
            "QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED disabled pretrained downloads from %s",
            HF_REPO_ID,
        )
        return False

    if not sys.stdin.isatty():
        _DOWNLOAD_APPROVED = False
        return False

    preview = ", ".join(missing_paths[:3])
    if len(missing_paths) > 3:
        preview += ", ..."
    try:
        answer = (
            input(
                "Pretrained model weights are not available locally and need to be "
                f"downloaded from https://huggingface.co/{HF_REPO_ID} "
                f"(missing: {preview}). Download now? [y/N]: "
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        _DOWNLOAD_APPROVED = False
        return False
    _DOWNLOAD_APPROVED = answer in {"y", "yes"}
    return _DOWNLOAD_APPROVED


def _resolve_pretrained_paths(rel_paths: list[str]) -> dict[str, str]:
    resolved = {}
    missing = []

    for rel_path in rel_paths:
        try:
            resolved[rel_path] = _hf_hub_download(rel_path, local_files_only=True)
        except ImportError:
            raise
        except Exception:
            missing.append(rel_path)

    if not missing:
        return resolved

    if not _allow_model_download(missing):
        for rel_path in missing:
            fallback = _packaged_model_path(rel_path)
            if fallback is not None:
                resolved[rel_path] = fallback
                continue
            raise RuntimeError(
                "Missing pretrained model in local cache. "
                "Set QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED=1 to auto-download, "
                f"or run interactively and accept download for '{rel_path}'."
            )
        return resolved

    for rel_path in missing:
        try:
            resolved[rel_path] = _hf_hub_download(rel_path, local_files_only=False)
        except ImportError:
            raise
        except Exception as exc:
            fallback = _packaged_model_path(rel_path)
            if fallback is not None:
                resolved[rel_path] = fallback
                continue
            raise RuntimeError(
                f"Unable to load pretrained model '{rel_path}' from "
                f"https://huggingface.co/{HF_REPO_ID}."
            ) from exc

    return resolved


def atom_model_predict(
    mols: list[Molecule],
    compile: bool = True,
    batch_size: int = 3,
    return_mol_arrays: bool = True,
):
    num_models = 5
    model_paths = _resolve_pretrained_paths(
        [f"am_ensemble/am_{i}.pt" for i in range(num_models)]
    )
    am = AtomModels.ap2_atom_model.AtomModel(
        pre_trained_model_path=model_paths["am_ensemble/am_0.pt"],
    )
    if compile:
        print("Compiling models...")
        am.compile_model()
    models = [copy.deepcopy(am) for _ in range(num_models)]
    for i in range(1, num_models):
        models[i].set_pretrained_model(
            model_path=model_paths[f"am_ensemble/am_{i}.pt"],
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
    mols: list[Molecule],
    compile: bool = True,
    batch_size: int = 16,
    ensemble_model_dir: str = model_dir,
    ap2_fused: bool = False,
):
    if ap2_fused:
        num_models = 4
        additional_models_start = 2
        model_paths = _resolve_pretrained_paths(
            ["ap2-fused_ensemble/ap2_1.pt"]
            + [f"ap2-fused_ensemble/ap2_{i}.pt" for i in range(2, num_models)]
        )
        ap2 = AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
            pre_trained_model_path=model_paths["ap2-fused_ensemble/ap2_1.pt"]
        )
    else:
        num_models = 5
        additional_models_start = 1
        model_paths = _resolve_pretrained_paths(
            [f"ap2_ensemble/ap2_{i}.pt" for i in range(num_models)]
            + [f"am_ensemble/am_{i}.pt" for i in range(num_models)]
        )
        ap2 = AtomPairwiseModels.apnet2.APNet2Model(
            pre_trained_model_path=model_paths["ap2_ensemble/ap2_0.pt"],
            atom_model_pre_trained_path=model_paths["am_ensemble/am_0.pt"],
        )
    if compile:
        print("Compiling models...")
        ap2.compile_model()
    models = [copy.deepcopy(ap2) for _ in range(num_models)]
    for i in range(additional_models_start, num_models):
        if ap2_fused:
            models[i].set_pretrained_model(
                ap2_model_path=model_paths[f"ap2-fused_ensemble/ap2_{i}.pt"]
            )
        else:
            models[i].set_pretrained_model(
                ap2_model_path=model_paths[f"ap2_ensemble/ap2_{i}.pt"],
                am_model_path=model_paths[f"am_ensemble/am_{i}.pt"],
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
    mols: list[Molecule],
    compile: bool = True,
    batch_size: int = 16,
    ensemble_model_dir: str = model_dir,
    fAs: list[dict[str, list[int]]] | None = None,
    fBs: list[dict[str, list[int]]] | None = None,
    print_results: bool = False,
    ap2_fused: bool = True,
):
    """
    Compute ensemble-averaged APNet2 pairwise interaction energies for specified fragment pairs and return per-molecule energies, per-atom pairwise arrays, and a fragment-pair breakdown DataFrame.

    Parameters:
        mols: Iterable of Molecule
            Molecules to evaluate.
        fAs: list[dict[str, list[int]]]
            Per-molecule fragment-A definitions: a list with one dict per molecule mapping fragment names to 1-based atom indices belonging to fragment A.
        fBs: list[dict[str, list[int]]]
            Per-molecule fragment-B definitions: a list with one dict per molecule mapping fragment names to 1-based atom indices belonging to fragment B.
        compile: bool, optional
            If True, compile models before prediction.
        batch_size: int, optional
            Prediction batch size used by the model.
        ensemble_model_dir: str, optional
            Directory containing ensemble model files.
        print_results: bool, optional
            If True, print a formatted per-fragment summary to stdout.
        ap2_fused: bool, optional
            If True, use the fused APNet2 variant; otherwise use the standard APNet2 ensemble.

    Returns:
        pred_IEs (numpy.ndarray):
            Array of shape (N, 5) where N is the number of molecules. Column 0 is the ensemble-averaged total interaction energy; columns 1–4 are the ensemble-averaged energy components in the order: electrostatics, exchange, induction, dispersion.
        pairwise_energies (list):
            Per-molecule pairwise energy arrays averaged over the ensemble. For each molecule this contains component arrays indexed by component (elst, exch, indu, disp) and atom indices for fragment A and fragment B (component, nA_atoms, nB_atoms).
        df (pandas.DataFrame):
            Fragment-pair breakdown with columns ["fA-fB", "total", "elst", "exch", "indu", "disp"], one row per fragment-A/fragment-B pair.
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
        model_paths = _resolve_pretrained_paths(
            ["ap2-fused_ensemble/ap2_1.pt"]
            + [f"ap2-fused_ensemble/ap2_{i}.pt" for i in range(2, num_models)]
        )
        ap2 = AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
            pre_trained_model_path=model_paths["ap2-fused_ensemble/ap2_1.pt"]
        )
    else:
        additional_models_start = 2
        num_models = 5
        model_paths = _resolve_pretrained_paths(
            [f"ap2_ensemble/ap2_{i}.pt" for i in range(num_models)]
            + [f"am_ensemble/am_{i}.pt" for i in range(num_models)]
        )
        ap2 = AtomPairwiseModels.apnet2.APNet2Model(
            pre_trained_model_path=model_paths["ap2_ensemble/ap2_0.pt"],
            atom_model_pre_trained_path=model_paths["am_ensemble/am_0.pt"],
        )
    if compile:
        print("Compiling models...")
        ap2.compile_model()
    models = [copy.deepcopy(ap2) for _ in range(num_models)]
    for i in range(additional_models_start, num_models):
        if ap2_fused:
            models[i].set_pretrained_model(
                ap2_model_path=model_paths[f"ap2-fused_ensemble/ap2_{i}.pt"]
            )
        else:
            models[i].set_pretrained_model(
                ap2_model_path=model_paths[f"ap2_ensemble/ap2_{i}.pt"],
                am_model_path=model_paths[f"am_ensemble/am_{i}.pt"],
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
    for model_idx in range(1, num_models):
        IEs, pairs = models[model_idx].predict_qcel_mols(
            mols,
            batch_size=batch_size,
            return_pairs=True,
        )
        for pair_idx in range(len(pairs)):
            for component_idx in range(len(pairs[pair_idx])):
                pairwise_energies[pair_idx][component_idx] += pairs[pair_idx][
                    component_idx
                ]
        pred_IEs[:, 1:] += IEs
        pred_IEs[:, 0] += np.sum(IEs, axis=1)
    for pair_idx in range(len(pairwise_energies)):
        for component_idx in range(len(pairwise_energies[pair_idx])):
            pairwise_energies[pair_idx][component_idx] /= num_models
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
        monB = mol.get_fragment([1])
        nA = len(monA.atomic_numbers)
        nB = len(monB.atomic_numbers)
        for kA, vA in fAs[i].items():
            for kB, vB in fBs[i].items():
                print(f"{kA}, {vA}, {kB}, {vB}")
                print(monA, nA, monB, nB)
                assert min(vB) > nA, (
                    "fB atom indices must be for fragment B. min(vB) <= nA, meaning atom index in fragment A. Please check your input."
                )
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
    mols: list[Molecule],
    compile: bool = True,
    batch_size: int = 16,
    ensemble_model_dir: str = model_dir,
    fAs: list[dict[str, list[int]]] | None = None,
    fBs: list[dict[str, list[int]]] | None = None,
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
    at_hf_vw_path = (
        f"{current_file_path}/../../tests/test_models/ap3_ensemble_0/am_h+1_3.pt"
    )
    at_elst_path = (
        f"{current_file_path}/../../tests/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
    )
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
        if print_results:
            print(mol.to_string("psi4"))
        monA = mol.get_fragment([0])
        nA = len(monA.atomic_numbers)
        for kA, vA in fAs[i].items():
            for kB, vB in fBs[i].items():
                assert min(vB) > nA, (
                    "fB atom indices must be for fragment B. min(vB) <= nA, meaning atom index in fragment A. Please check your input."
                )
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


DAPNET2_PRETRAINED_MODEL_FILENAMES = [
    "B2PLYP-D2aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "B2PLYP-D3aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "B2PLYP-D3aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "B2PLYPaug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "B2PLYPaug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "B3LYP-D2aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "B3LYP-D3BJ_adz_CCSD_LP_T_RP_CBSCP.pt",
    "B3LYP-D3aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "B3LYP-D3aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "B3LYPaug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "B3LYPaug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "B97-D2aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "B97-D3aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "B97-D3aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "B97aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "CCSD-F12aaug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "CCSD-F12baug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "CCSD_LP_T_RP_-F12aaug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "CCSD_LP_T_RP_-F12baug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "DW-CCSD_LP_T_RP_-F12aug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "DW-MP2aug-cc-pVDTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "DW-MP2aug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "DW-MP2aug-cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "DW-MP2aug-cc-pVTQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "DW-MP2aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "DW-MP2cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "HF-CABSaug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "HFaug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "HFaug-cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "HFaug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "HFcc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "M05-2Xaug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "M05-2Xaug-cc-pVDZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "MP2-F12aug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "MP2aug-cc-pVDTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "MP2aug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "MP2aug-cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "MP2aug-cc-pVTQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "MP2aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "MP2cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "PBE-D2aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "PBE-D3BJ_adz_CCSD_LP_T_RP_CBSCP.pt",
    "PBE-D3aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "PBE-D3aug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "PBEH3C_adz_CCSD_LP_T_RP_CBSCP.pt",
    "PBEaug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "PBEaug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "SAPT0aug-cc-pVDZSA_CCSD_LP_T_RP_CBSCP.pt",
    "SAPT0jun-cc-pVDZSA_CCSD_LP_T_RP_CBSCP.pt",
    "SAPT2aug-cc-pVDZSA_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-CCSD-F12aaug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-CCSD-F12baug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-MP2-F12aug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-MP2aug-cc-pVDTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-MP2aug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-MP2aug-cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-MP2aug-cc-pVTQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-MP2aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-MP2cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS-SAPT0jun-cc-pVDZSA_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_MI_RP_-CCSD-F12aaug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_MI_RP_-CCSD-F12baug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_MI_RP_-MP2aug-cc-pVDTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_MI_RP_-MP2aug-cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_MI_RP_-MP2aug-cc-pVTQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_MI_RP_-MP2aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_MI_RP_-MP2cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_N_RP_-MP2-F12aug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_N_RP_-MP2aug-cc-pVDTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_N_RP_-MP2aug-cc-pVDZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_N_RP_-MP2aug-cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_N_RP_-MP2aug-cc-pVTQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_N_RP_-MP2aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "SCS_LP_N_RP_-MP2cc-pVQZCP_CCSD_LP_T_RP_CBSCP.pt",
    "sSAPT0aug-cc-pVDZSA_CCSD_LP_T_RP_CBSCP.pt",
    "sSAPT0jun-cc-pVDZSA_CCSD_LP_T_RP_CBSCP.pt",
    "wB97X-Daug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "wB97X-Daug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
    "wB97X-Vaug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    "wB97X-Vaug-cc-pVTZunCP_CCSD_LP_T_RP_CBSCP.pt",
]


def dapnet2_levels_of_theory_pretrained():
    """
    Returns a list of possible m1 levels of theory with pretrained dAPNet2
    models. These pretrained models predict E=(m1-CCSD(T)/CBS/CP).
    """
    target = f"_{clean_str_for_filename('CCSD(T)/CBS/CP')}.pt"
    return [name.removesuffix(target) for name in DAPNET2_PRETRAINED_MODEL_FILENAMES]


def _resolve_dapnet2_pretrained_path(m1: str, m2: str) -> str:
    m1_clean = clean_str_for_filename(m1)
    m2_clean = clean_str_for_filename(m2)
    rel_paths = [
        f"dapnet2/{m1_clean}_to_{m2_clean}_0.pt",
        f"dapnet2/{m1_clean}_{m2_clean}.pt",
    ]
    rel_paths.extend(
        f"dapnet2/{name}"
        for name in DAPNET2_PRETRAINED_MODEL_FILENAMES
        if name == f"{m1_clean}_{m2_clean}.pt"
    )

    errors = []
    for rel_path in dict.fromkeys(rel_paths):
        try:
            return _resolve_pretrained_paths([rel_path])[rel_path]
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError(
        f"Pretrained dAPNet2 model for m1={m1!r}, m2={m2!r} not found. "
        "Available m1 values include: "
        f"{dapnet2_levels_of_theory_pretrained()}"
    ) from RuntimeError("\n".join(errors))


def dapnet2_model_predict(
    mols: list[Molecule],
    m1: str,
    m2: str,
    compile: bool = True,
    pre_trained_model_path: str = None,
    batch_size: int = 16,
    use_GPU: bool = None,
) -> np.ndarray:
    base_model_paths = _resolve_pretrained_paths(
        ["am_ensemble/am_0.pt", "ap2_ensemble/ap2_0.pt"]
    )
    atom_model = AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=use_GPU,
    ).set_pretrained_model(model_path=base_model_paths["am_ensemble/am_0.pt"])
    apnet2 = AtomPairwiseModels.apnet2.APNet2Model(
        atom_model=atom_model.model,
        use_GPU=use_GPU,
    ).set_pretrained_model(
        ap2_model_path=base_model_paths["ap2_ensemble/ap2_0.pt"],
        am_model_path=base_model_paths["am_ensemble/am_0.pt"],
    )
    apnet2.model.return_hidden_states = True
    if pre_trained_model_path is None:
        if m2 != "CCSD(T)/CBS/CP":
            raise ValueError("Pretrained models only predict m2=CCSD(T)/CBS/CP")
        pre_trained_model_path = _resolve_dapnet2_pretrained_path(m1, m2)
    dapnet2 = AtomPairwiseModels.dapnet2.dAPNet2Model(
        atom_model=atom_model,
        apnet2_model=apnet2,
        pre_trained_model_path=pre_trained_model_path,
        use_GPU=use_GPU,
    )
    return dapnet2.predict_qcel_mols(mols, batch_size=batch_size)
