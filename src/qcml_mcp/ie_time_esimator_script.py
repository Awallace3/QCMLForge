import re
import os
import psi4
import math
import numpy as np
import pandas as pd
import qcelemental as qcel
import apnet_pt                                                          
from qcml_mcp.timings.polynomial_fit_data import polynomial_expressions
from importlib import resources
from pprint import pprint as pp


def load_coeffs(
    restricted: bool
    ) -> pd.DataFrame:

    un = "" if restricted else "un"
    ref_path = resources.files("qcml_mcp.data").joinpath(
        f"time_fit_inference_df_{un}restricted.pkl"
    )
    with ref_path.open("rb") as handle:
        return pd.read_pickle(handle).set_index("method")


# _res_coeffs: pd.DataFrame = load_coeffs(1)
print("Warning, I have removed UHF functionality for now")
_coeffs: pd.DataFrame = load_coeffs(0)


def parse_geoms(
    path: str,
    ) -> pd.DataFrame:
    """
    Parses through a folder containing geometries in QCElemental recognized
    string format (xyz, xyz+, psi4, psi4+). Note that behaviour is largely undefined 
    if the provided molecular geometries are NOT in the Psi4 format (charge, multiplicity,
    atomic symbols, coordinates, and units specified):
    '''
    <charge_mon1> <multiplicity_mon1>
    <atom_symbol> <x> <y> <z>
    <atom_symbol> <x> <y> <z>
    --
    <charge_mon2> <multiplicity_mon2>
    units <unit>
    '''

    Returns a pd.DataFrame with the following columns:

    id:           filename string 
    n_atoms:      numpy.ndarray specifiying total number of atoms
    qcel_dimer:   qcelemental.models.Molecule representation of dimer 
    qcel_monA:    qcelemental.models.Molecule representation of first monomer
    qcel_monB:    qcelemental.models.Molecule representation of second monomer
    """
    id = []
    n_atoms = []
    qcel_dimer = []
    qcel_monA = []
    qcel_monB = []

    chgmult_pattern = re.compile(r'^\s*[-+]?\d+\s+[-+]?\d+\s*$')
    for file in os.listdir(path):
        filepath = os.path.join(path, file)

        if os.path.isfile(filepath):
            try:
                with open(filepath, "r", errors="ignore") as f:
                    raw_geom_str = f.read()

                    chgmult = 0 # will not catch 
                    for line in raw_geom_str.splitlines():
                        if chgmult_pattern.match(line):
                            chgmult = 1 
                            break
                    
                    units = 0
                    for line in reversed(raw_geom_str.splitlines()):
                        if "units" in line:
                            units = 1
                            break                    

                    if not units:
                        print("Warning: units may not be specified, assuming Angstroms by default")

                    if not chgmult:
                        print("Warning: charge/multiplicity may not be specified, molparse may default to incorrect values")

                    try:
                        mol_qcel = qcel.models.Molecule.from_data(raw_geom_str)

                    except Exception as e:
                        print(f"Error converting raw string to qcelemental.models.Molecule: \n {e}")
                        continue
                    
                    id.append(file.strip().split(".")[0])
                    n_atoms.append(len(mol_qcel.atomic_numbers))


                    fragments = mol_qcel.fragments
                    if len(fragments) != 2:
                        raise ValueError("input geometry must be a dimer")

                    qcel_dimer.append(mol_qcel)
                    qcel_monA.append(mol_qcel.get_fragment(0))
                    qcel_monB.append(mol_qcel.get_fragment(1))
                    print(f"succesfully built molecule and fragments for geometry found at {filepath}")

            except Exception as e:
                print(f"Error processing file at {filepath}: \n {e}")

    return pd.DataFrame({
        "id": id,
        "n_atoms": n_atoms,
        "qcel_dimer": qcel_dimer,
        "qcel_monA": qcel_monA,
        "qcel_monB": qcel_monB,
        })


def compute_psi4_time_estimation_variables(
        mol_qcel: qcel.models.Molecule, 
        basis_set: str, 
    ) -> np.array:
    """
    Builds the wavefunction for mol_qcel at the given basis_set
    and returns np.ndarray [[n_occupied, n_virtual, np_total, nbf_aux]].

    n_occupied: number of occupied orbitals in the primary basis
    n_vitual:   number of virtual orbitals in the primary basis
    np_total:   number of points in the DFT integration grid
    nbf_aux:    number of functions in the auxilary basis (JKFIT for methods.. this is an approximation)
    """
    try:
        mol = psi4.core.Molecule.from_schema(mol_qcel.dict())
    
    except Exception as e:
        print(f"Error when creating the Psi4 molecule object from QCElemental Schema: \n {e}")

    psi4.set_options({
        "basis": basis_set,
        "dft_pruning_scheme": "robust",
    })

    try: 
        wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
        bs = wfn.basisset()
        grid = psi4.core.DFTGrid.build(mol, bs)
        print("compute vars: built wfn & grid")

    except Exception as e:
        print(f"Error when building grid or wavefunction: \n {e}")
        return np.array([np.nan] * 4)

    n_occupied = math.ceil((wfn.nalpha() + wfn.nbeta()) / 2)
    n_virtual = bs.nbf() - n_occupied
    np_total = grid.npoints()

    try: 
        aux_basis = psi4.core.BasisSet.build(
            wfn.molecule(),
            "DF_BASIS_SCF",
            psi4.core.get_option("SCF", "DF_BASIS_SCF"),
            "JKFIT",
            psi4.core.get_global_option("BASIS"),
        )
        print("compute vars: built aux basis")

    except Exception as e:
        print(f"Error when building the auxillary basis: \n {e}")
        return np.array([np.nan] * 4)

    nbf_aux = aux_basis.nbf()
    psi4.core.clean()

    return np.array((
        n_occupied, 
        n_virtual, 
        np_total, 
        nbf_aux))

def build_inference_table(
        df: pd.DataFrame,
        methods: list[str],
        bases: list[str],
        cp: bool
    ) -> pd.DataFrame:
    """
    Modifies a pandas.Dataframe for batch prediction of (supermolecular) 
    interaction energy errors & timings. Timing variables are calculated
    per molecular system/basis-set pair and copied over to all LOTs considered. 
    """
    cp_str = "/unCP"

    if cp:
        print("Warning: using un-counterpoise corrected models for counterpoise corrected timing predictions")
        cp_str = "/CP"

    lotr_strings = [m + "/" + b + cp_str for b in bases for m in methods] # pretty sure everything I ran was not CP-corrected
    lotr_strings = lotr_strings * len(df)

    df_copy = df.copy()

    if df_copy.empty:
        return df_copy.reindex(
            columns=[
                *df_copy.columns,
                "dimer_tvars",
                "monA_tvars",
                "monB_tvars",
                "Level of Theory",
            ]
        )

    df_copy = df_copy.loc[df.index.repeat(len(bases))].copy()

    dimer_tvars = []
    monA_tvars = []
    monB_tvars = []

    print("Warning: using a JK auxiliary basis for MP2 and B2PLYP-D3 timing predictions")

    for idx in sorted(df_copy.index.unique()):
        rows = df_copy.loc[[idx]]   
        row = rows.iloc[0]          

        for basis in bases:
            dimer_tvars.append(compute_psi4_time_estimation_variables(row["qcel_dimer"], basis))
            monA_tvars.append(compute_psi4_time_estimation_variables(row["qcel_monA"], basis))
            monB_tvars.append(compute_psi4_time_estimation_variables(row["qcel_monB"], basis))
 
    df_copy[["dimer_tvars", "monA_tvars", "monB_tvars"]] = list(zip(dimer_tvars, monA_tvars, monB_tvars))

    # explode again and insert lotr strings 
    df_copy = df_copy.iloc[np.repeat(np.arange(len(df_copy)), len(methods))].copy()
    df_copy["Level of Theory"] = lotr_strings

    # reorder indicies
    a = len(df)
    b = len(df_copy) // a

    order = np.arange(len(df_copy)).reshape(a, b).T.ravel()

    df_copy = df_copy.iloc[order].copy()
    df_copy.index = np.repeat(np.arange(b), a)

    return df_copy

def predict_ie_errors_batch(
    df: pd.DataFrame,
) -> None: 
    """
    Predict error estimates for multiple molecular complexes.

    Estimates the interaction energy error between a starting level of theory
    and CCSD(T)/CBS/CP reference using the dAPNet2 model in QCMLForge for
    multiple molecular complexes. Each p4_string defines a molecular geometry
    in Psi4 format.

    Acceptable starting_level_of_theory values currently only include:

    ***NOTE: THIS LIST MAY NOT BE UP TO DATE***
    [
    "B3LYP-D3/aug-cc-pVTZ/unCP",
    "B2PLYP-D3/aug-cc-pVTZ/unCP",
    "wB97X-V/aug-cc-pVTZ/CP",
    "SAPT0/aug-cc-pVDZ/SA",
    "MP2/aug-cc-pVTZ/CP",
    "HF/aug-cc-pVDZ/CP",
    ]

    Input dataframe must contain dimer geometries as qcelemental.models.Molecule
    objects.
    """
    errors = []

    for idx in sorted(df.index.unique()):
        rows = df.loc[[idx]]
        mols = rows["qcel_dimer"].to_list()
        lotr = rows.iloc[0]["Level of Theory"]
        
        try:
            IE_pred = apnet_pt.pretrained_models.dapnet2_model_predict(
                mols,
                compile=False,
                m1=lotr,
                m2="CCSD(T)/CBS/CP",
            )
            errors.extend(IE_pred.tolist())
            print(f"completed interaction energy error estimation for all geometries at {lotr}")

        except Exception as e:
            print(f"dAPNet Error: \n {e}")
            errors.extend([np.nan] * len(mols))
            print(f"skipping {lotr} due to faliures")

    df["ERROR ESTIMATES (kcal/mol)"] = errors
    return

def predict_timing(
    method: str,
    basis: str,
    t_vars: np.ndarray,
) -> float:
    # if uhf_ref and (method == "FNO-CCSD" or method == "FNO-CCSD(T)"):
    #     raise ValueError(f"Polynomial expressions for unrestricted {method} not implemented yet")

    polynomial_lambda_expr = polynomial_expressions[method]["poly"]

    fit_label = "Augmented" if "aug" in basis else "Non-augmented"
    mask = (_coeffs["method"] == method) & (_coeffs["fit_label"] == fit_label)

    if not mask.any():
        mask = (_coeffs["method"] == method) & (_coeffs["fit_label"] == "All data")

    coeffs = _coeffs.loc[mask, "coefficients"].values[0]

    return np.log10(polynomial_lambda_expr(coeffs, t_vars))


def predict_timings_batch(
    df: pd.DataFrame,
) -> None: 
    """
    Predict timing for multiple data points using a pd.Dataframe.

    Returns the original Dataframe with added 'predicted_log_time' column 
    """
    # df_copy = df.copy()
    supermolecular_times = []

    for _, row in df.iterrows():
        method = row["Level of Theory"].split("/")[0]
        basis = row["Level of Theory"].split("/")[1]
        
        d_tvars = row["dimer_tvars"]
        a_tvars = row["monA_tvars"]
        b_tvars = row["monB_tvars"]

        try:
            a = predict_timing(
                method,
                basis,
                d_tvars,
            )

            b = predict_timing(
                method,
                basis,
                a_tvars,
            )

            c = predict_timing(
                method,
                basis,
                b_tvars,
            )

            supermolecular_times.append(np.log10(10**a + 10**b + 10**c))

        except Exception as e:
            print(f"timing polynomial error: \n {e}")
            supermolecular_times.append(np.nan)

    df["ESTIMATED CPU TIMES (log10(s))"] = supermolecular_times
    return

default_methods = [
    "HF",
    "PBE-D3",
    "wB97X-D",
    "wB97X-V",
    "MP2",
    "B3LYP-D3",
    "B2PLYP-D3",
    "M05-2X",
    "FNO-CCSD",
    "FNO-CCSD(T)",
]

default_bases = [
    "cc-pVDZ",
    "cc-pVTZ",
    "cc-pVQZ",
    "aug-cc-pVQZ",
    "aug-cc-pVTZ",
    "aug-cc-pVDZ",
]


def main(
    geom_path: str | None = None,
    n_threads: int = 4,
    using_cp: bool = False,
    methods: list[str] | None = None,
    bases: list[str] | None = None,
    auto_download: bool = False,    # set to True to avoid TTY prompting
) -> pd.DataFrame:
    if geom_path is None:
        raise ValueError("geom_path must be set")
    if methods is None:
        methods = default_methods
    if bases is None:
        bases = default_bases
    if auto_download:
        os.environ["QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED"] = "1"

    psi4.core.be_quiet()
    psi4.set_num_threads(n_threads)

    df1 = parse_geoms(geom_path)

    df2 = build_inference_table(df1, methods, bases, using_cp)

    predict_ie_errors_batch(df2)
    predict_timings_batch(df2)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pp(df2)

    return df2

if __name__ == "__main__":
    main()
