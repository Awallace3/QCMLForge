import math
import qcelemental as qcel
import numpy as np
import pandas as pd
from pprint import pprint as pp
from .all_polynomial_fits import fit_data


def compute_psi4_time_estimation_variables(
    mol_qcel,
    basis_set,
) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    import psi4

    psi4.core.be_quiet()
    n_atoms = len(mol_qcel.atomic_numbers)
    mol = psi4.core.Molecule.from_schema(mol_qcel.dict())
    psi4.set_options({"basis": basis_set})
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    bs = wfn.basisset()
    n_occupied = math.ceil((wfn.nalpha() + wfn.nbeta()) / 2)
    n_virtual = bs.nbf() - n_occupied
    np_total = 2 * n_atoms * 75 * 302 * 0.32
    aux_basis = psi4.core.BasisSet.build(
        wfn.molecule(),
        "DF_BASIS_MP2",
        psi4.core.get_option("DFMP2", "DF_BASIS_MP2"),
        "RIFIT",
        psi4.core.get_global_option("BASIS"),
    )
    nbf_aux = aux_basis.nbf()
    return n_occupied, n_virtual, np_total, nbf_aux


def _normalize_method(method: str) -> str:
    """Normalize a method name for lookup in timing fit data."""
    return method.lower()


def _evaluate_moo_time_polynomial(method_key: str, coeffs, input_values) -> float:
    """Evaluate the raw seconds polynomial from ``moo.py`` for one data point."""
    p = [float(x) for x in coeffs]
    nocc = float(input_values["nocc"])
    nvirt = float(input_values["nvirt"])
    nbf_aux = float(input_values["nbf_aux"])
    np_total = float(input_values.get("np_total", 0.0))
    nbf = nocc + nvirt

    if method_key == "hf":
        return (
            p[0]
            + p[1] * (nbf ** p[4] * nbf_aux**2)
            + p[2] * (nbf ** p[4] * nbf_aux)
            + p[3] * (nocc * nbf_aux * nbf**2)
        )
    if method_key == "pbe-d3":
        return (
            p[0] + p[1] * (np_total * nbf ** p[3]) + p[2] * (nbf ** p[4] * nbf_aux**2)
        )
    if method_key == "wb97x-d":
        return p[0] + p[1] * (np_total * nbf ** p[3]) + p[2] * (nocc * nbf_aux * nbf**2)
    if method_key == "wb97x-v":
        return (
            p[0]
            + p[1] * (nbf ** p[4] * nbf_aux**2)
            + p[2] * (nocc * nbf_aux * nbf**2)
            + p[3] * (np_total**2)
        )
    if method_key == "mp2":
        return p[0] + p[1] * (nocc * nbf**2 * nbf_aux) + p[2] * (nbf**2 * nbf_aux)
    if method_key in {"b3lyp-d3", "b2plyp-d3", "m05-2x"}:
        return p[0] + p[1] * (np_total * nbf ** p[3]) + p[2] * (nocc * nbf_aux * nbf**2)
    if method_key == "fno-ccsd":
        return p[0] + p[1] * (nocc * nbf**2 * nbf_aux) + p[2] * (nocc**2 * nvirt**4)
    if method_key == "fno-ccsd(t)":
        return (
            p[0]
            + p[1] * (nocc * nbf**2 * nbf_aux)
            + p[2] * (nocc**2 * nvirt**4)
            + p[3] * (nocc**3 * nvirt**4)
        )
    raise ValueError(f"No moo.py timing polynomial evaluator for method '{method_key}'")


def predict_timing(method, input_values):
    """
    Predict timing for a given method and input variables using saved fits.

    The current fits follow the ``moo.py`` logic: first evaluate a raw fitted
    polynomial for time in seconds, then compute ``log10(abs(raw_seconds))``
    with a lower clip of ``1e-12`` for the reported log value.
    """

    method_key = _normalize_method(method)
    if method_key not in fit_data["methods"]:
        available_methods = list(fit_data["methods"].keys())
        raise ValueError(
            f"Method '{method}' not found. Available methods: {available_methods}"
        )

    method_data = fit_data["methods"][method_key]
    variables = method_data["variables"]
    coefficients = method_data["coefficients"]

    missing_vars = [var for var in variables if var not in input_values]
    if missing_vars:
        raise ValueError(
            f"Missing required variables for method '{method_key}': {missing_vars}"
        )

    raw_time_pred = _evaluate_moo_time_polynomial(
        method_key,
        coefficients,
        input_values,
    )
    time_pred = float(np.clip(abs(raw_time_pred), 1.0e-12, None))
    log_time_pred = float(np.log10(time_pred))

    result = {
        "log_time": log_time_pred,
        "time_seconds": time_pred,
        "raw_time_seconds": float(raw_time_pred),
        "variables_used": variables,
        "method": method_key,
        "input_values": {var: input_values[var] for var in variables},
    }

    return result


def predict_timing_batch(method, input_dataframe):
    """
    Predict timing for multiple data points using a pandas DataFrame.

    Args:
        method (str): The computational method name
        input_dataframe (pd.DataFrame): DataFrame containing the input variables

    Returns:
        pd.DataFrame: Original dataframe with added 'predicted_log_time' and 'predicted_time_seconds' columns
    """

    df_copy = input_dataframe.copy()

    method_key = _normalize_method(method)
    if method_key not in fit_data["methods"]:
        available_methods = list(fit_data["methods"].keys())
        raise ValueError(
            f"Method '{method}' not found. Available methods: {available_methods}"
        )

    method_data = fit_data["methods"][method_key]
    variables = method_data["variables"]

    # Check if all required variables are in the dataframe
    missing_vars = [var for var in variables if var not in df_copy.columns]
    if missing_vars:
        raise ValueError(
            f"Missing required columns for method '{method}': {missing_vars}"
        )

    # Make predictions for each row
    predictions = []
    for _, row in df_copy.iterrows():
        input_values = {var: row[var] for var in variables}
        pred = predict_timing(method_key, input_values)
        predictions.append(pred)

    # Add predictions to dataframe
    df_copy["predicted_log_time"] = [pred["log_time"] for pred in predictions]
    df_copy["predicted_time_seconds"] = [pred["time_seconds"] for pred in predictions]

    return df_copy


def example_usage():
    """Example of how to use the prediction functions"""

    # Single prediction example
    input_vars = {"nocc": 12, "nvirt": 48, "nbf_aux": 180}
    try:
        result = predict_timing("MP2", input_vars)
        pp(result)
        print(f"Prediction for MP2:")
        print(f"  Input: {result['input_values']}")
        print(f"  Predicted log(time): {result['log_time']:.4f}")
        print(f"  Predicted time: {result['time_seconds']:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")

    # Batch prediction example (if you have a dataframe)
    df_test = pd.DataFrame(
        {"nocc": [10, 15, 20], "nvirt": [40, 60, 80], "nbf_aux": [150, 200, 250]}
    )

    df_with_predictions = predict_timing_batch("MP2", df_test)
    print(df_with_predictions[["nocc", "nvirt", "nbf_aux", "predicted_time_seconds"]])


def estimate_timing_for_qcel_molecule(
    qcel_molecule: qcel.models.Molecule,
    method: str = "MP2",
    basis_set: str = "aug-cc-pVDZ",
    manybody: bool = True,
):
    """
    Estimate timing for a given QCElemental molecule using the specified method and basis set.

    This function computes the necessary variables for the timing estimation
    for monomers and dimers and prints the results. When manybody is True,
    it computes the timing for dimer and the each monomer separately to
    mimic a supermolecular interaction energy calculation.

    Args:
        qcel_molecule (qcel.models.Molecule): The molecule to estimate timing for.
        method (str): The computational method to use (default: 'MP2').
        basis_set (str): The basis set to use (default: 'aug-cc-pVDZ').
        manybody (bool): Whether to compute timing for dimer and monomers separately (default: True).
    Returns:
        float: Estimated time in seconds for the computation.
    """
    mols = [qcel_molecule]
    if manybody and qcel_molecule.fragments_:
        for n, i in enumerate(qcel_molecule.fragments_):
            mols.append(qcel_molecule.get_fragment(n))

    time_seconds = 0.0
    for mol in mols:
        n_occupied, n_virtual, np_total, nbf_aux = (
            compute_psi4_time_estimation_variables(
                mol,
                basis_set,
            )
        )
        input_vars = {
            "nocc": n_occupied,
            "nvirt": n_virtual,
            "nbf_aux": nbf_aux,
            "np_total": np_total,
        }
        result = predict_timing(method, input_vars)
        time_seconds += result["time_seconds"]
    return time_seconds


def main():
    # example_usage()
    # return

    monA = qcel.models.Molecule.from_data("""
    0 1
    O 0.000000 0.000000  0.000000
    H 0.758602 0.000000  0.504284
    H 0.260455 0.000000 -0.872893
    """)

    dimer = qcel.models.Molecule.from_data("""
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
    r = estimate_timing_for_qcel_molecule(monA)
    print(f"time for monomer MP2/aDZ : {r:.2f} seconds")
    r = estimate_timing_for_qcel_molecule(monA, "B3LYP-D3", "aug-cc-pVTZ")
    print(f"time for monomer B3LYP-D3: {r:.2f} seconds")
    r = estimate_timing_for_qcel_molecule(dimer)
    print(f"time for dimer MP2/aDZ   : {r:.2f} seconds")
    return


if __name__ == "__main__":
    main()
