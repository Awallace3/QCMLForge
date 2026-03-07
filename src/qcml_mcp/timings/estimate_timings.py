import psi4
from psi4 import core
import math
import qcelemental as qcel
import numpy as np
import pandas as pd
from pprint import pprint as pp
from .all_polynomial_fits import fit_data, dft_methods, wfn_methods

psi4.core.be_quiet()


def compute_psi4_time_estimation_variables(
    mol_qcel,
    basis_set,
) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    n_atoms = len(mol_qcel.atomic_numbers)
    mol = psi4.core.Molecule.from_schema(mol_qcel.dict())
    psi4.set_options({"basis": basis_set})
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    bs = wfn.basisset()
    n_occupied = math.ceil((wfn.nalpha() + wfn.nbeta()) / 2)
    n_virtual = bs.nbf() - n_occupied
    np_total = 2 * n_atoms * 75 * 302 * 0.32
    aux_basis = core.BasisSet.build(
        wfn.molecule(),
        "DF_BASIS_MP2",
        core.get_option("DFMP2", "DF_BASIS_MP2"),
        "RIFIT",
        core.get_global_option("BASIS"),
    )
    nbf_aux = aux_basis.nbf()
    return n_occupied, n_virtual, np_total, nbf_aux


def predict_timing(method, input_values):
    """
    Predict timing for a given method and input variables using saved polynomial fits.

    Args:
        method (str): The computational method name (e.g., 'MP2', 'B3LYP-D3')
        input_values (dict): Dictionary with variable names as keys and values as values
                           e.g., {'nocc': 10, 'nvirt': 50, 'nbf_aux': 200}

    Returns:
        dict: Dictionary containing:
            - 'log_time': Predicted log10(time in seconds)
            - 'time_seconds': Predicted time in seconds
            - 'variables_used': List of variables used in the prediction
            - 'method': Method name used
    """

    # Check if method exists in the data
    if method not in fit_data["methods"]:
        available_methods = list(fit_data["methods"].keys())
        raise ValueError(
            f"Method '{method}' not found. Available methods: {available_methods}"
        )

    method_data = fit_data["methods"][method]

    # Extract polynomial parameters
    variables = method_data["variables"]
    degrees = method_data["degrees"]
    operators = method_data["operators"]
    coefficients = method_data["coefficients"]

    # Validate input variables
    missing_vars = [var for var in variables if var not in input_values]
    if missing_vars:
        raise ValueError(
            f"Missing required variables for method '{method}': {missing_vars}"
        )

    # Extract variable values in the correct order
    X_vars = []
    for var in variables:
        X_vars.append(input_values[var])

    # Evaluate the polynomial combination
    def evaluate_polynomial_combination(coeffs_list, X_values, degrees, operators):
        """Evaluate the polynomial model for a single data point"""
        prediction = 0.0

        for i, (X_val, degree, poly_coeffs) in enumerate(
            zip(X_values, degrees, coeffs_list)
        ):
            # Evaluate polynomial for this variable
            poly_val = 0.0
            for j, coeff in enumerate(poly_coeffs):
                poly_val += coeff * (X_val**j)

            if i == 0:
                prediction = poly_val
            elif operators[i - 1] == "*":
                prediction *= poly_val
            elif operators[i - 1] == "+":
                prediction += poly_val

        return prediction

    # Make prediction
    log_time_pred = evaluate_polynomial_combination(
        coefficients, X_vars, degrees, operators
    )
    time_pred = 10**log_time_pred  # Convert from log10 back to actual time

    result = {
        "log_time": log_time_pred,
        "time_seconds": time_pred,
        "variables_used": variables,
        "method": method,
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

    if method not in fit_data["methods"]:
        available_methods = list(fit_data["methods"].keys())
        raise ValueError(
            f"Method '{method}' not found. Available methods: {available_methods}"
        )

    method_data = fit_data["methods"][method]
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
        pred = predict_timing(method, input_values)
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
        n_occupied, n_virtual, np_total, nbf_aux = compute_psi4_time_estimation_variables(
            mol,
            "aug-cc-pVDZ",
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
