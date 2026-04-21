import numpy as np
import pytest
import qcelemental as qcel
from pprint import pprint as pp

import qcml_mcp


SERVER = qcml_mcp.server

DEFAULT_AM_P4_STRING = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
units angstrom
    """

DEFAULT_AP2_P4_STRING = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
units angstrom
    """

DEFAULT_DAPNET2_P4_STRINGS = [DEFAULT_AP2_P4_STRING]
DEFAULT_STARTING_LEVEL = "MP2/aug-cc-pVTZ/CP"
DEFAULT_TIMING_METHOD = "MP2"
DEFAULT_TIMING_BASIS = "aug-cc-pVDZ"

EXPECTED_AM_OUTPUT = {
    "AM-MBIS CHARGES": [
        -0.9731996059417725,
        0.4865997850894928,
        0.4865998387336731,
    ],
    "AM-MBIS DIPOLES": [
        [-0.20653732419013976, -0.0001249149441719055, 0.0745377004146576],
        [0.011094947159290314, -0.0001249149441719055, 0.023706956952810287],
        [-0.006747307255864143, -0.0001249149441719055, -0.025619836896657942],
    ],
    "AM-MBIS QUADRUPOLES": [
        [
            [0.10379766821861267, 0.0001027718186378479, -0.06015831157565117],
            [0.0001027718186378479, -0.06279769167304039, 0.0001027718186378479],
            [-0.06015831157565117, 0.0001027718186378479, -0.04099997952580452],
        ],
        [
            [0.0007394148036837578, 0.0001027718186378479, 0.0031744487583637237],
            [0.0001027718186378479, -0.003916253708302975, 0.0001027718186378479],
            [0.0031744487583637237, 0.0001027718186378479, 0.0031768389861099424],
        ],
        [
            [-0.0012837066780775786, 0.0001027718186378479, 0.0007437553256750106],
            [0.0001027718186378479, -0.003916237084195018, 0.0001027718186378479],
            [0.0007437553256750106, 0.0001027718186378479, 0.00519994399510324],
        ],
    ],
    "geometry": (
        "0 1\n"
        "O                     0.000000000000     0.000000000000"
        "     0.000000000000\n"
        "H                     1.433550020000     0.000000000000"
        "     0.952958650000\n"
        "H                     0.492188620000     0.000000000000"
        "    -1.649528710000\n"
        "units bohr\n"
    ),
}

EXPECTED_AP2_OUTPUT = {
    "APNet2 TOTAL INTERACTION (kcal/mol)": -1.1231137216091156,
    "APNet2 ELSTROSTATICS (kcal/mol)": -2.0493468761444094,
    "APNet2 EXCHANGE (kcal/mol)": 2.491750621795654,
    "APNet2 INDUCTION (kcal/mol)": -0.5718475759029389,
    "APNet2 DISPERSION (kcal/mol)": -0.9936698913574219,
    "geometry": (
        "0 1\n"
        "--\n"
        "0 1\n"
        "O                     0.000000000000     0.000000000000"
        "     0.000000000000\n"
        "H                     1.433550020000     0.000000000000"
        "     0.952958650000\n"
        "H                     0.492188620000     0.000000000000"
        "    -1.649528710000\n"
        "--\n"
        "0 1\n"
        "O                     5.669178380000     0.944863060000"
        "     0.000000000000\n"
        "H                     7.102728390000     0.944863060000"
        "     0.952958650000\n"
        "H                     6.161366990000     0.944863060000"
        "    -1.649528710000\n"
        "units bohr\n"
    ),
}

EXPECTED_DAPNET2_OUTPUT = {
    "ERROR ESTIMATES (kcal/mol)": [-0.32446742057800293],
}

EXPECTED_MP2_POLYNOMIAL_OUTPUT = {
    "input_values": {"nbf_aux": 50, "nocc": 10, "nvirt": 50},
    "log_time": 0.9166507282477321,
    "method": "mp2",
    "time_seconds": 8.25373893848617,
    "variables_used": ["nocc", "nvirt", "nbf_aux"],
}


def test_predict_am_multipoles_default_output():
    output = SERVER.predict_AM_multipoles_QCMLForge(
        p4_string=DEFAULT_AM_P4_STRING,
    )

    assert output["geometry"] == EXPECTED_AM_OUTPUT["geometry"]
    np.testing.assert_allclose(
        np.asarray(output["AM-MBIS CHARGES"]),
        np.asarray(EXPECTED_AM_OUTPUT["AM-MBIS CHARGES"]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(output["AM-MBIS DIPOLES"]),
        np.asarray(EXPECTED_AM_OUTPUT["AM-MBIS DIPOLES"]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(output["AM-MBIS QUADRUPOLES"]),
        np.asarray(EXPECTED_AM_OUTPUT["AM-MBIS QUADRUPOLES"]),
        atol=1e-6,
    )


def test_predict_apnet2_default_output():
    output = SERVER.predict_APNet2_IE_QCMLForge(
        p4_string=DEFAULT_AP2_P4_STRING,
    )

    assert output["geometry"] == EXPECTED_AP2_OUTPUT["geometry"]
    for key, expected in EXPECTED_AP2_OUTPUT.items():
        if key == "geometry":
            continue
        assert output[key] == pytest.approx(expected, abs=1e-6)


def test_predict_dapnet2_single_default_output():
    output = SERVER.predict_dAPNet2_error_estimates_QCMLForge(
        p4_string=DEFAULT_AP2_P4_STRING,
        starting_level_of_theory=DEFAULT_STARTING_LEVEL,
    )

    assert output == EXPECTED_DAPNET2_OUTPUT


def test_predict_dapnet2_multiple_default_output():
    output = SERVER.predict_dAPNet2_error_estimates_QCMLForge_molecules(
        p4_strings=DEFAULT_DAPNET2_P4_STRINGS.copy(),
        starting_level_of_theory=DEFAULT_STARTING_LEVEL,
    )

    assert output == EXPECTED_DAPNET2_OUTPUT


def test_estimate_timing_for_qcel_molecule_default_output():
    if not SERVER.is_psi4_installed():
        pytest.skip("Psi4 is not installed in this environment")

    output = SERVER.estimate_timing_for_qcel_molecule(
        p4_string=DEFAULT_AM_P4_STRING,
        method=DEFAULT_TIMING_METHOD,
        basis_set=DEFAULT_TIMING_BASIS,
        manybody=True,
    )

    assert isinstance(output["estimated_compute_time_seconds"], float)
    assert output["estimated_compute_time_seconds"] > 0.0
    assert output["geometry"] == EXPECTED_AM_OUTPUT["geometry"]
    pp(output)


def test_polynomial_eval():
    method = "mp2"
    input_vars = {
        "nocc": 10,
        "nvirt": 50,
        "nbf_aux": 50,
        "np_total": 100000,
    }
    result = qcml_mcp.timings.estimate_timings.predict_timing(method, input_vars)
    pp(result)
    assert result["method"] == EXPECTED_MP2_POLYNOMIAL_OUTPUT["method"]
    assert result["variables_used"] == EXPECTED_MP2_POLYNOMIAL_OUTPUT["variables_used"]
    assert result["input_values"] == EXPECTED_MP2_POLYNOMIAL_OUTPUT["input_values"]
    assert result["log_time"] == pytest.approx(EXPECTED_MP2_POLYNOMIAL_OUTPUT["log_time"], abs=1e-6)


def test_benzene_dimer_geometry():
    geometry = SERVER.benzene_dimer_geometry()
    mol = qcel.models.Molecule.from_data(geometry)

    assert geometry.endswith("units bohr\n")
    assert len(mol.fragments) == 2
    assert len(mol.atomic_numbers) == 24
    assert mol.atomic_numbers.tolist().count(6) == 12
    assert mol.atomic_numbers.tolist().count(1) == 12
    print(mol)


if __name__ == "__main__":
    test_polynomial_eval()
