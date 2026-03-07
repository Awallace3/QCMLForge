import numpy as np
import qcelemental as qcel
from typing import List, Tuple, Dict
from pprint import pprint as pp
from mcp.server.fastmcp import FastMCP
import apnet_pt

try:
    from .timings import is_psi4_installed
    from .timings import estimate_timings
except ImportError:
    # Fall back to absolute imports when run as a script
    from timings import is_psi4_installed
    from timings import estimate_timings


# Create an MCP server
mcp = FastMCP("QCMLForge", port=8001)


@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two numbers.

    Parameters
    ----------
    a : int
        First number to add.
    b : int
        Second number to add.

    Returns
    -------
    int
        Sum of a and b.
    """
    return a + b


@mcp.tool()
def predict_AM_multipoles_QCMLForge(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
units angstrom
    """,
) -> Dict:
    """
    Run a user defined molecule to get machine-learned atomic multipoles.

    Predicts atomic multipoles for evaluating electrostatics and polarization
    energies using the AtomicModule in QCMLForge. This approximates MBIS
    multipoles. The p4_string defines the molecular geometry in Psi4 format,
    which can be of the format:
    '''
    <charge_mon1> <multiplicity_mon1>
    <atom_symbol> <x> <y> <z>
    <atom_symbol> <x> <y> <z>
    --
    <charge_mon2> <multiplicity_mon2>
    units <unit>
    '''
    Note that "--" is used to separate different molecules in the input string
    but is not required for monomers.

    Parameters
    ----------
    p4_string : str, optional
        Molecular geometry in Psi4 format with charge, multiplicity, atomic
        symbols, coordinates, and units. Default is a water molecule geometry.

    Returns
    -------
    dict
        Dictionary containing:
        - "AM-MBIS CHARGES" : list
            Atomic charges from the AtomModel.
        - "AM-MBIS DIPOLES" : list
            Atomic dipoles from the AtomModel.
        - "AM-MBIS QUADRUPOLES" : list
            Atomic quadrupoles from the AtomModel.
        - "geometry" : str
            Molecular geometry in Psi4 format.
    """
    mol = qcel.models.Molecule.from_data(p4_string)
    charges, dipoles, quadrupoles, _ = apnet_pt.pretrained_models.atom_model_predict(
        [mol],
        compile=False,
        return_mol_arrays=False,
    )
    return {
        "AM-MBIS CHARGES": list(charges),
        "AM-MBIS DIPOLES": list(dipoles),
        "AM-MBIS QUADRUPOLES": list(quadrupoles),
        "geometry": mol.to_string("psi4"),
    }


@mcp.tool()
def secret_word():
    """
    Return the secret word for QCMLForge to test server.

    Parameters
    ----------
    None

    Returns
    -------
    str
    """
    return "QCMLForgeRocks!"


@mcp.tool()
def predict_APNet2_IE_QCMLForge(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
units angstrom
    """,
) -> Dict:
    """
    Predict machine-learned SAPT0 interaction energies for a molecular complex.

    Computes total interaction energy and its components (electrostatics,
    exchange, induction, and dispersion) using the APNet2 model in QCMLForge.
    The p4_string defines the molecular geometry in Psi4 format, which can be
    of the format:
    '''
    <charge_mon1> <multiplicity_mon1>
    <atom_symbol> <x> <y> <z>
    <atom_symbol> <x> <y> <z>
    --
    <charge_mon2> <multiplicity_mon2>
    units <unit>
    '''
    Note that "--" is used to separate different molecules in the input string
    but is not required for monomers.

    Parameters
    ----------
    p4_string : str, optional
        Molecular geometry in Psi4 format with charge, multiplicity, atomic
        symbols, coordinates, and units. Default is a water dimer geometry.

    Returns
    -------
    dict
        Dictionary containing:
        - "APNet2 TOTAL INTERACTION (kcal/mol)" : float
            Total interaction energy.
        - "APNet2 ELSTROSTATICS (kcal/mol)" : float
            Electrostatic component of interaction energy.
        - "APNet2 EXCHANGE (kcal/mol)" : float
            Exchange component of interaction energy.
        - "APNet2 INDUCTION (kcal/mol)" : float
            Induction component of interaction energy.
        - "APNet2 DISPERSION (kcal/mol)" : float
            Dispersion component of interaction energy.
        - "geometry" : str
            Molecular geometry in Psi4 format.
    """
    mol = qcel.models.Molecule.from_data(p4_string)
    IE_pred = apnet_pt.pretrained_models.apnet2_model_predict(
        [mol],
        compile=False,
    )
    return {
        "APNet2 TOTAL INTERACTION (kcal/mol)": float(IE_pred[0, 0]),
        "APNet2 ELSTROSTATICS (kcal/mol)": float(IE_pred[0, 1]),
        "APNet2 EXCHANGE (kcal/mol)": float(IE_pred[0, 2]),
        "APNet2 INDUCTION (kcal/mol)": float(IE_pred[0, 3]),
        "APNet2 DISPERSION (kcal/mol)": float(IE_pred[0, 4]),
        "geometry": mol.to_string("psi4"),
    }


@mcp.tool()
def predict_dAPNet2_error_estimates_QCMLForge(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000  0.000000
H 3.758602 0.500000  0.504284
H 3.260455 0.500000 -0.872893
units angstrom
    """,
    starting_level_of_theory: str = "MP2/aug-cc-pVTZ/CP",
) -> Dict:
    """
    Predict error between a starting level of theory and CCSD(T)/CBS/CP reference.

    Estimates the interaction energy error using the dAPNet2 model in QCMLForge
    for a single molecular complex. The p4_string defines the molecular geometry
    in Psi4 format.

    Acceptable starting_level_of_theory values currently only include:
    [
    "B3LYP-D3/aug-cc-pVTZ/unCP",
    "B2PLYP-D3/aug-cc-pVTZ/unCP",
    "wB97X-V/aug-cc-pVTZ/CP",
    "SAPT0/aug-cc-pVDZ/SA",
    "MP2/aug-cc-pVTZ/CP",
    "HF/aug-cc-pVDZ/CP",
    ]

    Parameters
    ----------
    p4_string : str, optional
        Molecular geometry in Psi4 format with charge, multiplicity, atomic
        symbols, coordinates, and units. Format:
        '''
        <charge_mon1> <multiplicity_mon1>
        <atom_symbol> <x> <y> <z>
        <atom_symbol> <x> <y> <z>
        --
        <charge_mon2> <multiplicity_mon2>
        units <unit>
        '''
        Note that "--" is used to separate different molecules in the input string
        but is not required for monomers. Default is a water dimer geometry.
    starting_level_of_theory : str, optional
        Level of theory for which to estimate error relative to CCSD(T)/CBS/CP.
        Default is "MP2/aug-cc-pVTZ/CP".

    Returns
    -------
    dict
        Dictionary containing:
        - "ERROR ESTIMATES (kcal/mol)" : array
            Predicted error between starting_level_of_theory and CCSD(T)/CBS/CP.
    """
    mol = qcel.models.Molecule.from_data(p4_string)
    IE_pred = apnet_pt.pretrained_models.dapnet2_model_predict(
        [mol],
        compile=False,
        m1=starting_level_of_theory,
        m2="CCSD(T)/CBS/CP",
    )
    return {
        "ERROR ESTIMATES (kcal/mol)": IE_pred.tolist(),
    }


@mcp.tool()
def predict_dAPNet2_error_estimates_QCMLForge_molecules(
    p4_strings: list[str] = [
        """0 1
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
    ],
    starting_level_of_theory: str = "MP2/aug-cc-pVTZ/CP",
) -> Dict:
    """
    Predict error estimates for multiple molecular complexes.

    Estimates the interaction energy error between a starting level of theory
    and CCSD(T)/CBS/CP reference using the dAPNet2 model in QCMLForge for
    multiple molecular complexes. Each p4_string defines a molecular geometry
    in Psi4 format.

    Acceptable starting_level_of_theory values currently only include:
    [
    "B3LYP-D3/aug-cc-pVTZ/unCP",
    "B2PLYP-D3/aug-cc-pVTZ/unCP",
    "wB97X-V/aug-cc-pVTZ/CP",
    "SAPT0/aug-cc-pVDZ/SA",
    "MP2/aug-cc-pVTZ/CP",
    "HF/aug-cc-pVDZ/CP",
    ]

    Parameters
    ----------
    p4_strings : list[str], optional
        List of molecular geometries in Psi4 format with charge, multiplicity,
        atomic symbols, coordinates, and units. Format:
        '''
        <charge_mon1> <multiplicity_mon1>
        <atom_symbol> <x> <y> <z>
        <atom_symbol> <x> <y> <z>
        --
        <charge_mon2> <multiplicity_mon2>
        units <unit>
        '''
        Note that "--" is used to separate different molecules in the input string
        but is not required for monomers. Default is a list with one water dimer.
    starting_level_of_theory : str, optional
        Level of theory for which to estimate error relative to CCSD(T)/CBS/CP.
        Default is "MP2/aug-cc-pVTZ/CP".

    Returns
    -------
    dict
        Dictionary containing:
        - "ERROR ESTIMATES (kcal/mol)" : array
            Predicted errors for each molecule in p4_strings.
    """
    mols = [qcel.models.Molecule.from_data(i) for i in p4_strings]
    IE_pred = apnet_pt.pretrained_models.dapnet2_model_predict(
        mols,
        compile=False,
        m1=starting_level_of_theory,
        m2="CCSD(T)/CBS/CP",
    )
    return {
        "ERROR ESTIMATES (kcal/mol)": IE_pred.tolist(),
    }


@mcp.tool()
def estimate_timing_for_qcel_molecule(
    p4_string: str = """0 1
O 0.000000 0.000000  0.000000
H 0.758602 0.000000  0.504284
H 0.260455 0.000000 -0.872893
units angstrom
    """,
    method: str = "MP2",
    basis_set: str = "aug-cc-pVDZ",
    manybody: bool = True,
) -> Dict:
    """
    Estimate computational timing for a molecular system.

    Computes the necessary variables for timing estimation for monomers and
    dimers. When manybody is True, it computes the timing for the dimer and
    each monomer separately to mimic a supermolecular interaction energy
    calculation. Manybody should also be set when counterpoise correction (CP)
    is requested.

    Allowed methods: 'mp2', 'hf', 'b2plyp-d3', 'b3lyp-d3', 'pbe-d3', 'm05-2x',
    'wb97x-v', 'wb97x-d', 'fno-ccsd', 'fno-ccsd(t)'.

    Parameters
    ----------
    p4_string : str, optional
        Molecular geometry in Psi4 format with charge, multiplicity, atomic
        symbols, coordinates, and units. Format:
        '''
        <charge_mon1> <multiplicity_mon1>
        <atom_symbol> <x> <y> <z>
        <atom_symbol> <x> <y> <z>
        --
        <charge_mon2> <multiplicity_mon2>
        units <unit>
        '''
        Note that "--" separates different molecules in the input string.
        Default is a water molecule geometry.
    method : str, optional
        Computational method to use. Default is 'MP2'.
    basis_set : str, optional
        Basis set to use. Default is 'aug-cc-pVDZ'.
    manybody : bool, optional
        Whether to compute timing for dimer and monomers separately.
        Default is True.

    Returns
    -------
    dict
        Dictionary containing:
        - "estimated_compute_time_seconds" : float
            Estimated computational time in seconds.
        - "geometry" : str
            Molecular geometry in Psi4 format.
    """
    qcel_molecule = qcel.models.Molecule.from_data(p4_string)
    if is_psi4_installed() is False:
        print("Psi4 is not installed. Please install Psi4 to use this function.")
    mols = [qcel_molecule]
    if manybody and qcel_molecule.fragments_:
        for n, i in enumerate(qcel_molecule.fragments_):
            mols.append(qcel_molecule.get_fragment(n))

    method = method.lower()
    time_seconds = 0.0
    for mol in mols:
        n_occupied, n_virtual, np_total, nbf_aux = (
            estimate_timings.compute_psi4_time_estimation_variables(
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
        result = estimate_timings.predict_timing(method, input_vars)
        time_seconds += result["time_seconds"]
    return {
        "estimated_compute_time_seconds": time_seconds,
        "geometry": mol.to_string("psi4"),
    }


@mcp.tool()
def benzene_dimer_geometry() -> str:
    """
    Provide parallel displaced benzene dimer geometry.

    Returns a pre-defined benzene dimer geometry in parallel displaced
    configuration.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Benzene dimer geometry in Psi4 format (p4_str).
    """
    mol = qcel.models.Molecule.from_data("""0 1
C	0.7500000000	-1.6000000000	-1.3915000000
C	1.9550743494	-1.6000000000	-0.6957500000
C	1.9550743494	-1.6000000000	0.6957500000
C	0.7500000000	-1.6000000000	1.3915000000
C	-0.4550743494	-1.6000000000	0.6957500000
C	-0.4550743494	-1.6000000000	-0.6957500000
H	0.7500000000	-1.6000000000	-2.4715000000
H	2.8903817855	-1.6000000000	-1.2357500000
H	2.8903817855	-1.6000000000	1.2357500000
H	0.7500000000	-1.6000000000	2.4715000000
H	-1.3903817855	-1.6000000000	1.2357500000
H	-1.3903817855	-1.6000000000	-1.2357500000
--
0 1
C	-0.7500000000	1.6000000000	1.3915000000
C	0.4550743494	1.6000000000	0.6957500000
C	0.4550743494	1.6000000000	-0.6957500000
C	-0.7500000000	1.6000000000	-1.3915000000
C	-1.9550743494	1.6000000000	-0.6957500000
C	-1.9550743494	1.6000000000	0.6957500000
H	-0.7500000000	1.6000000000	2.4715000000
H	1.3903817855	1.6000000000	1.2357500000
H	1.3903817855	1.6000000000	-1.2357500000
H	-0.7500000000	1.6000000000	-2.4715000000
H	-2.8903817855	1.6000000000	-1.2357500000
H	-2.8903817855	1.6000000000	1.2357500000
units angstrom
""")
    return mol.to_string("psi4")


if __name__ == "__main__":
    print("Starting MCP server...")
    # pp(estimate_timing_for_qcel_molecule(benzene_dimer_geometry(), method="hf", basis_set="aug-cc-pVDZ", manybody=True))
    # pp(estimate_timing_for_qcel_molecule(benzene_dimer_geometry(), method="fno-ccsd(t)", basis_set="aug-cc-pVDZ", manybody=True))
    # pp(predict_dAPNet2_error_estimates_QCMLForge(benzene_dimer_geometry(), starting_level_of_theory="HF/aug-cc-pVDZ/CP"))
    # pp(predict_AM_multipoles_QCMLForge())
    # pp(predict_APNet2_IE_QCMLForge())
    # pp(predict_dAPNet2_error_estimates_QCMLForge())
    # pp(estimate_timing_for_qcel_molecule())
