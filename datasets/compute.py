import os
import argparse
import pandas as pd
from tqdm import tqdm
from qcportal import PortalClient
import numpy as np

# from qcportal.manybody import ManybodyDataset, ManybodyDatasetEntry, ManybodySpecification, ManybodyKeywords, BSSECorrectionEnum
from qcportal.singlepoint import SinglepointDataset, SinglepointDatasetEntry
from qcportal.singlepoint import QCSpecification
from qcelemental.models import Molecule
from pprint import pprint as pp
from qm_tools_aw import tools
import qcelemental as qcel

client = PortalClient("http://localhost:7777", verify=False)
print(client)
ROOT = os.getcwd()

# df['qcel'] are qcelemental monomer Molecule objects
# df.head().to_pickle("./data_dir/raw/monomers_ap3_spec_1_head.pkl")
# df = pd.read_pickle("../data_dir/raw/monomers_ap3_spec_1_head.pkl")
# print(df.columns.values)


def add_QCdataset(
    df: pd.DataFrame | None,
    system: str = "AP2-monomers",
    method: str = "PBE0",
    basis: str = "aug-cc-pVTZ",
    mode: str = "create",
) -> SinglepointDataset:
    """Add a dataset with molecules and specification to a QCFractal client.

    This function creates a new `SinglepointDataset` on the connected QCFractal
    server or retrieves an existing one. It can then populate the dataset with
    molecular entries from a pandas DataFrame and add a computational
    specification for single-point energy calculations.

    Parameters
    ----------
    df : pd.DataFrame or None
        A DataFrame containing molecular information. It must have columns
        "name" and "qcel", where "qcel" contains QCElemental Molecule objects.
        This is only required when ``mode="create"``.
    system : str, optional
        The name of the dataset on the QCFractal server. Defaults to
        "AP2-monomers".
    method : str, optional
        The quantum chemistry method to be used for the calculations (e.g.,
        "PBE0"). Defaults to "PBE0".
    basis : str, optional
        The basis set to be used for the calculations (e.g., "aug-cc-pVTZ").
        Defaults to "aug-cc-pVTZ".
    mode : str, optional
        The operational mode. If "create", the function will add molecules
        and a specification to the dataset. Defaults to "create".

    Returns
    -------
    SinglepointDataset
        The created or retrieved QCFractal dataset object.

    """
    # client.delete_dataset(dataset_id=2, delete_records=True)
    try:
        ds = client.add_dataset("singlepoint", system, f"Dataset to contain {system}")
        print(f"Added {system} as dataset")
    except Exception:
        ds = client.get_dataset("singlepoint", system)
        print(f"Found {system} dataset, using this instead")
        print(ds)

    if mode == "create":
        if df is None:
            raise ValueError("A DataFrame is required when mode='create'.")
        entry_list = []
        for idx, row in df.iterrows():
            name = row["name"]
            extras = {
                "name": name,
                "idx": idx,
            }
            mol = row["qcel"]
            mol = Molecule.from_data(mol.dict(), extras=extras)
            ent = SinglepointDatasetEntry(name=name, molecule=mol)
            entry_list.append(ent)

        ds.add_entries(entry_list)
        print(f"Added {len(entry_list)} molecules to dataset")

        spec = QCSpecification(
            program="psi4",
            driver="energy",
            method=method,
            basis=basis,
            keywords={
                "d_convergence": 8,
                "dft_radial_points": 99,
                "dft_spherical_points": 590,
                "e_convergence": 10,
                "guess": "sad",
                "mbis_d_convergence": 9,
                "mbis_maxiter": 1000,
                "mbis_radial_points": 99,
                "mbis_spherical_points": 590,
                "scf_properties": ["mbis_charges", "MBIS_VOLUME_RATIOS"],
                "scf_type": "df",
            },
            protocols={"wavefunction": "orbitals_and_eigenvalues"},
        )
        ds.add_specification(name=f"psi4/{method}/{basis}", specification=spec)
        print(f"Added {method}/{basis} specification to dataset")
    return ds


def submit(ds: SinglepointDataset, tag: str = "short") -> None:
    """Submit a dataset for computation.

    This function submits all entries and specifications in a QCFractal
    dataset for computation. It is a simple wrapper around the `submit`
    method of the `SinglepointDataset` object.

    Parameters
    ----------
    ds : SinglepointDataset
        The QCFractal dataset to be submitted.
    tag : str, optional
        A tag to be associated with the submitted computations. Defaults to "short".

    """
    ds.submit(tag=tag)  # will default to submit all specifications for all entries
    print(f"Submitted {ds.name} dataset")


def analyze(ds: SinglepointDataset, output: str) -> pd.DataFrame:
    """Collect completed QCFractal results and save them to disk.

    Parameters
    ----------
    ds : SinglepointDataset
        Dataset to analyze.
    output : str
        Path where the processed DataFrame will be written.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed MBIS properties.

    """
    ds.status()
    ds.print_status()

    data = {
        "id": [],
        "name": [],
        "Z": [],
        "R": [],
        "cartesian_multipoles": [],
        "entry_name": [],
        "spec_name": [],
        "TQ": [],
        "molecular_multiplicity": [],
        "volume ratios": [],
        "valence widths": [],
        "radial moments <r^2>": [],
        "radial moments <r^3>": [],
        "radial moments <r^4>": [],
    }
    for entry_name, spec_name, record in tqdm(ds.iterate_records(status="complete")):
        record_dict = record.dict()
        qcvars = record_dict["properties"]
        charges = qcvars["mbis charges"]
        dipoles = qcvars["mbis dipoles"]
        quadrupoles = qcvars["mbis quadrupoles"]

        n = len(charges)
        charges = np.reshape(charges, (n, 1))
        dipoles = np.reshape(dipoles, (n, 3))
        quad = np.reshape(quadrupoles, (n, 3, 3))
        quad = [q[np.triu_indices(3)] for q in quad]
        multipoles = np.concatenate([charges, dipoles, np.array(quad)], axis=1)

        data["volume ratios"].append(qcvars["mbis volume ratios"])
        data["valence widths"].append(qcvars["mbis valence widths"])
        data["radial moments <r^2>"].append(qcvars["mbis radial moments <r^2>"])
        data["radial moments <r^3>"].append(qcvars["mbis radial moments <r^3>"])
        data["radial moments <r^4>"].append(qcvars["mbis radial moments <r^4>"])
        data["id"].append(record.molecule.extras["idx"])
        data["name"].append(record.molecule.extras["name"])
        data["Z"].append(record.molecule.atomic_numbers)
        data["R"].append(record.molecule.geometry * qcel.constants.bohr2angstroms)
        data["cartesian_multipoles"].append(multipoles)
        data["entry_name"].append(entry_name)
        data["spec_name"].append(spec_name)
        data["TQ"].append(int(record.molecule.molecular_charge))
        data["molecular_multiplicity"].append(record.molecule.molecular_multiplicity)

    df = pd.DataFrame(data)
    df.to_pickle(output)
    print(df)
    print(df.columns.values)
    return df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset workflow management."""
    parser = argparse.ArgumentParser(
        description="Manage QCFractal singlepoint datasets."
    )
    parser.add_argument(
        "mode",
        choices=["create", "run", "progress", "analyze"],
        help="Workflow stage to execute.",
    )
    parser.add_argument(
        "--input",
        default="../data_dir/raw/monomers_ap3_spec_1.pkl",
        help="Pickle file used to create the dataset.",
    )
    parser.add_argument(
        "--output",
        default="../data_dir/raw/monomers_ap3_spec_1_pbe0.pkl",
        help="Pickle file written during analyze mode.",
    )
    parser.add_argument(
        "--system",
        default="AP2-monomers",
        help="QCFractal dataset name.",
    )
    parser.add_argument("--method", default="PBE0", help="Quantum chemistry method.")
    parser.add_argument(
        "--basis", default="aug-cc-pVTZ", help="Basis set used for the dataset."
    )
    parser.add_argument(
        "--tag", default="normal", help="Submission tag used in run mode."
    )
    return parser.parse_args()


def main():
    """Main function to manage the QC dataset workflow.

    This function serves as the main entry point for a workflow that can
    create, run, monitor, and analyze a quantum chemistry dataset using
    QCFractal. The workflow is controlled by the `mode` variable, which
    can be set to "create", "run", "progress", or "analyze".

    The function performs the following steps based on the selected mode:
    - **create**: Creates a new dataset, adds molecules from a pickled
      DataFrame, and defines a computational specification.
    - **run**: Submits the created dataset for computation.
    - **progress**: Checks and prints the status of the computations in the
      dataset.
    - **analyze**: Retrieves the results of completed computations, processes
      them to extract properties like MBIS charges and multipoles, and saves
      the processed data to a new pickled DataFrame.

    The dataset, molecular data, and results are managed through a QCFractal
    server, and data is read from and saved to pickled pandas DataFrames.

    """
    args = parse_args()
    print(f"Running in {args.mode} mode")

    df = None
    if args.mode == "create":
        df = pd.read_pickle(args.input)

    ds = add_QCdataset(
        df,
        args.system,
        method=args.method,
        basis=args.basis,
        mode=args.mode,
    )

    if args.mode == "run":
        submit(ds, tag=args.tag)
    elif args.mode == "progress":
        ds.status()
        ds.print_status()
    elif args.mode == "analyze":
        analyze(ds, args.output)


if __name__ == "__main__":
    main()
