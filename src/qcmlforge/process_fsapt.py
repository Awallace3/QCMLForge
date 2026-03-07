import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint as pp
import qcelemental as qcel
from psi4.driver.procrouting.sapt import fsapt
from glob import glob


def gather_input_qcel_molecule(input_file_path: str) -> qcel.models.Molecule:
    with open(input_file_path, "r") as f:
        input_data = f.read()
    # split out mol definition until end of {} block
    mol_str = input_data.split("mol{")[1]
    mol_str = mol_str.split("}")[0]
    mol = qcel.models.Molecule.from_data(mol_str)
    return mol


def gather_fsapt_data(dirname: str) -> pd.DataFrame:
    data = fsapt.run_from_output(dirname=dirname)
    df = pd.DataFrame(data)
    return df


def fsapt_data(base_dir: str) -> pd.DataFrame:
    df = gather_fsapt_data(f"{base_dir}/fsapt")
    mol = gather_input_qcel_molecule(f"{base_dir}/input_file.in")
    df["qcel_molecule"] = [mol for i in range(len(df))]
    return df


def main():
    frames = []
    for d in glob("./pbd_dir/*"):
        print(f"Processing directory: {d}")
        df = fsapt_data(d)
        frames.append(df)
        break
    df = pd.concat(frames, ignore_index=True)
    pp(df.columns.tolist())
    df["F-Induction"] = df["IndAB"] + df["IndBA"]
    df.drop(columns=["IndAB", "IndBA"], inplace=True)
    df = df.rename(
        columns={
            "Elst": "F-Electrostatics",
            "Exch": "F-Exchange",
            "Disp": "F-Dispersion",
            "EDisp": "F-EDispersion",
            "Total": "F-Total",
        },
    )
    print(df)
    pp(df.columns.tolist())
    df.to_pickle("fsapt_data.pkl")
    return


if __name__ == "__main__":
    main()
