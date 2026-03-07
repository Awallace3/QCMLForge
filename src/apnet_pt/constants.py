"""
Fixed constants (units, atom types, etc.)
"""

import qcelemental as qcel
import pandas as pd
from importlib import resources
import torch

au2ang = qcel.constants.conversion_factor("bohr", "angstrom")
h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")

elem_to_z = { 
    'H'  : 1,
    'B'  : 5,
    'C'  : 6,
    'N'  : 7,
    'O'  : 8,
    'F'  : 9,
    'NA' : 11,
    'Na' : 11,
    'P'  : 15, 
    'S'  : 16, 
    'CL' : 17, 
    'Cl' : 17, 
    'BR' : 35, 
    'Br' : 35, 
}

z_to_elem = { 
    1  : 'H',
    5  : 'B',
    6  : 'C',
    7  : 'N',
    8  : 'O',
    9  : 'F',
    11 : 'Na',
    15 : 'P', 
    16 : 'S', 
    17 : 'Cl', 
    35 : 'Br', 
}

z_to_ind = {
    1  : 0,
    5  : 1,
    6  : 2,
    7  : 3,
    8  : 4,
    9  : 5,
    11 : 6,
    15 : 7,
    16 : 8,
    17 : 9,
    35 : 10,
}

pol_free = {
  'H' :  4.50,
  'He':  1.38,
  'C' : 12.00,
  'N' :  7.40,
  'O' :  5.40,
  'F' :  3.80,
  'Ne':  2.67,
  'Si': 37.00,
  'P' : 25.00,
  'S' : 19.60,
  'Cl': 15.00,
  'Ar': 11.10,
  'Br': 20.00,
  'Kr': 16.80,
  'I' : 35.00,
}

libmbd_vwd_params = pd.read_csv(
    # osp.join(current_file_path, "data", "vdw-params.csv"),
    resources.files(
        "apnet_pt",
    ).joinpath("data", "vdw-params.csv"),
    header=0,
    index_col=0,
    sep=",",
    nrows=102,
)
free_atom_polarizabilities = {
    # el: v for el, v in zip(libmbd_vwd_params['Z'], libmbd_vwd_params['alpha_0(TS)'])
    el: v
    for el, v in zip(libmbd_vwd_params["Z"], libmbd_vwd_params["alpha_0(BG)"])
}

# from pprint import pprint as pp
# pp(free_atom_polarizabilities)

# Create lookup table as a tensor
max_z = max(free_atom_polarizabilities.keys())
polarizability_table = torch.zeros(max_z + 1)
for z, value in free_atom_polarizabilities.items():
    polarizability_table[z] = value
