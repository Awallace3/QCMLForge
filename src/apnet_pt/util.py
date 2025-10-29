"""
General utility functions for pre-processing molecules
"""

import numpy as np
import pandas as pd
import qcelemental as qcel
import torch

from apnet_pt import constants

# from numba import jit
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


def scatter_sum_compile(src, index, dim_size, reduce="sum"):
    """
    Compile-friendly version of torch_geometric scatter for sum reduction.
    
    This function avoids graph breaks in torch.compile by using native PyTorch
    operations instead of torch_geometric.utils.scatter which contains .item()
    calls that cause dynamo graph breaks.
    
    Parameters
    ----------
    src : torch.Tensor
        Source tensor to scatter (any number of dimensions)
    index : torch.Tensor
        Index tensor indicating where to scatter each element (1D)
    dim_size : int
        Size of the output dimension 0
    reduce : str
        Reduction operation, must be "sum" or "add"
    
    Returns
    -------
    torch.Tensor
        Scattered tensor with shape (dim_size, *src.shape[1:])
    
    Notes
    -----
    This function only supports reduce="sum" and reduce="add" which are equivalent.
    For other reduction types, use torch_geometric.utils.scatter.
    """
    # if reduce not in ["sum", "add"]:
    #     raise ValueError(f"Only 'sum' and 'add' reductions supported, got '{reduce}'")
    
    out_shape = (dim_size,) + src.shape[1:]
    output = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    
    index_shape = (index.size(0),) + (1,) * (src.dim() - 1)
    index_expanded = index.view(index_shape).expand_as(src)
    
    result = output.scatter_add_(0, index_expanded, src)
    
    return result


def qcel_to_dimerdata(dimer):
    """proper qcel mol to ML-ready numpy arrays"""

    # this better be a dimer (not a monomer, trimer, etc.)
    if len(dimer.fragments) != 2:
        # raise AssertionError(f"A dimer must have exactly 2 molecular fragments, found {len(dimer.fragments)}")
        return None

    ZA = dimer.symbols[dimer.fragments[0]]
    ZB = dimer.symbols[dimer.fragments[1]]

    # only some elements allowed
    try:
        ZA = np.array([constants.elem_to_z[za] for za in ZA])
        ZB = np.array([constants.elem_to_z[zb] for zb in ZB])
    except Exception:
        return None

    RA = dimer.geometry[dimer.fragments[0]] * constants.au2ang
    RB = dimer.geometry[dimer.fragments[1]] * constants.au2ang

    nA = len(dimer.fragments[0])
    nB = len(dimer.fragments[1])
    aQA = dimer.fragment_charges[0] / nA
    aQB = dimer.fragment_charges[1] / nB

    return (RA, RB, ZA, ZB, aQA, aQB)


def qcel_to_monomerdata(monomer):
    """proper qcel mol to ML-ready numpy arrays"""

    # this better be a monomer
    if len(monomer.fragments) != 1:
        raise AssertionError(
            f"A monomer must have exactly 1 molecular fragment, found {len(monomer.fragments)}"
        )

    Z = monomer.symbols
    Z = np.array([constants.elem_to_z[z] for z in Z])

    R = monomer.geometry * constants.au2ang

    n = len(monomer.symbols)
    aQ = monomer.molecular_charge / n

    return (R, Z, aQ)


def dimerdata_to_qcel(RA, RB, ZA, ZB, aQA, aQB):
    """ML-ready numpy arrays to qcel mol"""

    nA = RA.shape[0]
    nB = RB.shape[0]
    ZA = [int(z) for z in ZA]
    ZB = [int(z) for z in ZB]

    tQA = int(round(aQA * nA))
    tQB = int(round(aQB * nB))

    assert abs(tQA - aQA * nA) < 1e-6
    assert abs(tQB - aQB * nB) < 1e-6

    blockA = f"{tQA} {1}\n"
    for ia in range(nA):
        blockA += f"{constants.z_to_elem[ZA[ia]]} {RA[ia, 0]} {RA[ia, 1]} {RA[ia, 2]}\n"

    blockB = f"{tQB} {1}\n"
    for ib in range(nB):
        blockB += f"{constants.z_to_elem[ZB[ib]]} {RB[ib, 0]} {RB[ib, 1]} {RB[ib, 2]}\n"

    dimer = blockA + "--\n" + blockB + "no_com\nno_reorient\nunits angstrom"
    dimer = qcel.models.Molecule.from_data(dimer)
    return dimer


def monomerdata_to_qcel(R, Z, aQ):
    """ML-ready numpy arrays to qcel mol"""

    n = R.shape[0]

    tQ = int(round(aQ * n))

    assert abs(tQ - aQ * n) < 1e-6

    block = f"{tQ} {1}\n"
    for i in range(n):
        block += f"{constants.z_to_elem[Z[i]]} {R[i, 0]} {R[i, 1]} {R[i, 2]}\n"

    monomer = block + "no_com\nno_reorient\nunits angstrom"
    monomer = qcel.models.Molecule.from_data(monomer)
    return monomer


def load_bms_dimer(file):
    """Load a single dimer from the BMS-xyz format

    This function expects an xyz file in the format used with the 1.66M dimer dataset.
    The first line contains the number of atoms.
    The second line contains a comma-separated list of values such as the dimer name, dimer and monomer charges, SAPT labels (at various levels of theory), and number of atoms in the first monomer.
    The next `natom` lines each contain an atomic symbol follwed by the x, y, and z cooordinates of the atom (Angstrom)

    Parameters
    ----------
    file : str
        The name of a file containing the xyz

    Returns
    -------
    dimer : :class:`~qcelemental.models.Molecule`
    labels : :class:`~numpy.ndarray`
        The SAPT0/aug-cc-pV(D+d)Z interaction energy labels: [total, electrostatics, exchange, induction, and dispersion].
    """

    lines = open(file, "r").readlines()

    natom = int(lines[0].strip())
    dimerinfo = ("".join(lines[1:-natom])).split(",")
    geom = lines[-natom:]

    nA = int(dimerinfo[-1])
    TQ = int(dimerinfo[1])
    TQA = int(dimerinfo[2])
    TQB = int(dimerinfo[3])
    assert TQ == (TQA + TQB)

    e_tot_aug = float(dimerinfo[14])
    e_elst_aug = float(dimerinfo[15])
    e_exch_aug = float(dimerinfo[16])
    e_ind_aug = float(dimerinfo[17])
    e_disp_aug = float(dimerinfo[18])

    assert abs(e_tot_aug - (e_elst_aug + e_exch_aug + e_ind_aug + e_disp_aug)) < 1e-6

    blockA = f"{TQA} 1\n" + "".join(geom[:nA])
    blockB = f"{TQB} 1\n" + "".join(geom[nA:])
    dimer = blockA + "--\n" + blockB + "no_com\nno_reorient\nunits angstrom"
    dimer = qcel.models.Molecule.from_data(dimer)

    label = np.array([e_tot_aug, e_elst_aug, e_exch_aug, e_ind_aug, e_disp_aug])
    return dimer, label


def load_dimer_dataset(
    file,
    max_size=None,
    columns=["Total_aug", "Elst_aug", "Exch_aug", "Ind_aug", "Disp_aug"],
    return_qcel_mols=True,
    return_qcel_mons=False,
    return_fragment_indices=False,
    random_seed_shuffle=None,
):
    """Load multiple dimers from a :class:`~pandas.DataFrame`

    Loads dimers from the :class:`~pandas.DataFrame` format associated with the original AP-Net publication.
    Each row of the :class:`~pandas.DataFrame` corresponds to a molecular dimer.

    The columns [`ZA`, `ZB`, `RA`, `RB`, `TQA`, `TQB`] are required.
    `ZA` and `ZB` are atom types (:class:`~numpy.ndarray` of `int` with shape (`n`,)).
    `RA` and `RB` are atomic positions in Angstrom (:class:`~numpy.ndarray` of `float` with shape (`n`,3.)).
    `TQA` and `TQB` are monomer charges (int).

    The columns [`Total_aug`, `Elst_aug`, `Exch_aug`, `Ind_aug`, and `Disp_aug`] are optional.
    Each column describes SAPT0/aug-cc-pV(D+d)Z labels in kcal / mol (`float`).

    Parameters
    ----------
    file : str
        The name of a file containing the :class:`~pandas.DataFrame`

    Returns
    -------
    dimers : list of :class:`~qcelemental.models.Molecule`
    labels : list of :class:`~numpy.ndarray` or None
        None is returned if SAPT0 label columns are not present in the :class:`~pandas.DataFrame`
    """

    df = pd.read_pickle(file)
    if len(columns) > 0:
        df.dropna(subset=columns, inplace=True)
    if random_seed_shuffle is not None:
        df = df.sample(frac=1, random_state=random_seed_shuffle).reset_index(drop=True)
    allowed_elements = constants.z_to_elem.keys()

    assert not ((return_qcel_mons != True) and (return_fragment_indices != True)), (
        "Cannot return both monomers and fragment indices simultaneously presently."
    )

    # if qcel_molecules is a column, use this as the source of truth to generate RA, RB, ZA, ZB, TQA, TQB
    if 'qcel_molecule' in df.columns:
        def extract_mol_data(mol):
            if all([z in allowed_elements for z in mol.atomic_numbers]) is False:
                return None, None, None, None, None, None
            monA, monB = mol.get_fragment(0), mol.get_fragment(1)

            RA = torch.tensor(monA.geometry, dtype=torch.float32) * constants.au2ang
            RB = torch.tensor(monB.geometry, dtype=torch.float32) * constants.au2ang
            ZA = torch.tensor(monA.atomic_numbers, dtype=torch.int64)
            ZB = torch.tensor(monB.atomic_numbers, dtype=torch.int64)

            TQA = torch.tensor(monA.molecular_charge, dtype=torch.float32)
            TQB = torch.tensor(monB.molecular_charge, dtype=torch.float32)
            return RA.numpy(), RB.numpy(), ZA.numpy(), ZB.numpy(), TQA.item(), TQB.item()
        v = df['qcel_molecule'].apply(extract_mol_data).tolist()
        # Filter out any None entries due to invalid elements
        v = [entry for entry in v if entry[0] is not None]
        N = len(v)
        RA, RB, ZA, ZB, TQA, TQB = zip(*v)
    else:
        len_df_start = len(df)
        df = df[df["ZA"].apply(lambda x: all([z in allowed_elements for z in x]))].copy()
        df = df[df["ZB"].apply(lambda x: all([z in allowed_elements for z in x]))].copy()
        len_df_end = len(df)
        if len_df_start != len_df_end:
            print(f"  Removed {len_df_start - len_df_end} rows with invalid elements")
        N = len(df.index)

        if max_size is not None and max_size < N:
            df = df.head(max_size)
            N = max_size

        RA = df.RA.tolist()
        RB = df.RB.tolist()
        ZA = df.ZA.tolist()
        ZB = df.ZB.tolist()
        TQA = df.TQA.tolist()
        TQB = df.TQB.tolist()
    aQA = [TQA[i] / np.sum(ZA[i] > 0) for i in range(N)]
    aQB = [TQB[i] / np.sum(ZB[i] > 0) for i in range(N)]
    try:
        labels = df[columns].to_numpy()
    except:
        labels = None

    if return_qcel_mons:
        monAs, monBs = [], []
        for i in range(N):
            monAs.append(monomerdata_to_qcel(RA[i], ZA[i], aQA[i]))
            monBs.append(monomerdata_to_qcel(RB[i], ZB[i], aQB[i]))
        return monAs, monBs, labels
    if return_qcel_mols:
        dimers = []
        for i in range(N):
            dimers.append(dimerdata_to_qcel(RA[i], RB[i], ZA[i], ZB[i], aQA[i], aQB[i]))

        return dimers, labels
    if return_fragment_indices:
        frag1_indices = df["Frag1_indices"].tolist()
        frag2_indices = df["Frag2_indices"].tolist()
        return RA, RB, ZA, ZB, TQA, TQB, labels, frag1_indices, frag2_indices
    return RA, RB, ZA, ZB, TQA, TQB, labels


def load_atomic_module_graph_dataset(
    file, R_c=5.0, edge_function="Bessel", max_size=None
):
    """Load multiple monomers from a :class:`~pandas.DataFrame`
    R_c is the cutoff radius for the atomic module graph construction (in Angstrom)

    Loads monomers from the :class:`~pandas.DataFrame` format associated with the original AP-Net publication.
    Each row of the :class:`~pandas.DataFrame` corresponds to a molecular dimer.

    The columns [`Z`, `R`, and `total_charge`] are required.
    `Z` is atom types (:class:`~numpy.ndarray` of `int` with shape (`n`,)).
    `R` is atomic positions in Angstrom (:class:`~numpy.ndarray` of `float` with shape (`n`,3)).
    `total_charge` are monomer charges (int).

    The columns [`cartesian_multipoles`, `volume_ratios`, and `valence_widths`] are optional.
    `cartesian_multipoles` describes atom-centered charges, dipoles, and quadrupoles (:class:`~numpy.ndarray` of `float` with shape (`n`, 10). The ordering convention is [q, u_x, u_y, u_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz], all in a.u.)
    `volume_ratios` is the ratio of the volume of the atom-in-molecule to the free atom (:class:`~numpy.ndarray` of `float` with shape (`n`, 1), unitless
    `valence_widths` is the width describing the valence electron density (:class:`~numpy.ndarray` of `float` with shape (`n`, 1), TODO: check units. a.u. ? inverse width?

    Parameters
    ----------
    file : str
        The name of a file containing the :class:`~pandas.DataFrame`

    Returns
    -------
    pyg_monomers
    """

    def distance_matrix(r):
        v = np.sqrt(
            np.sum(np.square(r[:, np.newaxis, :] - r[np.newaxis, :, :]), axis=-1)
        )
        return v

    if edge_function == "Bessel":
        # @jit(nopython=True, parallel=True)
        def vec_func(R_ij):
            edge_feature_vector = np.zeros((len(R_ij), len(R_ij)), dtype=np.float64)
            edge_index = []
            for i in range(R_ij.shape[0]):
                for j in range(R_ij.shape[1]):
                    if i != j and R_ij[i, j] < R_c:
                        r_ij = R_ij[i, j]
                        edge_feature_vector[i, j] = (
                            np.sqrt(2 / R_c) * np.sin(np.pi * r_ij / R_c) / r_ij
                        )
                        edge_index.append([i, j])
                        edge_index.append([j, i])
            return edge_feature_vector, edge_index

        def edge_function_system(R, r_c):
            edge_index = []
            dis_matrix = distance_matrix(R)
            print("dis_matrix:", dis_matrix)
            edge_feature_vector, edge_index = vec_func(dis_matrix)

            return edge_index, edge_feature_vector

        def edge_function_R(Rs, Zs, r_c):
            # edge_indices = []
            edge_feature_vectors = []
            dataset = []
            for i in range(len(Rs)):
                edge_index, edge_feature_vector = edge_function_system(Rs[i], r_c)
                zs = Zs[i]
                d = Data(x=zs, edge_attr=edge_feature_vector, edge_index=edge_index)
                print(d)
                dataset.append(d)
                # edge_indices.append(edge_index)
                # edge_feature_vectors.append(edge_feature_vector)
            # return edge_indices, edge_feature_vectors
            return dataset

    else:
        raise NotImplementedError(f"Edge function {edge_function} not implemented")

    df = pd.read_pickle(file)
    N = len(df.index)

    if max_size is not None and max_size < N:
        df = df.head(max_size)
        N = max_size
    print(df)

    R = df.R.tolist()
    Z = df.Z.tolist()
    TQ = df.total_charge.tolist()
    aQ = [TQ[i] / np.sum(Z[i] > 0) for i in range(N)]
    print(f"R: {R}")
    print(f"Z: {Z}")
    print(f"TQ: {TQ}")
    print(f"aQ: {aQ}")

    # Need edge_index [2, N] tensor for which nodes are connected (need to include bidirectionally) and edge_feature_vector [N, N]
    # edge_indices, edge_feature_vectors = edge_function_R(R, R_c)
    dataset = edge_function_R(R, Z, R_c)
    print(dataset)

    # print(f"edge_indices: {edge_indices}")
    # print(f"edge_feature_vectors: {edge_feature_vectors}")

    try:
        cartesian_multipoles = df["cartesian_multipoles"].to_numpy()
    except:
        cartesian_multipoles = None

    x_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    y_loader = DataLoader(
        [i for i in cartesian_multipoles], batch_size=32, shuffle=True
    )

    try:
        volume_ratios = df["volume_ratios"].to_numpy()
    except:
        volume_ratios = None

    try:
        valence_widths = df["valence_widths"].to_numpy()
    except:
        valence_widths = None

    monomers = []
    for i in range(N):
        monomers.append(monomerdata_to_qcel(R[i], Z[i], aQ[i]))
    return x_loader, y_loader


def load_monomer_dataset(
    file,
    max_size=None,
    R_label="R",
    Z_label="Z",
    TQ_label="TQ",
    cartesian_multipoles_label="cartesian_multipoles",
    hirshfeld_props=False,
    return_qcel_mons=False,
):
    """Load multiple monomers from a :class:`~pandas.DataFrame`

    Loads monomers from the :class:`~pandas.DataFrame` format associated with the original AP-Net publication.
    Each row of the :class:`~pandas.DataFrame` corresponds to a molecular dimer.

    The columns [`Z`, `R`, and `total_charge`] are required.
    `Z` is atom types (:class:`~numpy.ndarray` of `int` with shape (`n`,)).
    `R` is atomic positions in Angstrom (:class:`~numpy.ndarray` of `float` with shape (`n`,3)).
    `total_charge` are monomer charges (int).

    The columns [`cartesian_multipoles`, `volume_ratios`, and `valence_widths`] are optional.
    `cartesian_multipoles` describes atom-centered charges, dipoles, and quadrupoles (:class:`~numpy.ndarray` of `float` with shape (`n`, 10). The ordering convention is [q, u_x, u_y, u_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz], all in a.u.)
    `volume_ratios` is the ratio of the volume of the atom-in-molecule to the free atom (:class:`~numpy.ndarray` of `float` with shape (`n`, 1), unitless
    `valence_widths` is the width describing the valence electron density (:class:`~numpy.ndarray` of `float` with shape (`n`, 1), TODO: check units. a.u. ? inverse width?

    Parameters
    ----------
    file : str
        The name of a file containing the :class:`~pandas.DataFrame`

    Returns
    -------
    monomers : list of :class:`~qcelemental.models.Molecule`
    cartesian_multipoles : list of :class:`~numpy.ndarray` or None
        None is returned if the `cartesian_multipoles` column is not present in the :class:`~pandas.DataFrame`
    volume_ratios : list of :class:`~numpy.ndarray` or None
        None is returned if the `volume_ratios` column is not present in the :class:`~pandas.DataFrame`
    valence_widths : list of :class:`~numpy.ndarray` or None
        None is returned if the `valence_widths` column is not present in the :class:`~pandas.DataFrame`
    """

    df = pd.read_pickle(file)
    print(df.columns.values)
    N = len(df.index)
    print(f"Reading {file} with {N} rows")

    if max_size is not None and max_size < N:
        print("Truncating dataset to max_size:", max_size)
        df = df.head(max_size)
        N = max_size

    R = df[R_label].tolist()
    Z = df[Z_label].tolist()
    TQ = df[TQ_label].tolist()
    aQ = [TQ[i] / np.sum(Z[i] > 0) for i in range(N)]

    # cartesian_multipoles_in = df['multipoles_A'].to_numpy()
    if cartesian_multipoles_label is not None:
        cartesian_multipoles_in = df[cartesian_multipoles_label].to_numpy()
    else:
        cartesian_multipoles_in = None

    if hirshfeld_props:
        volume_ratios = df["volume ratios"].to_numpy()
        valence_widths = df["valence widths"].to_numpy()

    monomers = []
    cartesian_multipoles = []
    total_charge = []
    for i in range(N):
        # try will catch any errors in the data and skip that row
        try:
            monomers.append(monomerdata_to_qcel(R[i], Z[i], aQ[i]))
            total_charge.append(TQ[i])
            if cartesian_multipoles_in is not None:
                cartesian_multipoles.append(cartesian_multipoles_in[i])
        except Exception as e:
            pass

    if hirshfeld_props:
        return (
            monomers,
            cartesian_multipoles,
            total_charge,
            volume_ratios,
            valence_widths,
        )

    return monomers, cartesian_multipoles, total_charge


if __name__ == "__main__":
    mol = qcel.models.Molecule.from_data(
        """
    0 1
    O 0.000000 0.000000 0.100000
    H 1.000000 0.000000 0.000000
    CL 0.000000 1.000000 0.400000
    --
    0 1
    O -4.100000 0.000000 0.000000
    H -3.100000 0.000000 0.200000
    O -4.100000 1.000000 0.100000
    H -4.100000 2.000000 0.100000
    no_com
    no_reorient
    units angstrom
    """
    )
    print(mol.to_string("psi4"))
    print(mol)

    data = qcel_to_dimerdata(mol)

    mol2 = dimerdata_to_qcel(*data)
    print(mol2.to_string("psi4"))
    print(mol2)

    mol3 = qcel.models.Molecule.from_data(
        """
    -2 1
    O -4.100000 0.000000 0.000000
    H -3.100000 0.000000 0.200000
    O -4.100000 1.000000 0.100000
    H -4.100000 2.000000 0.100000
    no_com
    no_reorient
    units angstrom
    """
    )

    R, Z, aQ = qcel_to_monomerdata(mol3)
    print(mol3)
    print(R)
    print(Z)
    print(aQ)
