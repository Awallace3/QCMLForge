import pytest
import apnet_pt
import qcelemental as qcel
import os
import pandas as pd
from pprint import pprint
import numpy as np
import torch

# AMOEBA - water dimer
"""
 Enter the Desired Analysis Types [G,P,E,A,L,D,M,V,C] :  D

 Individual Bond Stretching Interactions :

 Type              Atom Names                      Ideal    Actual      Energy

 Bond            1-O        2-H                   0.9572    0.9572      0.0000
 Bond            4-O        5-H                   0.9572    0.9572      0.0000
 Bond            1-O        3-H                   0.9572    0.9572      0.0000
 Bond            4-O        6-H                   0.9572    0.9572      0.0000

 Individual Angle Bending Interactions :

 Type                  Atom Names                  Ideal    Actual      Energy

 Angle           2-H        1-O        3-H      108.5000  104.5200      0.2483
 Angle           5-H        4-O        6-H      108.5000  104.5200      0.2483

 Individual Urey-Bradley Interactions :

 Type                  Atom Names                  Ideal    Actual      Energy

 UreyBrad        2-H        1-O        3-H        1.5537    1.5139     -0.0120
 UreyBrad        5-H        4-O        6-H        1.5537    1.5139     -0.0120

 Individual Atomic Multipole Interactions :

 Type              Atom Names               Distance        Energy

 Mpole           1-O        4-O               4.0000       23.1068
 Mpole           1-O        5-H               4.1129      -10.8337
 Mpole           1-O        6-H               4.1129      -10.8337
 Mpole           2-H        4-O               4.1129      -10.8337
 Mpole           2-H        5-H               4.0000        5.5326
 Mpole           2-H        6-H               4.2769        5.0672
 Mpole           3-H        4-O               4.1129      -10.8337
 Mpole           3-H        5-H               4.2769        5.0672
 Mpole           3-H        6-H               4.0000        5.5326

 Determination of SCF Induced Dipole Moments :

    Iter       RMS Residual (Debye)

       1           0.0008888895
       2           0.0000550535
       3           0.0000046868
       4           0.0000000003

 Induced Dipoles :    Iterations    4       RMS Residual   0.0000000003

 Induced Dipole Moments (Debye) :

    Atom               X            Y            Z           Total

       1            0.0000      -0.0169       0.0199        0.0261
       2            0.0040      -0.0092      -0.0003        0.0100
       3           -0.0040      -0.0092      -0.0003        0.0100
       4            0.0000      -0.0169      -0.0199        0.0261
       5            0.0040      -0.0092       0.0003        0.0100
       6           -0.0040      -0.0092       0.0003        0.0100

 Individual Dipole Polarization Interactions :

 Type              Atom Names               Distance        Energy

 Polar           1-O        4-O               4.0000       -0.0480
 Polar           1-O        5-H               4.1129        0.0064
 Polar           1-O        6-H               4.1129        0.0064
 Polar           2-H        4-O               4.1129        0.0064
 Polar           2-H        5-H               4.0000       -0.0005
 Polar           2-H        6-H               4.2769        0.0012
 Polar           3-H        4-O               4.1129        0.0064
 Polar           3-H        5-H               4.2769        0.0012
 Polar           3-H        6-H               4.0000       -0.0005

 Individual van der Waals Interactions :

 Type              Atom Names                    Minimum    Actual      Energy

 VDW-Hal         1-O        4-O                   3.4050    4.0000     -0.0630
 VDW-Hal         1-O        5-H                   3.1214    4.0937     -0.0091
 VDW-Hal         1-O        6-H                   3.1214    4.0937     -0.0091
 VDW-Hal         2-H        4-O                   3.1214    4.0937     -0.0091
 VDW-Hal         2-H        5-H                   2.6550    4.0000     -0.0017
 VDW-Hal         2-H        6-H                   2.6550    4.2306     -0.0012
 VDW-Hal         3-H        4-O                   3.1214    4.0937     -0.0091
 VDW-Hal         3-H        5-H                   2.6550    4.2306     -0.0012
 VDW-Hal         3-H        6-H                   2.6550    4.0000     -0.0017

 Intermolecular Energy :                   0.8454 Kcal/mole

 Total Potential Energy :                  1.3179 Kcal/mole

 Energy Component Breakdown :           Kcal/mole        Interactions

 Bond Stretching                           0.0000                4
 Angle Bending                             0.4966                2
 Urey-Bradley                             -0.0241                2
 Van der Waals                            -0.1052                9
 Atomic Multipoles                         0.9718                9
 Polarization                             -0.0212                9
"""

# AMOEBA - water Monomer
"""
 Atom Definition Parameters :

   Atom  Symbol  Type  Class  Atomic   Mass  Valence  Description

     1     O       39     37     8    15.999    2     Water O                 
     2     H       40     38     1     1.008    1     Water H                 
     3     H       40     38     1     1.008    1     Water H                 

 Bond Stretching Parameters :

          Atom Numbers                         KS       Bond

     1        1     2                      556.850    0.9572
     2        1     3                      556.850    0.9572

 Angle Bending Parameters :

             Atom Numbers                      KB      Angle   Fold    Type

     1        2     1     3                 48.700   108.500

 Urey-Bradley Parameters :

             Atom Numbers                     KUB    Distance

     1        2     1     3                 -7.600    1.5537

 Van der Waals Parameters :

          Atom Number       Size   Epsilon   Size 1-4   Eps 1-4   Reduction

     1        1           3.4050    0.1100
     2        2           2.6550    0.0135                          0.9100
     3        3           2.6550    0.0135                          0.9100

 Atomic Multipole Parameters :

           Atom   Z-Axis X-Axis Y-Axis  Frame           Multipole Moments

     1        1       2      3          Bisector   -0.51966
                                                    0.00000  0.00000  0.14279
                                                    0.37928
                                                    0.00000 -0.41809
                                                    0.00000  0.00000  0.03881
     2        2       1      3          Z-then-X    0.25983
                                                   -0.03859  0.00000 -0.05818
                                                   -0.03673
                                                    0.00000 -0.10739
                                                   -0.00203  0.00000  0.14412
     3        3       1      2          Z-then-X    0.25983
                                                   -0.03859  0.00000 -0.05818
                                                   -0.03673
                                                    0.00000 -0.10739
                                                   -0.00203  0.00000  0.14412

 Dipole Polarizability Parameters :

          Atom Number     Alpha    Thole      Polarization Group

     1        1          0.8370    0.390        1     2     3
     2        2          0.4960    0.390        1     2     3
     3        3          0.4960    0.390        1     2     3

~/proj/tinker_tests > ls                                  took 6s py p4ein at 12:00:50
water.key  water.xyz  w.xyz

~/proj/tinker_tests > cat water.xyz                               py p4ein at 12:01:21
     3  Water
     1  O      0.000000    0.000000    0.000000     39     2     3
     2  H     -0.756950    0.585882    0.000000     40     1
     3  H      0.756950    0.585882    0.000000     40     1
"""

# Psi4 MBIS water monomer
"""
  MBIS Charges: (a.u.)
   Center  Symbol  Z      Pop.       Charge
      1       O    8    8.909280   -0.909280
      2       H    1    0.545360    0.454640
      3       H    1    0.545360    0.454640

# We must be using different sign convention for dipoles because AMOEBA dipoles
# are in +Y direction for O but ours are in -Y direction.
  MBIS Dipoles: [e a0]
   Center  Symbol  Z        X           Y           Z
      1       O    8    0.000000   -0.224397    0.000000
      2       H    1   -0.029279    0.001991    0.000000
      3       H    1    0.029279    0.001991   -0.000000

  MBIS Quadrupoles: [e a0^2]
   Center  Symbol  Z      XX        XY        XZ        YY        YZ        ZZ
      1       O    8   -4.6915   -0.0000   -0.0000   -4.9035   -0.0000   -5.0911
      2       H    1   -0.2588    0.0015    0.0000   -0.2645    0.0000   -0.2757
      3       H    1   -0.2588   -0.0015   -0.0000   -0.2645   -0.0000   -0.2757

  MBIS Octupoles: [e a0^3]
   Center  Symbol  Z      XXX       XXY       XXZ       XYY       XYZ       XZZ       YYY       YYZ       YZZ       ZZZ
      1       O    8   -0.0000   -0.4068   -0.0000    0.0000   -0.0000    0.0000   -0.7171   -0.0000   -0.2024   -0.0000
      2       H    1   -0.0294   -0.0094   -0.0000   -0.0107   -0.0000   -0.0077   -0.0338   -0.0000   -0.0104   -0.0000
      3       H    1    0.0294   -0.0094   -0.0000    0.0107   -0.0000    0.0077   -0.0338   -0.0000   -0.0104   -0.0000

  MBIS Radial Moments:
   Center  Symbol  Z      [a0^2]      [a0^3]      [a0^4]      
      1       O    8   14.686158   30.022731   75.576268
      2       H    1    0.798995    1.407824    3.022537
      3       H    1    0.798995    1.407824    3.022537

  MBIS Valence Widths: [a0]
   Center  Symbol  Z     Width
      1       O    8    0.411193
      2       H    1    0.349530
      3       H    1    0.349530


  MBIS Valence Charges: (a.u.)
   Center  Symbol  Z     Charge
      1       O    8   -7.273167
      2       H    1   -0.545360
      3       H    1   -0.545360


  MBIS Volume Ratios: 
   Center  Symbol  Z     
      1       O    8    1.392843
      2       H    1    0.185752
      3       H    1    0.185752
"""


def build_local_frame_rotation_matrix(
    mol_coords, atom_idx, z_axis_atom, x_axis_atom, frame_type
):
    """
    Build rotation matrix from local AMOEBA frame to global Cartesian XYZ frame.

    Parameters
    ----------
    mol_coords : np.ndarray
        Molecular coordinates, shape (n_atoms, 3)
    atom_idx : int
        Index of atom for which we're building the frame
    z_axis_atom : int
        Index of atom defining the local z-axis
    x_axis_atom : int
        Index of atom helping define the local x-axis
    frame_type : str
        Either "Bisector" or "Z-then-X"

    Returns
    -------
    R : np.ndarray
        Rotation matrix (3x3) that transforms from local frame to global frame.
        To transform a vector v_local to v_global: v_global = R @ v_local
    """
    pos = mol_coords[atom_idx]
    pos_z = mol_coords[z_axis_atom]
    pos_x = mol_coords[x_axis_atom]

    if frame_type == "Bisector":
        # Local z-axis bisects the angle between z_axis_atom and x_axis_atom
        vec_z = pos_z - pos  # vector to z_axis_atom
        vec_x = pos_x - pos  # vector to x_axis_atom

        # Normalize both vectors
        vec_z = vec_z / np.linalg.norm(vec_z)
        vec_x = vec_x / np.linalg.norm(vec_x)

        # Local z-axis is the bisector (average of the two unit vectors)
        z_local = vec_z + vec_x
        z_local = z_local / np.linalg.norm(z_local)

        # Local x-axis is perpendicular to z in the plane containing both bonds
        # First get the normal to the plane
        normal = np.cross(vec_z, vec_x)

        # Local y-axis is perpendicular to both z and the plane normal
        y_local = np.cross(z_local, normal)
        y_local = y_local / np.linalg.norm(y_local)

        # Local x-axis completes the right-handed system
        x_local = np.cross(y_local, z_local)
        x_local = x_local / np.linalg.norm(x_local)

    elif frame_type == "Z-then-X":
        # Local z-axis points from atom to z_axis_atom
        z_local = pos_z - pos
        z_local = z_local / np.linalg.norm(z_local)

        # Vector toward x_axis_atom helps define x-axis
        vec_to_x = pos_x - pos

        # Local y-axis is perpendicular to z and the vector to x_axis_atom
        y_local = np.cross(z_local, vec_to_x)
        y_local = y_local / np.linalg.norm(y_local)

        # Local x-axis completes the right-handed system
        x_local = np.cross(y_local, z_local)
        x_local = x_local / np.linalg.norm(x_local)
    else:
        raise ValueError(f"Unknown frame type: {frame_type}")

    # Rotation matrix has local axes as columns
    # R @ v_local = v_global
    R = np.column_stack([x_local, y_local, z_local])

    return R


def transform_multipoles_to_cartesian(mol_coords, atom_frames, multipoles_local):
    """
    Transform AMOEBA multipole moments from local frames to Cartesian XYZ.

    Parameters
    ----------
    mol_coords : np.ndarray
        Molecular coordinates, shape (n_atoms, 3)
    atom_frames : list of dict
        Each dict contains: {'z_axis': int, 'x_axis': int, 'frame_type': str}
    multipoles_local : list of dict
        Each dict contains: {'q': float, 'mu': array(3), 'theta': array(6)}
        where theta is in lower-triangular format [XX, XY, YY, XZ, YZ, ZZ]
        in the LOCAL frame

    Returns
    -------
    q_global : np.ndarray
        Charges (unchanged)
    mu_global : np.ndarray
        Dipole moments in Cartesian XYZ, shape (n_atoms, 3)
    theta_global : np.ndarray
        Quadrupole tensors in Cartesian XYZ, shape (n_atoms, 3, 3)
    """
    n_atoms = len(multipoles_local)
    q_global = np.zeros(n_atoms)
    mu_global = np.zeros((n_atoms, 3))
    theta_global = np.zeros((n_atoms, 3, 3))

    for i in range(n_atoms):
        # Get rotation matrix
        R = build_local_frame_rotation_matrix(
            mol_coords,
            i,
            atom_frames[i]["z_axis"],
            atom_frames[i]["x_axis"],
            atom_frames[i]["frame_type"],
        )

        # Charge is invariant
        q_global[i] = multipoles_local[i]["q"]

        # Dipole transforms as a vector: mu_global = R @ mu_local
        mu_local = multipoles_local[i]["mu"]  # shape (3,) in local frame
        mu_global[i] = R @ mu_local

        # Quadrupole transforms as rank-2 tensor: Q_global = R @ Q_local @ R^T
        # [XX, XY, YY, XZ, YZ, ZZ] in LOCAL frame
        theta_lt = multipoles_local[i]["theta"]
        theta_local = np.array(
            [
                [theta_lt[0], theta_lt[1], theta_lt[3]],  # XX, XY, XZ
                [theta_lt[1], theta_lt[2], theta_lt[4]],  # XY, YY, YZ
                [theta_lt[3], theta_lt[4], theta_lt[5]],  # XZ, YZ, ZZ
            ]
        )
        theta_global[i] = R @ theta_local @ R.T

    return q_global, mu_global, theta_global


def test_intramolecular_induced_dipole():
    molA = qcel.models.Molecule.from_data("""
0 1
O      0.000000    0.000000    0.000000
H     -0.756950    0.585882    0.000000
H      0.756950    0.585882    0.000000
units angstrom
    """)
    print(molA.to_string("psi4"))

    qA = np.array([-0.51966, 0.25983, 0.25983])
    # Need to switch ZXY to XYZ and multiply dipoles by -1 to match sign
    # convention
    muA = 1.0 * np.array(
        [
            [0.0, 0.14279, 0.0],
            [-0.06962867, 0.00509394, 0.0],
            [0.06962867, 0.00509394, 0.0],
        ]
    )
    thetaA = [
        # They have order of ZXY...
        [0.37928, 0.0, -0.41809, 0.0, 0.0, 0.03881],
        [-0.03673, 0.0, -0.10739, -0.00203, 0.0, 0.14412],
        [-0.03673, 0.0, -0.10739, -2.03e-03, 0.0, 0.14412],
    ]
    # expand thetaA to full 3x3 tensors from the lower-triangular representation
    # Lower-triangular format: [XX, XY, YY, XZ, YZ, ZZ]
    thetaA_full = []
    for theta_lt in thetaA:
        theta_3x3 = np.array(
            [
                [theta_lt[3], theta_lt[1], theta_lt[0]],  # XX, XY, XZ
                [theta_lt[4], theta_lt[2], theta_lt[1]],  # XY, YY, YZ
                [theta_lt[5], theta_lt[4], theta_lt[3]],  # XZ, YZ, ZZ
            ]
        )
        thetaA_full.append(theta_3x3)
    thetaA = np.array(thetaA_full)

    # vrA = r["vol_ratios_A pbe0/atz"]
    atomic_polarizabilities = np.array([0.8370, 0.4960, 0.4960])  # in angstrom^3
    # convert from angstrom^3 to atomic units (1 A^3 = 0.148184711 au)
    # ang2bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    # atomic_polarizabilities *= (ang2bohr ** 3)

    q_returned, mu_induced, theta_returned = (
        apnet_pt.multipole.intramolecular_induced_dipole(
            qcel_mol=molA,
            q=qA,
            mu=muA,
            theta=thetaA,
            atom_polarizabilities=atomic_polarizabilities,
            # hirshfeld_volume_ratio=vrA,
            # valence_widths=vwA,
            # AMOEBA+ uses a=0.75 for direct, a=39 for mutual
            # https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.7b00225?ref=article_openPDF
            thole_damping_param_direct=0.34,
            thole_damping_param_mutual=0.39,
            zero_quadrupoles=True,
        )
    )
    print(f"charges: {q_returned}")
    print(f"perm    dipoles:\n{muA}")
    print(f"Induced dipoles:\n{mu_induced}")
    return


def test_intramolecular_induced_dipole_MPID():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_pickle(
        current_file_path + os.sep + os.path.join("dataset_data", "water_dimer_pes3.pkl")
    )
    df = df[df["system_id"].str.contains("01_Water-Water")].copy()
    df = df.sort_values(by="system_id")
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    
    molA = mol.get_fragment(0)
    print(molA.to_string("psi4"))
    
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    vrA = r["vol_ratios_A pbe0/atz"]
    vwA = r["val_widths_A pbe0/atz"]
    np.set_printoptions(precision=6, suppress=True)
    
    q_returned, mu_induced, theta_returned = apnet_pt.multipole.intramolecular_induced_dipole(
        qcel_mol=molA,
        q=qA,
        mu=muA,
        theta=thetaA,
        hirshfeld_volume_ratio=vrA,
        valence_widths=vwA,
        thole_damping_param_mutual=0.39,
        thole_damping_param_direct=0.34,
    )
    mu_diff = mu_induced - muA.reshape(-1, 3)

    print(f"charges: {q_returned}")
    print(f"Quadrupoles:\n{theta_returned}")
    print(f"Original   dipoles:\n{muA}")
    print(f"Induced    dipoles:\n{mu_induced}")
    # get magnitudes of dipoles
    muA_magnitudes = np.linalg.norm(muA.reshape(-1, 3), axis=1)
    mu_induced_magnitudes = np.linalg.norm(mu_induced, axis=1)
    mu_diff_magnitudes = np.linalg.norm(mu_diff, axis=1)
    print(f"Original   dipole magnitudes: {muA_magnitudes}")
    print(f"Induced    dipole magnitudes: {mu_induced_magnitudes}")
    print(f"Difference dipole magnitudes: {mu_diff_magnitudes}")
    return


def test_intramolecular_induced_dipole_MPID_df_bz():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_pickle(
        current_file_path + os.sep + os.path.join("dataset_data", "df_bz_meoh_mbis.pkl")
    )
    print(df)
    df = df.sort_values(by="system_id")
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    
    molA = mol.get_fragment(0)
    print(molA.to_string("psi4"))
    
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    vrA = r["vol_ratios_A pbe0/atz"]
    vwA = r["val_widths_A pbe0/atz"]
    np.set_printoptions(precision=6, suppress=True)

    q_returned, mu_induced, theta_returned = apnet_pt.multipole.intramolecular_induced_dipole(
        qcel_mol=molA,
        q=qA,
        mu=muA,
        theta=thetaA,
        hirshfeld_volume_ratio=vrA,
        valence_widths=vwA,
        thole_damping_param_mutual=0.39,
        thole_damping_param_direct=0.34,
        compute_energies=True,
        heavy_atoms_only=False,
        verbose=True
    )
    
    q_returned, mu_induced, theta_returned = apnet_pt.multipole.intramolecular_induced_dipole(
        qcel_mol=molA,
        q=qA,
        mu=muA,
        theta=thetaA,
        hirshfeld_volume_ratio=vrA,
        valence_widths=vwA,
        thole_damping_param_mutual=0.39,
        thole_damping_param_direct=0.34,
        compute_energies=True,
        verbose=True
    )
    return


def test_intramolecular_induced_dipole_MPID_df_bz_torch():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_pickle(
        current_file_path + os.sep + os.path.join("dataset_data", "df_bz_meoh_mbis.pkl")
    )
    print(df)
    df = df.sort_values(by="system_id")
    r = df.iloc[0]
    mol = r["qcel_molecule"]
    
    molA = mol.get_fragment(0)
    print(molA.to_string("psi4"))
    
    qA = r["q_A pbe0/atz"]
    muA = r["mu_A pbe0/atz"]
    thetaA = r["theta_A pbe0/atz"]
    vrA = r["vol_ratios_A pbe0/atz"]
    vwA = r["val_widths_A pbe0/atz"]
    np.set_printoptions(precision=6, suppress=True)

    q_returned, mu_induced, theta_returned = apnet_pt.multipole.intramolecular_induced_dipole(
        qcel_mol=molA,
        q=qA,
        mu=muA,
        theta=thetaA,
        hirshfeld_volume_ratio=vrA,
        valence_widths=vwA,
        thole_damping_param_mutual=0.39,
        thole_damping_param_direct=0.34,
        compute_energies=True,
        heavy_atoms_only=False,
        verbose=True
    )
    
    q_returned, mu_induced, theta_returned = apnet_pt.multipole.intramolecular_induced_dipole(
        qcel_mol=molA,
        q=qA,
        mu=muA,
        theta=thetaA,
        hirshfeld_volume_ratio=vrA,
        valence_widths=vwA,
        thole_damping_param_mutual=0.39,
        thole_damping_param_direct=0.34,
        compute_energies=True,
        verbose=True
    )
    return


def amoeba_transform():
    # Example: Transform AMOEBA water multipoles to Cartesian XYZ
    # Water geometry
    coords = np.array(
        [
            [0.000000, 0.000000, 0.000000],  # O
            [-0.756950, 0.585882, 0.000000],  # H
            [0.756950, 0.585882, 0.000000],  # H
        ]
    )

    # Define local frames from AMOEBA parameters (lines 152-170)
    # Atom indices are 0-based (subtract 1 from AMOEBA 1-based indices)
    frames = [
        # O: bisector of H-O-H
        {"z_axis": 1, "x_axis": 2, "frame_type": "Bisector"},
        # H1: z toward O, x using H2
        {"z_axis": 0, "x_axis": 2, "frame_type": "Z-then-X"},
        # H2: z toward O, x using H1
        {"z_axis": 0, "x_axis": 1, "frame_type": "Z-then-X"},
    ]

    # AMOEBA multipoles in LOCAL frames
    multipoles_local = [
        {  # Atom 1 (O)
            "q": -0.51966,
            # in local frame (x, y, z)
            "mu": np.array([0.00000, 0.00000, 0.14279]),
            # XX, XY, YY, XZ, YZ, ZZ
            "theta": np.array([0.37928, 0.0, -0.41809, 0.0, 0.0, 0.03881]),
        },
        {  # Atom 2 (H)
            "q": 0.25983,
            "mu": np.array([-0.03859, 0.00000, -0.05818]),
            "theta": np.array([-0.03673, 0.0, -0.10739, -0.00203, 0.0, 0.14412]),
        },
        {  # Atom 3 (H)
            "q": 0.25983,
            "mu": np.array([-0.03859, 0.00000, -0.05818]),
            "theta": np.array([-0.03673, 0.0, -0.10739, -0.00203, 0.0, 0.14412]),
        },
    ]

    # Transform to Cartesian XYZ
    q_cart, mu_cart, theta_cart = transform_multipoles_to_cartesian(
        coords, frames, multipoles_local
    )

    print("=" * 60)
    print("AMOEBA Multipoles transformed to Cartesian XYZ:")
    print("=" * 60)
    print("\nCharges:")
    print(q_cart)
    print("\nDipole moments (Cartesian XYZ):")
    print(mu_cart)
    print("\nQuadrupole tensors (Cartesian XYZ):")
    for i, theta in enumerate(theta_cart):
        print(f"\nAtom {i}:")
        print(theta)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Now run the actual test
    # test_intramolecular_induced_dipole()
    # test_intramolecular_induced_dipole_MPID()
    test_intramolecular_induced_dipole_MPID_df_bz()
