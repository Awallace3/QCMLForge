import pytest
import apnet_pt
import os
import torch

"""
NOTE: OPENMM ENERGIES BELOW ARE NOT OPTIMIZED SINCE RAN ON CPU WHERE GPU
OPTIMIZATOIN CANNOT OCCUR!

%%%%%%%%%%% STARTING WATER U_IND CALCULATION %%%%%%%%%%%%
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
=-=-=-=-=-=-=-=-=-=-=-=-OpenMM Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
total Energy = -24.14799427986145 kJ/mol
<class 'openmm.openmm.NonbondedForce'>-26.1219482421875 kJ/mol
<class 'openmm.openmm.DrudeForce'>1.9739539623260498 kJ/mol
<class 'openmm.openmm.CMMotionRemover'>0.0 kJ/mol
Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
JAXOPT.BFGS Minimizer completed in 9.746 seconds!!
OpenMM U_ind = -24.1480 kJ/mol
Python U_ind = -24.1479 kJ/mol
0.00% Error
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
Log started at: 2025-02-14 13:57:37
%%%%%%%%%%% STARTING WATER U_IND CALCULATION %%%%%%%%%%%%
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
torchmin.BFGS Minimizer completed in 0.006 seconds!!
OpenMM U_ind = -22.3381 kJ/mol
PyTorch U_ind = -24.1479 kJ/mol
7.49% Error
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

Log started at: 2025-02-14 15:09:52
%%%%%%%%%%% STARTING ACNIT U_IND CALCULATION %%%%%%%%%%%%
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
=-=-=-=-=-=-=-=-=-=-=-=-OpenMM Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
total Energy = 9.742050170898438 kJ/mol
<class 'openmm.openmm.NonbondedForce'>9.742050170898438 kJ/mol
<class 'openmm.openmm.DrudeForce'>0.0 kJ/mol
<class 'openmm.openmm.CMMotionRemover'>0.0 kJ/mol
torchmin.BFGS Minimizer completed in 0.009 seconds!!
OpenMM U_ind = 9.7421 kJ/mol
PyTorch U_ind = 8.7294 kJ/mol
11.60% Error
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

Log started at: 2025-02-14 15:10:00
%%%%%%%%%%% STARTING IMIDAZOLE U_IND CALCULATION %%%%%%%%%%%%
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
=-=-=-=-=-=-=-=-=-=-=-=-OpenMM Output-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
total Energy = -4.921215534210205 kJ/mol
<class 'openmm.openmm.NonbondedForce'>-4.921215534210205 kJ/mol
<class 'openmm.openmm.DrudeForce'>0.0 kJ/mol
<class 'openmm.openmm.CMMotionRemover'>0.0 kJ/mol
torchmin.BFGS Minimizer completed in 0.009 seconds!!
OpenMM U_ind = -4.9212 kJ/mol
PyTorch U_ind = -19.8161 kJ/mol
75.17% Error
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
"""

data_dir = os.path.join(os.path.dirname(__file__), "drude_test_data/")

# Drude charge, (or q^shell_i) is what is assumed fixed, needs parameterization
# so MBIS charge != -q^shell_i for charged, but will for neutral

# Drude charge, polarizability, thole are fixed
# polarizability nm^3, from 2013 McDaniel first SAPT papers
# thole will largely be 1

# coupling of drude charge and polarizability, so might be issue with -D4
# Charges and polarizabilities should be from the same source, they used CAMCASP

# for acnit, stack N to N very close (2.5 A), need strong electric field to
# ensure thole model works correctly
# Could formulate in terms of interaction tensor E_elst = q_i * T_ij * q_j

# Need to get the non-additive 3body energy

# McDaniel - need to focus on crystal first before tackling generalization of
# atomic polarizabilities, cannot just use simple equations to map from one
# model to another. Energies will be bad


@pytest.mark.parametrize(
    "molecule,ref_uind",
    [
        ("water", -24.1479),
        ("acnit", 8.7294),
        ("imidazole", -19.8161),
    ],
)
@pytest.mark.skip(reason="Drude not used in codebase")
def skip_test_water_drude(
    molecule,  # name of molecule
    ref_uind,  # kJ/mol
):
    """
    Validate computed Drude oscillator induced energy for a molecule against a stored reference.
    
    Parameters:
    	molecule (str): Molecule name; used to load reference data from data_dir/<molecule>.pt.
    	ref_uind (float): Reference induced energy in kJ/mol to compare against.
    
    Raises:
    	AssertionError: If the absolute difference between the computed U_ind and ref_uind is greater than or equal to 1e-4.
    """
    ref_data = torch.load(
        data_dir + molecule + ".pt",
        weights_only=False,
    )
    for key in ref_data:
        ref_data[key] = torch.tensor(ref_data[key])
    U_ind = apnet_pt.classical_induction.compute_drude_oscillator_U_ind(
        ref_data["Dij"],
        ref_data["Rij"],
        ref_data["Qi_shell"],
        ref_data["Qj_shell"],
        ref_data["Qi_core"],
        ref_data["Qj_core"],
        ref_data["u_scale"],
        ref_data["k"],
    )
    print(U_ind)
    assert torch.abs(U_ind - ref_uind) < 1e-4


def main():
    test_water_drude("water", -24.1479)
    test_water_drude("acnit", 8.7294)
    test_water_drude("imidazole", -19.8161)
    return


if __name__ == "__main__":
    main()
