import apnet_pt
import numpy as np
import qcelemental as qcel
import torch
import os
from apnet_pt.AtomPairwiseModels.apnet3_fused import APNet3_AtomType_Model
from glob import glob
import pandas as pd
import pytest
import shutil
from pprint import pprint

torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"

am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_h+1_3.pt"
at_elst_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
ap3_path = f"{current_file_path}/test_models/ap3_ensemble_0/ap3_.pt"

water_water_dimer = qcel.models.Molecule.from_data("""
0 1
--
0 1
O                    -1.326958230000    -0.105938530000     0.018788150000
H                    -1.931665240000     1.600174320000    -0.021710520000
H                     0.486644280000     0.079598090000     0.009862480000
--
0 1
O                     4.287563290000     0.049775580000     0.000960040000
H                     4.999275000000    -0.778642690000     1.448725300000
H                     4.991040900000    -0.850136520000    -1.407646550000
units bohr
no_com
no_reorient
""")

menh2_water_dimer = qcel.models.Molecule.from_data("""
0 1
--
0 1
N                    -1.008100800000    -0.528355160000     0.202192680000
H                    -1.188923790000    -2.359180490000     0.723479130000
H                    -2.121413420000    -0.313995840000    -1.337480310000
C                    -1.921680310000     1.112077560000     2.243810640000
H                    -1.724865800000     3.071847600000     1.662054110000
H                    -3.882890690000     0.784391550000     2.793966870000
H                    -0.727588730000     0.848110780000     3.893996460000
--
0 1
O                     4.023106790000     1.759569490000     0.398270440000
H                     2.478222790000     0.821750410000     0.071058880000
H                     5.122741160000     1.271079320000    -0.954293650000
units bohr
no_com
no_reorient
""")

def test_ap3_fused_qcel_molecule():
    """
    Integration test that constructs atom-type and dimer parameter models, composes an APNet3 atom-type model, and runs a prediction on a water–water dimer.
    
    Builds an AtomTypeParamModel (HF+VW) and an AM_DimerParam_Model (electrostatics) using provided pretrained paths, assembles an APNet3_AtomType_Model wired with those components, calls predict_qcel_mols on the prepared water–water dimer with batch_size=1 and return_classical_pairs=True, and prints the dimer model's AtomTypeParam and the final prediction entry.
    """
    atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )
    print(atom_type_elst_model.dimer_model.AtomTypeParam)
    ap3 = APNet3_AtomType_Model(
        ds_root=None,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        am_dimer_param_model=atom_type_elst_model,
        use_precomputed_classical=False,
    )
    v = ap3.predict_qcel_mols([water_water_dimer,], batch_size=1, return_classical_pairs=True)
    print(v[-1])

if __name__ == "__main__":
    test_ap3_fused_qcel_molecule()