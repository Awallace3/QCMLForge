"""torch.compile must not change results for batches holding a single molecule
or dimer.

Inductor specializes a dimension of size 1, and on torch 2.10 an atomic-add
scatter into such an output was miscompiled on CUDA: the per-molecule charge
sum came back as zero (charges not neutralized) and single-dimer APNet2
energies came back as NaN. Eager runs were unaffected.
"""

import copy

import numpy as np
import qcelemental as qcel
import torch

from apnet_pt import atomic_datasets
from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model

MONOMER = qcel.models.Molecule.from_data(
    """
0 1
C -0.046520 0.663180 0.000000
O -0.046520 -0.756820 0.000000
H -1.086520 0.973180 0.000000
H 0.433480 1.063180 0.893000
H 0.433480 1.063180 -0.893000
H 0.863480 -1.066820 0.000000
units angstrom
"""
)
DIMER = qcel.models.Molecule.from_data(
    """
0 1
O 0.000000 0.000000 0.000000
H 0.758602 0.000000 0.504284
H 0.260455 0.000000 -0.872893
--
0 1
O 3.000000 0.500000 0.000000
H 3.500000 0.500000 0.866000
H 3.500000 0.500000 -0.866000
units angstrom
"""
)


def _atom_batch(n_mol, device):
    data = atomic_datasets.qcel_mon_to_pyg_data(MONOMER, r_cut=5.0)
    return atomic_datasets.atomic_collate_update_no_target([data] * n_mol).to(device)


def test_compiled_atom_model_neutralizes_single_molecule_batch(use_gpu):
    torch.manual_seed(0)
    harness = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=use_gpu)
    # The harness only moves its model inside predict_*; do it up front.
    harness.model.to(harness.device)
    eager = copy.deepcopy(harness.model).eval()
    harness.compile_model()
    compiled = harness.model.eval()

    with torch.no_grad():
        # A two-molecule batch first, then the one-molecule remainder: the
        # order a batch_size=2 prediction over three molecules produces.
        for n_mol in (2, 1):
            batch = _atom_batch(n_mol, harness.device)
            q_eager = eager(batch)[0]
            q_compiled = compiled(batch)[0]
            np.testing.assert_allclose(
                q_compiled.cpu().numpy(), q_eager.cpu().numpy(), atol=1e-5
            )
            assert abs(q_compiled.sum().item()) < 1e-5


def test_compiled_apnet2_single_dimer_matches_eager(use_gpu):
    torch.manual_seed(0)
    atom_model = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=use_gpu)
    pair_model = APNet2Model(
        atom_model=atom_model.model, ignore_database_null=True, use_GPU=use_gpu
    )
    expected = pair_model.predict_qcel_mols([DIMER] * 3, batch_size=2)

    pair_model.compile_model()
    actual = pair_model.predict_qcel_mols([DIMER] * 3, batch_size=2)

    assert np.isfinite(actual).all(), actual
    np.testing.assert_allclose(actual, expected, atol=1e-4)
