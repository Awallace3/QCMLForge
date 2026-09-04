"""Regression tests for atoms that carry no intramonomer edge.

A monatomic monomer (Na+, Cl-, a bare noble gas) produces an atom with no
entry in ``edge_index``. The atom models used to drop those atoms from message
passing and stitch them back in afterwards, which made a prediction depend on
which other monomers happened to share the batch. ``AtomMPNN`` no longer
filters, so anything reading its ``h_list`` must index the full atom list.
"""

import os

import numpy as np
import pytest
import qcelemental as qcel
import torch

import apnet_pt

file_dir = os.path.dirname(os.path.abspath(__file__))
am_path = f"{file_dir}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{file_dir}/test_models/ap3_ensemble_0/am_h+1_3.pt"

# float32 message passing reorders sums when the batch changes, so predictions
# agree to single precision rather than bit-exactly.
ATOL = 1e-5


def _mols():
    m = qcel.models.Molecule.from_data
    return (
        m("1 1\nNa 0.0 0.0 0.0"),
        m("-1 1\nCl 0.0 0.0 0.0"),
        m("0 1\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24"),
    )


@pytest.fixture(scope="module")
def atom_type_param_model():
    torch.manual_seed(42)
    return apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )


def test_atom_type_param_edgeless_atom_is_batch_invariant(atom_type_param_model):
    """K for a monatomic monomer must not depend on its batch neighbours."""
    na, cl, wat = _mols()
    model = atom_type_param_model

    alone = model.predict_qcel_mols([na], batch_size=1)
    all_edgeless = model.predict_qcel_mols([na, cl], batch_size=2)
    mixed = model.predict_qcel_mols([na, wat], batch_size=2)

    # entries 3 and 4 are the two predicted params (hirshfeld volume ratio,
    # valence width)
    for param_ind in (3, 4):
        ref = np.asarray(alone[0][param_ind])
        assert ref.shape == (1,)
        for tag, batched in (("all-edgeless", all_edgeless), ("mixed", mixed)):
            got = np.asarray(batched[0][param_ind])
            assert got.shape == ref.shape, tag
            assert np.allclose(got, ref, atol=ATOL), (
                f"{tag}: param {param_ind} drifted by "
                f"{np.abs(got - ref).max():.3e}"
            )


def test_atom_type_param_edge_bearing_atoms_unaffected(atom_type_param_model):
    """Adding an edgeless monomer must not perturb a normal monomer."""
    na, _, wat = _mols()
    model = atom_type_param_model

    wat_alone = model.predict_qcel_mols([wat], batch_size=1)
    mixed = model.predict_qcel_mols([na, wat], batch_size=2)

    for param_ind in (3, 4):
        ref = np.asarray(wat_alone[0][param_ind])
        got = np.asarray(mixed[1][param_ind])
        assert got.shape == ref.shape == (3,)
        assert np.allclose(got, ref, atol=ATOL), (
            f"param {param_ind} drifted by {np.abs(got - ref).max():.3e}"
        )


def test_atom_type_param_all_edgeless_batch_shapes(atom_type_param_model):
    """An all-edgeless batch must return the same shapes as any other batch.

    This path used to short-circuit and return ``charge.squeeze(-1)`` while the
    normal path returned ``charge``, so a batch of monatomic monomers came back
    with a different rank than a batch containing a polyatomic one.
    """
    na, cl, wat = _mols()
    model = atom_type_param_model

    all_edgeless = model.predict_qcel_mols([na, cl], batch_size=2)
    mixed = model.predict_qcel_mols([na, wat], batch_size=2)

    for out, natoms in ((all_edgeless[0], 1), (all_edgeless[1], 1), (mixed[0], 1)):
        charge, dipole, qpole = (np.asarray(out[i]) for i in range(3))
        assert charge.shape == (natoms,)
        assert dipole.shape == (natoms, 3)
        assert qpole.shape == (natoms, 3, 3)
