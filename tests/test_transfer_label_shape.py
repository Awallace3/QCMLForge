"""The transfer-learning loss must compare each dimer with its own label.

``batch.y`` is ``(n_dimer,)`` or ``(n_dimer, 1)`` depending on how labels were
given.  Against ``(n_dimer,)`` predictions the second used to broadcast to an
``n_dimer x n_dimer`` grid inside ``torch.nn.MSELoss`` without an error.
"""

import numpy as np
import pytest
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model
from apnet_pt.pairwise_datasets import apnet2_module_dataset

from .mols import mol_dimer

N_DIMERS = 8


def _dataset(root, labels):
    torch.manual_seed(0)
    return apnet2_module_dataset(
        root=str(root),
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atom_model=AtomModel(ds_root=None, ignore_database_null=True, use_GPU=False),
        atomic_batch_size=4,
        datapoint_storage_n_objects=4,
        batch_size=4,
        prebatched=False,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=0,
        qcel_molecules=[mol_dimer] * N_DIMERS,
        energy_labels=labels,
        in_memory=True,
        random_seed=None,
    )


def _transfer_weights(tmp_path, name, labels):
    root = tmp_path / name
    (root / "raw").mkdir(parents=True)
    torch.manual_seed(1)
    harness = APNet2Model(use_GPU=False)
    harness.train(
        _dataset(root, labels),
        n_epochs=2,
        lr=5e-3,
        skip_compile=True,
        transfer_learning=True,
        dataloader_num_workers=0,
        random_seed=7,
        wandb_config=None,
    )
    return harness.model.state_dict()


def test_column_labels_train_exactly_like_flat_labels(tmp_path):
    values = [0.1 * i for i in range(N_DIMERS)]
    flat = _transfer_weights(tmp_path, "flat", [np.array([v]) for v in values])
    column = _transfer_weights(tmp_path, "column", [np.array([[v]]) for v in values])
    assert flat.keys() == column.keys()
    for key in flat:
        assert torch.equal(flat[key], column[key]), key


def test_a_label_count_mismatch_raises(tmp_path):
    labels = [np.array([0.1, 0.2]) for _ in range(N_DIMERS)]
    with pytest.raises(RuntimeError, match="shape"):
        _transfer_weights(tmp_path, "pairs", labels)
