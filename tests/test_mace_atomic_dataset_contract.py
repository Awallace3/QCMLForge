import qcelemental as qcel
import torch

from apnet_pt.atomic_datasets import atomic_collate_update, qcel_mon_to_pyg_data


def test_atomic_dataset_preserves_multiplicity_for_mace():
    doublet = qcel.models.Molecule.from_data(
        "0 2\nH 0 0 0\nunits angstrom\nno_com\nno_reorient"
    )
    singlet = qcel.models.Molecule.from_data(
        "0 1\nHe 0 0 0\nunits angstrom\nno_com\nno_reorient"
    )
    records = []
    for molecule in (doublet, singlet):
        data = qcel_mon_to_pyg_data(molecule, full_indices=True)
        data.charges = torch.zeros(data.x.numel())
        data.dipoles = torch.zeros(data.x.numel(), 3)
        data.quadrupoles = torch.zeros(data.x.numel(), 3, 3)
        records.append(data)
    batch = atomic_collate_update(records)
    assert batch.total_spin.dtype.is_floating_point
    assert batch.total_spin.tolist() == [2.0, 1.0]
