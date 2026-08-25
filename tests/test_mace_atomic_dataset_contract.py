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


def test_processed_file_names_is_memoized_between_element_accesses(tmp_path):
    """len() runs per __getitem__, so an uncached glob makes iteration quadratic."""
    from apnet_pt import atomic_datasets

    processed = tmp_path / "processed"
    processed.mkdir()
    for index in range(5):
        (processed / f"data_spec_1_{index}.pt").touch()

    dataset = atomic_datasets.atomic_module_dataset.__new__(
        atomic_datasets.atomic_module_dataset
    )
    dataset.root = str(tmp_path)
    dataset.spec_type = 1
    dataset.split = "all"
    dataset.testing = False
    dataset.force_reprocess = False
    dataset.MAX_SIZE = None

    calls = []
    real_glob = atomic_datasets.glob

    def counting_glob(pattern):
        calls.append(pattern)
        return real_glob(pattern)

    atomic_datasets.glob = counting_glob
    try:
        first = dataset.processed_file_names
        for _ in range(10):
            assert dataset.processed_file_names == first
        assert len(calls) == 1, f"expected one glob, got {len(calls)}"

        # A new file is only picked up after explicit invalidation, which
        # process() performs once it finishes writing.
        (processed / "data_spec_1_5.pt").touch()
        assert dataset.processed_file_names == first
        dataset._invalidate_processed_file_names()
        assert len(dataset.processed_file_names) == len(first) + 1
    finally:
        atomic_datasets.glob = real_glob


def test_processed_file_names_orders_naturally_and_handles_empty(tmp_path):
    from apnet_pt import atomic_datasets

    processed = tmp_path / "processed"
    processed.mkdir()

    dataset = atomic_datasets.atomic_module_dataset.__new__(
        atomic_datasets.atomic_module_dataset
    )
    dataset.root = str(tmp_path)
    dataset.spec_type = 1
    dataset.split = "all"
    dataset.testing = False
    dataset.force_reprocess = False
    dataset.MAX_SIZE = None

    assert dataset.processed_file_names == ["data_missing_0.pt"]

    for index in (0, 2, 10, 1):
        (processed / f"data_spec_1_{index}.pt").touch()
    dataset._invalidate_processed_file_names()
    assert dataset.processed_file_names == [
        "data_spec_1_0.pt",
        "data_spec_1_1.pt",
        "data_spec_1_2.pt",
        "data_spec_1_10.pt",
    ]
