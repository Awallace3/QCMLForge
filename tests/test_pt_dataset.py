from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model
import apnet_pt
from apnet_pt.AtomPairwiseModels.dapnet2 import dAPNet2Model, APNet2_dAPNet2Model
from apnet_pt import AtomPairwiseModels
from apnet_pt import atomic_datasets
from apnet_pt import pairwise_datasets
from apnet_pt import AtomModels
from apnet_pt.pairwise_datasets import (
    apnet2_module_dataset,
    apnet2_collate_update,
    apnet2_collate_update_prebatched,
    APNet2_DataLoader,
    apnet3_module_dataset,
    apnet3_collate_update,
    apnet3_collate_update_prebatched,
)
from apnet_pt.pt_datasets.dapnet_ds import (
    dapnet2_module_dataset,
    dapnet2_module_dataset_apnetStored,
    dapnet2_collate_update_no_target,
)
import os
import shutil
import numpy as np
import pytest
from glob import glob
from pathlib import Path
import qcelemental as qcel
import torch
import pandas as pd
from pprint import pprint as pp
from apnet_pt.pt_datasets.ap2_fused_ds import ap2_fused_module_dataset_lmdb
from .mols import (
    # mol_mon,
    mol_dimer,
    # mol_dimer2,
    # mol_A,
    mol_dimer_ion,
    mol_cliff_water_close,
    # mol3,
    # mol_fsapt,
)


torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../src/apnet_pt/models/am_ensemble/am_0.pt"

am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
at_hf_vw_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_h+1_3.pt"
at_elst_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_elst_h+1_3.pt"
ap3_path = f"{current_file_path}/test_models/ap3_ensemble_0/ap3_.pt"
am_hf_path = f"{current_file_path}/test_models/am_hf_0.pt"


def test_apnet2_dataset_size_no_prebatched():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = False
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        # split="test",
        print_level=2,
    )
    print()
    print(ds)

    train_loader = APNet2_DataLoader(
        dataset=ds,
        # batch_size=1,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        # collate_fn=apnet2_collate_update_prebatched,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        cnt += i.y.shape[0]
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_8*.pt"):
        os.remove(i)
    assert ds_labels == cnt, f"Expected {len(ds)} points, but got {cnt} points"


def test_apnet2_dataset_size_prebatched():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = True
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        random_seed=None,
    )
    print()
    print(ds)
    print(ds.training_batch_size)

    train_loader = APNet2_DataLoader(
        dataset=ds,
        # batch_size=1,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        # collate_fn=apnet2_collate_update_prebatched,
        collate_fn=collate,
    )
    cnt = 0
    df = pd.read_pickle(f"{data_path}/raw/t_val_19.pkl")
    pp(df.columns.tolist())
    for i in train_loader:
        if cnt == 0:
            pp(i)
        inc = i.y.shape[0]
        r_RA, r_RB, r_ZA, r_ZB = [], [], [], []
        r_TQA, r_TQB = [], []
        r_labels = []
        for j in range(inc):
            r = df.iloc[cnt + j]
            r_RA.append(r["RA"])
            r_RB.append(r["RB"])
            r_ZA.append(r["ZA"])
            r_ZB.append(r["ZB"])
            y_data = np.array(
                [
                    r["Elst_aug"],
                    r["Exch_aug"],
                    r["Ind_aug"],
                    r["Disp_aug"],
                ]
            )
            r_labels.append(y_data)
            r_TQA.append(r["TQA"])
            r_TQB.append(r["TQB"])
        r_RA = np.concatenate(r_RA, axis=0)
        r_RB = np.concatenate(r_RB, axis=0)
        r_ZA = np.concatenate(r_ZA, axis=0)
        r_ZB = np.concatenate(r_ZB, axis=0)
        r_labels = np.array(r_labels)
        r_TQA = np.array(r_TQA)
        r_TQB = np.array(r_TQB)
        print(r_labels)
        print(i.y.numpy())
        assert np.allclose(i.RA.numpy(), r_RA, atol=1e-6), (
            f"Expected {i.RA.numpy()} but got {r.RA}"
        )
        assert np.allclose(i.RB.numpy(), r_RB, atol=1e-6), (
            f"Expected {i.RB.numpy()} but got {r.RB}"
        )
        assert np.allclose(i.ZA.numpy(), r_ZA, atol=1e-6), (
            f"Expected {i.ZA.numpy()} but got {r.ZA}"
        )
        assert np.allclose(i.ZB.numpy(), r_ZB, atol=1e-6), (
            f"Expected {i.ZB.numpy()} but got {r.ZB}"
        )
        assert np.allclose(i.y.numpy(), r_labels, atol=1e-6), (
            f"Expected {i.y.numpy()} but got {r_labels}"
        )
        assert np.allclose(i.total_charge_A.numpy(), r_TQA, atol=1e-6), (
            f"Expected {i.total_charge_A.numpy()} but got {r_TQA}"
        )
        assert np.allclose(i.total_charge_B.numpy(), r_TQB, atol=1e-6), (
            f"Expected {i.total_charge_B.numpy()} but got {r_TQB}"
        )
        cnt += inc
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_8*.pt"):
        os.remove(i)
    assert ds_labels * ds.batch_size == cnt, (
        f"Expected {len(ds) * ds.batch_size} points, but got {cnt} points"
    )


def test_apnet2_dataset_size_prebatched_qcel_molecules():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 6
    prebatched = True
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    qcel_molecules = [mol_dimer] * 32
    energy_labels = [np.array([1.0, 1.0, 1.0, 1.0]) for _ in range(len(qcel_molecules))]
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        random_seed=None,
    )
    print()
    print(ds)
    print(ds.training_batch_size)

    train_loader = APNet2_DataLoader(
        dataset=ds,
        # batch_size=1,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        # collate_fn=apnet2_collate_update_prebatched,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        cnt += i.y.shape[0]
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_None*.pt"):
        os.remove(i)
    assert ds_labels * ds.batch_size == cnt, (
        f"Expected {len(ds) * ds.batch_size} points, but got {cnt} points"
    )


def test_apnet2_dataset_size_qcel_molecules_in_memory():
    batch_size = 2
    atomic_batch_size = 8
    datapoint_storage_n_objects = 4
    prebatched = False
    number_dimers = 22
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    qcel_molecules = [mol_dimer] * number_dimers
    # energy_labels = [np.array([1.0, 1.0, 1.0, 1.0]) for _ in range(len(qcel_molecules))]
    energy_labels = [np.array([1.0]) for _ in range(len(qcel_molecules))]
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
        random_seed=None,
    )
    print(ds.training_batch_size)
    print(ds)
    train_loader = APNet2_DataLoader(
        dataset=ds,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        # collate_fn=apnet2_collate_update_prebatched,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        cnt += i.y.shape[0]
        # print(i)
        # print(i.y.shape)
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_None*.pt"):
        os.remove(i)
    assert number_dimers == cnt, (
        f"Expected {number_dimers} points, but got {cnt} points"
    )


def test_apnet2_dataset_size_prebatched_qcel_molecules_in_memory():
    batch_size = 4
    atomic_batch_size = 4
    datapoint_storage_n_objects = 4
    prebatched = True
    number_dimers = 31
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    qcel_molecules = [mol_dimer] * number_dimers
    energy_labels = [np.array([1.0, 1.0, 1.0, 1.0]) for _ in range(len(qcel_molecules))]
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
        random_seed=None,
    )
    print(ds)
    train_loader = APNet2_DataLoader(
        dataset=ds,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        # collate_fn=apnet2_collate_update_prebatched,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        cnt += i.y.shape[0]
        print(i.y.shape)
    print("Number of labels in dataset:", cnt)
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_None*.pt"):
        os.remove(i)
    assert (number_dimers - number_dimers % batch_size) == cnt, (
        f"Expected {number_dimers} points, but got {cnt} points"
    )


@pytest.mark.skip("Models no longer available vi PyPI")
def test_dapnet2_dataset_size_prebatched_qcel_molecules_in_memory():
    batch_size = 4
    datapoint_storage_n_objects = 4
    prebatched = True
    number_dimers = 31
    qcel_molecules = [mol_dimer] * number_dimers
    energy_labels = [np.array([1.0]) for _ in range(len(qcel_molecules))]
    ds = dapnet2_module_dataset_apnetStored(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
    )
    print(ds)
    train_loader = APNet2_DataLoader(
        dataset=ds,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=apnet2_collate_update_prebatched,
    )
    cnt = 0
    print("train_loader")
    for i in train_loader:
        print(i)
        cnt += i.y.shape[0]
        print(i.y.shape)
    print("Number of labels in dataset:", cnt)
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_None*.pt"):
        os.remove(i)
    assert (number_dimers) == cnt, (
        f"Expected {number_dimers} points, but got {cnt} points"
    )


@pytest.mark.skip("Models no longer available vi PyPI")
def test_dapnet2_dataset_size_qcel_molecules_in_memory():
    batch_size = 4
    datapoint_storage_n_objects = 4
    prebatched = False
    number_dimers = 31
    qcel_molecules = [mol_dimer] * number_dimers
    energy_labels = [np.array([1.0]) for _ in range(len(qcel_molecules))]
    ds = dapnet2_module_dataset_apnetStored(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
    )
    print(ds)
    train_loader = APNet2_DataLoader(
        dataset=ds,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=apnet2_collate_update_prebatched,
    )
    cnt = 0
    for i in train_loader:
        print(i)
        cnt += i.y.shape[0]
        print(i.y.shape)
    print("Number of labels in dataset:", cnt)
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_None*.pt"):
        os.remove(i)
    assert (number_dimers) == cnt, (
        f"Expected {number_dimers} points, but got {cnt} points"
    )


@pytest.mark.skip("Models no longer available vi PyPI")
def test_apnet2_train_qcel_molecules_in_memory_transfer():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 6
    prebatched = False
    qcel_molecules = [mol_dimer] * 31
    energy_labels = [1.0 for _ in range(len(qcel_molecules))]
    print(
        qcel_molecules[0],
        energy_labels[0],
    )
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
        random_seed=None,
    )
    ap2 = APNet2Model().set_pretrained_model(model_id=0)
    v_0 = ap2.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    ap2.train(
        ds,
        n_epochs=6,
        skip_compile=True,
        transfer_learning=True,
    )
    v = ap2.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    print(np.sum(v_0, axis=1), np.sum(v, axis=1))
    assert np.allclose(np.sum(v, axis=1), np.ones(2), atol=1e-1)


@pytest.mark.skip("Models no longer available vi PyPI")
def test_dapnet2_train_qcel_molecules_in_memory_transfer():
    batch_size = 4
    datapoint_storage_n_objects = 4
    prebatched = False
    number_dimers = 31
    qcel_molecules = [mol_dimer] * number_dimers
    # qcel_molecules_pair = [mol_dimer, mol_dimer2]
    qcel_molecules_pair = [mol_dimer, mol_dimer]
    energy_labels = [np.array([1.0]) for _ in range(len(qcel_molecules))]
    ds = dapnet2_module_dataset_apnetStored(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
    )
    dap2 = dAPNet2Model(
        atom_model=AtomModels.ap2_atom_model.AtomModel().set_pretrained_model(
            model_id=0
        ),
        apnet2_model=APNet2Model()
        .set_pretrained_model(model_id=0)
        .set_return_hidden_states(True),
    )
    v_0 = dap2.predict_qcel_mols(qcel_molecules_pair, batch_size=2)
    dap2.train(
        ds,
        n_epochs=6,
        skip_compile=True,
    )
    v = dap2.predict_qcel_mols(qcel_molecules_pair, batch_size=2)
    print(v_0, v)
    assert np.allclose(v, np.ones(2), atol=1e-1)


def test_apnet2_train_qcel_molecules_in_memory():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 6
    prebatched = False
    qcel_molecules = [mol_dimer] * 31
    energy_labels = [[1.0] * 4 for _ in range(len(qcel_molecules))]
    atom_model = AtomModels.ap2_atom_model.AtomModel().set_pretrained_model(model_id=0)
    ap2 = APNet2Model()
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=None,
        max_size=None,
        force_reprocess=True,
        atom_model=atom_model,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        qcel_molecules=qcel_molecules,
        energy_labels=energy_labels,
        in_memory=True,
        random_seed=None,
    )
    ap2.train(
        ds,
        n_epochs=3,
        skip_compile=True,
        transfer_learning=False,
        lr=0.005,
    )
    # This also tests to make sure only best model is returned
    v_0 = ap2.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    ap2.train(
        ds,
        n_epochs=1,
        skip_compile=True,
        transfer_learning=False,
        lr=0.5,
    )
    v = ap2.predict_qcel_mols(qcel_molecules[0:2], batch_size=2)
    print(v_0, v)
    assert np.allclose(v_0, v, atol=1e-6)


def test_apnet2_dataset_size_prebatched_train_spec8():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = True
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        # split="test",
        print_level=2,
        random_seed=None,
    )
    print()
    print(ds)
    print(ds.training_batch_size)
    ap2 = APNet2Model()
    print("Example input before training:")
    print(ap2.eval_fn(ap2.example_input()))
    ap2.train(
        ds,
        n_epochs=3,
        skip_compile=True,
    )
    print("Example input after training:")
    print(ap2.eval_fn(ap2.example_input()))


def test_apnet2_dataset_size_prebatched_train_spec9():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = True
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=9,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        split="train",
        print_level=2,
        random_seed=None,
    )
    print()
    print(ds)
    print(ds.training_batch_size)
    ap2 = APNet2Model()
    ap2.train(
        ds,
        n_epochs=3,
        skip_compile=True,
    )


def test_dapnet2_dataset_size_no_prebatched():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = False
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    ds = dapnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        # split="test",
        print_level=2,
        m1="Elst_aug",
        m2="Exch_aug",
    )
    print()
    print(ds)

    train_loader = APNet2_DataLoader(
        dataset=ds,
        # batch_size=1,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        # collate_fn=dapnet2_collate_update_prebatched,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        cnt += i.y.shape[0]
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(
        f"{data_path}/processed_delta/dimer_dap2_spec_8_Elstaug_to_Exchaug_*.pt"
    ):
        os.remove(i)
    assert ds_labels == cnt, f"Expected {len(ds)} points, but got {cnt} points"


def test_dapnet2_dataset_size_prebatched():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = True
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    for i in glob(
        f"{data_path}/processed_delta/dimer_dap2_spec_8_Elst_aug_to_Exch_aug_*.pt"
    ):
        os.remove(i)
    ds = dapnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        m1="Elst_aug",
        m2="Exch_aug",
    )
    print()
    print(ds)
    print(ds.training_batch_size)

    train_loader = APNet2_DataLoader(
        dataset=ds,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        print(i)
        cnt += i.y.shape[0]
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(
        f"{data_path}/processed_delta/dimer_dap2_spec_8_Elst_aug_to_Exch_aug_*.pt"
    ):
        os.remove(i)
    assert ds_labels * ds.batch_size == cnt, (
        f"Expected {len(ds) * ds.batch_size} points, but got {cnt} points"
    )


@pytest.mark.skip("Models no longer available vi PyPI")
def test_dapnet2_dataset_ap2_stored_size_prebatched():
    batch_size = 2
    datapoint_storage_n_objects = 8
    prebatched = True
    collate = apnet2_collate_update_prebatched if prebatched else apnet2_collate_update
    for i in glob(f"{data_path}/processed_delta/dimer_dap2_ap2_spec_8_*.pt"):
        os.remove(i)
    ds = dapnet2_module_dataset_apnetStored(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        batch_size=batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        m1="Elst_aug",
        m2="Exch_aug",
    )
    print()
    print(ds)
    print(ds.training_batch_size)

    train_loader = APNet2_DataLoader(
        dataset=ds,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        cnt += i.y.shape[0]
        print(i)
        print(i.y)
    print("Number of labels in dataset:", cnt)
    ds_labels = int(len(ds))
    print("Number of labels in dataset:", ds_labels)
    for i in glob(f"{data_path}/processed_delta/dimer_dap2_ap2_spec_8_*.pt"):
        os.remove(i)
    for i in glob(f"{data_path}/processed_delta/targets_Elst_aug_to_Exch_aug.pt"):
        os.remove(i)
    assert ds_labels * ds.batch_size - 1 == cnt, (
        f"Expected {ds_labels * ds.batch_size - 1} points, but got {cnt} points"
    )


@pytest.mark.skip("Models no longer available vi PyPI")
def test_dapnet2_dataset_ap2_stored_size_prebatched_train():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = True
    print(am_path)
    for i in glob(
        f"{data_path}/processed_delta/dimer_dap2_spec_8_Elst_aug_to_Exch_aug_*.pt"
    ):
        os.remove(i)
    ds = dapnet2_module_dataset_apnetStored(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        batch_size=batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        m1="Elst_aug",
        m2="Exch_aug",
    )
    apnet2_model = APNet2Model().set_pretrained_model(model_id=0)
    apnet2_model.model.return_hidden_states = True
    dapnet2 = dAPNet2Model(apnet2_model=apnet2_model, dataset=ds)
    dapnet2.train(
        n_epochs=3,
        skip_compile=True,
    )
    for i in glob(
        f"{data_path}/processed_delta/dimer_dap2_spec_8_Elst_aug_to_Exch_aug_*.pt"
    ):
        os.remove(i)
    return


@pytest.mark.skip("Models no longer available vi PyPI")
def test_dapnet2_dataset_size_prebatched_train():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = True
    print(am_path)
    for i in glob(
        f"{data_path}/processed_delta/dimer_dap2_spec_8_Elst_aug_to_Exch_aug_*.pt"
    ):
        os.remove(i)
    ds = dapnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        print_level=2,
        m1="Elst_aug",
        m2="Exch_aug",
    )
    apnet2_model = APNet2Model().set_pretrained_model(model_id=0).model
    apnet2_model.return_hidden_states = True
    dapnet2 = APNet2_dAPNet2Model(apnet2_mpnn=apnet2_model, dataset=ds)
    dapnet2.train(
        n_epochs=3,
        skip_compile=True,
    )
    for i in glob(
        f"{data_path}/processed_delta/dimer_dap2_spec_8_Elst_aug_to_Exch_aug_*.pt"
    ):
        os.remove(i)
    return


def test_apnet3_dataset_size_no_prebatched():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = False
    collate = apnet3_collate_update_prebatched if prebatched else apnet3_collate_update
    ds = apnet3_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_hf_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        # split="test",
        print_level=2,
    )
    print()
    print(ds)

    train_loader = APNet2_DataLoader(
        dataset=ds,
        # batch_size=1,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        # collate_fn=apnet2_collate_update_prebatched,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        cnt += i.y.shape[0]
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(f"{data_path}/processed/dimer_ap3_spec_8*.pt"):
        os.remove(i)
    assert ds_labels == cnt, f"Expected {len(ds)} points, but got {cnt} points"


def test_apnet3_dataset_size_prebatched():
    batch_size = 2
    atomic_batch_size = 4
    datapoint_storage_n_objects = 8
    prebatched = True
    collate = apnet3_collate_update_prebatched if prebatched else apnet3_collate_update
    ds = apnet3_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=8,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_hf_path,
        atomic_batch_size=atomic_batch_size,
        datapoint_storage_n_objects=datapoint_storage_n_objects,
        batch_size=batch_size,
        prebatched=prebatched,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        # split="test",
        print_level=2,
    )
    print()
    print(ds)
    print(ds.training_batch_size)

    train_loader = APNet2_DataLoader(
        dataset=ds,
        # batch_size=1,
        batch_size=ds.training_batch_size,
        shuffle=False,
        num_workers=1,
        # collate_fn=apnet3_collate_update_prebatched,
        collate_fn=collate,
    )
    cnt = 0
    for i in train_loader:
        cnt += i.y.shape[0]
    print("Number of labels in dataset:", cnt)
    ds_labels = len(ds)
    for i in glob(f"{data_path}/processed/dimer_ap3_spec_8*.pt"):
        os.remove(i)
    assert ds_labels * ds.batch_size == cnt, (
        f"Expected {len(ds) * ds.batch_size} points, but got {cnt} points"
    )


def test_apnet2_model_train():
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        atom_model_path=am_path,
        atomic_batch_size=1000,
        num_devices=1,
        skip_processed=False,
        split="train",
    )
    apnet2 = APNet2Model(
        dataset=ds,
        ds_root=data_path,
        ds_spec_type=spec_type,
        ds_force_reprocess=False,
        ignore_database_null=False,
        ds_atomic_batch_size=1000,
        ds_num_devices=1,
        ds_skip_process=False,
        # ds_max_size=10,
    )
    apnet2.train(
        model_path="./models/ap2_test.pt",
        n_epochs=1,
        world_size=1,
        omp_num_threads_per_process=8,
        lr=2e-3,
        lr_decay=0.10,
        # lr_decay=None,
        skip_compile=True,
    )
    return


def test_apnet2_model_train_small():
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        atom_model_path=am_path,
        batch_size=2,
        atomic_batch_size=4,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        split="train",
    )
    apnet2 = APNet2Model(
        dataset=ds,
        ds_root=data_path,
        ds_spec_type=spec_type,
        ds_force_reprocess=False,
        ignore_database_null=False,
        ds_atomic_batch_size=4,
        ds_num_devices=1,
        ds_skip_process=False,
        # ds_max_size=10,
    )
    apnet2.train(
        model_path="./models/ap2_test.pt",
        n_epochs=1,
        world_size=1,
        omp_num_threads_per_process=8,
        lr=2e-3,
        lr_decay=0.10,
        skip_compile=True,
        # lr_decay=None,
    )
    return


def test_apnet2_model_train_small_r_cut_im():
    r_cut_im = 16.0
    n_rbf = 12
    ds = apnet2_module_dataset(
        root=data_path,
        r_cut=5.0,
        r_cut_im=r_cut_im,
        spec_type=5,
        max_size=None,
        force_reprocess=True,
        atom_model_path=am_path,
        batch_size=2,
        atomic_batch_size=4,
        num_devices=1,
        skip_processed=False,
        skip_compile=True,
        split="train",
    )
    apnet2 = APNet2Model(
        dataset=ds,
        ds_root=data_path,
        ds_spec_type=spec_type,
        r_cut_im=r_cut_im,
        n_rbf=n_rbf,
        ds_force_reprocess=False,
        ignore_database_null=False,
        ds_atomic_batch_size=4,
        ds_num_devices=1,
        ds_skip_process=False,
        # ds_max_size=10,
    )
    apnet2.train(
        model_path="./models/ap2_test_r_cut_im.pt",
        n_epochs=1,
        world_size=1,
        omp_num_threads_per_process=8,
        lr=2e-3,
        lr_decay=0.10,
        skip_compile=False,
        # lr_decay=None,
    )
    return


def test_atom_model_train():
    ds = atomic_datasets.atomic_module_dataset(
        root=data_path,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=6,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=1,
    )
    print(ds)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "2"
    am = AtomModels.ap2_atom_model.AtomModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    am.train(
        n_epochs=3,
        batch_size=1,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=2,
        omp_num_threads_per_process=4,
        random_seed=42,
    )
    am = AtomModels.ap2_atom_model.AtomModel(
        use_GPU=True,
        ignore_database_null=False,
        dataset=ds,
    )
    print(am)
    # GPU
    am.train(
        n_epochs=3,
        batch_size=1,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        skip_compile=True,
        shuffle=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=None,
        random_seed=42,
    )
    return


def test_AtomTypeParamModel_train():
    """
    AtomTypeParamModel hirsfhfeld_valencewidth uses atomic_hirshfeld_module_dataset with ap2_atom_model
    """
    ds = atomic_datasets.atomic_hirshfeld_module_dataset(
        root=data_path,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=1,
    )
    print(ds)
    am = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    print(am)
    am.train(
        n_epochs=3,
        batch_size=1,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=None,
        random_seed=42,
    )


def test_AtomTypeParamModel_elst_train():
    """
    AtomTypeParamModel hirsfhfeld_valencewidth uses atomic_hirshfeld_module_dataset with ap2_atom_model
    """
    qcel_molecules = [mol_cliff_water_close] * 4
    energy_labels = [
        np.array([-10.779292828139122, -500, -3.414543432719425, 10000])
        for _ in range(len(qcel_molecules))
    ]
    am = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am.set_pretrained_model(model_id=0)
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        # atom_model_type='AtomTypeParamNN',
        atom_model_type="AtomMPNN",
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=None,
        ds_qcel_molecules=qcel_molecules,
        ds_energy_labels=energy_labels,
        param_start_mean=[3.2],
        param_start_std=[0.2],
        n_neuron=32,
        n_params=1,
        dimer_eval_type="elst_damping",
    )
    param_mod.train(
        n_epochs=3,
        # skip_compile=True,
        skip_compile=False,
        lr=5e-4,
        split_percent=0.5,
    )


def test_AtomTypeParamModel_ind_train():
    """ """
    qcel_molecules = [mol_cliff_water_close] * 4
    energy_labels = [
        np.array([-10.779292828139122, -500, -3.414543432719425, 10000])
        for _ in range(len(qcel_molecules))
    ]
    am = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
        # current_file_path + "/../models/ap_atomTypeParamModel/am_0.pt",
    )
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        atom_model_type="AtomTypeParamNN",
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=None,
        ds_qcel_molecules=qcel_molecules,
        ds_energy_labels=energy_labels,
        param_start_mean=[1.8],
        param_start_std=[0.05],
        n_neuron=32,
        n_params=1,
        dimer_eval_type="induced_dipole_param",
    )
    param_mod.train(
        # n_epochs=100,
        n_epochs=1,
        # skip_compile=True,
        skip_compile=False,
        lr=5e-4,
        split_percent=0.5,
    )


def test_AtomTypeParamModel_AM_DimerProp_train():
    df = pd.read_pickle(current_file_path + "/dataset_data/elst_damping_test.pkl")
    qcel_molecules = df["qcel_molecule"].to_list()
    energy_labels = (
        df[["SAPT0 ELST", "SAPT0 EXCH", "SAPT0 IND", "SAPT0 DISP"]].values
        * qcel.constants.hartree2kcalmol
    )
    print(energy_labels)
    am = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=current_file_path
        + "/../models/am_ensemble/am_0.pt",
        pre_trained_model_path=current_file_path
        + "/../models/ap_atomTypeParamModel/am_h+1_0.pt",
    )
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        atom_model_type="AtomTypeParamNN",
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=None,
        ds_qcel_molecules=qcel_molecules,
        ds_energy_labels=energy_labels,
        param_start_mean=[0.9, 1.8],
        param_start_std=[0.15, 0.05],
        n_neuron=64,
        n_params=2,
        dimer_eval_type="elst_damping__induced_dipole",
    )
    param_mod.train(
        n_epochs=3,
        # n_epochs=25,
        # skip_compile=True,
        skip_compile=False,
        lr=5e-5,
        split_percent=0.5,
    )


def test_AtomTypeParamModel_AM_DimerProp_train_elst_only():
    """
    AtomTypeParamModel hirsfhfeld_valencewidth uses atomic_hirshfeld_module_dataset with ap2_atom_model
    """
    df = pd.read_pickle(current_file_path + "/dataset_data/elst_damping_test.pkl")
    qcel_molecules = df["qcel_molecule"].to_list()
    for i in qcel_molecules:
        print(i.to_string("psi4"))
    energy_labels = (
        df[["SAPT0 ELST", "SAPT0 EXCH", "SAPT0 IND", "SAPT0 DISP"]].values
        * qcel.constants.hartree2kcalmol
    )
    print(energy_labels)

    am = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=current_file_path
        + "/../models/am_ensemble/am_0.pt",
        pre_trained_model_path=current_file_path
        + "/../models/ap_atomTypeParamModel/am_h+1_0.pt",
    )
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        atom_model_type="AtomTypeParamNN",
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=None,
        ds_qcel_molecules=qcel_molecules,
        ds_energy_labels=energy_labels,
        param_start_mean=[0.9],
        param_start_std=[0.15],
        n_neuron=64,
        n_params=1,
        dimer_eval_type="elst_damping",
    )
    param_mod.train(
        n_epochs=3,
        # n_epochs=25,
        # skip_compile=True,
        skip_compile=False,
        lr=5e-5,
        split_percent=0.5,
    )


def test_AtomTypeParamModel_AM_DimerProp_train_elst_only():
    """
    AtomTypeParamModel hirsfhfeld_valencewidth uses atomic_hirshfeld_module_dataset with ap2_atom_model
    """
    df = pd.read_pickle(current_file_path + "/dataset_data/elst_damping_test.pkl")
    qcel_molecules = df["qcel_molecule"].to_list()
    for i in qcel_molecules:
        print(i.to_string("psi4"))
    energy_labels = (
        df[["SAPT0 ELST", "SAPT0 EXCH", "SAPT0 IND", "SAPT0 DISP"]].values
        * qcel.constants.hartree2kcalmol
    )
    print(energy_labels)

    am = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=current_file_path
        + "/../models/am_ensemble/am_0.pt",
        pre_trained_model_path=current_file_path
        + "/../models/ap_atomTypeParamModel/am_h+1_0.pt",
    )
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        atom_model_type="AtomTypeParamNN",
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=None,
        ds_qcel_molecules=qcel_molecules,
        ds_energy_labels=energy_labels,
        param_start_mean=[0.9],
        param_start_std=[0.15],
        n_neuron=64,
        n_params=1,
        dimer_eval_type="elst_damping",
    )
    param_mod.train(
        n_epochs=3,
        # n_epochs=25,
        # skip_compile=True,
        skip_compile=False,
        lr=5e-5,
        split_percent=0.5,
    )


def test_AtomTypeNNParamModel_AM_DimerProp_train_elst_only():
    df = pd.read_pickle(current_file_path + "/dataset_data/elst_damping_test.pkl")
    qcel_molecules = df["qcel_molecule"].to_list()
    energy_labels = (
        df[["SAPT0 ELST", "SAPT0 EXCH", "SAPT0 IND", "SAPT0 DISP"]].values
        * qcel.constants.hartree2kcalmol
    )

    am = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=current_file_path
        + "/../models/am_ensemble/am_0.pt",
        pre_trained_model_path=current_file_path
        + "/../models/ap_atomTypeParamModel/am_h+1_0.pt",
    )
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        atom_model_type="AtomTypeParamNN",
        model_type="AtomTypeParamNN",
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=None,
        ds_qcel_molecules=qcel_molecules,
        ds_energy_labels=energy_labels,
        param_start_mean=[3.3],
        param_start_std=[0.3],
        n_neuron=64,
        n_params=1,
        dimer_eval_type="elst_damping",
    )
    param_mod.train(
        n_epochs=3,
        # skip_compile=True,
        skip_compile=False,
        lr=5e-4,
        split_percent=0.5,
    )


def test_AtomTypeParamNNModel_AM_DimerProp_train_elst_only_spec7():
    am = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=current_file_path
        + "/../models/am_ensemble/am_0.pt",
        pre_trained_model_path=current_file_path
        + "/../models/ap_atomTypeParamModel/am_h+1_0.pt",
    )
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        atom_model_type="AtomTypeParamNN",
        model_type="AtomTypeParamNN",
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=7,
        param_start_mean=[2.6],
        param_start_std=[0.3],
        n_neuron=32,
        n_params=1,
        dimer_eval_type="elst_damping",
    )
    param_mod.train(
        n_epochs=3,
        skip_compile=False,
        lr=5e-4,
    )


def test_AtomTypeParamModel_AM_DimerProp_train_elst_only_spec7():
    """
    AtomTypeParamModel hirsfhfeld_valencewidth uses atomic_hirshfeld_module_dataset with ap2_atom_model
    """
    am = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        atom_model_type="AtomTypeParamNN",
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=7,
        param_start_mean=[1.6],
        param_start_std=[0.15],
        n_neuron=64,
        n_params=1,
        dimer_eval_type="elst_damping",
    )
    param_mod.train(
        n_epochs=3,
        skip_compile=False,
        lr=5e-5,
        model_path=None,
    )


def test_ap3_spec7():
    atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=am_path,
        pre_trained_model_path=at_hf_vw_path,
    )
    atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        ds_root=data_path,
        use_GPU=False,
        n_neuron=64,
        n_params=1,
        ignore_database_null=True,
        atom_model=atom_type_hf_vw_model.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=at_elst_path,
    )
    ap3 = apnet_pt.AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model(
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=7,
        ds_in_memory=False,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        use_precomputed_classical=True,
    )
    ap3.train(
        n_epochs=3,
        skip_compile=True,
        transfer_learning=False,
        lr=5e-4,
        dataloader_num_workers=4,
    )
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_*.pt"):
        os.remove(i)


def test_ap3d3_spec5_non_precomputed_lmdb_training(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path(data_path) / "raw" / "t_train_19.pkl",
        raw_dir / "t_train.pkl",
    )
    shutil.copy2(
        Path(data_path) / "raw" / "t_test_19.pkl",
        raw_dir / "t_test.pkl",
    )

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
        freeze_atom_model=True,
    )
    ap3d3 = apnet_pt.AtomPairwiseModels.apnet3_d3_fused.APNet3D3_AtomType_Model(
        ds_root=str(tmp_path),
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=5,
        ds_in_memory=False,
        ds_class_type="lmdb",
        ds_atomic_batch_size=4,
        ds_batch_size=2,
        ds_datapoint_storage_n_objects=4,
        atom_type_model=atom_type_hf_vw_model.model,
        dimer_prop_model=atom_type_elst_model.dimer_model,
        use_precomputed_classical=False,
        freeze_dimer_prop_model=False,
        n_rbf=4,
        n_neuron=16,
        n_embed=4,
    )

    assert isinstance(ap3d3.dataset, list)
    assert isinstance(ap3d3.dataset[0], ap2_fused_module_dataset_lmdb)
    assert isinstance(ap3d3.dataset[1], ap2_fused_module_dataset_lmdb)
    assert len(ap3d3.dataset[0]) > 0
    assert len(ap3d3.dataset[1]) > 0

    ap3d3.train(
        n_epochs=1,
        skip_compile=True,
        lr=5e-4,
        dataloader_num_workers=0,
        shuffle=False,
    )

    assert (tmp_path / "processed" / "lmdb_ap2_fused_train_spec_5").exists()
    assert (tmp_path / "processed" / "lmdb_ap2_fused_test_spec_5").exists()


def test_ap2_spec7():
    atom_model = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        pre_trained_model_path=current_file_path + "/../models/am_ensemble/am_0.pt",
        ignore_database_null=True,
    )
    ap2 = apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
        atom_model=atom_model.model,
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=7,
    )
    ap2.train(
        n_epochs=5,
        skip_compile=False,
        transfer_learning=False,
        lr=5e-4,
        dataloader_num_workers=4,
    )
    for i in glob(f"{data_path}/processed/dimer_ap2_spec_*.pt"):
        os.remove(i)


def test_atomhirshfeld_model_train():
    ds = atomic_datasets.atomic_hirshfeld_module_dataset(
        root=data_path,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=1,
    )
    print(ds)
    am = AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    print(am)
    am.train(
        n_epochs=3,
        batch_size=1,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=None,
        random_seed=42,
    )
    return


def test_mtp_mtp_elst_qcel_mols():
    qcel_molecules = [mol_dimer] * 4
    energy_labels = [
        np.array([-10.779292828139122, 0, 0, 0]) for _ in range(len(qcel_molecules))
    ]
    print(energy_labels)
    am = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am.set_pretrained_model(model_id=0)
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=None,
        ds_qcel_molecules=qcel_molecules,
        ds_energy_labels=energy_labels,
        param_start_mean=2.0,
        param_start_std=0.1,
        n_neuron=16,
    )
    print(param_mod)
    param_mod.train(
        n_epochs=3,
        skip_compile=True,
        lr=5e-4,
        split_percent=0.5,
    )


def test_mtp_mtp_elst_dataset():
    am = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am.set_pretrained_model(model_id=0)
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        ignore_database_null=False,
        # pre_trained_model_path="nan.pt",
        ds_force_reprocess=True,
        ds_spec_type=7,
        use_GPU=False,
        ds_root=data_path,
        param_start_mean=1.5,
        param_start_std=0.1,
        n_neuron=32,
    )
    param_mod.train(
        n_epochs=2,
        skip_compile=False,
        lr=5e-3,
        # model_path='nan.pt',
    )


def test_mtp_mtp_elst_eval():
    am = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am.set_pretrained_model(model_id=0)
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        ignore_database_null=False,
        # pre_trained_model_path="nan.pt",
        ds_force_reprocess=True,
        ds_spec_type=7,
        use_GPU=False,
        ds_root=data_path,
        param_start_mean=1.5,
        param_start_std=0.1,
        n_neuron=32,
    )
    batch = param_mod._qcel_example_input([mol_dimer_ion])
    v = param_mod.model(batch)
    print(v[-1])
    batch = param_mod._qcel_dimer_example_input([mol_dimer_ion])
    v = param_mod.dimer_model(batch)
    print(v[-1])
    elst_energy = param_mod.predict_qcel_mols_dimer([mol_dimer_ion], batch_size=1)
    print(f"Predicted ELST energy: {elst_energy}")
    return


def test_induced_dipole_qcel_mols():
    qcel_molecules = [mol_cliff_water_close] * 4
    energy_labels = [
        np.array([-1000, -500, -3.414543432719425, 10000])
        for _ in range(len(qcel_molecules))
    ]
    print(energy_labels)
    am = apnet_pt.AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am.set_pretrained_model(current_file_path + "/../models/am_hf_ensemble/am_0.pt")
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        ds_root=data_path,
        ignore_database_null=False,
        ds_force_reprocess=True,
        use_GPU=False,
        ds_spec_type=None,
        ds_qcel_molecules=qcel_molecules,
        ds_energy_labels=energy_labels,
        param_start_mean=1.3,
        param_start_std=0.05,
        n_neuron=32,
        dimer_eval_type="induced_dipole",
    )
    print(param_mod)
    param_mod.train(
        n_epochs=3,
        # skip_compile=True,
        skip_compile=False,
        lr=5e-4,
        split_percent=0.5,
    )


def test_induced_dipole_dataset():
    am = apnet_pt.AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=True,
    )
    am.set_pretrained_model(am_hf_path)
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        ignore_database_null=False,
        # pre_trained_model_path="nan.pt",
        ds_force_reprocess=True,
        ds_spec_type=7,
        use_GPU=False,
        ds_root=data_path,
        param_start_mean=0.4,
        param_start_std=0.2,
        n_neuron=32,
        dimer_eval_type="induced_dipole",
    )
    param_mod.train(
        n_epochs=3,
        skip_compile=False,
        lr=5e-4,
        # model_path='nan.pt',
    )


def test_induced_dipole_eval():
    am = apnet_pt.AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am.set_pretrained_model(am_hf_path)
    param_mod = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
        atom_model=am.model,
        ignore_database_null=False,
        # pre_trained_model_path="nan.pt",
        ds_force_reprocess=True,
        ds_spec_type=7,
        use_GPU=False,
        ds_root=data_path,
        param_start_mean=1.5,
        param_start_std=0.1,
        n_neuron=32,
        dimer_eval_type="induced_dipole",
    )
    print("\nSingle Molecule Eval\n")
    batch = param_mod._qcel_example_input([mol_dimer_ion])
    v = param_mod.model(batch)
    print(v[-1])
    print("\nDimer Molecule Eval\n")
    batch = param_mod._qcel_dimer_example_input([mol_dimer])
    v = param_mod.dimer_model(batch)
    print(v[-1])
    print("\nDimer Molecule Ion Eval\n")
    batch = param_mod._qcel_dimer_example_input([mol_dimer_ion])
    v = param_mod.dimer_model(batch)
    print(v[-1])
    elst_energy = param_mod.predict_qcel_mols_dimer([mol_dimer_ion], batch_size=1)
    print(f"Predicted INDU energy: {elst_energy}")
    return


def test_ap2_elst_dataset():
    am = apnet_pt.AtomModels.ap2_atom_model.AtomModel(
        ds_root=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    am.set_pretrained_model(model_id=0)
    param_mod = apnet_pt.AtomPairwiseModels.apnet2_fused.APNet2_AM_Model(
        atom_model=am.model,
        ignore_database_null=False,
        ds_force_reprocess=True,
        ds_spec_type=7,
        use_GPU=False,
        ds_root=data_path,
        n_neuron=32,
    )
    param_mod.train(
        # n_epochs=500,
        n_epochs=3,
        skip_compile=True,
        lr=5e-4,
    )


if __name__ == "__main__":
    # test_AtomTypeParamModel_AM_DimerProp_train_elst_only_spec7()
    # test_AtomTypeParamNNModel_AM_DimerProp_train_elst_only_spec7()
    # test_AtomTypeParamModel_train()
    # test_induced_dipole_qcel_mols()
    # test_AtomTypeParamModel_AM_DimerProp_train()
    # test_induced_dipole_eval()
    # test_induced_dipole_dataset()

    # test_atomhirshfeld_model_train()
    # test_AtomTypeParamModel_elst_train()

    # test_AtomTypeParamModel_AM_DimerProp_train_elst_only_spec7()
    # test_ap2_spec7()
    test_ap3_spec7()
    # test_ap3_train()
    # test_AtomTypeParamModel_AM_DimerProp_train_elst_only()
    # test_AtomTypeParamModel_ind_train()

    # test_mtp_mtp_elst_qcel_mols()
    # test_mtp_mtp_elst_eval()
    # test_atom_model_train()
    # test_mtp_mtp_elst_dataset()

    # test_ap2_elst_dataset()
    # test_mtp_mtp_elst_dataset()
    # test_apnet2_train_qcel_molecules_in_memory()
    # test_apnet2_train_qcel_molecules_in_memory()
    # test_dapnet2_dataset_size_prebatched_qcel_molecules_in_memory()
    # test_apnet2_dataset_size_prebatched_train_spec8()
    # test_apnet2_dataset_size_prebatched()
    # test_dapnet2_dataset_size_prebatched()
    # test_dapnet2_train_qcel_molecules_in_memory_transfer()
    # test_apnet2_model_train()
    pass
