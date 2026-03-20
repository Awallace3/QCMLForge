import torch
import os
from .. import util
from ..pairwise_datasets import (
    AtomModel,
    Data,
    Dataset,
    pairwise_edges,
    pairwise_edges_im,
    natural_key,
    apnet2_collate_update,
)
from ..AtomPairwiseModels.apnet2 import (
    APNet2Model,
)
from ..hf_pretrained import resolve_pretrained_path
from typing import List, Optional, Sequence, Union
from .. import atomic_datasets
from time import time
from glob import glob
from pathlib import Path
import os.path as osp
import qcelemental as qcel
from apnet_pt import constants

# file dir
# prefix = os.path.dirname(os.path.abspath(__file__))
# pretrained_atom_model_path = prefix + "/../models/am_ensemble/"
# pretrained_pairwise_model_path = prefix + "/../models/ap2_ensemble/"
pretrained_atom_model_path = None
pretrained_pairwise_model_path = None


def clean_str_for_filename(string):
    """
    Remove all non-alphanumeric characters from a string and replace spaces with underscores.
    """
    # Remove all non-alphanumeric characters
    string = string.replace("(", "_LP_").replace(")", "_RP_")
    string = "".join(e for e in string if e.isalnum() or e.isspace() or e in ["-", "_"])
    # Replace spaces with underscores
    string = string.replace(" ", "_")
    return string


def dapnet2_collate_update_no_target(batch):
    batched_data = Data(
        # ZA=torch.cat([data.ZA for data in batch], dim=0),
        h_AB=torch.cat([data.h_AB for data in batch], dim=0),
        h_BA=torch.cat([data.h_BA for data in batch], dim=0),
        cutoff=torch.cat([data.cutoff for data in batch], dim=0),
        dimer_ind=torch.cat([data.dimer_ind for data in batch], dim=0),
        ndimer=torch.tensor([data.ndimer for data in batch], dtype=torch.long),
    )
    return batched_data


def dapnet2_collate_update(batch):
    y = torch.stack([data.y for data in batch], dim=0)
    batched_data = Data(
        y=y,
        h_AB=torch.cat([data.h_AB for data in batch], dim=0),
        h_BA=torch.cat([data.h_BA for data in batch], dim=0),
        cutoff=torch.cat([data.cutoff for data in batch], dim=0),
        dimer_ind=torch.cat([data.dimer_ind for data in batch], dim=0),
        ndimer=torch.tensor([data.ndimer for data in batch], dtype=torch.long),
    )
    return batched_data


class dapnet2_module_dataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=1,
        max_size=None,
        force_reprocess=True,
        skip_processed=True,
        # only need for processing
        atom_model_path=pretrained_atom_model_path,
        batch_size=16,
        atomic_batch_size=256,
        prebatched=False,
        skip_compile=False,
        # DO NOT CHANGE UNLESS YOU WANT TO RE-PROCESS THE DATASET
        datapoint_storage_n_objects=1000,
        in_memory=False,
        num_devices=1,
        split="all",  # train, test
        print_level=2,
        m1="B3LYP-D3/aug-cc-pVTZ/CP",
        m2="CCSD(T)/CBS/CP",
        qcel_molecules: Optional[List[qcel.models.Molecule]] = None,
        energy_labels: Optional[List[float]] = None,
    ):
        """
        spec_type definitions:
            1. dapnet dataset for bfdb interaction energies
        """
        self.print_level = print_level
        try:
            assert spec_type in [1, 2, 8]
        except Exception:
            print("Currently spec_type must be 1 or 2 for SAPT0/jun-cc-pVDZ")
            raise ValueError
        self.spec_type = spec_type
        self.prebatched = prebatched
        self.MAX_SIZE = max_size
        self.in_memory = in_memory
        self.m1 = m1
        self.m2 = m2
        self.data = []
        self.split = split
        self.r_cut = r_cut
        self.r_cut_im = r_cut_im
        self.qcel_molecules = None
        self.energy_labels = None
        # Store qcel_molecules and energy_labels if provided
        if qcel_molecules is not None and energy_labels is not None:
            self.qcel_molecules = qcel_molecules
            self.energy_labels = energy_labels
            if len(qcel_molecules) != len(energy_labels):
                raise ValueError(
                    "Length of qcel_molecules and energy_labels must match"
                )
            print(
                f"Received {len(qcel_molecules)} QCElemental molecules with energy labels"
            )
        self.force_reprocess = force_reprocess
        self.filename_methods = (
            clean_str_for_filename(m1) + "_to_" + clean_str_for_filename(m2)
        )
        self.datapoint_storage_n_objects = datapoint_storage_n_objects
        self.atomic_batch_size = atomic_batch_size
        self.batch_size = batch_size
        self.training_batch_size = batch_size if not prebatched else 1
        self.points_per_file = self.datapoint_storage_n_objects
        if self.prebatched:
            self.points_per_file *= self.batch_size
        elif self.in_memory:
            self.points_per_file = 1
        self.skip_processed = skip_processed
        if os.path.exists(root) is False:
            os.makedirs(root, exist_ok=True)
        if not self.skip_processed:
            if atom_model_path is None:
                atom_model_path = resolve_pretrained_path("am_ensemble/am_0.pt")
            self.atom_model = AtomModel(
                pre_trained_model_path=atom_model_path,
                ds_root=None,
                ignore_database_null=True,
            )
            self.atom_model.model.to(self.atom_model.device)
            if not skip_compile:
                self.atom_model.compile_model()
        super(dapnet2_module_dataset, self).__init__(root, transform, pre_transform)
        if self.force_reprocess:
            self.force_reprocess = False
            super(dapnet2_module_dataset, self).__init__(root, transform, pre_transform)
        print(f"{self.root=}, {self.spec_type=}, {self.in_memory=}")
        if self.in_memory:
            self.get = self.get_in_memory
        self.active_idx_data = None
        self.active_data = None

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed_delta")

    @property
    def raw_file_names(self):
        # TODO: enable users to specify data source via QCArchive, url, or local file
        # spec_1 = "spec_1" # 'SAPT0/jun-cc-pVDZ'
        if self.spec_type == 1:
            return [
                # 0.4 Train / 0.6 Test
                "3324_BFDBext_train_dimers.pkl",
                "3324_BFDBext_test_dimers.pkl",
            ]
        elif self.spec_type == 2:
            return [
                "DES370K_b3lyp_255586.pkl",
                # "DES370K_b3lyp_64.pkl",
            ]
        elif self.spec_type == 8:
            return [
                "t_val_19.pkl",
            ]
        else:
            return [
                "splinter_spec1.pkl",
            ]

    def reprocess_file_names(self):
        if self.force_reprocess:
            return ["file"]
        else:
            if self.split == "train":
                file_cmd = f"{self.processed_dir}/dimer_dap2_train_spec_{self.spec_type}_{self.filename_methods}*.pt"
            elif self.split == "test":
                file_cmd = f"{self.processed_dir}/dimer_dap2_test_spec_{self.spec_type}_{self.filename_methods}*.pt"
            else:
                file_cmd = f"{self.processed_dir}/dimer_dap2_spec_{self.spec_type}_{self.filename_methods}*.pt"
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                spec_files.sort(key=natural_key)
                if self.MAX_SIZE is not None:
                    max_size = int(self.MAX_SIZE / self.datapoint_storage_n_objects)
                if self.MAX_SIZE is not None:
                    if len(spec_files) > max_size and max_size > 0:
                        spec_files = spec_files[:max_size]
                    elif len(spec_files) > max_size:
                        spec_files = spec_files[:1]
                return spec_files
            else:
                return ["dimer_missing.pt"]

    @property
    def processed_file_names(self):
        return self.reprocess_file_names()

    def download(self):
        if self.energy_labels and self.qcel_molecules:
            return
        raise RuntimeError(
            "Dataset does not have a download method. Development usage only at the moment."
        )
        return

    def process(self):
        self.data = []
        idx = 0
        atomic_batch_size = self.atomic_batch_size
        data_objects = []
        RAs, RBs, ZAs, ZBs, TQAs, TQBs, targets = [], [], [], [], [], [], []
        if self.qcel_molecules is not None and self.energy_labels is not None:
            print("Processing directly from provided QCElemental molecules...")
            split_name = ""

            # Process directly from qcel_mols and energy_labels
            for mol in self.qcel_molecules:
                # Extract monomer data from dimer
                monA, monB = mol.get_fragment(0), mol.get_fragment(1)
                print(monA)

                # Get coordinates and atomic numbers for each monomer
                RA = torch.tensor(monA.geometry, dtype=torch.float32) * constants.au2ang
                RB = torch.tensor(monB.geometry, dtype=torch.float32) * constants.au2ang
                ZA = torch.tensor(monA.atomic_numbers, dtype=torch.int64)
                ZB = torch.tensor(monB.atomic_numbers, dtype=torch.int64)

                # Calculate total charges
                TQA = torch.tensor(monA.molecular_charge, dtype=torch.float32)
                TQB = torch.tensor(monB.molecular_charge, dtype=torch.float32)

                RAs.append(RA)
                RBs.append(RB)
                ZAs.append(ZA)
                ZBs.append(ZB)
                TQAs.append(TQA)
                TQBs.append(TQB)

            # Use provided energy labels
            targets = self.energy_labels
            # if targets[0] is not iterable, need to convert to list
            # if not isinstance(targets[0], (list, tuple)):
            #     targets = [[t] for t in targets]

            if self.MAX_SIZE is not None and len(RAs) > self.MAX_SIZE:
                RAs = RAs[: self.MAX_SIZE]
                RBs = RBs[: self.MAX_SIZE]
                ZAs = ZAs[: self.MAX_SIZE]
                ZBs = ZBs[: self.MAX_SIZE]
                TQAs = TQAs[: self.MAX_SIZE]
                TQBs = TQBs[: self.MAX_SIZE]
                targets = targets[: self.MAX_SIZE]

            print(
                f"Processing {len(RAs)} dimers from provided QCElemental molecules..."
            )
        else:
            for raw_path in self.raw_paths:
                split_name = ""
                if self.spec_type in [2, 5, 6, 7, 9]:
                    split_name = f"_{self.split}"
                    print(f"{split_name=}")
                    if self.split not in Path(raw_path).stem:
                        print(f"{self.split} is skipping {raw_path}")
                        continue
                print(f"raw_path: {raw_path}")
                print("Loading dimers...")
                RA, RB, ZA, ZB, TQA, TQB, label = util.load_dimer_dataset(
                    raw_path,
                    self.MAX_SIZE,
                    return_qcel_mols=False,
                    return_qcel_mons=False,
                    columns=[self.m1, self.m2],
                )
                labels = label[:, 0] - label[:, 1]
                RAs.extend(RA)
                RBs.extend(RB)
                ZAs.extend(ZA)
                ZBs.extend(ZB)
                TQAs.extend(TQA)
                TQBs.extend(TQB)
                targets.extend(labels)
            print("Creating data objects...")
            t1 = time()
            t2 = time()
            print(f"{len(RAs)=}, {self.atomic_batch_size=}, {self.batch_size=}")
            molA_data = []
            molB_data = []
            energies = []
            # targets = targets[:, 0] - targets[:, 1]
            for i in range(
                0,
                len(RAs) + len(RAs) % self.atomic_batch_size + 1,
                self.atomic_batch_size,
            ):
                if self.skip_processed:
                    datapath = osp.join(
                        self.processed_dir,
                        f"dimer_dap2{split_name}_spec_{self.spec_type}_{self.filename_methods}_{idx // self.points_per_file}.pt",
                    )
                    if osp.exists(datapath):
                        idx += atomic_batch_size
                        continue
                upper_bound = min(i + atomic_batch_size, len(RAs))
                for j in range(i, upper_bound):
                    monA_data = atomic_datasets.create_atomic_data(
                        ZAs[j], RAs[j], TQAs[j], r_cut=self.r_cut
                    )
                    monB_data = atomic_datasets.create_atomic_data(
                        ZBs[j], RBs[j], TQBs[j], r_cut=self.r_cut
                    )
                    molA_data.append(monA_data)
                    molB_data.append(monB_data)
                    energies.append(targets[j])
                if len(molA_data) != self.atomic_batch_size and j != len(RAs) - 1:
                    continue
                batch_A = atomic_datasets.atomic_collate_update_no_target(molA_data)
                qAs, muAs, thAs, hlistAs = self.atom_model.predict_multipoles_batch(
                    batch_A
                )
                batch_B = atomic_datasets.atomic_collate_update_no_target(molB_data)
                qBs, muBs, thBs, hlistBs = self.atom_model.predict_multipoles_batch(
                    batch_B
                )
                for j in range(len(molA_data)):
                    atomic_props_A = molA_data[j]
                    atomic_props_B = molB_data[j]
                    local_energies = energies[j]
                    qA, muA, quadA, hlistA = qAs[j], muAs[j], thAs[j], hlistAs[j]
                    qB, muB, quadB, hlistB = qBs[j], muBs[j], thBs[j], hlistBs[j]
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                    e_AA_source, e_AA_target = pairwise_edges(
                        atomic_props_A.R, self.r_cut
                    )
                    e_BB_source, e_BB_target = pairwise_edges(
                        atomic_props_B.R, self.r_cut
                    )
                    e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = (
                        pairwise_edges_im(
                            atomic_props_A.R, atomic_props_B.R, self.r_cut_im
                        )
                    )
                    y = torch.tensor(local_energies, dtype=torch.float32)
                    dimer_ind = torch.ones((1), dtype=torch.long) * j
                    data = Data(
                        y=y,
                        ZA=atomic_props_A.x,
                        RA=atomic_props_A.R,
                        ZB=atomic_props_B.x,
                        RB=atomic_props_B.R,
                        e_ABsr_source=e_ABsr_source,
                        e_ABsr_target=e_ABsr_target,
                        dimer_ind=dimer_ind,
                        e_ABlr_source=e_ABlr_source,
                        e_ABlr_target=e_ABlr_target,
                        dimer_ind_lr=dimer_ind,
                        e_AA_source=e_AA_source,
                        e_AA_target=e_AA_target,
                        e_BB_source=e_BB_source,
                        e_BB_target=e_BB_target,
                        total_charge_A=atomic_props_A.total_charge,
                        total_charge_B=atomic_props_B.total_charge,
                        qA=qA,
                        muA=muA,
                        quadA=quadA,
                        hlistA=hlistA,
                        qB=qB,
                        muB=muB,
                        quadB=quadB,
                        hlistB=hlistB,
                    )
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    data_objects.append(data)

                    if len(data_objects) == self.points_per_file:
                        datapath = osp.join(
                            self.processed_dir,
                            f"dimer_dap2{split_name}_spec_{self.spec_type}_{self.filename_methods}_{idx // self.points_per_file}.pt",
                        )
                        if self.print_level >= 2:
                            print(f"Saving to {datapath}")
                            print(len(data_objects))
                        # if we are pre-batching, we need to collate and save here.
                        if self.prebatched:
                            # collate based on batch_size
                            local_data_objects = []
                            for k in range(self.datapoint_storage_n_objects):
                                local_data_objects.append(
                                    apnet2_collate_update(
                                        data_objects[
                                            k * self.batch_size : (k + 1)
                                            * self.batch_size
                                        ]
                                    )
                                )
                            data_objects = local_data_objects
                        elif self.in_memory:
                            data_objects = data_objects[0]
                        if self.in_memory:
                            self.data.append(data_objects)
                        else:
                            torch.save(data_objects, datapath)
                        data_objects = []
                        if self.MAX_SIZE is not None and idx > self.MAX_SIZE:
                            break
                    idx += 1
                if self.print_level >= 2:
                    print(f"{i}/{len(RAs)}, {time() - t2:.2f}s, {time() - t1:.2f}s")
                elif self.print_level >= 1 and idx % 1000:
                    print(f"{i}/{len(RAs)}, {time() - t2:.2f}s, {time() - t1:.2f}s")
                t2 = time()
                molA_data = []
                molB_data = []
                energies = []
        if len(data_objects) > 0:
            if self.prebatched:
                # collate based on batch_size
                local_data_objects = []
                for k in range(len(data_objects) // self.batch_size):
                    local_data_objects.append(
                        apnet2_collate_update(
                            data_objects[
                                k * self.batch_size : (k + 1) * self.batch_size
                            ]
                        )
                    )
                data_objects = local_data_objects
            elif self.in_memory:
                data_objects = data_objects[0]
            if self.in_memory:
                self.data.append(data_objects)
            else:
                datapath = osp.join(
                    self.processed_dir,
                    f"dimer_dap2{split_name}_spec_{self.spec_type}_{self.filename_methods}_{idx // self.points_per_file}.pt",
                )
                if self.print_level >= 2:
                    print(f"Final Saving to {datapath}")
                    print(len(data_objects))
                torch.save(data_objects, datapath)
        return

    def len(self):
        if self.in_memory and self.prebatched:
            print((len(self.data) - 1) * len(self.data[0]), len(self.data[-1]))
            return (len(self.data) - 1) * len(self.data[0]) + len(self.data[-1])
        elif self.in_memory:
            return len(self.data)
        d = torch.load(
            osp.join(self.processed_dir, self.processed_file_names[-1]),
            weights_only=False,
        )
        if self.prebatched:
            return (
                len(self.processed_file_names) - 1
            ) * self.datapoint_storage_n_objects + len(d)
        return (
            len(self.processed_file_names) - 1
        ) * self.datapoint_storage_n_objects + len(d)

    def get(self, idx):
        idx_datapath = idx // self.datapoint_storage_n_objects
        obj_ind = idx % self.datapoint_storage_n_objects
        # issue is with 'train' split having only 3 data points each for some reason...
        # issue is with 'tets' split has 16
        # if self.active_idx_data == idx_datapath:
        #     return self.active_data[obj_ind]
        split_name = ""
        if self.spec_type in [1]:
            split_name = f"_{self.split}"
        datapath = os.path.join(
            self.processed_dir,
            f"dimer_dap2{split_name}_spec_{self.spec_type}_{self.filename_methods}_{idx_datapath}.pt",
        )
        self.active_data = torch.load(datapath, weights_only=False, map_location="cpu")
        self.active_data[obj_ind]
        try:
            self.active_data[obj_ind]
        except Exception:
            print(f"Error loading {datapath}\n  {idx=}, {idx_datapath=}, {obj_ind=}")
        # self.active_idx_data = idx_datapath
        return self.active_data[obj_ind]

    def get_in_memory(self, idx):
        """Method for retrieving data when in_memory=True"""
        if self.prebatched:
            idx_datapath = idx // self.datapoint_storage_n_objects
            obj_ind = idx % self.datapoint_storage_n_objects
            return self.data[idx_datapath][obj_ind]
        else:
            return self.data[idx]


class dapnet2_module_dataset_apnetStored(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=1,
        max_size=None,
        force_reprocess=True,
        skip_processed=True,
        # only need for processing
        atom_model_path=pretrained_atom_model_path,
        atom_model=None,
        apnet_model_path=pretrained_pairwise_model_path,
        apnet_model=None,
        batch_size=16,
        preprocessing_batch_size=256,
        prebatched=True,  # Note only operates as prebatched
        skip_compile=True,
        # DO NOT CHANGE UNLESS YOU WANT TO RE-PROCESS THE DATASET
        datapoint_storage_n_objects=1000,
        in_memory=False,
        num_devices=1,
        split="all",  # train, test
        print_level=2,
        m1="B3LYP-D3/aug-cc-pVTZ/CP",
        m2="CCSD(T)/CBS/CP",
        qcel_molecules: Optional[List[qcel.models.Molecule]] = None,
        energy_labels: Optional[List[float]] = None,
    ):
        """
        spec_type definitions:
            1. dapnet dataset for bfdb interaction energies
        """
        self.print_level = print_level
        try:
            assert spec_type in [1, 2, 8, None]
        except Exception:
            print("Currently spec_type must be 1 or 2 for SAPT0/jun-cc-pVDZ")
            raise ValueError
        self.spec_type = spec_type
        self.MAX_SIZE = max_size
        self.m1 = m1
        self.m2 = m2
        self.split = split
        self.r_cut = r_cut
        self.r_cut_im = r_cut_im
        self.force_reprocess = force_reprocess
        self.qcel_molecules = None
        self.energy_labels = None
        # Store qcel_molecules and energy_labels if provided
        if qcel_molecules is not None and energy_labels is not None:
            self.qcel_molecules = qcel_molecules
            self.energy_labels = energy_labels
            if len(qcel_molecules) != len(energy_labels):
                raise ValueError(
                    "Length of qcel_molecules and energy_labels must match"
                )
            print(
                f"Received {len(qcel_molecules)} QCElemental molecules with energy labels"
            )
        self.prebatched = True
        self.filename_methods = (
            clean_str_for_filename(m1) + "_to_" + clean_str_for_filename(m2)
        )
        self.datapoint_storage_n_objects = datapoint_storage_n_objects
        self.batch_size = batch_size
        self.training_batch_size = 1
        self.in_memory = in_memory
        self.skip_processed = skip_processed
        if os.path.exists(root) is False:
            os.makedirs(root, exist_ok=True)
        if atom_model is not None:
            self.atom_model = atom_model
            self.atom_model.model.to(self.atom_model.device)
            if not skip_compile:
                self.atom_model.compile_model()
        elif not self.skip_processed:
            if atom_model_path is None:
                atom_model_path = resolve_pretrained_path("am_ensemble/am_0.pt")
            self.atom_model = AtomModel(
                pre_trained_model_path=atom_model_path,
                ds_root=None,
                ignore_database_null=True,
            )
            self.atom_model.model.to(self.atom_model.device)
            if not skip_compile:
                self.atom_model.compile_model()

        if apnet_model is not None:
            self.ap_model = apnet_model
            self.ap_model.model.to(self.atom_model.device)
            if not skip_compile:
                self.ap_model.compile_model()
        elif not self.skip_processed:
            if apnet_model_path is None:
                apnet_model_path = resolve_pretrained_path("ap2_ensemble/ap2_0.pt")
            self.ap_model = APNet2Model(
                atom_model=self.atom_model.model,
                pre_trained_model_path=apnet_model_path,
                ds_root=None,
                ignore_database_null=True,
            )
            self.ap_model.model.return_hidden_states = True
            self.ap_model.model.to(self.atom_model.device)
            if not skip_compile:
                self.ap_model.compile_model()
        super(dapnet2_module_dataset_apnetStored, self).__init__(
            root, transform, pre_transform
        )
        if self.force_reprocess:
            self.force_reprocess = False
            super(dapnet2_module_dataset_apnetStored, self).__init__(
                root, transform, pre_transform
            )

        print(f"{self.root=}, {self.spec_type=}, {self.in_memory=}")
        if self.in_memory:
            self.get = self.get_in_memory
        # self.active_data = [None for i in self.processed_file_names]
        self.active_idx_data = None
        self.active_data = []
        self.process_m1_m2()
        targets_datapath = os.path.join(
            self.processed_dir, f"targets_{self.filename_methods}.pt"
        )
        if os.path.exists(targets_datapath) is False:
            print(f"Targets not found, processing targets...")
            self.process_m1_m2()
        else:
            self.target_data = torch.load(
                targets_datapath, weights_only=False, map_location="cpu"
            )

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed_delta")

    @property
    def raw_file_names(self):
        # TODO: enable users to specify data source via QCArchive, url, or local file
        # spec_1 = "spec_1" # 'SAPT0/jun-cc-pVDZ'
        if self.spec_type == 1:
            return [
                # 0.4 Train / 0.6 Test
                "3324_BFDBext_train_dimers.pkl",
                "3324_BFDBext_test_dimers.pkl",
            ]
        elif self.spec_type == 2:
            return [
                # "DES370K_b3lyp_255586.pkl",
                "DES370K_b3lyp_255586.pkl",
                # "DES370K_b3lyp_64.pkl",
            ]
        elif self.spec_type == 8:
            return [
                "t_val_19.pkl",
            ]
        else:
            return [
                "splinter_spec1.pkl",
            ]

    def reprocess_file_names(self):
        if self.force_reprocess:
            return ["file"]
        else:
            if self.split == "train":
                file_cmd = f"{self.processed_dir}/dimer_dap2_ap2_train_spec_{self.spec_type}*.pt"
            elif self.split == "test":
                file_cmd = f"{self.processed_dir}/dimer_dap2_ap2_test_spec_{self.spec_type}*.pt"
            else:
                file_cmd = (
                    f"{self.processed_dir}/dimer_dap2_ap2_spec_{self.spec_type}*.pt"
                )
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                spec_files.sort(key=natural_key)
                if self.MAX_SIZE is not None:
                    max_size = int(self.MAX_SIZE / self.datapoint_storage_n_objects)
                if self.MAX_SIZE is not None:
                    if len(spec_files) > max_size and max_size > 0:
                        spec_files = spec_files[:max_size]
                    elif len(spec_files) > max_size:
                        spec_files = spec_files[:1]
                return spec_files
            else:
                return ["dimer_missing.pt"]

    @property
    def processed_file_names(self):
        return self.reprocess_file_names()

    def download(self):
        if self.energy_labels and self.qcel_molecules:
            return
        raise RuntimeError(
            "Dataset does not have a download method. Development usage only at the moment."
        )
        return

    def process_m1_m2(self):
        target_data = []
        for raw_path in self.raw_paths:
            split_name = ""
            if self.spec_type in [1]:
                split_name = f"_{self.split}"
                print(f"{split_name=}")
                if self.split not in Path(raw_path).stem:
                    print(f"{self.split} is skipping {raw_path}")
                    continue
            print(f"raw_path: {raw_path}")
            print("Loading dimers...")
            if not self.in_memory:
                qcel_mols, targets = util.load_dimer_dataset(
                    raw_path,
                    self.MAX_SIZE,
                    return_qcel_mols=True,
                    return_qcel_mons=False,
                    columns=[self.m1, self.m2],
                )
                values = torch.tensor(
                    targets[:, 0] - targets[:, 1], dtype=torch.float32
                )
            else:
                values = torch.tensor(self.energy_labels, dtype=torch.float32)
                qcel_mols = self.qcel_molecules
            for i in range(
                0,
                len(qcel_mols) + len(qcel_mols) % self.batch_size + 1,
                self.batch_size,
            ):
                upper_bound = min(i + self.batch_size, len(qcel_mols))
                if len(qcel_mols[i:upper_bound]) == 0:
                    continue
                target_data.append(values[i:upper_bound])
        datapath = os.path.join(
            self.processed_dir, f"targets_{self.filename_methods}.pt"
        )
        if self.print_level >= 2:
            print(f"Saving to {datapath}")
        torch.save(target_data, datapath)
        self.target_data = target_data
        return

    def process(self):
        self.data = []
        idx = 0
        data_objects = []
        for raw_path in self.raw_paths:
            split_name = ""
            if self.spec_type in [1]:
                split_name = f"_{self.split}"
                print(f"{split_name=}")
                if self.split not in Path(raw_path).stem:
                    print(f"{self.split} is skipping {raw_path}")
                    continue
            print(f"raw_path: {raw_path}")
            print("Loading dimers...")
            if not self.in_memory:
                qcel_mols, targets = util.load_dimer_dataset(
                    raw_path,
                    self.MAX_SIZE,
                    return_qcel_mols=True,
                    return_qcel_mons=False,
                    columns=[],
                )
            else:
                qcel_mols = self.qcel_molecules
            print("Creating data objects...")
            print(f"{len(qcel_mols)=}, {self.batch_size=}")
            for i in range(
                0,
                len(qcel_mols) + len(qcel_mols) % self.batch_size + 1,
                self.batch_size,
            ):
                if self.skip_processed:
                    datapath = osp.join(
                        self.processed_dir,
                        f"dimer_dap2_ap2{split_name}_spec_{self.spec_type}_{idx // self.datapoint_storage_n_objects}.pt",
                    )
                    if osp.exists(datapath):
                        idx += self.batch_size
                        continue
                upper_bound = min(i + self.batch_size, len(qcel_mols))
                mols = qcel_mols[i:upper_bound]
                if len(mols) == 0:
                    continue
                predictions, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers = (
                    self.ap_model.predict_qcel_mols(
                        mols=mols,
                        batch_size=self.batch_size,
                        r_cut=self.ap_model.model.r_cut,
                        r_cut_im=self.ap_model.model.r_cut_im,
                    )
                )
                data = Data(
                    # no y data specified so that we can re-use the same general object for multiple datasets
                    h_AB=h_ABs[0],
                    h_BA=h_BAs[0],
                    cutoff=cutoffs[0],
                    dimer_ind=dimer_inds[0],
                    ndimer=ndimers[0],
                )
                data = data.cpu()
                data_objects.append(data)
                if len(data_objects) == self.datapoint_storage_n_objects:
                    if self.in_memory:
                        self.data.append(data_objects)
                    else:
                        idx_datapath = idx // self.datapoint_storage_n_objects
                        split_name = ""
                        if self.spec_type in [1]:
                            split_name = f"_{self.split}"
                        datapath = os.path.join(
                            self.processed_dir,
                            f"dimer_dap2_ap2{split_name}_spec_{self.spec_type}_{idx_datapath}.pt",
                        )
                        if self.print_level >= 2:
                            print(f"Saving to {datapath}")
                            print(len(data_objects))
                        torch.save(data_objects, datapath)
                    data_objects = []
                    if self.MAX_SIZE is not None and idx > self.MAX_SIZE:
                        break
                # idx += self.batch_size
                idx += 1
        if len(data_objects) > 0:
            if self.in_memory:
                self.data.append(data_objects)
            else:
                idx_datapath = idx // self.datapoint_storage_n_objects
                split_name = ""
                if self.spec_type in [1]:
                    split_name = f"_{self.split}"
                datapath = os.path.join(
                    self.processed_dir,
                    f"dimer_dap2_ap2{split_name}_spec_{self.spec_type}_{idx_datapath}.pt",
                )
                if self.print_level >= 2:
                    print(f"Saving to {datapath}")
                    print(len(data_objects))
                torch.save(data_objects, datapath)
        return

    def len(self):
        if self.in_memory and self.prebatched:
            return (len(self.data) - 1) * len(self.data[0]) + len(self.data[-1])
        elif self.in_memory:
            return len(self.data)
        d = torch.load(
            osp.join(self.processed_dir, self.processed_file_names[-1]),
            weights_only=False,
        )
        # NOTE: final incomplete batch size is counted as full
        return (
            len(self.processed_file_names) - 1
        ) * self.datapoint_storage_n_objects + len(d)

    def get(self, idx):
        idx_datapath = idx // self.datapoint_storage_n_objects
        obj_ind = idx % self.datapoint_storage_n_objects
        split_name = ""
        if self.spec_type in [1]:
            split_name = f"_{self.split}"
        datapath = os.path.join(
            self.processed_dir,
            f"dimer_dap2_ap2{split_name}_spec_{self.spec_type}_{idx_datapath}.pt",
        )
        local_data = torch.load(datapath, weights_only=False, map_location="cpu")
        try:
            local_data[obj_ind]
        except Exception:
            print(f"Error loading {datapath}\n  {idx=}, {idx_datapath=}, {obj_ind=}")
            raise ValueError
        try:
            self.target_data[obj_ind]
        except Exception:
            print(f"Error loading targets\n  {idx=}, {idx_datapath=}, {obj_ind=}")
            raise ValueError
        self.active_data = local_data
        self.active_data[obj_ind].y = self.target_data[idx]
        return self.active_data[obj_ind]

    def get_in_memory(self, idx):
        """Method for retrieving data when in_memory=True"""
        if self.prebatched:
            idx_datapath = idx // self.datapoint_storage_n_objects
            obj_ind = idx % self.datapoint_storage_n_objects
            self.data[idx_datapath][obj_ind].y = self.target_data[idx]
            return self.data[idx_datapath][obj_ind]
        else:
            self.data[idx].y = self.target_data[idx]
            return self.data[idx]
