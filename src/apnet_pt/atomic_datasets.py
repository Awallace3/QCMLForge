import os
import numpy as np
import pandas as pd
from typing import Any, List, Optional, Sequence, Union
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset
from torch_geometric.typing import TensorFrame, torch_frame
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping

from apnet_pt import constants

from torch_geometric.data import Data
from torch_geometric.data import Batch, Dataset
from . import util

import os.path as osp
import torch
from time import time
from qm_tools_aw import tools
import re
from . import multipole
from glob import glob

# from torch_geometric.data import download_url


def natural_key(text):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", text)]


def distance_matrix(r):
    v = np.sqrt(
        np.sum(np.square(r[:, np.newaxis, :] - r[np.newaxis, :, :]), axis=-1))
    return v


def distance_matrix_torch(r):
    v = torch.sqrt(torch.sum(torch.square(
        r[:, None, :] - r[None, :, :]), axis=-1))
    return v


def generate_monomer_multipole_dataset(file):
    monomers, cartesian_multipoles, _, _ = util.load_monomer_dataset(
        "mon200.pkl")
    return


def vec_func(R_ij, R_c=5.0, n_bessel=8):
    edge_feature_vector = np.zeros(
        (len(R_ij), len(R_ij), n_bessel), dtype=np.float32)
    edge_index = []
    for i in range(R_ij.shape[0]):
        for j in range(R_ij.shape[1]):
            if i != j and R_ij[i, j] < R_c:
                r_ij = R_ij[i, j]
                for n in range(n_bessel):
                    edge_feature_vector[i, j, n] = (
                        np.sqrt(2 / R_c) * np.sin(n *
                                                  np.pi * r_ij / R_c) / r_ij
                    )
                edge_index.append([i, j])
                # disagree with original apnet tf code here because we have bidirectional edges
                # edge_index.append([j, i])
    # if len(edge_index) == 0:
    #     edge_index = [[]]
    return edge_feature_vector, edge_index


def vec_func_index_only(R_ij, R_c=5.0):
    edge_index = []
    for i in range(R_ij.shape[0]):
        for j in range(i):
            if R_ij[j, i] < R_c:
                edge_index.append([j, i])
                edge_index.append([i, j])
    # for i in range(R_ij.shape[0]):
    #     for j in range(R_ij.shape[1]):
    #         if i != j and R_ij[i, j] < R_c:
    #             edge_index.append([i, j])
    #             # edge_index.append([j, i])
    return edge_index


def edge_function_system(R, r_c):
    dis_matrix = distance_matrix(R)
    edge_feature_vector, edge_index = vec_func(dis_matrix, R_c=r_c)
    return edge_index, edge_feature_vector


def edge_function_system_index_only(R, r_c):
    # dis_matrix = distance_matrix(R)
    dis_matrix = distance_matrix_torch(R)
    return vec_func_index_only(dis_matrix, R_c=r_c)


MAX_Z = 118  # largest atomic number


def atomic_collate_update_prebatched(batch):
    return batch[0]


def atomic_collate_update(batch):
    """
    Need to update the edge_index values so that each molecule has a unique
    set of indices. Then, the data.molecule_ind can be used to group
    atoms into molecules during a forward pass.
    """
    current_count = 0
    edge_indices = []
    # print('\nCollating')
    for i, data in enumerate(batch):
        # print(data.edge_index.shape)
        edge_indices.append(data.edge_index + current_count)
        data.molecule_ind = (
            torch.ones(data.molecule_ind.size(
                0), dtype=data.molecule_ind.dtype) * i
        )
        # data.molecule_ind.fill_(i)
        current_count += data.x.size(0)

    molecule_ind = torch.cat([data.molecule_ind for data in batch], dim=0)
    natom_per_mol = torch.bincount(molecule_ind)

    batched_data = Data(
        x=torch.cat([data.x for data in batch], dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        charges=torch.cat([data.charges for data in batch], dim=0),
        dipoles=torch.cat([data.dipoles for data in batch], dim=0),
        quadrupoles=torch.cat([data.quadrupoles for data in batch], dim=0),
        R=torch.cat([data.R for data in batch], dim=0),
        molecule_ind=molecule_ind,
        total_charge=torch.tensor(
            [data.total_charge for data in batch], dtype=batch[0].total_charge.dtype
        ),
        natom_per_mol=natom_per_mol,
    )
    return batched_data


def atomic_hirshfeld_collate_update(batch):
    """
    Need to update the edge_index values so that each molecule has a unique
    set of indices. Then, the data.molecule_ind can be used to group
    atoms into molecules during a forward pass.
    """
    current_count = 0
    edge_indices = []
    # print('\nCollating')
    for i, data in enumerate(batch):
        # print(data.edge_index.shape)
        edge_indices.append(data.edge_index + current_count)
        data.molecule_ind = (
            torch.ones(data.molecule_ind.size(
                0), dtype=data.molecule_ind.dtype) * i
        )
        # data.molecule_ind.fill_(i)
        current_count += data.x.size(0)

    molecule_ind = torch.cat([data.molecule_ind for data in batch], dim=0)
    natom_per_mol = torch.bincount(molecule_ind)

    batched_data = Data(
        x=torch.cat([data.x for data in batch], dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        charges=torch.cat([data.charges for data in batch], dim=0),
        dipoles=torch.cat([data.dipoles for data in batch], dim=0),
        quadrupoles=torch.cat([data.quadrupoles for data in batch], dim=0),
        R=torch.cat([data.R for data in batch], dim=0),
        molecule_ind=molecule_ind,
        total_charge=torch.tensor(
            [data.total_charge for data in batch], dtype=batch[0].total_charge.dtype
        ),
        natom_per_mol=natom_per_mol,
        volume_ratios=torch.cat([data.volume_ratios for data in batch], dim=0),
        valence_widths=torch.cat(
            [data.valence_widths for data in batch], dim=0),
    )
    return batched_data


def atomic_collate_update_no_target(batch):
    current_count = 0
    edge_indices = []
    # print('\nCollating')
    for i, data in enumerate(batch):
        edge_indices.append(data.edge_index + current_count)
        data.molecule_ind = (
            torch.ones(data.molecule_ind.size(
                0), dtype=data.molecule_ind.dtype) * i
        )
        # data.molecule_ind.fill_(i)
        current_count += data.x.size(0)

    molecule_ind = torch.cat([data.molecule_ind for data in batch], dim=0)
    natom_per_mol = torch.bincount(molecule_ind)

    batched_data = Data(
        x=torch.cat([data.x for data in batch], dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        R=torch.cat([data.R for data in batch], dim=0),
        molecule_ind=molecule_ind,
        total_charge=torch.tensor(
            [data.total_charge for data in batch], dtype=batch[0].total_charge.dtype
        ),
        natom_per_mol=natom_per_mol,
    )
    return batched_data


def atomic_pyg_to_qcel_mon(data):
    Z = data.x.numpy().astype(int)
    R = data.R.numpy()
    TQ = int(data.total_charge)
    qcel_mon = tools.convert_pos_carts_to_mol([Z], [R], charge=TQ)
    cartesian_multipoles = multipole.charge_dipole_qpoles_to_compact_multipoles(
        data.charges.numpy(), data.dipoles.numpy(), data.quadrupoles.numpy()
    )
    return qcel_mon, cartesian_multipoles


###############################
######   AtomicDataset   ######
###############################


class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, along="row")
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        if isinstance(self.dataset, OnDiskDataset):
            return self(self.dataset.multi_get(batch))
        return self(batch)


class AtomicDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        collate_fn=atomic_collate_update,
        # persistent_workers=False,
        **kwargs,
    ):
        if collate_fn is None:
            # Save for PyTorch Lightning < 1.6:
            self.follow_batch = follow_batch
            self.exclude_keys = exclude_keys

            self.collator_fn = Collater(dataset, follow_batch, exclude_keys)
            # self.collate_fn = self.collator.collate_fn
            # self.collate_fn = self.collator.collate_fn
        else:
            self.collate_fn = collate_fn

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            # persistent_workers=persistent_workers,
            **kwargs,
        )


def edges(R, r_cut):
    natom = np.shape(R)[0]
    RA = np.expand_dims(R, 0)
    RB = np.expand_dims(R, 1)
    RA = np.tile(RA, [natom, 1, 1])
    RB = np.tile(RB, [1, natom, 1])
    dist = np.linalg.norm(RA - RB, axis=2)
    mask = np.logical_and(dist < r_cut, dist > 0.0)
    edges = np.array(np.where(mask))  # dimensions [n_edge x 2]
    return edges


def qcel_mon_to_pyg_data(mon, r_cut=5.0, custom=False):
    Z = mon.atomic_numbers
    node_features = torch.tensor(np.array(Z), dtype=torch.int64)
    R = torch.tensor(np.array(mon.geometry) *
                     constants.au2ang, dtype=torch.float32)
    total_charge = torch.tensor(
        np.array(mon.molecular_charge), dtype=torch.int64)
    if custom:
        edge_index = edge_function_system_index_only(R, r_c=r_cut)
        edge_index = torch.tensor(np.array(edge_index)).t().long()
    else:
        edge_index = torch.tensor(edges(R, r_cut)).long()
    data = Data(
        x=node_features.long(),
        edge_index=edge_index.long(),
        R=R.float(),
        molecule_ind=torch.tensor(np.full(len(R), 0), dtype=torch.int64),
        total_charge=total_charge.long(),
        natom_per_mol=torch.tensor([len(R)], dtype=torch.int64),
    )
    return data


def create_atomic_data(
    Z,
    R,
    total_charge,
    cartesian_multipoles=None,
    r_cut=5.0,
    idx=None,
    edge_index_only=True,
    custom=False,
):
    node_features = np.array(Z, dtype=np.int64)
    node_features = torch.tensor(node_features)
    if isinstance(R, np.ndarray):
        R = torch.tensor(R, dtype=torch.float32)
    torch_total_charge = torch.tensor(total_charge, dtype=torch.int32)
    if custom:
        if edge_index_only:
            edge_index = edge_function_system_index_only(R, r_cut)
        else:
            edge_index, edge_feature_vector = edge_function_system(R, r_cut)
            edge_feature_vector = torch.tensor(edge_feature_vector).view(-1, 8)
        edge_index = torch.tensor(edge_index).t()
    else:
        edge_index = torch.tensor(edges(R, r_cut)).long()
    if idx is None:
        idx = 0
    if cartesian_multipoles is not None:
        return Data(
            x=node_features,
            edge_index=edge_index.long(),
            y=torch.tensor(cartesian_multipoles, dtype=torch.float32),
            R=R.float(),
            molecule_ind=torch.tensor(np.full(len(R), idx)),
            total_charge=torch_total_charge,
        )
    return Data(
        x=node_features,
        edge_index=edge_index.long(),
        R=R.float(),
        molecule_ind=torch.tensor(np.full(len(R), idx)),
        total_charge=torch_total_charge,
    )


class atomic_module_dataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=1,
        split="all",  # train, test
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=1,
    ):
        """ """
        try:
            assert spec_type in [1, 2, 3, 4, 6, 9, 10]
        except Exception:
            print(
                "Currently spec_type must be 1, 2, or 3 for HF/jun-cc-pV(D+d)Z (CMPNN), PBE0/aug-cc-pV(T+D)Z (CMPNN), or HF/jun-cc-pV(D+D)Z (APNET2) respectively. Only 1 and 2 are available for download at the moment."
            )
            raise ValueError
        self.testing = testing
        self.split = split
        if self.testing and max_size is None:
            self.MAX_SIZE = 200
        else:
            self.MAX_SIZE = max_size
        self.spec_type = spec_type
        self.force_reprocess = force_reprocess

        self.in_memory = in_memory
        if os.path.exists(root) is False:
            os.makedirs(root)

        if self.force_reprocess:
            file_cmd = f"{root}/processed/data_spec_{self.spec_type}_*.pt"
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                if self.force_reprocess:
                    self.force_reprocess = False
                    for i in spec_files:
                        os.remove(f"{root}/processed/{i}")

        super(atomic_module_dataset, self).__init__(
            root, transform, pre_transform)
        print(
            f"{self.root = }, {self.spec_type = }, {self.testing = }, {self.in_memory = }"
        )
        if self.in_memory:
            print("Loading data into memory")
            t = time()
            self.data = []
            for i in self.processed_file_names:
                self.data.append(
                    torch.load(osp.join(self.processed_dir, i),
                               weights_only=False)
                )
            total_time_seconds = int(time() - t)
            print(f"Loaded in {total_time_seconds:4d} seconds")
            self.get = self.get_in_memory
        self.batch_size = batch_size

    @property
    def raw_file_names(self):
        # TODO: enable users to specify data source via QCArchive, url, or local file

        # spec_1 = "spec_1" # 'hf/jun-cc-pv_dpd_z' CMPNN
        # spec_2 = "spec_2" # 'pbe0/aug-cc-pv_tpd_z' CMPNN
        # spec_3 = "spec_3" # 'hf/jun-cc-pv_dpd_z' APNET2
        # spec_4 = "spec_4" # 'pbe0/aug-cc-pvtz' APNET2
        if self.testing:
            return [
                "testing.pkl",
            ]
        else:
            if self.spec_type == 1 or self.spec_type == 2:
                return [
                    f"monomers_cmpnn_spec_{self.spec_type}.pkl",
                ]
            elif self.spec_type == 3:
                return [
                    f"monomers_apnet2_spec_{self.spec_type}.pkl",
                ]
            elif self.spec_type == 4:
                return [
                    "monomers_ap3_spec_1_pbe0.pkl",
                ]
            elif self.spec_type == 6:
                return [
                    "monomers_apnet2_spec_3_62.pkl",
                ]
            elif self.spec_type == 9:
                print('Using spec_type 9 for AP3 PBE0/aug-cc-pVDZ (with Hirshfeld volumes and widths')
                return [
                    "monomers_ap3_spec_5_pbe0.pkl",
                ]
            elif self.spec_type == 10:
                return [
                    f"monomers_ap3_spec_{self.spec_type}_HF.pkl",
                ]
        raise ValueError("spec_type must be 1, 2, or 3!")
        return []

    @property
    def processed_file_names(self):
        if self.force_reprocess:
            return ["file"]
        if self.testing:
            return [f"data_{i}.pt" for i in range(self.MAX_SIZE - 1)]
        else:
            if self.split == "train":
                file_cmd = f"{self.root}/processed/data_train_spec_{self.spec_type}_*.pt"
            elif self.split == "test":
                file_cmd = f"{self.root}/processed/data_test_spec_{self.spec_type}_*.pt"
            else:
                file_cmd = f"{self.root}/processed/data_spec_{self.spec_type}_*.pt"
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                # want to preserve idx ordering
                spec_files.sort(key=natural_key)
                if self.MAX_SIZE is not None and len(spec_files) > self.MAX_SIZE:
                    spec_files = spec_files[: self.MAX_SIZE]
                return spec_files
            else:
                return [f"data_missing_{i}.pt" for i in range(1)]

    def download(self):
        if self.spec_type in [1, 2]:
            import qcportal as ptl
            from tqdm import tqdm

            client = ptl.PortalClient("https://ml.qcarchive.molssi.org:443")
            ds = client.get_dataset("singlepoint", "StockholderMultipoles")
            cnt = 0
            data = {
                "id": [],
                "Z": [],
                "R": [],
                "cartesian_multipoles": [],
                "entry_name": [],
                "spec_name": [],
                "TQ": [],
                "molecular_multiplicity": [],
            }
            print("Downloading data from QCArchive")
            for entry_name, spec_name, record in tqdm(
                ds.iterate_records(status="complete",
                                   specification_names="spec_1")
            ):
                record_dict = record.dict()
                qcvars = record_dict["properties"]
                charges = qcvars["mbis charges"]
                dipoles = qcvars["mbis dipoles"]
                quadrupoles = qcvars["mbis quadrupoles"]
                level_of_theory = f"{record_dict['specification']['method']}/{record_dict['specification']['basis']}"

                n = len(charges)

                charges = np.reshape(charges, (n, 1))
                dipoles = np.reshape(dipoles, (n, 3))
                quad = np.reshape(quadrupoles, (n, 3, 3))

                quad = [q[np.triu_indices(3)] for q in quad]
                quadrupoles = np.array(quad)
                multipoles = np.concatenate(
                    [charges, dipoles, quadrupoles], axis=1)

                data["id"].append(cnt)
                data["Z"].append(record.molecule.atomic_numbers)
                data["R"].append(record.molecule.geometry * constants.au2ang)
                data["cartesian_multipoles"].append(multipoles)
                data["entry_name"].append(entry_name)
                data["spec_name"].append(spec_name)
                data["TQ"].append(int(record.molecule.molecular_charge))
                data["molecular_multiplicity"].append(
                    record.molecule.molecular_multiplicity
                )
                cnt += 1
            df = pd.DataFrame(data, index=data["id"])
            df1 = df[df["spec_name"] == "spec_1"]
            if os.path.exists(f"{self.root}/raw") is False:
                os.makedirs(f"{self.root}/raw")
            if os.path.exists(f"{self.root}/processed") is False:
                os.makedirs(f"{self.root}/processed")
            df1.to_pickle(f"{self.root}/raw/monomers_cmpnn_spec_1.pkl")
            df2 = df[df["spec_name"] == "spec_2"]
            assert len(df2) > 0
            df2.to_pickle(f"{self.root}/raw/monomers_cmpnn_spec_2.pkl")
            return
        else:
            raise ValueError("spec_type must be 1 or 2 for current downloads!")

    def process(self, r_cut=5.0, edge_index_only=True):
        idx = 0
        for raw_path in self.raw_paths:
            split_name = ""
            if self.spec_type in [7]:
                split_name = f"_{self.split}" if self.split != 'all' else ""
                print(f"{split_name=}")
            print(f"raw_path: {raw_path}")
            # converting to qcel monomer to crudely validate structure
            monomers, cartesian_multipoles, total_charge = util.load_monomer_dataset(
                raw_path, self.MAX_SIZE
            )
            t = time()
            for i in range(len(monomers)):
                if i % 1000 == 0:
                    print(f"{i}/{len(monomers)}, took {time() - t} seconds")
                    t = time()
                mol = monomers[i]
                data = qcel_mon_to_pyg_data(mol, r_cut=r_cut)
                cart_mult = np.array(
                    [j for j in cartesian_multipoles[i] if not np.all(j == 0)]
                )
                data.charges = torch.tensor(
                    cart_mult[:, 0], dtype=torch.float32)
                data.dipoles = torch.tensor(
                    cart_mult[:, 1:4], dtype=torch.float32)
                data.quadrupoles = torch.tensor(
                    multipole.make_quad_np(cart_mult[:, 4:]), dtype=torch.float32
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                if self.testing:
                    torch.save(data, osp.join(
                        self.processed_dir, f"data_{idx}.pt"))
                else:
                    torch.save(
                        data,
                        osp.join(
                            self.processed_dir, f"data{split_name}_spec_{self.spec_type}_{idx}.pt"
                        ),
                    )
                if self.MAX_SIZE is not None and idx > self.MAX_SIZE:
                    break
                idx += 1
        return

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.testing:
            return torch.load(
                osp.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False
            )
        else:
            split_name = ""
            if self.spec_type in [7]:
                split_name = f"_{self.split}" if self.split != 'all' else ""
            return torch.load(
                osp.join(self.processed_dir,
                         f"data{split_name}_spec_{self.spec_type}_{idx}.pt"),
                weights_only=False,
            )
        return

    def get_in_memory(self, idx):
        return self.data[idx]

    def train_test_loaders(self):
        indices = np.random.permutation(len(self))
        split = int(0.9 * len(self))
        train_indices = indices[:split]
        test_indices = indices[split:]
        return (
            AtomicDataLoader(
                self[train_indices],
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=atomic_collate_update,
            ),
            AtomicDataLoader(
                self[test_indices],
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=atomic_collate_update,
            ),
        )


class atomic_hirshfeld_module_dataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=1,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=1,
    ):
        try:
            assert spec_type in [1, 5, 10]
        except Exception:
            print(
                "Currently spec_type must be 1 for pbe0/aug-cc-pVDZ (APNET2) respectively. spec_type 5 is for testing. No downloads are available at the moment."
            )
            raise ValueError
        self.batch_size = batch_size
        self.testing = testing
        if self.testing and max_size is None:
            self.MAX_SIZE = 200
        else:
            self.MAX_SIZE = max_size
        self.spec_type = spec_type
        self.force_reprocess = force_reprocess
        self.root = root
        self.in_memory = in_memory
        if os.path.exists(root) is False:
            os.makedirs(root)
        print(
            f"{self.root = }, {self.spec_type = }, {self.testing = }, {self.in_memory = }"
        )
        super(atomic_hirshfeld_module_dataset, self).__init__(
            root, transform, pre_transform
        )
        if self.in_memory:
            print("Loading data into memory")
            t = time()
            self.data = []
            for i in self.processed_file_names:
                self.data.append(
                    torch.load(osp.join(self.processed_dir, i),
                               weights_only=False)
                )
            total_time_seconds = int(time() - t)
            print(f"Loaded in {total_time_seconds:4d} seconds")
            self.get = self.get_in_memory

    @property
    def raw_file_names(self):
        # spec_3 = "spec_3" # 'hf/jun-cc-pv_dpd_z' APNET2
        if self.spec_type in [1, 5]:
            print(
                f"monomers_ap3_spec_{self.spec_type}_pbe0.pkl",
                # "monomers_ap3_spec_1_pbe0_62.pkl",
            )
            return [
                f"monomers_ap3_spec_{self.spec_type}_pbe0.pkl",
                # "monomers_ap3_spec_1_pbe0_62.pkl",
            ]
        elif self.spec_type in [10]:
            return [
                f"monomers_ap3_spec_{self.spec_type}_HF.pkl",
            ]
        raise ValueError("spec_type must in [1, 5, 10]!")
        return []

    @property
    def processed_file_names(self):
        if self.force_reprocess:
            return ["file"]
        else:
            file_cmd = f"{self.root}/processed/monomer_ap3_{self.spec_type}_*.pt"
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                # want to preserve idx ordering
                spec_files.sort(key=natural_key)
                if self.MAX_SIZE is not None and len(spec_files) > self.MAX_SIZE:
                    spec_files = spec_files[: self.MAX_SIZE]
                return spec_files
            else:
                return [f"data_missing_{i}.pt" for i in range(1)]

    def download(self):
        print(self.raw_file_names)
        raise ValueError("Downloads are not available!")

    def process(self, r_cut=5.0, edge_index_only=True):
        idx = 0
        print(dir(self))
        batch_size = self.batch_size
        if self.spec_type in [1, 5]:
            print(
                f"ENSURE THAT {batch_size=} is the same as the batch size used in the AtomHirshfeldModel training! This mode avoids collating completely."
            )
        for raw_path in self.raw_paths:
            print(f"raw_path: {raw_path}")
            # converting to qcel monomer to crudely validate structure
            (
                monomers,
                cartesian_multipoles,
                total_charge,
                volume_ratios,
                valence_widths,
            ) = util.load_monomer_dataset(raw_path, self.MAX_SIZE, hirshfeld_props=True)
            t = time()
            for i in range(0, len(monomers), batch_size):
                batched_data = []
                upper_bound = min(i + batch_size, len(monomers))
                for j in range(i, upper_bound):
                    mol = monomers[i]
                    data = qcel_mon_to_pyg_data(mol, r_cut=r_cut)
                    cart_mult = np.array(
                        [j for j in cartesian_multipoles[i]
                            if not np.all(j == 0)]
                    )
                    data.charges = torch.tensor(
                        cart_mult[:, 0], dtype=torch.float32)
                    data.dipoles = torch.tensor(
                        cart_mult[:, 1:4], dtype=torch.float32)
                    data.quadrupoles = torch.tensor(
                        multipole.make_quad_np(cart_mult[:, 4:]), dtype=torch.float32
                    )
                    data.volume_ratios = torch.tensor(
                        volume_ratios[i], dtype=torch.float32
                    )
                    data.valence_widths = torch.tensor(
                        valence_widths[i], dtype=torch.float32
                    )
                    batched_data.append(data)
                batch = atomic_hirshfeld_collate_update(batched_data)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(
                    batch,
                    osp.join(
                        self.processed_dir,
                        f"monomer_ap3_{self.spec_type}_{idx}.pt",
                    ),
                    )
                if self.MAX_SIZE is not None and idx > self.MAX_SIZE:
                    break
                idx += 1
        return

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(
            osp.join(self.processed_dir,
                     f"monomer_ap3_{self.spec_type}_{idx}.pt"),
            weights_only=False,
        )

    def get_in_memory(self, idx):
        return self.data[idx]

    def train_test_loaders(self):
        indices = np.random.permutation(len(self))
        split = int(0.9 * len(self))
        train_indices = indices[:split]
        test_indices = indices[split:]
        return (
            AtomicDataLoader(
                self[train_indices],
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=atomic_hirshfeld_collate_update,
            ),
            AtomicDataLoader(
                self[test_indices],
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=atomic_hirshfeld_collate_update,
            ),
        )
