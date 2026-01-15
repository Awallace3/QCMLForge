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
import lmdb
import json

# from torch_geometric.data import download_url


def qcel_monomer_to_atomic_data(monomer, r_cut=5.0, **kwargs):
    return create_atomic_data(
        monomer.atomic_numbers,
        monomer.geometry * constants.au2ang,
        monomer.molecular_charge,
        r_cut=r_cut,
        **kwargs,
    )


def natural_key(text):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", text)]


def distance_matrix(r):
    v = np.sqrt(np.sum(np.square(r[:, np.newaxis, :] - r[np.newaxis, :, :]), axis=-1))
    return v


def distance_matrix_torch(r):
    v = torch.sqrt(torch.sum(torch.square(r[:, None, :] - r[None, :, :]), axis=-1))
    return v


def generate_monomer_multipole_dataset(file):
    monomers, cartesian_multipoles, _, _ = util.load_monomer_dataset("mon200.pkl")
    return


def vec_func(R_ij, R_c=5.0, n_bessel=8):
    edge_feature_vector = np.zeros((len(R_ij), len(R_ij), n_bessel), dtype=np.float32)
    edge_index = []
    for i in range(R_ij.shape[0]):
        for j in range(R_ij.shape[1]):
            if i != j and R_ij[i, j] < R_c:
                r_ij = R_ij[i, j]
                for n in range(n_bessel):
                    edge_feature_vector[i, j, n] = (
                        np.sqrt(2 / R_c) * np.sin(n * np.pi * r_ij / R_c) / r_ij
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
    Batch a list of per-molecule PyG Data objects into a single Data with reindexed edges and molecule indices.
    
    Constructs a batched Data by concatenating atom-level tensors (x, charges, dipoles, quadrupoles, R), reindexing each molecule's `edge_index` (and `edge_index_full` if present) so indices are unique across the batch, and producing `molecule_ind` and `natom_per_mol`. Also aggregates per-molecule `total_charge` into a tensor.
    
    Parameters:
        batch (list[Data]): List of PyG Data objects, one per molecule. Each item is expected to contain atom-level fields (`x`, `R`, `charges`, `dipoles`, `quadrupoles`), `edge_index`, `molecule_ind`, and `total_charge`. If present, `edge_index_full` will be propagated and reindexed.
    
    Returns:
        Data: A single PyG Data object containing concatenated atom features, reindexed `edge_index` (and optionally `edge_index_full`), `molecule_ind`, `natom_per_mol`, and aggregated `total_charge`.
    """
    current_count = 0
    edge_indices = []
    edge_indices_full = []
    has_full_edges = hasattr(batch[0], "edge_index_full")

    # print('\nCollating')
    for i, data in enumerate(batch):
        # print(data.edge_index.shape)
        edge_indices.append(data.edge_index + current_count)
        if has_full_edges:
            edge_indices_full.append(data.edge_index_full + current_count)
        data.molecule_ind = (
            torch.ones(data.molecule_ind.size(0), dtype=data.molecule_ind.dtype) * i
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

    if has_full_edges:
        batched_data.edge_index_full = torch.cat(edge_indices_full, dim=1)

    return batched_data


def atomic_hfvr_vw_collate_update(batch):
    """
    Batch collate for Data objects that include Hirshfeld volume ratios and valence widths.
    
    Parameters:
        batch (list of Data): List of per-molecule PyG `Data` objects containing at least
            `x`, `R`, `edge_index`, `molecule_ind`, `total_charge`, `volume_ratios`, and
            `valence_widths`. Each item may optionally include `edge_index_full`.
    
    Returns:
        Data: A single batched `Data` object with per-atom tensors concatenated, `edge_index`
        (and `edge_index_full` when present) reindexed so atom indices are unique across the
        batch, `molecule_ind` set to identify molecule membership, `natom_per_mol` as a
        per-molecule atom count, and `total_charge` collected per molecule.
    """
    current_count = 0
    edge_indices = []
    edge_indices_full = []
    has_full_edges = hasattr(batch[0], "edge_index_full")

    # print('\nCollating')
    for i, data in enumerate(batch):
        # print(data.edge_index.shape)
        edge_indices.append(data.edge_index + current_count)
        if has_full_edges:
            edge_indices_full.append(data.edge_index_full + current_count)
        data.molecule_ind = (
            torch.ones(data.molecule_ind.size(0), dtype=data.molecule_ind.dtype) * i
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
        volume_ratios=torch.cat([data.volume_ratios for data in batch], dim=0),
        valence_widths=torch.cat([data.valence_widths for data in batch], dim=0),
    )

    if has_full_edges:
        batched_data.edge_index_full = torch.cat(edge_indices_full, dim=1)

    return batched_data


def atomic_hirshfeld_collate_update(batch):
    """
    Batch a list of PyG Data objects representing molecules into a single Data with reindexed edges and per-molecule bookkeeping.
    
    Parameters:
        batch (list[torch_geometric.data.Data]): List of per-molecule Data objects. Each item must contain at minimum `x`, `edge_index`, `R`, `molecule_ind`, `total_charge`, `charges`, `dipoles`, `quadrupoles`, `volume_ratios`, and `valence_widths`. If present, `edge_index_full` will be included in the output.
    
    Returns:
        torch_geometric.data.Data: A single Data object where:
          - atom features (`x`, `R`, `charges`, `dipoles`, `quadrupoles`, `volume_ratios`, `valence_widths`) are concatenated along the atom dimension,
          - `edge_index` (and `edge_index_full` if present) are shifted so every molecule's atom indices are unique and then concatenated,
          - `molecule_ind` indicates each atom's molecule index,
          - `natom_per_mol` gives the atom counts per molecule,
          - `total_charge` is a tensor of per-molecule total charges.
    """
    current_count = 0
    edge_indices = []
    edge_indices_full = []
    has_full_edges = hasattr(batch[0], "edge_index_full")

    # print('\nCollating')
    for i, data in enumerate(batch):
        # print(data.edge_index.shape)
        edge_indices.append(data.edge_index + current_count)
        if has_full_edges:
            edge_indices_full.append(data.edge_index_full + current_count)
        data.molecule_ind = (
            torch.ones(data.molecule_ind.size(0), dtype=data.molecule_ind.dtype) * i
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
        valence_widths=torch.cat([data.valence_widths for data in batch], dim=0),
    )

    if has_full_edges:
        batched_data.edge_index_full = torch.cat(edge_indices_full, dim=1)

    return batched_data


def atomic_collate_update_no_target(batch):
    """
    Collate a list of PyG Data objects (no target properties) into a single batched Data for model input.
    
    Concatenates per-atom features and coordinates, reindexes and stacks short-range edges, computes per-molecule indices and atom counts, and collects total molecular charges. If any input item contains `edge_index_full`, the corresponding full all-pairs edge indices are concatenated and attached to the result as `edge_index_full`.
    
    Parameters:
        batch (Sequence[torch_geometric.data.Data]): Sequence of Data objects to batch. Each item is expected to contain `x`, `edge_index`, `R`, `molecule_ind`, and `total_charge`. `edge_index_full` is optional.
    
    Returns:
        torch_geometric.data.Data: Batched Data with fields:
            - x: concatenated atom features for all molecules.
            - edge_index: concatenated and reindexed short-range edge indices.
            - R: concatenated atomic coordinates.
            - molecule_ind: per-atom tensor mapping each atom to its molecule index in the batch.
            - total_charge: 1-D tensor of each molecule's total charge.
            - natom_per_mol: 1-D tensor of atom counts per molecule.
            - edge_index_full (optional): concatenated and reindexed full all-pairs edge indices if present in inputs.
    """
    current_count = 0
    edge_indices = []
    edge_indices_full = []
    has_full_edges = hasattr(batch[0], "edge_index_full")

    # print('\nCollating')
    for i, data in enumerate(batch):
        edge_indices.append(data.edge_index + current_count)
        if has_full_edges:
            edge_indices_full.append(data.edge_index_full + current_count)
        data.molecule_ind = (
            torch.ones(data.molecule_ind.size(0), dtype=data.molecule_ind.dtype) * i
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

    if has_full_edges:
        batched_data.edge_index_full = torch.cat(edge_indices_full, dim=1)

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
        """
        Initialize an AtomicDataLoader configured for atomic datasets with a flexible collate strategy.
        
        Parameters:
            dataset (Union[Dataset, Sequence[BaseData], DatasetAdapter]):
                Source dataset or sequence of Data objects. If an OnDiskDataset is provided, it will be replaced with a range iterator over its length to enable worker-safe indexing.
            batch_size (int): Number of samples per batch.
            shuffle (bool): If True, shuffle the data every epoch.
            follow_batch (Optional[List[str]]): Keys to include in the generated `follow_batch` mapping when using the builtin collator; ignored if a custom `collate_fn` is provided.
            exclude_keys (Optional[List[str]]): Keys to exclude from collation when using the builtin collator; ignored if a custom `collate_fn` is provided.
            collate_fn (callable): Collation function to aggregate samples into a batch. If None, a Collater using `follow_batch` and `exclude_keys` is constructed; otherwise the provided function is used.
            **kwargs: Forwarded to the base DataLoader constructor.
        """
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


def edges(R, r_cut, full_indices=False):
    """
    Compute pairwise atom index lists for short-range edges and optionally all non-self pairs.
    
    Short-range edges include ordered index pairs (i, j) for which the Euclidean distance between atoms i and j is greater than 0 and less than r_cut. When requested, also return all ordered non-self index pairs.
    
    Parameters:
        R (array-like): Atomic positions, shape (N, 3).
        r_cut (float): Cutoff distance for short-range edges.
        full_indices (bool): If True, also return full all-pairs (non-self) indices.
    
    Returns:
        edges_sr (ndarray): Short-range edge indices with shape [2, n_edges]; each column is (i, j).
        edges_full (ndarray, optional): All ordered non-self pair indices with shape [2, n_edges_full]; only returned if full_indices is True.
    """
    natom = np.shape(R)[0]
    RA = np.expand_dims(R, 0)
    RB = np.expand_dims(R, 1)
    RA = np.tile(RA, [natom, 1, 1])
    RB = np.tile(RB, [1, natom, 1])
    dist = np.linalg.norm(RA - RB, axis=2)
    mask_sr = np.logical_and(dist < r_cut, dist > 0.0)
    edges_sr = np.array(np.where(mask_sr))  # dimensions [2, n_edge]

    if full_indices:
        mask_full = dist > 0.0  # All pairs except self
        edges_full = np.array(np.where(mask_full))
        return edges_sr, edges_full
    return edges_sr


def qcel_mon_to_pyg_data(mon, r_cut=5.0, custom=False, full_indices=False):
    """
    Convert a QCElemental monomer into a PyTorch Geometric Data object representing the atomic graph.
    
    Parameters:
        mon: QCElemental monomer-like object containing `atomic_numbers`, `geometry`, and `molecular_charge`.
        r_cut (float): Distance cutoff in angstroms used to build short-range edges.
        custom (bool): If True, use the custom edge construction routine (`edge_function_system_index_only`) instead of the default cutoff-based edges.
        full_indices (bool): If True and `custom` is False, also compute and attach `edge_index_full` containing all pairwise (excluding self) atom index pairs.
    
    Returns:
        Data: A torch_geometric.data.Data instance with the following fields:
            - x: tensor of atomic numbers (int64).
            - R: tensor of Cartesian coordinates in angstroms (float32).
            - edge_index: short-range edge index tensor.
            - molecule_ind: tensor mapping atoms to molecule index (all zeros).
            - total_charge: tensor with the molecule total charge.
            - natom_per_mol: tensor containing the number of atoms in the monomer.
            - edge_index_full (optional): full all-pairs edge index tensor when `full_indices=True`.
    """
    Z = mon.atomic_numbers
    node_features = torch.tensor(np.array(Z), dtype=torch.int64)
    R = torch.tensor(np.array(mon.geometry) * constants.au2ang, dtype=torch.float32)
    total_charge = torch.tensor(np.array(mon.molecular_charge), dtype=torch.int64)

    edge_index_full = None
    if custom:
        edge_index = edge_function_system_index_only(R, r_c=r_cut)
        edge_index = torch.tensor(np.array(edge_index)).t().long()
    else:
        if full_indices:
            edge_index_sr, edge_index_full = edges(R, r_cut, full_indices=True)
            edge_index = torch.tensor(edge_index_sr).long()
            edge_index_full = torch.tensor(edge_index_full).long()
        else:
            edge_index = torch.tensor(edges(R, r_cut)).long()

    data_dict = {
        "x": node_features.long(),
        "edge_index": edge_index.long(),
        "R": R.float(),
        "molecule_ind": torch.tensor(np.full(len(R), 0), dtype=torch.int64),
        "total_charge": total_charge.long(),
        "natom_per_mol": torch.tensor([len(R)], dtype=torch.int64),
    }

    if edge_index_full is not None:
        data_dict["edge_index_full"] = edge_index_full

    return Data(**data_dict)


def create_atomic_data(
    Z,
    R,
    total_charge,
    cartesian_multipoles=None,
    r_cut=5.0,
    idx=None,
    edge_index_only=True,
    custom=False,
    full_indices=False,
):
    """
    Construct a PyG Data object representing an isolated molecule from atomic numbers, coordinates, and total charge.
    
    Parameters:
        Z (sequence[int]): Atomic numbers for each atom.
        R (array-like[N, 3] or torch.Tensor[N, 3]): Cartesian coordinates in ångströms.
        total_charge (int): Total molecular charge.
        cartesian_multipoles (array-like, optional): Per-atom cartesian multipoles to store as `y`.
        r_cut (float, optional): Short-range cutoff (angstroms) used to build short-range edges.
        idx (int, optional): Molecule index used to populate `molecule_ind` for each atom.
        edge_index_only (bool, optional): When True and `custom` is True, compute only edge indices (no edge features).
        custom (bool, optional): When True, use the custom edge construction routines (`edge_function_system*`) instead of `edges`.
        full_indices (bool, optional): When True and `custom` is False, compute and attach both short-range (`edge_index`) and all-pairs (`edge_index_full`) edge indices.
    
    Returns:
        torch_geometric.data.Data: A Data object containing at minimum:
            - x: atomic numbers as a tensor
            - edge_index: short-range edge index tensor
            - R: atomic positions tensor
            - molecule_ind: per-atom molecule index tensor
            - total_charge: total charge tensor
          Additionally, when provided:
            - y: cartesian multipoles tensor (if `cartesian_multipoles` supplied)
            - edge_index_full: all-pairs edge index tensor (if `full_indices` is True)
    """
    node_features = np.array(Z, dtype=np.int64)
    node_features = torch.tensor(node_features)
    if isinstance(R, np.ndarray):
        R = torch.tensor(R, dtype=torch.float32)
    torch_total_charge = torch.tensor(total_charge, dtype=torch.int32)

    edge_index_full = None
    if custom:
        if edge_index_only:
            edge_index = edge_function_system_index_only(R, r_cut)
        else:
            edge_index, edge_feature_vector = edge_function_system(R, r_cut)
            edge_feature_vector = torch.tensor(edge_feature_vector).view(-1, 8)
        edge_index = torch.tensor(edge_index).t()
    else:
        if full_indices:
            edge_index_sr, edge_index_full = edges(R, r_cut, full_indices=True)
            edge_index = torch.tensor(edge_index_sr).long()
            edge_index_full = torch.tensor(edge_index_full).long()
        else:
            edge_index = torch.tensor(edges(R, r_cut)).long()

    if idx is None:
        idx = 0

    data_dict = {
        "x": node_features,
        "edge_index": edge_index.long(),
        "R": R.float(),
        "molecule_ind": torch.tensor(np.full(len(R), idx)),
        "total_charge": torch_total_charge,
    }

    if edge_index_full is not None:
        data_dict["edge_index_full"] = edge_index_full

    if cartesian_multipoles is not None:
        data_dict["y"] = torch.tensor(cartesian_multipoles, dtype=torch.float32)

    return Data(**data_dict)


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
        """
        Initialize an atomic_module_dataset for converting and loading processed monomer data.
        
        Parameters:
        	root (str): Path to dataset root directory; processed data is expected under root/processed.
        	transform (callable, optional): Transform applied on-the-fly to examples.
        	pre_transform (callable, optional): Transform applied during processing before saving.
        	r_cut (float): Radial cutoff (angstroms) used for edge construction.
        	testing (bool): If True, enable testing mode which sets a default MAX_SIZE.
        	spec_type (int): Dataset specification variant; must be one of [1, 2, 3, 4, 6, 9, 10, 11, 12].
        	split (str): Which split to expose ("all", "train", or "test").
        	max_size (int|None): Maximum number of examples to load/consider; if None and testing is True, defaults to 200.
        	force_reprocess (bool): If True, remove existing processed files for this spec_type before processing.
        	in_memory (bool): If True, load all processed Data objects into memory and make get() return them directly.
        	batch_size (int): Default batch size used by helper loader utilities.
        
        Notes:
        	- The constructor validates spec_type and raises ValueError for unsupported values.
        	- When force_reprocess is True, existing processed files for the spec_type are deleted from root/processed.
        	- When in_memory is True, processed files are loaded into memory and self.get is set to self.get_in_memory.
        """
        try:
            assert spec_type in [1, 2, 3, 4, 6, 9, 10, 11, 12]
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

        super(atomic_module_dataset, self).__init__(root, transform, pre_transform)
        print(
            f"{self.root = }, {self.spec_type = }, {self.testing = }, {self.in_memory = }"
        )
        if self.in_memory:
            print("Loading data into memory")
            t = time()
            self.data = []
            for i in self.processed_file_names:
                self.data.append(
                    torch.load(osp.join(self.processed_dir, i), weights_only=False)
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
        """
        Return the list of expected raw data filenames for this dataset configuration.
        
        The returned filenames depend on the instance's `spec_type` and `testing` flag:
        - If `testing` is True, returns ["testing.pkl"].
        - For `spec_type` 1 or 2, returns ["monomers_cmpnn_spec_{spec_type}.pkl"].
        - For `spec_type` 3, returns ["monomers_apnet2_spec_3.pkl"].
        - For `spec_type` 4, returns ["monomers_ap3_spec_1_pbe0.pkl"].
        - For `spec_type` 6, returns ["monomers_apnet2_spec_3_62.pkl"].
        - For `spec_type` 9, returns ["monomers_ap3_spec_5_pbe0.pkl"].
        - For `spec_type` 10, returns ["monomers_ap3_spec_10_HF.pkl"].
        - For `spec_type` 11 or 12, returns ["SPICE_monomer_spec_{spec_type}.pkl"].
        
        Returns:
            list[str]: A list containing one filename (string) to be present in `raw_dir`.
        
        Raises:
            ValueError: If `spec_type` is not one of the supported values.
        """
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
                print(
                    "Using spec_type 9 for AP3 PBE0/aug-cc-pVDZ (with Hirshfeld volumes and widths"
                )
                return [
                    "monomers_ap3_spec_5_pbe0.pkl",
                ]
            elif self.spec_type == 10:
                return [
                    f"monomers_ap3_spec_{self.spec_type}_HF.pkl",
                ]
            elif self.spec_type in [11, 12]:
                return [
                    f"SPICE_monomer_spec_{self.spec_type}.pkl",
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
                file_cmd = (
                    f"{self.root}/processed/data_train_spec_{self.spec_type}_*.pt"
                )
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
                ds.iterate_records(status="complete", specification_names="spec_1")
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
                multipoles = np.concatenate([charges, dipoles, quadrupoles], axis=1)

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
        """
        Process raw monomer files into PyG Data objects and save the processed items to disk.
        
        Loads monomers from each raw path, converts each monomer into a PyG Data object (including atomic positions, atomic numbers, and cartesian multipoles converted to charges, dipoles, and quadrupoles), applies the optional `pre_filter` and `pre_transform`, and writes each processed item to the dataset's processed directory.
        
        Parameters:
            r_cut (float): Distance cutoff passed when constructing edge information for each Data object.
            edge_index_only (bool): If True, only short-range edge indices are intended to be retained; if False, full pairwise edge indices may be retained.
        
        Side effects:
            Writes processed Data objects to files under `self.processed_dir`. File names are:
              - "data_{idx}.pt" when `self.testing` is True
              - "data{_split}_spec_{spec_type}_{idx}.pt" otherwise (where `_split` is empty or `_{self.split}`)
        
        Returns:
            None
        """
        idx = 0
        for raw_path in self.raw_paths:
            split_name = ""
            if self.spec_type in [7]:
                split_name = f"_{self.split}" if self.split != "all" else ""
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
                data = qcel_mon_to_pyg_data(mol, r_cut=r_cut, full_indices=True)
                cart_mult = np.array(
                    [j for j in cartesian_multipoles[i] if not np.all(j == 0)]
                )
                data.charges = torch.tensor(cart_mult[:, 0], dtype=torch.float32)
                data.dipoles = torch.tensor(cart_mult[:, 1:4], dtype=torch.float32)
                data.quadrupoles = torch.tensor(
                    multipole.make_quad_np(cart_mult[:, 4:]), dtype=torch.float32
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                if self.testing:
                    torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
                else:
                    torch.save(
                        data,
                        osp.join(
                            self.processed_dir,
                            f"data{split_name}_spec_{self.spec_type}_{idx}.pt",
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
                split_name = f"_{self.split}" if self.split != "all" else ""
            return torch.load(
                osp.join(
                    self.processed_dir,
                    f"data{split_name}_spec_{self.spec_type}_{idx}.pt",
                ),
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
        """
        Initialize the atomic_hirshfeld_module_dataset.
        
        Parameters:
            root (str): Path to the dataset root directory; created if missing.
            transform (callable, optional): Transformation applied on-the-fly to each example.
            pre_transform (callable, optional): Transformation applied once before saving processed files.
            r_cut (float): Radial cutoff (in Å) used for edge construction.
            testing (bool): If True, enables testing mode which sets a default MAX_SIZE when max_size is None.
            spec_type (int): Dataset specification selector. Allowed values: 1, 5, 10, 11, 12.
                - 1: standard PBE0/aug-cc-pVDZ APNET2 set
                - 5,10,11,12: Hirshfeld-related variants (no automatic download available)
            max_size (int or None): Maximum number of processed items to consider; if None and testing is True,
                a default of 200 is used.
            force_reprocess (bool): If True, existing processed files may be reprocessed/overwritten.
            in_memory (bool): If True, loads all processed items into memory and replaces `get` with `get_in_memory`.
            batch_size (int): Default batch size used by helper loader constructors.
        
        Notes:
            - Raises ValueError if spec_type is not one of the allowed integers.
            - When in_memory is True, processed files under the processed directory are loaded with torch.load.
        """
        try:
            assert spec_type in [1, 5, 10, 11, 12]
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
                    torch.load(osp.join(self.processed_dir, i), weights_only=False)
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
            for i in range(len(monomers)):
                if i % 1000 == 0:
                    print(f"{i}/{len(monomers)}, took {time() - t} seconds")
                    t = time()
                mol = monomers[i]
                data = qcel_mon_to_pyg_data(mol, r_cut=r_cut)
                cart_mult = np.array(
                    [j for j in cartesian_multipoles[i] if not np.all(j == 0)]
                )
                data.charges = torch.tensor(cart_mult[:, 0], dtype=torch.float32)
                data.dipoles = torch.tensor(cart_mult[:, 1:4], dtype=torch.float32)
                data.quadrupoles = torch.tensor(
                    multipole.make_quad_np(cart_mult[:, 4:]), dtype=torch.float32
                )
                if np.isnan(volume_ratios[i]).any():
                    print(f"NaN in volume ratios for index {i}, skipping")
                    continue
                data.volume_ratios = torch.tensor(volume_ratios[i], dtype=torch.float32)
                data.valence_widths = torch.tensor(
                    valence_widths[i], dtype=torch.float32
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(
                    data,
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
            osp.join(self.processed_dir, f"monomer_ap3_{self.spec_type}_{idx}.pt"),
            weights_only=False,
        )

    def get_in_memory(self, idx):
        return self.data[idx]

    def train_test_loaders(self):
        """
        Create paired train and test DataLoaders from this dataset using a random 90/10 split.
        
        The dataset is randomly permuted before splitting; the training DataLoader shuffles batches and the test DataLoader does not. Both loaders use this instance's batch_size and atomic_hirshfeld_collate_update as the collate function.
        
        Returns:
            tuple: (train_loader, test_loader) where each element is an AtomicDataLoader over the respective split.
        """
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


class atomic_induced_dipole_precomputed_dataset(Dataset):
    """
    Dataset that pre-computes hirshfeld volume ratios and valence widths
    using an AtomTypeParamMPNN model during processing, storing them
    alongside multipole moments for efficient induced dipole training.

    This avoids the need to run atomtype_hfvr_model forward pass during training,
    significantly speeding up training by computing these values once during
    dataset processing.
    """

    def __init__(
        self,
        root,
        atomtype_hfvr_model,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=9,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=1,
    ):
        """
        Create a dataset that precomputes Hirshfeld volume ratios (hfvr) and valence widths (vw) using a provided AtomTypeHFVR model.
        
        This initializer stores and configures the provided model (sets eval mode and disables gradients), validates that the chosen spec_type supports Hirshfeld properties, prepares dataset paths, and optionally loads already-processed items into memory for fast access.
        
        Parameters:
            root (str): Root directory for the dataset (processed/raw files are stored under this path).
            atomtype_hfvr_model (torch.nn.Module): Pretrained model used to compute hfvr and vw during processing.
            transform (callable, optional): Transform applied on data when accessed.
            pre_transform (callable, optional): Transform applied to data during processing.
            r_cut (float, optional): Radial cutoff (Å) used when constructing edges and neighborhood features.
            testing (bool, optional): If True, a smaller default MAX_SIZE is used unless max_size is explicitly provided.
            spec_type (int, optional): Dataset specification type; must be one of 5, 9, 10, 11, or 12 (Hirshfeld-capable specs).
            max_size (int | None, optional): Maximum number of items to process/use; when None and testing is True, a small default is applied.
            force_reprocess (bool, optional): If True, previously processed files will be re-generated during processing.
            in_memory (bool, optional): If True, load processed items into memory and make get() return them directly.
            batch_size (int, optional): Default batch size used by helper loader constructors.
        """
        # Validate spec_type supports Hirshfeld properties
        try:
            assert spec_type in [5, 9, 10, 11, 12]
        except Exception:
            print(
                "spec_type must be 5, 9, or 10 for datasets with Hirshfeld properties."
            )
            raise ValueError

        # Store model for processing
        self.atomtype_hfvr_model = atomtype_hfvr_model
        self.atomtype_hfvr_model.eval()  # Set to eval mode
        self.atomtype_hfvr_model.requires_grad_(False)  # Disable gradients

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
        self.r_cut = r_cut

        if os.path.exists(root) is False:
            os.makedirs(root)

        print(
            f"atomic_induced_dipole_precomputed_dataset: {self.root = }, {self.spec_type = }, {self.testing = }, {self.in_memory = }"
        )

        super(atomic_induced_dipole_precomputed_dataset, self).__init__(
            root, transform, pre_transform
        )

        # After processing, reset force_reprocess so we can properly list files
        if self.force_reprocess:
            self.force_reprocess = False

        if self.in_memory:
            print("Loading pre-computed data into memory")
            t = time()
            self.data = []
            for i in self.processed_file_names:
                self.data.append(
                    torch.load(osp.join(self.processed_dir, i), weights_only=False)
                )
            total_time_seconds = int(time() - t)
            print(f"Loaded in {total_time_seconds:4d} seconds")
            self.get = self.get_in_memory

    @property
    def raw_file_names(self):
        """
        Return the list of expected raw filenames for this dataset based on its spec_type.
        
        Maps spec_type to dataset-specific raw file names:
        - 5 -> "monomers_ap3_spec_5_pbe0.pkl"
        - 9 -> "monomers_ap3_spec_5_pbe0.pkl" (spec_type 9 reuses spec 5 data)
        - 10 -> "monomers_ap3_spec_10_HF.pkl"
        - 11, 12 -> "SPICE_monomer_spec_{spec_type}.pkl"
        
        Returns:
            list[str]: One-element list containing the raw filename required for processing.
        
        Raises:
            ValueError: If spec_type is not one of 5, 9, 10, 11, or 12.
        """
        if self.spec_type in [5]:
            return [f"monomers_ap3_spec_{self.spec_type}_pbe0.pkl"]
        elif self.spec_type in [9]:
            # spec_type 9 uses spec_5 data
            return ["monomers_ap3_spec_5_pbe0.pkl"]
        elif self.spec_type in [10]:
            return [f"monomers_ap3_spec_{self.spec_type}_HF.pkl"]
        elif self.spec_type in [11, 12]:
            return [
                f"SPICE_monomer_spec_{self.spec_type}.pkl",
            ]
        raise ValueError("spec_type must be 5, 9, 10, 11!")

    @property
    def processed_file_names(self):
        """
        Return the list of processed filenames available for this dataset, honoring force_reprocess and MAX_SIZE.
        
        If force_reprocess is True, a sentinel list ["file"] is returned to trigger reprocessing. Otherwise the method searches the processed directory for files matching the dataset's spec type pattern, sorts them naturally, and trims the list to MAX_SIZE if set. If no matching files are found, a list with placeholder names ("data_missing_0.pt", ...) is returned.
        
        Returns:
            list[str]: Filenames (basename only) of processed data or sentinel/placeholder names.
        """
        if self.force_reprocess:
            return ["file"]
        else:
            file_cmd = f"{self.root}/processed/monomer_induced_dipole_precomputed_{self.spec_type}_*.pt"
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                spec_files.sort(key=natural_key)
                if self.MAX_SIZE is not None and len(spec_files) > self.MAX_SIZE:
                    spec_files = spec_files[: self.MAX_SIZE]
                return spec_files
            else:
                return [f"data_missing_{i}.pt" for i in range(1)]

    def download(self):
        """
        Signal that automatic downloads are not supported for this dataset.
        
        Raises:
            ValueError: Always raised to indicate downloads are unavailable.
        """
        print(self.raw_file_names)
        raise ValueError("Downloads are not available!")

    def process(self, r_cut=5.0, edge_index_only=True):
        """
        Precompute Hirshfeld volume ratios (hfvr) and valence widths (vw) for raw monomer data and save processed PyG Data objects with these properties.
        
        Processes each raw monomer file reachable via self.raw_paths: converts monomers to PyG Data (requesting full edge indices), attaches cartesian multipole targets (charges, dipoles, quadrupoles), computes hfvr and vw using self.atomtype_hfvr_model (model run with gradients disabled), optionally validates computed hfvr against raw values, applies self.pre_filter and self.pre_transform when present, and writes each resulting Data object to self.processed_dir with a monotonic index-based filename.
        
        Parameters:
            r_cut (float): Radial cutoff passed to qcel_mon_to_pyg_data when constructing the PyG Data object.
            edge_index_only (bool): Present for API compatibility but ignored by this method; full edge indices are requested during data construction.
        
        Returns:
            None
        """
        idx = 0
        for raw_path in self.raw_paths:
            print(f"Processing raw_path: {raw_path}")
            print(f"Pre-computing hfvr and vw using atomtype_hfvr_model...")

            # Load data with Hirshfeld properties (for validation/comparison)
            (
                monomers,
                cartesian_multipoles,
                total_charge,
                volume_ratios_raw,
                valence_widths_raw,
            ) = util.load_monomer_dataset(raw_path, self.MAX_SIZE, hirshfeld_props=True)

            t = time()
            for i in range(len(monomers)):
                if i % 100 == 0:
                    print(f"{i}/{len(monomers)}, took {time() - t:.2f} seconds")
                    t = time()

                mol = monomers[i]
                data = qcel_mon_to_pyg_data(mol, r_cut=r_cut, full_indices=True)

                # Store multipoles (targets for training)
                cart_mult = np.array(
                    [j for j in cartesian_multipoles[i] if not np.all(j == 0)]
                )
                data.charges = torch.tensor(cart_mult[:, 0], dtype=torch.float32)
                data.dipoles = torch.tensor(cart_mult[:, 1:4], dtype=torch.float32)
                data.quadrupoles = torch.tensor(
                    multipole.make_quad_np(cart_mult[:, 4:]), dtype=torch.float32
                )

                # PRE-COMPUTE hfvr and vw using the model
                with torch.no_grad():
                    Ks = self.atomtype_hfvr_model(data)  # [n_atoms, 2]
                    data.volume_ratios = Ks[:, 0].clone()  # hfvr
                    data.valence_widths = Ks[:, 1].clone()  # vw

                # Optional: Validate against raw values if available
                if not np.isnan(volume_ratios_raw[i]).any():
                    raw_vr = torch.tensor(volume_ratios_raw[i], dtype=torch.float32)
                    computed_vr = data.volume_ratios
                    if len(raw_vr) == len(computed_vr):
                        max_diff = torch.abs(raw_vr - computed_vr).max().item()
                        if max_diff > 0.1 and i % 100 == 0:
                            print(
                                f"  Note: Max difference between raw and computed hfvr: {max_diff:.4f}"
                            )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(
                    data,
                    osp.join(
                        self.processed_dir,
                        f"monomer_induced_dipole_precomputed_{self.spec_type}_{idx}.pt",
                    ),
                )

                if self.MAX_SIZE is not None and idx >= self.MAX_SIZE:
                    break
                idx += 1

        print(f"Finished processing {idx} molecules with pre-computed hfvr/vw")
        return

    def len(self):
        """
        Return the number of processed files available in the dataset.
        
        Returns:
            count (int): The number of entries in `processed_file_names`.
        """
        return len(self.processed_file_names)

    def get(self, idx):
        """
        Load the processed PyG Data object for a given item index.
        
        Parameters:
            idx (int): Index of the processed item to load.
        
        Returns:
            data (torch_geometric.data.Data): The processed Data object saved for the specified index.
        """
        return torch.load(
            osp.join(
                self.processed_dir,
                f"monomer_induced_dipole_precomputed_{self.spec_type}_{idx}.pt",
            ),
            weights_only=False,
        )

    def get_in_memory(self, idx):
        """
        Return the preloaded Data object for the given index from the in-memory cache.
        
        Parameters:
            idx (int): Index of the item to retrieve.
        
        Returns:
            data (Data): The cached PyG `Data` object stored at the specified index.
        """
        return self.data[idx]

    def train_test_loaders(self):
        """
        Create paired train and test DataLoaders from this dataset using a random 90/10 split.
        
        The dataset is randomly permuted before splitting; the training DataLoader shuffles batches and the test DataLoader does not. Both loaders use this instance's batch_size and atomic_hirshfeld_collate_update as the collate function.
        
        Returns:
            tuple: (train_loader, test_loader) where each element is an AtomicDataLoader over the respective split.
        """
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


class atomic_module_dataset_lmdb(Dataset):
    """
    LMDB-based dataset for atomic induced dipole training with efficient storage.

    This dataset uses LMDB (Lightning Memory-Mapped Database) for efficient
    storage and retrieval of processed atomic data, with worker-safe initialization
    and LRU caching for performance.
    """

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=9,
        max_size=None,
        force_reprocess=False,
        in_memory=False,
        batch_size=1,
        lmdb_map_size=1099511627776,
        lmdb_readonly=False,
        cache_size=1000,
        atomtype_hfvr_model=None,
    ):
        """
        LMDB-backed atomic dataset that stores and serves preprocessed PyG Data objects with optional in-memory caching and optional precomputation via an AtomType HFVR model.
        
        Parameters:
            root (str): Root directory for dataset storage and LMDB files.
            transform (callable, optional): Transform applied on each Data object on access.
            pre_transform (callable, optional): Transform applied to each Data object during processing.
            r_cut (float): Distance cutoff (Å) used for edge construction.
            testing (bool): If True, use a reduced default MAX_SIZE for faster tests.
            spec_type (int): Specification type selecting raw/processed file sets; must be one of [5, 9, 10, 11, 12].
            max_size (int, optional): Maximum number of molecules to process/load; None means no explicit limit.
            force_reprocess (bool): If True, re-run processing even if processed data exists.
            in_memory (bool): If True, load entire dataset into memory and make get() return from memory.
            batch_size (int): Default batch size used by helper loader factories.
            lmdb_map_size (int): Maximum LMDB map size in bytes (default 1 TB).
            lmdb_readonly (bool): If True, open the LMDB environment in read-only mode.
            cache_size (int): Number of recently accessed items to keep in an LRU-style in-process cache.
            atomtype_hfvr_model (torch.nn.Module, optional): Optional pretrained model used during processing to compute Hirshfeld volume ratios and valence widths; if provided it will be switched to eval() and gradients will be disabled.
        
        Notes:
            - Initializes and opens an LMDB environment under the provided root and prepares internal caches.
            - If in_memory is True, all items are loaded at construction and subsequent get() calls use the in-memory store.
            - Raises ValueError when spec_type is not in the allowed set.
        """
        try:
            assert spec_type in [5, 9, 10, 11, 12]
        except Exception:
            print(
                "spec_type must be 5, 9, or 10 for datasets with Hirshfeld properties."
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
        self.r_cut = r_cut

        # LMDB settings
        self.lmdb_map_size = lmdb_map_size
        self.lmdb_readonly = lmdb_readonly
        self.cache_size = cache_size
        self._cache = {}
        self._cache_keys = []

        # LMDB state
        self.lmdb_env = None
        self.lmdb_path = None
        self._length = None
        self._worker_id = None

        # Optional model for pre-computation
        self.atomtype_hfvr_model = atomtype_hfvr_model
        if self.atomtype_hfvr_model is not None:
            self.atomtype_hfvr_model.eval()
            self.atomtype_hfvr_model.requires_grad_(False)

        if os.path.exists(root) is False:
            os.makedirs(root, exist_ok=True)

        self._init_lmdb_path(root)
        self._init_lmdb()

        print(
            f"atomic_module_dataset_lmdb: {self.root = }, {self.spec_type = }, "
            f"{self.testing = }, {self.in_memory = }, {self.lmdb_path = }"
        )

        super(atomic_module_dataset_lmdb, self).__init__(root, transform, pre_transform)

        # Handle force_reprocess: close LMDB, re-init parent, reopen LMDB
        if self.force_reprocess:
            self.force_reprocess = False
            self._close_lmdb()
            super(atomic_module_dataset_lmdb, self).__init__(
                root, transform, pre_transform
            )
            self._init_lmdb()

        if self.in_memory:
            print("Loading LMDB data into memory...")
            t = time()
            self.data = []
            for i in range(len(self)):
                self.data.append(self.get(i))
            total_time_seconds = int(time() - t)
            print(f"Loaded {len(self.data)} items in {total_time_seconds:4d} seconds")
            self.get = self.get_in_memory

    def _init_lmdb_path(self, root):
        """
        Set the LMDB storage path for this dataset under the processed directory.
        
        Parameters:
            root (str or os.PathLike): Base dataset root directory.
        
        Effect:
            Creates and assigns `self.lmdb_path` as
            "<root>/processed/lmdb_atomic_induced_dipole_spec_{self.spec_type}".
        """
        self.lmdb_path = osp.join(
            root, "processed", f"lmdb_atomic_induced_dipole_spec_{self.spec_type}"
        )

    def _init_lmdb(self):
        """
        Initialize and open the LMDB environment for this dataset instance.
        
        Creates the lmdb_path directory if it does not exist, opens an LMDB environment using
        the instance's configuration attributes (e.g., lmdb_map_size, lmdb_readonly), and
        loads stored metadata (the "__metadata__" key) to set self._length. On failure, sets
        self.lmdb_env to None and self._length to 0 and prints an error message.
        """
        if not osp.exists(self.lmdb_path):
            os.makedirs(self.lmdb_path, exist_ok=True)

        try:
            self.lmdb_env = lmdb.open(
                self.lmdb_path,
                map_size=self.lmdb_map_size,
                readonly=self.lmdb_readonly,
                max_dbs=0,
                lock=not self.lmdb_readonly,
                max_readers=256,
            )

            # Read metadata
            with self.lmdb_env.begin() as txn:
                metadata_bytes = txn.get(b"__metadata__")
                if metadata_bytes:
                    metadata = json.loads(metadata_bytes.decode("utf-8"))
                    self._length = metadata.get("length", 0)
                else:
                    self._length = 0
        except Exception as e:
            print(f"Error initializing LMDB: {e}")
            self.lmdb_env = None
            self._length = 0

    def _close_lmdb(self):
        """
        Close and release the LMDB environment used by the dataset.
        
        If an LMDB environment is open, it is closed and the internal reference is cleared; calling this when no environment exists does nothing.
        """
        if self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

    def __del__(self):
        """
        Close the LMDB environment and release associated resources when the object is deleted.
        
        This method attempts to close any open LMDB handles; any exceptions raised during cleanup are suppressed.
        """
        try:
            self._close_lmdb()
        except:
            pass

    def __getstate__(self):
        """Prepare object for pickling by closing LMDB"""
        state = self.__dict__.copy()
        # Close LMDB environment before pickling
        if "lmdb_env" in state and state["lmdb_env"] is not None:
            try:
                state["lmdb_env"].close()
            except:
                pass
        # Remove unpicklable objects
        state["lmdb_env"] = None
        state["_cache"] = {}
        state["_cache_keys"] = []
        state["_worker_id"] = None
        return state

    def __setstate__(self, state):
        """
        Restore the dataset after unpickling and reinitialize its LMDB environment.
        
        Updates the instance dictionary from the unpickled state and re-opens the LMDB
        environment so the object is ready for use in the current process.
        """
        self.__dict__.update(state)
        # Reinitialize LMDB in the new process
        self._init_lmdb()

    @property
    def raw_file_names(self):
        """
        Map the dataset's spec_type to the expected raw filename(s).
        
        Returns:
            list[str]: A list containing the raw filename(s) required for the dataset. Mapping:
                - spec_type 5  -> "monomers_ap3_spec_5_pbe0.pkl"
                - spec_type 9  -> "monomers_ap3_spec_5_pbe0.pkl"
                - spec_type 10 -> "monomers_ap3_spec_10_HF.pkl"
                - spec_type 11 or 12 -> "SPICE_monomer_spec_<spec_type>.pkl"
        
        Raises:
            ValueError: If spec_type is not one of 5, 9, 10, 11, or 12.
        """
        if self.spec_type in [5]:
            return [f"monomers_ap3_spec_{self.spec_type}_pbe0.pkl"]
        elif self.spec_type in [9]:
            return ["monomers_ap3_spec_5_pbe0.pkl"]
        elif self.spec_type in [10]:
            return [f"monomers_ap3_spec_{self.spec_type}_HF.pkl"]
        elif self.spec_type in [11, 12]:
            return [f"SPICE_monomer_spec_{self.spec_type}.pkl"]
        raise ValueError("spec_type must be 5, 9, 10, or 11!")

    @property
    def processed_file_names(self):
        """
        Determine the processed file name list by checking for an LMDB database with valid metadata.
        
        If `self.force_reprocess` is True returns ["file"]. If no LMDB path is set or the LMDB is missing or invalid returns ["lmdb_missing"]. If an LMDB exists and its `__metadata__` contains a positive `length`, returns ["lmdb_atomic_induced_dipole_spec_{spec_type}"] where `{spec_type}` is taken from the instance.
        
        Returns:
            list: A single-item list with one of:
                - "file" when forcing reprocess,
                - "lmdb_missing" when no usable LMDB is found,
                - "lmdb_atomic_induced_dipole_spec_{spec_type}" when LMDB metadata reports length > 0.
        """
        if self.force_reprocess:
            return ["file"]

        if not hasattr(self, "lmdb_path") or self.lmdb_path is None:
            return ["lmdb_missing"]

        if osp.exists(self.lmdb_path):
            env = None
            try:
                env = lmdb.open(
                    self.lmdb_path,
                    readonly=True,
                    lock=False,
                    max_dbs=0,
                    create=False,
                    max_readers=256,
                )
                with env.begin() as txn:
                    metadata_bytes = txn.get(b"__metadata__")
                    if metadata_bytes:
                        metadata = json.loads(metadata_bytes.decode("utf-8"))
                        length = metadata.get("length", 0)

                        if length > 0:
                            return [f"lmdb_atomic_induced_dipole_spec_{self.spec_type}"]
            except Exception as e:
                print(f"Error checking LMDB: {e}")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except:
                        pass

        return ["lmdb_missing"]

    def download(self):
        """
        Prevent dataset downloads; this dataset does not support remote downloading.
        
        Raises:
            ValueError: Always raised to indicate downloads are not available.
        """
        print(self.raw_file_names)
        raise ValueError("Downloads are not available!")

    def _store_to_lmdb(self, data_objects, start_idx):
        """
        Store a sequence of PyG Data objects into the LMDB environment and update dataset metadata.
        
        Each object in `data_objects` is serialized and written under a sequential integer key starting at `start_idx`. After writing all entries, the LMDB `__metadata__` record is updated to reflect the new total length and to persist dataset-level parameters (`r_cut` and `spec_type`). The instance attribute `_length` is also updated.
        
        Parameters:
            data_objects (Iterable): Sequence of data objects to store (will be pickled).
            start_idx (int): Integer index at which to start writing keys in LMDB.
        
        Raises:
            RuntimeError: If the LMDB environment (`self.lmdb_env`) is not initialized.
        """
        import pickle

        if self.lmdb_env is None:
            raise RuntimeError("LMDB environment not initialized")

        with self.lmdb_env.begin(write=True) as txn:
            for i, data_obj in enumerate(data_objects):
                idx = start_idx + i
                key = str(idx).encode("utf-8")
                value = pickle.dumps(data_obj)
                txn.put(key, value)

            # Update metadata
            metadata = {
                "length": start_idx + len(data_objects),
                "r_cut": self.r_cut,
                "spec_type": self.spec_type,
            }
            txn.put(b"__metadata__", json.dumps(metadata).encode("utf-8"))

        self._length = start_idx + len(data_objects)

    def process(self, r_cut=5.0, edge_index_only=True):
        """
        Process raw monomer files, convert them to PyG Data objects with Hirshfeld properties, and store the results in the dataset's LMDB.
        
        Each monomer is converted to a Data object (including charges, dipoles, quadrupoles, and full-edge indices). If an AtomTypeHFVR model was provided to the dataset, volume_ratios and valence_widths are computed with that model; otherwise they are loaded from the raw source. Entries failing pre_filter or containing NaNs in required Hirshfeld data are skipped. Processed items are written to LMDB in batches and processing stops when MAX_SIZE is reached (if set).
        
        Parameters:
            r_cut (float): Distance cutoff used when converting monomers to PyG Data (passed to qcel_mon_to_pyg_data).
            edge_index_only (bool): Present for API compatibility; this method always produces full edge indices for stored data.
        """
        idx = 0
        data_objects = []
        batch_size_lmdb = 100  # Store in batches for efficiency

        for raw_path in self.raw_paths:
            print(f"Processing raw_path: {raw_path}")

            # Load data with Hirshfeld properties
            (
                monomers,
                cartesian_multipoles,
                total_charge,
                volume_ratios_raw,
                valence_widths_raw,
            ) = util.load_monomer_dataset(raw_path, self.MAX_SIZE, hirshfeld_props=True)

            t = time()
            for i in range(len(monomers)):
                if i % 100 == 0:
                    print(f"{i}/{len(monomers)}, took {time() - t:.2f} seconds")
                    t = time()

                mol = monomers[i]
                data = qcel_mon_to_pyg_data(mol, r_cut=r_cut, full_indices=True)

                # Store multipoles
                cart_mult = np.array(
                    [j for j in cartesian_multipoles[i] if not np.all(j == 0)]
                )
                data.charges = torch.tensor(cart_mult[:, 0], dtype=torch.float32)
                data.dipoles = torch.tensor(cart_mult[:, 1:4], dtype=torch.float32)
                data.quadrupoles = torch.tensor(
                    multipole.make_quad_np(cart_mult[:, 4:]), dtype=torch.float32
                )

                # Compute or load volume_ratios and valence_widths
                if self.atomtype_hfvr_model is not None:
                    # Pre-compute using model
                    with torch.no_grad():
                        Ks = self.atomtype_hfvr_model(data)
                        data.volume_ratios = Ks[:, 0].clone()
                        data.valence_widths = Ks[:, 1].clone()
                else:
                    # Load from raw data
                    if np.isnan(volume_ratios_raw[i]).any():
                        print(f"NaN in volume ratios for index {i}, skipping")
                        continue
                    data.volume_ratios = torch.tensor(
                        volume_ratios_raw[i], dtype=torch.float32
                    )
                    data.valence_widths = torch.tensor(
                        valence_widths_raw[i], dtype=torch.float32
                    )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_objects.append(data.cpu())

                # Store in batches
                if len(data_objects) >= batch_size_lmdb:
                    start_idx = idx - len(data_objects) + 1
                    self._store_to_lmdb(data_objects, start_idx)
                    data_objects = []

                if self.MAX_SIZE is not None and idx >= self.MAX_SIZE:
                    break
                idx += 1

        # Store remaining objects
        if len(data_objects) > 0:
            start_idx = idx - len(data_objects)
            self._store_to_lmdb(data_objects, start_idx)

        print(f"Finished processing {idx} molecules to LMDB")
        return

    def len(self):
        """
        Get the number of items stored in the LMDB-backed dataset.
        
        Returns:
            length (int): Number of entries recorded in LMDB metadata; returns 0 if the LMDB environment is not initialized or metadata is missing.
        """
        if self._length is not None:
            return self._length

        if self.lmdb_env is None:
            return 0

        with self.lmdb_env.begin() as txn:
            metadata_bytes = txn.get(b"__metadata__")
            if metadata_bytes:
                metadata = json.loads(metadata_bytes.decode("utf-8"))
                self._length = metadata.get("length", 0)
            else:
                self._length = 0

        return self._length

    def _check_worker_init(self):
        """Ensure LMDB env is initialized for current worker process"""
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
        else:
            worker_id = None

        # Reinitialize LMDB if worker changed
        if worker_id != self._worker_id:
            if self.lmdb_env is not None:
                self._close_lmdb()

            self._worker_id = worker_id
            self._init_lmdb()
            self._cache = {}
            self._cache_keys = []

    def get(self, idx):
        """
        Retrieve a dataset item by index from the LMDB store and update the local LRU cache.
        
        Parameters:
            idx (int): Index of the item to retrieve.
        
        Returns:
            data: The unpickled data object stored at the given index.
        
        Raises:
            RuntimeError: If the LMDB environment is not initialized.
            IndexError: If the index is not present in the LMDB database.
        """
        import pickle

        self._check_worker_init()

        # Check cache first
        if idx in self._cache:
            self._cache_keys.remove(idx)
            self._cache_keys.append(idx)
            return self._cache[idx]

        if self.lmdb_env is None:
            raise RuntimeError("LMDB environment not initialized")

        # Load from LMDB
        with self.lmdb_env.begin() as txn:
            key = str(idx).encode("utf-8")
            value_bytes = txn.get(key)

            if value_bytes is None:
                raise IndexError(f"Index {idx} not found in LMDB database")

            data = pickle.loads(value_bytes)

        # Update cache
        self._cache[idx] = data
        self._cache_keys.append(idx)

        # Evict oldest if cache full
        if len(self._cache) > self.cache_size:
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]

        return data

    def get_in_memory(self, idx):
        """
        Retrieve a processed Data object previously loaded into memory.
        
        Parameters:
        	idx (int): Index of the item in the in-memory cache.
        
        Returns:
        	data (Data): The stored PyG Data object at the given index.
        """
        return self.data[idx]

    def train_test_loaders(self):
        """
        Randomly split the dataset 90/10 into training and test subsets and return corresponding DataLoaders.
        
        Returns:
            tuple: (train_loader, test_loader)
                - train_loader (AtomicDataLoader): DataLoader over 90% of the dataset with shuffling enabled.
                - test_loader (AtomicDataLoader): DataLoader over remaining 10% of the dataset with shuffling disabled.
        """
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


class atomic_hirshfeld_valencewdith_only_module_dataset(Dataset):
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
        lmdb_map_size=1099511627776,
        lmdb_readonly=False,
        cache_size=1000,
    ):
        """
        Initialize an LMDB-backed dataset for Hirshfeld valence-width data.
        
        Parameters:
            root (str): Path to the dataset root directory where raw/processed/LMDB files are stored.
            transform (callable, optional): Transformation applied on a data object when accessed.
            pre_transform (callable, optional): Transformation applied to data objects during processing.
            r_cut (float): Distance cutoff (angstrom) used for constructing short-range edges.
            testing (bool): If True, use smaller default MAX_SIZE for quicker testing.
            spec_type (int): Dataset specification selector. Must be one of [1, 5, 10].
            max_size (int or None): Maximum number of examples to expose from the processed dataset.
            force_reprocess (bool): If True, forces reprocessing of raw data and reinitialization of LMDB.
            in_memory (bool): If True, load the entire dataset into memory after initialization.
            batch_size (int): Default batch size intended for downstream DataLoader creation.
            lmdb_map_size (int): Maximum LMDB map size in bytes (default 1 TB).
            lmdb_readonly (bool): If True, open the LMDB environment in read-only mode.
            cache_size (int): Number of recently accessed items to keep in an LRU memory cache.
        
        Raises:
            ValueError: If `spec_type` is not one of [1, 5, 10].
        """

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
        self.r_cut = r_cut

        self.lmdb_map_size = lmdb_map_size
        self.lmdb_readonly = lmdb_readonly
        self.cache_size = cache_size
        self._cache = {}
        self._cache_keys = []

        self.lmdb_env = None
        self.lmdb_path = None
        self._length = None
        self._worker_id = None

        if os.path.exists(root) is False:
            os.makedirs(root)

        print(
            f"{self.root = }, {self.spec_type = }, {self.testing = }, {self.in_memory = }"
        )

        self._init_lmdb_path(root)
        self._init_lmdb()

        super(atomic_hirshfeld_valencewdith_only_module_dataset, self).__init__(
            root, transform, pre_transform
        )

        if self.force_reprocess:
            self.force_reprocess = False
            self._close_lmdb()
            super(atomic_hirshfeld_valencewdith_only_module_dataset, self).__init__(
                root, transform, pre_transform
            )
            self._init_lmdb()

        if self.in_memory:
            print("Loading data into memory")
            t = time()
            self.data = []
            for i in range(len(self)):
                self.data.append(self.get(i))
            total_time_seconds = int(time() - t)
            print(f"Loaded in {total_time_seconds:4d} seconds")
            self.get = self.get_in_memory

    def _init_lmdb_path(self, root):
        """Initialize LMDB path before parent class init"""
        self.lmdb_path = osp.join(
            root, "processed", f"lmdb_monomer_ap3_spec_{self.spec_type}"
        )

    def _init_lmdb(self):
        """Initialize LMDB environment"""
        if not osp.exists(self.lmdb_path):
            os.makedirs(self.lmdb_path, exist_ok=True)

        try:
            self.lmdb_env = lmdb.open(
                self.lmdb_path,
                map_size=self.lmdb_map_size,
                readonly=self.lmdb_readonly,
                max_dbs=0,
                lock=not self.lmdb_readonly,
                max_readers=256,
            )

            with self.lmdb_env.begin() as txn:
                metadata_bytes = txn.get(b"__metadata__")
                if metadata_bytes:
                    metadata = json.loads(metadata_bytes.decode("utf-8"))
                    self._length = metadata.get("length", 0)
                else:
                    self._length = 0
        except Exception as e:
            print(f"Error initializing LMDB: {e}")
            self.lmdb_env = None
            self._length = 0

    def _close_lmdb(self):
        """Close LMDB environment"""
        if self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

    def __del__(self):
        """Cleanup LMDB on deletion"""
        try:
            self._close_lmdb()
        except:
            pass

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
        """Check if LMDB database exists and has data"""
        if self.force_reprocess:
            return ["file"]

        if not hasattr(self, "lmdb_path") or self.lmdb_path is None:
            return ["lmdb_missing"]

        if osp.exists(self.lmdb_path):
            env = None
            try:
                env = lmdb.open(
                    self.lmdb_path,
                    readonly=True,
                    lock=False,
                    max_dbs=0,
                    create=False,
                    max_readers=256,
                )
                with env.begin() as txn:
                    metadata_bytes = txn.get(b"__metadata__")
                    if metadata_bytes:
                        metadata = json.loads(metadata_bytes.decode("utf-8"))
                        length = metadata.get("length", 0)
                        if length > 0:
                            return [f"lmdb_monomer_ap3_spec_{self.spec_type}"]
            except Exception as e:
                print(f"Error checking LMDB: {e}")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except:
                        pass

        return ["lmdb_missing"]

    def download(self):
        print(self.raw_file_names)
        raise ValueError("Downloads are not available!")

    def _store_to_lmdb(self, data_objects, start_idx):
        """Store data objects to LMDB"""
        import pickle

        if self.lmdb_env is None:
            raise RuntimeError("LMDB environment not initialized")

        with self.lmdb_env.begin(write=True) as txn:
            for i, data_obj in enumerate(data_objects):
                idx = start_idx + i
                key = str(idx).encode("utf-8")
                value = pickle.dumps(data_obj)
                txn.put(key, value)

            metadata = {
                "length": start_idx + len(data_objects),
                "r_cut": self.r_cut,
                "spec_type": self.spec_type,
            }
            txn.put(b"__metadata__", json.dumps(metadata).encode("utf-8"))

        self._length = start_idx + len(data_objects)

    def process(self, r_cut=5.0, edge_index_only=True):
        """Process dataset and store in LMDB"""
        idx = 0
        data_objects = []
        batch_size = 256  # Store in batches for efficiency

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
            for i in range(len(monomers)):
                if i % 1000 == 0:
                    print(f"{i}/{len(monomers)}, took {time() - t} seconds")
                    t = time()
                mol = monomers[i]
                data = qcel_mon_to_pyg_data(mol, r_cut=r_cut)
                if np.isnan(volume_ratios[i]).any():
                    print(f"NaN in volume ratios for index {i}, skipping")
                    continue
                data.volume_ratios = torch.tensor(volume_ratios[i], dtype=torch.float32)
                data.valence_widths = torch.tensor(
                    valence_widths[i], dtype=torch.float32
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_objects.append(data)

                # Store in batches
                if len(data_objects) >= batch_size:
                    start_idx = idx - len(data_objects) + 1
                    self._store_to_lmdb(data_objects, start_idx)
                    data_objects = []

                if self.MAX_SIZE is not None and idx >= self.MAX_SIZE:
                    break
                idx += 1

            if self.MAX_SIZE is not None and idx >= self.MAX_SIZE:
                break

        # Store remaining data
        if len(data_objects) > 0:
            start_idx = idx - len(data_objects)
            self._store_to_lmdb(data_objects, start_idx)
            print(
                f"Final: Stored {len(data_objects)} objects to LMDB at index {start_idx}"
            )

        print(f"Processing complete. Total time: {time() - t:.2f}s")
        return

    def len(self):
        """Return dataset length from LMDB metadata"""
        if self._length is not None:
            return self._length

        if self.lmdb_env is None:
            return 0

        with self.lmdb_env.begin() as txn:
            metadata_bytes = txn.get(b"__metadata__")
            if metadata_bytes:
                metadata = json.loads(metadata_bytes.decode("utf-8"))
                self._length = metadata.get("length", 0)
            else:
                self._length = 0

        return self._length

    def _check_worker_init(self):
        """Ensure LMDB env is initialized for current worker process"""
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
        else:
            worker_id = None

        if worker_id != self._worker_id:
            if self.lmdb_env is not None:
                self._close_lmdb()

            self._worker_id = worker_id
            self._init_lmdb()
            self._cache = {}
            self._cache_keys = []

    def get(self, idx):
        """Retrieve item from LMDB with caching"""
        import pickle

        self._check_worker_init()

        if idx in self._cache:
            self._cache_keys.remove(idx)
            self._cache_keys.append(idx)
            return self._cache[idx]

        if self.lmdb_env is None:
            raise RuntimeError("LMDB environment not initialized")

        with self.lmdb_env.begin() as txn:
            key = str(idx).encode("utf-8")
            value_bytes = txn.get(key)

            if value_bytes is None:
                raise IndexError(f"Index {idx} not found in LMDB database")

            data = pickle.loads(value_bytes)

        self._cache[idx] = data
        self._cache_keys.append(idx)

        if len(self._cache) > self.cache_size:
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]

        return data

    def get_in_memory(self, idx):
        return self.data[idx]

    def prefetch(self, indices):
        """Prefetch multiple items into cache"""
        import pickle

        if self.lmdb_env is None:
            return

        with self.lmdb_env.begin() as txn:
            for idx in indices:
                if idx not in self._cache:
                    key = str(idx).encode("utf-8")
                    value_bytes = txn.get(key)
                    if value_bytes:
                        data = pickle.loads(value_bytes)
                        self._cache[idx] = data
                        self._cache_keys.append(idx)

        while len(self._cache) > self.cache_size:
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]

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