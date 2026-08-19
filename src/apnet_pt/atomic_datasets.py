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
from .lmdb_utils import acquire_lmdb_env, release_lmdb_env

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
    Batch a list of atomic PyG Data objects into a single Data with reindexed atoms and optional full-edge indices.

    Parameters:
        batch (list[Data]): Sequence of PyG Data objects representing molecules/monomers. Each item must provide `x`, `edge_index`, per-atom targets (`charges`, `dipoles`, `quadrupoles`), `R`, `molecule_ind`, and `total_charge`. If present, `edge_index_full` will also be preserved and reindexed.

    Returns:
        Data: A single PyG Data object containing:
            - concatenated `x`, `R`, `charges`, `dipoles`, and `quadrupoles`
            - `edge_index` reindexed so atom indices are unique across molecules
            - optional `edge_index_full` reindexed and concatenated when present
            - `molecule_ind` indicating molecule membership per atom
            - `total_charge` per molecule
            - `natom_per_mol` giving the number of atoms in each molecule
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
    Collate a list of PyG Data objects into a single batched Data, reindexing atom-level edge indices so each molecule occupies a unique index range.

    Concise description:
    - Reindexes `edge_index` (and `edge_index_full` if present) by offsetting atom indices so no collisions occur across molecules.
    - Sets `molecule_ind` so each atom is tagged with its source-molecule index.
    - Concatenates per-atom fields (`x`, `R`, `volume_ratios`, `valence_widths`) and computes `natom_per_mol`.
    - Aggregates per-molecule `total_charge` into a tensor.

    Returns:
        Data: A PyG Data object containing the batched graph with fields `x`, `edge_index`, `R`, `molecule_ind`, `total_charge`, `natom_per_mol`, `volume_ratios`, `valence_widths`, and optionally `edge_index_full`.
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
    Reindex per-atom indices across a list of atomic PyG Data objects and combine them into a single batched Data object.

    Parameters:
        batch (Sequence[Data]): Sequence of per-molecule PyG Data objects. Each item must contain `x`, `edge_index`, `charges`, `dipoles`, `quadrupoles`, `R`, `molecule_ind`, `total_charge`, `volume_ratios`, and `valence_widths`. Items may optionally include `edge_index_full`.

    Returns:
        Data: A single PyG Data object representing the batch with:
            - `x`: concatenated atom features
            - `edge_index`: reindexed and concatenated short-range edges
            - `edge_index_full` (optional): reindexed and concatenated full edges if present in inputs
            - `charges`, `dipoles`, `quadrupoles`: concatenated per-atom targets
            - `R`: concatenated coordinates
            - `molecule_ind`: per-atom molecule index (0..batch_size-1)
            - `total_charge`: tensor of per-molecule total charges
            - `natom_per_mol`: tensor with number of atoms per molecule
            - `volume_ratios`, `valence_widths`: concatenated per-atom Hirshfeld properties
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
    Collate a list of PyG Data objects (without per-atom target tensors) into a single batched Data object with reindexed atoms and edges.

    Parameters:
        batch (list): List of torch_geometric.data.Data objects. Each element must provide `x`, `R`, `edge_index`, `molecule_ind`, and `total_charge`. If present, `edge_index_full` will be preserved and concatenated.

    Returns:
        torch_geometric.data.Data: Batched Data containing:
            - `x`: concatenated atomic feature matrix,
            - `edge_index`: concatenated and reindexed edge indices,
            - `R`: concatenated coordinates,
            - `molecule_ind`: per-atom molecule indices (0..batch_size-1),
            - `total_charge`: tensor of per-molecule total charges,
            - `natom_per_mol`: tensor with number of atoms per molecule,
            - `edge_index_full` (optional): concatenated and reindexed full-pair edge indices if present in inputs.
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
        Initialize an AtomicDataLoader with dataset, batching, and collation configuration.

        Parameters:
            dataset (Dataset | Sequence[BaseData] | DatasetAdapter): Source of data examples; if an OnDiskDataset is provided, the loader will use an in-memory index range (range(len(dataset))) instead of the dataset object itself.
            batch_size (int): Number of examples per batch.
            shuffle (bool): Whether to shuffle the dataset each epoch.
            follow_batch (list[str], optional): Keys whose per-example tensors should be tracked with follow_batch semantics when an internal Collater is created.
            exclude_keys (list[str], optional): Keys to omit from batching when an internal Collater is created.
            collate_fn (callable | None): Collation function to assemble a batch. If `None`, an internal Collater is constructed using `dataset`, `follow_batch`, and `exclude_keys`; otherwise the provided callable is used directly.
            **kwargs: Additional keyword arguments forwarded to the underlying PyTorch DataLoader constructor.
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
    Compute edge index arrays for atom pairs within a cutoff and optionally for all off-diagonal pairs.

    Parameters:
        R (array-like): Shape (N, 3) of Cartesian positions for N atoms.
        r_cut (float): Distance cutoff; pairs with distance < r_cut and > 0 are included in the short-range edges.
        full_indices (bool): If True, also return indices for all atom pairs excluding self-pairs.

    Returns:
        edges (np.ndarray): Array of shape [2, n_edges] containing index pairs for atom pairs with distance < r_cut and > 0.
        full_edges (np.ndarray, optional): Array of shape [2, n_pairs] containing index pairs for all off-diagonal atom pairs; returned only when `full_indices` is True.
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
    Convert a QCel monomer into a PyTorch Geometric Data object for atomic graph representations.

    Creates a Data object with per-atom features and connectivity suitable for graph-based models. Coordinates are converted to Angstroms.

    Parameters:
        mon: QCel monomer
            Monomer object providing atomic_numbers, geometry (in atomic units), and molecular_charge.
        r_cut (float): cutoff distance in Angstroms used to build short-range edges.
        custom (bool): if True, build index-only edges via the custom edge builder; otherwise use the standard edge construction.
        full_indices (bool): if True and not using `custom`, include `edge_index_full` containing all atom-pair indices in addition to the short-range `edge_index`.

    Returns:
        Data: PyG Data containing the following fields:
            - x: atomic numbers (tensor, long)
            - edge_index: short-range edge indices (tensor)
            - R: atomic coordinates in Angstroms (tensor, float)
            - molecule_ind: per-atom molecule index (tensor, long)
            - total_charge: molecular charge (tensor, long)
            - natom_per_mol: number of atoms in the monomer (tensor, long)
            - edge_index_full (optional): all atom-pair indices when `full_indices=True`
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
    Construct a PyG Data object representing an isolated molecule from atomic inputs.

    Parameters:
        Z (Sequence[int]): Atomic numbers for each atom.
        R (array-like[N, 3] or torch.Tensor[N, 3]): Cartesian coordinates in angstroms.
        total_charge (int): Total molecular charge.
        cartesian_multipoles (array-like, optional): Per-atom cartesian multipole targets to store as `y`.
        r_cut (float, optional): Cutoff distance (angstrom) used when building short-range edges. Default 5.0.
        idx (int, optional): Molecule index used to populate `molecule_ind` for every atom. Default 0.
        edge_index_only (bool, optional): When `custom=True`, if True only compute edge indices (no edge features). Default True.
        custom (bool, optional): If True, use alternate edge builders (`edge_function_system`/`edge_function_system_index_only`) instead of `edges`. Default False.
        full_indices (bool, optional): If True and `custom=False`, compute and include both short-range `edge_index` and `edge_index_full` (all atom pairs). Default False.

    Returns:
        torch_geometric.data.Data: A Data object containing:
          - x (Tensor[N]): atomic numbers (int64)
          - R (Tensor[N,3]): positions (float)
          - edge_index (LongTensor[2, E]): short-range edge indices
          - molecule_ind (Tensor[N]): per-atom molecule index
          - total_charge (Tensor): scalar total molecular charge
          - y (Tensor[N,...], optional): cartesian multipoles when provided
          - edge_index_full (LongTensor[2, E_full], optional): full atom-pair indices when `full_indices=True`
    """
    if isinstance(Z, torch.Tensor):
        node_features = Z.long()
    else:
        node_features = torch.tensor(Z, dtype=torch.int64)
    # check to ensure node_features are all in ALLOWED_ELEMENTS
    constants.ALLOWED_ELEMENTS
    for z in node_features:
        if z.item() not in constants.ALLOWED_ELEMENTS:
            return None
    if isinstance(R, np.ndarray):
        R = torch.tensor(R, dtype=torch.float32)
    if isinstance(total_charge, torch.Tensor):
        torch_total_charge = total_charge.to(dtype=torch.int32)
    else:
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
        Initialize the atomic_module_dataset and prepare processed data access.

        Parameters:
            root (str): Root directory for raw and processed dataset files.
            transform, pre_transform: Optional PyG transform callables applied on load or before processing.
            r_cut (float): Neighbor cutoff distance used during processing (stored for use by processing routines).
            testing (bool): If True, enable testing mode which limits MAX_SIZE when max_size is not provided.
            spec_type (int): Dataset specification selector; must be one of [1, 2, 3, 4, 6, 9, 10, 11, 12]. Determines which raw/processed file names and processing behavior to use.
            split (str): Subset to expose; typically "all", "train", or "test".
            max_size (int or None): Maximum number of examples to load/process; when None and testing is True, MAX_SIZE defaults to 200.
            force_reprocess (bool): If True, remove existing processed files for the selected spec_type before processing.
            in_memory (bool): If True, load all processed Data objects into memory and replace get() with an in-memory accessor.
            batch_size (int): Default batch size to be used by train/test loader helpers.

        Behavior:
            - Validates spec_type and raises ValueError for unsupported values.
            - Creates the root directory if it does not exist.
            - If force_reprocess is enabled, removes existing processed files for the chosen spec_type.
            - When in_memory is True, loads all processed files into memory and sets get() to return from the in-memory cache.
            - Stores provided configuration on the instance (e.g., spec_type, testing, MAX_SIZE, batch_size).

        Raises:
            ValueError: If spec_type is not one of the supported integers.
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
        Map the dataset configuration to the expected raw input filenames.

        When `testing` is True returns a single testing filename; otherwise returns a list of expected raw pickle filenames based on `spec_type` (supports spec_type values 1, 2, 3, 4, 6, 9, 10, 11, and 12). The returned names correspond to the raw monomer data files the dataset will look for when processing.

        Returns:
            list[str]: Filenames expected to exist in the raw data directory.

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
        Process raw monomer files into PyG Data objects, apply optional filters/transforms, and save processed items to disk.

        This method:
        - Loads monomer data from each path in self.raw_paths.
        - Converts each monomer into a PyG Data object (including cartesian multipoles mapped to per-atom charges, dipoles, and quadrupoles).
        - Applies self.pre_filter and self.pre_transform when present.
        - Writes each processed Data to self.processed_dir with naming dependent on self.spec_type and self.split or testing mode.
        - Stops early when self.MAX_SIZE is reached.

        Parameters:
            r_cut (float): Radial cutoff (in the same distance units used by the dataset) passed to the conversion routine for building edges.
            edge_index_only (bool): If True, indicates that only short-range edge indices should be considered when building graph edges; when False, full pairwise indices may be retained. (The conversion call may override this flag.)

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
        Initialize the Hirshfeld multipole PyG dataset and optionally preload processed items into memory.

        Parameters:
            root (str): Path to the dataset root directory where raw and processed files live.
            transform (callable, optional): Transform applied to data objects on access.
            pre_transform (callable, optional): Transform applied to data objects during processing.
            r_cut (float): Cutoff radius (angstroms) used when constructing edge indices.
            testing (bool): If True, use testing mode (affects MAX_SIZE default).
            spec_type (int): Dataset specification variant; must be one of [1, 5, 10, 11, 12].
            max_size (int or None): Maximum number of examples to load/process; when None and testing is True, defaults to 200.
            force_reprocess (bool): If True, ignore existing processed files and reprocess raw data.
            in_memory (bool): If True, load all processed data into memory and replace get() with get_in_memory().
            batch_size (int): Default batch size used by convenience loader helpers.

        Raises:
            ValueError: If `spec_type` is not one of the allowed values.
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
        Create randomized train and test DataLoader pairs from the dataset.

        The dataset is shuffled and split with 90% of samples used for training and 10% for testing. The returned training DataLoader shuffles batches; the test DataLoader does not.

        Returns:
            (train_loader, test_loader): A tuple where `train_loader` provides batches drawn from the 90% training split with shuffling enabled, and `test_loader` provides batches drawn from the remaining 10% with shuffling disabled. Both loaders use `self.batch_size` and `atomic_hirshfeld_collate_update` as the collate function.
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
        Initialize the dataset and optionally preload processed items while configuring a model to precompute Hirshfeld volume ratios and valence widths.

        Parameters:
            root (str): Root directory for raw/processed dataset files.
            atomtype_hfvr_model (torch.nn.Module): Pretrained model used to compute per-atom Hirshfeld volume ratios (hfvr) and valence widths (vw); the model will be set to evaluation mode and gradients disabled.
            transform (callable, optional): Transformation applied on-the-fly to examples returned by get().
            pre_transform (callable, optional): Transformation applied once to examples during processing.
            r_cut (float, optional): Cutoff distance (angstrom) used when constructing neighbor/edge information. Default is 5.0.
            testing (bool, optional): If True, operate in testing mode which may reduce MAX_SIZE defaults. Default is False.
            spec_type (int, optional): Dataset specification identifier (must be one of 5, 9, 10, 11, 12). Default is 9.
            max_size (int or None, optional): Maximum number of examples to process/load; None means no explicit limit unless testing sets a default.
            force_reprocess (bool, optional): If True, force reprocessing of raw data into processed files. Default is False.
            in_memory (bool, optional): If True, load all processed examples into memory and override get() with an in-memory accessor. Default is True.
            batch_size (int, optional): Default batch size to be used by train/test loader helpers. Default is 1.
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
        Expected raw filenames for supported spec_type values.

        Returns:
            list[str]: A list containing the raw filename(s) that should exist for this dataset instance.

        Raises:
            ValueError: If `spec_type` is not one of 5, 9, 10, 11, or 12.
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
        Determine the list of processed filenames for the induced-dipole precomputed dataset.

        Searches the processed directory for files matching "monomer_induced_dipole_precomputed_{spec_type}_*.pt", returns their basenames sorted using natural ordering, and limits the result to `self.MAX_SIZE` if set. If `self.force_reprocess` is true, returns ["file"]. If no matching files are found, returns a single placeholder filename "data_missing_0.pt".

        Returns:
            list[str]: A list of processed file basenames (or a placeholder) for this dataset instance.
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
        Indicate that automatic downloading of raw files is unsupported.

        Prints the dataset's expected raw_file_names and raises a ValueError.

        @raises ValueError: Always raised to signal that downloads are not available.
        """
        print(self.raw_file_names)
        raise ValueError("Downloads are not available!")

    def process(self, r_cut=5.0, edge_index_only=True):
        """
        Process raw monomer files, precompute Hirshfeld volume ratios and valence widths using the configured model, and save processed PyG Data objects to disk.

        Loads monomers from each raw_path, converts each monomer into a PyG Data object, attaches cartesian multipole targets (charges, dipoles, quadrupoles), computes per-atom `volume_ratios` and `valence_widths` by calling `self.atomtype_hfvr_model` (with gradients disabled), applies optional `pre_filter` and `pre_transform`, and writes each processed item to self.processed_dir using the filename pattern "monomer_induced_dipole_precomputed_{spec_type}_{idx}.pt".

        Parameters:
            r_cut (float): Distance cutoff (in angstroms) used when constructing atomic neighborhood/edges for the PyG Data conversion.
            edge_index_only (bool): Present for API compatibility; not used by this processing routine.
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
        Get the number of available processed dataset items.

        Returns:
            n (int): Number of processed files described by `processed_file_names`.
        """
        return len(self.processed_file_names)

    def get(self, idx):
        """
        Load a processed precomputed induced-dipole monomer item by index.

        Parameters:
            idx (int): Index of the processed item to load.

        Returns:
            data (torch_geometric.data.Data): The loaded PyG Data object from
            processed_dir/monomer_induced_dipole_precomputed_{spec_type}_{idx}.pt.
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
        Retrieve a preloaded dataset item by its index.

        Parameters:
            idx (int): Integer index of the item in the in-memory cache.

        Returns:
            The cached data object stored at the given index.
        """
        return self.data[idx]

    def train_test_loaders(self):
        """
        Create randomized train and test DataLoader pairs from the dataset.

        The dataset is shuffled and split with 90% of samples used for training and 10% for testing. The returned training DataLoader shuffles batches; the test DataLoader does not.

        Returns:
            (train_loader, test_loader): A tuple where `train_loader` provides batches drawn from the 90% training split with shuffling enabled, and `test_loader` provides batches drawn from the remaining 10% with shuffling disabled. Both loaders use `self.batch_size` and `atomic_hirshfeld_collate_update` as the collate function.
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
        Initialize the LMDB-backed atomic dataset and prepare storage, caching, and optional in-memory loading.

        Parameters:
            root (str): Root directory for dataset files and LMDB storage.
            transform (callable, optional): Function applied to each example on access.
            pre_transform (callable, optional): Function applied to each example before saving to storage.
            r_cut (float): Cutoff radius (Å) used when constructing graph edges.
            testing (bool): If True, use a smaller default MAX_SIZE when max_size is not provided.
            spec_type (int): Dataset specification identifier; must be one of [5, 9, 10, 11, 12].
            max_size (int, optional): Maximum number of examples to process or load; None means no explicit limit.
            force_reprocess (bool): If True, re-run processing and recreate the LMDB/processed store.
            in_memory (bool): If True, load all dataset items into memory and use in-memory access.
            batch_size (int): Default batch size used by convenience loader helpers.
            lmdb_map_size (int): Maximum LMDB file size in bytes (map size) used when creating the LMDB environment.
            lmdb_readonly (bool): If True, open the LMDB in read-only mode.
            cache_size (int): Number of recently accessed items to keep in an internal LRU cache.
            atomtype_hfvr_model (torch.nn.Module, optional): Pretrained model used during processing to precompute
                Hirshfeld volume ratios and valence widths; if provided, the model is set to evaluation mode and
                gradients are disabled.

        Notes:
            - Validates spec_type and initializes LMDB paths and environment.
            - If in_memory is True, all items are loaded and self.get is overridden to return from the in-memory cache.
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
        Set the LMDB directory path for this dataset on the instance.

        Parameters:
            root (str or os.PathLike): Base dataset root directory; sets self.lmdb_path to
                root/processed/lmdb_atomic_induced_dipole_spec_{self.spec_type}.
        """
        self.lmdb_path = osp.join(
            root, "processed", f"lmdb_atomic_induced_dipole_spec_{self.spec_type}"
        )

    def _init_lmdb(self):
        """
        Initialize and open the LMDB environment for this dataset instance.

        Creates the LMDB directory if it does not exist, opens the LMDB environment using the instance's configuration (map size, readonly mode, etc.), and loads the stored metadata length into self._length (defaults to 0 if missing). On failure, sets self.lmdb_env to None, sets self._length to 0, and prints an error message.
        """
        if not osp.exists(self.lmdb_path):
            os.makedirs(self.lmdb_path, exist_ok=True)

        for attempt in range(2):
            try:
                self.lmdb_env = acquire_lmdb_env(
                    lmdb,
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
                return
            except Exception as e:
                if attempt == 0 and "already open in this process" in str(e):
                    import gc

                    if self.lmdb_env is not None:
                        release_lmdb_env(self.lmdb_path, self.lmdb_env)
                        self.lmdb_env = None
                        self._length = 0
                    gc.collect()
                    continue
                print(f"Error initializing LMDB: {e}")
                if self.lmdb_env is not None:
                    release_lmdb_env(self.lmdb_path, self.lmdb_env)
                self.lmdb_env = None
                self._length = 0
                return

    def _close_lmdb(self):
        """
        Close the LMDB environment and clear the cached environment reference.

        If an LMDB environment is open, it is closed and `self.lmdb_env` is set to `None`.
        """
        if self.lmdb_env is not None:
            release_lmdb_env(self.lmdb_path, self.lmdb_env)
            self.lmdb_env = None

    def __del__(self):
        """Cleanup LMDB on deletion"""
        try:
            self._close_lmdb()
        except:
            pass

    def __getstate__(self):
        """
        Return a picklable state dict with LMDB-related resources closed and removed.

        Closes the LMDB environment if present, clears the in-memory cache and cache keys, and resets the worker identifier so the resulting state contains only picklable entries.

        Returns:
            state (dict): A shallow copy of the object's __dict__ adapted for pickling:
                - `lmdb_env` set to None
                - `_cache` replaced with an empty dict
                - `_cache_keys` replaced with an empty list
                - `_worker_id` set to None
        """
        state = self.__dict__.copy()
        # Close LMDB environment before pickling
        if "lmdb_env" in state and state["lmdb_env"] is not None:
            try:
                release_lmdb_env(state["lmdb_path"], state["lmdb_env"])
            except Exception:
                pass
            self.lmdb_env = None
        # Remove unpicklable objects
        state["lmdb_env"] = None
        state["_cache"] = {}
        state["_cache_keys"] = []
        state["_worker_id"] = None
        return state

    def __setstate__(self, state):
        """
        Restore the object's state after unpickling and reinitialize its LMDB environment.

        The object's dictionary is updated from the given `state`, and any LMDB resources
        are reopened so the instance is ready for use in the current process.
        """
        self.__dict__.update(state)
        # Reinitialize LMDB in the new process
        self._init_lmdb()

    @property
    def raw_file_names(self):
        """
        Return the expected raw filename(s) for this dataset based on its `spec_type`.

        For spec_type 5 returns ["monomers_ap3_spec_5_pbe0.pkl"];
        for spec_type 9 returns ["monomers_ap3_spec_5_pbe0.pkl"];
        for spec_type 10 returns ["monomers_ap3_spec_10_HF.pkl"];
        for spec_type 11 or 12 returns ["SPICE_monomer_spec_{spec_type}.pkl"].

        Returns:
            list[str]: A list containing one raw filename expected to exist in the raw directory.

        Raises:
            ValueError: If `spec_type` is not one of 5, 9, 10, 11, or 12.
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
        Determine which processed-file marker to report for this dataset based on LMDB availability and metadata.

        Checks in order:
        - If force_reprocess is True, reports a placeholder indicating reprocessing is required.
        - If an LMDB path is configured and the LMDB exists with a stored metadata length greater than zero, returns the LMDB dataset marker name for the current spec_type.
        - Otherwise reports that LMDB is missing.

        Returns:
            list[str]: A single-item list containing one of:
                - "file" when force_reprocess is True.
                - "lmdb_atomic_induced_dipole_spec_{spec_type}" when a valid LMDB with length > 0 is found.
                - "lmdb_missing" when no usable LMDB is available or an error occurs while checking.
        """
        if self.force_reprocess:
            return ["file"]

        if not hasattr(self, "lmdb_path") or self.lmdb_path is None:
            return ["lmdb_missing"]

        if osp.exists(self.lmdb_path):
            lmdb_env = None
            try:
                lmdb_env = acquire_lmdb_env(
                    lmdb,
                    self.lmdb_path,
                    map_size=self.lmdb_map_size,
                    readonly=True,
                    lock=False,
                    max_dbs=0,
                    max_readers=256,
                )
                with lmdb_env.begin() as txn:
                    metadata_bytes = txn.get(b"__metadata__")
                    if metadata_bytes:
                        metadata = json.loads(metadata_bytes.decode("utf-8"))
                        length = metadata.get("length", 0)

                        if length > 0:
                            return [f"lmdb_atomic_induced_dipole_spec_{self.spec_type}"]
            except Exception as e:
                print(f"Error checking LMDB: {e}")
            finally:
                if lmdb_env is not None:
                    release_lmdb_env(self.lmdb_path, lmdb_env)

        return ["lmdb_missing"]

    def download(self):
        """
        Indicate that automatic downloading is not supported for this dataset.

        This method notifies callers that no remote download is available and prints the dataset's expected raw file names.

        Raises:
            ValueError: Always raised with the message "Downloads are not available!".
        """
        print(self.raw_file_names)
        raise ValueError("Downloads are not available!")

    def _store_to_lmdb(self, data_objects, start_idx):
        """
        Write the given sequence of data objects into the dataset's LMDB starting at the provided index.

        Each object is serialized and stored under an integer key derived from its index; the LMDB "__metadata__" entry is updated to reflect the new total length and dataset attributes.

        Parameters:
            data_objects (Iterable): Iterable of objects to store (each will be serialized).
            start_idx (int): Index at which the first object will be stored; subsequent objects use consecutive indices.

        Raises:
            RuntimeError: If the LMDB environment is not initialized.
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
        Process raw monomer files into PyG Data objects and store them in the dataset's LMDB.

        Processes each raw file in self.raw_paths, converts QCArchive/QCel monomers to PyG Data (with full edge indices), attaches cartesian multipole targets (charges, dipoles, quadrupoles), and either precomputes or loads Hirshfeld-derived volume_ratios and valence_widths. Applies self.pre_filter and self.pre_transform when present, batches processed items for efficient LMDB insertion via self._store_to_lmdb, and respects self.MAX_SIZE. Skips entries with NaNs in raw Hirshfeld properties when no precomputed model is provided.

        Parameters:
            r_cut (float): Distance cutoff (in angstroms) used when building interatomic edges.
            edge_index_only (bool): Present for API compatibility; when True/False does not change processing here (full edge indices are produced).
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

        If the LMDB environment is not initialized or the metadata entry is missing, returns 0. The value is cached on first successful read in `self._length`.

        Returns:
            int: Number of entries recorded in the LMDB metadata, or 0 if unavailable.
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
        Return the dataset item stored at the given index from the LMDB store, using and updating an LRU cache.

        Parameters:
            idx (int): Integer index of the item to retrieve.

        Returns:
            The deserialized Python object stored for the given index.

        Raises:
            RuntimeError: If the LMDB environment has not been initialized.
            IndexError: If the specified index is not present in the LMDB.
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
        Retrieve a cached dataset item by index.

        Parameters:
            idx (int): Index of the item in the in-memory cache.

        Returns:
            data: The stored data object at the given index.
        """
        return self.data[idx]

    def train_test_loaders(self):
        """
        Create randomized train and test DataLoader objects using a 90% / 10% split of the dataset.

        The split is performed by a random permutation of indices; the training loader is shuffled and the test loader is not.

        Returns:
            tuple: (train_loader, test_loader) — two AtomicDataLoader instances for the training and test subsets, respectively.
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
        Initialize an LMDB-backed dataset for Hirshfeld valence-width training.

        This constructor prepares LMDB storage (creating directories if needed), configures caching and in-memory loading, validates supported spec types, and optionally preloads all items into memory.

        Parameters:
            root (str): Filesystem path used as dataset root and LMDB location.
            transform (callable, optional): PyG transform applied on access (kept for compatibility).
            pre_transform (callable, optional): PyG pre-transform applied during processing (kept for compatibility).
            r_cut (float, optional): Radial cutoff (angstroms) used when constructing edge indices. Default 5.0.
            testing (bool, optional): If True, use a reduced MAX_SIZE useful for testing. Default False.
            spec_type (int, optional): Dataset specification identifier (supported: 1, 5, 10). Determines expected raw/processed file naming. Default 1.
            max_size (int or None, optional): Maximum number of examples to expose; if None and testing is False, no artificial limit is applied.
            force_reprocess (bool, optional): When True, reinitialize LMDB and force reprocessing of raw data. Default False.
            in_memory (bool, optional): When True, preload all dataset items into memory and make get() return in-memory copies. Default True.
            batch_size (int, optional): Default batch size stored on the dataset instance (used by helper loader factories). Default 1.
            lmdb_map_size (int, optional): Maximum LMDB map size in bytes used when creating the environment. Default 1099511627776 (1 TB).
            lmdb_readonly (bool, optional): If True, open LMDB in read-only mode. Default False.
            cache_size (int, optional): Size of the LRU cache (number of recently accessed items to retain in memory). Default 1000.

        Raises:
            ValueError: If spec_type is not one of the supported values (1, 5, 10).
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

        for attempt in range(2):
            try:
                self.lmdb_env = acquire_lmdb_env(
                    lmdb,
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
                return
            except Exception as e:
                if attempt == 0 and "already open in this process" in str(e):
                    import gc

                    if self.lmdb_env is not None:
                        release_lmdb_env(self.lmdb_path, self.lmdb_env)
                        self.lmdb_env = None
                        self._length = 0
                    gc.collect()
                    continue
                print(f"Error initializing LMDB: {e}")
                if self.lmdb_env is not None:
                    release_lmdb_env(self.lmdb_path, self.lmdb_env)
                self.lmdb_env = None
                self._length = 0
                return

    def _close_lmdb(self):
        """Close LMDB environment"""
        if self.lmdb_env is not None:
            release_lmdb_env(self.lmdb_path, self.lmdb_env)
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
            lmdb_env = None
            try:
                lmdb_env = acquire_lmdb_env(
                    lmdb,
                    self.lmdb_path,
                    map_size=self.lmdb_map_size,
                    readonly=True,
                    lock=False,
                    max_dbs=0,
                    max_readers=256,
                )
                with lmdb_env.begin() as txn:
                    metadata_bytes = txn.get(b"__metadata__")
                    if metadata_bytes:
                        metadata = json.loads(metadata_bytes.decode("utf-8"))
                        length = metadata.get("length", 0)
                        if length > 0:
                            return [f"lmdb_monomer_ap3_spec_{self.spec_type}"]
            except Exception as e:
                print(f"Error checking LMDB: {e}")
            finally:
                if lmdb_env is not None:
                    release_lmdb_env(self.lmdb_path, lmdb_env)

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
