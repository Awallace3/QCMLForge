import os
import qcelemental as qcel
from typing import List, Optional, Sequence, Union
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os.path as osp
import torch
from torch_geometric.data import download_url

from .. import util
from ..AtomModels.ap2_atom_model import AtomModel
from ..hf_pretrained import resolve_pretrained_path
from .. import atomic_datasets
from glob import glob
import tarfile
from time import time
import re
from pathlib import Path
from importlib import resources
from apnet_pt import constants
import h5py


AP2_FUSED_SPLIT_SPEC_TYPES = frozenset({2, 5, 6, 7, 9})


def spec_type_uses_split_files(spec_type):
    return spec_type in AP2_FUSED_SPLIT_SPEC_TYPES


def qcel_inputs_are_split_db(qcel_molecules):
    return (
        qcel_molecules is not None
        and len(qcel_molecules) == 2
        and isinstance(qcel_molecules[0], list)
    )


def qcel_dimer_to_fused_data(dimer, r_cut=5.0, r_cut_im=8.0, **kwargs):
    return dimer_fused_data(
        RA=dimer.get_fragment(0).geometry * constants.au2ang,
        ZA=dimer.get_fragment(0).atomic_numbers,
        TQA=dimer.get_fragment(0).molecular_charge,
        RB=dimer.get_fragment(1).geometry * constants.au2ang,
        ZB=dimer.get_fragment(1).atomic_numbers,
        TQB=dimer.get_fragment(1).molecular_charge,
        r_cut=r_cut,
        r_cut_im=r_cut_im,
        **kwargs,
    )


def assert_molecule_featurization_is_valid(atomic_props, dimer_ind):
    if len(atomic_props.molecule_ind) == 1:
        return True

    edge_index = atomic_props.edge_index
    atoms_with_edges = torch.cat([edge_index[0], edge_index[1]]).unique()
    keep_mask = torch.isin(
        torch.arange(
            len(atomic_props.molecule_ind), device=atomic_props.molecule_ind.device
        ),
        atoms_with_edges,
    )
    # all valid molecules will have edges
    if not keep_mask.all():
        print(
            "Molecule featurization is invalid. "
            "Some atoms in the molecule do not have edges."
            f"atomic_props:\n{atomic_props}. Skipping this data point."
        )
        torch.save(atomic_props, f"invalid_molecule_{dimer_ind}.pt")
        return False
    return True


def dimer_fused_data(
    RA,
    ZA,
    TQA,
    RB,
    ZB,
    TQB,
    dimer_ind,
    r_cut=5.0,
    r_cut_im=8.0,
    check_validity=True,
    **kwargs,
):
    """
    Constructs a PyG Data object representing a fused dimer from two monomer specifications.

    Creates per-monomer atomic features, optionally validates featurization, computes intra-monomer edges and inter-monomer short- and long-range edges, and returns a Data object containing atom tensors, edge index tensors, molecule indices, and total charges. Returns `None` if validity checks fail.

    Parameters:
        RA (Tensor): Positions for monomer A (cartesian coordinates, expected in angstroms).
        ZA (Tensor or sequence): Atomic numbers or one-hot/feature tensor for monomer A.
        TQA (Tensor or sequence): Per-atom quantum labels/properties for monomer A.
        RB (Tensor): Positions for monomer B.
        ZB (Tensor or sequence): Atomic numbers or one-hot/feature tensor for monomer B.
        TQB (Tensor or sequence): Per-atom quantum labels/properties for monomer B.
        dimer_ind (int): Integer index identifying the dimer (used to populate dimer index fields).
        r_cut (float, optional): Distance cutoff for intramonomer (AA/BB) edges. Default 5.0.
        r_cut_im (float, optional): Distance threshold separating short- and long-range inter-monomer AB edges. Default 8.0.
        check_validity (bool, optional): If True, validate that each atom in a monomer has at least one edge; return `None` when invalid. Default True.
        **kwargs: Additional attributes to include in the returned Data object.

    Returns:
        Data or None: A PyG Data object containing at least the following fields when successful:
            - ZA, RA, ZB, RB: atom features and positions for monomers A and B
            - e_ABsr_source, e_ABsr_target: short-range AB edge indices
            - dimer_ind: dimer index for short-range AB edges
            - e_ABlr_source, e_ABlr_target: long-range AB edge indices
            - dimer_ind_lr: dimer index for long-range AB edges
            - e_AA_source, e_AA_target: intramonomer A edge indices
            - e_BB_source, e_BB_target: intramonomer B edge indices
            - molecule_ind_A, molecule_ind_B: per-atom molecule indices for A and B
            - total_charge_A, total_charge_B: total charges for monomers A and B
        Returns `None` if validation fails.
    """
    atomic_props_A = atomic_datasets.create_atomic_data(ZA, RA, TQA, r_cut=r_cut)
    atomic_props_B = atomic_datasets.create_atomic_data(ZB, RB, TQB, r_cut=r_cut)
    if check_validity:
        valid = assert_molecule_featurization_is_valid(atomic_props_A, dimer_ind)
        if not valid:
            return None
        valid = assert_molecule_featurization_is_valid(atomic_props_B, dimer_ind)
        if not valid:
            return None
    e_AA_source, e_AA_target = pairwise_edges(atomic_props_A.R, r_cut)
    e_BB_source, e_BB_target = pairwise_edges(atomic_props_B.R, r_cut)
    e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = pairwise_edges_im(
        atomic_props_A.R, atomic_props_B.R, r_cut_im
    )
    dimer_ind = torch.ones((1), dtype=torch.long) * dimer_ind
    return Data(
        ZA=atomic_props_A.x,
        RA=atomic_props_A.R,
        ZB=atomic_props_B.x,
        RB=atomic_props_B.R,
        # short range, intermolecular edges
        e_ABsr_source=e_ABsr_source,
        e_ABsr_target=e_ABsr_target,
        dimer_ind=dimer_ind,
        # long range, intermolecular edges
        e_ABlr_source=e_ABlr_source,
        e_ABlr_target=e_ABlr_target,
        dimer_ind_lr=dimer_ind,
        # intramonomer edges (monomer A)
        e_AA_source=e_AA_source,
        e_AA_target=e_AA_target,
        molecule_ind_A=atomic_props_A.molecule_ind,
        # intramonomer edges (monomer B)
        e_BB_source=e_BB_source,
        e_BB_target=e_BB_target,
        molecule_ind_B=atomic_props_B.molecule_ind,
        # monomer charges
        total_charge_A=atomic_props_A.total_charge,
        total_charge_B=atomic_props_B.total_charge,
        **kwargs,  # allows for additional properties to be passed in
    )


def natural_key(text):
    """
    Create a sort key that splits a string into alternating non-numeric and numeric parts, converting numeric parts to integers for natural ordering.

    Parameters:
        text (str): Input string to be split into components.

    Returns:
        list: A list of parts where contiguous digit sequences are converted to `int` and all other segments remain `str`, suitable for natural/alphanumeric sorting.
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", text)]


###############################
#######   PairDataset   #######
###############################


def pairwise_edges(R, r_cut, full_indices=False):
    natom = R.size(0)

    RA = R.unsqueeze(0).repeat(natom, 1, 1)  # [natom x natom x 3]
    RB = R.unsqueeze(1).repeat(1, natom, 1)  # [natom x natom x 3]

    dist = torch.norm(RA - RB, dim=2)

    mask = (dist < r_cut) & (dist > 0.0)
    edges = torch.where(mask)  # indices where the mask is true
    if full_indices:
        full_edges = torch.where((dist > 0.0))
        return (
            edges[0].long(),
            edges[1].long(),
            full_edges[0].long(),
            full_edges[1].long(),
        )
    return edges[0].long(), edges[1].long()


def pairwise_edges_im(RA, RB, r_cut_im, full_indices=False):
    natomA = RA.shape[0]
    natomB = RB.shape[0]

    RA_temp = RA.unsqueeze(1).repeat(1, natomB, 1)  # [natomA x natomB x 3]
    RB_temp = RB.unsqueeze(0).repeat(natomA, 1, 1)  # [natomA x natomB x 3]

    dist = torch.norm(RA_temp - RB_temp, dim=2)

    mask = dist <= r_cut_im
    # dimensions [n_edge x 2]
    edges_sr = torch.nonzero(mask, as_tuple=False).long()
    # dimensions [n_edge x 2]
    edges_lr = torch.nonzero(~mask, as_tuple=False).long()

    if full_indices:
        full_edges = torch.cat([edges_sr, edges_lr], dim=0)
        return (
            edges_sr[:, 0],
            edges_sr[:, 1],
            edges_lr[:, 0],
            edges_lr[:, 1],
            full_edges[0].long(),
            full_edges[1].long(),
        )
    return edges_sr[:, 0], edges_sr[:, 1], edges_lr[:, 0], edges_lr[:, 1]


def ap2_fused_collate_update(batch):
    """
    Merge a list of per-dimer PyG Data objects into a single batched Data with unique, offset atom and edge indices.

    Parameters:
        batch (list[Data]): List of per-dimer Data objects containing fields for monomers A and B (ZA, RA, ZB, RB), per-monomer edge indices (e_AA_*, e_BB_*), inter-monomer edge lists (e_ABsr_*, e_ABlr_*), molecule and dimer index tensors, charges, and target y.

    Returns:
        Data: A single PyG Data object containing:
            - y: stacked targets for the batch.
            - ZA, RA, ZB, RB: concatenated atomic numbers and coordinates for monomers A and B.
            - e_AA_source, e_AA_target, e_BB_source, e_BB_target: concatenated intramonomer edge index lists with per-item offsets applied.
            - e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target: concatenated short- and long-range inter-monomer edge lists with atom-index offsets applied so every atom in the batch has a unique index.
            - dimer_ind, dimer_ind_lr: per-edge tensors indicating the originating dimer index for short- and long-range AB edges.
            - molecule_ind_A, molecule_ind_B: concatenated per-atom molecule indices for A and B.
            - natom_per_mol_A, natom_per_mol_B: number of atoms per molecule for A and B.
            - total_charge_A, total_charge_B: per-dimer total charges for A and B.
            - batch_atomic_A, batch_atomic_B: Data sub-objects for the batched monomer A and B atom graphs (x, edge_index, R, molecule_ind, total_charge, natom_per_mol).

    Behavior notes:
        - Per-item molecule and dimer index tensors are replaced with batch-local indices (0..batch_size-1).
        - All atom indices in edge lists are offset so atoms from different batch entries do not collide.
    """
    monA_edge_offset, monB_edge_offset = 0, 0
    local_e_ABsr_source = []
    local_e_ABsr_target = []
    local_e_ABlr_source = []
    local_e_ABlr_target = []

    local_e_AA_source = []
    local_e_AA_target = []
    local_e_BB_source = []
    local_e_BB_target = []

    for i, data in enumerate(batch):
        # need dimer ind to be index size of short-range edges
        data.dimer_ind = (
            torch.ones(data.e_ABsr_source.size(0), dtype=data.dimer_ind.dtype) * i
        )
        data.dimer_ind_lr = (
            torch.ones(data.e_ABlr_source.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
        data.molecule_ind_A = (
            torch.ones(data.molecule_ind_A.size(0), dtype=data.molecule_ind_A.dtype) * i
        )
        data.molecule_ind_B = (
            torch.ones(data.molecule_ind_B.size(0), dtype=data.molecule_ind_B.dtype) * i
        )
        local_e_ABsr_source.append(data.e_ABsr_source.clone() + monA_edge_offset)
        local_e_ABsr_target.append(data.e_ABsr_target.clone() + monB_edge_offset)
        local_e_ABlr_source.append(data.e_ABlr_source.clone() + monA_edge_offset)
        local_e_ABlr_target.append(data.e_ABlr_target.clone() + monB_edge_offset)

        # Monomer edges
        local_e_AA_source.append(data.e_AA_source.clone() + monA_edge_offset)
        local_e_AA_target.append(data.e_AA_target.clone() + monA_edge_offset)
        local_e_BB_source.append(data.e_BB_source.clone() + monB_edge_offset)
        local_e_BB_target.append(data.e_BB_target.clone() + monB_edge_offset)

        monA_edge_offset += data.RA.size(0)
        monB_edge_offset += data.RB.size(0)
    molecule_ind_A = torch.cat([data.molecule_ind_A for data in batch], dim=0)
    molecule_ind_B = torch.cat([data.molecule_ind_B for data in batch], dim=0)
    natom_per_mol_A = torch.bincount(molecule_ind_A)
    natom_per_mol_B = torch.bincount(molecule_ind_B)
    y = torch.stack([data.y for data in batch], dim=0)
    ZA_cat = torch.cat([data.ZA for data in batch], dim=0)
    RA_cat = torch.cat([data.RA for data in batch], dim=0)
    ZB_cat = torch.cat([data.ZB for data in batch], dim=0)
    RB_cat = torch.cat([data.RB for data in batch], dim=0)
    e_AA_source_cat = torch.cat(local_e_AA_source, dim=0)
    e_AA_target_cat = torch.cat(local_e_AA_target, dim=0)
    e_BB_source_cat = torch.cat(local_e_BB_source, dim=0)
    e_BB_target_cat = torch.cat(local_e_BB_target, dim=0)
    total_charge_A_tensor = torch.tensor(
        [data.total_charge_A for data in batch], dtype=batch[0].total_charge_A.dtype
    )
    total_charge_B_tensor = torch.tensor(
        [data.total_charge_B for data in batch], dtype=batch[0].total_charge_B.dtype
    )

    batch_atomic_A = Data(
        x=ZA_cat,
        edge_index=torch.vstack((e_AA_source_cat, e_AA_target_cat)),
        R=RA_cat,
        molecule_ind=molecule_ind_A,
        total_charge=total_charge_A_tensor,
        natom_per_mol=natom_per_mol_A,
    )

    batch_atomic_B = Data(
        x=ZB_cat,
        edge_index=torch.vstack((e_BB_source_cat, e_BB_target_cat)),
        R=RB_cat,
        molecule_ind=molecule_ind_B,
        total_charge=total_charge_B_tensor,
        natom_per_mol=natom_per_mol_B,
    )

    batched_data = Data(
        y=y,
        ZA=ZA_cat,
        RA=RA_cat,
        ZB=ZB_cat,
        RB=RB_cat,
        e_AA_source=e_AA_source_cat,
        e_AA_target=e_AA_target_cat,
        e_BB_source=e_BB_source_cat,
        e_BB_target=e_BB_target_cat,
        molecule_ind_A=molecule_ind_A,
        molecule_ind_B=molecule_ind_B,
        natom_per_mol_A=natom_per_mol_A,
        natom_per_mol_B=natom_per_mol_B,
        e_ABsr_source=torch.cat(local_e_ABsr_source, dim=0),
        e_ABsr_target=torch.cat(local_e_ABsr_target, dim=0),
        e_ABlr_source=torch.cat(local_e_ABlr_source, dim=0),
        e_ABlr_target=torch.cat(local_e_ABlr_target, dim=0),
        dimer_ind=torch.cat([data.dimer_ind for data in batch], dim=0),
        dimer_ind_lr=torch.cat([data.dimer_ind_lr for data in batch], dim=0),
        total_charge_A=total_charge_A_tensor,
        total_charge_B=total_charge_B_tensor,
        batch_atomic_A=batch_atomic_A,
        batch_atomic_B=batch_atomic_B,
    )
    return batched_data


def ap2_fused_collate_update_no_target(batch):
    """
    Batch-collate a list of fused Data objects into a single batched Data without target labels.

    This function offsets and concatenates per-item atom and edge indices so every atom in the batch has a unique global index, assembles per-monomer (A and B) batched subgraphs, and produces combined inter-monomer edge lists. The resulting Data includes short- and long-range AB edges, a concatenated full AB edge list, corresponding dimer index vectors (including `dimer_ind_full`), per-molecule indices and counts, total charges, and the per-monomer Data objects `batch_atomic_A` and `batch_atomic_B`.

    Returns:
        Data: A PyG Data object containing the following notable fields:
            - ZA, RA, ZB, RB: concatenated atomic numbers and coordinates for monomers A and B
            - e_AA_source/target, e_BB_source/target: concatenated intra-monomer edge indices
            - e_ABsr_source/target, e_ABlr_source/target: concatenated short- and long-range AB edge indices
            - e_ABfull_source/target: concatenated full AB edge indices (short + long)
            - dimer_ind, dimer_ind_lr, dimer_ind_full: dimer index vectors for AB edges
            - molecule_ind_A, molecule_ind_B: per-atom molecule indices for A and B
            - natom_per_mol_A, natom_per_mol_B: atom counts per molecule for A and B
            - total_charge_A, total_charge_B: per-dimer total charges for A and B
            - indA, indB: batch-local atom-to-dimer mapping for A and B
            - batch_atomic_A, batch_atomic_B: Data objects representing the batched A and B atomic graphs
    """
    monA_edge_offset, monB_edge_offset = 0, 0
    local_e_ABsr_source = []
    local_e_ABsr_target = []
    local_e_ABlr_source = []
    local_e_ABlr_target = []
    local_e_AA_source = []
    local_e_AA_target = []
    local_e_BB_source = []
    local_e_BB_target = []
    local_indA = []
    local_indB = []
    for i, data in enumerate(batch):
        data.dimer_ind = (
            torch.ones(data.e_ABsr_source.size(0), dtype=data.dimer_ind.dtype) * i
        )
        data.dimer_ind_lr = (
            torch.ones(data.e_ABlr_source.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
        data.molecule_ind_A = (
            torch.ones(data.molecule_ind_A.size(0), dtype=data.molecule_ind_A.dtype) * i
        )
        data.molecule_ind_B = (
            torch.ones(data.molecule_ind_B.size(0), dtype=data.molecule_ind_B.dtype) * i
        )
        local_e_ABsr_source.append(data.e_ABsr_source.clone() + monA_edge_offset)
        local_e_ABsr_target.append(data.e_ABsr_target.clone() + monB_edge_offset)
        local_e_ABlr_source.append(data.e_ABlr_source.clone() + monA_edge_offset)
        local_e_ABlr_target.append(data.e_ABlr_target.clone() + monB_edge_offset)
        local_e_AA_source.append(data.e_AA_source.clone() + monA_edge_offset)
        local_e_AA_target.append(data.e_AA_target.clone() + monA_edge_offset)
        local_e_BB_source.append(data.e_BB_source.clone() + monB_edge_offset)
        local_e_BB_target.append(data.e_BB_target.clone() + monB_edge_offset)

        monA_edge_offset += data.RA.size(0)
        monB_edge_offset += data.RB.size(0)

        local_indA.append(torch.ones(data.RA.size(0), dtype=data.dimer_ind.dtype) * i)
        local_indB.append(
            torch.ones(data.RB.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
    molecule_ind_A = torch.cat([data.molecule_ind_A for data in batch], dim=0)
    molecule_ind_B = torch.cat([data.molecule_ind_B for data in batch], dim=0)
    natom_per_mol_A = torch.bincount(molecule_ind_A)
    natom_per_mol_B = torch.bincount(molecule_ind_B)

    ZA_cat = torch.cat([data.ZA for data in batch], dim=0)
    RA_cat = torch.cat([data.RA for data in batch], dim=0)
    ZB_cat = torch.cat([data.ZB for data in batch], dim=0)
    RB_cat = torch.cat([data.RB for data in batch], dim=0)
    e_AA_source_cat = torch.cat(local_e_AA_source, dim=0)
    e_AA_target_cat = torch.cat(local_e_AA_target, dim=0)
    e_BB_source_cat = torch.cat(local_e_BB_source, dim=0)
    e_BB_target_cat = torch.cat(local_e_BB_target, dim=0)
    total_charge_A_tensor = torch.tensor(
        [data.total_charge_A for data in batch], dtype=batch[0].total_charge_A.dtype
    )
    total_charge_B_tensor = torch.tensor(
        [data.total_charge_B for data in batch], dtype=batch[0].total_charge_B.dtype
    )

    batch_atomic_A = Data(
        x=ZA_cat,
        edge_index=torch.vstack((e_AA_source_cat, e_AA_target_cat)),
        R=RA_cat,
        molecule_ind=molecule_ind_A,
        total_charge=total_charge_A_tensor,
        natom_per_mol=natom_per_mol_A,
    )

    batch_atomic_B = Data(
        x=ZB_cat,
        edge_index=torch.vstack((e_BB_source_cat, e_BB_target_cat)),
        R=RB_cat,
        molecule_ind=molecule_ind_B,
        total_charge=total_charge_B_tensor,
        natom_per_mol=natom_per_mol_B,
    )

    e_ABsr_source_cat = torch.cat(local_e_ABsr_source, dim=0)
    e_ABsr_target_cat = torch.cat(local_e_ABsr_target, dim=0)
    e_ABlr_source_cat = torch.cat(local_e_ABlr_source, dim=0)
    e_ABlr_target_cat = torch.cat(local_e_ABlr_target, dim=0)

    # Create full edge lists (short-range + long-range)
    e_ABfull_source = torch.cat([e_ABsr_source_cat, e_ABlr_source_cat], dim=0)
    e_ABfull_target = torch.cat([e_ABsr_target_cat, e_ABlr_target_cat], dim=0)

    # Create dimer_ind_full for full edge list
    dimer_ind_cat = torch.cat([data.dimer_ind for data in batch], dim=0)
    dimer_ind_lr_cat = torch.cat([data.dimer_ind_lr for data in batch], dim=0)
    dimer_ind_full = torch.cat([dimer_ind_cat, dimer_ind_lr_cat], dim=0)

    batched_data = Data(
        ZA=ZA_cat,
        RA=RA_cat,
        ZB=ZB_cat,
        RB=RB_cat,
        e_AA_source=e_AA_source_cat,
        e_AA_target=e_AA_target_cat,
        e_BB_source=e_BB_source_cat,
        e_BB_target=e_BB_target_cat,
        molecule_ind_A=molecule_ind_A,
        molecule_ind_B=molecule_ind_B,
        natom_per_mol_A=natom_per_mol_A,
        natom_per_mol_B=natom_per_mol_B,
        e_ABsr_source=e_ABsr_source_cat,
        e_ABsr_target=e_ABsr_target_cat,
        e_ABlr_source=e_ABlr_source_cat,
        e_ABlr_target=e_ABlr_target_cat,
        e_ABfull_source=e_ABfull_source,
        e_ABfull_target=e_ABfull_target,
        dimer_ind=dimer_ind_cat,
        dimer_ind_lr=dimer_ind_lr_cat,
        dimer_ind_full=dimer_ind_full,
        total_charge_A=total_charge_A_tensor,
        total_charge_B=total_charge_B_tensor,
        indA=torch.cat(local_indA, dim=0),
        indB=torch.cat(local_indB, dim=0),
        batch_atomic_A=batch_atomic_A,
        batch_atomic_B=batch_atomic_B,
    )
    return batched_data


def ap2_fused_collate_update_no_target_monomer_indices(batch):
    """
    Merge a list of fused-dimer Data objects (no target) into a single batched Data with unique global atom and edge indices and explicit per-monomer molecule indices.

    Parameters:
        batch (list[torch_geometric.data.Data]): List of per-dimer Data objects containing atomic features, intra- and inter-monomer edge indices, per-dimer indices, and monomer properties.

    Returns:
        torch_geometric.data.Data: A single Data object containing concatenated/offset fields:
            - ZA, RA, ZB, RB: concatenated atomic numbers and positions for monomers A and B.
            - e_AA_source, e_AA_target, e_BB_source, e_BB_target: batched intra-monomer edge index components.
            - e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target: batched short- and long-range inter-monomer edge index components.
            - dimer_ind, dimer_ind_lr: per-edge dimer membership indices for short- and long-range AB edges.
            - total_charge_A, total_charge_B: per-dimer total charges for monomers A and B.
            - qA, muA, quadA, hlistA, qB, muB, quadB, hlistB: concatenated per-monomer properties.
            - indA, indB (molecule_ind): per-atom tensors mapping each atom to its originating dimer (monomer index).
            - batch_atomic_A, batch_atomic_B: Data sub-objects representing the batched atomic graphs for monomers A and B (with x, edge_index, R, molecule_ind, total_charge).
    """
    monA_edge_offset, monB_edge_offset = 0, 0
    local_e_ABsr_source = []
    local_e_ABsr_target = []
    local_e_ABlr_source = []
    local_e_ABlr_target = []
    local_e_AA_source = []
    local_e_AA_target = []
    local_e_BB_source = []
    local_e_BB_target = []
    local_indA = []
    local_indB = []
    for i, data in enumerate(batch):
        data.dimer_ind = (
            torch.ones(data.e_ABsr_source.size(0), dtype=data.dimer_ind.dtype) * i
        )
        data.dimer_ind_lr = (
            torch.ones(data.e_ABlr_source.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
        local_e_ABsr_source.append(data.e_ABsr_source.clone() + monA_edge_offset)
        local_e_ABsr_target.append(data.e_ABsr_target.clone() + monB_edge_offset)
        local_e_ABlr_source.append(data.e_ABlr_source.clone() + monA_edge_offset)
        local_e_ABlr_target.append(data.e_ABlr_target.clone() + monB_edge_offset)
        local_e_AA_source.append(data.e_AA_source.clone() + monA_edge_offset)
        local_e_AA_target.append(data.e_AA_target.clone() + monA_edge_offset)
        local_e_BB_source.append(data.e_BB_source.clone() + monB_edge_offset)
        local_e_BB_target.append(data.e_BB_target.clone() + monB_edge_offset)

        monA_edge_offset += data.RA.size(0)
        monB_edge_offset += data.RB.size(0)
        local_indA.append(torch.ones(data.RA.size(0), dtype=data.dimer_ind.dtype) * i)
        local_indB.append(
            torch.ones(data.RB.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
    ZA_cat = torch.cat([data.ZA for data in batch], dim=0)
    RA_cat = torch.cat([data.RA for data in batch], dim=0)
    ZB_cat = torch.cat([data.ZB for data in batch], dim=0)
    RB_cat = torch.cat([data.RB for data in batch], dim=0)
    e_AA_source_cat = torch.cat(local_e_AA_source, dim=0)
    e_AA_target_cat = torch.cat(local_e_AA_target, dim=0)
    e_BB_source_cat = torch.cat(local_e_BB_source, dim=0)
    e_BB_target_cat = torch.cat(local_e_BB_target, dim=0)
    indA_cat = torch.cat(local_indA, dim=0)
    indB_cat = torch.cat(local_indB, dim=0)
    total_charge_A_tensor = torch.tensor(
        [data.total_charge_A for data in batch], dtype=batch[0].total_charge_A.dtype
    )
    total_charge_B_tensor = torch.tensor(
        [data.total_charge_B for data in batch], dtype=batch[0].total_charge_B.dtype
    )

    batch_atomic_A = Data(
        x=ZA_cat,
        edge_index=torch.vstack((e_AA_source_cat, e_AA_target_cat)),
        R=RA_cat,
        molecule_ind=indA_cat,
        total_charge=total_charge_A_tensor,
    )

    batch_atomic_B = Data(
        x=ZB_cat,
        edge_index=torch.vstack((e_BB_source_cat, e_BB_target_cat)),
        R=RB_cat,
        molecule_ind=indB_cat,
        total_charge=total_charge_B_tensor,
    )

    batched_data = Data(
        ZA=ZA_cat,
        RA=RA_cat,
        ZB=ZB_cat,
        RB=RB_cat,
        e_AA_source=e_AA_source_cat,
        e_AA_target=e_AA_target_cat,
        e_BB_source=e_BB_source_cat,
        e_BB_target=e_BB_target_cat,
        e_ABsr_source=torch.cat(local_e_ABsr_source, dim=0),
        e_ABsr_target=torch.cat(local_e_ABsr_target, dim=0),
        e_ABlr_source=torch.cat(local_e_ABlr_source, dim=0),
        e_ABlr_target=torch.cat(local_e_ABlr_target, dim=0),
        dimer_ind=torch.cat([data.dimer_ind for data in batch], dim=0),
        dimer_ind_lr=torch.cat([data.dimer_ind_lr for data in batch], dim=0),
        total_charge_A=total_charge_A_tensor,
        total_charge_B=total_charge_B_tensor,
        qA=torch.cat([data.qA for data in batch], dim=0),
        muA=torch.cat([data.muA for data in batch], dim=0),
        quadA=torch.cat([data.quadA for data in batch], dim=0),
        hlistA=torch.cat([data.hlistA for data in batch], dim=0),
        qB=torch.cat([data.qB for data in batch], dim=0),
        muB=torch.cat([data.muB for data in batch], dim=0),
        quadB=torch.cat([data.quadB for data in batch], dim=0),
        hlistB=torch.cat([data.hlistB for data in batch], dim=0),
        indA=indA_cat,
        indB=indB_cat,
        batch_atomic_A=batch_atomic_A,
        batch_atomic_B=batch_atomic_B,
    )
    return batched_data


def ap3_fused_collate_update(batch):
    """
    Batch-collate a list of fused dimer Data objects into a single batched Data with adjusted intra- and inter-monomer indices.

    Parameters:
        batch (list[Data]): List of per-dimer PyG Data objects containing atom features, coordinates, intra-monomer edges (e_AA*, e_BB*), inter-monomer edges (e_ABsr_*, e_ABlr_*), dimer index fields, charges, and target `y`.

    Returns:
        Data: A single PyG Data object that contains:
            - concatenated atom features and coordinates for monomers A and B (ZA, RA, ZB, RB)
            - adjusted intra-monomer edge lists (e_AA_source/target, e_BB_source/target)
            - concatenated short- and long-range inter-monomer edge lists (e_ABsr_*, e_ABlr_*)
            - full inter-monomer edge lists (e_ABfull_source, e_ABfull_target)
            - per-edge dimer indices (dimer_ind, dimer_ind_lr, dimer_ind_full)
            - per-monomer molecule indices and counts (molecule_ind_A/B, natom_per_mol_A/B)
            - total charges per monomer (total_charge_A/B)
            - stacked targets `y`
            - nested batch_atomic_A and batch_atomic_B Data objects for the A and B atom pools
    """
    monA_edge_offset, monB_edge_offset = 0, 0
    local_e_ABsr_source = []
    local_e_ABsr_target = []
    local_e_ABlr_source = []
    local_e_ABlr_target = []
    local_e_AA_source = []
    local_e_AA_target = []
    local_e_BB_source = []
    local_e_BB_target = []
    for i, data in enumerate(batch):
        data.dimer_ind = (
            torch.ones(data.e_ABsr_source.size(0), dtype=data.dimer_ind.dtype) * i
        )
        data.dimer_ind_lr = (
            torch.ones(data.e_ABlr_source.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
        data.dimer_ind_full = (
            torch.ones(
                data.e_ABsr_source.size(0) + data.e_ABlr_source.size(0),
                dtype=data.dimer_ind_lr.dtype,
            )
            * i
        )
        local_e_ABsr_source.append(data.e_ABsr_source.clone() + monA_edge_offset)
        local_e_ABsr_target.append(data.e_ABsr_target.clone() + monB_edge_offset)
        local_e_ABlr_source.append(data.e_ABlr_source.clone() + monA_edge_offset)
        local_e_ABlr_target.append(data.e_ABlr_target.clone() + monB_edge_offset)
        local_e_AA_source.append(data.e_AA_source.clone() + monA_edge_offset)
        local_e_AA_target.append(data.e_AA_target.clone() + monA_edge_offset)
        local_e_BB_source.append(data.e_BB_source.clone() + monB_edge_offset)
        local_e_BB_target.append(data.e_BB_target.clone() + monB_edge_offset)
        monA_edge_offset += data.RA.size(0)
        monB_edge_offset += data.RB.size(0)
    ZA_cat = torch.cat([data.ZA for data in batch], dim=0)
    RA_cat = torch.cat([data.RA for data in batch], dim=0)
    ZB_cat = torch.cat([data.ZB for data in batch], dim=0)
    RB_cat = torch.cat([data.RB for data in batch], dim=0)
    e_AA_source_cat = torch.cat(local_e_AA_source, dim=0)
    e_AA_target_cat = torch.cat(local_e_AA_target, dim=0)
    e_BB_source_cat = torch.cat(local_e_BB_source, dim=0)
    e_BB_target_cat = torch.cat(local_e_BB_target, dim=0)
    total_charge_A_tensor = torch.tensor(
        [data.total_charge_A for data in batch], dtype=batch[0].total_charge_A.dtype
    )
    total_charge_B_tensor = torch.tensor(
        [data.total_charge_B for data in batch], dtype=batch[0].total_charge_B.dtype
    )
    molecule_ind_A = torch.cat(
        [
            torch.ones(data.RA.size(0), dtype=torch.long) * i
            for i, data in enumerate(batch)
        ],
        dim=0,
    )
    molecule_ind_B = torch.cat(
        [
            torch.ones(data.RB.size(0), dtype=torch.long) * i
            for i, data in enumerate(batch)
        ],
        dim=0,
    )
    natom_per_mol_A = torch.tensor(
        [data.RA.size(0) for data in batch], dtype=torch.long
    )
    natom_per_mol_B = torch.tensor(
        [data.RB.size(0) for data in batch], dtype=torch.long
    )

    batch_atomic_A = Data(
        x=ZA_cat,
        edge_index=torch.vstack((e_AA_source_cat, e_AA_target_cat)),
        R=RA_cat,
        molecule_ind=molecule_ind_A,
        total_charge=total_charge_A_tensor,
        natom_per_mol=natom_per_mol_A,
    )

    batch_atomic_B = Data(
        x=ZB_cat,
        edge_index=torch.vstack((e_BB_source_cat, e_BB_target_cat)),
        R=RB_cat,
        molecule_ind=molecule_ind_B,
        total_charge=total_charge_B_tensor,
        natom_per_mol=natom_per_mol_B,
    )

    e_ABsr_source_cat = torch.cat(local_e_ABsr_source, dim=0)
    e_ABsr_target_cat = torch.cat(local_e_ABsr_target, dim=0)
    e_ABlr_source_cat = torch.cat(local_e_ABlr_source, dim=0)
    e_ABlr_target_cat = torch.cat(local_e_ABlr_target, dim=0)

    e_ABfull_source = torch.cat([e_ABsr_source_cat, e_ABlr_source_cat], dim=0)
    e_ABfull_target = torch.cat([e_ABsr_target_cat, e_ABlr_target_cat], dim=0)

    dimer_ind_cat = torch.cat([data.dimer_ind for data in batch], dim=0)
    dimer_ind_lr_cat = torch.cat([data.dimer_ind_lr for data in batch], dim=0)
    dimer_ind_full = torch.cat([data.dimer_ind_full for data in batch], dim=0)

    batched_data = Data(
        ZA=ZA_cat,
        RA=RA_cat,
        ZB=ZB_cat,
        RB=RB_cat,
        e_AA_source=e_AA_source_cat,
        e_AA_target=e_AA_target_cat,
        e_BB_source=e_BB_source_cat,
        e_BB_target=e_BB_target_cat,
        molecule_ind_A=molecule_ind_A,
        molecule_ind_B=molecule_ind_B,
        natom_per_mol_A=natom_per_mol_A,
        natom_per_mol_B=natom_per_mol_B,
        e_ABsr_source=e_ABsr_source_cat,
        e_ABsr_target=e_ABsr_target_cat,
        e_ABlr_source=e_ABlr_source_cat,
        e_ABlr_target=e_ABlr_target_cat,
        e_ABfull_source=e_ABfull_source,
        e_ABfull_target=e_ABfull_target,
        dimer_ind=dimer_ind_cat,
        dimer_ind_lr=dimer_ind_lr_cat,
        dimer_ind_full=dimer_ind_full,
        total_charge_A=total_charge_A_tensor,
        total_charge_B=total_charge_B_tensor,
        y=torch.stack([data.y for data in batch], dim=0),
        batch_atomic_A=batch_atomic_A,
        batch_atomic_B=batch_atomic_B,
    )
    return batched_data


class APNet2_fused_DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        collate_fn=None,  # atomic_collate_update,
        **kwargs,
    ):
        if collate_fn is None:
            # Save for PyTorch Lightning < 1.6:
            self.follow_batch = follow_batch
            self.exclude_keys = exclude_keys

            self.collator = atomic_datasets.Collater(
                dataset, follow_batch, exclude_keys
            )
            self.collate_fn = self.collator.collate_fn
        else:
            self.collate_fn = collate_fn

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )


def ap2_fused_setup(molA_data, molB_data, atom_model, r_cut, r_cut_im, index=0):
    batch_A = atomic_datasets.atomic_collate_update_no_target(molA_data)
    qAs, muAs, thAs, hlistAs = atom_model.predict_multipoles_batch(batch_A)
    batch_B = atomic_datasets.atomic_collate_update_no_target(molB_data)
    qBs, muBs, thBs, hlistBs = atom_model.predict_multipoles_batch(batch_B)
    dimer_data = []
    for j in range(len(molA_data)):
        atomic_props_A = molA_data[j]
        atomic_props_B = molB_data[j]
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
        e_AA_source, e_AA_target = pairwise_edges(atomic_props_A.R, r_cut)
        e_BB_source, e_BB_target = pairwise_edges(atomic_props_B.R, r_cut)
        e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = pairwise_edges_im(
            atomic_props_A.R, atomic_props_B.R, r_cut_im
        )
        dimer_ind = torch.ones((1), dtype=torch.long) * index
        data = Data(
            ZA=atomic_props_A.x.long(),
            RA=atomic_props_A.R,
            ZB=atomic_props_B.x.long(),
            RB=atomic_props_B.R,
            # short range, intermolecular edges
            e_ABsr_source=e_ABsr_source.long(),
            e_ABsr_target=e_ABsr_target.long(),
            dimer_ind=dimer_ind.long(),
            # long range, intermolecular edges
            e_ABlr_source=e_ABlr_source.long(),
            e_ABlr_target=e_ABlr_target.long(),
            dimer_ind_lr=dimer_ind.long(),
            # intramonomer edges (monomer A)
            e_AA_source=e_AA_source.long(),
            e_AA_target=e_AA_target.long(),
            # intramonomer edges (monomer B)
            e_BB_source=e_BB_source.long(),
            e_BB_target=e_BB_target.long(),
            # monomer charges
            total_charge_A=atomic_props_A.total_charge,
            total_charge_B=atomic_props_B.total_charge,
            # monomer A properties
            qA=qA,
            muA=muA,
            quadA=quadA,
            hlistA=hlistA,
            # monomer B properties
            qB=qB,
            muB=muB,
            quadB=quadB,
            hlistB=hlistB,
        )
        data = data.cpu()
        dimer_data.append(data)
    return dimer_data


def save_hdf5_data_objects(data_objects, filepath):
    """
    Save a list of PyG Data objects into an HDF5 file where each object is stored as a separate group.

    Each data object is written to a group named "data_<index>". The following attributes, when present on a data object, are persisted: ZA, RA, ZB, RB, e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target, e_AA_source, e_AA_target, e_BB_source, e_BB_target, dimer_ind, dimer_ind_lr, molecule_ind_A, molecule_ind_B, total_charge_A, total_charge_B, qA, muA, quadA, hlistA, qB, muB, quadB, hlistB, and y. Tensor attributes are converted to NumPy arrays and saved as datasets; numeric scalar attributes (int/float) are saved as group attributes.

    Parameters:
        data_objects (Sequence): Sequence of PyG Data-like objects to save. Objects should expose the attributes listed above when applicable.
        filepath (str or os.PathLike): Destination HDF5 file path.

    """
    with h5py.File(filepath, "w") as f:
        for i, data_obj in enumerate(data_objects):
            group = f.create_group(f"data_{i}")
            # Save essential tensor and scalar attributes
            essential_attrs = [
                "ZA",
                "RA",
                "ZB",
                "RB",
                "e_ABsr_source",
                "e_ABsr_target",
                "e_ABlr_source",
                "e_ABlr_target",
                "e_AA_source",
                "e_AA_target",
                "e_BB_source",
                "e_BB_target",
                "dimer_ind",
                "dimer_ind_lr",
                "molecule_ind_A",
                "molecule_ind_B",
                "total_charge_A",
                "total_charge_B",
                "qA",
                "muA",
                "quadA",
                "hlistA",
                "qB",
                "muB",
                "quadB",
                "hlistB",
                "y",  # Essential for training
            ]
            for attr_name in essential_attrs:
                if hasattr(data_obj, attr_name):
                    attr_value = getattr(data_obj, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        group.create_dataset(attr_name, data=attr_value.numpy())
                    elif isinstance(attr_value, (int, float)):
                        group.attrs[attr_name] = attr_value


def load_hdf5_data_objects(filepath):  # type: ignore
    """
    Load a list of PyG Data objects stored in an HDF5 file.

    Reads groups whose names start with "data_" (sorted lexicographically). For each such group, datasets are converted to torch tensors (array datasets become tensor arrays; scalar datasets become 0-dim tensors) and group attributes are copied as scalar fields. Each group's combined data is used to construct a torch_geometric.data.Data object; the function returns the list of these Data objects in the sorted group order.

    Parameters:
        filepath (str or os.PathLike): Path to the HDF5 file to read.

    Returns:
        list[torch_geometric.data.Data]: List of reconstructed Data objects.
    """
    data_objects = []
    with h5py.File(filepath, "r") as f:
        for key in sorted(f.keys()):
            if key.startswith("data_"):
                group = f[key]
                data_dict = {}
                # Load datasets (tensor data)
                for ds_name in group.keys():
                    try:
                        # Try to load as array first
                        data_dict[ds_name] = torch.from_numpy(group[ds_name][:])
                    except ValueError:
                        # If that fails, it's a scalar
                        data_dict[ds_name] = torch.tensor(group[ds_name][()])
                # Load attributes (scalar data)
                for attr_name, attr_value in group.attrs.items():
                    data_dict[attr_name] = attr_value
                data_objects.append(Data(**data_dict))
    return data_objects


class ap2_fused_module_dataset(Dataset):
    split_spec_types = AP2_FUSED_SPLIT_SPEC_TYPES

    @classmethod
    def is_split_db_config(cls, spec_type, qcel_molecules=None):
        return spec_type_uses_split_files(spec_type) or qcel_inputs_are_split_db(
            qcel_molecules
        )

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=1,
        max_size=None,
        force_reprocess=False,
        skip_processed=True,
        skip_compile=False,
        # only need for processing
        atom_model_path=None,
        atom_model=None,
        batch_size=16,
        atomic_batch_size=200,
        # DO NOT CHANGE UNLESS YOU WANT TO RE-PROCESS THE DATASET
        datapoint_storage_n_objects=1000,
        in_memory=False,
        num_devices=1,
        split="all",  # train, test
        print_level=1,
        qcel_molecules: Optional[List[qcel.models.Molecule]] = None,
        energy_labels: Optional[List[float]] = None,
        random_seed=42,
        check_monomer_validity=True,
        storage_type="pt",  # "pt" or "h5" for storage format
    ):
        """
        spec_type definitions:
            1. regular
            2. AP2 paper train/test split
            5. testing small
            6. testing 12k
            7. testing 12k but creating batch of 16 to avoid any collating and reduce large I/O issues (potentially)
            None: assumes that data is passed as qcel_molecules and energy labels
        """
        self.print_level = print_level
        try:
            assert spec_type in [1, 2, 5, 6, 7, 8, 9, None]
        except Exception:
            print("Currently spec_type must be 1 or 2 for SAPT0/jun-cc-pVDZ")
            raise ValueError
        self.spec_type = spec_type

        # Validate storage_type
        if storage_type not in ["pt", "h5"]:
            raise ValueError("storage_type must be 'pt' or 'h5'")
        self.storage_type = storage_type

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
                f"Received {len(qcel_molecules)}"
                " QCElemental molecules with energy labels"
            )

        self.MAX_SIZE = max_size
        self.random_seed = random_seed
        self.in_memory = in_memory
        self.split = split
        self.split_db = self.is_split_db_config(self.spec_type, self.qcel_molecules)
        self.r_cut = r_cut
        self.r_cut_im = r_cut_im
        self.force_reprocess = force_reprocess
        self.atomic_batch_size = atomic_batch_size
        self.check_monomer_validity = check_monomer_validity
        self.batch_size = batch_size
        self.training_batch_size = batch_size
        self.datapoint_storage_n_objects = datapoint_storage_n_objects
        self.points_per_file = self.datapoint_storage_n_objects
        self.skip_compile = skip_compile
        if self.in_memory:
            self.points_per_file = 1
        self.data = []
        self.skip_processed = skip_processed
        if os.path.exists(root) is False:
            os.makedirs(root, exist_ok=True)
        if atom_model is not None:
            if isinstance(atom_model, AtomModel):
                self.atom_model = atom_model
            else:
                self.atom_model = AtomModel(
                    ds_root=None,
                    ignore_database_null=True,
                )
                self.atom_model.model = atom_model
            if not skip_compile:
                self.atom_model.model = torch.compile(
                    self.atom_model.model, dynamic=True
                )
        elif not self.skip_processed:
            if atom_model_path is None:
                atom_model_path = resolve_pretrained_path("am_ensemble/am_0.pt")
            self.atom_model = AtomModel(
                pre_trained_model_path=atom_model_path,
                ds_root=None,
                ignore_database_null=True,
            )
            self.atom_model.model.to(self.atom_model.device)
            torch._dynamo.config.dynamic_shapes = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.capture_scalar_outputs = True
            if not skip_compile:
                self.atom_model.model = torch.compile(
                    self.atom_model.model, dynamic=True
                )
        print(f"{root=}, {self.spec_type=}, {self.in_memory=}")
        super(ap2_fused_module_dataset, self).__init__(root, transform, pre_transform)
        if self.force_reprocess:
            self.force_reprocess = False
            super(ap2_fused_module_dataset, self).__init__(
                root, transform, pre_transform
            )
        if self.in_memory:
            self.get = self.get_in_memory
        self.batch_size = batch_size
        self.active_idx_data = None
        self.active_data = None

    @property
    def file_extension(self):
        """Return the file extension based on storage type"""
        return ".h5" if self.storage_type == "h5" else ".pt"

    @property
    def split_name(self):
        return f"_{self.split}" if self.split_db and self.split != "all" else ""

    @property
    def raw_file_names(self):
        # TODO: enable users to specify data source via QCArchive, url, or local file
        # spec_1 = "spec_1" # 'SAPT0/jun-cc-pVDZ'
        if self.spec_type == 2:
            return [
                "1600K_train_dimers-fixed.pkl",
                "1600K_test_dimers-fixed.pkl",
            ]
        elif self.spec_type == 5:
            return [
                "t_train.pkl",
                "t_test.pkl",
            ]
        elif self.spec_type == 6:
            return [
                "t_train10k.pkl",
                "t_test2k.pkl",
            ]
        elif self.spec_type == 7:
            return [
                "t_train_100.pkl",
                "t_test_20.pkl",
            ]
        elif self.spec_type == 8:
            return [
                "t_val_19.pkl",
            ]
        elif self.spec_type == 9:
            return [
                "t_train_19.pkl",
                "t_test_19.pkl",
            ]
        elif self.spec_type is None:
            os.system(f"touch {self.raw_dir}/tmp.txt")
            return ["tmp.txt"]
        else:
            return [
                "splinter_spec1.pkl",
            ]

    def reprocess_file_names(self):
        """
        Determine which processed file names should be used (or signal that reprocessing is required).

        If force_reprocess is True, returns ["file"]. Otherwise, searches the processed directory for files matching the dataset split and spec_type, sorts them using natural_key, and trims the list according to MAX_SIZE and datapoint_storage_n_objects. If matching files are found, returns the ordered list of file basenames; if none are found, returns a single marker filename indicating missing data (e.g., "dimer_missing" plus the file extension).

        Returns:
            list[str]: Ordered list of processed file basenames to use, or a single-element list
            signaling reprocessing (either ["file"] for forced reprocess or ["dimer_missing{ext}"] when no files exist).
        """
        if self.force_reprocess:
            return ["file"]
        else:
            if self.split_db and self.split == "train":
                file_cmd = f"{self.root}/processed/dimer_ap2_fused_train_spec_{
                    self.spec_type
                }_*{self.file_extension}"
            elif self.split_db and self.split == "test":
                file_cmd = f"{self.root}/processed/dimer_ap2_fused_test_spec_{
                    self.spec_type
                }_*{self.file_extension}"
            else:
                file_cmd = f"{self.root}/processed/dimer_ap2_fused_spec_{self.spec_type}_*{self.file_extension}"
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                # want to preserve idx ordering
                spec_files.sort(key=natural_key)
                if self.MAX_SIZE is not None:
                    max_size = int(self.MAX_SIZE / self.datapoint_storage_n_objects)
                    # if max_size == 0:
                    #     raise ValueError(
                    #         "MAX_SIZE must be greater than datapoint_storage_n_objects"
                    #     )
                if self.MAX_SIZE is not None:
                    if len(spec_files) > max_size and max_size > 0:
                        spec_files = spec_files[:max_size]
                    elif len(spec_files) > max_size:
                        spec_files = spec_files[:1]
                return spec_files
            else:
                # Forces a re-processing of the dataset
                return [f"dimer_missing{self.file_extension}"]

    @property
    def processed_file_names(self):
        return self.reprocess_file_names()

    def download(self):
        if self.energy_labels and self.qcel_molecules:
            return

        print(
            "Downloading Splinter dataset of ~1.6M Dimers. This might take a while..."
        )
        splinter_spec_1_files = [
            "https://figshare.com/ndownloader/files/39449167",
            "https://figshare.com/ndownloader/files/40271983",
            "https://figshare.com/ndownloader/files/40271989",
            "https://figshare.com/ndownloader/files/40272001",
            "https://figshare.com/ndownloader/files/40552931",
            "https://figshare.com/ndownloader/files/40272022",
            "https://figshare.com/ndownloader/files/40272040",
            "https://figshare.com/ndownloader/files/40272052",
            "https://figshare.com/ndownloader/files/40272061",
            "https://figshare.com/ndownloader/files/40272064",
        ]
        for n, i in enumerate(splinter_spec_1_files):
            download_url(
                i,
                self.raw_dir,
                filename=f"splinter_spec1_{n}.tar.gz",
            )
        if not os.path.exists(f"{self.raw_dir}/dimerpairs"):
            for i in range(len(splinter_spec_1_files)):
                with tarfile.open(f"{self.raw_dir}/splinter_spec1_{i}.tar.gz") as tar:
                    tar.extractall(self.raw_dir)
        return

    def process(self):
        """
        Builds fused PyG Data objects from raw dimers or provided QCElemental molecules and writes them to disk or stores them in memory.

        Processes either self.qcel_molecules with self.energy_labels or files listed in self.raw_paths, converts each dimer into a fused Data object via dimer_fused_data (respecting r_cut, r_cut_im, and monomer-validity checks), applies self.pre_filter when provided, and accumulates data objects in chunks of self.points_per_file. Depending on configuration, the method:
        - appends chunks to self.data when in_memory is True, or
        - saves chunks to processed files under self.processed_dir using either HDF5 or PT format per self.storage_type.

        Behavioral details:
        - Honors self.MAX_SIZE to limit the number of processed dimers.
        - Skips already-processed files when self.skip_processed is True.
        - Uses self.print_level to control progress output.
        - Uses self.random_seed when loading raw datasets.
        - Populates self.data (when in_memory) and creates processed files named like dimer_ap2_fused_{split_name}_spec_{spec_type}_{file_index}{file_extension}.
        - Skips dimers for which dimer_fused_data returns None or that fail self.pre_filter.
        """
        self.data = []
        idx = 0
        data_objects = []
        # Handle direct qcel_mols input
        RAs, RBs, ZAs, ZBs, TQAs, TQBs, targets = [], [], [], [], [], [], []
        if self.qcel_molecules is not None and self.energy_labels is not None:
            print("Processing directly from provided QCElemental molecules...")
            split_name = self.split_name

            # Process directly from qcel_mols and energy_labels
            for mol in self.qcel_molecules:
                # Extract monomer data from dimer
                monA, monB = mol.get_fragment(0), mol.get_fragment(1)

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
            targets = self.energy_labels

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
                if self.split_db:
                    split_name = self.split_name
                    print(f"{split_name=}")
                    if self.split not in Path(raw_path).stem:
                        print(f"{self.split} is skipping {raw_path}")
                        continue
                print(f"raw_path: {raw_path}")
                print("Loading dimers...")
                RA, RB, ZA, ZB, TQA, TQB, target = util.load_dimer_dataset(
                    raw_path,
                    self.MAX_SIZE,
                    return_qcel_mols=False,
                    return_qcel_mons=False,
                    columns=["Elst_aug", "Exch_aug", "Ind_aug", "Disp_aug"],
                    random_seed_shuffle=self.random_seed,
                )
                RAs.extend(RA)
                RBs.extend(RB)
                ZAs.extend(ZA)
                ZBs.extend(ZB)
                TQAs.extend(TQA)
                TQBs.extend(TQB)
                targets.extend(target)
        print("Creating data objects...")
        t1 = time()
        t2 = time()
        print(f"{len(RAs)=}, {self.atomic_batch_size=}, {self.batch_size=}")
        for i in range(len(RAs)):
            if self.skip_processed:
                datapath = osp.join(
                    self.processed_dir,
                    f"dimer_ap2_fused{split_name}_spec_{self.spec_type}_{
                        idx // self.points_per_file
                    }{self.file_extension}",
                )
                if osp.exists(datapath):
                    idx += 1
                    continue
            y = torch.tensor(targets[i], dtype=torch.float32)
            data = dimer_fused_data(
                RAs[i],
                ZAs[i],
                TQAs[i],
                RBs[i],
                ZBs[i],
                TQBs[i],
                dimer_ind=i,
                r_cut=self.r_cut,
                r_cut_im=self.r_cut_im,
                check_validity=self.check_monomer_validity,
                y=y,
            )
            if data is None:
                print(data)
                print(f"Skipping invalid dimer index {i}")
                continue
            data = data.cpu()
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_objects.append(data)
            # Normally would store the data object to individual files,
            # but at 1.67M dimers, this is too many files. Need to
            # store self.datapoint_storage_n_objects (like 1000) dimers per file
            if len(data_objects) == self.points_per_file:
                if self.in_memory:
                    data_objects = data_objects[0]
                if self.in_memory:
                    self.data.append(data_objects)
                else:
                    datapath = osp.join(
                        self.processed_dir,
                        f"dimer_ap2_fused{split_name}_spec_{self.spec_type}_{
                            idx // self.points_per_file
                        }{self.file_extension}",
                    )
                    if self.print_level >= 2:
                        print(f"Saving to {datapath}")
                        print(len(data_objects))
                    if self.storage_type == "h5":
                        save_hdf5_data_objects(data_objects, datapath)
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
        if len(data_objects) > 0:
            if self.in_memory:
                data_objects = data_objects[0]
            if self.in_memory:
                self.data.append(data_objects)
            else:
                datapath = osp.join(
                    self.processed_dir,
                    f"dimer_ap2_fused{split_name}_spec_{self.spec_type}_{
                        idx // self.points_per_file
                    }{self.file_extension}",
                )
                if self.print_level >= 2:
                    print(f"Final Saving to {datapath}")
                    print(len(data_objects))
                if self.storage_type == "h5":
                    save_hdf5_data_objects(data_objects, datapath)
                else:
                    torch.save(data_objects, datapath)
        return

    def len(self):
        if self.in_memory:
            return len(self.data)

        if self.storage_type == "h5":
            d = load_hdf5_data_objects(
                osp.join(self.processed_dir, self.processed_file_names[-1])
            )
        else:
            d = torch.load(
                osp.join(self.processed_dir, self.processed_file_names[-1]),
                weights_only=False,
            )
        return (
            len(self.processed_file_names) - 1
        ) * self.datapoint_storage_n_objects + len(d)

    def get(self, idx):
        idx_datapath = idx // self.datapoint_storage_n_objects
        obj_ind = idx % self.datapoint_storage_n_objects
        if self.active_idx_data == idx_datapath:
            return self.active_data[obj_ind]
        split_name = ""
        if self.split_db:
            split_name = self.split_name
        datapath = osp.join(
            self.processed_dir,
            f"dimer_ap2_fused{split_name}_spec_{self.spec_type}_{idx_datapath}{self.file_extension}",
        )
        if self.storage_type == "h5":
            self.active_data = load_hdf5_data_objects(datapath)
        else:
            self.active_data = torch.load(datapath, weights_only=False)
        try:
            self.active_data[obj_ind]
        except Exception:
            print(
                f"Error loading {datapath}\n    at {
                    idx = }, {idx_datapath = }, {obj_ind = }"
            )
        return self.active_data[obj_ind]

    def get_in_memory(self, idx):
        """Method for retrieving data when in_memory=True"""
        return self.data[idx]
