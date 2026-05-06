"""
AP3-fused FSAPT dataset for training models on fragment-based SAPT energies.

This dataset stores FSAPT breakdown energies (F-Electrostatics, F-Exchange,
F-Dispersion, F-Induction, F-Total) along with fragment indices (Frag1_indices,
Frag2_indices) to enable training AP3 fused models on fragment-level energies.

During training, the model predicts atomic-level contributions which are summed
according to the fragment indices to compute fragment-level energies for loss
calculation against the FSAPT reference values.
"""

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
from ..lmdb_utils import acquire_lmdb_env, release_lmdb_env
import tarfile
from time import time
from pathlib import Path
from importlib import resources
from apnet_pt import constants
from .ap3_fused_ds import dimer_fused_data, qcel_inputs_are_split_db
import json

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


def ap3_fused_fsapt_collate_update(batch):
    """
    AP3 collate function with precomputed classical energies subtracted from targets.
    If E_classical_elst and E_classical_ind are present in the data objects,
    they will be subtracted from y so the model learns the residual.
    """
    monA_edge_offset, monB_edge_offset = 0, 0
    local_e_ABsr_source = []
    local_e_ABsr_target = []
    local_e_ABlr_source = []
    local_e_ABlr_target = []
    local_e_ABfull_source = []
    local_e_ABfull_target = []

    local_e_AA_source = []
    local_e_AA_target = []
    local_e_BB_source = []
    local_e_BB_target = []

    local_frag1_ind = []
    local_frag2_ind = []

    has_precomputed = hasattr(batch[0], "E_classical_elst") and hasattr(
        batch[0], "E_classical_ind"
    )
    local_indA = []
    local_indB = []

    for i, data in enumerate(batch):
        data.dimer_ind = (
            torch.ones(data.e_ABsr_source.size(0), dtype=data.dimer_ind.dtype) * i
        )
        data.dimer_ind_lr = (
            torch.ones(data.e_ABlr_source.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
        data.dimer_ind_full = (
            torch.ones(
                data.e_ABlr_source.size(0) + data.e_ABsr_source.size(0),
                dtype=data.dimer_ind_lr.dtype,
            )
            * i
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

        local_frag1_ind.append(data.frag1_ind.clone() + monA_edge_offset)
        local_frag2_ind.append(data.frag2_ind.clone() + monB_edge_offset)

        local_e_ABfull_source.append(
            torch.cat(
                [
                    data.e_ABsr_source.clone() + monA_edge_offset,
                    data.e_ABlr_source.clone() + monA_edge_offset,
                ]
            )
        )
        local_e_ABfull_target.append(
            torch.cat(
                [
                    data.e_ABsr_target.clone() + monB_edge_offset,
                    data.e_ABlr_target.clone() + monB_edge_offset,
                ]
            )
        )

        local_e_AA_source.append(data.e_AA_source.clone() + monA_edge_offset)
        local_e_AA_target.append(data.e_AA_target.clone() + monA_edge_offset)
        local_e_BB_source.append(data.e_BB_source.clone() + monB_edge_offset)
        local_e_BB_target.append(data.e_BB_target.clone() + monB_edge_offset)

        monA_edge_offset += data.RA.size(0)
        monB_edge_offset += data.RB.size(0)

        local_indA.append(torch.ones(data.RA.size(0), dtype=torch.long) * i)
        local_indB.append(torch.ones(data.RB.size(0), dtype=torch.long) * i)

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
    # local_frag1_ind_cat = torch.cat(local_frag1_ind, dim=0)
    # local_frag2_ind_cat = torch.cat(local_frag2_ind, dim=0)

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

    e_ABfull_source = torch.cat(local_e_ABfull_source, dim=0)
    e_ABfull_target = torch.cat(local_e_ABfull_target, dim=0)

    # e_ABfull_source = torch.cat([e_ABsr_source_cat, e_ABlr_source_cat], dim=0)
    # e_ABfull_target = torch.cat([e_ABsr_target_cat, e_ABlr_target_cat], dim=0)

    dimer_ind_cat = torch.cat([data.dimer_ind for data in batch], dim=0)
    dimer_ind_lr_cat = torch.cat([data.dimer_ind_lr for data in batch], dim=0)
    dimer_ind_full_cat = torch.cat([data.dimer_ind_full for data in batch], dim=0)

    indA_cat = torch.cat(local_indA, dim=0)
    indB_cat = torch.cat(local_indB, dim=0)

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
        e_ABsr_source=e_ABsr_source_cat,
        e_ABsr_target=e_ABsr_target_cat,
        e_ABlr_source=e_ABlr_source_cat,
        e_ABlr_target=e_ABlr_target_cat,
        e_ABfull_source=e_ABfull_source,
        e_ABfull_target=e_ABfull_target,
        dimer_ind=dimer_ind_cat,
        dimer_ind_lr=dimer_ind_lr_cat,
        dimer_ind_full=dimer_ind_full_cat,
        total_charge_A=total_charge_A_tensor,
        total_charge_B=total_charge_B_tensor,
        batch_atomic_A=batch_atomic_A,
        batch_atomic_B=batch_atomic_B,
        frag1_ind=local_frag1_ind,
        frag2_ind=local_frag2_ind,
        indA=indA_cat,
        indB=indB_cat,
    )

    if has_precomputed:
        batched_data.E_classical_elst = torch.tensor(
            [data.E_classical_elst for data in batch]
        )
        batched_data.E_classical_ind = torch.tensor(
            [data.E_classical_ind for data in batch]
        )

    return batched_data


def ap3_fused_fsapt_collate_update_no_target(batch):
    monA_edge_offset, monB_edge_offset = 0, 0
    local_e_ABsr_source = []
    local_e_ABsr_target = []
    local_e_ABlr_source = []
    local_e_ABlr_target = []
    local_e_ABfull_source = []
    local_e_ABfull_target = []
    local_e_AA_source = []
    local_e_AA_target = []
    local_e_BB_source = []
    local_e_BB_target = []
    local_indA = []
    local_indB = []

    local_frag1_ind = []
    local_frag2_ind = []
    for i, data in enumerate(batch):
        data.dimer_ind = (
            torch.ones(data.e_ABsr_source.size(0), dtype=data.dimer_ind.dtype) * i
        )
        data.dimer_ind_lr = (
            torch.ones(data.e_ABlr_source.size(0), dtype=data.dimer_ind_lr.dtype) * i
        )
        data.dimer_ind_full = (
            torch.ones(
                data.e_ABlr_source.size(0) + data.e_ABsr_source.size(0),
                dtype=data.dimer_ind_lr.dtype,
            )
            * i
        )

        local_e_ABsr_source.append(data.e_ABsr_source.clone() + monA_edge_offset)
        local_e_ABsr_target.append(data.e_ABsr_target.clone() + monB_edge_offset)
        local_e_ABlr_source.append(data.e_ABlr_source.clone() + monA_edge_offset)
        local_e_ABlr_target.append(data.e_ABlr_target.clone() + monB_edge_offset)
        local_frag1_ind.append(data.frag1_ind.clone() + monA_edge_offset)
        local_frag2_ind.append(data.frag2_ind.clone() + monB_edge_offset)
        local_e_ABfull_source.append(
            torch.cat(
                [
                    data.e_ABsr_source.clone() + monA_edge_offset,
                    data.e_ABlr_source.clone() + monA_edge_offset,
                ]
            )
        )
        local_e_ABfull_target.append(
            torch.cat(
                [
                    data.e_ABsr_target.clone() + monB_edge_offset,
                    data.e_ABlr_target.clone() + monB_edge_offset,
                ]
            )
        )
        local_e_AA_source.append(data.e_AA_source.clone() + monA_edge_offset)
        local_e_AA_target.append(data.e_AA_target.clone() + monA_edge_offset)
        local_e_BB_source.append(data.e_BB_source.clone() + monB_edge_offset)
        local_e_BB_target.append(data.e_BB_target.clone() + monB_edge_offset)
        monA_edge_offset += data.RA.size(0)
        monB_edge_offset += data.RB.size(0)
        local_indA.append(torch.ones(data.RA.size(0), dtype=torch.long) * i)
        local_indB.append(torch.ones(data.RB.size(0), dtype=torch.long) * i)
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

    e_ABfull_source = torch.cat(local_e_ABfull_source, dim=0)
    e_ABfull_target = torch.cat(local_e_ABfull_target, dim=0)

    dimer_ind_cat = torch.cat([data.dimer_ind for data in batch], dim=0)
    dimer_ind_lr_cat = torch.cat([data.dimer_ind_lr for data in batch], dim=0)
    dimer_ind_full = torch.cat([data.dimer_ind_full for data in batch], dim=0)

    local_frag1_ind_cat = torch.cat(local_frag1_ind, dim=0)
    local_frag2_ind_cat = torch.cat(local_frag2_ind, dim=0)

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
        indA=indA_cat,
        indB=indB_cat,
        batch_atomic_A=batch_atomic_A,
        batch_atomic_B=batch_atomic_B,
        frag1_ind=local_frag1_ind,
        frag2_ind=local_frag2_ind,
    )
    return batched_data


def ap3_fused_collate_update_no_target_monomer_indices(batch):
    monA_edge_offset, monB_edge_offset = 0, 0
    local_e_ABsr_source = []
    local_e_ABsr_target = []
    local_e_ABlr_source = []
    local_e_ABlr_target = []
    local_e_ABfull_source = []
    local_e_ABfull_target = []
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
        data.dimer_ind_full = (
            torch.ones(
                data.e_ABlr_source.size(0) + data.e_ABsr_source.size(0),
                dtype=data.dimer_ind_lr.dtype,
            )
            * i
        )
        local_e_ABsr_source.append(data.e_ABsr_source.clone() + monA_edge_offset)
        local_e_ABsr_target.append(data.e_ABsr_target.clone() + monB_edge_offset)
        local_e_ABlr_source.append(data.e_ABlr_source.clone() + monA_edge_offset)
        local_e_ABlr_target.append(data.e_ABlr_target.clone() + monB_edge_offset)
        local_e_ABfull_source.append(
            torch.cat(
                [
                    data.e_ABsr_source.clone() + monA_edge_offset,
                    data.e_ABlr_source.clone() + monA_edge_offset,
                ]
            )
        )
        local_e_ABfull_target.append(
            torch.cat(
                [
                    data.e_ABsr_target.clone() + monB_edge_offset,
                    data.e_ABlr_target.clone() + monB_edge_offset,
                ]
            )
        )
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

    e_ABfull_source = torch.cat(local_e_ABfull_source, dim=0)
    e_ABfull_target = torch.cat(local_e_ABfull_target, dim=0)

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


AP3_FUSED_FSAPT_SPLIT_SPEC_TYPES = frozenset({5, 6, 7, 8})


def fsapt_spec_type_uses_split_files(spec_type):
    return spec_type in AP3_FUSED_FSAPT_SPLIT_SPEC_TYPES


class ap3_fused_fsapt_module_dataset_lmdb(Dataset):
    split_spec_types = AP3_FUSED_FSAPT_SPLIT_SPEC_TYPES

    @classmethod
    def is_split_db_config(cls, spec_type, qcel_molecules=None):
        return fsapt_spec_type_uses_split_files(spec_type) or qcel_inputs_are_split_db(
            qcel_molecules
        )

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        r_cut_im=8.0,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        skip_processed=True,
        skip_compile=False,
        atom_model_path=None,
        atom_model=None,
        dimer_prop_model=None,
        batch_size=16,
        atomic_batch_size=256,
        datapoint_storage_n_objects=256,
        in_memory=False,
        num_devices=1,
        split="all",
        print_level=2,
        qcel_molecules: Optional[List[qcel.models.Molecule]] = None,
        energy_labels: Optional[List[float]] = None,
        frag1_indices: Optional[List[List[int]]] = None,
        frag2_indices: Optional[List[List[int]]] = None,
        random_seed=42,
        check_monomer_validity=True,
        device=None,
        lmdb_map_size=1099511627776,
        lmdb_readonly=False,
        fileserver_url=None,
        cache_size=1000,
    ):
        """
        Create an LMDB-backed dataset for AP3-fused FSAPT training that prepares and serves per-dimer fused data objects.

        Initializes dataset configuration, optional preloaded QCElemental inputs, model references (atomic model and optional dimer property model), LMDB storage, and an in-memory access cache. This constructor does not perform full dataset processing unless process() is invoked later; it only prepares runtime state and opens the LMDB environment when available.

        Parameters:
            root (str): Filesystem path for dataset storage and LMDB files.
            transform: Optional graph transform applied on-the-fly (kept for compatibility).
            pre_transform: Optional transform applied before saving processed examples.
            r_cut (float): Short-range cutoff (angstroms) used when building fused data objects.
            r_cut_im (float): Interaction-model cutoff (angstroms) used for long-range/short-range splitting.
            spec_type (int or None): Dataset specification type (allowed values: 5, 6, 7, 8, or None) that selects expected raw-file layout.
            max_size (int or None): Optional upper limit on number of processed dimers to store.
            force_reprocess (bool): If true, reinitialize LMDB and force reprocessing on next process() call.
            skip_processed (bool): If true, skip using already-processed LMDB files when loading models from disk.
            skip_compile (bool): If true, avoid compiling the atomic model with torch.compile.
            atom_model_path (path-like): Default path to a pre-trained atomic model checkpoint used when atom_model is not supplied.
            atom_model: Optional already-instantiated AtomModel or underlying model to use for per-atom predictions.
            dimer_prop_model: Optional model used to compute classical dimer properties (e.g., electrostatics, induction) during processing.
            batch_size (int): Number of dimer examples to group when computing dimer-level classical properties.
            atomic_batch_size (int): Number of atoms to batch when running atomic model inference; must be <= datapoint_storage_n_objects.
            datapoint_storage_n_objects (int): Number of processed examples bundled per LMDB write chunk.
            in_memory (bool): If true, prefer keeping dataset entries available in memory where feasible.
            num_devices (int): Number of devices intended for model execution (informational).
            split (str or int): Dataset split identifier (e.g., "all", train/val/test or numeric split); affects LMDB path selection.
            print_level (int): Verbosity level for informational prints.
            qcel_molecules (list[qcel.models.Molecule] or None): Optional pre-loaded QCElemental molecule objects to build dataset directly instead of reading raw files.
            energy_labels (list[float] or None): Corresponding target energy labels for qcel_molecules; if provided, frag1_indices and frag2_indices must also be provided.
            frag1_indices (list[list[int]] or None): Fragment indices for fragment 1 for each provided QCElemental molecule (required when qcel_molecules are supplied).
            frag2_indices (list[list[int]] or None): Fragment indices for fragment 2 for each provided QCElemental molecule (required when qcel_molecules are supplied).
            random_seed (int): RNG seed used for any dataset subsampling or ordering.
            check_monomer_validity (bool): If true, perform monomer validity checks during processing.
            device: Default device for model placement when loading/preparing models.
            lmdb_map_size (int): Maximum LMDB size in bytes (default 1 TB).
            lmdb_readonly (bool): Whether to open LMDB in read-only mode.
            fileserver_url (str or None): Optional URL for retrieving raw data from a fileserver.
            cache_size (int): Number of recently accessed items to retain in the in-memory LRU cache.

        Notes:
            - If qcel_molecules and energy_labels are both provided, frag1_indices and frag2_indices are required and lengths of molecules and labels must match.
            - The constructor will attempt to import and use the lmdb package; absence of lmdb will raise ImportError.
            - When an atomic model is supplied or loaded, the code may prepare it for inference (e.g., device placement or optional compilation) so provide compatible model objects.
        """
        try:
            import lmdb
            import json
        except ImportError:
            raise ImportError(
                "lmdb package is required. Install with: pip install lmdb"
            )

        self.lmdb = lmdb
        self.json = json
        self.print_level = print_level
        try:
            assert spec_type in [5, 6, 7, 8, None]
        except Exception:
            print("Currently spec_type must be 5 or None")
            raise ValueError
        self.spec_type = spec_type
        assert atomic_batch_size <= datapoint_storage_n_objects, (
            f"atomic_batch_size must be <= datapoint_storage_n_objects, "
            f"got {atomic_batch_size} and {datapoint_storage_n_objects}"
        )

        self.qcel_molecules = None
        self.energy_labels = None
        self.frag1_indices = None
        self.frag2_indices = None
        if qcel_molecules is not None and energy_labels is not None:
            assert frag1_indices is not None and frag2_indices is not None, (
                "If providing qcel_molecules and energy_labels, "
                "must also provide frag1_indices and frag2_indices for fsapt"
            )
            self.qcel_molecules = qcel_molecules
            self.energy_labels = energy_labels
            self.frag1_indices = frag1_indices
            self.frag2_indices = frag2_indices
            if len(qcel_molecules) != len(energy_labels):
                raise ValueError(
                    "Length of qcel_molecules and energy_labels must match"
                )
            print(
                f"Received {len(qcel_molecules)}"
                " QCElemental molecules with energy labels"
            )

        self.device = device
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
        self.skip_processed = skip_processed

        self.lmdb_map_size = lmdb_map_size
        self.lmdb_readonly = lmdb_readonly
        self.fileserver_url = fileserver_url
        self.cache_size = cache_size
        self._cache = {}
        self._cache_keys = []

        self.lmdb_env = None
        self.lmdb_path = None
        self._length = None
        self._raw_cursor = None
        self._worker_id = None

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

        self.dimer_prop_model = dimer_prop_model
        if self.dimer_prop_model is not None:
            if hasattr(self.dimer_prop_model, "eval"):
                self.dimer_prop_model.eval()
            elif hasattr(self.dimer_prop_model, "model"):
                self.dimer_prop_model.model.eval()
                if hasattr(self.dimer_prop_model, "dimer_model"):
                    self.dimer_prop_model.dimer_model.eval()
                if hasattr(self.dimer_prop_model, "dimer_model_elst"):
                    self.dimer_prop_model.dimer_model_elst.eval()

        print(f"{root=}, {self.spec_type=}, {self.in_memory=}")

        self._init_lmdb_path(root)
        self._init_lmdb()

        super(ap3_fused_fsapt_module_dataset_lmdb, self).__init__(
            root, transform, pre_transform
        )

        if self.force_reprocess:
            self.force_reprocess = False
            self._close_lmdb()
            super(ap3_fused_fsapt_module_dataset_lmdb, self).__init__(
                root, transform, pre_transform
            )
            self._init_lmdb()

    def _init_lmdb_path(self, root):
        """Initialize LMDB path before parent class init"""
        split_name = f"_{self.split}" if self.split != "all" else ""
        self.lmdb_path = osp.join(
            root, "processed", f"lmdb_ap3_fused_fsapt{split_name}_spec_{self.spec_type}"
        )

    def _init_lmdb(self):
        """Initialize LMDB environment"""
        if not osp.exists(self.lmdb_path):
            os.makedirs(self.lmdb_path, exist_ok=True)

        for attempt in range(2):
            try:
                self.lmdb_env = acquire_lmdb_env(
                    self.lmdb,
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
                        metadata = self.json.loads(metadata_bytes.decode("utf-8"))
                        self._length = metadata.get("length", 0)
                        self._raw_cursor = metadata.get(
                            "raw_cursor", self._length
                        )
                    else:
                        self._length = 0
                        self._raw_cursor = 0
                return
            except Exception as e:
                if attempt == 0 and "already open in this process" in str(e):
                    import gc

                    if self.lmdb_env is not None:
                        release_lmdb_env(self.lmdb_path, self.lmdb_env)
                        self.lmdb_env = None
                        self._length = 0
                        self._raw_cursor = 0
                    gc.collect()
                    continue
                print(f"Error initializing LMDB: {e}")
                if self.lmdb_env is not None:
                    release_lmdb_env(self.lmdb_path, self.lmdb_env)
                self.lmdb_env = None
                self._length = 0
                self._raw_cursor = 0
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
        """
        Return the expected raw dataset file names for the dataset's spec_type.

        The returned list contains the training and test filenames corresponding to
        the configured spec_type:
        - spec_type == 5: "fsapt_train_data.pkl", "fsapt_test_data.pkl"
        - spec_type == 6: "fsapt_train_simple.pkl", "fsapt_test_simple.pkl"
        - spec_type == 7: "fsapt_train_simple_2.pkl", "fsapt_test_simple_2.pkl"
        - spec_type == 8: "90K_fsaptpbe0-d4_fsapt_train.pkl", "90K_fsaptpbe0-d4_fsapt_test.pkl"

        Returns:
            list[str]: Two filenames [train, test] for the current spec_type.
        """
        if self.spec_type == 5:
            return [
                "fsapt_train_data.pkl",
                "fsapt_test_data.pkl",
            ]
        elif self.spec_type == 6:
            return [
                "fsapt_train_simple.pkl",
                "fsapt_test_simple.pkl",
            ]
        elif self.spec_type == 7:
            return [
                "fsapt_train_simple_2.pkl",
                "fsapt_test_simple_2.pkl",
            ]
        elif self.spec_type == 8:
            return [
                "90K_fsaptpbe0-d4_fsapt_train.pkl",
                "90K_fsaptpbe0-d4_fsapt_test.pkl",
            ]

    @property
    def processed_file_names(self):
        """
        Determine the dataset's processed file names based on the LMDB presence and metadata.

        Checks whether the LMDB at the configured lmdb_path exists and contains at least one entry; if so returns the expected MDB and lock file paths for this dataset split and spec type. If force_reprocess is set, returns ["file"] to signal reprocessing. If the LMDB is missing or empty, returns ["lmdb_missing"].

        Returns:
            list[str]: A list containing either the two expected LMDB filenames for this dataset (e.g.
            "lmdb_ap3_fused_fsapt[_<split>]_spec_<spec_type>/data.mdb" and the corresponding "lock.mdb"),
            or the marker lists ["file"] (when force_reprocess is true) or ["lmdb_missing"] (when no valid LMDB is found).
        """
        if self.force_reprocess:
            return ["file"]

        if not hasattr(self, "lmdb_path") or self.lmdb_path is None:
            return ["lmdb_missing"]

        if osp.exists(self.lmdb_path):
            env = None
            try:
                env = acquire_lmdb_env(
                    self.lmdb,
                    self.lmdb_path,
                    map_size=self.lmdb_map_size,
                    readonly=True,
                    lock=False,
                    max_dbs=0,
                    max_readers=256,
                )
                with env.begin() as txn:
                    metadata_bytes = txn.get(b"__metadata__")
                    if metadata_bytes:
                        metadata = json.loads(metadata_bytes.decode("utf-8"))
                        length = metadata.get("length", 0)

                        if length > 0:
                            split_name = f"_{self.split}" if self.split != "all" else ""
                            files_to_return = [
                                f"lmdb_ap3_fused_fsapt{split_name}_spec_{self.spec_type}/data.mdb",
                                f"lmdb_ap3_fused_fsapt{split_name}_spec_{self.spec_type}/lock.mdb",
                            ]
                            return files_to_return
            except Exception as e:
                print(f"Error checking LMDB: {e}")
            finally:
                if env is not None:
                    release_lmdb_env(self.lmdb_path, env)

        return ["lmdb_missing"]

    def download(self):
        """
        Prepare or fetch the raw data files required by the dataset.

        If both `energy_labels` and `qcel_molecules` are present on the instance, this method returns without action. Otherwise it prints `processed_paths` and `raw_file_names` and raises NotImplementedError indicating that the dataset must be provided via QCElemental inputs or available raw files for the chosen `spec_type`.

        Raises:
            NotImplementedError: If required data are not provided via `qcel_molecules` and `energy_labels`.
        """
        if self.energy_labels and self.qcel_molecules:
            return
        print(self.processed_paths)
        print(self.raw_file_names)
        raise NotImplementedError(
            "Download method not implemented. Provide qcel_molecules and energy_labels or named spec type with corresponding data files."
        )

    def _process_dimer_batch(self, batch_data_list):
        """Process dimer batch with classical energies - same as original"""
        from apnet_pt.util import scatter_sum_compile

        temp_batch = ap3_fused_fsapt_collate_update(batch_data_list)

        if self.device:
            temp_batch = temp_batch.to(self.device)

        with torch.no_grad():
            if hasattr(self.dimer_prop_model, "set_forward"):
                result = self.dimer_prop_model(temp_batch)
                E_classical = result[0]
            elif hasattr(self.dimer_prop_model, "dimer_model"):
                result = self.dimer_prop_model.dimer_model(temp_batch)
                E_classical = result[0]
            else:
                raise ValueError(
                    "dimer_prop_model must have either set_forward or dimer_model attribute"
                )
            if E_classical.ndim == 1:
                E_elst_pairs = E_classical
                E_ind_pairs = torch.zeros_like(E_classical)
            elif E_classical.ndim == 2:
                E_elst_pairs = E_classical[:, 0]
                E_ind_pairs = E_classical[:, 1]
            else:
                raise ValueError(
                    f"Expected E_classical to be 1D or 2D, got shape {E_classical.shape}"
                )

            ndimer = temp_batch.total_charge_A.size(0)
            E_elst_full_dimer = scatter_sum_compile(
                E_elst_pairs, temp_batch.dimer_ind_full, ndimer
            )
            E_elst_full_dimer = E_elst_full_dimer.unsqueeze(-1)
            N_full, num_cols = E_elst_full_dimer.shape
            full_expanded = E_elst_full_dimer.new_zeros((ndimer, num_cols))
            full_expanded[:N_full] = E_elst_full_dimer
            E_elst_dimer = full_expanded

            E_ind_full_dimer = scatter_sum_compile(
                E_ind_pairs, temp_batch.dimer_ind_full, ndimer
            )
            E_ind_full_dimer = E_ind_full_dimer.unsqueeze(-1)
            N_full, num_cols = E_ind_full_dimer.shape
            full_expanded = E_ind_full_dimer.new_zeros((ndimer, num_cols))
            full_expanded[:N_full] = E_ind_full_dimer
            E_ind_dimer = full_expanded

            for j, data in enumerate(batch_data_list):
                data.E_classical_elst = E_elst_dimer[j].cpu()
                data.E_classical_ind = E_ind_dimer[j].cpu()

    def _store_to_lmdb(self, data_objects, start_idx, raw_cursor):
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
                "raw_cursor": raw_cursor,
                "r_cut": self.r_cut,
                "r_cut_im": self.r_cut_im,
                "spec_type": self.spec_type,
            }
            txn.put(b"__metadata__", self.json.dumps(metadata).encode("utf-8"))

        self._length = start_idx + len(data_objects)
        self._raw_cursor = raw_cursor

    def process(self):
        """
        Process dataset entries into fused dimer Data objects and store them in the LMDB backend.

        This method reads dimers either from provided QCElemental molecules with corresponding energy labels
        or from the configured raw data paths, constructs fused dimer Data objects (via dimer_fused_data),
        optionally augments them by computing classical interaction components using the configured
        dimer_prop_model, applies the optional pre_filter, and writes validated objects into LMDB in
        chunks. Storage respects configuration flags and limits such as MAX_SIZE, skip_processed,
        datapoint_storage_n_objects, and atomic_batch_size. Progress and diagnostic messages are printed
        according to the instance's print_level.
        """
        stored_idx = self._length if self.skip_processed and self._length else 0
        resume_raw_cursor = (
            self._raw_cursor if self.skip_processed and self._raw_cursor else 0
        )
        data_objects = []

        RAs, RBs, ZAs, ZBs, TQAs, TQBs, targets = [], [], [], [], [], [], []
        frag1_inds, frag2_inds = [], []

        if self.qcel_molecules is not None and self.energy_labels is not None:
            print("Processing directly from provided QCElemental molecules...")

            for mol in self.qcel_molecules:
                monA, monB = mol.get_fragment(0), mol.get_fragment(1)

                RA = torch.tensor(monA.geometry, dtype=torch.float32) * constants.au2ang
                RB = torch.tensor(monB.geometry, dtype=torch.float32) * constants.au2ang
                ZA = torch.tensor(monA.atomic_numbers, dtype=torch.int64)
                ZB = torch.tensor(monB.atomic_numbers, dtype=torch.int64)

                TQA = torch.tensor(monA.molecular_charge, dtype=torch.float32)
                TQB = torch.tensor(monB.molecular_charge, dtype=torch.float32)

                RAs.append(RA)
                RBs.append(RB)
                ZAs.append(ZA)
                ZBs.append(ZB)
                TQAs.append(TQA)
                TQBs.append(TQB)
            targets = self.energy_labels
            frag1_inds = self.frag1_indices
            frag2_inds = self.frag2_indices

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
                if self.spec_type in [2, 5, 6, 7, 8, 9]:
                    split_name = f"_{self.split}" if self.split != "all" else ""
                    if self.split not in Path(raw_path).stem:
                        print(f"{self.split} is skipping {raw_path}")
                        continue

                print(f"raw_path: {raw_path}")
                print("Loading dimers...")
                RA, RB, ZA, ZB, TQA, TQB, target, frag1_ind, frag2_ind = (
                    util.load_dimer_dataset(
                        raw_path,
                        self.MAX_SIZE,
                        return_qcel_mols=False,
                        return_qcel_mons=False,
                        return_fragment_indices=True,
                        columns=[
                            "F-Electrostatics",
                            "F-Exchange",
                            "F-Induction",
                            "F-Dispersion",
                            "F-Total",
                        ],
                        random_seed_shuffle=self.random_seed,
                    )
                )
                RAs.extend(RA)
                RBs.extend(RB)
                ZAs.extend(ZA)
                ZBs.extend(ZB)
                TQAs.extend(TQA)
                TQBs.extend(TQB)
                targets.extend(target)
                frag1_inds.extend(frag1_ind)
                frag2_inds.extend(frag2_ind)

        print("Creating data objects...")
        t1 = time()
        print(f"{len(RAs)=}, {self.atomic_batch_size=}, {self.batch_size=}")

        batch_data_objects = []
        processed_raw_cursor = resume_raw_cursor
        for i in range(len(RAs)):
            if self.skip_processed and i < resume_raw_cursor:
                continue
            processed_raw_cursor = i + 1

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
                frag1_ind=torch.tensor(frag1_inds[i], dtype=torch.long),
                frag2_ind=torch.tensor(frag2_inds[i], dtype=torch.long),
            )

            if data is None:
                print(f"Skipping invalid dimer index {i}")
                continue

            if self.dimer_prop_model is not None:
                batch_data_objects.append(data)

                if len(batch_data_objects) >= self.atomic_batch_size:
                    self._process_dimer_batch(batch_data_objects)

                    for batch_data in batch_data_objects:
                        batch_data_cpu = batch_data.cpu()
                        if self.pre_filter is None or self.pre_filter(batch_data_cpu):
                            data_objects.append(batch_data_cpu)

                    batch_data_objects = []
            else:
                data = data.cpu()
                if self.pre_filter is None or self.pre_filter(data):
                    data_objects.append(data)

            if len(data_objects) >= self.datapoint_storage_n_objects:
                self._store_to_lmdb(data_objects, stored_idx, i + 1)

                if self.print_level >= 2:
                    print(
                        f"Stored {len(data_objects)} objects to LMDB at index {stored_idx}"
                    )

                stored_idx += len(data_objects)
                data_objects = []

                if self.MAX_SIZE is not None and i > self.MAX_SIZE:
                    break

        if self.dimer_prop_model is not None and len(batch_data_objects) > 0:
            self._process_dimer_batch(batch_data_objects)

            for batch_data in batch_data_objects:
                batch_data_cpu = batch_data.cpu()
                if self.pre_filter is None or self.pre_filter(batch_data_cpu):
                    data_objects.append(batch_data_cpu)

        if len(data_objects) > 0:
            self._store_to_lmdb(data_objects, stored_idx, processed_raw_cursor)

            if self.print_level >= 2:
                print(
                    f"Final: Stored {len(data_objects)} objects to LMDB at index {stored_idx}"
                )
        elif processed_raw_cursor > resume_raw_cursor:
            self._store_to_lmdb([], stored_idx, processed_raw_cursor)

        print(f"Processing complete. Total time: {time() - t1:.2f}s")

    def len(self):
        """Return dataset length from LMDB metadata"""
        if self._length is not None:
            return self._length

        if self.lmdb_env is None:
            return 0

        with self.lmdb_env.begin() as txn:
            metadata_bytes = txn.get(b"__metadata__")
            if metadata_bytes:
                metadata = self.json.loads(metadata_bytes.decode("utf-8"))
                self._length = metadata.get("length", 0)
                self._raw_cursor = metadata.get("raw_cursor", self._length)
            else:
                self._length = 0
                self._raw_cursor = 0

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
