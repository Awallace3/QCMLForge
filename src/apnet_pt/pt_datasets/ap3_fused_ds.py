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
from ..lmdb_utils import acquire_lmdb_env, release_lmdb_env
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


AP3_FUSED_SPLIT_SPEC_TYPES = frozenset({2, 5, 6, 7, 9, 10})


def spec_type_uses_split_files(spec_type):
    return spec_type in AP3_FUSED_SPLIT_SPEC_TYPES


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
    atomic_props_A = atomic_datasets.create_atomic_data(ZA, RA, TQA, r_cut=r_cut)
    atomic_props_B = atomic_datasets.create_atomic_data(ZB, RB, TQB, r_cut=r_cut)
    if atomic_props_A is None or atomic_props_B is None:
        print(
            "Unsupported element in dimer with "
            f"dimer_ind {dimer_ind}. Skipping this data point:\n"
            f"  {constants.ALLOWED_ELEMENTS=}\n  {ZA=}\n  {ZB=}"
        )
        return None
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


def ap3_fused_collate_update(batch):
    """
    AP3 collate function with precomputed classical energies attached.
    If classical terms are present in the data objects, they are batched so the
    model can learn residual corrections against the corresponding targets.
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

    local_indA = []
    local_indB = []

    has_precomputed = (
        hasattr(batch[0], "E_classical_elst")
        and hasattr(batch[0], "E_classical_ind")
        and hasattr(batch[0], "E_classical_disp")
    )
    # print(f"Batch has precomputed classical energies: {has_precomputed}")

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
        batched_data.E_classical_disp = torch.tensor(
            [data.E_classical_disp for data in batch]
        )

    return batched_data


def ap3_fused_collate_update_no_target(batch):
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


def save_hdf5_data_objects(data_objects, filepath):
    """Save list of data objects to HDF5 format"""
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
    """Load list of data objects from HDF5 format"""
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


class ap3_fused_module_dataset(Dataset):
    split_spec_types = AP3_FUSED_SPLIT_SPEC_TYPES

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
        dimer_prop_model=None,
        batch_size=16,
        atomic_batch_size=256,
        # DO NOT CHANGE UNLESS YOU WANT TO RE-PROCESS THE DATASET
        datapoint_storage_n_objects=256,
        in_memory=False,
        num_devices=1,
        split="all",  # train, test
        print_level=2,
        qcel_molecules: Optional[List[qcel.models.Molecule]] = None,
        energy_labels: Optional[List[float]] = None,
        random_seed=42,
        check_monomer_validity=True,
        storage_type="pt",  # "pt" or "h5" for storage format
        device=None,
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
            assert spec_type in [1, 2, 5, 6, 7, 8, 9, 10, None]
        except Exception:
            print("Currently spec_type must be 1 or 2 for SAPT0/jun-cc-pVDZ")
            raise ValueError
        self.spec_type = spec_type
        assert atomic_batch_size <= datapoint_storage_n_objects, (
            "atomic_batch_size must be <= datapoint_storage_n_objects, got {} and {}".format(
                atomic_batch_size, datapoint_storage_n_objects
            )
        )
        # assert datapoint_storage_n_objects % atomic_batch_size == 0, "datapoint_storage_n_objects must be multiple of atomic_batch_size, got {} and {}".format(
        #     datapoint_storage_n_objects, atomic_batch_size
        # )

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

        self.device = device
        self.MAX_SIZE = max_size
        self.random_seed = random_seed
        self.in_memory = in_memory
        self.split = split
        self.split_db = spec_type_uses_split_files(self.spec_type) or (
            self.qcel_molecules is not None and self.split != "all"
        )
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
        elif not self.skip_processed and dimer_prop_model is None:
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
        super(ap3_fused_module_dataset, self).__init__(root, transform, pre_transform)
        if self.force_reprocess:
            self.force_reprocess = False
            super(ap3_fused_module_dataset, self).__init__(
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
        # spec 10 SAPT(PBE0)-D4(I)/aug-cc-pVDZ
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
        elif self.spec_type == 10:
            return [
                "225K_saptpbe0-d4_totals_train.pkl",
                "225K_saptpbe0-d4_totals_test.pkl",
            ]
        elif self.spec_type is None:
            os.system(f"touch {self.raw_dir}/tmp.txt")
            return ["tmp.txt"]
        else:
            return [
                "splinter_spec1.pkl",
            ]

    def reprocess_file_names(self):
        if self.force_reprocess:
            return ["file"]
        else:
            if self.split_db and self.split == "train":
                file_cmd = f"{self.root}/processed/dimer_ap3_fused_train_spec_{self.spec_type}_*{self.file_extension}"
            elif self.split_db and self.split == "test":
                file_cmd = f"{self.root}/processed/dimer_ap3_fused_test_spec_{self.spec_type}_*{self.file_extension}"
            else:
                file_cmd = f"{self.root}/processed/dimer_ap3_fused_spec_{self.spec_type}_*{self.file_extension}"
            spec_files = glob(file_cmd)
            spec_files = [i.split("/")[-1] for i in spec_files]
            if len(spec_files) > 0:
                # want to preserve idx ordering
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

    def _process_dimer_batch(self, batch_data_list):
        from apnet_pt.util import scatter_sum_compile

        temp_batch = ap3_fused_collate_update(batch_data_list)

        if self.device:
            temp_batch = temp_batch.to(self.device)

        with torch.no_grad():
            if hasattr(self.dimer_prop_model, "set_forward"):
                self.dimer_prop_model.set_forward(
                    "ap3_elst_damping__induced_dipole__disp"
                )
                result = self.dimer_prop_model(temp_batch)
                self.dimer_prop_model.set_forward("ap3_atomMPNN")
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
                E_disp_pairs = torch.zeros_like(E_classical)
            elif E_classical.ndim == 2:
                E_elst_pairs = E_classical[:, 0]
                E_ind_pairs = E_classical[:, 1]
                E_disp_pairs = (
                    E_classical[:, 2]
                    if E_classical.shape[1] > 2
                    else torch.zeros_like(E_classical[:, 0])
                )
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
            # rows, cols = E_elst_dimer.shape
            # padded = E_elst_dimer.new_zeros((rows, cols + 3))
            # padded[:, :cols] = E_elst_dimer
            # E_elst_dimer = padded

            E_ind_full_dimer = scatter_sum_compile(
                E_ind_pairs, temp_batch.dimer_ind_full, ndimer
            )
            E_ind_full_dimer = E_ind_full_dimer.unsqueeze(-1)
            N_full, num_cols = E_ind_full_dimer.shape
            full_expanded = E_ind_full_dimer.new_zeros((ndimer, num_cols))
            full_expanded[:N_full] = E_ind_full_dimer
            E_ind_dimer = full_expanded

            E_disp_full_dimer = scatter_sum_compile(
                E_disp_pairs, temp_batch.dimer_ind_full, ndimer
            )
            E_disp_full_dimer = E_disp_full_dimer.unsqueeze(-1)
            N_full, num_cols = E_disp_full_dimer.shape
            full_expanded = E_disp_full_dimer.new_zeros((ndimer, num_cols))
            full_expanded[:N_full] = E_disp_full_dimer
            E_disp_dimer = full_expanded

            # rows, cols = E_ind_dimer.shape
            # padded = E_ind_dimer.new_zeros((rows, cols + 3))
            # padded[:, 2:3] = E_ind_dimer
            # E_ind_dimer = padded

            for j, data in enumerate(batch_data_list):
                data.E_classical_elst = E_elst_dimer[j].cpu()
                data.E_classical_ind = E_ind_dimer[j].cpu()
                data.E_classical_disp = E_disp_dimer[j].cpu()

    def process(self):
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
                if spec_type_uses_split_files(self.spec_type):
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
        batch_data_objects = []
        for i in range(len(RAs)):
            if self.skip_processed:
                datapath = osp.join(
                    self.processed_dir,
                    f"dimer_ap3_fused{split_name}_spec_{self.spec_type}_{
                        idx // self.points_per_file
                    }{self.file_extension}",
                )
                print(f"Saving to {datapath}")
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
            # Normally would store the data object to individual files,
            # but at 1.67M dimers, this is too many files. Need to
            # store self.datapoint_storage_n_objects (like 1000) dimers per file
            if len(data_objects) == self.points_per_file:
                if self.in_memory:
                    self.data.extend(data_objects)
                else:
                    datapath = osp.join(
                        self.processed_dir,
                        f"dimer_ap3_fused{split_name}_spec_{self.spec_type}_{
                            idx // self.points_per_file
                        }{self.file_extension}",
                    )
                    if self.print_level >= 2:
                        print(f"Saving to {datapath}")
                    if self.storage_type == "h5":
                        save_hdf5_data_objects(data_objects, datapath)
                    else:
                        torch.save(data_objects, datapath)
                data_objects = []
                if self.MAX_SIZE is not None and idx > self.MAX_SIZE:
                    break
            idx += 1

        if self.dimer_prop_model is not None and len(batch_data_objects) > 0:
            self._process_dimer_batch(batch_data_objects)

            for batch_data in batch_data_objects:
                batch_data_cpu = batch_data.cpu()
                if self.pre_filter is None or self.pre_filter(batch_data_cpu):
                    data_objects.append(batch_data_cpu)

        if self.print_level >= 2:
            print(f"{i}/{len(RAs)}, {time() - t2:.2f}s, {time() - t1:.2f}s")
        elif self.print_level >= 1 and idx % 1000:
            print(f"{i}/{len(RAs)}, {time() - t2:.2f}s, {time() - t1:.2f}s")
        t2 = time()
        if len(data_objects) > 0:
            if self.in_memory:
                self.data.extend(data_objects)
            else:
                datapath = osp.join(
                    self.processed_dir,
                    f"dimer_ap3_fused{split_name}_spec_{self.spec_type}_{
                        idx // self.points_per_file
                    }{self.file_extension}",
                )
                print(f"Saving to {datapath}")
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
            f"dimer_ap3_fused{split_name}_spec_{self.spec_type}_{idx_datapath}{self.file_extension}",
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


class ap3_fused_module_dataset_lmdb(Dataset):
    split_spec_types = AP3_FUSED_SPLIT_SPEC_TYPES

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
        random_seed=42,
        check_monomer_validity=True,
        device=None,
        lmdb_map_size=1099511627776,
        lmdb_readonly=False,
        fileserver_url=None,
        cache_size=1000,
    ):
        """
        LMDB-based dataset for AP3 fused module training.

        Args:
            lmdb_map_size: Maximum size of LMDB database in bytes (default 1TB)
            lmdb_readonly: Open LMDB in read-only mode
            fileserver_url: URL to fileserver for downloading data
            cache_size: Number of recently accessed items to keep in memory
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
            assert spec_type in [1, 2, 5, 6, 7, 8, 9, 10, None]
        except Exception:
            print("Currently spec_type must be 1 or 2 for SAPT0/jun-cc-pVDZ")
            raise ValueError
        self.spec_type = spec_type
        assert atomic_batch_size <= datapoint_storage_n_objects, (
            f"atomic_batch_size must be <= datapoint_storage_n_objects, "
            f"got {atomic_batch_size} and {datapoint_storage_n_objects}"
        )

        self.qcel_molecules = None
        self.energy_labels = None
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

        self.device = device
        self.MAX_SIZE = max_size
        self.random_seed = random_seed
        self.in_memory = in_memory
        self.split = split
        self.split_db = spec_type_uses_split_files(self.spec_type) or (
            self.qcel_molecules is not None and self.split != "all"
        )
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
        elif not self.skip_processed and dimer_prop_model is None:
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

        super(ap3_fused_module_dataset_lmdb, self).__init__(
            root, transform, pre_transform
        )

        if self.force_reprocess:
            self.force_reprocess = False
            self._close_lmdb()
            super(ap3_fused_module_dataset_lmdb, self).__init__(
                root, transform, pre_transform
            )
            self._init_lmdb()

    def _init_lmdb_path(self, root):
        """Initialize LMDB path before parent class init"""
        self.lmdb_path = osp.join(
            root, "processed", f"lmdb_ap3_fused{self.split_name}_spec_{self.spec_type}"
        )
        print(self.lmdb_path)

    @property
    def split_name(self):
        return f"_{self.split}" if self.split_db and self.split != "all" else ""

    def _init_lmdb(self):
        """Initialize LMDB environment"""
        if not osp.exists(self.lmdb_path):
            os.makedirs(self.lmdb_path, exist_ok=True)

        for attempt in range(2):
            env = None
            try:
                env = acquire_lmdb_env(
                    self.lmdb,
                    self.lmdb_path,
                    map_size=self.lmdb_map_size,
                    readonly=self.lmdb_readonly,
                    max_dbs=0,
                    lock=not self.lmdb_readonly,
                    max_readers=256,
                )

                with env.begin() as txn:
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
                self.lmdb_env = env
                return
            except Exception as e:
                if env is not None:
                    release_lmdb_env(self.lmdb_path, env)
                if attempt == 0 and "already open in this process" in str(e):
                    import gc

                    gc.collect()
                    continue
                print(f"Error initializing LMDB: {e}")
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
        """Same as original implementation"""
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
        elif self.spec_type == 10:
            return [
                "225K_saptpbe0-d4_totals_train.pkl",
                "225K_saptpbe0-d4_totals_test.pkl",
            ]
        elif self.spec_type is None:
            os.system(f"touch {self.raw_dir}/tmp.txt")
            return ["tmp.txt"]
        else:
            return [
                "splinter_spec1.pkl",
            ]

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
                env = acquire_lmdb_env(
                    self.lmdb,
                    self.lmdb_path,
                    readonly=True,
                    lock=False,
                    max_dbs=0,
                    map_size=self.lmdb_map_size,
                    max_readers=256,
                )
                with env.begin() as txn:
                    metadata_bytes = txn.get(b"__metadata__")
                    if metadata_bytes:
                        import json

                        metadata = json.loads(metadata_bytes.decode("utf-8"))
                        length = metadata.get("length", 0)

                        if length > 0:
                            if self.MAX_SIZE is not None and length >= self.MAX_SIZE:
                                return [
                                    f"lmdb_ap3_fused{self.split_name}_spec_{self.spec_type}"
                                ]
                            return [
                                f"lmdb_ap3_fused{self.split_name}_spec_{self.spec_type}"
                            ]
            except Exception as e:
                print(f"Error checking LMDB: {e}")
            finally:
                if env is not None:
                    release_lmdb_env(self.lmdb_path, env)

        return ["lmdb_missing"]

    def download(self):
        """Download data - same as original or from fileserver"""
        if self.energy_labels and self.qcel_molecules:
            return

        if self.fileserver_url:
            print(f"Downloading from fileserver: {self.fileserver_url}")
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

    def _process_dimer_batch(self, batch_data_list):
        """Process dimer batch with classical energies - same as original"""
        from apnet_pt.util import scatter_sum_compile

        temp_batch = ap3_fused_collate_update(batch_data_list)

        if self.device:
            temp_batch = temp_batch.to(self.device)

        with torch.no_grad():
            if hasattr(self.dimer_prop_model, "set_forward"):
                self.dimer_prop_model.set_forward(
                    "ap3_elst_damping__induced_dipole__disp"
                )
                result = self.dimer_prop_model(temp_batch)
                self.dimer_prop_model.set_forward("ap3_atomMPNN")
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
                E_disp_pairs = torch.zeros_like(E_classical)
            elif E_classical.ndim == 2:
                E_elst_pairs = E_classical[:, 0]
                E_ind_pairs = E_classical[:, 1]
                E_disp_pairs = (
                    E_classical[:, 2]
                    if E_classical.shape[1] > 2
                    else torch.zeros_like(E_classical[:, 0])
                )
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

            E_disp_full_dimer = scatter_sum_compile(
                E_disp_pairs, temp_batch.dimer_ind_full, ndimer
            )
            E_disp_full_dimer = E_disp_full_dimer.unsqueeze(-1)
            N_full, num_cols = E_disp_full_dimer.shape
            full_expanded = E_disp_full_dimer.new_zeros((ndimer, num_cols))
            full_expanded[:N_full] = E_disp_full_dimer
            E_disp_dimer = full_expanded

            for j, data in enumerate(batch_data_list):
                data.E_classical_elst = E_elst_dimer[j].cpu()
                data.E_classical_ind = E_ind_dimer[j].cpu()
                data.E_classical_disp = E_disp_dimer[j].cpu()

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
        """Process dataset and store in LMDB"""
        stored_idx = self._length if self.skip_processed and self._length else 0
        resume_raw_cursor = (
            self._raw_cursor if self.skip_processed and self._raw_cursor else 0
        )
        data_objects = []

        RAs, RBs, ZAs, ZBs, TQAs, TQBs, targets = [], [], [], [], [], [], []

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
                if spec_type_uses_split_files(self.spec_type):
                    split_name = self.split_name
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
