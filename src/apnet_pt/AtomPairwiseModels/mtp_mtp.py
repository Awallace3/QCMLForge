import torch
import torch.nn as nn

# from torch_scatter import scatter
from torch_geometric.utils import scatter
import numpy as np
import time
from ..AtomModels.ap2_atom_model import (
    AtomMPNN,
    isolate_atomic_property_predictions,
    qcel_mon_to_pyg_data,
    unwrap_model,
)
from ..atomic_datasets import (
    AtomicDataLoader,
    atomic_collate_update,
    atomic_collate_update_no_target,
    atomic_collate_update_prebatched,
)
from ..AtomModels.ap3_atom_model import (
    AtomHirshfeldMPNN,
    atomic_hirshfeld_module_dataset,
)
from ..pt_datasets.ap2_fused_ds import (
    ap2_fused_module_dataset,
    APNet2_fused_DataLoader,
    ap2_fused_collate_update,
    ap2_fused_collate_update_no_target,
    qcel_dimer_to_fused_data,
)
from .. import constants
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import qcelemental as qcel
from importlib import resources
from copy import deepcopy
from apnet_pt.torch_util import set_weights_to_value
from torch_geometric.data import Data


max_Z = 118


class NoisyConstantEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, mean=3.0, std=0.01):
        super().__init__(num_embeddings, embedding_dim)
        with torch.no_grad():
            # Use smaller, more controlled noise and clamp to reasonable bounds
            noise = torch.randn_like(self.weight) * min(std, 0.01)
            self.weight.copy_(mean + noise)
            # Clamp to prevent extreme values that could cause NaN
            self.weight.clamp_(min=mean-1.0, max=mean+1.0)


class DimerProp(nn.Module):
    def __init__(self, ATParam, dimer_eval="elst_damping"):
        super().__init__()
        self.AtomTypeParam = ATParam
        if dimer_eval == "elst_damping":
            self.forward = self._elst_damping_forward
        elif dimer_eval == "elst":
            self.forward = self._elst_forward
        elif dimer_eval == "induced_dipole":
            self.forward = self._indu_induced_dipole_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "elst_damping__induced_dipole":
            self.forward = self._elst_damping_indu_induced_dipole_forward
            self.polarizability_table = constants.polarizability_table.clone()
        else:
            raise ValueError(f"Unknown dimer_eval: {dimer_eval}")
        return

    def _elst_damping_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(
            Data(
                x=batch.ZA,
                R=batch.RA,
                edge_index=torch.vstack((batch.e_AA_source, batch.e_AA_target)),
                molecule_ind=batch.molecule_ind_A,
                total_charge=batch.total_charge_A,
                natom_per_mol=batch.natom_per_mol_A,
            )
        )
        v_B = self.AtomTypeParam(
            Data(
                x=batch.ZB,
                R=batch.RB,
                edge_index=torch.vstack((batch.e_BB_source, batch.e_BB_target)),
                molecule_ind=batch.molecule_ind_B,
                total_charge=batch.total_charge_B,
                natom_per_mol=batch.natom_per_mol_B,
            )
        )
        Elst = mtp_elst_damping(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=v_A[-1],
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=v_B[-1],
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
        )
        return Elst

    def _elst_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(
            Data(
                x=batch.ZA,
                R=batch.RA,
                edge_index=torch.vstack((batch.e_AA_source, batch.e_AA_target)),
                molecule_ind=batch.molecule_ind_A,
                total_charge=batch.total_charge_A,
                natom_per_mol=batch.natom_per_mol_A,
            )
        )
        v_B = self.AtomTypeParam(
            Data(
                x=batch.ZB,
                R=batch.RB,
                edge_index=torch.vstack((batch.e_BB_source, batch.e_BB_target)),
                molecule_ind=batch.molecule_ind_B,
                total_charge=batch.total_charge_B,
                natom_per_mol=batch.natom_per_mol_B,
            )
        )
        Elst = mtp_elst(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
        )
        return Elst

    def _indu_induced_dipole_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(
            Data(
                x=batch.ZA,
                R=batch.RA,
                edge_index=torch.vstack((batch.e_AA_source, batch.e_AA_target)),
                molecule_ind=batch.molecule_ind_A,
                total_charge=batch.total_charge_A,
                natom_per_mol=batch.natom_per_mol_A,
            )
        )
        v_B = self.AtomTypeParam(
            Data(
                x=batch.ZB,
                R=batch.RB,
                edge_index=torch.vstack((batch.e_BB_source, batch.e_BB_target)),
                molecule_ind=batch.molecule_ind_B,
                total_charge=batch.total_charge_B,
                natom_per_mol=batch.natom_per_mol_B,
            )
        )
        # print(f"{v_A[3] =}")
        # print(f"{v_A[4] =}")
        Indu = induced_dipole_induction_optimized(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=v_A[-1],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=v_B[-1],
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
            # Additional parameters for induction
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=v_A[3],
            hirshfeld_volume_ratio_B=v_B[3],
            valence_widths_A=v_A[4],
            valence_widths_B=v_B[4],
            polarizability_table=self.polarizability_table,
        )
        return Indu

    def _elst_damping_indu_induced_dipole_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(
            Data(
                x=batch.ZA,
                R=batch.RA,
                edge_index=torch.vstack((batch.e_AA_source, batch.e_AA_target)),
                molecule_ind=batch.molecule_ind_A,
                total_charge=batch.total_charge_A,
                natom_per_mol=batch.natom_per_mol_A,
            )
        )
        v_B = self.AtomTypeParam(
            Data(
                x=batch.ZB,
                R=batch.RB,
                edge_index=torch.vstack((batch.e_BB_source, batch.e_BB_target)),
                molecule_ind=batch.molecule_ind_B,
                total_charge=batch.total_charge_B,
                natom_per_mol=batch.natom_per_mol_B,
            )
        )
        # print(f"{v_A[-1] =}")
        # print(f"{v_A[-2] =}")
        Ka = torch.clamp(v_A[-1][:, 1], min=0.05, max=4.0)
        Kb = torch.clamp(v_B[-1][:, 1], min=0.05, max=4.0)
        Indu = induced_dipole_induction_optimized(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=Ka,
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=Kb,
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
            # Additional parameters for induction
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=v_A[-2][:, 0],
            hirshfeld_volume_ratio_B=v_B[-2][:, 0],
            valence_widths_A=v_A[-2][:, 1],
            valence_widths_B=v_B[-2][:, 1],
            polarizability_table=self.polarizability_table,
        )
        if Indu.isnan().any():
            print("Induced dipole energy is NaN, debugging info:")
            print(f"{v_A[-2] =}")
            print(f"{v_B[-2] =}")
            print(f"{v_A[-1] =}")
            print(f"{v_B[-1] =}")
            print(f"{Ka =}")
            print(f"{Kb =}")
            raise ValueError("Induced dipole energy is NaN")
        # Must compute Elst after Ind because we modify qA and qB in place... pain to debug
        Elst = mtp_elst_damping(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=v_A[-1][:, 0],
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=v_B[-1][:, 0],
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
        )
        if Elst.isnan().any():
            print("Electrostatic energy is NaN, debugging info:")
            print(f"{v_A[-1] =}")
            print(f"{v_B[-1] =}")
            raise ValueError("Electrostatic energy is NaN")
        return torch.vstack((Elst, Indu)).T


class AtomTypeParamNN(nn.Module):
    def __init__(
        self,
        atom_model: AtomMPNN = AtomMPNN(),
        n_message=3,
        n_neuron=128,
        n_embed=8,
        param_start_mean=1.8,
        param_start_std=0.01,
        n_params=1,
    ):
        super().__init__()
        self.atom_model = atom_model
        self.atom_model.requires_grad_(False)
        self.n_message = n_message
        if type(self.atom_model) in [AtomMPNN, AtomHirshfeldMPNN]:
            self.h_list_ind = -1
        elif type(self.atom_model) is AtomTypeParamNN:
            self.h_list_ind = 3
        else:
            raise ValueError("Unknown atom_model type")
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        # Convert to lists if scalars
        if not isinstance(param_start_mean, (list, tuple)):
            param_start_mean = [param_start_mean] * n_params
        if not isinstance(param_start_std, (list, tuple)):
            param_start_std = [param_start_std] * n_params
        # Ensure they are the right length
        if len(param_start_mean) != n_params:
            raise ValueError(f"param_start_mean length {len(param_start_mean)} doesn't match n_params {n_params}")
        if len(param_start_std) != n_params:
            raise ValueError(f"param_start_std length {len(param_start_std)} doesn't match n_params {n_params}")

        self.param_start_mean = param_start_mean
        self.param_start_std = param_start_std
        self.n_params = n_params
        self.guess_layer = nn.ModuleList(
            [
                NoisyConstantEmbedding(
                    max_Z + 1, 1, mean=self.param_start_mean[p], std=self.param_start_std[p]
                )
                for p in range(n_params)
            ]
        )

        # readout layers for predicting multipoles from hidden states
        self.param_readout_layers = nn.ModuleList(
            [nn.ModuleList() for _ in range(n_params)]
        )
        layer_nodes_readout = [
            n_embed,
            n_neuron * 2,
            n_neuron,
            n_neuron // 2,
            1,
        ]
        layer_activations = [
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            None,
        ]
        for p in range(n_params):
            for i in range(n_message):
                self.param_readout_layers[p].append(
                    self._make_layers(layer_nodes_readout, layer_activations)
                )

    def _make_layers(self, layer_nodes, activations):
        layers = []
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            # layers[-1].weight.data.normal_(1.0, 0.1)
            if activations[i] is not None:
                layers.append(activations[i])
        return nn.Sequential(*layers)

    def forward(
        self,
        batch,
    ):
        """
        Use each h_list to predict a correction to the initial guess, might be
        overkill for some properties...
        """
        x = batch.x
        edge_index = batch.edge_index
        molecule_ind = batch.molecule_ind
        am_out = self.atom_model(batch)
        charge, dipole, qpole, h_list = am_out[0], am_out[1], am_out[2], am_out[self.h_list_ind]
        Z = x
        # print(self.guess_layer[0].weight)
        # print(self.guess_layer[1].weight)
        K_list = [self.guess_layer[p](Z) for p in range(self.n_params)]
        K = torch.cat(K_list, dim=-1)  # shape (n_atoms, n_params)
        atoms_with_edges = torch.cat([edge_index[0], edge_index[1]]).unique()
        keep_mask = torch.isin(
            torch.arange(len(molecule_ind), device=molecule_ind.device),
            atoms_with_edges,
        )
        # Clamp K values to prevent extreme values that could lead to NaN
        K = torch.clamp(K, min=-10.0, max=10.0)
        if K.isnan().any():
            print("K has NaN values before readouts, debugging info:")
            print(f"{K =}")
            raise ValueError("K has NaN values")
        K_filtered = K[keep_mask]  # shape (n_atoms_filtered, n_params)
        for p in range(self.n_params):
            for i in range(self.n_message):
                param_update = self.param_readout_layers[p][i](h_list[:, i + 1, :])
                K_filtered[:, p] += param_update.squeeze(-1)
        K[keep_mask] = torch.relu(K_filtered)  # + 1.00001
        if K.isnan().any():
            print("K has NaN values, debugging info:")
            print(f"{K_filtered =}")
            print(f"{h_list=}")
            raise ValueError("K has NaN values")
        return (
            charge,
            dipole,
            qpole,
            *am_out[3:],
            K.squeeze(-1) if self.n_params == 1 else K,
        )


def get_distances(RA, RB, e_source, e_target):
    RA_source = RA.index_select(0, e_source)
    RB_target = RB.index_select(0, e_target)
    dR_xyz = RB_target - RA_source
    dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
    return dR, dR_xyz


@torch.compile
def elst_damping_mtp_mtp_torch(
    alpha_i: torch.tensor,
    alpha_j: torch.tensor,
    r: torch.tensor,
    e_source: torch.tensor,
    e_target: torch.tensor,
):
    """
    # MTP-MTP interaction
    """
    # need to have alpha_i repeated for each atom in j and vice versa
    alpha_i = alpha_i.index_select(0, e_source)
    alpha_j = alpha_j.index_select(0, e_target)
    r2 = r**2
    r3 = r2 * r
    a1_2 = alpha_i * alpha_i
    a2_2 = alpha_j * alpha_j
    a1_3 = a1_2 * alpha_i
    lam1 = torch.ones_like(r)
    lam3 = torch.ones_like(r)
    lam5 = torch.ones_like(r)
    e1r = torch.exp(-1.0 * alpha_i * r)
    e2r = torch.exp(-1.0 * alpha_j * r)
    diff = torch.abs(alpha_i - alpha_j) > 1e-6
    A = torch.where(diff, a2_2 / (a2_2 - a1_2), torch.zeros_like(r))
    B = torch.where(diff, a1_2 / (a1_2 - a2_2), torch.zeros_like(r))
    lam1 = torch.where(diff, 1 - A * e1r - B * e2r, 1 - (1.0 + 0.5 * alpha_i * r) * e1r)
    lam3 = torch.where(
        diff,
        1 - (1.0 + alpha_i * r) * A * e1r - (1.0 + alpha_j * r) * B * e2r,
        1 - (1.0 + alpha_i * r + 0.5 * a1_2 * r2) * e1r,
    )
    lam5 = torch.where(
        diff,
        1
        - (1.0 + alpha_i * r + (1.0 / 3.0) * a1_2 * r2) * A * e1r
        - (1.0 + alpha_j * r + (1.0 / 3.0) * a2_2 * r2) * B * e2r,
        1 - (1.0 + alpha_i * r + 0.5 * a1_2 * r2 + (1.0 / 6.0) * a1_3 * r3) * e1r,
    )
    return lam1, lam3, lam5


@torch.compile
def elst_damping_Z_mtp_torch(
    alpha_i: torch.tensor,
    alpha_j: torch.tensor,
    r: torch.tensor,
    e_source: torch.tensor,
    e_target: torch.tensor,
):
    """
    # Z-MTP interaction
    """
    # need to have alpha_i repeated for each atom in j and vice versa
    alpha_i = alpha_i.index_select(0, e_source)
    alpha_j = alpha_j.index_select(0, e_target)
    lam1_j = 1.0 - torch.exp(-1.0 * torch.multiply(alpha_j, r))
    lam3_j = 1.0 - (1.0 + torch.multiply(alpha_j, r)) * torch.exp(
        -1.0 * torch.multiply(alpha_j, r)
    )
    lam5_j = 1.0 - (
        1.0
        + torch.multiply(alpha_j, r)
        + (1.0 / 3.0) * torch.multiply(torch.square(alpha_j), r**2)
    ) * torch.exp(-1.0 * torch.multiply(alpha_j, r))
    lam1_i = 1.0 - torch.exp(-1.0 * torch.multiply(alpha_i, r))
    lam3_i = 1.0 - (1.0 + torch.multiply(alpha_i, r)) * torch.exp(
        -1.0 * torch.multiply(alpha_i, r)
    )
    lam5_i = 1.0 - (
        1.0
        + torch.multiply(alpha_i, r)
        + (1.0 / 3.0) * torch.multiply(torch.square(alpha_i), r**2)
    ) * torch.exp(-1.0 * torch.multiply(alpha_i, r))
    return lam1_j, lam3_j, lam5_j, lam1_i, lam3_i, lam5_i


@torch.compile
def mtp_elst(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    ZB,
    RB,
    qB,
    muB,
    quadB,
    e_AB_source,
    e_AB_target,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
):
    dR_ang, dR_xyz_ang = get_distances(RA, RB, e_AB_source, e_AB_target)
    dR = dR_ang / constants.au2ang
    dR_xyz = dR_xyz_ang / constants.au2ang
    oodR = 1.0 / dR
    delta = torch.eye(3, device=qA.device)

    ZA_q = ZA.index_select(0, e_AB_source)
    ZB_q = ZB.index_select(0, e_AB_target)
    qA -= ZA
    qB -= ZB

    # Identity for 3D
    delta = torch.eye(3, device=qA.device)

    # Extracting tensor elements
    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_source = qB.squeeze(-1).index_select(0, e_AB_target)

    muA_source = muA.index_select(0, e_AB_source)
    muB_source = muB.index_select(0, e_AB_target)

    # TF implementation uses 3/2 factor for quadrupoles
    # quadA_source = (3.0 / 2.0) * quadA.index_select(0, e_AB_source)
    # quadB_source = (3.0 / 2.0) * quadB.index_select(0, e_AB_target)
    quadA_source = quadA.index_select(0, e_AB_source)
    quadB_source = quadB.index_select(0, e_AB_target)

    E_qq = torch.einsum("x,x,x->x", qA_source, qB_source, oodR)

    T1 = torch.einsum("x,xy->xy", oodR**3, -1.0 * dR_xyz)
    qu = torch.einsum("x,xy->xy", qA_source, muB_source) - torch.einsum(
        "x,xy->xy", qB_source, muA_source
    )
    E_qu = torch.einsum("xy,xy->x", T1, qu)

    T2 = 3 * torch.einsum("xy,xz->xyz", dR_xyz, dR_xyz) - torch.einsum(
        "x,x,yz->xyz", dR, dR, delta
    )
    T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)

    E_uu = -1.0 * torch.einsum("xy,xz,xyz->x", muA_source, muB_source, T2)

    qA_quadB_source = torch.einsum("x,xyz->xyz", qA_source, quadB_source)
    qB_quadA_source = torch.einsum("x,xyz->xyz", qB_source, quadA_source)
    E_qQ = torch.einsum("xyz,xyz->x", T2, qA_quadB_source + qB_quadA_source) / Q_const

    # ZA-ZB
    E_ZA_ZB = torch.einsum("x,x,x->x", ZA_q, ZB_q, oodR)

    # TODO Z-M damping
    # ZA-MB
    E_ZA_qB = torch.einsum("x,x,x->x", ZA_q, qB_source, oodR)
    E_ZA_uB = torch.einsum("xy,x,xy->x", T1, ZA_q, muB_source)
    E_ZA_QB = torch.einsum("xyz,x,xyz->x", T2, ZA_q, quadB_source) / Q_const
    E_ZA_MB = E_ZA_qB + E_ZA_uB + E_ZA_QB
    # ZB-MA
    E_ZB_qA = torch.einsum("x,x,x->x", ZB_q, qA_source, oodR)
    E_ZB_uA = torch.einsum("xy,x,xy->x", -T1, ZB_q, muA_source)
    E_ZB_QA = torch.einsum("xyz,x,xyz->x", T2, ZB_q, quadA_source) / Q_const
    E_ZB_MA = E_ZB_qA + E_ZB_uA + E_ZB_QA

    E_elst = 627.509 * (E_qq + E_qu + E_qQ + E_uu + E_ZA_ZB + E_ZA_MB + E_ZB_MA)
    return E_elst


# @torch.compile
def mtp_elst_damping(
    ZA,
    RA,
    qA_0,
    muA,
    quadA,
    Ka,
    ZB,
    RB,
    qB_0,
    muB,
    quadB,
    Kb,
    e_AB_source,
    e_AB_target,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
):
    dR_ang, dR_xyz_ang = get_distances(RA, RB, e_AB_source, e_AB_target)
    dR = dR_ang / constants.au2ang
    dR_xyz = dR_xyz_ang / constants.au2ang
    oodR = 1.0 / dR
    delta = torch.eye(3, device=qA_0.device)

    lam1, lam3, lam5 = elst_damping_mtp_mtp_torch(Ka, Kb, dR, e_AB_source, e_AB_target)
    lam1_ZA_MB, lam3_ZA_MB, lam5_ZA_MB, lam1_ZB_MA, lam3_ZB_MA, lam5_ZB_MA = (
        elst_damping_Z_mtp_torch(Ka, Kb, dR, e_AB_source, e_AB_target)
    )

    # Nuclear Charge Subtraction - pre-compute all index selections
    ZA_q = ZA.index_select(0, e_AB_source)
    ZB_q = ZB.index_select(0, e_AB_target)
    qA = qA_0 - ZA
    qB = qB_0 - ZB
    # Extracting tensor elements - pre-compute all selections
    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_source = qB.squeeze(-1).index_select(0, e_AB_target)
    muA_source = muA.index_select(0, e_AB_source)
    muB_source = muB.index_select(0, e_AB_target)
    quadA_source = quadA.index_select(0, e_AB_source)
    quadB_source = quadB.index_select(0, e_AB_target)

    E_qq = torch.einsum("x,x,x,x->x", qA_source, qB_source, oodR, lam1)

    T1 = torch.einsum("x,xy->xy", oodR**3, -1.0 * dR_xyz)
    qu = torch.einsum("x,xy->xy", qA_source, muB_source) - torch.einsum(
        "x,xy->xy", qB_source, muA_source
    )
    E_qu = torch.einsum("xy,xy,x->x", T1, qu, lam3)

    # Pre-compute common T2 components to avoid redundant calculations
    # dR_xyz[:, :, None] * dR_xyz[:, None, :]
    dR_outer = torch.einsum("xy,xz->xyz", dR_xyz, dR_xyz)
    dR_squared_delta = torch.einsum("x,x,yz->xyz", dR, dR, delta)

    # Main T2 for E_uu and E_qQ
    T2_main = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3
    )
    T2_main = torch.einsum("x,xyz->xyz", oodR**5, T2_main)

    E_uu = -1.0 * torch.einsum("xy,xz,xyz->x", muA_source, muB_source, T2_main)

    qA_quadB_source = torch.einsum("x,xyz->xyz", qA_source, quadB_source)
    qB_quadA_source = torch.einsum("x,xyz->xyz", qB_source, quadA_source)
    E_qQ = (
        torch.einsum("xyz,xyz->x", T2_main, qA_quadB_source + qB_quadA_source) / Q_const
    )

    # ZA-ZB
    E_ZA_ZB = torch.einsum("x,x,x->x", ZA_q, ZB_q, oodR)

    # ZA-MB - reuse T1, compute specialized T2
    E_ZA_MB = torch.einsum("x,x,x,x->x", ZA_q, qB_source, oodR, lam1_ZA_MB)
    E_ZA_MB += torch.einsum("xy,x,x,xy->x", T1, lam3_ZA_MB, ZA_q, muB_source)
    T2_ZA_MB = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5_ZA_MB) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3_ZA_MB
    )
    T2_ZA_MB = torch.einsum("x,xyz->xyz", oodR**5, T2_ZA_MB)
    E_ZA_MB += torch.einsum("xyz,x,xyz->x", T2_ZA_MB, ZA_q, quadB_source) / Q_const

    # ZB-MA - reuse T1, compute specialized T2
    T2_ZB_MA = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5_ZB_MA) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3_ZB_MA
    )
    T2_ZB_MA = torch.einsum("x,xyz->xyz", oodR**5, T2_ZB_MA)
    E_ZB_MA = torch.einsum("x,x,x,x->x", ZB_q, qA_source, oodR, lam1_ZB_MA)
    E_ZB_MA += torch.einsum("xy,x,x,xy->x", -T1, lam3_ZB_MA, ZB_q, muA_source)
    E_ZB_MA += torch.einsum("xyz,x,xyz->x", T2_ZB_MA, ZB_q, quadA_source) / Q_const
    E_elst = 627.509 * (E_qq + E_qu + E_qQ + E_uu + E_ZA_ZB + E_ZA_MB + E_ZB_MA)
    return E_elst


@torch.compile
def distance_tensors(
    Ri, Rj, e_source, e_target, alpha_A=None, alpha_B=None, thole_damping_param=0.39
):
    dR_ang, dR_xyz_ang = get_distances(Ri, Rj, e_source, e_target)
    dR_xyz = dR_xyz_ang / constants.au2ang
    dR = dR_ang / constants.au2ang
    alpha_i = alpha_A.index_select(0, e_source)
    alpha_j = alpha_B.index_select(0, e_target)
    u = dR / ((alpha_i * alpha_j) ** (1.0 / 6.0))
    au3 = thole_damping_param * (u**3)
    lam_3 = 1 - torch.exp(-au3)
    lam_5 = 1 - (1 + au3) * torch.exp(-au3)
    delta = torch.eye(3, device=dR.device)
    oodR = 1.0 / dR
    T1 = torch.einsum("x,xy,x->xy", oodR**3, -1.0 * dR_xyz, lam_3)
    T2 = 3 * torch.einsum("xy,xz,x->xyz", dR_xyz, dR_xyz, lam_5) - torch.einsum(
        "x,x,yz,x->xyz", dR, dR, delta, lam_3
    )
    T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)
    return dR, dR_xyz, oodR, T1, T2


@torch.compile
def induced_dipole_induction(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    ZB,
    RB,
    qB,
    muB,
    quadB,
    e_AB_source,
    e_AB_target,
    e_AA_source,
    e_BB_source,
    e_AA_target,
    e_BB_target,
    hirshfeld_volume_ratio_A: torch.tensor,
    hirshfeld_volume_ratio_B: torch.tensor,
    valence_widths_A: torch.tensor,
    valence_widths_B: torch.tensor,
    Ka: torch.tensor,
    Kb: torch.tensor,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param: float = 0.39,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
    polarizability_table=constants.polarizability_table,
) -> float:
    """
    Calculate the induced dipole interaction energy between two molecules using
    their multipole moments and Hirshfeld volume ratios. Follow classical
    induction model from this paper:
    https://pubs.aip.org/aip/jcp/article/154/18/184110/200216/CLIFF-A-component-based-machine-learned
    """
    delta = torch.eye(3, device=qA.device)
    h2kcalmol = constants.h2kcalmol  # Hartree to kcal/mol conversion factor

    alpha_0_A = torch.zeros_like(hirshfeld_volume_ratio_A)
    alpha_0_B = torch.zeros_like(hirshfeld_volume_ratio_B)

    # Use index_select for vectorized lookup
    alpha_0_A = torch.index_select(polarizability_table, 0, ZA.long())
    alpha_0_B = torch.index_select(polarizability_table, 0, ZB.long())
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A ** (4 / 3.0)
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B ** (4 / 3.0)

    # Calculate interaction tensors between atoms
    dR_AB, dR_AB_xyz, T0_AB, T1_AB, T2_AB = distance_tensors(
        RA, RB, e_AB_source, e_AB_target, alpha_A, alpha_B, thole_damping_param
    )
    dR_AA, dR_AA_xyz, T0_AA, T1_AA, T2_AA = distance_tensors(
        RA, RA, e_AA_source, e_AA_target, alpha_A, alpha_A, thole_damping_param
    )
    dR_BB, dR_BB_xyz, T0_BB, T1_BB, T2_BB = distance_tensors(
        RB, RB, e_BB_source, e_BB_target, alpha_B, alpha_B, thole_damping_param
    )

    # TODO PASS DAMPING PARAM;
    # Select relevant tensors for atom pairs
    alpha_A_source = alpha_A.index_select(0, e_AB_source)
    alpha_B_target = alpha_B.index_select(0, e_AB_target)

    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    alpha_BB_target = alpha_B.index_select(0, e_BB_target)

    # Need to ensure that qA and qB are right shape even when ions
    qA = qA.reshape(-1, 1)
    qB = qB.reshape(-1, 1)
    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_target = qB.squeeze(-1).index_select(0, e_AB_target)

    muA_source = muA.index_select(0, e_AB_source)
    muB_target = muB.index_select(0, e_AB_target)

    # Initialize tensors for induced dipoles
    n_atoms_A = RA.shape[0]
    n_atoms_B = RB.shape[0]

    K_A_source = Ka.index_select(0, e_AB_source)
    K_B_target = Kb.index_select(0, e_AB_target)
    # Must have sigma be > 0 to avoid NaNs
    sigma_A_source = valence_widths_A.index_select(0, e_AB_source)
    sigma_B_target = valence_widths_B.index_select(0, e_AB_target)
    B_ij = torch.sqrt(1.0 / (sigma_A_source * sigma_B_target))
    S_ij = (1.0 / 3.0 * (B_ij * dR_AB) ** 2 + B_ij * dR_AB + 1.0) * torch.exp(
        -B_ij * dR_AB
    )
    E_ind_overlap = K_A_source * S_ij * K_B_target * h2kcalmol

    # Calculate initial induced dipoles
    # A: Induced by B's multipoles
    mu_induced_0_A = torch.zeros((n_atoms_A, 3), device=qA.device)
    mu_induced_0_B = torch.zeros((n_atoms_B, 3), device=qB.device)

    # Calculate initial induced dipoles from molecule B's multipoles on molecule A
    # Contribution from charges
    mu_charge_A = torch.einsum("a,ai,a->ai", alpha_A_source, T1_AB, qB_target)
    mu_induced_0_A = scatter(
        mu_charge_A, e_AB_source, dim=0, reduce="sum", dim_size=n_atoms_A
    )
    mu_dipole_A = torch.einsum("a,aij,aj->ai", alpha_A_source, T2_AB, muB_target)
    mu_induced_0_A += scatter(
        mu_dipole_A, e_AB_source, dim=0, reduce="sum", dim_size=n_atoms_A
    )

    mu_charge_B = torch.einsum("a,ai,a->ai", alpha_B_target, -T1_AB, qA_source)
    mu_induced_0_B = scatter(
        mu_charge_B, e_AB_target, dim=0, reduce="sum", dim_size=n_atoms_B
    )
    mu_dipole_B = torch.einsum("a,aij,aj->ai", alpha_B_target, T2_AB, muA_source)
    mu_induced_0_B += scatter(
        mu_dipole_B, e_AB_target, dim=0, reduce="sum", dim_size=n_atoms_B
    )

    # Self-consistent induced dipole iterations
    mu_induced_A = mu_induced_0_A.clone()
    mu_induced_B = mu_induced_0_B.clone()

    # Pre-compute index selections to avoid repeated operations in the loop
    mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
    mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
    mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
    mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

    # Iterative SCF procedure to converge induced dipoles
    for iteration in range(max_iterations):
        mu_induced_A_old = mu_induced_A.clone()
        mu_induced_B_old = mu_induced_B.clone()

        ####### (A) INDUCED DIPOLES ########
        # Induced dipoles on A due to induced dipoles on B
        mu_induced_A_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_A_source, T2_AB, mu_induced_B_at_AB_target
        )
        mu_induced_A_new = scatter(
            mu_induced_A_due_B, e_AB_source, dim=0, reduce="sum", dim_size=n_atoms_A
        )
        # Induced dipoles on A due to induced dipoles on A
        mu_induced_A_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_AA_target, T2_AA, mu_induced_A_at_AA_source
        )
        mu_induced_A_new += scatter(
            mu_induced_A_due_A, e_AA_target, dim=0, reduce="sum", dim_size=n_atoms_A
        )
        mu_induced_A_new += mu_induced_0_A

        ####### (B) INDUCED DIPOLES ########
        # Induced dipoles on B due to induced dipoles on A
        mu_induced_B_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_B_target, T2_AB, mu_induced_A_at_AB_source
        )
        mu_induced_B_new = scatter(
            mu_induced_B_due_A, e_AB_target, dim=0, reduce="sum", dim_size=n_atoms_B
        )
        # Induced dipoles on B due to induced dipoles on B
        mu_induced_B_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_BB_target, T2_BB, mu_induced_B_at_BB_source
        )
        mu_induced_B_new += scatter(
            mu_induced_B_due_B, e_BB_target, dim=0, reduce="sum", dim_size=n_atoms_B
        )
        mu_induced_B_new += mu_induced_0_B

        # Apply mixing
        mu_induced_A = (1 - omega) * mu_induced_A_old + omega * mu_induced_A_new
        mu_induced_B = (1 - omega) * mu_induced_B_old + omega * mu_induced_B_new

        # Update pre-computed index selections for next iteration
        mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
        mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
        mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
        mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

        # Check convergence
        delta_A = torch.norm(mu_induced_A - mu_induced_A_old)
        delta_B = torch.norm(mu_induced_B - mu_induced_B_old)
        delta = max(delta_A, delta_B)
        if delta < convergence_threshold:
            break
    muA_induced_source = mu_induced_A.index_select(0, e_AB_source)
    muB_induced_target = mu_induced_B.index_select(0, e_AB_target)
    qu = torch.einsum("x,xy->xy", qA_source, muB_induced_target) - torch.einsum(
        "x,xy->xy", qB_target, muA_induced_source
    )
    E_qu = torch.einsum("xy,xy->x", T1_AB, qu) * h2kcalmol
    E_uu = (
        -1.0
        * (
            torch.einsum("xy,xz,xyz->x", muA_induced_source, muB_target, T2_AB)
            + torch.einsum("xy,xz,xyz->x", muA_source, muB_induced_target, T2_AB)
        )
        * h2kcalmol
    )
    E_ind = (E_qu + E_uu) / 2.0
    E_ind -= E_ind_overlap
    return E_ind


@torch.compile
def induced_dipole_induction_optimized(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    ZB,
    RB,
    qB,
    muB,
    quadB,
    e_AB_source,
    e_AB_target,
    e_AA_source,
    e_BB_source,
    e_AA_target,
    e_BB_target,
    hirshfeld_volume_ratio_A: torch.tensor,
    hirshfeld_volume_ratio_B: torch.tensor,
    valence_widths_A: torch.tensor,
    valence_widths_B: torch.tensor,
    Ka: torch.tensor,
    Kb: torch.tensor,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param: float = 0.39,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
    polarizability_table=constants.polarizability_table,
) -> float:
    """
    Optimized version of induced_dipole_induction with reduced index_select and scatter operations.
    """

    delta = torch.eye(3, device=qA.device)
    h2kcalmol = constants.h2kcalmol  # Hartree to kcal/mol conversion factor

    alpha_0_A = torch.zeros_like(hirshfeld_volume_ratio_A)
    alpha_0_B = torch.zeros_like(hirshfeld_volume_ratio_B)

    # Use index_select for vectorized lookup
    alpha_0_A = torch.index_select(polarizability_table, 0, ZA.long())
    alpha_0_B = torch.index_select(polarizability_table, 0, ZB.long())
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A ** (4 / 3.0)
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B ** (4 / 3.0)

    # Calculate interaction tensors between atoms
    dR_AB, dR_AB_xyz, T0_AB, T1_AB, T2_AB = distance_tensors(
        RA, RB, e_AB_source, e_AB_target, alpha_A, alpha_B, thole_damping_param
    )
    dR_AA, dR_AA_xyz, T0_AA, T1_AA, T2_AA = distance_tensors(
        RA, RA, e_AA_source, e_AA_target, alpha_A, alpha_A, thole_damping_param
    )
    dR_BB, dR_BB_xyz, T0_BB, T1_BB, T2_BB = distance_tensors(
        RB, RB, e_BB_source, e_BB_target, alpha_B, alpha_B, thole_damping_param
    )

    # Select relevant tensors for atom pairs
    alpha_A_source = alpha_A.index_select(0, e_AB_source)
    alpha_B_target = alpha_B.index_select(0, e_AB_target)

    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    alpha_BB_target = alpha_B.index_select(0, e_BB_target)

    # Need to ensure that qA and qB are right shape even when ions
    qA = qA.reshape(-1, 1)
    qB = qB.reshape(-1, 1)
    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_target = qB.squeeze(-1).index_select(0, e_AB_target)

    muA_source = muA.index_select(0, e_AB_source)
    muB_target = muB.index_select(0, e_AB_target)

    # Initialize tensors for induced dipoles
    n_atoms_A = RA.shape[0]
    n_atoms_B = RB.shape[0]

    K_A_source = Ka.index_select(0, e_AB_source)
    K_B_target = Kb.index_select(0, e_AB_target)
    sigma_A_source = valence_widths_A.index_select(0, e_AB_source)
    sigma_B_target = valence_widths_B.index_select(0, e_AB_target)
    B_ij = torch.sqrt(1.0 / (sigma_A_source * sigma_B_target))
    S_ij = (1.0 / 3.0 * (B_ij * dR_AB) ** 2 + B_ij * dR_AB + 1.0) * torch.exp(
        -B_ij * dR_AB
    )
    E_ind_overlap = K_A_source * S_ij * K_B_target * h2kcalmol

    # Calculate initial induced dipoles
    mu_induced_0_A = torch.zeros((n_atoms_A, 3), device=qA.device)
    mu_induced_0_B = torch.zeros((n_atoms_B, 3), device=qB.device)

    # Calculate initial induced dipoles from molecule B's multipoles on molecule A
    mu_charge_A = torch.einsum("a,ai,a->ai", alpha_A_source, T1_AB, qB_target)
    mu_induced_0_A = scatter(
        mu_charge_A, e_AB_source, dim=0, reduce="sum", dim_size=n_atoms_A
    )
    mu_dipole_A = torch.einsum("a,aij,aj->ai", alpha_A_source, T2_AB, muB_target)
    mu_induced_0_A += scatter(
        mu_dipole_A, e_AB_source, dim=0, reduce="sum", dim_size=n_atoms_A
    )

    mu_charge_B = torch.einsum("a,ai,a->ai", alpha_B_target, -T1_AB, qA_source)
    mu_induced_0_B = scatter(
        mu_charge_B, e_AB_target, dim=0, reduce="sum", dim_size=n_atoms_B
    )
    mu_dipole_B = torch.einsum("a,aij,aj->ai", alpha_B_target, T2_AB, muA_source)
    mu_induced_0_B += scatter(
        mu_dipole_B, e_AB_target, dim=0, reduce="sum", dim_size=n_atoms_B
    )

    # Self-consistent induced dipole iterations
    mu_induced_A = mu_induced_0_A.clone()
    mu_induced_B = mu_induced_0_B.clone()

    # Pre-compute index selections to avoid repeated operations in the loop
    mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
    mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
    mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
    mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

    # Iterative SCF procedure to converge induced dipoles
    for iteration in range(max_iterations):
        mu_induced_A_old = mu_induced_A.clone()
        mu_induced_B_old = mu_induced_B.clone()

        # Update pre-computed selections
        mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
        mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
        mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
        mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

        ####### (A) INDUCED DIPOLES ########
        # Induced dipoles on A due to induced dipoles on B
        mu_induced_A_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_A_source, T2_AB, mu_induced_B_at_AB_target
        )
        mu_induced_A_new = scatter(
            mu_induced_A_due_B, e_AB_source, dim=0, reduce="sum", dim_size=n_atoms_A
        )
        # Induced dipoles on A due to induced dipoles on A
        mu_induced_A_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_AA_target, T2_AA, mu_induced_A_at_AA_source
        )
        mu_induced_A_new += scatter(
            mu_induced_A_due_A, e_AA_target, dim=0, reduce="sum", dim_size=n_atoms_A
        )
        mu_induced_A_new += mu_induced_0_A

        ####### (B) INDUCED DIPOLES ########
        # Induced dipoles on B due to induced dipoles on A
        mu_induced_B_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_B_target, T2_AB, mu_induced_A_at_AB_source
        )
        mu_induced_B_new = scatter(
            mu_induced_B_due_A, e_AB_target, dim=0, reduce="sum", dim_size=n_atoms_B
        )
        # Induced dipoles on B due to induced dipoles on B
        mu_induced_B_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_BB_target, T2_BB, mu_induced_B_at_BB_source
        )
        mu_induced_B_new += scatter(
            mu_induced_B_due_B, e_BB_target, dim=0, reduce="sum", dim_size=n_atoms_B
        )
        mu_induced_B_new += mu_induced_0_B

        # Apply mixing
        mu_induced_A = (1 - omega) * mu_induced_A_old + omega * mu_induced_A_new
        mu_induced_B = (1 - omega) * mu_induced_B_old + omega * mu_induced_B_new

        # Check convergence
        delta_A = torch.norm(mu_induced_A - mu_induced_A_old)
        delta_B = torch.norm(mu_induced_B - mu_induced_B_old)
        delta = max(delta_A, delta_B)
        if delta < convergence_threshold:
            break

    # Final energy calculation
    muA_induced_source = mu_induced_A.index_select(0, e_AB_source)
    muB_induced_target = mu_induced_B.index_select(0, e_AB_target)
    qu = torch.einsum("x,xy->xy", qA_source, muB_induced_target) - torch.einsum(
        "x,xy->xy", qB_target, muA_induced_source
    )
    E_qu = torch.einsum("xy,xy->x", T1_AB, qu) * h2kcalmol
    E_uu = (
        -1.0
        * (
            torch.einsum("xy,xz,xyz->x", muA_induced_source, muB_target, T2_AB)
            + torch.einsum("xy,xz,xyz->x", muA_source, muB_induced_target, T2_AB)
        )
        * h2kcalmol
    )
    E_ind = (E_qu + E_uu) / 2.0
    E_ind -= E_ind_overlap
    return E_ind


def isolate_atom_parameter_predictions(batch, output):
    batch_size = batch.natom_per_mol.size(0)
    q = output[0]
    mu = output[1]
    th = output[2]
    hlist = output[3]
    K = output[4]
    mol_charges = [[] for i in range(batch_size)]
    mol_dipoles = [[] for i in range(batch_size)]
    mol_qpoles = [[] for i in range(batch_size)]
    mol_hlist = [[] for i in range(batch_size)]
    mol_K = [[] for i in range(batch_size)]
    i_offset = 0
    for n, i in enumerate(batch.natom_per_mol):
        mol_charges[n] = q[i_offset : i_offset + i]
        mol_dipoles[n] = mu[i_offset : i_offset + i]
        mol_qpoles[n] = th[i_offset : i_offset + i]
        mol_hlist[n] = hlist[i_offset : i_offset + i]
        mol_K[n] = K[i_offset : i_offset + i]
        i_offset += i
    return mol_charges, mol_dipoles, mol_qpoles, mol_hlist, mol_K


def isolate_atom_parameter_predictions_ap3(batch, output):
    batch_size = batch.natom_per_mol.size(0)
    q = output[0]
    mu = output[1]
    th = output[2]
    hfvr = output[3]
    vw = output[4]
    hlist = output[5]
    K = output[6]
    mol_charges = [[] for i in range(batch_size)]
    mol_dipoles = [[] for i in range(batch_size)]
    mol_qpoles = [[] for i in range(batch_size)]
    mol_hfvr = [[] for i in range(batch_size)]
    mol_vw = [[] for i in range(batch_size)]
    mol_hlist = [[] for i in range(batch_size)]
    mol_K = [[] for i in range(batch_size)]
    i_offset = 0
    for n, i in enumerate(batch.natom_per_mol):
        mol_charges[n] = q[i_offset : i_offset + i]
        mol_dipoles[n] = mu[i_offset : i_offset + i]
        mol_qpoles[n] = th[i_offset : i_offset + i]
        mol_hfvr[n] = hfvr[i_offset : i_offset + i]
        mol_vw[n] = vw[i_offset : i_offset + i]
        mol_hlist[n] = hlist[i_offset : i_offset + i]
        mol_K[n] = K[i_offset : i_offset + i]
        i_offset += i
    return mol_charges, mol_dipoles, mol_qpoles, mol_hfvr, mol_vw, mol_hlist, mol_K


class AM_DimerParam_Model:
    def __init__(
        self,
        dataset=None,
        atom_model=None,
        atom_model_type="AtomMPNN",
        pre_trained_model_path=None,
        atom_model_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        param_start_mean=[1.7],
        param_start_std=[0.01],
        n_params=1,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_root="data",
        ds_max_size=None,
        ds_atomic_batch_size=200,
        ds_force_reprocess=False,
        ds_skip_process=False,
        ds_skip_compile=False,
        ds_num_devices=1,
        ds_datapoint_storage_n_objects=1000,
        ds_prebatched=False,
        ds_random_seed=42,
        print_lvl=0,
        ds_qcel_molecules=None,
        ds_energy_labels=None,
        dimer_eval_type="elst_damping",
    ):
        """
        If pre_trained_model_path is provided, the model will be loaded from
        the path and all other parameters will be ignored except for dataset.

        use_GPU will check for a GPU and use it if available unless set to false.
        """
        if torch.cuda.is_available() and use_GPU is not False:
            device = torch.device("cuda:0")
            print("running on the GPU")
        else:
            device = torch.device("cpu")
            print("running on the CPU")
        self.ds_spec_type = ds_spec_type
        # TODO UPDATE TO AP3
        if atom_model_type == "AtomMPNN":
            self.atom_model = AtomMPNN()
            am_type = AtomMPNN
        elif atom_model_type == "AtomHirshfeldMPNN":
            self.atom_model = AtomHirshfeldMPNN()
            am_type = AtomHirshfeldMPNN
        elif atom_model_type == "AtomTypeParamNN":
            self.atom_model = AtomTypeParamNN()
            am_type = AtomTypeParamNN
        else:
            raise ValueError(f"Unknown atom_model_type: {atom_model_type}")

        if atom_model_pre_trained_path:
            print(
                f"Loading pre-trained AtomMPNN model from {atom_model_pre_trained_path}"
            )
            checkpoint = torch.load(
                atom_model_pre_trained_path, map_location=device, weights_only=False
            )
            if atom_model_type in ["AtomHirshfeldMPNN", "AtomMPNN"]:
                self.atom_model = am_type(
                    n_message=checkpoint["config"]["n_message"],
                    n_rbf=checkpoint["config"]["n_rbf"],
                    n_neuron=checkpoint["config"]["n_neuron"],
                    n_embed=checkpoint["config"]["n_embed"],
                    r_cut=checkpoint["config"]["r_cut"],
                )
            elif atom_model_type == "AtomTypeParamNN":
                self.atom_model = am_type(
                    n_message=checkpoint["config"]["n_message"],
                    n_neuron=checkpoint["config"]["n_neuron"],
                    n_embed=checkpoint["config"]["n_embed"],
                    param_start_mean=checkpoint["config"]["param_start_mean"],
                    param_start_std=checkpoint["config"]["param_start_std"],
                    n_params=checkpoint["config"].get("n_params", 1),
                )
            # model_state_dict = checkpoint["model_state_dict"]
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.atom_model.load_state_dict(model_state_dict)
        elif atom_model:
            print("Using provided AtomMPNN model:", atom_model)
            self.atom_model = atom_model
        else:
            print(
                """No atom model provided.
    Assuming atomic multipoles and embeddings are
    pre-computed and passed as input to the model.
"""
            )
        if pre_trained_model_path:
            print(f"Loading pre-trained MTP-MTP model from {pre_trained_model_path}")
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = AtomTypeParamNN(
                atom_model=self.atom_model,
                n_message=checkpoint["config"]["n_message"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                param_start_mean=checkpoint["config"]["param_start_mean"],
                param_start_std=checkpoint["config"]["param_start_std"],
                n_params=checkpoint["config"].get("n_params", 1),
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = AtomTypeParamNN(
                atom_model=self.atom_model,
                n_message=n_message,
                n_neuron=n_neuron,
                n_embed=n_embed,
                param_start_mean=param_start_mean,
                param_start_std=param_start_std,
                n_params=n_params,
            )
        self.n_params = n_params
        self.dimer_eval_type = dimer_eval_type
        self.dimer_model = DimerProp(self.model, dimer_eval=dimer_eval_type)
        if self.dimer_eval_type in ["elst", "elst_damping"]:
            self.dimer_model_elst = DimerProp(self.model, dimer_eval="elst")
        else:
            self.dimer_model_elst = None

        if n_message != self.model.n_message:
            print(f"Changing n_mesage from {self.model.n_message} to {n_message}")
            self.model.n_message = n_message
        if n_neuron != self.model.n_neuron:
            print(f"Changing n_neuron from {self.model.n_neuron} to {n_neuron}")
            self.model.n_neuron = n_neuron
        if n_embed != self.model.n_embed:
            print(f"Changing n_embed from {self.model.n_embed} to {n_embed}")
            self.model.n_embed = n_embed
        # Handle param_start_mean/std as lists
        if isinstance(param_start_mean, (list, tuple)):
            if param_start_mean != self.model.param_start_mean:
                print(f"Changing param_start_mean to {param_start_mean}")
                self.model.param_start_mean = param_start_mean
        else:
            # Scalar case, check if different from all current values
            if not all(p == param_start_mean for p in self.model.param_start_mean):
                print(f"Changing param_start_mean to {param_start_mean}")
                self.model.param_start_mean = [param_start_mean] * self.model.n_params

        if isinstance(param_start_std, (list, tuple)):
            if param_start_std != self.model.param_start_std:
                print(f"Changing param_start_std to {param_start_std}")
                self.model.param_start_std = param_start_std
        else:
            # Scalar case
            if not all(p == param_start_std for p in self.model.param_start_std):
                print(f"Changing param_start_std to {param_start_std}")
                self.model.param_start_std = [param_start_std] * self.model.n_params

        self.device = device
        self.atom_model.to(device)
        self.model.to(device)
        self.dimer_model.to(device)

        split_dbs = [2, 5, 6, 7]
        ds_qcel_split_db = (
            ds_qcel_molecules is not None
            and len(ds_qcel_molecules) == 2
            and isinstance(ds_qcel_molecules[0], list)
        )
        self.dataset = dataset
        if (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type not in split_dbs
            and not ds_qcel_split_db
        ):

            def setup_ds(fp=ds_force_reprocess):
                return ap2_fused_module_dataset(
                    root=ds_root,
                    r_cut=r_cut,
                    r_cut_im=torch.inf,
                    spec_type=ds_spec_type,
                    max_size=ds_max_size,
                    force_reprocess=fp,
                    atom_model=self.atom_model,
                    atomic_batch_size=ds_atomic_batch_size,
                    num_devices=ds_num_devices,
                    skip_processed=ds_skip_process,
                    skip_compile=ds_skip_compile,
                    random_seed=ds_random_seed,
                    datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                    print_level=print_lvl,
                    qcel_molecules=ds_qcel_molecules,
                    energy_labels=ds_energy_labels,
                    # storage_type="h5",  # "pt" or "h5" for storage format
                    
                )

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset = self.dataset[:ds_max_size]
        elif (
            not ignore_database_null
            and self.dataset is None
            and (self.ds_spec_type in split_dbs or ds_qcel_split_db)
        ):
            print("Processing Split dataset...")
            if ds_qcel_molecules is None:
                ds_qcel_molecules = [None, None]
                ds_energy_labels = [None, None]

            def setup_ds(fp=ds_force_reprocess):
                return [
                    ap2_fused_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=torch.inf,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model=self.atom_model,
                        atomic_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        skip_compile=ds_skip_compile,
                        random_seed=ds_random_seed,
                        split="train",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        print_level=print_lvl,
                        qcel_molecules=ds_qcel_molecules[0],
                        energy_labels=ds_energy_labels[0],
                        # storage_type="h5",  # "pt" or "h5" for storage format
                    ),
                    ap2_fused_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=torch.inf,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model=self.atom_model,
                        atomic_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        skip_compile=ds_skip_compile,
                        random_seed=ds_random_seed,
                        split="test",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        print_level=print_lvl,
                        qcel_molecules=ds_qcel_molecules[1],
                        energy_labels=ds_energy_labels[1],
                        # storage_type="h5",  # "pt" or "h5" for storage format
                    ),
                ]

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset[0] = self.dataset[0][:ds_max_size]
                self.dataset[1] = self.dataset[1][:ds_max_size]
        print(f"{self.dataset=}")
        self.batch_size = None
        self.shuffle = False
        self.model_save_path = None
        return

    @torch.inference_mode()
    def predict_from_dataset(self):
        self.model.eval()
        for batch in self.dataset:
            batch = batch.to(self.device)
            self.model(batch)
        return

    def compile_model(self):
        self.model.to(self.device)
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
        torch._dynamo.config.capture_scalar_outputs = False
        # torch._dynamo.config.capture_scalar_outputs = True
        self.model = torch.compile(self.model, dynamic=True)
        return

    def set_all_weights_to_value(self, value: float):
        """
        Sets the weights of the model to a constant value for debugging.
        """
        batch = self.example_input()
        batch.to(self.device)
        self.model(batch)
        set_weights_to_value(self.model, value)
        return

    def set_pretrained_model(
        self, ap2_model_path=None, am_model_path=None, model_id=None
    ):
        if model_id is not None:
            ap2_model_path = resources.files("apnet_pt").joinpath(
                "models", "ap2-fused_ensemble", f"ap2_{model_id}.pt"
            )
        elif ap2_model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided.")

        checkpoint = torch.load(ap2_model_path)
        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        return self

    def _qcel_example_input(
        self,
        mols,
        batch_size=1,
        r_cut=999999,
    ):
        dimer_batch = ap2_fused_collate_update_no_target(
            [
                qcel_dimer_to_fused_data(
                    mol, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                )
                for n, mol in enumerate(mols)
            ]
        )
        batch = Data(
            x=dimer_batch.ZA,
            R=dimer_batch.RA,
            edge_index=torch.vstack((dimer_batch.e_AA_source, dimer_batch.e_AA_target)),
            molecule_ind=dimer_batch.molecule_ind_A,
            total_charge=dimer_batch.total_charge_A,
            natom_per_mol=dimer_batch.natom_per_mol_A,
        )
        batch.to(self.device)
        return batch

    def _qcel_dimer_example_input(
        self,
        mols,
        batch_size=1,
        r_cut=999999,
    ):
        batch = ap2_fused_collate_update_no_target(
            [
                qcel_dimer_to_fused_data(
                    mol, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                )
                for n, mol in enumerate(mols)
            ]
        )
        batch.to(self.device)
        return batch

    def _assemble_pairs(
        self,
        inp_batch,
        E_sr_dimer,
        E_sr,
        E_elst_sr,
        E_elst_lr,
    ):
        indA_to_dimer = []
        indB_to_dimer = []
        indA_to_atom = []
        indB_to_atom = []
        pair_energies_batch = []

        indsA_sr = inp_batch["e_ABsr_source"]
        indsB_sr = inp_batch["e_ABsr_target"]
        indsA_lr = inp_batch["e_ABlr_source"]
        indsB_lr = inp_batch["e_ABlr_target"]

        dimer_inds, atoms_per_dimer = torch.unique(
            inp_batch.dimer_ind, return_counts=True
        )
        indsA_monomer = inp_batch.indA
        indsB_monomer = inp_batch.indB

        for i in dimer_inds:
            size_A = torch.sum(indsA_monomer == i)
            size_B = torch.sum(indsB_monomer == i)
            indA_to_dimer.append(np.full((size_A,), i))
            indB_to_dimer.append(np.full((size_B,), i))
            indA_to_atom.append(np.arange(size_A))
            indB_to_atom.append(np.arange(size_B))
            pair_energies_batch.append(np.zeros((4, size_A, size_B)))

        indA_to_dimer = np.concatenate(indA_to_dimer)
        indB_to_dimer = np.concatenate(indB_to_dimer)
        indA_to_atom = np.concatenate(indA_to_atom)
        indB_to_atom = np.concatenate(indB_to_atom)

        # E_sr, E_elst_sr, E_elst_lr
        for e_pair, e_elst_sr, indA, indB in zip(E_sr, E_elst_sr, indsA_sr, indsB_sr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][0:4, atomA, atomB] += e_pair.numpy()
            pair_energies_batch[i][0, atomA, atomB] += e_elst_sr.numpy()

        for e_elst_lr, indA, indB in zip(E_elst_lr, indsA_lr, indsB_lr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][0, atomA, atomB] += e_elst_lr
        return pair_energies_batch

    def _assemble_mtp_pairs(
        self,
        inp_batch,
        E_elst_sr,
        E_elst_lr,
    ):
        indA_to_dimer = []
        indB_to_dimer = []
        indA_to_atom = []
        indB_to_atom = []
        pair_energies_batch = []

        indsA_sr = inp_batch["e_ABsr_source"]
        indsB_sr = inp_batch["e_ABsr_target"]
        indsA_lr = inp_batch["e_ABlr_source"]
        indsB_lr = inp_batch["e_ABlr_target"]

        dimer_inds, atoms_per_dimer = torch.unique(
            inp_batch.dimer_ind, return_counts=True
        )
        indsA_monomer = inp_batch.indA
        indsB_monomer = inp_batch.indB

        for i in dimer_inds:
            size_A = torch.sum(indsA_monomer == i)
            size_B = torch.sum(indsB_monomer == i)
            indA_to_dimer.append(np.full((size_A,), i))
            indB_to_dimer.append(np.full((size_B,), i))
            indA_to_atom.append(np.arange(size_A))
            indB_to_atom.append(np.arange(size_B))
            pair_energies_batch.append(np.zeros((size_A, size_B)))

        indA_to_dimer = np.concatenate(indA_to_dimer)
        indB_to_dimer = np.concatenate(indB_to_dimer)
        indA_to_atom = np.concatenate(indA_to_atom)
        indB_to_atom = np.concatenate(indB_to_atom)
        for e_elst_sr, indA, indB in zip(E_elst_sr, indsA_sr, indsB_sr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][atomA, atomB] += e_elst_sr.numpy()
        for e_elst_lr, indA, indB in zip(E_elst_lr, indsA_lr, indsB_lr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][atomA, atomB] += e_elst_lr
        return pair_energies_batch

    @torch.inference_mode()
    def predict_qcel_mols_dimer(
        self,
        mols,
        batch_size=1,
        r_cut=None,
        verbose=False,
        return_pairs=False,
        return_elst=False,
    ):
        assert not (return_elst and return_pairs), (
            "return_elst and return_pairs are not compatible"
        )
        if r_cut is None and hasattr(self.atom_model, "r_cut"):
            r_cut = self.atom_model.r_cut
        elif hasattr(self.atom_model.atom_model, "r_cut"):
            r_cut = self.atom_model.atom_model.r_cut

        N = len(mols)
        predictions = np.zeros((N, self.n_params))
        if return_pairs or return_elst:
            pairwise_energies = []
        self.atom_model.to(self.device)
        for i in range(0, N, batch_size):
            upper_bound = min(i + batch_size, N)
            dimer_batch = ap2_fused_collate_update_no_target(
                [
                    qcel_dimer_to_fused_data(
                        dimer, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                    )
                    for n, dimer in enumerate(mols[i:upper_bound])
                ]
            )
            dimer_batch.to(device=self.device)
            preds = self.dimer_model(dimer_batch)
            preds = scatter(
                preds,
                dimer_batch.dimer_ind,
                dim=0,
                reduce="add",
                dim_size=torch.tensor(
                    dimer_batch.total_charge_A.size(0), dtype=torch.long
                ),
            )
            predictions[i : i + batch_size] = preds[0].cpu().numpy()
        if verbose:
            print(f"Predictions for {i} to {i + batch_size} out of {N}")
        if return_pairs or return_elst:
            return predictions, pairwise_energies
        return predictions

    @torch.inference_mode()
    def predict_qcel_mols_monomer_props(
        self,
        mols,
        batch_size=1,
        r_cut=None,
        am_type="ap2",
        verbose=False,
    ):
        output_A = []
        output_B = []
        # check if atom_model has r_cut attribute
        if r_cut is None and hasattr(self.atom_model, "r_cut"):
            r_cut = self.atom_model.r_cut
        elif hasattr(self.atom_model.atom_model, "r_cut"):
            r_cut = self.atom_model.atom_model.r_cut
        N = len(mols)
        self.atom_model.to(self.device)
        if am_type == "ap2":
            isolate_fn = isolate_atom_parameter_predictions
        elif am_type == "ap3":
            isolate_fn = isolate_atom_parameter_predictions_ap3
        else:
            raise ValueError(f"Unknown am_type: {am_type}")
        for i in range(0, N, batch_size):
            upper_bound = min(i + batch_size, N)
            dimer_batch = ap2_fused_collate_update_no_target(
                [
                    qcel_dimer_to_fused_data(
                        dimer, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                    )
                    for n, dimer in enumerate(mols[i:upper_bound])
                ]
            )
            batch_A = Data(
                x=dimer_batch.ZA,
                R=dimer_batch.RA,
                edge_index=torch.vstack(
                    (dimer_batch.e_AA_source, dimer_batch.e_AA_target)
                ),
                molecule_ind=dimer_batch.molecule_ind_A,
                total_charge=dimer_batch.total_charge_A,
                natom_per_mol=dimer_batch.natom_per_mol_A,
            )
            with torch.no_grad():
                v = isolate_fn(batch_A, self.model(batch_A))
                output_A.extend(list(zip(*v)))
            batch_B = Data(
                x=dimer_batch.ZB,
                R=dimer_batch.RB,
                edge_index=torch.vstack(
                    (dimer_batch.e_BB_source, dimer_batch.e_BB_target)
                ),
                molecule_ind=dimer_batch.molecule_ind_B,
                total_charge=dimer_batch.total_charge_B,
                natom_per_mol=dimer_batch.natom_per_mol_B,
            )
            with torch.no_grad():
                v = isolate_fn(batch_B, self.model(batch_B))
                output_B.extend(list(zip(*v)))
        return output_A, output_B

    def example_input(
        self,
        mol=None,
        r_cut=5.0,
    ):
        if mol is None:
            mol = qcel.models.Molecule.from_data("""
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
--
0 1
8   2.268880784   0.026340101   0.000508029
1   2.645502399   -0.412039965   0.766632411
1   2.641145101   -0.449872874   -0.744894473
units angstrom
        """)
        return self._qcel_example_input(
            [mol],
            batch_size=1,
            r_cut=r_cut,
        )

    ########################################################################
    # TRAINING/VALIDATION HELPERS
    ########################################################################

    def __setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        if torch.cuda.is_available():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        else:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.manual_seed(43)

    def __cleanup(self):
        dist.destroy_process_group()

    def __train_batches_single_proc(
        self, dataloader, loss_fn, optimizer, rank_device, scheduler, y_ind=0
    ):
        """
        Single-process training loop body.
        """
        self.model.train()
        comp_errors_t = []
        total_loss = 0.0
        for n, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)  # minor speed-up
            batch = batch.to(rank_device, non_blocking=True)
            ref = batch.y[:, y_ind]
            preds = self.dimer_model(batch)
            preds = scatter(
                preds,
                batch.dimer_ind,
                dim=0,
                reduce="add",
                dim_size=torch.tensor(batch.total_charge_A.size(0), dtype=torch.long),
            )
            comp_errors = preds - ref
            # print(f" {preds = }\n {ref = }\n {comp_errors = }")
            batch_loss = (
                torch.mean(torch.square(comp_errors))
                if (loss_fn is None)
                else loss_fn(preds, ref)
            )
            batch_loss.backward()

            # Embedding-specific gradient clipping
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if 'guess_layer' in name:
                        # Debug: Print gradient statistics before clipping
                        grad_min = param.grad.data.min().item()
                        grad_max = param.grad.data.max().item()
                        grad_mean = param.grad.data.mean().item()
                        grad_std = param.grad.data.std().item()
                        has_nan = torch.isnan(param.grad.data).any().item()
                        has_inf = torch.isinf(param.grad.data).any().item()

                        # print(f"[{name}] Gradients - Min: {grad_min:.6f}, Max: {grad_max:.6f}, "
                        #       f"Mean: {grad_mean:.6f}, Std: {grad_std:.6f}, "
                        #       f"NaN: {has_nan}, Inf: {has_inf}")

                        # More aggressive clipping for embeddings
                        param.grad.data.clamp_(-0.001, 0.001)

                        # Debug: Print after clipping
                        # if has_nan or has_inf or abs(grad_max) > 0.01 or abs(grad_min) > 0.01:
                        #     print(f"[{name}] After clipping - Min: {param.grad.data.min().item():.6f}, "
                        #           f"Max: {param.grad.data.max().item():.6f}")
                    else:
                        # Standard clipping for other parameters
                        param.grad.data.clamp_(-1.0, 1.0)

            # More aggressive gradient clipping to prevent NaN in embeddings
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

            # Debug: Print parameter values before optimizer step
            # for name, param in self.model.named_parameters():
            #     if 'guess_layer' in name:
            #         param_min = param.data.min().item()
            #         param_max = param.data.max().item()
            #         param_mean = param.data.mean().item()
            #         param_has_nan = torch.isnan(param.data).any().item()
            #         param_has_inf = torch.isinf(param.data).any().item()
            #         print(f"[{name}] Before step - Min: {param_min:.6f}, Max: {param_max:.6f}, "
            #               f"Mean: {param_mean:.6f}, NaN: {param_has_nan}, Inf: {param_has_inf}")
            #
            optimizer.step()

            # Debug: Print parameter values after optimizer step
            # for name, param in self.model.named_parameters():
            #     if 'guess_layer' in name:
            #         param_min = param.data.min().item()
            #         param_max = param.data.max().item()
            #         param_mean = param.data.mean().item()
            #         param_has_nan = torch.isnan(param.data).any().item()
            #         param_has_inf = torch.isinf(param.data).any().item()
            #         print(f"[{name}] After step - Min: {param_min:.6f}, Max: {param_max:.6f}, "
            #               f"Mean: {param_mean:.6f}, NaN: {param_has_nan}, Inf: {param_has_inf}")
            #
            # NaN detection and recovery for embedding weights
            for name, param in self.model.named_parameters():
                if 'guess_layer' in name and torch.isnan(param).any():
                    print(f"NaN detected in {name}, {param=}")

            total_loss += batch_loss.item()
            comp_errors_t.append(comp_errors.detach().cpu())
        if scheduler is not None:
            scheduler.step()
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t), dim=0)
        return total_loss, total_MAE_t

    # @torch.inference_mode()
    def __evaluate_batches_single_proc(self, dataloader, loss_fn, rank_device, y_ind=0):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                preds = self.dimer_model(batch)
                ref = batch.y[:, y_ind]
                preds = scatter(
                    preds,
                    batch.dimer_ind,
                    dim=0,
                    reduce="add",
                    dim_size=torch.tensor(
                        batch.total_charge_A.size(0), dtype=torch.long
                    ),
                )
                comp_errors = preds - ref
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, ref)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t), dim=0)
        return total_loss, total_MAE_t

    def __evaluate_batches_single_proc_elst_no_damping(
        self, dataloader, loss_fn, rank_device
    ):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                preds = self.dimer_model_elst(batch)
                ref = batch.y[:, 0]
                preds = scatter(
                    preds,
                    batch.dimer_ind,
                    dim=0,
                    reduce="add",
                    dim_size=torch.tensor(
                        batch.total_charge_A.size(0), dtype=torch.long
                    ),
                )
                comp_errors = preds - ref
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, ref)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t), dim=0)
        return total_loss, total_MAE_t

    ########################################################################
    # SINGLE-PROCESS TRAINING
    ########################################################################
    def single_proc_train(
        self,
        train_dataset,
        test_dataset,
        n_epochs,
        batch_size,
        lr,
        pin_memory,
        num_workers,
        skip_compile=False,
    ):
        # (1) Compile Model
        rank_device = self.device
        # self.model.to(rank_device)
        batch = self.example_input()
        batch.to(rank_device)
        self.model(batch)
        best_model = deepcopy(self.model)
        if not skip_compile:
            print("Compiling model")
            self.compile_model()

        # (2) Dataloaders
        # if self.ds_spec_type in [1, 5, 6]:
        collate_fn = ap2_fused_collate_update
        train_loader = APNet2_fused_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test_loader = APNet2_fused_DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        # (3) Optim/Scheduler
        # Separate parameter groups for embeddings vs other parameters
        embedding_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'guess_layer' in name:
                embedding_params.append(param)
            else:
                other_params.append(param)

        # Use lower learning rate and weight decay for embeddings
        param_groups = []
        if embedding_params:
            param_groups.append({
                'params': embedding_params,
                'lr': lr * 0.01,  # 100x lower learning rate for embeddings
                'weight_decay': 1e-6  # L2 regularization for embeddings
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': lr,
                'weight_decay': 1e-6  # L2 regularization for other parameters
            })

        optimizer = torch.optim.Adam(param_groups)
        scheduler = None
        # criterion = None  # defaults to MSE
        criterion = torch.nn.MSELoss()

        # (4) Set eval functions
        __evaluate_batch = self.__evaluate_batches_single_proc
        __train_batch = self.__train_batches_single_proc
        if self.dimer_eval_type == "elst_damping":
            y_ind = 0
            term = "Elst"
        elif self.dimer_eval_type == "induced_dipole":
            y_ind = 2
            term = "Indu"
            self.dimer_model.polarizability_table = (
                self.dimer_model.polarizability_table.to(self.device)
            )
        elif self.dimer_eval_type == "elst_damping__induced_dipole":
            assert isinstance(self.atom_model, AtomTypeParamNN), (
                "elst_damping__induced_dipole is only compatible with "
                "AtomTypeParamNN atom models presently."
            )
            y_ind = torch.tensor([0, 2])
            term = "Elst      Ind"
            self.dimer_model.polarizability_table = (
                self.dimer_model.polarizability_table.to(self.device)
            )
        else:
            raise ValueError(f"Unknown dimer_eval_type: {self.dimer_eval_type}")
        print(
            f"                                       {term}",
            flush=True,
        )

        # (5) Evaluate once pre-training
        if self.dimer_model_elst is not None:
            t0 = time.time()
            _, no_damping_MAE_t = self.__evaluate_batches_single_proc_elst_no_damping(
                train_loader, criterion, rank_device
            )
            _, no_damping_MAE_v = self.__evaluate_batches_single_proc_elst_no_damping(
                test_loader, criterion, rank_device
            )
            print(
                f" (No Damping)  ({time.time() - t0: < 7.2f}s)"
                f" MAE: {no_damping_MAE_t: > 7.3f}/{no_damping_MAE_v: < 7.3f}",
                flush=True,
            )
        lowest_test_loss = float("inf")
        t0 = time.time()
        # t_out = __evaluate_batch(train_loader, criterion, rank_device, y_ind=y_ind)
        # v_out = __evaluate_batch(test_loader, criterion, rank_device, y_ind=y_ind)
        # train_loss, total_MAE_t = t_out
        # test_loss, total_MAE_v = v_out
        # if isinstance(y_ind, torch.Tensor):
        #     mae_string = " ".join([f"{mae_t: > 7.3f}/{mae_v: < 7.3f}" for mae_t, mae_v in zip(total_MAE_t, total_MAE_v)])
        # else:
        #     mae_string = f"{total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f}"
        # print(
        #     f" (Pre-training)({time.time() - t0: < 7.2f}s)"
        #     f" MAE: {mae_string}",
        #     flush=True,
        # )
        # lowest_test_loss = test_loss
        cpu_model = self.model.to("cpu")
        for epoch in range(n_epochs):
            t1 = time.time()
            t_out = __train_batch(
                train_loader, criterion, optimizer, rank_device, scheduler, y_ind=y_ind
            )
            v_out = __evaluate_batch(test_loader, criterion, rank_device, y_ind=y_ind)
            train_loss, total_MAE_t = t_out
            test_loss, total_MAE_v = v_out

            # Track best model
            star_marker = " "
            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                star_marker = "*"
                cpu_model = self.model.to("cpu")
                best_model = deepcopy(cpu_model)
                if self.model_save_path:
                    torch.save(
                        {
                            "model_state_dict": cpu_model.state_dict(),
                            "config": {
                                "n_message": cpu_model.n_message,
                                "n_neuron": cpu_model.n_neuron,
                                "n_embed": cpu_model.n_embed,
                                "n_params": cpu_model.n_params,
                                "param_start_mean": cpu_model.param_start_mean,
                                "param_start_std": cpu_model.param_start_std,
                            },
                        },
                        self.model_save_path,
                    )
                self.model.to(rank_device)

            if isinstance(y_ind, torch.Tensor):
                mae_string = " ".join([f"{mae_t: > 7.3f}/{mae_v: < 7.3f}" for mae_t, mae_v in zip(total_MAE_t, total_MAE_v)])
            else:
                mae_string = f"{total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f}"
            print(
                f"  EPOCH: {epoch:4d} ({time.time() - t1:<7.2f}s)  MAE: "
                f"{mae_string} {star_marker}",
                flush=True,
            )
            if not self.device == "CPU":
                torch.cuda.empty_cache()
            if torch.any(total_MAE_t.isnan()) or torch.any(total_MAE_v.isnan()):
                print("NaN detected, stopping training")
                torch.save(
                    {
                        "model_state_dict": cpu_model.state_dict(),
                        "config": {
                            "n_message": cpu_model.n_message,
                            "n_neuron": cpu_model.n_neuron,
                            "n_embed": cpu_model.n_embed,
                            "n_params": cpu_model.n_params,
                            "param_start_mean": cpu_model.param_start_mean,
                            "param_start_std": cpu_model.param_start_std,
                        },
                    },
                    "nan_crash_model.pt",
                )
                break
        self.model = best_model
        self.model.to(rank_device)
        return

    def train(
        self,
        dataset=None,
        n_epochs=50,
        lr=5e-4,
        split_percent=0.9,
        model_path=None,
        shuffle=True,
        dataloader_num_workers=4,
        world_size=1,
        omp_num_threads_per_process=6,
        random_seed=42,
        skip_compile=False,
        lr_decay=None,
    ):
        print("NOTE: lr_decay is not implemented.")
        if dataset is not None:
            self.dataset = dataset
        elif dataset is not None:
            print("Overriding self.dataset with passed dataset!")
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided")
        np.random.seed(random_seed)
        self.model_save_path = model_path
        print(f"Saving training results to...\n{model_path}")
        if isinstance(self.dataset, list):
            train_dataset = self.dataset[0]
            if shuffle:
                order_indices = np.random.permutation(len(train_dataset))
            else:
                order_indices = [i for i in range(len(train_dataset))]
            train_dataset = train_dataset[order_indices]

            test_dataset = self.dataset[1]
            if shuffle:
                order_indices = np.random.permutation(len(test_dataset))
            else:
                order_indices = [i for i in range(len(test_dataset))]
            test_dataset = test_dataset[order_indices]
            batch_size = train_dataset.training_batch_size
        else:
            if shuffle:
                order_indices = np.random.permutation(len(self.dataset))
            else:
                order_indices = np.arange(len(self.dataset))
            train_indices = order_indices[: int(len(self.dataset) * split_percent)]
            test_indices = order_indices[int(len(self.dataset) * split_percent) :]
            train_dataset = self.dataset[train_indices]
            test_dataset = self.dataset[test_indices]
            batch_size = train_dataset.training_batch_size
        self.batch_size = batch_size
        print("~~ Training Dimer Param ~~", flush=True)
        print(
            f"    Training on {len(train_dataset)} samples,"
            f" Testing on {len(test_dataset)} samples"
        )
        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {self.model.n_message=}", flush=True)
        print(f"  {self.model.n_neuron=}", flush=True)
        print(f"  {self.model.n_embed=}", flush=True)
        print(f"  {self.model.param_start_mean=}", flush=True)
        print(f"  {self.model.param_start_std=}", flush=True)
        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {lr=}\n", flush=True)
        print(f"  {batch_size=}", flush=True)

        if self.device.type == "cuda":
            pin_memory = False
        else:
            pin_memory = False

        self.shuffle = shuffle

        if world_size > 1:
            print("Running multi-process training", flush=True)
            raise NotImplementedError(
                "Multi-process training is not implemented for MTP-MTP models."
            )
        else:
            print("Running single-process training", flush=True)
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads_per_process)
            self.single_proc_train(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                pin_memory=pin_memory,
                num_workers=dataloader_num_workers,
                skip_compile=skip_compile,
            )
        return


### Atom Type Model Wrapper ####
class AtomTypeParamModel:
    def __init__(
        self,
        dataset=None,
        atom_model=None,
        atom_model_type="AtomMPNN",
        pre_trained_model_path=None,
        atom_model_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        param_start_mean=1.7,
        param_start_std=0.01,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_root="data_dir",
        ds_max_size=None,
        ds_random_seed=42,
        ds_batch_size=16,
        ds_testing=False,
        ds_force_reprocess=False,
        ds_in_memory=True,
        model_save_path=None,
        monomer_eval_type="hirshfeld_volume_ratio__valence_width",
    ):
        """
        If pre_trained_model_path is provided, the model will be loaded from
        the path and all other parameters will be ignored except for dataset.

        use_GPU will check for a GPU and use it if available unless set to false.
        """
        if torch.cuda.is_available() and use_GPU is not False:
            device = torch.device("cuda:0")
            print("running on the GPU")
        else:
            device = torch.device("cpu")
            print("running on the CPU")
        self.ds_spec_type = ds_spec_type
        # TODO UPDATE TO AP3
        if atom_model_type == "AtomMPNN":
            self.atom_model = AtomMPNN()
            am_type = AtomMPNN
        elif atom_model_type == "AtomHirshfeldMPNN":
            self.atom_model = AtomHirshfeldMPNN()
            am_type = AtomHirshfeldMPNN
        else:
            raise ValueError(f"Unknown atom_model_type: {atom_model_type}")

        self.n_params = 1
        if monomer_eval_type in ["hirshfeld_volume_ratio__valence_width"]:
            self.n_params = 2

        if atom_model_pre_trained_path:
            print(
                f"Loading pre-trained AtomMPNN model from {atom_model_pre_trained_path}"
            )
            checkpoint = torch.load(
                atom_model_pre_trained_path, map_location=device, weights_only=False
            )
            self.atom_model = am_type(
                n_message=checkpoint["config"]["n_message"],
                n_rbf=checkpoint["config"]["n_rbf"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                r_cut=checkpoint["config"]["r_cut"],
            )
            # model_state_dict = checkpoint["model_state_dict"]
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.atom_model.load_state_dict(model_state_dict)
        elif atom_model:
            print("Using provided AtomMPNN model:", atom_model)
            self.atom_model = atom_model
        else:
            print(
                """No atom model provided.
    Assuming atomic multipoles and embeddings are
    pre-computed and passed as input to the model.
"""
            )
        if pre_trained_model_path:
            print(f"Loading pre-trained MTP-MTP model from {pre_trained_model_path}")
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = AtomTypeParamNN(
                atom_model=self.atom_model,
                n_message=checkpoint["config"]["n_message"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                param_start_mean=checkpoint["config"]["param_start_mean"],
                param_start_std=checkpoint["config"]["param_start_std"],
                n_params=checkpoint["config"].get("n_params", 1),
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = AtomTypeParamNN(
                atom_model=self.atom_model,
                n_message=n_message,
                n_neuron=n_neuron,
                n_embed=n_embed,
                param_start_mean=param_start_mean,
                param_start_std=param_start_std,
                n_params=self.n_params,
            )
        self.n_params = self.n_params
        self.monomer_eval_type = monomer_eval_type
        self.device = device
        self.dataset = dataset
        mp.set_sharing_strategy("file_system")
        if not ignore_database_null and self.dataset is None:
            self.dataset = atomic_hirshfeld_module_dataset(
                root=ds_root,
                testing=ds_testing,
                spec_type=ds_spec_type,
                max_size=ds_max_size,
                force_reprocess=ds_force_reprocess,
                in_memory=ds_in_memory,
                batch_size=ds_batch_size,
            )
        # print(f"{self.dataset = }")
        self.rank = None
        self.world_size = None
        self.model_save_path = model_save_path
        self.train_shuffle = None
        # torch.jit.enable_onednn_fusion(True)
        return

    def set_pretrained_model(self, model_path):
        checkpoint = torch.load(model_path)
        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        return

    def compile_model(self):
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        self.model = torch.compile(self.model, dynamic=True)
        return

    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        # torch.manual_seed(42)

    def cleanup(self):
        dist.destroy_process_group()

    def evaluate_model_collate_train(self, data_loader, optimizer=None, loss_fn=None):
        charge_errors_t, dipole_errors_t, qpole_errors_t = [], [], []
        total_loss = 0.0
        self.model.train()
        for batch in data_loader:
            batch_loss = 0.0
            batch = batch.to(self.device)
            optimizer.zero_grad()
            charge, dipole, qpole, _ = self.model(batch)

            # Errors
            q_error = charge - batch.charges
            d_error = dipole - batch.dipoles
            qp_error = qpole - batch.quadrupoles
            if loss_fn is None:
                # perform mean squared error
                charge_loss = torch.mean(torch.square(q_error))
                dipole_loss = torch.mean(torch.square(d_error))
                qpole_loss = torch.mean(torch.square(qp_error))
            else:
                # perform custom loss function, or pytorch criterion loss_fn
                charge_loss = loss_fn(charge, batch.charges)
                dipole_loss = torch.mean(loss_fn(dipole, batch.dipoles))
                qpole_loss = torch.mean(loss_fn(qpole, batch.quadrupoles))

            batch_loss = charge_loss + dipole_loss + qpole_loss
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.detach().item()

            charge_errors_t.append(q_error.detach())
            dipole_errors_t.extend(d_error.detach())
            qpole_errors_t.extend(qp_error.detach())
        charge_errors_t = torch.cat(charge_errors_t)
        dipole_errors_t = torch.cat(dipole_errors_t)
        qpole_errors_t = torch.cat(qpole_errors_t)
        return total_loss, charge_errors_t, dipole_errors_t, qpole_errors_t

    def evaluate_model_collate_eval(self, data_loader, loss_fn=None):
        hfvr_errors_t, vw_errors_t = (
            [],
            [],
        )
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch_loss = 0.0
                params = self.model(batch)[-1]
                hirshfeld_volume_ratios = params[:, 0]
                valence_widths = params[:, 1]

                # Errors
                hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
                vw_error = valence_widths - batch.valence_widths
                if loss_fn is None:
                    # perform mean squared error
                    hfvr_loss = torch.mean(torch.square(hfvr_error))
                    vw_loss = torch.mean(torch.square(vw_error))
                else:
                    # perform custom loss function, or pytorch criterion loss_fn
                    hfvr_loss = torch.mean(
                        loss_fn(hirshfeld_volume_ratios, batch.volume_ratios)
                    )
                    vw_loss = torch.mean(loss_fn(valence_widths, batch.valence_widths))

                batch_loss = hfvr_loss + vw_loss
                total_loss += batch_loss.detach()

            hfvr_errors_t.extend(hfvr_error.detach())
            vw_errors_t.extend(vw_error.detach())
        hfvr_errors_t = torch.cat(hfvr_errors_t)
        vw_errors_t = torch.cat(vw_errors_t)
        return (
            total_loss,
            hfvr_errors_t,
            vw_errors_t,
        )

    def pretrain_statistics(self, train_loader, test_loader, criterion):
        t1 = time.time()
        with torch.no_grad():
            (
                _,
                hfvr_errors_t,
                vw_errors_t,
            ) = self.evaluate_model_collate_eval(
                train_loader,  # loss_fn=criterion
            )
            hfvr_MAE_t = np.mean(np.abs(hfvr_errors_t))
            vw_MAE_t = np.mean(np.abs(vw_errors_t))

            (
                hfvr_errors_t,
                vw_errors_t,
            ) = [], [], [], []
            (
                test_loss,
                hfvr_errors_v,
                vw_errors_v,
            ) = self.evaluate_model_collate_eval(
                test_loader,  # loss_fn=criterion
            )
            hfvr_MAE_v = np.mean(np.abs(hfvr_errors_v))
            vw_MAE_v = np.mean(np.abs(vw_errors_v))
            (
                hfvr_errors_v,
                vw_errors_v,
            ) = [], []
            dt = time.time() - t1
            print(
                f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {vw_MAE_t:>7.4f}/{vw_MAE_v:<7.4f}",
                flush=True,
            )
        return test_loss

    def train_batches_single_proc(
        self, rank, dataloader, criterion, optimizer, rank_device
    ):
        self.model.train()
        total_hfvr_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_vw_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        for batch in dataloader:
            batch = batch.to(rank_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            params = self.model(batch)[-1]
            hirshfeld_volume_ratios = params[:, 0]
            valence_widths = params[:, 1]

            hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
            vw_error = valence_widths - batch.valence_widths

            hfvr_loss = (hfvr_error**2).mean()
            vw_loss = (vw_error**2).mean()

            loss = hfvr_loss + vw_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += hfvr_error.numel()

            total_hfvr_error += hfvr_error.detach().abs().sum()
            total_vw_error += vw_error.detach().abs().sum()

        final_count = total_count.item()

        # Calculating MAEs
        hfvr_mae = total_hfvr_error.item() / final_count
        vw_mae = total_vw_error.item() / final_count
        return total_loss, hfvr_mae, vw_mae

    def train_batches(self, rank, dataloader, criterion, optimizer, rank_device):
        self.model.train()
        total_hfvr_error = 0
        total_vw_error = 0
        total_loss = 0
        count = 0

        for batch in dataloader:
            batch = batch.to(rank_device)
            optimizer.zero_grad()
            params = self.model(batch)[-1]
            hirshfeld_volume_ratios = params[:, 0]
            valence_widths = params[:, 1]

            hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
            vw_error = valence_widths - batch.valence_widths

            hfvr_loss = torch.mean(torch.square(hfvr_error))
            vw_loss = torch.mean(torch.square(vw_error))

            loss = hfvr_loss + vw_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += hfvr_error.numel()

            total_hfvr_error += torch.sum(torch.abs(hfvr_error)).item()
            total_vw_error += torch.sum(torch.abs(vw_error)).item()

        # Converting to tensors for all-reduce
        total_hfvr_error = torch.tensor(
            total_hfvr_error, dtype=torch.float32, device=rank_device
        )
        total_vw_error = torch.tensor(
            total_vw_error, dtype=torch.float32, device=rank_device
        )
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_hfvr_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_vw_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        # Calculating MAEs
        hfvr_mae = total_hfvr_error.item() / count.item()
        vw_mae = total_vw_error.item() / count.item()

        return total_loss, hfvr_mae, vw_mae

    def evaluate_batches_single_proc(self, rank, dataloader, criterion, rank_device):
        self.model.eval()
        total_hfvr_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_vw_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(rank_device, non_blocking=True)
                params = self.model(batch)[-1]
                hirshfeld_volume_ratios = params[:, 0]
                valence_widths = params[:, 1]

                hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
                vw_error = valence_widths - batch.valence_widths

                hfvr_loss = (hfvr_error**2).mean()
                vw_loss = (vw_error**2).mean()

                loss = hfvr_loss + vw_loss
                total_loss += loss.item()
                total_count += hfvr_error.numel()

                total_hfvr_error += hfvr_error.abs().sum()
                total_vw_error += vw_error.abs().sum()

        final_count = total_count.item()

        # Calculating MAEs
        hfvr_mae = total_hfvr_error.item() / final_count
        vw_mae = total_vw_error.item() / final_count
        return total_loss, hfvr_mae, vw_mae

    def evaluate_batches(self, rank, dataloader, criterion, rank_device):
        self.model.eval()
        total_hfvr_error = 0
        total_vw_error = 0
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(rank_device)
                params = self.model(batch)[-1]
                hirshfeld_volume_ratios = params[:, 0]
                valence_widths = params[:, 1]

                hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
                vw_error = valence_widths - batch.valence_widths

                hfvr_loss = (hfvr_error**2).mean()
                vw_loss = (vw_error**2).mean()
                hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
                vw_error = valence_widths - batch.valence_widths

                total_hfvr_error += torch.sum(torch.abs(hfvr_error)).item()
                total_vw_error += torch.sum(torch.abs(vw_error)).item()

                hfvr_loss = torch.mean(torch.square(hfvr_error))
                vw_loss = torch.mean(torch.square(vw_error))

                total_loss += hfvr_loss + vw_loss
                count += hfvr_error.numel()

        # Converting to tensors for all-reduce
        total_hfvr_error = torch.tensor(
            total_hfvr_error, dtype=torch.float32, device=rank_device
        )
        total_vw_error = torch.tensor(
            total_vw_error, dtype=torch.float32, device=rank_device
        )
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_hfvr_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_vw_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        total_loss = torch.tensor(total_loss.item(), device=rank_device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        # Calculating MAEs
        hfvr_mae = total_hfvr_error.item() / count.item()
        vw_mae = total_vw_error.item() / count.item()
        return total_loss, hfvr_mae, vw_mae

    def ddp_train(
        self,
        rank,
        world_size,
        train_dataset,
        test_dataset,
        n_epochs,
        batch_size,
        lr,
        pin_memory,
        num_workers,
    ):
        # print(f"{self.device.type = }")
        if self.device.type == "cpu":
            # rank = "cpu"
            rank_device = "cpu"
        else:
            rank_device = rank
        if world_size > 1:
            self.setup(rank, world_size)

        self.model.to(rank_device)
        if world_size > 1 and rank_device == "cpu":
            torch._dynamo.config.dynamic_shapes = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, dynamic=True)
            self.model = DDP(
                self.model,
            )

        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank
            )
            if world_size > 1
            else None
        )
        test_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if world_size > 1
            else None
        )

        train_loader = AtomicDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            collate_fn=atomic_collate_update_prebatched,
            # collate_fn=atomic_hirshfeld_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=atomic_collate_update_prebatched,
            # collate_fn=atomic_hirshfeld_collate_update,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        test_loss = self.pretrain_statistics(train_loader, test_loader, criterion)

        lowest_test_loss = test_loss

        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            train_loss, charge_MAE_t, dipole_MAE_t, qpole_MAE_t, hfvr_MAE_t = (
                self.train_batches(
                    rank, train_loader, criterion, optimizer, rank_device
                )
            )
            test_loss, charge_MAE_v, dipole_MAE_v, qpole_MAE_v, hfvr_MAE_v = (
                self.evaluate_batches(rank, test_loader, criterion, rank_device)
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        cpu_model = unwrap_model(self.model).to("cpu")
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "param_start_mean": cpu_model.param_start_mean,
                                    "param_start_std": cpu_model.param_start_std,
                                    "n_params": cpu_model.n_params,
                                },
                            },
                            self.model_save_path,
                        )
                        self.model.to(self.device)
                else:
                    test_lowered = " "
                dt = time.time() - t1
                test_loss = 0.0
                # if (world_size==1 or rank == 0):
                print(
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {test_lowered}",
                    flush=True,
                )
        if world_size > 1:
            self.cleanup()
        return

    def single_proc_train(
        self,
        rank,
        world_size,
        train_dataset,
        test_dataset,
        n_epochs,
        batch_size,
        lr,
        pin_memory,
        num_workers,
        skip_compile=False,
    ):
        if self.device.type == "cpu":
            rank_device = "cpu"
        else:
            rank_device = rank

        self.model.to(rank_device)
        if not skip_compile:
            self.compile_model()

        train_loader = AtomicDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=self.train_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=atomic_collate_update_prebatched,
            # collate_fn=atomic_hirshfeld_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=atomic_collate_update_prebatched,
            # collate_fn=atomic_hirshfeld_collate_update,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        lowest_test_loss = torch.tensor(float("inf"))
        t1 = time.time()
        train_loss, hfvr_MAE_t, vw_MAE_t = self.evaluate_batches_single_proc(
            rank, train_loader, criterion, rank_device
        )
        test_loss, hfvr_MAE_v, vw_MAE_v = self.evaluate_batches_single_proc(
            rank, test_loader, criterion, rank_device
        )
        dt = time.time() - t1
        print(
            f"                                Hirshfeld Vol Ratio   Valence Width"
        )
        print(
            f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {vw_MAE_t:>7.4f}/{vw_MAE_v:<7.4f}",
            flush=True,
        )
        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            (
                train_loss,
                hfvr_MAE_t,
                vw_MAE_t,
            ) = self.train_batches_single_proc(
                rank, train_loader, criterion, optimizer, rank_device
            )
            test_loss, hfvr_MAE_v, vw_MAE_v = self.evaluate_batches_single_proc(
                rank, test_loader, criterion, rank_device
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        # cpu_model = self.model.to("cpu")
                        cpu_model = unwrap_model(self.model).to("cpu")
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "param_start_mean": cpu_model.param_start_mean,
                                    "param_start_std": cpu_model.param_start_std,
                                    "n_params": cpu_model.n_params,
                                },
                            },
                            self.model_save_path,
                        )
                        self.model.to(self.device)
                else:
                    test_lowered = " "
                dt = time.time() - t1
                test_loss = 0.0
                print(
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {vw_MAE_t:>7.4f}/{vw_MAE_v:<7.4f} {test_lowered}",
                    flush=True,
                )

            # n = gc.collect()
            # print("    Garbage collector: collected %d objects." % n)
            # if rank_device != "cpu":
            #     torch.cuda.empty_cache()
        if world_size > 1:
            self.cleanup()
        return

    def train(
        self,
        dataset=None,
        n_epochs=500,
        batch_size=16,
        lr=5e-4,
        split_percent=0.9,
        model_path=None,
        skip_compile=True,
        shuffle=True,
        dataloader_num_workers=0,
        world_size=1,  # Default to 1 for single-core operation
        omp_num_threads_per_process=None,
        random_seed=42,
    ):
        self.model_save_path = model_path
        if self.model_save_path is not None:
            print(f"Saving model to {self.model_save_path}")
        if self.dataset is None and dataset is not None:
            self.dataset = dataset
        elif dataset is not None:
            print("Overriding self.dataset with passed dataset!")
            self.dataset = dataset
        if self.dataset is None:
            raise ValueError("No dataset provided")
        self.train_shuffle = shuffle

        np.random.seed(42)
        torch.manual_seed(42)
        random_indices = np.random.permutation(len(self.dataset))
        train_indices = random_indices[: int(len(self.dataset) * split_percent)]
        test_indices = random_indices[int(len(self.dataset) * split_percent) :]
        if random_seed:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            train_indices = np.random.permutation(train_indices)
        train_dataset = self.dataset[train_indices]
        test_dataset = self.dataset[test_indices]

        print("~~ Training Atom Model ~~", flush=True)
        print(
            f"    Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples",
            flush=True,
        )
        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {self.model.n_message=}", flush=True)
        print(f"  {self.model.n_neuron=}", flush=True)
        print(f"  {self.model.n_embed=}", flush=True)
        print(f"  {self.model.n_params=}", flush=True)
        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {batch_size=}", flush=True)
        print(f"  {lr=}\n", flush=True)

        # pin_memory = torch.cuda.is_available()
        pin_memory = True

        if skip_compile:
            torch.jit.enable_onednn_fusion(True)
            torch.autograd.set_detect_anomaly(False)

        if world_size > 1:
            # os.environ["OMP_NUM_THREADS"] = str(dataloader_num_workers + 1)
            print("Running multi-process training", flush=True)
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads_per_process)
            mp.spawn(
                self.ddp_train,
                args=(
                    world_size,
                    train_dataset,
                    test_dataset,
                    n_epochs,
                    batch_size,
                    lr,
                    pin_memory,
                    dataloader_num_workers,
                ),
                nprocs=world_size,
                join=True,
            )
        else:
            # Run single-process training directly
            print("Running single-process training", flush=True)
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads_per_process)
            self.single_proc_train(
                rank=0,
                world_size=world_size,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                pin_memory=pin_memory,
                num_workers=dataloader_num_workers,
                skip_compile=skip_compile,
            )

        return

    @torch.inference_mode()
    def predict_multipoles_batch(self, batch, isolate_predictions=True):
        batch.to(self.device)
        self.model.to(self.device)
        qA, muA, thA, hfvrA, vwA, hlistA = self.model_predict(batch)
        batch = batch.cpu()
        qA = qA.detach().detach().cpu()
        muA = muA.detach().detach().cpu()
        thA = thA.detach().detach().cpu()
        hfvrA = hfvrA.detach().detach().cpu()
        vwA = vwA.detach().detach().cpu()
        hlistA = hlistA.detach().cpu()
        if isolate_predictions:
            return isolate_atomic_property_predictions(
                batch, (qA, muA, thA, hfvrA, vwA, hlistA)
            )
        else:
            return qA, muA, thA, hfvrA, vwA, hlistA

    @torch.inference_mode()
    def predict_multipoles_dataset(
        self,
        batch_size=16,
        dataloader_num_workers=0,
        world_size=1,  # Default to 1 for single-process operation
        # omp_num_threads_per_process=None,
    ):
        output = []
        data = AtomicDataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        if world_size > 1:
            raise NotImplementedError(
                "Multi-process prediction not implemented yet due to needing to determine how to handle the output data merging."
            )
            # output = mp.spawn(
            #     self.predict_multipoles_dataset_process,
            #     args=(data, batch_size, dataloader_num_workers),
            #     nprocs=world_size,
            #     join=True,
            # )
        else:
            for batch in data:
                (
                    charges,
                    dipoles,
                    qpoles,
                    hirshfeld_volume_ratios,
                    valence_widths,
                    hlists,
                ) = self.model_predict(batch)
                # need to use batch.molecule_ind to reassemble the output
                mol_charges = [[] for i in range(batch_size)]
                mol_dipoles = [[] for i in range(batch_size)]
                mol_qpoles = [[] for i in range(batch_size)]
                mol_hfvr = [[] for i in range(batch_size)]
                mol_vw = [[] for i in range(batch_size)]
                for n, i in enumerate(batch.molecule_ind):
                    mol_charges[i].append(charges[n])
                    mol_dipoles[i].append(dipoles[n])
                    mol_qpoles[i].append(qpoles[n])
                    mol_hfvr[i].append(hirshfeld_volume_ratios[n])
                    mol_vw[i].append(valence_widths[n])
                output.append(
                    (mol_charges, mol_dipoles, mol_qpoles, mol_hfvr, mol_vw, hlists)
                )
        return output

    @torch.inference_mode()
    def predict_qcel_mols(self, mols, batch_size=2):
        output = []
        mol_data = []
        cnt = 0
        for mol in mols:
            data = qcel_mon_to_pyg_data(mol)
            mol_data.append(data)
            cnt += 1
            if len(mol_data) == batch_size or cnt == len(mols):
                batch = atomic_collate_update_no_target(mol_data)
                with torch.no_grad():
                    charge, dipole, qpole, hfvr, vw, hlist = self.model(batch)
                    # Isolate atomic properties by molecule
                    (
                        mol_charges,
                        mol_dipoles,
                        mol_qpoles,
                        mol_hfvrs,
                        mol_vws,
                        mol_hlists,
                    ) = isolate_atomic_property_predictions(
                        batch, (charge, dipole, qpole, hfvr, vw, hlist)
                    )
                    output.extend(
                        list(
                            zip(
                                mol_charges,
                                mol_dipoles,
                                mol_qpoles,
                                mol_hfvrs,
                                mol_vws,
                                mol_hlists,
                            )
                        )
                    )
                mol_data = []
        return output

    @torch.inference_mode()
    def model_predict(self, data):
        charge, dipole, qpole, hirshfeld_volume_ratios, valence_widths, hlist = (
            self.model(data)
        )
        return charge, dipole, qpole, hirshfeld_volume_ratios, valence_widths, hlist
