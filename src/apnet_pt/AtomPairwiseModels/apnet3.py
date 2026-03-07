import torch
import torch.nn as nn
from apnet_pt.util import scatter_sum_compile
from torch_geometric.data import Data
import numpy as np
import warnings
import time
from ..AtomModels.ap2_hirshfeld_atom_model import (
    AtomHirshfeldMPNN,
    isolate_atomic_property_predictions,
)
from .. import atomic_datasets
from .. import pairwise_datasets
from ..pairwise_datasets import (
    apnet3_module_dataset,
    APNet2_DataLoader,
    apnet3_collate_update,
    apnet3_collate_update_prebatched,
    pairwise_edges,
    pairwise_edges_im,
    qcel_dimer_to_pyg_data,
    free_atom_polarizabilities,
)
from .. import constants
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import qcelemental as qcel
# from ..multipole import T_cart_torch

hartree2kcal = qcel.constants.conversion_factor("hartree", "kcal/mol")


def inverse_time_decay(step, initial_lr, decay_steps, decay_rate, staircase=True):
    p = step / decay_steps
    if staircase:
        p = np.floor(p)
    return initial_lr / (1 + decay_rate * p)


class InverseTimeDecayLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, initial_lr, decay_steps, decay_rate):
        super().__init__(
            optimizer,
            lr_lambda=lambda step: inverse_time_decay(
                step, initial_lr, decay_steps, decay_rate
            ),
        )


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

max_Z = 118


def lr_lambda(epoch, decay_factor, initial_lr, min_lr=4e-5):
    lr = initial_lr * (decay_factor**epoch)
    return max(lr, min_lr) / initial_lr


class AsymptoticDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_coefficient, last_epoch=-1):
        self.decay_coefficient = decay_coefficient
        super(AsymptoticDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr / (1 + self.last_epoch / self.decay_coefficient)
            for base_lr in self.base_lrs
        ]


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff in PyTorch.
    """

    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.exponent = exponent

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs):
        # Envelope function divided by r
        env_val = (
            1 / inputs
            + self.a * inputs ** (self.p - 1)
            + self.b * inputs**self.p
            + self.c * inputs ** (self.p + 1)
        )
        env_val = torch.where(inputs < 1, env_val, torch.zeros_like(inputs))
        return env_val


class DistanceLayer(nn.Module):
    """
    Projects a distance 0 < r < r_cut into an orthogonal basis of Bessel functions in PyTorch.
    """

    def __init__(self, num_radial=8, r_cut=5.0, envelope_exponent=5):
        super(DistanceLayer, self).__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1.0 / r_cut
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        freq_init = torch.FloatTensor(
            np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
        )
        self.frequencies = nn.Parameter(freq_init, requires_grad=True)

    def forward(self, inputs):
        # scale to range [0, 1]
        d_scaled = inputs * self.inv_cutoff
        d_scaled = d_scaled.unsqueeze(-1)
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


# Maybe MessagePassing inheritance is not necessary and slowing down...
# class APNet2_MPNN(MessagePassing):
class APNet3_MPNN(nn.Module):
    def __init__(
        self,
        # atom_model,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut_im=8.0,
        r_cut=5.0,
    ):
        # super().__init__(aggr="add")
        super().__init__()
        # self.atom_model = atom_model

        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut_im = r_cut_im
        self.r_cut = r_cut

        layer_nodes_hidden = [
            # input_layer_size,
            n_neuron * 2,
            n_neuron,
            n_neuron // 2,
            n_embed,
        ]
        layer_nodes_readout = [
            # n_embed,
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
        ]  # None represents a linear activation

        # embed interatomic distances into large orthogonal basis
        self.distance_layer_im = DistanceLayer(n_rbf, self.r_cut_im)
        self.distance_layer = DistanceLayer(n_rbf, self.r_cut)

        # embed atom types
        self.embed_layer = nn.Embedding(max_Z + 1, n_embed)

        # readout layers for predicting final interaction energies
        self.readout_layer_elst = self._make_layers(
            layer_nodes_readout, layer_activations
        )
        self.readout_layer_exch = self._make_layers(
            layer_nodes_readout, layer_activations
        )
        self.readout_layer_exch_quotient = self._make_layers(
            layer_nodes_readout, layer_activations
        )
        self.readout_layer_indu = self._make_layers(
            layer_nodes_readout, layer_activations
        )
        self.readout_layer_disp = self._make_layers(
            layer_nodes_readout, layer_activations
        )

        # update layers for hidden states
        self.update_layers = nn.ModuleList()
        self.directional_layers = nn.ModuleList()
        for i in range(n_message):
            self.update_layers.append(
                self._make_layers(layer_nodes_hidden, layer_activations)
            )
            self.directional_layers.append(
                self._make_layers(layer_nodes_hidden, layer_activations)
            )

    def _make_layers(self, layer_nodes, activations):
        layers = []
        # Start with a LazyLinear so we don't have to fix input dim
        layers.append(nn.LazyLinear(layer_nodes[0]))
        layers.append(activations[0])
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            if activations[i + 1] is not None:
                layers.append(activations[i + 1])
        return nn.Sequential(*layers)

    def mtp_elst(
        self,
        qA,
        muA,
        quadA,
        qB,
        muB,
        quadB,
        e_ABsr_source,
        e_ABsr_target,
        dR_ang,
        dR_xyz_ang,
    ):
        dR = dR_ang / constants.au2ang
        dR_xyz = dR_xyz_ang / constants.au2ang
        oodR = 1.0 / dR

        # Identity for 3D
        delta = torch.eye(3, device=qA.device)

        # Extracting tensor elements
        qA_source = qA.squeeze(-1).index_select(0, e_ABsr_source)
        qB_source = qB.squeeze(-1).index_select(0, e_ABsr_target)

        muA_source = muA.index_select(0, e_ABsr_source)
        muB_source = muB.index_select(0, e_ABsr_target)

        quadA_source = (3.0 / 2.0) * quadA.index_select(0, e_ABsr_source)
        quadB_source = (3.0 / 2.0) * quadB.index_select(0, e_ABsr_target)

        E_qq = qA_source * qB_source * oodR

        T1 = -1.0 * dR_xyz * (oodR**3).unsqueeze(1)
        qu = (qA_source.unsqueeze(1) * muB_source) - (
            qB_source.unsqueeze(1) * muA_source
        )
        E_qu = (T1 * qu).sum(dim=1)

        # T2 = 3(dR_xyz x dR_xyz) - dR * delta
        T2 = 3 * torch.einsum("ij,ik->ijk", dR_xyz, dR_xyz) - torch.einsum(
            "i,jk->ijk", dR, delta
        )
        T2 = T2 * (oodR**5).unsqueeze(1).unsqueeze(2)

        # E_uu should be close to zero
        E_uu = -1.0 * torch.einsum("ij,ik,ijk->i", muA_source, muB_source, T2)

        qA_quadB_source = qA_source.unsqueeze(1).unsqueeze(2) * quadB_source
        qB_quadA_source = qB_source.unsqueeze(1).unsqueeze(2) * quadA_source
        E_qQ = (T2 * (qA_quadB_source + qB_quadA_source)).sum(dim=(1, 2)) / 3.0

        E_elst = 627.509 * (E_qq + E_qu + E_qQ + E_uu)

        return E_elst

    def valence_width_exch(self, e_source, e_target, vwA, vwB, r_ij):
        # TODO: Implement valence width exchange;
        # vwA and vwB are the valence widths of monomer A and B,
        # respectively. r_ij is the interatomic distance between atoms i
        # and j. Use distance matrix to compute r_ij.
        vwA = torch.where(vwA > 0.1, vwA, 0.1)
        vwB = torch.where(vwB > 0.1, vwB, 0.1)
        sigma_A_source = vwA.index_select(0, e_source)
        sigma_B_target = vwB.index_select(0, e_target)
        # print(e_source)
        # print(e_target)
        # print(f"{sigma_A_source = }")
        # print(f"{sigma_B_target = }")
        # sigma = torch.einsum("i,j->ij", vwA, vwB)
        # https://vergil.chemistry.gatech.edu/static/pdfs/schriber_2021_184110.pdf
        sigma_ij = sigma_A_source * sigma_B_target
        # print(f"{sigma_ij = }")
        # print(f"{sigma_A_source.size() = }, {sigma_B_target.size() = }, {sigma_ij.size() = }")
        # dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
        B_ij = (1.0 / sigma_ij).squeeze()
        # print(f"{B_ij = }")
        # print(f"{r_ij = }")
        S_ij = (
            (1.0 / 3.0 * (B_ij * r_ij) ** 2 + B_ij * r_ij + 1.0)
            * torch.exp(-B_ij * r_ij)
            * hartree2kcal
        )
        # print(f"{S_ij.size() = }")
        # print(f"{S_ij = }")
        return S_ij

    def induced_dipole_indu(
        self,
        RA,
        RB,
        qA,
        muA,
        quadA,
        qB,
        muB,
        quadB,
        hfvrA,
        hfvrB,
        alpha_0_A,
        alpha_0_B,
        S_ij,
        e_AB_source_all,
        e_AB_target_all,
        omega=0.7,
        smearing=0.39,
    ):
        # TODO: Implement induced dipole induction
        # https://github.com/jeffschriber/cliff/blob/660871c3949fcea5d907fe8cbe54352fd071e841/cliff/components/induction_calc.py#L122
        print(f"{RA.size() = }, {RB.size() = }")
        print(f"{e_AB_source_all = }")
        print(f"{e_AB_target_all = }")
        # # intra+intermolecular distances
        dR, dR_xyz = self.get_distances(RA, RB, e_AB_source_all, e_AB_target_all)
        # print(f"{dR.size() = }")
        # print(f"{dR = }")
        # T0, T1, T2, T3, T4 = T_cart_torch(RA, RB)
        # print(f"{T0 = }")
        # print(f"{T1 = }")
        # print(f"{T2 = }")
        # print(f"{T3 = }")
        # print(f"{T4 = }")
        alpha_0s = torch.cat([alpha_0_A, alpha_0_B], dim=0)
        print(f"{alpha_0_A = }")
        print(f"{alpha_0_B = }")
        alphas = torch.outer(alpha_0s, alpha_0s)
        # Need to mask out i interacting with itself
        # alphas = alphas.index_select(0, e_AB_source_all)
        print(f"{alphas = }")
        print(f"{alphas.size() = }")
        print(f"{dR.size() = }")
        # u = dR_xyz / ((alpha_0_A * alpha_0_B) ** (1.0 / 6.0))
        # print(f"{e_AA_source = }")
        # print(f"{e_AA_target = }")
        # u = dR_ang / ((alpha_0_A * alpha_0_B) ** (1.0 / 6.0))
        # print(f"{u = }")
        # f_Thole = 3.0 * smearing / (4.0 * torch.pi) * torch.exp(-smearing * u**3)
        # print(f"{f_Thole = }")
        return

    def get_messages(self, h0, h, rbf, e_source, e_target):
        nedge = e_source.numel()
        if nedge == 0:
            # No intramolecular edges
            return torch.zeros(
                0, self.n_embed * 4 * self.n_rbf + self.n_embed * 4 + self.n_rbf
            )

        h0_source = h0.index_select(0, e_source)
        h0_target = h0.index_select(0, e_target)
        h_source = h.index_select(0, e_source)
        h_target = h.index_select(0, e_target)

        # [edges x 4 * n_embed]
        h_all = torch.cat([h0_source, h0_target, h_source, h_target], dim=-1)

        # print(nedge)
        # print(h_all.size())
        # [edges, 4 * n_embed, n_rbf]
        h_all_dot = torch.einsum("ez,er->ezr", h_all, rbf).view(nedge, -1)
        # h_all_dot = h_all_dot.view(nedge, -1)

        # [edges,  n_embed * 4 * n_rbf + n_embed * 4 + n_rbf]
        m_ij = torch.cat([h_all, h_all_dot, rbf], dim=-1)
        return m_ij

    def get_pair(self, hA, hB, qA, qB, rbf, e_source, e_target):
        hA_source = hA.index_select(0, e_source)
        hB_target = hB.index_select(0, e_target)

        qA_source = qA.index_select(0, e_source)
        qB_target = qB.index_select(0, e_target)
        # print(f"{hA_source.size() = }, {hB_target.size() = }, {qA_source.size() = }, {qB_target.size() = }, {rbf.size() = }")
        return torch.cat([hA_source, hB_target, qA_source, qB_target, rbf], dim=-1)

    def get_distances(self, RA, RB, e_source, e_target):
        RA_source = RA.index_select(0, e_source)
        RB_target = RB.index_select(0, e_target)
        dR_xyz = RB_target - RA_source

        # Compute distances with safe operation for square root
        # dR = torch.sqrt(nn.functional.relu(torch.sum(dR_xyz**2, dim=-1)))
        dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
        return dR, dR_xyz

    def forward(
        self,
        ZA,
        RA,
        ZB,
        RB,
        # short range, intermolecular edges
        e_ABsr_source,
        e_ABsr_target,
        dimer_ind,
        # long range, intermolecular edges
        e_ABlr_source,
        e_ABlr_target,
        dimer_ind_lr,
        # intramonomer edges (monomer A)
        e_AA_source,
        e_AA_target,
        # intramonomer edges (monomer B)
        e_BB_source,
        e_BB_target,
        # monomer charges
        total_charge_A,
        total_charge_B,
        # monomer A properties
        qA,
        muA,
        quadA,
        hfvrA,
        vwA,
        alpha_0_A,
        hlistA,
        # monomer B properties
        qB,
        muB,
        quadB,
        hfvrB,
        vwB,
        alpha_0_B,
        hlistB,
        # intermolecular edges (full)
        e_AB_source_all,
        e_AB_target_all,
        # intramonomer edges (full)
        e_AA_source_all,
        e_AA_target_all,
        e_BB_source_all,
        e_BB_target_all,
    ):
        # counts
        # natomA = ZA.size(0)
        # natomB = ZB.size(0)
        # ndimer = total_charge_A.size(0)
        """
        Compute SAPT-like energy components for a batch of dimers from atomic inputs and graph edges.
        
        Generates short-range pairwise predictions via intramonomer message passing, applies directional projections and classical corrections (exchange, induction placeholder), computes multipole electrostatics for short- and long-range edges, aggregates per-dimer energies, pads/aligns electrostatic terms, and returns combined and intermediate outputs.
        
        Returns:
            E_output (Tensor): Per-dimer total energy array with shape [ndimer, C_total] combining short-range and electrostatic contributions.
            E_sr (Tensor): Per-edge short-range SAPT component predictions with shape [n_edges_sr, 4] (elst, exch, indu, disp) after distance scaling and classical exchange correction.
            E_elst_sr (Tensor): Per-edge short-range multipole electrostatic interactions with shape [n_edges_sr, ...].
            E_elst_lr (Tensor): Per-edge long-range multipole electrostatic interactions with shape [n_edges_lr, ...].
            hAB (Tensor): Learned atom-pair feature matrix for A->B edges used by short-range readouts.
            hBA (Tensor): Learned atom-pair feature matrix for B->A edges used by short-range readouts.
        """
        natomA = torch.tensor(ZA.size(0), dtype=torch.long)
        natomB = torch.tensor(ZB.size(0), dtype=torch.long)
        ndimer = torch.tensor(total_charge_A.size(0), dtype=torch.long)

        # interatomic distances
        dR_sr, dR_sr_xyz = self.get_distances(RA, RB, e_ABsr_source, e_ABsr_target)
        dR_lr, dR_lr_xyz = self.get_distances(RA, RB, e_ABlr_source, e_ABlr_target)

        # intramonomer distances
        dRA, dRA_xyz = self.get_distances(RA, RA, e_AA_source, e_AA_target)
        dRB, dRB_xyz = self.get_distances(RB, RB, e_BB_source, e_BB_target)


        # interatomic unit vectors
        dR_sr_unit = dR_sr_xyz / dR_sr.unsqueeze(1)
        dRA_unit = dRA_xyz / dRA.unsqueeze(1)
        dRB_unit = dRB_xyz / dRB.unsqueeze(1)

        # interatomic distances for short range, long range, and intramonomer edges

        # distance encodings
        rbf_sr = self.distance_layer_im(dR_sr)
        rbfA = self.distance_layer(dRA)
        rbfB = self.distance_layer(dRB)

        ################################################################
        ### predict SAPT components via intramonomer message passing ###
        ################################################################

        # invariant hidden state lists
        hA_list = [self.embed_layer(ZA).view(ZA.size(0), -1)]
        hB_list = [self.embed_layer(ZB).view(ZB.size(0), -1)]

        # directional hidden state lists
        hA_dir_list = []
        hB_dir_list = []

        # TODO: need to determine how to handle all monA in batch having no
        # monomer edges (single atoms)
        for i in range(self.n_message):
            mA_ij = self.get_messages(
                hA_list[0], hA_list[-1], rbfA, e_AA_source, e_AA_target
            )
            mB_ij = self.get_messages(
                hB_list[0], hB_list[-1], rbfB, e_BB_source, e_BB_target
            )
            if mA_ij is None or mB_ij is None:
                # Single-atom corner case; skip
                hA_list.append(hA_list[-1])
                hB_list.append(hB_list[-1])
                continue

            #################
            ### invariant ###
            #################

            # sum each atom's messages
            mA_i = scatter_sum_compile(mA_ij, e_AA_source, reduce="sum", dim_size=natomA)
            mB_i = scatter_sum_compile(mB_ij, e_BB_source, reduce="sum", dim_size=natomB)

            # get the next hidden state of the atom
            hA_next = self.update_layers[i](mA_i)
            hB_next = self.update_layers[i](mB_i)

            hA_list.append(hA_next)
            hB_list.append(hB_next)

            ###################
            ### directional ###
            ###################

            mA_ij_dir = self.directional_layers[i](mA_ij)
            mB_ij_dir = self.directional_layers[i](mB_ij)
            mA_ij_dir = torch.einsum("ex,em->exm", dRA_unit, mA_ij_dir)
            mB_ij_dir = torch.einsum("ex,em->exm", dRB_unit, mB_ij_dir)

            # sum directional messages to get directional atomic hidden states
            # NOTE: this summation must be linear to guarantee equivariance.
            #       because of this constraint, we applied a dense net before
            #       the summation, not after
            hA_dir = scatter_sum_compile(
                mA_ij_dir, e_AA_source, dim=0, reduce="sum", dim_size=natomA
            )
            hB_dir = scatter_sum_compile(
                mB_ij_dir, e_BB_source, dim=0, reduce="sum", dim_size=natomB
            )
            hA_dir_list.append(hA_dir)
            hB_dir_list.append(hB_dir)

        # concatenate hidden states over MP iterations
        hA = torch.cat(hA_list, dim=-1)
        hB = torch.cat(hB_list, dim=-1)

        # mock right sized output with N_dimer, 4 components

        # atom-pair features are a combo of atomic hidden states and the interatomic distance
        hAB = self.get_pair(hA, hB, qA, qB, rbf_sr, e_ABsr_source, e_ABsr_target)
        hBA = self.get_pair(hB, hA, qB, qA, rbf_sr, e_ABsr_target, e_ABsr_source)

        # project the directional atomic hidden states along the interatomic axis
        hA_dir = torch.cat(hA_dir_list, dim=-1)
        hB_dir = torch.cat(hB_dir_list, dim=-1)
        # hA_dir = torch.cat(hA_dir_list, dim=-1) if len(hA_dir_list) > 0 else None
        # hB_dir = torch.cat(hB_dir_list, dim=-1) if len(hB_dir_list) > 0 else None
        # if (hA_dir is not None) and (hB_dir is not None):
        #     hA_dir_s = hA_dir.index_select(0, e_ABsr_source)
        #     hB_dir_t = hB_dir.index_select(0, e_ABsr_target)
        #
        #     # Dot with ± unit vector
        #     hA_dir_blah = torch.einsum("axf,ax->af", hA_dir_s, dR_sr_unit)
        #     hB_dir_blah = torch.einsum("axf,ax->af", hB_dir_t, -dR_sr_unit)
        #     # Concatenate
        #     hAB = torch.cat([hAB, hA_dir_blah, hB_dir_blah], dim=1)
        #     hBA = torch.cat([hBA, hB_dir_blah, hA_dir_blah], dim=1)

        hA_dir_source = hA_dir.index_select(0, e_ABsr_source)
        hB_dir_target = hB_dir.index_select(0, e_ABsr_target)

        hA_dir_blah = torch.einsum("axf,ax->af", hA_dir_source, dR_sr_unit)
        hB_dir_blah = torch.einsum("axf,ax->af", hB_dir_target, -dR_sr_unit)

        hAB = torch.cat([hAB, hA_dir_blah, hB_dir_blah], dim=1)
        hBA = torch.cat([hBA, hB_dir_blah, hA_dir_blah], dim=1)

        # run atom-pair features through a dense net to predict SAPT components
        EAB_sr = torch.cat(
            [
                self.readout_layer_elst(hAB),
                self.readout_layer_exch(hAB),
                self.readout_layer_indu(hAB),
                self.readout_layer_disp(hAB),
            ],
            dim=1,
        )
        EBA_sr = torch.cat(
            [
                self.readout_layer_elst(hBA),
                self.readout_layer_exch(hBA),
                self.readout_layer_indu(hBA),
                self.readout_layer_disp(hBA),
            ],
            dim=1,
        )

        E_sr = EAB_sr + EBA_sr

        cutoff = (1.0 / (dR_sr**3)).unsqueeze(-1)
        E_sr *= cutoff
        # When natomsA == 1, this causes nans...
        # print("Exit")
        # return torch.zeros(ndimer, 4), torch.zeros(ndimer, 4), torch.zeros(ndimer, 4), torch.zeros(ndimer, 4), hAB, hBA
        # if ndimer < 1:
        #     return torch.zeros(ndimer, 4), torch.zeros(ndimer, 4), torch.zeros(ndimer, 4), torch.zeros(ndimer, 4), hAB, hBA
        # print(E_sr.dtype, dimer_ind.dtype, ndimer.dtype)

        # CLASSICAL EXCHANGE
        S_ij = self.valence_width_exch(e_ABsr_source, e_ABsr_target, vwA, vwB, dR_sr)
        EAB_exch_quotient = self.readout_layer_exch_quotient(hAB)
        EBA_exch_quotient = self.readout_layer_exch_quotient(hBA)
        E_exch_quotient = ((EAB_exch_quotient + EBA_exch_quotient) / 2).squeeze(-1)
        E_exch_classical = torch.mul(S_ij, E_exch_quotient)
        E_sr[:, 1] = E_exch_classical + E_sr[:, 1]

        # CLASSICAL INDUCTION - INDUCED DIPOLE

        E_indu = self.induced_dipole_indu(
            RA,
            RB,
            qA,
            muA,
            quadA,
            qB,
            muB,
            quadB,
            hfvrA,
            hfvrB,
            alpha_0_A,
            alpha_0_B,
            S_ij,
            e_AB_source_all,
            e_AB_target_all,
        )
        print(f"{E_indu = }")

        E_sr_dimer = scatter_sum_compile(E_sr, dimer_ind, dim=0, reduce="add", dim_size=ndimer)

        # print(f"{E_sr_dimer.size() = }")
        # print(f"{E_sr.size() = }")
        ####################################################
        ### predict multipole electrostatic interactions ###
        ####################################################

        E_elst_sr = self.mtp_elst(
            qA,
            muA,
            quadA,
            qB,
            muB,
            quadB,
            e_ABsr_source,
            e_ABsr_target,
            dR_sr,
            dR_sr_xyz,
        )
        # print(vwA.size(), vwB.size(), dR_sr.size())
        # print(f"{vwA = }")
        # print(f"{vwB = }")
        #
        # print()
        # print(f"{S_ij.size() = }")
        # print(S_ij)
        #
        # print()
        # print(f"{E_elst_sr.size() = }")
        # print(E_elst_sr)

        E_elst_sr_dimer = scatter_sum_compile(
            E_elst_sr, dimer_ind, dim=0, reduce="add", dim_size=ndimer
        )
        # print()
        # print(f"{E_elst_sr_dimer.size() = }")
        # print(E_elst_sr_dimer)
        E_elst_sr_dimer = E_elst_sr_dimer.unsqueeze(-1)

        E_elst_lr = self.mtp_elst(
            qA,
            muA,
            quadA,
            qB,
            muB,
            quadB,
            e_ABlr_source,
            e_ABlr_target,
            dR_lr,
            dR_lr_xyz,
        )
        E_elst_lr_dimer = scatter_sum_compile(
            E_elst_lr, dimer_ind_lr, dim=0, reduce="add", dim_size=ndimer
        )
        E_elst_lr_dimer = E_elst_lr_dimer.unsqueeze(-1)

        # Example shapes for clarity:
        # E_elst_sr_dimer : shape [N_sr, C]  (some # of rows, e.g. 4 columns)
        # E_elst_lr_dimer : shape [N_lr, C]
        # ndimer          : desired # of rows (N_sr, N_lr <= ndimer)

        # 1) Expand E_elst_sr_dimer up to ndimer rows if needed
        N_sr, num_cols = E_elst_sr_dimer.shape
        sr_expanded = E_elst_sr_dimer.new_zeros((ndimer, num_cols))
        sr_expanded[:N_sr] = E_elst_sr_dimer
        E_elst_sr_dimer = sr_expanded

        # 2) Expand E_elst_lr_dimer similarly
        N_lr, num_cols = E_elst_lr_dimer.shape
        lr_expanded = E_elst_lr_dimer.new_zeros((ndimer, num_cols))
        lr_expanded[:N_lr] = E_elst_lr_dimer
        E_elst_lr_dimer = lr_expanded

        # 3) Sum them
        E_elst_dimer = E_elst_sr_dimer + E_elst_lr_dimer

        # 4) Finally, pad columns by 3 if you want to go from shape [ndimer, 4] to [ndimer, 7]
        rows, cols = E_elst_dimer.shape
        padded = E_elst_dimer.new_zeros((rows, cols + 3))
        padded[:, :cols] = E_elst_dimer
        E_elst_dimer = padded
        E_output = E_sr_dimer + E_elst_dimer
        return E_output, E_sr, E_elst_sr, E_elst_lr, hAB, hBA


class APNet3Model:
    def __init__(
        self,
        dataset=None,
        atom_model=None,
        pre_trained_model_path=None,
        atom_model_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut_im=8.0,
        r_cut=5.0,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_root="data",
        ds_max_size=None,
        ds_atomic_batch_size=200,
        ds_force_reprocess=False,
        ds_skip_process=False,
        ds_num_devices=1,
        ds_datapoint_storage_n_objects=1000,
        ds_prebatched=False,
        print_lvl=0,
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
        self.atom_model = AtomHirshfeldMPNN()
        if atom_model_pre_trained_path:
            print(
                f"Loading pre-trained AtomHirshfeldMPNN model from {atom_model_pre_trained_path}"
            )
            checkpoint = torch.load(
                atom_model_pre_trained_path, map_location=device, weights_only=False
            )
            self.atom_model = AtomHirshfeldMPNN(
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
            self.atom_model = atom_model
        else:
            print(
                """No atom model provided.
    Assuming atomic multipoles and embeddings are
    pre-computed and passed as input to the model.
"""
            )
        self.atom_model.to(device)
        if pre_trained_model_path:
            print(
                f"Loading pre-trained APNet3_MPNN model from {pre_trained_model_path}"
            )
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = APNet3_MPNN(
                n_message=checkpoint["config"]["n_message"],
                n_rbf=checkpoint["config"]["n_rbf"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                r_cut_im=checkpoint["config"]["r_cut_im"],
                r_cut=checkpoint["config"]["r_cut"],
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = APNet3_MPNN(
                # atom_model=self.atom_model,
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut_im=r_cut_im,
                r_cut=r_cut,
            )
        split_dbs = [2, 5, 6, 7]
        self.dataset = dataset
        if (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type not in split_dbs
        ):

            def setup_ds(fp=ds_force_reprocess):
                return apnet3_module_dataset(
                    root=ds_root,
                    r_cut=r_cut,
                    r_cut_im=r_cut_im,
                    spec_type=ds_spec_type,
                    max_size=ds_max_size,
                    force_reprocess=fp,
                    atom_model_path=atom_model_pre_trained_path,
                    atomic_batch_size=ds_atomic_batch_size,
                    num_devices=ds_num_devices,
                    skip_processed=ds_skip_process,
                    datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                    prebatched=ds_prebatched,
                    print_level=print_lvl,
                )

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset = self.dataset[:ds_max_size]
        elif (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type in split_dbs
        ):
            print("Processing Split dataset...")

            def setup_ds(fp=ds_force_reprocess):
                return [
                    apnet3_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model_path=atom_model_pre_trained_path,
                        atomic_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        split="train",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        prebatched=ds_prebatched,
                        print_level=print_lvl,
                    ),
                    apnet3_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model_path=atom_model_pre_trained_path,
                        atomic_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        split="test",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        prebatched=ds_prebatched,
                        print_level=print_lvl,
                    ),
                ]

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset[0] = self.dataset[0][:ds_max_size]
                self.dataset[1] = self.dataset[1][:ds_max_size]
        print(f"{self.dataset=}")
        self.model.to(device)
        self.device = device
        self.batch_size = None
        self.shuffle = False
        self.model_save_path = None
        self.prebatched = ds_prebatched
        return

    @torch.inference_mode()
    def predict_from_dataset(self):
        self.model.eval()
        for batch in self.dataset:
            batch = batch.to(self.device)
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
        return

    def compile_model(self):
        self.model.to(self.device)
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
        torch._dynamo.config.capture_scalar_outputs = False
        self.model = torch.compile(self.model)
        return

    def set_pretrained_model(self, ap3_model_path, am_model_path):
        checkpoint = torch.load(ap3_model_path, weights_only=False)
        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint = torch.load(am_model_path, weights_only=False)
        if "_orig_mod" not in list(self.atom_model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.atom_model.load_state_dict(model_state_dict)
        else:
            self.atom_model.load_state_dict(checkpoint["model_state_dict"])
        return

    ############################################################################
    # The main forward/eval function
    ############################################################################
    def eval_fn(self, batch):
        return self.model(
            ZA=batch.ZA,
            RA=batch.RA,
            ZB=batch.ZB,
            RB=batch.RB,
            e_ABsr_source=batch.e_ABsr_source,
            e_ABsr_target=batch.e_ABsr_target,
            dimer_ind=batch.dimer_ind,
            e_ABlr_source=batch.e_ABlr_source,
            e_ABlr_target=batch.e_ABlr_target,
            dimer_ind_lr=batch.dimer_ind_lr,
            e_AA_source=batch.e_AA_source,
            e_AA_target=batch.e_AA_target,
            e_BB_source=batch.e_BB_source,
            e_BB_target=batch.e_BB_target,
            total_charge_A=batch.total_charge_A,
            total_charge_B=batch.total_charge_B,
            qA=batch.qA,
            muA=batch.muA,
            quadA=batch.quadA,
            hfvrA=batch.hfvrA,
            vwA=batch.vwA,
            alpha_0_A=batch.alpha_0_A,
            hlistA=batch.hlistA,
            qB=batch.qB,
            muB=batch.muB,
            quadB=batch.quadB,
            hfvrB=batch.hfvrB,
            vwB=batch.vwB,
            alpha_0_B=batch.alpha_0_B,
            hlistB=batch.hlistB,
            e_AB_source_all=batch.e_AB_source_all,
            e_AB_target_all=batch.e_AB_target_all,
            e_AA_source_all=batch.e_AA_source_all,
            e_AA_target_all=batch.e_AA_target_all,
            e_BB_source_all=batch.e_BB_source_all,
            e_BB_target_all=batch.e_BB_target_all,
        )

    def _qcel_example_input(
        self,
        mols,
        batch_size=1,
        r_cut=5.0,
        r_cut_im=8.0,
    ):
        mol_data = [[*qcel_dimer_to_pyg_data(mol)] for mol in mols]
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            data_A = [d[0] for d in batch_mol_data]
            data_B = [d[1] for d in batch_mol_data]
            batch_A = atomic_datasets.atomic_collate_update_no_target(data_A)
            batch_B = atomic_datasets.atomic_collate_update_no_target(data_B)
            with torch.no_grad():
                am_out_A = self.atom_model(batch_A)
                am_out_B = self.atom_model(batch_B)
                qAs, muAs, quadAs, hfvrAs, vwAs, hlistAs = (
                    isolate_atomic_property_predictions(batch_A, am_out_A)
                )
                qBs, muBs, quadBs, hfvrBs, vwBs, hlistBs = (
                    isolate_atomic_property_predictions(batch_B, am_out_B)
                )
                if len(batch_A.total_charge.size()) == 0:
                    batch_A.total_charge = batch_A.total_charge.unsqueeze(0)
                if len(batch_B.total_charge.size()) == 0:
                    batch_B.total_charge = batch_B.total_charge.unsqueeze(0)
                dimer_ls = []
                for j in range(len(batch_mol_data)):
                    qA, muA, quadA, hfvrA, vwA, hlistA = (
                        qAs[j],
                        muAs[j],
                        quadAs[j],
                        hfvrAs[j],
                        vwAs[j],
                        hlistAs[j],
                    )
                    qB, muB, quadB, hfvrB, vwB, hlistB = (
                        qBs[j],
                        muBs[j],
                        quadBs[j],
                        hfvrBs[j],
                        vwBs[j],
                        hlistBs[j],
                    )
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                        hfvrA = hfvrA.unsqueeze(0).unsqueeze(0)
                        vwA = vwA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                        hfvrA = hfvrA.unsqueeze(-1)
                        vwA = vwA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                        hfvrB = hfvrB.unsqueeze(0).unsqueeze(0)
                        vwB = vwB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                        hfvrB = hfvrB.unsqueeze(-1)
                        vwB = vwB.unsqueeze(-1)
                    e_AA_source, e_AA_target, e_AA_source_all, e_AA_target_all = (
                        pairwise_edges(data_A[j].R, r_cut, full_indices=True)
                    )
                    e_BB_source, e_BB_target, e_BB_source_all, e_BB_target_all = (
                        pairwise_edges(data_B[j].R, r_cut, full_indices=True)
                    )
                    (
                        e_ABsr_source,
                        e_ABsr_target,
                        e_ABlr_source,
                        e_ABlr_target,
                        e_AB_source_all,
                        e_AB_target_all,
                    ) = pairwise_edges_im(
                        data_A[j].R, data_B[j].R, r_cut_im, full_indices=True
                    )
                    dimer_ind = torch.ones((1), dtype=torch.long) * 0
                    alpha_0_A = torch.tensor(
                        [free_atom_polarizabilities[int(z)] for z in batch_A.x]
                    )
                    alpha_0_B = torch.tensor(
                        [free_atom_polarizabilities[int(z)] for z in batch_B.x]
                    )
                    data = Data(
                        ZA=data_A[j].x,
                        RA=data_A[j].R,
                        ZB=data_B[j].x,
                        RB=data_B[j].R,
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
                        # intramonomer edges (monomer B)
                        e_BB_source=e_BB_source,
                        e_BB_target=e_BB_target,
                        # monomer charges
                        total_charge_A=data_A[j].total_charge,
                        total_charge_B=data_B[j].total_charge,
                        # monomer A properties
                        qA=qA,
                        muA=muA,
                        quadA=quadA,
                        hfvrA=hfvrA,
                        vwA=vwA,
                        alpha_0_A=alpha_0_A,
                        hlistA=hlistA,
                        # monomer B properties
                        qB=qB,
                        muB=muB,
                        quadB=quadB,
                        hfvrB=hfvrB,
                        vwB=vwB,
                        alpha_0_B=alpha_0_B,
                        hlistB=hlistB,
                        # intermolecular edges (full)
                        e_AB_source_all=e_AB_source_all,
                        e_AB_target_all=e_AB_target_all,
                        # intramonomer edges (full)
                        e_AA_source_all=e_AA_source_all,
                        e_AA_target_all=e_AA_target_all,
                        e_BB_source_all=e_BB_source_all,
                        e_BB_target_all=e_BB_target_all,
                    )
                    dimer_ls.append(data)
                dimer_batch = pairwise_datasets.apnet3_collate_update_no_target(
                    dimer_ls
                )
        return dimer_batch

    @torch.inference_mode()
    def predict_qcel_mols(
        self,
        mols,
        batch_size=1,
        r_cut=5.0,
        r_cut_im=8.0,
    ):
        mol_data = [[*qcel_dimer_to_pyg_data(mol)] for mol in mols]
        predictions = np.zeros((len(mol_data), 4))
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            data_A = [d[0] for d in batch_mol_data]
            data_B = [d[1] for d in batch_mol_data]
            batch_A = atomic_datasets.atomic_collate_update_no_target(data_A)
            batch_B = atomic_datasets.atomic_collate_update_no_target(data_B)
            with torch.no_grad():
                am_out_A = self.atom_model(batch_A)
                am_out_B = self.atom_model(batch_B)
                qAs, muAs, quadAs, hfvrAs, vwAs, hlistAs = (
                    isolate_atomic_property_predictions(batch_A, am_out_A)
                )
                qBs, muBs, quadBs, hfvrBs, vwBs, hlistBs = (
                    isolate_atomic_property_predictions(batch_B, am_out_B)
                )
                if len(batch_A.total_charge.size()) == 0:
                    batch_A.total_charge = batch_A.total_charge.unsqueeze(0)
                if len(batch_B.total_charge.size()) == 0:
                    batch_B.total_charge = batch_B.total_charge.unsqueeze(0)
                dimer_ls = []
                for j in range(len(batch_mol_data)):
                    qA, muA, quadA, hfvrA, vwA, hlistA = (
                        qAs[j],
                        muAs[j],
                        quadAs[j],
                        hfvrAs[j],
                        vwAs[j],
                        hlistAs[j],
                    )
                    qB, muB, quadB, hfvrB, vwB, hlistB = (
                        qBs[j],
                        muBs[j],
                        quadBs[j],
                        hfvrBs[j],
                        vwBs[j],
                        hlistBs[j],
                    )
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                        hfvrA = hfvrA.unsqueeze(0).unsqueeze(0)
                        vwA = vwA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                        hfvrA = hfvrA.unsqueeze(-1)
                        vwA = vwA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                        hfvrB = hfvrB.unsqueeze(0).unsqueeze(0)
                        vwB = vwB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                        hfvrB = hfvrB.unsqueeze(-1)
                        vwB = vwB.unsqueeze(-1)
                    e_AA_source, e_AA_target, e_AA_source_all, e_AA_target_all = (
                        pairwise_edges(data_A[j].R, r_cut, full_indices=True)
                    )
                    e_BB_source, e_BB_target, e_BB_source_all, e_BB_target_all = (
                        pairwise_edges(data_B[j].R, r_cut, full_indices=True)
                    )
                    (
                        e_ABsr_source,
                        e_ABsr_target,
                        e_ABlr_source,
                        e_ABlr_target,
                        e_AB_source_all,
                        e_AB_target_all,
                    ) = pairwise_edges_im(
                        data_A[j].R, data_B[j].R, r_cut_im, full_indices=True
                    )
                    dimer_ind = torch.ones((1), dtype=torch.long) * 0
                    alpha_0_A = torch.tensor(
                        [free_atom_polarizabilities[int(z)] for z in batch_A.x]
                    )
                    alpha_0_B = torch.tensor(
                        [free_atom_polarizabilities[int(z)] for z in batch_B.x]
                    )
                    data = Data(
                        ZA=data_A[j].x,
                        RA=data_A[j].R,
                        ZB=data_B[j].x,
                        RB=data_B[j].R,
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
                        # intramonomer edges (monomer B)
                        e_BB_source=e_BB_source,
                        e_BB_target=e_BB_target,
                        # monomer charges
                        total_charge_A=data_A[j].total_charge,
                        total_charge_B=data_B[j].total_charge,
                        # monomer A properties
                        qA=qA,
                        muA=muA,
                        quadA=quadA,
                        hfvrA=hfvrA,
                        vwA=vwA,
                        alpha_0_A=alpha_0_A,
                        hlistA=hlistA,
                        # monomer B properties
                        qB=qB,
                        muB=muB,
                        quadB=quadB,
                        hfvrB=hfvrB,
                        vwB=vwB,
                        alpha_0_B=alpha_0_B,
                        hlistB=hlistB,
                        # intermolecular edges (full)
                        e_AB_source_all=e_AB_source_all,
                        e_AB_target_all=e_AB_target_all,
                        # intramonomer edges (full)
                        e_AA_source_all=e_AA_source_all,
                        e_AA_target_all=e_AA_target_all,
                        e_BB_source_all=e_BB_source_all,
                        e_BB_target_all=e_BB_target_all,
                    )
                    dimer_ls.append(data)
                dimer_batch = pairwise_datasets.apnet3_collate_update_no_target(
                    dimer_ls
                )
                dimer_batch.to(self.device)
                preds = self.eval_fn(dimer_batch)
                predictions[i : i + batch_size] = preds[0].cpu().numpy()
        return predictions

    def example_input(self):
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
        return self._qcel_example_input([mol], batch_size=1)

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
        self, dataloader, loss_fn, optimizer, rank_device, scheduler
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
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
            preds = E_sr_dimer.reshape(-1, 4)
            comp_errors = preds - batch.y
            batch_loss = (
                torch.mean(torch.square(comp_errors))
                if (loss_fn is None)
                else loss_fn(preds, batch.y)
            )
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            comp_errors_t.append(comp_errors.detach().cpu())
        if scheduler is not None:
            scheduler.step()

        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, 4)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t))
        elst_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 0]))
        exch_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 1]))
        indu_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 2]))
        disp_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 3]))
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    # @torch.inference_mode()
    def __evaluate_batches_single_proc(self, dataloader, loss_fn, rank_device):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        # print time every 1% of data
        t = time.time()
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer, _, _, _, _, _ = self.eval_fn(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                comp_errors = preds - batch.y
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, batch.y)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
                # if n % 50 == 0:
                #     print(f"    Time for {n/len(dataloader)*100:.2f}%", time.time() - t)

        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, 4)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t))
        elst_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 0]))
        exch_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 1]))
        indu_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 2]))
        disp_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 3]))
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    ########################################################################
    # SINGLE-PROCESS TRAINING
    ########################################################################

    def __train_batches(
        self, rank, dataloader, loss_fn, optimizer, rank_device, scheduler
    ):
        self.model.train()
        total_loss = 0.0
        total_error = 0.0
        elst_error = 0.0
        exch_error = 0.0
        indu_error = 0.0
        disp_error = 0.0
        count = 0
        for n, batch in enumerate(dataloader):
            batch_loss = 0.0
            optimizer.zero_grad()
            batch = batch.to(rank_device)
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
            preds = E_sr_dimer.reshape(-1, 4)
            comp_errors = preds - batch.y
            if loss_fn is None:
                batch_loss = torch.mean(torch.square(comp_errors))
            else:
                batch_loss = loss_fn(preds, batch.y)

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            total_errors = preds.sum(dim=1) - batch.y.sum(dim=1)
            total_error += torch.sum(torch.abs(total_errors)).item()
            elst_error += torch.sum(torch.abs(comp_errors[:, 0])).item()
            exch_error += torch.sum(torch.abs(comp_errors[:, 1])).item()
            indu_error += torch.sum(torch.abs(comp_errors[:, 2])).item()
            disp_error += torch.sum(torch.abs(comp_errors[:, 3])).item()
            count += preds.numel()
        if scheduler is not None:
            scheduler.step()

        total_loss = torch.tensor(total_loss, dtype=torch.float32, device=rank_device)
        total_error = torch.tensor(total_error, dtype=torch.float32, device=rank_device)
        elst_error = torch.tensor(elst_error, dtype=torch.float32, device=rank_device)
        exch_error = torch.tensor(exch_error, dtype=torch.float32, device=rank_device)
        indu_error = torch.tensor(indu_error, dtype=torch.float32, device=rank_device)
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(elst_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(exch_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(indu_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        total_MAE_t = (total_error / count).cpu()
        elst_MAE_t = (elst_error / count).cpu()
        exch_MAE_t = (exch_error / count).cpu()
        indu_MAE_t = (indu_error / count).cpu()
        disp_MAE_t = (disp_error / count).cpu()
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    # @torch.inference_mode()
    def __evaluate_batches(self, rank, dataloader, loss_fn, rank_device):
        self.model.eval()
        total_loss = 0.0
        total_error = 0.0
        elst_error = 0.0
        exch_error = 0.0
        indu_error = 0.0
        disp_error = 0.0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_loss = 0.0
                batch = batch.to(rank_device)
                E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                comp_errors = preds - batch.y
                if loss_fn is None:
                    batch_loss = torch.mean(torch.square(comp_errors))
                else:
                    batch_loss = loss_fn(preds, batch.y)

                total_loss += batch_loss.item()
                total_errors = preds.sum(dim=1) - batch.y.sum(dim=1)
                total_error += torch.sum(torch.abs(total_errors)).item()
                elst_error += torch.sum(torch.abs(comp_errors[:, 0])).item()
                exch_error += torch.sum(torch.abs(comp_errors[:, 1])).item()
                indu_error += torch.sum(torch.abs(comp_errors[:, 2])).item()
                disp_error += torch.sum(torch.abs(comp_errors[:, 3])).item()
                count += preds.numel()

        total_loss = torch.tensor(total_loss, device=rank_device)
        total_error = torch.tensor(total_error, device=rank_device)
        elst_error = torch.tensor(elst_error, device=rank_device)
        exch_error = torch.tensor(exch_error, device=rank_device)
        indu_error = torch.tensor(indu_error, device=rank_device)
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(elst_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(exch_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(indu_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        total_MAE_t = (total_error / count).cpu()
        elst_MAE_t = (elst_error / count).cpu()
        exch_MAE_t = (exch_error / count).cpu()
        indu_MAE_t = (indu_error / count).cpu()
        disp_MAE_t = (disp_error / count).cpu()
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

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
        lr_decay=None,
    ):
        print(f"{self.device.type=}")
        if self.device.type == "cpu":
            rank_device = "cpu"
        else:
            rank_device = rank
        if world_size > 1:
            self.__setup(rank, world_size)
        if rank == 0:
            print("Setup complete")

        self.model = self.model.to(rank_device)
        print(f"{rank=}, {world_size=}, {rank_device=}")
        if rank == 0:
            print("Model Transferred to device")
        if world_size > 1:
            first_pass_data = APNet2_DataLoader(
                dataset=test_dataset[:batch_size],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=apnet3_collate_update,
            )
            for b in first_pass_data:
                b.to(rank_device)
                self.eval_fn(b)
                break
            self.model = DDP(
                self.model,
            )

        if rank == 0:
            print("Model DDP wrapped")

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

        train_loader = APNet2_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            collate_fn=apnet3_collate_update,
        )

        test_loader = APNet2_DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=apnet3_collate_update,
        )
        if rank == 0:
            print("Loaders setup\n")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if lr_decay:
            scheduler = InverseTimeDecayLR(
                optimizer, lr, len(train_loader) * 60, lr_decay
            )
        else:
            scheduler = None
        criterion = None
        lowest_test_loss = torch.tensor(float("inf"))
        self.model = self.model.to(rank_device)

        if rank == 0:
            print(
                "                                       Total            Elst            Exch            Ind            Disp",
                flush=True,
            )
        t1 = time.time()
        with torch.no_grad():
            train_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t = (
                self.__evaluate_batches(rank, train_loader, criterion, rank_device)
            )
            test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
                self.__evaluate_batches(rank, test_loader, criterion, rank_device)
            )
            dt = time.time() - t1
            if rank == 0:
                print(
                    f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} {indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} {disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f}",
                    flush=True,
                )
        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            train_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t = (
                self.__train_batches(
                    rank,
                    train_loader,
                    criterion,
                    optimizer,
                    rank_device,
                    scheduler,
                )
            )
            test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
                self.__evaluate_batches(rank, test_loader, criterion, rank_device)
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        print("Saving model")
                        cpu_model = unwrap_model(self.model).to("cpu")
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_rbf": cpu_model.n_rbf,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "r_cut_im": cpu_model.r_cut_im,
                                    "r_cut": cpu_model.r_cut,
                                },
                            },
                            self.model_save_path,
                        )
                        self.model.to(rank_device)
                else:
                    test_lowered = " "
                dt = time.time() - t1
                test_loss = 0.0
                print(
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} {indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} {disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f} {test_lowered}",
                    flush=True,
                )

        if world_size > 1:
            self.__cleanup()
        return

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
        lr_decay=None,
        skip_compile=False,
    ):
        # (1) Compile Model
        rank_device = self.device
        self.model.to(rank_device)
        batch = self.example_input()
        batch.to(rank_device)
        self.model(**batch)
        if not skip_compile:
            print("Compiling model")
            self.compile_model()

        # (2) Dataloaders
        if train_dataset.prebatched:
            collate_fn = apnet3_collate_update_prebatched
        else:
            collate_fn = apnet3_collate_update
        train_loader = APNet2_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test_loader = APNet2_DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        # (3) Optim/Scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # scheduler = ModLambdaDecayLR(optimizer, lr_decay, lr) if lr_decay else None
        scheduler = (
            InverseTimeDecayLR(optimizer, lr, len(train_loader) * 2, lr_decay)
            if lr_decay
            else None
        )
        criterion = None  # defaults to MSE

        # (4) Print table header
        print(
            "                                       Total            Elst            Exch            Ind            Disp",
            flush=True,
        )

        # (5) Evaluate once pre-training
        t0 = time.time()
        train_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t = (
            self.__evaluate_batches_single_proc(train_loader, criterion, rank_device)
        )
        test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
            self.__evaluate_batches_single_proc(test_loader, criterion, rank_device)
        )

        print(
            f"  (Pre-training) ({time.time() - t0:<7.2f}s)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} "
            f"{elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} "
            f"{indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} {disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f}",
            flush=True,
        )

        # (6) Main training loop
        lowest_test_loss = test_loss
        for epoch in range(n_epochs):
            t1 = time.time()
            train_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t = (
                self.__train_batches_single_proc(
                    train_loader, criterion, optimizer, rank_device, scheduler
                )
            )
            test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
                self.__evaluate_batches_single_proc(test_loader, criterion, rank_device)
            )

            # Track best model
            star_marker = " "
            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                star_marker = "*"
                if self.model_save_path:
                    cpu_model = unwrap_model(self.model).to("cpu")
                    torch.save(
                        {
                            "model_state_dict": cpu_model.state_dict(),
                            "config": {
                                "n_message": cpu_model.n_message,
                                "n_rbf": cpu_model.n_rbf,
                                "n_neuron": cpu_model.n_neuron,
                                "n_embed": cpu_model.n_embed,
                                "r_cut_im": cpu_model.r_cut_im,
                                "r_cut": cpu_model.r_cut,
                            },
                        },
                        self.model_save_path,
                    )
                    self.model.to(rank_device)

            print(
                f"  EPOCH: {epoch:4d} ({time.time() - t1:<7.2f}s)  MAE: "
                f"{total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} "
                f"{exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} {indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} "
                f"{disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f} {star_marker}",
                flush=True,
            )

    def train(
        self,
        dataset=None,
        n_epochs=50,
        lr=5e-4,
        split_percent=0.9,
        model_path=None,
        shuffle=False,
        dataloader_num_workers=4,
        optimize_for_speed=True,
        world_size=1,
        omp_num_threads_per_process=6,
        lr_decay=None,
        random_seed=42,
        skip_compile=False,
    ):
        """
        hyperparameters match the defaults in the original code:
        https://chemrxiv.org/engage/chemrxiv/article-details/65ccd41866c1381729a2b885
        """
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

        print("~~ Training APNet2Model ~~", flush=True)
        print(
            f"    Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples"
        )
        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {self.model.n_message=}", flush=True)
        print(f"  {self.model.n_neuron=}", flush=True)
        print(f"  {self.model.n_embed=}", flush=True)
        print(f"  {self.model.n_rbf=}", flush=True)
        print(f"  {self.model.r_cut=}", flush=True)
        print(f"  {self.model.r_cut_im=}", flush=True)
        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {lr=}\n", flush=True)
        print(f"  {lr_decay=}\n", flush=True)
        if self.prebatched:
            print(f"  Prebatched training data: setting batch_size=1", flush=True)
            batch_size = 1
        print(f"  {batch_size=}", flush=True)

        if self.device.type == "cuda":
            pin_memory = True
        else:
            pin_memory = False

        # if optimize_for_speed:
        # torch.jit.enable_onednn_fusion(False)
        # torch.autograd.set_detect_anomaly(True)

        self.shuffle = shuffle

        if world_size > 1:
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
                    lr_decay,
                ),
                nprocs=world_size,
                join=True,
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
                lr_decay=lr_decay,
                skip_compile=skip_compile,
            )
        return
