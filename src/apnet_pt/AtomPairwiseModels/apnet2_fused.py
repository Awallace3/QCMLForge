import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import warnings
import time
from ..AtomModels.ap2_atom_model import AtomMPNN
from ..hf_pretrained import resolve_pretrained_path
from apnet_pt.util import scatter_sum_compile
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
from torch.nn.parallel import DistributedDataParallel as DDP
import qcelemental as qcel
from copy import deepcopy
from apnet_pt.torch_util import set_weights_to_value


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


class APNet2_AM_MPNN(nn.Module):
    def __init__(
        self,
        atom_model: AtomMPNN,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut_im=8.0,
        r_cut=5.0,
        return_hidden_states=False,
    ):
        # super().__init__(aggr="add")
        super().__init__()
        self.atom_model = atom_model
        # set atom_model parameters to be not trainable
        self.atom_model.requires_grad_(False)

        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut_im = r_cut_im
        self.r_cut = r_cut
        self.return_hidden_states = return_hidden_states

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

        # TF implementation uses 3/2 factor for quadrupoles
        # quadA_source = (3.0 / 2.0) * quadA.index_select(0, e_ABsr_source)
        # quadB_source = (3.0 / 2.0) * quadB.index_select(0, e_ABsr_target)
        quadA_source = quadA.index_select(0, e_ABsr_source)
        quadB_source = quadB.index_select(0, e_ABsr_target)

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
        E_qQ = torch.einsum("xyz,xyz->x", T2, qA_quadB_source + qB_quadA_source) / 3.0

        E_elst = 627.509 * (E_qq + E_qu + E_qQ + E_uu)
        return E_elst

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

    @torch.compile
    def readouts(self, H):
        return torch.cat(
            [
                self.readout_layer_elst(H),
                self.readout_layer_exch(H),
                self.readout_layer_indu(H),
                self.readout_layer_disp(H),
            ],
            dim=1,
        )

    def forward(
        self,
        batch,
    ):
        """
        Compute short-range SAPT components via intramonomer message passing, evaluate multipole electrostatic interactions for intermonomer pairs, and assemble per-edge and per-dimer energy outputs; optionally return atom-pair feature tensors and directional cutoff factors.

        Parameters:
            batch: fused dimer batch object providing the following attributes used by this method:
                ZA, RA, ZB, RB (atom types and coordinates for monomers A and B);
                e_ABsr_source, e_ABsr_target, dimer_ind (short-range intermolecular edge sources, targets, and per-edge dimer indices);
                e_ABlr_source, e_ABlr_target, dimer_ind_lr (long-range intermolecular edges and dimer indices);
                e_AA_source, e_AA_target, e_BB_source, e_BB_target (intramonomer edge indices for A and B);
                batch_atomic_A, batch_atomic_B (inputs consumed by the frozen atom_model);
                total_charge_A (used for determining ndimer size).

        Returns:
            If self.return_hidden_states is True:
                (E_output, E_sr_dimer, E_elst_sr_dimer, E_elst_lr_dimer, hAB, hBA, cutoff)
                - E_output (torch.Tensor): per-dimer assembled energy tensor (rows = ndimer, columns = 7: 4 SAPT components + padded electrostatic columns).
                - E_sr_dimer (torch.Tensor): per-dimer summed short-range SAPT components (shape [ndimer, 4]).
                - E_elst_sr_dimer (torch.Tensor): per-dimer summed short-range multipole electrostatic contributions (unsqueezed to [,1] before expansion).
                - E_elst_lr_dimer (torch.Tensor): per-dimer summed long-range multipole electrostatic contributions (unsqueezed to [,1] before expansion).
                - hAB, hBA (torch.Tensor): atom-pair feature matrices used as readout inputs for A→B and B→A edges.
                - cutoff (torch.Tensor): per-edge distance-based scaling factor used for short-range SAPT (shape matches number of short-range edges).
            Otherwise:
                (E_output, E_sr, E_elst_sr, E_elst_lr, hAB, hBA)
                - E_output (torch.Tensor): per-dimer assembled energy tensor as above.
                - E_sr (torch.Tensor): per-edge short-range SAPT component predictions (shape [n_edges_sr, 4]).
                - E_elst_sr (torch.Tensor): per-edge short-range multipole electrostatic contributions.
                - E_elst_lr (torch.Tensor): per-edge long-range multipole electrostatic contributions.
                - hAB, hBA (torch.Tensor): atom-pair feature matrices as above.

        Notes:
            - The function expects the contained atom_model to provide per-atom multipole properties (monopole, dipole, quadrupole) and uses internal distance encodings and message-passing to predict SAPT components.
            - Per-edge values are summed to per-dimer aggregates; multipole electrostatic outputs are expanded/padded as needed to match the per-dimer output shape.
        """
        ZA = batch.ZA
        RA = batch.RA
        ZB = batch.ZB
        RB = batch.RB
        # short range intermolecular edges
        e_ABsr_source = batch.e_ABsr_source
        e_ABsr_target = batch.e_ABsr_target
        dimer_ind = batch.dimer_ind
        # batch.long range intermolecular edges
        e_ABlr_source = batch.e_ABlr_source
        e_ABlr_target = batch.e_ABlr_target
        dimer_ind_lr = batch.dimer_ind_lr
        # batch.intramonomer edges (monomer A)
        e_AA_source = batch.e_AA_source
        e_AA_target = batch.e_AA_target
        # batch.intramonomer edges (monomer B)
        e_BB_source = batch.e_BB_source
        e_BB_target = batch.e_BB_target
        # counts
        natomA = ZA.size(0)
        natomB = ZB.size(0)
        ndimer = batch.total_charge_A.size(0)

        # interatomic distances
        dR_sr, dR_sr_xyz = self.get_distances(RA, RB, e_ABsr_source, e_ABsr_target)
        dR_lr, dR_lr_xyz = self.get_distances(RA, RB, e_ABlr_source, e_ABlr_target)
        # TODO: need to handle single atoms correctly without self edge because
        # this goes to zero causing nans later...
        dRA, dRA_xyz = self.get_distances(RA, RA, e_AA_source, e_AA_target)
        dRB, dRB_xyz = self.get_distances(RB, RB, e_BB_source, e_BB_target)

        # interatomic unit vectors
        dR_sr_unit = dR_sr_xyz / dR_sr.unsqueeze(1)
        dRA_unit = dRA_xyz / dRA.unsqueeze(1)
        dRB_unit = dRB_xyz / dRB.unsqueeze(1)

        # distance encodings
        rbf_sr = self.distance_layer_im(dR_sr)
        rbfA = self.distance_layer(dRA)
        rbfB = self.distance_layer(dRB)

        ##########################################################
        ### predict monomer properties w/ pretrained AtomModel ###
        ##########################################################

        qA, muA, quadA, _ = self.atom_model(batch.batch_atomic_A)
        qA = qA.reshape(-1, 1)
        qB, muB, quadB, _ = self.atom_model(batch.batch_atomic_B)
        qB = qB.reshape(-1, 1)

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
            mA_i = scatter_sum_compile(
                mA_ij, e_AA_source, dim_size=natomA, reduce="sum"
            )
            mB_i = scatter_sum_compile(
                mB_ij, e_BB_source, dim_size=natomB, reduce="sum"
            )

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
                mA_ij_dir, e_AA_source, dim_size=natomA, reduce="sum"
            )
            hB_dir = scatter_sum_compile(
                mB_ij_dir, e_BB_source, dim_size=natomB, reduce="sum"
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
        # EAB_sr = torch.cat(
        #     [
        #         self.readout_layer_elst(hAB),
        #         self.readout_layer_exch(hAB),
        #         self.readout_layer_indu(hAB),
        #         self.readout_layer_disp(hAB),
        #     ],
        #     dim=1,
        # )
        # EBA_sr = torch.cat(
        #     [
        #         self.readout_layer_elst(hBA),
        #         self.readout_layer_exch(hBA),
        #         self.readout_layer_indu(hBA),
        #         self.readout_layer_disp(hBA),
        #     ],
        #     dim=1,
        # )
        EAB_sr = self.readouts(hAB)
        EBA_sr = self.readouts(hBA)

        E_sr = EAB_sr + EBA_sr

        cutoff = (1.0 / (dR_sr**3)).unsqueeze(-1)
        E_sr *= cutoff
        # cutoff = torch.pow(torch.reciprocal(dR_sr), 3)
        # E_sr = torch.einsum('xy,x->xy', E_sr, cutoff)
        E_sr_dimer = scatter_sum_compile(E_sr, dimer_ind, dim_size=ndimer, reduce="add")

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

        E_elst_sr_dimer = scatter_sum_compile(
            E_elst_sr, dimer_ind, dim_size=ndimer, reduce="add"
        )
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
            E_elst_lr, dimer_ind_lr, dim_size=ndimer, reduce="add"
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
        if self.return_hidden_states:
            return (
                E_output,
                E_sr_dimer,
                E_elst_sr_dimer,
                E_elst_lr_dimer,
                hAB,
                hBA,
                cutoff,
            )
        return E_output, E_sr, E_elst_sr, E_elst_lr, hAB, hBA


class APNet2_AM_Model:
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
        ds_skip_compile=False,
        ds_num_devices=1,
        ds_datapoint_storage_n_objects=1000,
        ds_prebatched=False,
        ds_random_seed=42,
        print_lvl=0,
        ds_qcel_molecules=None,
        ds_energy_labels=None,
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
        self.atom_model = AtomMPNN()

        if atom_model_pre_trained_path:
            print(
                f"Loading pre-trained AtomMPNN model from {atom_model_pre_trained_path}"
            )
            checkpoint = torch.load(
                atom_model_pre_trained_path, map_location=device, weights_only=False
            )
            self.atom_model = AtomMPNN(
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
            print(
                f"Loading pre-trained APNet2_MPNN model from {pre_trained_model_path}"
            )
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = APNet2_AM_MPNN(
                atom_model=self.atom_model,
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
            self.model = APNet2_AM_MPNN(
                atom_model=self.atom_model,
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut_im=r_cut_im,
                r_cut=r_cut,
            )
        if n_rbf != self.model.n_rbf:
            print(f"Changing n_rbf from {self.model.n_rbf} to {n_rbf}")
            self.model.n_rbf = n_rbf
        if n_message != self.model.n_message:
            print(f"Changing n_message from {self.model.n_message} to {n_message}")
            self.model.n_message = n_message
        if n_neuron != self.model.n_neuron:
            print(f"Changing n_neuron from {self.model.n_neuron} to {n_neuron}")
            self.model.n_neuron = n_neuron
        if n_embed != self.model.n_embed:
            print(f"Changing n_embed from {self.model.n_embed} to {n_embed}")
            self.model.n_embed = n_embed
        if r_cut_im != self.model.r_cut_im:
            print(f"Changing r_cut_im from {self.model.r_cut_im} to {r_cut_im}")
            self.model.r_cut_im = r_cut_im
        if r_cut != self.model.r_cut:
            print(f"Changing r_cut from {self.model.r_cut} to {r_cut}")
            self.model.r_cut = r_cut

        self.device = device
        self.atom_model.to(device)
        self.model.to(device)

        ds_split_db = ap2_fused_module_dataset.is_split_db_config(
            self.ds_spec_type, ds_qcel_molecules
        )
        self.dataset = dataset
        if self.dataset is not None:
            ds_split_db = getattr(self.dataset, "split_db", ds_split_db)
        if not ignore_database_null and self.dataset is None and not ds_split_db:

            def setup_ds(fp=ds_force_reprocess):
                return ap2_fused_module_dataset(
                    root=ds_root,
                    r_cut=r_cut,
                    r_cut_im=r_cut_im,
                    spec_type=ds_spec_type,
                    max_size=ds_max_size,
                    force_reprocess=fp,
                    atom_model=self.atom_model,
                    # atom_model_path=atom_model_pre_trained_path,
                    atomic_batch_size=ds_atomic_batch_size,
                    num_devices=ds_num_devices,
                    skip_processed=ds_skip_process,
                    skip_compile=ds_skip_compile,
                    random_seed=ds_random_seed,
                    datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                    print_level=print_lvl,
                    qcel_molecules=ds_qcel_molecules,
                    energy_labels=ds_energy_labels,
                )

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset = self.dataset[:ds_max_size]
        elif not ignore_database_null and self.dataset is None and ds_split_db:
            print("Processing Split dataset...")
            if ds_qcel_molecules is None:
                ds_qcel_molecules = [None, None]
                ds_energy_labels = [None, None]

            def setup_ds(fp=ds_force_reprocess):
                return [
                    ap2_fused_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
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
                    ),
                    ap2_fused_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
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
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.model(batch)
        return

    def compile_model(self):
        """
        Compile the model for optimized execution with dynamic shape support and move it to the configured device.

        This enables PyTorch Dynamo's dynamic-shape configuration, compiles the model using torch.compile(dynamic=True), and ensures the model is placed on self.device.
        """
        self.model.to(self.device)
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
        torch._dynamo.config.capture_scalar_outputs = False
        # torch._dynamo.config.capture_scalar_outputs = True
        self.model = torch.compile(self.model, dynamic=True)
        return

    def set_all_weights_to_value(self, value: float):
        """
        Set every learnable parameter of the model to the given constant value.

        Parameters:
            value (float): Scalar to assign to all model parameters.

        Notes:
            Performs one forward pass on an example input to ensure lazy-initialized parameters are created before assignment.
        """
        batch = self.example_input()
        batch.to(self.device)
        self.model(batch)
        set_weights_to_value(self.model, value)
        return

    def set_pretrained_model(
        self, ap2_model_path=None, am_model_path=None, model_id=None
    ):
        """
        Load a pretrained APNet2_AM_MPNN checkpoint into this model instance.

        Parameters:
            ap2_model_path (str or pathlib.Path, optional): Path to a saved APNet2_AM_MPNN checkpoint file. Mutually exclusive with `model_id`.
            am_model_path: (ignored) Present for API compatibility but not used by this method.
            model_id (str or int, optional): If provided, selects a bundled model resource by id; overrides `ap2_model_path`.

        Returns:
            self: The current APNet2_AM_Model instance with weights loaded from the checkpoint.

        Raises:
            ValueError: If neither `ap2_model_path` nor `model_id` is provided.
        """
        if model_id is not None:
            ap2_model_path = resolve_pretrained_path(
                f"ap2-fused_ensemble/ap2_{model_id}.pt"
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
        r_cut=5.0,
        r_cut_im=8.0,
    ):
        dimer_batch = ap2_fused_collate_update_no_target(
            [
                qcel_dimer_to_fused_data(
                    mol, r_cut=r_cut, r_cut_im=r_cut_im, dimer_ind=n
                )
                for n, mol in enumerate(mols)
            ]
        )
        dimer_batch.to(self.device)
        return dimer_batch

    def set_return_hidden_states(self, value=True):
        self.model.return_hidden_states = value
        return self

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
            indA_to_atom.append(np.arange(size_A.item()))
            indB_to_atom.append(np.arange(size_B.item()))
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
            indA_to_atom.append(np.arange(size_A.item()))
            indB_to_atom.append(np.arange(size_B.item()))
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
    def predict_qcel_mols(
        self,
        mols,
        batch_size=1,
        r_cut=None,
        r_cut_im=None,
        verbose=False,
        return_pairs=False,
        return_elst=False,
    ):
        assert not (return_elst and return_pairs), (
            "return_elst and return_pairs are not compatible"
        )
        if r_cut is None:
            r_cut = self.model.r_cut
        if r_cut_im is None:
            r_cut_im = self.model.r_cut_im

        N = len(mols)
        predictions = np.zeros((N, 4))
        if return_pairs or return_elst:
            pairwise_energies = []
        if self.model.return_hidden_states:
            # need to capture output
            h_ABs, h_BAs, cutoffs, dimer_inds, ndimers = [], [], [], [], []
        # self.model.to(self.device)
        self.atom_model.to(self.device)
        for i in range(0, N, batch_size):
            upper_bound = min(i + batch_size, N)
            data = [
                qcel_dimer_to_fused_data(
                    dimer, r_cut=r_cut, r_cut_im=r_cut_im, dimer_ind=n
                )
                for n, dimer in enumerate(mols[i:upper_bound])
            ]
            valid_indices = [j for j, d in enumerate(data) if d is not None]
            all_indices = list(range(len(data)))
            if len(valid_indices) < len(data):
                if verbose:
                    print(
                        f"Skipping {len(data) - len(valid_indices)} invalid dimers in batch {i} to {upper_bound}"
                    )
                data = [data[j] for j in valid_indices]
            if len(data) == 0:
                for idx in all_indices:
                    predictions[i + idx] = np.array([np.nan, np.nan, np.nan, np.nan])
                    if return_pairs or return_elst:
                        pairwise_energies.append([])
                continue
            dimer_batch = ap2_fused_collate_update_no_target(data)
            dimer_batch.to(device=self.device)
            preds = self.model(dimer_batch)
            if self.model.return_hidden_states:
                E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA, cutoff = preds
                h_ABs.append(hAB)
                h_BAs.append(hBA)
                cutoffs.append(cutoff)
                dimer_inds.append(dimer_batch.dimer_ind)
                ndimers.append(dimer_batch.total_charge_A.size(0))
                cnt = 0
                for idx in all_indices:
                    if idx in valid_indices:
                        predictions[i + idx] = E_sr_dimer[cnt].cpu().numpy()
                        cnt += 1
                    else:
                        predictions[i + idx] = np.array(
                            [np.nan, np.nan, np.nan, np.nan]
                        )
            elif return_pairs:
                E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = preds
                v = self._assemble_pairs(
                    dimer_batch.cpu(),
                    E_sr_dimer.cpu(),
                    E_sr.cpu(),
                    E_elst_sr.cpu(),
                    E_elst_lr.cpu(),
                )
                cnt = 0
                for idx in all_indices:
                    if idx in valid_indices:
                        predictions[i + idx] = E_sr_dimer[cnt].cpu().numpy()
                        pairwise_energies.append(v[cnt])
                        cnt += 1
                    else:
                        predictions[i + idx] = np.array(
                            [np.nan, np.nan, np.nan, np.nan]
                        )
                        pairwise_energies.append([])
            elif return_elst:
                E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = preds
                v = self._assemble_mtp_pairs(
                    dimer_batch,
                    E_elst_sr,
                    E_elst_lr,
                )
                cnt = 0
                for idx in all_indices:
                    if idx in valid_indices:
                        predictions[i + idx] = E_sr_dimer[cnt].cpu().numpy()
                        pairwise_energies.append(v[cnt])
                        cnt += 1
                    else:
                        predictions[i + idx] = np.array(
                            [np.nan, np.nan, np.nan, np.nan]
                        )
                        pairwise_energies.append([])
            else:
                cnt = 0
                for idx in all_indices:
                    if idx in valid_indices:
                        predictions[i + idx] = preds[0][cnt].cpu().numpy()
                        cnt += 1
                    else:
                        predictions[i + idx] = np.array(
                            [np.nan, np.nan, np.nan, np.nan]
                        )
        if verbose:
            print(f"Predictions for {i} to {i + batch_size} out of {N}")
        if self.model.return_hidden_states:
            return predictions, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers
        if return_pairs or return_elst:
            return predictions, pairwise_energies
        return predictions

    def example_input(
        self,
        mol=None,
        r_cut=5.0,
        r_cut_im=8.0,
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
            [mol], batch_size=1, r_cut=r_cut, r_cut_im=r_cut_im
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
        self, dataloader, loss_fn, optimizer, rank_device, scheduler
    ):
        """
        Single-process training loop body.
        """
        self.model.train()
        comp_errors_t = []
        total_loss = 0.0
        for n, batch in enumerate(dataloader):
            # optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()
            batch = batch.to(rank_device, non_blocking=True)
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.model(batch)
            preds = E_sr_dimer.reshape(-1, 4)
            comp_errors = preds - batch.y
            batch_loss = (
                torch.mean(torch.square(comp_errors))
                if (loss_fn is None)
                else loss_fn(preds, batch.y)
            )
            batch_loss.backward()
            optimizer.step()
            # print(preds[0][0].item(), batch.y[0].numpy())
            # print(f"    Loss value: {batch_loss.item()}")
            total_loss += batch_loss.item()
            comp_errors_t.append(comp_errors.detach().cpu())
        if scheduler is not None:
            scheduler.step()

        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, 4)
        total_MAE_t = torch.mean(torch.abs(torch.sum(comp_errors_t, axis=1)))
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
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer, _, _, _, _, _ = self.model(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                comp_errors = preds - batch.y
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, batch.y)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, 4)
        total_MAE_t = torch.mean(torch.abs(torch.sum(comp_errors_t, axis=1)))
        elst_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 0]))
        exch_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 1]))
        indu_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 2]))
        disp_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 3]))
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    def __train_batches_single_proc_transfer(
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
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.model(batch)
            preds = E_sr_dimer.reshape(-1, 4)
            preds = torch.sum(preds, dim=1)
            comp_errors = preds - batch.y.squeeze(-1)
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

        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t))
        return total_loss, total_MAE_t

    # @torch.inference_mode()
    def __evaluate_batches_single_proc_transfer(self, dataloader, loss_fn, rank_device):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer, _, _, _, _, _ = self.model(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                preds = torch.sum(preds, dim=1)
                comp_errors = preds - batch.y.squeeze(-1)
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds.flatten(), batch.y.flatten())
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t))
        return total_loss, total_MAE_t

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
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.model(batch)
            preds = E_sr_dimer.reshape(-1, 4)
            comp_errors = preds - batch.y
            if loss_fn is None:
                batch_loss = torch.mean(torch.square(comp_errors))
            else:
                batch_loss = loss_fn(preds.flatten(), batch.y.flatten())

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
                E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.model(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                comp_errors = preds - batch.y
                if loss_fn is None:
                    batch_loss = torch.mean(torch.square(comp_errors))
                else:
                    batch_loss = loss_fn(preds.flatten(), batch.y.flatten())

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
        """
        Run a distributed-data-parallel (DDP) training loop for the model, evaluate on validation data, and save the best checkpoint to self.model_save_path.

        Performs an initial evaluation (pre-training), trains for n_epochs across the provided datasets using PyTorch DDP when world_size > 1, evaluates after each epoch, and saves the model checkpoint when validation loss improves. When using multiple processes, this method will initialize and clean up the process group. The method uses self.device to choose the target device and may move the model to/from CPU for checkpointing.

        Parameters:
            rank (int): Process rank in the distributed group (0 is the primary process).
            world_size (int): Total number of processes participating in DDP; if 1, runs single-process training.
            train_dataset (torch.utils.data.Dataset): Dataset used for training.
            test_dataset (torch.utils.data.Dataset): Dataset used for validation/evaluation.
            n_epochs (int): Number of training epochs to run.
            batch_size (int): Batch size for data loaders.
            lr (float): Initial learning rate for the optimizer.
            pin_memory (bool): If true, data loaders will use pinned memory.
            num_workers (int): Number of worker processes for data loading.
            lr_decay (float | None): Optional decay factor for the inverse-time LR scheduler; if None no scheduler is used.

        Side effects:
            - May call self.__setup(...) and self.__cleanup() to manage the process group when world_size > 1.
            - Moves the model to/from devices and may wrap it with torch.distributed.DDP.
            - Saves the best-performing model checkpoint to self.model_save_path when validation loss decreases.
        """
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
            first_pass_data = APNet2_fused_DataLoader(
                dataset=test_dataset[:batch_size],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=ap2_fused_collate_update,
            )
            for b in first_pass_data:
                b.to(rank_device)
                self.model(b)
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

        train_loader = APNet2_fused_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            collate_fn=ap2_fused_collate_update,
        )

        test_loader = APNet2_fused_DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=ap2_fused_collate_update,
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
                    f"  EPOCH: {epoch: 4d}({dt: < 7.2f} sec)  MAE: {total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f} {elst_MAE_t: > 7.3f}/{elst_MAE_v: < 7.3f} {exch_MAE_t: > 7.3f}/{exch_MAE_v: < 7.3f} {indu_MAE_t: > 7.3f}/{indu_MAE_v: < 7.3f} {disp_MAE_t: > 7.3f}/{disp_MAE_v: < 7.3f} {test_lowered}",
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
        transfer_learning=False,
        pretrain_test_loss=True,
    ):
        # (1) Compile Model
        """
        Train the model on a single process using provided train and test datasets.

        Performs an optional compile/warmup, constructs dataloaders and optimizer/scheduler, runs an initial evaluation, then executes the main training loop for n_epochs while tracking and saving the best model by test loss.

        Parameters:
            train_dataset: Dataset providing training fused-dimer batches.
            test_dataset: Dataset providing validation fused-dimer batches.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size for data loaders.
            lr (float): Initial learning rate for the optimizer.
            pin_memory (bool): Passed to DataLoader to pin memory.
            num_workers (int): Number of worker processes for data loading.
            lr_decay (float | None): Decay factor used to construct the learning-rate scheduler; if None no scheduler is used.
            skip_compile (bool): If True, skip torch.compile step.
            transfer_learning (bool): If True, use transfer-learning evaluation/training routines that aggregate component predictions before loss.
            pretrain_test_loss (bool): If True, initialize best-model selection using the pre-training test loss; if False, require strictly better test loss to replace the saved best model.

        """
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # scheduler = ModLambdaDecayLR(optimizer, lr_decay, lr) if lr_decay else None
        scheduler = (
            InverseTimeDecayLR(optimizer, lr, len(train_loader) * 2, lr_decay)
            if lr_decay
            else None
        )
        # criterion = None  # defaults to MSE
        criterion = torch.nn.MSELoss()

        # (4) Set eval functions
        if not transfer_learning:
            __evaluate_batch = self.__evaluate_batches_single_proc
            __train_batch = self.__train_batches_single_proc
            print(
                "                                       Total            Elst            Exch            Ind            Disp",
                flush=True,
            )
        else:
            __evaluate_batch = self.__evaluate_batches_single_proc_transfer
            __train_batch = self.__train_batches_single_proc_transfer
            print(
                "                                       Total",
                flush=True,
            )

        # (5) Evaluate once pre-training
        t0 = time.time()
        t_out = __evaluate_batch(train_loader, criterion, rank_device)
        v_out = __evaluate_batch(test_loader, criterion, rank_device)
        if not transfer_learning:
            train_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t = (
                t_out
            )
            test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
                v_out
            )
            print(
                f"  (Pre-training)({time.time() - t0: < 7.2f}s)  MAE: {
                    total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f} "
                f"{elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} "
                f"{indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} {disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f}",
                flush=True,
            )
        else:
            train_loss, total_MAE_t = t_out
            test_loss, total_MAE_v = v_out
            print(
                f"  (Pre-training)({time.time() - t0: < 7.2f}s)  MAE: {
                    total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f}",
                flush=True,
            )

        # (6) Main training loop
        if pretrain_test_loss:
            lowest_test_loss = test_loss
        else:
            lowest_test_loss = torch.tensor(float("inf"))
        # print(f"{lowest_test_loss=:.6f}")
        for epoch in range(n_epochs):
            t1 = time.time()
            t_out = __train_batch(
                train_loader, criterion, optimizer, rank_device, scheduler
            )
            v_out = __evaluate_batch(test_loader, criterion, rank_device)
            if not transfer_learning:
                (
                    train_loss,
                    total_MAE_t,
                    elst_MAE_t,
                    exch_MAE_t,
                    indu_MAE_t,
                    disp_MAE_t,
                ) = t_out
                (
                    test_loss,
                    total_MAE_v,
                    elst_MAE_v,
                    exch_MAE_v,
                    indu_MAE_v,
                    disp_MAE_v,
                ) = v_out
            else:
                train_loss, total_MAE_t = t_out
                test_loss, total_MAE_v = v_out

            # Track best model
            star_marker = " "
            # print(f"{test_loss=:.6f}")
            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                star_marker = "*"
                cpu_model = unwrap_model(self.model).to("cpu")
                best_model = deepcopy(cpu_model)
                if self.model_save_path:
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

            if not transfer_learning:
                print(
                    f"  EPOCH: {epoch:4d} ({time.time() - t1:<7.2f}s)  MAE: "
                    f"{total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} "
                    f"{exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} {indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} "
                    f"{disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f} {star_marker}",
                    flush=True,
                )
            else:
                print(
                    f"  EPOCH: {epoch:4d} ({time.time() - t1:<7.2f}s)  MAE: "
                    f"{total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {star_marker}",
                    flush=True,
                )
            if not self.device == "CPU":
                torch.cuda.empty_cache()
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
        lr_decay=None,
        random_seed=42,
        skip_compile=False,
        transfer_learning=False,
        pretrain_test_loss=True,
    ):
        """
        Train the APNet2_AM_Model on the provided dataset using the configured device and training settings.

        Parameters:
            dataset (optional): Dataset or [train_dataset, test_dataset] to use for training; if omitted the instance's dataset must be set beforehand.
            n_epochs (int): Number of training epochs.
            lr (float): Initial learning rate.
            split_percent (float): Fraction of dataset assigned to training when a single dataset is provided.
            model_path (str|None): Path where training results and the best model will be saved.
            shuffle (bool): Whether to shuffle datasets before splitting and batching.
            dataloader_num_workers (int): Number of worker processes for data loading.
            world_size (int): Number of processes for distributed (DDP) training; uses single-process training when 1.
            omp_num_threads_per_process (int): Value for OMP_NUM_THREADS for each training process.
            lr_decay (callable|None): Learning-rate scheduler factory or configuration passed to training; when None no LR scheduler is used.
            random_seed (int): Seed used for dataset shuffling and reproducible splits.
            skip_compile (bool): When True, skip model compilation/optimization steps prior to training.
            transfer_learning (bool): When True, use the transfer-learning training variant that sums sub-component predictions before loss computation.
            pretrain_test_loss (bool): Controls best-model tracking behavior when continuing from a pretrained state; affects initialization and updating of the saved best test loss.

        Raises:
            ValueError: If no dataset is provided or available on the instance.

        Returns:
            None
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
        print(f"Saving training results to...\n{self.model_save_path}")
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
            f"    Training on {len(train_dataset)} samples, Testing on {
                len(test_dataset)
            } samples"
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
        print(f"  {batch_size=}", flush=True)

        if self.device.type == "cuda":
            pin_memory = False
        else:
            pin_memory = False

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
                transfer_learning=transfer_learning,
                pretrain_test_loss=pretrain_test_loss,
            )
        return

    def freeze_parameters_except_readouts(self):
        """
        Freeze all model parameters except the AP2 model readout layers named with suffixes 'elst', 'exch', 'indu', or 'disp'.

        This sets requires_grad = False for every parameter not belonging to a readout layer whose top-level module name ends with one of the listed suffixes, and sets requires_grad = True for parameters in those readout layers.
        """
        for name, param in self.model.named_parameters():
            term = name.split(".")[0]
            if "readout" in name and term[-4:] in ["elst", "exch", "indu", "disp"]:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return

    def unfreeze_all_parameters(self):
        """
        Enable gradient updates for all parameters of the AP2 model.

        Sets `requires_grad = True` on every parameter in `self.model`, allowing all layers to be trained.
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        return
