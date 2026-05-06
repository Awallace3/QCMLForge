import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import warnings
import time
from ..AtomModels.ap2_atom_model import AtomMPNN, isolate_atomic_property_predictions
from .. import atomic_datasets
from ..hf_pretrained import resolve_pretrained_paths
from .. import pairwise_datasets
from .. import model_io
from ..pairwise_datasets import (
    apnet2_module_dataset,
    APNet2_DataLoader,
    apnet2_collate_update,
    apnet2_collate_update_prebatched,
    pairwise_edges,
    pairwise_edges_im,
    qcel_dimer_to_pyg_data,
)
from .. import constants
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import qcelemental as qcel
from copy import deepcopy
from apnet_pt.torch_util import set_weights_to_value
from apnet_pt.util import scatter_sum_compile


def inverse_time_decay(step, initial_lr, decay_steps, decay_rate, staircase=True):
    """
    Compute an inverse-time decayed learning rate.

    Parameters:
        step (int or float): Current training step or epoch.
        initial_lr (float): Initial learning rate before decay.
        decay_steps (int or float): Normalization factor for the step (controls decay speed).
        decay_rate (float): Multiplicative decay factor.
        staircase (bool): If True, use discrete (staircase) decay by applying floor to step/decay_steps.

    Returns:
        (float): The learning rate after applying inverse time decay.
    """
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


def compute_component_mse_loss(preds, labels, loss_fn=None, include_total_mse=False):
    """Compute component MSE and optionally add a total-energy MSE term."""
    comp_errors = preds - labels
    batch_loss = (
        torch.mean(torch.square(comp_errors))
        if loss_fn is None
        else loss_fn(preds, labels)
    )
    if include_total_mse:
        total_preds = torch.sum(preds, dim=1)
        total_labels = torch.sum(labels, dim=1)
        batch_loss = batch_loss + torch.mean(torch.square(total_preds - total_labels))
    return batch_loss, comp_errors


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


class APNet2_MPNN(nn.Module):
    def __init__(
        self,
        # atom_model,
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
        # self.atom_model = atom_model

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

    def get_config(self) -> dict:
        """
        Return the configuration dictionary for this model.

        Returns
        -------
        dict
            Dictionary containing all hyperparameters needed to reconstruct
            this model architecture.
        """
        return {
            "n_message": self.n_message,
            "n_rbf": self.n_rbf,
            "n_neuron": self.n_neuron,
            "n_embed": self.n_embed,
            "r_cut_im": self.r_cut_im,
            "r_cut": self.r_cut,
        }

    def get_model_info(self):
        """Return a ModelInfo describing this module for print_model_tree."""
        from apnet_pt.model_print import ModelInfo, _safe_numel

        n_total = sum(_safe_numel(p) for p in self.parameters())
        n_train = sum(_safe_numel(p) for p in self.parameters() if p.requires_grad)
        return ModelInfo(
            name="APNet2_MPNN",
            role="Predicts SAPT interaction components from monomer multipoles and pair features",
            inputs=["h_A", "h_B", "q_A", "mu_A", "Q_A", "q_B", "mu_B", "Q_B"],
            outputs=["E_elst", "E_exch", "E_indu", "E_disp"],
            frozen=(n_train == 0),
            n_params=n_train,
            n_params_total=n_total,
            n_calls=1,
            call_note=(
                "intra-monomer update layers run independently on monomer A and B "
                "with shared weights before pairwise readout"
            ),
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
        hlistA,
        # monomer B properties
        qB,
        muB,
        quadB,
        hlistB,
    ):
        # counts
        # natomA = ZA.size(0)
        # natomB = ZB.size(0)
        # ndimer = total_charge_A.size(0)
        """
        Compute atom-pair SAPT component predictions and multipole electrostatic energies for a batch of dimers given per-atom features, coordinates, and edge connectivity.

        Parameters:
            ZA: Tensor of atomic numbers for monomer A.
            RA: Tensor of Cartesian coordinates for monomer A.
            ZB: Tensor of atomic numbers for monomer B.
            RB: Tensor of Cartesian coordinates for monomer B.
            e_ABsr_source, e_ABsr_target: Edge index arrays (source, target) for short-range intermolecular atom pairs (A->B).
            dimer_ind: Mapping from short-range intermolecular edges to dimer indices.
            e_ABlr_source, e_ABlr_target: Edge index arrays for long-range intermolecular atom pairs.
            dimer_ind_lr: Mapping from long-range intermolecular edges to dimer indices.
            e_AA_source, e_AA_target: Edge index arrays for intramonomer A connectivity.
            e_BB_source, e_BB_target: Edge index arrays for intramonomer B connectivity.
            total_charge_A, total_charge_B: Per-dimer total charges for monomer A and B.
            qA, qB: Per-atom monopoles (charges) for monomers A and B.
            muA, muB: Per-atom dipole vectors for monomers A and B.
            quadA, quadB: Per-atom quadrupole tensors (or flattened representation) for monomers A and B.
            hlistA, hlistB: Optional per-atom auxiliary features for monomers A and B (precomputed atom-model outputs).

        Returns:
            If self.return_hidden_states is True:
                (E_output, E_sr_dimer, E_elst_sr_dimer, E_elst_lr_dimer, hAB, hBA, cutoff)
                - E_output: Per-dimer aggregated energy array combining short-range and multipole electrostatics.
                - E_sr_dimer: Per-dimer summed short-range SAPT component energies.
                - E_elst_sr_dimer: Per-dimer summed short-range multipole electrostatic contributions (expanded and padded).
                - E_elst_lr_dimer: Per-dimer summed long-range multipole electrostatic contributions (expanded and padded).
                - hAB, hBA: Per-edge atom-pair feature tensors for A->B and B->A used by readouts.
                - cutoff: Per-edge short-range distance-based cutoff factors applied to short-range energies.
            Otherwise:
                (E_output, E_sr, E_elst_sr, E_elst_lr, hAB, hBA)
                - E_output: Per-dimer aggregated energy array combining short-range and multipole electrostatics.
                - E_sr: Per-edge short-range SAPT component values before per-dimer aggregation.
                - E_elst_sr: Per-edge short-range multipole electrostatic contributions.
                - E_elst_lr: Per-edge long-range multipole electrostatic contributions.
                - hAB, hBA: Per-edge atom-pair feature tensors for A->B and B->A.

        Notes:
            - The function performs intramonomer message passing to produce invariant and directional hidden states, assembles atom-pair features, predicts per-pair SAPT components via readout networks, applies a distance-based short-range cutoff, aggregates per-pair energies to per-dimer energies, and computes multipole electrostatic contributions using supplied multipole moments.
            - Shapes of returned tensors depend on the number of dimers, number of intermolecular edges, and model configuration.
        """
        natomA = ZA.size(0)
        natomB = ZB.size(0)
        ndimer = total_charge_A.size(0)

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
                mA_ij, e_AA_source, reduce="sum", dim_size=natomA
            )
            mB_i = scatter_sum_compile(
                mB_ij, e_BB_source, reduce="sum", dim_size=natomB
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
                mA_ij_dir, e_AA_source, reduce="sum", dim_size=natomA
            )
            hB_dir = scatter_sum_compile(
                mB_ij_dir, e_BB_source, reduce="sum", dim_size=natomB
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
        # cutoff = torch.pow(torch.reciprocal(dR_sr), 3)
        # E_sr = torch.einsum('xy,x->xy', E_sr, cutoff)
        E_sr_dimer = scatter_sum_compile(E_sr, dimer_ind, reduce="add", dim_size=ndimer)

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
            E_elst_sr, dimer_ind, reduce="add", dim_size=ndimer
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
            E_elst_lr, dimer_ind_lr, reduce="add", dim_size=ndimer
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
        # E_sr_dimer[:, 0] = 0.0
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


class APNet2Model:
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

        # Track whether we loaded the atom_model from an embedded submodel
        atom_model_loaded_from_embed = False

        if pre_trained_model_path:
            print(
                f"Loading pre-trained APNet2_MPNN model from {pre_trained_model_path}"
            )
            checkpoint = model_io.load_checkpoint(
                pre_trained_model_path, map_location=device
            )
            version = model_io.get_checkpoint_version(checkpoint)

            # Check for embedded atom_model in v2 checkpoint
            if version >= 2 and model_io.has_embedded_submodel(
                checkpoint, "atom_model"
            ):
                if atom_model_pre_trained_path:
                    model_io.warn_submodel_override(
                        "atom_model",
                        embedded_type="AtomMPNN",
                        external_path=atom_model_pre_trained_path,
                    )
                # Load embedded atom_model
                am_checkpoint = model_io.get_submodel_checkpoint(
                    checkpoint, "atom_model"
                )
                am_config = am_checkpoint["config"]
                self.atom_model = AtomMPNN(
                    n_message=am_config["n_message"],
                    n_rbf=am_config["n_rbf"],
                    n_neuron=am_config["n_neuron"],
                    n_embed=am_config["n_embed"],
                    r_cut=am_config["r_cut"],
                )
                am_state_dict = model_io.load_state_dict_from_checkpoint(am_checkpoint)
                self.atom_model.load_state_dict(am_state_dict)
                atom_model_loaded_from_embed = True
                print("Loaded embedded AtomMPNN from checkpoint")

            # Load the main APNet2 model
            config = model_io.load_config_from_checkpoint(checkpoint) or {}
            self.model = APNet2_MPNN(
                n_message=config.get("n_message", n_message),
                n_rbf=config.get("n_rbf", n_rbf),
                n_neuron=config.get("n_neuron", n_neuron),
                n_embed=config.get("n_embed", n_embed),
                r_cut_im=config.get("r_cut_im", r_cut_im),
                r_cut=config.get("r_cut", r_cut),
            )
            state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
            self.model.load_state_dict(state_dict)
        else:
            self.model = APNet2_MPNN(
                # atom_model=self.atom_model,
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut_im=r_cut_im,
                r_cut=r_cut,
            )

        # Load atom_model from external path if not already loaded from embedded
        if not atom_model_loaded_from_embed:
            if atom_model_pre_trained_path:
                print(
                    f"Loading pre-trained AtomMPNN model from {atom_model_pre_trained_path}"
                )
                am_checkpoint = model_io.load_checkpoint(
                    atom_model_pre_trained_path, map_location=device
                )
                am_config = model_io.load_config_from_checkpoint(am_checkpoint)
                if am_config is None:
                    am_config = am_checkpoint.get("config", {})
                self.atom_model = AtomMPNN(
                    n_message=am_config["n_message"],
                    n_rbf=am_config["n_rbf"],
                    n_neuron=am_config["n_neuron"],
                    n_embed=am_config["n_embed"],
                    r_cut=am_config["r_cut"],
                )
                am_state_dict = model_io.load_state_dict_from_checkpoint(am_checkpoint)
                self.atom_model.load_state_dict(am_state_dict)
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

        ds_split_db = apnet2_module_dataset.is_split_db_config(
            self.ds_spec_type, ds_qcel_molecules
        )
        self.dataset = dataset
        if self.dataset is not None:
            ds_split_db = getattr(self.dataset, "split_db", ds_split_db)
        if not ignore_database_null and self.dataset is None and not ds_split_db:

            def setup_ds(fp=ds_force_reprocess):
                return apnet2_module_dataset(
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
                    prebatched=ds_prebatched,
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
                    apnet2_module_dataset(
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
                        prebatched=ds_prebatched,
                        print_level=print_lvl,
                        qcel_molecules=ds_qcel_molecules[0],
                        qcel_molecules_raw=ds_qcel_molecules,
                        energy_labels=ds_energy_labels[0],
                    ),
                    apnet2_module_dataset(
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
                        prebatched=ds_prebatched,
                        print_level=print_lvl,
                        qcel_molecules=ds_qcel_molecules[1],
                        qcel_molecules_raw=ds_qcel_molecules,
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
        self.prebatched = ds_prebatched
        return

    def get_model_info(self):
        """Return a ModelInfo tree for the APNet2 harness."""
        from apnet_pt.model_print import ModelInfo, get_model_info

        atom_info = get_model_info(self.atom_model)
        atom_info.n_calls = 2
        atom_info.call_note = (
            "run separately for monomer A and monomer B (shared weights)"
        )
        mpnn_info = get_model_info(self.model)
        children = [atom_info, mpnn_info]
        return ModelInfo(
            name="APNet2Model",
            frozen=all(child.frozen for child in children),
            n_params=sum(child.n_params for child in children),
            n_params_total=sum(child.n_params_total for child in children),
            children=children,
        )

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

    def set_all_weights_to_value(self, value: float):
        """
        Sets the weights of the model to a constant value for debugging.
        """
        batch = self.example_input()
        batch.to(self.device)
        self.model(**batch)
        set_weights_to_value(self.model, value)
        return

    def set_pretrained_model(
        self, ap2_model_path=None, am_model_path=None, model_id=None
    ):
        """
        Load pretrained weights for the APNet2 and AtomMPNN models.

        For v2 checkpoints with embedded atom_model, the embedded submodel will
        be used and am_model_path will be ignored (with a warning).

        Parameters
        ----------
        ap2_model_path : str or Path, optional
            Path to APNet2 checkpoint file.
        am_model_path : str or Path, optional
            Path to AtomMPNN checkpoint file. Ignored if ap2_model_path is a v2
            checkpoint with embedded atom_model.
        model_id : int, optional
            If provided, loads from bundled ensemble models (am_{model_id}.pt
            and ap2_{model_id}.pt).

        Returns
        -------
        self
            Returns self for method chaining.
        """
        if model_id is not None:
            model_paths = resolve_pretrained_paths(
                [
                    f"am_ensemble/am_{model_id}.pt",
                    f"ap2_ensemble/ap2_{model_id}.pt",
                ]
            )
            am_model_path = model_paths[f"am_ensemble/am_{model_id}.pt"]
            ap2_model_path = model_paths[f"ap2_ensemble/ap2_{model_id}.pt"]
        elif ap2_model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided.")

        # Load APNet2 checkpoint
        checkpoint = model_io.load_checkpoint(ap2_model_path, map_location=self.device)
        version = model_io.get_checkpoint_version(checkpoint)

        # Check for embedded atom_model in v2 checkpoint
        atom_model_loaded_from_embed = False
        if version >= 2 and model_io.has_embedded_submodel(checkpoint, "atom_model"):
            if am_model_path is not None:
                model_io.warn_submodel_override(
                    "atom_model",
                    embedded_type="AtomMPNN",
                    external_path=str(am_model_path),
                )
            # Load embedded atom_model
            am_checkpoint = model_io.get_submodel_checkpoint(checkpoint, "atom_model")
            am_config = am_checkpoint["config"]
            self.atom_model = AtomMPNN(
                n_message=am_config["n_message"],
                n_rbf=am_config["n_rbf"],
                n_neuron=am_config["n_neuron"],
                n_embed=am_config["n_embed"],
                r_cut=am_config["r_cut"],
            )
            am_state_dict = model_io.load_state_dict_from_checkpoint(am_checkpoint)
            self.atom_model.load_state_dict(am_state_dict)
            self.atom_model.to(self.device)
            atom_model_loaded_from_embed = True
            print("Loaded embedded AtomMPNN from checkpoint")

        # Load APNet2 weights
        ap2_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
        self.model.load_state_dict(ap2_state_dict)

        # Load external atom_model if not loaded from embedded
        if not atom_model_loaded_from_embed and am_model_path is not None:
            am_checkpoint = model_io.load_checkpoint(
                am_model_path, map_location=self.device
            )
            am_state_dict = model_io.load_state_dict_from_checkpoint(am_checkpoint)
            self.atom_model.load_state_dict(am_state_dict)

        return self

    def _create_checkpoint(
        self,
        model: nn.Module = None,
        atom_model: nn.Module = None,
        embed_atom_model: bool = True,
        metadata: dict = None,
    ) -> dict:
        """
        Create a v2 checkpoint dictionary for the APNet2 model.

        Parameters
        ----------
        model : nn.Module, optional
            The APNet2_MPNN model to checkpoint. If None, uses self.model.
        atom_model : nn.Module, optional
            The AtomMPNN model to embed. If None, uses self.atom_model.
        embed_atom_model : bool, default True
            Whether to embed the atom_model as a submodel in the checkpoint.
        metadata : dict, optional
            Additional metadata to include in the checkpoint.

        Returns
        -------
        dict
            A v2 checkpoint dictionary ready for saving.
        """
        if model is None:
            model = self.model
        if atom_model is None:
            atom_model = self.atom_model

        # Unwrap DDP if needed
        model = model_io.unwrap_model(model)
        atom_model = model_io.unwrap_model(atom_model)

        # Get model config
        if hasattr(model, "get_config"):
            model_config = model.get_config()
        else:
            model_config = {
                "n_message": getattr(model, "n_message", 3),
                "n_rbf": getattr(model, "n_rbf", 8),
                "n_neuron": getattr(model, "n_neuron", 128),
                "n_embed": getattr(model, "n_embed", 8),
                "r_cut_im": getattr(model, "r_cut_im", 8.0),
                "r_cut": getattr(model, "r_cut", 5.0),
            }

        # Create submodels dict
        submodels = {}
        if embed_atom_model and atom_model is not None:
            if hasattr(atom_model, "get_config"):
                atom_config = atom_model.get_config()
            else:
                atom_config = {
                    "n_message": getattr(atom_model, "n_message", 3),
                    "n_rbf": getattr(atom_model, "n_rbf", 8),
                    "n_neuron": getattr(atom_model, "n_neuron", 128),
                    "n_embed": getattr(atom_model, "n_embed", 8),
                    "r_cut": getattr(atom_model, "r_cut", 5.0),
                }
            submodels["atom_model"] = model_io.create_submodel_checkpoint(
                model=atom_model,
                config=atom_config,
                model_type="AtomMPNN",
            )

        return model_io.create_checkpoint(
            model=model,
            config=model_config,
            model_type="APNet2_MPNN",
            submodels=submodels if submodels else None,
            metadata=metadata,
        )

    def save_model(
        self,
        path: str,
        embed_atom_model: bool = True,
        metadata: dict = None,
    ) -> None:
        """
        Save the APNet2 model to a v2 checkpoint file.

        Parameters
        ----------
        path : str
            Path where the checkpoint will be saved.
        embed_atom_model : bool, default True
            Whether to embed the atom_model in the checkpoint. If True, the
            checkpoint is self-contained and doesn't require a separate
            atom_model file.
        metadata : dict, optional
            Additional metadata to include in the checkpoint.
        """
        checkpoint = self._create_checkpoint(
            embed_atom_model=embed_atom_model,
            metadata=metadata,
        )
        model_io.save_checkpoint(checkpoint, path)
        print(f"Saved APNet2 model to {path}")

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
            hlistA=batch.hlistA,
            qB=batch.qB,
            muB=batch.muB,
            quadB=batch.quadB,
            hlistB=batch.hlistB,
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
            batch_A.to(self.device)
            batch_B.to(self.device)
            with torch.no_grad():
                am_out_A = self.atom_model(batch_A)
                am_out_B = self.atom_model(batch_B)
                qAs, muAs, quadAs, hlistAs = isolate_atomic_property_predictions(
                    batch_A, am_out_A
                )
                qBs, muBs, quadBs, hlistBs = isolate_atomic_property_predictions(
                    batch_B, am_out_B
                )
                if len(batch_A.total_charge.size()) == 0:
                    batch_A.total_charge = batch_A.total_charge.unsqueeze(0)
                if len(batch_B.total_charge.size()) == 0:
                    batch_B.total_charge = batch_B.total_charge.unsqueeze(0)
                dimer_ls = []
                for j in range(len(batch_mol_data)):
                    qA, muA, quadA, hlistA = qAs[j], muAs[j], quadAs[j], hlistAs[j]
                    qB, muB, quadB, hlistB = qBs[j], muBs[j], quadBs[j], hlistBs[j]
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                    e_AA_source, e_AA_target = pairwise_edges(data_A[j].R, r_cut)
                    e_BB_source, e_BB_target = pairwise_edges(data_B[j].R, r_cut)
                    e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = (
                        pairwise_edges_im(data_A[j].R, data_B[j].R, r_cut_im)
                    )
                    dimer_ind = torch.ones((1), dtype=torch.long) * 0
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
                        hlistA=hlistA,
                        # monomer B properties
                        qB=qB,
                        muB=muB,
                        quadB=quadB,
                        hlistB=hlistB,
                    )
                    dimer_ls.append(data)
                dimer_batch = pairwise_datasets.apnet2_collate_update_no_target(
                    dimer_ls
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
        """
        Predict per-dimer SAPT and related energies for a list of QCEngine dimer molecules.

        Parameters:
                mols (Sequence): Iterable of QCEngine dimer objects (each convertible via qcel_dimer_to_pyg_data).
                batch_size (int): Number of dimers processed per forward pass.
                r_cut (float | None): Intramonomer cutoff distance; defaults to self.model.r_cut when None.
                r_cut_im (float | None): Intermolecular cutoff distance; defaults to self.model.r_cut_im when None.
                verbose (bool): If True, prints progress messages for each processed batch.
                return_pairs (bool): If True, also return assembled per-pair short-range and long-range energy contributions.
                return_elst (bool): If True, also return assembled multipole electrostatic pair contributions. Mutually exclusive with return_pairs.

        Returns:
                When self.model.return_hidden_states is True:
                        tuple: (predictions, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers)
                                predictions (ndarray): (N, 4) array of per-dimer predicted SAPT component energies.
                                h_ABs, h_BAs (list): per-batch hidden-state tensors for A->B and B->A.
                                cutoffs (list): per-batch cutoff masks or values returned by the model.
                                dimer_inds (list): per-batch dimer index tensors.
                                ndimers (list): per-batch counts of dimers.
                When return_pairs or return_elst is True (and hidden states are not returned):
                        tuple: (predictions, pairwise_energies)
                                predictions (ndarray): (N, 4) array of per-dimer predicted SAPT component energies.
                                pairwise_energies (list): assembled per-pair energy records (format produced by _assemble_pairs or _assemble_mtp_pairs).
                Otherwise:
                        ndarray: (N, 4) array of per-dimer predicted SAPT component energies.

        Notes:
                - return_elst and return_pairs are mutually exclusive; an assertion is raised if both are True.
                - Inputs are processed in batches; atomic properties are produced by self.atom_model and assembled into dimer Data objects before evaluation.
        """
        assert not (return_elst and return_pairs), (
            "return_elst and return_pairs are not compatible"
        )
        if r_cut is None:
            r_cut = self.model.r_cut
        if r_cut_im is None:
            r_cut_im = self.model.r_cut_im

        mol_data = [[*qcel_dimer_to_pyg_data(mol)] for mol in mols]
        predictions = np.zeros((len(mol_data), 4))
        if return_pairs or return_elst:
            pairwise_energies = []
        if self.model.return_hidden_states:
            # need to capture output
            h_ABs, h_BAs, cutoffs, dimer_inds, ndimers = [], [], [], [], []
        # self.model.to(self.device)
        self.atom_model.to(self.device)
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            data_A = [d[0] for d in batch_mol_data]
            data_B = [d[1] for d in batch_mol_data]
            batch_A = atomic_datasets.atomic_collate_update_no_target(data_A)
            batch_B = atomic_datasets.atomic_collate_update_no_target(data_B)
            with torch.no_grad():
                batch_A.to(self.device)
                am_out_A = self.atom_model(batch_A)
                batch_B.to(self.device)
                am_out_B = self.atom_model(batch_B)
                qAs, muAs, quadAs, hlistAs = isolate_atomic_property_predictions(
                    batch_A, am_out_A
                )
                qBs, muBs, quadBs, hlistBs = isolate_atomic_property_predictions(
                    batch_B, am_out_B
                )
                if len(batch_A.total_charge.size()) == 0:
                    batch_A.total_charge = batch_A.total_charge.unsqueeze(0)
                if len(batch_B.total_charge.size()) == 0:
                    batch_B.total_charge = batch_B.total_charge.unsqueeze(0)
                dimer_ls = []
                for j in range(len(batch_mol_data)):
                    qA, muA, quadA, hlistA = qAs[j], muAs[j], quadAs[j], hlistAs[j]
                    qB, muB, quadB, hlistB = qBs[j], muBs[j], quadBs[j], hlistBs[j]
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                        e_AA_source, e_AA_target = pairwise_edges(data_A[j].R, r_cut)
                        e_BB_source, e_BB_target = pairwise_edges(data_B[j].R, r_cut)
                        e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = (
                            pairwise_edges_im(data_A[j].R, data_B[j].R, r_cut_im)
                        )
                        dimer_ind = torch.ones((1), dtype=torch.long) * 0
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
                            hlistA=hlistA,
                            # monomer B properties
                            qB=qB,
                            muB=muB,
                            quadB=quadB,
                            hlistB=hlistB,
                        )
                        dimer_ls.append(data)
                dimer_batch = (
                    pairwise_datasets.apnet2_collate_update_no_target_monomer_indices(
                        dimer_ls
                    )
                )
                dimer_batch.to(device=self.device)
                preds = self.eval_fn(dimer_batch)
                if self.model.return_hidden_states:
                    E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA, cutoff = preds
                    h_ABs.append(hAB)
                    h_BAs.append(hBA)
                    cutoffs.append(cutoff)
                    dimer_inds.append(dimer_batch.dimer_ind)
                    ndimers.append(
                        torch.tensor(
                            dimer_batch.total_charge_A.size(0), dtype=torch.long
                        )
                    )
                    predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
                elif return_pairs:
                    E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = preds
                    predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
                    pairwise_energies.extend(
                        self._assemble_pairs(
                            dimer_batch.cpu(),
                            E_sr_dimer.cpu(),
                            E_sr.cpu(),
                            E_elst_sr.cpu(),
                            E_elst_lr.cpu(),
                        )
                    )
                elif return_elst:
                    E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = preds
                    predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
                    pairwise_energies.extend(
                        self._assemble_mtp_pairs(
                            dimer_batch,
                            E_elst_sr,
                            E_elst_lr,
                        )
                    )
                else:
                    predictions[i : i + batch_size] = preds[0].cpu().numpy()
            if verbose:
                print(f"Predictions for {i} to {i + batch_size} out of {len(mol_data)}")
        if self.model.return_hidden_states:
            return predictions, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers
        if return_pairs or return_elst:
            return predictions, pairwise_energies
        return predictions

    @torch.inference_mode()
    def predict_qcel_mols_indexing(
        self,
        mols,
        batch_size=1,
        r_cut=None,
        r_cut_im=None,
        verbose=False,
        return_pairs=False,
        return_elst=False,
    ):
        """
        Generate predictions for a list of QCEL dimer molecules using an indexing-based batching workflow.

        Parameters:
            mols (Sequence): Iterable of QCEL dimer molecule objects (each convertible by qcel_dimer_to_pyg_data).
            batch_size (int): Number of dimers processed per atom-model batching step.
            r_cut (float | None): Short-range cutoff distance; defaults to self.model.r_cut when None.
            r_cut_im (float | None): Inter-monomer cutoff distance; defaults to self.model.r_cut_im when None.
            verbose (bool): If True, print progress messages for each processed batch.
            return_pairs (bool): If True, also return assembled pairwise SAPT component energies for each dimer.
            return_elst (bool): If True, also return assembled multipole electrostatic pair energies for each dimer.
                Note: return_pairs and return_elst are mutually exclusive (an assertion enforces this).

        Returns:
            If self.model.return_hidden_states is True:
                (predictions, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers)
                - predictions (ndarray, shape (N, 4)): Per-dimer predicted energy components (N = number of input dimers).
                - h_ABs, h_BAs (list): Lists of hidden-state tensors for A->B and B->A outputs per batch.
                - cutoffs (list): Cutoff values returned per batch.
                - dimer_inds (list): Batched dimer index tensors.
                - ndimers (list): Number of dimers per batch as tensors.
            Else if return_pairs is True:
                (predictions, pairwise_energies)
                - predictions (ndarray, shape (N, 4)): Per-dimer predicted energy components.
                - pairwise_energies (list): Assembled per-pair SAPT component entries produced by _assemble_pairs.
            Else if return_elst is True:
                (predictions, pairwise_energies)
                - predictions (ndarray, shape (N, 4)): Per-dimer predicted energy components.
                - pairwise_energies (list): Assembled multipole electrostatic pair entries produced by _assemble_mtp_pairs.
            Else:
                predictions (ndarray, shape (N, 4)): Per-dimer predicted energy components.

        """
        assert not (return_elst and return_pairs), (
            "return_elst and return_pairs are not compatible"
        )
        if r_cut is None:
            r_cut = self.model.r_cut
        if r_cut_im is None:
            r_cut_im = self.model.r_cut_im

        mol_data = [[*qcel_dimer_to_pyg_data(mol)] for mol in mols]
        predictions = np.zeros((len(mol_data), 4))
        if return_pairs or return_elst:
            pairwise_energies = []
        if self.model.return_hidden_states:
            # need to capture output
            h_ABs, h_BAs, cutoffs, dimer_inds, ndimers, e_ABsr_source, e_ABsr_target = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
        # self.model.to(self.device)
        self.atom_model.to(self.device)
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            data_A = [d[0] for d in batch_mol_data]
            data_B = [d[1] for d in batch_mol_data]
            batch_A = atomic_datasets.atomic_collate_update_no_target(data_A)
            batch_B = atomic_datasets.atomic_collate_update_no_target(data_B)
            with torch.no_grad():
                batch_A.to(self.device)
                am_out_A = self.atom_model(batch_A)
                batch_B.to(self.device)
                am_out_B = self.atom_model(batch_B)
                qAs, muAs, quadAs, hlistAs = isolate_atomic_property_predictions(
                    batch_A, am_out_A
                )
                qBs, muBs, quadBs, hlistBs = isolate_atomic_property_predictions(
                    batch_B, am_out_B
                )
                if len(batch_A.total_charge.size()) == 0:
                    batch_A.total_charge = batch_A.total_charge.unsqueeze(0)
                if len(batch_B.total_charge.size()) == 0:
                    batch_B.total_charge = batch_B.total_charge.unsqueeze(0)
                dimer_ls = []
                for j in range(len(batch_mol_data)):
                    qA, muA, quadA, hlistA = qAs[j], muAs[j], quadAs[j], hlistAs[j]
                    qB, muB, quadB, hlistB = qBs[j], muBs[j], quadBs[j], hlistBs[j]
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                        e_AA_source, e_AA_target = pairwise_edges(data_A[j].R, r_cut)
                        e_BB_source, e_BB_target = pairwise_edges(data_B[j].R, r_cut)
                        e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = (
                            pairwise_edges_im(data_A[j].R, data_B[j].R, r_cut_im)
                        )
                        dimer_ind = torch.ones((1), dtype=torch.long) * 0
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
                            hlistA=hlistA,
                            # monomer B properties
                            qB=qB,
                            muB=muB,
                            quadB=quadB,
                            hlistB=hlistB,
                        )
                        dimer_ls.append(data)
                dimer_batch = (
                    pairwise_datasets.apnet2_collate_update_no_target_monomer_indices(
                        dimer_ls
                    )
                )
                dimer_batch.to(device=self.device)
                preds = self.eval_fn(dimer_batch)
                if self.model.return_hidden_states:
                    E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA, cutoff = preds
                    h_ABs.append(hAB)
                    h_BAs.append(hBA)
                    cutoffs.append(cutoff)
                    dimer_inds.append(dimer_batch.dimer_ind)
                    ndimers.append(
                        torch.tensor(
                            dimer_batch.total_charge_A.size(0), dtype=torch.long
                        )
                    )
                    predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
                elif return_pairs:
                    E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = preds
                    predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
                    pairwise_energies.extend(
                        self._assemble_pairs(
                            dimer_batch.cpu(),
                            E_sr_dimer.cpu(),
                            E_sr.cpu(),
                            E_elst_sr.cpu(),
                            E_elst_lr.cpu(),
                        )
                    )
                elif return_elst:
                    E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = preds
                    predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
                    pairwise_energies.extend(
                        self._assemble_mtp_pairs(
                            dimer_batch,
                            E_elst_sr,
                            E_elst_lr,
                        )
                    )
                else:
                    predictions[i : i + batch_size] = preds[0].cpu().numpy()
            if verbose:
                print(f"Predictions for {i} to {i + batch_size} out of {len(mol_data)}")
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
        """
        Create a single-batch example input for the model from a QCEngine/QCElemental molecule.

        Parameters:
            mol (qcel.models.Molecule | None): Molecule to build the example from. If None, a small default water-like dimer is used.
            r_cut (float): Short-range cutoff distance (angstrom) used when assembling pairwise inputs.
            r_cut_im (float): Intramonomer cutoff distance (angstrom) used for internal-distance encodings.

        Returns:
            A collated batch object formatted for the APNet2 model's evaluation/prediction methods, containing atom features, coordinates, and pairwise assembly for the specified cutoffs.
        """
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
        self,
        dataloader,
        loss_fn,
        optimizer,
        rank_device,
        scheduler,
        include_total_mse=False,
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
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
            preds = E_sr_dimer.reshape(-1, 4)
            batch_loss, comp_errors = compute_component_mse_loss(
                preds,
                batch.y,
                loss_fn=loss_fn,
                include_total_mse=include_total_mse,
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
    def __evaluate_batches_single_proc(
        self, dataloader, loss_fn, rank_device, include_total_mse=False
    ):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer, _, _, _, _, _ = self.eval_fn(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                batch_loss, comp_errors = compute_component_mse_loss(
                    preds,
                    batch.y,
                    loss_fn=loss_fn,
                    include_total_mse=include_total_mse,
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
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
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
                E_sr_dimer, _, _, _, _, _ = self.eval_fn(batch)
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
        self,
        rank,
        dataloader,
        loss_fn,
        optimizer,
        rank_device,
        scheduler,
        include_total_mse=False,
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
            batch_loss, comp_errors = compute_component_mse_loss(
                preds,
                batch.y,
                loss_fn=loss_fn,
                include_total_mse=include_total_mse,
            )

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
    def __evaluate_batches(
        self, rank, dataloader, loss_fn, rank_device, include_total_mse=False
    ):
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
                batch_loss, comp_errors = compute_component_mse_loss(
                    preds,
                    batch.y,
                    loss_fn=loss_fn,
                    include_total_mse=include_total_mse,
                )

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
        include_total_mse=False,
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
                collate_fn=apnet2_collate_update,
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
            collate_fn=apnet2_collate_update,
        )

        test_loader = APNet2_DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=apnet2_collate_update,
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
                self.__evaluate_batches(
                    rank,
                    train_loader,
                    criterion,
                    rank_device,
                    include_total_mse=include_total_mse,
                )
            )
            test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
                self.__evaluate_batches(
                    rank,
                    test_loader,
                    criterion,
                    rank_device,
                    include_total_mse=include_total_mse,
                )
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
                    include_total_mse=include_total_mse,
                )
            )
            test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
                self.__evaluate_batches(
                    rank,
                    test_loader,
                    criterion,
                    rank_device,
                    include_total_mse=include_total_mse,
                )
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        print("Saving model")
                        cpu_model = model_io.unwrap_model(self.model).to("cpu")
                        cpu_atom_model = model_io.unwrap_model(self.atom_model).to(
                            "cpu"
                        )
                        checkpoint = self._create_checkpoint(
                            model=cpu_model,
                            atom_model=cpu_atom_model,
                            embed_atom_model=True,
                        )
                        model_io.save_checkpoint(checkpoint, self.model_save_path)
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
        transfer_learning=False,
        include_total_mse=False,
    ):
        # (1) Compile Model
        rank_device = self.device
        # self.model.to(rank_device)
        batch = self.example_input()
        batch.to(rank_device)
        print(batch)
        self.model(**batch)
        best_model = deepcopy(self.model)
        if not skip_compile:
            print("Compiling model")
            self.compile_model()

        # (2) Dataloaders
        # if self.ds_spec_type in [1, 5, 6]:
        if train_dataset.prebatched:
            collate_fn = apnet2_collate_update_prebatched
        else:
            collate_fn = apnet2_collate_update
        train_loader = APNet2_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # shuffle=False,
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
        component_batch_kwargs = (
            {"include_total_mse": include_total_mse} if not transfer_learning else {}
        )
        t_out = __evaluate_batch(
            train_loader,
            criterion,
            rank_device,
            **component_batch_kwargs,
        )
        v_out = __evaluate_batch(
            test_loader,
            criterion,
            rank_device,
            **component_batch_kwargs,
        )
        if not transfer_learning:
            train_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t = (
                t_out
            )
            test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
                v_out
            )
            print(
                f"  (Pre-training) ({time.time() - t0:<7.2f}s)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} "
                f"{elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} "
                f"{indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} {disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f}",
                flush=True,
            )
        else:
            train_loss, total_MAE_t = t_out
            test_loss, total_MAE_v = v_out
            print(
                f"  (Pre-training) ({time.time() - t0:<7.2f}s)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f}",
                flush=True,
            )

        # (6) Main training loop
        lowest_test_loss = test_loss
        for epoch in range(n_epochs):
            t1 = time.time()
            t_out = __train_batch(
                train_loader,
                criterion,
                optimizer,
                rank_device,
                scheduler,
                **component_batch_kwargs,
            )
            v_out = __evaluate_batch(
                test_loader,
                criterion,
                rank_device,
                **component_batch_kwargs,
            )
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
            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                star_marker = "*"
                cpu_model = model_io.unwrap_model(self.model).to("cpu")
                best_model = deepcopy(cpu_model)
                if self.model_save_path:
                    cpu_atom_model = model_io.unwrap_model(self.atom_model).to("cpu")
                    checkpoint = self._create_checkpoint(
                        model=cpu_model,
                        atom_model=cpu_atom_model,
                        embed_atom_model=True,
                    )
                    model_io.save_checkpoint(checkpoint, self.model_save_path)
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
        include_total_mse=False,
    ):
        """
        Train the APNet2Model on a dataset using single-process or multi-process (DDP) execution.

        Parameters:
            dataset (Dataset | list, optional): If provided, overrides the model's dataset. Can be a single dataset or a [train, test] pair.
            n_epochs (int): Number of training epochs.
            lr (float): Initial learning rate.
            split_percent (float): Fraction of `dataset` to use for training when a single dataset is provided (0-1).
            model_path (str | None): Directory path where training results and checkpoints will be saved.
            shuffle (bool): Whether to shuffle dataset indices before splitting and batching.
            dataloader_num_workers (int): Number of worker processes for data loading in DataLoader.
            world_size (int): Number of processes for distributed training; if >1 runs DDP training via mp.spawn.
            omp_num_threads_per_process (int): Value to set OMP_NUM_THREADS for each training process.
            lr_decay (callable | None): Learning rate scheduler or decay configuration passed to training routines (optional).
            random_seed (int): RNG seed used to initialize numpy permutation for dataset shuffling/splitting.
            skip_compile (bool): If True, skip optional torch.compile model compilation in single-process training.
            transfer_learning (bool): If True, run training in transfer-learning mode (alters loss/aggregation behavior).
            include_total_mse (bool): If True, add an extra MSE term on the total energy in addition to the four component terms.

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
        print(f"  {include_total_mse=}\n", flush=True)
        if self.prebatched:
            print(f"  Prebatched training data: setting batch_size=1", flush=True)
            batch_size = 1
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
                    include_total_mse,
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
                include_total_mse=include_total_mse,
            )
        return
