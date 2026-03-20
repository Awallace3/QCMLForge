import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import warnings
import time
from ..pt_datasets.ap2_fused_ds import (
    ap2_fused_module_dataset,
    APNet2_fused_DataLoader,
    qcel_dimer_to_fused_data,
)
from ..pt_datasets.ap3_fused_ds import (
    ap3_fused_module_dataset_lmdb,
    ap3_fused_module_dataset,
    ap3_fused_collate_update,
    ap3_fused_collate_update_no_target,
    qcel_inputs_are_split_db,
)
from ..pt_datasets.ap3_fused_fsapt_ds import (
    ap3_fused_fsapt_collate_update,
    ap3_fused_fsapt_module_dataset_lmdb,
)
from .. import constants
from ..hf_pretrained import resolve_pretrained_path
from .. import model_io
from ..util import scatter_sum_compile
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import qcelemental as qcel
from importlib import resources
from copy import deepcopy
from apnet_pt.torch_util import set_weights_to_value
from qcml_dftd3.d3 import resolve_d3_damping_parameters
from .mtp_mtp import (
    AtomTypeParamNN,
    DimerProp,
    AtomTypeParamModel,
    load_dimer_prop_from_checkpoint,
)


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


# Need to pass in dimer_prop_model that has dispersion packed into it.
class APNet3D3_AtomType_MPNN(nn.Module):
    def __init__(
        self,
        dimer_prop_model: DimerProp,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut_im=8.0,
        r_cut=5.0,
        return_hidden_states=False,
        use_precomputed_classical=False,
        use_atom_props=True,
        no_disp_nn=False,
        freeze_dimer_prop_model=None,
        d3_damping_parameters=None,
    ):
        super().__init__()
        self.dimer_prop_model = dimer_prop_model
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut_im = r_cut_im
        self.r_cut = r_cut
        self.return_hidden_states = return_hidden_states
        self.use_precomputed_classical = use_precomputed_classical
        self.use_atom_props = use_atom_props
        self.no_disp_nn = no_disp_nn
        self.freeze_dimer_prop_model = freeze_dimer_prop_model
        self.d3_damping_parameters = resolve_d3_damping_parameters(
            d3_damping_parameters
        )

        if self.freeze_dimer_prop_model:
            if self.dimer_prop_model is not None:
                if hasattr(self.dimer_prop_model, "parameters"):
                    for param in self.dimer_prop_model.parameters():
                        param.requires_grad = False
                elif hasattr(self.dimer_prop_model, "model"):
                    for param in self.dimer_prop_model.model.parameters():
                        param.requires_grad = False
                    if hasattr(self.dimer_prop_model, "dimer_model"):
                        for param in self.dimer_prop_model.dimer_model.parameters():
                            param.requires_grad = False
                    if hasattr(self.dimer_prop_model, "dimer_model_elst"):
                        for (
                            param
                        ) in self.dimer_prop_model.dimer_model_elst.parameters():
                            param.requires_grad = False

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
        if not no_disp_nn:
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

    def get_config(self) -> dict:
        """
        Return model hyperparameters required to reconstruct this model.
        """
        return {
            "n_message": self.n_message,
            "n_rbf": self.n_rbf,
            "n_neuron": self.n_neuron,
            "n_embed": self.n_embed,
            "r_cut_im": self.r_cut_im,
            "r_cut": self.r_cut,
            "use_atom_props": self.use_atom_props,
            "use_precomputed_classical": self.use_precomputed_classical,
            "no_disp_nn": self.no_disp_nn,
            "freeze_dimer_prop_model": self.freeze_dimer_prop_model,
            "d3_damping_parameters": deepcopy(self.d3_damping_parameters),
        }

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

    def get_pair_params(
        self, hA, hB, qA, qB, hfvrA, hfvrB, vwA, vwB, rbf, e_source, e_target
    ):
        hA_source = hA.index_select(0, e_source)
        hB_target = hB.index_select(0, e_target)

        qA_source = qA.index_select(0, e_source)
        qB_target = qB.index_select(0, e_target)

        if self.use_atom_props:
            hfvrA_source = hfvrA.index_select(0, e_source)
            hfvrB_target = hfvrB.index_select(0, e_target)

            vwA_source = vwA.index_select(0, e_source)
            vwB_target = vwB.index_select(0, e_target)
            return torch.cat(
                [
                    hA_source,
                    hB_target,
                    qA_source,
                    qB_target,
                    hfvrA_source,
                    hfvrB_target,
                    vwA_source,
                    vwB_target,
                    rbf,
                ],
                dim=-1,
            )
        else:
            return torch.cat([hA_source, hB_target, qA_source, qB_target, rbf], dim=-1)

    def get_distances(self, RA, RB, e_source, e_target):
        RA_source = RA.index_select(0, e_source)
        RB_target = RB.index_select(0, e_target)
        dR_xyz = RB_target - RA_source

        # Compute distances with safe operation for square root
        # dR = torch.sqrt(nn.functional.relu(torch.sum(dR_xyz**2, dim=-1)))
        dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
        return dR, dR_xyz

    # @torch.compile
    def readouts(self, H):
        parts = [
            self.readout_layer_elst(H),
            self.readout_layer_exch(H),
            self.readout_layer_indu(H),
        ]
        if not self.no_disp_nn:
            parts.append(self.readout_layer_disp(H))
        return torch.cat(parts, dim=1)

    def forward(
        self,
        batch,
    ):
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
        # dimer_ind_lr = batch.dimer_ind_lr
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

        if self.use_precomputed_classical:
            mA, mB = self.dimer_prop_model(batch)
        else:
            E_classical, mA, mB = self.dimer_prop_model(batch)
            E_elst = E_classical[:, 0]
            E_ind = E_classical[:, 1]
            if not self.no_disp_nn:
                E_disp = E_classical[:, 2]
        qA = mA[0]
        qB = mB[0]
        qA = qA.view(-1, 1)
        qB = qB.view(-1, 1)
        hfvrA = mA[-2][:, 0].view(-1, 1)
        hfvrB = mB[-2][:, 0].view(-1, 1)
        vwA = mA[-2][:, 1].view(-1, 1)
        vwB = mB[-2][:, 1].view(-1, 1)
        # print(f"{hfvrA.shape = }, {hfvrB.shape = }, {vwA.shape = }, {vwB.shape = }")
        # print(f"{qB.shape = }")
        # print(f"{qA.shape = }, {muA.shape = }, {quadA.shape = }")
        # print(f"{Elst.shape = }")

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
            mA_i = scatter_sum_compile(mA_ij, e_AA_source, int(natomA))
            mB_i = scatter_sum_compile(mB_ij, e_BB_source, int(natomB))

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
            hA_dir = scatter_sum_compile(mA_ij_dir, e_AA_source, int(natomA))
            hB_dir = scatter_sum_compile(mB_ij_dir, e_BB_source, int(natomB))
            hA_dir_list.append(hA_dir)
            hB_dir_list.append(hB_dir)

        # concatenate hidden states over MP iterations
        hA = torch.cat(hA_list, dim=-1)
        hB = torch.cat(hB_list, dim=-1)

        # atom-pair features are a combo of atomic hidden states and the interatomic distance
        hAB = self.get_pair_params(
            hA, hB, qA, qB, hfvrA, hfvrB, vwA, vwB, rbf_sr, e_ABsr_source, e_ABsr_target
        )
        hBA = self.get_pair_params(
            hB, hA, qB, qA, hfvrB, hfvrA, vwB, vwA, rbf_sr, e_ABsr_target, e_ABsr_source
        )
        # hAB = self.get_pair(hA, hB, qA, qB, rbf_sr, e_ABsr_source, e_ABsr_target)
        # hBA = self.get_pair(hB, hA, qB, qA, rbf_sr, e_ABsr_target, e_ABsr_source)

        # project the directional atomic hidden states along the interatomic axis
        hA_dir = torch.cat(hA_dir_list, dim=-1)
        hB_dir = torch.cat(hB_dir_list, dim=-1)

        hA_dir_source = hA_dir.index_select(0, e_ABsr_source)
        hB_dir_target = hB_dir.index_select(0, e_ABsr_target)

        hA_dir_blah = torch.einsum("axf,ax->af", hA_dir_source, dR_sr_unit)
        hB_dir_blah = torch.einsum("axf,ax->af", hB_dir_target, -dR_sr_unit)

        hAB = torch.cat([hAB, hA_dir_blah, hB_dir_blah], dim=1)
        hBA = torch.cat([hBA, hB_dir_blah, hA_dir_blah], dim=1)

        EAB_sr = self.readouts(hAB)
        EBA_sr = self.readouts(hBA)

        E_sr = EAB_sr + EBA_sr

        cutoff = (1.0 / (dR_sr**3)).unsqueeze(-1)
        E_sr *= cutoff
        E_sr_dimer = scatter_sum_compile(E_sr, dimer_ind, ndimer)
        if self.use_precomputed_classical:
            E_output = E_sr_dimer
            return E_output, E_sr, 0, 0, 0, hAB, hBA
        else:
            E_elst_full_dimer = scatter_sum_compile(
                E_elst, batch.dimer_ind_full, ndimer
            )
            E_elst_full_dimer = E_elst_full_dimer.unsqueeze(-1)
            N_full, num_cols = E_elst_full_dimer.shape
            full_expanded = E_elst_full_dimer.new_zeros((ndimer, num_cols))
            full_expanded[:N_full] = E_elst_full_dimer
            E_elst_dimer = full_expanded
            rows, cols = E_elst_dimer.shape
            padded = E_elst_dimer.new_zeros((rows, cols + 3))
            padded[:, :cols] = E_elst_dimer
            E_elst_dimer = padded

            E_ind_full_dimer = scatter_sum_compile(E_ind, batch.dimer_ind_full, ndimer)
            E_ind_full_dimer = E_ind_full_dimer.unsqueeze(-1)
            N_full, num_cols = E_ind_full_dimer.shape
            full_expanded = E_ind_full_dimer.new_zeros((ndimer, num_cols))
            full_expanded[:N_full] = E_ind_full_dimer
            E_ind_dimer = full_expanded

            rows, cols = E_ind_dimer.shape
            padded = E_ind_dimer.new_zeros((rows, cols + 3))
            padded[:, 2:3] = E_ind_dimer
            E_ind_dimer = padded

            if self.no_disp_nn:
                # no_disp_nn mode: keep 3 columns (elst at col 0, ind at col 2)
                # E_elst_dimer has shape (ndimer, 4) with elst at col 0
                # E_ind_dimer has shape (ndimer, 4) with ind at col 2
                # E_sr_dimer has shape (ndimer, 3)
                # Truncate to 3 columns
                # print(f"{E_elst_dimer = }")
                # print(f"{E_ind_dimer = }")
                E_output = E_sr_dimer + E_elst_dimer[:, :3] + E_ind_dimer[:, :3]
                E_disp = torch.tensor(0.0, device=E_sr_dimer.device)
            else:
                # Default 4-col mode: build dispersion and sum all
                E_disp_full_dimer = scatter_sum_compile(
                    E_disp, batch.dimer_ind_full, ndimer
                )
                E_disp_full_dimer = E_disp_full_dimer.unsqueeze(-1)
                N_full, num_cols = E_disp_full_dimer.shape
                full_expanded = E_disp_full_dimer.new_zeros((ndimer, num_cols))
                full_expanded[:N_full] = E_disp_full_dimer
                E_disp_dimer = full_expanded

                rows, cols = E_disp_dimer.shape
                padded = E_disp_dimer.new_zeros((rows, cols + 3))
                padded[:, 3:4] = E_disp_dimer
                E_disp_dimer = padded

                E_output = E_sr_dimer + E_elst_dimer + E_ind_dimer + E_disp_dimer
        if self.return_hidden_states:
            return (
                E_output,
                E_sr_dimer,
                E_elst,
                E_ind,
                E_disp,
                hAB,
                hBA,
                cutoff,
            )
        return E_output, E_sr, E_elst, E_ind, E_disp, hAB, hBA


class APNet3D3_AtomType_Model:
    def __init__(
        self,
        dataset=None,
        atom_type_model=None,
        dimer_prop_model=None,
        am_dimer_param_model=None,
        pre_trained_model_path=None,
        dimer_prop_model_pre_trained_path=None,
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
        ds_batch_size=16,
        ds_force_reprocess=False,
        ds_skip_process=False,
        ds_skip_compile=False,
        ds_in_memory=False,
        ds_num_devices=1,
        ds_datapoint_storage_n_objects=1000,
        ds_prebatched=False,
        ds_random_seed=42,
        ds_type="total_component_energies",
        print_lvl=0,
        ds_qcel_molecules=None,
        ds_energy_labels=None,
        use_precomputed_classical=False,
        ds_class_type="lmdb",  # "pt" or "lmdb"
        use_atom_props=True,
        no_disp_nn=False,
        freeze_dimer_prop_model=True,
        d3_damping_parameters=None,
    ):
        """
        the path and all other parameters will be ignored except for dataset.

        use_GPU will check for a GPU and use it if available unless set to false.
        """
        if torch.cuda.is_available() and use_GPU is not False:
            device = torch.device("cuda:0")
            print("running on the GPU")
        else:
            device = torch.device("cpu")
            print("running on the CPU")
        self.device = device
        self.ds_spec_type = ds_spec_type
        self.atom_type_model = AtomTypeParamModel()
        self.dimer_prop_model = DimerProp(ATParam=self.atom_type_model.model)
        self.am_dimer_param_model = am_dimer_param_model
        dimer_prop_model_loaded_from_embed = False
        config = {}

        self.ds_class_type = ds_class_type
        if self.ds_class_type not in ["pt", "lmdb"]:
            raise ValueError("ds_class_type must be 'pt' or 'lmdb'")
        elif self.ds_class_type == "lmdb" and ds_type == "total_component_energies":
            print("Using LMDB dataset class")
            self.dataset_class = ap3_fused_module_dataset_lmdb
        elif self.ds_class_type == "pt" and ds_type == "total_component_energies":
            self.dataset_class = ap3_fused_module_dataset
        elif self.ds_class_type == "lmdb" and ds_type == "fsapt_energies":
            self.dataset_class = ap3_fused_fsapt_module_dataset_lmdb
        elif self.ds_class_type == "pt" and ds_type == "fsapt_energies":
            raise NotImplementedError(
                "PT dataset class for fsapt_energies not implemented yet. Use LMDB."
            )
        self.ds_type = ds_type
        print(f"{self.ds_type = }")
        print(f"{self.ds_class_type = }")
        print(f"{self.dataset_class = }")

        if dimer_prop_model_pre_trained_path:
            print(
                f"Loading pre-trained DimerProp model from {dimer_prop_model_pre_trained_path}"
            )
            checkpoint = torch.load(
                dimer_prop_model_pre_trained_path,
                map_location=device,
                weights_only=False,
            )
            self.dimer_prop_model = DimerProp(
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
            self.dimer_prop_model.load_state_dict(model_state_dict)
        elif dimer_prop_model:
            print("Using provided DimerProp model:", dimer_prop_model)
            self.dimer_prop_model = dimer_prop_model
        else:
            print(
                """No atom model provided.
    Assuming atomic multipoles and embeddings are
    pre-computed and passed as input to the model.
"""
            )
        self.use_precomputed_classical = use_precomputed_classical
        if pre_trained_model_path:
            print(
                f"Loading pre-trained APNet3D3_AtomType_MPNN model from {pre_trained_model_path}"
            )
            checkpoint = model_io.load_checkpoint(
                pre_trained_model_path,
                map_location=device,
            )
            version = model_io.get_checkpoint_version(checkpoint)
            if version >= 2 and model_io.has_embedded_submodel(
                checkpoint,
                "dimer_prop_model",
            ):
                if dimer_prop_model_pre_trained_path:
                    model_io.warn_submodel_override(
                        "dimer_prop_model",
                        embedded_type="DimerProp",
                        external_path=dimer_prop_model_pre_trained_path,
                    )
                dimer_checkpoint = model_io.get_submodel_checkpoint(
                    checkpoint,
                    "dimer_prop_model",
                )
                self.dimer_prop_model = load_dimer_prop_from_checkpoint(
                    dimer_checkpoint,
                    freeze_atom_model=False,
                )
                dimer_prop_model_loaded_from_embed = True
            config = model_io.load_config_from_checkpoint(checkpoint) or {}
            use_atom_props = config.get("use_atom_props", True)
            no_disp_nn = config.get("no_disp_nn", False)
            if freeze_dimer_prop_model is None:
                freeze_dimer_prop_model = config.get("freeze_dimer_prop_model", True)
            resolved_d3_damping_parameters = resolve_d3_damping_parameters(
                d3_damping_parameters
                if d3_damping_parameters is not None
                else config.get("d3_damping_parameters")
            )
            self.model = APNet3D3_AtomType_MPNN(
                dimer_prop_model=self.dimer_prop_model,
                n_message=config["n_message"],
                n_rbf=config["n_rbf"],
                n_neuron=config["n_neuron"],
                n_embed=config["n_embed"],
                r_cut_im=config["r_cut_im"],
                r_cut=config["r_cut"],
                use_precomputed_classical=use_precomputed_classical,
                use_atom_props=use_atom_props,
                no_disp_nn=no_disp_nn,
                freeze_dimer_prop_model=freeze_dimer_prop_model,
                d3_damping_parameters=resolved_d3_damping_parameters,
            )
            model_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
            self.model.load_state_dict(model_state_dict)
        else:
            if freeze_dimer_prop_model is None:
                freeze_dimer_prop_model = True
            resolved_d3_damping_parameters = resolve_d3_damping_parameters(
                d3_damping_parameters
            )
            self.model = APNet3D3_AtomType_MPNN(
                dimer_prop_model=self.dimer_prop_model,
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut_im=r_cut_im,
                r_cut=r_cut,
                use_precomputed_classical=use_precomputed_classical,
                use_atom_props=use_atom_props,
                no_disp_nn=no_disp_nn,
                freeze_dimer_prop_model=freeze_dimer_prop_model,
                d3_damping_parameters=resolved_d3_damping_parameters,
            )
        self.d3_damping_parameters = deepcopy(resolved_d3_damping_parameters)
        self.model.d3_damping_parameters = deepcopy(resolved_d3_damping_parameters)
        if hasattr(self.dimer_prop_model, "set_d3_damping_parameters"):
            self.dimer_prop_model.set_d3_damping_parameters(
                resolved_d3_damping_parameters
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
        if hasattr(self.dimer_prop_model, "set_forward"):
            if self.model.no_disp_nn:
                self.dimer_prop_model.set_forward("ap3_elst_damping__induced_dipole")
            else:
                self.dimer_prop_model.set_forward(
                    "ap3_elst_damping__induced_dipole__disp"
                )
            self.dimer_prop_model.to(device)
            self.dimer_prop_model.polarizability_table = (
                self.dimer_prop_model.polarizability_table.to(self.device)
            )
        elif hasattr(self.dimer_prop_model, "dimer_model"):
            self.dimer_prop_model.dimer_model.set_forward(
                "ap3_elst_damping__induced_dipole"
            )
            if hasattr(self.dimer_prop_model, "model"):
                self.dimer_prop_model.model.to(device)
            self.dimer_prop_model.dimer_model.to(device)
            if hasattr(self.dimer_prop_model, "dimer_model_elst"):
                self.dimer_prop_model.dimer_model_elst.to(device)
            self.dimer_prop_model.dimer_model.polarizability_table = (
                self.dimer_prop_model.dimer_model.polarizability_table.to(self.device)
            )

        if dimer_prop_model_loaded_from_embed:
            print("Loaded embedded DimerProp from checkpoint")

        self.model.to(device)

        is_split_db_config = getattr(self.dataset_class, "is_split_db_config", None)
        ds_split_db = (
            is_split_db_config(self.ds_spec_type, ds_qcel_molecules)
            if is_split_db_config is not None
            else False
        )
        ds_qcel_split_db = qcel_inputs_are_split_db(ds_qcel_molecules)
        self.dataset = dataset
        if self.dataset is not None:
            ds_split_db = getattr(self.dataset, "split_db", ds_split_db)
        print(
            not ignore_database_null,
            self.dataset is None,
            not ds_split_db,
            not ds_qcel_split_db,
        )
        if (
            not ignore_database_null
            and self.dataset is None
            and not ds_split_db
            and not ds_qcel_split_db
        ):

            def setup_ds(fp=ds_force_reprocess):
                if use_precomputed_classical:
                    return self.dataset_class(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model=self.dimer_prop_model,
                        dimer_prop_model=self.dimer_prop_model,
                        atomic_batch_size=ds_atomic_batch_size,
                        batch_size=ds_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        skip_compile=ds_skip_compile,
                        random_seed=ds_random_seed,
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        print_level=print_lvl,
                        qcel_molecules=ds_qcel_molecules,
                        energy_labels=ds_energy_labels,
                        in_memory=ds_in_memory,
                        device=self.device,
                    )
                else:
                    return ap2_fused_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model=self.dimer_prop_model,
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
                        in_memory=ds_in_memory,
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
                if use_precomputed_classical or ds_type == "fsapt_energies":
                    return [
                        self.dataset_class(
                            root=ds_root,
                            r_cut=r_cut,
                            r_cut_im=r_cut_im,
                            spec_type=ds_spec_type,
                            max_size=ds_max_size,
                            force_reprocess=fp,
                            atom_model=self.dimer_prop_model,
                            dimer_prop_model=self.dimer_prop_model,
                            atomic_batch_size=ds_atomic_batch_size,
                            batch_size=ds_batch_size,
                            num_devices=ds_num_devices,
                            skip_processed=ds_skip_process,
                            skip_compile=ds_skip_compile,
                            random_seed=ds_random_seed,
                            split="train",
                            datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                            print_level=print_lvl,
                            qcel_molecules=ds_qcel_molecules[0],
                            energy_labels=ds_energy_labels[0],
                            in_memory=ds_in_memory,
                            device=self.device,
                        ),
                        self.dataset_class(
                            root=ds_root,
                            r_cut=r_cut,
                            r_cut_im=r_cut_im,
                            spec_type=ds_spec_type,
                            max_size=ds_max_size,
                            force_reprocess=fp,
                            atom_model=self.dimer_prop_model,
                            dimer_prop_model=self.dimer_prop_model,
                            atomic_batch_size=ds_atomic_batch_size,
                            batch_size=ds_batch_size,
                            num_devices=ds_num_devices,
                            skip_processed=ds_skip_process,
                            skip_compile=ds_skip_compile,
                            random_seed=ds_random_seed,
                            split="test",
                            datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                            print_level=print_lvl,
                            qcel_molecules=ds_qcel_molecules[1],
                            energy_labels=ds_energy_labels[1],
                            in_memory=ds_in_memory,
                            device=self.device,
                        ),
                    ]
                else:
                    return [
                        ap2_fused_module_dataset(
                            root=ds_root,
                            r_cut=r_cut,
                            r_cut_im=r_cut_im,
                            spec_type=ds_spec_type,
                            max_size=ds_max_size,
                            force_reprocess=fp,
                            atom_model=self.dimer_prop_model,
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
                            in_memory=ds_in_memory,
                        ),
                        ap2_fused_module_dataset(
                            root=ds_root,
                            r_cut=r_cut,
                            r_cut_im=r_cut_im,
                            spec_type=ds_spec_type,
                            max_size=ds_max_size,
                            force_reprocess=fp,
                            atom_model=self.dimer_prop_model,
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
                            in_memory=ds_in_memory,
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
            ap2_model_path = resolve_pretrained_path(
                f"ap2-fused_ensemble/ap2_{model_id}.pt"
            )
        elif ap2_model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided.")

        checkpoint = model_io.load_checkpoint(ap2_model_path, map_location=self.device)
        state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)

        # Handle torch.compile prefix mismatch
        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            self.model.load_state_dict(state_dict)
        else:
            # Model was compiled, need to add prefix back
            state_dict_with_prefix = {
                f"_orig_mod.{k}": v for k, v in state_dict.items()
            }
            self.model.load_state_dict(state_dict_with_prefix)
        return self

    def _create_checkpoint(self, metadata: dict | None = None) -> dict:
        """
        Create a v2 checkpoint dictionary for this model.

        The checkpoint embeds the dimer_prop_model as a submodel when
        available.
        """
        cpu_model = model_io.unwrap_model(self.model).to("cpu")
        config = cpu_model.get_config()

        submodels = {}
        if (
            hasattr(self, "dimer_prop_model")
            and self.dimer_prop_model is not None
            and hasattr(self.dimer_prop_model, "state_dict")
        ):
            if hasattr(self.dimer_prop_model, "get_config"):
                dimer_prop_config = self.dimer_prop_model.get_config()
            else:
                dimer_prop_config = {
                    "n_message": getattr(self.dimer_prop_model, "n_message", 3),
                    "n_rbf": getattr(self.dimer_prop_model, "n_rbf", 8),
                    "n_neuron": getattr(self.dimer_prop_model, "n_neuron", 128),
                    "n_embed": getattr(self.dimer_prop_model, "n_embed", 8),
                    "r_cut": getattr(self.dimer_prop_model, "r_cut", 5.0),
                }

            submodels["dimer_prop_model"] = model_io.create_submodel_checkpoint(
                model=self.dimer_prop_model,
                config=dimer_prop_config,
                model_type="DimerProp",
            )

        checkpoint = model_io.create_checkpoint(
            model=cpu_model,
            config=config,
            model_type="APNet3D3_AtomType_MPNN",
            submodels=submodels if submodels else None,
            metadata=metadata,
        )

        self.model.to(self.device)
        return checkpoint

    def save_model(self, path: str, metadata: dict = None) -> None:
        """
        Save the model to a checkpoint file.
        """
        checkpoint = self._create_checkpoint(metadata=metadata)
        model_io.save_checkpoint(checkpoint, path)
        print(f"Model saved to {path}")

    def load_ap2_pretrained_weights(self, ap2_model_path):
        print(f"Loading AP2 pretrained weights from {ap2_model_path}")
        checkpoint = torch.load(
            ap2_model_path, map_location=self.device, weights_only=False
        )

        ap2_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }

        ap3_state_dict = self.model.state_dict()

        shared_layers = [
            "embed_layer",
            "distance_layer",
            "distance_layer_im",
            "readout_layer_elst",
            "readout_layer_exch",
            "readout_layer_indu",
            "readout_layer_disp",
            "update_layers",
            "directional_layers",
        ]

        loaded_params = []
        for layer_name in shared_layers:
            for key in ap2_state_dict.keys():
                if key.startswith(layer_name):
                    if key in ap3_state_dict:
                        ap3_state_dict[key] = ap2_state_dict[key]
                        loaded_params.append(key)

        self.model.load_state_dict(ap3_state_dict)
        print(f"Loaded {len(loaded_params)} parameters from AP2 model:")
        for param in loaded_params:
            print(f"  - {param}")
        return self

    def _qcel_example_input(
        self,
        mols,
        batch_size=1,
        r_cut=5.0,
        r_cut_im=8.0,
    ):
        dimer_batch = ap3_fused_collate_update_no_target(
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
        E_elst_mtp,
        E_ind_mtp,
        E_disp,
    ):
        indA_to_dimer = []
        indB_to_dimer = []
        indA_to_atom = []
        indB_to_atom = []
        pair_energies_batch = []

        indsA_sr = inp_batch["e_ABsr_source"]
        indsB_sr = inp_batch["e_ABsr_target"]
        indsA = inp_batch["e_ABfull_source"]
        indsB = inp_batch["e_ABfull_target"]

        dimer_inds, atoms_per_dimer = torch.unique(
            inp_batch.dimer_ind_full, return_counts=True
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
        for e_elst, e_ind, indA, indB in zip(E_elst_mtp, E_ind_mtp, indsA, indsB):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][0, atomA, atomB] += e_elst.numpy()
            pair_energies_batch[i][2, atomA, atomB] += e_ind.numpy()

        for e_pair, e_elst, e_disp, indA, indB in zip(
            E_sr, E_elst_mtp, E_disp, indsA_sr, indsB_sr
        ):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][0:4, atomA, atomB] += e_pair.numpy()
            pair_energies_batch[i][0, atomA, atomB] += e_elst.numpy()
            pair_energies_batch[i][3, atomA, atomB] += e_disp.numpy()

        indsA_lr = inp_batch["e_ABlr_source"]
        indsB_lr = inp_batch["e_ABlr_target"]

        for e_ind, e_disp, indA, indB in zip(E_ind_mtp, E_disp, indsA_lr, indsB_lr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][2, atomA, atomB] += e_ind
            pair_energies_batch[i][3, atomA, atomB] += e_disp
        return pair_energies_batch

    def _assemble_mtp_pairs(
        self,
        inp_batch,
        E_elst_mtp,
        E_ind_mtp,
        E_disp,
    ):
        indA_to_dimer = []
        indB_to_dimer = []
        indA_to_atom = []
        indB_to_atom = []
        pair_elst_batch = []
        pair_ind_batch = []
        pair_disp_batch = []

        indsA = inp_batch["e_ABfull_source"]
        indsB = inp_batch["e_ABfull_target"]

        dimer_inds, atoms_per_dimer = torch.unique(
            inp_batch.dimer_ind_full, return_counts=True
        )
        indsA_monomer = inp_batch.indA
        indsB_monomer = inp_batch.indB

        for i in dimer_inds:
            size_A = torch.sum(indsA_monomer == i)  # Finding size of each monomer
            size_B = torch.sum(indsB_monomer == i)
            indA_to_dimer.append(np.full((size_A,), i))
            indB_to_dimer.append(np.full((size_B,), i))
            indA_to_atom.append(np.arange(size_A))
            indB_to_atom.append(np.arange(size_B))
            pair_elst_batch.append(np.zeros((size_A, size_B)))
            pair_ind_batch.append(np.zeros((size_A, size_B)))
            pair_disp_batch.append(np.zeros((size_A, size_B)))

        indA_to_dimer = np.concatenate(indA_to_dimer)
        indB_to_dimer = np.concatenate(indB_to_dimer)
        indA_to_atom = np.concatenate(indA_to_atom)
        indB_to_atom = np.concatenate(indB_to_atom)

        for e_elst, indA, indB in zip(E_elst_mtp, indsA, indsB):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_elst_batch[i][atomA, atomB] += e_elst.numpy()
        for e_ind, indA, indB in zip(E_ind_mtp, indsA, indsB):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_ind_batch[i][atomA, atomB] += e_ind

        # D3 pairwise energies
        for e_disp, indA, indB in zip(E_disp, indsA, indsB):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_disp_batch[i][atomA, atomB] += e_disp

        return pair_elst_batch, pair_ind_batch, pair_disp_batch

    @torch.inference_mode()
    def predict_qcel_mols(
        self,
        mols,
        batch_size=1,
        r_cut=None,
        r_cut_im=None,
        verbose=False,
        return_pairs=False,
        return_classical_pairs=False,
    ):
        assert not (return_classical_pairs and return_pairs), (
            "return_classical_pairs and return_pairs are not compatible"
        )
        if r_cut is None:
            r_cut = self.model.r_cut
        if r_cut_im is None:
            r_cut_im = self.model.r_cut_im

        N = len(mols)
        predictions = np.zeros((N, 4))
        if return_pairs:
            pairwise_energies = []
        if return_classical_pairs:
            pairwise_elst_energies = []
            pairwise_ind_energies = []
            pairwise_disp_energies = []
        if self.model.return_hidden_states:
            # need to capture output
            h_ABs, h_BAs, cutoffs, dimer_inds, ndimers = [], [], [], [], []
        # self.model.to(self.device)
        self.dimer_prop_model.to(self.device)
        for i in range(0, N, batch_size):
            upper_bound = min(i + batch_size, N)
            # Need to capture what dimers are invalid and return None to report nan for these systems
            data = [
                qcel_dimer_to_fused_data(
                    dimer,
                    r_cut=r_cut,
                    r_cut_im=r_cut_im,
                    dimer_ind=n,
                    check_validity=True,
                )
                for n, dimer in enumerate(mols[i:upper_bound])
            ]
            # get indices that are None
            valid_indices = [j for j, d in enumerate(data) if d is not None]
            all_indices = list(range(len(data)))
            if len(valid_indices) < len(data):
                if verbose:
                    print(
                        f"Skipping {len(data) - len(valid_indices)} invalid dimers in batch {i} to {upper_bound}"
                    )
                # create a new data list with only valid data
                data = [data[j] for j in valid_indices]
            dimer_batch = ap3_fused_collate_update_no_target(data)
            dimer_batch.to(device=self.device)
            preds = self.model(dimer_batch)
            # If no_disp_nn=True, model outputs 3 cols; expand to 4 by computing D3 at predict time
            if self.model.no_disp_nn:
                # Switch dimer_prop_model to full classical (elst + ind + D3 disp)
                self.dimer_prop_model.set_forward(
                    "ap3_elst_damping__induced_dipole__disp"
                )
                with torch.no_grad():
                    E_classical, _, _ = self.dimer_prop_model(dimer_batch)
                E_disp = E_classical[:, 2]
                ndimer = dimer_batch.total_charge_A.size(0)
                E_disp_dimer = scatter_sum_compile(
                    E_disp, dimer_batch.dimer_ind_full, ndimer
                )
                # Expand predictions from 3 cols to 4 cols
                E_sr_dimer_3col = preds[0]
                E_out4 = torch.zeros((ndimer, 4), device=E_sr_dimer_3col.device)
                E_out4[:, :3] = E_sr_dimer_3col
                E_out4[:, 3] = E_disp_dimer
                E_sr_3col = preds[1]
                E_sr_out4 = torch.zeros((E_sr_3col.size(0), 4), device=E_sr_3col.device)
                E_sr_out4[:, :3] = E_sr_3col
                if self.model.return_hidden_states:
                    preds = (
                        E_out4,
                        E_sr_out4,
                        preds[2],
                        preds[3],
                        E_disp,
                        preds[5],
                        preds[6],
                        preds[7],
                    )
                else:
                    preds = (
                        E_out4,
                        E_sr_out4,
                        preds[2],
                        preds[3],
                        E_disp,
                        preds[5],
                        preds[6],
                    )
                # Restore dimer_prop_model to training mode (elst + ind only)
                self.dimer_prop_model.set_forward("ap3_elst_damping__induced_dipole")
            if self.model.return_hidden_states:
                E_sr_dimer, E_sr, E_elst, E_ind, E_disp, hAB, hBA, cutoff = preds
                h_ABs.append(hAB)
                h_BAs.append(hBA)
                cutoffs.append(cutoff)
                dimer_inds.append(dimer_batch.dimer_ind)
                ndimers.append(
                    torch.tensor(dimer_batch.total_charge_A.size(0), dtype=torch.long)
                )
                # update correct indices in predictions
                for idx, valid_idx in enumerate(valid_indices):
                    predictions[i + valid_idx] = E_sr_dimer[idx].cpu().numpy()
                # predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
            elif return_pairs:
                E_sr_dimer, E_sr, E_elst, E_ind, E_disp, hAB, hBA = preds
                # predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
                v = self._assemble_pairs(
                    dimer_batch.cpu(),
                    E_sr_dimer.cpu(),
                    E_sr.cpu(),
                    E_elst.cpu(),
                    E_ind.cpu(),
                    E_disp.cpu(),
                )
                for idx, valid_idx in enumerate(valid_indices):
                    predictions[i + valid_idx] = E_sr_dimer[idx].cpu().numpy()
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
            elif return_classical_pairs:
                E_sr_dimer, E_sr, E_elst, E_ind, E_disp, hAB, hBA = preds
                v = self._assemble_mtp_pairs(
                    dimer_batch,
                    E_elst,
                    E_ind,
                    E_disp,
                )
                cnt = 0
                for idx in all_indices:
                    if idx in valid_indices:
                        predictions[i + idx] = E_sr_dimer[cnt].cpu().numpy()
                        pairwise_elst_energies.append(v[0][cnt])
                        pairwise_ind_energies.append(v[1][cnt])
                        pairwise_disp_energies.append(v[2][cnt])
                        cnt += 1
                    else:
                        predictions[i + idx] = np.array(
                            [np.nan, np.nan, np.nan, np.nan]
                        )
                        pairwise_elst_energies.append([])
                        pairwise_ind_energies.append([])
                        pairwise_disp_energies.append([])
            else:
                for cnt, idx in enumerate(all_indices):
                    if idx in valid_indices:
                        predictions[i + idx] = preds[0][cnt].cpu().numpy()
                    else:
                        predictions[i + idx] = np.array(
                            [np.nan, np.nan, np.nan, np.nan]
                        )
        if verbose:
            print(f"Predictions for {i} to {i + batch_size} out of {N}")
        if self.model.return_hidden_states:
            return predictions, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers
        if return_pairs:
            return predictions, pairwise_energies
        if return_classical_pairs:
            return (
                predictions,
                pairwise_elst_energies,
                pairwise_ind_energies,
                pairwise_disp_energies,
            )
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
        n_comp = 3 if self.model.no_disp_nn else 4
        self.model.train()
        comp_errors_t = []
        total_loss = 0.0
        for n, batch in enumerate(dataloader):
            # optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()
            batch = batch.to(rank_device, non_blocking=True)
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, E_disp_lr, hAB, hBA = self.model(
                batch
            )
            preds = E_sr_dimer.reshape(-1, n_comp)
            labels = batch.y[:, :n_comp]
            if self.use_precomputed_classical:
                labels[:, 0] -= batch.E_classical_elst
                labels[:, 2] -= batch.E_classical_ind
            comp_errors = preds - labels
            batch_loss = (
                torch.mean(torch.square(comp_errors))
                if (loss_fn is None)
                else loss_fn(preds, labels)
            )
            batch_loss.backward()
            optimizer.step()
            # print(preds[0][0].item(), batch.y[0].numpy())
            # print(f"    Loss value: {batch_loss.item()}")
            total_loss += batch_loss.item()
            comp_errors_t.append(comp_errors.detach().cpu())
        if scheduler is not None:
            scheduler.step()

        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, n_comp)
        total_MAE_t = torch.mean(torch.abs(torch.sum(comp_errors_t, axis=1)))
        elst_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 0]))
        exch_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 1]))
        indu_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 2]))
        disp_MAE_t = (
            torch.tensor(0.0)
            if self.model.no_disp_nn
            else torch.mean(torch.abs(comp_errors_t[:, 3]))
        )
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    # @torch.inference_mode()
    def __evaluate_batches_single_proc(self, dataloader, loss_fn, rank_device):
        n_comp = 3 if self.model.no_disp_nn else 4
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer, _, _, _, _, _, _ = self.model(batch)
                preds = E_sr_dimer.reshape(-1, n_comp)
                labels = batch.y[:, :n_comp]
                if self.use_precomputed_classical:
                    labels[:, 0] -= batch.E_classical_elst
                    labels[:, 2] -= batch.E_classical_ind
                comp_errors = preds - labels
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, labels)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, n_comp)
        total_MAE_t = torch.mean(torch.abs(torch.sum(comp_errors_t, axis=1)))
        elst_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 0]))
        exch_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 1]))
        indu_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 2]))
        disp_MAE_t = (
            torch.tensor(0.0)
            if self.model.no_disp_nn
            else torch.mean(torch.abs(comp_errors_t[:, 3]))
        )
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

    def __train_batches_fsapt_single_proc(
        self, dataloader, loss_fn, optimizer, rank_device, scheduler
    ):
        """
        Single-process training loop for FSAPT fragment energies.

        For FSAPT training, we aggregate atomic pair contributions to fragment-level
        energies using frag1_ind and frag2_ind before computing loss.
        """
        n_comp = 3 if self.model.no_disp_nn else 4
        self.model.train()
        comp_errors_t = []
        total_loss = 0.0
        for n, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch.to(rank_device, non_blocking=True)
            E_sr_dimer, E_sr, E_elst, E_ind, hAB, hBA = self.model(batch)
            # For FSAPT training, use only MPNN predictions (E_sr),
            # not classical frozen components (E_elst, E_ind)
            full_pairwise_energies = torch.zeros(
                E_elst.size(0), n_comp, device=rank_device
            )
            full_pairwise_energies[:, 0] = E_elst
            full_pairwise_energies[:, 2] = E_ind
            # Everything is ordered based on e_ABfull_source/target, so we
            # need to map e_ABsr edges to full edges. We can do this by
            # learning the mapping from e_ABsr to e_ABfull.
            e_ABsr_source = batch.e_ABsr_source
            e_ABsr_target = batch.e_ABsr_target
            e_ABfull_source = batch.e_ABfull_source
            e_ABfull_target = batch.e_ABfull_target
            # For each edge in e_ABsr, find the corresponding index in e_ABfull
            mapping_indices = []
            for src, tgt in zip(e_ABsr_source, e_ABsr_target):
                mask_source = e_ABfull_source == src
                mask_target = e_ABfull_target == tgt
                mask = mask_source & mask_target
                index = torch.nonzero(mask, as_tuple=False).squeeze()
                mapping_indices.append(index)
            # if only long-range edges, mapping_indices will be empty.
            # Generally, long-range models will not be trainable here, but
            # we need to handle this case and not adjust model outputs.
            if len(mapping_indices) > 0:
                mapping_indices = torch.stack(mapping_indices)
                # Now we add the short-range energies to the full pairwise
                # energies to assemble all pairwise contributions in one tensor
                full_pairwise_energies[mapping_indices, :] += E_sr
            # Okay, now we want to only sum over SPECIFIC pairwise contributions
            # defined by frag1_ind and frag2_ind. We will loop over dimers
            # in the batch and sum only the relevant pairwise contributions. Note,
            # frag1_ind and frag2_ind are lists of atom indices for each fragment that
            # are comparable to the atom indices in e_ABfull_source/target.
            ndimer = batch.total_charge_A.size(0)
            preds = torch.zeros(ndimer, n_comp, device=rank_device)
            for i in range(ndimer):
                frag1_idx = batch.frag1_ind[i]
                frag2_idx = batch.frag2_ind[i]
                # Find edges where source is in frag1 AND target is in frag2
                mask_source = torch.isin(e_ABfull_source, frag1_idx)
                mask_target = torch.isin(e_ABfull_target, frag2_idx)
                mask = mask_source & mask_target
                # Sum the edge contributions for this fragment pair
                preds[i, :] = full_pairwise_energies[mask, :].sum(dim=0)

            # Labels are [batch_size, 5], we use first n_comp components
            labels = batch.y[:, :n_comp]
            comp_errors = preds - labels
            batch_loss = (
                torch.mean(torch.square(comp_errors))
                if (loss_fn is None)
                else loss_fn(preds, labels)
            )
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            comp_errors_t.append(comp_errors.detach().cpu())

        if scheduler is not None:
            scheduler.step()

        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, n_comp)
        total_MAE_t = torch.mean(torch.abs(torch.sum(comp_errors_t, axis=1)))
        elst_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 0]))
        exch_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 1]))
        indu_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 2]))
        disp_MAE_t = (
            torch.tensor(0.0)
            if self.model.no_disp_nn
            else torch.mean(torch.abs(comp_errors_t[:, 3]))
        )
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    def __evaluate_batches_fsapt_single_proc(self, dataloader, loss_fn, rank_device):
        """
        Single-process evaluation loop for FSAPT fragment energies.
        """
        n_comp = 3 if self.model.no_disp_nn else 4
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer, E_sr, E_elst, E_ind, hAB, hBA = self.model(batch)
                # For FSAPT evaluation, use only MPNN predictions (E_sr),
                # not classical frozen components (E_elst, E_ind)
                full_pairwise_energies = torch.zeros(
                    E_elst.size(0), n_comp, device=rank_device
                )
                # Don't initialize with frozen classical values
                full_pairwise_energies[:, 0] = E_elst
                full_pairwise_energies[:, 2] = E_ind
                # Everything is ordered based on e_ABfull_source/target, so we
                # need to map e_ABsr edges to full edges. We can do this by
                # learning the mapping from e_ABsr to e_ABfull.
                e_ABsr_source = batch.e_ABsr_source
                e_ABsr_target = batch.e_ABsr_target
                e_ABfull_source = batch.e_ABfull_source
                e_ABfull_target = batch.e_ABfull_target
                # For each edge in e_ABsr, find the corresponding index in e_ABfull
                mapping_indices = []
                for src, tgt in zip(e_ABsr_source, e_ABsr_target):
                    mask_source = e_ABfull_source == src
                    mask_target = e_ABfull_target == tgt
                    mask = mask_source & mask_target
                    index = torch.nonzero(mask, as_tuple=False).squeeze()
                    mapping_indices.append(index)
                # if only long-range edges, mapping_indices will be empty.
                # Generally, long-range models will not be trainable here, but
                # we need to handle this case and not adjust model outputs.
                if len(mapping_indices) > 0:
                    mapping_indices = torch.stack(mapping_indices)
                    # Now we add the short-range energies to the full pairwise
                    # energies to assemble all pairwise contributions in one tensor
                    full_pairwise_energies[mapping_indices, :] += E_sr
                ndimer = batch.total_charge_A.size(0)
                preds = torch.zeros(ndimer, n_comp, device=rank_device)
                # Okay, now we want to only sum over SPECIFIC pairwise contributions
                # defined by frag1_ind and frag2_ind. We will loop over dimers
                # in the batch and sum only the relevant pairwise contributions. Note,
                # frag1_ind and frag2_ind are lists of atom indices for each fragment that
                # are comparable to the atom indices in e_ABfull_source/target.
                for i in range(ndimer):
                    frag1_idx = batch.frag1_ind[i]
                    frag2_idx = batch.frag2_ind[i]
                    # Find edges where source is in frag1 AND target is in frag2
                    mask_source = torch.isin(e_ABfull_source, frag1_idx)
                    mask_target = torch.isin(e_ABfull_target, frag2_idx)
                    mask = mask_source & mask_target
                    # Sum the edge contributions for this fragment pair
                    preds[i, :] = full_pairwise_energies[mask, :].sum(dim=0)

                # Labels are [batch_size, 5], we use first n_comp components
                labels = batch.y[:, :n_comp]

                # No precomputed classical correction for FSAPT supported currently
                # if self.use_precomputed_classical:
                #     labels[:, 0] -= batch.E_classical_elst if hasattr(batch, 'E_classical_elst') else 0
                #     labels[:, 2] -= batch.E_classical_ind if hasattr(batch, 'E_classical_ind') else 0

                comp_errors = preds - labels
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, labels)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())

        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, n_comp)
        total_MAE_t = torch.mean(torch.abs(torch.sum(comp_errors_t, axis=1)))
        elst_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 0]))
        exch_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 1]))
        indu_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 2]))
        disp_MAE_t = (
            torch.tensor(0.0)
            if self.model.no_disp_nn
            else torch.mean(torch.abs(comp_errors_t[:, 3]))
        )
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    ########################################################################
    # SINGLE-PROCESS TRAINING
    ########################################################################

    def __train_batches(
        self, rank, dataloader, loss_fn, optimizer, rank_device, scheduler
    ):
        n_comp = 3 if self.model.no_disp_nn else 4
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
            preds = E_sr_dimer.reshape(-1, n_comp)
            comp_errors = preds - batch.y[:, :n_comp]
            if loss_fn is None:
                batch_loss = torch.mean(torch.square(comp_errors))
            else:
                batch_loss = loss_fn(preds.flatten(), batch.y[:, :n_comp].flatten())

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            total_errors = preds.sum(dim=1) - batch.y[:, :n_comp].sum(dim=1)
            total_error += torch.sum(torch.abs(total_errors)).item()
            elst_error += torch.sum(torch.abs(comp_errors[:, 0])).item()
            exch_error += torch.sum(torch.abs(comp_errors[:, 1])).item()
            indu_error += torch.sum(torch.abs(comp_errors[:, 2])).item()
            if not self.model.no_disp_nn:
                disp_error += torch.sum(torch.abs(comp_errors[:, 3])).item()
            count += preds.numel()
        if scheduler is not None:
            scheduler.step()

        total_loss = torch.tensor(total_loss, dtype=torch.float32, device=rank_device)
        total_error = torch.tensor(total_error, dtype=torch.float32, device=rank_device)
        elst_error = torch.tensor(elst_error, dtype=torch.float32, device=rank_device)
        exch_error = torch.tensor(exch_error, dtype=torch.float32, device=rank_device)
        indu_error = torch.tensor(indu_error, dtype=torch.float32, device=rank_device)
        disp_error = torch.tensor(disp_error, dtype=torch.float32, device=rank_device)
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(elst_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(exch_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(indu_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(disp_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        total_MAE_t = (total_error / count).cpu()
        elst_MAE_t = (elst_error / count).cpu()
        exch_MAE_t = (exch_error / count).cpu()
        indu_MAE_t = (indu_error / count).cpu()
        disp_MAE_t = (
            torch.tensor(0.0) if self.model.no_disp_nn else (disp_error / count).cpu()
        )
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    # @torch.inference_mode()
    def __evaluate_batches(self, rank, dataloader, loss_fn, rank_device):
        n_comp = 3 if self.model.no_disp_nn else 4
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
                preds = E_sr_dimer.reshape(-1, n_comp)
                comp_errors = preds - batch.y[:, :n_comp]
                if loss_fn is None:
                    batch_loss = torch.mean(torch.square(comp_errors))
                else:
                    batch_loss = loss_fn(preds.flatten(), batch.y[:, :n_comp].flatten())

                total_loss += batch_loss.item()
                total_errors = preds.sum(dim=1) - batch.y[:, :n_comp].sum(dim=1)
                total_error += torch.sum(torch.abs(total_errors)).item()
                elst_error += torch.sum(torch.abs(comp_errors[:, 0])).item()
                exch_error += torch.sum(torch.abs(comp_errors[:, 1])).item()
                indu_error += torch.sum(torch.abs(comp_errors[:, 2])).item()
                if not self.model.no_disp_nn:
                    disp_error += torch.sum(torch.abs(comp_errors[:, 3])).item()
                count += preds.numel()

        total_loss = torch.tensor(total_loss, device=rank_device)
        total_error = torch.tensor(total_error, device=rank_device)
        elst_error = torch.tensor(elst_error, device=rank_device)
        exch_error = torch.tensor(exch_error, device=rank_device)
        indu_error = torch.tensor(indu_error, device=rank_device)
        disp_error = torch.tensor(disp_error, device=rank_device)
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(elst_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(exch_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(indu_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(disp_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        total_MAE_t = (total_error / count).cpu()
        elst_MAE_t = (elst_error / count).cpu()
        exch_MAE_t = (exch_error / count).cpu()
        indu_MAE_t = (indu_error / count).cpu()
        disp_MAE_t = (
            torch.tensor(0.0) if self.model.no_disp_nn else (disp_error / count).cpu()
        )
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
            first_pass_data = APNet2_fused_DataLoader(
                dataset=test_dataset[:batch_size],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=ap3_fused_collate_update,
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
            collate_fn=ap3_fused_collate_update,
        )

        test_loader = APNet2_fused_DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=ap3_fused_collate_update,
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
            if self.model.no_disp_nn:
                print(
                    "                                       Total            Elst            Exch            Ind",
                    flush=True,
                )
            else:
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
                if self.model.no_disp_nn:
                    print(
                        f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} {indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f}",
                        flush=True,
                    )
                else:
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
                        self.save_model(
                            self.model_save_path,
                            metadata={"training_mode": "ddp", "epoch": epoch},
                        )
                else:
                    test_lowered = " "
                dt = time.time() - t1
                test_loss = 0.0
                if self.model.no_disp_nn:
                    print(
                        f"  EPOCH: {epoch: 4d}({dt: < 7.2f} sec)  MAE: {
                            total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f} {
                            elst_MAE_t: > 7.3f}/{elst_MAE_v: < 7.3f} {
                            exch_MAE_t: > 7.3f}/{exch_MAE_v: < 7.3f} {
                            indu_MAE_t: > 7.3f}/{indu_MAE_v: < 7.3f} {test_lowered}",
                        flush=True,
                    )
                else:
                    print(
                        f"  EPOCH: {epoch: 4d}({dt: < 7.2f} sec)  MAE: {
                            total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f} {
                            elst_MAE_t: > 7.3f}/{elst_MAE_v: < 7.3f} {
                            exch_MAE_t: > 7.3f}/{exch_MAE_v: < 7.3f} {
                            indu_MAE_t: > 7.3f}/{indu_MAE_v: < 7.3f} {
                            disp_MAE_t: > 7.3f}/{disp_MAE_v: < 7.3f} {test_lowered}",
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
        # Detect if we're using FSAPT dataset (handle Subset wrapper from random_split)
        actual_dataset = (
            train_dataset.dataset
            if hasattr(train_dataset, "dataset")
            else train_dataset
        )
        is_fsapt = isinstance(actual_dataset, (ap3_fused_fsapt_module_dataset_lmdb))

        # Use FSAPT collate function if needed
        if is_fsapt:
            collate_fn = ap3_fused_fsapt_collate_update
            # TODO: remove in production
            # batch_size = 1
        else:
            collate_fn = (
                ap3_fused_collate_update
                if self.model.use_precomputed_classical
                else ap3_fused_collate_update
            )

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
        if is_fsapt:
            # FSAPT fragment energy training
            # ensure pre-compute is not enabled
            assert not self.use_precomputed_classical, (
                "Precomputed classical corrections not supported for FSAPT training."
            )
            __evaluate_batch = self.__evaluate_batches_fsapt_single_proc
            __train_batch = self.__train_batches_fsapt_single_proc
            if self.model.no_disp_nn:
                print(
                    "                                       Total            Elst            Exch            Ind",
                    flush=True,
                )
            else:
                print(
                    "                                       Total            Elst            Exch            Ind            Disp",
                    flush=True,
                )
        elif not transfer_learning:
            __evaluate_batch = self.__evaluate_batches_single_proc
            __train_batch = self.__train_batches_single_proc
            if self.model.no_disp_nn:
                print(
                    "                                       Total            Elst            Exch            Ind",
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
        if is_fsapt or not transfer_learning:
            train_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t = (
                t_out
            )
            test_loss, total_MAE_v, elst_MAE_v, exch_MAE_v, indu_MAE_v, disp_MAE_v = (
                v_out
            )
            if self.model.no_disp_nn:
                print(
                    f"  (Pre-training)({time.time() - t0: < 7.2f}s)  MAE: {
                        total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f} "
                    f"{elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} "
                    f"{indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f}",
                    flush=True,
                )
            else:
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
        lowest_test_loss = test_loss
        for epoch in range(n_epochs):
            t1 = time.time()
            t_out = __train_batch(
                train_loader, criterion, optimizer, rank_device, scheduler
            )
            v_out = __evaluate_batch(test_loader, criterion, rank_device)
            if is_fsapt or not transfer_learning:
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
                    self.save_model(
                        self.model_save_path,
                        metadata={
                            "training_mode": "single_proc",
                            "epoch": epoch,
                        },
                    )
                self.model.to(rank_device)

            if is_fsapt or not transfer_learning:
                if self.model.no_disp_nn:
                    print(
                        f"  EPOCH: {epoch:4d} ({time.time() - t1:<7.2f}s)  MAE: "
                        f"{total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} "
                        f"{exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} {indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} "
                        f"{star_marker}",
                        flush=True,
                    )
                else:
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
        skip_compile=True,
        transfer_learning=False,
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
        print("~~ Training APNet3-fused Model ~~", flush=True)
        print(f"   Labeled data for {self.ds_type}", flush=True)
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

        # Now that dataset has computed classical terms in dataset, we can set
        # to only atomMPNN for training
        if self.use_precomputed_classical:
            self.dimer_prop_model.set_forward("ap3_atomMPNN")
            self.dimer_prop_model.to(self.device)

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
            )
        return

    def freeze_parameters_except_readouts(self):
        """
        Freeze all model parameters except those in the readout layers for AP2 model
        """
        for name, param in self.model.named_parameters():
            term = name.split(".")[0]
            if "readout" in name and term[-4:] in ["elst", "exch", "indu", "disp"]:
                # Only allow disp if not in no_disp_nn mode
                if term[-4:] == "disp" and self.model.no_disp_nn:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
        return

    def unfreeze_all_parameters(self):
        """
        Unfreeze all model parameters for AP3 model
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = True
