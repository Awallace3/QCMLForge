import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import warnings
import time
from ..AtomModels.ap2_atom_model import AtomMPNN
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
from apnet_pt.util import scatter_sum_compile
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import qcelemental as qcel
from importlib import resources
from copy import deepcopy
from apnet_pt.torch_util import set_weights_to_value
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


class APNet3_AtomType_MPNN(nn.Module):
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
        freeze_dimer_prop_model=True,
    ):
        # super().__init__(aggr="add")
        """
        Initialize the APNet3_AtomType_MPNN module and configure its architecture and behavior.

        Parameters:
            dimer_prop_model (DimerProp): Optional pretrained dimer property model whose parameters will be frozen if provided (controlled by freeze_dimer_prop_model).
            n_message (int): Number of message-passing iterations.
            n_rbf (int): Number of radial basis functions for distance embeddings.
            n_neuron (int): Base width for hidden layers in MLP blocks.
            n_embed (int): Size of atom-type embedding vectors.
            r_cut_im (float): Cutoff distance for inter-monomer (long-range) distance encoding.
            r_cut (float): Cutoff distance for short-range distance encoding.
            return_hidden_states (bool): If True, forward will return intermediate hidden representations alongside predictions.
            use_precomputed_classical (bool): If True, model expects precomputed classical energy contributions to be provided/used.
            use_atom_props (bool): If True, atom-level classical properties (e.g., charges, polarizabilities) will be included in pair feature construction.
            freeze_dimer_prop_model (bool): If True (default), freeze all parameters of dimer_prop_model to prevent gradient updates.

        Behavior:
            - Freezes parameters of the provided dimer_prop_model (if any and if freeze_dimer_prop_model is True) to prevent gradient updates.
            - Constructs distance embedding layers, an atom embedding layer, readout MLPs for energy components (electrostatics, exchange, induction, dispersion), and per-iteration update and directional MLP stacks according to the provided hyperparameters.
        """
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
        self.freeze_dimer_prop_model = freeze_dimer_prop_model

        if self.freeze_dimer_prop_model and self.dimer_prop_model is not None:
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
                    for param in self.dimer_prop_model.dimer_model_elst.parameters():
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
        Return the model configuration as a dictionary.

        Returns
        -------
        dict
            Configuration dictionary containing all hyperparameters needed
            to reconstruct this model.
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
            "freeze_dimer_prop_model": self.freeze_dimer_prop_model,
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
        """
        Build per-edge feature vectors by gathering source/target atom features and concatenating them with per-atom properties and edge radial-basis features.

        Parameters:
            hA (Tensor): Atom feature matrix for set A with shape (nA, d_h).
            hB (Tensor): Atom feature matrix for set B with shape (nB, d_h).
            qA (Tensor): Per-atom property tensor for set A with shape (nA, d_q).
            qB (Tensor): Per-atom property tensor for set B with shape (nB, d_q).
            rbf (Tensor): Per-edge radial-basis features with shape (n_edges, d_rbf).
            e_source (LongTensor): Indices of source atoms into A for each edge with shape (n_edges,).
            e_target (LongTensor): Indices of target atoms into B for each edge with shape (n_edges,).

        Returns:
            Tensor: Per-edge feature tensor with shape (n_edges, d_h + d_h + d_q + d_q + d_rbf) formed by concatenating
            [hA[source], hB[target], qA[source], qB[target], rbf] along the last dimension.
        """
        hA_source = hA.index_select(0, e_source)
        hB_target = hB.index_select(0, e_target)

        qA_source = qA.index_select(0, e_source)
        qB_target = qB.index_select(0, e_target)
        # print(f"{hA_source.size() = }, {hB_target.size() = }, {qA_source.size() = }, {qB_target.size() = }, {rbf.size() = }")
        return torch.cat([hA_source, hB_target, qA_source, qB_target, rbf], dim=-1)

    def get_pair_params(
        self, hA, hB, qA, qB, hfvrA, hfvrB, vwA, vwB, rbf, e_source, e_target
    ):
        """
        Construct per-edge feature vectors by gathering source and target atom features and concatenating them with radial-basis encodings and, optionally, additional atom properties.

        Parameters:
            hA (Tensor): Per-atom hidden states for set A.
            hB (Tensor): Per-atom hidden states for set B.
            qA (Tensor): Per-atom multipole/charge features for set A.
            qB (Tensor): Per-atom multipole/charge features for set B.
            hfvrA (Tensor): Per-atom fluctuating-valence (or other atomic) properties for set A; used only if `self.use_atom_props` is True.
            hfvrB (Tensor): Per-atom fluctuating-valence properties for set B; used only if `self.use_atom_props` is True.
            vwA (Tensor): Per-atom additional scalar/vector properties for set A; used only if `self.use_atom_props` is True.
            vwB (Tensor): Per-atom additional scalar/vector properties for set B; used only if `self.use_atom_props` is True.
            rbf (Tensor): Radial-basis function encodings for each edge (shape aligned to edges).
            e_source (LongTensor): 1D index tensor mapping each edge to its source atom index in A.
            e_target (LongTensor): 1D index tensor mapping each edge to its target atom index in B.

        Returns:
            Tensor: Per-edge feature tensor with shape (num_edges, F) where each row is the concatenation of:
                hA[source], hB[target], qA[source], qB[target], (hfvrA[source], hfvrB[target], vwA[source], vwB[target] if `self.use_atom_props`), and rbf.
        """
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
        Compute pairwise SAPT-like energy components for a fused batch using intramonomer message passing and distance-based pair features.

        Parameters:
            batch: A fused-batch object containing atom indices, coordinates, and edge lists required by the model (expected attributes include ZA, RA, ZB, RB, e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target, e_AA_source, e_AA_target, e_BB_source, e_BB_target, dimer_ind, dimer_ind_full, total_charge_A, ...).

        Returns:
            Tuple whose contents depend on model configuration:
            - If self.use_precomputed_classical is True:
                (E_output, E_sr, 0, 0, hAB, hBA)
                where E_output is per-dimer short-range energy (summed), E_sr are per-pair short-range contributions, and hAB/hBA are pair feature tensors.
            - Otherwise, if self.return_hidden_states is False:
                (E_output, E_sr, E_elst, E_ind, hAB, hBA)
                where E_output is the sum of short-range, electrostatic, and induction per-dimer energies; E_sr are per-pair short-range contributions; E_elst and E_ind are per-pair classical electrostatic and induction contributions; and hAB/hBA are pair feature tensors.
            - Otherwise (self.return_hidden_states is True):
                (E_output, E_sr_dimer, E_elst, E_ind, hAB, hBA, cutoff)
                where E_sr_dimer is short-range energy aggregated per dimer and cutoff is the distance-dependent scaling applied to short-range pair terms.

        Notes:
            - The forward pass performs per-monomer message passing to build atomic invariant and directional hidden states, constructs pair features (optionally including atomic properties predicted by an internal dimer_prop_model), computes readout energies for both pair orderings, applies a 1/r^3 short-range cutoff, and assembles per-dimer outputs.
            - The precise shapes and presence of returned arrays depend on model flags (use_precomputed_classical, return_hidden_states) and on batch contents.
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
            mA_i = scatter_sum_compile(mA_ij, e_AA_source, dim_size=natomA)
            mB_i = scatter_sum_compile(mB_ij, e_BB_source, dim_size=natomB)

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
            return E_output, E_sr, 0, 0, hAB, hBA
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

            E_output = E_sr_dimer + E_elst_dimer + E_ind_dimer
        if self.return_hidden_states:
            return (
                E_output,
                E_sr_dimer,
                E_elst,
                E_ind,
                hAB,
                hBA,
                cutoff,
            )
        return E_output, E_sr, E_elst, E_ind, hAB, hBA


class APNet3_AtomType_Model:
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
        freeze_dimer_prop_model=True,
    ):
        """
        Initialize the APNet3_AtomType_Model, set device and model components, optionally load pretrained weights, and prepare or load the dataset(s).

        When `pre_trained_model_path` is provided, the APNet3_AtomType_MPNN is loaded from that checkpoint and the remaining model hyperparameters are ignored (dataset-related arguments are still applied). Device selection prefers CUDA if available and `use_GPU` is not False. The initializer constructs or loads an AtomType parameter model and a DimerProp model (optionally from `dimer_prop_model_pre_trained_path` or a provided `dimer_prop_model`), configures the dimer forward mode required for classical predictions, builds or loads the APNet3 model (or loads it from `pre_trained_model_path`), moves models and relevant tensors to the selected device, and prepares dataset objects according to `ds_class_type`, `ds_type`, and split options.

        Parameters:
            dataset: Optional existing dataset or None to construct or load datasets inside the initializer.
            pre_trained_model_path: Path to a checkpoint for the APNet3 model; when provided the model is loaded from this checkpoint.
            dimer_prop_model_pre_trained_path: Path to a checkpoint for the DimerProp model to load instead of constructing/using a provided instance.
            dimer_prop_model: Optional preconstructed DimerProp instance to use instead of creating one.
            am_dimer_param_model: Optional auxiliary dimer parameter model (stored but not otherwise documented here).
            use_GPU: If False, force CPU; otherwise a GPU is used when available.
            use_precomputed_classical: If True, dataset creation and forward passes will expect/use precomputed classical (atomic multipole) inputs.
            ds_class_type: Dataset backend type, either "pt" or "lmdb"; "pt" or "lmdb" controls which dataset class is selected.
            ds_type: Dataset content type, e.g., "total_component_energies" or "fsapt_energies"; affects dataset class selection and processing.
            ds_qcel_molecules: Optional list (or split lists) of qcel molecules to build datasets from; supports split (train/test) when provided as two lists.
            ds_* and other hyperparameters: Various dataset construction and model hyperparameters (n_message, n_rbf, n_neuron, n_embed, r_cut_im, r_cut, batch sizes, reprocessing flags, random seed, max size, etc.) are used to configure the dataset and model; when a checkpoint is loaded, hyperparameters come from the checkpoint config.

        Behavior notes:
            - Validates `ds_class_type` must be "pt" or "lmdb".
            - If `dimer_prop_model_pre_trained_path` is provided, the DimerProp is reconstructed from checkpoint config and its state dict loaded.
            - If `pre_trained_model_path` is provided, the APNet3 model is reconstructed from checkpoint config and its state dict loaded.
            - Model attributes (n_rbf, n_message, n_neuron, n_embed, r_cut_im, r_cut) are adjusted to match the initializer arguments if they differ after loading.
            - If the dataset argument is None and `ignore_database_null` is False, the initializer will construct dataset(s) according to the spec type and split configuration; supports both single and split (train/test) dataset creation.
            - Moves models and relevant tensors (e.g., polarizability tables) to the chosen device.

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
                f"Loading pre-trained APNet3_AtomType_MPNN model from {pre_trained_model_path}"
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
            freeze_dimer_prop_model = config.get("freeze_dimer_prop_model", True)
            self.model = APNet3_AtomType_MPNN(
                dimer_prop_model=self.dimer_prop_model,
                n_message=config["n_message"],
                n_rbf=config["n_rbf"],
                n_neuron=config["n_neuron"],
                n_embed=config["n_embed"],
                r_cut_im=config["r_cut_im"],
                r_cut=config["r_cut"],
                use_precomputed_classical=use_precomputed_classical,
                use_atom_props=use_atom_props,
                freeze_dimer_prop_model=freeze_dimer_prop_model,
            )
            model_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = APNet3_AtomType_MPNN(
                dimer_prop_model=self.dimer_prop_model,
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut_im=r_cut_im,
                r_cut=r_cut,
                use_precomputed_classical=use_precomputed_classical,
                use_atom_props=use_atom_props,
                freeze_dimer_prop_model=freeze_dimer_prop_model,
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

        if hasattr(self.dimer_prop_model, "set_forward"):
            self.dimer_prop_model.set_forward("ap3_elst_damping__induced_dipole")
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
                """
                Create and return a configured dataset instance for this model.

                When `use_precomputed_classical` is true, returns an instance of `self.dataset_class` configured with the model's dimer property model and dataset options; otherwise returns an `ap2_fused_module_dataset` instance configured with equivalent dataset options.

                Parameters:
                        fp (bool): If true, force reprocessing of on-disk dataset artifacts; defaults to `ds_force_reprocess`.

                Returns:
                        Dataset: A dataset object ready for training or evaluation (type depends on `use_precomputed_classical`).
                """
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
                """
                Create and return paired train and test dataset instances configured for the current model and dataset settings.

                When `use_precomputed_classical` is true or `ds_type` equals "fsapt_energies", this returns instances of `self.dataset_class`; otherwise it returns instances of `ap2_fused_module_dataset`. Both datasets are constructed with the same configuration but with `split` set to "train" and "test", respectively.

                Parameters:
                    fp (bool): Override for the `force_reprocess` flag passed to the dataset constructors.

                Returns:
                    list: A two-element list [train_dataset, test_dataset] containing the constructed dataset objects.
                """
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
        """
        Load pretrained model weights from a checkpoint file.

        Parameters
        ----------
        ap2_model_path : str, optional
            Path to the checkpoint file
        am_model_path : str, optional
            Not used (for API compatibility)
        model_id : int, optional
            Model ID to load from bundled ensemble models

        Returns
        -------
        self
            The model instance with loaded weights
        """
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

        The checkpoint embeds the dimer_prop_model as a submodel when available.

        Parameters
        ----------
        metadata : dict, optional
            Additional metadata to include in the checkpoint

        Returns
        -------
        dict
            Complete v2 checkpoint dictionary
        """
        # Get model on CPU for saving
        cpu_model = model_io.unwrap_model(self.model).to("cpu")
        config = cpu_model.get_config()

        # Create submodel checkpoint for dimer_prop_model if available
        submodels = {}
        if (
            hasattr(self, "dimer_prop_model")
            and self.dimer_prop_model is not None
            and hasattr(self.dimer_prop_model, "state_dict")
        ):
            # Get dimer_prop_model config if it has get_config method
            if hasattr(self.dimer_prop_model, "get_config"):
                dimer_prop_config = self.dimer_prop_model.get_config()
            else:
                # Fallback to extracting config from attributes
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
            model_type="APNet3_AtomType_MPNN",
            submodels=submodels if submodels else None,
            metadata=metadata,
        )

        # Move model back to device
        self.model.to(self.device)

        return checkpoint

    def save_model(self, path: str, metadata: dict = None) -> None:
        """
        Save the model to a checkpoint file.

        Parameters
        ----------
        path : str
            Path to save the checkpoint to
        metadata : dict, optional
            Additional metadata to include
        """
        checkpoint = self._create_checkpoint(metadata=metadata)
        model_io.save_checkpoint(checkpoint, path)
        print(f"Model saved to {path}")

    def load_ap2_pretrained_weights(self, ap2_model_path):
        """
        Load matching parameter tensors from an AP2 checkpoint into the current AP3 model instance.

        This method opens the checkpoint at `ap2_model_path`, strips any `_orig_mod.` prefix from AP2 state-dict keys, and copies parameter tensors whose names match keys in the AP3 model's state dictionary for a predefined set of shared layer name prefixes (embed_layer, distance_layer, distance_layer_im, readout_layer_*, update_layers, directional_layers). After copying, the updated AP3 state is loaded into `self.model`. Matching parameter names that were loaded are printed.

        Parameters:
            ap2_model_path (str): Filesystem path to the AP2 model checkpoint containing a `model_state_dict`.

        Returns:
            self: The same model wrapper instance with AP2 parameters merged into the AP3 model where names matched.
        """
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
    ):
        """
        Assemble per-dimer pairwise energy tensors from per-pair energy contributions.

        Parameters:
            inp_batch: batch object containing mapping indices:
                - e_ABsr_source / e_ABsr_target: source/target atom indices for short-range pairs
                - e_ABfull_source / e_ABfull_target: source/target atom indices for full (classical) pairs
                - dimer_ind_full, indA, indB: dimer and within-dimer atom indexing used to group atoms into dimers
            E_sr: iterable of per-pair short-range energy arrays (added across four channels)
            E_elst_mtp: iterable of per-pair electrostatic (MTP) energy values to add to channel 0
            E_ind_mtp: iterable of per-pair induction (MTP) energy values to add to channel 2

        Returns:
            list_of_pair_arrays (list[numpy.ndarray]): list with one array per dimer of shape (4, n_A, n_B),
            where the first dimension indexes energy channels (ordered as elst, exch, indu, disp).
            Each array contains summed contributions from short-range predictions and classical (MTP)
            electrostatic and induction terms accumulated at the corresponding atom-pair positions.
        """
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
            indA_to_atom.append(np.arange(size_A.item()))
            indB_to_atom.append(np.arange(size_B.item()))
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

        # E_sr, E_elst_sr, E_elst_lr
        for e_pair, indA, indB in zip(E_sr, indsA_sr, indsB_sr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][0:4, atomA, atomB] += e_pair.numpy()

        return pair_energies_batch

    def _assemble_pairs_torch(
        self,
        inp_batch,
        E_sr_dimer,
        E_sr,
        E_elst_mtp,
        E_ind_mtp,
    ):
        """
        Assemble per-dimer pairwise SAPT component energies using PyTorch tensors while preserving gradients.

        Parameters:
            inp_batch: batch-like object/namespace containing indexing tensors. Required keys/attributes:
                - "e_ABsr_source", "e_ABsr_target": edge source/target indices for short-range edges.
                - "e_ABlr_source", "e_ABlr_target": edge source/target indices for long-range edges.
                - dimer_ind_full: tensor mapping each atom to its dimer id.
                - indA, indB: monomer-local atom indices for A and B.
            E_sr (Tensor): short-range edge energies with shape [n_edges, 4], ordered as [elst, exch, indu, disp].
            E_elst_mtp (Tensor): long-range electrostatic per-edge energies with shape [n_edges_lr].
            E_ind_mtp (Tensor): long-range induction per-edge energies with shape [n_edges_lr].
            E_sr_dimer: (unused in this implementation) provided for API compatibility.

        Returns:
            list of Tensors: one tensor per dimer with shape [4, size_A, size_B], where dim 0 indexes SAPT components in order [elst, exch, indu, disp]. The returned tensors are PyTorch tensors and preserve gradient flow.
        """
        device = E_sr.device

        indsA_sr = inp_batch["e_ABsr_source"]
        indsB_sr = inp_batch["e_ABsr_target"]
        indsA_lr = inp_batch["e_ABlr_source"]
        indsB_lr = inp_batch["e_ABlr_target"]

        dimer_inds, atoms_per_dimer = torch.unique(
            inp_batch.dimer_ind_full, return_counts=True
        )
        indsA_monomer = inp_batch.indA
        indsB_monomer = inp_batch.indB

        # Build mapping tensors using PyTorch
        indA_to_dimer_list = []
        indA_to_atom_list = []
        indB_to_atom_list = []
        pair_energies_batch = []

        for i in dimer_inds:
            size_A = torch.sum(indsA_monomer == i).item()
            size_B = torch.sum(indsB_monomer == i).item()

            # Create mapping tensors (these are just for indexing, not part of computation graph)
            indA_to_dimer_list.append(
                torch.full((size_A,), i.item(), dtype=torch.long, device=device)
            )
            indA_to_atom_list.append(
                torch.arange(size_A, dtype=torch.long, device=device)
            )
            indB_to_atom_list.append(
                torch.arange(size_B, dtype=torch.long, device=device)
            )

            # Initialize pairwise energy tensor for this dimer
            pair_energies_batch.append(
                torch.zeros((4, size_A, size_B), dtype=E_sr.dtype, device=device)
            )

        indA_to_dimer = torch.cat(indA_to_dimer_list)
        indA_to_atom = torch.cat(indA_to_atom_list)
        indB_to_atom = torch.cat(indB_to_atom_list)

        # Assemble short-range energies (E_sr has shape [n_edges, 4])
        for edge_idx, (indA, indB) in enumerate(zip(indsA_sr, indsB_sr)):
            i = indA_to_dimer[indA].item()
            atomA = indA_to_atom[indA].item()
            atomB = indB_to_atom[indB].item()

            # Add all 4 SAPT components from E_sr
            pair_energies_batch[i][0:4, atomA, atomB] += E_sr[edge_idx]

        # Assemble long-range induction energies
        for edge_idx, (indA, indB) in enumerate(zip(indsA_lr, indsB_lr)):
            i = indA_to_dimer[indA].item()
            atomA = indA_to_atom[indA].item()
            atomB = indB_to_atom[indB].item()

            # Add elst + ind component
            pair_energies_batch[i][0, atomA, atomB] += E_elst_mtp[edge_idx]
            pair_energies_batch[i][2, atomA, atomB] += E_ind_mtp[edge_idx]

        return pair_energies_batch

    def _assemble_mtp_pairs(
        self,
        inp_batch,
        E_elst_mtp,
        E_ind_mtp,
    ):
        """
        Assemble per-dimer matrices of multipole electrostatic and induction pair energies from edge-wise MTP contributions.

        Parameters:
            inp_batch: Batch object containing dimer and per-edge indexing fields:
                - dimer_ind_full: 1D tensor mapping each atom (in concatenated monomer lists) to its dimer index.
                - indA, indB: 1D tensors mapping atoms to their per-batch monomer-local indices.
                - e_ABfull_source, e_ABfull_target: 1D tensors of edge source/target atom indices for the full AB pairing.
            E_elst_mtp: Iterable of scalar electrostatic MTP contributions aligned with edges (same order as e_ABfull_source / e_ABfull_target).
            E_ind_mtp: Iterable of scalar induction MTP contributions aligned with edges (same order as e_ABfull_source / e_ABfull_target).

        Returns:
            pair_elst_batch, pair_ind_batch:
                - pair_elst_batch: list of 2D numpy arrays, one per dimer, where entry (a, b) is the summed electrostatic MTP energy for atom a in monomer A and atom b in monomer B.
                - pair_ind_batch: list of 2D arrays (numpy or torch-compatible), one per dimer, where entry (a, b) is the summed induction MTP energy for that atom pair.
        """
        indA_to_dimer = []
        indB_to_dimer = []
        indA_to_atom = []
        indB_to_atom = []
        pair_elst_batch = []
        pair_ind_batch = []

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
            indA_to_atom.append(np.arange(size_A.item()))
            indB_to_atom.append(np.arange(size_B.item()))
            pair_elst_batch.append(np.zeros((size_A, size_B)))
            pair_ind_batch.append(np.zeros((size_A, size_B)))

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
        return pair_elst_batch, pair_ind_batch

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
        """
        Predict energies for a list of QCEngine dimer molecules in batches using the model's current weights.

        Parameters:
            mols (Sequence): Iterable of QCEngine dimer objects to predict.
            batch_size (int): Number of dimers to process per batch.
            r_cut (float | None): Short-range cutoff distance; uses model.r_cut if None.
            r_cut_im (float | None): Intermonomer cutoff distance; uses model.r_cut_im if None.
            verbose (bool): If True, print progress and warnings about skipped invalid dimers.
            return_pairs (bool): If True, also return per-dimer pairwise short-range energy lists.
            return_classical_pairs (bool): If True, also return per-dimer classical pairwise energies
                as two lists (electrostatics, induction). Mutually exclusive with `return_pairs`.

        Returns:
            If `model.return_hidden_states` is True:
                Tuple (predictions, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers)
                - predictions (ndarray[N,4]): per-dimer aggregated energies [sr, elst, ind, disp] or NaNs for invalid dimers.
                - h_ABs, h_BAs (lists): hidden-state tensors for each processed batch.
                - cutoffs (list): cutoff tensors per batch.
                - dimer_inds (list): original dimer indices tensors per batch.
                - ndimers (list): per-batch number of dimers as tensors.
            Elif `return_pairs` is True:
                Tuple (predictions, pairwise_energies)
                - pairwise_energies (list): per-dimer lists of short-range pair energy tensors (empty for invalid dimers).
            Elif `return_classical_pairs` is True:
                Tuple (predictions, pairwise_elst_energies, pairwise_ind_energies)
                - pairwise_elst_energies, pairwise_ind_energies (lists): per-dimer classical pair energy lists (empty for invalid dimers).
            Else:
                ndarray[N,4]: per-dimer aggregated energies [sr, elst, ind, disp], with NaNs for invalid dimers.

        Notes:
            - Invalid dimers are skipped and represented as NaNs in the predictions array; corresponding pair lists are empty.
            - `return_classical_pairs` and `return_pairs` cannot both be True.
        """
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
            # print(dimer_batch)
            dimer_batch.to(device=self.device)
            preds = self.model(dimer_batch)
            if self.model.return_hidden_states:
                E_sr_dimer, E_sr, E_elst, E_ind, hAB, hBA, cutoff = preds
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
                E_sr_dimer, E_sr, E_elst, E_ind, hAB, hBA = preds
                # predictions[i : i + batch_size] = E_sr_dimer.cpu().numpy()
                v = self._assemble_pairs(
                    dimer_batch.cpu(),
                    E_sr_dimer.cpu(),
                    E_sr.cpu(),
                    E_elst.cpu(),
                    E_ind.cpu(),
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
                E_sr_dimer, E_sr, E_elst, E_ind, hAB, hBA = preds
                v = self._assemble_mtp_pairs(
                    dimer_batch,
                    E_elst,
                    E_ind,
                )
                cnt = 0
                for idx in all_indices:
                    if idx in valid_indices:
                        predictions[i + idx] = E_sr_dimer[cnt].cpu().numpy()
                        pairwise_elst_energies.append(v[0][cnt])
                        pairwise_ind_energies.append(v[1][cnt])
                        cnt += 1
                    else:
                        predictions[i + idx] = np.array(
                            [np.nan, np.nan, np.nan, np.nan]
                        )
                        pairwise_elst_energies.append([])
                        pairwise_ind_energies.append([])
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
            return predictions, pairwise_elst_energies, pairwise_ind_energies
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
            labels = batch.y
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
                labels = batch.y
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
        Perform a single-process training epoch for the transfer-learning workflow, updating model parameters and accumulating epoch loss and mean absolute error.

        Parameters:
            dataloader: Iterable of training batches.
            loss_fn (callable or None): Optional loss function; if None, mean squared error is used.
            optimizer: Optimizer used to update model parameters.
            rank_device: Device to move each batch to (e.g., CPU or CUDA device).
            scheduler: Optional learning-rate scheduler to step after the epoch.

        Returns:
            total_loss (float): Sum of the per-batch loss values accumulated over the epoch.
            total_MAE_t (torch.Tensor): Mean absolute error of the component prediction errors across all batches.
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
            labels = batch.y.squeeze(-1)
            if self.use_precomputed_classical:
                labels -= batch.E_classical_elst
                labels -= batch.E_classical_ind
            comp_errors = preds - labels
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
        """
        Evaluate the model on a dataloader for transfer-learning style targets and compute aggregate loss and mean absolute error.

        Parameters:
            dataloader: Iterable of batches; each batch must have attributes `to(device)`, `y`, and, when `use_precomputed_classical` is True, `E_classical_elst` and `E_classical_ind`.
            loss_fn: Optional loss function accepting (preds, targets). If None, mean squared error between predictions and adjusted labels is used.
            rank_device: Device to move batches to before inference.

        Returns:
            total_loss (float): Sum of per-batch losses accumulated over the dataloader (uses .item()).
            total_MAE_t (torch.Tensor): Mean absolute error across all evaluated samples (scalar tensor).

        Notes:
            Predictions are computed by summing per-dimer short-range pair contributions produced by the model. If `use_precomputed_classical` is True, classical electrostatic and induction terms are subtracted from labels before error computation.
        """
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer, _, _, _, _, _ = self.model(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                preds = torch.sum(preds, dim=1)
                labels = batch.y.squeeze(-1)
                if self.use_precomputed_classical:
                    labels -= batch.E_classical_elst
                    labels -= batch.E_classical_ind
                comp_errors = preds - labels
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
        Train the model for one epoch on FSAPT fragment-energy batches, aggregating MPNN pair contributions into fragment-level predictions and returning the epoch loss and per-component MAEs.

        This method:
        - Maps short-range pairwise predictions (E_sr) into the full pairwise edge index set (e_ABfull) when present.
        - Assembles full pairwise contributions (MPNN short-range plus classical elst/ind slots) and sums only the edges connecting fragment index sets frag1_ind and frag2_ind for each dimer to produce per-dimer predictions.
        - Computes the batch loss using `loss_fn` if provided or mean squared error otherwise, backpropagates, and performs an optimizer step for each batch. If `scheduler` is provided, steps it once after the epoch.
        - Computes mean absolute errors (MAEs) for the summed total and each of the four components.

        Returns:
            total_loss (float): Sum of batch losses accumulated over the epoch.
            total_MAE_t (torch.Tensor): MAE of the summed per-dimer errors across components.
            elst_MAE_t (torch.Tensor): MAE for the electrostatic component.
            exch_MAE_t (torch.Tensor): MAE for the exchange component.
            indu_MAE_t (torch.Tensor): MAE for the induction component.
            disp_MAE_t (torch.Tensor): MAE for the dispersion component.
        """
        self.model.train()
        comp_errors_t = []
        total_loss = 0.0
        for n, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch.to(rank_device, non_blocking=True)
            E_sr_dimer, E_sr, E_elst, E_ind, hAB, hBA = self.model(batch)
            # For FSAPT training, use only MPNN predictions (E_sr),
            # not classical frozen components (E_elst, E_ind)
            full_pairwise_energies = torch.zeros(E_elst.size(0), 4, device=rank_device)
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
            preds = torch.zeros(ndimer, 4, device=rank_device)
            for i in range(ndimer):
                frag1_idx = batch.frag1_ind[i]
                frag2_idx = batch.frag2_ind[i]
                # Find edges where source is in frag1 AND target is in frag2
                mask_source = torch.isin(e_ABfull_source, frag1_idx)
                mask_target = torch.isin(e_ABfull_target, frag2_idx)
                mask = mask_source & mask_target
                # Sum the edge contributions for this fragment pair
                preds[i, :] = full_pairwise_energies[mask, :].sum(dim=0)

            # Labels are [batch_size, 5], we use first 4 components
            labels = batch.y[:, :4]
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

        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, 4)
        total_MAE_t = torch.mean(torch.abs(torch.sum(comp_errors_t, axis=1)))
        elst_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 0]))
        exch_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 1]))
        indu_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 2]))
        disp_MAE_t = torch.mean(torch.abs(comp_errors_t[:, 3]))
        return total_loss, total_MAE_t, elst_MAE_t, exch_MAE_t, indu_MAE_t, disp_MAE_t

    def __evaluate_batches_fsapt_single_proc(self, dataloader, loss_fn, rank_device):
        """
        Evaluate the model on FSAPT fragment batches and compute component-wise errors and loss.

        Processes each batch without gradient updates, assembles per-fragment pairwise predictions from short-range MPNN outputs, sums contributions for fragment pairs defined by frag1_ind/frag2_ind, and compares the resulting 4-component predictions (elst, exch, indu, disp) against labels.

        Returns:
            total_loss (float): Accumulated batch loss summed over the evaluation pass.
            total_MAE_t (Tensor): Mean absolute error of the total interaction per dimer (mean over |sum of component errors|).
            elst_MAE_t (Tensor): Mean absolute error for the electrostatics component.
            exch_MAE_t (Tensor): Mean absolute error for the exchange component.
            indu_MAE_t (Tensor): Mean absolute error for the induction component.
            disp_MAE_t (Tensor): Mean absolute error for the dispersion component.
        """
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
                    E_elst.size(0), 4, device=rank_device
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
                preds = torch.zeros(ndimer, 4, device=rank_device)
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

                # Labels are [batch_size, 5], we use first 4 components
                labels = batch.y[:, :4]

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

        comp_errors_t = torch.cat(comp_errors_t, dim=0).reshape(-1, 4)
        total_MAE_t = torch.mean(torch.abs(torch.sum(comp_errors_t, axis=1)))
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
                        self.save_model(
                            self.model_save_path,
                            metadata={"training_mode": "ddp", "epoch": epoch},
                        )
                else:
                    test_lowered = " "
                dt = time.time() - t1
                test_loss = 0.0
                print(
                    f"  EPOCH: {epoch: 4d}({dt: < 7.2f} sec)  MAE: {
                        total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f} {
                        elst_MAE_t: > 7.3f}/{elst_MAE_v: < 7.3f} {exch_MAE_t: > 7.3f}/{
                        exch_MAE_v: < 7.3f} {indu_MAE_t: > 7.3f}/{indu_MAE_v: < 7.3f} {
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
        """
        Run single-process training and evaluation loop for the model, updating self.model to the best-performing checkpoint and optionally saving it.

        Parameters:
            train_dataset: Training dataset or a Subset-wrapped dataset.
            test_dataset: Validation/test dataset or a Subset-wrapped dataset.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size for data loaders.
            lr (float): Initial learning rate for the optimizer.
            pin_memory (bool): Whether DataLoader should pin memory.
            num_workers (int): Number of worker subprocesses for data loading.
            lr_decay (float or None): Decay rate used to construct the step scheduler; if None no scheduler is used.
            skip_compile (bool): If True, skip compiling the model before training.
            transfer_learning (bool): If True, use transfer-learning training/evaluation paths (different metrics and training functions).

        Notes:
            - Detects FSAPT datasets and switches to FSAPT-specific training/evaluation routines when applicable.
            - Updates self.model in-place to the best validation-checkpoint found during training; if self.model_save_path is set, the best model state and a minimal config are saved to that path.
            - This function performs device placement and may call self.compile_model() unless skip_compile is True.
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
            print(
                "                                       Total            Elst            Exch            Ind            Disp",
                flush=True,
            )
        elif not transfer_learning:
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
        if is_fsapt or not transfer_learning:
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
        Train the APNet3 fused model using the provided dataset and training hyperparameters.

        Starts or spawns a training run (single-process or multi-process DDP) that prepares train/test splits, configures device and data loaders, optionally loads precomputed classical terms into the dimer property model, and runs the selected training loop. On entry this method sets self.dataset (if provided), self.model_save_path, and self.batch_size and may modify environment thread settings; it raises ValueError if no dataset is available.

        Parameters:
            dataset (optional): Dataset or a two-item list [train_dataset, test_dataset]. If None, uses self.dataset.
            n_epochs (int): Number of training epochs.
            lr (float): Initial learning rate.
            split_percent (float): Fraction of dataset to use for training when a single dataset is provided.
            model_path (str or None): Filesystem path where model/checkpoints will be saved; assigned to self.model_save_path.
            shuffle (bool): Whether to randomize dataset order before splitting/creating train/test subsets.
            dataloader_num_workers (int): Number of worker processes for PyTorch DataLoader in single-process mode.
            world_size (int): Number of processes for distributed training; >1 enables multi-process DDP via mp.spawn.
            omp_num_threads_per_process (int): Value to set OMP_NUM_THREADS for each spawned process.
            lr_decay (optional): Learning-rate scheduler configuration passed through to training loop (format handled by inner methods).
            random_seed (int): Seed used for numpy RNG when shuffling/splitting datasets.
            skip_compile (bool): If True, skip model compilation step before training in single-process mode.
            transfer_learning (bool): If True, enable transfer-learning training pathway in the single-process training routine.

        Raises:
            ValueError: If no dataset is provided or available on self.dataset.

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
        Freeze all model parameters except readout-layer parameters for elst, exch, indu, and disp.

        Keeps trainable only parameters whose name contains "readout" and whose top-level module name ends with one of: "elst", "exch", "indu", "disp". Sets requires_grad=False for all other parameters.
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
        Unfreeze all parameters of the wrapped AP3 model so they can be trained.

        Sets requires_grad = True for every parameter in self.model.
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        return
