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
)
from ..pt_datasets.ap3_fused_fsapt_ds import (
    ap3_fused_fsapt_collate_update,
    ap3_fused_fsapt_module_dataset_lmdb,
)
from .. import constants
from ..util import scatter_sum_compile
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import qcelemental as qcel
from importlib import resources
from copy import deepcopy
from apnet_pt.torch_util import set_weights_to_value
from .mtp_mtp import AtomTypeParamNN, DimerProp, AtomTypeParamModel


def inverse_time_decay(step, initial_lr, decay_steps, decay_rate, staircase=True):
    """
    Compute a learning rate following an inverse-time decay schedule.
    
    Parameters:
        step (int or float): Current training step or epoch.
        initial_lr (float): Initial learning rate at step zero.
        decay_steps (int or float): Normalization factor for the step (controls decay speed).
        decay_rate (float): Coefficient multiplying the normalized step in the denominator.
        staircase (bool): If True, use discrete (staircase) decay by flooring step/decay_steps.
    
    Returns:
        decayed_lr (float): The learning rate after applying inverse-time decay.
    """
    p = step / decay_steps
    if staircase:
        p = np.floor(p)
    return initial_lr / (1 + decay_rate * p)


class InverseTimeDecayLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, initial_lr, decay_steps, decay_rate):
        """
        Create a learning-rate scheduler that applies an inverse-time decay to an optimizer's learning rate.
        
        Parameters:
            optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
            initial_lr (float): Learning rate value used at step 0.
            decay_steps (float): Scale factor controlling the rate at which the learning rate decays.
            decay_rate (float): Decay coefficient used in the inverse-time schedule.
        
        Description:
            The scheduler scales the learning rate by the factor 1 / (1 + (step / decay_steps) * decay_rate) (or the equivalent used by the module's inverse-time implementation) for each step.
        """
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
    """
    Compute a multiplicative factor for exponential per-epoch learning-rate decay with a lower bound.
    
    Parameters:
        epoch (int): Current epoch index.
        decay_factor (float): Per-epoch multiplicative decay (e.g., 0.98).
        initial_lr (float): Learning rate at epoch 0 used to compute the floor fraction.
        min_lr (float): Minimum allowed learning rate.
    
    Returns:
        factor (float): Scalar multiplier to apply to `initial_lr`. Equals `decay_factor**epoch` but floored so the resulting learning rate is at least `min_lr`.
    """
    lr = initial_lr * (decay_factor**epoch)
    return max(lr, min_lr) / initial_lr


class AsymptoticDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_coefficient, last_epoch=-1):
        """
        Create an asymptotic learning-rate scheduler that decays the learning rate as 1 / (1 + epoch / decay_coefficient).
        
        Parameters:
            optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
            decay_coefficient (float): Positive scalar controlling the decay speed; larger values produce slower decay.
            last_epoch (int, optional): Index of last epoch. Set to -1 to start from the beginning (default).
        """
        self.decay_coefficient = decay_coefficient
        super(AsymptoticDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rates scaled by an asymptotic decay factor 1 / (1 + epoch / decay_coefficient) for each base learning rate.
        
        @returns
            list: Current learning rates for each parameter group, where each value is base_lr / (1 + self.last_epoch / self.decay_coefficient).
        """
        return [
            base_lr / (1 + self.last_epoch / self.decay_coefficient)
            for base_lr in self.base_lrs
        ]


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff in PyTorch.
    """

    def __init__(self, exponent):
        """
        Initialize an Envelope and precompute polynomial coefficients used by the envelope cutoff.
        
        The instance stores `exponent` and computes coefficients `p`, `a`, `b`, and `c` for the polynomial envelope that smoothly transitions to zero at the unit cutoff.
        
        Parameters:
            exponent (int): Exponent controlling the sharpness of the envelope; larger values produce a steeper cutoff.
        """
        super(Envelope, self).__init__()
        self.exponent = exponent

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs):
        # Envelope function divided by r
        """
        Compute the envelope function divided by r for input radial values, zeroing outputs for inputs greater than or equal to 1.
        
        Parameters:
            inputs (torch.Tensor): Elementwise radial values (same device/dtype as coefficients). Values >= 1 are treated as outside the cutoff and produce zero.
        
        Returns:
            torch.Tensor: Tensor of the same shape as `inputs` containing 1/inputs + a * inputs^(p-1) + b * inputs^p + c * inputs^(p+1) for entries where inputs < 1, and 0 for entries where inputs >= 1.
        """
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
        """
        Create a DistanceLayer that encodes interatomic distances into a learnable radial basis truncated by a smooth envelope cutoff.
        
        Parameters:
            num_radial (int): Number of radial basis components.
            r_cut (float): Cutoff distance (units of length); values beyond this are suppressed by the envelope.
            envelope_exponent (int): Exponent controlling the sharpness of the envelope cutoff.
        
        Attributes:
            num_radial (int): Copy of the provided `num_radial`.
            inv_cutoff (float): Precomputed reciprocal of `r_cut` (1 / r_cut) for efficient scaling.
            envelope (Envelope): Envelope module used to smoothly zero basis contributions beyond the cutoff.
            frequencies (torch.nn.Parameter): Learnable 1-D tensor of length `num_radial` containing initial canonical frequencies (π * [1..num_radial]).
        """
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
        """
        Project interatomic distances into an envelope-weighted sinusoidal radial basis.
        
        Parameters:
            inputs (torch.Tensor): Tensor of distances (any shape). Values are expected in the same units as the layer cutoff.
        
        Returns:
            torch.Tensor: Tensor where the last dimension is the radial-basis features produced by applying a smooth envelope and `sin(frequencies * scaled_distance)`. The output shape is inputs.shape + (n_frequencies,).
        """
        d_scaled = inputs * self.inv_cutoff
        d_scaled = d_scaled.unsqueeze(-1)
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)


def unwrap_model(model):
    """
    Return the underlying module when the model is wrapped by PyTorch DistributedDataParallel.
    
    Parameters:
    	model: A model instance, possibly wrapped in `torch.distributed.DistributedDataParallel` (DDP).
    
    Returns:
    	The wrapped `nn.Module` if `model` is a DDP wrapper, otherwise the original `model`.
    """
    return model.module if isinstance(model, DDP) else model


class APNet3_AtomType_MPNN(nn.Module):
    def __init__(
        self,
        dimer_prop_model: DimerProp,
        n_message=3,
        n_rbf=16,
        n_neuron=256,
        n_embed=10,
        r_cut_im=10.0,
        r_cut=5.0,
        return_hidden_states=False,
        use_precomputed_classical=False,
        use_atom_props=True,
    ):
        # super().__init__(aggr="add")
        """
        Initialize the APNet3 atom-type message-passing network and construct its embedding, distance encoders, update and readout layers.
        
        Parameters:
            dimer_prop_model: Optional pretrained DimerProp model whose parameters (and any nested model/dimer_model/dimer_model_elst attributes) will be frozen and used for optional classical terms.
            n_message: Number of message-passing iterations (hops).
            n_rbf: Number of radial basis functions for distance encodings.
            n_neuron: Base hidden layer width used to size internal MLPs.
            n_embed: Dimension of the atom-type embedding and final per-edge embedding.
            r_cut_im: Cutoff distance for intramonomer (long-range) radial basis encoding.
            r_cut: Cutoff distance for intermonomer (short-range) radial basis encoding.
            return_hidden_states: If True, forward will return internal hidden states alongside energies.
            use_precomputed_classical: If True, the model will expect and incorporate classical precomputed dimer terms from the dimer_prop_model instead of predicting them.
            use_atom_props: If True, include per-atom classical properties (e.g., higher-order multipole features) when constructing pair features.
        """
        super().__init__()
        self.dimer_prop_model = dimer_prop_model
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
                    for param in self.dimer_prop_model.dimer_model_elst.parameters():
                        param.requires_grad = False

        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut_im = r_cut_im
        self.r_cut = r_cut
        self.return_hidden_states = return_hidden_states
        self.use_precomputed_classical = use_precomputed_classical
        self.use_atom_props = use_atom_props

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
            nn.SiLU(),
            nn.SiLU(),
            nn.SiLU(),
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
        """
        Constructs a Sequential MLP module using the provided layer sizes and activation layers, starting with a lazy input Linear to allow unspecified input dimensionality.
        
        Parameters:
            layer_nodes (Sequence[int]): List of neuron counts for each layer in the MLP (length >= 1).
            activations (Sequence[Optional[nn.Module]]): Activation modules corresponding to each layer position; an entry of None means no activation after that layer.
        
        Returns:
            nn.Sequential: A Sequential module containing a LazyLinear followed by alternating Linear and activation modules according to `layer_nodes` and `activations`.
        """
        layers = []
        # Start with a LazyLinear so we don't have to fix input dim
        layers.append(nn.LazyLinear(layer_nodes[0]))
        layers.append(activations[0])
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            if activations[i + 1] is not None:
                layers.append(activations[i + 1])
        return nn.Sequential(*layers)

    def get_messages(self, h0, h, rbf, e_source, e_target):
        """
        Construct per-edge message features by combining source/target node embeddings and radial basis encodings.
        
        Parameters:
            h0 (Tensor): Initial node embeddings of shape [N, n_embed].
            h (Tensor): Current node embeddings of shape [N, n_embed].
            rbf (Tensor): Radial basis features for edges of shape [E, n_rbf].
            e_source (LongTensor): Source node indices for each edge of shape [E].
            e_target (LongTensor): Target node indices for each edge of shape [E].
        
        Returns:
            Tensor: Per-edge feature tensor of shape [E, 4 * n_embed + 4 * n_embed * n_rbf + n_rbf] where each row is
            [h0_source, h0_target, h_source, h_target, (h0_source/h0_target/h_source/h_target projected onto rbf), rbf].
            If there are no edges returns an empty tensor with shape [0, 4 * n_embed + 4 * n_embed * n_rbf + n_rbf].
        """
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
        Builds concatenated per-edge feature vectors by gathering source/target atom features and radial basis features.
        
        Parameters:
            hA (Tensor): Per-atom hidden states for atom set A with shape [n_atoms_A, ...].
            hB (Tensor): Per-atom hidden states for atom set B with shape [n_atoms_B, ...].
            qA (Tensor): Per-atom scalar features for A (e.g., charges) with shape [n_atoms_A, ...].
            qB (Tensor): Per-atom scalar features for B with shape [n_atoms_B, ...].
            rbf (Tensor): Per-edge radial basis/features with shape [n_edges, rbf_dim].
            e_source (LongTensor): Edge-to-source-atom indices selecting rows from A with shape [n_edges].
            e_target (LongTensor): Edge-to-target-atom indices selecting rows from B with shape [n_edges].
        
        Returns:
            Tensor: Per-edge feature tensor with shape [n_edges, D] formed by concatenating
            [hA_source, hB_target, qA_source, qB_target, rbf] along the last dimension.
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
        Assemble concatenated per-edge pair features by selecting source/target atom rows and joining atomic hidden states, charges, optional atom properties, and radial basis features.
        
        Parameters:
            hA (Tensor): Atom hidden states for monomer A.
            hB (Tensor): Atom hidden states for monomer B.
            qA (Tensor): Atomic charges for monomer A.
            qB (Tensor): Atomic charges for monomer B.
            hfvrA (Tensor): Higher-order atom properties for monomer A (included only if self.use_atom_props is True).
            hfvrB (Tensor): Higher-order atom properties for monomer B (included only if self.use_atom_props is True).
            vwA (Tensor): van der Waals / volume-related atom properties for monomer A (included only if self.use_atom_props is True).
            vwB (Tensor): van der Waals / volume-related atom properties for monomer B (included only if self.use_atom_props is True).
            rbf (Tensor): Radial basis feature tensor for each edge.
            e_source (LongTensor): Source atom indices for each edge.
            e_target (LongTensor): Target atom indices for each edge.
        
        Returns:
            Tensor: Concatenated per-edge feature tensor with columns in the following order:
            - when self.use_atom_props is False: [hA_source, hB_target, qA_source, qB_target, rbf]
            - when self.use_atom_props is True: [hA_source, hB_target, qA_source, qB_target, hfvrA_source, hfvrB_target, vwA_source, vwB_target, rbf]
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
        """
        Compute pairwise displacement vectors and Euclidean distances for indexed source-target atom pairs.
        
        Parameters:
            RA (torch.Tensor): Coordinates of source atoms with shape [N_A, 3].
            RB (torch.Tensor): Coordinates of target atoms with shape [N_B, 3].
            e_source (torch.LongTensor): 1D tensor of source atom indices into `RA` for each pair.
            e_target (torch.LongTensor): 1D tensor of target atom indices into `RB` for each pair.
        
        Returns:
            dR (torch.Tensor): 1D tensor of Euclidean distances for each source-target pair with shape [num_pairs].
            dR_xyz (torch.Tensor): 2D tensor of displacement vectors (target minus source) with shape [num_pairs, 3].
        """
        RA_source = RA.index_select(0, e_source)
        RB_target = RB.index_select(0, e_target)
        dR_xyz = RB_target - RA_source

        # Compute distances with safe operation for square root
        dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
        return dR, dR_xyz

    # @torch.compile
    def readouts(self, H):
        """
        Compute the four SAPT component predictions (electrostatics, exchange, induction, dispersion) from input feature vectors.
        
        Parameters:
            H (torch.Tensor): Input feature tensor for each element (e.g., atom or pair), shape [N, F].
        
        Returns:
            torch.Tensor: Tensor of shape [N, 4] where columns are the predicted components in order: electrostatics, exchange, induction, dispersion.
        """
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
        Compute per-dimer SAPT-like energy components and atom-pair features from a fused batch using intramonomer message passing, directional projections, and pairwise readouts.
        
        Parameters:
            batch: A fused batch object containing monomer atom types, coordinates, edge indices for short- and long-range inter- and intra-monomer graphs, dimer indices, and any precomputed classical inputs required by the dimer property model. The batch must provide attributes accessed by this method (e.g., ZA, RA, ZB, RB, e_ABsr_source/target, e_ABlr_source/target, e_AA_source/target, e_BB_source/target, dimer_ind, dimer_ind_full, total_charge_A).
        
        Returns:
            If use_precomputed_classical is True:
                (E_output, E_sr, 0, 0, hAB, hBA)
                - E_output: per-dimer total energy built from short-range model contributions.
                - E_sr: per-short-range-edge energy contributions (edges x components).
                - 0, 0: placeholders for classical elst/ind when precomputed externals are used.
                - hAB, hBA: per-edge pair feature tensors for A->B and B->A readouts.
            Otherwise:
                (E_output, E_sr, E_elst, E_ind, hAB, hBA)
                - E_output: per-dimer total energy combining short-range and classical components.
                - E_sr: per-short-range-edge energy contributions (edges x components).
                - E_elst: per-dimer classical electrostatic contributions (flattened before assembly).
                - E_ind: per-dimer classical induction contributions (flattened before assembly).
                - hAB, hBA: per-edge pair feature tensors for A->B and B->A readouts.
        
            If return_hidden_states is enabled, the method returns:
                (E_output, E_sr_dimer, E_elst, E_ind, hAB, hBA, cutoff_5)
                where E_sr_dimer is the short-range contribution aggregated per dimer and cutoff_5 is the 1/r^5 scaling applied to edge energies.
        
        Notes:
            - The function assembles per-edge predictions into per-dimer energies using the dimer indices supplied in the batch.
            - The shapes and exact presence of returned tensors depend on model flags (use_precomputed_classical, return_hidden_states).
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

        # cutoff_1 = (1.0 / (dR_sr))
        # cutoff_2 = (1.0 / (dR_sr**2))
        # cutoff_3 = (1.0 / (dR_sr**3))
        # cutoff_4 = (1.0 / (dR_sr**4))
        cutoff_5 = (1.0 / (dR_sr**5))
        # cutoff_6 = (1.0 / (dR_sr**6))
        # cutoff_12 = (1.0 / (dR_sr**12))
        E_sr[:, 0] *= cutoff_5
        E_sr[:, 1] *= cutoff_5
        E_sr[:, 2] *= cutoff_5
        E_sr[:, 3] *= cutoff_5
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
                cutoff_5,
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
        n_rbf=16,
        n_neuron=256,
        n_embed=10,
        r_cut_im=10.0,
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
    ):
        """
        Initialize the APNet3_AtomType_Model wrapper, construct or load submodels, configure device placement, and (optionally) build the dataset(s).
        
        This constructor will:
        - Load a pretrained APNet3 model if `pre_trained_model_path` is provided (in which case other model hyperparameters are ignored except `dataset`).
        - Load or accept a DimerProp model via `dimer_prop_model` or `dimer_prop_model_pre_trained_path`.
        - Select the dataset class based on `ds_class_type` and `ds_type`, optionally creating and returning a dataset or split datasets.
        - Place models and relevant tensors on a CUDA device if available unless `use_GPU` is explicitly False.
        - Configure internal model hyperparameters (n_message, n_rbf, n_neuron, n_embed, r_cut_im, r_cut) to match the constructed/loaded APNet3 model.
        
        Parameters:
            dataset: Optional dataset object to use instead of constructing one from disk.
            atom_type_model: Optional AtomTypeParamModel instance to provide atomic parameters (if not provided a default is created).
            dimer_prop_model: Optional preconstructed DimerProp-like model to supply classical atomic/multipole inputs.
            am_dimer_param_model: Optional alternate dimer parameter model (stored for later use).
            pre_trained_model_path (str | None): Path to an APNet3 checkpoint. When provided, the saved model is loaded and hyperparameters from the checkpoint are used.
            dimer_prop_model_pre_trained_path (str | None): Path to a pretrained DimerProp checkpoint to load and replace the default DimerProp.
            n_message (int): Number of message-passing iterations (used only when not loading a pretrained APNet3 model).
            n_rbf (int): Number of radial basis functions for distance encoding (used only when not loading a pretrained APNet3 model).
            n_neuron (int): Width of intermediate MLP layers (used only when not loading a pretrained APNet3 model).
            n_embed (int): Atom-type embedding dimensionality (used only when not loading a pretrained APNet3 model).
            r_cut_im (float): Short-range interaction cutoff (used for dataset construction and model when not loading a pretrained model).
            r_cut (float): Long-range cutoff for pair interactions (used for dataset construction and model when not loading a pretrained model).
            use_GPU (bool | None): If True, prefer GPU; if False, force CPU; if None, GPU is used only if available.
            ignore_database_null (bool): If False and `dataset` is None, attempt to construct dataset(s) from disk.
            ds_spec_type (int): Dataset specification type controlling dataset splitting/selection behavior.
            ds_root (str): Root path for dataset files.
            ds_max_size (int | None): If set, truncate constructed datasets to this size.
            ds_atomic_batch_size (int): Atomic-batch size used when constructing datasets.
            ds_batch_size (int): Per-sample batch size for dataset construction.
            ds_force_reprocess (bool): Force dataset preprocessing when constructing datasets.
            ds_skip_process (bool): Skip dataset processing when constructing datasets.
            ds_skip_compile (bool): Skip dataset compilation step when constructing datasets.
            ds_in_memory (bool): If True, attempt to load dataset fully into memory.
            ds_num_devices (int): Number of devices to assume when constructing dataset sharding.
            ds_datapoint_storage_n_objects (int): Dataset storage parameter controlling internal sharding.
            ds_prebatched (bool): Not currently used; reserved for prebatched dataset interfaces.
            ds_random_seed (int): Seed used by dataset preprocessing.
            ds_type (str): Type of dataset contents; accepted values include "total_component_energies" and "fsapt_energies".
            print_lvl (int): Verbosity for dataset construction and model messages.
            ds_qcel_molecules: Optional specification of QCEL molecules used to construct dataset entries (may be per-split).
            ds_energy_labels: Optional list of energy label names for dataset construction.
            use_precomputed_classical (bool): If True, dataset construction and model initialization assume classical multipoles/polarizabilities are provided by `dimer_prop_model`.
            ds_class_type (str): Either "pt" or "lmdb" selecting which dataset class implementation to use; a ValueError is raised for other values.
            use_atom_props (bool): If True, atom properties (e.g., hfvr, vw) are included in pairwise features.
        
        Raises:
            ValueError: If `ds_class_type` is not "pt" or "lmdb".
            NotImplementedError: If `ds_type` is "fsapt_energies" and `ds_class_type` is "pt" (PT dataset class for FSAPT not implemented).
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
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            config = checkpoint["config"]
            use_atom_props = config.get("use_atom_props", True)
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
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
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

        self.model.to(device)

        split_dbs = [2, 5, 6, 7]
        ds_qcel_split_db = (
            ds_qcel_molecules is not None
            and len(ds_qcel_molecules) == 2
            and isinstance(ds_qcel_molecules[0], list)
        )
        self.dataset = dataset
        print(
            not ignore_database_null,
            self.dataset is None,
            self.ds_spec_type not in split_dbs,
            not ds_qcel_split_db,
        )
        if (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type not in split_dbs
            and not ds_qcel_split_db
        ):

            def setup_ds(fp=ds_force_reprocess):
                """
                Constructs and returns the dataset instance configured for this model.
                
                Selects between the configured dataset_class (when precomputed classical terms are used)
                and the fused AP2-compatible dataset constructor otherwise, passing through shared
                dataset configuration options captured from the enclosing scope.
                
                Parameters:
                    fp (bool): If True, force reprocessing of raw data files for dataset creation.
                
                Returns:
                    dataset: A dataset object constructed and initialized according to the model's
                    dataset configuration (either `self.dataset_class(...)` or `ap2_fused_module_dataset(...)`).
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
                """
                Create and return paired training and test dataset instances configured for the current model and dataset settings.
                
                Parameters:
                	fp (bool): If True, force reprocessing of raw data for the created dataset objects.
                
                Returns:
                	list: A two-element list [train_dataset, test_dataset] containing instantiated dataset objects. If `use_precomputed_classical` is True or `ds_type == "fsapt_energies"`, the datasets are created via `self.dataset_class` and include `dimer_prop_model`/`atom_model`; otherwise they are created via `ap2_fused_module_dataset`.
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
        """
        Run the model over the configured dataset in evaluation mode.
        
        Moves each batch to the configured device, performs a forward pass to produce per-batch predictions (short-range, total short-range, short-range electrostatics, long-range electrostatics, and hidden states), and does not retain or return those outputs.
        """
        self.model.eval()
        for batch in self.dataset:
            batch = batch.to(self.device)
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.model(batch)
        return

    def compile_model(self):
        """
        Prepare and compile the model for optimized execution with PyTorch Dynamo.
        
        Moves the model to the configured device, enables Dynamo dynamic-shape mode and sets related capture flags, then compiles the model with torch.compile(dynamic=True) for runtime optimization.
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
        Set every trainable parameter of the model to a constant value.
        
        Performs a forward pass using example_input() to ensure any lazy-initialized parameters are created, then assigns `value` to all model parameters (weights and biases).
        
        Parameters:
            value (float): The constant value to write into every parameter tensor.
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
        Load pretrained AP2 weights into this APNet3 wrapper from a checkpoint file or a built-in model identifier.
        
        Parameters:
            ap2_model_path (str | pathlib.Path | None): Path to a saved AP2 checkpoint file. Ignored if `model_id` is provided.
            am_model_path (Any): Unused placeholder kept for API compatibility.
            model_id (str | None): Identifier of a bundled AP2 checkpoint to load from package resources when provided.
        
        Returns:
            self: The current APNet3_AtomType_Model instance with weights loaded from the specified checkpoint.
        
        Raises:
            ValueError: If neither `ap2_model_path` nor `model_id` is provided.
        """
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

    def load_ap2_pretrained_weights(self, ap2_model_path):
        """
        Load shared parameter blocks from an AP2 checkpoint into the current APNet3 model.
        
        Loads a checkpoint from ap2_model_path (mapped to the model device), finds parameters whose keys begin with any of the AP2/AP3 shared layer prefixes (embed_layer, distance_layer, distance_layer_im, readout_layer_elst/exch/indu/disp, update_layers, directional_layers), and copies matching tensors into this model's state dict while leaving nonmatching parameters unchanged. Prints a brief summary of which parameter keys were loaded.
        
        Parameters:
            ap2_model_path (str): Filesystem path to the AP2 model checkpoint.
        
        Returns:
            self: The same APNet3_AtomType_Model instance with updated weights for matched parameters.
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
        """
        Builds a fused dimer Batch from QCEl molecules for use as model input.
        
        Parameters:
            mols (iterable): Iterable of QCEl dimer molecules (objects accepted by qcel_dimer_to_fused_data).
            batch_size (int): Unused placeholder parameter retained for API compatibility.
            r_cut (float): Short-range cutoff passed to qcel_dimer_to_fused_data.
            r_cut_im (float): Intermediate/long-range cutoff passed to qcel_dimer_to_fused_data.
        
        Returns:
            dimer_batch: A fused dataset Batch (as produced by ap3_fused_collate_update_no_target) moved to self.device.
        """
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
        """
        Enable or disable returning hidden states from the underlying model.
        
        When enabled, the model's forward outputs will include hidden state tensors alongside the usual outputs.
        
        Parameters:
            value (bool): If `True`, forward will return hidden states; if `False`, it will not.
        
        Returns:
            self: The caller object, allowing method chaining.
        """
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
        Assemble per-dimer, per-atom-pair energy matrices from per-edge short-range and classical pair contributions.
        
        Parameters:
            inp_batch: batch object with per-edge index mappings. Must provide:
                - "e_ABsr_source", "e_ABsr_target": indices for short-range edges
                - "e_ABfull_source", "e_ABfull_target": indices for full (all) edges
                - dimer_ind_full, indA, indB: tensors mapping atoms/edges to dimer and monomer-local atom indices
            E_sr_dimer: (unused by this function) per-dimer short-range totals (can be None)
            E_sr: iterable of per-pair short-range energy arrays or tensors; each entry supplies up to four component values for a specific short-range atom pair
            E_elst_mtp: iterable of per-pair classical electrostatic (multipole) scalar energies for full edges
            E_ind_mtp: iterable of per-pair classical induction scalar energies for full edges
        
        Returns:
            pair_energies_batch (list of ndarray): one numpy array per dimer with shape [4, size_A, size_B], where size_A/size_B are the number of atoms
            in monomers A and B for that dimer. The first axis indexes energy components in the order:
            [electrostatics, exchange, induction, dispersion]. Each array accumulates contributions from both
            classical (multipole/induction) and learned short-range pair terms.
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
        Assemble per-dimer, per-atom-pair SAPT component energies using PyTorch operations so gradients are preserved.
        
        Constructs a list of tensors (one tensor per dimer) with shape [4, size_A, size_B] containing the four SAPT components ordered as [electrostatics, exchange, induction, dispersion]. Short-range edge contributions from `E_sr` (shape [n_sr_edges, 4]) are placed into the corresponding atom-pair entries; long-range multipole induction/electrostatic contributions from `E_elst_mtp` and `E_ind_mtp` (each shape [n_lr_edges]) are added to the electrostatics and induction components respectively.
        
        Parameters:
            inp_batch: batch object/dict providing indexing tensors:
                - "e_ABsr_source", "e_ABsr_target": short-range source/target atom indices per SR edge
                - "e_ABlr_source", "e_ABlr_target": long-range source/target atom indices per LR edge
                - dimer_ind_full, indA, indB: tensors mapping atoms to dimer indices and monomer-local atom ordering
            E_sr (Tensor): short-range edge energies with shape [n_sr_edges, 4], components ordered [elst, exch, indu, disp].
            E_sr_dimer: (unused in this implementation) kept for API compatibility.
            E_elst_mtp (Tensor): long-range electrostatic per-LR-edge contributions with shape [n_lr_edges].
            E_ind_mtp (Tensor): long-range induction per-LR-edge contributions with shape [n_lr_edges].
        
        Returns:
            list[Tensor]: one tensor per dimer with shape [4, size_A, size_B], dtype matching the input energies, where index 0=electrostatics, 1=exchange, 2=induction, 3=dispersion.
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
        Assemble per-dimer pairwise matrices of multipole electrostatic and induction contributions from edge-wise values.
        
        Parameters:
            inp_batch: Batch-like object containing index mappings:
                - `dimer_ind_full` (Tensor): per-edge dimer indices for full graph.
                - `indA` (Tensor): per-atom indices mapping atoms of monomer A to dimers.
                - `indB` (Tensor): per-atom indices mapping atoms of monomer B to dimers.
                - `e_ABfull_source` / `e_ABfull_target` (Tensors): source/target edge atom indices for edges.
            E_elst_mtp (iterable of Tensor or numeric): electrostatic contribution per short-range edge, ordered to match `e_ABfull_source` / `e_ABfull_target`.
            E_ind_mtp (iterable of Tensor or numeric): induction contribution per short-range edge, ordered to match `e_ABfull_source` / `e_ABfull_target`.
        
        Returns:
            pair_elst_batch (list of ndarray): list of (size_A, size_B) arrays, one per dimer, where each entry contains the summed electrostatic pair contributions for atom pairs (A_i, B_j).
            pair_ind_batch (list of ndarray): list of (size_A, size_B) arrays, one per dimer, where each entry contains the summed induction pair contributions for atom pairs (A_i, B_j).
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
            indA_to_atom.append(np.arange(size_A))
            indB_to_atom.append(np.arange(size_B))
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
        Predict energies for a list of QCEngine (QCEL) dimer molecules using the model's fused APNet3 pipeline.
        
        Parameters:
            mols (Sequence): Iterable of QCEL dimer objects to predict.
            batch_size (int): Number of dimers to process per model forward pass.
            r_cut (float | None): Long-range cutoff to use; defaults to self.model.r_cut when None.
            r_cut_im (float | None): Short-range (intramonomer) cutoff; defaults to self.model.r_cut_im when None.
            verbose (bool): If True, print progress and warnings about skipped invalid dimers.
            return_pairs (bool): If True, also return per-dimer pairwise short-range contributions as lists of pair tensors.
            return_classical_pairs (bool): If True, return per-dimer classical pairwise components (electrostatic and induction) as two lists.
                Note: return_pairs and return_classical_pairs are mutually exclusive.
        
        Returns:
            If neither return_pairs nor return_classical_pairs and model.return_hidden_states is False:
                numpy.ndarray of shape (N, 4) containing per-dimer predicted energies (columns correspond to the four energy components used by this model). Invalid dimers produce rows of NaN.
            If return_pairs is True:
                (predictions, pairwise_energies)
                - predictions: numpy.ndarray (N, 4) as above.
                - pairwise_energies: list of length N where each element is a list/collection of pairwise short-range energy contributions for that dimer (empty list for invalid dimers).
            If return_classical_pairs is True:
                (predictions, pairwise_elst_energies, pairwise_ind_energies)
                - predictions: numpy.ndarray (N, 4).
                - pairwise_elst_energies, pairwise_ind_energies: lists of length N containing per-dimer classical electrostatic and induction pair contributions respectively (empty list for invalid dimers).
            If model.return_hidden_states is True:
                (predictions, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers)
                - predictions: numpy.ndarray (N, 4).
                - h_ABs, h_BAs: lists of hidden-state tensors produced for AB and BA directed pair sets.
                - cutoffs: list of cutoff tensors produced by the model for each batch.
                - dimer_inds: list of dimer index tensors mapping batch entries to original mol indices.
                - ndimers: list of tensors with per-batch monomer sizes.
        
        Behavioral notes:
            - Invalid dimers detected during preprocessing are skipped and yield NaN prediction rows and empty pair lists.
            - The function moves the internal dimer_prop_model to self.device and runs the model in evaluation-forward mode for each batch.
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
        """
        Builds a single-batch example input from a QCElemental Molecule (or a default water dimer) for use with the model.
        
        Parameters:
            mol (qcel.models.Molecule | None): QCElemental Molecule to convert into a fused batch. If None, a default water–water dimer geometry is used.
            r_cut (float): Short-range cutoff radius (angstrom) used when constructing pair lists.
            r_cut_im (float): Intramonomer/long-range cutoff radius (angstrom) used when constructing pair lists.
        
        Returns:
            batch: A fused dataset batch (the same input format expected by the model's forward method) containing one example constructed from `mol`.
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
        """
        Initialize the distributed training environment for the given process rank and total world size.
        
        Sets MASTER_ADDR to "localhost" and MASTER_PORT to "12355", selects the backend ("nccl" when a CUDA device is available, otherwise "gloo"), initializes the process group with the provided rank and world_size, and seeds PyTorch RNG with 43.
        
        Parameters:
            rank (int): Rank of the current process within the distributed group.
            world_size (int): Total number of processes in the distributed group.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        if torch.cuda.is_available():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        else:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.manual_seed(43)

    def __cleanup(self):
        """
        Tears down the active PyTorch distributed process group for the current process.
        
        This releases resources associated with the distributed group and should be called during shutdown of distributed training.
        """
        dist.destroy_process_group()

    def __train_batches_single_proc(
        self, dataloader, loss_fn, optimizer, rank_device, scheduler
    ):
        """
        Run a single-process training epoch over the provided dataloader and return summed loss plus per-component mean absolute errors.
        
        Processes each batch by moving it to rank_device, running the model to obtain short-range dimer component predictions, optionally subtracting precomputed classical terms from targets, computing the batch loss (using loss_fn when provided, otherwise mean squared error), performing backpropagation and an optimizer step, and stepping the scheduler once after the epoch.
        
        Parameters:
            dataloader: Iterable that yields training batches compatible with the model's forward.
            loss_fn: Callable loss function accepting (preds, labels) or None to use mean squared error.
            optimizer: torch.optim.Optimizer used for parameter updates.
            rank_device: torch.device to which batches and model outputs are moved.
            scheduler: Optional LR scheduler whose step() is called once after the epoch, or None.
        
        Returns:
            total_loss (float): Sum of per-batch loss.item() values accumulated over the epoch.
            total_MAE_t (torch.Tensor): Mean absolute error of the total dimer energy (mean over dimers).
            elst_MAE_t (torch.Tensor): Mean absolute error for the electrostatic component.
            exch_MAE_t (torch.Tensor): Mean absolute error for the exchange component.
            indu_MAE_t (torch.Tensor): Mean absolute error for the induction component.
            disp_MAE_t (torch.Tensor): Mean absolute error for the dispersion component.
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
        """
        Evaluate the model on a dataloader and compute aggregated loss and mean absolute errors for total and component energies.
        
        Parameters:
            dataloader: Iterable yielding batched examples compatible with the model (each batch must have .to(device) and .y targets).
            loss_fn (callable or None): Optional loss function taking (preds, labels). If None, mean squared error per batch is used.
            rank_device (torch.device): Device to which each batch will be moved before inference.
        
        Returns:
            tuple: (
                total_loss (float): Sum of per-batch losses accumulated over the dataloader.
                total_MAE_t (torch.Tensor): Mean absolute error of the total dimer energies (|sum of component errors|), scalar tensor.
                elst_MAE_t (torch.Tensor): Mean absolute error for the electrostatics component, scalar tensor.
                exch_MAE_t (torch.Tensor): Mean absolute error for the exchange component, scalar tensor.
                indu_MAE_t (torch.Tensor): Mean absolute error for the induction component, scalar tensor.
                disp_MAE_t (torch.Tensor): Mean absolute error for the dispersion component, scalar tensor
            )
        
        Notes:
            - If self.use_precomputed_classical is True, classical electrostatics and induction terms from the batch are subtracted from the labels before computing errors.
            - The model is evaluated with torch.no_grad() and set to eval() mode inside the method.
        """
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
        Run a single-process training epoch for the transfer-learning workflow and return accumulated loss and mean absolute error.
        
        This iterates over `dataloader`, moves each batch to `rank_device`, performs a forward pass with `self.model`, constructs scalar predictions by summing the four short-range per-dimer components, computes a batch loss (mean-squared error by default or `loss_fn` if provided), backpropagates, and updates model parameters via `optimizer`. If `scheduler` is provided it is stepped once after the epoch.
        
        Parameters:
            dataloader: Iterable that yields training batches. Each batch must contain inputs accepted by `self.model` and a target `y`.
            loss_fn (callable or None): Optional loss function taking (preds, targets). If `None`, mean-squared error is used.
            optimizer: Optimizer used to update model parameters.
            rank_device: torch.device to which batches are moved before the forward pass.
            scheduler: Optional learning-rate scheduler stepped once after the epoch.
        
        Returns:
            total_loss: Sum of per-batch loss values accumulated over the epoch (float).
            total_MAE_t: Mean absolute error across all examples in the epoch (scalar tensor).
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
        """
        Evaluate the wrapped model on `dataloader` for the transfer-learning workflow and return the aggregated loss and mean absolute error.
        
        Runs the model in evaluation mode without gradient computation, obtains per-dimer short-range predictions, collapses per-dimer component predictions to total energies, and computes per-batch losses which are accumulated. If `loss_fn` is None, per-batch mean squared error is used; otherwise `loss_fn` is applied to predictions and targets.
        
        Parameters:
            dataloader: Iterable of batches providing input data and targets (batch.y).
            loss_fn (callable or None): Optional loss function taking (preds, targets). If None, mean squared error is used.
            rank_device: Device to which each batch will be moved before inference.
        
        Returns:
            total_loss (float): Sum of per-batch losses accumulated over `dataloader`.
            total_MAE_t (torch.Tensor): Mean absolute error across all examples (scalar tensor).
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
        Perform a single-process training epoch for FSAPT fragment energies using MPNN-derived short-range pair predictions.
        
        This method trains the model on a dataloader of fused FSAPT batches by assembling per-edge short-range predictions into fragment-level pairwise energies (using frag1_ind and frag2_ind), computing a component-wise loss against the first four target components, and applying optimizer updates. When short-range-to-full-edge mapping is empty the model skips adding short-range contributions. The scheduler is stepped once after the epoch when provided.
        
        Parameters:
            dataloader: Iterable of batches produced by the fused dataset collate function; each batch must contain e_ABsr_source/target, e_ABfull_source/target, frag1_ind, frag2_ind, and y among other fields.
            loss_fn: Optional loss function taking (preds, labels). If None, mean squared error is used.
            optimizer: Optimizer used for parameter updates; a single step is taken per batch.
            rank_device: Device on which tensors and model operate.
            scheduler: Optional LR scheduler to step once after the epoch.
        
        Returns:
            total_loss (float): Sum of batch losses accumulated over the epoch.
            total_MAE_t (torch.Tensor): Mean absolute error of the sum over components per dimer.
            elst_MAE_t (torch.Tensor): Mean absolute error for the electrostatics component.
            exch_MAE_t (torch.Tensor): Mean absolute error for the exchange component.
            indu_MAE_t (torch.Tensor): Mean absolute error for the induction component.
            disp_MAE_t (torch.Tensor): Mean absolute error for the dispersion component.
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
        Evaluate the model on FSAPT fragment energies for a single process and compute component-wise and total MAE metrics.
        
        This method runs the model in evaluation mode over `dataloader`, assembles per-pair energies by mapping short-range predicted edges into the full edge set, sums contributions for the fragment pairs defined by each dimer, and computes mean-squared loss (or provided loss_fn) plus mean absolute errors for total and each SAPT component (electrostatics, exchange, induction, dispersion).
        
        Parameters:
            dataloader: Iterable of batches providing fused/FSAPT batch tensors and index mappings required for assembling fragment pair contributions (must include attributes used in the routine such as e_ABsr_source/target, e_ABfull_source/target, frag1_ind, frag2_ind, total_charge_A, and y).
            loss_fn: Optional loss function applied to (preds, labels). If None, mean squared error is used.
            rank_device: Device on which tensors and model are placed for evaluation.
        
        Returns:
            total_loss (float): Sum of batch losses accumulated over the dataset (uses .item() per batch).
            total_MAE_t (torch.Tensor): Mean absolute error of the summed component predictions per dimer.
            elst_MAE_t (torch.Tensor): Mean absolute error for the electrostatics component.
            exch_MAE_t (torch.Tensor): Mean absolute error for the exchange component.
            indu_MAE_t (torch.Tensor): Mean absolute error for the induction component.
            disp_MAE_t (torch.Tensor): Mean absolute error for the dispersion component.
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
        """
        Performs a training pass over `dataloader`, updates model parameters, and computes per-component and total mean absolute errors.
        
        Runs the model in training mode for one epoch: for each batch it computes the loss (using `loss_fn` if provided, otherwise MSE), backpropagates, steps `optimizer`, and accumulates absolute errors for electrostatics, exchange, induction, and dispersion components. After iterating batches, reduces losses and error counts across distributed processes (if any) and returns aggregated metrics.
        
        Parameters:
            rank (int): Process rank in distributed training.
            dataloader (Iterable): Iterable of training batches; each batch must have `.to(device)` and `.y` target tensor.
            loss_fn (callable or None): Loss function accepting (preds, targets). If `None`, mean squared error is used.
            optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
            rank_device (torch.device): Device to place tensors and reductions on for this rank.
            scheduler (torch.optim.lr_scheduler._LRScheduler or None): Optional LR scheduler stepped once after the epoch.
        
        Returns:
            total_loss (torch.Tensor): Sum of batch losses (scalar tensor) reduced across processes.
            total_MAE_t (torch.Tensor): Mean absolute error of the total energy per pair (scalar tensor, CPU).
            elst_MAE_t (torch.Tensor): Mean absolute error for electrostatics component per pair (scalar tensor, CPU).
            exch_MAE_t (torch.Tensor): Mean absolute error for exchange component per pair (scalar tensor, CPU).
            indu_MAE_t (torch.Tensor): Mean absolute error for induction component per pair (scalar tensor, CPU).
            disp_MAE_t (torch.Tensor): Mean absolute error for dispersion component per pair (scalar tensor, CPU).
        """
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
        """
        Evaluate the model on a dataloader and aggregate loss plus overall and per-component mean absolute errors (MAE) across distributed processes.
        
        Parameters:
            rank: Process rank used for distributed evaluation.
            dataloader: Iterable of batches to evaluate.
            loss_fn: Optional loss function accepting flattened predictions and targets; if None, uses mean squared error on component errors.
            rank_device: torch.device to place intermediate tensors for this process.
        
        Returns:
            total_loss (torch.Tensor): Sum-reduced scalar loss across all processes (on rank_device).
            total_MAE_t (torch.Tensor): Overall MAE per predicted component (total absolute error divided by total element count), CPU tensor.
            elst_MAE_t (torch.Tensor): MAE for the electrostatics component, CPU tensor.
            exch_MAE_t (torch.Tensor): MAE for the exchange component, CPU tensor.
            indu_MAE_t (torch.Tensor): MAE for the induction component, CPU tensor.
            disp_MAE_t (torch.Tensor): MAE for the dispersion component, CPU tensor.
        """
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
        Train the model using Distributed Data Parallel (DDP) across multiple processes and GPUs.
        
        Performs setup for distributed training (when world_size > 1), constructs distributed data loaders, runs pre-training evaluation, and executes epoch-wise training and validation loops with optional learning-rate decay. If a lower validation loss is found on rank 0, the current model state is saved to self.model_save_path (if set). After training, distributed resources are cleaned up when used.
        
        Parameters:
            rank (int): Process rank within the distributed group (0..world_size-1).
            world_size (int): Total number of processes participating in DDP.
            train_dataset (Dataset): Training dataset instance.
            test_dataset (Dataset): Validation/test dataset instance.
            n_epochs (int): Number of training epochs to run.
            batch_size (int): Batch size per process.
            lr (float): Initial learning rate for the optimizer.
            pin_memory (bool): Passed to data loader to enable pin_memory.
            num_workers (int): Number of worker processes for data loading.
            lr_decay (float | None): If provided, used as the decay_rate for an inverse-time LR scheduler; if None no scheduler is used.
        
        Side effects:
            - Moves and (when world_size > 1) wraps self.model in torch.nn.parallel.DistributedDataParallel.
            - May save a model checkpoint to self.model_save_path from rank 0 when validation improves.
            - Calls self.__setup and self.__cleanup for distributed initialization/teardown when applicable.
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
                                    "use_atom_props": cpu_model.use_atom_props,
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
        Train the model in a single-process training loop and update self.model with the best-performing weights.
        
        Performs optional model compilation, constructs data loaders, optimizer, and scheduler, runs a single pre-training evaluation, then iterates epoch-wise training and evaluation. Tracks the lowest test loss, saves the best model to self.model_save_path if provided, and replaces self.model with the best CPU-copied state before moving it back to the configured device.
        
        Parameters:
            train_dataset: Dataset or Subset
                Training dataset (may be a Subset wrapper); supports FSAPT and fused dataset types.
            test_dataset: Dataset or Subset
                Validation/test dataset matching train_dataset format.
            n_epochs: int
                Number of training epochs to run.
            batch_size: int
                Batch size for both training and validation loaders.
            lr: float
                Initial learning rate for the optimizer.
            pin_memory: bool
                Passed to DataLoader to enable pinned memory for CUDA transfers.
            num_workers: int
                Number of worker processes for the DataLoader.
            lr_decay: float or None, optional
                Decay factor used to construct an inverse-time decay scheduler; if None no scheduler is used.
            skip_compile: bool, optional
                If True, skip torch.compile step; otherwise attempt to compile the model before training.
            transfer_learning: bool, optional
                If True, use transfer-learning-specific training/evaluation routines and logging (reduced metric set).
        
        Side effects:
            - Trains and modifies self.model parameters.
            - May save the best model checkpoint to self.model_save_path.
            - Moves models and sample batches to self.device and performs CUDA cache cleanup during training.
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
                                "use_atom_props": cpu_model.use_atom_props,
                            },
                        },
                        self.model_save_path,
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
        Train the APNet3-fused model on a dataset, optionally using distributed training, and save training results to model_path.
        
        This method prepares train/test splits (or accepts a two-element [train, test] list), configures device and threading, optionally freezes/uses precomputed classical components, and launches either a single-process training loop or a multi-process DDP training routine. It prints model and training hyperparameters and sets up data loaders and learning-rate decay as requested.
        
        Parameters:
            dataset (optional): Dataset object or a two-element list [train_dataset, test_dataset]. If omitted, self.dataset must be set prior to calling.
            n_epochs (int): Number of training epochs.
            lr (float): Initial learning rate.
            split_percent (float): Fraction of data used for training when a single dataset is provided.
            model_path (str): Path where training outputs and the best model will be saved.
            shuffle (bool): Whether to shuffle dataset indices before splitting.
            dataloader_num_workers (int): Number of worker processes for data loading.
            world_size (int): Number of processes for distributed training; if >1, spawns DDP workers.
            omp_num_threads_per_process (int): Value to set OMP_NUM_THREADS for each process.
            lr_decay (optional): Learning-rate scheduler configuration (passed through to training routines).
            random_seed (int): Seed for NumPy RNG to control dataset shuffling/splitting.
            skip_compile (bool): If True, skip model compilation step when using single-process training.
            transfer_learning (bool): If True, enable transfer-learning training routines or behaviors.
        
        Raises:
            ValueError: If no dataset is provided (neither via argument nor self.dataset).
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
        Freeze model parameters except the SAPT component readout layers.
        
        Sets requires_grad = False for all parameters of self.model, and sets requires_grad = True
        for parameters whose name contains "readout" and whose top-level module name ends with
        one of: "elst", "exch", "indu", "disp".
        """
        for name, param in self.model.named_parameters():
            term = name.split('.')[0]
            if "readout" in name and term[-4:] in ['elst', 'exch', 'indu', 'disp']:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return

    def unfreeze_all_parameters(self):
        """
        Unfreeze all parameters of the AP3 model so they become trainable.
        
        Sets `requires_grad = True` on every parameter in `self.model`.
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        return