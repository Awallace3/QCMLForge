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
from apnet_pt.util import scatter_sum_compile
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
    Compute an inverse-time learning-rate scaling factor.

    Parameters:
        step (float): Current step or epoch number.
        initial_lr (float): Base learning rate before decay.
        decay_steps (float): Normalization factor for the step (controls decay speed).
        decay_rate (float): Decay coefficient controlling how quickly the learning rate decreases.
        staircase (bool): If True, use floor(step / decay_steps) so the decay is applied in discrete intervals.

    Returns:
        float: The scaled learning rate computed as initial_lr / (1 + decay_rate * p), where
        p = step / decay_steps (or floor(step / decay_steps) when `staircase` is True).
    """
    p = step / decay_steps
    if staircase:
        p = np.floor(p)
    return initial_lr / (1 + decay_rate * p)


class InverseTimeDecayLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, initial_lr, decay_steps, decay_rate):
        """
        Create a learning-rate scheduler that applies inverse-time decay to the optimizer.

        Parameters:
            optimizer: Optimizer whose learning rate will be scheduled.
            initial_lr (float): Reference learning rate used by the decay function.
            decay_steps (int): Number of steps used to scale the decay progression.
            decay_rate (float): Decay coefficient controlling the rate of reduction.

        Behavior:
            The scheduler scales the optimizer's base learning rates by an inverse-time
            decay factor computed from the provided parameters.
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
    Compute a multiplicative learning-rate factor that decays exponentially with epoch and is floored by a minimum learning rate.

    Parameters:
        epoch (int): Current epoch index used to compute decay exponent.
        decay_factor (float): Per-epoch multiplicative decay (e.g., 0.9 for 10% decay per epoch).
        initial_lr (float): Reference learning rate before decay.
        min_lr (float): Minimum allowed learning rate; the decayed value will not fall below this.

    Returns:
        float: The factor (in (0, 1]) to multiply `initial_lr` by; equals max(initial_lr * decay_factor**epoch, min_lr) divided by `initial_lr`.
    """
    lr = initial_lr * (decay_factor**epoch)
    return max(lr, min_lr) / initial_lr


class AsymptoticDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_coefficient, last_epoch=-1):
        """
        Create an AsymptoticDecayLR scheduler that decays each base learning rate asymptotically.

        Parameters:
            optimizer (torch.optim.Optimizer): Wrapped optimizer whose learning rate will be scheduled.
            decay_coefficient (float): Positive coefficient controlling the decay speed; learning rate at epoch t is computed as base_lr / (1 + t / decay_coefficient).
            last_epoch (int, optional): The index of the last epoch. Defaults to -1, which starts scheduling from epoch 0.
        """
        self.decay_coefficient = decay_coefficient
        super(AsymptoticDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rates using asymptotic decay based on the current epoch.

        Returns:
            list[float]: Learning rates for each parameter group, computed as
            base_lr / (1 + last_epoch / decay_coefficient).
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
        Create an Envelope configured by the given exponent and precompute polynomial coefficients used by the envelope function.

        Parameters:
            exponent (int or float): Exponent that controls the smooth cutoff shape; used to compute internal parameter `p` and polynomial coefficients `a`, `b`, and `c`.
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
        Compute a smooth cutoff envelope divided by r for normalized distances.

        Parameters:
            inputs (torch.Tensor): Elementwise distances scaled so that 1.0 corresponds to the cutoff; may be any shape.

        Returns:
            torch.Tensor: Envelope values with the same shape as `inputs`: for elements < 1 the value is
            1/inputs + a * inputs^(p-1) + b * inputs^p + c * inputs^(p+1), and for elements >= 1 the value is 0.
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
        Create a DistanceLayer that encodes interatomic distances into a learnable radial basis with a smooth cutoff.

        Parameters:
            num_radial (int): Number of radial basis functions.
            r_cut (float): Cutoff distance used to scale inputs; distances are mapped relative to this cutoff.
            envelope_exponent (int): Exponent controlling the smooth envelope that damps the basis near the cutoff.

        Notes:
            - A precomputed inverse cutoff (1 / r_cut) and an Envelope instance are stored for use in forward.
            - Frequency parameters are initialized to pi * [1..num_radial] and are learnable.
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
        Compute enveloped sinusoidal radial-basis embeddings for interatomic distances.

        Parameters:
            inputs (torch.Tensor): Distances (same units as the layer's cutoff) with arbitrary shape.

        Returns:
            torch.Tensor: Radial-basis embeddings with shape inputs.shape + (num_radial,), equal to envelope(inputs/r_cut) * sin(frequencies * (inputs/r_cut)). The envelope is zero for distances greater than or equal to the cutoff, so embeddings are zero there.
        """
        d_scaled = inputs * self.inv_cutoff
        d_scaled = d_scaled.unsqueeze(-1)
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)


def unwrap_model(model):
    """
    Retrieve the underlying nn.Module when a model is wrapped in DistributedDataParallel.

    Returns:
        nn.Module: The unwrapped module if `model` is a DDP wrapper, otherwise `model` itself.
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
        Initialize the APNet3 atom-type MPNN for predicting SAPT-like dimer interaction components.

        Parameters:
            dimer_prop_model: Optional pre-trained DimerProp model whose parameters will be frozen when provided.
            n_message: Number of message-passing iterations.
            n_rbf: Number of radial basis functions used for distance embeddings.
            n_neuron: Base hidden size used to construct internal MLP layers.
            n_embed: Size of the atom-type embedding vectors.
            r_cut_im: Cutoff distance for intramonomer (intra-monomer) radial basis embeddings.
            r_cut: Cutoff distance for intermonomer (inter-monomer) radial basis embeddings.
            return_hidden_states: If True, forward will return intermediate hidden-state tensors for inspection.
            use_precomputed_classical: If True, the model will expect and prefer externally provided classical (Elst/Eind) terms.
            use_atom_props: If True, include additional per-atom properties (e.g., HFVR or VW features) when building pair features.
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
        Constructs a torch.nn.Sequential feed-forward block from given layer sizes and activation modules.

        Parameters:
            layer_nodes (Sequence[int]): Sequence of layer output sizes; the first entry is the size of the initial (lazy) linear layer and subsequent entries are sizes for each following linear layer.
            activations (Sequence[Optional[nn.Module]]): Sequence of activation modules (or None) aligned with layers; the first activation follows the initial lazy linear layer, and each subsequent non-None entry follows the corresponding Linear layer.

        Returns:
            nn.Sequential: A Sequential module containing a LazyLinear for the first layer followed by the specified activations and Linear layers in order.
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
        Construct per-edge invariant message vectors by combining node states and radial basis features.

        Parameters:
            h0 (Tensor): Initial node embeddings with shape [N, n_embed].
            h (Tensor): Current node embeddings with shape [N, n_embed].
            rbf (Tensor): Radial basis features for edges with shape [E, n_rbf].
            e_source (LongTensor): Source node indices for each edge with shape [E].
            e_target (LongTensor): Target node indices for each edge with shape [E].

        Returns:
            Tensor: Per-edge message tensor with shape [E, n_embed*4*n_rbf + n_embed*4 + n_rbf]. Each row is the concatenation of
            [h0_source, h0_target, h_source, h_target, projections_of_these_on_rbf (flattened), rbf].
            If E == 0, returns a zero-sized tensor with shape [0, n_embed*4*n_rbf + n_embed*4 + n_rbf].
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
        Builds per-edge pair feature vectors by gathering source and target node features, per-node scalars, and edge radial-basis features.

        Parameters:
            hA (Tensor): Node feature tensor for monomer A with shape [n_nodes_A, n_h].
            hB (Tensor): Node feature tensor for monomer B with shape [n_nodes_B, n_h].
            qA (Tensor): Per-node scalar/vector features for monomer A with shape [n_nodes_A, ...].
            qB (Tensor): Per-node scalar/vector features for monomer B with shape [n_nodes_B, ...].
            rbf (Tensor): Edge radial-basis features with shape [n_edges, n_rbf].
            e_source (LongTensor): Source node indices for each edge (length n_edges).
            e_target (LongTensor): Target node indices for each edge (length n_edges).

        Returns:
            Tensor: Concatenated edge features with shape [n_edges, n_h + n_h + qA_dim + qB_dim + n_rbf],
            formed as [hA[source], hB[target], qA[source], qB[target], rbf] for each edge.
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
        Construct per-edge pair features by selecting source and target atom descriptors and concatenating them with radial-basis features.

        Parameters:
            e_source: LongTensor of source atom indices for each edge; used to index into per-atom tensors.
            e_target: LongTensor of target atom indices for each edge; used to index into per-atom tensors.
            hfvrA, hfvrB, vwA, vwB: Per-atom property tensors that are included only when `self.use_atom_props` is True.

        Returns:
            A tensor of per-edge concatenated features with shape [num_edges, F], where each row contains:
            - source atom embedding,
            - target atom embedding,
            - source charge,
            - target charge,
            - optional per-atom properties (hfvr and vw) for source and target when `use_atom_props` is True,
            - the radial-basis features `rbf`.
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
        Compute pairwise displacement vectors and their Euclidean distances for edges defined by index tensors.

        Parameters:
            RA (Tensor): Coordinates of atoms in set A with shape [nA, 3].
            RB (Tensor): Coordinates of atoms in set B with shape [nB, 3].
            e_source (LongTensor): 1-D tensor of source atom indices (into RA) for each edge with shape [E].
            e_target (LongTensor): 1-D tensor of target atom indices (into RB) for each edge with shape [E].

        Returns:
            dR (Tensor): 1-D tensor of Euclidean distances for each edge with shape [E]; distances computed as the L2 norm of the displacement and clamped to be at least 1e-10 before square-rooting to avoid numerical issues.
            dR_xyz (Tensor): Tensor of displacement vectors RB[e_target] - RA[e_source] with shape [E, 3].
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
        Compute component-wise readout predictions from per-node feature representations.

        Parameters:
            H: Node feature tensor used as input to the component readout heads.

        Returns:
            Tensor of concatenated component predictions with the four components ordered as
            electrostatics, exchange, induction, and dispersion concatenated along feature axis (dim=1).
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
        Compute per-dimer SAPT-like energy components and optional hidden-state tensors from a fused dimer batch.

        Parameters:
            batch: fused batch object containing atom and edge data for a set of dimers. Required attributes used:
                ZA, ZB (LongTensor): atomic numbers for monomer A and B (concatenated across batch).
                RA, RB (FloatTensor): atomic coordinates for monomer A and B.
                e_ABsr_source, e_ABsr_target (LongTensor): short-range intermolecular edge source/target indices.
                e_ABlr_source, e_ABlr_target (LongTensor): long-range intermolecular edge source/target indices.
                e_AA_source, e_AA_target (LongTensor): intramonomer A edge source/target indices.
                e_BB_source, e_BB_target (LongTensor): intramonomer B edge source/target indices.
                dimer_ind (LongTensor): mapping from short-range intermolecular edges to dimer indices.
                dimer_ind_full (LongTensor): mapping from classical per-atom contributions to dimer indices.
                total_charge_A (Tensor): used to infer number of dimers (only its length is used).

        Returns:
            If self.use_precomputed_classical is True:
                (E_output, E_sr, 0, 0, hAB, hBA)
                  - E_output (FloatTensor [ndimer, ?]): final per-dimer energy (model short-range sum; classical terms not provided).
                  - E_sr (FloatTensor [n_pairs, 4]): per-pair short-range predicted components prior to dimer aggregation and component padding.
                  - 0, 0: placeholders for electrostatic and induction classical components.
                  - hAB, hBA (FloatTensor): per-pair learned pair-feature tensors (A->B and B->A).

            If self.use_precomputed_classical is False and self.return_hidden_states is False:
                (E_output, E_sr, E_elst, E_ind, hAB, hBA)
                  - E_output (FloatTensor [ndimer, ?]): final per-dimer energy (short-range + classical electrostatic + classical induction).
                  - E_sr (FloatTensor [n_pairs, 4]): per-pair short-range predicted components.
                  - E_elst (FloatTensor [n_atoms?] or per-atom electrostatic contributions): classical electrostatic contributions (before dimer aggregation/padding).
                  - E_ind (FloatTensor [n_atoms?] or per-atom induction contributions): classical induction contributions (before dimer aggregation/padding).
                  - hAB, hBA: per-pair learned pair-feature tensors.

            If self.use_precomputed_classical is False and self.return_hidden_states is True:
                (E_output, E_sr_dimer, E_elst, E_ind, hAB, hBA, cutoff_3)
                  - E_sr_dimer (FloatTensor [ndimer, ?]): short-range components after summation into dimers.
                  - cutoff_3 (FloatTensor [n_pairs, 1] or [n_pairs]): 1 / (r^3) factor used to scale certain components (returned for inspection).

        Notes:
            - The exact shapes of returned tensors depend on model configuration (number of readout outputs, batching, and padding).
            - The method requires that the batch provides the listed attributes; behaviour is undefined if attributes are missing.
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
        cutoff_3 = 1.0 / (dR_sr**3)
        # cutoff_4 = (1.0 / (dR_sr**4))
        # cutoff_5 = (1.0 / (dR_sr**5))
        # cutoff_6 = (1.0 / (dR_sr**6))
        # cutoff_12 = (1.0 / (dR_sr**12))
        E_sr[:, 0] *= cutoff_3
        E_sr[:, 1] *= cutoff_3**2
        E_sr[:, 2] *= cutoff_3
        E_sr[:, 3] *= cutoff_3
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
                cutoff_3,
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
        Construct the APNet3_AtomType_Model, configure device and dataset classes, and load or instantiate model and supporting components.

        Parameters:
            dataset: Optional preconstructed dataset or dataset list to use; if None and ignore_database_null is False the constructor will attempt to build dataset(s) from disk.
            atom_type_model: Optional AtomTypeParamModel instance to use (if omitted a default AtomTypeParamModel is created internally).
            dimer_prop_model: Optional pre-instantiated DimerProp model to use instead of constructing/loading one.
            am_dimer_param_model: Optional auxiliary dimer parameter model passed through to the instance.
            pre_trained_model_path: Path to a saved APNet3_AtomType_MPNN checkpoint; when provided the APNet3 model and its config are loaded and other model-related init parameters are ignored.
            dimer_prop_model_pre_trained_path: Path to a saved DimerProp checkpoint to load instead of using or constructing a DimerProp instance.
            n_message: Number of message-passing steps for newly constructed models (ignored when loading from pre_trained_model_path).
            n_rbf: Number of radial basis functions for newly constructed models (ignored when loading from pre_trained_model_path).
            n_neuron: Hidden neuron count for newly constructed models (ignored when loading from pre_trained_model_path).
            n_embed: Atom-type embedding dimension for newly constructed models (ignored when loading from pre_trained_model_path).
            r_cut_im: Intramonomer distance cutoff used for dataset construction and model configuration.
            r_cut: Intermonomer distance cutoff used for dataset construction and model configuration.
            use_GPU: If not False and CUDA is available, selects CUDA device; set to False to force CPU.
            ignore_database_null: When False and dataset is None, triggers automatic dataset construction from disk according to ds_spec_type and ds_class_type.
            ds_spec_type: Dataset specification type (internal dataset selection logic uses this to decide dataset layout/splits).
            ds_root: Root directory for datasets.
            ds_max_size: Optional cap on dataset size after construction (slices constructed dataset(s) to this length).
            ds_atomic_batch_size: Atomic batching parameter passed to dataset constructors.
            ds_batch_size: Batch size passed to dataset constructors.
            ds_force_reprocess: Force datasets to reprocess raw data when True.
            ds_skip_process: Skip processing of raw inputs when True (dataset-dependent).
            ds_skip_compile: Skip compilation steps for dataset preprocessing when True.
            ds_in_memory: If True, dataset is loaded into memory.
            ds_num_devices: Number of devices advertised to dataset constructors.
            ds_datapoint_storage_n_objects: Dataset internal storage tuning parameter.
            ds_prebatched: Dataset is already pre-batched (passed through).
            ds_random_seed: Random seed used by dataset constructors.
            ds_type: Logical dataset content type; e.g., "total_component_energies" or "fsapt_energies".
            print_lvl: Verbosity level forwarded to datasets.
            ds_qcel_molecules: Optional molecule lists (or split lists) used to restrict dataset contents; a two-entry list of lists triggers split dataset handling.
            ds_energy_labels: Optional energy label filters passed to dataset constructors.
            use_precomputed_classical: When True, dataset/model assume classical (Elst/Ind) components are precomputed and provided as inputs.
            ds_class_type: Dataset backend type, must be either "pt" or "lmdb"; selects the dataset class implementation.
            use_atom_props: If True, atom-level properties (e.g., HF/VR, van-der-Waals) are included in model inputs when supported.

        Raises:
            ValueError: If ds_class_type is not "pt" or "lmdb".
            NotImplementedError: If ds_type == "fsapt_energies" and ds_class_type == "pt" (PT backend for FSAPT is not implemented).

        Returns:
            None
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
                Construct and return a dataset instance configured for the model.

                Chooses between the configured dataset_class (when precomputed classical terms are used)
                and the legacy ap2_fused_module_dataset otherwise, passing through dataset configuration
                options (cuts, sizes, batching, seed, storage and device/memory flags).

                Parameters:
                    fp: Override for the dataset `force_reprocess` flag; passed through to the dataset constructor.

                Returns:
                    A dataset object instance created by either `self.dataset_class(...)` or `ap2_fused_module_dataset(...)`,
                    configured with the model's dataset-related parameters.
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
                Constructs and returns train/test dataset instances based on current configuration and whether precomputed classical terms or FSAPT data are used.

                Parameters:
                        fp (bool): Whether to force reprocessing of raw data when creating the datasets.

                Returns:
                        list: A two-element list [train_dataset, test_dataset] of dataset objects configured with the model's dataset parameters, ready for training and evaluation.
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
        Run the model in evaluation mode over the configured dataset, performing a forward pass for each batch.

        This method sets the model to eval() and iterates through self.dataset, moving each batch to self.device and calling the model. The forward outputs from each batch are not returned or stored by this method.
        """
        self.model.eval()
        for batch in self.dataset:
            batch = batch.to(self.device)
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.model(batch)
        return

    def compile_model(self):
        """
        Move the model to the configured device and enable TorchDynamo/TorchCompile optimizations.

        This sets a few TorchDynamo configuration flags for dynamic shapes and output-shape capture, then replaces `self.model` with a compiled, dynamically-traced version suitable for runtime acceleration.

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
        Set every learnable parameter of the wrapped model to a constant value.

        This runs a single example batch through the model to ensure any lazy parameters
        are created and moved to the configured device, then assigns `value` to all
        model weights. Intended for debugging and deterministic behavior.

        Parameters:
            value (float): Constant value to assign to all learnable parameters.

        Returns:
            None
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
        Load AP2 pretrained weights into this APNet3 model instance.

        Parameters:
            ap2_model_path (str or os.PathLike, optional): Path to an AP2 checkpoint file. Ignored if `model_id` is provided.
            am_model_path: Reserved for compatibility (not used).
            model_id (str or int, optional): Identifier for a bundled AP2 checkpoint; when provided, the corresponding packaged checkpoint is loaded instead of `ap2_model_path`.

        Returns:
            self: The same model instance with weights loaded from the checkpoint.

        Raises:
            ValueError: If neither `ap2_model_path` nor `model_id` is provided.

        Notes:
            If checkpoint parameter keys include the prefix `_orig_mod.`, that prefix will be removed to match the current model's state dict keys before loading.
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

    def load_ap2_pretrained_weights(self, ap2_model_path):
        """
        Load weights from an AP2 checkpoint into the current AP3 model, copying parameters for matching shared layers.

        Parameters:
            ap2_model_path (str): Path to the AP2 checkpoint file containing a 'model_state_dict'.

        Returns:
            self: The current instance with the AP3 model's state updated with any matching AP2 parameters.
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
        Construct a fused dimer batch from QCElemental-style molecule objects and move it to the model device.

        Parameters:
            mols (list): Iterable of QCElemental-style dimer objects to convert into fused data.
            batch_size (int): Ignored in this implementation; present for API compatibility.
            r_cut (float): Inter-monomer distance cutoff passed to the conversion routine.
            r_cut_im (float): Intra-monomer distance cutoff passed to the conversion routine.

        Returns:
            dimer_batch: A fused dimer batch (output of ap3_fused_collate_update_no_target) with tensors moved to self.device.
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
        Enable or disable returning internal hidden-state tensors from the wrapped model.

        Parameters:
            value (bool): If `True`, the model will include internal hidden-state tensors in its outputs; if `False`, hidden states will be omitted.

        Returns:
            self: The object instance, allowing method chaining.
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
        Assemble per-dimer pairwise energy matrices from flat pair lists and per-pair component arrays.

        Parameters:
            inp_batch: batch object with indexing tensors:
                - e_ABsr_source, e_ABsr_target: source/target atom indices for short-range pairs
                - e_ABfull_source, e_ABfull_target: source/target atom indices for full (MTP) pairs
                - dimer_ind_full: per-atom dimer index used to group atoms into dimers
                - indA, indB: per-atom monomer indices mapping atoms to their dimer membership
            E_sr_dimer: unused in this function (kept for signature compatibility)
            E_sr: iterable of 4-component short-range pair energies aligned with e_ABsr_source/target
            E_elst_mtp: iterable of electrostatic (MTP) pair energies aligned with e_ABfull_source/target
            E_ind_mtp: iterable of induction pair energies aligned with e_ABfull_source/target

        Returns:
            pair_energies_batch (list of numpy.ndarray): one array per dimer with shape (4, nA, nB),
            where nA and nB are the number of atoms in monomers A and B for that dimer.
            The four channels correspond to energy components assembled from the inputs (electrostatics, exchange, induction, dispersion),
            accumulated into the appropriate atom-atom entries.
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
        Assemble pairwise energies using pure PyTorch operations to preserve gradients.

        Returns a list of tensors, one per dimer, each with shape [4, size_A, size_B]
        containing the pairwise interaction energies for each SAPT component.
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
        Assemble per-dimer atom-pair matrices of electrostatic and induction MTP energies from flattened edge lists.

        Parameters:
            inp_batch: Batch object containing edge mapping fields:
                - "e_ABfull_source" and "e_ABfull_target": source/target edge index tensors aligning with energy lists.
                - attributes `dimer_ind_full` (per-edge dimer index), `indA` (per-edge index into concatenated monomer A atoms),
                  and `indB` (per-edge index into concatenated monomer B atoms).
            E_elst_mtp (iterable): Per-edge electrostatic MTP energies aligned with inp_batch edge order.
            E_ind_mtp (iterable): Per-edge induction MTP energies aligned with inp_batch edge order.

        Returns:
            pair_elst_batch (list of ndarray): One 2D NumPy array per dimer of shape (size_A, size_B) with assembled electrostatic pair energies.
            pair_ind_batch (list of ndarray): One 2D NumPy array per dimer of shape (size_A, size_B) with assembled induction pair energies.
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
        Compute model predictions for a list of QCElemental dimer objects and optionally return pairwise or hidden-state details.

        Parameters:
            mols (Sequence): Iterable of QCElemental dimer objects to predict.
            batch_size (int): Number of dimers to process per model batch.
            r_cut (float | None): Intermolecular distance cutoff to use; defaults to self.model.r_cut when None.
            r_cut_im (float | None): Intramolecular distance cutoff to use; defaults to self.model.r_cut_im when None.
            verbose (bool): If True, print progress messages about skipped invalid dimers and batch progress.
            return_pairs (bool): If True, also return pairwise short-range energy decompositions for each dimer.
            return_classical_pairs (bool): If True, return per-dimer classical pairwise electrostatic and induction energies. Incompatible with return_pairs.

        Returns:
            If the model is configured to return hidden states:
                tuple:
                    predictions (ndarray): Array of shape (N, 4) with four-component dimer energies for each input dimer.
                    h_ABs (list): Per-batch tensors of hidden-state matrices for A->B pairs.
                    h_BAs (list): Per-batch tensors of hidden-state matrices for B->A pairs.
                    cutoffs (list): Per-batch cutoff tensors used by the model.
                    dimer_inds (list): Per-batch dimer index tensors.
                    ndimers (list): Per-batch tensors with the number of atoms in monomer A for each dimer.
            Else if return_pairs is True:
                tuple:
                    predictions (ndarray): Array of shape (N, 4) with four-component dimer energies.
                    pairwise_energies (list): List of per-dimer arrays containing pairwise short-range energy contributions.
            Else if return_classical_pairs is True:
                tuple:
                    predictions (ndarray): Array of shape (N, 4) with four-component dimer energies.
                    pairwise_elst_energies (list): List of per-dimer arrays of pairwise electrostatic energies.
                    pairwise_ind_energies (list): List of per-dimer arrays of pairwise induction energies.
            Else:
                predictions (ndarray): Array of shape (N, 4) with four-component dimer energies.

        Notes:
            - Invalid dimers detected during preprocessing are skipped and their prediction rows are filled with NaNs.
            - The function moves the internal dimer property model to the configured device before inference.
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
        Construct a sample batched input for the model from a QCElemental Molecule or a default dimer.

        Parameters:
            mol (qcel.models.Molecule | None): Molecule to convert into a fused batch. If `None`, a small default dimer example is used.
            r_cut (float): Inter-monomer distance cutoff (angstrom).
            r_cut_im (float): Intra-monomer distance cutoff (angstrom).

        Returns:
            A processed batch (the same structure produced by `_qcel_example_input`) containing a single example ready for model prediction or training.
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
        Initialize the distributed process group for multi-process training and fix the random seed for reproducibility.

        Parameters:
            rank (int): The rank of the current process within the distributed group.
            world_size (int): Total number of processes participating in the distributed group.
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
        Terminate the current PyTorch distributed process group and release its resources.

        This calls torch.distributed.destroy_process_group() to clean up the initialized distributed group for the current process.
        """
        dist.destroy_process_group()

    def __train_batches_single_proc(
        self, dataloader, loss_fn, optimizer, rank_device, scheduler
    ):
        """
        Run one training epoch over `dataloader` on a single process and return loss and component MAEs.

        Parameters:
            dataloader: Iterable of training batches already collated for the model.
            loss_fn: Optional loss function; if `None`, mean squared error between predicted and target components is used.
            optimizer: Optimizer used to update model parameters.
            rank_device: Device to move batches and model tensors to (e.g., torch.device).
            scheduler: Optional learning-rate scheduler; `step()` is called once after the epoch when provided.

        Returns:
            total_loss (float): Sum of per-batch loss values accumulated over the epoch.
            total_MAE_t (torch.Tensor): Mean absolute error of the total predicted dimer energy (sum of components).
            elst_MAE_t (torch.Tensor): Mean absolute error for the electrostatics component.
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
        Evaluate the model over a dataloader and compute aggregate loss plus component-wise mean absolute errors.

        When use_precomputed_classical is enabled, the classical electrostatics and induction components from the batch are subtracted from the labels before loss/MAE computation.

        Parameters:
            dataloader: Iterable yielding batches with .to(device) and attributes `y` and, when applicable, `E_classical_elst`, `E_classical_ind`.
            loss_fn (callable or None): Loss function accepting (preds, labels). If None, mean squared error is used.
            rank_device (torch.device): Device to move each batch to for evaluation.

        Returns:
            tuple: (total_loss, total_MAE, elst_MAE, exch_MAE, indu_MAE, disp_MAE)
                total_loss (float): Sum of per-batch losses over the dataloader.
                total_MAE (torch.Tensor): Mean absolute error of the total predicted energy per sample.
                elst_MAE (torch.Tensor): Mean absolute error for the electrostatics component.
                exch_MAE (torch.Tensor): Mean absolute error for the exchange component.
                indu_MAE (torch.Tensor): Mean absolute error for the induction component.
                disp_MAE (torch.Tensor): Mean absolute error for the dispersion component.
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
        Train the model for one pass over `dataloader` in single-process transfer-learning mode.

        Runs forward/backward updates for each batch, accumulates the scalar training loss sum and the per-datapoint mean absolute error for the epoch, and steps the optional scheduler once after the loop.

        Parameters:
            dataloader: Iterable yielding batches compatible with the model; each batch must expose `.to(device)` and contain `y` target values.
            loss_fn: Optional loss function; if `None`, mean squared error between predictions and targets is used.
            optimizer: Optimizer used for parameter updates.
            rank_device: Device to move batches to (e.g., torch.device).
            scheduler: Optional learning-rate scheduler to step once after the epoch.

        Returns:
            total_loss (float): Sum of individual batch loss scalars accumulated over the loop.
            total_MAE_t (torch.Tensor): Mean absolute error across all datapoints processed (scalar tensor).
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
        Evaluate the model on `dataloader` using transfer-style component aggregation and compute loss and mean absolute error.

        Parameters:
            dataloader: Iterable of batches; each batch must support `.to(device)` and contain `.y` target values.
            loss_fn: Callable(loss_pred, target) -> scalar loss or None to use mean squared error on component-summed predictions.
            rank_device: Device to move batches to for evaluation.

        Returns:
            total_loss (float): Sum of per-batch loss values accumulated over the dataloader.
            total_MAE_t (torch.Tensor): Mean absolute error (scalar tensor) of (sum of predicted components - target) across all examples.
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
        Perform a single-process training epoch for FSAPT fragment energies and return training loss and MAE metrics.

        Trains the wrapped model over the provided dataloader using the given optimizer and optional scheduler. For each batch, the method assembles fragment-level energies by summing pairwise contributions (using fragment index lists in the batch), computes the component-wise loss (using loss_fn when provided, otherwise mean squared error), backpropagates, and steps the optimizer. If a scheduler is supplied, it is stepped once after the epoch.

        Parameters:
            dataloader: Iterable that yields batches with attributes expected by the model (including fragment index lists, pair edge indices, and labels).
            loss_fn: Optional loss callable(pref: (preds, labels) -> scalar). If None, mean squared error is used.
            optimizer: Optimizer used to update model parameters.
            rank_device: Device on which tensors and model reside (e.g., torch.device).
            scheduler: Optional learning-rate scheduler to step after the epoch.

        Returns:
            total_loss (float): Sum of batch losses accumulated over the epoch.
            total_MAE_t (torch.Tensor): Mean absolute error of the total (sum over components) per-dimer predictions.
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
        Run a single-process evaluation pass over a dataloader computing FSAPT fragment-level prediction errors and loss.

        Parameters:
            dataloader: An iterable of batched FSAPT examples containing pair-edge indices, fragment index lists, labels, and any precomputed classical terms.
            loss_fn (callable or None): Loss function to apply to batch predictions and labels; if None, mean squared error is computed.
            rank_device (torch.device): Device where model inputs and tensors are placed for evaluation.

        Returns:
            total_loss (float): Sum of per-batch loss values accumulated over the dataloader.
            total_MAE_t (torch.Tensor): Mean absolute error of the summed four-component fragment predictions per dimer.
            elst_MAE_t (torch.Tensor): Mean absolute error for the electrostatic component.
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
        Run one training pass over `dataloader`, perform optimizer updates, and aggregate loss and mean absolute errors across distributed processes.

        Parameters:
            rank (int): Process rank in distributed training.
            dataloader (Iterable): Yields batches with attributes `.to(device)` and `.y` targets.
            loss_fn (callable or None): Loss function accepting flattened predictions and targets; if `None`, mean squared error is used.
            optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
            rank_device (torch.device): Device to which batches and aggregated tensors are moved.
            scheduler (torch.optim.lr_scheduler._LRScheduler or None): Optional learning-rate scheduler stepped after the epoch.

        Returns:
            total_loss (torch.Tensor): Sum of batch losses across all processes.
            total_MAE_t (torch.Tensor): Mean absolute error of total (summed) predicted vs target energies per component.
            elst_MAE_t (torch.Tensor): Mean absolute error for electrostatics component.
            exch_MAE_t (torch.Tensor): Mean absolute error for exchange component.
            indu_MAE_t (torch.Tensor): Mean absolute error for induction component.
            disp_MAE_t (torch.Tensor): Mean absolute error for dispersion component.
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
        Evaluate the model on a dataloader and compute aggregated loss and component-wise mean absolute errors across distributed workers.

        Parameters:
            rank (int): Process rank in distributed evaluation.
            dataloader (Iterable): Iterable of batches to evaluate; each batch must be moved to `rank_device`.
            loss_fn (callable or None): Loss function to reduce predictions and targets; if `None`, mean squared error on component errors is used.
            rank_device (torch.device): Device where model and batch tensors reside for this rank.

        Returns:
            total_loss (torch.Tensor): Sum of per-batch loss values aggregated across all ranks.
            total_MAE_t (torch.Tensor): Mean absolute error of the total (sum over components) prediction per sample, averaged across all ranks.
            elst_MAE_t (torch.Tensor): Mean absolute error for the electrostatics component across all ranks.
            exch_MAE_t (torch.Tensor): Mean absolute error for the exchange component across all ranks.
            indu_MAE_t (torch.Tensor): Mean absolute error for the induction component across all ranks.
            disp_MAE_t (torch.Tensor): Mean absolute error for the dispersion component across all ranks.
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
        Run distributed (or single-device) training and evaluation loop for this model, optionally saving the best checkpoint.

        Parameters:
            rank (int): Process rank in the distributed group (0 is the main process).
            world_size (int): Total number of processes participating in distributed training.
            train_dataset (Dataset): Dataset used for training.
            test_dataset (Dataset): Dataset used for evaluation.
            n_epochs (int): Number of training epochs to run.
            batch_size (int): Batch size per process.
            lr (float): Initial learning rate for the optimizer.
            pin_memory (bool): Whether data loaders should use pinned memory.
            num_workers (int): Number of worker subprocesses for data loading.
            lr_decay (float | None): If provided, decay rate used to build an inverse-time LR scheduler; if None, no scheduler is used.

        Notes:
            - Sets up and tears down the process group when world_size > 1.
            - Moves the model to the appropriate device for the given rank and wraps it with DistributedDataParallel when applicable.
            - Uses Adam as the optimizer and an optional inverse-time decay scheduler when `lr_decay` is provided.
            - Evaluates model performance before training and after each epoch; the best-evaluated model may be saved to `self.model_save_path`.
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
        Train the model on a single process using the provided training and test datasets.

        This method runs a full single-process training loop: it optionally compiles the model, constructs data loaders with the appropriate collate function, evaluates once before training, performs epoch-wise training and validation while tracking the best model by test loss, and restores/saves the best model when training finishes.

        Parameters:
            train_dataset: Dataset or Subset
                Dataset used for training.
            test_dataset: Dataset or Subset
                Dataset used for validation.
            n_epochs (int):
                Number of training epochs to run.
            batch_size (int):
                Batch size for both training and validation data loaders.
            lr (float):
                Initial learning rate used by the optimizer.
            pin_memory (bool):
                Passed to the data loaders' pin_memory argument.
            num_workers (int):
                Number of worker processes for data loading.
            lr_decay (float or None, optional):
                If provided, enables an inverse-time learning-rate decay schedule using this decay rate.
            skip_compile (bool, optional):
                If True, skip the model compilation step prior to training.
            transfer_learning (bool, optional):
                If True, use the transfer-learning training/evaluation routines which compute only total MAE; otherwise use full component-wise routines.

        Side effects:
            - Updates self.model to the best-performing model (by test loss) found during training.
            - If self.model_save_path is set, saves the best model's state dict and minimal config to that path.
            - Moves model and batches to the device indicated by self.device.
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
        Train the APNet3 fused model using the provided dataset and training configuration.

        Parameters:
            dataset (optional): Dataset or list [train_dataset, test_dataset] to train on; if omitted the instance's dataset must be set beforehand.
            n_epochs (int): Number of training epochs.
            lr (float): Initial learning rate.
            split_percent (float): Fraction of `dataset` used for training when a single dataset is provided (value in (0,1]).
            model_path (str | None): Directory or file path to save training outputs and model checkpoints.
            shuffle (bool): Whether to shuffle datasets before splitting or when using provided train/test lists.
            dataloader_num_workers (int): Number of worker processes for data loading in single-process training.
            world_size (int): Number of processes for distributed (DDP) training; when >1, training is run via mp.spawn and ddp_train is used.
            omp_num_threads_per_process (int): Value set to OMP_NUM_THREADS for each training process (controls OpenMP threads per process).
            lr_decay (optional): Learning-rate scheduler configuration passed through to training routines (format expected by internal scheduler handlers).
            random_seed (int): Seed for NumPy RNG used when shuffling/splitting data.
            skip_compile (bool): When True, skip any model compilation steps before training in the single-process path.
            transfer_learning (bool): When True, run the single-process transfer-learning training path that uses a different update/evaluation behaviour.

        Notes:
            - If `dataset` is a list, it is interpreted as [train_dataset, test_dataset]; otherwise a split is performed using `split_percent`.
            - If `use_precomputed_classical` is set on the instance, the dimer property model is switched to the atom-MPNN forward mode before training.
            - For world_size > 1, this method launches distributed training via mp.spawn; for world_size == 1 it calls the single-process training routine.
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
        Freeze model parameters except the AP2 readout layers.

        Sets each parameter's `requires_grad` to False except for parameters belonging to readout modules whose top-level term ends with `elst`, `exch`, `indu`, or `disp`, which are left with `requires_grad = True`.
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
        Unfreeze all model parameters for AP3 model
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        return
