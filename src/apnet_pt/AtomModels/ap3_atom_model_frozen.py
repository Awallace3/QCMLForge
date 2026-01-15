import torch
import torch.nn as nn
from ..util import scatter_sum_compile
import numpy as np
import warnings
from .. import multipole
import time
from apnet_pt.atomic_datasets import (
    atomic_module_dataset,
    AtomicDataLoader,
    atomic_collate_update,
    qcel_mon_to_pyg_data,
    atomic_collate_update_no_target,
)
from apnet_pt.multipole import thole_damping_mutual_torch, thole_damping_direct_torch
from apnet_pt import constants

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from importlib import resources
import qcelemental as qcel
from .ap2_atom_model import (
    unsorted_segment_sum_3d,
    get_distances,
    DistanceLayer,
    max_Z,
    isolate_atomic_property_predictions,
    unwrap_model,
)
from .ap3_atomtype_mpnn import AtomTypeParamMPNN

warnings.filterwarnings("ignore")


class InducedDipoleMPNN(torch.nn.Module):
    def __init__(
        self,
        atomtype_hfvr_model=AtomTypeParamMPNN(),
        atom_mpnn_model=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        use_nn_screening=False,
        precompute_hfvr=False,
    ):
        """
        Initialize an InducedDipoleMPNN instance, configuring pretrained submodules (optional), embedding and readout architectures, and screening/damping options.
        
        Parameters:
            atomtype_hfvr_model (AtomTypeParamMPNN): Model to compute Hirshfeld-volume ratios and related per-atom features; stored and frozen unless `precompute_hfvr` is True.
            atom_mpnn_model (Optional[Module]): Pretrained AtomMPNN to reuse for distance/embed/guess/update/readout layers. If provided, its parameters are frozen except dipole-related update and readout layers which remain trainable.
            n_message (int): Number of message-passing steps / layers to create when not using a pretrained AtomMPNN.
            n_rbf (int): Number of radial basis functions for distance embedding.
            n_neuron (int): Base width used to construct hidden layer sizes for new layers.
            n_embed (int): Size of atom-type embedding vectors.
            r_cut (float): Cutoff distance for distance embedding.
            use_nn_screening (bool): If True, create additional damping update/readout networks used for NN-based screening of induction interactions.
            precompute_hfvr (bool): If True, do not store `atomtype_hfvr_model` (expect hfvr/vw provided in input batches) to save memory.
        
        Behavior:
            - Stores hyperparameters on the instance.
            - When `precompute_hfvr` is False, stores and freezes `atomtype_hfvr_model`; otherwise the model is not retained.
            - When `atom_mpnn_model` is provided, reuses its distance/embed/guess/update/readout layers and freezes all of them except dipole update/readout layers; otherwise constructs new distance, embedding, guess, update, and readout layers sized according to provided hyperparameters.
            - If `use_nn_screening` is True, constructs per-message-step damping update and readout networks.
        """
        super().__init__()
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut = r_cut
        self.use_nn_screening = use_nn_screening
        self.precompute_hfvr = precompute_hfvr

        print(f"Precompute HFVR: {precompute_hfvr}")
        # Only store and freeze model if NOT precomputing
        if not precompute_hfvr:
            self.atomtype_hfvr_model = atomtype_hfvr_model
            self.atomtype_hfvr_model.requires_grad_(False)
        else:
            # Don't store model to save memory - will read from batch
            self.atomtype_hfvr_model = None

        # Store and freeze the pretrained AtomMPNN model for predicting charges, dipoles, and quadrupoles
        if atom_mpnn_model is not None:
            print(
                "Using pretrained AtomMPNN model for charge, dipole, and quadrupole predictions"
            )
            self.atom_mpnn_model = atom_mpnn_model
            # Freeze all layers in the pretrained model
            self.atom_mpnn_model.requires_grad_(False)
            # We'll selectively unfreeze dipole layers later
        else:
            self.atom_mpnn_model = None

        self.polarizability_table = constants.polarizability_table.clone()
        # If we have a pretrained AtomMPNN model, use its layers for charge/dipole/qpole prediction
        # Otherwise, create new layers
        if self.atom_mpnn_model is not None:
            # Use pretrained model's layers - these will be frozen except dipole layers
            self.distance_layer = self.atom_mpnn_model.distance_layer
            self.embed_layer = self.atom_mpnn_model.embed_layer
            self.guess_layer = self.atom_mpnn_model.guess_layer
            self.charge_update_layers = self.atom_mpnn_model.charge_update_layers
            self.dipole_update_layers = self.atom_mpnn_model.dipole_update_layers
            self.qpole1_update_layers = self.atom_mpnn_model.qpole1_update_layers
            self.qpole2_update_layers = self.atom_mpnn_model.qpole2_update_layers
            self.charge_readout_layers = self.atom_mpnn_model.charge_readout_layers
            self.dipole_readout_layers = self.atom_mpnn_model.dipole_readout_layers
            self.qpole_readout_layers = self.atom_mpnn_model.qpole_readout_layers

            # Freeze all layers from the pretrained model
            for param in self.distance_layer.parameters():
                param.requires_grad = False
            for param in self.embed_layer.parameters():
                param.requires_grad = False
            for param in self.guess_layer.parameters():
                param.requires_grad = False
            for param in self.charge_update_layers.parameters():
                param.requires_grad = False
            for param in self.qpole1_update_layers.parameters():
                param.requires_grad = False
            for param in self.qpole2_update_layers.parameters():
                param.requires_grad = False
            for param in self.charge_readout_layers.parameters():
                param.requires_grad = False
            for param in self.qpole_readout_layers.parameters():
                param.requires_grad = False

            # Unfreeze only dipole update and readout layers
            for param in self.dipole_update_layers.parameters():
                param.requires_grad = True
            for param in self.dipole_readout_layers.parameters():
                param.requires_grad = True
        else:
            # Create new layers (original behavior)
            # embed interatomic distances into large orthogonal basis
            self.distance_layer = DistanceLayer(n_rbf, r_cut)

            # embed atom types
            self.embed_layer = nn.Embedding(max_Z + 1, n_embed)

            # zero-th order charge guess, based solely on atom type
            self.guess_layer = nn.Embedding(max_Z + 1, 1)

            # update layers for hidden states
            self.charge_update_layers = nn.ModuleList()
            self.dipole_update_layers = nn.ModuleList()
            self.qpole1_update_layers = nn.ModuleList()
            self.qpole2_update_layers = nn.ModuleList()

            # readout layers for predicting multipoles from hidden states
            self.charge_readout_layers = nn.ModuleList()
            self.dipole_readout_layers = nn.ModuleList()
            self.qpole_readout_layers = nn.ModuleList()

        # damping layers for NN screening (only used if use_nn_screening=True)
        if use_nn_screening:
            self.damping_update_layers = nn.ModuleList()
            self.damping_readout_layers = nn.ModuleList()

        # + 4 for hfvr (2), vw (2)
        input_layer_size = n_embed * 4 * n_rbf + n_embed * 4 + n_rbf + 4
        # + 6 for hfvr (2), vw (2), q (2)
        input_layer_size_damping = n_embed * 4 * n_rbf + n_embed * 4 + n_rbf + 6

        layer_nodes_hidden = [
            input_layer_size,
            n_neuron * 2,
            n_neuron,
            n_neuron // 2,
            n_embed,
        ]
        layer_nodes_hidden_damping = [
            input_layer_size_damping,
            n_neuron * 2,
            n_neuron,
            n_neuron // 2,
            n_embed,
        ]
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
        ]  # None represents a linear activation

        # Only initialize layers if we don't have a pretrained model
        if self.atom_mpnn_model is None:
            for i in range(n_message):
                self.charge_update_layers.append(
                    self._make_layers(layer_nodes_hidden, layer_activations)
                )
                self.dipole_update_layers.append(
                    self._make_layers(layer_nodes_hidden, layer_activations)
                )
                self.qpole1_update_layers.append(
                    self._make_layers(layer_nodes_hidden, layer_activations)
                )
                self.qpole2_update_layers.append(
                    self._make_layers(layer_nodes_hidden, layer_activations)
                )

                self.charge_readout_layers.append(
                    self._make_layers(layer_nodes_readout, layer_activations)
                )
                self.dipole_readout_layers.append(nn.Linear(n_embed, 1))
                self.qpole_readout_layers.append(nn.Linear(n_embed, 1))

        # Add damping layers for NN screening (always created, not from pretrained model)
        if use_nn_screening:
            for i in range(n_message):
                self.damping_update_layers.append(
                    self._make_layers(layer_nodes_hidden, layer_activations)
                )
                self.damping_readout_layers.append(
                    self._make_layers(layer_nodes_readout, layer_activations)
                )

    def _make_layers(self, layer_nodes, activations):
        """
        Builds a sequential feed-forward network from specified layer sizes and activation modules.
        
        Parameters:
        	layer_nodes (list[int]): Sizes of consecutive linear layers; length must be at least 2 (defines input and output sizes).
        	activations (list[Optional[torch.nn.Module]]): Activation modules inserted after each corresponding linear layer except the final; an entry of `None` leaves that position without an activation. Length must be len(layer_nodes) - 1.
        
        Returns:
        	torch.nn.Sequential: A Sequential container of Linear layers interleaved with the provided activation modules.
        """
        layers = []
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            if activations[i] is not None:
                layers.append(activations[i])
        return nn.Sequential(*layers)

    def get_messages_without_hfvr(self, h0, h, rbf, e_source, e_target):
        """
        Build edge-wise message features excluding Hirshfeld-volume ratio and van-der-Waals terms for use with pretrained AtomMPNN layers.
        
        Parameters:
            h0 (Tensor): Initial per-atom embeddings with shape (N, n_embed).
            h (Tensor): Current per-atom hidden embeddings with shape (N, n_embed).
            rbf (Tensor): Per-edge radial basis features with shape (E, n_rbf).
            e_source (LongTensor): Source atom indices for each edge with shape (E,).
            e_target (LongTensor): Target atom indices for each edge with shape (E,).
        
        Returns:
            Tensor: Edge-wise message tensor m_ij with shape (E, 4*n_embed + 4*n_embed*n_rbf + n_rbf),
            containing [h0_source, h0_target, h_source, h_target] concatenated with their elementwise
            products with `rbf` (flattened) and the raw `rbf`.
        """
        nedge = e_source.size(0)

        h0_source = h0.index_select(0, e_source)
        h0_target = h0.index_select(0, e_target)
        h_source = h.index_select(0, e_source)
        h_target = h.index_select(0, e_target)

        # [edges x 4 * n_embed]
        h_all = torch.cat([h0_source, h0_target, h_source, h_target], dim=-1)

        # [edges, 4 * n_embed, n_rbf]
        h_all_dot = torch.einsum("ez,er->ezr", h_all, rbf)
        # [edges, 4 * n_embed * n_rbf]
        h_all_dot = h_all_dot.view(nedge, -1)
        m_ij = torch.cat([h_all, h_all_dot, rbf], dim=-1)
        return m_ij

    def get_messages(self, h0, h, rbf, hfvr, vw, e_source, e_target):
        """
        Build per-edge message features by combining source/target node embeddings, radial basis expansions, and Hirshfeld-volume/VW parameters.
        
        Parameters:
            h0 (Tensor): Initial node embeddings with shape [n_nodes, n_embed].
            h (Tensor): Current node embeddings with shape [n_nodes, n_embed].
            rbf (Tensor): Radial basis features for each edge with shape [n_edges, n_rbf].
            hfvr (Tensor): Per-atom Hirshfeld volume ratio embeddings with shape [n_nodes, n_embed].
            vw (Tensor): Per-atom van der Waals (or similar) embeddings with shape [n_nodes, n_embed].
            e_source (LongTensor): Source node indices for each edge with shape [n_edges].
            e_target (LongTensor): Target node indices for each edge with shape [n_edges].
        
        Returns:
            Tensor: Per-edge message tensor m_ij with shape [n_edges, feature_dim], where each row is the concatenation of:
                - source and target initial embeddings (h0_source, h0_target) and current embeddings (h_source, h_target),
                - pairwise projections of those embeddings onto the radial basis (expanded and flattened),
                - the radial basis vector rbf for the edge,
                - concatenated hfvr and vw embeddings for source and target.
        """
        nedge = e_source.size(0)

        h0_source = h0.index_select(0, e_source)
        h0_target = h0.index_select(0, e_target)
        h_source = h.index_select(0, e_source)
        h_target = h.index_select(0, e_target)
        # hfvr, vw
        hfvr_source = hfvr.index_select(0, e_source)
        hfvr_target = hfvr.index_select(0, e_target)
        vw_source = vw.index_select(0, e_source)
        vw_target = vw.index_select(0, e_target)
        param_all = torch.cat([hfvr_source, hfvr_target, vw_source, vw_target], dim=-1)
        # [edges x 4 * n_embed]
        h_all = torch.cat([h0_source, h0_target, h_source, h_target], dim=-1)

        # [edges, 4 * n_embed, n_rbf]
        h_all_dot = torch.einsum("ez,er->ezr", h_all, rbf)
        # [edges, 4 * n_embed * n_rbf]
        h_all_dot = h_all_dot.view(nedge, -1)
        m_ij = torch.cat([h_all, h_all_dot, rbf, param_all], dim=-1)
        return m_ij

    def get_messages_charges(self, h0, h, rbf, hfvr, vw, q, e_source, e_target):
        """
        Constructs per-edge message features used for charge prediction by concatenating source/ and target-based embeddings, their interactions with radial basis functions, the raw radial basis vector, and Hirshfeld-volume-ratio (hfvr) and van-der-Waals (vw) descriptors.
        
        Parameters:
            h0: Node initial embeddings; used for both source and target contributions.
            h: Node hidden embeddings after message updates; used for both source and target contributions.
            rbf: Radial basis function vector per edge.
            hfvr: Per-atom Hirshfeld volume ratio descriptors.
            vw: Per-atom van-der-Waals descriptors.
            q: Per-atom partial charges (not used by this function but kept for API compatibility).
            e_source: Edge source node indices.
            e_target: Edge target node indices.
        
        Returns:
            m_ij (torch.Tensor): Per-edge feature tensor of shape [n_edges, feature_dim] formed by concatenating:
                - source/target initial and hidden embeddings,
                - the elementwise interactions of those embeddings with the RBF vector,
                - the raw RBF vector,
                - concatenated hfvr and vw descriptors from source and target.
        """
        nedge = e_source.size(0)

        h0_source = h0.index_select(0, e_source)
        h0_target = h0.index_select(0, e_target)
        h_source = h.index_select(0, e_source)
        h_target = h.index_select(0, e_target)
        # hfvr, vw
        hfvr_source = hfvr.index_select(0, e_source)
        hfvr_target = hfvr.index_select(0, e_target)
        vw_source = vw.index_select(0, e_source)
        vw_target = vw.index_select(0, e_target)
        # q_source = q.index_select(0, e_source)
        # q_target = q.index_select(0, e_target)
        param_all = torch.cat(
            [
                hfvr_source,
                hfvr_target,
                vw_source,
                vw_target,
                # q_source,
                # q_target,
            ],
            dim=-1,
        )
        # [edges x 4 * n_embed]
        h_all = torch.cat([h0_source, h0_target, h_source, h_target], dim=-1)

        # [edges, 4 * n_embed, n_rbf]
        h_all_dot = torch.einsum("ez,er->ezr", h_all, rbf)
        # [edges, 4 * n_embed * n_rbf]
        h_all_dot = h_all_dot.view(nedge, -1)
        m_ij = torch.cat([h_all, h_all_dot, rbf, param_all], dim=-1)
        return m_ij

    def monomer_induced_dipole_torch(
        self,
        Z,
        R,
        q,
        mu,
        quad,
        e_source,
        e_target,
        hirshfeld_volume_ratio: torch.Tensor,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-8,
        omega: float = 0.7,
        thole_damping_param_mutual: float = 0.39,
        thole_damping_param_direct: float = 0.34,
        screening: bool = True,
        screening_distance: float = 2.0,
        compute_energies: bool = False,
    ) -> torch.Tensor:
        """
        Compute intramolecular induced atomic dipoles using a Thole-damped self-consistent field (SCF) induction model.
        
        This function uses permanent multipoles (charges, dipoles, quadrupoles) together with atomwise polarizabilities scaled by the Hirshfeld volume ratio to iteratively converge induced dipoles via Thole-damped interaction tensors. Hydrogen atoms are excluded from mutual induction contributions.
        
        Parameters:
            Z (torch.Tensor): Atomic numbers, shape (n_atoms,).
            R (torch.Tensor): Atomic positions in Bohr, shape (n_atoms, 3).
            q (torch.Tensor): Atomic charges, shape (n_atoms,) or (n_atoms, 1).
            mu (torch.Tensor): Permanent atomic dipoles, shape (n_atoms, 3).
            quad (torch.Tensor): Permanent atomic quadrupoles, shape (n_atoms, 3, 3).
            e_source (torch.Tensor): Source atom indices for pair interactions (edge list).
            e_target (torch.Tensor): Target atom indices matching e_source.
            hirshfeld_volume_ratio (torch.Tensor): Per-atom Hirshfeld volume ratios used to scale gas-phase polarizabilities, shape (n_atoms,).
            max_iterations (int, optional): Maximum SCF iterations (default: 200).
            convergence_threshold (float, optional): L2-norm threshold for SCF convergence (default: 1e-8).
            omega (float, optional): SCF mixing parameter between old and new induced dipoles (default: 0.7).
            thole_damping_param_mutual (float, optional): Thole parameter for induced–induced interactions (default: 0.39).
            thole_damping_param_direct (float, optional): Thole parameter for permanent→induced interactions (default: 0.34).
            screening (bool, optional): If True, apply short-range distance screening to direct (permanent→induced) tensors to exclude very close 1-2/1-3 contributions (default: True).
            screening_distance (float, optional): Distance threshold in Angstroms used for screening direct interactions (default: 2.0).
            compute_energies (bool, optional): Reserved flag to compute induction energies (not used by this implementation; default: False).
        
        Returns:
            torch.Tensor: Converged induced dipoles for each atom, shape (n_atoms, 3).
        """

        # torch.set_printoptions(profile="full")
        # Calculate atomic polarizabilities
        alpha_0 = torch.index_select(self.polarizability_table, 0, Z.long())
        alpha = alpha_0 * torch.abs(hirshfeld_volume_ratio) ** (4 / 3.0)
        # torch.set_printoptions(precision=6)
        # Torch print full ternor
        # print(f"{alpha = }")
        # print(f"{Z = }")
        # print(f"{alpha_0 = }")
        # # get index of nan values of alpha_0
        # nan_indices = torch.isnan(alpha)
        # print(f"{nan_indices = }")
        # print(f"{Z[nan_indices] = }")
        # print(f"{hirshfeld_volume_ratio = }")

        # Define helper function to calculate distance tensors with Thole damping
        def distance_tensors(
            Ri,
            Rj,
            e_source,
            e_target,
            alpha_i,
            alpha_j,
            thole_param,
            apply_screening=False,
        ):
            """
            Compute interatomic distance metrics and Thole-damped interaction tensors for source→target atom pairs.
            
            Parameters:
                Ri (Tensor): Atom positions for source atoms (shape [..., 3]).
                Rj (Tensor): Atom positions for target atoms (shape [..., 3]).
                e_source (LongTensor): Indices selecting source atoms for each edge.
                e_target (LongTensor): Indices selecting target atoms for each edge.
                alpha_i (Tensor): Per-atom polarizabilities for source atoms.
                alpha_j (Tensor): Per-atom polarizabilities for target atoms.
                thole_param (float or Tensor): Thole damping parameter controlling short-range damping.
                apply_screening (bool): If True, use the "direct" Thole damping variant; otherwise use the mutual variant.
            
            Returns:
                tuple: (
                    dR (Tensor): Interatomic distances for each edge in angstroms.
                    dR_xyz (Tensor): Interatomic displacement vectors (source→target) for each edge in angstroms.
                    oodR (Tensor): Elementwise inverse of `dR` for each edge (1/angstrom).
                    T1 (Tensor): Rank-1 field tensors for each edge describing dipole field contributions at the target.
                    T2 (Tensor): Rank-2 field-gradient tensors for each edge describing dipole–dipole coupling gradients.
                )
            
            Notes:
                - Distances are converted to angstrom units inside the function.
                - Thole damping factors modulate the tensors to avoid singular short-range behavior.
            """
            dR_ang, dR_xyz_ang = get_distances(Ri, Rj, e_source, e_target)
            dR_xyz = dR_xyz_ang / constants.au2ang
            dR = dR_ang / constants.au2ang

            alpha_source = alpha_i.index_select(0, e_source)
            alpha_target = alpha_j.index_select(0, e_target)

            # Apply Thole damping
            if apply_screening:
                au3, lam_3, lam_5 = thole_damping_direct_torch(
                    dR, alpha_source, alpha_target, thole_param
                )
            else:
                au3, lam_3, lam_5 = thole_damping_mutual_torch(
                    dR, alpha_source, alpha_target, thole_param
                )

            # Apply distance-based screening for direct interactions (exclude 1-2, 1-3 bonds)
            if apply_screening and screening:
                screening_mask = dR_ang < screening_distance
                lam_3 = torch.where(screening_mask, torch.zeros_like(lam_3), lam_3)
                lam_5 = torch.where(screening_mask, torch.zeros_like(lam_5), lam_5)
                dR = torch.where(screening_mask, torch.ones_like(dR), dR)

            delta = torch.eye(3, device=dR.device)
            oodR = 1.0 / dR

            # T1: field tensor (rank 1)
            # Note: dR_xyz points FROM source TO target, which is the correct direction
            # for the field at target due to source. No negation needed.
            T1 = torch.einsum("x,xy,x->xy", oodR**3, dR_xyz, lam_3)

            # T2: field gradient tensor (rank 2)
            T2 = 3 * torch.einsum("xy,xz,x->xyz", dR_xyz, dR_xyz, lam_5) - torch.einsum(
                "x,x,yz,x->xyz", dR, dR, delta, lam_3
            )
            T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)

            return dR, dR_xyz, oodR, T1, T2

        # Calculate direct tensors (permanent → induced) with screening
        dR_direct, dR_xyz_direct, T0_direct, T1_direct, T2_direct = distance_tensors(
            R,
            R,
            e_source,
            e_target,
            alpha,
            alpha,
            thole_damping_param_direct,
            apply_screening=True,
        )

        # Calculate mutual tensors (induced ↔ induced) without screening
        dR_mutual, dR_xyz_mutual, T0_mutual, T1_mutual, T2_mutual = distance_tensors(
            R,
            R,
            e_source,
            e_target,
            alpha,
            alpha,
            thole_damping_param_mutual,
            apply_screening=False,
        )

        # Initialize induced dipoles
        # print(f"{R.size() = }")
        n_atoms = R.shape[0]
        mu_induced_0 = torch.zeros((n_atoms, 3), device=q.device)

        # Select relevant tensors for atom pairs
        # alpha_source = alpha.index_select(0, e_source)
        alpha_target = alpha.index_select(0, e_target)
        q_source = q.squeeze(-1).index_select(0, e_source)
        mu_source = mu.index_select(0, e_source)

        # Calculate initial induced dipoles from permanent multipoles (using direct tensors)
        # Contribution from charges: mu_ind = alpha * T1 * q
        mu_charge = torch.einsum("a,ai,a->ai", alpha_target, T1_direct, q_source)
        mu_induced_0 = scatter_sum_compile(mu_charge, e_target, n_atoms)

        # Contribution from dipoles: mu_ind += alpha * T2 * mu
        mu_dipole = torch.einsum("a,aij,aj->ai", alpha_target, T2_direct, mu_source)
        mu_dipole_summed = scatter_sum_compile(mu_dipole, e_target, n_atoms)
        mu_induced_0 += mu_dipole_summed

        # print(f"{mu_induced_0.size()}")
        # print(f"{Z.size()}")
        # print(f"{Z = }")
        # print(f"{e_source = }")
        # print(f"{e_target = }")
        # apply heavy_atoms_only logic in torch
        mu_induced_0 = torch.where(
            Z.unsqueeze(-1) == 1, torch.zeros_like(mu_induced_0), mu_induced_0
        )
        Z_source = Z.index_select(0, e_source)
        Z_target = Z.index_select(0, e_target)
        # set all T tenors to zero where either source or target is hydrogen
        hydrogen_mask = (Z_source == 1) | (Z_target == 1)
        T2_mutual = torch.where(
            hydrogen_mask.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(T2_mutual),
            T2_mutual,
        )
        T1_direct = torch.where(
            hydrogen_mask.unsqueeze(-1),
            torch.zeros_like(T1_direct),
            T1_direct,
        )

        # Self-consistent field (SCF) iteration to converge induced dipoles
        mu_induced = mu_induced_0.clone()

        for iteration in range(max_iterations):
            mu_induced_old = mu_induced.clone()

            # Induced dipoles due to other induced dipoles (using mutual tensors)
            mu_induced_contrib = torch.einsum(
                "a,aij,aj->ai",
                alpha_target,
                T2_mutual,
                mu_induced.index_select(0, e_source),
            )
            mu_induced_new = scatter_sum_compile(mu_induced_contrib, e_target, n_atoms)
            # Add initial induced dipoles from permanent multipoles
            mu_induced_new += mu_induced_0

            # Apply mixing for numerical stability
            mu_induced = (1 - omega) * mu_induced_old + omega * mu_induced_new

            # Check convergence
            delta = torch.norm(mu_induced - mu_induced_old)
            if delta < convergence_threshold:
                break

        return mu_induced

    def expand_screening_to_full_edges(
        self,
        screening_factors_short: torch.Tensor,
        e_source_short: torch.Tensor,
        e_target_short: torch.Tensor,
        e_source_full: torch.Tensor,
        e_target_full: torch.Tensor,
        default_value: float = 1.0,
    ) -> torch.Tensor:
        """
        Map short-range edge screening factors onto a full edge list by matching (source, target) pairs.
        
        Parameters:
            screening_factors_short (torch.Tensor): Screening values for short-range edges with shape (n_short_edges,).
            e_source_short (torch.Tensor): Source atom indices for short-range edges.
            e_target_short (torch.Tensor): Target atom indices for short-range edges.
            e_source_full (torch.Tensor): Source atom indices for the full edge set.
            e_target_full (torch.Tensor): Target atom indices for the full edge set.
            default_value (float): Value to assign for full edges that have no corresponding short-range entry (default: 1.0).
        
        Returns:
            torch.Tensor: Screening factors for the full edge set with shape (n_full_edges,); each entry is taken from
            `screening_factors_short` when a matching (source, target) pair exists, otherwise `default_value`.
        """
        n_full_edges = e_source_full.size(0)
        screening_factors_full = torch.full(
            (n_full_edges,),
            default_value,
            device=screening_factors_short.device,
            dtype=screening_factors_short.dtype,
        )

        # Create edge pair hashes for efficient lookup
        # Hash: source * max_nodes + target (assumes max_nodes < sqrt(max_long))
        # Use torch.max to avoid .item() which creates data-dependent control flow
        max_nodes = (
            torch.max(
                torch.max(e_source_short.max(), e_target_short.max()),
                torch.max(e_source_full.max(), e_target_full.max()),
            )
            + 1
        )
        hash_short = e_source_short * max_nodes + e_target_short
        hash_full = e_source_full * max_nodes + e_target_full

        # Vectorized edge matching using broadcasting and comparison
        # Shape: (n_full_edges, n_short_edges)
        matches = hash_full.unsqueeze(1) == hash_short.unsqueeze(0)

        # Find indices where matches occur
        # any_match: (n_full_edges,) - whether each full edge has a match
        # which_match: (n_full_edges,) - index of matching short edge (undefined if no match)
        any_match = matches.any(dim=1)
        which_match = matches.long().argmax(dim=1)  # Gets first match index

        # Update screening factors only where matches exist
        screening_factors_full[any_match] = screening_factors_short[
            which_match[any_match]
        ]

        return screening_factors_full

    def monomer_induced_dipole_torch_NN_screening(
        self,
        Z,
        R,
        q,
        mu,
        quad,
        e_source_short: torch.Tensor,
        e_target_short: torch.Tensor,
        e_source_full: torch.Tensor,
        e_target_full: torch.Tensor,
        h_list,
        rbf_short: torch.Tensor,
        hfvr,
        vw,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-8,
        omega: float = 0.7,
        thole_damping_param_mutual: float = 0.39,
        thole_damping_param_direct: float = 0.34,
        compute_energies: bool = False,
    ) -> torch.Tensor:
        """
        Compute intramolecular induced dipoles for a single molecule using permanent multipoles, Hirshfeld-volume-scaled polarizabilities, and NN-derived short-range screening.
        
        Parameters:
            Z (torch.Tensor): Atomic numbers, shape (n_atoms,).
            R (torch.Tensor): Atomic positions in Bohr, shape (n_atoms, 3).
            q (torch.Tensor): Atomic charges, shape (n_atoms,) or (n_atoms, 1).
            mu (torch.Tensor): Permanent atomic dipoles, shape (n_atoms, 3).
            quad (torch.Tensor): Permanent atomic quadrupoles, shape (n_atoms, 3, 3).
            e_source_short (torch.Tensor): Source indices for short-range edges used by the NN screening.
            e_target_short (torch.Tensor): Target indices for short-range edges used by the NN screening.
            e_source_full (torch.Tensor): Source indices for the full edge set used in induction tensors.
            e_target_full (torch.Tensor): Target indices for the full edge set used in induction tensors.
            h_list (iterable of torch.Tensor): Hidden states from message passing for each message step (indexed 0..n_message).
            rbf_short (torch.Tensor): RBF features for short-range edges (used by screening NN), shape (n_short_edges, n_rbf).
            hfvr (torch.Tensor): Hirshfeld volume ratio features for polarizability scaling, shape (n_atoms,) or (n_atoms, 1).
            vw (torch.Tensor): Van-der-Waals features used by message/screening networks.
            max_iterations (int): Maximum SCF iterations (default 200).
            convergence_threshold (float): Convergence threshold on induced dipole norm (default 1e-8).
            omega (float): SCF mixing parameter in (0, 1] for numerical stability (default 0.7).
            thole_damping_param_mutual (float): Thole parameter for induced–induced interactions (default 0.39).
            thole_damping_param_direct (float): Thole parameter for permanent→induced interactions (default 0.34).
            compute_energies (bool): Placeholder flag for computing induction energies; not used by this implementation.
        
        Returns:
            torch.Tensor: Converged induced dipole moments for each atom, shape (n_atoms, 3).
        """

        # Calculate atomic polarizabilities
        alpha_0 = torch.index_select(self.polarizability_table, 0, Z.long())
        alpha = alpha_0 * torch.abs(hfvr.squeeze(1)) ** (4 / 3.0)
        # torch.set_printoptions(precision=6)
        # Torch print full ternor
        # print(f"{alpha = }")
        # print(f"{Z = }")
        # print(f"{alpha_0 = }")
        # # get index of nan values of alpha_0
        # nan_indices = torch.isnan(alpha)
        # print(f"{nan_indices = }")
        # print(f"{Z[nan_indices] = }")
        # print(f"{hfvr = }")

        # Compute NN-based screening factors using damping layers on SHORT-RANGE edges
        # Initialize screening factors tensor (size: n_short_edges)
        n_short_edges = e_source_short.size(0)
        screening_factors_short = torch.zeros(
            n_short_edges,
            device=e_source_short.device,
            dtype=rbf_short.dtype,
        )

        # Accumulate screening factors across message passing steps
        for i in range(self.n_message):
            m_ij = self.get_messages(
                h_list[0],
                h_list[i + 1],
                rbf_short,
                hfvr,
                vw,
                # q,
                e_source_short,
                e_target_short,
            )
            h_damping = self.damping_update_layers[i](m_ij)
            # Get screening factor (sigmoid to constrain to [0, 1])
            # screen_factor = torch.sigmoid(self.damping_readout_layers[i](h_damping))
            # # Invert so that 0 = fully screened, 1 = no screening
            # screen_factor = 1.0 - screen_factor
            # # Accumulate screening factors
            # screening_factors_short += screen_factor.squeeze(-1)
            screening_factors_short += self.damping_readout_layers[i](
                h_damping
            ).squeeze(-1)
        # Apply 1-screening accumulation
        screening_factors_short = 1.0 - screening_factors_short
        # Apply sigmoid function on everything to constrain to [0, 1]
        screening_factors_short = torch.sigmoid(screening_factors_short)

        # Expand screening factors from short-range to full edges
        # Long-range edges get default value of 1.0 (no screening)
        # print(f"{e_source_short = }")
        # print(f"{e_target_short = }")
        # print(f"{e_source_full = }")
        # print(f"{e_target_full = }")
        # print(f"{screening_factors_short = }")
        screening_factors = self.expand_screening_to_full_edges(
            screening_factors_short,
            e_source_short,
            e_target_short,
            e_source_full,
            e_target_full,
            default_value=1.0,
        )

        # screening_factors = screening_factors_short
        # e_source_full = e_source_short
        # e_target_full = e_target_short
        # print(f"{screening_factors = }")
        # print(f"{screening_factors = }")
        # Define helper function to calculate distance tensors with Thole damping
        def distance_tensors(
            Ri,
            Rj,
            e_source,
            e_target,
            alpha_i,
            alpha_j,
            thole_param,
            apply_screening=False,
            screening_factors=None,
        ):
            """
            Compute distance-based interaction tensors between atom pairs, optionally applying Thole damping and NN-provided screening.
            
            Parameters:
                Ri (Tensor): Atomic positions (N_i x 3) for source set, in angstroms.
                Rj (Tensor): Atomic positions (N_j x 3) for target set, in angstroms.
                e_source (LongTensor): Index array selecting source atom for each edge (E,).
                e_target (LongTensor): Index array selecting target atom for each edge (E,).
                alpha_i (Tensor): Per-atom polarizabilities for source atoms (N_i,).
                alpha_j (Tensor): Per-atom polarizabilities for target atoms (N_j,).
                thole_param (float or Tensor): Thole damping parameter (scalar or per-edge tensor) used to compute damping factors.
                apply_screening (bool): If True, compute direct-interaction damping; otherwise compute mutual-interaction damping.
                screening_factors (Tensor or None): Optional per-edge factors in [0, 1] to scale direct-interaction damping (0 = fully screened, 1 = no screening).
            
            Returns:
                dR (Tensor): Interatomic distances for each edge in atomic units (E,).
                dR_xyz (Tensor): Interatomic displacement vectors for each edge in atomic units (E, 3).
                oodR (Tensor): Elementwise reciprocal of dR (E,).
                T1 (Tensor): Rank-1 interaction (field) tensor per edge (E, 3).
                T2 (Tensor): Rank-2 interaction (field-gradient) tensor per edge (E, 3, 3).
            """
            dR_ang, dR_xyz_ang = get_distances(Ri, Rj, e_source, e_target)
            dR_xyz = dR_xyz_ang / constants.au2ang
            dR = dR_ang / constants.au2ang

            alpha_source = alpha_i.index_select(0, e_source)
            alpha_target = alpha_j.index_select(0, e_target)

            # Apply Thole damping
            if apply_screening:
                au3, lam_3, lam_5 = thole_damping_direct_torch(
                    dR, alpha_source, alpha_target, thole_param
                )
            else:
                au3, lam_3, lam_5 = thole_damping_mutual_torch(
                    dR, alpha_source, alpha_target, thole_param
                )

            # Apply NN-based screening for direct interactions if provided
            if apply_screening and screening_factors is not None:
                # screening_factors should be in range [0, 1] where 0 = fully screened, 1 = no screening
                lam_3 = lam_3 * screening_factors
                # print(f"{lam_3 = }")
                lam_5 = lam_5 * screening_factors

            delta = torch.eye(3, device=dR.device)
            oodR = 1.0 / dR

            # T1: field tensor (rank 1)
            T1 = torch.einsum("x,xy,x->xy", oodR**3, dR_xyz, lam_3)

            # T2: field gradient tensor (rank 2)
            T2 = 3 * torch.einsum("xy,xz,x->xyz", dR_xyz, dR_xyz, lam_5) - torch.einsum(
                "x,x,yz,x->xyz", dR, dR, delta, lam_3
            )
            T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)

            return dR, dR_xyz, oodR, T1, T2

        # Calculate direct tensors (permanent → induced) with NN screening
        # Use FULL EDGES for induced dipole calculations (long-range interactions)
        dR_direct, dR_xyz_direct, T0_direct, T1_direct, T2_direct = distance_tensors(
            R,
            R,
            e_source_full,
            e_target_full,
            alpha,
            alpha,
            thole_damping_param_direct,
            apply_screening=True,
            screening_factors=screening_factors,
        )

        # Calculate mutual tensors (induced ↔ induced) without screening
        dR_mutual, dR_xyz_mutual, T0_mutual, T1_mutual, T2_mutual = distance_tensors(
            R,
            R,
            e_source_full,
            e_target_full,
            alpha,
            alpha,
            thole_damping_param_mutual,
            apply_screening=False,
        )

        # Initialize induced dipoles
        n_atoms = R.shape[0]
        mu_induced_0 = torch.zeros((n_atoms, 3), device=q.device)

        # Select relevant tensors for atom pairs (using full edges)
        alpha_target = alpha.index_select(0, e_target_full)
        q_source = q.squeeze(-1).index_select(0, e_source_full)
        mu_source = mu.index_select(0, e_source_full)

        # Calculate initial induced dipoles from permanent multipoles (using direct tensors with NN screening)
        # Contribution from charges: mu_ind = alpha * T1 * q
        mu_charge = torch.einsum("a,ai,a->ai", alpha_target, T1_direct, q_source)
        mu_induced_0 = scatter_sum_compile(mu_charge, e_target_full, n_atoms)

        # Contribution from dipoles: mu_ind += alpha * T2 * mu
        mu_dipole = torch.einsum("a,aij,aj->ai", alpha_target, T2_direct, mu_source)
        mu_dipole_summed = scatter_sum_compile(mu_dipole, e_target_full, n_atoms)
        mu_induced_0 += mu_dipole_summed

        # Apply heavy_atoms_only logic in torch
        mu_induced_0 = torch.where(
            Z.unsqueeze(-1) == 1, torch.zeros_like(mu_induced_0), mu_induced_0
        )
        Z_source = Z.index_select(0, e_source_full)
        Z_target = Z.index_select(0, e_target_full)
        # Set all T tensors to zero where either source or target is hydrogen
        hydrogen_mask = (Z_source == 1) | (Z_target == 1)
        T2_mutual = torch.where(
            hydrogen_mask.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(T2_mutual),
            T2_mutual,
        )
        T1_direct = torch.where(
            hydrogen_mask.unsqueeze(-1),
            torch.zeros_like(T1_direct),
            T1_direct,
        )

        # Self-consistent field (SCF) iteration to converge induced dipoles
        mu_induced = mu_induced_0.clone()

        for iteration in range(max_iterations):
            mu_induced_old = mu_induced.clone()

            # Induced dipoles due to other induced dipoles (using mutual tensors on full edges)
            mu_induced_contrib = torch.einsum(
                "a,aij,aj->ai",
                alpha_target,
                T2_mutual,
                mu_induced.index_select(0, e_source_full),
            )
            mu_induced_new = scatter_sum_compile(
                mu_induced_contrib, e_target_full, n_atoms
            )
            # Add initial induced dipoles from permanent multipoles
            mu_induced_new += mu_induced_0

            # Apply mixing for numerical stability
            mu_induced = (1 - omega) * mu_induced_old + omega * mu_induced_new

            # Check convergence
            delta = torch.norm(mu_induced - mu_induced_old)
            if delta < convergence_threshold:
                break

        return mu_induced

    # @torch.jit.trace
    def forward(
        self,
        batch,
    ):
        # Extract variables from batch
        """
        Compute per-atom multipoles (charges, dipoles, quadrupoles) and hidden-state embeddings for a batched set of molecules.
        
        Processes the input batch of graph-structured molecules through the model's embedding, iterative message-passing updates for charge/dipole/quadrupole components, enforces traceless quadrupoles and global charge conservation per molecule, and computes induced dipoles via either the standard Thole-style SCF or an NN-screened SCF depending on model configuration.
        
        Parameters:
            batch: A batched data object containing at least the following attributes:
                - x (LongTensor or Tensor): atomic numbers or initial atom features, shape [n_atoms].
                - edge_index (LongTensor): short-range edge indices, shape [2, n_edges].
                - R (Tensor): atomic positions, shape [n_atoms, 3].
                - molecule_ind (LongTensor): mapping from atom index to molecule index, shape [n_atoms].
                - total_charge (Tensor): total charge per molecule, shape [n_molecules] or [n_molecules, 1].
                - natom_per_mol (LongTensor): number of atoms per molecule, shape [n_molecules].
                - If self.precompute_hfvr is True: volume_ratios (Tensor) and valence_widths (Tensor) of shape [n_atoms, 1].
                - If self.use_nn_screening is True: edge_index_full (LongTensor) for full-edge induced-dipole computation, shape [2, n_edges_full].
                Additional fields the model may read when configured (e.g., for on-the-fly hfvr computation).
        
        Returns:
            charge (Tensor): per-atom partial charges after charge-conservation adjustment, shape [n_atoms].
            dipole (Tensor): per-atom total dipoles including induced contribution, shape [n_atoms, 3].
            qpole (Tensor): per-atom traceless quadrupole tensors, shape [n_atoms, 3, 3].
            h_list_stacked (Tensor): stacked hidden states from each message-passing layer, shape [n_atoms, n_message+1, hidden_dim].
        """
        x = batch.x
        edge_index = batch.edge_index
        R = batch.R
        molecule_ind = batch.molecule_ind
        total_charge = batch.total_charge
        natom_per_mol = batch.natom_per_mol

        # edge_index has shape [(e_source, e_target), n_edges]
        Z = x
        natom = Z.size(0)

        h_list_0 = [self.embed_layer(Z)]

        # Initial guesses
        charge = self.guess_layer(Z)

        dipole = torch.zeros(natom, 3, dtype=torch.float32, device=Z.device)
        qpole = torch.zeros(natom, 3, 3, dtype=torch.float32, device=Z.device)

        if edge_index.size(1) == 0:
            # need h_list to have the same number of dimensions as the number of message passing layers
            h_list = [h_list_0[0] for i in range(self.n_message + 1)]
            h_list = torch.stack(h_list, dim=1)
            molecule_ind.requires_grad_(False)
            molecule_ind = molecule_ind.long()
            num_mols = (
                int(molecule_ind.max().item()) + 1 if molecule_ind.numel() > 0 else 1
            )
            total_charge_pred = scatter_sum_compile(
                charge, molecule_ind, num_mols, reduce="sum"
            )
            total_charge_pred = total_charge_pred.squeeze(-1)
            total_charge_err = total_charge_pred - total_charge
            charge_err = torch.repeat_interleave(
                total_charge_err / natom_per_mol.float(), natom_per_mol
            ).unsqueeze(1)
            charge = charge - charge_err
            return charge, dipole, qpole, h_list

        # Conditional hfvr/vw retrieval based on precompute_hfvr flag
        if self.precompute_hfvr:
            # Read directly from batch (pre-computed during dataset processing)
            hfvr = batch.volume_ratios.unsqueeze(1)  # [n_atoms, 1]
            vw = batch.valence_widths.unsqueeze(1)  # [n_atoms, 1]
        else:
            # Compute on-the-fly using model (existing behavior)
            Ks = self.atomtype_hfvr_model(batch)
            hfvr = Ks[:, 0].unsqueeze(1)
            vw = Ks[:, 1].unsqueeze(1)

        # 1) Filter out atoms that don't have edges
        # Create keep_mask directly from edge_index without using torch.isin
        # This is more compile-friendly than torch.isin with unbacked symbolic shapes
        natom = len(molecule_ind)
        keep_mask = torch.zeros(natom, dtype=torch.bool, device=molecule_ind.device)
        if edge_index.size(1) > 0:
            # Mark all atoms that appear in edge_index as True
            keep_mask.scatter_(0, edge_index[0], True)
            keep_mask.scatter_(0, edge_index[1], True)
        filtered_charge = charge[keep_mask]

        # Now `filtered_charge` contains only atoms from molecules that have >= 2 atoms and edges
        h_list = [h_list_0[0][keep_mask]]

        # Now we need to filter the edge_index to only include edges between
        # atoms in molecules with >= 2 atoms.
        e_source = edge_index[0]
        e_target = edge_index[1]
        edge_keep = keep_mask[e_source] & keep_mask[e_target]
        e_source = e_source[edge_keep]
        e_target = e_target[edge_keep]
        # shape [N], each kept atom -> new index
        idx_map = torch.cumsum(keep_mask, dim=0) - 1
        idx_map = idx_map.long()  # ensure integer
        e_source = idx_map[e_source]
        e_target = idx_map[e_target]

        R_mask = R[keep_mask, :]
        natom_filtered = keep_mask.sum()

        #  [edges]
        dR, dR_xyz = get_distances(R_mask, R_mask, e_source, e_target)

        # [edges x 3]
        dr_unit = dR_xyz / dR.unsqueeze(1)
        rbf = self.distance_layer(dR)

        for i in range(self.n_message):
            #####################
            ### charge update ###
            #####################

            # [edges x message_embedding_dim]
            # Use get_messages_without_hfvr when using pretrained AtomMPNN layers
            if self.atom_mpnn_model is not None:
                m_ij = self.get_messages_without_hfvr(
                    h_list[0], h_list[-1], rbf, e_source, e_target
                )
            else:
                m_ij = self.get_messages(
                    h_list[0], h_list[-1], rbf, hfvr, vw, e_source, e_target
                )

            # [atoms x message_embedding_dim]
            m_i = scatter_sum_compile(m_ij, e_source, int(natom_filtered), reduce="sum")  # type: ignore

            # [atomx x hidden_dim]
            h_next = self.charge_update_layers[i](m_i)
            h_list.append(h_next)
            charge_update = self.charge_readout_layers[i](h_list[i + 1])
            filtered_charge += charge_update

            #####################
            ### dipole update ###
            #####################

            # [edges x n_embed]
            m_ij_dipole = self.dipole_update_layers[i](m_ij)
            # [edges x 3 x n_embed]
            m_ij_dipole = torch.einsum("ex,em->exm", dr_unit, m_ij_dipole)
            # [atoms x 3 x n_embed]
            m_i_dipole = unsorted_segment_sum_3d(m_ij_dipole, e_source, natom)
            # [atoms x 3 x 1]
            d_dipole = self.dipole_readout_layers[i](m_i_dipole)
            # [atoms x 3]
            d_dipole = d_dipole.view(natom, 3)
            dipole += d_dipole

            #########################
            ### quadrupole update ###
            #########################

            # [edges x n_embed]
            m_ij_qpole1 = self.qpole1_update_layers[i](m_ij)
            # [edges x 3 x n_embed]
            m_ij_qpole1 = torch.einsum("ex,em->exm", dr_unit, m_ij_qpole1)
            # [atoms x 3 x n_embed]
            m_i_qpole1 = unsorted_segment_sum_3d(m_ij_qpole1, e_source, natom)

            # [edges x n_embed]
            m_ij_qpole2 = self.qpole2_update_layers[i](m_ij)
            # [edges x 3 x n_embed]
            m_ij_qpole2 = torch.einsum("ex,em->exm", dr_unit, m_ij_qpole2)
            # [atoms x 3 x n_embed]
            m_i_qpole2 = unsorted_segment_sum_3d(m_ij_qpole2, e_source, natom)
            d_qpole = torch.einsum("axf,ayf->axyf", m_i_qpole1, m_i_qpole2)
            d_qpole = d_qpole + d_qpole.permute(0, 2, 1, 3)
            # Paper states 0.5 factor is applied to the sum
            # d_qpole = 0.5 * (d_qpole + d_qpole.permute(0, 2, 1, 3))
            d_qpole = self.qpole_readout_layers[i](d_qpole)
            d_qpole = d_qpole.view(natom, 3, 3)
            qpole += d_qpole

        ####################################
        ### enforce traceless quadrupole ###
        ####################################

        qpole = multipole.ensure_traceless_qpole(qpole)

        ###################################
        ### enforce charge conservation ###
        ###################################

        charge[keep_mask] = filtered_charge
        molecule_ind.requires_grad_(False)
        molecule_ind = molecule_ind.long()
        num_mols = int(molecule_ind.max().item()) + 1 if molecule_ind.numel() > 0 else 1
        total_charge_pred = scatter_sum_compile(
            charge, molecule_ind, num_mols, reduce="sum"
        )
        # return charge, dipole, qpole, h_list

        total_charge_pred = total_charge_pred.squeeze(-1)
        total_charge_err = total_charge_pred - total_charge
        charge_err = torch.repeat_interleave(
            total_charge_err / natom_per_mol.float(), natom_per_mol
        ).unsqueeze(1)
        charge = charge - charge_err
        charge = charge.squeeze(-1)
        # changed to dim=0 from dim=1 for usage in Param fitting # AMW 8/20/25
        # Breaks test_apnet2_train_qcel_molecules_in_memory_transfer test,
        # dimensions no longer correct... figure out another way to fix this. reverting back to dim=1 # AMW 9/17/25
        h_list_stacked = torch.stack(h_list, dim=1)

        # Choose induced dipole calculation method based on use_nn_screening flag
        if self.use_nn_screening:
            # Use full edge indices for induced dipole calculations with NN screening
            e_source_full = batch.edge_index_full[0]
            e_target_full = batch.edge_index_full[1]
            # For NN screening, we need to expand h_list back to full size for get_messages
            # Create full h_list by scattering filtered values back
            h_list_full = []
            for h_layer in h_list:
                h_full = torch.zeros(
                    natom, h_layer.size(-1), device=h_layer.device, dtype=h_layer.dtype
                )
                h_full[keep_mask] = h_layer
                h_list_full.append(h_full)

            # Compute RBF for SHORT-RANGE edges (for NN screening)
            rbf_short = self.distance_layer(
                get_distances(R, R, edge_index[0], edge_index[1])[0]
            )

            induced_dipoles = self.monomer_induced_dipole_torch_NN_screening(
                Z,
                R,
                charge.unsqueeze(1),
                dipole,
                qpole,
                edge_index[0],  # Short-range edges for NN screening
                edge_index[1],
                e_source_full,  # Full edges for induced dipole calculations
                e_target_full,
                h_list=h_list_full,
                rbf_short=rbf_short,  # RBF for short-range edges
                hfvr=hfvr,
                vw=vw,
            )
        else:
            # For standard induced dipole, use regular short-range edges
            induced_dipoles = self.monomer_induced_dipole_torch(
                Z,
                R,
                charge.unsqueeze(1),
                dipole,
                qpole,
                edge_index[0],  # Use short-range edges
                edge_index[1],
                hirshfeld_volume_ratio=hfvr.squeeze(1),
            )
        # print(f"{dipole = }")
        # print(f"{induced_dipoles = }")
        dipole += induced_dipoles
        return charge, dipole, qpole, h_list_stacked


class InducedDipoleModel:
    def __init__(
        self,
        dataset=None,
        pre_trained_model_path=None,
        atomtype_hfvr_model=None,
        atomtype_hfvr_pre_trained_path=None,
        atom_mpnn_model=None,
        atom_mpnn_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        use_nn_screening=False,
        precompute_hfvr=False,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_root="data",
        ds_max_size=None,
        ds_testing=False,
        ds_force_reprocess=False,
        ds_in_memory=True,
        ds_use_lmdb=False,
        model_save_path=None,
    ):
        """
        Create an InducedDipoleModel manager that constructs or restores an InducedDipoleMPNN and prepares dataset, device, and training configuration.
        
        Parameters:
            dataset (optional): Dataset object to use for training/evaluation. If None and ignore_database_null is False, a dataset will be created from ds_root and dataset flags.
            pre_trained_model_path (str, optional): Path to a saved InducedDipoleModel checkpoint. When provided, the model and (optionally) atomtype/AtomMPNN submodules are restored from the checkpoint; other model-building arguments are ignored.
            atomtype_hfvr_model (torch.nn.Module, optional): Model that predicts Hirshfeld volume ratios (HFVR) and valence widths. Used unless precompute_hfvr is True or a checkpoint overrides it.
            atomtype_hfvr_pre_trained_path (str, optional): Path to a pretrained AtomTypeParamMPNN checkpoint to restore an atomtype_hfvr_model.
            atom_mpnn_model (torch.nn.Module, optional): Pretrained AtomMPNN instance to provide charge/dipole/qpole predictive layers. If provided, its weights are used; a checkpoint may also supply this submodule.
            atom_mpnn_pre_trained_path (str, optional): Path to a pretrained AtomMPNN checkpoint to load when atom_mpnn_model is not provided.
            n_message (int): Number of message-passing steps used when constructing a new InducedDipoleMPNN.
            n_rbf (int): Number of radial basis functions for distance encoding.
            n_neuron (int): Hidden size for internal dense layers.
            n_embed (int): Size of atom-type embedding vectors.
            r_cut (float): Cutoff radius (in angstroms) for neighbor edges and distance-based features.
            use_nn_screening (bool): When True, enable additional NN-based short-range screening networks used during induced-dipole computation.
            precompute_hfvr (bool): When True, expect HFVR and valence width fields in the dataset/batch and skip restoring/using an atomtype_hfvr_model.
            use_GPU (bool or None): If True attempts to use CUDA; if False forces CPU; if None auto-detects and uses GPU when available.
            ignore_database_null (bool): If False and dataset is None, the constructor will build/load dataset(s) according to ds_spec_type and ds_use_lmdb.
            ds_spec_type (int): Dataset specification/type selector passed to dataset builders.
            ds_root (str): Root path for dataset storage or preprocessing.
            ds_max_size (int, optional): Maximum number of examples to include when building/processing the dataset.
            ds_testing (bool): If True, build/testing-mode dataset variants (smaller or faster processing).
            ds_force_reprocess (bool): If True, force dataset reprocessing even if cached data exists.
            ds_in_memory (bool): If True, load dataset contents into memory for faster access.
            ds_use_lmdb (bool): If True, prefer LMDB-backed dataset builder for more efficient I/O; can be combined with precompute_hfvr.
            model_save_path (str, optional): Path where trained model checkpoints will be saved during training.
        """
        # if (
        #     not precompute_hfvr
        #     and atomtype_hfvr_model is None
        #     and atomtype_hfvr_pre_trained_path is None
        # ):
        #     raise ValueError(
        #         "Either atomtypeparam_hfvr_model or atomtypeparam_hfvr_pre_trained_path must be provided.\n"
        #         "Without a model predicting hirshfeld volumes, induced dipoles cannot be computed correctly.\n"
        #         "Alternatively, set precompute_hfvr=True and use atomic_induced_dipole_precomputed_dataset."
        #     )
        if torch.cuda.is_available() and use_GPU is not False:
            device = torch.device("cuda:0")
            print("running on the GPU")
        else:
            device = torch.device("cpu")
            print("running on the CPU")

        if atomtype_hfvr_model is not None:
            self.atomtype_hfvr_model = atomtype_hfvr_model

        if atomtype_hfvr_pre_trained_path is not None:
            print(
                f"Loading pre-trained AtomTypeParamMPNN hfvr model from {atomtype_hfvr_pre_trained_path}"
            )
            checkpoint = torch.load(atomtype_hfvr_pre_trained_path, weights_only=False)
            self.atomtype_hfvr_model = AtomTypeParamMPNN(
                n_message=checkpoint["config"]["n_message"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                param_start_mean=checkpoint["config"]["param_start_mean"],
                param_start_std=checkpoint["config"]["param_start_std"],
                n_params=checkpoint["config"].get("n_params", 1),
                r_cut=checkpoint["config"]["r_cut"],
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.atomtype_hfvr_model.load_state_dict(model_state_dict)

        # Load pretrained AtomMPNN model if provided
        # Note: If pre_trained_model_path is provided, it takes priority and
        # atom_mpnn_pre_trained_path will be ignored (model is loaded from saved state)
        if pre_trained_model_path:
            # When loading a pretrained InducedDipoleModel, the AtomMPNN weights
            # are already stored in the model state_dict, so we don't load separately
            print(
                f"Loading pre-trained InducedDipoleModel from {pre_trained_model_path}"
            )
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            # Prioritize user-provided precompute_hfvr over checkpoint value
            # This allows users to switch modes when loading old checkpoints
            checkpoint_precompute = checkpoint["config"].get("precompute_hfvr", False)
            # use_precompute = (
            #     precompute_hfvr if precompute_hfvr else checkpoint_precompute
            # )
            use_precompute = precompute_hfvr

            # Check if checkpoint has atomtype_hfvr_model config
            # If so, restore it from checkpoint (override any passed model/path)
            atomtype_config = checkpoint["config"].get("atomtype_hfvr_config", None)
            if atomtype_config is not None and not use_precompute:
                # Restore atomtype_hfvr_model from checkpoint
                self.atomtype_hfvr_model = AtomTypeParamMPNN(
                    n_message=atomtype_config["n_message"],
                    n_neuron=atomtype_config["n_neuron"],
                    n_embed=atomtype_config["n_embed"],
                    param_start_mean=atomtype_config["param_start_mean"],
                    param_start_std=atomtype_config["param_start_std"],
                    n_params=atomtype_config.get("n_params", 1),
                    r_cut=atomtype_config["r_cut"],
                )
                print(
                    "Note: Checkpoint contains atomtype_hfvr_model, restoring from checkpoint"
                )

            # Check if checkpoint was saved with a pretrained AtomMPNN
            has_pretrained_atom_mpnn = checkpoint["config"].get(
                "has_pretrained_atom_mpnn", False
            )

            # If checkpoint had a pretrained AtomMPNN, create a dummy one to ensure correct architecture
            # The actual weights will be loaded from the state_dict
            if has_pretrained_atom_mpnn:
                from .ap2_atom_model import AtomMPNN

                self.atom_mpnn_model = AtomMPNN(
                    n_message=checkpoint["config"]["n_message"],
                    n_rbf=checkpoint["config"]["n_rbf"],
                    n_neuron=checkpoint["config"]["n_neuron"],
                    n_embed=checkpoint["config"]["n_embed"],
                    r_cut=checkpoint["config"]["r_cut"],
                )
                print(
                    "Note: Checkpoint contains pretrained AtomMPNN, restoring architecture"
                )
            else:
                self.atom_mpnn_model = None

            self.model = InducedDipoleMPNN(
                atomtype_hfvr_model=self.atomtype_hfvr_model
                if not use_precompute
                else None,
                atom_mpnn_model=self.atom_mpnn_model,
                n_message=checkpoint["config"]["n_message"],
                n_rbf=checkpoint["config"]["n_rbf"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                r_cut=checkpoint["config"]["r_cut"],
                use_nn_screening=checkpoint["config"].get("use_nn_screening", False),
                precompute_hfvr=use_precompute,
            )

            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }

            # If switching between precompute modes, filter atomtype_hfvr_model keys
            if use_precompute and not checkpoint_precompute:
                # User wants precompute=True but checkpoint has atomtype_hfvr_model keys
                # Filter them out since new model doesn't have that submodule
                model_state_dict = {
                    k: v
                    for k, v in model_state_dict.items()
                    if not k.startswith("atomtype_hfvr_model.")
                }
                print(
                    "Note: Filtered out atomtype_hfvr_model weights (switching to precompute mode)"
                )
            elif not use_precompute and checkpoint_precompute:
                # User wants precompute=False but checkpoint doesn't have atomtype_hfvr_model
                # This is fine, we'll just load what's available
                print(
                    "Note: Checkpoint was saved with precompute mode, loading available weights"
                )

            self.model.load_state_dict(model_state_dict, strict=False)
            # Store the precompute flag used for model creation
            self.precompute_hfvr = use_precompute
            print("Precompute HFVR:", self.precompute_hfvr)
            print(
                "Using pretrained AtomMPNN model from checkpoint"
                if self.model.atom_mpnn_model is not None
                else "Training InducedDipoleMPNN from scratch"
            )
        else:
            # No pre_trained_model_path, so check if we should load atom_mpnn separately
            if atom_mpnn_model is not None:
                self.atom_mpnn_model = atom_mpnn_model
            elif atom_mpnn_pre_trained_path is not None:
                print(
                    f"Loading pre-trained AtomMPNN model from {atom_mpnn_pre_trained_path}"
                )
                # Import AtomMPNN from ap2_atom_model
                from .ap2_atom_model import AtomMPNN

                checkpoint = torch.load(atom_mpnn_pre_trained_path, weights_only=False)
                self.atom_mpnn_model = AtomMPNN(
                    n_message=checkpoint["config"]["n_message"],
                    n_rbf=checkpoint["config"]["n_rbf"],
                    n_neuron=checkpoint["config"]["n_neuron"],
                    n_embed=checkpoint["config"]["n_embed"],
                    r_cut=checkpoint["config"]["r_cut"],
                )
                model_state_dict = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in checkpoint["model_state_dict"].items()
                }
                self.atom_mpnn_model.load_state_dict(model_state_dict)
            else:
                self.atom_mpnn_model = None

            self.model = InducedDipoleMPNN(
                atomtype_hfvr_model=self.atomtype_hfvr_model
                if not precompute_hfvr
                else None,
                atom_mpnn_model=self.atom_mpnn_model,
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut=r_cut,
                use_nn_screening=use_nn_screening,
                precompute_hfvr=precompute_hfvr,
            )
            # Store the precompute flag used for model creation
            self.precompute_hfvr = precompute_hfvr
            print("Precompute HFVR:", self.precompute_hfvr)
            print(
                "Using pretrained AtomMPNN model for charge, dipole, and quadrupole predictions"
                if self.atom_mpnn_model is not None
                else "Training InducedDipoleMPNN from scratch"
            )
        self.device = device
        self.dataset = dataset
        self.ds_spec_type = ds_spec_type
        mp.set_sharing_strategy("file_system")
        split_dbs = [7]
        if (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type not in split_dbs
        ):
            print("Setting up dataset...")

            def setup_ds(fp=ds_force_reprocess):
                """
                Select and construct the appropriate dataset object based on configuration flags.
                
                Parameters:
                    fp (bool): If true, force reprocessing of the dataset files.
                
                Returns:
                    dataset: An instantiated dataset object. Chooses:
                      - LMDB-backed dataset when `ds_use_lmdb` is true.
                      - Precomputed induced-dipole dataset when `precompute_hfvr` is true.
                      - Standard atomic module dataset otherwise.
                      The chosen dataset is constructed with `root`, `testing`, `spec_type`, `max_size`,
                      `force_reprocess=fp`, and `in_memory` from the surrounding scope. When using the
                      LMDB or precomputed paths, `atomtype_hfvr_model` from `self` is passed if
                      `precompute_hfvr` is true.
                """
                if ds_use_lmdb:
                    # Use LMDB-based dataset
                    from apnet_pt.atomic_datasets import (
                        atomic_module_dataset_lmdb,
                    )

                    return atomic_module_dataset_lmdb(
                        root=ds_root,
                        atomtype_hfvr_model=self.atomtype_hfvr_model
                        if precompute_hfvr
                        else None,
                        testing=ds_testing,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        in_memory=ds_in_memory,
                    )
                elif precompute_hfvr:
                    # Use pre-computed dataset
                    from apnet_pt.atomic_datasets import (
                        atomic_induced_dipole_precomputed_dataset,
                    )

                    return atomic_induced_dipole_precomputed_dataset(
                        root=ds_root,
                        atomtype_hfvr_model=self.atomtype_hfvr_model,
                        testing=ds_testing,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        in_memory=ds_in_memory,
                    )
                else:
                    # Use regular atomic_module_dataset
                    return atomic_module_dataset(
                        root=ds_root,
                        testing=ds_testing,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        in_memory=ds_in_memory,
                    )

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
        elif (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type in split_dbs
        ):
            print("Processing Split dataset...")

            def setup_ds(fp=ds_force_reprocess):
                """
                Create and return training and testing atomic_module_dataset instances.
                
                Parameters:
                    fp: Force-reprocess flag forwarded to the dataset builder (used as `force_reprocess`).
                
                Returns:
                    list: A two-element list [train_dataset, test_dataset] of atomic_module_dataset instances for the "train" and "test" splits.
                """
                return [
                    atomic_module_dataset(
                        root=ds_root,
                        testing=ds_testing,
                        spec_type=ds_spec_type,
                        split="train",
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        in_memory=ds_in_memory,
                    ),
                    atomic_module_dataset(
                        root=ds_root,
                        testing=ds_testing,
                        spec_type=ds_spec_type,
                        split="test",
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        in_memory=ds_in_memory,
                    ),
                ]

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
        print(f"{self.dataset = }")
        self.rank = None
        self.world_size = None
        self.model_save_path = model_save_path
        self.train_shuffle = None
        # torch.jit.enable_onednn_fusion(True)
        return

    def set_pretrained_model(self, model_path=None, model_id=None):
        """
        Load a pretrained checkpoint into this model either from a filesystem path or from a bundled resource ID.
        
        Parameters:
            model_path (str | pathlib.Path | None): Filesystem path to a saved checkpoint file. Ignored if `model_id` is provided.
            model_id (int | str | None): Identifier of a packaged model stored under apnet_pt/models/am_ensemble/am_{model_id}.pt. When provided, this resource path is used instead of `model_path`.
        
        Returns:
            self: The same object with its `model` state dict updated from the checkpoint (allows method chaining).
        
        Raises:
            ValueError: If neither `model_path` nor `model_id` is provided.
        """
        if model_id is not None:
            # model_path = f"{file_dir}/../models/am_ensemble/am_{model_id}.pt"
            model_path = resources.files("apnet_pt").joinpath(
                "models", "am_ensemble", f"am_{model_id}.pt"
            )
        elif model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided.")

        checkpoint = torch.load(model_path, weights_only=False)
        # pp(checkpoint)
        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        return self

    def compile_model(self):
        """
        Enable TorchDynamo dynamic-shape support and compile the internal model for optimized execution.
        
        Configures TorchDynamo to allow dynamic shapes and related capture behavior, then replaces
        self.model with a compiled version suitable for runtime acceleration.
        """
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        self.model = torch.compile(self.model, dynamic=True)
        return

    def setup(self, rank, world_size):
        """
        Initialize distributed process group.

        Uses MASTER_ADDR and MASTER_PORT from environment if set (for SLURM),
        otherwise defaults to localhost:12355 for local testing.

        Parameters
        ----------
        rank : int
            Rank of this process
        world_size : int
            Total number of processes
        """
        # Use environment variables if set (SLURM sets these), otherwise use defaults
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        # Initialize process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        # torch.manual_seed(42)

    def cleanup(self):
        """
        Shut down the PyTorch distributed process group and release associated distributed resources.
        
        This terminates the current process group used for DDP, making the process no longer participate in distributed communication.
        """
        dist.destroy_process_group()

    def _qcel_example_input(self, mols, batch_size=1):
        """
        Create batched PyG data objects from QCElemental molecules for model input.
        
        Converts each molecule in `mols` into a PyG-style data object with full atom indexing and groups them into batches of size `batch_size`. Returns a list where each element is a batched data object ready for the model's collate/forward pipeline.
        
        Parameters:
            mols (Sequence): Iterable of QCElemental molecule objects convertible to PyG data.
            batch_size (int): Number of molecules per returned batch.
        
        Returns:
            list: Batched PyG data objects, one entry per batch.
        """
        mol_data = [qcel_mon_to_pyg_data(mol, full_indices=True) for mol in mols]
        batches = []
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            batch_A = atomic_collate_update_no_target(batch_mol_data)
            batches.append(batch_A)
        return batches

    def example_input(self):
        """
        Create a minimal example batch containing a single water-like molecule for use as model input.
        
        Returns:
            batch: A batched data object (converted from a QCEngine Molecule) representing one molecule, ready to pass to the model.
        """
        mol = qcel.models.Molecule.from_data("""
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
units angstrom
        """)
        return self._qcel_example_input([mol], batch_size=1)

    def evaluate_model_collate_train(self, data_loader, optimizer=None, loss_fn=None):
        """
        Performs a training pass over data_loader, updates model parameters, and returns aggregated per-atom prediction errors.
        
        The model is put into training mode and for each batch the function computes charge, dipole, and quadrupole losses (sum of the three), backpropagates, and steps the optimizer. If loss_fn is None, elementwise mean squared error is used; otherwise the provided loss_fn is applied (dipole and quadrupole losses are averaged over their components).
        
        Parameters:
            data_loader: Iterable of batched data objects placed on self.device. Each batch must provide .charges, .dipoles, and .quadrupoles for supervision.
            optimizer: Optimizer used to update model parameters (required).
            loss_fn (optional): Loss function or criterion taking (prediction, target). If None, mean squared error is used.
        
        Returns:
            total_loss (float): Sum of batch losses accumulated over the entire data_loader (detached Python float).
            charge_errors_t (torch.Tensor): Concatenated per-atom charge prediction errors (prediction - target) for all batches.
            dipole_errors_t (torch.Tensor): Concatenated per-atom dipole prediction errors for all batches.
            qpole_errors_t (torch.Tensor): Concatenated per-atom quadrupole prediction errors for all batches.
        """
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
        """
        Evaluate the model on an evaluation DataLoader and collect per-component errors and aggregate loss.
        
        Runs the model in eval mode without gradient tracking over every batch from `data_loader`, computes per-batch losses for charges, dipoles, and quadrupoles (using `loss_fn` if provided, otherwise mean squared error), accumulates the total loss, and gathers prediction errors for each component.
        
        Parameters:
            data_loader: Iterable[torch.Tensor] or DataLoader
                Batched evaluation data objects providing attributes `charges`, `dipoles`, and `quadrupoles`.
            loss_fn: callable, optional
                Loss function to apply to predictions (e.g., a PyTorch criterion). If omitted, mean squared error is used.
        
        Returns:
            total_loss (torch.Tensor):
                Sum of batch losses accumulated over the dataset (detached).
            charge_errors_t (torch.Tensor):
                Concatenated per-atom charge prediction errors moved to CPU.
            dipole_errors_t (torch.Tensor):
                Concatenated per-atom dipole prediction errors moved to CPU.
            qpole_errors_t (torch.Tensor):
                Concatenated per-atom quadrupole prediction errors moved to CPU.
        """
        charge_errors_t, dipole_errors_t, qpole_errors_t = [], [], []
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                # print('saving batch for debug')
                # torch.save(batch, "debug_batch.pt")
                batch_loss = 0.0
                batch = batch.to(self.device)
                charge, dipole, qpole, hlist = self.model(batch)

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
                total_loss += batch_loss.detach()

            charge_errors_t.append(q_error.detach().cpu())
            dipole_errors_t.extend(d_error.detach().cpu())
            qpole_errors_t.extend(qp_error.detach().cpu())
        charge_errors_t = torch.cat(charge_errors_t)
        dipole_errors_t = torch.cat(dipole_errors_t)
        qpole_errors_t = torch.cat(qpole_errors_t)
        return total_loss, charge_errors_t, dipole_errors_t, qpole_errors_t

    def pretrain_statistics(self, train_loader, test_loader, criterion):
        """
        Evaluate pretraining performance on the provided training and test loaders and print mean absolute errors (MAE) for charges, dipoles, and quadrupoles.
        
        Parameters:
            train_loader: Iterable or DataLoader yielding training batches for evaluation.
            test_loader: Iterable or DataLoader yielding validation/test batches for evaluation.
            criterion: Loss function (unused directly here but kept for API compatibility).
        
        Returns:
            test_loss: Scalar loss computed over the test_loader.
        """
        t1 = time.time()
        with torch.no_grad():
            _, charge_errors_t, dipole_errors_t, qpole_errors_t = (
                self.evaluate_model_collate_eval(
                    train_loader,  # loss_fn=criterion
                )
            )
            charge_MAE_t = np.mean(np.abs(charge_errors_t.numpy()))
            dipole_MAE_t = np.mean(np.abs(dipole_errors_t.numpy()))
            qpole_MAE_t = np.mean(np.abs(qpole_errors_t.numpy()))

            charge_errors_t, dipole_errors_t, qpole_errors_t = [], [], []
            test_loss, charge_errors_v, dipole_errors_v, qpole_errors_v = (
                self.evaluate_model_collate_eval(
                    test_loader,  # loss_fn=criterion
                )
            )
            charge_MAE_v = np.mean(np.abs(charge_errors_v.numpy()))
            dipole_MAE_v = np.mean(np.abs(dipole_errors_v.numpy()))
            qpole_MAE_v = np.mean(np.abs(qpole_errors_v.numpy()))
            charge_errors_v, dipole_errors_v, qpole_errors_v = [], [], []
            dt = time.time() - t1
            print(
                f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {charge_MAE_t:>7.4f}/{charge_MAE_v:<7.4f} {dipole_MAE_t:>7.4f}/{dipole_MAE_v:<7.4f} {qpole_MAE_t:>7.4f}/{qpole_MAE_v:<7.4f}",
                flush=True,
            )
        return test_loss

    def pretrain_statistics_ddp(
        self, rank, train_loader, test_loader, criterion, rank_device, world_size
    ):
        """
        Compute pretraining MAE statistics across distributed processes and print results on the main rank.
        
        Evaluates the provided training and test dataloaders, aggregates metrics across all processes, and prints aggregated MAEs on rank 0 (or when world_size == 1).
        
        Parameters:
            rank (int): Process rank within the distributed group.
            train_loader (DataLoader): DataLoader for the training partition.
            test_loader (DataLoader): DataLoader for the validation/test partition.
            criterion (torch.nn.Module): Loss function used to evaluate batches.
            rank_device (torch.device): Device assigned to this rank.
            world_size (int): Total number of distributed processes.
        
        Returns:
            test_loss (float): Test loss averaged across all processes.
        """
        t1 = time.time()

        with torch.no_grad():
            # Use evaluate_batches which handles all_reduce across processes
            _, charge_MAE_t, dipole_MAE_t, qpole_MAE_t = self.evaluate_batches(
                rank, train_loader, criterion, rank_device
            )

            test_loss, charge_MAE_v, dipole_MAE_v, qpole_MAE_v = self.evaluate_batches(
                rank, test_loader, criterion, rank_device
            )

            # Only print on rank 0
            if rank == 0 or world_size == 1:
                dt = time.time() - t1
                print(
                    f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {charge_MAE_t:>7.4f}/{charge_MAE_v:<7.4f} "
                    f"{dipole_MAE_t:>7.4f}/{dipole_MAE_v:<7.4f} {qpole_MAE_t:>7.4f}/{qpole_MAE_v:<7.4f}",
                    flush=True,
                )

        return test_loss

    def train_batches_single_proc(
        self, rank, dataloader, criterion, optimizer, rank_device
    ):
        """
        Train the model for one epoch on a single process and accumulate mean absolute errors.
        
        Parameters:
            rank: Process rank identifier (unused for computation but used for logging/coordination).
            dataloader: Iterable of batched training examples moved to the device; each batch must provide .charges, .dipoles, .quadrupoles and be compatible with model(batch).
            criterion: Loss function placeholder (not used directly; losses are computed inside the method).
            optimizer: Optimizer used to update model parameters.
            rank_device: torch.device where computations and metric accumulations are performed.
        
        Returns:
            total_loss (float): Sum of per-batch scalar losses over the epoch.
            charge_mae (float): Mean absolute error per scalar charge across all atoms in the epoch.
            dipole_mae (float): Mean absolute error per dipole component (x,y,z) across all atoms in the epoch.
            qpole_mae (float): Mean absolute error per quadrupole component (9 components) across all atoms in the epoch.
        """
        self.model.train()
        total_charge_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_dipole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_qpole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        for batch in dataloader:
            batch = batch.to(rank_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # print('saving batch for debug')
            charge, dipole, qpole, _ = self.model(batch)
            if torch.isnan(dipole).any():
                torch.save(batch, "debug_batch.pt")
                raise ValueError("NaN detected in dipole predictions")

            q_error = charge - batch.charges
            d_error = dipole - batch.dipoles
            qp_error = qpole - batch.quadrupoles

            charge_loss = (q_error**2).mean()
            dipole_loss = (d_error**2).mean()
            qpole_loss = (qp_error**2).mean()

            loss = charge_loss + dipole_loss + qpole_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += q_error.numel()

            total_charge_error += q_error.abs().sum()
            total_dipole_error += d_error.abs().sum()
            total_qpole_error += qp_error.abs().sum()

        final_count = total_count.item()

        # Calculating MAEs
        charge_mae = total_charge_error.item() / final_count
        dipole_mae = total_dipole_error.item() / (final_count * 3)
        qpole_mae = total_qpole_error.item() / (final_count * 9)
        return total_loss, charge_mae, dipole_mae, qpole_mae

    def train_batches(self, rank, dataloader, criterion, optimizer, rank_device):
        """
        Perform a training pass over the provided dataloader, updating model weights and computing distributed mean absolute errors.
        
        Parameters:
            rank (int): Process rank within distributed training.
            dataloader (Iterable): Iterable of batched data objects; each batch must contain target attributes `.charges`, `.dipoles`, and `.quadrupoles`.
            criterion: Loss function (not used explicitly here; kept for API consistency).
            optimizer: Optimizer used for parameter updates.
            rank_device (torch.device): Device to move batches and reduction tensors to.
        
        Returns:
            total_loss (float): Sum of per-batch loss values accumulated on this process.
            charge_mae (float): Mean absolute error for charges across all processes.
            dipole_mae (float): Mean absolute error for dipoles (averaged over 3 components) across all processes.
            qpole_mae (float): Mean absolute error for quadrupoles (averaged over 9 components) across all processes.
        """
        self.model.train()
        total_charge_error = 0
        total_dipole_error = 0
        total_qpole_error = 0
        total_loss = 0
        count = 0

        for batch in dataloader:
            batch = batch.to(rank_device)
            optimizer.zero_grad()
            charge, dipole, qpole, _ = self.model(batch)

            q_error = charge - batch.charges
            d_error = dipole - batch.dipoles
            qp_error = qpole - batch.quadrupoles

            charge_loss = torch.mean(torch.square(q_error))
            dipole_loss = torch.mean(torch.square(d_error))
            qpole_loss = torch.mean(torch.square(qp_error))

            loss = charge_loss + dipole_loss + qpole_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += q_error.numel()

            total_charge_error += torch.sum(torch.abs(q_error)).item()
            total_dipole_error += torch.sum(torch.abs(d_error)).item()
            total_qpole_error += torch.sum(torch.abs(qp_error)).item()

        # Converting to tensors for all-reduce
        total_charge_error = torch.tensor(
            total_charge_error, dtype=torch.float32, device=rank_device
        )
        total_dipole_error = torch.tensor(
            total_dipole_error, dtype=torch.float32, device=rank_device
        )
        total_qpole_error = torch.tensor(
            total_qpole_error, dtype=torch.float32, device=rank_device
        )
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_charge_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_dipole_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_qpole_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        # Calculating MAEs
        charge_mae = total_charge_error.item() / count.item()
        dipole_mae = total_dipole_error.item() / (count.item() * 3)
        qpole_mae = total_qpole_error.item() / (count.item() * 9)

        return total_loss, charge_mae, dipole_mae, qpole_mae

    def evaluate_batches_single_proc(self, rank, dataloader, criterion, rank_device):
        """
        Evaluate the model on a dataloader in single-process mode and compute aggregate loss and mean absolute errors for predicted multipoles.
        
        Parameters:
            rank (int): Process rank for this evaluation (used for device placement).
            dataloader (torch.utils.data.DataLoader): Iterable providing batches compatible with the model.
            criterion (callable or None): Loss function (accepted for API compatibility; not used by this implementation).
            rank_device (torch.device): Device to run the evaluation on.
        
        Returns:
            total_loss (float): Sum of per-batch squared losses (charge + dipole + quadrupole) accumulated over the dataloader.
            charge_mae (float): Mean absolute error per atomic charge (averaged over all atoms in the dataloader).
            dipole_mae (float): Mean absolute error per dipole component (averaged over all atoms and the three Cartesian components).
            qpole_mae (float): Mean absolute error per quadrupole component (averaged over all atoms and the nine tensor components).
        """
        self.model.eval()
        total_charge_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_dipole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_qpole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        with torch.no_grad():
            for batch in dataloader:
                # print('saving batch for debug')
                # torch.save(batch, "debug_batch.pt")
                batch = batch.to(rank_device, non_blocking=True)
                charge, dipole, qpole, _ = self.model(batch)

                q_error = charge - batch.charges
                d_error = dipole - batch.dipoles
                qp_error = qpole - batch.quadrupoles

                charge_loss = (q_error**2).mean()
                dipole_loss = (d_error**2).mean()
                qpole_loss = (qp_error**2).mean()

                loss = charge_loss + dipole_loss + qpole_loss
                total_loss += loss.item()
                total_count += q_error.numel()

                total_charge_error += q_error.abs().sum()
                total_dipole_error += d_error.abs().sum()
                total_qpole_error += qp_error.abs().sum()

        final_count = total_count.item()

        # Calculating MAEs
        charge_mae = total_charge_error.item() / final_count
        dipole_mae = total_dipole_error.item() / (final_count * 3)
        qpole_mae = total_qpole_error.item() / (final_count * 9)
        return total_loss, charge_mae, dipole_mae, qpole_mae

    def evaluate_batches(self, rank, dataloader, criterion, rank_device):
        """
        Evaluate the model on batches from a dataloader and return the aggregated loss and per-component mean absolute errors (MAEs) across distributed processes.
        
        Runs the model in eval mode without gradient computation, accumulates per-atom absolute errors and squared-loss terms for charges, dipoles, and quadrupoles over the provided dataloader, performs an all-reduce to aggregate these values across processes, and computes MAEs per atom for each multipole component (dipole MAE is averaged over 3 components, quadrupole MAE over 9 components).
        
        Parameters:
            rank (int): Process rank (unused in computation but preserved for API symmetry).
            dataloader (Iterable): DataLoader yielding batches compatible with the model; each batch must expose .to(device) and contain .charges, .dipoles, and .quadrupoles targets.
            criterion: Loss function placeholder (not used inside this method but kept for interface compatibility).
            rank_device (torch.device): Device on which tensors for this rank are allocated and where all-reduce is performed.
        
        Returns:
            total_loss (torch.Tensor): Sum of per-batch losses aggregated across all distributed processes.
            charge_mae (float): Mean absolute error per atom for charges.
            dipole_mae (float): Mean absolute error per atom per dipole component (averaged over 3 components).
            qpole_mae (float): Mean absolute error per atom per quadrupole component (averaged over 9 components).
        """
        self.model.eval()
        total_charge_error = 0
        total_dipole_error = 0
        total_qpole_error = 0
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                # print('saving batch for debug')
                # torch.save(batch, "debug_batch.pt")
                batch = batch.to(rank_device)
                charge, dipole, qpole, _ = self.model(batch)

                q_error = charge - batch.charges
                d_error = dipole - batch.dipoles
                qp_error = qpole - batch.quadrupoles

                total_charge_error += torch.sum(torch.abs(q_error)).item()
                total_dipole_error += torch.sum(torch.abs(d_error)).item()
                total_qpole_error += torch.sum(torch.abs(qp_error)).item()

                charge_loss = torch.mean(torch.square(q_error))
                dipole_loss = torch.mean(torch.square(d_error))
                qpole_loss = torch.mean(torch.square(qp_error))

                total_loss += charge_loss + dipole_loss + qpole_loss
                count += q_error.numel()

        # Converting to tensors for all-reduce
        total_charge_error = torch.tensor(
            total_charge_error, dtype=torch.float32, device=rank_device
        )
        total_dipole_error = torch.tensor(
            total_dipole_error, dtype=torch.float32, device=rank_device
        )
        total_qpole_error = torch.tensor(
            total_qpole_error, dtype=torch.float32, device=rank_device
        )
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_charge_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_dipole_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_qpole_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        total_loss = torch.tensor(total_loss.item(), device=rank_device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        # Calculating MAEs
        charge_mae = total_charge_error.item() / count.item()
        dipole_mae = total_dipole_error.item() / (count.item() * 3)
        qpole_mae = total_qpole_error.item() / (count.item() * 9)

        return total_loss, charge_mae, dipole_mae, qpole_mae

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
        """
        Run distributed (or single-process) training loop for the internal model, evaluating on a validation set and saving the best checkpoint.
        
        Parameters:
            rank (int): Process rank for distributed training (0 for main process).
            world_size (int): Total number of processes participating in DDP; use 1 for single-process.
            train_dataset (torch.utils.data.Dataset): Dataset used for training.
            test_dataset (torch.utils.data.Dataset): Dataset used for validation/testing.
            n_epochs (int): Number of training epochs to run.
            batch_size (int): Batch size for both training and testing dataloaders.
            lr (float): Learning rate for the Adam optimizer.
            pin_memory (bool): Passed to DataLoader to enable pinned memory.
            num_workers (int): Number of worker processes for data loading.
        
        Behavior:
            - Initializes distributed process group when world_size > 1 and places the model on the appropriate device.
            - Builds train and test AtomicDataLoader instances with the correct collate function depending on HFVR precomputation.
            - Runs the training loop for n_epochs, performing training and evaluation each epoch.
            - Tracks the lowest validation loss on rank 0 and saves the model checkpoint (including minimal atomtype HFVR config when present) to self.model_save_path when improved.
            - Cleans up the distributed process group when used.
        
        Side effects:
            - May call self.setup and self.cleanup for DDP.
            - May wrap the model in DDP.
            - May save model state to disk at self.model_save_path.
        """
        if self.device.type == "cpu":
            # rank = "cpu"
            rank_device = "cpu"
        else:
            rank_device = rank
        if world_size > 1:
            self.setup(rank, world_size)

        self.model.to(rank_device)
        if world_size > 1 and rank_device == "cpu":
            # NOTE: torch.compile() with DDP on CPU can cause segfaults
            # Skip compilation for CPU DDP training
            # torch._dynamo.config.dynamic_shapes = True
            # torch._dynamo.config.capture_dynamic_output_shape_ops = True
            # torch._dynamo.config.capture_scalar_outputs = True
            # self.model = torch.compile(self.model, dynamic=True)
            self.model = DDP(
                self.model,
            )
        # elif rank_device != "cpu":
        #     self.model = DDP(
        #         self.model,
        #         device_ids=[rank],
        #         output_device=rank_device,
        #     )

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

        # Use appropriate collate function based on precompute_hfvr
        if self.precompute_hfvr:
            from apnet_pt.atomic_datasets import atomic_hirshfeld_collate_update

            collate_fn = atomic_hirshfeld_collate_update
        else:
            collate_fn = atomic_collate_update

        train_loader = AtomicDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            collate_fn=collate_fn,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=collate_fn,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        test_loss = self.pretrain_statistics_ddp(
            rank, train_loader, test_loader, criterion, rank_device, world_size
        )

        lowest_test_loss = test_loss

        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            train_loss, charge_MAE_t, dipole_MAE_t, qpole_MAE_t = self.train_batches(
                rank, train_loader, criterion, optimizer, rank_device
            )
            test_loss, charge_MAE_v, dipole_MAE_v, qpole_MAE_v = self.evaluate_batches(
                rank, test_loader, criterion, rank_device
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        # cpu_model = self.model.to("cpu")
                        cpu_model = unwrap_model(self.model).to("cpu")

                        # Save atomtype_hfvr_model config if it exists
                        atomtype_config = None
                        if cpu_model.atomtype_hfvr_model is not None:
                            atomtype_config = {
                                "n_message": cpu_model.atomtype_hfvr_model.n_message,
                                "n_neuron": cpu_model.atomtype_hfvr_model.n_neuron,
                                "n_embed": cpu_model.atomtype_hfvr_model.n_embed,
                                "param_start_mean": cpu_model.atomtype_hfvr_model.param_start_mean,
                                "param_start_std": cpu_model.atomtype_hfvr_model.param_start_std,
                                "n_params": cpu_model.atomtype_hfvr_model.n_params,
                                "r_cut": cpu_model.atomtype_hfvr_model.r_cut,
                            }
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_rbf": cpu_model.n_rbf,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "r_cut": cpu_model.r_cut,
                                    "use_nn_screening": cpu_model.use_nn_screening,
                                    "precompute_hfvr": cpu_model.precompute_hfvr,
                                    "has_pretrained_atom_mpnn": cpu_model.atom_mpnn_model
                                    is not None,
                                    "atomtype_hfvr_config": atomtype_config,
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
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {charge_MAE_t:>7.4f}/{charge_MAE_v:<7.4f} {dipole_MAE_t:>7.4f}/{dipole_MAE_v:<7.4f} {qpole_MAE_t:>7.4f}/{qpole_MAE_v:<7.4f} {test_lowered}",
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
        skip_compile=True,
    ):
        """
        Train the model on a single process and save the best-performing checkpoint.
        
        Runs a training loop over the provided datasets for the given number of epochs, evaluates on the test set each epoch, and saves the model state (and atomtype HFVR config if present) when test loss improves. Selects the collate function based on the instance's precompute_hfvr setting and moves the model to the appropriate device; optionally compiles the model before training.
        
        Parameters:
            rank: Process rank used for device assignment and logging (0 prints/save).
            world_size: Total number of processes (cleanup is invoked if > 1 after training).
            train_dataset: Dataset used for training.
            test_dataset: Dataset used for validation/testing.
            n_epochs: Number of training epochs to run.
            batch_size: Batch size for data loaders.
            lr: Learning rate for the Adam optimizer.
            pin_memory: Passed to data loaders to enable pinned memory.
            num_workers: Number of worker processes for data loading.
            skip_compile: If False, compile the model before training; if True, skip compilation.
        """
        if self.device.type == "cpu":
            rank_device = "cpu"
        else:
            rank_device = rank

        self.model.to(rank_device)
        if not skip_compile:
            self.compile_model()

        # Use appropriate collate function based on precompute_hfvr
        if self.precompute_hfvr:
            from apnet_pt.atomic_datasets import atomic_hirshfeld_collate_update

            collate_fn = atomic_hirshfeld_collate_update
        else:
            collate_fn = atomic_collate_update

        train_loader = AtomicDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=self.train_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        lowest_test_loss = torch.tensor(float("inf"))
        # test_loss = self.pretrain_statistics(train_loader, test_loader, criterion)

        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            train_loss, charge_MAE_t, dipole_MAE_t, qpole_MAE_t = (
                self.train_batches_single_proc(
                    rank, train_loader, criterion, optimizer, rank_device
                )
            )
            test_loss, charge_MAE_v, dipole_MAE_v, qpole_MAE_v = (
                self.evaluate_batches_single_proc(
                    rank, test_loader, criterion, rank_device
                )
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        # cpu_model = self.model.to("cpu")
                        cpu_model = unwrap_model(self.model).to("cpu")

                        # Save atomtype_hfvr_model config if it exists
                        atomtype_config = None
                        if cpu_model.atomtype_hfvr_model is not None:
                            atomtype_config = {
                                "n_message": cpu_model.atomtype_hfvr_model.n_message,
                                "n_neuron": cpu_model.atomtype_hfvr_model.n_neuron,
                                "n_embed": cpu_model.atomtype_hfvr_model.n_embed,
                                "param_start_mean": cpu_model.atomtype_hfvr_model.param_start_mean,
                                "param_start_std": cpu_model.atomtype_hfvr_model.param_start_std,
                                "n_params": cpu_model.atomtype_hfvr_model.n_params,
                                "r_cut": cpu_model.atomtype_hfvr_model.r_cut,
                            }
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_rbf": cpu_model.n_rbf,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "r_cut": cpu_model.r_cut,
                                    "use_nn_screening": cpu_model.use_nn_screening,
                                    "precompute_hfvr": cpu_model.precompute_hfvr,
                                    "has_pretrained_atom_mpnn": cpu_model.atom_mpnn_model
                                    is not None,
                                    "atomtype_hfvr_config": atomtype_config,
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
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {charge_MAE_t:>7.4f}/{charge_MAE_v:<7.4f} {dipole_MAE_t:>7.4f}/{dipole_MAE_v:<7.4f} {qpole_MAE_t:>7.4f}/{qpole_MAE_v:<7.4f} {test_lowered}",
                    flush=True,
                )
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
        skip_compile=False,
        shuffle=True,
        dataloader_num_workers=0,
        world_size=1,  # Default to 1 for single-core operation
        omp_num_threads_per_process=None,
        random_seed=42,
    ):
        """
        Train the InducedDipoleModel using the provided or stored dataset.
        
        Splits the dataset into training and test subsets (unless a two-element list of pre-split datasets is provided), sets random seeds, prepares dataloaders, and runs either single-process or multi-process (DDP) training. Saves the model to `model_path` if provided and prints network/training hyperparameters.
        
        Parameters:
            dataset (optional): Dataset or sequence containing two datasets [train, test]. If None, uses self.dataset.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and evaluation.
            lr (float): Learning rate.
            split_percent (float): Fraction of data to use for training when a single dataset is provided.
            model_path (str or None): File path to save the trained model state; if provided, sets self.model_save_path.
            skip_compile (bool): If True, skips certain compilation/optimization steps before training.
            shuffle (bool): If True, shuffles datasets before splitting or sampling.
            dataloader_num_workers (int): Number of worker processes for data loading.
            world_size (int): Number of processes for distributed data-parallel training (1 for single-process).
            omp_num_threads_per_process (int or None): Value to set OMP_NUM_THREADS for each spawned process.
            random_seed (int or None): Random seed for NumPy and PyTorch; if falsy, random seeding is skipped.
        """
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

        if random_seed:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

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
        else:
            if shuffle:
                order_indices = np.random.permutation(len(self.dataset))
            else:
                order_indices = np.arange(len(self.dataset))
            train_indices = order_indices[: int(len(self.dataset) * split_percent)]
            test_indices = order_indices[int(len(self.dataset) * split_percent) :]
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
        print(f"  {self.model.n_rbf=}", flush=True)
        print(f"  {self.model.r_cut=}", flush=True)
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
        """
        Run the model on a batched graph and return predicted atomic multipoles and hidden states.
        
        Parameters:
            batch: A batched PyG Data object placed on the CPU; this function will move it to the model device.
            isolate_predictions (bool): If True, group per-atom predictions by molecule and return the grouped structure produced by isolate_atomic_property_predictions; if False, return raw per-atom tensors.
        
        Returns:
            If isolate_predictions is True:
                The grouped per-molecule atomic predictions as returned by isolate_atomic_property_predictions for (charge, dipole, quadrupole, hidden_states).
            If isolate_predictions is False:
                A 4-tuple of tensors (qA, muA, thA, hlistA), all on CPU:
                    qA: per-atom charges tensor,
                    muA: per-atom dipole tensor,
                    thA: per-atom quadrupole tensor,
                    hlistA: per-atom hidden-state tensor.
        """
        batch.to(self.device)
        self.model.to(self.device)
        qA, muA, thA, hlistA = self.model(batch)
        batch = batch.cpu()
        qA = qA.detach().detach().cpu()
        muA = muA.detach().detach().cpu()
        thA = thA.detach().detach().cpu()
        hlistA = hlistA.detach().cpu()
        if isolate_predictions:
            return isolate_atomic_property_predictions(batch, (qA, muA, thA, hlistA))
        else:
            return qA, muA, thA, hlistA

    @torch.inference_mode()
    def predict_multipoles_dataset(
        self,
        batch_size=16,
        dataloader_num_workers=0,
        world_size=1,  # Default to 1 for single-process operation
        # omp_num_threads_per_process=None,
    ):
        """
        Predict atomic multipoles for the entire dataset and return grouped per-molecule predictions for each batch.
        
        This iterates over the dataset with a DataLoader (no shuffling), runs the model on each batch, and reassembles per-atom predictions into per-molecule lists using batch.molecule_ind. Multi-process prediction (world_size > 1) is not supported and will raise NotImplementedError.
        
        Parameters:
            batch_size (int): Number of examples per batch passed to the DataLoader.
            dataloader_num_workers (int): Number of worker processes for the DataLoader (currently unused in single-process mode).
            world_size (int): Number of parallel processes for prediction; only 1 (single-process) is supported.
        
        Returns:
            output (list): A list with one entry per processed batch. Each entry is a tuple
                (mol_charges, mol_dipoles, mol_qpoles, hlists) where
                - mol_charges: list of length `batch_size` of lists of per-atom charge predictions for each molecule in the batch,
                - mol_dipoles: list of length `batch_size` of lists of per-atom dipole predictions for each molecule,
                - mol_qpoles: list of length `batch_size` of lists of per-atom quadrupole predictions for each molecule,
                - hlists: the model-produced hidden-state list for the batch (structure as returned by model_predict).
        """
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
                charges, dipoles, qpoles, hlists = self.model_predict(batch)
                # need to use batch.molecule_ind to reassemble the output
                mol_charges = [[] for i in range(batch_size)]
                mol_dipoles = [[] for i in range(batch_size)]
                mol_qpoles = [[] for i in range(batch_size)]
                for n, i in enumerate(batch.molecule_ind):
                    mol_charges[i].append(charges[n])
                    mol_dipoles[i].append(dipoles[n])
                    mol_qpoles[i].append(qpoles[n])
                output.append((mol_charges, mol_dipoles, mol_qpoles, hlists))
        return output

    @torch.inference_mode()
    def predict_qcel_mols(self, mols, batch_size=2):
        """
        Predict atomic multipoles for a sequence of qcel molecules in batches.
        
        Parameters:
            mols (iterable): Iterable of qcel molecule objects to predict.
            batch_size (int): Number of molecules processed per inference batch.
        
        Returns:
            list: A list of tuples (charges, dipoles, qpoles, hlist) for each input molecule.
                  Each tuple contains the per-atom predictions produced by the model:
                  - charges: per-atom charge tensor
                  - dipoles: per-atom dipole tensor
                  - qpoles: per-atom quadrupole tensor
                  - hlist: per-atom hidden-state list or tensor
        """
        output = []
        mol_data = []
        cnt = 0
        for mol in mols:
            data = qcel_mon_to_pyg_data(mol, full_indices=True)
            mol_data.append(data)
            cnt += 1
            if len(mol_data) == batch_size or cnt == len(mols):
                batch = atomic_collate_update_no_target(mol_data)
                with torch.no_grad():
                    charge, dipole, qpole, hlist = self.model(batch)
                    # Isolate atomic properties by molecule
                    mol_charges, mol_dipoles, mol_qpoles, mol_hlists = (
                        isolate_atomic_property_predictions(
                            batch, (charge, dipole, qpole, hlist)
                        )
                    )
                    output.extend(
                        list(zip(mol_charges, mol_dipoles, mol_qpoles, mol_hlists))
                    )
                mol_data = []
        return output

    @torch.inference_mode()
    def predict_qcel_mols_dimer(self, mols, batch_size=2):
        """
        Generate model predictions for dimers and their constituent monomers.
        
        Parameters:
            mols (Sequence): Iterable of QCEngine-compatible dimer molecules where each molecule contains two fragments.
            batch_size (int): Number of molecules per batch for prediction.
        
        Returns:
            tuple: A 3-tuple (dimer_output, monA_output, monB_output) where each element is a sequence of per-molecule model predictions produced by predict_qcel_mols for the corresponding inputs:
                - dimer_output: predictions for the full dimers in `mols`.
                - monA_output: predictions for the first fragment of each dimer.
                - monB_output: predictions for the second fragment of each dimer.
        """
        monA = [mol.get_fragment([0]) for mol in mols]
        monB = [mol.get_fragment([1]) for mol in mols]
        dimer_output = self.predict_qcel_mols(mols, batch_size=batch_size)
        monA_output = self.predict_qcel_mols(monA, batch_size=batch_size)
        monB_output = self.predict_qcel_mols(monB, batch_size=batch_size)
        return dimer_output, monA_output, monB_output

    def predict_elst_ind_dimer(self, mols, batch_size=2):
        """
        Compute electrostatic and induction energies for dimers formed from input molecules.
        
        For each input molecule, predicts multipoles for isolated monomers and for the dimer, evaluates the electrostatic energy of the monomer pair and of the dimer pair, and returns the induction energy as the difference (dimer minus monomer pair).
        
        Parameters:
            mols (Sequence): Iterable of QCEl-compatible molecule objects describing dimers (each must expose `fragments` to index sub-molecules).
            batch_size (int): Number of molecules to process per prediction batch.
        
        Returns:
            tuple:
                - E_elst (list[float]): Electrostatic energies computed using monomer-predicted multipoles for each input molecule.
                - E_elst_dimer (list[float]): Electrostatic energies computed using multipoles predicted on the full dimer for each input molecule.
                - E_induction (list[float]): Induction energies computed as E_elst_dimer - E_elst for each input molecule.
        """
        E_elst, E_elst_dimer, E_induction = [], [], []
        dimer, monA, monB = self.predict_qcel_mols_dimer(
            mols,
            batch_size=batch_size,
        )
        for i, m in enumerate(mols):
            qA, muA, thetaA = (
                monA[i][0].detach().numpy(),
                monA[i][1].numpy(),
                monA[i][2].numpy(),
            )
            qB, muB, thetaB = (
                monB[i][0].detach().numpy(),
                monB[i][1].numpy(),
                monB[i][2].numpy(),
            )
            elst = multipole.eval_qcel_dimer(
                m,
                qA,
                muA,
                thetaA,
                qB,
                muB,
                thetaB,
            )
            qD, muD, thetaD = dimer[i][0], dimer[i][1], dimer[i][2]
            qA, muA, thetaA = (
                qD[m.fragments[0]].detach().numpy(),
                muD[m.fragments[0], :].numpy(),
                thetaD[m.fragments[0], :, :].numpy(),
            )
            qB, muB, thetaB = (
                qD[m.fragments[1]].detach().numpy(),
                muD[m.fragments[1], :].numpy(),
                thetaD[m.fragments[1], :, :].numpy(),
            )
            elst_dimer = multipole.eval_qcel_dimer(
                m,
                qA,
                muA,
                thetaA,
                qB,
                muB,
                thetaB,
            )
            indu = elst_dimer - elst
            E_elst.append(elst)
            E_elst_dimer.append(elst_dimer)
            E_induction.append(indu)
        return E_elst, E_elst_dimer, E_induction

    @torch.inference_mode()
    def model_predict(self, data):
        """
        Run the internal InducedDipoleMPNN on a batched graph data object to produce per-atom multipole predictions and intermediate hidden representations.
        
        Parameters:
            data (object): A batched graph-like object (e.g., PyG Data/batch) that must provide the following attributes used by the model:
                - x: node feature tensor
                - edge_index: edge index tensor
                - R: node coordinate tensor
                - molecule_ind: tensor mapping nodes to molecule indices
                - total_charge: per-molecule total charge tensor
                - natom_per_mol: number of atoms per molecule tensor
        
        Returns:
            tuple: (charge, dipole, qpole, hlist)
            - charge: Tensor of per-atom partial charges (shape: [N_atoms])
            - dipole: Tensor of per-atom dipole vectors (shape: [N_atoms, 3])
            - qpole: Tensor of per-atom quadrupole tensors or descriptors (shape: [N_atoms, ...], typically [N_atoms, 3, 3] or a flattened equivalent)
            - hlist: Tensor containing per-atom hidden representations across message-passing steps (shape: [n_steps, N_atoms, hidden_dim] or similar)
        """
        charge, dipole, qpole, hlist = self.model(
            data.x,
            data.edge_index,
            # data.edge_attr,
            R=data.R,
            molecule_ind=data.molecule_ind,
            total_charge=data.total_charge,
            natom_per_mol=data.natom_per_mol,
        )
        return charge, dipole, qpole, hlist