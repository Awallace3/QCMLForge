import torch
import torch.nn as nn
import torch.nn.functional as F
from apnet_pt.util import scatter_sum_compile
from torch_geometric.nn import MessagePassing
import numpy as np
import warnings
from .. import multipole
from .. import model_io
from ..hf_pretrained import resolve_pretrained_path
import time
from ..atomic_datasets import (
    atomic_module_dataset,
    AtomicDataLoader,
    atomic_collate_update,
    qcel_mon_to_pyg_data,
    atomic_collate_update_no_target,
)

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import qcelemental as qcel
from pprint import pprint as pp

warnings.filterwarnings("ignore")

max_Z = 118  # largest atomic number

# file_dir = os.path.dirname(os.path.realpath(__file__))


def unsorted_segment_sum_2d(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. This function supports 2D data.

    Parameters:
    - data: Tensor of shape (N, F)
    - segment_ids: Tensor of shape (N,)
    - num_segments: int, Number of unique segments

    Returns:
    - Tensor of sum along segments of shape (num_segments, D, F) or (num_segments, F)
    """
    N, F = data.size()
    result = torch.zeros((num_segments, F), dtype=data.dtype, device=data.device)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, F)
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_sum_3d(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. This function supports 3D data.

    Parameters:
    - data: Tensor of shape (N, D, F)
    - segment_ids: Tensor of shape (N,)
    - num_segments: int, Number of unique segments

    Returns:
    - Tensor of sum along segments of shape (num_segments, D, F) or (num_segments, F)
    """
    N, D, F = data.size()
    result = torch.zeros((num_segments, D, F), dtype=data.dtype, device=data.device)
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, D, F)
    result.scatter_add_(0, segment_ids, data)
    return result


def make_quad(flat_quad):
    natom = flat_quad.size()[0]
    full_quad = torch.zeros(
        (natom, 3, 3), device=flat_quad.device, dtype=flat_quad.dtype
    )
    full_quad[:, 0, 0] = flat_quad[:, 0]  # xx
    full_quad[:, 0, 1] = flat_quad[:, 1]  # xy
    full_quad[:, 1, 0] = flat_quad[:, 1]  # xy
    full_quad[:, 0, 2] = flat_quad[:, 2]  # xz
    full_quad[:, 2, 0] = flat_quad[:, 2]  # xz
    full_quad[:, 1, 1] = flat_quad[:, 3]  # yy
    full_quad[:, 1, 2] = flat_quad[:, 4]  # yz
    full_quad[:, 2, 1] = flat_quad[:, 4]  # yz
    full_quad[:, 2, 2] = flat_quad[:, 5]  # zz

    trace = full_quad[:, 0, 0] + full_quad[:, 1, 1] + full_quad[:, 2, 2]

    full_quad[:, 0, 0] -= trace / 3.0
    full_quad[:, 1, 1] -= trace / 3.0
    full_quad[:, 2, 2] -= trace / 3.0

    return full_quad


# @torch.jit.script
def get_distances(RA, RB, e_source, e_target):
    RA_source = torch.index_select(RA, dim=0, index=e_source)
    # print(RA_source[:10])
    # print(np.load(f"tf_outputs/am/am__RA_source.npy")[:10])
    # assert np.allclose(RA_source, np.load(f"tf_outputs/am/am__RA_source.npy"))
    RB_target = torch.index_select(RB, dim=0, index=e_target)
    dR_xyz = RB_target - RA_source
    # print(np.array(dR_xyz[:10]))
    # print(np.load(f"tf_outputs/am/am__dR_xyz0.npy")[:10])
    # assert np.allclose(dR_xyz, np.load(f"tf_outputs/am/am__dR_xyz0.npy"), rtol=1e-4)

    # Compute distances with safe operation for square root
    dR = torch.sqrt(F.relu(torch.sum(dR_xyz**2, dim=-1)))
    return dR, dR_xyz


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

        # Apply conditional where inputs < 1
        env_val = torch.where(inputs < 1, env_val, torch.zeros_like(inputs))
        return env_val


class DistanceLayer(nn.Module):
    """
    Projects a distance 0 < r < r_cut into an orthogonal basis of Bessel functions in PyTorch.
    """

    def __init__(self, num_radial=8, r_cut=5.0, envelope_exponent=5):
        super(DistanceLayer, self).__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / r_cut
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        freq_init = torch.FloatTensor(
            np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
        )
        self.frequencies = nn.Parameter(freq_init, requires_grad=True)

    def forward(self, inputs):
        # scale to range [0, 1]
        d_scaled = inputs * self.inv_cutoff
        # # print(d_scaled[:10], np.load(f"tf_outputs/am/am__d_scaled.npy")[:10])
        # assert np.allclose(d_scaled.detach().numpy(), np.load(f"tf_outputs/am/am__d_scaled.npy"))
        d_scaled = d_scaled.unsqueeze(-1)
        d_cutoff = self.envelope(d_scaled)
        # # print(d_cutoff[:10], np.load(f"tf_outputs/am/am__d_cutoff.npy")[:10])
        # assert np.allclose(self.frequencies.detach().numpy(), np.load(f"tf_outputs/am/am__frequencies.npy"))
        # assert np.allclose(d_cutoff.detach().numpy(), np.load(f"tf_outputs/am/am__d_cutoff.npy"))
        return d_cutoff * torch.sin(self.frequencies * d_scaled)


class AtomMPNN(MessagePassing):
    def __init__(self, n_message=3, n_rbf=8, n_neuron=128, n_embed=8, r_cut=5.0):
        super().__init__(aggr="add")
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut = r_cut

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

        input_layer_size = n_embed * 4 * n_rbf + n_embed * 4 + n_rbf

        layer_nodes_hidden = [
            input_layer_size,
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
            "r_cut": self.r_cut,
        }

    def _make_layers(self, layer_nodes, activations):
        layers = []
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            if activations[i] is not None:
                layers.append(activations[i])
        return nn.Sequential(*layers)

    def get_messages(self, h0, h, rbf, e_source, e_target):
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

        # [edges,  n_embed * 4 * n_rbf + n_embed * 4 + n_rbf]
        m_ij = torch.cat([h_all, h_all_dot, rbf], dim=-1)
        return m_ij

    # @torch.jit.trace
    def forward(
        self,
        batch,
    ):
        # Extract variables from batch
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
            total_charge_pred = total_charge_pred.squeeze()
            total_charge_err = total_charge_pred - total_charge
            charge_err = torch.repeat_interleave(
                total_charge_err / natom_per_mol.float(), natom_per_mol
            ).unsqueeze(1)
            charge = charge - charge_err
            return charge, dipole, qpole, h_list

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
        idx_map = (
            torch.cumsum(keep_mask, dim=0) - 1
        )  # shape [N], each kept atom -> new index
        idx_map = idx_map.long()  # ensure integer
        e_source = idx_map[e_source]
        e_target = idx_map[e_target]

        R = R[keep_mask, :]
        natom_filtered = keep_mask.sum()

        #  [edges]
        dR, dR_xyz = get_distances(R, R, e_source, e_target)

        # [edges x 3]
        dr_unit = dR_xyz / dR.unsqueeze(1)
        rbf = self.distance_layer(dR)

        for i in range(self.n_message):
            #####################
            ### charge update ###
            #####################

            # [edges x message_embedding_dim]
            m_ij = self.get_messages(h_list[0], h_list[-1], rbf, e_source, e_target)

            # [atoms x message_embedding_dim]
            # m_i = unsorted_segment_sum_2d(m_ij, e_source, natom)
            # write unsorted_segment_sum_2d using scatter
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

        total_charge_pred = total_charge_pred.squeeze()
        total_charge_err = total_charge_pred - total_charge
        charge_err = torch.repeat_interleave(
            total_charge_err / natom_per_mol.float(), natom_per_mol
        ).unsqueeze(1)
        charge = charge - charge_err
        charge = charge.squeeze()
        # changed to dim=0 from dim=1 for usage in Param fitting # AMW 8/20/25
        # Breaks test_apnet2_train_qcel_molecules_in_memory_transfer test,
        # dimensions no longer correct... figure out another way to fix this. reverting back to dim=1 # AMW 9/17/25
        # print(len(h_list), h_list[0].size())
        h_list = torch.stack(h_list, dim=1)
        # print(h_list.size())
        return charge, dipole, qpole, h_list


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def isolate_atomic_property_predictions(batch, output):
    batch_size = batch.natom_per_mol.size(0)
    qA = output[0]
    muA = output[1]
    thA = output[2]
    hlistA = output[3]
    mol_charges = [[] for i in range(batch_size)]
    mol_dipoles = [[] for i in range(batch_size)]
    mol_qpoles = [[] for i in range(batch_size)]
    mol_hlist = [[] for i in range(batch_size)]
    i_offset = 0
    for n, i in enumerate(batch.natom_per_mol):
        mol_charges[n] = qA[i_offset : i_offset + i]
        mol_dipoles[n] = muA[i_offset : i_offset + i]
        mol_qpoles[n] = thA[i_offset : i_offset + i]
        mol_hlist[n] = hlistA[i_offset : i_offset + i]
        i_offset += i
    return mol_charges, mol_dipoles, mol_qpoles, mol_hlist


def isolate_atomic_property_predictions_q_mu_theta_hlist_hfvr_vw(batch, output):
    batch_size = batch.natom_per_mol.size(0)
    qA = output[0]
    muA = output[1]
    thA = output[2]
    hlistA = output[3]
    mol_charges = [[] for i in range(batch_size)]
    mol_dipoles = [[] for i in range(batch_size)]
    mol_qpoles = [[] for i in range(batch_size)]
    mol_hlist = [[] for i in range(batch_size)]
    i_offset = 0
    for n, i in enumerate(batch.natom_per_mol):
        mol_charges[n] = qA[i_offset : i_offset + i]
        mol_dipoles[n] = muA[i_offset : i_offset + i]
        mol_qpoles[n] = thA[i_offset : i_offset + i]
        mol_hlist[n] = hlistA[i_offset : i_offset + i]
        i_offset += i
    return mol_charges, mol_dipoles, mol_qpoles, mol_hlist


class AtomModel:
    def __init__(
        self,
        dataset=None,
        pre_trained_model_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_root="data",
        ds_max_size=None,
        ds_testing=False,
        ds_force_reprocess=False,
        ds_in_memory=True,
        model_save_path=None,
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

        if pre_trained_model_path:
            # print(f"Loading pre-trained AtomMPNN model from {pre_trained_model_path}")
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = AtomMPNN(
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
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = AtomMPNN(
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut=r_cut,
            )
        # self.model.to(device)
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
        Load a pretrained model from a checkpoint file.

        Supports both v1 (legacy) and v2 checkpoint formats.

        Parameters
        ----------
        model_path : str, optional
            Path to a checkpoint file
        model_id : int, optional
            ID of a bundled pretrained model (0-9)

        Returns
        -------
        self
            Returns self for method chaining
        """
        if model_id is not None:
            model_path = resolve_pretrained_path(f"am_ensemble/am_{model_id}.pt")
        elif model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided.")

        checkpoint = model_io.load_checkpoint(model_path)
        version = model_io.get_checkpoint_version(checkpoint)

        # For v2, optionally validate checkpoint type
        if version >= 2:
            model_io.validate_checkpoint(checkpoint, expected_type="AtomMPNN")

        # Load the state dict
        state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)

        # Handle compiled model prefix mismatch
        model_keys = list(self.model.state_dict().keys())
        state_keys = list(state_dict.keys())
        model_is_compiled = model_keys and "_orig_mod" in model_keys[0]
        state_is_compiled = state_keys and "_orig_mod" in state_keys[0]

        if model_is_compiled and not state_is_compiled:
            # Loading non-compiled state_dict into compiled model
            state_dict = model_io.add_prefix_to_state_dict(state_dict)
        elif not model_is_compiled and state_is_compiled:
            # Loading compiled state_dict into non-compiled model
            state_dict = model_io.strip_prefix_from_state_dict(state_dict)

        self.model.load_state_dict(state_dict)
        return self

    def _create_checkpoint(self, metadata: dict | None = None) -> dict:
        """
        Create a v2 checkpoint dictionary for this model.

        Parameters
        ----------
        metadata : dict, optional
            Additional metadata to include in the checkpoint

        Returns
        -------
        dict
            Checkpoint dictionary in v2 format
        """
        return model_io.create_checkpoint(
            model=self.model,
            config=model_io.unwrap_model(self.model).get_config(),
            model_type="AtomMPNN",
            metadata=metadata,
        )

    def save_model(self, path: str, metadata: dict | None = None) -> None:
        """
        Save the model to a checkpoint file in v2 format.

        Parameters
        ----------
        path : str
            Path to save the checkpoint
        metadata : dict, optional
            Additional metadata to include
        """
        checkpoint = self._create_checkpoint(metadata=metadata)
        model_io.save_checkpoint(checkpoint, path)

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

    def _qcel_example_input(self, mols, batch_size=1):
        mol_data = [qcel_mon_to_pyg_data(mol) for mol in mols]
        batches = []
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            batch_A = atomic_collate_update_no_target(batch_mol_data)
            batches.append(batch_A)
        return batches

    def example_input(self):
        mol = qcel.models.Molecule.from_data("""
0 1
8   -0.702196054   -0.056060256   0.009942262
1   -1.022193224   0.846775782   -0.011488714
1   0.257521062   0.042121496   0.005218999
units angstrom
        """)
        return self._qcel_example_input([mol], batch_size=1)

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
        charge_errors_t, dipole_errors_t, qpole_errors_t = [], [], []
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
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
        t1 = time.time()
        with torch.no_grad():
            _, charge_errors_t, dipole_errors_t, qpole_errors_t = (
                self.evaluate_model_collate_eval(
                    train_loader,  # loss_fn=criterion
                )
            )
            print(charge_errors_t.shape, dipole_errors_t.shape, qpole_errors_t.shape)
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

    def train_batches_single_proc(
        self, rank, dataloader, criterion, optimizer, rank_device
    ):
        self.model.train()
        total_charge_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_dipole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_qpole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        for batch in dataloader:
            batch = batch.to(rank_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            charge, dipole, qpole, _ = self.model(batch)

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
        self.model.eval()
        total_charge_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_dipole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_qpole_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        with torch.no_grad():
            for batch in dataloader:
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
        self.model.eval()
        total_charge_error = 0
        total_dipole_error = 0
        total_qpole_error = 0
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
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

        train_loader = AtomicDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            collate_fn=atomic_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=atomic_collate_update,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        test_loss = self.pretrain_statistics(train_loader, test_loader, criterion)

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
                        cpu_model = model_io.unwrap_model(self.model).to("cpu")
                        checkpoint = model_io.create_checkpoint(
                            model=cpu_model,
                            config=cpu_model.get_config(),
                            model_type="AtomMPNN",
                        )
                        model_io.save_checkpoint(checkpoint, self.model_save_path)
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
            collate_fn=atomic_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=atomic_collate_update,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        lowest_test_loss = torch.tensor(float("inf"))
        print(f"{rank=}")

        test_loss = self.pretrain_statistics(train_loader, test_loader, criterion)

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
                        cpu_model = model_io.unwrap_model(self.model).to("cpu")
                        checkpoint = model_io.create_checkpoint(
                            model=cpu_model,
                            config=cpu_model.get_config(),
                            model_type="AtomMPNN",
                        )
                        model_io.save_checkpoint(checkpoint, self.model_save_path)
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
        batch.to(self.device)
        self.model.to(self.device)
        qA, muA, thA, hlistA = self.model(batch)
        batch = batch.cpu()
        qA = qA.detach().detach().cpu()
        # print("predict_multipoles_batch")
        # print(qA)
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
        Predict per-atom multipole properties for a list of QCEL-formatted molecules by batching them through the model.

        Parameters:
            mols (Iterable): Iterable of molecules in QCEL/monomer format accepted by qcel_mon_to_pyg_data.
            batch_size (int): Number of molecules to process per model forward pass.

        Returns:
            List[Tuple]: A list, one entry per input molecule, of tuples (charges, dipoles, qpoles, hlists) where:
                - charges: Tensor of shape (n_atoms,) containing predicted atomic charges.
                - dipoles: Tensor of shape (n_atoms, 3) containing predicted atomic dipole vectors.
                - qpoles: Tensor of shape (n_atoms, 3, 3) containing predicted atomic quadrupole tensors.
                - hlists: Tensor or list containing per-atom hidden-state vectors returned by the model.
        """
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
        Predict multipole outputs for each dimer and its two monomer fragments.

        Parameters:
            mols (Sequence): Sequence of QCEngine/QCEL molecule objects representing dimers; each molecule must support `get_fragment`.
            batch_size (int): Number of molecules to process per inference batch.

        Returns:
            tuple: (dimer_output, monA_output, monB_output) where each element is the list of per-molecule prediction tuples produced by `predict_qcel_mols` (e.g., (charges, dipoles, qpoles, hlists) per molecule).
        """
        monA = [mol.get_fragment([0]) for mol in mols]
        monB = [mol.get_fragment([1]) for mol in mols]
        dimer_output = self.predict_qcel_mols(mols, batch_size=batch_size)
        monA_output = self.predict_qcel_mols(monA, batch_size=batch_size)
        monB_output = self.predict_qcel_mols(monB, batch_size=batch_size)
        return dimer_output, monA_output, monB_output

    def predict_elst_ind_dimer(self, mols, batch_size=2):
        """
        Compute per-dimer electrostatic energies, dimer electrostatic energies after model prediction, and their difference (induction) for a list of QCELEM dimer molecules.

        Parameters:
            mols (Sequence): Iterable of QCEL dimer molecules; each molecule must contain fragment indexing in `fragments`.
            batch_size (int): Number of molecules to process per prediction batch.

        Returns:
            E_elst (list of float): Electrostatic energies computed from monomer reference multipoles for each dimer.
            E_elst_dimer (list of float): Electrostatic energies computed from model-predicted multipoles on the full dimer for each dimer.
            E_induction (list of float): Difference `E_elst_dimer - E_elst` for each dimer, representing the induction-like contribution.
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
        Produce atomic multipole predictions and hidden states for a batched input.

        Parameters:
            data: Batched graph-like object containing at least the attributes
                `x` (atom types), `edge_index` (edge indices), `R` (positions),
                `molecule_ind` (molecule indices per atom), `total_charge` (per-batch total charge),
                and `natom_per_mol` (number of atoms per molecule).

        Returns:
            charge (Tensor): Per-atom scalar charges with shape (N,).
            dipole (Tensor): Per-atom dipole vectors with shape (N, 3).
            qpole (Tensor): Per-atom quadrupole tensors with shape (N, 3, 3), traceless.
            hlist (Tensor or list[Tensor]): Per-atom hidden representations produced by the model.
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
