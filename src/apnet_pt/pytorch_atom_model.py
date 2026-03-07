import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DistanceLayer, FeedForwardLayer  

#################

target_dim = 1  # target property dimension
max_Z = 118  # largest atomic number

#################


def get_distances(RA, RB, e_source, e_target):
    """
    Compute pairwise displacement vectors and their Euclidean norms for specified edges.
    
    Parameters:
        RA (torch.Tensor): Source node coordinates with shape (..., 3) or (n_nodes, 3).
        RB (torch.Tensor): Target node coordinates with shape compatible with RA.
        e_source (Tensor or LongTensor): Index tensor selecting source positions from RA (shape (n_edges,)).
        e_target (Tensor or LongTensor): Index tensor selecting target positions from RB (shape (n_edges,)).
    
    Returns:
        tuple:
            dR (torch.Tensor): 1D tensor of Euclidean distances for each edge (shape (n_edges,)).
            dR_xyz (torch.Tensor): 2D tensor of displacement vectors RB_target - RA_source for each edge (shape (n_edges, 3)).
    """
    RA_source = RA[e_source]
    RB_target = RB[e_target]

    dR_xyz = RB_target - RA_source
    dR = torch.sqrt(F.relu(torch.sum(dR_xyz ** 2, dim=-1)))

    return dR, dR_xyz

def get_messages(h0, h, rbf, e_source, e_target):
    nedge = e_source.shape[0]

    h0_source = h0[e_source]
    h0_target = h0[e_target]
    h_source = h[e_source]
    h_target = h[e_target]

    h_all = torch.cat([h0_source, h0_target, h_source, h_target], dim=-1)
    h_all_dot = torch.einsum('ez,er->ezr', h_all, rbf)
    h_all_dot = h_all_dot.view(nedge, -1)

    return torch.cat([h_all, h_all_dot, rbf], dim=-1)

class PyTorchAtomModel(nn.Module):

    def __init__(self, n_message=3, n_rbf=8, n_neuron=128, n_embed=8, r_cut=5.0):
        super(PyTorchAtomModel, self).__init__()

        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut = r_cut

        self.distance_layer = DistanceLayer(n_rbf, r_cut)
        self.embed_layer = nn.Embedding(max_Z + 1, n_embed)
        self.guess_layer = nn.Embedding(max_Z + 1, target_dim)

        self.charge_update_layers = nn.ModuleList()
        self.dipole_update_layers = nn.ModuleList()
        self.qpole1_update_layers = nn.ModuleList()
        self.qpole2_update_layers = nn.ModuleList()

        self.charge_readout_layers = nn.ModuleList()
        self.dipole_readout_layers = nn.ModuleList()
        self.qpole_readout_layers = nn.ModuleList()

        layer_nodes_hidden = [n_neuron * 2, n_neuron, n_neuron // 2, n_embed]
        layer_nodes_readout = [n_neuron * 2, n_neuron, n_neuron // 2, target_dim]
        layer_activations = [nn.ReLU(), nn.ReLU(), nn.ReLU(), None]  # Assuming linear activation for the last layer

        for i in range(n_message):
            self.charge_update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations))
            self.dipole_update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations))
            self.qpole1_update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations))
            self.qpole2_update_layers.append(FeedForwardLayer(layer_nodes_hidden, layer_activations))

            self.charge_readout_layers.append(FeedForwardLayer(layer_nodes_readout, layer_activations))
            self.dipole_readout_layers.append(nn.Linear(layer_nodes_readout[-2], 1))
            self.qpole_readout_layers.append(nn.Linear(layer_nodes_readout[-2], 1))

    def forward(self, inputs):
        # Unpack the inputs (assuming inputs is a dictionary as in the TensorFlow code)
        Z = inputs['Z']
        R = inputs['R']
        e_source = inputs['e_source']
        e_target = inputs['e_target']
        molecule_ind = inputs['molecule_ind']
        total_charge = inputs['total_charge']

        natom = Z.size(0)
        # PyTorch doesn't have a direct equivalent of tf.math.segment_sum, so you might need to implement a custom function
        # For simplicity, let's assume natom_per_mol is precomputed or you have a workaround
        natom_per_mol = ...  # Implement your logic or workaround here

        # Example operations adapted to PyTorch
        dR, dR_xyz = get_distances(R, R, e_source, e_target)  # Make sure get_distances is adapted to PyTorch
        dr_unit = dR_xyz / dR.unsqueeze(1)

        rbf = self.distance_layer(dR)  # Ensure distance_layer is a PyTorch module

        h_list = [self.embed_layer(Z).view(natom, -1)]
        charge = self.guess_layer(Z)  # Ensure guess_layer is adapted to PyTorch
        dipole = torch.zeros(natom, 3, dtype=torch.float32)
        qpole = torch.zeros(natom, 3, 3, dtype=torch.float32)

        for i in range(self.n_message):
            m_ij = get_messages(h_list[0], h_list[-1], rbf, e_source, e_target)  # Adapt get_messages to PyTorch
            # Implement the logic for charge, dipole, and quadrupole updates using PyTorch operations
            # m_i = tf.math.unsorted_segment_sum(m_ij, e_source, natom)
            m_i = torch.math.fsum(m_ij, e_source, natom)  
            # TODO finish...

        # Enforce traceless quadrupole and charge conservation in PyTorch
        # This may involve custom implementations for operations not directly available in PyTorch

        return charge, dipole, qpole  # Adjust return values as needed

    def get_config(self):
        # This method is specific to TensorFlow's Keras. In PyTorch, you might manually implement configuration saving/loading.
        pass

if __name__ == "__main__":
    model = PyTorchAtomModel()
    print(model.r_cut)