import torch
import torch.nn as nn

from e3nn import o3

from apnet_pt.util import scatter_sum_compile

from .. import model_io
from .. import multipole
from .ap2_atom_model import (
    AtomModel,
    DistanceLayer,
    get_distances,
    max_Z,
    unsorted_segment_sum_3d,
)


class AtomE3MPNN(nn.Module):
    """E(3)-aware atom model using e3nn spherical harmonics."""

    def __init__(self, n_message=3, n_rbf=8, n_neuron=128, n_embed=8, r_cut=5.0):
        super().__init__()
        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut = r_cut

        self.distance_layer = DistanceLayer(n_rbf, r_cut)
        self.embed_layer = nn.Embedding(max_Z + 1, n_embed)
        self.guess_layer = nn.Embedding(max_Z + 1, 1)

        self.charge_update_layers = nn.ModuleList()
        self.dipole_update_layers = nn.ModuleList()
        self.qpole1_update_layers = nn.ModuleList()
        self.qpole2_update_layers = nn.ModuleList()

        self.charge_readout_layers = nn.ModuleList()
        self.dipole_edge_readout_layers = nn.ModuleList()
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
        layer_activations = [nn.ReLU(), nn.ReLU(), nn.ReLU(), None]

        for _ in range(n_message):
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
            self.dipole_edge_readout_layers.append(nn.Linear(n_embed, 1))
            self.qpole_readout_layers.append(nn.Linear(n_embed, 1))

    def get_config(self) -> dict:
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

        h_all = torch.cat([h0_source, h0_target, h_source, h_target], dim=-1)
        h_all_dot = torch.einsum("ez,er->ezr", h_all, rbf)
        h_all_dot = h_all_dot.view(nedge, -1)

        return torch.cat([h_all, h_all_dot, rbf], dim=-1)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        R = batch.R
        molecule_ind = batch.molecule_ind
        total_charge = batch.total_charge
        natom_per_mol = batch.natom_per_mol

        Z = x
        natom = Z.size(0)
        h0 = self.embed_layer(Z)

        charge = self.guess_layer(Z)
        dipole = torch.zeros(natom, 3, dtype=torch.float32, device=Z.device)
        qpole = torch.zeros(natom, 3, 3, dtype=torch.float32, device=Z.device)

        if edge_index.size(1) == 0:
            h_list = torch.stack([h0 for _ in range(self.n_message + 1)], dim=1)
            molecule_ind.requires_grad_(False)
            molecule_ind = molecule_ind.long()
            num_mols = int(molecule_ind.max().item()) + 1 if molecule_ind.numel() > 0 else 1
            total_charge_pred = scatter_sum_compile(
                charge, molecule_ind, num_mols, reduce="sum"
            ).squeeze()
            total_charge_err = total_charge_pred - total_charge
            charge_err = torch.repeat_interleave(
                total_charge_err / natom_per_mol.float(), natom_per_mol
            ).unsqueeze(1)
            charge = charge - charge_err
            return charge, dipole, qpole, h_list

        keep_mask = torch.zeros(natom, dtype=torch.bool, device=Z.device)
        keep_mask.scatter_(0, edge_index[0], True)
        keep_mask.scatter_(0, edge_index[1], True)

        filtered_charge = charge[keep_mask]
        filtered_dipole = torch.zeros(
            int(keep_mask.sum().item()), 3, dtype=torch.float32, device=Z.device
        )
        filtered_qpole = torch.zeros(
            int(keep_mask.sum().item()), 3, 3, dtype=torch.float32, device=Z.device
        )

        h_list = [h0[keep_mask]]

        e_source = edge_index[0]
        e_target = edge_index[1]
        edge_keep = keep_mask[e_source] & keep_mask[e_target]
        e_source = e_source[edge_keep]
        e_target = e_target[edge_keep]
        idx_map = (torch.cumsum(keep_mask, dim=0) - 1).long()
        e_source = idx_map[e_source]
        e_target = idx_map[e_target]

        R = R[keep_mask, :]
        natom_filtered = int(keep_mask.sum().item())

        dR, dR_xyz = get_distances(R, R, e_source, e_target)
        dr_unit = dR_xyz / dR.unsqueeze(1)
        rbf = self.distance_layer(dR)

        y_l1 = o3.spherical_harmonics(
            1, dr_unit, normalize=True, normalization="component"
        )

        for i in range(self.n_message):
            m_ij = self.get_messages(h_list[0], h_list[-1], rbf, e_source, e_target)
            m_i = scatter_sum_compile(m_ij, e_source, natom_filtered, reduce="sum")

            h_next = self.charge_update_layers[i](m_i)
            h_list.append(h_next)
            filtered_charge = filtered_charge + self.charge_readout_layers[i](h_list[i + 1])

            m_ij_dipole = self.dipole_update_layers[i](m_ij)
            dipole_amp = self.dipole_edge_readout_layers[i](m_ij_dipole)
            edge_dipole = dipole_amp * y_l1
            filtered_dipole = filtered_dipole + scatter_sum_compile(
                edge_dipole, e_source, natom_filtered, reduce="sum"
            )

            m_ij_qpole1 = self.qpole1_update_layers[i](m_ij)
            m_ij_qpole1 = torch.einsum("ex,em->exm", dr_unit, m_ij_qpole1)
            m_i_qpole1 = unsorted_segment_sum_3d(m_ij_qpole1, e_source, natom_filtered)

            m_ij_qpole2 = self.qpole2_update_layers[i](m_ij)
            m_ij_qpole2 = torch.einsum("ex,em->exm", dr_unit, m_ij_qpole2)
            m_i_qpole2 = unsorted_segment_sum_3d(m_ij_qpole2, e_source, natom_filtered)

            d_qpole = torch.einsum("axf,ayf->axyf", m_i_qpole1, m_i_qpole2)
            d_qpole = d_qpole + d_qpole.permute(0, 2, 1, 3)
            d_qpole = self.qpole_readout_layers[i](d_qpole).view(natom_filtered, 3, 3)
            filtered_qpole = filtered_qpole + d_qpole

        filtered_qpole = multipole.ensure_traceless_qpole(filtered_qpole)

        charge[keep_mask] = filtered_charge
        dipole[keep_mask] = filtered_dipole
        qpole[keep_mask] = filtered_qpole

        molecule_ind.requires_grad_(False)
        molecule_ind = molecule_ind.long()
        num_mols = int(molecule_ind.max().item()) + 1 if molecule_ind.numel() > 0 else 1
        total_charge_pred = scatter_sum_compile(
            charge, molecule_ind, num_mols, reduce="sum"
        ).squeeze()

        total_charge_err = total_charge_pred - total_charge
        charge_err = torch.repeat_interleave(
            total_charge_err / natom_per_mol.float(), natom_per_mol
        ).unsqueeze(1)
        charge = (charge - charge_err).squeeze()

        h_list = torch.stack(h_list, dim=1)
        return charge, dipole, qpole, h_list


class AtomE3Model(AtomModel):
    """Atom model harness for AtomE3MPNN with AtomModel-compatible API."""

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
        super().__init__(
            dataset=dataset,
            pre_trained_model_path=None,
            n_message=n_message,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            use_GPU=use_GPU,
            ignore_database_null=ignore_database_null,
            ds_spec_type=ds_spec_type,
            ds_root=ds_root,
            ds_max_size=ds_max_size,
            ds_testing=ds_testing,
            ds_force_reprocess=ds_force_reprocess,
            ds_in_memory=ds_in_memory,
            model_save_path=model_save_path,
        )

        if pre_trained_model_path:
            checkpoint = model_io.load_checkpoint(pre_trained_model_path)
            version = model_io.get_checkpoint_version(checkpoint)
            config = checkpoint.get("config", {})

            if version >= 2:
                model_io.validate_checkpoint(checkpoint, expected_type="AtomE3MPNN")

            self.model = AtomE3MPNN(
                n_message=config.get("n_message", n_message),
                n_rbf=config.get("n_rbf", n_rbf),
                n_neuron=config.get("n_neuron", n_neuron),
                n_embed=config.get("n_embed", n_embed),
                r_cut=config.get("r_cut", r_cut),
            )
            state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
            self.model.load_state_dict(state_dict)
        else:
            self.model = AtomE3MPNN(
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut=r_cut,
            )

    def set_pretrained_model(self, model_path=None, model_id=None):
        if model_id is not None:
            raise ValueError("AtomE3Model does not provide bundled model_id checkpoints.")
        if model_path is None:
            raise ValueError("model_path must be provided for AtomE3Model.")

        checkpoint = model_io.load_checkpoint(model_path)
        version = model_io.get_checkpoint_version(checkpoint)

        if version >= 2:
            model_io.validate_checkpoint(checkpoint, expected_type="AtomE3MPNN")

        state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
        self.model.load_state_dict(state_dict)
        return self

    def _create_checkpoint(self, metadata: dict | None = None) -> dict:
        config = {
            "n_message": self.model.n_message,
            "n_rbf": self.model.n_rbf,
            "n_neuron": self.model.n_neuron,
            "n_embed": self.model.n_embed,
            "r_cut": self.model.r_cut,
        }
        return model_io.create_checkpoint(
            model=self.model,
            config=config,
            model_type="AtomE3MPNN",
            metadata=metadata,
        )
