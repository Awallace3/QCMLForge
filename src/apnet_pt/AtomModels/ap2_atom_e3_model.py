import torch
import torch.nn as nn
import time
import warnings

from e3nn import o3

from apnet_pt.util import scatter_sum_compile

from .. import model_io
from .. import multipole
from .ap2_atom_model import (
    AtomModel,
    AtomicDataLoader,
    DistanceLayer,
    get_distances,
    max_Z,
    atomic_collate_update,
    unsorted_segment_sum_3d,
)
from torch.nn.parallel import DistributedDataParallel as DDP


def _sh_dim_from_lmax(lmax: int) -> int:
    return sum(2 * l + 1 for l in range(1, lmax + 1))


def _fallback_sh_components(dr_unit: torch.Tensor, lmax: int) -> torch.Tensor:
    x = dr_unit[:, 0:1]
    y = dr_unit[:, 1:2]
    z = dr_unit[:, 2:3]

    parts = [dr_unit]

    if lmax >= 2:
        l2 = torch.cat(
            [
                x * y,
                y * z,
                z * x,
                x * x - y * y,
                0.5 * (3.0 * z * z - 1.0),
            ],
            dim=-1,
        )
        parts.append(l2)

    if lmax >= 3:
        l3 = torch.cat(
            [
                x * y * z,
                x * (x * x - 3.0 * y * y),
                y * (3.0 * x * x - y * y),
                z * (x * x - y * y),
                x * (5.0 * z * z - 1.0),
                y * (5.0 * z * z - 1.0),
                z * (5.0 * z * z - 3.0),
            ],
            dim=-1,
        )
        parts.append(l3)

    return torch.cat(parts, dim=-1)


class AtomE3MPNN(nn.Module):
    """Configurable e3nn-based atom model for multipole prediction."""

    def __init__(
        self,
        n_message: int = 3,
        n_rbf: int = 8,
        n_neuron: int = 128,
        n_embed: int = 8,
        r_cut: float = 5.0,
        e3_lmax: int = 1,
        e3_contraction: str = "einsum",
        e3_dipole_mode: str = "l1",
        e3_qpole_mode: str = "legacy",
        e3_message_mode: str = "none",
        e3_normalization: str = "component",
    ):
        super().__init__()
        if e3_lmax < 1:
            raise ValueError("e3_lmax must be >= 1")
        valid_contraction = {"einsum", "tensor_product", "fully_connected_tp"}
        if e3_contraction not in valid_contraction:
            raise ValueError(
                f"Invalid e3_contraction={e3_contraction}. "
                f"Expected one of {sorted(valid_contraction)}"
            )
        valid_dipole_mode = {"l1", "multi_l"}
        if e3_dipole_mode not in valid_dipole_mode:
            raise ValueError(
                f"Invalid e3_dipole_mode={e3_dipole_mode}. "
                f"Expected one of {sorted(valid_dipole_mode)}"
            )
        valid_qpole_mode = {"legacy", "l2"}
        if e3_qpole_mode not in valid_qpole_mode:
            raise ValueError(
                f"Invalid e3_qpole_mode={e3_qpole_mode}. "
                f"Expected one of {sorted(valid_qpole_mode)}"
            )
        valid_message_mode = {"none", "concat_sh"}
        if e3_message_mode not in valid_message_mode:
            raise ValueError(
                f"Invalid e3_message_mode={e3_message_mode}. "
                f"Expected one of {sorted(valid_message_mode)}"
            )

        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut = r_cut

        self.e3_lmax = e3_lmax
        self.e3_contraction = e3_contraction
        self.e3_dipole_mode = e3_dipole_mode
        self.e3_qpole_mode = e3_qpole_mode
        self.e3_message_mode = e3_message_mode
        self.e3_normalization = e3_normalization

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

        self.sh_dim = _sh_dim_from_lmax(self.e3_lmax)
        message_extra = 0
        if self.e3_message_mode == "concat_sh":
            message_extra = self.sh_dim + self.sh_dim * self.n_rbf

        input_layer_size = n_embed * 4 * n_rbf + n_embed * 4 + n_rbf + message_extra

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

        self.dipole_feature_project = nn.Linear(n_embed, 3)
        self.dipole_tp = o3.ElementwiseTensorProduct("1x0e", "1x1o")
        self.dipole_fc_tp = o3.FullyConnectedTensorProduct("1x0e", "1x1o", "1x1o")
        self.multi_l_to_l1 = nn.Linear(self.sh_dim, 3)

        self.qpole_l2_readout = nn.Linear(n_embed, 1)
        self.qpole_l2_to_cart = nn.Linear(5, 9, bias=False)

    def get_config(self) -> dict:
        return {
            "n_message": self.n_message,
            "n_rbf": self.n_rbf,
            "n_neuron": self.n_neuron,
            "n_embed": self.n_embed,
            "r_cut": self.r_cut,
            "e3_lmax": self.e3_lmax,
            "e3_contraction": self.e3_contraction,
            "e3_dipole_mode": self.e3_dipole_mode,
            "e3_qpole_mode": self.e3_qpole_mode,
            "e3_message_mode": self.e3_message_mode,
            "e3_normalization": self.e3_normalization,
        }

    def _make_layers(self, layer_nodes, activations):
        layers = []
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            if activations[i] is not None:
                layers.append(activations[i])
        return nn.Sequential(*layers)

    def get_messages(self, h0, h, rbf, e_source, e_target, sh_features=None):
        nedge = e_source.size(0)

        h0_source = h0.index_select(0, e_source)
        h0_target = h0.index_select(0, e_target)
        h_source = h.index_select(0, e_source)
        h_target = h.index_select(0, e_target)

        h_all = torch.cat([h0_source, h0_target, h_source, h_target], dim=-1)
        h_all_dot = torch.einsum("ez,er->ezr", h_all, rbf)
        h_all_dot = h_all_dot.view(nedge, -1)

        base = [h_all, h_all_dot, rbf]
        if sh_features is not None:
            base.append(sh_features)
        return torch.cat(base, dim=-1)

    def _get_sh_features(self, dr_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.e3_lmax == 1:
            return dr_unit, dr_unit

        l_values = list(range(1, self.e3_lmax + 1))
        try:
            sh_all = o3.spherical_harmonics(
                l_values,
                dr_unit,
                normalize=True,
                normalization=self.e3_normalization,
            )
        except RuntimeError:
            sh_all = _fallback_sh_components(dr_unit, self.e3_lmax)
        sh_l1 = sh_all[:, :3]
        return sh_all, sh_l1

    def _contract_dipole(self, dipole_amp, dipole_feat, sh_used):
        if self.e3_contraction == "einsum":
            return dipole_amp * sh_used

        if self.e3_contraction == "tensor_product":
            return self.dipole_tp(dipole_amp, sh_used)

        feat_scalar = dipole_feat.mean(dim=-1, keepdim=True)
        return self.dipole_fc_tp(feat_scalar, sh_used)

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

        filtered_charge = charge
        filtered_dipole = dipole
        filtered_qpole = qpole
        h_list = [h0]

        e_source = edge_index[0]
        e_target = edge_index[1]

        if e_source.numel() == 0:
            h_stack = torch.stack([h0 for _ in range(self.n_message + 1)], dim=1)
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
            return charge, dipole, qpole, h_stack

        dR, dR_xyz = get_distances(R, R, e_source, e_target)
        dR_safe = torch.clamp(dR, min=1.0e-8)
        dr_unit = dR_xyz / dR_safe.unsqueeze(1)
        dr_unit = torch.nan_to_num(dr_unit, nan=0.0, posinf=0.0, neginf=0.0)
        rbf = self.distance_layer(dR)

        sh_all, sh_l1 = self._get_sh_features(dr_unit)
        if self.e3_dipole_mode == "multi_l":
            sh_used = self.multi_l_to_l1(sh_all)
        else:
            sh_used = sh_l1

        sh_message_features = None
        if self.e3_message_mode == "concat_sh":
            sh_dot_rbf = torch.einsum("el,er->elr", sh_all, rbf).reshape(
                rbf.size(0), -1
            )
            sh_message_features = torch.cat([sh_all, sh_dot_rbf], dim=-1)

        sh_l2 = None
        if self.e3_qpole_mode == "l2":
            if self.e3_lmax < 2:
                raise ValueError("e3_qpole_mode='l2' requires e3_lmax >= 2")
            sh_l2 = sh_all[:, 3:8]

        for i in range(self.n_message):
            m_ij = self.get_messages(
                h_list[0], h_list[-1], rbf, e_source, e_target, sh_message_features
            )
            m_i = scatter_sum_compile(m_ij, e_source, natom, reduce="sum")

            h_next = self.charge_update_layers[i](m_i)
            h_list.append(h_next)
            filtered_charge = filtered_charge + self.charge_readout_layers[i](h_list[i + 1])

            m_ij_dipole = self.dipole_update_layers[i](m_ij)
            dipole_amp = self.dipole_edge_readout_layers[i](m_ij_dipole)
            dipole_feat = self.dipole_feature_project(m_ij_dipole)
            edge_dipole = self._contract_dipole(dipole_amp, dipole_feat, sh_used)
            filtered_dipole = filtered_dipole + scatter_sum_compile(
                edge_dipole, e_source, natom, reduce="sum"
            )

            if self.e3_qpole_mode == "l2" and sh_l2 is not None:
                q_amp = self.qpole_l2_readout(m_ij_dipole)
                edge_q_l2 = q_amp * sh_l2
                edge_q_cart = self.qpole_l2_to_cart(edge_q_l2).view(-1, 3, 3)
                edge_q_cart = 0.5 * (edge_q_cart + edge_q_cart.transpose(1, 2))
                d_qpole = scatter_sum_compile(
                    edge_q_cart, e_source, natom, reduce="sum"
                )
                filtered_qpole = filtered_qpole + d_qpole
            else:
                m_ij_qpole1 = self.qpole1_update_layers[i](m_ij)
                m_ij_qpole1 = torch.einsum("ex,em->exm", dr_unit, m_ij_qpole1)
                m_i_qpole1 = unsorted_segment_sum_3d(m_ij_qpole1, e_source, natom)

                m_ij_qpole2 = self.qpole2_update_layers[i](m_ij)
                m_ij_qpole2 = torch.einsum("ex,em->exm", dr_unit, m_ij_qpole2)
                m_i_qpole2 = unsorted_segment_sum_3d(m_ij_qpole2, e_source, natom)

                d_qpole = torch.einsum("axf,ayf->axyf", m_i_qpole1, m_i_qpole2)
                d_qpole = d_qpole + d_qpole.permute(0, 2, 1, 3)
                d_qpole = self.qpole_readout_layers[i](d_qpole).view(natom, 3, 3)
                filtered_qpole = filtered_qpole + d_qpole

        filtered_qpole = multipole.ensure_traceless_qpole(filtered_qpole)
        charge = filtered_charge
        dipole = filtered_dipole
        qpole = filtered_qpole

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
        e3_lmax=1,
        e3_contraction="einsum",
        e3_dipole_mode="l1",
        e3_qpole_mode="legacy",
        e3_message_mode="none",
        e3_normalization="component",
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

        self.e3_lmax = e3_lmax
        self.e3_contraction = e3_contraction
        self.e3_dipole_mode = e3_dipole_mode
        self.e3_qpole_mode = e3_qpole_mode
        self.e3_message_mode = e3_message_mode
        self.e3_normalization = e3_normalization

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
                e3_lmax=config.get("e3_lmax", e3_lmax),
                e3_contraction=config.get("e3_contraction", e3_contraction),
                e3_dipole_mode=config.get("e3_dipole_mode", e3_dipole_mode),
                e3_qpole_mode=config.get("e3_qpole_mode", e3_qpole_mode),
                e3_message_mode=config.get("e3_message_mode", e3_message_mode),
                e3_normalization=config.get("e3_normalization", e3_normalization),
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
                e3_lmax=e3_lmax,
                e3_contraction=e3_contraction,
                e3_dipole_mode=e3_dipole_mode,
                e3_qpole_mode=e3_qpole_mode,
                e3_message_mode=e3_message_mode,
                e3_normalization=e3_normalization,
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

    def compile_model(self):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        try:
            self.model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=False,
                mode="reduce-overhead",
            )
        except Exception as exc:
            warnings.warn(
                f"torch.compile failed for AtomE3Model; using eager mode. {exc}",
                RuntimeWarning,
            )
        return

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
        if self.device.type == "cpu":
            rank_device = "cpu"
        else:
            rank_device = rank
        if world_size > 1:
            self.setup(rank, world_size)

        self.model.to(rank_device)
        if world_size > 1:
            if rank_device == "cpu":
                self.model = DDP(self.model, find_unused_parameters=True)
            else:
                self.model = DDP(
                    self.model,
                    device_ids=[rank],
                    output_device=rank_device,
                    find_unused_parameters=True,
                )

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
                        cpu_model = model_io.unwrap_model(self.model).to("cpu")
                        checkpoint = model_io.create_checkpoint(
                            model=cpu_model,
                            config=cpu_model.get_config(),
                            model_type="AtomE3MPNN",
                        )
                        model_io.save_checkpoint(checkpoint, self.model_save_path)
                        self.model.to(self.device)
                else:
                    test_lowered = " "

                dt = time.time() - t1
                print(
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {charge_MAE_t:>7.4f}/{charge_MAE_v:<7.4f} {dipole_MAE_t:>7.4f}/{dipole_MAE_v:<7.4f} {qpole_MAE_t:>7.4f}/{qpole_MAE_v:<7.4f} {test_lowered}",
                    flush=True,
                )
        if world_size > 1:
            self.cleanup()
        return

    def _create_checkpoint(self, metadata: dict | None = None) -> dict:
        config = {
            "n_message": self.model.n_message,
            "n_rbf": self.model.n_rbf,
            "n_neuron": self.model.n_neuron,
            "n_embed": self.model.n_embed,
            "r_cut": self.model.r_cut,
            "e3_lmax": self.model.e3_lmax,
            "e3_contraction": self.model.e3_contraction,
            "e3_dipole_mode": self.model.e3_dipole_mode,
            "e3_qpole_mode": self.model.e3_qpole_mode,
            "e3_message_mode": self.model.e3_message_mode,
            "e3_normalization": self.model.e3_normalization,
        }
        return model_io.create_checkpoint(
            model=self.model,
            config=config,
            model_type="AtomE3MPNN",
            metadata=metadata,
        )
