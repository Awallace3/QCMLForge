import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from apnet_pt.util import scatter_sum_compile
import numpy as np
import time
from .ap2_atom_model import (
    isolate_atomic_property_predictions,
    qcel_mon_to_pyg_data,
    unwrap_model,
    DistanceLayer,
    get_distances,
)
from apnet_pt.atomic_datasets import (
    AtomicDataLoader,
    atomic_collate_update_no_target,
    atomic_hfvr_vw_collate_update,
    atomic_hirshfeld_valencewdith_only_module_dataset,
)
import os
import torch.distributed as dist
import torch.multiprocessing as mp


max_Z = 118


class NoisyConstantEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, mean=3.0, std=0.01):
        super().__init__(num_embeddings, embedding_dim)
        with torch.no_grad():
            if isinstance(mean, (list, tuple)):
                # If mean is a list, use it directly (assuming it's the right shape)
                mean_tensor = torch.tensor(
                    mean, dtype=self.weight.dtype, device=self.weight.device
                )
                if len(mean_tensor) == 1:
                    mean_tensor = mean_tensor.expand_as(self.weight)
                elif len(mean_tensor) == self.weight.shape[0]:
                    mean_tensor = mean_tensor.unsqueeze(-1).expand_as(self.weight)
                else:
                    raise ValueError(
                        f"mean list length {len(mean_tensor)} doesn't match num_embeddings {num_embeddings}"
                    )
                self.weight.copy_(mean_tensor + std * torch.randn_like(self.weight))
            else:
                # Scalar case
                self.weight.copy_(mean + std * torch.randn_like(self.weight))


class AtomTypeParamMPNN(nn.Module):
    def __init__(
        self,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        param_start_mean=1.8,
        param_start_std=0.01,
        n_params=1,
    ):
        super().__init__()
        self.n_message = n_message
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.n_rbf = n_rbf
        self.r_cut = r_cut
        # Convert to lists if scalars
        if not isinstance(param_start_mean, (list, tuple)):
            param_start_mean = [param_start_mean] * n_params
        if not isinstance(param_start_std, (list, tuple)):
            param_start_std = [param_start_std] * n_params
        # Ensure they are the right length
        if len(param_start_mean) != n_params:
            raise ValueError(
                f"param_start_mean length {len(param_start_mean)} doesn't match n_params {n_params}"
            )
        if len(param_start_std) != n_params:
            raise ValueError(
                f"param_start_std length {len(param_start_std)} doesn't match n_params {n_params}"
            )

        # embed interatomic distances into large orthogonal basis
        self.distance_layer = DistanceLayer(n_rbf, r_cut)

        self.param_start_mean = param_start_mean
        self.param_start_std = param_start_std
        self.n_params = n_params
        self.embed_layer = nn.ModuleList(
            [nn.Embedding(max_Z + 1, n_embed) for p in range(n_params)]
        )
        self.guess_layer = nn.ModuleList(
            [
                NoisyConstantEmbedding(
                    max_Z + 1,
                    1,
                    mean=self.param_start_mean[p],
                    std=self.param_start_std[p],
                )
                for p in range(n_params)
            ]
        )
        # self.set_weights_excluding_guess(0.01)

        # readout layers for predicting multipoles from hidden states
        self.param_update_layers = nn.ModuleList(
            [nn.ModuleList() for _ in range(n_params)]
        )
        self.param_readout_layers = nn.ModuleList(
            [nn.ModuleList() for _ in range(n_params)]
        )
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
        ]
        input_layer_size = n_embed * 4 * n_rbf + n_embed * 4 + n_rbf
        layer_nodes_hidden = [
            input_layer_size,
            n_neuron * 2,
            n_neuron,
            n_neuron // 2,
            n_embed,
        ]
        for p in range(n_params):
            for i in range(n_message + 1):
                self.param_readout_layers[p].append(
                    self._make_layers(layer_nodes_readout, layer_activations)
                )
                self.param_update_layers[p].append(
                    self._make_layers(layer_nodes_hidden, layer_activations)
                )

    def set_weights_excluding_guess(self, value=0.01):
        """Sets all weights and biases in the model to a specific value."""
        with torch.no_grad():
            for name, param in self.state_dict().items():
                if "guess_layer" not in name:
                    param.fill_(value)

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

    def _make_layers(self, layer_nodes, activations):
        layers = []
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            # layers[-1].weight.data.normal_(1.0, 0.1)
            if activations[i] is not None:
                layers.append(activations[i])
        return nn.Sequential(*layers)

    def forward(
        self,
        batch,
    ):
        """
        Use each h_list to predict a correction to the initial guess, might be
        overkill for some properties...
        """
        x = batch.x
        edge_index = batch.edge_index
        molecule_ind = batch.molecule_ind
        R = batch.R
        Z = x
        K_list = [self.guess_layer[p](Z) for p in range(self.n_params)]
        K = torch.cat(K_list, dim=-1)  # shape (n_atoms, n_params)
        # print(f"{K = }")
        # Create keep_mask directly from edge_index without using torch.isin
        # This is more compile-friendly than torch.isin with unbacked symbolic shapes
        natom = len(molecule_ind)
        keep_mask = torch.zeros(natom, dtype=torch.bool, device=molecule_ind.device)
        if edge_index.size(1) > 0:
            # Mark all atoms that appear in edge_index as True
            keep_mask.scatter_(0, edge_index[0], True)
            keep_mask.scatter_(0, edge_index[1], True)
        if not keep_mask.any():
            return K.squeeze(-1) if self.n_params == 1 else K
        K_filtered = K[keep_mask]
        e_source = edge_index[0]
        e_target = edge_index[1]
        edge_keep = keep_mask[e_source] & keep_mask[e_target]
        e_source = e_source[edge_keep]
        e_target = e_target[edge_keep]
        idx_map = (torch.cumsum(keep_mask, dim=0) - 1).long()
        e_source = idx_map[e_source]
        e_target = idx_map[e_target]

        R = R[keep_mask, :]
        natom_filtered = keep_mask.sum()

        dR, dR_xyz = get_distances(R, R, e_source, e_target)
        rbf = self.distance_layer(dR)

        hlists = []
        for p in range(self.n_params):
            h_list_0 = [self.embed_layer[p](Z)]
            h_list = [h_list_0[0][keep_mask]]
            for i in range(self.n_message):
                # [edges x message_embedding_dim]
                m_ij = self.get_messages(h_list[0], h_list[-1], rbf, e_source, e_target)
                m_i = scatter_sum_compile(
                    m_ij, e_source, int(natom_filtered), reduce="sum"
                )
                # [atomx x hidden_dim]
                h_next = self.param_update_layers[p][i](m_i)
                h_list.append(h_next)
                param_update = self.param_readout_layers[p][i](h_list[i + 1])
                K_filtered[:, p] += param_update.squeeze(-1)
            hlists.append(torch.stack(h_list, dim=1))
        h_list = torch.stack(hlists, dim=2)
        K[keep_mask] = K_filtered
        return K.squeeze(-1) if self.n_params == 1 else K


### Atom Type Model Wrapper ####


class AtomTypeParamModel:
    def __init__(
        self,
        dataset=None,
        pre_trained_model_path=None,
        atom_model_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        param_start_mean=1.7,
        param_start_std=0.01,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_root="data_dir",
        ds_max_size=None,
        ds_random_seed=42,
        ds_batch_size=16,
        ds_testing=False,
        ds_force_reprocess=False,
        ds_in_memory=True,
        model_save_path=None,
        monomer_eval_type="hirshfeld_volume_ratio__valence_width",
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

        self.n_params = 1
        if monomer_eval_type in ["hirshfeld_volume_ratio__valence_width"]:
            self.n_params = 2

        if pre_trained_model_path:
            print(f"Loading pre-trained MTP-MTP model from {pre_trained_model_path}")
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = AtomTypeParamMPNN(
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
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = AtomTypeParamMPNN(
                n_message=n_message,
                n_neuron=n_neuron,
                n_embed=n_embed,
                param_start_mean=param_start_mean,
                param_start_std=param_start_std,
                n_params=self.n_params,
                r_cut=r_cut,
            )
        self.r_cut = r_cut
        self.n_params = self.n_params
        self.monomer_eval_type = monomer_eval_type
        self.device = device
        self.dataset = dataset
        mp.set_sharing_strategy("file_system")
        if not ignore_database_null and self.dataset is None:
            self.dataset = atomic_hirshfeld_valencewdith_only_module_dataset(
                root=ds_root,
                testing=ds_testing,
                spec_type=ds_spec_type,
                max_size=ds_max_size,
                force_reprocess=ds_force_reprocess,
                in_memory=ds_in_memory,
                batch_size=ds_batch_size,
            )
        # print(f"{self.dataset = }")
        self.rank = None
        self.world_size = None
        self.model_save_path = model_save_path
        self.train_shuffle = True
        # torch.jit.enable_onednn_fusion(True)
        return

    def set_pretrained_model(self, model_path):
        checkpoint = torch.load(model_path)
        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        return

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
        hfvr_errors_t, vw_errors_t = (
            [],
            [],
        )
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch_loss = 0.0
                params = self.model(batch)
                hirshfeld_volume_ratios = params[:, 0]
                valence_widths = params[:, 1]

                # Errors
                hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
                vw_error = valence_widths - batch.valence_widths
                if loss_fn is None:
                    # perform mean squared error
                    hfvr_loss = torch.mean(torch.square(hfvr_error))
                    vw_loss = torch.mean(torch.square(vw_error))
                else:
                    # perform custom loss function, or pytorch criterion loss_fn
                    hfvr_loss = torch.mean(
                        loss_fn(hirshfeld_volume_ratios, batch.volume_ratios)
                    )
                    vw_loss = torch.mean(loss_fn(valence_widths, batch.valence_widths))

                batch_loss = hfvr_loss + vw_loss
                total_loss += batch_loss.detach()

            hfvr_errors_t.extend(hfvr_error.detach())
            vw_errors_t.extend(vw_error.detach())
        hfvr_errors_t = torch.cat(hfvr_errors_t)
        vw_errors_t = torch.cat(vw_errors_t)
        return (
            total_loss,
            hfvr_errors_t,
            vw_errors_t,
        )

    def pretrain_statistics(self, train_loader, test_loader, criterion):
        t1 = time.time()
        with torch.no_grad():
            (
                _,
                hfvr_errors_t,
                vw_errors_t,
            ) = self.evaluate_model_collate_eval(
                train_loader,  # loss_fn=criterion
            )
            hfvr_MAE_t = np.mean(np.abs(hfvr_errors_t))
            vw_MAE_t = np.mean(np.abs(vw_errors_t))

            (
                hfvr_errors_t,
                vw_errors_t,
            ) = [], [], [], []
            (
                test_loss,
                hfvr_errors_v,
                vw_errors_v,
            ) = self.evaluate_model_collate_eval(
                test_loader,  # loss_fn=criterion
            )
            hfvr_MAE_v = np.mean(np.abs(hfvr_errors_v))
            vw_MAE_v = np.mean(np.abs(vw_errors_v))
            (
                hfvr_errors_v,
                vw_errors_v,
            ) = [], []
            dt = time.time() - t1
            print(
                f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {vw_MAE_t:>7.4f}/{vw_MAE_v:<7.4f}",
                flush=True,
            )
        return test_loss

    def train_batches_single_proc(
        self, rank, dataloader, criterion, optimizer, rank_device
    ):
        self.model.train()
        total_hfvr_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_vw_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        for batch in dataloader:
            batch = batch.to(rank_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            params = self.model(batch)
            hirshfeld_volume_ratios = params[:, 0]
            valence_widths = params[:, 1]

            hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
            vw_error = valence_widths - batch.valence_widths

            hfvr_loss = (hfvr_error**2).mean()
            vw_loss = (vw_error**2).mean()

            loss = hfvr_loss + vw_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += hfvr_error.numel()

            total_hfvr_error += hfvr_error.detach().abs().sum()
            total_vw_error += vw_error.detach().abs().sum()

        final_count = total_count.item()

        # Calculating MAEs
        hfvr_mae = total_hfvr_error.item() / final_count
        vw_mae = total_vw_error.item() / final_count
        return total_loss, hfvr_mae, vw_mae

    def train_batches(self, rank, dataloader, criterion, optimizer, rank_device):
        self.model.train()
        total_hfvr_error = 0
        total_vw_error = 0
        total_loss = 0
        count = 0

        for batch in dataloader:
            batch = batch.to(rank_device)
            optimizer.zero_grad()
            params = self.model(batch)
            hirshfeld_volume_ratios = params[:, 0]
            valence_widths = params[:, 1]

            hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
            vw_error = valence_widths - batch.valence_widths

            hfvr_loss = torch.mean(torch.square(hfvr_error))
            vw_loss = torch.mean(torch.square(vw_error))

            loss = hfvr_loss + vw_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += hfvr_error.numel()

            total_hfvr_error += torch.sum(torch.abs(hfvr_error)).item()
            total_vw_error += torch.sum(torch.abs(vw_error)).item()

        # Converting to tensors for all-reduce
        total_hfvr_error = torch.tensor(
            total_hfvr_error, dtype=torch.float32, device=rank_device
        )
        total_vw_error = torch.tensor(
            total_vw_error, dtype=torch.float32, device=rank_device
        )
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_hfvr_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_vw_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        # Calculating MAEs
        hfvr_mae = total_hfvr_error.item() / count.item()
        vw_mae = total_vw_error.item() / count.item()

        return total_loss, hfvr_mae, vw_mae

    def evaluate_batches_single_proc(self, rank, dataloader, criterion, rank_device):
        self.model.eval()
        total_hfvr_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_vw_error = torch.zeros([], dtype=torch.float32, device=rank_device)
        total_loss = 0.0

        total_count = torch.zeros([], dtype=torch.int, device=rank_device)

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(rank_device, non_blocking=True)
                params = self.model(batch)
                hirshfeld_volume_ratios = params[:, 0]
                valence_widths = params[:, 1]

                hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
                vw_error = valence_widths - batch.valence_widths

                hfvr_loss = (hfvr_error**2).mean()
                vw_loss = (vw_error**2).mean()

                loss = hfvr_loss + vw_loss
                total_loss += loss.item()
                total_count += hfvr_error.numel()

                total_hfvr_error += hfvr_error.abs().sum()
                total_vw_error += vw_error.abs().sum()

        final_count = total_count.item()

        # Calculating MAEs
        hfvr_mae = total_hfvr_error.item() / final_count
        vw_mae = total_vw_error.item() / final_count
        return total_loss, hfvr_mae, vw_mae

    def evaluate_batches(self, rank, dataloader, criterion, rank_device):
        self.model.eval()
        total_hfvr_error = 0
        total_vw_error = 0
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(rank_device)
                params = self.model(batch)
                hirshfeld_volume_ratios = params[:, 0]
                valence_widths = params[:, 1]

                hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
                vw_error = valence_widths - batch.valence_widths

                hfvr_loss = (hfvr_error**2).mean()
                vw_loss = (vw_error**2).mean()
                hfvr_error = hirshfeld_volume_ratios - batch.volume_ratios
                vw_error = valence_widths - batch.valence_widths

                total_hfvr_error += torch.sum(torch.abs(hfvr_error)).item()
                total_vw_error += torch.sum(torch.abs(vw_error)).item()

                hfvr_loss = torch.mean(torch.square(hfvr_error))
                vw_loss = torch.mean(torch.square(vw_error))

                total_loss += hfvr_loss + vw_loss
                count += hfvr_error.numel()

        # Converting to tensors for all-reduce
        total_hfvr_error = torch.tensor(
            total_hfvr_error, dtype=torch.float32, device=rank_device
        )
        total_vw_error = torch.tensor(
            total_vw_error, dtype=torch.float32, device=rank_device
        )
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_hfvr_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_vw_error, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

        total_loss = torch.tensor(total_loss.item(), device=rank_device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        # Calculating MAEs
        hfvr_mae = total_hfvr_error.item() / count.item()
        vw_mae = total_vw_error.item() / count.item()
        return total_loss, hfvr_mae, vw_mae

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
            # collate_fn=atomic_collate_update,
            collate_fn=atomic_hfvr_vw_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            # collate_fn=atomic_collate_update,
            collate_fn=atomic_hfvr_vw_collate_update,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        test_loss = self.pretrain_statistics(train_loader, test_loader, criterion)

        lowest_test_loss = test_loss

        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            train_loss, charge_MAE_t, dipole_MAE_t, qpole_MAE_t, hfvr_MAE_t = (
                self.train_batches(
                    rank, train_loader, criterion, optimizer, rank_device
                )
            )
            test_loss, charge_MAE_v, dipole_MAE_v, qpole_MAE_v, hfvr_MAE_v = (
                self.evaluate_batches(rank, test_loader, criterion, rank_device)
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        cpu_model = unwrap_model(self.model).to("cpu")
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "param_start_mean": cpu_model.param_start_mean,
                                    "param_start_std": cpu_model.param_start_std,
                                    "n_params": cpu_model.n_params,
                                    "r_cut": cpu_model.r_cut,
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
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {test_lowered}",
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
        skip_compile=False,
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
            # collate_fn=atomic_collate_update,
            collate_fn=atomic_hfvr_vw_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # collate_fn=atomic_collate_update,
            collate_fn=atomic_hfvr_vw_collate_update,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        lowest_test_loss = torch.tensor(float("inf"))
        t1 = time.time()
        train_loss, hfvr_MAE_t, vw_MAE_t = self.evaluate_batches_single_proc(
            rank, train_loader, criterion, rank_device
        )
        test_loss, hfvr_MAE_v, vw_MAE_v = self.evaluate_batches_single_proc(
            rank, test_loader, criterion, rank_device
        )
        dt = time.time() - t1
        print(f"                                Hirshfeld Vol Ratio   Valence Width")
        print(
            f"  (Pre-training) ({dt:<7.2f} sec)  MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {vw_MAE_t:>7.4f}/{vw_MAE_v:<7.4f}",
            flush=True,
        )
        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            (
                train_loss,
                hfvr_MAE_t,
                vw_MAE_t,
            ) = self.train_batches_single_proc(
                rank, train_loader, criterion, optimizer, rank_device
            )
            test_loss, hfvr_MAE_v, vw_MAE_v = self.evaluate_batches_single_proc(
                rank, test_loader, criterion, rank_device
            )

            if rank == 0:
                if test_loss < lowest_test_loss:
                    lowest_test_loss = test_loss
                    test_lowered = "*"
                    if self.model_save_path:
                        # cpu_model = self.model.to("cpu")
                        cpu_model = unwrap_model(self.model).to("cpu")
                        torch.save(
                            {
                                "model_state_dict": cpu_model.state_dict(),
                                "config": {
                                    "n_message": cpu_model.n_message,
                                    "n_neuron": cpu_model.n_neuron,
                                    "n_embed": cpu_model.n_embed,
                                    "param_start_mean": cpu_model.param_start_mean,
                                    "param_start_std": cpu_model.param_start_std,
                                    "n_params": cpu_model.n_params,
                                    "r_cut": cpu_model.r_cut,
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
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {vw_MAE_t:>7.4f}/{vw_MAE_v:<7.4f} {test_lowered}",
                    flush=True,
                )

            # n = gc.collect()
            # print("    Garbage collector: collected %d objects." % n)
            # if rank_device != "cpu":
            #     torch.cuda.empty_cache()
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
        skip_compile=True,
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

        np.random.seed(42)
        torch.manual_seed(42)
        random_indices = np.random.permutation(len(self.dataset))
        train_indices = random_indices[: int(len(self.dataset) * split_percent)]
        test_indices = random_indices[int(len(self.dataset) * split_percent) :]
        if random_seed:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            train_indices = np.random.permutation(train_indices)
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
        print(f"  {self.model.n_params=}", flush=True)
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
        qA, muA, thA, hfvrA, vwA, hlistA = self.model_predict(batch)
        batch = batch.cpu()
        qA = qA.detach().detach().cpu()
        muA = muA.detach().detach().cpu()
        thA = thA.detach().detach().cpu()
        hfvrA = hfvrA.detach().detach().cpu()
        vwA = vwA.detach().detach().cpu()
        hlistA = hlistA.detach().cpu()
        if isolate_predictions:
            return isolate_atomic_property_predictions(
                batch, (qA, muA, thA, hfvrA, vwA, hlistA)
            )
        else:
            return qA, muA, thA, hfvrA, vwA, hlistA

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
                (
                    charges,
                    dipoles,
                    qpoles,
                    hirshfeld_volume_ratios,
                    valence_widths,
                    hlists,
                ) = self.model_predict(batch)
                # need to use batch.molecule_ind to reassemble the output
                mol_charges = [[] for i in range(batch_size)]
                mol_dipoles = [[] for i in range(batch_size)]
                mol_qpoles = [[] for i in range(batch_size)]
                mol_hfvr = [[] for i in range(batch_size)]
                mol_vw = [[] for i in range(batch_size)]
                for n, i in enumerate(batch.molecule_ind):
                    mol_charges[i].append(charges[n])
                    mol_dipoles[i].append(dipoles[n])
                    mol_qpoles[i].append(qpoles[n])
                    mol_hfvr[i].append(hirshfeld_volume_ratios[n])
                    mol_vw[i].append(valence_widths[n])
                output.append(
                    (mol_charges, mol_dipoles, mol_qpoles, mol_hfvr, mol_vw, hlists)
                )
        return output

    @torch.inference_mode()
    def predict_qcel_mols(self, mols, batch_size=2):
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
                    charge, dipole, qpole, hfvr, vw, hlist = self.model(batch)
                    # Isolate atomic properties by molecule
                    (
                        mol_charges,
                        mol_dipoles,
                        mol_qpoles,
                        mol_hfvrs,
                        mol_vws,
                        mol_hlists,
                    ) = isolate_atomic_property_predictions(
                        batch, (charge, dipole, qpole, hfvr, vw, hlist)
                    )
                    output.extend(
                        list(
                            zip(
                                mol_charges,
                                mol_dipoles,
                                mol_qpoles,
                                mol_hfvrs,
                                mol_vws,
                                mol_hlists,
                            )
                        )
                    )
                mol_data = []
        return output

    @torch.inference_mode()
    def model_predict(self, data):
        charge, dipole, qpole, hirshfeld_volume_ratios, valence_widths, hlist = (
            self.model(data)
        )
        return charge, dipole, qpole, hirshfeld_volume_ratios, valence_widths, hlist
