import torch
import torch.nn as nn
from apnet_pt.util import scatter_sum_compile
from torch_geometric.data import Data
import numpy as np
import time
from ..AtomModels.ap2_atom_model import AtomMPNN, isolate_atomic_property_predictions
from .. import atomic_datasets
from ..hf_pretrained import resolve_pretrained_paths
from .. import pairwise_datasets
from .. import model_io
from ..pairwise_datasets import (
    APNet2_DataLoader,
    apnet2_collate_update,
    apnet2_collate_update_prebatched,
    pairwise_edges,
    pairwise_edges_im,
    qcel_dimer_to_pyg_data,
)
from ..pt_datasets.dapnet_ds import (
    dapnet2_module_dataset,
    dapnet2_module_dataset_apnetStored,
    dapnet2_collate_update_no_target,
)
from ..AtomPairwiseModels.apnet2 import (
    APNet2_MPNN,
    InverseTimeDecayLR,
)
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import qcelemental as qcel
from copy import deepcopy


class APNet2_dAPNet2_MPNN(nn.Module):
    def __init__(
        self,
        apnet2_model: APNet2_MPNN,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut_im=8.0,
        r_cut=5.0,
    ):
        super().__init__()

        self.n_message = n_message
        self.n_rbf = n_rbf
        self.n_neuron = n_neuron
        self.n_embed = n_embed
        self.r_cut_im = r_cut_im
        self.r_cut = r_cut
        self.apnet2_model = apnet2_model
        for param in self.apnet2_model.parameters():
            # Freeze the APNet2 model parameters to only train the readout
            # layer
            param.requires_grad = False

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
        ]
        self.readout_layer_energy = self._make_layers(
            layer_nodes_readout, layer_activations
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
        }

    def forward(
        self,
        ZA,
        RA,
        ZB,
        RB,
        # short range, intermolecular edges
        e_ABsr_source,
        e_ABsr_target,
        dimer_ind,
        # long range, intermolecular edges
        e_ABlr_source,
        e_ABlr_target,
        dimer_ind_lr,
        # intramonomer edges (monomer A)
        e_AA_source,
        e_AA_target,
        # intramonomer edges (monomer B)
        e_BB_source,
        e_BB_target,
        # monomer charges
        total_charge_A,
        total_charge_B,
        # monomer A properties
        qA,
        muA,
        quadA,
        hlistA,
        # monomer B properties
        qB,
        muB,
        quadB,
        hlistB,
    ):
        """
        Compute per-dimer energy outputs by running the inner APNet2 model and applying the frozen readout to short-range interaction embeddings.

        Parameters:
            ZA (Tensor): Atomic numbers for monomer A, batched for all dimers.
            RA (Tensor): Positions for atoms in monomer A.
            ZB (Tensor): Atomic numbers for monomer B.
            RB (Tensor): Positions for atoms in monomer B.
            e_ABsr_source, e_ABsr_target (Tensor): Short-range intermolecular edge index tensors (source and target).
            dimer_ind (Tensor): Mapping from short-range edge/global entries to dimer indices used for aggregation.
            e_ABlr_source, e_ABlr_target (Tensor): Long-range intermolecular edge index tensors (source and target).
            dimer_ind_lr (Tensor): Mapping for long-range entries to dimer indices.
            e_AA_source, e_AA_target (Tensor): Intramonomer edge indices for monomer A.
            e_BB_source, e_BB_target (Tensor): Intramonomer edge indices for monomer B.
            total_charge_A, total_charge_B (Tensor): Total charges for monomer A and B for each dimer.
            qA, muA, quadA, hlistA (Tensor): Predicted atomic multipole and descriptor arrays for monomer A.
            qB, muB, quadB, hlistB (Tensor): Predicted atomic multipole and descriptor arrays for monomer B.

        Returns:
            E_output (Tensor): Per-dimer aggregated energy tensor expanded to match the original number of dimers; shape (ndimer, C) where C is readout output channels.
            E_sr (Tensor): Short-range energy contributions from the inner APNet2 model prior to readout aggregation.
            E_elst_sr (Tensor): Short-range electrostatic energy components from the inner APNet2 model.
            E_elst_lr (Tensor): Long-range electrostatic energy components from the inner APNet2 model.
            hAB (Tensor): Short-range embedding/features for interactions A->B produced by the inner APNet2 model.
            hBA (Tensor): Short-range embedding/features for interactions B->A produced by the inner APNet2 model.
        """
        E_pairmodel, E_sr, E_elst_sr, E_elst_lr, hAB, hBA, cutoff = self.apnet2_model(
            ZA,
            RA,
            ZB,
            RB,
            # short range, intermolecular edges
            e_ABsr_source,
            e_ABsr_target,
            dimer_ind,
            # long range, intermolecular edges
            e_ABlr_source,
            e_ABlr_target,
            dimer_ind_lr,
            # intramonomer edges (monomer A)
            e_AA_source,
            e_AA_target,
            # intramonomer edges (monomer B)
            e_BB_source,
            e_BB_target,
            # monomer charges
            total_charge_A,
            total_charge_B,
            # monomer A properties
            qA,
            muA,
            quadA,
            hlistA,
            # monomer B properties
            qB,
            muB,
            quadB,
            hlistB,
        )
        EAB_sr = self.readout_layer_energy(hAB)
        EBA_sr = self.readout_layer_energy(hBA)

        delta_E = EAB_sr + EBA_sr
        delta_E *= cutoff
        E = scatter_sum_compile(
            delta_E, dimer_ind, dim_size=int(dimer_ind.max()) + 1, reduce="sum"
        )

        # Need to ensure that the output is the same size as input dimers
        ndimer = torch.tensor(total_charge_A.size(0), dtype=torch.long)
        N_sr, num_cols = E.shape
        E_expanded = E.new_zeros((ndimer, num_cols))
        E_expanded[:N_sr] = E
        E_output = E_expanded
        return E_output, E_sr, E_elst_sr, E_elst_lr, hAB, hBA


class dAPNet2_MPNN(nn.Module):
    def __init__(
        self,
        n_neuron=128,
    ):
        super().__init__()

        self.n_neuron = n_neuron
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
        ]
        self.readout_layer_energy = self._make_layers(
            layer_nodes_readout, layer_activations
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
            "n_neuron": self.n_neuron,
        }

    def forward(
        self,
        h_AB,
        h_BA,
        cutoff,
        dimer_ind,
        ndimer,
    ):
        """
        Compute aggregated per-dimer energy predictions from pairwise readout embeddings.

        Parameters:
            h_AB (Tensor): Readout embeddings for AB-directed pairs.
            h_BA (Tensor): Readout embeddings for BA-directed pairs.
            cutoff (Tensor or float): Per-pair multiplicative cutoff weights applied to predicted pair energies.
            dimer_ind (LongTensor): 1D index tensor mapping each pair row to a dimer index for aggregation.
            ndimer (int): Total number of dimers in the original batch; determines the first dimension of the output.

        Returns:
            Tensor: Aggregated energy tensor of shape (ndimer, C) where C is the number of output channels from the readout; each row is the sum of scaled pair contributions for that dimer.
        """
        EAB_sr = self.readout_layer_energy(h_AB)
        EBA_sr = self.readout_layer_energy(h_BA)

        delta_E = EAB_sr + EBA_sr
        delta_E *= cutoff
        E = scatter_sum_compile(
            delta_E, dimer_ind, dim_size=int(dimer_ind.max()) + 1, reduce="sum"
        )
        # Need to ensure that the output is the same size as input dimers
        N_sr, num_cols = E.shape
        E_expanded = E.new_zeros((ndimer, num_cols))
        E_expanded[:N_sr] = E
        E_output = E_expanded
        return E_output


class APNet2_dAPNet2Model:
    def __init__(
        self,
        apnet2_mpnn,
        dataset=None,
        atom_model=None,
        pre_trained_model_path=None,
        atom_model_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut_im=8.0,
        r_cut=5.0,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_m1="",
        ds_m2="",
        ds_root="data",
        ds_max_size=None,
        ds_atomic_batch_size=200,
        ds_force_reprocess=False,
        ds_skip_process=False,
        ds_num_devices=1,
        ds_datapoint_storage_n_objects=1000,
        ds_prebatched=False,
        print_lvl=0,
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
        self.atom_model = AtomMPNN()
        self.apnet2_mpnn = apnet2_mpnn

        if atom_model_pre_trained_path:
            print(
                f"Loading pre-trained AtomMPNN model from {atom_model_pre_trained_path}"
            )
            checkpoint = torch.load(
                atom_model_pre_trained_path, map_location=device, weights_only=False
            )
            self.atom_model = AtomMPNN(
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
            self.atom_model.load_state_dict(model_state_dict)
        elif atom_model:
            self.atom_model = atom_model
        else:
            print(
                """No atom model provided.
    Assuming atomic multipoles and embeddings are
    pre-computed and passes as input to the model.
"""
            )
        if pre_trained_model_path:
            print(
                f"Loading pre-trained APNet2_MPNN model from {pre_trained_model_path}"
            )
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = APNet2_dAPNet2_MPNN(
                apnet2_model=apnet2_mpnn,
                n_message=checkpoint["config"]["n_message"],
                n_rbf=checkpoint["config"]["n_rbf"],
                n_neuron=checkpoint["config"]["n_neuron"],
                n_embed=checkpoint["config"]["n_embed"],
                r_cut_im=checkpoint["config"]["r_cut_im"],
                r_cut=checkpoint["config"]["r_cut"],
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = APNet2_dAPNet2_MPNN(
                # atom_model=self.atom_model,
                apnet2_model=apnet2_mpnn,
                n_message=n_message,
                n_rbf=n_rbf,
                n_neuron=n_neuron,
                n_embed=n_embed,
                r_cut_im=r_cut_im,
                r_cut=r_cut,
            )
        split_dbs = [1]
        self.dataset = dataset
        if (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type not in split_dbs
        ):

            def setup_ds(fp=ds_force_reprocess):
                return dapnet2_module_dataset(
                    root=ds_root,
                    r_cut=r_cut,
                    r_cut_im=r_cut_im,
                    spec_type=ds_spec_type,
                    max_size=ds_max_size,
                    force_reprocess=fp,
                    atom_model_path=atom_model_pre_trained_path,
                    atomic_batch_size=ds_atomic_batch_size,
                    num_devices=ds_num_devices,
                    skip_processed=ds_skip_process,
                    datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                    prebatched=ds_prebatched,
                    print_level=print_lvl,
                    m1=ds_m1,
                    m2=ds_m2,
                )

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset = self.dataset[:ds_max_size]
        elif (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type in split_dbs
        ):
            print("Processing Split dataset...")

            def setup_ds(fp=ds_force_reprocess):
                return [
                    dapnet2_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model_path=atom_model_pre_trained_path,
                        atomic_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        split="train",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        prebatched=ds_prebatched,
                        print_level=print_lvl,
                        m1=ds_m1,
                        m2=ds_m2,
                    ),
                    dapnet2_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model_path=atom_model_pre_trained_path,
                        atomic_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        split="test",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        prebatched=ds_prebatched,
                        print_level=print_lvl,
                        m1=ds_m1,
                        m2=ds_m2,
                    ),
                ]

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset[0] = self.dataset[0][:ds_max_size]
                self.dataset[1] = self.dataset[1][:ds_max_size]
        print(self.dataset)
        self.model.to(device)
        self.device = device
        self.batch_size = None
        self.shuffle = False
        self.model_save_path = None
        self.prebatched = ds_prebatched
        return

    @torch.inference_mode()
    def predict_from_dataset(self):
        self.model.eval()
        for batch in self.dataset:
            batch = batch.to(self.device)
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
        return

    def compile_model(self):
        # self.model.to(self.device)
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
        torch._dynamo.config.capture_scalar_outputs = False
        self.model = torch.compile(self.model)
        return

    def set_pretrained_model(
        self, ap2_model_path=None, am_model_path=None, model_id=None
    ):
        """
        Load pretrained model weights from checkpoint files.

        Parameters
        ----------
        ap2_model_path : str, optional
            Path to the dAPNet2 checkpoint file
        am_model_path : str, optional
            Path to the atom model checkpoint file
        model_id : int, optional
            Model ID to load from bundled ensemble models

        Returns
        -------
        self
            The model instance with loaded weights
        """
        if model_id is not None:
            model_paths = resolve_pretrained_paths(
                [
                    f"am_ensemble/am_{model_id}.pt",
                    f"ap2_ensemble/ap2_{model_id}.pt",
                ]
            )
            am_model_path = model_paths[f"am_ensemble/am_{model_id}.pt"]
            ap2_model_path = model_paths[f"ap2_ensemble/ap2_{model_id}.pt"]
        elif ap2_model_path is None or am_model_path is None:
            raise ValueError(
                "Provide both ap2_model_path and am_model_path, or set model_id."
            )

        # Load main model checkpoint
        checkpoint = model_io.load_checkpoint(ap2_model_path, map_location=self.device)
        state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)

        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            self.model.load_state_dict(state_dict)
        else:
            state_dict_with_prefix = {
                f"_orig_mod.{k}": v for k, v in state_dict.items()
            }
            self.model.load_state_dict(state_dict_with_prefix)

        # Load atom model checkpoint
        am_checkpoint = model_io.load_checkpoint(
            am_model_path, map_location=self.device
        )
        am_state_dict = model_io.load_state_dict_from_checkpoint(am_checkpoint)

        if "_orig_mod" not in list(self.atom_model.state_dict().keys())[0]:
            self.atom_model.load_state_dict(am_state_dict)
        else:
            am_state_dict_with_prefix = {
                f"_orig_mod.{k}": v for k, v in am_state_dict.items()
            }
            self.atom_model.load_state_dict(am_state_dict_with_prefix)
        return self

    def _create_checkpoint(self, metadata: dict = None) -> dict:
        """
        Create a v2 checkpoint dictionary for this model.

        The checkpoint embeds the apnet2_model and atom_model as submodels.

        Parameters
        ----------
        metadata : dict, optional
            Additional metadata to include in the checkpoint

        Returns
        -------
        dict
            Complete v2 checkpoint dictionary
        """
        cpu_model = model_io.unwrap_model(self.model).to("cpu")
        config = cpu_model.get_config()

        # Create submodel checkpoints
        submodels = {}

        # Embed the APNet2 model
        if hasattr(cpu_model, "apnet2_model") and cpu_model.apnet2_model is not None:
            if hasattr(cpu_model.apnet2_model, "get_config"):
                apnet2_config = cpu_model.apnet2_model.get_config()
            else:
                apnet2_config = {
                    "n_message": getattr(cpu_model.apnet2_model, "n_message", 3),
                    "n_rbf": getattr(cpu_model.apnet2_model, "n_rbf", 8),
                    "n_neuron": getattr(cpu_model.apnet2_model, "n_neuron", 128),
                    "n_embed": getattr(cpu_model.apnet2_model, "n_embed", 8),
                    "r_cut_im": getattr(cpu_model.apnet2_model, "r_cut_im", 8.0),
                    "r_cut": getattr(cpu_model.apnet2_model, "r_cut", 5.0),
                }
            submodels["apnet2_model"] = model_io.create_submodel_checkpoint(
                model=cpu_model.apnet2_model,
                config=apnet2_config,
                model_type="APNet2_MPNN",
            )

        # Embed the atom model
        if hasattr(self, "atom_model") and self.atom_model is not None:
            if hasattr(self.atom_model, "get_config"):
                atom_config = self.atom_model.get_config()
            else:
                atom_config = {
                    "n_message": getattr(self.atom_model, "n_message", 3),
                    "n_rbf": getattr(self.atom_model, "n_rbf", 8),
                    "n_neuron": getattr(self.atom_model, "n_neuron", 128),
                    "n_embed": getattr(self.atom_model, "n_embed", 8),
                    "r_cut": getattr(self.atom_model, "r_cut", 5.0),
                }
            submodels["atom_model"] = model_io.create_submodel_checkpoint(
                model=self.atom_model,
                config=atom_config,
                model_type="AtomMPNN",
            )

        checkpoint = model_io.create_checkpoint(
            model=cpu_model,
            config=config,
            model_type="APNet2_dAPNet2_MPNN",
            submodels=submodels if submodels else None,
            metadata=metadata,
        )

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

    ############################################################################
    # The main forward/eval function
    ############################################################################
    def eval_fn(self, batch):
        return self.model(
            ZA=batch.ZA,
            RA=batch.RA,
            ZB=batch.ZB,
            RB=batch.RB,
            e_ABsr_source=batch.e_ABsr_source,
            e_ABsr_target=batch.e_ABsr_target,
            dimer_ind=batch.dimer_ind,
            e_ABlr_source=batch.e_ABlr_source,
            e_ABlr_target=batch.e_ABlr_target,
            dimer_ind_lr=batch.dimer_ind_lr,
            e_AA_source=batch.e_AA_source,
            e_AA_target=batch.e_AA_target,
            e_BB_source=batch.e_BB_source,
            e_BB_target=batch.e_BB_target,
            total_charge_A=batch.total_charge_A,
            total_charge_B=batch.total_charge_B,
            qA=batch.qA,
            muA=batch.muA,
            quadA=batch.quadA,
            hlistA=batch.hlistA,
            qB=batch.qB,
            muB=batch.muB,
            quadB=batch.quadB,
            hlistB=batch.hlistB,
        )

    def _qcel_example_input(
        self,
        mols,
        batch_size=1,
        r_cut=5.0,
        r_cut_im=8.0,
    ):
        mol_data = [[*qcel_dimer_to_pyg_data(mol)] for mol in mols]
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            data_A = [d[0] for d in batch_mol_data]
            data_B = [d[1] for d in batch_mol_data]
            batch_A = atomic_datasets.atomic_collate_update_no_target(data_A)
            batch_B = atomic_datasets.atomic_collate_update_no_target(data_B)
            with torch.no_grad():
                am_out_A = self.atom_model(batch_A)
                am_out_B = self.atom_model(batch_B)
                qAs, muAs, quadAs, hlistAs = isolate_atomic_property_predictions(
                    batch_A, am_out_A
                )
                qBs, muBs, quadBs, hlistBs = isolate_atomic_property_predictions(
                    batch_B, am_out_B
                )
                if len(batch_A.total_charge.size()) == 0:
                    batch_A.total_charge = batch_A.total_charge.unsqueeze(0)
                if len(batch_B.total_charge.size()) == 0:
                    batch_B.total_charge = batch_B.total_charge.unsqueeze(0)
                dimer_ls = []
                for j in range(len(batch_mol_data)):
                    qA, muA, quadA, hlistA = qAs[j], muAs[j], quadAs[j], hlistAs[j]
                    qB, muB, quadB, hlistB = qBs[j], muBs[j], quadBs[j], hlistBs[j]
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                    e_AA_source, e_AA_target = pairwise_edges(data_A[j].R, r_cut)
                    e_BB_source, e_BB_target = pairwise_edges(data_B[j].R, r_cut)
                    e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = (
                        pairwise_edges_im(data_A[j].R, data_B[j].R, r_cut_im)
                    )
                    dimer_ind = torch.ones((1), dtype=torch.long) * 0
                    data = Data(
                        ZA=data_A[j].x,
                        RA=data_A[j].R,
                        ZB=data_B[j].x,
                        RB=data_B[j].R,
                        # short range, intermolecular edges
                        e_ABsr_source=e_ABsr_source,
                        e_ABsr_target=e_ABsr_target,
                        dimer_ind=dimer_ind,
                        # long range, intermolecular edges
                        e_ABlr_source=e_ABlr_source,
                        e_ABlr_target=e_ABlr_target,
                        dimer_ind_lr=dimer_ind,
                        # intramonomer edges (monomer A)
                        e_AA_source=e_AA_source,
                        e_AA_target=e_AA_target,
                        # intramonomer edges (monomer B)
                        e_BB_source=e_BB_source,
                        e_BB_target=e_BB_target,
                        # monomer charges
                        total_charge_A=data_A[j].total_charge,
                        total_charge_B=data_B[j].total_charge,
                        # monomer A properties
                        qA=qA,
                        muA=muA,
                        quadA=quadA,
                        hlistA=hlistA,
                        # monomer B properties
                        qB=qB,
                        muB=muB,
                        quadB=quadB,
                        hlistB=hlistB,
                    )
                    dimer_ls.append(data)
                dimer_batch = pairwise_datasets.apnet2_collate_update_no_target(
                    dimer_ls
                )
        return dimer_batch

    @torch.inference_mode()
    def predict_qcel_mols(
        self,
        mols,
        batch_size=1,
        r_cut=5.0,
        r_cut_im=8.0,
    ):
        mol_data = [[*qcel_dimer_to_pyg_data(mol)] for mol in mols]
        predictions = np.zeros((len(mol_data), 1))
        for i in range(0, len(mol_data), batch_size):
            batch_mol_data = mol_data[i : i + batch_size]
            data_A = [d[0] for d in batch_mol_data]
            data_B = [d[1] for d in batch_mol_data]
            batch_A = atomic_datasets.atomic_collate_update_no_target(data_A)
            batch_B = atomic_datasets.atomic_collate_update_no_target(data_B)
            with torch.no_grad():
                am_out_A = self.atom_model(batch_A)
                am_out_B = self.atom_model(batch_B)
                qAs, muAs, quadAs, hlistAs = isolate_atomic_property_predictions(
                    batch_A, am_out_A
                )
                qBs, muBs, quadBs, hlistBs = isolate_atomic_property_predictions(
                    batch_B, am_out_B
                )
                if len(batch_A.total_charge.size()) == 0:
                    batch_A.total_charge = batch_A.total_charge.unsqueeze(0)
                if len(batch_B.total_charge.size()) == 0:
                    batch_B.total_charge = batch_B.total_charge.unsqueeze(0)
                dimer_ls = []
                for j in range(len(batch_mol_data)):
                    qA, muA, quadA, hlistA = qAs[j], muAs[j], quadAs[j], hlistAs[j]
                    qB, muB, quadB, hlistB = qBs[j], muBs[j], quadBs[j], hlistBs[j]
                    if len(qA.size()) == 0:
                        qA = qA.unsqueeze(0).unsqueeze(0)
                    elif len(qA.size()) == 1:
                        qA = qA.unsqueeze(-1)
                    if len(qB.size()) == 0:
                        qB = qB.unsqueeze(0).unsqueeze(0)
                    elif len(qB.size()) == 1:
                        qB = qB.unsqueeze(-1)
                        e_AA_source, e_AA_target = pairwise_edges(data_A[j].R, r_cut)
                        e_BB_source, e_BB_target = pairwise_edges(data_B[j].R, r_cut)
                        e_ABsr_source, e_ABsr_target, e_ABlr_source, e_ABlr_target = (
                            pairwise_edges_im(data_A[j].R, data_B[j].R, r_cut_im)
                        )
                        dimer_ind = torch.ones((1), dtype=torch.long) * 0
                        data = Data(
                            ZA=data_A[j].x,
                            RA=data_A[j].R,
                            ZB=data_B[j].x,
                            RB=data_B[j].R,
                            # short range, intermolecular edges
                            e_ABsr_source=e_ABsr_source,
                            e_ABsr_target=e_ABsr_target,
                            dimer_ind=dimer_ind,
                            # long range, intermolecular edges
                            e_ABlr_source=e_ABlr_source,
                            e_ABlr_target=e_ABlr_target,
                            dimer_ind_lr=dimer_ind,
                            # intramonomer edges (monomer A)
                            e_AA_source=e_AA_source,
                            e_AA_target=e_AA_target,
                            # intramonomer edges (monomer B)
                            e_BB_source=e_BB_source,
                            e_BB_target=e_BB_target,
                            # monomer charges
                            total_charge_A=data_A[j].total_charge,
                            total_charge_B=data_B[j].total_charge,
                            # monomer A properties
                            qA=qA,
                            muA=muA,
                            quadA=quadA,
                            hlistA=hlistA,
                            # monomer B properties
                            qB=qB,
                            muB=muB,
                            quadB=quadB,
                            hlistB=hlistB,
                        )
                        dimer_ls.append(data)
                dimer_batch = pairwise_datasets.apnet2_collate_update_no_target(
                    dimer_ls
                )
                dimer_batch.to(self.device)
                preds = self.eval_fn(dimer_batch)
                predictions[i : i + batch_size] = preds[0].cpu().numpy()
        return predictions

    def example_input(self):
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
        return self._qcel_example_input([mol], batch_size=1)

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
            optimizer.zero_grad(set_to_none=True)  # minor speed-up
            batch = batch.to(rank_device, non_blocking=True)
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
            preds = E_sr_dimer.flatten()
            # print(f"{preds=}")
            # print(f"{batch.y=}")
            comp_errors = preds - batch.y
            # print(f"{comp_errors=}")
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
    def __evaluate_batches_single_proc(self, dataloader, loss_fn, rank_device):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer, _, _, _, _, _ = self.eval_fn(batch)
                preds = E_sr_dimer.flatten()
                try:
                    comp_errors = preds - batch.y
                except Exception as e:
                    print(f"Error in batch {n}: {e}")
                    print(batch)
                    print(batch.y)
                    print(batch.qA)
                    print(batch.dimer_ind)
                    raise e
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, batch.y)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t))
        return total_loss, total_MAE_t

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
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
            preds = E_sr_dimer.reshape(-1, 4)
            comp_errors = preds - batch.y
            if loss_fn is None:
                batch_loss = torch.mean(torch.square(comp_errors))
            else:
                batch_loss = loss_fn(preds, batch.y)

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
                E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                comp_errors = preds - batch.y
                if loss_fn is None:
                    batch_loss = torch.mean(torch.square(comp_errors))
                else:
                    batch_loss = loss_fn(preds, batch.y)

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
            first_pass_data = APNet2_DataLoader(
                dataset=test_dataset[:batch_size],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=apnet2_collate_update,
            )
            for b in first_pass_data:
                b.to(rank_device)
                self.eval_fn(b)
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

        train_loader = APNet2_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            collate_fn=apnet2_collate_update,
        )

        test_loader = APNet2_DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=apnet2_collate_update,
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
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} {indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} {disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f} {test_lowered}",
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
    ):
        # (1) Compile Model
        rank_device = self.device
        self.model.to(rank_device)
        batch = self.example_input()
        batch.to(rank_device)
        self.model(**batch)
        best_model = deepcopy(self.model)
        if not skip_compile:
            print("Compiling model")
            self.compile_model()

        # (2) Dataloaders
        if train_dataset.prebatched:
            collate_fn = apnet2_collate_update_prebatched
        else:
            collate_fn = apnet2_collate_update
        print(f"{num_workers = }")
        train_loader = APNet2_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test_loader = APNet2_DataLoader(
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
        criterion = None  # defaults to MSE

        # (4) Print table header
        print(
            "                                       Energy",
            flush=True,
        )

        # (5) Evaluate once pre-training
        t0 = time.time()
        train_loss, total_MAE_t = self.__evaluate_batches_single_proc(
            train_loader, criterion, rank_device
        )
        test_loss, total_MAE_v = self.__evaluate_batches_single_proc(
            test_loader, criterion, rank_device
        )

        print(
            f"  (Pre-training) ({time.time() - t0:<7.2f}s)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f}",
            flush=True,
        )

        # (6) Main training loop
        lowest_test_loss = test_loss
        for epoch in range(n_epochs):
            t1 = time.time()
            train_loss, total_MAE_t = self.__train_batches_single_proc(
                train_loader, criterion, optimizer, rank_device, scheduler
            )
            test_loss, total_MAE_v = self.__evaluate_batches_single_proc(
                test_loader, criterion, rank_device
            )

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

            print(
                f"  EPOCH: {epoch:4d} ({time.time() - t1:<7.2f}s)  MAE: "
                f"{total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {star_marker}",
                flush=True,
            )
        self.model = best_model

    def train(
        self,
        dataset=None,
        n_epochs=50,
        lr=5e-4,
        split_percent=0.9,
        model_path=None,
        shuffle=False,
        dataloader_num_workers=4,
        optimize_for_speed=True,
        world_size=1,
        omp_num_threads_per_process=6,
        lr_decay=None,
        random_seed=42,
        skip_compile=False,
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

        print("~~ Training APNet2Model ~~", flush=True)
        print(
            f"    Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples"
        )
        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {self.model.n_message=}", flush=True)
        print(f"  {self.model.n_neuron=}", flush=True)
        print(f"  {self.model.n_embed=}", flush=True)
        print(f"  {self.model.n_rbf=}", flush=True)
        print(f"  {self.model.r_cut=}", flush=True)
        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {lr=}\n", flush=True)
        print(f"  {lr_decay=}\n", flush=True)
        print(f"  {batch_size=}", flush=True)

        if self.device.type == "cuda":
            pin_memory = True
        else:
            pin_memory = False

        # if optimize_for_speed:
        # torch.jit.enable_onednn_fusion(False)
        # torch.autograd.set_detect_anomaly(True)

        self.shuffle = shuffle

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
            )
        return


class dAPNet2Model:
    def __init__(
        self,
        apnet2_model=None,
        dataset=None,
        atom_model=None,
        pre_trained_model_path=None,
        atom_model_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut_im=8.0,
        r_cut=5.0,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_m1="",
        ds_m2="",
        ds_root="data",
        ds_max_size=None,
        ds_atomic_batch_size=200,
        ds_force_reprocess=False,
        ds_skip_process=False,
        ds_num_devices=1,
        ds_datapoint_storage_n_objects=1000,
        ds_prebatched=False,
        print_lvl=0,
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
        self.atom_model = AtomMPNN()
        self.apnet2_model = apnet2_model

        if atom_model_pre_trained_path:
            print(
                f"Loading pre-trained AtomMPNN model from {atom_model_pre_trained_path}"
            )
            checkpoint = torch.load(
                atom_model_pre_trained_path, map_location=device, weights_only=False
            )
            self.atom_model = AtomMPNN(
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
            self.atom_model.load_state_dict(model_state_dict)
        elif atom_model:
            self.atom_model = atom_model
        else:
            print(
                """No atom model provided.
    Assuming atomic multipoles and embeddings are
    pre-computed and passes as input to the model.
"""
            )
        if pre_trained_model_path:
            print(
                f"Loading pre-trained APNet2_MPNN model from {pre_trained_model_path}"
            )
            checkpoint = torch.load(pre_trained_model_path, weights_only=False)
            self.model = dAPNet2_MPNN(
                n_neuron=checkpoint["config"]["n_neuron"],
            )
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = dAPNet2_MPNN(
                n_neuron=n_neuron,
            )
        split_dbs = [1]
        self.dataset = dataset
        if (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type not in split_dbs
        ):

            def setup_ds(fp=ds_force_reprocess):
                return dapnet2_module_dataset_apnetStored(
                    root=ds_root,
                    r_cut=r_cut,
                    r_cut_im=r_cut_im,
                    spec_type=ds_spec_type,
                    max_size=ds_max_size,
                    force_reprocess=fp,
                    atom_model_path=atom_model_pre_trained_path,
                    preprocessing_batch_size=ds_atomic_batch_size,
                    num_devices=ds_num_devices,
                    skip_processed=ds_skip_process,
                    datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                    prebatched=ds_prebatched,
                    print_level=print_lvl,
                    m1=ds_m1,
                    m2=ds_m2,
                )

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset = self.dataset[:ds_max_size]
        elif (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type in split_dbs
        ):
            print("Processing Split dataset...")

            def setup_ds(fp=ds_force_reprocess):
                return [
                    dapnet2_module_dataset_apnetStored(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model_path=atom_model_pre_trained_path,
                        preprocessing_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        split="train",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        prebatched=ds_prebatched,
                        print_level=print_lvl,
                        m1=ds_m1,
                        m2=ds_m2,
                    ),
                    dapnet2_module_dataset_apnetStored(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=r_cut_im,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model_path=atom_model_pre_trained_path,
                        preprocessing_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        split="test",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        prebatched=ds_prebatched,
                        print_level=print_lvl,
                        m1=ds_m1,
                        m2=ds_m2,
                    ),
                ]

            self.dataset = setup_ds()
            self.dataset = setup_ds(False)
            if ds_max_size:
                self.dataset[0] = self.dataset[0][:ds_max_size]
                self.dataset[1] = self.dataset[1][:ds_max_size]
        print(self.dataset)
        self.model.to(device)
        self.device = device
        self.batch_size = None
        self.shuffle = False
        self.model_save_path = None
        self.prebatched = ds_prebatched
        return

    @torch.inference_mode()
    def predict_from_dataset(self):
        self.model.eval()
        for batch in self.dataset:
            batch = batch.to(self.device)
            E_sr_dimer = self.eval_fn(batch)
        return

    def compile_model(self):
        self.model.to(self.device)
        torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
        torch._dynamo.config.capture_scalar_outputs = False
        self.model = torch.compile(self.model)
        return

    def set_pretrained_model(
        self, ap2_model_path=None, am_model_path=None, model_id=None
    ):
        if model_id is not None:
            model_paths = resolve_pretrained_paths(
                [
                    f"am_ensemble/am_{model_id}.pt",
                    f"ap2_ensemble/ap2_{model_id}.pt",
                ]
            )
            am_model_path = model_paths[f"am_ensemble/am_{model_id}.pt"]
            ap2_model_path = model_paths[f"ap2_ensemble/ap2_{model_id}.pt"]
        elif ap2_model_path is None or am_model_path is None:
            raise ValueError(
                "Provide both ap2_model_path and am_model_path, or set model_id."
            )

        checkpoint = torch.load(ap2_model_path)
        print(checkpoint)
        if "_orig_mod" not in list(self.model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.apnet2_model.load_state_dict(model_state_dict)
        else:
            self.apnet2_model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint = torch.load(am_model_path)
        if "_orig_mod" not in list(self.atom_model.state_dict().keys())[0]:
            model_state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            self.atom_model.load_state_dict(model_state_dict)
        else:
            self.atom_model.load_state_dict(checkpoint["model_state_dict"])
        return self

    def _create_checkpoint(self, metadata: dict = None) -> dict:
        """
        Create a v2 checkpoint dictionary for this model.

        The checkpoint embeds the apnet2_model and atom_model as submodels.

        Parameters
        ----------
        metadata : dict, optional
            Additional metadata to include in the checkpoint

        Returns
        -------
        dict
            Complete v2 checkpoint dictionary
        """
        cpu_model = model_io.unwrap_model(self.model).to("cpu")
        config = cpu_model.get_config()

        # Create submodel checkpoints
        submodels = {}

        # Embed the APNet2 model if available
        if hasattr(self, "apnet2_model") and self.apnet2_model is not None:
            if hasattr(self.apnet2_model, "model"):
                # apnet2_model is a harness, get the underlying model
                apnet2_nn = self.apnet2_model.model
            else:
                apnet2_nn = self.apnet2_model

            if hasattr(apnet2_nn, "get_config"):
                apnet2_config = apnet2_nn.get_config()
            else:
                apnet2_config = {
                    "n_message": getattr(apnet2_nn, "n_message", 3),
                    "n_rbf": getattr(apnet2_nn, "n_rbf", 8),
                    "n_neuron": getattr(apnet2_nn, "n_neuron", 128),
                    "n_embed": getattr(apnet2_nn, "n_embed", 8),
                    "r_cut_im": getattr(apnet2_nn, "r_cut_im", 8.0),
                    "r_cut": getattr(apnet2_nn, "r_cut", 5.0),
                }
            submodels["apnet2_model"] = model_io.create_submodel_checkpoint(
                model=apnet2_nn,
                config=apnet2_config,
                model_type="APNet2_MPNN",
            )

        # Embed the atom model
        if hasattr(self, "atom_model") and self.atom_model is not None:
            if hasattr(self.atom_model, "get_config"):
                atom_config = self.atom_model.get_config()
            else:
                atom_config = {
                    "n_message": getattr(self.atom_model, "n_message", 3),
                    "n_rbf": getattr(self.atom_model, "n_rbf", 8),
                    "n_neuron": getattr(self.atom_model, "n_neuron", 128),
                    "n_embed": getattr(self.atom_model, "n_embed", 8),
                    "r_cut": getattr(self.atom_model, "r_cut", 5.0),
                }
            submodels["atom_model"] = model_io.create_submodel_checkpoint(
                model=self.atom_model,
                config=atom_config,
                model_type="AtomMPNN",
            )

        checkpoint = model_io.create_checkpoint(
            model=cpu_model,
            config=config,
            model_type="dAPNet2_MPNN",
            submodels=submodels if submodels else None,
            metadata=metadata,
        )

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

    ############################################################################
    # The main forward/eval function
    ############################################################################
    def eval_fn(self, batch):
        return self.model(
            h_AB=batch.h_AB,
            h_BA=batch.h_BA,
            cutoff=batch.cutoff,
            dimer_ind=batch.dimer_ind,
            ndimer=batch.ndimer,
        )

    def _qcel_example_input(
        self,
        mols,
        batch_size=1,
        r_cut=5.0,
        r_cut_im=8.0,
    ):
        dimers = []
        for i in range(0, len(mols) + len(mols) % batch_size + 1, batch_size):
            upper_bound = min(i + batch_size, len(mols))
            local_mols = mols[i:upper_bound]
            if len(local_mols) == 0:
                break
            _, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers = (
                self.apnet2_model.predict_qcel_mols(
                    mols=local_mols,
                    batch_size=batch_size,
                    r_cut=self.apnet2_model.model.r_cut,
                    r_cut_im=self.apnet2_model.model.r_cut_im,
                )
            )
            dimer_data = Data(
                h_AB=h_ABs[0],
                h_BA=h_BAs[0],
                cutoff=cutoffs[0],
                dimer_ind=dimer_inds[0],
                ndimer=ndimers[0],
            )
            dimers.append(dimer_data)
        dimer_batch = dapnet2_collate_update_no_target(dimers)
        return dimer_batch

    @torch.inference_mode()
    def predict_qcel_mols(
        self,
        mols,
        batch_size=1,
        r_cut=5.0,
        r_cut_im=8.0,
    ) -> np.ndarray:
        predictions = np.zeros((len(mols)))
        for i in range(0, len(mols) + len(mols) % batch_size + 1, batch_size):
            upper_bound = min(i + batch_size, len(mols))
            local_mols = mols[i:upper_bound]
            if len(local_mols) == 0:
                break
            _, h_ABs, h_BAs, cutoffs, dimer_inds, ndimers = (
                self.apnet2_model.predict_qcel_mols(
                    mols=local_mols,
                    batch_size=batch_size,
                    r_cut=self.apnet2_model.model.r_cut,
                    r_cut_im=self.apnet2_model.model.r_cut_im,
                )
            )
            dimer_batch = Data(
                h_AB=h_ABs[0],
                h_BA=h_BAs[0],
                cutoff=cutoffs[0],
                dimer_ind=dimer_inds[0],
                ndimer=ndimers[0],
            )
            dimer_batch.to(self.device)
            preds = self.eval_fn(dimer_batch)
            preds = preds.flatten()
            predictions[i : i + batch_size] = preds.cpu().numpy()
        return predictions

    def example_input(self):
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
        return self._qcel_example_input([mol], batch_size=1)

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
            optimizer.zero_grad(set_to_none=True)  # minor speed-up
            batch = batch.to(rank_device, non_blocking=True)
            E_sr_dimer = self.eval_fn(batch)
            preds = E_sr_dimer.flatten()
            comp_errors = preds - batch.y
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
    def __evaluate_batches_single_proc(self, dataloader, loss_fn, rank_device):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                E_sr_dimer = self.eval_fn(batch)
                preds = E_sr_dimer.flatten()
                try:
                    comp_errors = preds - batch.y
                except Exception as e:
                    print(f"Error in batch {n}: {e}")
                    print(batch)
                    print(batch.y)
                    print(batch.qA)
                    print(batch.dimer_ind)
                    raise e
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, batch.y)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t))
        return total_loss, total_MAE_t

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
            E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
            preds = E_sr_dimer.reshape(-1, 4)
            comp_errors = preds - batch.y
            if loss_fn is None:
                batch_loss = torch.mean(torch.square(comp_errors))
            else:
                batch_loss = loss_fn(preds, batch.y)

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
                E_sr_dimer, E_sr, E_elst_sr, E_elst_lr, hAB, hBA = self.eval_fn(batch)
                preds = E_sr_dimer.reshape(-1, 4)
                comp_errors = preds - batch.y
                if loss_fn is None:
                    batch_loss = torch.mean(torch.square(comp_errors))
                else:
                    batch_loss = loss_fn(preds, batch.y)

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
            first_pass_data = APNet2_DataLoader(
                dataset=test_dataset[:batch_size],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=apnet2_collate_update,
            )
            for b in first_pass_data:
                b.to(rank_device)
                self.eval_fn(b)
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

        train_loader = APNet2_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            collate_fn=apnet2_collate_update,
        )

        test_loader = APNet2_DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=apnet2_collate_update,
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
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {elst_MAE_t:>7.3f}/{elst_MAE_v:<7.3f} {exch_MAE_t:>7.3f}/{exch_MAE_v:<7.3f} {indu_MAE_t:>7.3f}/{indu_MAE_v:<7.3f} {disp_MAE_t:>7.3f}/{disp_MAE_v:<7.3f} {test_lowered}",
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
    ):
        # (1) Compile Model
        rank_device = self.device
        self.model.to(rank_device)
        batch = self.example_input()
        batch.to(rank_device)
        self.model(**batch)
        # if False:
        if not skip_compile:
            print("Compiling model")
            self.compile_model()

        # (2) Dataloaders
        if train_dataset.prebatched:
            collate_fn = apnet2_collate_update_prebatched
        else:
            collate_fn = apnet2_collate_update
        print(f"{num_workers = }")
        train_loader = APNet2_DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test_loader = APNet2_DataLoader(
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
        criterion = None  # defaults to MSE

        # (4) Print table header
        print(
            "                                       Energy",
            flush=True,
        )

        # (5) Evaluate once pre-training
        t0 = time.time()
        train_loss, total_MAE_t = self.__evaluate_batches_single_proc(
            train_loader, criterion, rank_device
        )
        test_loss, total_MAE_v = self.__evaluate_batches_single_proc(
            test_loader, criterion, rank_device
        )

        print(
            f"  (Pre-training) ({time.time() - t0:<7.2f}s)  MAE: {total_MAE_t:>7.3f}/{total_MAE_v:<7.3f}",
            flush=True,
        )

        # (6) Main training loop
        lowest_test_loss = test_loss
        for epoch in range(n_epochs):
            t1 = time.time()
            train_loss, total_MAE_t = self.__train_batches_single_proc(
                train_loader, criterion, optimizer, rank_device, scheduler
            )
            test_loss, total_MAE_v = self.__evaluate_batches_single_proc(
                test_loader, criterion, rank_device
            )

            # Track best model
            star_marker = " "
            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                star_marker = "*"
                if self.model_save_path:
                    self.save_model(
                        self.model_save_path,
                        metadata={
                            "training_mode": "single_proc",
                            "epoch": epoch,
                        },
                    )

            print(
                f"  EPOCH: {epoch:4d} ({time.time() - t1:<7.2f}s)  MAE: "
                f"{total_MAE_t:>7.3f}/{total_MAE_v:<7.3f} {star_marker}",
                flush=True,
            )

    def train(
        self,
        dataset=None,
        n_epochs=50,
        lr=5e-4,
        split_percent=0.9,
        model_path=None,
        shuffle=False,
        dataloader_num_workers=4,
        optimize_for_speed=True,
        world_size=1,
        omp_num_threads_per_process=6,
        lr_decay=None,
        random_seed=42,
        skip_compile=False,
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

        print("~~ Training APNet2Model ~~", flush=True)
        print(
            f"    Training on {len(train_dataset)} samples, Testing on {len(test_dataset)} samples"
        )
        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {self.model.n_neuron=}", flush=True)
        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {lr=}\n", flush=True)
        print(f"  {lr_decay=}\n", flush=True)
        print(f"  {batch_size=}", flush=True)

        if self.device.type == "cuda":
            pin_memory = True
        else:
            pin_memory = False

        # if optimize_for_speed:
        # torch.jit.enable_onednn_fusion(False)
        # torch.autograd.set_detect_anomaly(True)

        self.shuffle = shuffle

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
            )
        return
