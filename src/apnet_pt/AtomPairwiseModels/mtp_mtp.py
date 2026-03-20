import os
import re
import time
from copy import deepcopy
from importlib import resources

import numpy as np
import qcelemental as qcel
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data

from apnet_pt.torch_util import set_weights_to_value
from qcml_dftd3.d3 import d3, resolve_d3_damping_parameters

from .. import constants
from .. import model_io
from ..atomic_datasets import (
    AtomicDataLoader,
    atomic_collate_update,
    atomic_collate_update_no_target,
    atomic_collate_update_prebatched,
)
from ..AtomModels.ap2_atom_model import (  # isolate_atomic_property_predictions,; DistanceLayer,
    AtomMPNN,
    qcel_mon_to_pyg_data,
    unwrap_model,
)
from ..AtomModels.ap2_hirshfeld_atom_model import (
    AtomHirshfeldMPNN,
    atomic_hirshfeld_module_dataset,
    isolate_atomic_property_predictions,
)
from ..hf_pretrained import resolve_pretrained_path
from ..multipole import thole_damping_direct_torch, thole_damping_mutual_torch
from ..pt_datasets.ap2_fused_ds import (
    APNet2_fused_DataLoader,
    ap2_fused_collate_update,
    ap2_fused_collate_update_no_target,
    ap2_fused_module_dataset,
    qcel_dimer_to_fused_data,
)
from ..util import scatter_sum_compile

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


class DimerProp(nn.Module):
    def __init__(
        self,
        ATParam,
        dimer_eval="elst_damping",
        elst_damping_type="CLIFF",
        d3_damping_parameters=None,
        freeze_atom_model=True,
    ):
        """
        Create a DimerProp configured with an AtomTypeParam and selected evaluation and damping modes.

        Parameters:
            ATParam: An AtomTypeParam instance providing per-atom parameter tensors and an `atom_model` used for multipole predictions. The `atom_model` will be frozen (requires_grad set to False) when freeze_atom_model is True (default).
            dimer_eval (str): Name of the dimer evaluation forward mode to use (e.g., "elst_damping", "induced_dipole", "elst").
            elst_damping_type (str): Electrostatic damping scheme to apply for damped elst evaluations; supported values include "CLIFF" and "AMOEBA".
        """
        super().__init__()
        self.AtomTypeParam = ATParam
        if freeze_atom_model:
            self.AtomTypeParam.atom_model.requires_grad_(False)
        self.elst_damping_type = elst_damping_type
        self.set_d3_damping_parameters(d3_damping_parameters)
        self.set_forward(dimer_eval)
        return

    def set_d3_damping_parameters(self, d3_damping_parameters=None):
        self.d3_damping_parameters = resolve_d3_damping_parameters(
            d3_damping_parameters
        )
        return

    def set_forward(self, dimer_eval):
        """
        Configure which forward method the instance will use and set related resources.

        Parameters:
            dimer_eval (str): Mode selector for the dimer evaluation. Accepted values:
                - "elst_damping": use damped electrostatics (_elst_damping_forward)
                - "elst_damping_AMOEBA": use AMOEBA-style damped electrostatics (_elst_damping_AMOEBA_forward)
                - "elst": use undamped electrostatics (_elst_forward)
                - "induced_dipole": compute induction via induced dipoles (_indu_induced_dipole_forward)
                - "induced_dipole_param": induction using parameterized polarizabilities (_indu_induced_dipole_param_forward)
                - "elst_damping__induced_dipole": combined damped electrostatics and induction (_elst_damping_indu_induced_dipole_forward)
                - "ap3_elst_damping__induced_dipole": AP3-specific damped electrostatics plus induction (_ap3_elst_damping_indu_induced_dipole_forward)
                - "ap3_atomMPNN": return AP3 atom multipole parameters only (_ap3_atomMPNN)

        Notes:
            - This method sets self.forward to the corresponding internal forward implementation.
            - For modes that compute induction ("induced_dipole", "induced_dipole_param", and the combined induction modes), this method also clones the global polarizability table into self.polarizability_table.
            - Raises ValueError if dimer_eval is not one of the accepted mode strings.
        """
        if dimer_eval == "elst_damping":
            self.forward = self._elst_damping_forward
        elif dimer_eval == "elst_damping_AMOEBA":
            self.forward = self._elst_damping_AMOEBA_forward
        elif dimer_eval == "elst":
            self.forward = self._elst_forward
        elif dimer_eval == "induced_dipole":
            self.forward = self._indu_induced_dipole_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "induced_dipole_param":
            self.forward = self._indu_induced_dipole_param_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "elst_damping__induced_dipole":
            self.forward = self._elst_damping_indu_induced_dipole_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "ap3_elst_damping__induced_dipole":
            self.forward = self._ap3_elst_damping_indu_induced_dipole_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "ap3_elst_damping__induced_dipole__disp":
            self.forward = self._ap3_elst_damping_indu_induced_dipole_disp_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "disp":
            self.forward = self._disp_forward
        elif dimer_eval == "ap3_atomMPNN":
            self.forward = self._ap3_atomMPNN
        else:
            raise ValueError(f"Unknown dimer_eval: {dimer_eval}")

    def get_config(self) -> dict:
        """
        Return a reconstruction config for this DimerProp hierarchy.
        """

        def _infer_nested_r_cut(model):
            current = model
            while current is not None:
                if hasattr(current, "r_cut"):
                    return getattr(current, "r_cut")
                current = getattr(current, "atom_model", None)
            return None

        atom_type_param_config = None
        atom_type_param_type = None
        atom_model_config = None
        atom_model_type = None

        if hasattr(self, "AtomTypeParam") and self.AtomTypeParam is not None:
            atom_type_param_type = type(self.AtomTypeParam).__name__
            if hasattr(self.AtomTypeParam, "get_config"):
                atom_type_param_config = self.AtomTypeParam.get_config()

            if hasattr(self.AtomTypeParam, "atom_model"):
                atom_model = self.AtomTypeParam.atom_model
                atom_model_type = type(atom_model).__name__
                if hasattr(atom_model, "get_config"):
                    atom_model_config = atom_model.get_config()
                    nested_r_cut = _infer_nested_r_cut(atom_model)
                    if nested_r_cut is not None and "r_cut" not in atom_model_config:
                        atom_model_config["r_cut"] = nested_r_cut

        return {
            "dimer_eval": getattr(getattr(self, "forward", None), "__name__", None),
            "elst_damping_type": self.elst_damping_type,
            "d3_damping_parameters": deepcopy(self.d3_damping_parameters),
            "atom_type_param_type": atom_type_param_type,
            "atom_type_param_config": atom_type_param_config,
            "atom_model_type": atom_model_type,
            "atom_model_config": atom_model_config,
        }

    def _elst_damping_forward(
        self,
        batch,
    ):
        """
        Compute the damped electrostatic energy for a batched dimer and return per-atom parameter outputs.

        Parameters:
            batch: Batched dimer data containing at least the following attributes used for the evaluation:
                - ZA, ZB: nuclear charges for fragments A and B
                - RA, RB: Cartesian coordinates for fragments A and B
                - batch_atomic_A, batch_atomic_B: atom index mappings for AtomTypeParam lookup
                - e_ABsr_source, e_ABsr_target: edge source/target indices for short-range A–B interactions
                The function also uses the AtomTypeParam module attached to self and self.elst_damping_type to select the damping variant.

        Returns:
            Elst: Tensor of electrostatic energy values for the batch (damped MTP–MTP A–B interactions).
            v_A: Tuple/list of per-atom parameter tensors produced for fragment A (e.g., monopole, dipole, quadrupole, ...).
            v_B: Tuple/list of per-atom parameter tensors produced for fragment B (e.g., monopole, dipole, quadrupole, ...).
        """
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        Ka = torch.abs(v_A[-1])
        Kb = torch.abs(v_B[-1])
        # print(f"{Ka =}")
        # print(f"{v_A[0] =}")

        # Select damping function based on elst_damping_type
        if self.elst_damping_type == "AMOEBA":
            damping_fn = mtp_elst_damping_AMOEBA
        else:  # Default to CLIFF
            damping_fn = mtp_elst_damping

        Elst = damping_fn(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=Ka,
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=Kb,
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
        )
        return Elst, v_A, v_B

    def _elst_damping_forward_AMOEBA(
        self,
        batch,
    ):
        """
        Compute the AMOEBA-damped multipole electrostatic energy for a batched dimer and return per-atom parameter tensors.

        Parameters:
            batch: Batched dimer data object containing at least ZA, ZB (atomic numbers), RA, RB (coordinates), e_ABsr_source, e_ABsr_target (short-range inter-molecular edge index arrays), and batch_atomic_A / batch_atomic_B indices used by the AtomTypeParam module.

        Returns:
            Elst (torch.Tensor): Batched AMOEBA-damped electrostatic energy for each dimer in the input batch.
            v_A (tuple): Per-atom multipole parameter tensors produced for molecule A (q, mu, quad, ..., last element used to derive Ka).
            v_B (tuple): Per-atom multipole parameter tensors produced for molecule B (q, mu, quad, ..., last element used to derive Kb).
        """
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        Ka = torch.abs(v_A[-1])
        Kb = torch.abs(v_B[-1])
        # print(f"{Ka =}")
        # print(f"{v_A[0] =}")

        Elst = mtp_elst_damping_AMOEBA(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=Ka,
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=Kb,
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
        )
        return Elst, v_A, v_B

    def _elst_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        # print(f"{v_A[-1] =}")
        Elst = mtp_elst(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
        )
        return Elst, v_A, v_B

    def _elst_ind_ap3_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        # print(f"{v_A[-1] =}")
        Elst = mtp_elst(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
        )
        return Elst, v_A, v_B

    def _indu_induced_dipole_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        # print(f"{v_A[3] =}")
        # print(f"{v_A[4] =}")
        # Ka = torch.tensor([1.8398, 2.4643, 2.5112, 1.8398, 2.4643, 2.5112], requires_grad=True)
        # Kb = torch.tensor([1.8398, 2.4643, 2.5112, 1.8398, 2.4643, 2.5112], requires_grad=True)
        Ka = v_A[-1]
        Kb = v_B[-1]
        Indu = induced_dipole_induction_optimized(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            # Ka=v_A[-1],
            Ka=Ka,
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            # Kb=v_B[-1],
            Kb=Kb,
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
            # Additional parameters for induction
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=torch.abs(v_A[3]),
            hirshfeld_volume_ratio_B=torch.abs(v_B[3]),
            valence_widths_A=v_A[4],
            valence_widths_B=v_B[4],
            polarizability_table=self.polarizability_table,
        )
        return Indu, v_A, v_B

    def _indu_induced_dipole_param_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        # Ka = torch.tensor([1.8398, 2.4643, 2.5112, 1.8398, 2.4643, 2.5112], requires_grad=True)
        # Kb = torch.tensor([1.8398, 2.4643, 2.5112, 1.8398, 2.4643, 2.5112], requires_grad=True)
        # print(f"{Ka =}")
        Ka = v_A[-1]
        Kb = v_B[-1]
        Indu = induced_dipole_induction_optimized(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=Ka,
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=Kb,
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
            # Additional parameters for induction
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=torch.abs(v_A[-2][:, 0]),
            hirshfeld_volume_ratio_B=torch.abs(v_B[-2][:, 0]),
            valence_widths_A=v_A[-2][:, 1],
            valence_widths_B=v_B[-2][:, 1],
            polarizability_table=self.polarizability_table,
        )
        # if Indu.isnan().any():
        #     print("Induced dipole energy is NaN, debugging info:")
        #     print(f"{v_A[-2] =}")
        #     print(f"{v_B[-2] =}")
        #     print(f"{v_A[-1] =}")
        #     print(f"{v_B[-1] =}")
        #     print(f"{Ka =}")
        #     print(f"{Kb =}")
        #     raise ValueError("Induced dipole energy is NaN")
        return Indu, v_A, v_B

    def _elst_damping_indu_induced_dipole_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        Kas = torch.abs(v_A[-1])
        Kbs = torch.abs(v_B[-1])
        # print(f"{Kas =}")
        # print(f"{v_A[-1] =}")
        # print(f"{v_A[-2] =}")
        # Ka = Kas[:, 1]
        # Kb = Kbs[:, 1]
        # print(f"{Kas =}")
        # print(f"{Kbs =}")
        # Ka = torch.clamp(v_A[-1][:, 1], min=0.0001, max=20.0)
        # Kb = torch.clamp(v_B[-1][:, 1], min=0.0001, max=20.0)
        # Ka = torch.tensor([1.8398, 2.4643, 2.5112, 1.8398, 2.4643, 2.5112], requires_grad=True)
        # Kb = torch.tensor([1.8398, 2.4643, 2.5112, 1.8398, 2.4643, 2.5112], requires_grad=True)

        Indu = induced_dipole_induction_optimized(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=Kas[:, 1],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=Kbs[:, 1],
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
            # Additional parameters for induction
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=torch.abs(v_A[-2][:, 0]),
            hirshfeld_volume_ratio_B=torch.abs(v_B[-2][:, 0]),
            valence_widths_A=v_A[-2][:, 1],
            valence_widths_B=v_B[-2][:, 1],
            polarizability_table=self.polarizability_table,
        )
        # if Indu.isnan().any():
        #     print("Induced dipole energy is NaN, debugging info:")
        #     print(f"{Indu = }")
        #     print(f"{v_A[-2] =}")
        #     print(f"{v_B[-2] =}")
        #     print(f"{v_A[-1] =}")
        #     print(f"{v_B[-1] =}")
        #     raise ValueError("Induced dipole energy is NaN")
        # Must compute Elst after Ind because we modify qA and qB in place... pain to debug

        Elst = mtp_elst_damping(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=Kas[:, 0],
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=Kbs[:, 0],
            e_AB_source=batch.e_ABsr_source,
            e_AB_target=batch.e_ABsr_target,
        )
        # if Elst.isnan().any():
        #     print("Electrostatic energy is NaN, debugging info:")
        #     print(f"{v_A[-1] =}")
        #     print(f"{v_B[-1] =}")
        #     raise ValueError("Electrostatic energy is NaN")
        return torch.vstack((Elst, Indu)).T, v_A, v_B

    def _ap3_elst_damping_indu_induced_dipole_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        Kas = torch.abs(v_A[-1])
        Kbs = torch.abs(v_B[-1])
        # print(f"{Kas =}")
        # print(f"{v_A[-1] =}")
        # print(f"{v_A[-2] =}")
        # print(batch.e_ABsr_source)
        # print(batch.e_ABlr_source)
        # print(batch.e_ABfull_source)
        Indu = induced_dipole_induction_optimized_no_correction(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
            # Additional parameters for induction
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=torch.abs(v_A[-2][:, 0]),
            hirshfeld_volume_ratio_B=torch.abs(v_B[-2][:, 0]),
            polarizability_table=self.polarizability_table,
        )
        # if Indu.isnan().any():
        #     print("Induced dipole energy is NaN, debugging info:")
        #     torch.save(batch, "ind_nan_batch.pt")
        #     print(f"{v_A[-2] =}")
        #     print(f"{v_B[-2] =}")
        #     print(f"{v_A[-1] =}")
        #     print(f"{v_B[-1] =}")
        #     raise ValueError("Induced dipole energy is NaN")
        # Must compute Elst after Ind because we modify qA and qB in place... pain to debug

        Elst = mtp_elst_damping(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=Kas,
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=Kbs,
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
        )
        # if Elst.isnan().any():
        #     print("Electrostatic energy is NaN, debugging info:")
        #     torch.save(batch, "elst_nan_batch.pt")
        #     print(f"{v_A[-1] =}")
        #     print(f"{v_B[-1] =}")
        #     raise ValueError("Electrostatic energy is NaN")
        return torch.vstack((Elst, Indu)).T, v_A, v_B

    def _ap3_atomMPNN(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)

        return v_A, v_B

    def _disp_forward(
        self,
        batch,
    ):
        """
        Compute only the dispersion energy using DFTD3.
        """
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)

        Disp = d3(batch, params=self.d3_damping_parameters)
        return Disp, v_A, v_B

    def _ap3_elst_damping_indu_induced_dipole_disp_forward(
        self,
        batch,
    ):
        v_A = self.AtomTypeParam(batch.batch_atomic_A)
        v_B = self.AtomTypeParam(batch.batch_atomic_B)
        Kas = torch.abs(v_A[-1])
        Kbs = torch.abs(v_B[-1])
        # print(f"{Kas =}")
        # print(f"{v_A[-1] =}")
        # print(f"{v_A[-2] =}")
        Indu = induced_dipole_induction_optimized_no_correction(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
            # Additional parameters for induction
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=torch.abs(v_A[-2][:, 0]),
            hirshfeld_volume_ratio_B=torch.abs(v_B[-2][:, 0]),
            polarizability_table=self.polarizability_table,
        )
        if Indu.isnan().any():
            print("Induced dipole energy is NaN, debugging info:")
            torch.save(batch, "ind_nan_batch.pt")
            print(f"{v_A[-2] =}")
            print(f"{v_B[-2] =}")
            print(f"{v_A[-1] =}")
            print(f"{v_B[-1] =}")
            raise ValueError("Induced dipole energy is NaN")
        # Must compute Elst after Ind because we modify qA and qB in place... pain to debug

        Elst = mtp_elst_damping(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=v_A[0],
            muA=v_A[1],
            quadA=v_A[2],
            Ka=Kas,
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=v_B[0],
            muB=v_B[1],
            quadB=v_B[2],
            Kb=Kbs,
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
        )
        if Elst.isnan().any():
            print("Electrostatic energy is NaN, debugging info:")
            torch.save(batch, "elst_nan_batch.pt")
            print(f"{v_A[-1] =}")
            print(f"{v_B[-1] =}")
            raise ValueError("Electrostatic energy is NaN")

        Disp = d3(batch, params=self.d3_damping_parameters)
        return torch.vstack((Elst, Indu, Disp)).T, v_A, v_B


def _substate_dict(state_dict: dict, prefix: str) -> dict:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _infer_max_index(state_dict: dict, pattern: str) -> int | None:
    matches = []
    regex = re.compile(pattern)
    for key in state_dict:
        match = regex.match(key)
        if match:
            matches.append(int(match.group(1)))
    if not matches:
        return None
    return max(matches)


def _infer_atommpnn_from_state_dict(
    state_dict: dict,
    r_cut: float = 5.0,
):
    n_message_max = _infer_max_index(
        state_dict,
        r"^charge_update_layers\.(\d+)\.0\.weight$",
    )
    n_message = 0 if n_message_max is None else n_message_max + 1
    n_rbf = int(state_dict["distance_layer.frequencies"].shape[0])
    n_embed = int(state_dict["embed_layer.weight"].shape[1])
    n_neuron = int(state_dict["charge_readout_layers.0.0.weight"].shape[0] // 2)
    return AtomMPNN(
        n_message=n_message,
        n_rbf=n_rbf,
        n_neuron=n_neuron,
        n_embed=n_embed,
        r_cut=r_cut,
    )


def _infer_atomtypeparamnn_from_state_dict(
    state_dict: dict,
    r_cut: float = 5.0,
):
    nested_atom_type = any(
        key.startswith("atom_model.param_readout_layers.") for key in state_dict
    )
    atom_model_state = _substate_dict(state_dict, "atom_model.")
    if nested_atom_type:
        atom_model = _infer_atomtypeparamnn_from_state_dict(
            atom_model_state, r_cut=r_cut
        )
    else:
        atom_model = _infer_atommpnn_from_state_dict(atom_model_state, r_cut=r_cut)

    n_params_max = _infer_max_index(
        state_dict,
        r"^param_readout_layers\.(\d+)\.\d+\.0\.weight$",
    )
    n_params = 1 if n_params_max is None else n_params_max + 1
    n_message_max = _infer_max_index(
        state_dict,
        r"^param_readout_layers\.\d+\.(\d+)\.0\.weight$",
    )
    n_message = 0 if n_message_max is None else n_message_max
    first_weight = state_dict["param_readout_layers.0.0.0.weight"]
    n_embed = int(first_weight.shape[1])
    n_neuron = int(first_weight.shape[0] // 2)

    return AtomTypeParamNN(
        atom_model=atom_model,
        n_message=n_message,
        n_neuron=n_neuron,
        n_embed=n_embed,
        param_start_mean=[0.0] * n_params,
        param_start_std=[0.01] * n_params,
        n_params=n_params,
        freeze_atom_model=False,
    )


def load_dimer_prop_from_checkpoint(
    checkpoint: dict,
    freeze_atom_model: bool = False,
):
    """
    Reconstruct a DimerProp model from a v1/v2 checkpoint.

    This supports older v2 checkpoints that embedded the full recursive state
    dict but did not yet store enough nested config to rebuild the hierarchy
    directly from config alone.
    """
    config = model_io.load_config_from_checkpoint(checkpoint) or {}
    state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
    atom_type_param_state = _substate_dict(state_dict, "AtomTypeParam.")

    atom_type_param_config = config.get("atom_type_param_config") or {}
    atom_type_param_type = config.get("atom_type_param_type", "AtomTypeParamNN")
    atom_model_config = config.get("atom_model_config") or {}
    atom_model_type = config.get("atom_model_type")
    r_cut = atom_model_config.get("r_cut", config.get("r_cut", 5.0))

    if atom_type_param_type == "AtomTypeParamNN" and atom_type_param_config:
        atom_model_state = _substate_dict(atom_type_param_state, "atom_model.")
        if atom_model_type == "AtomMPNN" and atom_model_config:
            atom_model = AtomMPNN(
                n_message=atom_model_config["n_message"],
                n_rbf=atom_model_config["n_rbf"],
                n_neuron=atom_model_config["n_neuron"],
                n_embed=atom_model_config["n_embed"],
                r_cut=atom_model_config["r_cut"],
            )
        else:
            atom_model = _infer_atomtypeparamnn_from_state_dict(
                atom_model_state, r_cut=r_cut
            )

        atom_type_param = AtomTypeParamNN(
            atom_model=atom_model,
            n_message=atom_type_param_config["n_message"],
            n_neuron=atom_type_param_config["n_neuron"],
            n_embed=atom_type_param_config["n_embed"],
            param_start_mean=atom_type_param_config["param_start_mean"],
            param_start_std=atom_type_param_config["param_start_std"],
            n_params=atom_type_param_config.get("n_params", 1),
            freeze_atom_model=False,
        )
    else:
        atom_type_param = _infer_atomtypeparamnn_from_state_dict(
            atom_type_param_state,
            r_cut=r_cut,
        )

    dimer_prop = DimerProp(
        ATParam=atom_type_param,
        freeze_atom_model=freeze_atom_model,
        elst_damping_type=config.get("elst_damping_type", "CLIFF"),
        d3_damping_parameters=config.get("d3_damping_parameters"),
    )
    dimer_prop.load_state_dict(state_dict)
    return dimer_prop


class AtomTypeParamNN(nn.Module):
    def __init__(
        self,
        atom_model: AtomMPNN = AtomMPNN(),
        n_message=3,
        n_neuron=128,
        n_embed=8,
        param_start_mean=1.8,
        param_start_std=0.01,
        n_params=1,
        freeze_atom_model=True,
    ):
        super().__init__()
        self.atom_model = atom_model
        if freeze_atom_model:
            self.atom_model.requires_grad_(False)
        self.n_message = n_message
        if type(self.atom_model) in [AtomMPNN, AtomHirshfeldMPNN]:
            self.h_list_ind = -1
        elif type(self.atom_model) is AtomTypeParamNN:
            self.h_list_ind = 3
        else:
            raise ValueError("Unknown atom_model type")
        self.n_neuron = n_neuron
        self.n_embed = n_embed
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

        self.param_start_mean = param_start_mean
        self.param_start_std = param_start_std
        self.n_params = n_params
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
        for p in range(n_params):
            for i in range(n_message + 1):
                self.param_readout_layers[p].append(
                    self._make_layers(layer_nodes_readout, layer_activations)
                )

    def set_weights_excluding_guess(self, value=0.01):
        """Sets all weights and biases in the model to a specific value."""
        with torch.no_grad():
            for name, param in self.state_dict().items():
                if "guess_layer" not in name:
                    param.fill_(value)

    def _make_layers(self, layer_nodes, activations):
        layers = []
        for i in range(len(layer_nodes) - 1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
            # layers[-1].weight.data.normal_(1.0, 0.1)
            if activations[i] is not None:
                layers.append(activations[i])
        return nn.Sequential(*layers)

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
            "n_neuron": self.n_neuron,
            "n_embed": self.n_embed,
            "param_start_mean": self.param_start_mean,
            "param_start_std": self.param_start_std,
            "n_params": self.n_params,
        }

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
        # current_model_device = next(self.parameters()).device
        # model_device = next(self.atom_model.parameters()).device
        am_out = self.atom_model(batch)
        charge, dipole, qpole, h_list = (
            am_out[0],
            am_out[1],
            am_out[2],
            am_out[self.h_list_ind],
        )
        Z = x
        K_list = [self.guess_layer[p](Z) for p in range(self.n_params)]
        K = torch.cat(K_list, dim=-1)  # shape (n_atoms, n_params)
        # print(f"{K = }")
        atoms_with_edges = torch.cat([edge_index[0], edge_index[1]]).unique()
        keep_mask = torch.isin(
            torch.arange(len(molecule_ind), device=molecule_ind.device),
            atoms_with_edges,
        )
        if not keep_mask.any():
            return (
                charge.squeeze(-1),
                dipole,
                qpole,
                *am_out[3:],
                K.squeeze(-1) if self.n_params == 1 else K,
            )
        K_filtered = K[keep_mask]  # shape (n_atoms_filtered, n_params)
        for p in range(self.n_params):
            for i in range(self.n_message + 1):
                param_update = self.param_readout_layers[p][i](h_list[:, i, :])
                K_filtered[:, p] += param_update.squeeze(-1)
        # K[keep_mask] = torch.relu(K_filtered)  # + 1.00001
        K[keep_mask] = K_filtered  # + 1.00001
        # if K.isnan().any():
        #     print("K has NaN values, debugging info:")
        #     print(f"{K_filtered =}")
        #     print(f"{Z =}")
        #     print(f"{h_list=}")
        #     raise ValueError("K has NaN values")
        return (
            charge,
            dipole,
            qpole,
            *am_out[3:],
            K.squeeze(-1) if self.n_params == 1 else K,
        )


def get_distances(RA, RB, e_source, e_target):
    RA_source = RA.index_select(0, e_source)
    RB_target = RB.index_select(0, e_target)
    dR_xyz = RB_target - RA_source
    dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
    return dR, dR_xyz


# @torch.compile
def elst_damping_mtp_mtp_torch(
    alpha_i: torch.tensor,
    alpha_j: torch.tensor,
    r: torch.tensor,
    e_source: torch.tensor,
    e_target: torch.tensor,
):
    """
    Compute Gordon1-style damping factors for multipole–multipole interactions per edge.

    Parameters:
        alpha_i (torch.Tensor): Per-atom alpha values for the source ensemble (shape [N_atoms]).
        alpha_j (torch.Tensor): Per-atom alpha values for the target ensemble (shape [M_atoms]).
        r (torch.Tensor): Interatomic distances for each edge (shape [n_edges]).
        e_source (torch.Tensor): Source atom indices for each edge (shape [n_edges]).
        e_target (torch.Tensor): Target atom indices for each edge (shape [n_edges]).

    Returns:
        tuple: (lam1, lam3, lam5) — three torch.Tensors of damping factors for each edge (each shape [n_edges]).
    """
    # need to have alpha_i repeated for each atom in j and vice versa
    alpha_i = alpha_i.index_select(0, e_source)
    alpha_j = alpha_j.index_select(0, e_target)
    r2 = r**2
    r3 = r2 * r
    a1_2 = alpha_i * alpha_i
    a2_2 = alpha_j * alpha_j
    a1_3 = a1_2 * alpha_i
    lam1 = torch.ones_like(r)
    lam3 = torch.ones_like(r)
    lam5 = torch.ones_like(r)
    e1r = torch.exp(-1.0 * alpha_i * r)
    e2r = torch.exp(-1.0 * alpha_j * r)
    diff = torch.abs(alpha_i - alpha_j) > 1e-6
    # Add small epsilon to denominator to prevent NaN during backprop
    # (torch.where evaluates both branches, so division happens even when diff=False)
    eps = 1e-10
    denom = a2_2 - a1_2
    safe_denom = torch.where(torch.abs(denom) > eps, denom, torch.full_like(denom, eps))
    A = torch.where(diff, a2_2 / safe_denom, torch.zeros_like(r))
    B = torch.where(diff, a1_2 / (-safe_denom), torch.zeros_like(r))
    lam1 = torch.where(diff, 1 - A * e1r - B * e2r, 1 - (1.0 + 0.5 * alpha_i * r) * e1r)
    lam3 = torch.where(
        diff,
        1 - (1.0 + alpha_i * r) * A * e1r - (1.0 + alpha_j * r) * B * e2r,
        1 - (1.0 + alpha_i * r + 0.5 * a1_2 * r2) * e1r,
    )
    lam5 = torch.where(
        diff,
        1
        - (1.0 + alpha_i * r + (1.0 / 3.0) * a1_2 * r2) * A * e1r
        - (1.0 + alpha_j * r + (1.0 / 3.0) * a2_2 * r2) * B * e2r,
        1 - (1.0 + alpha_i * r + 0.5 * a1_2 * r2 + (1.0 / 6.0) * a1_3 * r3) * e1r,
    )
    return lam1, lam3, lam5


# @torch.compile
def elst_damping_Z_mtp_torch(
    alpha_i: torch.tensor,
    alpha_j: torch.tensor,
    r: torch.tensor,
    e_source: torch.tensor,
    e_target: torch.tensor,
):
    """
    Compute Gordon1-style damping factors for Z (nuclear charge) to multipole (MTP) interactions for each pair defined by edge indices.

    Parameters:
        alpha_i (torch.Tensor): Per-atom polarizabilities for atoms in set A (shape [n_atoms_A]).
        alpha_j (torch.Tensor): Per-atom polarizabilities for atoms in set B (shape [n_atoms_B]).
        r (torch.Tensor): Pairwise scalar distances for edges (shape [n_edges]).
        e_source (torch.Tensor): Source atom indices for each edge (maps entries in `r` to indices in `alpha_i`).
        e_target (torch.Tensor): Target atom indices for each edge (maps entries in `r` to indices in `alpha_j`).

    Returns:
        lam1_j (torch.Tensor): First-order damping factor for the j-side (shape [n_edges]).
        lam3_j (torch.Tensor): Third-order damping factor for the j-side (shape [n_edges]).
        lam5_j (torch.Tensor): Fifth-order damping factor for the j-side (shape [n_edges]).
        lam1_i (torch.Tensor): First-order damping factor for the i-side (shape [n_edges]).
        lam3_i (torch.Tensor): Third-order damping factor for the i-side (shape [n_edges]).
        lam5_i (torch.Tensor): Fifth-order damping factor for the i-side (shape [n_edges]).
    """
    # need to have alpha_i repeated for each atom in j and vice versa
    alpha_i = alpha_i.index_select(0, e_source)
    alpha_j = alpha_j.index_select(0, e_target)
    exp_i = torch.exp(-1.0 * torch.multiply(alpha_i, r))
    exp_j = torch.exp(-1.0 * torch.multiply(alpha_j, r))
    damp_i = torch.multiply(alpha_i, r)
    damp_j = torch.multiply(alpha_j, r)
    lam1_j = 1.0 - exp_j
    lam3_j = 1.0 - (1.0 + damp_j) * exp_j
    lam5_j = (
        1.0
        - (1.0 + damp_j + (1.0 / 3.0) * torch.multiply(torch.square(alpha_j), r**2))
        * exp_j
    )

    lam1_i = 1.0 - exp_i
    lam3_i = 1.0 - (1.0 + torch.multiply(alpha_i, r)) * exp_i
    lam5_i = (
        1.0
        - (1.0 + damp_i + (1.0 / 3.0) * torch.multiply(torch.square(alpha_i), r**2))
        * exp_i
    )
    return lam1_j, lam3_j, lam5_j, lam1_i, lam3_i, lam5_i


def elst_damping_AMOEBA_mtp_mtp_torch(
    alpha_i: torch.tensor,
    alpha_j: torch.tensor,
    r: torch.tensor,
    e_source: torch.tensor,
    e_target: torch.tensor,
):
    """
    Compute AMOEBA-style Gordon1 damping factors for multipole–multipole interactions.

    Computes per-edge damping scaling factors lam1, lam3, and lam5 for pairs of atomic sites using their effective damping parameters (alpha_i, alpha_j), inter-site distances r, and edge index mappings (e_source selects sites from alpha_i, e_target selects sites from alpha_j). Handles both same-alpha and different-alpha cases with numerical safeguards.

    Parameters:
        alpha_i (torch.Tensor): Per-atom damping parameter tensor for the "i" set.
        alpha_j (torch.Tensor): Per-atom damping parameter tensor for the "j" set.
        r (torch.Tensor): Interatomic distances for each edge (matches length of e_source/e_target).
        e_source (torch.Tensor): Index tensor selecting source atoms from alpha_i for each edge.
        e_target (torch.Tensor): Index tensor selecting target atoms from alpha_j for each edge.

    Returns:
        tuple: (lam1, lam3, lam5) tensors of the same shape as r containing the computed damping factors.
    """
    # need to have alpha_i repeated for each atom in j and vice versa
    alpha_i = alpha_i.index_select(0, e_source)
    alpha_j = alpha_j.index_select(0, e_target)

    # dampi = alpha_i * r, dampk = alpha_j * r
    damp_i = alpha_i * r
    damp_k = alpha_j * r
    damp_i2 = damp_i * damp_i
    damp_i3 = damp_i2 * damp_i
    damp_i4 = damp_i2 * damp_i2
    damp_i5 = damp_i2 * damp_i3
    damp_k2 = damp_k * damp_k
    damp_k3 = damp_k2 * damp_k

    exp_i = torch.exp(-damp_i)
    exp_k = torch.exp(-damp_k)

    a1_2 = alpha_i * alpha_i
    a2_2 = alpha_j * alpha_j

    diff = torch.abs(alpha_i - alpha_j) > 1e-3  # eps = 0.001 in Fortran

    # termi = alphak2 / (alphak2 - alphai2)
    # termk = alphai2 / (alphai2 - alphak2)
    # Add small epsilon to denominator to prevent NaN during backprop
    # (torch.where evaluates both branches, so division happens even when diff=False)
    eps = 1e-10
    denom = a2_2 - a1_2
    safe_denom = torch.where(torch.abs(denom) > eps, denom, torch.full_like(denom, eps))
    term_i = torch.where(diff, a2_2 / safe_denom, torch.zeros_like(r))
    term_k = torch.where(diff, a1_2 / (-safe_denom), torch.zeros_like(r))
    term_i2 = term_i * term_i
    term_k2 = term_k * term_k

    lam1_same = 1.0 - exp_i * (1 + 11 / 16 * damp_i + 3 / 16 * damp_i2 + damp_i3 / 48)
    lam1_diff = (
        1.0
        - exp_i
        * alpha_j**4
        / (alpha_i**2 - alpha_j**2) ** 2
        * (1.0 - 2.0 * alpha_i**2 / (alpha_j**2 - alpha_i**2) + 0.5 * damp_i)
        - exp_k
        * alpha_i**4
        / (alpha_j**2 - alpha_i**2) ** 2
        * (1.0 - 2.0 * alpha_j**2 / (alpha_i**2 - alpha_j**2) + 0.5 * damp_k)
        # - exp_i * term_i2 * (1.0 - 2.0 * term_k2 +  0.5 * damp_i)
        # - exp_k * term_k2 * (1.0 - 2.0 * term_i2 +  0.5 * damp_k)
    )
    lam1 = torch.where(diff, lam1_diff, lam1_same)

    lam3_same = (
        1.0
        - (1.0 + damp_i + 0.5 * damp_i2 + 7.0 * damp_i3 / 48.0 + damp_i4 / 48.0) * exp_i
    )
    # Different alpha case:
    lam3_diff = (
        1.0
        - term_i2 * (1.0 + damp_i + 0.5 * damp_i2) * exp_i
        - term_k2 * (1.0 + damp_k + 0.5 * damp_k2) * exp_k
        - 2.0 * term_i2 * term_k * (1.0 + damp_i) * exp_i
        - 2.0 * term_k2 * term_i * (1.0 + damp_k) * exp_k
    )
    lam3 = torch.where(diff, lam3_diff, lam3_same)

    # GORDON1 lam5 (dmpik(5))
    # Same alpha case:
    lam5_same = (
        1.0
        - (
            1.0
            + damp_i
            + 0.5 * damp_i2
            + damp_i3 / 6.0
            + damp_i4 / 24.0
            + damp_i5 / 144.0
        )
        * exp_i
    )
    # Different alpha case:
    lam5_diff = (
        1.0
        - term_i2 * (1.0 + damp_i + 0.5 * damp_i2 + damp_i3 / 6.0) * exp_i
        - term_k2 * (1.0 + damp_k + 0.5 * damp_k2 + damp_k3 / 6.0) * exp_k
        - 2.0 * term_i2 * term_k * (1.0 + damp_i + damp_i2 / 3.0) * exp_i
        - 2.0 * term_k2 * term_i * (1.0 + damp_k + damp_k2 / 3.0) * exp_k
    )
    lam5 = torch.where(diff, lam5_diff, lam5_same)

    return lam1, lam3, lam5


# @torch.compile
def elst_damping_AMOEBA_Z_mtp_torch(
    alpha_i: torch.tensor,
    alpha_j: torch.tensor,
    r: torch.tensor,
    e_source: torch.tensor,
    e_target: torch.tensor,
):
    """
    Compute AMOEBA-style Gordon1 damping factors for Z–MTP (core–valence) interactions per edge.

    Parameters:
        alpha_i (torch.Tensor): Per-atom alpha values for the "i" set (will be indexed by e_source).
        alpha_j (torch.Tensor): Per-atom alpha values for the "j" set (will be indexed by e_target).
        r (torch.Tensor): Distance scalar per edge (aligned with e_source/e_target).
        e_source (torch.Tensor): Edge source indices selecting entries from alpha_i.
        e_target (torch.Tensor): Edge target indices selecting entries from alpha_j.

    Returns:
        lam1_j, lam3_j, lam5_j, lam1_i, lam3_i, lam5_i (torch.Tensor):
            Damping factors of orders 1, 3, and 5 for the j-side followed by the i-side,
            each tensor aligned to the input edge list. If alpha_i and alpha_j differ by
            less than 1e-3 the j-side uses the same damping values as the i-side.
    """
    # need to have alpha_i repeated for each atom in j and vice versa
    alpha_i = alpha_i.index_select(0, e_source)
    alpha_j = alpha_j.index_select(0, e_target)

    # dampi = alpha_i * r, dampk = alpha_j * r
    damp_i = alpha_i * r
    damp_k = alpha_j * r
    damp_i2 = damp_i * damp_i
    damp_i3 = damp_i2 * damp_i
    damp_k2 = damp_k * damp_k
    damp_k3 = damp_k2 * damp_k

    exp_i = torch.exp(-damp_i)
    exp_k = torch.exp(-damp_k)

    diff = torch.abs(alpha_i - alpha_j) > 1e-3  # eps = 0.001 in Fortran

    # GORDON1 damping for alpha_i (dmpi)
    lam1_i = 1.0 - (1.0 + 0.5 * damp_i) * exp_i
    lam3_i = 1.0 - (1.0 + damp_i + 0.5 * damp_i2) * exp_i
    lam5_i = 1.0 - (1.0 + damp_i + 0.5 * damp_i2 + damp_i3 / 6.0) * exp_i

    # GORDON1 damping for alpha_j (dmpk)
    # Same alpha case: dmpk = dmpi
    lam1_j_same = lam1_i
    lam3_j_same = lam3_i
    lam5_j_same = lam5_i
    # Different alpha case: compute separately
    lam1_j_diff = 1.0 - (1.0 + 0.5 * damp_k) * exp_k
    lam3_j_diff = 1.0 - (1.0 + damp_k + 0.5 * damp_k2) * exp_k
    lam5_j_diff = 1.0 - (1.0 + damp_k + 0.5 * damp_k2 + damp_k3 / 6.0) * exp_k

    lam1_j = torch.where(diff, lam1_j_diff, lam1_j_same)
    lam3_j = torch.where(diff, lam3_j_diff, lam3_j_same)
    lam5_j = torch.where(diff, lam5_j_diff, lam5_j_same)

    return lam1_j, lam3_j, lam5_j, lam1_i, lam3_i, lam5_i


# @torch.compile
def mtp_elst(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    ZB,
    RB,
    qB,
    muB,
    quadB,
    e_AB_source,
    e_AB_target,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
):
    dR_ang, dR_xyz_ang = get_distances(RA, RB, e_AB_source, e_AB_target)
    dR = dR_ang / constants.au2ang
    dR_xyz = dR_xyz_ang / constants.au2ang
    oodR = 1.0 / dR
    delta = torch.eye(3, device=qA.device)

    ZA_q = ZA.index_select(0, e_AB_source)
    ZB_q = ZB.index_select(0, e_AB_target)
    qA -= ZA
    qB -= ZB

    # Identity for 3D
    delta = torch.eye(3, device=qA.device)

    # Extracting tensor elements
    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_source = qB.squeeze(-1).index_select(0, e_AB_target)

    muA_source = muA.index_select(0, e_AB_source)
    muB_source = muB.index_select(0, e_AB_target)

    # TF implementation uses 3/2 factor for quadrupoles
    # quadA_source = (3.0 / 2.0) * quadA.index_select(0, e_AB_source)
    # quadB_source = (3.0 / 2.0) * quadB.index_select(0, e_AB_target)
    quadA_source = quadA.index_select(0, e_AB_source)
    quadB_source = quadB.index_select(0, e_AB_target)

    E_qq = torch.einsum("x,x,x->x", qA_source, qB_source, oodR)

    T1 = torch.einsum("x,xy->xy", oodR**3, -1.0 * dR_xyz)
    qu = torch.einsum("x,xy->xy", qA_source, muB_source) - torch.einsum(
        "x,xy->xy", qB_source, muA_source
    )
    E_qu = torch.einsum("xy,xy->x", T1, qu)

    T2 = 3 * torch.einsum("xy,xz->xyz", dR_xyz, dR_xyz) - torch.einsum(
        "x,x,yz->xyz", dR, dR, delta
    )
    T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)

    E_uu = -1.0 * torch.einsum("xy,xz,xyz->x", muA_source, muB_source, T2)

    qA_quadB_source = torch.einsum("x,xyz->xyz", qA_source, quadB_source)
    qB_quadA_source = torch.einsum("x,xyz->xyz", qB_source, quadA_source)
    E_qQ = torch.einsum("xyz,xyz->x", T2, qA_quadB_source + qB_quadA_source) / Q_const

    # ZA-ZB
    E_ZA_ZB = torch.einsum("x,x,x->x", ZA_q, ZB_q, oodR)

    # TODO Z-M damping
    # ZA-MB
    E_ZA_qB = torch.einsum("x,x,x->x", ZA_q, qB_source, oodR)
    E_ZA_uB = torch.einsum("xy,x,xy->x", T1, ZA_q, muB_source)
    E_ZA_QB = torch.einsum("xyz,x,xyz->x", T2, ZA_q, quadB_source) / Q_const
    E_ZA_MB = E_ZA_qB + E_ZA_uB + E_ZA_QB
    # ZB-MA
    E_ZB_qA = torch.einsum("x,x,x->x", ZB_q, qA_source, oodR)
    E_ZB_uA = torch.einsum("xy,x,xy->x", -T1, ZB_q, muA_source)
    E_ZB_QA = torch.einsum("xyz,x,xyz->x", T2, ZB_q, quadA_source) / Q_const
    E_ZB_MA = E_ZB_qA + E_ZB_uA + E_ZB_QA

    E_elst = 627.509 * (E_qq + E_qu + E_qQ + E_uu + E_ZA_ZB + E_ZA_MB + E_ZB_MA)
    return E_elst


# @torch.compile
def mtp_elst_damping(
    ZA,
    RA,
    qA_0,
    muA,
    quadA,
    Ka,
    ZB,
    RB,
    qB_0,
    muB,
    quadB,
    Kb,
    e_AB_source,
    e_AB_target,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
):
    """
    Compute damped multipole electrostatic interactions for paired atoms using the Gordon2 (CLIFF) damping scheme.

    Parameters:
        ZA (Tensor): Nuclear charges for atoms in A.
        RA (Tensor): Cartesian coordinates for atoms in A (au or consistent internal units).
        qA_0 (Tensor): Monopole (formal) charges for atoms in A.
        muA (Tensor): Dipole vectors for atoms in A.
        quadA (Tensor): Quadrupole tensors for atoms in A.
        Ka (Tensor): Per-atom damping/size parameters for atoms in A used by the Gordon2 scheme.
        ZB (Tensor): Nuclear charges for atoms in B.
        RB (Tensor): Cartesian coordinates for atoms in B (same units as RA).
        qB_0 (Tensor): Monopole (formal) charges for atoms in B.
        muB (Tensor): Dipole vectors for atoms in B.
        quadB (Tensor): Quadrupole tensors for atoms in B.
        Kb (Tensor): Per-atom damping/size parameters for atoms in B used by the Gordon2 scheme.
        e_AB_source (LongTensor): Source atom indices into A for each A–B interacting pair.
        e_AB_target (LongTensor): Target atom indices into B for each A–B interacting pair.
        Q_const (float, optional): Scaling constant applied to quadrupole contributions (default 3.0).

    Returns:
        Tensor: Per-pair electrostatic interaction energies (one value per entry in the edge index arrays), scaled by the factor 627.509.
    """
    dR_ang, dR_xyz_ang = get_distances(RA, RB, e_AB_source, e_AB_target)
    dR = dR_ang / constants.au2ang
    dR_xyz = dR_xyz_ang / constants.au2ang
    oodR = 1.0 / dR
    delta = torch.eye(3, device=qA_0.device)

    lam1, lam3, lam5 = elst_damping_mtp_mtp_torch(Ka, Kb, dR, e_AB_source, e_AB_target)
    lam1_ZA_MB, lam3_ZA_MB, lam5_ZA_MB, lam1_ZB_MA, lam3_ZB_MA, lam5_ZB_MA = (
        elst_damping_Z_mtp_torch(Ka, Kb, dR, e_AB_source, e_AB_target)
    )
    # print(f"{Ka = }\n{Kb = }")
    # print(f"{lam1 = }\n{lam3 = }\n{lam5 = }")
    # print(f"{lam1_ZA_MB = }\n{lam3_ZA_MB = }\n{lam5_ZA_MB = }")
    # print(f"{lam1_ZB_MA = }\n{lam3_ZB_MA = }\n{lam5_ZB_MA = }")

    # Nuclear Charge Subtraction - pre-compute all index selections
    ZA_q = ZA.index_select(0, e_AB_source)
    ZB_q = ZB.index_select(0, e_AB_target)

    qA = qA_0 - ZA
    qB = qB_0 - ZB
    # Extracting tensor elements - pre-compute all selections
    qA_source = (
        qA.squeeze(-1).index_select(0, e_AB_source)
        if qA.dim() > 1
        else qA.index_select(0, e_AB_source)
    )
    qB_source = (
        qB.squeeze(-1).index_select(0, e_AB_target)
        if qB.dim() > 1
        else qB.index_select(0, e_AB_target)
    )
    muA_source = muA.index_select(0, e_AB_source)
    muB_source = muB.index_select(0, e_AB_target)
    quadA_source = quadA.index_select(0, e_AB_source)
    quadB_source = quadB.index_select(0, e_AB_target)

    E_qq = torch.einsum("x,x,x,x->x", qA_source, qB_source, oodR, lam1)

    T1 = torch.einsum("x,xy->xy", oodR**3, -1.0 * dR_xyz)
    qu = torch.einsum("x,xy->xy", qA_source, muB_source) - torch.einsum(
        "x,xy->xy", qB_source, muA_source
    )
    E_qu = torch.einsum("xy,xy,x->x", T1, qu, lam3)

    # Pre-compute common T2 components to avoid redundant calculations
    # dR_xyz[:, :, None] * dR_xyz[:, None, :]
    dR_outer = torch.einsum("xy,xz->xyz", dR_xyz, dR_xyz)
    dR_squared_delta = torch.einsum("x,x,yz->xyz", dR, dR, delta)

    # Main T2 for E_uu and E_qQ
    T2_main = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3
    )
    T2_main = torch.einsum("x,xyz->xyz", oodR**5, T2_main)

    E_uu = -1.0 * torch.einsum("xy,xz,xyz->x", muA_source, muB_source, T2_main)

    qA_quadB_source = torch.einsum("x,xyz->xyz", qA_source, quadB_source)
    qB_quadA_source = torch.einsum("x,xyz->xyz", qB_source, quadA_source)
    E_qQ = (
        torch.einsum("xyz,xyz->x", T2_main, qA_quadB_source + qB_quadA_source) / Q_const
    )

    # ZA-ZB
    E_ZA_ZB = torch.einsum("x,x,x->x", ZA_q, ZB_q, oodR)

    # ZA-MB - reuse T1, compute specialized T2
    E_ZA_MB = torch.einsum("x,x,x,x->x", ZA_q, qB_source, oodR, lam1_ZA_MB)
    E_ZA_MB += torch.einsum("xy,x,x,xy->x", T1, lam3_ZA_MB, ZA_q, muB_source)
    T2_ZA_MB = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5_ZA_MB) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3_ZA_MB
    )
    T2_ZA_MB = torch.einsum("x,xyz->xyz", oodR**5, T2_ZA_MB)
    E_ZA_MB += torch.einsum("xyz,x,xyz->x", T2_ZA_MB, ZA_q, quadB_source) / Q_const

    # ZB-MA - reuse T1, compute specialized T2
    T2_ZB_MA = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5_ZB_MA) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3_ZB_MA
    )
    T2_ZB_MA = torch.einsum("x,xyz->xyz", oodR**5, T2_ZB_MA)
    E_ZB_MA = torch.einsum("x,x,x,x->x", ZB_q, qA_source, oodR, lam1_ZB_MA)
    E_ZB_MA += torch.einsum("xy,x,x,xy->x", -T1, lam3_ZB_MA, ZB_q, muA_source)
    E_ZB_MA += torch.einsum("xyz,x,xyz->x", T2_ZB_MA, ZB_q, quadA_source) / Q_const
    E_elst = 627.509 * (E_qq + E_qu + E_qQ + E_uu + E_ZA_ZB + E_ZA_MB + E_ZB_MA)
    return E_elst


def mtp_elst_damping_AMOEBA(
    ZA,
    RA,
    qA_0,
    muA,
    quadA,
    Ka,
    ZB,
    RB,
    qB_0,
    muB,
    quadB,
    Kb,
    e_AB_source,
    e_AB_target,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
):
    """
    Compute the AMOEBA-style damped electrostatic interaction energy between multipole-expanded atoms for each A-B edge.

    Parameters:
        ZA (Tensor): Nuclear charges for atoms in A, shape (nA, 1) or (nA,).
        RA (Tensor): Coordinates for atoms in A, shape (nA, 3) (atomic units).
        qA_0 (Tensor): Total monopoles for atoms in A (including nuclear), shape (nA, 1) or (nA,).
        muA (Tensor): Dipole vectors for atoms in A, shape (nA, 3).
        quadA (Tensor): Quadrupole tensors for atoms in A, shape (nA, 3, 3).
        Ka (Tensor): Per-atom damping/alpha-like parameters for atoms in A, shape (nA, ...) as required by damping helpers.
        ZB (Tensor): Nuclear charges for atoms in B, shape (nB, 1) or (nB,).
        RB (Tensor): Coordinates for atoms in B, shape (nB, 3) (atomic units).
        qB_0 (Tensor): Total monopoles for atoms in B (including nuclear), shape (nB, 1) or (nB,).
        muB (Tensor): Dipole vectors for atoms in B, shape (nB, 3).
        quadB (Tensor): Quadrupole tensors for atoms in B, shape (nB, 3, 3).
        Kb (Tensor): Per-atom damping/alpha-like parameters for atoms in B, shape (nB, ...) as required by damping helpers.
        e_AB_source (LongTensor): Source atom indices from A for each A-B interaction edge, shape (n_edges,).
        e_AB_target (LongTensor): Target atom indices from B for each A-B interaction edge, shape (n_edges,).
        Q_const (float, optional): Quadrupole scaling constant; default 3.0 (set to 1.0 to match CLIFF/Gordon2 convention).

    Returns:
        Tensor: Per-edge electrostatic interaction energies in kcal/mol, shape (n_edges,).
    """
    dR_ang, dR_xyz_ang = get_distances(RA, RB, e_AB_source, e_AB_target)
    dR = dR_ang / constants.au2ang
    dR_xyz = dR_xyz_ang / constants.au2ang
    oodR = 1.0 / dR
    delta = torch.eye(3, device=qA_0.device)

    lam1, lam3, lam5 = elst_damping_AMOEBA_mtp_mtp_torch(
        Ka, Kb, dR, e_AB_source, e_AB_target
    )
    lam1_ZA_MB, lam3_ZA_MB, lam5_ZA_MB, lam1_ZB_MA, lam3_ZB_MA, lam5_ZB_MA = (
        elst_damping_AMOEBA_Z_mtp_torch(Ka, Kb, dR, e_AB_source, e_AB_target)
    )
    # print(f"{Ka = }\n{Kb = }")
    # print(f"{lam1 = }\n")
    # print(f"{lam1 = }\n{lam3 = }\n{lam5 = }")
    # print(f"{lam1_ZA_MB = }\n{lam3_ZA_MB = }\n{lam5_ZA_MB = }")
    # print(f"{lam1_ZB_MA = }\n{lam3_ZB_MA = }\n{lam5_ZB_MA = }")

    # Nuclear Charge Subtraction - pre-compute all index selections
    ZA_q = ZA.index_select(0, e_AB_source)
    ZB_q = ZB.index_select(0, e_AB_target)

    qA = qA_0 - ZA
    qB = qB_0 - ZB
    # Extracting tensor elements - pre-compute all selections
    qA_source = (
        qA.squeeze(-1).index_select(0, e_AB_source)
        if qA.dim() > 1
        else qA.index_select(0, e_AB_source)
    )
    qB_source = (
        qB.squeeze(-1).index_select(0, e_AB_target)
        if qB.dim() > 1
        else qB.index_select(0, e_AB_target)
    )
    muA_source = muA.index_select(0, e_AB_source)
    muB_source = muB.index_select(0, e_AB_target)
    quadA_source = quadA.index_select(0, e_AB_source)
    quadB_source = quadB.index_select(0, e_AB_target)

    E_qq = torch.einsum("x,x,x,x->x", qA_source, qB_source, oodR, lam1)

    T1 = torch.einsum("x,xy->xy", oodR**3, -1.0 * dR_xyz)
    qu = torch.einsum("x,xy->xy", qA_source, muB_source) - torch.einsum(
        "x,xy->xy", qB_source, muA_source
    )
    E_qu = torch.einsum("xy,xy,x->x", T1, qu, lam3)

    # Pre-compute common T2 components to avoid redundant calculations
    # dR_xyz[:, :, None] * dR_xyz[:, None, :]
    dR_outer = torch.einsum("xy,xz->xyz", dR_xyz, dR_xyz)
    dR_squared_delta = torch.einsum("x,x,yz->xyz", dR, dR, delta)

    # Main T2 for E_uu and E_qQ
    T2_main = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3
    )
    T2_main = torch.einsum("x,xyz->xyz", oodR**5, T2_main)

    E_uu = -1.0 * torch.einsum("xy,xz,xyz->x", muA_source, muB_source, T2_main)

    qA_quadB_source = torch.einsum("x,xyz->xyz", qA_source, quadB_source)
    qB_quadA_source = torch.einsum("x,xyz->xyz", qB_source, quadA_source)
    E_qQ = (
        torch.einsum("xyz,xyz->x", T2_main, qA_quadB_source + qB_quadA_source) / Q_const
    )

    # ZA-ZB
    E_ZA_ZB = torch.einsum("x,x,x->x", ZA_q, ZB_q, oodR)

    # ZA-MB - reuse T1, compute specialized T2
    E_ZA_MB = torch.einsum("x,x,x,x->x", ZA_q, qB_source, oodR, lam1_ZA_MB)
    E_ZA_MB += torch.einsum("xy,x,x,xy->x", T1, lam3_ZA_MB, ZA_q, muB_source)
    T2_ZA_MB = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5_ZA_MB) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3_ZA_MB
    )
    T2_ZA_MB = torch.einsum("x,xyz->xyz", oodR**5, T2_ZA_MB)
    E_ZA_MB += torch.einsum("xyz,x,xyz->x", T2_ZA_MB, ZA_q, quadB_source) / Q_const

    # ZB-MA - reuse T1, compute specialized T2
    T2_ZB_MA = 3 * torch.einsum("xyz,x->xyz", dR_outer, lam5_ZB_MA) - torch.einsum(
        "xyz,x->xyz", dR_squared_delta, lam3_ZB_MA
    )
    T2_ZB_MA = torch.einsum("x,xyz->xyz", oodR**5, T2_ZB_MA)
    E_ZB_MA = torch.einsum("x,x,x,x->x", ZB_q, qA_source, oodR, lam1_ZB_MA)
    E_ZB_MA += torch.einsum("xy,x,x,xy->x", -T1, lam3_ZB_MA, ZB_q, muA_source)
    E_ZB_MA += torch.einsum("xyz,x,xyz->x", T2_ZB_MA, ZB_q, quadA_source) / Q_const
    E_elst = 627.509 * (E_qq + E_qu + E_qQ + E_uu + E_ZA_ZB + E_ZA_MB + E_ZB_MA)
    return E_elst


# @torch.compile
def distance_tensors(
    Ri, Rj, e_source, e_target, alpha_A=None, alpha_B=None, thole_damping_param=0.39
):
    """
    Compute Thole-damped distance and interaction tensors for pairs of atoms.

    Parameters:
        Ri (Tensor): Coordinates of atom set A with shape (N_A, 3).
        Rj (Tensor): Coordinates of atom set B with shape (N_B, 3).
        e_source (LongTensor): Source indices into Ri for each interacting pair.
        e_target (LongTensor): Target indices into Rj for each interacting pair.
        alpha_A (Tensor): Per-atom polarizabilities for Ri (shape (N_A,) or (N_A,1)).
        alpha_B (Tensor): Per-atom polarizabilities for Rj (shape (N_B,) or (N_B,1)).
        thole_damping_param (float): Thole damping parameter controlling short-range screening (default 0.39).

    Returns:
        dR (Tensor): Pairwise scalar distances for each edge (in atomic units).
        dR_xyz (Tensor): Pairwise displacement vectors for each edge (in atomic units), shape (E,3).
        oodR (Tensor): Elementwise inverse of dR (1 / dR).
        T1 (Tensor): Thole-damped rank-2 interaction tensor components used for dipole interactions (shape (E,3,3) or broadcastable).
        T2 (Tensor): Thole-damped rank-3 interaction tensor components used for higher-order interactions (shape (E,3,3) or broadcastable).
    """
    dR_ang, dR_xyz_ang = get_distances(Ri, Rj, e_source, e_target)
    dR_xyz = dR_xyz_ang / constants.au2ang
    dR = dR_ang / constants.au2ang
    alpha_i = alpha_A.index_select(0, e_source)
    alpha_j = alpha_B.index_select(0, e_target)
    u = dR / ((alpha_i * alpha_j) ** (1.0 / 6.0))
    au3 = thole_damping_param * (u**3)
    lam_3 = 1 - torch.exp(-au3)
    lam_5 = 1 - (1 + au3) * torch.exp(-au3)
    delta = torch.eye(3, device=dR.device)
    oodR = 1.0 / dR
    T1 = torch.einsum("x,xy,x->xy", oodR**3, -1.0 * dR_xyz, lam_3)
    T2 = 3 * torch.einsum("xy,xz,x->xyz", dR_xyz, dR_xyz, lam_5) - torch.einsum(
        "x,x,yz,x->xyz", dR, dR, delta, lam_3
    )
    T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)
    return dR, dR_xyz, oodR, T1, T2


# @torch.compile
def induced_dipole_induction(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    ZB,
    RB,
    qB,
    muB,
    quadB,
    e_AB_source,
    e_AB_target,
    e_AA_source,
    e_BB_source,
    e_AA_target,
    e_BB_target,
    hirshfeld_volume_ratio_A: torch.tensor,
    hirshfeld_volume_ratio_B: torch.tensor,
    valence_widths_A: torch.tensor,
    valence_widths_B: torch.tensor,
    Ka: torch.tensor,
    Kb: torch.tensor,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param: float = 0.39,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
    polarizability_table=constants.polarizability_table,
) -> float:
    """
    Compute per-edge induced-dipole induction energies for a dimer using Hirshfeld-scaled atomic polarizabilities and Thole damping.

    Per-edge energies include mutual induction between atoms in molecule A and B, computed via a self-consistent field (SCF) on induced dipoles with optional Thole damping and an exponential overlap correction. The function returns one energy value per A–B interaction edge (indexed by e_AB_source / e_AB_target) in kilocalories per mole.

    Parameters:
        ZA (Tensor): Atomic numbers for molecule A.
        RA (Tensor): Coordinates for molecule A (N_A x 3).
        qA (Tensor): Monopoles for A (N_A).
        muA (Tensor): Permanent dipoles for A (N_A x 3).
        quadA (Tensor): Quadrupoles for A (unused by induction here but kept for API parity).
        ZB, RB, qB, muB, quadB: Same as above for molecule B.
        e_AB_source, e_AB_target (LongTensor): Per-edge source/target atom indices mapping A->B for intermolecular edges.
        e_AA_source, e_AA_target, e_BB_source, e_BB_target (LongTensor): Index tensors for intra-molecular interaction edges used in SCF.
        hirshfeld_volume_ratio_A, hirshfeld_volume_ratio_B (Tensor): Per-atom Hirshfeld volume ratios used to scale free-atom polarizabilities.
        valence_widths_A, valence_widths_B (Tensor): Per-atom valence-width parameters used for the exponential overlap correction.
        Ka, Kb (Tensor): Per-atom prefactors used in the overlap correction term for A and B.
        max_iterations (int): Maximum SCF iterations to converge induced dipoles.
        convergence_threshold (float): L2-norm threshold for SCF convergence.
        omega (float): SCF mixing parameter in (0,1] applied to induced-dipole updates.
        thole_damping_param (float): Thole damping parameter controlling short-range screening.
        Q_const (float): Multiplicative constant for electrostatic prefactors (keeps internal scaling; default chosen for unit conventions).
        polarizability_table (Tensor): Lookup table of free-atom isotropic polarizabilities indexed by atomic number.

    Returns:
        Tensor: Per-interaction induced induction energies (kcal/mol) for each A–B edge (shape equals number of entries in e_AB_source).
    """
    delta = torch.eye(3, device=qA.device)
    h2kcalmol = constants.h2kcalmol  # Hartree to kcal/mol conversion factor

    alpha_0_A = torch.zeros_like(hirshfeld_volume_ratio_A)
    alpha_0_B = torch.zeros_like(hirshfeld_volume_ratio_B)

    # Use index_select for vectorized lookup
    alpha_0_A = torch.index_select(polarizability_table, 0, ZA.long())
    alpha_0_B = torch.index_select(polarizability_table, 0, ZB.long())
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A ** (4 / 3.0)
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B ** (4 / 3.0)

    # Calculate interaction tensors between atoms
    dR_AB, dR_AB_xyz, T0_AB, T1_AB, T2_AB = distance_tensors(
        RA, RB, e_AB_source, e_AB_target, alpha_A, alpha_B, thole_damping_param
    )
    dR_AA, dR_AA_xyz, T0_AA, T1_AA, T2_AA = distance_tensors(
        RA, RA, e_AA_source, e_AA_target, alpha_A, alpha_A, thole_damping_param
    )
    dR_BB, dR_BB_xyz, T0_BB, T1_BB, T2_BB = distance_tensors(
        RB, RB, e_BB_source, e_BB_target, alpha_B, alpha_B, thole_damping_param
    )

    # TODO PASS DAMPING PARAM;
    # Select relevant tensors for atom pairs
    alpha_A_source = alpha_A.index_select(0, e_AB_source)
    alpha_B_target = alpha_B.index_select(0, e_AB_target)

    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    alpha_BB_target = alpha_B.index_select(0, e_BB_target)

    # Need to ensure that qA and qB are right shape even when ions
    qA = qA.reshape(-1, 1)
    qB = qB.reshape(-1, 1)
    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_target = qB.squeeze(-1).index_select(0, e_AB_target)

    muA_source = muA.index_select(0, e_AB_source)
    muB_target = muB.index_select(0, e_AB_target)

    # Initialize tensors for induced dipoles
    n_atoms_A = RA.shape[0]
    n_atoms_B = RB.shape[0]

    K_A_source = Ka.index_select(0, e_AB_source)
    K_B_target = Kb.index_select(0, e_AB_target)
    # Must have sigma be > 0 to avoid NaNs
    sigma_A_source = valence_widths_A.index_select(0, e_AB_source)
    sigma_B_target = valence_widths_B.index_select(0, e_AB_target)
    B_ij = torch.sqrt(1.0 / (sigma_A_source * sigma_B_target))
    S_ij = (1.0 / 3.0 * (B_ij * dR_AB) ** 2 + B_ij * dR_AB + 1.0) * torch.exp(
        -B_ij * dR_AB
    )
    E_ind_overlap = K_A_source * S_ij * K_B_target * h2kcalmol

    # Calculate initial induced dipoles
    # A: Induced by B's multipoles
    mu_induced_0_A = torch.zeros((n_atoms_A, 3), device=qA.device)
    mu_induced_0_B = torch.zeros((n_atoms_B, 3), device=qB.device)

    # Calculate initial induced dipoles from molecule B's multipoles on molecule A
    # Contribution from charges
    mu_charge_A = torch.einsum("a,ai,a->ai", alpha_A_source, T1_AB, qB_target)
    mu_induced_0_A = scatter_sum_compile(mu_charge_A, e_AB_source, n_atoms_A)
    mu_dipole_A = torch.einsum("a,aij,aj->ai", alpha_A_source, T2_AB, muB_target)
    mu_induced_0_A += scatter_sum_compile(mu_dipole_A, e_AB_source, n_atoms_A)

    mu_charge_B = torch.einsum("a,ai,a->ai", alpha_B_target, -T1_AB, qA_source)
    mu_induced_0_B = scatter_sum_compile(mu_charge_B, e_AB_target, n_atoms_B)
    mu_dipole_B = torch.einsum("a,aij,aj->ai", alpha_B_target, T2_AB, muA_source)
    mu_induced_0_B += scatter_sum_compile(mu_dipole_B, e_AB_target, n_atoms_B)

    # Self-consistent induced dipole iterations
    mu_induced_A = mu_induced_0_A.clone()
    mu_induced_B = mu_induced_0_B.clone()

    # Pre-compute index selections to avoid repeated operations in the loop
    mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
    mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
    mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
    mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

    # Iterative SCF procedure to converge induced dipoles
    for iteration in range(max_iterations):
        mu_induced_A_old = mu_induced_A.clone()
        mu_induced_B_old = mu_induced_B.clone()

        ####### (A) INDUCED DIPOLES ########
        # Induced dipoles on A due to induced dipoles on B
        mu_induced_A_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_A_source, T2_AB, mu_induced_B_at_AB_target
        )
        mu_induced_A_new = scatter_sum_compile(
            mu_induced_A_due_B, e_AB_source, dim_size=n_atoms_A
        )
        # Induced dipoles on A due to induced dipoles on A
        mu_induced_A_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_AA_target, T2_AA, mu_induced_A_at_AA_source
        )
        mu_induced_A_new += scatter_sum_compile(
            mu_induced_A_due_A, e_AA_target, dim_size=n_atoms_A
        )
        mu_induced_A_new += mu_induced_0_A

        ####### (B) INDUCED DIPOLES ########
        # Induced dipoles on B due to induced dipoles on A
        mu_induced_B_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_B_target, T2_AB, mu_induced_A_at_AB_source
        )
        mu_induced_B_new = scatter_sum_compile(
            mu_induced_B_due_A, e_AB_target, dim_size=n_atoms_B
        )
        # Induced dipoles on B due to induced dipoles on B
        mu_induced_B_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_BB_target, T2_BB, mu_induced_B_at_BB_source
        )
        mu_induced_B_new += scatter_sum_compile(
            mu_induced_B_due_B, e_BB_target, dim_size=n_atoms_B
        )
        mu_induced_B_new += mu_induced_0_B

        # Apply mixing
        mu_induced_A = (1 - omega) * mu_induced_A_old + omega * mu_induced_A_new
        mu_induced_B = (1 - omega) * mu_induced_B_old + omega * mu_induced_B_new

        # Update pre-computed index selections for next iteration
        mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
        mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
        mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
        mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

        # Check convergence
        delta_A = torch.norm(mu_induced_A - mu_induced_A_old)
        delta_B = torch.norm(mu_induced_B - mu_induced_B_old)
        delta = max(delta_A, delta_B)
        if delta < convergence_threshold:
            break
    muA_induced_source = mu_induced_A.index_select(0, e_AB_source)
    muB_induced_target = mu_induced_B.index_select(0, e_AB_target)
    qu = torch.einsum("x,xy->xy", qA_source, muB_induced_target) - torch.einsum(
        "x,xy->xy", qB_target, muA_induced_source
    )
    E_qu = torch.einsum("xy,xy->x", T1_AB, qu) * h2kcalmol
    E_uu = (
        -1.0
        * (
            torch.einsum("xy,xz,xyz->x", muA_induced_source, muB_target, T2_AB)
            + torch.einsum("xy,xz,xyz->x", muA_source, muB_induced_target, T2_AB)
        )
        * h2kcalmol
    )
    E_ind = (E_qu + E_uu) / 2.0
    E_ind -= E_ind_overlap
    return E_ind


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
    valence_widths: torch.Tensor = None,
    atom_polarizabilities: torch.Tensor = None,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param_mutual: float = 0.39,
    thole_damping_param_direct: float = 0.34,
    screening: bool = True,
    screening_distance: float = 1.8,
    compute_energies: bool = False,
    verbose: int = 0,
) -> tuple:
    """
    Calculate intramolecular induced dipoles for a single molecule using
    its multipole moments and Hirshfeld volume ratios. This is the PyTorch
    version of the intramolecular_induced_dipole function, following the
    classical induction model from CLIFF paper.

    Reference: https://pubs.aip.org/aip/jcp/article/154/18/184110/200216/CLIFF-A-component-based-machine-learned

    Parameters
    ----------
    Z : torch.Tensor
        Atomic numbers (n_atoms,)
    R : torch.Tensor
        Atomic positions in Bohr (n_atoms, 3)
    q : torch.Tensor
        Atomic charges (n_atoms, 1) or (n_atoms,)
    mu : torch.Tensor
        Atomic dipole moments (n_atoms, 3)
    quad : torch.Tensor
        Atomic quadrupole moments (n_atoms, 3, 3)
    e_source : torch.Tensor
        Source atom indices for intramolecular pairs
    e_target : torch.Tensor
        Target atom indices for intramolecular pairs
    hirshfeld_volume_ratio : torch.Tensor
        Hirshfeld volume ratios for polarizability scaling (n_atoms,)
    valence_widths : torch.Tensor, optional
        Valence widths for each atom (n_atoms,)
    atom_polarizabilities : torch.Tensor, optional
        Explicit atomic polarizabilities. If None, calculated from Hirshfeld ratios
    max_iterations : int
        Maximum number of SCF iterations (default: 200)
    convergence_threshold : float
        Convergence threshold for induced dipoles (default: 1e-8)
    omega : float
        Damping parameter for SCF convergence (default: 0.7, recommended)
    thole_damping_param_mutual : float
        Thole damping parameter for induced-induced interactions (default: 0.39)
    thole_damping_param_direct : float
        Thole damping parameter for permanent-induced interactions (default: 0.34)
    screening : bool
        Enable distance-based screening for 1-2, 1-3 interactions (default: True)
    screening_distance : float
        Distance threshold in Angstroms for screening (default: 1.8)
    compute_energies : bool
        If True, compute and return intramolecular induction energy (default: False)
    verbose : int
        Verbosity level: 0=quiet, 1=basic, 2=detailed (default: 0)

    Returns
    -------
    tuple
        (charges, induced_dipoles, quadrupoles) or
        (charges, induced_dipoles, quadrupoles, energy) if compute_energies=True
        - charges: original charges (n_atoms,)
        - induced_dipoles: converged induced dipole moments (n_atoms, 3)
        - quadrupoles: original quadrupoles (n_atoms, 3, 3)
        - energy (optional): intramolecular induction energy in kcal/mol
    """
    # Calculate atomic polarizabilities
    alpha_0 = torch.index_select(self.polarizability_table, 0, Z.long())
    alpha = alpha_0 * hirshfeld_volume_ratio ** (4 / 3.0)

    # Define helper function to calculate distance tensors with Thole damping
    def distance_tensors(
        Ri,
        Rj,
        e_source,
        e_target,
        alpha_i,
        alpha_j,
        thole_param,
        thole_type="direct",
    ):
        """
        Compute Thole-damped interaction tensors and distance measures between two sets of atoms for a list of pair indices.

        Parameters:
            Ri (Tensor): Coordinates of source atoms (units: atomic units).
            Rj (Tensor): Coordinates of target atoms (units: atomic units).
            e_source (LongTensor): 1D indices selecting source atoms for each pair.
            e_target (LongTensor): 1D indices selecting target atoms for each pair.
            alpha_i (Tensor): Per-atom polarizabilities for source atoms.
            alpha_j (Tensor): Per-atom polarizabilities for target atoms.
            thole_param (float or Tensor): Thole damping parameter (scalar or per-pair).
            thole_type (str): Either "direct" or "mutual", selects the Thole damping variant.

        Returns:
            dR (Tensor): Pairwise scalar distances (Angstrom).
            dR_xyz (Tensor): Pairwise displacement vectors (Angstrom) from source to target.
            oodR (Tensor): 1.0 / dR (inverse distances).
            T1 (Tensor): Rank-1 interaction tensor (field) for each pair, Thole-damped.
            T2 (Tensor): Rank-2 interaction tensor (field gradient) for each pair, Thole-damped.

        Notes:
            - Distances and displacement vectors are converted from atomic units to Angstrom.
            - Short-range pairs below the configured screening distance are replaced with safe values and their damping factors set to zero to avoid singularities.
            - `thole_type="mutual"` applies mutual Thole damping; otherwise direct Thole damping is used.
        """
        dR_ang, dR_xyz_ang = get_distances(Ri, Rj, e_source, e_target)
        dR_xyz = dR_xyz_ang / constants.au2ang
        dR = dR_ang / constants.au2ang

        alpha_source = alpha_i.index_select(0, e_source)
        alpha_target = alpha_j.index_select(0, e_target)

        # Apply Thole damping and screening
        if thole_type == "mutual":
            au3, lam_3, lam_5 = thole_damping_mutual_torch(
                dR, alpha_source, alpha_target, thole_param
            )
        else:
            au3, lam_3, lam_5 = thole_damping_direct_torch(
                dR, alpha_source, alpha_target, thole_param
            )

        # Apply distance-based screening for direct interactions (excluding 1-2, 1-3 type terms)
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
        thole_type="direct",
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
        thole_type="mutual",
    )

    # Initialize induced dipoles
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


@torch.compile
def induced_dipole_induction_optimized(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    ZB,
    RB,
    qB,
    muB,
    quadB,
    e_AB_source,
    e_AB_target,
    e_AA_source,
    e_BB_source,
    e_AA_target,
    e_BB_target,
    hirshfeld_volume_ratio_A: torch.tensor,
    hirshfeld_volume_ratio_B: torch.tensor,
    valence_widths_A: torch.tensor,
    valence_widths_B: torch.tensor,
    Ka: torch.tensor,
    Kb: torch.tensor,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param: float = 0.39,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
    polarizability_table=constants.polarizability_table,
) -> float:
    """
    Compute per-pair induced-dipole induction energies for a dimer using an optimized SCF procedure.

    Parameters:
        ZA (Tensor): atomic numbers for molecule A (shape [n_A]).
        RA (Tensor): Cartesian coordinates for molecule A (shape [n_A, 3]).
        qA (Tensor): monopoles for A (shape [n_A] or [n_A,1]).
        muA (Tensor): permanent dipoles for A (shape [n_A, 3]).
        quadA (Tensor): quadrupoles for A (shape [n_A, ...]) — used by the interaction tensors.
        ZB (Tensor): atomic numbers for molecule B (shape [n_B]).
        RB (Tensor): Cartesian coordinates for molecule B (shape [n_B, 3]).
        qB (Tensor): monopoles for B (shape [n_B] or [n_B,1]).
        muB (Tensor): permanent dipoles for B (shape [n_B, 3]).
        quadB (Tensor): quadrupoles for B (shape [n_B, ...]) — used by the interaction tensors.
        e_AB_source (LongTensor): source indices into A for A↔B pair list (shape [n_pairs]).
        e_AB_target (LongTensor): target indices into B for A↔B pair list (shape [n_pairs]).
        e_AA_source (LongTensor): source indices into A for intramolecular A–A interactions.
        e_BB_source (LongTensor): source indices into B for intramolecular B–B interactions.
        e_AA_target (LongTensor): target indices into A for intramolecular A–A interactions.
        e_BB_target (LongTensor): target indices into B for intramolecular B–B interactions.
        hirshfeld_volume_ratio_A (Tensor): Hirshfeld volume ratios for A (shape [n_A]).
        hirshfeld_volume_ratio_B (Tensor): Hirshfeld volume ratios for B (shape [n_B]).
        valence_widths_A (Tensor): valence width parameters for A (shape [n_A]).
        valence_widths_B (Tensor): valence width parameters for B (shape [n_B]).
        Ka (Tensor): per-atom short-range correction amplitudes for A (shape [n_A]).
        Kb (Tensor): per-atom short-range correction amplitudes for B (shape [n_B]).
        max_iterations (int): maximum SCF iterations (default 200).
        convergence_threshold (float): SCF convergence threshold on induced-dipole change (default 1e-8).
        omega (float): DIIS-like mixing factor applied each iteration (default 0.7).
        thole_damping_param (float): Thole damping parameter for interaction tensors (default 0.39).
        Q_const (float): scaling constant applied in tensor construction (default 3.0).
        polarizability_table (Tensor): lookup table of free-atom polarizabilities indexed by atomic number.

    Returns:
        Tensor: per-pair induced induction energy (kcal/mol) for each A–B pair in the order given by e_AB_source/e_AB_target.
    """

    delta = torch.eye(3, device=qA.device)
    h2kcalmol = constants.h2kcalmol  # Hartree to kcal/mol conversion factor

    alpha_0_A = torch.zeros_like(hirshfeld_volume_ratio_A)
    alpha_0_B = torch.zeros_like(hirshfeld_volume_ratio_B)

    # Use index_select for vectorized lookup
    alpha_0_A = torch.index_select(polarizability_table, 0, ZA.long())
    alpha_0_B = torch.index_select(polarizability_table, 0, ZB.long())
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A ** (4 / 3.0)
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B ** (4 / 3.0)

    # Calculate interaction tensors between atoms
    dR_AB, dR_AB_xyz, T0_AB, T1_AB, T2_AB = distance_tensors(
        RA, RB, e_AB_source, e_AB_target, alpha_A, alpha_B, thole_damping_param
    )
    dR_AA, dR_AA_xyz, T0_AA, T1_AA, T2_AA = distance_tensors(
        RA, RA, e_AA_source, e_AA_target, alpha_A, alpha_A, thole_damping_param
    )
    dR_BB, dR_BB_xyz, T0_BB, T1_BB, T2_BB = distance_tensors(
        RB, RB, e_BB_source, e_BB_target, alpha_B, alpha_B, thole_damping_param
    )

    # Select relevant tensors for atom pairs
    alpha_A_source = alpha_A.index_select(0, e_AB_source)
    alpha_B_target = alpha_B.index_select(0, e_AB_target)

    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    alpha_BB_target = alpha_B.index_select(0, e_BB_target)

    # Need to ensure that qA and qB are right shape even when ions
    qA = qA.reshape(-1, 1)
    qB = qB.reshape(-1, 1)
    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_target = qB.squeeze(-1).index_select(0, e_AB_target)

    muA_source = muA.index_select(0, e_AB_source)
    muB_target = muB.index_select(0, e_AB_target)

    # Initialize tensors for induced dipoles
    n_atoms_A = RA.shape[0]
    n_atoms_B = RB.shape[0]

    K_A_source = Ka.index_select(0, e_AB_source)
    K_B_target = Kb.index_select(0, e_AB_target)
    sigma_A_source = valence_widths_A.index_select(0, e_AB_source)
    sigma_B_target = valence_widths_B.index_select(0, e_AB_target)
    B_ij = torch.sqrt(1.0 / (sigma_A_source * sigma_B_target))
    S_ij = (1.0 / 3.0 * (B_ij * dR_AB) ** 2 + B_ij * dR_AB + 1.0) * torch.exp(
        -B_ij * dR_AB
    )
    E_ind_overlap = K_A_source * S_ij * K_B_target * h2kcalmol

    # Calculate initial induced dipoles
    mu_induced_0_A = torch.zeros((n_atoms_A, 3), device=qA.device)
    mu_induced_0_B = torch.zeros((n_atoms_B, 3), device=qB.device)

    # Calculate initial induced dipoles from molecule B's multipoles on molecule A
    mu_charge_A = torch.einsum("a,ai,a->ai", alpha_A_source, T1_AB, qB_target)
    mu_induced_0_A = scatter_sum_compile(mu_charge_A, e_AB_source, dim_size=n_atoms_A)
    mu_dipole_A = torch.einsum("a,aij,aj->ai", alpha_A_source, T2_AB, muB_target)
    mu_induced_0_A += scatter_sum_compile(mu_dipole_A, e_AB_source, dim_size=n_atoms_A)

    mu_charge_B = torch.einsum("a,ai,a->ai", alpha_B_target, -T1_AB, qA_source)
    mu_induced_0_B = scatter_sum_compile(mu_charge_B, e_AB_target, dim_size=n_atoms_B)
    mu_dipole_B = torch.einsum("a,aij,aj->ai", alpha_B_target, T2_AB, muA_source)
    mu_induced_0_B += scatter_sum_compile(mu_dipole_B, e_AB_target, dim_size=n_atoms_B)

    # Self-consistent induced dipole iterations
    mu_induced_A = mu_induced_0_A.clone()
    mu_induced_B = mu_induced_0_B.clone()

    # Pre-compute index selections to avoid repeated operations in the loop
    mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
    mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
    mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
    mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

    # Iterative SCF procedure to converge induced dipoles
    for iteration in range(max_iterations):
        mu_induced_A_old = mu_induced_A.clone()
        mu_induced_B_old = mu_induced_B.clone()

        # Update pre-computed selections
        mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
        mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
        mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
        mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

        ####### (A) INDUCED DIPOLES ########
        # Induced dipoles on A due to induced dipoles on B
        mu_induced_A_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_A_source, T2_AB, mu_induced_B_at_AB_target
        )
        mu_induced_A_new = scatter_sum_compile(
            mu_induced_A_due_B, e_AB_source, dim_size=n_atoms_A
        )
        # Induced dipoles on A due to induced dipoles on A
        mu_induced_A_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_AA_target, T2_AA, mu_induced_A_at_AA_source
        )
        mu_induced_A_new += scatter_sum_compile(
            mu_induced_A_due_A, e_AA_target, dim_size=n_atoms_A
        )
        mu_induced_A_new += mu_induced_0_A

        ####### (B) INDUCED DIPOLES ########
        # Induced dipoles on B due to induced dipoles on A
        mu_induced_B_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_B_target, T2_AB, mu_induced_A_at_AB_source
        )
        mu_induced_B_new = scatter_sum_compile(
            mu_induced_B_due_A, e_AB_target, dim_size=n_atoms_B
        )
        # Induced dipoles on B due to induced dipoles on B
        mu_induced_B_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_BB_target, T2_BB, mu_induced_B_at_BB_source
        )
        mu_induced_B_new += scatter_sum_compile(
            mu_induced_B_due_B, e_BB_target, dim_size=n_atoms_B
        )
        mu_induced_B_new += mu_induced_0_B

        # Apply mixing
        mu_induced_A = (1 - omega) * mu_induced_A_old + omega * mu_induced_A_new
        mu_induced_B = (1 - omega) * mu_induced_B_old + omega * mu_induced_B_new

        # Check convergence
        delta_A = torch.norm(mu_induced_A - mu_induced_A_old)
        delta_B = torch.norm(mu_induced_B - mu_induced_B_old)
        delta = max(delta_A, delta_B)
        if delta < convergence_threshold:
            break

    # Final energy calculation
    muA_induced_source = mu_induced_A.index_select(0, e_AB_source)
    muB_induced_target = mu_induced_B.index_select(0, e_AB_target)
    qu = torch.einsum("x,xy->xy", qA_source, muB_induced_target) - torch.einsum(
        "x,xy->xy", qB_target, muA_induced_source
    )
    E_qu = torch.einsum("xy,xy->x", T1_AB, qu) * h2kcalmol
    E_uu = (
        -1.0
        * (
            torch.einsum("xy,xz,xyz->x", muA_induced_source, muB_target, T2_AB)
            + torch.einsum("xy,xz,xyz->x", muA_source, muB_induced_target, T2_AB)
        )
        * h2kcalmol
    )
    E_ind = (E_qu + E_uu) / 2.0
    E_ind -= E_ind_overlap
    return E_ind


# @torch.compile
def induced_dipole_induction_optimized_no_correction(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    ZB,
    RB,
    qB,
    muB,
    quadB,
    e_AB_source,
    e_AB_target,
    e_AA_source,
    e_BB_source,
    e_AA_target,
    e_BB_target,
    hirshfeld_volume_ratio_A: torch.tensor,
    hirshfeld_volume_ratio_B: torch.tensor,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param: float = 0.39,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
    polarizability_table=constants.polarizability_table,
) -> float:
    """
    Compute induction energy from self-consistent induced dipoles for a dimer without overlap/valence-width correction.

    Performs a Thole-damped SCF to converge induced dipoles on each monomer due to permanent multipoles and mutual induced dipoles, then returns the induction energy per A–B interaction edge.

    Parameters:
        ZA (Tensor): Atomic numbers for molecule A (N_A,).
        RA (Tensor): Cartesian coordinates for molecule A (N_A, 3).
        qA (Tensor): Nuclear+electronic monopoles for A (N_A,) or (N_A,1).
        muA (Tensor): Permanent dipoles for A (N_A, 3).
        quadA (Tensor): Permanent quadrupoles for A (per-atom multipole representation).
        ZB, RB, qB, muB, quadB: Same as above for molecule B.
        e_AB_source (LongTensor): Source atom indices in A for A–B interaction edges (n_edges,).
        e_AB_target (LongTensor): Target atom indices in B for A–B interaction edges (n_edges,).
        e_AA_source, e_AA_target (LongTensor): Intra-A interaction edge index pairs for AA interactions.
        e_BB_source, e_BB_target (LongTensor): Intra-B interaction edge index pairs for BB interactions.
        hirshfeld_volume_ratio_A (Tensor): Per-atom Hirshfeld volume ratios for A (N_A,).
        hirshfeld_volume_ratio_B (Tensor): Per-atom Hirshfeld volume ratios for B (N_B,).
        max_iterations (int): Maximum SCF iterations.
        convergence_threshold (float): Convergence threshold on induced-dipole change.
        omega (float): DIIS-like mixing factor for SCF updates (0..1).
        thole_damping_param (float): Thole damping parameter for short-range screening.
        Q_const (float): Scaling constant applied in electrostatic/tensor prefactors (kept for compatibility).
        polarizability_table (Tensor): Lookup table mapping atomic number to base polarizability.

    Returns:
        Tensor: Induction energy per A–B interaction edge (n_edges,) in kcal/mol.
    """

    delta = torch.eye(3, device=qA.device)
    h2kcalmol = constants.h2kcalmol  # Hartree to kcal/mol conversion factor

    alpha_0_A = torch.zeros_like(hirshfeld_volume_ratio_A)
    alpha_0_B = torch.zeros_like(hirshfeld_volume_ratio_B)

    # print(f"{alpha_0_A = }")
    # print(f"{alpha_0_B = }")
    # Use index_select for vectorized lookup
    alpha_0_A = torch.index_select(polarizability_table, 0, ZA.long())
    alpha_0_B = torch.index_select(polarizability_table, 0, ZB.long())
    # print(f"{alpha_0_A = }")
    # print(f"{alpha_0_A = }")
    # print(f"{hirshfeld_volume_ratio_A = }")
    # print(f"{hirshfeld_volume_ratio_B = }")
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A ** (4 / 3.0)
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B ** (4 / 3.0)

    # Calculate interaction tensors between atoms
    dR_AB, dR_AB_xyz, T0_AB, T1_AB, T2_AB = distance_tensors(
        RA, RB, e_AB_source, e_AB_target, alpha_A, alpha_B, thole_damping_param
    )
    dR_AA, dR_AA_xyz, T0_AA, T1_AA, T2_AA = distance_tensors(
        RA, RA, e_AA_source, e_AA_target, alpha_A, alpha_A, thole_damping_param
    )
    dR_BB, dR_BB_xyz, T0_BB, T1_BB, T2_BB = distance_tensors(
        RB, RB, e_BB_source, e_BB_target, alpha_B, alpha_B, thole_damping_param
    )

    # Select relevant tensors for atom pairs
    alpha_A_source = alpha_A.index_select(0, e_AB_source)
    alpha_B_target = alpha_B.index_select(0, e_AB_target)

    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    alpha_BB_target = alpha_B.index_select(0, e_BB_target)

    # Need to ensure that qA and qB are right shape even when ions
    qA = qA.reshape(-1, 1)
    qB = qB.reshape(-1, 1)
    qA_source = qA.squeeze(-1).index_select(0, e_AB_source)
    qB_target = qB.squeeze(-1).index_select(0, e_AB_target)

    muA_source = muA.index_select(0, e_AB_source)
    muB_target = muB.index_select(0, e_AB_target)

    # Initialize tensors for induced dipoles
    n_atoms_A = RA.shape[0]
    n_atoms_B = RB.shape[0]

    # Calculate initial induced dipoles
    mu_induced_0_A = torch.zeros((n_atoms_A, 3), device=qA.device)
    mu_induced_0_B = torch.zeros((n_atoms_B, 3), device=qB.device)

    # Calculate initial induced dipoles from molecule B's multipoles on molecule A
    mu_charge_A = torch.einsum("a,ai,a->ai", alpha_A_source, T1_AB, qB_target)
    mu_induced_0_A = scatter_sum_compile(mu_charge_A, e_AB_source, dim_size=n_atoms_A)
    mu_dipole_A = torch.einsum("a,aij,aj->ai", alpha_A_source, T2_AB, muB_target)
    mu_induced_0_A += scatter_sum_compile(mu_dipole_A, e_AB_source, dim_size=n_atoms_A)

    # Nan is in part of T1_AB tensor...
    mu_charge_B = torch.einsum("a,ai,a->ai", alpha_B_target, -T1_AB, qA_source)
    mu_induced_0_B = scatter_sum_compile(mu_charge_B, e_AB_target, dim_size=n_atoms_B)
    mu_dipole_B = torch.einsum("a,aij,aj->ai", alpha_B_target, T2_AB, muA_source)
    mu_induced_0_B += scatter_sum_compile(mu_dipole_B, e_AB_target, dim_size=n_atoms_B)

    # Self-consistent induced dipole iterations
    mu_induced_A = mu_induced_0_A.clone()
    mu_induced_B = mu_induced_0_B.clone()

    # Pre-compute index selections to avoid repeated operations in the loop
    mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
    mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
    mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
    mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

    # Iterative SCF procedure to converge induced dipoles
    for iteration in range(max_iterations):
        mu_induced_A_old = mu_induced_A.clone()
        mu_induced_B_old = mu_induced_B.clone()

        # Update pre-computed selections
        mu_induced_B_at_AB_target = mu_induced_B.index_select(0, e_AB_target)
        mu_induced_A_at_AB_source = mu_induced_A.index_select(0, e_AB_source)
        mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)
        mu_induced_B_at_BB_source = mu_induced_B.index_select(0, e_BB_source)

        ####### (A) INDUCED DIPOLES ########
        # Induced dipoles on A due to induced dipoles on B
        mu_induced_A_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_A_source, T2_AB, mu_induced_B_at_AB_target
        )
        mu_induced_A_new = scatter_sum_compile(
            mu_induced_A_due_B, e_AB_source, dim_size=n_atoms_A
        )
        # Induced dipoles on A due to induced dipoles on A
        mu_induced_A_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_AA_target, T2_AA, mu_induced_A_at_AA_source
        )
        mu_induced_A_new += scatter_sum_compile(
            mu_induced_A_due_A, e_AA_target, dim_size=n_atoms_A
        )
        mu_induced_A_new += mu_induced_0_A

        ####### (B) INDUCED DIPOLES ########
        # Induced dipoles on B due to induced dipoles on A
        mu_induced_B_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_B_target, T2_AB, mu_induced_A_at_AB_source
        )
        mu_induced_B_new = scatter_sum_compile(
            mu_induced_B_due_A, e_AB_target, dim_size=n_atoms_B
        )
        # Induced dipoles on B due to induced dipoles on B
        mu_induced_B_due_B = torch.einsum(
            "a,aij,aj->ai", alpha_BB_target, T2_BB, mu_induced_B_at_BB_source
        )
        mu_induced_B_new += scatter_sum_compile(
            mu_induced_B_due_B, e_BB_target, dim_size=n_atoms_B
        )
        mu_induced_B_new += mu_induced_0_B

        # Apply mixing
        mu_induced_A = (1 - omega) * mu_induced_A_old + omega * mu_induced_A_new
        mu_induced_B = (1 - omega) * mu_induced_B_old + omega * mu_induced_B_new

        # Check convergence
        delta_A = torch.norm(mu_induced_A - mu_induced_A_old)
        delta_B = torch.norm(mu_induced_B - mu_induced_B_old)
        delta = max(delta_A, delta_B)
        if delta < convergence_threshold:
            break

    # Final energy calculation
    muA_induced_source = mu_induced_A.index_select(0, e_AB_source)
    muB_induced_target = mu_induced_B.index_select(0, e_AB_target)
    qu = torch.einsum("x,xy->xy", qA_source, muB_induced_target) - torch.einsum(
        "x,xy->xy", qB_target, muA_induced_source
    )
    E_qu = torch.einsum("xy,xy->x", T1_AB, qu) * h2kcalmol
    E_uu = (
        -1.0
        * (
            torch.einsum("xy,xz,xyz->x", muA_induced_source, muB_target, T2_AB)
            + torch.einsum("xy,xz,xyz->x", muA_source, muB_induced_target, T2_AB)
        )
        * h2kcalmol
    )
    E_ind = (E_qu + E_uu) / 2.0
    return E_ind


def induced_dipole(
    ZA,
    RA,
    qA,
    muA,
    quadA,
    e_AA_source,
    e_AA_target,
    hirshfeld_volume_ratio_A: torch.tensor,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    thole_damping_param: float = 0.39,
    Q_const=3.0,  # set to 1.0 to agree with CLIFF
    polarizability_table=constants.polarizability_table,
) -> float:
    """
    Compute self-consistent induced dipoles for a single molecule A from its permanent multipoles and Hirshfeld volume ratios.

    Performs a Thole-damped SCF to converge induced dipoles on each atom of molecule A using its charges (qA), permanent dipoles (muA), quadrupoles (quadA), and atomic coordinates (RA). Iteration continues until the change in induced dipoles falls below convergence_threshold or max_iterations is reached.

    Parameters:
        ZA (Tensor): Atomic number indices for atoms in molecule A.
        RA (Tensor): Atomic coordinates for molecule A with shape (n_atoms, 3).
        qA (Tensor): Atomic charges for A with shape (n_atoms,).
        muA (Tensor): Permanent atomic dipoles for A with shape (n_atoms, 3).
        quadA (Tensor): Atomic quadrupoles for A (unused by this function but required for consistency).
        e_AA_source (Tensor): Source indices for pairwise A->A interactions (edge source).
        e_AA_target (Tensor): Target indices for pairwise A->A interactions (edge target).
        hirshfeld_volume_ratio_A (Tensor): Per-atom Hirshfeld volume ratios used to scale free-atom polarizabilities.
        max_iterations (int, optional): Maximum SCF iterations. Default 200.
        convergence_threshold (float, optional): Convergence norm threshold for induced dipole changes. Default 1e-8.
        omega (float, optional): Damping/mixing parameter for SCF updates. Default 0.7.
        thole_damping_param (float, optional): Thole damping parameter. Default 0.39.
        Q_const (float, optional): Multiplicative constant for electrostatic scaling (kept for consistency). Default 3.0.
        polarizability_table (Tensor or array-like, optional): Table of free-atom polarizabilities indexed by ZA.

    Notes:
        - Polarizabilities are scaled as alpha = alpha_free * (hirshfeld_volume_ratio)^(4/3).
        - Thole damping is applied via distance_tensors.
        - This function performs the SCF and computes induced dipoles but does not return a value (implicit None).
    """

    delta = torch.eye(3, device=qA.device)
    h2kcalmol = constants.h2kcalmol  # Hartree to kcal/mol conversion factor

    alpha_0_A = torch.zeros_like(hirshfeld_volume_ratio_A)

    # Use index_select for vectorized lookup
    alpha_0_A = torch.index_select(polarizability_table, 0, ZA.long())
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A ** (4 / 3.0)

    # Calculate interaction tensors between atoms
    dR_AA, dR_AA_xyz, T0_AA, T1_AA, T2_AA = distance_tensors(
        RA, RA, e_AA_source, e_AA_target, alpha_A, alpha_A, thole_damping_param
    )

    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    alpha_AA_source = alpha_A.index_select(0, e_AA_source)

    # Need to ensure that qA and qB are right shape even when ions
    qA = qA.reshape(-1, 1)
    qA_source = qA.squeeze(-1).index_select(0, e_AA_source)
    qA_target = qA.squeeze(-1).index_select(0, e_AA_target)

    muA_source = muA.index_select(0, e_AA_source)
    muA_target = muA.index_select(0, e_AA_target)

    # Initialize tensors for induced dipoles
    n_atoms_A = RA.shape[0]

    # Calculate initial induced dipoles
    mu_induced_0_A = torch.zeros((n_atoms_A, 3), device=qA.device)

    # Calculate initial induced dipoles from molecule B's multipoles on molecule A
    mu_charge_A = torch.einsum("a,ai,a->ai", alpha_AA_source, T1_AA, qA_target)
    mu_induced_0_A = scatter_sum_compile(mu_charge_A, e_AA_source, dim_size=n_atoms_A)
    mu_dipole_A = torch.einsum("a,aij,aj->ai", alpha_AA_source, T2_AA, muA_target)
    mu_induced_0_A += scatter_sum_compile(mu_dipole_A, e_AA_source, dim_size=n_atoms_A)

    # Self-consistent induced dipole iterations
    mu_induced_A = mu_induced_0_A.clone()

    # Pre-compute index selections to avoid repeated operations in the loop
    mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)

    # Iterative SCF procedure to converge induced dipoles
    for iteration in range(max_iterations):
        mu_induced_A_old = mu_induced_A.clone()

        # Update pre-computed selections
        mu_induced_A_at_AA_source = mu_induced_A.index_select(0, e_AA_source)

        ####### (A) INDUCED DIPOLES ########
        # Induced dipoles on A due to induced dipoles on A
        mu_induced_A_due_A = torch.einsum(
            "a,aij,aj->ai", alpha_AA_target, T2_AA, mu_induced_A_at_AA_source
        )
        mu_induced_A_new = scatter_sum_compile(
            mu_induced_A_due_A, e_AA_target, dim_size=n_atoms_A
        )
        mu_induced_A_new += mu_induced_0_A

        mu_induced_A = (1 - omega) * mu_induced_A_old + omega * mu_induced_A_new

        # Check convergence
        delta_A = torch.norm(mu_induced_A - mu_induced_A_old)
        delta = max(delta_A)
        if delta < convergence_threshold:
            break

    # Final energy calculation
    muA_induced_source = mu_induced_A.index_select(0, e_AA_source)
    muB_induced_target = mu_induced_A.index_select(0, e_AA_target)
    return


def isolate_atom_parameter_predictions(batch, output):
    """
    Split batched per-atom prediction tensors into per-molecule lists.

    Parameters:
        batch: object with attribute `natom_per_mol`, a 1D tensor-like giving the number of atoms for each molecule in the batch.
        output: sequence where
            - output[0] is per-atom charges (tensor of length total_atoms),
            - output[1] is per-atom dipoles,
            - output[2] is per-atom quadrupoles,
            - output[3] is per-atom `hlist`,
            - output[-1] is per-atom parameter tensor `K`.

    Returns:
        mol_charges, mol_dipoles, mol_qpoles, mol_hlist, mol_K:
            Five lists of length `batch.natom_per_mol.size(0)`. Each element is a tensor containing the corresponding property restricted to the atoms of that molecule.
    """
    batch_size = batch.natom_per_mol.size(0)
    q = output[0]
    mu = output[1]
    th = output[2]
    hlist = output[3]
    K = output[-1]
    mol_charges = [[] for i in range(batch_size)]
    mol_dipoles = [[] for i in range(batch_size)]
    mol_qpoles = [[] for i in range(batch_size)]
    mol_hlist = [[] for i in range(batch_size)]
    mol_K = [[] for i in range(batch_size)]
    i_offset = 0
    for n, i in enumerate(batch.natom_per_mol):
        mol_charges[n] = q[i_offset : i_offset + i]
        mol_dipoles[n] = mu[i_offset : i_offset + i]
        mol_qpoles[n] = th[i_offset : i_offset + i]
        mol_hlist[n] = hlist[i_offset : i_offset + i]
        mol_K[n] = K[i_offset : i_offset + i]
        i_offset += i
    return mol_charges, mol_dipoles, mol_qpoles, mol_hlist, mol_K


def isolate_atom_parameter_predictions_ap3(batch, output):
    batch_size = batch.natom_per_mol.size(0)
    q = output[0]
    mu = output[1]
    th = output[2]
    hlist = output[3]
    K_hfvr = output[-2][:, 0]
    K_vw = output[-2][:, 1]
    K_elst = output[-1]
    mol_charges = [[] for i in range(batch_size)]
    mol_dipoles = [[] for i in range(batch_size)]
    mol_qpoles = [[] for i in range(batch_size)]
    mol_hfvr = [[] for i in range(batch_size)]
    mol_vw = [[] for i in range(batch_size)]
    mol_hlist = [[] for i in range(batch_size)]
    mol_K = [[] for i in range(batch_size)]
    i_offset = 0
    for n, i in enumerate(batch.natom_per_mol):
        mol_charges[n] = q[i_offset : i_offset + i]
        mol_dipoles[n] = mu[i_offset : i_offset + i]
        mol_qpoles[n] = th[i_offset : i_offset + i]
        mol_hfvr[n] = K_hfvr[i_offset : i_offset + i]
        mol_vw[n] = K_vw[i_offset : i_offset + i]
        mol_hlist[n] = hlist[i_offset : i_offset + i]
        mol_K[n] = K_elst[i_offset : i_offset + i]
        i_offset += i
    return mol_charges, mol_dipoles, mol_qpoles, mol_hlist, mol_hfvr, mol_vw, mol_K


class AM_DimerParam_Model:
    def __init__(
        self,
        dataset=None,
        atom_model=None,
        atom_model_type="AtomMPNN",
        model_type="AtomTypeParamNN",
        pre_trained_model_path=None,
        atom_model_pre_trained_path=None,
        n_message=3,
        n_rbf=8,
        n_neuron=64,
        n_embed=8,
        r_cut=5.0,
        param_start_mean=[1.6],
        param_start_std=[0.25],
        n_params=1,
        use_GPU=None,
        ignore_database_null=True,
        ds_spec_type=1,
        ds_root="data",
        ds_max_size=None,
        ds_atomic_batch_size=200,
        ds_force_reprocess=False,
        ds_skip_process=False,
        ds_skip_compile=False,
        ds_num_devices=1,
        ds_datapoint_storage_n_objects=1000,
        ds_prebatched=False,
        ds_random_seed=42,
        ds_in_memory=False,
        print_lvl=0,
        ds_qcel_molecules=None,
        ds_energy_labels=None,
        dimer_eval_type="elst_damping",
        elst_damping_type="CLIFF",
        freeze_atom_model=True,
    ):
        """
        Construct an AtomTypeParamModel wrapper that builds or loads an atom-level model, a parameter-predicting model, and optional dimer evaluators and dataset.

        This initializer will:
        - Prefer loading a full pretrained model if `pre_trained_model_path` is given (all other model-building parameters are ignored except `dataset`).
        - Optionally load a pretrained atom model via `atom_model_pre_trained_path`.
        - Instantiate or use the provided `atom_model` and `model` (controlled by `atom_model_type` and `model_type`), then create dimer evaluators (DimerProp) configured by `dimer_eval_type` and `elst_damping_type`.
        - Select device automatically (GPU if available unless `use_GPU` is False), move models to that device, and optionally construct an on-disk/in-memory dataset unless `ignore_database_null` is True.

        Parameters:
            dataset (optional): Preconstructed dataset object to use instead of building one.
            atom_model (optional): Preconstructed atom-level model instance to use.
            atom_model_type (str): Type name for constructing a default atom model when `atom_model` is not provided (e.g., "AtomMPNN", "AtomHirshfeldMPNN", "AtomTypeParamNN").
            model_type (str): Type name for the parameter-predicting model to construct when no pretrained model is loaded (e.g., "AtomTypeParamNN").
            pre_trained_model_path (str, optional): Path to a checkpoint for the full AtomTypeParam model; when provided this checkpoint is loaded and model-building kwargs are ignored.
            atom_model_pre_trained_path (str, optional): Path to a checkpoint for an atom-level model; when provided the atom model is re-instantiated to match checkpoint config and its weights are loaded.
            n_message (int): Number of message-passing steps for the atom/parameter models.
            n_rbf (int): Number of radial basis functions (used by some atom model types).
            n_neuron (int): Hidden neuron count used in MLP readouts.
            n_embed (int): Embedding dimensionality for per-atom embeddings.
            r_cut (float): Cutoff distance used when constructing datasets.
            param_start_mean (float or list): Initial mean(s) for parameter embeddings.
            param_start_std (float or list): Initial stddev(s) for parameter embeddings.
            n_params (int): Number of per-atom parameters to predict.
            use_GPU (bool or None): If False, force CPU; if None, use GPU if available.
            ignore_database_null (bool): If False and no `dataset` is provided, build the dataset(s) from `ds_root` and related dataset args.
            ds_spec_type (int): Dataset specification / split type forwarded to dataset constructor.
            ds_root (str): Root directory for datasets.
            ds_max_size (int, optional): Max dataset size (truncates when set).
            ds_atomic_batch_size (int): Atomic batch size used by dataset construction.
            ds_force_reprocess (bool): Force dataset reprocessing.
            ds_skip_process (bool): Skip dataset processing.
            ds_skip_compile (bool): Skip any compilation steps when building dataset.
            ds_num_devices (int): Number of devices used when building dataset metadata.
            ds_datapoint_storage_n_objects (int): Dataset storage chunking parameter.
            ds_prebatched (bool): Whether dataset inputs are already prebatched.
            ds_random_seed (int): RNG seed for dataset construction.
            ds_in_memory (bool): Whether dataset should be loaded in memory.
            print_lvl (int): Verbosity level for dataset construction.
            ds_qcel_molecules (optional): Optional qcel molecules passed into dataset builder.
            ds_energy_labels (optional): Energy label specifications for dataset builder.
            dimer_eval_type (str): Dimer evaluation mode used by created DimerProp (e.g., "elst_damping", "elst").
            elst_damping_type (str): Electrostatic damping variant to use ("CLIFF" or "AMOEBA"); can be overridden by loaded checkpoint config.

        Notes:
            - When loading checkpoints, model constructor parameters (n_message, n_neuron, n_embed, param_start_*) are read from the checkpoint config to reinstantiate compatible model instances.
            - The constructed instance exposes `self.model`, `self.atom_model`, `self.dimer_model`, `self.dimer_model_elst` (when applicable), `self.dataset`, and `self.device`.
        """
        if torch.cuda.is_available() and use_GPU is not False:
            device = torch.device("cuda:0")
            print("running on the GPU")
        else:
            device = torch.device("cpu")
            print("running on the CPU")
        self.ds_spec_type = ds_spec_type
        if atom_model_type == "AtomMPNN":
            self.atom_model = AtomMPNN()
            am_type = AtomMPNN
        elif atom_model_type == "AtomHirshfeldMPNN":
            self.atom_model = AtomHirshfeldMPNN()
            am_type = AtomHirshfeldMPNN
        elif atom_model_type == "AtomTypeParamNN":
            self.atom_model = AtomTypeParamNN()
            am_type = AtomTypeParamNN
        # elif atom_model_type == "AtomTypeParamMPNN":
        #     self.atom_model = AtomTypeParamMPNN()
        #     am_type = AtomTypeParamMPNN
        else:
            raise ValueError(f"Unknown atom_model_type: {atom_model_type}")

        if atom_model_pre_trained_path:
            print(
                f"Loading pre-trained AtomMPNN model from {atom_model_pre_trained_path}"
            )
            checkpoint = model_io.load_checkpoint(
                atom_model_pre_trained_path, map_location=device
            )
            am_config = model_io.load_config_from_checkpoint(checkpoint)
            if am_config is None:
                am_config = checkpoint.get("config", {})
            if atom_model_type in ["AtomHirshfeldMPNN", "AtomMPNN"]:
                self.atom_model = am_type(
                    n_message=am_config["n_message"],
                    n_rbf=am_config["n_rbf"],
                    n_neuron=am_config["n_neuron"],
                    n_embed=am_config["n_embed"],
                    r_cut=am_config["r_cut"],
                )
            elif atom_model_type == "AtomTypeParamNN":
                self.atom_model = am_type(
                    n_message=am_config["n_message"],
                    n_neuron=am_config["n_neuron"],
                    n_embed=am_config["n_embed"],
                    param_start_mean=am_config["param_start_mean"],
                    param_start_std=am_config["param_start_std"],
                    n_params=am_config["n_params"],
                )
            # elif atom_model_type == "AtomTypeParamMPNN":
            #     self.atom_model = am_type(
            #         n_message=checkpoint["config"]["n_message"],
            #         n_rbf=checkpoint["config"]["n_rbf"],
            #         n_neuron=checkpoint["config"]["n_neuron"],
            #         n_embed=checkpoint["config"]["n_embed"],
            #         r_cut=checkpoint["config"]["r_cut"],
            #         param_start_mean=checkpoint["config"]["param_start_mean"],
            #         param_start_std=checkpoint["config"]["param_start_std"],
            #         n_params=checkpoint["config"]["n_params"],
            #     )
            # model_state_dict = checkpoint["model_state_dict"]
            model_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
            self.atom_model.load_state_dict(model_state_dict)
        elif atom_model:
            print("Using provided AtomMPNN model:", atom_model)
            self.atom_model = atom_model
        else:
            print(
                """No atom model provided.
    Assuming atomic multipoles and embeddings are
    pre-computed and passed as input to the model.
"""
            )
        if pre_trained_model_path:
            print(
                f"Loading pre-trained MTP-MTP {model_type} from {pre_trained_model_path}"
            )
            checkpoint = model_io.load_checkpoint(pre_trained_model_path)
            config = model_io.load_config_from_checkpoint(checkpoint)
            if config is None:
                config = checkpoint.get("config", {})
            # Load elst_damping_type from checkpoint if available, otherwise use default
            elst_damping_type = config.get("elst_damping_type", elst_damping_type)
            if model_type == "AtomTypeParamNN":
                self.model = AtomTypeParamNN(
                    atom_model=self.atom_model,
                    n_message=config["n_message"],
                    n_neuron=config["n_neuron"],
                    n_embed=config["n_embed"],
                    param_start_mean=config["param_start_mean"],
                    param_start_std=config["param_start_std"],
                    n_params=config.get("n_params", 1),
                )
            # elif model_type == "AtomTypeParamMPNN":
            #     self.model = AtomTypeParamMPNN(
            #         atom_model=self.atom_model,
            #         n_message=checkpoint["config"]["n_message"],
            #         n_rbf=checkpoint["config"]["n_rbf"],
            #         n_neuron=checkpoint["config"]["n_neuron"],
            #         n_embed=checkpoint["config"]["n_embed"],
            #         r_cut=checkpoint["config"]["r_cut"],
            #         param_start_mean=checkpoint["config"]["param_start_mean"],
            #         param_start_std=checkpoint["config"]["param_start_std"],
            #         n_params=checkpoint["config"].get("n_params", 1),
            #     )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            model_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
            self.model.load_state_dict(model_state_dict)
        else:
            if model_type == "AtomTypeParamNN":
                self.model = AtomTypeParamNN(
                    atom_model=self.atom_model,
                    n_message=n_message,
                    n_neuron=n_neuron,
                    n_embed=n_embed,
                    param_start_mean=param_start_mean,
                    param_start_std=param_start_std,
                    n_params=n_params,
                )
            # elif model_type == "AtomTypeParamMPNN":
            #     self.model = AtomTypeParamMPNN(
            #         atom_model=self.atom_model,
            #         n_message=n_message,
            #         n_rbf=n_rbf,
            #         n_neuron=n_neuron,
            #         n_embed=n_embed,
            #         r_cut=r_cut,
            #         param_start_mean=param_start_mean,
            #         param_start_std=param_start_std,
            #         n_params=n_params,
            #     )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        self.n_params = n_params
        self.dimer_eval_type = dimer_eval_type
        self.elst_damping_type = elst_damping_type
        self.dimer_model = DimerProp(
            self.model,
            dimer_eval=dimer_eval_type,
            elst_damping_type=elst_damping_type,
            freeze_atom_model=freeze_atom_model,
        )
        if self.dimer_eval_type in ["elst", "elst_damping"]:
            self.dimer_model_elst = DimerProp(
                self.model,
                dimer_eval="elst",
                elst_damping_type=elst_damping_type,
                freeze_atom_model=freeze_atom_model,
            )
        else:
            self.dimer_model_elst = None

        if n_message != self.model.n_message:
            print(f"Changing n_mesage from {self.model.n_message} to {n_message}")
            self.model.n_message = n_message
        if n_neuron != self.model.n_neuron:
            print(f"Changing n_neuron from {self.model.n_neuron} to {n_neuron}")
            self.model.n_neuron = n_neuron
        if n_embed != self.model.n_embed:
            print(f"Changing n_embed from {self.model.n_embed} to {n_embed}")
            self.model.n_embed = n_embed
        # Handle param_start_mean/std as lists
        if isinstance(param_start_mean, (list, tuple)):
            if param_start_mean != self.model.param_start_mean:
                print(f"Changing param_start_mean to {param_start_mean}")
                self.model.param_start_mean = param_start_mean
        else:
            # Scalar case, check if different from all current values
            if not all(p == param_start_mean for p in self.model.param_start_mean):
                print(f"Changing param_start_mean to {param_start_mean}")
                self.model.param_start_mean = [param_start_mean] * self.model.n_params

        if isinstance(param_start_std, (list, tuple)):
            if param_start_std != self.model.param_start_std:
                print(f"Changing param_start_std to {param_start_std}")
                self.model.param_start_std = param_start_std
        else:
            # Scalar case
            if not all(p == param_start_std for p in self.model.param_start_std):
                print(f"Changing param_start_std to {param_start_std}")
                self.model.param_start_std = [param_start_std] * self.model.n_params

        self.device = device
        self.atom_model.to(device)
        self.model.to(device)
        self.dimer_model.to(device)
        self.dimer_model.AtomTypeParam.to(device)
        if hasattr(self.dimer_model.AtomTypeParam, "atom_model"):
            self.dimer_model.AtomTypeParam.atom_model.to(device)

        split_dbs = [2, 5, 6, 7]
        ds_qcel_split_db = (
            ds_qcel_molecules is not None
            and len(ds_qcel_molecules) == 2
            and isinstance(ds_qcel_molecules[0], list)
        )
        self.dataset = dataset
        if (
            not ignore_database_null
            and self.dataset is None
            and self.ds_spec_type not in split_dbs
            and not ds_qcel_split_db
        ):

            def setup_ds(fp=ds_force_reprocess):
                return ap2_fused_module_dataset(
                    root=ds_root,
                    r_cut=r_cut,
                    r_cut_im=torch.inf,
                    spec_type=ds_spec_type,
                    max_size=ds_max_size,
                    force_reprocess=fp,
                    atom_model=self.atom_model,
                    atomic_batch_size=ds_atomic_batch_size,
                    num_devices=ds_num_devices,
                    skip_processed=ds_skip_process,
                    skip_compile=ds_skip_compile,
                    random_seed=ds_random_seed,
                    in_memory=ds_in_memory,
                    datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                    print_level=print_lvl,
                    qcel_molecules=ds_qcel_molecules,
                    energy_labels=ds_energy_labels,
                    # storage_type="h5",  # "pt" or "h5" for storage format
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
                return [
                    ap2_fused_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=torch.inf,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model=self.atom_model,
                        atomic_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        skip_compile=ds_skip_compile,
                        in_memory=ds_in_memory,
                        random_seed=ds_random_seed,
                        split="train",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        print_level=print_lvl,
                        qcel_molecules=ds_qcel_molecules[0],
                        energy_labels=ds_energy_labels[0],
                        # storage_type="h5",  # "pt" or "h5" for storage format
                    ),
                    ap2_fused_module_dataset(
                        root=ds_root,
                        r_cut=r_cut,
                        r_cut_im=torch.inf,
                        spec_type=ds_spec_type,
                        max_size=ds_max_size,
                        force_reprocess=fp,
                        atom_model=self.atom_model,
                        atomic_batch_size=ds_atomic_batch_size,
                        num_devices=ds_num_devices,
                        skip_processed=ds_skip_process,
                        skip_compile=ds_skip_compile,
                        random_seed=ds_random_seed,
                        in_memory=ds_in_memory,
                        split="test",
                        datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
                        print_level=print_lvl,
                        qcel_molecules=ds_qcel_molecules[1],
                        energy_labels=ds_energy_labels[1],
                        # storage_type="h5",  # "pt" or "h5" for storage format
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
            self.model(batch)
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
        self,
        ap2_model_path=None,
        am_model_path=None,
        model_id=None,
        ap2_fused: bool = False,
    ):
        if model_id is not None:
            ensemble_prefix = "ap2-fused_ensemble" if ap2_fused else "ap2_ensemble"
            ap2_model_path = resolve_pretrained_path(
                f"{ensemble_prefix}/ap2_{model_id}.pt"
            )
        elif ap2_model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided.")

        checkpoint = model_io.load_checkpoint(ap2_model_path, map_location=self.device)
        model_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
        self.model.load_state_dict(model_state_dict)

        if am_model_path is not None:
            am_checkpoint = model_io.load_checkpoint(
                am_model_path, map_location=self.device
            )
            am_state_dict = model_io.load_state_dict_from_checkpoint(am_checkpoint)
            self.atom_model.load_state_dict(am_state_dict)
        return self

    def _create_checkpoint(
        self,
        model: nn.Module = None,
        atom_model: nn.Module = None,
        embed_atom_model: bool = True,
        metadata: dict | None = None,
    ) -> dict:
        """
        Create a v2 checkpoint dictionary for this model.
        """
        if model is None:
            model = self.model
        if atom_model is None:
            atom_model = self.atom_model

        model = model_io.unwrap_model(model)
        atom_model = model_io.unwrap_model(atom_model)

        if hasattr(model, "get_config"):
            model_config = model.get_config()
        else:
            model_config = {
                "n_message": getattr(model, "n_message", 3),
                "n_neuron": getattr(model, "n_neuron", 128),
                "n_embed": getattr(model, "n_embed", 8),
                "param_start_mean": getattr(model, "param_start_mean", [1.8]),
                "param_start_std": getattr(model, "param_start_std", [0.01]),
                "n_params": getattr(model, "n_params", 1),
            }
        model_config["elst_damping_type"] = self.elst_damping_type
        model_config["dimer_eval_type"] = self.dimer_eval_type

        submodels = None
        if embed_atom_model and atom_model is not None:
            if hasattr(atom_model, "get_config"):
                atom_config = atom_model.get_config()
            else:
                atom_config = {
                    "n_message": getattr(atom_model, "n_message", 3),
                    "n_rbf": getattr(atom_model, "n_rbf", 8),
                    "n_neuron": getattr(atom_model, "n_neuron", 128),
                    "n_embed": getattr(atom_model, "n_embed", 8),
                    "r_cut": getattr(atom_model, "r_cut", 5.0),
                }
            submodels = {
                "atom_model": model_io.create_submodel_checkpoint(
                    model=atom_model,
                    config=atom_config,
                    model_type=type(atom_model).__name__,
                )
            }

        return model_io.create_checkpoint(
            model=model,
            config=model_config,
            model_type=type(model).__name__,
            submodels=submodels,
            metadata=metadata,
        )

    def save_model(
        self,
        path: str,
        embed_atom_model: bool = True,
        metadata: dict | None = None,
    ) -> None:
        """
        Save the model to a checkpoint file in v2 format.
        """
        checkpoint = self._create_checkpoint(
            embed_atom_model=embed_atom_model,
            metadata=metadata,
        )
        model_io.save_checkpoint(checkpoint, path)

    def _qcel_example_input(
        self,
        mols,
        batch_size=1,
        r_cut=5,
    ):
        dimer_batch = ap2_fused_collate_update_no_target(
            [
                qcel_dimer_to_fused_data(
                    mol, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                )
                for n, mol in enumerate(mols)
            ]
        )
        batch = Data(
            x=dimer_batch.ZA,
            R=dimer_batch.RA,
            edge_index=torch.vstack((dimer_batch.e_AA_source, dimer_batch.e_AA_target)),
            molecule_ind=dimer_batch.molecule_ind_A,
            total_charge=dimer_batch.total_charge_A,
            natom_per_mol=dimer_batch.natom_per_mol_A,
        )
        batch.to(self.device)
        return batch

    def _qcel_dimer_example_input(
        self,
        mols,
        batch_size=1,
        r_cut=5,
    ):
        batch = ap2_fused_collate_update_no_target(
            [
                qcel_dimer_to_fused_data(
                    mol, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                )
                for n, mol in enumerate(mols)
            ]
        )
        batch.to(self.device)
        return batch

    def _assemble_pairs(
        self,
        inp_batch,
        E_sr_dimer,
        E_sr,
        E_elst_sr,
        E_elst_lr,
    ):
        indA_to_dimer = []
        indB_to_dimer = []
        indA_to_atom = []
        indB_to_atom = []
        pair_energies_batch = []

        indsA_sr = inp_batch["e_ABsr_source"]
        indsB_sr = inp_batch["e_ABsr_target"]
        indsA_lr = inp_batch["e_ABlr_source"]
        indsB_lr = inp_batch["e_ABlr_target"]

        dimer_inds, atoms_per_dimer = torch.unique(
            inp_batch.dimer_ind, return_counts=True
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

        # E_sr, E_elst_sr, E_elst_lr
        for e_pair, e_elst_sr, indA, indB in zip(E_sr, E_elst_sr, indsA_sr, indsB_sr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][0:4, atomA, atomB] += e_pair.numpy()
            pair_energies_batch[i][0, atomA, atomB] += e_elst_sr.numpy()

        for e_elst_lr, indA, indB in zip(E_elst_lr, indsA_lr, indsB_lr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][0, atomA, atomB] += e_elst_lr
        return pair_energies_batch

    def _assemble_mtp_pairs(
        self,
        inp_batch,
        E_elst_sr,
        E_elst_lr,
    ):
        indA_to_dimer = []
        indB_to_dimer = []
        indA_to_atom = []
        indB_to_atom = []
        pair_energies_batch = []

        indsA_sr = inp_batch["e_ABsr_source"]
        indsB_sr = inp_batch["e_ABsr_target"]
        indsA_lr = inp_batch["e_ABlr_source"]
        indsB_lr = inp_batch["e_ABlr_target"]

        dimer_inds, atoms_per_dimer = torch.unique(
            inp_batch.dimer_ind, return_counts=True
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
            pair_energies_batch.append(np.zeros((size_A, size_B)))

        indA_to_dimer = np.concatenate(indA_to_dimer)
        indB_to_dimer = np.concatenate(indB_to_dimer)
        indA_to_atom = np.concatenate(indA_to_atom)
        indB_to_atom = np.concatenate(indB_to_atom)
        for e_elst_sr, indA, indB in zip(E_elst_sr, indsA_sr, indsB_sr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][atomA, atomB] += e_elst_sr.numpy()
        for e_elst_lr, indA, indB in zip(E_elst_lr, indsA_lr, indsB_lr):
            i = indA_to_dimer[indA]
            assert i == indB_to_dimer[indB]
            atomA = indA_to_atom[indA]
            atomB = indB_to_atom[indB]
            pair_energies_batch[i][atomA, atomB] += e_elst_lr
        return pair_energies_batch

    @torch.inference_mode()
    def predict_qcel_mols_dimer(
        self,
        mols,
        batch_size=1,
        r_cut=None,
        verbose=False,
        return_pairs=False,
        return_elst=False,
    ):
        """
        Predict per-dimer energies for a list of qcel dimer molecules using the configured dimer model.

        Parameters:
            mols (Sequence): Iterable of qcel dimer objects convertible by qcel_dimer_to_fused_data.
            batch_size (int): Number of dimers to process per forward pass.
            r_cut (float or None): Cutoff radius for assembling graph edges; when None, uses the atom model's default if available.
            verbose (bool): If true, prints a brief progress message after processing batches.
            return_pairs (bool): If true, also return per-pair (atom-pair) energy components alongside per-dimer totals.
            return_elst (bool): If true, also return pairwise electrostatic components. Mutually exclusive with `return_pairs`.

        Returns:
            numpy.ndarray or (numpy.ndarray, list): If neither `return_pairs` nor `return_elst` is set, returns a NumPy array of shape (N, M) with per-dimer predictions (N = number of dimers, M = model-determined number of outputs). If `return_pairs` or `return_elst` is set, returns a tuple (predictions, pairwise_energies) where `pairwise_energies` is a list of per-dimer pairwise energy entries produced during prediction.

        Notes:
            - Moves the atom_model to the wrapper's configured device.
            - The number of output columns M is determined from the first batch forward pass.
            - `return_pairs` and `return_elst` cannot both be true (the function asserts against this).
        """
        assert not (return_elst and return_pairs), (
            "return_elst and return_pairs are not compatible"
        )
        if r_cut is None and hasattr(self.atom_model, "r_cut"):
            r_cut = self.atom_model.r_cut
        elif hasattr(self.atom_model.atom_model, "r_cut"):
            r_cut = self.atom_model.atom_model.r_cut

        N = len(mols)
        # Determine number of output columns from model (e.g., 2 for Elst + Indu)
        # Will be determined after first forward pass
        predictions = None
        if return_pairs or return_elst:
            pairwise_energies = []
        self.atom_model.to(self.device)
        for i in range(0, N, batch_size):
            upper_bound = min(i + batch_size, N)
            dimer_batch = ap2_fused_collate_update_no_target(
                [
                    qcel_dimer_to_fused_data(
                        dimer, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                    )
                    for n, dimer in enumerate(mols[i:upper_bound])
                ]
            )
            dimer_batch.to(device=self.device)
            preds = self.dimer_model(dimer_batch)[0]
            # Use dimer_ind_full for scatter_sum to include both sr and lr edges
            preds = scatter_sum_compile(
                preds,
                dimer_batch.dimer_ind_full,
                dim_size=torch.tensor(
                    dimer_batch.total_charge_A.size(0), dtype=torch.long
                ),
            )
            preds_np = preds.cpu().numpy()
            # Initialize predictions array on first batch
            if predictions is None:
                n_outputs = preds_np.shape[1] if preds_np.ndim > 1 else 1
                predictions = np.zeros((N, n_outputs))
            predictions[i:upper_bound] = preds_np.reshape(upper_bound - i, -1)
        if verbose:
            print(f"Predictions for {i} to {i + batch_size} out of {N}")
        if return_pairs or return_elst:
            return predictions, pairwise_energies
        print(f"Predictions: {predictions}")
        return predictions

    @torch.inference_mode()
    def predict_qcel_mols_monomer_props(
        self,
        mols,
        batch_size=1,
        r_cut=None,
        am_type="ap2",
        verbose=False,
        model_type="atom_model",
    ):
        output_A = []
        output_B = []
        if model_type == "atom_model":
            model = self.atom_model
        elif model_type == "model":
            model = self.model
        # check if atom_model has r_cut attribute
        if r_cut is None and hasattr(model, "r_cut") and model.r_cut is not None:
            r_cut = model.r_cut
        elif hasattr(model.atom_model, "r_cut"):
            r_cut = model.atom_model.r_cut
        elif hasattr(self.atom_model, "atom_model") and hasattr(
            self.atom_model.atom_model, "r_cut"
        ):
            r_cut = self.atom_model.atom_model.r_cut
        else:
            raise ValueError("r_cut must be provided if not defined in the model.")

        N = len(mols)
        model.to(self.device)
        if am_type == "ap2":
            isolate_fn = isolate_atom_parameter_predictions
        elif am_type == "ap3":
            isolate_fn = isolate_atom_parameter_predictions_ap3
        else:
            raise ValueError(f"Unknown am_type: {am_type}")
        for i in range(0, N, batch_size):
            upper_bound = min(i + batch_size, N)
            dimer_batch = ap2_fused_collate_update_no_target(
                [
                    qcel_dimer_to_fused_data(
                        dimer, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                    )
                    for n, dimer in enumerate(mols[i:upper_bound])
                ]
            )
            batch_A = Data(
                x=dimer_batch.ZA,
                R=dimer_batch.RA,
                edge_index=torch.vstack(
                    (dimer_batch.e_AA_source, dimer_batch.e_AA_target)
                ),
                molecule_ind=dimer_batch.molecule_ind_A,
                total_charge=dimer_batch.total_charge_A,
                natom_per_mol=dimer_batch.natom_per_mol_A,
            )
            with torch.no_grad():
                v = isolate_fn(batch_A, self.model(batch_A))
                output_A.extend(list(zip(*v)))
            batch_B = Data(
                x=dimer_batch.ZB,
                R=dimer_batch.RB,
                edge_index=torch.vstack(
                    (dimer_batch.e_BB_source, dimer_batch.e_BB_target)
                ),
                molecule_ind=dimer_batch.molecule_ind_B,
                total_charge=dimer_batch.total_charge_B,
                natom_per_mol=dimer_batch.natom_per_mol_B,
            )
            with torch.no_grad():
                v = isolate_fn(batch_B, self.model(batch_B))
                output_B.extend(list(zip(*v)))
        return output_A, output_B

    def example_input(
        self,
        mol=None,
        r_cut=5.0,
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
            [mol],
            batch_size=1,
            r_cut=r_cut,
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
        self, dataloader, loss_fn, optimizer, rank_device, scheduler, y_ind=0
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
            ref = batch.y[:, y_ind]
            preds = self.dimer_model(batch)[0]
            # print(f"{preds = }")
            preds = scatter_sum_compile(
                preds,
                batch.dimer_ind,
                dim_size=torch.tensor(batch.total_charge_A.size(0), dtype=torch.long),
            )
            comp_errors = preds - ref
            # print(f"{preds = }")
            # print(f"{ref = }")
            batch_loss = (
                torch.mean(torch.square(comp_errors))
                if (loss_fn is None)
                else loss_fn(preds, ref)
            )
            batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.dimer_model.parameters(), max_norm=0.2)
            optimizer.step()
            total_loss += batch_loss.item()
            comp_errors_t.append(comp_errors.detach().cpu())
        if scheduler is not None:
            scheduler.step()
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t), dim=0)
        return total_loss, total_MAE_t

    # @torch.inference_mode()
    def __evaluate_batches_single_proc(self, dataloader, loss_fn, rank_device, y_ind=0):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                preds = self.dimer_model(batch)[0]
                ref = batch.y[:, y_ind]
                preds = scatter_sum_compile(
                    preds,
                    batch.dimer_ind,
                    dim_size=torch.tensor(
                        batch.total_charge_A.size(0), dtype=torch.long
                    ),
                )
                comp_errors = preds - ref
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, ref)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t), dim=0)
        return total_loss, total_MAE_t

    def __evaluate_batches_single_proc_elst_no_damping(
        self, dataloader, loss_fn, rank_device
    ):
        self.model.eval()
        comp_errors_t = []
        total_loss = 0.0
        with torch.no_grad():
            for n, batch in enumerate(dataloader):
                batch = batch.to(rank_device, non_blocking=True)
                preds = self.dimer_model_elst(batch)[0]
                ref = batch.y[:, 0]
                preds = scatter_sum_compile(
                    preds,
                    batch.dimer_ind,
                    dim_size=torch.tensor(
                        batch.total_charge_A.size(0), dtype=torch.long
                    ),
                )
                # print(f"{preds=}")
                comp_errors = preds - ref
                # print(f"{comp_errors=}")
                batch_loss = (
                    torch.mean(torch.square(comp_errors))
                    if (loss_fn is None)
                    else loss_fn(preds, ref)
                )
                total_loss += batch_loss.item()
                comp_errors_t.append(comp_errors.detach().cpu())
        comp_errors_t = torch.cat(comp_errors_t, dim=0)
        total_MAE_t = torch.mean(torch.abs(comp_errors_t), dim=0)
        return total_loss, total_MAE_t

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
        skip_compile=False,
    ):
        # (1) Compile Model
        """
        Train the model in a single process using provided datasets and hyperparameters.

        Performs optional model compilation, constructs data loaders, runs epoch-wise training and evaluation, tracks and saves the best model (by test loss) to self.model_save_path, and stops early if NaNs are detected. The saved checkpoint includes model state and a config that captures architecture and the active elst_damping_type.

        Parameters:
            train_dataset: Dataset
                Training dataset compatible with APNet2_fused_DataLoader.
            test_dataset: Dataset
                Validation/test dataset compatible with APNet2_fused_DataLoader.
            n_epochs (int):
                Number of training epochs to run.
            batch_size (int):
                Batch size for both training and test loaders.
            lr (float):
                Initial learning rate for the Adam optimizer.
            pin_memory (bool):
                Passed to DataLoader; whether to pin memory.
            num_workers (int):
                Number of worker processes for data loading.
            skip_compile (bool):
                If True, skip torch compilation step before training.
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
        # if self.ds_spec_type in [1, 5, 6]:
        collate_fn = ap2_fused_collate_update
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
        scheduler = None
        # criterion = None  # defaults to MSE
        criterion = torch.nn.MSELoss()

        # (4) Set eval functions
        if self.dimer_eval_type == "elst_damping":
            y_ind = 0
            term = "Elst"
        elif self.dimer_eval_type in ["induced_dipole", "induced_dipole_param"]:
            y_ind = 2
            term = "Indu"
            self.dimer_model.polarizability_table = (
                self.dimer_model.polarizability_table.to(self.device)
            )
        elif self.dimer_eval_type in [
            "elst_damping__induced_dipole",
            "ap3_elst_damping__induced_dipole",
        ]:
            assert isinstance(self.atom_model, AtomTypeParamNN), (
                f"{self.dimer_eval_type} is only compatible with "
                "AtomTypeParamNN atom models presently."
            )
            self.model.to(self.device)
            self.model.atom_model.to(self.device)
            self.model.atom_model.atom_model.to(self.device)
            print(self.device)
            y_ind = torch.tensor([0, 2])
            term = "Elst      Ind"
            self.dimer_model.polarizability_table = (
                self.dimer_model.polarizability_table.to(self.device)
            )
            self.dimer_model.to(rank_device)
            self.dimer_model.AtomTypeParam.to(rank_device)
            if hasattr(self.dimer_model.AtomTypeParam, "atom_model"):
                self.dimer_model.AtomTypeParam.atom_model.to(rank_device)
        else:
            raise ValueError(f"Unknown dimer_eval_type: {self.dimer_eval_type}")
        print(
            f"                                       {term}",
            flush=True,
        )

        # (5) Evaluate once pre-training
        if self.dimer_model_elst is not None:
            t0 = time.time()
            # _, no_damping_MAE_t = self.__evaluate_batches_single_proc_elst_no_damping(
            #     train_loader, criterion, rank_device
            # )
            # _, no_damping_MAE_v = self.__evaluate_batches_single_proc_elst_no_damping(
            #     test_loader, criterion, rank_device
            # )
            # print(
            #     f" (No Damping)  ({time.time() - t0: < 7.2f}s)"
            #     f" MAE: {no_damping_MAE_t: > 7.3f}/{no_damping_MAE_v: < 7.3f}",
            #     flush=True,
            # )
        t0 = time.time()
        # t_out = self.__evaluate_batches_single_proc(train_loader, criterion, rank_device, y_ind=y_ind)
        # v_out = self.__evaluate_batches_single_proc(test_loader, criterion, rank_device, y_ind=y_ind)
        # train_loss, total_MAE_t = t_out
        # test_loss, total_MAE_v = v_out
        # if isinstance(y_ind, torch.Tensor):
        #     mae_string = " ".join([f"{mae_t: > 7.3f}/{mae_v: < 7.3f}" for mae_t, mae_v in zip(total_MAE_t, total_MAE_v)])
        # else:
        #     mae_string = f"{total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f}"
        # print(
        #     f" (Pre-training)({time.time() - t0: < 7.2f}s)"
        #     f" MAE: {mae_string}",
        #     flush=True,
        # )
        # lowest_test_loss = test_loss
        lowest_test_loss = float("inf")
        # cpu_model = self.model.to("cpu")
        # self.model.to(rank_device)
        for epoch in range(n_epochs):
            t1 = time.time()
            t_out = self.__train_batches_single_proc(
                train_loader,
                loss_fn=criterion,
                optimizer=optimizer,
                rank_device=rank_device,
                scheduler=scheduler,
                y_ind=y_ind,
            )
            v_out = self.__evaluate_batches_single_proc(
                test_loader, loss_fn=criterion, rank_device=rank_device, y_ind=y_ind
            )
            train_loss, total_MAE_t = t_out
            test_loss, total_MAE_v = v_out

            # Track best model
            star_marker = " "
            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                star_marker = "*"
                cpu_model = model_io.unwrap_model(self.model).to("cpu")
                cpu_atom_model = model_io.unwrap_model(self.atom_model).to("cpu")
                best_model = deepcopy(cpu_model)
                if self.model_save_path:
                    checkpoint = self._create_checkpoint(
                        model=cpu_model,
                        atom_model=cpu_atom_model,
                        embed_atom_model=True,
                    )
                    model_io.save_checkpoint(checkpoint, self.model_save_path)
                self.model.to(rank_device)

            if isinstance(y_ind, torch.Tensor):
                mae_string = " ".join(
                    [
                        f"{mae_t: > 7.3f}/{mae_v: < 7.3f}"
                        for mae_t, mae_v in zip(total_MAE_t, total_MAE_v)
                    ]
                )
            else:
                mae_string = f"{total_MAE_t: > 7.3f}/{total_MAE_v: < 7.3f}"
            print(
                f"  EPOCH: {epoch:4d} ({time.time() - t1:<7.2f}s)  MAE: "
                f"{mae_string} {star_marker}",
                flush=True,
            )
            if not self.device == "CPU":
                torch.cuda.empty_cache()
            if torch.any(total_MAE_t.isnan()) or torch.any(total_MAE_v.isnan()):
                cpu_model = model_io.unwrap_model(self.model).to("cpu")
                cpu_atom_model = model_io.unwrap_model(self.atom_model).to("cpu")
                print("NaN detected, stopping training")
                checkpoint = self._create_checkpoint(
                    model=cpu_model,
                    atom_model=cpu_atom_model,
                    embed_atom_model=True,
                    metadata={"nan_crash": True},
                )
                model_io.save_checkpoint(checkpoint, "nan_crash_model.pt")
                break
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
        random_seed=42,
        skip_compile=False,
        lr_decay=None,
    ):
        print("NOTE: lr_decay is not implemented.")
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
        print("~~ Training Dimer Param ~~", flush=True)
        print(f"{self.model}", flush=True)
        print(
            f"    Training on {len(train_dataset)} samples,"
            f" Testing on {len(test_dataset)} samples"
        )
        print("\nNetwork Hyperparameters:", flush=True)
        print(f"  {self.model.n_message=}", flush=True)
        print(f"  {self.model.n_neuron=}", flush=True)
        print(f"  {self.model.n_embed=}", flush=True)
        print(f"  {self.model.param_start_mean=}", flush=True)
        print(f"  {self.model.param_start_std=}", flush=True)
        print("\nTraining Hyperparameters:", flush=True)
        print(f"  {n_epochs=}", flush=True)
        print(f"  {lr=}\n", flush=True)
        print(f"  {batch_size=}", flush=True)

        if self.device.type == "cuda":
            pin_memory = False
        else:
            pin_memory = False

        self.shuffle = shuffle

        if world_size > 1:
            print("Running multi-process training", flush=True)
            raise NotImplementedError(
                "Multi-process training is not implemented for MTP-MTP models."
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
                skip_compile=skip_compile,
            )
        return


### Atom Type Model Wrapper ####
class AtomTypeParamModel:
    def __init__(
        self,
        dataset=None,
        atom_model=None,
        atom_model_type="AtomMPNN",
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
        freeze_atom_model=True,
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
        if atom_model_type == "AtomMPNN":
            self.atom_model = AtomMPNN()
            am_type = AtomMPNN
        elif atom_model_type == "AtomHirshfeldMPNN":
            self.atom_model = AtomHirshfeldMPNN()
            am_type = AtomHirshfeldMPNN
        else:
            raise ValueError(f"Unknown atom_model_type: {atom_model_type}")

        self.n_params = 1
        if monomer_eval_type in ["hirshfeld_volume_ratio__valence_width"]:
            self.n_params = 2

        if atom_model_pre_trained_path:
            print(
                f"Loading pre-trained AtomMPNN model from {atom_model_pre_trained_path}"
            )
            checkpoint = model_io.load_checkpoint(
                atom_model_pre_trained_path, map_location=device
            )
            am_config = model_io.load_config_from_checkpoint(checkpoint)
            if am_config is None:
                am_config = checkpoint.get("config", {})
            self.atom_model = am_type(
                n_message=am_config["n_message"],
                n_rbf=am_config["n_rbf"],
                n_neuron=am_config["n_neuron"],
                n_embed=am_config["n_embed"],
                r_cut=am_config["r_cut"],
            )
            model_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
            self.atom_model.load_state_dict(model_state_dict)
        elif atom_model:
            print("Using provided AtomMPNN model:", atom_model)
            self.atom_model = atom_model
        else:
            print(
                """No atom model provided.
    Assuming atomic multipoles and embeddings are
    pre-computed and passed as input to the model.
"""
            )
        if pre_trained_model_path:
            print(f"Loading pre-trained MTP-MTP model from {pre_trained_model_path}")
            checkpoint = model_io.load_checkpoint(pre_trained_model_path)
            config = model_io.load_config_from_checkpoint(checkpoint)
            if config is None:
                config = checkpoint.get("config", {})
            self.model = AtomTypeParamNN(
                atom_model=self.atom_model,
                n_message=config["n_message"],
                n_neuron=config["n_neuron"],
                n_embed=config["n_embed"],
                param_start_mean=config["param_start_mean"],
                param_start_std=config["param_start_std"],
                n_params=config.get("n_params", 1),
                freeze_atom_model=freeze_atom_model,
            )
            model_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
            self.model.load_state_dict(model_state_dict)
        else:
            self.model = AtomTypeParamNN(
                atom_model=self.atom_model,
                n_message=n_message,
                n_neuron=n_neuron,
                n_embed=n_embed,
                param_start_mean=param_start_mean,
                param_start_std=param_start_std,
                n_params=self.n_params,
                freeze_atom_model=freeze_atom_model,
            )
        self.n_params = self.n_params
        self.monomer_eval_type = monomer_eval_type
        self.device = device
        self.dataset = dataset
        mp.set_sharing_strategy("file_system")
        if not ignore_database_null and self.dataset is None:
            self.dataset = atomic_hirshfeld_module_dataset(
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
        self.train_shuffle = None
        # torch.jit.enable_onednn_fusion(True)
        return

    def set_pretrained_model(self, model_path):
        checkpoint = model_io.load_checkpoint(model_path, map_location=self.device)
        model_state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
        self.model.load_state_dict(model_state_dict)
        return self

    def _create_checkpoint(
        self,
        model: nn.Module = None,
        atom_model: nn.Module = None,
        embed_atom_model: bool = True,
        metadata: dict | None = None,
    ) -> dict:
        """
        Create a v2 checkpoint dictionary for this model.
        """
        if model is None:
            model = self.model
        if atom_model is None:
            atom_model = self.atom_model

        model = model_io.unwrap_model(model)
        atom_model = model_io.unwrap_model(atom_model)

        model_config = (
            model.get_config()
            if hasattr(model, "get_config")
            else {
                "n_message": getattr(model, "n_message", 3),
                "n_neuron": getattr(model, "n_neuron", 128),
                "n_embed": getattr(model, "n_embed", 8),
                "param_start_mean": getattr(model, "param_start_mean", [1.8]),
                "param_start_std": getattr(model, "param_start_std", [0.01]),
                "n_params": getattr(model, "n_params", 1),
            }
        )
        model_config["monomer_eval_type"] = self.monomer_eval_type

        submodels = None
        if embed_atom_model and atom_model is not None:
            atom_config = (
                atom_model.get_config()
                if hasattr(atom_model, "get_config")
                else {
                    "n_message": getattr(atom_model, "n_message", 3),
                    "n_rbf": getattr(atom_model, "n_rbf", 8),
                    "n_neuron": getattr(atom_model, "n_neuron", 128),
                    "n_embed": getattr(atom_model, "n_embed", 8),
                    "r_cut": getattr(atom_model, "r_cut", 5.0),
                }
            )
            submodels = {
                "atom_model": model_io.create_submodel_checkpoint(
                    model=atom_model,
                    config=atom_config,
                    model_type=type(atom_model).__name__,
                )
            }

        return model_io.create_checkpoint(
            model=model,
            config=model_config,
            model_type=type(model).__name__,
            submodels=submodels,
            metadata=metadata,
        )

    def save_model(
        self,
        path: str,
        embed_atom_model: bool = True,
        metadata: dict | None = None,
    ) -> None:
        """
        Save the model to a checkpoint file in v2 format.
        """
        checkpoint = self._create_checkpoint(
            embed_atom_model=embed_atom_model,
            metadata=metadata,
        )
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
                params = self.model(batch)[-1]
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
            params = self.model(batch)[-1]
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
            params = self.model(batch)[-1]
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
                params = self.model(batch)[-1]
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
                params = self.model(batch)[-1]
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
            collate_fn=atomic_collate_update_prebatched,
            # collate_fn=atomic_hirshfeld_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=test_sampler,
            collate_fn=atomic_collate_update_prebatched,
            # collate_fn=atomic_hirshfeld_collate_update,
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
                        cpu_model = model_io.unwrap_model(self.model).to("cpu")
                        cpu_atom_model = model_io.unwrap_model(self.atom_model).to(
                            "cpu"
                        )
                        checkpoint = self._create_checkpoint(
                            model=cpu_model,
                            atom_model=cpu_atom_model,
                            embed_atom_model=True,
                        )
                        model_io.save_checkpoint(checkpoint, self.model_save_path)
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
            collate_fn=atomic_collate_update_prebatched,
            # collate_fn=atomic_hirshfeld_collate_update,
        )

        test_loader = AtomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=atomic_collate_update_prebatched,
            # collate_fn=atomic_hirshfeld_collate_update,
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
                        cpu_model = model_io.unwrap_model(self.model).to("cpu")
                        cpu_atom_model = model_io.unwrap_model(self.atom_model).to(
                            "cpu"
                        )
                        checkpoint = self._create_checkpoint(
                            model=cpu_model,
                            atom_model=cpu_atom_model,
                            embed_atom_model=True,
                        )
                        model_io.save_checkpoint(checkpoint, self.model_save_path)
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
                    charge, dipole, qpole, hlist, Ks = self.model(batch)
                    hfvr = Ks[:, 0]
                    vw = Ks[:, 1]
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
        """
        Run the atom-level model on a batch and return predicted per-atom multipole parameters and related atom properties.

        Parameters:
            data: A batched graph or input compatible with the wrapped atom model containing node features and batch indices.

        Returns:
            charge: Per-atom monopole (charge) tensor.
            dipole: Per-atom dipole tensor.
            qpole: Per-atom quadrupole (or higher multipole) tensor.
            hirshfeld_volume_ratios: Per-atom Hirshfeld volume ratio tensor used for scaling polarizabilities.
            valence_widths: Per-atom valence-width tensor used in overlap/width corrections.
            hlist: Internal per-atom feature list (message-passing hidden states) used by downstream readouts.
        """
        charge, dipole, qpole, hirshfeld_volume_ratios, valence_widths, hlist = (
            self.model(data)
        )
        return charge, dipole, qpole, hirshfeld_volume_ratios, valence_widths, hlist
