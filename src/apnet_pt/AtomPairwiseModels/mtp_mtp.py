import math
import os
import re
import time
import warnings
from copy import deepcopy
from importlib import resources

import numpy as np
import qcelemental as qcel
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data

from apnet_pt.torch_util import set_weights_to_value
from apnet_pt.util import resolve_split_indices
from qcml_dftd3.d3 import d3, resolve_d3_damping_parameters

from .. import constants
from .. import model_io
from ..training_tracking import (
    TrackerBackend,
    WandbConfig,
    configure_distributed_tracking,
    run_tracked_single_process,
    stage_final_weights,
    track_epoch_from_locals,
    track_pretraining_from_locals,
    tracked_ddp_worker,
)
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

RACKERS_PARAMETER_NAMES = (
    "elst",
    "thole_direct",
    "thole_mutual",
    "ind_overlap",
)
RACKERS_INITIAL_VALUES = (1.8, 0.34, 0.39, 1.8)
RACKERS_INITIAL_STDS = (0.01, 0.01, 0.01, 0.01)
RACKERS_POSITIVITY_EPSILON = 1e-8

RACKERS_ELST_INDEX = 0
RACKERS_THOLE_DIRECT_INDEX = 1
RACKERS_THOLE_MUTUAL_INDEX = 2
RACKERS_IND_OVERLAP_INDEX = 3

# CLIFF classical-exchange parameter contract.  ``K_exch`` is a single positive
# per-atom parameter combined *multiplicatively* (``K_i K_j``), unlike the
# Thole parameters which use a geometric mean.  ``2.5`` sits near the centre of
# the CLIFF Table I ``K_exch`` range (0.60-7.60, mean ~3.2) and reproduces
# water-dimer-scale exchange at initialization.
CLIFF_EXCH_PARAMETER_NAMES = ("exch",)
CLIFF_EXCH_INITIAL_VALUES = (2.5,)
# Raw-space (pre-softplus) initialization spread.  0.25 rather than the Rackers
# 0.01: `K_exch` spans 0.60-7.60 across CLIFF's atom types, and the per-element
# seeds below only resolve the element, not CLIFF's coordination-number
# refinement within it (C4/C3/C2 are 2.26/2.46/2.80; N3/N2/N1 are
# 4.47/4.63/3.49).  0.25 in raw space is about the width of those
# within-element spreads, so the readout starts with room to separate atom
# types rather than having to climb out of a delta function.
CLIFF_EXCH_INITIAL_STDS = (0.25,)

# Per-element ``K_exch`` seeds, from the CLIFF Table I fitted values (Schriber
# et al., J. Chem. Phys. 154, 184110 (2021)).  CLIFF assigns 17 atom types by
# element *and* coordination number; these are the per-element centres of those
# type groups, which is the resolution a per-``Z`` embedding can represent --
# the MPNN readout correction supplies the coordination-number dependence.
#
# Seeding per element matters far more than it looks.  ``K_exch`` enters Eq. (8)
# as the product ``K_i K_j``, so a uniform 2.5 makes an H-H pair
# ``2.5 * 2.5 = 6.25`` when CLIFF's hydrogen types give ``0.77 * 0.77 = 0.59``:
# an order of magnitude too repulsive on the *most common* pair in the data.
# The optimizer's fastest way to undo that is to drive every ``K_exch`` toward
# zero, and because ``softplus`` saturates, once it gets there the head has no
# gradient left to come back.  A 100-epoch run at the uniform seed collapsed to
# a mean ``K_exch`` of 0.025 with 23% of atoms pinned at ``positivity_epsilon``
# and an exchange MAE equal to the predict-zero baseline.  Elements absent here
# fall back to ``CLIFF_EXCH_INITIAL_VALUES``.
CLIFF_EXCH_INITIAL_VALUES_BY_Z = {
    1: 0.7676,   # mean of HC 0.9890, HN 0.6910, HO 0.5996, HS 0.7909
    6: 2.5079,   # mean of C4 2.2649, C3 2.4566, C2 2.8023
    7: 4.1936,   # mean of N3 4.4660, N2 4.6251, N1 3.4896
    8: 5.5987,   # mean of O2 5.8538, O1 5.3435
    9: 7.6036,   # F
    16: 3.2308,  # mean of S2 3.2842, S1 3.1773
    17: 3.8152,  # Cl
    35: 4.1008,  # Br
}

# CLIFF Table I covers only these eight elements.  This dataset also contains
# Na (Z=11) and P (Z=15), which fall back to the scalar
# :data:`CLIFF_EXCH_INITIAL_VALUES` guess.  See :data:`OVERLAP_WIDTH_CEILING`
# for why that is survivable.
CLIFF_TABLE_I_ELEMENTS = frozenset(CLIFF_EXCH_INITIAL_VALUES_BY_Z)

# Combined classical contract.  Columns 0-3 intentionally mirror
# ``RACKERS_PARAMETER_NAMES`` so the electrostatics and induction physics paths
# are reused unchanged and a Rackers checkpoint's learned columns remain
# interpretable; column 4 adds exchange.
CLIFF_CLASSICAL_PARAMETER_NAMES = (
    "elst",
    "thole_direct",
    "thole_mutual",
    "ind_overlap",
    "exch",
)
CLIFF_CLASSICAL_INITIAL_VALUES = (1.8, 0.34, 0.39, 1.8, 2.5)
CLIFF_CLASSICAL_INITIAL_STDS = (0.01, 0.01, 0.01, 0.01, 0.25)

# Per-element seeds for the combined route.  Only the exchange column has
# published per-element values; columns 0-3 keep their scalar Rackers seeds.
CLIFF_CLASSICAL_INITIAL_VALUES_BY_Z = {
    "exch": CLIFF_EXCH_INITIAL_VALUES_BY_Z,
}

# Straight-through bounds on the *raw* (pre-softplus) per-atom parameters,
# expressed relative to each column's scalar seed: ``[fraction * seed,
# multiple * seed]``.
#
# These exist because ``K = softplus(raw) + epsilon`` cannot recover from
# collapse.  ``dK/draw = sigmoid(raw)``, so as ``K -> 0`` the gradient that
# would lift it back vanishes too; a head driven to ``raw ~ -18`` is dead for
# the rest of training no matter what the loss wants.  Clamping ``raw`` in the
# *raw* domain with a straight-through gradient (see :func:`_ste_clamp`) parks a
# collapsing head at ``fraction * seed`` -- where ``sigmoid(raw)`` is still
# order 0.1, not 1e-8 -- so it stays trainable and can climb back out.
#
# The ceiling is the mirror image: the same 100-epoch run drove the ``elst``
# damping column to a mean of 22.2 and a maximum of 164.7 from a seed of 1.8,
# which is not a physical damping width.  ``10x`` the seed is loose enough to
# let a column adapt by an order of magnitude and tight enough to catch a
# runaway.
CLIFF_PARAM_FLOOR_FRACTION = 0.05
CLIFF_PARAM_CEILING_MULTIPLE = 10.0

# Multiplier applied to the *output* layer of each per-message readout MLP at
# construction, shrinking the random correction so the per-element seed actually
# governs the initial prediction.
#
# Without it the seeding work above is largely wasted.  Measured at
# initialization on held-out dimers, exchange MAE against SAPT ``Exch``:
#
#     uniform K = 2.5, full-size readout     12.4    (predict-zero: 18.6)
#     per-element K,   full-size readout      5.5
#     per-element K,   readout x 0.1          3.3
#
# So the random readout was contributing more error than the entire
# uniform-versus-per-element difference.  ``AtomTypeParamNN.__init__`` has
# carried a commented-out ``set_weights_excluding_guess(0.01)`` since it was
# written, which is the same intent; that helper fills *every* weight with a
# constant (including the frozen atom model's), so this scales only the readout
# output layers, which makes the correction's magnitude scale linearly and
# leaves the rest of the hierarchy alone.
CLIFF_READOUT_INIT_SCALE = 0.1

# Column indices into the 2-D parameter tensors returned by ``CliffExchangeNN``
# and ``CliffClassicalNN``.  Both classes present a uniform ``[n_atoms, k]``
# contract, so these indices are the only sanctioned way to read a column.
CLIFF_EXCH_INDEX = 0
CLIFF_CLASSICAL_ELST_INDEX = 0
CLIFF_CLASSICAL_THOLE_DIRECT_INDEX = 1
CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX = 2
CLIFF_CLASSICAL_IND_OVERLAP_INDEX = 3
CLIFF_CLASSICAL_EXCH_INDEX = 4

# Lower bound applied to predicted valence widths before they enter the
# ``rsqrt`` in :func:`atomic_overlap_S_ij`.  ``AtomHirshfeldMPNN`` emits
# ``relu(...) + 1e-4`` (ap2_hirshfeld_atom_model.py:403), so a predicted width
# can legitimately approach zero and blow ``B_ij`` up.  ``0.1`` matches the
# floor that ``apnet3.valence_width_exch`` has always applied.
OVERLAP_WIDTH_FLOOR = 0.1

# Upper bound on predicted valence widths before they enter the ``rsqrt``.
#
# The floor above guards ``B_ij -> inf``; this guards the opposite and far more
# damaging direction.  ``S_ij`` decays as ``exp(-r / sqrt(sigma_i sigma_j))``,
# so an *over*-large width flattens the exponential and the pair energy
# explodes.  The frozen HFVR/valence-width atom model emits
# ``sigma = 1.8952`` for every Na atom in this dataset -- identical to four
# decimal places across all of them, i.e. an untrained per-element embedding
# bias rather than a prediction -- against 0.40-0.52 for C/N/O.  That is a
# ~100x inflation of ``S_ij``, and combined with the fallback ``K = 2.5`` it
# produced single-dimer exchange predictions of 2960 kcal/mol against a
# reference of 42.  Chlorine shows the same failure less uniformly
# (sigma 0.52-2.11).  Measured over 1280 dimers, the ten worst carried 95% of
# the squared error and every one of the fifteen worst contained Na or Cl.
#
# 1.0 bohr sits above every legitimate width observed (P 0.70, S 0.67, C 0.52,
# N 0.48, O 0.43, H 0.39) and below the pathological ones, so it caps the
# atom model's out-of-domain extrapolation without touching any element it was
# actually trained on.  This is a guard on an input the exchange term cannot
# defend itself against, not a fitted parameter; the real fix is an atom model
# that covers these elements.
OVERLAP_WIDTH_CEILING = 1.0

# Dimer evaluation modes whose per-edge energies live on the *full*
# intermolecular edge domain (``e_ABfull_source`` / ``e_ABfull_target``).  Those
# energies must be scatter-aggregated with ``batch.dimer_ind_full``; the
# short-range ``batch.dimer_ind`` covers only ``e_ABsr_*`` and would silently
# attribute long-range edges to the wrong dimer (and drop the tail entirely) if
# used here.  ``_dimer_index_for_output`` is the sole consumer, so keeping the
# membership in one named place is what stops the edge domain and the
# aggregation index from drifting apart.  Every mode listed here evaluates its
# kernels over ``e_ABfull_*``; conversely, any forward that switches to
# ``e_ABfull_*`` must be added here in the same change.
FULL_EDGE_DIMER_EVAL_MODES = frozenset(
    {
        "rackers_thole",
        "rackers_thole_overlap",
        "cliff_exch",
        "cliff_classical",
        "cliff_classical_overlap",
        "cliff_classical_d3",
    }
)

# Trainable multi-component CLIFF routes.  These are the only ``dimer_eval``
# values for which a total/component loss split (``component_gamma``) is
# meaningful: ``cliff_exch`` predicts a single component, and
# ``cliff_classical_d3`` is inference-only.
# Maps the short MAE-report term labels onto the tracker's metric names, so a
# route's target columns, report header, and logged metric names are all derived
# from one labelled selection instead of three parallel literals.
TRACKER_METRIC_LABELS_BY_TERM = {
    "Elst": "electrostatics",
    "Exch": "exchange",
    "Ind": "induction",
    "Indu": "induction",
    "Disp": "dispersion",
}

COMBINED_CLIFF_DIMER_EVAL_MODES = frozenset(
    {
        "cliff_classical",
        "cliff_classical_overlap",
    }
)

# Checkpoint contract for every positive per-atom parameter head:
# ``{model_type: parameter_names}``.  ``AM_DimerParam_Model.__init__`` looks the
# expected ``parameter_names`` up here instead of hard-coding one contract, so a
# checkpoint can never silently reassign the physical meaning of a column.  The
# ordering is load-bearing -- a reordered list is rejected, not remapped.
POSITIVE_PARAMETER_CONTRACTS: dict[str, tuple[str, ...]] = {
    "RackersTholeDampingNN": RACKERS_PARAMETER_NAMES,
    "CliffExchangeNN": CLIFF_EXCH_PARAMETER_NAMES,
    "CliffClassicalNN": CLIFF_CLASSICAL_PARAMETER_NAMES,
}

# Human-readable prefix used in the checkpoint-contract error messages.  The
# Rackers strings predate the generalization and are asserted on verbatim by
# ``tests/test_rackers_thole_damping.py``, so "Rackers" must stay mapped here;
# every other contract falls back to its own ``model_type``.
_POSITIVE_PARAMETER_ERROR_LABELS: dict[str, str] = {
    "RackersTholeDampingNN": "Rackers",
}


def _positive_parameter_error_label(model_type: str) -> str:
    return _POSITIVE_PARAMETER_ERROR_LABELS.get(model_type, model_type)


# Per-column width of the MAE progress report.  The header and the numeric rows
# are both generated from the column *labels*, so widening the target selection
# from one to two to three columns adds a label and nothing else -- no format
# string anywhere assumes a particular number of columns.
_MAE_REPORT_COLUMN_WIDTH = 10


def _mae_report_header(labels) -> str:
    """Left-justified per-column MAE header for any number of columns.

    Byte-identical to the literals it replaces for the pre-existing selections
    (``("Elst",)`` -> ``"Elst"``, ``("Elst", "Ind")`` -> ``"Elst      Ind"``);
    ``tests/test_cliff_classical_exchange.py`` pins that equivalence.
    """
    return "".join(
        f"{label:<{_MAE_REPORT_COLUMN_WIDTH}}" for label in labels
    ).rstrip()


def _polarizability_table_on_device(
    polarizability_table: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return polarizability_table.to(device=device)


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

    def info(self):
        """Print a Unicode model tree for this model."""
        from apnet_pt.model_print import model_tree_string

        print(model_tree_string(self, unicode=True))

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
                - "rackers_thole": combined Rackers electrostatics and pure
                  induced-point-dipole induction (_rackers_thole_forward)
                - "rackers_thole_overlap": combined Rackers electrostatics and
                  overlap-augmented induction (_rackers_thole_overlap_forward)
                - "cliff_exch": CLIFF classical exchange repulsion alone
                  (_cliff_exch_forward), returning [n_edges]
                - "cliff_classical": combined CLIFF electrostatics, exchange,
                  and pure induced-point-dipole induction
                  (_cliff_classical_forward), returning [n_edges, 3] with
                  columns (Elst, Exch, Indu)
                - "cliff_classical_overlap": the same three terms with the
                  short-range induction overlap correction enabled
                  (_cliff_classical_overlap_forward), returning [n_edges, 3]
                - "cliff_classical_d3": the three classical terms plus DFT-D3
                  dispersion (_cliff_classical_d3_forward), returning
                  [n_edges, 4] with columns (Elst, Exch, Indu, Disp)
                - "ap3_elst_damping__induced_dipole": AP3-specific damped electrostatics plus induction (_ap3_elst_damping_indu_induced_dipole_forward)
                - "ap3_atomMPNN": return AP3 atom multipole parameters only (_ap3_atomMPNN)

        Notes:
            - This method sets self.forward to the corresponding internal forward implementation.
            - Induction modes clone the global polarizability table into
              self.polarizability_table. Both Rackers modes scale
              polarizabilities with Hirshfeld volume ratios; only
              "rackers_thole_overlap" uses valence widths in its energy
              expression.
            - "cliff_exch" is exchange-only: it runs no induction and therefore
              deliberately does *not* clone a polarizability table. The three
              "cliff_classical*" modes do induction and clone it exactly as the
              Rackers modes do.
            - The combined CLIFF column order is fixed at (Elst, Exch, Indu)
              and, for "cliff_classical_d3", (Elst, Exch, Indu, Disp). That
              matches the pairwise dataset's y = [Elst, Exch, Ind, Disp]
              layout, so target slicing is a plain column select.
            - Every "cliff_*" mode evaluates over the full intermolecular edge
              domain and is a member of FULL_EDGE_DIMER_EVAL_MODES.
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
        elif dimer_eval == "rackers_thole":
            self.forward = self._rackers_thole_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "rackers_thole_overlap":
            self.forward = self._rackers_thole_overlap_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "cliff_exch":
            # Exchange-only: no induction, so no polarizability table.
            self.forward = self._cliff_exch_forward
        elif dimer_eval == "cliff_classical":
            self.forward = self._cliff_classical_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "cliff_classical_overlap":
            self.forward = self._cliff_classical_overlap_forward
            self.polarizability_table = constants.polarizability_table.clone()
        elif dimer_eval == "cliff_classical_d3":
            self.forward = self._cliff_classical_d3_forward
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

    def _rackers_thole_forward(self, batch):
        return self._rackers_thole_common_forward(
            batch, include_overlap=False
        )

    def _rackers_thole_overlap_forward(self, batch):
        return self._rackers_thole_common_forward(
            batch, include_overlap=True
        )

    def _rackers_thole_common_forward(self, batch, include_overlap):
        output_A = self.AtomTypeParam(batch.batch_atomic_A)
        output_B = self.AtomTypeParam(batch.batch_atomic_B)
        parameters_A = output_A[-1]
        parameters_B = output_B[-1]
        hfvr_A = torch.abs(output_A[-2][:, 0])
        hfvr_B = torch.abs(output_B[-2][:, 0])
        valence_widths_A = output_A[-2][:, 1]
        valence_widths_B = output_B[-2][:, 1]

        if self.elst_damping_type == "AMOEBA":
            damping_fn = mtp_elst_damping_AMOEBA
        elif self.elst_damping_type == "CLIFF":
            damping_fn = mtp_elst_damping
        else:
            raise ValueError(
                "Unsupported elst_damping_type: "
                f"{self.elst_damping_type}"
            )

        Elst = damping_fn(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=output_A[0].clone(),
            muA=output_A[1],
            quadA=output_A[2],
            Ka=parameters_A[:, RACKERS_ELST_INDEX],
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=output_B[0].clone(),
            muB=output_B[1],
            quadB=output_B[2],
            Kb=parameters_B[:, RACKERS_ELST_INDEX],
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
        )
        Indu = rackers_thole_induction(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=output_A[0],
            muA=output_A[1],
            quadA=output_A[2],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=output_B[0],
            muB=output_B[1],
            quadB=output_B[2],
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=hfvr_A,
            hirshfeld_volume_ratio_B=hfvr_B,
            valence_widths_A=valence_widths_A,
            valence_widths_B=valence_widths_B,
            thole_direct_A=parameters_A[:, RACKERS_THOLE_DIRECT_INDEX],
            thole_direct_B=parameters_B[:, RACKERS_THOLE_DIRECT_INDEX],
            thole_mutual_A=parameters_A[:, RACKERS_THOLE_MUTUAL_INDEX],
            thole_mutual_B=parameters_B[:, RACKERS_THOLE_MUTUAL_INDEX],
            ind_overlap_A=parameters_A[:, RACKERS_IND_OVERLAP_INDEX],
            ind_overlap_B=parameters_B[:, RACKERS_IND_OVERLAP_INDEX],
            include_overlap=include_overlap,
            polarizability_table=self.polarizability_table,
        )
        return torch.vstack((Elst, Indu)).T, output_A, output_B

    def _overlap_width_floor(self) -> float:
        """Valence-width floor for the exchange overlap.

        ``CliffExchangeNN`` / ``CliffClassicalNN`` record ``width_floor`` in
        their config, so the value used at inference follows the checkpoint
        rather than the module default.  Parameter heads that do not declare
        one (the Rackers head, or a test stub) fall back to
        :data:`OVERLAP_WIDTH_FLOOR`.
        """
        return getattr(
            self.AtomTypeParam, "width_floor", OVERLAP_WIDTH_FLOOR
        )

    def _cliff_exch_forward(self, batch):
        """CLIFF classical exchange repulsion alone, ``[n_edges]`` kcal/mol.

        Touches neither electrostatics nor induction, so no polarizability
        table is required (``set_forward`` deliberately does not clone one for
        this mode).  ``cliff_exchange`` is called without ``dR_AB``, so it
        performs the single intermolecular distance reduction of this forward
        pass itself.
        """
        output_A = self.AtomTypeParam(batch.batch_atomic_A)
        output_B = self.AtomTypeParam(batch.batch_atomic_B)
        Exch = cliff_exchange(
            RA=batch.RA,
            RB=batch.RB,
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
            valence_widths_A=output_A[-2][:, 1],
            valence_widths_B=output_B[-2][:, 1],
            K_exch_A=output_A[-1][:, CLIFF_EXCH_INDEX],
            K_exch_B=output_B[-1][:, CLIFF_EXCH_INDEX],
            width_floor=self._overlap_width_floor(),
        )
        return Exch, output_A, output_B

    def _cliff_classical_forward(self, batch):
        return self._cliff_classical_common_forward(
            batch, include_overlap=False, include_d3=False
        )

    def _cliff_classical_overlap_forward(self, batch):
        return self._cliff_classical_common_forward(
            batch, include_overlap=True, include_d3=False
        )

    def _cliff_classical_d3_forward(self, batch):
        return self._cliff_classical_common_forward(
            batch, include_overlap=True, include_d3=True
        )

    def _cliff_classical_common_forward(
        self, batch, include_overlap, include_d3
    ):
        """Combined CLIFF electrostatics, exchange, and induction.

        Mirrors :meth:`_rackers_thole_common_forward` and adds exchange.
        Returns ``(edge_energy, output_A, output_B)`` with ``edge_energy`` of
        shape ``[n_edges, 3]`` in column order ``(Elst, Exch, Indu)``, or
        ``[n_edges, 4]`` appending ``Disp`` when ``include_d3`` is set.  All
        columns live on the full intermolecular edge domain, so the caller must
        aggregate with ``batch.dimer_ind_full``
        (see :data:`FULL_EDGE_DIMER_EVAL_MODES`).

        Parameter columns are read through the ``CLIFF_CLASSICAL_*_INDEX``
        constants: column 0 damps electrostatics, columns 1-3 feed Thole direct
        / Thole mutual / induction overlap, and column 4 is the exchange
        amplitude.
        """
        output_A = self.AtomTypeParam(batch.batch_atomic_A)
        output_B = self.AtomTypeParam(batch.batch_atomic_B)
        parameters_A = output_A[-1]
        parameters_B = output_B[-1]
        hfvr_A = torch.abs(output_A[-2][:, 0])
        hfvr_B = torch.abs(output_B[-2][:, 0])
        valence_widths_A = output_A[-2][:, 1]
        valence_widths_B = output_B[-2][:, 1]

        if self.elst_damping_type == "AMOEBA":
            damping_fn = mtp_elst_damping_AMOEBA
        elif self.elst_damping_type == "CLIFF":
            damping_fn = mtp_elst_damping
        else:
            raise ValueError(
                "Unsupported elst_damping_type: "
                f"{self.elst_damping_type}"
            )

        # One intermolecular distance reduction per forward pass, shared with
        # exchange below.
        #
        # UNITS: `get_distances` returns ANGSTROM, while `atomic_overlap_S_ij`
        # (reached through `cliff_exchange`) needs BOHR because `B_ij` is built
        # from bohr valence widths.  Convert here exactly as `cliff_exchange`'s
        # own `dR_AB=None` path does; skipping this yields a wrong-but-plausible
        # exchange energy rather than an error.
        dR_AB_ang, _ = get_distances(
            batch.RA,
            batch.RB,
            batch.e_ABfull_source,
            batch.e_ABfull_target,
        )
        dR_AB = dR_AB_ang / constants.au2ang

        # `mtp_elst_damping` mutates its `qA_0` / `qB_0` in place, so pass a
        # clone and leave `output_*[0]` intact for the induction call below.
        Elst = damping_fn(
            ZA=batch.ZA,
            RA=batch.RA,
            qA_0=output_A[0].clone(),
            muA=output_A[1],
            quadA=output_A[2],
            Ka=parameters_A[:, CLIFF_CLASSICAL_ELST_INDEX],
            ZB=batch.ZB,
            RB=batch.RB,
            qB_0=output_B[0].clone(),
            muB=output_B[1],
            quadB=output_B[2],
            Kb=parameters_B[:, CLIFF_CLASSICAL_ELST_INDEX],
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
        )
        Exch = cliff_exchange(
            RA=batch.RA,
            RB=batch.RB,
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
            valence_widths_A=valence_widths_A,
            valence_widths_B=valence_widths_B,
            K_exch_A=parameters_A[:, CLIFF_CLASSICAL_EXCH_INDEX],
            K_exch_B=parameters_B[:, CLIFF_CLASSICAL_EXCH_INDEX],
            dR_AB=dR_AB,
            width_floor=self._overlap_width_floor(),
        )
        Indu = rackers_thole_induction(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=output_A[0],
            muA=output_A[1],
            quadA=output_A[2],
            ZB=batch.ZB,
            RB=batch.RB,
            qB=output_B[0],
            muB=output_B[1],
            quadB=output_B[2],
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
            e_AA_source=batch.e_AA_source,
            e_BB_source=batch.e_BB_source,
            e_AA_target=batch.e_AA_target,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=hfvr_A,
            hirshfeld_volume_ratio_B=hfvr_B,
            valence_widths_A=valence_widths_A,
            valence_widths_B=valence_widths_B,
            thole_direct_A=parameters_A[
                :, CLIFF_CLASSICAL_THOLE_DIRECT_INDEX
            ],
            thole_direct_B=parameters_B[
                :, CLIFF_CLASSICAL_THOLE_DIRECT_INDEX
            ],
            thole_mutual_A=parameters_A[
                :, CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX
            ],
            thole_mutual_B=parameters_B[
                :, CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX
            ],
            ind_overlap_A=parameters_A[
                :, CLIFF_CLASSICAL_IND_OVERLAP_INDEX
            ],
            ind_overlap_B=parameters_B[
                :, CLIFF_CLASSICAL_IND_OVERLAP_INDEX
            ],
            include_overlap=include_overlap,
            polarizability_table=self.polarizability_table,
        )
        if include_d3:
            Disp = d3(batch, params=self.d3_damping_parameters)
            return (
                torch.vstack((Elst, Exch, Indu, Disp)).T,
                output_A,
                output_B,
            )
        return torch.vstack((Elst, Exch, Indu)).T, output_A, output_B

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


def _validate_scan_multiple(value, name="ds_exclude_scan_multiple"):
    """Bound the exclusion scan, and with it how much raw data gets processed."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number (got {value!r})")
    value = float(value)
    if not math.isfinite(value) or value < 1.0:
        raise ValueError(f"{name} must be finite and >= 1 (got {value})")
    return value


# Test seam: the caller above runs inside a 600-line __init__ that needs a
# dataset and an atom model to construct, so the validation is reachable here.
_validate_scan_multiple_for_test = _validate_scan_multiple


def normalize_excluded_elements(excluded_elements):
    """Validate an element-exclusion spec into a frozenset of atomic numbers.

    Accepts None (nothing excluded), a bare atomic number, or an iterable of
    them.  Element *symbols* are rejected on purpose: the dataset stores Z, and
    silently mapping "Cl" to 17 here would hide a typo like "CL" as an empty
    exclusion set that quietly trains on the data you meant to drop.
    """
    if excluded_elements is None:
        return frozenset()
    if isinstance(excluded_elements, (str, bytes)):
        raise TypeError(
            "excluded_elements must be atomic numbers, not element symbols "
            f"(got {excluded_elements!r})"
        )
    if isinstance(excluded_elements, (bool, np.bool_)):
        # Reject before the iterable branch, which would otherwise surface as
        # "'bool' object is not iterable" and say nothing about the real
        # mistake.
        raise TypeError(
            "excluded_elements entries must be atomic numbers "
            f"(got {excluded_elements!r})"
        )
    if isinstance(excluded_elements, (int, np.integer)):
        excluded_elements = (excluded_elements,)
    normalized = set()
    for value in excluded_elements:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                "excluded_elements entries must be atomic numbers "
                f"(got {value!r})"
            )
        z = int(value)
        if z < 1:
            raise ValueError(
                f"excluded_elements entries must be >= 1 (got {z})"
            )
        normalized.add(z)
    return frozenset(normalized)


def dimer_indices_excluding_elements(
    dataset,
    excluded_elements,
    max_size=None,
    print_level=1,
    label="",
):
    """Indices of the dimers in ``dataset`` that contain none of the excluded Z.

    Scanning stops as soon as ``max_size`` survivors have been found, so asking
    for a filtered subset costs a scan proportional to the subset size and not
    to the whole store.

    ``dataset.get`` is called directly rather than indexing ``dataset[i]``:
    ``Dataset.__getitem__`` routes through ``indices()``, which re-derives
    ``len()`` on every access, and ``len()`` on this dataset globs the entire
    processed directory.  For the uncapped 1.5M-dimer store that turns a linear
    scan quadratic.
    """
    excluded = normalize_excluded_elements(excluded_elements)
    total = dataset.len()
    if not excluded:
        return list(range(total if max_size is None else min(total, max_size)))

    getter = getattr(dataset, "get", None)
    if not callable(getter):
        raise TypeError(
            f"{type(dataset).__name__} has no get() method to scan for "
            "element exclusion"
        )
    keep = []
    scanned = 0
    for idx in range(total):
        data = getter(idx)
        try:
            present = set(data.ZA.reshape(-1).tolist())
            present |= set(data.ZB.reshape(-1).tolist())
        except AttributeError as exc:
            raise TypeError(
                "element exclusion needs per-dimer ZA/ZB atomic numbers; "
                f"datapoint {idx} of {type(dataset).__name__} has none"
            ) from exc
        scanned += 1
        if present.isdisjoint(excluded):
            keep.append(idx)
            if max_size is not None and len(keep) >= max_size:
                break
    if print_level:
        where = f" ({label})" if label else ""
        dropped = scanned - len(keep)
        print(
            f"element exclusion{where}: excluded Z="
            f"{sorted(excluded)}; scanned {scanned} dimers, kept {len(keep)}, "
            f"dropped {dropped}"
            + (f" ({100.0 * dropped / scanned:.2f}%)" if scanned else "")
        )
    if max_size is not None and len(keep) < max_size:
        warnings.warn(
            f"element exclusion{f' ({label})' if label else ''} exhausted the "
            f"dataset at {len(keep)} dimers, short of the requested "
            f"{max_size}; only {total} were available before filtering",
            RuntimeWarning,
            stacklevel=2,
        )
    return keep


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
    atom_model_config = config.get("atom_model_config") or {}
    r_cut = atom_model_config.get("r_cut", config.get("r_cut", 5.0))

    inferred_atom_type_param = _infer_atomtypeparamnn_from_state_dict(
        atom_type_param_state,
        r_cut=r_cut,
    )
    atom_type_param = AtomTypeParamNN(
        atom_model=inferred_atom_type_param.atom_model,
        n_message=inferred_atom_type_param.n_message,
        n_neuron=inferred_atom_type_param.n_neuron,
        n_embed=inferred_atom_type_param.n_embed,
        param_start_mean=atom_type_param_config.get(
            "param_start_mean",
            inferred_atom_type_param.param_start_mean,
        ),
        param_start_std=atom_type_param_config.get(
            "param_start_std",
            inferred_atom_type_param.param_start_std,
        ),
        n_params=atom_type_param_config.get(
            "n_params",
            inferred_atom_type_param.n_params,
        ),
        freeze_atom_model=False,
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

    def get_model_info(self):
        """Return a ModelInfo describing this module for print_model_tree."""
        from apnet_pt.model_print import ModelInfo, _safe_numel, get_model_info

        n_total = sum(_safe_numel(p) for p in self.parameters())
        n_train = sum(_safe_numel(p) for p in self.parameters() if p.requires_grad)
        source_name = (
            type(self.atom_model).__name__
            if hasattr(self, "atom_model") and self.atom_model is not None
            else "atom_model"
        )
        children = []
        if hasattr(self, "atom_model") and self.atom_model is not None:
            children.append(get_model_info(self.atom_model))
        return ModelInfo(
            name="AtomTypeParamNN",
            role=(
                "Predicts electrostatic damping exponent K from atom hidden "
                "states for one monomer"
            ),
            inputs=[f"h_list [from {source_name}]"],
            outputs=["K"],
            passes=["q", "\u03bc", "Q", "HFVR", "VW"],
            frozen=(n_train == 0),
            n_params=n_train,
            n_params_total=n_total,
            n_calls=1,
            children=children,
        )

    def info(self):
        """Print a Unicode model tree for this model."""
        from apnet_pt.model_print import model_tree_string

        print(model_tree_string(self, unicode=True))

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
        n_message_steps = min(self.n_message + 1, h_list.size(1))
        for p in range(self.n_params):
            for i in range(n_message_steps):
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


def _serialize_nested_atom_model(model: nn.Module) -> dict:
    if type(model) is AtomMPNN:
        return {
            "model_type": "AtomMPNN",
            "config": model.get_config(),
        }
    if type(model) is AtomTypeParamNN:
        return {
            "model_type": "AtomTypeParamNN",
            "config": model.get_config(),
            "atom_model": _serialize_nested_atom_model(model.atom_model),
        }
    raise ValueError(
        "Unsupported nested atom model type: "
        f"{type(model).__name__}"
    )


def _rebuild_nested_atom_model(
    metadata: dict,
    freeze_atom_model: bool,
) -> nn.Module:
    if not isinstance(metadata, dict):
        raise ValueError("nested_atom_model metadata must be a dictionary")
    model_type = metadata.get("model_type")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            f"Nested {model_type!r} metadata must contain a config dictionary"
        )
    if model_type == "AtomMPNN":
        return AtomMPNN(**config)
    if model_type == "AtomTypeParamNN":
        if "atom_model" not in metadata:
            raise ValueError(
                "Nested AtomTypeParamNN metadata must contain atom_model"
            )
        atom_model = _rebuild_nested_atom_model(
            metadata["atom_model"], freeze_atom_model
        )
        return AtomTypeParamNN(
            atom_model=atom_model,
            freeze_atom_model=freeze_atom_model,
            **config,
        )
    raise ValueError(f"Unsupported nested atom model type: {model_type!r}")


def _inverse_softplus(value: float) -> float:
    """Return inverse softplus without overflowing for large finite inputs."""
    if value > 20.0:
        return value + math.log1p(-math.exp(-value))
    return math.log(math.expm1(value))


_COUNT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)


def _expected_count_phrase(n_params: int) -> str:
    """Render ``"exactly four values"``-style text for a required list length.

    The Rackers case (``n_params == 4``) must render byte-for-byte as it
    always has: ``tests/test_rackers_thole_damping.py`` matches on
    ``"exactly four"`` in several places and ``train_models.py`` surfaces the
    message verbatim.
    """
    word = (
        _COUNT_WORDS[n_params]
        if 0 <= n_params < len(_COUNT_WORDS)
        else str(n_params)
    )
    noun = "value" if n_params == 1 else "values"
    return f"exactly {word} {noun}"


def _validate_positive_initialization(
    parameter_names,
    param_start_mean,
    param_start_std,
    positivity_epsilon,
) -> tuple[list[float], list[float], float, list[float]]:
    """Validate and normalize a positive per-atom parameter initialization.

    Shared by every ``AtomTypeParamNN`` subclass that exposes its parameters
    through ``softplus(raw) + positivity_epsilon``: the Rackers Thole route and
    the CLIFF exchange / classical routes.  ``parameter_names`` fixes both the
    expected list length and the error text's reported count.

    Returns ``(positive_means, raw_stds, epsilon, raw_means)`` where
    ``raw_means`` are the inverse-softplus pre-images that make a zeroed
    correction head reproduce ``positive_means`` exactly.
    """
    n_params = len(parameter_names)
    count_phrase = _expected_count_phrase(n_params)
    if not isinstance(param_start_mean, (list, tuple)) or len(
        param_start_mean
    ) != n_params:
        raise ValueError(f"param_start_mean must contain {count_phrase}")
    if not isinstance(param_start_std, (list, tuple)) or len(
        param_start_std
    ) != n_params:
        raise ValueError(f"param_start_std must contain {count_phrase}")

    try:
        epsilon = float(positivity_epsilon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "positivity_epsilon must be finite and strictly greater than zero"
        ) from exc
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            "positivity_epsilon must be finite and strictly greater than zero"
        )

    try:
        positive_means = [float(value) for value in param_start_mean]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "param_start_mean values must be finite and strictly greater than "
            "positivity_epsilon"
        ) from exc
    if any(
        not math.isfinite(value) or value <= epsilon
        for value in positive_means
    ):
        raise ValueError(
            "param_start_mean values must be finite and strictly greater than "
            "positivity_epsilon"
        )

    try:
        raw_stds = [float(value) for value in param_start_std]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "param_start_std values must be finite and greater than or equal "
            "to zero"
        ) from exc
    if any(not math.isfinite(value) or value < 0.0 for value in raw_stds):
        raise ValueError(
            "param_start_std values must be finite and greater than or equal "
            "to zero"
        )

    raw_means = [
        _inverse_softplus(value - epsilon) for value in positive_means
    ]
    embedding_dtype = torch.get_default_dtype()
    embedding_limit = torch.finfo(embedding_dtype).max
    if any(
        not math.isfinite(value) or abs(value) > embedding_limit
        for value in raw_means
    ):
        raise ValueError(
            "transformed param_start_mean values must be finite and "
            f"representable in the {embedding_dtype} embedding dtype"
        )
    if any(value > embedding_limit for value in raw_stds):
        raise ValueError(
            "param_start_std values must be representable in the "
            f"{embedding_dtype} embedding dtype"
        )

    return positive_means, raw_stds, epsilon, raw_means


def _validate_rackers_initialization(
    param_start_mean,
    param_start_std,
    positivity_epsilon,
) -> tuple[list[float], list[float], float, list[float]]:
    """Validate and normalize the Rackers positive-parameter initialization.

    Thin wrapper binding :data:`RACKERS_PARAMETER_NAMES`.  Retained as its own
    name because ``train_models.py`` calls it directly to validate CLI
    overrides before any model is constructed, and its error strings are
    asserted on verbatim by ``tests/test_rackers_thole_damping.py``.
    """
    return _validate_positive_initialization(
        RACKERS_PARAMETER_NAMES,
        param_start_mean,
        param_start_std,
        positivity_epsilon,
    )


def _validate_model_width_floor(width_floor) -> float:
    """Validate a model-configuration ``width_floor``.

    A *model config* width floor must be strictly positive and finite: it is
    the guard that keeps a degenerate predicted valence width from blowing up
    the ``rsqrt`` in :func:`atomic_overlap_S_ij`, so disabling it via config
    would silently remove that guard.

    This is deliberately stricter than the :func:`atomic_overlap_S_ij`
    *argument*, which accepts ``0.0`` as a documented legacy-parity bypass for
    the three pre-existing induction-overlap call sites (flooring those is not
    behavior-preserving).  Those call sites pass ``0.0`` positionally and never
    route through this validator.
    """
    try:
        value = float(width_floor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "width_floor must be finite and strictly greater than zero"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "width_floor must be finite and strictly greater than zero"
        )
    return value


def _ste_clamp(
    x: torch.Tensor,
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
) -> torch.Tensor:
    """Clamp ``x`` to ``[lower, upper]`` while passing gradient through as 1.

    A plain ``clamp`` would zero the gradient outside the interval, which is
    precisely the failure this is meant to prevent: a parameter that has been
    pushed out of range would then be frozen there permanently.  Adding a
    *detached* correction gives ``max``/``min`` semantics on the value and an
    identity Jacobian, so a clamped parameter keeps receiving the full gradient
    signal and moves back inside as soon as the loss asks it to.

    ``lower`` and ``upper`` broadcast against ``x``; either may be ``None``.

    The value is clamped on a *detached* copy and the gradient is reattached
    through ``x - x.detach()``, which is exactly ``0.0`` in floating point
    because both terms are bit-identical.  Writing it the obvious way instead --
    ``x + (bound - x).detach()`` -- is algebraically the same but loses the
    bound to catastrophic cancellation once ``|x|`` is large enough that
    ``bound - x`` rounds back to ``-x``: a readout driven to ``raw = 3e7`` came
    out of that form at ``32`` rather than the requested ``25``.
    """
    if lower is None and upper is None:
        return x
    clamped = x.detach().clamp(min=lower, max=upper)
    return clamped + (x - x.detach())


def _validate_bound_scale(value, name: str, *, allow_none: bool = True):
    """Validate a raw-parameter bound scale (``fraction`` or ``multiple``)."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be finite and strictly greater than zero")
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be finite and strictly greater than zero"
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be finite and strictly greater than zero")
    return scale


def _validate_param_start_mean_by_Z(param_start_mean_by_Z, parameter_names):
    """Validate and normalize per-element parameter seeds.

    Accepts ``None`` or a mapping ``{parameter_name: {Z: value}}``.  Returned
    keys are normalized to ``str`` parameter names and ``int`` atomic numbers so
    a config round-trip through JSON (which stringifies integer keys) rebuilds
    the same table.  Values are only range-checked here; the inverse-softplus
    transform happens at embedding-seed time against the same
    ``positivity_epsilon`` as the scalar means.
    """
    if param_start_mean_by_Z is None:
        return None
    if not isinstance(param_start_mean_by_Z, dict):
        raise ValueError(
            "param_start_mean_by_Z must be None or a mapping of parameter name "
            "to {Z: value}"
        )
    allowed = set(parameter_names)
    normalized: dict[str, dict[int, float]] = {}
    for raw_name, table in param_start_mean_by_Z.items():
        name = str(raw_name)
        if name not in allowed:
            raise ValueError(
                f"param_start_mean_by_Z key {name!r} is not one of "
                f"{tuple(parameter_names)}"
            )
        if not isinstance(table, dict):
            raise ValueError(
                f"param_start_mean_by_Z[{name!r}] must be a mapping of Z to value"
            )
        column: dict[int, float] = {}
        for raw_z, raw_value in table.items():
            try:
                z = int(raw_z)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"param_start_mean_by_Z[{name!r}] keys must be atomic numbers"
                ) from exc
            if not 0 <= z <= max_Z:
                raise ValueError(
                    f"param_start_mean_by_Z[{name!r}] atomic number {z} is "
                    f"outside [0, {max_Z}]"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"param_start_mean_by_Z[{name!r}][{z}] must be finite and "
                    "strictly positive"
                ) from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"param_start_mean_by_Z[{name!r}][{z}] must be finite and "
                    "strictly positive"
                )
            column[z] = value
        if column:
            normalized[name] = column
    return normalized or None


class RackersTholeDampingNN(AtomTypeParamNN):
    def __init__(
        self,
        atom_model: AtomTypeParamNN,
        n_message: int = 3,
        n_neuron: int = 128,
        n_embed: int = 8,
        param_start_mean: tuple[
            float, float, float, float
        ] = RACKERS_INITIAL_VALUES,
        param_start_std: tuple[
            float, float, float, float
        ] = RACKERS_INITIAL_STDS,
        positivity_epsilon: float = RACKERS_POSITIVITY_EPSILON,
        freeze_atom_model: bool = True,
    ):
        if type(atom_model) is not AtomTypeParamNN:
            raise ValueError("atom_model must be an AtomTypeParamNN")
        positive_means, raw_stds, positivity_epsilon, raw_means = (
            _validate_rackers_initialization(
                param_start_mean,
                param_start_std,
                positivity_epsilon,
            )
        )
        super().__init__(
            atom_model=atom_model,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            param_start_mean=raw_means,
            param_start_std=raw_stds,
            n_params=4,
            freeze_atom_model=freeze_atom_model,
        )
        if any(
            not torch.isfinite(layer.weight).all().item()
            for layer in self.guess_layer
        ):
            raise ValueError(
                "Rackers embedding initialization produced non-finite parameters"
            )
        self.raw_param_start_mean = raw_means
        self.param_start_mean = positive_means
        self.param_start_std = raw_stds
        self.positivity_epsilon = positivity_epsilon
        self.atom_model.requires_grad_(not freeze_atom_model)

    def forward(self, batch):
        output = super().forward(batch)
        raw_parameters = output[-1]
        parameters = F.softplus(raw_parameters) + self.positivity_epsilon
        return (*output[:-1], parameters)

    def get_config(self) -> dict:
        return {
            "model_type": "RackersTholeDampingNN",
            "parameter_names": list(RACKERS_PARAMETER_NAMES),
            "param_start_mean": list(self.param_start_mean),
            "param_start_std": list(self.param_start_std),
            "positivity_epsilon": self.positivity_epsilon,
            "n_message": self.n_message,
            "n_neuron": self.n_neuron,
            "n_embed": self.n_embed,
            "nested_atom_model": _serialize_nested_atom_model(
                self.atom_model
            ),
        }


class _CliffPositiveParamNN(AtomTypeParamNN):
    """Shared base for the CLIFF positive per-atom parameter heads.

    Follows :class:`RackersTholeDampingNN` exactly -- reject a nested model
    that is not an ``AtomTypeParamNN``, validate initialization through
    :func:`_validate_positive_initialization`, seed the per-element embeddings
    with inverse-softplus pre-images so a zeroed correction head reproduces the
    requested positive values, and expose
    ``K = softplus(K_raw) + positivity_epsilon`` from ``forward`` -- and adds
    two things the Rackers head does not need:

    * a ``width_floor`` config entry, since these heads feed
      :func:`atomic_overlap_S_ij`, and
    * normalization of the parameter tensor to two dimensions, so
      ``CLIFF_EXCH_INDEX`` / ``CLIFF_CLASSICAL_*_INDEX`` column indexing works
      identically for a one-column and a five-column head.

    Subclasses fix ``MODEL_TYPE`` and ``PARAMETER_NAMES``; the parameter count
    is derived from ``PARAMETER_NAMES`` and ``n_params`` is deliberately absent
    from every constructor signature.
    """

    MODEL_TYPE: str = ""
    PARAMETER_NAMES: tuple[str, ...] = ()

    def __init__(
        self,
        atom_model: AtomTypeParamNN,
        n_message: int,
        n_neuron: int,
        n_embed: int,
        param_start_mean,
        param_start_std,
        positivity_epsilon: float,
        width_floor: float,
        freeze_atom_model: bool,
        param_start_mean_by_Z=None,
        param_floor_fraction=CLIFF_PARAM_FLOOR_FRACTION,
        param_ceiling_multiple=CLIFF_PARAM_CEILING_MULTIPLE,
        readout_init_scale=CLIFF_READOUT_INIT_SCALE,
    ):
        if type(atom_model) is not AtomTypeParamNN:
            raise ValueError("atom_model must be an AtomTypeParamNN")
        positive_means, raw_stds, positivity_epsilon, raw_means = (
            _validate_positive_initialization(
                self.PARAMETER_NAMES,
                param_start_mean,
                param_start_std,
                positivity_epsilon,
            )
        )
        width_floor = _validate_model_width_floor(width_floor)
        param_start_mean_by_Z = _validate_param_start_mean_by_Z(
            param_start_mean_by_Z, self.PARAMETER_NAMES
        )
        param_floor_fraction = _validate_bound_scale(
            param_floor_fraction, "param_floor_fraction"
        )
        param_ceiling_multiple = _validate_bound_scale(
            param_ceiling_multiple, "param_ceiling_multiple"
        )
        readout_init_scale = _validate_bound_scale(
            readout_init_scale, "readout_init_scale"
        )
        if (
            param_floor_fraction is not None
            and param_ceiling_multiple is not None
            and param_floor_fraction >= param_ceiling_multiple
        ):
            raise ValueError(
                "param_floor_fraction must be strictly less than "
                "param_ceiling_multiple"
            )
        super().__init__(
            atom_model=atom_model,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            param_start_mean=raw_means,
            param_start_std=raw_stds,
            n_params=len(self.PARAMETER_NAMES),
            freeze_atom_model=freeze_atom_model,
        )
        self.param_start_mean_by_Z = param_start_mean_by_Z
        self.param_floor_fraction = param_floor_fraction
        self.param_ceiling_multiple = param_ceiling_multiple
        self.readout_init_scale = readout_init_scale
        self._seed_guess_layer_by_Z(param_start_mean_by_Z, positivity_epsilon)
        self._scale_readout_output_layers(readout_init_scale)
        if any(
            not torch.isfinite(layer.weight).all().item()
            for layer in self.guess_layer
        ):
            raise ValueError(
                f"{self.MODEL_TYPE} embedding initialization produced "
                "non-finite parameters"
            )
        self.raw_param_start_mean = raw_means
        self.param_start_mean = positive_means
        self.param_start_std = raw_stds
        self.positivity_epsilon = positivity_epsilon
        self.width_floor = width_floor
        self._register_raw_parameter_bounds(
            positive_means,
            positivity_epsilon,
            param_floor_fraction,
            param_ceiling_multiple,
        )
        self.atom_model.requires_grad_(not freeze_atom_model)

    def _seed_guess_layer_by_Z(self, param_start_mean_by_Z, positivity_epsilon):
        """Overwrite per-element rows of the seed embeddings.

        ``AtomTypeParamNN.__init__`` has already filled every row of
        ``guess_layer[p]`` with the scalar pre-image plus noise; this replaces
        the rows named in ``param_start_mean_by_Z`` with that element's own
        pre-image, keeping the same noise scale so the two paths are
        statistically identical apart from the centre.  Elements not named keep
        the scalar seed.
        """
        if not param_start_mean_by_Z:
            return
        index_by_name = {
            name: idx for idx, name in enumerate(self.PARAMETER_NAMES)
        }
        with torch.no_grad():
            for name, table in param_start_mean_by_Z.items():
                p = index_by_name[name]
                weight = self.guess_layer[p].weight
                std = float(self.param_start_std[p])
                for z, value in table.items():
                    raw = _inverse_softplus(value - positivity_epsilon)
                    if not math.isfinite(raw):
                        raise ValueError(
                            f"param_start_mean_by_Z[{name!r}][{z}] = {value} "
                            "has no finite inverse-softplus pre-image"
                        )
                    noise = std * torch.randn(
                        weight.shape[1:],
                        dtype=weight.dtype,
                        device=weight.device,
                    )
                    weight[z].copy_(raw + noise)

    def _scale_readout_output_layers(self, readout_init_scale):
        """Shrink the random correction so the per-element seed dominates at init.

        Only the *output* ``Linear`` of each per-message readout MLP is scaled,
        so the correction's contribution scales linearly in
        ``readout_init_scale``.  Scaling every layer instead would compound
        across the four-layer stack (``s ** 4``) and make the knob unreadable.
        See :data:`CLIFF_READOUT_INIT_SCALE` for the measurement motivating it.
        """
        if readout_init_scale is None or readout_init_scale == 1.0:
            return
        with torch.no_grad():
            for head in self.param_readout_layers:
                for readout in head:
                    output_layer = None
                    for module in readout.modules():
                        if isinstance(module, nn.Linear):
                            output_layer = module
                    if output_layer is None:
                        raise ValueError(
                            f"{self.MODEL_TYPE} readout stack contains no "
                            "Linear output layer to scale"
                        )
                    output_layer.weight.mul_(readout_init_scale)
                    if output_layer.bias is not None:
                        output_layer.bias.mul_(readout_init_scale)

    def _register_raw_parameter_bounds(
        self,
        positive_means,
        positivity_epsilon,
        param_floor_fraction,
        param_ceiling_multiple,
    ):
        """Precompute the raw-domain bounds handed to :func:`_ste_clamp`.

        The bounds are specified in the *positive* domain (a fraction and a
        multiple of each column's scalar seed) because that is where they are
        interpretable, then mapped once through ``_inverse_softplus`` so the
        forward pass does no transcendental work.  They are registered as
        non-persistent buffers: they are fully determined by config, so keeping
        them out of ``state_dict`` leaves checkpoints loadable by builds that
        predate this bound and avoids two sources of truth for the same number.
        """
        def _raw_bound(scale, kind):
            if scale is None:
                return None
            values = []
            for name, mean in zip(self.PARAMETER_NAMES, positive_means):
                target = scale * mean
                if target <= positivity_epsilon:
                    raise ValueError(
                        f"param_{kind} for {name!r} resolves to {target}, which "
                        "is not above positivity_epsilon"
                    )
                values.append(_inverse_softplus(target - positivity_epsilon))
            return torch.tensor(values, dtype=torch.get_default_dtype()).reshape(
                1, -1
            )

        self.register_buffer(
            "raw_parameter_floor",
            _raw_bound(param_floor_fraction, "floor_fraction"),
            persistent=False,
        )
        self.register_buffer(
            "raw_parameter_ceiling",
            _raw_bound(param_ceiling_multiple, "ceiling_multiple"),
            persistent=False,
        )

    def forward(self, batch):
        output = super().forward(batch)
        raw_parameters = output[-1]
        # `AtomTypeParamNN.forward` returns `K.squeeze(-1)` when
        # `n_params == 1`, so a single-parameter head arrives here as
        # `[n_atoms]`.  Restore the column axis *before* softplus so both CLIFF
        # heads return `[n_atoms, len(PARAMETER_NAMES)]` and callers can always
        # index a column.  `Tensor.dim()` is a static rank, not data, so this
        # branch is folded away under `torch.compile`.
        if raw_parameters.dim() == 1:
            raw_parameters = raw_parameters.unsqueeze(-1)
        # Bound the raw parameter *before* softplus, with gradient passed
        # through.  Clamping the positive output instead would leave a
        # collapsing head sitting at ``sigmoid(raw) ~ 0`` and unable to
        # recover; see :func:`_ste_clamp` and
        # :data:`CLIFF_PARAM_FLOOR_FRACTION`.
        raw_parameters = _ste_clamp(
            raw_parameters,
            self.raw_parameter_floor,
            self.raw_parameter_ceiling,
        )
        parameters = F.softplus(raw_parameters) + self.positivity_epsilon
        return (*output[:-1], parameters)

    def get_config(self) -> dict:
        return {
            "model_type": self.MODEL_TYPE,
            "parameter_names": list(self.PARAMETER_NAMES),
            "param_start_mean": list(self.param_start_mean),
            "param_start_std": list(self.param_start_std),
            "param_start_mean_by_Z": self.param_start_mean_by_Z,
            "param_floor_fraction": self.param_floor_fraction,
            "param_ceiling_multiple": self.param_ceiling_multiple,
            "readout_init_scale": self.readout_init_scale,
            "positivity_epsilon": self.positivity_epsilon,
            "width_floor": self.width_floor,
            "n_message": self.n_message,
            "n_neuron": self.n_neuron,
            "n_embed": self.n_embed,
            "nested_atom_model": _serialize_nested_atom_model(
                self.atom_model
            ),
        }


class CliffExchangeNN(_CliffPositiveParamNN):
    """Predict the single positive per-atom ``K_exch`` of CLIFF Eq. (8).

    ``forward`` returns ``(*nested_output, parameters)`` with ``parameters`` of
    shape ``[n_atoms, 1]``; read the column with
    :data:`CLIFF_EXCH_INDEX`.  Valence widths remain available from the nested
    output as ``output[-2][:, 1]`` and Hirshfeld volume ratios as
    ``abs(output[-2][:, 0])``.
    """

    MODEL_TYPE = "CliffExchangeNN"
    PARAMETER_NAMES = CLIFF_EXCH_PARAMETER_NAMES

    def __init__(
        self,
        atom_model: AtomTypeParamNN,
        n_message: int = 3,
        n_neuron: int = 128,
        n_embed: int = 8,
        param_start_mean: tuple[float] = CLIFF_EXCH_INITIAL_VALUES,
        param_start_std: tuple[float] = CLIFF_EXCH_INITIAL_STDS,
        positivity_epsilon: float = RACKERS_POSITIVITY_EPSILON,
        width_floor: float = OVERLAP_WIDTH_FLOOR,
        freeze_atom_model: bool = True,
        param_start_mean_by_Z=None,
        param_floor_fraction=CLIFF_PARAM_FLOOR_FRACTION,
        param_ceiling_multiple=CLIFF_PARAM_CEILING_MULTIPLE,
        readout_init_scale=CLIFF_READOUT_INIT_SCALE,
    ):
        if param_start_mean_by_Z is None:
            param_start_mean_by_Z = {"exch": CLIFF_EXCH_INITIAL_VALUES_BY_Z}
        super().__init__(
            atom_model=atom_model,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            positivity_epsilon=positivity_epsilon,
            width_floor=width_floor,
            freeze_atom_model=freeze_atom_model,
            param_start_mean_by_Z=param_start_mean_by_Z,
            param_floor_fraction=param_floor_fraction,
            param_ceiling_multiple=param_ceiling_multiple,
            readout_init_scale=readout_init_scale,
        )


class CliffClassicalNN(_CliffPositiveParamNN):
    """Predict the five positive per-atom parameters of the classical route.

    ``forward`` returns ``(*nested_output, parameters)`` with ``parameters`` of
    shape ``[n_atoms, 5]``.  Columns 0-3 carry the same physical meaning as the
    Rackers head (``elst``, ``thole_direct``, ``thole_mutual``,
    ``ind_overlap``) and column 4 carries ``exch``; read them with the
    ``CLIFF_CLASSICAL_*_INDEX`` constants rather than literals.
    """

    MODEL_TYPE = "CliffClassicalNN"
    PARAMETER_NAMES = CLIFF_CLASSICAL_PARAMETER_NAMES

    def __init__(
        self,
        atom_model: AtomTypeParamNN,
        n_message: int = 3,
        n_neuron: int = 128,
        n_embed: int = 8,
        param_start_mean: tuple[
            float, float, float, float, float
        ] = CLIFF_CLASSICAL_INITIAL_VALUES,
        param_start_std: tuple[
            float, float, float, float, float
        ] = CLIFF_CLASSICAL_INITIAL_STDS,
        positivity_epsilon: float = RACKERS_POSITIVITY_EPSILON,
        width_floor: float = OVERLAP_WIDTH_FLOOR,
        freeze_atom_model: bool = True,
        param_start_mean_by_Z=None,
        param_floor_fraction=CLIFF_PARAM_FLOOR_FRACTION,
        param_ceiling_multiple=CLIFF_PARAM_CEILING_MULTIPLE,
        readout_init_scale=CLIFF_READOUT_INIT_SCALE,
    ):
        if param_start_mean_by_Z is None:
            param_start_mean_by_Z = CLIFF_CLASSICAL_INITIAL_VALUES_BY_Z
        super().__init__(
            atom_model=atom_model,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            positivity_epsilon=positivity_epsilon,
            width_floor=width_floor,
            freeze_atom_model=freeze_atom_model,
            param_start_mean_by_Z=param_start_mean_by_Z,
            param_floor_fraction=param_floor_fraction,
            param_ceiling_multiple=param_ceiling_multiple,
            readout_init_scale=readout_init_scale,
        )


# Constructor for each CLIFF parameter head, keyed by the ``model_type`` its
# ``get_config`` records.  ``AM_DimerParam_Model.__init__`` builds through this
# mapping so adding a head is one entry rather than two more ``elif`` blocks.
_CLIFF_PARAMETER_HEADS: dict[str, type[_CliffPositiveParamNN]] = {
    CliffExchangeNN.MODEL_TYPE: CliffExchangeNN,
    CliffClassicalNN.MODEL_TYPE: CliffClassicalNN,
}

# `None` is a meaningful value for every one of the four initialization knobs
# below -- it disables the per-element table, a bound, or the readout scaling --
# so "not specified" needs a distinct sentinel rather than reusing `None`.
_CLIFF_HEAD_DEFAULT = object()


def _cliff_head_overrides(**kwargs) -> dict:
    """Drop unspecified CLIFF head initialization knobs.

    Passing these through as `None` would silently disable each feature for
    every caller that did not name it, so the ones left at
    :data:`_CLIFF_HEAD_DEFAULT` are omitted and the head's own default applies.
    """
    return {
        name: value
        for name, value in kwargs.items()
        if value is not _CLIFF_HEAD_DEFAULT
    }


def get_distances(RA, RB, e_source, e_target):
    RA_source = RA.index_select(0, e_source)
    RB_target = RB.index_select(0, e_target)
    dR_xyz = RB_target - RA_source
    dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
    return dR, dR_xyz


def geometric_mean_edge_values(
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    e_source: torch.Tensor,
    e_target: torch.Tensor,
) -> torch.Tensor:
    # `torch.isfinite(...).all()` in a Python predicate forces a device sync and
    # a graph break, and this helper runs six times per Rackers forward pass.
    # `torch.compiler.is_compiling()` is folded to a constant while tracing, so
    # the validation stays in eager execution and disappears under compilation.
    if not torch.compiler.is_compiling():
        if not torch.isfinite(source_values).all():
            raise ValueError("source per-atom values must be finite")
        if not torch.isfinite(target_values).all():
            raise ValueError("target per-atom values must be finite")

    source_edge_values = source_values.index_select(0, e_source)
    target_edge_values = target_values.index_select(0, e_target)
    return torch.sqrt(source_edge_values * target_edge_values)


def atomic_overlap_S_ij(
    valence_widths_A: torch.Tensor,
    valence_widths_B: torch.Tensor,
    e_AB_source: torch.Tensor,
    e_AB_target: torch.Tensor,
    dR_AB: torch.Tensor,
    width_floor: float = OVERLAP_WIDTH_FLOOR,
    width_ceiling: float | None = None,
) -> torch.Tensor:
    """Dimensionless per-edge atomic overlap ``S_ij``.

    Implements the Van Vleet effective-width overlap used by CLIFF Eq. (11)::

        B_ij = 1 / sqrt(sigma_i * sigma_j)
        S_ij = (1/3 (B_ij r_ij)^2 + B_ij r_ij + 1) exp(-B_ij r_ij)

    ``B_ij`` uses the *square root* of the width product.  CLIFF Eq. (10) is
    typeset as ``1 / (sigma_i sigma_j)``, which is neither dimensionally nor
    numerically viable: for a water-dimer hydrogen bond the literal form
    underpredicts exchange by six orders of magnitude.  See
    ``apnet3.AtomPairwiseMPNN3.valence_width_exch`` for the one place the
    literal form is deliberately retained as a *learned* shape factor.

    Units
    -----
    ``valence_widths_*`` are Slater valence widths in bohr, so ``B_ij`` is in
    bohr^-1 and ``dR_AB`` **must already be in bohr** (atomic units).  Every
    in-repo caller obtains ``dR_AB`` from ``_rackers_distance_tensors`` or
    ``distance_tensors``, both of which divide by ``constants.au2ang`` before
    returning.  ``get_distances`` returns Angstrom and must be converted by the
    caller (``cliff_exchange`` does this).  No unit conversion happens here and
    no ``h2kcalmol`` factor is applied: the return value is dimensionless and
    callers own the conversion to kcal/mol.

    Parameters
    ----------
    valence_widths_A, valence_widths_B
        Per-atom valence widths for monomer A / B, shape ``[n_A]`` / ``[n_B]``
        (a trailing singleton dimension is tolerated and reduced).
    e_AB_source, e_AB_target
        Edge index into A and B respectively, shape ``[n_edges]``.
    dR_AB
        Per-edge interatomic distance in bohr, shape ``[n_edges]``.
    width_floor
        Widths are ``clamp_min``-ed to this value before the ``rsqrt``, which
        guards against a degenerate predicted width.  ``0.0`` disables the
        floor (``clamp_min(0.0)`` is an exact no-op for the strictly positive
        widths every atom model emits).

        .. note::
           The three induction-overlap call sites in this module pass
           ``width_floor=0.0`` to preserve their historical numerics.  They
           have never applied a floor, and ``AtomHirshfeldMPNN`` emits
           ``relu(...) + 1e-4`` (``ap2_hirshfeld_atom_model.py:403``), so
           sub-``0.1`` widths really do occur in trained models; introducing
           the floor there would change existing predictions and invalidate
           trained Rackers checkpoints.  New physics (``cliff_exchange``) uses
           the ``OVERLAP_WIDTH_FLOOR`` default.

    Returns
    -------
    torch.Tensor
        Dimensionless ``S_ij`` of shape ``[n_edges]``, in the dtype and on the
        device of the inputs.

    Notes
    -----
    Exactly one ``index_select`` per monomer is performed, so no ``[n_A, n_B]``
    outer product is ever materialized.  There is no data-dependent control
    flow, keeping the helper ``torch.compile``-safe.
    """
    sigma_i = valence_widths_A.reshape(-1).index_select(0, e_AB_source)
    sigma_j = valence_widths_B.reshape(-1).index_select(0, e_AB_target)
    if width_floor:
        sigma_i = sigma_i.clamp_min(width_floor)
        sigma_j = sigma_j.clamp_min(width_floor)
    if width_ceiling is not None:
        # Defaults to `None` so the three legacy induction-overlap call sites
        # keep their exact pre-existing numerics; only `cliff_exchange` opts in.
        sigma_i = sigma_i.clamp_max(width_ceiling)
        sigma_j = sigma_j.clamp_max(width_ceiling)
    # rsqrt fuses the reciprocal and the square root into one op.
    B_ij = torch.rsqrt(sigma_i * sigma_j)
    x = B_ij * dR_AB
    # Horner form of (x^2 / 3 + x + 1); one fewer multiply than the literal.
    return (x * (x / 3.0 + 1.0) + 1.0) * torch.exp(-x)


def cliff_exchange(
    RA: torch.Tensor,
    RB: torch.Tensor,
    e_AB_source: torch.Tensor,
    e_AB_target: torch.Tensor,
    valence_widths_A: torch.Tensor,
    valence_widths_B: torch.Tensor,
    K_exch_A: torch.Tensor,
    K_exch_B: torch.Tensor,
    dR_AB: torch.Tensor | None = None,
    width_floor: float = OVERLAP_WIDTH_FLOOR,
    width_ceiling: float | None = OVERLAP_WIDTH_CEILING,
) -> torch.Tensor:
    """CLIFF classical exchange repulsion, per intermolecular edge, kcal/mol.

    Implements CLIFF Eq. (8) with the Eq. (11) overlap::

        E_ij^exch = K_i^exch K_j^exch S_ij * h2kcalmol

    The pair rule is the **product** ``K_i * K_j``, matching CLIFF's
    ``K^exch`` / ``K^indu`` / ``K^disp`` parameters.  It is *not* the geometric
    mean that :func:`geometric_mean_edge_values` applies to the Thole damping
    widths; the two must never be interchanged.  Positivity of ``K`` is a
    softplus contract owned by the parameter model, so no ``abs`` or clamp is
    applied here and the result is strictly positive, matching the SAPT
    ``Exch`` sign convention.

    Parameters
    ----------
    RA, RB
        Monomer coordinates in **Angstrom**, shape ``[n_A, 3]`` / ``[n_B, 3]``.
        Used only when ``dR_AB`` is ``None``.
    e_AB_source, e_AB_target
        Full intermolecular edge index (``e_ABfull_source`` / ``_target``),
        consistent with the Rackers routes.
    valence_widths_A, valence_widths_B
        Per-atom valence widths in bohr.
    K_exch_A, K_exch_B
        Per-atom exchange amplitudes.
    dR_AB
        Optional precomputed per-edge distance in **bohr** (atomic units).
        Combined-mode forwards pass the distances already computed for
        electrostatics so the intermolecular distance reduction happens once
        per batch.  When ``None``, distances are computed with
        :func:`get_distances` (which returns Angstrom) and converted to bohr
        here.
    width_floor
        Forwarded to :func:`atomic_overlap_S_ij`.
    width_ceiling
        Forwarded to :func:`atomic_overlap_S_ij`.  Defaults to
        :data:`OVERLAP_WIDTH_CEILING` here (and to ``None`` in the helper), so
        exchange is guarded while the legacy induction-overlap call sites keep
        their exact pre-existing numerics.

    Returns
    -------
    torch.Tensor
        Strictly positive per-edge exchange energy in kcal/mol, shape
        ``[n_edges]``.
    """
    if dR_AB is None:
        dR_ang, _ = get_distances(RA, RB, e_AB_source, e_AB_target)
        dR_AB = dR_ang / constants.au2ang
    elif dR_AB.shape[0] != e_AB_source.shape[0]:
        raise ValueError(
            "cliff_exchange received dR_AB of length "
            f"{dR_AB.shape[0]} for {e_AB_source.shape[0]} edges"
        )
    S_ij = atomic_overlap_S_ij(
        valence_widths_A,
        valence_widths_B,
        e_AB_source,
        e_AB_target,
        dR_AB,
        width_floor=width_floor,
        width_ceiling=width_ceiling,
    )
    K_i = K_exch_A.reshape(-1).index_select(0, e_AB_source)
    K_j = K_exch_B.reshape(-1).index_select(0, e_AB_target)
    return K_i * K_j * S_ij * constants.h2kcalmol


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


def _rackers_distance_tensors(
    Ri: torch.Tensor,
    Rj: torch.Tensor,
    e_source: torch.Tensor,
    e_target: torch.Tensor,
    alpha_i: torch.Tensor,
    alpha_j: torch.Tensor,
    thole_edge_values: torch.Tensor,
    damping_type: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build direct or mutual Thole tensors from edge damping values."""
    dR_ang, dR_xyz_ang = get_distances(Ri, Rj, e_source, e_target)
    dR_xyz = dR_xyz_ang / constants.au2ang
    dR = dR_ang / constants.au2ang
    alpha_source = alpha_i.index_select(0, e_source)
    alpha_target = alpha_j.index_select(0, e_target)

    if damping_type == "direct":
        _, lam_3, lam_5 = thole_damping_direct_torch(
            dR,
            alpha_source,
            alpha_target,
            thole_edge_values,
        )
    elif damping_type == "mutual":
        _, lam_3, lam_5 = thole_damping_mutual_torch(
            dR,
            alpha_source,
            alpha_target,
            thole_edge_values,
        )
    else:
        raise ValueError(
            f"Invalid Rackers damping type: {damping_type!r}"
        )

    delta = torch.eye(3, device=dR.device, dtype=dR.dtype)
    oodR = 1.0 / dR
    T1 = torch.einsum(
        "x,xy,x->xy", oodR**3, -1.0 * dR_xyz, lam_3
    )
    T2 = 3 * torch.einsum(
        "xy,xz,x->xyz", dR_xyz, dR_xyz, lam_5
    ) - torch.einsum("x,x,yz,x->xyz", dR, dR, delta, lam_3)
    T2 = torch.einsum("x,xyz->xyz", oodR**5, T2)
    return dR, dR_xyz, oodR, T1, T2


def _rackers_initial_permanent_fields(
    alpha_A: torch.Tensor,
    alpha_B: torch.Tensor,
    qA: torch.Tensor,
    muA: torch.Tensor,
    qB: torch.Tensor,
    muB: torch.Tensor,
    e_AB_source: torch.Tensor,
    e_AB_target: torch.Tensor,
    e_AA_source: torch.Tensor,
    e_AA_target: torch.Tensor,
    e_BB_source: torch.Tensor,
    e_BB_target: torch.Tensor,
    T1_AB: torch.Tensor,
    T2_AB: torch.Tensor,
    T1_AA: torch.Tensor,
    T2_AA: torch.Tensor,
    T1_BB: torch.Tensor,
    T2_BB: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build permanent-multipole fields using only direct tensors."""
    n_atoms_A = alpha_A.shape[0]
    n_atoms_B = alpha_B.shape[0]
    qA = qA.reshape(-1)
    qB = qB.reshape(-1)

    alpha_A_source = alpha_A.index_select(0, e_AB_source)
    alpha_B_target = alpha_B.index_select(0, e_AB_target)
    qA_source = qA.index_select(0, e_AB_source)
    qB_target = qB.index_select(0, e_AB_target)
    muA_source = muA.index_select(0, e_AB_source)
    muB_target = muB.index_select(0, e_AB_target)

    mu_charge_A = torch.einsum(
        "a,ai,a->ai", alpha_A_source, T1_AB, qB_target
    )
    mu_induced_0_A = scatter_sum_compile(
        mu_charge_A, e_AB_source, dim_size=n_atoms_A
    )
    mu_dipole_A = torch.einsum(
        "a,aij,aj->ai", alpha_A_source, T2_AB, muB_target
    )
    mu_induced_0_A += scatter_sum_compile(
        mu_dipole_A, e_AB_source, dim_size=n_atoms_A
    )

    mu_charge_B = torch.einsum(
        "a,ai,a->ai", alpha_B_target, -T1_AB, qA_source
    )
    mu_induced_0_B = scatter_sum_compile(
        mu_charge_B, e_AB_target, dim_size=n_atoms_B
    )
    mu_dipole_B = torch.einsum(
        "a,aij,aj->ai", alpha_B_target, T2_AB, muA_source
    )
    mu_induced_0_B += scatter_sum_compile(
        mu_dipole_B, e_AB_target, dim_size=n_atoms_B
    )

    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    qA_AA_source = qA.index_select(0, e_AA_source)
    muA_AA_source = muA.index_select(0, e_AA_source)
    mu_charge_AA = torch.einsum(
        "a,ai,a->ai", alpha_AA_target, -T1_AA, qA_AA_source
    )
    mu_dipole_AA = torch.einsum(
        "a,aij,aj->ai", alpha_AA_target, T2_AA, muA_AA_source
    )
    mu_induced_0_A += scatter_sum_compile(
        mu_charge_AA + mu_dipole_AA,
        e_AA_target,
        dim_size=n_atoms_A,
    )

    alpha_BB_target = alpha_B.index_select(0, e_BB_target)
    qB_BB_source = qB.index_select(0, e_BB_source)
    muB_BB_source = muB.index_select(0, e_BB_source)
    mu_charge_BB = torch.einsum(
        "a,ai,a->ai", alpha_BB_target, -T1_BB, qB_BB_source
    )
    mu_dipole_BB = torch.einsum(
        "a,aij,aj->ai", alpha_BB_target, T2_BB, muB_BB_source
    )
    mu_induced_0_B += scatter_sum_compile(
        mu_charge_BB + mu_dipole_BB,
        e_BB_target,
        dim_size=n_atoms_B,
    )
    return mu_induced_0_A, mu_induced_0_B


def _rackers_scf_update(
    alpha_A: torch.Tensor,
    alpha_B: torch.Tensor,
    e_AB_source: torch.Tensor,
    e_AB_target: torch.Tensor,
    e_AA_source: torch.Tensor,
    e_AA_target: torch.Tensor,
    e_BB_source: torch.Tensor,
    e_BB_target: torch.Tensor,
    T2_AB: torch.Tensor,
    T2_AA: torch.Tensor,
    T2_BB: torch.Tensor,
    mu_induced_A: torch.Tensor,
    mu_induced_B: torch.Tensor,
    mu_induced_0_A: torch.Tensor,
    mu_induced_0_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one unmixed induced-dipole update using mutual tensors."""
    n_atoms_A = alpha_A.shape[0]
    n_atoms_B = alpha_B.shape[0]
    alpha_A_source = alpha_A.index_select(0, e_AB_source)
    alpha_B_target = alpha_B.index_select(0, e_AB_target)
    alpha_AA_target = alpha_A.index_select(0, e_AA_target)
    alpha_BB_target = alpha_B.index_select(0, e_BB_target)

    mu_induced_A_due_B = torch.einsum(
        "a,aij,aj->ai",
        alpha_A_source,
        T2_AB,
        mu_induced_B.index_select(0, e_AB_target),
    )
    mu_induced_A_new = scatter_sum_compile(
        mu_induced_A_due_B, e_AB_source, dim_size=n_atoms_A
    )
    mu_induced_A_due_A = torch.einsum(
        "a,aij,aj->ai",
        alpha_AA_target,
        T2_AA,
        mu_induced_A.index_select(0, e_AA_source),
    )
    mu_induced_A_new += scatter_sum_compile(
        mu_induced_A_due_A, e_AA_target, dim_size=n_atoms_A
    )
    mu_induced_A_new += mu_induced_0_A

    mu_induced_B_due_A = torch.einsum(
        "a,aij,aj->ai",
        alpha_B_target,
        T2_AB,
        mu_induced_A.index_select(0, e_AB_source),
    )
    mu_induced_B_new = scatter_sum_compile(
        mu_induced_B_due_A, e_AB_target, dim_size=n_atoms_B
    )
    mu_induced_B_due_B = torch.einsum(
        "a,aij,aj->ai",
        alpha_BB_target,
        T2_BB,
        mu_induced_B.index_select(0, e_BB_source),
    )
    mu_induced_B_new += scatter_sum_compile(
        mu_induced_B_due_B, e_BB_target, dim_size=n_atoms_B
    )
    mu_induced_B_new += mu_induced_0_B
    return mu_induced_A_new, mu_induced_B_new


def rackers_thole_induction(
    ZA: torch.Tensor,
    RA: torch.Tensor,
    qA: torch.Tensor,
    muA: torch.Tensor,
    quadA: torch.Tensor,
    ZB: torch.Tensor,
    RB: torch.Tensor,
    qB: torch.Tensor,
    muB: torch.Tensor,
    quadB: torch.Tensor,
    e_AB_source: torch.Tensor,
    e_AB_target: torch.Tensor,
    e_AA_source: torch.Tensor,
    e_BB_source: torch.Tensor,
    e_AA_target: torch.Tensor,
    e_BB_target: torch.Tensor,
    hirshfeld_volume_ratio_A: torch.Tensor,
    hirshfeld_volume_ratio_B: torch.Tensor,
    valence_widths_A: torch.Tensor,
    valence_widths_B: torch.Tensor,
    thole_direct_A: torch.Tensor,
    thole_direct_B: torch.Tensor,
    thole_mutual_A: torch.Tensor,
    thole_mutual_B: torch.Tensor,
    ind_overlap_A: torch.Tensor,
    ind_overlap_B: torch.Tensor,
    include_overlap: bool = False,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-8,
    omega: float = 0.7,
    polarizability_table: torch.Tensor = constants.polarizability_table,
) -> torch.Tensor:
    """Compute Rackers induction with distinct direct and mutual damping."""
    del quadA, quadB
    polarizability_table = _polarizability_table_on_device(
        polarizability_table,
        ZA.device,
    )
    alpha_0_A = torch.index_select(polarizability_table, 0, ZA.long())
    alpha_0_B = torch.index_select(polarizability_table, 0, ZB.long())
    alpha_A = alpha_0_A * hirshfeld_volume_ratio_A ** (4 / 3.0)
    alpha_B = alpha_0_B * hirshfeld_volume_ratio_B ** (4 / 3.0)

    direct_AB = geometric_mean_edge_values(
        thole_direct_A,
        thole_direct_B,
        e_AB_source,
        e_AB_target,
    )
    direct_AA = geometric_mean_edge_values(
        thole_direct_A,
        thole_direct_A,
        e_AA_source,
        e_AA_target,
    )
    direct_BB = geometric_mean_edge_values(
        thole_direct_B,
        thole_direct_B,
        e_BB_source,
        e_BB_target,
    )
    direct_tensors_AB = _rackers_distance_tensors(
        RA,
        RB,
        e_AB_source,
        e_AB_target,
        alpha_A,
        alpha_B,
        direct_AB,
        "direct",
    )
    direct_tensors_AA = _rackers_distance_tensors(
        RA,
        RA,
        e_AA_source,
        e_AA_target,
        alpha_A,
        alpha_A,
        direct_AA,
        "direct",
    )
    direct_tensors_BB = _rackers_distance_tensors(
        RB,
        RB,
        e_BB_source,
        e_BB_target,
        alpha_B,
        alpha_B,
        direct_BB,
        "direct",
    )

    mutual_AB = geometric_mean_edge_values(
        thole_mutual_A,
        thole_mutual_B,
        e_AB_source,
        e_AB_target,
    )
    mutual_AA = geometric_mean_edge_values(
        thole_mutual_A,
        thole_mutual_A,
        e_AA_source,
        e_AA_target,
    )
    mutual_BB = geometric_mean_edge_values(
        thole_mutual_B,
        thole_mutual_B,
        e_BB_source,
        e_BB_target,
    )
    mutual_tensors_AB = _rackers_distance_tensors(
        RA,
        RB,
        e_AB_source,
        e_AB_target,
        alpha_A,
        alpha_B,
        mutual_AB,
        "mutual",
    )
    mutual_tensors_AA = _rackers_distance_tensors(
        RA,
        RA,
        e_AA_source,
        e_AA_target,
        alpha_A,
        alpha_A,
        mutual_AA,
        "mutual",
    )
    mutual_tensors_BB = _rackers_distance_tensors(
        RB,
        RB,
        e_BB_source,
        e_BB_target,
        alpha_B,
        alpha_B,
        mutual_BB,
        "mutual",
    )

    mu_induced_0_A, mu_induced_0_B = (
        _rackers_initial_permanent_fields(
            alpha_A,
            alpha_B,
            qA,
            muA,
            qB,
            muB,
            e_AB_source,
            e_AB_target,
            e_AA_source,
            e_AA_target,
            e_BB_source,
            e_BB_target,
            direct_tensors_AB[3],
            direct_tensors_AB[4],
            direct_tensors_AA[3],
            direct_tensors_AA[4],
            direct_tensors_BB[3],
            direct_tensors_BB[4],
        )
    )
    mu_induced_A = mu_induced_0_A.clone()
    mu_induced_B = mu_induced_0_B.clone()

    for _ in range(max_iterations):
        mu_induced_A_old = mu_induced_A.clone()
        mu_induced_B_old = mu_induced_B.clone()
        mu_induced_A_new, mu_induced_B_new = _rackers_scf_update(
            alpha_A,
            alpha_B,
            e_AB_source,
            e_AB_target,
            e_AA_source,
            e_AA_target,
            e_BB_source,
            e_BB_target,
            mutual_tensors_AB[4],
            mutual_tensors_AA[4],
            mutual_tensors_BB[4],
            mu_induced_A,
            mu_induced_B,
            mu_induced_0_A,
            mu_induced_0_B,
        )
        mu_induced_A = (
            (1 - omega) * mu_induced_A_old + omega * mu_induced_A_new
        )
        mu_induced_B = (
            (1 - omega) * mu_induced_B_old + omega * mu_induced_B_new
        )
        delta_A = torch.norm(mu_induced_A - mu_induced_A_old)
        delta_B = torch.norm(mu_induced_B - mu_induced_B_old)
        if max(delta_A, delta_B) < convergence_threshold:
            break

    qA_source = qA.reshape(-1).index_select(0, e_AB_source)
    qB_target = qB.reshape(-1).index_select(0, e_AB_target)
    muA_source = muA.index_select(0, e_AB_source)
    muB_target = muB.index_select(0, e_AB_target)
    muA_induced_source = mu_induced_A.index_select(0, e_AB_source)
    muB_induced_target = mu_induced_B.index_select(0, e_AB_target)
    qu = torch.einsum(
        "x,xy->xy", qA_source, muB_induced_target
    ) - torch.einsum("x,xy->xy", qB_target, muA_induced_source)
    E_qu = (
        torch.einsum("xy,xy->x", direct_tensors_AB[3], qu)
        * constants.h2kcalmol
    )
    E_uu = -1.0 * (
        torch.einsum(
            "xy,xz,xyz->x",
            muA_induced_source,
            muB_target,
            direct_tensors_AB[4],
        )
        + torch.einsum(
            "xy,xz,xyz->x",
            muA_source,
            muB_induced_target,
            direct_tensors_AB[4],
        )
    ) * constants.h2kcalmol
    E_ind = (E_qu + E_uu) / 2.0

    if include_overlap:
        # width_floor=0.0 preserves the historical numerics of this route: it
        # has never floored predicted widths, and flooring them now would
        # change every trained Rackers-overlap checkpoint's predictions.  See
        # atomic_overlap_S_ij for the full rationale.
        S_ij = atomic_overlap_S_ij(
            valence_widths_A,
            valence_widths_B,
            e_AB_source,
            e_AB_target,
            direct_tensors_AB[0],  # bohr
            width_floor=0.0,
        )
        K_A = ind_overlap_A.index_select(0, e_AB_source)
        K_B = ind_overlap_B.index_select(0, e_AB_target)
        E_ind -= K_A * S_ij * K_B * constants.h2kcalmol
    return E_ind


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
    polarizability_table = _polarizability_table_on_device(
        polarizability_table,
        ZA.device,
    )
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
    # width_floor=0.0 preserves this route's historical numerics; see
    # atomic_overlap_S_ij for why the floor is not applied here.
    S_ij = atomic_overlap_S_ij(
        valence_widths_A,
        valence_widths_B,
        e_AB_source,
        e_AB_target,
        dR_AB,  # bohr
        width_floor=0.0,
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
    polarizability_table = _polarizability_table_on_device(
        polarizability_table,
        ZA.device,
    )
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
    # width_floor=0.0 preserves this route's historical numerics; see
    # atomic_overlap_S_ij for why the floor is not applied here.
    S_ij = atomic_overlap_S_ij(
        valence_widths_A,
        valence_widths_B,
        e_AB_source,
        e_AB_target,
        dR_AB,  # bohr
        width_floor=0.0,
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
    polarizability_table = _polarizability_table_on_device(
        polarizability_table,
        ZA.device,
    )
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
    polarizability_table = _polarizability_table_on_device(
        polarizability_table,
        ZA.device,
    )
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
        ds_exclude_elements=None,
        ds_exclude_scan_multiple=2.0,
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
        positivity_epsilon=RACKERS_POSITIVITY_EPSILON,
        width_floor=OVERLAP_WIDTH_FLOOR,
        param_start_mean_by_Z=_CLIFF_HEAD_DEFAULT,
        param_floor_fraction=_CLIFF_HEAD_DEFAULT,
        param_ceiling_multiple=_CLIFF_HEAD_DEFAULT,
        readout_init_scale=_CLIFF_HEAD_DEFAULT,
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
            ds_max_size (int, optional): Max dataset size (truncates when set). With `ds_exclude_elements` it caps the count *after* filtering.
            ds_exclude_elements (iterable[int] or None): Atomic numbers to exclude; any dimer containing one is dropped before `ds_max_size` is applied. Atomic numbers only -- element symbols are rejected.
            ds_exclude_scan_multiple (float): How much raw data to make available to the exclusion scan, as a multiple of `ds_max_size`. Must be >= 1. This bounds both the scan and, on an unprocessed store, how many dimers get processed.
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
            width_floor (float): Valence-width floor recorded on `CliffExchangeNN` / `CliffClassicalNN` and applied inside `atomic_overlap_S_ij`; ignored by every other `model_type`, and overridden by a loaded checkpoint's own value.

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
        param_checkpoint = None
        param_config = None
        if pre_trained_model_path and model_type in POSITIVE_PARAMETER_CONTRACTS:
            # One contract-driven validation path for every positive per-atom
            # parameter head.  `label` reproduces the pre-existing Rackers
            # message prefix byte-for-byte and names the model type otherwise;
            # every check below is the original Rackers check with the
            # hard-coded model type and parameter list replaced by the looked-up
            # contract.
            label = _positive_parameter_error_label(model_type)
            expected_parameter_names = list(
                POSITIVE_PARAMETER_CONTRACTS[model_type]
            )
            param_checkpoint = model_io.load_checkpoint(
                pre_trained_model_path, map_location=device
            )
            checkpoint_version = param_checkpoint.get("checkpoint_version")
            if checkpoint_version != model_io.CHECKPOINT_VERSION:
                raise ValueError(
                    f"{label} checkpoint_version mismatch: expected "
                    f"{model_io.CHECKPOINT_VERSION}, got "
                    f"{checkpoint_version!r}"
                )
            if param_checkpoint.get("model_type") != model_type:
                raise ValueError(
                    f"{label} checkpoint model_type mismatch: expected "
                    f"{model_type}, got "
                    f"{param_checkpoint.get('model_type')!r}"
                )
            model_io.validate_checkpoint(
                param_checkpoint,
                expected_type=model_type,
            )
            param_config = param_checkpoint["config"]
            if param_config.get("parameter_names") != expected_parameter_names:
                raise ValueError(
                    f"{label} checkpoint parameter_names must exactly match "
                    f"{expected_parameter_names}"
                )
            if param_config.get("dimer_eval") != dimer_eval_type:
                raise ValueError(
                    f"{label} checkpoint dimer_eval mismatch: expected "
                    f"{dimer_eval_type}, got "
                    f"{param_config.get('dimer_eval')!r}"
                )
            if "nested_atom_model" not in param_config:
                raise ValueError(
                    f"{label} checkpoint missing nested_atom_model metadata"
                )
            self.atom_model = _rebuild_nested_atom_model(
                param_config["nested_atom_model"], freeze_atom_model
            )
            am_type = AtomTypeParamNN
        elif atom_model_type == "AtomMPNN":
            self.atom_model = AtomMPNN()
            am_type = AtomMPNN
        elif atom_model_type == "AtomHirshfeldMPNN":
            self.atom_model = AtomHirshfeldMPNN()
            am_type = AtomHirshfeldMPNN
        elif atom_model_type == "AtomTypeParamNN":
            self.atom_model = AtomTypeParamNN(
                freeze_atom_model=freeze_atom_model,
            )
            am_type = AtomTypeParamNN
        # elif atom_model_type == "AtomTypeParamMPNN":
        #     self.atom_model = AtomTypeParamMPNN()
        #     am_type = AtomTypeParamMPNN
        else:
            raise ValueError(f"Unknown atom_model_type: {atom_model_type}")

        if param_checkpoint is not None:
            pass
        elif atom_model_pre_trained_path:
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
                    freeze_atom_model=freeze_atom_model,
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
        self.pre_trained_model_path = pre_trained_model_path
        if pre_trained_model_path:
            print(
                f"Loading pre-trained MTP-MTP {model_type} from {pre_trained_model_path}"
            )
            checkpoint = param_checkpoint or model_io.load_checkpoint(
                pre_trained_model_path
            )
            config = param_config or model_io.load_config_from_checkpoint(
                checkpoint
            )
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
                    freeze_atom_model=freeze_atom_model,
                )
            elif model_type == "RackersTholeDampingNN":
                self.model = RackersTholeDampingNN(
                    atom_model=self.atom_model,
                    n_message=config["n_message"],
                    n_neuron=config["n_neuron"],
                    n_embed=config["n_embed"],
                    param_start_mean=config["param_start_mean"],
                    param_start_std=config["param_start_std"],
                    positivity_epsilon=config["positivity_epsilon"],
                    freeze_atom_model=freeze_atom_model,
                )
            elif model_type in _CLIFF_PARAMETER_HEADS:
                # `width_floor` follows the checkpoint rather than the caller,
                # so a reloaded model reproduces the overlap it was trained
                # with.  The raw-parameter bounds follow it for the same reason,
                # and more sharply: they change `forward` output, so defaulting
                # them on for a checkpoint trained without them would silently
                # alter its predictions.  Absent keys therefore mean "no
                # bounds", which is what a pre-bounds checkpoint was trained
                # with.  `param_start_mean_by_Z` only seeds embeddings that
                # `load_state_dict` immediately overwrites, so `{}` here is
                # purely about `get_config` reporting the truth.
                self.model = _CLIFF_PARAMETER_HEADS[model_type](
                    atom_model=self.atom_model,
                    n_message=config["n_message"],
                    n_neuron=config["n_neuron"],
                    n_embed=config["n_embed"],
                    param_start_mean=config["param_start_mean"],
                    param_start_std=config["param_start_std"],
                    positivity_epsilon=config["positivity_epsilon"],
                    width_floor=config.get("width_floor", width_floor),
                    freeze_atom_model=freeze_atom_model,
                    param_start_mean_by_Z=config.get(
                        "param_start_mean_by_Z", {}
                    ) or {},
                    param_floor_fraction=config.get("param_floor_fraction"),
                    param_ceiling_multiple=config.get("param_ceiling_multiple"),
                    readout_init_scale=config.get("readout_init_scale"),
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
                    freeze_atom_model=freeze_atom_model,
                )
            elif model_type == "RackersTholeDampingNN":
                self.model = RackersTholeDampingNN(
                    atom_model=self.atom_model,
                    n_message=n_message,
                    n_neuron=n_neuron,
                    n_embed=n_embed,
                    param_start_mean=param_start_mean,
                    param_start_std=param_start_std,
                    positivity_epsilon=positivity_epsilon,
                    freeze_atom_model=freeze_atom_model,
                )
            elif model_type in _CLIFF_PARAMETER_HEADS:
                self.model = _CLIFF_PARAMETER_HEADS[model_type](
                    atom_model=self.atom_model,
                    n_message=n_message,
                    n_neuron=n_neuron,
                    n_embed=n_embed,
                    param_start_mean=param_start_mean,
                    param_start_std=param_start_std,
                    positivity_epsilon=positivity_epsilon,
                    width_floor=width_floor,
                    freeze_atom_model=freeze_atom_model,
                    # Omitted rather than passed as `None`, because for these
                    # four `None` is a meaningful value ("no per-element table",
                    # "no bound", "no readout scaling") distinct from "use the
                    # head's default".
                    **_cliff_head_overrides(
                        param_start_mean_by_Z=param_start_mean_by_Z,
                        param_floor_fraction=param_floor_fraction,
                        param_ceiling_multiple=param_ceiling_multiple,
                        readout_init_scale=readout_init_scale,
                    ),
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
        self.width_floor = getattr(self.model, "width_floor", width_floor)
        # CLIFF Eq. (23) component/total loss weighting.  `None` selects the
        # historical plain MSE over the selected columns and is the default for
        # every route; any float in [0, 1] selects the Eq. (23) functional.
        # `train()` overrides both, and a combined-route checkpoint carries
        # whatever was last used so a resumed run reproduces it -- including a
        # recorded `None`, which must not silently become a float.
        loaded_config = param_config if pre_trained_model_path else None
        loaded_gamma = (loaded_config or {}).get("component_gamma")
        self.component_gamma = (
            None if loaded_gamma is None else float(loaded_gamma)
        )
        self.total_includes_d3 = bool(
            (loaded_config or {}).get("total_includes_d3", False)
        )
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

        if not pre_trained_model_path:
            if n_message != self.model.n_message:
                print(
                    f"Changing n_mesage from {self.model.n_message} to {n_message}"
                )
                self.model.n_message = n_message
            if n_neuron != self.model.n_neuron:
                print(
                    f"Changing n_neuron from {self.model.n_neuron} to {n_neuron}"
                )
                self.model.n_neuron = n_neuron
            if n_embed != self.model.n_embed:
                print(f"Changing n_embed from {self.model.n_embed} to {n_embed}")
                self.model.n_embed = n_embed
            if isinstance(param_start_mean, (list, tuple)):
                if param_start_mean != self.model.param_start_mean:
                    print(f"Changing param_start_mean to {param_start_mean}")
                    self.model.param_start_mean = param_start_mean
            elif not all(
                p == param_start_mean for p in self.model.param_start_mean
            ):
                print(f"Changing param_start_mean to {param_start_mean}")
                self.model.param_start_mean = [
                    param_start_mean
                ] * self.model.n_params

            if isinstance(param_start_std, (list, tuple)):
                if param_start_std != self.model.param_start_std:
                    print(f"Changing param_start_std to {param_start_std}")
                    self.model.param_start_std = param_start_std
            elif not all(
                p == param_start_std for p in self.model.param_start_std
            ):
                print(f"Changing param_start_std to {param_start_std}")
                self.model.param_start_std = [
                    param_start_std
                ] * self.model.n_params

        self.device = device
        self.atom_model.to(device)
        self.model.to(device)
        self.dimer_model.to(device)
        self.dimer_model.AtomTypeParam.to(device)
        if hasattr(self.dimer_model.AtomTypeParam, "atom_model"):
            self.dimer_model.AtomTypeParam.atom_model.to(device)

        split_dbs = [2, 5, 6, 7]
        # Element exclusion filters whole dimers, so it has to run before
        # ds_max_size is applied -- otherwise a 5000-dimer request silently
        # returns however many survive. The raw cap is therefore loosened, and
        # the requested size moves onto the filtered index list.
        #
        # Loosened, NOT removed. `max_size` also bounds how much of the raw
        # store gets *processed* on first use, so passing None here would turn
        # "give me 5000 filtered dimers" into a full 1.6M-dimer processing job
        # on any machine whose processed store is not already built.
        ds_excluded_elements = normalize_excluded_elements(ds_exclude_elements)
        ds_exclude_scan_multiple = _validate_scan_multiple(
            ds_exclude_scan_multiple
        )
        # Recorded so the tracked run config says which elements were dropped.
        # Without it a filtered run is indistinguishable from a full one on the
        # dashboard, which is the whole reason for logging the run.
        self.ds_excluded_elements = sorted(ds_excluded_elements)
        if not ds_excluded_elements or ds_max_size is None:
            ds_raw_max_size = ds_max_size
        else:
            ds_raw_max_size = int(
                math.ceil(ds_max_size * float(ds_exclude_scan_multiple))
            )
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
                    max_size=ds_raw_max_size,
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
            if ds_excluded_elements:
                self.dataset = self.dataset[
                    dimer_indices_excluding_elements(
                        self.dataset,
                        ds_excluded_elements,
                        max_size=ds_max_size,
                        print_level=1,
                        label="all",
                    )
                ]
            elif ds_max_size:
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
                        max_size=ds_raw_max_size,
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
                        max_size=ds_raw_max_size,
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
            if ds_excluded_elements:
                for split_idx, split_label in ((0, "train"), (1, "test")):
                    self.dataset[split_idx] = self.dataset[split_idx][
                        dimer_indices_excluding_elements(
                            self.dataset[split_idx],
                            ds_excluded_elements,
                            max_size=ds_max_size,
                            print_level=1,
                            label=split_label,
                        )
                    ]
            elif ds_max_size:
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
        if type(model).__name__ in POSITIVE_PARAMETER_CONTRACTS:
            # `dimer_eval` is the key the contract validation in `__init__`
            # cross-checks, so every positive-parameter head records it.
            model_config["dimer_eval"] = self.dimer_eval_type
        if type(model).__name__ in _CLIFF_PARAMETER_HEADS:
            model_config["d3_damping_parameters"] = deepcopy(
                self.dimer_model.d3_damping_parameters
            )
            if self.dimer_eval_type in COMBINED_CLIFF_DIMER_EVAL_MODES:
                # `component_gamma` is recorded as-is, including `None` (the
                # legacy plain-MSE sentinel).  Coercing it to a float here
                # would silently switch a resumed run onto the Eq. (23)
                # functional, whose `gamma == 1.0` endpoint is `k` times the
                # legacy loss.
                gamma, includes_d3 = self._component_loss_weighting()
                model_config["component_gamma"] = gamma
                model_config["total_includes_d3"] = includes_d3

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

    def _dimer_index_for_output(self, batch):
        # Membership is defined once in FULL_EDGE_DIMER_EVAL_MODES so a new
        # full-edge mode cannot be added to `DimerProp.set_forward` while this
        # aggregation index silently keeps using the short-range `dimer_ind`.
        if self.dimer_eval_type in FULL_EDGE_DIMER_EVAL_MODES:
            return batch.dimer_ind_full
        return batch.dimer_ind

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
            preds = scatter_sum_compile(
                preds,
                self._dimer_index_for_output(dimer_batch),
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

    def _component_loss_weighting(self) -> tuple[float | None, bool]:
        """Resolved ``(component_gamma, total_includes_d3)`` for this harness.

        ``component_gamma is None`` means the legacy plain MSE.  Read through
        ``getattr`` so a bare ``AM_DimerParam_Model.__new__`` instance (the
        pattern the training-loop unit tests use) keeps the historical defaults
        without having to know about this feature.
        """
        gamma = getattr(self, "component_gamma", None)
        return (
            None if gamma is None else float(gamma),
            bool(getattr(self, "total_includes_d3", False)),
        )

    def _batch_loss(self, preds, ref, comp_errors, batch, loss_fn):
        """Per-batch loss: legacy plain MSE, or CLIFF Eq. (23) weighted.

        ``component_gamma is None`` -- the default for every route -- takes the
        *original* expression verbatim, so the historical plain (multi-column)
        MSE is reproduced bitwise, not merely to within floating-point
        tolerance.

        Any float in ``[0.0, 1.0]`` selects the CLIFF Eq. (23) functional:

            L = (1 - gamma) * MSE(E_total) + gamma * sum_C MSE(E_C)

        ``sum_C`` is deliberately *unnormalized*, so ``component_gamma`` keeps
        the paper's meaning and ``0.4`` is CLIFF's fitted value.  Because the
        legacy default is a separate sentinel rather than an overloaded
        ``gamma == 1.0``, the Eq. (23) family is continuous over the whole
        ``[0, 1]`` sweep of CLIFF Fig. 3; ``gamma == 1.0`` is the honest
        endpoint "fit the components only, zero weight on the total"
        (``sum_C MSE(E_C)``), which is ``k`` times the plain mean MSE and
        therefore *not* the legacy loss.

        ``E_total`` is the *partial* total ``Elst + Exch + Indu`` compared
        against the summed reference columns, which keeps the total term
        differentiable with respect to trained parameters only.  With
        ``total_includes_d3`` the detached dispersion is added to the predicted
        total and the reference becomes all four SAPT columns; D3 has no
        trainable parameters and is detached so it can contribute no gradient
        by either route.
        """
        gamma, includes_d3 = self._component_loss_weighting()
        if gamma is None:
            return (
                torch.mean(torch.square(comp_errors))
                if (loss_fn is None)
                else loss_fn(preds, ref)
            )
        if preds.dim() < 2:
            raise ValueError(
                "component_gamma is only defined for multi-component "
                f"dimer_eval_type values, not {self.dimer_eval_type!r}"
            )
        component_mse = torch.square(comp_errors).mean(dim=0).sum()
        pred_total = preds.sum(dim=-1)
        if includes_d3:
            disp_edges = d3(
                batch, params=self.dimer_model.d3_damping_parameters
            ).detach()
            disp = scatter_sum_compile(
                disp_edges,
                self._dimer_index_for_output(batch),
                dim_size=batch.total_charge_A.size(0),
            )
            pred_total = pred_total + disp
            ref_total = batch.y.sum(dim=-1)
        else:
            ref_total = ref.sum(dim=-1)
        total_mse = torch.mean(torch.square(pred_total - ref_total))
        return (1.0 - gamma) * total_mse + gamma * component_mse

    def __train_batches_single_proc(
        self,
        dataloader,
        loss_fn,
        optimizer,
        rank_device,
        scheduler,
        y_ind=0,
        grad_clip_norm=None,
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
                self._dimer_index_for_output(batch),
                dim_size=batch.total_charge_A.size(0),
            )
            comp_errors = preds - ref
            # print(f"{preds = }")
            # print(f"{ref = }")
            batch_loss = self._batch_loss(
                preds, ref, comp_errors, batch, loss_fn
            )
            batch_loss.backward()
            if grad_clip_norm is not None:
                # SAPT components reach ~240 kcal/mol on close contacts, so a
                # single such dimer produces a gradient orders of magnitude
                # larger than a typical batch's under MSE.  Adam rescales by the
                # running second moment rather than clipping, so those spikes
                # both take a full-size step in an outlier direction and inflate
                # `v` enough to stall the following steps.  Bounding the global
                # norm keeps one batch from setting the trajectory.
                torch.nn.utils.clip_grad_norm_(
                    self.dimer_model.parameters(), max_norm=grad_clip_norm
                )
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
                    self._dimer_index_for_output(batch),
                    dim_size=torch.tensor(
                        batch.total_charge_A.size(0), dtype=torch.long
                    ),
                )
                comp_errors = preds - ref
                batch_loss = self._batch_loss(
                    preds, ref, comp_errors, batch, loss_fn
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
        grad_clip_norm=None,
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
            term = _mae_report_header(("Elst",))
            metric_labels = ("electrostatics",)
        elif self.dimer_eval_type in ["induced_dipole", "induced_dipole_param"]:
            y_ind = 2
            term = _mae_report_header(("Indu",))
            metric_labels = ("induction",)
            self.dimer_model.polarizability_table = (
                self.dimer_model.polarizability_table.to(self.device)
            )
        elif self.dimer_eval_type == "cliff_exch":
            # Standalone exchange fits SAPT `Exch` (column 1) alone.  It runs
            # neither electrostatics nor induction, so there is no
            # polarizability table to place and none of the induction device
            # moves below apply; `__init__` has already moved the hierarchy.
            assert isinstance(self.atom_model, AtomTypeParamNN), (
                f"{self.dimer_eval_type} is only compatible with "
                "AtomTypeParamNN atom models presently."
            )
            y_ind = 1
            term = _mae_report_header(("Exch",))
            metric_labels = ("exchange",)
        elif self.dimer_eval_type in [
            "elst_damping__induced_dipole",
            "ap3_elst_damping__induced_dipole",
            "rackers_thole",
            "rackers_thole_overlap",
            "cliff_classical",
            "cliff_classical_overlap",
        ]:
            assert isinstance(self.atom_model, AtomTypeParamNN), (
                f"{self.dimer_eval_type} is only compatible with "
                "AtomTypeParamNN atom models presently."
            )
            self.model.to(self.device)
            self.model.atom_model.to(self.device)
            self.model.atom_model.atom_model.to(self.device)
            print(self.device)
            # The combined CLIFF routes add an exchange column between elst and
            # induction, matching the dataset's [Elst, Exch, Ind, Disp] layout
            # so target slicing stays a plain column select.  The target index,
            # the report header, and the tracker metric labels all come from one
            # labelled selection, so nothing downstream is specialized to a
            # column count and the three cannot drift apart.
            if self.dimer_eval_type in COMBINED_CLIFF_DIMER_EVAL_MODES:
                target_columns = (0, 1, 2)
                target_labels = ("Elst", "Exch", "Ind")
            else:
                target_columns = (0, 2)
                target_labels = ("Elst", "Ind")
            y_ind = torch.tensor(target_columns)
            term = _mae_report_header(target_labels)
            metric_labels = tuple(
                TRACKER_METRIC_LABELS_BY_TERM[label] for label in target_labels
            )
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
                grad_clip_norm=grad_clip_norm,
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

            dt = time.time() - t1
            track_epoch_from_locals(self, locals(), metric_labels=metric_labels)
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
        # Publish the real final-epoch weights before restoring the best ones.
        stage_final_weights(self)
        # Restore into the existing module rather than rebinding self.model, so
        # compiled wrappers and the DimerProp references below keep pointing at
        # the live object.
        underlying_model = model_io.unwrap_model(self.model)
        underlying_model.load_state_dict(best_model.state_dict())
        underlying_model.to(rank_device)
        self.atom_model = underlying_model.atom_model
        if self.dimer_model.AtomTypeParam is not underlying_model:
            self.dimer_model.AtomTypeParam = underlying_model
        if (
            self.dimer_model_elst is not None
            and self.dimer_model_elst.AtomTypeParam is not underlying_model
        ):
            self.dimer_model_elst.AtomTypeParam = underlying_model
        return

    def _validate_component_loss_weighting(
        self,
        component_gamma,
        total_includes_d3,
    ) -> tuple[float | None, bool]:
        """Validate the CLIFF Eq. (23) weighting for this route.

        ``None`` (the default) selects the legacy plain MSE and is accepted on
        every route.  A float must lie in ``[0.0, 1.0]`` and selects the
        Eq. (23) functional, which is only meaningful on the combined CLIFF
        routes: a single-component route such as ``cliff_exch`` has no
        total-versus-component split, and neither do the pre-existing
        two-column routes, whose loss must stay unchanged.
        """
        includes_d3 = bool(total_includes_d3)
        if (
            component_gamma is not None or includes_d3
        ) and self.dimer_eval_type not in COMBINED_CLIFF_DIMER_EVAL_MODES:
            raise ValueError(
                "component_gamma/total_includes_d3 are only supported for the "
                "combined CLIFF routes "
                f"{sorted(COMBINED_CLIFF_DIMER_EVAL_MODES)}, not "
                f"{self.dimer_eval_type!r}"
            )
        if component_gamma is None:
            if includes_d3:
                # Without an explicit gamma there is no total term at all, so
                # a D3 evaluation per batch would buy nothing.  Rejecting is
                # clearer than silently ignoring the flag.
                raise ValueError(
                    "total_includes_d3 requires an explicit component_gamma; "
                    "the default (None) loss has no total term"
                )
            return None, False
        try:
            gamma = float(component_gamma)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "component_gamma must be None or a float in [0.0, 1.0]"
            ) from exc
        if not math.isfinite(gamma) or gamma < 0.0 or gamma > 1.0:
            raise ValueError(
                f"component_gamma must be in [0.0, 1.0], got {component_gamma!r}"
            )
        return gamma, includes_d3

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
        component_gamma=None,
        total_includes_d3=False,
        grad_clip_norm=None,
        wandb_config: WandbConfig | None = None,
        _tracker_backend=TrackerBackend.WANDB,
        _tracker_event_directory=None,
    ):
        """Fit the parameter head.

        ``component_gamma=None`` (the default) keeps the legacy plain MSE; a
        float in ``[0.0, 1.0]`` selects CLIFF Eq. (23).  Both it and
        ``total_includes_d3`` are declared here as named parameters
        *deliberately*, as is ``grad_clip_norm`` (``None`` keeps the legacy
        unclipped update): ``train_models.py`` filters ``train_kwargs`` through
        ``inspect.signature(apnet.train).parameters`` before calling, so
        anything absent from this signature is silently dropped rather than
        raising.  See :meth:`_batch_loss` for the weighting itself.
        """
        grad_clip_norm = _validate_bound_scale(grad_clip_norm, "grad_clip_norm")
        # Validated before any dataset work so a misconfigured route fails
        # immediately rather than after a dataset build.
        self.component_gamma, self.total_includes_d3 = (
            self._validate_component_loss_weighting(
                component_gamma, total_includes_d3
            )
        )
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

        tracking_config = {
            "training/epochs": n_epochs,
            "training/learning_rate_initial": lr,
            "training/random_seed": random_seed,
            "training/skip_compile": skip_compile,
            "training/grad_clip_norm": grad_clip_norm,
            # Always logged, even when empty: an absent key would have to be
            # read as "unknown", while [] is a positive statement that nothing
            # was filtered.
            "data/excluded_elements": list(
                getattr(self, "ds_excluded_elements", []) or []
            ),
        }
        if world_size > 1:
            print("Running multi-process training", flush=True)
            raise NotImplementedError(
                "Multi-process training is not implemented for MTP-MTP models."
            )
        else:
            print("Running single-process training", flush=True)
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads_per_process)
            run_tracked_single_process(
                self,
                lambda: self.single_proc_train(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    pin_memory=pin_memory,
                    num_workers=dataloader_num_workers,
                    skip_compile=skip_compile,
                    grad_clip_norm=grad_clip_norm,
                ),
                wandb_config,
                model_family="parameter",
                train_dataset=train_dataset,
                validation_dataset=test_dataset,
                effective_batch_size=batch_size,
                world_size=world_size,
                initial_config=tracking_config,
                backend=_tracker_backend,
                event_directory=_tracker_event_directory,
            )
        return


class _RackersTholeDampingModelBase(AM_DimerParam_Model):
    DIMER_EVAL: str

    def __init__(
        self,
        dataset=None,
        atom_model: AtomTypeParamNN | None = None,
        pre_trained_model_path=None,
        n_message: int = 3,
        n_neuron: int = 64,
        n_embed: int = 8,
        param_start_mean=RACKERS_INITIAL_VALUES,
        param_start_std=RACKERS_INITIAL_STDS,
        positivity_epsilon: float = RACKERS_POSITIVITY_EPSILON,
        freeze_atom_model: bool = True,
        **dataset_kwargs,
    ):
        param_start_mean, param_start_std, positivity_epsilon, _ = (
            _validate_rackers_initialization(
                param_start_mean,
                param_start_std,
                positivity_epsilon,
            )
        )
        super().__init__(
            dataset=dataset,
            atom_model=atom_model,
            atom_model_type="AtomTypeParamNN",
            model_type="RackersTholeDampingNN",
            pre_trained_model_path=pre_trained_model_path,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            positivity_epsilon=positivity_epsilon,
            n_params=4,
            dimer_eval_type=self.DIMER_EVAL,
            freeze_atom_model=freeze_atom_model,
            **dataset_kwargs,
        )


class RackersTholeDampingModel(_RackersTholeDampingModelBase):
    DIMER_EVAL = "rackers_thole"


class RackersTholeDampingOverlapModel(_RackersTholeDampingModelBase):
    DIMER_EVAL = "rackers_thole_overlap"


class _CliffParamModelBase(AM_DimerParam_Model):
    """Shared plumbing for the CLIFF training harnesses.

    Mirrors :class:`_RackersTholeDampingModelBase`: each concrete harness fixes
    its ``MODEL_TYPE``, ``DIMER_EVAL``, and ``PARAMETER_NAMES`` (hence its
    parameter count *and* column ordering) as class attributes and exposes none
    of ``n_params`` / ``model_type`` / ``dimer_eval_type`` in its public
    constructor.  Initialization always routes through
    :func:`_validate_positive_initialization`, which reports the expected count
    from ``PARAMETER_NAMES``.

    The concrete subclasses keep explicit signatures (rather than inheriting one
    with sentinel defaults) so the CLIFF defaults are visible where a reader
    looks for them, and share only the ``AM_DimerParam_Model`` call below.
    """

    DIMER_EVAL: str
    MODEL_TYPE: str
    PARAMETER_NAMES: tuple[str, ...]

    def _init_cliff_harness(
        self,
        *,
        dataset,
        atom_model,
        pre_trained_model_path,
        n_message,
        n_neuron,
        n_embed,
        param_start_mean,
        param_start_std,
        positivity_epsilon,
        width_floor,
        freeze_atom_model,
        dataset_kwargs,
    ):
        param_start_mean, param_start_std, positivity_epsilon, _ = (
            _validate_positive_initialization(
                self.PARAMETER_NAMES,
                param_start_mean,
                param_start_std,
                positivity_epsilon,
            )
        )
        AM_DimerParam_Model.__init__(
            self,
            dataset=dataset,
            atom_model=atom_model,
            atom_model_type="AtomTypeParamNN",
            model_type=self.MODEL_TYPE,
            pre_trained_model_path=pre_trained_model_path,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            positivity_epsilon=positivity_epsilon,
            width_floor=width_floor,
            n_params=len(self.PARAMETER_NAMES),
            dimer_eval_type=self.DIMER_EVAL,
            freeze_atom_model=freeze_atom_model,
            **dataset_kwargs,
        )


class CliffExchangeModel(_CliffParamModelBase):
    """Fit CLIFF classical exchange repulsion alone against SAPT ``Exch``.

    One positive per-atom parameter (``K_exch``).  Being a single-component
    route it has no total/component split, so ``train`` rejects any
    ``component_gamma`` other than the default ``None``.
    """

    DIMER_EVAL = "cliff_exch"
    MODEL_TYPE = "CliffExchangeNN"
    PARAMETER_NAMES = CLIFF_EXCH_PARAMETER_NAMES

    def __init__(
        self,
        dataset=None,
        atom_model: AtomTypeParamNN | None = None,
        pre_trained_model_path=None,
        n_message: int = 3,
        n_neuron: int = 64,
        n_embed: int = 8,
        param_start_mean=CLIFF_EXCH_INITIAL_VALUES,
        param_start_std=CLIFF_EXCH_INITIAL_STDS,
        positivity_epsilon: float = RACKERS_POSITIVITY_EPSILON,
        width_floor: float = OVERLAP_WIDTH_FLOOR,
        freeze_atom_model: bool = True,
        **dataset_kwargs,
    ):
        self._init_cliff_harness(
            dataset=dataset,
            atom_model=atom_model,
            pre_trained_model_path=pre_trained_model_path,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            positivity_epsilon=positivity_epsilon,
            width_floor=width_floor,
            freeze_atom_model=freeze_atom_model,
            dataset_kwargs=dataset_kwargs,
        )


class _CliffClassicalModelBase(_CliffParamModelBase):
    """Fit electrostatics, exchange, and induction jointly.

    Five positive per-atom parameters in the fixed order
    :data:`CLIFF_CLASSICAL_PARAMETER_NAMES`, whose first four columns
    intentionally match the Rackers ordering.  Subclasses select whether the
    short-range induction overlap correction is enabled via ``DIMER_EVAL``.
    """

    MODEL_TYPE = "CliffClassicalNN"
    PARAMETER_NAMES = CLIFF_CLASSICAL_PARAMETER_NAMES

    def __init__(
        self,
        dataset=None,
        atom_model: AtomTypeParamNN | None = None,
        pre_trained_model_path=None,
        n_message: int = 3,
        n_neuron: int = 64,
        n_embed: int = 8,
        param_start_mean=CLIFF_CLASSICAL_INITIAL_VALUES,
        param_start_std=CLIFF_CLASSICAL_INITIAL_STDS,
        positivity_epsilon: float = RACKERS_POSITIVITY_EPSILON,
        width_floor: float = OVERLAP_WIDTH_FLOOR,
        freeze_atom_model: bool = True,
        **dataset_kwargs,
    ):
        self._init_cliff_harness(
            dataset=dataset,
            atom_model=atom_model,
            pre_trained_model_path=pre_trained_model_path,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            positivity_epsilon=positivity_epsilon,
            width_floor=width_floor,
            freeze_atom_model=freeze_atom_model,
            dataset_kwargs=dataset_kwargs,
        )


class CliffClassicalModel(_CliffClassicalModelBase):
    DIMER_EVAL = "cliff_classical"


class CliffClassicalOverlapModel(_CliffClassicalModelBase):
    DIMER_EVAL = "cliff_classical_overlap"


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
        self.pre_trained_model_path = pre_trained_model_path
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
        track_pretraining_from_locals(self, locals())
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
        total_loss = torch.tensor(total_loss, dtype=torch.float32, device=rank_device)
        count = torch.tensor(count, dtype=torch.int, device=rank_device)

        # All-reduce across processes
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
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

        with torch.no_grad():
            train_loss, hfvr_MAE_t, vw_MAE_t = self.evaluate_batches(
                rank, train_loader, criterion, rank_device
            )
            test_loss, hfvr_MAE_v, vw_MAE_v = self.evaluate_batches(
                rank, test_loader, criterion, rank_device
            )
        if rank == 0:
            print(
                "  (Pre-training)  MAE: "
                f"{hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} "
                f"{vw_MAE_t:>7.4f}/{vw_MAE_v:<7.4f}",
                flush=True,
            )
        track_pretraining_from_locals(self, locals())

        lowest_test_loss = test_loss

        for epoch in range(n_epochs):
            t1 = time.time()
            test_lowered = False
            train_loss, hfvr_MAE_t, vw_MAE_t = self.train_batches(
                rank, train_loader, criterion, optimizer, rank_device
            )
            test_loss, hfvr_MAE_v, vw_MAE_v = self.evaluate_batches(
                rank, test_loader, criterion, rank_device
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
                track_epoch_from_locals(self, locals())
                test_loss = 0.0
                # if (world_size==1 or rank == 0):
                print(
                    f"  EPOCH: {epoch:4d} ({dt:<7.2f} sec)     MAE: {hfvr_MAE_t:>7.4f}/{hfvr_MAE_v:<7.4f} {vw_MAE_t:>7.4f}/{vw_MAE_v:<7.4f} {test_lowered}",
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
        track_pretraining_from_locals(self, locals())
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
                track_epoch_from_locals(self, locals())
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
        train_indices=None,
        test_indices=None,
        model_path=None,
        skip_compile=True,
        shuffle=True,
        dataloader_num_workers=0,
        world_size=1,  # Default to 1 for single-core operation
        omp_num_threads_per_process=None,
        random_seed=42,
        wandb_config: WandbConfig | None = None,
        _tracker_backend=TrackerBackend.WANDB,
        _tracker_event_directory=None,
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

        train_indices, test_indices, explicit_split = resolve_split_indices(
            self.dataset,
            split_percent=split_percent,
            train_indices=train_indices,
            test_indices=test_indices,
            label="AtomTypeParamNN monomers",
        )
        if random_seed:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            # Shuffles the order within the training set only; the split itself
            # is untouched, so an explicit split stays exactly as designed.
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

        tracking_config = {
            "training/epochs": n_epochs,
            "training/learning_rate_initial": lr,
            "training/random_seed": random_seed,
            "training/skip_compile": skip_compile,
            "data/split_kind": "explicit" if explicit_split else "uniform",
            "data/split_percent": None if explicit_split else split_percent,
        }
        if world_size > 1:
            # os.environ["OMP_NUM_THREADS"] = str(dataloader_num_workers + 1)
            print("Running multi-process training", flush=True)
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads_per_process)
            configure_distributed_tracking(
                self,
                wandb_config,
                model_family="parameter",
                initial_config=tracking_config,
                backend=_tracker_backend,
                event_directory=_tracker_event_directory,
            )
            mp.spawn(
                tracked_ddp_worker,
                args=(
                    self.ddp_train,
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
            run_tracked_single_process(
                self,
                lambda: self.single_proc_train(
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
                ),
                wandb_config,
                model_family="parameter",
                train_dataset=train_dataset,
                validation_dataset=test_dataset,
                effective_batch_size=batch_size,
                world_size=world_size,
                initial_config={
                    "training/epochs": n_epochs,
                    "training/learning_rate_initial": lr,
                    "training/random_seed": random_seed,
                    "training/skip_compile": skip_compile,
                },
                backend=_tracker_backend,
                event_directory=_tracker_event_directory,
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
