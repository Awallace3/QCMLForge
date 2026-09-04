import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


_PRETRAINED_MODEL_GROUPS = {
    "am": ["am_ensemble/am_0.pt"],
    "am_ensemble": [f"am_ensemble/am_{index}.pt" for index in range(5)],
    "ap2": ["am_ensemble/am_0.pt", "ap2_ensemble/ap2_0.pt"],
    "ap2_ensemble": [
        *[f"am_ensemble/am_{index}.pt" for index in range(5)],
        *[f"ap2_ensemble/ap2_{index}.pt" for index in range(5)],
    ],
    "ap2_tf_paper": [
        "ap2_tf_paper/atom_models/atom0.pt",
        "ap2_tf_paper/pair_models/pair0.pt",
    ],
    "ap2_tf_paper_ensemble": [
        *[f"ap2_tf_paper/atom_models/atom{index}.pt" for index in range(5)],
        *[f"ap2_tf_paper/pair_models/pair{index}.pt" for index in range(5)],
    ],
    "ap2_fused_ensemble": [
        "ap2-fused_ensemble/ap2_1.pt",
        "ap2-fused_ensemble/ap2_2.pt",
        "ap2-fused_ensemble/ap2_3.pt",
    ],
    "dapnet2": [
        "dapnet2/backbone/am_0.pt",
        "dapnet2/backbone/ap2_0.pt",
        "dapnet2/B3LYP-D3aug-cc-pVTZCP_CCSD_LP_T_RP_CBSCP.pt",
    ],
}
_TRUE_ENV_VALUES = {"1", "true", "yes", "y"}


@pytest.fixture(autouse=True)
def require_pretrained_models(request):
    """Skip marked tests unless all requested pretrained artifacts resolve."""
    marker = request.node.get_closest_marker("pretrained_models")
    if marker is None:
        return

    env_value = os.getenv("QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED", "").strip().lower()
    if env_value not in _TRUE_ENV_VALUES:
        pytest.skip(
            "pretrained model downloads are disabled; set "
            "QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED=1"
        )

    unknown_groups = set(marker.args) - _PRETRAINED_MODEL_GROUPS.keys()
    if unknown_groups:
        raise ValueError(
            f"Unknown pretrained model groups: {sorted(unknown_groups)}"
        )

    rel_paths = [
        rel_path
        for group in marker.args
        for rel_path in _PRETRAINED_MODEL_GROUPS[group]
    ]
    resolved_paths = []
    if rel_paths:
        try:
            from apnet_pt.hf_pretrained import resolve_pretrained_paths

            resolved = resolve_pretrained_paths(list(dict.fromkeys(rel_paths)))
            resolved_paths.extend(resolved.values())
        except (ImportError, OSError, RuntimeError) as exc:
            pytest.skip(f"required pretrained models could not be resolved: {exc}")

    resolved_paths.extend(ROOT / path for path in marker.kwargs.get("local", []))
    missing_paths = [str(path) for path in resolved_paths if not Path(path).is_file()]
    if missing_paths:
        pytest.skip(
            "required pretrained model paths do not exist: "
            + ", ".join(missing_paths)
        )


# The sys.path insertion above must happen before any apnet_pt import, so the
# shared fixtures below are imported after it rather than at the top of the
# file.
import pytest  # noqa: E402
import qcelemental as qcel  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN  # noqa: E402
from apnet_pt.AtomPairwiseModels.mtp_mtp import AtomTypeParamNN  # noqa: E402
from apnet_pt.pt_datasets.ap2_fused_ds import (  # noqa: E402
    ap2_fused_collate_update,
)
from apnet_pt.torch_util import set_weights_to_value  # noqa: E402


def _make_collate_item(y_scale: float) -> Data:
    """One un-collated pairwise dimer item.

    Shared by ``test_rackers_thole_damping.py`` (which imports it directly to
    build one-off batches) and by the ``synthetic_dimer_batch`` fixture below,
    which ``test_cliff_classical_exchange.py`` also uses.
    """
    return Data(
        y=torch.tensor(
            [-1.0, 2.0, -3.0, 4.0], dtype=torch.float32
        ) * y_scale,
        ZA=torch.tensor([8, 1], dtype=torch.long),
        RA=torch.tensor(
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        ZB=torch.tensor([8, 1], dtype=torch.long),
        RB=torch.tensor(
            [[3.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        e_ABsr_source=torch.tensor([0, 1], dtype=torch.long),
        e_ABsr_target=torch.tensor([0, 0], dtype=torch.long),
        e_ABlr_source=torch.tensor([0, 1], dtype=torch.long),
        e_ABlr_target=torch.tensor([1, 1], dtype=torch.long),
        e_AA_source=torch.tensor([0, 1], dtype=torch.long),
        e_AA_target=torch.tensor([1, 0], dtype=torch.long),
        e_BB_source=torch.tensor([0, 1], dtype=torch.long),
        e_BB_target=torch.tensor([1, 0], dtype=torch.long),
        dimer_ind=torch.zeros(2, dtype=torch.long),
        dimer_ind_lr=torch.zeros(2, dtype=torch.long),
        molecule_ind_A=torch.zeros(2, dtype=torch.long),
        molecule_ind_B=torch.zeros(2, dtype=torch.long),
        total_charge_A=torch.tensor(0.0),
        total_charge_B=torch.tensor(0.0),
    )


@pytest.fixture
def synthetic_dimer_batch() -> Data:
    """A two-dimer collated batch with close-contact monomers.

    Lives here because both ``test_rackers_thole_damping.py`` and
    ``test_cliff_classical_exchange.py`` drive ``DimerProp`` forwards with it.
    """
    items = [_make_collate_item(1.0), _make_collate_item(2.0)]
    for item in items:
        item.RB = torch.tensor(
            [[1.8, 0.3, 0.0], [2.7, -0.2, 0.0]],
            dtype=torch.float32,
        )
    return ap2_fused_collate_update(items)


@pytest.fixture
def synthetic_qcel_dimers():
    """Two qcel water dimers for ``predict_qcel_mols_dimer`` round trips.

    Shared by ``test_rackers_thole_damping.py`` and
    ``test_cliff_classical_exchange.py``.
    """
    first = qcel.models.Molecule.from_data("""
0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.260455  0.000000 -0.872893
--
0 1
O  3.000000  0.500000  0.000000
H  3.758602  0.500000  0.504284
H  2.739545  0.500000 -0.872893
units angstrom
""")
    second = qcel.models.Molecule.from_data("""
0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.260455  0.000000 -0.872893
--
0 1
O  3.500000 -0.250000  0.100000
H  4.258602 -0.250000  0.604284
H  3.239545 -0.250000 -0.772893
units angstrom
""")
    return [first, second]


@pytest.fixture
def atomic_batch() -> Data:
    """A single three-atom (water) monomer batch for atom-model forwards.

    Shared by ``test_rackers_thole_damping.py`` and
    ``test_cliff_classical_exchange.py``.
    """
    return Data(
        x=torch.tensor([8, 1, 1], dtype=torch.long),
        R=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [-0.3, 0.8, 0.0],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
            dtype=torch.long,
        ),
        molecule_ind=torch.zeros(3, dtype=torch.long),
        total_charge=torch.tensor([0.0], dtype=torch.float32),
        natom_per_mol=torch.tensor([3], dtype=torch.long),
    )


@pytest.fixture
def nested_hfvr_vw_model() -> AtomTypeParamNN:
    """A tiny deterministic HFVR / valence-width ``AtomTypeParamNN``.

    Its two parameter columns are the Hirshfeld volume ratio (column 0) and the
    valence width (column 1), which is the nested contract every positive
    parameter head (``RackersTholeDampingNN``, ``CliffExchangeNN``,
    ``CliffClassicalNN``) wraps.  All weights are set to a constant so
    downstream tests are deterministic without seeding.
    """
    atom_model = AtomMPNN(
        n_message=1,
        n_rbf=2,
        n_neuron=8,
        n_embed=4,
        r_cut=5.0,
    )
    nested = AtomTypeParamNN(
        atom_model=atom_model,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        param_start_mean=[1.0, 0.4],
        param_start_std=[0.0, 0.0],
        n_params=2,
        freeze_atom_model=False,
    )
    set_weights_to_value(nested, 0.01)
    return nested
