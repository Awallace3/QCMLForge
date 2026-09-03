"""Numerical parity between the original TensorFlow AP-Net2 and QCMLForge.

``models/ap2_tf_paper/**`` is a direct conversion of the SavedModels published with
``github.com/zachglick/apnet`` (branch ``sparse``).  These tests assert that the
conversion is faithful: loading those checkpoints and predicting must reproduce
what TensorFlow itself produced for the same dimers.

The TensorFlow numbers live in ``tests/dataset_data/ap2_tf_parity/``, generated
once by ``scripts/ap2_tf/tf_reference_predictions.py`` inside the legacy
TensorFlow 2.3 / python 3.8 environment.  Recording them as a fixture is what
lets this run in ordinary CI, which has no TensorFlow, no python 3.8, and none
of the 100 MB of SavedModels.

Tolerances are set well above the observed disagreement (3e-6 on multipoles,
1.3e-4 kcal/mol on energy components) but far below anything physically
meaningful, so genuine regressions in the conversion or in the forward pass are
caught while float32 reduction-order differences across platforms are not.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
FIXTURE_DIR = os.path.join(TESTS_DIR, "dataset_data", "ap2_tf_parity")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "ap2_tf"))
import parity_common  # noqa: E402  (needs the sys.path entry above)

DIMERS_NPZ = os.path.join(FIXTURE_DIR, "parity_dimers.npz")
REFERENCE_NPZ = os.path.join(FIXTURE_DIR, "tf_reference.npz")
REFERENCE_MANIFEST = os.path.join(FIXTURE_DIR, "tf_reference.manifest.json")

# TensorFlow returns the six unique quadrupole components in this order; the
# PyTorch models return the full symmetric 3x3 tensor.
QUADRUPOLE_INDICES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

ATOM_ATOL = 1.0e-4
PAIR_ATOL = 2.0e-3  # kcal/mol

MODEL_INDICES = (0, 1, 2, 3, 4)

pytestmark = pytest.mark.skipif(
    not os.path.isfile(REFERENCE_NPZ),
    reason="TensorFlow parity fixture is not present",
)


def atom_model_path(index):
    return os.path.join(PROJECT_ROOT, "models/ap2_tf_paper/atom_models/atom%d.pt" % index)


def pair_model_path(index):
    return os.path.join(PROJECT_ROOT, "models/ap2_tf_paper/pair_models/pair%d.pt" % index)


@pytest.fixture(scope="module")
def parity_inputs():
    records, labels = parity_common.load_dimers(DIMERS_NPZ)
    dimers, monomers_a, monomers_b = [], [], []
    for record in records:
        dimer = parity_common.build_dimer(record)
        # The fixture is only meaningful if the geometry survives the qcel
        # round trip identically here and in the TensorFlow environment.
        parity_common.verify_round_trip(record, dimer)
        dimers.append(dimer)
        monomers_a.append(
            parity_common.build_monomer(record["ZA"], record["RA"], record["TQA"])
        )
        monomers_b.append(
            parity_common.build_monomer(record["ZB"], record["RB"], record["TQB"])
        )
    return {
        "records": records,
        "labels": labels,
        "dimers": dimers,
        "monomers_A": monomers_a,
        "monomers_B": monomers_b,
        "reference": np.load(REFERENCE_NPZ),
    }


def stack_multipoles(predictions):
    """Flatten ``predict_qcel_mols`` output into the TensorFlow (natoms, 10) layout."""
    rows = []
    for charges, dipoles, quadrupoles, _ in predictions:
        charges = np.asarray(charges.detach().cpu(), dtype=np.float64).reshape(-1, 1)
        dipoles = np.asarray(dipoles.detach().cpu(), dtype=np.float64)
        quadrupoles = np.asarray(quadrupoles.detach().cpu(), dtype=np.float64)
        compact = np.stack(
            [quadrupoles[:, i, j] for i, j in QUADRUPOLE_INDICES], axis=1
        )
        rows.append(np.concatenate([charges, dipoles, compact], axis=1))
    return np.concatenate(rows, axis=0)


def test_reference_manifest_records_provenance():
    manifest = json.load(open(REFERENCE_MANIFEST))
    assert manifest["versions"]["tensorflow"].startswith("2.3")
    assert manifest["vintage"] == "new"
    assert manifest["device"] == "cpu"
    # A float32-exact Angstrom round trip is the premise of the whole fixture.
    assert manifest["max_angstrom_round_trip_error"] < 1e-12
    for index in MODEL_INDICES:
        for kind in ("atom", "pair"):
            entry = manifest["models"]["%s%d" % (kind, index)]
            assert len(entry["saved_model_pb_sha256"]) == 64


@pytest.mark.parametrize("index", MODEL_INDICES)
def test_atom_model_reproduces_tensorflow(parity_inputs, index):
    """Converted AtomModel multipoles must match KerasAtomModel."""
    atom_model = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=False)
    atom_model.set_pretrained_model(model_path=atom_model_path(index))
    atom_model.model.eval()

    reference = parity_inputs["reference"]
    for monomer_key, reference_key in (
        ("monomers_A", "atom%d_multipoles_A" % index),
        ("monomers_B", "atom%d_multipoles_B" % index),
    ):
        predicted = stack_multipoles(
            atom_model.predict_qcel_mols(parity_inputs[monomer_key], batch_size=4)
        )
        expected = reference[reference_key]
        assert predicted.shape == expected.shape
        np.testing.assert_allclose(predicted, expected, atol=ATOM_ATOL, rtol=0)


@pytest.mark.parametrize("index", MODEL_INDICES)
def test_pair_model_reproduces_tensorflow(parity_inputs, index):
    """Converted APNet2Model component energies must match KerasPairModel.

    This is also what rules out the embedded-atom-submodel worry. The reference
    was recorded with ``PairModel.from_file``, which is ``pretrained`` with an
    explicit path: both call ``tf.keras.models.load_model`` with
    ``atom_model=None``, so TensorFlow predicted these components using the atom
    network embedded in the pair SavedModel by
    ``KerasPairModel(atom_model.model)``. This test feeds the converted pair
    network multipoles from the *standalone* ``atom_models/atom{index}``
    instead. Electrostatics consumes those multipoles analytically, so the two
    multipole sources agreeing to ~1e-4 kcal/mol means the embedded submodel and
    the standalone atom model are the same network. Do not re-open that question
    without breaking this test first.
    """
    pair_model = APNet2Model(
        pre_trained_model_path=pair_model_path(index),
        atom_model_pre_trained_path=atom_model_path(index),
        ignore_database_null=True,
        use_GPU=False,
    )
    pair_model.model.eval()
    # The TensorFlow pair model scales both quadrupole tensors by 3/2 inside
    # mtp_elst; losing that factor moves electrostatics on these dimers by 0.086
    # kcal/mol on average and up to 0.498, and would otherwise pass every shape
    # and load check.
    assert pair_model.model.quadrupole_scale == pytest.approx(1.5)

    predicted = np.asarray(
        pair_model.predict_qcel_mols(parity_inputs["dimers"], batch_size=4),
        dtype=np.float64,
    )
    expected = parity_inputs["reference"]["pair%d_components" % index]
    assert predicted.shape == expected.shape
    np.testing.assert_allclose(predicted, expected, atol=PAIR_ATOL, rtol=0)


@pytest.mark.parametrize("index", MODEL_INDICES)
def test_set_pretrained_model_adopts_quadrupole_scale(index):
    """``set_pretrained_model`` must not silently drop the checkpoint's scale.

    It loads weights into an already-constructed model, so a forward-pass
    constant that lives in the config rather than the state dict is easy to
    lose -- and losing it produces plausible-looking, wrong energies.
    """
    atom_model = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=False)
    atom_model.set_pretrained_model(model_path=atom_model_path(index))
    pair_model = APNet2Model(
        atom_model=atom_model.model, ignore_database_null=True, use_GPU=False
    )
    assert pair_model.model.quadrupole_scale == pytest.approx(1.0)
    pair_model.set_pretrained_model(
        ap2_model_path=pair_model_path(index),
        am_model_path=atom_model_path(index),
    )
    assert pair_model.model.quadrupole_scale == pytest.approx(1.5)


@pytest.mark.parametrize("index", MODEL_INDICES)
def test_pair_model_has_no_uninitialized_parameters(index):
    """Every pair tensor must come from TensorFlow, not from a lazy-init draw.

    The first conversion of these checkpoints omitted 64 of 83 tensors.
    ``load_state_dict`` accepted it, ``nn.LazyLinear`` then materialised
    ``readout_layer_indu.0.*`` from the global torch seed, and the induction
    component became seed-dependent noise.
    """
    checkpoint = torch.load(pair_model_path(index), weights_only=False)
    state = checkpoint["model_state_dict"]
    assert len(state) == 83
    for name, tensor in state.items():
        assert not isinstance(tensor, torch.nn.parameter.UninitializedParameter), name
        assert torch.isfinite(tensor).all(), name

    pair_model = APNet2Model(
        pre_trained_model_path=pair_model_path(index),
        atom_model_pre_trained_path=atom_model_path(index),
        ignore_database_null=True,
        use_GPU=False,
    )
    for name, parameter in pair_model.model.named_parameters():
        assert not isinstance(parameter, torch.nn.parameter.UninitializedParameter), name


def test_conversion_provenance_is_recorded():
    """A checkpoint that claims paper-exact numerics must say where it came from."""
    for index in MODEL_INDICES:
        for path, expected_tensors in (
            (atom_model_path(index), 135),
            (pair_model_path(index), 83),
        ):
            checkpoint = torch.load(path, weights_only=False)
            provenance = checkpoint["tf_provenance"]
            assert provenance["source_repo"]["commit"]
            assert len(provenance["saved_model_pb_sha256"]) == 64
            assert provenance["tensorflow"].startswith("2.3")
            assert provenance["n_pt_tensors"] == expected_tensors
