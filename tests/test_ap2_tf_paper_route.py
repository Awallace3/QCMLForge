"""
Cover the ``weights="ap2_tf_paper"`` route onto the published AP-Net2 ensemble.

The registry tests are pure and always run.  The loading tests need the
checkpoints, either from the Hugging Face cache or from ``models/ap2_tf_paper``,
and assert that the named route reaches exactly the same weights -- including
``quadrupole_scale = 1.5``, which lives in the checkpoint config rather than the
state dict and silently shifts electrostatics by ~0.5 kcal/mol when dropped.
"""
import os

import numpy as np
import pytest
import qcelemental as qcel

from apnet_pt import pretrained_models
from apnet_pt.AtomModels.ap2_atom_model import AtomModel
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model
from apnet_pt.hf_pretrained import (
    DEFAULT_APNET2_WEIGHTS,
    apnet2_weight_paths,
    apnet2_weight_set_size,
    apnet2_weight_sets,
)

current_file_path = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(current_file_path)
TF_ATOM_MODELS = [f"models/ap2_tf_paper/atom_models/atom{i}.pt" for i in range(5)]
TF_PAIR_MODELS = [f"models/ap2_tf_paper/pair_models/pair{i}.pt" for i in range(5)]
TF_MODEL_ZERO = [TF_ATOM_MODELS[0], TF_PAIR_MODELS[0]]

mol_water_dimer = qcel.models.Molecule.from_data("""
0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H  0.260455  0.000000 -0.872893
--
0 1
O  3.000000  0.500000  0.000000
H  3.758602  0.500000  0.504284
H  3.260455  0.500000 -0.872893
""")


def test_weight_set_registry():
    """The paper ensemble is selectable and the default is unchanged."""
    assert DEFAULT_APNET2_WEIGHTS == "qcmlforge"
    assert "ap2_tf_paper" in apnet2_weight_sets()
    assert apnet2_weight_set_size("ap2_tf_paper") == 5
    assert apnet2_weight_paths(2, "ap2_tf_paper") == {
        "atom": "ap2_tf_paper/atom_models/atom2.pt",
        "pair": "ap2_tf_paper/pair_models/pair2.pt",
    }


def test_weight_set_paths_mirror_the_repository_layout():
    """Every registry path names a file that ships in the repository."""
    for model_id in range(apnet2_weight_set_size("ap2_tf_paper")):
        for rel_path in apnet2_weight_paths(model_id, "ap2_tf_paper").values():
            assert os.path.isfile(os.path.join(project_root, "models", rel_path))


def test_unknown_weight_set_is_rejected():
    with pytest.raises(ValueError, match="Unknown APNet2 weight set"):
        apnet2_weight_paths(0, "ap2_tf")


@pytest.mark.parametrize("model_id", [-1, 5])
def test_out_of_range_model_id_is_rejected(model_id):
    with pytest.raises(ValueError, match=r"model_id must be in \[0, 4\]"):
        apnet2_weight_paths(model_id, "ap2_tf_paper")


def test_non_integer_model_id_is_rejected():
    with pytest.raises(TypeError, match="model_id must be an integer"):
        apnet2_weight_paths("0", "ap2_tf_paper")


def test_fused_ensemble_rejects_the_paper_weights():
    """The fused state dict cannot hold the separate paper checkpoints."""
    with pytest.raises(ValueError, match="has no fused ensemble"):
        pretrained_models.apnet2_model_predict(
            [mol_water_dimer], ap2_fused=True, weights="ap2_tf_paper"
        )
    with pytest.raises(ValueError, match="has no fused ensemble"):
        pretrained_models.apnet2_model_predict_pairs(
            [mol_water_dimer], ap2_fused=True, weights="ap2_tf_paper"
        )


@pytest.mark.pretrained_models("ap2_tf_paper", local=[TF_ATOM_MODELS[0]])
def test_atom_route_matches_direct_path():
    """``weights="ap2_tf_paper"`` loads the same atom weights as the file."""
    by_name = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=False)
    by_name.set_pretrained_model(model_id=0, weights="ap2_tf_paper")

    by_path = AtomModel(ds_root=None, ignore_database_null=True, use_GPU=False)
    by_path.set_pretrained_model(
        model_path=os.path.join(project_root, TF_ATOM_MODELS[0])
    )

    charges_name = by_name.predict_qcel_mols(
        [mol_water_dimer.get_fragment(0)], batch_size=1
    )[0][0]
    charges_path = by_path.predict_qcel_mols(
        [mol_water_dimer.get_fragment(0)], batch_size=1
    )[0][0]
    assert np.allclose(np.asarray(charges_name), np.asarray(charges_path), atol=0.0)


@pytest.mark.pretrained_models("ap2_tf_paper", local=TF_MODEL_ZERO)
def test_pair_route_matches_direct_path():
    """The named route carries both checkpoints and the 1.5 quadrupole scale."""
    by_name = APNet2Model(ignore_database_null=True, use_GPU=False)
    by_name.set_pretrained_model(model_id=0, weights="ap2_tf_paper")
    assert by_name.model.quadrupole_scale == pytest.approx(1.5)

    by_path = APNet2Model(ignore_database_null=True, use_GPU=False)
    by_path.set_pretrained_model(
        ap2_model_path=os.path.join(project_root, TF_PAIR_MODELS[0]),
        am_model_path=os.path.join(project_root, TF_ATOM_MODELS[0]),
    )

    pred_name = by_name.predict_qcel_mols([mol_water_dimer], batch_size=1)
    pred_path = by_path.predict_qcel_mols([mol_water_dimer], batch_size=1)
    assert np.allclose(
        np.asarray(pred_name), np.asarray(pred_path), atol=1e-6
    ), f"{pred_name = }\n{pred_path = }"


@pytest.mark.pretrained_models("ap2_tf_paper", local=TF_ATOM_MODELS + TF_PAIR_MODELS)
def test_ensemble_route_averages_the_paper_models():
    """The ensemble route reproduces a hand-built average of all five members."""
    pred = pretrained_models.apnet2_model_predict(
        [mol_water_dimer], compile=False, ap2_fused=False, weights="ap2_tf_paper"
    )

    members = []
    for model_id in range(5):
        model = APNet2Model(ignore_database_null=True, use_GPU=False)
        model.set_pretrained_model(
            ap2_model_path=os.path.join(project_root, TF_PAIR_MODELS[model_id]),
            am_model_path=os.path.join(project_root, TF_ATOM_MODELS[model_id]),
        )
        members.append(
            np.asarray(model.predict_qcel_mols([mol_water_dimer], batch_size=1))
        )
    expected_components = np.mean(members, axis=0)
    # Column 0 of the route's output is the total; columns 1-4 are the
    # components in (elst, exch, indu, disp) order.
    expected = np.column_stack(
        [expected_components.sum(axis=1), expected_components]
    )

    assert np.allclose(
        np.asarray(pred), expected, atol=1e-5
    ), f"{pred = }\n{expected = }"
