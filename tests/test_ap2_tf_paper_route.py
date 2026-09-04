"""The ``weights="ap2_tf_paper"`` route onto the published AP-Net2 ensemble.

The registry test is pure; the inference test pins the ensemble's SAPT0
prediction, which shifts by ~0.5 kcal/mol if ``quadrupole_scale = 1.5`` (a
checkpoint-config value, not a state-dict entry) is not adopted on load.
"""
import os

import numpy as np
import pytest
import qcelemental as qcel

from apnet_pt import pretrained_models
from apnet_pt.hf_pretrained import (
    DEFAULT_APNET2_WEIGHTS,
    apnet2_atom_weight_path,
    apnet2_weight_paths,
    apnet2_weight_set_size,
    apnet2_weight_sets,
)

project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TF_MODELS = [
    f"models/ap2_tf_paper/{kind}_models/{kind}{i}.pt"
    for i in range(5)
    for kind in ("atom", "pair")
]

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
    """The paper ensemble is selectable, validated, and the default is intact."""
    assert DEFAULT_APNET2_WEIGHTS == "qcmlforge"
    assert "ap2_tf_paper" in apnet2_weight_sets()
    assert apnet2_weight_set_size("ap2_tf_paper") == 5
    assert apnet2_atom_weight_path(9) == "am_ensemble/am_9.pt"
    for model_id in range(5):
        for rel_path in apnet2_weight_paths(model_id, "ap2_tf_paper").values():
            assert os.path.isfile(os.path.join(project_root, "models", rel_path))

    with pytest.raises(ValueError, match="Unknown APNet2 weight set"):
        apnet2_weight_paths(0, "ap2_tf")
    with pytest.raises(ValueError, match=r"atom model_id must be in \[0, 9\]"):
        apnet2_atom_weight_path(10)
    for bad_id in (-1, 5):
        with pytest.raises(ValueError, match=r"model_id must be in \[0, 4\]"):
            apnet2_weight_paths(bad_id, "ap2_tf_paper")
    with pytest.raises(TypeError, match="model_id must be an integer"):
        apnet2_weight_paths("0", "ap2_tf_paper")
    # The fused ensemble is a single state dict; it cannot hold these members.
    for predict in (
        pretrained_models.apnet2_model_predict,
        pretrained_models.apnet2_model_predict_pairs,
    ):
        with pytest.raises(ValueError, match="has no fused ensemble"):
            predict([mol_water_dimer], ap2_fused=True, weights="ap2_tf_paper")


@pytest.mark.pretrained_models("ap2_tf_paper_ensemble", local=TF_MODELS)
def test_paper_ensemble_reproduces_reference_interaction_energy():
    """Pinned five-member ensemble SAPT0/aug-cc-pV(D+d)Z prediction, kcal/mol."""
    pred = pretrained_models.apnet2_model_predict(
        [mol_water_dimer], compile=False, ap2_fused=False, weights="ap2_tf_paper"
    )
    # total, elst, exch, indu, disp
    expected = [[-2.61708314, -3.52547402, 2.46066155, -0.58232477, -0.96994591]]
    np.testing.assert_allclose(np.asarray(pred), expected, atol=1e-5)
