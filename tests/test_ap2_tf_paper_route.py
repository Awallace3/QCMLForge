"""The ``weights="ap2_tf_paper"`` route onto the published AP-Net2 ensemble.

The registry test is pure; the inference test pins the ensemble's SAPT0
prediction, which shifts by ~0.5 kcal/mol if ``quadrupole_scale = 1.5`` (a
checkpoint-config value, not a state-dict entry) is not adopted on load.
"""
import importlib.util
import os
import pathlib

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
    assert apnet2_weight_set_size("qcmlforge") == 5
    assert apnet2_atom_weight_path(4) == "qcmlforge/atom_models/am_4.pt"
    assert apnet2_weight_paths(4, "qcmlforge") == {
        "atom": "qcmlforge/atom_models/am_4.pt",
        "pair": "qcmlforge/pair_models/ap2_4.pt",
    }
    # The pre-AtomMPNN-fix ensemble stays reachable for older results.
    assert apnet2_weight_paths(4, "qcmlforge_v1") == {
        "atom": "am_ensemble/am_4.pt",
        "pair": "ap2_ensemble/ap2_4.pt",
    }
    for model_id in range(5):
        for rel_path in apnet2_weight_paths(model_id, "ap2_tf_paper").values():
            assert os.path.isfile(os.path.join(project_root, "models", rel_path))

    with pytest.raises(ValueError, match="Unknown APNet2 weight set"):
        apnet2_weight_paths(0, "ap2_tf")
    with pytest.raises(ValueError, match=r"atom model_id must be in \[0, 4\]"):
        apnet2_atom_weight_path(5)
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


def test_upload_script_resolves_both_registry_layouts(tmp_path):
    """``planned_uploads`` handles prefixed and unprefixed registry templates.

    Some weight sets' Hugging Face paths start with the set name
    (``ap2_tf_paper/...``) and some do not (``qcmlforge_v1`` reuses the flat
    ``am_ensemble/...`` paths). Both must resolve from a directory mirroring the
    remote tree, and a prefixed set must also resolve from its own directory
    with the prefix stripped -- that is what a staging directory looks like.
    """
    spec = importlib.util.spec_from_file_location(
        "_upload_paper_models_to_hf",
        os.path.join(project_root, "scripts", "ap2_tf", "upload_paper_models_to_hf.py"),
    )
    upload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upload)

    def stage(weights, strip_prefix):
        root = tmp_path / f"{weights}-{'stripped' if strip_prefix else 'mirror'}"
        for model_id in range(apnet2_weight_set_size(weights)):
            for rel in apnet2_weight_paths(model_id, weights).values():
                rel_path = pathlib.Path(rel)
                if strip_prefix:
                    if rel_path.parts[0] != weights:
                        return None
                    rel_path = rel_path.relative_to(weights)
                target = root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"")
        return root

    for weights in ("qcmlforge", "qcmlforge_v1", "ap2_tf_paper"):
        expected = sorted(
            rel
            for model_id in range(apnet2_weight_set_size(weights))
            for rel in apnet2_weight_paths(model_id, weights).values()
        )
        # A directory mirroring the remote tree works for either template style.
        mirror = stage(weights, strip_prefix=False)
        uploads = upload.planned_uploads(weights, mirror)
        assert sorted(rel for _, rel in uploads) == expected
        assert all(local.is_file() for local, _ in uploads)

        # The set's own directory works too, when the paths carry the prefix.
        stripped = stage(weights, strip_prefix=True)
        if stripped is not None:
            uploads = upload.planned_uploads(weights, stripped)
            assert sorted(rel for _, rel in uploads) == expected
            assert all(local.is_file() for local, _ in uploads)

    # A missing checkpoint names every path tried rather than uploading a
    # partial set.
    with pytest.raises(FileNotFoundError, match="is missing but the registry maps"):
        upload.planned_uploads("qcmlforge", tmp_path / "empty")
